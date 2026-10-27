"""Versioned MovieLens Small DuckDB/Parquet preparation with immutable inputs/outputs."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import tempfile

import duckdb
import yaml

from .small_data import COLUMNS, DATASET_ID, DataValidationError, sha256, verify_provenance
from .small_protocol import (PARTITIONS, POLICY, CROSS_USER_LIMITATION, assign,
                             summaries, evaluation_tables, cohort_summary, inner_diagnostics)

ROOT = Path(__file__).resolve().parents[2]
LIMITATIONS = [
    'Rating timestamps are submissions, not verified consumption or exposure events.',
    'Observed low ratings and unobserved pairs are distinct; missing pairs are not verified negatives.',
    'Selection, popularity and submission-burst biases limit generalization; no online or causal claims.',
    'Validation histories are not appended before Test; only frozen Train context is permitted.',
]
SCHEMA = dict(interactionId='BIGINT', userId='INTEGER', movieId='INTEGER', rating='FLOAT',
              timestamp='BIGINT', submitted_at='TIMESTAMP WITH TIME ZONE', positive='BOOLEAN')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def json_bytes(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def write_immutable_json(path,value):
    path=Path(path);content=json_bytes(value)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=content:
            raise FileExistsError(f'Immutable output differs: {path}. Use a new version; no overwrite performed.')
        return
    with path.open('xb') as stream:stream.write(content)


def settings(path='params.yaml'):
    config=yaml.safe_load(Path(path).read_text())
    d,p=config['dataset'],config['protocol']
    if d['id']!=DATASET_ID or p['primary']!='per_user_chronological_70_15_15' or p['secondary']!='global_chronological_80_10_10':
        raise ValueError('Dataset/protocol not supported by this versioned pipeline')
    for key,expected in [('positive_threshold',4.0),('catalog','train_observed'),('seen','all_train_rated'),
                         ('history','frozen_train'),('timestamp_ties','whole_groups_nearest_floor_target_earlier_tie'),
                         ('short_history','one_group_all_train_two_groups_train_test'),('inner_history_percent',80)]:
        if p.get(key)!=expected:raise ValueError(f'Incompatible protocol assumption: {key}')
    return config


def raw_connection(raw_dir, dataset):
    """Check frozen hashes and CSV headers, then validate/cast in DuckDB SQL, never repair."""
    raw_dir=Path(raw_dir)
    manifest=verify_provenance(raw_dir)
    expected={'provenance.json':dataset['provenance_sha256'],
              'ratings.csv':dataset['ratings_sha256'],'movies.csv':dataset['movies_sha256']}
    for filename,value in expected.items():
        if sha256(raw_dir/filename)!=value:
            raise DataValidationError(f'Frozen raw SHA256 mismatch: {filename}; create a reviewed new data version')
    for filename,columns in COLUMNS.items():
        with (raw_dir/filename).open(encoding='utf-8-sig',newline='') as f:
            if next(csv.reader(f),[])!=columns:raise DataValidationError(f'Incompatible CSV schema: {filename}')
    con=duckdb.connect(':memory:')
    con.execute("SET TimeZone='UTC'; SET threads=1; SET memory_limit='1GB'")
    con.read_csv(str(raw_dir/'ratings.csv'),header=True,auto_detect=False,quotechar='"',escapechar='"',
                 columns={'userId':'BIGINT','movieId':'BIGINT','rating':'DOUBLE','timestamp':'BIGINT'}).create('raw_ratings')
    con.read_csv(str(raw_dir/'movies.csv'),header=True,auto_detect=False,quotechar='"',escapechar='"',
                 columns={'movieId':'BIGINT','title':'VARCHAR','genres':'VARCHAR'}).create('raw_movies')
    checks={
        'invalid_rating_rows':'''SELECT count(*) FROM raw_ratings WHERE userId IS NULL OR movieId IS NULL OR rating IS NULL OR timestamp IS NULL
            OR userId<=0 OR movieId<=0 OR userId>2147483647 OR movieId>2147483647 OR timestamp<0
            OR NOT isfinite(rating) OR rating<0.5 OR rating>5 OR rating*2<>floor(rating*2)''',
        'invalid_movie_rows':'''SELECT count(*) FROM raw_movies WHERE movieId IS NULL OR movieId<=0 OR movieId>2147483647
            OR title IS NULL OR genres IS NULL OR trim(title)='' OR trim(genres)='' ''',
        'duplicate_pairs':'SELECT count(*) FROM (SELECT userId,movieId FROM raw_ratings GROUP BY ALL HAVING count(*)>1)',
        'duplicate_movies':'SELECT count(*) FROM (SELECT movieId FROM raw_movies GROUP BY movieId HAVING count(*)>1)',
        'missing_movie_references':'SELECT count(*) FROM raw_ratings r ANTI JOIN raw_movies m USING(movieId)',
    }
    counts={key:int(con.sql(sql).fetchone()[0]) for key,sql in checks.items()}
    if any(counts.values()):
        con.close();raise DataValidationError(f'DuckDB raw quality checks failed: {counts}')
    observed=dict(zip(['ratings','users','metadata_movies'],con.sql('''SELECT count(*),count(DISTINCT userId),
        (SELECT count(*) FROM raw_movies) FROM raw_ratings''').fetchone()))
    for key,value in observed.items():
        if value!=dataset['expected_'+key]:
            con.close();raise DataValidationError(f'Expected {key}={dataset["expected_"+key]}, observed {value}')
    con.execute('''CREATE TABLE ratings AS SELECT
        (row_number() OVER(ORDER BY userId,timestamp,movieId)-1)::BIGINT AS interactionId,
        userId::INTEGER AS userId,movieId::INTEGER AS movieId,rating::FLOAT AS rating,timestamp::BIGINT AS timestamp,
        to_timestamp(timestamp) AS submitted_at,rating>=4.0 AS positive FROM raw_ratings ORDER BY userId,timestamp,movieId''')
    con.execute('CREATE TABLE movies AS SELECT movieId::INTEGER AS movieId,title,genres FROM raw_movies ORDER BY movieId')
    # Force normalization to be evaluated now, including out-of-range timestamp failures.
    con.sql('SELECT min(submitted_at),max(submitted_at) FROM ratings').fetchone()
    return con,dict(dataset_id=DATASET_ID,raw_sha256=expected,observed=observed,
                    duckdb_quality_checks=counts,provenance_quality=manifest['quality'])


def assignment_integrity(con,source='ratings',table='assigned',mode='per_user',inner=False):
    from .small_protocol import name
    source,table=name(source),name(table)
    # EXCEPT ALL checks exact source values/multiplicity, not just total counts.
    differences=con.sql(f'''SELECT count(*) FROM (
        (SELECT * FROM {source} EXCEPT ALL SELECT * EXCLUDE(partition) FROM {table}) UNION ALL
        (SELECT * EXCLUDE(partition) FROM {table} EXCEPT ALL SELECT * FROM {source}))''').fetchone()[0]
    if differences:raise DataValidationError('Dropped, duplicated or changed source interactions')
    if con.sql(f'SELECT count(*)-count(DISTINCT interactionId) FROM {table}').fetchone()[0]:
        raise DataValidationError('Duplicate interaction IDs')
    if con.sql(f'SELECT count(*) FROM (SELECT userId,timestamp FROM {table} GROUP BY ALL HAVING count(DISTINCT partition)>1)').fetchone()[0]:
        raise DataValidationError('Same-user timestamp group split across windows')
    code="CASE partition WHEN 'train' THEN 0 WHEN 'validation' THEN 1 ELSE 2 END"
    violations=con.sql(f'''WITH seq AS (SELECT timestamp,{code} AS p,
        lag({code}) OVER(PARTITION BY userId ORDER BY timestamp,movieId) AS prior FROM {table})
        SELECT count(*) FROM seq WHERE prior>p''').fetchone()[0]
    if violations:raise DataValidationError('Within-user chronology violated')
    if mode=='global':
        if con.sql(f'SELECT count(*) FROM (SELECT timestamp FROM {table} GROUP BY timestamp HAVING count(DISTINCT partition)>1)').fetchone()[0]:
            raise DataValidationError('Global timestamp ties split')
        bounds=con.sql(f'SELECT partition,min(timestamp),max(timestamp) FROM {table} GROUP BY partition').fetchall()
        order=['train','labels'] if inner else list(PARTITIONS)
        nonempty=sorted(bounds,key=lambda row:order.index(row[0]))
        if any(a[2]>=b[1] for a,b in zip(nonempty,nonempty[1:])):
            raise DataValidationError('Global chronological boundary violated')
    return dict(source_rows=con.sql(f'SELECT count(*) FROM {source}').fetchone()[0],
                exact_source_coverage=True,unique_interactions=True,whole_timestamp_groups=True,
                within_user_chronology=True,global_chronology=(mode=='global'))


def records(frame):
    return json.loads(frame.to_json(orient='records',double_precision=15))


def code_hashes():
    return {f'src/recsys/{filename}':sha256(Path(__file__).parent/filename)
            for filename in ['small_data.py','small_protocol.py','small_pipeline.py']}


def compare_protocols(con,inner_percent=80):
    comparisons=[]
    for mode,ratios in [('per_user',(70,15,15)),('per_user',(80,10,10)),('global',(70,15,15)),('global',(80,10,10))]:
        assign(con,mode=mode,ratios=ratios)
        integrity=assignment_integrity(con,mode=mode)
        evaluation_tables(con)
        catalog=con.sql('SELECT count(*) FROM item_eligibility WHERE eligible').fetchone()[0]
        cohorts=cohort_summary(con)
        inner=inner_diagnostics(con,mode=mode,history_percent=inner_percent)
        comparisons.append(dict(protocol=mode+'_'+('_'.join(map(str,ratios))),mode=mode,ratio_targets=list(ratios),
            partitions=records(summaries(con)),train_known_movies=catalog,evaluation_windows=cohorts,
            historical_ranker_upper_bounds=inner,integrity=integrity))
    return dict(decision='Per-user 70/15/15 primary, global 80/10/10 sensitivity, specified before model fitting',
                no_model_metrics=True,comparisons=comparisons,cross_user_temporal_disclosure=CROSS_USER_LIMITATION)


def export(con,query,path):
    # DuckDB COPY uses deterministic single-threaded ordering and the pinned writer version.
    con.execute(f'COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 10000)',[str(path)])


def verify_files(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'split_manifest.json').read_text())
    for filename,record in manifest['files'].items():
        if Path(filename).name!=filename:raise DataValidationError('Invalid split artifact name')
        path=directory/filename
        if not path.is_file() or sha256(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:
            raise DataValidationError(f'Split artifact hash mismatch: {filename}')
    actual={p.name for p in directory.iterdir() if p.is_file()}
    if actual!=set(manifest['files'])|{'split_manifest.json'}:
        raise DataValidationError('Unexpected or missing split files')
    return manifest


def materialize(con,destination,raw_report,config,mode,ratios,version):
    destination=Path(destination)
    processing=code_hashes()
    lock_hash=sha256(ROOT/'uv.lock') if (ROOT/'uv.lock').exists() else None
    identity=dict(dataset=raw_report['raw_sha256'],config_sha256=digest(config),mode=mode,
                  ratios=list(ratios),version=version,processing_source_hashes=processing,
                  environment_lock_sha256=lock_hash,python=sys.version.split()[0],duckdb=duckdb.__version__)
    evaluation_id=digest(identity)
    if destination.exists():
        existing=verify_files(destination)
        if existing['evaluation_id']!=evaluation_id:
            raise FileExistsError(f'{destination} is another immutable version; choose a new output version')
        return existing
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='small-prepare-',dir=destination.parent) as work:
        work=Path(work)
        assign(con,mode=mode,ratios=ratios)
        integrity=assignment_integrity(con,mode=mode)
        partition_stats=records(summaries(con))
        evaluation_tables(con)
        cohorts=cohort_summary(con)
        outputs={p+'.parquet':f"SELECT * EXCLUDE(partition) FROM assigned WHERE partition='{p}' ORDER BY userId,timestamp,movieId" for p in PARTITIONS}
        outputs.update({
            'movies.parquet':'SELECT * FROM movies ORDER BY movieId',
            'item_eligibility.parquet':'SELECT * FROM item_eligibility ORDER BY movieId',
            'user_eval_cohorts.parquet':'SELECT * FROM user_eval_cohorts ORDER BY partition,userId',
            'evaluation_targets.parquet':'SELECT * FROM evaluation_targets ORDER BY partition,userId,movieId',
            'user_mapping.parquet':"SELECT userId,(row_number() OVER(ORDER BY userId)-1)::INTEGER AS user_index FROM (SELECT DISTINCT userId FROM assigned WHERE partition='train') ORDER BY userId",
            'movie_mapping.parquet':'SELECT movieId,(row_number() OVER(ORDER BY movieId)-1)::INTEGER AS item_index FROM movies ORDER BY movieId',
            'user_cutoffs.parquet':'''SELECT userId,count(*) AS total_rows,count(DISTINCT timestamp) AS timestamp_groups,
                max(timestamp) FILTER(WHERE partition='train') AS train_max_timestamp,
                min(timestamp) FILTER(WHERE partition='validation') AS validation_min_timestamp,
                max(timestamp) FILTER(WHERE partition='validation') AS validation_max_timestamp,
                min(timestamp) FILTER(WHERE partition='test') AS test_min_timestamp
                FROM assigned GROUP BY userId ORDER BY userId''',
        })
        for filename,query in outputs.items():export(con,query,work/filename)
        inner=inner_diagnostics(con,mode=mode,history_percent=config['protocol']['inner_history_percent'])
        inner_integrity=assignment_integrity(con,'outer_train','inner_assigned',mode,inner=True)
        for partition,filename in [('train','ranker_snapshot_train.parquet'),('labels','ranker_label_window.parquet')]:
            export(con,f"SELECT * EXCLUDE(partition) FROM inner_assigned WHERE partition='{partition}' ORDER BY userId,timestamp,movieId",work/filename)
        export(con,'SELECT * FROM user_eval_cohorts ORDER BY userId',work/'ranker_query_diagnostics.parquet')
        export(con,'''SELECT userId,max(timestamp) FILTER(WHERE partition='train') AS snapshot_max_timestamp,
               min(timestamp) FILTER(WHERE partition='labels') AS label_min_timestamp,
               max(timestamp) FILTER(WHERE partition='labels') AS label_max_timestamp
               FROM inner_assigned GROUP BY userId ORDER BY userId''',work/'ranker_user_cutoffs.parquet')
        files={}
        for path in sorted(work.glob('*.parquet')):
            relation=con.read_parquet(str(path))
            files[path.name]=dict(bytes=path.stat().st_size,sha256=sha256(path),rows=relation.count('*').fetchone()[0],
                                  schema=dict(zip(relation.columns,map(str,relation.types))))
        manifest=dict(dataset_id=DATASET_ID,version=version,protocol_name=mode+'_chronological_'+('_'.join(map(str,ratios))),
            evaluation_id=evaluation_id,raw_sha256=raw_report['raw_sha256'],
            raw_provenance_manifest_sha256=raw_report['raw_sha256']['provenance.json'],
            policy=POLICY,ratio_targets=list(ratios),timestamp_policy=config['protocol']['timestamp_ties'],
            rounding_rule='floor(n*cumulative_percent/100); nearest cumulative whole-timestamp prefix; earlier prefix wins equal distance',
            short_history_rule=config['protocol']['short_history'],
            nonempty_rule='>=3 groups reserve one group per partition; two groups Train/Test; one group Train only',
            processing_source_hashes=processing,config_sha256=digest(config),
            python_version=sys.version.split()[0],duckdb_version=duckdb.__version__,
            environment_lock_sha256=lock_hash,
            partition_statistics=partition_stats,evaluation_cohorts=cohorts,historical_ranker_feasibility=inner,
            integrity=integrity,historical_integrity=inner_integrity,files=files,
            limitations=LIMITATIONS+([CROSS_USER_LIMITATION] if mode=='per_user' else ['Strict global chronology has limited warm-user coverage.']),
            selection_policy='Validation only for future tuning/early stopping; Test locked after protocol diagnostics; no models fitted',
            mappings='users sorted from Train only; movies sorted from raw metadata; eligibility is independently Train-derived; unknown user index -1',
            ranker_query_key='(snapshot evaluation_id, userId); one snapshot here; never merge snapshots by userId alone')
        write_immutable_json(work/'split_manifest.json',manifest)
        verify_files(work)
        shutil.copytree(work,destination)  # exclusive: never overwrite a user's existing directory
    return verify_files(destination)


def prepare(raw_dir,output_root,config,diagnostics_path=None):
    con,raw_report=raw_connection(raw_dir,config['dataset'])
    try:
        comparison=compare_protocols(con,config['protocol']['inner_history_percent'])
        results={}
        for mode,ratios,key in [('per_user',(70,15,15),'version'),('global',(80,10,10),'secondary_version')]:
            version=config['protocol'][key]
            results[version]=materialize(con,Path(output_root)/version,raw_report,config,mode,ratios,version)
        if diagnostics_path:write_immutable_json(diagnostics_path,comparison)
        return results,comparison
    finally:con.close()


def validate_splits(raw_dir,directory,config):
    """Recompute assignments and label-side tables from immutable raw data; never fit."""
    directory=Path(directory);manifest=verify_files(directory)
    con,raw_report=raw_connection(raw_dir,config['dataset'])
    try:
        if manifest['dataset_id']!=DATASET_ID or manifest['raw_sha256']!=raw_report['raw_sha256']:
            raise DataValidationError('Split/raw provenance disagreement')
        if manifest['config_sha256']!=digest(config) or manifest['processing_source_hashes']!=code_hashes():
            raise DataValidationError('Frozen split/config/processing source mismatch; review a new version')
        mode='per_user' if manifest['protocol_name'].startswith('per_user_') else 'global'
        ratios=tuple(manifest['ratio_targets'])
        if (mode,ratios) not in [('per_user',(70,15,15)),('global',(80,10,10))]:
            raise DataValidationError('Unexpected frozen protocol')
        assign(con,mode=mode,ratios=ratios)
        check=assignment_integrity(con,mode=mode)
        def same(query,filename):
            actual=con.read_parquet(str(directory/filename));actual.create_view('_actual',replace=True)
            con.execute('CREATE OR REPLACE TEMP VIEW _expected AS '+query)
            n=con.sql('''SELECT count(*) FROM ((SELECT * FROM _actual EXCEPT ALL SELECT * FROM _expected)
                  UNION ALL (SELECT * FROM _expected EXCEPT ALL SELECT * FROM _actual))''').fetchone()[0]
            if n:raise DataValidationError(f'Incorrect frozen artifact values: {filename}')
        for partition in PARTITIONS:
            filename=partition+'.parquet'
            relation=con.read_parquet(str(directory/filename))
            if dict(zip(relation.columns,map(str,relation.types)))!=SCHEMA:
                raise DataValidationError('Partition schema/dtype mismatch: '+filename)
            same(f"SELECT * EXCLUDE(partition) FROM assigned WHERE partition='{partition}'",filename)
            rows=relation.project('userId,timestamp,movieId').fetchall()
            if rows!=sorted(rows):raise DataValidationError('Unsorted partition: '+filename)
        evaluation_tables(con)
        for table in ['item_eligibility','user_eval_cohorts','evaluation_targets']:
            same('SELECT * FROM '+table,table+'.parquet')
        same('SELECT * FROM movies','movies.parquet')
        same("SELECT userId,(row_number() OVER(ORDER BY userId)-1)::INTEGER AS user_index FROM (SELECT DISTINCT userId FROM assigned WHERE partition='train')",'user_mapping.parquet')
        same('SELECT movieId,(row_number() OVER(ORDER BY movieId)-1)::INTEGER AS item_index FROM movies','movie_mapping.parquet')
        if records(summaries(con))!=manifest['partition_statistics'] or cohort_summary(con)!=manifest['evaluation_cohorts']:
            raise DataValidationError('Manifest count/cohort mismatch')
        inner=inner_diagnostics(con,mode=mode,history_percent=config['protocol']['inner_history_percent'])
        assignment_integrity(con,'outer_train','inner_assigned',mode,inner=True)
        for partition,filename in [('train','ranker_snapshot_train.parquet'),('labels','ranker_label_window.parquet')]:
            same(f"SELECT * EXCLUDE(partition) FROM inner_assigned WHERE partition='{partition}'",filename)
        same('SELECT * FROM user_eval_cohorts','ranker_query_diagnostics.parquet')
        if inner!=manifest['historical_ranker_feasibility']:
            raise DataValidationError('Historical ranker feasibility mismatch')
        return dict(dataset_id=DATASET_ID,version=manifest['version'],evaluation_id=manifest['evaluation_id'],
                    split_manifest_sha256=sha256(directory/'split_manifest.json'),
                    files_verified=len(manifest['files']),recomputed_from_raw=True,**check)
    finally:con.close()
