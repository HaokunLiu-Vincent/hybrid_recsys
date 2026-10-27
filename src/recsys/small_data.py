"""MovieLens Latest Small raw-data validation and provenance, with no model or split fitting."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import csv
import hashlib
import importlib.metadata
import json
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile

import numpy as np
import pandas as pd

DATASET_ID = 'grouplens/movielens-latest-small'
DATASET_URL = 'https://www.kaggle.com/datasets/'+DATASET_ID
COLUMNS = {'ratings.csv':['userId','movieId','rating','timestamp'],
           'movies.csv':['movieId','title','genres']}
DTYPES = {'ratings.csv':{'userId':'int64','movieId':'int64','rating':'float32','timestamp':'int64'},
          'movies.csv':{'movieId':'int64','title':'string','genres':'string'}}


class DataValidationError(ValueError):
    """Raw files fail a declared schema or integrity invariant; no automatic repair."""


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            result.update(block)
    return result.hexdigest()


def validate_frames(ratings, movies):
    for name,frame in [('ratings.csv',ratings),('movies.csv',movies)]:
        if list(frame.columns) != COLUMNS[name]:
            raise DataValidationError(f'{name}: expected columns {COLUMNS[name]}, observed {list(frame.columns)}')
        if frame.empty or frame.isna().any().any():
            raise DataValidationError(f'{name}: empty file or missing values; inspect raw data, do not impute silently')
    for frame,cols in [(ratings,['userId','movieId','timestamp']),(movies,['movieId'])]:
        for col in cols:
            if not pd.api.types.is_integer_dtype(frame[col]):
                raise DataValidationError(f'{col}: integer values required')
            if ((frame[col] < 0) if col=='timestamp' else (frame[col] <= 0)).any():
                raise DataValidationError(f'{col}: invalid negative/zero identifier or negative timestamp')
    if not pd.api.types.is_numeric_dtype(ratings.rating):
        raise DataValidationError('rating: numeric values required')
    values = ratings.rating.to_numpy()
    if not np.isfinite(values).all() or ((values < .5)|(values > 5)|(values*2 % 1 != 0)).any():
        raise DataValidationError('rating: finite half-star values in [0.5, 5.0] required')
    if ratings.duplicated(['userId','movieId']).any():
        raise DataValidationError('Duplicate user/movie rating pairs; no automatic deduplication')
    if movies.movieId.duplicated().any():
        raise DataValidationError('Duplicate metadata movie IDs')
    unknown = ~ratings.movieId.isin(movies.movieId)
    if unknown.any():
        raise DataValidationError(f'{int(unknown.sum())} ratings reference movies absent from movies.csv')
    if not all(movies[col].map(lambda x:isinstance(x,str)).all() for col in ['title','genres']):
        raise DataValidationError('Title and genre values must be strings')
    if (movies.title.str.strip().eq('')|movies.genres.str.strip().eq('')).any():
        raise DataValidationError('Empty title or genre; the explicit no-genres sentinel is allowed')
    try:
        timestamps = pd.to_datetime(ratings.timestamp,unit='s',utc=True,errors='raise')
    except (ValueError,OverflowError) as exc:
        raise DataValidationError('Timestamp cannot be represented as a UTC submission time') from exc
    return dict(passed=True,missing_values={name:frame.isna().sum().astype(int).to_dict()
                for name,frame in [('ratings',ratings),('movies',movies)]},
                duplicate_user_movie_pairs=0,duplicate_movie_ids=0,unknown_movie_ratings=0,
                ratings=int(len(ratings)),metadata_movies=int(len(movies)),
                users=int(ratings.userId.nunique()),rated_movies=int(ratings.movieId.nunique()),
                metadata_movies_without_ratings=int((~movies.movieId.isin(ratings.movieId)).sum()),
                observed_positive_ratings=int((ratings.rating>=4).sum()),positive_definition='rating >= 4.0',
                observed_low_ratings=int((ratings.rating<4).sum()),
                rating_min=float(ratings.rating.min()),rating_max=float(ratings.rating.max()),
                timestamp_min=int(ratings.timestamp.min()),timestamp_max=int(ratings.timestamp.max()),
                timestamp_min_utc=timestamps.min().isoformat(),timestamp_max_utc=timestamps.max().isoformat())


def load_raw(data_dir, verify=True):
    root = Path(data_dir)
    if verify:
        verify_provenance(root)
    frames = []
    for name in COLUMNS:
        path = root/name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Required nonempty raw file missing: {path}. Run tools/download_movielens_small.py.')
        with path.open(encoding='utf-8-sig',newline='') as f:
            header = next(csv.reader(f),[])
        if header != COLUMNS[name]:
            raise DataValidationError(f'{name}: expected {COLUMNS[name]}, observed {header}')
        try:
            frames.append(pd.read_csv(path,dtype=DTYPES[name]))
        except (ValueError,TypeError,OverflowError) as exc:
            raise DataValidationError(f'{name}: invalid typed CSV values') from exc
    ratings,movies = frames
    quality = validate_frames(ratings,movies)
    for frame in [ratings,movies]:
        for col in ['userId','movieId']:
            if col in frame and frame[col].max() <= np.iinfo(np.int32).max:
                frame[col] = frame[col].astype('int32')
    return ratings,movies,quality


def verify_provenance(data_dir):
    root = Path(data_dir)
    manifest = json.loads((root/'provenance.json').read_text())
    if manifest.get('source',{}).get('dataset_id') != DATASET_ID:
        raise DataValidationError('Provenance belongs to another dataset/source')
    for required in COLUMNS:
        if required not in manifest['files']:
            raise DataValidationError(f'Manifest omits {required}')
    for name,record in manifest['files'].items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise DataValidationError('Unsafe manifest file path')
        path = root.joinpath(*rel.parts)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise DataValidationError(f'Raw file differs from acquisition manifest: {name}')
    return manifest


def safe_extract(archive_path, destination, max_uncompressed_bytes=256<<20):
    """Reject traversal, symlinks, duplicates, encrypted entries and unreasonable expansion."""
    root = Path(destination)
    root.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if sum(e.file_size for e in entries) > max_uncompressed_bytes:
            raise DataValidationError('Archive exceeds the explicit Small-dataset extraction budget')
        names = set()
        for entry in entries:
            rel = PurePosixPath(entry.filename)
            mode = entry.external_attr >> 16
            if (rel.is_absolute() or '..' in rel.parts or '\\' in entry.filename or not rel.parts or
                stat.S_ISLNK(mode) or entry.flag_bits & 1 or entry.filename in names):
                raise DataValidationError('Unsafe/duplicate/encrypted archive member')
            names.add(entry.filename)
        for entry in entries:
            path = root.joinpath(*PurePosixPath(entry.filename).parts)
            if entry.is_dir():
                path.mkdir(parents=True,exist_ok=True)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(entry) as src,path.open('xb') as dst:
                    shutil.copyfileobj(src,dst)


def required_root(extracted):
    matches = {name:list(Path(extracted).rglob(name)) for name in COLUMNS}
    if any(len(paths)!=1 for paths in matches.values()):
        raise DataValidationError('Expected exactly one ratings.csv and one movies.csv; layout is ambiguous')
    parents = {paths[0].parent for paths in matches.values()}
    if len(parents)!=1:
        raise DataValidationError('Required CSVs must share a dataset directory')
    return parents.pop()


def publish_archive(archive_path, data_dir, receipt):
    """Normalize a known Kaggle CLI download without overwriting any existing data directory."""
    destination = Path(data_dir)
    if destination.exists():
        raise FileExistsError(f'Destination already exists, refusing replacement: {destination}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ml-small-normalize-') as work:
        work = Path(work)
        safe_extract(archive_path,work/'extracted')
        source_root = required_root(work/'extracted')
        payload = work/'payload'
        shutil.copytree(source_root,payload)
        if (payload/'provenance.json').exists() or (payload/'_source').exists():
            raise DataValidationError('Archive conflicts with reserved provenance paths')
        source = payload/'_source';source.mkdir()
        shutil.copyfile(archive_path,source/'kaggle-download.zip')
        ratings,movies,quality = load_raw(payload,verify=False)
        readmes = [p for p in payload.iterdir() if p.name.lower() in ('readme.md','readme.txt')]
        expected = {}
        publisher_date = None
        if len(readmes)==1:
            text = readmes[0].read_text(encoding='utf-8')
            for key,word in [('ratings','ratings'),('metadata_movies','movies'),('users','users')]:
                match = re.search(r'([0-9][0-9,]*)\s+'+word,text,re.IGNORECASE)
                if match:expected[key]=int(match.group(1).replace(',',''))
            date = re.search(r'generated on ([^.]+)\.',text)
            publisher_date = date.group(1) if date else None
        discrepancies = {key:dict(publisher_expected=value,observed=quality[key]) for key,value in expected.items() if quality[key]!=value}
        records = {}
        for path in sorted(payload.rglob('*')):
            if path.is_file():
                records[str(path.relative_to(payload))]=dict(bytes=path.stat().st_size,sha256=sha256(path))
        for name,frame in [('ratings.csv',ratings),('movies.csv',movies)]:
            records[name].update(rows=len(frame),schema={col:str(dtype) for col,dtype in frame.dtypes.items()},
                                 csv_columns=list(frame.columns))
        manifest = dict(manifest_version='small_raw_v1',
            source=dict(dataset_id=DATASET_ID,url=DATASET_URL,kaggle_dataset_version=None,
                        publisher_generated_date=publisher_date,**receipt),
            downloaded_at_utc=receipt.get('downloaded_at_utc'),
            verified_at_utc=datetime.now(timezone.utc).isoformat(),
            files=records,quality=quality,publisher_expected_counts=expected,discrepancies=discrepancies,
            licence=dict(kaggle_label=receipt.get('kaggle_licence_label'),reference=readmes[0].name if readmes else None,
                         notice='Keep and consult the original bundled publisher README licence/citation conditions; not a CC0 assumption.'),
            modeling_inputs=list(COLUMNS),ancillary_files='Preserved raw; tags.csv and links.csv are not loaded for analysis/modeling')
        (payload/'provenance.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        # copytree's exclusive destination creation refuses a pre-existing or concurrent dataset directory.
        shutil.copytree(payload,destination)
        verify_provenance(destination)
    return manifest


def download_dataset(data_dir, kaggle_executable='kaggle'):
    destination = Path(data_dir)
    if destination.exists():
        manifest = verify_provenance(destination)
        load_raw(destination)
        print('Existing Kaggle dataset verified; no download or overwrite.')
        return manifest
    with tempfile.TemporaryDirectory(prefix='kaggle-ml-small-') as work:
        command = [str(kaggle_executable),'datasets','download',DATASET_ID,'-p',work,'--quiet']
        result = subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=180)
        if result.returncode:
            raise RuntimeError('Kaggle download failed (authentication/network). Configure Kaggle outside the repository; '
                               'no alternative dataset source was attempted. CLI exit code: '+str(result.returncode))
        archives = list(Path(work).glob('*.zip'))
        if len(archives)!=1:
            raise DataValidationError('Kaggle CLI did not produce exactly one ZIP archive')
        licence = re.search(r'License\(s\):\s*([^\r\n]+)',result.stdout)
        receipt = dict(kaggle_licence_label=licence.group(1).strip() if licence else None,acquisition_tool='official Kaggle CLI',kaggle_cli_version=importlib.metadata.version('kaggle'),
                       command=['kaggle','datasets','download',DATASET_ID,'-p','<temporary directory>','--quiet'],
                       downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                       version_status='Kaggle version unavailable from download; immutable content identified by SHA256')
        return publish_archive(archives[0],destination,receipt)
