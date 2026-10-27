"""DuckDB split assignment and label-side feasibility; no fitting or candidate generation."""
import re

PARTITIONS = ('train', 'validation', 'test')
CROSS_USER_LIMITATION = (
    'Per-user chronology only: other users\' Train events can be later than this user\'s '
    'held-out events. Cross-user future information is accepted for this educational '
    'offline benchmark; it is not a globally leakage-free production-time evaluation.'
)
POLICY = dict(positive_threshold=4.0, catalog='train_observed', seen='all_train_rated',
              history='frozen_train', primary_metric_denominator='eligible held-out positives',
              alternative_metric_denominator='all observed held-out positives; separate diagnostic',
              positive_history_cohorts=['cold_0', 'sparse_1_4', 'moderate_5_19', 'warm_20_plus'])


def name(value):
    if not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*', value):
        raise ValueError('SQL table identifier must be a simple internal name')
    return value


def assign(con, source='ratings', output='assigned', mode='per_user', ratios=(70, 15, 15)):
    """Nearest cumulative group to floor(n*p/100), reserving nonempty windows when possible.

    Ties choose the earlier prefix. For three windows, >=3 timestamp groups reserve at
    least one per window; two groups become Train/Test; one group is entirely Train.
    Two-window inner snapshots reserve one group per window when >=2 groups exist.
    """
    source, output = name(source), name(output)
    if mode not in ('per_user', 'global') or len(ratios) not in (2, 3):
        raise ValueError('Expected per_user/global and two or three ratio components')
    if sum(ratios) != 100 or any(type(p) is not int or p <= 0 for p in ratios):
        raise ValueError('Positive integer ratio percentages must sum to 100')
    key = 'userId' if mode == 'per_user' else '0'
    con.execute(f'''CREATE OR REPLACE TEMP TABLE _groups AS
        SELECT {key} AS split_group, timestamp, count(*) AS n
        FROM {source} GROUP BY split_group,timestamp''')
    con.execute('''CREATE OR REPLACE TEMP TABLE _prefix AS SELECT *,
        row_number() OVER w AS g, count(*) OVER(PARTITION BY split_group) AS ng,
        sum(n) OVER w AS prefix, sum(n) OVER(PARTITION BY split_group) AS total
        FROM _groups WINDOW w AS(PARTITION BY split_group ORDER BY timestamp ROWS UNBOUNDED PRECEDING)''')
    reserve = len(ratios) - 1
    con.execute(f'''CREATE OR REPLACE TEMP TABLE _first AS SELECT * FROM _prefix
        WHERE g <= CASE WHEN ng >= {len(ratios)} THEN ng-{reserve} ELSE 1 END
        QUALIFY row_number() OVER(PARTITION BY split_group
            ORDER BY abs(prefix-(total*{ratios[0]}//100)),prefix)=1''')
    if len(ratios) == 3:
        con.execute(f'''CREATE OR REPLACE TEMP TABLE _second AS SELECT p.* FROM _prefix p
            JOIN _first f USING(split_group) WHERE p.ng>=3 AND p.g>f.g AND p.g<p.ng
            QUALIFY row_number() OVER(PARTITION BY split_group
                ORDER BY abs(p.prefix-(p.total*{sum(ratios[:2])}//100)),p.prefix)=1''')
        con.execute('''CREATE OR REPLACE TEMP TABLE _bounds AS
            SELECT f.split_group,f.ng,f.timestamp AS train_last,
                   coalesce(s.timestamp,f.timestamp) AS validation_last
            FROM _first f LEFT JOIN _second s USING(split_group)''')
        condition = "CASE WHEN r.timestamp<=b.train_last THEN 'train' WHEN b.ng=2 THEN 'test' WHEN r.timestamp<=b.validation_last THEN 'validation' ELSE 'test' END"
    else:
        con.execute('CREATE OR REPLACE TEMP TABLE _bounds AS SELECT split_group,ng,timestamp AS train_last FROM _first')
        condition = "CASE WHEN r.timestamp<=b.train_last THEN 'train' ELSE 'labels' END"
    join = 'r.userId=b.split_group' if mode == 'per_user' else 'b.split_group=0'
    con.execute(f'''CREATE OR REPLACE TEMP TABLE {output} AS
        SELECT r.*, {condition}::VARCHAR AS partition FROM {source} r JOIN _bounds b ON {join}''')


def summaries(con, table='assigned'):
    table = name(table)
    return con.sql(f'''SELECT partition,count(*) AS interactions,
        count(DISTINCT userId) AS users,count(DISTINCT movieId) AS movies,
        count(*) FILTER(WHERE rating>=4) AS positives,min(timestamp) AS min_timestamp,
        max(timestamp) AS max_timestamp,count(*)*100.0/sum(count(*)) OVER() AS percent
        FROM {table} GROUP BY partition ORDER BY partition''').df()


def evaluation_tables(con, table='assigned', label_partitions=('validation', 'test')):
    """All users x label windows retained; catalog/history aggregates use Train only."""
    table = name(table)
    if not label_partitions or not set(label_partitions) <= {'validation', 'test', 'labels'}:
        raise ValueError('Unexpected label partition')
    values = ','.join(f"('{p}')" for p in label_partitions)
    con.execute(f'''CREATE OR REPLACE TEMP TABLE item_eligibility AS
        SELECT m.movieId,count(r.userId)::BIGINT AS train_interactions,
               count(r.userId) FILTER(WHERE r.rating>=4)::BIGINT AS train_positives,
               count(r.userId)>0 AS eligible
        FROM movies m LEFT JOIN {table} r ON m.movieId=r.movieId AND r.partition='train'
        GROUP BY m.movieId''')
    con.execute(f'''CREATE OR REPLACE TEMP TABLE evaluation_targets AS WITH annotated AS (
        SELECT r.partition,r.userId,r.movieId,e.eligible AS train_known,
               EXISTS(SELECT 1 FROM {table} h WHERE h.partition='train' AND h.userId=r.userId AND h.movieId=r.movieId) AS seen_train,
               e.eligible AS catalog_eligible
        FROM {table} r JOIN item_eligibility e USING(movieId)
        WHERE r.partition IN ({','.join(repr(p) for p in label_partitions)}) AND r.rating>=4)
        SELECT * EXCLUDE(catalog_eligible),catalog_eligible AND NOT seen_train AS eligible FROM annotated''')
    con.execute(f'''CREATE OR REPLACE TEMP TABLE user_eval_cohorts AS
        WITH history AS (SELECT userId,count(*) AS train_rows,
            count(*) FILTER(WHERE rating>=4) AS train_positives FROM {table} WHERE partition='train' GROUP BY userId),
        future AS (SELECT partition,userId,count(*) AS future_rows,
            count(*) FILTER(WHERE rating>=4) AS future_positives FROM {table} GROUP BY partition,userId),
        targets AS (SELECT partition,userId,count(*) FILTER(WHERE train_known) AS catalog_known_positives,
            count(*) FILTER(WHERE eligible) AS eligible_positives FROM evaluation_targets GROUP BY partition,userId),
        base AS (
        SELECT u.userId,w.partition,coalesce(h.train_rows,0)::BIGINT AS train_rows,
            coalesce(h.train_positives,0)::BIGINT AS train_positives,
            coalesce(f.future_rows,0)::BIGINT AS future_rows,coalesce(f.future_positives,0)::BIGINT AS future_positives,
            coalesce(t.catalog_known_positives,0)::BIGINT AS catalog_known_positives,
            coalesce(t.eligible_positives,0)::BIGINT AS eligible_positives,
            (SELECT count(*) FROM item_eligibility WHERE eligible)-coalesce(h.train_rows,0) AS available_candidates
        FROM (SELECT DISTINCT userId FROM {table}) u CROSS JOIN (VALUES {values}) w(partition)
        LEFT JOIN history h USING(userId) LEFT JOIN future f ON f.userId=u.userId AND f.partition=w.partition
        LEFT JOIN targets t ON t.userId=u.userId AND t.partition=w.partition)
        SELECT *,train_rows=0 AS no_train_history,train_positives=0 AS no_positive_history,
            CASE WHEN train_positives=0 THEN 'cold_0' WHEN train_positives<5 THEN 'sparse_1_4'
                 WHEN train_positives<20 THEN 'moderate_5_19' ELSE 'warm_20_plus' END AS cohort,
            eligible_positives>0 AS metric_eligible,available_candidates=0 AS no_candidates,
            future_positives=0 AS no_positive_targets,
            future_positives>0 AND eligible_positives=0 AS no_retrievable_targets FROM base''')


def cohort_summary(con):
    frame = con.sql('''SELECT partition,count(*) FILTER(WHERE future_rows>0) AS future_users,
        count(*) FILTER(WHERE future_rows>0 AND no_train_history) AS users_no_train_history,
        count(*) FILTER(WHERE future_rows>0 AND no_positive_history) AS users_no_positive_history,
        count(*) FILTER(WHERE future_rows>0 AND train_positives BETWEEN 1 AND 4) AS users_1_4_train_positive,
        count(*) FILTER(WHERE future_rows>0 AND train_positives BETWEEN 5 AND 19) AS users_5_19_train_positive,
        count(*) FILTER(WHERE future_rows>0 AND train_positives>=20) AS users_ge20_train_positive,
        count(*) FILTER(WHERE future_positives>0) AS users_with_heldout_positive,
        count(*) FILTER(WHERE metric_eligible) AS eligible_positive_users,
        count(*) FILTER(WHERE no_retrievable_targets) AS users_no_retrievable_targets,
        count(*) FILTER(WHERE future_rows>0 AND no_candidates) AS users_no_available_candidates,
        sum(future_positives) AS heldout_positives,sum(catalog_known_positives) AS train_known_positives,
        sum(eligible_positives) AS eligible_positives,
        sum(future_positives-catalog_known_positives) AS catalog_excluded_positives,
        sum(future_positives) FILTER(WHERE no_train_history) AS positives_from_no_history_users,
        sum(future_positives) FILTER(WHERE no_positive_history) AS positives_from_no_positive_history_users,
        count(*) FILTER(WHERE metric_eligible AND available_candidates>eligible_positives) AS queries_with_possible_contrast,
        count(*) FILTER(WHERE metric_eligible AND train_rows>=10) AS eligible_queries_ge10_history_rows,
        count(*) FILTER(WHERE metric_eligible AND train_positives>=5) AS eligible_queries_ge5_history_positive,
        sum(least(eligible_positives,100)) AS ideal_top100_positive_hits,
        sum(least(eligible_positives,600)) AS ideal_union600_positive_hits
        FROM user_eval_cohorts GROUP BY partition ORDER BY partition''').df().fillna(0)
    records = []
    for row in frame.to_dict('records'):
        row = {k:int(v) if k!='partition' else v for k,v in row.items()}
        def fraction(a,b):return a/b if b else None
        row.update(cold_history_user_share=fraction(row['users_no_train_history'],row['future_users']),
                   cold_history_positive_share=fraction(row['positives_from_no_history_users'],row['heldout_positives']),
                   train_known_positive_coverage=fraction(row['train_known_positives'],row['heldout_positives']),
                   eligible_target_coverage=fraction(row['eligible_positives'],row['heldout_positives']))
        records.append(row)
    return records


def inner_diagnostics(con, outer='assigned', mode='per_user', history_percent=80):
    outer = name(outer)
    con.execute(f"CREATE OR REPLACE TEMP TABLE outer_train AS SELECT * EXCLUDE(partition) FROM {outer} WHERE partition='train'")
    assign(con,'outer_train','inner_assigned',mode,(history_percent,100-history_percent))
    evaluation_tables(con,'inner_assigned',('labels',))
    result = cohort_summary(con)[0]
    result.update(snapshot_history_rows=con.sql("SELECT count(*) FROM inner_assigned WHERE partition='train'").fetchone()[0],
                  label_window_rows=con.sql("SELECT count(*) FROM inner_assigned WHERE partition='labels'").fetchone()[0],
                  train_known_movies=con.sql('SELECT count(*) FROM item_eligibility WHERE eligible').fetchone()[0],
                  sufficient_history_convention='Report >=10 historical ratings and >=5 historical positives separately; not an inclusion rule',
                  actual_effective_ranker_queries=None,actual_candidate_pool_recall=None,
                  limitation='Catalog eligibility/possible contrast are upper bounds; no candidates, models or ranker examples were produced.',
                  temporal_interpretation=CROSS_USER_LIMITATION if mode=='per_user' else 'Strict global historical prefix inside external Train')
    return result
