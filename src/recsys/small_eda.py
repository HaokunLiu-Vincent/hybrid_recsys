"""Read-only DuckDB exploratory diagnostics. No saved splits, candidate generation or model fitting."""
import duckdb
import pandas as pd
from .small_data import load_raw


def analysis_connection(data_dir):
    ratings,movies,quality = load_raw(data_dir)
    connection = duckdb.connect(':memory:')
    connection.execute("SET TimeZone='UTC'")
    connection.register('ratings',ratings)
    connection.register('movies',movies)
    return connection,quality


def cutoff_options(connection):
    maximum = connection.sql('SELECT max(timestamp) FROM ratings').fetchone()[0]
    end = pd.Timestamp(maximum,unit='s',tz='UTC').normalize()+pd.Timedelta(days=1)
    result=[]
    for name,val_months,test_months in [('calendar_3m_3m',3,3),('calendar_6m_6m',6,6),('calendar_6m_12m',6,12)]:
        test_start = end-pd.DateOffset(months=test_months)
        validation_start = test_start-pd.DateOffset(months=val_months)
        result.append(dict(alternative=name,validation_start=int(validation_start.timestamp()),
                           test_start=int(test_start.timestamp()),end_exclusive=int(end.timestamp()),
                           definition=f'{val_months}-month Validation then {test_months}-month Test; rolling calendar offsets, UTC midnight'))
    validation_start,test_start=connection.sql('SELECT quantile_disc(timestamp,0.8),quantile_disc(timestamp,0.9) FROM ratings').fetchone()
    result.append(dict(alternative='timestamp_80_10_10',validation_start=int(validation_start),test_start=int(test_start),
                       end_exclusive=int(end.timestamp()),definition='Observed timestamp 80th/90th discrete quantiles; all boundary ties stay together'))
    return result


def window_diagnostics(connection,history_end,label_start,label_end):
    if not history_end <= label_start < label_end:
        raise ValueError('Require history_end <= label_start < label_end; history uses strictly earlier timestamps')
    # Bind values: no interpolation of arbitrary SQL or future data into training-derived aggregates.
    query='''
    WITH history AS (SELECT * FROM ratings WHERE timestamp < ?),
    future AS (SELECT * FROM ratings WHERE timestamp >= ? AND timestamp < ?),
    catalog AS (SELECT DISTINCT movieId FROM history),
    user_history AS (
      SELECT userId,count(*) AS train_ratings,count(*) FILTER(WHERE rating >= 4) AS train_positives,
             max(timestamp) AS last_train_submission FROM history GROUP BY userId
    ), annotated AS (
      SELECT f.*, c.movieId IS NOT NULL AS train_known_item,
             EXISTS(SELECT 1 FROM history h WHERE h.userId=f.userId AND h.movieId=f.movieId) AS seen_train
      FROM future f LEFT JOIN catalog c USING(movieId)
    ), labels AS (
      SELECT userId,count(*) AS future_rows,
             count(*) FILTER(WHERE rating >= 4) AS observed_future_positive,
             count(*) FILTER(WHERE rating < 4) AS observed_future_low,
             count(*) FILTER(WHERE rating >= 4 AND train_known_item AND NOT seen_train) AS eligible_future_positive,
             count(*) FILTER(WHERE rating >= 4 AND NOT train_known_item) AS positive_not_train_known,
             count(*) FILTER(WHERE rating >= 4 AND train_known_item AND seen_train) AS positive_seen_train,
             min(timestamp) AS first_future_submission
      FROM annotated GROUP BY userId
    )
    SELECT l.*,coalesce(h.train_ratings,0) AS train_ratings,coalesce(h.train_positives,0) AS train_positives,
           first_future_submission-h.last_train_submission AS train_to_label_gap_seconds,
           CASE WHEN coalesce(h.train_positives,0)=0 THEN 'cold_positive_history'
                WHEN h.train_positives < 5 THEN 'sparse_1_4'
                WHEN h.train_positives < 20 THEN 'moderate_5_19' ELSE 'warm_20_plus' END AS exploratory_cohort
    FROM labels l LEFT JOIN user_history h USING(userId) ORDER BY userId
    '''
    users=connection.execute(query,[int(history_end),int(label_start),int(label_end)]).df()
    n_rows=int(users.future_rows.sum())
    positives=int(users.observed_future_positive.sum())
    eligible=int(users.eligible_future_positive.sum())
    history=connection.execute('''SELECT count(*),count(DISTINCT userId),count(DISTINCT movieId),
                               count(*) FILTER(WHERE rating>=4) FROM ratings WHERE timestamp < ?''',[int(history_end)]).fetchone()
    burst=users.train_to_label_gap_seconds.notna() & users.train_to_label_gap_seconds.le(1800)
    summary=dict(history_rows=int(history[0]),history_users=int(history[1]),history_items=int(history[2]),
                 history_positive=int(history[3]),future_rows=n_rows,future_users=len(users),
                 future_positive=positives,eligible_positive=eligible,
                 positive_not_train_known=int(users.positive_not_train_known.sum()),
                 positive_seen_train=int(users.positive_seen_train.sum()),
                 train_known_positive_coverage=(positives-int(users.positive_not_train_known.sum()))/positives if positives else None,
                 eligible_positive_fraction=eligible/positives if positives else None,
                 eligible_users=int(users.eligible_future_positive.gt(0).sum()),
                 eligible_users_ge5_train_positive=int((users.eligible_future_positive.gt(0)&users.train_positives.ge(5)).sum()),
                 users_without_train_ratings=int(users.train_ratings.eq(0).sum()),
                 users_without_positive_history=int(users.train_positives.eq(0).sum()),
                 future_users_without_positive=int(users.observed_future_positive.eq(0).sum()),
                 future_users_without_eligible_positive=int(users.eligible_future_positive.eq(0).sum()),
                 users_crossing_boundary_within_30m=int(burst.sum()))
    return summary,users


def split_feasibility(connection,options=None):
    """Compare alternative views only. No assignment table or Parquet split is created."""
    options=cutoff_options(connection) if options is None else options
    summaries,cohorts,internal=[] ,[],[]
    for option in options:
        name=option['alternative'];v,t,e=[option[k] for k in ['validation_start','test_start','end_exclusive']]
        for phase,start,end in [('validation',v,t),('test',t,e)]:
            summary,users=window_diagnostics(connection,v,start,end)
            summaries.append(dict(alternative=name,window=phase,
                train_end_exclusive_utc=pd.Timestamp(v,unit='s',tz='UTC').isoformat(),
                label_start_utc=pd.Timestamp(start,unit='s',tz='UTC').isoformat(),
                label_end_exclusive_utc=pd.Timestamp(end,unit='s',tz='UTC').isoformat(),**summary))
            if len(users):
                grouped=users.groupby('exploratory_cohort',observed=True).agg(
                    users=('userId','size'),future_positive=('observed_future_positive','sum'),
                    eligible_positive=('eligible_future_positive','sum'),
                    eligible_users=('eligible_future_positive',lambda x:int(x.gt(0).sum()))).reset_index()
                grouped['alternative'],grouped['window']=name,phase
                cohorts.append(grouped)
        cutoff=connection.execute('SELECT quantile_disc(timestamp,0.8) FROM ratings WHERE timestamp < ?',[v]).fetchone()[0]
        if cutoff is None or cutoff>=v:
            internal.append(dict(alternative=name,status='insufficient historical timestamps'))
        else:
            summary,_=window_diagnostics(connection,int(cutoff),int(cutoff),v)
            internal.append(dict(alternative=name,status='exploratory; no candidates or training examples generated',
                snapshot_end_exclusive_utc=pd.Timestamp(cutoff,unit='s',tz='UTC').isoformat(),
                internal_labels_end_exclusive_utc=pd.Timestamp(v,unit='s',tz='UTC').isoformat(),
                potential_ranker_queries_upper_bound=summary['eligible_users'],**summary))
    return pd.DataFrame(summaries),pd.concat(cohorts,ignore_index=True) if cohorts else pd.DataFrame(),pd.DataFrame(internal)
