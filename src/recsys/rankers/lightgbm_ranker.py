"""LambdaMART learns only from out-of-time internal Train candidate labels."""
import numpy as np
from ..ranking_features import FEATURES
from ..runtime import require_memory


def fit_ranker(train, validation, config):
    import lightgbm as lgb
    require_memory(cpu=(len(train)+len(validation))*len(FEATURES)*4*8, stage='LightGBM candidate features')
    train = train.sort_values(['userId','movieId'],kind='stable')
    validation = validation.sort_values(['userId','movieId'],kind='stable')
    if train.empty or train.label.sum() == 0 or train.userId.nunique() < 2:
        raise ValueError('Insufficient effective internal ranking queries; no invented/leaked examples')
    if validation.empty or validation.label.sum() == 0:
        raise ValueError('Validation has no candidate positives for ranker early stopping')
    params = config['ranker']
    model = lgb.LGBMRanker(objective='lambdarank',metric='ndcg',eval_at=[20],
                           random_state=config['seed'],n_jobs=config['cpu_threads'],
                           deterministic=True,force_col_wise=True,verbosity=-1,
                           n_estimators=params['trees'],num_leaves=params['num_leaves'],
                           learning_rate=params['lr'],max_bin=127)
    model.fit(train[FEATURES],train.label,
              group=train.groupby('userId',sort=False).size().to_numpy(),
              eval_set=[(validation[FEATURES],validation.label)],
              eval_group=[validation.groupby('userId',sort=False).size().to_numpy()],
              eval_at=[20],callbacks=[lgb.early_stopping(params['patience'])])
    return model.booster_


def rank(model, frame):
    result = frame.copy()
    result['ranking_score'] = model.predict(result[FEATURES])
    return result.sort_values(['ranking_score','movieId'],ascending=[False,True],kind='stable').movieId.tolist()


def fit_from_shards(train_paths, validation_paths, feature_names, config):
    """LambdaMART over streamed Parquet features; LightGBM keeps compressed bins, not raw tables.

    Uses the native Dataset/Sequence API because sklearn's fit interface materializes X.
    The objective, query grouping and weak-label convention are the same as LGBMRanker.
    """
    import lightgbm as lgb
    from .dataset import ParquetRows, labels_and_groups
    rows = ParquetRows(train_paths,feature_names,config.get('sequence_batch_rows',4096))
    validation_rows = ParquetRows(validation_paths,feature_names,config.get('sequence_batch_rows',4096))
    require_memory(cpu=(len(rows)+len(validation_rows))*(len(feature_names)*4+24)*4,
                   stage='LightGBM compressed bins, labels, gradients and histogram workspace')
    labels,groups = labels_and_groups(train_paths)
    validation_labels,validation_groups = labels_and_groups(validation_paths)
    if len(groups) < 2 or labels.sum() == 0:
        raise ValueError('Insufficient effective historical ranking queries; no in-sample substitute')
    if not len(validation_groups) or not validation_labels.sum():
        raise ValueError('Validation candidates contain no retrieved positives')
    class Sequence(lgb.Sequence):
        def __init__(self,data):
            self.data,self.batch_size = data,data.batch_size
        def __len__(self): return len(self.data)
        def __getitem__(self,idx): return self.data[idx]
    train = lgb.Dataset(Sequence(rows),label=labels,group=groups,feature_name=feature_names,free_raw_data=True,
                        params={'max_bin':127,'data_random_seed':config['seed']})
    valid = lgb.Dataset(Sequence(validation_rows),label=validation_labels,group=validation_groups,
                        feature_name=feature_names,reference=train,free_raw_data=True)
    params = dict(objective='lambdarank',metric='ndcg',eval_at=[20],learning_rate=config['lr'],
                  num_leaves=config['num_leaves'],seed=config['seed'],num_threads=config['cpu_threads'],
                  deterministic=True,force_col_wise=True,verbosity=-1,max_bin=127)
    history = {}
    model = lgb.train(params,train,num_boost_round=config['trees'],valid_sets=[valid],valid_names=['validation'],
                     callbacks=[lgb.early_stopping(config['patience']),lgb.record_evaluation(history)])
    return model,history
