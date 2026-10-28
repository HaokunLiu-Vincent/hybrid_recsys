"""Independent CPU train/evaluate/benchmark commands. No implicit fitting in evaluation."""
import argparse
import json
import os
from pathlib import Path


def main():
    # Set limits before importing numerical libraries; --no-sync preserves inherited training wheels.
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key]='1'
    os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'.venv/mplconfig'))
    from .contracts import CHANNELS
    from .retrieval_workflow import ROOT,train,evaluate,benchmark
    parser=argparse.ArgumentParser(prog='recsys')
    commands=parser.add_subparsers(dest='command',required=True)
    fitting=commands.add_parser('train')
    fitting.add_argument('--model',choices=CHANNELS,required=True)
    fitting.add_argument('--config',type=Path,required=True)
    fitting.add_argument('--data-dir',type=Path,default=ROOT/'data/splits/small_v1')
    fitting.add_argument('--artifact-root',type=Path,default=ROOT/'artifacts')
    fitting.add_argument('--run-id')
    scoring=commands.add_parser('evaluate')
    scoring.add_argument('--model',choices=CHANNELS,required=True)
    scoring.add_argument('--split',choices=['validation'],default='validation')
    scoring.add_argument('--checkpoint',type=Path,required=True)
    scoring.add_argument('--data-dir',type=Path,default=ROOT/'data/splits/small_v1')
    scoring.add_argument('--output-root',type=Path,default=ROOT/'artifacts/candidates')
    scoring.add_argument('--run-id')
    table=commands.add_parser('benchmark')
    table.add_argument('--split',choices=['validation'],default='validation')
    table.add_argument('--evaluations',type=Path,nargs='+',required=True)
    table.add_argument('--output-root',type=Path,default=ROOT/'outputs/phase3')
    table.add_argument('--run-id')
    fusion=commands.add_parser('fuse-validation')
    fusion.add_argument('--evaluations',type=Path,nargs='+',required=True)
    fusion.add_argument('--data-dir',type=Path,default=ROOT/'data/splits/small_v1')
    fusion.add_argument('--output-root',type=Path,default=ROOT/'artifacts/fusion')
    fusion.add_argument('--run-id')
    historical=commands.add_parser('historical-train')
    historical.add_argument('--model',choices=CHANNELS,required=True)
    historical.add_argument('--config',type=Path,required=True)
    historical.add_argument('--data-dir',type=Path,default=ROOT/'data/splits/small_v1')
    historical.add_argument('--output-root',type=Path,default=ROOT/'artifacts/historical')
    historical.add_argument('--run-id')
    hc=commands.add_parser('historical-candidates')
    hc.add_argument('--checkpoint',type=Path,required=True)
    hc.add_argument('--data-dir',type=Path,default=ROOT/'data/splits/small_v1')
    hc.add_argument('--output-root',type=Path,default=ROOT/'artifacts/historical_candidates')
    hc.add_argument('--run-id')
    hf=commands.add_parser('fuse-historical')
    hf.add_argument('--candidates',type=Path,nargs='+',required=True)
    hf.add_argument('--data-dir',type=Path,default=ROOT/'data/splits/small_v1')
    hf.add_argument('--output-root',type=Path,default=ROOT/'artifacts/fusion')
    hf.add_argument('--run-id')
    features=commands.add_parser('rank-features')
    features.add_argument('--historical-fusion',type=Path,required=True)
    features.add_argument('--validation-fusion',type=Path,required=True)
    features.add_argument('--config',type=Path,default=ROOT/'configs/lambdamart.yaml')
    features.add_argument('--data-dir',type=Path,default=ROOT/'data/splits/small_v1')
    features.add_argument('--output-root',type=Path,default=ROOT/'artifacts/ranking_features')
    features.add_argument('--training-output-root',type=Path,default=ROOT/'artifacts/ranking_training')
    features.add_argument('--run-id')
    ranker=commands.add_parser('rank-train')
    ranker.add_argument('--training-data',type=Path,required=True)
    ranker.add_argument('--config',type=Path,default=ROOT/'configs/lambdamart.yaml')
    ranker.add_argument('--data-dir',type=Path,default=ROOT/'data/splits/small_v1')
    ranker.add_argument('--output-root',type=Path,default=ROOT/'artifacts/lightgbm')
    ranker.add_argument('--run-id')
    reval=commands.add_parser('rank-evaluate')
    reval.add_argument('--checkpoint',type=Path,required=True)
    reval.add_argument('--training-data',type=Path,required=True)
    reval.add_argument('--output-root',type=Path,default=ROOT/'artifacts/ranking_evaluations')
    reval.add_argument('--run-id')
    commands.add_parser('mlflow-backfill')
    packaging=commands.add_parser('package-release')
    packaging.add_argument('--run-id')
    registration=commands.add_parser('register-release')
    registration.add_argument('--release',type=Path,required=True)
    registration.add_argument('--parity',type=Path,required=True)
    registration.add_argument('--backfill-summary',type=Path,required=True)
    registration.add_argument('--name',default='MovieLensHybridRecommender')
    verification=commands.add_parser('verify-release')
    verification.add_argument('--model-uri',default='models:/MovieLensHybridRecommender@candidate')
    verification.add_argument('--parity',type=Path,required=True)
    prediction=commands.add_parser('recommend')
    prediction.add_argument('--release',type=Path,required=True)
    prediction.add_argument('--user-ids',type=int,nargs='+',required=True)
    prediction.add_argument('--top-k',type=int,default=20)
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='mlflow-backfill':
        from .mlflow_tracking import backfill
        print(json.dumps(backfill(),indent=2));return
    if command=='package-release':
        from .releases import package_release
        print(json.dumps(package_release(**args),indent=2));return
    if command in ['register-release','verify-release']:
        from .mlflow_registry import register_release,verify_release
        print(json.dumps((register_release if command=='register-release' else verify_release)(**args),indent=2));return
    if command=='recommend':
        from .inference import load_hybrid_recommender
        print(load_hybrid_recommender(args['release']).recommend(args['user_ids'],args['top_k']).to_json(orient='records'));return
    from .fusion.small_rrf import fuse_validation
    from .historical_retrieval import train_historical,historical_candidates,fuse_historical
    from .rankers.small_lambdamart import prepare_ranking_features,train_lambdamart,evaluate_lambdamart
    result={'train':train,'evaluate':evaluate,'benchmark':benchmark,'fuse-validation':fuse_validation,
            'historical-train':train_historical,'historical-candidates':historical_candidates,'fuse-historical':fuse_historical,
            'rank-features':prepare_ranking_features,'rank-train':train_lambdamart,'rank-evaluate':evaluate_lambdamart}[command](**args)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
