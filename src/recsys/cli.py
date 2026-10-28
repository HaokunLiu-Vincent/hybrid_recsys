"""Independent CPU train/evaluate/benchmark commands. No implicit fitting in evaluation."""
import argparse
import json
import os
from pathlib import Path


def main():
    # Set limits before importing numerical libraries; --no-sync preserves inherited training wheels.
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key]='1'
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
    args=vars(parser.parse_args());command=args.pop('command')
    result={'train':train,'evaluate':evaluate,'benchmark':benchmark}[command](**args)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
