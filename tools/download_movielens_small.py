"""Download only the declared public Kaggle source; never overwrite an existing raw dataset."""
from pathlib import Path
import argparse
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from recsys.small_data import download_dataset

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data'/'raw'/'ml-latest-small')
    local_cli=Path(sys.executable).parent/'kaggle'
    parser.add_argument('--kaggle',default=str(local_cli) if local_cli.is_file() else shutil.which('kaggle'))
    args=parser.parse_args()
    if not args.kaggle:parser.error('Kaggle CLI missing; install requirements-eda.txt in the selected environment.')
    manifest=download_dataset(args.data_dir,args.kaggle)
    print('Verified source:',manifest['source']['dataset_id'])
    print('Observed dataset:',manifest['quality'])
    print('Provenance:',args.data_dir/'provenance.json')
