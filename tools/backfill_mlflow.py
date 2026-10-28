"""Persist a machine-readable backfill summary despite framework console messages."""
from pathlib import Path
import argparse
import json
from recsys.mlflow_tracking import backfill

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError('Import summary exists; choose a new output file')
result=backfill()
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x') as stream:json.dump(result,stream,indent=2)
print(f"Verified {result['run_count']} historical records; summary: {args.output}")
