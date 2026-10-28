"""Read-only comparison from imported MLflow metrics, not transcribed benchmark tables."""
from pathlib import Path
import argparse
import json
import pandas as pd
from recsys.mlflow_tracking import client_for

parser=argparse.ArgumentParser();parser.add_argument('--backfill-summary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();source=json.loads(args.backfill_summary.read_text());source=source.get('first',source)
client=client_for();rows=[]
for key,rid in source['runs'].items():
    run=client.get_run(rid)
    if 'validation/policy_b/ndcg_at_20' not in run.data.metrics:continue
    row=dict(original_key=key,mlflow_run_id=rid,retrospective_import=run.data.tags['retrospective_import'])
    row.update(run.data.metrics);rows.append(row)
if args.output.exists():raise FileExistsError('Comparison output exists; choose another path')
pd.DataFrame(rows).to_csv(args.output,index=False)
print(pd.DataFrame(rows)[['original_key','validation/policy_b/recall_at_20','validation/policy_b/ndcg_at_20','validation/policy_b/hitrate_at_20']].to_string(index=False))
