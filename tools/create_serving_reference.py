"""Fresh Version-1 predictions for all Train users plus one unknown; no target reads."""
from pathlib import Path
import argparse
import json
import time
import pandas as pd
from recsys_serving.model_loader import load_model,sha256
from recsys_serving.settings import Settings,MODEL_URI

parser=argparse.ArgumentParser();parser.add_argument('--tracking-uri',required=True);parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
if args.output_dir.exists():raise FileExistsError('Reference directory exists; choose a new output path')
loaded=load_model(Settings(model_source='registry',tracking_uri=args.tracking_uri))
users=list(map(int,loaded.model.unwrap_python_model().recommender.ctx.users))
inputs=pd.DataFrame({'userId':users+[max(users)+1000000],'top_k':20},dtype='int64')
started=time.perf_counter();predictions=loaded.model.predict(inputs);seconds=time.perf_counter()-started
args.output_dir.mkdir(parents=True)
inputs.to_parquet(args.output_dir/'inputs.parquet',index=False);predictions.to_parquet(args.output_dir/'predictions.parquet',index=False)
record=dict(model_uri=MODEL_URI,release_id=loaded.info['release_id'],queries=len(inputs),rows=len(predictions),prediction_seconds=seconds,
    selection='All serialized Train users and an unknown ID; no Validation/Test labels or candidates read',
    files={p.name:sha256(p) for p in args.output_dir.glob('*.parquet')})
(args.output_dir/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
