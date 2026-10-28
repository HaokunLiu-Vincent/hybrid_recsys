"""Standalone registry load: -I, no recsys installation, deny source-checkout reads."""
import argparse
import json
from pathlib import Path
import sys
import importlib.util

parser=argparse.ArgumentParser()
for name in ['tracking-uri','model-uri','inputs','reference','deny-root','output']:parser.add_argument('--'+name,required=True)
args=parser.parse_args()
assert importlib.util.find_spec('recsys') is None,'Independent check requires no installed recsys package'
denied=Path(args.deny_root).resolve()
def guard(event,values):
    if event=='open' and values and isinstance(values[0],(str,bytes)):
        path=Path(values[0]).resolve()
        if path==denied or path.is_relative_to(denied):raise RuntimeError('Independent inference tried reading the original checkout: '+str(path))
sys.addaudithook(guard)
import mlflow
import pandas as pd
mlflow.set_tracking_uri(args.tracking_uri);mlflow.set_registry_uri(args.tracking_uri)
model=mlflow.pyfunc.load_model(args.model_uri)
import recsys
assert not Path(recsys.__file__).resolve().is_relative_to(denied)
inputs=pd.read_parquet(args.inputs);reference=pd.read_parquet(args.reference)
actual=model.predict(inputs);pd.testing.assert_frame_equal(actual,reference,check_exact=True)
for bad in [pd.DataFrame({'userId':[0],'top_k':[20]}),pd.DataFrame({'userId':[1],'top_k':[0]})]:
    try:model.predict(bad)
    except (ValueError,mlflow.exceptions.MlflowException):pass
    else:raise AssertionError('Invalid inference input accepted')
result=dict(model_uri=args.model_uri,tracking_uri=args.tracking_uri,queries=len(inputs),rows=len(actual),exact_prediction_parity=True,
    imported_application=str(Path(recsys.__file__).resolve()),original_checkout_reads_denied=True,
    feature_count=len(model.unwrap_python_model().recommender.schema['names']))
Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
