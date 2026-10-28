"""Export exactly registered Version 1, retaining the complete outer MLflow layout."""
from pathlib import Path
import argparse
import json
from recsys_serving.model_loader import export_model

parser=argparse.ArgumentParser();parser.add_argument('--tracking-uri',required=True);parser.add_argument('--destination',type=Path,required=True)
args=parser.parse_args();print(json.dumps(export_model(args.destination,args.tracking_uri),indent=2))
