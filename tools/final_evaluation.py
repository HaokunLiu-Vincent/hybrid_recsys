"""Explicit sequential gates for the already-selected Version 1; never fits models."""
from pathlib import Path
import argparse
import json
import os
import sys

for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))
from recsys_evaluation.workflow import preflight, seal, gate_c, log_final

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gate', choices=['preflight', 'seal', 'evaluate', 'log'])
    parser.add_argument('--output', type=Path, default=root/'outputs/phase7')
    parser.add_argument('--tracking-uri', help='Required for preflight/log; inspect owned server first')
    args = parser.parse_args()
    if args.gate in ('preflight', 'log') and not args.tracking_uri:
        parser.error('--tracking-uri is required for this gate')
    if args.gate == 'preflight':
        result = preflight(root, args.output, args.tracking_uri)
    elif args.gate == 'seal':
        result = seal(root, args.output)
    elif args.gate == 'evaluate':
        result = gate_c(root, args.output)
    else:
        result = log_final(root, args.output, args.tracking_uri)
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
