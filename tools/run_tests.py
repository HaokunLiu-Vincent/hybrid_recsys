"""Run synthetic tests in a fresh, CPU-thread-bounded process (no dataset training)."""
from pathlib import Path
import os
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    env[key]='1'
env['PYTHONPATH']=str(root/'src')+os.pathsep+env.get('PYTHONPATH','')
raise SystemExit(subprocess.call([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-v'],
                                 cwd=root,env=env))
