"""Single-worker launcher; set numerical limits before importing model libraries."""
import os

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ.setdefault('MLFLOW_ENABLE_USAGE_TRACKING','false')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')

if __name__=='__main__':
    import logging
    import uvicorn
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    uvicorn.run('recsys_serving.app:create_app',factory=True,host=os.getenv('RECSYS_HOST','127.0.0.1'),
        port=int(os.getenv('RECSYS_PORT','8000')),workers=1,reload=False,access_log=False,limit_concurrency=64,
        timeout_keep_alive=5,timeout_graceful_shutdown=30)
