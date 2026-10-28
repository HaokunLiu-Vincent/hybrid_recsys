"""Owned localhost MLflow lifecycle; persistent SQLite and artifact serving."""
from pathlib import Path
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
import psutil

ROOT=Path(__file__).resolve().parents[2]
HOME=ROOT/'outputs/mlflow'


def state():
    path=HOME/'server.json'
    return json.loads(path.read_text()) if path.exists() else None


def owned(record):
    try:
        process=psutil.Process(record['pid']);command=process.cmdline()
        return process if record['backend_uri'] in command and 'mlflow' in ' '.join(command) else None
    except (psutil.NoSuchProcess,psutil.AccessDenied):return None


def responsive(uri):
    try:
        with urlopen(uri+'/health',timeout=2) as response:return response.status==200
    except OSError:return False


def status():
    record=state()
    return dict(record or {},running=bool(record and owned(record)),healthy=bool(record and responsive(record['tracking_uri'])))


def start():
    HOME.mkdir(parents=True,exist_ok=True)
    old=state()
    if old and owned(old):
        if not responsive(old['tracking_uri']):raise RuntimeError('Owned MLflow exists but is unhealthy; inspect server.log')
        return status()
    preferred=old['port'] if old else 5000
    for port in range(preferred,preferred+20):
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                sock.bind(('127.0.0.1',port))
            break
        except PermissionError:raise
        except OSError:
            if port==preferred+19:raise RuntimeError('No available local MLflow port; no process was killed')
    database=HOME/'mlflow.db';artifacts=HOME/'artifacts';artifacts.mkdir(exist_ok=True)
    backend='sqlite:///'+str(database)
    command=[sys.executable,'-m','mlflow','server','--host','127.0.0.1','--port',str(port),
        '--workers','1','--backend-store-uri',backend,'--registry-store-uri',backend,
        '--serve-artifacts','--artifacts-destination',str(artifacts)]
    env={**os.environ,'MLFLOW_ENABLE_USAGE_TRACKING':'false','OMP_NUM_THREADS':'1',
         'OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':str(ROOT/'.venv/mplconfig')}
    with (HOME/'server.log').open('a') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=HOME,env=env)
    record=dict(pid=process.pid,port=port,tracking_uri=f'http://127.0.0.1:{port}',backend_uri=backend,
                artifact_root=str(artifacts),command=command)
    (HOME/'server.json').write_text(json.dumps(record,indent=2)+'\n')
    for _ in range(100):
        if process.poll() is not None:raise RuntimeError('MLflow exited; inspect outputs/mlflow/server.log')
        if responsive(record['tracking_uri']):return status()
        time.sleep(.2)
    raise RuntimeError('MLflow health timeout; inspect server.log before stopping the owned server')


def stop():
    record=state()
    if not record:return dict(running=False)
    process=owned(record)
    if process:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=15)
        except psutil.TimeoutExpired:raise RuntimeError('Owned server has not finished stopping; inspect server.log')
    return status()


def tracking_uri():
    record=status()
    if not record['healthy'] or not record['running']:raise RuntimeError('Start the owned local server first: scripts/start_mlflow.sh start')
    return record['tracking_uri']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','stop','status'])
    print(json.dumps(globals()[parser.parse_args().action](),indent=2))
