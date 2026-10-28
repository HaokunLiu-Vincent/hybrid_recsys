"""Real Linux Docker HTTP/parity/lifecycle checks. Never substitutes a host venv."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
import httpx
from verify_serving_http import verify
from recsys.mlflow_local import tracking_uri,stop as stop_mlflow,start as start_mlflow,responsive


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',default='movielens-recsys-api:phase6_v1');parser.add_argument('--name',default='movielens-phase6-v1')
    parser.add_argument('--port',type=int,default=8001);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--keep-running',action='store_true')
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError('Docker verification output exists; choose new output/name')
    out.mkdir(parents=True);commands=[]
    def run(arguments,check=True):
        started=time.perf_counter();result=subprocess.run(['docker',*arguments],capture_output=True,text=True)
        record=dict(argv=['docker',*arguments],returncode=result.returncode,seconds=time.perf_counter()-started,
                    stdout=result.stdout,stderr=result.stderr);commands.append(record)
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if check and result.returncode:raise RuntimeError(result.stdout+result.stderr)
        return result
    if run(['container','inspect',args.name],False).returncode==0:raise FileExistsError('Container name exists; no replacement performed')
    image=json.loads(run(['image','inspect',args.image]).stdout)[0]
    if image['Architecture']!='arm64' or image['Os']!='linux':raise ValueError('Expected native Linux ARM64 image')
    mount=f'type=bind,source={args.model.resolve()},target=/models/movielens_hybrid_v1,readonly'
    base=['--platform','linux/arm64','--label','io.recsys.phase=6','--cpus','2','--memory','3g','--pids-limit','256',
          '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
          '--tmpfs','/tmp:rw,noexec,nosuid,size=536870912','--mount',mount]
    url=f'http://127.0.0.1:{args.port}'
    command=['run','--detach','--name',args.name,*base,'--publish',f'127.0.0.1:{args.port}:8000',args.image]
    started=time.perf_counter();cid=run(command).stdout.strip();keep=False;registry_stopped=False
    def ready():
        started=time.perf_counter()
        with httpx.Client(timeout=2,trust_env=False) as client:
            for _ in range(180):
                state=json.loads(run(['inspect','--format','{{json .State}}',cid]).stdout)
                if not state['Running']:raise RuntimeError('Container exited: '+json.dumps(state)+'\n'+run(['logs',cid]).stdout)
                try:
                    if client.get(url+'/health/ready').status_code==200:return time.perf_counter()-started
                except httpx.HTTPError:pass
                time.sleep(.5)
        raise TimeoutError('Container did not become ready')
    try:
        ready();startup=time.perf_counter()-started
        first=verify(url,args.reference_dir,latency=True)
        runtime_script="import os,platform,importlib.metadata as m,json,torch; print(json.dumps(dict(uid=os.getuid(),python=platform.python_version(),machine=platform.machine(),cuda_available=torch.cuda.is_available(),torch_build=torch.__version__,packages={p:m.version(p) for p in ['numpy','pandas','scipy','pyarrow','psutil','PyYAML','pytz','torch','implicit','threadpoolctl','lightgbm','mlflow','fastapi','uvicorn','httpx']})))"
        runtime=json.loads(run(['exec',cid,'/opt/venv/bin/python','-B','-c',runtime_script]).stdout)
        assert runtime['uid']==10001 and not runtime['cuda_available']
        inventory=run(['exec',cid,'find','/app','-type','f']).stdout
        assert '/app/data/' not in inventory and '/app/artifacts/' not in inventory and '/app/recsys/' not in inventory
        stats=run(['stats','--no-stream','--format','{{json .}}',cid]).stdout
        # Required offline continuation and restart, controlling only the owned project server.
        registry_uri=tracking_uri();stop_mlflow();registry_stopped=True;assert not responsive(registry_uri)
        with httpx.Client(timeout=30,trust_env=False) as client:
            assert client.post(url+'/recommend',json={'user_ids':[1,1000610],'top_k':20}).status_code==200
        run(['restart','--time','30',cid]);restart_ready_seconds=ready()
        restarted=verify(url,args.reference_dir,latency=False);assert not responsive(registry_uri)
        restored=start_mlflow();registry_stopped=False;assert restored['tracking_uri']==registry_uri and restored['healthy']
        health=None
        for _ in range(40):
            state=json.loads(run(['inspect','--format','{{json .State}}',cid]).stdout);health=state['Health']['Status']
            if health=='healthy':break
            time.sleep(.5)
        assert health=='healthy'
        # Optional Registry topology is probed honestly; no host exposure workaround.
        probe="import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:5002/health',timeout=3).status)"
        connectivity=run(['exec',cid,'/opt/venv/bin/python','-B','-c',probe],False)
        badid=run(['run','--detach','--name',args.name+'-wrong-release',*base,'--env','RECSYS_EXPECTED_RELEASE_ID=wrong',args.image]).stdout.strip()
        badcode=int(run(['wait',badid]).stdout.strip());assert badcode!=0
        badlogs=run(['logs',badid]);(out/'wrong_release.log').write_text(badlogs.stdout+badlogs.stderr)
        result=dict(gate_c='passed',gate_d='passed',image_id=image['Id'],image_size_bytes=image['Size'],platform='linux/arm64',
            container_id=cid,url=url,startup_wall_seconds=startup,first_http=first,restart_http=restarted,
            offline_mlflow_unreachable_verified=True,restart_ready_seconds=restart_ready_seconds,runtime=runtime,
            health=health,stats=stats,app_file_inventory=inventory,readonly_model_mount=True,
            optional_registry_connectivity=dict(returncode=connectivity.returncode,stdout=connectivity.stdout,stderr=connectivity.stderr),
            wrong_release_exit_code=badcode,registered_model_mutated=False)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        run(['logs',cid]);run(['inspect',cid]);keep=args.keep_running
        print(json.dumps({k:result[k] for k in ['gate_c','gate_d','image_id','image_size_bytes','container_id','url','health']},indent=2))
    finally:
        if registry_stopped:start_mlflow()
        if not keep:run(['stop','--time','30',cid],False)


if __name__=='__main__':main()
