"""Real HTTP tracking/artifact/registry check and optional owned-server restart."""
from pathlib import Path
import argparse
import json
import tempfile
from recsys.mlflow_local import tracking_uri,start,stop
from recsys.mlflow_tracking import client_for,experiment


def verify(restart=False):
    original_uri=tracking_uri()
    client=client_for();eid=experiment(client,'MovieLens-Small-Health')
    run=client.create_run(eid,tags={'mlflow.runName':'phase5_local_health','integration_action':'health_check'})
    rid=run.info.run_id;client.log_param(rid,'probe','persistent_local');client.log_metric(rid,'integration/health',1.)
    with tempfile.TemporaryDirectory() as temporary:
        p=Path(temporary)/'health.txt';p.write_text('MLflow local persistent artifact probe\n')
        client.log_artifact(rid,str(p))
        got=client.download_artifacts(rid,'health.txt',str(Path(temporary)/'download'))
        assert Path(got).read_bytes()==p.read_bytes()
    name='MovieLensLocalHealthProbe'
    if not any(m.name==name for m in client.search_registered_models(filter_string=f"name = '{name}'")):
        client.create_registered_model(name,tags={'purpose':'Local registry health probe, not a recommender'})
    version=client.create_model_version(name,f'runs:/{rid}/health.txt',run_id=rid,tags={'purpose':'health_probe'})
    client.set_terminated(rid)
    if restart:
        stop();start();client=client_for()
        assert tracking_uri()==original_uri,'Tracking URI must remain stable across restart'
    saved=client.get_run(rid)
    assert saved.data.params['probe']=='persistent_local' and saved.data.metrics['integration/health']==1.
    assert client.get_model_version(name,version.version).run_id==rid
    with tempfile.TemporaryDirectory() as temporary:
        path=client.download_artifacts(rid,'health.txt',temporary)
        assert Path(path).read_text()=='MLflow local persistent artifact probe\n'
    return dict(tracking_uri=tracking_uri(),run_id=rid,experiment_id=eid,registry_probe_version=version.version,
                http_tracking_artifacts_registry=True,restarted=restart,persistence_verified=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--restart',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=verify(args.restart)
    if args.output:
        with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
