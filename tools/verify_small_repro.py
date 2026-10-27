"""Real-data rerun plus cold-cache local DVC recovery in disposable copies, never original deletion."""
from pathlib import Path
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from recsys.small_pipeline import ROOT,settings,prepare,validate_splits,verify_files,json_bytes
from recsys.small_data import sha256,verify_provenance


def main():
    config=settings(ROOT/'params.yaml')
    raw=ROOT/config['dataset']['raw_dir'];verify_provenance(raw)
    directories=[raw,ROOT/'data/splits/small_v1',ROOT/'data/splits/small_global_v1']
    files=[p for d in directories for p in d.rglob('*') if p.is_file()]
    files += [ROOT/'data/diagnostics/small_protocol_v1.json',
              ROOT/'data/validation/small_raw_v1.json',ROOT/'data/validation/small_splits_v1.json']
    expected={str(p.relative_to(ROOT)):sha256(p) for p in files}
    mtimes={str(p.relative_to(ROOT)):p.stat().st_mtime_ns for p in files}
    dvc=str(ROOT/'.venv/bin/dvc');commands=[]
    env=os.environ.copy();env.update(DVC_NO_ANALYTICS='1',PYTHONDONTWRITEBYTECODE='1')
    def run(args,cwd=ROOT):
        local_env={**env,'DVC_SITE_CACHE_DIR':str(cwd/'.dvc/tmp/site-cache')}
        result=subprocess.run([dvc,*args],cwd=cwd,env=local_env,capture_output=True,text=True)
        commands.append(dict(cwd=str(cwd),argv=[dvc,*args],returncode=result.returncode,
                             stdout=result.stdout,stderr=result.stderr))
        if result.returncode:raise RuntimeError(f'DVC verification failed: {args}: {result.stderr}')
        return result.stdout
    remote_text=run(['config','remote.local.url']).strip()
    if '://' in remote_text:raise ValueError('Only a local filesystem remote is allowed')
    remote=Path(remote_text).expanduser().resolve()
    if not remote.is_dir() or remote==ROOT or remote.is_relative_to(ROOT):
        raise ValueError('DVC remote must exist outside the source tree')
    before_status=json.loads(run(['status','--json']))
    if before_status:raise RuntimeError(f'Pipeline not clean before rerun: {before_status}')
    no_change_output=run(['repro'])
    assert all(sha256(ROOT/p)==s for p,s in expected.items())
    assert all((ROOT/p).stat().st_mtime_ns==n for p,n in mtimes.items()),'No-change repro rewrote outputs'
    with tempfile.TemporaryDirectory(prefix='hybrid-recsys-repro-') as temporary:
        work=Path(temporary)
        results,comparison=prepare(raw,work/'independent_splits',config)
        for version in results:
            a,b=work/'independent_splits'/version,ROOT/'data/splits'/version
            assert {p.name:sha256(p) for p in a.iterdir()}=={p.name:sha256(p) for p in b.iterdir()}
            validate_splits(raw,a,config)
        assert json_bytes(comparison)==(ROOT/'data/diagnostics/small_protocol_v1.json').read_bytes()
        # A disposable DVC-only workspace, not a second Git repository or implementation.
        scratch=work/'recovery';scratch.mkdir()
        run(['init','--no-scm'],scratch)
        run(['config','core.analytics','false'],scratch)
        run(['config','cache.type','copy'],scratch)
        run(['remote','add','--default','--local','recovery',str(remote)],scratch)
        for relative in ['dvc.yaml','dvc.lock','params.yaml','data/raw/ml-latest-small.dvc']:
            dst=scratch/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dst)
        # Source dependencies make status/invalidation checks meaningful without executing scratch stages.
        stages=yaml.safe_load((ROOT/'dvc.yaml').read_text())['stages']
        for stage in stages.values():
            for relative in stage.get('deps',[]):
                source=ROOT/relative;dst=scratch/relative
                if source.is_file() and not relative.startswith('data/'):
                    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dst)
        assert not (scratch/'.git').exists()
        assert all(not (scratch/p).exists() for p in expected)
        cache=scratch/'.dvc/cache'
        assert not cache.exists() or not any(p.is_file() for p in cache.rglob('*'))
        cold_pull=run(['pull'],scratch)
        assert all(sha256(scratch/p)==s for p,s in expected.items()),'Cold remote recovery differs'
        clean_scratch=json.loads(run(['status','--json'],scratch));assert not clean_scratch,clean_scratch
        victim=scratch/'data/splits/small_v1/train.parquet'
        victim_hash=sha256(victim)
        victim.unlink()  # only a disposable restored copy; original/cache/remote remain intact
        missing_pull=run(['pull'],scratch)
        assert sha256(victim)==victim_hash
        params_path=scratch/'params.yaml';params_original=params_path.read_bytes()
        changed=yaml.safe_load(params_original);changed['protocol']['inner_history_percent']=75
        params_path.write_text(yaml.safe_dump(changed))
        config_status=json.loads(run(['status','--json'],scratch))
        assert 'prepare_splits' in config_status and 'validate_splits' in config_status
        assert 'validate_raw' not in config_status,config_status
        params_path.write_bytes(params_original)
        csv=scratch/'data/raw/ml-latest-small/ratings.csv'
        csv.chmod(stat.S_IMODE(csv.stat().st_mode)|stat.S_IWUSR)
        with csv.open('ab') as stream:stream.write(b'\n')
        raw_status=json.loads(run(['status','--json'],scratch))
        assert {'validate_raw','prepare_splits','validate_splits'}<=set(raw_status)
        # No repro after deliberately invalid fixture inputs; original files were never edited.
        evidence=dict(remote=str(remote),cold_cache_pull=cold_pull,missing_disposable_output_pull=missing_pull,
                      restored_files=len(expected),all_restored_sha256_match=True,
                      config_invalidation=config_status,raw_invalidation=raw_status)
    assert all(sha256(ROOT/p)==s for p,s in expected.items())
    assert all((ROOT/p).stat().st_mtime_ns==n for p,n in mtimes.items())
    final_status=json.loads(run(['status','--json']));assert not final_status
    report=dict(passed=True,original_files_unchanged=True,no_change_repro_rewrote_outputs=False,
                independent_real_data_build_byte_identical=True,checked_sha256=expected,
                no_change_repro_stdout=no_change_output,local_dvc_recovery=evidence,
                final_dvc_status=final_status,commands=commands,
                scope='Data preparation/recovery only. No models trained, scored or selected.')
    path=ROOT/'outputs/phase2'/('reproducibility_'+uuid.uuid4().hex+'.json')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(json_bytes(report))
    print(json.dumps(dict(passed=True,files_checked=len(expected),independent_build='byte-identical',
                         cold_cache_recovery='all SHA256 match',report=str(path)),indent=2))


if __name__=='__main__':main()
