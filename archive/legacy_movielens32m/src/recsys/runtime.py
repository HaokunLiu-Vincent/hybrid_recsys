"""Resource budgets, reproducible fingerprints, and immutable verified checkpoints."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
import threading
import time
import uuid

import psutil


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def environment(root='/content'):
    try:
        import torch
    except ImportError:
        torch = None
    versions = {}
    for name in ['numpy','scipy','pandas','pyarrow','psutil','torch','implicit','lightgbm','matplotlib','nbformat']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    gpu = None
    if torch is not None and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = dict(name=torch.cuda.get_device_name(), free_bytes=free, total_bytes=total)
    return dict(python=platform.python_version(), torch=torch.__version__ if torch else None, cuda=torch.version.cuda if torch else None,
                gpu=gpu, cpu_cores=os.cpu_count(), available_ram=psutil.virtual_memory().available,
                disk_free=shutil.disk_usage(Path(root)).free,
                packages=versions)


def require_memory(cpu=0, gpu=0, disk=0, root='/tmp', fraction=.65, stage='stage'):
    available = psutil.virtual_memory().available
    if cpu > available * fraction:
        raise MemoryError(f'{stage}: estimated additional RAM {cpu/2**30:.2f} GiB exceeds '
                          f'{fraction:.0%} of {available/2**30:.2f} GiB available. '
                          'Use a larger RAM runtime or reduce the documented block/batch size.')
    if disk > shutil.disk_usage(root).free * fraction:
        raise OSError(f'{stage}: insufficient disk; need {disk/2**30:.2f} GiB plus reserve in {root}.')
    if gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f'{stage}: CUDA required for full-scale neural training; select the L4 runtime.')
        free, _ = torch.cuda.mem_get_info()
        if gpu > free * fraction:
            raise MemoryError(f'{stage}: GPU estimate {gpu/2**30:.2f} GiB exceeds budget; '
                              'reduce embedding/batch dimensions or use a larger GPU, then version the run.')


@contextmanager
def profile():
    """Sample actual process RSS including native allocations; CUDA allocator peak separately."""
    try:
        import torch
    except ImportError:
        torch = None
    done = threading.Event()
    samples = [psutil.Process().memory_info().rss]
    def poll():
        while not done.wait(.1):
            samples.append(psutil.Process().memory_info().rss)
    thread = threading.Thread(target=poll, daemon=True)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    thread.start()
    report = {}
    start = time.perf_counter()
    try:
        yield report
    finally:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        done.set()
        thread.join()
        samples.append(psutil.Process().memory_info().rss)
        report.update(seconds=time.perf_counter()-start, peak_rss_bytes=max(samples),
                      cuda_peak_bytes=torch.cuda.max_memory_allocated() if torch is not None and torch.cuda.is_available() else 0)


class Store:
    """Immutable stage directories. Only a checksummed COMPLETE manifest marks success.

    Stages are built locally, copied to a hidden destination, verified, then renamed.
    An interrupted hidden directory is ignored on restart. Completed stages never overwrite.
    """
    def __init__(self, root, signature):
        self.root = Path(root) / signature
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, stage):
        path = self.root / stage
        if not path.exists():
            return None
        marker = path / 'COMPLETE.json'
        if not marker.is_file():
            raise RuntimeError(f'Incomplete visible checkpoint: {path}. Inspect/move it explicitly before retrying.')
        meta = json.loads(marker.read_text())
        for name, info in meta['files'].items():
            p = path / name
            if not p.is_file() or p.stat().st_size != info['bytes'] or digest(p) != info['sha256']:
                raise IOError(f'Corrupt checkpoint {p}; no automatic overwrite.')
        return path

    def write(self, stage, writer, metadata=None):
        old = self.load(stage)
        if old is not None:
            return old
        with tempfile.TemporaryDirectory(prefix='recsys-') as tmp:
            local = Path(tmp)
            writer(local)
            files = {str(p.relative_to(local)): dict(bytes=p.stat().st_size, sha256=digest(p))
                     for p in sorted(local.rglob('*')) if p.is_file()}
            if not files:
                raise ValueError('Refuse empty checkpoint')
            (local/'COMPLETE.json').write_text(json.dumps(dict(files=files, metadata=metadata or {}),
                                                        indent=2, default=str))
            staging = self.root / ('.'+stage.replace('/', '_')+'-'+uuid.uuid4().hex)
            require_memory(disk=sum(x['bytes'] for x in files.values()), root=self.root, stage='checkpoint')
            shutil.copytree(local, staging)
            for name, info in files.items():
                if digest(staging/name) != info['sha256']:
                    raise IOError(f'Drive copy verification failed: {staging/name}')
            target = self.root/stage
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f'Concurrent checkpoint at {target}; refusing replacement')
            staging.rename(target)
        return self.load(stage)

    def json(self, stage, value):
        return self.write(stage, lambda p: (p/'report.json').write_text(
            json.dumps(value, indent=2, default=str, allow_nan=False)))
