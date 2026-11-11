"""Registered, single-use F03-K0 diagnostic inventory executor.

The planar screen cannot qualify the full six-channel eccentric line law.
Execution requires a separately frozen registration containing exact source,
environment, resource and output identities. An absent terminal record is an
aborted execution, not permission to restart a consumed registration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback

import anyfesolver
import numpy
import psutil
import scipy
import scipy.linalg
import threadpoolctl

from probe_b3_ge_e4_pl_g0_f03_kkt import complete_inventory


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT.parents[1] / '.qualification' / 'b3-ge-e4-pl'
SOURCES = (
    'scripts/supervise_b3_ge_e4_pl_g0_f03_k0.py',
    'scripts/run_b3_ge_e4_pl_g0_f03_k0.py',
    'scripts/probe_b3_ge_e4_pl_g0_f03_kkt.py',
    'scripts/_b3_e4_f03_native_fixture.py',
    'scripts/_b3_e4_f03_shell_trace.py',
    'src/anyfesolver/_ge_beam3_retained_generalized.py',
    'src/anyfesolver/_ge_beam3_retained_generalized_state.py',
    'src/anyfesolver/_ge_beam3_generalized_static_boundary.py',
    'src/anyfesolver/_ge_beam3_native_generalized_element.py',
    'src/anyfesolver/_ge_beam3_centered_reference.py',
    'src/anyfesolver/e4_pl_element.py',
    'src/anyfesolver/e4_pl_s3_element.py',
    'src/anyfesolver/e4_pl_s3_v2d_element.py',
    'src/anyfesolver/elements.py',
)
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actual_identity() -> dict:
    numpy.linalg.svd(numpy.eye(2))
    scipy.linalg.svd(numpy.eye(2))
    pools = threadpoolctl.threadpool_info()
    imported = Path(anyfesolver.__file__).resolve()
    if not imported.is_relative_to(ROOT / 'src'):
        raise RuntimeError('anyfesolver import is not from the candidate checkout')
    head = subprocess.check_output(('git','rev-parse','HEAD'),cwd=ROOT,
                                   text=True).strip()
    dirty = subprocess.check_output(
        ('git','status','--porcelain','--untracked-files=all'),cwd=ROOT,
        text=True).strip()
    if dirty:
        raise RuntimeError('F03-K0 execution requires a clean source checkout')
    source_hashes = {name:sha256(ROOT / name) for name in SOURCES}
    tracked = subprocess.check_output(
        ('git','ls-files','-z','--','src/anyfesolver'),cwd=ROOT).split(b'\0')
    source_tree = hashlib.sha256()
    for raw_name in filter(None,tracked):
        name = os.fsdecode(raw_name)
        source_tree.update(raw_name+b'\0'+bytes.fromhex(sha256(ROOT/name)))
    return {'candidate_commit':head,'source_sha256':source_hashes,
            'tracked_solver_source_tree_sha256':source_tree.hexdigest(),
            'imported_anyfesolver':str(imported),
            'source_anyfesolver_version':anyfesolver.__version__,
            'numba_cache_directory':os.environ.get('NUMBA_CACHE_DIR'),
            'pythonpath':os.environ.get('PYTHONPATH'),
            'python_executable':str(Path(sys.executable).resolve()),
            'python_version':platform.python_version(),
            'packages':{'anyfesolver':importlib.metadata.version('anyfesolver'),
                        'numpy':numpy.__version__,
                        'scipy':scipy.__version__,
                        'numba':importlib.metadata.version('numba'),
                        'llvmlite':importlib.metadata.version('llvmlite'),
                        'psutil':psutil.__version__,
                        'threadpoolctl':importlib.metadata.version('threadpoolctl')},
            'loaded_threadpools':pools,
            'platform':platform.platform(),
            'processor':platform.processor(),
            'logical_cpus':psutil.cpu_count(logical=True),
            'physical_cpus':psutil.cpu_count(logical=False),
            'total_memory_bytes':psutil.virtual_memory().total,
            'threads':{name:os.environ.get(name) for name in THREADS}}


def verify_registration(registration: dict, identity: dict, evidence: Path):
    if registration.get('gate') != 'G0_F03_K0_V1' or registration.get('status') != 'FROZEN':
        raise ValueError('exact frozen F03-K0 v1 registration required')
    if set(registration.get('identity',{})) != set(identity):
        raise ValueError('registration identity fields differ from runner')
    for key,value in identity.items():
        if registration['identity'][key] != value:
            raise ValueError(f'registration identity mismatch: {key}')
    limits = registration.get('limits',{})
    if (set(limits) != {'wall_seconds','sampled_rss_limit_bytes'}
            or type(limits['wall_seconds']) is not int
            or not 60 <= limits['wall_seconds'] <= 86400
            or type(limits['sampled_rss_limit_bytes']) is not int
            or not 256*1024**2 <= limits['sampled_rss_limit_bytes'] <=
                identity['total_memory_bytes']):
        raise ValueError('invalid frozen resource limits')
    if (str(evidence) != registration.get('evidence_directory')
            or evidence.exists()
            or evidence.parent != EVIDENCE_ROOT):
        raise ValueError('new exact evidence directory within registered root required')
    if any(identity['threads'][name] != '1' for name in THREADS[:4]):
        raise ValueError('F03-K0 requires four explicit one-thread settings')
    if any(value not in (None,'1') for value in identity['threads'].values()):
        raise ValueError('extra backend thread settings conflict')
    if (not identity['loaded_threadpools']
            or any(pool.get('num_threads') != 1
                   for pool in identity['loaded_threadpools'])):
        raise ValueError('actual loaded numerical threadpools are not single-threaded')


class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.maximum = 0
        self.thread = threading.Thread(target=self._run,name='f03-memory',daemon=True)

    def _run(self):
        while not self.stop.wait(.1):
            try:
                self.maximum = max(self.maximum,self.process.memory_info().rss)
            except psutil.Error:
                break

    def __enter__(self):
        self.maximum = self.process.memory_info().rss
        self.thread.start()
        return self

    def __exit__(self,*_):
        self.stop.set()
        self.thread.join(timeout=1.)
        self.maximum = max(self.maximum,self.process.memory_info().rss)


def execute(registration_path: Path, evidence: Path) -> int:
    registration_path = registration_path.resolve(strict=True)
    evidence = evidence.resolve()
    registration = json.loads(registration_path.read_text(encoding='utf-8'))
    identity = actual_identity()
    verify_registration(registration,identity,evidence)
    evidence.mkdir(exist_ok=False)
    raw_path = evidence / 'raw.jsonl'
    started = time.monotonic()
    registration_hash = sha256(registration_path)
    with raw_path.open('x',encoding='utf-8') as raw, MemoryMonitor() as memory:
        def emit(record):
            raw.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
            raw.flush()
            os.fsync(raw.fileno())
        emit({'type':'start','registration_sha256':registration_hash,
              'registration':registration,'identity':identity})
        completed = 0
        def on_mesh(row):
            nonlocal completed
            completed += 1
            elapsed = time.monotonic()-started
            emit({'type':'mesh','index':completed,'elapsed_seconds':elapsed,
                  'rss_bytes':memory.process.memory_info().rss,'data':row})
            if elapsed > registration['limits']['wall_seconds']:
                raise TimeoutError('registered cooperative wall limit exceeded')
            if memory.maximum > registration['limits']['sampled_rss_limit_bytes']:
                raise MemoryError('registered peak RSS limit exceeded')
        try:
            result = complete_inventory(on_mesh=on_mesh)
            if result['mesh_count'] != 18 or result['candidate_count'] != 90 \
                    or result['load_rows'] != 180:
                raise AssertionError('registered inventory incomplete')
            emit({'type':'trends','data':result['trends']})
            if (time.monotonic()-started > registration['limits']['wall_seconds']
                    or memory.maximum > registration['limits']['sampled_rss_limit_bytes']):
                raise RuntimeError('registered cooperative resource limit exceeded')
            if (actual_identity() != identity
                    or sha256(registration_path) != registration_hash):
                raise RuntimeError('source, environment or registration changed during run')
            emit({'type':'terminal','status':result['status'],
                  'completed_meshes':completed,
                  'status_counts':result['status_counts'],
                  'trend_dispositions':{
                      name:sum(t['disposition'] == name for t in result['trends'])
                      for name in sorted({t['disposition'] for t in result['trends']})},
                  'elapsed_seconds':time.monotonic()-started,
                  'sampled_peak_rss_bytes':memory.maximum})
            return 0
        except BaseException as exc:
            emit({'type':'terminal','status':'ABORTED_NOT_QUALIFIED',
                  'completed_meshes':completed,
                  'elapsed_seconds':time.monotonic()-started,
                  'sampled_peak_rss_bytes':memory.maximum,
                  'exception_type':type(exc).__name__,
                  'exception':str(exc),
                  'traceback':traceback.format_exc()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    args = parser.parse_args()
    return execute(args.registration,args.evidence)


if __name__ == '__main__':
    raise SystemExit(main())
