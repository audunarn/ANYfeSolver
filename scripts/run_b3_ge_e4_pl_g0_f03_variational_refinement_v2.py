"""One-time independent beam/mortar/shell refinement of the classical pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

PARENT=Path(__file__).resolve().parent
sys.path.insert(0,str(PARENT.parent/'src'))

import anyfesolver
import numpy as np
import psutil
import scipy

import run_b3_ge_e4_pl_g0_f03_variational_line_v1 as previous


SOURCES=(Path(__file__).name,
         'run_b3_ge_e4_pl_g0_f03_variational_line_v1.py',
         'probe_b3_ge_e4_pl_g0_f03_variational_line_v1.py',
         'probe_b3_ge_e4_pl_g0_f03_resolved_continuum.py',
         'probe_b3_ge_e4_pl_g0_f03_resolved_sparse_v5.py',
         'probe_b3_ge_e4_pl_g0_f03_independent_membrane.py')
MAX_RSS=8*1024**3
MAX_CASE_SECONDS=900
MAX_TOTAL_SECONDS=7200
V1_ROOT=Path('C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/f03-variational-line-v1')
V1_HASHES={
    'manifest.json':'3895a90ec6e94dc92b9c3738a8fc880fca20f56c51dd182b58664446660e0483',
    'preflight.json':'8a6dc936d517a2fb060208d086d51a36c71d3efb93fc6989e2075491e8904d78',
    'records.jsonl':'e4925757567209706512f0aeec5ce234871eafdd1b669822b4b049bdf6511204',
}
BASE=((16,8,8,16),   # Nm=8/16/32 with identical 128 beam subcells.
      (16,8,16,2),   # Nm=16 with beam subcells 32/64/128/256.
      (16,8,16,4),
      (16,8,16,8),
      (16,8,16,16),
      (16,8,32,4),
      (32,16,16,8), # Joint shell length/width refinement.
      (64,32,16,8))
SHELL_CONTROLS=((16,16,16,8),(32,8,16,8),
                (32,32,16,8),(64,16,16,8))
CASES=(tuple((topology,ns,ny,nm,subcells,load,law)
             for topology in ('Q4','S3-V2D')
             for load in ('smooth','sine')
             for law in ('weak-p0','weak-p1')
             for ns,ny,nm,subcells in BASE)
       +tuple((topology,ns,ny,nm,subcells,load,law)
              for topology in ('Q4','S3-V2D')
              for load in ('smooth','sine')
              for law in ('weak-p0','weak-p1')
              for ns,ny,nm,subcells in SHELL_CONTROLS))


def child(index,case,root,started):
    topology,ns,ny,nm,subcells,load,law=case
    command=[sys.executable,str(PARENT/'probe_b3_ge_e4_pl_g0_f03_variational_line_v1.py'),
             '--topology',topology,'--n-shell',str(ns),'--n-y',str(ny),
             '--n-mortar',str(nm),'--subcells',str(subcells),
             '--load',load,'--law',law]
    environment=os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    began=time.monotonic()
    process=subprocess.Popen(command,cwd=PARENT.parent,env=environment,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    observed=psutil.Process(process.pid)
    peak=0
    failure=None
    try:
        while process.poll() is None:
            try:
                peak=max(peak,observed.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if peak>MAX_RSS:
                failure='sampled child RSS exceeded 8 GiB'
                process.terminate()
                break
            if time.monotonic()-began>MAX_CASE_SECONDS or time.monotonic()-started>MAX_TOTAL_SECONDS:
                failure='registered wall-time limit exceeded'
                process.terminate()
                break
            time.sleep(.5)
        try:
            stdout,stderr=process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout,stderr=process.communicate()
            failure='owned child did not terminate after limit'
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        raise
    (root/f'{index:02d}.stdout.json').write_text(stdout,encoding='utf-8')
    (root/f'{index:02d}.stderr.txt').write_text(stderr,encoding='utf-8')
    record={'case':dict(topology=topology,n_shell=ns,n_y=ny,n_mortar=nm,
                        subcells=subcells,load=load,law=law),
            'exit_code':process.returncode,'sampled_peak_child_rss_bytes':peak,
            'wall_seconds':time.monotonic()-began}
    if failure or process.returncode!=0:
        record.update(status='EXPLORATORY_FAILURE',failure=failure or 'child exited nonzero')
    else:
        try:
            item=json.loads(stdout)
            if tuple(item[key] for key in (
                'topology','n_shell','n_y','n_mortar','subcells','load','law'))!=case:
                raise ValueError('child case identity mismatch')
            record.update(status='EXPLORATORY_COMPLETE',response=item)
        except (ValueError,KeyError,TypeError) as exc:
            record.update(status='EXPLORATORY_FAILURE',failure=repr(exc))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    if not os.environ.get('NUMBA_CACHE_DIR'):
        parser.error('set a dedicated writable NUMBA_CACHE_DIR')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PARENT.parent,
                                     text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],
                                  cwd=PARENT.parent,text=True).strip()
    if dirty:
        raise ValueError('freeze all integration worktree inputs before execution')
    hashes={name:hashlib.sha256((PARENT/name).read_bytes()).hexdigest()
            for name in SOURCES}
    predecessor={name:hashlib.sha256((V1_ROOT/name).read_bytes()).hexdigest()
                 for name in V1_HASHES}
    if predecessor!=V1_HASHES:
        raise ValueError('immutable v1 comparator evidence differs')
    prior_manifest=json.loads((V1_ROOT/'manifest.json').read_text(encoding='utf-8'))
    if prior_manifest['revision']!='0003a91639c8622f4fc1790277a0dcded961c733':
        raise ValueError('historical v1 source revision differs')
    if any(hashes.get(name)!=digest
           for name,digest in prior_manifest['source_sha256'].items()):
        raise ValueError('current inherited source bytes differ from frozen v1')
    # The pilot preflight tests four/six point polynomial coupling parity,
    # section flexibility, energy, and the four-host rigid pose.
    preflight_rows=previous.preflight()
    if len(CASES)!=96 or len(set(CASES))!=96:
        raise ValueError('incorrect independent refinement inventory')
    args.root.mkdir(parents=True,exist_ok=False)
    manifest={'purpose':'INDEPENDENT_CLASSICAL_VARIATIONAL_REFINEMENT_NOT_F03_ACCEPTANCE',
              'revision':revision,'source_sha256':hashes,'cases':CASES,
              'predecessor_v1_sha256':predecessor,
              'anyfesolver_import':str(Path(anyfesolver.__file__).resolve()),
              'independent_membrane_import':str(Path(
                  previous.probe.reference.continuum_stiffness.__code__.co_filename).resolve()),
              'memory_limit_bytes':MAX_RSS,'case_wall_seconds':MAX_CASE_SECONDS,
              'total_wall_seconds':MAX_TOTAL_SECONDS,'rss_sampling_seconds':.5,
              'blas_threads':1,'python':platform.python_version(),
              'numpy':np.__version__,'scipy':scipy.__version__,
              'psutil':psutil.__version__,
              'limitations':['independent classical planar model is not native B3/E4-PL',
                             'fixed shell/beam/mortar axes are not a diagonal convergence proof',
                             'P0/P1 row rank does not establish signed inf-sup',
                             'sampled pointwise gaps are lower bounds',
                             'root bond is not a separately enforced point condition']}
    (args.root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',
                                          encoding='utf-8')
    (args.root/'preflight.json').write_text(json.dumps(preflight_rows,sort_keys=True,indent=2)+'\n',
                                           encoding='utf-8')
    started=time.monotonic()
    failed=False
    completed=0
    with (args.root/'records.jsonl').open('x',encoding='utf-8') as stream:
        for index,case in enumerate(CASES):
            record=child(index,case,args.root,started)
            stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
            stream.flush()
            completed+=1
            print(json.dumps({'case':record['case'],'status':record['status'],
                              'wall_seconds':record['wall_seconds'],
                              'sampled_rss_bytes':record['sampled_peak_child_rss_bytes'],
                              'failure':record.get('failure')},sort_keys=True),flush=True)
            if record['status']!='EXPLORATORY_COMPLETE':
                failed=True
                if record.get('failure') in (
                    'sampled child RSS exceeded 8 GiB',
                    'registered wall-time limit exceeded'):
                    break
    return 1 if failed or completed!=len(CASES) else 0


if __name__=='__main__':
    raise SystemExit(main())
