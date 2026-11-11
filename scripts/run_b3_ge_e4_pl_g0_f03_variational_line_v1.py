"""One-time bounded independent classical variational line-law pilot."""

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

import numpy as np
import psutil
import scipy
import anyfesolver

import probe_b3_ge_e4_pl_g0_f03_variational_line_v1 as probe


SOURCES=(Path(__file__).name,
         'probe_b3_ge_e4_pl_g0_f03_variational_line_v1.py',
         'probe_b3_ge_e4_pl_g0_f03_resolved_continuum.py',
         'probe_b3_ge_e4_pl_g0_f03_resolved_sparse_v5.py',
         'probe_b3_ge_e4_pl_g0_f03_independent_membrane.py')
MAX_RSS=8*1024**3
MAX_CASE_SECONDS=900
MAX_TOTAL_SECONDS=3600
CASES=(tuple((topology,8,4,8,subcells,load,law)
             for topology in ('Q4','S3-V2D') for load in ('smooth','sine')
             for subcells in (1,4)
             for law in (('weak-p0','weak-p1') if subcells==1 else
                         ('weak-p0','weak-p1','station','station-no-root')))
       +tuple((topology,16,8,8,4,load,law)
              for topology in ('Q4','S3-V2D') for load in ('smooth','sine')
              for law in ('weak-p0','weak-p1')))


def preflight():
    probe.reference.self_check()
    rows=[]
    for topology in ('Q4','S3-V2D'):
        for law in ('weak-p0','weak-p1'):
            a=probe.coupling(topology,8,4,8,8,'smooth',law,order=4)
            b=probe.coupling(topology,8,4,8,8,'smooth',law,order=6)
            cs4,cb4=a[7:9]
            cs6,cb6=b[7:9]
            shell_rel=float(np.linalg.norm((cs4-cs6).data)/
                            max(np.linalg.norm(cs4.data),np.finfo(float).tiny))
            beam_rel=float(np.linalg.norm((cb4-cb6).data)/
                           max(np.linalg.norm(cb4.data),np.finfo(float).tiny))
            if max(shell_rel,beam_rel)>1e-11:
                raise ValueError('mortar quadrature depends on polynomial order')
            rows.append({'topology':topology,'law':law,
                         'shell_quadrature_relative':shell_rel,
                         'beam_quadrature_relative':beam_rel})
    return rows


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
    imported=str(Path(probe.reference.continuum_stiffness.__code__.co_filename).resolve())
    preflight_rows=preflight()
    args.root.mkdir(parents=True,exist_ok=False)
    manifest={'purpose':'EXPLORATORY_CLASSICAL_VARIATIONAL_LINE_NOT_F03_ACCEPTANCE',
              'revision':revision,'source_sha256':hashes,'cases':CASES,
              'independent_membrane_import':imported,
              'anyfesolver_import':str(Path(anyfesolver.__file__).resolve()),
              'memory_limit_bytes':MAX_RSS,'case_wall_seconds':MAX_CASE_SECONDS,
              'total_wall_seconds':MAX_TOTAL_SECONDS,'rss_sampling_seconds':.5,
              'blas_threads':1,'python':platform.python_version(),
              'numpy':np.__version__,'scipy':scipy.__version__,
              'psutil':psutil.__version__,
              'limitations':['classical Q4/CST, not E4-PL',
                             'conforming Timoshenko subcells are not native retained B3',
                             'P0/P1 weak tests and station ties differ at finite resolution',
                             'station includes root; station-no-root excludes it',
                             'rank is not inf-sup or signed KKT stability',
                             'pointwise gap samples are lower bounds']}
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
