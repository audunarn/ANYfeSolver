"""One-time v2 bounded sparse-application parity supervisor for F03."""

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

import numpy as np
import psutil
import scipy


PARENT=Path(__file__).resolve().parent
CASES=(tuple(('native',topology,n,n) for topology in ('Q4','S3-V2D')
             for n in (4,8,16))
       +tuple(('native',topology,8,n) for topology in ('Q4','S3-V2D')
              for n in (1,64))
       +tuple(('synthetic',n) for n in (64,256,1024,4096,16384)))
MAX_RSS=8*1024**3
MAX_CASE_SECONDS=900
MAX_TOTAL_SECONDS=7200
PREVIOUS=Path('C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/f03-banded-moment-v1')
PREVIOUS_HASHES={
    'manifest.json':'422c8d875686988d7df4a5d823c9872fea2f16de378a17483a435d984ec08380',
    'records.jsonl':'989a68ef008cca3af46fa1e793dcbb54d085f9fbf1850bb47eb572965d216886',
}
SOURCES=(Path(__file__).name,'probe_b3_ge_e4_pl_g0_f03_banded_moment_v2.py',
         'run_b3_ge_e4_pl_g0_f03_banded_moment_v1.py',
         'probe_b3_ge_e4_pl_g0_f03_banded_moment_v1.py',
         'run_b3_ge_e4_pl_g0_f03_shell_port_lift_v1.py',
         'probe_b3_ge_e4_pl_g0_f03_shell_port_lift_v1.py',
         'run_b3_ge_e4_pl_g0_f03_local_trace_norm_v1.py',
         'probe_b3_ge_e4_pl_g0_f03_local_trace_norm_v1.py',
         'probe_b3_ge_e4_pl_g0_f03_trace_dual_v1.py',
         'probe_b3_ge_e4_pl_g0_f03_broken_trace_v1.py',
         'probe_b3_ge_e4_pl_g0_f03_signed_norm_v1.py',
         'probe_b3_ge_e4_pl_g0_f03_kkt.py',
         'probe_b3_ge_e4_pl_g0_f03_k1_rank.py',
         'probe_b3_ge_e4_pl_g0_f03_local_dual_v1.py',
         '_b3_e4_f03_native_fixture.py','_b3_e4_f03_shell_trace.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child(index,case,root,started):
    command=[sys.executable,str(PARENT/'probe_b3_ge_e4_pl_g0_f03_banded_moment_v2.py')]
    if case[0]=='native':
        _,topology,n_beam,n_shell=case
        command+=['--topology',topology,'--n-beam',str(n_beam),'--n-shell',str(n_shell)]
        identity={'kind':'native','topology':topology,'n_beam':n_beam,'n_shell':n_shell}
    else:
        command+=['--synthetic-n',str(case[1])]
        identity={'kind':'synthetic','n_beam':case[1]}
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
    record={'case':identity,'exit_code':process.returncode,
            'sampled_peak_child_rss_bytes':peak,'wall_seconds':time.monotonic()-began}
    if failure or process.returncode!=0:
        record.update(status='EXPLORATORY_FAILURE',failure=failure or 'child exited nonzero')
    else:
        try:
            item=json.loads(stdout)
            if any(item[k]!=v for k,v in identity.items()):
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
        raise ValueError('freeze all inputs before execution')
    hashes={name:sha(PARENT/name) for name in SOURCES}
    predecessor={name:sha(PREVIOUS/name) for name in PREVIOUS_HASHES}
    if predecessor!=PREVIOUS_HASHES:
        raise ValueError('sealed previous evidence differs')
    prior=json.loads((PREVIOUS/'manifest.json').read_text(encoding='utf-8'))
    if prior['revision']!='b3a70d1870fe127cd7c87a6faebd698bc4095662':
        raise ValueError('sealed predecessor revision differs')
    if any(hashes.get(name)!=digest for name,digest in prior['source_sha256'].items()):
        raise ValueError('inherited source differs from sealed predecessor')
    if len(CASES)!=15 or len(set(CASES))!=15:
        raise ValueError('registered inventory differs')
    imported=subprocess.check_output(
        [sys.executable,'-c','import sys;sys.path.insert(0,"src");'
         'import anyfesolver;print(anyfesolver.__file__)'],cwd=PARENT.parent,text=True).strip()
    if not Path(imported).resolve().is_relative_to((PARENT.parent/'src').resolve()):
        raise ValueError('ANYfeSolver import outside frozen integration source')
    args.root.mkdir(parents=True,exist_ok=False)
    manifest={'purpose':'BANDED_MOMENT_V2_PARITY_NOT_F03_ACCEPTANCE',
              'revision':revision,'source_sha256':hashes,'cases':CASES,
              'predecessor_sha256':predecessor,'anyfesolver_import':imported,
              'memory_limit_bytes':MAX_RSS,'case_wall_seconds':MAX_CASE_SECONDS,
              'total_wall_seconds':MAX_TOTAL_SECONDS,'rss_sampling_seconds':.5,
              'blas_threads':1,'python':platform.python_version(),
              'numpy':np.__version__,'scipy':scipy.__version__,'psutil':psutil.__version__,
              'limitations':['bounded native dense oracle only',
                             'synthetic sizes prove no physical response',
                             'banded inverse has global action, not fixed-stencil locality',
                             'no signed-KKT or six-channel acceptance']}
    (args.root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',
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
                if record.get('failure') in ('sampled child RSS exceeded 8 GiB',
                                             'registered wall-time limit exceeded'):
                    break
    return 1 if failed or completed!=len(CASES) else 0


if __name__=='__main__':
    raise SystemExit(main())
