"""One-time bounded joint/crossed refinement of the independent comparator."""

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

import probe_b3_ge_e4_pl_g0_f03_resolved_continuum as old
import probe_b3_ge_e4_pl_g0_f03_resolved_sparse_v5 as new


MAX_RSS=24*1024**3
MAX_CASE_SECONDS=1800
MAX_TOTAL_SECONDS=7200
CASES=tuple((t,l,n,n,n) for n in (64,96,128)
            for t in ('Q4','S3-V2D') for l in ('smooth','sine')) + tuple(
            (t,'smooth',ns,ny,nb) for t in ('Q4','S3-V2D')
            for ns,ny,nb in ((128,64,128),(128,128,64)))
PARENT=Path(__file__).resolve().parent
V4=Path('C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/f03-resolved-refinement-v4')
PREDECESSOR_SHA={
    'manifest.json':'8e016729c7e3c863642f3938d375584c249b0368eab0f49d4b2876e1bf03c83b',
    'records.jsonl':'82d53a2487db070717113c3dd3a87ddc1ae9b21bfa326739784679b8ef449d83',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight():
    new.self_check()
    parity=[]
    for topology in ('Q4','S3-V2D'):
        for load in ('smooth','sine'):
            for n in (4,8):
                before=old.solve(topology,n,n,load,n)
                after=new.solve(topology,n,n,load,n)
                if (before['matrix_nnz']!=after['matrix_nnz'] or
                    before['multiplier_rows']!=after['multiplier_rows']):
                    raise AssertionError('small dense/sparse pattern disagreement')
                motion=max(abs(a-b) for x,y in zip(before['station_probes'],
                                                      after['station_probes'])
                           for key in ('shell','beam','gap')
                           for a,b in zip(x[key],y[key]))
                extrema=max(abs(a-b) for key in (
                    'sampled_max_gap_lower_bound_components',
                    'polynomial_gap_extrema_estimate_components')
                    for a,b in zip(before[key],after[key]))
                scalars=('shell_energy_j','beam_energy_j','external_work_j',
                         'support_moment_z_nm','shell_interface_work_j',
                         'beam_interface_work_j')
                relative=max(abs(before[k]-after[k])/max(1.,abs(before[k]))
                             for k in scalars)
                if motion>5e-11 or extrema>5e-11 or relative>1e-9:
                    raise AssertionError('small physical-response parity failed')
                parity.append(dict(topology=topology,load=load,n=n,
                                   max_motion_difference=motion,
                                   max_gap_extrema_difference=extrema,
                                   max_scalar_relative_difference=relative))
    return parity


def child_case(index,case,root,started):
    t,l,ns,ny,nb=case
    child_code=('import json,sys; '
                'from probe_b3_ge_e4_pl_g0_f03_resolved_sparse_v5 import solve; '
                'print(json.dumps(solve(sys.argv[1],int(sys.argv[2]),'
                'int(sys.argv[3]),sys.argv[4],int(sys.argv[5])),allow_nan=False))')
    command=[sys.executable,'-c',child_code,t,str(ns),str(nb),l,str(ny)]
    environment=os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    environment['PYTHONPATH']=str(PARENT)+os.pathsep+environment.get('PYTHONPATH','')
    began=time.monotonic()
    process=subprocess.Popen(command,cwd=PARENT.parent,env=environment,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                             text=True)
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
                failure='sampled child RSS exceeded 24 GiB'
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
            failure='owned child failed to terminate after limit'
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
    record={'case':dict(topology=t,load=l,n_shell=ns,n_y=ny,n_beam=nb),
            'exit_code':process.returncode,'sampled_peak_child_rss_bytes':peak,
            'wall_seconds':time.monotonic()-began}
    if failure or process.returncode!=0:
        record.update(status='EXPLORATORY_FAILURE',failure=failure or 'child exited nonzero')
    else:
        try:
            item=json.loads(stdout)
            if (item['topology'],item['load'],item['n_shell'],item['n_y'],
                item['n_beam']) != case:
                raise ValueError('child case identity mismatch')
            record.update(status='EXPLORATORY_COMPLETE',response=item)
        except (ValueError,KeyError,TypeError) as exc:
            record.update(status='EXPLORATORY_FAILURE',failure=repr(exc))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=False)
    source_names=(Path(__file__).name,
                  'probe_b3_ge_e4_pl_g0_f03_resolved_sparse_v5.py',
                  'probe_b3_ge_e4_pl_g0_f03_resolved_continuum.py',
                  'probe_b3_ge_e4_pl_g0_f03_independent_membrane.py')
    hashes={name:sha(PARENT/name) for name in source_names}
    previous={name:sha(V4/name) for name in PREDECESSOR_SHA}
    if previous != PREDECESSOR_SHA:
        raise ValueError('historical v4 anchor changed')
    v4_manifest=json.loads((V4/'manifest.json').read_text())
    if hashes['probe_b3_ge_e4_pl_g0_f03_resolved_continuum.py']!=v4_manifest['source_sha256']:
        raise ValueError('reference implementation differs from v4 anchor')
    manifest={'purpose':'EXPLORATORY_INDEPENDENT_COMPARATOR_REFINEMENT_NOT_F03_ACCEPTANCE',
              'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],
                                                       text=True).strip(),
              'source_sha256':hashes,'predecessor_v4_sha256':previous,
              'cases':CASES,'memory_limit_bytes':MAX_RSS,
              'case_wall_seconds':MAX_CASE_SECONDS,
              'total_wall_seconds':MAX_TOTAL_SECONDS,
              'rss_sampling_seconds':.5,'blas_threads':1,
              'python':platform.python_version(),'numpy':np.__version__,
              'scipy':scipy.__version__,'psutil':psutil.__version__,
              'limitations':['finite-station collocation is not a continuous bond',
                             'classical membrane Q4/S3 differ from E4-PL',
                             'conventional beam differs from native retained B3',
                             'homogeneous beam interpolation lacks load particular field',
                             'gap extrema are numerical estimates, not certified enclosures']}
    (args.root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',
                                          encoding='utf-8')
    parity=preflight()
    (args.root/'preflight.json').write_text(json.dumps(parity,sort_keys=True,indent=2)+'\n',
                                           encoding='utf-8')
    started=time.monotonic()
    failed=False
    with (args.root/'records.jsonl').open('x',encoding='utf-8') as stream:
        for index,case in enumerate(CASES):
            record=child_case(index,case,args.root,started)
            stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
            stream.flush()
            print(json.dumps({'case':record['case'],'status':record['status'],
                              'wall_seconds':record['wall_seconds'],
                              'sampled_rss_bytes':record['sampled_peak_child_rss_bytes'],
                              'failure':record.get('failure')},sort_keys=True),flush=True)
            if record['status']!='EXPLORATORY_COMPLETE':
                failed=True
                if record.get('failure') in ('sampled child RSS exceeded 24 GiB',
                                             'registered wall-time limit exceeded'):
                    break
    return 1 if failed or index+1!=len(CASES) else 0


if __name__=='__main__':
    raise SystemExit(main())
