"""Broken trace/yaw-jump graph diagnostic; not uniform F03 qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if not os.environ.get('NUMBA_CACHE_DIR'):
    raise RuntimeError('set a writable NUMBA_CACHE_DIR before source import')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import anyfesolver
import numba
import numpy as np
import scipy
from scipy import linalg
from threadpoolctl import threadpool_limits

from _b3_e4_f03_native_fixture import build
from _b3_e4_f03_shell_trace import make_shell
from probe_b3_ge_e4_pl_g0_f03_k1_rank import candidate_spaces
from probe_b3_ge_e4_pl_g0_f03_kkt import ELL, assemble, scales, supported_stiffness
from probe_b3_ge_e4_pl_g0_f03_signed_norm_v1 import F0,L0,U,W0


CASES=tuple((topology,n,n) for topology in ('Q4','S3-V2D')
            for n in (4,8,16)) + tuple((topology,8,n)
            for topology in ('Q4','S3-V2D') for n in (1,64))


def trace(beam,shell,s):
    bt=beam.planar_trace_jacobian(float(s))
    st=shell.trace(float(s))
    return np.diag((1.,1.,ELL))@np.hstack(
        (-st[:,shell.free],bt[:,beam.physical],bt[:,beam.internal]))


def interval(beam,shell,a,b):
    h=b-a
    s1,s2=a+.25*h,a+.75*h
    t1,t2=trace(beam,shell,s1),trace(beam,shell,s2)
    d=(t2-t1)/(.5*h)
    midpoint_error=float(np.linalg.norm(trace(beam,shell,(a+b)/2)-.5*(t1+t2))/
                         max(np.linalg.norm(t1),np.linalg.norm(t2),1.))
    return t1,t2,d,midpoint_error


def graph(beam,shell,overlay):
    n=shell.free_count+len(beam.physical)+len(beam.internal)
    g=np.zeros((n,n))
    maximum_midpoint=0.
    roots,weights=np.polynomial.legendre.leggauss(4)
    intervals=[]
    for a,b in zip(overlay[:-1],overlay[1:]):
        t1,t2,d,error=interval(beam,shell,float(a),float(b))
        maximum_midpoint=max(maximum_midpoint,error)
        intervals.append((a,b,t1,t2,d))
        for root,weight in zip(roots,weights):
            s=(a+b)/2+(b-a)*root/2
            t=t1+d*(s-(a+.25*(b-a)))
            w=(b-a)*weight/2
            g += (w/L0)*t.T@t + (L0*w)*d.T@d
    jumps=[]
    for j in range(1,beam.n_half_cells):
        station=j/beam.n_half_cells
        index=int(np.argmin(np.abs(overlay-station)))
        if abs(overlay[index]-station)>1e-12 or index in (0,len(overlay)-1):
            raise ValueError('native yaw jump is not resolved by overlay')
        a,b,t1,t2,d=intervals[index-1]
        left=t2+d*(station-(a+.75*(b-a)))
        a,b,t1,t2,d=intervals[index]
        right=t1+d*(station-(a+.25*(b-a)))
        jump=right-left
        if np.linalg.norm(jump[:2])>1e-10:
            raise ValueError('translation trace unexpectedly jumps')
        g += (L0*beam.n_half_cells)*jump.T@jump
        jumps.append(float(np.linalg.norm(jump)))
    if maximum_midpoint>1e-10:
        raise ValueError('trace is not affine on registered overlay')
    return g,maximum_midpoint,jumps


def operator_norm(beam,shell):
    k=supported_stiffness(beam,shell).toarray()
    sq,_=scales(beam,shell,1)
    a=(sq[:,None]*k)*sq[None,:]/W0
    p=shell.free_count+len(beam.physical)
    app,apr,arp,arr=a[:p,:p],a[:p,p:],a[p:,:p],a[p:,p:]
    r=linalg.solve(arr,arp,assume_a='sym')
    ke=app-apr@r
    ev,vec=linalg.eigh(arr)
    if min(linalg.eigvalsh(ke))<=1e-12*max(linalg.eigvalsh(ke)):
        raise ValueError('physical Schur not SPD in scaled graph diagnostic')
    if np.min(np.abs(ev))<=1e-12*np.max(np.abs(ev)):
        raise ValueError('native retained block has a near-zero mode')
    absolute=(vec*np.abs(ev)[None,:])@vec.T
    h=np.block([[ke+r.T@absolute@r,r.T@absolute],
                [absolute@r,absolute]])
    if np.linalg.norm(h-h.T)/np.linalg.norm(h)>1e-10:
        raise ValueError('operator-adapted H is nonsymmetric')
    return h,sq


def measure(topology,n_beam,n_shell):
    beam,shell=build(n_beam),make_shell(n_shell,topology)
    spaces={s.name:s for s in candidate_spaces(beam,shell)}
    overlay=assemble(beam,shell,spaces['p1-beam']).integration_knots
    h,sq=operator_norm(beam,shell)
    if len(h)>1200:
        raise ValueError('bounded dense graph diagnostic exceeds 1200 primal rows')
    g,mid,jumps=graph(beam,shell,overlay)
    scaled=(sq[:,None]*g)*sq[None,:]/U**2
    spectrum=linalg.eigh(scaled,h,eigvals_only=True,check_finite=True)
    largest=float(spectrum[-1])
    if spectrum[0]<-1e-10*largest or largest<=0:
        raise ValueError('invalid broken trace generalized spectrum')
    positive=spectrum[spectrum>1e-10*largest]
    return {'topology':topology,'n_beam':n_beam,'n_shell':n_shell,
            'overlay_intervals':len(overlay)-1,'primal_count':len(h),
            'max_affine_midpoint_error':mid,
            'native_yaw_jump_count':len(jumps),
            'max_raw_jump_row_frobenius_mixed_units':max(jumps,default=0.),
            'graph_rank_at_relative_1e_minus_10':len(positive),
            'graph_smallest_resolved_eigen':float(positive[0]),
            'graph_largest_eigen':largest,
            'trace_continuity_upper_in_operator_norm':float(np.sqrt(largest)),
            'status':'FINITE_BROKEN_GRAPH_SCREEN_NOT_F03_ACCEPTANCE'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=False)
    names=(Path(__file__).name,'_b3_e4_f03_native_fixture.py',
           '_b3_e4_f03_shell_trace.py','probe_b3_ge_e4_pl_g0_f03_kkt.py',
           'probe_b3_ge_e4_pl_g0_f03_k1_rank.py',
           'probe_b3_ge_e4_pl_g0_f03_signed_norm_v1.py')
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in names}
    manifest={'purpose':'EXPLORATORY_BROKEN_TRACE_NOT_F03_ACCEPTANCE',
              'cases':CASES,'source_sha256':hashes,
              'import':str(Path(anyfesolver.__file__).resolve()),
              'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
              'numba':numba.__version__,'cache':os.environ['NUMBA_CACHE_DIR'],
              'blas_threads':1,'max_dense_primal_rows':1200,
              'scales':{'force_density_n_per_m':F0,'attachment_length_m':L0,
                        'displacement_m':U,'work_j':W0,'yaw_length_m':ELL}}
    (args.root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    failed=False
    with threadpool_limits(limits=1), (args.root/'results.jsonl').open('x',encoding='utf-8') as out:
        for case in CASES:
            try:
                row=measure(*case)
            except Exception as exc:
                row={'case':case,'failure_type':type(exc).__name__,
                     'failure':str(exc),'status':'EXPLORATORY_FAILURE'}
                failed=True
            out.write(json.dumps(row,sort_keys=True)+'\n')
            out.flush()
            print(json.dumps({'case':case,'status':row['status'],
                              'upper':row.get('trace_continuity_upper_in_operator_norm'),
                              'failure':row.get('failure')},sort_keys=True),flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
