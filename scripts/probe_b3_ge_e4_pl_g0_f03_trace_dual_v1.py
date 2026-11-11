"""Restricted broken-trace dual/signed KKT oracle; not F03 qualification."""

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
from probe_b3_ge_e4_pl_g0_f03_broken_trace_v1 import graph
from probe_b3_ge_e4_pl_g0_f03_k1_rank import candidate_spaces
from probe_b3_ge_e4_pl_g0_f03_kkt import assemble,scales,supported_stiffness
from probe_b3_ge_e4_pl_g0_f03_signed_norm_v1 import (
    CASES,F0,L0,U,W0,couplings,reciprocal_sqrt,
)


CUTOFFS=(1e-12,1e-10,1e-8)


def primal_factors(beam,shell):
    k=supported_stiffness(beam,shell).toarray()
    if len(k)+3*beam.n_half_cells>1200:
        raise ValueError('bounded dense KKT exceeds 1200 rows')
    sq,_=scales(beam,shell,3*beam.n_half_cells)
    a=(sq[:,None]*k)*sq[None,:]/W0
    skew=float(np.linalg.norm(a-a.T)/np.linalg.norm(a))
    if skew>1e-10:
        raise ValueError('supported signed tangent is nonsymmetric')
    p=shell.free_count+len(beam.physical)
    app,apr,arp,arr=a[:p,:p],a[:p,p:],a[p:,:p],a[p:,p:]
    r=linalg.solve(arr,arp,assume_a='sym')
    ke=app-apr@r
    ep,up=linalg.eigh(ke)
    er,ur=linalg.eigh(arr)
    if np.min(np.abs(er))<=1e-12*np.max(np.abs(er)):
        raise ValueError('signed native Krr near zero')
    pp=reciprocal_sqrt(ep,up,'physical Ke')
    rr=ur/np.sqrt(np.abs(er))[None,:]
    return sq,p,r,pp,rr,er,skew


def dual_gram(c_scaled,eg,vg,cutoff):
    maximum=float(eg[-1])
    if maximum<=0 or eg[0]<-1e-10*maximum:
        raise ValueError('broken graph is not semidefinite')
    supported=eg>cutoff*maximum
    dropped=vg[:,~supported]
    kernel_residual=float(np.linalg.norm(c_scaled@dropped)/
                          max(np.linalg.norm(c_scaled),np.finfo(float).tiny))
    lifted=(c_scaled@vg[:,supported])/np.sqrt(eg[supported])[None,:]
    gram=lifted@lifted.T
    return gram,int(np.count_nonzero(supported)),kernel_residual


def normalized_metrics(gram,b,signs):
    ed,ud=linalg.eigh(gram)
    if ed[0]<=1e-10*ed[-1]:
        return None
    w=reciprocal_sqrt(ed,ud,'broken trace dual gram')
    bw=w.T@b
    infsup=linalg.svdvals(bw)
    schur=(bw*signs[None,:])@bw.T
    schur_eigen=linalg.eigvalsh(schur)
    saddle=np.zeros((len(signs)+len(gram),)*2)
    saddle[np.diag_indices(len(signs))]=signs
    saddle[:len(signs),len(signs):]=bw.T
    saddle[len(signs):,:len(signs)]=bw
    singular=linalg.svdvals(saddle)
    return {'constraint_infsup_floor':float(infsup[-1]),
            'signed_schur_min_abs_eigen':float(np.min(np.abs(schur_eigen))),
            'signed_schur_negative_modes':int(np.count_nonzero(schur_eigen<0)),
            'signed_kkt_min_singular':float(singular[-1]),
            'signed_kkt_max_singular':float(singular[0])}


def space_result(c,eg,vg,primal):
    sq,p,r,pp,rr,er,skew=primal
    cs=c*sq[None,:]/(U*L0)
    cp,cr=cs[:,:p],cs[:,p:]
    b=np.hstack(((cp-cr@r)@pp,cr@rr))
    signs=np.r_[np.ones(p),np.sign(er)]
    variants={}
    for cutoff in CUTOFFS:
        gram,rank,residual=dual_gram(cs,eg,vg,cutoff)
        eig=linalg.eigvalsh(gram)
        record={'graph_rank':rank,
                'kernel_annihilation_residual':residual,
                'dual_gram_min_eigen':float(eig[0]),
                'dual_gram_max_eigen':float(eig[-1])}
        if residual<=1e-8:
            metrics=normalized_metrics(gram,b,signs)
            if metrics is not None:
                record.update(metrics)
        record['admissible']=('signed_kkt_min_singular' in record)
        variants[str(cutoff)]=record
    primary=variants[str(CUTOFFS[0])]
    if primary['kernel_annihilation_residual']>1e-8:
        raise ValueError('constraint does not annihilate dropped trace modes')
    if not primary['admissible']:
        raise ValueError('multiplier space has deficient broken-trace dual norm')
    return {'rows':len(c),'graph_rank':primary['graph_rank'],
            'dual_cutoff_sensitivity':variants,
            **{key:primary[key] for key in (
                'constraint_infsup_floor','signed_schur_min_abs_eigen',
                'signed_schur_negative_modes','signed_kkt_min_singular',
                'signed_kkt_max_singular')},
            'primal_skew':skew,
            'status':'RESTRICTED_DISCRETE_TRACE_DUAL_ONLY'}


def measure(topology,n_beam,n_shell):
    beam,shell=build(n_beam),make_shell(n_shell,topology)
    spaces={s.name:s for s in candidate_spaces(beam,shell)}
    overlay=assemble(beam,shell,spaces['p1-beam']).integration_knots
    primal=primal_factors(beam,shell)
    sq=primal[0]
    physical,mid,jumps=graph(beam,shell,overlay)
    gs=(sq[:,None]*physical)*sq[None,:]/U**2
    eg,vg=linalg.eigh(gs,check_finite=True)
    result={'topology':topology,'n_beam':n_beam,'n_shell':n_shell,
            'primal_count':len(gs),'overlay_intervals':len(overlay)-1,
            'max_affine_midpoint_error':mid,'jump_count':len(jumps),'spaces':{}}
    for name,(c,m) in couplings(beam,shell).items():
        result['spaces'][name]=space_result(c,eg,vg,primal)
        result['spaces'][name]['l2_mass_min_eigen']=float(linalg.eigvalsh(m)[0])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=False)
    names=(Path(__file__).name,'_b3_e4_f03_native_fixture.py',
           '_b3_e4_f03_shell_trace.py','probe_b3_ge_e4_pl_g0_f03_kkt.py',
           'probe_b3_ge_e4_pl_g0_f03_k1_rank.py',
           'probe_b3_ge_e4_pl_g0_f03_local_dual_v1.py',
           'probe_b3_ge_e4_pl_g0_f03_signed_norm_v1.py',
           'probe_b3_ge_e4_pl_g0_f03_broken_trace_v1.py')
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in names}
    manifest={'purpose':'RESTRICTED_TRACE_DUAL_EXPLORATION_NOT_F03_ACCEPTANCE',
              'cases':CASES,'spaces':('k4','local-dual'),'cutoffs':CUTOFFS,
              'source_sha256':hashes,'import':str(Path(anyfesolver.__file__).resolve()),
              'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
              'numba':numba.__version__,'cache':os.environ['NUMBA_CACHE_DIR'],
              'blas_threads':1,'max_dense_kkt_rows':1200,
              'scales':{'force_density_n_per_m':F0,'attachment_length_m':L0,
                        'displacement_m':U,'work_j':W0}}
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
            print(json.dumps({'case':case,'failure':row.get('failure'),
                              'summary':{name:{'beta':v['constraint_infsup_floor'],
                                               'signed_sigma':v['signed_kkt_min_singular']}
                                         for name,v in row.get('spaces',{}).items()}},
                             sort_keys=True),flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
