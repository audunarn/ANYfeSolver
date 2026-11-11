"""Smallest signed KKT mode in the restricted broken-trace dual; diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if not os.environ.get('NUMBA_CACHE_DIR'):
    raise RuntimeError('set an explicit writable NUMBA_CACHE_DIR before import')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import anyfesolver
import numba
import numpy as np
import scipy
from scipy import linalg
from threadpoolctl import threadpool_limits

from _b3_e4_f03_native_fixture import build
from _b3_e4_f03_shell_trace import make_shell
from probe_b3_ge_e4_pl_g0_f03_broken_trace_v1 import graph,interval,trace
from probe_b3_ge_e4_pl_g0_f03_k1_rank import candidate_spaces
from probe_b3_ge_e4_pl_g0_f03_kkt import assemble,supported_stiffness
from probe_b3_ge_e4_pl_g0_f03_signed_norm_v1 import CASES,U,L0,W0,couplings,reciprocal_sqrt
from probe_b3_ge_e4_pl_g0_f03_trace_dual_v1 import dual_gram,primal_factors


PREDECESSOR=Path('C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/f03-trace-dual-v1')
PREDECESSOR_SHA={
    'manifest.json':'ea783b9f4df3d89c94d01564c4799fb9e5414f4eb29b2c351381764acd0e59ec',
    'results.jsonl':'c2a54018fe6a81e151066610dc0f1ba5a855dfc4c99a4e473c62c65487b784dc',
}


def jump_amplitudes(beam,shell,overlay,physical_mode):
    pieces=[interval(beam,shell,float(a),float(b)) for a,b in
            zip(overlay[:-1],overlay[1:])]
    result=[]
    for j in range(1,beam.n_half_cells):
        station=j/beam.n_half_cells
        index=int(np.argmin(abs(overlay-station)))
        if abs(overlay[index]-station)>1e-12 or index in (0,len(overlay)-1):
            raise ValueError('unresolved native jump station')
        a,b=overlay[index-1:index+1]
        t1,t2,d,_=pieces[index-1]
        left=t2+d*(station-(a+.75*(b-a)))
        a,b=overlay[index:index+2]
        t1,t2,d,_=pieces[index]
        right=t1+d*(station-(a+.25*(b-a)))
        result.append(float(abs((right[2]-left[2])@physical_mode)/U))
    return result


def signed_coupling(bw,signs,dual_vector):
    actions=bw.T@dual_vector
    positive=float(np.sum(actions[signs>0]**2))
    negative=float(np.sum(actions[signs<0]**2))
    total=positive+negative
    dual_norm=float(dual_vector@dual_vector)
    return {'positive_action_squared':positive,
            'negative_action_squared':negative,
            'dual_norm_squared':dual_norm,
            'signed_schur_quadratic':positive-negative,
            'signed_schur_rayleigh':((positive-negative)/dual_norm
                                     if dual_norm>np.finfo(float).tiny else None),
            'cancellation_ratio':abs(positive-negative)/max(total,np.finfo(float).tiny)}


def mode(topology,n_beam,n_shell,previous):
    beam,shell=build(n_beam),make_shell(n_shell,topology)
    choices={s.name:s for s in candidate_spaces(beam,shell)}
    overlay=assemble(beam,shell,choices['p1-beam']).integration_knots
    physical,mid,jumps=graph(beam,shell,overlay)
    primal=primal_factors(beam,shell)
    sq,p,r,pp,rr,er,skew=primal
    stiffness=supported_stiffness(beam,shell).toarray()
    ks=(sq[:,None]*stiffness)*sq[None,:]/W0
    gs=(sq[:,None]*physical)*sq[None,:]/U**2
    eg,vg=linalg.eigh(gs,check_finite=True)
    results={'topology':topology,'n_beam':n_beam,'n_shell':n_shell,
             'primal_count':len(sq),'overlay_intervals':len(overlay)-1,
             'max_affine_midpoint_error':mid,'jump_count':len(jumps),
             'spaces':{}}
    for name,(c,_) in couplings(beam,shell).items():
        if len(sq)+len(c)>1200:
            raise ValueError('bounded dense signed KKT exceeds 1200 rows')
        cs=c*sq[None,:]/(U*L0)
        gram,rank,kernel=dual_gram(cs,eg,vg,1e-12)
        if kernel>1e-8:
            raise ValueError('constraint does not annihilate graph kernel')
        ed,ud=linalg.eigh(gram)
        if ed[0]<=1e-10*ed[-1]:
            raise ValueError('deficient discrete-image dual')
        w=reciprocal_sqrt(ed,ud,'trace dual')
        cp,cr=cs[:,:p],cs[:,p:]
        b=np.hstack(((cp-cr@r)@pp,cr@rr))
        bw=w.T@b
        signs=np.r_[np.ones(p),np.sign(er)]
        positive_infsup=float(linalg.svdvals(bw)[-1])
        spd_comparator=float((np.sqrt(1+4*positive_infsup**2)-1)/2)
        schur=(bw*signs[None,:])@bw.T
        schur_values,schur_vectors=linalg.eigh(schur)
        schur_weakest=int(np.argmin(abs(schur_values)))
        schur_dual=schur_vectors[:,schur_weakest]
        saddle=np.zeros((len(sq)+len(c),)*2)
        saddle[np.diag_indices(len(sq))]=signs
        saddle[:len(sq),len(sq):]=bw.T
        saddle[len(sq):,:len(sq)]=bw
        values,vectors=linalg.eigh(saddle,check_finite=True)
        weakest=int(np.argmin(abs(values)))
        value=float(values[weakest])
        second=float(np.partition(abs(values),1)[1])
        expected=previous[(topology,n_beam,n_shell)]['spaces'][name][
            'signed_kkt_min_singular']
        if abs(abs(value)-expected)>max(1e-12,1e-9*expected):
            raise ValueError('signed weakest mode does not match prior singular floor')
        vector=vectors[:,weakest]
        xp=vector[:p]
        xr=vector[p:len(sq)]
        xl=vector[len(sq):]
        if abs(float(xp@xp+xr@xr+xl@xl)-1.)>1e-10:
            raise ValueError('weak eigenvector mode is not normalized')
        p_scaled=pp@xp
        r_scaled=rr@xr-r@p_scaled
        q_scaled=np.r_[p_scaled,r_scaled]
        q=sq*q_scaled
        # Eigenvectors have arbitrary sign. Fix it by the largest physical
        # coordinate, then report invariant amplitudes and signed samples.
        if q[np.argmax(abs(q[:p]))]<0:
            vector=-vector
            xp,xr,xl,p_scaled,r_scaled,q_scaled,q=(
                -xp,-xr,-xl,-p_scaled,-r_scaled,-q_scaled,-q)
        sampled=[(trace(beam,shell,s)@q/U).tolist()
                 for s in (0.,.125,.25,.375,.5,.625,.75,.875,1.)]
        jump=jump_amplitudes(beam,shell,overlay,q)
        lam=w@xl
        scaled_primal_residual=ks@q_scaled+cs.T@lam
        scaled_constraint_residual=cs@q_scaled
        pulled_back=np.r_[
            pp.T@(scaled_primal_residual[:p]-r.T@scaled_primal_residual[p:]),
            rr.T@scaled_primal_residual[p:],
            w.T@scaled_constraint_residual]
        pulled_back_error=float(np.linalg.norm(
            pulled_back-value*np.r_[xp,xr,xl]))
        if pulled_back_error>1e-9 or not np.isfinite(pulled_back_error):
            raise ValueError('inverse factorization does not reproduce eigenmode')
        alternation={label:float(np.mean((lam[start:start+n_beam-1]*
                    lam[start+1:start+n_beam])<0))
                     for label,start in (('x',0),('y',n_beam),('yaw',2*n_beam))}
        gap_norm=float(np.sqrt(max(0.,q_scaled@gs@q_scaled)))
        residual=float(np.linalg.norm(saddle@vector-value*vector))
        if residual>1e-9 or not np.isfinite(residual):
            raise ValueError('signed eigenmode residual exceeds tolerance')
        results['spaces'][name]={
            'rows':len(c),'graph_rank':rank,'kernel_annihilation_residual':kernel,
            'signed_weak_eigenvalue':value,'signed_kkt_min_singular':abs(value),
            'second_smallest_abs_eigenvalue':second,
            'weak_mode_nearly_repeated':bool(second<1.00001*abs(value)),
            'normalized_eigen_residual':residual,
            'pulled_back_scaled_eigen_residual':pulled_back_error,
            'p_norm_squared':float(xp@xp),
            'retained_norm_squared':float(xr@xr),
            'retained_positive_norm_squared':float(np.sum(xr[er>0]**2)),
            'retained_negative_norm_squared':float(np.sum(xr[er<0]**2)),
            'dual_norm_squared':float(xl@xl),
            'positive_constraint_infsup_floor':positive_infsup,
            'all_positive_saddle_comparator':spd_comparator,
            'weak_kkt_dual_signed_coupling':signed_coupling(bw,signs,xl),
            'weak_schur_eigenvalue':float(schur_values[schur_weakest]),
            'weak_schur_signed_coupling':signed_coupling(bw,signs,schur_dual),
            'shell_scaled_coordinate_norm':float(np.linalg.norm(q_scaled[:shell.free_count])),
            'beam_scaled_coordinate_norm':float(np.linalg.norm(q_scaled[shell.free_count:p])),
            'retained_scaled_coordinate_norm':float(np.linalg.norm(q_scaled[p:])),
            'trace_graph_norm':gap_norm,
            'scaled_constraint_gap_norm':float(np.linalg.norm(cs@q_scaled)),
            'max_native_yaw_jump_over_U':max(jump,default=0.),
            'native_yaw_jump_amplitudes_over_U':jump,
            'nine_one_sided_gap_samples_over_U':sampled,
            'multiplier_row_alternation':alternation,
            'multiplier_row_max_abs_in_F0_units':{
                label:float(np.max(abs(lam[start:start+n_beam])))
                for label,start in (('x',0),('y',n_beam),('yaw',2*n_beam))},
            'status':'FINITE_RESTRICTED_DUAL_WEAK_MODE_NOT_F03_ACCEPTANCE'}
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    for name,expected in PREDECESSOR_SHA.items():
        actual=hashlib.sha256((PREDECESSOR/name).read_bytes()).hexdigest()
        if actual!=expected:
            raise ValueError('trace-dual predecessor evidence changed')
    previous_rows=[json.loads(x) for x in (PREDECESSOR/'results.jsonl').read_text(
        encoding='utf-8').splitlines()]
    previous={(x['topology'],x['n_beam'],x['n_shell']):x for x in previous_rows}
    if len(previous)!=len(CASES) or set(previous)!=set(CASES):
        raise ValueError('predecessor case inventory mismatch')
    args.root.mkdir(parents=True,exist_ok=False)
    names=(Path(__file__).name,'_b3_e4_f03_native_fixture.py',
           '_b3_e4_f03_shell_trace.py','probe_b3_ge_e4_pl_g0_f03_kkt.py',
           'probe_b3_ge_e4_pl_g0_f03_k1_rank.py',
           'probe_b3_ge_e4_pl_g0_f03_local_dual_v1.py',
           'probe_b3_ge_e4_pl_g0_f03_signed_norm_v1.py',
           'probe_b3_ge_e4_pl_g0_f03_broken_trace_v1.py',
           'probe_b3_ge_e4_pl_g0_f03_trace_dual_v1.py')
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in names}
    manifest={'purpose':'EXPLORATORY_SIGNED_WEAK_MODE_NOT_F03_ACCEPTANCE',
              'cases':CASES,'spaces':('k4','local-dual'),
              'source_sha256':hashes,'anyfesolver_import':str(Path(anyfesolver.__file__).resolve()),
              'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
              'numba':numba.__version__,'cache':os.environ['NUMBA_CACHE_DIR'],
              'blas_threads':1,'max_dense_kkt_rows':1200,
              'graph_eigen_cutoff':1e-12,'yaw_conversion_length_m':.46,
              'predecessor_sha256':PREDECESSOR_SHA}
    (args.root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',
                                          encoding='utf-8')
    failed=False
    with threadpool_limits(limits=1),(args.root/'results.jsonl').open('x',encoding='utf-8') as out:
        for case in CASES:
            try:
                row=mode(*case,previous)
            except Exception as exc:
                row={'case':case,'failure_type':type(exc).__name__,
                     'failure':str(exc),'status':'EXPLORATORY_FAILURE'}
                failed=True
            out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            out.flush()
            print(json.dumps({'case':case,'failure':row.get('failure'),
                              'summary':{name:{'eigenvalue':v['signed_weak_eigenvalue'],
                                               'retained_fraction':v['retained_norm_squared'],
                                               'dual_fraction':v['dual_norm_squared']}
                                         for name,v in row.get('spaces',{}).items()}},
                             sort_keys=True),flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
