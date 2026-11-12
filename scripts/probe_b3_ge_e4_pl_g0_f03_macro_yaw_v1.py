"""Two-cell yaw moment mortar candidate; bounded planar falsification only."""

from __future__ import annotations

from dataclasses import replace
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

if not os.environ.get('NUMBA_CACHE_DIR'):
    raise RuntimeError('set an explicit writable NUMBA_CACHE_DIR before import')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import anyfesolver
import numba
import numpy as np
import scipy
from scipy import linalg,sparse
from threadpoolctl import threadpool_limits

from _b3_e4_f03_native_fixture import build
from _b3_e4_f03_shell_trace import make_shell
from probe_b3_ge_e4_pl_g0_f03_broken_trace_v1 import graph
from probe_b3_ge_e4_pl_g0_f03_k1_rank import candidate_spaces
from probe_b3_ge_e4_pl_g0_f03_kkt import (
    ELL,CandidateSingular,PrimalNotAdmissible,assemble,physical_support_balance,
    scales,solve_exact,
)
from probe_b3_ge_e4_pl_g0_f03_k2p import U,fixed_load,physical_stations
from probe_b3_ge_e4_pl_g0_f03_signed_norm_v1 import CASES,couplings
from probe_b3_ge_e4_pl_g0_f03_trace_dual_v1 import primal_factors,space_result


def macro_yaw_tests(n,s):
    if n%2 or n<2 or not (0<=s<=1):
        raise ValueError('even native half-cell count and material station required')
    cell=min(int(s*n),n-1)
    macro=cell//2
    h=1/n
    center=(2*macro+1)*h
    out=np.zeros(n)
    out[2*macro]=1.
    out[2*macro+1]=2*(s-center)/h
    return out


def analytic_pairing(n):
    h=1/n
    expected=np.zeros((n,n))
    for macro in range(n//2):
        block=h*np.array(((1.,1.),(-1.,1.)))
        sl=slice(2*macro,2*macro+2)
        expected[sl,sl]=block
    mass=np.zeros((n,n))
    actual=np.zeros((n,n))
    roots,weights=np.polynomial.legendre.leggauss(4)
    for cell in range(n):
        for root,weight in zip(roots,weights):
            s=(cell+(root+1)/2)*h
            tests=macro_yaw_tests(n,s)
            w=h*weight/2
            actual[:,cell] += w*tests
            mass += w*np.outer(tests,tests)
    expected_mass=np.zeros((n,n))
    for macro in range(n//2):
        sl=slice(2*macro,2*macro+2)
        expected_mass[sl,sl]=np.diag((2*h,8*h/3))
    return actual,expected,mass,expected_mass


def affine_yaw_patch(n):
    h=1/n
    center=h
    roots,weights=np.polynomial.legendre.leggauss(4)
    p0=np.zeros(2)
    macro=np.zeros(2)
    for cell in (0,1):
        for root,weight in zip(roots,weights):
            s=(cell+(root+1)/2)*h
            yaw=.37+(s-center)
            p0[cell] += h*weight*yaw/2
            macro += h*weight*yaw*macro_yaw_tests(n,s)[:2]/2
    return float(abs((p0[1]-p0[0])-h*h)),float(abs(macro[1]-4*h*h/3))


def integrate_yaw(beam,shell,overlay,order):
    n=beam.n_half_cells
    ncols=shell.free_count+len(beam.physical)+len(beam.internal)
    c=np.zeros((n,ncols))
    full=np.zeros((n,shell.dof_count+beam.context.count))
    mass=np.zeros((n,n))
    roots,weights=np.polynomial.legendre.leggauss(order)
    for a,b in zip(overlay[:-1],overlay[1:]):
        for root,weight in zip(roots,weights):
            s=float(.5*(a+b)+.5*(b-a)*root)
            test=macro_yaw_tests(n,s)
            st=shell.trace(s)[2]
            bt=beam.planar_trace_jacobian(s)[2]
            gap=ELL*np.r_[(-st[shell.free],bt[beam.physical],bt[beam.internal])]
            gap_full=ELL*np.r_[-st,bt]
            w=.5*(b-a)*weight
            c += w*test[:,None]*gap[None,:]
            full += w*test[:,None]*gap_full[None,:]
            mass += w*np.outer(test,test)
    return c,full,mass


def yaw_virtual_work_error(beam,shell,overlay,assembled):
    """Independent direct multiplier work on a non-equilibrium trial field."""
    n=beam.n_half_cells
    ncols=assembled.shape[1]
    sq,_=scales(beam,shell,3*n)
    q=sq*np.sin(.37+np.arange(ncols)*.23)
    lam=np.cos(.19+np.arange(n)*.31)
    roots,weights=np.polynomial.legendre.leggauss(4)
    direct=0.
    for a,b in zip(overlay[:-1],overlay[1:]):
        for root,weight in zip(roots,weights):
            s=float(.5*(a+b)+.5*(b-a)*root)
            shell_yaw=shell.trace(s)[2,shell.free]
            beam_yaw=beam.planar_trace_jacobian(s)[2]
            gap=ELL*(-shell_yaw@q[:shell.free_count]
                     +beam_yaw[beam.physical]@q[shell.free_count:
                                                   shell.free_count+len(beam.physical)]
                     +beam_yaw[beam.internal]@q[shell.free_count+len(beam.physical):])
            direct += .5*(b-a)*weight*(lam@macro_yaw_tests(n,s))*gap
    assembled_work=float(lam@assembled[-n:]@q)
    return float(abs(direct-assembled_work)/max(abs(direct),abs(assembled_work),
                                               np.finfo(float).tiny))


def build_spaces(beam,shell):
    native=beam.n_half_cells
    spaces={s.name:s for s in candidate_spaces(beam,shell)}
    p1=assemble(beam,shell,spaces['p1-beam'])
    p0=assemble(beam,shell,spaces['p0-beam'])
    x=[3*i for i in range(1,native+1)]
    y=[3*i+1 for i in range(1,native+1)]
    yaw=[3*i+2 for i in range(native)]
    k4=np.vstack((p1.matrix[x].toarray(),p1.matrix[y].toarray(),
                  p0.matrix[yaw].toarray()))
    k4full=np.vstack((p1.full_constraint[x],p1.full_constraint[y],
                      p0.full_constraint[yaw]))
    k4mass=linalg.block_diag(p1.mass[np.ix_(x,x)],
                              p1.mass[np.ix_(y,y)],p0.mass[np.ix_(yaw,yaw)])
    macro,macrofull,macromass=integrate_yaw(beam,shell,p1.integration_knots,4)
    macro2,_,mass2=integrate_yaw(beam,shell,p1.integration_knots,2)
    c=np.vstack((p1.matrix[x].toarray(),p1.matrix[y].toarray(),macro))
    full=np.vstack((p1.full_constraint[x],p1.full_constraint[y],macrofull))
    mass=linalg.block_diag(p1.mass[np.ix_(x,x)],
                            p1.mass[np.ix_(y,y)],macromass)
    sq,_=scales(beam,shell,len(c))
    quadrature=float(np.linalg.norm((macro-macro2)*sq[None,:])/
                     max(np.linalg.norm(macro*sq[None,:]),np.finfo(float).tiny))
    mass_quadrature=float(np.linalg.norm(macromass-mass2)/np.linalg.norm(macromass))
    work_error=yaw_virtual_work_error(beam,shell,p1.integration_knots,c)
    k4coupling=replace(p1,matrix=sparse.csr_matrix(k4),mass=k4mass,
                       full_constraint=k4full)
    macrocoupling=replace(p1,matrix=sparse.csr_matrix(c),mass=mass,
                          full_constraint=full,
                          max_row_incidence=int(np.max(np.count_nonzero(abs(c)>1e-15,axis=1))),
                          quadrature_error={**p1.quadrature_error,
                                            'macro_C':quadrature,'macro_M':mass_quadrature})
    return (k4,k4coupling),(c,macrocoupling),quadrature,mass_quadrature,work_error


def rank(values):
    sv=linalg.svdvals(values)
    return int(np.count_nonzero(sv>1e-10*sv[0])),float(sv[-1]/sv[0])


def physical(beam,shell,coupling):
    result={}
    for kind in ('smooth','sine'):
        load,full_load=fixed_load(beam,shell,coupling,kind,8)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',linalg.LinAlgWarning)
                diagnostics,solution=solve_exact(beam,shell,coupling,load)
            stations=physical_stations(beam,shell,solution)
            balance=physical_support_balance(
                beam,shell,replace(coupling,full_load_smooth=full_load),
                solution,'smooth')
            result[kind]={
                'max_nine_station_gap_over_U':float(max(
                    np.linalg.norm(s['gap_m']) for s in stations)/U),
                'stations':stations,
                'active_shell_energy_j':balance['active_physical_shell_energy_j'],
                'applied_wrench':balance['applied_wrench'],
                'support_wrench':balance['support_wrench'],
                'max_free_residual':balance['max_free_residual'],
                'normalized_force_balance':balance['normalized_force_balance'],
                'normalized_moment_balance':balance['normalized_moment_balance'],
                'scaled_solution_parity':diagnostics['scaled_solution_parity'],
                'scaled_free_equilibrium':diagnostics['scaled_free_equilibrium'],
                'normalized_weak_residual':diagnostics['normalized_weak_residual']}
        except (linalg.LinAlgError,linalg.LinAlgWarning,
                CandidateSingular,PrimalNotAdmissible) as exc:
            result[kind]={'failure_type':type(exc).__name__,'failure':str(exc)}
    return result


def measure(topology,n_beam,n_shell):
    beam,shell=build(n_beam),make_shell(n_shell,topology)
    if n_beam%2:
        raise ValueError('macrocell yaw requires paired native half-cells')
    actual,expected,mass,expected_mass=analytic_pairing(n_beam)
    pairing_error=float(np.max(abs(actual-expected)))
    mass_error=float(np.max(abs(mass-expected_mass)))
    if max(pairing_error,mass_error)>2e-14:
        raise ValueError('analytical macrocell yaw pairing or mass failed')
    patch_error=affine_yaw_patch(n_beam)
    if max(patch_error)>2e-14:
        raise ValueError('affine shell yaw moment distinction failed')
    (k4,k4coupling),(c,macrocoupling),quadrature,mass_quadrature,work_error=(
        build_spaces(beam,shell))
    if max(quadrature,mass_quadrature,work_error)>1e-11:
        raise ValueError('macrocell yaw quadrature or multiplier work failed')
    sq=primal_factors(beam,shell)[0]
    shell_count=shell.free_count
    k4rank,_=rank(k4*sq[None,:])
    full_rank,ratio=rank(c*sq[None,:])
    native_rank,native_ratio=rank(c[:,shell_count:]*sq[None,shell_count:])
    combined_rank,_=rank(np.vstack((k4,c))*sq[None,:])
    if len(c)!=3*n_beam or full_rank!=len(c) or native_rank!=len(c):
        raise ValueError('macrocell candidate loses full/native row rank')
    distinct=(combined_rank>k4rank)
    p1spaces={s.name:s for s in candidate_spaces(beam,shell)}
    overlay=assemble(beam,shell,p1spaces['p1-beam']).integration_knots
    g,mid,jumps=graph(beam,shell,overlay)
    gs=(sq[:,None]*g)*sq[None,:]/U**2
    eg,vg=linalg.eigh(gs)
    primal=primal_factors(beam,shell)
    space_result_macro=space_result(c,eg,vg,primal)
    space_result_k4=space_result(k4,eg,vg,primal)
    macro_loads=physical(beam,shell,macrocoupling)
    k4_loads=physical(beam,shell,k4coupling)
    load_failures=any('failure_type' in result for loads in (macro_loads,k4_loads)
                      for result in loads.values())
    return {'topology':topology,'n_beam':n_beam,'n_shell':n_shell,
            'rows':len(c),'k4_rank':k4rank,'macro_full_rank':full_rank,
            'macro_native_rank':native_rank,'macro_row_ratio':ratio,
            'macro_native_row_ratio':native_ratio,
            'combined_k4_macro_rank':combined_rank,
            'new_full_span_rows':combined_rank-k4rank,
            'analytic_pairing_max_error':pairing_error,
            'analytic_mass_max_error':mass_error,
            'affine_yaw_patch_errors':patch_error,
            'macro_quadrature_error':quadrature,
            'macro_mass_quadrature_error':mass_quadrature,
            'macro_yaw_virtual_work_rel_error':work_error,
            'max_affine_midpoint_error':mid,'native_yaw_jump_count':len(jumps),
            'maximum_macro_row_incidence':macrocoupling.max_row_incidence,
            'macro_trace_dual':space_result_macro,
            'k4_trace_dual':space_result_k4,
            'macro_loads':macro_loads,
            'k4_loads':k4_loads,
            'status':('EXPLORATORY_LOAD_FAILURE_NOT_F03_ACCEPTANCE' if load_failures
                      else 'NO_DISTINCT_CONSTRAINT_SPAN_NOT_F03_ACCEPTANCE' if not distinct
                      else 'EXPLORATORY_MACROCELL_YAW_NOT_F03_ACCEPTANCE')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=False)
    names=(Path(__file__).name,'_b3_e4_f03_native_fixture.py',
           '_b3_e4_f03_shell_trace.py','probe_b3_ge_e4_pl_g0_f03_kkt.py',
           'probe_b3_ge_e4_pl_g0_f03_k1_rank.py',
           'probe_b3_ge_e4_pl_g0_f03_k2p.py',
           'probe_b3_ge_e4_pl_g0_f03_signed_norm_v1.py',
           'probe_b3_ge_e4_pl_g0_f03_broken_trace_v1.py',
           'probe_b3_ge_e4_pl_g0_f03_trace_dual_v1.py')
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in names}
    manifest={'purpose':'EXPLORATORY_DISTINCT_LOCAL_YAW_SPAN_NOT_F03_ACCEPTANCE',
              'cases':CASES,'spaces':('k4','macro-yaw'),
              'source_sha256':hashes,'anyfesolver_import':str(Path(anyfesolver.__file__).resolve()),
              'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
              'numba':numba.__version__,'cache':os.environ['NUMBA_CACHE_DIR'],
              'blas_threads':1,'max_dense_kkt_rows':1200,
              'force_density_n_per_m':1e6,'yaw_conversion_length_m':ELL}
    (args.root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',
                                          encoding='utf-8')
    failed=False
    with threadpool_limits(limits=1),(args.root/'results.jsonl').open('x',encoding='utf-8') as out:
        for case in CASES:
            try:
                row=measure(*case)
            except Exception as exc:
                row={'case':case,'failure_type':type(exc).__name__,
                     'failure':str(exc),'status':'EXPLORATORY_FAILURE'}
                failed=True
            out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            out.flush()
            failed |= row['status']!='EXPLORATORY_MACROCELL_YAW_NOT_F03_ACCEPTANCE'
            print(json.dumps({'case':case,'failure':row.get('failure'),
                              'span_added':row.get('new_full_span_rows'),
                              'macro_floor':row.get('macro_trace_dual',{}).get(
                                  'signed_kkt_min_singular')},sort_keys=True),flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
