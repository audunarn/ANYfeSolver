"""One fixed-mesh BL5 work allocation; diagnostic only, no product condensation.

Compare exact local stationary elimination, a counterfactual frozen retained
lift, and strong nodal ties. The latter includes five endpoint rows at the
root, unlike integrated BL5. No SPD interpretation or multiplier selection.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import subprocess
import sys
from math import fsum, fma

import anyfesolver
from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_ge_experimental import _linear_equilibrium, _scaled_constraint_gap
from anyfesolver._stiffened_shell_ge_experimental import _require_independent_supported_rows
import numpy as np
import numba
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info, threadpool_limits

from b3_e4_f03_bl5_native_nodal_comparator_dev import (
    INTERVALS, LOADS, _load, _metrics, _nodal_rows, _strong_case,
)
from test_stiffened_shell_l1_assembly_experimental import _bl5_joint_mesh_definition

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticNumericalFailure(RuntimeError):
    """A computed counterfactual did not satisfy the existing numerical checks."""


def _quality(owner,k,c,b,w,free,load,q,lam,row_scale):
    residual = k@q+b@lam-load
    equilibrium = _linear_equilibrium(residual[free],load[free],owner._coordinate_scale[free])
    gap = _scaled_constraint_gap(c@q-w@lam,row_scale)
    if equilibrium > 1e-8 or gap > 1e-9:
        raise DiagnosticNumericalFailure(f"equilibrium={equilibrium}, constraint={gap}")
    actual, adjoint = b@lam, c.T@lam
    scale = owner._coordinate_scale[:len(q)]
    relative = float(np.linalg.norm((actual-adjoint)*scale)/max(
        np.linalg.norm(actual*scale),np.linalg.norm(adjoint*scale),1e-20))
    if relative > 1e-8:
        raise DiagnosticNumericalFailure(f"reduced coupling/adjoint work-map defect={relative}")
    return {'scaled_equilibrium':equilibrium,'scaled_constraint':gap,
            'coupling_adjoint_relative_work_norm':relative,
            'coupling_adjoint_actual_work_difference_j':float((actual-adjoint)@q)}


def _add(rows, columns, values, ri, ci, block):
    """Scatter one locally bounded block; never construct a global dense array."""
    rows.extend(np.repeat(ri, len(ci)).tolist())
    columns.extend(np.tile(ci, len(ri)).tolist())
    values.extend(block.ravel().tolist())


def _assemble(shape, triples):
    rows, columns, values = triples
    result = sparse.coo_matrix((values, (rows, columns)), shape=shape).tocsr()
    result.sum_duplicates(); result.eliminate_zeros()
    return result


def _seed(matrix):
    entries = matrix.tocoo()
    return entries.row.tolist(), entries.col.tolist(), entries.data.tolist()


def _reduce(owner):
    k, c = owner._K.tocsr(), owner._C.tocsr()
    n = owner._shell_count + 6*len(owner._beam_ids)
    blocks = len(owner._beam_parts)
    if k.shape != (n+24*blocks, n+24*blocks):
        raise ValueError("diagnostic requires unchanged native generalized layout")
    m = c.shape[0]
    kk, cc, bb, ww = _seed(k[:n, :n]), _seed(c[:, :n]), _seed(c[:, :n].T), ([], [], [])
    maps, retained_blocks = [], []
    for i in range(blocks):
        r = np.arange(n+24*i, n+24*(i+1))
        rr = k[r, :][:, r].tocsc()
        retained_blocks.append(rr)
        rn = k[r, :n]
        p = np.unique(rn.nonzero()[1])
        cr = c[:, r]
        rows = np.unique(cr.nonzero()[0])
        if len(p) > 18 or len(rows) > 10:
            raise ValueError("fixed P0 fixture local scatter exceeded its declared layout")
        factor = splu(rr)
        lift = factor.solve(-rn[:, p].toarray())
        response = factor.solve(cr[rows, :].T.toarray())
        nr = k[p, :][:, r].toarray()
        local_c = cr[rows, :].toarray()
        _add(*kk, p, p, nr@lift)
        _add(*cc, rows, p, local_c@lift)
        _add(*bb, p, rows, -nr@response)
        _add(*ww, rows, rows, local_c@response)
        maps.append((r, p, rows, lift, response, factor))
    defect = k[n:, n:] - sparse.block_diag(retained_blocks, format="csr")
    if np.max(np.abs(defect.data), initial=0.) != 0.:
        raise ValueError("retained blocks are not independent in this fixture")
    return n, _assemble((n,n),kk), _assemble((m,n),cc), _assemble((n,m),bb), _assemble((m,m),ww), maps


def _reconstruct(n, q, lam, maps):
    full = np.zeros(n+24*len(maps)); full[:n] = q
    for r, p, rows, lift, response, _ in maps:
        full[r] = lift@q[p] - response@lam[rows]
    return full


def _solve(k, c, b, w, free, load):
    matrix = sparse.bmat([[k[free,:][:,free], b[free,:]],
                          [c[:,free], -w]], format="csc")
    rhs = np.r_[load[free], np.zeros(c.shape[0])]
    factor = splu(matrix)
    result = factor.solve(rhs)
    for _ in range(2): result += factor.solve(rhs-matrix@result)
    if not np.isfinite(result).all():
        raise ValueError("finite diagnostic solution required")
    q = np.zeros(k.shape[0]); q[free] = result[:len(free)]
    residual = matrix@result-rhs
    return q, result[len(free):], matrix, float(np.linalg.norm(residual)/max(np.linalg.norm(rhs),1.)), factor


def _accurate_residual(matrix,rhs,x):
    """Signed RHS minus products, retaining FMA product tails before fsum."""
    a=matrix.tocsr(); result=np.empty(a.shape[0])
    for row in range(a.shape[0]):
        terms=[float(rhs[row])]
        for index in range(a.indptr[row],a.indptr[row+1]):
            v,u=float(a.data[index]),float(x[a.indices[index]])
            product=v*u
            terms.extend((-product,-fma(v,u,-product)))
        result[row]=fsum(terms)
    return result


def _full_refine(owner,c,load,q,lam):
    free=owner._free
    matrix=sparse.bmat([[owner._K[free,:][:,free],c[:,free].T],
                       [c[:,free],None]],format='csc')
    residual=_accurate_residual(matrix,np.r_[load[free],np.zeros(len(lam))],np.r_[q[free],lam])
    correction=splu(matrix).solve(residual)
    changed=q.copy();changed[free]+=correction[:len(free)]
    return changed,lam+correction[len(free):],residual,correction


def _eliminated_refine(owner,c,n,maps,load,q,lam,matrix,factor,*,retained_action=True):
    original=sparse.bmat([[owner._K,c.T],[c,None]],format='csr')
    residual=_accurate_residual(original,np.r_[load,np.zeros(len(lam))],np.r_[q,lam])
    z=np.zeros(len(q)-n)
    for r,p,rows,lift,response,local_factor in maps:
        z[r-n]=local_factor.solve(residual[r])
    free=owner._free[owner._free<n]
    rn=residual[:n]-owner._K[:n,n:]@z
    rc=residual[len(q):]-c[:,n:]@z
    correction=factor.solve(np.r_[rn[free],rc])
    dq=np.zeros(n);dq[free]=correction[:len(free)]
    dl=correction[len(free):]
    dr=_reconstruct(n,dq,dl if retained_action else np.zeros(owner._C.shape[0]),maps)
    dr[n:]+=z
    return q+dr,lam+dl,residual,np.r_[dr,dl]


def _reduced_refine(matrix,factor,q,lam,free,load):
    residual=_accurate_residual(matrix,np.r_[load[free],np.zeros(len(lam))],np.r_[q[free],lam])
    correction=factor.solve(residual)
    changed=q.copy();changed[free]+=correction[:len(free)]
    return changed,lam+correction[len(free):],residual,correction


def _parity(a, b, n, *, require=True):
    tail_a, tail_b = a[n:].reshape(-1,24), b[n:].reshape(-1,24)
    pairs = (("nodal", a[:n], b[:n]),
             ("retained_rotations", tail_a[:,:6], tail_b[:,:6]),
             ("retained_resultants", tail_a[:,6:], tail_b[:,6:]))
    ratios = {name:float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20))
              for name,x,y in pairs}
    if require and max(ratios.values()) > 1e-10:
        raise ValueError(f"stationary reconstruction fails registered small-case parity: {ratios}")
    return ratios


def main(destination):
    if destination.exists(): raise FileExistsError("preserve previous decision output")
    destination.mkdir()
    expected = (ROOT/'src/anyfesolver/__init__.py').resolve()
    if Path(anyfesolver.__file__).resolve() != expected:
        raise ValueError("decision imported another solver checkout")
    definition = _bl5_joint_mesh_definition(INTERVALS, INTERVALS, 1)
    report = {"status":"IN_PROGRESS", "source_git":subprocess.check_output(
        ['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "imported_anyfesolver":str(expected), "physical_shell":asdict(definition.shell),
        "runtime":{"python":sys.version,"numba_jit_disabled":bool(numba.config.DISABLE_JIT),
                   "numba_threads":numba.config.NUMBA_NUM_THREADS},
        "beam_definition_hashes":[b.sha256 for b in definition.beams],
        "load":LOADS['mx'], "load_node":101+INTERVALS,
        "supports":[asdict(s) for s in definition.fixed],
        "line":definition.lines[0].descriptor(),
        "limitations":"One fixed reference-linear native torque case; frozen lift is a counterfactual, not product physics. Strong ties include root endpoints. No SPD, external continuous-trace bridge, multiplier selection, F03 or gate acceptance."}
    owner = None
    try:
        with threadpool_limits(limits=1):
            report['threadpools'] = threadpool_info()
            if (not report['threadpools'] or any(p['num_threads']!=1 for p in report['threadpools'])
                    or not numba.config.DISABLE_JIT or numba.config.NUMBA_NUM_THREADS!=1):
                raise ValueError("bounded decision requires effective one-thread/JIT-off settings")
            owner = stiffened_shell.create_analysis(definition=definition)
            report['model_identity'] = owner._identity
            full = owner.solve_linear((stiffened_shell.NodalAction('beam',101+INTERVALS,LOADS['mx']),))
            qw = np.r_[full.shell_displacement,full.beam_coordinates]
            lw = full.multipliers; load = _load(owner,'mx')
            n,k,c,b,w,maps = _reduce(owner)
            for name,value in (('K',owner._K),('C',owner._C),
                               ('Kbar',k),('Cbar',c),('Bbar',b),('W',w)):
                sparse.save_npz(destination/(name+'.npz'),value)
            free = owner._free[owner._free<n]
            qr,lr,matrix,defect,rfactor = _solve(k,c,b,w,free,load[:n])
            reconstructed = _reconstruct(n,qr,lr,maps)
            report['raw_full_reconstruction_parity'] = _parity(reconstructed,qw,n,require=False)
            records={'raw_full':qw.copy(),'raw_full_lambda':lw.copy(),
                     'raw_reconstructed':reconstructed.copy(),'raw_reconstructed_lambda':lr.copy()}
            qw,lw,records['full_residual_before'],records['full_correction'] = _full_refine(owner,owner._C,load,qw,lw)
            reconstructed,lr,records['eliminated_residual_before'],records['eliminated_correction'] = _eliminated_refine(
                owner,owner._C,n,maps,load,reconstructed,lr,matrix,rfactor)
            qr=reconstructed[:n]
            records.update(corrected_full=qw,corrected_full_lambda=lw,
                           corrected_reconstructed=reconstructed,corrected_reconstructed_lambda=lr,
                           load=load,free=owner._free,nodal_free=free)
            np.savez_compressed(destination/'refinement.npz',**records)
            report['arithmetic']='One compensated-residual correction per independent full/local-elimination route; no full factor applied to the eliminated answer. Raw v2 failure preserved; thresholds unchanged.'
            report['full_reconstruction_parity'] = _parity(reconstructed,qw,n)
            report['full_multiplier_parity'] = float(np.linalg.norm(lr-lw)/max(np.linalg.norm(lw),1e-20))
            if report['full_multiplier_parity'] > 1e-10:
                raise ValueError("stationary reconstruction fails multiplier parity")
            report['raw_full_reduced_equilibrium_relative'] = defect
            report['full_original_checks'] = _quality(owner,owner._K,owner._C,owner._C.T,
                sparse.csr_matrix(w.shape),owner._free,load,qw,lw,owner._constraint_scale)
            report['reconstructed_original_checks'] = _quality(owner,owner._K,owner._C,owner._C.T,
                sparse.csr_matrix(w.shape),owner._free,load,reconstructed,lr,owner._constraint_scale)
            report['full_reduced_checks'] = _quality(owner,k,c,b,w,free,load[:n],qr,lr,owner._constraint_scale)
            report['full'] = _metrics(owner,qw,lw,owner._C,load)
            nodal = _nodal_rows(owner)
            sm,_,qs,ls = _strong_case(owner,nodal,'mx')
            cn = nodal[:,:n].tocsr()
            qsn,lsn,smatrix,sdefect,sfactor = _solve(k,cn,cn.T,sparse.csr_matrix((len(ls),len(ls))),free,load[:n])
            strong_reconstruction = _reconstruct(n,qsn,np.zeros_like(lw),maps)
            report['raw_strong_reconstruction_parity'] = _parity(strong_reconstruction,qs,n,require=False)
            records.update(raw_strong=qs.copy(),raw_strong_lambda=ls.copy(),
                           raw_strong_reconstruction=strong_reconstruction.copy(),raw_strong_reconstruction_lambda=lsn.copy())
            qs,ls,records['strong_residual_before'],records['strong_correction'] = _full_refine(owner,nodal,load,qs,ls)
            strong_reconstruction,lsn,records['strong_eliminated_residual_before'],records['strong_eliminated_correction'] = _eliminated_refine(
                owner,nodal,n,maps,load,strong_reconstruction,lsn,smatrix,sfactor,retained_action=False)
            qsn=strong_reconstruction[:n]
            sm=_metrics(owner,qs,ls,nodal,load)
            records.update(corrected_strong=qs,corrected_strong_lambda=ls,
                           corrected_strong_reconstruction=strong_reconstruction,
                           corrected_strong_reconstruction_lambda=lsn)
            np.savez_compressed(destination/'refinement.npz',**records)
            report['strong_reconstruction_parity'] = _parity(strong_reconstruction,qs,n)
            report['raw_strong_reduced_equilibrium_relative'] = sdefect
            strong_scale=np.asarray(abs(nodal)@owner._coordinate_scale).ravel()
            report['strong_original_checks'] = _quality(owner,owner._K,nodal,nodal.T,
                sparse.csr_matrix((len(ls),len(ls))),owner._free,load,qs,ls,strong_scale)
            report['strong_reconstructed_original_checks'] = _quality(owner,owner._K,nodal,nodal.T,
                sparse.csr_matrix((len(ls),len(ls))),owner._free,load,strong_reconstruction,lsn,strong_scale)
            report['strong_reduced_checks'] = _quality(owner,k,cn,cn.T,
                sparse.csr_matrix((len(ls),len(ls))),free,load[:n],qsn,lsn,
                np.asarray(abs(cn)@owner._coordinate_scale[:n]).ravel())
            report['strong_nodal'] = sm
            cross_strong = ls*(nodal@qw)
            cross_weak = lw*(owner._C@qs)
            difference = float(load@(qw-qs))
            closure = float(np.sum(cross_strong)-np.sum(cross_weak)-difference)
            report['reciprocal_cross_work'] = {
                'applied_difference_j':difference,'strong_root_term_j':float(sum(cross_strong[:5])),
                'strong_nonroot_term_j':float(sum(cross_strong[5:])),
                'weak_on_strong_term_j':float(sum(cross_weak)),
                'weak_on_strong_by_channel_j':cross_weak.reshape(5,-1).sum(axis=1).tolist(),
                'closure_j':closure}
            if abs(closure) > 1e-8*max(abs(load@qw),abs(load@qs),1e-20):
                raise ValueError("reciprocal work allocation does not close")
            induced = _reconstruct(n,qw[:n],lw,maps)-_reconstruct(n,qw[:n],np.zeros_like(lw),maps)
            report['retained_correction_work_j'] = float(lw@(owner._C@induced))
            report['negative_lambda_W_lambda_j'] = float(-lw@(w@lw))
            if abs(report['retained_correction_work_j']-report['negative_lambda_W_lambda_j']) > 1e-8*max(abs(load@qw),1e-20):
                raise ValueError("retained correction work does not match exact W action")
            report['effective_trace_on_strong_by_channel'] = np.max(
                np.abs(c@qs[:n]).reshape(5,-1),axis=1).tolist()
            report['projected_action_work_j'] = float((c.T@lw)@qw[:n])
            qf,lf,frozen_full = np.empty(0),np.empty(0),np.empty(0)
            try:
                qf,lf,fmatrix,fdefect,ffactor = _solve(k,c,b,sparse.csr_matrix(w.shape),free,load[:n])
                records.update(raw_frozen=qf.copy(),raw_frozen_lambda=lf.copy())
                qf,lf,records['frozen_residual_before'],records['frozen_correction'] = _reduced_refine(
                    fmatrix,ffactor,qf,lf,free,load[:n])
                records.update(corrected_frozen=qf,corrected_frozen_lambda=lf)
                np.savez_compressed(destination/'refinement.npz',**records)
                frozen_full = _reconstruct(n,qf,np.zeros_like(lf),maps)
                checks = _quality(owner,k,c,b,sparse.csr_matrix(w.shape),free,load[:n],qf,lf,owner._constraint_scale)
                report['frozen_lift'] = {'status':'completed','applied_work_j':float(load[:n]@qf),
                    'checks':checks,
                    'raw_reduced_equilibrium_relative':fdefect,
                    'effective_gap_max':float(np.max(np.abs(c@qf))),
                    'full_minus_frozen_work_j':float(load[:n]@(qr-qf)),
                    'frozen_minus_strong_work_j':float(load[:n]@(qf-qsn)),
                    'interface_nodal_action':(-c.T@lf).tolist()}
            except RuntimeError as error:
                report['frozen_lift'] = {'status':('numerical_checks_failed' if
                    type(error) is DiagnosticNumericalFailure else 'backend_failed'),
                    'failure_type':type(error).__name__,'details':str(error)}
            for name,value in (('K',owner._K),('C',owner._C),('C_nodal',nodal),
                               ('Kbar',k),('Cbar',c),('Bbar',b),('W',w)):
                sparse.save_npz(destination/(name+'.npz'),value)
            np.savez_compressed(destination/'states.npz',full=qw,full_lambda=lw,
                reconstructed=reconstructed,reconstructed_lambda=lr,strong=qs,strong_lambda=ls,
                frozen=frozen_full,frozen_lambda=lf,load=load,free=owner._free,
                full_projected_nodal_action=-c.T@lw)
            report['sizes'] = {'nodal':n,'retained':len(qw)-n,'multipliers':len(lw),
                'full_primal':owner._K.shape[0],'reduced_mixed':matrix.shape[0],
                'local_blocks':len(maps),'max_local_retained':24,
                'reduced_nnz':matrix.nnz,'W_nnz':w.nnz}
            report['status'] = 'COMPLETED_NATIVE_ALLOCATION_DIAGNOSTIC'
    except Exception as error:
        report['status'] = 'FAILED';report['failure'] = {'type':type(error).__name__,'details':str(error)}
        raise
    finally:
        if owner is not None: owner.close()
        (destination/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'destination':str(destination)}))


def boundary_control(destination, previous):
    """Add comparator nodal root restraint, not a physical section endpoint law."""
    if destination.exists(): raise FileExistsError("preserve prior boundary output")
    destination.mkdir()
    source=json.loads((previous/'report.json').read_text(encoding='utf-8'))
    if source['status'] != 'COMPLETED_NATIVE_ALLOCATION_DIAGNOSTIC':
        raise ValueError("completed original allocation required")
    report={'status':'IN_PROGRESS','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parent_report_sha256':hashlib.sha256((previous/'report.json').read_bytes()).hexdigest(),
            'consumed_parent_sha256':{name:hashlib.sha256((previous/name).read_bytes()).hexdigest()
                for name in ('states.npz','K.npz','C.npz','C_nodal.npz')},
            'source_git':source['source_git'],'runtime':{'python':sys.version},
            'semantics':'Unchanged integrated BL5 plus five comparator nodal-triad root rows. Not retained section endpoint enforcement or product support. No multiplier selection or F03 acceptance.'}
    owner=None
    try:
        with threadpool_limits(limits=1):
            report['threadpools']=threadpool_info()
            if (Path(anyfesolver.__file__).resolve()!=(ROOT/'src/anyfesolver/__init__.py').resolve()
                    or any(p['num_threads']!=1 for p in report['threadpools'])
                    or not report['threadpools'] or not numba.config.DISABLE_JIT
                    or numba.config.NUMBA_NUM_THREADS!=1):
                raise ValueError("effective source/one-thread/JIT-off contract required")
            owner=stiffened_shell.create_analysis(definition=_bl5_joint_mesh_definition(INTERVALS,INTERVALS,1))
            if owner._identity!=source['model_identity']: raise ValueError("changed parent model")
            for name,current in (('K',owner._K),('C',owner._C)):
                difference=current-sparse.load_npz(previous/(name+'.npz'))
                if np.max(abs(difference.data),initial=0.)!=0.: raise ValueError("changed original operator")
            nodal=_nodal_rows(owner);root=nodal[:5]
            root_spin=owner._shell_count+6*owner._beam_index[101]+5
            if root[:,root_spin].nnz: raise ValueError("root counterfactual constrains normal spin")
            report['root_normal_spin_column_is_zero']=True
            rows=sparse.vstack((owner._C,root),format='csr')
            scale=np.asarray(abs(rows)@owner._coordinate_scale).ravel()
            _require_independent_supported_rows(rows,owner._free,owner._coordinate_scale,scale)
            # Bounded sparse numerical row independence, not an inf-sup metric.
            scaled=sparse.diags(1./scale)@rows[:,owner._free]@sparse.diags(owner._coordinate_scale[owner._free])
            from scipy.sparse.linalg import svds
            low=float(svds(scaled,k=1,which='SM',solver='propack',return_singular_vectors=False,
                           rng=0,maxiter=scaled.shape[0])[0])
            high=float(svds(scaled,k=1,which='LM',solver='propack',return_singular_vectors=False,
                            rng=0,maxiter=scaled.shape[0])[0])
            floor=float(np.finfo(float).eps*max(scaled.shape)*high)
            if not np.isfinite((low,high,floor)).all() or high<=0.:
                raise DiagnosticNumericalFailure("invalid supported rank diagnostic")
            report['bounded_row_independence']={'smallest':low,'largest':high,'roundoff_floor':floor,
                                              'rows':scaled.shape[0],'columns':scaled.shape[1]}
            if low<=floor: raise DiagnosticNumericalFailure("stacked root rows numerically dependent")
            load=_load(owner,'mx')
            q,lam,matrix,rawdefect,_=_solve(owner._K,rows,rows.T,
                sparse.csr_matrix((rows.shape[0],rows.shape[0])),owner._free,load)
            records={'raw':q.copy(),'raw_lambda':lam.copy(),'free':owner._free,'load':load}
            q,lam,records['residual_before'],records['correction']=_full_refine(owner,rows,load,q,lam)
            records.update(corrected=q,corrected_lambda=lam)
            np.savez_compressed(destination/'states.npz',**records)
            sparse.save_npz(destination/'C_augmented.npz',rows)
            report['raw_equilibrium_relative']=rawdefect
            report['checks']=_quality(owner,owner._K,rows,rows.T,
                sparse.csr_matrix((rows.shape[0],rows.shape[0])),owner._free,load,q,lam,scale)
            report['augmented']=_metrics(owner,q,lam,rows,load)
            work=float(load@q)
            total_work_error=float(q@(owner._K@q)-work)
            shell_work_error=report['augmented']['shell_work_error_j']
            wrench=report['augmented']['support_action']
            force=np.asarray(wrench['force_n'])+np.asarray(LOADS['mx'][:3])
            moment=np.asarray(wrench['moment_nm'])+np.asarray(LOADS['mx'][3:])
            balance=max(np.linalg.norm(force)*owner._reference_length,np.linalg.norm(moment))/10.
            report['physical_checks']={'total_work_error_j':total_work_error,
                'shell_work_error_j':shell_work_error,'scaled_global_wrench_balance':float(balance)}
            if (max(abs(total_work_error),abs(shell_work_error))>1e-8*max(abs(work),1e-20)
                    or balance>1e-8):
                raise DiagnosticNumericalFailure("augmented work or global wrench balance failed")
            for name,c,multipliers in (('integrated_line',owner._C,lam[:owner._C.shape[0]]),
                                        ('concentrated_root',root,lam[-5:])):
                partial=_metrics(owner,q,multipliers,c,load)
                report[name]={'shell_action_work_j':partial['shell_action_work_j'],
                              'interface_shell_action':partial['interface_shell_action'],
                              'semantics':'Partial interface action only, not separate equilibrium or support reaction.'}
            with np.load(previous/'states.npz',allow_pickle=False) as parent:
                weak,strong=parent['full'].copy(),parent['strong'].copy()
                lw,ls=parent['full_lambda'].copy(),parent['strong_lambda'].copy()
            for name,state,multipliers,c in (('weak',weak,lw,owner._C),('strong',strong,ls,nodal)):
                rs=np.asarray(abs(c)@owner._coordinate_scale).ravel()
                report[name+'_reused_checks']=_quality(owner,owner._K,c,c.T,
                    sparse.csr_matrix((c.shape[0],c.shape[0])),owner._free,load,state,multipliers,rs)
            weakwork,strongwork,augwork=(float(load@x) for x in (weak,strong,q))
            comparison={'weak_work_j':weakwork,'strong_work_j':strongwork,'augmented_work_j':augwork,
                'weak_minus_augmented_j':weakwork-augwork,'augmented_minus_strong_j':augwork-strongwork,
                'remaining_fraction_of_original_difference':(augwork-strongwork)/(weakwork-strongwork),
                'nodal_root_gap_before':(root@weak).tolist(),'nodal_root_gap_after':(root@q).tolist(),
                'weak_line_gap_after_scaled':_scaled_constraint_gap(owner._C@q,owner._constraint_scale),
                'root_gap_after_scaled':_scaled_constraint_gap(root@q,scale[-5:])}
            nw=owner._shell_count
            if owner._shell_index[1]!=0 or owner._beam_index[101]!=0:
                raise ValueError("changed fixture root ordering")
            weakaction=-owner._C.T@lw;strongaction=-nodal.T@ls;augaction=-rows.T@lam
            nonroot=np.arange(6,nw);ds=owner._coordinate_scale[:nw]
            denominator=max(np.linalg.norm(strongaction[nonroot]*ds[nonroot]),1e-20)
            for name,state,action in (('weak',weak,weakaction),('strong',strong,strongaction),('augmented',q,augaction)):
                comparison[name+'_beam_nodal_root_spin_rad']=float(state[nw+5])
                comparison[name+'_nonroot_scaled_shell_action_relative']=float(
                    np.linalg.norm((action-strongaction)[nonroot]*ds[nonroot])/denominator)
            report['comparison']=comparison
            report['status']='COMPLETED_NODAL_ROOT_COUNTERFACTUAL'
    except Exception as error:
        report['status']='FAILED';report['failure']={'type':type(error).__name__,'details':str(error)}
        raise
    finally:
        if owner is not None: owner.close()
        (destination/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'destination':str(destination)}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--boundary-source',type=Path)
    args=parser.parse_args()
    if args.boundary_source: boundary_control(args.destination,args.boundary_source)
    else: main(args.destination)
