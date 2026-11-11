"""Bounded saved native mixed-system check for the projected density candidate.

Reuse the existing <=1000-physical-coordinate oracle contract. Eliminate only
checked local stress blocks, retaining every nodal/cell rotation. This is not
production condensation, a full-K SPD interpretation, or F03 acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import anyfesolver
import numba
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info
from anyfesolver._stiffened_shell_ge_experimental import _linear_equilibrium, _scaled_constraint_gap
from scripts.r8_bl5_projected_density_patch_dev import compile_saved_projection, IDENTITY

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


# Same recorded numerical uncertainty measures as the bounded mechanical oracle;
# do not import its owner-construction/test-fixture graph for a saved-matrix check.
def relative(a,b,floor=1e-14):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),floor))


def norm2(a):
    return float(linalg.norm(a,2))


def lower_completion_difference(a):
    return norm2(a-(np.tril(a)+np.tril(a,-1).T))


def evaluate(root, output):
    start=time.monotonic()
    (inputs,k,old_c,reconstructed,c,rigid,shell_work,beam_work,fixed,state,
     records,_,_,_)=compile_saved_projection(root)
    assert inputs['layout']['nx']==16 and c.shape==(80,1092)
    original_parity=float(sparse.linalg.norm(reconstructed-old_c)/max(1.,sparse.linalg.norm(old_c)))
    assert original_parity<=1e-12
    base=inputs['layout']['counts']['physical_dofs'];members=inputs['layout']['counts']['members']
    with np.load(root/'native/operators.npz',allow_pickle=False) as z:
        free=z['free'].copy();coordinate=z['coordinate_scale'].copy();constraint=z['constraint_scale'].copy()
    blocks=[base+24*i+np.arange(6,24) for i in range(members)]
    pp=np.concatenate(blocks);x=np.setdiff1d(free,pp)
    assert len(x)<=1000 and len(free)+c.shape[0]<=1500
    assert np.isin(pp,free).all() and c[:,pp].nnz==0
    independent=k[pp][:,pp]-sparse.block_diag([k[p][:,p] for p in blocks])
    assert np.max(np.abs(independent.data),initial=0.)==0.
    full_skew=float(sparse.linalg.norm(k-k.T)/sparse.linalg.norm(k));assert full_skew<=1e-9
    s=k[x][:,x].tocsr();free_map={int(v):i for i,v in enumerate(free)}
    rows=[free_map[int(v)] for v in x];columns=list(range(len(x)));values=[1.]*len(x)
    maps=[];local_info=[]
    for p in blocks:
        d=k[p][:,p].toarray();assert relative(d,d.T)<=1e-9
        factor=linalg.cho_factor(-d,lower=True);chol=np.tril(factor[0]);factor_error=d+chol@chol.T
        reconstruction=norm2(factor_error);local_eigenvalues=linalg.eigvalsh(d,lower=True)
        budget=reconstruction+64*np.finfo(float).eps*len(d)*norm2(d)
        assert -local_eigenvalues[-1]>budget
        coupling=k[p][:,x];neighbours=np.unique(coupling.nonzero()[1]);assert len(neighbours)<=24
        px=coupling[:,neighbours].toarray();lift=linalg.cho_solve(factor,px)
        residual=d@lift+px
        backward=float(np.linalg.norm(residual)/max(np.linalg.norm(d)*np.linalg.norm(lift)+np.linalg.norm(px),1e-30))
        assert backward<=1e-10
        update=k[x[neighbours]][:,p].toarray()@lift
        s+=sparse.coo_matrix((update.ravel(),(np.repeat(neighbours,len(neighbours)),np.tile(neighbours,len(neighbours)))),shape=s.shape).tocsr()
        rows.extend(np.repeat([free_map[int(v)] for v in p],len(neighbours)))
        columns.extend(np.tile(neighbours,len(p)));values.extend(lift.ravel())
        maps.append((neighbours,lift,factor,residual,factor_error))
        local_info.append(dict(maximum_negative_eigenvalue=float(local_eigenvalues[-1]),
            negative_block_budget=budget,stationary_backward_residual=backward))
    lifted=sparse.coo_matrix((values,(rows,columns)),shape=(len(free),len(x))).tocsr()
    kfree=k[free][:,free];cx=c[:,x].toarray();work_pullback=lifted.T@kfree@lifted
    work_parity=float(sparse.linalg.norm(work_pullback-s)/max(sparse.linalg.norm(s),1e-30))
    constraint_parity=float(np.linalg.norm(c[:,free]@lifted-cx)/max(np.linalg.norm(cx),1e-30))
    assert max(work_parity,constraint_parity)<=1e-10
    scale=coordinate[x].copy();diagonal=np.abs(s.diagonal()*scale**2)
    scale/=np.sqrt(np.where(diagonal>0.,diagonal,np.max(diagonal)))
    sd=sparse.diags(scale);scaled=(sd@s@sd).toarray();assert relative(scaled,scaled.T)<=1e-9
    cscaled=cx*scale;row_norm=np.linalg.norm(cscaled,axis=1);assert np.all(row_norm>0.)
    cscaled/=row_norm[:,None];singular=linalg.svdvals(cscaled);eps=np.finfo(float).eps
    rank_budget=64*eps*max(cscaled.shape)*singular[0];assert singular[-1]>rank_budget
    qbasis,_=linalg.qr(cscaled.T,mode='full');z=qbasis[:,c.shape[0]:]
    restricted=z.T@scaled@z;ev,vectors=linalg.eigh(restricted,lower=True)
    snorm=float(np.max(np.abs(linalg.eigvalsh(scaled,lower=True))))
    orthogonality=norm2(z.T@z-np.eye(z.shape[1]));leakage=norm2(cscaled@z);eta=leakage/singular[-1]
    eliminated_error=norm2((sd@(work_pullback-s)@sd).toarray())
    for neighbours,lift,factor,residual,factor_error in maps:
        r=residual*scale[neighbours];eliminated_error+=norm2(r.T@linalg.cho_solve(factor,r))
        sl=lift*scale[neighbours];eliminated_error+=norm2(sl.T@factor_error@sl)
    terms=dict(rounding=64*eps*len(x)*snorm,elimination_and_reconstruction=eliminated_error,
        kernel_leakage=(2*eta+eta**2)*snorm,kernel_orthogonality=(2*orthogonality+orthogonality**2)*snorm,
        physical_triangle_completion=lower_completion_difference(scaled),
        eigen_residual=norm2(restricted@vectors-vectors*ev))
    uncertainty=sum(terms.values());kernel_positive=bool(ev[0]>uncertainty)
    saddle=np.block([[scaled,cscaled.T],[cscaled,np.zeros((c.shape[0],c.shape[0]))]])
    saddle_ev=linalg.eigvalsh(saddle,lower=True)
    saddle_budget=64*eps*len(saddle)*float(np.max(np.abs(saddle_ev)))+uncertainty+lower_completion_difference(saddle)
    inertia=dict(positive=int(np.sum(saddle_ev>saddle_budget)),negative=int(np.sum(saddle_ev<-saddle_budget)),
        unresolved=int(np.sum(np.abs(saddle_ev)<=saddle_budget)))
    expected=dict(positive=len(x),negative=c.shape[0],unresolved=0)
    # Full general KKT and reduced symmetric oracle each factor ONCE for 3 RHS.
    cases=('fy','mx','mz');loads=[]
    for case in cases:
        with np.load(root/'native'/case/'raw-state.npz',allow_pickle=False) as a: loads.append(a['load'].copy())
    forces=np.column_stack(loads);assert np.max(np.abs(forces[pp]))==0.
    full=sparse.bmat([[kfree,c[:,free].T],[c[:,free],None]],format='csc')
    full_factor=splu(full);solution=full_factor.solve(np.vstack((forces[free],np.zeros((c.shape[0],3)))))
    reduced=linalg.solve(saddle,np.vstack((forces[x]*scale[:,None],np.zeros((c.shape[0],3)))),assume_a='sym',lower=True)
    shell_end=6*inputs['layout']['counts']['shell_nodes']
    roles=dict(shell=x[x<shell_end],beam_nodes=x[(x>=shell_end)&(x<base)],
               cell_rotations=x[x>=base])
    assert all(len(slots)>0 for slots in roles.values())
    results={};native_states=[]
    for j,case in enumerate(cases):
        native=np.zeros(k.shape[0]);native[free]=lifted@(reduced[:len(x),j]*scale)
        multipliers=reduced[len(x):,j]/row_norm
        direct=np.zeros_like(native);direct[free]=solution[:len(free),j];lm=solution[len(free):,j]
        residual=k@native+c.T@multipliers-forces[:,j]
        support=np.zeros_like(native);support[fixed]=residual[fixed]
        direct_residual=k@direct+c.T@lm-forces[:,j]
        direct_support=np.zeros_like(native);direct_support[fixed]=direct_residual[fixed]
        parity=dict(physical=relative(native[x]/coordinate[x],direct[x]/coordinate[x]),
            resultants=relative(native[pp]/coordinate[pp],direct[pp]/coordinate[pp]),
            multipliers=relative(multipliers*constraint,lm*constraint),
            support=relative(support*coordinate,direct_support*coordinate))
        parity.update({name:relative(native[slots]/coordinate[slots],
                                    direct[slots]/coordinate[slots])
                       for name,slots in roles.items()})
        action=c.T@multipliers
        resisting=np.vstack((shell_work@action,beam_work@action))
        results[case]=dict(parity=parity,equilibrium=_linear_equilibrium(residual[free],forces[free,j],coordinate[free]),
            weak_constraint=_scaled_constraint_gap(c@native,constraint),
            energy_work_relative=relative(np.array(native@(k@native)),np.array(native@forces[:,j])),
            work_J=float(native@forces[:,j]),interface_resisting_wrenches=resisting.tolist(),
            interface_action_balance=float(np.max(np.abs(resisting.sum(axis=0)))))
        native_states.append(dict(q=native,load=forces[:,j],multipliers=multipliers,reactions=support,
            direct_q=direct,direct_multipliers=lm,direct_reactions=direct_support))
    predicates=dict(kernel_positive=kernel_positive,saddle_inertia=inertia==expected,
        solution_parity=all(max(v['parity'].values())<=1e-10 for v in results.values()),
        equilibrium=all(v['equilibrium']<=1e-8 for v in results.values()),
        weak_constraints=all(v['weak_constraint']<=1e-9 for v in results.values()),
        energy_work=all(v['energy_work_relative']<=1e-8 for v in results.values()),
        action_balance=all(v['interface_action_balance']<=1e-8 for v in results.values()))
    arrays=dict(physical_slots=x,resultant_slots=pp,physical_scale=scale,constraint_row_scale=row_norm,
        kernel_values=ev,kernel_basis=z,kernel_vectors=vectors,saddle_values=saddle_ev)
    arrays.update({case+'_'+key:value for case,item in zip(cases,native_states) for key,value in item.items()})
    for name,matrix in (('C',c),('S',s),('lift',lifted)):
        arrays.update({name+'_data':matrix.data,name+'_indices':matrix.indices,name+'_indptr':matrix.indptr,name+'_shape':matrix.shape})
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    source_files={name:sha(root/'native'/name) for name in ('inputs.json','operators.npz')}
    source_files.update({case+'/raw-state.npz':sha(root/'native'/case/'raw-state.npz') for case in cases})
    return dict(status='COMPLETED_DEVELOPMENT_ONLY' if all(predicates.values()) else 'UNRESOLVED_DEVELOPMENT_CHECK',
        scope='one saved coarse virgin native mixed-system discriminator; no material-history parity, growing-space, locking, accuracy or F03 acceptance',
        candidate=IDENTITY,source_sha256=sha(__file__),source_files=source_files,
        helpers={name:sha(Path(__file__).parent/name) for name in ('b3_e4_bl5_mechanical_kernel_dev.py','r8_bl5_projected_density_patch_dev.py')},
        anyfesolver_import=anyfesolver.__file__,numba_threads=numba.get_num_threads(),jit_disabled=bool(numba.config.DISABLE_JIT),
        pools=threadpool_info(),physical_order=len(x),native_supported_order=len(free),constraint_rows=c.shape[0],
        original_constraint_parity=original_parity,local_blocks=local_info,
        work_pullback_relative=work_parity,constraint_pullback_relative=constraint_parity,
        constraint_min_singular=float(singular[-1]),rank_budget=float(rank_budget),minimum_kernel_eigenvalue=float(ev[0]),
        spectral_uncertainty_terms=terms,spectral_uncertainty=uncertainty,reduced_saddle_inertia=inertia,
        expected_reduced_saddle_inertia=expected,saddle_budget=saddle_budget,
        implied_native_inertia=dict(positive=len(x),negative=len(pp)+c.shape[0],unresolved=0) if all(predicates.values()) else None,
        full_general_factorizations=1,reduced_factorizations=1,loads=results,predicates=predicates,elapsed_seconds=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    assert numba.get_num_threads()==1 and numba.config.DISABLE_JIT
    assert all(v['num_threads']==1 for v in threadpool_info())
    r=evaluate(a.input,a.output);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r[k] for k in ('status','minimum_kernel_eigenvalue','spectral_uncertainty','reduced_saddle_inertia','elapsed_seconds')}))
    raise SystemExit(0 if all(r['predicates'].values()) else 1)
