"""Independent attachment/affine static oracle, explicitly bounded harness only."""
from dataclasses import replace,asdict
from pathlib import Path
import json,os
import numpy as np
import pytest
from scipy import sparse,linalg
from anyfesolver import stiffened_shell as api
from examples.stiffened_shell_b2_physical_rigid_footprint import definition

pytestmark=pytest.mark.skipif('B2_FOUNDATION_OUTPUT' not in os.environ or 'B2_FOUNDATION_SUPERVISED' not in os.environ,
    reason='Requires explicitly bounded B2 static-foundation harness')
SELECT=api.B2PhysicalRigidFootprintSelection(('physical-left',))

def save(name,metrics,**arrays):
    out=Path(os.environ['B2_FOUNDATION_OUTPUT']);(out/(name+'.json')).write_text(json.dumps(metrics,indent=2)+'\n')
    if arrays:np.savez(out/(name+'.npz'),**arrays)

def independent_rows(d):
    """Apply physical translational/director variations to each Cartesian basis."""
    shell_ids=sorted(d.shell_nodes_m);positions={}
    for member in d.members:
        positions.update(zip(member.node_ids,member.reference_points_m))
    beam_ids=sorted(positions);rows=[];inventory=[];local=[]
    for group in d.physical_footprints:
        n=np.asarray(group.reference_director);e=np.asarray(group.tangent_basis)
        for node in group.shell_nodes:
            xs=np.asarray(d.shell_nodes_m[node]);xb=np.asarray(positions[group.beam_node]);arm=xs-xb
            s=6*shell_ids.index(node);b=6*len(shell_ids)+6*beam_ids.index(group.beam_node)
            block=np.zeros((5,54))
            for j in range(54):
                v=np.eye(54)[j];us,ws=v[s:s+3],v[s+3:s+6];ub,wb=v[b:b+3],v[b+3:b+6]
                block[:,j]=np.r_[us-ub-np.cross(wb,arm),e.T@np.cross(ws-wb,n)]
            local.append((node,group.beam_node,s,b,arm,n,e));start=len(rows)
            rows.extend(block)
            inventory.extend((start+j,group.footprint_id,node,group.beam_node,c) for j,c in enumerate(('translation_x','translation_y','translation_z','director_1','director_2')))
    return np.asarray(rows),tuple(inventory),local

def case_definition(kind):
    d=definition();u0=np.zeros(54)
    if kind=='coordinate':return d
    # Genuine affine relation outside pad: u_z(node3)=.25*u_x(node3)+.0003m.
    free=[i for i in range(54) if i not in (*range(36,42),14)]
    T=sparse.eye(54,format='lil')[:,free];T[14,free.index(12)]=.25
    u0[[37,41,14]]=(.001,.0002,.0003)
    return replace(d,transformation=T.tocsr(),prescribed_offset=u0)

def load(second=False):
    f=np.zeros(54);f[[12,13,14,30,48,53]]=(.4,.7,.2,-.3,.25,.03)
    if second:f[[12,13,53]]=(-.2,.1,-.05)
    return f

def oracle(owner,d,f):
    C,inventory,local=independent_rows(d);K=owner._K.toarray();T=d.transformation.toarray();u0=np.asarray(d.prescribed_offset)
    A=np.block([[T.T@K@T,T.T@C.T],[C@T,np.zeros((len(C),len(C)))]])
    rhs=np.r_[T.T@(f-K@u0),-C@u0]
    # Congruence equilibration only inside this independent tiny-model oracle.
    scale=1/np.sqrt(np.max(abs(A),axis=1));answer=scale*linalg.solve(scale[:,None]*A*scale[None,:],scale*rhs,assume_a='gen')
    q=answer[:-len(C)];lam=answer[-len(C):];u=T@q+u0;r=K@u+C.T@lam-f
    return dict(C=C,K=K,T=T,u0=u0,A=A,rhs=rhs,q=q,lam=lam,u=u,reaction=r,inventory=inventory,local=local)

def block_error(a,b,parts):
    return max(float(np.max(abs(a[ix]-b[ix]))/max(np.max(abs(a[ix])),np.max(abs(b[ix])),np.finfo(float).tiny)) for ix in parts)

def verify(name,owner,d,f,result,reference):
    o=reference
    save(name+'-public',dict(status=result.status,failure_reason=getattr(result,'failure_reason',None)))
    assert result.status=='completed',getattr(result,'failure_reason',None)
    u=np.asarray(result.displacements);lam=np.asarray(result.constraint_multipliers);reaction=np.asarray(result.affine_reactions)
    force=o['K']@u;transfer=o['C'].T@lam;gap=o['C']@u;projected=o['T'].T@(force+transfer-f)
    physicalparts=[np.array([i for i in range(54) if i%6<3]),np.array([i for i in range(54) if i%6>=3])]
    dualparts=[np.array([i for i in range(20) if i%5<3]),np.array([i for i in range(20) if i%5>=3])]
    equilibrium=[]
    for ix in physicalparts:
        bounds=abs(o['K'])@abs(u)+abs(o['C'].T)@abs(lam)+abs(f)+abs(reaction)
        equilibrium.append(float(np.max(abs((force+transfer-f-reaction)[ix]))/max(np.max(bounds[ix]),np.finfo(float).tiny)))
    strain=float(u@force);external=float(f@u);support=float(o['u0']@reaction)
    dual_reduced=float(lam@(o['C']@(u-o['u0'])));dual_offset=float(lam@(o['C']@o['u0']))
    reference_dual_reduced=float(o['lam']@(o['C']@o['T']@o['q']))
    reference_dual_offset=float(o['lam']@(o['C']@o['u0']))
    # Every admitted fixture column has one physical dual unit. A general
    # translation/rotation mixture needs declared scales, absent in this oracle.
    reduced_parts=[[],[]]
    for j in range(o['T'].shape[1]):
        touched=np.flatnonzero(o['T'][:,j]);units={int(i%6>=3) for i in touched}
        if len(units)!=1:raise ValueError('independent oracle requires declared scales for mixed-unit affine columns')
        reduced_parts[units.pop()].append(j)
    reduced_bound=abs(o['T'].T)@(abs(force)+abs(transfer)+abs(f))
    reduced_errors=[float(np.max(abs(projected[ix]))/max(np.max(reduced_bound[ix]),np.finfo(float).tiny)) for ix in reduced_parts]
    virtual=np.linspace(-.2,.3,o['T'].shape[1]);v=o['T']@virtual
    virtual_gap=float(lam@(o['C']@v));virtual_action=float(v@transfer)
    metrics=dict(C_error=float(np.max(abs(owner._C.toarray()-o['C']))),row_inventory_equal=tuple(result.constraint_rows)==o['inventory'],
        displacement_parity=block_error(u,o['u'],physicalparts),multiplier_parity=block_error(lam,o['lam'],dualparts),reaction_parity=block_error(reaction,o['reaction'],physicalparts),
        full_physical_equilibrium_by_force_moment=equilibrium,reduced_equilibrium_by_force_moment=reduced_errors,reduced_columns_by_translation_rotation=reduced_parts,
        translation_gap=float(np.max(abs(gap.reshape(-1,5)[:,:3]))),director_gap=float(np.max(abs(gap.reshape(-1,5)[:,3:]))),
        strain_work=strain,external_work=external,prescribed_affine_dual_work=support,work_error=abs(strain-external-support)/max(abs(strain),abs(external),abs(support),np.finfo(float).tiny),
        multiplier_reduced_work=dual_reduced,multiplier_prescribed_work=dual_offset,multiplier_affine_cancellation=abs(dual_reduced+dual_offset)/max(abs(dual_reduced),abs(dual_offset),np.finfo(float).tiny),
        reference_multiplier_reduced_work=reference_dual_reduced,reference_multiplier_prescribed_work=reference_dual_offset,
        constraint_virtual_work_error=abs(virtual_gap-virtual_action)/max(abs(virtual_gap),abs(virtual_action),np.finfo(float).tiny),internal_energy_error=abs(2*result.internal_strain_energy-strain)/max(abs(strain),np.finfo(float).tiny))
    save(name,metrics,**{k:o[k] for k in ('C','K','T','u0','A','rhs','q')},reference_u=o['u'],reference_lam=o['lam'],reference_reaction=o['reaction'],u=u,lam=lam,reaction=reaction,force=force,transfer=transfer,gap=gap,load=f)
    assert metrics['C_error']<=1e-12 and metrics['row_inventory_equal']
    assert max(metrics[k] for k in ('displacement_parity','multiplier_parity','reaction_parity'))<=1e-10,metrics
    assert max(equilibrium+reduced_errors+[metrics['work_error'],metrics['constraint_virtual_work_error'],metrics['internal_energy_error']])<=1e-8,metrics
    assert max(metrics['translation_gap'],metrics['director_gap'])<=1e-9
    if np.any(o['u0']):
        assert abs(support)>1e-6 and abs(dual_offset)>1e-6,'prescribed and constraint dual work must be exercised'
        assert metrics['multiplier_affine_cancellation']<=1e-8
    return metrics

def test_independent_physical_rows_and_work():
    d=definition();owner=api.create_analysis(definition=d)
    try:
        C,inventory,local=independent_rows(d);v=np.linspace(-.3,.4,54);lam=np.linspace(.2,1.3,20)
        # Direct physical force/moment balance, independently assembled by arms.
        physical=np.zeros(54)
        for i,(_,_,s,b,arm,n,e) in enumerate(local):
            f=lam[5*i:5*i+3];m=lam[5*i+3:5*i+5]
            moment=np.cross(n,e@m);physical[s:s+3]+=f;physical[b:b+3]-=f
            physical[s+3:s+6]+=moment;physical[b+3:b+6]-=moment+np.cross(arm,f)
        spin=np.zeros(54)
        for _,_,s,_,_,n,_ in local:spin[s+3:s+6]=n
        metric=dict(C_error=float(np.max(abs(owner._C.toarray()-C))),transpose_action_error=float(np.max(abs(C.T@lam-physical))),
            virtual_work_error=abs(float(lam@(C@v)-v@physical)),normal_spin_gap=float(np.max(abs(C@spin))))
        save('rows-work',metric,C=C,direction=v,multipliers=lam,physical_action=physical,spin=spin)
        assert max(metric.values())<=1e-12
        altered=C.copy();altered[0,39:42]=0
        assert np.max(abs(altered-C))>.1,'omitted physical beam arm must be distinguishable'
        assert np.max(abs(C.T@(-lam)-physical))>.1,'multiplier sign error must be distinguishable'
        # Unit counterexample: a quiet moment defect cannot borrow a large
        # independent force scale. This old mixed maximum incorrectly passes.
        residual=np.array((0.,1e-6));force_moment_bounds=np.array((1e6,1e-6))
        old_mixed=float(np.max(abs(residual))/np.max(force_moment_bounds))
        separate=float(np.max(abs(residual)/force_moment_bounds))
        save('unit-masking-control',dict(old_mixed_error=old_mixed,separate_error=separate,limit=1e-8))
        assert old_mixed<=1e-8 and separate>1e-8
    finally:owner.close()

@pytest.mark.parametrize('kind',('coordinate','affine-offset'))
def test_independent_affine_kkt_public_static(kind):
    d=case_definition(kind);owner=api.create_analysis(definition=d)
    try:
        f=load();o=oracle(owner,d,f);result=owner.solve_linear(f,selected_footprints=SELECT)
        verify(kind,owner,d,f,result,o)
        if kind=='coordinate':
            # Reuse the same public state; this is a synthetic assessor refusal,
            # not a new physical model/solve or a modified admitted definition.
            mixed=dict(o);mixed['T']=o['T'].copy();mixed['T'][3,0]=.1
            with pytest.raises(ValueError,match='declared scales for mixed-unit'):
                verify('mixed-unit-refusal',owner,d,f,result,mixed)
            save('mixed-unit-refusal-control',dict(status='expected refusal',translation_row=0,rotation_row=3,column=0))
    finally:owner.close()

def test_independent_affine_offset_public_batch():
    d=case_definition('affine-offset');owner=api.create_analysis(definition=d)
    try:
        loads=(load(),load(True));refs=[oracle(owner,d,f) for f in loads]
        result=owner.solve_linear_many(tuple((f'case-{i}',f) for i,f in enumerate(loads)),selected_dofs=tuple(range(54)),selected_footprints=SELECT)
        save('batch-public',dict(status=result.status,failure_reason=result.failure_reason,cases=len(result.cases),factorization_count=result.factorization_count,solve_count=result.solve_count))
        assert result.status=='completed' and len(result.cases)==2
        for i,(f,o,case) in enumerate(zip(loads,refs,result.cases)):
            # Selected public batch returns detached full selected coordinates.
            class Full:pass
            full=Full();full.status='completed';full.displacements=case.selected_displacements;full.affine_reactions=case.selected_affine_reactions
            full.constraint_multipliers=case.constraint_multipliers;full.constraint_rows=result.constraint_rows;full.internal_strain_energy=case.internal_strain_energy
            verify(f'batch-{i}',owner,d,f,full,o)
        assert result.factorization_count==1
    finally:owner.close()
