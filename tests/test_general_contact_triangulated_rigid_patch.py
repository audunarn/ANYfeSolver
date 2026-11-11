"""Independent P1 patch law and actual ordinary static/restart controls."""
import itertools
import json
import os
from pathlib import Path
import numpy as np
import pytest
from examples.general_contact_triangulated_rigid_patch import case,analysis
from anyfesolver import (BoundaryCondition,ContactPathControl,ContactPathProgram,solve_static_nonlinear)
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.general_contact.triangulated_rigid_static import (TriangulatedRigidManager,
    PATCH_PROFILE,_storage_guard,derivative_work_storage_bound,LOCAL_DERIVATIVE_WORK_LIMIT_BYTES,_source_overlap_guard)


def independent_mass(model):
    m=np.zeros((4,4));s=np.zeros(4);bending=np.zeros((4,4));areas=[];slopes=[]
    for ids in ((0,1,2),(1,3,2)):
        xy=np.array([(model.mesh.nodes[i+1].x,model.mesh.nodes[i+1].y) for i in ids])
        area=abs(np.linalg.det(np.array([xy[1]-xy[0],xy[2]-xy[0]])))/2
        local=area/12*(np.eye(3)+np.ones((3,3)))
        m[np.ix_(ids,ids)]+=local;s[list(ids)]+=area/3
        b=np.linalg.inv(np.column_stack((np.ones(3),xy)))[1:,:]
        bending[np.ix_(ids,ids)]+=area*b.T@b
        areas.append(area);slopes.append((ids,b))
    return m,s,bending,areas,slopes


def test_shared_pressure_cross_face_work_and_storage():
    model,definition,settings=case();manager=TriangulatedRigidManager(model,definition,settings)
    assert manager.profile==PATCH_PROFILE and len(model.mesh.nodes)==4
    source,target=manager._positions(np.zeros(manager.n_dof));source[:,2]=-2e-5
    result=manager.law.work(source,target)
    m,s,b,areas,_=independent_mass(model);area=sum(areas)
    assert abs(result.state.energy-.5*1e6*area*(2e-5)**2)<1e-15
    np.testing.assert_allclose(result.state.reference_support,s,rtol=1e-12,atol=1e-15)
    # Shared pressures enforce once after summing both source-face integrals.
    normal=1e6*m.T@np.diag(1/s)@m+1e6*(2e-5)**2*b
    z=(2,5,8,11)
    np.testing.assert_allclose(result.tangent[np.ix_(z,z)],normal,rtol=1e-12,atol=1e-8)
    assert abs(normal[0,3])>0  # Cross-face coupling through shared pressure nodes.
    v=np.zeros_like(result.residual);v[list(z)]=(.1,-.2,.3,.4);v[0]=.1
    coords=np.r_[source.reshape(-1),target.reshape(-1)];h=1e-7
    work=[]
    for sign in (-1,1):
        x=coords+sign*h*v;work.append(manager.law.work(x[:12].reshape(4,3),x[12:].reshape(-1,3)))
    assert abs((work[1].state.energy-work[0].state.energy)/(2*h)-result.residual@v)/max(1.,abs(result.residual@v))<1e-5
    assert np.linalg.norm((work[1].residual-work[0].residual)/(2*h)-result.tangent@v)/max(1.,np.linalg.norm(result.tangent@v))<1e-5
    assert derivative_work_storage_bound(4,24,4)<LOCAL_DERIVATIVE_WORK_LIMIT_BYTES
    with pytest.raises(ValueError,match='before dense allocation'): _storage_guard(200,1200,256)


def test_patch_force_response_and_lifecycle_restart(monkeypatch):
    model,definition,settings=case();m,s,_,areas,slopes=independent_mass(model)
    baseline,_=assemble_stiffness_matrix(model)
    dofs=[model.mesh.dof_manager.get_node_dofs(i)[2] for i in (1,2,3,4)]
    k=baseline[np.ix_(dofs,dofs)].toarray();free=(1,2,3);load=np.array((0.,-.05,-.05,-.05));gap=np.ones(4)*1e-5
    references=[]
    for mask in itertools.product((False,True),repeat=4):
        active=np.array(mask);ma=m[active];sa=s[active]
        contact=1e6*ma.T@np.diag(1/sa)@ma if any(active) else np.zeros((4,4))
        u=np.zeros(4);u[list(free)]=np.linalg.solve((k+contact)[np.ix_(free,free)],(load-contact@gap)[list(free)])
        weighted=m@(gap+u)
        if np.all(weighted[active]<0) and np.all(weighted[~active]>=0):references.append((u,active,contact))
    assert len(references)==1
    reference,active,contact=references[0]
    import importlib
    solver=importlib.import_module('anyfesolver.nonlinear_static');original=solver._support_reaction_resultants;captured={}
    def reaction(owner_model,imbalance,**kwargs):
        captured.update(model=owner_model,residual=np.array(imbalance,copy=True))
        return original(owner_model,imbalance,**kwargs)
    monkeypatch.setattr(solver,'_support_reaction_resultants',reaction)
    full=analysis();monkeypatch.setattr(solver,'_support_reaction_resultants',original)
    assert full.status=='completed',full.failure_reason
    actual=full.displacements[dofs]
    scale=1e-5  # Declared initial physical clearance; fixed before solving.
    error=float(np.linalg.norm(actual-reference)/scale)
    assert error<1e-6
    # Leading-order correction estimate; not a uniform rigorous enclosure.
    # The independent measured response gate above remains the required1e-6 check.
    max_slope=max(np.linalg.norm(b@reference[list(ids)]) for ids,b in slopes)
    wmax=max(abs(gap+reference));u_norm=np.linalg.norm(reference)
    bnorm=max(np.linalg.norm(b,2) for _,b in slopes)
    angle_force=4e6*sum(areas)*(wmax*max_slope**2+wmax**2*bnorm*max_slope)
    membrane_force=4*1e5*.01/(1-.3**2)*sum(areas)*bnorm**4*u_norm**3
    eig=np.linalg.eigvalsh((k+contact)[np.ix_(free,free)])[0]
    displacement_estimate=(angle_force+membrane_force)/eig
    assert displacement_estimate/scale<1e-6
    certificate=full.info['general_contact'];assert certificate['gap_tolerance_met'] and certificate['active_pressure_dofs']>0
    assert np.linalg.norm(np.array(certificate['integration_force'])+certificate['target_force'])/.15<1e-6
    assert np.linalg.norm(np.array(certificate['integration_moment'])+certificate['target_moment'])/.15<1e-6
    support_force=np.zeros(3);support_moment=np.zeros(3)
    for node in (1,2,3,4):
        owner=captured['model'];ids=owner.mesh.dof_manager.get_node_dofs(node)
        position=np.array((owner.mesh.nodes[node].x,owner.mesh.nodes[node].y,owner.mesh.nodes[node].z))+full.displacements[ids[:3]]
        residual=captured['residual'][ids].copy()
        if node!=1:residual[2]=0.
        support_force+=residual[:3];support_moment+=residual[3:]+np.cross(position,residual[:3])
        if node!=1:
            declared_load=np.array((0.,0.,-.05));support_force+=declared_load;support_moment+=np.cross(position,declared_load)
    support_force+=certificate['integration_force'];support_moment+=certificate['integration_moment']
    equilibrium={'force_error':float(np.linalg.norm(support_force)/.15),'moment_error':float(np.linalg.norm(support_moment)/.15)}
    assert equilibrium['force_error']<1e-6 and equilibrium['moment_error']<1e-6
    model,definition,settings=case()
    model.boundary_conditions[0]=BoundaryCondition('guide',(1,2,3,4),{k:0. for k in ('uy','rx','ry','rz')})
    model.add_boundary_condition(BoundaryCondition('xguide',(1,3,4),{'ux':0.}))
    program=ContactPathProgram(controls=(ContactPathControl.node(2,'ux'),ContactPathControl.node(2,'uz')),
        waypoints=((0.,0.),(0.,-4e-5),(0.,0.),(0.,-4e-5)),steps_per_segment=1)
    common=dict(contact=definition,contact_settings=settings,contact_path_program=program,max_iterations=15,emit_restart_checkpoint=True)
    path=solve_static_nonlinear(model,**common);assert path.status=='completed',path.failure_reason
    phases=path.info['force_displacement_history'];assert [p['contact']['active_pressure_dofs']>0 for p in phases]==[True,False,True]
    assert all(p['contact']['gap_tolerance_met'] for p in phases)
    half=solve_static_nonlinear(model,max_load_factor=1/3,**common);assert half.status=='completed',half.failure_reason
    resumed=solve_static_nonlinear(model,restart_checkpoint=half.to_restart_checkpoint(),**common)
    assert resumed.status=='completed',resumed.failure_reason
    assert json.dumps(path.to_restart_checkpoint(),sort_keys=True)==json.dumps(resumed.to_restart_checkpoint(),sort_keys=True)
    if os.environ.get('TRIANGULATED_RIGID_RESULT'):
        Path(os.environ['TRIANGULATED_RIGID_RESULT']).write_text(json.dumps({'force':certificate,'reference_u_m':reference.tolist(),'actual_u_m':actual.tolist(),
            'relative_response_error':error,'nonlinear_correction_estimate_m':displacement_estimate,'estimate_relative_to_gap':displacement_estimate/scale,
            'phases':phases,'exact_checkpoint_equal':True,'support_load_contact_equilibrium':equilibrium},indent=2,default=lambda v:v.item()))


def test_source_ownership_and_trace_refusal():
    with pytest.raises(ValueError,match='overlapping'):
        _source_overlap_guard(np.array(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(.1,.1,0.),(.9,.1,0.),(.1,.9,0.))),((0,1,2),(3,4,5)))
    model,definition,settings=case()
    model.mesh.nodes[4].x=.25;model.mesh.nodes[4].y=.3
    with pytest.raises(ValueError):TriangulatedRigidManager(model,definition,settings)
    manager=TriangulatedRigidManager(*case())
    u=np.zeros(manager.n_dof);v=u.copy()
    for node in (1,2,3,4):u[manager.model.mesh.dof_manager.get_node_dofs(node)[2]]=-3e-5
    shared=manager.model.mesh.dof_manager.get_node_dofs(2);v[shared[3]]=1.;v[shared[2]]=.1
    assert np.asarray(manager.map@v).reshape(-1,3)[1,1]==.005
    f,k,_=manager.evaluate(u);h=1e-7
    fp,_,rp=manager.evaluate(u+h*v);fm,_,rm=manager.evaluate(u-h*v)
    assert abs((rp.penalty_energy-rm.penalty_energy)/(2*h)-f@v)/max(1.,abs(f@v))<1e-5
    assert np.linalg.norm((fp-fm)/(2*h)-k@v)/max(1.,np.linalg.norm(k@v))<1e-5


def test_shared_trace_near_mismatch_refuses():
    model,definition,settings=case()
    model.mesh.elements[2].thickness=.010000000000000002
    with pytest.raises(ValueError,match='exactly'):TriangulatedRigidManager(model,definition,settings)
