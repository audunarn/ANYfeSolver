import importlib.util,pathlib
import numpy as np
import pytest
from anyfesolver import solve_static_nonlinear
from anyfesolver._plate_stability_continuation import trust_steps,segment_work,admit
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
spec=importlib.util.spec_from_file_location('plate_case',pathlib.Path(__file__).parents[1]/'scripts/investigate_plate_edge_shear.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

@pytest.mark.parametrize('gradient',[np.zeros(3),np.array([1e-15,-2e-15,.1]),np.array([1e-30,-2e-30,.1]),np.array([.2,-.4,.3])])
def test_indefinite_repeated_low_space_trust_step(gradient):
    values=np.array([-2.,-2.,3.]);steps=trust_steps(values,np.eye(3),gradient,.2)
    for step in steps:
        assert np.linalg.norm(step)<=.2*(1+1e-12)
        assert np.linalg.norm(step)>=.2*(1-1e-12)
        assert gradient@step+.5*step@(values*step)<0
    if np.any(gradient[:2]):assert len(steps)==1

def test_nonzero_endpoint_energy_work_gradient_reversal_additivity():
    m=fixture.plate(.01,True);rng=np.random.default_rng(27);size=m.mesh.dof_manager.total_dofs
    a=.002*rng.standard_normal(size);b=.003*rng.standard_normal(size);c=.001*rng.standard_normal(size)
    check=lambda:None
    ab,_=segment_work(m,a,b,check);ba,_=segment_work(m,b,a,check);bc,_=segment_work(m,b,c,check);ac,_=segment_work(m,a,c,check)
    assert abs(ab+ba)<1e-10*max(abs(ab),1)
    assert abs(ab+bc-ac)<1e-10*max(abs(ab)+abs(bc),1)
    d=rng.standard_normal(size);d/=np.linalg.norm(d);h=1e-6
    work,_=segment_work(m,a-h*d,a+h*d,check)
    force,_,_=_assemble_nonlinear_system(m,a,{},5,tangent=False,require_full_coordinates=True,kinematics='corotational',corotational_tangent='work_conjugate')
    assert abs(work/(2*h)-force@d)<1e-6*max(abs(force@d),1)

def test_stable_tiny_public_route():
    m=fixture.plate(1e-6,True)
    result=solve_static_nonlinear(m,None,num_steps=2,kinematics='corotational',corotational_tangent='work_conjugate',stability_control='plate_energy',record_increment_snapshots=True)
    assert result.converged
    d=result.info['stability_control'];assert d['accepted_curvatures']
    assert all(c['minimum']>c['threshold'] for c in d['accepted_curvatures'])

@pytest.mark.parametrize('change',['yield','mode','load'])
def test_unsupported_route_refuses(change):
    m=fixture.plate(.01,True);kwargs=dict(kinematics='corotational',corotational_tangent='work_conjugate',stability_control='plate_energy')
    if change=='yield':m.get_material('ideal-elastic').yield_stress=1.
    if change=='mode':kwargs['corotational_tangent']='consistent'
    if change=='load':kwargs['load_case']=object()
    with pytest.raises(ValueError):solve_static_nonlinear(m,**kwargs)

@pytest.mark.parametrize('where',['status','token'])
def test_hostile_observation_cannot_change_admitted_policy(where):
    m=fixture.plate(1e-6,True)
    kwargs={}
    if where=='status':
        def status(message):m.get_material('ideal-elastic').yield_stress=1.
        kwargs['status_callback']=status
    else:
        class Token:
            def raise_if_cancelled(self,stage):
                if stage=='nonlinear_static.plate_energy':m.boundary_conditions[1].dof_constraints['uy']*=2
        kwargs['cancellation_token']=Token()
    with pytest.raises((ValueError,RuntimeError)):
        solve_static_nonlinear(m,None,kinematics='corotational',corotational_tangent='work_conjugate',stability_control='plate_energy',**kwargs)

def test_cancellation_during_work_propagates_and_preserves_reference_geometry():
    from anyfesolver.control import SolveCancelled,CancellationToken
    m=fixture.plate(.01,True);original=[(n.x,n.y,n.z) for n in m.mesh.nodes.values()]
    class Token(CancellationToken):
        calls=0
        def raise_if_cancelled(self,stage=''):
            if stage=='nonlinear_static.plate_energy':
                self.calls+=1
                if self.calls==8:raise SolveCancelled('during energy')
    with pytest.raises(SolveCancelled,match='during energy'):
        solve_static_nonlinear(m,None,num_steps=1,kinematics='corotational',corotational_tangent='work_conjugate',stability_control='plate_energy',cancellation_token=Token())
    assert original==[(n.x,n.y,n.z) for n in m.mesh.nodes.values()]


def test_exhausted_trial_returns_failure_without_committing_unstable_state():
    m=fixture.plate(.01,True)
    r=solve_static_nonlinear(m,None,num_steps=1,max_iterations=1,min_step_fraction=1.,kinematics='corotational',corotational_tangent='work_conjugate',stability_control='plate_energy',record_increment_snapshots=True)
    assert not r.converged and not r.steps and not r.snapshots
    assert not r.info['stability_control']['accepted_curvatures']

def test_large_mesh_refuses_before_solving():
    m=fixture.plate(.01,True,16)
    with pytest.raises(ValueError,match='512'):admit(m)


@pytest.mark.parametrize('binding',['trust_steps','eigh','np','math'])
def test_callback_cannot_replace_stability_algorithm(monkeypatch,binding):
    import anyfesolver._plate_stability_continuation as helper
    m=fixture.plate(1e-6,True)
    def status(message):monkeypatch.setattr(helper,binding,lambda *a: [np.zeros(1)])
    with pytest.raises(ValueError,match='authority changed'):
        solve_static_nonlinear(m,None,kinematics='corotational',corotational_tangent='work_conjugate',stability_control='plate_energy',status_callback=status)


def test_callback_cannot_replace_algorithm_code(monkeypatch):
    import anyfesolver._plate_stability_continuation as helper
    m=fixture.plate(1e-6,True)
    def status(message):
        monkeypatch.setattr(helper.trust_steps,'__code__',(lambda *args: []). __code__)
    with pytest.raises(ValueError,match='authority changed'):
        solve_static_nonlinear(m,None,kinematics='corotational',corotational_tangent='work_conjugate',stability_control='plate_energy',status_callback=status)


@pytest.mark.parametrize('amplitude,delta_scale',[(.002,None),(.01,None),(.1,None),(.01,1e-9)])
def test_two_point_local_work_matches_three_point_reference(amplitude,delta_scale):
    import math
    from anyfesolver.corotational import _corotational_cache,_corotational_deformation_state
    m=fixture.plate(.01,True);rng=np.random.default_rng(918)
    size=m.mesh.dof_manager.total_dofs
    old=amplitude*rng.standard_normal(size);new=amplitude*rng.standard_normal(size)
    if delta_scale is not None:
        new=old+delta_scale*rng.standard_normal(size)
    value,_=segment_work(m,old,new,lambda:None)
    refs=_corotational_cache(m);points,weights=np.polynomial.legendre.leggauss(3)
    terms=[]
    for eid,element in m.mesh.elements.items():
        dofs=element.get_dof_mapping(m.mesh)
        a,_,_=_corotational_deformation_state(refs[eid],element,old[dofs],stable_shell_rotation=True)
        b,_,_=_corotational_deformation_state(refs[eid],element,new[dofs],stable_shell_rotation=True)
        delta=b-a
        for point,weight in zip(points,weights):
            f,_,_=element.compute_nonlinear_response(m.mesh,m.get_material(element.material_name),a+.5*(point+1)*delta,None,5,False)
            terms.append(.5*weight*float(delta@f))
    reference=math.fsum(terms)
    assert abs(value-reference)<=256*np.finfo(float).eps*max(math.fsum(abs(v) for v in terms),1.)
