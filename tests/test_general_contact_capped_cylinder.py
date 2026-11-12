"""Independent finite-cylinder distance/work checks and integrated line owner."""
from dataclasses import replace
import json
import os
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad

from anyfesolver import (RigidCappedCylinder, validate_general_contact, solve_static_nonlinear,
    BoundaryCondition, DisplacementControl, LoadCase)
from anyfesolver.general_contact.capped_cylinder import (signed_distance_work,_frame,
    curve_gap_lower,CappedCylinderBeamManager,signed_distance_value)
from test_general_contact_beam_radial import _model,_trial


def _case(quadratic=False):
    model,definition,settings=_model(kind='cylinder',quadratic=quadratic)
    body=RigidCappedCylinder('obstacle',(0.,0.,0.),(1.,0.,0.),.01,.1)
    return model,replace(definition,rigid_bodies=(body,)),settings


def _oracle(point,R=.01,h=.1):
    # Independent closest point on the closed solid, with interior face distance.
    axial=float(point[0]); rho=np.linalg.norm(point[1:])
    if abs(axial)<=h and rho<=R: return -min(h-abs(axial),R-rho)
    nearest=np.array(point,dtype=float)
    nearest[0]=np.clip(axial,-h,h)
    if rho>R: nearest[1:]*=R/rho
    return np.linalg.norm(np.asarray(point)-nearest)


@pytest.mark.parametrize('point',[(0.,.03,.01),(.13,.003,.001),(.13,.03,.01),(.095,.001,.001)])
def test_analytic_distance_virtual_work_and_hessian(point):
    frame=_frame((1.,0.,0.)); point=np.array(point)
    d,g,H=signed_distance_work(point,np.zeros(3),frame,.01,.1)
    assert d==pytest.approx(_oracle(point),abs=1e-15)
    direction=np.array([.3,-.4,.2])
    for step in (1e-6,3e-7,1e-7):
        plus=signed_distance_work(point+step*direction,np.zeros(3),frame,.01,.1)
        minus=signed_distance_work(point-step*direction,np.zeros(3),frame,.01,.1)
        assert (plus[0]-minus[0])/(2*step)==pytest.approx(g@direction,rel=1e-6,abs=1e-9)
        np.testing.assert_allclose((plus[1]-minus[1])/(2*step),H@direction,rtol=1e-5,atol=1e-8)


@pytest.mark.parametrize('point,R,h',[((0.,0.,0.),.01,.1),((0.,.01,0.),.1,.005),((.4375,.0625,0.),.125,.5)])
def test_medial_or_equal_distance_configurations_reject(point,R,h):
    with pytest.raises(ValueError,match='nonsmooth|undefined'):
        signed_distance_work(np.array(point),np.zeros(3),_frame((1.,0.,0.)),R,h)


def test_quadratic_full_curve_bound_interior_and_budget():
    positions=np.array([[-.3,0.,.04],[0.,0.,.018],[.3,0.,.04]])
    frame=_frame((1.,0.,0.))
    lower=curve_gap_lower(positions,np.zeros(3),frame,.01,.1,.01,accuracy=1e-6,maximum_depth=14)
    grid=np.linspace(0,1,4001)
    points=np.array([(2*s*s-3*s+1)*positions[0]+4*s*(1-s)*positions[1]+(2*s*s-s)*positions[2] for s in grid])
    independent=min(_oracle(point)-.01 for point in points)
    assert lower<=independent+1e-15
    assert independent-lower<=1e-6
    assert lower<0 and _oracle(positions[0])-.01>0 and _oracle(positions[-1])-.01>0
    with pytest.raises(MemoryError):
        curve_gap_lower(positions,np.zeros(3),frame,.01,.1,.01,accuracy=1e-12,maximum_depth=0)


@pytest.mark.parametrize('quadratic',[False,True])
def test_b2_b3_actual_line_energy_force_tangent_and_finite_ends(quadratic):
    model,definition,settings=_case(quadratic)
    assert validate_general_contact(model,definition,settings).valid
    manager=CappedCylinderBeamManager(model,definition,settings)
    u=_trial(model,range(1,4 if quadratic else 3),-.012)
    f,K,result=manager.evaluate(u)
    reference=.5*1e4*quad(lambda x:min(_oracle((x,0.,.018))-.01,0.)**2,-.5,.5,
        points=[-.1,.1],epsabs=1e-13)[0]
    assert result.line_contact_energy==pytest.approx(reference,rel=1e-6)
    assert result.active_line_length<.25 # infinite cylinder would activate the full metre
    np.testing.assert_allclose(np.array(result.beam_force)+result.rigid_force,0.,atol=1e-12)
    direction=_trial(model,range(1,4 if quadratic else 3),.7)
    step=1e-7
    fd=(manager.evaluate(u+step*direction,tangent=False)[0]-manager.evaluate(u-step*direction,tangent=False)[0])/(2*step)
    np.testing.assert_allclose(fd,K@direction,rtol=1e-5,atol=1e-6)
    through=_trial(model,range(1,4 if quadratic else 3),-.06)
    assert not manager.swept_increment_admissible(np.zeros_like(u),through,0.,1.)


def _static_case():
    model,definition,settings=_case(True)
    model.add_boundary_condition(BoundaryCondition('ends',(1,3),
        {name:0. for name in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('guide',(2,),
        {name:0. for name in ('ux','uy','rx','ry','rz')}))
    load=LoadCase('middle'); load.add_nodal_load(2,forces=(0.,0.,-1.))
    return model,definition,settings,load


def test_actual_static_displacement_control_active_restart_and_changed_length():
    model,definition,settings,load=_static_case()
    options=dict(contact=definition,contact_settings=settings,control='displacement',
        displacement_control=DisplacementControl(node_id=2,dof='uz',target_displacement=-.012),
        num_steps=4,max_iterations=20,emit_restart_checkpoint=True)
    full=solve_static_nonlinear(model,load,**options)
    assert full.status=='completed',full.failure_reason
    assert full.info['general_contact']['active_line_length']>0
    assert full.info['general_contact']['gap_tolerance_met']
    fresh,_,_,fresh_load=_static_case()
    first=solve_static_nonlinear(fresh,fresh_load,max_load_factor=.9,**options)
    assert first.status=='completed',first.failure_reason
    assert first.info['general_contact']['active_line_length']>0
    fresh,_,_,fresh_load=_static_case()
    resumed=solve_static_nonlinear(fresh,fresh_load,restart_checkpoint=first.restart_checkpoint,**options)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,full.displacements,atol=1e-12)
    if os.environ.get('CONTACT_APP_RESULT_RECORD'):
        Path(os.environ['CONTACT_APP_RESULT_RECORD']).with_name('capped_displacement_restart.json').write_text(
            json.dumps({'full':full.info,'prefix':first.info,'resumed':resumed.info,
                        'full_displacements':full.displacements.tolist()},indent=2),encoding='utf-8')
    changed=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],half_length=.11),))
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(fresh,fresh_load,restart_checkpoint=first.restart_checkpoint,
                              **{**options,'contact':changed})


def test_actual_static_force_equilibrium_and_independent_line_reaction():
    model,definition,settings,load=_static_case()
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
                                 num_steps=4,max_iterations=25,tolerance=1e-8)
    assert result.status=='completed',result.failure_reason
    report=result.info['general_contact']
    assert report['active_line_length']>0 and report['gap_tolerance_met']
    displacement=result.displacements[model.mesh.dof_manager.get_node_dofs(2)[2]]
    def force(x):
        s=x+.5; z=.03+4*s*(1-s)*displacement
        eps=1e-7
        g=_oracle((x,0.,z))-.01
        derivative=(_oracle((x,0.,z+eps))-_oracle((x,0.,z-eps)))/(2*eps)
        return -1e4*min(g,0.)*derivative
    independent=quad(force,-.5,.5,epsabs=1e-8,points=[-.1,.1])[0]
    assert report['beam_force'][2]==pytest.approx(independent,rel=1e-5,abs=1e-8)
    if os.environ.get('CONTACT_APP_RESULT_RECORD'):
        Path(os.environ['CONTACT_APP_RESULT_RECORD']).with_name('capped_force.json').write_text(
            json.dumps({'info':result.info,'independent_force_n':independent,
                        'displacements':result.displacements.tolist()},indent=2),encoding='utf-8')


def test_full_local_translation_work_and_tangent_cross_blocks():
    model,definition,settings=_case(True)
    manager=CappedCylinderBeamManager(model,definition,settings)
    u=_trial(model,(1,2,3),-.012)
    for node in (1,2,3): u[model.mesh.dof_manager.get_node_dofs(node)[1]]=.001
    f,K,result=manager.evaluate(u)
    direction=np.zeros_like(u)
    for node in (1,2,3):
        direction[model.mesh.dof_manager.get_node_dofs(node)[:3]]=[.03*node,.04*(4-node),-.1*node]
    for step in (3e-7,1e-7,3e-8):
        fp,_,rp=manager.evaluate(u+step*direction,tangent=False)
        fm,_,rm=manager.evaluate(u-step*direction,tangent=False)
        relative=np.linalg.norm((fp-fm)/(2*step)-K@direction)/np.linalg.norm(K@direction)
        assert relative<=1e-5
        assert (rp.line_contact_energy-rm.line_contact_energy)/(2*step)==pytest.approx(f@direction,rel=1e-5,abs=1e-9)


def test_cap_rim_branch_one_sided_derivatives_and_rigid_covariance():
    frame=_frame((1.,0.,0.)); origin=np.zeros(3)
    point=np.array([.1,.03,0.]); direction=np.array([1.,0.,0.])
    _,normal,_=signed_distance_work(point,origin,frame,.01,.1)
    for step in (1e-6,1e-7):
        _,plus,_=signed_distance_work(point+step*direction,origin,frame,.01,.1)
        _,minus,_=signed_distance_work(point-step*direction,origin,frame,.01,.1)
        np.testing.assert_allclose((normal-minus)/step,0.,atol=1e-8)
        np.testing.assert_allclose((plus-normal)/step,[50.,0.,0.],rtol=1e-4,atol=.002)
    from scipy.spatial.transform import Rotation
    rotation=Rotation.from_rotvec([.3,-.4,.2]).as_matrix(); shift=np.array([2.,-3.,.7])
    point=np.array([.13,.03,.01])
    d,g,H=signed_distance_work(point,origin,frame,.01,.1)
    moved_frame=_frame(rotation@np.array([1.,0.,0.]))
    md,mg,mH=signed_distance_work(rotation@point+shift,shift,moved_frame,.01,.1)
    assert md==pytest.approx(d,abs=1e-15)
    np.testing.assert_allclose(mg,rotation@g,atol=1e-14)
    np.testing.assert_allclose(mH,rotation@H@rotation.T,atol=1e-12)


def test_prescribed_tool_work_and_motion_restart_identity():
    model,definition,settings=_case(True)
    definition=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],path_translation=(0.,0.,.012)),))
    manager=CappedCylinderBeamManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    # A fixed beam and translated tool have potential work with a scalar oracle.
    def energy(t):
        return .5*1e4*quad(lambda x:min(_oracle((x,0.,.03-.012*t))-.01,0.)**2,
            -.5,.5,points=[-.1,.1],epsabs=1e-13)[0]
    errors=[]
    for count in (8,32):
        work=sum(manager.tool_work_increment(u,u,i/count,(i+1)/count) for i in range(count))
        errors.append(abs(work+energy(1.)-energy(0.)))
    assert errors[1]<errors[0]
    assert manager.path_factor==0. # tool integration restores algorithmic pose
    manager.set_path_factor(1.)
    assert manager.evaluate(u,tangent=False)[2].active_line_length>0
    assert manager.restart_contract()['body']['half_length']==.1


def test_unsupported_settings_and_declared_resource_exhaustion():
    model,definition,settings=_case(True)
    assert not validate_general_contact(model,definition,replace(settings,line_integration_relative_tolerance=None)).valid
    assert not validate_general_contact(model,definition,replace(settings,beam_shell_geometry='jet2_deformable_s3_dev')).valid
    assert not validate_general_contact(model,definition,settings,analysis_control='unsupported').valid
    manager=CappedCylinderBeamManager(model,definition,replace(settings,line_integration_relative_tolerance=1e-15,maximum_line_subdivisions=1))
    with pytest.raises(MemoryError,match='integration budget'):
        manager.evaluate(_trial(model,(1,2,3),-.012))


def test_review_interior_medial_crossing_is_excluded_before_force_mapping():
    model,definition,settings=_case(False)
    large=replace(definition,interactions=(replace(definition.interactions[0],gap_tolerance=.016),))
    assert not validate_general_contact(model,large,settings).valid
    manager=CappedCylinderBeamManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,position in ((1,(.092,0.,.005)),(2,(.098,0.,.005))):
        reference=model.mesh.nodes[node].coords()
        u[model.mesh.dof_manager.get_node_dofs(node)[:3]]=np.array(position)-reference
    with pytest.raises(ValueError,match='whole centreline outside'):
        manager.evaluate(u)


def test_repeated_side_grazing_root_and_quadratic_tool_through_crossing():
    model,definition,settings=_case(True)
    manager=CappedCylinderBeamManager(model,definition,settings)
    grazing=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,height in ((1,.025),(2,.02),(3,.025)):
        grazing[model.mesh.dof_manager.get_node_dofs(node)[2]]=height-model.mesh.nodes[node].z
    assert model.mesh.nodes[2].z+grazing[model.mesh.dof_manager.get_node_dofs(2)[2]]==2*definition.rigid_bodies[0].radius
    force,_,report=manager.evaluate(grazing)
    assert report.line_contact_energy==0.
    np.testing.assert_array_equal(force,np.zeros_like(force))
    moving=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],path_translation=(0.,0.,.06)),))
    manager=CappedCylinderBeamManager(model,moving,settings)
    start=np.zeros_like(grazing)
    end=_trial(model,(2,),-.004)
    manager.set_path_factor(1.)
    assert manager.evaluate(end,tangent=False)[2].minimum_gap>0
    manager.set_path_factor(0.)
    assert manager.evaluate(start,tangent=False)[2].minimum_gap>0
    assert not manager.swept_increment_admissible(start,end,0.,1.)


def test_narrow_quadratic_rim_pocket_with_open_cell_ends():
    model,definition,settings=_case(True)
    manager=CappedCylinderBeamManager(model,definition,settings)
    positions=np.array([[.1055,0.,.025],[.106,0.,.01799],[.1065,0.,.025]])
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,point in enumerate(positions,1):
        u[model.mesh.dof_manager.get_node_dofs(node)[:3]]=point-model.mesh.nodes[node].coords()
    _,_,report=manager.evaluate(u)
    assert _oracle(positions[0])-.01>0 and _oracle(positions[-1])-.01>0
    def density(s):
        point=(2*s*s-3*s+1)*positions[0]+4*s*(1-s)*positions[1]+(2*s*s-s)*positions[2]
        return .5*1e4*min(_oracle(point)-.01,0.)**2
    reference=quad(density,0.,1.,points=[.45,.5,.55],epsabs=1e-16,epsrel=1e-8)[0]
    assert reference>0 and report.active_line_length>0
    assert report.line_contact_energy==pytest.approx(reference,rel=1e-6,abs=1e-16)


def test_two_separated_quadratic_contact_pockets_match_independent_integral():
    model,definition,settings=_case(True)
    for node in model.mesh.nodes.values(): node.y=.03
    definition=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],normal=(0.,0.,1.)),))
    manager=CappedCylinderBeamManager(model,definition,settings)
    positions=np.array([[.04,.018,-.05],[-.01,.018,0.],[.04,.018,.05]])
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,point in enumerate(positions,1):
        u[model.mesh.dof_manager.get_node_dofs(node)[:3]]=point-model.mesh.nodes[node].coords()
    _,_,report=manager.evaluate(u)
    def density(s):
        x=.04-.2*s*(1-s)
        gap=np.hypot(x,.018)-.02
        return .5*1e4*min(gap,0.)**2
    independent=quad(density,0.,1.,points=[.2,.3,.4,.6,.7,.8],epsabs=1e-13)[0]
    assert report.active_intervals>=2
    assert report.line_contact_energy==pytest.approx(independent,rel=1e-6)


def test_actual_prescribed_tool_force_control_and_work():
    model,definition,settings,load=_static_case()
    definition=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],path_translation=(0.,0.,.012)),))
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
                                 num_steps=4,max_iterations=25,tolerance=1e-8)
    assert result.status=='completed',result.failure_reason
    contact=result.info['general_contact']
    assert contact['active_line_length']>0 and contact['gap_tolerance_met']
    assert contact['rigid_tool_position'][2]==pytest.approx(.012)
    assert contact['rigid_contact_work_trapezoid']<0
    assert contact['prescribed_actuator_work_trapezoid']==pytest.approx(-contact['rigid_contact_work_trapezoid'])


def test_public_development_example_runs_actual_static_route():
    import runpy
    example=Path(__file__).resolve().parents[1]/'examples/general_contact_round_beam_capped_cylinder.py'
    result=runpy.run_path(str(example))['run']()
    assert result.status=='completed'
    contact=result.info['general_contact']
    assert contact['profile']=='CLASSICAL_B2_B3_ROUND_TUBE_CAPPED_CYLINDER_LINE_PENALTY_STATIC_DEV_V0'
    assert contact['active_line_length']>0 and contact['gap_tolerance_met']


def test_capability_reports_development_geometry_without_application_admission():
    from anyfesolver import report_general_contact_capability
    model,definition,settings=_case(True)
    report=report_general_contact_capability(model,definition,settings)
    assert report.implemented and report.rigid_profiles==(('obstacle','capped_cylinder'),)
    assert not report.tested and not report.independently_accepted
    assert not report.released and not report.admitted_for_application


def test_nonmidpoint_b3_reference_rejects_chord_measure_substitution():
    model,definition,settings=_case(True)
    model.mesh.nodes[2].y=.001
    report=validate_general_contact(model,definition,settings)
    assert not report.valid and any('non-midpoint' in issue for issue in report.issues)
    with pytest.raises(ValueError,match='non-midpoint'):
        CappedCylinderBeamManager(model,definition,settings)


def test_actual_import_root_preserves_lazy_no_contact_solve():
    import anyfesolver
    import subprocess
    import sys
    root=Path(anyfesolver.__file__).resolve().parent.parent
    environment=dict(os.environ,PYTHONPATH=str(root),CAPS_EXPECTED_IMPORT_ROOT=str(root))
    code='''import os,sys
from pathlib import Path
import anyfesolver
assert Path(anyfesolver.__file__).resolve().is_relative_to(Path(os.environ['CAPS_EXPECTED_IMPORT_ROOT']))
assert 'anyfesolver.general_contact' not in sys.modules
from anyfesolver.boundary import LoadCase
from anyfesolver.mesh_gen import generate_beam_mesh
model=generate_beam_mesh(1.,num_divisions=1)
load=LoadCase('axial'); load.add_nodal_load(2,[10.,0.,0.,0.,0.,0.])
result=anyfesolver.solve_static_nonlinear(model,load,num_steps=1,max_iterations=5)
assert result.status=='completed',result.failure_reason
assert 'anyfesolver.general_contact.core' not in sys.modules
assert 'anyfesolver.general_contact.capped_cylinder' not in sys.modules
from anyfesolver import RigidCappedCylinder
from anyfesolver.general_contact import RigidCappedCylinder as exported
assert RigidCappedCylinder is exported
'''
    completed=subprocess.run([sys.executable,'-c',code],env=environment,
        capture_output=True,text=True,timeout=30)
    assert completed.returncode==0,completed.stdout+completed.stderr
