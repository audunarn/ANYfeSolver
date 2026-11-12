"""Independent hemisphere endpoint work, topology, affine ownership and solves."""
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import os
import math

import numpy as np
import pytest
from scipy.optimize import brentq
from anyfesolver import (ContactDefinition,ContactInteraction,ContactSolverSettings,ContactSurface,
    RigidCappedCylinder,BoundaryCondition,LoadCase,DisplacementControl,solve_static_nonlinear,
    validate_general_contact)
from anyfesolver.elements import BeamElement,QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.capped_cylinder import (CappedCylinderBeamManager,PROFILE_END_CAPS,
    _cap_orientation_range,_frame,_controls,_end_geometry_controls,_point_gap_sign)
from test_general_contact_capped_cylinder import _oracle


def _case(quadratic=False,feature='cap',reverse=False,endpoint_stiffness=1000.):
    model=FEModel('declared_beam_hemisphere_end')
    model.add_material('elastic',1e3,.3,density=1.)
    if feature=='cap':point=np.array([.112,0.,0.]);normal=np.array([1.,0.,0.])
    elif feature=='side':point=np.array([0.,0.,.022]);normal=np.array([0.,0.,1.])
    else:
        normal=np.array([1.,0.,1.])/np.sqrt(2)
        point=np.array([.1,0.,.01])+.012*normal
    end=point+.2*normal
    points=(point,(point+end)/2,end) if quadratic else (point,end)
    for node,p in enumerate(points,1):model.add_node(node,*p)
    nodes=tuple(range(1,len(points)+1));nodes=nodes[::-1] if reverse else nodes
    section={'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}
    model.add_element(1,(QuadraticBeamElement if quadratic else BeamElement)(1,nodes,'elastic',section))
    definition=ContactDefinition((ContactSurface('tube',(1,),beam_radius=.01,beam_end_caps='hemisphere'),),
        (RigidCappedCylinder('obstacle',(0.,0.,0.),(1.,0.,0.),.01,.1),),
        (ContactInteraction('tube_solid','tube','obstacle',None,.004,line_stiffness=1e4,
            endpoint_stiffness=endpoint_stiffness),))
    settings=ContactSolverSettings(pair_motion='general',line_integration_relative_tolerance=1e-7,
        maximum_line_subdivisions=12,maximum_swept_subdivisions=14)
    return model,definition,settings,normal


def _closed(model,normal,quadratic=False):
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(1)[:3]]=-.004*normal
    if quadratic:u[model.mesh.dof_manager.get_node_dofs(2)[:3]]=-.001*normal+np.array([0.,.002,0.])
    return u


def _without_caps(definition):
    return replace(definition,surfaces=(replace(definition.surfaces[0],beam_end_caps='none'),),
        interactions=(replace(definition.interactions[0],endpoint_stiffness=None),))


@pytest.mark.parametrize('quadratic,feature',[(q,f) for q in (False,True) for f in ('cap','side','rim')])
def test_independent_endpoint_energy_work_and_tangent(quadratic,feature):
    model,definition,settings,normal=_case(quadratic,feature)
    manager=CappedCylinderBeamManager(model,definition,settings)
    plain=CappedCylinderBeamManager(model,_without_caps(definition),settings)
    u=_closed(model,normal,quadratic)
    f,K,result=manager.evaluate(u);f0,K0,line=plain.evaluate(u)
    point=model.mesh.nodes[1].coords()+u[model.mesh.dof_manager.get_node_dofs(1)[:3]]
    gap=_oracle(point)-.01
    assert result.profile==PROFILE_END_CAPS and result.active_endpoints==1
    assert result.endpoint_contact_energy==pytest.approx(.5*1000*gap*gap,rel=1e-12)
    assert result.line_contact_energy==line.line_contact_energy
    assert result.to_dict()['algorithmic_contact_potential']==pytest.approx(line.line_contact_energy+result.endpoint_contact_energy)
    fields=result.to_dict()['endpoint_fields']
    assert {field['node_id'] for field in fields}=={1,3 if quadratic else 2}
    assert sum(field['state']=='owned_active' for field in fields)==1
    assert result.to_dict()['moment_origin']==[0.,0.,0.]
    np.testing.assert_allclose(np.asarray(result.beam_force)+result.rigid_force,0.,atol=1e-12)
    direction=np.zeros_like(u);direction[model.mesh.dof_manager.get_node_dofs(1)[:3]]=(.3,-.4,.2)
    errors=[]
    full_errors=[]
    for h in (1e-6,3e-7,1e-7):
        plus=manager.evaluate(u+h*direction,tangent=False)[0]
        minus=manager.evaluate(u-h*direction,tangent=False)[0]
        full_expected=K@direction
        full_errors.append(np.linalg.norm((plus-minus)/(2*h)-full_expected)/max(np.linalg.norm(full_expected),1e-12))
        fp=plus-plain.evaluate(u+h*direction,tangent=False)[0]
        fm=minus-plain.evaluate(u-h*direction,tangent=False)[0]
        expected=(K-K0)@direction
        errors.append(np.linalg.norm((fp-fm)/(2*h)-expected)/max(np.linalg.norm(expected),1e-12))
        def energy(q):
            p=model.mesh.nodes[1].coords()+q[model.mesh.dof_manager.get_node_dofs(1)[:3]]
            return .5*1000*min(_oracle(p)-.01,0.)**2
        assert (energy(u+h*direction)-energy(u-h*direction))/(2*h)==pytest.approx((f-f0)@direction,rel=1e-5,abs=1e-9)
    assert min(errors)<=1e-5,errors
    assert min(full_errors)<=1e-5,full_errors


@pytest.mark.parametrize('quadratic,reverse',[(q,r) for q in (False,True) for r in (False,True)])
def test_physical_end_and_b3_midpoint_junction_ownership(quadratic,reverse):
    model,definition,settings,_=_case(quadratic,reverse=reverse)
    p=model.mesh.nodes[1].coords()
    model.add_node(10,*(p+(0.,-.1,0.)));model.add_node(11,*(p+(0.,.1,0.)))
    model.add_element(2,QuadraticBeamElement(2,(10,1,11),'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    manager=CappedCylinderBeamManager(model,definition,settings)
    assert set(manager.endpoints)=={3 if quadratic else 2}
    # The same topology correction applies to the existing plane owner.
    from anyfesolver import RigidContactBody
    from anyfesolver.general_contact.beam_plane import _BeamPlaneManager
    plane=replace(definition,rigid_bodies=(RigidContactBody('obstacle','plane',(.1,0.,0.),(1.,0.,0.)),))
    plane_manager=_BeamPlaneManager(model,plane,settings)
    assert set(plane_manager.endpoints)==set(manager.endpoints)
    assert plane_manager.restart_contract()['end_topology_revision']=='all_classical_material_node_incidence_v1'


def test_endpoint_law_and_restart_identity_are_explicit():
    model,definition,settings,_=_case()
    missing=replace(definition,interactions=(replace(definition.interactions[0],endpoint_stiffness=None),))
    assert not validate_general_contact(model,missing,settings).valid
    with pytest.raises(ValueError,match='endpoint_stiffness'):CappedCylinderBeamManager(model,missing,settings)
    report=validate_general_contact(model,definition,settings)
    assert report.valid and report.profile==PROFILE_END_CAPS
    contract=CappedCylinderBeamManager(model,definition,settings).restart_contract()
    assert contract['interaction']['endpoint_stiffness']==1000.
    assert len(contract['physical_endpoints'])==2
    old=CappedCylinderBeamManager(model,_without_caps(definition),settings).restart_contract()
    assert 'end_cap_geometry_revision' not in old
    from anyfesolver import report_general_contact_capability
    capability=report_general_contact_capability(model,definition,settings)
    assert capability.implemented and capability.validation.profile==PROFILE_END_CAPS
    assert not capability.tested and not capability.independently_accepted
    assert not capability.admitted_for_application and not capability.released


def test_exact_affine_ownership_rejects_active_switch_and_permits_open_switch():
    model,definition,settings,_=_case()
    manager=CappedCylinderBeamManager(model,definition,settings)
    # Restrict to one actual physical end, so this independently checks the
    # endpoint certificate; the full manager additionally certifies every line.
    manager.endpoints={1:manager.endpoints[1]}
    def pose(x,y,outward):
        values=np.zeros(model.mesh.dof_manager.total_dofs)
        p=np.array([x,y,0.]);other=p-np.array(outward)
        for node,point in ((1,p),(2,other)):
            values[model.mesh.dof_manager.get_node_dofs(node)[:3]]=point-model.mesh.nodes[node].coords()
        return values
    a=pose(.108,0.,(-.0001,.02,0.));b=pose(.108,0.,(.0001,.02,0.))
    assert not manager._end_cap_swept_admissible(a,b,np.zeros(3),np.zeros(3))
    a=pose(.13,0.,(-.0001,.02,0.));b=pose(.13,0.,(.0001,.02,0.))
    assert manager._end_cap_swept_admissible(a,b,np.zeros(3),np.zeros(3))
    # Both endpoint samples are open; the midpoint is closed at ownership zero.
    a=pose(.108,-.1,(-.0001,.02,0.));b=pose(.108,.1,(.0001,.02,0.))
    assert _oracle((.108,-.1,0.))>.01 and _oracle((.108,.1,0.))>.01
    assert not manager._end_cap_swept_admissible(a,b,np.zeros(3),np.zeros(3))


def _static_case(endpoint_stiffness=1e5):
    model,definition,settings,_=_case(endpoint_stiffness=endpoint_stiffness)
    definition=replace(definition,interactions=(replace(definition.interactions[0],gap_tolerance=1e-5),))
    fixed={d:0. for d in ('ux','uy','uz','rx','ry','rz')}
    model.add_boundary_condition(BoundaryCondition('root',(2,),fixed))
    model.add_boundary_condition(BoundaryCondition('guide',(1,),{d:0. for d in ('uy','uz','rx','ry','rz')}))
    load=LoadCase('press');load.add_nodal_load(1,forces=(-.2,0.,0.))
    return model,definition,settings,load


def test_actual_force_displacement_active_restart_and_analytic_bar_reaction():
    model,definition,settings,load=_static_case()
    common=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,tolerance=1e-10)
    full=solve_static_nonlinear(model,load,**common)
    assert full.status=='completed',full.failure_reason
    # Independent exact axial-bar + line integral + endpoint spring equation.
    L=.2;g0=.002;EA=1.;k=1e4;ke=1e5;P=.2
    def balance(delta):
        length=L+g0+delta
        return EA/L*(g0+delta)+ke*delta+k*L*(delta**2/(2*length)-delta**3/(6*length**2))-P
    delta=brentq(balance,0.,1e-5,xtol=1e-18)
    actual=full.displacements[model.mesh.dof_manager.get_node_dofs(1)[0]]
    assert actual==pytest.approx(-g0-delta,rel=1e-6,abs=1e-10)
    result=full.info['general_contact']
    assert result['endpoint_contact_energy']==pytest.approx(.5*ke*delta*delta,rel=1e-6)
    assert result['minimum_gap']>=-1e-5 and result['active_endpoints']==1
    model,definition,settings,load=_static_case()
    prefix=solve_static_nonlinear(model,load,**{**common,'num_steps':2},max_load_factor=.75,emit_restart_checkpoint=True)
    assert prefix.status=='completed',prefix.failure_reason
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint()))
    model,definition,settings,load=_static_case()
    resumed=solve_static_nonlinear(model,load,**{**common,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,full.displacements,rtol=1e-6,atol=1e-10)
    changed=replace(definition,interactions=(replace(definition.interactions[0],endpoint_stiffness=2e5),))
    with pytest.raises(ValueError,match='restart|incompatible|contract'):
        solve_static_nonlinear(model,load,**{**common,'contact':changed},restart_checkpoint=checkpoint)
    model,definition,settings,load=_static_case()
    prescribed=solve_static_nonlinear(model,load,**common,control='displacement',
        displacement_control=DisplacementControl(node_id=1,dof='ux',target_displacement=-.002005))
    assert prescribed.status=='completed',prescribed.failure_reason
    assert prescribed.info['general_contact']['active_endpoints']==1
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).write_text(json.dumps({'scope':'axial bar + distributed line + declared hemisphere endpoint development',
        'displacement_m':actual,'independent_displacement_m':-g0-delta,'force_result':result,
        'displacement_result':prescribed.info['general_contact']},indent=2))


def test_actual_prescribed_tool_work_with_activation_stage_and_open_restart():
    model,definition,settings,load=_static_case()
    definition=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],path_translation=(.004,0.,0.)),))
    load=LoadCase('tool only')
    # Prospectively prescribed physical activation stage. The unaligned
    # four-step work nonpass remains at endcaps_tool_discriminator; its independent
    # force-history work matched production, exposing trapezoidal event error.
    activation=float((Fraction.from_float(model.mesh.nodes[1].x)-Fraction.from_float(.1)
        -Fraction.from_float(.01))/Fraction.from_float(.004))
    manager=CappedCylinderBeamManager(model,definition,settings)
    for _ in range(8):
        point,_=_end_geometry_controls(manager.cells[0].reference,manager.point_at(activation),
            manager.frame,0,-1)
        if _point_gap_sign(point,.01,.1,.01)>=0:break
        activation=math.nextafter(activation,0.)
    else:raise AssertionError('activation stage has no represented open-side pose')
    prefix=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=2,max_load_factor=activation,max_iterations=20,tolerance=1e-10,
        emit_restart_checkpoint=True)
    assert prefix.status=='completed',prefix.failure_reason
    assert prefix.info['general_contact']['active_endpoints']==0
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint()))
    model,definition,settings,load=_static_case()
    definition=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],path_translation=(.004,0.,0.)),))
    load=LoadCase('tool only')
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=2,max_iterations=20,tolerance=1e-10,restart_checkpoint=checkpoint)
    assert result.status=='completed',result.failure_reason
    info=result.info['general_contact']
    assert info['active_endpoints']==1
    preflights=[row for row in result.info['convergence_adaptation']
        if row.get('residual_evaluated') is False]
    assert preflights
    assert all(row['residual_norm'] is None and row['iterations']==0
        and row['failure_reason']=='contact_gap_exceeded' for row in preflights)
    json.dumps(result.to_restart_checkpoint(),allow_nan=False)
    u=result.displacements[model.mesh.dof_manager.get_node_dofs(1)[0]]
    storage=.5*5*u*u+info['endpoint_contact_energy']+info['line_contact_energy']
    history=result.info['force_displacement_history']
    previous_factor=previous_force=0.;oracle_work=0.;oracle_steps=[]
    for step in history:
        factor=step['load_factor'];displacement=step['control_value']
        delta=max(.004*factor-.002-displacement,0.)
        force=1e5*delta+1e4*.2*delta*delta/(2*(.2-displacement))
        oracle_work+=.5*(force+previous_force)*.004*(factor-previous_factor)
        oracle_steps.append({'factor':factor,'displacement':displacement,'force':force})
        previous_factor,previous_force=factor,force
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).with_name('tool_work.json').write_text(json.dumps({'storage':storage,
        'reported_work':info['prescribed_actuator_work_trapezoid'],'independent_work':oracle_work,
        'contact':info,'history':history,'independent_steps':oracle_steps,'solver_info':result.info},indent=2))
    assert info['prescribed_actuator_work_trapezoid']==pytest.approx(storage,rel=1e-6,abs=1e-12)
    assert info['rigid_contact_work_trapezoid']==pytest.approx(-storage,rel=1e-6,abs=1e-12)


@pytest.mark.parametrize('feature',('cap','side'))
def test_atomic_endpoint_feature_seams_reject_distinct_one_sided_hessians(feature):
    model,definition,settings,normal=_case(False,feature)
    # Cap/rim: rho=R, z=h+d. Side/rim: z=h, rho=R+d.
    offset=np.array([0.,0.,.01]) if feature=='cap' else np.array([.1,0.,0.])
    for node in model.mesh.nodes.values():
        p=node.coords()+offset;node.x,node.y,node.z=p
    manager=CappedCylinderBeamManager(model,definition,settings)
    u=_closed(model,normal);point=model.mesh.nodes[1].coords()-.004*normal
    perturb=np.array([0.,0.,1.]) if feature=='cap' else np.array([1.,0.,0.])
    stiffness=definition.interactions[0].endpoint_stiffness
    def energy(p):
        distance=_oracle(p,.01,.1)
        return .5*stiffness*(distance-.01)**2
    expected=stiffness*(.008-.01)/.008
    for h in (1e-6,3e-7):
        inward=(energy(point)-2*energy(point-h*perturb)+energy(point-2*h*perturb))/h**2
        outward=(energy(point+2*h*perturb)-2*energy(point+h*perturb)+energy(point))/h**2
        assert inward==pytest.approx(0.,abs=1e-4)
        assert outward==pytest.approx(expected,rel=1e-5)
    for tangent in (False,True):
        with pytest.raises(ValueError,match='feature seam'):
            manager.evaluate(u,tangent=tangent)


@pytest.mark.parametrize('quadratic',(False,True))
def test_common_rigid_motion_preserves_complete_endpoint_work(quadratic):
    model,definition,settings,normal=_case(quadratic,'rim')
    u=_closed(model,normal,quadratic)
    f,K,r=CappedCylinderBeamManager(model,definition,settings).evaluate(u)
    angle=.31;axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis)
    cross=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    Q=np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross);shift=np.array([.37,-.21,.11])
    moved,md,_,_= _case(quadratic,'rim')
    for node in moved.mesh.nodes.values():
        point=Q@node.coords()+shift;node.x,node.y,node.z=point
    md=replace(md,rigid_bodies=(replace(md.rigid_bodies[0],point=tuple(shift),normal=tuple(Q@np.array([1.,0.,0.]))),))
    T=np.kron(np.eye(len(u)//3),Q)
    mf,mK,mr=CappedCylinderBeamManager(moved,md,settings).evaluate(T@u)
    np.testing.assert_allclose(mf,T@f,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(mK.toarray(),T@K.toarray()@T.T,rtol=1e-8,atol=1e-7)
    assert mr.endpoint_contact_energy==pytest.approx(r.endpoint_contact_energy,rel=1e-9)
    expected=Q@np.array(r.beam_moment)+np.cross(shift,Q@np.array(r.beam_force))
    np.testing.assert_allclose(mr.beam_moment,expected,rtol=1e-9,atol=1e-10)


def test_public_endpoint_example_and_installed_no_contact_compatibility():
    from examples.general_contact_beam_end_capped_cylinder import run
    result=run()
    assert result.info['general_contact']['profile']==PROFILE_END_CAPS
    assert result.info['general_contact']['active_endpoints']==1
    # This child follows the effective source/installed runtime, unlike the
    # historical lazy-import test whose child explicitly requests source.
    import subprocess,sys
    import anyfesolver
    root=str(Path(anyfesolver.__file__).resolve().parent)
    environment=os.environ.copy()
    environment['PYTHONPATH']=os.pathsep.join(
        (str(Path(root).parent),environment.get('PYTHONPATH','')))
    command='''import sys
from pathlib import Path
import anyfesolver
assert Path(anyfesolver.__file__).resolve().is_relative_to(Path(sys.argv[1]))
from anyfesolver.boundary import LoadCase
from anyfesolver.mesh_gen import generate_beam_mesh
model=generate_beam_mesh(1.,num_divisions=1)
load=LoadCase('axial');load.add_nodal_load(2,[10.,0.,0.,0.,0.,0.])
result=anyfesolver.solve_static_nonlinear(model,load,num_steps=1,max_iterations=5)
assert result.status=='completed',result.failure_reason
assert not any(n in sys.modules for n in ('anyfesolver.general_contact.capped_cylinder',
    'anyfesolver.general_contact.beam_plane','anyfesolver.general_contact.beam_end_topology'))
'''
    process=subprocess.run([sys.executable,'-c',command,root],env=environment,
                           capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stdout+process.stderr
