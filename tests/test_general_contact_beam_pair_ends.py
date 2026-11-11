"""Independent work and two-body response for the bounded physical-end row."""
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.integrate import quad

from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    solve_static_nonlinear, validate_general_contact, report_general_contact_capability)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_pair_ends import BeamPairEndManager, PROFILE
from anyfesolver.general_contact.beam_pair_end_geometry import minimum_gap, endpoint_feature


def fixture(interior=False):
    model=FEModel('two physical beam ends')
    model.add_material('elastic',1e3,.3,density=1.)
    positions=((0.,.224,0.),(0.,.024,0.),(-.2,0.,0.),(.2,0.,0.)) if interior else (
        (-.212,0.,0.),(-.012,0.,0.),(.012,0.,0.),(.212,0.,0.))
    for node,p in enumerate(positions,1):model.add_node(node,*p)
    section={'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}
    model.add_element(1,BeamElement(1,(1,2),'elastic',section))
    model.add_element(2,BeamElement(2,(3,4),'elastic',section))
    model.add_boundary_condition(BoundaryCondition('roots',(1,4),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('axial guides',(2,3),
        {d:0. for d in ('uy','uz','rx','ry','rz')}))
    definition=ContactDefinition(tuple(ContactSurface(name,(element,),beam_radius=.01,
        beam_end_caps='hemisphere') for name,element in (('a',1),('b',2))),(),
        (ContactInteraction('pair','a','b',None,1e-5,line_stiffness=1e4,endpoint_stiffness=1e5),))
    settings=ContactSolverSettings(pair_motion='general',line_integration_relative_tolerance=1e-7,
        maximum_line_subdivisions=14,maximum_swept_subdivisions=16)
    load=LoadCase('symmetric axial closure')
    load.add_nodal_load(2,forces=(.2,0.,0.));load.add_nodal_load(3,forces=(-.2,0.,0.))
    return model,definition,settings,load


def independent_potential(positions, radius, kl, ke, genuine=((0,1),(0,1))):
    # Scalar oracle has its own projections, work, quadrature and ownership.
    def closest(p,target):
        v=target[1]-target[0];t=float(np.clip((p-target[0])@v/(v@v),0.,1.))
        return t,p-((1.-t)*target[0]+t*target[1])
    line=0.
    for side in (0,1):
        source,target=positions[side],positions[1-side]
        length=independent_potential.lengths[side]
        def value(s):
            _,d=closest((1.-s)*source[0]+s*source[1],target)
            return .5*kl*min(np.linalg.norm(d)-radius,0.)**2
        w=source[0]-target[0];u=source[1]-source[0];v=target[1]-target[0]
        raw=w@v/(v@v);slope=u@v/(v@v);cuts=[0.,1.]
        if slope:
            cuts.extend(t for t in (-raw/slope,(1.-raw)/slope) if 0.<t<1.)
        initial=sorted(cuts)
        for left,right in zip(initial,initial[1:]):
            projection=raw+slope*(left+right)/2
            t0,t1=((0.,0.) if projection<=0 else (1.,0.) if projection>=1 else (raw,slope))
            c,d=w-t0*v,u-t1*v
            roots=np.roots((d@d,2*c@d,c@c-radius**2))
            cuts.extend(float(r.real) for r in roots if abs(r.imag)<1e-14 and left<r.real<right)
        cuts=sorted(set(cuts))
        line+=.5*length*sum(quad(value,l,r,epsabs=1e-15,epsrel=1e-10,limit=100)[0]
            for l,r in zip(cuts,cuts[1:]))
    end=0.;seen=set()
    for side in (0,1):
        source,target=positions[side],positions[1-side]
        for i in genuine[side]:
            t,d=closest(source[i],target);gap=np.linalg.norm(d)-radius
            outward=(source[1]-source[0])*(1 if i==1 else -1)
            if gap>=0. or d@outward>=0.:continue
            if t in (0.,1.):
                other=int(t);key=tuple(sorted(((side,i),(1-side,other))))
            else:key=(side,i,'interior')
            if key in seen:continue
            seen.add(key);end+=.5*ke*gap**2
    return line+end


@pytest.mark.parametrize('interior',(False,True))
def test_complete_both_body_work_tangent_wrench_and_unique_point_law(interior):
    model,definition,settings,_=fixture(interior)
    manager=BeamPairEndManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(2)[1 if interior else 0]]=-.0042 if interior else .0021
    if not interior:u[model.mesh.dof_manager.get_node_dofs(3)[0]]=-.0021
    f,K,r=manager.evaluate(u);info=r.to_dict()
    assert info['active_endpoints']==1
    assert info['endpoint_contact_energy']==pytest.approx(.5*1e5*.0002**2,rel=1e-10)
    assert sum(row['state']=='active' for row in info['endpoint_fields'])==1
    independent_potential.lengths=tuple(s.length for s in manager.segments)
    poses=manager._positions(u)
    assert r.contact_energy==pytest.approx(independent_potential(poses,.02,1e4,1e5),rel=1e-8,abs=1e-11)
    direction=np.zeros_like(u)
    values=np.array(((.21,-.14,.31),(-.29,.36,-.11),(.13,-.26,.19),(-.32,.17,.23)))
    for node,v in enumerate(values,1):direction[model.mesh.dof_manager.get_node_dofs(node)[:3]]=v
    for step in (1e-7,3e-8):
        plus=manager.evaluate(u+step*direction,tangent=False)[0]
        minus=manager.evaluate(u-step*direction,tangent=False)[0]
        assert np.linalg.norm((plus-minus)/(2*step)-K@direction)/np.linalg.norm(K@direction)<1e-5
        Eplus=independent_potential(manager._positions(u+step*direction),.02,1e4,1e5)
        Eminus=independent_potential(manager._positions(u-step*direction),.02,1e4,1e5)
        assert (Eplus-Eminus)/(2*step)==pytest.approx(f@direction,rel=1e-5,abs=1e-7)
    assert np.linalg.norm(np.array(r.beam_a_force)+r.beam_b_force)<1e-6
    assert np.linalg.norm(np.array(r.beam_a_moment)+r.beam_b_moment)<1e-6
    np.testing.assert_allclose(K.toarray(),K.toarray().T,rtol=1e-12,atol=1e-9)


def test_actual_two_body_bar_force_control_displacement_and_split_restart():
    model,definition,settings,load=fixture()
    args=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=25,
        tolerance=1e-10,emit_restart_checkpoint=True)
    result=solve_static_nonlinear(model,load,**args)
    assert result.status=='completed',result.failure_reason
    # Each bar has EA=1,L=.2; closure per end a=(g0+delta)/2.
    def balance(delta):
        a=(.004+delta)/2;D=.2+a
        return 5*a+1e5*delta+1e4*.2*(delta**2/(2*D)-delta**3/(12*D**2))-.2
    delta=brentq(balance,0.,1e-5,xtol=1e-18);a=(.004+delta)/2
    dofs=[model.mesh.dof_manager.get_node_dofs(n)[0] for n in (2,3)]
    np.testing.assert_allclose(result.displacements[dofs],(a,-a),rtol=1e-6,atol=1e-10)
    assert result.info['general_contact']['active_endpoints']==1
    prefix_model,_,_,_=fixture()
    prefix=solve_static_nonlinear(prefix_model,load,**{**args,'num_steps':2},max_load_factor=.75)
    assert prefix.status=='completed',prefix.failure_reason
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_=fixture()
    resumed=solve_static_nonlinear(fresh,load,**{**args,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    changed=replace(definition,interactions=(replace(definition.interactions[0],endpoint_stiffness=1.01e5),))
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(fresh,load,**{**args,'contact':changed},restart_checkpoint=checkpoint)
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(fresh,load,**{**args,'contact_settings':replace(settings,
            maximum_swept_subdivisions=15)},restart_checkpoint=checkpoint)
    joined,_,_,_=fixture()
    joined.add_node(5,.012,-.1,0.);joined.add_node(6,.012,.1,0.)
    joined.add_element(3,QuadraticBeamElement(3,(5,3,6),'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(joined,load,**args,restart_checkpoint=checkpoint)
    control_model,_,_,_=fixture()
    control=solve_static_nonlinear(control_model,load,**args,control='displacement',
        displacement_control=DisplacementControl(node_id=2,dof='ux',target_displacement=a))
    assert control.status=='completed',control.failure_reason
    np.testing.assert_allclose(control.displacements[dofs],result.displacements[dofs],rtol=1e-6,atol=1e-10)
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).with_name('independent_two_bar.json').write_text(json.dumps({
        'independent_displacement_m':[a,-a],'force_result_m':result.displacements[dofs].tolist(),
        'contact':result.info['general_contact']},indent=2))


def test_gap_exact_parallel_near_parallel_and_interior_minimum():
    a=np.array(((0.,0.,0.),(1.,0.,0.)))
    assert minimum_gap(a,a+(0.,.03,0.),.02)==pytest.approx(.01,abs=1e-15)
    b=np.array(((0.,-.5,.03),(0.,.5,.03)))
    assert minimum_gap(a,b,.02)==pytest.approx(.01,abs=1e-15)
    b=np.array(((0.,.03,0.),(1.,.03+1e-15,0.)))
    assert minimum_gap(a,b,.02)==pytest.approx(.01,abs=1e-15)


def test_full_path_hidden_crossing_collapse_and_bounded_trial_limiting():
    model,definition,settings,_=fixture()
    manager=BeamPairEndManager(model,definition,settings)
    zero=np.zeros(model.mesh.dof_manager.total_dofs);cross=zero.copy()
    for node in (3,4):cross[model.mesh.dof_manager.get_node_dofs(node)[0]]=-.5
    assert not manager.swept_increment_admissible(zero,cross,0.,1.)
    fraction=manager.admissible_step_fraction(zero,cross)
    assert 0.<fraction<1.
    collapse=zero.copy();collapse[model.mesh.dof_manager.get_node_dofs(2)[0]]=-.4
    assert not manager.swept_increment_admissible(zero,collapse,0.,1.)
    assert 0.<manager.admissible_step_fraction(zero,collapse)<.5


def test_active_line_projection_plateau_rejects_even_when_source_ends_open():
    model,definition,settings,_=fixture()
    manager=BeamPairEndManager(model,definition,settings)
    source=np.array(((0.,-.1,.01),(0.,.1,.01)))
    target=np.array(((0.,0.,0.),(.2,0.,0.)))
    with pytest.raises(ValueError,match='plateau'):
        manager._integrate_direction(source,target,.2,True)


def test_path_rejects_hidden_active_line_plateau_with_open_physical_ends():
    from anyfesolver.general_contact.beam_pair_end_geometry import admissible_line_projection_path
    from anyfesolver.general_contact.beam_end_radial_geometry import regular_curve_path
    source0=np.array(((-.003,-.1,.024),(-.003,.1,.024)))
    source1=source0+(.006,0.,0.)
    target=np.array(((0.,0.,0.),(.2,0.,0.)))
    assert admissible_line_projection_path(source0,source1,target,target,.02,8)  # Entirely open.
    source0[:,2]=source1[:,2]=.019999
    assert regular_curve_path(source0,source1,8)
    assert not admissible_line_projection_path(source0,source1,target,target,.02,8)


@pytest.mark.parametrize('bad',('missing_point_law','one_cap','B3','candidate_limit'))
def test_public_profile_is_bounded_and_no_application_acceptance(bad):
    model,definition,settings,_=fixture()
    if bad=='missing_point_law':definition=replace(definition,interactions=(
        replace(definition.interactions[0],endpoint_stiffness=None),))
    elif bad=='one_cap':definition=replace(definition,surfaces=(
        definition.surfaces[0],replace(definition.surfaces[1],beam_end_caps='none')))
    elif bad=='B3':
        model.add_node(5,-.112,.001,0.)
        model.mesh.elements[1]=QuadraticBeamElement(1,(1,5,2),'elastic',
            {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4})
    if bad=='candidate_limit':
        # Resource limit is tested by an unsupported multi-cell selection; no contacts are dropped.
        definition=replace(definition,surfaces=(replace(definition.surfaces[0],element_ids=(1,2)),definition.surfaces[1]))
        settings=ContactSolverSettings(pair_motion='general',line_integration_relative_tolerance=1e-7,maximum_candidate_pairs=1)
    assert not validate_general_contact(model,definition,settings).valid
    with pytest.raises(ValueError):BeamPairEndManager(model,definition,settings)


def test_supported_capability_keeps_development_disposition():
    model,definition,settings,_=fixture()
    capability=report_general_contact_capability(model,definition,settings)
    assert capability.validation.profile==PROFILE and capability.implemented
    assert not capability.admitted_for_application and not capability.independently_accepted
    assert not capability.released


def test_body_exchange_connectivity_reversal_and_rigid_covariance():
    model,definition,settings,_=fixture(True)
    manager=BeamPairEndManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(2)[1]]=-.0042
    f,K,r=manager.evaluate(u)
    exchanged=replace(definition,surfaces=tuple(reversed(definition.surfaces)),
        interactions=(replace(definition.interactions[0],surface='b',target='a'),))
    ef,eK,er=BeamPairEndManager(model,exchanged,settings).evaluate(u)
    np.testing.assert_allclose(ef,f,atol=1e-10);np.testing.assert_allclose(eK.toarray(),K.toarray(),atol=1e-8)
    assert er.contact_energy==pytest.approx(r.contact_energy,rel=1e-12)
    for cell in model.mesh.elements.values():cell.node_ids=tuple(reversed(cell.node_ids))
    rf,rK,rr=BeamPairEndManager(model,definition,settings).evaluate(u)
    np.testing.assert_allclose(rf,f,atol=1e-10);np.testing.assert_allclose(rK.toarray(),K.toarray(),atol=1e-8)
    assert rr.contact_energy==pytest.approx(r.contact_energy,rel=1e-12)
    angle=.37;Q=np.array(((np.cos(angle),-np.sin(angle),0.),(np.sin(angle),np.cos(angle),0.),(0.,0.,1.)))
    shift=np.array((.7,-.4,.2));rotated=np.zeros_like(u)
    for node in model.mesh.nodes.values():
        node.x,node.y,node.z=Q@node.coords()+shift
        dofs=model.mesh.dof_manager.get_node_dofs(node.id)[:3]
        rotated[dofs]=Q@u[dofs]
    tf,tK,tr=BeamPairEndManager(model,definition,settings).evaluate(rotated)
    assert tr.contact_energy==pytest.approx(r.contact_energy,rel=1e-10)
    for node in model.mesh.nodes:
        dofs=model.mesh.dof_manager.get_node_dofs(node)[:3]
        np.testing.assert_allclose(tf[dofs],Q@f[dofs],rtol=1e-10,atol=1e-10)
    assert np.linalg.norm(np.array(tr.beam_a_moment)+tr.beam_b_moment)<1e-6


def test_unselected_middle_node_junction_is_not_a_physical_cap():
    model,definition,settings,_=fixture()
    model.add_node(5,.012,-.1,0.);model.add_node(6,.012,.1,0.)
    model.add_element(3,QuadraticBeamElement(3,(5,3,6),'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    manager=BeamPairEndManager(model,definition,settings)
    assert manager.ends==((0,1),(1,))
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(2)[0]]=.0021
    u[model.mesh.dof_manager.get_node_dofs(3)[0]]=-.0021
    with pytest.raises(ValueError,match='nonphysical'):
        manager.evaluate(u)


def test_hidden_fractional_double_ownership_root_and_open_bypass():
    from fractions import Fraction as F
    from anyfesolver.general_contact.beam_pair_end_geometry import admissible_endpoint_path
    b=((0,0,-1),(0,0,1));root=F(1,3);height=F(1,100)
    a0=((0,height,-1),(-root,height,0));a1=((0,height,-1),(1-root,height,0))
    assert endpoint_feature(a0,b,1,F(1,50)).gap_sign>0
    assert endpoint_feature(a1,b,1,F(1,50)).gap_sign>0
    assert not admissible_endpoint_path(a0,a1,b,b,(1,),(),F(1,50),8)
    assert admissible_endpoint_path(a0,a1,b,b,(1,),(),0,0)


def test_failed_candidate_has_no_state_and_public_example():
    model,definition,settings,_=fixture()
    manager=BeamPairEndManager(model,definition,settings)
    zero=np.zeros(model.mesh.dof_manager.total_dofs)
    with pytest.raises(ValueError,match='invalid'):manager.evaluate(np.full_like(zero,np.nan))
    f,K,r=manager.evaluate(zero)
    np.testing.assert_array_equal(f,0.);assert r.contact_energy==0.
    from examples.general_contact_beam_pair_ends import run
    result=run()
    assert result.info['general_contact']['profile']==PROFILE


@pytest.mark.parametrize('cancel',(False,True))
def test_candidate_exception_and_cancellation_restore_two_body_model(cancel,monkeypatch):
    from anyfesolver import CancellationToken,SolveCancelled
    model,definition,settings,load=fixture()
    args=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=25,
        tolerance=1e-10,emit_restart_checkpoint=True)
    token=CancellationToken();armed=[False];raised=[False];original=BeamPairEndManager.evaluate
    def interrupt(event):
        if event.event_type=='nonlinear_static_step' and event.metadata['load_factor']>=.5:
            if cancel:token.cancel('two body end interruption')
            else:armed[0]=True
    def fail_after_candidate(self,*values,**options):
        candidate=original(self,*values,**options)
        if armed[0] and not raised[0]:
            raised[0]=True
            raise RuntimeError('two body end interruption')
        return candidate
    with monkeypatch.context() as patch:
        if not cancel:patch.setattr(BeamPairEndManager,'evaluate',fail_after_candidate)
        with pytest.raises(SolveCancelled if cancel else RuntimeError,match='two body end interruption'):
            solve_static_nonlinear(model,load,**args,progress_callback=interrupt,cancellation_token=token)
    if not cancel:assert raised[0]
    rerun=solve_static_nonlinear(model,load,**args)
    fresh,_,_,_=fixture();reference=solve_static_nonlinear(fresh,load,**args)
    assert rerun.status==reference.status=='completed'
    np.testing.assert_array_equal(rerun.displacements,reference.displacements)
    assert rerun.to_restart_checkpoint()==reference.to_restart_checkpoint()


@pytest.mark.parametrize('ownership_tie',(False,True))
def test_closed_active_endpoint_ties_reject(ownership_tie):
    model,definition,settings,_=fixture(True)
    manager=BeamPairEndManager(model,definition,settings)
    if ownership_tie:
        source=np.array(((-.2,.0198,0.),(0.,.0198,0.)))
    else:source=np.array(((-.2,.2198,0.),(-.2,.0198,0.)))
    target=np.array(((-.2,0.,0.),(.2,0.,0.)))
    with pytest.raises(ValueError,match='tie'):
        manager._endpoint_work((source,target))


def test_manager_rejects_plateau_path_at_an_unselected_material_junction():
    from anyfesolver.general_contact.beam_pair_end_geometry import admissible_endpoint_path
    model,definition,settings,_=fixture()
    positions=((- .003,-.1,.019999),(-.003,.1,.019999),(0.,0.,0.),(.2,0.,0.))
    for node,p in zip(model.mesh.nodes.values(),positions):node.x,node.y,node.z=p
    model.add_node(5,0.,-.1,0.);model.add_node(6,0.,.1,0.)
    model.add_element(3,QuadraticBeamElement(3,(5,3,6),'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    manager=BeamPairEndManager(model,definition,settings)
    first=np.zeros(model.mesh.dof_manager.total_dofs);last=first.copy()
    for node in (1,2):
        first[model.mesh.dof_manager.get_node_dofs(node)[0]]=.003-1e-6
        last[model.mesh.dof_manager.get_node_dofs(node)[0]]=.003+1e-6
    a0,b0=manager._positions(first);a1,b1=manager._positions(last)
    assert manager.ends==((0,1),(1,))
    assert admissible_endpoint_path(a0,a1,b0,b1,*manager.ends,.02,16)
    assert manager.evaluate(first)[2].gap_tolerance_met
    assert manager.evaluate(last)[2].gap_tolerance_met
    assert not manager.swept_increment_admissible(first,last,0.,1.)
