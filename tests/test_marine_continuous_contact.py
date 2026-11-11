"""Opt-in continuous contact law and actual fixed-step coupled integration."""
from dataclasses import replace
import copy

import numpy as np
import pytest

from anyfesolver.marine_seabed import (SeabedContact, SeabedContactError,
    seabed_contact_force, API_VERSION, CONTINUOUS_API_VERSION)
from anyfesolver.marine_rigid import RigidBody6D
from anyfesolver.marine_mooring import (FixedEndpoint, DynamicLineNodeEndpoint,
    MooringNodeState, MooringLine, MooringLineAssembler)
from anyfesolver.marine_mechanical import (MarineMechanicalModel,
    MarineMechanicalRunSpec, solve_marine_mechanical, model_to_dict, model_from_dict)


def fixture(q0=.0002):
    body=RigidBody6D('marker',5.,(0.,0.,0.),np.diag([2.,3.,4.]))
    line=MooringLine('slack-contact',
        (FixedEndpoint((-1.,0.,-2.)),DynamicLineNodeEndpoint(0),FixedEndpoint((1.,0.,-2.))),
        np.full(2,2.),np.full(2,100.),np.zeros(2),np.full(2,2.),np.zeros(2),contact_diameters=np.full(2,.2))
    bed=SeabedContact(-2.1,1e5,100.,activation_depth_m=1e-4)
    assembler=MooringLineAssembler((body,),(line,),1,(0.,0.,0.),1000.,0.,seabed_contact=bed)
    model=MarineMechanicalModel((body,),mooring=assembler)
    nodes=MooringNodeState([[0.,0.,-2.+q0]],[[0.,0.,-.1]],[[123.,123.,123.]])
    return model,{'marker':body.state(position=(0.,0.,-1.))},nodes


def test_explicit_v2_and_exact_legacy_constructor_serialization():
    legacy=SeabedContact(-2.,1000.,20.)
    assert legacy.to_dict()==dict(api_version=API_VERSION,elevation=-2.,normal_stiffness=1000.,normal_damping=20.)
    assert SeabedContact.from_dict(legacy.to_dict())==legacy
    new=replace(legacy,activation_depth_m=1e-4)
    data=new.to_dict()
    assert data['api_version']==CONTINUOUS_API_VERSION and data['activation_depth_m']==1e-4
    assert SeabedContact.from_dict(copy.deepcopy(data))==new
    model,_,_=fixture()
    assert model_to_dict(model_from_dict(model_to_dict(model)))==model_to_dict(model)


@pytest.mark.parametrize('bad',[0.,-1.,True,'1e-4',float('nan'),float('inf')])
def test_v2_depth_is_positive_finite_physical_input(bad):
    with pytest.raises(SeabedContactError):SeabedContact(-2.,1000.,20.,activation_depth_m=bad)
    data=SeabedContact(-2.,1000.,20.,activation_depth_m=1e-4).to_dict()
    data['activation_depth_m']=bad
    with pytest.raises(SeabedContactError):SeabedContact.from_dict(data)


@pytest.mark.parametrize('bad',['missing','unknown','legacy-with-depth'])
def test_versions_and_fields_fail_closed(bad):
    data=SeabedContact(-2.,1000.,20.,activation_depth_m=1e-4).to_dict()
    if bad=='missing':del data['activation_depth_m']
    elif bad=='unknown':data['api_version']='future-contact'
    else:data['api_version']=API_VERSION
    with pytest.raises(SeabedContactError):SeabedContact.from_dict(data)


def test_independent_ramp_saturated_clipping_and_passive_power_values():
    bed=SeabedContact(0.,1e5,100.,activation_depth_m=1e-4)
    penetration=np.array([5e-5,5e-5,5e-5,2e-4,0.,-1e-4])
    velocity=np.array([-.2,.05,.2,-.2,-.2,-.2])
    p=np.zeros((6,3));p[:,2]=.1-penetration
    v=np.zeros_like(p);v[:,2]=velocity
    actual=seabed_contact_force(bed,.2,2.,p,v)
    np.testing.assert_allclose(actual.force[:,2],[6.,1.,0.,16.,0.,0.],atol=3e-11,rtol=0.)
    np.testing.assert_allclose(actual.dissipation,[.8,.05,.4,1.6,0.,0.],atol=1e-11,rtol=0.)
    np.testing.assert_allclose(actual.force[:,2]*velocity-actual.elastic_force[:,2]*velocity+actual.dissipation,0.,atol=1e-12)
    assert np.min(actual.dissipation)>=0 and np.min(actual.force[:,2])>=0
    assert actual.potential_energy[2]>0 and actual.force[2,2]==0  # retained clipping storage


@pytest.mark.parametrize('penetration,velocity',[(5e-5,-.02),(5e-5,.02),(2e-4,-.02),(5e-5,.2)])
def test_active_tangents_against_independent_force_differences(penetration,velocity):
    bed=SeabedContact(0.,1e5,100.,activation_depth_m=1e-4)
    p=np.array([[0.,0.,.1-penetration]]);v=np.array([[0.,0.,velocity]])
    actual=seabed_contact_force(bed,.2,2.,p,v)
    for target,field in ((actual.stiffness,p),(actual.damping,v)):
        h=1e-9;plus=field.copy();minus=field.copy();plus[0,2]+=h;minus[0,2]-=h
        kwargs=(plus,v) if field is p else (p,plus)
        a=seabed_contact_force(bed,.2,2.,*kwargs).force
        kwargs=(minus,v) if field is p else (p,minus)
        b=seabed_contact_force(bed,.2,2.,*kwargs).force
        np.testing.assert_allclose(target[0,:,2],-(a-b)[0]/(2*h),rtol=1e-7,atol=1e-7)
    assert actual.stiffness[0,2,2]>=0 and actual.damping[0,2,2]>=0


def test_corner_conventions_and_force_continuity():
    bed=SeabedContact(-2.,1.,1.,activation_depth_m=1.)
    # Exactly representable surface, ramp corner and clipped raw-force corner.
    p=[[0.,0.,-1.],[0.,0.,-2.],[0.,0.,-1.5]];v=[[0.,0.,-1.],[0.,0.,-1.],[0.,0.,1.]]
    r=seabed_contact_force(bed,2.,1.,p,v)
    np.testing.assert_array_equal(r.force[:,2],[0.,4.,0.])
    np.testing.assert_array_equal(r.stiffness[:,2,2],[0.,2.,0.])
    np.testing.assert_array_equal(r.damping[:,2,2],[0.,2.,0.])
    for gap in (-1e-8,1e-8):
        near=seabed_contact_force(bed,2.,1.,[[0.,0.,-1.+gap]],[[0.,0.,-1.]])
        assert near.force[0,2]<5e-8


def test_original_damped_landing_now_closes_fixed_step_boundary():
    model,states,nodes=fixture()
    result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.0005,12),initial_nodes=nodes)
    assert result.status=='completed' and len(result.accepted)==13
    assert result.accepted[0].nodes.accelerations[0,2]==0.
    assert any(frame.nodes.positions[0,2]<-2. for frame in result.accepted)
    for frame in result.accepted:
        assert frame.residual_norm<=1e-10
        assert frame.mooring_diagnostics['contact']['dissipation']>=0
        np.testing.assert_allclose(frame.positions['marker'],[0.,0.,-1.],atol=1e-14)
    assert np.all(nodes.accelerations==123.)


@pytest.mark.parametrize('epsilon',[1e-8,5e-8,1e-7])
def test_actual_finite_no_root_interval_has_continuous_law_roots(epsilon):
    model,states,nodes=fixture(q0=.00005-epsilon)
    result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.0005,1),initial_nodes=nodes)
    assert result.status=='completed'
    assert result.accepted[-1].nodes.positions[0,2]<-2.
    assert result.accepted[-1].nodes.accelerations[0,2]>0.
    assert result.accepted[-1].residual_norm<=1e-10


def test_v2_cancel_preserves_accepted_prefix_and_intent():
    model,states,nodes=fixture();saved=model_to_dict(model);stop=False
    def progress(frame):
        nonlocal stop
        if frame.step==1:stop=True
    result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.0005,12),
        initial_nodes=nodes,progress=progress,cancel=lambda:stop)
    assert result.status=='cancelled' and [frame.step for frame in result.accepted]==[0,1]
    assert model_to_dict(model)==saved
    with pytest.raises(ValueError):result.accepted[-1].nodes.positions[0,2]=0.
