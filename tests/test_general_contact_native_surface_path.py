"""Analytic circular-arc counterexamples to endpoint-only history admission."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum

from anyfesolver.general_contact.native_surface_path import (
    certify_material_path, material_path_pose, locate_affine_contact_event,
)
from test_general_contact_native_surface_s3_friction import _case


def _start(pose, angle, z, dx=0.):
    nodes=pose.positions.copy(); nodes[:, 2]=z; nodes[:, 0]+=dx
    cells=pose.cell_rotations.copy(); cells[0]=Rotation.from_rotvec((angle,0,0)).as_matrix()
    return replace(pose, positions=nodes, cell_rotations=cells)


def _touch(pair, pose):
    high,low=split_sum((float(pair.target.adapter.reference_positions[0,2]),pair.source.radius))
    pose=_start(pose,0.,high)
    lows=pose.position_low.copy(); lows[:,2]=low
    return replace(pose,position_low=lows)


@pytest.mark.parametrize('closed,z',[(True,-.02),(False,.02)])
def test_certifies_persistent_rotating_material_point(closed,z):
    _,pair,pose,_=_case()
    pose=_start(pose,0.,z)
    result=certify_material_path(pair,pose,np.zeros((3,3)),np.array(((.1,0,0),(0,0,0))),
        np.zeros(18),closed=closed)
    assert result.certified
    assert 0<result.intervals<=256


def test_hidden_opening_between_closed_endpoints_is_rejected():
    _,pair,pose,_=_case()
    pose=_start(pose,2.,-.06)
    # g(theta)=-.02-.03*cos(theta); endpoints negative, theta=pi positive.
    assert -.02-.03*np.cos(2.)<0 and -.02-.03*np.cos(4.)<0
    assert -.02-.03*np.cos(np.pi)>0
    result=certify_material_path(pair,pose,np.zeros((3,3)),np.array(((2.,0,0),(0,0,0))),
        np.zeros(18),closed=True)
    assert not result.certified
    assert 'opening/closure' in result.reason


def test_hidden_facet_exit_between_interior_endpoints_is_rejected():
    _,pair,pose,_=_case()
    pose=_start(pose,.6,-.1,.48)
    # Target diagonal is x+y=0. x=-.02, y=.03*sin(theta).
    assert -.02+.03*np.sin(.6)<0 and -.02+.03*np.sin(2.6)<0
    assert -.02+.03*np.sin(np.pi/2)>0
    result=certify_material_path(pair,pose,np.zeros((3,3)),np.array(((2.,0,0),(0,0,0))),
        np.zeros(18),closed=True)
    assert not result.certified
    assert 'facet exit' in result.reason


def test_unresolved_budget_and_rotation_chart_fail_explicitly():
    _,pair,pose,_=_case()
    result=certify_material_path(pair,pose,np.zeros((3,3)),np.array(((1.,0,0),(0,0,0))),
        np.zeros(18),closed=True,maximum_intervals=1)
    assert not result.certified
    assert result.intervals==1
    with pytest.raises(ValueError,match='rotation chart'):
        material_path_pose(pose,np.zeros((3,3)),np.array(((2*np.pi,0,0),(0,0,0))),np.zeros(18),1.)
    np.testing.assert_array_equal(pose.cell_rotations,np.tile(np.eye(3),(2,1,1)))


def test_interior_target_collapse_is_not_hidden_by_valid_endpoints():
    _,pair,pose,_=_case()
    corners=pair.target.adapter.current_positions(pose.shell_displacement)
    # A 180-degree in-plane endpoint rotation collapses an affine triangle at half time.
    delta=np.zeros(18)
    for i in range(3):delta[6*i:6*i+2]=-2*corners[i,:2]
    result=certify_material_path(pair,pose,np.zeros((3,3)),np.zeros((2,3)),delta,closed=True)
    assert not result.certified
    assert 'orientation/area' in result.reason


def test_path_does_not_admit_point_law_excluded_near_rim_strip():
    _,pair,pose,_=_case()
    pose=_start(pose,0.,-.02,.5-1e-13)
    result=certify_material_path(pair,pose,np.zeros((3,3)),np.zeros((2,3)),np.zeros(18),closed=True)
    assert not result.certified


def test_affine_target_tilt_and_edge_contraction_with_rotating_source():
    _,pair,pose,_=_case()
    omega=np.array((.002,.003,.01)); rotation=Rotation.from_rotvec(omega).as_matrix()
    pivot=np.array((-.5,0.,-.04)); shift=np.array((.001,-.002,.001))
    corners=pair.target.adapter.current_positions(pose.shell_displacement)
    delta=np.zeros(18)
    for i in range(3):
        delta[6*i:6*i+3]=rotation@(corners[i]-pivot)+pivot+shift-corners[i]
    translations=(pose.positions-pivot)@rotation.T+pivot+shift-pose.positions
    angular=np.tile(omega,(2,1))
    value=certify_material_path(pair,pose,translations,angular,delta,closed=True)
    assert value.certified
    # For an affine chord between equal-length rotated edges, its squared
    # length has its minimum exactly at half time. This exercises that bound.
    edge=corners[1]-corners[0]; last=rotation@edge
    assert np.linalg.norm(.5*(edge+last))<np.linalg.norm(edge)
    # Independent Rodrigues/chord evaluation is a diagnostic, not the certificate.
    arm=np.array((0.,0.,-.03)); base=np.array((-.5,0.,-.02))
    for alpha in (0.,.25,.5,.75,1.):
        vertices=corners+alpha*np.array([delta[6*i:6*i+3] for i in range(3)])
        n=np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]); n/=np.linalg.norm(n)
        x=base+alpha*.5*(translations[0]+translations[1])+Rotation.from_rotvec(alpha*omega).apply(arm)
        assert n@(x-vertices[0])<-.009
    changed=certify_material_path(pair,pose,translations,angular,delta,closed=False)
    assert not changed.certified
    assert value.path_sha256!=changed.path_sha256


@pytest.mark.parametrize('z,vertical,kind,fraction',[
    (0.,-.02,'first_touch',.5), (-.02,.04,'release',.25),
])
def test_affine_event_includes_target_motion_and_preserves_geometric_path(z,vertical,kind,fraction):
    _,pair,pose,_=_case()
    pose=_start(pose,0.,z)
    translations=np.tile((.002,.001,vertical+.003),(3,1))
    shell=np.zeros(18)
    for i in range(3):shell[6*i:6*i+3]=(.001,-.002,.003)
    event=locate_affine_contact_event(pair,pose,translations,np.zeros((2,3)),shell)
    assert event.transition==kind
    assert event.fraction==pytest.approx(fraction,abs=1e-14)
    point=pair.source.evaluate(event.pose.positions,event.pose.position_low,event.pose.cell_rotations).surface_position
    target=pair.target.physical_pose(event.pose.shell_displacement)
    assert (point-np.asarray(target.position))@np.asarray(target.material_frame)[:,2]==pytest.approx(0.,abs=2e-17)
    np.testing.assert_allclose(event.pose.shell_displacement,fraction*shell,atol=1e-16)
    assert len(event.path_sha256)==64
    # Geometric proposals do not mutate/commit mechanical or friction state.
    np.testing.assert_array_equal(pose.shell_displacement,np.zeros(18))


def test_exact_touch_affine_activation_and_release_do_not_stall():
    _,pair,pose,_=_case()
    pose=_touch(pair,pose)
    for vertical,closed in ((-.005,True),(.005,False)):
        translation=np.tile((0.,0.,vertical),(3,1))
        event=locate_affine_contact_event(pair,pose,translation,np.zeros((2,3)),np.zeros(18))
        assert event.fraction==0.
        result=certify_material_path(pair,pose,translation,np.zeros((2,3)),np.zeros(18),closed=closed)
        assert result.certified
        assert result.intervals==1


def test_affine_event_rejects_rotation_facet_exit_and_missing_transition():
    _,pair,pose,_=_case()
    with pytest.raises(ValueError,match='affine physical'):
        locate_affine_contact_event(pair,pose,np.zeros((3,3)),np.array(((.1,0,0),(0,0,0))),np.zeros(18))
    with pytest.raises(ValueError,match='no unique'):
        locate_affine_contact_event(pair,pose,np.zeros((3,3)),np.zeros((2,3)),np.zeros(18))
    with pytest.raises(ValueError,match='interior ownership'):
        locate_affine_contact_event(pair,pose,np.tile((1.,0.,.1),(3,1)),np.zeros((2,3)),np.zeros(18))


def test_certified_contribution_uses_committed_history_and_rejects_hidden_opening():
    _,pair,pose,history=_case()
    pose=_touch(pair,pose)
    translation=np.tile((.002,0.,-.01),(3,1))
    response=pair.evaluate_path(pose,history,translation,np.zeros((2,3)),np.zeros(18))
    assert response.contribution.trial.branch=='stick'
    assert response.contribution.trial.traction[0]==pytest.approx(-2.,abs=1e-12)
    assert response.contribution.normal_pressure_Pa==pytest.approx(10.,abs=1e-12)
    assert response.certified_intervals==1
    assert history.partner_key is None
    # A second disposable candidate uses the same physical origin, not the
    # preceding trial's elastic slip. This is the Newton transaction boundary.
    response2=pair.evaluate_path(pose,history,.5*translation,np.zeros((2,3)),np.zeros(18))
    assert response2.contribution.trial.traction[0]==pytest.approx(-1.,abs=1e-12)
    assert response.path_sha256!=response2.path_sha256
    hidden=_start(pose,2.,-.06)
    with pytest.raises(ValueError,match='requires subdivision'):
        pair.evaluate_path(hidden,history,np.zeros((3,3)),np.array(((2.,0,0),(0,0,0))),np.zeros(18))


def test_certified_contribution_release_and_recontact_reset_traction():
    _,pair,pose,empty=_case()
    touch=_touch(pair,pose)
    loaded=pair.evaluate_path(touch,empty,np.tile((.002,0.,-.01),(3,1)),
                             np.zeros((2,3)),np.zeros(18))
    state=loaded.contribution.trial.candidate_state
    # End exactly at the known original contact plane. Release occurs at zero
    # pressure, with stored elastic energy explicitly reported by the law.
    released=pair.evaluate_path(loaded.configuration,state,np.tile((0.,0.,.01),(3,1)),
                               np.zeros((2,3)),np.zeros(18))
    assert released.contribution.trial.branch=='open'
    assert released.contribution.trial.released_storage_density==pytest.approx(.002,rel=1e-10)
    assert released.contribution.trial.candidate_state.partner_key is None
    recontact=pair.evaluate_path(released.configuration,released.contribution.trial.candidate_state,
        np.tile((0.,0.,-.01),(3,1)),np.zeros((2,3)),np.zeros(18))
    np.testing.assert_array_equal(recontact.contribution.trial.traction,(0.,0.,0.))


def test_rounded_zero_does_not_override_compensated_affine_gap_sign():
    _,pair,pose,_=_case()
    # Ordinary double evaluation rounds this station to the skin, but the
    # represented native base+arm lies slightly above it. It is not closed.
    pose=_start(pose,0.,-.01)
    point=pair.source.evaluate(pose.positions,pose.position_low,pose.cell_rotations).surface_position
    assert point[2]==pair.target.adapter.reference_positions[0,2]
    closed=certify_material_path(pair,pose,np.zeros((3,3)),np.zeros((2,3)),np.zeros(18),closed=True)
    opened=certify_material_path(pair,pose,np.zeros((3,3)),np.zeros((2,3)),np.zeros(18),closed=False)
    assert not closed.certified
    assert opened.certified
