"""Independent scalar controls for the internal material-frame Coulomb law."""

import numpy as np
import pytest

from anyfesolver.general_contact.friction_point import (
    FrictionPointState, initial_friction_point,
    objective_relative_tangent_increment, trial_coulomb_point,
    trial_coulomb_point_directional,
)


FRAME = np.eye(3)
PRESSURE = 2.0e6
MU = 0.3
STIFFNESS = 1.0e9
PARTNER = "shell:17:material-point:0.25,0.5"


def test_frame_validation_reuse_rechecks_changed_bytes_and_does_not_share_arrays():
    from anyfesolver.general_contact.friction_point import _frame3, _validate_frame_bytes
    _validate_frame_bytes.cache_clear()
    frame=np.eye(3)
    assert _frame3(frame) is frame
    assert _frame3(frame.copy()) is not frame
    assert _validate_frame_bytes.cache_info().hits == 1
    frame[0,0]=1.01
    with pytest.raises(ValueError,match='orthonormal'):
        _frame3(frame)
    with pytest.raises(ValueError,match='finite'):
        _frame3(np.full((3,3),np.nan))
    with pytest.raises(ValueError,match='right-handed'):
        _frame3(np.diag((-1.,1.,1.)))
    assert np.array_equal(_frame3(np.eye(3)),np.eye(3))


def trial(committed, increment, *, frame=FRAME, pressure=PRESSURE,
          partner=PARTNER):
    return trial_coulomb_point(
        committed, material_frame=frame, relative_increment=increment,
        normal_pressure=pressure, friction_coefficient=MU,
        tangential_stiffness=STIFFNESS, partner_key=partner,
    )


def test_slider_stick_slip_and_reversal_from_same_committed_origin():
    origin = initial_friction_point(FRAME)
    stick = trial(origin, (0.0002, 0.0, 0.0))
    assert stick.branch == "stick"
    np.testing.assert_allclose(stick.traction, (-0.2e6, 0.0, 0.0))
    assert stick.stored_energy_density == pytest.approx(20.0)
    assert stick.dissipated_increment_density == 0.0
    assert origin.partner_key is None and origin.elastic_slip == (0.0, 0.0, 0.0)

    # The slip threshold is mu*p/k = 0.0006 m. This second increment takes
    # the same material point to 0.0012 m of total relative displacement.
    slip = trial(stick.candidate_state, (0.001, 0.0, 0.0))
    assert slip.branch == "slip"
    np.testing.assert_allclose(slip.traction, (-0.6e6, 0.0, 0.0))
    np.testing.assert_allclose(slip.plastic_slip_increment, (0.0006, 0.0, 0.0))
    assert slip.dissipated_increment_density == pytest.approx(360.0)
    assert slip.candidate_state.cumulative_plastic_slip == pytest.approx(0.0006)
    assert slip.candidate_state.cumulative_dissipation_density == pytest.approx(360.0)
    assert slip.stored_energy_density == pytest.approx(180.0)

    reverse_stick = trial(slip.candidate_state, (-0.0004, 0.0, 0.0))
    assert reverse_stick.branch == "stick"
    np.testing.assert_allclose(reverse_stick.traction, (-0.2e6, 0.0, 0.0))
    assert reverse_stick.dissipated_increment_density == 0.0
    reverse_slip = trial(slip.candidate_state, (-0.0014, 0.0, 0.0))
    assert reverse_slip.branch == "slip"
    np.testing.assert_allclose(reverse_slip.traction, (0.6e6, 0.0, 0.0))
    assert reverse_slip.dissipated_increment_density == pytest.approx(120.0)
    assert slip.candidate_state.cumulative_dissipation_density == pytest.approx(360.0)


def test_rigid_material_spin_transports_traction_without_work():
    origin = initial_friction_point(FRAME)
    committed = trial(origin, (0.0002, 0.0001, 0.0)).candidate_state
    rotation = np.array(((0.0, -1.0, 0.0),
                         (1.0, 0.0, 0.0),
                         (0.0, 0.0, 1.0)))
    rotated = trial(committed, (0.0, 0.0, 0.0), frame=rotation)
    assert rotated.branch == "stick"
    np.testing.assert_allclose(rotated.traction, (0.1e6, -0.2e6, 0.0))
    assert rotated.stored_energy_density == pytest.approx(25.0)
    assert rotated.dissipated_increment_density == 0.0
    assert rotated.released_storage_density == 0.0
    angle = 0.37
    c, s = np.cos(angle), np.sin(angle)
    tilted = np.array(((c, 0.0, s),
                       (0.0, 1.0, 0.0),
                       (-s, 0.0, c)))
    normal_rotated = trial(committed, (0.0, 0.0, 0.0), frame=tilted)
    np.testing.assert_allclose(normal_rotated.traction,
                               tilted @ np.array((-0.2e6, -0.1e6, 0.0)))
    assert normal_rotated.stored_energy_density == pytest.approx(25.0)
    assert normal_rotated.dissipated_increment_density == 0.0


def test_opening_and_changed_partner_clear_elastic_traction():
    committed = trial(initial_friction_point(FRAME),
                      (0.0002, 0.0, 0.0)).candidate_state
    opened = trial(committed, (0.01, 0.0, 0.0), pressure=0.0,
                   partner=None)
    assert opened.branch == "open"
    assert opened.traction == (0.0, 0.0, 0.0)
    assert opened.stored_energy_density == 0.0
    assert opened.released_storage_density == pytest.approx(20.0)
    assert opened.candidate_state.partner_key is None
    recontact = trial(opened.candidate_state, (0.0001, 0.0, 0.0))
    np.testing.assert_allclose(recontact.traction, (-0.1e6, 0.0, 0.0))
    other = trial(committed, (0.0001, 0.0, 0.0), partner="new-face")
    np.testing.assert_allclose(other.traction, (-0.1e6, 0.0, 0.0))
    assert other.released_storage_density == pytest.approx(20.0)


@pytest.mark.parametrize("increment,branch", [
    ((0.0001, 0.0002, 0.003), "stick"),
    ((0.0010, 0.0002, 0.003), "slip"),
])
def test_smooth_tangential_and_pressure_derivatives(increment, branch):
    origin = initial_friction_point(FRAME)
    base = trial(origin, increment)
    assert base.branch == branch
    direction = np.array((0.3, -0.7, 0.2))
    step = 1.0e-8
    plus = trial(origin, np.asarray(increment) + step*direction)
    minus = trial(origin, np.asarray(increment) - step*direction)
    assert plus.branch == minus.branch == branch
    finite = (np.asarray(plus.traction)-minus.traction)/(2.0*step)
    tangent = np.asarray(base.displacement_tangent)
    np.testing.assert_allclose(tangent @ direction, finite,
                               rtol=1.0e-7, atol=1.0e-3)
    pressure_step = 1.0
    plus_p = trial(origin, increment, pressure=PRESSURE+pressure_step)
    minus_p = trial(origin, increment, pressure=PRESSURE-pressure_step)
    finite_p = (np.asarray(plus_p.traction)-minus_p.traction)/(2.0*pressure_step)
    np.testing.assert_allclose(base.pressure_derivative, finite_p,
                               rtol=1.0e-7, atol=1.0e-10)
    if branch == "stick":
        np.testing.assert_allclose(tangent, np.diag((-STIFFNESS, -STIFFNESS, 0.0)))


def test_changed_normal_load_slips_and_invalid_frames_fail():
    origin = initial_friction_point(FRAME)
    committed = trial(origin, (0.0005, 0.0, 0.0)).candidate_state
    reduced = trial(committed, (0.0, 0.0, 0.0), pressure=1.0e6)
    assert reduced.branch == "slip"
    np.testing.assert_allclose(reduced.traction, (-0.3e6, 0.0, 0.0))
    assert reduced.dissipated_increment_density == pytest.approx(60.0)
    with pytest.raises(ValueError, match="right-handed"):
        initial_friction_point(np.diag((1.0, 1.0, -1.0)))
    with pytest.raises(ValueError, match="open friction point"):
        FrictionPointState(tuple(map(tuple, FRAME)), (0.0001, 0.0, 0.0))
    with pytest.raises(ValueError, match="nonnegative"):
        trial(origin, (0.0, 0.0, 0.0), pressure=-1.0)
    with pytest.raises(ValueError, match="partner_key"):
        trial(origin, (0.0, 0.0, 0.0), partner=None)


def test_objective_pair_motion_removes_rotation_of_a_finite_open_gap():
    angle = 0.47
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array(((c, 0.0, s),
                         (0.0, 1.0, 0.0),
                         (-s, 0.0, c)))
    translation = np.array((0.3, -0.2, 0.7))
    source = np.array((0.0, 0.0, 0.02))
    target = np.zeros(3)
    common = objective_relative_tangent_increment(
        source_committed=source, target_committed=target,
        source_candidate=rotation @ source + translation,
        target_candidate=translation,
        target_frame_committed=FRAME,
        target_frame_candidate=rotation,
    )
    np.testing.assert_allclose(common, 0.0, atol=1.0e-15)
    # Directly subtracting source/target displacements would create false
    # tangential slip from the rotation of the 20 mm normal opening.
    naive = (rotation @ source) - source
    projector = rotation[:, :2] @ rotation[:, :2].T
    assert np.linalg.norm(projector @ naive) > 0.005

    real_material_slip = np.array((0.001, -0.0004, 0.0))
    moving = objective_relative_tangent_increment(
        source_committed=source, target_committed=target,
        source_candidate=rotation @ (source + real_material_slip) + translation,
        target_candidate=translation,
        target_frame_committed=FRAME,
        target_frame_candidate=rotation,
    )
    np.testing.assert_allclose(moving, rotation @ real_material_slip,
                               atol=1.0e-15)


def test_objective_pair_motion_equal_and_independent_translations():
    old_source = np.array((0.3, 0.1, 0.02))
    old_target = np.array((0.3, 0.1, 0.0))
    translation = np.array((0.002, -0.003, 0.004))
    same = objective_relative_tangent_increment(
        source_committed=old_source, target_committed=old_target,
        source_candidate=old_source + translation,
        target_candidate=old_target + translation,
        target_frame_committed=FRAME, target_frame_candidate=FRAME,
    )
    np.testing.assert_allclose(same, 0.0, atol=1.0e-15)
    target_only = objective_relative_tangent_increment(
        source_committed=old_source, target_committed=old_target,
        source_candidate=old_source,
        target_candidate=old_target + (0.001, -0.002, 0.003),
        target_frame_committed=FRAME, target_frame_candidate=FRAME,
    )
    np.testing.assert_allclose(target_only, (-0.001, 0.002, 0.0),
                               atol=1.0e-15)
    with pytest.raises(ValueError, match="orthonormal"):
        objective_relative_tangent_increment(
            source_committed=old_source, target_committed=old_target,
            source_candidate=old_source, target_candidate=old_target,
            target_frame_committed=FRAME,
            target_frame_candidate=np.diag((2.0, 1.0, 1.0)),
        )


@pytest.mark.parametrize("increment,branch", [
    ((0.00005, -0.00002, 0.0), "stick"),
    ((0.001, -0.0001, 0.0), "slip"),
])
def test_rotating_committed_spring_traction_directional(increment, branch):
    committed = trial(initial_friction_point(FRAME),
                      (0.00025, 0.0, 0.0)).candidate_state

    def rotation(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array(((c, -s, 0.0),
                         (s, c, 0.0),
                         (0.0, 0.0, 1.0)))

    angle = 0.21
    frame = rotation(angle)
    generator = np.array(((0.0, -1.0, 0.0),
                          (1.0, 0.0, 0.0),
                          (0.0, 0.0, 0.0)))
    d_frame = generator @ frame
    d_increment = np.array((0.0003, -0.0002, 0.0001))
    d_pressure = 2.0e5
    analytic = trial_coulomb_point_directional(
        committed, material_frame=frame, material_frame_variation=d_frame,
        relative_increment=increment,
        relative_increment_variation=d_increment,
        normal_pressure=PRESSURE, pressure_variation=d_pressure,
        friction_coefficient=MU, tangential_stiffness=STIFFNESS,
        partner_key=PARTNER,
    )
    assert trial(committed, increment, frame=frame).branch == branch
    for step in (1.0e-6, 3.0e-7):
        plus = trial(committed, np.asarray(increment) + step*d_increment,
                     frame=rotation(angle + step),
                     pressure=PRESSURE + step*d_pressure)
        minus = trial(committed, np.asarray(increment) - step*d_increment,
                      frame=rotation(angle - step),
                      pressure=PRESSURE - step*d_pressure)
        assert plus.branch == minus.branch == branch
        finite = (np.asarray(plus.traction)-minus.traction)/(2.0*step)
        np.testing.assert_allclose(analytic, finite,
                                   rtol=1.0e-6, atol=1.0e-3)


def test_traction_directional_rejects_nonsmooth_activation_and_yield():
    origin = initial_friction_point(FRAME)
    data = dict(
        material_frame=FRAME, material_frame_variation=np.zeros((3, 3)),
        relative_increment=(0.0006, 0.0, 0.0),
        relative_increment_variation=(1.0, 0.0, 0.0),
        normal_pressure=PRESSURE, pressure_variation=0.0,
        friction_coefficient=MU, tangential_stiffness=STIFFNESS,
        partner_key=PARTNER,
    )
    with pytest.raises(ValueError, match="stick/slip equality"):
        trial_coulomb_point_directional(origin, **data)
    data.update(normal_pressure=0.0, pressure_variation=1.0,
                partner_key=None)
    with pytest.raises(ValueError, match="normal activation"):
        trial_coulomb_point_directional(origin, **data)
    data["friction_coefficient"] = 0.0
    assert trial_coulomb_point_directional(origin, **data) == (0.0, 0.0, 0.0)
