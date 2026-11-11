"""Finite-area shell pose candidate; no coupled or material qualification."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_pose_joint import _exp_terms

from scripts._b3_e4_f03_shell_finite_pose_candidate import (
    directional_pose, directional_pose_from_e4_total, pose,
    pose_from_e4_total, reference_nodes, reference_tangent)
from scripts._b3_e4_f03_shell_trace import make_shell


def _directors(count):
    return np.tile((0., 0., 1.), (count, 1))


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_reference_and_large_common_rigid_motion(topology):
    panel = make_shell(2, topology)
    nodes = reference_nodes(panel)
    directors = _directors(panel.node_count)
    for station in (0., .37, .5, 1.):
        center, frame = pose(panel, station, nodes, directors)
        np.testing.assert_allclose(center, (station, 0., 0.),
                                   rtol=0, atol=2e-15)
        np.testing.assert_allclose(frame, np.eye(3), rtol=0, atol=2e-15)
        rotation = Rotation.from_rotvec((.94, -.72, .48)).as_matrix()
        shift = np.array((.23, -.17, .31))
        moved = pose(panel, station, nodes@rotation.T+shift,
                     directors@rotation.T)
        np.testing.assert_allclose(moved[0], rotation@center+shift,
                                   rtol=0, atol=3e-15)
        np.testing.assert_allclose(moved[1], rotation@frame,
                                   rtol=0, atol=4e-15)


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_nonrigid_state_is_objective_and_keeps_independent_director(topology):
    panel = make_shell(2, topology)
    nodes = reference_nodes(panel)
    deformed = nodes.copy()
    deformed[:, 0] += .05*nodes[:, 1]
    deformed[:, 2] += .09*nodes[:, 1]-.025*nodes[:, 0]
    nodal_tilt = np.column_stack((.02*nodes[:, 0],
                                  -.03*np.ones(panel.node_count),
                                  np.zeros(panel.node_count)))
    directors = np.array([Rotation.from_rotvec(v).apply((0., 0., 1.))
                          for v in nodal_tilt])
    center, frame = pose(panel, .37, deformed, directors)
    np.testing.assert_allclose(frame[:, 2],
        (make_shell(2, topology).sample_weights(.37).mean(axis=0)@directors)/
        np.linalg.norm(make_shell(2, topology).sample_weights(.37).mean(axis=0)@directors),
        rtol=0, atol=3e-15)
    rotation = Rotation.from_rotvec((-.43, .27, .73)).as_matrix()
    shift = np.array((.3, -.4, .1))
    moved = pose(panel, .37, deformed@rotation.T+shift,
                 directors@rotation.T)
    np.testing.assert_allclose(moved[0], rotation@center+shift,
                               rtol=0, atol=3e-15)
    np.testing.assert_allclose(moved[1], rotation@frame,
                               rtol=0, atol=5e-15)


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_actual_finite_pose_linearizes_to_six_channel_reference_port(topology):
    panel = make_shell(2, topology)
    nodes = reference_nodes(panel)
    count = panel.node_count
    rng = np.random.default_rng(20260924)
    direction = rng.normal(size=(count, 6))*.08
    h = 1e-6
    evaluations = []
    for sign in (1., -1.):
        translations = nodes+sign*h*direction[:, :3]
        directors = np.array([
            Rotation.from_rotvec(sign*h*v).apply((0., 0., 1.))
            for v in direction[:, 3:]])
        evaluations.append(pose(panel, .37, translations, directors))
    plus, minus = evaluations
    observed = np.r_[(plus[0]-minus[0])/(2*h),
        Rotation.from_matrix(plus[1]@minus[1].T).as_rotvec()/(2*h)]
    expected = reference_tangent(panel, .37)@direction.ravel()
    np.testing.assert_allclose(observed, expected, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_slope_and_drilling_do_not_replace_physical_director(topology):
    panel = make_shell(2, topology)
    nodes = reference_nodes(panel)
    directors = _directors(panel.node_count)
    slope = nodes.copy()
    slope[:, 2] = .12*nodes[:, 1]
    _, frame = pose(panel, .37, slope, directors)
    np.testing.assert_allclose(frame, np.eye(3), rtol=0, atol=2e-14)
    # A pure nodal rz coordinate leaves each physical director unchanged;
    # with unchanged positions, the port cannot gain numerical drilling work.
    _, unchanged = pose(panel, .37, nodes, directors.copy())
    np.testing.assert_allclose(unchanged, np.eye(3), rtol=0, atol=2e-15)
    with pytest.raises(ValueError, match="degenerate material rail"):
        pose(panel, .37, np.zeros_like(nodes), directors)
    with pytest.raises(ValueError, match="antipodal"):
        pose(panel, .37, nodes, -directors)


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_director_parallel_transverse_shear_does_not_create_spin(topology):
    panel = make_shell(2, topology)
    nodes = reference_nodes(panel)
    tilted = Rotation.from_rotvec((.4, -.17, .08)).apply((0., 0., 1.))
    directors = np.tile(tilted, (panel.node_count, 1))
    base_center, base_frame = pose(panel, .37, nodes, directors)
    for gradient in (-.2, .14):
        sheared = nodes+gradient*nodes[:, 0, None]*tilted
        center, frame = pose(panel, .37, sheared, directors)
        np.testing.assert_allclose(frame, base_frame, rtol=0, atol=2e-14)
        np.testing.assert_allclose(center-base_center,
                                   gradient*.37*tilted, rtol=0, atol=3e-15)


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_first_and_second_variations_follow_spatial_left_updates(topology):
    panel = make_shell(2, topology)
    nodes = reference_nodes(panel)
    positions = nodes.copy()
    positions[:, 0] += .04*nodes[:, 1]
    positions[:, 2] += .07*nodes[:, 0]-.03*nodes[:, 1]
    directors = np.array([Rotation.from_rotvec((.12+.02*x, -.09, .03))
                          .apply((0., 0., 1.)) for x in nodes[:, 0]])
    rng = np.random.default_rng(3051)
    first = rng.normal(size=(panel.node_count, 6))*.04
    second = rng.normal(size=(panel.node_count, 6))*.04
    analytic = directional_pose(panel, .37, positions, directors,
                                first, second)

    def moved(a, b):
        step = a*first+b*second
        altered = positions+step[:, :3]
        tilted = np.array([Rotation.from_rotvec(v).apply(d)
                           for v, d in zip(step[:, 3:], directors)])
        return pose(panel, .37, altered, tilted)

    h = 1e-5
    plus, minus = moved(h, 0.), moved(-h, 0.)
    np.testing.assert_allclose(analytic.position_first,
                               (plus[0]-minus[0])/(2*h),
                               rtol=2e-7, atol=2e-9)
    np.testing.assert_allclose(analytic.frame_first,
                               (plus[1]-minus[1])/(2*h),
                               rtol=2e-7, atol=2e-9)
    h = 2e-3
    pp, pm, mp, mm = (moved(h, h), moved(h, -h),
                      moved(-h, h), moved(-h, -h))
    np.testing.assert_allclose(analytic.position_mixed,
        (pp[0]-pm[0]-mp[0]+mm[0])/(4*h*h), rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(analytic.frame_mixed,
        (pp[1]-pm[1]-mp[1]+mm[1])/(4*h*h), rtol=2e-5, atol=2e-8)


def test_physical_drill_does_not_move_tilted_director_port():
    panel = make_shell(2, "Q4")
    nodes = reference_nodes(panel)
    director = Rotation.from_rotvec((.36, -.21, .1)).apply((0., 0., 1.))
    directors = np.tile(director, (panel.node_count, 1))
    drill = np.zeros((panel.node_count, 6))
    drill[:, 3:] = director
    result = directional_pose(panel, .37, nodes, directors, drill, drill)
    for value in (result.position_first, result.position_mixed,
                  result.frame_first, result.frame_mixed):
        assert np.linalg.norm(value) < 1e-13


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_every_reference_coordinate_column_matches_independent_trace(topology):
    panel = make_shell(1, topology)
    nodes = reference_nodes(panel)
    directors = _directors(panel.node_count)
    trace = reference_tangent(panel, .37)
    zero = np.zeros((panel.node_count, 6))
    for column in range(trace.shape[1]):
        direction = zero.copy()
        direction.ravel()[column] = 1.
        result = directional_pose(panel, .37, nodes, directors,
                                  direction, zero)
        derivative = result.frame_first
        angular = np.array((derivative[2, 1]-derivative[1, 2],
                            derivative[0, 2]-derivative[2, 0],
                            derivative[1, 0]-derivative[0, 1]))/2
        observed = np.r_[result.position_first, angular]
        np.testing.assert_allclose(observed, trace[:, column],
                                   rtol=0, atol=2e-14)


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_additive_e4_total_chart_first_and_second_variations(topology):
    panel = make_shell(2, topology)
    nodes = reference_nodes(panel)
    total = np.zeros((panel.node_count, 6))
    total[:, 0] = .02*nodes[:, 1]
    total[:, 2] = .04*nodes[:, 0]
    total[:, 3] = .17+.02*nodes[:, 0]
    total[:, 4] = -.08
    total[:, 5] = .06
    rng = np.random.default_rng(1904)
    first = rng.normal(size=total.shape)*.04
    second = rng.normal(size=total.shape)*.04
    analytic = directional_pose_from_e4_total(
        panel, .37, total.ravel(), first, second)

    def moved(a, b):
        return pose_from_e4_total(
            panel, .37, (total+a*first+b*second).ravel())

    h = 1e-5
    plus, minus = moved(h, 0.), moved(-h, 0.)
    np.testing.assert_allclose(analytic.position_first,
                               (plus[0]-minus[0])/(2*h),
                               rtol=2e-7, atol=2e-9)
    np.testing.assert_allclose(analytic.frame_first,
                               (plus[1]-minus[1])/(2*h),
                               rtol=2e-7, atol=2e-9)
    h = 2e-3
    pp, pm, mp, mm = (moved(h, h), moved(h, -h),
                      moved(-h, h), moved(-h, -h))
    np.testing.assert_allclose(analytic.position_mixed,
        (pp[0]-pm[0]-mp[0]+mm[0])/(4*h*h), rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(analytic.frame_mixed,
        (pp[1]-pm[1]-mp[1]+mm[1])/(4*h*h), rtol=2e-5, atol=2e-8)


def test_additive_chart_physical_drill_is_not_naive_rz_column():
    panel = make_shell(2, "Q4")
    total = np.zeros((panel.node_count, 6))
    total[:, 3:] = (.36, -.21, .1)
    direction = np.zeros_like(total)
    theta = total[0, 3:]
    current = Rotation.from_rotvec(theta).apply((0., 0., 1.))
    _rotation, left_map, _derivative = _exp_terms(theta)
    direction[:, 3:] = np.linalg.solve(left_map, current)
    result = directional_pose_from_e4_total(
        panel, .37, total.ravel(), direction, np.zeros_like(direction))
    assert np.linalg.norm(result.position_first) < 1e-13
    assert np.linalg.norm(result.frame_first) < 1e-12
    naive_rz = np.zeros_like(direction)
    naive_rz[:, 5] = 1.
    naive = directional_pose_from_e4_total(
        panel, .37, total.ravel(), naive_rz, np.zeros_like(direction))
    assert np.linalg.norm(naive.frame_first) > 1e-3
