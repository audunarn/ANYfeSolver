"""Segmented six-channel weak work from actual E4 and native B3 ports."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint

from scripts._b3_e4_f03_e4_trial_panel import FlatE4TrialPanel
from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts._b3_e4_f03_shell_finite_pose_candidate import (
    pose_from_e4_total, reference_nodes)
from scripts._b3_e4_f03_shell_local_translation_candidate import (
    pose_from_e4_total_local_translation,
    pose_from_e4_total_local_director,
    pose_from_e4_total_local_frame)
from scripts._b3_e4_f03_shell_chord_pose_candidate import pose_from_e4_total_chord
from scripts._b3_e4_f03_shell_trace import make_shell
from scripts._b3_e4_f03_shell_spatial_linear_port import physical_shell_stiffness
from scripts._b3_e4_f03_weak_line_candidate import (
    LineMultiplierSpace, assemble_reference_kkt, integrate_weak_line,
    segments)


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
@pytest.mark.parametrize("kind", ("piecewise_constant", "continuous_linear"))
@pytest.mark.parametrize("eccentricity", (0., .11))
def test_segmented_reference_gap_and_multiplier_work(
        topology, kind, eccentricity):
    shell = FlatE4TrialPanel(make_shell(1, topology))
    beam = build(4)
    space = LineMultiplierSpace(kind, (0., .5, 1.))
    reference = np.zeros((shell.panel.node_count, 6))
    reference[:, 2] = -eccentricity
    coefficients = np.arange(1, 6*space.count+1, dtype=float).reshape(-1, 6)
    zeros_shell = np.zeros_like(reference)
    zeros_beam = np.zeros(beam.context.count)
    result = integrate_weak_line(
        shell, beam, space, coefficients, reference.ravel(),
        reference.ravel(), zeros_shell, zeros_shell,
        beam.context.initial.mechanical, zeros_beam, zeros_beam)
    np.testing.assert_allclose(result.weak_constraints, 0.,
                               rtol=0, atol=3e-13)
    assert result.potential == pytest.approx(0., abs=3e-12)
    assert result.potential == pytest.approx(
        np.sum(coefficients*result.weak_constraints), abs=3e-12)
    assert result.first == pytest.approx(0., abs=3e-12)
    assert result.second == pytest.approx(0., abs=3e-12)
    assert result.quadrature_count == 4*result.segment_count
    assert result.segment_count >= beam.n_half_cells
    assert len(result.shell_trial.candidate_states) == shell.panel.element_count
    np.testing.assert_allclose(result.shell_trial.residual, 0.,
                               rtol=0, atol=1e-4)
    edges = segments(shell, beam, space)
    assert set(np.round(space.knots, 14)).issubset(set(edges))
    assert set(np.round(np.arange(5)/4, 14)).issubset(set(edges))
    assert set(np.round(shell.panel.breakpoints(), 14)).issubset(set(edges))


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_reference_joint_varies_along_rigidly_rotated_shell(topology):
    shell = FlatE4TrialPanel(make_shell(1, topology))
    beam = build(4)
    space = LineMultiplierSpace("continuous_linear", (0., 1.))
    angle = np.array((0., .17, 0.))
    rotation = Rotation.from_rotvec(angle).as_matrix()
    nodes = reference_nodes(shell.panel)
    reference = np.zeros((shell.panel.node_count, 6))
    reference[:, :3] = nodes@rotation.T-nodes
    reference[:, 3:] = angle
    shell_direction = np.zeros_like(reference)
    beam_direction = np.zeros(beam.context.count)
    coefficients = np.arange(1, 13, dtype=float).reshape(2, 6)
    result = integrate_weak_line(
        shell, beam, space, coefficients, reference.ravel(),
        reference.ravel(), shell_direction, shell_direction,
        beam.context.initial.mechanical, beam_direction, beam_direction)
    np.testing.assert_allclose(result.weak_constraints, 0.,
                               rtol=0, atol=3e-13)
    assert result.potential == pytest.approx(0., abs=3e-12)
    shell_a = pose_from_e4_total(shell.panel, .2, reference.ravel())[0]
    shell_b = pose_from_e4_total(shell.panel, .8, reference.ravel())[0]
    beam_a = beam.pose(.2)[0]
    beam_b = beam.pose(.8)[0]
    assert np.linalg.norm((beam_a-shell_a)-(beam_b-shell_b)) > .05


@pytest.mark.parametrize("topology,kind", (
    ("Q4", "piecewise_constant"),
    ("S3-V2D", "continuous_linear"),
    ("Q4", "dual_linear"),
    ("S3-V2D", "dual_linear"),
))
@pytest.mark.parametrize("shell_port", (
    "rail", "local_translation", "local_director", "local_frame",
    "line_chord_020", "line_chord_confined_020"))
def test_segmented_line_directional_work_matches_direct_potential(
        topology, kind, shell_port):
    shell = FlatE4TrialPanel(make_shell(1, topology))
    beam = build(4)
    space = LineMultiplierSpace(kind, (0., .5, 1.))
    reference = np.zeros((shell.panel.node_count, 6))
    reference[:, 2] = -.07
    rng = np.random.default_rng(3972)
    current = reference+rng.normal(size=reference.shape)*.002
    shell_first = rng.normal(size=reference.shape)*.02
    shell_second = rng.normal(size=reference.shape)*.02
    beam_base = np.zeros(beam.context.count)
    beam_base[6:12] = (.001, -.002, .0015, .01, -.007, .004)
    beam_base[beam.context.nodal_count+3:beam.context.nodal_count+6] = (
        .012, -.008, .006)
    base_state = beam.context.advance(
        beam.context.initial.mechanical, beam_base)
    beam_first = np.zeros_like(beam_base)
    beam_second = np.zeros_like(beam_base)
    beam_first[6:12] = (.02, -.01, .015, .03, -.02, .01)
    beam_first[beam.context.nodal_count+3:beam.context.nodal_count+6] = (
        .04, -.03, .02)
    beam_second[12:18] = (-.015, .02, -.01, -.02, .01, .025)
    beam_second[beam.context.nodal_count+6:beam.context.nodal_count+9] = (
        -.02, .03, -.01)
    coefficients = np.arange(1, 6*space.count+1, dtype=float).reshape(-1, 6)
    coefficients = np.sin(coefficients)*21.
    result = integrate_weak_line(
        shell, beam, space, coefficients, reference.ravel(),
        current.ravel(), shell_first, shell_second,
        base_state, beam_first, beam_second, shell_port=shell_port)
    pose_map = {"rail": pose_from_e4_total,
                "local_translation": pose_from_e4_total_local_translation,
                "local_director": pose_from_e4_total_local_director,
                "local_frame": pose_from_e4_total_local_frame,
                "line_chord_020": pose_from_e4_total_chord,
                "line_chord_confined_020": lambda p, s, u: (
                    pose_from_e4_total_chord(
                        p, s, u, confined=True))}[shell_port]

    edges = segments(shell, beam, space, shell_port=shell_port)
    points, weights = np.polynomial.legendre.leggauss(4)

    def direct_quantities(first_factor, second_factor):
        shell_total = (current+first_factor*shell_first+
                       second_factor*shell_second).ravel()
        beam_step = first_factor*beam_first+second_factor*beam_second
        state = beam.context.advance(base_state, beam_step)
        total = 0.
        weak = np.zeros((space.count, 6))
        for left, right in zip(edges[:-1], edges[1:]):
            half, center = .5*(right-left), .5*(left+right)
            for point, weight in zip(points, weights):
                station = center+half*point
                element, cell, fraction, _ = beam._locate(station)
                port = NativeHalfCellSpatialPort(
                    beam.context, element, cell, fraction)
                shell_ref = pose_map(
                    shell.panel, station, reference.ravel())
                high_ref, low_ref, beam_frame_ref = port.pose_pair(
                    beam.context.initial.mechanical)
                joint = RigidPoseJoint(
                    np.array((shell_ref[0], high_ref+low_ref)),
                    np.array((shell_ref[1], beam_frame_ref)))
                shell_position, shell_frame = pose_map(
                    shell.panel, station, shell_total)
                high, low, beam_frame = port.pose_pair(state)
                multiplier = space.values(station)@coefficients
                work = joint.evaluate(
                    np.array((shell_position, high)),
                    np.array((shell_frame, beam_frame)), multiplier,
                    position_low=np.array((np.zeros(3), low)))
                total += half*weight*work.potential
                weak += (half*weight*space.values(station)[:, None]*
                         work.constraints[None, :])
        return total, weak

    assert result.potential == pytest.approx(direct_quantities(0., 0.)[0],
                                             abs=2e-11)
    assert result.potential == pytest.approx(
        np.sum(coefficients*result.weak_constraints), abs=2e-11)
    assert result.first == pytest.approx(
        np.sum(coefficients*result.weak_first), abs=2e-10)
    assert result.second == pytest.approx(
        np.sum(coefficients*result.weak_second), abs=2e-10)
    h = 1e-5
    first_plus = direct_quantities(h, 0.)
    first_minus = direct_quantities(-h, 0.)
    second_plus = direct_quantities(0., h)
    second_minus = direct_quantities(0., -h)
    np.testing.assert_allclose(
        (result.first, result.second),
        ((first_plus[0]-first_minus[0])/(2*h),
         (second_plus[0]-second_minus[0])/(2*h)),
        rtol=8e-7, atol=1e-8)
    np.testing.assert_allclose(result.weak_first,
                               (first_plus[1]-first_minus[1])/(2*h),
                               rtol=8e-7, atol=2e-9)
    np.testing.assert_allclose(result.weak_second,
                               (second_plus[1]-second_minus[1])/(2*h),
                               rtol=8e-7, atol=2e-9)
    h = 2e-3
    observed = (direct_quantities(h, h)[0]-direct_quantities(h, -h)[0]-
                direct_quantities(-h, h)[0]+direct_quantities(-h, -h)[0])/(4*h*h)
    np.testing.assert_allclose(result.mixed, observed,
                               rtol=3e-5, atol=5e-7)


def test_invalid_line_space_and_multiplier_rejected():
    with pytest.raises(ValueError, match="valid fixed"):
        LineMultiplierSpace("piecewise_constant", (0., .5, .5, 1.))
    shell = FlatE4TrialPanel(make_shell(1, "Q4"))
    beam = build(4)
    space = LineMultiplierSpace("piecewise_constant", (0., .5, 1.))
    total = np.zeros(6*shell.panel.node_count)
    shell_direction = np.zeros((shell.panel.node_count, 6))
    beam_direction = np.zeros(beam.context.count)
    with pytest.raises(ValueError, match="multiplier coefficients"):
        integrate_weak_line(
            shell, beam, space, np.zeros((1, 6)), total, total,
            shell_direction, shell_direction,
            beam.context.initial.mechanical, beam_direction,
            beam_direction)


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
@pytest.mark.parametrize("kind", ("piecewise_constant", "continuous_linear"))
@pytest.mark.parametrize("shell_port", ("rail", "local_translation"))
def test_reference_kkt_matches_weak_derivative_and_scaled_solve(
        topology, kind, shell_port):
    shell = FlatE4TrialPanel(make_shell(1, topology))
    beam = build(4)
    space = LineMultiplierSpace(kind, (0., .5, 1.))
    kkt = assemble_reference_kkt(shell, beam, space, shell_port=shell_port)
    rng = np.random.default_rng(1138)
    shell_direction = rng.normal(size=(shell.panel.node_count, 6))*.001
    beam_direction = rng.normal(size=beam.context.count)*.001
    beam_direction[list(beam.context.fixed)] = 0.
    for node in (0, 2*shell.panel.n_s+1, 2*(2*shell.panel.n_s+1)):
        shell_direction[node] = 0.
    zeros_shell = np.zeros_like(shell_direction)
    zeros_beam = np.zeros_like(beam_direction)
    multipliers = np.zeros((space.count, 6))
    weak = integrate_weak_line(
        shell, beam, space, multipliers,
        zeros_shell.ravel(), zeros_shell.ravel(),
        shell_direction, zeros_shell,
        beam.context.initial.mechanical,
        beam_direction, zeros_beam, shell_port=shell_port)
    direction = np.r_[shell_direction.ravel(), beam_direction]
    np.testing.assert_allclose(
        kkt.constraint_jacobian@direction,
        weak.weak_first.ravel(), rtol=2e-11, atol=2e-12)
    assert np.linalg.norm(kkt.scaled_system-kkt.scaled_system.T) <= (
        1e-10*np.linalg.norm(kkt.scaled_system))

    shell_size = 6*shell.panel.node_count
    full_load = np.zeros(shell_size+beam.context.count)
    full_load[shell_size+6*beam.n_half_cells+2] = 1000.
    free_count = len(kkt.free_primal)
    rhs = np.r_[full_load[kkt.free_primal]*
                kkt.coordinate_scale[:free_count]/kkt.energy_scale,
                np.zeros(6*space.count)]
    solution = np.linalg.solve(kkt.scaled_system, rhs)
    scaled_residual = kkt.scaled_system@solution-rhs
    assert np.linalg.norm(scaled_residual) <= 1e-9*max(1., np.linalg.norm(rhs))
    physical = kkt.coordinate_scale*solution
    physical_rhs = np.r_[full_load[kkt.free_primal],
                         np.zeros(6*space.count)]
    physical_residual = kkt.system@physical-physical_rhs
    assert np.linalg.norm(physical_residual) <= (
        1e-10*max(1., np.linalg.norm(physical_rhs)))
    gaps = kkt.constraint_jacobian[:, kkt.free_primal]@physical[:free_count]
    assert np.linalg.norm(gaps) <= 1e-9
    assert np.isfinite(physical).all()


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
@pytest.mark.parametrize("shell_port", ("rail", "local_translation"))
def test_reference_line_annihilates_six_common_rigid_modes(
        topology, shell_port):
    shell = FlatE4TrialPanel(make_shell(1, topology))
    beam = build(4)
    space = LineMultiplierSpace("continuous_linear", (0., .5, 1.))
    coupling = assemble_reference_kkt(
        shell, beam, space, shell_port=shell_port).constraint_jacobian
    shell_nodes = reference_nodes(shell.panel)
    beam_nodes = (beam.context.initial.mechanical.positions+
                  beam.context.initial.mechanical.position_low)
    for axis in range(3):
        basis = np.eye(3)[axis]
        shell_translation = np.zeros((shell.panel.node_count, 6))
        beam_translation = np.zeros(beam.context.count)
        shell_translation[:, axis] = 1.
        beam_translation[axis:beam.context.nodal_count:6] = 1.
        assert np.linalg.norm(coupling@np.r_[
            shell_translation.ravel(), beam_translation]) < 2e-12

        shell_rotation = np.zeros_like(shell_translation)
        beam_rotation = np.zeros_like(beam_translation)
        shell_rotation[:, :3] = np.cross(basis, shell_nodes)
        shell_rotation[:, 3:] = basis
        for node, position in enumerate(beam_nodes):
            beam_rotation[6*node:6*node+3] = np.cross(basis, position)
            beam_rotation[6*node+3:6*node+6] = basis
        for element in range(beam.n_half_cells//2):
            start = beam.context.nodal_count+24*element
            beam_rotation[start:start+3] = basis
            beam_rotation[start+3:start+6] = basis
        assert np.linalg.norm(coupling@np.r_[
            shell_rotation.ravel(), beam_rotation]) < 2e-12


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
@pytest.mark.parametrize("n_s,n_half_cells,n_y,kind,support_policy,intervals", (
    (1, 4, 2, "piecewise_constant", "legacy_three_left_nodes", 2),
    (2, 8, 2, "piecewise_constant", "legacy_three_left_nodes", 4),
    (4, 16, 2, "piecewise_constant", "legacy_three_left_nodes", 8),
    (4, 16, 4, "piecewise_constant", "both_xy_edges", 4),
    (4, 16, 4, "continuous_linear", "both_xy_edges", 4),
    (4, 16, 4, "dual_linear", "both_xy_edges", 4),
    (8, 32, 8, "piecewise_constant", "both_xy_edges", 8),
    (8, 32, 8, "continuous_linear", "both_xy_edges", 8),
    (8, 32, 8, "dual_linear", "both_xy_edges", 8),
))
def test_torque_reaches_physical_shell_without_drilling_completion(
        topology, n_s, n_half_cells, n_y, kind, support_policy, intervals):
    panel = make_shell(n_s, topology, n_y=n_y)
    shell = FlatE4TrialPanel(panel)
    beam = build(n_half_cells)
    space = LineMultiplierSpace(
        kind, tuple(np.linspace(0., 1., intervals+1)))
    kkt = assemble_reference_kkt(
        shell, beam, space, shell_support_policy=support_policy)
    shell_size = 6*panel.node_count
    tip_rx = shell_size+6*beam.n_half_cells+3
    load = np.zeros(shell_size+beam.context.count)
    load[tip_rx] = 100.

    def solve(system, free, scale):
        count = len(free)
        rhs = np.r_[load[free]*scale[:count]/kkt.energy_scale,
                    np.zeros(6*space.count)]
        scaled = scale[:, None]*system*scale[None, :]/kkt.energy_scale
        solution = scale*np.linalg.solve(scaled, rhs)
        full = np.zeros(len(load))
        full[free] = solution[:count]
        return full, solution[count:]

    full_response, full_multipliers = solve(
        kkt.system, kkt.free_primal, kkt.coordinate_scale)
    physical = physical_shell_stiffness(panel).toarray()
    physical_primal = kkt.primal_tangent.copy()
    physical_primal[:shell_size, :shell_size] = physical
    # A drilling-free physical shell has no independent rz coordinates.
    # The objective line's yaw comes from physical in-plane translations.
    free = np.array([index for index in kkt.free_primal
                     if not (index < shell_size and index % 6 == 5)])
    keep = np.isin(kkt.free_primal, free)
    rows = kkt.constraint_jacobian[:, free]
    physical_system = np.block([
        [physical_primal[np.ix_(free, free)], rows.T],
        [rows, np.zeros((len(rows), len(rows)))],
    ])
    scale = np.r_[kkt.coordinate_scale[:len(kkt.free_primal)][keep],
                  kkt.coordinate_scale[len(kkt.free_primal):]]
    physical_response, physical_multipliers = solve(physical_system, free, scale)
    shell_energy = .5*physical_response[:shell_size]@physical@(
        physical_response[:shell_size])
    assert shell_energy > 1e-6
    assert abs(physical_response[tip_rx]) > 1e-7
    np.testing.assert_allclose(physical_response[tip_rx],
                               full_response[tip_rx], rtol=1e-8, atol=1e-12)
    physical_shell_dofs = np.array([
        index for index in range(shell_size) if index % 6 != 5])
    np.testing.assert_allclose(physical_response[physical_shell_dofs],
                               full_response[physical_shell_dofs],
                               rtol=1e-8, atol=1e-12)
    assert np.linalg.norm(rows@physical_response[free]) < 1e-9

    support_residual = (kkt.primal_tangent@full_response+
                        kkt.constraint_jacobian.T@full_multipliers-load)
    assert np.linalg.norm(support_residual[kkt.free_primal]) < 1e-7
    shell_residual = support_residual[:shell_size].reshape(-1, 6)
    beam_residual = support_residual[
        shell_size:shell_size+beam.context.nodal_count].reshape(-1, 6)
    shell_positions = reference_nodes(panel)
    beam_positions = (beam.context.initial.mechanical.positions+
                      beam.context.initial.mechanical.position_low)
    shell_force = shell_residual[:, :3].sum(axis=0)
    beam_force = beam_residual[:, :3].sum(axis=0)
    shell_moment = np.sum(shell_residual[:, 3:]+np.cross(
        shell_positions, shell_residual[:, :3]), axis=0)
    physical_residual = (physical_primal@physical_response+
                         kkt.constraint_jacobian.T@physical_multipliers-load)
    physical_shell_residual = physical_residual[:shell_size].reshape(-1, 6)
    physical_shell_moment = np.sum(
        physical_shell_residual[:, 3:]+np.cross(
            shell_positions, physical_shell_residual[:, :3]), axis=0)
    np.testing.assert_allclose(physical_shell_moment[0], shell_moment[0],
                               rtol=1e-8, atol=1e-7)
    beam_moment = np.sum(beam_residual[:, 3:]+np.cross(
        beam_positions, beam_residual[:, :3]), axis=0)
    np.testing.assert_allclose(shell_force+beam_force, 0., atol=1e-7)
    np.testing.assert_allclose(shell_moment+beam_moment,
                               (-100., 0., 0.), atol=1e-7)
    assert abs(shell_moment[0]) > 1e-4


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_shell_thinning_moves_torque_response_toward_bare_beam(topology):
    beam = build(8)
    space = LineMultiplierSpace(
        "piecewise_constant", (0., .25, .5, .75, 1.))
    beam_load = np.zeros(beam.context.count)
    beam_tip_rx = 6*beam.n_half_cells+3
    beam_load[beam_tip_rx] = 100.
    beam_free = np.array([index for index in range(beam.context.count)
                          if index not in beam.context.fixed])
    bare_beam = np.zeros_like(beam_load)
    bare_beam[beam_free] = np.linalg.solve(
        beam.tangent.toarray()[np.ix_(beam_free, beam_free)],
        beam_load[beam_free])
    bare_tip = float(bare_beam[beam_tip_rx])
    tip_rotations = []
    for thickness in (.07, .035, .0175):
        panel = make_shell(2, topology, thickness=thickness)
        shell = FlatE4TrialPanel(panel)
        kkt = assemble_reference_kkt(shell, beam, space)
        shell_size = 6*panel.node_count
        tip_rx = shell_size+6*beam.n_half_cells+3
        load = np.zeros(shell_size+beam.context.count)
        load[tip_rx] = 100.
        free_count = len(kkt.free_primal)
        rhs = np.r_[load[kkt.free_primal]*
                    kkt.coordinate_scale[:free_count]/kkt.energy_scale,
                    np.zeros(6*space.count)]
        scaled_solution = np.linalg.solve(kkt.scaled_system, rhs)
        physical_solution = kkt.coordinate_scale*scaled_solution
        full = np.zeros(len(load))
        full[kkt.free_primal] = physical_solution[:free_count]
        multipliers = physical_solution[free_count:]
        physical = physical_shell_stiffness(panel).toarray()
        shell_u = full[:shell_size]
        tip_rotations.append(float(full[tip_rx]))
        shell_energy = float(.5*shell_u@physical@shell_u)
        residual = (kkt.primal_tangent@full+
                    kkt.constraint_jacobian.T@multipliers-load)
        shell_residual = residual[:shell_size].reshape(-1, 6)
        beam_residual = residual[
            shell_size:shell_size+beam.context.nodal_count].reshape(-1, 6)
        shell_moment = np.sum(
            shell_residual[:, 3:]+np.cross(
                reference_nodes(panel), shell_residual[:, :3]), axis=0)
        beam_positions = (beam.context.initial.mechanical.positions+
                          beam.context.initial.mechanical.position_low)
        beam_moment = np.sum(
            beam_residual[:, 3:]+np.cross(
                beam_positions, beam_residual[:, :3]), axis=0)
        assert np.linalg.norm(kkt.scaled_system@scaled_solution-rhs) < 1e-9
        assert shell_energy > 0.
        np.testing.assert_allclose(shell_moment+beam_moment,
                                   (-100., 0., 0.), atol=1e-7)
    assert all(a < b for a, b in zip(tip_rotations, tip_rotations[1:])), tip_rotations
    assert 0. < tip_rotations[0] < tip_rotations[-1] < bare_tip


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
@pytest.mark.parametrize("kind", ("piecewise_constant", "continuous_linear"))
@pytest.mark.parametrize("n_s,n_half_cells", ((2, 8), (4, 16)))
def test_extended_mesh_reference_kkt_executes_without_shape_or_rank_loss(
        topology, kind, n_s, n_half_cells):
    shell = FlatE4TrialPanel(make_shell(n_s, topology))
    beam = build(n_half_cells)
    space = LineMultiplierSpace(
        kind, tuple(np.linspace(0., 1., 2*n_s+1)))
    kkt = assemble_reference_kkt(shell, beam, space)
    assert kkt.system.shape[0] > 100*n_s
    shell_size = 6*shell.panel.node_count
    load = np.zeros(shell_size+beam.context.count)
    load[shell_size+6*beam.n_half_cells+2] = 1000.
    count = len(kkt.free_primal)
    rhs = np.r_[load[kkt.free_primal]*
                kkt.coordinate_scale[:count]/kkt.energy_scale,
                np.zeros(6*space.count)]
    solution = np.linalg.solve(kkt.scaled_system, rhs)
    assert np.linalg.norm(kkt.scaled_system@solution-rhs) < 1e-9
    physical = kkt.coordinate_scale*solution
    assert np.linalg.norm(kkt.constraint_jacobian[:, kkt.free_primal]@
                          physical[:count]) < 1e-9


def test_transverse_tip_force_has_physical_support_wrench_and_work_balance():
    from scripts.b3_e4_f03_weak_line_resolution_screen import one_case

    result = one_case(
        "Q4", 4, "continuous_linear", load_kind="transverse_force_z",
        n_s=2, native_half_cells=4, shell_support_policy="both_xy_edges")
    assert result["applied_force_z_n"] == 1000.
    assert result["applied_moment_y_about_origin_n_m"] == -1000.
    assert result["beam_tip_uz_m"] > 0.
    assert result["shell_full_minus_physical_inplane_energy_j"] > 0.
    assert abs(result["global_force_z_balance_n"]) < 1e-7
    assert abs(result["global_moment_y_balance_n_m"]) < 1e-7
    assert abs(result["global_work_error_j"]) < 1e-8


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_local_frame_yaw_load_recovers_actual_interface_moment(topology):
    from scripts.b3_e4_f03_weak_line_resolution_screen import one_case

    result = one_case(
        topology, 2, "piecewise_constant", shell_port="local_frame",
        load_kind="yaw_torque_z", n_s=2, n_y=2,
        native_half_cells=4, shell_support_policy="full_left_edge")
    assert result["applied_moment_z_about_origin_n_m"] == pytest.approx(100.)
    assert result["tip_rz_rad"] > 0.
    assert result["shell_component_physical_energy_j"] > 0.
    assert result["shell_numerical_energy_j"] >= 0.
    assert abs(result["global_moment_z_balance_n_m"]) < 1e-8
    assert abs(result["shell_support_reaction_mz_about_origin_n_m"]-
               result["shell_interface_mz_about_origin_n_m"]) < 1e-8
    assert abs(result["global_work_error_j"]) < 1e-8


def test_local_frame_segments_odd_shell_hosts_before_quadrature():
    shell = FlatE4TrialPanel(make_shell(3, "Q4"))
    beam = build(4)
    space = LineMultiplierSpace("piecewise_constant", (0., 1.))
    edges = segments(shell, beam, space, shell_port="local_frame")
    for center_host_edge in (1/6, .5, 5/6):
        assert np.min(np.abs(edges-center_host_edge)) < 1e-13
    assert np.min(np.diff(edges)) > 1e-12
