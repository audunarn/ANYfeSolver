"""One bounded finite-chart gate for the six-resultant surface candidate."""

from fractions import Fraction
from math import comb

import numpy as np
import pytest
from scipy import linalg

from scripts._b3_e4_f03_native_fixture import build
from scripts.b3_e4_f03_distributed_surface_finite import (
    FiniteSurfaceLineHybrid, FiniteSurfaceMomentPatch,
    FiniteSurfaceTopFaceLine, c1_hybrid_line_space,
    bounded_physical_compliance_screen, finite_chart_screen,
)
from anyfesolver.corotational import rotation_matrix_from_vector
from scripts.b3_e4_f03_local_footprint_work import (
    bond_x_segments, distributed_surface_patch_trace,
    graded_terminal_coupling, structured_c1_mixed_point_shape,
    structured_q4_point_shape,
)
from scripts.b3_e4_f03_plate_patch_work import (
    c1_mixed_s3_cells, local_grid, raw_e4_mixed_c1_tangent,
    raw_e4_tangent,
)
from scripts.b3_e4_f03_square_section_control import square_torsion_constant
from scripts.b3_e4_f03_common_trace_finite_work import _CommonRawWork
from scripts.b3_e4_f03_common_trace_spaces import _assemble
from scripts.b3_e4_f03_top_face_common_trace import _raw_rows


def test_common_top_face_spaces_leave_physical_top_fixed_pitch_free():
    xs, ys = local_grid(4, transverse_cells=4)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    shell_size = 6*len(xs)*len(ys)
    shell = np.zeros((len(xs)*len(ys), 6))
    terminal = np.array([(x >= .9-1e-12)
                         for y in ys for x in xs])
    shell[terminal, 0] = -.0175e-5
    shell[terminal, 4] = 1e-5
    field = np.r_[shell.ravel(), np.zeros(beam.context.count)]
    _, physical = raw_e4_tangent(xs, ys, thickness=.035)
    assert physical.shape == (shell_size, shell_size)
    assert float(shell.ravel()@physical@shell.ravel()) > 0.
    for kind, index in (("piecewise_constant", 1),
                        ("continuous_linear", 2),
                        ("dual_linear", 0)):
        patch = _CommonRawWork(xs, ys, beam, kind=kind, index=index)
        assert np.max(np.abs(_raw_rows(patch, np.eye(3))@field)) < 1e-12
    hybrid = FiniteSurfaceLineHybrid(xs, ys, beam, thickness=.035)
    assert abs((hybrid.reference_trace()@field)[4]-1e-5) < 1e-12


def test_common_top_face_signed_basis_moments_match_exact_polynomials():
    xs, ys = local_grid(16, transverse_cells=4)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    intervals = 8
    edges = np.linspace(.9, 1., intervals+1)
    patches = [FiniteSurfaceTopFaceLine(
        xs, ys, beam, thickness=.035,
        x_interval=(float(low), float(high)))
        for low, high in zip(edges[:-1], edges[1:])]
    h = Fraction(1, 10*intervals)
    center = Fraction(19, 20)
    for kind in ("piecewise_constant", "continuous_linear", "dual_linear"):
        _, masses, first, second = _assemble(
            patches, np.eye(3), kind, intervals)
        expected = [[Fraction(0) for _ in range(len(masses))]
                    for _ in range(3)]
        for cell in range(intervals):
            left = Fraction(9, 10)+cell*h
            basis = ({cell: (1, 0)} if kind == "piecewise_constant" else
                     {cell: (1, -1), cell+1: (0, 1)}
                     if kind == "continuous_linear" else
                     {cell: (2, -3), cell+1: (-1, 3)})
            for index, (constant, slope) in basis.items():
                for power in range(3):
                    expected[power][index] += h*sum(
                        comb(power, term)*(left-center)**(power-term)*
                        h**term*(Fraction(constant, term+1)+
                                 Fraction(slope, term+2))
                        for term in range(power+1))
        for actual, exact in zip((masses, first, second), expected):
            np.testing.assert_allclose(actual, [float(v) for v in exact],
                                       rtol=0., atol=2e-15)


def test_actual_e4_b3_surface_work_has_consistent_finite_derivatives():
    result = finite_chart_screen()
    assert result["reference_gap_max_m_rad"] < 1e-12
    assert result["reference_jacobian_direction_error"] < 1e-12
    assert result["nonzero_gap_norm_m_rad"] > 1e-4
    assert abs(result["nonzero_multiplier_mixed_work_j"]) > 1e-3
    assert result["first_direction_error_j"] < 1e-7
    assert result["mixed_direction_error_j"] < 1e-6
    assert result["finite_common_rigid_gap_error"] < 1e-12
    assert abs(result["common_rigid_virtual_work_j"]) < 1e-9
    assert result["quadrature_points"] == 18


def test_reference_integration_splits_native_half_cell_inside_shell_host():
    assert bond_x_segments(.925, .95, 16) == (
        (.925, .9375), (.9375, .95))
    c1_stations = np.array((0., .75, .7875, .825, .8625, .9, 1.))
    assert bond_x_segments(.78, .85, c1_stations) == (
        (.78, .7875), (.7875, .825), (.825, .85))
    xs, ys = local_grid(4)
    beam = build(16, z_offset=.0675,
                 torsion_constant=square_torsion_constant())
    patch = FiniteSurfaceMomentPatch(xs, ys, beam, thickness=.035)
    assert len(patch.points) == 180
    trace, reference = distributed_surface_patch_trace(
        xs, ys, beam, plate_thickness=.035)
    assert abs(reference["area_m2"]-.01) < 1e-12
    first = np.zeros((patch.shell_count, 6))
    first[:, 2] = 1e-5*patch.reference_nodes[:, 1]
    beam_first = np.zeros(beam.context.count)
    direction = (first, beam_first)
    frames = np.broadcast_to(np.eye(3), (patch.shell_count, 3, 3)).copy()
    derivative = patch.evaluate(
        patch.reference_nodes, frames, beam.context.initial.mechanical,
        direction, direction)[:, 1]
    assert np.max(np.abs(derivative-trace@np.r_[first.ravel(), beam_first])) < 1e-12


def test_interval_surface_means_partition_without_reprojecting_reference():
    xs, ys = local_grid(2)
    beam = build(8, z_offset=.0675,
                 torsion_constant=square_torsion_constant())
    whole, full = distributed_surface_patch_trace(
        xs, ys, beam, plate_thickness=.035, x_interval=(0., 1.))
    left, first = distributed_surface_patch_trace(
        xs, ys, beam, plate_thickness=.035, x_interval=(0., .5))
    right, second = distributed_surface_patch_trace(
        xs, ys, beam, plate_thickness=.035, x_interval=(.5, 1.))
    assert max(full["reference_gap_m"], first["reference_gap_m"],
               second["reference_gap_m"]) < 1e-12
    assert abs(full["area_m2"]-.1) < 1e-12
    assert np.max(np.abs(whole[:3]-.5*(left[:3]+right[:3]))) < 1e-12


def test_c1_surface_moment_chart_matches_linearized_native_trace():
    xs, ys = local_grid(2)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    patch = FiniteSurfaceMomentPatch(xs, ys, beam, thickness=.035)
    trace, reference = distributed_surface_patch_trace(
        xs, ys, beam, plate_thickness=.035)
    assert reference["reference_gap_m"] < 1e-12
    frames = np.broadcast_to(np.eye(3), (patch.shell_count, 3, 3)).copy()
    shell_direction = np.zeros((patch.shell_count, 6))
    shell_direction[:, 2] = 1e-5*patch.reference_nodes[:, 1]
    beam_direction = np.zeros(beam.context.count)
    beam_direction[6*beam.n_half_cells+2] = 2e-5
    direction = (shell_direction, beam_direction)
    actual = patch.evaluate(
        patch.reference_nodes, frames, beam.context.initial.mechanical,
        direction, direction)
    assert np.max(np.abs(actual[:, 0])) < 1e-12
    expected = trace@np.r_[shell_direction.ravel(), beam_direction]
    assert np.max(np.abs(actual[:, 1]-expected)) < 1e-12


def test_c1_surface_moment_finite_chart_preserves_work_and_objectivity():
    result = finite_chart_screen("c1_tangent_2um")
    assert result["reference_profile"] == "c1_tangent_2um"
    assert result["reference_gap_max_m_rad"] < 1e-12
    assert result["reference_jacobian_direction_error"] < 1e-12
    assert abs(result["nonzero_multiplier_mixed_work_j"]) > 1e-3
    assert result["first_direction_error_j"] < 1e-7
    assert result["mixed_direction_error_j"] < 1e-6
    assert result["finite_common_rigid_gap_error"] < 1e-12
    assert abs(result["common_rigid_virtual_work_j"]) < 1e-9
    assert result["quadrature_points"] == 36


def test_six_surface_moments_leave_a_real_local_slip_mode():
    xs, ys = local_grid(2)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    patch = FiniteSurfaceMomentPatch(xs, ys, beam, thickness=.035)
    shell_step = np.zeros((patch.shell_count, 6))
    slope = 1e-5
    shell_step[:, 0] = slope*(patch.reference_nodes[:, 0]-.95)
    stats = patch.linearized_gap_statistics(
        shell_step, np.zeros(beam.context.count))
    assert np.max(np.abs(stats["weak_six_gap_m_rad"])) < 1e-12
    assert abs(stats["quadrature_rms_position_gap_m"]-
               slope*.1/np.sqrt(12)) < 1e-12
    assert stats["quadrature_max_position_gap_m"] > 0.


def test_c1_hybrid_six_channel_reference_is_distinct_and_non_drilling():
    xs, ys = local_grid(2)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    hybrid = FiniteSurfaceLineHybrid(xs, ys, beam, thickness=.035)
    old, _ = distributed_surface_patch_trace(
        xs, ys, beam, plate_thickness=.035)
    trace = hybrid.reference_trace()
    frames = np.broadcast_to(np.eye(3),
                             (hybrid.shell_count, 3, 3)).copy()
    state = beam.context.initial.mechanical
    assert len(hybrid.points) == 36
    assert np.max(np.abs(hybrid.evaluate(
        hybrid.reference_nodes, frames, state)[:, 0])) < 1e-12
    assert np.linalg.matrix_rank(trace) == 6
    assert np.max(np.abs(trace[:, 5:hybrid.shell_size:6])) < 1e-12

    # Terminal shell top positions stay fixed under this pitch field.
    # The old six position moments cannot see it; the new physical
    # director-pitch channel does.
    shell_step = np.zeros((hybrid.shell_count, 6))
    terminal = hybrid.reference_nodes[:, 0] >= .9-1e-12
    angle = 1e-5
    shell_step[terminal, 0] = -.0175*angle
    shell_step[terminal, 4] = angle
    direction = np.r_[shell_step.ravel(), np.zeros(beam.context.count)]
    assert np.max(np.abs(old@direction)) < 1e-12
    response = trace@direction
    assert abs(response[4]-angle) < 1e-12
    assert np.max(np.abs(np.delete(response, 4))) < 1e-12
    continuous = hybrid.linearized_continuous_gap_statistics(
        shell_step.ravel(), np.zeros(beam.context.count))
    assert abs(continuous["quadrature_mean_director_pitch_gap_rad"]-angle) < 1e-12
    assert abs(continuous["quadrature_max_abs_director_pitch_gap_rad"]-angle) < 1e-12
    assert continuous["quadrature_max_position_gap_m"] < 1e-12

    # A common transverse displacement slope with unchanged directors
    # leaves the connection closed while allowing shell shear.
    shell_step[:] = 0.
    shell_step[:, 2] = angle*hybrid.reference_nodes[:, 0]
    beam_step = np.zeros(beam.context.count)
    for i in range(beam.context.nodal_count//6):
        beam_step[6*i+2] = angle*beam.abscissae[i]
    shear = trace@np.r_[shell_step.ravel(), beam_step]
    assert np.max(np.abs(shear)) < 1e-12
    continuous = hybrid.linearized_continuous_gap_statistics(
        shell_step.ravel(), beam_step)
    assert continuous["quadrature_max_abs_director_pitch_gap_rad"] < 1e-12
    assert continuous["quadrature_max_position_gap_m"] < 1e-12


def test_c1_pitch_couple_drives_physical_shell_work_and_balances():
    xs, ys = local_grid(2)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    line = FiniteSurfaceLineHybrid(xs, ys, beam, thickness=.035)
    pitch_couple = -100.*line.reference_trace()[4, :line.shell_size]
    assert np.max(np.abs(pitch_couple[5::6])) < 1e-12
    tangent, physical = raw_e4_tangent(xs, ys, thickness=.035)
    fixed = {6*j*len(xs)+axis for j in range(len(ys)) for axis in range(6)}
    free = np.array([i for i in range(len(pitch_couple)) if i not in fixed])
    solution = np.zeros(len(pitch_couple))
    solution[free] = linalg.solve(
        tangent[np.ix_(free, free)], pitch_couple[free], assume_a="gen")
    reaction = tangent@solution-pitch_couple
    assert np.linalg.norm(reaction[free])/max(
        1., np.linalg.norm(pitch_couple[free])) < 1e-8
    locations = line.reference_nodes

    def wrench(force):
        nodal = force.reshape(-1, 6)
        return np.r_[nodal[:, :3].sum(axis=0),
                     (np.cross(locations, nodal[:, :3])+nodal[:, 3:]).sum(axis=0)]

    np.testing.assert_allclose(wrench(pitch_couple),
                               (0., 0., 0., 0., -100., 0.), atol=1e-10)
    np.testing.assert_allclose(wrench(pitch_couple+reaction),
                               0., atol=1e-7)
    work = float(pitch_couple@solution)
    physical_work = float(solution@(physical@solution))
    numerical_work = float(solution@((tangent-physical)@solution))
    assert work > 0.
    assert abs(work-physical_work-numerical_work)/work < 1e-8
    assert abs(numerical_work)/work < 1e-8


def test_c1_loaded_native_my_channel_allocation_and_pitch_ablation():
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    result = graded_terminal_coupling(
        4, beam, transverse_cells=4, plate_thickness=.035,
        include_line_hybrid_spaces=True, sparse_line_only=True,
        include_tip_my=True, include_pitch_ablation=True, force_sign=-1.)
    base_names = {
        "line_hybrid_p0_weighted", "line_hybrid_p1_weighted",
        "line_hybrid_dual_p1_weighted"}
    assert set(result["results"]) == base_names | {
        name+"_without_pitch" for name in base_names}
    for name in base_names:
        family = result["results"][name]
        case = family["cases"]["tip_my"]
        assert case["tip_ry_rad"] > 0.
        assert case["mechanics_error"] < 1e-8
        assert case["physical_shell_work_j"] > 0.
        assert case["total_external_work_j"] == pytest.approx(
            100.*case["tip_ry_rad"], rel=1e-10)
        shell_my = case["shell_support_wrench_N_Nm"][4]
        root_my = case["beam_root_wrench_N_Nm"][4]
        assert shell_my < -1.
        np.testing.assert_allclose(shell_my+root_my, -100., atol=1e-7)
        np.testing.assert_allclose(
            np.asarray(case["interface_shell_wrench_N_Nm"])+
            np.asarray(case["interface_beam_wrench_N_Nm"]),
            0., atol=1e-8)
        channels = case["continuous_line_recovery"][
            "channel_interface_side_work_and_wrench"]
        assert [item["channel"] for item in channels] == [
            "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        assert sum(item["shell_work_j"] for item in channels) == pytest.approx(
            case["interface_shell_work_j"], abs=1e-9)
        np.testing.assert_allclose(
            np.sum([item["shell_wrench_N_Nm"] for item in channels], axis=0),
            case["interface_shell_wrench_N_Nm"], atol=1e-8)

        ablated = result["results"][name+"_without_pitch"]
        changed = ablated["cases"]["tip_my"]
        assert ablated["constraint_rank"] == family["constraint_rank"] - (
            2 if name.endswith("p0_weighted") else 3)
        assert changed["mechanics_error"] < 1e-8
        assert changed["tip_ry_rad"] > case["tip_ry_rad"]
        assert changed["physical_shell_work_j"] > 0.
        assert abs(changed["omitted_pitch_gap"][
            "quadrature_mean_director_pitch_gap_rad"]) > 1e-7
        np.testing.assert_allclose(
            changed["shell_support_wrench_N_Nm"][4]+
            changed["beam_root_wrench_N_Nm"][4], -100., atol=1e-7)


def test_c1_hybrid_physical_wrench_includes_all_six_channels_and_offset():
    """Check reference resultant and beam/shell rigid work conjugacy."""
    xs, ys = local_grid(2)
    beam = build(16, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="straight", assemble=False)
    line = FiniteSurfaceLineHybrid(xs, ys, beam, thickness=.035)
    multipliers = np.array((21., -32., 47., 11., -13., 19.))
    trace = line.reference_trace()
    reaction = -(trace.T@multipliers)
    nodal = reaction[:line.shell_size].reshape(line.shell_count, 6)

    # The physical reaction is -C.T lambda. Moments are taken about the
    # center of the *top* connection, so the three force and three couple
    # channels have separately known, work-conjugate resultants.
    connection_center = np.array((.95, 0., .035/2))
    force = nodal[:, :3].sum(axis=0)
    moment = (np.cross(line.reference_nodes-connection_center,
                       nodal[:, :3])+nodal[:, 3:]).sum(axis=0)
    np.testing.assert_allclose(force, -multipliers[:3], rtol=0., atol=1e-11)
    np.testing.assert_allclose(moment, -multipliers[3:], rtol=0., atol=1e-11)

    translation = np.array((.003, -.002, .004))
    spin = np.array((.005, .007, -.011))
    virtual = np.zeros((line.shell_count, 6))
    virtual[:, :3] = (translation+
                      np.cross(spin, line.reference_nodes-connection_center))
    virtual[:, 3:] = spin
    np.testing.assert_allclose(
        float(np.sum(nodal*virtual)),
        float(force@translation+moment@spin), rtol=0., atol=1e-12)

    # A native B3 rigid variation also rotates each retained half-cell
    # frame. Omitting those coordinates gives a false beam-side moment.
    beam_virtual = np.zeros(beam.context.count)
    beam_nodes = (beam.context.initial.mechanical.positions+
                  beam.context.initial.mechanical.position_low)
    beam_virtual[:beam.context.nodal_count] = np.column_stack((
        translation+np.cross(spin, beam_nodes-connection_center),
        np.broadcast_to(spin, beam_nodes.shape))).ravel()
    for element in range(len(beam.context.elements)):
        start = beam.context.nodal_count+24*element
        beam_virtual[start:start+6] = np.tile(spin, 2)
    np.testing.assert_allclose(
        trace@np.r_[virtual.ravel(), beam_virtual], 0.,
        rtol=0., atol=1e-12)
    np.testing.assert_allclose(
        float(reaction[line.shell_size:]@beam_virtual),
        -float(force@translation+moment@spin), rtol=0., atol=1e-12)

    # Moving the report origin down to the shell reference surface restores
    # the eccentric lever arm without altering the underlying nodal work.
    reference_moment = (np.cross(line.reference_nodes-
                                 np.array((.95, 0., 0.)), nodal[:, :3])+
                        nodal[:, 3:]).sum(axis=0)
    np.testing.assert_allclose(
        reference_moment,
        moment+np.cross(np.array((0., 0., .035/2)), force),
        rtol=0., atol=1e-11)


def test_c1_continuous_pitch_sample_matches_actual_native_port_derivative():
    xs, ys = local_grid(2)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    line = FiniteSurfaceLineHybrid(xs, ys, beam, thickness=.035)
    rng = np.random.default_rng(1907)
    shell = rng.normal(scale=1e-5, size=(line.shell_count, 6))
    shell[line.reference_nodes[:, 0] == xs[0]] = 0.
    native = rng.normal(scale=1e-5, size=beam.context.count)
    native[:6] = 0.
    sampled = line.linearized_continuous_gap_statistics(shell.ravel(), native)
    frames = np.broadcast_to(np.eye(3), (line.shell_count, 3, 3))
    derivative = line.evaluate(
        line.reference_nodes, frames, beam.context.initial.mechanical,
        first=(shell, native), second=(shell, native))[4, 1]
    np.testing.assert_allclose(
        sampled["quadrature_mean_director_pitch_gap_rad"],
        derivative, rtol=0., atol=1e-12)
    assert sampled["quadrature_max_abs_director_pitch_gap_rad"] > 0.


def test_c1_six_channel_finite_state_matches_physical_side_actions():
    xs, ys = local_grid(2)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    patch = FiniteSurfaceLineHybrid(xs, ys, beam, thickness=.035)
    rng = np.random.default_rng(1921)
    total_shell = rng.normal(size=(patch.shell_count, 6))*np.array(
        (.001, .001, .001, .06, .06, .06))
    shell_direction = rng.normal(size=total_shell.shape)*np.array(
        (.002, .002, .002, .03, .03, .03))
    fixed = patch.reference_nodes[:, 0] == xs[0]
    total_shell[fixed] = 0.
    shell_direction[fixed] = 0.
    beam_total = rng.normal(scale=.001, size=beam.context.count)
    beam_direction = rng.normal(scale=.002, size=beam.context.count)
    beam_total[:6] = 0.
    beam_direction[:6] = 0.
    state = beam.context.advance(beam.context.initial.mechanical, beam_total)
    coordinates = patch.reference_nodes+total_shell[:, :3]
    frames = np.array([rotation_matrix_from_vector(theta)
                       for theta in total_shell[:, 3:]])
    multiplier = np.array((18., -23., 31., 7., -100., 13.))
    # KKT interface reaction is -C.T @ multiplier, so physical side
    # actions use its negative material-frame row coefficients.
    action = -multiplier
    native = float(-multiplier@patch.evaluate(
        coordinates, frames, state,
        (shell_direction, beam_direction),
        (shell_direction, beam_direction))[:, 1])

    shell_work = beam_work = 0.
    shell_wrench = np.zeros(6)
    beam_wrench = np.zeros(6)
    for point in patch.points:
        high, low, beam_frame = point.port.pose_pair(state)
        offset = np.array((0., point.y, -.05))
        beam_position = high+low+beam_frame@offset
        shell_position = patch._shell(coordinates, frames, point)
        gap = shell_position-beam_position
        force_material = action[:3]+np.array((
            -action[5]*point.y/patch.transverse_inertia,
            0., action[3]*point.y/patch.transverse_inertia))
        force = beam_frame@force_material
        port_step = point.port.tangent(state)@beam_direction
        spin = port_step[3:]
        beam_velocity = port_step[:3]+np.cross(spin, beam_frame@offset)
        shell_velocity = sum((shape*(shell_direction[int(node), :3]+
            np.cross(shell_direction[int(node), 3:],
                     frames[int(node)]@np.array((0., 0., patch.top))))
            for node, shape in zip(point.nodes, point.shape)), np.zeros(3))

        center_nodes, center_shape, director = patch._director(frames, point.x)
        x_axis, z_axis = beam_frame[:, 0], beam_frame[:, 2]
        a, c = float(x_axis@director), float(z_axis@director)
        pitch_gradient = (c*x_axis-a*z_axis)/(a*a+c*c)
        director_velocity = sum((shape*np.cross(
            shell_direction[int(node), 3:], frames[int(node), :, 2])
            for node, shape in zip(center_nodes, center_shape)), np.zeros(3))
        couple = action[4]*np.cross(director, pitch_gradient)
        weight = point.weight
        shell_work += weight*(force@shell_velocity+
                              action[4]*pitch_gradient@director_velocity)
        beam_work += weight*(-force@beam_velocity-
                             np.cross(gap, force)@spin-couple@spin)
        shell_wrench += weight*np.r_[force,
            np.cross(shell_position, force)+couple]
        beam_wrench += weight*np.r_[-force,
            -np.cross(beam_position, force)-np.cross(gap, force)-couple]

    np.testing.assert_allclose(native, shell_work+beam_work,
                               rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(shell_wrench+beam_wrench, 0.,
                               rtol=0., atol=1e-10)
    assert abs(shell_work) > 1e-6
    assert abs(beam_work) > 1e-6


def test_c1_segmented_top_face_law_preserves_positional_work():
    xs, ys = local_grid(2)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    top = FiniteSurfaceTopFaceLine(xs, ys, beam, thickness=.035)
    hybrid = FiniteSurfaceLineHybrid(xs, ys, beam, thickness=.035)
    state = beam.context.initial.mechanical
    zero = np.zeros(top.shell_size)
    assert np.max(np.abs(top.evaluate_e4_total(zero, state)[:, 0])) < 1e-12
    trace = top.reference_trace()
    assert np.linalg.matrix_rank(trace) == 6

    rng = np.random.default_rng(731)
    shell_step = rng.normal(size=top.shell_size)*1e-5
    beam_step = rng.normal(size=beam.context.count)*1e-5
    direction = np.r_[shell_step, beam_step]
    actual = top.evaluate_e4_total(
        zero, state, first=(shell_step, beam_step),
        second=(shell_step, beam_step))[:, 1]
    np.testing.assert_allclose(actual, trace@direction, rtol=0., atol=1e-10)

    # A supported, top-fixed pitch field is invisible to positional work,
    # while the distinct two-sided director-pitch weld does work.
    shell_step = np.zeros((top.shell_count, 6))
    terminal = top.reference_nodes[:, 0] >= .9-1e-12
    shell_step[terminal, 0] = -.0175e-5
    shell_step[terminal, 4] = 1e-5
    field = np.r_[shell_step.ravel(), np.zeros(beam.context.count)]
    assert np.max(np.abs(trace@field)) < 1e-12
    assert abs((hybrid.reference_trace()@field)[4]-1e-5) < 1e-12

    intervals = ((.9, .95), (.95, 1.))
    pieces = [FiniteSurfaceTopFaceLine(
        xs, ys, beam, thickness=.035, x_interval=interval)
        for interval in intervals]
    rows = np.vstack([piece.reference_trace() for piece in pieces])
    fixed_shell = {6*j*len(xs)+axis for j in range(len(ys))
                   for axis in range(6)}
    fixed_beam = {top.shell_size+index for index in beam.context.fixed}
    free = [i for i in range(rows.shape[1])
            if i not in fixed_shell and i not in fixed_beam]
    # One linear host per interval makes the two mean-z gaps and two
    # longitudinal pitch moments algebraically dependent. This is a
    # genuine discrete continuity relation, not a mixed-unit rank cutoff.
    np.testing.assert_allclose(
        rows[2]-rows[8]-.025*(rows[4]+rows[10]), 0.,
        rtol=0., atol=1e-12)
    assert np.linalg.matrix_rank(rows[:, free]) == 11

    # Two hosts per interval permit curvature inside each interval. This
    # distinguishes coarse primal representability from a duplicate in
    # the continuous six-channel positional law.
    finer_xs, finer_ys = local_grid(4)
    finer_beam = build(24, z_offset=.0675,
                       torsion_constant=square_torsion_constant(),
                       reference_profile="c1_tangent_2um", assemble=False)
    finer = [FiniteSurfaceTopFaceLine(
        finer_xs, finer_ys, finer_beam, thickness=.035,
        x_interval=interval) for interval in intervals]
    finer_rows = np.vstack([piece.reference_trace() for piece in finer])
    finer_shell_size = finer[0].shell_size
    finer_fixed = {6*j*len(finer_xs)+axis for j in range(len(finer_ys))
                   for axis in range(6)}
    finer_fixed |= {finer_shell_size+index
                    for index in finer_beam.context.fixed}
    finer_free = [i for i in range(finer_rows.shape[1])
                  if i not in finer_fixed]
    assert np.linalg.matrix_rank(finer_rows[:, finer_free]) == 12


def test_c1_top_face_line_actual_e4_chart_work_at_nonzero_rotation():
    xs, ys = local_grid(2)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    top = FiniteSurfaceTopFaceLine(xs, ys, beam, thickness=.035)
    state = beam.context.initial.mechanical
    rng = np.random.default_rng(910)
    q = np.zeros((top.shell_count, 6))
    q[:, 3:] = .02*rng.normal(size=(top.shell_count, 3))
    u = rng.normal(size=q.shape)*np.array((.001, .001, .001, .02, .02, .02))
    v = rng.normal(size=q.shape)*np.array((.001, .001, .001, .02, .02, .02))
    beam_zero = np.zeros(beam.context.count)
    multipliers = np.array((18., -23., 31., 7., -100., 13.))
    response = top.evaluate_e4_total(
        q.ravel(), state, first=(u.ravel(), beam_zero),
        second=(v.ravel(), beam_zero))

    def work(a, b):
        return float(multipliers@top.evaluate_e4_total(
            (q+a*u+b*v).ravel(), state)[:, 0])

    h = 1e-5
    first_fd = (work(h, 0.)-work(-h, 0.))/(2*h)
    mixed_h = 2e-3
    mixed_fd = (work(mixed_h, mixed_h)-work(mixed_h, -mixed_h)-
                work(-mixed_h, mixed_h)+work(-mixed_h, -mixed_h))/(4*mixed_h**2)
    assert abs(float(multipliers@response[:, 1])-first_fd) < 1e-7
    assert abs(float(multipliers@response[:, 3])-mixed_fd) < 1e-6


def test_c1_top_face_interval_work_with_finite_b3_port_motion():
    xs, ys = local_grid(2)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    patch = FiniteSurfaceTopFaceLine(
        xs, ys, beam, thickness=.035, x_interval=(.9, .95))
    rng = np.random.default_rng(917)
    shell = rng.normal(size=(patch.shell_count, 6))*np.array(
        (.001, .001, .001, .03, .03, .03))
    shell_u = rng.normal(size=shell.shape)*np.array(
        (.002, .002, .002, .04, .04, .04))
    shell_v = rng.normal(size=shell.shape)*np.array(
        (.002, .002, .002, .04, .04, .04))
    beam_base = np.zeros(beam.context.count)
    beam_base[6:] = rng.normal(size=beam.context.count-6)*.001
    beam_u = rng.normal(size=beam.context.count)*.001
    beam_v = rng.normal(size=beam.context.count)*.001
    beam_u[:6] = beam_v[:6] = 0.
    state = beam.context.advance(beam.context.initial.mechanical, beam_base)
    multipliers = np.array((18., -23., 31., 7., -100., 13.))
    actual = patch.evaluate_e4_total(
        shell.ravel(), state,
        first=(shell_u.ravel(), beam_u),
        second=(shell_v.ravel(), beam_v))

    def work(a, b):
        candidate = beam.context.advance(state, a*beam_u+b*beam_v)
        return float(multipliers@patch.evaluate_e4_total(
            (shell+a*shell_u+b*shell_v).ravel(), candidate)[:, 0])

    h1, h2 = 1e-5, 2e-3
    first_fd = (work(h1, 0.)-work(-h1, 0.))/(2*h1)
    second_fd = (work(0., h1)-work(0., -h1))/(2*h1)
    mixed_fd = (work(h2, h2)-work(h2, -h2)-
                work(-h2, h2)+work(-h2, -h2))/(4*h2*h2)
    assert abs(float(multipliers@actual[:, 1])-first_fd) < 1e-7
    assert abs(float(multipliers@actual[:, 2])-second_fd) < 1e-7
    assert abs(float(multipliers@actual[:, 3])-mixed_fd) < 1e-6


def test_c1_hybrid_has_a_support_admissible_transverse_section_null():
    xs, ys = local_grid(4)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    hybrid = FiniteSurfaceLineHybrid(xs, ys, beam, thickness=.035)
    nodal_y2 = hybrid.reference_nodes[:, 1]**2
    sampled_y2 = np.array([
        point.shape @ nodal_y2[point.nodes] for point in hybrid.points])
    weights = np.array([point.weight for point in hybrid.points])
    mean_y2 = float(weights @ sampled_y2)
    shell_step = np.zeros((hybrid.shell_count, 6))
    terminal = ((hybrid.reference_nodes[:, 0] >= .9-1e-12) &
                (np.abs(hybrid.reference_nodes[:, 1]) <= .05+1e-12))
    shell_step[terminal, 2] = .01*(nodal_y2[terminal]-mean_y2)
    direction = np.r_[shell_step.ravel(), np.zeros(beam.context.count)]

    # The even transverse field is nonzero only on free terminal nodes.
    # It probes the continuum section reduction; it is not a claim that
    # an exact weak line law must enforce pointwise surface no-slip.
    weak_gap = hybrid.reference_trace() @ direction
    sampled_gap = np.array([
        point.shape @ shell_step[point.nodes, 2] for point in hybrid.points])
    rms_gap = float(np.sqrt(weights @ (sampled_gap**2)))
    assert np.max(np.abs(weak_gap)) < 1e-12
    assert rms_gap > 1e-6


def test_c1_hybrid_actual_chart_has_objective_first_and_mixed_work():
    result = finite_chart_screen("c1_tangent_2um", hybrid=True)
    assert result["hybrid"] is True
    assert result["quadrature_points"] == 36
    assert result["reference_gap_max_m_rad"] < 1e-12
    assert result["reference_jacobian_direction_error"] < 1e-12
    assert result["nonzero_gap_norm_m_rad"] > 1e-4
    assert abs(result["nonzero_multiplier_mixed_work_j"]) > 1e-3
    assert result["first_direction_error_j"] < 1e-7
    assert result["mixed_direction_error_j"] < 1e-6
    assert result["finite_common_rigid_gap_error"] < 1e-12
    assert abs(result["common_rigid_virtual_work_j"]) < 1e-9


def test_c1_hybrid_two_host_intervals_preserve_chart_work_and_rank():
    xs, ys = local_grid(2)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    intervals = ((.9, .95), (.95, 1.))
    pieces = [FiniteSurfaceLineHybrid(
        xs, ys, beam, thickness=.035, x_interval=interval)
        for interval in intervals]
    trace = np.vstack([piece.reference_trace() for piece in pieces])
    fixed_shell = {6*j*len(xs)+axis for j in range(len(ys))
                   for axis in range(6)}
    fixed_beam = {pieces[0].shell_size+index
                  for index in beam.context.fixed}
    free = [i for i in range(trace.shape[1])
            if i not in fixed_shell and i not in fixed_beam]
    assert [len(piece.points) for piece in pieces] == [18, 18]
    assert np.linalg.matrix_rank(trace[:, free]) == 12
    assert np.max(np.abs(trace[:, 5:pieces[0].shell_size:6])) < 1e-12
    for interval in intervals:
        result = finite_chart_screen(
            "c1_tangent_2um", hybrid=True, x_interval=interval)
        assert result["reference_gap_max_m_rad"] < 1e-12
        assert result["reference_jacobian_direction_error"] < 1e-12
        assert result["first_direction_error_j"] < 1e-7
        assert result["mixed_direction_error_j"] < 1e-6
        assert result["finite_common_rigid_gap_error"] < 1e-12
        assert abs(result["common_rigid_virtual_work_j"]) < 1e-9


def test_c1_hybrid_uses_actual_additive_e4_total_rotation_chart():
    xs, ys = local_grid(2)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    patch = FiniteSurfaceLineHybrid(
        xs, ys, beam, thickness=.035, x_interval=(.9, .95))
    rng = np.random.default_rng(2047)
    count = patch.shell_count
    total = rng.normal(size=(count, 6))*np.array(
        (.001, .001, .001, .08, .08, .08))
    first_shell = rng.normal(size=(count, 6))*np.array(
        (.002, .002, .002, .04, .04, .04))
    second_shell = rng.normal(size=(count, 6))*np.array(
        (.002, .002, .002, .04, .04, .04))
    beam_base = np.zeros(beam.context.count)
    beam_base[6:beam.context.nodal_count] = rng.normal(
        size=beam.context.nodal_count-6)*.001
    first_beam = rng.normal(size=beam.context.count)*.001
    second_beam = rng.normal(size=beam.context.count)*.001
    first_beam[:6] = 0.
    second_beam[:6] = 0.
    state = beam.context.advance(beam.context.initial.mechanical, beam_base)
    first = (first_shell, first_beam)
    second = (second_shell, second_beam)
    multipliers = np.array((18., -23., 31., 7., -100., 13.))

    actual = patch.evaluate_e4_total(total.ravel(), state, first, second)

    def work(a, b):
        candidate = beam.context.advance(
            state, a*first_beam+b*second_beam)
        shell = total+a*first_shell+b*second_shell
        return float(multipliers@patch.evaluate_e4_total(
            shell.ravel(), candidate)[:, 0])

    h1, h2 = 1e-5, 2e-3
    first_fd = (work(h1, 0.)-work(-h1, 0.))/(2*h1)
    second_fd = (work(0., h1)-work(0., -h1))/(2*h1)
    mixed_fd = (work(h2, h2)-work(h2, -h2)-
                work(-h2, h2)+work(-h2, -h2))/(4*h2*h2)
    assert abs(float(multipliers@actual[:, 1])-first_fd) < 1e-7
    assert abs(float(multipliers@actual[:, 2])-second_fd) < 1e-7
    assert abs(float(multipliers@actual[:, 3])-mixed_fd) < 1e-6
    assert np.linalg.norm(total[:, 3:]) > .1


def test_c1_weighted_line_spaces_preserve_p0_and_are_distinct():
    xs, ys = local_grid(4)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    traces = {}
    for kind in ("piecewise_constant", "continuous_linear", "dual_linear"):
        rows, masses, patches = c1_hybrid_line_space(
            xs, ys, beam, kind=kind)
        traces[kind] = rows
        expected = ((.05, .05) if kind == "piecewise_constant" else
                    (.025, .05, .025))
        assert np.allclose(masses, expected, atol=1e-14)
        assert np.linalg.matrix_rank(rows) == len(rows)
        frames = np.broadcast_to(np.eye(3),
                                 (patches[0].shell_count, 3, 3))
        for candidate in patches:
            assert np.max(np.abs(candidate.evaluate(
                candidate.reference_nodes, frames,
                beam.context.initial.mechanical)[:, 0])) < 1e-12
    old = np.vstack([FiniteSurfaceLineHybrid(
        xs, ys, beam, thickness=.035, x_interval=interval).reference_trace()
        for interval in ((.9, .95), (.95, 1.))])
    assert np.max(np.abs(traces["piecewise_constant"]-old)) < 1e-12
    shell_size = 6*len(xs)*len(ys)
    fixed_shell = {6*j*len(xs)+axis for j in range(len(ys))
                   for axis in range(6)}
    fixed_beam = {shell_size+i for i in beam.context.fixed}
    free = [i for i in range(old.shape[1])
            if i not in fixed_shell and i not in fixed_beam]
    assert all(np.linalg.matrix_rank(rows[:, free]) == len(rows)
               for rows in traces.values())
    assert np.linalg.matrix_rank(np.vstack((
        traces["continuous_linear"],
        traces["dual_linear"]))[:, free]) == 24


def test_c1_weighted_line_spaces_reject_other_plate_reference():
    xs, ys = local_grid(4)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    with pytest.raises(ValueError, match="35 mm C1 reference"):
        c1_hybrid_line_space(xs, ys, beam,
                             kind="continuous_linear", thickness=.07)
    with pytest.raises(ValueError, match="35 mm plate"):
        graded_terminal_coupling(
            4, beam, plate_thickness=.07,
            include_line_hybrid_spaces=True)


def test_c1_weighted_spaces_use_own_density_norm():
    from threadpoolctl import threadpool_limits

    with pytest.raises(ValueError, match="segmented hybrid"):
        bounded_physical_compliance_screen(line_space_kind="continuous_linear")
    with threadpool_limits(limits=1):
        old = bounded_physical_compliance_screen(
            hybrid=True, segmented=True)[0]["p0_density_norm"]
        for kind, count in (("piecewise_constant", 2),
                            ("continuous_linear", 3),
                            ("dual_linear", 3)):
            result = bounded_physical_compliance_screen(
                hybrid=True, segmented=True,
                line_space_kind=kind)[0]
            norm = result["line_density_norm"]
            assert result["constraint_rank_on_free"] == 6*count
            assert len(norm["basis_integrals_m"]) == count
            assert norm["maximum_gram_bracket_ratio"] < 1.000001
            assert norm["smallest_nonaxial_signed_quotient_upper"] > 0.
            assert norm["smallest_nonaxial_mode_shell_action"] > 0.
            assert norm["smallest_nonaxial_mode_beam_action"] > 0.
            if kind == "piecewise_constant":
                assert norm["smallest_nonaxial_signed_quotient_upper"] == (
                    pytest.approx(old["smallest_nonaxial_signed_quotient_upper"],
                                  rel=1e-11))


def test_c1_sparse_line_action_preserves_dense_and_bounds_eight_cells():
    from threadpoolctl import threadpool_limits

    with pytest.raises(ValueError, match="sparse weighted"):
        bounded_physical_compliance_screen(
            hybrid=True, segmented=True,
            line_space_kind="piecewise_constant", include_shell_eight=True)
    with threadpool_limits(limits=1):
        dense = bounded_physical_compliance_screen(
            hybrid=True, segmented=True,
            line_space_kind="piecewise_constant")
        sparse = bounded_physical_compliance_screen(
            hybrid=True, segmented=True,
            line_space_kind="piecewise_constant",
            sparse_action=True, include_shell_eight=True)
    assert [r["patch_cells_per_side"] for r in sparse] == [2, 4, 8]
    for old, new in zip(dense, sparse[:2]):
        for key in ("smallest_signed_quotient_upper",
                    "smallest_nonaxial_signed_quotient_upper"):
            assert new["line_density_norm"][key] == pytest.approx(
                old["line_density_norm"][key], rel=1e-9)
    fine = sparse[-1]
    assert fine["constraint_rank_on_free"] == 12
    assert fine["line_density_norm"]["smallest_nonaxial_mode_shell_action"] > 0.
    assert fine["line_density_norm"]["smallest_nonaxial_mode_beam_action"] > 0.


def test_c1_reciprocal_bond_refinement_preserves_exterior_and_reference():
    from threadpoolctl import threadpool_limits

    with pytest.raises(ValueError, match="reciprocal screen"):
        bounded_physical_compliance_screen(
            hybrid=True, segmented=True, sparse_action=True,
            line_space_kind="piecewise_constant", c1_beam_half_cells=26)
    with threadpool_limits(limits=1):
        for half_cells, bond_cells in ((14, 4), (18, 8), (26, 16)):
            beam = build(half_cells, z_offset=.0675,
                         torsion_constant=square_torsion_constant(),
                         reference_profile="c1_tangent_2um")
            np.testing.assert_allclose(beam.abscissae[:11],
                                       (0., .125, .25, .375, .5, .625,
                                        .75, .7875, .825, .8625, .9),
                                       rtol=0., atol=0.)
            assert len(beam.abscissae) == half_cells+1
            assert np.count_nonzero(np.diff(beam.abscissae[-bond_cells-1:]) > 0) == bond_cells
            assert beam.dense_oracle_max_error < 1e-9
        result = bounded_physical_compliance_screen(
            hybrid=True, segmented=True, sparse_action=True,
            line_space_kind="piecewise_constant", c1_beam_half_cells=26,
            reciprocal_shell_x_cells=2)[0]
        p1 = bounded_physical_compliance_screen(
            hybrid=True, segmented=True, sparse_action=True,
            line_space_kind="continuous_linear", c1_beam_half_cells=26,
            reciprocal_shell_x_cells=2)[0]
    assert (result["bond_x_cells"], result["bond_y_cells"]) == (2, 4)
    assert result["constraint_rank_on_free"] == 12
    assert result["reference_gap_m"] < 1e-12
    assert result["line_density_norm"]["smallest_nonaxial_mode_shell_action"] > 0.
    assert result["line_density_norm"]["smallest_nonaxial_mode_beam_action"] > 0.
    assert p1["constraint_rank_on_free"] == 18
    norm = p1["line_density_norm"]
    assert (norm["smallest_nonaxial_mode_shell_action"]+
            norm["smallest_nonaxial_mode_beam_action"] == pytest.approx(
                norm["smallest_nonaxial_signed_quotient_upper"], abs=1e-8))


def test_c1_line_quadrature_control_preserves_reference_and_default():
    from threadpoolctl import threadpool_limits

    xs, ys = local_grid(2, transverse_cells=4)
    with threadpool_limits(limits=1):
        beam = build(12, z_offset=.0675,
                     torsion_constant=square_torsion_constant(),
                     reference_profile="c1_tangent_2um")
        for kind in ("piecewise_constant", "continuous_linear", "dual_linear"):
            default, masses, _ = c1_hybrid_line_space(
                xs, ys, beam, kind=kind)
            explicit, explicit_masses, _ = c1_hybrid_line_space(
                xs, ys, beam, kind=kind, quadrature_order=3)
            fine, fine_masses, patches = c1_hybrid_line_space(
                xs, ys, beam, kind=kind, quadrature_order=5)
            np.testing.assert_array_equal(default, explicit)
            np.testing.assert_array_equal(masses, explicit_masses)
            assert np.max(np.abs(fine-default)) < 1e-8
            assert np.max(np.abs(fine_masses-masses)) < 1e-12
            assert all(len(patch.points) > 0 for patch in patches)
    with pytest.raises(ValueError, match="fifth-order quadrature"):
        graded_terminal_coupling(2, beam, transverse_cells=4,
                                 plate_thickness=.035,
                                 include_line_hybrid_spaces=True,
                                 line_quadrature_order=5)


def test_c1_weighted_endpoint_work_uses_actual_e4_chart():
    # Minimal valid shell grid keeps the same two bond hosts and beam trace.
    xs = np.array((-.5, .9, .95, 1., 1.5))
    ys = np.array((-.5, -.05, 0., .05, .5))
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    rng = np.random.default_rng(2048)
    shell = rng.normal(size=(len(xs)*len(ys), 6))*np.array(
        (.001, .001, .001, .08, .08, .08))
    du = rng.normal(size=shell.shape)*np.array(
        (.002, .002, .002, .04, .04, .04))
    dv = rng.normal(size=shell.shape)*np.array(
        (.002, .002, .002, .04, .04, .04))
    beam_base = np.zeros(beam.context.count)
    beam_base[6:beam.context.nodal_count] = rng.normal(
        size=beam.context.nodal_count-6)*.001
    bu = rng.normal(size=beam.context.count)*.001
    bv = rng.normal(size=beam.context.count)*.001
    bu[:6] = 0.
    bv[:6] = 0.
    state = beam.context.advance(beam.context.initial.mechanical, beam_base)
    multiplier = np.array((18., -23., 31., 7., -100., 13.))
    for kind in ("continuous_linear", "dual_linear"):
        patch = c1_hybrid_line_space(xs, ys, beam, kind=kind)[2][0]
        exact = patch.evaluate_e4_total(
            shell.ravel(), state, (du, bu), (dv, bv))

        def work(a, b):
            candidate = beam.context.advance(state, a*bu+b*bv)
            return float(multiplier@patch.evaluate_e4_total(
                (shell+a*du+b*dv).ravel(), candidate)[:, 0])

        h1, h2 = 1e-5, 2e-3
        first = (work(h1, 0.)-work(-h1, 0.))/(2*h1)
        mixed = (work(h2, h2)-work(h2, -h2)-
                 work(-h2, h2)+work(-h2, -h2))/(4*h2*h2)
        assert abs(float(multiplier@exact[:, 1])-first) < 1e-7
        assert abs(float(multiplier@exact[:, 3])-mixed) < 1e-6


@pytest.mark.parametrize("diagonal", ("bl_tr", "br_tl"))
def test_c1_mixed_s3_host_reference_and_actual_chart_work(diagonal):
    xs, ys = local_grid(2, transverse_cells=4)
    selected = c1_mixed_s3_cells(xs, ys)
    assert len(selected) == 4
    nodes_xyz = np.array([(x, y, 0.) for y in ys for x in xs])
    for x, y in ((.925, -.0375), (.9125, -.0125),
                 (.925, .0125), (.9125, .0375)):
        nodes, shape = structured_c1_mixed_point_shape(
            xs, ys, x, y, selected, diagonal=diagonal)
        assert len(nodes) == 3
        assert np.min(shape) >= -1e-14
        np.testing.assert_allclose(shape.sum(), 1., atol=1e-14)
        np.testing.assert_allclose(shape@nodes_xyz[nodes, :2],
                                   (x, y), atol=1e-14)
    # Both adjacent triangle traces have the same interpolated field on
    # their diagonal, including the rotational nodal chart variables.
    x, y = .925, -.0375
    nodes, shape = structured_c1_mixed_point_shape(
        xs, ys, x, y, selected, diagonal=diagonal)
    full, _ = structured_q4_point_shape(xs, ys, x, y)
    arbitrary_field = np.arange(len(nodes_xyz), dtype=float)**2
    endpoints = full[[0, 2]] if diagonal == "bl_tr" else full[[1, 3]]
    diagonal_value = .5*sum(arbitrary_field[endpoints])
    assert shape@arbitrary_field[nodes] == pytest.approx(
        diagonal_value, abs=1e-11)
    opposite = full[[0, 2, 3]] if diagonal == "bl_tr" else full[[1, 2, 3]]
    opposite_weights = (.5, .5, 0.) if diagonal == "bl_tr" else (.5, 0., .5)
    assert np.array(opposite_weights) @ arbitrary_field[opposite] == pytest.approx(
        diagonal_value, abs=1e-14)

    shell_k, physical = raw_e4_mixed_c1_tangent(
        xs, ys, diagonal=diagonal)
    assert shell_k.shape == physical.shape == (6*len(xs)*len(ys),)*2
    assert np.isfinite(shell_k).all() and np.isfinite(physical).all()
    assert np.linalg.norm(shell_k-shell_k.T)/np.linalg.norm(shell_k) < 1e-9

    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    rng = np.random.default_rng(2094)
    shell = rng.normal(size=(len(xs)*len(ys), 6))*np.array(
        (.001, .001, .001, .08, .08, .08))
    du = rng.normal(size=shell.shape)*np.array(
        (.002, .002, .002, .04, .04, .04))
    dv = rng.normal(size=shell.shape)*np.array(
        (.002, .002, .002, .04, .04, .04))
    base = np.zeros(beam.context.count)
    base[6:beam.context.nodal_count] = rng.normal(
        size=beam.context.nodal_count-6)*.001
    bu = rng.normal(size=beam.context.count)*.001
    bv = rng.normal(size=beam.context.count)*.001
    bu[:6] = bv[:6] = 0.
    state = beam.context.advance(beam.context.initial.mechanical, base)
    multiplier = np.array((18., -23., 31., 7., -100., 13.))
    for kind in ("piecewise_constant", "continuous_linear", "dual_linear"):
        rows, masses, patches = c1_hybrid_line_space(
            xs, ys, beam, kind=kind, mixed_s3=True,
            mixed_s3_diagonal=diagonal)
        assert rows.shape[0] == 6*len(masses)
        assert np.all(masses > 0.)
        patch = patches[0]
        assert len(patch.points) > 0
        assert np.max(np.abs(patch.evaluate_e4_total(
            np.zeros(6*len(nodes_xyz)),
            beam.context.initial.mechanical)[:, 0])) < 1e-12
        exact = patch.evaluate_e4_total(
            shell.ravel(), state, (du, bu), (dv, bv))

        def work(a, b):
            candidate = beam.context.advance(state, a*bu+b*bv)
            return float(multiplier@patch.evaluate_e4_total(
                (shell+a*du+b*dv).ravel(), candidate)[:, 0])

        h1, h2 = 1e-5, 2e-3
        first = (work(h1, 0.)-work(-h1, 0.))/(2*h1)
        mixed = (work(h2, h2)-work(h2, -h2)-
                 work(-h2, h2)+work(-h2, -h2))/(4*h2*h2)
        assert abs(float(multiplier@exact[:, 1])-first) < 1e-7
        assert abs(float(multiplier@exact[:, 3])-mixed) < 1e-6


@pytest.mark.parametrize("diagonal", ("bl_tr", "br_tl"))
def test_c1_mixed_s3_line_modes_and_loaded_balance(diagonal):
    from threadpoolctl import threadpool_limits

    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    with threadpool_limits(limits=1):
        loaded = graded_terminal_coupling(
            2, beam, transverse_cells=4, plate_thickness=.035,
            include_line_hybrid_spaces=True, sparse_line_only=True,
            mixed_s3=True, mixed_s3_diagonal=diagonal,
            force_sign=-1.)
        for kind, count in (("piecewise_constant", 2),
                            ("continuous_linear", 3),
                            ("dual_linear", 3)):
            result = bounded_physical_compliance_screen(
                hybrid=True, segmented=True, sparse_action=True,
                reciprocal_shell_x_cells=None, line_space_kind=kind,
                mixed_s3=True, mixed_s3_diagonal=diagonal)[0]
            assert result["shell_topology"] == "Q4/S3-V2D"
            assert result["s3_diagonal"] == diagonal
            assert result["constraint_rank_on_free"] == 6*count
            assert result["reference_gap_m"] < 1e-12
            norm = result["line_density_norm"]
            assert norm["smallest_nonaxial_mode_shell_action"] > 0.
            assert norm["smallest_nonaxial_mode_beam_action"] > 0.
    assert loaded["shell_topology"] == "Q4/S3-V2D"
    assert loaded["s3_diagonal"] == diagonal
    assert loaded["e4_elements"] == 124
    for family in loaded["results"].values():
        assert family["constraint_rank"] in (12, 18)
        for case in family["cases"].values():
            assert case["mechanics_error"] < 1e-7
            assert case["physical_shell_work_j"] > 0.
            assert case["total_external_work_j"] > 0.
            tip = case["terminal_station_kinematics"]["1000"]
            np.testing.assert_allclose(
                tip["shell_top_edge_sample_y_m"],
                (-.05,-.025,0.,.025,.05), atol=1e-14)
            assert len(tip["shell_top_edge_mean_translation_m"]) == 3
            assert np.isfinite(tip["shell_top_edge_mean_translation_m"]).all()
            assert tip["beam_nodal_rotation_rad"][0] == pytest.approx(
                tip["beam_nodal_rx_rad"], abs=1e-14)
            assert tip["beam_nodal_translation_m"][2] == pytest.approx(
                case["tip_uz_m"], abs=1e-14)
            assert np.isfinite((tip["shell_top_slope_z_x_rad"],
                                tip["shell_director_ry_rad"],
                                tip["shell_local_patch_mean_ry_rad"])).all()


def test_c1_hybrid_physical_director_spin_is_null_at_finite_tilt():
    xs, ys = local_grid(2)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    patch = FiniteSurfaceLineHybrid(xs, ys, beam, thickness=.035)
    rng = np.random.default_rng(710)
    theta = rng.normal(size=(patch.shell_count, 3))*.12
    frames = np.array([rotation_matrix_from_vector(value)
                       for value in theta])
    coordinates = patch.reference_nodes.copy()
    shell_spin = np.zeros((patch.shell_count, 6))
    shell_spin[:, 3:] = frames[:, :, 2]
    direction = (shell_spin, np.zeros(beam.context.count))
    response = patch.evaluate(
        coordinates, frames, beam.context.initial.mechanical,
        direction, direction)
    assert np.max(np.abs(response[:, 1:])) < 1e-12
