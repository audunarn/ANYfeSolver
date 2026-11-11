"""Focused standard element-edge beam-shell coupling transfer tests.

These tests exercise the actual production constraint transformation
(``assembly.build_constraint_transformation``) and force diagnostics
(``assembly.compute_constraint_force_diagnostics``) for the interpolated
edge route (``mesh_gen.InterpolatedBeamShellMPCElement``) and the one-to-one
eccentric route (``elements.CoupledBeamShellElement``), together with the
fail-closed input admission adopted for the coupling boundary.

Adopted mechanical contract (fixed common-offset map):

    u_b    = sum_i N_i u_i + (sum_i N_i theta_i) x r
    theta_b = sum_i N_i theta_i

The oracles below restate this map independently; they do not call the
production constraint builders.  All identity residuals are dimensionless
and bounded by 1e-10 against the separately stated physical scales:

    TRANSLATION_SCALE = 1.0     m
    ROTATION_SCALE    = 1e-3    rad
    FORCE_SCALE       = 1e3     N
    COUPLE_SCALE      = 1e3     N*m
    WORK_SCALE        = 1e3     N*m  (FORCE_SCALE * TRANSLATION_SCALE)
"""

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import audit_constraints
from anyfesolver.analysis_session import AnalysisSession
from anyfesolver.assembly import (
    build_constraint_transformation,
    compute_constraint_force_diagnostics,
    reconstruct_full_solution,
)
from anyfesolver.boundary import FixedSupport, LoadCase
from anyfesolver.constraint_audit import require_valid_constraints
from anyfesolver.elements import BeamElement, CoupledBeamShellElement
from anyfesolver.fe_core import FEModel
from anyfesolver.mesh_gen import (
    COUPLING_PARTITION_OF_UNITY_ATOL,
    InterpolatedBeamShellMPCElement,
    generate_simple_panel_mesh,
)
from anyfesolver.validation import mpc_constraint_residuals

ATOL = 1.0e-10
TRANSLATION_SCALE = 1.0
ROTATION_SCALE = 1.0e-3
FORCE_SCALE = 1.0e3
COUPLE_SCALE = 1.0e3
WORK_SCALE = FORCE_SCALE * TRANSLATION_SCALE


def _steel(model):
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)


def _interpolated_model(node_coords, beam_node_id, shell_node_ids, weights, r, *, fixed=()):
    model = FEModel("standard_edge_transfer")
    _steel(model)
    for node_id, coords in node_coords.items():
        model.add_node(int(node_id), *[float(c) for c in coords])
    element = InterpolatedBeamShellMPCElement(
        1,
        int(beam_node_id),
        [int(n) for n in shell_node_ids],
        np.asarray(weights, dtype=float),
        np.asarray(r, dtype=float),
    )
    model.add_element(1, element)
    if fixed:
        model.add_boundary_condition(FixedSupport("fixed_masters", [int(n) for n in fixed]))
    return model, element


def _production_transform(model):
    total = int(model.mesh.dof_manager.total_dofs)
    K = sparse.eye(total, format="csr")
    zero = np.zeros(total, dtype=float)
    _K, _F, T, u0, independent, info = build_constraint_transformation(K, zero, model)
    return T, u0, np.asarray(independent, dtype=int), info


def _full_from_master_states(model, master_states):
    total = int(model.mesh.dof_manager.total_dofs)
    full = np.zeros(total, dtype=float)
    for node_id, (u, theta) in master_states.items():
        dofs = model.mesh.get_node(int(node_id)).dofs
        full[dofs[:3]] = u
        full[dofs[3:6]] = theta
    return full


def _beam_state(model, beam_node_id, u):
    dofs = model.mesh.get_node(int(beam_node_id)).dofs
    return u[dofs[:3]], u[dofs[3:6]]


def _oracle_beam_state(master_states_in_order, weights, r):
    """Independent restatement of the adopted common-offset map."""
    u_sum = np.zeros(3, dtype=float)
    theta_sum = np.zeros(3, dtype=float)
    for (u, theta), weight in zip(master_states_in_order, weights):
        u_sum = u_sum + float(weight) * np.asarray(u, dtype=float)
        theta_sum = theta_sum + float(weight) * np.asarray(theta, dtype=float)
    return u_sum + np.cross(theta_sum, np.asarray(r, dtype=float)), theta_sum


def _assert_beam_matches_oracle(model, beam_node_id, shell_node_ids, weights, r, master_states):
    """Drive the production T with the master field and compare to the oracle."""
    ordered = [(np.asarray(master_states[n][0]), np.asarray(master_states[n][1])) for n in shell_node_ids]
    expected_u, expected_theta = _oracle_beam_state(ordered, weights, r)
    T, u0, independent, _info = _production_transform(model)
    full = _full_from_master_states(model, master_states)
    u = reconstruct_full_solution(T, full[independent], u0)
    got_u, got_theta = _beam_state(model, beam_node_id, u)
    translation_error = float(np.linalg.norm(got_u - expected_u)) / TRANSLATION_SCALE
    rotation_error = float(np.linalg.norm(got_theta - expected_theta)) / ROTATION_SCALE
    assert translation_error <= ATOL
    assert rotation_error <= ATOL
    residuals = mpc_constraint_residuals(model, u)
    max_residual = max((abs(v) for v in residuals.values()), default=0.0)
    assert max_residual <= ATOL
    return u, got_u, got_theta


# ---------------------------------------------------------------------------
# Actual interpolated edge route: kinematics through the production T
# ---------------------------------------------------------------------------

def test_interpolated_linear_weights_reproduce_affine_field_with_common_offset():
    # Actual 2-node collinear edge: the linear edge shape weights are
    # [1 - t, t] at a stated interior position 0 < t < 1.
    r = np.array([0.03, -0.02, 0.12])
    coords = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)}
    t = 0.3
    weights = np.array([1.0 - t, t])
    shell_ids = [1, 2]
    attachment = sum(w * np.asarray(coords[n]) for n, w in zip(shell_ids, weights))
    coords[10] = tuple(attachment + r)
    model, _element = _interpolated_model(coords, 10, shell_ids, weights, r)

    # Affine translation and rotation fields along the edge.
    master_states = {}
    for n in shell_ids:
        x = np.asarray(coords[n])
        master_states[n] = (
            np.array([0.02, -0.01, 0.03]) + 0.01 * x,
            np.array([1.0e-3, -2.0e-3, 5.0e-4]) + 4.0e-4 * x[0] * np.ones(3),
        )
    _assert_beam_matches_oracle(model, 10, shell_ids, weights, r, master_states)


def test_interpolated_quadratic_weights_admit_negative_and_zero_values():
    r = np.array([0.0, 0.05, -0.1])
    coords = {1: (0.0, 0.0, 0.0), 2: (0.5, 0.0, 0.0), 3: (1.0, 0.0, 0.0)}
    shell_ids = [1, 2, 3]
    # Quadratic edge shape weights at xi=0.3: one weight is negative.
    negative_weights = np.array([0.28, 0.84, -0.12])
    assert abs(negative_weights.sum() - 1.0) <= COUPLING_PARTITION_OF_UNITY_ATOL
    assert np.any(negative_weights < 0.0)
    coords[10] = (0.3, 0.0, 0.0)
    model, element = _interpolated_model(coords, 10, shell_ids, negative_weights, r)
    assert element.shape_weights.shape == (3,)

    master_states = {
        1: (np.array([0.01, 0.0, -0.02]), np.array([1.0e-3, 0.0, 0.0])),
        2: (np.array([-0.03, 0.02, 0.01]), np.array([0.0, 2.0e-3, -1.0e-3])),
        3: (np.array([0.05, -0.04, 0.0]), np.array([-2.0e-3, 1.0e-3, 3.0e-3])),
    }
    _assert_beam_matches_oracle(model, 10, shell_ids, negative_weights, r, master_states)

    # Zero weights are legitimate (midpoint of a quadratic edge).
    zero_weights = np.array([0.0, 1.0, 0.0])
    model_zero, _element = _interpolated_model(coords, 10, shell_ids, zero_weights, r)
    _assert_beam_matches_oracle(model_zero, 10, shell_ids, zero_weights, r, master_states)


def test_interpolated_map_is_infinitesimal_rigid_motion_consistent():
    # Physically matched record: the beam node sits at attachment + r, so a
    # rigid infinitesimal motion of the shell edge must move the beam node
    # with the same motion.
    r = np.array([0.0, 0.0, 0.2])
    coords = {1: (0.0, 0.0, 0.0), 2: (0.5, 0.0, 0.0), 3: (1.0, 0.0, 0.0)}
    weights = np.array([0.375, 0.25, 0.375])
    shell_ids = [1, 2, 3]
    attachment = sum(w * np.asarray(coords[n]) for n, w in zip(shell_ids, weights))
    coords[10] = tuple(attachment + r)
    model, _element = _interpolated_model(coords, 10, shell_ids, weights, r)

    u0 = np.array([0.004, -0.002, 0.001])
    theta0 = np.array([1.0e-3, -3.0e-4, 2.0e-3])
    master_states = {
        n: (u0 + np.cross(theta0, np.asarray(coords[n])), theta0) for n in shell_ids
    }
    u, got_u, got_theta = _assert_beam_matches_oracle(
        model, 10, shell_ids, weights, r, master_states
    )
    expected_u = u0 + np.cross(theta0, np.asarray(coords[10]))
    assert np.linalg.norm(got_u - expected_u) / TRANSLATION_SCALE <= ATOL
    assert np.linalg.norm(got_theta - theta0) / ROTATION_SCALE <= ATOL


def test_interpolated_map_is_frame_covariant():
    rng = np.random.default_rng(20261016)
    r = np.array([0.04, 0.01, -0.07])
    coords = {1: (0.0, 0.0, 0.0), 2: (0.4, 0.2, 0.0), 3: (0.8, 0.4, 0.1)}
    weights = np.array([0.2, 0.5, 0.3])
    shell_ids = [1, 2, 3]
    attachment = sum(w * np.asarray(coords[n]) for n, w in zip(shell_ids, weights))
    coords[10] = tuple(attachment + r)
    master_states = {
        n: (
            rng.uniform(-0.02, 0.02, 3),
            rng.uniform(-5.0e-3, 5.0e-3, 3),
        )
        for n in shell_ids
    }

    def rotation_matrix():
        cz, sz = np.cos(0.5), np.sin(0.5)
        cx, sx = np.cos(0.3), np.sin(0.3)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        return Rz @ Rx

    R = rotation_matrix()
    rotated_coords = {n: tuple(R @ np.asarray(c)) for n, c in coords.items()}
    rotated_states = {
        n: (R @ master_states[n][0], R @ master_states[n][1]) for n in shell_ids
    }

    model_a, _ = _interpolated_model(coords, 10, shell_ids, weights, r)
    u_a, got_u_a, got_theta_a = _assert_beam_matches_oracle(
        model_a, 10, shell_ids, weights, r, master_states
    )
    model_b, _ = _interpolated_model(rotated_coords, 10, shell_ids, weights, R @ r)
    u_b, got_u_b, got_theta_b = _assert_beam_matches_oracle(
        model_b, 10, shell_ids, weights, R @ r, rotated_states
    )
    assert np.linalg.norm(got_u_b - R @ got_u_a) / TRANSLATION_SCALE <= ATOL
    assert np.linalg.norm(got_theta_b - R @ got_theta_a) / ROTATION_SCALE <= ATOL


def test_interpolated_map_is_invariant_to_edge_reversal():
    r = np.array([0.02, -0.03, 0.06])
    coords = {1: (0.0, 0.0, 0.0), 2: (0.5, 0.0, 0.0), 3: (1.0, 0.0, 0.0)}
    weights = np.array([0.28, 0.84, -0.12])
    shell_ids = [1, 2, 3]
    coords[10] = (0.3, 0.0, 0.0)
    master_states = {
        1: (np.array([0.01, 0.02, -0.01]), np.array([1.0e-3, 0.0, -2.0e-3])),
        2: (np.array([0.0, -0.01, 0.03]), np.array([-1.0e-3, 2.0e-3, 0.0])),
        3: (np.array([-0.02, 0.0, 0.01]), np.array([0.0, -1.0e-3, 3.0e-3])),
    }

    forward_model, _ = _interpolated_model(coords, 10, shell_ids, weights, r)
    _u_f, forward_u, forward_theta = _assert_beam_matches_oracle(
        forward_model, 10, shell_ids, weights, r, master_states
    )

    reversed_ids = list(reversed(shell_ids))
    reversed_weights = weights[::-1].copy()
    reversed_model, _ = _interpolated_model(coords, 10, reversed_ids, reversed_weights, r)
    _u_r, reversed_u, reversed_theta = _assert_beam_matches_oracle(
        reversed_model, 10, reversed_ids, reversed_weights, r, master_states
    )
    assert np.linalg.norm(forward_u - reversed_u) / TRANSLATION_SCALE <= ATOL
    assert np.linalg.norm(forward_theta - reversed_theta) / ROTATION_SCALE <= ATOL


def test_corner_shared_by_two_edges_keeps_unique_slave_ownership():
    # Node 3 is the corner shared by edge (1,2,3) and edge (3,4,5); each edge
    # owns its own beam slave.
    r_a = np.array([0.0, 0.0, 0.15])
    r_b = np.array([0.01, 0.0, -0.05])
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (0.5, 0.0, 0.0),
        3: (1.0, 0.0, 0.0),
        4: (1.0, 0.5, 0.0),
        5: (1.0, 1.0, 0.0),
        10: (0.3, 0.0, 0.15),
        11: (1.0, 0.7, -0.05),
    }
    weights_a = np.array([0.28, 0.84, -0.12])
    weights_b = np.array([0.375, 0.25, 0.375])

    model = FEModel("corner_junction")
    _steel(model)
    for node_id, xyz in coords.items():
        model.add_node(node_id, *[float(c) for c in xyz])
    model.add_element(
        1,
        InterpolatedBeamShellMPCElement(1, 10, [1, 2, 3], weights_a, r_a),
    )
    model.add_element(
        2,
        InterpolatedBeamShellMPCElement(2, 11, [3, 4, 5], weights_b, r_b),
    )
    report = audit_constraints(model)
    assert report.status == "ok"
    assert report.feasible
    assert report.mpc_slave_dofs == 12

    master_states = {
        1: (np.array([0.01, 0.0, 0.0]), np.array([1.0e-3, 0.0, 0.0])),
        2: (np.array([0.0, 0.01, 0.0]), np.array([0.0, 1.0e-3, 0.0])),
        3: (np.array([0.0, 0.0, 0.01]), np.array([0.0, 0.0, 1.0e-3])),
        4: (np.array([-0.01, 0.0, 0.0]), np.array([-1.0e-3, 0.0, 0.0])),
        5: (np.array([0.0, -0.01, 0.0]), np.array([0.0, -1.0e-3, 0.0])),
    }
    T, u0, independent, info = _production_transform(model)
    assert info["num_mpc_slave_dofs"] == 12
    full = _full_from_master_states(model, master_states)
    u = reconstruct_full_solution(T, full[independent], u0)

    expected_u_a, expected_theta_a = _oracle_beam_state(
        [(master_states[n][0], master_states[n][1]) for n in [1, 2, 3]], weights_a, r_a
    )
    expected_u_b, expected_theta_b = _oracle_beam_state(
        [(master_states[n][0], master_states[n][1]) for n in [3, 4, 5]], weights_b, r_b
    )
    got_u_a, got_theta_a = _beam_state(model, 10, u)
    got_u_b, got_theta_b = _beam_state(model, 11, u)
    assert np.linalg.norm(got_u_a - expected_u_a) / TRANSLATION_SCALE <= ATOL
    assert np.linalg.norm(got_theta_a - expected_theta_a) / ROTATION_SCALE <= ATOL
    assert np.linalg.norm(got_u_b - expected_u_b) / TRANSLATION_SCALE <= ATOL
    assert np.linalg.norm(got_theta_b - expected_theta_b) / ROTATION_SCALE <= ATOL
    residuals = mpc_constraint_residuals(model, u)
    assert max(abs(v) for v in residuals.values()) <= ATOL


def test_endpoint_degenerates_to_one_to_one_map():
    # A corner endpoint of an edge (weights [1, 0, 0]) must reproduce the
    # one-to-one CoupledBeamShellElement map with the same offset.
    r = np.array([0.05, -0.01, 0.02])
    coords = {1: (0.0, 0.0, 0.0), 2: (0.5, 0.0, 0.0), 3: (1.0, 0.0, 0.0), 10: (0.05, -0.01, 0.02)}
    master_state = (np.array([0.02, -0.03, 0.01]), np.array([1.0e-3, -2.0e-3, 5.0e-4]))

    interp_model, _ = _interpolated_model(
        coords, 10, [1, 2, 3], np.array([1.0, 0.0, 0.0]), r
    )
    states = {1: master_state, 2: (np.zeros(3), np.zeros(3)), 3: (np.zeros(3), np.zeros(3))}
    _u_i, interp_u, interp_theta = _assert_beam_matches_oracle(
        interp_model, 10, [1, 2, 3], np.array([1.0, 0.0, 0.0]), r, states
    )

    one_to_one = FEModel("one_to_one_control")
    _steel(one_to_one)
    one_to_one.add_node(1, 0.0, 0.0, 0.0)
    one_to_one.add_node(10, 0.05, -0.01, 0.02)
    one_to_one.add_element(
        1, CoupledBeamShellElement(1, beam_node_id=10, shell_node_id=1, material_name="steel", eccentricity=r)
    )
    T, u0, independent, _info = _production_transform(one_to_one)
    full = np.zeros(int(one_to_one.mesh.dof_manager.total_dofs), dtype=float)
    dofs = one_to_one.mesh.get_node(1).dofs
    full[dofs[:3]] = master_state[0]
    full[dofs[3:6]] = master_state[1]
    u = reconstruct_full_solution(T, full[independent], u0)
    got_u, got_theta = _beam_state(one_to_one, 10, u)
    assert np.linalg.norm(interp_u - got_u) / TRANSLATION_SCALE <= ATOL
    assert np.linalg.norm(interp_theta - got_theta) / ROTATION_SCALE <= ATOL


def test_shared_node_control_rejects_self_coupling():
    # A beam node that is already a shell node must not acquire a second
    # constraint; the production audit rejects the self-referential MPC.
    model = FEModel("shared_node_control")
    _steel(model)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_element(
        1, CoupledBeamShellElement(1, beam_node_id=1, shell_node_id=1, material_name="steel")
    )
    report = audit_constraints(model)
    assert report.status == "invalid"
    assert not report.feasible
    with pytest.raises(ValueError):
        require_valid_constraints(model)


def test_shared_node_direct_composition_installs_no_mpc_constraints():
    # Positive shared-node control through the supported direct FEModel
    # composition: a standard shell panel plus a standard BeamElement on the
    # SAME two shell edge node IDs.  The beam and the shell edge share the
    # exact same DOF mapping and no coupling element or MPC slave constraint
    # is installed.  (generate_stiffened_panel_mesh refuses
    # use_shared_nodes=True for eccentric stiffeners; that refused flag is
    # not a positive route.  This is a coincident beam on the shell edge, no
    # eccentric capability.)
    panel = generate_simple_panel_mesh(
        1.0, 1.0, 0.01, num_divisions_x=1, num_divisions_y=1
    )
    shell_element = next(iter(panel.mesh.elements.values()))
    edge_node_ids = [int(shell_element.node_ids[0]), int(shell_element.node_ids[1])]
    beam = BeamElement(
        100,
        edge_node_ids,
        material_name="steel",
        cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
    )
    panel.add_element(100, beam)
    assert len(panel.mesh.elements) == 2

    edge_dofs = []
    for node_id in edge_node_ids:
        edge_dofs.extend(int(dof) for dof in panel.mesh.get_node(node_id).dofs)
    assert beam.get_dof_mapping(panel.mesh) == edge_dofs
    assert edge_dofs == shell_element.get_dof_mapping(panel.mesh)[: len(edge_dofs)]

    report = audit_constraints(panel)
    assert report.status == "ok"
    assert report.feasible
    assert report.mpc_slave_dofs == 0

    T, u0, independent, info = _production_transform(panel)
    assert info["num_mpc_slave_dofs"] == 0
    assert T.shape[0] == int(panel.mesh.dof_manager.total_dofs)


# ---------------------------------------------------------------------------
# Wrench transfer and virtual work through the production diagnostics
# ---------------------------------------------------------------------------

def test_interpolated_wrench_transfer_and_virtual_work():
    r = np.array([0.12, -0.04, 0.18])
    coords = {1: (0.0, 0.0, 0.0), 2: (0.5, 0.0, 0.0), 3: (1.0, 0.0, 0.0)}
    weights = np.array([0.28, 0.84, -0.12])
    shell_ids = [1, 2, 3]
    # Physically matched record: the beam node sits at sum_i N_i x_i + r.
    attachment = sum(w * np.asarray(coords[n]) for n, w in zip(shell_ids, weights))
    beam_position = attachment + r
    coords[10] = tuple(beam_position)
    # Nonzero common reference origin for the moment transport below.
    o = np.array([0.7, -0.15, 0.35])
    model, _element = _interpolated_model(
        coords, 10, shell_ids, weights, r, fixed=shell_ids
    )

    force = np.array([1400.0, -320.0, 210.0])
    moment = np.array([18.0, -7.0, 31.0])
    load = LoadCase("interpolated_slave_load")
    load.add_nodal_load(10, np.concatenate([force, moment]))
    u_zero = np.zeros(int(model.mesh.dof_manager.total_dofs), dtype=float)

    # Residual convention is K u - F on the unreduced system; with u = 0 and
    # zero coupling stiffness the slave bucket carries minus the applied load
    # (internal residual sign), and the master-equivalent bucket carries the
    # transpose of the displacement map, again with the internal sign.
    diagnostics = compute_constraint_force_diagnostics(model, u_zero, load)
    slave_force = np.asarray(diagnostics["mpc_slave_forces"].get(10, np.zeros(6)), dtype=float)
    assert np.linalg.norm(slave_force[:3] + force) / FORCE_SCALE <= ATOL
    assert np.linalg.norm(slave_force[3:] + moment) / COUPLE_SCALE <= ATOL

    # Per-node distribution map (moments about each master node's own
    # origin): master i carries -N_i [F, M + r x F].
    transported = np.concatenate([force, moment + np.cross(r, force)])
    reported_wrenches = {}
    for node_id, weight in zip(shell_ids, weights):
        expected = -float(weight) * transported
        got = np.asarray(
            diagnostics["mpc_master_equivalent_forces"].get(node_id, np.zeros(6)), dtype=float
        )
        reported_wrenches[node_id] = got
        assert np.linalg.norm(got[:3] - expected[:3]) / FORCE_SCALE <= ATOL
        assert np.linalg.norm(got[3:] - expected[3:]) / COUPLE_SCALE <= ATOL

    # Resultant about the declared common origin o, from the ACTUAL reported
    # per-node wrenches: sum_i f_i and sum_i [m_i + (x_i - o) x f_i].  The
    # nodal couples alone are an algebraic sum about different node origins,
    # not a global moment.  The transported resultant must equal the negative
    # applied beam wrench about o: -[F, M + (x_b - o) x F].
    total_force = np.zeros(3, dtype=float)
    total_couple_about_o = np.zeros(3, dtype=float)
    for node_id in shell_ids:
        wrench = reported_wrenches[node_id]
        xi = np.asarray(coords[node_id], dtype=float)
        total_force = total_force + wrench[:3]
        total_couple_about_o = total_couple_about_o + wrench[3:] + np.cross(xi - o, wrench[:3])
    assert np.linalg.norm(total_force + force) / FORCE_SCALE <= ATOL
    expected_couple_about_o = moment + np.cross(beam_position - o, force)
    assert np.linalg.norm(total_couple_about_o + expected_couple_about_o) / COUPLE_SCALE <= ATOL

    # The direct support bucket must not double-count the MPC transfer.
    for node_id in shell_ids:
        support = np.asarray(
            diagnostics["support_reactions"].get(node_id, np.zeros(6)), dtype=float
        )
        assert np.linalg.norm(support[:3]) / FORCE_SCALE <= ATOL
        assert np.linalg.norm(support[3:]) / COUPLE_SCALE <= ATOL

    # Virtual work on a free-master model (identical node IDs and constraint
    # map, so the reported master-equivalent wrenches above apply): the
    # applied beam load contracted with the beam state must equal the
    # analytic transported wrench contracted with an independent virtual
    # master field, and the ACTUAL reported master-equivalent wrenches
    # contracted with the same field must give the negative of it (internal
    # sign).
    free_model, _free_element = _interpolated_model(coords, 10, shell_ids, weights, r)
    master_states = {
        1: (np.array([0.01, 0.0, -0.02]), np.array([1.0e-3, 0.0, 0.0])),
        2: (np.array([-0.03, 0.02, 0.01]), np.array([0.0, 2.0e-3, -1.0e-3])),
        3: (np.array([0.05, -0.04, 0.0]), np.array([-2.0e-3, 1.0e-3, 3.0e-3])),
    }
    T, u0, independent, _info = _production_transform(free_model)
    full = _full_from_master_states(free_model, master_states)
    u = reconstruct_full_solution(T, full[independent], u0)
    got_u, got_theta = _beam_state(free_model, 10, u)
    beam_work = float(force @ got_u + moment @ got_theta)
    analytic_work = 0.0
    reported_work = 0.0
    for node_id, weight in zip(shell_ids, weights):
        node_u, node_theta = master_states[node_id]
        node_wrench = float(weight) * transported
        analytic_work += float(node_wrench[:3] @ node_u + node_wrench[3:] @ node_theta)
        reported = reported_wrenches[node_id]
        reported_work += float(reported[:3] @ node_u + reported[3:] @ node_theta)
    assert abs(beam_work - analytic_work) / WORK_SCALE <= ATOL
    assert abs(reported_work + beam_work) / WORK_SCALE <= ATOL


# ---------------------------------------------------------------------------
# Fail-closed input admission through the production audit and assembly
# ---------------------------------------------------------------------------

def _bare_model():
    model = FEModel("malformed_admission")
    _steel(model)
    return model


def test_constructor_rejects_malformed_weights_and_offsets():
    with pytest.raises(ValueError, match="shape-weight"):
        InterpolatedBeamShellMPCElement(1, 10, [1, 2, 3], np.array([0.5, 0.5]), np.zeros(3))
    with pytest.raises(ValueError, match="shape-weight"):
        InterpolatedBeamShellMPCElement(1, 10, [1, 2, 3], np.array([[0.5, 0.5, 0.0]]), np.zeros(3))
    with pytest.raises(ValueError, match="non-finite"):
        InterpolatedBeamShellMPCElement(1, 10, [1, 2, 3], np.array([np.nan, 1.0, 1.0]), np.zeros(3))
    with pytest.raises(ValueError, match="partition of unity"):
        InterpolatedBeamShellMPCElement(1, 10, [1, 2, 3], np.array([0.9, 0.05, 0.04]), np.zeros(3))
    with pytest.raises(ValueError, match="eccentricity"):
        InterpolatedBeamShellMPCElement(1, 10, [1, 2, 3], np.array([1.0, 0.0, 0.0]), np.zeros(2))
    with pytest.raises(ValueError, match="non-finite"):
        InterpolatedBeamShellMPCElement(1, 10, [1, 2, 3], np.array([1.0, 0.0, 0.0]), np.array([0.0, np.inf, 0.0]))
    with pytest.raises(ValueError, match="eccentricity"):
        CoupledBeamShellElement(1, 10, 1, eccentricity=np.zeros(2))
    with pytest.raises(ValueError, match="non-finite"):
        CoupledBeamShellElement(1, 10, 1, eccentricity=np.array([0.0, np.nan, 0.0]))


def test_overflowing_finite_weights_reject_construction_and_emission():
    # Finite mixed-sign weights whose pairwise reduction overflows to NaN
    # must fail the partition-of-unity admission: NaN compares False against
    # the tolerance, so an explicit finite-sum check is required.  The
    # intentional overflow is confined to this test's errstate.
    overflow_weights = np.array([1.0e308, 1.0e308, -1.0e308, -1.0e308] * 2)
    assert np.all(np.isfinite(overflow_weights))
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.isnan(overflow_weights.sum())

    shell_ids = list(range(1, 9))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="partition of unity"):
            InterpolatedBeamShellMPCElement(1, 10, shell_ids, overflow_weights, np.zeros(3))

    # Emission after an aliased in-place mutation to the same overflowing
    # values must fail closed through the production audit as well.
    weights = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    element = InterpolatedBeamShellMPCElement(1, 10, shell_ids, weights, np.zeros(3))
    model = _bare_model()
    for node_id in shell_ids:
        model.add_node(node_id, 0.1 * node_id, 0.0, 0.0)
    model.add_node(10, 0.45, 0.0, 0.0)
    model.add_element(1, element)
    assert audit_constraints(model).status == "ok"
    with np.errstate(over="ignore", invalid="ignore"):
        weights[:] = overflow_weights
        report = audit_constraints(model)
        with pytest.raises(ValueError, match="partition of unity"):
            element.get_mpc_constraints(model.mesh)
    assert report.status == "invalid"
    assert any(issue.code == "CONSTRAINT001" for issue in report.issues)


def _model_with_coupling(element, node_count=5):
    model = _bare_model()
    for node_id in range(1, node_count + 1):
        model.add_node(node_id, float(node_id - 1), 0.0, 0.0)
    model.add_element(1, element)
    return model


def test_missing_referenced_nodes_fail_closed_through_audit():
    # Missing beam node.
    element = InterpolatedBeamShellMPCElement(
        1, 99, [1, 2, 3], np.array([0.28, 0.84, -0.12]), np.zeros(3)
    )
    model = _model_with_coupling(element)
    report = audit_constraints(model)
    assert report.status == "invalid"
    assert not report.feasible
    assert report.equations == ()
    assert any(issue.code == "CONSTRAINT001" for issue in report.issues)
    with pytest.raises(ValueError, match="Invalid constraint system"):
        require_valid_constraints(model)
    with pytest.raises(ValueError):
        build_constraint_transformation(
            sparse.eye(int(model.mesh.dof_manager.total_dofs), format="csr"),
            np.zeros(int(model.mesh.dof_manager.total_dofs)),
            model,
        )

    # Missing nonzero-weight shell master.
    element = InterpolatedBeamShellMPCElement(
        1, 4, [1, 2, 99], np.array([0.28, 0.84, -0.12]), np.zeros(3)
    )
    model = _model_with_coupling(element)
    report = audit_constraints(model)
    assert report.status == "invalid"
    assert any(issue.code == "CONSTRAINT001" for issue in report.issues)

    # Missing zero-weight shell master must also fail closed.
    element = InterpolatedBeamShellMPCElement(
        1, 4, [1, 2, 99], np.array([1.0, 0.0, 0.0]), np.zeros(3)
    )
    model = _model_with_coupling(element)
    report = audit_constraints(model)
    assert report.status == "invalid"
    assert any(issue.code == "CONSTRAINT001" for issue in report.issues)

    # One-to-one route: missing beam or shell node fails closed.
    model = _model_with_coupling(CoupledBeamShellElement(1, 99, 1, material_name="steel"))
    report = audit_constraints(model)
    assert report.status == "invalid"
    assert any(issue.code == "CONSTRAINT001" for issue in report.issues)
    model = _model_with_coupling(CoupledBeamShellElement(1, 4, 99, material_name="steel"))
    report = audit_constraints(model)
    assert report.status == "invalid"
    assert any(issue.code == "CONSTRAINT001" for issue in report.issues)


def test_post_construction_corruption_fails_closed_at_emission():
    weights = np.array([0.28, 0.84, -0.12])
    element = InterpolatedBeamShellMPCElement(1, 4, [1, 2, 3], weights, np.zeros(3))
    model = _model_with_coupling(element)
    assert audit_constraints(model).status == "ok"

    # Aliased array corrupted in place: partition of unity violated.
    weights[0] = 0.5
    report = audit_constraints(model)
    assert report.status == "invalid"
    assert any(issue.code == "CONSTRAINT001" for issue in report.issues)
    weights[0] = 0.28
    assert audit_constraints(model).status == "ok"

    # Attribute-level corruption: count mismatch.
    element.shape_weights = np.array([0.5, 0.5])
    report = audit_constraints(model)
    assert report.status == "invalid"
    assert any(issue.code == "CONSTRAINT001" for issue in report.issues)
    element.shape_weights = weights

    # Non-finite corruption.
    weights[1] = np.nan
    report = audit_constraints(model)
    assert report.status == "invalid"
    assert any(issue.code == "CONSTRAINT001" for issue in report.issues)
    weights[1] = 0.84
    assert audit_constraints(model).status == "ok"


# ---------------------------------------------------------------------------
# Aliasing, legitimate updates and refresh-versus-cold parity
# ---------------------------------------------------------------------------

def test_constructor_preserves_array_aliasing_and_copies_node_ids():
    weights = np.array([0.28, 0.84, -0.12])
    r = np.array([0.0, 0.0, 0.2])
    shell_ids = [1, 2, 3]
    element = InterpolatedBeamShellMPCElement(1, 4, shell_ids, weights, r)
    assert element.shape_weights is weights
    assert element.eccentricity is r
    shell_ids.append(4)
    assert element.shell_node_ids == [1, 2, 3]

    eccentricity = np.array([0.1, 0.0, 0.0])
    coupled = CoupledBeamShellElement(1, 4, 1, eccentricity=eccentricity)
    assert coupled.eccentricity is eccentricity


def test_mutable_weight_refresh_matches_cold_session_parity():
    coords = {1: (0.0, 0.0, 0.0), 2: (0.5, 0.0, 0.0), 3: (1.0, 0.0, 0.0), 10: (0.3, 0.0, 0.0)}
    r = np.array([0.0, 0.0, 0.1])
    weights = np.array([0.28, 0.84, -0.12])
    shell_ids = [1, 2, 3]

    # A real reused AnalysisSession on the bare zero-stiffness coupling
    # model: constraint_plan() needs no solve or factorization.
    model, _element = _interpolated_model(coords, 10, shell_ids, weights, r)
    session = AnalysisSession(model)
    plan_before = session.constraint_plan()
    assert session.constraint_plan() is plan_before
    structure_key_before = plan_before.structure_key

    # Legitimate model update through the aliased array (still a valid
    # partition of unity); the refreshed session plan must match a cold
    # session built with the new weights exactly.
    weights[:] = [0.375, 0.25, 0.375]
    plan_weights = session.constraint_plan()
    assert plan_weights is not plan_before
    assert plan_weights.structure_key != structure_key_before
    assert (plan_before.T != plan_weights.T).nnz > 0

    cold_model, _cold = _interpolated_model(coords, 10, shell_ids, weights, r)
    cold_session = AnalysisSession(cold_model)
    plan_cold = cold_session.constraint_plan()
    assert np.array_equal(plan_weights.independent_dofs, plan_cold.independent_dofs)
    assert (plan_weights.T != plan_cold.T).nnz == 0
    assert np.array_equal(plan_weights.u0, plan_cold.u0)

    # The stale pre-update plan is rejected through the existing session API.
    beam_dofs = np.asarray(model.mesh.get_node(10).dofs, dtype=np.intp)
    with pytest.raises(ValueError, match="stale or foreign ConstraintPlan"):
        session.output_selection_plan(beam_dofs, plan_before)

    master_states = {
        1: (np.array([0.01, 0.0, 0.0]), np.array([1.0e-3, 0.0, 0.0])),
        2: (np.array([0.0, 0.01, 0.0]), np.array([0.0, 1.0e-3, 0.0])),
        3: (np.array([0.0, 0.0, 0.01]), np.array([0.0, 0.0, 1.0e-3])),
    }
    full = _full_from_master_states(model, master_states)
    u_refreshed = reconstruct_full_solution(
        plan_weights.T, full[plan_weights.independent_dofs], plan_weights.u0
    )
    full_cold = _full_from_master_states(cold_model, master_states)
    u_cold = reconstruct_full_solution(
        plan_cold.T, full_cold[plan_cold.independent_dofs], plan_cold.u0
    )
    got_u_refreshed, _ = _beam_state(model, 10, u_refreshed)
    got_u_cold, _ = _beam_state(cold_model, 10, u_cold)
    assert np.linalg.norm(got_u_refreshed - got_u_cold) / TRANSLATION_SCALE <= ATOL

    # Legitimate aliased offset update, separately: same rebuild and
    # cold-session parity route.
    r[:] = [0.05, -0.02, 0.03]
    plan_offset = session.constraint_plan()
    assert plan_offset.structure_key != plan_weights.structure_key

    cold_model_offset, _cold_offset = _interpolated_model(coords, 10, shell_ids, weights, r)
    cold_session_offset = AnalysisSession(cold_model_offset)
    plan_cold_offset = cold_session_offset.constraint_plan()
    assert (plan_offset.T != plan_cold_offset.T).nnz == 0
    assert np.array_equal(plan_offset.u0, plan_cold_offset.u0)

    diagnostics = session.diagnostics()
    assert diagnostics["invalidation_reasons"]["constraint_structure"] == 2
    assert diagnostics["counters"]["constraint_builds"] == 3
    assert diagnostics["counters"]["constraint_hits"] == 2
    assert diagnostics["counters"]["constraint_plan_rejections"] == 1
