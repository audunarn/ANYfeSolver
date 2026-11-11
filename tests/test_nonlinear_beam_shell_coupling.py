"""Verification of Slice 2: Direct nonlinear master-slave beam-shell kinematic coupling.

Tests the exact finite-rotation master-slave kinematic elimination.  On the
active rotating-offset route the slave orientation and offset frame are
governed by the intrinsic weighted SO(3) mean R_m = mean(R_i, w_i):
    u_b     = sum_i w_i u_{s,i} + (R_m - I) r_0
    theta_b = theta_m = Log(R_m)
which is rigid-spin equivariant for distinct master rotations (the direct
single-master route and identical-rotation limit reproduce the legacy
averaged map exactly), with state-dependent Jacobian J(q) = du/dq, reduced
residual r_red(q) = J(q)^T (F_ext - F_int(u(q))), and symmetric geometric
MPC tangent K_mpc,geom(q, F_int - F_ext), using zero Lagrange multipliers.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import (
    BoundaryCondition,
    DisplacementControl,
    FEModel,
    LoadCase,
    create_shell_element,
    solve_static_nonlinear,
)
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.matrix_assembly import (
    assemble_load_vector,
    assemble_stiffness_matrix,
)
from anyfesolver.corotational import (
    _rotation_left_jacobian_inverse,
    _so3_weighted_mean,
    build_nonlinear_rotating_mpc_map,
    has_nonlinear_rotating_mpc,
    rotating_mpc_geometric_hessian,
    rotating_mpc_lever_arm_jacobian,
    rotation_matrix_from_vector,
    rotation_vector_from_matrix,
    spatial_moment_from_rotation_covector,
)
from anyfesolver.constraint_audit import constraint_residual_summary
from anyfesolver.elements import BeamElement, CoupledBeamShellElement
from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement
from anyfesolver.nonlinear_restart import model_configuration_fingerprint
from anyfesolver.nonlinear_static import _assemble_nonlinear_system


def _build_coupled_cantilever(
    route: str,
    *,
    nx: int = 2,
    ny: int = 1,
    length: float = 1.0,
    width: float = 0.2,
    thickness: float = 0.02,
    eccentricity: tuple[float, float, float] = (0.0, 0.0, 0.05),
) -> tuple[FEModel, dict]:
    """Build a cantilever Q4 shell strip with a coupled eccentric beam stiffener."""
    model = FEModel(f"nonlinear_coupled_{route}")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    r_vec = np.asarray(eccentricity, dtype=float)

    # Create shell grid nodes: x in [0, length], y in [-width/2, width/2]
    shell_node_grid: dict[tuple[int, int], int] = {}
    node_id = 1
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(-0.5 * width, 0.5 * width, ny + 1)
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            model.add_node(node_id, float(x), float(y), 0.0)
            shell_node_grid[(ix, iy)] = node_id
            node_id += 1

    elem_id = 1
    for ix in range(nx):
        for iy in range(ny):
            n1 = shell_node_grid[(ix, iy)]
            n2 = shell_node_grid[(ix + 1, iy)]
            n3 = shell_node_grid[(ix + 1, iy + 1)]
            n4 = shell_node_grid[(ix, iy + 1)]
            model.add_element(
                elem_id,
                create_shell_element(elem_id, [n1, n2, n3, n4], "steel", thickness=thickness),
            )
            elem_id += 1

    beam_nodes: list[int] = []
    couplings: list[dict] = []
    beam_base_id = 100
    mpc_base_id = 200

    if route == "eccentric":
        # Attach beam along y = -width/2 (iy = 0) one-to-one with shell nodes
        for ix, x in enumerate(xs):
            master_id = shell_node_grid[(ix, 0)]
            master_node = model.mesh.get_node(master_id)
            b_id = beam_base_id + ix
            b_xyz = np.array([master_node.x, master_node.y, master_node.z], dtype=float) + r_vec
            model.add_node(b_id, float(b_xyz[0]), float(b_xyz[1]), float(b_xyz[2]))
            beam_nodes.append(b_id)
            mpc_id = mpc_base_id + ix
            model.add_element(
                mpc_id,
                CoupledBeamShellElement(
                    mpc_id,
                    beam_node_id=b_id,
                    shell_node_id=master_id,
                    material_name="steel",
                    eccentricity=r_vec,
                ),
            )
            couplings.append(
                {
                    "slave": b_id,
                    "masters": [master_id],
                    "weights": np.array([1.0], dtype=float),
                    "r": r_vec.copy(),
                }
            )
    elif route == "interpolated":
        # Attach beam along the centerline y = 0 interpolated between iy=0 and iy=ny
        weights = np.array([0.6, 0.4], dtype=float)
        for ix, x in enumerate(xs):
            m1 = shell_node_grid[(ix, 0)]
            m2 = shell_node_grid[(ix, ny)]
            n1_obj = model.mesh.get_node(m1)
            n2_obj = model.mesh.get_node(m2)
            attach_xyz = weights[0] * np.array([n1_obj.x, n1_obj.y, n1_obj.z]) + weights[1] * np.array(
                [n2_obj.x, n2_obj.y, n2_obj.z]
            )
            b_id = beam_base_id + ix
            b_xyz = attach_xyz + r_vec
            model.add_node(b_id, float(b_xyz[0]), float(b_xyz[1]), float(b_xyz[2]))
            beam_nodes.append(b_id)
            mpc_id = mpc_base_id + ix
            model.add_element(
                mpc_id,
                InterpolatedBeamShellMPCElement(
                    mpc_id,
                    b_id,
                    [m1, m2],
                    weights,
                    eccentricity=r_vec,
                ),
            )
            couplings.append(
                {
                    "slave": b_id,
                    "masters": [m1, m2],
                    "weights": weights.copy(),
                    "r": r_vec.copy(),
                }
            )
    else:
        raise ValueError(f"Unknown route {route!r}")

    cross_section = {
        "area": 0.01 * 0.04,
        "Iy": 0.01 * (0.04**3) / 12.0,
        "Iz": 0.04 * (0.01**3) / 12.0,
        "J": 1.0e-8,
    }
    for ix in range(nx):
        b_elem_id = 300 + ix
        model.add_element(
            b_elem_id,
            BeamElement(
                b_elem_id,
                [beam_nodes[ix], beam_nodes[ix + 1]],
                material_name="steel",
                cross_section=cross_section,
            ),
        )

    root_shell_nodes = [shell_node_grid[(0, iy)] for iy in range(ny + 1)]
    tip_shell_nodes = [shell_node_grid[(nx, iy)] for iy in range(ny + 1)]
    model.add_boundary_condition(
        BoundaryCondition(
            "root",
            root_shell_nodes,
            {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
        )
    )
    model.apply_boundary_conditions()
    meta = {
        "route": route,
        "beam_nodes": beam_nodes,
        "root_shell_nodes": root_shell_nodes,
        "tip_shell_nodes": tip_shell_nodes,
        "couplings": couplings,
        "eccentricity": r_vec,
    }
    return model, meta


@pytest.mark.parametrize("route", ("eccentric", "interpolated"))
def test_mpc_descriptor_preserves_model_fingerprint(route: str) -> None:
    """Calling get_nonlinear_mpc_descriptor must never mutate element.__dict__ or model fingerprint."""
    model, _meta = _build_coupled_cantilever(route)
    K0, _ = assemble_stiffness_matrix(model)
    zero = np.zeros(K0.shape[0], dtype=float)
    _, _, T, _, _, _ = build_constraint_transformation(K0, zero, model)

    fp_before = model_configuration_fingerprint(model)
    dicts_before = {
        eid: dict(elem.__dict__) for eid, elem in model.mesh.elements.items()
    }

    assert has_nonlinear_rotating_mpc(model) is True
    mpc_map = build_nonlinear_rotating_mpc_map(model, T)
    assert mpc_map is not None
    assert len(mpc_map.entries) == 3

    fp_after = model_configuration_fingerprint(model)
    assert fp_after == fp_before
    for eid, elem in model.mesh.elements.items():
        assert set(elem.__dict__.keys()) == set(dicts_before[eid].keys())


@pytest.mark.parametrize("route", ("eccentric", "interpolated"))
def test_rigid_body_motion_zero_strain_up_to_45_degrees(route: str) -> None:
    """Coupled beam-shell assembly has zero strain and zero residual under 45-deg 3D rigid motion."""
    model, meta = _build_coupled_cantilever(
        route,
        eccentricity=(0.03, -0.02, 0.08),
    )
    # Remove root clamp so all shell nodes can undergo a global rigid-body motion
    model.boundary_conditions.clear()
    model.apply_boundary_conditions()

    K0, _ = assemble_stiffness_matrix(model)
    zero = np.zeros(K0.shape[0], dtype=float)
    _, _, T, u0, independent_dofs, constraint_info = build_constraint_transformation(
        K0, zero, model
    )
    assert constraint_info["method"] == "transformation"
    mpc_map = build_nonlinear_rotating_mpc_map(model, T)
    assert mpc_map is not None

    # 45-degree rotation about an oblique 3D axis + finite translation
    axis = np.array([1.0, -2.0, 1.5], dtype=float)
    axis /= np.linalg.norm(axis)
    omega = np.deg2rad(45.0) * axis
    R0 = rotation_matrix_from_vector(omega)
    c0 = np.array([0.4, -0.25, 0.7], dtype=float)
    pivot = np.array([0.2, 0.05, -0.1], dtype=float)

    # Prescribe rigid-body motion on all shell nodes
    u_shell_prescribed = np.zeros(K0.shape[0], dtype=float)
    slave_nodes = {c["slave"] for c in meta["couplings"]}
    for nid, node in model.mesh.nodes.items():
        if nid in slave_nodes:
            continue
        X = np.array([node.x, node.y, node.z], dtype=float)
        dofs = model.mesh.dof_manager.get_node_dofs(nid)
        u_shell_prescribed[dofs[:3]] = c0 + (R0 - np.eye(3)) @ (X - pivot)
        u_shell_prescribed[dofs[3:6]] = omega

    q = u_shell_prescribed[independent_dofs]
    u_lin = np.asarray(T @ q + u0, dtype=float).reshape(-1)
    u_nl = mpc_map.apply_kinematics(u_lin)

    # Verify exact rigid-body position of every coupled beam node
    for coupling in meta["couplings"]:
        slave_id = coupling["slave"]
        slave_node = model.mesh.get_node(slave_id)
        X_b = np.array([slave_node.x, slave_node.y, slave_node.z], dtype=float)
        expected_u_b = c0 + (R0 - np.eye(3)) @ (X_b - pivot)
        slave_dofs = model.mesh.dof_manager.get_node_dofs(slave_id)
        np.testing.assert_allclose(u_nl[slave_dofs[:3]], expected_u_b, atol=1.0e-14)
        np.testing.assert_allclose(u_nl[slave_dofs[3:6]], omega, atol=1.0e-14)

    # Evaluate corotational internal force at u_nl
    F_int_nl, _K_nl, _states = _assemble_nonlinear_system(
        model,
        u_nl,
        {},
        5,
        tangent=True,
        kinematics="corotational",
        corotational_tangent="work_conjugate",
    )
    r_red_nl = mpc_map.reduce_force(u_nl, F_int_nl)
    k_scale = float(np.max(np.abs(K0.diagonal())))
    assert float(np.max(np.abs(F_int_nl))) < 1.0e-15 * k_scale
    assert float(np.max(np.abs(r_red_nl))) < 1.0e-15 * k_scale

    # Contrast: the linear MPC displacement u_lin misses the true rigid-body beam position
    # by ||(skew(omega) - (R0 - I)) r_0|| > 15 mm at 45 deg, causing > 1e5 N of spurious
    # beam force when the tip beam node is at its exact rigid-body position.
    tip_slave_dofs = model.mesh.dof_manager.get_node_dofs(meta["couplings"][-1]["slave"])
    tip_slave_node = model.mesh.get_node(meta["couplings"][-1]["slave"])
    tip_X_b = np.array([tip_slave_node.x, tip_slave_node.y, tip_slave_node.z], dtype=float)
    tip_expected_u_b = c0 + (R0 - np.eye(3)) @ (tip_X_b - pivot)
    assert float(np.linalg.norm(u_lin[tip_slave_dofs[:3]] - tip_expected_u_b)) > 0.015

    u_lin_pinned_tip = u_lin.copy()
    u_lin_pinned_tip[tip_slave_dofs[:3]] = tip_expected_u_b
    F_int_lin, _, _ = _assemble_nonlinear_system(
        model,
        u_lin_pinned_tip,
        {},
        5,
        tangent=False,
        kinematics="corotational",
        corotational_tangent="work_conjugate",
    )
    assert float(np.max(np.abs(F_int_lin))) > 1.0e5


@pytest.mark.parametrize("route", ("eccentric", "interpolated"))
def test_diffeomorphism_and_constraint_residual_postcheck(route: str) -> None:
    """to_linear_displacement is the exact inverse of apply_kinematics and preserves MPC residual checks."""
    model, meta = _build_coupled_cantilever(
        route,
        eccentricity=(0.02, 0.03, -0.06),
    )
    K0, _ = assemble_stiffness_matrix(model)
    zero = np.zeros(K0.shape[0], dtype=float)
    _, _, T, u0, _, _ = build_constraint_transformation(K0, zero, model)
    mpc_map = build_nonlinear_rotating_mpc_map(model, T)
    assert mpc_map is not None

    rng = np.random.default_rng(20261017)
    q = rng.uniform(-0.35, 0.35, int(T.shape[1]))
    u_lin = np.asarray(T @ q + u0, dtype=float).reshape(-1)
    u_nl = mpc_map.apply_kinematics(u_lin)
    u_lin_recovered = mpc_map.to_linear_displacement(u_nl)
    np.testing.assert_allclose(u_lin_recovered, u_lin, rtol=1.0e-14, atol=1.0e-15)
    np.testing.assert_allclose(
        mpc_map.apply_kinematics(u_lin_recovered), u_nl, rtol=1.0e-14, atol=1.0e-15
    )

    # Compatible nonlinear state has zero MPC residual when mapped via to_linear_displacement
    summary_ok = constraint_residual_summary(model, u_lin_recovered)
    assert summary_ok["status"] == "passed"
    assert summary_ok["max_absolute_residual"] < 1.0e-14
    assert summary_ok["max_relative_residual"] < 1.0e-14

    # Incompatible slave perturbation is detected with its exact magnitude
    u_bad = u_nl.copy()
    slave_dofs = model.mesh.dof_manager.get_node_dofs(meta["beam_nodes"][-1])
    u_bad[slave_dofs[0]] += 1.25e-3
    summary_bad = constraint_residual_summary(
        model, mpc_map.to_linear_displacement(u_bad)
    )
    assert summary_bad["status"] == "failed"
    assert summary_bad["max_absolute_residual"] == pytest.approx(1.25e-3, rel=1.0e-12)


def test_lever_arm_jacobian_and_geometric_hessian_match_finite_differences() -> None:
    """Verify A(theta_bar, r_0) and H_b(theta_bar, r_0, f_b) against central finite differences."""
    rng = np.random.default_rng(90210)
    for theta_scale in (0.0, 1.0e-7, 1.0e-4, 0.25, 0.85):
        theta_bar = theta_scale * rng.standard_normal(3)
        r0 = np.array([0.04, -0.025, 0.09], dtype=float)
        f_b = np.array([120.0, -350.0, 480.0], dtype=float)

        A = rotating_mpc_lever_arm_jacobian(theta_bar, r0)
        H = rotating_mpc_geometric_hessian(theta_bar, r0, f_b)
        np.testing.assert_allclose(H, H.T, atol=1.0e-14)

        eps = 1.0e-7
        A_fd = np.zeros((3, 3), dtype=float)
        H_fd = np.zeros((3, 3), dtype=float)
        for j in range(3):
            e_j = np.zeros(3, dtype=float)
            e_j[j] = eps
            tp = theta_bar + e_j
            tm = theta_bar - e_j
            up = (rotation_matrix_from_vector(tp) - np.eye(3)) @ r0
            um = (rotation_matrix_from_vector(tm) - np.eye(3)) @ r0
            A_fd[:, j] = (up - um) / (2.0 * eps)

            gp = rotating_mpc_lever_arm_jacobian(tp, r0).T @ f_b
            gm = rotating_mpc_lever_arm_jacobian(tm, r0).T @ f_b
            H_fd[:, j] = (gp - gm) / (2.0 * eps)

        np.testing.assert_allclose(A, A_fd, rtol=1.0e-8, atol=1.0e-10)
        np.testing.assert_allclose(H, H_fd, rtol=1.0e-7, atol=1.0e-8)


@pytest.mark.parametrize("route", ("eccentric", "interpolated"))
def test_reduced_coupled_tangent_matches_central_finite_differences(route: str) -> None:
    """Reduced coupled tangent J^T K_T J + K_mpc,geom is symmetric and matches FD to < 1e-6."""
    model, meta = _build_coupled_cantilever(
        route,
        nx=2,
        ny=1,
        eccentricity=(0.03, -0.015, 0.06),
    )
    # Apply nonzero external forces and moments directly to the eccentric beam nodes
    load = LoadCase("eccentric_beam_load")
    for idx, b_id in enumerate(meta["beam_nodes"][1:], start=1):
        load.add_nodal_load(
            b_id,
            forces=(150.0 * idx, -220.0 * idx, 400.0 * idx),
            moments=(12.0 * idx, -18.0 * idx, 9.0 * idx),
        )
    F_ext, _ = assemble_load_vector(model, load)

    K0, _ = assemble_stiffness_matrix(model)
    _, _, T, u0, _, _ = build_constraint_transformation(K0, F_ext, model)
    mpc_map = build_nonlinear_rotating_mpc_map(model, T)
    assert mpc_map is not None

    rng = np.random.default_rng(1337 if route == "eccentric" else 7331)
    n_red = int(T.shape[1])
    # Finite deformation state with ~25-35 deg rotations and finite translations
    q = np.zeros(n_red, dtype=float)
    for col in range(n_red):
        dof = int(T.indices[T.indptr[col]])
        if dof % 6 < 3:
            q[col] = rng.uniform(-0.04, 0.04)
        else:
            q[col] = rng.uniform(-0.45, 0.45)

    def full_u(q_vec: np.ndarray) -> np.ndarray:
        return mpc_map.apply_kinematics(np.asarray(T @ q_vec + u0, dtype=float).reshape(-1))

    u = full_u(q)
    J = mpc_map.jacobian(u).toarray()

    # Check J(q) = du/dq against central finite differences.  The step is
    # large enough that the intrinsic SO(3) mean's floating-point fixed-point
    # noise (a few ulp of theta_m) stays far below the comparison tolerance,
    # and small enough that truncation remains negligible at rtol 1e-8.
    eps_j = 2.0e-6
    J_fd = np.zeros_like(J)
    for j in range(n_red):
        dq = np.zeros(n_red, dtype=float)
        dq[j] = eps_j
        J_fd[:, j] = (full_u(q + dq) - full_u(q - dq)) / (2.0 * eps_j)
    np.testing.assert_allclose(J, J_fd, rtol=1.0e-8, atol=1.0e-10)

    # Evaluate internal force and consistent work-conjugate tangent
    F_int, K_T, _ = _assemble_nonlinear_system(
        model,
        u,
        {},
        5,
        tangent=True,
        kinematics="corotational",
        corotational_tangent="work_conjugate",
    )
    K_red = mpc_map.reduce_tangent(u, K_T, F_int - F_ext).toarray()

    # Symmetry check
    sym_err = np.linalg.norm(K_red - K_red.T, ord="fro") / max(
        np.linalg.norm(K_red, ord="fro"), 1.0
    )
    assert sym_err < 1.0e-12

    # Central finite difference of net reduced residual r_net(q) = J(q)^T (F_int(u(q)) - F_ext)
    def net_reduced_force(q_vec: np.ndarray) -> np.ndarray:
        u_q = full_u(q_vec)
        f_q, _, _ = _assemble_nonlinear_system(
            model,
            u_q,
            {},
            5,
            tangent=False,
            kinematics="corotational",
            corotational_tangent="work_conjugate",
        )
        return mpc_map.reduce_force(u_q, f_q - F_ext)

    eps = 1.0e-6
    K_fd = np.zeros_like(K_red)
    for j in range(n_red):
        dq = np.zeros(n_red, dtype=float)
        dq[j] = eps
        K_fd[:, j] = (net_reduced_force(q + dq) - net_reduced_force(q - dq)) / (2.0 * eps)

    rel_err = np.linalg.norm(K_red - K_fd, ord="fro") / np.linalg.norm(K_fd, ord="fro")
    assert rel_err < 1.0e-6

    # Confirm K_mpc,geom is non-vacuous: omitting it leaves a large O(||f_b|| ||r_0||) error
    K_without_geom = (J.T @ K_T.toarray() @ J)
    err_without_geom = np.linalg.norm(K_without_geom - K_fd, ord="fro")
    assert err_without_geom > 5.0e2 * np.linalg.norm(K_red - K_fd, ord="fro")


@pytest.mark.parametrize("route", ("eccentric", "interpolated"))
@pytest.mark.parametrize("control_mode", ("force", "displacement"))
def test_cantilever_stiffened_shell_finite_rotation_solve_and_wrench_balance(
    route: str,
    control_mode: str,
) -> None:
    """Solve coupled stiffened shell past 30 deg rotation and verify wrench & reaction balance."""
    model, meta = _build_coupled_cantilever(
        route,
        nx=4,
        ny=1,
        length=1.0,
        width=0.2,
        thickness=0.02,
        eccentricity=(0.0, 0.0, 0.05),
    )
    tip_beam_node = meta["beam_nodes"][-1]
    load = LoadCase("tip_bending_and_axial")
    # Apply transverse + axial force on the eccentric tip beam node so lever-arm
    # moment (R(theta_s) r_0) x f_b is strongly active at finite rotation (> 30 deg).
    load.add_nodal_load(
        tip_beam_node,
        forces=(-4.0e4, 1.5e4, -3.2e5),
        moments=(0.0, 2.0e4, 0.0),
    )

    if control_mode == "force":
        result = solve_static_nonlinear(
            model,
            load,
            num_steps=8,
            max_iterations=25,
            tolerance=1.0e-9,
            kinematics="corotational",
            corotational_tangent="auto",
        )
    else:
        ctrl = DisplacementControl(
            node_id=meta["tip_shell_nodes"][0],
            dof="uz",
            target_displacement=-0.45,
        )
        result = solve_static_nonlinear(
            model,
            load,
            control="displacement",
            displacement_control=ctrl,
            num_steps=8,
            max_iterations=25,
            tolerance=1.0e-9,
            kinematics="corotational",
            corotational_tangent="auto",
        )

    assert result.converged
    assert result.status == "completed"
    assert result.info["corotational_tangent"] == "work_conjugate"
    assert result.info["constraint_info"]["method"] == "transformation"
    assert result.info["constraint_postcheck"]["status"] == "passed"
    assert result.info["constraint_postcheck"]["max_absolute_residual"] <= 1.0e-12
    assert result.info["constraint_postcheck"]["max_relative_residual"] <= 1.0e-12

    u = np.asarray(result.displacements, dtype=float)
    tip_beam_dofs = model.mesh.dof_manager.get_node_dofs(tip_beam_node)
    tip_rot_norm = float(np.linalg.norm(u[tip_beam_dofs[3:6]]))
    # Verify finite rotation exceeds 30 degrees (0.5236 rad)
    assert tip_rot_norm > np.deg2rad(30.0)

    # Verify exact nonlinear rotating-arm kinematic relation at all coupled nodes
    K0_kin, _ = assemble_stiffness_matrix(model)
    _zero = np.zeros(K0_kin.shape[0], dtype=float)
    _, _, T_kin, _, _, _ = build_constraint_transformation(K0_kin, _zero, model)
    kin_map = build_nonlinear_rotating_mpc_map(model, T_kin)
    assert kin_map is not None
    for entry in kin_map.entries:
        slave_u_dofs = list(entry.slave_u_dofs)
        slave_theta_dofs = list(entry.slave_theta_dofs)
        theta_m, _G = entry.mean_state(u)
        u_s_bar = entry.master_translation(u)
        expected_u_b = u_s_bar + (rotation_matrix_from_vector(theta_m) - np.eye(3)) @ entry.eccentricity
        np.testing.assert_allclose(u[slave_u_dofs], expected_u_b, rtol=1.0e-12, atol=1.0e-14)
        np.testing.assert_allclose(u[slave_theta_dofs], theta_m, rtol=1.0e-12, atol=1.0e-14)

    # Verify spatial wrench transfer identity and global support reaction balance
    K0, _ = assemble_stiffness_matrix(model)
    F_prop, _ = assemble_load_vector(model, load)
    F_ext = float(result.load_factor) * F_prop
    _, _, T, _, _, _ = build_constraint_transformation(K0, F_prop, model)
    mpc_map = build_nonlinear_rotating_mpc_map(model, T)
    assert mpc_map is not None

    F_int, _, _ = _assemble_nonlinear_system(
        model,
        u,
        result.element_states,
        5,
        tangent=False,
        kinematics="corotational",
        corotational_tangent="work_conjugate",
    )
    out_of_balance = F_int - F_ext
    reduced_residual = mpc_map.reduce_force(u, out_of_balance)
    ref_norm = max(float(np.linalg.norm(mpc_map.reduce_force(u, F_ext))), 1.0)
    assert float(np.linalg.norm(reduced_residual)) <= 1.0e-8 * ref_norm

    # Verify per-slave spatial wrench equivalence:
    # Since t_i = J_l(theta_i)^{-T} m_i and J_r(theta) J_l(theta)^T = R(theta),
    # J_l(theta_bar)^{-T} (m_b + A(theta_bar, r_0)^T f_b)
    #   = t_b - J_l(theta_bar)^{-T} J_r(theta_bar)^T (-hat(r_0)) R(theta_bar)^T f_b
    #   = t_b + R(theta_bar) hat(r_0) R(theta_bar)^T f_b
    #   = t_b + (R(theta_bar) r_0) x f_b !
    for entry in mpc_map.entries:
        f_b = out_of_balance[list(entry.slave_u_dofs)]
        m_b = out_of_balance[list(entry.slave_theta_dofs)]
        theta_m, _G = entry.mean_state(u)
        A = rotating_mpc_lever_arm_jacobian(theta_m, entry.eccentricity)
        spatial_torque_from_pullback = spatial_moment_from_rotation_covector(
            theta_m, m_b + A.T @ f_b
        )
        t_b = spatial_moment_from_rotation_covector(theta_m, m_b)
        r_spatial = rotation_matrix_from_vector(theta_m) @ entry.eccentricity
        spatial_torque_expected = t_b + np.cross(r_spatial, f_b)
        np.testing.assert_allclose(
            spatial_torque_from_pullback,
            spatial_torque_expected,
            rtol=1.0e-11,
            atol=1.0e-11,
        )

    # Verify global support reaction force equilibrium (including root slave beam transfer)
    root_reaction = np.asarray(result.steps[-1].support_reactions["root"][:3], dtype=float)
    applied_total_force = np.array([-4.0e4, 1.5e4, -3.2e5], dtype=float) * float(
        result.load_factor
    )
    np.testing.assert_allclose(
        root_reaction + applied_total_force,
        np.zeros(3, dtype=float),
        atol=1.0e-7 * np.linalg.norm(applied_total_force),
    )


@pytest.mark.parametrize("route", ("eccentric", "interpolated"))
def test_coupled_corotational_split_restart_matches_uninterrupted(route: str) -> None:
    """Split restart at finite rotation reproduces the uninterrupted solve."""
    model, meta = _build_coupled_cantilever(
        route,
        nx=2,
        ny=1,
        eccentricity=(0.0, 0.0, 0.05),
    )
    half_load = LoadCase("half_tip_load")
    half_load.add_nodal_load(
        meta["beam_nodes"][-1],
        forces=(-1.0e4, 0.5e4, -1.25e5),
    )
    common = dict(
        num_steps=2,
        max_iterations=25,
        tolerance=1.0e-10,
        kinematics="corotational",
        corotational_tangent="auto",
    )

    half = solve_static_nonlinear(
        model,
        half_load,
        max_load_factor=1.0,
        **common,
    )
    assert half.converged
    resumed = solve_static_nonlinear(
        model,
        half_load,
        constant_load_case=half_load,
        max_load_factor=1.0,
        initial_displacements=half.displacements,
        initial_element_states=half.element_states,
        equilibrate_initial_state=False,
        **common,
    )
    assert resumed.converged

    fresh_model, fresh_meta = _build_coupled_cantilever(
        route,
        nx=2,
        ny=1,
        eccentricity=(0.0, 0.0, 0.05),
    )
    fresh_load = LoadCase("full_tip_load")
    fresh_load.add_nodal_load(
        fresh_meta["beam_nodes"][-1],
        forces=(-2.0e4, 1.0e4, -2.5e5),
    )
    uninterrupted = solve_static_nonlinear(
        fresh_model,
        fresh_load,
        max_load_factor=1.0,
        num_steps=4,
        max_iterations=25,
        tolerance=1.0e-10,
        kinematics="corotational",
        corotational_tangent="auto",
    )
    assert uninterrupted.converged
    np.testing.assert_allclose(
        resumed.displacements,
        uninterrupted.displacements,
        rtol=1.0e-8,
        atol=1.0e-10,
    )


# ---------------------------------------------------------------------------
# Rigid-spin objectivity of the intrinsic SO(3) mean (adopted contract)
# ---------------------------------------------------------------------------


def test_so3_mean_counterexample_is_rigid_spin_equivariant() -> None:
    """An admitted non-antipodal counterexample to rotation-vector averaging."""
    a = 0.4 * np.pi
    thetas = [np.array([a, 0.0, 0.0]), np.array([-a, 0.0, 0.0])]
    weights = [0.5, 0.5]
    theta_m, G = _so3_weighted_mean(thetas, weights, "test")
    # Commuting rotations: the mean is the exact weighted average.
    np.testing.assert_allclose(theta_m, np.zeros(3), atol=1.0e-14)

    # Differential rigid-spin identity: sum_i G_i J_l(theta_i)^-1 = J_l(theta_m)^-1.
    lhs = np.zeros((3, 3))
    for G_i, t_i in zip(G, thetas):
        lhs += G_i @ _rotation_left_jacobian_inverse(t_i)
    np.testing.assert_allclose(lhs, _rotation_left_jacobian_inverse(theta_m), atol=1.0e-12)

    # Superpose Q = Ry(a) on both physical masters.
    Q = rotation_matrix_from_vector(np.array([0.0, a, 0.0]))
    spun = [rotation_vector_from_matrix(Q @ rotation_matrix_from_vector(t)) for t in thetas]
    theta_m_spun, G_spun = _so3_weighted_mean(spun, weights, "test")
    np.testing.assert_allclose(
        rotation_matrix_from_vector(theta_m_spun), Q, rtol=0.0, atol=1.0e-12
    )
    lhs_spun = np.zeros((3, 3))
    for G_i, t_i in zip(G_spun, spun):
        lhs_spun += G_i @ _rotation_left_jacobian_inverse(t_i)
    np.testing.assert_allclose(
        lhs_spun, _rotation_left_jacobian_inverse(theta_m_spun), atol=1.0e-12
    )

    # Discriminator: the arithmetic average of the spun principal rotation
    # vectors is NOT equivariant.
    lin_spun = weights[0] * spun[0] + weights[1] * spun[1]
    assert np.linalg.norm(lin_spun - np.array([0.0, a, 0.0])) > 1.0e-2


def test_so3_mean_direct_identical_limits_and_refusals() -> None:
    """Direct, identical-rotation and first-order limits are exact; unsupported cases refuse."""
    t = np.array([0.3, -0.2, 0.15])
    theta_m, G = _so3_weighted_mean([t], [1.0], "test")
    np.testing.assert_array_equal(theta_m, t)
    np.testing.assert_allclose(G[0], np.eye(3), atol=0.0)

    theta_m2, G2 = _so3_weighted_mean([t, t, t], [0.2, 0.5, 0.3], "test")
    np.testing.assert_array_equal(theta_m2, t)
    for G_i, w in zip(G2, (0.2, 0.5, 0.3)):
        np.testing.assert_allclose(G_i, w * np.eye(3), atol=1.0e-15)

    # First-order weighted linear relation for small rotations.
    small = [np.array([1.0e-3, -2.0e-3, 5.0e-4]), np.array([-1.5e-3, 1.0e-3, -3.0e-4])]
    theta_small, _G_small = _so3_weighted_mean(small, [0.6, 0.4], "test")
    lin_small = 0.6 * small[0] + 0.4 * small[1]
    assert np.linalg.norm(theta_small - lin_small) <= 1.0e-7

    with pytest.raises(ValueError, match="nonnegative"):
        _so3_weighted_mean([t, -0.5 * t], [1.5, -0.5], "test")
    with pytest.raises(ValueError, match="partition of unity"):
        _so3_weighted_mean([t, t], [0.5, 0.4], "test")


def _interp_map_with_distinct_master_rotations(eccentricity=(0.02, 0.03, -0.06)):
    """Interpolated fixture map plus a CONSISTENT linear state with distinct master rotations.

    The master rotations are replaced by distinct finite values and the slave
    linear MPC entries are recomputed from them, so ``u_lin`` satisfies the
    linear constraint exactly and the map's corrections apply to a consistent
    state.  The returned ``u_lin`` is deliberately NOT in the image of ``T``
    (the master rotations were replaced), so finite-difference tests of
    ``du/dq`` must construct their own ``T @ q + u0`` state.
    """
    model, meta = _build_coupled_cantilever("interpolated", eccentricity=eccentricity)
    K0, _ = assemble_stiffness_matrix(model)
    zero = np.zeros(K0.shape[0], dtype=float)
    _, _, T, u0, _, _ = build_constraint_transformation(K0, zero, model)
    mpc_map = build_nonlinear_rotating_mpc_map(model, T)
    assert mpc_map is not None
    rng = np.random.default_rng(20261018)
    q = rng.uniform(-0.05, 0.05, int(T.shape[1]))
    u_lin = np.asarray(T @ q + u0, dtype=float).reshape(-1)
    for entry in mpc_map.entries:
        for k, (_m_u, m_theta, _w) in enumerate(entry.masters):
            u_lin[m_theta] = np.roll(np.array([0.35, -0.27, 0.19]), k) * (1.0 + 0.13 * k)
        theta_lin = entry.master_rotation(u_lin)
        u_lin[entry.slave_theta_dofs] = theta_lin
        u_lin[entry.slave_u_dofs] = (
            entry.master_translation(u_lin) + np.cross(theta_lin, entry.eccentricity)
        )
    return model, meta, T, u0, q, mpc_map, u_lin


def test_interp_map_finite_rigid_spin_covariance() -> None:
    """Weighted attachment point and slave orientation transform under a common rigid rotation."""
    model, meta, T, u0, q, mpc_map, u_lin = _interp_map_with_distinct_master_rotations()
    u_nl = mpc_map.apply_kinematics(u_lin)

    psi = np.array([0.23, -0.31, 0.42])
    Q = rotation_matrix_from_vector(psi)
    u_lin_spun = u_lin.copy()
    dof_manager = model.mesh.dof_manager
    for coupling in meta["couplings"]:
        master_states = []
        for m_id, w in zip(coupling["masters"], coupling["weights"]):
            m_dofs = dof_manager.get_node_dofs(m_id)
            m_node = model.mesh.get_node(m_id)
            X = np.array([m_node.x, m_node.y, m_node.z], dtype=float)
            u_lin_spun[m_dofs[:3]] = (Q - np.eye(3)) @ X + Q @ u_lin[m_dofs[:3]]
            u_lin_spun[m_dofs[3:6]] = rotation_vector_from_matrix(
                Q @ rotation_matrix_from_vector(u_lin[m_dofs[3:6]])
            )
            master_states.append(
                (np.array(u_lin_spun[m_dofs[:3]]), np.array(u_lin_spun[m_dofs[3:6]]), float(w))
            )
        s_dofs = dof_manager.get_node_dofs(coupling["slave"])
        u_bar = sum(w * mu for mu, _mt, w in master_states)
        theta_lin = sum(w * mt for _mu, mt, w in master_states)
        u_lin_spun[s_dofs[:3]] = u_bar + np.cross(theta_lin, coupling["r"])
        u_lin_spun[s_dofs[3:6]] = theta_lin

    u_nl_spun = mpc_map.apply_kinematics(u_lin_spun)
    for coupling in meta["couplings"]:
        s_dofs = dof_manager.get_node_dofs(coupling["slave"])
        s_node = model.mesh.get_node(coupling["slave"])
        X_b = np.array([s_node.x, s_node.y, s_node.z], dtype=float)
        x_b = X_b + u_nl[s_dofs[:3]]
        x_b_spun = X_b + u_nl_spun[s_dofs[:3]]
        np.testing.assert_allclose(x_b_spun, Q @ x_b, rtol=0.0, atol=1.0e-12)
        R_b = rotation_matrix_from_vector(u_nl[s_dofs[3:6]])
        R_b_spun = rotation_matrix_from_vector(u_nl_spun[s_dofs[3:6]])
        np.testing.assert_allclose(R_b_spun, Q @ R_b, rtol=0.0, atol=1.0e-12)


def test_interp_transfer_preserves_physical_spatial_wrench() -> None:
    """The exact moment-defect discriminator: the new transfer preserves the spatial wrench.

    The old common-covector transfer changed the total physical spatial moment
    by Delta_M = D^T (m_b + A^T f_b) with D = sum_i w_i J_l(theta_i)^-1 -
    J_l(theta_bar)^-1; the mean-tangent transfer G_i^T (m_b + A^T f_b) removes
    that defect exactly.
    """
    model, meta, T, u0, q, mpc_map, u_lin = _interp_map_with_distinct_master_rotations()
    u_nl = mpc_map.apply_kinematics(u_lin)
    dof_manager = model.mesh.dof_manager

    # dof -> node id lookup for translation dofs.
    node_of_dof = {}
    for coupling in meta["couplings"]:
        for nid in [coupling["slave"]] + list(coupling["masters"]):
            for d in dof_manager.get_node_dofs(nid):
                node_of_dof[int(d)] = int(nid)

    def node_position(nid):
        node = model.mesh.get_node(nid)
        return np.array([node.x, node.y, node.z], dtype=float)

    rng = np.random.default_rng(20261019)
    imbalance = np.zeros_like(u_nl)
    for entry in mpc_map.entries:
        imbalance[entry.slave_u_dofs] = rng.uniform(-2.0e3, 2.0e3, 3)
        imbalance[entry.slave_theta_dofs] = rng.uniform(-5.0e2, 5.0e2, 3)
    transferred = mpc_map.transfer_slave_imbalance_to_masters(u_nl, imbalance)

    for entry in mpc_map.entries:
        f_b = imbalance[entry.slave_u_dofs]
        m_b = imbalance[entry.slave_theta_dofs]
        theta_m, G_list = entry.mean_state(u_nl)
        A = rotating_mpc_lever_arm_jacobian(theta_m, entry.eccentricity)
        v = m_b + A.T @ f_b
        slave_id = node_of_dof[int(entry.slave_u_dofs[0])]
        x_b = node_position(slave_id) + u_nl[entry.slave_u_dofs]
        F_before = f_b
        M_before = spatial_moment_from_rotation_covector(
            u_nl[entry.slave_theta_dofs], m_b
        ) + np.cross(x_b, f_b)
        F_after = np.zeros(3)
        M_after = np.zeros(3)
        for (m_u, m_theta, _w), _G in zip(entry.masters, G_list):
            f_i = transferred[m_u]
            m_i = transferred[m_theta]
            x_i = node_position(node_of_dof[int(m_u[0])]) + u_nl[m_u]
            F_after += f_i
            M_after += spatial_moment_from_rotation_covector(
                u_nl[m_theta], m_i
            ) + np.cross(x_i, f_i)
        scale = max(np.linalg.norm(M_before), np.linalg.norm(F_before), 1.0)
        np.testing.assert_allclose(F_after, F_before, rtol=0.0, atol=1.0e-9 * scale)
        np.testing.assert_allclose(M_after, M_before, rtol=0.0, atol=1.0e-9 * scale)
        # The old averaged-map defect D is nonzero on this state: the removed
        # mechanism is real, not vacuous.
        D = np.zeros((3, 3))
        theta_lin = entry.master_rotation(u_nl)
        for (_m_u, m_theta, w) in entry.masters:
            D += w * _rotation_left_jacobian_inverse(u_nl[m_theta])
        D -= _rotation_left_jacobian_inverse(theta_lin)
        delta_M = D.T @ v
        assert np.linalg.norm(delta_M) > 1.0e-6 * scale


def test_interp_map_jacobian_reduced_force_and_geometric_tangent_match_fd() -> None:
    """FD of the whole forward map, reduced force and mean derivative (nonzero m_b)."""
    model, meta = _build_coupled_cantilever(
        "interpolated", eccentricity=(0.02, 0.03, -0.06)
    )
    K0, _ = assemble_stiffness_matrix(model)
    zero = np.zeros(K0.shape[0], dtype=float)
    _, _, T, u0, _, _ = build_constraint_transformation(K0, zero, model)
    mpc_map = build_nonlinear_rotating_mpc_map(model, T)
    assert mpc_map is not None
    n_red = int(T.shape[1])
    # Distinct finite master rotations arise from q itself so that
    # u_lin = T q + u0 stays the exact parameterization used by the FD.
    rng = np.random.default_rng(20261021)
    q = np.zeros(n_red, dtype=float)
    for col in range(n_red):
        dof = int(T.indices[T.indptr[col]])
        if dof % 6 < 3:
            q[col] = rng.uniform(-0.03, 0.03)
        else:
            q[col] = rng.uniform(-0.4, 0.4)

    def full_u(q_vec):
        return mpc_map.apply_kinematics(np.asarray(T @ q_vec + u0, dtype=float).reshape(-1))

    u_nl = full_u(q)
    J = mpc_map.jacobian(u_nl).toarray()
    eps = 1.0e-7
    J_fd = np.zeros_like(J)
    for j in range(n_red):
        dq = np.zeros(n_red, dtype=float)
        dq[j] = eps
        J_fd[:, j] = (full_u(q + dq) - full_u(q - dq)) / (2.0 * eps)
    np.testing.assert_allclose(J, J_fd, rtol=1.0e-7, atol=1.0e-8)

    # reduce_force is exactly J^T f, including the slave moment covector.
    rng = np.random.default_rng(20261020)
    f = rng.uniform(-1.0, 1.0, u_nl.shape[0])
    for entry in mpc_map.entries:
        f[entry.slave_u_dofs] = rng.uniform(-2.0e3, 2.0e3, 3)
        f[entry.slave_theta_dofs] = rng.uniform(-5.0e2, 5.0e2, 3)
    g = mpc_map.reduce_force(u_nl, f)
    np.testing.assert_allclose(g, J.T @ f, rtol=0.0, atol=1.0e-9)

    # Geometric tangent: complete FD of the slave-correction covector g(q),
    # including mean second derivatives and the m_b-weighted orientation
    # Hessian (the old force-only lever Hessian omits them).
    K_geom = mpc_map.geometric_tangent(u_nl, f).toarray()
    T_dense = T.toarray()
    base = g - T_dense.T @ f
    assert np.linalg.norm(base) > 0.0

    def corr(q_vec):
        u_q = full_u(q_vec)
        return mpc_map.reduce_force(u_q, f) - T_dense.T @ f

    eps_k = 1.0e-6
    K_fd = np.zeros_like(K_geom)
    for j in range(n_red):
        dq = np.zeros(n_red, dtype=float)
        dq[j] = eps_k
        K_fd[:, j] = (corr(q + dq) - corr(q - dq)) / (2.0 * eps_k)
    np.testing.assert_allclose(K_geom, K_fd, rtol=1.0e-6, atol=1.0e-6)
    # The m_b orientation Hessian is non-vacuous: a force-only geometric
    # tangent would leave a large residual on this state.
    f_force_only = f.copy()
    for entry in mpc_map.entries:
        f_force_only[entry.slave_theta_dofs] = 0.0
    K_force_only = mpc_map.geometric_tangent(u_nl, f_force_only).toarray()
    assert np.linalg.norm(K_geom - K_force_only) > 1.0e-6


def test_interp_downstream_rotation_dependency_refused_and_direct_retained() -> None:
    """Multi-master slave rotations are nonlinear and refuse downstream consumption;
    single-master direct slave rotations stay linear and admissible."""
    model, meta = _build_coupled_cantilever("interpolated")
    dof_manager = model.mesh.dof_manager
    slave_beam = meta["beam_nodes"][1]
    extra = 500
    model.add_node(extra, 0.5, 0.05, 0.0)
    ry_slave = int(dof_manager.get_node_dofs(slave_beam)[4])
    ry_extra = int(dof_manager.get_node_dofs(extra)[4])
    model.add_constraint_equation(
        terms=((ry_extra, 1.0), (ry_slave, -1.0)),
        rhs=0.0,
        source_id="downstream_interp_ry_eq",
        dependent_dof=ry_extra,
    )
    K0, _ = assemble_stiffness_matrix(model)
    zero = np.zeros(K0.shape[0], dtype=float)
    _, _, T, _, _, _ = build_constraint_transformation(K0, zero, model)
    with pytest.raises(ValueError, match="rotating slave"):
        build_nonlinear_rotating_mpc_map(model, T)

    # Direct single-master route: slave rotation stays linear, dependency remains supported.
    model_d, meta_d = _build_coupled_cantilever("eccentric")
    slave_beam_d = meta_d["beam_nodes"][1]
    model_d.add_node(extra, 0.5, 0.05, 0.0)
    ry_slave_d = int(dof_manager.get_node_dofs(slave_beam_d)[4])
    ry_extra_d = int(dof_manager.get_node_dofs(extra)[4])
    model_d.add_constraint_equation(
        terms=((ry_extra_d, 1.0), (ry_slave_d, -1.0)),
        rhs=0.0,
        source_id="downstream_direct_ry_eq",
        dependent_dof=ry_extra_d,
    )
    K0d, _ = assemble_stiffness_matrix(model_d)
    _, _, Td, _, _, _ = build_constraint_transformation(K0d, zero, model_d)
    mpc_map = build_nonlinear_rotating_mpc_map(model_d, Td)
    assert mpc_map is not None and mpc_map.active


def test_interp_negative_shape_weights_refused_at_map_build() -> None:
    """Mixed-sign weights admitted by the partition validator are refused on the mean route."""
    model, meta = _build_coupled_cantilever("interpolated")
    elem_id = 200
    old_elem = model.mesh.elements[elem_id]
    model.mesh.elements[elem_id] = InterpolatedBeamShellMPCElement(
        elem_id,
        old_elem.beam_node_id,
        old_elem.shell_node_ids,
        np.array([1.5, -0.5], dtype=float),
        eccentricity=old_elem.eccentricity,
    )
    K0, _ = assemble_stiffness_matrix(model)
    zero = np.zeros(K0.shape[0], dtype=float)
    _, _, T, _, _, _ = build_constraint_transformation(K0, zero, model)
    with pytest.raises(ValueError, match="nonnegative"):
        build_nonlinear_rotating_mpc_map(model, T)
