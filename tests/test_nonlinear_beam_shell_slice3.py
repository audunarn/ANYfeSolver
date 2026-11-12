"""Slice 3 source-repair regressions: R1, R2 guard, R3, R4, R5.

Covers the confirmed non-author review findings adopted in
``.development/nonlinear-master-slave-coupling-20261017/supervisor_findings.md``:

R1  downstream affine dependencies on rotating slave translations are refused
    (fail closed) while ordinary linear chains, upstream shell row ties and
    rotating slave *rotation* dependencies remain supported;
R2  initial stress/prestrain fields stay refused for corotational kinematics
    (existing admission guard retained, no new capability);
R3  the fully constrained (n_red == 0) return uses the rotating-map physical
    state, not the linearized lever-arm vector;
R4  rotating-MPC ``auto`` selects the work-conjugate internal force even with
    follower loads, explicit legacy ``rotated``/``consistent`` modes are
    refused on the new route, and legacy models without a rotating descriptor
    keep their explicit-mode semantics;
R5  displacement-control row and proportional-load admission use the actual
    initial mapped state ``full_u(q)`` and its Jacobian.
Final repair: the intrinsic-mean domain guards raise a typed
    ``SO3MeanDomainError`` (a ``ValueError``), a displacement-control
    line-search trial that leaves the supported rotation chart is refused
    and backtracked instead of aborting the solve, and a step with no
    admissible sufficient decrease at a residual-dominated state is a
    classified failure with rollback and bounded subdivision, never a
    committed full step that sustains a Newton limit cycle.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition,
    DisplacementControl,
    FEModel,
    LoadCase,
    create_shell_element,
    solve_static_nonlinear,
)
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.corotational import (
    SO3MeanDomainError,
    _so3_weighted_mean,
    build_nonlinear_rotating_mpc_map,
    rotation_matrix_from_vector,
)
from anyfesolver.elements import BeamElement, CoupledBeamShellElement
from anyfesolver.matrix_assembly import (
    assemble_load_vector,
    assemble_stiffness_matrix,
)
from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement
from anyfesolver.nonlinear_static import (
    ShellInitialField,
    _assemble_nonlinear_system,
    _weighted_external_load_system,
)

CROSS_SECTION = {
    "area": 0.01 * 0.04,
    "Iy": 0.01 * (0.04**3) / 12.0,
    "Iz": 0.04 * (0.01**3) / 12.0,
    "J": 1.0e-8,
}


def _synthetic_postyield_controller_history(model):
    """Accumulated plastic history for synthetic postyield controller laws."""
    state = model.mesh.elements[1].init_nonlinear_state(5)
    state["alpha"][:] = 1e-4
    return {1: state}


def _build_coupled_cantilever(
    route: str,
    *,
    nx: int = 2,
    ny: int = 1,
    length: float = 1.0,
    width: float = 0.2,
    thickness: float = 0.02,
    eccentricity: tuple[float, float, float] = (0.0, 0.0, 0.05),
    root_supports: bool = True,
) -> tuple[FEModel, dict]:
    """Cantilever Q4 shell strip with a coupled eccentric beam stiffener."""
    model = FEModel(f"slice3_{route}_{nx}x{ny}")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    r_vec = np.asarray(eccentricity, dtype=float)

    grid: dict[tuple[int, int], int] = {}
    node_id = 1
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(-0.5 * width, 0.5 * width, ny + 1)
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            model.add_node(node_id, float(x), float(y), 0.0)
            grid[(ix, iy)] = node_id
            node_id += 1

    elem_id = 1
    for ix in range(nx):
        for iy in range(ny):
            model.add_element(
                elem_id,
                create_shell_element(
                    elem_id,
                    [
                        grid[(ix, iy)],
                        grid[(ix + 1, iy)],
                        grid[(ix + 1, iy + 1)],
                        grid[(ix, iy + 1)],
                    ],
                    "steel",
                    thickness=thickness,
                ),
            )
            elem_id += 1

    beam_nodes: list[int] = []
    couplings: list[dict] = []
    if route == "eccentric":
        for ix, x in enumerate(xs):
            master_id = grid[(ix, 0)]
            b_id = 100 + ix
            model.add_node(b_id, float(x), -0.5 * width, float(r_vec[2]))
            beam_nodes.append(b_id)
            mpc_id = 200 + ix
            model.add_element(
                mpc_id,
                CoupledBeamShellElement(
                    mpc_id, beam_node_id=b_id, shell_node_id=master_id,
                    material_name="steel", eccentricity=r_vec,
                ),
            )
            couplings.append(
                {"slave": b_id, "masters": [master_id],
                 "weights": np.array([1.0]), "r": r_vec.copy()}
            )
    elif route == "interpolated":
        weights = np.array([0.6, 0.4], dtype=float)
        for ix, x in enumerate(xs):
            m1 = grid[(ix, 0)]
            m2 = grid[(ix, ny)]
            b_id = 100 + ix
            model.add_node(b_id, float(x), 0.0, float(r_vec[2]))
            beam_nodes.append(b_id)
            mpc_id = 200 + ix
            model.add_element(
                mpc_id,
                InterpolatedBeamShellMPCElement(
                    mpc_id, b_id, [m1, m2], weights, eccentricity=r_vec,
                ),
            )
            couplings.append(
                {"slave": b_id, "masters": [m1, m2],
                 "weights": weights.copy(), "r": r_vec.copy()}
            )
    else:
        raise ValueError(f"Unknown route {route!r}")

    for ix in range(nx):
        model.add_element(
            300 + ix,
            BeamElement(300 + ix, [beam_nodes[ix], beam_nodes[ix + 1]],
                       "steel", cross_section=dict(CROSS_SECTION)),
        )

    root_nodes = [grid[(0, iy)] for iy in range(ny + 1)]
    if root_supports:
        model.add_boundary_condition(
            BoundaryCondition(
                "root", root_nodes,
                {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
            )
        )
    model.apply_boundary_conditions()
    meta = {
        "route": route,
        "grid": grid,
        "beam_nodes": beam_nodes,
        "root_nodes": root_nodes,
        "couplings": couplings,
        "eccentricity": r_vec,
        "nx": nx,
        "ny": ny,
    }
    return model, meta


def _transformation(model: FEModel, load: LoadCase | None = None):
    K0, _ = assemble_stiffness_matrix(model)
    F = np.zeros(K0.shape[0], dtype=float)
    if load is not None:
        F, _ = assemble_load_vector(model, load)
    return build_constraint_transformation(K0, F, model)


def _assert_element_states_equal(left, right) -> None:
    """Exact deep equality for caller-supplied element-state mappings."""
    assert set(left) == set(right)
    for key in left:
        a, b = left[key], right[key]
        assert type(a) is type(b)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            _assert_element_states_equal(a, b)
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for item_a, item_b in zip(a, b):
                if isinstance(item_a, np.ndarray):
                    np.testing.assert_array_equal(item_a, item_b)
                else:
                    assert item_a == item_b
        else:
            assert a == b


# ---------------------------------------------------------------------------
# R1: fail closed for downstream dependencies on rotating slave translations
# ---------------------------------------------------------------------------

def test_r1_downstream_generic_equation_on_rotating_slave_translation_refused() -> None:
    model, meta = _build_coupled_cantilever("eccentric")
    dofs = model.mesh.dof_manager.get_node_dofs
    slave_beam = meta["beam_nodes"][1]
    extra = 500
    model.add_node(extra, 0.5, 0.05, 0.0)
    uz_slave = int(dofs(slave_beam)[2])
    uz_extra = int(dofs(extra)[2])
    model.add_constraint_equation(
        terms=((uz_extra, 1.0), (uz_slave, -1.0)),
        rhs=0.0,
        source_id="downstream_uz_eq",
        dependent_dof=uz_extra,
    )
    _, _, T, _, _, _ = _transformation(model)
    with pytest.raises(ValueError, match="rotating slave"):
        build_nonlinear_rotating_mpc_map(model, T)
    load = LoadCase("tip")
    load.add_nodal_load(meta["beam_nodes"][-1], forces=(0.0, 0.0, -1.0e4))
    model.add_load_case(load)
    with pytest.raises(ValueError, match="rotating slave"):
        solve_static_nonlinear(
            model, load, kinematics="corotational", num_steps=2,
            max_iterations=10,
        )


def test_r1_second_rotating_coupling_downstream_of_slave_refused() -> None:
    model, meta = _build_coupled_cantilever("eccentric")
    slave_beam = meta["beam_nodes"][1]
    downstream_beam = 500
    model.add_node(downstream_beam, 0.5, -0.1, 0.08)
    model.add_element(
        260,
        CoupledBeamShellElement(
            260, beam_node_id=downstream_beam, shell_node_id=slave_beam,
            material_name="steel",
            eccentricity=np.array([0.0, 0.0, 0.03]),
        ),
    )
    _, _, T, _, _, _ = _transformation(model)
    with pytest.raises(ValueError, match="rotating slave"):
        build_nonlinear_rotating_mpc_map(model, T)


def test_r1_downstream_rotation_covector_dependency_remains_supported() -> None:
    model, meta = _build_coupled_cantilever("eccentric")
    dofs = model.mesh.dof_manager.get_node_dofs
    slave_beam = meta["beam_nodes"][1]
    extra = 500
    model.add_node(extra, 0.5, 0.05, 0.0)
    ry_slave = int(dofs(slave_beam)[4])
    ry_extra = int(dofs(extra)[4])
    model.add_constraint_equation(
        terms=((ry_extra, 1.0), (ry_slave, -1.0)),
        rhs=0.0,
        source_id="downstream_ry_eq",
        dependent_dof=ry_extra,
    )
    _, _, T, _, _, _ = _transformation(model)
    mpc_map = build_nonlinear_rotating_mpc_map(model, T)
    assert mpc_map is not None and mpc_map.active


def test_r1_upstream_shell_ties_and_linear_chains_remain_supported() -> None:
    # Upstream shell row tie: a shell node translation tied to another shell
    # node translation. Both are coupling masters, not rotating slaves.
    model, meta = _build_coupled_cantilever("eccentric", nx=2, ny=1)
    dofs = model.mesh.dof_manager.get_node_dofs
    uy_a = int(dofs(meta["grid"][(1, 0)])[1])
    uy_b = int(dofs(meta["grid"][(1, 1)])[1])
    model.add_constraint_equation(
        terms=((uy_b, 1.0), (uy_a, -1.0)),
        rhs=0.0,
        source_id="upstream_row_tie",
        dependent_dof=uy_b,
    )
    _, _, T, _, _, _ = _transformation(model)
    mpc_map = build_nonlinear_rotating_mpc_map(model, T)
    assert mpc_map is not None and mpc_map.active

    # Ordinary linear chain without any rotating descriptor still solves.
    shell_only = FEModel("shell_only_chain")
    shell_only.add_material("steel", 2.1e11, 0.3, density=7850.0)
    shell_only.add_node(1, 0.0, 0.0, 0.0)
    shell_only.add_node(2, 1.0, 0.0, 0.0)
    shell_only.add_node(3, 1.0, 0.2, 0.0)
    shell_only.add_node(4, 0.0, 0.2, 0.0)
    shell_only.add_element(
        1, create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.02)
    )
    sd = shell_only.mesh.dof_manager.get_node_dofs
    shell_only.add_constraint_equation(
        terms=((int(sd(3)[2]), 1.0), (int(sd(2)[2]), -1.0)),
        rhs=0.0,
        source_id="linear_chain",
        dependent_dof=int(sd(3)[2]),
    )
    shell_only.add_boundary_condition(
        BoundaryCondition(
            "root", [1, 4],
            {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
        )
    )
    shell_only.apply_boundary_conditions()
    load = LoadCase("tip")
    load.add_nodal_load(3, forces=(0.0, 0.0, -50.0))
    shell_only.add_load_case(load)
    _, _, T2, _, _, _ = _transformation(shell_only, load)
    assert build_nonlinear_rotating_mpc_map(shell_only, T2) is None
    result = solve_static_nonlinear(
        shell_only, load, kinematics="corotational", num_steps=1,
        max_iterations=10,
    )
    assert result.status == "completed"


# ---------------------------------------------------------------------------
# R2: initial stress/prestrain fields stay refused for corotational kinematics
# ---------------------------------------------------------------------------

def test_r2_initial_fields_refused_for_corotational_kinematics() -> None:
    model, meta = _build_coupled_cantilever("eccentric")
    load = LoadCase("tip")
    load.add_nodal_load(meta["beam_nodes"][-1], forces=(0.0, 0.0, -1.0e4))
    model.add_load_case(load)
    with pytest.raises(NotImplementedError, match="von_karman"):
        solve_static_nonlinear(
            model,
            load,
            kinematics="corotational",
            num_steps=2,
            max_iterations=10,
            initial_fields={1: ShellInitialField(membrane_stress=[5.0e7, 0.0, 0.0])},
        )


# ---------------------------------------------------------------------------
# R3: fully constrained finite motion uses the rotating map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ("eccentric", "interpolated"))
def test_r3_fully_constrained_finite_rotation_uses_rotating_map(route: str) -> None:
    alpha = 0.35
    model, meta = _build_coupled_cantilever(
        route, nx=2, ny=1, root_supports=False
    )
    rot = rotation_matrix_from_vector(np.array([0.0, alpha, 0.0]))
    dofs = model.mesh.dof_manager.get_node_dofs

    # Prescribe a rigid rotation about +y through the origin on EVERY shell
    # node; all shell DOFs become fixed and the beam slaves are eliminated by
    # their coupling, so n_red == 0.
    for (ix, iy), node in meta["grid"].items():
        x = float(model.mesh.get_node(node).x)
        y = float(model.mesh.get_node(node).y)
        z = float(model.mesh.get_node(node).z)
        disp = (rot - np.eye(3)) @ np.array([x, y, z])
        model.add_boundary_condition(
            BoundaryCondition(
                f"rigid_{node}",
                [node],
                {
                    "ux": float(disp[0]),
                    "uy": float(disp[1]),
                    "uz": float(disp[2]),
                    "rx": 0.0,
                    "ry": alpha,
                    "rz": 0.0,
                },
            )
        )
    model.apply_boundary_conditions()
    load = LoadCase("none")
    load.add_nodal_load(meta["grid"][(2, 0)], forces=(0.0, 0.0, -1.0))
    model.add_load_case(load)

    result = solve_static_nonlinear(
        model, load, kinematics="corotational", corotational_tangent="auto",
        num_steps=1, max_iterations=5,
    )
    assert result.info["stop_reason"] == "empty_reduced_system"
    u = np.asarray(result.displacements, dtype=float)

    # The returned state must carry the prescribed shell motion itself; an
    # all-zero returned state cannot pass through self-consistency alone.
    for (ix, iy), node in meta["grid"].items():
        x = float(model.mesh.get_node(node).x)
        y = float(model.mesh.get_node(node).y)
        z = float(model.mesh.get_node(node).z)
        disp = (rot - np.eye(3)) @ np.array([x, y, z])
        node_dofs = dofs(node)
        np.testing.assert_allclose(
            u[node_dofs[:3]], disp, rtol=1.0e-12, atol=1.0e-14
        )
        np.testing.assert_allclose(
            u[node_dofs[3:6]], [0.0, alpha, 0.0], rtol=1.0e-12, atol=1.0e-14
        )

    for coupling in meta["couplings"]:
        slave_dofs = dofs(coupling["slave"])
        u_bar = np.zeros(3)
        theta_bar = np.zeros(3)
        for m_id, w in zip(coupling["masters"], coupling["weights"]):
            m_dofs = dofs(m_id)
            u_bar += float(w) * u[m_dofs[:3]]
            theta_bar += float(w) * u[m_dofs[3:6]]
        expected_u_b = u_bar + (rot - np.eye(3)) @ coupling["r"]
        np.testing.assert_allclose(
            u[slave_dofs[:3]], expected_u_b, rtol=1.0e-12, atol=1.0e-14
        )
        np.testing.assert_allclose(
            u[slave_dofs[3:6]], [0.0, alpha, 0.0], rtol=1.0e-12, atol=1.0e-14
        )

    # The rigid motion must be strain free: the internal force at the mapped
    # state vanishes while the linearized lever-arm state carries O(e * alpha)
    # strain energy.
    _, _, T, u0, _, _ = _transformation(model, load)
    f_mapped, _, _ = _assemble_nonlinear_system(
        model, u, {}, 5, tangent=False, kinematics="corotational",
        corotational_tangent="work_conjugate",
    )
    f_linearized, _, _ = _assemble_nonlinear_system(
        model, np.asarray(u0, dtype=float), {}, 5, tangent=False,
        kinematics="corotational", corotational_tangent="work_conjugate",
    )
    assert float(np.linalg.norm(f_mapped)) < 1.0e-6 * max(
        float(np.linalg.norm(f_linearized)), 1.0
    )


# ---------------------------------------------------------------------------
# R4: work-conjugate dispatch with follower loads; legacy modes on the route
# ---------------------------------------------------------------------------

def _coupled_with_follower(pressure: float):
    model, meta = _build_coupled_cantilever(
        "eccentric", nx=2, ny=1, eccentricity=(0.0, 0.0, 0.05)
    )
    load = LoadCase("tip_and_pressure", follower_pressure=True)
    load.add_nodal_load(
        meta["beam_nodes"][-1], forces=(-4.0e4, 1.5e4, -3.2e5)
    )
    if pressure != 0.0:
        for element_id in (1, 2):
            load.add_pressure_load(element_id, pressure)
    model.add_load_case(load)
    return model, meta, load


@pytest.mark.parametrize("pressure", (200.0, 1.0e-6))
def test_r4_auto_with_follower_loads_selects_work_conjugate(pressure: float) -> None:
    model, meta, load = _coupled_with_follower(pressure)
    result = solve_static_nonlinear(
        model, load, num_steps=6, max_iterations=25, tolerance=1.0e-9,
        kinematics="corotational", corotational_tangent="auto",
    )
    assert result.converged
    assert result.info["follower_pressure"] is True
    assert result.info["corotational_tangent"] == "work_conjugate"


def test_r4_internal_force_dispatch_does_not_jump_as_pressure_tends_to_zero() -> None:
    modes = []
    for pressure in (1.0e-6, 0.0):
        model, meta, load = _coupled_with_follower(pressure)
        result = solve_static_nonlinear(
            model, load, num_steps=6, max_iterations=25, tolerance=1.0e-9,
            kinematics="corotational", corotational_tangent="auto",
        )
        assert result.converged
        modes.append(result.info["corotational_tangent"])
    assert modes == ["work_conjugate", "work_conjugate"]


@pytest.mark.parametrize("mode", ("rotated", "consistent"))
def test_r4_explicit_legacy_modes_refused_on_rotating_route(mode: str) -> None:
    model, meta, load = _coupled_with_follower(0.0)
    with pytest.raises(ValueError, match="rotating-MPC route"):
        solve_static_nonlinear(
            model, load, num_steps=2, max_iterations=10,
            kinematics="corotational", corotational_tangent=mode,
        )


def test_r4_legacy_explicit_modes_retained_without_rotating_descriptor() -> None:
    model = FEModel("legacy_shell")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.2, 0.0), (0.0, 0.2, 0.0)]
    for i, c in enumerate(coords, start=1):
        model.add_node(i, *c)
    model.add_element(
        1, create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.02)
    )
    model.add_boundary_condition(
        BoundaryCondition(
            "root", [1, 4],
            {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
        )
    )
    model.apply_boundary_conditions()
    dead = LoadCase("dead")
    dead.add_nodal_load(2, forces=(0.0, 0.0, -2.0e3))
    model.add_load_case(dead)
    rotated_result = solve_static_nonlinear(
        model, dead, num_steps=2, max_iterations=15,
        kinematics="corotational", corotational_tangent="rotated",
    )
    assert rotated_result.info["corotational_tangent"] == "rotated"

    follower = LoadCase("follower", follower_pressure=True)
    follower.add_pressure_load(1, 50.0)
    follower.add_nodal_load(2, forces=(0.0, 0.0, -2.0e3))
    model.add_load_case(follower)
    consistent_result = solve_static_nonlinear(
        model, follower, num_steps=2, max_iterations=15,
        kinematics="corotational", corotational_tangent="consistent",
    )
    assert consistent_result.info["corotational_tangent"] == "consistent"
    assert consistent_result.info["follower_pressure"] is True


def test_r4_follower_complete_reduced_derivative_matches_finite_differences() -> None:
    model, meta, load = _coupled_with_follower(200.0)
    load.add_nodal_load(
        meta["beam_nodes"][1], moments=(12.0, -18.0, 9.0)
    )
    K0, _ = assemble_stiffness_matrix(model)
    F_ext_ref, _ = assemble_load_vector(model, load)
    _, _, T, u0, _, _ = build_constraint_transformation(K0, F_ext_ref, model)
    mpc_map = build_nonlinear_rotating_mpc_map(model, T)
    assert mpc_map is not None

    rng = np.random.default_rng(4242)
    n_red = int(T.shape[1])
    q = np.zeros(n_red, dtype=float)
    for col in range(n_red):
        dof = int(T.indices[T.indptr[col]])
        if dof % 6 < 3:
            q[col] = rng.uniform(-0.03, 0.03)
        else:
            q[col] = rng.uniform(-0.35, 0.35)

    def full_u(q_vec: np.ndarray) -> np.ndarray:
        return mpc_map.apply_kinematics(
            np.asarray(T @ q_vec + u0, dtype=float).reshape(-1)
        )

    def net_reduced_force(q_vec: np.ndarray) -> np.ndarray:
        u_q = full_u(q_vec)
        f_int, _, _ = _assemble_nonlinear_system(
            model, u_q, {}, 5, tangent=False, kinematics="corotational",
            corotational_tangent="work_conjugate",
        )
        f_ext, _ = _weighted_external_load_system(
            model, [(load, 1.0)], u_q, tangent=False
        )
        return mpc_map.reduce_force(u_q, f_int - f_ext)

    u = full_u(q)
    f_int, K_T, _ = _assemble_nonlinear_system(
        model, u, {}, 5, tangent=True, kinematics="corotational",
        corotational_tangent="work_conjugate",
    )
    f_ext, K_ext = _weighted_external_load_system(
        model, [(load, 1.0)], u, tangent=True
    )
    K_red = mpc_map.reduce_tangent(
        u, K_T, f_int - f_ext, K_ext if K_ext is not None and K_ext.nnz else None
    ).toarray()

    eps = 1.0e-6
    K_fd = np.zeros_like(K_red)
    for j in range(n_red):
        dq = np.zeros(n_red, dtype=float)
        dq[j] = eps
        K_fd[:, j] = (
            net_reduced_force(q + dq) - net_reduced_force(q - dq)
        ) / (2.0 * eps)

    rel_err = np.linalg.norm(K_red - K_fd, ord="fro") / np.linalg.norm(
        K_fd, ord="fro"
    )
    assert rel_err < 1.0e-6

    # External reduced-force derivative on its own scale: with steel
    # stiffness a relative full-system tolerance can hide an absent external
    # load tangent.  Differentiate J(u)^T F_ext(u) and compare against
    # J^T K_ext J + geometric_tangent(u, F_ext), normalized by that matrix.
    def ext_reduced_force(q_vec: np.ndarray) -> np.ndarray:
        u_q = full_u(q_vec)
        f_ext_q, _ = _weighted_external_load_system(
            model, [(load, 1.0)], u_q, tangent=False
        )
        return mpc_map.reduce_force(u_q, f_ext_q)

    J = mpc_map.jacobian(u)
    K_ext_red = (
        (J.T @ K_ext @ J).toarray()
        + np.asarray(
            mpc_map.geometric_tangent(u, f_ext).todense()
        )
    )
    K_ext_fd = np.zeros_like(K_ext_red)
    for j in range(n_red):
        dq = np.zeros(n_red, dtype=float)
        dq[j] = eps
        K_ext_fd[:, j] = (
            ext_reduced_force(q + dq) - ext_reduced_force(q - dq)
        ) / (2.0 * eps)

    ext_scale = np.linalg.norm(K_ext_fd, ord="fro")
    assert ext_scale > 0.0
    rel_err_ext = np.linalg.norm(
        K_ext_red - K_ext_fd, ord="fro"
    ) / ext_scale
    assert rel_err_ext < 1.0e-6


# ---------------------------------------------------------------------------
# R5: displacement-control admission at the actual initial mapped state
# ---------------------------------------------------------------------------

def _r5_model():
    """Root clamp plus fixed master uz; the tip beam slave uz is the control.

    The linear MPC gives u_b,z = u_s,z (the lever arm (theta x r0)_z is zero
    for r0 = (0, 0, e)), so with the master uz fixed the linear control row
    vanishes.  The nonlinear position u_b,z = ... + (cos(ry) - 1) e has a
    nonzero finite-state derivative -e sin(ry), so the same control coordinate
    is usable at a finite-rotation restart.
    """
    model, meta = _build_coupled_cantilever(
        "eccentric", nx=2, ny=1, eccentricity=(0.0, 0.0, 0.05)
    )
    tip_master = meta["grid"][(meta["nx"], 0)]
    model.add_boundary_condition(
        BoundaryCondition("fix_master_uz", [tip_master], {"uz": 0.0})
    )
    model.apply_boundary_conditions()
    pre_load = LoadCase("bending")
    pre_load.add_nodal_load(
        meta["grid"][(meta["nx"], meta["ny"])], forces=(0.0, 0.0, -1.0e5)
    )
    model.add_load_case(pre_load)
    slave_load = LoadCase("slave_uz_load")
    slave_load.add_nodal_load(meta["beam_nodes"][-1], forces=(0.0, 0.0, -5.0e3))
    model.add_load_case(slave_load)
    return model, meta, pre_load, slave_load


def test_r5_zero_jacobian_control_refused_at_zero_rotation() -> None:
    model, meta, _pre_load, slave_load = _r5_model()
    control = DisplacementControl(
        node_id=meta["beam_nodes"][-1], dof="uz", target_displacement=-0.01
    )
    with pytest.raises(ValueError, match="zero control Jacobian"):
        solve_static_nonlinear(
            model, slave_load, control="displacement",
            displacement_control=control,
            kinematics="corotational", num_steps=2, max_iterations=20,
            tolerance=1.0e-8,
        )


def test_r5_finite_state_admits_nonlinear_only_control_and_load() -> None:
    model, meta, pre_load, slave_load = _r5_model()
    dofs = model.mesh.dof_manager.get_node_dofs

    # Reach a genuine finite-rotation equilibrium with a dead shell load.
    pre = solve_static_nonlinear(
        model, pre_load, num_steps=6, max_iterations=25, tolerance=1.0e-9,
        kinematics="corotational", corotational_tangent="auto",
    )
    assert pre.converged
    assert pre.status == "completed"
    assert abs(float(pre.load_factor) - 1.0) <= 1.0e-12
    pre_u = np.asarray(pre.displacements, dtype=float)
    tip_master_ry = float(pre_u[dofs(meta["grid"][(meta["nx"], 0)])[4]])
    assert tip_master_ry != 0.0

    # The regression is the admission itself, not a particular large angle:
    # the control coordinate is nonlinear-only, so the linear row is exactly
    # zero while the finite-state kinematic Jacobian row and the proportional
    # load projection are both nonzero at this equilibrium (and both vanish
    # at the zero-rotation origin, which the refusal test covers).
    K0, _ = assemble_stiffness_matrix(model)
    F_prop, _ = assemble_load_vector(model, slave_load)
    _, _, T, _u0, _, _ = build_constraint_transformation(K0, F_prop, model)
    mpc_map = build_nonlinear_rotating_mpc_map(model, T)
    assert mpc_map is not None and mpc_map.active
    control = DisplacementControl(
        node_id=meta["beam_nodes"][-1], dof="uz", target_displacement=-0.01
    )
    row_full = control.full_row(model)
    row_linear = np.asarray(row_full @ T, dtype=float).reshape(-1)
    assert float(np.linalg.norm(row_linear)) == 0.0
    row_finite = np.asarray(row_full @ mpc_map.jacobian(pre_u)).reshape(-1)
    assert float(np.linalg.norm(row_finite)) > 0.0
    zero_u = np.zeros_like(pre_u)
    assert float(np.linalg.norm(mpc_map.reduce_force(zero_u, F_prop))) == 0.0
    assert float(np.linalg.norm(mpc_map.reduce_force(pre_u, F_prop))) > 0.0

    # Controlled increment near that state: the presolved equilibrium is held
    # by the same dead load as a constant load case, so the displacement-
    # control unknown only carries the slave-only proportional load and the
    # initial state is an exact equilibrium of the restarted system.
    initial_control = float(row_full @ pre_u)
    target = initial_control - 8.0e-6
    control_increment = DisplacementControl(
        node_id=meta["beam_nodes"][-1], dof="uz", target_displacement=float(target)
    )
    u_input = np.array(pre_u, dtype=float, copy=True)
    states_input = copy.deepcopy(dict(pre.element_states))
    u_snapshot = np.array(pre_u, dtype=float, copy=True)
    states_snapshot = copy.deepcopy(dict(pre.element_states))
    result = solve_static_nonlinear(
        model, slave_load, constant_load_case=pre_load,
        control="displacement", displacement_control=control_increment,
        kinematics="corotational", num_steps=2, max_iterations=30,
        tolerance=1.0e-8, initial_displacements=u_input,
        initial_element_states=states_input,
        equilibrate_initial_state=False,
    )
    assert result.status == "completed"
    # Caller-supplied restart inputs are preserved by the solve.
    np.testing.assert_array_equal(u_input, u_snapshot)
    _assert_element_states_equal(states_input, states_snapshot)
    uz = float(result.displacements[dofs(meta["beam_nodes"][-1])[2]])
    assert abs(uz - target) <= 1.0e-8
    assert abs(float(result.load_factor)) > 0.0
    # The accepted control motion is carried by the nonlinear map: the fixed
    # master translation stays exactly zero while the master rotation vector
    # changes enough to move the slave through the finite-rotation lever arm.
    master_dofs = dofs(meta["grid"][(meta["nx"], 0)])
    assert float(result.displacements[master_dofs[2]]) == 0.0
    theta_change = float(np.linalg.norm(
        np.asarray(result.displacements[master_dofs[3:6]], dtype=float)
        - pre_u[master_dofs[3:6]]
    ))
    assert theta_change > 1.0e-4


def test_d1_displacement_control_augmented_factorization_declared_general(
    monkeypatch,
) -> None:
    """D1: the assembled augmented DC system is declared MatrixClass.GENERAL.

    The augmented control system [K, -F; row, 0] is nonsymmetric even for a
    symmetric tangent (the proportional-load column and the control row are
    independent), so a symmetric declaration would silently consume only
    one triangle.  A spy backend records the declared class of every
    factorization of an augmented (n_red + 1)-shaped matrix and delegates the
    actual solve unchanged, so the regression is independent of optional
    PARDISO availability and does not count a bare SciPy pass as coverage.
    """
    import anyfesolver.nonlinear_static as nonlinear_static
    from anyfesolver.linalg import MatrixClass

    model, meta = _build_coupled_cantilever("eccentric")
    load = LoadCase("tip")
    load.add_nodal_load(
        meta["grid"][(meta["nx"], meta["ny"])], forces=(0.0, 0.0, -1.0e4)
    )
    model.add_load_case(load)
    control = DisplacementControl(
        node_id=meta["grid"][(meta["nx"], 0)], dof="uz",
        target_displacement=-1.0e-3,
    )
    K0, _ = assemble_stiffness_matrix(model)
    F_prop, _ = assemble_load_vector(model, load)
    _, _, T, _u0, _, _ = build_constraint_transformation(K0, F_prop, model)
    n_red = int(T.shape[1])
    aug_shape = (n_red + 1, n_red + 1)

    recorded: list = []
    real_factorize = nonlinear_static.factorize

    def spying_factorize(matrix, matrix_class, **kwargs):
        recorded.append((tuple(matrix.shape), matrix_class))
        return real_factorize(matrix, matrix_class, **kwargs)

    monkeypatch.setattr(nonlinear_static, "factorize", spying_factorize)
    result = solve_static_nonlinear(
        model, load, control="displacement", displacement_control=control,
        kinematics="corotational", num_steps=2, max_iterations=20,
        tolerance=1.0e-8,
    )
    assert result.status == "completed"
    aug_classes = [cls for shape, cls in recorded if shape == aug_shape]
    assert aug_classes, "no augmented displacement-control factorization observed"
    assert all(
        cls is MatrixClass.GENERAL for cls in aug_classes
    ), aug_classes


def test_intrinsic_mean_two_master_independent_geodesic_and_differential() -> None:
    from scipy.spatial.transform import Rotation

    thetas = [np.array([0.4, -0.2, 0.3]), np.array([-0.3, 0.5, 0.1])]
    weights = [0.35 * (1.0 + 1.0e-9), 0.65 * (1.0 + 1.0e-9)]
    def independent_geodesic(vectors):
        a, b = [Rotation.from_rotvec(t) for t in vectors]
        return (a * Rotation.from_rotvec(0.65 * (a.inv() * b).as_rotvec())).as_rotvec()

    expected = independent_geodesic(thetas)
    actual, blocks = _so3_weighted_mean(thetas, weights, "test")
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)
    for i in range(2):
        fd = np.empty((3, 3))
        for j in range(3):
            # The centered four-point stencil balances O(h**4) truncation
            # against O(eps/h) roundoff on the independent SciPy geodesic.
            h = np.finfo(float).eps ** 0.2
            samples = []
            for step in (-2, -1, 1, 2):
                displaced = [t.copy() for t in thetas]
                displaced[i][j] += step * h
                samples.append(independent_geodesic(displaced))
            fd[:, j] = ((samples[0] - samples[3])
                        + 8.0 * (samples[2] - samples[1])) / (12.0 * h)
        np.testing.assert_allclose(blocks[i], fd, rtol=1.0e-8, atol=1.0e-10)


@pytest.mark.parametrize("thetas,weights,reason", [
    ([[np.pi / 2, 0, 0], [-np.pi / 2, 0, 0]], [.5, .5], "ambiguous"),
    ([[np.pi + .01, 0, 0]] * 2, [.5, .5], "principal"),
    ([[0, 0, 0], [0, 1.0, 0], [0, .1, 0]], [.2, .4, .4], "convex"),
    ([[0, 0, 0], [0, .1, 0]], [-.1, 1.1], "nonnegative"),
])
def test_intrinsic_mean_ambiguous_or_unadmitted_domain_refused(thetas, weights, reason):
    with pytest.raises(ValueError, match=reason):
        _so3_weighted_mean(thetas, weights, "test")


def test_dc_zero_span_failed_attempt_is_bounded_and_rolls_back(monkeypatch) -> None:
    import anyfesolver.nonlinear_static as nonlinear_static

    model, meta = _build_coupled_cantilever("eccentric", nx=2, ny=1)
    tip = meta["grid"][(meta["nx"], 0)]
    load = LoadCase("tip")
    load.add_nodal_load(tip, forces=(0.0, 0.0, -100.0))
    model.add_load_case(load)
    real_assemble = nonlinear_static._assemble_nonlinear_system
    calls = []

    def unreconciled_force(*args, **kwargs):
        force, tangent, state = real_assemble(*args, **kwargs)
        force = force.copy()
        force[model.mesh.dof_manager.get_node_dofs(tip)[2]] += 1.0
        calls.append(1)
        assert len(calls) <= 10, "zero control span retried without progress"
        return force, tangent, state

    monkeypatch.setattr(nonlinear_static, "_assemble_nonlinear_system", unreconciled_force)
    result = solve_static_nonlinear(
        model, load, control="displacement",
        displacement_control=DisplacementControl(node_id=tip, dof="uz", target_displacement=0.0),
        kinematics="corotational", num_steps=1, max_iterations=1,
        tolerance=1.0e-8, record_increment_snapshots=True,
    )
    assert result.status != "completed"
    assert calls
    assert not result.steps and result.snapshots == ()
    np.testing.assert_array_equal(result.displacements, np.zeros_like(result.displacements))


def test_dc_accepted_subincrements_receive_independent_cutback_budget(monkeypatch) -> None:
    """Retry limits belong to a failed interval, not completed subdivisions.

    A deterministic corrector failure for oversized intervals requires more
    than eight cumulative cutbacks to complete one scheduled milestone. Small
    intervals use the actual element assembly, tangent and state transaction.
    """
    import inspect
    import anyfesolver.nonlinear_static as nonlinear_static

    model, meta = _build_coupled_cantilever("eccentric", nx=2, ny=1)
    tip = meta["grid"][(meta["nx"], 0)]
    load = LoadCase("tip")
    load.add_nodal_load(tip, forces=(0.0, 0.0, -100.0))
    model.add_load_case(load)
    target = -6.4e-5
    maximum_interval = abs(target) / 8
    real_factorize = nonlinear_static.factorize
    refused_spans = []

    class NoCorrection:
        def solve(self, rhs):
            return np.zeros_like(rhs)

    def bounded_corrector(matrix, matrix_class, **kwargs):
        frame = inspect.currentframe().f_back
        try:
            assert frame.f_code.co_name == "_solve_static_displacement_control"
            values = frame.f_locals
            span = abs(values["target"] - values["attempt_start_control"])
            if span > maximum_interval * (1 + 1.0e-10):
                refused_spans.append(span)
                return NoCorrection()
        finally:
            del frame
        return real_factorize(matrix, matrix_class, **kwargs)

    monkeypatch.setattr(nonlinear_static, "factorize", bounded_corrector)
    result = solve_static_nonlinear(
        model, load, control="displacement",
        displacement_control=DisplacementControl(
            node_id=tip, dof="uz", target_displacement=target),
        kinematics="corotational", num_steps=1, max_iterations=4,
        tolerance=1.0e-8, record_increment_snapshots=True,
        emit_restart_checkpoint=False,
    )
    assert refused_spans
    assert result.status == "completed", (result.status, result.stop_reason)
    assert result.info["displacement_control_adaptation"]["cutbacks"] > 8
    controls = [0.0, *[s.control_value for s in result.snapshots]]
    assert len(controls) > 8
    assert all(0 < earlier - later <= maximum_interval * (1 + 1.0e-8)
               for earlier, later in zip(controls, controls[1:]))
    assert abs(controls[-1] - target) <= 1.0e-8 * abs(target)


def test_dc_subdivision_cannot_accept_zero_progress_with_loose_tolerance() -> None:
    """A tiny retried target inside tolerance must not reset retries forever."""
    model, meta = _build_coupled_cantilever("eccentric", nx=2, ny=1)
    tip = meta["grid"][(meta["nx"], 0)]
    load = LoadCase("tip")
    load.add_nodal_load(tip, forces=(0.0, 0.0, -100.0))
    model.add_load_case(load)
    # Each nonconverged one-iteration attempt is rolled back. Eventually its
    # subdivided target lies inside the deliberately loose control tolerance
    # at the unchanged origin, which is not an advancing accepted interval.
    result = solve_static_nonlinear(
        model, load, control="displacement",
        displacement_control=DisplacementControl(
            node_id=tip, dof="uz", target_displacement=-.01),
        kinematics="corotational", num_steps=1, max_iterations=1,
        tolerance=.01, record_increment_snapshots=True,
        emit_restart_checkpoint=False,
    )
    assert result.status == "diverged"
    assert result.stop_reason == "minimum_control_step_reached"
    assert not result.steps and not result.snapshots
    assert result.info["displacement_control_adaptation"]["cutbacks"] <= 8
    np.testing.assert_array_equal(result.displacements, np.zeros_like(result.displacements))


@pytest.mark.parametrize("control_target", [0.0, -.001])
def test_dc_quadratic_search_reaches_small_descent_step_within_eight_trials(
    monkeypatch, control_target,
) -> None:
    """An exact, steep cubic residual needs alpha below the old 1/128 floor.

    Only this controller regression substitutes a synthetic conservative force
    and its exact tangent. Actual constraint maps and state transactions run
    through the public solve, with unchanged finite iteration/search limits.
    """
    import inspect
    from scipy import sparse
    import anyfesolver.nonlinear_static as nonlinear_static

    model, meta = _build_coupled_cantilever("eccentric", nx=2, ny=1)
    tip = meta["grid"][(meta["nx"], 0)]
    dofs = model.mesh.dof_manager.get_node_dofs(tip)
    load = LoadCase("tip")
    load.add_nodal_load(tip, forces=(0.0, 0.0, -100.0))
    model.add_load_case(load)
    real_assemble = nonlinear_static._assemble_nonlinear_system
    trial_alphas = []
    corrector_controls = []

    def cubic_force(*args, **kwargs):
        _force, _tangent, state = real_assemble(*args, **kwargs)
        u = np.asarray(args[1], dtype=float)
        force = u.copy()
        force[dofs[0]] += 1.0e12 * u[dofs[0]] ** 3 - 1.0
        diagonal = np.ones_like(u)
        diagonal[dofs[0]] += 3.0e12 * u[dofs[0]] ** 2
        frame = inspect.currentframe().f_back
        try:
            for _ in range(6):
                if frame is None or frame.f_code.co_name == "_solve_static_displacement_control":
                    break
                frame = frame.f_back
            if frame is not None and frame.f_code.co_name == "_solve_static_displacement_control":
                if not kwargs.get("tangent", True) and "alpha" in frame.f_locals:
                    trial_alphas.append(float(frame.f_locals["alpha"]))
                elif kwargs.get("tangent", True):
                    corrector_controls.append(float(u[dofs[2]]))
        finally:
            del frame
        return force, sparse.diags(diagonal, format="csr"), state

    monkeypatch.setattr(nonlinear_static, "_assemble_nonlinear_system", cubic_force)
    result = solve_static_nonlinear(
        model, load, control="displacement",
        displacement_control=DisplacementControl(
            node_id=tip, dof="uz", target_displacement=control_target),
        kinematics="corotational", num_steps=1, max_iterations=40,
        tolerance=1.0e-8, emit_restart_checkpoint=False,
        initial_element_states=_synthetic_postyield_controller_history(model),
        equilibrate_initial_state=False,
    )
    assert result.status == "completed", (result.status, result.stop_reason)
    if control_target == 0.0:
        assert trial_alphas and min(trial_alphas) < 1.0 / 128.0
    else:
        # A constraint-dominated first corrector retains its full control
        # predictor; interpolation must not turn it into microscopic accepted
        # intervals before the residual-dominated correction begins.
        assert len(result.steps) == 1
        assert len(corrector_controls) > 1
        assert abs(corrector_controls[1] - control_target) <= 1.0e-12
    assert abs(result.displacements[dofs[2]] - control_target) <= 1.0e-12
    assert result.steps[-1].residual_norm <= 1.0e-6


# ---------------------------------------------------------------------------
# Final repair: typed domain refusal, trial backtracking, classified failure
# ---------------------------------------------------------------------------


def test_dc_search_crosses_a_rising_constitutive_shoulder_within_eight_trials(monkeypatch):
    """A continuous piecewise response has a narrow descent region at a kink.

    This intentionally has no equilibrium root.  The contract is finding the
    existing local descent, followed by finite failure/rollback, not falsely
    declaring that every exhausted eight-trial search has no improving state.
    """
    import inspect
    from scipy import sparse
    import anyfesolver.nonlinear_static as nonlinear_static
    model, meta = _build_coupled_cantilever("eccentric", nx=2, ny=1)
    tip = meta["grid"][(meta["nx"], 0)]
    dofs = model.mesh.dof_manager.get_node_dofs(tip)
    load = LoadCase("tip")
    load.add_nodal_load(tip, forces=(0., 0., -100.))
    model.add_load_case(load)
    real_assemble = nonlinear_static._assemble_nonlinear_system
    corrections = []
    def shoulder_force(*args, **kwargs):
        _, _, state = real_assemble(*args, **kwargs)
        u = np.asarray(args[1])
        force = u.copy(); diagonal = np.ones_like(u)
        # Do not put a second parallel synthetic ux spring on each dependent
        # beam translation: it would mask the master response's shoulder.
        for node_id in model.mesh.nodes:
            if node_id >= 100:
                ux = model.mesh.dof_manager.get_node_dofs(node_id)[0]
                force[ux] = diagonal[ux] = 0.
        x = float(u[dofs[0]])
        force[dofs[0]] = -1+x if x <= 2e-6 else -1+2e-6-.1*(x-2e-6)
        diagonal[dofs[0]] = 1. if x <= 2e-6 else -.1
        frame = inspect.currentframe().f_back
        try:
            for _ in range(6):
                if frame is None or frame.f_code.co_name == "_solve_static_displacement_control": break
                frame = frame.f_back
            if frame is not None and kwargs.get("tangent", True):
                corrections.append(x)
        finally:
            del frame
        return force, sparse.diags(diagonal, format="csr"), state
    monkeypatch.setattr(nonlinear_static, "_assemble_nonlinear_system", shoulder_force)
    result = solve_static_nonlinear(
        model, load, control="displacement",
        displacement_control=DisplacementControl(node_id=tip, dof="uz", target_displacement=0.),
        kinematics="corotational", num_steps=1, max_iterations=3, tolerance=1e-8,
        emit_restart_checkpoint=False, record_increment_snapshots=True,
        initial_element_states=_synthetic_postyield_controller_history(model),
        equilibrate_initial_state=False)
    # Beyond the kink f=-1+2e-6-.1*(x-2e-6), so |f|<|f(0)|
    # exactly when 0<x<22e-6 (the turning point itself is at2e-6).
    assert len(corrections) >= 2 and 0. < corrections[1] < 22e-6, corrections
    assert result.status != "completed" and not result.steps and not result.snapshots
    np.testing.assert_array_equal(result.displacements, np.zeros_like(result.displacements))


def test_so3_mean_domain_refusal_is_typed() -> None:
    """The specific internal domain failure carries its own type.

    The displacement-control loop must classify exactly this refusal as
    trial inadmissibility; input-contract violations stay plain ValueError.
    """
    with pytest.raises(SO3MeanDomainError) as excinfo:
        _so3_weighted_mean(
            [np.array([np.pi, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])],
            [0.5, 0.5],
            "unit",
        )
    assert "principal rotation-vector chart" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _so3_weighted_mean(
            [np.array([0.1, 0.0, 0.0]), np.array([0.2, 0.0, 0.0])],
            [-0.5, 1.5],
            "unit",
        )
    assert not isinstance(excinfo.value, SO3MeanDomainError)


def test_dc_trial_domain_refusal_is_backtracked_not_aborted(monkeypatch) -> None:
    """A chart-inadmissible line-search trial must not abort the solve.

    Reproduces the interp full-target abort mechanism on a small fixture:
    the augmented step is inflated so the alpha=1 trial leaves the supported
    rotation chart.  The trial must be refused and backtracked, the commit
    must be refused, and the solve must return a classified failure with the
    committed state preserved, within the bounded subdivision budget.
    """
    import anyfesolver.nonlinear_static as nonlinear_static

    model, meta = _build_coupled_cantilever("interpolated", nx=2, ny=1)
    tip = meta["grid"][(meta["nx"], 0)]
    load = LoadCase("tip")
    load.add_nodal_load(tip, forces=(0.0, 0.0, -100.0))
    model.add_load_case(load)

    real_factorize = nonlinear_static.factorize
    factorize_calls = []

    class _OvershootingHandle:
        def __init__(self, inner):
            self._inner = inner

        def solve(self, rhs):
            return (
                np.asarray(self._inner.solve(rhs), dtype=float).reshape(-1)
                # Keep every one of the eight finite domain trials outside
                # the chart, even at the smallest permitted control interval.
                # A smaller inflation can now be recovered by interpolation.
                * 1.0e12
            )

    def overshooting_factorize(matrix, matrix_class, signature=None, **kwargs):
        factorize_calls.append(1)
        assert len(factorize_calls) <= 20, (
            "trial-domain refusal retried without bound"
        )
        inner = real_factorize(
            matrix, matrix_class, signature=signature, **kwargs
        )
        return _OvershootingHandle(inner)

    monkeypatch.setattr(nonlinear_static, "factorize", overshooting_factorize)
    result = solve_static_nonlinear(
        model, load, control="displacement",
        displacement_control=DisplacementControl(
            node_id=tip, dof="uz", target_displacement=-0.01
        ),
        kinematics="corotational", num_steps=1, max_iterations=5,
        tolerance=1.0e-8,
    )
    assert result.info["failure_reason"] == "trial_state_domain_refused"
    assert result.status != "completed"
    assert not result.steps and result.snapshots == ()
    np.testing.assert_array_equal(
        result.displacements, np.zeros_like(result.displacements)
    )
    adaptation = result.info["displacement_control_adaptation"]
    assert adaptation["cutbacks"] <= 8


@pytest.mark.parametrize("plastic_origin", ("elastic", "committed", "trial"))
def test_dc_merit_search_requires_actual_trial_or_committed_plasticity(monkeypatch, plastic_origin):
    import inspect
    import anyfesolver.nonlinear_static as ns
    model, meta = _build_coupled_cantilever("eccentric", nx=2, ny=1)
    tip = meta["grid"][(meta["nx"], 0)]
    load = LoadCase("tip"); load.add_nodal_load(tip, forces=(0., 0., -100.))
    model.add_load_case(load)
    real = ns._assemble_nonlinear_system
    merit_trials = []
    def observe(*args, **kwargs):
        force, tangent, state = real(*args, **kwargs)
        if plastic_origin == "trial":
            state = copy.deepcopy(dict(state))
            state[1]["alpha"][:] = 1e-4
        frame = inspect.currentframe().f_back
        try:
            for _ in range(6):
                if frame is None or frame.f_code.co_name == "_solve_static_displacement_control": break
                frame = frame.f_back
            if (frame is not None and not kwargs.get("tangent", True)
                    and frame.f_locals.get("u_trial") is args[1]):
                merit_trials.append(1)
        finally:
            del frame
        return force, tangent, state
    monkeypatch.setattr(ns, "_assemble_nonlinear_system", observe)
    initial = _synthetic_postyield_controller_history(model) if plastic_origin == "committed" else {}
    before = copy.deepcopy(initial)
    result = solve_static_nonlinear(model, load, control="displacement",
        displacement_control=DisplacementControl(node_id=tip, dof="uz", target_displacement=-1e-5),
        kinematics="corotational", num_steps=1, max_iterations=10, tolerance=1e-8,
        initial_element_states=initial, equilibrate_initial_state=False, emit_restart_checkpoint=False)
    assert result.status == "completed", (result.status, result.stop_reason)
    assert bool(merit_trials) == (plastic_origin != "elastic")
    for eid in initial:
        for field in initial[eid]:
            if isinstance(initial[eid][field], np.ndarray):
                np.testing.assert_array_equal(initial[eid][field], before[eid][field])


@pytest.mark.parametrize("length", (1.0 / 3.0, 0.5, 1.3))
def test_fiber_beam_unyielded_limit_matches_full_elastic_timoshenko_matrix(length):
    from test_nonlinear_beam_shell_slice3_continuation import _add_ep_material, CS
    model = FEModel("fiber_elastic_limit")
    _add_ep_material(model, 1.0)
    elastic = model.add_material("elastic", 2.1e11, 0.3)
    model.add_node(1, 0., 0., 0.)
    model.add_node(2, length, 0., 0.)
    element = BeamElement(1, [1, 2], "steel", dict(CS))
    model.add_element(1, element)
    plastic = model.get_material("steel")
    config = element._fiber_plasticity_config(plastic)
    y, z, weights = element._fiber_section_grid(config)
    np.testing.assert_allclose([sum(weights), weights@y, weights@z,
                               weights@(y*y), weights@(z*z), weights@(y*z)],
                              [CS["area"], 0., 0., CS["Iz"], CS["Iy"], 0.],
                              rtol=1e-14, atol=1e-21)
    for u in (np.zeros(12), np.array([1., -2., 3., 2., 1., -1.,
                                     2., 3., -1., -1., 2., 3.])*1e-6):
        before = u.copy()
        fp, kp, state = element.compute_nonlinear_response(model.mesh, plastic, u)
        fe, ke, _ = element.compute_nonlinear_response(model.mesh, elastic, u)
        np.testing.assert_array_equal(u, before)
        assert np.max(state["alpha"]) == 0.
        assert np.linalg.norm(fp-fe) <= 1e-12*max(np.linalg.norm(fe), 1.)
        assert np.linalg.norm(kp-ke) <= 1e-12*np.linalg.norm(ke)
    # A separate energy expression checks both bending/shear planes, axial
    # geometric terms and torsion, rather than reproducing force assembly.
    _, transform = element._beam_frame_and_transform(element.get_node_coordinates(model.mesh))
    E, G = 2.1e11, 2.1e11/(2*(1+.3))
    def energy(u):
        v = transform@u
        eps = (v[6]-v[0])/length + ((v[7]-v[1])**2+(v[8]-v[2])**2)/(2*length**2)
        value = .5*E*CS["area"]*length*eps**2 + .5*G*CS["J"]*(v[9]-v[3])**2/length
        for a, b, ra, rb, inertia, sign, kappa in (
            (1, 7, 5, 11, CS["Iz"], -1., element._ky),
            (2, 8, 4, 10, CS["Iy"], 1., element._kz),
        ):
            shear = (v[b]-v[a])/length + sign*(v[ra]+v[rb])/2
            effective = 1/(1/(G*CS["area"]*kappa)+length**2/(12*E*inertia))
            value += .5*E*inertia*(v[rb]-v[ra])**2/length + .5*effective*length*shear**2
        return value
    u = np.linspace(-2e-5, 3e-5, 12)
    f, k, state = element.compute_nonlinear_response(model.mesh, plastic, u)
    state_before = copy.deepcopy(state)
    numeric_f, numeric_k = np.empty(12), np.empty((12, 12))
    for j in range(12):
        perturbation = np.eye(12)[j]*1e-7
        plus, minus = u+perturbation, u-perturbation
        numeric_f[j] = (energy(plus)-energy(minus))/(2e-7)
        numeric_k[:, j] = (
            element.compute_nonlinear_response(model.mesh, plastic, plus, state)[0]
            - element.compute_nonlinear_response(model.mesh, plastic, minus, state)[0])/(2e-7)
    for key in state:
        if isinstance(state[key], np.ndarray):
            np.testing.assert_array_equal(state[key], state_before[key])
    assert np.linalg.norm(f-numeric_f)/np.linalg.norm(f) <= 1e-7
    assert np.linalg.norm(k-numeric_k)/np.linalg.norm(k) <= 1e-7


def test_fiber_beam_elastic_limit_retains_objective_rigid_motion():
    from test_nonlinear_beam_shell_slice3_continuation import _add_ep_material, CS
    import anyfesolver.corotational as cr
    model = FEModel("fiber_rigid_motion")
    _add_ep_material(model, 1.)
    model.add_node(1, 0., 0., 0.)
    model.add_node(2, .5, 0., 0.)
    element = BeamElement(1, [1, 2], "steel", dict(CS))
    model.add_element(1, element)
    theta = np.array([.12, -.08, .17])
    rotation = rotation_matrix_from_vector(theta)
    coords = element.get_node_coordinates(model.mesh)
    u = np.zeros((2, 6)); u[:, :3] = coords@rotation.T-coords + [.1, -.2, .3]
    u[:, 3:] = theta
    before = u.copy()
    force, _, state = cr.corotational_work_conjugate_response(model, 1, element, u.ravel())
    np.testing.assert_array_equal(u, before)
    assert np.linalg.norm(force) <= 1e-6
    assert np.max(state["alpha"]) == 0.


@pytest.mark.parametrize("boundary", ("absolute", "relative"))
def test_rotating_geometric_tangent_stencil_stays_in_admitted_domain(boundary):
    from test_nonlinear_beam_shell_coupling import _interp_map_with_distinct_master_rotations

    _, _, _, _, _, mapping, linear_u = _interp_map_with_distinct_master_rotations()
    for entry in mapping.entries:
        for i, (_, theta_dofs, _) in enumerate(entry.masters):
            angle = np.pi - 1.5e-6 if boundary == "absolute" else (np.pi - 1.5e-6) / 2
            linear_u[theta_dofs] = [angle * (-1 if boundary == "relative" and i else 1), 0, 0]
    u = mapping.apply_kinematics(linear_u)
    imbalance = np.zeros(mapping.total_dofs)
    for entry in mapping.entries:
        imbalance[entry.slave_u_dofs] = [30.0, -20.0, 10.0]
        imbalance[entry.slave_theta_dofs] = [2.0, -3.0, 4.0]
    tangent = mapping.geometric_tangent(u, imbalance).toarray()
    assert np.any(tangent) and np.all(np.isfinite(tangent))
    np.testing.assert_allclose(tangent, tangent.T, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("control_mode", ("force", "force_plain", "displacement"))
@pytest.mark.parametrize("refusal", ("once", "all", "unrelated"))
def test_trial_domain_recovery_and_unrelated_error_propagation(monkeypatch, control_mode, refusal):
    from anyfesolver.corotational import NonlinearRotatingMPCMap

    model, meta = _build_coupled_cantilever("interpolated", nx=2, ny=1)
    tip = meta["grid"][(meta["nx"], 0)]
    load = LoadCase("tip")
    load.add_nodal_load(tip, forces=(0.0, 0.0, -100.0))
    model.add_load_case(load)
    real_apply = NonlinearRotatingMPCMap.apply_kinematics
    refusals = []

    def refuse_trial(self, linear_u):
        if np.linalg.norm(linear_u) > 1.0e-14 and (refusal != "once" or not refusals):
            refusals.append(1)
            assert len(refusals) <= 150, "trial rejection did not remain bounded"
            if refusal == "unrelated":
                raise ValueError("sentinel_unrelated_error")
            raise SO3MeanDomainError("synthetic trial domain refusal")
        return real_apply(self, linear_u)

    monkeypatch.setattr(NonlinearRotatingMPCMap, "apply_kinematics", refuse_trial)
    u_input = np.zeros(model.mesh.dof_manager.total_dofs)
    states_input = {}
    settings = dict(
        kinematics="corotational", num_steps=1, max_iterations=5, tolerance=1.0e-8,
        initial_displacements=u_input, initial_element_states=states_input,
        equilibrate_initial_state=False, record_increment_snapshots=True,
        convergence_settings={"line_search": "always", "max_line_search_cuts": 3,
                              "min_step_fraction": 0.25},
    )
    if control_mode == "force_plain":
        settings["convergence_settings"]["line_search"] = "never"
    if control_mode == "displacement":
        settings.update(control="displacement", displacement_control=DisplacementControl(
            node_id=tip, dof="uz", target_displacement=-1.0e-4))
    if refusal == "unrelated":
        with pytest.raises(ValueError, match="sentinel_unrelated_error"):
            solve_static_nonlinear(model, load, **settings)
    else:
        result = solve_static_nonlinear(model, load, **settings)
        assert refusals
        if refusal == "once":
            assert result.status == "completed", (result.status, result.stop_reason)
        else:
            assert result.status != "completed"
            if control_mode.startswith("force"):
                assert result.failure_reason == "minimum_load_increment_reached"
                assert result.info["first_failed_iteration_reason"] == "trial_state_domain_refused"
            else:
                assert result.failure_reason == "trial_state_domain_refused"
            assert not result.steps and not result.snapshots
            np.testing.assert_array_equal(result.displacements, u_input)
    np.testing.assert_array_equal(u_input, np.zeros_like(u_input))
    assert states_input == {}


def test_slice3_portable_effective_runtime_settings():
    import os
    if "ANYSOLVER_DEVELOPMENT_SOURCE" not in os.environ:
        pytest.skip("effective portable settings are checked by the development runner")
    import numba
    from threadpoolctl import threadpool_info

    assert os.environ.get("ANYSOLVER_RUN_RETIRED_PROGRAMME_TESTS") == "0"
    assert numba.config.DISABLE_JIT and numba.get_num_threads() == 1
    assert all(pool["num_threads"] == 1 for pool in threadpool_info())


def test_dc_line_search_no_improvement_is_classified_and_rolls_back(
    monkeypatch,
) -> None:
    """No improving step found in the bounded postyield trial set fails
    the attempt with a classified reason instead of committing a full step
    that sustains a Newton limit cycle; the committed state is preserved.
    """
    from anyfesolver.corotational import NonlinearRotatingMPCMap

    model, meta = _build_coupled_cantilever("eccentric", nx=2, ny=1)
    tip = meta["grid"][(meta["nx"], 0)]
    load = LoadCase("tip")
    load.add_nodal_load(tip, forces=(0.0, 0.0, -100.0))
    model.add_load_case(load)

    real_reduce = NonlinearRotatingMPCMap.reduce_force
    fixed = {}

    def irreducible_residual(self, u, force):
        # A fixed irreducible reduced residual: once the control row is
        # satisfied no line-search trial can decrease the augmented merit,
        # which previously fell back to the full Newton step forever (the
        # measured period-3 limit cycle mechanism).
        reduced = real_reduce(self, u, force)
        if "v" not in fixed:
            fixed["v"] = np.full_like(
                np.asarray(reduced, dtype=float).reshape(-1), 1.0e6
            )
        return fixed["v"]

    monkeypatch.setattr(
        NonlinearRotatingMPCMap, "reduce_force", irreducible_residual
    )
    result = solve_static_nonlinear(
        model, load, control="displacement",
        displacement_control=DisplacementControl(
            node_id=tip, dof="uz", target_displacement=-0.01
        ),
        kinematics="corotational", num_steps=1, max_iterations=5,
        tolerance=1.0e-8,
        initial_element_states=_synthetic_postyield_controller_history(model),
        equilibrate_initial_state=False,
    )
    assert result.info["failure_reason"] == "line_search_no_improvement"
    assert result.status != "completed"
    assert not result.steps and result.snapshots == ()
    np.testing.assert_array_equal(
        result.displacements, np.zeros_like(result.displacements)
    )
    adaptation = result.info["displacement_control_adaptation"]
    assert adaptation["cutbacks"] <= 8


@pytest.mark.parametrize("refusal", ["none", "domain", "unrepresentable", "unrelated"])
def test_material_event_localization_is_bounded_pure_and_fails_closed(refusal):
    from anyfesolver._nonlinear_events import MaterialEvents, crossed_material_event
    event = 6.4e-8
    q = np.array([1.0e20 if refusal == "unrepresentable" else .1])
    direction = np.ones(1)
    before = q.copy()
    base = MaterialEvents(np.array([1.]), np.array([.001]))
    upper = MaterialEvents(np.array([1.-1e-7/event]), np.array([.001+1e-7]))
    calls = []
    def evaluate(alpha):
        calls.append(alpha)
        if refusal == "domain":
            return None
        if refusal == "unrelated":
            raise RuntimeError("unrelated callback failure")
        return {"events": MaterialEvents(np.array([1.-alpha/event]), np.array([.001+alpha])),
                "merit": 1.-alpha}
    if refusal == "unrelated":
        with pytest.raises(RuntimeError, match="unrelated callback failure"):
            crossed_material_event(base, upper, 1e-7, q, direction, evaluate)
    else:
        result = crossed_material_event(base, upper, 1e-7, q, direction, evaluate)
        if refusal == "none":
            assert event < result["alpha"] < 1e-7
            assert not result["events"].signature[0]
            assert not np.array_equal(q + result["alpha"] * direction, q)
        else:
            assert result is None
    assert len(calls) <= 50
    np.testing.assert_array_equal(q, before)
    assert crossed_material_event(base, base, 1e-7, q, direction, evaluate) is None


def test_material_event_indicators_match_actual_fiber_entry_and_copy_inputs():
    from anyfesolver._nonlinear_events import material_events
    from test_nonlinear_beam_shell_slice3_continuation import _build_ep_model
    model, *_ = _build_ep_model("direct")
    element = next(e for e in model.mesh.elements.values() if isinstance(e, BeamElement))
    material = model.get_material(element.material_name)
    curve = material.hardening_curve
    E = material.elastic_modulus
    n = 25
    old = {"plastic_strain": np.zeros(n), "alpha": np.zeros(n)}
    flow = float(curve.flow_stress(np.array([0.]))[0])
    strain = np.full(n, (flow * (1.+1e-9) - 1.) / E)
    strain[1] = (flow * (1.+1e-9) + 1.) / E
    entry = (flow + 1e-9 * max(flow, 1.)) / E
    adjacent = entry
    for j in range(2, n):
        adjacent = np.nextafter(adjacent, np.inf if j % 2 else -np.inf)
        strain[j] = adjacent if j % 3 else -adjacent
    stress, tangent, plastic, alpha = element._uniaxial_return_map(strain, old, E, curve)
    trial = {"fiber_strain": strain, "plastic_strain": plastic, "alpha": alpha}
    before = copy.deepcopy((old, trial))
    events = material_events(model, {element.element_id: old}, {element.element_id: trial})
    np.testing.assert_array_equal(events.signature[:n], alpha > old["alpha"])
    assert not events.signature[0] and events.signature[1]
    for actual, expected in zip((old, trial), before):
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
    boundary = curve.eps_p_y1
    hardening = copy.deepcopy(trial)
    hardening["alpha"][:3] = [np.nextafter(boundary, -np.inf), boundary,
                              np.nextafter(boundary, np.inf)]
    segmented = material_events(model, {element.element_id: old}, {element.element_id: hardening})
    np.testing.assert_array_equal(segmented.signature[n:n+3], [False, False, True])
    assert material_events(model, {}, {element.element_id: trial}) is None


@pytest.mark.parametrize("cancel", [False, True])
def test_dc_resolved_material_event_crossing_preserves_rollback_and_cancellation(monkeypatch, cancel):
    """A narrow resolved event is crossed, but no non-equilibrium state commits."""
    import inspect
    from scipy import sparse
    import anyfesolver.nonlinear_static as nonlinear_static
    import anyfesolver._nonlinear_events as events
    from anyfesolver.control import CancellationToken, SolveCancelled
    model, meta = _build_coupled_cantilever("eccentric", nx=2, ny=1)
    tip = meta["grid"][(meta["nx"], 0)]
    dofs = model.mesh.dof_manager.get_node_dofs(tip)
    load = LoadCase("tip")
    load.add_nodal_load(tip, forces=(0., 0., -100.))
    model.add_load_case(load)
    real_assemble = nonlinear_static._assemble_nonlinear_system
    token = CancellationToken()
    corrections, observations = [], []
    current = [0.]
    kink = 6.4e-8
    def force(*args, **kwargs):
        _, _, state = real_assemble(*args, **kwargs)
        u = np.asarray(args[1]); f = u.copy(); diagonal = np.ones_like(u)
        for node_id in model.mesh.nodes:
            if node_id >= 100:
                ux = model.mesh.dof_manager.get_node_dofs(node_id)[0]
                f[ux] = diagonal[ux] = 0.
        x = current[0] = float(u[dofs[0]])
        f[dofs[0]] = -1+x if x <= kink else -1+kink-10.*(x-kink)
        diagonal[dofs[0]] = 1. if x <= kink else -10.
        if kwargs.get("tangent", True):
            corrections.append(x)
        return f, sparse.diags(diagonal, format="csr"), state
    def indicators(*args):
        observations.append(current[0])
        if cancel and len(observations) == 3:
            token.cancel("during material event localization")
        return events.MaterialEvents(np.array([1.-current[0]/kink]), np.array([current[0]]))
    monkeypatch.setattr(nonlinear_static, "_assemble_nonlinear_system", force)
    monkeypatch.setattr(events, "material_events", indicators)
    initial = _synthetic_postyield_controller_history(model)
    before = copy.deepcopy(initial)
    def solve():
        return solve_static_nonlinear(
            model, load, control="displacement",
            displacement_control=DisplacementControl(node_id=tip, dof="uz", target_displacement=0.),
            kinematics="corotational", num_steps=1, max_iterations=3, tolerance=1e-8,
            emit_restart_checkpoint=False, record_increment_snapshots=True,
            initial_element_states=initial, equilibrate_initial_state=False, cancellation_token=token)
    if cancel:
        with pytest.raises(SolveCancelled, match="material event"):
            solve()
    else:
        result = solve()
        assert len(corrections) >= 2 and kink < corrections[1] < 1e-7, corrections
        assert result.info["displacement_control_adaptation"]["material_event_steps"] >= 1
        assert result.status != "completed" and not result.steps and not result.snapshots
        np.testing.assert_array_equal(result.displacements, np.zeros_like(result.displacements))
    _assert_element_states_equal(initial, before)


def _return_band_fixture(kind, alpha_old):
    from anyfesolver.material_curves import LinearHardeningCurve, PiecewiseLinearCurve
    scale = 1.25
    if kind == "constant":
        curve = LinearHardeningCurve(355e6/scale, 0.)
        flow, hardening = 355e6, 0.
    else:
        curve = PiecewiseLinearCurve((0., .002, .004),
                                     (300e6/scale, 355e6/scale, 390e6/scale))
        if alpha_old < .002:
            flow, hardening = 300e6+27.5e9*alpha_old, 27.5e9
        elif alpha_old < .004:
            flow, hardening = 355e6+17.5e9*(alpha_old-.002), 17.5e9
        else:
            flow, hardening = 390e6, 0.
    state = {"plastic_strain": np.array([.0004]), "alpha": np.array([alpha_old])}
    return curve, scale, flow, hardening, state


def _immutable_return_evaluation(strain, state, curve, scale):
    value = np.array([strain]); value_before = value.copy()
    history = copy.deepcopy(state); history_before = copy.deepcopy(history)
    result = BeamElement._uniaxial_return_map(value, history, 210e9, curve, scale)
    np.testing.assert_array_equal(value, value_before)
    for name in history:
        np.testing.assert_array_equal(history[name], history_before[name])
    assert all(np.asarray(field).shape == (1,) and np.all(np.isfinite(field)) for field in result)
    return tuple(float(field[0]) for field in result)


@pytest.mark.parametrize("sign", (-1., 1.))
@pytest.mark.parametrize("kind,alpha_old", (("constant", .003), ("piecewise", .001),
    ("piecewise", .002), ("piecewise", .003), ("piecewise", .004)))
def test_beam_return_map_positive_update_matches_exact_law_inside_closure_band(sign, kind, alpha_old):
    E = 210e9
    curve, scale, flow, hardening, state = _return_band_fixture(kind, alpha_old)
    before = copy.deepcopy(state)
    strain = state["plastic_strain"][0] + sign*(flow+1.)/E
    trial = E*(strain-state["plastic_strain"][0])
    overstress = abs(trial)-flow
    assert 1e-9*flow < overstress < 1e-8*flow
    gamma = overstress/(E+hardening)
    stress, tangent, plastic, alpha = _immutable_return_evaluation(strain, state, curve, scale)
    assert sign*(plastic-state["plastic_strain"][0]) > 0. and alpha > alpha_old
    # Exact affine return, including fixed nonzero history and flow scaling.
    expected_stress = sign*(flow+hardening*gamma)
    assert abs(stress-expected_stress) <= 8*np.spacing(abs(expected_stress))
    assert abs(plastic-(state["plastic_strain"][0]+sign*gamma)) <= 8*np.spacing(abs(plastic))
    assert abs(alpha-(alpha_old+gamma)) <= 8*np.spacing(alpha)
    np.testing.assert_allclose(tangent, E*hardening/(E+hardening), rtol=1e-14, atol=0.)
    h = .5/E
    plus = _immutable_return_evaluation(strain+h, state, curve, scale)[0]
    minus = _immutable_return_evaluation(strain-h, state, curve, scale)[0]
    for shifted in (strain+h, strain-h):
        margin = abs(E*(shifted-state["plastic_strain"][0]))-flow
        assert 1e-9*flow < margin < 1e-8*flow
    # E provides a nonzero constitutive scale also for perfect plasticity.
    # The sub-Pa perturbation stays within the original problematic branch.
    assert abs((plus-minus)/(2*h)-tangent)/E <= 1e-7
    for name in state:
        np.testing.assert_array_equal(state[name], before[name])


@pytest.mark.parametrize("kind", ("constant", "piecewise"))
@pytest.mark.parametrize("sign", (-1., 1.))
def test_beam_return_map_keeps_yield_entry_and_closure_band_boundaries(kind, sign):
    E = 210e9
    curve, scale, flow, hardening, state = _return_band_fixture(kind, .003)
    for factor in (.5, 1.25, 7.5, 12.5):
        strain = state["plastic_strain"][0]+sign*(flow+factor*1e-9*flow)/E
        trial = E*(strain-state["plastic_strain"][0])
        stress, tangent, plastic, alpha = _immutable_return_evaluation(strain, state, curve, scale)
        margin = abs(trial)-flow
        if factor < 1.:
            assert margin < 1e-9*flow
            assert stress == trial and tangent == E
            assert plastic == state["plastic_strain"][0] and alpha == state["alpha"][0]
        else:
            assert margin > 1e-9*flow
            gamma = margin/(E+hardening)
            expected = sign*(flow+hardening*gamma)
            assert abs(stress-expected) <= 8*np.spacing(abs(expected))
            assert sign*(plastic-state["plastic_strain"][0]) > 0. and alpha > state["alpha"][0]
            np.testing.assert_allclose(tangent, E*hardening/(E+hardening), rtol=1e-14, atol=0.)


def _secant_seed_cubic(maximum_solves=20, *, evaluate_override=None, solve_override=None):
    from anyfesolver._nonlinear_seed import secant_plane_seed
    q = np.array([1.]); previous = (np.array([0.]), 0.)
    calls = []
    def evaluate(values, lam, tangent):
        if evaluate_override is not None:
            return evaluate_override(values, lam, tangent)
        return dict(residual=np.array([values[0]**3-lam]), reference=1.,
                    tangent=np.array([[3*values[0]**2]]), load_column=np.array([-1.]))
    def solve(A, rhs):
        calls.append((A.copy(), rhs.copy()))
        return np.linalg.solve(A, rhs) if solve_override is None else solve_override(A, rhs)
    result, info = secant_plane_seed(q, 1., previous, evaluate, solve,
        maximum_solves=maximum_solves, tolerance=1e-8, force_scale=1.,
        domain_error=SO3MeanDomainError)
    np.testing.assert_array_equal(q, [1.])
    np.testing.assert_array_equal(previous[0], [0.])
    assert previous[1] == 0.
    assert info['solves'] == len(calls) <= maximum_solves
    return result, info, calls


def test_uncommitted_secant_plane_seed_solves_actual_force_and_fixed_plane():
    result, info, calls = _secant_seed_cubic()
    assert info['status'] == 'equilibrated_uncommitted_seed'
    # Independent analytic equations: q^3=lambda and q+lambda=4.
    assert abs(result[0]**3-result[1]) <= 1e-8
    assert abs(result.sum()-4.) <= 1e-8
    assert np.linalg.norm(result-[1., 1.]) > .1
    assert 0 < len(calls) < 20
    # This is a general border, not a symmetric [K,-F;-F^T,0] solve.
    assert not np.allclose(calls[0][0], calls[0][0].T)


def test_uncommitted_secant_plane_seed_budget_failure_restores_origin():
    result, info, _ = _secant_seed_cubic(maximum_solves=1)
    assert info['status'] == 'seed_solve_budget' and info['solves'] == 1
    np.testing.assert_array_equal(result, [1., 1.])


@pytest.mark.parametrize('mode', ('domain', 'nonfinite_solve', 'no_merit_descent'))
def test_uncommitted_secant_plane_seed_refusal_is_bounded_and_unaccepted(mode):
    def refused(*args): raise SO3MeanDomainError('synthetic seed chart')
    result, info, calls = _secant_seed_cubic(
        evaluate_override=refused if mode == 'domain' else None,
        solve_override=(lambda A, r: np.full_like(r, np.nan)) if mode == 'nonfinite_solve'
                       else ((lambda A, r: np.zeros_like(r)) if mode == 'no_merit_descent' else None))
    assert info['status'] == {'domain': 'seed_domain_refused',
        'nonfinite_solve': 'seed_linear_solve_failed', 'no_merit_descent': 'seed_merit_no_improvement'}[mode]
    assert len(calls) <= 1
    np.testing.assert_array_equal(result, [1., 1.])


@pytest.mark.parametrize('exception', (ValueError('unrelated assembly error'),
    __import__('anyfesolver.control', fromlist=['SolveCancelled']).SolveCancelled('cancel seed')))
def test_uncommitted_secant_plane_seed_propagates_unrelated_error_and_cancellation(exception):
    def fail(*args): raise exception
    with pytest.raises(type(exception), match=str(exception)):
        _secant_seed_cubic(evaluate_override=fail)


@pytest.mark.parametrize('retained_rows', (True, False))
def test_restart_reduction_inverts_affine_constraints_and_refuses_incompatible_state(retained_rows):
    from scipy import sparse
    from anyfesolver.nonlinear_static import _reduced_coordinates

    T = sparse.csr_matrix([[1., 0.], [0., 1.], [.75, .25]] if retained_rows
                          else [[2., 0.], [0., 3.], [.75, .25]])
    q = np.array([.125, -.0625])
    u0 = np.array([.03125, 0., -.015625])
    supplied = T @ q + u0
    before, affine_before = supplied.copy(), u0.copy()
    recovered = _reduced_coordinates(T, u0, supplied)
    np.testing.assert_allclose(T @ recovered + u0, supplied, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(supplied, before)
    np.testing.assert_array_equal(u0, affine_before)
    incompatible = supplied.copy()
    incompatible[-1] += 1e-4
    incompatible_before = incompatible.copy()
    with pytest.raises(ValueError, match='incompatible with the active supports/MPC'):
        _reduced_coordinates(T, u0, incompatible)
    np.testing.assert_array_equal(incompatible, incompatible_before)
