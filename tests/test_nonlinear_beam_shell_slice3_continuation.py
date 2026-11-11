"""Slice 3 continuation regressions: EP edge coupling on the restart route.

Product-level acceptance for the adopted Slice 3 restart contract
(``.development/nonlinear-master-slave-coupling-20261017``):

* EP isolation geometry (L=1, W=0.24, t=0.012, shell 2x1, stiffener 10x40 mm
  at e=+0.04, S355 canonical curve, fiber plasticity) with the two original
  edge couplings: direct (beam nodes on shell edge nodes) and interpolated
  (genuinely interpolated mid-span beam nodes).
* Revised displacement endpoint uz=-11/60 m at the original tip probe, 11
  original -1/60 m scheduled increments (explicit 2026-10-19 amendment),
  max 40 iterations, tolerance 1e-8: completion, control error, positive
  plasticity, independent residual at the achieved load factor, constraint
  postcheck, accepted-increment snapshots and force/wrench balance.
* Restart via the public initial-state route only: 6 steps to -0.10,
  resume 5 steps to -11/60 with ordinary deep-copied initial displacements
  and element states, ``equilibrate_initial_state=False`` and no serialized
  checkpoint.  The resumed run must reproduce the uninterrupted physical
  state, load factor and plastic-strain history.
* Caller restart inputs are preserved on success, rejection and
  cancellation; cancellation fires through accepted progress events and the
  same inputs must still deliver the fresh resume afterwards.

Expensive immutable solves are cached at module level so several assertions
inspect the same result; the final regression starts a fresh process
naturally.

The historical -0.20 m / 12-increment tests and failed evidence are preserved
in the task evidence. This amended endpoint does not relabel those stopped
results as completed or establish the remaining analytical gates.
"""

from __future__ import annotations

import copy
import json
import pickle
from collections.abc import Mapping

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
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver.constraint_audit import constraint_residual_summary
from anyfesolver.corotational import build_nonlinear_rotating_mpc_map
from anyfesolver.elements import BeamElement, CoupledBeamShellElement
from anyfesolver.matrix_assembly import (
    assemble_load_vector,
    assemble_stiffness_matrix,
)
from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement
from anyfesolver.nonlinear_static import _assemble_nonlinear_system

L = 1.0
W = 0.24
T = 0.012
NX = 2
NY = 1
R = np.array([0.0, 0.0, 0.04])
CS = {
    "area": 4.0e-4,
    "Iy": 0.01 * 0.04**3 / 12.0,
    "Iz": 0.04 * 0.01**3 / 12.0,
    "J": 1.0e-8,
    "b": 0.01,
    "h": 0.04,
    "fiber_plasticity": True,
}
TIP_LOAD_PER_NODE = -2000.0
DC_TARGET = -11.0 / 60.0
# Displacement-seed fixture (re-seeded 2026-10-22).  The secant-plane seed
# only runs when the accepted load factor magnitude decreases along an
# unchanged control direction.  The -0.10 -> -11/60 resume used to soften only
# because the qualified Q4 plastic correction made the shells unstable; with
# the plastic-deficit fix that path hardens monotonically and never seeds.
# Elastic unloading -0.10 -> -0.07 decreases |lambda| on every increment and
# exercises the seed on both the historical and the fixed formulation.
DC_SEED_TARGET = -0.07
DC_STEPS = 11
CHECKPOINT_STEPS = 6
RESUME_STEPS = 5
DC_MAX_ITER = 40
DC_TOL = 1.0e-8
FC_PARAMS = {"num_steps": 6, "max_iterations": 30}

_CACHE: dict = {}


def _cached(key, factory):
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def _add_ep_material(model: FEModel, scale: float) -> None:
    from anyfesolver.material_curves import (
        DNVC208MaterialCurve,
        dnv_c208_steel_curve,
    )

    base = dnv_c208_steel_curve("S355", T).as_dict()
    hardening = DNVC208MaterialCurve(
        base["sigma_prop"] * scale,
        base["sigma_yield"] * scale,
        base["sigma_yield_2"] * scale,
        base["eps_p_y1"],
        base["eps_p_y2"],
        base["K"] * scale,
        base["n"],
    )
    model.add_material(
        "steel",
        2.1e11,
        0.3,
        density=7850.0,
        yield_stress=base["sigma_yield"] * scale,
        hardening_curve=hardening,
    )


def _build_ep_model(coupling: str, scale: float = 1.0):
    """EP isolation geometry; coupling in {'direct', 'interp'}.

    Node identities are the deterministic construction IDs (no coordinate
    search): shell grid nodes 1..(NX+1)*(NY+1), beam nodes 100+ib, MPC
    elements 200+ib, beam elements 300+ib.
    """
    if coupling not in ("direct", "interp"):
        raise ValueError(f"unknown coupling {coupling!r}")
    model = FEModel(f"ep_edge_{coupling}")
    _add_ep_material(model, scale)
    grid: dict[tuple[int, int], int] = {}
    node_id = 1
    for ix, x in enumerate(np.linspace(0.0, L, NX + 1)):
        for iy, y in enumerate(np.linspace(-W / 2.0, W / 2.0, NY + 1)):
            model.add_node(node_id, float(x), float(y), 0.0)
            grid[(ix, iy)] = node_id
            node_id += 1
    element_id = 1
    for ix in range(NX):
        for iy in range(NY):
            model.add_element(
                element_id,
                create_shell_element(
                    element_id,
                    [
                        grid[(ix, iy)],
                        grid[(ix + 1, iy)],
                        grid[(ix + 1, iy + 1)],
                        grid[(ix, iy + 1)],
                    ],
                    "steel",
                    thickness=T,
                ),
            )
            element_id += 1
    n_beam = 2 if coupling == "direct" else 3
    beam_nodes: list[int] = []
    for ib, xb in enumerate(np.linspace(0.0, L, n_beam + 1)):
        beam_node = 100 + ib
        model.add_node(beam_node, float(xb), -W / 2.0, float(R[2]))
        beam_nodes.append(beam_node)
        sx = xb / (L / NX)
        ix = min(int(np.floor(sx)), NX - 1)
        xi = sx - ix
        masters = [(grid[(ix, 0)], 1.0 - xi), (grid[(ix + 1, 0)], xi)]
        masters = [(n, w) for n, w in masters if w > 1.0e-12]
        mpc_id = 200 + ib
        if len(masters) == 1:
            model.add_element(
                mpc_id,
                CoupledBeamShellElement(
                    mpc_id, beam_node, masters[0][0], "steel", eccentricity=R
                ),
            )
        else:
            weights = np.array([w for _, w in masters], dtype=float)
            model.add_element(
                mpc_id,
                InterpolatedBeamShellMPCElement(
                    mpc_id, beam_node, [n for n, _ in masters], weights,
                    eccentricity=R,
                ),
            )
    for ib in range(n_beam):
        model.add_element(
            300 + ib,
            BeamElement(300 + ib, [beam_nodes[ib], beam_nodes[ib + 1]],
                        "steel", dict(CS)),
        )
    model.add_boundary_condition(
        BoundaryCondition(
            "root",
            [grid[(0, iy)] for iy in range(NY + 1)],
            {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
        )
    )
    model.apply_boundary_conditions()
    load = LoadCase("tip")
    probe = grid[(NX, 0)]
    return model, load, probe, grid


def _add_tip_load(model: FEModel, load: LoadCase, grid: dict) -> None:
    for iy in range(NY + 1):
        load.add_nodal_load(
            grid[(NX, iy)],
            [0.0, 0.0, TIP_LOAD_PER_NODE, 0.0, 0.0, 0.0],
        )
    model.add_load_case(load)


def _alpha_max(states) -> float:
    values = [
        float(np.max(np.abs(state.get("alpha", 0.0))))
        for state in dict(states).values()
        if isinstance(state, dict)
    ]
    return max(values, default=0.0)


def _probe_uz(case: dict) -> float:
    dofs = case["model"].mesh.dof_manager.get_node_dofs(case["probe"])
    return float(np.asarray(case["result"].displacements)[dofs[2]])


def _dc_case(coupling: str, target: float, num_steps: int, **extra) -> dict:
    model, load, probe, grid = _build_ep_model(coupling)
    _add_tip_load(model, load, grid)
    control = DisplacementControl(
        node_id=probe, dof="uz", target_displacement=target
    )
    result = solve_static_nonlinear(
        model, load, control="displacement", displacement_control=control,
        kinematics="corotational", num_steps=num_steps,
        max_iterations=DC_MAX_ITER, tolerance=DC_TOL,
        record_increment_snapshots=True, **extra,
    )
    return {
        "model": model, "load": load, "probe": probe, "grid": grid,
        "result": result, "coupling": coupling,
    }


def _preyield_case(coupling: str, scale: float, *, true_elastic: bool = False) -> dict:
    model, load, probe, grid = _build_ep_model(coupling, scale)
    if true_elastic:
        # No hardening curve: shells bypass return mapping and beams bypass
        # the fiber-plasticity integration, while retaining E, nu and geometry.
        model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    _add_tip_load(model, load, grid)
    result = solve_static_nonlinear(
        model, load, kinematics="corotational", tolerance=DC_TOL,
        record_increment_snapshots=True, **FC_PARAMS
    )
    return {
        "model": model, "load": load, "probe": probe, "grid": grid,
        "result": result, "coupling": coupling,
    }


def _half_case(coupling: str) -> dict:
    return _cached(
        ("half", coupling), lambda: _dc_case(coupling, -0.10, CHECKPOINT_STEPS)
    )


def _full_case(coupling: str) -> dict:
    return _cached(("full", coupling), lambda: _dc_case(coupling, DC_TARGET, DC_STEPS))


def _resume_case(coupling: str, target: float = DC_TARGET, **extra) -> dict:
    """Resume -0.10 -> target (default -11/60) from copied state."""
    half = _half_case(coupling)
    model, load, probe, grid = _build_ep_model(coupling)
    _add_tip_load(model, load, grid)
    u_input = np.array(half["result"].displacements, dtype=float, copy=True)
    states_input = copy.deepcopy(dict(half["result"].element_states))
    u_snapshot = np.array(u_input, copy=True)
    states_snapshot = copy.deepcopy(states_input)
    control = DisplacementControl(
        node_id=probe, dof="uz", target_displacement=target
    )
    result = solve_static_nonlinear(
        model, load, control="displacement", displacement_control=control,
        kinematics="corotational", num_steps=RESUME_STEPS,
        max_iterations=DC_MAX_ITER, tolerance=DC_TOL,
        initial_displacements=u_input, initial_element_states=states_input,
        equilibrate_initial_state=False,
        record_increment_snapshots=True, **extra,
    )
    np.testing.assert_array_equal(u_input, u_snapshot)
    _assert_states_equal(states_input, states_snapshot)
    return {
        "model": model, "load": load, "probe": probe, "grid": grid,
        "result": result, "u_input": u_input, "states_input": states_input,
        "u_snapshot": u_snapshot, "states_snapshot": states_snapshot,
        "coupling": coupling,
    }


def _resumed_case(coupling: str) -> dict:
    return _cached(("resume", coupling), lambda: _resume_case(coupling))


def _seed_resumed_case(coupling: str) -> dict:
    return _cached(
        ("seed_resume", coupling),
        lambda: _resume_case(coupling, target=DC_SEED_TARGET),
    )


def _assert_states_equal(left, right) -> None:
    if not isinstance(left, dict):
        assert type(left) is type(right)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                _assert_states_equal(a, b)
        else:
            assert left == right
        return
    assert set(left) == set(right)
    for key in left:
        a, b = left[key], right[key]
        assert type(a) is type(b)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            _assert_states_equal(a, b)
        elif isinstance(a, (list, tuple)):
            _assert_states_equal(a, b)
        else:
            assert a == b


def _assert_states_allclose(left, right, *, physical=False) -> None:
    """Compare complete physical material histories at restart tolerances.

    T3: scalar alpha maxima alone can conceal changed distributions or a
    partially reset state, so every integration-point/fiber entry of the
    plastic-strain tensors, alpha arrays and fiber histories is compared
    with the existing strict restart tolerances (rtol 1e-6, atol 1e-9).
    """
    if not isinstance(left, dict):
        assert type(left) is type(right)
        if isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                _assert_states_allclose(a, b, physical=physical)
        elif physical:
            np.testing.assert_allclose(left, right, rtol=1.0e-6, atol=1.0e-9)
        else:
            _assert_states_equal(left, right)
        return
    assert set(left) == set(right)
    # Element-state metadata includes constitutive setup and bookkeeping.
    # Compare all physical state fields, including stresses and strains;
    # exact caller immutability is tested separately with _assert_states_equal.
    fields = {"plastic_strain", "alpha", "fiber_strain", "kinematic_fiber_strain",
              "fiber_stress", "axial_force", "layer_strain", "layer_strain_material",
              "kinematic_layer_strain", "layer_stress", "layer_stress_material"}
    for key in left:
        a, b = left[key], right[key]
        if not isinstance(key, (int, np.integer)) and not physical and key not in fields:
            continue
        assert type(a) is type(b)
        if isinstance(a, np.ndarray):
            assert a.shape == b.shape
            np.testing.assert_allclose(a, b, rtol=1.0e-6, atol=1.0e-9)
        elif isinstance(a, dict):
            _assert_states_allclose(a, b, physical=physical or key in fields)
        elif isinstance(a, (list, tuple)):
            _assert_states_allclose(a, b, physical=physical or key in fields)
        elif isinstance(a, float):
            np.testing.assert_allclose(a, b, rtol=1.0e-6, atol=1.0e-9)
        else:
            assert a == b


def _assert_material_schema(case, states) -> None:
    n_beam = 2 if case["coupling"] == "direct" else 3
    expected = set(range(1, NX * NY + 1)) | set(range(300, 300 + n_beam))
    assert expected <= set(states), "missing physical element history"
    for eid in expected:
        state = states[eid]
        if eid < 300:
            n = len(case["model"].mesh.elements[eid].gauss_points) * 5
            shapes = {"plastic_strain": (n, 3), "alpha": (n,)}
            shapes.update({name: (n, 3) for name in (
                "layer_strain", "layer_strain_material", "kinematic_layer_strain", "layer_stress")})
        else:
            # Fixture declares the default rectangular 5x5 fiber grid.
            shapes = {name: (25,) for name in (
                "plastic_strain", "alpha", "fiber_strain", "kinematic_fiber_strain", "fiber_stress")}
        for name, shape in shapes.items():
            assert name in state, (eid, name)
            value = np.asarray(state[name])
            assert value.shape == shape and value.size > 0, (eid, name, value.shape)
            assert np.all(np.isfinite(value)), (eid, name)


def _snapshot_at_control(snapshots, control_value: float):
    matches = [
        s for s in snapshots if abs(s.control_value - control_value) <= 1.0e-6
    ]
    assert len(matches) == 1, (
        control_value,
        [round(s.control_value, 9) for s in snapshots],
    )
    return matches[0]


def _equilibrium_oracle(case: dict) -> dict:
    """Cached assembly/map context shared by all equilibrium reconstructions."""
    if "_oracle" not in case:
        model, load = case["model"], case["load"]
        K0, _ = assemble_stiffness_matrix(model)
        F_prop, _ = assemble_load_vector(model, load)
        _, _, T, _u0, _, _ = build_constraint_transformation(K0, F_prop, model)
        mpc_map = build_nonlinear_rotating_mpc_map(model, T)
        assert mpc_map is not None and mpc_map.active
        case["_oracle"] = {
            "mpc_map": mpc_map,
            "F_prop": np.asarray(F_prop, dtype=float),
        }
    return case["_oracle"]


def _independent_equilibrium(
    case: dict,
    *,
    displacements=None,
    load_factor=None,
    element_states=None,
) -> dict:
    """Residual at the achieved load factor plus explicit force/wrench balance.

    The reduced residual is reconstructed independently of the solver's own
    convergence bookkeeping: J(u)^T (f_int(u, states) - lam * F_prop)
    relative to max(norm(J(u)^T lam * F_prop), 1) at the SAME achieved load
    factor (T2: an achieved-load claim must be normalized by the actual
    load, not the unit pattern).  The wrench balance is the physical support
    equilibrium: root-node residuals after rotating slave imbalance
    transfer, with deformed force arms and spatial moments (root rotations
    are zero, so the moment covectors are spatial moments), against the
    applied tip loads at their actual deformed loaded nodes.

    ``displacements``/``load_factor``/``element_states`` parameterize the
    reconstruction over an explicit accepted snapshot; the supplied state
    mappings are deep-copied so the stored snapshot is observed unchanged.
    """
    oracle = _equilibrium_oracle(case)
    mpc_map = oracle["mpc_map"]
    F_prop = oracle["F_prop"]
    model = case["model"]
    grid = case["grid"]
    result = case["result"]
    u = np.array((
        np.asarray(result.displacements, dtype=float)
        if displacements is None
        else np.asarray(displacements, dtype=float)
    ), dtype=float, copy=True)
    lam = float(result.load_factor if load_factor is None else load_factor)
    states = copy.deepcopy(dict(
        result.element_states if element_states is None else element_states
    ))
    _assert_material_schema(case, states)
    u_before, states_before = u.copy(), copy.deepcopy(states)
    f_int, _, _ = _assemble_nonlinear_system(
        model, u, states, 5, tangent=False, kinematics="corotational",
        corotational_tangent="work_conjugate",
    )
    np.testing.assert_array_equal(u, u_before)
    _assert_states_equal(states, states_before)
    reduced = mpc_map.reduce_force(u, f_int - lam * F_prop)
    reference = mpc_map.reduce_force(u, lam * F_prop)
    relative = float(np.linalg.norm(reduced)) / max(
        float(np.linalg.norm(reference)), 1.0
    )
    imbalance = np.asarray(f_int, dtype=float) - lam * np.asarray(
        F_prop, dtype=float
    )
    imbalance = mpc_map.transfer_slave_imbalance_to_masters(u, imbalance)
    dof_manager = model.mesh.dof_manager
    nodes = model.mesh.nodes

    def deformed_position(node_id: int) -> np.ndarray:
        node = nodes[node_id]
        dofs = dof_manager.get_node_dofs(node_id)
        return np.array([
            node.x + u[dofs[0]],
            node.y + u[dofs[1]],
            node.z + u[dofs[2]],
        ], dtype=float)

    support = np.zeros(6, dtype=float)
    for iy in range(NY + 1):
        dofs = dof_manager.get_node_dofs(grid[(0, iy)])
        force = imbalance[dofs[:3]]
        moment = imbalance[dofs[3:6]]
        support[:3] += force
        support[3:] += np.cross(deformed_position(grid[(0, iy)]), force) + moment
    applied = np.zeros(6, dtype=float)
    for iy in range(NY + 1):
        node_id = grid[(NX, iy)]
        dofs = dof_manager.get_node_dofs(node_id)
        force = lam * np.asarray(F_prop, dtype=float)[dofs[:3]]
        moment = lam * np.asarray(F_prop, dtype=float)[dofs[3:6]]
        applied[:3] += force
        applied[3:] += np.cross(deformed_position(node_id), force) + moment
    balance = support + applied
    return {
        "relative_residual": relative,
        "balance": balance,
        "support_wrench": support,
        "applied_wrench": applied,
    }


def _assert_completed_dc_case(
    case: dict, target: float, control_direction: int = -1
) -> None:
    result = case["result"]
    _assert_material_schema(case, result.element_states)
    assert result.converged
    uz = _probe_uz(case)
    postcheck = result.info.get("constraint_postcheck", {})
    assert postcheck.get("status") == "passed"
    assert float(postcheck.get("max_relative_residual", 1.0)) <= float(
        postcheck.get("relative_tolerance", 1.0)
    )
    assert len(result.snapshots) == len(result.steps)
    assert [s.control_value for s in result.snapshots] == [
        s.control_value for s in result.steps
    ]
    control_values = [s.control_value for s in result.steps]
    # Loading runs move the probe downward; the re-seeded unloading fixture
    # (DC_SEED_TARGET) moves it upward.  Either way the control is monotone.
    assert all(
        control_direction * (later - earlier) > 0.0
        for earlier, later in zip(control_values, control_values[1:])
    )
    equilibrium = _independent_equilibrium(case)
    assert equilibrium["relative_residual"] <= 1.0e-7
    applied_norm = float(np.linalg.norm(equilibrium["applied_wrench"]))
    balance_norm = float(np.linalg.norm(equilibrium["balance"]))
    assert balance_norm <= 1.0e-6 * max(applied_norm, 1.0e-30)

    # T2: EVERY accepted snapshot is an equilibrium at its own achieved load
    # factor, lies on the rotating-MPC manifold, and retains its stored
    # displacement/material history through the reconstruction.  Inserted
    # adaptive subdivisions are included; the final snapshot must agree with
    # the public final state.
    mpc_map = _equilibrium_oracle(case)["mpc_map"]
    assert result.snapshots, "completed DC case must record accepted snapshots"
    for snapshot in result.snapshots:
        stored_u = np.array(snapshot.displacements, dtype=float, copy=True)
        stored_states = copy.deepcopy(dict(snapshot.element_states))
        input_u = stored_u.copy()
        input_states = copy.deepcopy(stored_states)
        snapshot_equilibrium = _independent_equilibrium(
            case,
            displacements=input_u,
            load_factor=float(snapshot.load_factor),
            element_states=input_states,
        )
        assert snapshot_equilibrium["relative_residual"] <= 1.0e-7, (
            snapshot.step_index,
            snapshot_equilibrium["relative_residual"],
        )
        linear_u = mpc_map.to_linear_displacement(stored_u) if mpc_map else stored_u
        assert constraint_residual_summary(case["model"], linear_u)["status"] == "passed"
        # The stored snapshot was not mutated by the reconstruction.
        np.testing.assert_array_equal(
            np.asarray(snapshot.displacements, dtype=float), stored_u
        )
        _assert_states_equal(dict(snapshot.element_states), stored_states)
        np.testing.assert_array_equal(input_u, stored_u)
        _assert_states_equal(input_states, stored_states)
        # Per-snapshot physical wrench balance on BOTH routes: the intrinsic
        # SO(3) mean made the interpolated route rigid-spin equivariant, so
        # its previously open wrench-objectivity finding is now held at every
        # accepted snapshot, not only at the final state.
        snap_applied = float(np.linalg.norm(
            snapshot_equilibrium["applied_wrench"]
        ))
        snap_balance = float(np.linalg.norm(
            snapshot_equilibrium["balance"]
        ))
        assert snap_balance <= 1.0e-6 * max(snap_applied, 1.0e-30), (
            snapshot.step_index, snap_balance, snap_applied,
        )
    np.testing.assert_allclose(
        np.asarray(result.snapshots[-1].displacements, dtype=float),
        np.asarray(result.displacements, dtype=float),
        rtol=0.0, atol=1.0e-9,
    )
    assert abs(
        float(result.snapshots[-1].load_factor) - float(result.load_factor)
    ) <= 1.0e-12
    # Assess every accepted partial state before reporting incomplete target
    # attainment. Public converged also includes stopped_at_limit.
    assert result.status == "completed", (
        result.status, result.stop_reason, uz, result.load_factor,
        equilibrium["relative_residual"], len(result.snapshots),
    )
    assert abs(uz - target) <= 1.0e-8


# ---------------------------------------------------------------------------
# Pre-yield controls (maintained from the passed diagnostic evidence)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coupling", ("direct", "interp"))
def test_preyield_plastic_matches_elastic_control_exactly(coupling: str) -> None:
    plastic = _cached(
        ("preyield", coupling, "plastic"),
        lambda: _preyield_case(coupling, 1.0),
    )
    elastic = _cached(
        ("preyield", coupling, "elastic"),
        lambda: _preyield_case(coupling, 1.0e4),
    )
    for case in (plastic, elastic):
        assert case["result"].status == "completed"
        assert abs(float(case["result"].load_factor) - 1.0) <= 1.0e-12
        assert _alpha_max(case["result"].element_states) == 0.0
        assert case["result"].steps[-1].max_equivalent_plastic_strain == 0.0
        assert len(case["result"].snapshots) == len(case["result"].steps) > 0
        mapping = _equilibrium_oracle(case)["mpc_map"]
        for snapshot in case["result"].snapshots:
            u_before = np.array(snapshot.displacements, copy=True)
            states_before = copy.deepcopy(dict(snapshot.element_states))
            checked = _independent_equilibrium(
                case, displacements=snapshot.displacements, load_factor=snapshot.load_factor,
                element_states=snapshot.element_states)
            assert checked["relative_residual"] <= 1.0e-7
            assert np.linalg.norm(checked["balance"]) <= 1.0e-6 * max(
                float(np.linalg.norm(checked["applied_wrench"])), 1.0e-30)
            assert constraint_residual_summary(
                case["model"], mapping.to_linear_displacement(snapshot.displacements))["status"] == "passed"
            np.testing.assert_array_equal(snapshot.displacements, u_before)
            _assert_states_equal(dict(snapshot.element_states), states_before)
    assert _probe_uz(plastic) == _probe_uz(elastic)
    np.testing.assert_array_equal(plastic["result"].displacements, elastic["result"].displacements)
    _assert_states_allclose(dict(plastic["result"].element_states), dict(elastic["result"].element_states))


@pytest.mark.parametrize("coupling", ("direct", "interp"))
def test_preyield_plastic_matches_distinct_linear_elastic_material(coupling: str) -> None:
    plastic = _cached(("preyield", coupling, "plastic"), lambda: _preyield_case(coupling, 1.0))
    elastic = _cached(
        ("preyield", coupling, "true_elastic"),
        lambda: _preyield_case(coupling, 1.0, true_elastic=True),
    )
    assert elastic["model"].get_material("steel").hardening_curve is None
    assert plastic["model"].get_material("steel").hardening_curve is not None
    assert _alpha_max(plastic["result"].element_states) == 0.0
    for state in plastic["result"].element_states.values():
        if isinstance(state, Mapping) and "plastic_strain" in state:
            assert np.count_nonzero(state["plastic_strain"]) == 0
    for case in (plastic, elastic):
        assert case["result"].status == "completed"
        assert abs(float(case["result"].load_factor) - 1.0) <= 1.0e-12
    u_p, u_e = (np.asarray(c["result"].displacements) for c in (plastic, elastic))
    assert np.all(np.isfinite(u_p)) and np.all(np.isfinite(u_e))
    displacement_error = float(np.linalg.norm(u_p-u_e) / np.linalg.norm(u_e))
    assert np.isfinite(displacement_error)
    stress_errors = {}
    fiber_errors = {}
    stress_witnesses = {}
    # Compare physical stress responses, allowing distinct state schemas.
    from anyfesolver.corotational import _corotational_cache, _corotational_deformation_state
    for eid, element in elastic["model"].mesh.elements.items():
        if isinstance(element, BeamElement):
            assert element._fiber_plasticity_config(elastic["model"].get_material("steel")) is None
            stresses = []
            for case in (plastic, elastic):
                model = case["model"]
                beam = model.mesh.elements[eid]
                dofs = np.concatenate([model.mesh.dof_manager.get_node_dofs(n) for n in beam.node_ids])
                ud, _, _ = _corotational_deformation_state(
                    _corotational_cache(model)[eid], beam, case["result"].displacements[dofs])
                local_u = np.zeros_like(case["result"].displacements)
                local_u[dofs] = ud
                stress = beam.compute_stresses(model.mesh, local_u, model.get_material("steel"))
                stresses.append(np.array([stress[k] for k in (
                    "axial_stress", "bending_stress_y", "bending_stress_z",
                    "shear_stress_y", "shear_stress_z", "torsional_stress")]))
            # Observe the ACTUAL plastic constitutive output, independently of
            # the common elastic stress-recovery helper used above.
            pbeam = plastic["model"].mesh.elements[eid]
            config = pbeam._fiber_plasticity_config(plastic["model"].get_material("steel"))
            y, z, _ = pbeam._fiber_section_grid(config)
            length, transform = beam._beam_frame_and_transform(beam.get_node_coordinates(model.mesh))
            v = transform @ ud  # final loop member is the true elastic state
            eps0 = (v[6]-v[0])/length + ((v[7]-v[1])**2+(v[8]-v[2])**2)/(2*length**2)
            expected = 2.1e11*(eps0 + z*(v[10]-v[4])/length + y*(v[11]-v[5])/length)
            actual = np.asarray(plastic["result"].element_states[eid]["fiber_stress"])
            assert actual.shape == expected.shape == (25,)
            assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), eid
            stress_witnesses[eid] = actual.size
            fiber_errors[eid] = float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1.0))
            assert np.isfinite(fiber_errors[eid]), eid
        elif eid in range(1, NX*NY+1):
            # The true elastic shell shortcut has no stored layer stresses.
            # Recover them directly from its own mapped displacement and the
            # isotropic elastic law; do not send it through return mapping.
            from anyfesolver.plasticity import lobatto_layers
            model = elastic["model"]
            shell = model.mesh.elements[eid]
            dofs = np.concatenate([model.mesh.dof_manager.get_node_dofs(n) for n in shell.node_ids])
            ud, _, _ = _corotational_deformation_state(
                _corotational_cache(model)[eid], shell, elastic["result"].displacements[dofs])
            geometry = shell._nonlinear_geometry(model.mesh)
            v = geometry["T0"] @ ud
            zs, _ = lobatto_layers(5, shell.thickness)
            C = 2.1e11/(1-.3**2)*np.array([[1., .3, 0.], [.3, 1., 0.], [0., 0., (1-.3)/2]])
            # A plastic-capable qualified Q4 evaluates its layers at the hybrid
            # assumed strain (compatible strain + dE u); its elastic stress is
            # C times that strain.  The oracle uses the same kinematics.
            from anyfesolver import _q4_assumed_strain_plasticity as assumed_strain
            pshell = plastic["model"].mesh.elements[eid]
            pmaterial = plastic["model"].get_material("steel")
            offsets = (assumed_strain.offset_operator(pshell, plastic["model"].mesh, pmaterial)
                       if assumed_strain.supported(pshell, plastic["model"].mesh, pmaterial)
                       else np.zeros((len(geometry["gp"]), 6, 24)))
            layers = []
            for index, gp in enumerate(geometry["gp"]):
                theta = gp["Gw"] @ v
                offset = offsets[index] @ v
                membrane = gp["B_m"] @ v + [.5*theta[0]**2, .5*theta[1]**2, theta[0]*theta[1]] + offset[:3]
                layers.extend((membrane + zs[:, None]*(gp["B_b"]@v + offset[3:])) @ C.T)
            expected = np.asarray(layers)
            actual = np.asarray(plastic["result"].element_states[eid]["layer_stress"])
            assert actual.shape == expected.shape == (len(geometry["gp"])*5, 3)
            assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), eid
            stresses = [actual, expected]
            stress_witnesses[eid] = actual.shape[0]
        else:
            continue
        assert all(np.all(np.isfinite(value)) for value in stresses), eid
        stress_errors[eid] = float(np.linalg.norm(stresses[0]-stresses[1]) / max(np.linalg.norm(stresses[1]), 1.0))
        assert np.isfinite(stress_errors[eid]), eid
    assert set(stress_errors) == set(range(1, NX*NY+1)) | set(range(300, 302 if coupling == "direct" else 303))
    assert set(stress_witnesses) == set(stress_errors)
    assert set(fiber_errors) == set(range(300, 302 if coupling == "direct" else 303))
    assert displacement_error <= 1.0e-7, {"displacement_relative_error": displacement_error,
                                         "stress_relative_errors": stress_errors}
    assert max(stress_errors.values()) <= 1.0e-7, stress_errors
    assert max(fiber_errors.values()) <= 1.0e-7, fiber_errors


# ---------------------------------------------------------------------------
# Uninterrupted displacement control to the amended -11/60 m endpoint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coupling", ("direct", "interp"))
def test_dc_full_target_plasticity_and_equilibrium(coupling: str, monkeypatch, tmp_path) -> None:
    # Persist accepted physical arrays before checking completion. A bounded
    # timeout must retain the actual path, not merely a pytest progress marker.
    import anyfesolver.nonlinear_static as nonlinear_static
    real_snapshot = nonlinear_static._increment_snapshot
    accepted = []
    real_factorize = nonlinear_static.factorize
    rhs_per_attempt = {}
    import anyfesolver._nonlinear_seed as nonlinear_seed
    real_seed = nonlinear_seed.secant_plane_seed
    observed_failed_seed_origins = set()

    def observe_seed_origin(q, lam, previous, *args, **kwargs):
        origin = (q.tobytes(), float(lam), previous[0].tobytes(), float(previous[1]))
        assert origin not in observed_failed_seed_origins, 'identical failed plane was recomputed'
        result, info = real_seed(q, lam, previous, *args, **kwargs)
        if info['status'] in ('seed_domain_refused', 'seed_solve_budget',
                              'seed_merit_no_improvement', 'invalid_accepted_secant'):
            observed_failed_seed_origins.add(origin)
        return result, info

    def count_global_rhs(matrix, *args, **kwargs):
        import inspect
        frame = inspect.currentframe().f_back
        try:
            while frame is not None and frame.f_code.co_name != '_solve_static_displacement_control':
                frame = frame.f_back
            assert frame is not None
            key = (frame.f_locals['step_index'], frame.f_locals['attempts_this_step'])
            limit = frame.f_locals['max_iterations']
        finally:
            del frame
        handle = real_factorize(matrix, *args, **kwargs)
        class Counted:
            def __getattr__(self, name): return getattr(handle, name)
            def solve(self, rhs):
                rhs_per_attempt[key] = rhs_per_attempt.get(key, 0)+1
                assert rhs_per_attempt[key] <= limit
                return handle.solve(rhs)
        return Counted()

    def record_snapshot(*args, **kwargs):
        snapshot = real_snapshot(*args, **kwargs)
        payload = dict(
            step_index=int(snapshot.step_index), control=float(snapshot.control_value),
            lam=float(snapshot.load_factor),
            displacements=np.array(snapshot.displacements, copy=True),
            element_states=copy.deepcopy(dict(snapshot.element_states)),
        )
        path = tmp_path / f"accepted-{len(accepted)+1:04d}.pkl"
        with path.open("xb") as stream:
            pickle.dump(payload, stream, protocol=5)
        accepted.append(path)
        print(json.dumps(dict(kind="dc_accepted", coupling=coupling,
                              control=payload["control"], lam=payload["lam"],
                              artifact=str(path))), flush=True)
        return snapshot

    with monkeypatch.context() as context:
        context.setattr(nonlinear_static, "_increment_snapshot", record_snapshot)
        context.setattr(nonlinear_static, 'factorize', count_global_rhs)
        context.setattr(nonlinear_seed, 'secant_plane_seed', observe_seed_origin)
        case = _full_case(coupling)
    # Preserve terminal adaptation/reason/history even when completion fails.
    # Accepted-state scientific assertions below remain unchanged.
    (tmp_path / "terminal.pkl").write_bytes(pickle.dumps(dict(
        displacements=case["result"].displacements,
        element_states=case["result"].element_states,
        load_factor=case["result"].load_factor,
        status=case["result"].status, stop_reason=case["result"].stop_reason,
        info=case["result"].info, rhs_per_attempt=rhs_per_attempt), protocol=5))
    _assert_completed_dc_case(case, DC_TARGET)
    result = case["result"]
    assert _alpha_max(result.element_states) > 0.0
    assert result.steps[-1].max_equivalent_plastic_strain > 0.0
    assert rhs_per_attempt and max(rhs_per_attempt.values()) <= DC_MAX_ITER
    for seed in result.info['displacement_control_adaptation']['uncommitted_seeds']:
        assert seed['strategy_engaged'] or abs(seed['accepted_load_factor']) < abs(seed['previous_accepted_load_factor'])
        assert (seed['accepted_control']-seed['previous_accepted_control'])*(seed['target']-seed['accepted_control']) > 0.
        assert seed['solves']+seed['remaining_iterations'] == DC_MAX_ITER
    strains = [s.max_equivalent_plastic_strain for s in result.steps]
    assert all(later >= earlier for earlier, later in zip(strains, strains[1:]))


@pytest.mark.parametrize('stage', ('probe', 'solve', 'complete'))
def test_dc_seed_cancellation_preserves_restart_inputs(stage):
    half = _half_case('direct')
    model, load, probe, grid = _build_ep_model('direct'); _add_tip_load(model, load, grid)
    u = half['result'].displacements.copy(); states = copy.deepcopy(dict(half['result'].element_states))
    before_u, before_states = u.copy(), copy.deepcopy(states)
    class SeedToken(CancellationToken):
        hits = 0
        def raise_if_cancelled(self, location=''):
            if location == 'nonlinear_static.displacement.seed.'+stage:
                self.hits += 1; self.cancel('cancel inside uncommitted seed')
            super().raise_if_cancelled(location)
    token = SeedToken()
    with pytest.raises(SolveCancelled, match='cancel inside uncommitted seed'):
        solve_static_nonlinear(model, load, control='displacement',
            displacement_control=DisplacementControl(node_id=probe, dof='uz', target_displacement=DC_SEED_TARGET),
            num_steps=RESUME_STEPS, max_iterations=DC_MAX_ITER, tolerance=DC_TOL, kinematics='corotational',
            initial_displacements=u, initial_element_states=states, equilibrate_initial_state=False,
            emit_restart_checkpoint=False, cancellation_token=token)
    assert token.hits == 1
    np.testing.assert_array_equal(u, before_u); _assert_states_equal(states, before_states)
    # Reuse the SAME model and caller objects after cancellation inside the
    # private seed, then compare against a clean uninterrupted public route.
    reused = solve_static_nonlinear(model, load, control='displacement',
        displacement_control=DisplacementControl(node_id=probe, dof='uz', target_displacement=DC_SEED_TARGET),
        num_steps=RESUME_STEPS, max_iterations=DC_MAX_ITER, tolerance=DC_TOL, kinematics='corotational',
        initial_displacements=u, initial_element_states=states, equilibrate_initial_state=False,
        emit_restart_checkpoint=False, record_increment_snapshots=True)
    reused_case = dict(model=model, load=load, probe=probe, grid=grid, result=reused, coupling='direct')
    _assert_completed_dc_case(reused_case, DC_SEED_TARGET, control_direction=1)
    reference = _seed_resumed_case('direct')
    _assert_completed_dc_case(reference, DC_SEED_TARGET, control_direction=1)
    np.testing.assert_allclose(reused.displacements, reference['result'].displacements, rtol=1e-6, atol=1e-9)
    assert abs(reused.load_factor-reference['result'].load_factor) <= 1e-6*max(abs(reference['result'].load_factor), 1.)
    _assert_states_allclose(dict(reused.element_states), dict(reference['result'].element_states), physical=True)
    np.testing.assert_array_equal(u, before_u); _assert_states_equal(states, before_states)


def test_dc_seed_unrelated_linear_error_propagates_and_preserves_inputs(monkeypatch):
    import anyfesolver.nonlinear_static as driver
    half = _half_case('direct')
    model, load, probe, grid = _build_ep_model('direct'); _add_tip_load(model, load, grid)
    u = half['result'].displacements.copy(); states = copy.deepcopy(dict(half['result'].element_states))
    before_u, before_states = u.copy(), copy.deepcopy(states)
    factorize = driver.factorize
    def broken_provider(matrix, *args, **kwargs):
        if str(kwargs.get('signature', '')).startswith('nonlinear.displacement_seed:'):
            raise ValueError('unrelated seed provider defect')
        return factorize(matrix, *args, **kwargs)
    monkeypatch.setattr(driver, 'factorize', broken_provider)
    with pytest.raises(ValueError, match='unrelated seed provider defect'):
        solve_static_nonlinear(model, load, control='displacement',
            displacement_control=DisplacementControl(node_id=probe, dof='uz', target_displacement=DC_SEED_TARGET),
            num_steps=RESUME_STEPS, max_iterations=DC_MAX_ITER, tolerance=DC_TOL, kinematics='corotational',
            initial_displacements=u, initial_element_states=states, equilibrate_initial_state=False,
            emit_restart_checkpoint=False)
    np.testing.assert_array_equal(u, before_u); _assert_states_equal(states, before_states)


# ---------------------------------------------------------------------------
# Split restart via the public initial-state route
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coupling", ("direct", "interp"))
def test_dc_half_reaches_minus_0_10_with_plasticity(coupling: str) -> None:
    case = _half_case(coupling)
    _assert_completed_dc_case(case, -0.10)
    assert case["result"].steps[-1].max_equivalent_plastic_strain > 0.0


@pytest.mark.parametrize("coupling", ("direct", "interp"))
def test_dc_resume_reproduces_uninterrupted_run(coupling: str) -> None:
    resumed = _resumed_case(coupling)
    _assert_completed_dc_case(resumed, DC_TARGET)
    full = _full_case(coupling)
    _assert_completed_dc_case(full, DC_TARGET)
    half = _half_case(coupling)

    # Caller restart inputs are preserved by the successful resumed solve.
    np.testing.assert_array_equal(resumed["u_input"], resumed["u_snapshot"])
    _assert_states_equal(resumed["states_input"], resumed["states_snapshot"])

    resumed_result = resumed["result"]
    full_result = full["result"]
    u_resumed = np.asarray(resumed_result.displacements, dtype=float)
    u_full = np.asarray(full_result.displacements, dtype=float)
    assert float(np.max(np.abs(u_resumed - u_full))) <= 1.0e-6
    lam_resumed = float(resumed_result.load_factor)
    lam_full = float(full_result.load_factor)
    assert abs(lam_resumed - lam_full) <= 1.0e-6 * max(abs(lam_full), 1.0)

    # Physical histories: the resumed increments continue the uninterrupted
    # accepted states.  Both runs preserve every scheduled control milestone;
    # accepted adaptive subdivisions are inserted increments, so the runs are
    # compared on their scheduled milestones with monotone in-between history.
    resumed_steps = resumed_result.steps
    half_target = -0.10
    tail_milestones = [
        half_target + (DC_TARGET - half_target) * k / RESUME_STEPS
        for k in range(1, RESUME_STEPS + 1)
    ]

    def milestone_steps(steps, milestones):
        selected = []
        for milestone in milestones:
            matches = [
                s for s in steps if abs(s.control_value - milestone) <= 1.0e-6
            ]
            assert len(matches) == 1, (
                milestone, [round(s.control_value, 9) for s in steps]
            )
            selected.append(matches[0])
        return selected

    resumed_control = [s.control_value for s in resumed_steps]
    assert all(later < earlier for earlier, later in zip(
        resumed_control, resumed_control[1:]
    ))
    full_tail = [
        s for s in full_result.steps if s.control_value <= half_target - 1.0e-6
    ]
    resumed_milestones = milestone_steps(resumed_steps, tail_milestones)
    full_milestones = milestone_steps(full_tail, tail_milestones)
    np.testing.assert_allclose(
        [s.control_value for s in resumed_milestones],
        [s.control_value for s in full_milestones],
        rtol=0.0, atol=1.0e-8,
    )
    np.testing.assert_allclose(
        [s.load_factor for s in resumed_milestones],
        [s.load_factor for s in full_milestones],
        rtol=1.0e-6, atol=1.0e-12,
    )
    np.testing.assert_allclose(
        [s.max_equivalent_plastic_strain for s in resumed_milestones],
        [s.max_equivalent_plastic_strain for s in full_milestones],
        rtol=1.0e-6, atol=1.0e-9,
    )
    resumed_strains = [s.max_equivalent_plastic_strain for s in resumed_steps]
    assert all(later >= earlier for earlier, later in zip(
        resumed_strains, resumed_strains[1:]
    ))
    assert resumed_steps[-1].max_equivalent_plastic_strain > 0.0

    # T3: full physical material histories (plastic-strain tensors, alpha
    # arrays, fiber histories; every element and integration point) agree
    # between the split restart and the uninterrupted run at the split
    # state, at every common scheduled milestone and at the final state.
    # The half-run prefix is compared against the uninterrupted prefix on
    # the same scheduled control grid.
    half_result = half["result"]
    half_milestones = [
        -0.10 * k / CHECKPOINT_STEPS for k in range(1, CHECKPOINT_STEPS + 1)
    ]
    for milestone in half_milestones:
        _assert_states_allclose(
            dict(_snapshot_at_control(
                half_result.snapshots, milestone
            ).element_states),
            dict(_snapshot_at_control(
                full_result.snapshots, milestone
            ).element_states),
        )
    for milestone in tail_milestones:
        resumed_snapshot = _snapshot_at_control(
            resumed_result.snapshots, milestone
        )
        full_snapshot = _snapshot_at_control(full_result.snapshots, milestone)
        _assert_states_allclose(
            dict(resumed_snapshot.element_states),
            dict(full_snapshot.element_states),
        )
    _assert_states_allclose(
        dict(resumed_result.element_states),
        dict(full_result.element_states),
    )


def test_dc_rejected_restart_input_refused_and_preserved() -> None:
    half = _half_case("direct")
    model, load, probe, grid = _build_ep_model("direct")
    _add_tip_load(model, load, grid)
    u_input = np.array(half["result"].displacements, dtype=float, copy=True)
    # Violate the beam-shell MPC: the slave translation no longer matches its
    # master plus lever arm, so the restart state must be refused.
    slave_dofs = model.mesh.dof_manager.get_node_dofs(100 + NX)
    u_input[slave_dofs[2]] += 1.0e-3
    u_snapshot = np.array(u_input, copy=True)
    # T4: BOTH caller-owned inputs are named deep copies and observed after
    # the refusal (an inline copy cannot be compared afterwards).
    states_input = copy.deepcopy(dict(half["result"].element_states))
    states_snapshot = copy.deepcopy(states_input)
    control = DisplacementControl(
        node_id=probe, dof="uz", target_displacement=DC_TARGET
    )
    with pytest.raises(ValueError):
        solve_static_nonlinear(
            model, load, control="displacement", displacement_control=control,
            kinematics="corotational", num_steps=RESUME_STEPS,
            max_iterations=DC_MAX_ITER, tolerance=DC_TOL,
            initial_displacements=u_input,
            initial_element_states=states_input,
            equilibrate_initial_state=False,
        )
    np.testing.assert_array_equal(u_input, u_snapshot)
    _assert_states_equal(states_input, states_snapshot)


def test_dc_cancelled_resume_preserves_inputs_and_same_inputs_still_deliver(monkeypatch) -> None:
    half = _half_case("direct")
    model, load, probe, grid = _build_ep_model("direct")
    _add_tip_load(model, load, grid)
    u_input = np.array(half["result"].displacements, dtype=float, copy=True)
    states_input = copy.deepcopy(dict(half["result"].element_states))
    u_snapshot = np.array(u_input, copy=True)
    states_snapshot = copy.deepcopy(states_input)

    token = CancellationToken()
    accepted_events: list = []

    def cancel_after_two_accepted(event) -> None:
        if event.event_type == "nonlinear_static_step":
            accepted_events.append(event)
            if len(accepted_events) >= 2:
                token.cancel("test cancellation after two accepted increments")

    control = DisplacementControl(
        node_id=probe, dof="uz", target_displacement=DC_TARGET
    )
    common = dict(
        control="displacement", displacement_control=control,
        kinematics="corotational", num_steps=RESUME_STEPS,
        max_iterations=DC_MAX_ITER, tolerance=DC_TOL,
        initial_displacements=u_input, initial_element_states=states_input,
        equilibrate_initial_state=False,
        record_increment_snapshots=True,
    )
    import anyfesolver.nonlinear_static as nonlinear_static
    real_snapshot = nonlinear_static._increment_snapshot
    observed_snapshots = []

    def observe_snapshot(*args, **kwargs):
        snapshot = real_snapshot(*args, **kwargs)
        observed_snapshots.append((snapshot, np.array(snapshot.displacements, copy=True),
                                   copy.deepcopy(dict(snapshot.element_states))))
        return snapshot

    with monkeypatch.context() as context:
        context.setattr(nonlinear_static, "_increment_snapshot", observe_snapshot)
        with pytest.raises(SolveCancelled):
            solve_static_nonlinear(
                model, load, cancellation_token=token,
                progress_callback=cancel_after_two_accepted, **common,
            )
    # Cancellation happened through accepted progress events only.
    assert len(accepted_events) == 2
    assert len(observed_snapshots) == 2
    for snapshot, before_u, before_states in observed_snapshots:
        np.testing.assert_array_equal(snapshot.displacements, before_u)
        _assert_states_equal(dict(snapshot.element_states), before_states)
        _assert_material_schema(half, snapshot.element_states)
    # Caller restart inputs are preserved by the cancelled solve.
    np.testing.assert_array_equal(u_input, u_snapshot)
    _assert_states_equal(states_input, states_snapshot)

    # The very same input objects still deliver the fresh resumed run on a
    # clean model.
    reuse_model, reuse_load, reuse_probe, _grid = _build_ep_model("direct")
    _add_tip_load(reuse_model, reuse_load, _grid)
    reuse_control = DisplacementControl(
        node_id=reuse_probe, dof="uz", target_displacement=DC_TARGET
    )
    reuse = solve_static_nonlinear(
        reuse_model, reuse_load, control="displacement",
        displacement_control=reuse_control,
        kinematics="corotational", num_steps=RESUME_STEPS,
        max_iterations=DC_MAX_ITER, tolerance=DC_TOL,
        initial_displacements=u_input, initial_element_states=states_input,
        equilibrate_initial_state=False,
    )
    assert reuse.status == "completed"
    assert abs(_probe_uz({"model": reuse_model, "result": reuse, "probe": reuse_probe}) - DC_TARGET) <= 1.0e-8
    np.testing.assert_array_equal(u_input, u_snapshot)
    _assert_states_equal(states_input, states_snapshot)

    # And that fresh resume reproduces the cached resumed result bit-for-bit
    # up to roundoff (same immutable inputs, same deterministic route).
    resumed = _resumed_case("direct")
    np.testing.assert_allclose(
        np.asarray(reuse.displacements, dtype=float),
        np.asarray(resumed["result"].displacements, dtype=float),
        rtol=0.0, atol=1.0e-12,
    )
    assert abs(float(reuse.load_factor) - float(resumed["result"].load_factor)) <= 1.0e-12
    for snapshot, before_u, before_states in observed_snapshots:
        np.testing.assert_array_equal(snapshot.displacements, before_u)
        _assert_states_equal(dict(snapshot.element_states), before_states)


def test_dc_stopped_resume_snapshots_exclude_rejected_history() -> None:
    half = _half_case("direct")
    model, load, probe, grid = _build_ep_model("direct")
    _add_tip_load(model, load, grid)
    control = DisplacementControl(
        node_id=probe, dof="uz", target_displacement=DC_TARGET
    )
    u_input = np.array(half["result"].displacements, dtype=float, copy=True)
    states_input = copy.deepcopy(dict(half["result"].element_states))
    u_before, states_before = u_input.copy(), copy.deepcopy(states_input)
    result = solve_static_nonlinear(
        model, load, control="displacement", displacement_control=control,
        kinematics="corotational", num_steps=1, max_iterations=1,
        tolerance=DC_TOL,
        initial_displacements=u_input,
        initial_element_states=states_input,
        equilibrate_initial_state=False,
        record_increment_snapshots=True,
    )
    np.testing.assert_array_equal(u_input, u_before)
    _assert_states_equal(states_input, states_before)
    # A single Newton iteration per step cannot converge the plastic
    # increment, so the run must stop without completing; whatever was
    # accepted is exactly the recorded snapshot/rejected-free history.
    assert result.status != "completed"
    assert len(result.snapshots) == len(result.steps)
    if result.steps:
        assert [s.control_value for s in result.snapshots] == [
            s.control_value for s in result.steps
        ]
        uz = float(
            np.asarray(result.displacements)[
                model.mesh.dof_manager.get_node_dofs(probe)[2]
            ]
        )
        assert abs(uz - float(result.steps[-1].control_value)) <= 1.0e-8
    else:
        assert result.snapshots == ()
        uz = float(
            np.asarray(result.displacements)[
                model.mesh.dof_manager.get_node_dofs(probe)[2]
            ]
        )
        # Rolled back to the unchanged restart state.
        assert abs(uz - (-0.10)) <= 1.0e-7
        # Returned coordinates pass through the established map round trip;
        # caller-owned inputs above remain EXACTLY unchanged. Use the
        # existing map round-trip tolerance for the reconstructed result.
        np.testing.assert_allclose(result.displacements, u_before, rtol=1.0e-14, atol=1.0e-15)
        _assert_states_allclose(dict(result.element_states), states_before)
        checked = _independent_equilibrium({
            "model": model, "load": load, "probe": probe, "grid": grid,
            "result": result, "coupling": "direct"})
        assert checked["relative_residual"] <= 1.0e-7


# Reverse plastic flow keeps the secant-plane seed engaged: from -0.10 to
# -0.03 the probe unloads elastically, then the shells/beams yield in reverse
# while the seed strategy stays engaged.  This exercises cancellation inside a
# seed whose increment is actively yielding (the -0.07 fixture above only
# covers elastic unloading).
DC_REVERSE_YIELD_TARGET = -0.03


def _reverse_yield_case() -> dict:
    return _cached(("reverse_yield", "direct"),
                   lambda: _resume_case("direct", target=DC_REVERSE_YIELD_TARGET))


def _first_yielding_seed(case: dict) -> int:
    result = case["result"]
    seeds = result.info["displacement_control_adaptation"]["uncommitted_seeds"]
    controls = [s.control_value for s in result.steps]
    strains = [s.max_equivalent_plastic_strain for s in result.steps]
    for index, seed in enumerate(seeds):
        if seed["accepted_control"] not in controls:
            continue
        # accepted_control is the seed's starting state; the seeded (and later
        # cancelled) increment is the next accepted step, which must yield.
        step = controls.index(seed["accepted_control"])
        if step + 1 < len(strains) and strains[step + 1] > strains[step]:
            return index
    raise AssertionError("no seeded increment with active plastic flow")


@pytest.mark.parametrize('stage', ('probe', 'solve', 'complete'))
def test_dc_seed_cancellation_during_actively_yielding_increment(stage):
    reference = _reverse_yield_case()
    assert reference["result"].status == "completed"
    target_seed = _first_yielding_seed(reference)
    half = _half_case('direct')
    model, load, probe, grid = _build_ep_model('direct'); _add_tip_load(model, load, grid)
    u = half['result'].displacements.copy(); states = copy.deepcopy(dict(half['result'].element_states))
    before_u, before_states = u.copy(), copy.deepcopy(states)

    class YieldingSeedToken(CancellationToken):
        completed = 0
        hits = 0

        def raise_if_cancelled(self, location=''):
            if location == 'nonlinear_static.displacement.seed.complete' and self.hits == 0:
                if self.completed == target_seed and stage == 'complete':
                    self.hits += 1
                    self.cancel('cancel inside a yielding seed')
                self.completed += 1
            elif (location == 'nonlinear_static.displacement.seed.' + stage
                  and self.completed == target_seed and self.hits == 0):
                self.hits += 1
                self.cancel('cancel inside a yielding seed')
            super().raise_if_cancelled(location)

    token = YieldingSeedToken()
    with pytest.raises(SolveCancelled, match='cancel inside a yielding seed'):
        solve_static_nonlinear(model, load, control='displacement',
            displacement_control=DisplacementControl(node_id=probe, dof='uz', target_displacement=DC_REVERSE_YIELD_TARGET),
            num_steps=RESUME_STEPS, max_iterations=DC_MAX_ITER, tolerance=DC_TOL, kinematics='corotational',
            initial_displacements=u, initial_element_states=states, equilibrate_initial_state=False,
            emit_restart_checkpoint=False, cancellation_token=token)
    assert token.hits == 1
    np.testing.assert_array_equal(u, before_u); _assert_states_equal(states, before_states)
