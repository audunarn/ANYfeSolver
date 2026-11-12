"""Nonlinear/contact reliability tests for the standard edge beam-shell routes.

The three standard attachment routes (shared beam/shell node IDs, one-to-one
eccentric MPC, interpolated edge MPC) are exercised together with the
qualified planar rigid-plane penalty contact profile
(``PLANAR_Q4_RIGID_PLANE_PENALTY_MORTAR_V1``) on a qualified Q4 cantilever
with a standard ``BeamElement`` along the free shell edge.  Contact acts only
on the shell.  The coupling keeps the existing exact fixed-reference MPC law
with the single common offset ``r`` and no penalty coupling; von Karman
small-rotation kinematics are used for the lifecycle and restart solves.

Fixture contract (fixed before running):

- The successful small-rotation cycle and restart fixtures use a rigid plane
  at the reference shell midsurface ``z = 0`` (the admitted plane-fixture
  configuration of the existing plane module's guided-plate restart), gap
  tolerance 0.001, contact stiffness 1e5 and press force 0.1 per beam node.
  The root lies on the plane: an open accepted state may therefore report
  ``minimum_gap == 0`` at the supported root, but must carry zero active
  pressure/force and a positive free-edge gap.  Accepted rotations are
  explicitly bounded by ``MAX_ACCEPTED_ROTATION = 0.01`` rad.
- The deliberately rejected-trial fixture and the manufactured derivative
  states keep the separated plane ``z = -0.04`` with the same tolerance, so
  the firmly open/closed finite-difference base is unchanged.

Checks in this module:

- the production reduced force/tangent scatter
  (``nonlinear_reduced_assembly.assemble_reduced_system`` through a real
  constraint transformation ``T``) equals the explicit pullback of the
  production full assembly, for every route including an algebraic
  extrapolation compatibility control with negative weights (1.4, -0.4)
  outside the Q4 edge; on-edge quadratic negative-weight coverage already
  exists in the standard transfer tests and is not duplicated here;
- independent central differences of the production reduced internal force
  ``T^T F_int`` (the quantity whose equilibrium against ``T^T F_ext`` forms
  the actual Newton residual) at firmly open and firmly closed contact
  states, in translation, rotation and mixed retained directions that
  activate the lever arm and the shape weights.  The three perturbation
  sizes below are locked before running:

      PERTURBATION_SIZES = (1e-4, 1e-5, 1e-6)
      LENGTH_SCALE       = 1.0    m   (panel edge)
      ROTATION_SCALE    = 1.0    rad (small-rotation scale)

  The scaled tangent error must stay <= 1e-6 at the two smallest adjacent
  sizes.  No evaluation differentiates across a contact branch: every
  perturbed evaluation re-checks the active-pressure count, the minimum-gap
  branch margin and the penalty energy against its base state, and the
  committed state mapping is deep-copied, explicitly restored around every
  +/- evaluation and verified unchanged;
- one public ``NonlinearLoadProgram`` press/release/recontact cycle per
  route on a single model: every accepted increment snapshot is checked for
  its load-stage factors, MPC map, contact branch and equilibrium from the
  actual nonlinear internal + contact force against the staged external load
  (never tangent-times-u); the committed stage endpoints are closed, open
  (exactly zero contact force, positive free-edge gap) and closed again, and
  the final state reproduces the press-only reference;
- actual virtual work through the production ``T`` and the MPC wrench
  transfer about a common origin from the accepted pressed state: the
  per-slave out-of-balance force/couple (including the applied beam load)
  transfers to the masters exactly as the analytic sum
  ``N_i [F, M + r x F]``, with nonzero force and couple required, and the
  nonlinear support-reaction bucket closes global equilibrium without
  double counting;
- a deliberately rejected trial (a cause-coherent terminal failure: gap
  refusal or exhausted stepping, never a completed solve) retains exactly
  the last accepted snapshot/checkpoint state and that state re-evaluates
  as an equilibrium; mid-solve cancellation fires only after one observed
  accepted increment (progress event ``nonlinear_static_step``), leaves the
  captured state owner with no active trial, unchanged generation and
  payload equal to the accepted baseline, and the SAME model then solves
  to the clean reference; callback failure leaves the contact scope
  discarded and a fresh solve reproduces the reference;
- one focused controlled driver-failure pair on the shared route injects
  each terminal cause after an accepted step through the production
  failure surfaces: a refused swept increment ends
  ``contact_gap_exceeded`` with the gap cause carried end to end, and a
  wrong-direction factorization ends ``stopped_at_limit`` /
  ``minimum_load_increment_reached`` without gap evidence; the two
  statuses are distinct and neither claims completion;
- split restart versus uninterrupted solve (rtol 1e-8 / atol 1e-10).  With
  the plane at the reference midsurface the half-load state is already
  closed; the actual branch is inspected and compared with the first
  committed increment of the uninterrupted path instead of claiming a
  closure crossing.  A legitimate aliased weight or offset update rejects
  the checkpoint before any new Newton work.

The finite-rotation MPC restriction is preserved unchanged: the finite
contact path in ``general_contact/static_path.py`` refuses weighted MPC rows,
repeated selector columns and nonzero reference offsets before assembly
("finite shell path requires zero-offset selector constraints without MPC
rows").  This module reviews that guard statically and executes no finite
programme; no finite-rotation attachment development is performed here.

Dimensionless identity residuals are bounded by 1e-10 against the explicit
scales stated below.
"""

import copy

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import (
    BoundaryCondition,
    ContactDefinition,
    ContactInteraction,
    ContactSurface,
    LoadCase,
    RigidContactBody,
    create_shell_element,
    solve_static_nonlinear,
)
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver.elements import BeamElement, CoupledBeamShellElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.core import (
    active_contact_manager,
    contact_scope,
)
from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement
from anyfesolver.nonlinear_performance_bootstrap import (
    clear_nonlinear_assembly_cache,
    get_nonlinear_assembly_plan,
    install_nonlinear_performance_optimizations,
)
from anyfesolver.nonlinear_reduced_assembly import (
    assemble_reduced_system,
    build_reduced_assembly_plan,
)
from anyfesolver.nonlinear_static import (
    NonlinearLoadProgram,
    NonlinearLoadStage,
    _assemble_nonlinear_system,
)
from anyfesolver.nonlinear_state import NonlinearStateStore
from anyfesolver.validation import mpc_constraint_residuals

ATOL = 1.0e-10
TRANSLATION_SCALE = 1.0
ACCEPTED_ROTATION_SCALE = 0.1
# Explicit accepted-rotation bound for the small-rotation lifecycle fixture.
MAX_ACCEPTED_ROTATION = 0.01
FORCE_SCALE = 1.0e3
COUPLE_SCALE = 1.0e3
WORK_SCALE = FORCE_SCALE * TRANSLATION_SCALE

# Locked derivative-test controls (chosen before any run).
PERTURBATION_SIZES = (1.0e-4, 1.0e-5, 1.0e-6)
LENGTH_SCALE = 1.0
ROTATION_SCALE = 1.0
FD_TANGENT_TOL = 1.0e-6
FD_LOOSE_TOL = 1.0e-4

# Prospective stronger solve input for every SUCCESSFUL comparator solve
# (_press_reference, _cycle_result, split restart and the same-model
# re-solve parity): one fixed tighter Newton tolerance through the public
# ``tolerance`` parameter, applied consistently so restart/checkpoint
# identities stay compatible.  This is a stronger solve input, not a
# relaxation of any scientific residual, derivative, parity or gap
# assertion.
SOLVE_TOLERANCE = 1.0e-10

# Successful cycle/restart fixture: rigid plane at the reference shell
# midsurface.  Separated-plane fixture for the rejected trial and the
# manufactured derivative states.
CYCLE_PLANE_Z = 0.0
SEPARATED_PLANE_Z = -0.04
CONTACT_STIFFNESS = 1.0e5
CONTACT_TOLERANCE = 1.0e-3
PRESS_FORCE = 0.1
# Small in-plane edge component so the MPC transfer couple M + r x F is
# nonzero (a purely axial edge load is parallel to the eccentricity r and
# transfers a vanishing couple).
EDGE_INPLANE_FORCE = 0.02
# Release/recontact stage forces: the open endpoint carries a small net
# uplift (+0.002 per node) so the free-edge gap is positive while accepted
# rotations stay far below MAX_ACCEPTED_ROTATION, and the recontact endpoint
# returns to exactly the press-only total load.
RELEASE_FORCE = 0.102
RECONTACT_FORCE = -0.102
RELEASE_INPLANE = -0.04
RECONTACT_INPLANE = 0.04

ROUTES = ("shared", "eccentric", "interpolated")
MPC_ROUTES = ("eccentric", "interpolated")


def _contact(plane_z=CYCLE_PLANE_Z, stiffness=CONTACT_STIFFNESS,
             tolerance=CONTACT_TOLERANCE):
    return ContactDefinition(
        surfaces=(ContactSurface("skin", (1,)),),
        rigid_bodies=(RigidContactBody("floor", "plane", (0, 0, plane_z),
                                       (0, 0, 1)),),
        interactions=(ContactInteraction("support", "skin", "floor",
                                         stiffness, tolerance),),
    )


def _edge_load(model, beam_nodes, per_node_force_z,
               per_node_force_y=EDGE_INPLANE_FORCE):
    load = LoadCase(f"edge_{'press' if per_node_force_z < 0 else 'lift'}")
    for node_id in beam_nodes:
        load.add_nodal_load(int(node_id),
                            forces=(0.0, float(per_node_force_y),
                                    float(per_node_force_z)))
    return load


def _stage_force_vector(model, beam_nodes, per_node_force_y,
                        per_node_force_z):
    """Independent reconstruction of one stage's full external load vector."""
    total = int(model.mesh.dof_manager.total_dofs)
    force = np.zeros(total, dtype=float)
    for node_id in beam_nodes:
        dofs = _node_dofs(model, node_id)
        force[int(dofs[1])] += float(per_node_force_y)
        force[int(dofs[2])] += float(per_node_force_z)
    return force


def _cantilever(route):
    """Qualified Q4 cantilever with a standard beam on the free shell edge.

    The shell root edge (nodes 1, 4) is fully fixed; the beam runs along the
    free edge (nodes 2, 3).  ``shared`` uses the SAME node IDs, the
    eccentric and interpolated routes add offset beam nodes coupled by the
    standard MPC elements with the single common offset ``r``.
    """
    r = np.array([0.0, 0.0, 0.05])
    coords = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0),
              3: (1.0, 1.0, 0.0), 4: (0.0, 1.0, 0.0)}
    couplings = []
    if route == "shared":
        beam_nodes = [2, 3]
    elif route == "eccentric":
        for slave, master in ((10, 2), (11, 3)):
            coords[slave] = tuple(np.asarray(coords[master]) + r)
        beam_nodes = [10, 11]
    elif route in ("interpolated", "algebraic_extrapolated"):
        if route == "interpolated":
            weights_pair = ((0.75, 0.25), (0.25, 0.75))
        else:
            # Algebraic extrapolation compatibility control: the weights do
            # not lie on the Q4 edge; only the exact fixed T law's algebraic
            # consistency for negative weights is asserted here.
            weights_pair = ((1.4, -0.4), (-0.4, 1.4))
        for slave, weights in zip((10, 11), weights_pair):
            attachment = (weights[0] * np.asarray(coords[2])
                          + weights[1] * np.asarray(coords[3]))
            coords[slave] = tuple(attachment + r)
        beam_nodes = [10, 11]
    else:
        raise ValueError(f"unknown route {route!r}")

    model = FEModel(f"edge_beam_contact_{route}")
    model.add_material("steel", 1.0e5, 0.3, density=1.0)
    for node_id, xyz in coords.items():
        model.add_node(int(node_id), *[float(c) for c in xyz])
    model.add_element(1, create_shell_element(1, [1, 2, 3, 4], "steel",
                                              thickness=0.05))
    model.add_element(100, BeamElement(
        100, [int(n) for n in beam_nodes], material_name="steel",
        cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
    ))
    if route == "eccentric":
        for element_id, (slave, master) in ((101, (10, 2)), (102, (11, 3))):
            model.add_element(element_id, CoupledBeamShellElement(
                element_id, beam_node_id=slave, shell_node_id=master,
                material_name="steel", eccentricity=r))
            couplings.append({"slave": slave, "masters": [master],
                              "weights": np.array([1.0]), "r": r})
    elif route in ("interpolated", "algebraic_extrapolated"):
        for element_id, (slave, weights) in zip(
                (101, 102),
                ((10, weights_pair[0]), (11, weights_pair[1]))):
            weights_array = np.asarray(weights, dtype=float)
            model.add_element(element_id, InterpolatedBeamShellMPCElement(
                element_id, slave, [2, 3], weights_array, r))
            couplings.append({"slave": slave, "masters": [2, 3],
                              "weights": weights_array, "r": r})
    model.add_boundary_condition(BoundaryCondition(
        "root", [1, 4],
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")}))
    model.apply_boundary_conditions()
    meta = {"route": route, "beam_nodes": beam_nodes, "couplings": couplings,
            "coords": coords}
    return model, meta


def _production_transform(model):
    total = int(model.mesh.dof_manager.total_dofs)
    K = sparse.eye(total, format="csr")
    zero = np.zeros(total, dtype=float)
    _K, _F, T, u0, independent, info = build_constraint_transformation(
        K, zero, model)
    return T, np.asarray(u0, dtype=float), np.asarray(independent,
                                                      dtype=int), info


def _node_dofs(model, node_id):
    return np.asarray(model.mesh.dof_manager.get_node_dofs(int(node_id)),
                      dtype=int)


def _full_field(model, shell_components):
    """Build a full displacement field from prescribed shell-node dofs."""
    full = np.zeros(int(model.mesh.dof_manager.total_dofs), dtype=float)
    for node_id, components in shell_components.items():
        dofs = _node_dofs(model, node_id)
        for name, value in components.items():
            index = {"ux": 0, "uy": 1, "uz": 2,
                     "rx": 3, "ry": 4, "rz": 5}[name]
            full[dofs[index]] = float(value)
    return full


def _committed_snapshot(committed):
    """Owned deep copy of the committed state mapping."""
    return copy.deepcopy(dict(committed))


def _restore_committed(committed, snapshot):
    """Explicit restore of the committed mapping from an owned snapshot."""
    committed.clear()
    committed.update(copy.deepcopy(snapshot))


def _states_equal(left, right):
    """Exact structural comparison for committed state payloads."""
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return (set(left) == set(right)
                and all(_states_equal(left[key], right[key])
                        for key in left))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return (len(left) == len(right)
                and all(_states_equal(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _reduced_internal_force(model, T, u0, q, committed):
    """Production reduced internal force ``T^T F_int`` at one state.

    Must be called inside an active ``contact_scope`` so the contact force
    is part of the actual residual.  The caller owns ``committed``: it
    restores the baseline mapping around every call and verifies the
    baseline is unchanged.
    """
    u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
    F_int, _K, _states = _assemble_nonlinear_system(
        model, u, committed, 5, tangent=False, kinematics="von_karman")
    return np.asarray(T.T @ F_int, dtype=float).reshape(-1)


def _reduced_equilibrium_in_scope(model, T, u, committed_states, F_ext):
    """Actual nonlinear reduced residual ``T^T (F_int + F_contact - F_ext)``.

    Must be called inside an active ``contact_scope``.  The committed states
    are deep-copied so the accepted snapshot payload is never mutated.
    """
    states = copy.deepcopy(dict(committed_states))
    F_int, _K, _states = _assemble_nonlinear_system(
        model, np.asarray(u, dtype=float).reshape(-1), states, 5,
        tangent=False, kinematics="von_karman")
    residual = np.asarray(F_int, dtype=float).reshape(-1) - F_ext
    return np.asarray(T.T @ residual, dtype=float).reshape(-1)


def _assert_reduced_equilibrium(residual, T, F_ext):
    """The production convergence criterion, re-derived independently."""
    reference = max(float(np.linalg.norm(np.asarray(
        T.T @ F_ext, dtype=float).reshape(-1))), 1.0)
    assert float(np.linalg.norm(residual)) <= 1.0e-6 * reference


def _max_abs_rotation(model, u):
    u = np.asarray(u, dtype=float).reshape(-1)
    worst = 0.0
    for node_id in model.mesh.nodes:
        dofs = _node_dofs(model, node_id)
        if dofs.size >= 6:
            worst = max(worst, float(np.max(np.abs(u[dofs[3:6]]))))
    return worst


def _assert_accepted_coupling_kinematics(model, meta, displacements):
    """The accepted state satisfies the MPC map and constraint residuals."""
    u = np.asarray(displacements, dtype=float)
    residuals = mpc_constraint_residuals(model, u)
    max_residual = max((abs(v) for v in residuals.values()), default=0.0)
    assert max_residual <= ATOL
    for coupling in meta["couplings"]:
        slave = coupling["slave"]
        weights = np.asarray(coupling["weights"], dtype=float)
        r = np.asarray(coupling["r"], dtype=float)
        u_sum = np.zeros(3, dtype=float)
        theta_sum = np.zeros(3, dtype=float)
        for master, weight in zip(coupling["masters"], weights):
            dofs = _node_dofs(model, master)
            u_sum = u_sum + float(weight) * u[dofs[:3]]
            theta_sum = theta_sum + float(weight) * u[dofs[3:6]]
        expected_u = u_sum + np.cross(theta_sum, r)
        expected_theta = theta_sum
        slave_dofs = _node_dofs(model, slave)
        got_u = u[slave_dofs[:3]]
        got_theta = u[slave_dofs[3:6]]
        assert np.linalg.norm(got_u - expected_u) / TRANSLATION_SCALE <= ATOL
        assert (np.linalg.norm(got_theta - expected_theta)
                / ACCEPTED_ROTATION_SCALE <= ATOL)


def _assert_contact_branch(manager, u, branch, base=None):
    """Every perturbed evaluation stays firmly inside its contact branch.

    Returns the base evidence ``(state, pressure mask)``.  Compare the
    actual weighted-gap active mask: the nodal-force support can stay
    unchanged when a different pressure degree of freedom becomes active.
    Every pressure gap must stay strictly away from first touch.
    """
    force, _tangent, state = manager.evaluate(
        np.asarray(u, dtype=float).reshape(-1), tangent=False)
    plane_shift = -float((manager.point - manager.reference_point)
                         @ manager.normal)
    pressure_gap = (manager.reference_weighted_gap
                    + plane_shift * manager.support
                    + np.asarray(manager.gap_gradient @ u).reshape(-1)
                    ) / manager.support
    assert np.min(np.abs(pressure_gap)) > 1.0e-3
    mask = np.flatnonzero(pressure_gap <= 0.0)
    if branch == "open":
        assert state.active_pressure_dofs == 0
        assert state.minimum_gap > 0.02
        assert state.penalty_energy == 0.0
        np.testing.assert_allclose(state.shell_force, 0.0, atol=1.0e-12)
    else:
        assert state.active_pressure_dofs > 0
        assert state.minimum_gap < -0.005
        assert state.penalty_energy > 0.0
        assert mask.size > 0
    if base is not None:
        base_state, base_mask = base
        assert state.active_pressure_dofs == base_state.active_pressure_dofs
        np.testing.assert_array_equal(mask, base_mask)
    return state, mask


_PRESS_CACHE = {}
_CYCLE_CACHE = {}


def _press_reference(route):
    """Cached press-only public solve per route (deterministic reference)."""
    if route not in _PRESS_CACHE:
        model, meta = _cantilever(route)
        result = solve_static_nonlinear(
            model, _edge_load(model, meta["beam_nodes"], -PRESS_FORCE),
            contact=_contact(CYCLE_PLANE_Z), num_steps=4, max_iterations=20,
            kinematics="von_karman", tolerance=SOLVE_TOLERANCE,
        )
        _PRESS_CACHE[route] = (model, meta, result)
    return _PRESS_CACHE[route]


def _cycle_program(model, meta):
    """One public three-stage press/release/recontact load program."""
    stage_forces = (
        ("press", EDGE_INPLANE_FORCE, -PRESS_FORCE),
        ("release", RELEASE_INPLANE, RELEASE_FORCE),
        ("recontact", RECONTACT_INPLANE, RECONTACT_FORCE),
    )
    stages = []
    for name, per_node_y, per_node_z in stage_forces:
        load = LoadCase(name)
        for node_id in meta["beam_nodes"]:
            load.add_nodal_load(int(node_id),
                                forces=(0.0, float(per_node_y),
                                        float(per_node_z)))
        stages.append(NonlinearLoadStage(name, load, 1.0))
    return NonlinearLoadProgram(tuple(stages)), dict(
        (name, (per_node_y, per_node_z))
        for name, per_node_y, per_node_z in stage_forces)


def _cycle_result(route):
    """Cached single-model press/release/recontact cycle per route."""
    if route not in _CYCLE_CACHE:
        model, meta = _cantilever(route)
        program, stage_forces = _cycle_program(model, meta)
        result = solve_static_nonlinear(
            model, load_program=program, contact=_contact(CYCLE_PLANE_Z),
            num_steps=6, max_iterations=20, kinematics="von_karman",
            record_increment_snapshots=True, tolerance=SOLVE_TOLERANCE,
        )
        _CYCLE_CACHE[route] = (model, meta, program, stage_forces, result)
    return _CYCLE_CACHE[route]


def _snapshot_at(result, load_factor):
    for snapshot in result.snapshots:
        if abs(snapshot.load_factor - load_factor) <= 1.0e-9:
            return snapshot
    raise AssertionError(
        f"no accepted snapshot committed at load factor {load_factor}; "
        f"committed factors: "
        f"{[float(s.load_factor) for s in result.snapshots]}")


# ---------------------------------------------------------------------------
# Production reduced force/tangent versus full assembly with the real T
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ROUTES + ("algebraic_extrapolated",))
def test_reduced_scatter_matches_full_assembly_pullback(route):
    model, meta = _cantilever(route)
    T, u0, independent, info = _production_transform(model)
    if route == "shared":
        assert info["num_mpc_slave_dofs"] == 0
    else:
        assert info["num_mpc_slave_dofs"] == 12

    field = _full_field(model, {
        2: {"uz": -0.04, "ry": 3.0e-3, "ux": 1.0e-3},
        3: {"uz": -0.02, "rx": -2.0e-3, "rz": 5.0e-4},
    })
    q = field[independent]
    u = np.asarray(T @ q + u0, dtype=float).reshape(-1)

    install_nonlinear_performance_optimizations()
    clear_nonlinear_assembly_cache(model)
    plan = get_nonlinear_assembly_plan(model, 5)
    f_full, K_full, _states = plan.assemble(u, {}, tangent=True)

    # Cross-check the batched production assembly against the reference
    # production assembler at the same nonlinear state (the repository's
    # demonstrated plan-versus-reference tolerances).
    f_ref, K_ref, _ref_states = _assemble_nonlinear_system(
        model, u, {}, 5, tangent=True, kinematics="von_karman")
    np.testing.assert_allclose(f_full, f_ref, rtol=1.0e-11, atol=2.0e-5)
    np.testing.assert_allclose(K_full.toarray(), np.asarray(K_ref.todense()),
                              rtol=1.0e-11, atol=3.0e-5)

    # Production reduced scatter against the explicit pullback of the same
    # full assembly through the real constraint transformation.
    reduced_plan = build_reduced_assembly_plan(plan, T)
    f_red, K_red, _red_states = assemble_reduced_system(
        plan, reduced_plan, u, {}, tangent=True)
    f_pullback = np.asarray(T.T @ f_full, dtype=float).reshape(-1)
    K_pullback = (T.T @ K_full @ T).toarray()

    force_error = float(np.linalg.norm(f_red - f_pullback))
    assert force_error / FORCE_SCALE <= ATOL
    assert force_error / max(float(np.linalg.norm(f_pullback)), 1.0e-30) <= ATOL
    tangent_error = float(np.linalg.norm((K_red.toarray() - K_pullback),
                                        ord="fro"))
    tangent_scale = max(float(np.max(np.abs(K_pullback))), 1.0e-30)
    assert tangent_error / (tangent_scale * max(K_pullback.shape)) <= ATOL

    # The reduced pullback is the exact virtual-work projection of the full
    # assembly: for arbitrary retained fields the full internal work equals
    # the reduced contraction.
    rng = np.random.default_rng(20261016)
    for _ in range(3):
        d = rng.uniform(-1.0, 1.0, int(T.shape[1]))
        full_work = float(f_full @ np.asarray(T @ d).reshape(-1))
        reduced_work = float(f_pullback @ d)
        assert abs(full_work - reduced_work) / WORK_SCALE <= ATOL


# ---------------------------------------------------------------------------
# Independent central differences of the production residual
# ---------------------------------------------------------------------------

def _reduced_direction(model, independent, entries):
    direction = np.zeros(int(independent.size), dtype=float)
    for node_id, name, value in entries:
        dof = int(_node_dofs(model, node_id)[
            {"ux": 0, "uy": 1, "uz": 2, "rx": 3, "ry": 4, "rz": 5}[name]])
        positions = np.flatnonzero(independent == dof)
        assert positions.size == 1, f"dof {dof} is not retained"
        direction[int(positions[0])] = float(value)
    return direction


@pytest.mark.parametrize("route", ROUTES)
def test_central_differences_of_production_reduced_residual(route):
    model, meta = _cantilever(route)
    T, u0, independent, info = _production_transform(model)
    contact = _contact(SEPARATED_PLANE_Z)
    committed = {}
    baseline = _committed_snapshot(committed)

    open_field = _full_field(model, {
        2: {"uz": -0.01, "ry": 1.0e-3},
        3: {"uz": -0.005, "rx": -8.0e-4},
    })
    # Manufactured closed residual state (a local state, not an accepted
    # solve): the free edge is pressed firmly onto the separated plane so
    # the mortar weighted pressure is active there.  The fixed root nodes
    # stay at their reference configuration (their dofs are removed by the
    # root constraint, so prescribing them would have no effect).
    closed_field = _full_field(model, {
        2: {"uz": -0.09, "ry": 2.0e-3},
        3: {"uz": -0.09, "rx": -1.5e-3},
    })

    directions = (
        ("translation", _reduced_direction(model, independent,
                                           ((2, "uz", 1.0),))),
        ("rotation", _reduced_direction(model, independent,
                                        ((3, "ry", 1.0),))),
        ("mixed", _reduced_direction(model, independent,
                                     ((2, "uz", 1.0), (3, "ry", 1.0),
                                      (3, "ux", 1.0)))),
    )

    with contact_scope(model, contact) as manager:
        for state_name, base_field in (("open", open_field),
                                       ("closed", closed_field)):
            q_base = base_field[independent]
            u_base = np.asarray(T @ q_base + u0, dtype=float).reshape(-1)
            base_branch = _assert_contact_branch(manager, u_base, state_name,
                                                 None)
            _restore_committed(committed, baseline)
            F_int, K_full, _states = _assemble_nonlinear_system(
                model, u_base, committed, 5, tangent=True,
                kinematics="von_karman")
            K_red = np.asarray((T.T @ K_full @ T).todense())
            full_reduced_force = np.asarray(T.T @ F_int).reshape(-1).copy()
            assert _states_equal(_committed_snapshot(committed), baseline)

            # Contact-inclusive projection parity at BOTH base states.
            # The direct reduced assembler owns structural scatter; the
            # contact manager supplies its full-coordinate force/tangent,
            # composed through the very same production transformation.
            structural_plan = get_nonlinear_assembly_plan(model, 5)
            reduced_plan = build_reduced_assembly_plan(structural_plan, T)
            structural_force, structural_tangent, _ = assemble_reduced_system(
                structural_plan, reduced_plan, u_base,
                copy.deepcopy(baseline), tangent=True)
            contact_force, contact_tangent, _ = manager.evaluate(u_base)
            composed_force = (structural_force.copy()
                              + np.asarray(T.T @ contact_force).reshape(-1))
            composed_tangent = (structural_tangent
                                + T.T @ contact_tangent @ T).toarray()
            assert (np.linalg.norm(composed_force - full_reduced_force)
                    / max(np.linalg.norm(full_reduced_force), 1.0)
                    <= ATOL)
            assert (np.linalg.norm(composed_tangent - K_red)
                    / max(np.linalg.norm(K_red), 1.0) <= ATOL)

            for name, direction in directions:
                target = K_red @ direction
                target_scale = max(float(np.linalg.norm(target)), 1.0e-30)
                errors = []
                for size in PERTURBATION_SIZES:
                    step = size * LENGTH_SCALE * direction
                    _restore_committed(committed, baseline)
                    plus_u = np.asarray(
                        T @ (q_base + step) + u0, dtype=float).reshape(-1)
                    _assert_contact_branch(
                        manager, plus_u, state_name, base_branch)
                    plus = _reduced_internal_force(
                        model, T, u0, q_base + step, committed)
                    _restore_committed(committed, baseline)
                    minus_u = np.asarray(
                        T @ (q_base - step) + u0, dtype=float).reshape(-1)
                    _assert_contact_branch(
                        manager, minus_u, state_name, base_branch)
                    minus = _reduced_internal_force(
                        model, T, u0, q_base - step, committed)
                    finite_difference = (plus - minus) / (2.0 * size
                                                          * LENGTH_SCALE)
                    errors.append(float(np.linalg.norm(
                        finite_difference - target)) / target_scale)
                    # The committed baseline survived the +/- evaluations.
                    assert _states_equal(_committed_snapshot(committed),
                                         baseline)
                # No differentiation across a contact branch: every
                # perturbed evaluation was checked against the base branch.
                assert errors[1] <= FD_TANGENT_TOL, (route, state_name,
                                                     name, errors)
                assert errors[2] <= FD_TANGENT_TOL, (route, state_name,
                                                      name, errors)
                assert errors[0] <= FD_LOOSE_TOL, (route, state_name, name,
                                                   errors)

    _restore_committed(committed, baseline)
    assert _states_equal(_committed_snapshot(committed), baseline)


# ---------------------------------------------------------------------------
# One public press/release/recontact load-program cycle per route
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ROUTES)
def test_press_release_recontact_cycle_through_public_load_program(route):
    model, meta, program, stage_forces, result = _cycle_result(route)
    assert result.status == "completed"
    assert program.total_factor == pytest.approx(3.0)
    assert len(result.snapshots) >= 3

    contact = _contact(CYCLE_PLANE_Z)
    T, u0, independent, info = _production_transform(model)
    stage_vectors = {
        name: _stage_force_vector(model, meta["beam_nodes"],
                                  per_node_y, per_node_z)
        for name, (per_node_y, per_node_z) in stage_forces.items()
    }

    with contact_scope(model, contact) as manager:
        for snapshot in result.snapshots:
            u = np.asarray(snapshot.displacements, dtype=float)
            factors = program.stage_factors(snapshot.load_factor)

            # Load-stage factors of the accepted increment.
            for value in factors.values():
                assert -1.0e-12 <= value <= 1.0 + 1.0e-12
            assert abs(sum(factors.values()) - snapshot.load_factor) <= 1.0e-9

            # Accepted kinematics: MPC map and explicit rotation bound.
            _assert_accepted_coupling_kinematics(model, meta, u)
            assert _max_abs_rotation(model, u) <= MAX_ACCEPTED_ROTATION

            # Contact branch consistency at the accepted state.
            state = manager.evaluate(u, tangent=False)[2]
            assert state.gap_tolerance_met
            if state.active_pressure_dofs > 0:
                assert state.minimum_gap < 0.0
                assert state.penalty_energy > 0.0
            else:
                assert state.penalty_energy == 0.0
                np.testing.assert_allclose(state.shell_force, 0.0,
                                           atol=1.0e-12)

            # Equilibrium from the actual nonlinear internal + contact force
            # against the staged external load (never tangent-times-u).
            F_ext = np.zeros(
                int(model.mesh.dof_manager.total_dofs), dtype=float)
            for name, value in factors.items():
                F_ext = F_ext + value * stage_vectors[name]
            residual = _reduced_equilibrium_in_scope(
                model, T, u, snapshot.element_states, F_ext)
            _assert_reduced_equilibrium(residual, T, F_ext)

        # Committed stage endpoints: closed, open, closed.
        pressed = _snapshot_at(result, 1.0)
        opened = _snapshot_at(result, 2.0)
        recontacted = _snapshot_at(result, 3.0)

        press_state = manager.evaluate(
            np.asarray(pressed.displacements, dtype=float),
            tangent=False)[2]
        assert press_state.active_pressure_dofs > 0
        assert press_state.minimum_gap < 0.0
        assert press_state.shell_force[2] > 0.0

        open_u = np.asarray(opened.displacements, dtype=float)
        open_state = manager.evaluate(open_u, tangent=False)[2]
        assert open_state.active_pressure_dofs == 0
        assert open_state.penalty_energy == 0.0
        np.testing.assert_allclose(open_state.shell_force, 0.0, atol=1.0e-12)
        # The root lies on the plane: the open state may touch there with
        # zero pressure, but the free edge must be clear of the plane.
        assert open_state.minimum_gap >= 0.0
        for node_id in (2, 3):
            free_edge_gap = float(open_u[_node_dofs(model, node_id)[2]])
            assert free_edge_gap > 1.0e-4

        recontact_state = manager.evaluate(
            np.asarray(recontacted.displacements, dtype=float),
            tangent=False)[2]
        assert recontact_state.active_pressure_dofs > 0
        assert recontact_state.shell_force[2] > 0.0

    # Global z equilibrium from the actual nonlinear support-reaction bucket:
    # support + physical contact force balance the staged applied load
    # (2 * (-PRESS_FORCE + RELEASE_FORCE + RECONTACT_FORCE) = -0.2), so the
    # MPC transfer is not double counted as an extra external load.
    final_step = result.steps[-1]
    support_z = float(final_step.support_reactions["root"][2])
    contact_z = float(result.info["general_contact"]["shell_force"][2])
    assert support_z + contact_z == pytest.approx(2.0 * PRESS_FORCE,
                                                  abs=1.0e-6)

    # Recontact consistency: the cycle's final state reproduces the
    # press-only public reference on a fresh model (elastic path
    # independence of the admitted profile).
    _reference_model, _reference_meta, reference = _press_reference(route)
    assert reference.status == "completed"
    np.testing.assert_allclose(result.displacements,
                               reference.displacements,
                               rtol=1.0e-8, atol=1.0e-10)
    assert result.info["general_contact"]["shell_force"] == pytest.approx(
        reference.info["general_contact"]["shell_force"], abs=1.0e-8)


# ---------------------------------------------------------------------------
# Virtual work and MPC wrench transfer at the accepted pressed state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", MPC_ROUTES)
def test_accepted_state_virtual_work_and_wrench_transfer(route):
    model, meta, result = _press_reference(route)
    assert result.status == "completed"
    u = np.asarray(result.displacements, dtype=float)
    T, u0, independent, info = _production_transform(model)
    F_ext = _stage_force_vector(model, meta["beam_nodes"],
                                EDGE_INPLANE_FORCE, -PRESS_FORCE)

    with contact_scope(model, _contact(CYCLE_PLANE_Z)):
        states = copy.deepcopy(dict(result.element_states))
        F_int, _K, _states = _assemble_nonlinear_system(
            model, u, states, 5, tangent=False, kinematics="von_karman")

    # Actual accepted nonlinear equilibrium: the reduced residual of the
    # true internal + contact force against the applied load satisfies the
    # production convergence criterion.
    residual = np.asarray(T.T @ (np.asarray(F_int, dtype=float).reshape(-1)
                                 - F_ext), dtype=float).reshape(-1)
    _assert_reduced_equilibrium(residual, T, F_ext)

    # Actual virtual work through the production transformation: the full
    # internal work on admissible variations equals the reduced contraction.
    reduced_internal = np.asarray(T.T @ F_int, dtype=float).reshape(-1)
    rng = np.random.default_rng(4242)
    for _ in range(3):
        d = rng.uniform(-1.0, 1.0, int(T.shape[1]))
        full_work = float(F_int @ np.asarray(T @ d).reshape(-1))
        reduced_work = float(reduced_internal @ d)
        assert abs(full_work - reduced_work) / WORK_SCALE <= ATOL

    # Per-slave out-of-balance force/couple from the true nonlinear residual
    # (including the applied beam load; contact acts only on the shell, so
    # the slave rows are contact-free).
    out_of_balance = (np.asarray(F_int, dtype=float).reshape(-1) - F_ext)
    transferred = np.asarray(T.T @ out_of_balance, dtype=float).reshape(-1)

    origin = np.array([0.7, -0.15, 0.35])
    total_master_force = np.zeros(3, dtype=float)
    total_master_couple = np.zeros(3, dtype=float)
    master_nodes = sorted({master for coupling in meta["couplings"]
                           for master in coupling["masters"]})
    for master in master_nodes:
        dofs = _node_dofs(model, master)
        positions = np.array([int(np.flatnonzero(independent == dof)[0])
                              for dof in dofs], dtype=int)
        # Production T mapping: the constraint contribution at this master
        # is the reduced residual minus the master's own residual rows.
        production_force = transferred[positions[:3]] - out_of_balance[
            dofs[:3]]
        production_couple = transferred[positions[3:6]] - out_of_balance[
            dofs[3:6]]

        # Independent analytic transfer N_i [F, M + r x F] summed over the
        # slaves attached to this master.
        analytic_force = np.zeros(3, dtype=float)
        analytic_couple = np.zeros(3, dtype=float)
        for coupling in meta["couplings"]:
            weights = np.asarray(coupling["weights"], dtype=float)
            if master not in coupling["masters"]:
                continue
            weight = float(weights[coupling["masters"].index(master)])
            slave_dofs = _node_dofs(model, coupling["slave"])
            f_s = out_of_balance[slave_dofs[:3]]
            m_s = out_of_balance[slave_dofs[3:6]]
            r = np.asarray(coupling["r"], dtype=float)
            analytic_force = analytic_force + weight * f_s
            analytic_couple = analytic_couple + weight * (m_s
                                                          + np.cross(r, f_s))

        np.testing.assert_allclose(production_force, analytic_force,
                                   rtol=1.0e-9, atol=1.0e-12)
        np.testing.assert_allclose(production_couple, analytic_couple,
                                   rtol=1.0e-9, atol=1.0e-12)
        total_master_force = total_master_force + production_force
        total_master_couple = (total_master_couple + production_couple
                               + np.cross(
                                   np.asarray(meta["coords"][master],
                                              dtype=float) - origin,
                                   production_force))

    # Common-origin wrench of the slave side: the transfer preserves the
    # wrench about any common origin, and both force and couple are
    # nonzero, so the identity is not vacuous.
    total_slave_force = np.zeros(3, dtype=float)
    total_slave_couple = np.zeros(3, dtype=float)
    for coupling in meta["couplings"]:
        slave_dofs = _node_dofs(model, coupling["slave"])
        f_s = out_of_balance[slave_dofs[:3]]
        m_s = out_of_balance[slave_dofs[3:6]]
        weights = np.asarray(coupling["weights"], dtype=float)
        r = np.asarray(coupling["r"], dtype=float)
        attachment = sum(
            float(w) * np.asarray(meta["coords"][m], dtype=float)
            for m, w in zip(coupling["masters"], weights))
        x_s = attachment + r
        assert np.linalg.norm(f_s) > 1.0e-3
        assert np.linalg.norm(m_s + np.cross(r, f_s)) > 1.0e-4
        total_slave_force = total_slave_force + f_s
        total_slave_couple = (total_slave_couple + m_s
                              + np.cross(x_s - origin, f_s))
    assert np.linalg.norm(total_slave_force) > 1.0e-3
    assert np.linalg.norm(total_slave_couple) > 1.0e-4
    np.testing.assert_allclose(total_master_force, total_slave_force,
                               rtol=1.0e-9, atol=1.0e-12)
    np.testing.assert_allclose(total_master_couple, total_slave_couple,
                               rtol=1.0e-9, atol=1.0e-12)

    # Actual nonlinear result buckets close global z equilibrium; the
    # support bucket contains only the fixed root nodes, disjoint from the
    # MPC master/slave nodes, so the transferred wrench is not counted twice.
    support_z = float(result.steps[-1].support_reactions["root"][2])
    contact_z = float(result.info["general_contact"]["shell_force"][2])
    assert support_z + contact_z == pytest.approx(2.0 * PRESS_FORCE,
                                                  abs=1.0e-6)
    support_nodes = {1, 4}
    mpc_nodes = set(meta["beam_nodes"]) | set(master_nodes)
    assert support_nodes.isdisjoint(mpc_nodes)


# ---------------------------------------------------------------------------
# Rejected trials, cancellation and callback failure preserve accepted state
# ---------------------------------------------------------------------------

# The documented finalization seal of a qualified Q4 state
# (_seal_final_qualified_q4_states): the finalized result adds exactly the
# committed-displacement binding and its integrity digest to the sealed
# element state; every physical history field is unchanged.
Q4_SEALED_STATE_METADATA_KEYS = ("qualified_q4_committed_binding",
                                 "state_integrity_sha256")

# The deliberately rejected trial terminates through the production
# cause-mapped failure branch: a gap refusal is ``contact_gap_exceeded``
# end to end, and exhausted stepping after an accepted step is
# ``stopped_at_limit`` with ``minimum_load_increment_reached``.  The
# former per-route tuple was an observation of one incidental historical
# route (shared once ended ``stopped_at_limit``/``line_search_failed``),
# not a mandated status, and is deliberately not asserted: the terminal
# cause itself carries the oracle.
TERMINAL_FAILURE_STATUSES = ("contact_gap_exceeded", "stopped_at_limit")


def _assert_finalized_states_match_snapshot(model, result_states,
                                            snapshot_states, displacements):
    """Exact physical state comparison with explicit documented seal metadata.

    The finalized result's qualified Q4 states are sealed: every physical
    history field must equal the last accepted snapshot's exactly, and
    exactly the documented binding/digest metadata may be added.  Arbitrary
    key differences fail; the added metadata is checked explicitly.
    """
    assert set(result_states) == set(snapshot_states)
    for element_id, final_state in result_states.items():
        snapshot_state = snapshot_states[element_id]
        assert isinstance(final_state, dict)
        assert isinstance(snapshot_state, dict)
        extra = set(final_state) - set(snapshot_state)
        missing = set(snapshot_state) - set(final_state)
        assert extra <= set(Q4_SEALED_STATE_METADATA_KEYS), (element_id, extra)
        assert not missing, (element_id, missing)
        for key in sorted(set(final_state) & set(snapshot_state)):
            assert _states_equal(final_state[key], snapshot_state[key]), \
                (element_id, key)
        if "qualified_q4_committed_binding" in extra:
            binding = final_state["qualified_q4_committed_binding"]
            assert isinstance(binding, dict)
            element = model.mesh.elements[int(element_id)]
            dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=int)
            np.testing.assert_array_equal(
                np.asarray(binding["committed_total_u"], dtype=float),
                np.asarray(displacements, dtype=float)[dofs])
        if "state_integrity_sha256" in extra:
            digest = final_state["state_integrity_sha256"]
            assert isinstance(digest, str) and digest
            element = model.mesh.elements[int(element_id)]
            dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=int)
            validated = element.validate_committed_current_tangent_state(
                model.mesh, model.get_material(element.material_name),
                np.asarray(displacements, dtype=float)[dofs], final_state, 5)
            assert validated == digest


@pytest.mark.parametrize("route", ROUTES)
def test_rejected_trial_keeps_last_accepted_state(route):
    model, meta = _cantilever(route)
    load = _edge_load(model, meta["beam_nodes"], -PRESS_FORCE)
    contact = _contact(SEPARATED_PLANE_Z, stiffness=1.0e7)
    result = solve_static_nonlinear(
        model, load, contact=contact,
        num_steps=3, max_iterations=20, kinematics="von_karman",
        record_increment_snapshots=True, emit_restart_checkpoint=True,
    )
    # Cause-coherent terminal failure contract: the public status and both
    # failure-reason fields must agree with the actual terminal cause, and
    # neither failure may claim completion.  A gap refusal is
    # ``contact_gap_exceeded`` end to end; exhausted stepping after an
    # accepted step is ``stopped_at_limit`` with the exhaustion reason and
    # an iteration-level Newton/line-search cause, never a hidden gap
    # refusal.  A partial retained accepted step is not a completed solve.
    assert result.status in TERMINAL_FAILURE_STATUSES
    assert result.status != "completed"
    assert result.failure_reason == result.info["failure_reason"]
    assert 0.0 < result.load_factor < 1.0
    assert result.info["general_contact"]["gap_tolerance_met"]
    iteration_reason = result.info["first_failed_iteration_reason"]
    if result.status == "contact_gap_exceeded":
        assert result.failure_reason == "contact_gap_exceeded"
        assert iteration_reason == "contact_gap_exceeded"
        assert result.converged is False
    else:
        assert result.failure_reason == "minimum_load_increment_reached"
        assert isinstance(iteration_reason, str) and iteration_reason
        assert iteration_reason != "contact_gap_exceeded"
        # Retained-step convergence semantics are not an independent
        # successful-endpoint claim: the status above carries the oracle.
        assert result.converged is True

    # Evidence of the actual rejected trial and cutbacks: a trial beyond the
    # retained accepted load factor was attempted and rejected, and the
    # increment was cut back before the abort.
    adaptation = result.info["convergence_adaptation"]
    assert adaptation
    assert any(entry["action"] == "cutback_after_nonconvergence"
               for entry in adaptation)
    assert result.info["first_failed_load_factor"] > result.load_factor
    assert result.info["first_failed_step_size"] > 0.0

    # The retained state is exactly the last accepted snapshot: no rejected
    # trial state was admitted.
    assert result.snapshots
    last = result.snapshots[-1]
    assert result.load_factor == last.load_factor
    np.testing.assert_allclose(result.displacements, last.displacements,
                               rtol=0.0, atol=0.0)
    _assert_finalized_states_match_snapshot(
        model, result.element_states, last.element_states,
        result.displacements)
    assert result.restart_checkpoint is not None
    assert result.restart_checkpoint["version"] == 2
    np.testing.assert_allclose(
        np.asarray(result.restart_checkpoint["displacements"], dtype=float),
        last.displacements, rtol=1.0e-12, atol=1.0e-15)

    # The retained accepted state still satisfies the coupling map exactly.
    _assert_accepted_coupling_kinematics(model, meta, result.displacements)

    # Re-evaluated equilibrium of the retained state against the retained
    # load factor, from the actual nonlinear internal + contact force.
    T, u0, independent, info = _production_transform(model)
    F_ext = result.load_factor * _stage_force_vector(
        model, meta["beam_nodes"], EDGE_INPLANE_FORCE, -PRESS_FORCE)
    with contact_scope(model, contact) as manager:
        state = manager.evaluate(
            np.asarray(result.displacements, dtype=float), tangent=False)[2]
        assert state.gap_tolerance_met
        residual = _reduced_equilibrium_in_scope(
            model, T, result.displacements, last.element_states, F_ext)
    _assert_reduced_equilibrium(residual, T, F_ext)

    # Terminal-failure checkpoints retain the accepted state but are not
    # continuable targets under the existing public restart contract.
    # Preserve that refusal; completed-target recovery is checked below
    # on all three routes by test_split_restart_matches_uninterrupted.
    def fail_if_newton_reached(_message):
        raise AssertionError("terminal checkpoint refusal must precede Newton")

    with pytest.raises(ValueError, match="continuable completed static target"):
        solve_static_nonlinear(
            model, load, contact=contact, num_steps=3, max_iterations=20,
            kinematics="von_karman", max_load_factor=result.load_factor,
            restart_checkpoint=result.restart_checkpoint,
            status_callback=fail_if_newton_reached,
        )


def test_terminal_failure_causes_are_distinct_and_cause_coherent(
        monkeypatch):
    """One controlled driver, two injected terminal causes.

    Both scenarios use the standard completing shared-route fixture and
    let exactly one increment accept; the failure injected through a
    production failure surface after that accepted step then drives the
    solver to its terminal state.  A gap refusal must terminate as
    ``contact_gap_exceeded`` with the gap cause carried end to end; a
    Newton failure must exhaust stepping and terminate as
    ``stopped_at_limit``/``minimum_load_increment_reached`` with no
    gap-refusal evidence.  The two terminal statuses are distinct, neither
    claims completion, and both retain exactly the accepted step.  No
    physical parameter combination is searched.
    """
    import anyfesolver.nonlinear_static as nonlinear_static

    def solve_with_injection(inject_after_accepted_step):
        model, meta = _cantilever("shared")
        load = _edge_load(model, meta["beam_nodes"], -PRESS_FORCE)
        armed = []

        def on_progress(event):
            if (event.event_type == "nonlinear_static_step"
                    and not armed):
                armed.append(True)
                inject_after_accepted_step()

        return solve_static_nonlinear(
            model, load, contact=_contact(CYCLE_PLANE_Z), num_steps=4,
            max_iterations=20, kinematics="von_karman",
            progress_callback=on_progress, record_increment_snapshots=True,
        )

    def assert_retained_accepted_step(result):
        # Terminal cause AND retained-state proof carry the oracle: the
        # partial accepted step is retained exactly and never claimed
        # completed.
        assert result.status != "completed"
        assert 0.0 < result.load_factor < 1.0
        assert result.snapshots
        last = result.snapshots[-1]
        assert result.load_factor == last.load_factor
        np.testing.assert_allclose(result.displacements, last.displacements,
                                   rtol=0.0, atol=0.0)
        assert any(entry["action"] == "cutback_after_nonconvergence"
                   for entry in result.info["convergence_adaptation"])
        assert result.info["first_failed_load_factor"] > result.load_factor
        assert result.info["first_failed_step_size"] > 0.0

    # Gap refusal injected on the production contact-admission surface:
    # every swept increment after the accepted step is refused.
    def refuse_swept_increments():
        manager = active_contact_manager()
        assert manager is not None
        manager.swept_increment_admissible = lambda *args, **kwargs: False

    gap = solve_with_injection(refuse_swept_increments)
    assert gap.status == "contact_gap_exceeded"
    assert gap.failure_reason == "contact_gap_exceeded"
    assert gap.info["failure_reason"] == "contact_gap_exceeded"
    assert gap.info["first_failed_iteration_reason"] == "contact_gap_exceeded"
    assert gap.converged is False
    assert_retained_accepted_step(gap)

    # Newton failure injected on the production factorization surface: the
    # wrong-direction handle makes every post-acceptance trial diverge, so
    # stepping exhausts the minimum load increment after the accepted
    # step.  Established failure-injection handle shape.
    real_factorize = nonlinear_static.factorize

    class _WrongDirectionHandle:
        status = "ok"
        backend_name = "slice3-wrong-direction"
        failure_reason = None

        @staticmethod
        def solve(residual):
            return -np.asarray(residual, dtype=float)

    exhausting = []

    def gated_factorize(*args, **kwargs):
        if exhausting:
            return _WrongDirectionHandle()
        return real_factorize(*args, **kwargs)

    monkeypatch.setattr(nonlinear_static, "factorize", gated_factorize)

    def arm_wrong_direction():
        exhausting.append(True)

    exhausted = solve_with_injection(arm_wrong_direction)
    assert exhausted.status == "stopped_at_limit"
    assert exhausted.failure_reason == "minimum_load_increment_reached"
    assert (exhausted.info["failure_reason"]
            == "minimum_load_increment_reached")
    exhausted_iteration_reason = exhausted.info["first_failed_iteration_reason"]
    assert (isinstance(exhausted_iteration_reason, str)
            and exhausted_iteration_reason)
    assert exhausted_iteration_reason != "contact_gap_exceeded"
    # Retained-step convergence semantics hold, yet the solve is not a
    # completed endpoint.
    assert exhausted.converged is True
    assert_retained_accepted_step(exhausted)

    # The two terminal causes are distinct public statuses.
    assert gap.status != exhausted.status


@pytest.mark.parametrize("route", ROUTES)
def test_cancellation_after_accepted_step_preserves_committed_state(
        route, monkeypatch):
    _reference_model, _reference_meta, reference = _press_reference(route)
    assert reference.status == "completed"

    import anyfesolver.nonlinear_static as nonlinear_static
    original_commit = nonlinear_static._commit_nonlinear_state_candidate
    commits = []

    def wrapped_commit(*args, **kwargs):
        # Compatible instrumentation of the original production commit: the
        # original runs first and its RETURN value is the committed state
        # owner (the persistent NonlinearStateStore or the legacy mapping
        # fallback).  args[0] is only the pre-commit container and must not
        # be mistaken for the accepted state.
        committed = original_commit(*args, **kwargs)
        owner = committed
        commits.append({
            "owner": owner,
            "backend": type(owner).__name__,
            "generation": getattr(owner, "generation", None),
            "has_trial": getattr(owner, "has_active_trial", None),
            "payload": {int(key): copy.deepcopy(value)
                        for key, value in owner.items()},
        })
        return committed

    monkeypatch.setattr(nonlinear_static,
                        "_commit_nonlinear_state_candidate", wrapped_commit)

    model, meta = _cantilever(route)
    load = _edge_load(model, meta["beam_nodes"], -PRESS_FORCE)
    token = CancellationToken()
    events = []
    accepted_commits = 0

    def on_progress(event):
        nonlocal accepted_commits
        events.append(event.event_type)
        if event.event_type == "nonlinear_static_step":
            # Fires after the increment is committed: this is a known
            # accepted step.  Cancel now so the next trial is refused.
            accepted_commits = len(commits)
            token.cancel(
                "deliberate cancellation after the first accepted step")

    with pytest.raises(SolveCancelled) as exc_info:
        solve_static_nonlinear(
            model, load, contact=_contact(CYCLE_PLANE_Z), num_steps=4,
            max_iterations=20, kinematics="von_karman",
            progress_callback=on_progress, cancellation_token=token,
        )
    assert "deliberate cancellation" in exc_info.value.reason
    assert active_contact_manager() is None
    # A genuinely accepted step was observed before the cancellation.
    assert "nonlinear_static_step" in events
    assert accepted_commits >= 1

    baseline = commits[accepted_commits - 1]
    owner = baseline["owner"]
    # Record the observed backend: the production commit returns either the
    # persistent NonlinearStateStore or the legacy mapping fallback; both
    # are Mapping owners whose payload is the actual accepted state.
    backend = baseline["backend"]
    assert backend in ("NonlinearStateStore", "dict"), backend
    # The captured accepted state is nonempty and is preserved exactly by
    # the cancellation: no further commit mutated it.
    assert baseline["payload"]
    assert len(commits) == accepted_commits
    current = {int(key): copy.deepcopy(value) for key, value in owner.items()}
    assert _states_equal(current, baseline["payload"])
    if isinstance(owner, NonlinearStateStore):
        assert owner.has_active_trial is False
        assert owner.generation == baseline["generation"]

    # The SAME model solves to completion and reproduces the clean
    # reference (the fresh reference is a comparator; the preserved accepted
    # state and same-model reuse are the evidence above).
    resolved = solve_static_nonlinear(
        model, load, contact=_contact(CYCLE_PLANE_Z), num_steps=4,
        max_iterations=20, kinematics="von_karman",
        tolerance=SOLVE_TOLERANCE,
    )
    assert resolved.status == "completed"
    np.testing.assert_allclose(resolved.displacements,
                               reference.displacements,
                               rtol=1.0e-10, atol=1.0e-12)


def test_callback_failure_discards_contact_scope_allows_fresh_solve():
    model, meta = _cantilever("eccentric")
    load = _edge_load(model, meta["beam_nodes"], -PRESS_FORCE)

    def fail_callback(_message):
        raise RuntimeError("deliberate callback failure")

    with pytest.raises(RuntimeError, match="deliberate callback failure"):
        solve_static_nonlinear(
            model, load, contact=_contact(CYCLE_PLANE_Z), num_steps=4,
            max_iterations=20, kinematics="von_karman",
            status_callback=fail_callback,
        )
    assert active_contact_manager() is None

    fresh_model, fresh_meta = _cantilever("eccentric")
    fresh = solve_static_nonlinear(
        fresh_model,
        _edge_load(fresh_model, fresh_meta["beam_nodes"], -PRESS_FORCE),
        contact=_contact(CYCLE_PLANE_Z), num_steps=4, max_iterations=20,
        kinematics="von_karman",
    )
    assert fresh.status == "completed"


# ---------------------------------------------------------------------------
# Split restart, and checkpoint rejection after legitimate model updates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ROUTES)
def test_split_restart_matches_uninterrupted(route):
    model, meta = _cantilever(route)
    load = _edge_load(model, meta["beam_nodes"], -PRESS_FORCE)
    contact = _contact(CYCLE_PLANE_Z)
    common = dict(num_steps=1, max_iterations=20, kinematics="von_karman",
                  tolerance=SOLVE_TOLERANCE)

    half = solve_static_nonlinear(
        model, load, contact=contact, max_load_factor=0.5,
        emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed"
    assert half.restart_checkpoint["version"] == 2
    assert half.restart_checkpoint["contact_state"] == {}
    # With the plane at the reference midsurface the half-load state is
    # already closed: inspect the actual committed branch.
    assert half.info["general_contact"]["active_pressure_dofs"] > 0

    resumed = solve_static_nonlinear(
        model, load, contact=contact, max_load_factor=1.0,
        restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    uninterrupted_model, uninterrupted_meta = _cantilever(route)
    uninterrupted = solve_static_nonlinear(
        uninterrupted_model,
        _edge_load(uninterrupted_model,
                   uninterrupted_meta["beam_nodes"], -PRESS_FORCE),
        contact=_contact(CYCLE_PLANE_Z), num_steps=2, max_iterations=20,
        kinematics="von_karman", record_increment_snapshots=True,
        tolerance=SOLVE_TOLERANCE,
    )
    assert resumed.status == uninterrupted.status == "completed"
    # State continuity: the half-load accepted state is the first committed
    # increment of the uninterrupted path.
    assert len(uninterrupted.snapshots) >= 2
    np.testing.assert_allclose(half.displacements,
                               uninterrupted.snapshots[0].displacements,
                               rtol=1.0e-8, atol=1.0e-10)
    np.testing.assert_allclose(resumed.displacements,
                               uninterrupted.displacements,
                               rtol=1.0e-8, atol=1.0e-10)
    assert resumed.info["general_contact"]["shell_force"] == pytest.approx(
        uninterrupted.info["general_contact"]["shell_force"], abs=1.0e-8)


@pytest.mark.parametrize("route", MPC_ROUTES)
def test_restart_rejects_changed_weights_or_offset_before_newton(route):
    # The shared route has no MPC rows to update: its checkpoint
    # compatibility is exercised by the split-restart test across all
    # routes.  The eccentric route has a single-master partition of unity
    # with no legitimate aliased weight alternative, so its refusal
    # coverage is the offset update below.
    model, meta = _cantilever(route)
    load = _edge_load(model, meta["beam_nodes"], -PRESS_FORCE)
    contact = _contact(CYCLE_PLANE_Z)
    common = dict(num_steps=1, max_iterations=20, kinematics="von_karman")

    half = solve_static_nonlinear(
        model, load, contact=contact, max_load_factor=0.5,
        emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed"

    weights = meta["couplings"][0]["weights"]
    original_weights = weights.copy()
    r = meta["couplings"][0]["r"]
    original_r = r.copy()

    def fail_if_reached(_message):
        raise AssertionError("checkpoint rejection must precede Newton work")

    if weights.size > 1:
        # A legitimate aliased weight update (still a valid partition of
        # unity) invalidates the checkpoint before any new Newton work.
        weights[:] = [0.7, 0.3]
        with pytest.raises(ValueError, match="incompatible"):
            solve_static_nonlinear(
                model, load, contact=contact, max_load_factor=1.0,
                restart_checkpoint=half.to_restart_checkpoint(),
                status_callback=fail_if_reached, **common,
            )
        weights[:] = original_weights

    # A legitimate aliased offset update is rejected the same way.
    r[:] = [0.01, 0.0, 0.05]
    with pytest.raises(ValueError, match="incompatible"):
        solve_static_nonlinear(
            model, load, contact=contact, max_load_factor=1.0,
            restart_checkpoint=half.to_restart_checkpoint(),
            status_callback=fail_if_reached, **common,
        )
    r[:] = original_r

    # With the model restored, the checkpoint remains usable.
    resumed = solve_static_nonlinear(
        model, load, contact=contact, max_load_factor=1.0,
        restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed"
