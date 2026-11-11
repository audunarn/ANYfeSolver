"""Focused mechanism diagnosis of the original-fixture seed QP refusal.

The first actual decision on the original 300-coordinate fixture (captured
in ``test_general_contact_original_workflow_seed_decision.py``) refuses the
seed with ``qp_iteration_cap``: the internal bounded nonnegative pressure QP
consumed its full 4*modes+16 = 116 iteration cap although the constrained
factorization backward error (2.7e-13) and the symmetrized gap response
eigenvalues (4.4e-8 .. 1.93, positive definite) passed every operator gate.
This module captures the exact production QP arguments (h, g) through a
private wrapper around ``initial_contact_seed._bounded_nonnegative_qp``,
then replays the identical active-set iteration with instrumentation to
record the true mechanism: repeated working sets, iterate fixed points,
the face-solve residual and the componentwise free gradient against the
1e-12*scale optimality tolerance. The replay is test-side observation of
the captured production data; no production gate or acceptance predicate
is evaluated or changed. The same capture additionally runs the repaired
v2 release-alternatives QP on the exact saved operator and verifies the
actual KKT obligations of the returned iterate within the unchanged
tolerance: exact nonnegativity/finiteness, stationarity on the
positive-pressure support (gradient ~ 0 where z > 0) and a valid bound
gradient (gradient >= -tolerance where z == 0). A positive gradient at
a zero-pressure bound is valid, so the all-gradient maximum is kept only
as a separate descriptive field, never as the KKT norm; the KKT residual
is the maximum of the separate stationarity and bound-gradient
violations. Any non-optimal terminal is recorded honestly as a refusal
terminal with its computed residual.
"""
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import solve_static_nonlinear
from anyfesolver.control import SolveCancelled
from anyfesolver.general_contact import initial_contact_seed, semismooth_path

FIXTURE_DIR = Path("C:/ANYworkspaceWork/contact_joint_dual_full8_20261010")
DESCRIPTOR_PATH = FIXTURE_DIR / "force-full.json"
CAPTURE_NAME = "original-workflow-qp-mechanism-v1.json"


def _original_case():
    spec = importlib.util.spec_from_file_location(
        "candidate_fixture", FIXTURE_DIR / "candidate_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    descriptor = json.loads(DESCRIPTOR_PATH.read_text(encoding="utf-8"))
    model, definition, settings, program = module.build_candidate(descriptor)
    from dataclasses import replace
    return model, definition, replace(
        settings, initial_contact_seed="linearized_kkt_dev"), program


def _instrumented_replay(h, g, iteration_cap, tolerance):
    """Replay the production active-set loop with mechanism logging."""
    modes = len(g)
    z = np.zeros(modes)
    working = np.ones(modes, dtype=bool)
    trace = []
    seen_working = {}
    fixed_point_iteration = None
    terminal = None
    for iteration in range(1, iteration_cap + 1):
        gradient = h @ z + g
        bound_violation = working & (gradient < -tolerance)
        free = ~working
        row = {"iteration": iteration,
               "free_count": int(np.count_nonzero(free)),
               "bound_violations": int(np.count_nonzero(bound_violation)),
               "free_gradient_max_abs": float(
                   np.max(np.abs(gradient[free]), initial=0.0)),
               "clamped_gradient_min": float(np.min(gradient[working])),
               "z_norm": float(np.linalg.norm(z)),
               "z_min": float(np.min(z))}
        signature = tuple(int(i) for i in np.flatnonzero(~working))
        row["working_signature"] = signature
        if signature in seen_working:
            row["repeated_working_set"] = seen_working[signature]
        seen_working.setdefault(signature, iteration)
        if (not np.any(bound_violation)
                and not np.any(free & (np.abs(gradient) > tolerance))):
            terminal = "optimal"
            trace.append(row)
            break
        if np.any(bound_violation):
            entering = int(np.argmin(np.where(bound_violation, gradient,
                                              np.inf)))
            working[entering] = False
            free = ~working
        face = h[np.ix_(free, free)]
        target = np.linalg.solve(face, -g[free])
        face_residual = float(np.max(np.abs(face @ target + g[free]),
                                    initial=0.0))
        row["face_solve_residual_max_abs"] = face_residual
        if np.all(target >= 0.0):
            new_z = np.zeros(modes)
            new_z[free] = target
            row["full_face_step"] = True
            row["target_equals_z"] = bool(np.array_equal(new_z, z))
            if np.array_equal(new_z, z):
                fixed_point_iteration = iteration
                trace.append(row)
                terminal = "fixed_point_full_face"
                z = new_z
                break
            z = new_z
            trace.append(row)
            continue
        direction = np.zeros(modes)
        direction[free] = target - z[free]
        blocking = free & (direction < 0.0) & (z > 0.0)
        row["full_face_step"] = False
        row["blocking_count"] = int(np.count_nonzero(blocking))
        if not np.any(blocking):
            entering = int(np.argmin(np.where(
                free & (direction < 0.0), direction, np.inf)))
            working[entering] = True
            trace.append(row)
            continue
        fractions = z[blocking]/(-direction[blocking])
        index = int(np.flatnonzero(blocking)[int(np.argmin(fractions))])
        alpha = min(1.0, float(np.min(fractions)))
        if alpha <= 0.0 or not np.isfinite(alpha):
            terminal = "degenerate_step"
            trace.append(row)
            break
        z = z + alpha*direction
        z[index] = 0.0
        working[index] = True
        trace.append(row)
    if terminal is None:
        terminal = "iteration_cap"
    return z, terminal, trace, fixed_point_iteration


def _instrumented_v2_replay(h, g, iteration_cap, tolerance):
    """Mirror the production v2 (release_alternatives) active-set loop with
    mechanism logging; must reproduce the production v2 result exactly."""
    modes = len(g)
    z = np.zeros(modes)
    working = np.ones(modes, dtype=bool)
    blocked = np.zeros(modes, dtype=bool)
    trace = []
    terminal = None
    for iteration in range(1, iteration_cap + 1):
        gradient = h @ z + g
        free = ~working
        free_residual = (float(np.max(np.abs(gradient[free]), initial=0.0))
                         if np.any(free) else 0.0)
        bound_violation = working & (gradient < -tolerance)
        row = {"iteration": iteration,
               "free": [int(i) for i in np.flatnonzero(free)],
               "blocked": [int(i) for i in np.flatnonzero(blocked)],
               "free_residual": free_residual,
               "bound_violations": int(np.count_nonzero(bound_violation)),
               "bound_gradient_min": float(np.min(gradient[working]))
               if np.any(working) else 0.0,
               "gradient_max_abs": float(np.max(np.abs(gradient),
                                                initial=0.0)),
               "z_norm": float(np.linalg.norm(z))}
        if not np.any(bound_violation) and free_residual <= tolerance:
            terminal = "optimal"
            row["branch"] = "certified_optimal"
            trace.append(row)
            break
        released = False
        entering = -1
        if np.any(bound_violation):
            if np.any(bound_violation & ~blocked) and free_residual <= tolerance:
                entering = int(np.argmin(np.where(bound_violation & ~blocked,
                                                  gradient, np.inf)))
            if entering >= 0:
                working[entering] = False
                released = True
                free = ~working
        row["released"] = released
        row["entering"] = entering
        face = h[np.ix_(free, free)]
        target = np.linalg.solve(face, -g[free])
        if np.all(target >= 0.0):
            candidate = np.zeros(modes)
            candidate[free] = target
            if np.array_equal(candidate, z):
                if released:
                    row["branch"] = "degenerate_release"
                    working[entering] = True
                    blocked[entering] = True
                    trace.append(row)
                    continue
                terminal = "kkt_fixed_point"
                row["branch"] = "kkt_fixed_point"
                trace.append(row)
                break
            row["branch"] = "full_face_step"
            z = candidate
            blocked[:] = False
            trace.append(row)
            continue
        direction = np.zeros(modes)
        direction[free] = target - z[free]
        blocking = free & (direction < 0.0) & (z > 0.0)
        if not np.any(blocking):
            row["branch"] = "entering_blocks_at_zero"
            entering = int(np.argmin(np.where(
                free & (direction < 0.0), direction, np.inf)))
            working[entering] = True
            blocked[entering] = True
            trace.append(row)
            continue
        row["branch"] = "partial_step"
        fractions = z[blocking]/(-direction[blocking])
        index = int(np.flatnonzero(blocking)[int(np.argmin(fractions))])
        alpha = min(1.0, float(np.min(fractions)))
        if alpha <= 0.0 or not np.isfinite(alpha):
            terminal = "degenerate_step"
            trace.append(row)
            break
        z = z + alpha*direction
        z[index] = 0.0
        working[index] = True
        blocked[:] = False
        trace.append(row)
    if terminal is None:
        terminal = "iteration_cap"
    return z, terminal, trace


@pytest.mark.skipif(not (FIXTURE_DIR / "candidate_fixture.py").is_file()
                    or not DESCRIPTOR_PATH.is_file(),
                    reason='historical reproducer inputs absent: original-workflow '
                           'QP mechanism capture requires candidate_fixture.py and '
                           'force-full.json under C:/ANYworkspaceWork/'
                           'contact_joint_dual_full8_20261010')
def test_qp_iteration_cap_mechanism_is_recorded():
    model, definition, seeded, program = _original_case()
    captured = {}

    original_qp = initial_contact_seed._bounded_nonnegative_qp

    def wrapper(h, g, *, iteration_cap, release_alternatives=False):
        result = original_qp(h, g, iteration_cap=iteration_cap,
                             release_alternatives=release_alternatives)
        captured["h"] = np.array(h, dtype=float)
        captured["g"] = np.array(g, dtype=float)
        captured["iteration_cap"] = iteration_cap
        captured["release_alternatives"] = bool(release_alternatives)
        captured["result"] = result
        return result

    initial_contact_seed._bounded_nonnegative_qp = wrapper

    def hook(payload):
        captured["decision"] = payload["record"]
        raise SolveCancelled("qp mechanism capture",
                             stage="contact_path.step:1.semismooth."
                                   "initial_seed.decision")

    assert semismooth_path._INITIAL_SEED_DECISION_HOOK is None
    semismooth_path._INITIAL_SEED_DECISION_HOOK = hook
    try:
        with pytest.raises(SolveCancelled, match="qp mechanism capture"):
            solve_static_nonlinear(
                model, contact=definition, contact_settings=seeded,
                contact_path_program=program, max_iterations=18,
                max_load_factor=1.0/8.0)
    finally:
        semismooth_path._INITIAL_SEED_DECISION_HOOK = None
        initial_contact_seed._bounded_nonnegative_qp = original_qp
    decision = captured["decision"]
    assert decision["outcome"] == "refused"
    assert decision["reason"] == "qp_iteration_cap"
    h, g = captured["h"], captured["g"]
    modes = len(g)
    scale = max(1.0, float(np.max(np.abs(g), initial=0.0)),
                float(np.max(np.abs(h), initial=0.0)))
    tolerance = initial_contact_seed.SEED_QP_OPTIMALITY_TOLERANCE*scale
    z, terminal, trace, fixed_point_iteration = _instrumented_replay(
        h, g, captured["iteration_cap"], tolerance)
    production_z, production_iterations, production_terminal = (
        captured["result"])
    assert production_terminal == "iteration_cap"
    assert np.array_equal(z, production_z), "replay must match production"
    # The repaired v2 QP on the exact same captured original operator: an
    # `optimal` terminal must certify the full unmasked KKT residual within
    # the unchanged tolerance; any other terminal is honest and the
    # production caller refuses on the reported residual. No false
    # optimal from a release mask or a repeated-face fixed point.
    v2_z, v2_iterations, v2_terminal = original_qp(
        h, g, iteration_cap=captured["iteration_cap"],
        release_alternatives=True)
    v2_gradient = h @ v2_z + g
    # Correct KKT oracle for z >= 0: the obligations are exact
    # nonnegativity/finiteness, stationarity on the positive-pressure
    # support (gradient ~ 0 where z > 0) and a valid bound gradient
    # (gradient >= -tolerance where z == 0). A positive gradient at a
    # zero-pressure bound is valid, so the all-gradient maximum stays a
    # separate descriptive field and is never the KKT norm. The solver
    # clamps bound entries exactly to zero, so the support split is exact.
    v2_positive = v2_z > 0.0
    v2_free_stationarity = (float(np.max(np.abs(v2_gradient[v2_positive]),
                                         initial=0.0))
                            if np.any(v2_positive) else 0.0)
    v2_bound_gradient_min = (float(np.min(v2_gradient[~v2_positive]))
                             if np.any(~v2_positive) else 0.0)
    v2_bound_violation = max(0.0, -v2_bound_gradient_min)
    v2_complementarity = float(np.max(np.abs(v2_z*v2_gradient),
                                      initial=0.0))
    v2_kkt_residual = max(v2_free_stationarity, v2_bound_violation)
    v2_all_gradient_max_abs = float(np.max(np.abs(v2_gradient),
                                           initial=0.0))
    v2_all_gradient_min = float(np.min(v2_gradient))
    v2_mirror_z, v2_mirror_terminal, v2_trace = _instrumented_v2_replay(
        h, g, captured["iteration_cap"], tolerance)
    assert v2_mirror_terminal == v2_terminal, (
        "v2 mirror must reproduce the production terminal")
    assert np.array_equal(v2_mirror_z, v2_z), (
        "v2 mirror must reproduce the production iterate")
    assert np.all(np.isfinite(v2_z)) and np.all(v2_z >= 0.0)
    repeated = [row for row in trace if "repeated_working_set" in row]
    full_steps = [row for row in trace if row.get("full_face_step")]
    last = trace[-1]
    production_gradient = h @ production_z + g
    production_positive = production_z > 0.0
    production_free_stationarity = (
        float(np.max(np.abs(production_gradient[production_positive]),
                     initial=0.0))
        if np.any(production_positive) else 0.0)
    production_bound_gradient_min = (
        float(np.min(production_gradient[~production_positive]))
        if np.any(~production_positive) else 0.0)
    production_kkt_residual = max(production_free_stationarity,
                                  max(0.0, -production_bound_gradient_min))
    record = {
        "schema": "ORIGINAL_WORKFLOW_QP_MECHANISM_V1",
        "modes": int(modes),
        "iteration_cap": int(captured["iteration_cap"]),
        "production_terminal": production_terminal,
        "production_iterations": int(production_iterations),
        "replay_terminal": terminal,
        "replay_iterations": len(trace),
        "repeated_working_set_first_iteration": (
            repeated[0]["iteration"] if repeated else None),
        "repeated_working_set_count": len(repeated),
        "fixed_point_iteration": fixed_point_iteration,
        "full_face_step_count": len(full_steps),
        "last_free_gradient_max_abs": last["free_gradient_max_abs"],
        "last_clamped_gradient_min": last["clamped_gradient_min"],
        "last_bound_violations": last["bound_violations"],
        "optimality_tolerance": float(tolerance),
        "final_z_norm": float(np.linalg.norm(production_z)),
        "final_z_min": float(np.min(production_z)),
        "final_z_max": float(np.max(production_z)),
        "final_gradient_max_abs": float(np.max(np.abs(
            production_gradient), initial=0.0)),
        "final_gradient_min": float(np.min(production_gradient)),
        "final_positive_support_count": int(
            np.count_nonzero(production_positive)),
        "final_free_stationarity_max_abs": production_free_stationarity,
        "final_bound_gradient_min": production_bound_gradient_min,
        "final_kkt_residual": production_kkt_residual,
        "v2_repaired_terminal": v2_terminal,
        "v2_repaired_iterations": int(v2_iterations),
        "v2_repaired_positive_support_count": int(
            np.count_nonzero(v2_positive)),
        "v2_repaired_free_stationarity_max_abs": v2_free_stationarity,
        "v2_repaired_bound_gradient_min": v2_bound_gradient_min,
        "v2_repaired_bound_gradient_violation": v2_bound_violation,
        "v2_repaired_complementarity_max_abs": v2_complementarity,
        "v2_repaired_kkt_residual": v2_kkt_residual,
        "v2_repaired_all_gradient_max_abs": v2_all_gradient_max_abs,
        "v2_repaired_all_gradient_min": v2_all_gradient_min,
        "v2_repaired_z_norm": float(np.linalg.norm(v2_z)),
        "v2_repaired_z": [float(v) for v in v2_z],
        "v2_trace": v2_trace,
        "trace": trace,
    }
    # Diagnostics are written BEFORE the KKT assertions so any failure
    # keeps the full computed evidence for the classification.
    with open(os.path.join((os.environ.get("TEMP") or tempfile.gettempdir()), CAPTURE_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    # The corrected KKT oracle on the exact saved original operator: an
    # `optimal` terminal must satisfy the actual KKT obligations within
    # the unchanged tolerance; any other terminal is honest and the
    # production caller refuses on the reported residual.
    if v2_terminal == "optimal":
        assert v2_free_stationarity <= tolerance, (
            f"positive-pressure stationarity {v2_free_stationarity:.6e} "
            f"exceeds tolerance {tolerance:.6e}")
        assert v2_bound_gradient_min >= -tolerance, (
            f"bound gradient {v2_bound_gradient_min:.6e} is below "
            f"-tolerance {tolerance:.6e}")
        assert v2_kkt_residual <= tolerance
    else:
        assert v2_terminal in {"iteration_cap", "singular_free_face",
                               "degenerate_step", "kkt_fixed_point"}
