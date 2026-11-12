"""Bounded same-predecessor diagnostic for the saved 40 -> 45 N J2 step.

This is development evidence, not a material-domain or G3 acceptance test.
It restores the existing 40 N checkpoint and observes one 45 N attempt.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import anyfesolver
from anyfesolver import corotational
from anymaterial import DNVC208MaterialCurve
from anyfesolver._stiffened_shell_ge_experimental import ExperimentalGEAnalysis
from test_stiffened_shell_ge_finite_experimental import _actions, _panel


def _copy_state(state):
    return replace(state, q=state.q.copy(), frames=state.frames.copy(),
                   cells=state.cells.copy(), resultants=state.resultants.copy(),
                   multipliers=state.multipliers.copy())


def _active(candidate):
    rows = [np.asarray(item["alpha"]).ravel() for item in candidate
            if isinstance(item, dict) and "alpha" in item]
    values = np.concatenate(rows) if rows else np.zeros(1)
    return {"positive_count": int(np.count_nonzero(values > 0.)),
            "maximum": float(np.max(values)), "sum": float(np.sum(values))}


def _blocks(owner, size):
    count = owner._K.shape[0]
    return {"shell": slice(0, owner._shell_count),
            "beam_retained": slice(owner._shell_count, count),
            "multipliers": slice(count, size),
            "free_physical": owner._free,
            "free_all": np.r_[owner._free, np.arange(count, size)]}


def _install_stable_shell_pullback():
    """Match the diagnostic chart used to create the accepted 40 N state."""
    original_pullback = corotational._corotational_deformation_state

    def stable(reference, element, u_element):
        if reference.category != "shell":
            return original_pullback(reference, element, u_element)
        count = int(element.num_nodes)
        u = np.asarray(u_element, dtype=float).reshape(count, 6)
        translation, rotations = u[:, :3], u[:, 3:]
        deformed = reference.coordinates+translation
        rigid = (
            corotational._shell_center_frame(element, deformed)@
            reference.frame.T)
        local = np.zeros((count, 6))
        arm = reference.coordinates-reference.centroid
        shifted = translation-translation.mean(axis=0)
        rotation_difference = rigid.T-np.eye(3)
        for node in range(count):
            local[node, :3] = (
                rigid.T@shifted[node]+rotation_difference@arm[node])
            nodal = corotational.rotation_matrix_from_vector(rotations[node])
            local[node, 3:] = (
                corotational.rotation_vector_from_matrix(rigid.T@nodal))
        return local.reshape(-1), rigid, deformed

    corotational._corotational_deformation_state = stable


def run(checkpoint: Path, intermediate: float | None = None,
        *, stable_pullback: bool = False):
    expected = (ROOT / "src" / "anyfesolver" / "__init__.py").resolve()
    if Path(anyfesolver.__file__).resolve() != expected:
        raise RuntimeError("diagnostic imported another checkout")
    if stable_pullback:
        _install_stable_shell_pullback()
    raw = checkpoint.read_bytes()
    base = _panel()
    pressure = 20_000.
    curve = DNVC208MaterialCurve(
        pressure, 1.05*pressure, 1.1*pressure, .005, .01, 4.*pressure, .2)
    definition = replace(base, shell=replace(
        base.shell, q4_elements=((1, 2, 5, 4),),
        s3_elements=((2, 3, 6), (2, 6, 5)), j2_curve=curve))
    owner = ExperimentalGEAnalysis(definition)
    owner.restore(raw)
    driver = owner._finite_driver
    assert driver is not None and driver._accepted_factors[-1] == .04
    before = owner.checkpoint()
    assert before == raw
    pattern = driver._load(_actions())
    started = time.perf_counter()
    preliminary = None
    if intermediate is not None:
        if not .04 < intermediate < .045:
            raise ValueError("intermediate factor must be between 40 and 45 N")
        try:
            first = owner.solve(_actions(), (intermediate,),
                                max_iterations=20, max_backtracks=12)
            preliminary = {"status": first.status,
                           "accepted_factors": list(first.accepted_factors)}
        except RuntimeError as exc:
            preliminary = {"status": "failed", "type": type(exc).__name__,
                           "detail": str(exc),
                           "accepted_factors": list(driver._accepted_factors)}
            return {"classification": "bounded cutback diagnostic; no F05/G3 acceptance",
                    "source_git": subprocess.check_output(
                        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
                    "source": str(expected), "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256(raw).hexdigest(),
                    "threadpools": threadpool_info(), "preliminary": preliminary,
                    "stable_pullback_counterfactual": stable_pullback,
                    "outcome": {"status": "not_attempted"},
                    "accepted_checkpoint_unchanged": owner.checkpoint() == before,
                    "assemblies_in_attempt": 0,
                    "elapsed_s": time.perf_counter()-started,
                    "diagnosis": None}
        before = owner.checkpoint()
    assembly = driver._assemble
    advance = driver._advance
    last = {}
    candidates = []
    assemblies = 0

    def observed_advance(state, increment, amount):
        last["base"] = _copy_state(state)
        last["direction"] = increment.copy()
        last["alpha"] = float(amount)
        return advance(state, increment, amount)

    def observed_assemble(state, factor, *args, **kwargs):
        nonlocal assemblies
        result = assembly(state, factor, *args, **kwargs)
        assemblies += 1
        if "alpha" in last and factor == .045:
            candidates.append({"alpha": last["alpha"],
                               "merit": float(max(result[2], result[3])),
                               "free_ratio": float(result[2]),
                               "gap": float(result[3]),
                               "shell_active": _active(result[4])})
        return result

    driver._advance = observed_advance
    driver._assemble = observed_assemble
    try:
        result = owner.solve(_actions(), (.045,), max_iterations=20,
                             max_backtracks=12)
        outcome = {"status": result.status,
                   "accepted_factors": list(result.accepted_factors)}
    except RuntimeError as exc:
        outcome = {"status": "failed", "type": type(exc).__name__,
                   "detail": str(exc),
                   "accepted_factors": list(driver._accepted_factors)}
    finally:
        driver._advance = advance
        driver._assemble = assembly
    after = owner.checkpoint()
    unchanged = after == before
    if outcome["status"] == "failed" and not unchanged:
        raise AssertionError("failed 45 N trial changed the accepted checkpoint")

    diagnosis = None
    if "base" in last:
        state = last["base"]
        direction = last["direction"]
        args = (.045, driver._shell_origins, driver._beam_origins)
        residual, tangent, relative, gap, shell, beam = assembly(
            state, *args, pattern=pattern)
        predicted = tangent @ direction
        blocks = _blocks(owner, len(direction))
        derivatives = []
        for step in (1e-4, 1e-5, 1e-6):
            positive = assembly(advance(state, direction, step), *args,
                                pattern=pattern)
            negative = assembly(advance(state, direction, -step), *args,
                                pattern=pattern)
            observed = (positive[0]-negative[0])/(2*step)
            errors = {}
            for name, indices in blocks.items():
                norm = float(np.linalg.norm(predicted[indices]))
                errors[name] = {
                    "relative": float(np.linalg.norm(
                        (observed-predicted)[indices])/max(norm, 1e-12)),
                    "absolute": float(np.linalg.norm(
                        (observed-predicted)[indices])),
                    "predicted_norm": norm}
            derivatives.append({"step": step, "blocks": errors,
                                "positive_merit": float(max(positive[2], positive[3])),
                                "negative_merit": float(max(negative[2], negative[3])),
                                "merit_central_slope": float(
                                    (max(positive[2], positive[3])-
                                     max(negative[2], negative[3]))/(2*step)),
                                "shell_plus": _active(positive[4]),
                                "shell_minus": _active(negative[4])})
        count = owner._K.shape[0]
        multiplier_probes = {}
        zero_value = replace(state, multipliers=np.zeros_like(state.multipliers))
        zero_increment = direction.copy()
        zero_increment[count:] = 0.
        for name, probe_state, probe_direction in (
                ("zero_multiplier_value", zero_value, direction),
                ("zero_multiplier_increment", state, zero_increment)):
            probe = assembly(probe_state, *args, pattern=pattern)
            predicted_probe = probe[1] @ probe_direction
            step_size = 1e-5
            plus = assembly(advance(probe_state, probe_direction, step_size),
                            *args, pattern=pattern)
            minus = assembly(advance(probe_state, probe_direction, -step_size),
                             *args, pattern=pattern)
            observed_probe = (plus[0]-minus[0])/(2*step_size)
            multiplier_probes[name] = {
                "multiplier_value_norm": float(np.linalg.norm(
                    probe_state.multipliers)),
                "multiplier_increment_norm": float(np.linalg.norm(
                    probe_direction[count:])),
                "step": step_size,
                "blocks": {
                    block: {
                        "absolute": float(np.linalg.norm(
                            (observed_probe-predicted_probe)[indices])),
                        "relative": float(np.linalg.norm(
                            (observed_probe-predicted_probe)[indices]) /
                            max(np.linalg.norm(predicted_probe[indices]), 1e-12)),
                        "predicted_norm": float(np.linalg.norm(
                            predicted_probe[indices])),
                    } for block, indices in blocks.items()},
                "shell_plus": _active(plus[4]),
                "shell_minus": _active(minus[4]),
            }
        free = np.r_[owner._free, np.arange(count, tangent.shape[0])]
        linear = spsolve(tangent[free, :][:, free], -residual[free])
        diagnosis = {
            "base_merit": float(max(relative, gap)),
            "base_free_ratio": float(relative), "base_gap": float(gap),
            "shell_active": _active(shell),
            "beam_accumulated_max": float(max(
                station.accumulated[side] for history in beam
                for station in history.stations for side in range(2))),
            "stored_direction_linear_residual_ratio": float(np.linalg.norm(
                tangent[free, :][:, free] @ direction[free]+residual[free]) /
                max(np.linalg.norm(residual[free]), 1e-12)),
            "fresh_direction_difference_ratio": float(np.linalg.norm(
                linear-direction[free])/max(np.linalg.norm(linear), 1e-12)),
            "last_backtracks": candidates[-13:],
            "actual_update_derivatives": derivatives,
            "multiplier_probes": multiplier_probes}

    return {"classification": "bounded development diagnostic; no F05/G3 acceptance",
            "source_git": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "source": str(expected), "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(raw).hexdigest(),
            "threadpools": threadpool_info(), "preliminary": preliminary,
            "stable_pullback_counterfactual": stable_pullback,
            "outcome": outcome,
            "accepted_checkpoint_unchanged": unchanged,
            "assemblies_in_attempt": assemblies,
            "elapsed_s": time.perf_counter()-started,
            "diagnosis": diagnosis}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--intermediate", type=float)
    parser.add_argument("--stable-pullback", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing diagnostic evidence")
    output = run(args.checkpoint, args.intermediate,
                 stable_pullback=args.stable_pullback)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True,
                                      allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": output["outcome"]["status"],
                      "assemblies": output["assemblies_in_attempt"],
                      "checkpoint_unchanged": output["accepted_checkpoint_unchanged"],
                      "elapsed_s": output["elapsed_s"]}))
