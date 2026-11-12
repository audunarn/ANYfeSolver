"""Bounded unchanged-model load-path discriminator for the failed 50 N case.

This development run tests whether ten explicit 5 N steps reach the 50 N
state that did not converge from the virgin state. It changes neither the
material curve nor the finite mixed owner's Newton/acceptance algorithm.
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
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import anyfesolver
from anyfesolver import corotational
from anymaterial import DNVC208MaterialCurve
from anyfesolver._stiffened_shell_ge_experimental import (
    ExperimentalGEAnalysis,
)
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from test_stiffened_shell_ge_finite_experimental import _actions, _panel


def run(step_n: int = 5, max_force_n: int = 50, *,
        elastic_shell: bool = False,
        stable_pullback: bool = False) -> tuple[dict, bytes | None]:
    if (type(step_n) is not int or step_n not in (5, 10) or
            type(max_force_n) is not int or max_force_n not in (10, 50) or
            max_force_n % step_n or type(elastic_shell) is not bool or
            type(stable_pullback) is not bool):
        raise ValueError("bounded 5/10 N step and 10/50 N target required")
    if stable_pullback:
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
                    rigid.T@shifted[node]+
                    rotation_difference@arm[node])
                nodal = corotational.rotation_matrix_from_vector(
                    rotations[node])
                local[node, 3:] = (
                    corotational.rotation_vector_from_matrix(
                        rigid.T@nodal))
            return local.reshape(-1), rigid, deformed

        corotational._corotational_deformation_state = stable
    expected = (ROOT / "src" / "anyfesolver" / "__init__.py").resolve()
    if Path(anyfesolver.__file__).resolve() != expected:
        raise RuntimeError("step-path probe imported another checkout")
    base = _panel()
    pressure = 20_000.
    curve = DNVC208MaterialCurve(
        pressure, 1.05*pressure, 1.1*pressure, .005, .01,
        4.*pressure, .2)
    definition = replace(
        base, shell=replace(
            base.shell, q4_elements=((1, 2, 5, 4),),
            s3_elements=((2, 3, 6), (2, 6, 5)),
            j2_curve=None if elastic_shell else curve))
    owner = ExperimentalGEAnalysis(definition)
    driver = FinitePointDriver(owner)
    owner._finite_driver = driver
    original = driver._assemble
    counter = 0
    best_merit = float("inf")
    best_free = float("inf")
    best_gap = float("inf")
    active_max = 0.
    best_rows = []
    pattern = driver._load(_actions())
    physical_free = owner._free
    physical_scale = owner._coordinate_scale[physical_free]

    def observed(*args, **kwargs):
        nonlocal counter, best_merit, best_free, best_gap, active_max
        nonlocal best_rows
        result = original(*args, **kwargs)
        counter += 1
        free, gap = float(result[2]), float(result[3])
        if max(free, gap) < best_merit:
            best_merit, best_free, best_gap = max(free, gap), free, gap
            factor = float(args[1])
            residual = result[0][physical_free]
            applied = factor*pattern[physical_free]
            weighted = residual*physical_scale
            order = np.argsort(np.abs(weighted))[-8:][::-1]
            best_rows = [{
                "global_dof": int(physical_free[i]),
                "domain": (
                    "shell" if physical_free[i] < owner._shell_count
                    else "beam_or_retained"),
                "residual": float(residual[i]),
                "applied": float(applied[i]),
                "coordinate_scale": float(physical_scale[i]),
                "weighted_residual": float(weighted[i]),
                "active_ratio": (
                    float(abs(residual[i]/applied[i]))
                    if applied[i] != 0. else None),
            } for i in order]
        active_max = max(active_max, *(
            float(np.max(state["alpha"])) for state in result[4]
            if isinstance(state, dict) and "alpha" in state))
        return result

    driver._assemble = observed
    records = []
    started = time.perf_counter()
    targets = tuple(
        k*step_n/1000. for k in range(1, max_force_n//step_n+1))
    for factor in targets:
        if time.perf_counter()-started > 500.:
            records.append({
                "factor": factor, "status": "resource_time_guard",
                "accepted_generation": owner._generation,
                "accepted_factors": list(driver._accepted_factors),
            })
            break
        before = counter
        best_merit = best_free = best_gap = float("inf")
        active_max = 0.
        best_rows = []
        tick = time.perf_counter()
        try:
            result = owner.solve(
                _actions(), (factor,),
                max_iterations=20, max_backtracks=12)
        except Exception as error:
            records.append({
                "factor": factor, "target_force_n": factor*1000.,
                "status": "failed",
                "failure_type": type(error).__name__,
                "failure_detail": str(error),
                "assemblies": counter-before,
                "best_free_ratio": best_free,
                "best_gap": best_gap,
                "active_alpha_max": active_max,
                "best_dominant_rows": best_rows,
                "elapsed_s": time.perf_counter()-tick,
                "accepted_generation": owner._generation,
                "accepted_factors": list(driver._accepted_factors),
            })
            break
        records.append({
            "factor": factor, "target_force_n": factor*1000.,
            "status": result.status,
            "assemblies": counter-before,
            "newton_iterations": result.iterations,
            "free_ratio": result.free_equilibrium_ratio,
            "gap": result.constraint_max,
            "active_alpha_max": active_max,
            "best_dominant_rows": best_rows,
            "tip_uz_m": float(result.beam_positions[-1, 2]-
                              driver._reference_positions[-1, 2]),
            "elapsed_s": time.perf_counter()-tick,
            "accepted_generation": owner._generation,
            "accepted_factors": list(result.accepted_factors),
        })
    checkpoint = owner.checkpoint() if owner._generation else None
    return {
        "classification": "development load-path discriminator; no F05/G3 acceptance",
        "source_git": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "imported_anyfesolver": str(expected),
        "threadpools": threadpool_info(),
        "targets": list(targets),
        "step_n": step_n,
        "max_force_n": max_force_n,
        "elastic_shell": elastic_shell,
        "stable_pullback_counterfactual": stable_pullback,
        "max_iterations": 20,
        "max_backtracks": 12,
        "accepted_generation": owner._generation,
        "accepted_factors": list(driver._accepted_factors),
        "records": records,
        "elapsed_s": time.perf_counter()-started,
        "accepted_checkpoint_sha256": (
            sha256(checkpoint).hexdigest() if checkpoint is not None else None),
        "accepted_checkpoint_bytes": (
            len(checkpoint) if checkpoint is not None else 0),
    }, checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step-n", type=int, choices=(5, 10), default=5)
    parser.add_argument("--max-force-n", type=int, choices=(10, 50), default=50)
    parser.add_argument("--elastic-shell", action="store_true")
    parser.add_argument("--stable-pullback", action="store_true")
    args = parser.parse_args()
    output = args.output
    if output.exists() or output.with_suffix(".checkpoint").exists():
        raise FileExistsError("preserve earlier development evidence")
    result, checkpoint = run(
        args.step_n, args.max_force_n, elastic_shell=args.elastic_shell,
        stable_pullback=args.stable_pullback)
    if checkpoint is not None:
        output.with_suffix(".checkpoint").write_bytes(checkpoint)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["records"][-1]["status"],
        "last_factor": result["records"][-1]["factor"],
        "accepted_factors": result["accepted_factors"],
        "elapsed_s": result["elapsed_s"],
    }))
