"""Bounded BL5 shell/beam co-refinement screen; no space selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import anyfesolver
from anyfesolver import stiffened_shell
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from test_stiffened_shell_l1_assembly_experimental import (
    _bl5_joint_mesh_definition,
)


def _model(halfcells: int, span: int):
    if halfcells not in (2, 4, 8) or span not in (1, 2):
        raise ValueError("registered small co-refinement levels required")
    return _bl5_joint_mesh_definition(halfcells, halfcells, span)


def _support_wrench(owner, result):
    shell_positions = np.asarray(owner._shell_xyz)
    beam_positions = np.array([owner._beam_nodes[node][0]
                               for node in owner._beam_ids])
    positions = np.vstack((shell_positions, beam_positions))
    nodal = result.reactions[:6*len(positions)].reshape(-1, 6)
    force = nodal[:, :3].sum(axis=0)
    moment = (nodal[:, 3:]+np.cross(positions, nodal[:, :3])).sum(axis=0)
    return force, moment


def _sampled_gap(owner, result):
    positions = np.array([owner._beam_nodes[node][0]
                          for node in owner._beam_ids])
    frames = np.array([owner._beam_nodes[node][1]
                       for node in owner._beam_ids])
    state = Mechanical(
        positions, np.zeros_like(positions), frames,
        np.tile(np.eye(3), (len(owner._beam_parts), 2, 1, 1)),
        np.zeros((len(owner._beam_parts), 18)))
    q = np.r_[result.shell_displacement, result.beam_coordinates]
    gaps = []
    for local, slots, _, _, _ in owner._line_samples:
        reference = local.evaluate(np.zeros((2, 6)), state, np.zeros(5))
        gaps.append(reference.gap_jacobian@q[slots])
    gaps = np.asarray(gaps)
    return {"translation_max_m": float(np.max(np.abs(gaps[:, :3]))),
            "director_max": float(np.max(np.abs(gaps[:, 3:])))}


def _case(owner, terminal: int, name: str, load: tuple[float, ...]):
    action = stiffened_shell.NodalAction("beam", terminal, load)
    try:
        result = owner.solve_linear((action,))
    except ValueError as exc:
        if str(exc) != "single-support BL5 leaves a free native beam rigid mode":
            raise
        return {"status": "UNSUPPORTED_FREE_RIGID_MODE",
                "failure_reason": str(exc)}
    except RuntimeError as exc:
        if "Factor is exactly singular" not in str(exc):
            raise
        return {"status": "FAILED_SINGULAR_FACTORIZATION",
                "failure_reason": str(exc)}
    if result.status != "completed":
        raise AssertionError(f"{name}: {result.status}")
    if result.free_equilibrium_ratio > 1e-8 or result.constraint_max > 1e-9:
        raise AssertionError(f"{name}: equilibrium or weak constraint failed")
    q = np.r_[result.shell_displacement, result.beam_coordinates]
    applied = owner._accepted[1]
    work = float(applied@q)
    internal = float(q@(owner._K@q))
    if abs(work-internal) > 1e-8*max(abs(work), 1e-12):
        raise AssertionError(f"{name}: internal/applied work mismatch")
    force, moment = _support_wrench(owner, result)
    expected_force = np.array(load[:3])
    tip = owner._beam_nodes[terminal][0]
    expected_moment = np.array(load[3:])+np.cross(tip, expected_force)
    force_error = float(np.max(np.abs(force+expected_force)))
    moment_error = float(np.max(np.abs(moment+expected_moment)))
    if force_error > 1e-6 or moment_error > 1e-5:
        raise AssertionError(f"{name}: physical support balance failed")
    beam_slot = owner._beam_index[terminal]
    return {"beam_tip": result.beam_coordinates[
                6*beam_slot:6*beam_slot+6].tolist(),
            "status": "COMPLETED",
            "load_work_j_twice_energy": work,
            "sampled_gap": _sampled_gap(owner, result),
            "weak_gap": result.constraint_max,
            "free_equilibrium_ratio": result.free_equilibrium_ratio,
            "force_balance_max_n": force_error,
            "moment_balance_max_nm": moment_error}


def analyze():
    results = {}
    for halfcells in (2, 4, 8):
        for span in (1, 2):
            model = _model(halfcells, span)
            owner = stiffened_shell.create_analysis(definition=model)
            try:
                terminal = 101+halfcells
                key = f"shell{halfcells}_beam{halfcells}_span{span}"
                results[key] = {
                    "identity": owner.preflight()["identity"],
                    "matrix_order": int(owner._A.shape[0]),
                    "matrix_nnz": int(owner._A.nnz),
                    "multiplier_rows": int(owner._C.shape[0]),
                    "fy": _case(owner, terminal, "fy", (0., 100., 0., 0., 0., 0.)),
                    "mx": _case(owner, terminal, "mx", (0., 0., 0., 10., 0., 0.)),
                    "factorizations": owner.preflight()[
                        "reference_linear_factorizations"],
                }
            finally:
                owner.close()
    ratios = {}
    for n in (2, 4, 8):
        first, second = (results[f"shell{n}_beam{n}_span{s}"] for s in (1, 2))
        ratios[str(n)] = {
            case: (second[case]["load_work_j_twice_energy"]/
                   first[case]["load_work_j_twice_energy"]
                   if first[case]["status"] == second[case]["status"] ==
                   "COMPLETED" else None)
            for case in ("fy", "mx")}
    return {"status": "BL5_JOINT_GROWTH_DEVELOPMENT_ONLY",
            "source_git": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), text=True).strip(),
            "imported_anyfesolver": str(Path(anyfesolver.__file__).resolve()),
            "threadpools": threadpool_info(),
            "geometry_m": {"length": 1., "width": .2,
                           "shell_thickness": .035,
                           "beam_axis_height": .0675},
            "results": results, "span2_to_span1_work_ratio": ratios,
            "limitations": ("Reference-linear flat Q4/native B3 point-free BL5 "
                            "development screen only; same-law discretization "
                            "sensitivity, not physical reference, stability, "
                            "locking, F03 assessment or space selection.")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(limits=1):
        report = analyze()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                      allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"],
                      "ratios": report["span2_to_span1_work_ratio"],
                      "output": str(args.output)}))
