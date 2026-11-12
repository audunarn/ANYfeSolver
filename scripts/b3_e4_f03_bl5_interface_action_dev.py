"""Compare native BL5 and external nodal shell actions on one native shell.

This bounded 16-interval transfer diagnostic uses the same native shell
compliance for both actions. It does not select a multiplier space or qualify
the continuous bond. Root-node interface action is excluded from comparison:
the external fixed-root reaction cannot separate its constraint contribution.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import anyfesolver
from anyfesolver import stiffened_shell
import numpy as np
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info, threadpool_limits

from test_stiffened_shell_l1_assembly_experimental import (
    _bl5_joint_mesh_definition,
)


def _edge_inventory(owner, force):
    """Supported shell-edge action excluding the fixed root."""
    nodes = range(2, 18)
    rows = np.array([force[6*owner._shell_index[node]:
                           6*owner._shell_index[node]+6] for node in nodes])
    positions = np.array([owner._shell_xyz[owner._shell_index[node]]
                          for node in nodes])
    x = positions[:, 0]
    return {"force_n": rows[:, :3].sum(axis=0).tolist(),
            "moment_nm": (rows[:, 3:]+np.cross(
                positions, rows[:, :3])).sum(axis=0).tolist(),
            "x_force_nm": (x[:, None]*rows[:, :3]).sum(axis=0).tolist(),
            "x_moment_nm2": (x[:, None]*rows[:, 3:]).sum(axis=0).tolist()}


def _run_case(owner, external, common, case):
    if external["intervals"] != 16 or external["case"] != case:
        raise ValueError("declared 16-interval external action required")
    if common["intervals"] != 16 or common["case"] != case:
        raise ValueError("matching common-action shell control required")
    action = stiffened_shell.NodalAction(
        "beam", 117, (0., 100., 0., 0., 0., 0.) if case == "fy"
        else (0., 0., 0., 10., 0., 0.))
    result = owner.solve_linear((action,))
    if result.status != "completed":
        raise ValueError("completed native mixed response required")
    count = owner._shell_count
    k = owner._K[:count, :count].tocsc()
    native_force = np.asarray(
        -owner._C.T[:count, :]@result.multipliers).reshape(-1)
    native_u = result.shell_displacement
    free = np.setdiff1d(np.arange(count), owner._fixed[owner._fixed < count])
    native_residual = float(np.max(np.abs(
        (k@native_u-native_force)[free])))
    if native_residual > 1e-7*max(np.max(np.abs(native_force)), 1.):
        raise ValueError("native shell action is not work/equilibrium conjugate")
    external_force = np.zeros(count)
    keys = {str(node) for node in range(2, 18)}
    if set(external["shell_interface_action"]) != keys:
        raise ValueError("complete nonroot external shell-edge action required")
    for key, wrench in external["shell_interface_action"].items():
        index = owner._shell_index[int(key)]
        external_force[6*index:6*index+6] = wrench
    external_u = np.zeros(count)
    external_u[free] = splu(k[free, :][:, free]).solve(external_force[free])
    native_work = float(native_force@native_u)
    external_on_native_work = float(external_force@external_u)
    cross_work = float(external_force@native_u)
    difference_work = native_work+external_on_native_work-2*cross_work
    floor = 1e-10*max(native_work, external_on_native_work)
    if (not np.isfinite((native_work, external_on_native_work,
                         cross_work, difference_work)).all() or
            native_work <= 0. or external_on_native_work <= 0. or
            difference_work < -floor or
            abs(external_on_native_work-common["native_shell_work_j"]) >
            1e-8*max(external_on_native_work, 1e-12)):
        raise ValueError("common-shell compliance work identity failed")
    delta = native_force-external_force
    lower = np.array([owner._shell_index[node] for node in range(2, 18)])
    rows = delta.reshape(-1, 6)[lower]
    return {"case": case,
            "native_shell_work_j": native_work,
            "external_action_on_native_shell_work_j": external_on_native_work,
            "cross_work_j": cross_work,
            "action_difference_compliance_work_j": max(difference_work, 0.),
            "relative_action_difference_compliance_norm": float(np.sqrt(
                max(difference_work, 0.)/external_on_native_work)),
            "native_action_nonroot": _edge_inventory(owner, native_force),
            "external_action_nonroot": _edge_inventory(owner, external_force),
            "nodewise_delta_force_l2_n": float(np.linalg.norm(rows[:, :3])),
            "nodewise_delta_moment_l2_nm": float(np.linalg.norm(rows[:, 3:])),
            "native_shell_free_residual_max": native_residual,
            "native_weak_gap": result.constraint_max,
            "native_free_equilibrium_ratio": result.free_equilibrium_ratio}


def main(external_path: Path, common_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("preserve prior interface-action evidence")
    external_bytes = external_path.read_bytes()
    common_bytes = common_path.read_bytes()
    external = json.loads(external_bytes)
    common = json.loads(common_bytes)
    if external.get("failure") is not None or common.get("failure") is not None:
        raise ValueError("completed external/common-action evidence required")
    reference = {(entry["intervals"], entry["case"]): entry
                 for entry in external["cases"]}
    controls = {(entry["intervals"], entry["case"]): entry
                for entry in common["results"]}
    report = {"status": "BL5_INTERFACE_ACTION_DEVELOPMENT_ONLY",
              "source_git": subprocess.check_output(
                  ("git", "rev-parse", "HEAD"), text=True).strip(),
              "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "external_sha256": sha256(external_bytes).hexdigest(),
              "common_sha256": sha256(common_bytes).hexdigest(),
              "imported_anyfesolver": str(Path(anyfesolver.__file__).resolve()),
              "cases": [], "failure": None,
              "limitations": ("Nonroot supported edge actions on one native "
                              "shell; strong external versus weak native "
                              "enforcement. Root action and continuum bond "
                              "fidelity are not determined.")}
    try:
        with threadpool_limits(limits=1):
            report["threadpools"] = threadpool_info()
            for span in (1, 2):
                owner = stiffened_shell.create_analysis(
                    definition=_bl5_joint_mesh_definition(16, 16, span))
                try:
                    for case in ("fy", "mx"):
                        entry = _run_case(owner, reference[16, case],
                                          controls[16, case], case)
                        entry["span"] = span
                        report["cases"].append(entry)
                finally:
                    owner.close()
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                     allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"cases": len(report["cases"]),
                      "failure": report["failure"],
                      "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("external", type=Path)
    parser.add_argument("common", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.external, args.common, args.output)
