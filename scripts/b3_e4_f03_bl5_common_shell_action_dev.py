"""Bounded common-action shell comparison for the BL5 nodal reference.

Apply independently recovered OpenSees shell-edge actions to the native Q4
shell stiffness with identical reference geometry and root support. This
allocates shell response; it does not qualify the line law or select P0 space.
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


def _wrench(nodes: np.ndarray, nodal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = nodal.reshape(-1, 6)
    force = rows[:, :3].sum(axis=0)
    moment = (rows[:, 3:]+np.cross(nodes, rows[:, :3])).sum(axis=0)
    return force, moment


def _case(owner, external: dict) -> dict:
    n = external["intervals"]
    if n not in (4, 8, 16) or external["case"] not in ("fy", "mx"):
        raise ValueError("declared reference case required")
    count = 2*(n+1)
    if tuple(owner._shell_ids) != tuple(range(1, count+1)):
        raise ValueError("native/external shell node identity mismatch")
    if owner._shell_count != 6*count:
        raise ValueError("native shell coordinate count mismatch")
    expected_fixed = np.r_[np.arange(6), 6*(n+1)+np.arange(6)]
    fixed = owner._fixed[owner._fixed < owner._shell_count]
    if not np.array_equal(fixed, expected_fixed):
        raise ValueError("native/external root support mismatch")
    expected_edge = {str(node) for node in range(2, n+2)}
    action = external["shell_interface_action"]
    edge_displacement = external["shell_edge_displacement"]
    if set(action) != expected_edge or set(edge_displacement) != expected_edge:
        raise ValueError("complete external shell-edge action required")
    f = np.zeros(owner._shell_count)
    ext_u = np.zeros(owner._shell_count)
    for key in expected_edge:
        index = owner._shell_index[int(key)]
        a = np.asarray(action[key], dtype=float)
        u = np.asarray(edge_displacement[key], dtype=float)
        if a.shape != (6,) or u.shape != (6,) or not (
                np.isfinite(a).all() and np.isfinite(u).all()):
            raise ValueError("finite external shell action/displacement required")
        f[6*index:6*index+6] = a
        ext_u[6*index:6*index+6] = u
    ext_work = float(f@ext_u)
    if abs(ext_work-external["shell_internal_work_j"]) > (
            1e-8*max(abs(ext_work), 1e-12)):
        raise ValueError("external shell work is not conjugate to edge action")
    k = owner._K[:owner._shell_count, :owner._shell_count].tocsc()
    free = np.setdiff1d(np.arange(owner._shell_count), fixed)
    u = np.zeros(owner._shell_count)
    u[free] = splu(k[free, :][:, free]).solve(f[free])
    if not np.isfinite(u).all():
        raise ValueError("finite native common-action shell response required")
    r = np.asarray(k@u-f).reshape(-1)
    free_error = float(np.max(np.abs(r[free])))
    native_work = float(f@u)
    internal_work = float(u@(k@u))
    applied_force, applied_moment = _wrench(owner._shell_xyz, f)
    fixed_reactions = np.zeros_like(f)
    fixed_reactions[fixed] = r[fixed]
    support_force, support_moment = _wrench(owner._shell_xyz, fixed_reactions)
    force_error = float(np.max(np.abs(applied_force+support_force)))
    moment_error = float(np.max(np.abs(applied_moment+support_moment)))
    if (free_error > 1e-7*max(np.max(np.abs(f)), 1.) or
            abs(native_work-internal_work) >
            1e-8*max(abs(native_work), 1e-12) or
            force_error > 1e-5 or moment_error > 1e-5 or
            native_work <= 0. or ext_work <= 0.):
        raise ValueError("native common-action shell equilibrium/work failed: "
                         f"{free_error=}, {force_error=}, {moment_error=}")
    tip = owner._shell_index[n+1]
    return {"intervals": n, "case": external["case"],
            "external_shell_work_j": ext_work,
            "native_shell_work_j": native_work,
            "native_to_external_work_ratio": native_work/ext_work,
            "external_edge_tip": ext_u[6*tip:6*tip+6].tolist(),
            "native_edge_tip": u[6*tip:6*tip+6].tolist(),
            "native_free_residual_max": free_error,
            "native_work_error_j": abs(native_work-internal_work),
            "native_support_force_error_n": force_error,
            "native_support_moment_error_nm": moment_error,
            "shell_coordinates": owner._shell_count,
            "shell_matrix_nnz": int(k.nnz)}


def main(reference: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("preserve prior common-action evidence")
    raw = reference.read_bytes()
    external = json.loads(raw)
    if (external.get("failure") is not None or
            external.get("constraint_form") != "chain" or
            external.get("scope") != "4/8/16 Fy/Mx" or
            len(external.get("cases", ())) != 6):
        raise ValueError("complete chained six-case external reference required")
    report = {"status": "BL5_COMMON_SHELL_ACTION_DEVELOPMENT_ONLY",
              "source_git": subprocess.check_output(
                  ("git", "rev-parse", "HEAD"), text=True).strip(),
              "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference_sha256": sha256(raw).hexdigest(),
              "imported_anyfesolver": str(Path(anyfesolver.__file__).resolve()),
              "results": [], "failure": None,
              "limitations": ("Same recovered nodal shell action with identical "
                              "root supports; different shell formulations. "
                              "Not a full coupled same-law oracle, physical "
                              "bond qualification or multiplier selection.")}
    try:
        with threadpool_limits(limits=1):
            report["threadpools"] = threadpool_info()
            for n in (4, 8, 16):
                owner = stiffened_shell.create_analysis(
                    definition=_bl5_joint_mesh_definition(n, n, 1))
                try:
                    for case in ("fy", "mx"):
                        record = next(value for value in external["cases"]
                                      if value["intervals"] == n and
                                      value["case"] == case)
                        report["results"].append(_case(owner, record))
                finally:
                    owner.close()
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                     allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"cases": len(report["results"]),
                      "failure": report["failure"],
                      "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.reference, args.output)
