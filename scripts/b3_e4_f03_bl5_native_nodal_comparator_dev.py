"""Compare weak BL5 with five-row nodal ties on the same native K.

This is a bounded reference-linear allocation experiment, not a production
attachment, external qualification, or multiplier-space selection.
"""

from __future__ import annotations

from hashlib import sha256
import argparse
import json
from pathlib import Path
import subprocess

import anyfesolver
from anyfesolver import stiffened_shell
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info, threadpool_limits

from test_stiffened_shell_l1_assembly_experimental import (
    _bl5_joint_mesh_definition,
)


ROOT = Path(__file__).resolve().parents[1]
INTERVALS = 16
RS = .035/2.
RB = -.05
LOADS = {"fy": (0., 100., 0., 0., 0., 0.),
         "mx": (0., 0., 0., 10., 0., 0.)}


def _nodal_rows(owner) -> sparse.csr_matrix:
    rows, columns, values = [], [], []

    def put(row, column, value):
        rows.append(row)
        columns.append(column)
        values.append(value)

    for edge in range(INTERVALS+1):
        shell = 6*owner._shell_index[edge+1]
        beam = owner._shell_count+6*owner._beam_index[101+edge]
        row = 5*edge
        # Relative top/bottom position; eccentric rotational arms are
        # included once. The two physical director rotations are tied.
        put(row, beam, 1.)
        put(row, shell, -1.)
        put(row, beam+4, RB)
        put(row, shell+4, -RS)
        put(row+1, beam+1, 1.)
        put(row+1, shell+1, -1.)
        put(row+1, beam+3, -RB)
        put(row+1, shell+3, RS)
        put(row+2, beam+2, 1.)
        put(row+2, shell+2, -1.)
        put(row+3, beam+3, 1.)
        put(row+3, shell+3, -1.)
        put(row+4, beam+4, 1.)
        put(row+4, shell+4, -1.)
    result = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(5*(INTERVALS+1), owner._K.shape[0])).tocsr()
    result.sum_duplicates()
    return result


def _inventory(owner, shell_action):
    nodes = []
    for edge in range(INTERVALS+1):
        node = edge+1
        row = owner._shell_index[node]
        wrench = np.asarray(shell_action[6*row:6*row+6])
        xyz = owner._shell_xyz[row]
        nodes.append((node, xyz, wrench))

    def reduced(items):
        force = sum((w[:3] for _, _, w in items), np.zeros(3))
        moment = sum((w[3:]+np.cross(x, w[:3])
                      for _, x, w in items), np.zeros(3))
        return {"force_n": force.tolist(), "moment_nm": moment.tolist()}

    return {"all": reduced(nodes),
            "nonroot": reduced(nodes[1:]),
            "root": reduced(nodes[:1])}


def _support_inventory(owner, equilibrium):
    force = np.zeros(3)
    moment = np.zeros(3)
    for node in owner._shell_ids:
        row = owner._shell_index[node]
        slot = 6*row
        if not any(slot+axis in owner._fixed for axis in range(6)):
            continue
        wrench = np.asarray(equilibrium[slot:slot+6])
        xyz = owner._shell_xyz[row]
        force += wrench[:3]
        moment += wrench[3:]+np.cross(xyz, wrench[:3])
    return {"force_n": force.tolist(), "moment_nm": moment.tolist()}


def _load(owner, case):
    full = np.zeros(owner._K.shape[0])
    slot = owner._shell_count+6*owner._beam_index[101+INTERVALS]
    full[slot:slot+6] = LOADS[case]
    return full


def _metrics(owner, q, multipliers, rows, load):
    q = np.asarray(q)
    lam = np.asarray(multipliers)
    equilibrium = np.asarray(owner._K@q+rows.T@lam-load).ravel()
    gap = np.asarray(rows@q).ravel()
    shell_action = np.asarray(-rows.T[:owner._shell_count, :]@lam).ravel()
    shell_work = float(shell_action@q[:owner._shell_count])
    shell_internal = float(q[:owner._shell_count]@
                           (owner._K[:owner._shell_count,
                                     :owner._shell_count]@
                            q[:owner._shell_count]))
    free = owner._free
    return {
        "applied_work_j": float(load@q),
        "shell_action_work_j": shell_work,
        "shell_internal_work_j": shell_internal,
        "shell_work_error_j": shell_work-shell_internal,
        "gap_max": float(np.max(np.abs(gap))),
        "free_equilibrium_max": float(np.max(np.abs(equilibrium[free]))),
        "free_equilibrium_relative": float(np.linalg.norm(
            equilibrium[free])/max(
                np.linalg.norm(load[free]),
                np.linalg.norm((owner._K@q)[free]), 1.)),
        "support_action": _support_inventory(owner, equilibrium),
        "interface_shell_action": _inventory(owner, shell_action),
        "tip_translation_m": q[owner._shell_count+
                               6*owner._beam_index[101+INTERVALS]:
                               owner._shell_count+
                               6*owner._beam_index[101+INTERVALS]+3].tolist(),
        "tip_rotation_rad": q[owner._shell_count+
                              6*owner._beam_index[101+INTERVALS]+3:
                              owner._shell_count+
                              6*owner._beam_index[101+INTERVALS]+6].tolist(),
    }


def _strong_case(owner, rows, case):
    free = owner._free
    reduced = rows[:, free]
    matrix = sparse.bmat([
        [owner._K[free, :][:, free], reduced.T],
        [reduced, None]], format="csc")
    load = _load(owner, case)
    rhs = np.r_[load[free], np.zeros(rows.shape[0])]
    factor = splu(matrix)
    solution = factor.solve(rhs)
    solution += factor.solve(rhs-matrix@solution)
    if not np.isfinite(solution).all():
        raise ValueError("finite native nodal comparison required")
    q = np.zeros(owner._K.shape[0])
    q[free] = solution[:len(free)]
    lam = solution[len(free):]
    return _metrics(owner, q, lam, rows, load), matrix, q, lam


def _same_shell_action(owner, rows, q, lam, external):
    count = owner._shell_count
    native = np.asarray(-rows.T[:count, :]@lam).ravel()
    reference = np.zeros(count)
    keys = {str(node) for node in range(2, INTERVALS+2)}
    if set(external["shell_interface_action"]) != keys:
        raise ValueError("complete external nonroot edge action required")
    for key, wrench in external["shell_interface_action"].items():
        index = owner._shell_index[int(key)]
        reference[6*index:6*index+6] = wrench
    free = np.setdiff1d(np.arange(count),
                        owner._fixed[owner._fixed < count])
    shell = owner._K[:count, :count].tocsc()
    response = np.zeros(count)
    response[free] = splu(shell[free, :][:, free]).solve(reference[free])
    own_work = float(native@q[:count])
    reference_work = float(reference@response)
    cross_work = float(reference@q[:count])
    difference = own_work+reference_work-2*cross_work
    if (own_work <= 0. or reference_work <= 0. or
            difference < -1e-10*max(own_work, reference_work)):
        raise ValueError("positive same-shell action work required")
    delta = (native-reference).reshape(-1, 6)[
        [owner._shell_index[node] for node in range(2, INTERVALS+2)]]
    return {
        "external_on_native_shell_work_j": reference_work,
        "native_shell_action_work_j": own_work,
        "cross_work_j": cross_work,
        "relative_action_difference_compliance_norm": float(np.sqrt(
            max(difference, 0.)/reference_work)),
        "nonroot_force_difference_l2_n": float(np.linalg.norm(delta[:, :3])),
        "nonroot_moment_difference_l2_nm": float(np.linalg.norm(delta[:, 3:])),
    }


def main(external_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("preserve prior nodal comparator evidence")
    expected = (ROOT / "src" / "anyfesolver" / "__init__.py").resolve()
    if Path(anyfesolver.__file__).resolve() != expected:
        raise RuntimeError("native nodal comparator imported another checkout")
    external_bytes = external_path.read_bytes()
    external = json.loads(external_bytes)
    if external.get("failure") is not None or external.get("constraint_form") != "chain":
        raise ValueError("completed chained five-row external reference required")
    references = {(item["intervals"], item["case"]): item
                  for item in external["cases"]}
    if any((INTERVALS, case) not in references for case in LOADS):
        raise ValueError("both 16-interval external actions required")
    report = {
        "status": "BL5_NATIVE_NODAL_COMPARATOR_DEVELOPMENT_ONLY",
        "source_git": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "external_sha256": sha256(external_bytes).hexdigest(),
        "imported_anyfesolver": str(expected),
        "cases": [],
        "limitations": ("Reference-linear same-native-K comparison. The five-row "
                        "nodal comparator omits native retained-port work, "
                        "whereas BL5 couples it. External ASDShellQ4 and "
                        "ElasticTimoshenkoBeam are distinct formulations. "
                        "No connection or multiplier selection follows alone.")}
    with threadpool_limits(limits=1):
        report["threadpools"] = threadpool_info()
        for span in (1, 2):
            owner = stiffened_shell.create_analysis(
                definition=_bl5_joint_mesh_definition(
                    INTERVALS, INTERVALS, span))
            try:
                nodal = _nodal_rows(owner)
                for case in LOADS:
                    action = stiffened_shell.NodalAction(
                        "beam", 101+INTERVALS, LOADS[case])
                    weak = owner.solve_linear((action,))
                    weak_q = np.r_[weak.shell_displacement,
                                   weak.beam_coordinates]
                    weak_metrics = _metrics(
                        owner, weak_q, weak.multipliers,
                        owner._C, _load(owner, case))
                    strong_metrics, matrix, strong_q, strong_lam = (
                        _strong_case(owner, nodal, case))
                    if (strong_metrics["free_equilibrium_relative"] > 1e-8 or
                            strong_metrics["gap_max"] > 1e-9 or
                            abs(strong_metrics["shell_work_error_j"]) > 1e-8):
                        raise ValueError("native nodal solve fails work/equilibrium")
                    report["cases"].append({
                        "span": span, "case": case,
                        "weak": weak_metrics, "native_nodal": strong_metrics,
                        "weak_vs_external_action": _same_shell_action(
                            owner, owner._C, weak_q, weak.multipliers,
                            references[INTERVALS, case]),
                        "nodal_vs_external_action": _same_shell_action(
                            owner, nodal, strong_q, strong_lam,
                            references[INTERVALS, case]),
                        "native_nodal_matrix_order": int(matrix.shape[0]),
                        "native_nodal_matrix_nnz": int(matrix.nnz),
                        "external_applied_work_j": float(
                            references[INTERVALS, case]["applied_work_j"]),
                    })
            finally:
                owner.close()
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"cases": len(report["cases"]), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("external", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.external, args.output)
