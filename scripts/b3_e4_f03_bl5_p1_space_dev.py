"""One coherent five-channel continuous-P1 BL5 development comparison.

This uses the existing native sample work, K and supports. It asks whether
P0 resolution or retained-port coupling explains the prior transfer gap.
It is not a multiplier-space selection or formal F03 execution.
"""

from __future__ import annotations

from hashlib import sha256
import argparse
import json
from pathlib import Path
import subprocess

import anyfesolver
from anyfesolver import stiffened_shell
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

from b3_e4_f03_bl5_native_nodal_comparator_dev import (
    INTERVALS, LOADS, _same_shell_action, _strong_case,
)
from test_stiffened_shell_l1_assembly_experimental import (
    _bl5_joint_mesh_definition,
)


ROOT = Path(__file__).resolve().parents[1]


def _continuous_p1(owner) -> sparse.csr_matrix:
    positions = np.array([owner._beam_nodes[node][0]
                          for node in owner._beam_ids])
    frames = np.array([owner._beam_nodes[node][1]
                       for node in owner._beam_ids])
    state = Mechanical(
        positions, np.zeros_like(positions), frames,
        np.tile(np.eye(3), (len(owner._beam_parts), 2, 1, 1)),
        np.zeros((len(owner._beam_parts), 18)))
    row_indices, columns, values = [], [], []
    for local_operator, slots, weight, edge, alpha in owner._line_samples:
        local = local_operator.evaluate(
            np.zeros((2, 6)), state, np.zeros(5))
        if np.max(np.abs(local.gap)) > 2e-13:
            raise ValueError("stress-free native BL5 trace required")
        for channel in range(5):
            for node, coefficient in ((edge, 1.-alpha), (edge+1, alpha)):
                row = channel*(INTERVALS+1)+node
                row_indices.extend([row]*len(slots))
                columns.extend(slots.tolist())
                values.extend((weight*coefficient*
                               local.gap_jacobian[channel]).tolist())
    result = sparse.coo_matrix(
        (values, (row_indices, columns)),
        shape=(5*(INTERVALS+1), owner._K.shape[0])).tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    return result


def main(external_path: Path, prior_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("preserve prior P1 evidence")
    expected = (ROOT / "src" / "anyfesolver" / "__init__.py").resolve()
    if Path(anyfesolver.__file__).resolve() != expected:
        raise RuntimeError("P1 diagnostic imported another checkout")
    external_bytes, prior_bytes = external_path.read_bytes(), prior_path.read_bytes()
    external, prior = json.loads(external_bytes), json.loads(prior_bytes)
    if (external.get("failure") is not None or
            prior.get("status") != "BL5_NATIVE_NODAL_COMPARATOR_DEVELOPMENT_ONLY"):
        raise ValueError("completed exact-input references required")
    references = {(item["intervals"], item["case"]): item
                  for item in external["cases"]}
    prior_cases = {(item["span"], item["case"]): item
                   for item in prior["cases"]}
    report = {
        "status": "BL5_P1_SPACE_DEVELOPMENT_ONLY",
        "source_git": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparator_runner_sha256": sha256((ROOT / "scripts" /
            "b3_e4_f03_bl5_native_nodal_comparator_dev.py").read_bytes()).hexdigest(),
        "external_sha256": sha256(external_bytes).hexdigest(),
        "prior_sha256": sha256(prior_bytes).hexdigest(),
        "imported_anyfesolver": str(expected), "cases": [],
        "limitations": ("One 16-interval five-channel P1 space only. "
                        "Reference-linear work/transfer, without the registered "
                        "seven ratios, growing-space stability, locking, "
                        "finite law, or formal F03 assessment.")}
    with threadpool_limits(limits=1):
        report["threadpools"] = threadpool_info()
        owner = stiffened_shell.create_analysis(
            definition=_bl5_joint_mesh_definition(
                INTERVALS, INTERVALS, 1))
        try:
            rows = _continuous_p1(owner)
            report["p1_rows"] = int(rows.shape[0])
            report["p1_nnz"] = int(rows.nnz)
            for case in LOADS:
                prior_strong = prior_cases[1, case]["native_nodal"]
                if (abs(prior_strong["applied_work_j"]-
                        prior_cases[2, case]["native_nodal"]["applied_work_j"]) >
                        1e-14):
                    raise ValueError("prior same-native nodal comparator changed")
                metrics, matrix, q, lam = _strong_case(owner, rows, case)
                if (metrics["free_equilibrium_relative"] > 1e-8 or
                        metrics["gap_max"] > 1e-9 or
                        abs(metrics["shell_work_error_j"]) > 1e-8):
                    raise ValueError("P1 native solve fails work/equilibrium")
                report["cases"].append({
                    "case": case, "p1": metrics,
                    "p1_vs_external_action": _same_shell_action(
                        owner, rows, q, lam, references[INTERVALS, case]),
                    "matrix_order": int(matrix.shape[0]),
                    "matrix_nnz": int(matrix.nnz),
                    "prior_p0_span_1_work_j": float(
                        prior_cases[1, case]["weak"]["applied_work_j"]),
                    "prior_p0_span_2_work_j": float(
                        prior_cases[2, case]["weak"]["applied_work_j"]),
                    "prior_native_nodal_work_j": float(
                        prior_strong["applied_work_j"]),
                    "external_nodal_work_j": float(
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
    parser.add_argument("prior", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.external, args.prior, args.output)
