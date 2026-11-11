"""Decompose loaded BL5 multiplier work on native retained coordinates.

This bounded algebraic diagnostic compares the same native K under complete
P0, P1 and nodal five-channel constraints. It does not qualify or select one.
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
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info, threadpool_limits

from b3_e4_f03_bl5_native_nodal_comparator_dev import (
    INTERVALS, LOADS, _nodal_rows, _strong_case,
)
from b3_e4_f03_bl5_p1_space_dev import _continuous_p1
from test_stiffened_shell_l1_assembly_experimental import (
    _bl5_joint_mesh_definition,
)


ROOT = Path(__file__).resolve().parents[1]


def _decompose(owner, rows, q, lam, applied_work):
    retained_start = owner._shell_count+6*len(owner._beam_ids)
    qn, qr = q[:retained_start], q[retained_start:]
    cn, cr = rows[:, :retained_start], rows[:, retained_start:]
    krn = owner._K[retained_start:, :retained_start]
    krr = owner._K[retained_start:, retained_start:].tocsc()
    factor = splu(krr)
    uncoupled = factor.solve(-krn@qn)
    correction = qr-uncoupled
    retained_force = np.asarray(cr.T@lam).ravel()
    retained_equilibrium = np.asarray(
        krn@qn+krr@qr+retained_force).ravel()
    correction_equilibrium = np.asarray(
        krr@correction+retained_force).ravel()
    nodal_work = float(lam@(cn@qn))
    retained_work = float(lam@(cr@qr))
    lift_work = float(lam@(cr@uncoupled))
    correction_work = float(lam@(cr@correction))
    if (not np.isfinite((nodal_work, retained_work, lift_work,
                         correction_work)).all()):
        raise ValueError("finite retained work split required")
    return {
        "applied_work_j": float(applied_work),
        "nodal_constraint_work_j": nodal_work,
        "retained_constraint_work_j": retained_work,
        "uncoupled_lift_constraint_work_j": lift_work,
        "connection_induced_retained_work_j": correction_work,
        "retained_work_to_applied_abs_ratio": float(
            abs(retained_work)/max(abs(applied_work), 1e-20)),
        "connection_induced_work_to_applied_abs_ratio": float(
            abs(correction_work)/max(abs(applied_work), 1e-20)),
        "retained_equilibrium_absolute": float(np.linalg.norm(
            retained_equilibrium)),
        "correction_equilibrium_absolute": float(np.linalg.norm(
            correction_equilibrium)),
        "constraint_work_balance_j": nodal_work+retained_work,
        "weak_gap_max": float(np.max(np.abs(rows@q))),
        "retained_port_columns_nonzero": int(np.count_nonzero(
            np.asarray(abs(cr).sum(axis=0)).ravel() > 1e-14)),
    }


def _solve(owner, rows, case, *, owner_rows=False):
    if owner_rows:
        result = owner.solve_linear((stiffened_shell.NodalAction(
            "beam", 101+INTERVALS, LOADS[case]),))
        q = np.r_[result.shell_displacement, result.beam_coordinates]
        return q, result.multipliers, float(
            np.dot(LOADS[case], q[owner._shell_count+
                                  6*owner._beam_index[101+INTERVALS]:
                                  owner._shell_count+
                                  6*owner._beam_index[101+INTERVALS]+6]))
    metrics, _, q, lam = _strong_case(owner, rows, case)
    if metrics["free_equilibrium_relative"] > 1e-8:
        raise ValueError("reference-linear KKT solution required")
    return q, lam, metrics["applied_work_j"]


def main(output: Path):
    if output.exists():
        raise FileExistsError("preserve prior retained-work evidence")
    expected = (ROOT / "src" / "anyfesolver" / "__init__.py").resolve()
    if Path(anyfesolver.__file__).resolve() != expected:
        raise RuntimeError("retained-work diagnostic imported another checkout")
    report = {
        "status": "BL5_LOADED_RETAINED_WORK_DEVELOPMENT_ONLY",
        "source_git": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "imported_anyfesolver": str(expected), "cases": [],
        "limitations": ("Work-conjugate algebraic split at one 16-interval "
                        "linear model. Raw retained coordinate/equation norms "
                        "have mixed units and are not physical stability "
                        "metrics. No law or multiplier-space qualification.")}
    with threadpool_limits(limits=1):
        report["threadpools"] = threadpool_info()
        for span in (1, 2):
            owner = stiffened_shell.create_analysis(
                definition=_bl5_joint_mesh_definition(
                    INTERVALS, INTERVALS, span))
            try:
                rows = owner._C
                for case in LOADS:
                    q, lam, work = _solve(owner, rows, case, owner_rows=True)
                    report["cases"].append({
                        "policy": f"complete_p0_span_{span}", "case": case,
                        "decomposition": _decompose(owner, rows, q, lam, work)})
                if span != 1:
                    continue
                for name, rows in (("complete_continuous_p1",
                                    _continuous_p1(owner)),
                                   ("strong_nodal_five_row", _nodal_rows(owner))):
                    for case in LOADS:
                        q, lam, work = _solve(owner, rows, case)
                        report["cases"].append({
                            "policy": name, "case": case,
                            "decomposition": _decompose(
                                owner, rows, q, lam, work)})
            finally:
                owner.close()
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"cases": len(report["cases"]), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
