"""Installed native counterpart of the independently derived fixed B2 reference.

Only the reference fixture generator is imported; its OpenSees main is not run.
Connection rows must agree before an external loaded comparison is meaningful.
No mesh convergence or complete E01 qualification is claimed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
import numba
import threadpoolctl
from anyfesolver import stiffened_shell
import b2_e01_opensees_finite_point as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pools = threadpoolctl.threadpool_info()
    assert pools and all(p["num_threads"] == 1 for p in pools)
    assert numba.config.DISABLE_JIT and numba.config.NUMBA_NUM_THREADS == 1
    summary = {"status": "FAILED", "python": sys.version,
               "executable": sys.executable, "threads": pools, "jit_disabled": True,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "reference_source_sha256": hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
               "imported_facade": stiffened_shell.__file__, "panels": []}
    try:
        law = stiffened_shell.LinearSection.isotropic(
            E=reference.E, nu=reference.NU, area=reference.A,
            Iy=reference.IY, Iz=reference.IZ, J=reference.J,
            ky=reference.KY, kz=reference.KZ)
        for sign in (-1, 1):
            model = reference.panel_input(sign)
            nodes = model["coordinates_m"]
            shells = tuple(stiffened_shell.Q4PanelShell(
                f"shell-{i}", cell, reference.E, reference.NU, reference.H)
                for i, cell in enumerate(reference.CELLS))
            members = tuple(stiffened_shell.PhysicalB2Member(
                f"beam-{i}", f"reference/beam-{i}", beam,
                tuple(nodes[node] for node in beam), (0., 1., 0.),
                "scalar-elastic", law)
                for i, beam in enumerate(reference.BEAMS))
            points = tuple(stiffened_shell.Q4B2Point(
                f"point-{i}", f"shell-{i}", f"beam-{i}", .5)
                for i in range(2))
            free = [i for i in range(54) if i not in model["fixed_coordinates"]]
            definition = stiffened_shell.B2Q4PanelDefinition(
                stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
                {node: nodes[node] for node in reference.SHELL_NODES},
                shells, members, points, sparse.eye(54, format="csr")[:, free],
                np.zeros(54))
            owner = stiffened_shell.create_analysis(definition=definition)
            try:
                # Private matrix access is a diagnostic law-equivalence check;
                # construction and solves go through the installed public API.
                observed = owner._C.toarray()
                expected = np.asarray(model["constraint_rows"])
                row_error = float(np.max(np.abs(observed-expected)))
                assert np.isfinite(row_error) and row_error <= 1e-10, row_error
                record = {"sign": sign, "constraint_row_max_error": row_error,
                          "constraint_rows": observed.tolist(), "cases": []}
                summary["panels"].append(record)
                for name, component, magnitude in reference.LOADS:
                    load = np.zeros(54)
                    load[48+component] = magnitude
                    result = owner.solve_linear(load)
                    case = {"name": name, "result": asdict(result)}
                    record["cases"].append(case)
                    assert result.status == "completed", result.failure_reason
                assert owner._handle.factorization_count == 1
            finally:
                owner.close()
        summary["status"] = "PASSED_NATIVE_FIXED_SCREEN_ONLY"
    finally:
        (output/"native.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": summary["status"], "panels": len(summary["panels"]),
                      "cases": sum(len(p["cases"]) for p in summary["panels"])}))


if __name__ == "__main__":
    main()
