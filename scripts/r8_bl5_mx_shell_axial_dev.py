"""One fixed-beam/weak-action C3D20 shell-axial discriminator at 32x8."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import anyfesolver
import numba
import numpy as np
import psutil
from scipy import sparse
from threadpoolctl import threadpool_info

from scripts import r8_bl5_mx_matched_refinement_dev as matched
from scripts import r8_bl5_mx_coupled_shell_refinement_dev as shell_ref
from scripts import r8_bl5_mx_torsion_refinement_dev as beam_ref
from scripts.r8_bl5_mx_shell_thickness_dev import diagnostic_solve

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA256 = "c81720f503d972dff685ec114446eece5abeef4c1d7d9a1d9df56cf43ff79729"
BEAM_LEVEL = (16, 8, 4)
SHELL_LEVEL = (32, 8, 4)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous discriminator output")
    prior = json.loads(args.independent.read_text(encoding="utf-8"))
    if (digest(args.independent) != PARENT_SHA256 or
            prior["status"] != "DEVELOPMENT_MATCHED_REFINEMENT_NOT_QUALIFIED" or
            tuple(prior["level"]) != BEAM_LEVEL or prior["multiplier_rows"] != 80):
        raise ValueError("exact saved 16x8x4 matched case required")
    pools = threadpool_info()
    runtime = {"cpu_affinity": psutil.Process().cpu_affinity(),
               "numba_disable_jit": int(numba.config.DISABLE_JIT),
               "numba_threads": int(numba.get_num_threads()),
               "threadpools": pools,
               "anyfesolver_import": str(Path(anyfesolver.__file__).resolve())}
    if (runtime["cpu_affinity"] != [0] or runtime["numba_disable_jit"] != 1 or
            runtime["numba_threads"] != 1 or not pools or
            any(item["num_threads"] != 1 for item in pools) or
            Path(anyfesolver.__file__).resolve().parent != ROOT / "src" / "anyfesolver"):
        raise ValueError("effective source/runtime does not match bounded contract")
    started = time.perf_counter()
    beam_nodes, beam_elements = beam_ref._mesh(*BEAM_LEVEL)
    beam = beam_ref._body(beam_nodes, beam_elements, BEAM_LEVEL)
    tip, meta = beam_ref._port(beam_nodes, beam_elements, 1., BEAM_LEVEL)
    shell_nodes, shell_elements = shell_ref._shell_mesh(*SHELL_LEVEL)
    shell = shell_ref._shell_body(shell_nodes, shell_elements, SHELL_LEVEL)
    model = shell_ref._combined(shell_nodes, shell_elements, beam_nodes, beam_elements)
    body = sparse.block_diag((shell, beam), format="csr")
    c, scale = matched._weak_rows(model, BEAM_LEVEL[0])
    load = np.zeros(body.shape[0])
    load[-beam.shape[0]:] = tip.T @ np.array((0., 0., 0., 10., 0., 0.))
    result = diagnostic_solve(model, body, c, scale, load, tip)
    old = float(prior["resolved"]["applied_work_J"])
    new = float(result["applied_work_J"])
    record = {"identity": "R8_BL5_MX_FIXED_BEAM_SHELL_AXIAL_DEV_V1",
              "status": ("DEVELOPMENT_DISCRIMINATOR_NOT_QUALIFIED" if result["status"] == "PASSED"
                         else "DEVELOPMENT_CONTROL_FAILED"),
              "source_sha256": digest(Path(__file__)),
              "independent_sha256": digest(args.independent),
              "fixed_beam_level": BEAM_LEVEL,
              "fixed_interface_rows": int(c.shape[0]),
              "old_shell_level": BEAM_LEVEL,
              "new_shell_level": SHELL_LEVEL,
              "old_resolved_work_J": old,
              "new_resolved_work_J": new,
              "relative_shell_axial_change": (new-old)/old,
              "native_work_J": prior["native"]["applied_work_J"],
              "new_native_over_resolved": prior["native"]["applied_work_J"]/new,
              "new_shell_elements": len(shell_elements),
              "fixed_beam_elements": len(beam_elements),
              "tip_port": meta, "result": result,
              "runtime": runtime, "solve_wall_seconds": time.perf_counter()-started,
              "limitation": "One shell-axial refinement only; does not establish convergence or select BL5"}
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in (
        "status", "relative_shell_axial_change", "new_native_over_resolved",
        "solve_wall_seconds")}))
    if record["status"] != "DEVELOPMENT_DISCRIMINATOR_NOT_QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
