"""Bounded full-panel same-law development comparison for the sparse L1 owner.

This holds the physical panel, eccentric material line, supports, section and
two independent loads fixed while refining shell and native beam together.
It is neither a converged reference nor a formal F03 qualification run.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import psutil
import scipy
from threadpoolctl import threadpool_info

import anyfesolver
from anyfesolver import b3_ge, stiffened_shell
from anyfesolver._stiffened_shell_l1_assembly_experimental import SPACE_A, SPACE_B


ROOT = Path(__file__).resolve().parents[1]
LINE = (.5, 1.)
LIMIT_BYTES = 8*1024**3


def definition(segments: int, space: str):
    if type(segments) is not int or segments not in (2, 4, 8, 16):
        raise ValueError("registered bounded L1 refinement level required")
    e, nu = 210e9, .3
    g = e/(2*(1+nu))
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((e*.01, g*.008, g*.008, g*(2*8.333333333333333e-6),
                 e*8.333333333333333e-6, e*8.333333333333333e-6)),
        np.eye(6), 1e20, 1e9)
    xs = np.linspace(0., 1., 2*segments+1)
    ys = np.linspace(-.1, .1, segments+1)
    beam_points = np.column_stack((xs, np.zeros(len(xs)),
                                   np.full(len(xs), .0675)))
    beams = tuple(b3_ge.define_beam(
        "b3-ge", element+1,
        (101+2*element, 102+2*element, 103+2*element),
        beam_points[2*element:2*element+3],
        np.tile(np.eye(3), (3, 1, 1)), section, np.eye(6))
        for element in range(segments))
    width = len(xs)
    coordinates = tuple((float(x), float(y), 0.) for y in ys for x in xs)
    q4 = tuple((1+j*width+i, 2+j*width+i,
                2+(j+1)*width+i, 1+(j+1)*width+i)
               for j in range(segments) for i in range(2*segments))
    shell = stiffened_shell.ShellPanel(
        tuple(range(1, len(coordinates)+1)), coordinates, q4,
        .035, e, nu, 7850.)
    center = segments//2
    line_nodes = tuple(1+center*width+i for i in range(segments, 2*segments+1))
    halfcells = tuple(divmod(half, 2) for half in range(segments, 2*segments))
    line = stiffened_shell.ExperimentalL1Line(
        line_nodes, halfcells, space)
    fixed = tuple(stiffened_shell.FixedCoordinate("shell", 1+j*width, axis)
                  for j in range(len(ys)) for axis in range(6)) + tuple(
                  stiffened_shell.FixedCoordinate("beam", 101, axis)
                  for axis in range(6))
    model = stiffened_shell.LinearDefinition(
        stiffened_shell.B3_GE_E4_PL_STIFFENED_SHELL_V1,
        shell, beams, (), fixed, (line,))
    return model, line_nodes[-1], 101+2*segments


def one_variant(segments: int, space: str):
    model, shell_tip, beam_tip = definition(segments, space)
    start = time.perf_counter()
    owner = stiffened_shell.create_analysis(definition=model)
    compiled = time.perf_counter()
    preflight = owner.preflight()
    if (owner._A.shape[0] > 100_000 or owner._A.nnz == 0 or
            not np.isfinite(owner._A.data).all()):
        raise RuntimeError("bounded finite sparse L1 operator required")
    if psutil.Process().memory_info().rss > LIMIT_BYTES:
        raise MemoryError("L1 resource ceiling exceeded after compilation")
    actions = (
        ("shell_fy", stiffened_shell.NodalAction(
            "shell", shell_tip, (0., 1000., 0., 0., 0., 0.))),
        ("beam_tip_mx", stiffened_shell.NodalAction(
            "beam", beam_tip, (0., 0., 0., 100., 0., 0.))),
    )
    results = {}
    for label, action in actions:
        result = owner.solve_linear((action,))
        if (result.status != "completed" or
                result.free_equilibrium_ratio > 1e-8 or
                result.constraint_max > 1e-9):
            raise RuntimeError("supported L1 equilibrium/compatibility failed")
        if label == "shell_fy":
            shell_slot = model.shell.node_ids.index(shell_tip)
            work = 1000.*result.shell_displacement[6*shell_slot+1]
        else:
            beam_slot = owner._beam_index[beam_tip]
            work = 100.*result.beam_coordinates[6*beam_slot+3]
        results[label] = {
            "load_displacement_work_Nm": float(work),
            "free_equilibrium_ratio": result.free_equilibrium_ratio,
            "constraint_max": result.constraint_max,
            "generation": result.generation,
            "support_reaction_norm": float(np.linalg.norm(result.reactions)),
        }
        if psutil.Process().memory_info().rss > LIMIT_BYTES:
            raise MemoryError("L1 resource ceiling exceeded after solve")
    solved = time.perf_counter()
    report = {
        "space": space, "shell_elements": len(model.shell.q4_elements),
        "beam_elements": len(model.beams),
        "line_interval_m": list(LINE),
        "line_segments": segments,
        "line_quadrature_samples": preflight["experimental_line_samples"],
        "matrix_rows": int(owner._A.shape[0]),
        "matrix_nnz": int(preflight["matrix_nnz"]),
        "factorizations": preflight["reference_linear_factorizations"],
        "factorizations_after_solve": owner.preflight()[
            "reference_linear_factorizations"],
        "compile_seconds": compiled-start,
        "two_solve_seconds": solved-compiled,
        "rss_after_bytes": psutil.Process().memory_info().rss,
        "loads": results,
    }
    owner.close()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    threads = threadpool_info()
    if any(item.get("num_threads", 0) > 8 for item in threads):
        raise RuntimeError("L1 numerical-thread ceiling exceeded")
    started = time.perf_counter()
    cases = [one_variant(args.segments, space) for space in (SPACE_A, SPACE_B)]
    if time.perf_counter()-started > 600.:
        raise RuntimeError("L1 run exceeded bounded diagnostic ceiling")
    source = Path(__file__).read_bytes()
    report = {
        "status": "development_not_qualified",
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "runner_sha256": sha256(source).hexdigest(),
        "segments": args.segments,
        "python": __import__("sys").version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "imported_package": str(Path(anyfesolver.__file__).resolve()),
        "thread_environment": {key: os.environ.get(key) for key in (
            "NUMBA_DISABLE_JIT", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "effective_threadpools": [
            {key: pool.get(key) for key in ("internal_api", "num_threads",
                                             "filepath")}
            for pool in threads],
        "elapsed_seconds": time.perf_counter()-started,
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True),
                           encoding="utf-8")
    print(json.dumps({"segments": args.segments,
                      "work_Nm": {c["space"]: {k: v[
                          "load_displacement_work_Nm"] for k, v in c[
                              "loads"].items()} for c in cases},
                      "elapsed_seconds": report["elapsed_seconds"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
