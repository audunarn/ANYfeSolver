"""Bounded BL5 versus strong-nodal five-row common-field work discriminator.

This compares only the reference-linear work maps on identical node sets.
It is development diagnosis, not a bonded-line qualification or a loaded solve.
"""

from __future__ import annotations

from hashlib import sha256
import argparse
import json
import math
from pathlib import Path
import subprocess

import anyfesolver
from anyfesolver import stiffened_shell
import numpy as np
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_info, threadpool_limits

from test_stiffened_shell_l1_assembly_experimental import (
    _bl5_joint_mesh_definition,
)


ROOT = Path(__file__).resolve().parents[1]
SHELL_TOP = .035/2.
BEAM_BOTTOM = -.05


def _shell_field(x: float, kind: str) -> np.ndarray:
    if kind == "affine":
        return np.array((.0002*x, .001*x, 0., 0., 0., 0.))
    if kind == "smooth":
        return np.array((.0002*x*x, .001*math.sin(math.pi*x),
                         .0003*math.sin(math.pi*x),
                         .01*math.sin(math.pi*x),
                         .005*math.sin(2*math.pi*x), 0.))
    raise ValueError("declared common field required")


def _run(owner, intervals: int, span: int, kind: str,
         lift: str) -> dict:
    q = np.zeros(owner._K.shape[0])
    direct_rows = []
    for edge in range(intervals+1):
        shell_id = edge+1
        beam_id = 101+edge
        x = edge/intervals
        shell = _shell_field(x, kind)
        beam = shell.copy()
        beam[:3] += np.cross(shell[3:], (0., 0., SHELL_TOP))
        beam[:3] -= np.cross(beam[3:], (0., 0., BEAM_BOTTOM))
        shell_slot = 6*owner._shell_index[shell_id]
        beam_slot = owner._shell_count+6*owner._beam_index[beam_id]
        q[shell_slot:shell_slot+6] = shell
        q[beam_slot:beam_slot+6] = beam
        direct_rows.append(np.r_[
            shell[:3]+np.cross(shell[3:], (0., 0., SHELL_TOP))-
            beam[:3]-np.cross(beam[3:], (0., 0., BEAM_BOTTOM)),
            shell[3:5]-beam[3:5]])
    direct = np.asarray(direct_rows)
    if np.max(np.abs(direct)) > 1e-14:
        raise AssertionError("the imposed strong nodal constraints are not zero")
    if np.max(np.abs(q[owner._fixed])) > 1e-14:
        raise AssertionError("common field violates a declared support")
    retained_start = owner._shell_count+6*len(owner._beam_ids)
    if lift == "equilibrated":
        block = owner._K[retained_start:, retained_start:].tocsc()
        rhs = -owner._K[retained_start:, :retained_start]@q[:retained_start]
        q[retained_start:] = spsolve(block, rhs)
        if not np.all(np.isfinite(q[retained_start:])):
            raise ValueError("finite native retained lift required")
        defect = np.asarray(block@q[retained_start:]-rhs).ravel()
        retained_relative = float(np.linalg.norm(defect)/max(
            np.linalg.norm(rhs),
            np.linalg.norm(block@q[retained_start:]), 1e-20))
        retained_absolute = float(np.linalg.norm(defect))
    elif lift == "zero":
        retained_relative = None
        retained_absolute = None
    else:
        raise ValueError("declared native retained lift required")
    weak = np.asarray(owner._C@q).reshape(5, intervals//span)
    # Coefficients are illustrative physical line densities on the integrated
    # weak rows: 100 N/m in y and 10 Nm/m on the second director channel.
    dual = np.zeros_like(weak)
    dual[1] = 100.
    dual[4] = 10.
    retained = owner._C[:, retained_start:]
    retained_column_norms = np.asarray(abs(retained).sum(axis=0)).ravel()
    return {
        "intervals": intervals, "span": span, "field": kind,
        "retained_lift": lift,
        "retained_equilibrium_relative": retained_relative,
        "retained_equilibrium_absolute": retained_absolute,
        "retained_coordinate_norm": float(np.linalg.norm(q[retained_start:])),
        "strong_nodal_gap_max": float(np.max(np.abs(direct))),
        "weak_integrated_gap_max_by_channel":
            np.max(np.abs(weak), axis=1).tolist(),
        "weak_test_work_j": float(np.sum(dual*weak)),
        "retained_columns": int(retained.shape[1]),
        "retained_columns_affecting_bl5":
            int(np.count_nonzero(retained_column_norms > 1e-14)),
        "maximum_retained_column_sum":
            float(np.max(retained_column_norms)),
        "weak_rows": int(owner._C.shape[0]),
    }


def main(output: Path) -> None:
    if output.exists():
        raise FileExistsError("preserve prior common-field evidence")
    expected = (ROOT / "src" / "anyfesolver" / "__init__.py").resolve()
    if Path(anyfesolver.__file__).resolve() != expected:
        raise RuntimeError("common-field diagnostic imported another checkout")
    result = {
        "status": "BL5_COMMON_FIELD_DEVELOPMENT_ONLY",
        "source_git": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "imported_anyfesolver": str(expected),
        "cases": [],
        "limitations": ("Strong nodal and weak native work are compared only "
                        "on imposed common nodal fields with either zero or "
                        "uncoupled-equilibrated native retained increments. "
                        "OpenSees has no coordinate counterpart for native "
                        "retained-only modes. No loaded or nonlinear solve.")}
    with threadpool_limits(limits=1):
        result["threadpools"] = threadpool_info()
        for intervals in (4, 8):
            for span in (1, 2):
                owner = stiffened_shell.create_analysis(
                    definition=_bl5_joint_mesh_definition(
                        intervals, intervals, span))
                try:
                    for kind in ("affine", "smooth"):
                        for lift in ("zero", "equilibrated"):
                            result["cases"].append(_run(
                                owner, intervals, span, kind, lift))
                finally:
                    owner.close()
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"cases": len(result["cases"]), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
