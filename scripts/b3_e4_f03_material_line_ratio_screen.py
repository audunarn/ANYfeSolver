"""First bounded reciprocal beam:shell ratio screen for material-line rows.

Four multiplier intervals follow the union of native half-cell and shell
reference cuts. This is a development discriminator, not a stability or
physical-accuracy qualification.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import anyfesolver
import numpy as np
import scipy

from scripts._b3_e4_f03_native_fixture import build
from scripts.b3_e4_f03_material_line_channel_screen import _one_topology
from scripts.b3_e4_f03_plate_patch_work import local_grid
from scripts.b3_e4_f03_square_section_control import square_torsion_constant


def run() -> dict:
    expected = (Path(__file__).resolve().parents[1]/"src"/"anyfesolver").resolve()
    actual = Path(anyfesolver.__file__).resolve().parent
    if actual != expected:
        raise RuntimeError("ratio screen imported another ANYfeSolver checkout")
    knots = (0., .25, .5, .75, 1.)
    results = {}
    for label, shell_cells, beam_half_cells in (
            ("beam_to_shell_1_to_2", 4, 12),
            ("beam_to_shell_2_to_1", 2, 14)):
        xs, ys = local_grid(shell_cells, transverse_cells=4)
        beam = build(beam_half_cells, z_offset=.0675,
                     torsion_constant=square_torsion_constant(),
                     reference_profile="c1_tangent_2um")
        beam_bond_cuts = beam.abscissae[
            (beam.abscissae >= .9-1e-12) &
            (beam.abscissae <= 1.+1e-12)]
        shell_bond_cuts = xs[(xs >= .9-1e-12) & (xs <= 1.+1e-12)]
        expected_cuts = .9+.1*np.asarray(knots)
        if (len(beam_bond_cuts)-1 != beam_half_cells-10 or
                len(shell_bond_cuts)-1 != shell_cells or
                not np.allclose(np.unique(np.r_[beam_bond_cuts,
                                                shell_bond_cuts]),
                                expected_cuts, rtol=0., atol=1e-12)):
            raise ArithmeticError("declared reciprocal reference cuts changed")
        p0, p1 = "piecewise_constant", "continuous_linear"
        policy = ((p1, p0, p1, p1, p1, p0) if shell_cells == 4 else
                  (p1, p1, p1, p0, p0, p0))
        results[label] = {
            "shell_bond_cells": shell_cells,
            "native_bond_half_cells": int(len(beam_bond_cuts)-1),
            "shell_bond_cuts_m": shell_bond_cuts.tolist(),
            "native_bond_cuts_m": beam_bond_cuts.tolist(),
            "q4": _one_topology(xs, ys, beam, mixed_s3=False,
                                 diagonal="bl_tr", knots=knots,
                                 extra_policies=(("rank_compatible_ablation",
                                                  policy),)),
        }
    return {
        "classification": "private first reciprocal material-line ratio screen",
        "qualification": "development only; not mesh-uniform stability or G0B pass",
        "anyfesolver_import_path": str(Path(anyfesolver.__file__).resolve()),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
        "thread_settings": {key: os.environ.get(key) for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "result": results,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
