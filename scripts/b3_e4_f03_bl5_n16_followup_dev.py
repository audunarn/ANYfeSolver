"""Targeted 16-interval native BL5 follow-up to an independent nodal reference.

This is a Level-A response trend, not a formal F03 run or space selection.
The historical 2/4/8 runner and its evidence remain unchanged.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import anyfesolver
from anyfesolver import stiffened_shell
from threadpoolctl import threadpool_info, threadpool_limits

from b3_e4_f03_bl5_joint_growth_dev import _case
from test_stiffened_shell_l1_assembly_experimental import (
    _bl5_joint_mesh_definition,
)


def main(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"preserve prior follow-up evidence: {output}")
    report = {
        "status": "BL5_N16_REFERENCE_FOLLOWUP_DEVELOPMENT_ONLY",
        "source_git": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True).strip(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "imported_anyfesolver": str(Path(anyfesolver.__file__).resolve()),
        "geometry_m": {"length": 1., "width": .2,
                       "shell_thickness": .035,
                       "beam_axis_height": .0675},
        "loads": {"fy": [0., 100., 0., 0., 0., 0.],
                  "mx": [0., 0., 0., 10., 0., 0.]},
        "results": {},
        "failure": None,
        "limitations": ("One 16/16 Q4/native B3 co-refinement level, complete "
                        "P0 BL5 spans 1 and 2. Compare only with the saved "
                        "4/8 native and independent 4/8/16 nodal screens. "
                        "Not an F03 assessment, physical bond qualification, "
                        "mixed-system stability proof, or space selection."),
    }
    try:
        with threadpool_limits(limits=1):
            report["threadpools"] = threadpool_info()
            for span in (1, 2):
                owner = stiffened_shell.create_analysis(
                    definition=_bl5_joint_mesh_definition(16, 16, span))
                try:
                    key = f"shell16_beam16_span{span}"
                    report["results"][key] = {
                        "identity": owner.preflight()["identity"],
                        "matrix_order": int(owner._A.shape[0]),
                        "matrix_nnz": int(owner._A.nnz),
                        "multiplier_rows": int(owner._C.shape[0]),
                        "fy": _case(owner, 117, "fy", tuple(report["loads"]["fy"])),
                        "mx": _case(owner, 117, "mx", tuple(report["loads"]["mx"])),
                        "factorizations": owner.preflight()[
                            "reference_linear_factorizations"],
                    }
                finally:
                    owner.close()
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                     allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "cases": len(report["results"])*2,
                      "failure": report["failure"],
                      "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.output)
