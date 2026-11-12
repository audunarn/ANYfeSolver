"""Write plasticity and nonlinear tangent qualification report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.plasticity_qualification import (  # noqa: E402
    DEFAULT_PLASTICITY_QUALIFICATION_PATH,
    write_plasticity_qualification_report,
)


def _markdown(report: Dict[str, Any]) -> str:
    material = report["material_point"]
    element = report["element_tangents"]
    shell_plastic = element["shell_layered_plastic"]
    lines = [
        "# Plasticity And Tangent Qualification Report",
        "",
        f"- Generated: {report.get('generated_at')}",
        f"- Status: {report.get('status')}",
        f"- Max yield residual: {material['max_abs_yield_residual']:.3e}",
        f"- Max algorithmic element tangent error: {element['max_algorithmic_tangent_error']:.3e}",
        f"- Beam fiber tangent error: {element['beam_fiber_plastic']['tangent_fd_relative_error']:.3e}",
        f"- Shell elastic tangent error: {element['shell_elastic']['tangent_fd_relative_error']:.3e}",
        f"- Shell plastic tangent error: {shell_plastic['tangent_fd_relative_error']:.3e}",
        f"- Shell plastic tangent status: {shell_plastic.get('tangent_status')}",
        "",
        "## Material Paths",
        "",
    ]
    for name, data in material["plastic_paths"].items():
        lines.append(f"- {name}: yield_residual={data.get('yield_residual', 0.0):.3e}")
    lines.extend(["", "## Known Limitations", ""])
    for item in report.get("known_limitations", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLASTICITY_QUALIFICATION_PATH)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    report = write_plasticity_qualification_report(args.output)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report.get("status"), "output": str(args.output)}, indent=2))
    return 0 if report.get("status") in {"passed", "diagnostic"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
