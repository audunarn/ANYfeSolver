"""Write Q8, beam, and mass qualification report."""

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

from anyfesolver.element_qualification import DEFAULT_ELEMENT_QUALIFICATION_PATH, write_element_qualification_report  # noqa: E402


def _markdown(report: Dict[str, Any]) -> str:
    q8_square = report["q8"]["geometry_metrics"]["square"]
    q8r_square = report["q8r"]["geometry_metrics"]["square"]
    patch = q8_square["patch"]
    q8r_patch = q8r_square.get("patch", {})
    q8r_hourglass = q8r_square["hourglass_assessment"]
    beam = report["beam"]
    lines = [
        "# Element Qualification Report",
        "",
        f"- Generated: {report.get('generated_at')}",
        f"- Q8 square zero modes: {q8_square['free_modes']['zero_mode_count']}",
        f"- Q8 membrane patch max error: {patch['membrane_max_relative_error']:.3e}",
        f"- Q8 bending patch error: {patch['bending_relative_error']:.3e}",
        f"- Q8 shear patch error: {patch['shear_relative_error']:.3e}",
        f"- Q8R/S8R QC status: {report['q8r'].get('qc_status')}",
        f"- Q8R/S8R square zero modes: {q8r_square['free_modes']['zero_mode_count']}",
        f"- Q8R/S8R extra zero-energy modes: {q8r_hourglass['extra_zero_energy_modes']}",
        f"- Q8R/S8R membrane patch max error: {q8r_patch.get('membrane_max_relative_error', float('nan')):.3e}",
        f"- Beam max response relative error: {beam['max_relative_error']:.3e}",
        f"- Beam total mass smoke case: {beam['beam_mass']['total_mass']:.6g}",
        "",
        "## Q4/Q8 Sweep",
        "",
    ]
    for row in report["q8"]["q4_q8_convergence_cost_sweep"]:
        lines.append(
            f"- div={row['division']}: q4_dofs={row['q4_dofs']}, q8_dofs={row['q8_dofs']}, "
            f"disp_ratio={row['displacement_ratio_q4_to_q8']:.4f}, stress_ratio={row['stress_ratio_q4_to_q8']:.4f}"
        )
    lines.extend(["", "## Known Limitations", ""])
    for item in report.get("known_limitations", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ELEMENT_QUALIFICATION_PATH)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    report = write_element_qualification_report(args.output)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
