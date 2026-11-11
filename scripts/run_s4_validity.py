"""Generate local S4 shell validity evidence report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.s4_validity import DEFAULT_S4_VALIDITY_PATH, write_s4_validity_report


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# S4 Validity Report",
        "",
        f"- Generated: {report.get('generated_at')}",
        f"- Commit: {report.get('commit') or 'unknown'}",
        f"- Element: {report.get('element')}",
        "",
        "## Geometry Metrics",
        "",
    ]
    for name, metrics in report.get("geometry_metrics", {}).items():
        lines.append(f"### {name}")
        free = metrics.get("free_modes", {})
        lines.append(f"- Zero mode count: {free.get('zero_mode_count')}")
        if "membrane_patch" in metrics:
            errors = metrics["membrane_patch"].get("relative_errors", {})
            lines.append(f"- Membrane max relative error: {max(errors.values()) if errors else 0.0:.3e}")
        if "bending_patch" in metrics:
            lines.append(f"- Bending relative error: {metrics['bending_patch'].get('relative_error', 0.0):.3e}")
        if "shear_patch" in metrics:
            lines.append(f"- Shear relative error: {metrics['shear_patch'].get('relative_error', 0.0):.3e}")
        if "warped_quad" in metrics:
            lines.append(f"- Warped stiffness finite: {metrics['warped_quad'].get('stiffness_finite')}")
            lines.append(f"- Warped symmetry error: {metrics['warped_quad'].get('relative_symmetry_error', 0.0):.3e}")
        lines.append("")

    lines.extend(["## Thin Plate Locking Sweep", ""])
    for row in report.get("thin_plate_locking_sweep", []):
        lines.append(
            f"- L/t={row['span_to_thickness']:.0f}: ratio={row['ratio_to_reference']:.4f}, "
            f"relative_error={row['relative_error']:.3e}"
        )

    lines.extend(["", "## S4/S8 Comparison", ""])
    for row in report.get("s4_s8_comparison", []):
        recovered = row.get("recovered_stress_ratio_s4_to_s8")
        recovered_text = "" if recovered is None else f", recovered_stress_ratio={float(recovered):.4f}"
        lines.append(
            f"- div={row['division']}: disp_ratio={row['displacement_ratio_s4_to_s8']:.4f}, "
            f"stress_ratio={row['stress_ratio_s4_to_s8']:.4f}{recovered_text}"
        )
    lines.append("")
    lines.append(
        "Note: coarse-mesh S8 von Mises overshoots the converged value near hard "
        "simply-supported corners, so low S4/S8 stress ratios at coarse divisions "
        "reflect S8 overshoot more than S4 deficiency."
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_S4_VALIDITY_PATH, help="JSON report output path.")
    parser.add_argument("--markdown", type=Path, default=None, help="Optional Markdown report output path.")
    args = parser.parse_args()

    report = write_s4_validity_report(args.output)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(report, args.markdown)
    print(json.dumps({"status": "completed", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
