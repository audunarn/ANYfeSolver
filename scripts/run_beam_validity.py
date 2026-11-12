"""Generate local beam/member geometric validity evidence report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.beam_validity import DEFAULT_BEAM_VALIDITY_PATH, write_beam_validity_report


def _write_markdown(report: dict, path: Path) -> None:
    rigid = report["corotational_v1"]["rigid_rotation"]
    axial = report["corotational_v1"]["axial_extension"]
    lines = [
        "# Beam Validity Report",
        "",
        f"- Generated: {report.get('generated_at')}",
        f"- Commit: {report.get('commit') or 'unknown'}",
        "",
        "## Corotational V1",
        "",
        f"- Rigid rotation angle: {rigid['angle_degrees']:.3f} deg",
        f"- Default force norm: {rigid['default_force_norm']:.6e}",
        f"- Corotational force norm: {rigid['corotational_force_norm']:.6e}",
        f"- Corot/default force ratio: {rigid['force_norm_ratio_corot_to_default']:.6e}",
        f"- Axial extension relative error: {axial['relative_error']:.6e}",
        "",
        "## Known Limitations",
        "",
    ]
    for item in report.get("known_limitations", []):
        lines.append(f"- {item}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_BEAM_VALIDITY_PATH, help="JSON report output path.")
    parser.add_argument("--markdown", type=Path, default=None, help="Optional Markdown report output path.")
    args = parser.parse_args()

    report = write_beam_validity_report(args.output)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(report, args.markdown)
    print(json.dumps({"status": "completed", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
