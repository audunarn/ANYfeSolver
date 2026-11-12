"""Write Batch 06 mass-property and modal-analysis verification report."""

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

from anyfesolver import (  # noqa: E402
    BoundaryCondition,
    FEModel,
    FixedSupport,
    LoadCase,
    calculate_mass_properties,
    generate_simple_panel_mesh,
    solve_free_vibration,
)
from anyfesolver.elements import BeamElement  # noqa: E402


DEFAULT_OUTPUT = Path("reports/mass_modal/mass_modal_validity_report.json")


def _axial_bar_model() -> FEModel:
    model = FEModel("modal_axial_bar")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("slider", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    return model


def generate_mass_modal_report() -> Dict[str, Any]:
    beam = _axial_bar_model()
    beam_mass = calculate_mass_properties(beam)
    modal = solve_free_vibration(beam, num_modes=1)
    expected_frequency = (100.0 / 1.0) ** 0.5 / (2.0 * 3.141592653589793)

    shell = generate_simple_panel_mesh(1.0, 1.0, 0.01, num_divisions_x=1, num_divisions_y=1)
    shell.materials["steel"].density = 7850.0
    shell_mass = calculate_mass_properties(shell)

    free = FEModel("free_beam")
    free.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    free.add_node(1, 0.0, 0.0, 0.0)
    free.add_node(2, 1.0, 0.0, 0.0)
    free.add_element(1, BeamElement(1, [1, 2], "steel", {"area": 1.0, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4}))
    free_modal = solve_free_vibration(free, num_modes=6)

    return {
        "status": "passed" if modal.solver_status == "ok" and free_modal.diagnostics["num_rigid_body_modes"] == 6 else "failed",
        "beam_mass": beam_mass.to_dict(),
        "shell_mass": shell_mass.to_dict(),
        "modal_axial_bar": {
            "frequency_hz": modal.frequencies_hz.tolist(),
            "expected_frequency_hz": expected_frequency,
            "relative_error": abs(modal.frequencies_hz[0] - expected_frequency) / expected_frequency,
            "diagnostics": modal.diagnostics,
        },
        "free_free_modal": {
            "frequencies_hz": free_modal.frequencies_hz.tolist(),
            "num_rigid_body_modes": free_modal.diagnostics["num_rigid_body_modes"],
            "diagnostics": free_modal.diagnostics,
        },
        "known_limitations": [
            "2-node beam mass remains lumped in this batch; 3-node beam and shell masses use consistent integration.",
            "Shifted sparse modal analysis exposes an explicit cached shift-invert factorization; unshifted eigsh still uses SciPy's internal operator policy.",
            "Deterministic repeated-mode basis stabilization is only sign-stabilized in this foundation batch.",
        ],
    }


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Mass And Modal Validity Report",
        "",
        f"- Status: {report['status']}",
        f"- Beam total mass: {report['beam_mass']['total_mass']:.6g}",
        f"- Shell total mass: {report['shell_mass']['total_mass']:.6g}",
        f"- Axial modal frequency: {report['modal_axial_bar']['frequency_hz'][0]:.6g} Hz",
        f"- Axial frequency relative error: {report['modal_axial_bar']['relative_error']:.3e}",
        f"- Free-free rigid modes returned: {report['free_free_modal']['num_rigid_body_modes']}",
        "",
        "## Known Limitations",
        "",
    ]
    for item in report["known_limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    report = generate_mass_modal_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {args.output}")
    if args.markdown is not None:
        print(f"Wrote {args.markdown}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
