"""Run a local nonlinear capacity workflow smoke case."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.capacity_workflow import CapacityWorkflowConfig, DEFAULT_CAPACITY_WORKFLOW_PATH, run_nonlinear_capacity_workflow, write_capacity_workflow_report
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


def _beam_column_model(num_elements: int = 6) -> FEModel:
    model = FEModel("capacity_workflow_column")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    length = 4.0
    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for i in range(num_elements + 1):
        model.add_node(i + 1, length * i / num_elements, 0.0, 0.0)
    for i in range(num_elements):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", section))
    all_nodes = list(range(1, num_elements + 2))
    model.add_boundary_condition(BoundaryCondition("suppress", all_nodes, {"uz": 0.0, "rx": 0.0, "ry": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pin_left", [1], {"ux": 0.0, "uy": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pin_right", [num_elements + 1], {"uy": 0.0}))
    return model


def _compression_load(model: FEModel) -> LoadCase:
    right = max(model.mesh.nodes)
    load_case = LoadCase("unit_compression")
    load_case.add_nodal_load(right, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return load_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_CAPACITY_WORKFLOW_PATH, help="JSON workflow report output path.")
    parser.add_argument("--amplitude", type=float, default=0.005, help="Eigenmode imperfection amplitude in metres.")
    parser.add_argument("--steps", type=int, default=6, help="Nonlinear load steps.")
    parser.add_argument("--max-load-factor", type=float, default=1.0, help="Maximum nonlinear load factor for the smoke solve.")
    args = parser.parse_args()

    model = _beam_column_model()
    load_case = _compression_load(model)
    config = CapacityWorkflowConfig(
        eigenmode_imperfection_amplitude=args.amplitude,
        nonlinear_num_steps=args.steps,
        nonlinear_max_load_factor=args.max_load_factor,
        nonlinear_max_iterations=20,
        nonlinear_num_layers=3,
    )
    result = run_nonlinear_capacity_workflow(model, load_case, config=config)
    write_capacity_workflow_report(result, args.output)
    print(json.dumps({"status": result.status, "capacity_factor": result.capacity_factor, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
