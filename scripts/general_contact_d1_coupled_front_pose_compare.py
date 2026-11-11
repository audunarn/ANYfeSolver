"""Read-only comparison of saved coupled Q4/S3 source poses near contact onset.

Rebuilds only the case geometry and work maps; no contact solve is rerun.
This diagnostic compares physical nodal gaps and structural displacements on
the same reference strip, then reports saved pressure and multiplier values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anyfesolver.general_contact.q4_skin_work import Q4SkinWorkAdapter
from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def _extract(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    columns = report["columns"]
    if (type(columns) is not int or columns < 2
            or report["status"] != "completed"
            or report["steps_per_segment"] not in (2, 3)
            or report.get("four_phase", False)
            or report["load_profile"] != "distributed"
            or report["total_downward_force_n"] != 0.3
            or report["target_right_x_m"] != 1.3
            or report.get("target_reference_slope", 0.0) != 0.0
            or report.get("target_columns", 1) != 1
            or report.get("pressure_basis", "fine_nodal_q1") != "fine_nodal_q1"
            or len(report["history"]) != report["steps_per_segment"]
            or report["accepted_steps"] != report["steps_per_segment"]):
        raise ValueError(f"incompatible distributed full-press report: {path}")
    model, _definition, _settings, _program = build_case(
        columns, steps_per_segment=report["steps_per_segment"],
        load_profile="distributed",
        total_force_n=0.3, target_right_x_m=1.3,
    )
    displacement = np.asarray(report["final_displacements"], dtype=float)
    if (displacement.shape != (model.mesh.dof_manager.total_dofs,)
            or not np.all(np.isfinite(displacement))):
        raise ValueError("saved displacement vector is incompatible with case")
    stride = columns + 1
    source = {}
    reference = {}
    for element_id in range(1, columns + 1):
        adapter = Q4SkinWorkAdapter.from_model(model, 1, element_id, "bottom")
        current = adapter.current_positions(displacement)
        for node, initial, point in zip(
                adapter.node_ids, adapter.reference_positions, current):
            if node in source and (np.linalg.norm(source[node]-point) > 1e-12
                                   or np.linalg.norm(reference[node]-initial) > 1e-12):
                raise ValueError("shared source skin node is discontinuous")
            source[node] = point
            reference[node] = initial
    target = S3SkinWorkAdapter.from_model(model, 2, columns+1, "top")
    target_pose = target.current_positions(displacement)
    normal = np.cross(target_pose[1]-target_pose[0],
                      target_pose[2]-target_pose[0])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.99:
        raise ValueError("unexpected target orientation")
    ids = list(range(1, 2*stride+1))
    positions = np.asarray([source[node] for node in ids]).reshape(2, stride, 3)
    initial = np.asarray([reference[node] for node in ids]).reshape(2, stride, 3)
    physical_gap = (positions-target_pose[0]) @ normal
    initial_gap = (initial-target_pose[0]) @ normal
    certificate = report["history"][-1]["contact"]
    if certificate["source_pressure_node_ids"] != ids:
        raise ValueError("saved pressure node order differs from geometry nodes")
    pressure = np.asarray(certificate["numerical_pressure_pa"], dtype=float)
    multiplier = np.asarray(certificate["normal_multiplier_estimate_pa"], dtype=float)
    weighted_gap = np.asarray(certificate["normalized_weighted_gap_m"], dtype=float)
    if any(array.shape != (len(ids),) or not np.all(np.isfinite(array))
           for array in (pressure, multiplier, weighted_gap)):
        raise ValueError("saved pressure, multiplier or gap vector is invalid")
    return {
        "path": str(path), "columns": columns,
        "steps_per_segment": report["steps_per_segment"],
        "physical_gap": physical_gap,
        "initial_gap": initial_gap,
        "vertical_displacement": positions[:, :, 2]-initial[:, :, 2],
        "horizontal_displacement": positions[:, :, 0]-initial[:, :, 0],
        "pressure": pressure.reshape(2, stride),
        "multiplier": multiplier.reshape(2, stride),
        "weighted_gap": weighted_gap.reshape(2, stride),
    }


def _squared_difference(first: dict, later: dict, name: str,
                        left: float, right: float) -> float:
    n1, n2 = first["columns"], later["columns"]
    breaks = sorted({left, right} | {
        index/n for n in (n1, n2) for index in range(n+1)
        if left < index/n < right})
    total = 0.0
    for a, b in zip(breaks, breaks[1:]):
        for x in (a+(b-a)*(1-1/np.sqrt(3))/2,
                  a+(b-a)*(1+1/np.sqrt(3))/2):
            earlier = np.asarray([
                np.interp(x, np.linspace(0, 1, n1+1), row)
                for row in first[name]
            ])
            newer = np.asarray([
                np.interp(x, np.linspace(0, 1, n2+1), row)
                for row in later[name]
            ])
            total += 0.5*(b-a)*float(np.mean((earlier-newer)**2))
    return total


def analyze(paths: list[Path]) -> dict:
    if len(paths) < 2:
        raise ValueError("provide at least two saved coupled reports")
    cases = [_extract(path) for path in paths]
    if len({case["columns"] for case in cases}) != len(cases):
        raise ValueError("source column counts must be distinct")
    rows = []
    for case in cases:
        active = case["pressure"] > 0
        first_active = [
            (int(np.flatnonzero(row)[0])/case["columns"] if np.any(row)
             else None)
            for row in active
        ]
        rows.append({
            "report": case["path"], "columns": case["columns"],
            "steps_per_segment": case["steps_per_segment"],
            "initial_nodal_gap_range_m": [
                float(np.min(case["initial_gap"])),
                float(np.max(case["initial_gap"]))],
            "full_press_nodal_gap_range_m": [
                float(np.min(case["physical_gap"])),
                float(np.max(case["physical_gap"]))],
            "first_active_reference_x_by_row": first_active,
            "peak_pressure_pa": float(np.max(case["pressure"])),
            "max_pressure_minus_multiplier_pa": float(np.max(np.abs(
                case["pressure"]-case["multiplier"]))),
            "max_active_weighted_gap_m": float(np.max(np.abs(
                case["weighted_gap"][active]))) if np.any(active) else None,
        })
    comparisons = []
    for earlier, later in zip(cases[:-1], cases[1:]):
        item = {"earlier_columns": earlier["columns"],
                "later_columns": later["columns"]}
        for name in ("physical_gap", "vertical_displacement",
                     "horizontal_displacement", "multiplier"):
            for region, left, right in (("full", 0.0, 1.0),
                                        ("front", 0.625, 0.75)):
                item[f"{name}_{region}_l2"] = float(np.sqrt(
                    _squared_difference(earlier, later, name, left, right),
                ))
        comparisons.append(item)
    return {
        "purpose": "development saved-state coupled contact-front pose comparison",
        "reference_coordinate": "source xi=(x_reference-0.1 m)/(1 m)",
        "l2_measure": "two source rows averaged, exact squared piecewise-linear integration over reference xi",
        "cases": rows,
        "adjacent_comparisons": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.reports)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
