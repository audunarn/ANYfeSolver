"""Read-only preflight for a proposed transported D1 pressure-space family.

This is a development screen, not a contact equilibrium or acceptance run.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse.linalg import lsqr

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import linearize
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from scripts.general_contact_d1_interior_press_dev import build_interior_press


def _fixture(cells: int, target_y0: float):
    model, definition, settings, program = build_interior_press(
        force_n=0.04, reference_slope=1.0e-4,
        source_support="perimeter", source_columns=cells, source_rows=cells,
        target_columns=4, target_rows=4, target_young_multiplier=5.0,
        target_y0_m=target_y0, target_reference_full_left_clamp=True,
        load_shape="physical_bump", slide_body="target",
    )
    return model, definition, replace(
        settings, q4_source_gap_certificate="overlap_local_dev"), program


def _node_values(model, u: np.ndarray, node: int) -> np.ndarray:
    return u[np.asarray(model.mesh.dof_manager.get_node_dofs(node), dtype=int)]


def _transport(seed_model, seed_u: np.ndarray, model, cells: int) -> np.ndarray:
    source_seed = np.empty((13, 13, 6))
    for row in range(13):
        for column in range(13):
            source_seed[row, column] = _node_values(
                seed_model, seed_u, 1 + row*13 + column)
    result = np.zeros(model.mesh.dof_manager.total_dofs)
    for row in range(cells+1):
        for column in range(cells+1):
            y, x = 12*row/cells, 12*column/cells
            iy, ix = min(int(y), 11), min(int(x), 11)
            fy, fx = y-iy, x-ix
            value = ((1-fy)*((1-fx)*source_seed[iy, ix]
                             + fx*source_seed[iy, ix+1])
                     + fy*((1-fx)*source_seed[iy+1, ix]
                           + fx*source_seed[iy+1, ix+1]))
            result[np.asarray(model.mesh.dof_manager.get_node_dofs(
                1 + row*(cells+1) + column), dtype=int)] = value
    seed_target_start = 170
    target_start = (cells+1)**2 + 1
    for local in range(25):
        result[np.asarray(model.mesh.dof_manager.get_node_dofs(
            target_start+local), dtype=int)] = _node_values(
                seed_model, seed_u, seed_target_start+local)
    return result


def _smooth_pose(model, cells: int) -> np.ndarray:
    """One analytic physical field evaluated on each reference grid."""
    result = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in range(1, (cells+1)**2+1):
        point = model.mesh.nodes[node]
        xi, eta = point.x-0.1, point.y-0.1
        result[model.mesh.dof_manager.get_node_dofs(node)[2]] = (
            -5.5e-5*np.sin(np.pi*xi)*np.sin(np.pi*eta))
    for node in range((cells+1)**2+1, (cells+1)**2+26):
        point = model.mesh.nodes[node]
        xi, eta = point.x/1.3, (point.y+0.113)/1.313
        result[model.mesh.dof_manager.get_node_dofs(node)[2]] = (
            5.0e-6*xi*np.sin(np.pi*eta))
    return result


def main(seed_path: Path, output: Path, *, analytic: bool = False) -> None:
    if output.exists():
        raise FileExistsError(output)
    saved = json.loads(seed_path.read_text(encoding="utf-8"))
    if (saved.get("status") != "completed"
            or saved.get("source_columns") != 12
            or saved.get("source_rows") != 12
            or saved.get("target_columns") != 4
            or saved.get("target_rows") != 4):
        raise ValueError("seed is not the expected source12/target4 result")
    seed_model, _, _, _ = _fixture(12, -0.1)
    seed_u = np.asarray(saved["displacements"], dtype=float)
    if seed_u.shape != (seed_model.mesh.dof_manager.total_dofs,):
        raise ValueError("seed displacement size changed")
    records = []
    for cells in (4, 8, 12):
        started = time.perf_counter()
        entry: dict[str, object] = {"source_cells": cells}
        try:
            model, definition, settings, program = _fixture(cells, -0.113)
            u = (_smooth_pose(model, cells) if analytic else
                 _transport(seed_model, seed_u, model, cells))
            source_nodes = (cells+1)**2
            target_start = source_nodes+1
            entry["maximum_source_displacement_m"] = float(np.max(
                np.abs(u[:6*source_nodes])))
            entry["maximum_target_displacement_m"] = float(np.max(
                np.abs(u[6*source_nodes:])))
            entry["path_controls"] = [float(sum(
                weight*u[model.mesh.dof_manager.get_node_dofs(node)[
                    {"ux": 0, "uy": 1, "uz": 2}[component]]]
                for node, component, weight in control.terms))
                for control in program.controls]
            model.apply_boundary_conditions()
            stiffness, _ = assemble_stiffness_matrix(model)
            _, _, transform, offset, _, _ = build_constraint_transformation(
                stiffness, np.zeros(len(u)), model)
            coordinates = lsqr(transform, u-offset,
                               atol=1e-14, btol=1e-14)[0]
            entry["structural_constraint_error"] = float(np.max(np.abs(
                u-(offset+transform @ coordinates))))
            if entry["structural_constraint_error"] > 1e-12:
                raise ValueError("transported pose violates structural constraints")
            manager = GeneralS3PatchManager(model, definition, settings)
            source, target = manager._positions(u)
            manager.require_d1_coverage(source, target)
            state = linearize(manager, u,
                              np.zeros(len(manager.law.pressure_mode_ids)),
                              tangent=False).state
            interior = tuple(1+row*(cells+1)+column
                             for row in range(1, cells)
                             for column in range(1, cells))
            selected = [manager.law.source_node_ids.index(node)
                        for node in interior]
            gap = np.asarray(state.normalized_gap)[selected]
            support = np.asarray(state.reference_support)[selected]
            entry.update({
                "status": "geometry_preflight_complete",
                "interior_modes": len(selected),
                "interior_negative_gap_nodes": int(np.count_nonzero(gap < 0)),
                "minimum_interior_gap_m": float(np.min(gap)),
                "minimum_interior_support_m2": float(np.min(support)),
                "reference_overlap_area_m2": float(state.reference_overlap_area),
                "current_overlap_area_m2": float(state.overlap_area),
                "source_nodes": source_nodes,
                "target_start": target_start,
            })
        except Exception as exc:
            entry.update({"status": "preflight_failed",
                          "error": f"{type(exc).__name__}: {exc}"})
        entry["elapsed_seconds"] = time.perf_counter()-started
        records.append(entry)
        print(json.dumps(entry), flush=True)
    output.write_text(json.dumps({
        "status": "development_preflight_only",
        "seed": str(seed_path), "pose_rule": (
            "smooth_analytic" if analytic else "q1_accepted_source12_transport"),
        "records": records,
    }, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] != "--analytic"):
        raise SystemExit("usage: transport_preflight SEED OUTPUT [--analytic]")
    main(Path(sys.argv[1]), Path(sys.argv[2]), analytic=len(sys.argv) == 4)
