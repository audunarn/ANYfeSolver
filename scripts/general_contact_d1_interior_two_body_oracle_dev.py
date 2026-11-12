"""Compare an accepted two-body D1 press with independent linear Q1 mortar."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from docs.d1_review_support.d1_two_body_reference_oracle import (
    independent_gap_work, solve_reference, solve_reference_singleton,
)
from scripts.general_contact_d1_interior_press_dev import build_interior_press


def main(saved_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    expected = {
        "status": "completed", "accepted_steps": 2,
        "source_columns": 4, "source_rows": 4, "target_s3_faces": 2,
        "source_support": "perimeter", "load_shape": "interior_distributed",
        "force_n": 0.04, "reference_slope": 1.0e-4,
        "slide_body": "target", "four_phase": False,
    }
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("saved nonlinear press differs from bounded oracle case")
    model, definition, _settings, program = build_interior_press(
        source_support="perimeter", source_columns=4, source_rows=4,
        target_columns=1, target_rows=1, force_n=0.04,
        reference_slope=1.0e-4, load_shape="interior_distributed",
        slide_body="target",
    )
    reference = solve_reference(
        model, definition, program, 2, physical_parallel_skin=True,
    )
    production = saved["history"][-1]["contact"]
    displacement = np.asarray(saved["displacements"], dtype=float)
    source_ids = tuple(production["source_pressure_node_ids"])
    if source_ids != reference["source_pressure_nodes"]:
        raise ValueError("oracle and solver source pressure order differ")
    pressure = np.asarray(production["numerical_pressure_pa"])
    active_production = [node for node, value in zip(source_ids, pressure)
                         if value > 0.0]
    active_reference = [node for node, value in zip(source_ids,
                                                    reference["pressure"])
                        if value > 0.0]
    force_prod = float(production["integration_force"][2])
    force_ref = float(reference["integrated_force"])
    norm_u = max(float(np.linalg.norm(reference["displacement"])), 1.0e-15)
    norm_p = max(float(np.linalg.norm(reference["pressure"])), 1.0e-15)
    normal_gap_at_production = (
        reference["initial_gap"]
        + reference["weighted_gap_work"] @ displacement/reference["support"]
    )
    result = {
        "status": "development_two_body_linear_oracle_comparison",
        "production_input": str(saved_path),
        "reference_scope": "frozen_xy_footprint_linear_two_body_q1_s3",
        "independent_initial_parallel_skin_gap_m": reference["initial_gap"],
        "independent_reference_overlap_area_m2": reference["integrated_area"],
        "oracle_active_set_iterations": reference["active_set_iterations"],
        "production_active_node_ids": active_production,
        "oracle_active_node_ids": active_reference,
        "production_integrated_normal_force_n": force_prod,
        "oracle_integrated_normal_force_n": force_ref,
        "integrated_force_relative_difference": abs(force_prod-force_ref)
                                              /max(abs(force_ref), 1.0e-15),
        "displacement_relative_l2_difference": float(np.linalg.norm(
            displacement-reference["displacement"])/norm_u),
        "pressure_relative_l2_difference": float(np.linalg.norm(
            pressure-reference["pressure"])/norm_p),
        "support_relative_l2_difference": float(np.linalg.norm(
            np.asarray(production["reference_support_m2"])
            -reference["support"])/np.linalg.norm(reference["support"])),
        "oracle_gap_at_production_minimum_m": float(np.min(
            normal_gap_at_production)),
        "production_normalized_gap_minimum_m": float(np.min(
            production["normalized_weighted_gap_m"])),
        "oracle_pressure_pa": reference["pressure"].tolist(),
        "oracle_displacements_m": reference["displacement"].tolist(),
    }
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if not key.startswith("oracle_pressure")
                      and not key.startswith("oracle_displacements")}))


def singleton_family(output: Path, inputs: list[Path]) -> None:
    if output.exists() or len(inputs) != 3:
        raise ValueError("singleton family needs new output and three saved poses")
    cases = []
    for path in inputs:
        saved = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "status": "completed", "accepted_steps": 2,
            "source_support": "perimeter", "load_shape": "interior_distributed",
            "force_n": 0.04, "reference_slope": 1.0e-4,
            "slide_body": "target", "four_phase": False,
        }
        if (any(saved.get(key) != value for key, value in expected.items())
                or saved.get("source_columns") not in (4, 8)
                or saved.get("source_rows") not in (4, 8)
                or saved.get("target_s3_faces") not in (2, 4)):
            raise ValueError(f"saved pose is outside singleton family: {path}")
        model, definition, _settings, program = build_interior_press(
            source_support="perimeter",
            source_columns=saved["source_columns"],
            source_rows=saved["source_rows"],
            target_columns=saved["target_s3_faces"]//2,
            target_rows=1, force_n=0.04, reference_slope=1.0e-4,
            load_shape="interior_distributed", slide_body="target",
        )
        reference = solve_reference_singleton(model, definition, program, 2)
        production = saved["history"][-1]["contact"]
        pressure = np.asarray(production["numerical_pressure_pa"])
        if tuple(production["source_pressure_node_ids"]) != (
                reference["source_pressure_nodes"]):
            raise ValueError("oracle pressure node order differs")
        force_production = float(production["integration_force"][2])
        force_reference = float(reference["integrated_force"])
        displacement_production = np.asarray(saved["displacements"])
        displacement_reference = reference["displacement"]
        cases.append({
            "input": str(path),
            "source_q4_faces": saved["source_columns"]*saved["source_rows"],
            "target_s3_faces": saved["target_s3_faces"],
            "oracle_active_node_id": reference["active_node_id"],
            "production_active_node_ids": [node for node, value in zip(
                production["source_pressure_node_ids"], pressure)
                if value > 0.0],
            "oracle_integrated_normal_force_n": force_reference,
            "production_integrated_normal_force_n": force_production,
            "integrated_force_relative_difference": abs(
                force_production-force_reference)/max(abs(force_reference), 1e-15),
            "displacement_relative_l2_difference": float(np.linalg.norm(
                displacement_production-displacement_reference)
                /max(np.linalg.norm(displacement_reference), 1e-15)),
            "pressure_relative_l2_difference": float(np.linalg.norm(
                pressure-reference["pressure"])
                /max(np.linalg.norm(reference["pressure"]), 1e-15)),
            "oracle_initial_gap_m": reference["initial_gap"],
            "oracle_structural_free_coordinates": reference[
                "structural_free_coordinates"],
        })
    output.write_text(json.dumps({"cases": cases}, indent=2)+"\n",
                      encoding="utf-8")
    print(json.dumps(cases))


def _source_vertical_samples(model, displacement, rows, sample_points,
                             *, columns=8):
    values_at_points = []
    for x, y in sample_points:
        local_x = (x-0.1)*columns
        local_y = (y-0.1)*rows
        column = min(int(local_x), columns-1)
        row = min(int(local_y), rows-1)
        x_fraction = local_x-column
        y_fraction = local_y-row
        values = np.array([
            displacement[model.mesh.dof_manager.get_node_dofs(
                1+j*(columns+1)+i)[2]]
            for j, i in ((row, column), (row, column+1),
                         (row+1, column), (row+1, column+1))
        ])
        weights = np.array([(1-x_fraction)*(1-y_fraction),
                            x_fraction*(1-y_fraction),
                            (1-x_fraction)*y_fraction,
                            x_fraction*y_fraction])
        values_at_points.append(float(weights @ values))
    return values_at_points


def offline_y_reference(output: Path, *, load_shape: str = "interior_distributed") -> None:
    """Screen source-y resolution with one independently clipped linear oracle.

    This keeps x resolution, target mesh, loads, support and initial opening
    fixed. The 16-row case is guarded in the fixture builder and is never an
    ordinary coupled-solve admission.
    """
    if output.exists():
        raise FileExistsError(output)
    cases = []
    # Physical source coordinates start at (0.1, 0.1), unlike local fractions.
    sample_points = ((0.6, 0.6), (0.6, 0.7), (0.6, 0.9))
    for rows in (4, 8, 16):
        model, definition, _settings, program = build_interior_press(
            source_support="perimeter", source_columns=8, source_rows=rows,
            source_y_profile="uniform_ref16" if rows == 16 else None,
            offline_structural_reference_y16=(rows == 16),
            target_columns=2, target_rows=1, force_n=0.04,
            reference_slope=1.0e-4, load_shape=load_shape,
            slide_body="target",
        )
        nodal_forces = np.asarray(program.force_values_at(2), dtype=float)
        if abs(float(np.sum(nodal_forces))+0.04) > 1.0e-12:
            raise ValueError("reference load does not balance to 0.04 N")
        centroid = [float(sum(
            -force*getattr(model.mesh.nodes[item.node_id], axis)
            for item, force in zip(program.nodal_forces, nodal_forces,
                                   strict=True))/0.04)
            for axis in ("x", "y")]
        if load_shape == "physical_bump" and np.max(
                np.abs(np.asarray(centroid)-0.6)) > 1.0e-12:
            raise ValueError("physical bump load centroid changed")
        try:
            reference = solve_reference_singleton(
                model, definition, program, 2,
                allow_multimode=(rows == 16 or load_shape == "physical_bump"))
        except ValueError as error:
            output.write_text(json.dumps({"cases": cases, "failed_rows": rows,
                                          "failure": str(error)}, indent=2)+"\n",
                              encoding="utf-8")
            raise
        displacement = reference["displacement"]
        source_values = _source_vertical_samples(
            model, displacement, rows, sample_points)
        cases.append({
            "source_columns": 8, "source_rows": rows,
            "target_s3_faces": 4,
            "load_shape": load_shape,
            "load_centroid_xy_m": centroid,
            "oracle_active_node_id": reference["active_node_id"],
            "oracle_active_node_ids": reference["active_node_ids"],
            "oracle_method": reference["reference_method"],
            "integrated_normal_force_n": reference["integrated_force"],
            "sample_xy_m": sample_points,
            "source_vertical_displacement_m": source_values,
            "structural_free_coordinates": reference[
                "structural_free_coordinates"],
        })
    output.write_text(json.dumps({"cases": cases}, indent=2)+"\n",
                      encoding="utf-8")
    print(json.dumps(cases))


def offline_x_reference(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    cases = []
    points = ((0.6, 0.6), (0.6, 0.7), (0.6, 0.9))
    for columns in (8, 16):
        model, definition, _settings, program = build_interior_press(
            source_support="perimeter", source_columns=columns,
            source_rows=16, source_y_profile="uniform_ref16",
            offline_structural_reference_y16=True,
            target_columns=2, target_rows=1, force_n=0.04,
            reference_slope=1.0e-4, load_shape="physical_bump",
            slide_body="target",
        )
        reference = solve_reference_singleton(
            model, definition, program, 2, allow_multimode=True)
        forces = np.asarray(program.force_values_at(2))
        if abs(float(np.sum(forces))+0.04) > 1.0e-12:
            raise ValueError("x-reference load does not balance")
        cases.append({
            "source_columns": columns, "source_rows": 16,
            "target_s3_faces": 4, "load_shape": "physical_bump",
            "integrated_normal_force_n": reference["integrated_force"],
            "active_node_ids": reference["active_node_ids"],
            "sample_xy_m": points,
            "source_vertical_displacement_m": _source_vertical_samples(
                model, reference["displacement"], 16, points,
                columns=columns),
            "structural_free_coordinates": reference[
                "structural_free_coordinates"],
        })
    output.write_text(json.dumps({"cases": cases}, indent=2)+"\n",
                      encoding="utf-8")
    print(json.dumps(cases))


def offline_target_reference(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    from examples.general_contact_d1_interior_controls_dev import (
        _independent_projected_gap_bound,
    )

    cases = []
    points = ((0.6, 0.6), (0.6, 0.7), (0.6, 0.9))
    for target_rows in (1, 2, 3):
        model, definition, _settings, program = build_interior_press(
            source_support="perimeter", source_columns=16,
            source_rows=16, source_y_profile="uniform_ref16",
            offline_structural_reference_y16=True,
            target_columns=2, target_rows=target_rows, force_n=0.04,
            reference_slope=1.0e-4, load_shape="physical_bump",
            slide_body="target",
            target_reference_full_left_clamp=(target_rows > 1),
        )
        target_first = 17*17+1
        middle_left = tuple(target_first+3*row
                            for row in range(1, target_rows))
        reference = solve_reference_singleton(
            model, definition, program, 2, allow_multimode=True)
        try:
            lower_gap = _independent_projected_gap_bound(
                model, definition, reference["displacement"],
                definition.interactions[0].gap_tolerance)
        except RuntimeError as error:
            output.write_text(json.dumps({
                "cases": cases, "failed_target_rows": target_rows,
                "failure": str(error)}, indent=2)+"\n", encoding="utf-8")
            raise
        cases.append({
            "source_columns": 16, "source_rows": 16,
            "target_columns": 2, "target_rows": target_rows,
            "target_s3_faces": 4*target_rows,
            "fixed_middle_left_nodes": middle_left,
            "load_shape": "physical_bump",
            "integrated_normal_force_n": reference["integrated_force"],
            "active_node_ids": reference["active_node_ids"],
            "independent_projected_lower_gap_m": lower_gap,
            "sample_xy_m": points,
            "source_vertical_displacement_m": _source_vertical_samples(
                model, reference["displacement"], 16, points,
                columns=16),
            "structural_free_coordinates": reference[
                "structural_free_coordinates"],
        })
    output.write_text(json.dumps({"cases": cases}, indent=2)+"\n",
                      encoding="utf-8")
    print(json.dumps(cases))


def _spatial_source32_identity():
    import anyfesolver
    import hashlib
    import subprocess

    root = Path(__file__).resolve().parents[1]
    module_path = Path(anyfesolver.__file__).resolve()
    if module_path != (root/"src/anyfesolver/__init__.py").resolve():
        raise ValueError("source32 reference must import the intended worktree source")
    paths = (
        "scripts/general_contact_d1_interior_two_body_oracle_dev.py",
        "scripts/general_contact_d1_interior_press_dev.py",
        "scripts/general_contact_d1_mixed_mesh_jet2_schur_press.py",
        "docs/d1_review_support/d1_two_body_reference_oracle.py",
        "examples/general_contact_d1_interior_controls_dev.py",
        "src/anyfesolver/assembly.py", "src/anyfesolver/matrix_assembly.py",
        "docs/d1_review_support/d1_bounded_single.py",
    )
    return {
        "anyfesolver_module": str(module_path),
        "python_version": sys.version, "numpy_version": np.__version__,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "helper_sha256": {path: hashlib.sha256((root/path).read_bytes()).hexdigest()
                          for path in paths},
        "inputs": {
            "source_columns": 32, "source_rows": 32,
            "source_y_profile": "uniform_ref32", "offline_structural_reference_y32": True,
            "target_columns": 6, "target_rows": 6, "target_young_multiplier": 5.0,
            "target_y0_m": -0.1, "force_n": 0.04, "reference_slope": 1.0e-4,
            "source_support": "perimeter", "load_shape": "physical_bump",
            "slide_body": "target", "target_reference_full_left_clamp": True,
            "source_reference_downshift_m": 0.0, "accepted_step": 2,
            "expected_initial_gap_m": 4.0e-5,
        },
    }


def _build_spatial_reference_case(source_cells, target_cells,
                                  target_young_multiplier, force_n,
                                  target_y0_m, source_reference_downshift_m):
    return build_interior_press(
        source_support="perimeter", source_columns=source_cells,
        source_rows=source_cells,
        source_y_profile=("uniform_ref16" if source_cells == 16 else
                          "uniform_ref24" if source_cells == 24 else
                          "uniform_ref32" if source_cells == 32 else None),
        offline_structural_reference_y16=source_cells == 16,
        offline_structural_reference_y24=source_cells == 24,
        offline_structural_reference_y32=source_cells == 32,
        target_columns=target_cells, target_rows=target_cells,
        target_young_multiplier=target_young_multiplier, target_y0_m=target_y0_m,
        source_reference_downshift_m=source_reference_downshift_m,
        force_n=force_n, reference_slope=1.0e-4, load_shape="physical_bump",
        slide_body="target", target_reference_full_left_clamp=True,
    )


def spatial_source32_allocation_inventory(output: Path) -> None:
    """Build sparse matrices and inventory future dense buffers before a solve.

    The current oracle retains a null-space SVD basis, physical variations,
    energy, work, compliance and pressure response. Phase estimates include
    simultaneous layout/symmetry copies and factorization buffers. A fixed
    512 MiB native-workspace planning allowance is added; native allocation is
    not rigorously bounded, so the actual process-tree cap remains mandatory.
    This is a resource estimate, never a scientific uncertainty estimator.
    """
    import psutil
    from anyfesolver.assembly import build_constraint_transformation
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case

    if output.exists():
        raise FileExistsError(output)
    for kwargs in (
            {"source_y_profile": "uniform_ref32"},
            {"offline_structural_reference_y32": True}):
        try:
            build_case(32, source_rows=32, **kwargs)
        except ValueError:
            pass
        else:
            raise ValueError("uniform32 must require both an offline opt-in and explicit profile")
    model, definition, settings, program = _build_spatial_reference_case(
        32, 6, 5.0, 0.04, -0.1, 0.0)
    if settings.maximum_general_s3_energy_evaluations != 14144:
        raise ValueError("production contact energy cap changed")
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    d = model.mesh.dof_manager.total_dofs
    _, _, transform, offset, _, _ = build_constraint_transformation(
        stiffness, np.zeros(d), model)
    if np.linalg.norm(offset) != 0.0:
        raise ValueError("inventory requires homogeneous supports")
    rows = np.zeros((2, d))
    for row, control in enumerate(program.controls):
        for node, name, weight in control.terms:
            if name != "ux":
                raise ValueError("inventory requires target-x controls")
            rows[row, model.mesh.dof_manager.get_node_dofs(node)[0]] += weight
    rank = int(np.linalg.matrix_rank(np.asarray(rows @ transform)))
    if rank != 2 or np.max(np.abs(program.target_at(2))) != 0.0:
        raise ValueError("inventory requires two independent zero controls")
    q = int(transform.shape[1])
    n = q-rank
    p = len({node for element in definition.surfaces[0].element_ids
             for node in model.mesh.elements[element].node_ids})

    def allocation(shape, copies=1):
        return {"shape": list(shape), "copies": copies,
                "bytes": int(np.prod(shape))*8*copies}

    buffers = {
        "full_work": allocation((p, d)),
        "null_space_vh_and_returned_basis": allocation((q, q), 2),
        "physical_variations": allocation((d, n)), "energy": allocation((n, n)),
        "weighted_work": allocation((p, n)), "compliance": allocation((n, p)),
        "gap_response_and_symmetry_copy": allocation((p, p), 2),
        "solution_vectors": allocation((d+q+6*n+10*p,)),
    }
    retained = sum(item["bytes"] for item in buffers.values())
    scratch = {
        "energy_assembly_symmetry": {
            "sparse_product_and_layout_copy": allocation((d, n), 2),
            "old_energy_sum_and_final_copy": allocation((n, n), 2)},
        "energy_factorization_compliance": {
            "layout_finite_check_factor_and_work_copy": allocation((n, n), 3),
            "rhs_and_result_copy": allocation((p, n), 2)},
        "contact_cholesky_nnls": {
            "eligible_response_factor_nnls_copies": allocation(((32-1)**2,)*2, 6)},
    }
    base = int(psutil.Process().memory_info().rss)
    native_allowance = 512*1024**2
    phases = {name: base+native_allowance+retained
              +sum(item["bytes"] for item in copies.values())
              for name, copies in scratch.items()}
    predicted = max(phases.values())
    record = {
        "status": "allocation_inventory_fits" if predicted < 4_000_000_000
                  else "allocation_inventory_exceeds_cap",
        "question": "nested source8->16->32 contraction versus mesh/contact-front cancellation",
        "identity": _spatial_source32_identity(),
        "dimensions": {"full_dofs": d, "uncontrolled_free_coordinates": q,
                       "control_rank": rank, "structural_free_coordinates": n,
                       "source_pressure_nodes": p},
        "dense_buffers": buffers, "phase_scratch_buffers": scratch,
        "phase_predicted_bytes": phases, "measured_inventory_base_rss_bytes": base,
        "native_workspace_allowance_bytes": native_allowance,
        "predicted_peak_bytes": predicted, "run_memory_limit_bytes": 4_000_000_000,
        "production_energy_evaluation_cap": settings.maximum_general_s3_energy_evaluations,
        "physical_load_sum_n": float(sum(program.force_values_at(2))),
        "limitations": "allocation planning; actual hard process-tree RSS cap remains required",
    }
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in
                      ("status", "dimensions", "predicted_peak_bytes",
                       "production_energy_evaluation_cap", "physical_load_sum_n")}))
    if predicted >= 4_000_000_000:
        raise ValueError("source32 reference allocation inventory exceeds cap")


def offline_target_xy_reference(output: Path, baseline_path: Path | None,
                                target_cells_to_run: tuple[int, ...] = (3, 4),
                                target_young_multiplier: float = 1.0,
                                source_cells: int = 16,
                                force_n: float = 0.04,
                                target_y0_m: float = -0.1,
                                source_reference_downshift_m: float = 0.0,
                                allocation_inventory_path: Path | None = None,
                                ) -> None:
    """Screen x/y S3 refinement under the same fixed physical source load."""
    if output.exists():
        raise FileExistsError(output)
    from examples.general_contact_d1_interior_controls_dev import (
        _independent_projected_gap_bound,
    )

    if target_young_multiplier not in (1.0, 5.0):
        raise ValueError("unsupported target contrast")
    if force_n not in (0.001, 0.04, 0.08):
        raise ValueError("unsupported bounded force screen")
    if source_reference_downshift_m not in (0.0, 35.0e-6):
        raise ValueError("unsupported source reference downshift")
    if force_n == 0.001 and not (
            source_reference_downshift_m == 35.0e-6
            and target_young_multiplier == 5.0 and source_cells == 4
            and target_cells_to_run == (4,) and target_y0_m == -0.1
            and baseline_path is None):
        raise ValueError("low-load oracle is limited to the 5 µm opening fixture")
    if force_n != 0.001 and source_reference_downshift_m != 0.0:
        raise ValueError("downshift is limited to the low-load fixture")
    if target_y0_m not in (-0.1, -0.113):
        raise ValueError("unsupported bounded target row offset")
    if source_cells not in (4, 8, 12, 16, 24, 32):
        raise ValueError("unsupported source contrast screen")
    identity = None
    if source_cells == 32:
        if (target_cells_to_run != (6,) or target_young_multiplier != 5.0
                or force_n != 0.04 or target_y0_m != -0.1
                or source_reference_downshift_m != 0.0 or baseline_path is not None
                or allocation_inventory_path is None):
            raise ValueError("source32 is limited to the inventoried offline target6 reference")
        inventory = json.loads(allocation_inventory_path.read_text(encoding="utf-8"))
        identity = _spatial_source32_identity()
        if (inventory["status"] != "allocation_inventory_fits"
                or inventory["predicted_peak_bytes"] >= 4_000_000_000
                or inventory["identity"] != identity):
            raise ValueError("source32 allocation inventory is failed or stale")
    if baseline_path is None:
        cases = []
    else:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        cases = [next(item for item in baseline["cases"]
                      if item["target_columns"] == item["target_rows"] == 2)]
        if (cases[0]["source_columns"] != source_cells
                or cases[0]["source_rows"] != source_cells
                or cases[0]["load_shape"] != "physical_bump"
                or target_young_multiplier != 1.0):
            raise ValueError("baseline target reference has incompatible mechanics")
    points = ((0.6, 0.6), (0.6, 0.7), (0.6, 0.9))
    for target_cells in target_cells_to_run:
        if target_cells not in (3, 4, 6):
            raise ValueError("unsupported offline target-XY screen")
        model, definition, _settings, program = _build_spatial_reference_case(
            source_cells, target_cells, target_young_multiplier, force_n,
            target_y0_m, source_reference_downshift_m)
        reference = solve_reference_singleton(
            model, definition, program, 2, allow_multimode=True,
            expected_initial_gap_m=(5.0e-6 if source_reference_downshift_m
                                    else 4.0e-5))
        lower_gap = _independent_projected_gap_bound(
            model, definition, reference["displacement"],
            definition.interactions[0].gap_tolerance)
        target_first = (source_cells+1)**2+1
        cases.append({
            "source_columns": source_cells, "source_rows": source_cells,
            "target_columns": target_cells, "target_rows": target_cells,
            "target_s3_faces": 2*target_cells**2,
            "target_young_multiplier": target_young_multiplier,
            "target_y0_m": target_y0_m,
            "force_n": force_n,
            "source_reference_downshift_m": source_reference_downshift_m,
            "fixed_middle_left_nodes": tuple(
                target_first+(target_cells+1)*row
                for row in range(1, target_cells)),
            "load_shape": "physical_bump",
            "integrated_normal_force_n": reference["integrated_force"],
            "active_node_ids": reference["active_node_ids"],
            "independent_projected_lower_gap_m": lower_gap,
            "sample_xy_m": points,
            "source_vertical_displacement_m": _source_vertical_samples(
                model, reference["displacement"], source_cells, points,
                columns=source_cells),
            "structural_free_coordinates": reference[
                "structural_free_coordinates"],
            "target_maximum_vertical_displacement_m": max(abs(float(
                reference["displacement"][model.mesh.dof_manager.get_node_dofs(
                    node)[2]])) for node in range(
                        target_first, target_first+(target_cells+1)**2)),
        })
    payload = {"baseline": str(baseline_path) if baseline_path is not None else None,
               "cases": cases}
    if identity is not None:
        payload.update({"identity": identity,
                        "allocation_inventory": str(allocation_inventory_path),
                        "reference_scope": "independent_frozen_footprint_linear_q1_s3_mortar_kkt",
                        "status": "development_reference_not_spatial_acceptance"})
    output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps([{key: item[key] for key in (
        "target_columns", "target_rows", "integrated_normal_force_n",
        "independent_projected_lower_gap_m")}
        for item in cases]))


def target8s3_coupled_oracle(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    from examples.general_contact_d1_interior_controls_dev import (
        _independent_projected_gap_bound,
    )

    model, definition, _settings, program = build_interior_press(
        source_support="perimeter", source_columns=8, source_rows=8,
        target_columns=2, target_rows=2,
        target_reference_full_left_clamp=True,
        force_n=0.04, reference_slope=1.0e-4,
        load_shape="physical_bump", slide_body="target",
    )
    reference = solve_reference_singleton(
        model, definition, program, 2, allow_multimode=True)
    gap = _independent_projected_gap_bound(
        model, definition, reference["displacement"],
        definition.interactions[0].gap_tolerance)
    record = {
        "source_q4_faces": 64, "target_s3_faces": 8,
        "load_shape": "physical_bump",
        "fixed_middle_left_node_id": 1+9*9+3,
        "integrated_normal_force_n": reference["integrated_force"],
        "active_node_ids": reference["active_node_ids"],
        "independent_projected_lower_gap_m": gap,
        "sample_xy_m": ((0.6, 0.6), (0.6, 0.7), (0.6, 0.9)),
        "source_vertical_displacement_m": _source_vertical_samples(
            model, reference["displacement"], 8,
            ((0.6, 0.6), (0.6, 0.7), (0.6, 0.9))),
    }
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record))


def target8s3_coupled_comparison(output: Path, reference_path: Path,
                                saved_path: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    from examples.general_contact_d1_interior_controls_dev import (
        _independent_projected_gap_bound,
    )

    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (saved["status"] != "completed" or saved["accepted_steps"] != 2
            or saved["source_columns"] != 8 or saved["source_rows"] != 8
            or saved["target_s3_faces"] != 8
            or saved["target_reference_full_left_clamp"] is not True
            or saved["load_shape"] != "physical_bump"
            or saved["q4_gap_certificate"] != "overlap_local_dev"):
        raise ValueError("saved pose is outside selected target8s3 comparison")
    model, definition, _settings, _program = build_interior_press(
        source_support="perimeter", source_columns=8, source_rows=8,
        target_columns=2, target_rows=2,
        target_reference_full_left_clamp=True,
        force_n=0.04, reference_slope=1.0e-4,
        load_shape="physical_bump", slide_body="target",
    )
    displacement = np.asarray(saved["displacements"])
    contact = saved["history"][-1]["contact"]
    pressure = np.asarray(contact["numerical_pressure_pa"])
    force = float(contact["integration_force"][2])
    reference_force = reference["integrated_normal_force_n"]
    samples = _source_vertical_samples(
        model, displacement, 8, reference["sample_xy_m"])
    independent_lower = _independent_projected_gap_bound(
        model, definition, displacement,
        definition.interactions[0].gap_tolerance)
    record = {
        "saved_pose": str(saved_path), "oracle": str(reference_path),
        "solver_status": saved["status"],
        "accepted_steps": saved["accepted_steps"],
        "certificate": saved["q4_gap_certificate"],
        "production_integrated_normal_force_n": force,
        "oracle_integrated_normal_force_n": reference_force,
        "relative_integrated_force_difference": abs(
            force-reference_force)/abs(reference_force),
        "production_active_node_ids": [node for node, value in zip(
            contact["source_pressure_node_ids"], pressure) if value > 0.0],
        "oracle_active_node_ids": reference["active_node_ids"],
        "relative_source_displacement_difference": [
            abs(a-b)/max(abs(b), 1.0e-15) for a, b in zip(
                samples, reference["source_vertical_displacement_m"])],
        "independent_projected_lower_gap_m": independent_lower,
        "registered_gap_tolerance_m": (
            definition.interactions[0].gap_tolerance),
        "production_contact_force_balance_n": [
            a+b for a, b in zip(contact["integration_force"],
                                contact["target_force"])],
        "production_contact_moment_balance_nm": [
            a+b for a, b in zip(contact["integration_moment"],
                                contact["target_moment"])],
        "active_rim_pressure_pa": saved["history"][-1]["rim_pressure_pa"],
    }
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record))


def coupled_contrast_comparison(output: Path, reference_path: Path,
                                saved_path: Path) -> None:
    """Compare one bounded nonlinear contrast pose with its linear Q1 oracle."""
    if output.exists():
        raise FileExistsError(output)
    from examples.general_contact_d1_interior_controls_dev import (
        _independent_projected_gap_bound,
    )

    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (saved["status"] != "completed" or saved["accepted_steps"] != 2
            or saved["target_young_multiplier"] != 5.0
            or saved["source_support"] != "perimeter"
            or saved["load_shape"] != "physical_bump"
            or saved["target_reference_full_left_clamp"] is not True
            or saved["q4_gap_certificate"] != "overlap_local_dev"):
        raise ValueError("saved pose is outside the declared contrast case")
    matching = [case for case in reference["cases"]
                if case["source_columns"] == saved["source_columns"]
                and case["source_rows"] == saved["source_rows"]
                and case["target_columns"] == saved["target_columns"]
                and case["target_rows"] == saved["target_rows"]
                and case["target_young_multiplier"] == 5.0
                and case.get("target_y0_m", -0.1) == saved.get("target_y0_m", -0.1)
                and case.get("force_n", 0.04) == saved["force_n"]
                and case.get("source_reference_downshift_m", 0.0) ==
                saved.get("source_reference_downshift_m", 0.0)]
    if len(matching) != 1:
        raise ValueError("one matching independent reference is required")
    oracle = matching[0]
    model, definition, _settings, _program = build_interior_press(
        source_support="perimeter",
        source_columns=saved["source_columns"],
        source_rows=saved["source_rows"],
        source_y_profile=saved["source_y_profile"],
        target_columns=saved["target_columns"],
        target_rows=saved["target_rows"],
        target_young_multiplier=5.0,
        target_y0_m=saved.get("target_y0_m", -0.1),
        source_reference_downshift_m=saved.get(
            "source_reference_downshift_m", 0.0),
        target_reference_full_left_clamp=True,
        force_n=saved["force_n"],
        reference_slope=saved["reference_slope"],
        load_shape="physical_bump", slide_body="target",
    )
    displacement = np.asarray(saved["displacements"], dtype=float)
    contact = saved["history"][-1]["contact"]
    pressure = np.asarray(contact["numerical_pressure_pa"], dtype=float)
    force = float(contact["integration_force"][2])
    reference_force = float(oracle["integrated_normal_force_n"])
    samples = _source_vertical_samples(
        model, displacement, saved["source_rows"], oracle["sample_xy_m"],
        columns=saved["source_columns"])
    independent_lower = _independent_projected_gap_bound(
        model, definition, displacement,
        definition.interactions[0].gap_tolerance)
    record = {
        "saved_pose": str(saved_path), "oracle": str(reference_path),
        "oracle_scope": "same_mesh_small_motion_linear_q1_mortar",
        "solver_status": saved["status"],
        "accepted_steps": saved["accepted_steps"],
        "certificate": saved["q4_gap_certificate"],
        "production_integrated_normal_force_n": force,
        "oracle_integrated_normal_force_n": reference_force,
        "relative_integrated_force_difference": abs(
            force-reference_force)/abs(reference_force),
        "production_active_node_ids": [node for node, value in zip(
            contact["source_pressure_node_ids"], pressure) if value > 0.0],
        "oracle_active_node_ids": oracle["active_node_ids"],
        "production_source_vertical_displacement_m": samples,
        "oracle_source_vertical_displacement_m": (
            oracle["source_vertical_displacement_m"]),
        "relative_source_displacement_difference": [
            abs(a-b)/max(abs(b), 1.0e-15) for a, b in zip(
                samples, oracle["source_vertical_displacement_m"])],
        "independent_projected_lower_gap_m": independent_lower,
        "registered_gap_tolerance_m": (
            definition.interactions[0].gap_tolerance),
        "production_contact_force_balance_n": [
            a+b for a, b in zip(contact["integration_force"],
                                contact["target_force"])],
        "production_contact_moment_balance_nm": [
            a+b for a, b in zip(contact["integration_moment"],
                                contact["target_moment"])],
        "active_rim_pressure_pa": saved["history"][-1]["rim_pressure_pa"],
    }
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in (
        "relative_integrated_force_difference",
        "relative_source_displacement_difference",
        "independent_projected_lower_gap_m",
        "production_active_node_ids", "oracle_active_node_ids",
        "production_contact_force_balance_n",
        "production_contact_moment_balance_nm",
        "active_rim_pressure_pa")}))


def lowload_geometry_decomposition(output: Path, saved_path: Path) -> None:
    """Compare moving contact geometry with the independent frozen map."""
    if output.exists():
        raise FileExistsError(output)
    from scipy import sparse
    from anyfesolver.assembly import build_constraint_transformation
    from anyfesolver.general_contact.core import suspend_contact
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.semismooth import linearize
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system

    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    if (saved["status"] != "completed" or saved["accepted_steps"] != 2
            or saved["force_n"] != 0.001
            or saved["source_reference_downshift_m"] != 35.0e-6
            or saved["source_columns"] != 4 or saved["source_rows"] != 4
            or saved["target_columns"] != 4 or saved["target_rows"] != 4
            or saved["q4_gap_certificate"] != "overlap_local_dev"):
        raise ValueError("saved pose is outside the low-load discriminator")
    model, definition, settings, _program = build_interior_press(
        force_n=0.001, reference_slope=1.0e-4,
        source_support="perimeter", source_columns=4, source_rows=4,
        target_columns=4, target_rows=4, target_young_multiplier=5.0,
        target_reference_full_left_clamp=True,
        source_reference_downshift_m=35.0e-6,
        load_shape="physical_bump", slide_body="target",
    )
    manager = GeneralS3PatchManager(
        model, definition,
        replace(settings, q4_source_gap_certificate="overlap_local_dev"),
    )
    contact = saved["history"][-1]["contact"]
    pressure = np.asarray(contact["numerical_pressure_pa"], dtype=float)
    support_saved = np.asarray(contact["reference_support_m2"], dtype=float)
    displacement = np.asarray(saved["displacements"], dtype=float)
    state = linearize(manager, displacement, pressure*support_saved,
                      tangent=False)
    node_ids, support_reference, reference_work, reference_area = (
        independent_gap_work(model, definition, 0.0))
    if (tuple(node_ids) != tuple(manager.law.source_node_ids)
            or tuple(contact["source_pressure_node_ids"]) != tuple(node_ids)):
        raise ValueError("production and oracle pressure identities differ")
    support_production = np.asarray(state.state.reference_support)
    np.testing.assert_allclose(support_production, support_saved,
                               rtol=1e-9, atol=1e-12)
    reference_work = (reference_work.toarray() if sparse.issparse(reference_work)
                      else np.asarray(reference_work))
    gradient = (state.gap_gradient.toarray()
                if sparse.issparse(state.gap_gradient)
                else np.asarray(state.gap_gradient))
    production_work = support_production[:, None]*gradient
    reference_gap = 5.0e-6 + reference_work @ displacement/support_reference
    production_gap = np.asarray(state.state.normalized_gap)
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    _, _, transform, _offset, _, _ = build_constraint_transformation(
        stiffness, np.zeros(len(displacement)), model,
    )
    with suspend_contact():
        nonlinear_internal, _, _ = _assemble_nonlinear_system(
            model, displacement, {}, 5, tangent=False,
            kinematics="von_karman", require_full_coordinates=True,
        )
    structural_difference = np.asarray(nonlinear_internal-stiffness @ displacement)
    reduced_difference = np.asarray(transform.T @ structural_difference).reshape(-1)
    active = np.flatnonzero(pressure > 0.0)
    if active.size != 1 or node_ids[int(active[0])] != 13:
        raise ValueError("saved low-load active pressure identity changed")
    i = int(active[0])
    record = {
        "status": "development_lowload_geometry_decomposition",
        "saved_pose": str(saved_path),
        "oracle_scope": "frozen_reference_xy_work_map",
        "active_node_id": node_ids[i],
        "production_gap_m": float(production_gap[i]),
        "frozen_oracle_gap_at_production_pose_m": float(reference_gap[i]),
        "active_gap_difference_m": float(production_gap[i]-reference_gap[i]),
        "active_support_relative_difference": float(
            (support_production[i]-support_reference[i])/
            support_reference[i]),
        "active_work_row_relative_l2_difference": float(np.linalg.norm(
            production_work[i]-reference_work[i])/
            np.linalg.norm(reference_work[i])),
        "active_work_on_production_pose_difference_m3": float(
            (production_work[i]-reference_work[i]) @ displacement),
        "structural_nonlinear_minus_linear_free_l2_n": float(
            np.linalg.norm(reduced_difference)),
        "structural_nonlinear_minus_linear_active_uz_n": float(
            structural_difference[model.mesh.dof_manager.get_node_dofs(13)[2]]),
        "independent_reference_overlap_area_m2": float(reference_area),
    }
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record))


def lowload_current_skin_check(output: Path, saved_path: Path) -> None:
    """Check moving contact values/work against independent current-skin integration."""
    if output.exists():
        raise FileExistsError(output)
    from dataclasses import replace
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.semismooth import linearize
    from docs.d1_review_support.d1_two_body_current_skin_quadrature import (
        integrate_current_skin,
    )

    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    if (saved["status"] != "completed" or saved["accepted_steps"] != 2
            or saved["force_n"] != 0.001
            or saved["source_reference_downshift_m"] != 35.0e-6
            or saved["source_columns"] != 4 or saved["source_rows"] != 4
            or saved["target_columns"] != 4 or saved["target_rows"] != 4
            or saved["q4_gap_certificate"] != "overlap_local_dev"):
        raise ValueError("saved pose is outside the low-load discriminator")
    model, definition, settings, _program = build_interior_press(
        force_n=0.001, reference_slope=1.0e-4,
        source_support="perimeter", source_columns=4, source_rows=4,
        target_columns=4, target_rows=4, target_young_multiplier=5.0,
        target_reference_full_left_clamp=True,
        source_reference_downshift_m=35.0e-6,
        load_shape="physical_bump", slide_body="target",
    )
    manager = GeneralS3PatchManager(
        model, definition,
        replace(settings, q4_source_gap_certificate="overlap_local_dev"),
    )
    contact = saved["history"][-1]["contact"]
    pose = np.asarray(saved["displacements"], dtype=float)
    support_saved = np.asarray(contact["reference_support_m2"], dtype=float)
    pressure = np.asarray(contact["numerical_pressure_pa"], dtype=float)
    state = linearize(manager, pose, pressure*support_saved, tangent=False)
    independent_support, weighted_gap, _area, count, signature = (
        integrate_current_skin(model, definition, pose))
    if tuple(contact["source_pressure_node_ids"]) != tuple(manager.law.source_node_ids):
        raise ValueError("pressure node ordering changed")
    active = np.flatnonzero(pressure > 0.0)
    if len(active) != 1 or manager.law.source_node_ids[int(active[0])] != 13:
        raise ValueError("active node changed")
    row = int(active[0])
    production_gradient = (state.gap_gradient.toarray()
                           if hasattr(state.gap_gradient, "toarray")
                           else np.asarray(state.gap_gradient))
    target_ids = sorted({node for element_id in definition.surfaces[1].element_ids
                         for node in model.mesh.elements[element_id].node_ids})
    dofs = model.mesh.dof_manager
    checks = [("source13_ux", int(dofs.get_node_dofs(13)[0]), 1e-7),
              ("source13_uz", int(dofs.get_node_dofs(13)[2]), 1e-7)]
    checks.extend(("source13_ry", int(dofs.get_node_dofs(13)[4]), step)
                  for step in (1e-6, 1e-5, 1e-4))
    for axis, name in ((0, "ux"), (2, "uz"), (4, "ry")):
        target_dof = max((int(dofs.get_node_dofs(node)[axis])
                          for node in target_ids),
                         key=lambda dof: abs(production_gradient[row, dof]))
        checks.extend((f"target_{target_dof}_{name}", target_dof, step)
                      for step in ((1e-6, 1e-5, 1e-4) if axis == 4
                                   else (1e-7,)))
    derivatives = []
    for label, dof, step in checks:
        plus = pose.copy()
        minus = pose.copy()
        plus[dof] += step
        minus[dof] -= step
        sp, wp, _area_p, _count_p, signature_p = integrate_current_skin(
            model, definition, plus)
        sm, wm, _area_m, _count_m, signature_m = integrate_current_skin(
            model, definition, minus)
        if signature_p != signature or signature_m != signature:
            raise ValueError(f"clip topology changed for {label}")
        independent = float(((wp/sp-wm/sm)/(2*step))[row])
        production = float(production_gradient[row, dof])
        derivatives.append({
            "direction": label, "dof": dof, "step": step,
            "independent_gap_derivative": independent,
            "production_gap_derivative": production,
            "relative_difference": abs(independent-production)
                /max(abs(independent), abs(production), 1e-8),
        })
    record = {
        "status": "development_lowload_independent_current_skin_check",
        "saved_pose": str(saved_path),
        "quadrature_points": count,
        "active_node_id": 13,
        "max_support_relative_difference": float(np.max(np.abs(
            independent_support-np.asarray(state.state.reference_support))
            /independent_support)),
        "active_gap_difference_m": float(
            weighted_gap[row]/independent_support[row]
            -np.asarray(state.state.normalized_gap)[row]),
        "derivatives": derivatives,
    }
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record))


def interior_saved_kkt_check(output: Path, saved_path: Path) -> None:
    """Independently integrate an interior active contact row on all free DOFs."""
    if output.exists():
        raise FileExistsError(output)
    import time
    from anyfesolver.assembly import build_constraint_transformation
    from anyfesolver.general_contact.core import suspend_contact
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.semismooth import linearize
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    from docs.d1_review_support.d1_two_body_current_skin_quadrature import (
        integrate_current_skin,
    )

    started = time.perf_counter()
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    if (saved["status"] != "completed" or saved["accepted_steps"] != 1
            or saved["force_n"] != 0.04 or saved["source_columns"] != 4
            or saved["source_rows"] != 4 or saved["target_columns"] != 1
            or saved["target_rows"] != 1
            or saved["load_shape"] != "interior_distributed"
            or saved["q4_gap_certificate"] != "overlap_local_dev"):
        raise ValueError("saved pose is outside the D1 interior KKT fixture")
    model, definition, settings, program = build_interior_press(
        force_n=0.04, reference_slope=1.0e-4,
        source_support="perimeter", source_columns=4, source_rows=4,
        target_columns=1, target_rows=1,
        load_shape="interior_distributed", slide_body="target",
        steps_per_segment=1,
    )
    manager = GeneralS3PatchManager(
        model, definition,
        replace(settings, q4_source_gap_certificate="overlap_local_dev"),
    )
    model.apply_boundary_conditions()
    ndof = model.mesh.dof_manager.total_dofs
    stiffness, _ = assemble_stiffness_matrix(model)
    _, _, transform, offset, _, _ = build_constraint_transformation(
        stiffness, np.zeros(ndof), model)
    if np.linalg.norm(offset) != 0.0:
        raise ValueError("KKT fixture requires zero model boundary values")
    transform = transform.tocsr()
    control_values = np.asarray(program.target_at(1), dtype=float)
    if len(control_values) != len(program.controls) or np.any(control_values):
        raise ValueError("KKT fixture requires zero-valued path controls")
    controlled = []
    for control in program.controls:
        if len(control.terms) != 1 or control.terms[0][1:] != ("ux", 1.0):
            raise ValueError("KKT fixture needs simple nodal ux controls")
        dof = int(model.mesh.dof_manager.get_node_dofs(
            control.terms[0][0])[0])
        row = transform.getrow(dof)
        if row.nnz != 1 or row.data[0] != 1.0:
            raise ValueError("control is not one retained coordinate")
        controlled.append(int(row.indices[0]))
    free = np.setdiff1d(np.arange(transform.shape[1]), controlled)
    columns = np.asarray(transform[:, free].toarray())
    if np.any(np.sum(np.abs(columns) > 0.0, axis=0) != 1):
        raise ValueError("KKT fixture requires one physical DOF per free column")
    pose = np.asarray(saved["displacements"], dtype=float)
    if pose.shape != (ndof,) or np.max(abs(columns @ (columns.T @ pose)-pose)) > 1e-12:
        raise ValueError("saved pose is inconsistent with fixture constraints")
    contact = saved["history"][-1]["contact"]
    support_saved = np.asarray(contact["reference_support_m2"], dtype=float)
    pressure = np.asarray(contact["numerical_pressure_pa"], dtype=float)
    active = np.flatnonzero(pressure > 0.0)
    if (len(active) != 1 or manager.law.source_node_ids[int(active[0])]
            not in saved["interior_pressure_node_ids"]):
        raise ValueError("saved active mode is not one interior source node")
    row = int(active[0])
    force = float(pressure[row]*support_saved[row])
    independent_support, weighted, _area, points, signature = (
        integrate_current_skin(model, definition, pose))
    independent_gap = weighted/independent_support
    active_source_faces = {element_id
        for element_id in definition.surfaces[0].element_ids
        if manager.law.source_node_ids[row]
        in model.mesh.elements[element_id].node_ids}
    active_signature = tuple(item for item in signature
                             if item[0] in active_source_faces)
    gradient = np.empty(len(free))
    for index in range(len(free)):
        full_dof = int(np.flatnonzero(columns[:, index])[0])
        step = 1e-5 if full_dof % 6 >= 3 else 1e-7
        plus = pose+step*columns[:, index]
        minus = pose-step*columns[:, index]
        support_plus, weighted_plus, _, _, signature_plus = (
            integrate_current_skin(model, definition, plus))
        support_minus, weighted_minus, _, _, signature_minus = (
            integrate_current_skin(model, definition, minus))
        if (tuple(item for item in signature_plus
                  if item[0] in active_source_faces) != active_signature
                or tuple(item for item in signature_minus
                         if item[0] in active_source_faces) != active_signature):
            raise ValueError(
                f"active-support clipping changed at free column {index}")
        gradient[index] = (
            (weighted_plus[row]/support_plus[row]
             -weighted_minus[row]/support_minus[row])/(2*step))
    with suspend_contact():
        internal, _, _ = _assemble_nonlinear_system(
            model, pose, {}, 5, tangent=False,
            kinematics="von_karman", require_full_coordinates=True)
    external = np.zeros(ndof)
    for load, value in zip(program.nodal_forces,
                           program.force_values_at(1), strict=True):
        if load.dof != "uz":
            raise ValueError("KKT fixture only admits vertical dead forces")
        external[model.mesh.dof_manager.get_node_dofs(load.node_id)[2]] += value
    structural = columns.T @ (np.asarray(internal)-external)
    independent_residual = structural-gradient*force
    seed = solve_reference_singleton(
        model, definition, program, 1, allow_multimode=True)
    seed_pose = np.asarray(seed["displacement"])
    seed_pressure = np.asarray(seed["pressure"])
    production_state = linearize(manager, pose, pressure*support_saved,
                                 tangent=False)
    production_gradient = (production_state.gap_gradient.toarray()
                           if hasattr(production_state.gap_gradient, "toarray")
                           else np.asarray(production_state.gap_gradient))
    production_reduced_gradient = production_gradient[row] @ columns
    record = {
        "status": "development_d1_interior_saved_pose_independent_kkt",
        "saved_pose": str(saved_path),
        "active_node_id": manager.law.source_node_ids[row],
        "active_force_n": force,
        "free_coordinates": len(free),
        "independent_geometry_evaluations": 1+2*len(free),
        "quadrature_points_per_evaluation": points,
        "maximum_support_relative_difference": float(np.max(np.abs(
            independent_support-np.asarray(production_state.state.reference_support))
            /independent_support)),
        "active_gap_difference_m": float(independent_gap[row]
            -np.asarray(production_state.state.normalized_gap)[row]),
        "active_gradient_relative_l2_difference": float(np.linalg.norm(
            gradient-production_reduced_gradient)
            /np.linalg.norm(production_reduced_gradient)),
        "independent_free_equilibrium_residual_n": float(np.linalg.norm(
            independent_residual)),
        "independent_free_equilibrium_residual_over_force_scale": float(
            np.linalg.norm(independent_residual)/0.04),
        "minimum_inactive_normalized_gap_m": float(np.min(np.delete(
            independent_gap, active))),
        "linear_seed_active_node_ids": [manager.law.source_node_ids[i]
            for i in np.flatnonzero(seed_pressure > 0.0)],
        "linear_seed_relative_displacement_l2_difference": float(
            np.linalg.norm(seed_pose-pose)/np.linalg.norm(pose)),
        "linear_seed_relative_integrated_force_difference": float(
            abs(float(seed["integrated_force"])-force)/force),
        "elapsed_seconds": time.perf_counter()-started,
    }
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record))


def interior_nonlinear_kkt_compare(output: Path, saved_path: Path) -> None:
    """Solve the interior fixed active branch with independent current contact."""
    if output.exists():
        raise FileExistsError(output)
    import time
    from scipy import sparse
    from anyfesolver.assembly import build_constraint_transformation
    from anyfesolver.general_contact.core import suspend_contact
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    from docs.d1_review_support.d1_two_body_current_skin_quadrature import (
        integrate_current_skin,
    )

    started = time.perf_counter()
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    if (saved["status"] != "completed" or saved["accepted_steps"] != 1
            or saved["force_n"] != 0.04 or saved["source_columns"] != 4
            or saved["source_rows"] != 4 or saved["target_columns"] != 1
            or saved["target_rows"] != 1
            or saved["load_shape"] != "interior_distributed"
            or saved["q4_gap_certificate"] != "overlap_local_dev"):
        raise ValueError("saved pose is outside the D1 interior KKT fixture")
    model, definition, _settings, program = build_interior_press(
        force_n=0.04, reference_slope=1.0e-4,
        source_support="perimeter", source_columns=4, source_rows=4,
        target_columns=1, target_rows=1,
        load_shape="interior_distributed", slide_body="target",
        steps_per_segment=1,
    )
    model.apply_boundary_conditions()
    ndof = model.mesh.dof_manager.total_dofs
    stiffness, _ = assemble_stiffness_matrix(model)
    _, _, transform, offset, _, _ = build_constraint_transformation(
        stiffness, np.zeros(ndof), model)
    if np.linalg.norm(offset) != 0.0:
        raise ValueError("KKT fixture requires zero model boundary values")
    transform = transform.tocsr()
    controls = np.asarray(program.target_at(1), dtype=float)
    if len(controls) != len(program.controls) or np.any(controls):
        raise ValueError("KKT fixture requires zero path-control values")
    controlled = []
    for control in program.controls:
        if len(control.terms) != 1 or control.terms[0][1:] != ("ux", 1.0):
            raise ValueError("KKT fixture needs simple nodal ux controls")
        dof = int(model.mesh.dof_manager.get_node_dofs(
            control.terms[0][0])[0])
        row = transform.getrow(dof)
        if row.nnz != 1 or row.data[0] != 1.0:
            raise ValueError("control is not one retained coordinate")
        controlled.append(int(row.indices[0]))
    free = np.setdiff1d(np.arange(transform.shape[1]), controlled)
    columns = np.asarray(transform[:, free].toarray())
    if np.any(np.sum(np.abs(columns) > 0.0, axis=0) != 1):
        raise ValueError("KKT fixture requires one physical DOF per free column")
    seed = solve_reference_singleton(
        model, definition, program, 1, allow_multimode=True)
    seed_pressure = np.asarray(seed["pressure"])
    seed_active = np.flatnonzero(seed_pressure > 0.0)
    saved_contact = saved["history"][-1]["contact"]
    saved_pressure = np.asarray(saved_contact["numerical_pressure_pa"])
    saved_active = np.flatnonzero(saved_pressure > 0.0)
    source_nodes = tuple(seed["source_pressure_nodes"])
    if (len(seed_active) != 1 or not np.array_equal(seed_active, saved_active)
            or source_nodes[int(seed_active[0])]
            not in saved["interior_pressure_node_ids"]):
        raise ValueError("independent seed does not select the saved interior branch")
    active = int(seed_active[0])
    active_faces = {element_id
        for element_id in definition.surfaces[0].element_ids
        if source_nodes[active] in model.mesh.elements[element_id].node_ids}
    pose = np.asarray(seed["displacement"])
    q = columns.T @ pose
    if np.max(abs(columns @ q-pose)) > 1e-12:
        raise ValueError("seed does not satisfy the declared constraints")
    z = float(seed_pressure[active]*np.asarray(seed["support"])[active])
    external = np.zeros(ndof)
    for load, value in zip(program.nodal_forces,
                           program.force_values_at(1), strict=True):
        if load.dof != "uz":
            raise ValueError("KKT fixture only admits vertical dead forces")
        external[model.mesh.dof_manager.get_node_dofs(load.node_id)[2]] += value
    steps = np.array([1e-5 if int(np.flatnonzero(columns[:, index])[0]) % 6 >= 3
                      else 1e-7 for index in range(len(free))])
    evaluations = 0
    initial_signature = None

    def evaluate(coordinates, nodal_force):
        nonlocal evaluations, initial_signature
        if time.perf_counter()-started > 165.0:
            raise TimeoutError("interior independent KKT budget exhausted")
        candidate_pose = columns @ coordinates
        support, weighted, _area, points, signature = (
            integrate_current_skin(model, definition, candidate_pose))
        evaluations += 1
        active_signature = tuple(item for item in signature
                                 if item[0] in active_faces)
        if initial_signature is None:
            initial_signature = active_signature
        elif active_signature != initial_signature:
            raise ValueError("independent KKT crossed an active clipping branch")
        gaps = weighted/support
        gradient = np.empty(len(free))
        for index, step in enumerate(steps):
            plus_support, plus_weighted, _, _, plus_signature = (
                integrate_current_skin(model, definition,
                                       candidate_pose+step*columns[:, index]))
            minus_support, minus_weighted, _, _, minus_signature = (
                integrate_current_skin(model, definition,
                                       candidate_pose-step*columns[:, index]))
            evaluations += 2
            for perturbed in (plus_signature, minus_signature):
                if tuple(item for item in perturbed
                         if item[0] in active_faces) != active_signature:
                    raise ValueError("independent KKT derivative crossed a clip branch")
            gradient[index] = (
                plus_weighted[active]/plus_support[active]
                -minus_weighted[active]/minus_support[active])/(2*step)
        with suspend_contact():
            internal, tangent, _ = _assemble_nonlinear_system(
                model, candidate_pose, {}, 5, tangent=True,
                kinematics="von_karman", require_full_coordinates=True)
        structural = columns.T @ (np.asarray(internal)-external)
        residual = structural-gradient*nodal_force
        reduced_tangent = (columns.T @ sparse.csr_matrix(tangent) @ columns)
        merit = float(np.hypot(np.linalg.norm(residual)/0.04,
                               abs(gaps[active])/5e-5))
        return (candidate_pose, support, gaps, gradient, residual,
                np.asarray(reduced_tangent), merit, points)

    trace = []
    previous_merit = float("inf")
    for iteration in range(4):
        (pose, support, gaps, gradient, residual, tangent,
         merit, points) = evaluate(q, z)
        trace.append({"iteration": iteration,
                      "scaled_merit": merit,
                      "free_residual_n": float(np.linalg.norm(residual)),
                      "active_gap_m": float(gaps[active]),
                      "active_force_n": z})
        if merit >= previous_merit:
            raise RuntimeError("independent nonlinear KKT merit did not decrease")
        if merit <= 1e-8:
            break
        previous_merit = merit
        matrix = np.block([
            [tangent, -gradient[:, None]],
            [gradient[None, :], np.zeros((1, 1))],
        ])
        correction = np.linalg.solve(matrix, -np.r_[residual, gaps[active]])
        candidate_z = z+float(correction[-1])
        if candidate_z <= 0.0:
            raise RuntimeError("independent KKT left the active pressure branch")
        q += correction[:-1]
        z = candidate_z
    else:
        raise RuntimeError("independent nonlinear KKT iteration budget exhausted")
    production_pose = np.asarray(saved["displacements"])
    production_force = float(saved_pressure[active]
                             *np.asarray(saved_contact["reference_support_m2"])[active])
    (face_check_support, _weighted, _face_area, _face_points,
     _face_signature, support_by_face) = integrate_current_skin(
         model, definition, pose, support_by_source_face=True)
    if abs(np.sum(support_by_face[:, active])
           -face_check_support[active]) > 1e-12:
        raise AssertionError("independent per-face support does not partition")
    independent_face_force = (z*support_by_face[:, active]
                              /face_check_support[active])
    production_face_force = np.zeros(len(independent_face_force))
    reference_face_support = np.zeros(len(independent_face_force))
    for index, face_id in enumerate(definition.surfaces[0].element_ids):
        element = model.mesh.elements[face_id]
        if source_nodes[active] not in element.node_ids:
            continue
        vertices = np.asarray([(model.mesh.nodes[node].x,
                                model.mesh.nodes[node].y,
                                model.mesh.nodes[node].z)
                               for node in element.node_ids])
        area = float(np.linalg.norm(np.cross(vertices[1]-vertices[0],
                                             vertices[3]-vertices[0])))
        reference_face_support[index] = area/4.0
        production_face_force[index] = saved_pressure[active]*area/4.0
    support_face_error = float(np.max(np.abs(
        support_by_face[:, active]-reference_face_support)))
    face_force_l1_error = float(np.sum(np.abs(
        independent_face_force-production_face_force))
        /np.sum(np.abs(production_face_force)))
    record = {
        "status": "development_d1_interior_independent_contact_nonlinear_kkt",
        "saved_pose": str(saved_path),
        "active_node_id": source_nodes[active],
        "free_coordinates": len(free),
        "geometry_evaluations": evaluations,
        "quadrature_points_per_evaluation": points,
        "trace": trace,
        "maximum_displacement_difference_m_or_rad": float(np.max(abs(
            pose-production_pose))),
        "relative_displacement_l2_difference": float(np.linalg.norm(
            pose-production_pose)/np.linalg.norm(production_pose)),
        "independent_integrated_force_n": z,
        "production_integrated_force_n": production_force,
        "relative_integrated_force_difference": float(abs(z-production_force)
            /abs(production_force)),
        "maximum_per_face_support_difference_m2": support_face_error,
        "relative_per_face_bin_force_l1_difference": face_force_l1_error,
        "active_source_face_ids": [int(face_id)
            for face_id, force in zip(definition.surfaces[0].element_ids,
                                      independent_face_force, strict=True)
            if force > 0.0],
        "minimum_inactive_gap_m": float(np.min(np.delete(gaps, active))),
        "elapsed_seconds": time.perf_counter()-started,
    }
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record))


def coupled_y_comparison(output: Path, reference_path: Path,
                         saved_paths: list[Path]) -> None:
    if output.exists() or len(saved_paths) not in (1, 2):
        raise ValueError("coupled y comparison needs new output and one or two saved poses")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    cases = []
    for path in saved_paths:
        saved = json.loads(path.read_text(encoding="utf-8"))
        rows = saved["source_rows"]
        if (rows not in (8, 16) or saved["source_columns"] != 8
                or saved["target_s3_faces"] != 4
                or saved["status"] != "completed"
                or saved["accepted_steps"] != 2):
            raise ValueError(f"saved pose is outside bounded y comparison: {path}")
        corresponding = next(item for item in reference["cases"]
                             if item["source_rows"] == rows)
        if saved["load_shape"] != corresponding.get(
                "load_shape", "interior_distributed"):
            raise ValueError("coupled and oracle load fields differ")
        model, _definition, _settings, _program = build_interior_press(
            source_support="perimeter", source_columns=8,
            source_rows=rows,
            source_y_profile="uniform_ref16" if rows == 16 else None,
            offline_structural_reference_y16=(rows == 16),
            target_columns=2, target_rows=1, force_n=0.04,
            reference_slope=1.0e-4, load_shape=saved["load_shape"],
            slide_body="target",
        )
        production = saved["history"][-1]["contact"]
        pressure = np.asarray(production["numerical_pressure_pa"])
        production_force = float(production["integration_force"][2])
        reference_force = corresponding["integrated_normal_force_n"]
        samples = _source_vertical_samples(
            model, np.asarray(saved["displacements"]), rows,
            corresponding["sample_xy_m"])
        reference_samples = corresponding["source_vertical_displacement_m"]
        cases.append({
            "saved_pose": str(path), "source_rows": rows,
            "source_sample_xy_m": corresponding["sample_xy_m"],
            "production_source_vertical_displacement_m": samples,
            "oracle_source_vertical_displacement_m": reference_samples,
            "relative_source_displacement_difference": [
                abs(a-b)/max(abs(b), 1e-15)
                for a, b in zip(samples, reference_samples)],
            "production_active_node_ids": [node for node, value in zip(
                production["source_pressure_node_ids"], pressure)
                if value > 0.0],
            "oracle_active_node_ids": corresponding["oracle_active_node_ids"],
            "production_integrated_normal_force_n": production_force,
            "oracle_integrated_normal_force_n": reference_force,
            "relative_integrated_force_difference": abs(
                production_force-reference_force)/max(abs(reference_force), 1e-15),
        })
    output.write_text(json.dumps({"cases": cases}, indent=2)+"\n",
                      encoding="utf-8")
    print(json.dumps(cases))


def bump_gap_preflight(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    from examples.general_contact_d1_interior_controls_dev import (
        _independent_projected_gap_bound,
    )

    model, definition, _settings, program = build_interior_press(
        source_support="perimeter", source_columns=8, source_rows=16,
        source_y_profile="uniform_ref16",
        offline_structural_reference_y16=True,
        target_columns=2, target_rows=1, force_n=0.04,
        reference_slope=1.0e-4, load_shape="physical_bump",
        slide_body="target",
    )
    reference = solve_reference_singleton(
        model, definition, program, 2, allow_multimode=True)
    tolerance = definition.interactions[0].gap_tolerance
    try:
        gap = _independent_projected_gap_bound(
            model, definition, reference["displacement"], tolerance)
    except RuntimeError as error:
        output.write_text(json.dumps({
            "status": "development_preflight_failed",
            "failure": str(error),
            "active_node_ids": reference["active_node_ids"],
            "integrated_normal_force_n": reference["integrated_force"],
            "gap_tolerance_m": tolerance,
        }, indent=2)+"\n", encoding="utf-8")
        raise
    record = {
        "status": "development_preflight_passed",
        "active_node_ids": reference["active_node_ids"],
        "integrated_normal_force_n": reference["integrated_force"],
        "independent_projected_lower_gap_m": gap,
        "gap_tolerance_m": tolerance,
    }
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--singleton-family":
        singleton_family(Path(sys.argv[2]), [Path(path) for path in sys.argv[3:]])
    elif len(sys.argv) > 1 and sys.argv[1] == "--offline-y-reference":
        offline_y_reference(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--offline-y-reference-bump":
        offline_y_reference(Path(sys.argv[2]), load_shape="physical_bump")
    elif len(sys.argv) > 1 and sys.argv[1] == "--offline-x-reference":
        offline_x_reference(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--offline-target-reference":
        offline_target_reference(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--target8s3-coupled-oracle":
        target8s3_coupled_oracle(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--offline-target-xy-reference":
        offline_target_xy_reference(
            Path(sys.argv[2]), Path(sys.argv[3]),
            tuple(map(int, sys.argv[4:])) if len(sys.argv) > 4 else (3, 4))
    elif len(sys.argv) > 1 and sys.argv[1] == "--offline-target-contrast":
        offline_target_xy_reference(Path(sys.argv[2]), None, (4, 6),
                                    target_young_multiplier=5.0)
    elif len(sys.argv) > 1 and sys.argv[1] == "--offline-target-contrast-source8":
        offline_target_xy_reference(Path(sys.argv[2]), None, (6,),
                                    target_young_multiplier=5.0,
                                    source_cells=8)
    elif len(sys.argv) > 1 and sys.argv[1] == "--offline-target-contrast-source12":
        offline_target_xy_reference(Path(sys.argv[2]), None, (4, 6),
                                    target_young_multiplier=5.0,
                                    source_cells=12)
    elif len(sys.argv) > 1 and sys.argv[1] == "--offline-rotation-load-screen":
        offline_target_xy_reference(Path(sys.argv[2]), None, (4,),
                                    target_young_multiplier=5.0,
                                    source_cells=12, force_n=0.08)
    elif len(sys.argv) > 1 and sys.argv[1] == "--pressure-family-source4":
        offline_target_xy_reference(Path(sys.argv[2]), None, (4,),
                                    target_young_multiplier=5.0,
                                    source_cells=4)
    elif len(sys.argv) > 1 and sys.argv[1] == "--pressure-family-source8":
        offline_target_xy_reference(Path(sys.argv[2]), None, (4,),
                                    target_young_multiplier=5.0,
                                    source_cells=8)
    elif len(sys.argv) > 1 and sys.argv[1] == "--pressure-family-shifted-source8":
        offline_target_xy_reference(Path(sys.argv[2]), None, (4,),
                                    target_young_multiplier=5.0,
                                    source_cells=8, target_y0_m=-0.113)
    elif len(sys.argv) > 1 and sys.argv[1] == "--offline-target-contrast-source24":
        offline_target_xy_reference(Path(sys.argv[2]), None, (6,),
                                    target_young_multiplier=5.0,
                                    source_cells=24)
    elif len(sys.argv) > 1 and sys.argv[1] == "--target8s3-coupled-comparison":
        target8s3_coupled_comparison(Path(sys.argv[2]), Path(sys.argv[3]),
                                    Path(sys.argv[4]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--coupled-contrast-comparison":
        coupled_contrast_comparison(Path(sys.argv[2]), Path(sys.argv[3]),
                                    Path(sys.argv[4]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--lowload-geometry-decomposition":
        lowload_geometry_decomposition(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--lowload-current-skin-check":
        lowload_current_skin_check(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--interior-saved-kkt-check":
        interior_saved_kkt_check(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--interior-nonlinear-kkt-compare":
        interior_nonlinear_kkt_compare(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--coupled-y-comparison":
        coupled_y_comparison(Path(sys.argv[2]), Path(sys.argv[3]),
                             [Path(path) for path in sys.argv[4:]])
    elif len(sys.argv) > 1 and sys.argv[1] == "--bump-gap-preflight":
        bump_gap_preflight(Path(sys.argv[2]))
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
