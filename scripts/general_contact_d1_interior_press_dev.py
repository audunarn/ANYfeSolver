"""Bounded development press for an interior Q4/S3 two-body contact patch.

This is a feasibility fixture, not a qualification case. The source rotations
are free, the S3 target bends, and source loads act on declared interior nodes.
Report active rim pressure explicitly rather than hiding it.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import anyfesolver
import numpy as np
from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathNodalForce,
    ContactPathProgram, solve_static_nonlinear, validate_general_contact,
)

from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import (
    _json_default, build_case,
)


def _physical_bump_axis_weights(count: int) -> np.ndarray:
    """Consistent Q1 loads from one fixed physical-axis traction shape."""
    roots, quadrature = np.polynomial.legendre.leggauss(8)
    nodal = np.zeros(count+1)
    for element in range(count):
        cell_left, cell_right = element/count, (element+1)/count
        left, right = max(cell_left, 0.25), min(cell_right, 0.75)
        if right <= left:
            continue
        position = 0.5*(left+right)+0.5*(right-left)*roots
        weight = 0.5*(right-left)*quadrature
        density = np.sin(2*np.pi*(position-0.25))**2
        local = (position-cell_left)*count
        nodal[element] += float(np.sum(weight*density*(1-local)))
        nodal[element+1] += float(np.sum(weight*density*local))
    # Integral of sin² over the fixed half-width support is 1/4.
    if abs(float(np.sum(nodal))-0.25) > 1.0e-12:
        raise RuntimeError("physical bump load quadrature did not converge")
    return nodal/0.25


def build_interior_press(*, force_n: float = 0.08,
                         reference_slope: float = 0.01,
                         target_columns: int = 1,
                         target_rows: int = 1,
                         target_young_multiplier: float = 1.0,
                         target_y0_m: float = -0.1,
                         source_reference_downshift_m: float = 0.0,
                         source_support: str = "left",
                         source_columns: int = 4,
                         source_rows: int = 2,
                         source_y_profile: str | None = None,
                         offline_structural_reference_y16: bool = False,
                         offline_structural_reference_y24: bool = False,
                         offline_structural_reference_y32: bool = False,
                         load_shape: str = "point",
                         four_phase: bool = False,
                         steps_per_segment: int = 2,
                         slide_distance_m: float = 0.05,
                         slide_intervals: int = 1,
                         slide_body: str = "source",
                         target_reference_full_left_clamp: bool = False,
                         target_left_edge_motion: bool = False,
                         shared_edge_tie: bool = False):
    if not np.isfinite(slide_distance_m) or slide_distance_m <= 0.0:
        raise ValueError("slide distance must be positive and finite")
    if source_reference_downshift_m not in (0.0, 35.0e-6):
        raise ValueError("bounded source downshift must be 0 or 35 µm")
    if type(slide_intervals) is not int or not 1 <= slide_intervals <= 10:
        raise ValueError("slide intervals must be an integer in [1, 10]")
    if slide_body not in {"source", "target", "target_diagonal_dev"}:
        raise ValueError("slide body must be source, target or target_diagonal_dev")
    if (slide_body == "target_diagonal_dev"
            and (target_columns != 1 or target_rows != 1)):
        raise ValueError("diagonal target development control needs two S3 faces")
    if target_reference_full_left_clamp and (
            target_rows <= 1 or slide_body != "target"):
        raise ValueError("full target left clamp requires multirow target control")
    if target_left_edge_motion and not target_reference_full_left_clamp:
        raise ValueError("moving target edge requires the full left clamp")
    columns, rows = source_columns, source_rows
    stride = columns + 1
    model, definition, settings, _program = build_case(
        columns, source_rows=rows, full_width_target=True,
        full_width_target_y0_m=target_y0_m,
        target_right_x_m=1.3, common_reference_slope=reference_slope,
        target_columns=target_columns, target_rows=target_rows,
        target_young_multiplier=target_young_multiplier,
        load_profile="distributed", total_force_n=force_n,
        source_y_profile=source_y_profile,
        offline_structural_reference_y16=offline_structural_reference_y16,
        offline_structural_reference_y24=offline_structural_reference_y24,
        offline_structural_reference_y32=offline_structural_reference_y32,
    )
    target_start = (rows + 1) * (columns + 1) + 1
    if source_reference_downshift_m:
        for node_id in range(1, target_start):
            node = model.mesh.nodes[node_id]
            model.set_node_coordinates(
                node_id, node.x, node.y,
                node.z-source_reference_downshift_m,
            )
    target_left_top = target_start + (
        3 if target_columns == 1 and target_rows == 1
        else target_rows*(target_columns+1))
    target_control_top = (target_start + 2 if slide_body == "target_diagonal_dev"
                          else target_left_top)
    model.boundary_conditions = [
        bc for bc in model.boundary_conditions
        if bc.name not in {"target_fixed", "source_guide"}
        and not (source_support == "perimeter"
                 and bc.name == "source_left_support")
    ]
    model.add_boundary_condition(BoundaryCondition(
        "source_lateral_guide", tuple(range(1, target_start)), {"uy": 0.0},
    ))
    if source_support == "left":
        model.add_boundary_condition(BoundaryCondition(
            "source_left_bending_clamp",
            tuple(1 + row * stride for row in range(rows + 1)),
            {"ry": 0.0},
        ))
    elif source_support == "perimeter":
        rim = (tuple(range(1, stride + 1))
               + tuple(node for row in range(1, rows)
                       for node in (1 + row * stride, (row + 1) * stride))
               + tuple(range(1 + rows * stride,
                             1 + (rows + 1) * stride)))
        model.add_boundary_condition(BoundaryCondition(
            "source_perimeter_support", rim, {"uz": 0.0},
        ))
    else:
        raise ValueError("source support must be left or perimeter")
    model.add_boundary_condition(BoundaryCondition(
        "target_control_support", (target_start, target_control_top),
        {name: 0.0 for name in (("ux", "uy", "uz", "rx", "ry", "rz")
                                    if slide_body == "source"
                                    else ("uy", "uz", "rx", "ry", "rz"))},
    ))
    if target_reference_full_left_clamp:
        middle_left = tuple(target_start+row*(target_columns+1)
                            for row in range(1, target_rows))
        model.add_boundary_condition(BoundaryCondition(
            "target_reference_middle_left_clamp", middle_left,
            {name: 0.0 for name in (("uy", "uz", "rx", "ry", "rz")
                                   if target_left_edge_motion else
                                   ("ux", "uy", "uz", "rx", "ry", "rz"))},
        ))
        if target_left_edge_motion:
            anchor = model.mesh.dof_manager.get_node_dofs(target_start)[0]
            for node in middle_left:
                dependent = model.mesh.dof_manager.get_node_dofs(node)[0]
                model.add_constraint_equation(
                    terms=((dependent, 1.0), (anchor, -1.0)),
                    source_id=f"moving_target_left_ux_{node}")
    if slide_body != "source":
        model.add_boundary_condition(BoundaryCondition(
            "source_x_anchor", (1,), {"ux": 0.0},
        ))
    settings = replace(
        settings, enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=force_n,
        augmentation_acceleration="none",
        augmentation_schur_maximum_modes=None,
        augmentation_schur_maximum_pressure_pa=None,
        augmentation_schur_maximum_step_pa=None,
        augmentation_schur_difference_step_m=None,
        augmentation_schur_maximum_attempts=1,
        augmentation_schur_maximum_relinearizations=0,
        d1_interior_profile=True,
        d1_interior_coverage_tolerance_m2=1.0e-10,
        d1_interior_boundary_force_tolerance_n=1.0e-10,
        **({
            "q4_s3_shared_edge_tie_step_m": 3.0e-8,
            "q4_s3_shared_edge_tie_direction": (1.0, 0.0, 0.0),
            "q4_s3_shared_edge_tie_relative_tolerance": 1.0e-6,
        } if shared_edge_tie else {}),
    )
    interior = tuple(1 + row * stride + column
                     for row in range(1, rows)
                     for column in range(1, columns))
    if load_shape == "point":
        load_weights = {1 + (rows // 2) * stride + columns // 2: 1.0}
    elif load_shape == "interior_distributed":
        raw = {node: min(row, rows - row) * min(column, columns - column)
               for row in range(1, rows) for column in range(1, columns)
               for node in (1 + row * stride + column,)}
        total = float(sum(raw.values()))
        load_weights = {node: weight / total for node, weight in raw.items()}
    elif load_shape == "interior_uniform":
        load_weights = {node: 1.0 / len(interior) for node in interior}
    elif load_shape == "physical_bump":
        if source_y_profile not in (None, "uniform_ref16", "uniform_ref24", "uniform_ref32"):
            raise ValueError("physical bump requires uniform source coordinates")
        x_weights = _physical_bump_axis_weights(columns)
        y_weights = _physical_bump_axis_weights(rows)
        load_weights = {
            1+row*stride+column: float(x_weights[column]*y_weights[row])
            for row in range(rows+1) for column in range(columns+1)
            if x_weights[column]*y_weights[row] > 1.0e-15
        }
        if abs(sum(load_weights.values())-1.0) > 1.0e-11:
            raise RuntimeError("physical bump nodal loads do not balance")
    else:
        raise ValueError("unknown bounded interior load shape")
    controlled_nodes = ((1, columns + 2) if slide_body == "source" else
                        (target_start, target_control_top))
    waypoints = ((0.0,) * len(controlled_nodes),) * 2
    force_factors = (0.0, 1.0)
    if four_phase:
        waypoints += tuple((slide_distance_m * i / slide_intervals,) * len(controlled_nodes)
                           for i in range(1, slide_intervals + 1))
        waypoints += ((slide_distance_m,) * len(controlled_nodes),) * 2
        force_factors += (1.0,) * slide_intervals + (0.0, 1.0)
    program = ContactPathProgram(
        controls=tuple(ContactPathControl.node(node, "ux")
                       for node in controlled_nodes),
        waypoints=waypoints,
        steps_per_segment=steps_per_segment,
        nodal_forces=tuple(ContactPathNodalForce(
            node, "uz", tuple(-force_n * weight * factor
                              for factor in force_factors),
        ) for node, weight in load_weights.items()),
    )
    return model, definition, settings, program


def validate_restart_poses(source, checkpoint):
    """Validate caller-supplied prefix inventory and its checkpoint endpoint.

    Earlier poses are caller-supplied evidence, not authenticated by a terminal
    checkpoint. A guarded full-route comparison can verify their correspondence.
    """
    count=checkpoint['path_state']['accepted_step']
    poses=np.asarray(source.get('accepted_pose_displacements',[]),dtype=float)
    terminal=np.asarray(checkpoint['displacements'],dtype=float)
    if (source.get('status')!='completed' or source.get('accepted_steps')!=count
            or type(count) is not int or count<=0 or poses.shape!=(count,len(terminal))
            or not np.all(np.isfinite(poses)) or not np.all(np.isfinite(terminal))
            or not np.array_equal(poses[-1],terminal)):
        raise ValueError('restart source has incompatible prefix poses or checkpoint endpoint')
    return source['accepted_pose_displacements']

def main(output: Path, *, force_n: float, reference_slope: float,
         target_columns: int, target_rows: int, source_support: str,
         target_young_multiplier: float,
         target_y0_m: float,
         source_reference_downshift_m: float,
         source_columns: int, source_rows: int, load_shape: str,
         source_y_profile: str | None,
         offline_structural_reference_y16: bool,
         max_path_fraction: float, four_phase: bool,
         steps_per_segment: int, slide_distance_m: float,
         slide_intervals: int, slide_body: str,
         target_reference_full_left_clamp: bool,
         shared_edge_tie: bool,
         capture_swept_failure: bool,
         record_swept_fractions: bool,
         q4_gap_certificate: str | None,
         record_accepted_poses: bool = False,
         restart_checkpoint_path: Path | None = None,
         target_left_edge_motion: bool = False):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    contact_directory = Path(anyfesolver.__file__).resolve().parent / "general_contact"
    changed_kernel_identity = {
        name: hashlib.sha256((contact_directory / name).read_bytes()).hexdigest()
        for name in ("d1_interior_profile.py", "q4_overlap_local_bound.py",
                     "s3_patch_static.py")}
    model, definition, settings, program = build_interior_press(
        force_n=force_n, reference_slope=reference_slope,
        target_columns=target_columns, target_rows=target_rows,
        target_young_multiplier=target_young_multiplier,
        target_y0_m=target_y0_m,
        source_reference_downshift_m=source_reference_downshift_m,
        source_support=source_support,
        source_columns=source_columns, source_rows=source_rows,
        source_y_profile=source_y_profile,
        offline_structural_reference_y16=offline_structural_reference_y16,
        load_shape=load_shape,
        four_phase=four_phase, steps_per_segment=steps_per_segment,
        slide_distance_m=slide_distance_m,
        slide_intervals=slide_intervals,
        slide_body=slide_body,
        target_reference_full_left_clamp=target_reference_full_left_clamp,
        target_left_edge_motion=target_left_edge_motion,
        shared_edge_tie=shared_edge_tie,
    )
    if q4_gap_certificate is not None:
        settings = replace(settings,
                           q4_source_gap_certificate=q4_gap_certificate)
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError(f"fixture capability rejected: {report.issues}")
    restart = None
    restart_source_payload = None
    if restart_checkpoint_path is not None:
        restart_source_payload = json.loads(restart_checkpoint_path.read_text(
            encoding="utf-8"))
        restart = restart_source_payload["restart_checkpoint"]
        if restart is None:
            raise ValueError("restart source contains no committed checkpoint")
        if record_accepted_poses:
            validate_restart_poses(restart_source_payload,restart)
    captured = []
    swept_fractions = []
    original_admissible_fraction = None
    if capture_swept_failure or record_swept_fractions:
        from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager

        original_admissible_fraction = (
            GeneralS3PatchManager.admissible_step_fraction)

        def capture_fraction(owner, start, end):
            fraction = original_admissible_fraction(owner, start, end)
            if record_swept_fractions:
                swept_fractions.append(float(fraction))
            if capture_swept_failure and fraction == 0.0:
                captured.append((owner, np.asarray(start).copy(),
                                 np.asarray(end).copy()))
            return fraction

        GeneralS3PatchManager.admissible_step_fraction = capture_fraction
    try:
        result = solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            max_load_factor=max_path_fraction,
            emit_restart_checkpoint=True,
            restart_checkpoint=restart,
            record_increment_snapshots=record_accepted_poses,
        )
    finally:
        if original_admissible_fraction is not None:
            GeneralS3PatchManager.admissible_step_fraction = (
                original_admissible_fraction)
    step_records = []
    stride = source_columns + 1
    interior_nodes = tuple(1 + row * stride + column
                           for row in range(1, source_rows)
                           for column in range(1, source_columns))
    for row in result.info.get("force_displacement_history", ()):
        contact = row.get("contact", {})
        pressure = dict(zip(contact.get("source_pressure_node_ids", ()),
                            contact.get("numerical_pressure_pa", ())))
        step_records.append({
            **row,
            "step": row.get("step_index"),
            "accepted": row.get("accepted"),
            "backtracks": row.get("backtracks"),
            "semismooth_iterations": row.get("semismooth_iterations"),
            "contact": contact,
            "rim_pressure_pa": {
                str(node): value for node, value in pressure.items()
                if node not in interior_nodes and value > 0.0
            },
        })
    payload = {
        "candidate": "d1_interior_free_rotation_press_development",
        "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
        "changed_kernel_sha256": changed_kernel_identity,
        "capability_profile": report.profile,
        "source_q4_faces": source_columns * source_rows,
        "target_s3_faces": 2 * target_columns * target_rows,
        "target_rows": target_rows,
        "target_columns": target_columns,
        "target_young_multiplier": target_young_multiplier,
        "target_y0_m": target_y0_m,
        "source_reference_downshift_m": source_reference_downshift_m,
        "interior_pressure_node_ids": list(interior_nodes),
        "force_n": force_n,
        "reference_slope": reference_slope,
        "source_support": source_support,
        "source_rows": source_rows,
        "source_columns": source_columns,
        "source_y_profile": source_y_profile,
        "offline_structural_reference_y16": offline_structural_reference_y16,
        "load_shape": load_shape,
        "max_path_fraction": max_path_fraction,
        "four_phase": four_phase,
        "steps_per_segment": steps_per_segment,
        "slide_distance_m": slide_distance_m,
        "slide_intervals": slide_intervals,
        "slide_body": slide_body,
        "target_reference_full_left_clamp": target_reference_full_left_clamp,
        "target_left_edge_motion": target_left_edge_motion,
        "shared_edge_tie": shared_edge_tie,
        "q4_gap_certificate": settings.q4_source_gap_certificate,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "elapsed_seconds": time.perf_counter() - started,
        "displacements": result.displacements.tolist(),
        "history": step_records,
        "contact_failure": result.info.get("contact_failure"),
    }
    if record_swept_fractions:
        payload["swept_fraction_trace"] = swept_fractions
    if record_accepted_poses:
        new_poses = [
            np.asarray(snapshot.displacements).tolist()
            for snapshot in result.snapshots]
        prefix_poses = []
        if restart_source_payload is not None:
            prefix_poses = validate_restart_poses(restart_source_payload,restart)
        payload["accepted_pose_displacements"] = prefix_poses + new_poses
        payload["new_increment_count"] = len(new_poses)
        payload["restart_prefix_pose_count"] = len(prefix_poses)
        payload["prefix_pose_provenance"] = "caller-supplied earlier poses; checkpoint endpoint validated; input-file hash retained"
        payload["restart_source_sha256"] = (hashlib.sha256(
            restart_checkpoint_path.read_bytes()).hexdigest() if restart_checkpoint_path else None)
        payload["restart_checkpoint"] = (
            result.to_restart_checkpoint() if result.restart_checkpoint is not None
            else None)
        payload["restart_checkpoint_available"] = result.restart_checkpoint is not None
        payload["restart_source"] = (
            str(restart_checkpoint_path) if restart_checkpoint_path else None)
    if capture_swept_failure:
        from anyfesolver.general_contact.q4_overlap_local_bound import (
            overlap_local_swept_lower,
        )
        from anyfesolver.general_contact.s3_pair_static import _normal_controls
        from examples.general_contact_d1_interior_controls_dev import (
            _independent_projected_gap_bound,
        )

        diagnostic = {"zero_fraction_proposals": len(captured)}
        if captured:
            owner, start, proposed = captured[-1]
            start_position = owner._positions(start)
            trials = []
            for fraction in (1/32, 1/16, 1/8):
                endpoint = start+fraction*(proposed-start)
                end_position = owner._positions(endpoint)
                local_lowers = [overlap_local_swept_lower(
                    start_position[0][si], end_position[0][si],
                    start_position[1][ti], end_position[1][ti], 3,
                )[0] for si in owner.law.source_face_indices
                    for ti in owner.law.target_face_indices]
                orientation_min = min(float(control @ reference)
                                      for side, indices, reference in (
                                          (0, owner.law.source_face_indices,
                                           owner.reference_normals[0]),
                                          (1, owner.law.target_face_indices,
                                           owner.reference_normals[1]))
                                      for face in indices
                                      for triple in (((0, 1, 2),)
                                                     if len(face) == 3 else
                                                     ((0, 1, 2), (1, 2, 3),
                                                      (2, 3, 0), (3, 0, 1)))
                                      for control in _normal_controls(
                                          start_position[side][face[list(triple)]],
                                          end_position[side][face[list(triple)]]))
                try:
                    independent_lower = _independent_projected_gap_bound(
                        model, definition, endpoint,
                        definition.interactions[0].gap_tolerance)
                    independent_issue = None
                except RuntimeError as error:
                    independent_lower = None
                    independent_issue = str(error)
                trials.append({
                    "fraction": fraction,
                    "projected_endpoint_lower_gap_m": owner._gap_lower(
                        end_position),
                    "overlap_local_swept_lower_gap_m": min(local_lowers),
                    "swept_orientation_minimum": orientation_min,
                    "independent_endpoint_lower_gap_m": independent_lower,
                    "independent_endpoint_issue": independent_issue,
                })
            diagnostic.update({
                "captured_start_displacement": start.tolist(),
                "captured_proposed_displacement": proposed.tolist(),
                "start_projected_lower_gap_m": owner._gap_lower(
                    start_position),
                "trials": trials,
            })
        payload["swept_failure_discriminator"] = diagnostic
    output.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "status", "failure_reason", "accepted_steps", "elapsed_seconds",
        "contact_failure",
    )}, default=_json_default))


def build_saved_cycle(saved):
    """Reconstruct a recorded development fixture without changing its inputs."""
    keys = ("force_n", "reference_slope", "source_columns", "source_rows",
            "target_columns", "target_rows", "target_young_multiplier",
            "target_y0_m", "source_reference_downshift_m", "source_support",
            "source_y_profile", "offline_structural_reference_y16", "load_shape",
            "four_phase", "steps_per_segment", "slide_distance_m",
            "slide_intervals", "slide_body", "target_reference_full_left_clamp",
            "shared_edge_tie")
    model, definition, settings, program = build_interior_press(
        **{key: saved[key] for key in keys},
        target_left_edge_motion=saved.get("target_left_edge_motion", False))
    return model, definition, replace(
        settings, q4_source_gap_certificate=saved["q4_gap_certificate"]), program


def verify_saved_cycle(saved_path: Path, output: Path):
    """Independently check physical gaps and retained equilibrium of a full cycle."""
    if output.exists():
        raise FileExistsError(output)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    if (saved["status"] != "completed" or saved["accepted_steps"] != 8
            or len(saved.get("accepted_pose_displacements", ())) != 8):
        raise ValueError("cycle verification requires all eight accepted poses")
    from examples.general_contact_d1_interior_controls_dev import (
        _independent_projected_gap_bound,
    )
    model, definition, settings, _ = build_saved_cycle(saved)
    gap_budget = definition.interactions[0].gap_tolerance
    rows = saved["restart_checkpoint"]["path_state"]["history"]
    records = []
    for pose, row in zip(saved["accepted_pose_displacements"], rows, strict=True):
        contact = row["contact"]
        lower = _independent_projected_gap_bound(
            model, definition, np.asarray(pose), gap_budget)
        force_error = float(np.linalg.norm(np.asarray(contact["integration_force"])
                                          + contact["target_force"]))
        moment_error = float(np.linalg.norm(np.asarray(contact["integration_moment"])
                                           + contact["target_moment"]))
        records.append(dict(step=row["step_index"], independent_gap_lower_m=lower,
                            equilibrium_relative_force=row["residual_norm"]/saved["force_n"],
                            complementarity_relative_force=row["complementarity_relative_force"],
                            contact_balance_relative_force=force_error/saved["force_n"],
                            contact_balance_relative_moment=moment_error/(saved["force_n"]*1.0),
                            active_modes=contact["active_pressure_dofs"]))
    checks = (all(r["independent_gap_lower_m"] >= -gap_budget
                  and r["equilibrium_relative_force"] <= 1e-6
                  and r["complementarity_relative_force"] <= 1e-8
                  and r["contact_balance_relative_force"] <= 1e-6
                  and r["contact_balance_relative_moment"] <= 1e-6 for r in records)
              and records[5]["active_modes"] == 0 and records[7]["active_modes"] > 0
              and all(value == 0.0 for value in rows[5]["contact"]["numerical_pressure_pa"]))
    output.write_text(json.dumps(dict(status="development_cycle_checks",
        saved_input=str(saved_path), input_sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest(),
        force_scale_n=saved["force_n"], moment_scale_nm=saved["force_n"]*1.0,
        dimensional_gap_budget_m=gap_budget, development_checks_met=checks,
        rows=records), indent=2), encoding="utf-8")
    if not checks:
        raise RuntimeError("saved cycle failed its independent development checks")


def compare_saved_cycles(reference_path: Path, candidate_path: Path, output: Path):
    """Compare a complete accepted route with its recorded split continuation."""
    if output.exists():
        raise FileExistsError(output)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    for data in (reference, candidate):
        if data["status"] != "completed" or data["accepted_steps"] != 8:
            raise ValueError("restart comparison needs two complete accepted cycles")
    physics = ("force_n", "reference_slope", "source_columns", "source_rows",
               "target_columns", "target_rows", "target_young_multiplier", "target_y0_m",
               "source_support", "load_shape", "slide_body", "slide_distance_m",
               "target_reference_full_left_clamp", "target_left_edge_motion",
               "q4_gap_certificate")
    if any(reference.get(key) != candidate.get(key) for key in physics):
        raise ValueError("restart comparison changed physical fixture inputs")
    displacement_error = float(np.max(np.abs(
        np.asarray(reference["displacements"])-candidate["displacements"])))
    errors = dict(displacement_m=displacement_error)
    for first, last in zip(reference["restart_checkpoint"]["path_state"]["history"],
                           candidate["restart_checkpoint"]["path_state"]["history"], strict=True):
        for key in ("integration_force", "target_force", "integration_moment",
                    "target_moment", "minimum_gap", "numerical_pressure_pa"):
            value = float(np.max(np.abs(np.asarray(first["contact"][key])
                                       -last["contact"][key])))
            errors[key] = max(errors.get(key, 0.0), value)
        errors["reaction_generalized_n"] = max(errors.get("reaction_generalized_n", 0.0),
            float(np.max(np.abs(np.asarray(first["reaction_generalized"])
                                -last["reaction_generalized"]))))
        if first["contact"]["active_pressure_dofs"] != last["contact"]["active_pressure_dofs"]:
            raise RuntimeError("split restart changed an accepted active set")
    checks = all(value <= 1e-10 for value in errors.values())
    output.write_text(json.dumps(dict(status="development_split_restart_comparison",
        reference=str(reference_path), candidate=str(candidate_path),
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        absolute_development_tolerance=1e-10, maximum_differences=errors,
        development_checks_met=checks), indent=2), encoding="utf-8")
    if not checks:
        raise RuntimeError("split restart differs beyond the declared development tolerance")


def build_radial_guided_case(*, target_cells=4, target_y0_m=-0.113,
                            target_right_x_m=1.3):
    """A physical-load spatial benchmark; target MIN3 shear is retained."""
    from docs.d1_review_support.d1_radial_two_body_reference import RadialTwoBodyReference
    model, definition, settings, _ = build_case(
        12, source_rows=12, target_columns=target_cells, target_rows=target_cells,
        full_width_target=True, target_right_x_m=target_right_x_m,
        full_width_target_y0_m=target_y0_m,
        target_young_multiplier=5.0,
        guided_radial_target_refinement=target_cells == 12)
    ds = (5.0/6.0)*0.05*1e5/(2*(1+0.3))
    coefficients, phi_values, scalar_errors = [], [], []
    for element_id in definition.surfaces[1].element_ids:
        face = model.mesh.elements[element_id]
        component = face.compute_stiffness_components(
            model.mesh, model.materials[face.material_name])
        phi = component["phi_squared"]
        dt = phi*5*ds
        points = np.array([(model.mesh.nodes[node].x, model.mesh.nodes[node].y)
                           for node in face.node_ids])
        inverse = np.linalg.inv(np.column_stack((np.ones(3), points)))
        gradients = inverse[1:, :]
        area = abs(np.linalg.det(np.column_stack((points[1]-points[0],
                                                 points[2]-points[0]))))/2
        expected = dt*area*(gradients.T @ gradients)
        actual = component["total"][np.ix_((2, 8, 14), (2, 8, 14))]
        scalar_errors.append(float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)))
        phi_values.append(float(phi))
        coefficients.append(float(dt))
    dt = coefficients[0]
    if (max(scalar_errors) > 1e-11
            or max(abs(value-dt) for value in coefficients) > 1e-11*dt):
        raise ValueError("target does not match one frozen scalar MIN3 operator")
    reference = RadialTwoBodyReference(ds, dt)
    from examples.general_contact_d1_interior_controls_dev import _physical_skin_vertices
    source_ids = tuple(range(1, 170))
    target_ids = tuple(range(170, 170+(target_cells+1)**2))
    zero = np.zeros(model.mesh.dof_manager.total_dofs)
    source_face = model.mesh.elements[definition.surfaces[0].element_ids[0]]
    target_face = model.mesh.elements[definition.surfaces[1].element_ids[0]]
    current_gap = (_physical_skin_vertices(model, source_face, zero, "bottom")[0, 2]
                   -_physical_skin_vertices(model, target_face, zero, "top")[0, 2])
    for node in source_ids:
        point = model.mesh.nodes[node]
        model.set_node_coordinates(node, point.x, point.y,
                                   point.z+reference.gap_m-current_gap)
    model.boundary_conditions = []
    model.add_boundary_condition(BoundaryCondition("radial_guidance", source_ids+target_ids,
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")}))
    anchors = []
    for body, ids, cells in (("source_w", source_ids, 12),
                             ("target_w", target_ids, target_cells)):
        anchor_node = ids[0]
        anchor_point = model.mesh.nodes[anchor_node]
        anchor_value = float(reference.fields_xy([[anchor_point.x, anchor_point.y]])[body][0])
        if anchor_value == 0.0:
            raise ValueError("radial trace anchor must have nonzero motion")
        anchors.append((anchor_node, anchor_value))
        anchor_dof = model.mesh.dof_manager.get_node_dofs(anchor_node)[2]
        for local, node in enumerate(ids):
            row, column = divmod(local, cells+1)
            if row in (0, cells) or column in (0, cells):
                point = model.mesh.nodes[node]
                value = float(reference.fields_xy([[point.x, point.y]])[body][0])
                if node != anchor_node:
                    dependent = model.mesh.dof_manager.get_node_dofs(node)[2]
                    model.add_constraint_equation(
                        terms=((dependent, 1.), (anchor_dof, -value/anchor_value)),
                        source_id=f"radial_trace_{node}")
    weights = np.ones((13, 13)); weights[[0, -1], :] *= .5; weights[:, [0, -1]] *= .5
    program = ContactPathProgram(
        controls=tuple(ContactPathControl.node(node, "uz") for node, _ in anchors),
        waypoints=((0., 0.), tuple(value for _, value in anchors)), steps_per_segment=2,
        nodal_forces=tuple(ContactPathNodalForce(node, "uz", (0., -reference.load_pa*weight/144))
            for node, weight in zip(source_ids, weights.ravel(), strict=True)))
    settings = replace(settings, enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=reference.load_pa, augmentation_acceleration="none",
        augmentation_schur_maximum_modes=None,
        augmentation_schur_maximum_pressure_pa=None,
        augmentation_schur_maximum_step_pa=None,
        augmentation_schur_difference_step_m=None,
        augmentation_schur_maximum_attempts=1,
        augmentation_schur_maximum_relinearizations=0,
        d1_interior_profile=True, d1_interior_coverage_tolerance_m2=1e-10,
        d1_interior_boundary_force_tolerance_n=1e-10,
        q4_source_gap_certificate="overlap_local_dev")
    return model, definition, settings, program, reference, dict(
        source_guided_shear_n_m=ds, target_guided_shear_n_m=dt,
        target_phi_squared_range=[min(phi_values), max(phi_values)],
        target_p1_energy_max_relative_error=max(scalar_errors),
        target_cells=target_cells,
        target_y0_m=target_y0_m,
        target_right_x_m=target_right_x_m,
        gap_m=reference.gap_m, exact_force_n=reference.force_n)


def radial_structure_check(output: Path, linear_record: Path):
    """Check the actual nonlinear structural owner against the guided operator."""
    if output.exists():
        raise FileExistsError(output)
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.nonlinear_static import (_prepare_initial_states,
        _prepare_qualified_q4_states_for_nonlinear_solve,
        _activate_nonlinear_state_storage, _assemble_nonlinear_system,
        _discard_nonlinear_state_candidate)
    saved = json.loads(linear_record.read_text())
    model, _definition, _settings, _program, _reference, record = build_radial_guided_case(
        target_cells=saved["target_cells"],
        target_y0_m=saved.get("target_y0_m", -0.113),
        target_right_x_m=saved.get("target_right_x_m", 1.3))
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    zero = np.zeros(model.mesh.dof_manager.total_dofs)
    info = {}
    states, _ = _prepare_initial_states(model, None, None, 5)
    states = _prepare_qualified_q4_states_for_nonlinear_solve(
        model, zero, states, 5, info, supplied_element_ids=(), ordinary_restart=False)
    states = _activate_nonlinear_state_storage(model, states, 5, info,
        kinematics="von_karman", committed_displacements=zero)
    internal, actual, _ = _assemble_nonlinear_system(model, zero, states, 5)
    normal = [model.mesh.dof_manager.get_node_dofs(node)[2] for node in model.mesh.nodes]
    expected = stiffness[normal][:, normal].toarray()
    actual_normal = actual[normal][:, normal].toarray()
    record["zero_structural_normal_tangent_relative_error"] = float(
        np.linalg.norm(actual_normal-expected)/np.linalg.norm(expected))
    record["zero_internal_force_norm_n"] = float(np.linalg.norm(internal))
    _discard_nonlinear_state_candidate(states)
    pose = np.asarray(saved["steps"][0]["displacements"])
    internal, _, _ = _assemble_nonlinear_system(model, pose, states, 5, tangent=False)
    linear = stiffness @ pose
    record["first_step_normal_internal_relative_difference"] = float(
        np.linalg.norm(internal[normal]-linear[normal])/np.linalg.norm(linear[normal]))
    record["normal_operator_target_met"] = (
        record["zero_structural_normal_tangent_relative_error"] <= 1e-11)
    record.update(status="development_actual_structural_operator_check",
        input_sha256=hashlib.sha256(linear_record.read_bytes()).hexdigest(), num_layers=5)
    _discard_nonlinear_state_candidate(states)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


def radial_preflight(output: Path, *, target_cells=4, target_right_x_m=1.3):
    if output.exists():
        raise FileExistsError(output)
    model, definition, settings, program, reference, record = build_radial_guided_case(
        target_cells=target_cells, target_right_x_m=target_right_x_m)
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError(f"radial fixture rejected: {report.issues}")
    record.update(status="development_radial_preflight", anyfesolver_module=str(Path(anyfesolver.__file__).resolve()),
                  total_dofs=model.mesh.dof_manager.total_dofs,
                  source_faces=len(definition.surfaces[0].element_ids),
                  target_faces=len(definition.surfaces[1].element_ids),
                  physical_load_n=-sum(force.values_n[-1] for force in program.nodal_forces),
                  boundary_trace_scope="continuous analytic traces; nodal interpolation is spatial discretization")
    record["source_trace_error_bound"] = reference.rectangle_trace_error_bound(
        (.1, 1.1), (.1, 1.1), 12, 12, "source")
    record["target_trace_error_bound"] = reference.rectangle_trace_error_bound(
        (0., record["target_right_x_m"]), (record["target_y0_m"], 1.2),
        record["target_cells"], record["target_cells"], "target")
    output.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")


def radial_linear_discriminator(output: Path, *, target_cells=4,
                                target_right_x_m=1.3):
    """Separate a bad fixture/predictor from nonlinear globalization failure."""
    if output.exists():
        raise FileExistsError(output)
    from scipy import linalg, optimize
    from anyfesolver.assembly import build_constraint_transformation
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from docs.d1_review_support.d1_two_body_reference_oracle import independent_gap_work
    started = time.perf_counter()
    model, definition, settings, program, reference, record = build_radial_guided_case(
        target_cells=target_cells, target_right_x_m=target_right_x_m)
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    size = model.mesh.dof_manager.total_dofs
    _, _, transform, offset, _, _ = build_constraint_transformation(
        stiffness, np.zeros(size), model)
    if np.linalg.norm(offset) != 0:
        raise ValueError("discriminator requires homogeneous structural constraints")
    controls = np.zeros((2, size))
    for row, control in enumerate(program.controls):
        for node, name, coefficient in control.terms:
            if name != "uz":
                raise ValueError("expected guided normal controls")
            controls[row, model.mesh.dof_manager.get_node_dofs(node)[2]] = coefficient
    reduced_controls = np.asarray(controls @ transform)
    null = linalg.null_space(reduced_controls)
    free_map = np.asarray(transform @ null)
    k = free_map.T @ stiffness @ free_map
    linalg.cholesky(k)
    _ids, support, work, area = independent_gap_work(model, definition, 0.)
    gap_map = work/support[:, None]
    records = []
    for step in (1, 2):
        targets = np.asarray(program.target_at(step))
        prescribed = np.asarray(transform @ (reduced_controls.T @ np.linalg.solve(
            reduced_controls @ reduced_controls.T, targets)))
        external = np.zeros(size)
        for force, value in zip(program.nodal_forces, program.force_values_at(step), strict=True):
            external[model.mesh.dof_manager.get_node_dofs(force.node_id)[2]] += value
        base = prescribed + free_map @ np.linalg.solve(k, free_map.T @ (external-stiffness @ prescribed))
        response = free_map @ np.linalg.solve(k, (gap_map @ free_map).T)
        h = gap_map @ response
        g = reference.gap_m + gap_map @ base
        fscale, lscale = reference.load_pa, reference.gap_m
        matrix, vector = h*fscale/lscale, g/lscale
        solved = optimize.minimize(lambda y: (.5*y @ matrix @ y+vector @ y,
            matrix @ y+vector), np.zeros(len(support)), jac=True, method="L-BFGS-B",
            bounds=[(0., None)]*len(support), options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000})
        z = fscale*solved.x
        pose = base+response @ z
        gap = reference.gap_m+gap_map @ pose
        ncp = z-np.maximum(0., z-definition.interactions[0].normal_stiffness*support*gap)
        records.append(dict(step=step, optimizer_success=bool(solved.success),
            optimizer_message=str(solved.message), optimizer_iterations=solved.nit,
            contact_free_displacement_norm_m=float(np.linalg.norm(base)),
            displacement_norm_m=float(np.linalg.norm(pose)),
            max_absolute_displacement_m=float(np.max(abs(pose))),
            contact_force_n=float(np.sum(z)), active_modes=int(np.count_nonzero(z)),
            minimum_normalized_gap_m=float(np.min(gap)),
            complementarity_relative_force=float(np.sum(abs(ncp))/fscale),
            force_relative_error=(abs(np.sum(z)-reference.force_n)/reference.force_n if step == 2 else None),
            displacements=pose.tolist(), nodal_force_n=z.tolist()))
    record.update(status="development_independent_linear_discriminator", elapsed_seconds=time.perf_counter()-started,
        reference_scope="frozen parallel-skin linear finite-element KKT; not nonlinear/spatial acceptance",
        independent_overlap_area_m2=area, admissible_normal_dimensions=null.shape[1], steps=records)
    output.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"status": record["status"], "elapsed_seconds": record["elapsed_seconds"],
        "steps": [{k: v for k, v in item.items() if k not in ("displacements", "nodal_force_n")} for item in records]}))


def radial_swept_profile(output: Path, saved_path: Path):
    """One saved full-path certificate, with flushed scalar cost traces."""
    from functools import wraps
    from anyfesolver.general_contact import s3_patch_static as owner
    if output.exists():
        raise FileExistsError(output)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    model, definition, settings, _program, _reference, record = build_radial_guided_case(
        target_cells=saved["target_cells"], target_y0_m=saved["target_y0_m"],
        target_right_x_m=saved.get("target_right_x_m", 1.3))
    pose = np.asarray(saved["steps"][-1]["displacements"])
    counters, originals = {}, []
    started = time.perf_counter()
    print(json.dumps(dict(event="preparation_begin", imported_source=anyfesolver.__file__)), flush=True)
    manager = owner.GeneralS3PatchManager(model, definition, settings)
    print(json.dumps(dict(event="preparation_end", elapsed_seconds=time.perf_counter()-started)), flush=True)

    def instrument(obj, name):
        original = getattr(obj, name)
        counters[name] = dict(calls=0, seconds=0.)
        @wraps(original)
        def measured(*args, **kwargs):
            before = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                item = counters[name]
                item["calls"] += 1
                item["seconds"] += time.perf_counter()-before
                if item["calls"] == 1 or item["calls"] % 500 == 0:
                    print(json.dumps(dict(event=name, **item,
                        elapsed_seconds=time.perf_counter()-started)), flush=True)
        originals.append((obj, name, original))
        setattr(obj, name, measured)
    instrument(owner.GeneralS3PatchManager, "_gap_lower")
    instrument(owner, "swept_projected_coverage_admissible")
    instrument(owner, "overlap_local_swept_lower")
    try:
        admitted = manager._swept_admissible(np.zeros_like(pose), pose)
        record.update(status="development_saved_swept_profile", admissible=bool(admitted))
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
        record.update(elapsed_seconds=time.perf_counter()-started,
            coarse_inclusive_counters=counters,
            input_sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest(),
            scope="one zero-to-saved-linear-pose path; no solve or swept acceptance claim")
        output.write_text(json.dumps(record, indent=2), encoding="utf-8")


def radial_candidate_profile(output: Path, saved_path: Path):
    """One saved-pose residual evaluation; no advancement or mesh sweep."""
    import cProfile
    import pstats
    from functools import wraps
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.semismooth import linearize
    from anyfesolver.general_contact import s3_patch_augmented as augmented
    from anyfesolver.general_contact.s3_patch_mortar_energy import MovingS3PatchPrimalMortarPenalty
    from anyfesolver.general_contact.q4_source_mortar_energy import MovingQ4PrimalMortarPenalty
    if output.exists() or output.with_suffix(".pstats").exists():
        raise FileExistsError(output)
    saved = json.loads(saved_path.read_text())
    model, definition, settings, _program, _reference, record = build_radial_guided_case(
        target_cells=saved["target_cells"], target_y0_m=saved["target_y0_m"],
        target_right_x_m=saved.get("target_right_x_m", 1.3))
    pose = np.asarray(saved["steps"][-1]["displacements"])
    forces = np.asarray(saved["steps"][-1]["nodal_force_n"])
    counters = {}
    originals = []
    def instrument(owner, name, label):
        original = getattr(owner, name)
        originals.append((owner, name, original))
        @wraps(original)
        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                counter = counters.setdefault(label, dict(calls=0, seconds=0.))
                counter["calls"] += 1
                counter["seconds"] += time.perf_counter()-started
        setattr(owner, name, measured)
    instrument(MovingS3PatchPrimalMortarPenalty, "evaluate", "float_geometry")
    instrument(MovingQ4PrimalMortarPenalty, "__init__", "pair_preparation")
    instrument(augmented, "q4_s3_overlap_derivatives", "pair_jet2")
    instrument(augmented, "q4_s3_overlap_first_derivatives", "pair_gradient_only")
    instrument(augmented.MovingS3PatchProjectedNormal, "_jet2_s3_work", "work_assembly_inclusive")
    profiler, started = cProfile.Profile(), time.perf_counter()
    try:
        profiler.enable()
        preparation_started = time.perf_counter()
        manager = GeneralS3PatchManager(model, definition, settings)
        record["manager_preparation_seconds"] = time.perf_counter()-preparation_started
        result = linearize(manager, pose, forces, tangent=False)
        record.update(status="development_saved_candidate_profile", residual_norm_n=float(np.linalg.norm(result.force)),
                      energy_evaluations=result.evaluations,
                      gradient_only_pairs=result.gradient_only_pairs,
                      residual=result.force.tolist(), gap_gradient=result.gap_gradient.tolist())
    except Exception as exc:
        record.update(status="candidate_exception", failure_reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        profiler.disable()
        elapsed = time.perf_counter()-started
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
        profiler.dump_stats(str(output.with_suffix(".pstats")))
        with output.with_suffix(".profile.txt").open("x", encoding="utf-8") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(40)
        record.update(elapsed_seconds=elapsed, coarse_inclusive_counters=counters,
            input_sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest(),
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            changed_kernel_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(anyfesolver.__file__).parent / "general_contact").glob("*.py")},
            scope="one profiled residual at saved independent linear pose; no route speed claim")
        output.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")


def radial_wrench_check(output: Path, saved_path: Path):
    """Recover complete guided reactions at saved elastic poses, without solve."""
    if output.exists():
        raise FileExistsError(output)
    from scipy import linalg
    from anyfesolver.assembly import build_constraint_transformation
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.nonlinear_static import (_prepare_initial_states,
        _prepare_qualified_q4_states_for_nonlinear_solve,
        _activate_nonlinear_state_storage, _assemble_nonlinear_system,
        _discard_nonlinear_state_candidate)
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.semismooth import linearize
    from docs.d1_review_support.d1_guided_reaction_wrench import guided_wrench_diagnostics
    saved = json.loads(saved_path.read_text())
    poses, history = saved.get("accepted_pose_displacements", ()), saved.get("history", ())
    if saved["status"] != "completed" or len(poses) != len(history) or len(poses) != 2:
        raise ValueError("wrench recovery requires the completed two-step radial fixture")
    model, definition, settings, program, _reference, record = build_radial_guided_case(
        target_cells=saved["target_cells"], target_y0_m=saved["target_y0_m"],
        target_right_x_m=saved.get("target_right_x_m", 1.3))
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    size = model.mesh.dof_manager.total_dofs
    _, _, transform, offset, _, _ = build_constraint_transformation(
        stiffness, np.zeros(size), model)
    if np.any(offset != 0.):
        raise ValueError("radial fixture requires homogeneous structural constraints")
    controls = np.zeros((len(program.controls), size))
    for row, control in enumerate(program.controls):
        for node, name, coefficient in control.terms:
            if name != "uz":
                raise ValueError("guided radial path requires vertical controls")
            controls[row, model.mesh.dof_manager.get_node_dofs(node)[2]] += coefficient
    free_map = np.asarray(transform @ linalg.null_space(np.asarray(controls @ transform)))
    if free_map.shape[1] != 121+(saved["target_cells"]-1)**2:
        raise ValueError("guided joint free span has an unexpected nullity")
    zero, info = np.zeros(size), {}
    states, _ = _prepare_initial_states(model, None, None, 5)
    states = _prepare_qualified_q4_states_for_nonlinear_solve(
        model, zero, states, 5, info, supplied_element_ids=(), ordinary_restart=False)
    states = _activate_nonlinear_state_storage(model, states, 5, info,
        kinematics="von_karman", committed_displacements=zero)
    manager = GeneralS3PatchManager(model, definition, settings)
    origin = np.array((.6, .6, 0.))
    records = []
    for pose, row in zip(poses, history, strict=True):
        u, certificate = np.asarray(pose), row["contact"]
        internal, _, _ = _assemble_nonlinear_system(model, u, states, 5,
            tangent=False, kinematics="von_karman", require_full_coordinates=True)
        _discard_nonlinear_state_candidate(states)
        z = (np.asarray(certificate["numerical_pressure_pa"])
             * np.asarray(certificate["reference_support_m2"]))
        if np.any(z != 0.):
            contact = linearize(manager, u, z, tangent=False).force
        else:
            contact = np.zeros(size)
        external = np.zeros(size)
        for force, value in zip(program.nodal_forces,
                                program.force_values_at(row["step_index"]), strict=True):
            external[model.mesh.dof_manager.get_node_dofs(force.node_id)[2]] += value
        physical_force = np.asarray(certificate["integration_force"])+certificate["target_force"]
        physical_moment = (np.asarray(certificate["integration_moment"])
                           +certificate["target_moment"]-np.cross(origin, physical_force))
        bodies = {}
        for name, surface, prefix in zip(('source', 'target'), definition.surfaces,
                                        ('integration', 'target'), strict=True):
            force = np.asarray(certificate[prefix+'_force'])
            bodies[name] = dict(node_ids=sorted({node
                for element in surface.element_ids
                for node in model.mesh.elements[element].node_ids}),
                physical_contact_force=force,
                physical_contact_moment=np.asarray(certificate[prefix+'_moment'])
                                        -np.cross(origin, force))
        diagnostics = guided_wrench_diagnostics(model, u, internal, contact, external,
            free_map, physical_contact_force=physical_force,
            physical_contact_moment=physical_moment, origin=origin, bodies=bodies)
        records.append(dict(step=row["step_index"], **diagnostics))
    record.update(status="development_saved_guided_wrench_check", rows=records,
        input_sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest(),
        joint_free_dimension=free_map.shape[1],
        development_checks_met=all(max(r["constraint_reconstruction_relative_force"],
            r["balance_relative_force"], r["balance_relative_moment"],
            r["contact_conversion_max_scaled"], r["structural_internal_max_scaled"]) <= 1e-6
            and all(max(body['balance_relative_force'], body['balance_relative_moment'],
                body['contact_conversion_max_scaled'], body['structural_internal_max_scaled'])
                <= 1e-6 for body in r['bodies'].values()) for r in records))
    output.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("status", "development_checks_met", "joint_free_dimension")}))


def radial_source_bin_check(output: Path, saved_path: Path):
    """Assess one completed saved pose; do not repeat its coupled solve."""
    if output.exists():
        raise FileExistsError(output)
    from dataclasses import asdict
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from docs.d1_review_support.d1_radial_source_bin_work import check_guided_source_bins
    from docs.d1_review_support.d1_radial_reference_observables import uniform_disk_pressure_observables
    from docs.d1_review_support.d1_radial_bin_assessment import assess_physical_source_bins
    started = time.perf_counter()
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    if (saved.get("status") != "completed" or saved.get("accepted_steps") != 2
            or len(saved.get("accepted_pose_displacements", ())) != 2):
        raise ValueError("physical bins require the completed two-step radial fixture")
    for name, expected in saved["changed_kernel_sha256"].items():
        path = Path(anyfesolver.__file__).parent / "general_contact" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"saved-pose contact source changed: {name}")
    model, definition, settings, _program, reference, record = build_radial_guided_case(
        target_cells=saved["target_cells"], target_y0_m=saved["target_y0_m"],
        target_right_x_m=saved["target_right_x_m"])
    certificate = saved["history"][-1]["contact"]
    z = dict(zip(certificate["source_pressure_node_ids"],
        np.asarray(certificate["numerical_pressure_pa"])
        * np.asarray(certificate["reference_support_m2"]), strict=True))
    record.update(status="development_saved_physical_source_bins",
        input_sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest(),
        anyfesolver_module=str(Path(anyfesolver.__file__).resolve()),
        quadrature_tolerance_n=reference.force_n*1e-8, maximum_triangles=14144)
    try:
        manager = GeneralS3PatchManager(model, definition, settings)
        evidence = check_guided_source_bins(model, definition,
            np.asarray(saved["accepted_pose_displacements"][-1]), z,
            quadrature_tolerance_n=record["quadrature_tolerance_n"],
            maximum_triangles=record["maximum_triangles"],
            production_geometry=manager.law.geometry)
        exact = uniform_disk_pressure_observables(reference.pressure_pa,
            material_force_threshold_n=8e-8)
        exact_bins = [[item.force_n for item in row] for row in exact.bins]
        record.update(evidence=evidence, exact_disk_observables=asdict(exact),
            assessment=assess_physical_source_bins(evidence, exact_bins,
                certificate["integration_force"][2], exact_bin_error_n=[
                    [item.estimated_absolute_error[0] for item in row] for row in exact.bins]))
    except (ValueError, MemoryError) as exc:
        record.update(status="development_saved_physical_source_bins_failure",
            failure_reason=f"{type(exc).__name__}: {exc}")
    record["elapsed_seconds"] = time.perf_counter()-started
    output.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in
        ("status", "elapsed_seconds", "failure_reason", "assessment")}))


def radial_spatial_assess(output: Path, saved_path: Path, bin_path: Path,
                          wrench_path: Path, linear_path: Path):
    """Aggregate existing evidence without another solve or reassembly."""
    if output.exists():
        raise FileExistsError(output)
    from docs.d1_review_support.d1_radial_spatial_assessment import assess_saved_radial_spatial
    paths = (saved_path, bin_path, wrench_path, linear_path)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assessment = assess_saved_radial_spatial(*records,
        input_sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest())
    assessment["evidence_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in paths}
    assessment.update(independently_accepted=False, released=False)
    output.write_text(json.dumps(assessment, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({key: assessment.get(key) for key in
        ("status", "development_checks_met", "remaining_gates", "missing_or_invalid_evidence")}))


def radial_solve(output: Path, *, target_cells=4, target_right_x_m=1.3):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    model, definition, settings, program, reference, record = build_radial_guided_case(
        target_cells=target_cells, target_right_x_m=target_right_x_m)
    record.update(anyfesolver_module=str(Path(anyfesolver.__file__).resolve()),
        changed_kernel_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(anyfesolver.__file__).parent / "general_contact").glob("*.py")},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        from docs.d1_review_support.d1_radial_scalar_trace import radial_solver_trace
        trace_path = output.with_suffix('.trace.jsonl')
        record['scalar_trace'] = str(trace_path)
        with radial_solver_trace(trace_path):
            result = solve_static_nonlinear(model, contact=definition, contact_settings=settings,
                contact_path_program=program, max_iterations=18, emit_restart_checkpoint=True,
                record_increment_snapshots=True)
    except Exception as exc:
        record.update(status="solver_exception", elapsed_seconds=time.perf_counter()-started,
            failure_reason=f"{type(exc).__name__}: {exc}", restart_checkpoint=None)
        output.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")
        raise
    record.update(status=result.status, failure_reason=result.failure_reason,
        accepted_steps=len(result.steps), elapsed_seconds=time.perf_counter()-started,
        contact_failure=result.info.get("contact_failure"),
        history=result.info.get("force_displacement_history", ()),
        displacements=result.displacements.tolist(),
        accepted_pose_displacements=[snapshot.displacements.tolist()
                                    for snapshot in result.snapshots],
        restart_checkpoint=result.restart_checkpoint,
        analysis_scope="flat elastic guided fixed-MIN3 operator; development only")
    # Preserve the mechanical outcome before potentially failing diagnostics.
    # A missing diagnostic is not a successful reference comparison.
    output.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")
    if result.status == "completed":
        from dataclasses import asdict
        from docs.d1_review_support.d1_radial_reference_observables import (
            q1_reference_pressure_observables, uniform_disk_pressure_observables,
            DIAGNOSTIC_BIN_EDGES)
        from examples.general_contact_d1_interior_controls_dev import _independent_projected_gap_bound
        final = record["history"][-1]["contact"]
        record["independent_gap_lower_m"] = _independent_projected_gap_bound(
            model, definition, result.displacements, definition.interactions[0].gap_tolerance)
        record["force_relative_error"] = abs(final["integration_force"][2]-reference.force_n)/reference.force_n
        pressure_by_node = dict(zip(final["source_pressure_node_ids"],
                                    final["numerical_pressure_pa"], strict=True))
        pressure = np.array([pressure_by_node[node] for node in range(1, 170)]).reshape(13, 13)
        coordinates = np.linspace(.1, 1.1, 13)
        numerical = q1_reference_pressure_observables(coordinates, coordinates, pressure,
            material_force_threshold_n=8e-8)
        exact_bins = uniform_disk_pressure_observables(reference.pressure_pa,
            material_force_threshold_n=8e-8)
        numerical_bins = np.array([[item.force_n for item in row] for row in numerical.bins])
        reference_bins = np.array([[item.force_n for item in row] for row in exact_bins.bins])
        physical_force = np.asarray(final["integration_force"])
        physical_moment = np.asarray(final["integration_moment"])-np.cross(
            np.array((.6, .6, 0.)), physical_force)
        record.update(numerical_reference_pressure_observables=asdict(numerical),
            exact_disk_observables=asdict(exact_bins),
            reference_pressure_bin_force_l1_error=float(np.sum(abs(numerical_bins-reference_bins))/np.sum(abs(reference_bins))),
            reference_pressure_bin_force_l2_error=float(np.linalg.norm(numerical_bins-reference_bins)/np.linalg.norm(reference_bins)),
            reference_pressure_to_physical_force_relative_difference=float(
                abs(numerical.total.force_n-physical_force[2])/reference.force_n),
            reference_pressure_to_physical_moment_scaled_difference=float(
                np.linalg.norm(np.asarray(numerical.total.moment_nm)-physical_moment)/reference.force_n),
            physical_contact_moment_about_disk_center_nm=physical_moment.tolist(),
            numerical_reference_pressure_diagnostic_4x4=asdict(q1_reference_pressure_observables(
                coordinates, coordinates, pressure, x_bin_edges=DIAGNOSTIC_BIN_EDGES,
                y_bin_edges=DIAGNOSTIC_BIN_EDGES, material_force_threshold_n=8e-8)),
            pressure_bin_scope="numerical reference-area pressure; physical local-bin conversion remains to assess")
        u = np.array([result.displacements[model.mesh.dof_manager.get_node_dofs(node)[2]]
                      for node in range(1, 170)]).reshape(13, 13)
        samples = []
        for x, y in ((.6, .6), (.6, .85), (.6, 1.0)):
            sx, sy = (x-.1)*12, (y-.1)*12
            column, row = min(int(sx), 11), min(int(sy), 11)
            ax, ay = sx-column, sy-row
            actual = ((1-ax)*(1-ay)*u[row, column]+ax*(1-ay)*u[row, column+1]
                      +ax*ay*u[row+1, column+1]+(1-ax)*ay*u[row+1, column])
            exact = float(reference.fields_xy([[x, y]])["source_w"][0])
            samples.append(dict(x_m=x, y_m=y, actual_uz_m=float(actual), exact_uz_m=exact,
                                relative_error=abs(actual-exact)/abs(exact)))
        record["source_displacement_samples"] = samples
        target_samples = []
        for x, y in ((.6, .6), (.6, .85), (.6, 1.0)):
            for element_id in definition.surfaces[1].element_ids:
                element = model.mesh.elements[element_id]
                xy = np.array([(model.mesh.nodes[node].x, model.mesh.nodes[node].y)
                               for node in element.node_ids])
                shape = np.array((1., x, y)) @ np.linalg.inv(np.column_stack((np.ones(3), xy)))
                if np.min(shape) < -1e-12:
                    continue
                actual = sum(weight*result.displacements[model.mesh.dof_manager.get_node_dofs(node)[2]]
                             for weight, node in zip(shape, element.node_ids, strict=True))
                exact = float(reference.fields_xy([[x, y]])["target_w"][0])
                target_samples.append(dict(x_m=x, y_m=y, actual_uz_m=float(actual), exact_uz_m=exact,
                    relative_error=abs(actual-exact)/abs(exact)))
                break
            else:
                raise ValueError("target response sample is outside represented S3 faces")
        record["target_displacement_samples"] = target_samples
    output.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in (
        "status", "failure_reason", "accepted_steps", "elapsed_seconds", "force_relative_error")}))


def replay_captured_sweep(capture: Path, output: Path):
    """Check a saved rejected proposal without advancing the nonlinear solve."""
    if output.exists():
        raise FileExistsError(output)
    saved = json.loads(capture.read_text(encoding="utf-8"))
    if saved.get("swept_failure_discriminator", {}).get(
            "zero_fraction_proposals") != 1:
        raise ValueError("expected one captured zero-fraction proposal")
    model, definition, settings, _ = build_interior_press(
        force_n=saved["force_n"], reference_slope=saved["reference_slope"],
        target_columns=saved.get(
            "target_columns", saved["target_s3_faces"] //
            (2*saved["target_rows"])),
        target_rows=saved["target_rows"],
        target_young_multiplier=saved.get("target_young_multiplier", 1.0),
        target_y0_m=saved.get("target_y0_m", -0.1),
        source_reference_downshift_m=saved.get(
            "source_reference_downshift_m", 0.0),
        source_support=saved["source_support"],
        source_columns=saved["source_columns"],
        source_rows=saved["source_rows"],
        source_y_profile=saved["source_y_profile"],
        offline_structural_reference_y16=(
            saved["offline_structural_reference_y16"]),
        load_shape=saved["load_shape"],
        slide_intervals=saved["slide_intervals"],
        slide_body=saved["slide_body"],
        target_reference_full_left_clamp=(
            saved["target_reference_full_left_clamp"]),
        target_left_edge_motion=saved.get("target_left_edge_motion", False),
        shared_edge_tie=saved["shared_edge_tie"],
    )
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from examples.general_contact_d1_interior_controls_dev import (
        _independent_projected_gap_bound,
    )

    manager = GeneralS3PatchManager(
        model, definition,
        replace(settings, q4_source_gap_certificate="overlap_local_dev"),
    )
    captured = saved["swept_failure_discriminator"]
    start = np.asarray(captured["captured_start_displacement"], dtype=float)
    proposed = np.asarray(captured["captured_proposed_displacement"], dtype=float)
    if start.size != manager.n_dof or proposed.size != manager.n_dof:
        raise ValueError("saved proposal does not fit reconstructed fixture")
    trials = []
    for fraction in (1/8, 1/4, 1/2, 1.0):
        endpoint = start+fraction*(proposed-start)
        source, target = manager._positions(endpoint)
        try:
            manager.require_d1_coverage(source, target)
            coverage_issue = None
        except ValueError as error:
            coverage_issue = str(error)
        try:
            independent_lower = _independent_projected_gap_bound(
                model, definition, endpoint,
                definition.interactions[0].gap_tolerance)
            independent_issue = None
        except RuntimeError as error:
            independent_lower = None
            independent_issue = str(error)
        trials.append({
            "fraction": fraction,
            "overlap_local_swept_admissible": manager._swept_admissible(
                start, endpoint),
            "overlap_local_endpoint_lower_gap_m": manager._gap_lower(
                (source, target)),
            "independent_endpoint_lower_gap_m": independent_lower,
            "independent_endpoint_issue": independent_issue,
            "coverage_issue": coverage_issue,
        })
    payload = {
        "capture": str(capture),
        "certificate": "overlap_local_dev",
        "gap_tolerance_m": manager.gap_tolerance,
        "trials": trials,
    }
    output.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n",
                      encoding="utf-8")
    print(json.dumps(payload, default=_json_default))


def saved_rotation_probe(saved_path: Path, output: Path) -> None:
    """Read-only contact-work probe at a saved accepted development pose."""
    if output.exists():
        raise FileExistsError(output)
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager

    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    if (saved["status"] != "completed" or saved["accepted_steps"] != 2
            or saved["source_support"] != "perimeter"
            or saved["q4_gap_certificate"] != "overlap_local_dev"):
        raise ValueError("rotation probe requires an accepted interior pose")
    columns, rows = saved["source_columns"], saved["source_rows"]
    model, definition, settings, _ = build_interior_press(
        force_n=saved["force_n"], reference_slope=saved["reference_slope"],
        source_support=saved["source_support"],
        source_columns=columns, source_rows=rows,
        source_y_profile=saved["source_y_profile"],
        target_columns=saved["target_columns"],
        target_rows=saved["target_rows"],
        target_young_multiplier=saved.get("target_young_multiplier", 1.0),
        target_y0_m=saved.get("target_y0_m", -0.1),
        source_reference_downshift_m=saved.get(
            "source_reference_downshift_m", 0.0),
        target_reference_full_left_clamp=(
            saved["target_reference_full_left_clamp"]),
        target_left_edge_motion=saved.get("target_left_edge_motion", False),
        load_shape=saved["load_shape"], slide_body=saved["slide_body"],
    )
    manager = GeneralS3PatchManager(
        model, definition,
        replace(settings, q4_source_gap_certificate="overlap_local_dev"),
    )
    manager.set_normal_multipliers(
        saved["history"][-1]["contact"]["normal_multiplier_estimate_pa"])
    displacement = np.asarray(saved["displacements"], dtype=float)
    free_nodes = tuple(1+row*(columns+1)+column
                       for row in range(1, rows)
                       for column in range(1, columns))
    directions = tuple((node, axis, model.mesh.dof_manager.get_node_dofs(node)[axis])
                       for node in free_nodes for axis in (3, 4))
    force, tangent, center = manager.evaluate(displacement, tangent=True)
    selected = int(np.argmax(np.abs(np.asarray(force)[
        [entry[2] for entry in directions]])))
    node, axis, direction = directions[selected]
    h = 1.0e-5
    plus_displacement = displacement.copy()
    plus_displacement[direction] += h
    plus, _, plus_state = manager.evaluate(plus_displacement, tangent=False)
    minus_displacement = displacement.copy()
    minus_displacement[direction] -= h
    minus, _, minus_state = manager.evaluate(minus_displacement, tangent=False)
    difference = (np.asarray(plus)-np.asarray(minus))/(2*h)
    derived = np.asarray(tangent[:, direction].toarray()).reshape(-1)
    branch_stable = (center.active_pressure_dofs ==
                     plus_state.active_pressure_dofs ==
                     minus_state.active_pressure_dofs and
                     center.integration_triangles ==
                     plus_state.integration_triangles ==
                     minus_state.integration_triangles)
    record = {
        "saved_pose": str(saved_path),
        "selected_source_node": node,
        "selected_rotation_axis": "rx" if axis == 3 else "ry",
        "selected_rotation_dof": direction,
        "source_free_rotation_count": len(directions),
        "maximum_free_rx_contact_moment_nm": float(np.max(np.abs(
            np.asarray(force)[[entry[2] for entry in directions
                               if entry[1] == 3]]))),
        "maximum_free_ry_contact_moment_nm": float(np.max(np.abs(
            np.asarray(force)[[entry[2] for entry in directions
                               if entry[1] == 4]]))),
        "selected_contact_moment_nm": float(force[direction]),
        "contact_directional_response_norm": float(np.linalg.norm(difference)),
        "relative_tangent_direction_error": float(
            np.linalg.norm(derived-difference)/
            max(np.linalg.norm(difference), 1.0e-30)),
        "branch_stable": branch_stable,
        "active_pressure_modes": center.active_pressure_dofs,
        "integration_triangles": center.integration_triangles,
        "perturbation_rad": h,
    }
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--force-n", type=float, default=0.08)
    parser.add_argument("--reference-slope", type=float, default=0.01)
    parser.add_argument("--target-columns", type=int, choices=(1, 2, 3, 4, 6),
                        default=1)
    parser.add_argument("--target-rows", type=int, choices=(1, 2, 3, 4, 6),
                        default=1)
    parser.add_argument("--target-young-multiplier", type=float,
                        choices=(1.0, 5.0), default=1.0)
    parser.add_argument("--target-y0-m", type=float,
                        choices=(-0.1, -0.113), default=-0.1)
    parser.add_argument("--source-reference-downshift-m", type=float,
                        choices=(0.0, 35.0e-6), default=0.0)
    parser.add_argument("--source-support", choices=("left", "perimeter"),
                        default="left")
    parser.add_argument("--source-rows", type=int, choices=(2, 4, 6, 8, 12, 16), default=2)
    parser.add_argument("--source-columns", type=int, choices=(4, 6, 8, 12), default=4)
    parser.add_argument("--source-y-profile", choices=("uniform_ref16",),
                        default=None)
    parser.add_argument("--offline-structural-reference-y16",
                        action="store_true")
    parser.add_argument("--load-shape", choices=("point", "interior_distributed",
                                                 "interior_uniform", "physical_bump"),
                        default="point")
    parser.add_argument("--max-path-fraction", type=float,
                        default=1.0)
    parser.add_argument("--four-phase", action="store_true")
    parser.add_argument("--slide-distance-m", type=float, default=0.05)
    parser.add_argument("--slide-intervals", type=int, default=1)
    parser.add_argument("--slide-body", choices=("source", "target",
                                                 "target_diagonal_dev"),
                        default="source")
    parser.add_argument("--shared-edge-tie", action="store_true")
    parser.add_argument("--target-reference-full-left-clamp",
                        action="store_true")
    parser.add_argument("--target-left-edge-motion", action="store_true")
    parser.add_argument("--capture-swept-failure", action="store_true")
    parser.add_argument("--record-swept-fractions", action="store_true")
    parser.add_argument("--record-accepted-poses", action="store_true")
    parser.add_argument("--restart-checkpoint", type=Path)
    parser.add_argument("--q4-gap-certificate",
                        choices=("projected_matching", "overlap_local_dev"))
    parser.add_argument("--replay-captured-sweep", type=Path)
    parser.add_argument("--saved-rotation-probe", type=Path)
    parser.add_argument("--verify-saved-cycle", type=Path)
    parser.add_argument("--compare-saved-cycle", type=Path)
    parser.add_argument("--cycle-reference", type=Path)
    parser.add_argument("--radial-preflight", action="store_true")
    parser.add_argument("--radial-solve", action="store_true")
    parser.add_argument("--radial-linear-discriminator", action="store_true")
    parser.add_argument("--radial-target-cells", type=int, choices=(4, 12), default=4)
    parser.add_argument("--radial-target-right-x-m", type=float,
                        choices=(1.3, 1.297), default=1.3)
    parser.add_argument("--radial-structure-check", type=Path)
    parser.add_argument("--radial-wrench-check", type=Path)
    parser.add_argument("--radial-source-bin-check", type=Path)
    parser.add_argument("--radial-spatial-assess", type=Path)
    parser.add_argument("--radial-bin-evidence", type=Path)
    parser.add_argument("--radial-wrench-evidence", type=Path)
    parser.add_argument("--radial-linear-assessment", type=Path)
    parser.add_argument("--radial-candidate-profile", type=Path)
    parser.add_argument("--radial-swept-profile", type=Path)
    parser.add_argument("--steps-per-segment", type=int, choices=(1, 2),
                        default=2)
    arguments = parser.parse_args()
    if arguments.radial_spatial_assess is not None:
        if any(p is None for p in (arguments.radial_bin_evidence,
                arguments.radial_wrench_evidence, arguments.radial_linear_assessment)):
            parser.error("radial spatial assessment requires bin, wrench and linear assessment evidence")
        radial_spatial_assess(arguments.output, arguments.radial_spatial_assess,
            arguments.radial_bin_evidence, arguments.radial_wrench_evidence,
            arguments.radial_linear_assessment)
        raise SystemExit(0)
    if arguments.radial_source_bin_check is not None:
        radial_source_bin_check(arguments.output, arguments.radial_source_bin_check)
        raise SystemExit(0)
    if arguments.radial_swept_profile is not None:
        radial_swept_profile(arguments.output, arguments.radial_swept_profile)
        raise SystemExit(0)
    if arguments.radial_candidate_profile is not None:
        radial_candidate_profile(arguments.output, arguments.radial_candidate_profile)
        raise SystemExit(0)
    if arguments.radial_wrench_check is not None:
        radial_wrench_check(arguments.output, arguments.radial_wrench_check)
        raise SystemExit(0)
    if arguments.radial_structure_check is not None:
        radial_structure_check(arguments.output, arguments.radial_structure_check)
        raise SystemExit(0)
    if arguments.radial_linear_discriminator:
        radial_linear_discriminator(arguments.output, target_cells=arguments.radial_target_cells,
                                    target_right_x_m=arguments.radial_target_right_x_m)
        raise SystemExit(0)
    if arguments.radial_solve:
        radial_solve(arguments.output, target_cells=arguments.radial_target_cells,
                     target_right_x_m=arguments.radial_target_right_x_m)
        raise SystemExit(0)
    if arguments.radial_preflight:
        radial_preflight(arguments.output, target_cells=arguments.radial_target_cells,
                         target_right_x_m=arguments.radial_target_right_x_m)
        raise SystemExit(0)
    if arguments.compare_saved_cycle is not None:
        if arguments.cycle_reference is None:
            parser.error("cycle comparison requires --cycle-reference")
        compare_saved_cycles(arguments.cycle_reference, arguments.compare_saved_cycle,
                             arguments.output)
        raise SystemExit(0)
    if arguments.verify_saved_cycle is not None:
        verify_saved_cycle(arguments.verify_saved_cycle, arguments.output)
        raise SystemExit(0)
    if arguments.replay_captured_sweep is not None:
        replay_captured_sweep(arguments.replay_captured_sweep,
                              arguments.output)
        raise SystemExit(0)
    if arguments.saved_rotation_probe is not None:
        saved_rotation_probe(arguments.saved_rotation_probe,
                             arguments.output)
        raise SystemExit(0)
    main(arguments.output, force_n=arguments.force_n,
         reference_slope=arguments.reference_slope,
         target_columns=arguments.target_columns,
         target_rows=arguments.target_rows,
         target_young_multiplier=arguments.target_young_multiplier,
         target_y0_m=arguments.target_y0_m,
         source_reference_downshift_m=(
             arguments.source_reference_downshift_m),
         source_support=arguments.source_support,
         source_columns=arguments.source_columns,
         source_rows=arguments.source_rows, load_shape=arguments.load_shape,
         source_y_profile=arguments.source_y_profile,
         offline_structural_reference_y16=(
             arguments.offline_structural_reference_y16),
         max_path_fraction=arguments.max_path_fraction,
         four_phase=arguments.four_phase,
         steps_per_segment=arguments.steps_per_segment,
         slide_distance_m=arguments.slide_distance_m,
         slide_intervals=arguments.slide_intervals,
         slide_body=arguments.slide_body,
         target_reference_full_left_clamp=(
             arguments.target_reference_full_left_clamp),
         target_left_edge_motion=arguments.target_left_edge_motion,
         shared_edge_tie=arguments.shared_edge_tie,
         capture_swept_failure=arguments.capture_swept_failure,
         record_swept_fractions=arguments.record_swept_fractions,
         q4_gap_certificate=arguments.q4_gap_certificate,
         record_accepted_poses=arguments.record_accepted_poses,
         restart_checkpoint_path=arguments.restart_checkpoint)
