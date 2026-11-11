"""Bounded interior full-cover shell contact against a 1D exact obstacle."""

from __future__ import annotations

from dataclasses import replace
import json
import sys
import time
from pathlib import Path

import numpy as np

from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathNodalForce,
    ContactPathProgram, solve_static_nonlinear,
)
from anyfesolver.general_contact import validate_general_contact
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main(columns: int, output: Path, *, shared_edge_tie: bool = False,
         shifted_target: bool = False):
    if columns not in (16, 32):
        raise ValueError("analytic strip preflight admits 16 or 32 cells")
    if output.exists():
        raise FileExistsError(output)
    thickness = 0.05
    young = 1e5
    poisson = 0.3
    shear = (5/6)*thickness*young/(2*(1+poisson))
    opening = 40e-6
    front = 0.25
    load = 2*shear*opening/front**2
    expected_reaction = load*(1-2*front)
    model, definition, settings, program = build_case(
        columns, load_profile="distributed", total_force_n=load,
        target_right_x_m=1.3, full_width_target=True,
        full_width_target_y0_m=-0.113 if shifted_target else -0.1,
        source_rows=1,
    )
    stride = columns+1
    model.add_boundary_condition(BoundaryCondition(
        "source_right_support", (stride, 2*stride), {"uz": 0.0},
    ))
    settings = replace(
        settings, enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=load,
        augmentation_acceleration="none",
        augmentation_schur_maximum_modes=None,
        augmentation_schur_maximum_pressure_pa=None,
        augmentation_schur_maximum_step_pa=None,
        augmentation_schur_difference_step_m=None,
        augmentation_schur_maximum_attempts=1,
        augmentation_schur_maximum_relinearizations=0,
        d1_interior_profile=True,
        d1_interior_coverage_tolerance_m2=1e-10,
        d1_interior_boundary_force_tolerance_n=1e-10,
        **({
            "q4_s3_shared_edge_tie_step_m": 3e-8,
            "q4_s3_shared_edge_tie_direction": (1.0, 0.0, 0.0),
            "q4_s3_shared_edge_tie_relative_tolerance": 1e-6,
        } if shared_edge_tie else {}),
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError(f"analytic strip capability rejected: {report.issues}")
    started = time.perf_counter()
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=False,
    )
    contact = result.info.get("force_displacement_history", [])
    final = contact[-1]["contact"] if contact else {}
    reaction = float(np.asarray(final.get("integration_force", (0., 0., 0.)))[2])
    record = {
        "candidate": "d1_interior_analytic_guided_strip_dev",
        "shared_edge_tie": shared_edge_tie,
        "shifted_target": shifted_target,
        "columns": columns,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "contact_failure": result.info.get("contact_failure"),
        "accepted_steps": len(result.steps),
        "elapsed_seconds": time.perf_counter()-started,
        "shear_resultant_n_per_m": shear,
        "initial_opening_m": opening,
        "analytical_front_from_each_end_m": front,
        "total_distributed_load_n": load,
        "analytical_integrated_reaction_n": expected_reaction,
        "computed_integrated_reaction_n": reaction,
        "reaction_relative_error": abs(reaction-expected_reaction)/expected_reaction,
        "minimum_gap_m": final.get("minimum_gap"),
        "active_pressure_node_ids": [node for node, pressure in zip(
            final.get("source_pressure_node_ids", ()),
            final.get("numerical_pressure_pa", ())) if pressure > 0.0],
        "history": contact,
    }
    output.write_text(json.dumps(record, indent=2, default=_json_default)+"\n",
                      encoding="utf-8")
    print(json.dumps({key: record[key] for key in (
        "status", "failure_reason", "accepted_steps", "elapsed_seconds",
        "computed_integrated_reaction_n", "reaction_relative_error",
    )}))


def _plateau(value: float) -> tuple[float, float]:
    edge = 0.25
    if value < edge:
        r = value/edge
    elif value > 1-edge:
        r = (1-value)/edge
    else:
        return 1.0, 0.0
    return 3*r*r-2*r*r*r, (6-12*r)/edge**2


def _pressure_shape(value: float) -> float:
    if value <= 0.25 or value >= 0.75:
        return 0.0
    return float(np.sin(2*np.pi*(value-0.25))**2)


def _exact_q1_plateau(value: float) -> tuple[float, float]:
    if value < 0.25:
        return 4*value, 4.0
    if value > 0.75:
        return 4*(1-value), -4.0
    return 1.0, 0.0


def _exact_q1_pressure(value: float) -> float:
    if value <= 0.25 or value >= 0.75:
        return 0.0
    return 4*(value-0.25) if value < 0.5 else 4*(0.75-value)


def manufactured_2d(columns: int, output: Path, *, exact_q1: bool = False):
    """Exact guided-shear obstacle with a smooth interior contact pressure."""
    if columns not in (4, 8, 16) or output.exists():
        raise ValueError("unsupported or already saved manufactured case")
    shear = (5/6)*0.05*1e5/(2*(1+0.3))
    opening = 40e-6
    pressure_peak = 4.0
    expected_reaction = pressure_peak/16.0
    model, definition, settings, _program = build_case(
        columns, source_rows=columns,
        source_y_profile="uniform_ref16" if columns == 16 else None,
        offline_structural_reference_y16=columns == 16,
        full_width_target=True, full_width_target_y0_m=-0.113,
        target_right_x_m=1.3, load_profile="distributed",
        total_force_n=expected_reaction,
    )
    stride = columns+1
    model.boundary_conditions = [
        bc for bc in model.boundary_conditions
        if bc.name != "source_left_support"
    ]
    rim = (tuple(range(1, stride+1))
           + tuple(node for row in range(1, columns)
                   for node in (1+row*stride, (row+1)*stride))
           + tuple(range(1+columns*stride, 1+(columns+1)*stride)))
    model.add_boundary_condition(BoundaryCondition(
        "source_perimeter_support", rim, {"uz": 0.0},
    ))
    settings = replace(
        settings, enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=expected_reaction,
        augmentation_acceleration="none",
        augmentation_schur_maximum_modes=None,
        augmentation_schur_maximum_pressure_pa=None,
        augmentation_schur_maximum_step_pa=None,
        augmentation_schur_difference_step_m=None,
        augmentation_schur_maximum_attempts=1,
        augmentation_schur_maximum_relinearizations=0,
        d1_interior_profile=True,
        d1_interior_coverage_tolerance_m2=1e-10,
        d1_interior_boundary_force_tolerance_n=1e-10,
    )
    nodes = np.zeros((columns+1, columns+1))
    roots, weights = np.polynomial.legendre.leggauss(8)
    points = 0.5*(roots+1)
    weights = 0.5*weights
    h = 1/columns
    for row in range(columns):
        for column in range(columns):
            for i, wi in zip(points, weights):
                x = (column+i)*h
                px, pxx = (_exact_q1_plateau(x) if exact_q1
                           else _plateau(x))
                sx = (_exact_q1_pressure(x) if exact_q1
                      else _pressure_shape(x))
                for j, wj in zip(points, weights):
                    y = (row+j)*h
                    py, pyy = (_exact_q1_plateau(y) if exact_q1
                               else _plateau(y))
                    sy = (_exact_q1_pressure(y) if exact_q1
                          else _pressure_shape(y))
                    shape = np.asarray((
                        ((1-i)*(1-j), i*(1-j)),
                        ((1-i)*j, i*j),
                    ))
                    if exact_q1:
                        dx = np.asarray((
                            (-(1-j)/h, (1-j)/h), (-j/h, j/h),
                        ))
                        dy = np.asarray((
                            (-(1-i)/h, -i/h), ((1-i)/h, i/h),
                        ))
                        wx, wy = -opening*pxx*py, -opening*px*pyy
                        local_load = shear*(wx*dx+wy*dy) - (
                            pressure_peak*sx*sy)*shape
                    else:
                        load_density = (shear*opening*(pxx*py+px*pyy)
                                        - pressure_peak*sx*sy)
                        local_load = load_density*shape
                    nodes[row:row+2, column:column+2] += (
                        wi*wj*h*h*local_load
                    )
    if abs(float(np.sum(nodes))+expected_reaction) > 1e-10:
        raise RuntimeError("manufactured load quadrature is inconsistent")
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(stride+1, "ux")),
        waypoints=((0.0, 0.0), (0.0, 0.0)),
        steps_per_segment=2,
        nodal_forces=tuple(ContactPathNodalForce(
            1+row*stride+column, "uz", (0.0, float(nodes[row, column])),
        ) for row in range(stride) for column in range(stride)
            if abs(nodes[row, column]) > 1e-15),
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError(f"manufactured patch capability rejected: {report.issues}")
    started = time.perf_counter()
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=False,
    )
    history = result.info.get("force_displacement_history", [])
    final = history[-1]["contact"] if history else {}
    reaction = float(np.asarray(final.get("integration_force", (0., 0., 0.)))[2])
    center_node = 1+(columns//2)*stride+columns//2
    center_uz = float(result.displacements[
        model.mesh.dof_manager.get_node_dofs(center_node)[2]])
    source_vertical = [float(result.displacements[
        model.mesh.dof_manager.get_node_dofs(node)[2]])
        for node in range(1, stride*stride+1)]
    expected_vertical = [
        -opening*(_exact_q1_plateau(column/columns)[0] if exact_q1
                  else _plateau(column/columns)[0])
        *(_exact_q1_plateau(row/columns)[0] if exact_q1
          else _plateau(row/columns)[0])
        for row in range(stride) for column in range(stride)
    ]
    pressure_errors = []
    rim_pressure = []
    for node, pressure in zip(final.get("source_pressure_node_ids", ()),
                              final.get("numerical_pressure_pa", ())):
        column, row = (node-1) % stride, (node-1)//stride
        shape = _exact_q1_pressure if exact_q1 else _pressure_shape
        pressure_errors.append(abs(float(pressure)-pressure_peak
                                   *shape(column/columns)*shape(row/columns)))
        if row in (0, columns) or column in (0, columns):
            rim_pressure.append(abs(float(pressure)))
    record = {
        "candidate": ("d1_interior_exact_q1_weak_manufacture_dev"
                      if exact_q1 else "d1_interior_manufactured_2d_dev"),
        "columns": columns,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "contact_failure": result.info.get("contact_failure"),
        "accepted_steps": len(result.steps),
        "elapsed_seconds": time.perf_counter()-started,
        "analytical_integrated_reaction_n": expected_reaction,
        "computed_integrated_reaction_n": reaction,
        "reaction_relative_error": abs(reaction-expected_reaction)/expected_reaction,
        "analytical_center_displacement_m": -opening,
        "computed_center_displacement_m": center_uz,
        "source_vertical_displacements_m": source_vertical,
        "maximum_analytical_vertical_error_m": max(abs(a-b) for a,b in zip(
            source_vertical, expected_vertical)),
        "independent_nodal_gap_lower_m": min(opening+value
                                              for value in source_vertical),
        "maximum_analytical_pressure_error_pa": max(pressure_errors,
                                                     default=None),
        "maximum_rim_pressure_pa": max(rim_pressure, default=None),
        "minimum_gap_m": final.get("minimum_gap"),
        "active_pressure_node_ids": [node for node, pressure in zip(
            final.get("source_pressure_node_ids", ()),
            final.get("numerical_pressure_pa", ())) if pressure > 0.0],
        "history": history,
    }
    output.write_text(json.dumps(record, indent=2, default=_json_default)+"\n",
                      encoding="utf-8")
    print(json.dumps({key: record[key] for key in (
        "status", "failure_reason", "accepted_steps", "elapsed_seconds",
        "computed_integrated_reaction_n", "reaction_relative_error",
        "contact_failure",
    )}))


if __name__ == "__main__":
    if "--manufactured-2d" in sys.argv[3:]:
        manufactured_2d(int(sys.argv[1]), Path(sys.argv[2]),
                        exact_q1="--exact-q1" in sys.argv[3:])
    else:
        main(int(sys.argv[1]), Path(sys.argv[2]),
             shared_edge_tie="--shared-edge-tie" in sys.argv[3:],
             shifted_target="--shifted-target" in sys.argv[3:])
