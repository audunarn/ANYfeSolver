"""D1 interior displacement-control press through four open source nodes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

import anyfesolver
from anyfesolver import (
    ContactPathControl, ContactPathProgram, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.general_contact.static_path import (
    _elastic_displacement_force_predictor,
)
from anyfesolver.matrix_assembly import assemble_stiffness_matrix

from scripts.general_contact_d1_interior_press_dev import build_interior_press
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import _json_default


_DOF = {"ux": 0, "uy": 1, "uz": 2}
_OPEN_RING_NODES = (7, 8, 13, 17)
_REFERENCE_ACTIVE_NODE = 12


def build_case(reference: Path, *, full_cycle: bool = False):
    saved = json.loads(reference.read_text(encoding="utf-8"))
    expected = {
        "status": "completed", "accepted_steps": 2,
        "source_columns": 4, "source_rows": 4,
        "target_s3_faces": 2, "force_n": 0.04,
        "reference_slope": 1.0e-4, "slide_body": "target",
        "source_support": "perimeter", "load_shape": "interior_distributed",
    }
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("displacement fixture needs the accepted matched force pose")
    model, definition, settings, force_program = build_interior_press(
        source_support="perimeter", source_rows=4, source_columns=4,
        target_columns=1, force_n=0.04, reference_slope=1.0e-4,
        load_shape="interior_distributed", slide_body="target",
    )
    reference_u = np.asarray(saved["displacements"], dtype=float)
    dof_manager = model.mesh.dof_manager
    ring_dofs = [dof_manager.get_node_dofs(node)[2]
                 for node in _OPEN_RING_NODES]
    active_dof = dof_manager.get_node_dofs(_REFERENCE_ACTIVE_NODE)[2]
    if reference_u.shape != (dof_manager.total_dofs,):
        raise ValueError("accepted force pose has incompatible structural coordinates")
    if saved["history"][-1]["contact"]["active_pressure_dofs"] != 1:
        raise ValueError("accepted force pose no longer has one interior active mode")
    pressure = saved["history"][-1]["contact"]
    positive_nodes = tuple(int(node) for node, value in zip(
        pressure["source_pressure_node_ids"], pressure["numerical_pressure_pa"],
    ) if value > 0.0)
    if positive_nodes != (_REFERENCE_ACTIVE_NODE,):
        raise ValueError("accepted force pose changed its single active node")
    if _REFERENCE_ACTIVE_NODE in _OPEN_RING_NODES:
        raise ValueError("prescribed ring contains an active pressure node")
    ring_motion = float(np.mean(reference_u[ring_dofs]))
    ring = ContactPathControl(tuple((node, "uz", 0.25)
                                    for node in _OPEN_RING_NODES))
    press_waypoints = ((0.0, 0.0, 0.0), (0.0, 0.0, ring_motion))
    waypoints = (press_waypoints + (
        (0.05, 0.05, ring_motion),
        (0.05, 0.05, 0.0),
        (0.05, 0.05, ring_motion),
    ) if full_cycle else press_waypoints)
    program = ContactPathProgram(
        controls=(*force_program.controls, ring),
        waypoints=waypoints,
        steps_per_segment=2,
    )
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    zero = np.zeros(dof_manager.total_dofs)
    _, _, transform, offset, _, _ = build_constraint_transformation(
        stiffness, zero, model,
    )
    rows = np.zeros((len(program.controls), len(zero)))
    for index, control in enumerate(program.controls):
        for node, dof, weight in control.terms:
            rows[index, dof_manager.get_node_dofs(node)[_DOF[dof]]] += weight
    predictor = _elastic_displacement_force_predictor(
        stiffness, transform, np.asarray(rows @ transform),
        np.asarray(program.waypoints),
    )
    press_predictor = _elastic_displacement_force_predictor(
        stiffness, transform, np.asarray(rows @ transform),
        np.asarray(press_waypoints),
    )
    settings = replace(settings,
                       complementarity_force_scale_n=min(0.04, 0.5*press_predictor))
    return (model, definition, settings, program, predictor, press_predictor,
            ring_motion,
            float(reference_u[active_dof]))


def main(reference: Path, output: Path, *, full_cycle: bool = False) -> None:
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    (model, definition, settings, program, predictor, press_predictor,
     ring_motion, reference_active) = build_case(
        reference, full_cycle=full_cycle,
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError(f"open-ring displacement fixture rejected: {report.issues}")
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    active_dof = model.mesh.dof_manager.get_node_dofs(_REFERENCE_ACTIVE_NODE)[2]
    payload = {
        "candidate": "d1_open_ring_source_displacement_control_development",
        "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
        "reference_pose": str(reference),
        "source_control_nodes": list(_OPEN_RING_NODES),
        "source_control_weights": [0.25]*4,
        "ring_motion_m": ring_motion,
        "full_cycle": full_cycle,
        "force_reference_active_node_id": _REFERENCE_ACTIVE_NODE,
        "force_reference_active_node_motion_m": reference_active,
        "contact_free_elastic_force_predictor_n": predictor,
        "contact_free_press_force_predictor_n": press_predictor,
        "declared_complementarity_force_scale_n":
            settings.complementarity_force_scale_n,
        "profile": report.profile,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "elapsed_seconds": time.perf_counter()-started,
        "active_node_motion_m": float(result.displacements[active_dof]),
        "history": result.info.get("force_displacement_history"),
        "contact_failure": result.info.get("contact_failure"),
    }
    output.write_text(json.dumps(payload, indent=2, default=_json_default)+"\n",
                      encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "status", "failure_reason", "accepted_steps", "elapsed_seconds",
        "ring_motion_m", "contact_free_elastic_force_predictor_n",
    )}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--full-cycle", action="store_true")
    args = parser.parse_args()
    main(args.reference, args.output, full_cycle=args.full_cycle)
