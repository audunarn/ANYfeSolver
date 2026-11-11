"""Development example: prescribed displacement of a Q4 patch onto S3 skins.

This bounded planar, frictionless case exercises the ordinary nonlinear static
API. It is an implemented/tested development profile, not an accepted contact
capability or a general recommendation for penalty and gap settings.
"""

from __future__ import annotations

import json

import anyfesolver
from anyfesolver import (
    BoundaryCondition,
    ContactDefinition,
    ContactInteraction,
    ContactSolverSettings,
    ContactSurface,
    create_shell_element,
    solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear_static import DisplacementControl


def build_model():
    model = FEModel("general_contact_ordinary_displacement_dev")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for row, y in enumerate((0.1, 1.1)):
        for column in range(5):
            model.add_node(1 + 5*row + column, 0.1 + column/4, y, 0.04004)
    for node, (x, y) in zip((11, 12, 13, 14),
                             ((0.0, 0.0), (1.3, 0.0),
                              (1.3, 1.0), (0.0, 1.0))):
        model.add_node(node, x, y, 0.0)
    for element_id in range(1, 5):
        model.add_element(element_id, create_shell_element(
            element_id,
            (element_id, element_id+1, element_id+6, element_id+5),
            "elastic", thickness=0.05,
            reference_normal=(0.0, 0.0, 1.0),
        ))
    for element_id, nodes in ((5, (11, 12, 14)), (6, (12, 13, 14))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d",
            reference_normal=(0.0, 0.0, 1.0),
            reference_surface_offset=0.01,
        ))
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (11, 12, 13, 14),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", tuple(range(1, 11)),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_left_support", (1, 6), {"uz": 0.0},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_x_support", (1, 6), {"ux": 0.0},
    ))
    definition = ContactDefinition(
        surfaces=(
            ContactSurface("q4_source", (1, 2, 3, 4), "bottom"),
            ContactSurface("s3_target", (5, 6), "top"),
        ),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "q4_source", "s3_target", 10000.0, 5.0e-5,
            normal_law="augmented",
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        q4_source_gap_certificate="projected_matching",
        general_s3_difference_step=2.0e-5,
        maximum_general_s3_energy_evaluations=14144,
        maximum_candidate_pairs=32,
        q4_source_area_tolerance=1.0e-9,
        q4_source_weighted_gap_tolerance=1.0e-10,
        maximum_q4_source_integration_triangles=300,
        general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1.0e-5,
        q4_source_support_hessian_tolerance=1.0e-5,
        q4_source_weighted_gap_gradient_tolerance=1.0e-5,
        q4_source_weighted_gap_hessian_tolerance=1.0e-5,
        pressure_tolerance=1.0e-5,
        maximum_augmentations=7,
        enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=0.01,
        monolithic_linear_solver="block_response_dev",
    )
    return model, definition, settings


def main() -> None:
    model, definition, settings = build_model()
    admission = validate_general_contact(model, definition, settings)
    if not admission.valid:
        raise ValueError(admission.issues)
    result = solve_static_nonlinear(
        model,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=5, dof="uz", target_displacement=-8.0e-5,
        ),
        num_steps=4, max_iterations=18,
        contact=definition, contact_settings=settings,
        emit_restart_checkpoint=True,
    )
    if result.status != "completed":
        raise RuntimeError(f"contact solve failed: {result.failure_reason}")
    contact = result.info["general_contact"]
    print(json.dumps({
        "status": result.status,
        "accepted_steps": len(result.steps),
        "contact_profile": contact["profile"],
        "minimum_gap_m": float(contact["minimum_gap"]),
        "gap_tolerance_met": bool(contact["gap_tolerance_met"]),
        "active_pressure_modes": int(contact["active_pressure_dofs"]),
        "integrated_normal_force_n": float(contact["integration_force"][2]),
        "control_generalized_force_n": float(
            result.info["force_displacement_history"][-1][
                "reaction_generalized"][0]),
        "contact_free_elastic_force_predictor_n": float(
            result.info["contact_free_elastic_force_predictor_n"]),
        "has_restart_checkpoint": result.restart_checkpoint is not None,
        "anyfesolver_module": str(anyfesolver.__file__),
    }, indent=2))


if __name__ == "__main__":
    main()
