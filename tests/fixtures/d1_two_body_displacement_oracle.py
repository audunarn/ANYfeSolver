"""Independent small-motion Q1/S3 KKT oracle for prescribed source deflection."""

from __future__ import annotations

import numpy as np

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.matrix_assembly import assemble_stiffness_matrix

from scripts.d1_reference_gap_work import independent_gap_work


def solve_displacement_reference(model, definition, control):
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    ndof = model.mesh.dof_manager.total_dofs
    _, _, transform, offset, _, _ = build_constraint_transformation(
        stiffness, np.zeros(ndof), model)
    assert np.linalg.norm(offset) == 0.0
    ids, support, gap_work, area = independent_gap_work(
        model, definition, 0.0)
    source = model.mesh.elements[definition.surfaces[0].element_ids[0]]
    target = model.mesh.elements[definition.surfaces[1].element_ids[0]]
    source_node = model.mesh.nodes[source.node_ids[0]]
    target_node = model.mesh.nodes[target.node_ids[0]]
    reference_gap = (
        source_node.z-0.5*source.thickness
        -target_node.z-0.5*target.thickness
        +target.reference_surface_offset)
    assert abs(reference_gap-40e-6) < 1e-12
    assert control.dof == "uz"
    control_row = np.zeros(ndof)
    control_dof = model.mesh.dof_manager.get_node_dofs(control.node_id)[2]
    control_row[control_dof] = 1.0
    k = (transform.T @ stiffness @ transform).toarray()
    g = np.asarray(gap_work @ transform)
    c = np.asarray(control_row @ transform).reshape(1, -1)

    def equilibrate(active):
        selected = np.flatnonzero(active)
        ga = g[selected]
        size = k.shape[0]
        matrix = np.block([
            [k, -ga.T, c.T],
            [ga, np.zeros((len(selected), len(selected))),
             np.zeros((len(selected), 1))],
            [c, np.zeros((1, len(selected))), np.zeros((1, 1))],
        ])
        rhs = np.r_[np.zeros(size), -reference_gap*support[selected],
                    control.target_displacement]
        solved = np.linalg.solve(matrix, rhs)
        pressure = np.zeros(len(support))
        pressure[selected] = solved[size:size+len(selected)]
        return solved[:size], pressure, float(solved[-1])

    active = np.zeros(len(support), dtype=bool)
    for iteration in range(30):
        coordinates, pressure, control_reaction = equilibrate(active)
        normalized_gap = reference_gap+(g @ coordinates)/support
        updated = (pressure-definition.interactions[0].normal_stiffness
                   *normalized_gap) > 0.0
        if np.array_equal(updated, active):
            break
        active = updated
    else:
        raise RuntimeError("independent displacement active set did not converge")
    equilibrium = k @ coordinates-g.T @ pressure+c.T.ravel()*control_reaction
    assert np.linalg.norm(equilibrium) < 1e-9
    assert np.min(pressure) >= -1e-10
    assert np.min(normalized_gap) >= -1e-10
    assert np.max(abs(pressure*normalized_gap)) < 1e-10
    pose = np.asarray(transform @ coordinates).reshape(-1)
    contact_force = gap_work.T @ pressure
    return {
        "displacement": pose,
        "pressure": pressure,
        "support": support,
        "weighted_gap_work": gap_work,
        "reference_gap": reference_gap,
        "normalized_gap": normalized_gap,
        "source_pressure_nodes": ids,
        "area": area,
        "integrated_force": float(support @ pressure),
        "control_reaction": control_reaction,
        "endpoint_contact_force_displacement_pairing": float(
            contact_force @ pose),
        "active_set_iterations": iteration+1,
    }
