"""Focused probe: the active contact work is exactly independent of the control dofs.

Companion to d1_two_body_control_reaction_decomposition.py. The two
displacement-control dofs (ux of source nodes 1 and 6) belong to source
element neighbourhoods that are disjoint from the source elements supporting
the active pressure modes, and the control nodes are absent from the target
skin. The active modes' current-skin gap quadrature therefore executes
bitwise-identical arithmetic with and without the control perturbation, so
the independent finite-difference control gradient and the production
contact force at the control dofs are exactly zero for the registered
half/nominal/double perturbations on the admitted fixture topology.
The control reaction is consequently a purely structural cancelled sum, and
its independent-vs-production comparison is limited by arithmetic-realization
rounding of that sum, not by contact work or finite-difference error.
"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

import anyfesolver
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import linearize
from d1_current_skin import integrate_current_skin
from scripts.general_contact_d1_free_target_press_probe import build_free_target_case


path = (Path(__file__).resolve().parents[2] / "docs/d1_review_support"
        / "d1_two_body_reference_compare_full_20260928.json")
saved_bytes = path.read_bytes()
saved = json.loads(saved_bytes)
assert saved["production_status"] == "completed"
assert saved["production_accepted_steps"] == 4
model, definition, settings, _program = build_free_target_case(
    4, steps_per_segment=2, four_phase=True,
    enforcement_solver="monolithic_active_set",
)
model.apply_boundary_conditions()
ndof = model.mesh.dof_manager.total_dofs
control_nodes = (1, 6)
control_dofs = tuple(int(model.mesh.dof_manager.get_node_dofs(node)[0])
                     for node in control_nodes)
control_list = list(control_dofs)
pressure_nodes = sorted({
    node for element_id in definition.surfaces[0].element_ids
    for node in model.mesh.elements[element_id].node_ids
})
target_skin_nodes = sorted({
    node for element_id in definition.surfaces[1].element_ids
    for node in model.mesh.elements[element_id].node_ids
})
production_pressure = np.asarray(saved["production_pressure_pa"])
oracle_pressure = np.asarray(saved["oracle_pressure_pa"])
active = np.flatnonzero(production_pressure > 0.0)
assert np.array_equal(active, np.flatnonzero(oracle_pressure > 0.0))
active_nodes = [pressure_nodes[index] for index in active]


def source_elements_of(node):
    return {int(element_id) for element_id in definition.surfaces[0].element_ids
            if node in model.mesh.elements[element_id].node_ids}


control_elements = set().union(*(source_elements_of(node)
                                 for node in control_nodes))
active_elements = set().union(*(source_elements_of(node)
                                for node in active_nodes))
disjointness = {
    "control_nodes": list(control_nodes),
    "active_mode_nodes": active_nodes,
    "control_source_elements": sorted(control_elements),
    "active_source_elements": sorted(active_elements),
    "active_elements_disjoint_from_control_elements":
        bool(active_elements.isdisjoint(control_elements)),
    "control_nodes_on_target_skin":
        [bool(node in target_skin_nodes) for node in control_nodes],
}
assert disjointness["active_elements_disjoint_from_control_elements"]
assert not any(disjointness["control_nodes_on_target_skin"])

pose = np.asarray(saved["production_displacements"])
support, weighted, _area, _points, signature = integrate_current_skin(
    model, definition, pose)
control_columns = np.eye(ndof)[:, control_list]
gradients = {}
bitwise_invariance = {}
for label, step in (("half", 1e-6), ("nominal", 2e-6), ("double", 4e-6)):
    gradient = np.zeros((len(pressure_nodes), len(control_dofs)))
    invariant = True
    for column, dof in enumerate(control_dofs):
        plus = pose.copy()
        minus = pose.copy()
        plus[dof] += step
        minus[dof] -= step
        plus_support, plus_weighted, _a, _p, plus_signature = (
            integrate_current_skin(model, definition, plus))
        minus_support, minus_weighted, _a, _p, minus_signature = (
            integrate_current_skin(model, definition, minus))
        assert plus_signature == signature == minus_signature
        invariant = invariant and bool(
            np.array_equal(plus_support[active], support[active])
            and np.array_equal(plus_weighted[active], weighted[active])
            and np.array_equal(minus_support[active], support[active])
            and np.array_equal(minus_weighted[active], weighted[active]))
        gradient[:, column] = ((plus_weighted/plus_support
                                -minus_weighted/minus_support)/(2*step))
    gradients[label] = gradient
    bitwise_invariance[label] = invariant
    assert not gradient[active].any()
    assert invariant

production_support = np.asarray(saved["production_support_m2"])
production_z = production_pressure*production_support
manager = GeneralS3PatchManager(model, definition, settings)
contact = linearize(manager, pose, production_z, tangent=False)
production_force_control = np.asarray(contact.force[control_list])
production_gradient_control = np.asarray(
    contact.gap_gradient[:, control_list])
assert not production_force_control.any()
assert not production_gradient_control[active].any()

external = np.zeros(ndof)
for force, value in zip(_program.nodal_forces,
                        _program.force_values_at(4), strict=True):
    assert force.dof == "uz"
    external[model.mesh.dof_manager.get_node_dofs(force.node_id)[2]] += value
external_control = external[control_list]
assert not external_control.any()

record = {
    "status": "development_control_reaction_route_probe",
    "source_commit": subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True).strip(),
    "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "saved_source_sha256": hashlib.sha256(saved_bytes).hexdigest(),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
    "active_pressure_modes": active.tolist(),
    "disjointness": disjointness,
    "independent_control_gradient_half_step": gradients["half"].tolist(),
    "independent_control_gradient_nominal": gradients["nominal"].tolist(),
    "independent_control_gradient_double_step": gradients["double"].tolist(),
    "active_rows_bitwise_invariance": bitwise_invariance,
    "production_contact_force_control_n": production_force_control.tolist(),
    "production_gap_gradient_control_columns":
        production_gradient_control.tolist(),
    "production_gap_gradient_control_max_abs": float(
        np.max(abs(production_gradient_control))),
    "external_control_n": external_control.tolist(),
}
print(json.dumps(record, indent=2))
