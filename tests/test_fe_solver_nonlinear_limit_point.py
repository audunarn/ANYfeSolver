import numpy as np
import pytest

from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.buckling import solve_eigenvalue_buckling
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear import solve_nonlinear_load_stepping


def _beam_column_model(num_elements=8):
    length = 4.0
    model = FEModel("beam_column_limit_point")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.current_material = "steel"

    for i in range(num_elements + 1):
        model.add_node(i + 1, length * i / num_elements, 0.0, 0.0)

    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for i in range(num_elements):
        element_id = i + 1
        model.add_element(element_id, BeamElement(element_id, [i + 1, i + 2], "steel", section))

    all_nodes = list(range(1, num_elements + 2))
    end_nodes = [1, num_elements + 1]
    model.add_boundary_condition(
        BoundaryCondition("suppress_unrelated_dofs", all_nodes, {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0})
    )
    model.add_boundary_condition(BoundaryCondition("pinned_lateral_ends", end_nodes, {"uy": 0.0}))
    model.apply_boundary_conditions()
    return model


def _unit_lateral_load(model):
    middle_node = 1 + len(model.mesh.nodes) // 2
    load_case = LoadCase("unit_lateral")
    load_case.add_nodal_load(middle_node, forces=np.array([0.0, 1.0, 0.0]))
    return load_case


def test_nonlinear_load_stepping_stops_near_eigenvalue_limit_point():
    model = _beam_column_model(num_elements=10)
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    buckling = solve_eigenvalue_buckling(model, states, num_modes=1)

    result = solve_nonlinear_load_stepping(
        model,
        _unit_lateral_load(model),
        states,
        max_load_factor=1.2 * buckling.critical_load_factor,
        num_steps=48,
        stability_tolerance=0.03,
    )

    assert result.status in {"near_limit_point", "limit_point_detected"}
    assert result.critical_load_factor_estimate == pytest.approx(buckling.critical_load_factor, rel=0.08)
    assert result.last_load_factor <= 1.03 * buckling.critical_load_factor
    assert result.steps[0].tangent_stability_index == pytest.approx(1.0)
    assert result.steps[-1].tangent_status in {"near_limit", "unstable"}
    # Residual of the solved tangent system, relative to the applied load level.
    assert result.steps[-1].residual_norm <= 1.0e-9 * max(1.0, result.steps[-1].load_factor)


def test_tangent_stability_index_decreases_monotonically_for_proportional_compression():
    model = _beam_column_model(num_elements=6)
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    buckling = solve_eigenvalue_buckling(model, states, num_modes=1)

    result = solve_nonlinear_load_stepping(
        model,
        _unit_lateral_load(model),
        states,
        max_load_factor=0.8 * buckling.critical_load_factor,
        num_steps=12,
        stability_tolerance=0.0,
    )

    indices = [step.tangent_stability_index for step in result.steps]
    assert result.status == "completed"
    assert all(earlier >= later for earlier, later in zip(indices, indices[1:]))
    assert indices[-1] > 0.0


def test_load_stepping_without_geometric_stiffness_completes_without_limit_point():
    model = _beam_column_model(num_elements=2)

    result = solve_nonlinear_load_stepping(
        model,
        _unit_lateral_load(model),
        element_states=None,
        max_load_factor=10.0,
        num_steps=4,
    )

    assert result.status == "completed"
    assert result.critical_load_factor_estimate is None
    assert len(result.steps) == 5
    assert all(step.tangent_status == "stable" for step in result.steps)
