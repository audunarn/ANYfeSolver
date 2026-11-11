from __future__ import annotations

import numpy as np

from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import LinearHardeningCurve
from anyfesolver.nonlinear_static import ShellInitialField, solve_static_nonlinear


def _single_coordinate_model(*, plastic: bool = False) -> tuple[FEModel, LoadCase]:
    model = FEModel("qualified-s3-native-workflow")
    model.add_material(
        "steel",
        210.0e9,
        0.3,
        density=7850.0,
        hardening_curve=(
            LinearHardeningCurve(250.0e6, 2.0e9) if plastic else None
        ),
    )
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 0.2, 0.9, 0.0)
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.01,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    model.add_boundary_condition(FixedSupport("node-1", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "node-2-one-coordinate",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(FixedSupport("node-3", [3]))
    load = LoadCase("pull")
    load.add_nodal_load(
        2,
        [2.0e6 if plastic else 1.0e3, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    return model, load


def _assert_state_matches_accepted_displacement(
    model: FEModel,
    displacement: np.ndarray,
    state: dict[str, object],
) -> None:
    element = model.mesh.get_element(1)
    assert element is not None
    mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    np.testing.assert_array_equal(
        state["committed_total_u"],
        np.asarray(displacement, dtype=np.float64)[mapping],
    )
    assert state["state_schema"] == "anysolver.e4_pl_s3.committed_state.v2"


def test_layered_s3_runs_through_the_production_static_state_transaction() -> None:
    model, load = _single_coordinate_model()

    result = solve_static_nonlinear(
        model,
        load,
        num_steps=2,
        num_layers=3,
    )

    assert result.status == "completed"
    assert result.load_factor == 1.0
    _assert_state_matches_accepted_displacement(
        model,
        result.displacements,
        result.element_states[1],
    )
    assert result.info["nonlinear_state_storage"]["native_rotation_activated"] is True
    assert result.info["nonlinear_state_storage"]["native_rotation_generation"] > 0


def test_layered_s3_material_history_commits_only_the_accepted_plastic_state() -> None:
    model, load = _single_coordinate_model(plastic=True)

    result = solve_static_nonlinear(
        model,
        load,
        num_steps=8,
        num_layers=3,
        max_iterations=30,
    )

    assert result.status == "completed"
    state = result.element_states[1]
    _assert_state_matches_accepted_displacement(model, result.displacements, state)
    assert float(np.max(np.asarray(state["alpha"], dtype=float))) > 0.0
    assert float(np.max(np.abs(np.asarray(state["plastic_strain"], dtype=float)))) > 0.0


def test_layered_s3_initial_field_is_equilibrated_and_committed_with_provenance() -> None:
    model, _load = _single_coordinate_model()
    model.boundary_conditions.clear()
    model.add_boundary_condition(FixedSupport("all", [1, 2, 3]))

    result = solve_static_nonlinear(
        model,
        num_steps=1,
        num_layers=3,
        initial_fields={
            1: ShellInitialField(
                membrane_stress=(1.0e6, 0.0, 0.0),
                source="qualified-s3-native-field-test",
            )
        },
    )

    assert result.status == "empty_reduced_system"
    state = result.element_states[1]
    assert state["initial_field_provenance"] == {
        "kind": "shell",
        "source": "qualified-s3-native-field-test",
        "components": ["initial_membrane_stress"],
    }
    assert float(np.linalg.norm(state["committed_internal_force"])) > 0.0
    assert result.info["initial_state_equilibration"]["native_tl_evaluated"] is True


def test_layered_s3_runs_through_arc_length_with_the_same_native_transaction() -> None:
    model, load = _single_coordinate_model()
    control = ArcLengthControl(
        initial_load_increment=0.05,
        minimum_load_increment=0.001,
        maximum_load_increment=0.10,
        maximum_absolute_load_factor=0.10,
        max_steps=2,
    )

    result = solve_static_arc_length(
        model,
        load,
        control=control,
        num_layers=3,
    )

    assert result.status == "load_factor_limit_reached"
    assert len(result.steps) == 2
    _assert_state_matches_accepted_displacement(
        model,
        result.displacements,
        result.element_states[1],
    )
    assert result.info["nonlinear_state_storage"]["native_rotation_activated"] is True
