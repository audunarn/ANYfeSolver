from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from anyfesolver.assembly import solve_nonlinear
from anyfesolver.boundary import LoadCase
from anyfesolver.corotational import (
    corotational_element_response,
    validate_corotational_scope,
)
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.elements import ShellElement
from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear_static import (
    ShellInitialField,
    _prepare_initial_states,
    _state_has_initial_field,
)
from anyfesolver.nonlinear_performance import (
    _state_has_initial_fields as _performance_state_has_initial_fields,
)
from anyfesolver.nonlinear import solve_nonlinear_load_stepping
from anyfesolver.nonlinear_state import StateTransactionError


_TRIANGLE = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.2, 0.9, 0.0),
)


def _model_with(element: ShellElement) -> FEModel:
    model = FEModel("s3-model-bound-state-hook")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(_TRIANGLE, start=1):
        model.add_node(node_id, *coordinates)
    model.add_element(1, element)
    return model


class _ModelBoundHookShell(ShellElement):
    legacy_nonlinear_batch_eligible = False

    def __init__(self) -> None:
        super().__init__(1, [1, 2, 3], "steel", thickness=0.012)
        self.hook_calls: list[dict[str, Any]] = []

    def init_nonlinear_state(self, num_layers: int) -> dict[str, Any]:
        raise AssertionError(
            "legacy layer-count-only initialization ran despite the model-bound hook"
        )

    def init_model_bound_nonlinear_state(
        self,
        mesh: Any,
        material: Any,
        num_layers: int,
        *,
        initial_fields: dict[str, np.ndarray],
        initial_field_provenance: dict[str, Any],
    ) -> dict[str, Any]:
        self.hook_calls.append(
            {
                "mesh": mesh,
                "material": material,
                "num_layers": num_layers,
                "initial_fields": {
                    key: np.asarray(value, dtype=float).copy()
                    for key, value in initial_fields.items()
                },
                "initial_field_provenance": dict(initial_field_provenance),
            }
        )
        return {
            "initialized_by_model_bound_hook": True,
            **{
                key: np.asarray(value, dtype=float).copy()
                for key, value in initial_fields.items()
            },
            "initial_field_provenance": dict(initial_field_provenance),
        }


def test_prepare_initial_states_passes_model_and_field_identity_atomically_to_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = _ModelBoundHookShell()
    model = _model_with(element)
    validation_calls: list[dict[str, Any]] = []

    def record_validation(
        validated_element: Any,
        material: Any,
        state: dict[str, Any],
        num_layers: int,
        mesh: Any,
    ) -> None:
        validation_calls.append(
            {
                "element": validated_element,
                "material": material,
                "state": state,
                "num_layers": num_layers,
                "mesh": mesh,
            }
        )

    monkeypatch.setattr(
        "anyfesolver.elements.validate_initial_field_state",
        record_validation,
    )
    states, provenance = _prepare_initial_states(
        model,
        initial_element_states=None,
        initial_fields={
            1: ShellInitialField(
                membrane_stress=[12.0, -3.0, 2.0],
                source="model-bound-map-v1",
            )
        },
        num_layers=3,
    )

    assert len(element.hook_calls) == 1
    call = element.hook_calls[0]
    assert call["mesh"] is model.mesh
    assert call["material"] is model.get_material("steel")
    assert call["num_layers"] == 3
    assert set(call["initial_fields"]) == {"initial_membrane_stress"}
    np.testing.assert_array_equal(
        call["initial_fields"]["initial_membrane_stress"],
        [12.0, -3.0, 2.0],
    )
    assert call["initial_field_provenance"] == {
        "kind": "shell",
        "source": "model-bound-map-v1",
        "components": ["initial_membrane_stress"],
    }
    assert states[1]["initialized_by_model_bound_hook"] is True
    assert validation_calls[0]["state"] is states[1]
    assert provenance[0]["source"] == "model-bound-map-v1"


def test_zero_filled_s3_initial_field_storage_without_provenance_is_inactive() -> None:
    state = {
        name: np.zeros((7, 3), dtype=float)
        for name in (
            "initial_membrane_stress",
            "initial_bending_stress",
            "initial_membrane_prestrain",
            "initial_curvature_prestrain",
        )
    }

    assert _state_has_initial_field(state) is False
    assert _state_has_initial_field({**state, "initial_field_provenance": {}}) is False
    assert _state_has_initial_field(
        {**state, "initial_field_provenance": None}
    ) is True
    assert _performance_state_has_initial_fields(state) is False
    assert (
        _performance_state_has_initial_fields(
            {**state, "initial_field_provenance": {}}
        )
        is False
    )
    assert (
        _performance_state_has_initial_fields(
            {**state, "initial_field_provenance": None}
        )
        is True
    )


def test_nonzero_or_provenanced_s3_initial_field_storage_is_active() -> None:
    zeros = {
        name: np.zeros((7, 3), dtype=float)
        for name in (
            "initial_membrane_stress",
            "initial_bending_stress",
            "initial_membrane_prestrain",
            "initial_curvature_prestrain",
        )
    }
    nonzero = {key: value.copy() for key, value in zeros.items()}
    nonzero["initial_curvature_prestrain"][4, 1] = np.nextafter(0.0, 1.0)
    provenanced = {
        **zeros,
        "initial_field_provenance": {
            "kind": "shell",
            "source": "manufacturing-map-v2",
            "components": ["initial_membrane_stress"],
        },
    }

    assert _state_has_initial_field(nonzero) is True
    assert _state_has_initial_field(provenanced) is True
    assert _performance_state_has_initial_fields(nonzero) is True
    assert _performance_state_has_initial_fields(provenanced) is True


def test_legacy_shell_initialization_and_explicit_zero_field_remain_unchanged() -> None:
    element = ShellElement(1, [1, 2, 3], "steel", thickness=0.012)
    model = _model_with(element)
    states, provenance = _prepare_initial_states(
        model,
        initial_element_states=None,
        initial_fields={
            1: ShellInitialField(
                membrane_prestrain=[0.0, 0.0, 0.0],
                source="explicit-zero-legacy-field",
            )
        },
        num_layers=3,
    )

    state = states[1]
    assert state["plastic_strain"].shape == (len(element.gauss_points) * 3, 3)
    assert state["alpha"].shape == (len(element.gauss_points) * 3,)
    np.testing.assert_array_equal(state["initial_membrane_prestrain"], np.zeros(3))
    assert state["initial_field_provenance"] == {
        "kind": "shell",
        "source": "explicit-zero-legacy-field",
        "components": ["initial_membrane_prestrain"],
    }
    assert _state_has_initial_field(state) is True
    assert provenance[0]["source"] == "explicit-zero-legacy-field"


def test_corotational_scope_rejects_formulation_native_s3_but_not_legacy_shell() -> None:
    native = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.012,
        reference_normal=[0.0, 0.0, 1.0],
    )
    with pytest.raises(
        ValueError,
        match=r"qualified S3.*formulation-native.*corotational",
    ):
        validate_corotational_scope(_model_with(native))

    legacy = ShellElement(1, [1, 2, 3], "steel", thickness=0.012)
    assert validate_corotational_scope(_model_with(legacy)) is None


def test_generic_corotational_wrapper_rejects_native_s3_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.012,
        reference_normal=[0.0, 0.0, 1.0],
    )
    model = _model_with(element)
    calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("mechanics")
        raise AssertionError("generic corotational wrapper evaluated native S3 mechanics")

    monkeypatch.setattr(element, "compute_nonlinear_response", forbidden)
    with pytest.raises(
        ValueError,
        match=r"qualified S3.*formulation-native.*corotational",
    ):
        corotational_element_response(
            model,
            1,
            element,
            np.zeros(18, dtype=float),
            tangent=True,
            committed_state=None,
            num_layers=3,
        )
    assert calls == []


def test_legacy_nonlinear_entry_points_reject_native_s3_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.012,
        reference_normal=[0.0, 0.0, 1.0],
    )
    model = _model_with(element)
    load_case = LoadCase("zero")
    calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("mechanics")
        raise AssertionError("legacy nonlinear entry evaluated native S3 mechanics")

    monkeypatch.setattr(element, "compute_nonlinear_response", forbidden)
    with pytest.warns(DeprecationWarning):
        with pytest.raises(
            StateTransactionError,
            match=r"deprecated solve_nonlinear.*solver-owned native.*element IDs \[1\]",
        ):
            solve_nonlinear(model, load_case, max_iterations=1)
    with pytest.raises(
        ValueError,
        match=(
            "qualified S3 class authority has instance shadows: "
            "compute_nonlinear_response"
        ),
    ):
        solve_nonlinear_load_stepping(model, load_case, num_steps=1)
    assert calls == []


def test_qualified_s3_builds_and_validates_initial_fields_atomically() -> None:
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.012,
        reference_normal=[0.0, 0.0, 1.0],
    )
    model = _model_with(element)
    with pytest.raises(ValueError, match="model-bound.*mesh and material"):
        element.init_nonlinear_state(3)

    states, provenance = _prepare_initial_states(
        model,
        initial_element_states=None,
        initial_fields={
            1: ShellInitialField(
                membrane_prestrain=[1.0e-5, -2.0e-5, 3.0e-5],
                source="qualified-s3-native-map-v1",
            )
        },
        num_layers=3,
    )
    state = states[1]
    validated = element.validate_model_bound_nonlinear_state(
        model.mesh,
        model.get_material("steel"),
        state,
        3,
    )
    np.testing.assert_array_equal(
        validated["initial_membrane_prestrain"],
        np.broadcast_to([1.0e-5, -2.0e-5, 3.0e-5], (7, 3)),
    )
    assert validated["initial_field_provenance"] == {
        "kind": "shell",
        "source": "qualified-s3-native-map-v1",
        "components": ["initial_membrane_prestrain"],
    }
    assert _state_has_initial_field(validated) is True
    assert _performance_state_has_initial_fields(validated) is True
    assert provenance[0]["source"] == "qualified-s3-native-map-v1"
