from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pytest

from anyfesolver import (
    FEModel,
    QualifiedE4PLShellElement,
    solve_eigenvalue_buckling,
    solve_free_vibration,
)
from anyfesolver.boundary import FixedSupport
from anyfesolver.current_state_tangent import (
    COMMITTED_CURRENT_TANGENT_INPUT_OWNERSHIP_POLICY_ID,
    assemble_committed_current_tangent_components,
    validate_committed_current_tangent_inputs,
)
from anyfesolver.element_capabilities import ElementCapabilityError


class _SingleItemsMapping(Mapping[Any, Any]):
    """A mapping that proves consumers detach its one authoritative view."""

    def __init__(self, values: Mapping[Any, Any]) -> None:
        self._values = dict(values)
        self.items_reads = 0

    def items(self):  # type: ignore[override]
        self.items_reads += 1
        if self.items_reads != 1:
            raise AssertionError("caller-controlled mapping was consumed twice")
        return tuple(self._values.items())

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[Any]:
        raise AssertionError("owned snapshot bypassed the authoritative items view")

    def __getitem__(self, key: Any) -> Any:
        del key
        raise AssertionError("mechanics consulted caller-controlled mapping state")

    def get(self, key: Any, default: Any = None) -> Any:
        del key, default
        raise AssertionError("mechanics consulted caller-controlled mapping state")


def _model_and_zero_state() -> tuple[FEModel, np.ndarray, dict[str, object]]:
    model = FEModel("q4-current-state-owned-input")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    element = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "steel",
        thickness=0.02,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)
    model.add_boundary_condition(FixedSupport("left", [1, 4]))
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64)
    local = displacement[np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)]
    state = element.seal_committed_current_tangent_state(
        model.mesh,
        model.get_material("steel"),
        local,
        element.init_nonlinear_state(3),
        3,
    )
    return model, displacement, state


@pytest.mark.parametrize("route", ("assembly", "modal", "buckling"))
def test_current_state_routes_consume_hostile_inputs_once(route: str) -> None:
    model, displacement, state = _model_and_zero_state()
    state_view = _SingleItemsMapping(state)
    states_view = _SingleItemsMapping({1: state_view})

    if route == "assembly":
        _material, _geometric, total, info = (
            assemble_committed_current_tangent_components(
                model, displacement, states_view, 3
            )
        )
        assert total.shape == (24, 24)
        assert info["input_ownership_policy_id"] == (
            COMMITTED_CURRENT_TANGENT_INPUT_OWNERSHIP_POLICY_ID
        )
    elif route == "modal":
        result = solve_free_vibration(
            model,
            num_modes=1,
            current_state_displacements=displacement,
            current_state_element_states=states_view,
            current_state_num_layers=3,
        )
        assert result.solver_status == "ok"
    else:
        result = solve_eigenvalue_buckling(
            model,
            current_state_displacements=displacement,
            current_state_element_states=states_view,
            current_state_num_layers=3,
        )
        assert result.solver_status == "zero_geometric_stiffness"

    assert states_view.items_reads == 1
    assert state_view.items_reads == 1


@pytest.mark.parametrize(
    "raw_element_id",
    (True, 1.0, "01", " 1", "+1", "1.0"),
)
def test_current_state_rejects_noncanonical_element_ids(
    raw_element_id: object,
) -> None:
    model, displacement, state = _model_and_zero_state()
    with pytest.raises(ValueError, match="canonical integers"):
        validate_committed_current_tangent_inputs(
            model,
            displacement,
            {raw_element_id: state},
            3,
            context="test noncanonical current-state ID",
        )


def test_current_state_accepts_one_canonical_json_id_but_rejects_aliases() -> None:
    model, displacement, state = _model_and_zero_state()
    route = validate_committed_current_tangent_inputs(
        model,
        displacement,
        {"1": state},
        3,
        context="test canonical JSON current-state ID",
    )
    assert route["route"] == "qualified_q4"

    with pytest.raises(ValueError, match="ambiguous"):
        validate_committed_current_tangent_inputs(
            model,
            displacement,
            {1: state, "1": state},
            3,
            context="test aliased current-state IDs",
        )


def test_displacement_array_protocol_is_guarded_before_finite_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, displacement, state = _model_and_zero_state()
    reached: list[str] = []

    def forbidden_numeric(*_args: object, **_kwargs: object) -> bool:
        reached.append("numeric")
        raise AssertionError("changed numerical operation was invoked")

    class ObservedArray:
        def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
            del copy
            reached.append("array")
            monkeypatch.setattr(np, "all", forbidden_numeric)
            return np.asarray(displacement, dtype=dtype)

    with pytest.raises(ElementCapabilityError, match="NUMERICAL_AUTHORITY_MISMATCH"):
        validate_committed_current_tangent_inputs(
            model,
            ObservedArray(),
            {1: state},
            3,
            context="test displacement observation ordering",
        )
    assert reached == ["array"]


def test_nested_state_mapping_is_guarded_before_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, displacement, state = _model_and_zero_state()
    reached: list[str] = []

    def forbidden_numeric(*_args: object, **_kwargs: object) -> bool:
        reached.append("numeric")
        raise AssertionError("changed numerical operation was invoked")

    class ObservedState(dict[str, object]):
        def items(self):  # type: ignore[override]
            reached.append("mapping")
            monkeypatch.setattr(np, "all", forbidden_numeric)
            return super().items()

    with pytest.raises(ElementCapabilityError, match="NUMERICAL_AUTHORITY_MISMATCH"):
        validate_committed_current_tangent_inputs(
            model,
            displacement,
            {1: ObservedState(state)},
            3,
            context="test nested state observation ordering",
        )
    assert reached == ["mapping"]
