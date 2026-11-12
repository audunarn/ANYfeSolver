"""Frozen solver-chart and node-shared SO(3) tests for GE-Beam3 P2."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anyfesolver._ge_beam3_mixed_ad import RotationDomainError
from anyfesolver._native_rotation_state import (
    NativeRotationStateStore,
    create_native_rotation_state_store,
    rotation_exponential,
)
from anyfesolver.beam_sections import GeneralizedBeamSection
from anyfesolver.fe_core import FEModel
from anyfesolver.ge_beam3_mixed_element import (
    GeBeam3MixedLocalSolveError,
    GeometricallyExactBeam3D3NElement,
)
from anyfesolver.ge_beam3_mixed_state import (
    GeBeam3MixedCommittedStateError,
    deserialize_ge_beam3_mixed_state,
    seal_committed_ge_beam3_mixed_state,
    serialize_ge_beam3_mixed_state,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "docs" / "reference_cases" / "ge_beam3_mixed_p2_cases.json"


def _manifest() -> dict[str, Any]:
    return json.loads(CASES.read_text(encoding="utf-8"))


def _record(group: str, record_id: str) -> dict[str, Any]:
    records = _manifest()["cases"][group]
    id_key = {"geometry": "geometry_id", "sections": "section_id"}[group]
    return next(record for record in records if record[id_key] == record_id)


def _solver_case(case_id: str) -> dict[str, Any]:
    return next(
        record
        for record in _manifest()["cases"]["solver_state"]
        if record["case_id"] == case_id
    )


def _floats(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _section(section_id: str) -> GeneralizedBeamSection:
    record = _record("sections", section_id)
    return GeneralizedBeamSection(
        stiffness=_floats(record["stiffness_matrix"]),
        mass_matrix=_floats(record["mass_matrix_per_reference_length"]),
        name=section_id,
    )


def _model_element(
    geometry_id: str,
    section_id: str,
    *,
    element_id: int = 1,
) -> tuple[FEModel, GeometricallyExactBeam3D3NElement]:
    geometry = _record("geometry", geometry_id)
    model = FEModel(f"ge-beam3-p2-{geometry_id.lower()}")
    model.add_material("mat", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in zip(geometry["connectivity"], geometry["nodes"]):
        model.add_node(int(node_id), *map(float, coordinates))
    element = GeometricallyExactBeam3D3NElement(
        element_id,
        geometry["connectivity"],
        "mat",
        section=_section(section_id),
        reference_orientation=_floats(geometry["reference_orientation"]),
        reference_axis_direction=_floats(geometry["reference_axis_direction"]),
    )
    return model, element


def _pack(translations: Any, rotations: Any) -> np.ndarray:
    made = np.empty((3, 6), dtype=np.float64)
    made[:, :3] = _floats(translations)
    made[:, 3:] = _floats(rotations)
    return made.reshape(18)


def _store(
    element: GeometricallyExactBeam3D3NElement,
    model: FEModel,
    committed_u: np.ndarray,
    committed_operators: Any,
) -> NativeRotationStateStore:
    nodes = tuple(int(node) for node in element.node_ids)
    reference = element.get_node_coordinates(model.mesh)
    total = np.asarray(committed_u, dtype=np.float64).reshape(3, 6)
    made = create_native_rotation_state_store(
        nodes,
        rotational_dofs={node: (6 * row + 3, 6 * row + 4, 6 * row + 5) for row, node in enumerate(nodes)},
        coordinate_rows={node: row for row, node in enumerate(nodes)},
        committed_full_displacement=total.reshape(18),
        committed_full_coordinates=reference + total[:, :3],
        committed_rotation_matrices=np.asarray(committed_operators, dtype=np.float64),
        coordinate_node_ids=nodes,
    )
    assert isinstance(made, NativeRotationStateStore)
    return made


def _committed_state(
    element: GeometricallyExactBeam3D3NElement,
    model: FEModel,
    committed_u: np.ndarray,
    committed_operators: Any,
) -> dict[str, Any]:
    # This private P2 helper deliberately exercises the formulation-owned replay
    # path: nonzero committed kinematics must carry their recomputed local fields,
    # not a zero-field state with a merely updated displacement seal.
    return element._state_from_configuration(
        model.mesh,
        np.asarray(committed_u),
        np.asarray(committed_operators),
    )


def _trial_coordinates(
    element: GeometricallyExactBeam3D3NElement,
    model: FEModel,
    trial_u: np.ndarray,
) -> np.ndarray:
    return element.get_node_coordinates(model.mesh) + np.asarray(trial_u).reshape(3, 6)[:, :3]


def _evaluate_trial(
    element: GeometricallyExactBeam3D3NElement,
    model: FEModel,
    store: NativeRotationStateStore,
    committed_state: dict[str, Any],
    trial_u: np.ndarray,
    *,
    tangent: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    coordinates = _trial_coordinates(element, model, trial_u)
    with store.candidate(trial_u, coordinates) as candidate:
        view = candidate.element_view(
            int(element.element_id),
            element.node_ids,
            element.native_reference_directors(model.mesh),
        )
        force, matrix, state = element.compute_nonlinear_response(
            model.mesh,
            model.get_material("mat"),
            trial_u,
            committed_state,
            1,
            tangent,
            native_rotation_trial=view,
        )
    return np.asarray(force), None if matrix is None else np.asarray(matrix), state


def _commit_trial(
    element: GeometricallyExactBeam3D3NElement,
    model: FEModel,
    store: NativeRotationStateStore,
    committed_state: Mapping[str, Any],
    trial_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate and atomically accept one frozen solver-owned candidate."""

    coordinates = _trial_coordinates(element, model, trial_u)
    with store.candidate(trial_u, coordinates) as candidate:
        view = candidate.element_view(
            int(element.element_id),
            element.node_ids,
            element.native_reference_directors(model.mesh),
        )
        force, matrix, state = element.compute_nonlinear_response(
            model.mesh,
            model.get_material("mat"),
            trial_u,
            committed_state,
            1,
            True,
            native_rotation_trial=view,
        )
        assert matrix is not None
        candidate.commit(trial_u, coordinates)
    return np.asarray(force), np.asarray(matrix), state


def _snapshot(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return ("ndarray", array.dtype.str, array.shape, array.tobytes())
    if isinstance(value, Mapping):
        return tuple((key, _snapshot(value[key])) for key in sorted(value))
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    return copy.deepcopy(value)


def test_nonzero_solver_chart_directional_tangent_uses_additive_increment_chart() -> None:
    case = _solver_case("NONZERO_SOLVER_CHART")
    model, element = _model_element(case["geometry_id"], case["section_id"])
    committed_u = _pack(
        case["committed_translations"], case["committed_rotation_coordinates"]
    )
    trial_u = _pack(case["trial_translations"], case["trial_rotation_coordinates"])
    committed_operators = _floats(case["committed_spatial_operators"])
    committed_state = _committed_state(
        element, model, committed_u, committed_operators
    )
    store = _store(element, model, committed_u, committed_operators)

    force, tangent, _state = _evaluate_trial(
        element, model, store, committed_state, trial_u
    )
    assert tangent is not None
    direction = _floats(case["direction"]).reshape(18)
    direction /= np.linalg.norm(direction)
    step = 2.0e-6
    plus, _unused, _plus_state = _evaluate_trial(
        element,
        model,
        store,
        committed_state,
        trial_u + step * direction,
        tangent=False,
    )
    minus, _unused, _minus_state = _evaluate_trial(
        element,
        model,
        store,
        committed_state,
        trial_u - step * direction,
        tangent=False,
    )
    finite = (plus - minus) / (2.0 * step)
    analytic = tangent @ direction
    relative = np.linalg.norm(finite - analytic) / max(
        1.0, np.linalg.norm(finite), np.linalg.norm(analytic)
    )
    assert np.all(np.isfinite(force))
    assert relative <= float(case["expected"]["directional_tangent_relative_max"])


def test_two_noncommuting_commits_and_split_restart_are_bitwise_identical() -> None:
    case = _solver_case("TWO_NONCOMMUTING_COMMITS")
    model, element = _model_element(case["geometry_id"], case["section_id"])
    zero = np.zeros(18)
    initial = _floats(case["initial_spatial_operators"])
    increments = [_floats(value) for value in case["increments"]]

    def make_store() -> NativeRotationStateStore:
        return _store(element, model, zero, initial)

    continuous = make_store()
    first_u = _pack(np.zeros((3, 3)), increments[0])
    with continuous.candidate(
        first_u, _trial_coordinates(element, model, first_u)
    ) as candidate:
        candidate.commit(first_u, _trial_coordinates(element, model, first_u))
    first_operators = np.array(continuous.committed_rotation_matrices, copy=True)
    second_u = _pack(np.zeros((3, 3)), increments[0] + increments[1])
    with continuous.candidate(
        second_u, _trial_coordinates(element, model, second_u)
    ) as candidate:
        candidate.commit(second_u, _trial_coordinates(element, model, second_u))

    expected = np.asarray(
        [
            rotation_exponential(increments[1][node])
            @ rotation_exponential(increments[0][node])
            @ initial[node]
            for node in range(3)
        ]
    )
    np.testing.assert_allclose(
        continuous.committed_rotation_matrices,
        expected,
        rtol=0.0,
        atol=3.0e-16,
    )

    restarted = _store(element, model, first_u, first_operators)
    with restarted.candidate(
        second_u, _trial_coordinates(element, model, second_u)
    ) as candidate:
        candidate.commit(second_u, _trial_coordinates(element, model, second_u))
    np.testing.assert_array_equal(
        restarted.committed_rotation_matrices,
        continuous.committed_rotation_matrices,
    )

    opposite = make_store()
    opposite_first = _pack(np.zeros((3, 3)), increments[1])
    with opposite.candidate(
        opposite_first, _trial_coordinates(element, model, opposite_first)
    ) as candidate:
        candidate.commit(
            opposite_first, _trial_coordinates(element, model, opposite_first)
        )
    opposite_second = _pack(np.zeros((3, 3)), increments[1] + increments[0])
    with opposite.candidate(
        opposite_second, _trial_coordinates(element, model, opposite_second)
    ) as candidate:
        candidate.commit(
            opposite_second, _trial_coordinates(element, model, opposite_second)
        )
    difference = np.linalg.norm(
        opposite.committed_rotation_matrices
        - continuous.committed_rotation_matrices
    )
    assert difference > float(
        case["expected"]["opposite_order_must_differ_by_frobenius_more_than"]
    )


def test_split_restart_replays_both_frozen_increments_and_rejects_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _solver_case("SPLIT_RESTART_IDENTITY")
    model, element = _model_element(case["geometry_id"], case["section_id"])
    identity = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    zero = np.zeros(18)
    increments = [
        _pack(record["translations"], record["rotations"])
        for record in case["increments"]
    ]
    first_u = increments[0]
    final_u = increments[0] + increments[1]

    continuous_store = _store(element, model, zero, identity)
    continuous_state = element.init_model_bound_nonlinear_state(
        model.mesh, model.get_material("mat"), 1
    )
    _first_force, _first_tangent, continuous_state = _commit_trial(
        element,
        model,
        continuous_store,
        continuous_state,
        first_u,
    )
    continuous_force, continuous_tangent, continuous_final = _commit_trial(
        element,
        model,
        continuous_store,
        continuous_state,
        final_u,
    )

    split_store = _store(element, model, zero, identity)
    split_state = element.init_model_bound_nonlinear_state(
        model.mesh, model.get_material("mat"), 1
    )
    _split_first_force, _split_first_tangent, split_state = _commit_trial(
        element,
        model,
        split_store,
        split_state,
        first_u,
    )
    checkpoint = serialize_ge_beam3_mixed_state(split_state)
    restarted_state = deserialize_ge_beam3_mixed_state(checkpoint)
    assert serialize_ge_beam3_mixed_state(restarted_state) == checkpoint
    restarted_store = _store(
        element,
        model,
        np.asarray(restarted_state["committed_total_u"]),
        np.asarray(restarted_state["committed_nodal_rotation_matrices"]),
    )
    split_force, split_tangent, split_final = _commit_trial(
        element,
        model,
        restarted_store,
        restarted_state,
        final_u,
    )

    assert serialize_ge_beam3_mixed_state(split_final) == serialize_ge_beam3_mixed_state(
        continuous_final
    )
    for continuous, split in (
        (continuous_force, split_force),
        (continuous_tangent, split_tangent),
    ):
        error = np.linalg.norm(continuous - split, ord=np.inf) / max(
            1.0,
            np.linalg.norm(continuous, ord=np.inf),
            np.linalg.norm(split, ord=np.inf),
        )
        assert error <= 1.0e-11

    mutated = dict(restarted_state)
    changed_geometry = np.array(mutated["reference_geometry"], copy=True)
    changed_geometry[0, 0] += 0.125
    mutated["reference_geometry"] = changed_geometry
    mutated = seal_committed_ge_beam3_mixed_state(mutated)

    def mechanics_must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("mutated restart reached element mechanics")

    monkeypatch.setattr(element, "evaluate_candidate", mechanics_must_not_run)
    with pytest.raises(
        GeBeam3MixedCommittedStateError,
        match="reference geometry bit pattern mismatch",
    ):
        element.validate_model_bound_nonlinear_state(
            model.mesh,
            model.get_material("mat"),
            mutated,
            1,
            expected_committed_total_u=first_u,
        )


def test_rejected_trial_discards_state_and_cutback_starts_from_committed_base() -> None:
    case = _solver_case("REJECTED_TRIAL_ROLLBACK_AND_CUTBACK")
    model, element = _model_element(case["geometry_id"], case["section_id"])
    zero_u = np.zeros(18)
    identity = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    store = _store(element, model, zero_u, identity)
    state = element.init_model_bound_nonlinear_state(
        model.mesh, model.get_material("mat"), 1
    )

    accepted_u = _pack(
        case["accepted_translations"], case["accepted_rotation_coordinates"]
    )
    accepted_coordinates = _trial_coordinates(element, model, accepted_u)
    with store.candidate(accepted_u, accepted_coordinates) as candidate:
        view = candidate.element_view(
            int(element.element_id),
            element.node_ids,
            element.native_reference_directors(model.mesh),
        )
        _force, _tangent, accepted_state = element.compute_nonlinear_response(
            model.mesh,
            model.get_material("mat"),
            accepted_u,
            state,
            1,
            True,
            native_rotation_trial=view,
        )
        candidate.commit(accepted_u, accepted_coordinates)

    committed_u = np.array(store.committed_full_displacement, copy=True)
    committed_coordinates = np.array(store.committed_full_coordinates, copy=True)
    committed_operators = np.array(store.committed_rotation_matrices, copy=True)
    state_snapshot = _snapshot(accepted_state)
    rejected_u = _pack(
        case["rejected_translations"], case["rejected_rotation_coordinates"]
    )
    rejected_coordinates = _trial_coordinates(element, model, rejected_u)
    with pytest.raises((GeBeam3MixedLocalSolveError, RotationDomainError)):
        with store.candidate(rejected_u, rejected_coordinates) as candidate:
            view = candidate.element_view(
                int(element.element_id),
                element.node_ids,
                element.native_reference_directors(model.mesh),
            )
            element.compute_nonlinear_response(
                model.mesh,
                model.get_material("mat"),
                rejected_u,
                accepted_state,
                1,
                True,
                native_rotation_trial=view,
            )

    np.testing.assert_array_equal(store.committed_full_displacement, committed_u)
    np.testing.assert_array_equal(store.committed_full_coordinates, committed_coordinates)
    np.testing.assert_array_equal(store.committed_rotation_matrices, committed_operators)
    assert _snapshot(accepted_state) == state_snapshot

    cutback_u = accepted_u + 0.05 * (rejected_u - accepted_u)
    with store.candidate(
        cutback_u, _trial_coordinates(element, model, cutback_u)
    ) as candidate:
        view = candidate.element_view(
            int(element.element_id),
            element.node_ids,
            element.native_reference_directors(model.mesh),
        )
        np.testing.assert_array_equal(
            view.committed_rotation_matrices, committed_operators
        )
        np.testing.assert_allclose(
            view.rotation_coordinate_increment,
            (cutback_u - committed_u).reshape(3, 6)[:, 3:],
            rtol=0.0,
            atol=0.0,
        )
    np.testing.assert_array_equal(store.committed_rotation_matrices, committed_operators)


def test_shared_node_has_one_operator_and_distinct_absolute_material_triads() -> None:
    case = _solver_case("SHARED_NODE_DISTINCT_MATERIAL_TRIADS")
    model = FEModel("ge-beam3-p2-shared-node")
    model.add_material("mat", 210.0e9, 0.3, density=7850.0)
    for node, x in enumerate((0.0, 0.5, 1.0, 1.5, 2.0), start=1):
        model.add_node(node, x, 0.0, 0.0)
    section = _section(case["section_id"])
    first = GeometricallyExactBeam3D3NElement(
        1,
        case["elements"][0]["node_ids"],
        "mat",
        section=section,
        reference_orientation=(0.0, 1.0, 0.0),
        reference_axis_direction=(1.0, 0.0, 0.0),
    )
    second = GeometricallyExactBeam3D3NElement(
        2,
        case["elements"][1]["node_ids"],
        "mat",
        section=section,
        reference_orientation=(0.0, 0.0, 1.0),
        reference_axis_direction=(1.0, 0.0, 0.0),
    )
    nodes = (1, 2, 3, 4, 5)
    zero = np.zeros(30)
    coordinates = np.asarray(
        [model.mesh.get_node(node).coords() for node in nodes], dtype=np.float64
    )
    store = create_native_rotation_state_store(
        nodes,
        rotational_dofs={node: (6 * row + 3, 6 * row + 4, 6 * row + 5) for row, node in enumerate(nodes)},
        coordinate_rows={node: row for row, node in enumerate(nodes)},
        committed_full_displacement=zero,
        committed_full_coordinates=coordinates,
        coordinate_node_ids=nodes,
    )
    assert isinstance(store, NativeRotationStateStore)
    trial = zero.copy()
    trial[2 * 6 + 3 : 2 * 6 + 6] = _floats(case["shared_node_increment"])
    with store.candidate(trial, coordinates) as candidate:
        view_first = candidate.element_view(
            1, first.node_ids, first.native_reference_directors(model.mesh)
        )
        view_second = candidate.element_view(
            2, second.node_ids, second.native_reference_directors(model.mesh)
        )
        first_operator = view_first.trial_rotation_matrices[2]
        second_operator = view_second.trial_rotation_matrices[0]
        np.testing.assert_array_equal(first_operator, second_operator)
        first_material = first_operator @ first.reference_triad(model.mesh)
        second_material = second_operator @ second.reference_triad(model.mesh)
        assert not np.array_equal(first_material, second_material)


def test_reversal_covariance_and_zero_chart_preserve_static_core() -> None:
    case = _solver_case("CONNECTIVITY_REVERSAL_STATE")
    forward_model, forward = _model_element(
        case["forward_geometry_id"], case["section_id"]
    )
    reverse_model, reverse = _model_element(
        case["reversed_geometry_id"], case["section_id"]
    )
    forward_u = _pack(
        case["forward_translations"], case["forward_rotation_coordinates"]
    )
    reverse_u = forward_u.reshape(3, 6)[::-1].reshape(18)
    identity = np.repeat(np.eye(3)[None, :, :], 3, axis=0)

    def evaluate(
        model: FEModel,
        element: GeometricallyExactBeam3D3NElement,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        state = element.init_model_bound_nonlinear_state(
            model.mesh, model.get_material("mat"), 1
        )
        store = _store(element, model, np.zeros(18), identity)
        force, matrix, candidate = _evaluate_trial(
            element, model, store, state, values
        )
        assert matrix is not None
        return force, matrix, candidate

    force_f, tangent_f, state_f = evaluate(forward_model, forward, forward_u)
    force_r, tangent_r, state_r = evaluate(reverse_model, reverse, reverse_u)
    permutation = np.zeros((18, 18))
    for new_node, old_node in enumerate((2, 1, 0)):
        permutation[
            6 * new_node : 6 * new_node + 6,
            6 * old_node : 6 * old_node + 6,
        ] = np.eye(6)
    np.testing.assert_allclose(
        permutation.T @ force_r, force_f, rtol=2.0e-10, atol=2.0e-11
    )
    np.testing.assert_allclose(
        permutation.T @ tangent_r @ permutation,
        tangent_f,
        rtol=2.0e-9,
        atol=2.0e-10,
    )
    reversal = forward.REVERSAL_STRAIN_MAP
    np.testing.assert_allclose(
        np.asarray(state_r["station_generalized_strain"])[::-1] @ reversal,
        np.asarray(state_f["station_generalized_strain"]),
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        np.asarray(state_r["station_generalized_resultant"])[::-1] @ reversal,
        np.asarray(state_f["station_generalized_resultant"]),
        rtol=2.0e-10,
        atol=2.0e-11,
    )

    zero_state = forward.init_model_bound_nonlinear_state(
        forward_model.mesh, forward_model.get_material("mat"), 1
    )
    zero_store = _store(forward, forward_model, np.zeros(18), identity)
    static_energy, static_force, static_tangent, static_fields = (
        forward.evaluate_candidate(
            forward_model.mesh,
            np.zeros(18),
            rotation_matrices=np.repeat(
                forward.reference_triad(forward_model.mesh)[None, :, :], 3, axis=0
            ),
            tangent=True,
        )
    )
    solver_force, solver_tangent, solver_state = _evaluate_trial(
        forward,
        forward_model,
        zero_store,
        zero_state,
        np.zeros(18),
    )
    assert static_energy == 0.0
    np.testing.assert_array_equal(solver_force, static_force)
    np.testing.assert_array_equal(solver_tangent, static_tangent)
    np.testing.assert_array_equal(
        solver_state["station_generalized_strain"],
        static_fields["generalized_strain"],
    )
    np.testing.assert_array_equal(
        solver_state["station_generalized_resultant"],
        static_fields["generalized_resultant"],
    )
