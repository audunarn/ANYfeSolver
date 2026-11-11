"""Focused contract tests for solver-owned native multiplicative rotations."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from anyfesolver._native_rotation_state import (
    AcceptedConfigurationMismatchError,
    NativeRotationStateStore,
    NativeRotationTransactionError,
    NativeRotationValidationError,
    StaleNativeRotationTokenError,
    create_native_rotation_state_store,
    rotation_exponential,
    validate_proper_rotation_matrices,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import ShellElement
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.e4_pl_s3_state import (
    reconstruct_director_triad,
    seal_committed_s3_state,
)
from anyfesolver.nonlinear_state import (
    NonlinearStateStore,
    StateMaterializationPolicy,
    begin_state_evaluation,
    commit_state_candidate,
    create_model_native_rotation_store,
    discard_active_state_candidate,
    finish_state_evaluation,
    materialize_state_mapping,
)


class _NativeStub(ShellElement):
    formulation_native_total_lagrangian = True

    def compute_stiffness_matrix(self, mesh, material):  # pragma: no cover
        raise NotImplementedError

    def native_reference_directors(self, mesh) -> np.ndarray:
        del mesh
        return np.asarray(((0.0, 0.0, 1.0),) * len(self.node_ids), dtype=float)


def _coordinates() -> np.ndarray:
    return np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=float,
    )


def _store(*, node_ids=(3, 1, 2)) -> NativeRotationStateStore:
    return NativeRotationStateStore(
        node_ids,
        rotational_dofs={1: (3, 4, 5), 2: (9, 10, 11), 3: (15, 16, 17)},
        coordinate_rows={1: 0, 2: 1, 3: 2},
        committed_full_displacement=np.zeros(24),
        committed_full_coordinates=_coordinates(),
    )


def test_factory_has_true_no_native_node_fast_path() -> None:
    class MustNotBeTouched:
        def __array__(self, *args, **kwargs):  # pragma: no cover - must stay unused
            raise AssertionError("empty native-node path touched an array input")

    assert (
        create_native_rotation_state_store(
            (),
            committed_full_displacement=MustNotBeTouched(),
            committed_full_coordinates=MustNotBeTouched(),
        )
        is None
    )


def test_trial_is_sorted_node_shared_and_relative_to_committed_base() -> None:
    store = _store()
    assert store.node_ids == (1, 2, 3)

    trial = np.zeros(24)
    trial[3:6] = (0.3, -0.2, 0.1)
    trial[9:12] = (-0.1, 0.4, 0.2)
    trial[15:18] = (0.2, 0.1, -0.3)
    moved = _coordinates() + np.asarray((0.5, -0.2, 0.8))
    token = store.begin_trial(trial, moved)

    expected = np.stack(
        (
            rotation_exponential(trial[3:6]),
            rotation_exponential(trial[9:12]),
            rotation_exponential(trial[15:18]),
        )
    )
    view = store.element_view(
        10,
        (3, 1, 2),
        np.asarray(((0.0, 0.0, 1.0),) * 3),
        trial_token=token,
    )
    np.testing.assert_allclose(view.rotation_matrices, expected[[2, 0, 1]], atol=2e-15)
    np.testing.assert_array_equal(view.coordinates, moved[[2, 0, 1]])
    np.testing.assert_array_equal(view.committed_coordinates, _coordinates()[[2, 0, 1]])
    np.testing.assert_array_equal(
        view.coordinate_increment,
        moved[[2, 0, 1]] - _coordinates()[[2, 0, 1]],
    )
    np.testing.assert_array_equal(
        view.rotation_coordinate_increment,
        trial[np.asarray(((15, 16, 17), (3, 4, 5), (9, 10, 11)))],
    )
    assert view.node_ids == (3, 1, 2)
    assert view.generation == 0
    assert view.trial_serial == token.serial

    assert store.commit(token, trial, moved) == 1
    np.testing.assert_array_equal(store.committed_rotation_matrices, expected)
    np.testing.assert_array_equal(store.committed_coordinates, moved[[0, 1, 2]])


def test_two_committed_steps_are_multiplicative_and_noncommutative() -> None:
    store = _store(node_ids=(1,))
    coordinates = _coordinates()
    first = np.zeros(24)
    first[3] = 0.7
    first_token = store.begin_trial(first, coordinates)
    store.commit(first_token, first.copy(), coordinates.copy())

    second = first.copy()
    second[4] += 0.5
    second_token = store.begin_trial(second, coordinates)
    store.commit(second_token, second, coordinates)

    rotate_x = rotation_exponential((0.7, 0.0, 0.0))
    rotate_y = rotation_exponential((0.0, 0.5, 0.0))
    np.testing.assert_allclose(
        store.committed_rotation_matrices[0],
        rotate_y @ rotate_x,
        atol=2e-15,
    )
    assert not np.allclose(
        store.committed_rotation_matrices[0],
        rotate_x @ rotate_y,
        atol=1e-4,
    )
    assert not np.allclose(
        store.committed_rotation_matrices[0],
        rotation_exponential((0.7, 0.5, 0.0)),
        atol=1e-4,
    )


def test_rejected_candidate_and_exception_leave_committed_state_unchanged() -> None:
    store = _store(node_ids=(1,))
    baseline_coordinates = store.committed_coordinates.copy()
    baseline_rotations = store.committed_rotation_matrices.copy()
    candidate_displacement = np.zeros(24)
    candidate_displacement[3:6] = (0.0, 0.0, 1.1)
    candidate_coordinates = _coordinates().copy()
    candidate_coordinates[0, 2] = 0.25

    with pytest.raises(RuntimeError, match="reject this line-search point"):
        with store.candidate(candidate_displacement, candidate_coordinates):
            raise RuntimeError("reject this line-search point")

    assert store.has_active_trial is False
    assert store.generation == 0
    np.testing.assert_array_equal(store.committed_coordinates, baseline_coordinates)
    np.testing.assert_array_equal(store.committed_rotation_matrices, baseline_rotations)

    token = store.begin_trial(candidate_displacement, candidate_coordinates)
    store.discard_trial(token)
    with pytest.raises(StaleNativeRotationTokenError):
        store.validate_trial_token(token)
    np.testing.assert_array_equal(store.committed_rotation_matrices, baseline_rotations)


def test_exclusive_generation_tokens_and_stale_or_foreign_rejection() -> None:
    store = _store(node_ids=(1,))
    other = _store(node_ids=(1,))
    displacement = np.zeros(24)
    coordinates = _coordinates()
    token = store.begin_trial(displacement, coordinates)

    with pytest.raises(NativeRotationTransactionError, match="already active"):
        store.begin_trial(displacement, coordinates)
    with pytest.raises(StaleNativeRotationTokenError):
        other.validate_trial_token(token)

    store.commit(token, displacement, coordinates)
    with pytest.raises(StaleNativeRotationTokenError):
        store.element_view(
            1,
            (1,),
            ((0.0, 0.0, 1.0),),
            trial_token=token,
        )

    next_token = store.begin_trial(displacement, coordinates)
    assert next_token.generation == 1
    assert next_token.serial > token.serial
    store.discard(next_token)


def test_commit_requires_exact_full_displacement_and_coordinate_match() -> None:
    store = _store(node_ids=(1,))
    displacement = np.zeros(24)
    displacement[3:6] = (0.2, 0.3, 0.4)
    coordinates = _coordinates()

    with store.candidate(displacement, coordinates) as candidate:
        wrong_displacement = displacement.copy()
        # This is deliberately a non-native translational DOF.  Native-only
        # comparison would miss it, while the full accepted-vector gate must not.
        wrong_displacement[23] = 1.0e-30
        with pytest.raises(AcceptedConfigurationMismatchError, match="displacement"):
            candidate.commit(wrong_displacement, coordinates)
        assert store.has_active_trial is True
    assert store.has_active_trial is False

    with store.candidate(displacement, coordinates) as candidate:
        wrong_coordinates = coordinates.copy()
        # Row 3 is not one of this store's native coordinate rows.
        wrong_coordinates[3, 0] = 1.0e-30
        with pytest.raises(AcceptedConfigurationMismatchError, match="coordinates"):
            candidate.commit(displacement, wrong_coordinates)
    assert store.generation == 0


def test_shared_node_views_preserve_distinct_crease_directors_and_are_immutable() -> None:
    store = _store()
    displacement = np.zeros(24)
    displacement[9:12] = (0.4, -0.3, 0.2)
    coordinates = _coordinates()
    token = store.begin_trial(displacement, coordinates)

    first = store.element_view(
        "left",
        (1, 2, 3),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        trial_token=token,
    )
    second = store.element_view(
        "right",
        (2, 1, 3),
        ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        trial_token=token,
    )
    np.testing.assert_array_equal(first.rotation_matrices[1], second.rotation_matrices[0])
    np.testing.assert_array_equal(first.coordinates[1], second.coordinates[0])
    np.testing.assert_array_equal(first.reference_directors[1], (0.0, 0.0, 1.0))
    np.testing.assert_array_equal(second.reference_directors[0], (0.0, 1.0, 0.0))
    np.testing.assert_allclose(
        first.trial_directors[1],
        first.trial_rotation_matrices[1] @ first.reference_directors[1],
        atol=2e-15,
    )
    np.testing.assert_allclose(
        second.trial_directors[0],
        second.trial_rotation_matrices[0] @ second.reference_directors[0],
        atol=2e-15,
    )
    assert not np.allclose(first.trial_directors[1], second.trial_directors[0])

    for array in (
        first.coordinates,
        first.rotation_matrices,
        first.reference_directors,
        first.committed_coordinates,
        first.coordinate_increment,
        first.committed_rotation_coordinates,
        first.trial_rotation_coordinates,
        first.rotation_coordinate_increment,
        first.committed_rotation_matrices,
        first.committed_directors,
        first.trial_directors,
        store.committed_coordinates,
        store.committed_rotation_matrices,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(ValueError):
        first.rotation_matrices[0, 0, 0] = 7.0
    store.discard(token)


@pytest.mark.parametrize(
    "bad_matrix",
    (
        np.diag((1.0, 1.0, -1.0)),
        np.asarray(((1.0, 0.2, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        np.full((3, 3), np.nan),
    ),
)
def test_proper_rotation_validation_rejects_reflection_drift_and_nonfinite(
    bad_matrix: np.ndarray,
) -> None:
    with pytest.raises(NativeRotationValidationError):
        validate_proper_rotation_matrices(bad_matrix)
    with pytest.raises(NativeRotationValidationError):
        NativeRotationStateStore(
            (1,),
            rotational_dofs={1: (3, 4, 5)},
            coordinate_rows={1: 0},
            committed_full_displacement=np.zeros(6),
            committed_full_coordinates=np.zeros((1, 3)),
            committed_rotation_matrices={1: bad_matrix},
        )


def test_proper_rotation_validation_accepts_exponential_and_no_silent_projection() -> None:
    matrix = rotation_exponential((2.4, -0.8, 1.3))
    checked = validate_proper_rotation_matrices(matrix)
    np.testing.assert_array_equal(checked, matrix)
    assert checked.flags.writeable is False
    assert np.linalg.det(checked) == pytest.approx(1.0, abs=2e-15)


def _model_bound_store() -> tuple[FEModel, NonlinearStateStore]:
    model = FEModel("native-rotation-parent-transaction")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 0.0, 1.0, 0.0)
    full_displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    coordinate_node_ids = tuple(model.mesh.nodes)
    full_coordinates = np.asarray(
        [model.mesh.nodes[node_id].coords() for node_id in coordinate_node_ids]
    )
    rotation_store = NativeRotationStateStore(
        (1, 2, 3),
        rotational_dofs={
            node_id: tuple(model.mesh.nodes[node_id].dofs[3:6])
            for node_id in (1, 2, 3)
        },
        coordinate_rows={node_id: row for row, node_id in enumerate(coordinate_node_ids)},
        coordinate_node_ids=coordinate_node_ids,
        committed_full_displacement=full_displacement,
        committed_full_coordinates=full_coordinates,
    )
    state_store = NonlinearStateStore.from_shell_layouts(
        (), {7: {"history": np.asarray([1.0])}}
    )
    state_store.attach_native_rotation_store(rotation_store)
    return model, state_store


def test_parent_state_transaction_atomically_binds_material_and_native_kinematics() -> None:
    model, store = _model_bound_store()
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    displacement[model.mesh.nodes[1].dofs[:3]] = (0.2, -0.1, 0.05)
    displacement[model.mesh.nodes[2].dofs[3:6]] = (0.4, -0.2, 0.1)
    token = begin_state_evaluation(store, model=model, displacements=displacement)
    assert token is not None
    view = store.native_element_rotation_view(
        token,
        7,
        (1, 2, 3),
        np.asarray(((0.0, 0.0, 1.0),) * 3),
    )
    np.testing.assert_array_equal(view.coordinate_increment[0], (0.2, -0.1, 0.05))
    np.testing.assert_allclose(
        view.trial_rotation_matrices[1],
        rotation_exponential((0.4, -0.2, 0.1)),
        atol=2.0e-15,
    )
    candidate = finish_state_evaluation(
        store,
        token,
        {7: {"history": np.asarray([2.0])}},
    )

    wrong = displacement.copy()
    wrong[-1] = 1.0e-30
    with pytest.raises(AcceptedConfigurationMismatchError):
        commit_state_candidate(
            store,
            candidate,
            model=model,
            accepted_full_displacement=wrong,
        )
    np.testing.assert_array_equal(store[7]["history"], (1.0,))
    assert store.generation == 0
    assert store.native_rotation_store is not None
    assert store.native_rotation_store.generation == 0
    discard_active_state_candidate(store)

    token = begin_state_evaluation(store, model=model, displacements=displacement)
    candidate = finish_state_evaluation(
        store,
        token,
        {7: {"history": np.asarray([2.0])}},
    )
    committed = commit_state_candidate(
        store,
        candidate,
        model=model,
        accepted_full_displacement=displacement.copy(),
    )
    assert committed is store
    np.testing.assert_array_equal(store[7]["history"], (2.0,))
    assert store.generation == 1
    assert store.native_rotation_store.generation == 1


def test_non_native_parent_store_preserves_model_free_legacy_transaction_api() -> None:
    store = NonlinearStateStore.from_shell_layouts(
        (), {1: {"history": np.asarray([1.0])}}
    )
    token = begin_state_evaluation(store)
    candidate = finish_state_evaluation(
        store,
        token,
        {1: {"history": np.asarray([3.0])}},
    )
    assert commit_state_candidate(store, candidate) is store
    np.testing.assert_array_equal(store[1]["history"], (3.0,))


def _two_native_element_model() -> FEModel:
    model = FEModel("shared-native-rotation-restart")
    for node_id, coordinates in {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (0.0, 1.0, 0.0),
        4: (1.0, 1.0, 0.0),
    }.items():
        model.add_node(node_id, *coordinates)
    model.add_element(1, _NativeStub(1, [1, 2, 3], "steel"))
    model.add_element(2, _NativeStub(2, [2, 4, 3], "steel"))
    return model


def test_model_builder_reconstructs_and_cross_checks_shared_restart_operators() -> None:
    model = _two_native_element_model()
    full = np.zeros(model.mesh.dof_manager.total_dofs)
    rotations = {
        node_id: rotation_exponential((0.03 * node_id, -0.01 * node_id, 0.02))
        for node_id in model.mesh.nodes
    }
    states = {
        1: {
            "committed_total_u": full[
                model.mesh.elements[1].get_dof_mapping(model.mesh)
            ],
            "committed_nodal_rotation_matrices": np.asarray(
                [rotations[node_id] for node_id in (1, 2, 3)]
            ),
        },
        2: {
            "committed_total_u": full[
                model.mesh.elements[2].get_dof_mapping(model.mesh)
            ],
            "committed_nodal_rotation_matrices": np.asarray(
                [rotations[node_id] for node_id in (2, 4, 3)]
            ),
        },
    }
    store = create_model_native_rotation_store(model, states, full)
    assert store is not None
    assert store.node_ids == (1, 2, 3, 4)
    np.testing.assert_array_equal(store.committed_rotation_matrices[1], rotations[2])

    conflicting = {
        key: {field: np.asarray(value).copy() for field, value in state.items()}
        for key, state in states.items()
    }
    conflicting[2]["committed_nodal_rotation_matrices"][0] = rotation_exponential(
        (0.8, 0.0, 0.0)
    )
    with pytest.raises(NativeRotationValidationError, match="conflicting"):
        create_model_native_rotation_store(model, conflicting, full)


def test_model_builder_rejects_nonzero_additive_restart_without_rotation_history() -> None:
    model = _two_native_element_model()
    full = np.zeros(model.mesh.dof_manager.total_dofs)
    full[model.mesh.nodes[2].dofs[4]] = 0.25
    with pytest.raises(NativeRotationValidationError, match="no multiplicative"):
        create_model_native_rotation_store(model, {}, full)


def _qualified_s3_state_model(
    *,
    two_elements: bool = False,
) -> tuple[FEModel, NonlinearStateStore, dict[int, dict[str, object]], np.ndarray]:
    model = FEModel("qualified-s3-native-state-consistency")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    coordinates = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (0.0, 1.0, 0.0),
        4: (1.0, 1.0, 0.0),
    }
    for node_id in ((1, 2, 3, 4) if two_elements else (1, 2, 3)):
        model.add_node(node_id, *coordinates[node_id])
    elements = {
        1: QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.012,
            reference_normal=[0.0, 0.0, 1.0],
        )
    }
    if two_elements:
        elements[2] = QualifiedE4PLS3ShellElement(
            2,
            [2, 4, 3],
            "steel",
            thickness=0.012,
            reference_normal=[0.0, 0.0, 1.0],
        )
    material = model.get_material("steel")
    states: dict[int, dict[str, object]] = {}
    for element_id, element in elements.items():
        model.add_element(element_id, element)
        states[element_id] = element.init_model_bound_nonlinear_state(
            model.mesh,
            material,
            3,
        )
    full = np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64)
    rotation_store = create_model_native_rotation_store(model, states, full)
    assert rotation_store is not None
    store = NonlinearStateStore.from_shell_layouts((), states)
    store.attach_native_rotation_store(rotation_store)
    return model, store, states, full


def _accepted_s3_state(
    committed: dict[str, object],
    view: object,
    accepted_element_u: np.ndarray,
) -> dict[str, object]:
    result = copy.deepcopy(committed)
    result["committed_total_u"] = np.asarray(
        accepted_element_u, dtype=np.float64
    ).copy()
    result["reference_corner_directors"] = np.asarray(
        view.reference_directors, dtype=np.float64
    ).copy()
    result["committed_nodal_rotation_matrices"] = np.asarray(
        view.trial_rotation_matrices, dtype=np.float64
    ).copy()
    triads = np.asarray(result["committed_director_triads"], dtype=np.float64).copy()
    for node in range(3):
        triads[node] = reconstruct_director_triad(view.trial_directors[node])
    result["committed_director_triads"] = triads
    return seal_committed_s3_state(result)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("committed_total_u", "committed_total_u"),
        ("reference_corner_directors", "reference_corner_directors"),
        (
            "committed_nodal_rotation_matrices",
            "committed_nodal_rotation_matrices",
        ),
        ("committed_director_triads", "corner director normals"),
    ),
)
def test_parent_transaction_rejects_s3_state_disagreeing_with_native_view(
    field: str,
    message: str,
) -> None:
    model, store, initial, displacement = _qualified_s3_state_model()
    displacement[model.mesh.nodes[1].dofs[:3]] = (0.05, -0.02, 0.01)
    displacement[model.mesh.nodes[2].dofs[3:6]] = (0.2, -0.1, 0.05)
    token = begin_state_evaluation(store, model=model, displacements=displacement)
    assert token is not None
    element = model.mesh.elements[1]
    view = store.native_element_rotation_view(
        token,
        1,
        element.node_ids,
        element.native_reference_directors(model.mesh),
    )
    mapping = element.get_dof_mapping(model.mesh)
    trial_state = _accepted_s3_state(initial[1], view, displacement[mapping])
    if field == "committed_total_u":
        trial_state[field][0] = np.nextafter(trial_state[field][0], 1.0)
    elif field == "reference_corner_directors":
        trial_state[field][0, 0] = np.nextafter(trial_state[field][0, 0], 1.0)
    elif field == "committed_nodal_rotation_matrices":
        trial_state[field][0] = rotation_exponential((0.1, 0.0, 0.0))
    else:
        trial_state[field][0, 0, 2] += 1.0e-6

    with pytest.raises(NativeRotationValidationError, match=message):
        finish_state_evaluation(store, token, {1: trial_state})
    assert store.generation == 0
    assert store.native_rotation_store is not None
    assert store.native_rotation_store.generation == 0
    np.testing.assert_array_equal(
        store[1]["committed_total_u"], initial[1]["committed_total_u"]
    )
    discard_active_state_candidate(store)


def test_committed_restart_preserves_exact_deletion_state_for_frozen_s3() -> None:
    model, store, initial, displacement = _qualified_s3_state_model(
        two_elements=True
    )
    store.freeze_deleted((2,))
    displacement[model.mesh.nodes[2].dofs[:3]] = (0.03, -0.01, 0.02)
    displacement[model.mesh.nodes[2].dofs[3:6]] = (0.25, -0.15, 0.1)
    displacement[model.mesh.nodes[3].dofs[3:6]] = (-0.1, 0.05, 0.2)
    token = begin_state_evaluation(store, model=model, displacements=displacement)
    assert token is not None
    first = model.mesh.elements[1]
    first_view = store.native_element_rotation_view(
        token,
        1,
        first.node_ids,
        first.native_reference_directors(model.mesh),
    )
    first_mapping = first.get_dof_mapping(model.mesh)
    candidate = finish_state_evaluation(
        store,
        token,
        {
            1: _accepted_s3_state(
                initial[1],
                first_view,
                displacement[first_mapping],
            )
        },
    )
    assert (
        commit_state_candidate(
            store,
            candidate,
            model=model,
            accepted_full_displacement=displacement.copy(),
        )
        is store
    )

    restart = materialize_state_mapping(
        store,
        policy=StateMaterializationPolicy.RESTART,
    )
    material = model.get_material("steel")
    active = model.mesh.elements[1]
    active_mapping = active.get_dof_mapping(model.mesh)
    np.testing.assert_array_equal(
        restart[1]["committed_total_u"], displacement[active_mapping]
    )
    np.testing.assert_array_equal(
        store[1]["committed_total_u"], displacement[active_mapping]
    )
    active.validate_model_bound_nonlinear_state(
        model.mesh,
        material,
        restart[1],
        3,
        expected_committed_total_u=displacement[active_mapping],
    )

    # A deleted element is constitutively frozen at its last accepted active
    # state.  Advancing a shared node and committing an active neighbour must
    # not silently rebind that frozen history to the later global displacement
    # or to the neighbour's newer nodal rotation matrices.
    np.testing.assert_array_equal(
        restart[2]["committed_total_u"], initial[2]["committed_total_u"]
    )
    np.testing.assert_array_equal(
        store[2]["committed_total_u"], initial[2]["committed_total_u"]
    )
    np.testing.assert_array_equal(
        restart[2]["committed_nodal_rotation_matrices"],
        initial[2]["committed_nodal_rotation_matrices"],
    )
    np.testing.assert_array_equal(
        store[2]["committed_nodal_rotation_matrices"],
        initial[2]["committed_nodal_rotation_matrices"],
    )
    np.testing.assert_array_equal(restart[2]["alpha"], initial[2]["alpha"])
