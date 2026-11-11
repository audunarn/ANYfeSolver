from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver._native_rotation_state import (
    NativeElementRotationView,
    create_native_rotation_state_store,
    rotation_exponential,
)
from anyfesolver.e4_pl_s3_element import (
    QualifiedE4PLS3ShellElement,
    invariant_drilling_scale,
)
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import (
    DNVC208MaterialCurve,
    LinearHardeningCurve,
    PiecewiseLinearCurve,
    PowerLawHardeningCurve,
)


_REFERENCE = np.asarray(
    ((0.0, 0.0, 0.0), (1.1, 0.0, 0.0), (0.25, 0.95, 0.0)),
    dtype=np.float64,
)


def _model(
    *,
    plastic: bool = False,
    curve: object | None = None,
) -> tuple[FEModel, QualifiedE4PLS3ShellElement]:
    model = FEModel("qualified-s3-direct-native-response")
    if plastic and curve is not None:
        raise ValueError("test model accepts either plastic=True or an explicit curve")
    selected_curve = (
        DNVC208MaterialCurve(
            sigma_prop=245.0e6,
            sigma_yield=250.0e6,
            sigma_yield_2=260.0e6,
            eps_p_y1=0.004,
            eps_p_y2=0.02,
            K=500.0e6,
            n=0.2,
        )
        if plastic
        else curve
    )
    model.add_material(
        "steel",
        210.0e9,
        0.3,
        density=7850.0,
        hardening_curve=selected_curve,
    )
    for node_id, coordinates in enumerate(_REFERENCE, start=1):
        model.add_node(node_id, *coordinates)
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.018,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)
    return model, element


def _view_for_total_u(
    model: FEModel,
    element: QualifiedE4PLS3ShellElement,
    total_u: np.ndarray,
    committed_state: dict[str, object] | None,
) -> NativeElementRotationView:
    trial_u = np.asarray(total_u, dtype=np.float64).reshape(18)
    if committed_state is None:
        committed_u = np.zeros(18, dtype=np.float64)
        committed_q = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    else:
        committed_u = np.asarray(
            committed_state["committed_total_u"], dtype=np.float64
        ).reshape(18)
        committed_q = np.asarray(
            committed_state["committed_nodal_rotation_matrices"],
            dtype=np.float64,
        ).reshape(3, 3, 3)
    committed_coordinates = _REFERENCE + committed_u.reshape(3, 6)[:, :3]
    trial_coordinates = _REFERENCE + trial_u.reshape(3, 6)[:, :3]
    node_ids = tuple(element.node_ids)
    store = create_native_rotation_state_store(
        node_ids,
        rotational_dofs={
            node_id: (6 * row + 3, 6 * row + 4, 6 * row + 5)
            for row, node_id in enumerate(node_ids)
        },
        coordinate_rows={node_id: row for row, node_id in enumerate(node_ids)},
        coordinate_node_ids=node_ids,
        committed_full_displacement=committed_u,
        committed_full_coordinates=committed_coordinates,
        committed_rotation_matrices={
            node_id: committed_q[row] for row, node_id in enumerate(node_ids)
        },
    )
    assert store is not None
    token = store.begin_trial(trial_u, trial_coordinates)
    return store.element_view(
        element.element_id,
        node_ids,
        element.native_reference_directors(model.mesh),
        trial_token=token,
    )


def _direct_response(
    model: FEModel,
    element: QualifiedE4PLS3ShellElement,
    total_u: np.ndarray,
    state: dict[str, object] | None = None,
    *,
    tangent: bool = True,
):
    view = _view_for_total_u(model, element, total_u, state)
    response = element.compute_nonlinear_response(
        model.mesh,
        model.get_material("steel"),
        total_u,
        state,
        3,
        tangent,
        native_rotation_trial=view,
    )
    return (*response, view)


def test_direct_zero_response_recovers_linear_tangent_and_honours_tangent_false() -> None:
    model, element = _model()
    zero = np.zeros(18, dtype=np.float64)
    force, tangent, state, view = _direct_response(model, element, zero)

    np.testing.assert_array_equal(force, np.zeros(18))
    assert tangent is not None
    np.testing.assert_allclose(
        tangent,
        element.compute_stiffness_matrix(model.mesh, model.get_material("steel")),
        rtol=4.0e-13,
        atol=2.0e-5,
    )
    np.testing.assert_array_equal(
        state["committed_nodal_rotation_matrices"],
        view.trial_rotation_matrices,
    )
    assert canonical_json_bytes(state)

    force_without, no_tangent, repeated, _view = _direct_response(
        model, element, zero, state, tangent=False
    )
    np.testing.assert_array_equal(force_without, force)
    assert no_tangent is None
    assert canonical_json_bytes(repeated) == canonical_json_bytes(state)


def test_direct_finite_rigid_rotation_is_objective_and_commits_shared_q() -> None:
    model, element = _model()
    rotation_vector = np.asarray((0.23, -0.17, 0.19), dtype=np.float64)
    rotation = rotation_exponential(rotation_vector)
    total = np.zeros(18, dtype=np.float64)
    by_node = total.reshape(3, 6)
    by_node[:, :3] = (rotation @ _REFERENCE.T).T - _REFERENCE
    by_node[:, 3:6] = rotation_vector

    force, _tangent, state, view = _direct_response(model, element, total)

    np.testing.assert_allclose(force, np.zeros(18), rtol=0.0, atol=2.0e-4)
    np.testing.assert_allclose(
        state["station_generalized_strain"], np.zeros((7, 8)), rtol=0.0, atol=3.0e-15
    )
    np.testing.assert_allclose(
        state["committed_pl_twist"], np.zeros(3), rtol=0.0, atol=3.0e-15
    )
    np.testing.assert_array_equal(
        state["committed_nodal_rotation_matrices"], view.trial_rotation_matrices
    )
    for node in range(3):
        np.testing.assert_array_equal(
            np.asarray(state["committed_director_triads"])[node, :, 2],
            view.trial_directors[node],
        )


def test_direct_total_force_directional_derivative_matches_returned_tangent() -> None:
    model, element = _model()
    rng = np.random.default_rng(20260825)
    centre_u = 2.0e-5 * rng.standard_normal(18)
    direction = rng.standard_normal(18)
    direction /= np.linalg.norm(direction)
    step = 1.0e-7

    centre_force, tangent, _state, _view = _direct_response(
        model, element, centre_u
    )
    plus = _direct_response(model, element, centre_u + step * direction)[0]
    minus = _direct_response(model, element, centre_u - step * direction)[0]

    assert tangent is not None
    np.testing.assert_allclose(
        tangent @ direction,
        (plus - minus) / (2.0 * step),
        rtol=2.0e-6,
        atol=1.0,
    )
    assert np.linalg.norm(centre_force) > 0.0


def test_direct_pl_multiplier_energy_and_physical_resultant_separation() -> None:
    model, element = _model()
    angle = 0.21
    total = np.zeros(18, dtype=np.float64)
    total[5::6] = angle

    force, _tangent, state, _view = _direct_response(model, element, total)
    frame = element._model_bound_nonlinear_context(
        model.mesh, model.get_material("steel")
    )[1]
    membrane = element._constitutive(model.get_material("steel"), frame)[1]
    k_d = invariant_drilling_scale(membrane)
    constraint = np.asarray(state["committed_pl_twist"], dtype=np.float64)
    multiplier = np.asarray(state["committed_pl_multiplier"], dtype=np.float64)
    area = 0.5 * float(
        np.linalg.norm(np.cross(_REFERENCE[1] - _REFERENCE[0], _REFERENCE[2] - _REFERENCE[0]))
    )
    gram = (area / 12.0) * np.asarray(((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0)))

    np.testing.assert_array_equal(multiplier, k_d * constraint)
    mixed = np.asarray(
        [
            sum(
                float(gram[row, column]) * float(multiplier[column])
                for column in range(3)
            )
            for row in range(3)
        ]
    )
    expected_energy = sum(
        0.5 * float(constraint[row]) * float(mixed[row]) for row in range(3)
    )
    assert state["committed_pl_energy"] == expected_energy
    np.testing.assert_array_equal(
        state["station_generalized_resultant"], np.zeros((7, 8))
    )
    np.testing.assert_array_equal(state["committed_internal_force"], force)
    np.testing.assert_array_equal(
        state["bubble_rotation_last_increment"], np.zeros(2)
    )


def test_direct_plastic_trial_does_not_mutate_committed_history() -> None:
    model, element = _model(plastic=True)
    material = model.get_material("steel")
    committed = element.init_model_bound_nonlinear_state(model.mesh, material, 3)
    before = canonical_json_bytes(committed)
    total = np.zeros(18, dtype=np.float64)
    total[6] = 0.008
    total[12] = 0.003

    _force, _tangent, trial, _view = _direct_response(
        model, element, total, committed
    )

    assert canonical_json_bytes(committed) == before
    assert np.max(np.asarray(trial["alpha"])) > 0.0
    assert np.max(np.abs(np.asarray(trial["plastic_strain"]))) > 0.0


@pytest.mark.parametrize(
    "curve",
    (
        DNVC208MaterialCurve(
            sigma_prop=245.0e6,
            sigma_yield=250.0e6,
            sigma_yield_2=260.0e6,
            eps_p_y1=0.004,
            eps_p_y2=0.02,
            K=500.0e6,
            n=0.2,
        ),
        LinearHardeningCurve(250.0e6, 1.0e9),
        PowerLawHardeningCurve.from_yield(250.0e6, 700.0e6, 0.2),
        PiecewiseLinearCurve(
            (0.0, 0.01, 0.05),
            (250.0e6, 270.0e6, 310.0e6),
        ),
    ),
    ids=("dnv-c208", "linear", "power-law", "piecewise-linear"),
)
def test_direct_admitted_isotropic_curve_has_elastic_plastic_history_and_fd_tangent(
    curve: object,
) -> None:
    model, element = _model(curve=curve)
    material = model.get_material("steel")
    committed = element.init_model_bound_nonlinear_state(model.mesh, material, 3)
    committed_bytes = canonical_json_bytes(committed)

    elastic_u = np.zeros(18, dtype=np.float64)
    elastic_u[6] = 1.0e-6
    _elastic_force, _elastic_tangent, elastic_state, _elastic_view = (
        _direct_response(model, element, elastic_u, committed)
    )
    np.testing.assert_array_equal(elastic_state["alpha"], np.zeros(21))
    np.testing.assert_array_equal(
        elastic_state["plastic_strain"], np.zeros((21, 3))
    )

    plastic_u = np.zeros(18, dtype=np.float64)
    plastic_u[6] = 0.008
    plastic_u[12] = 0.003
    force, tangent, plastic_state, _plastic_view = _direct_response(
        model, element, plastic_u, committed
    )
    assert tangent is not None
    alpha = np.asarray(plastic_state["alpha"], dtype=np.float64)
    assert np.max(alpha) > 0.0
    assert np.max(np.abs(np.asarray(plastic_state["plastic_strain"]))) > 0.0
    assert canonical_json_bytes(committed) == committed_bytes
    stress = np.asarray(plastic_state["layer_stress"], dtype=np.float64)
    equivalent = np.sqrt(
        stress[:, 0] ** 2
        - stress[:, 0] * stress[:, 1]
        + stress[:, 1] ** 2
        + 3.0 * stress[:, 2] ** 2
    )
    flow = np.asarray(curve.flow_stress(alpha), dtype=np.float64)  # type: ignore[attr-defined]
    yielded = alpha > 0.0
    np.testing.assert_allclose(
        equivalent[yielded], flow[yielded], rtol=2.0e-10, atol=0.1
    )

    direction = np.zeros(18, dtype=np.float64)
    direction[6] = 0.8
    direction[12] = -0.6
    step = 1.0e-7
    plus = _direct_response(
        model, element, plastic_u + step * direction, committed
    )[0]
    minus = _direct_response(
        model, element, plastic_u - step * direction, committed
    )[0]
    np.testing.assert_allclose(
        tangent @ direction,
        (plus - minus) / (2.0 * step),
        rtol=2.0e-5,
        atol=2.0,
    )
    assert np.linalg.norm(force) > 0.0

    next_u = plastic_u.copy()
    next_u[6] += 2.0e-4
    _next_force, _next_tangent, next_state, _next_view = _direct_response(
        model, element, next_u, plastic_state
    )
    assert np.all(
        np.asarray(next_state["alpha"]) >= np.asarray(plastic_state["alpha"])
    )
    assert canonical_json_bytes(plastic_state)


def test_direct_two_noncommuting_steps_rebase_on_committed_shared_q() -> None:
    model, element = _model()
    first_vector = np.asarray((0.18, -0.07, 0.03))
    first = np.zeros(18, dtype=np.float64)
    first.reshape(3, 6)[:, 3:6] = first_vector
    _force1, _tangent1, state1, view1 = _direct_response(model, element, first)

    second_vector = np.asarray((-0.04, 0.16, 0.09))
    second_total = np.asarray(state1["committed_total_u"]).copy()
    second_total.reshape(3, 6)[:, 3:6] += second_vector
    _force2, _tangent2, state2, view2 = _direct_response(
        model, element, second_total, state1
    )
    expected = rotation_exponential(second_vector) @ rotation_exponential(first_vector)

    np.testing.assert_array_equal(view1.trial_rotation_matrices[0], rotation_exponential(first_vector))
    np.testing.assert_array_equal(view2.trial_rotation_matrices[0], expected)
    np.testing.assert_array_equal(
        state2["committed_nodal_rotation_matrices"][0], expected
    )
    assert not np.allclose(
        expected,
        rotation_exponential(first_vector + second_vector),
        rtol=0.0,
        atol=1.0e-5,
    )
    assert canonical_json_bytes(state2)


def test_direct_response_rejects_missing_malformed_and_stale_native_views() -> None:
    model, element = _model()
    total = np.zeros(18, dtype=np.float64)
    with pytest.raises(TypeError, match="native_rotation_trial"):
        element.compute_nonlinear_response(
            model.mesh, model.get_material("steel"), total, None, 3, True
        )

    view = _view_for_total_u(model, element, total, None)
    stale = replace(view, trial_serial=None)
    with pytest.raises(ValueError, match="active native rotation trial"):
        element.compute_nonlinear_response(
            model.mesh,
            model.get_material("steel"),
            total,
            None,
            3,
            True,
            native_rotation_trial=stale,
        )
    nonvirgin = replace(view, generation=1)
    with pytest.raises(ValueError, match="unambiguous zero committed"):
        element.compute_nonlinear_response(
            model.mesh,
            model.get_material("steel"),
            total,
            None,
            3,
            True,
            native_rotation_trial=nonvirgin,
        )
    wrong_q = np.asarray(view.trial_rotation_matrices).copy()
    wrong_q[0] = rotation_exponential((0.01, 0.0, 0.0))
    malformed = replace(view, trial_rotation_matrices=wrong_q)
    with pytest.raises(ValueError, match=r"trial Q|Q/reference"):
        element.compute_nonlinear_response(
            model.mesh,
            model.get_material("steel"),
            total,
            None,
            3,
            True,
            native_rotation_trial=malformed,
        )

    committed = element.init_model_bound_nonlinear_state(
        model.mesh, model.get_material("steel"), 3
    )
    corrupted = copy.deepcopy(committed)
    corrupted["committed_total_u"][0] = 1.0e-8
    with pytest.raises(ValueError, match="integrity"):
        element.compute_nonlinear_response(
            model.mesh,
            model.get_material("steel"),
            total,
            corrupted,
            3,
            True,
            native_rotation_trial=view,
        )
