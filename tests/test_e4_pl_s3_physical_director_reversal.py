from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest

from anyfesolver import FEModel, QualifiedE4PLS3ShellElement, shell_element_from_dict
from anyfesolver._native_rotation_state import create_native_rotation_state_store
from anyfesolver.boundary import LoadCase
from anyfesolver.e4_pl_s3_state import (
    DIRECTOR_POLARITY_POLICY_ID,
    DIRECTOR_REVERSAL_TRANSFORM_ID,
    S3CommittedStateError,
    seal_committed_s3_state,
)
from anyfesolver.material_curves import LinearHardeningCurve
from anyfesolver.shell_sections import GeneralizedShellSection
from anyfesolver.recovery import _recover_qualified_s3_committed_state


REFERENCE = np.asarray(
    ((0.0, 0.0, 0.0), (1.1, 0.0, 0.0), (0.24, 0.94, 0.0)),
    dtype=np.float64,
)
OWNER = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)


def _section() -> GeneralizedShellSection:
    return GeneralizedShellSection(
        name="director-reversal-B-coupled",
        A=np.asarray(
            (
                (2.4e8, 0.42e8, 0.13e8),
                (0.42e8, 1.35e8, -0.08e8),
                (0.13e8, -0.08e8, 0.71e8),
            )
        ),
        B=np.asarray(
            (
                (2.1e4, -0.7e4, 0.3e4),
                (0.4e4, 1.6e4, -0.2e4),
                (-0.5e4, 0.1e4, 0.8e4),
            )
        ),
        D=np.asarray(
            (
                (3.2e4, 0.55e4, 0.12e4),
                (0.55e4, 2.1e4, -0.09e4),
                (0.12e4, -0.09e4, 1.15e4),
            )
        ),
        As=np.asarray(((0.82e8, 0.11e8), (0.11e8, 0.59e8))),
        mass_per_area=41.0,
        rotary_inertia_per_area=0.014,
    )


def _model(
    *,
    polarity: int,
    node_order: tuple[int, int, int] = (1, 2, 3),
    generalized: bool = False,
) -> tuple[FEModel, QualifiedE4PLS3ShellElement]:
    model = FEModel("qualified-s3-director-reversal")
    model.add_material("carrier", 70.0e9, 0.31, density=2700.0)
    for node_id, coordinates in enumerate(REFERENCE, start=1):
        model.add_node(node_id, *coordinates)
    element = QualifiedE4PLS3ShellElement(
        1,
        list(node_order),
        "carrier",
        thickness=0.027,
        material_direction=(0.91, 0.37, 0.0),
        material_angle_deg=13.0,
        shell_section=_section() if generalized else None,
        reference_normal=OWNER,
        director_polarity=polarity,
    )
    model.add_element(1, element)
    return model, element


def _native_response(
    model: FEModel,
    element: QualifiedE4PLS3ShellElement,
    total_u: np.ndarray,
    state: dict[str, object] | None = None,
):
    total = np.asarray(total_u, dtype=np.float64).reshape(18)
    if state is None:
        committed_u = np.zeros(18, dtype=np.float64)
        committed_q = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    else:
        committed_u = np.asarray(state["committed_total_u"], dtype=np.float64)
        committed_q = np.asarray(
            state["committed_nodal_rotation_matrices"], dtype=np.float64
        )
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
        committed_full_coordinates=(
            element.get_node_coordinates(model.mesh)
            + committed_u.reshape(3, 6)[:, :3]
        ),
        committed_rotation_matrices={
            node_id: committed_q[row] for row, node_id in enumerate(node_ids)
        },
    )
    assert store is not None
    reference = element.get_node_coordinates(model.mesh)
    token = store.begin_trial(total, reference + total.reshape(3, 6)[:, :3])
    view = store.element_view(
        element.element_id,
        node_ids,
        element.native_reference_directors(model.mesh),
        trial_token=token,
    )
    return element.compute_nonlinear_response(
        model.mesh,
        model.get_material("carrier"),
        total,
        state,
        3,
        True,
        native_rotation_trial=view,
    )


def _relative(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = max(
        float(np.linalg.norm(actual, ord=np.inf)),
        float(np.linalg.norm(expected, ord=np.inf)),
        1.0,
    )
    return float(np.linalg.norm(actual - expected, ord=np.inf)) / scale


def test_director_polarity_is_strict_serialized_and_distinct_from_owner() -> None:
    for invalid in (True, False, 0, 2, -2, 1.0, "-1"):
        with pytest.raises(ValueError, match="director_polarity"):
            QualifiedE4PLS3ShellElement(
                1,
                [1, 2, 3],
                reference_normal=OWNER,
                director_polarity=invalid,  # type: ignore[arg-type]
            )

    model, element = _model(polarity=-1)
    np.testing.assert_array_equal(element.reference_normal, OWNER)
    np.testing.assert_array_equal(element.physical_reference_director, -OWNER)
    np.testing.assert_array_equal(
        element.native_reference_directors(model.mesh),
        np.repeat((-OWNER)[None, :], 3, axis=0),
    )
    payload = element.to_dict()
    assert payload["director_polarity"] == -1
    assert payload["director_polarity_policy_id"] == DIRECTOR_POLARITY_POLICY_ID
    assert payload["director_reversal_transform_id"] == DIRECTOR_REVERSAL_TRANSFORM_ID
    rebuilt = shell_element_from_dict(payload)
    assert isinstance(rebuilt, QualifiedE4PLS3ShellElement)
    assert rebuilt.director_polarity == -1
    for key in (
        "director_polarity",
        "director_polarity_policy_id",
        "director_reversal_transform_id",
    ):
        malformed = dict(payload)
        malformed.pop(key)
        with pytest.raises(ValueError, match="director"):
            shell_element_from_dict(malformed)


def test_all_six_d3_actions_preserve_both_physical_directors() -> None:
    for polarity in (-1, 1):
        for ordering in itertools.permutations((1, 2, 3)):
            model, element = _model(
                polarity=polarity,
                node_order=ordering,
            )
            np.testing.assert_allclose(
                element.native_reference_directors(model.mesh),
                np.repeat((polarity * OWNER)[None, :], 3, axis=0),
                rtol=0.0,
                atol=0.0,
            )
            expected_sign = 1.0 if ordering in {
                (1, 2, 3),
                (2, 3, 1),
                (3, 1, 2),
            } else -1.0
            assert element.sheet_area_orientation_sign(model.mesh) == expected_sign


def test_all_six_d3_operators_are_covariant_for_both_polarities_and_B_coupling() -> None:
    for polarity in (-1, 1):
        baseline_model, baseline_element = _model(
            polarity=polarity,
            generalized=True,
        )
        baseline = np.asarray(
            baseline_element._compute_stiffness_components(
                baseline_model.mesh,
                baseline_model.get_material("carrier"),
                enforce_positive_winding=False,
            )["total"]
        )
        for ordering in itertools.permutations((1, 2, 3)):
            model, element = _model(
                polarity=polarity,
                node_order=ordering,
                generalized=True,
            )
            actual = np.asarray(
                element._compute_stiffness_components(
                    model.mesh,
                    model.get_material("carrier"),
                    enforce_positive_winding=False,
                )["total"]
            )
            dofs = [
                6 * (node_id - 1) + component
                for node_id in ordering
                for component in range(6)
            ]
            assert _relative(actual, baseline[np.ix_(dofs, dofs)]) <= 3.0e-14


def test_B_coupled_linear_operator_and_virtual_work_are_polarity_invariant() -> None:
    rng = np.random.default_rng(20260825)
    displacement = 1.0e-4 * rng.standard_normal(18)
    virtual = rng.standard_normal(18)
    baseline: np.ndarray | None = None
    for polarity in (-1, 1):
        model, element = _model(polarity=polarity, generalized=True)
        components = element.compute_stiffness_components(
            model.mesh, model.get_material("carrier")
        )
        constitutive = np.asarray(components["constitutive"])
        rotated = element._generalized_section_in_frame(components["frame"])
        assert rotated is not None
        np.testing.assert_allclose(
            constitutive[:3, 3:6], polarity * rotated.B, rtol=0.0, atol=0.0
        )
        stiffness = np.asarray(components["total"])
        if baseline is None:
            baseline = stiffness
        else:
            assert _relative(stiffness, baseline) <= 3.0e-14
        force = stiffness @ displacement
        assert float(virtual @ force) == pytest.approx(
            float(displacement @ stiffness @ virtual), rel=2.0e-13, abs=1.0e-12
        )


def test_linear_recovery_reverses_local_first_moments_and_physical_surfaces() -> None:
    rng = np.random.default_rng(7701)
    displacement = 2.0e-5 * rng.standard_normal(18)
    recovered: dict[int, dict[str, object]] = {}
    for polarity in (-1, 1):
        model, element = _model(polarity=polarity)
        recovered[polarity] = element.compute_stresses(
            model.mesh,
            displacement,
            model.get_material("carrier"),
            return_global=True,
        )
    plus = recovered[1]
    minus = recovered[-1]
    for key in ("membrane_strain", "membrane_resultants"):
        np.testing.assert_allclose(minus[key], plus[key], rtol=2.0e-13, atol=1.0e-12)
    for key in (
        "curvature",
        "bending_resultants",
        "transverse_shear_strain",
        "transverse_shear_resultants",
    ):
        np.testing.assert_allclose(minus[key], -np.asarray(plus[key]), rtol=2.0e-13, atol=1.0e-12)
    for key in (
        "global_membrane_resultant_tensors",
        "global_bending_resultant_tensors",
        "global_transverse_shear_resultants",
    ):
        np.testing.assert_allclose(minus[key], plus[key], rtol=2.0e-13, atol=1.0e-8)
    for component in ("xx", "yy", "xy"):
        np.testing.assert_allclose(
            minus[f"local_{component}_top"],
            plus[f"local_{component}_bot"],
            rtol=2.0e-13,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            minus[f"local_{component}_bot"],
            plus[f"local_{component}_top"],
            rtol=2.0e-13,
            atol=1.0e-8,
        )


def test_polarity_is_bound_into_layered_and_generalized_state_identity() -> None:
    initial = {
        "initial_membrane_stress": np.asarray((3.0, 4.0, 5.0)),
        "initial_bending_stress": np.asarray((6.0, 7.0, 8.0)),
        "initial_membrane_prestrain": np.asarray((0.1, 0.2, 0.3)),
        "initial_curvature_prestrain": np.asarray((0.4, 0.5, 0.6)),
    }
    states: dict[int, dict[str, object]] = {}
    for polarity in (-1, 1):
        model, element = _model(polarity=polarity)
        states[polarity] = element.init_model_bound_nonlinear_state(
            model.mesh,
            model.get_material("carrier"),
            3,
            initial_fields=initial,
            initial_field_provenance={"authority": "director-reversal-test"},
        )
        state = states[polarity]
        assert state["director_polarity"] == polarity
        np.testing.assert_array_equal(
            state["reference_corner_directors"],
            np.repeat((polarity * OWNER)[None, :], 3, axis=0),
        )
        np.testing.assert_array_equal(
            state["initial_bending_stress"],
            polarity * np.broadcast_to(initial["initial_bending_stress"], (7, 3)),
        )
        np.testing.assert_array_equal(
            state["initial_curvature_prestrain"],
            polarity * np.broadcast_to(initial["initial_curvature_prestrain"], (7, 3)),
        )
    assert states[-1]["element_configuration_fingerprint"] != states[1]["element_configuration_fingerprint"]
    assert states[-1]["reference_corner_directors_fingerprint"] != states[1]["reference_corner_directors_fingerprint"]

    negative_model, negative_element = _model(polarity=-1)
    positive_model, positive_element = _model(polarity=1)
    with pytest.raises(S3CommittedStateError, match="identity does not match"):
        negative_element.validate_model_bound_nonlinear_state(
            negative_model.mesh,
            negative_model.get_material("carrier"),
            states[1],
            3,
        )
    for generalized in (False, True):
        positive_model, positive_element = _model(
            polarity=1,
            generalized=generalized,
        )
        state = positive_element.init_model_bound_nonlinear_state(
            positive_model.mesh,
            positive_model.get_material("carrier"),
            3,
        )
        mutated = copy.deepcopy(state)
        mutated["director_polarity"] = -1
        mutated = seal_committed_s3_state(mutated)
        with pytest.raises(S3CommittedStateError):
            positive_element.validate_model_bound_nonlinear_state(
                positive_model.mesh,
                positive_model.get_material("carrier"),
                mutated,
                3,
            )


def test_generalized_native_response_is_polarity_invariant_with_signed_fields() -> None:
    rng = np.random.default_rng(99173)
    total = 1.2e-5 * rng.standard_normal(18)
    responses = {}
    for polarity in (-1, 1):
        model, element = _model(polarity=polarity, generalized=True)
        responses[polarity] = _native_response(model, element, total)
    force_plus, tangent_plus, state_plus = responses[1]
    force_minus, tangent_minus, state_minus = responses[-1]
    np.testing.assert_allclose(force_minus, force_plus, rtol=2.0e-10, atol=2.0e-5)
    np.testing.assert_allclose(tangent_minus, tangent_plus, rtol=2.0e-10, atol=2.0e-2)
    np.testing.assert_allclose(
        state_minus["station_generalized_strain"][:, :3],
        state_plus["station_generalized_strain"][:, :3],
        rtol=2.0e-11,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        state_minus["station_generalized_strain"][:, 3:],
        -state_plus["station_generalized_strain"][:, 3:],
        rtol=2.0e-11,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        state_minus["station_generalized_resultant"][:, :3],
        state_plus["station_generalized_resultant"][:, :3],
        rtol=2.0e-10,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        state_minus["station_generalized_resultant"][:, 3:],
        -state_plus["station_generalized_resultant"][:, 3:],
        rtol=2.0e-10,
        atol=1.0e-5,
    )


def test_layered_native_state_reverses_physical_layer_order() -> None:
    rng = np.random.default_rng(453)
    total = 8.0e-6 * rng.standard_normal(18)
    states = {}
    forces = {}
    tangents = {}
    for polarity in (-1, 1):
        model, element = _model(polarity=polarity)
        force, tangent, state = _native_response(model, element, total)
        forces[polarity] = force
        tangents[polarity] = tangent
        states[polarity] = state
    np.testing.assert_allclose(forces[-1], forces[1], rtol=3.0e-10, atol=2.0e-4)
    np.testing.assert_allclose(tangents[-1], tangents[1], rtol=3.0e-10, atol=0.2)
    plus_layers = np.asarray(states[1]["layer_stress"]).reshape(7, 3, 3)
    minus_layers = np.asarray(states[-1]["layer_stress"]).reshape(7, 3, 3)
    np.testing.assert_allclose(
        minus_layers,
        plus_layers[:, ::-1, :],
        rtol=3.0e-10,
        atol=2.0e-4,
    )


def test_layered_plastic_history_is_ordered_bottom_to_top_along_each_director() -> None:
    total = np.zeros(18, dtype=np.float64)
    total[6] = 0.008
    total[12] = 0.003
    total[10] = 0.03
    total[16] = 0.012
    states: dict[int, dict[str, object]] = {}
    for polarity in (-1, 1):
        model, element = _model(polarity=polarity)
        model.add_material(
            "carrier",
            70.0e9,
            0.31,
            density=2700.0,
            hardening_curve=LinearHardeningCurve(90.0e6, 1.5e9),
        )
        _force, _tangent, states[polarity] = _native_response(
            model,
            element,
            total,
        )
    plus_alpha = np.asarray(states[1]["alpha"]).reshape(7, 3)
    minus_alpha = np.asarray(states[-1]["alpha"]).reshape(7, 3)
    plus_plastic = np.asarray(states[1]["plastic_strain"]).reshape(7, 3, 3)
    minus_plastic = np.asarray(states[-1]["plastic_strain"]).reshape(7, 3, 3)
    assert float(np.max(plus_alpha)) > 0.0
    assert float(np.max(np.abs(plus_plastic))) > 0.0
    np.testing.assert_allclose(
        minus_alpha,
        plus_alpha[:, ::-1],
        rtol=3.0e-9,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        minus_plastic,
        plus_plastic[:, ::-1, :],
        rtol=3.0e-9,
        atol=2.0e-12,
    )


def test_committed_layer_recovery_swaps_physical_surfaces_and_preserves_global_work() -> None:
    rng = np.random.default_rng(611)
    total = 7.0e-6 * rng.standard_normal(18)
    recovered: dict[int, dict[str, object]] = {}
    for polarity in (-1, 1):
        model, element = _model(polarity=polarity)
        _force, _tangent, state = _native_response(model, element, total)
        actual, reason, _sources = _recover_qualified_s3_committed_state(
            model,
            1,
            element,
            state,
            displacements=total,
            return_global=True,
        )
        assert reason == ""
        assert actual is not None
        recovered[polarity] = actual
    plus = recovered[1]
    minus = recovered[-1]
    for key in (
        "global_membrane_resultant_tensors",
        "global_bending_resultant_tensors",
        "global_transverse_shear_resultants",
    ):
        np.testing.assert_allclose(
            minus[key], plus[key], rtol=4.0e-10, atol=2.0e-4
        )
    for component in ("xx", "yy", "xy", "xz", "yz"):
        np.testing.assert_allclose(
            minus[f"local_{component}_top"],
            plus[f"local_{component}_bot"],
            rtol=4.0e-10,
            atol=2.0e-4,
        )
        np.testing.assert_allclose(
            minus[f"local_{component}_bot"],
            plus[f"local_{component}_top"],
            rtol=4.0e-10,
            atol=2.0e-4,
        )


def test_shared_node_Q_is_identical_while_element_directors_are_opposite() -> None:
    model, plus_element = _model(polarity=1)
    minus_element = QualifiedE4PLS3ShellElement(
        2,
        [1, 2, 3],
        "carrier",
        thickness=plus_element.thickness,
        material_direction=plus_element.material_direction,
        material_angle_deg=plus_element.material_angle_deg,
        reference_normal=OWNER,
        director_polarity=-1,
    )
    total = np.zeros(18, dtype=np.float64)
    total.reshape(3, 6)[:, 3:6] = np.asarray(
        ((0.07, -0.03, 0.02), (0.07, -0.03, 0.02), (0.07, -0.03, 0.02))
    )
    store = create_native_rotation_state_store(
        (1, 2, 3),
        rotational_dofs={node: (6 * row + 3, 6 * row + 4, 6 * row + 5) for row, node in enumerate((1, 2, 3))},
        coordinate_rows={node: row for row, node in enumerate((1, 2, 3))},
        coordinate_node_ids=(1, 2, 3),
        committed_full_displacement=np.zeros(18),
        committed_full_coordinates=REFERENCE,
    )
    token = store.begin_trial(total, REFERENCE.copy())
    plus_view = store.element_view(
        1,
        (1, 2, 3),
        plus_element.native_reference_directors(model.mesh),
        trial_token=token,
    )
    minus_view = store.element_view(
        2,
        (1, 2, 3),
        minus_element.native_reference_directors(model.mesh),
        trial_token=token,
    )
    np.testing.assert_array_equal(
        minus_view.trial_rotation_matrices,
        plus_view.trial_rotation_matrices,
    )
    np.testing.assert_allclose(
        minus_view.trial_directors,
        -plus_view.trial_directors,
        rtol=0.0,
        atol=2.0e-16,
    )


def test_pressure_load_and_follower_tangent_use_owner_not_director_or_d3() -> None:
    pressure = 731.0
    baseline_load: np.ndarray | None = None
    for polarity in (-1, 1):
        for ordering in itertools.permutations((1, 2, 3)):
            model, element = _model(polarity=polarity, node_order=ordering)
            load_case = LoadCase("owner-oriented-pressure")
            load_case.add_pressure_load(1, pressure)
            actual = load_case.get_load_vector(
                model.mesh, model.mesh.dof_manager, model.get_material
            )
            if baseline_load is None:
                baseline_load = actual
            else:
                np.testing.assert_allclose(actual, baseline_load, rtol=0.0, atol=2.0e-13)

    model, element = _model(polarity=-1, node_order=(2, 1, 3))
    load_case = LoadCase("follower-pressure")
    coordinates = element.get_node_coordinates(model.mesh) + np.asarray(
        ((0.01, -0.02, 0.03), (-0.01, 0.03, 0.02), (0.02, 0.01, -0.01))
    )
    direction = np.asarray(
        ((0.4, -0.1, 0.2), (-0.3, 0.5, -0.2), (0.1, -0.4, 0.3))
    )
    tangent = load_case._consistent_pressure_tangent(
        element, model.mesh, pressure, coordinates
    )
    step = 2.0e-7
    plus = load_case._consistent_pressure_load(
        element, model.mesh, pressure, coordinates + step * direction
    )
    minus = load_case._consistent_pressure_load(
        element, model.mesh, pressure, coordinates - step * direction
    )
    local_direction = np.zeros(18)
    local_direction.reshape(3, 6)[:, :3] = direction
    np.testing.assert_allclose(
        tangent @ local_direction,
        (plus - minus) / (2.0 * step),
        rtol=2.0e-9,
        atol=2.0e-7,
    )
    virtual = np.linspace(-0.7, 0.9, 18)
    assert float(virtual @ plus) == pytest.approx(
        float(plus @ virtual), rel=0.0, abs=0.0
    )


def test_direct_global_couples_are_polarity_independent() -> None:
    expected: np.ndarray | None = None
    applied = np.zeros(18)
    applied[3::6] = (2.0, -3.0, 4.0)
    applied[4::6] = (-1.0, 5.0, 0.5)
    applied[5::6] = (7.0, -2.0, 1.0)
    for polarity in (-1, 1):
        model, _element = _model(polarity=polarity)
        load_case = LoadCase("global-couples")
        load_case.element_loads[1] = applied.copy()
        actual = load_case.get_load_vector(
            model.mesh, model.mesh.dof_manager, model.get_material
        )
        if expected is None:
            expected = actual
        else:
            np.testing.assert_array_equal(actual, expected)
