"""Qualified S3 committed tangent decomposition and current-state buckling."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import anyfesolver.e4_pl_s3_element as s3
from anyfesolver import (
    AnalysisSession,
    FEModel,
    QualifiedE4PLS3ShellElement,
    QualifiedE4PLShellElement,
    solve_eigenvalue_buckling,
    solve_static_nonlinear,
)
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.buckling import CURRENT_STATE_BUCKLING_POLICY_ID
from anyfesolver.current_state_tangent import (
    assemble_committed_current_tangent_components,
)
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from anyfesolver.element_capabilities import (
    ElementCapabilityError,
    STATEFUL_MATERIAL_RESPONSE_MODE,
    STATELESS_FIXED_GENERALIZED_SECTION_RESPONSE_MODE,
    require_model_element_capabilities,
    require_model_nonlinear_workflow_capabilities,
)
from anyfesolver.linalg import FactorizationCache
from anyfesolver.nonlinear_state import (
    NonlinearStateStore,
    begin_state_evaluation,
    create_model_native_rotation_store,
    discard_active_state_candidate,
)

import test_e4_pl_s3_direct_nonlinear_response as direct
import test_e4_pl_s3_native_tl_full_d3 as d3
import test_e4_pl_s3_physical_director_reversal as reversal
import test_e4_pl_s3_prestressed_modal_buckling as eigen
from _e4_pl_s3_native_trial import native_trial_for_increment


def _relative(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = max(
        float(np.linalg.norm(actual, ord="fro")),
        float(np.linalg.norm(expected, ord="fro")),
        1.0,
    )
    return float(np.linalg.norm(actual - expected, ord="fro") / scale)


def _committed_element_components(
    model: FEModel,
    element: QualifiedE4PLS3ShellElement,
    total_u: np.ndarray,
    state: dict[str, object],
) -> dict[str, object]:
    before = canonical_json_bytes(state)
    store = NonlinearStateStore.from_shell_layouts((), {element.element_id: state})
    rotations = create_model_native_rotation_store(model, store, total_u)
    assert rotations is not None
    store.attach_native_rotation_store(rotations)
    token = begin_state_evaluation(store, model=model, displacements=total_u)
    assert token is not None
    try:
        view = store.native_element_rotation_view(
            token,
            element.element_id,
            tuple(element.node_ids),
            element.native_reference_directors(model.mesh),
        )
        result = dict(
            element.compute_committed_current_tangent_components(
                model.mesh,
                model.get_material(element.material_name),
                np.asarray(total_u)[element.get_dof_mapping(model.mesh)],
                state,
                3,
                native_rotation_trial=view,
            )
        )
    finally:
        discard_active_state_candidate(store)
    assert canonical_json_bytes(state) == before
    return result


@pytest.mark.parametrize("plastic_history", (False, True))
def test_committed_layered_components_close_project_and_match_force_difference(
    plastic_history: bool,
) -> None:
    model, element = direct._model(plastic=plastic_history)
    material = model.get_material("steel")
    initial = element.init_model_bound_nonlinear_state(model.mesh, material, 3)
    if plastic_history:
        total = np.zeros(18, dtype=np.float64)
        total[6] = 0.008
        total[12] = 0.003
        _force, _tangent, first, _view = direct._direct_response(
            model, element, total, initial
        )
        total = total.copy()
        total[6] += 2.0e-4
        _force, _tangent, committed, _view = direct._direct_response(
            model, element, total, first
        )
        assert np.max(np.asarray(committed["alpha"], dtype=float)) > 0.0
    else:
        total = np.zeros(18, dtype=np.float64)
        total[6] = 1.7e-5
        total[12] = -0.6e-5
        total[8] = 0.9e-5
        _force, _tangent, committed, _view = direct._direct_response(
            model, element, total, initial
        )

    frozen = canonical_json_bytes(committed)
    components = _committed_element_components(model, element, total, committed)
    material_tangent = np.asarray(components["material"])
    geometric_tangent = np.asarray(components["geometric"])
    total_tangent = np.asarray(components["total"])
    projection = np.asarray(components["bubble_projection"])
    sensitivity = np.asarray(components["bubble_total_sensitivity"])
    uncondensed_material = np.asarray(components["uncondensed_material"])
    uncondensed_geometric = np.asarray(components["uncondensed_geometric"])
    uncondensed_total = np.asarray(components["uncondensed_total"])
    physical_material = np.asarray(components["physical_material"])
    physical_geometric = np.asarray(components["physical_geometric"])
    pl_material = np.asarray(components["objective_pl_material_numerical"])

    np.testing.assert_array_equal(projection[:18], np.eye(18))
    np.testing.assert_array_equal(projection[18:], sensitivity)
    np.testing.assert_allclose(
        uncondensed_material + uncondensed_geometric,
        uncondensed_total,
        rtol=2.0e-15,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        physical_material,
        projection.T @ uncondensed_material @ projection,
        rtol=2.0e-15,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        physical_geometric,
        projection.T @ uncondensed_geometric @ projection,
        rtol=2.0e-15,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        material_tangent,
        0.5
        * (
            physical_material
            + pl_material
            + physical_material.T
            + pl_material.T
        ),
        rtol=3.0e-15,
        atol=3.0e-7,
    )
    np.testing.assert_allclose(
        geometric_tangent,
        0.5 * (physical_geometric + physical_geometric.T),
        rtol=3.0e-15,
        atol=3.0e-7,
    )
    np.testing.assert_allclose(
        material_tangent + geometric_tangent,
        total_tangent,
        rtol=3.0e-15,
        atol=5.0e-7,
    )
    for matrix in (material_tangent, geometric_tangent, total_tangent):
        assert _relative(matrix, matrix.T) <= 512.0 * np.finfo(float).eps

    direction = (
        np.asarray(
            (0.7, -0.2, 0.3, -0.1, 0.4, -0.25) * 3,
            dtype=np.float64,
        )
        if not plastic_history
        else np.asarray(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.8, 0.0, 0.0, 0.0, 0.0, 0.0,
             -0.6, 0.0, 0.0, 0.0, 0.0, 0.0),
            dtype=np.float64,
        )
    )
    direction /= np.linalg.norm(direction)
    step = 1.0e-7 if plastic_history else 5.0e-8
    plus = direct._direct_response(
        model, element, total + step * direction, committed
    )[0]
    minus = direct._direct_response(
        model, element, total - step * direction, committed
    )[0]
    # A state lying exactly on the active yield surface is nonsmooth.  The
    # zero-increment algorithmic tangent selects its elastic/unloading branch,
    # so prove that one-sided derivative there; elastic states use the stronger
    # centred difference.
    force_difference = (
        (np.asarray(components["force"]) - minus) / step
        if plastic_history
        else (plus - minus) / (2.0 * step)
    )
    np.testing.assert_allclose(
        total_tangent @ direction,
        force_difference,
        rtol=3.0e-5 if plastic_history else 3.0e-6,
        atol=500.0 if plastic_history else 1.0,
    )
    assert canonical_json_bytes(committed) == frozen
    assert components["state_digest"] in {
        committed.get("state_digest"),
        committed.get("state_integrity_sha256"),
    }
    assert all(
        not np.asarray(components[name]).flags.writeable
        for name in ("material", "geometric", "total", "bubble_projection")
    )


@pytest.mark.parametrize("generalized", (False, True))
def test_committed_components_cover_generalized_sections_and_both_polarities(
    generalized: bool,
) -> None:
    rng = np.random.default_rng(1403 + int(generalized))
    total = 1.1e-5 * rng.standard_normal(18)
    by_polarity: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for polarity in (-1, 1):
        model, element = reversal._model(
            polarity=polarity,
            generalized=generalized,
        )
        _force, _tangent, committed = reversal._native_response(
            model, element, total
        )
        frozen = canonical_json_bytes(committed)
        components = _committed_element_components(
            model, element, total, committed
        )
        material_tangent = np.asarray(components["material"])
        geometric_tangent = np.asarray(components["geometric"])
        total_tangent = np.asarray(components["total"])
        np.testing.assert_allclose(
            material_tangent + geometric_tangent,
            total_tangent,
            rtol=3.0e-15,
            atol=1.0e-7,
        )
        direction = rng.standard_normal(18)
        direction /= np.linalg.norm(direction)
        step = 5.0e-8
        plus = reversal._native_response(
            model, element, total + step * direction, committed
        )[0]
        minus = reversal._native_response(
            model, element, total - step * direction, committed
        )[0]
        np.testing.assert_allclose(
            total_tangent @ direction,
            (plus - minus) / (2.0 * step),
            rtol=4.0e-6,
            atol=0.15,
        )
        assert canonical_json_bytes(committed) == frozen
        by_polarity[polarity] = (
            material_tangent,
            geometric_tangent,
            total_tangent,
        )
        if generalized:
            assert "alpha" not in committed
            assert committed["state_mode"] == "stateless_generalized_section"

    for negative, positive in zip(by_polarity[-1], by_polarity[1]):
        assert _relative(negative, positive) <= 4.0e-10


def _d3_component_response(
    reference_nodes: np.ndarray,
    current_nodes: np.ndarray,
    frame: np.ndarray,
    triads: np.ndarray,
    material_direction: np.ndarray,
    material: object,
    state: dict[str, np.ndarray],
    external: np.ndarray,
):
    direction_components = frame[:, :2].T @ material_direction
    material_angle = float(
        math.atan2(direction_components[1], direction_components[0])
    )

    def builder(increment: np.ndarray):
        native_trial, exact, _store = native_trial_for_increment(
            current_nodes, triads, increment
        )
        return s3._native_layered_uncondensed_response_components(
            current_nodes,
            triads,
            exact,
            reference_nodes,
            frame,
            material,
            material_angle,
            d3._THICKNESS,
            state,
            d3._LAYER_COUNT,
            native_rotation_trial=native_trial,
        )

    return s3._solve_native_bubble_equilibrium(
        external,
        np.zeros(2),
        builder,
    )


def test_material_and_stress_hessian_pieces_are_full_d3_covariant() -> None:
    (
        reference_nodes,
        current_nodes,
        owner,
        triads,
        material_direction,
        material,
        baseline_state,
        external,
    ) = d3._base_fixture()
    baseline_frame = d3._independent_frame(reference_nodes, owner)
    _force, _total, _trial, baseline_meta = _d3_component_response(
        reference_nodes,
        current_nodes,
        baseline_frame,
        triads,
        material_direction,
        material,
        baseline_state,
        external,
    )
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = d3._external_permutation(permutation)
        numbered_reference = reference_nodes[list(permutation)]
        numbered_current = current_nodes[list(permutation)]
        numbered_frame = d3._independent_frame(numbered_reference, owner)
        numbered_triads = np.concatenate(
            (triads[:3][list(permutation)], triads[3:]), axis=0
        )
        rotation = numbered_frame[:, :2].T @ baseline_frame[:, :2]
        engineering = d3._engineering_transform(rotation)
        resultant = d3._resultant_transform(rotation)
        station_map = d3._station_map(permutation)
        numbered_state = d3._numbered_state(
            baseline_state,
            station_map,
            engineering,
            resultant,
            rotation,
        )
        _force, total, _trial, metadata = _d3_component_response(
            numbered_reference,
            numbered_current,
            numbered_frame,
            numbered_triads,
            material_direction,
            material,
            numbered_state,
            permutation_matrix @ external,
        )
        projection = np.asarray(metadata["bubble_projection"])
        for condensed_key, uncondensed_key in (
            ("condensed_material_tangent", "uncondensed_material_tangent"),
            ("condensed_geometric_tangent", "uncondensed_geometric_tangent"),
        ):
            condensed = np.asarray(metadata[condensed_key])
            uncondensed = np.asarray(metadata[uncondensed_key])
            np.testing.assert_allclose(
                condensed,
                projection.T @ uncondensed @ projection,
                rtol=2.0e-15,
                atol=2.0e-7,
            )
            expected = (
                permutation_matrix
                @ np.asarray(baseline_meta[condensed_key])
                @ permutation_matrix.T
            )
            assert d3._relative_error(condensed, expected) <= 4.0e-11
        np.testing.assert_allclose(
            np.asarray(metadata["condensed_material_tangent"])
            + np.asarray(metadata["condensed_geometric_tangent"]),
            total,
            rtol=3.0e-15,
            atol=6.0e-7,
        )


def _compressed_committed_state():
    model, load = eigen._axial_nonlinear_model()
    load.nodal_loads[2][0] = -1.0e3
    result = solve_static_nonlinear(
        model,
        load,
        num_steps=2,
        num_layers=3,
    )
    assert result.status == "completed"
    return model, result


def test_current_state_buckling_matches_native_pencil_scaling_and_session() -> None:
    model, static = _compressed_committed_state()
    frozen = canonical_json_bytes(static.element_states[1])
    material, internal_geometric, total, info = (
        assemble_committed_current_tangent_components(
            model,
            static.displacements,
            static.element_states,
            3,
        )
    )
    model.apply_boundary_conditions()
    reduced_material, _, transform, _, _, _ = build_constraint_transformation(
        material,
        np.zeros(material.shape[0], dtype=float),
        model,
    )
    reduced_geometric = (
        transform.T @ (-internal_geometric) @ transform
    ).toarray()
    oracle = eigen._finite_descriptor_values(
        reduced_material.toarray(), reduced_geometric
    )
    oracle = oracle[oracle > 0.0]

    expected = solve_eigenvalue_buckling(
        model,
        num_modes=2,
        dense_size_limit=1000,
        allow_free_mechanisms=True,
        current_state_displacements=static.displacements,
        current_state_element_states=static.element_states,
        current_state_num_layers=3,
    )
    doubled = solve_eigenvalue_buckling(
        model,
        num_modes=2,
        dense_size_limit=1000,
        allow_free_mechanisms=True,
        current_state_displacements=static.displacements,
        current_state_element_states=static.element_states,
        current_state_num_layers=3,
        current_state_load_scale=2.0,
    )
    with AnalysisSession(model) as session:
        first = solve_eigenvalue_buckling(
            model,
            num_modes=2,
            dense_size_limit=1000,
            allow_free_mechanisms=True,
            session=session,
            current_state_displacements=static.displacements,
            current_state_element_states=static.element_states,
            current_state_num_layers=3,
        )
        repeated = solve_eigenvalue_buckling(
            model,
            num_modes=2,
            dense_size_limit=1000,
            allow_free_mechanisms=True,
            session=session,
            current_state_displacements=static.displacements,
            current_state_element_states=static.element_states,
            current_state_num_layers=3,
        )

    factors = np.asarray([mode.load_factor for mode in expected.modes])
    doubled_factors = np.asarray([mode.load_factor for mode in doubled.modes])
    assert expected.solver_status == "ok"
    np.testing.assert_allclose(factors, oracle[: len(factors)], rtol=5.0e-12)
    np.testing.assert_allclose(doubled_factors, 0.5 * factors, rtol=5.0e-12)
    np.testing.assert_allclose(
        [mode.load_factor for mode in first.modes], factors, rtol=5.0e-12
    )
    np.testing.assert_allclose(
        [mode.load_factor for mode in repeated.modes], factors, rtol=5.0e-12
    )
    assert repeated.assembly_info["analysis_session_bypass_reason"] == (
        "committed_current_state_matrices_and_factors_are_not_cacheable"
    )
    assert repeated.assembly_info["current_state_buckling_policy_id"] == (
        CURRENT_STATE_BUCKLING_POLICY_ID
    )
    component_info = repeated.assembly_info["current_state_tangent_components"]
    assert component_info["state_digests"] == {
        "1": static.element_states[1]["state_integrity_sha256"]
    }
    assert component_info["state_immutability_verified"] is True
    assert component_info["matrix_persistence"] == "none"
    assert component_info["factorization_persistence"] == "none"
    assert component_info["buckling_total_closure_relative_error"] <= (
        512.0 * np.finfo(float).eps
    )
    assert canonical_json_bytes(static.element_states[1]) == frozen
    np.testing.assert_allclose(
        (material + internal_geometric).toarray(),
        total.toarray(),
        rtol=3.0e-15,
        atol=5.0e-7,
    )


def _mixed_model() -> FEModel:
    model = FEModel("mixed-q4-s3-current-buckling-guard")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.5, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1, [1, 2, 3, 4], "steel", thickness=0.02
        ),
    )
    model.add_element(
        2,
        QualifiedE4PLS3ShellElement(
            2,
            [2, 5, 3],
            "steel",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    model.add_boundary_condition(FixedSupport("left", [1, 4]))
    return model


def test_current_state_guards_fail_before_mechanics_and_mixed_q4_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = eigen._model()
    element = model.mesh.elements[1]
    state = element.init_model_bound_nonlinear_state(
        model.mesh, model.get_material("steel"), 3
    )

    def forbidden() -> None:
        raise AssertionError("current-state guard reached model mechanics")

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(ValueError, match="requires both committed"):
        solve_eigenvalue_buckling(
            model,
            current_state_displacements=np.zeros(18),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        solve_eigenvalue_buckling(
            model,
            {1: eigen._compression()},
            current_state_displacements=np.zeros(18),
            current_state_element_states={1: state},
        )
    with pytest.raises(ValueError, match="reference_elastic_only"):
        solve_eigenvalue_buckling(
            model,
            reference_elastic_only=True,
            current_state_displacements=np.zeros(18),
            current_state_element_states={1: state},
        )
    with pytest.raises(ValueError, match="follower-load"):
        solve_eigenvalue_buckling(
            model,
            reference_load_case=LoadCase("follower"),
            current_state_displacements=np.zeros(18),
            current_state_element_states={1: state},
        )
    with pytest.raises(ValueError, match="factorization_cache"):
        solve_eigenvalue_buckling(
            model,
            factorization_cache=FactorizationCache("forbidden-current-cache"),
            current_state_displacements=np.zeros(18),
            current_state_element_states={1: state},
        )
    with pytest.raises(ValueError, match="num_layers"):
        solve_eigenvalue_buckling(
            model,
            current_state_displacements=np.zeros(18),
            current_state_element_states={1: state},
            current_state_num_layers=0,
        )
    with pytest.raises(ValueError, match="exactly one model-bound state"):
        solve_eigenvalue_buckling(
            model,
            current_state_displacements=np.zeros(18),
            current_state_element_states={},
        )
    with pytest.raises(ValueError, match="available only"):
        solve_eigenvalue_buckling(model, current_state_load_scale=2.0)
    with pytest.raises(ValueError, match="current_state_num_layers is available only"):
        solve_eigenvalue_buckling(model, current_state_num_layers=3)

    mixed = _mixed_model()
    monkeypatch.setattr(mixed, "apply_boundary_conditions", forbidden)
    with pytest.raises(ValueError, match="exactly one model-bound state"):
        solve_eigenvalue_buckling(
            mixed,
            current_state_displacements=np.zeros(
                mixed.mesh.dof_manager.total_dofs
            ),
            current_state_element_states={},
        )

    matrix = element.capability_matrix()
    assert matrix["current_state_buckling_s3"] == "PARITY_REPLACED"
    assert matrix["mixed_current_state_buckling"] == "PARITY_REPLACED"
    assert matrix["buckling"] == (
        "EXPLICIT_REFERENCE_ELASTIC_OR_CURRENT_STATE_AUTHORITY_REQUIRED"
    )
    assert matrix["material_nonlinearity"] == "PARITY_REPLACED"


def test_stateless_generalized_nonlinear_workflow_has_no_material_history_waiver() -> None:
    generalized_model, generalized_element = reversal._model(
        polarity=1,
        generalized=True,
    )
    assert generalized_element.nonlinear_material_response_mode == (
        STATELESS_FIXED_GENERALIZED_SECTION_RESPONSE_MODE
    )
    assert generalized_element.capability_matrix()[
        "stateless_generalized_section_nonlinear_geometry"
    ] == "PARITY_REPLACED"
    require_model_nonlinear_workflow_capabilities(
        generalized_model,
        context="stateless-generalized-test",
    )
    with pytest.raises(ElementCapabilityError, match="material_nonlinearity"):
        require_model_element_capabilities(
            generalized_model,
            "material_nonlinearity",
            context="material-history-test",
        )

    layered_model, layered_element = reversal._model(
        polarity=1,
        generalized=False,
    )
    assert layered_element.nonlinear_material_response_mode == (
        STATEFUL_MATERIAL_RESPONSE_MODE
    )
    require_model_nonlinear_workflow_capabilities(
        layered_model,
        context="stateful-layered-test",
    )
