from __future__ import annotations

import copy
import itertools
from collections.abc import Mapping

import numpy as np
import pytest
from scipy import linalg

import anyfesolver.e4_pl_element as q4_module
import anyfesolver.e4_pl_s3_element as s3_module
import anyfesolver.e4_pl_s3_state as s3_state_module
import anyfesolver.elements as elements_module
import anyfesolver.shell_sections as shell_sections_module
from anyfesolver import (
    AnalysisSession,
    FEModel,
    assemble_stiffness_matrix,
    solve_eigenvalue_buckling,
    solve_free_vibration,
    solve_static_nonlinear,
)
from anyfesolver.activity import ElementActivity
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.boundary import FixedSupport, LoadCase
from anyfesolver.buckling import (
    QUALIFIED_Q4_S3_CURRENT_STATE_BUCKLING_POLICY_ID,
)
from anyfesolver.current_state_tangent import (
    COMMITTED_CURRENT_TANGENT_ASSEMBLY_POLICY_ID,
    COMMITTED_CURRENT_TANGENT_ROUTE_POLICY_ID,
    assemble_committed_current_tangent_components,
    require_exact_qualified_component_lifecycle_api,
    validate_committed_current_tangent_inputs,
)
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_element import (
    REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
    QualifiedE4PLS3ShellElement,
)
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import ShellElement
from anyfesolver.linalg import FactorizationCache
from anyfesolver.modal import CURRENT_STATE_MODAL_POLICY_ID
from anyfesolver.nonlinear_state import (
    NonlinearStateStore,
    begin_state_evaluation,
    create_model_native_rotation_store,
    discard_active_state_candidate,
)
from anyfesolver.shell_sections import GeneralizedShellSection


def _b_coupled_section() -> GeneralizedShellSection:
    return GeneralizedShellSection(
        name="mixed-route-b-coupled",
        A=np.asarray(
            (
                (2.4e8, 0.42e8, 0.13e8),
                (0.42e8, 1.35e8, -0.08e8),
                (0.13e8, -0.08e8, 0.71e8),
            ),
            dtype=np.float64,
        ),
        B=np.asarray(
            (
                (2.1e4, -0.7e4, 0.3e4),
                (0.4e4, 1.6e4, -0.2e4),
                (-0.5e4, 0.1e4, 0.8e4),
            ),
            dtype=np.float64,
        ),
        D=np.asarray(
            (
                (3.2e4, 0.55e4, 0.12e4),
                (0.55e4, 2.1e4, -0.09e4),
                (0.12e4, -0.09e4, 1.15e4),
            ),
            dtype=np.float64,
        ),
        As=np.asarray(((0.82e8, 0.11e8), (0.11e8, 0.59e8))),
        mass_per_area=41.0,
        rotary_inertia_per_area=0.014,
    )


def _mixed_model(
    *,
    q4_nodes: tuple[int, int, int, int] = (1, 2, 3, 4),
    s3_nodes: tuple[int, int, int] = (2, 5, 3),
    rotation: np.ndarray | None = None,
    q4_section: GeneralizedShellSection | None = None,
    s3_reference_surface_offset: float = 0.0,
) -> FEModel:
    model = FEModel("qualified-q4-s3-current-state-route")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    proper = (
        np.eye(3, dtype=np.float64)
        if rotation is None
        else np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    )
    base_coordinates = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (2.0, 0.5, 0.0),
    )
    transformed_coordinates = {
        node_id: proper @ np.asarray(coordinates, dtype=float)
        for node_id, coordinates in enumerate(base_coordinates, start=1)
    }
    for node_id, coordinates in transformed_coordinates.items():
        model.add_node(node_id, *coordinates)
    owner = proper @ np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    triangle = np.asarray(
        [transformed_coordinates[node_id] for node_id in s3_nodes], dtype=float
    )
    winding = 1 if float(
        np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]) @ owner
    ) > 0.0 else -1
    s3_reference_normal = float(winding) * owner
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1,
            list(q4_nodes),
            "steel",
            thickness=0.02,
            material_direction=(proper[:, 0] if q4_section is not None else None),
            shell_section=q4_section,
            reference_normal=owner,
        ),
    )
    model.add_element(
        2,
        QualifiedE4PLS3ShellElement(
            2,
            list(s3_nodes),
            "steel",
            thickness=0.02,
            reference_normal=s3_reference_normal,
            director_polarity=winding,
            reference_surface_offset=s3_reference_surface_offset,
        ),
    )
    model.add_boundary_condition(FixedSupport("left", [1, 4]))
    return model


def _zero_states(
    model: FEModel, layers: int = 3
) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64)
    material = model.get_material("steel")
    q4 = model.mesh.elements[1]
    s3 = model.mesh.elements[2]
    q4_state = q4.seal_committed_current_tangent_state(
        model.mesh,
        material,
        displacement[q4.get_dof_mapping(model.mesh)],
        q4.init_nonlinear_state(layers),
        layers,
    )
    s3_state = s3.init_model_bound_nonlinear_state(
        model.mesh, material, layers
    )
    return displacement, {1: q4_state, 2: s3_state}


def _matrices(
    model: FEModel,
    displacement: np.ndarray,
    states: dict[int, dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    material, geometric, total, info = (
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )
    )
    return (
        material.toarray(),
        geometric.toarray(),
        total.toarray(),
        info,
    )


def _compressed_states(
    model: FEModel,
    layers: int = 3,
    strain: float = 1.0e-4,
    compression_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    total_dofs = model.mesh.dof_manager.total_dofs
    displacement = np.zeros(total_dofs, dtype=np.float64)
    axis = np.asarray(
        (1.0, 0.0, 0.0)
        if compression_axis is None
        else compression_axis,
        dtype=np.float64,
    )
    axis /= np.linalg.norm(axis)
    for node in model.mesh.nodes.values():
        coordinates = np.asarray(node.coords(), dtype=np.float64)
        displacement[np.asarray(node.dofs[:3], dtype=np.intp)] = (
            -float(strain) * float(coordinates @ axis) * axis
        )
    material = model.get_material("steel")
    q4 = model.mesh.elements[1]
    s3 = model.mesh.elements[2]

    q4_initial = q4.init_nonlinear_state(layers)
    _q4_force, _q4_tangent, q4_trial = q4.compute_nonlinear_response(
        model.mesh,
        material,
        displacement[q4.get_dof_mapping(model.mesh)],
        q4_initial,
        layers,
        True,
    )
    q4_state = q4.seal_committed_current_tangent_state(
        model.mesh,
        material,
        displacement[q4.get_dof_mapping(model.mesh)],
        q4_trial,
        layers,
    )

    s3_initial = s3.init_model_bound_nonlinear_state(
        model.mesh, material, layers
    )
    zero = np.zeros(total_dofs, dtype=np.float64)
    store = NonlinearStateStore.from_shell_layouts((), {2: s3_initial})
    rotations = create_model_native_rotation_store(model, store, zero)
    assert rotations is not None
    store.attach_native_rotation_store(rotations)
    token = begin_state_evaluation(store, model=model, displacements=displacement)
    assert token is not None
    try:
        view = store.native_element_rotation_view(
            token,
            2,
            tuple(s3.node_ids),
            s3.native_reference_directors(model.mesh),
        )
        _s3_force, _s3_tangent, s3_state = s3.compute_nonlinear_response(
            model.mesh,
            material,
            displacement[s3.get_dof_mapping(model.mesh)],
            s3_initial,
            layers,
            True,
            native_rotation_trial=view,
        )
    finally:
        discard_active_state_candidate(store)
    return displacement, {1: q4_state, 2: s3_state}


def _manual_component_scatter(
    model: FEModel,
    displacement: np.ndarray,
    states: dict[int, dict[str, object]],
    layers: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = model.mesh.dof_manager.total_dofs
    outputs = tuple(np.zeros((size, size), dtype=np.float64) for _ in range(3))
    store = NonlinearStateStore.from_shell_layouts((), states)
    rotations = create_model_native_rotation_store(model, store, displacement)
    assert rotations is not None
    store.attach_native_rotation_store(rotations)
    token = begin_state_evaluation(store, model=model, displacements=displacement)
    assert token is not None
    try:
        for element_id, element in sorted(model.mesh.elements.items()):
            dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
            kwargs: dict[str, object] = {}
            if bool(
                getattr(element, "formulation_native_total_lagrangian", False)
            ):
                kwargs["native_rotation_trial"] = store.native_element_rotation_view(
                    token,
                    int(element_id),
                    tuple(element.node_ids),
                    element.native_reference_directors(model.mesh),
                )
            components = element.compute_committed_current_tangent_components(
                model.mesh,
                model.get_material(element.material_name),
                displacement[dofs],
                states[int(element_id)],
                layers,
                **kwargs,
            )
            for target, key in zip(outputs, ("material", "geometric", "total")):
                target[np.ix_(dofs, dofs)] += np.asarray(
                    components[key], dtype=np.float64
                )
    finally:
        discard_active_state_candidate(store)
    return outputs


def test_mixed_route_assembles_24_and_18_dof_native_components_read_only() -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    frozen = {key: canonical_json_bytes(value) for key, value in states.items()}

    material, geometric, total, info = _matrices(model, displacement, states)

    np.testing.assert_allclose(
        material + geometric, total, rtol=2.0e-15, atol=1.0e-7
    )
    assert info["policy_id"] == COMMITTED_CURRENT_TANGENT_ASSEMBLY_POLICY_ID
    assert info["route"] == {
        "route": "mixed_qualified_q4_s3",
        "route_policy_id": COMMITTED_CURRENT_TANGENT_ROUTE_POLICY_ID,
        "families": ["qualified_q4", "qualified_s3"],
        "formulation_counts": {
            "E4_PL_QUALIFIED_Q4_HYBRID_V2": 1,
            "E4_PL_QUALIFIED_S3_COMPANION_V1": 1,
        },
        "native_rotation_required": True,
        "kinematic_scope": {
            "qualified_q4": "additive_rotation_von_karman",
            "qualified_s3": "native_multiplicative_total_lagrangian",
        },
        "reference_surface_offset_scope": {
            "qualified_q4": "q4_zero_offset_only",
            "qualified_s3": "s3_native_signed_offset",
        },
    }
    assert info["element_components"]["1"]["local_dofs"] == 24
    assert info["element_components"]["1"]["native_rotation_required"] is False
    assert info["element_components"]["1"]["state_binding_verified"] is True
    assert info["element_components"]["1"]["algorithmic_origin_verified"] is True
    assert info["element_components"]["1"]["algorithmic_origin_schema_id"] == (
        q4_module.Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
    )
    assert info["element_components"]["2"]["local_dofs"] == 18
    assert info["element_components"]["2"]["native_rotation_required"] is True
    assert info["element_components"]["2"]["state_binding_verified"] is True
    assert info["state_storage"]["native_rotation_activated"] is True
    assert {key: canonical_json_bytes(value) for key, value in states.items()} == frozen


def test_nonzero_compressed_mixed_pencil_matches_manual_scatter_and_scales() -> None:
    model = _mixed_model()
    displacement, states = _compressed_states(model)
    frozen = {key: canonical_json_bytes(value) for key, value in states.items()}
    material, geometric, total, _info = _matrices(
        model, displacement, states
    )
    manual = _manual_component_scatter(model, displacement, states)
    for assembled, independent in zip(
        (material, geometric, total), manual
    ):
        np.testing.assert_allclose(
            assembled, independent, rtol=2.0e-15, atol=2.0e-7
        )
    assert np.linalg.norm(geometric, ord="fro") > 0.0
    np.testing.assert_allclose(
        material + geometric, total, rtol=2.0e-15, atol=2.0e-7
    )

    model.apply_boundary_conditions()
    material_sparse, geometric_sparse, _total_sparse, _ = (
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )
    )
    reduced_material, _, transform, _, _, _ = build_constraint_transformation(
        material_sparse,
        np.zeros(material.shape[0], dtype=np.float64),
        model,
    )
    reduced_geometric = (
        transform.T @ (-geometric_sparse) @ transform
    ).toarray()
    eigenvalues = linalg.eigvals(
        reduced_material.toarray(), reduced_geometric
    )
    finite = eigenvalues[np.isfinite(eigenvalues)]
    real = np.asarray(
        [
            float(value.real)
            for value in finite
            if abs(float(value.imag))
            <= 1.0e-8 * max(abs(float(value.real)), 1.0)
            and float(value.real) > 0.0
        ],
        dtype=np.float64,
    )
    oracle = np.sort(real)
    assert oracle.size >= 2

    result = solve_eigenvalue_buckling(
        model,
        num_modes=2,
        dense_size_limit=1000,
        allow_free_mechanisms=True,
        current_state_displacements=displacement,
        current_state_element_states=states,
        current_state_num_layers=3,
    )
    doubled = solve_eigenvalue_buckling(
        model,
        num_modes=2,
        dense_size_limit=1000,
        allow_free_mechanisms=True,
        current_state_displacements=displacement,
        current_state_element_states=states,
        current_state_num_layers=3,
        current_state_load_scale=2.0,
    )
    with AnalysisSession(model) as session:
        repeated = solve_eigenvalue_buckling(
            model,
            num_modes=2,
            dense_size_limit=1000,
            allow_free_mechanisms=True,
            session=session,
            current_state_displacements=displacement,
            current_state_element_states=states,
            current_state_num_layers=3,
        )
        session_diagnostics = session.diagnostics()
    factors = np.asarray([mode.load_factor for mode in result.modes])
    doubled_factors = np.asarray(
        [mode.load_factor for mode in doubled.modes]
    )
    assert result.solver_status == doubled.solver_status == "ok"
    np.testing.assert_allclose(factors, oracle[: factors.size], rtol=2.0e-10)
    np.testing.assert_allclose(doubled_factors, 0.5 * factors, rtol=2.0e-10)
    assert max(mode.residual_norm for mode in result.modes) < 1.0e-8
    assert repeated.assembly_info["current_state_buckling_policy_id"] == (
        QUALIFIED_Q4_S3_CURRENT_STATE_BUCKLING_POLICY_ID
    )
    assert repeated.assembly_info["analysis_session_bypass_reason"] == (
        "committed_current_state_matrices_and_factors_are_not_cacheable"
    )
    assert session_diagnostics["plan_builds"] == 0
    assert session_diagnostics["plan_hits"] == 0
    assert session_diagnostics["estimated_retained_bytes"] == 0
    assert {key: canonical_json_bytes(value) for key, value in states.items()} == frozen


def test_nonzero_mixed_components_cover_all_d4_d3_numberings_and_global_rotation() -> None:
    baseline_model = _mixed_model()
    baseline_u, baseline_states = _compressed_states(baseline_model)
    baseline = _matrices(baseline_model, baseline_u, baseline_states)[:3]

    d4 = (
        (1, 2, 3, 4),
        (2, 3, 4, 1),
        (3, 4, 1, 2),
        (4, 1, 2, 3),
        (1, 4, 3, 2),
        (4, 3, 2, 1),
        (3, 2, 1, 4),
        (2, 1, 4, 3),
    )
    for numbering in d4:
        model = _mixed_model(q4_nodes=numbering)
        displacement, states = _compressed_states(model)
        actual = _matrices(model, displacement, states)[:3]
        for candidate, expected in zip(actual, baseline):
            np.testing.assert_allclose(
                candidate, expected, rtol=3.0e-10, atol=3.0e-5
            )

    for numbering in itertools.permutations((2, 5, 3)):
        model = _mixed_model(s3_nodes=numbering)
        displacement, states = _compressed_states(model)
        actual = _matrices(model, displacement, states)[:3]
        for candidate, expected in zip(actual, baseline):
            np.testing.assert_allclose(
                candidate, expected, rtol=3.0e-10, atol=3.0e-5
            )

    cx, sx = np.cos(0.37), np.sin(0.37)
    cz, sz = np.cos(-0.51), np.sin(-0.51)
    rotation = np.asarray(
        (
            (cz, -sz * cx, sz * sx),
            (sz, cz * cx, -cz * sx),
            (0.0, sx, cx),
        ),
        dtype=np.float64,
    )
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    rotated_model = _mixed_model(rotation=rotation)
    rotated_u, rotated_states = _compressed_states(
        rotated_model, compression_axis=rotation[:, 0]
    )
    rotated = _matrices(rotated_model, rotated_u, rotated_states)[:3]
    transform = np.zeros_like(rotated[0])
    for node in rotated_model.mesh.nodes.values():
        translation = np.asarray(node.dofs[:3], dtype=np.intp)
        rotational = np.asarray(node.dofs[3:6], dtype=np.intp)
        transform[np.ix_(translation, translation)] = rotation
        transform[np.ix_(rotational, rotational)] = rotation
    for candidate, original in zip(rotated, baseline):
        expected = transform @ original @ transform.T
        np.testing.assert_allclose(
            candidate, expected, rtol=8.0e-10, atol=8.0e-5
        )


def test_mixed_pencil_closes_for_s3_offset_and_zero_offset_b_coupled_q4() -> None:
    model = _mixed_model(
        q4_section=_b_coupled_section(),
        s3_reference_surface_offset=0.003,
    )
    displacement, states = _compressed_states(model)
    material, geometric, total, info = _matrices(
        model, displacement, states
    )
    manual = _manual_component_scatter(model, displacement, states)
    for candidate, expected in zip(
        (material, geometric, total), manual
    ):
        np.testing.assert_allclose(
            candidate, expected, rtol=3.0e-15, atol=3.0e-7
        )
    np.testing.assert_allclose(
        material + geometric, total, rtol=3.0e-15, atol=3.0e-7
    )
    assert info["route"]["reference_surface_offset_scope"] == {
        "qualified_q4": "q4_zero_offset_only",
        "qualified_s3": "s3_native_signed_offset",
    }
    assert info["element_components"]["1"]["kinematics"] == (
        "additive_rotation_von_karman"
    )
    assert info["element_components"]["1"][
        "reference_surface_offset_scope"
    ] == "q4_zero_offset_only"
    assert info["element_components"]["2"][
        "reference_surface_offset_scope"
    ] == "s3_native_signed_offset"

    result = solve_eigenvalue_buckling(
        model,
        num_modes=1,
        dense_size_limit=1000,
        allow_free_mechanisms=True,
        current_state_displacements=displacement,
        current_state_element_states=states,
        current_state_num_layers=3,
    )
    assert result.solver_status == "ok"
    assert result.critical_load_factor is not None
    assert result.critical_load_factor > 0.0
    assert result.assembly_info["current_state_route"][
        "reference_surface_offset_scope"
    ]["qualified_q4"] == "q4_zero_offset_only"


def test_activity_scales_every_component_and_deleted_elements_are_exact_zero() -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)

    q4_activity = ElementActivity([1, 2])
    q4_activity.hard_delete([2])
    model.set_element_activity(q4_activity)
    q4_only = _matrices(model, displacement, states)[:3]
    s3_activity = ElementActivity([1, 2])
    s3_activity.hard_delete([1])
    model.set_element_activity(s3_activity)
    s3_only = _matrices(model, displacement, states)[:3]
    model.set_element_activity(ElementActivity([1, 2], [0.5, 0.25]))
    fractional = _matrices(model, displacement, states)

    for q4_piece, s3_piece, actual in zip(
        q4_only, s3_only, fractional[:3]
    ):
        np.testing.assert_allclose(
            actual,
            0.5 * q4_piece + 0.25 * s3_piece,
            rtol=2.0e-15,
            atol=1.0e-7,
        )
    assert fractional[3]["element_components"]["1"][
        "activity_stiffness_scale"
    ] == 0.5
    assert fractional[3]["element_components"]["2"][
        "activity_stiffness_scale"
    ] == 0.25

    deleted = ElementActivity([1, 2])
    deleted.hard_delete([1, 2])
    model.set_element_activity(deleted)
    zero = _matrices(model, displacement, states)
    assert all(not np.any(matrix) for matrix in zero[:3])
    assert zero[3]["element_activity"]["zero_contribution_count"] == 2


@pytest.mark.parametrize("solver", ("modal", "buckling"))
@pytest.mark.parametrize("disposition", ("softened", "deleted"))
def test_current_eigen_requires_exact_active_element_lifecycle_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    solver: str,
    disposition: str,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    activity = ElementActivity([1, 2], [1.0, 1.0])
    if disposition == "softened":
        activity.set_activity(2, 0.5)
    else:
        activity.hard_delete([2])
    model.set_element_activity(activity)

    def forbidden() -> None:
        raise AssertionError("inactive current eigen route reached mechanics")

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(ElementCapabilityError, match="exact ACTIVE"):
        if solver == "modal":
            solve_free_vibration(
                model,
                current_state_displacements=displacement,
                current_state_element_states=states,
                current_state_num_layers=3,
            )
        else:
            solve_eigenvalue_buckling(
                model,
                current_state_displacements=displacement,
                current_state_element_states=states,
                current_state_num_layers=3,
            )


def test_route_and_state_binding_mutations_fail_before_component_mechanics() -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    q4 = model.mesh.elements[1]

    class SpoofedQualifiedQ4(QualifiedE4PLShellElement):
        pass

    spoofed = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "steel",
        thickness=0.02,
        reference_normal=(0.0, 0.0, 1.0),
    )
    # Production construction correctly rejects qualified descendants.  The
    # route validator still needs an invalid runtime object for this negative
    # test, so inject that identity only after exact construction.
    object.__setattr__(spoofed, "__class__", SpoofedQualifiedQ4)
    model.mesh.elements[1] = spoofed
    with pytest.raises(
        ElementCapabilityError, match="FORMULATION_ID_CLASS_MISMATCH|explicitly routed"
    ):
        validate_committed_current_tangent_inputs(
            model, displacement, states, 3, context="mutation"
        )
    model.mesh.elements[1] = q4

    q4.element_id = 99
    with pytest.raises(ElementCapabilityError, match="ID_MISMATCH"):
        validate_committed_current_tangent_inputs(
            model, displacement, states, 3, context="mutation"
        )
    q4.element_id = 1

    q4.formulation_id = "E4_PL_QUALIFIED_S3_COMPANION_V1"
    with pytest.raises(
        ElementCapabilityError, match="INSTANCE_SHADOW|explicitly routed"
    ):
        validate_committed_current_tangent_inputs(
            model, displacement, states, 3, context="mutation"
        )
    del q4.formulation_id

    q4.implementation_id = "SPOOFED_IMPLEMENTATION"
    with pytest.raises(
        ElementCapabilityError, match="IMPLEMENTATION_MISMATCH|INSTANCE_SHADOW"
    ):
        validate_committed_current_tangent_inputs(
            model, displacement, states, 3, context="mutation"
        )
    del q4.implementation_id

    q4.current_state_projection_policy_id = "SPOOFED_POLICY"
    with pytest.raises(
        ElementCapabilityError, match="COMPONENT_POLICY_MISMATCH|INSTANCE_SHADOW"
    ):
        validate_committed_current_tangent_inputs(
            model, displacement, states, 3, context="mutation"
        )
    del q4.current_state_projection_policy_id

    q4.reference_surface_offset = 0.001
    with pytest.raises(ElementCapabilityError, match="OFFSET_SCOPE_MISMATCH"):
        validate_committed_current_tangent_inputs(
            model, displacement, states, 3, context="mutation"
        )
    del q4.reference_surface_offset

    displaced = copy.deepcopy(states)
    displaced[1]["qualified_q4_committed_binding"]["committed_total_u"][0] = 1.0
    with pytest.raises(ValueError, match="exact displacement/state pairing"):
        validate_committed_current_tangent_inputs(
            model, displacement, displaced, 3, context="mutation"
        )

    malformed = copy.deepcopy(states)
    malformed[2]["state_integrity_sha256"] = "0" * 63
    with pytest.raises(ValueError, match="canonical SHA-256"):
        validate_committed_current_tangent_inputs(
            model, displacement, malformed, 3, context="mutation"
        )


@pytest.mark.parametrize(
    "constant_name",
    (
        "Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID",
        "Q4_CURRENT_STATE_PROJECTION_POLICY_ID",
    ),
)
def test_component_policy_mutations_cannot_enter_the_aggregate(
    monkeypatch: pytest.MonkeyPatch, constant_name: str
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    q4 = model.mesh.elements[1]
    monkeypatch.setattr(q4_module, constant_name, "SPOOFED_POLICY_ID")
    with pytest.raises(ValueError, match="authority"):
        q4.seal_committed_current_tangent_state(
            model.mesh,
            model.get_material("steel"),
            displacement[q4.get_dof_mapping(model.mesh)],
            states[1],
            3,
        )


def test_q4_algorithmic_origin_schema_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    monkeypatch.setattr(
        q4_module,
        "Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID",
        "SPOOFED_ALGORITHMIC_ORIGIN_SCHEMA",
    )

    with pytest.raises(
        ElementCapabilityError,
        match="algorithmic|origin|binding|schema|authority|MODULE_DATA",
    ):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )


@pytest.mark.parametrize(
    "constant_name",
    (
        "CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID",
        "CURRENT_STATE_BUBBLE_PROJECTION_POLICY_ID",
    ),
)
def test_s3_component_policy_mutations_cannot_enter_the_aggregate(
    monkeypatch: pytest.MonkeyPatch, constant_name: str
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    monkeypatch.setattr(s3_module, constant_name, "SPOOFED_POLICY_ID")

    with pytest.raises(
        ElementCapabilityError, match="authority|MODULE_DATA_MISMATCH"
    ):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )


def test_every_state_seal_is_prevalidated_before_the_first_component_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    model = _mixed_model()
    displacement, states = _zero_states(model)
    q4 = model.mesh.elements[1]
    s3 = model.mesh.elements[2]
    s3.thickness *= 1.1
    with pytest.raises(ValueError, match="state|identity|configuration"):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )
    assert getattr(q4, "_nl_cache", None) is None
    assert q4._qualified_components is None


def test_pure_q4_route_never_creates_a_native_rotation_transaction() -> None:
    model = _mixed_model()
    model.mesh.elements.pop(2)
    model.mesh.bump_revision("topology")
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64)
    q4 = model.mesh.elements[1]
    material = model.get_material("steel")
    state = q4.seal_committed_current_tangent_state(
        model.mesh,
        material,
        displacement[q4.get_dof_mapping(model.mesh)],
        q4.init_nonlinear_state(3),
        3,
    )

    _material, _geometric, _total, info = _matrices(
        model, displacement, {1: state}
    )

    assert info["route"]["route"] == "qualified_q4"
    assert info["route"]["native_rotation_required"] is False
    assert info["state_storage"]["native_rotation_activated"] is False

    result = solve_eigenvalue_buckling(
        model,
        current_state_displacements=displacement,
        current_state_element_states={1: state},
        current_state_num_layers=3,
    )
    assert result.solver_status == "zero_geometric_stiffness"
    assert result.assembly_info["current_state_buckling_policy_id"] == (
        QUALIFIED_Q4_S3_CURRENT_STATE_BUCKLING_POLICY_ID
    )
    assert result.assembly_info["current_state_route"]["route"] == "qualified_q4"


def test_solver_committed_mixed_states_feed_current_state_buckling_directly() -> None:
    model = _mixed_model()
    load = LoadCase("mixed-edge-compression")
    load.add_nodal_load(2, [-2.0e4, 0.0, 0.0, 0.0, 0.0, 0.0])
    load.add_nodal_load(3, [-2.0e4, 0.0, 0.0, 0.0, 0.0, 0.0])
    load.add_nodal_load(5, [-1.0e4, 0.0, 0.0, 0.0, 0.0, 0.0])

    nonlinear = solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-8,
        convergence_settings="legacy",
    )
    assert nonlinear.status == "completed"
    assert nonlinear.info["qualified_q4_committed_state_lifecycle"][
        "sealed_final_element_ids"
    ] == [1]
    route = validate_committed_current_tangent_inputs(
        model,
        nonlinear.displacements,
        nonlinear.element_states,
        3,
        context="solver-produced mixed committed state",
    )
    assert route["route"] == "mixed_qualified_q4_s3"

    result = solve_eigenvalue_buckling(
        model,
        num_modes=1,
        dense_size_limit=1000,
        allow_free_mechanisms=True,
        current_state_displacements=nonlinear.displacements,
        current_state_element_states=nonlinear.element_states,
        current_state_num_layers=3,
    )
    assert result.solver_status == "ok"
    assert result.critical_load_factor is not None
    assert result.critical_load_factor > 0.0
    assert result.assembly_info["current_state_buckling_policy_id"] == (
        QUALIFIED_Q4_S3_CURRENT_STATE_BUCKLING_POLICY_ID
    )
    assert result.assembly_info["current_state_route"]["route"] == (
        "mixed_qualified_q4_s3"
    )


def test_explicit_mixed_capability_guard_precedes_buckling_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    original = QualifiedE4PLS3ShellElement.capability_gaps.fget
    assert original is not None
    monkeypatch.setattr(
        QualifiedE4PLS3ShellElement,
        "capability_gaps",
        property(
            lambda element: original(element)
            | frozenset({"mixed_current_state_buckling"})
        ),
    )

    def forbidden() -> None:
        raise AssertionError("mixed capability guard reached buckling mechanics")

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(
        ElementCapabilityError,
        match="mixed_current_state_buckling|capability_gaps|CLASS_NAMESPACE|DEPENDENCY",
    ):
        solve_eigenvalue_buckling(
            model,
            current_state_displacements=displacement,
            current_state_element_states=states,
            current_state_num_layers=3,
        )


def test_mixed_current_state_modal_uses_exact_total_route_without_persistence() -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    frozen = {
        key: canonical_json_bytes(value) for key, value in states.items()
    }

    with AnalysisSession(model) as session:
        result = solve_free_vibration(
            model,
            num_modes=3,
            dense_size_limit=1000,
            session=session,
            current_state_displacements=displacement,
            current_state_element_states=states,
            current_state_num_layers=3,
        )
        session_diagnostics = session.diagnostics()

    assert result.solver_status == "ok"
    assert result.assembly_info["current_state_modal_policy_id"] == (
        CURRENT_STATE_MODAL_POLICY_ID
    )
    assert result.assembly_info["current_state_route"]["route"] == (
        "mixed_qualified_q4_s3"
    )
    tangent_info = result.assembly_info["current_state_tangent"]
    assert tangent_info["policy_id"] == (
        COMMITTED_CURRENT_TANGENT_ASSEMBLY_POLICY_ID
    )
    assert tangent_info["matrix_type"] == "committed_current_total_tangent"
    assert tangent_info["matrix_persistence"] == "none"
    assert tangent_info["factorization_persistence"] == "none"
    assert result.assembly_info["analysis_session_bypass_reason"] == (
        "committed_current_state_matrices_and_factors_are_not_cacheable"
    )
    assert session_diagnostics["plan_builds"] == 0
    assert session_diagnostics["plan_hits"] == 0
    assert session_diagnostics["estimated_retained_bytes"] == 0
    assert {key: canonical_json_bytes(value) for key, value in states.items()} == frozen


def test_current_state_modal_rejects_persistent_factorization_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)

    def forbidden() -> None:
        raise AssertionError("persistent-cache guard reached modal mechanics")

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(ValueError, match="persistent factorization_cache"):
        solve_free_vibration(
            model,
            factorization_cache=FactorizationCache("forbidden-current-modal"),
            current_state_displacements=displacement,
            current_state_element_states=states,
            current_state_num_layers=3,
        )


def test_explicit_mixed_modal_capability_guard_precedes_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    original = QualifiedE4PLS3ShellElement.capability_gaps.fget
    assert original is not None
    monkeypatch.setattr(
        QualifiedE4PLS3ShellElement,
        "capability_gaps",
        property(
            lambda element: original(element)
            | frozenset({"mixed_current_state_modal"})
        ),
    )

    def forbidden() -> None:
        raise AssertionError("mixed modal capability guard reached mechanics")

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(
        ElementCapabilityError,
        match="mixed_current_state_modal|capability_gaps|CLASS_NAMESPACE|DEPENDENCY",
    ):
        solve_free_vibration(
            model,
            current_state_displacements=displacement,
            current_state_element_states=states,
            current_state_num_layers=3,
        )


@pytest.mark.parametrize("element_id", (1, 2))
def test_instance_shadowed_connectivity_route_rejects_before_mechanics(
    element_id: int,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    element = model.mesh.elements[element_id]
    wrong_nodes = (2, 3, 4, 5) if element_id == 1 else (1, 2, 3)
    wrong_dofs = np.asarray(
        [
            dof
            for node_id in wrong_nodes
            for dof in model.mesh.dof_manager.get_node_dofs(node_id)
        ],
        dtype=np.intp,
    )
    element.get_dof_mapping = lambda _mesh: wrong_dofs.copy()

    with pytest.raises(ElementCapabilityError, match="INSTANCE_SHADOW"):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )


@pytest.mark.parametrize(
    ("element_id", "method_name"),
    (
        (1, "_qualified_linear_correction"),
        (1, "_accepted_algorithmic_update_fingerprint"),
        (1, "_constitutive_and_drill_stiffness"),
        (1, "_compute_4node_shape_functions"),
        (2, "compute_nonlinear_response"),
        (2, "_validate_model_bound_nonlinear_state_core"),
        (2, "init_model_bound_nonlinear_state"),
        (2, "_director_generalized_transform"),
        (2, "_material_angle"),
    ),
)
def test_class_level_critical_api_spoof_rejects_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    element_id: int,
    method_name: str,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    element_type = type(model.mesh.elements[element_id])

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("spoofed critical mechanics reached")

    monkeypatch.setattr(element_type, method_name, forbidden)
    with pytest.raises(ElementCapabilityError, match="CRITICAL_API_MISMATCH"):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )


def test_formulation_descriptor_is_rejected_without_invocation_current_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    expected = vars(QualifiedE4PLShellElement)["formulation_id"]

    class SplitFormulationDescriptor:
        reads = 0

        def __get__(self, instance: object, _owner: object) -> str:
            self.reads += 1
            return expected if instance is None else str(expected)

    descriptor = SplitFormulationDescriptor()
    monkeypatch.setattr(
        QualifiedE4PLShellElement, "formulation_id", descriptor
    )

    with pytest.raises(ElementCapabilityError, match="FORMULATION_ID_CLASS_MISMATCH"):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )
    assert descriptor.reads == 0


@pytest.mark.parametrize(
    "helper_name",
    (
        "_local_frame_and_derivatives",
        "_mitc4_shear_b_matrix",
        "to_dict",
    ),
)
def test_q4_base_kernel_spoof_rejects_before_current_state_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("spoofed Q4 base kernel reached mechanics")

    monkeypatch.setattr(ShellElement, helper_name, forbidden)
    with pytest.raises(
        ElementCapabilityError,
        match="BASE_CRITICAL_API_MISMATCH|DEPENDENCY_AUTHORITY_MISMATCH",
    ):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )


@pytest.mark.parametrize(
    ("owner", "element_id"),
    (
        (ShellElement, 1),
        (QualifiedE4PLS3ShellElement, 2),
    ),
)
def test_quadrature_descriptor_rejects_current_state_without_invocation(
    monkeypatch: pytest.MonkeyPatch,
    owner: type[object],
    element_id: int,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    expected = vars(owner)["gauss_points"]

    class StatefulQuadratureDescriptor:
        reads = 0

        def __get__(self, instance: object, _owner: object) -> object:
            self.reads += 1
            if instance is None:
                return expected
            return np.zeros((1, 2), dtype=np.float64)

    descriptor = StatefulQuadratureDescriptor()
    monkeypatch.setattr(owner, "gauss_points", descriptor)

    with pytest.raises(
        ElementCapabilityError,
        match="CRITICAL_API_MISMATCH|DEPENDENCY_AUTHORITY_MISMATCH",
    ):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )
    assert element_id in model.mesh.elements
    assert descriptor.reads == 0


def test_stale_q4_geometry_identity_helper_spoof_rejects_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    q4 = model.mesh.elements[1]
    old_identity = q4._stable_state_identity_payload(
        model.mesh,
        model.get_material("steel"),
        3,
    )
    model.mesh.nodes[1].x += 0.125

    monkeypatch.setattr(
        QualifiedE4PLShellElement,
        "_stable_state_identity_payload",
        lambda *_args, **_kwargs: copy.deepcopy(old_identity),
    )
    with pytest.raises(ElementCapabilityError, match="CRITICAL_API_MISMATCH"):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )


def test_split_class_instance_identity_descriptor_is_never_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    q4 = model.mesh.elements[1]
    old_identity = q4._stable_state_identity_payload(
        model.mesh,
        model.get_material("steel"),
        3,
    )
    original = QualifiedE4PLShellElement._stable_state_identity_payload

    class SplitIdentityDescriptor:
        reads = 0

        def __get__(self, instance: object, _owner: object) -> object:
            self.reads += 1
            if instance is None:
                return original
            return lambda *_args, **_kwargs: copy.deepcopy(old_identity)

    descriptor = SplitIdentityDescriptor()
    monkeypatch.setattr(
        QualifiedE4PLShellElement,
        "_stable_state_identity_payload",
        descriptor,
    )
    model.mesh.nodes[1].x += 0.125

    with pytest.raises(ElementCapabilityError, match="CRITICAL_API_MISMATCH"):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )
    assert descriptor.reads == 0


@pytest.mark.parametrize("profile", ("current", "prestress", "reference"))
def test_dynamic_descriptor_class_spoof_rejects_without_invocation(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)

    class ChangingNullity:
        reads = 0

        def __get__(self, _instance: object, _owner: object) -> int:
            self.reads += 1
            return 3 if self.reads == 1 else 0

    descriptor = ChangingNullity()
    monkeypatch.setattr(
        QualifiedE4PLS3ShellElement,
        "dynamic_algebraic_nullity",
        descriptor,
    )

    with pytest.raises(ElementCapabilityError, match="dynamic_identity|algebraic"):
        if profile == "current":
            solve_free_vibration(
                model,
                current_state_displacements=displacement,
                current_state_element_states=states,
                current_state_num_layers=3,
            )
        elif profile == "prestress":
            solve_free_vibration(
                model,
                prestress_states={
                    1: None,
                    2: {
                        "bubble_linearization_policy": (
                            REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID
                        )
                    },
                },
            )
        else:
            solve_free_vibration(model)
    assert descriptor.reads == 0


@pytest.mark.parametrize("profile", ("current", "prestress", "reference"))
def test_q4_forbidden_dynamic_descriptor_is_rejected_statically(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)

    class ChangingNullity:
        reads = 0

        def __get__(self, _instance: object, _owner: object) -> object:
            self.reads += 1
            return None if self.reads == 1 else 3

    descriptor = ChangingNullity()
    monkeypatch.setattr(
        QualifiedE4PLShellElement,
        "dynamic_algebraic_nullity",
        descriptor,
        raising=False,
    )

    with pytest.raises(ElementCapabilityError, match="dynamic_identity|algebraic"):
        if profile == "current":
            solve_free_vibration(
                model,
                current_state_displacements=displacement,
                current_state_element_states=states,
                current_state_num_layers=3,
            )
        elif profile == "prestress":
            solve_free_vibration(
                model,
                prestress_states={
                    1: None,
                    2: {
                        "bubble_linearization_policy": (
                            REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID
                        )
                    },
                },
            )
        else:
            solve_free_vibration(model)
    assert descriptor.reads == 0


def test_split_prestress_api_descriptor_is_never_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()
    original = QualifiedE4PLS3ShellElement._compute_stiffness_components

    class SplitMechanicsDescriptor:
        reads = 0

        def __get__(self, instance: object, _owner: object) -> object:
            self.reads += 1
            if instance is None:
                return original
            return lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("descriptor mechanics reached")
            )

    descriptor = SplitMechanicsDescriptor()
    monkeypatch.setattr(
        QualifiedE4PLS3ShellElement,
        "_compute_stiffness_components",
        descriptor,
    )
    with pytest.raises(
        ElementCapabilityError,
        match="_compute_stiffness_components|operator|authority",
    ):
        solve_free_vibration(
            model,
            prestress_states={
                1: None,
                2: {
                    "bubble_linearization_policy": (
                        REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID
                    )
                },
            },
        )
    assert descriptor.reads == 0


def test_current_modal_mass_and_algebraic_authority_rejects_spoofs() -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    s3 = model.mesh.elements[2]
    s3.compute_mass_matrix = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("spoofed mass mechanics reached")
    )
    with pytest.raises(ElementCapabilityError, match="mass|authority|SHADOW"):
        solve_free_vibration(
            model,
            current_state_displacements=displacement,
            current_state_element_states=states,
            current_state_num_layers=3,
        )

    del s3.compute_mass_matrix
    s3.dynamic_algebraic_nullity = 2
    with pytest.raises(
        ElementCapabilityError,
        match="algebraic-coordinate|CLASS_NAMESPACE_INSTANCE_SHADOW",
    ):
        solve_free_vibration(
            model,
            current_state_displacements=displacement,
            current_state_element_states=states,
            current_state_num_layers=3,
        )


def test_current_eigen_bypass_rejects_foreign_and_closed_sessions() -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    foreign_model = _mixed_model()
    foreign = AnalysisSession(foreign_model)
    try:
        with pytest.raises(ValueError, match="different FEModel"):
            solve_free_vibration(
                model,
                session=foreign,
                current_state_displacements=displacement,
                current_state_element_states=states,
                current_state_num_layers=3,
            )
        with pytest.raises(ValueError, match="different FEModel"):
            solve_eigenvalue_buckling(
                model,
                session=foreign,
                current_state_displacements=displacement,
                current_state_element_states=states,
                current_state_num_layers=3,
            )
    finally:
        foreign.close()

    closed = AnalysisSession(model)
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        solve_free_vibration(
            model,
            session=closed,
            current_state_displacements=displacement,
            current_state_element_states=states,
            current_state_num_layers=3,
        )
    with pytest.raises(RuntimeError, match="closed"):
        solve_eigenvalue_buckling(
            model,
            session=closed,
            current_state_displacements=displacement,
            current_state_element_states=states,
            current_state_num_layers=3,
        )


@pytest.mark.parametrize(
    ("family", "symbol", "route"),
    (
        ("q4", "_stationary_blocks", "current"),
        ("q4", "_stationary_blocks", "reference"),
        ("s3", "qualified_s3_triangle_frame", "current"),
        ("s3", "qualified_s3_triangle_frame", "reference"),
        ("s3", "_validate_s3_quadrature_values", "reference"),
    ),
)
def test_transitive_module_helper_spoof_rejects_before_any_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    symbol: str,
    route: str,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    module = q4_module if family == "q4" else s3_module
    reached: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        reached.append(symbol)
        raise AssertionError("mutated module helper reached mechanics")

    monkeypatch.setattr(module, symbol, forbidden)
    monkeypatch.setattr(
        model, "apply_boundary_conditions", lambda: reached.append("boundary")
    )

    with pytest.raises(ElementCapabilityError, match="MODULE_HELPER_MISMATCH"):
        if route == "current":
            assemble_committed_current_tangent_components(
                model, displacement, states, 3
            )
        else:
            solve_free_vibration(model, num_modes=1)
    assert reached == []


@pytest.mark.parametrize(
    ("family", "name"),
    (
        ("q4", "element_id"),
        ("q4", "node_ids"),
        ("q4", "material_name"),
        ("q4", "reference_surface_offset"),
        ("q4", "implementation_id"),
        ("q4", "current_state_projection_policy_id"),
        ("s3", "element_id"),
        ("s3", "node_ids"),
        ("s3", "material_name"),
        ("s3", "reference_surface_offset"),
        ("s3", "formulation_native_total_lagrangian"),
    ),
)
def test_route_owned_data_descriptors_are_never_invoked(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    name: str,
) -> None:
    model = _mixed_model(s3_reference_surface_offset=0.02)
    displacement, states = _zero_states(model)
    owner = (
        QualifiedE4PLShellElement
        if family == "q4"
        else QualifiedE4PLS3ShellElement
    )

    class ForbiddenDataDescriptor:
        reads = 0

        def __get__(self, _instance: object, _owner: object) -> object:
            self.reads += 1
            raise AssertionError("route invoked attacker-controlled descriptor")

        def __set__(self, _instance: object, _value: object) -> None:
            raise AssertionError("route wrote attacker-controlled descriptor")

    descriptor = ForbiddenDataDescriptor()
    monkeypatch.setattr(owner, name, descriptor, raising=False)

    with pytest.raises(ElementCapabilityError, match="MISMATCH|SHADOW"):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )
    assert descriptor.reads == 0


def test_s3_scientific_data_and_imported_class_alias_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()
    original_tying = s3_module.TYING_POINTS
    monkeypatch.setattr(s3_module, "TYING_POINTS", tuple(reversed(original_tying)))
    with pytest.raises(ElementCapabilityError, match="MODULE_DATA_MISMATCH.*TYING_POINTS"):
        solve_free_vibration(model, num_modes=1)

    monkeypatch.undo()
    model = _mixed_model()
    monkeypatch.setattr(q4_module, "GeneralizedShellSection", object)
    with pytest.raises(ElementCapabilityError, match="MODULE_HELPER_MISMATCH"):
        solve_free_vibration(model, num_modes=1)


@pytest.mark.parametrize(
    ("element_id", "name", "value"),
    (
        (1, "_MITC4_SAMPLE_POINTS", {"A": (0.0, 0.0)}),
        (2, "gauss_points", np.zeros((7, 2), dtype=np.float64)),
    ),
)
def test_instance_scientific_data_shadow_rejects_before_mechanics(
    element_id: int,
    name: str,
    value: object,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    model.mesh.elements[element_id].__dict__[name] = value

    with pytest.raises(
        ElementCapabilityError,
        match="CLASS_NAMESPACE_INSTANCE_SHADOW",
    ):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )


@pytest.mark.parametrize(
    ("module", "name"),
    (
        (elements_module, "_cross3"),
        (s3_state_module, "np"),
    ),
)
def test_transitive_dependency_module_spoof_rejects_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    name: str,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    reached: list[str] = []

    class ForbiddenProxy:
        def __getattr__(self, attribute: str) -> object:
            reached.append(attribute)
            raise AssertionError("mutated dependency reached mechanics")

    replacement: object = ForbiddenProxy()
    if name == "_cross3":
        def forbidden(*_args: object, **_kwargs: object) -> object:
            reached.append(name)
            raise AssertionError("mutated dependency reached mechanics")

        replacement = forbidden
    monkeypatch.setattr(module, name, replacement)

    with pytest.raises(
        ElementCapabilityError, match="DEPENDENCY_AUTHORITY_MISMATCH"
    ):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )
    assert reached == []


def test_generalized_section_method_namespace_is_frozen_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model(q4_section=_b_coupled_section())
    displacement, states = _zero_states(model)
    reached: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        reached.append("rotated")
        raise AssertionError("mutated generalized section reached mechanics")

    monkeypatch.setattr(
        shell_sections_module.GeneralizedShellSection, "rotated", forbidden
    )
    with pytest.raises(
        ElementCapabilityError, match="DEPENDENCY_AUTHORITY_MISMATCH"
    ):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )
    assert reached == []


def test_scientific_mapping_spoof_rejects_without_iterating_attacker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()
    reached: list[str] = []

    class ForbiddenMapping(Mapping[object, object]):
        def __getitem__(self, key: object) -> object:
            reached.append(f"getitem:{key}")
            raise AssertionError("attacker mapping evaluated")

        def __iter__(self) -> object:
            reached.append("iter")
            raise AssertionError("attacker mapping evaluated")

        def __len__(self) -> int:
            reached.append("len")
            raise AssertionError("attacker mapping evaluated")

    monkeypatch.setattr(s3_module, "TYING_POINTS", ForbiddenMapping())
    with pytest.raises(ElementCapabilityError, match="MODULE_DATA_MISMATCH"):
        solve_free_vibration(model, num_modes=1)
    assert reached == []


@pytest.mark.parametrize("element_id", (1, 2))
def test_route_rejects_attacker_configuration_without_conversion(
    element_id: int,
) -> None:
    model = _mixed_model()
    displacement, states = _zero_states(model)
    reached: list[str] = []

    class ForbiddenFloat:
        def __float__(self) -> float:
            reached.append("float")
            raise AssertionError("attacker configuration reached mechanics")

    object.__setattr__(model.mesh.elements[element_id], "thickness", ForbiddenFloat())
    with pytest.raises(ElementCapabilityError, match="CONFIGURATION_AUTHORITY"):
        assemble_committed_current_tangent_components(
            model, displacement, states, 3
        )
    assert reached == []


def test_direct_q4_s3_mechanics_bind_exact_numpy_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()
    material = model.get_material("steel")
    original_cross = np.cross

    def delegating_cross(*args: object, **kwargs: object) -> np.ndarray:
        return original_cross(*args, **kwargs)

    monkeypatch.setattr(np, "cross", delegating_cross)
    for element_id in (1, 2):
        with pytest.raises(ValueError, match="exact numerical runtime"):
            model.mesh.elements[element_id].compute_stiffness_matrix(
                model.mesh,
                material,
            )


def test_warm_assembly_rechecks_q4_and_s3_quadrature_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()
    assemble_stiffness_matrix(model)

    with monkeypatch.context() as context:
        context.setattr(
            q4_module,
            "_validate_q4_quadrature_authority",
            lambda _element: None,
        )
        context.setattr(q4_module, "_GAUSS", ((0.0, 0.0),))
        with pytest.raises(ValueError, match="qualified shell authority"):
            assemble_stiffness_matrix(model)

    with monkeypatch.context() as context:
        context.setattr(
            s3_module,
            "_validate_s3_quadrature_values",
            lambda _element: None,
        )
        context.setattr(s3_module, "TRIANGLE_QUADRATURE", ((0.0, 0.0, 0.5),))
        with pytest.raises(ValueError, match="qualified shell authority"):
            assemble_stiffness_matrix(model)


@pytest.mark.parametrize("element_id", (1, 2))
def test_qualified_mapping_key_must_match_owned_element_id(
    element_id: int,
) -> None:
    model = _mixed_model()
    object.__setattr__(model.mesh.elements[element_id], "element_id", 99)
    with pytest.raises(ElementCapabilityError, match="ELEMENT_MAPPING_ID_MISMATCH"):
        require_exact_qualified_component_lifecycle_api(
            model,
            context="mapping identity regression",
        )


@pytest.mark.parametrize("element_id", (1, 2))
def test_legacy_pickle_state_normalizes_connectivity_and_clears_caches(
    element_id: int,
) -> None:
    source = _mixed_model().mesh.elements[element_id]
    state = source.__getstate__()
    state["node_ids"] = list(state["node_ids"])
    state.pop("_qualified_plan_state_revision", None)
    state["_qualified_components"] = {"stale": True}
    restored = object.__new__(type(source))

    restored.__setstate__(state)

    assert type(restored.node_ids) is tuple
    assert restored._qualified_plan_state_revision == 0
    assert restored._qualified_components is None
