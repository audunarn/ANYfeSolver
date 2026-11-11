"""Focused qualification of the bounded qualified-S3 reference batch."""

from __future__ import annotations

import copy
from dataclasses import asdict
import inspect
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import warnings
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pytest

import anyfesolver.matrix_assembly as matrix_assembly_module
import anyfesolver.e4_pl_s3_element as s3_element_module
import anyfesolver.s3_reference_batch as s3_batch_module
from anyfesolver.assembly import (
    compute_constraint_force_diagnostics,
    compute_reactions,
)
from anyfesolver.activity import ElementActivity
from anyfesolver.boundary import LoadCase
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.fe_core import FEModel, FEMesh
from anyfesolver.matrix_assembly import AssemblyError, assemble_stiffness_matrix
from anyfesolver.materials import Hill48Yield
from anyfesolver.recovery import (
    RecoveryConfig,
    ResourceConfig,
    _compute_one_element_stress,
    recover_element_stresses_with_report,
    recover_stress_result,
)
from anyfesolver.recovery_batches import get_recovery_batch_plan
from anyfesolver.s3_reference_batch import (
    MIN_REFERENCE_S3_RECOVERY_GROUP,
    REFERENCE_S3_BATCH_POLICY_ID,
    REFERENCE_S3_FORMULATION_ID,
    get_reference_s3_stiffness_components,
    prepare_reference_s3_components,
    reference_s3_eligibility,
)
from anyfesolver.shell_sections import GeneralizedShellSection


def _build_model(
    count: int,
    *,
    include_q4: bool = False,
    shell_section: GeneralizedShellSection | None = None,
    material_name: str = "steel",
    reverse_winding: bool = False,
    reference_surface_offset: float = 0.0,
) -> FEModel:
    model = FEModel("qualified-s3-reference-batch")
    if material_name == "steel":
        model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    elif material_name == "history":
        model.add_material("history", 210.0e9, 0.3, density=7850.0)
        model.materials["history"].hardening_curve = object()
    elif material_name == "hill":
        model.add_material("hill", 210.0e9, 0.3, density=7850.0)
        model.materials["hill"].hill_yield = Hill48Yield(
            400.0e6,
            350.0e6,
            300.0e6,
            220.0e6,
            210.0e6,
            200.0e6,
        )
    elif material_name == "orthotropic":
        model.add_orthotropic_material(
            "orthotropic",
            150.0e9,
            90.0e9,
            70.0e9,
            0.18,
            0.16,
            0.15,
            48.0e9,
            42.0e9,
            36.0e9,
            density=1600.0,
        )
    else:  # pragma: no cover - helper contract
        raise AssertionError(material_name)

    next_node = 1
    elements = []
    for index in range(int(count)):
        x = float(8 * index)
        node_ids = [next_node, next_node + 1, next_node + 2]
        for node_id, coordinate in zip(
            node_ids,
            ((x, 0.0, 0.0), (x + 3.0, 0.0, 0.0), (x, 3.0, 0.0)),
        ):
            model.add_node(node_id, *coordinate)
        if reverse_winding:
            node_ids = [node_ids[0], node_ids[2], node_ids[1]]
        element_id = index + 1
        elements.append(
            (
                element_id,
                QualifiedE4PLS3ShellElement(
                    element_id,
                    node_ids,
                    material_name,
                    thickness=0.05,
                    shell_section=shell_section,
                    material_direction=(
                        np.asarray((1.0, 0.0, 0.0))
                        if material_name == "orthotropic"
                        else None
                    ),
                    reference_normal=(0.0, 0.0, 1.0),
                    reference_surface_offset=reference_surface_offset,
                ),
            )
        )
        next_node += 3

    if include_q4:
        q4_nodes = [next_node + offset for offset in range(4)]
        x = float(8 * count + 8)
        for node_id, coordinate in zip(
            q4_nodes,
            (
                (x, 0.0, 0.0),
                (x + 3.0, 0.0, 0.0),
                (x + 3.0, 3.0, 0.0),
                (x, 3.0, 0.0),
            ),
        ):
            model.add_node(node_id, *coordinate)
        element_id = int(count) + 1
        elements.append(
            (
                element_id,
                QualifiedE4PLShellElement(
                    element_id,
                    q4_nodes,
                    "steel",
                    thickness=0.05,
                ),
            )
        )
    for element_id, element in elements:
        model.add_element(element_id, element)
    return model


def _payload_bytes_equal(
    actual: Mapping[int, Mapping[str, Any]],
    expected: Mapping[int, Mapping[str, Any]],
) -> None:
    assert tuple(actual) == tuple(expected)
    for element_id in expected:
        assert tuple(actual[element_id]) == tuple(expected[element_id])
        for name, expected_value in expected[element_id].items():
            actual_value = actual[element_id][name]
            if isinstance(expected_value, np.ndarray):
                assert isinstance(actual_value, np.ndarray)
                assert actual_value.dtype == expected_value.dtype
                assert actual_value.shape == expected_value.shape
                assert actual_value.tobytes(order="C") == expected_value.tobytes(
                    order="C"
                )
            else:
                assert actual_value == expected_value


def _section() -> GeneralizedShellSection:
    return GeneralizedShellSection(
        A=np.diag((2.0e8, 2.0e8, 8.0e7)),
        B=np.zeros((3, 3)),
        D=np.diag((2.0e4, 2.0e4, 8.0e3)),
        As=np.diag((6.0e7, 6.0e7)),
        name="test-generalized",
    )


def _set_array_strides_for_authority_test(
    array: np.ndarray,
    strides: tuple[int, ...],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        array.strides = strides


def test_stiffness_batch_uses_one_native_component_evaluation_and_copied_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(8, include_q4=True)
    stiffness, info = assemble_stiffness_matrix(model)
    assert stiffness.shape == (
        model.mesh.dof_manager.total_dofs,
        model.mesh.dof_manager.total_dofs,
    )
    diagnostics = info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]
    assert diagnostics == {
        "policy_id": REFERENCE_S3_BATCH_POLICY_ID,
        "formulation_id": REFERENCE_S3_FORMULATION_ID,
        "scope": "reference_elastic_isotropic_positive_winding",
        "path": "formulation_native_shared_components",
        "candidate_element_count": 8,
        "element_count": 8,
        "translation_group_element_count": 8,
        "exact_element_cache_reuse_count": 0,
        "exact_translation_group_count": 1,
        "component_evaluation_count": 1,
        "matrix_shape_finite_symmetry_prevalidated": True,
        "element_ids": list(range(1, 9)),
        "group_element_ids": [list(range(1, 9))],
        "fallback_reasons": {},
        "revision_key": [
            model.mesh.revisions["topology"],
            model.mesh.revisions["geometry"],
            model.mesh.revisions["material"],
        ],
        "parallel_kernel": False,
        "legacy_stiffness_batch_eligible": False,
        "legacy_nonlinear_batch_eligible": False,
        "speedup_claimed": False,
        "plan_reused": False,
    }
    first = model.mesh.elements[1]
    second = model.mesh.elements[2]
    material = model.get_material(first.material_name)
    for element_id in range(1, 9):
        element = model.mesh.elements[element_id]
        cached_total = s3_element_module._try_s3_fast_assembly_cached_stiffness(
            element,
            model.mesh,
            material,
        )
        assert type(cached_total) is bytes
        assert len(cached_total) == 18 * 18 * 8
    assert first._qualified_cache_key == second._qualified_cache_key
    assert first._qualified_components is not second._qualified_components
    with pytest.raises(TypeError):
        first._qualified_components["total"] = np.zeros((18, 18))
    with pytest.raises(ValueError, match="WRITEABLE flag"):
        first._qualified_components["total"].setflags(write=True)
    with pytest.raises(TypeError):
        first._qualified_components["assumed_shear_samples"]["A"] = np.zeros(
            (2, 17)
        )
    with pytest.raises(ValueError, match="WRITEABLE flag"):
        first._qualified_components["assumed_shear_samples"]["A"].setflags(
            write=True
        )
    retained = model.mesh._qualified_s3_reference_stiffness_plan.matrices[1]
    assert retained.flags.owndata is False
    with pytest.raises(ValueError, match="WRITEABLE flag"):
        retained.setflags(write=True)
    assert not np.shares_memory(
        first._qualified_components["total"],
        second._qualified_components["total"],
    )
    np.testing.assert_array_equal(
        first._qualified_components["total"],
        second._qualified_components["total"],
    )
    assert first.legacy_stiffness_batch_eligible is False
    assert first.legacy_nonlinear_batch_eligible is False

    scalar_model = _build_model(8, include_q4=True)
    original_get = s3_batch_module.get_reference_s3_stiffness_components

    def force_scalar(candidate_model, items, **_kwargs):
        return (
            prepare_reference_s3_components(
                candidate_model,
                items,
                minimum_group_size=10_000,
            ),
            False,
        )

    monkeypatch.setattr(
        s3_batch_module,
        "get_reference_s3_stiffness_components",
        force_scalar,
    )
    scalar_stiffness, scalar_info = assemble_stiffness_matrix(scalar_model)
    np.testing.assert_array_equal(stiffness.toarray(), scalar_stiffness.toarray())
    assert scalar_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["element_count"] == 0
    monkeypatch.setattr(
        s3_batch_module,
        "get_reference_s3_stiffness_components",
        original_get,
    )

    # Activity/deletion is applied after the native matrices are prepared.
    activity = ElementActivity(range(1, 10))
    model.set_element_activity(activity)
    activity.set_activity([2], [0.25], reason="batch-qualification")
    activity.hard_delete([8], reason="batch-qualification")
    scaled, scaled_info = assemble_stiffness_matrix(model)
    dofs_2 = np.asarray(model.mesh.elements[2].get_dof_mapping(model.mesh))
    dofs_8 = np.asarray(model.mesh.elements[8].get_dof_mapping(model.mesh))
    baseline = stiffness.toarray()
    made = scaled.toarray()
    np.testing.assert_array_equal(
        made[np.ix_(dofs_2, dofs_2)],
        0.25 * baseline[np.ix_(dofs_2, dofs_2)],
    )
    assert not np.any(made[np.ix_(dofs_8, dofs_8)])
    assert scaled_info["diagnostics"]["element_activity"][
        "zero_contribution_count"
    ] == 1
    assert scaled_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True


def test_warm_immutable_s3_plan_reuses_one_time_matrix_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(8)
    baseline, _baseline_info = assemble_stiffness_matrix(model)
    plan = model.mesh._qualified_s3_reference_stiffness_plan
    assert plan.matrices_prevalidated is True
    assert all(
        matrix.flags.writeable is False for matrix in plan.matrices.values()
    )

    calls = 0
    original = matrix_assembly_module._relative_symmetry_error

    def counted(matrix: Any) -> float:
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(
        matrix_assembly_module,
        "_relative_symmetry_error",
        counted,
    )
    repeated, info = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(baseline.toarray(), repeated.toarray())
    assert info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True
    # The warm plan retains its exact local-matrix certificates and the
    # assembly operation uses the closure-bound symmetry kernel captured at
    # import.  Replacing the mutable module alias therefore cannot instrument
    # or influence either the certified local matrices or the global check.
    assert calls == 0

    element = model.mesh.elements[1]
    original_mapping = np.asarray(element.get_dof_mapping(model.mesh))
    monkeypatch.setattr(
        element,
        "get_dof_mapping",
        lambda _mesh: original_mapping[:-1],
    )
    with pytest.raises(
        matrix_assembly_module.AssemblyError,
        match="incompatible qualified shell authority",
    ):
        assemble_stiffness_matrix(model)


@pytest.mark.parametrize(
    "mutation",
    ("shape", "strides", "writeable", "value"),
)
def test_warm_plan_rebuilds_after_retained_matrix_authority_changes(
    mutation: str,
) -> None:
    model = _build_model(8)
    baseline, _ = assemble_stiffness_matrix(model)
    _warm, warm_info = assemble_stiffness_matrix(model)
    assert warm_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True

    plan = model.mesh._qualified_s3_reference_stiffness_plan
    original_matrices = plan.matrices
    retained = original_matrices[1]
    original_shape = retained.shape
    original_strides = retained.strides
    if mutation == "shape":
        retained.shape = (9, 36)
    elif mutation == "strides":
        _set_array_strides_for_authority_test(retained, (0, 8))
    else:
        replacement = (
            np.zeros((18, 18), dtype=np.float64)
            if mutation == "writeable"
            else np.frombuffer(bytes(18 * 18 * 8), dtype=np.float64).reshape(
                (18, 18)
            )
        )
        object.__setattr__(
            plan,
            "matrices",
            MappingProxyType({**dict(original_matrices), 1: replacement}),
        )
    try:
        rebuilt, rebuilt_info = assemble_stiffness_matrix(model)
    finally:
        retained.shape = original_shape
        _set_array_strides_for_authority_test(retained, original_strides)
        object.__setattr__(plan, "matrices", original_matrices)
    np.testing.assert_array_equal(rebuilt.toarray(), baseline.toarray())
    assert rebuilt_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is False

    next_clean, next_clean_info = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(next_clean.toarray(), baseline.toarray())
    assert next_clean_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True


def test_warm_plan_matrix_metadata_restore_during_use_cannot_change_output() -> None:
    model = _build_model(8)
    baseline, _ = assemble_stiffness_matrix(model)
    _warm, warm_info = assemble_stiffness_matrix(model)
    assert warm_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True
    plan = model.mesh._qualified_s3_reference_stiffness_plan
    retained = plan.matrices[1]
    original_strides = retained.strides

    implementation = matrix_assembly_module._assemble_element_matrix_under_lease_impl
    source, start = inspect.getsourcelines(implementation)
    mutate_line = start + next(
        index
        for index, text in enumerate(source)
        if "rows = _ndarray_constructor(" in text
    )
    restore_line = start + next(
        index
        for index, text in enumerate(source)
        if index > mutate_line - start
        and "data = _ndarray_constructor(" in text
    )
    state = {"mutated": False, "restored": False}

    def trace(frame: Any, event: str, _argument: Any) -> Any:
        if event == "line" and frame.f_code is implementation.__code__:
            if frame.f_lineno == mutate_line:
                _set_array_strides_for_authority_test(retained, (0, 8))
                state["mutated"] = True
            elif frame.f_lineno == restore_line and state["mutated"]:
                _set_array_strides_for_authority_test(
                    retained,
                    original_strides,
                )
                state["restored"] = True
        return trace

    sys.settrace(trace)
    try:
        actual, info = assemble_stiffness_matrix(model)
    finally:
        sys.settrace(None)
        _set_array_strides_for_authority_test(retained, original_strides)
    assert state == {"mutated": True, "restored": True}
    np.testing.assert_array_equal(actual.toarray(), baseline.toarray())
    assert info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True


def test_stiffness_batch_is_revision_bound_and_small_groups_fall_back() -> None:
    small = _build_model(7)
    _matrix, small_info = assemble_stiffness_matrix(small)
    small_diagnostics = small_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]
    assert small_diagnostics["element_count"] == 0
    assert small_diagnostics["translation_group_element_count"] == 0
    assert small_diagnostics["exact_element_cache_reuse_count"] == 0
    assert small_diagnostics["component_evaluation_count"] == 0
    assert small_diagnostics["fallback_reasons"] == {
        "group_below_minimum_size": list(range(1, 8))
    }
    assert small_info["diagnostics"]["scalar_shell_element_count"] == 7

    _warm_matrix, warm_info = assemble_stiffness_matrix(small)
    warm_diagnostics = warm_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]
    assert warm_diagnostics["path"] == "formulation_native_exact_cache_reuse"
    assert warm_diagnostics["translation_group_element_count"] == 0
    assert warm_diagnostics["exact_element_cache_reuse_count"] == 7
    cache_groups = [
        group
        for group in warm_info["diagnostics"]["vectorized_shell_groups"]
        if group["kernel"] == "qualified_s3_exact_element_cache_reuse"
    ]
    assert len(cache_groups) == 1
    assert cache_groups[0]["unique_geometry_count"] == 1

    model = _build_model(8)
    _first, first_info = assemble_stiffness_matrix(model)
    first_revision = first_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["revision_key"]
    for node_id, node in tuple(model.mesh.nodes.items()):
        model.set_node_coordinates(node_id, node.x + 16.0, node.y, node.z)
    _second, second_info = assemble_stiffness_matrix(model)
    second_diagnostics = second_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]
    assert second_diagnostics["element_count"] == 8
    assert second_diagnostics["component_evaluation_count"] == 1
    assert second_diagnostics["revision_key"] != first_revision


def test_partial_helper_request_never_reuses_a_complete_candidate_plan() -> None:
    model = _build_model(8)
    items = list(model.mesh.elements.items())
    full, full_reused = get_reference_s3_stiffness_components(
        model,
        items,
        complete_candidate_items=True,
    )
    assert full_reused is False
    assert len(full.matrices) == 8
    repeated, repeated_reused = get_reference_s3_stiffness_components(
        model,
        items,
        complete_candidate_items=True,
    )
    assert repeated_reused is True
    assert len(repeated.matrices) == 8

    partial, partial_reused = get_reference_s3_stiffness_components(
        model,
        items[:4],
    )
    assert partial_reused is False
    assert partial.candidate_element_ids == (1, 2, 3, 4)
    assert tuple(partial.matrices) == (1, 2, 3, 4)


def test_warm_plan_reuses_exact_binary64_element_caches_without_rounding() -> None:
    model = _build_model(8)
    # Decimal grid translations produce distinct binary64 centered-coordinate
    # keys even though the triangles are nominally translated copies.  The
    # cache plan must reuse each element's own exact matrix, never round the
    # geometry or substitute a neighbouring element's matrix.
    for node_id, node in tuple(model.mesh.nodes.items()):
        model.set_node_coordinates(
            node_id,
            0.05 * float(node.x),
            0.05 * float(node.y),
            float(node.z),
        )

    first, first_info = assemble_stiffness_matrix(model)
    first_diagnostics = first_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]
    assert first_diagnostics["element_count"] == 0
    assert first_diagnostics["fallback_reasons"] == {
        "group_below_minimum_size": list(range(1, 9))
    }

    second, second_info = assemble_stiffness_matrix(model)
    second_diagnostics = second_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]
    assert second_diagnostics["path"].endswith("exact_cache_reuse")
    assert second_diagnostics["element_count"] == 8
    assert second_diagnostics["translation_group_element_count"] == 0
    assert second_diagnostics["exact_element_cache_reuse_count"] == 8
    assert second_diagnostics["component_evaluation_count"] == 0
    assert second_diagnostics["fallback_reasons"] == {}
    assert second_diagnostics["plan_reused"] is False

    third, third_info = assemble_stiffness_matrix(model)
    assert third_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True
    np.testing.assert_array_equal(first.toarray(), second.toarray())
    np.testing.assert_array_equal(second.toarray(), third.toarray())

    # Direct element-property edits do not advance the mesh revision.  The
    # plan must therefore revalidate the formulation's complete cache key and
    # must never serve the stale pre-edit matrix.
    element = model.mesh.elements[1]
    element.thickness *= 2.0
    changed_thickness, thickness_info = assemble_stiffness_matrix(model)
    assert thickness_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is False
    assert not np.array_equal(third.toarray(), changed_thickness.toarray())

    # The next pass may capture the newly populated exact element cache; only
    # the following pass is a reusable, fully rebound plan.
    rebound, rebound_info = assemble_stiffness_matrix(model)
    assert rebound_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is False
    reused, reused_info = assemble_stiffness_matrix(model)
    assert reused_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True
    np.testing.assert_array_equal(changed_thickness.toarray(), rebound.toarray())
    np.testing.assert_array_equal(rebound.toarray(), reused.toarray())

    # Eligibility is also live state rather than a mesh revision.  Switching
    # to an oriented material path must evict the element from this narrow
    # isotropic reference plan before evaluation.
    element.material_direction = np.asarray((1.0, 0.0, 0.0))
    _oriented, oriented_info = assemble_stiffness_matrix(model)
    oriented_diagnostics = oriented_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]
    assert oriented_diagnostics["plan_reused"] is False
    assert oriented_diagnostics["fallback_reasons"]["oriented_material"] == [1]

    # Revision invalidation clears the exact element caches and the mesh-owned
    # plan; the next assembly must evaluate every element again.
    node = model.mesh.nodes[1]
    model.set_node_coordinates(1, node.x - 1.0e-4, node.y, node.z)
    _changed, changed_info = assemble_stiffness_matrix(model)
    assert changed_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["element_count"] == 0


def test_warm_plan_detects_direct_node_edits_without_a_mesh_revision() -> None:
    model = _build_model(8)
    baseline, _baseline_info = assemble_stiffness_matrix(model)
    _warm, warm_info = assemble_stiffness_matrix(model)
    assert warm_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True

    geometry_revision = model.mesh.revisions["geometry"]
    coordinate_revision = model.mesh.nodes[1]._coordinate_revision
    direct_revision = model.mesh._qualified_direct_state_token[0]
    model.mesh.nodes[1].x -= 1.0e-4
    assert model.mesh._qualified_direct_state_token[0] == direct_revision + 1
    changed, changed_info = assemble_stiffness_matrix(model)
    assert model.mesh.revisions["geometry"] == geometry_revision
    assert model.mesh.nodes[1]._coordinate_revision == coordinate_revision + 1
    assert changed_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is False
    assert not np.array_equal(baseline.toarray(), changed.toarray())

    element = model.mesh.elements[1]
    with pytest.raises(ValueError, match="read-only"):
        element.reference_normal[0] = 0.25


def test_warm_plan_binds_direct_elastic_material_edits() -> None:
    model = _build_model(8)
    baseline, _baseline_info = assemble_stiffness_matrix(model)
    _warm, warm_info = assemble_stiffness_matrix(model)
    assert warm_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True

    material = model.materials["steel"]
    material_revision = model.mesh.revisions["material"]
    material.elastic_modulus *= 0.75
    changed, changed_info = assemble_stiffness_matrix(model)
    assert model.mesh.revisions["material"] == material_revision
    assert changed_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is False
    assert not np.array_equal(baseline.toarray(), changed.toarray())


def test_warm_plan_detects_and_rebinds_public_mapping_replacements() -> None:
    node_model = _build_model(8)
    baseline, _baseline_info = assemble_stiffness_matrix(node_model)
    _warm, warm_info = assemble_stiffness_matrix(node_model)
    assert warm_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True

    replacement_node = copy.deepcopy(node_model.mesh.nodes[1])
    replacement_node.x -= 1.0e-4
    node_model.mesh.nodes[1] = replacement_node
    changed_node, node_info = assemble_stiffness_matrix(node_model)
    assert node_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is False
    assert not np.array_equal(baseline.toarray(), changed_node.toarray())
    assert replacement_node._qualified_direct_state_token is (
        node_model.mesh._qualified_direct_state_token
    )
    token_before_node_edit = node_model.mesh._qualified_direct_state_token[0]
    replacement_node.x -= 1.0e-4
    assert (
        node_model.mesh._qualified_direct_state_token[0]
        == token_before_node_edit + 1
    )

    element_model = _build_model(8)
    baseline_element, _baseline_element_info = assemble_stiffness_matrix(
        element_model
    )
    _warm_element, warm_element_info = assemble_stiffness_matrix(element_model)
    assert warm_element_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True
    replacement_element = copy.deepcopy(element_model.mesh.elements[1])
    replacement_element.thickness *= 2.0
    element_model.mesh.elements[1] = replacement_element
    changed_element, element_info = assemble_stiffness_matrix(element_model)
    assert element_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is False
    assert not np.array_equal(
        baseline_element.toarray(), changed_element.toarray()
    )
    assert replacement_element._qualified_direct_state_token is (
        element_model.mesh._qualified_direct_state_token
    )
    token_before_element_edit = element_model.mesh._qualified_direct_state_token[0]
    replacement_element.thickness *= 1.1
    assert (
        element_model.mesh._qualified_direct_state_token[0]
        == token_before_element_edit + 1
    )


def test_warm_plan_binds_reference_normal_bytes_even_if_writes_are_reenabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(8)
    baseline, _baseline_info = assemble_stiffness_matrix(model)
    repeated, repeated_info = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(baseline.toarray(), repeated.toarray())
    assert repeated_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True

    element = model.mesh.elements[1]
    with pytest.raises(ValueError, match="WRITEABLE flag"):
        element.reference_normal.setflags(write=True)
    element.reference_normal = np.asarray((0.0, 0.0, -1.0))
    assert element.reference_normal.flags.writeable is False
    assert reference_s3_eligibility(model, element) == (
        False,
        "nonpositive_winding",
    )
    with pytest.raises(ValueError, match="winding opposes"):
        assemble_stiffness_matrix(model)


def test_warm_assembly_lease_rejects_transient_runtime_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(8)
    baseline, _baseline_info = assemble_stiffness_matrix(model)
    _warm, warm_info = assemble_stiffness_matrix(model)
    assert warm_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True

    original_revision_signature = model.mesh.revision_signature
    original_triangle_frame = s3_element_module.triangle_frame
    callback_count = 0

    def transient_revision_signature() -> dict[str, int]:
        nonlocal callback_count
        callback_count += 1
        setattr(s3_element_module, "triangle_frame", lambda *args, **kwargs: None)
        setattr(s3_element_module, "triangle_frame", original_triangle_frame)
        return original_revision_signature()

    monkeypatch.setattr(
        model.mesh,
        "revision_signature",
        transient_revision_signature,
    )
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        assemble_stiffness_matrix(model)
    # Exact warm assembly rejects the provider shadow at the owned-input
    # boundary.  The callback therefore cannot run far enough to perform its
    # transient formulation mutation.
    assert callback_count == 0
    assert all(
        element._qualified_components is not None
        for element in model.mesh.elements.values()
    )

    monkeypatch.delattr(model.mesh, "revision_signature")
    restored, _restored_info = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(restored.toarray(), baseline.toarray())


@pytest.mark.parametrize("entrypoint", ("get", "prepare"))
def test_direct_batch_lease_rejects_transient_runtime_mutation(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    model = _build_model(8)
    assemble_stiffness_matrix(model)
    assemble_stiffness_matrix(model)
    items = tuple(model.mesh.elements.items())

    original_revision_signature = model.mesh.revision_signature
    original_triangle_frame = s3_element_module.triangle_frame
    callback_count = 0

    def transient_revision_signature() -> dict[str, int]:
        nonlocal callback_count
        if callback_count == 0:
            setattr(
                s3_element_module,
                "triangle_frame",
                lambda *args, **kwargs: None,
            )
            setattr(
                s3_element_module,
                "triangle_frame",
                original_triangle_frame,
            )
        callback_count += 1
        return original_revision_signature()

    monkeypatch.setattr(
        model.mesh,
        "revision_signature",
        transient_revision_signature,
    )
    with pytest.raises(ValueError, match="authority changed"):
        if entrypoint == "get":
            get_reference_s3_stiffness_components(
                model,
                items,
                complete_candidate_items=True,
            )
        else:
            prepare_reference_s3_components(model, items)

    assert callback_count >= 1
    assert not hasattr(
        model.mesh,
        "_qualified_s3_reference_stiffness_plan",
    )
    assert all(
        element._qualified_components is None
        for element in model.mesh.elements.values()
    )


@pytest.mark.parametrize(
    "entrypoint",
    (compute_constraint_force_diagnostics, compute_reactions),
)
def test_constraint_routes_reject_transient_load_provider_mutation(
    entrypoint: Any,
) -> None:
    model = _build_model(1)
    displacements = np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64)
    clean_load = LoadCase("clean-zero")
    baseline = entrypoint(model, displacements, clean_load)
    original_triangle_frame = s3_element_module.triangle_frame
    callback_count = 0

    class TransientLoadProvider:
        def get_load_vector(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
            nonlocal callback_count
            callback_count += 1
            setattr(
                s3_element_module,
                "triangle_frame",
                lambda *args, **kwargs: None,
            )
            setattr(
                s3_element_module,
                "triangle_frame",
                original_triangle_frame,
            )
            return np.zeros(
                model.mesh.dof_manager.total_dofs,
                dtype=np.float64,
            )

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        entrypoint(model, displacements, TransientLoadProvider())
    assert callback_count == 1

    restored = entrypoint(model, displacements, clean_load)
    assert tuple(restored) == tuple(baseline)
    for name, value in baseline.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(restored[name], value)


def test_direct_load_case_rejects_transient_material_getter_mutation() -> None:
    model = _build_model(1)
    load_case = LoadCase("qualified-s3-gravity")
    load_case.set_gravity(gz=-9.81)
    baseline = load_case.get_load_vector(
        model.mesh,
        model.mesh.dof_manager,
        model.get_material,
    )
    original_triangle_frame = s3_element_module.triangle_frame
    callback_count = 0

    def transient_material_getter(name: str) -> Any:
        nonlocal callback_count
        callback_count += 1
        setattr(
            s3_element_module,
            "triangle_frame",
            lambda *args, **kwargs: None,
        )
        setattr(
            s3_element_module,
            "triangle_frame",
            original_triangle_frame,
        )
        return model.get_material(name)

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        load_case.get_load_vector(
            model.mesh,
            model.mesh.dof_manager,
            transient_material_getter,
        )
    assert callback_count == 1

    restored = load_case.get_load_vector(
        model.mesh,
        model.mesh.dof_manager,
        model.get_material,
    )
    np.testing.assert_array_equal(restored, baseline)


def test_model_deepcopy_drops_derived_plan_and_rebinds_direct_state_tokens() -> None:
    model = _build_model(8)
    baseline, _baseline_info = assemble_stiffness_matrix(model)
    _warm, warm_info = assemble_stiffness_matrix(model)
    assert warm_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True
    original_direct_revision = model.mesh._qualified_direct_state_token[0]

    copied = copy.deepcopy(model)
    assert not hasattr(copied.mesh, "_qualified_s3_reference_stiffness_plan")
    assert copied.mesh._qualified_direct_state_token is not (
        model.mesh._qualified_direct_state_token
    )
    assert all(
        node._qualified_direct_state_token
        is copied.mesh._qualified_direct_state_token
        and len(node._qualified_direct_state_tokens) == 1
        for node in copied.mesh.nodes.values()
    )
    assert all(
        element._qualified_direct_state_token
        is copied.mesh._qualified_direct_state_token
        and len(element._qualified_direct_state_tokens) == 1
        for element in copied.mesh.elements.values()
    )
    assert all(
        element.reference_normal.flags.writeable is False
        and (
            element.material_direction is None
            or element.material_direction.flags.writeable is False
        )
        for element in copied.mesh.elements.values()
    )
    with pytest.raises(ValueError, match="WRITEABLE flag"):
        copied.mesh.elements[1].reference_normal.setflags(write=True)
    copied_matrix, _copied_info = assemble_stiffness_matrix(copied)
    np.testing.assert_array_equal(baseline.toarray(), copied_matrix.toarray())

    original_x = model.mesh.nodes[1].x
    copied_direct_revision = copied.mesh._qualified_direct_state_token[0]
    copied.mesh.nodes[1].x -= 1.0e-4
    assert model.mesh.nodes[1].x == original_x
    assert copied.mesh._qualified_direct_state_token[0] == copied_direct_revision + 1
    assert (
        model.mesh._qualified_direct_state_token[0]
        == original_direct_revision
    )


def test_tracked_mesh_mappings_remain_dataclass_and_json_serializable() -> None:
    payload = asdict(FEMesh())
    encoded = json.dumps(
        {"nodes": payload["nodes"], "elements": payload["elements"]},
        sort_keys=True,
    )
    assert '"nodes"' in encoded
    assert '"elements"' in encoded

    mesh = FEMesh()
    mesh.add_node(1, 0.0, 0.0, 0.0)
    restored = pickle.loads(pickle.dumps(mesh))
    assert restored.nodes[1]._qualified_direct_state_token is (
        restored._qualified_direct_state_token
    )

    warm_model = _build_model(8)
    baseline, _baseline_info = assemble_stiffness_matrix(warm_model)
    assert hasattr(
        warm_model.mesh, "_qualified_s3_reference_stiffness_plan"
    )
    restored_model = pickle.loads(pickle.dumps(warm_model))
    assert not hasattr(
        restored_model.mesh, "_qualified_s3_reference_stiffness_plan"
    )
    assert all(
        element._qualified_components is None
        for element in restored_model.mesh.elements.values()
    )
    restored_matrix, _restored_info = assemble_stiffness_matrix(restored_model)
    np.testing.assert_array_equal(
        baseline.toarray(), restored_matrix.toarray()
    )


def test_standalone_element_deepcopy_stays_immutable_when_added() -> None:
    source_model = _build_model(1)
    standalone = copy.deepcopy(source_model.mesh.elements[1])
    assert not hasattr(standalone, "_qualified_direct_state_token")
    assert standalone.reference_normal.flags.writeable is False
    with pytest.raises(ValueError, match="WRITEABLE flag"):
        standalone.reference_normal.setflags(write=True)

    target = _build_model(8)
    replacement_id = 1
    standalone.element_id = replacement_id
    target.mesh.elements.pop(replacement_id)
    target.mesh.add_element(replacement_id, standalone)
    assert standalone._qualified_direct_state_token is (
        target.mesh._qualified_direct_state_token
    )
    assert standalone.reference_normal.flags.writeable is False
    _baseline, _baseline_info = assemble_stiffness_matrix(target)
    _warm, warm_info = assemble_stiffness_matrix(target)
    assert warm_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True
    with pytest.raises(ValueError, match="WRITEABLE flag"):
        standalone.reference_normal.setflags(write=True)


def test_shared_element_invalidates_every_owning_mesh_plan() -> None:
    first = _build_model(8)
    shared = first.mesh.elements[1]
    assemble_stiffness_matrix(first)
    assemble_stiffness_matrix(first)

    second = _build_model(8)
    second.mesh.elements.pop(1)
    second.mesh.add_element(1, shared)
    baseline_second, _baseline_second_info = assemble_stiffness_matrix(second)
    _warm_second, warm_second_info = assemble_stiffness_matrix(second)
    assert warm_second_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True
    assemble_stiffness_matrix(first)
    _warm_first, warm_first_info = assemble_stiffness_matrix(first)
    assert warm_first_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True

    first_token = first.mesh._qualified_direct_state_token[0]
    second_token = second.mesh._qualified_direct_state_token[0]
    shared.thickness *= 2.0
    assert first.mesh._qualified_direct_state_token[0] == first_token + 1
    assert second.mesh._qualified_direct_state_token[0] == second_token + 1
    changed_second, changed_second_info = assemble_stiffness_matrix(second)
    assert changed_second_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is False
    assert not np.array_equal(
        baseline_second.toarray(), changed_second.toarray()
    )

def test_generalized_orthotropic_history_and_winding_never_enter_batch() -> None:
    cases = (
        (_build_model(1, shell_section=_section()), "generalized_section"),
        (_build_model(1, material_name="orthotropic"), "orthotropic_or_anisotropic_material"),
        (_build_model(1, material_name="hill"), "hill_material"),
        (_build_model(1, material_name="history"), "material_history"),
        (_build_model(1, reverse_winding=True), "nonpositive_winding"),
    )
    for model, expected_reason in cases:
        element = model.mesh.elements[1]
        eligible, reason = reference_s3_eligibility(model, element)
        assert eligible is False
        assert reason == expected_reason
        prepared = prepare_reference_s3_components(
            model,
            [(1, element)],
            minimum_group_size=1,
        )
        assert prepared.matrices == {}
        assert prepared.component_evaluation_count == 0
        assert prepared.fallback_reasons == {expected_reason: (1,)}


def test_reference_surface_offset_never_enters_reference_batch() -> None:
    model = _build_model(1, reference_surface_offset=0.125)
    element = model.mesh.elements[1]

    eligible, reason = reference_s3_eligibility(model, element)
    assert eligible is False
    assert reason == "nonzero_reference_surface_offset"

    prepared = prepare_reference_s3_components(
        model,
        [(1, element)],
        minimum_group_size=1,
    )
    assert prepared.matrices == {}
    assert prepared.component_evaluation_count == 0
    assert prepared.fallback_reasons == {"nonzero_reference_surface_offset": (1,)}


def test_large_mixed_recovery_is_byte_identical_to_scalar_and_reuses_plan() -> None:
    model = _build_model(MIN_REFERENCE_S3_RECOVERY_GROUP, include_q4=True)
    vector = 2.0e-5 * np.sin(
        np.arange(model.mesh.dof_manager.total_dofs, dtype=float) + 0.375
    )
    scalar = {}
    for element_id in model.mesh.elements:
        item = _compute_one_element_stress(
            model,
            vector,
            element_id,
            return_global=True,
        )
        assert item is not None
        scalar[item[0]] = item[1]

    # Force a clean revision-bound batch construction after the independent
    # per-element scalar oracle populated its own caches.
    model.bump_revision("material")
    batched, first_report = recover_element_stresses_with_report(
        model,
        vector,
        RecoveryConfig(),
        return_global=True,
        resource_config=ResourceConfig(recovery_threads=4),
    )
    repeated, second_report = recover_element_stresses_with_report(
        model,
        vector,
        RecoveryConfig(),
        return_global=True,
        resource_config=ResourceConfig(recovery_threads=1),
    )
    _payload_bytes_equal(batched, scalar)
    _payload_bytes_equal(repeated, batched)
    assert list(batched) == list(model.mesh.elements)
    assert first_report.metadata["recovery_backend"] == (
        "hybrid_qualified_s3_reference_thread_pool"
    )
    assert first_report.metadata["qualified_s3_reference_batch_count"] == 1
    assert first_report.metadata["compiled_batch_count"] == 0
    assert first_report.metadata["eligible_element_count"] == (
        MIN_REFERENCE_S3_RECOVERY_GROUP
    )
    assert first_report.metadata["fallback_element_count"] == 1
    assert first_report.metadata["fallback_reasons"] == {
        "batch_below_minimum_size": {
            "element_ids": [MIN_REFERENCE_S3_RECOVERY_GROUP + 1],
            "minimum_size": 100,
        }
    }
    batch_diagnostics = first_report.metadata[
        "qualified_s3_reference_batch"
    ]
    assert batch_diagnostics["policy_id"] == REFERENCE_S3_BATCH_POLICY_ID
    assert batch_diagnostics["formulation_id"] == REFERENCE_S3_FORMULATION_ID
    assert batch_diagnostics["component_evaluation_count"] == 1
    assert batch_diagnostics["exact_translation_group_count"] == 1
    assert batch_diagnostics["speedup_claimed"] is False
    assert batch_diagnostics["legacy_stiffness_batch_eligible"] is False
    assert batch_diagnostics["legacy_nonlinear_batch_eligible"] is False
    assert first_report.metadata["plan_reused"] is False
    assert second_report.metadata["plan_reused"] is True


def test_recovery_plan_invalidates_on_direct_qualified_state_mutation() -> None:
    model = _build_model(MIN_REFERENCE_S3_RECOVERY_GROUP)
    vector = 2.0e-5 * np.sin(
        np.arange(model.mesh.dof_manager.total_dofs, dtype=float) + 0.375
    )
    config = RecoveryConfig()
    resources = ResourceConfig(recovery_threads=1)
    baseline, _baseline_report = recover_element_stresses_with_report(
        model,
        vector,
        config,
        return_global=True,
        resource_config=resources,
    )
    _warm, _warm_report = recover_element_stresses_with_report(
        model,
        vector,
        config,
        return_global=True,
        resource_config=resources,
    )
    _warm_plan, warm_plan_reused = get_recovery_batch_plan(model)
    assert warm_plan_reused is True

    model.mesh.elements[1].thickness *= 2.0
    changed, changed_report = recover_element_stresses_with_report(
        model,
        vector,
        config,
        return_global=True,
        resource_config=resources,
    )
    assert changed_report.metadata["plan_reused"] is False
    assert any(
        isinstance(value, np.ndarray)
        and not np.array_equal(value, baseline[1][name])
        for name, value in changed[1].items()
    )
    scalar_item = _compute_one_element_stress(
        model,
        vector,
        1,
        return_global=True,
    )
    assert scalar_item is not None
    _payload_bytes_equal({1: changed[1]}, {1: scalar_item[1]})

    model.materials["steel"].elastic_modulus *= 0.9
    changed_material, changed_material_report = (
        recover_element_stresses_with_report(
            model,
            vector,
            config,
            return_global=True,
            resource_config=resources,
        )
    )
    assert changed_material_report.metadata["plan_reused"] is False
    scalar_material_item = _compute_one_element_stress(
        model,
        vector,
        1,
        return_global=True,
    )
    assert scalar_material_item is not None
    _payload_bytes_equal(
        {1: changed_material[1]},
        {1: scalar_material_item[1]},
    )

    # Use a fresh homogeneous group for material-option eligibility.  The
    # thickness mutation above intentionally split ``model`` into exact groups
    # of 1 and 127, neither of which meets the 128-element recovery threshold.
    material_model = _build_model(MIN_REFERENCE_S3_RECOVERY_GROUP)
    baseline_plan, baseline_reused = get_recovery_batch_plan(material_model)
    assert baseline_reused is False
    assert baseline_plan.reference_s3 is not None
    material = material_model.materials["steel"]
    material.hardening_curve = object()
    hardening_plan, hardening_reused = get_recovery_batch_plan(material_model)
    assert hardening_reused is False
    assert hardening_plan.reference_s3 is None
    material.hardening_curve = None
    restored_plan, restored_reused = get_recovery_batch_plan(material_model)
    assert restored_reused is False
    assert restored_plan.reference_s3 is not None
    material.hill_yield = object()
    hill_plan, hill_reused = get_recovery_batch_plan(material_model)
    assert hill_reused is False
    assert hill_plan.reference_s3 is None


def test_benchmark_rejects_unbounded_group_and_thread_configuration() -> None:
    root = Path(__file__).resolve().parents[1]
    benchmark = root / "scripts" / "benchmark_e4_pl_s3_reference_batch.py"
    base_command = [sys.executable, str(benchmark), "--repeats", "11"]

    invalid_threads = dict(os.environ)
    thread_names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in thread_names:
        invalid_threads[name] = "4"
    thread_result = subprocess.run(
        [*base_command, "--elements", str(MIN_REFERENCE_S3_RECOVERY_GROUP)],
        cwd=root,
        env=invalid_threads,
        capture_output=True,
        text=True,
        check=False,
    )
    assert thread_result.returncode != 0
    assert "thread environment variables to equal 1" in thread_result.stderr

    bounded_threads = dict(os.environ)
    for name in thread_names:
        bounded_threads[name] = "1"
    group_result = subprocess.run(
        [*base_command, "--elements", str(MIN_REFERENCE_S3_RECOVERY_GROUP - 1)],
        cwd=root,
        env=bounded_threads,
        capture_output=True,
        text=True,
        check=False,
    )
    assert group_result.returncode != 0
    assert "recovery batch minimum" in group_result.stderr


def test_recovery_rebinds_wholesale_public_mapping_replacements() -> None:
    model = _build_model(MIN_REFERENCE_S3_RECOVERY_GROUP)
    vector = 2.0e-5 * np.sin(
        np.arange(model.mesh.dof_manager.total_dofs, dtype=float) + 0.375
    )
    config = RecoveryConfig()
    resources = ResourceConfig(recovery_threads=1)
    recover_element_stresses_with_report(
        model,
        vector,
        config,
        return_global=True,
        resource_config=resources,
    )
    _warm, _warm_report = recover_element_stresses_with_report(
        model,
        vector,
        config,
        return_global=True,
        resource_config=resources,
    )
    _warm_plan, warm_plan_reused = get_recovery_batch_plan(model)
    assert warm_plan_reused is True

    model.mesh.elements = {
        element_id: copy.deepcopy(element)
        for element_id, element in model.mesh.elements.items()
    }
    _rebound_plan, rebound_plan_reused = get_recovery_batch_plan(model)
    assert rebound_plan_reused is False
    recover_element_stresses_with_report(
        model,
        vector,
        config,
        return_global=True,
        resource_config=resources,
    )
    assert all(
        element._qualified_direct_state_token is (
            model.mesh._qualified_direct_state_token
        )
        for element in model.mesh.elements.values()
    )
    token = model.mesh._qualified_direct_state_token[0]
    model.mesh.elements[1].thickness *= 2.0
    assert model.mesh._qualified_direct_state_token[0] == token + 1
    _changed_plan, changed_plan_reused = get_recovery_batch_plan(model)
    assert changed_plan_reused is False


def test_committed_state_recovery_stays_outside_reference_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(1)
    element = model.mesh.elements[1]
    material = model.get_material("steel")
    state = element.init_model_bound_nonlinear_state(model.mesh, material, 3)
    vector = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("committed recovery entered the reference-elastic batch")

    monkeypatch.setattr("anyfesolver.recovery.recover_reference_s3", forbidden)
    result = recover_stress_result(
        model,
        vector,
        RecoveryConfig(),
        element_states={1: state},
        return_global=True,
    )
    assert result.provenance.mode == "material_history"
    assert result.provenance.per_element_source == {
        1: "committed_qualified_s3_native_state"
    }
    assert result.execution_report.metadata.get(
        "qualified_s3_reference_batch_count", 0
    ) == 0
