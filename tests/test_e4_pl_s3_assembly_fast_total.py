from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import numpy as np
import pytest

import anyfesolver.e4_pl_s3_element as s3_module
import anyfesolver.s3_reference_batch as s3_batch_module
from anyfesolver import (
    FEModel,
    QualifiedE4PLS3ShellElement,
    QualifiedE4PLShellElement,
    assemble_stiffness_matrix,
)
from anyfesolver.matrix_assembly import AssemblyError
import anyfesolver.matrix_assembly as matrix_module


def _model(*, s3_count: int = 4, include_q4: bool = False) -> FEModel:
    model = FEModel("qualified-s3-fast-assembly")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    next_node = 1
    next_element = 1
    for index in range(s3_count):
        offset = 4.0 * index
        node_ids = []
        for coordinate in (
            (offset, 0.0, 0.0),
            (offset + 1.0, 0.0, 0.0),
            (offset, 1.0, 0.0),
        ):
            model.add_node(next_node, *coordinate)
            node_ids.append(next_node)
            next_node += 1
        model.add_element(
            next_element,
            QualifiedE4PLS3ShellElement(
                next_element,
                node_ids,
                "steel",
                thickness=0.02,
                reference_normal=(0.0, 0.0, 1.0),
            ),
        )
        next_element += 1
    if include_q4:
        node_ids = []
        for coordinate in (
            (-2.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
        ):
            model.add_node(next_node, *coordinate)
            node_ids.append(next_node)
            next_node += 1
        model.add_element(
            next_element,
            QualifiedE4PLShellElement(
                next_element,
                node_ids,
                "steel",
                thickness=0.02,
            ),
        )
    return model


def _warm(model: FEModel) -> np.ndarray:
    first, _ = assemble_stiffness_matrix(model)
    second, _ = assemble_stiffness_matrix(model)
    third, info = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(first.toarray(), second.toarray())
    np.testing.assert_array_equal(second.toarray(), third.toarray())
    assert info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True
    return third.toarray()


def _warm_q4(model: FEModel) -> np.ndarray:
    first, _ = assemble_stiffness_matrix(model)
    second, _ = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(first.toarray(), second.toarray())
    return second.toarray()


def _numpy_coordinate_model(scalar_type: type[Any]) -> FEModel:
    model = FEModel("qualified-s3-numpy-coordinate-assembly")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
        ),
        start=1,
    ):
        model.add_node(
            node_id,
            *(scalar_type(value) for value in coordinate),
        )
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    return model


@pytest.mark.parametrize(
    "scalar_type",
    (np.float32, np.float64, np.int32, np.int64),
)
def test_warm_s3_assembly_accepts_exact_numpy_coordinate_scalars(
    scalar_type: type[Any],
) -> None:
    builtin_expected = _warm(_model(s3_count=1))
    model = _numpy_coordinate_model(scalar_type)
    assert all(
        type(value) is scalar_type
        for node in model.mesh.nodes.values()
        for value in (node.x, node.y, node.z)
    )

    actual = _warm(model)

    np.testing.assert_array_equal(actual, builtin_expected)


def test_warm_s3_numpy_coordinate_authority_rejects_raw_value_or_type_change() -> None:
    model = _numpy_coordinate_model(np.float64)
    expected = _warm(model)
    node = model.mesh.nodes[1]
    original_x = node.x

    object.__setattr__(node, "x", np.float64(0.25))
    with pytest.raises(AssemblyError, match="incompatible qualified shell authority"):
        assemble_stiffness_matrix(model)
    object.__setattr__(node, "x", original_x)
    np.testing.assert_array_equal(_warm(model), expected)

    object.__setattr__(node, "x", float(original_x))
    with pytest.raises(AssemblyError, match="incompatible qualified shell authority"):
        assemble_stiffness_matrix(model)
    object.__setattr__(node, "x", original_x)
    np.testing.assert_array_equal(_warm(model), expected)


@pytest.mark.parametrize("scalar_type", (bool, np.bool_))
def test_qualified_assembly_rejects_boolean_coordinate_scalars(
    scalar_type: type[Any],
) -> None:
    model = _numpy_coordinate_model(scalar_type)

    with pytest.raises(AssemblyError, match="incompatible qualified shell authority"):
        assemble_stiffness_matrix(model)


def test_prepared_s3_execution_accepts_a_fresh_value_equal_component_guard() -> None:
    model = _model(s3_count=1)
    expected = _warm(model)
    element = model.mesh.elements[1]
    material = model.materials["steel"]
    original_guard = element._qualified_component_guard

    element._bind_qualified_component_guard(model.mesh, material)

    assert element._qualified_component_guard is not original_guard
    assert element._qualified_component_guard == original_guard
    actual, info = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(actual.toarray(), expected)
    assert info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True

    lease = matrix_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
        model,
        context="value-equal guard lease preflight",
        allow_q4_cached_stiffness=True,
    )
    element._bind_qualified_component_guard(model.mesh, material)
    with pytest.raises(AssemblyError, match="incompatible qualified shell authority"):
        lease(model, context="value-equal guard lease output", final=True)
    np.testing.assert_array_equal(_warm(model), expected)


def test_warm_q4_assembly_binds_raw_coordinates_and_tracked_generation() -> None:
    model = _model(s3_count=0, include_q4=True)
    expected = _warm_q4(model)
    node = model.mesh.nodes[1]
    original_x = node.x

    object.__setattr__(node, "x", original_x + 0.25)
    try:
        with pytest.raises(
            AssemblyError,
            match="incompatible qualified shell authority",
        ):
            assemble_stiffness_matrix(model)
    finally:
        object.__setattr__(node, "x", original_x)
    np.testing.assert_array_equal(_warm_q4(model), expected)

    changed_model = _model(s3_count=0, include_q4=True)
    object.__setattr__(changed_model.mesh.nodes[1], "x", original_x + 0.25)
    changed, _ = assemble_stiffness_matrix(changed_model)
    assert not np.array_equal(changed.toarray(), expected)

    lease = matrix_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
        model,
        context="Q4 tracked geometry preflight",
        allow_q4_cached_stiffness=True,
    )
    node.x = original_x + 0.125
    node.x = original_x
    with pytest.raises(
        AssemblyError,
        match="incompatible qualified shell authority",
    ):
        lease(model, context="Q4 tracked geometry output", final=True)
    np.testing.assert_array_equal(_warm_q4(model), expected)


def test_shared_node_mixed_compiled_csr_stays_canonical_and_exact() -> None:
    model = FEModel("qualified-shared-node-mixed-csr")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1,
            (1, 2, 3, 4),
            "steel",
            thickness=0.02,
        ),
    )
    model.add_element(
        2,
        QualifiedE4PLS3ShellElement(
            2,
            (1, 2, 3),
            "steel",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )

    baseline, _ = assemble_stiffness_matrix(model)
    expected = baseline.toarray()
    expected_nnz = baseline.nnz
    for _ in range(30):
        current, _ = assemble_stiffness_matrix(model)
        np.testing.assert_array_equal(current.toarray(), expected)
        assert current.nnz == expected_nnz
        assert current.has_sorted_indices is True
        assert current.has_canonical_format is True
        for row in range(current.shape[0]):
            start = int(current.indptr[row])
            stop = int(current.indptr[row + 1])
            assert np.all(np.diff(current.indices[start:stop]) > 0)


def test_warm_s3_assembly_consumes_closure_total_without_batch_reentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    expected = _warm(model)
    reached: list[str] = []

    def tripwire(*_args: Any, **_kwargs: Any) -> Any:
        reached.append("batch")
        raise AssertionError("warm exact S3 assembly re-entered the batch")

    monkeypatch.setattr(
        s3_batch_module,
        "get_reference_s3_stiffness_components",
        tripwire,
    )
    actual, info = assemble_stiffness_matrix(model)
    assert reached == []
    np.testing.assert_array_equal(actual.toarray(), expected)
    assert info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is True
    assert info["diagnostics"]["scalar_shell_element_count"] == 0


def test_s3_assembly_rejects_accessor_metadata_change_and_recovers() -> None:
    model = _model()
    expected = _warm(model)
    helper = s3_module._try_s3_fast_assembly_cached_stiffness
    original = helper.__kwdefaults__
    helper.__kwdefaults__ = {"forged": object()}
    try:
        with pytest.raises(
            AssemblyError,
            match="incompatible qualified shell authority",
        ):
            assemble_stiffness_matrix(model)
    finally:
        helper.__kwdefaults__ = original
    recovered = _warm(model)
    np.testing.assert_array_equal(recovered, expected)


def test_s3_public_total_metadata_cannot_poison_warm_assembly() -> None:
    model = _model()
    expected = _warm(model)
    element = model.mesh.elements[1]
    material = model.materials["steel"]
    disposable = element.compute_stiffness_matrix(model.mesh, material)
    disposable.shape = (9, 36)

    actual, _ = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(actual.toarray(), expected)

    components = element._qualified_components
    assert components is not None
    internal = components["total"]
    original_shape = internal.shape
    internal.shape = (9, 36)
    try:
        with pytest.raises(
            AssemblyError,
            match="incompatible qualified shell authority",
        ):
            assemble_stiffness_matrix(model)
    finally:
        internal.shape = original_shape
    recovered = _warm(model)
    np.testing.assert_array_equal(recovered, expected)


def test_mixed_warm_assembly_rejects_s3_runtime_change_before_dispatch() -> None:
    model = _model(s3_count=3, include_q4=True)
    expected = _warm(model)
    original = s3_module.triangle_frame
    reached: list[str] = []

    def tripwire(*_args: Any, **_kwargs: Any) -> Any:
        reached.append("triangle_frame")
        raise AssertionError("changed S3 helper was dispatched")

    setattr(s3_module, "triangle_frame", tripwire)
    try:
        with pytest.raises(
            AssemblyError,
            match="incompatible qualified shell authority",
        ):
            assemble_stiffness_matrix(model)
    finally:
        setattr(s3_module, "triangle_frame", original)
    assert reached == []
    recovered = _warm(model)
    np.testing.assert_array_equal(recovered, expected)


def test_warm_s3_assembly_rejects_raw_node_and_connectivity_changes() -> None:
    model = _model()
    expected = _warm(model)
    node = model.mesh.nodes[1]
    original_x = node.x
    lease = matrix_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
        model,
        context="raw node lease preflight",
        allow_q4_cached_stiffness=True,
    )
    object.__setattr__(node, "x", original_x + 0.25)
    with pytest.raises(AssemblyError, match="incompatible qualified shell authority"):
        lease(model, context="raw node lease output", final=True)
    object.__setattr__(node, "x", original_x)
    np.testing.assert_array_equal(_warm(model), expected)

    object.__setattr__(node, "x", original_x + 0.25)
    with pytest.raises(AssemblyError, match="incompatible qualified shell authority"):
        assemble_stiffness_matrix(model)
    object.__setattr__(node, "x", original_x)
    np.testing.assert_array_equal(_warm(model), expected)

    element = model.mesh.elements[1]
    original_ids = element.node_ids
    object.__setattr__(element, "node_ids", [1, 2, 4])
    with pytest.raises(AssemblyError, match="incompatible qualified shell authority"):
        assemble_stiffness_matrix(model)
    object.__setattr__(element, "node_ids", original_ids)
    np.testing.assert_array_equal(_warm(model), expected)


def test_warm_s3_assembly_rejects_replaced_dof_manager() -> None:
    model = _model()
    expected = _warm(model)
    original = model.mesh.dof_manager
    object.__setattr__(
        model.mesh,
        "dof_manager",
        SimpleNamespace(total_dofs=original.total_dofs + 6),
    )
    with pytest.raises(AssemblyError, match="incompatible qualified shell authority"):
        assemble_stiffness_matrix(model)
    object.__setattr__(model.mesh, "dof_manager", original)
    actual, _ = assemble_stiffness_matrix(model)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.toarray(), expected)


def test_warm_s3_assembly_ignores_mutable_public_sparsity_arrays() -> None:
    model = _model()
    expected = _warm(model)
    cache = model.mesh.__dict__["_sparsity_cache"]["stiffness"]
    rows = cache["rows"]
    cols = cache["cols"]
    original_row = int(rows[0])
    original_col_shape = cols.shape
    rows[0] = original_row + 6
    cols.shape = (cols.size, 1)
    try:
        actual, _ = assemble_stiffness_matrix(model)
        np.testing.assert_array_equal(actual.toarray(), expected)
    finally:
        rows[0] = original_row
        cols.shape = original_col_shape
    recovered, _ = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(recovered.toarray(), expected)


def test_warm_s3_assembly_never_invokes_reference_plan_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    expected = _warm(model)
    reached: list[str] = []

    def forged_diagnostics(_self: Any) -> dict[str, Any]:
        reached.append("diagnostics")
        raise AssertionError("reference-plan diagnostics callback ran")

    monkeypatch.setattr(
        s3_batch_module.PreparedReferenceS3Components,
        "diagnostics",
        forged_diagnostics,
    )
    with pytest.raises(AssemblyError, match="incompatible qualified shell authority"):
        assemble_stiffness_matrix(model)
    assert reached == []
    monkeypatch.undo()
    np.testing.assert_array_equal(_warm(model), expected)


def test_warm_s3_execution_ignores_caller_replaced_lease_providers() -> None:
    model = _model()
    expected = _warm(model)
    lease = matrix_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
        model,
        context="provider isolation preflight",
        allow_q4_cached_stiffness=True,
    )
    original_items = tuple(model.mesh.elements.items())
    lease._qualified_fast_element_items = lambda: tuple(reversed(original_items))
    lease._qualified_s3_cached_total = lambda _element: np.zeros((18, 18))
    actual, _ = matrix_module._assemble_element_matrix_under_lease(
        model,
        "stiffness",
        lambda element, mesh, material: element.compute_stiffness_matrix(
            mesh,
            material,
        ),
        lease,
    )
    lease(model, context="provider isolation output", final=True)
    np.testing.assert_array_equal(actual.toarray(), expected)
