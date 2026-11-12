"""Qualified immutable-vector identity contract for the freeze helper.

Covers the demonstrated PR35 defect: recovery reinsertion recreated already
canonical immutable vector inputs, changing their object identity and
invalidating warm qualified assembly plans bound to that identity.  The
freeze helper must preserve identity for the exact supported vector form and
must keep every other value on the original normalizer.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.fe_core import (
    FEModel,
    _freeze_qualified_element_vector_inputs,
    _is_canonical_qualified_vector,
)
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.recovery import (
    RecoveryConfig,
    ResourceConfig,
    recover_element_stresses_with_report,
)
from anyfesolver.s3_reference_batch import MIN_REFERENCE_S3_RECOVERY_GROUP


def _canonical_bytes_vector(values: tuple[float, ...]) -> np.ndarray:
    made = np.ascontiguousarray(np.asarray(values, dtype=float))
    return np.frombuffer(made.tobytes(order="C"), dtype=float).reshape(
        made.shape
    )


def _build_model(count: int, *, with_direction: bool = False) -> FEModel:
    model = FEModel("qualified-vector-identity")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    next_node = 1
    for index in range(int(count)):
        x = float(8 * index)
        node_ids = [next_node, next_node + 1, next_node + 2]
        for node_id, coordinate in zip(
            node_ids,
            ((x, 0.0, 0.0), (x + 3.0, 0.0, 0.0), (x, 3.0, 0.0)),
        ):
            model.add_node(node_id, *coordinate)
        element_id = index + 1
        model.add_element(
            element_id,
            QualifiedE4PLS3ShellElement(
                element_id,
                node_ids,
                "steel",
                thickness=0.05,
                material_direction=(
                    np.asarray((1.0, 0.0, 0.0)) if with_direction else None
                ),
                reference_normal=(0.0, 0.0, 1.0),
            ),
        )
        next_node += 3
    return model


def _terminal_base(value: np.ndarray) -> Any:
    current: Any = value
    while type(current) is np.ndarray:
        current = current.base
    return current


def _readonly_owned() -> np.ndarray:
    made = np.array((1.0, 0.0, 0.0), dtype=float)
    made.setflags(write=False)
    return made


def _readonly_view_over_mutable() -> np.ndarray:
    base = np.zeros(6, dtype=float)
    view = base[:3]
    view.setflags(write=False)
    return view


def _readonly_over_bytearray() -> np.ndarray:
    view = np.frombuffer(bytearray(24), dtype=float)
    view.setflags(write=False)
    return view


def _nonnative_bytes_vector() -> np.ndarray:
    payload = np.asarray((1.0, 0.0, 0.0), dtype=">f8").tobytes(order="C")
    return np.frombuffer(payload, dtype=">f8")


def _metadata_bytes_vector() -> np.ndarray:
    payload = np.asarray((1.0, 0.0, 0.0), dtype=float).tobytes(order="C")
    dtype = np.dtype(np.float64, metadata={"pr35": "vector"})
    return np.frombuffer(payload, dtype=dtype)


def _unaligned_bytes_vector() -> np.ndarray:
    payload = np.asarray((0.0, 1.0, 0.0, 0.0), dtype=float).tobytes(order="C")
    return np.frombuffer(payload, dtype=float, count=3, offset=4)


def _vector_subclass() -> np.ndarray:
    return _canonical_bytes_vector((1.0, 0.0, 0.0)).view(
        type("VectorSubclass", (np.ndarray,), {})
    )


def _strided_bytes_vector() -> np.ndarray:
    payload = np.asarray(
        (1.0, 0.0, 0.0, 2.0, 0.0, 0.0), dtype=float
    ).tobytes(order="C")
    return np.frombuffer(payload, dtype=float)[::2]


def test_recognizer_accepts_proven_immutable_bytes_vectors() -> None:
    canonical = _canonical_bytes_vector((1.0, 0.0, 0.0))
    assert _is_canonical_qualified_vector(canonical) is True
    assert canonical.flags.writeable is False
    assert type(_terminal_base(canonical)) is bytes

    chained = canonical[:]
    assert _is_canonical_qualified_vector(chained) is True
    assert chained.flags.writeable is False
    assert type(_terminal_base(chained)) is bytes

    payload = np.ascontiguousarray(np.asarray((0.0, 0.0, 1.0), dtype=float))
    view = np.frombuffer(memoryview(payload.tobytes(order="C")), dtype=float)
    assert _is_canonical_qualified_vector(view) is True


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.array([1.0, 0.0, 0.0]), id="mutable"),
        pytest.param(_readonly_owned(), id="readonly-owned"),
        pytest.param(_readonly_view_over_mutable(), id="readonly-over-mutable"),
        pytest.param(_readonly_over_bytearray(), id="readonly-over-bytearray"),
        pytest.param(
            np.frombuffer(bytearray(24), dtype=float),
            id="bytearray-memoryview",
        ),
        pytest.param(_nonnative_bytes_vector(), id="nonnative-dtype"),
        pytest.param(_metadata_bytes_vector(), id="metadata-dtype"),
        pytest.param(_vector_subclass(), id="subclass"),
        pytest.param(_strided_bytes_vector(), id="strided"),
        pytest.param(_unaligned_bytes_vector(), id="unaligned-offset-view"),
        pytest.param(
            np.frombuffer(
                np.asarray(2.0, dtype=float).tobytes(order="C"), dtype=float
            ).reshape(()),
            id="scalar-shape",
        ),
        pytest.param(
            np.frombuffer(
                np.asarray((1.0, 0.0, 0.0), dtype=float).tobytes(order="C"),
                dtype=float,
            ).reshape((1, 3)),
            id="unexpected-shape",
        ),
    ],
)
def test_recognizer_rejects_noncanonical_vectors(value: Any) -> None:
    assert _is_canonical_qualified_vector(value) is False


def test_recognizer_rejects_non_arrays() -> None:
    assert _is_canonical_qualified_vector(None) is False
    assert _is_canonical_qualified_vector([1.0, 0.0, 0.0]) is False


def test_mesh_lifecycle_preserves_canonical_identity() -> None:
    model = _build_model(1, with_direction=True)
    element = model.mesh.elements[1]
    normal = element.reference_normal
    direction = element.material_direction
    assert _is_canonical_qualified_vector(normal) is True
    assert _is_canonical_qualified_vector(direction) is True
    revision = element._qualified_plan_state_revision

    model.mesh.elements[1] = element
    assert element.reference_normal is normal
    assert element.material_direction is direction
    assert element._qualified_plan_state_revision == revision

    copied = copy.deepcopy(model)
    copied_element = copied.mesh.elements[1]
    copied_normal = copied_element.reference_normal
    assert copied_normal is not normal
    assert _is_canonical_qualified_vector(copied_normal) is True
    copied.mesh.elements[1] = copied_element
    assert copied_element.reference_normal is copied_normal


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.array([1.0, 0.0, 0.0]), id="mutable"),
        pytest.param(_readonly_owned(), id="readonly-owned"),
        pytest.param(_readonly_view_over_mutable(), id="readonly-over-mutable"),
        pytest.param(_readonly_over_bytearray(), id="readonly-over-bytearray"),
        pytest.param(
            np.frombuffer(bytearray(24), dtype=float),
            id="bytearray-memoryview",
        ),
        pytest.param(
            np.frombuffer(
                np.asarray((1.0, 0.0, 0.0), dtype=">f8").tobytes(order="C"),
                dtype=">f8",
            ),
            id="nonnative-dtype",
        ),
        pytest.param(
            _canonical_bytes_vector((1.0, 0.0, 0.0)).view(
                type("Sub", (np.ndarray,), {})
            ),
            id="subclass",
        ),
        pytest.param(
            np.frombuffer(
                np.asarray((1.0, 0.0, 0.0, 2.0, 0.0, 0.0), dtype=float).tobytes(
                    order="C"
                ),
                dtype=float,
            )[::2],
            id="strided",
        ),
        pytest.param(_unaligned_bytes_vector(), id="unaligned-offset-view"),
    ],
)
def test_freeze_rebuilds_noncanonical_vectors(value: Any) -> None:
    model = _build_model(1)
    element = model.mesh.elements[1]
    object.__setattr__(element, "material_direction", value)
    expected = np.asarray(value, dtype=float).tobytes(order="C")

    model.mesh.elements[1] = element
    frozen = element.material_direction
    assert frozen is not value
    assert type(frozen) is np.ndarray
    assert frozen.dtype == np.dtype(np.float64)
    assert frozen.flags.c_contiguous is True
    assert frozen.flags.writeable is False
    assert type(_terminal_base(frozen)) is bytes
    assert frozen.tobytes(order="C") == expected


def test_freeze_discards_dtype_metadata_on_fallback() -> None:
    model = _build_model(1)
    element = model.mesh.elements[1]
    value = _metadata_bytes_vector()
    object.__setattr__(element, "material_direction", value)

    model.mesh.elements[1] = element
    frozen = element.material_direction
    assert frozen is not value
    assert frozen.dtype == np.dtype(np.float64)
    assert frozen.dtype.metadata is None
    assert frozen.flags.writeable is False
    assert type(_terminal_base(frozen)) is bytes
    assert frozen.tobytes(order="C") == value.tobytes(order="C")


def test_freeze_keeps_unexpected_shapes_on_original_normalizer() -> None:
    model = _build_model(1)
    element = model.mesh.elements[1]

    object.__setattr__(element, "material_direction", np.array(2.0))
    model.mesh.elements[1] = element
    scalar = element.material_direction
    assert scalar.shape == (1,)
    assert scalar.flags.writeable is False
    assert scalar[0] == 2.0

    object.__setattr__(element, "material_direction", np.ones((1, 3)))
    model.mesh.elements[1] = element
    assert element.material_direction.shape == (1, 3)
    assert element.material_direction.flags.writeable is False


def test_freeze_skips_none_and_nonqualified_inputs() -> None:
    model = _build_model(1)
    element = model.mesh.elements[1]
    object.__setattr__(element, "material_direction", None)
    model.mesh.elements[1] = element
    assert element.material_direction is None

    mutable = np.array([1.0, 0.0, 0.0])
    unqualified = type("Unqualified", (), {})()
    unqualified.material_direction = mutable
    unqualified.reference_normal = None
    _freeze_qualified_element_vector_inputs(unqualified)
    assert unqualified.material_direction is mutable


def test_frozen_vector_is_copy_isolated() -> None:
    source = np.array([0.0, 0.0, 2.0])
    model = FEModel("qualified-vector-identity")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in zip(
        (1, 2, 3), ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0))
    ):
        model.add_node(node_id, *coordinate)
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.05,
        reference_normal=source,
    )
    model.add_element(1, element)
    frozen = element.reference_normal
    assert frozen is not source
    assert frozen.flags.writeable is False
    np.testing.assert_array_equal(frozen, (0.0, 0.0, 1.0))

    source[0] = 5.0
    np.testing.assert_array_equal(frozen, (0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        frozen[0] = 1.0


def test_recovery_reinsertion_preserves_warm_stiffness_identity() -> None:
    count = MIN_REFERENCE_S3_RECOVERY_GROUP
    model = _build_model(count)
    vector = 2.0e-5 * np.sin(
        np.arange(model.mesh.dof_manager.total_dofs, dtype=float) + 0.375
    )
    baseline, _baseline_info = assemble_stiffness_matrix(model)
    element = model.mesh.elements[1]
    normal = element.reference_normal
    assert normal.flags.writeable is False

    recovered, report = recover_element_stresses_with_report(
        model,
        vector,
        RecoveryConfig(),
        return_global=True,
        resource_config=ResourceConfig(recovery_threads=1),
    )
    assert len(recovered) == count
    assert report.metadata["qualified_s3_reference_batch_count"] == 1
    assert element.reference_normal is normal

    warm, warm_info = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(warm.toarray(), baseline.toarray())
    assert warm_info["diagnostics"]["qualified_s3_reference_elastic_stiffness"][
        "plan_reused"
    ] is True

    element.reference_normal = (0.0, 0.0, 1.0)
    assert element.reference_normal is not normal
    refreshed, refreshed_info = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(refreshed.toarray(), baseline.toarray())
    assert refreshed_info["diagnostics"][
        "qualified_s3_reference_elastic_stiffness"
    ]["plan_reused"] is False
