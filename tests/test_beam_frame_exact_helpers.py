"""The beam frame fast helpers are bit-identical to the NumPy calls they replace."""

from __future__ import annotations

import numpy as np
import pytest

import anyfesolver.elements as elements
from anyfesolver.elements import BeamElement


def _bits(value: object) -> bytes:
    array = np.asarray(value)
    return array.dtype.str.encode() + repr(array.shape).encode() + array.tobytes()


def _reference_rotation(e1: np.ndarray, orientation: np.ndarray | None) -> np.ndarray:
    """The pre-change ``_beam_rotation_matrix`` body, verbatim."""
    if orientation is not None:
        candidate = orientation - np.dot(orientation, e1) * e1
        norm = float(np.linalg.norm(candidate))
        if norm > 1.0e-6 * float(np.linalg.norm(orientation)):
            e3 = candidate / norm
            e2 = elements._cross3(e3, e1)
            e2 /= np.linalg.norm(e2)
            return np.column_stack((e1, e2, e3))
    trial = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(e1, trial))) > 0.95:
        trial = np.array([0.0, 0.0, 1.0])
    e2 = trial - np.dot(trial, e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = elements._cross3(e1, e2)
    e3 /= np.linalg.norm(e3)
    return np.column_stack((e1, e2, e3))


def _reference_arm(offsets: np.ndarray) -> np.ndarray:
    arm = np.eye(12)
    for i, d in enumerate(offsets):
        x, y, z = d
        arm[i * 6 : i * 6 + 3, i * 6 + 3 : i * 6 + 6] = np.array(
            [[0, z, -y], [-z, 0, x], [y, -x, 0]]
        )
    return arm


def _vectors(count: int) -> np.ndarray:
    rng = np.random.default_rng(20261124)
    scale = 10.0 ** rng.integers(-9, 9, size=(count, 1))
    values = rng.normal(size=(count, 3)) * scale
    axes = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, -0.0, 0.0],
            [0.0, 1.0, -0.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [3.0, 4.0, 12.0],
            [1e-300, 1e-300, 0.0],
            [1e300, 1e300, 1.0],
            [0.95, 0.31224989991991997, 0.0],
        ]
    )
    return np.vstack([axes, values])


def test_vector_norm_matches_linalg_norm_bitwise() -> None:
    for vector in _vectors(20000):
        vector = np.ascontiguousarray(vector)
        expected = np.linalg.norm(vector)
        got = elements._contiguous_vector_norm(vector)
        assert type(got) is type(expected)
        assert _bits(got) == _bits(expected)


@pytest.mark.parametrize(
    "vector",
    [
        np.arange(6.0)[::2],  # non-contiguous view
        np.ones((3, 1)),  # two-dimensional
        np.array([1, 2, 2]),  # integer
        np.array([1.0, 2.0, 2.0], dtype=np.float32),
        [3.0, 4.0, 0.0],  # not an ndarray
    ],
)
def test_vector_norm_other_inputs_take_linalg_norm(vector: object) -> None:
    expected = np.linalg.norm(vector)
    got = elements._contiguous_vector_norm(vector)
    assert type(got) is type(expected)
    assert _bits(got) == _bits(expected)


def test_rotation_matrix_matches_reference_bitwise() -> None:
    vectors = _vectors(6000)
    rng = np.random.default_rng(7)
    for index, raw in enumerate(vectors):
        length = np.linalg.norm(raw)
        if not np.isfinite(length) or length == 0.0:
            continue
        e1 = raw / length
        for orientation in (
            None,
            np.array([0.0, 0.0, 1.0]),
            np.array(raw, dtype=float),  # parallel: heuristic fallback
            rng.normal(size=3),
        ):
            expected = _reference_rotation(e1, orientation)
            got = elements._beam_rotation_matrix(e1, orientation)
            assert _bits(got) == _bits(expected), (index, orientation)
            assert got.flags.c_contiguous == expected.flags.c_contiguous


def test_default_rigid_arm_matches_zero_offset_loop_bitwise() -> None:
    element = BeamElement(1, [1, 2], "steel", {})
    assert not hasattr(element, "_linear_nodal_offsets")
    got = element._linear_rigid_arm_map()
    expected = _reference_arm(np.zeros((2, 3)))
    assert _bits(got) == _bits(expected)
    assert np.signbit(got).sum() == 6
    # Callers may own the returned array; it must be a fresh one each time.
    assert got is not element._linear_rigid_arm_map()


def test_explicit_offsets_keep_the_loop() -> None:
    element = BeamElement(1, [1, 2], "steel", {})
    offsets = np.array([[0.0, -0.0, 0.1], [0.2, 0.05, -0.3]])
    element.set_linear_nodal_offsets(offsets)
    assert _bits(element._linear_rigid_arm_map()) == _bits(_reference_arm(offsets))
