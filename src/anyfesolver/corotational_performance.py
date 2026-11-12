"""Compiled block rotations for the qualified corotational tangent.

The rotated corotational path applies the same 3x3 rigid rotation to every
translational and rotational vector block.  Building a dense block-diagonal
matrix and evaluating two dense matrix products performs mostly operations on
known zeros.  These kernels apply the identical algebra directly to the 3x3
blocks while the consistent tangent retains its existing dense chain-rule
implementation.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .jit_compiler import JIT_BACKEND, JIT_DISABLED_REASON, JIT_ENABLED, njit
from .nonlinear_analysis_diagnostics import record_corotational_analysis_execution


@njit(cache=True)
def _rotate_force_blocks_jit(force: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.empty_like(force)
    for block in range(force.shape[0] // 3):
        start = 3 * block
        for row in range(3):
            value = 0.0
            for column in range(3):
                value += rotation[row, column] * force[start + column]
            result[start + row] = value
    return result


@njit(cache=True)
def _rotate_force_tangent_blocks_jit(
    force: np.ndarray,
    tangent: np.ndarray,
    rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    force_result = _rotate_force_blocks_jit(force, rotation)
    tangent_result = np.empty_like(tangent)
    block_count = force.shape[0] // 3
    for row_block in range(block_count):
        row_start = 3 * row_block
        for column_block in range(block_count):
            column_start = 3 * column_block
            for row in range(3):
                for column in range(3):
                    value = 0.0
                    for inner_row in range(3):
                        for inner_column in range(3):
                            value += (
                                rotation[row, inner_row]
                                * tangent[row_start + inner_row, column_start + inner_column]
                                * rotation[column, inner_column]
                            )
                    tangent_result[row_start + row, column_start + column] = value
    return force_result, tangent_result


_STATUS_LOCK = threading.Lock()
_STATUS = {
    "force_block_rotations": 0,
    "tangent_block_rotations": 0,
    "dense_consistent_rotations": 0,
}


def _validated_inputs(
    force: np.ndarray,
    rotation: np.ndarray,
    tangent: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    force_array = np.asarray(force, dtype=float).reshape(-1)
    rotation_array = np.asarray(rotation, dtype=float)
    if rotation_array.shape != (3, 3):
        raise ValueError("corotational rotation must have shape (3, 3)")
    if force_array.size == 0 or force_array.size % 3:
        raise ValueError("corotational force size must be a positive multiple of 3")
    tangent_array: Optional[np.ndarray]
    if tangent is None:
        tangent_array = None
    else:
        tangent_array = np.asarray(tangent, dtype=float)
        expected = (force_array.size, force_array.size)
        if tangent_array.shape != expected:
            raise ValueError(
                f"corotational tangent shape {tangent_array.shape} does not match {expected}"
            )
    return force_array, rotation_array, tangent_array


def rotate_corotational_force_blocks(
    force: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Return ``E @ force`` without constructing block-diagonal ``E``."""

    force_array, rotation_array, _ = _validated_inputs(force, rotation)
    result = _rotate_force_blocks_jit(force_array, rotation_array)
    with _STATUS_LOCK:
        _STATUS["force_block_rotations"] += 1
    record_corotational_analysis_execution(force_blocks=1)
    return result


def rotate_corotational_force_tangent_blocks(
    force: np.ndarray,
    tangent: np.ndarray,
    rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(E f, E K E.T)`` through direct 3x3 block operations."""

    force_array, rotation_array, tangent_array = _validated_inputs(
        force,
        rotation,
        tangent,
    )
    assert tangent_array is not None
    result = _rotate_force_tangent_blocks_jit(
        force_array,
        tangent_array,
        rotation_array,
    )
    with _STATUS_LOCK:
        _STATUS["force_block_rotations"] += 1
        _STATUS["tangent_block_rotations"] += 1
    record_corotational_analysis_execution(force_blocks=1, tangent_blocks=1)
    return result


def note_dense_consistent_rotation() -> None:
    """Record the intentional dense fallback used by the consistent tangent."""

    with _STATUS_LOCK:
        _STATUS["dense_consistent_rotations"] += 1
    record_corotational_analysis_execution(dense_consistent=1)


def reset_corotational_performance_status() -> None:
    with _STATUS_LOCK:
        for key in _STATUS:
            _STATUS[key] = 0


def corotational_performance_status() -> Dict[str, Any]:
    with _STATUS_LOCK:
        counters = dict(_STATUS)
    return {
        "fast_path_name": "corotational_direct_3x3_blocks",
        "eligible": True,
        "backend": JIT_BACKEND,
        "jit_enabled": bool(JIT_ENABLED),
        "fallback_reason": JIT_DISABLED_REASON,
        **counters,
    }
