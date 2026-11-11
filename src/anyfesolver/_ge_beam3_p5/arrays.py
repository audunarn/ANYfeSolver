"""Internal P5 package candidate; unqualified and not publicly registered.

Extracted mechanically from the bound source map. Do not edit copied mechanics
without a reviewed successor mapping and renewed equivalence checks.
"""

import numpy as np

def _readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _array(value, shape, label):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _frames(value, count, label):
    frames = _array(value, (count, 3, 3), label)
    for frame in frames:
        if np.linalg.norm(frame.T @ frame-np.eye(3)) > 1e-11 or abs(
            np.linalg.det(frame)-1.0
        ) > 1e-11:
            raise ValueError(f"{label} must contain proper rotations")
    return frames
