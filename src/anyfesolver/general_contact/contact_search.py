"""Opt-in contact broad phase backed by ANYgeometry's public spatial index.

Callers supply conservative physical boxes and stable feature identities.
This module owns neither contact mechanics nor the bounds calculation.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np


ContactFeatureKey = tuple[str, int]


def _validated_boxes(boxes: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(boxes, dtype=float)
    if (values.shape != (count, 6) or not np.all(np.isfinite(values))
            or np.any(values[:, :3] > values[:, 3:])):
        raise ValueError("contact search needs finite ordered 3D boxes")
    return values.copy()


def _validated_keys(keys: Sequence[ContactFeatureKey]) -> tuple[ContactFeatureKey, ...]:
    result = []
    for key in keys:
        if (not isinstance(key, tuple) or len(key) != 2
                or not isinstance(key[0], str) or not key[0]
                or isinstance(key[1], bool) or not isinstance(key[1], Integral)
                or key[1] < 0):
            raise ValueError("contact feature key needs a kind and nonnegative integer ID")
        result.append((key[0], int(key[1])))
    if len(set(result)) != len(result):
        raise ValueError("contact search feature keys must be unique")
    return tuple(result)


class ContactAABBIndex:
    """Stable-key exact-box search; refit preserves the represented features."""

    def __init__(self, keys: Sequence[ContactFeatureKey], boxes: np.ndarray):
        self.keys = _validated_keys(keys)
        self.boxes = _validated_boxes(boxes, len(self.keys))
        try:
            from anygeometry import AABB, AABBTree
        except ModuleNotFoundError as exc:
            if exc.name != "anygeometry":
                raise
            raise RuntimeError(
                "contact search requires ANYgeometry; install "
                "ANYfeSolver[general-contact]"
            ) from exc
        self._aabb = AABB
        self._tree_type = AABBTree
        self._tree = AABBTree(
            (key, self._box(value)) for key, value in zip(
                self.keys, self.boxes, strict=True)
        )

    def _box(self, row: np.ndarray):
        return self._aabb(tuple(map(float, row[:3])),
                          tuple(map(float, row[3:])))

    def query(self, box: np.ndarray, *, padding: float = 0.0
              ) -> tuple[ContactFeatureKey, ...]:
        return self.query_with_diagnostics(box, padding=padding).keys

    def query_with_diagnostics(self, box: np.ndarray, *, padding: float = 0.0):
        """Return owner query keys and work counters together."""
        row = _validated_boxes(np.asarray(box).reshape(1, -1), 1)[0]
        margin = float(padding)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("contact search padding must be finite and nonnegative")
        return self._tree.query(self._box(row).expanded(margin))

    @property
    def diagnostics(self):
        """Return owner tree build/update counters."""
        return self._tree.diagnostics

    def refit(self, boxes: np.ndarray) -> None:
        """Replace exact current boxes without changing feature ownership."""
        values = _validated_boxes(boxes, len(self.keys))
        try:
            for key, row in zip(self.keys, values, strict=True):
                self._tree.update(key, self._box(row))
        except Exception:
            self._tree = self._tree_type(
                (key, self._box(row)) for key, row in zip(
                    self.keys, self.boxes, strict=True)
            )
            raise
        self.boxes = values
