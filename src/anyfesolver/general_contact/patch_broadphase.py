"""Deterministic projected-footprint search for restricted planar patches.

The tree only rejects pairs whose tangential AABBs are disjoint. The physical
polygon overlap and contact law remain the narrow-phase authorities.
"""

from __future__ import annotations

import numpy as np


class PatchAABBTree:
    """Contiguous 2D AABB hierarchy with stable feature-index ordering."""

    def __init__(self, boxes: np.ndarray, *, leaf_size: int = 8):
        values = np.asarray(boxes, dtype=float)
        if (values.ndim != 2 or values.shape[1] != 4 or not len(values)
                or not np.all(np.isfinite(values))
                or np.any(values[:, :2] > values[:, 2:])
                or type(leaf_size) is not int or leaf_size < 1):
            raise ValueError("patch AABB tree needs finite ordered boxes and leaf size")
        self.boxes = values.copy()
        bounds: list[np.ndarray] = []
        left: list[int] = []
        right: list[int] = []
        first: list[int] = []
        stop: list[int] = []
        order: list[int] = []

        def build(indices: list[int]) -> int:
            node = len(bounds)
            local = self.boxes[indices]
            lower = np.min(local[:, :2], axis=0)
            upper = np.max(local[:, 2:], axis=0)
            bounds.append(np.r_[lower, upper])
            left.append(-1)
            right.append(-1)
            first.append(-1)
            stop.append(-1)
            if len(indices) <= leaf_size:
                first[node] = len(order)
                order.extend(sorted(indices))
                stop[node] = len(order)
                return node
            centroids = 0.5 * (local[:, :2] + local[:, 2:])
            axis = int(np.argmax(np.ptp(centroids, axis=0)))
            ordered = sorted(indices, key=lambda index: (
                0.5 * (self.boxes[index, axis] + self.boxes[index, axis + 2]),
                index,
            ))
            middle = len(ordered) // 2
            left[node] = build(ordered[:middle])
            right[node] = build(ordered[middle:])
            return node

        build(list(range(len(values))))
        self.bounds = np.asarray(bounds, dtype=float)
        self.left = np.asarray(left, dtype=np.intp)
        self.right = np.asarray(right, dtype=np.intp)
        self.first = np.asarray(first, dtype=np.intp)
        self.stop = np.asarray(stop, dtype=np.intp)
        self.order = np.asarray(order, dtype=np.intp)

    def query(self, box: np.ndarray, *, padding: float = 0.0) -> tuple[int, ...]:
        window = np.asarray(box, dtype=float)
        if (window.shape != (4,) or not np.all(np.isfinite(window))
                or np.any(window[:2] > window[2:])
                or not np.isfinite(padding) or padding < 0.0):
            raise ValueError("patch AABB query needs an ordered finite box")
        query = window + np.array((-padding, -padding, padding, padding))
        stack = [0]
        found: list[int] = []
        while stack:
            node = stack.pop()
            bound = self.bounds[node]
            if (np.any(bound[:2] > query[2:])
                    or np.any(query[:2] > bound[2:])):
                continue
            if self.first[node] >= 0:
                for index in self.order[self.first[node]:self.stop[node]]:
                    box = self.boxes[index]
                    if (np.all(box[:2] <= query[2:])
                            and np.all(query[:2] <= box[2:])):
                        found.append(int(index))
            else:
                stack.append(int(self.right[node]))
                stack.append(int(self.left[node]))
        return tuple(sorted(found))


def projected_face_boxes(faces, normal: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return conservative 2D boxes in one deterministic normal frame."""

    direction = np.asarray(normal, dtype=float).copy()
    if (direction.shape != (3,) or not np.all(np.isfinite(direction))
            or float(np.linalg.norm(direction)) == 0.0):
        raise ValueError("patch broadphase needs a finite normal")
    direction /= np.linalg.norm(direction)
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(direction)))] = 1.0
    tangent = seed - float(seed @ direction) * direction
    tangent /= np.linalg.norm(tangent)
    binormal = np.cross(direction, tangent)
    boxes = []
    for face in faces:
        corners = face.geometry.corners
        projected = np.column_stack((corners @ tangent, corners @ binormal))
        boxes.append(np.r_[np.min(projected, axis=0),
                           np.max(projected, axis=0)])
    return np.asarray(boxes, dtype=float), tangent, binormal
