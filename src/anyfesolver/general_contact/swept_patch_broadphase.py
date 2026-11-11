"""Conservative search for affine physical-corner Q4/S3 projection sweeps.

This selects pairs only; the overlap-local kernel remains the gap authority.
An unresolved normal cone returns ``None`` (evaluate every pair). No endpoint
candidate union or time sample participates in the selection.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from .contact_search import ContactAABBIndex
from .overlap import PlanarFace
from .q4_overlap_local_bound import (
    _dot, _enclose_positions, _interval, _normal_controls_interval,
    _norm_upper, _outward,
)
from .q4_source_overlap import projected_q4_source_face


def _indices(values, count):
    result = []
    for face in values:
        local = tuple(face)
        if (not local or any(isinstance(i, (bool, np.bool_))
                             or not isinstance(i, Integral)
                             or i < 0 or i >= count for i in local)
                or len(set(local)) != len(local)):
            raise ValueError("swept search needs unique in-range face node indices")
        result.append(np.asarray(local, dtype=np.intp))
    if not result:
        raise ValueError("swept search needs source and target faces")
    return tuple(result)


def conservative_swept_pair_indices(
        a0, a1, b0, b1, source_indices, target_indices, *, endpoint_errors=None
        ) -> set[int] | None:
    """Return row-major candidates or ``None`` for exhaustive fallback.

    Endpoints are finite ``(nodes, 3)`` physical positions. Each error is a
    finite nonnegative scalar or array broadcastable to its endpoint, ordered
    ``(a0, a1, b0, b1)``. The affine path includes the interpolated endpoint
    enclosures. Malformed inputs/degenerate endpoint faces raise ``ValueError``;
    unsupported topology or unresolved finite cone bounds request fallback.

    All-pair parent and physical-corner angular conditions match those in
    ``overlap_local_swept_lower`` before any pair can be excluded.
    """
    poses = tuple(np.asarray(x, dtype=float) for x in (a0, a1, b0, b1))
    if (any(x.ndim != 2 or x.shape[1] != 3 or not len(x)
            or not np.all(np.isfinite(x)) for x in poses)
            or poses[0].shape != poses[1].shape
            or poses[2].shape != poses[3].shape):
        raise ValueError("swept search needs matching finite physical endpoints")
    errors = (0.0,)*4 if endpoint_errors is None else tuple(endpoint_errors)
    if len(errors) != 4:
        raise ValueError("swept search needs four endpoint enclosures")
    try:
        errors = tuple(np.broadcast_to(np.asarray(e, dtype=float), x.shape)
                       for e, x in zip(errors, poses, strict=True))
    except ValueError as exc:
        raise ValueError("swept endpoint enclosure shape does not match positions") from exc
    if any(not np.all(np.isfinite(e)) or np.any(e < 0.0) for e in errors):
        raise ValueError("swept endpoint enclosures must be finite and nonnegative")
    sources = _indices(source_indices, len(poses[0]))
    targets = _indices(target_indices, len(poses[2]))
    if any(len(i) != 4 for i in sources) or any(len(i) != 3 for i in targets):
        return None
    # Validate remote and singleton faces as well as represented candidates.
    references = []
    for nodes in sources:
        _, normal = projected_q4_source_face(poses[0][nodes])
        projected_q4_source_face(poses[1][nodes])
        references.append(normal)
    axes = []
    for nodes in targets:
        for pose in poses[2:]:
            PlanarFace((1, 1), pose[nodes])
        face = poses[2][nodes]
        axis = np.cross(face[1]-face[0], face[2]-face[0])
        length = np.linalg.norm(axis)
        if not np.isfinite(length) or length <= 0.0:
            return None
        axes.append(axis/length)
    axes = np.asarray(axes)
    cones = []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        bounds = tuple(_enclose_positions(x, e)
                       for x, e in zip(poses, errors, strict=True))
        if any(not np.all(np.isfinite(v)) for pair in bounds for v in pair):
            return None
        for nodes, reference in zip(sources, references, strict=True):
            controls = tuple(_normal_controls_interval(
                poses[0][nodes], poses[1][nodes], corner,
                errors[0][nodes], errors[1][nodes]) for corner in range(4))
            if any(not np.all(np.isfinite(v)) for pair in controls for v in pair):
                return None
            parent = controls[0]
            height = np.min(_dot(parent, _interval(reference))[0])
            if not np.isfinite(height) or height <= 0.0:
                return None
            aligned = np.where((axes @ reference < 0.0)[:, None], -axes, axes)
            axis_norm = _norm_upper(_interval(aligned))
            for values in controls:
                maximum = np.max(_norm_upper(values))
                denominator = _outward(maximum*axis_norm, np.inf)
                expanded = (values[0][:, None, :], values[1][:, None, :])
                numerator = np.min(_dot(expanded, _interval(aligned))[0], axis=0)
                cosine = _outward(numerator/denominator, -np.inf)
                if (not np.all(np.isfinite(cosine))
                        or np.any(denominator <= 0.0) or np.any(cosine <= 0.75)):
                    return None
            low, high = np.min(parent[0], axis=0), np.max(parent[1], axis=0)
            # Bernstein interval controls enclose n0(t) for the entire affine
            # corner sweep. A same-sign component gives |n_k(t)| >= d > 0.
            distance = np.where(low > 0.0, low, np.where(high < 0.0, -high, 0.0))
            usable = np.flatnonzero(distance > 0.0)
            if not len(usable):
                return None
            axis = int(usable[np.argmax(distance[usable])])
            ratio = _outward(np.maximum(np.abs(low), np.abs(high))/distance[axis], np.inf)
            if not np.all(np.isfinite(ratio)):
                return None
            cones.append((axis, ratio))
        source_low = np.minimum(bounds[0][0], bounds[1][0])
        source_high = np.maximum(bounds[0][1], bounds[1][1])
        target_low = np.minimum(bounds[2][0], bounds[3][0])
        target_high = np.maximum(bounds[2][1], bounds[3][1])
        scene_low, scene_high = np.min(target_low, axis=0), np.max(target_high, axis=0)
        scale = max(1.0, *(float(np.max(np.abs(v))) for pair in bounds for v in pair))
        margin = _outward(256.0*np.finfo(float).eps*scale, np.inf)
        boxes = np.asarray([np.r_[
            _outward(np.min(target_low[nodes], axis=0)-margin, -np.inf),
            _outward(np.max(target_high[nodes], axis=0)+margin, np.inf)
        ] for nodes in targets])
        windows = []
        for nodes, (axis, ratio) in zip(sources, cones, strict=True):
            low = np.min(source_low[nodes], axis=0)
            high = np.max(source_high[nodes], axis=0)
            # Physical Q1/P1 points remain in the swept corner boxes. For a
            # matched projection b-a=lambda*n0, |b_k-a_k| <= H_k, hence
            # |b_j-a_j| <= H_k*max|n_j|/d. The k window covers the target
            # scene itself; the remaining axes contain every matched point.
            separation = max(_outward(high[axis]-scene_low[axis], np.inf),
                             _outward(scene_high[axis]-low[axis], np.inf))
            expansion = _outward(separation*ratio, np.inf)
            window_low = _outward(_outward(low-expansion, -np.inf)-margin, -np.inf)
            window_high = _outward(_outward(high+expansion, np.inf)+margin, np.inf)
            window_low[axis] = _outward(scene_low[axis]-margin, -np.inf)
            window_high[axis] = _outward(scene_high[axis]+margin, np.inf)
            windows.append(np.r_[window_low, window_high])
        if (not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(windows))):
            return None
    search = ContactAABBIndex(tuple(("target_face", i) for i in range(len(targets))), boxes)
    return {source_index*len(targets)+key[1]
            for source_index, window in enumerate(windows)
            for key in search.query(window)}
