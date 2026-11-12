"""Conservative whole-face bounds for affine S3 / original Q4 skin paths.

Every projected Q4 corner Jacobian is checked against the varying S3 normal.
This is a sufficient interval certificate, not a sampled collision test.
"""
import numpy as np
from .q4_overlap_local_bound import (
    _normal_controls_interval, _dot, _interval, _enclose_positions,
    _sub, _norm_upper,
)


def s3_q4_affine_lower(a0, a1, b0, b1, *, endpoint_errors=None):
    arrays = tuple(np.asarray(x, dtype=float) for x in (a0, a1, b0, b1))
    if tuple(x.shape for x in arrays) != ((3, 3), (3, 3), (4, 3), (4, 3)):
        raise ValueError("S3/Q4 affine certificate requires original three/four corners")
    if not all(np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("S3/Q4 affine certificate requires finite coordinates")
    errors = (0.,)*4 if endpoint_errors is None else endpoint_errors
    if len(errors) != 4:
        raise ValueError("four endpoint enclosures required")
    normals = _normal_controls_interval(arrays[0], arrays[1],
        start_error=errors[0], end_error=errors[1])
    reference = np.cross(arrays[0][1]-arrays[0][0], arrays[0][2]-arrays[0][0])
    magnitude = np.linalg.norm(reference)
    if not np.isfinite(magnitude) or magnitude <= 0.:
        return -np.inf
    reference /= magnitude
    # The rounded direction need not have exactly unit norm. Cauchy's
    # inequality uses its outward upper norm, including that rounding.
    reference_norm_upper = float(_norm_upper(_interval(reference)))
    denominator_lower = float(np.nextafter(
        np.min(_dot(normals, _interval(reference))[0])/reference_norm_upper, -np.inf))
    if denominator_lower <= 0.:
        return -np.inf
    # Dot products of all quadratic Bernstein controls bound the product
    # through the complete affine path, including rotating source normals.
    for corner in range(4):
        target = _normal_controls_interval(arrays[2], arrays[3], corner,
            start_error=errors[2], end_error=errors[3])
        pairs = _dot((normals[0][:, None], normals[1][:, None]),
                     (target[0][None, :], target[1][None, :]))[0]
        if not np.all(np.isfinite(pairs)) or np.min(pairs) <= 0.:
            return -np.inf
    denominator_upper = float(np.max(_norm_upper(normals)))
    endpoint_boxes = [_enclose_positions(x, e) for x, e in zip(arrays, errors)]
    source = (np.minimum(endpoint_boxes[0][0], endpoint_boxes[1][0]),
              np.maximum(endpoint_boxes[0][1], endpoint_boxes[1][1]))
    target = (np.minimum(endpoint_boxes[2][0], endpoint_boxes[3][0]),
              np.maximum(endpoint_boxes[2][1], endpoint_boxes[3][1]))
    separation = _sub((source[0][:, None], source[1][:, None]),
                      (target[0][None, :], target[1][None, :]))
    normal_box = (np.min(normals[0], axis=0), np.max(normals[1], axis=0))
    numerator = float(np.min(_dot(separation, normal_box)[0]))
    denominator = denominator_lower if numerator < 0. else denominator_upper
    return float(np.nextafter(numerator/denominator, -np.inf))
