"""Fixed-overlap geometry for the explicit guided shell-pair transient slice.

The motion contract makes each physical and pressure gap affine in the full
displacement. The Hermite enclosure bounds the complete represented cubic for
the supplied floating gap/rate values; upstream gap arithmetic is separate.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .plane_newmark_events import hermite_branch_extrema


_MOTION_TOLERANCE = 256.0 * np.finfo(float).eps


def _motion_vector(manager, value, name):
    try:
        if np.iscomplexobj(value):
            raise ValueError("motion vector must be real")
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite full-DOF vector") from exc
    if (vector.shape != (manager.model.mesh.dof_manager.total_dofs,)
            or not np.all(np.isfinite(vector))):
        raise ValueError(f"{name} must be a finite full-DOF vector")
    # Normalize the contact components themselves. A small column or a large
    # unrelated structural component must not hide an inadmissible contact mode.
    scale = max(float(np.max(np.abs(vector[dofs])))
                for dofs in manager.motion_faces)
    if scale == 0.0:
        return vector
    common_tangent = None
    for dofs in manager.motion_faces:
        local = vector[dofs] / scale
        movement = local[:, :3]
        normal_movement = movement @ manager.normal
        tangent = movement - normal_movement[:, None] * manager.normal
        if common_tangent is None:
            common_tangent = tangent[0]
        if (np.max(np.linalg.norm(tangent - common_tangent, axis=1))
                > _MOTION_TOLERANCE
                or np.ptp(normal_movement) > _MOTION_TOLERANCE):
            raise ValueError("guided pair requires uniform normal translation of each face and common tangential motion")
        if np.max(np.linalg.norm(local[:, 3:], axis=1)) > _MOTION_TOLERANCE:
            raise ValueError("guided pair requires fixed shell rotations")
    return vector


def _pressure_support(manager):
    support = np.asarray(manager.support, dtype=float)
    if (support.ndim != 1 or not support.size
            or not np.all(np.isfinite(support)) or np.any(support <= 0.0)):
        raise ValueError("guided pair requires finite positive pressure supports")
    if support.size > manager.settings.maximum_patch_normal_dofs:
        raise MemoryError("guided pair pressure DOF limit exceeded; no positive support was discarded")
    if len(manager.pressure_nodes) != support.size:
        raise ValueError("guided pair pressure feature identity does not match its supports")
    return support


def prepare_guided_shell_pair_transient(manager, T):
    """Check every represented generator and return stable pressure identities.

    Checking one trial displacement cannot establish fixed-overlap eligibility.
    Every nonzero transformation column must obey the motion contract, with a
    dimensionless epsilon tolerance independent of column and geometry scale.
    This is a per-generator floating contract, not an exact certificate for all
    linear combinations: cancellation can expose a tolerated generator residual.
    Actual endpoint and rate vectors are checked again before path bounding.
    Zero contact components are admissible for unrelated structural modes.
    """
    _pressure_support(manager)
    expected = manager.model.mesh.dof_manager.total_dofs
    try:
        if np.iscomplexobj(T):
            raise ValueError("constraint subspace must be real")
        matrix = T.tocsc(copy=True) if sparse.issparse(T) else np.asarray(T, dtype=float)
        if sparse.issparse(matrix):
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
        values = matrix.data if sparse.issparse(matrix) else matrix
        if (matrix.ndim != 2 or matrix.shape[0] != expected
                or matrix.shape[1] == 0 or not np.all(np.isfinite(values))
                or not np.any(values != 0.0)):
            raise ValueError("guided pair requires a represented nonempty finite constraint subspace")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("guided pair requires a represented nonempty finite constraint subspace") from exc
    for column in range(matrix.shape[1]):
        vector = (matrix[:, column].toarray().reshape(-1) if sparse.issparse(matrix)
                  else matrix[:, column])
        _motion_vector(manager, vector, "constraint mode")
    name = manager.definition.interactions[0].name
    return tuple((name, int(node)) for node in manager.pressure_nodes)


def guided_shell_pair_path_minimum_gap(manager, start, end, v0, v1, dt,
                                       check_cancel=lambda: None):
    """Bound every pressure and physical gap over the entire Hermite step path.

    Physical samples include the overlap vertices and quadrature of the manager.
    Under admitted translation the reference overlap remains fixed and these
    gaps are affine. Endpoint-only gap checks cannot exclude an interior crossing.
    """
    check_cancel()
    start, end, v0, v1 = (_motion_vector(manager, value, name) for value, name in (
        (start, "start displacement"), (end, "end displacement"),
        (v0, "start velocity"), (v1, "end velocity")))
    support = _pressure_support(manager)
    if not manager.samples:
        raise ValueError("guided pair has no physical gap samples")
    pressure = manager.gap_gradient
    reference = np.asarray(manager.reference_weighted_gap, dtype=float)
    gaps0 = list((reference + pressure @ start) / support)
    gaps1 = list((reference + pressure @ end) / support)
    rates0 = list((pressure @ v0) / support)
    rates1 = list((pressure @ v1) / support)
    for reference_gap, source_shape, target_shape, source_dofs, target_dofs, source_gradient, target_gradient in manager.samples:
        check_cancel()
        def relative_motion(vector):
            return float(source_shape @ (vector[source_dofs] @ source_gradient)
                         - target_shape @ (vector[target_dofs] @ target_gradient))
        gaps0.append(reference_gap + relative_motion(start))
        gaps1.append(reference_gap + relative_motion(end))
        rates0.append(relative_motion(v0))
        rates1.append(relative_motion(v1))
    minimum = float("inf")
    # Keep cancellation responsive even for many physical overlap samples.
    for first in range(0, len(gaps0), 256):
        check_cancel()
        selected = slice(first, first + 256)
        bounds = hermite_branch_extrema(gaps0[selected], gaps1[selected],
                                        rates0[selected], rates1[selected], dt)
        minimum = min(minimum, float(np.min(bounds.minimum_gaps)))
    check_cancel()
    return minimum
