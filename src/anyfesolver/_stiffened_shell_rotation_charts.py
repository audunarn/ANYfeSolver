"""Owned shell pose charts; no material-history or support-law transport."""

import numpy as np

from scipy.spatial.transform import Rotation
from ._ge_beam3_pose_joint import _exp_terms


POLICY = "STIFFENED_SHELL_OWNED_ROTATION_ANCHORS_V1"


def owned_anchors(value, nodes):
    anchors = np.array(value, dtype=float, copy=True)
    if (anchors.shape != (nodes, 3, 3) or not np.isfinite(anchors).all() or
            any(np.linalg.norm(a.T@a-np.eye(3)) > 1e-10 or
                abs(np.linalg.det(a)-1.) > 1e-10 for a in anchors)):
        raise ValueError("owned proper shell rotation anchors required")
    # Read-only flags on an owning ndarray can be reset by its caller. An
    # immutable backing store makes this ownership boundary enforceable.
    return np.frombuffer(anchors.tobytes(), dtype=float).reshape(anchors.shape)


def physical_frames(q, anchors=None):
    rows = np.asarray(q, dtype=float).reshape(-1, 6)
    if not np.isfinite(rows).all():
        raise ValueError("finite shell chart coordinates required")
    if anchors is not None:
        anchors = owned_anchors(anchors, len(rows))
    return np.array([_exp_terms(row[3:])[0] if anchors is None else
                     _exp_terms(row[3:])[0]@anchors[i]
                     for i, row in enumerate(rows)])


def total_coordinates(q, anchors):
    """Total-chart values and exact local column map for the existing Q4 route.

    The physical configuration is authoritative. This principal logarithm is
    an evaluation chart, not a reconstruction of an accepted loading history.
    Its pi branch is rejected independently of the unchanged relative-strain
    log guard. Large common physical rotations are not relative strains.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or q.size % 6 or not np.isfinite(q).all():
        raise ValueError("complete finite shell chart required")
    anchors = owned_anchors(anchors, q.size//6)
    values = q.copy()
    differential = np.eye(q.size)
    for i, row in enumerate(q.reshape(-1, 6)):
        made, jacobian, _ = _exp_terms(row[3:])
        theta = Rotation.from_matrix(made@anchors[i]).as_rotvec()
        if abs(np.linalg.norm(theta)-np.pi) < 1e-7:
            raise ValueError("shell total evaluation chart is on its pi branch")
        total_jacobian = _exp_terms(theta)[1]
        values[6*i+3:6*i+6] = theta
        differential[6*i+3:6*i+6, 6*i+3:6*i+6] = np.linalg.solve(total_jacobian, jacobian)
    if not np.isfinite(values).all() or not np.isfinite(differential).all():
        raise ValueError("shell total evaluation chart is not representable")
    return values, differential
