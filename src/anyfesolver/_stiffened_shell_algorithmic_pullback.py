"""Spatial rows for a complete, fixed-predecessor shell algorithmic pullback."""

import numpy as np

from ._ge_beam3_pose_joint import _exp_terms


def spatial_response(displacement, force, tangent, differential, second, *, stable_rotation=False):
    """Convert D.T f and its complete derivative through the actual chart.

    A consistent material derivative is not an assertion of a plastic energy
    potential. No symmetry repair is applied to the algorithmic operator.
    """
    u = np.asarray(displacement, dtype=float)
    if type(stable_rotation) is not bool:
        raise ValueError('typed shell rotation numerical selection required')
    exp_terms = _exp_terms
    if stable_rotation:
        from ._ge_beam3_g3c_stable.joint import _exp_terms as exp_terms
    covector = differential.T@force
    derivative = differential.T@tangent@differential+np.einsum(
        "i,ijk->jk", force, second)
    chart = np.eye(len(u))
    chart_derivatives = []
    for node, row in enumerate(u.reshape(-1, 6)):
        _, jacobian, curvature = exp_terms(row[3:])
        singular = np.linalg.svd(jacobian, compute_uv=False)
        if singular[-1] <= 1e-11*max(1., singular[0]):
            raise ValueError("singular additive shell chart; owner rebase required")
        slots = slice(6*node+3, 6*node+6)
        chart[slots, slots] = jacobian
        chart_derivatives.append((slots, curvature))
    residual = np.linalg.solve(chart.T, covector)
    correction = np.zeros((len(u), len(u)))
    for slots, curvature in chart_derivatives:
        correction[slots, slots] = np.einsum("ijk,i->jk", curvature, residual[slots])
    spatial_tangent = np.linalg.solve(chart.T, derivative-correction)
    return residual, spatial_tangent
