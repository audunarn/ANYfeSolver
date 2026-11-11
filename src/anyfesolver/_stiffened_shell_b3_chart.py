"""Private sparse native-B3 additive-chart pullback for G0 F06.

The input trial uses spatial rotation increments.  The output is the
coordinate residual/Jacobian for left-Exp charts at the supplied nodal and
retained-cell rotation coordinates.  This is not a mixed production owner.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ._ge_beam3_p5.chart import exp_chart_terms
from ._stiffened_shell_b3_local_adapter import NativeB3SparseTrial


def pullback_sparse_trial(layout, trial: NativeB3SparseTrial,
                          coordinates) -> NativeB3SparseTrial:
    """Apply the native left-Exp coordinate map without dense global arrays."""
    if not isinstance(trial, NativeB3SparseTrial):
        raise ValueError("native B3 sparse spatial trial required")
    count = layout.count
    q = np.asarray(coordinates, dtype=float)
    if (q.shape != (count,) or not np.isfinite(q).all() or
            trial.residual.shape != (count,) or
            trial.tangent.shape != (count, count) or
            not np.isfinite(trial.residual).all() or
            not np.isfinite(trial.tangent.data).all()):
        raise ValueError("finite native B3 chart and spatial trial required")
    if layout.nodal_count != 6 * len(layout.node_ids):
        raise ValueError("native B3 nodal rotation layout required")
    if count != layout.nodal_count + 24 * len(layout.elements):
        raise ValueError("native B3 retained-cell layout required")

    starts = [6 * node + 3 for node in range(len(layout.node_ids))]
    for element in range(len(layout.elements)):
        first = layout.nodal_count + 24 * element
        starts.extend((first, first + 3))
    rotation_dofs = np.array([start + i for start in starts for i in range(3)])
    diagonal = np.ones(count, dtype=bool)
    diagonal[rotation_dofs] = False
    other = np.flatnonzero(diagonal)
    rows = list(other)
    columns = list(other)
    values = [1.] * len(other)
    curvature_rows, curvature_columns, curvature_values = [], [], []
    for start in starts:
        block = slice(start, start + 3)
        a, da = exp_chart_terms(q[block])
        moment = trial.residual[block]
        for i in range(3):
            for j in range(3):
                rows.append(start + i)
                columns.append(start + j)
                values.append(float(a[i, j]))
                curvature_rows.append(start + i)
                curvature_columns.append(start + j)
                curvature_values.append(float(da[:, i, j] @ moment))
    chart = sparse.coo_matrix(
        (values, (rows, columns)), shape=(count, count)).tocsr()
    curvature = sparse.coo_matrix(
        (curvature_values, (curvature_rows, curvature_columns)),
        shape=(count, count)).tocsr()
    tangent = (chart.T @ trial.tangent @ chart + curvature).tocsr()
    tangent.eliminate_zeros()
    residual = np.asarray(chart.T @ trial.residual).reshape(-1).copy()
    if not np.isfinite(residual).all() or not np.isfinite(tangent.data).all():
        raise ValueError("finite native B3 chart response required")
    return NativeB3SparseTrial(residual, tangent,
                               trial.candidate_histories, trial.family)
