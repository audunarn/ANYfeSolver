"""Private stable scalar evaluation of the unchanged affine Q4 pose chart.

Source equations: _ge_beam3_g3c_local_shell.deformation and
_ge_beam3_pose_joint._exp_terms. Only their Exp/Log evaluator dependencies
change under GE_BEAM3_Q4_AFFINE_STABLE_CHART_ADDENDUM.md. The original
Davenport fit, Jet2 class, physical chart equations and guards are retained.
No public caller, registry arithmetic or accepted historical module changes.
"""
import numpy as np
from ._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, matvec, transpose
from ._ge_beam3_variational_shell import rotation_jets
from ._ge_beam3_g3c_local_shell import Kinematics, owned, array, rotations
from ._ge_beam3_g3c_so3_numerics import so3_exp, so3_log

NUMERICS_ID = 'GE_BEAM3_Q4_AFFINE_STABLE_CHART_NUMERICS_V1'


def deformation(reference, displacement, accepted_rotations):
    """Analytic value, first and second derivatives; no numerical fit derivative."""
    n = len(reference)
    if n not in (3, 4):
        raise ValueError('registered shell topology required')
    ref = array(reference, (n, 3)); u = array(displacement, (6*n,))
    qa = rotations(accepted_rotations, n)
    if np.any(np.linalg.norm(u.reshape(n, 6)[:, 3:], axis=1) >= .9*np.pi):
        raise ValueError('trial shell increment requires cutback')
    rotation = rotation_jets(ref, ref+u.reshape(n, 6)[:, :3])
    rt = transpose(rotation)
    z = [[Jet2.variable(u[6*i+j], 6*i+j, 6*n) for j in range(6)] for i in range(n)]
    x = [[z[i][j]+ref[i, j] for j in range(3)] for i in range(n)]
    centre = [sum(row[j] for row in x)/n for j in range(3)]
    refc = ref-ref.mean(axis=0)
    q = [matmul(so3_exp(z[i][3:]), constant_matrix(qa[i], 6*n)) for i in range(n)]
    values = []
    for i in range(n):
        local = matvec(rt, [x[i][j]-centre[j] for j in range(3)])
        values.extend(local[j]-refc[i, j] for j in range(3))
        values.extend(so3_log(matmul(rt, q[i])))
    return Kinematics(owned([v.value for v in values]), owned([v.gradient for v in values]),
        owned([v.hessian for v in values]), owned([[v.value for v in row] for row in rotation]),
        owned([[[v.value for v in row] for row in qi] for qi in q]),
        ref, owned(ref+u.reshape(n,6)[:,:3]))


def _exp_terms(vector):
    """Same analytic spatial Exp differential/derivative, stable scalar factors."""
    jets = so3_exp([Jet2.variable(float(v), i, 3) for i, v in enumerate(vector)])
    q = np.array([[v.value for v in row] for row in jets])
    dq = np.array([[v.gradient for v in row] for row in jets])
    ddq = np.array([[v.hessian for v in row] for row in jets])
    def axial(m):
        return np.array([m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1]])/2
    a = np.column_stack([axial(dq[:, :, j]@q.T) for j in range(3)])
    da = np.empty((3, 3, 3))
    for j in range(3):
        for k in range(3):
            da[:, j, k] = axial(ddq[:, :, j, k]@q.T+dq[:, :, j]@dq[:, :, k].T)
    return q, a, da
