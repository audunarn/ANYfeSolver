"""Bounded Jacobi SVD for private reference-energy calculations.

The SciPy mapping is verified against v1.16.3 flapack_other.pyf.src:
CEFGAR / UFWN / VJWN and 0=N for range, transpose and perturbation.
No fallback, numerical-rank truncation permission or structured perturbation.
"""

import numpy as np
from scipy.linalg import lapack


def relative_reference_svd(matrix):
    from ._native_modal_capacity import limits
    coordinates,rows,_=limits()
    a = np.array(matrix, dtype=float, copy=True)
    if (a.ndim != 2 or not 1 <= a.shape[1] <= min(a.shape[0], coordinates)
            or a.shape[0] > rows or not np.isfinite(a).all()):
        raise ValueError('finite tall bounded reference matrix required')
    # G estimates accuracy with both row and column preprocessing; U/J
    # explicitly accumulates right rotations. N/N/N forbids optional killing
    # of small columns, speculative transposition and structured perturbation.
    s, u, v, work, iwork, info = lapack.dgejsv(a, joba=3, jobu=0, jobv=1,
        jobr=0, jobt=0, jobp=0, overwrite_a=0)
    if info != 0 or iwork[2] != 0:
        raise ValueError('Jacobi reference SVD failure or accuracy warning')
    if (work[1] <= 0. or work[0] <= 0. or not np.isfinite(work[:2]).all()
            or s.shape != (a.shape[1],) or u.shape != a.shape
            or v.shape != (a.shape[1], a.shape[1])):
        raise ValueError('invalid Jacobi reference scaling or shape')
    with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
        try:
            s = s*(work[0]/work[1])
        except FloatingPointError as exc:
            raise ValueError('unrepresentable Jacobi reference scaling') from exc
    if (not np.isfinite(s).all() or not np.isfinite(u).all() or not np.isfinite(v).all()
            or np.any(s < 0.) or np.any(np.diff(s) > 0.)):
        raise ValueError('invalid Jacobi reference spectrum')
    reconstruction = float(np.linalg.norm((u*s)@v.T-a))
    scale = max(1., float(np.linalg.norm(a)))
    if not np.isfinite(reconstruction) or not np.isfinite(scale) or reconstruction > 1e-11*scale:
        raise ValueError('Jacobi reference reconstruction failed')
    if np.linalg.norm(v.T@v-np.eye(len(s))) > 1e-11:
        raise ValueError('Jacobi reference right-vector orthogonality failed')
    return u, s, v.T
