import numpy as np
from anyfesolver._ge_beam3_mixed_ad import Jet2


def _davenport(c):
    trace = np.trace(c)
    z = np.array([c[2, 1]-c[1, 2], c[0, 2]-c[2, 0], c[1, 0]-c[0, 1]])
    return np.block([[np.array([[trace]]), z[None, :]],
                     [z[:, None], c+c.T-trace*np.eye(3)]])


def rotation_jets(reference, current):
    """Equal nodal-weight proper Procrustes fit and analytic two derivatives.

Centered vectors retain their lengths (no unit-vector normalization). The
normalized largest quaternion eigenpair is differentiated implicitly. Only
the top eigenvalue must be simple; repeated lower eigenvalues are harmless.
"""
    ref = np.asarray(reference, dtype=float)
    x = np.asarray(current, dtype=float)
    if (ref.ndim != 2 or ref.shape[1:] != (3,) or len(ref) not in (3, 4)
            or x.shape != ref.shape or not np.isfinite(ref).all() or not np.isfinite(x).all()):
        raise ValueError('finite three/four-node shell fit required')
    ref = ref-ref.mean(axis=0)
    x = x-x.mean(axis=0)
    n = 6*len(ref)
    # Constant reference scaling improves eigenspace conditioning without
    # introducing a displacement-dependent coefficient into differentiation.
    scale = np.linalg.norm(ref)
    if not np.isfinite(scale) or scale == 0.:
        raise ValueError('degenerate reference shell fit')
    ref = ref/scale
    c = (x/scale).T@ref
    eigenvalues, vectors = np.linalg.eigh(_davenport(c))
    gap = eigenvalues[-1]-eigenvalues[-2]
    if not gap > 1e-11*max(np.finfo(float).tiny, np.max(np.abs(eigenvalues))):
        raise ValueError('nonunique proper shell fit; refine or cut back')
    q = vectors[:, -1]
    inverse = (vectors[:, :-1]/(eigenvalues[-1]-eigenvalues[:-1]))@vectors[:, :-1].T
    derivatives = np.zeros((n, 4, 4))
    for node, r in enumerate(ref):
        for axis in range(3):
            dc = np.zeros((3, 3)); dc[axis] = r/scale
            derivatives[6*node+axis] = _davenport(dc)
    dq = np.einsum('ab,jbc,c->aj', inverse, derivatives, q)
    dl = np.einsum('a,jab,b->j', q, derivatives, q)
    reduced = derivatives-dl[:, None, None]*np.eye(4)
    rhs = np.einsum('jab,bk->ajk', reduced, dq)
    ddq = np.einsum('ab,bjk->ajk', inverse, rhs+rhs.transpose(0, 2, 1))
    ddq -= q[:, None, None]*(dq.T@dq)[None, :, :]
    w, a, b, d = [Jet2(q[i], dq[i], ddq[i]) for i in range(4)]
    return [[w*w+a*a-b*b-d*d, 2*(a*b-w*d), 2*(a*d+w*b)],
            [2*(a*b+w*d), w*w-a*a+b*b-d*d, 2*(b*d-w*a)],
            [2*(a*d-w*b), 2*(b*d+w*a), w*w-a*a-b*b+d*d]]
