"""Private conservative shell pullback for the GE-B3 connection successor.

Unchanged local Q4/S3 operators supply forces, material history and tangent.
The new connection policy differentiates the COMPLETE deformational map, not
just its rigid rotation. It neither replaces the existing corotational route
nor grants production qualification. See GE_BEAM3_VARIATIONAL_SHELL_MAP.md.
"""
import numpy as np

from ._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, matvec, transpose, so3_exp, so3_log
from ._ge_beam3_pose_joint import _exp_terms

POLICY = 'GE_BEAM3_PROCRUSTES_VARIATIONAL_SHELL_PULLBACK_V2'


def _davenport(c):
    trace = np.trace(c)
    z = np.array([c[2, 1]-c[1, 2], c[0, 2]-c[2, 0], c[1, 0]-c[0, 1]])
    return np.block([[np.array([[trace]]), z[None, :]],
                     [z[:, None], c+c.T-trace*np.eye(3)]])


def _quaternion_jets(reference, current, *, displacement=None):
    """Equal nodal-weight proper Procrustes fit and analytic two derivatives.

Centered vectors retain their lengths (no unit-vector normalization). The
normalized largest quaternion eigenpair is differentiated implicitly. Only
the top eigenvalue must be simple; repeated lower eigenvalues are harmless.
"""
    ref = np.asarray(reference, dtype=float)
    x = np.asarray(current if displacement is None else displacement, dtype=float)
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
    if displacement is None:
        matrix = _davenport((x/scale).T@ref)
    else:
        # Algebraically the same centered covariance, without forming ref+u.
        # Build the linear Davenport contributions separately: subtracting
        # O(1) covariance entries would erase tiny antisymmetric increments.
        matrix = _davenport(ref.T@ref)+_davenport((x/scale).T@ref)
    eigenvalues, vectors = np.linalg.eigh(matrix)
    gap = eigenvalues[-1]-eigenvalues[-2]
    if not gap > 1e-11*max(np.finfo(float).tiny, np.max(np.abs(eigenvalues))):
        raise ValueError('nonunique proper shell fit; refine or cut back')
    q = vectors[:, -1]
    if displacement is None:
        inverse = (vectors[:, :-1]/(eigenvalues[-1]-eigenvalues[:-1]))@vectors[:, :-1].T
    else:
        # Recover small quaternion components from the preserved covariance
        # increment. An eigensolver guarantees absolute eigenvector accuracy,
        # which can swamp a tiny physical rotation. The largest-component
        # chart avoids division by a small quaternion component.
        pivot = int(np.argmax(np.abs(q)))
        rest = np.array([i for i in range(4) if i != pivot])
        refined = np.zeros(4)
        refined[pivot] = 1.
        refined[rest] = np.linalg.solve(
            eigenvalues[-1]*np.eye(3)-matrix[np.ix_(rest, rest)],
            matrix[rest, pivot])
        refined /= np.linalg.norm(refined)
        q = refined if q[pivot] >= 0. else -refined
        # Differentiate on THIS quaternion's orthogonal complement. Reusing
        # the old lower eigenvectors would break the normalization identities.
        eigenvalue = float(q@matrix@q)
        bordered = np.block([[eigenvalue*np.eye(4)-matrix, q[:, None]],
                             [q[None, :], np.zeros((1, 1))]])
        inverse = np.linalg.solve(bordered, np.vstack((np.eye(4), np.zeros((1, 4)))))[:4]
    derivatives = np.zeros((n, 4, 4))
    derivative_reference = ref if displacement is None else ref-ref.mean(axis=0)
    for node, r in enumerate(derivative_reference):
        for axis in range(3):
            dc = np.zeros((3, 3)); dc[axis] = r/scale
            derivatives[6*node+axis] = _davenport(dc)
    dq = np.einsum('ab,jbc,c->aj', inverse, derivatives, q)
    dl = np.einsum('a,jab,b->j', q, derivatives, q)
    reduced = derivatives-dl[:, None, None]*np.eye(4)
    rhs = np.einsum('jab,bk->ajk', reduced, dq)
    ddq = np.einsum('ab,bjk->ajk', inverse, rhs+rhs.transpose(0, 2, 1))
    ddq -= q[:, None, None]*(dq.T@dq)[None, :, :]
    return [Jet2(q[i], dq[i], ddq[i]) for i in range(4)]


def rotation_jets(reference, current):
    """Proper Procrustes rotation with the historical two-argument contract."""
    w, a, b, d = _quaternion_jets(reference, current)
    return [[w*w+a*a-b*b-d*d, 2*(a*b-w*d), 2*(a*d+w*b)],
            [2*(a*b+w*d), w*w-a*a+b*b-d*d, 2*(b*d-w*a)],
            [2*(a*d-w*b), 2*(b*d+w*a), w*w-a*a-b*b+d*d]]


def deformation(reference, displacement, *, stable_translation=False, rotation_anchors=None,
                stable_rotation=False):
    """Value, differential and Hessian of all corotated nodal coordinates."""
    ref = np.asarray(reference, dtype=float)
    u = np.asarray(displacement, dtype=float)
    if type(stable_rotation) is not bool:
        raise ValueError('typed shell rotation numerical selection required')
    exponential, logarithm = so3_exp, so3_log
    if stable_rotation:
        from ._ge_beam3_g3c_so3_numerics import so3_exp as exponential, so3_log as logarithm
    if u.shape != (6*len(ref),) or not np.isfinite(u).all():
        raise ValueError('finite complete shell coordinates required')
    n = len(u)
    if rotation_anchors is not None:
        from ._stiffened_shell_rotation_charts import owned_anchors
        rotation_anchors = owned_anchors(rotation_anchors, len(ref))
    if stable_translation:
        w, a, b, d = _quaternion_jets(
            ref, None, displacement=u.reshape(-1, 6)[:, :3])
        # Unit-quaternion R-I, including its full two derivatives. Avoid
        # subtracting one from a near-unit diagonal after rotation evaluation.
        increment = [[-2*(b*b+d*d), 2*(a*b-w*d), 2*(a*d+w*b)],
                     [2*(a*b+w*d), -2*(a*a+d*d), 2*(b*d-w*a)],
                     [2*(a*d-w*b), 2*(b*d+w*a), -2*(a*a+b*b)]]
        rotation = [[increment[i][j]+float(i == j) for j in range(3)]
                    for i in range(3)]
    else:
        current = ref+u.reshape(-1, 6)[:, :3]
        rotation = rotation_jets(ref, current)
    rt = transpose(rotation)
    jets = [[Jet2.variable(u[6*i+j], 6*i+j, n) for j in range(6)] for i in range(len(ref))]
    x = [[jets[i][j]+(0. if stable_translation else ref[i, j])
          for j in range(3)] for i in range(len(ref))]
    centre = [sum(row[j] for row in x)/len(x) for j in range(3)]
    refc = ref-ref.mean(axis=0)
    values = []
    for i in range(len(ref)):
        local = matvec(rt, [x[i][j]-centre[j] for j in range(3)])
        if stable_translation:
            rotated_reference = [sum(increment[k][j]*refc[i, k]
                                     for k in range(3)) for j in range(3)]
            values.extend(local[j]+rotated_reference[j] for j in range(3))
        else:
            values.extend(local[j]-refc[i, j] for j in range(3))
        physical = exponential(jets[i][3:])
        if rotation_anchors is not None:
            physical = matmul(physical, constant_matrix(rotation_anchors[i], n))
        values.extend(logarithm(matmul(rt, physical)))
    return (np.array([v.value for v in values]),
            np.array([v.gradient for v in values]),
            np.array([v.hessian for v in values]))


def response(model, element, displacement, origin, layers, *, split=False,
             rotation_anchors=None, stable_translation=False):
    """Spatial wrench rows, additive columns, and unchanged local trial state.

g=D.T f; H=D.T K D + sum(f_i Hess(d_i)). Spatial rows then follow
P.T r=g and P.T J=H-dP.T r. No post-hoc symmetrization is performed.
"""
    if type(stable_translation) is not bool:
        raise ValueError('explicit shell translation arithmetic policy required')
    ref = np.array([model.mesh.nodes[i].coords() for i in element.node_ids])
    u = np.asarray(displacement, dtype=float)
    local, differential, second = deformation(ref, u, rotation_anchors=rotation_anchors,
                                               stable_translation=stable_translation)
    material = model.get_material(element.material_name)
    # This successor currently owns virgin elastic shells only. Do not imply
    # a conservative potential for active plastic/unsupported section states.
    if material.yield_stress != 0. or material.hardening_curve is not None or element.shell_section is not None:
        raise ValueError('variational shell successor requires its owned elastic section')
    if split and element.num_nodes == 4:
        f, k, candidate, components = element.compute_nonlinear_response(model.mesh, material,
            local, origin, layers, True, _return_tangent_components=True)
        km, kg = components['material'], components['geometric']
    else:
        f, k, candidate = element.compute_nonlinear_response(model.mesh, material,
            local, origin, layers, True)
        # The admitted elastic S3 V2D local strains are linear in local_u.
        # Its finite geometric variation belongs entirely to this pullback.
        km, kg = k, np.zeros_like(k)
    f, k = np.asarray(f), np.asarray(k)
    if not np.isfinite(f).all() or not np.isfinite(k).all():
        raise ValueError('nonfinite local shell response')
    if np.linalg.norm(k-k.T) > 1e-11*max(1., np.linalg.norm(k)):
        raise ValueError('nonconservative local shell tangent')
    g = differential.T@f
    h = differential.T@k@differential+np.einsum('i,ijk->jk', f, second)
    p = np.eye(len(u)); blocks = []
    for node, row in enumerate(u.reshape(-1, 6)):
        _, a, da = _exp_terms(row[3:])
        singular = np.linalg.svd(a, compute_uv=False)
        if singular[-1] <= 1e-11*max(1., singular[0]):
            raise ValueError('singular additive shell chart; owner rebase required')
        s = slice(6*node+3, 6*node+6)
        p[s, s] = a; blocks.append((s, da))
    residual = np.linalg.solve(p.T, g)
    correction = np.zeros_like(h)
    for s, da in blocks:
        correction[s, s] = np.einsum('ijk,i->jk', da, residual[s])
    tangent = np.linalg.solve(p.T, h-correction)
    if split:
        inverse = np.linalg.solve(p, np.eye(len(u)))
        b = differential@inverse
        material = b.T@km@b
        geometric = np.linalg.solve(p.T, differential.T@kg@differential
            +np.einsum('i,ijk->jk', f, second)-correction)@inverse
        if np.linalg.norm(material+geometric-tangent@inverse) > 1e-11*max(1., np.linalg.norm(tangent@inverse)):
            raise ValueError('shell variational material/geometric split mismatch')
        return residual, tangent, candidate, material, geometric
    return residual, tangent, candidate
