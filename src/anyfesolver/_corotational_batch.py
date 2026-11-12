"""Batch the existing work-conjugate geometric-tangent FD stencils.

Same equations, step sizes and scalar fallback as the original evaluation.
The scalar production kernel remains the comparison oracle and fallback.
"""
from __future__ import annotations

import numpy as np
from . import corotational as cr

ORIGINAL = cr._scalar_work_conjugate_corotational_tangent


def skew(v):
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def exp(v):
    a2 = np.sum(v*v, axis=-1)
    a = np.sqrt(a2)
    small = a < 1e-2
    c1, c2 = np.empty_like(a), np.empty_like(a)
    x = a2[small]
    c1[small] = 1-x*(1/6-x*(1/120-x/5040))
    c2[small] = .5-x*(1/24-x*(1/720-x/40320))
    c1[~small] = np.sin(a[~small])/a[~small]
    c2[~small] = 2*np.sin(.5*a[~small])**2/a2[~small]
    K = skew(v)
    return np.eye(3)+c1[..., None, None]*K+c2[..., None, None]*(K@K)


def log(R):
    v = .5*np.stack((R[..., 2, 1]-R[..., 1, 2],
                     R[..., 0, 2]-R[..., 2, 0],
                     R[..., 1, 0]-R[..., 0, 1]), axis=-1)
    sin_angle = np.linalg.norm(v, axis=-1)
    cos_angle = np.clip(.5*(np.trace(R, axis1=-2, axis2=-1)-1), -1, 1)
    angle = np.arctan2(sin_angle, cos_angle)
    small, near_pi = angle < 1e-4, angle > np.pi-1e-6
    factor = np.ones_like(angle)
    s2 = sin_angle[small]**2
    factor[small] = 1+s2*(1/6+.075*s2)
    ordinary = ~(small | near_pi)
    factor[ordinary] = angle[ordinary]/sin_angle[ordinary]
    result = factor[..., None]*v
    for index in zip(*np.nonzero(near_pi)):
        result[index] = cr.rotation_vector_from_matrix(R[index])
    return result


def inverse_jacobian(v, left=False):
    a2 = np.sum(v*v, axis=-1)
    a = np.sqrt(a2)
    small = a < 1e-2
    c = np.empty_like(a)
    x = a2[small]
    c[small] = 1/12+x*(1/720+x*(1/30240+x/1209600))
    c[~small] = 1/a2[~small]-.5/a[~small]*(np.cos(.5*a[~small])/np.sin(.5*a[~small]))
    K = skew(v)
    return np.eye(3)+(-.5 if left else .5)*K+c[..., None, None]*(K@K)


def right_jacobian(v):
    a2 = np.sum(v*v, axis=-1)
    a = np.sqrt(a2)
    small = a < 1e-2
    c1, c2 = np.empty_like(a), np.empty_like(a)
    x = a2[small]
    c1[small] = .5-x*(1/24-x*(1/720-x/40320))
    c2[small] = 1/6-x*(1/120-x*(1/5040-x/362880))
    c1[~small] = 2*np.sin(.5*a[~small])**2/a2[~small]
    c2[~small] = (a[~small]-np.sin(a[~small]))/(a2[~small]*a[~small])
    K = skew(v)
    return np.eye(3)-c1[..., None, None]*K+c2[..., None, None]*(K@K)


def minimal(a, b):
    a, b = np.broadcast_arrays(a, b)
    a, b = a.copy(), b.copy()
    for vec in (a, b):
        norms = np.linalg.norm(vec, axis=-1)
        mask = norms > cr._SMALL
        vec[mask] /= norms[mask, None]
    dot = np.sum(a*b, axis=-1)
    special = dot < -1+1e-12
    K = skew(np.cross(a, b))
    denominator = np.where(special, 1, 1+dot)
    R = np.eye(3)+K+(K@K)/denominator[..., None, None]
    for index in zip(*np.nonzero(special)):
        R[index] = cr._minimal_rotation(a[index], b[index])
    return R


def beam_rotation(reference, deformed, rotations):
    # Beam callers provide a translation-centered geometry gauge. All frame,
    # sensitivity and Hessian stencils retain it, matching the scalar pullback.
    t_ref = reference.coordinates[-1]-reference.coordinates[0]
    t_ref = t_ref/max(np.linalg.norm(t_ref), cr._SMALL)
    t_def = deformed[..., -1, :]-deformed[..., 0, :]
    t_def /= np.maximum(np.linalg.norm(t_def, axis=-1, keepdims=True), cr._SMALL)
    nodes = exp(rotations)
    aligned = minimal(nodes@t_ref, t_def[..., None, :]) @ nodes
    R0 = aligned[..., 0, :, :]
    relative = np.swapaxes(R0, -1, -2)[..., None, :, :]@aligned
    sin_psi = .5*((relative[..., 2, 1]-relative[..., 1, 2])*t_ref[0]
                   +(relative[..., 0, 2]-relative[..., 2, 0])*t_ref[1]
                   +(relative[..., 1, 0]-relative[..., 0, 1])*t_ref[2])
    cos_psi = .5*(np.trace(relative, axis1=-2, axis2=-1)
                   -np.einsum('i,...ij,j->...', t_ref, relative, t_ref))
    twists = np.arctan2(sin_psi, cos_psi)
    twists[..., 0] = 0
    mean = np.mean(twists, axis=-1)
    return R0@exp(mean[..., None]*t_ref)


def shell_rotation(reference, deformed):
    xi, eta = reference.center_dN
    tx = np.einsum('n,mni->mi', xi, deformed)
    te = np.einsum('n,mni->mi', eta, deformed)
    e3 = np.cross(tx, te)
    e3 /= np.linalg.norm(e3, axis=-1, keepdims=True)
    e1 = tx-np.sum(tx*e3, axis=-1, keepdims=True)*e3
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(e3, e1)
    return np.stack((e1, e2, e3), axis=-1)@reference.frame.T


def sensitivity(reference, deformed, rotations, R, h):
    count, n = deformed.shape[:2]
    if reference.category == 'shell':
        xi, eta = reference.center_dN
        tx = np.einsum('n,mni->mi', xi, deformed)
        te = np.einsum('n,mni->mi', eta, deformed)
        frame = R@reference.frame
        e1, e2, e3 = frame[..., 0], frame[..., 1], frame[..., 2]
        L1 = np.sum(tx*e1, axis=-1)
        alpha = np.sum(te*e1, axis=-1)
        beta = np.sum(te*e2, axis=-1)
        if not np.all((L1 > cr._SMALL) & (np.abs(beta) > cr._SMALL)
                      & (np.abs(np.sum(tx*e3, axis=-1)) <= 1e-9*L1)):
            raise ArithmeticError('fast shell stencil leaves its scalar admission')
        jxi = np.stack((-alpha[:, None]/(L1*beta)[:, None]*e3,
                        -e3/L1[:, None], e2/L1[:, None]), axis=1)
        jeta = np.zeros_like(jxi)
        jeta[:, 0] = e3/beta[:, None]
        gx, ge = frame@jxi, frame@jeta
        blocks = xi[None, :, None, None]*gx[:, None]+eta[None, :, None, None]*ge[:, None]
        blocks -= np.mean(blocks, axis=1, keepdims=True)
        G = np.zeros((count, 3, n, 6))
        G[..., :3] = np.transpose(blocks, (0, 2, 1, 3))
        return G.reshape(count, 3, 6*n)
    cols = np.array((0, 1, 2, 3, 4, 5, 9, 10, 11))
    widths = np.where(cols % 6 < 3, h, 1e-6)
    dd = np.repeat(deformed[:, None], 2*len(cols), axis=1)
    rr = np.repeat(rotations[:, None], 2*len(cols), axis=1)
    for j, (col, width) in enumerate(zip(cols, widths)):
        target = dd if col % 6 < 3 else rr
        target[:, 2*j, col//6, col % 3] += width
        target[:, 2*j+1, col//6, col % 3] -= width
    Rp = beam_rotation(reference, dd, rr)
    relative = log(Rp@np.swapaxes(R, -1, -2)[:, None])
    derivative = (relative[:, ::2]-relative[:, 1::2])/(2*widths[None, :, None])
    G = np.zeros((count, 3, 12))
    G[:, :, cols] = np.transpose(derivative, (0, 2, 1))
    G[:, :, 6:9] = -G[:, :, :3]
    blocks = G.reshape(count, 3, n, 6)
    blocks[..., :3] -= np.mean(blocks[..., :3], axis=2, keepdims=True)
    return G


def pullback(reference, deformed, rotations, R, f_trans, f_rot, h):
    Rt = np.swapaxes(R, -1, -2)
    ft = f_trans[None]@Rt
    pt = ft-np.mean(ft, axis=1, keepdims=True)
    arms = deformed-np.mean(deformed, axis=1, keepdims=True)
    uf = np.sum(np.cross(ft, arms), axis=1)
    defrot = log(Rt[:, None]@exp(rotations))
    D = inverse_jacobian(defrot)@right_jacobian(rotations)
    U = -inverse_jacobian(defrot, left=True)@Rt[:, None]
    pr = np.einsum('mnij,ni->mnj', D, f_rot)
    uf += np.einsum('mnij,ni->mj', U, f_rot)
    G = sensitivity(reference, deformed, rotations, R, h)
    p = np.concatenate((pt, pr), axis=-1).reshape(deformed.shape[0], -1)
    return p+np.einsum('mij,mi->mj', G, uf)


def work_conjugate_tangent(reference, element, deformed, rotations, R_rig, G, B, f_ref, k_ref):
    if (reference.category == 'shell' and reference.center_dN is None
            or reference.category == 'beam' and deformed.shape[0] != 2):
        return ORIGINAL(reference, element, deformed, rotations, R_rig, G, B, f_ref, k_ref)
    material = B.T@k_ref@B
    n = deformed.shape[0]
    if np.linalg.norm(f_ref) <= cr._SMALL:
        return material, np.zeros((6*n, 6*n))
    spread = np.max(np.linalg.norm(reference.coordinates-reference.centroid, axis=1))
    ht = 1e-5*max(spread, 1e-3)
    cols = np.arange(6*n) if reference.category == 'shell' else np.array((0, 1, 2, 3, 4, 5, 9, 10, 11))
    widths = np.where(cols % 6 < 3, ht, 1e-5)
    dd = np.repeat(deformed[None], 2*len(cols), axis=0)
    rr = np.repeat(rotations[None], 2*len(cols), axis=0)
    for j, (col, width) in enumerate(zip(cols, widths)):
        target = dd if col % 6 < 3 else rr
        target[2*j, col//6, col % 3] += width
        target[2*j+1, col//6, col % 3] -= width
    R = shell_rotation(reference, dd) if reference.category == 'shell' else beam_rotation(reference, dd, rr)
    f = f_ref.reshape(n, 6)
    try:
        p = pullback(reference, dd, rr, R, f[:, :3], f[:, 3:], 1e-6*max(spread, 1e-3))
    except ArithmeticError:
        return ORIGINAL(reference, element, deformed, rotations, R_rig, G, B, f_ref, k_ref)
    geom = np.zeros((6*n, 6*n))
    geom[:, cols] = ((p[::2]-p[1::2])/(2*widths[:, None])).T
    if reference.category == 'beam':
        geom[:, 6:9] = -geom[:, :3]
    return material, .5*(geom+geom.T)
