"""Equivalent nodal vectors of a pressure acting on PHYSICAL overlap regions of one shell element.

The geometry owner supplies physical triangles (pieces of source-panel / wet-region overlap polygons) in the element's
own coordinate frame.  This module integrates the element's OWN pressure virtual-work density over exactly those
triangles: owner quadrature (``pressure_regions._rule``), every physical quadrature point is INVERTED through the actual
element mapping (no assumption that reference polygon edges stay straight under a non-affine quad map), and the density per
unit physical area ``vectors / magnitude`` of ``pressure_regions._integrand`` is used, so the native S3 path keeps its
full translational AND rotational (l, m) work (6 DOF per node).  No averaging over the element, no whole-element scalar.
"""
from dataclasses import dataclass
import numpy as np

from .pressure_regions import (PressureRegionError, _check_element, _integrand, _is_native, _rule, element_kernel)

_INVERT_TOL = 1e-10


@dataclass(frozen=True)
class RegionVectors:
    node_ids: tuple
    vectors: np.ndarray       # (n_nodes, 6): [Fx, Fy, Fz, Mx, My, Mz] (real or complex, same dtype as the field)
    area: float               # physical area integrated


def _node_coordinates(element, mesh):
    if hasattr(element, 'get_node_coordinates'):
        return np.asarray(element.get_node_coordinates(mesh), dtype=float)
    return np.array([mesh.nodes[int(i)].coords() for i in element.node_ids], dtype=float)


def _invert(element, mesh, coords, native, points):
    """Reference coordinates of physical ``points`` on the element surface (refuses points off the element)."""
    n = len(element.node_ids)
    scale = max(1.0, float(np.abs(coords).max()))
    if native:
        c0, c1, c2 = coords[:3]
        a, b = c1 - c0, c2 - c0
        g = np.array([[a @ a, a @ b], [a @ b, b @ b]])
        rhs = np.stack([(points - c0) @ a, (points - c0) @ b])
        xi = np.linalg.solve(g, rhs).T
        resid = points - (c0 + np.outer(xi[:, 0], a) + np.outer(xi[:, 1], b))
        if np.abs(resid).max() > 1e-8 * scale or np.any(xi < -1e-9) or np.any(xi.sum(axis=1) > 1 + 1e-9):
            raise PressureRegionError('overlap region lies off the native shell element')
        return xi
    tri = n in (3, 6)
    out = np.zeros((len(points), 2))
    for k, x in enumerate(points):
        xi = np.array([1 / 3, 1 / 3]) if tri else np.zeros(2)
        for _ in range(60):
            N, dxi, deta = (np.asarray(v, dtype=float) for v in element.compute_shape_functions(float(xi[0]), float(xi[1])))
            pos = N @ coords
            J = np.stack([dxi @ coords, deta @ coords], axis=1)       # (3,2)
            r = x - pos
            step = np.linalg.lstsq(J, r, rcond=None)[0]
            xi = xi + step
            if np.abs(step).max() < 1e-13:
                break
        N = np.asarray(element.compute_shape_functions(float(xi[0]), float(xi[1]))[0], dtype=float)
        if np.abs(x - N @ coords).max() > 1e-8 * scale:
            raise PressureRegionError('overlap region lies off the shell element surface (inversion failed)')
        lim = 1e-9
        inside = (xi[0] >= -lim and xi[1] >= -lim and xi.sum() <= 1 + lim) if tri else bool(np.all(np.abs(xi) <= 1 + lim))
        if not inside:
            raise PressureRegionError('overlap region lies outside the element reference domain')
        out[k] = xi
    return out


def prove_owner_kernel(element, mesh, coordinates, proof=None):
    """Run the PUBLIC owner kernel proof (``pressure_regions.element_kernel`` -> ``_require_native_equivalence``): the interpolation
    used here must equal the element's own uniform dead-pressure load in every component (translations AND the native S3 18-DOF
    moments); otherwise the formulation is refused.  ``proof`` is an optional caller-owned (job-lifetime) dict: a proof is reused only
    for the identical element object AND identical node coordinates, never across jobs."""
    key = id(element)
    identity = (len(element.node_ids), coordinates.tobytes())
    if proof is not None and proof.get(key) == identity:
        return
    element_kernel(element, mesh)
    if proof is not None:
        proof[key] = identity


def region_nodal_vectors(element, mesh, triangles, field, *, sign=1.0, kernel_proof=None) -> RegionVectors:
    """``sign * integral p(x) g_i dA`` over the physical ``triangles`` (k,3,3); ``field(points, tri_index) -> (m,)``.

    The owner kernel equivalence proof (all components, native S3 included) is evaluated first (see :func:`prove_owner_kernel`)."""
    if sign not in (-1.0, 1.0):
        raise PressureRegionError('orientation sign must be +1 or -1')
    _check_element(element)
    prove_owner_kernel(element, mesh, _node_coordinates(element, mesh), kernel_proof)
    native = _is_native(element)
    T = np.asarray(triangles, dtype=float)
    if T.ndim != 3 or T.shape[1:] != (3, 3) or not np.isfinite(T).all():
        raise PressureRegionError('finite (k,3,3) physical triangles required')
    coords = _node_coordinates(element, mesh)
    rp, rw = _rule(element)
    bary = np.column_stack([1.0 - rp[:, 0] - rp[:, 1], rp[:, 0], rp[:, 1]])                  # (q,3)
    pts = np.einsum('qa,kad->kqd', bary, T).reshape(-1, 3)                                    # physical points
    cross = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    weights = (2.0 * area[:, None] * rw[None, :]).reshape(-1)
    tri_index = np.repeat(np.arange(len(T)), len(rw))
    if not len(T):
        return RegionVectors(tuple(int(i) for i in element.node_ids), np.zeros((len(element.node_ids), 6)), 0.0)
    ref = _invert(element, mesh, coords, native, pts)
    vectors, magnitude = _integrand(element, mesh).evaluate(ref)                              # (m,n,6), (m,)
    density = vectors / magnitude[:, None, None]                                              # per unit physical area
    values = np.asarray(field(pts, tri_index))
    if values.shape != (len(pts),) or not np.isfinite(values).all():
        raise PressureRegionError('pressure field must return one finite value per quadrature point')
    nodal = sign * np.einsum('m,mnk->nk', weights * values, density)
    return RegionVectors(tuple(int(i) for i in element.node_ids), nodal, float(area.sum()))


def full_element_vectors(element, mesh, pressure):
    """Reference: constant pressure on the WHOLE element through the owner kernel (consistency checks / tests)."""
    kernel, total_area, _ = element_kernel(element, mesh)
    n_corner = kernel.shape[2]
    return np.einsum('nkc,c->nk', kernel, np.full(n_corner, float(pressure))), total_area
