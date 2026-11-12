"""Smooth cut integration of an ACTUAL B4-enriched Q4 against a physical S3.

37 local coordinates = 12 skin corners + 9 target corners + 4 bubbles
plus 12 source MID surface corners.  The bubble body frame is independent
of contact side; the physical director offset is applied after that field.
The boundary bubbles vanish, but inversion, interior position, normal and
reference pullback include them before differentiation.  No appended row is
used to imitate contact work.  Vertex/edge coincidences and unresolved
quadrature are explicit unsupported branches, not accepted zero derivatives.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from anyfesolver._ge_beam3_mixed_ad import Jet2, sqrt
from anyfesolver.control import SolveCancelled
from anyfesolver.e4_pl_enriched import EnrichedProfileError, bubble_basis, readonly

SIZE = 37


class EnrichedGeometryError(EnrichedProfileError):
    pass


def const(x):
    return Jet2.constant(x, SIZE)


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), const(0))


def sub(a, b):
    return [x-y for x, y in zip(a, b)]


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def cross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def unit(a):
    length = sqrt(dot(a, a))
    return [x/length for x in a]


def shape(r, s):
    return (((1-r)*(1-s)/4, (1+r)*(1-s)/4,
             (1+r)*(1+s)/4, (1-r)*(1+s)/4),
            (-(1-s)/4, (1-s)/4, (1+s)/4, -(1+s)/4),
            (-(1-r)/4, -(1+r)/4, (1+r)/4, (1-r)/4))


def interpolate(n, points):
    return [sum((n[i]*points[i][k] for i in range(len(n))), const(0))
            for k in range(len(points[0]))]


def physical(r, s, corners, beta, direction):
    n, nr, ns = shape(r, s)
    b, br, bs = bubble_basis(r, s)
    out = []
    for basis, extra in ((n, b), (nr, br), (ns, bs)):
        nodal = interpolate(basis, corners)
        amount = sum((extra[i]*beta[i] for i in range(4)), const(0))
        out.append([nodal[k]+amount*direction[k] for k in range(3)])
    return n, out[0], out[1], out[2]


@dataclass(frozen=True)
class QuadraturePolicy:
    """Explicit numerical integration controls, not scientific tolerances.

    The caller must bind these controls; they are not inferred from pressure
    size, a cut fraction, or a desired stability result.
    """
    value_absolute: float
    gradient_absolute: float
    hessian_absolute: float
    maximum_triangles: int

    def validate(self):
        if (not all(np.isfinite(x) and x > 0 for x in
                    (self.value_absolute, self.gradient_absolute, self.hessian_absolute))
                or type(self.maximum_triangles) is not int or self.maximum_triangles < 5):
            raise EnrichedGeometryError("positive finite, explicitly bound quadrature controls required")


@dataclass(frozen=True)
class EnrichedMoments:
    support: np.ndarray
    weighted_gap: np.ndarray
    pressure_gram: np.ndarray
    support_gradient: np.ndarray
    gap_integral_gradient: np.ndarray
    support_hessian: np.ndarray
    gap_integral_hessian: np.ndarray
    branch: tuple
    triangles: int
    quadrature_difference: tuple


def moments(reference_source, current_source, current_target, beta, direction,
            policy, *, current_midsurface, cancel=lambda: False):
    """Reference-area moments M_i, b_i and int(N_i N_j), with complete jets."""
    policy.validate()
    ref = readonly(reference_source, (4, 3))
    source = readonly(current_source, (4, 3))
    target = readonly(current_target, (3, 3))
    beta = readonly(beta, (4,))
    midsurface = readonly(current_midsurface, (4, 3))
    direction = readonly(direction, (3,))
    if not np.isclose(direction @ direction, 1., rtol=0, atol=64*np.finfo(float).eps):
        raise EnrichedGeometryError("unit physical reference director required")
    variables = [Jet2.variable(x, i, SIZE) for i, x in enumerate(
        np.concatenate((source.ravel(), target.ravel(), beta, midsurface.ravel())))]
    q = [variables[3*i:3*i+3] for i in range(4)]
    t = [variables[12+3*i:15+3*i] for i in range(3)]
    bb = variables[21:25]
    mid = [variables[25+3*i:28+3*i] for i in range(4)]
    body_normal = unit(cross(sub(mid[1], mid[0]), sub(mid[3], mid[0])))
    normal = unit(cross(sub(q[1], q[0]), sub(q[3], q[0])))
    tangent = unit(sub(q[1], q[0]))
    second = cross(normal, tangent)
    reference_normal = np.cross(ref[1]-ref[0], ref[3]-ref[0])
    reference_normal /= np.linalg.norm(reference_normal)
    if not np.allclose(direction, reference_normal, rtol=0, atol=128*np.finfo(float).eps):
        raise EnrichedGeometryError("reference director does not match source orientation")
    target_normal = cross(sub(t[1], t[0]), sub(t[2], t[0]))
    if dot(target_normal, normal).value >= 0:
        raise EnrichedGeometryError("target S3 must face the source skin")

    def project(p):
        return [dot(sub(p, q[0]), tangent), dot(sub(p, q[0]), second)]

    quad = [project(p) for p in q]
    triangle = [project(p) for p in t]
    scale = max(float(np.max(np.abs(source-source[0]))),
                float(np.max(np.abs(target-source[0]))), np.finfo(float).tiny)
    resolution = 256*np.finfo(float).eps*scale*scale
    area2 = cross2(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]))
    if abs(area2.value) <= resolution:
        raise EnrichedGeometryError("degenerate projected S3")
    if area2.value < 0:
        triangle = [triangle[0], triangle[2], triangle[1]]
        t = [t[0], t[2], t[1]]
        area2 = -area2
    for i in range(4):
        if cross2(sub(quad[(i+1) % 4], quad[i]),
                  sub(quad[(i+2) % 4], quad[(i+1) % 4])).value <= resolution:
            raise EnrichedGeometryError("nonconvex projected Q4")
    polygon = quad
    signature = []
    for edge in range(3):
        if not polygon:
            break
        start, end = triangle[edge], triangle[(edge+1) % 3]
        line = sub(end, start)
        old = polygon[-1]
        old_d = cross2(line, sub(old, start))
        out = []
        for point in polygon:
            d = cross2(line, sub(point, start))
            # This version admits smooth cut branches only.  Coincident
            # boundaries require an explicit later branch contract.
            if min(abs(old_d.value), abs(d.value)) <= resolution:
                raise EnrichedGeometryError("vertex on clipping boundary")
            old_in, inside = old_d.value > 0, d.value > 0
            signature.append(int(inside))
            if old_in != inside:
                alpha = old_d/(old_d-d)
                out.append([a+alpha*b for a, b in zip(old, sub(point, old))])
            if inside:
                out.append(point)
            old, old_d = point, d
        polygon = out
    if len(polygon) < 3:
        zeros = np.zeros(4)
        return EnrichedMoments(readonly(zeros), readonly(zeros), readonly(np.zeros((4, 4))),
                               readonly(np.zeros((4, SIZE))), readonly(np.zeros((4, SIZE))),
                               readonly(np.zeros((4, SIZE, SIZE))),
                               readonly(np.zeros((4, SIZE, SIZE))), tuple(signature), 0, (0., 0., 0.))

    def integrand(xy):
        # Solve the ENRICHED projected map, including all beta dependencies.
        r, s = const(0), const(0)
        for iteration in range(12):
            n, p, pr, ps = physical(r, s, q, bb, body_normal)
            mapped = project(p)
            residual = sub(mapped, xy)
            ar = [dot(pr, tangent), dot(pr, second)]
            ass = [dot(ps, tangent), dot(ps, second)]
            det = cross2(ar, ass)
            if det.value <= resolution:
                raise EnrichedGeometryError("folded enriched projected skin")
            r, s = r-cross2(residual, ass)/det, s-cross2(ar, residual)/det
            if iteration >= 2 and (max(abs(x.value) for x in residual) <= 1e-12*scale
                    and max(np.max(np.abs(x.gradient)) for x in residual) <= 1e-10
                    and max(np.max(np.abs(x.hessian)) for x in residual) <= 1e-8/scale):
                break
        n, p, pr, ps = physical(r, s, q, bb, body_normal)
        residual = sub(project(p), xy)
        if (max(abs(x.value) for x in residual) > 1e-11*scale
                or max(np.max(np.abs(x.gradient)) for x in residual) > 1e-9
                or max(np.max(np.abs(x.hessian)) for x in residual) > 1e-7/scale
                or max(abs(r.value), abs(s.value)) > 1+1e-10):
            raise EnrichedGeometryError("enriched projected inversion unresolved")
        cross_skin = cross(pr, ps)
        projected_jac = dot(cross_skin, normal)
        if projected_jac.value <= resolution:
            raise EnrichedGeometryError("nonpositive physical skin Jacobian")
        skin_normal = unit(cross_skin)
        # Target barycentric correspondence is in the same moving frame.
        v = sub(xy, triangle[0])
        l1 = cross2(v, sub(triangle[2], triangle[0]))/area2
        l2 = cross2(sub(triangle[1], triangle[0]), v)/area2
        target_point = interpolate((1-l1-l2, l1, l2), t)
        gap = dot(skin_normal, sub(p, target_point))
        _, nr, ns = shape(r, s)
        reference_jac = sqrt(dot(cross(interpolate(nr, ref), interpolate(ns, ref)),
                                  cross(interpolate(nr, ref), interpolate(ns, ref))))
        weight = reference_jac/projected_jac
        return [weight*x for x in n]+[weight*x*gap for x in n]+[
            weight*x*y for x in n for y in n]

    used = 0
    difference = np.zeros(3)
    # B4*Q1 moments have degree eight.  The old degree-five triangle rule
    # would need avoidable subdivisions even at a flat zero-beta state.
    # A 5x5 Duffy rule integrates those polynomial moments; adaptive checks
    # below still govern curved/rational moments and ALL derivatives.
    gauss, weights = np.polynomial.legendre.leggauss(5)
    triangle_rule = []
    for x, wx in zip(gauss, weights):
        a = (x+1)/2
        for y, wy in zip(gauss, weights):
            b = (y+1)/2
            triangle_rule.append(((1-a-(1-a)*b, a, (1-a)*b), wx*wy*(1-a)/2))

    def rule(vertices):
        nonlocal used
        if cancel():
            raise SolveCancelled("enriched integration cancelled")
        used += 1
        if used > policy.maximum_triangles:
            raise EnrichedGeometryError("enriched quadrature budget exhausted")
        area = cross2(sub(vertices[1], vertices[0]), sub(vertices[2], vertices[0]))/2
        if area.value <= resolution:
            raise EnrichedGeometryError("unresolved integration triangle")
        out = [const(0) for _ in range(24)]
        for bary, relative in triangle_rule:
            xy = [sum((bary[i]*vertices[i][k] for i in range(3)), const(0)) for k in range(2)]
            vals = integrand(xy)
            out = [a+area*relative*b for a, b in zip(out, vals)]
        return out

    def split(v):
        a, b, c = v
        ab = [(x+y)/2 for x, y in zip(a, b)]
        bc = [(x+y)/2 for x, y in zip(b, c)]
        ca = [(x+y)/2 for x, y in zip(c, a)]
        return ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))

    def integrate(vertices, tolerance, coarse=None):
        coarse = rule(vertices) if coarse is None else coarse
        children = split(vertices)
        parts = [rule(v) for v in children]
        fine = [sum((part[i] for part in parts), const(0)) for i in range(24)]
        error = np.array((max(abs(a.value-b.value) for a, b in zip(fine, coarse)),
                          max(np.max(np.abs(a.gradient-b.gradient)) for a, b in zip(fine, coarse)),
                          max(np.max(np.abs(a.hessian-b.hessian)) for a, b in zip(fine, coarse))))
        if np.all(error <= tolerance):
            difference[:] += error
            return fine
        refined = [integrate(v, tolerance/4, part) for v, part in zip(children, parts)]
        return [sum((part[i] for part in refined), const(0)) for i in range(24)]

    roots = [(polygon[0], polygon[i], polygon[i+1]) for i in range(1, len(polygon)-1)]
    tolerance = np.array((policy.value_absolute, policy.gradient_absolute,
                          policy.hessian_absolute))/len(roots)
    parts = [integrate(v, tolerance) for v in roots]
    out = [sum((part[i] for part in parts), const(0)) for i in range(24)]
    values = np.array([x.value for x in out])
    gradients = np.array([x.gradient for x in out[:8]])
    hessians = np.array([x.hessian for x in out[:8]])
    if not all(np.all(np.isfinite(x)) for x in (values, gradients, hessians)):
        raise EnrichedGeometryError("nonfinite enriched moments")
    if np.any(values[:4] <= 0):
        raise EnrichedGeometryError("nonpositive supported Q1 hat")
    return EnrichedMoments(readonly(values[:4]), readonly(values[4:8]),
                           readonly(values[8:].reshape(4, 4)), readonly(gradients[:4]),
                           readonly(gradients[4:8]), readonly(hessians[:4]),
                           readonly(hessians[4:8]), tuple(signature), used,
                           tuple(map(float, difference)))
