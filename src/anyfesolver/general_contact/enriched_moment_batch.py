"""Dedicated nonlinear-profile point-batched equivalent physical moments.

Protected legacy evaluators remain untouched. Clipping, domain guards,
adaptive tolerances/allocation and triangle count semantics are retained;
only independent quadrature-point arithmetic is vectorized.
"""
from __future__ import annotations
import numpy as np
from .enriched_overlap import (Jet2,sqrt,SIZE,const,dot,sub,cross,cross2,unit,
    shape,interpolate,physical,readonly,EnrichedGeometryError,EnrichedMoments,SolveCancelled)
from .moment_batch_ad import PointJet
from .moment_batch_points import make_integrand

def batch_moments(reference_source, current_source, current_target, beta, direction,
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

    integrand = make_integrand(ref,q,t,bb,body_normal,normal,tangent,second,triangle,area2,scale,resolution)

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
        xys = [[sum((bary[i]*vertices[i][k] for i in range(3)), const(0))
                for k in range(2)] for bary, relative in triangle_rule]
        vals = integrand(xys)
        relative = np.array([relative for bary, relative in triangle_rule])
        weighted_area = PointJet.broadcast(area, len(triangle_rule))*relative
        return [(weighted_area*value).sum_to_jet() for value in vals]

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

def selected_batch_moments(reference_source, current_source, current_target, beta, direction,
            policy, *, current_midsurface, internal_edge_nodes, target_node_ids, cancel=lambda: False):
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
    target_order = (0,2,1) if area2.value < 0 else (0,1,2)
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
    # triangle/t were reversed together above; retain actual node identities.
    ids = [target_node_ids[i] for i in target_order]
    internal = frozenset(internal_edge_nodes)
    admitted_internal_tie = False
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
            # Only a pre-certified INTERNAL vertex tie is permitted here.
            # External ties and shared-edge collinearity remain unsupported.
            if min(abs(old_d.value), abs(d.value)) <= resolution:
                if frozenset((ids[edge],ids[(edge+1)%3])) != internal:
                    raise EnrichedGeometryError("external vertex on clipping boundary")
                admitted_internal_tie = True
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

    integrand = make_integrand(ref,q,t,bb,body_normal,normal,tangent,second,triangle,area2,scale,resolution)

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
        if area.value < 0 or (area.value <= resolution and not admitted_internal_tie):
            raise EnrichedGeometryError("unresolved integration triangle")
        xys = [[sum((bary[i]*vertices[i][k] for i in range(3)), const(0))
                for k in range(2)] for bary, relative in triangle_rule]
        vals = integrand(xys)
        relative = np.array([relative for bary, relative in triangle_rule])
        weighted_area = PointJet.broadcast(area, len(triangle_rule))*relative
        return [(weighted_area*value).sum_to_jet() for value in vals]

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
    parts = []
    for root_index, vertices in enumerate(roots):
        root_area = cross2(sub(vertices[1],vertices[0]),sub(vertices[2],vertices[0]))/2
        if 0 < root_area.value <= resolution and not admitted_internal_tie:
            from .enriched_small_cut import integrate_positive_cut
            values, additional, error, certificate = integrate_positive_cut(
                vertices,quad,triangle,polygon,signature,root_index,integrand,
                triangle_rule,tolerance,policy.maximum_triangles-used,cancel)
            used += additional
            difference[:] += error
            parts.append(values)
        else:
            parts.append(integrate(vertices,tolerance))
    out = [sum((part[i] for part in parts), const(0)) for i in range(24)]
    values = np.array([x.value for x in out])
    gradients = np.array([x.gradient for x in out[:8]])
    hessians = np.array([x.hessian for x in out[:8]])
    if not all(np.all(np.isfinite(x)) for x in (values, gradients, hessians)):
        raise EnrichedGeometryError("nonfinite enriched moments")
    if np.any(values[:4] < 0) or (np.any(values[:4] == 0) and not admitted_internal_tie):
        raise EnrichedGeometryError("nonpositive supported Q1 hat")
    return EnrichedMoments(readonly(values[:4]), readonly(values[4:8]),
                           readonly(values[8:].reshape(4, 4)), readonly(gradients[:4]),
                           readonly(gradients[4:8]), readonly(hessians[:4]),
                           readonly(hessians[4:8]), tuple(signature)+(int(admitted_internal_tie),), used,
                           tuple(map(float, difference)))
