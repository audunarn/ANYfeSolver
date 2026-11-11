"""Restricted paired-C0-union selected derivative branch, development only.

The smooth-pair integrand/quadrature is copied exactly from protected V2.
Only certified INTERNAL shared-edge clipping ties are admitted. Exterior,
projection, nonconforming or folded branches remain typed refusals. A
selected piecewise Newton matrix is an internal model, NOT a Hessian proof.
"""
from __future__ import annotations
import numpy as np
from .enriched_overlap import (Jet2,sqrt,SIZE,const,dot,sub,cross,cross2,unit,
    shape,interpolate,physical,readonly,EnrichedGeometryError,EnrichedMoments,
    SolveCancelled)

def selected_pair_moments(reference_source, current_source, current_target, beta, direction,
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
        if area.value < 0 or (area.value <= resolution and not admitted_internal_tie):
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
    if np.any(values[:4] < 0) or (np.any(values[:4] == 0) and not admitted_internal_tie):
        raise EnrichedGeometryError("nonpositive supported Q1 hat")
    return EnrichedMoments(readonly(values[:4]), readonly(values[4:8]),
                           readonly(values[8:].reshape(4, 4)), readonly(gradients[:4]),
                           readonly(gradients[4:8]), readonly(hessians[:4]),
                           readonly(hessians[4:8]), tuple(signature)+(int(admitted_internal_tie),), used,
                           tuple(map(float, difference)))


def certify_union(first, second, first_map, second_map):
    """Finite whole-domain certificate BEFORE any internal tie is admitted.

    The shared physical edge has identical point AND variation maps. The
    source-side hat, normal and reference/projected Jacobian are independent
    of target choice. Consequently BOTH integrand traces (M,b,Gram) agree
    for every point on that moving edge, so internal first-variation fluxes
    cancel. This does not assert smooth second variation at a vertex tie.

    B vanishes on the whole source boundary. For the interior, the projected
    map is Q1+v*b with constant v and b=sum(beta*B). The determinant lemma
    cancels the quadratic rank-one term. |dB_a/dr|,|dB_a/ds|<=2 for all four
    modes, hence the explicit bound below proves strictly positive Jacobian
    over the ENTIRE square, not a sampled determinant test. Strictly convex
    unchanged boundary plus positive interior Jacobian gives a one-to-one
    planar source map. Convex target boundary and opposite shared-edge sides
    certify a simple, nonoverlapping two-triangle union.
    """
    if (first.definition.source_element!=second.definition.source_element
            or first.definition.source_offset!=second.definition.source_offset
            or first.definition.target_offset!=second.definition.target_offset
            or not np.array_equal(first.reference_corners[:4],second.reference_corners[:4])
            or not np.array_equal(first.reference_directors[:4],second.reference_directors[:4])
            or first.mode_indices!=second.mode_indices):
        raise EnrichedGeometryError('nonconforming paired source/offset/pressure definition')
    ids1,ids2=first.target_node_ids,second.target_node_ids
    shared=frozenset(ids1)&frozenset(ids2)
    if len(shared)!=2 or len(set(ids1)|set(ids2))!=4:
        raise EnrichedGeometryError('exact two-triangle target topology required')
    x1,x2=first_map[0],second_map[0]
    if not np.array_equal(x1[:12],x2[:12]) or not np.array_equal(x1[21:],x2[21:]):
        raise EnrichedGeometryError('paired physical source maps differ')
    vertices={}
    for pair,x in ((first,x1),(second,x2)):
        for j,node in enumerate(pair.target_node_ids):
            value=x[12+3*j:15+3*j]
            signature=(pair.reference_corners[4+j].tobytes(),
                       pair.reference_directors[4+j].tobytes(),
                       pair.indices[24+6*j:30+6*j])
            if node in vertices:
                old,old_signature=vertices[node]
                if not np.array_equal(old,value) or old_signature!=signature:
                    raise EnrichedGeometryError('shared physical edge/work maps are not identical')
            else:vertices[node]=(value,signature)
    q=x1[:12].reshape(4,3);mid=x1[25:37].reshape(4,3)
    tangent=q[1]-q[0];tangent/=np.linalg.norm(tangent)
    normal=np.cross(q[1]-q[0],q[3]-q[0]);normal/=np.linalg.norm(normal)
    second_axis=np.cross(normal,tangent)
    frame=np.column_stack((tangent,second_axis))
    projected={node:(v[0]-q[0]) @ frame for node,v in vertices.items()}
    source=(q-q[0]) @ frame
    scale=max(1.,float(np.max(np.abs(q))),float(np.max(np.abs(source))),
              max(float(np.max(np.abs(v[0])))for v in vertices.values()),
              max(float(np.max(np.abs(v)))for v in projected.values()))
    resolution=512*np.finfo(float).eps*scale*scale
    def area(a,b):return float(a[0]*b[1]-a[1]*b[0])
    edges={}
    for ids in (ids1,ids2):
        for j in range(3):
            edge=frozenset((ids[j],ids[(j+1)%3]));edges[edge]=edges.get(edge,0)+1
    boundary=[tuple(e)for e,count in edges.items()if count==1]
    if len(boundary)!=4 or edges.get(shared)!=2:
        raise EnrichedGeometryError('target union boundary is not simple')
    adjacency={i:[]for i in vertices}
    for a,b in boundary:adjacency[a].append(b);adjacency[b].append(a)
    if any(len(v)!=2 for v in adjacency.values()):
        raise EnrichedGeometryError('target boundary is not one four-edge cycle')
    cycle=[min(vertices)]
    while len(cycle)<4:
        options=[i for i in adjacency[cycle[-1]]if i not in cycle]
        if not options:raise EnrichedGeometryError('target boundary cycle is incomplete')
        cycle.append(min(options))
    if cycle[0]not in adjacency[cycle[-1]]:
        raise EnrichedGeometryError('target boundary is not closed')
    polygon=np.array([projected[i]for i in cycle])
    if sum(area(polygon[i],polygon[(i+1)%4])for i in range(4))<0:
        cycle.reverse();polygon=polygon[::-1]
    turns=[area(polygon[(i+1)%4]-polygon[i],polygon[(i+2)%4]-polygon[(i+1)%4])for i in range(4)]
    if min(turns)<=resolution:
        raise EnrichedGeometryError('target union must be strictly convex/nonfolded')
    a,b=sorted(shared);edge=projected[b]-projected[a]
    distances=[area(edge,p-projected[a])for p in source]
    if any(abs(distances[i])<=resolution and abs(distances[(i+1)%4])<=resolution for i in range(4)):
        raise EnrichedGeometryError('source-edge/shared-edge collinearity is unsupported')
    third=[next(i for i in ids if i not in shared)for ids in (ids1,ids2)]
    if area(edge,projected[third[0]]-projected[a])*area(edge,projected[third[1]]-projected[a])>=0:
        raise EnrichedGeometryError('target triangles overlap or collapse')
    clearance=min(area(polygon[(i+1)%4]-polygon[i],p-polygon[i])/
        np.linalg.norm(polygon[(i+1)%4]-polygon[i])for i in range(4)for p in source)
    if clearance<=512*np.finfo(float).eps*scale:
        raise EnrichedGeometryError('source footprint lacks strict exterior clearance')
    ar=[];ass=[];det=[]
    for r in (-1.,1.):
        for s in (-1.,1.):
            _,nr,ns=shape(r,s)
            dr=np.asarray(nr) @ source;ds=np.asarray(ns) @ source
            ar.append(float(np.linalg.norm(dr)));ass.append(float(np.linalg.norm(ds)));det.append(area(dr,ds))
    body=np.cross(mid[1]-mid[0],mid[3]-mid[0]);body/=np.linalg.norm(body)
    perturb=float(np.linalg.norm(body @ frame))*2*float(np.sum(np.abs(x1[21:25])))*(max(ar)+max(ass))
    lower=min(det)-perturb
    if not np.isfinite(lower) or lower<=resolution:
        raise EnrichedGeometryError('whole-cell enriched projection positivity is unproved')
    return dict(shared_nodes=tuple(sorted(shared)),exterior_clearance_m=float(clearance),
                projected_jacobian_lower_bound=float(lower),rank_one_bound=float(perturb),
                first_variation_internal_flux='IDENTICAL_PHYSICAL_TRACE',
                tie_matrix='SELECTED_PIECEWISE_NEWTON_MATRIX_NOT_HESSIAN_CERTIFICATE')
