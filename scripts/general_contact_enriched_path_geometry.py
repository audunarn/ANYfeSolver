"""Independent enriched physical contact integration, V14 development reference.

No production contact geometry, moments, derivatives or inverse map are used.
The shared FE structural operator is deliberately outside this module.
"""
from __future__ import annotations
import copy
import hashlib
import numpy as np
from scripts.general_contact_enriched_reference_polynomial import (
    EXPONENTS, control_polynomials, triangle_moments)


class ReferenceRefused(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ReferenceRefused(message)


def finite(value, shape=None):
    a = np.asarray(value, dtype=float)
    require((shape is None or a.shape == shape) and np.all(np.isfinite(a)),
            'finite reference array and exact shape required')
    return a


def cross2(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def unit(v):
    length = np.linalg.norm(v)
    require(np.isfinite(length) and length > 0, 'nondegenerate physical direction')
    return v/length


def q4(r, s):
    r, s = np.asarray(r), np.asarray(s)
    n = np.stack(((1-r)*(1-s), (1+r)*(1-s),
                  (1+r)*(1+s), (1-r)*(1+s)), axis=-1)/4
    nr = np.stack((-(1-s), 1-s, 1+s, -(1+s)), axis=-1)/4
    ns = np.stack((-(1-r), -(1+r), 1+r, 1-r), axis=-1)/4
    return n, nr, ns


def bubbles(r, s):
    r, s = np.asarray(r), np.asarray(s)
    b = (1-r*r)*(1-s*s)
    br, bs = -2*r*(1-s*s), -2*s*(1-r*r)
    a = np.stack((np.ones_like(r), r, s, r*s), axis=-1)
    ar = np.stack((np.zeros_like(r), np.ones_like(r), np.zeros_like(r), s), axis=-1)
    ass = np.stack((np.zeros_like(r), np.zeros_like(r), np.ones_like(r), r), axis=-1)
    return b[..., None]*a, br[..., None]*a+b[..., None]*ar, bs[..., None]*a+b[..., None]*ass


def skin(r, s, corners, beta, normal):
    n, nr, ns = q4(r, s)
    b, br, bs = bubbles(r, s)
    return (n @ corners+(b @ beta)[..., None]*normal,
            nr @ corners+(br @ beta)[..., None]*normal,
            ns @ corners+(bs @ beta)[..., None]*normal, n)


def clip(subject, triangle):
    """Independent scalar half-plane clipping; incidence is retained, not frozen."""
    triangle = np.array(triangle, copy=True)
    if cross2(triangle[1]-triangle[0], triangle[2]-triangle[0]) < 0:
        triangle = triangle[[0, 2, 1]]
    polygon = [p.copy() for p in subject]
    signature = []
    for start, end in zip(triangle, np.roll(triangle, -1, axis=0)):
        if not polygon:
            break
        edge = end-start
        old = polygon[-1]
        od = float(cross2(edge, old-start))
        result = []
        for p in polygon:
            d = float(cross2(edge, p-start))
            signature.append(int(d > 0))
            if (od > 0) != (d > 0):
                require(od != d, 'unresolved clipping denominator')
                result.append(old+(od/(od-d))*(p-old))
            if d > 0:
                result.append(p)
            old, od = p, d
        polygon = result
    return np.asarray(polygon).reshape(-1, 2), tuple(signature)


def inverse_projected(points, corners, beta, normal, frame):
    """Independent batched numeric inversion of the full enriched map."""
    rs = np.zeros((len(points), 2))
    scale = max(float(np.max(np.abs(corners-corners[0]))), np.finfo(float).tiny)
    for _ in range(24):
        p, pr, ps, _ = skin(rs[:, 0], rs[:, 1], corners, beta, normal)
        residual = (p-corners[0]) @ frame-points
        a, b = pr @ frame, ps @ frame
        det = cross2(a, b)
        require(np.all(det > 0), 'folded independent enriched projected map')
        delta = np.column_stack((cross2(residual, b)/det, cross2(a, residual)/det))
        rs -= delta
        if np.max(np.abs(delta)) <= 4*np.finfo(float).eps:
            break
    p, pr, ps, n = skin(rs[:, 0], rs[:, 1], corners, beta, normal)
    error = np.max(np.abs((p-corners[0]) @ frame-points))
    require(error <= 1e-12*scale and np.max(np.abs(rs)) <= 1+1e-10,
            'independent enriched inverse did not converge in reference square')
    return p, pr, ps, n, rs


def _raw_identity(model):
    """Read-only model identity, independent of a contact definition's cache."""
    from anyfesolver.nonlinear_restart import model_configuration_fingerprint
    return model_configuration_fingerprint(model)


class ContactOracle:
    """All25 Q1 pressures on16 physical B4 Q4s and original paired S3 skin."""
    def __init__(self, model, definitions, pose, forces, order, *, cancel=lambda: False,
                 cache_pose_results=False):
        self.model, self.definitions = model, tuple(definitions)
        self.identity = _raw_identity(model)
        self.order, self.cancel = order, cancel
        require(order in (4, 8), 'registered independent quadrature orders4/8 only')
        self.sources = tuple(sorted({d.source_element for d in self.definitions}))
        self.targets = tuple(sorted({d.target_element for d in self.definitions}))
        self.nodal = model.mesh.dof_manager.total_dofs
        self.size = self.nodal+4*len(self.sources)
        self.base_pose = finite(pose, (self.size,)).copy()
        self.z = finite(forces).copy()
        self.base_pose.setflags(write=False); self.z.setflags(write=False)
        self.cache_pose_results = bool(cache_pose_results)
        self._pose_cache = {}
        self.cache_hits = 0
        self.pressure_ids = tuple(sorted({n for e in self.sources for n in model.mesh.elements[e].node_ids}))
        require(self.z.shape == (len(self.pressure_ids),), 'all independent supported force IDs')
        self.mode = {n: i for i, n in enumerate(self.pressure_ids)}
        self.source_data, self.target_data = {}, {}
        self.pair_offsets = {}
        for d in self.definitions:
            self.pair_offsets[d.source_element, d.target_element] = (d.source_offset, d.target_offset)
        require(len(self.pair_offsets) == len(self.definitions)
                and set(self.pair_offsets) == {(s, t) for s in self.sources for t in self.targets},
                'complete nonduplicate physical source/target pair inventory')
        for ordinal, eid in enumerate(self.sources):
            e = model.mesh.elements[eid]
            x = np.array([[model.mesh.nodes[i].x, model.mesh.nodes[i].y, model.mesh.nodes[i].z] for i in e.node_ids])
            require(x.shape == (4, 3), 'independent source must be Q4')
            actual = getattr(e, 'physical_reference_director', None)
            director = unit(np.cross(x[1]-x[0], x[3]-x[0])) if actual is None else finite(actual, (3,))
            require(abs(np.linalg.norm(director)-1) <= 1e-10, 'actual unit physical source director')
            offsets = {self.pair_offsets[eid, t][0] for t in self.targets}
            require(len(offsets) == 1, 'same physical source surface for both targets')
            dofs = np.asarray([model.mesh.dof_manager.get_node_dofs(i) for i in e.node_ids])
            self.source_data[eid] = (x, director, offsets.pop(), dofs,
                np.arange(self.nodal+4*ordinal, self.nodal+4*ordinal+4))
        for eid in self.targets:
            e = model.mesh.elements[eid]
            x = np.array([[model.mesh.nodes[i].x, model.mesh.nodes[i].y, model.mesh.nodes[i].z] for i in e.node_ids])
            require(x.shape == (3, 3), 'independent target must be S3')
            # Physical director comes from the actual element, never integration ordering.
            actual = getattr(e, 'physical_reference_director', None)
            director = unit(np.cross(x[1]-x[0], x[2]-x[0])) if actual is None else finite(actual, (3,))
            require(abs(np.linalg.norm(director)-1) <= 1e-10, 'actual unit physical target director')
            offsets = {self.pair_offsets[s, eid][1] for s in self.sources}
            require(len(offsets) == 1, 'same physical target surface')
            self.target_data[eid] = (x, director, offsets.pop(),
                np.asarray([model.mesh.dof_manager.get_node_dofs(i) for i in e.node_ids]))
        roots, weights = np.polynomial.legendre.leggauss(order)
        t, w = (roots+1)/2, weights/2
        a, b = np.meshgrid(t, t, indexing='ij')
        self.ga, self.gb = a.ravel(), b.ravel()
        self.weights = (w[:, None]*w[None, :]*(1-a)).ravel()
        self.base_sources, self.base_targets = self.vertices(self.base_pose)
        self.base_pairs = {key: self.pair(*key, self.base_sources[key[0]], self.base_targets[key[1]])
                           for key in self.pair_offsets}
        self.base_value = self.accumulate(self.base_sources, self.base_targets, self.base_pairs)
        if self.cache_pose_results:
            self._pose_cache[self.base_pose.tobytes()] = self._cache_snapshot(self.base_value)

    @staticmethod
    def _cache_snapshot(value):
        out = dict(value)
        out['union_certificate'] = copy.deepcopy(value['union_certificate'])
        return out

    def require_exact_reuse(self, model, definitions, pose, forces, order):
        self.verify()
        signature = lambda ds: tuple((d.source_element, d.target_element,
            float(d.source_offset).hex(), float(d.target_offset).hex()) for d in ds)
        require(self.cache_pose_results and model is self.model and order == self.order
                and signature(definitions) == signature(self.definitions)
                and finite(pose, self.base_pose.shape).tobytes() == self.base_pose.tobytes()
                and finite(forces, self.z.shape).tobytes() == self.z.tobytes(),
                'exact model/definition/ALLpose/force/order cache reuse required')

    def verify(self):
        require(_raw_identity(self.model) == self.identity, 'independent reference model mutated')
        self.check_cancel()

    def check_cancel(self):
        if self.cancel():
            from anyfesolver.control import SolveCancelled
            raise SolveCancelled('independent reference cancelled')

    def vertices(self, pose):
        sources, targets = {}, {}
        for eid, (x, director, offset, dofs, bi) in self.source_data.items():
            local = pose[dofs]
            mid = x+local[:, :3]
            corners = mid+offset*(director+np.cross(local[:, 3:], director))
            body = unit(np.cross(mid[1]-mid[0], mid[3]-mid[0]))
            sources[eid] = (corners, pose[bi], body)
        for eid, (x, director, offset, dofs) in self.target_data.items():
            local = pose[dofs]
            targets[eid] = x+local[:, :3]+offset*(director+np.cross(local[:, 3:], director))
        return sources, targets

    def pair(self, sid, tid, source, target):
        # Fingerprint/immutable-model admission is paired at EVERY full pose
        # entry/accumulation/exit, rather than redundantly hashing each pair.
        self.check_cancel()
        corners, beta, body = source
        tangent = unit(corners[1]-corners[0])
        normal = unit(np.cross(corners[1]-corners[0], corners[3]-corners[0]))
        frame = np.column_stack((tangent, np.cross(normal, tangent)))
        q = (corners-corners[0]) @ frame
        t = (target-corners[0]) @ frame
        polygon, signature = clip(q, t)
        points, weights, triangles = [], [], []
        for i in range(1, len(polygon)-1):
            a, b, c = polygon[[0, i, i+1]]
            jac = abs(float(cross2(b-a, c-a)))
            if jac == 0:
                continue  # Scalar zero-measure integration only; derivatives are whole-union FD.
            points.append(a+self.ga[:, None]*(b-a)+(1-self.ga[:, None])*self.gb[:, None]*(c-a))
            weights.append(jac*self.weights)
            triangles.append((np.asarray((a, b, c)), jac/2))
        if not points:
            return np.zeros(4), np.zeros(4), signature, 0
        points, weights = np.concatenate(points), np.concatenate(weights)
        # The SAME deterministic polynomial is constructed at EVERY pose,
        # independently of literal order4/8; no geometric chain is frozen.
        chart_r, chart_s = (q[1]-q[0])/2, (q[3]-q[0])/2
        center = q[0]+chart_r+chart_s
        chart_det = float(cross2(chart_r, chart_s))
        require(np.isfinite(chart_det) and chart_det > 0, 'positive current affine control chart')
        def affine_coordinates(xy):
            delta = xy-center
            return np.column_stack((cross2(delta, chart_s)/chart_det,
                                    cross2(chart_r, delta)/chart_det))
        x, director, offset, _, _ = self.source_data[sid]
        reference = x+offset*director
        reference_corner_jac = float(np.linalg.norm(np.cross(reference[1]-reference[0], reference[3]-reference[0]))/4)
        pm, pb = control_polynomials(corners.tolist(), beta.tolist(), body.tolist(), normal.tolist(),
            target.tolist(), t.tolist(), center.tolist(), chart_r.tolist(), chart_s.tolist(), reference_corner_jac)
        coefficients = np.array([[poly.get(key, 0.) for key in EXPONENTS] for poly in (*pm, *pb)])
        moments = np.zeros(len(EXPONENTS))
        for triangle, area in triangles:
            all_moments = triangle_moments(affine_coordinates(triangle).tolist(), area)
            moments += np.asarray([all_moments[key] for key in EXPONENTS])
        analytic = coefficients @ moments
        affine = affine_coordinates(points)
        monomials = np.array([affine[:, 0]**i*affine[:, 1]**j for i, j in EXPONENTS])
        polynomial_values = (coefficients @ monomials).T
        p, pr, ps, n, rs = inverse_projected(points, corners, beta, body, frame)
        a, b = t[1]-t[0], t[2]-t[0]
        det = float(cross2(a, b))
        require(det != 0, 'nondegenerate independent projected S3')
        delta = points-t[0]
        b1, b2 = cross2(delta, b)/det, cross2(a, delta)/det
        target_shape = np.column_stack((1-b1-b2, b1, b2))
        physical_normal = np.cross(pr, ps)
        projected = physical_normal @ normal
        require(np.all(projected > 0), 'positive enriched projection Jacobian')
        physical_normal /= np.linalg.norm(physical_normal, axis=1)[:, None]
        gap = np.sum(physical_normal*(p-target_shape @ target), axis=1)
        _, rr, ss = q4(rs[:, 0], rs[:, 1])
        reference_jac = np.linalg.norm(np.cross(rr @ reference, ss @ reference), axis=1)
        full_support = (reference_jac/projected)[:, None]*n
        full_weighted = full_support*gap[:, None]
        # Exact algebraic decomposition I(F)=I(P)+I(F-P); F retains the FULL
        # inverse/varying normal/target/domain/reference-Jacobian computation.
        support = analytic[:4]+np.sum(weights[:, None]*(full_support-polynomial_values[:, :4]), axis=0)
        weighted = analytic[4:]+np.sum(weights[:, None]*(full_weighted-polynomial_values[:, 4:]), axis=0)
        require(np.all(np.isfinite(support)) and np.all(np.isfinite(weighted)), 'finite independent moments')
        return support, weighted, signature, len(points)

    def accumulate(self, sources, targets, pairs):
        self.verify()
        support, weighted = np.zeros(len(self.z)), np.zeros(len(self.z))
        signatures, points = [], 0
        for (sid, tid), (m, b, signature, count) in pairs.items():
            indices = [self.mode[n] for n in self.model.mesh.elements[sid].node_ids]
            support[indices] += m
            weighted[indices] += b
            signatures.append((sid, tid, signature)); points += count
        certificate = union_certificate(self, sources, targets)
        require(np.all(support > 0), 'ALL supported independent source modes positive')
        gap = weighted/support
        for array in (support, weighted, gap):
            array.setflags(write=False)
        result = dict(work=float(-self.z @ gap), support=support, weighted_gap=weighted,
                    gap=gap, signature=tuple(signatures), quadrature_points=points,
                    integration_method='ANALYTIC_PI8_PLUS_LITERAL_GAUSS_REMAINDER_V15',
                    union_certificate=certificate)
        self.verify()
        return result

    def evaluate(self, pose):
        self.verify()
        pose = finite(pose, (self.size,))
        key = pose.tobytes()
        if self.cache_pose_results and key in self._pose_cache:
            self.cache_hits += 1
            value = self._cache_snapshot(self._pose_cache[key])
            self.verify()
            return value
        if self.cache_pose_results:
            require(len(self._pose_cache) < 1+2*self.size, 'bounded one238 central FD sweep cache')
        sources, targets = self.vertices(pose)
        dirty_s = {i for i, v in sources.items() if any(not np.array_equal(a, b) for a, b in zip(v, self.base_sources[i]))}
        dirty_t = {i for i, v in targets.items() if not np.array_equal(v, self.base_targets[i])}
        pairs = dict(self.base_pairs)
        for s, t in pairs:
            if s in dirty_s or t in dirty_t:
                pairs[s, t] = self.pair(s, t, sources[s], targets[t])
        value = self.accumulate(sources, targets, pairs)
        self.verify()
        if self.cache_pose_results:
            self._pose_cache[key] = self._cache_snapshot(value)
        return value


def union_certificate(oracle,sources,targets):
    """Independent node-star/disk embedding and complete physical field check.

    No production topology/clip/AD/contact helper is imported. The common
    simplex topology theorem is checked from original element node ordering.
    """
    require(len(targets)>=2,'actual multi-triangle target required')
    ids={t:tuple(oracle.model.mesh.elements[t].node_ids) for t in targets}
    vertices,variations={},{}
    edges={};face_edges={}
    for tid,face in targets.items():
        reference,director,offset,dofs=oracle.target_data[tid]
        for j,node in enumerate(ids[tid]):
            signature=(reference[j].tobytes(),(offset*director).tobytes(),tuple(dofs[j]))
            if node in vertices:
                require(np.array_equal(vertices[node],face[j]) and variations[node]==signature,'independent physical/virtual-work trace mismatch')
            vertices[node]=face[j];variations[node]=signature
        for j,a in enumerate(ids[tid]):
            b=ids[tid][(j+1)%3];key=tuple(sorted((a,b)))
            edges.setdefault(key,[]).append((tid,a,b))
    require(all(len(v)in(1,2) for v in edges.values()),'independent nonmanifold target')
    require(all(v[0][1:]==v[1][1:][::-1] for v in edges.values() if len(v)==2),'independent inconsistent target orientation')
    boundary=[v[0][1:] for v in edges.values() if len(v)==1]
    successor=dict(boundary)
    require(len(successor)==len(boundary) and set(successor)=={b for a,b in boundary},'independent branched exterior')
    require(len(vertices)-len(edges)+len(targets)==1,'independent target Euler disk')
    cycle=[];node=min(successor)
    while node not in cycle: cycle.append(node);node=successor[node]
    require(node==cycle[0] and len(cycle)==len(boundary),'independent exterior cycle')
    adjacent={tid:set() for tid in targets}
    for rows in edges.values():
        if len(rows)==2:
            a,b=rows[0][0],rows[1][0];adjacent[a].add(b);adjacent[b].add(a)
    seen=set();pending=[next(iter(targets))]
    while pending:
        tid=pending.pop()
        if tid not in seen: seen.add(tid);pending.extend(adjacent[tid]-seen)
    require(len(seen)==len(targets),'independent disconnected target disk')
    for node in vertices:
        links={}
        for face in ids.values():
            if node in face:
                a,b=[x for x in face if x!=node]
                links.setdefault(a,set()).add(b);links.setdefault(b,set()).add(a)
        counts=sorted(map(len,links.values()))
        require(all(c in(1,2) for c in counts) and counts.count(1)==(2 if node in successor else 0),'independent nonmanifold vertex')
        seen=set();pending=[next(iter(links))]
        while pending:
            x=pending.pop()
            if x not in seen:seen.add(x);pending.extend(links[x]-seen)
        require(len(seen)==len(links),'independent disconnected vertex star')
    def disjoint(a,b,c,d,eps):
        areas=(cross2(b-a,c-a),cross2(b-a,d-a),cross2(d-c,a-c),cross2(d-c,b-c))
        if min(areas[:2])<-eps and max(areas[:2])>eps and min(areas[2:])<-eps and max(areas[2:])>eps:return False
        def distance(p,a,b):
            v=b-a;fraction=np.clip((p-a) @ v/(v @ v),0.,1.)
            return np.linalg.norm(p-a-fraction*v)
        return min(distance(a,c,d),distance(b,c,d),distance(c,a,b),distance(d,a,b))>eps/max(np.linalg.norm(b-a),np.linalg.norm(d-c))
    def winding(point,poly):
        count=0
        for a,b in zip(poly,np.roll(poly,-1,axis=0)):
            if a[1]<=point[1]<b[1] and cross2(b-a,point-a)>0:count+=1
            if b[1]<=point[1]<a[1] and cross2(b-a,point-a)<0:count-=1
        return count
    certificates=[]
    for sid,(q,beta,body) in sources.items():
        tangent=unit(q[1]-q[0]);normal=unit(np.cross(q[1]-q[0],q[3]-q[0]))
        frame=np.column_stack((tangent,np.cross(normal,tangent)))
        projected={n:(p-q[0]) @ frame for n,p in vertices.items()}
        poly=np.array([projected[n] for n in cycle]);source=(q-q[0]) @ frame
        scale=max(1.,float(np.max(np.abs(q))),*(float(np.max(np.abs(p))) for p in vertices.values()))
        eps=512*np.finfo(float).eps*scale*scale
        require(all(cross2(projected[f[1]]-projected[f[0]],projected[f[2]]-projected[f[0]])>eps for f in ids.values()),'independent folded target projection')
        for i in range(len(poly)):
            for j in range(i+1,len(poly)):
                if j==i+1 or i==0 and j==len(poly)-1:continue
                require(disjoint(poly[i],poly[(i+1)%len(poly)],poly[j],poly[(j+1)%len(poly)],eps),'independent exterior self-intersection')
        require(all(winding(p,poly)==1 for p in source),'independent source footprint outside target')
        for i in range(4):
            for a,b in zip(poly,np.roll(poly,-1,axis=0)):
                require(disjoint(source[i],source[(i+1)%4],a,b,eps),'independent source/target exterior event')
        require(all(cross2(source[(i+1)%4]-source[i],source[(i+2)%4]-source[(i+1)%4])>eps for i in range(4)),'independent source boundary fold')
        dets,ars,ass=[],[],[]
        for r,s in ((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)):
            _,nr,ns=q4(r,s);a,b=nr @ source,ns @ source
            dets.append(cross2(a,b));ars.append(np.linalg.norm(a));ass.append(np.linalg.norm(b))
        bound=min(dets)-np.linalg.norm(body @ frame)*2*np.sum(np.abs(beta))*(max(ars)+max(ass))
        require(bound>eps,'independent whole-cell source map positivity unproved')
        for (a,b),rows in edges.items():
            if len(rows)==2:
                v=projected[b]-projected[a];distance=[cross2(v,p-projected[a]) for p in source]
                require(not any(abs(distance[i])<=eps and abs(distance[(i+1)%4])<=eps for i in range(4)),'independent internal/source-edge collinearity unsupported')
        certificates.append(dict(source=sid,projected_jacobian_lower=float(bound)))
    return dict(admitted=True,target_faces=len(targets),target_vertices=len(vertices),boundary=list(cycle),
        internal_edges=[list(e) for e,v in edges.items() if len(v)==2],cells=certificates,
        scope='independent conforming PL disk/globalembedding/coverage FIRST variation, not a smooth Hessian theorem')


class PhysicalField(ContactOracle):
    """Independent original-node physical field WITHOUT contact integration."""
    def __init__(self,model,definitions,pose,*,cancel=lambda:False):
        order=8;cache_pose_results=False
        forces=np.zeros(len({n for d in definitions for n in model.mesh.elements[d.source_element].node_ids}))
        self.model, self.definitions = model, tuple(definitions)
        self.identity = _raw_identity(model)
        self.order, self.cancel = order, cancel
        require(order in (4, 8), 'registered independent quadrature orders4/8 only')
        self.sources = tuple(sorted({d.source_element for d in self.definitions}))
        self.targets = tuple(sorted({d.target_element for d in self.definitions}))
        self.nodal = model.mesh.dof_manager.total_dofs
        self.size = self.nodal+4*len(self.sources)
        self.base_pose = finite(pose, (self.size,)).copy()
        self.z = finite(forces).copy()
        self.base_pose.setflags(write=False); self.z.setflags(write=False)
        self.cache_pose_results = bool(cache_pose_results)
        self._pose_cache = {}
        self.cache_hits = 0
        self.pressure_ids = tuple(sorted({n for e in self.sources for n in model.mesh.elements[e].node_ids}))
        require(self.z.shape == (len(self.pressure_ids),), 'all independent supported force IDs')
        self.mode = {n: i for i, n in enumerate(self.pressure_ids)}
        self.source_data, self.target_data = {}, {}
        self.pair_offsets = {}
        for d in self.definitions:
            self.pair_offsets[d.source_element, d.target_element] = (d.source_offset, d.target_offset)
        require(len(self.pair_offsets) == len(self.definitions)
                and set(self.pair_offsets) == {(s, t) for s in self.sources for t in self.targets},
                'complete nonduplicate physical source/target pair inventory')
        for ordinal, eid in enumerate(self.sources):
            e = model.mesh.elements[eid]
            x = np.array([[model.mesh.nodes[i].x, model.mesh.nodes[i].y, model.mesh.nodes[i].z] for i in e.node_ids])
            require(x.shape == (4, 3), 'independent source must be Q4')
            actual = getattr(e, 'physical_reference_director', None)
            director = unit(np.cross(x[1]-x[0], x[3]-x[0])) if actual is None else finite(actual, (3,))
            require(abs(np.linalg.norm(director)-1) <= 1e-10, 'actual unit physical source director')
            offsets = {self.pair_offsets[eid, t][0] for t in self.targets}
            require(len(offsets) == 1, 'same physical source surface for both targets')
            dofs = np.asarray([model.mesh.dof_manager.get_node_dofs(i) for i in e.node_ids])
            self.source_data[eid] = (x, director, offsets.pop(), dofs,
                np.arange(self.nodal+4*ordinal, self.nodal+4*ordinal+4))
        for eid in self.targets:
            e = model.mesh.elements[eid]
            x = np.array([[model.mesh.nodes[i].x, model.mesh.nodes[i].y, model.mesh.nodes[i].z] for i in e.node_ids])
            require(x.shape == (3, 3), 'independent target must be S3')
            # Physical director comes from the actual element, never integration ordering.
            actual = getattr(e, 'physical_reference_director', None)
            director = unit(np.cross(x[1]-x[0], x[2]-x[0])) if actual is None else finite(actual, (3,))
            require(abs(np.linalg.norm(director)-1) <= 1e-10, 'actual unit physical target director')
            offsets = {self.pair_offsets[s, eid][1] for s in self.sources}
            require(len(offsets) == 1, 'same physical target surface')
            self.target_data[eid] = (x, director, offsets.pop(),
                np.asarray([model.mesh.dof_manager.get_node_dofs(i) for i in e.node_ids]))
