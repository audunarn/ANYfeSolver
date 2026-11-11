"""Versioned oriented target-disk and physical-trace admission, V16.

No legacy paired guard is changed. A positive oriented PL disk with a simple
projected boundary has degree one, hence no overlapping triangle interiors.
The certificate is rebuilt at every pose before internal-edge ties are used.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .enriched_overlap import EnrichedGeometryError, shape


def require(ok, message):
    if not ok:
        raise EnrichedGeometryError(message)


def disk_topology(faces):
    """Pure node/edge combinatorics; no geometry or mechanical computation."""
    faces = tuple(tuple(f) for f in faces)
    require(faces and all(len(f) == 3 and len(set(f)) == 3 for f in faces), 'triangular target faces required')
    require(len({frozenset(f) for f in faces}) == len(faces), 'duplicate target face')
    edges, links = defaultdict(list), defaultdict(list)
    for index, face in enumerate(faces):
        for j, node in enumerate(face):
            a, b = face[(j+1)%3], face[(j+2)%3]
            links[node].append((a, b))
            edges[tuple(sorted((node, a)))].append((index, node, a))
    require(all(len(v) in (1, 2) for v in edges.values()), 'nonmanifold target edge')
    for incident in edges.values():
        if len(incident) == 2:
            require(incident[0][1:] == incident[1][1:][::-1], 'target edge orientations disagree')
    boundary = [incident[0][1:] for incident in edges.values() if len(incident) == 1]
    successor = {a: b for a, b in boundary}
    require(len(successor) == len(boundary) and set(successor) == {b for a,b in boundary}, 'branched target boundary')
    require(bool(successor), 'closed target surface is not a disk')
    cycle, node = [], min(successor)
    while node not in cycle:
        cycle.append(node); node = successor[node]
    require(node == cycle[0] and len(cycle) == len(boundary), 'multiple target boundary loops')
    vertices = set(links)
    require(len(vertices)-len(edges)+len(faces) == 1, 'target Euler characteristic is not a disk')
    adjacency = defaultdict(set)
    for incident in edges.values():
        if len(incident) == 2:
            a,b = incident[0][0],incident[1][0]
            adjacency[a].add(b);adjacency[b].add(a)
    visited, pending = set(), [0]
    while pending:
        i = pending.pop()
        if i not in visited: visited.add(i);pending.extend(adjacency[i]-visited)
    require(len(visited) == len(faces), 'disconnected target faces')
    for vertex, segments in links.items():
        graph = defaultdict(set)
        for a,b in segments: graph[a].add(b);graph[b].add(a)
        degrees = sorted(len(v) for v in graph.values())
        require(all(d in (1,2) for d in degrees), 'nonmanifold target vertex link')
        require(degrees.count(1) == (2 if vertex in successor else 0), 'target vertex link is not path/cycle')
        seen,pending = set(),[next(iter(graph))]
        while pending:
            a=pending.pop()
            if a not in seen: seen.add(a);pending.extend(graph[a]-seen)
        require(len(seen) == len(graph), 'disconnected target vertex star')
    return dict(boundary=tuple(cycle), internal_edges=tuple(sorted(e for e,v in edges.items() if len(v)==2)),
                face_edges=tuple(tuple(tuple(sorted((f[j],f[(j+1)%3]))) for j in range(3)) for f in faces))


def cross(a,b): return float(a[0]*b[1]-a[1]*b[0])


def segment_distance(p,a,b):
    v=b-a;t=float(np.clip((p-a) @ v/(v @ v),0.,1.))
    return float(np.linalg.norm(p-(a+t*v)))


def segments_disjoint(a,b,c,d,resolution):
    # Every nonincident boundary edge pair must be separated, including
    # collinear/touching cases; no epsilon relabeling as an internal tie.
    values=(cross(b-a,c-a),cross(b-a,d-a),cross(d-c,a-c),cross(d-c,b-c))
    if min(values[:2]) < -resolution and max(values[:2]) > resolution and min(values[2:]) < -resolution and max(values[2:]) > resolution:
        return False
    length=max(float(np.linalg.norm(b-a)),float(np.linalg.norm(d-c)),np.finfo(float).tiny)
    return min(segment_distance(a,c,d),segment_distance(b,c,d),segment_distance(c,a,b),segment_distance(d,a,b)) > resolution/length


def inside(p, polygon):
    winding=0
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        if a[1] <= p[1] < b[1] and cross(b-a,p-a)>0: winding+=1
        elif b[1] <= p[1] < a[1] and cross(b-a,p-a)<0: winding-=1
    return winding == 1


def certify_disk(pairs, maps):
    """Physical point AND complete affine virtual-work traces at each node."""
    from .enriched_nonlinear import AffinePhysicalPair,_AFFINE_MAP_HESSIAN
    require(len(pairs)==len(maps) and len(pairs)>=2, 'complete target disk pair maps required')
    require(all(type(p) is AffinePhysicalPair for p in pairs), 'exact protected affine physical-map ABI required')
    first,x0=pairs[0],maps[0][0]
    q=x0[:12].reshape(4,3);mid=x0[25:37].reshape(4,3)
    tangent=q[1]-q[0];tangent=tangent/np.linalg.norm(tangent)
    normal=np.cross(q[1]-q[0],q[3]-q[0]);normal=normal/np.linalg.norm(normal)
    frame=np.column_stack((tangent,np.cross(normal,tangent)))
    vertices, signatures, faces = {},{},[]
    orientation=None
    for pair,mapping in zip(pairs,maps):
        x=mapping[0]
        require(pair.definition.source_element==first.definition.source_element
            and pair.definition.source_offset==first.definition.source_offset
            and pair.mode_indices==first.mode_indices
            and pair.indices[:24]==first.indices[:24] and pair.indices[42:]==first.indices[42:]
            and np.array_equal(pair.reference_corners[:4],first.reference_corners[:4])
            and np.array_equal(pair.reference_directors[:4],first.reference_directors[:4])
            and np.array_equal(pair.reference_source,first.reference_source)
            and np.array_equal(x[:12],x0[:12]) and np.array_equal(x[21:],x0[21:]), 'nonconforming source physical/work maps')
        require(mapping[1].shape==(37,46) and mapping[2] is _AFFINE_MAP_HESSIAN, 'protected affine physical-map layout required')
        face=list(pair.target_node_ids)
        for j,node in enumerate(face):
            point=x[12+3*j:15+3*j]
            signature=(pair.reference_corners[4+j].tobytes(),pair.reference_directors[4+j].tobytes(),
                float(pair.definition.target_offset).hex(),tuple(pair.indices[24+6*j:30+6*j]))
            if node in vertices:
                require(np.array_equal(vertices[node],point) and signatures[node]==signature, 'nonconforming shared physical point/variation trace')
            vertices[node]=point;signatures[node]=signature
        triangle=np.array([(vertices[n]-q[0]) @ frame for n in face])
        signed=cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
        require(np.isfinite(signed) and signed!=0, 'collapsed target projected face')
        if orientation is None: orientation=1 if signed>0 else -1
        require(signed*orientation>0, 'target disk contains a folded face')
        if orientation<0: face.reverse()  # ONE numerical chart, never physical side
        faces.append(tuple(face))
    topology=disk_topology(faces)
    projected={n:(p-q[0]) @ frame for n,p in vertices.items()}
    scale=max(1.,float(np.max(np.abs(q))),*(float(np.max(np.abs(p))) for p in vertices.values()))
    resolution=512*np.finfo(float).eps*scale*scale
    for face in faces:
        a,b,c=(projected[n] for n in face)
        require(cross(b-a,c-a)>resolution, 'target projected orientation unresolved')
    polygon=np.array([projected[n] for n in topology['boundary']])
    require(sum(cross(a,b) for a,b in zip(polygon,np.roll(polygon,-1,axis=0)))>resolution, 'target boundary orientation')
    for i in range(len(polygon)):
        for j in range(i+1,len(polygon)):
            if j==i+1 or i==0 and j==len(polygon)-1: continue
            require(segments_disjoint(polygon[i],polygon[(i+1)%len(polygon)],polygon[j],polygon[(j+1)%len(polygon)],resolution), 'target projected boundary not simple/separated')
    source=(q-q[0]) @ frame
    require(all(cross(source[(i+1)%4]-source[i],source[(i+2)%4]-source[(i+1)%4])>resolution for i in range(4)), 'nonconvex source boundary')
    require(all(inside(p,polygon) for p in source), 'source outside target disk')
    clearance=min(segment_distance(p,a,b) for p in source for a,b in zip(polygon,np.roll(polygon,-1,axis=0)))
    require(clearance>512*np.finfo(float).eps*scale, 'source footprint lacks exterior clearance')
    for i in range(4):
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            require(segments_disjoint(source[i],source[(i+1)%4],a,b,resolution), 'source boundary crosses target exterior')
    drs,dss,dets=[],[],[]
    for r in (-1.,1.):
        for s in (-1.,1.):
            _,nr,ns=shape(r,s);dr=np.asarray(nr) @ source;ds=np.asarray(ns) @ source
            drs.append(np.linalg.norm(dr));dss.append(np.linalg.norm(ds));dets.append(cross(dr,ds))
    body=np.cross(mid[1]-mid[0],mid[3]-mid[0]);body/=np.linalg.norm(body)
    perturb=float(np.linalg.norm(body @ frame))*2*float(np.sum(np.abs(x0[21:25])))*(max(drs)+max(dss))
    lower=min(dets)-perturb
    require(np.isfinite(lower) and lower>resolution, 'whole-source projected injectivity unproved')
    for a,b in topology['internal_edges']:
        direction=projected[b]-projected[a]
        distance=[cross(direction,p-projected[a]) for p in source]
        require(not any(abs(distance[i])<=resolution and abs(distance[(i+1)%4])<=resolution for i in range(4)), 'source-edge/internal-edge collinearity unsupported')
    internal=set(topology['internal_edges'])
    return tuple(dict(internal_edges=tuple(e for e in edges if e in internal),
        projected_jacobian_lower_bound=float(lower),exterior_clearance_m=float(clearance),
        oriented_faces=len(faces),target_vertices=len(vertices),boundary_nodes=len(polygon),
        first_variation_internal_flux='IDENTICAL_PHYSICAL_AND_VIRTUAL_WORK_TRACE',
        tie_matrix='SELECTED_PIECEWISE_NEWTON_NOT_SMOOTH_HESSIAN') for edges in topology['face_edges'])
