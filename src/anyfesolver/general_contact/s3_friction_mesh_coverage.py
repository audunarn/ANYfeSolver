"""Rounded whole-chord union certificate for a conforming planar S3 disk.

Every facet has positive oriented area, the exterior is strictly convex and
every source corner/dense point lies in both projection domains. Coplanarity
is enclosed on the entire original affine coordinate path. Dyadic restriction
tightens interval dependency; it is not a sampled-position test.
"""
from fractions import Fraction as F
from math import comb
import numpy as np
from .s3_friction_coverage import add, cross, mapped_path, material_point_paths, magnitude, positive, vsub
from .s3_friction_moving_coverage import _raw_normal, _dot, _widen
from . import s3_friction_chart_interval as I


def restrict(p, a, b):
    out=[]
    for j in range(len(p)):
        weights=[F(comb(i,j))*a**(i-j)*(b-a)**j for i in range(j,len(p))]
        out.append((sum(p[i][0]*w for i,w in zip(range(j,len(p)),weights)),
                    sum(p[i][1]*w for i,w in zip(range(j,len(p)),weights))))
    return tuple(out)


def _safe(raw):
    minimum=F(float(np.finfo(float).tiny**.25))
    # Subtract a positive floor (not an uncertainty widening).
    def pos(p):return positive(((p[0][0]-minimum,p[0][1]-minimum),*p[1:]))
    return (any(pos(p) or pos(tuple((-hi,-lo) for lo,hi in p)) for p in raw)
            and max(map(magnitude,raw))<F(float(np.finfo(float).max**.25)))


def _sum_normals(facets):
    total=tuple(((F(0),F(0)),)*3 for _ in range(3))
    u=F(1,2**53);floor=F(1,2**1074)
    for face in facets:
        raw=_raw_normal(face)
        total=tuple(_widen(add(a,b),u*(magnitude(a)+magnitude(b))/(1-u)+floor)
                    for a,b in zip(total,raw))
    return total


def _chart_check(source, facets, boundary, dense):
    # Preserve affine-map correlations BEFORE boxing. Rounded chart normal,
    # tangent and projected edge arithmetic are still enclosed explicitly.
    u=F(1,2**53);floor=F(1,2**1074);gamma=16*u/(1-16*u)
    def rounded_edge(p,q):
        return tuple(_widen(d,u*(magnitude(a)+magnitude(b))/(1-u)+floor)
                     for d,a,b in zip(vsub(p,q),p,q))
    for chart_index,chart in enumerate((source,facets[0])):
        edge=rounded_edge(chart[1],chart[0])
        raw=_raw_normal(chart) if chart_index==0 else _sum_normals(facets)
        n=I.normalize(tuple(I.box(p) for p in raw))
        tangent=I.normalize(tuple(I.box(p) for p in edge));binormal=I.cross(n,tangent)
        axes=(tangent,binormal)
        def relative(p,q):
            delta=rounded_edge(p,q)
            result=[]
            for axis in axes:
                value=I.dot(tuple(I.box(v) for v in delta),axis)
                # xy_j-xy_i in the clipper differs from projecting the
                # physical edge by the two rounded origin-subtractions/dots
                # and final xy subtraction. gamma16 bounds these operations;
                # this term also covers source-origin exact-zero handling.
                error=gamma*sum((magnitude(a)+magnitude(b)+2*magnitude(o))*max(map(abs,t))
                                for a,b,o,t in zip(p,q,chart[0],axis))+32*floor*sum(max(map(abs,t)) for t in axis)
                result.append((value[0]-error,value[1]+error))
            return result
        projected=[relative(p,chart[0]) for p in (*source,*boundary)]
        scale=2*max(sum(max(map(abs,c)) for c in p) for p in projected)
        threshold=I.mul(I.mul((F(1e-10),F(1e-10)),(scale,scale)),(scale,scale))[1]
        if any(I.cross2(relative(f[1],f[0]),relative(f[2],f[0]))[0]<=2*threshold for f in [source,*facets]):return False
        for j,p in enumerate(boundary):
            edge=relative(boundary[(j+1)%len(boundary)],p)
            # All nonincident exterior vertices establish global strict
            # convexity/simple boundary, not merely positive local turns.
            others=[q for k,q in enumerate(boundary) if k not in (j,(j+1)%len(boundary))]
            if any(I.cross2(edge,relative(q,p))[0]<=threshold for q in [*others,*source,*dense]):return False
    return True


def mesh_moving_coverage(owner, first, last, *, physical_paths=None):
    first=np.asarray(first,float);last=np.asarray(last,float)
    if first.shape!=(owner.n_dof,) or last.shape!=first.shape:return False
    arrays=[first,last,owner.point_reference,owner.point_maps]
    adapters=(*owner.source_adapters,*owner.target_adapters.values())
    for a in adapters:arrays.extend((a.reference_positions,a.work_map.data))
    if any(not np.isfinite(a).all() or np.max(np.abs(a),initial=0)>np.finfo(float).max**.125 for a in arrays):return False
    if physical_paths is None:
        target={e:mapped_path(a,first,last)[0] for e,a in owner.target_adapters.items()}
        source=[mapped_path(a,first,last)[0] for a in owner.source_adapters]
        dense=material_point_paths(owner,first,last)
    else:
        source,target,dense=physical_paths
    vertices={};edges={}
    for e,a in owner.target_adapters.items():
        for node,p in zip(a.node_ids,target[e]):
            if node in vertices and p!=vertices[node]:return False
            vertices[node]=p
        for j,node in enumerate(a.node_ids):
            edge=(node,a.node_ids[(j+1)%3]);edges.setdefault(tuple(sorted(edge)),[]).append(edge)
    if any(len(v)>2 or (len(v)==2 and v[0]!=v[1][::-1]) for v in edges.values()):return False
    boundary=[v[0] for v in edges.values() if len(v)==1];following=dict(boundary)
    if not boundary or len(following)!=len(boundary) or len(vertices)-len(edges)+len(target)!=1:return False
    loop=[];node=boundary[0][0]
    while node not in loop and node in following:loop.append(node);node=following[node]
    if node!=loop[0] or len(loop)!=len(boundary):return False
    # A finite sufficient checker: try 1,2,4,8,16 exact dyadic restrictions.
    for slices in (1,2,4,8,16):
        success=True
        for k in range(slices):
            a,b=F(k,slices),F(k+1,slices)
            rv=lambda v:tuple(restrict(p,a,b) for p in v)
            facets=[tuple(rv(v) for v in f) for f in target.values()]
            sources=[tuple(rv(v) for v in f) for f in source]
            points=[rv(v) for v in dense];rim=[rv(vertices[n]) for n in loop]
            raw=_sum_normals(facets)
            if not _safe(raw):success=False;break
            origin=facets[0][0]
            # Lower distance scale and raw-normal norm use a signed component.
            lower=lambda p:max(F(0),I.box(p)[0],-I.box(p)[1])
            norm_lower=max(map(lower,raw))
            scale_lower=max(lower(p) for face in facets for v in face for p in vsub(v,origin))
            tol=F(owner.friction_settings.geometry_relative_tolerance)*norm_lower*scale_lower/2
            u=F(1,2**53);floor=F(1,2**1074);gamma=8*u/(1-8*u)
            def plane_bound(v):
                delta=vsub(v,origin)
                # Rounded subtraction, normalized components and 3-term dot;
                # multiplied by the positive norm. The factor-two scale lower
                # retains slack for the rounded Euclidean scale computation.
                error=sum((gamma*magnitude(d)+u*(magnitude(p)+magnitude(o))/(1-u)+8*floor)*magnitude(n)
                          for d,p,o,n in zip(delta,v,origin,raw))+8*floor*sum(map(magnitude,raw))
                return magnitude(_dot(delta,raw))+error
            if tol<=0 or any(plane_bound(v)>tol for face in facets for v in face):success=False;break
            reference=[np.cross(ad.reference_positions[1]-ad.reference_positions[0],ad.reference_positions[2]-ad.reference_positions[0]) for ad in owner.target_adapters.values()]
            from .s3_friction_coverage import dot_axis
            if any(not positive(dot_axis(_raw_normal(f),tuple(F(float(x)) for x in ref))) or not positive(_dot(_raw_normal(f),raw)) for f,ref in zip(facets,reference)):success=False;break
            for f in sources:
                sn=_raw_normal(f)
                try: chart_ok=_chart_check(f,facets,rim,points)
                except ValueError: chart_ok=False
                if not _safe(sn) or not positive(_dot(sn,raw)) or not chart_ok:success=False;break
            if not success:break
        if success:return True
    return False
