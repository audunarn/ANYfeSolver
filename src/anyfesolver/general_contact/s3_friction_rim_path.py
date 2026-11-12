"""Sufficient rounded fixed-axis planar rim-path geometry; no history claim."""
from fractions import Fraction as F
import numpy as np
from .s3_friction_coverage import mapped_path,cross,vsub,positive
from .s3_friction_moving_coverage import _raw_normal,_dot
from .s3_friction_mesh_coverage import _safe

def require_fixed_plane_path(template,first,last):
    first=np.asarray(first,float);last=np.asarray(last,float)
    if first.shape!=(template.n_dof,) or last.shape!=first.shape or not np.isfinite(first).all() or not np.isfinite(last).all():raise ValueError("invalid rim path endpoints")
    cap=np.finfo(float).max**.125
    if max(np.max(abs(first)),np.max(abs(last)))>cap:raise ValueError("rim path arithmetic range")
    sources=[mapped_path(a,first,last)[0] for a in template.source_adapters]
    targets={e:mapped_path(a,first,last)[0] for e,a in template.target_adapters.items()}
    reference=next(iter(template.target_adapters.values())).reference_positions
    raw=np.cross(reference[1]-reference[0],reference[2]-reference[0])
    axes=np.flatnonzero(raw!=0.)
    if len(axes)!=1 or raw[axes[0]]<=0:raise ValueError('rim replay needs positive fixed axis target plane')
    axis=int(axes[0]);level=F(float(reference[0,axis]))
    slide_axis=min(j for j in range(3) if j!=axis)
    fixed_tangent=next(j for j in range(3) if j not in (axis,slide_axis))
    for adapter in (*template.source_adapters,*template.target_adapters.values()):
        face=mapped_path(adapter,first,last)[0]
        for point,ref in zip(face,adapter.reference_positions):
            value=F(float(ref[fixed_tangent]));polynomial=point[fixed_tangent]
            if polynomial[0]!=(value,value) or any(v!=(F(0),F(0)) for v in polynomial[1:]):raise ValueError('rim replay first row requires fixed second tangential coordinate')
    def fixed(p):return p[0]==(level,level) and all(v==(F(0),F(0)) for v in p[1:])
    all_nodes={}
    for e,face in targets.items():
        adapter=template.target_adapters[e]
        if any(not fixed(point[axis]) for point in face):raise ValueError('rim target plane changes or is not exactly mapped')
        normal=_raw_normal(face)
        if not _safe(normal) or not positive(normal[axis]):raise ValueError('rim target facet path orientation unresolved')
        for node,point in zip(adapter.node_ids,face):
            if node in all_nodes and all_nodes[node]!=point:raise ValueError('rim target shared material map differs')
            all_nodes[node]=point
    # Every nonincident boundary vertex lies strictly inside each edge. This
    # proves a convex/simple union rather than relying on consecutive turns.
    edges=[]
    for e,a in template.target_adapters.items():
        for j,node in enumerate(a.node_ids):
            end=a.node_ids[(j+1)%3]
            if tuple(sorted((node,end))) in template.boundary_edges:edges.append((node,end))
    boundary={node for edge in edges for node in edge}
    for node,end in edges:
        edge=vsub(all_nodes[end],all_nodes[node])
        for other in boundary-{node,end}:
            if not positive(cross(edge,vsub(all_nodes[other],all_nodes[node]))[axis]):raise ValueError('rim target convex path unresolved')
    # Equal perpendicular rows and equal binary endpoint coordinates imply
    # equal rounded coordinates on the actual chord. Do not infer this from
    # approximately planar mapped endpoints or positive normal alignment.
    for adapter,face in zip(template.source_adapters,sources):
        refs=adapter.reference_positions[:,axis]
        if not np.all(refs==refs[0]):raise ValueError('rim source reference normals must be parallel to target')
        values=[]
        for j in range(3):
            row=adapter.work_map.getrow(3*j+axis)
            nonzero=row.data!=0
            if np.count_nonzero(nonzero)!=1 or row.data[nonzero][0]!=1.:raise ValueError('rim source perpendicular map must be a translation selector')
            col=int(row.indices[nonzero][0]);values.append((first[col],last[col]))
        if any(v!=values[0] for v in values[1:]):raise ValueError('rim source/target normals must remain parallel over the actual chord')
        normal=_raw_normal(face)
        if not _safe(normal) or not positive(normal[axis]):raise ValueError('rim source path polarity/alignment unresolved')
    from .s3_friction_mesh_gap import pair_lower
    from .s3_friction_dyadic_interval import outward_dyadic_160
    with outward_dyadic_160():
        lower=min(pair_lower(s,t) for s in sources for t in targets.values())
    if lower < -template.normal.gap_tolerance:raise ValueError('rim original rounded-map source-ray gap budget unresolved')
    return True
