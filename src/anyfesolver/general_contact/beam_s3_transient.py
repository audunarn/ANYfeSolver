"""Private complete cubic-time hull bounds for moving B2/B3 and affine S3."""
from fractions import Fraction as Q
from math import sqrt,nextafter,inf
from contextlib import contextmanager
import hashlib
import numpy as np
from .deformable_beam_s3_static import DeformableBeamS3Manager
from .runtime import _ACTIVE_CONTACT

PROFILE='ELASTIC_B2_B3_DEFORMABLE_FIXED_DIRECTOR_S3_NEWMARK_DEV_V1'
RESTART_SCHEMA='ANYSOLVER_BEAM_DEFORMABLE_S3_TRANSIENT_DEV_CHECKPOINT_V17'


def _dot(a,b):return sum(x*y for x,y in zip(a,b))
def _sub(a,b):return [x-y for x,y in zip(a,b)]
def _cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def _norm(a,upper=True):
    square=_dot(a,a)
    if square==0:return Q(0)
    value=sqrt(float(square))
    if not np.isfinite(value) or value==0:raise ValueError('unresolved path norm precision')
    while (Q(value)**2<square if upper else Q(value)**2>square):
        value=nextafter(value,inf if upper else -inf)
    return Q(value)
def _down(value):
    result=float(value)
    if not np.isfinite(result):raise ValueError('unresolved path bound range')
    return nextafter(result,-inf) if Q(result)>value else result


def _binding(manager):
    arrays=(manager.beam_reference,manager.shell.reference_positions,
        manager.work_map.data,manager.work_map.indices,manager.work_map.indptr,
        manager.shell.work_map.data,manager.shell.work_map.indices,manager.shell.work_map.indptr,
        manager.beam_dofs,manager.shell_dofs,manager.kernel.reference_normal)
    return (repr(manager.restart_contract()),tuple(manager.beam_ids),manager.work_map.shape,manager.shell.work_map.shape,tuple((a.shape,a.dtype.str,hashlib.sha256(a.tobytes()).hexdigest()) for a in arrays),
        manager.kernel.count,manager.kernel.length,manager.kernel.radius,
        manager.kernel.stiffness,manager.kernel.orientation,manager.kernel.rim_clearance)


class BeamS3TransientManager(DeformableBeamS3Manager):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._captured_binding=_binding(self)
    def guard(self):
        if _binding(self)!=self._captured_binding:raise ValueError('beam-S3 transient operator mutated')
    def evaluate(self,*args,**kwargs):
        self.guard()
        return super().evaluate(*args,**kwargs)


@contextmanager
def beam_s3_scope(model,definition,settings,initial_displacement=None):
    manager=BeamS3TransientManager(model,definition,settings,initial_displacement)
    token=_ACTIVE_CONTACT.set(manager)
    try:yield manager
    finally:_ACTIVE_CONTACT.reset(token)


def _spatial(beam):
    return beam if len(beam)==2 else [beam[0],[2*b-(a+c)/2 for a,b,c in zip(beam[0],beam[1],beam[2])],beam[2]]


def _unit_anchor(vector):
    low,high=_norm(vector,False),_norm(vector)
    if low<=0:raise ValueError('unresolved path unit direction')
    anchor=[Q(float(x)/float(high)) for x in vector]
    error=sum(max(abs(a-min(x/low,x/high)),abs(a-max(x/low,x/high))) for a,x in zip(anchor,vector))
    return anchor,error


def _box_lower(manager,controls,chart_error):
    nb=len(manager.beam_ids)
    level=controls
    while len(level)>1:
        level=[(a+b)/2 for a,b in zip(level,level[1:])]
    middle=level[0]
    beam=_spatial(middle[:nb]);skin=middle[nb:]
    btime=[_spatial(row[:nb]) for row in controls]
    db=max(_norm(_sub(row[j],beam[j])) for row in btime for j in range(len(beam)))+3*chart_error
    ds=[max(_norm(_sub(row[nb+j],skin[j])) for row in controls)+chart_error for j in range(3)]
    e1,e2=_sub(skin[1],skin[0]),_sub(skin[2],skin[0])
    C=_cross(e1,e2);dc=_norm(e1)*(ds[0]+ds[2])+_norm(e2)*(ds[0]+ds[1])+(ds[0]+ds[1])*(ds[0]+ds[2])
    floor=_norm(C,False)-dc
    reference=[Q(float(x)) for x in manager.kernel.reference_normal]
    if floor<=0 or _dot(C,reference)-dc*_norm(reference)<=0:return None
    normal,anchor_error=_unit_anchor(C)
    dn=2*dc/floor+anchor_error
    physical=[Q(float(manager.kernel.orientation))*x for x in normal]
    U=max(_norm(_sub(p,skin[0])) for p in beam)
    gap=min(_dot(_sub(p,skin[0]),physical) for p in beam)-Q(manager.kernel.radius)-db-ds[0]-U*dn
    for i in range(3):
        edge=_sub(skin[(i+1)%3],skin[i]);delta=ds[i]+ds[(i+1)%3]
        edge_floor=_norm(edge,False)-delta
        if edge_floor<=0:return None
        unit,unit_error=_unit_anchor(edge)
        inward=_cross(normal,unit)
        d_inward=dn+_norm(normal)*(2*delta/edge_floor+unit_error)
        extent=max(_norm(_sub(p,skin[i])) for p in beam)
        rim=min(_dot(_sub(p,skin[i]),inward) for p in beam)-db-ds[i]-extent*d_inward
        if rim<Q(manager.kernel.rim_clearance):return None
    return _down(gap)


def hermite_gap_bounds(manager,start,end,vstart,vend,dt,*,maximum_intervals=512,maximum_depth=20,check_cancel=lambda:None):
    """Complete tensor hull, never a certificate based on hull edges alone."""
    manager.guard()
    if not np.isfinite(dt) or dt<=0:raise ValueError('positive path duration required')
    vectors=[np.asarray(x,float).reshape(-1) for x in (start,end,vstart,vend)]
    if any(x.shape!=(manager.work_map.shape[1],) or not np.isfinite(x).all() for x in vectors):raise ValueError('invalid complete path vector')
    if not getattr(manager,'rotational_descriptor',False) and any(np.any(x[manager.shell_dofs[:,3:]]) for x in vectors):raise ValueError('target path rotations must remain zero')
    a,b,va,vb=([Q(float(x)) for x in v] for v in vectors);h=Q(float(dt))
    nodal=[a,[x+h*y/3 for x,y in zip(a,va)],[x-h*y/3 for x,y in zip(b,vb)],b]
    refs=np.r_[manager.beam_reference.ravel(),manager.shell.reference_positions.ravel()]
    controls=[];magnitude=Q(1)
    for values in nodal:
        row=[]
        for i,ref in enumerate(refs):
            first,last=manager.work_map.indptr[i:i+2]
            terms=[Q(float(manager.work_map.data[j]))*values[manager.work_map.indices[j]] for j in range(first,last)]
            row.append(Q(float(ref))+sum(terms));magnitude=max(magnitude,abs(Q(float(ref)))+sum(abs(x) for x in terms))
        controls.append(np.asarray(row,dtype=object).reshape(-1,3))
    # Enclose standard chart/Hermite evaluation before any cancellation. Fixed
    # admission has <=36 coordinates and <=4 nonzero shape terms per position.
    axis_error=Q(128*np.finfo(float).eps)*magnitude
    chart_error=_norm([axis_error]*3)
    queue=[(controls,0)];visited=0;lower=inf
    while queue:
        check_cancel();box,depth=queue.pop();visited+=1
        if visited>maximum_intervals:return -inf,False,visited
        bound=_box_lower(manager,box,chart_error)
        if bound is not None and bound>=-manager.gap_tolerance:
            lower=min(lower,bound);continue
        if depth>=maximum_depth:return -inf,False,visited
        levels=[box]
        while len(levels[-1])>1:levels.append([(x+y)/2 for x,y in zip(levels[-1],levels[-1][1:])])
        left=[level[0] for level in levels];right=[level[-1] for level in levels[::-1]]
        queue.extend(((right,depth+1),(left,depth+1)))
    return lower,True,visited
