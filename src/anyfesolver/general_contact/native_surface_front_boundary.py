"""Development angular partition at captured-time stationary-front transitions.

At fixed time, N=A'B-AB' is exactly a constant plus sine and cosine of the
material angle for the admitted straight round geometry. Regular bands here
exclude only endpoint N zeros. They do not certify interior temporal branches
or admit friction integration. Every uncertain band remains represented.
"""
from dataclasses import dataclass
from math import acos,atan2,ceil,cos,floor,pi,sin
import numpy as np

from .native_surface_front import axial_contact_front,stationary_front_slab
from .native_surface_path import material_path_pose
from .native_surface_support import SupportIntervalBudget,_support_path,_base


@dataclass(frozen=True)
class StationaryEndpointAngularPartition:
    field_identity: str
    cell: int
    angle_bounds: tuple
    endpoint_polynomials: tuple
    guards: tuple
    bands: tuple
    intervals_used: int


def endpoint_numerator_range(partition,field,cell,index,time,angle_bounds):
    """Analytic range over a sub-band of the bound endpoint partition."""
    if (type(partition) is not StationaryEndpointAngularPartition or
        partition.field_identity!=field.identity or partition.cell!=cell):
        raise ValueError('bound endpoint angular partition required')
    lower,upper=map(float,angle_bounds)
    if not partition.angle_bounds[0]<=lower<=upper<=partition.angle_bounds[1]:
        raise ValueError('endpoint range outside bound angular partition')
    entries=[entry for entry in partition.endpoint_polynomials if entry[:2]==(index,time)]
    if len(entries)!=1:raise ValueError('unique captured endpoint polynomial required')
    _,_,(q0,qc,qs),error=entries[0];phase=atan2(qs,qc)
    first=floor((lower-phase)/pi);last=ceil((upper-phase)/pi)
    candidates=[lower,upper,*(phase+k*pi for k in range(first,last+1) if lower<=phase+k*pi<=upper)]
    values=[q0+qc*cos(angle)+qs*sin(angle) for angle in candidates]
    error+=1024*np.finfo(float).eps*(abs(q0)+np.hypot(qc,qs))
    return (float(np.nextafter(min(values)-error,-np.inf)),
            float(np.nextafter(max(values)+error,np.inf)))


def _endpoint_polynomial(field,cell,index,time,angle):
    """Direct geometric coefficients, without angular sampling/interpolation."""
    value=axial_contact_front(field,cell,index,time,angle)
    start,segments=_support_path(field,index);segment=segments[0]
    pose=material_path_pose(start,segment.translations,segment.angular,segment.shell_increment,time)
    target=field.patch.target;corners=target.adapter.current_positions(pose.shell_displacement)
    velocities=np.asarray(target.adapter.work_map@segment.shell_increment).reshape(3,3)
    e1,e2=corners[1]-corners[0],corners[2]-corners[0]
    v1,v2=velocities[1]-velocities[0],velocities[2]-velocities[0]
    c=np.cross(e1,e2);cd=np.cross(v1,e2)+np.cross(e1,v2)
    frame=pose.cell_rotations[cell]@field.patch.source.reference.nodal_triads[0]
    base=_base(pose,cell,0.,corners[0]);t=1. if cell==0 else 0.
    base_rate=(1-t)*segment.translations[cell]+t*segment.translations[cell+1]-velocities[0]
    radius=field.patch.rule.definition['radius'];omega=segment.angular[cell]
    arms=(radius*frame[:,1],radius*frame[:,2])
    a=(float(c@base),*(float(c@arm) for arm in arms))
    ap=(float(cd@base+c@base_rate),*(float(cd@arm+c@np.cross(omega,arm)) for arm in arms))
    coefficients=tuple(p*value.b-x*value.b_rate for x,p in zip(a,ap))
    if not np.all(np.isfinite(coefficients)):raise ValueError('nonfinite endpoint angular polynomial')
    return coefficients


def stationary_angular_coefficients(field,cell,index,time,angle):
    """Exact sinusoidal A, A_t and A_tt coefficients at one path time.

    These are coefficients of the declared round surface, not fitted samples.
    B has no material-angle dependence. Callers supply rounding bounds before
    using the resulting stationary numerator and its time derivative.
    """
    value=axial_contact_front(field,cell,index,time,angle)
    start,segments=_support_path(field,index);segment=segments[0]
    pose=material_path_pose(start,segment.translations,segment.angular,segment.shell_increment,time)
    corners=field.patch.target.adapter.current_positions(pose.shell_displacement)
    velocities=np.asarray(field.patch.target.adapter.work_map@segment.shell_increment).reshape(3,3)
    e1,e2=corners[1]-corners[0],corners[2]-corners[0]
    v1,v2=velocities[1]-velocities[0],velocities[2]-velocities[0]
    c=np.cross(e1,e2);cd=np.cross(v1,e2)+np.cross(e1,v2);cdd=2*np.cross(v1,v2)
    frame=pose.cell_rotations[cell]@field.patch.source.reference.nodal_triads[0]
    base=_base(pose,cell,0.,corners[0]);t=1. if cell==0 else 0.
    velocity=(1-t)*segment.translations[cell]+t*segment.translations[cell+1]-velocities[0]
    radius=field.patch.rule.definition['radius'];omega=segment.angular[cell]
    arms=(radius*frame[:,1],radius*frame[:,2])
    a=(float(c@base),*(float(c@arm) for arm in arms))
    ap=(float(cd@base+c@velocity),*(float(cd@arm+c@np.cross(omega,arm)) for arm in arms))
    app=(float(cdd@base+2*cd@velocity),
         *(float(cdd@arm+2*cd@np.cross(omega,arm)+c@np.cross(omega,np.cross(omega,arm))) for arm in arms))
    if not np.all(np.isfinite((a,ap,app))):raise ValueError('nonfinite stationary angular coefficients')
    return a,ap,app,value


def _polynomial_guards(coefficients,error,lower,upper):
    """Enclose all zeros under uniform function error, including tangencies."""
    q0,qc,qs=coefficients;amplitude=float(np.hypot(qc,qs))
    rounding=1024*np.finfo(float).eps*(abs(q0)+amplitude)
    error=float(np.nextafter(error+rounding,np.inf))
    if abs(q0)>amplitude+error:return ()
    # Degenerate/tangent cases are kept as a whole unresolved band. No root
    # is manufactured, and no potentially contributing material is removed.
    if amplitude<=error or abs(q0)+error>=amplitude:return ((lower,upper),)
    slope=float(np.sqrt((amplitude-abs(q0)-error)*(amplitude+abs(q0)+error)))
    if not np.isfinite(slope) or slope<=0:return ((lower,upper),)
    phase=atan2(qs,qc);alpha=acos(-q0/amplitude)
    padding=1024*np.finfo(float).eps*max(1.,abs(lower),abs(upper))
    # |Q'| >= slope wherever |Q| <= error. The mean-value theorem encloses
    # each perturbed zero about a computed zero; the factor2 also leaves a
    # strict sign margin beyond that set. Angle-rounding padding is separate.
    width=float(np.nextafter(2*error/slope+padding,np.inf));guards=[]
    for offset in (phase-alpha,phase+alpha):
        first=floor((lower-width-offset)/(2*pi));last=ceil((upper+width-offset)/(2*pi))
        for k in range(first,last+1):
            root=offset+2*pi*k
            left=max(lower,float(np.nextafter(root-width,-np.inf)))
            right=min(upper,float(np.nextafter(root+width,np.inf)))
            if left<right:guards.append((left,right))
    return tuple(sorted(guards))


def stationary_endpoint_angular_partition(field,cell,angle_bounds,*,budget):
    """Bound all captured-time endpoint transitions, with complete band coverage.

    Each endpoint consumes one shared support interval. The six-coefficient
    propagation supplies a uniform N evaluation error before cancellation.
    Each constant/cosine/sine coefficient is bounded by that control; their
    combined evaluation uses three times that bound. These are padded binary64
    development controls, not a formal arithmetic proof or admission policy.
    """
    from anyfesolver._ge_beam3_p5_seeded.core import sha
    if type(budget) is not SupportIntervalBudget:raise TypeError('shared support budget required')
    if type(cell) is not int or cell not in (0,1):raise ValueError('one native half-cell required')
    if sha(field._definition())!=field.identity:raise ValueError('native material history field changed')
    lower,upper=map(float,angle_bounds);domain=field.patch.rule.definition['angle_range']
    if not np.isfinite(lower+upper) or not domain[0]<=lower<upper<=domain[1]:
        raise ValueError('endpoint angular partition outside material selection')
    middle=(lower+upper)/2;before=budget.intervals;polynomials=[];guards=[]
    for index in range(len(field.segments)):
        for time in (0.,1.):
            if not budget.consume():raise MemoryError('endpoint angular partition interval cap exhausted')
            coefficients=_endpoint_polynomial(field,cell,index,time,middle)
            _,(_,nerror)=stationary_front_slab(field,cell,index,middle,time,time,
                angle_bounds=(lower,upper),_return_error_controls=True)
            error=float(np.nextafter(3*nerror,np.inf))
            polynomials.append((index,time,coefficients,error))
            guards.extend((left,right,index,time) for left,right in
                _polynomial_guards(coefficients,error,lower,upper))
    edges=sorted({lower,upper,*(v for left,right,_,_ in guards for v in (left,right))})
    bands=[]
    for left,right in zip(edges[:-1],edges[1:]):
        owners=tuple((index,time) for a,b,index,time in guards if a<=left and right<=b)
        bands.append((left,right,'unresolved' if owners else 'regular',owners))
    if not bands or bands[0][0]!=lower or bands[-1][1]!=upper or any(
            a[1]!=b[0] for a,b in zip(bands,bands[1:])):
        raise ValueError('incomplete endpoint angular partition')
    return StationaryEndpointAngularPartition(field.identity,cell,(lower,upper),
        tuple(polynomials),tuple(sorted(guards)),tuple(bands),budget.intervals-before)
