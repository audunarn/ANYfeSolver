"""Whole curved patch admission controls for fixed-plane development work.

These are sufficient conservative controls, independent of contact quadrature.
An unresolved orientation bound is not a proof of physical inversion. Current
plane-gap enclosures do not certify a moving path or a collision time.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import sqrt

import numpy as np

from anyfesolver._ge_beam3_p5.arrays import _array
from .capped_cylinder import _sqrt_bound
from .native_surface_bounds import _reference_radial_gain, _rotation_gain, _restrict_q2
from .native_retained_station import _readonly


@dataclass(frozen=True)
class CurvedPatchGeometryControl:
    orientation_lower: float
    cell_orientation_lower: tuple
    gap_lower_m: float
    gap_upper_m: float
    search_box: np.ndarray


def _reference_curvature_coefficients(reference):
    """Exact binary Q2 curvature, matching the centered reference lift."""
    F=lambda v:Fraction(float(v))
    return tuple(F(reference.coordinates[0,j])-2*F(reference.coordinates[1,j])+
                 F(reference.coordinates[2,j]) for j in range(3))


def _chord_difference_norm(high,low,rotation,reference,cell):
    """Outward norm of exact represented chord minus rotated reference chord."""
    F=lambda v:Fraction(float(v))
    cref=[F(reference.coordinates[cell+1,j])-F(reference.coordinates[cell,j]) for j in range(3)]
    difference=[F(high[cell+1,i])+F(low[cell+1,i])-F(high[cell,i])-F(low[cell,i])-
        sum(F(rotation[i,j])*cref[j] for j in range(3)) for i in range(3)]
    return float(_sqrt_bound(sum(v*v for v in difference),upper=True))


def _reference_curvature_norm(reference):
    b=_reference_curvature_coefficients(reference)
    return float(_sqrt_bound(sum(v*v for v in b),upper=True))


def _reference_norm_bounds(reference, cell, *, actual_roll=False):
    """Affine Q2 tangent norm and frame-derivative norm bounds.

    Normalized affine tangent has |n'|<=|b|/min|v|. Shortest transport
    T=I+K+K²/d has |T'|<=A*N1+2*A²*N1/d+A³*N1/d².
    Roll derivative is bounded by5*N1+2*roll_rate, with the formulation's
    frozen roll_rate<0.9*pi. The Q2 tangent remains in the endpoint cone;
    nodal tangent misalignment and arithmetic reduce the denominator bound.
    """
    F=lambda v:Fraction(float(v))
    X=[[F(v) for v in row] for row in reference.coordinates]
    a=[(X[2][j]-X[0][j])/2 for j in range(3)]
    b=[X[0][j]-2*X[1][j]+X[2][j] for j in range(3)]
    left,right=(Fraction(-1),Fraction(0)) if cell==0 else (Fraction(0),Fraction(1))
    bb=sum(v*v for v in b)
    ab=sum(v*w for v,w in zip(a,b))
    minimum=left if bb==0 else max(left,min(right,-ab/bb))
    square=lambda xi:sum((v+xi*w)**2 for v,w in zip(a,b))
    jmin=float(_sqrt_bound(square(minimum)))
    jmax=float(_sqrt_bound(max(square(left),square(right)),upper=True))
    B=float(_sqrt_bound(bb,upper=True))
    eps=np.finfo(float).eps
    scale=max(1.,jmax,B)
    jmin-=128*eps*scale
    jmax+=128*eps*scale;B+=128*eps*scale
    if jmin<=0. or not np.all(np.isfinite((jmin,jmax,B))):
        raise ValueError('curved material reference derivative bound unresolved')
    nodes=reference.nodal_triads
    k0=reference.tangent(float(left));k1=reference.tangent(float(right))
    A=_rotation_gain(nodes[cell])
    mismatch=float(np.linalg.norm(nodes[cell,:,0]-k0))+128*eps
    # Compensated affine evaluation still rounds products. Near a small
    # Jacobian, its angular uncertainty must reduce the whole-cell cone bound.
    tangent_error=32*eps*scale/jmin
    denominator=1+float(k0@k1)-mismatch-8*tangent_error-256*eps
    if denominator<=1+np.cos(.9*np.pi):
        raise ValueError('curved material frame interval denominator unresolved')
    N1=B/jmin
    T=1+A+A*A/denominator
    T1=A*N1+2*A*A*N1/denominator+A**3*N1/denominator**2
    # The work support control may use the formulation-owned constant roll
    # rate; the historical whole-patch control retains its frozen worst case.
    roll=abs(float(reference._half_frame_data[cell][1])) if actual_roll else .9*np.pi
    if not np.isfinite(roll) or (actual_roll and roll>=.9*np.pi):
        raise ValueError('curved material reference roll bound unresolved')
    # At exactly zero owned roll, its axis rotation is identically I and its
    # derivative is zero. Only shortest transport contributes; no fitted bound.
    roll_derivative=0. if actual_roll and roll==0. else 5*N1+2*roll
    prime=(roll_derivative*T+(1+128*eps)*T1)*A
    prime*=1+4096*eps/denominator**2
    gram=max(float(np.linalg.norm(f.T@f-np.eye(3))) for f in nodes)
    gram+=128*eps*(1+max(float(np.linalg.norm(f))**2 for f in nodes))
    # Transported first column error, including clipped dot products and roll.
    first_error=64*(gram+128*eps)/denominator**2+8*tangent_error
    return jmin,jmax,prime,first_error


def _reference_second_bound(reference,cell):
    """Second derivative of the owned smooth half-cell frame formula.

    n=v/|v| for affine v has |n''|<=3*(B/jmin)^2. Twice differentiate
    T=I+K+K²/d and the constant-roll Rodrigues rotation, then their product.
    Root isolation separately certifies no transport clipping switch and owns
    nodal one-sided reconstruction uncertainty; this is not a global C2 claim.
    """
    jmin,jmax,_,_=_reference_norm_bounds(reference,cell,actual_roll=True)
    eps=np.finfo(float).eps
    curvature=_reference_curvature_norm(reference)
    B=curvature+128*eps*max(1.,jmax,curvature)
    N1=B/jmin;N2=3*N1*N1
    left,right=(-1.,0.) if cell==0 else (0.,1.)
    A=_rotation_gain(reference.nodal_triads[cell])
    k0=reference.tangent(left);k1=reference.tangent(right)
    mismatch=float(np.linalg.norm(reference.nodal_triads[cell,:,0]-k0))+128*eps
    tangent_error=32*eps*max(1.,jmax,B)/jmin
    d=1+float(k0@k1)-mismatch-8*tangent_error-256*eps
    if d<=1+np.cos(.9*np.pi):
        raise ValueError('curved material second derivative denominator unresolved')
    T=1+A+A*A/d
    T1=A*N1+2*A*A*N1/d+A**3*N1/d**2
    T2=(A*N2+2*A*A*(N2+N1*N1)/d+
        A**3*(4*N1*N1+N2)/d**2+2*A**4*N1*N1/d**3)
    roll=abs(float(reference._half_frame_data[cell][1]))
    Q1=0. if roll==0. else 5*N1+2*roll
    Q2=0. if roll==0. else 2*roll*roll+6*roll*N1+5*N2+4*N1*N1
    value=(Q2*T+2*Q1*T1+(1+128*eps)*T2)*A
    value*=1+8192*eps/d**3
    if not np.isfinite(value):
        raise ValueError('curved material second derivative bound overflow')
    return float(value)


def _reference_plane_derivative_bounds(reference,cell,rotation,normal):
    """Projected smooth-frame bounds, eligible only on an unclipped interval.

    For exactly zero owned roll, the affine tangent and its derivatives lie
    in one plane. Projecting shortest transport before taking norms retains
    its binormal cancellation and the actual stored tangent misalignment.
    Nonzero roll or an unresolved plane uses the existing full-frame bounds.
    Returned bounds exclude radius and include the represented current map.
    """
    F=lambda v:Fraction(float(v))
    jmin,jmax,F1,_=_reference_norm_bounds(reference,cell,actual_roll=True)
    F2=_reference_second_bound(reference,cell)
    q=[sum(F(rotation[i,j])*F(normal[i]) for i in range(3)) for j in range(3)]
    qnorm=float(_sqrt_bound(sum(v*v for v in q),upper=True))
    fallback=(qnorm*F1,qnorm*F2)
    if float(reference._half_frame_data[cell][1])!=0.:return fallback
    a=[(F(reference.coordinates[2,j])-F(reference.coordinates[0,j]))/2 for j in range(3)]
    b=_reference_curvature_coefficients(reference)
    cross=lambda u,v:[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
    plane=cross(a,b);pnorm=float(_sqrt_bound(sum(v*v for v in plane)))
    if pnorm<=0.:return fallback
    l=[F(v) for v in reference.nodal_triads[cell,:,0]]
    A=_rotation_gain(reference.nodal_triads[cell])
    projected=float(_sqrt_bound(sum(v*v for v in cross(q,plane)),upper=True))
    D=float(np.nextafter(float((F(A)*F(projected)+
        abs(sum(v*w for v,w in zip(l,plane)))*F(qnorm))/F(pnorm)),np.inf))
    eps=np.finfo(float).eps;curvature=_reference_curvature_norm(reference)
    B=curvature+128*eps*max(1.,jmax,curvature);N1=B/jmin;N2=3*N1*N1
    left,right=(-1.,0.) if cell==0 else (0.,1.)
    k0=reference.tangent(left);k1=reference.tangent(right)
    mismatch=float(np.linalg.norm(reference.nodal_triads[cell,:,0]-k0))+128*eps
    tangent_error=32*eps*max(1.,jmax,B)/jmin
    d=1+float(k0@k1)-mismatch-8*tangent_error-256*eps
    first=A*D*N1*(1+2*A/d+A*A/d**2)
    second=A*D*(N2+2*A*(N2+N1*N1)/d+
        A*A*(4*N1*N1+N2)/d**2+2*A**3*N1*N1/d**3)
    # Same denominator/normalization representation allowances as the full
    # frame controls; cancellation never removes arithmetic uncertainty.
    pad=8192*eps*qnorm*(1+A)**5*(1+N1+N2)*(1+jmax/jmin)/d**3
    if not np.all(np.isfinite((first,second,pad))):return fallback
    return tuple(min(old,float(np.nextafter(value+pad,np.inf)))
                 for old,value in zip(fallback,(first,second)))


def _reference_unclipped_interval(reference,cell,left,right):
    """Exact affine-tangent test for the upper clipping branch, with margin."""
    F=lambda v:Fraction(float(v))
    X=[[F(v) for v in row] for row in reference.coordinates]
    a=[(X[2][j]-X[0][j])/2 for j in range(3)]
    b=[X[0][j]-2*X[1][j]+X[2][j] for j in range(3)]
    if not any(b):return True  # Constant tangent: no clip branch can change.
    m=[F(v) for v in reference.nodal_triads[cell,:,0]]
    aa=sum(v*v for v in a);ab=sum(v*w for v,w in zip(a,b));bb=sum(v*v for v in b)
    ma=sum(v*w for v,w in zip(m,a));mb=sum(v*w for v,w in zip(m,b))
    lo,hi=F(left),F(right);points=[lo,hi]
    slope=mb*ab-ma*bb
    if slope:
        critical=(ma*ab-mb*aa)/slope
        if lo<critical<hi:points.append(critical)
    for xi in points:
        numerator=ma+xi*mb;square=aa+2*xi*ab+xi*xi*bb
        if numerator>0 and square-numerator*numerator<=F(1024*np.finfo(float).eps)*square:
            return False
    return True


def _orientation_cell_lower(reference,cell,radius,gain,EU,difference,coordinate_scale,Fgain):
    """Sufficient signed area/radius² bound for one complete material cell."""
    jmin,jmax,prime,first_error=_reference_norm_bounds(reference,cell)
    eps=np.finfo(float).eps
    E=4*((gain*Fgain)**2-1)+1024*eps
    if E>=.1 or not np.isfinite(E):
        raise ValueError('curved material orientation frame bound unresolved')
    error=2*E+gain*first_error
    padding=8192*eps*max(1.,jmax,difference,radius*prime,coordinate_scale)
    return (jmin*(1-EU-128*eps)-difference*gain-
            (gain*jmax+difference)*error-
            radius*gain*prime*(gain*Fgain)**2-padding)


def curved_patch_geometry_control(rule, positions, position_low, cell_rotations,
                                  origin, normal):
    """Require positive oriented material area throughout all cells/arcs.

    Let W=(Uc*R0 e2) cross (Uc*R0 e3), independent of arc angle.
    Signed area/r² is (c+Uc*L'+Uc*R0'*rho).dot(W).
    c+Uc*L'=Uc*J*n + (c-Uc*c_ref). A near-rotation Gram bound
    encloses W-Uc*n; no sampled positivity establishes this test.
    """
    origin=_array(origin,(3,),'curved material plane origin')
    normal=_array(normal,(3,),'curved material plane normal')
    if abs(np.linalg.norm(normal)-1.)>1e-12:
        raise ValueError('unit curved material plane normal required')
    bounds=rule.current_bounds(positions,position_low,cell_rotations)
    reference=rule.samples[0].station.reference
    radius=float(rule.definition['radius'])
    high=np.asarray(positions);low=np.asarray(position_low);U=np.asarray(cell_rotations)
    Fgain=_reference_radial_gain(reference)
    cell_bounds=[]
    for cell in bounds.cells:
        gain=_rotation_gain(U[cell])
        eps=np.finfo(float).eps
        EU=float(np.linalg.norm(U[cell].T@U[cell]-np.eye(3)))+512*eps
        difference=_chord_difference_norm(high,low,U[cell],reference,cell)
        lower=_orientation_cell_lower(reference,cell,radius,gain,EU,difference,
            float(np.max(np.abs(reference.coordinates))),Fgain)
        if not np.isfinite(lower) or lower<=0.:
            raise ValueError('curved material whole-patch orientation unresolved')
        cell_bounds.append((cell,float(lower)))
    # Direct oriented projection avoids axis-box inflation under rigid motion.
    F=lambda v:Fraction(float(v))
    X=[[F(v) for v in row] for row in reference.coordinates]
    b=[X[0][j]-2*X[1][j]+X[2][j] for j in range(3)]
    p=[[F(h)+F(l) for h,l in zip(hr,lr)] for hr,lr in zip(high,low)]
    lo,hi=rule.definition['xi_range'];slabs=[]
    for cell,left,right in ((0,max(lo,-1.),min(hi,0.)),(1,max(lo,0.),min(hi,1.))):
        if right<=left:continue
        turn=[sum(F(U[cell,i,j])*b[j] for j in range(3)) for i in range(3)]
        controls=[p[cell],[(p[cell][i]+p[cell+1][i])/2-turn[i]/4 for i in range(3)],p[cell+1]]
        lane=tuple(sum((point[i]-F(origin[i]))*F(normal[i]) for i in range(3)) for point in controls)
        l=F(left)+(1 if cell==0 else 0);r=F(right)+(1 if cell==0 else 0)
        restricted=_restrict_q2((lane,),l,r)[0]
        extent=F(radius*Fgain*_rotation_gain(U[cell])*float(np.linalg.norm(normal)))
        extent+=F(bounds.roundoff_padding_m*max(1.,float(np.linalg.norm(normal,1))))
        slabs.append((min(restricted)-extent,max(restricted)+extent))
    lower=float(np.nextafter(float(min(s[0] for s in slabs)),-np.inf))
    upper=float(np.nextafter(float(max(s[1] for s in slabs)),np.inf))
    if not np.all(np.isfinite((lower,upper))):
        raise ValueError('curved material plane enclosure overflow')
    return CurvedPatchGeometryControl(min(v for _,v in cell_bounds),tuple(cell_bounds),
                                     lower,upper,bounds.as_search_box())


@dataclass(frozen=True)
class CurvedPatchPathControl:
    certified: bool
    reason: str
    intervals: int
    orientation_lower: float | None
    gap_lower_m: float | None
    path_sha256: str


def curved_patch_path_control(rule,start,translations,angular,origin,normal,*,
                              gap_tolerance_m,maximum_intervals,cancellation_token=None,
                              orientation_only=False):
    """Whole material patch along the existing affine/Exp path, not pose samples.

    On an interval of half-width h, d=|c-U*c_ref| is at most its midpoint
    value+h*(|dc|+|omega|*gain*|c_ref|). Apply the same complete-cell outward
    bound with that upper d and a path-wide Gram allowance. Plane gap variation
    is bounded by projected nodal translation plus rotating lift/section radius.
    Positive controls retain the declared normal/side. Unresolved controls never
    commit a state or return an inferred collision time.
    """
    from anyfesolver.control import cancellation_safe_point
    from anyfesolver._ge_beam3_p5_seeded.core import sha
    from .native_surface_s3_friction import NativeS3MaterialConfiguration
    from .native_surface_path import material_path_pose
    if type(start) is not NativeS3MaterialConfiguration or start.shell_displacement.size:
        raise TypeError('fixed-plane native path needs exact native-only configuration')
    if type(orientation_only) is not bool:
        raise TypeError('orientation-only control must be explicit boolean')
    if (type(gap_tolerance_m) not in (int,float) or not np.isfinite(gap_tolerance_m)
        or gap_tolerance_m<=0 or type(maximum_intervals) is not int or maximum_intervals<1):
        raise ValueError('positive dimensional gap tolerance and finite path interval cap required')
    move=_array(translations,(3,3),'curved path translations')
    spin=_array(angular,(2,3),'curved path angular increments')
    origin=_array(origin,(3,),'curved material plane origin')
    normal=_array(normal,(3,),'curved material plane normal')
    # Validates the same actual motion/chart as the structural path adapter.
    end=material_path_pose(start,move,spin,np.zeros(0),1.)
    rule.current_bounds(start.positions,start.position_low,start.cell_rotations)
    rule.current_bounds(end.positions,end.position_low,end.cell_rotations)
    identity=sha(dict(profile='NATIVE_CURVED_MATERIAL_AFFINE_EXP_PATH_DEV_V1',
        rule=rule.definition,start=start,translations=move,angular=spin,
        origin=origin,normal=normal,gap_tolerance_m=float(gap_tolerance_m),
        maximum_intervals=maximum_intervals))
    if orientation_only:
        identity=sha(dict(base=identity,admission='whole_native_orientation_only_v1'))
    ref=rule.samples[0].station.reference;radius=float(rule.definition['radius'])
    Fgain=_reference_radial_gain(ref);eps=np.finfo(float).eps;F=lambda v:Fraction(float(v))
    norm=lambda v:float(_sqrt_bound(sum(x*x for x in v),upper=True))
    cells=[];gap_rate=0.
    arm=_reference_curvature_norm(ref)/8+radius*Fgain
    scale=max(1.,float(np.max(np.abs(start.positions))),float(np.max(np.abs(start.position_low))),
        float(np.max(np.abs(move))),float(np.max(np.abs(ref.coordinates))))
    for cell in rule.current_bounds(start.positions,start.position_low,start.cell_rotations).cells:
        gain=_rotation_gain(start.cell_rotations[cell])*(1+2048*eps)
        gram=float(np.linalg.norm(start.cell_rotations[cell].T@start.cell_rotations[cell]-np.eye(3)))
        gram+=4096*eps*(1+gain*gain)
        dc=[F(move[cell+1,j])-F(move[cell,j]) for j in range(3)]
        cref=[F(ref.coordinates[cell+1,j])-F(ref.coordinates[cell,j]) for j in range(3)]
        speed=norm([F(v) for v in spin[cell]])
        rate=norm(dc)+gain*speed*norm(cref)
        translation=max(abs(float(sum(F(move[i,j])*F(normal[j]) for j in range(3))))
                        for i in (cell,cell+1))
        gap_rate=max(gap_rate,translation+gain*speed*arm*float(np.linalg.norm(normal)))
        cells.append((cell,gain,gram,rate))
    pad=32768*eps*scale*(1+Fgain)*max(1.,float(np.linalg.norm(normal,1)),arm)
    pending=[(0.,1.)];visited=0;orientation=gap=np.inf
    def result(certified,reason):
        return CurvedPatchPathControl(certified,reason,visited,
            float(orientation) if np.isfinite(orientation) else None,
            float(gap) if np.isfinite(gap) else None,identity)
    while pending:
        cancellation_safe_point(cancellation_token,'curved material whole path')
        if visited>=maximum_intervals:
            return result(False,'whole patch path interval budget exhausted')
        left,right=pending.pop();mid=(left+right)/2;half=(right-left)/2;visited+=1
        if mid in (left,right):
            return result(False,'whole patch path resolution exhausted')
        pose=material_path_pose(start,move,spin,np.zeros(0),mid)
        try:
            geometry=curved_patch_geometry_control(rule,pose.positions,pose.position_low,
                pose.cell_rotations,origin,normal)
        except ValueError as error:
            if ('orientation' not in str(error) and
                'native round surface singular or reversed at sample' not in str(error)):raise
            return result(False,'whole patch path orientation unresolved')
        lower=np.inf
        for cell,gain,gram,rate in cells:
            difference=_chord_difference_norm(pose.positions,pose.position_low,
                pose.cell_rotations[cell],ref,cell)+half*rate+pad
            lower=min(lower,_orientation_cell_lower(ref,cell,radius,gain,gram,difference,scale,Fgain))
        lower_gap=geometry.gap_lower_m-half*gap_rate-pad
        if not np.all(np.isfinite((lower,lower_gap))):
            raise ValueError('curved whole patch path bound overflow')
        if lower>0 and (orientation_only or lower_gap>=-gap_tolerance_m):
            orientation=min(orientation,lower);gap=min(gap,lower_gap)
            continue
        pending.extend(((left,mid),(mid,right)))
    return result(True,'whole native orientation bounded; gap not admitted' if orientation_only
                  else 'whole patch orientation and gap bounded')
