"""Curved native material normal work against a fixed plane.

Local development kernel, not an analysis owner. Actual material reference area
and retained work maps are preserved. Gauss8/16 tensor differences control
energy, force and spatial tangent under existing dimensioned settings. They
are error estimates, not rigorous integration error certificates.
"""
from dataclasses import asdict,dataclass
from fractions import Fraction
from decimal import Decimal, localcontext
from math import atan2,cos,sin,fsum,isfinite,pi,sqrt
import numpy as np
from numpy.polynomial.legendre import leggauss

from anyfesolver._ge_beam3_mixed_ad import Jet2
from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver._ge_beam3_p5_seeded.core import sha
from .native_retained_station import _readonly
from .native_surface_quadrature import NativeRoundSurfaceRule,_prepare_reference_area
from .native_surface_normal import StraightNormalIntegrationSettings
from .native_curved_geometry import (curved_patch_geometry_control,curved_patch_path_control,_reference_norm_bounds,
    _reference_second_bound,_reference_unclipped_interval,_reference_plane_derivative_bounds,
    _reference_curvature_coefficients,_reference_curvature_norm)
from .native_surface_bounds import _rotation_gain,_reference_radial_gain
from .native_normal_angles import PRECISION,PreciseNormalAngles

PROFILE='NATIVE_RETAINED_CURVED_MATERIAL_PLANE_LOCAL_WORK_DEV_V1'


def _projected_gap(high, low, cell, t, arm, origin, normal):
    """Project exact binary products before converting to one scalar value.

    Shape weights form an exact partition of unity in this scalar projection.
    A tangential change of plane origin cannot lose coordinate differences.
    """
    F=lambda v:Fraction(float(v))
    weight=F(t)
    value=sum(((1-weight)*(F(high[cell,i])+F(low[cell,i])-F(origin[i]))+
               weight*(F(high[cell+1,i])+F(low[cell+1,i])-F(origin[i]))+
               F(arm[i]))*F(normal[i]) for i in range(3))
    try:result=float(value)
    except (OverflowError,ValueError) as exc:
        raise ValueError('curved material scalar projection unresolved') from exc
    if not isfinite(result) or (value!=0 and result==0):
        raise ValueError('curved material scalar projection unresolved')
    return result


def _station_field(high,low,cell,t,lift,frame,rotation,radius,origin,normal):
    """Circular gap with exact binary projection and bounded scalar precision."""
    F=lambda v:Fraction(float(v))
    weight=F(t)
    base=sum(((1-weight)*(F(high[cell,i])+F(low[cell,i])-F(origin[i]))+
              weight*(F(high[cell+1,i])+F(low[cell+1,i])-F(origin[i]))+
              sum(F(rotation[i,j])*F(lift[j]) for j in range(3)))*F(normal[i])
             for i in range(3))
    def radial(column):
        return F(radius)*sum(F(normal[i])*F(rotation[i,j])*F(frame[j,column])
                             for i in range(3) for j in range(3))
    with localcontext() as ctx:
        ctx.prec=PRECISION
        D=lambda v:Decimal(v.numerator)/Decimal(v.denominator)
        return PreciseNormalAngles(D(base),0.,0.,D(radial(1)),D(radial(2)))


def _critical_gaps(field,left,right):
    phase=atan2(float(field.sine),float(field.cosine))
    angles=[left,right]+[phase+k*pi for k in range(-2,5) if left<phase+k*pi<right]
    return [field.value(angle) for angle in angles]


def _projected_center_derivative(reference,high,low,U,cell,xi,normal):
    F=lambda v:Fraction(float(v))
    b=_reference_curvature_coefficients(reference)
    t=F(xi)+(1 if cell==0 else 0)-Fraction(1,2)
    return float(sum(F(normal[i])*(F(high[cell+1,i])+F(low[cell+1,i])-
        F(high[cell,i])-F(low[cell,i])+sum(F(U[cell,i,j])*b[j]*t for j in range(3)))
        for i in range(3)))


def _projected_center_second(reference,U,cell,normal):
    F=lambda v:Fraction(float(v));b=_reference_curvature_coefficients(reference)
    return float(sum(F(normal[i])*F(U[cell,i,j])*b[j] for i in range(3) for j in range(3)))


def _rectangle_support(rule,high,low,U,cell,left,right,al,ar,origin,normal):
    """Continuous axial Lipschitz control; quadrature zeros are insufficient.

    x'=current_chord-U*reference_chord+U*r0'+U*R0'*rho.
    The whole-cell reference derivative bounds enclose every represented angle.
    Enclose variation of the angular minimum on the whole axial interval.
    Both empty and nonempty rectangles require a continuous support proof:
    one broad sampled pocket cannot hide another thin, unobserved pocket.
    Unresolved axial fronts lead to subdivision/non-success.
    """
    ref=rule.samples[0].station.reference;radius=rule.definition['radius']
    xi=(left+right)/2;t=xi+1 if cell==0 else xi
    field=_station_field(high,low,cell,t,ref.half_cell_lift(cell,t),ref.frame(xi),
                         U[cell],radius,origin,normal)
    F=lambda v:Fraction(float(v))
    chord=np.array([float(F(high[cell+1,i])+F(low[cell+1,i])-F(high[cell,i])-F(low[cell,i]))
                    for i in range(3)])
    _,jmax,prime,_=_reference_norm_bounds(ref,cell,actual_roll=True)
    gain=_rotation_gain(U[cell])
    # The projected center derivative is affine. Evaluate its exact binary
    # endpoint products; using a full tangent norm needlessly inflates it.
    center=max(abs(_projected_center_derivative(ref,high,low,U,cell,x,normal)) for x in (left,right))
    bound=center+radius*gain*prime*float(np.linalg.norm(normal))
    scale=max(1.,jmax,np.linalg.norm(chord),np.max(np.abs(ref.coordinates)),
              abs(float(field.offset)),abs(float(field.cosine)),abs(float(field.sine)))
    padding=8192*np.finfo(float).eps*scale
    minimum=float(min(_critical_gaps(field,al,ar)))
    variation=padding+bound*(right-left)/2
    if minimum-variation>0:return 'open'
    # Endpoints can create/disappear a second truncated arc while the angular
    # minimum remains negative. Maximum changes can merge two negative arcs.
    # Enclose these signs as well, rather than trusting sampled topology.
    critical=_critical_gaps(field,al,ar)
    endpoints=(float(field.value(al)),float(field.value(ar)))
    maximum=float(max(critical))
    if (minimum+variation<0 and abs(maximum)>variation
            and all(abs(g)>variation for g in endpoints)):
        return 'active'
    return 'unresolved'


def _angular_rule_grazing(field,left,right):
    with localcontext() as ctx:
        ctx.prec=PRECISION
        resolution=Decimal(float(64*np.finfo(float).eps))*(
            abs(field.offset)+abs(field.cosine)+abs(field.sine))
        return any(abs(g)<=resolution for g in _critical_gaps(field,left,right))


def _active_angular_rule(field,left,right,points,weights,*,root_band=False):
    """Moving roots need no boundary terms through second potential variation.

    At g=0 both k*g_-²/2 and k*g_-*delta(g) vanish. Integration uses the
    current negative arcs and the complete interior work derivatives. Decimal
    root coordinates and scalar gaps share the existing bounded precision.
    """
    with localcontext() as ctx:
        ctx.prec=PRECISION
        if _angular_rule_grazing(field,left,right):
            if not root_band:
                raise ValueError('curved material angular endpoint/extremum unresolved; grazing excluded')
            # A certified simple axial-root bracket may contain an angular
            # grazing station. Its full positive-weight work norm is bounded
            # separately; it is not an unaccounted empty/support assumption.
            lo,hi=Decimal(left),Decimal(right);mid=(lo+hi)/2;half=(hi-lo)/2
            for point,weight in zip(points,weights):
                angle=mid+half*Decimal(float(point));scalar=float(field.value(angle))
                measure=float(half*Decimal(float(weight)))
                if not isfinite(scalar) or not isfinite(measure) or measure<=0:
                    raise ValueError('curved axial root-band angular measure unresolved')
                if scalar<0:yield float(angle),measure,scalar
            return
        breaks=field.breakpoints(left,right)
        for lo,hi in zip(breaks[:-1],breaks[1:]):
            mid=(lo+hi)/2;half=(hi-lo)/2
            if field.value(mid)>=0:continue
            for point,weight in zip(points,weights):
                angle=mid+half*Decimal(float(point))
                scalar=float(field.value(angle));measure=float(half*Decimal(float(weight)))
                if (not isfinite(scalar) or scalar>=0 or measure<=0 or not isfinite(measure)
                    or not float(lo)<float(angle)<float(hi)):
                    raise ValueError('curved material active angular quadrature unresolved')
                yield float(angle),measure,scalar


def _front_sample(rule,high,low,U,cell,xi,al,ar,origin,normal):
    ref=rule.samples[0].station.reference;r=rule.definition['radius']
    t=xi+1 if cell==0 else xi
    field=_station_field(high,low,cell,t,ref.half_cell_lift(cell,t),ref.frame(xi),
                         U[cell],r,origin,normal)
    trace=('LEFT' if cell==0 else 'RIGHT') if xi==0. else 'VALUE'
    prime=ref.frame_derivative(xi,trace=trace)
    center=_projected_center_derivative(ref,high,low,U,cell,xi,normal)
    cp=float(r*normal@U[cell]@prime[:,1]);sp=float(r*normal@U[cell]@prime[:,2])
    with localcontext() as ctx:
        ctx.prec=PRECISION
        amplitude=(field.cosine*field.cosine+field.sine*field.sine).sqrt()
        values=np.array([float(field.value(al)),float(field.value(ar)),
                         float(field.offset-amplitude),float(field.offset+amplitude)])
    amplitude=float(amplitude)
    ap=(float(field.cosine)*cp+float(field.sine)*sp)/amplitude if amplitude else np.nan
    derivatives=np.array((center+cp*cos(al)+sp*sin(al),center+cp*cos(ar)+sp*sin(ar),
                           center-ap,center+ap))
    return field,values,derivatives,amplitude


def _axial_front_bands(rule,high,low,U,cell,left,right,al,ar,origin,normal,budget):
    """Isolate simple endpoint/extremum roots, retaining uncertain brackets.

    Every unresolved isolation interval and integration rectangle shares the
    caller's finite interval budget. Refine monotone proposals only inside a
    bracket. Representation uncertainty is retained as a band, never an exact
    root assertion or silently merged feature. Tangent/degenerate roots reject.
    """
    ref=rule.samples[0].station.reference;r=rule.definition['radius']
    jmin,jmax,F1,_=_reference_norm_bounds(ref,cell,actual_roll=True)
    F2=_reference_second_bound(ref,cell);gain=_rotation_gain(U[cell])
    nn=float(np.linalg.norm(normal));full_radial1=r*gain*nn*F1;full_radial2=r*gain*nn*F2
    projected1,projected2=_reference_plane_derivative_bounds(ref,cell,U[cell],normal)
    center2=abs(_projected_center_second(ref,U,cell,normal))
    nodes=(-1.,0.) if cell==0 else (0.,1.)
    discrepancy=0.
    for endpoint,other in ((nodes[0],nodes[1]),(nodes[1],nodes[0])):
        inside=float(np.nextafter(endpoint,other))
        discrepancy=max(discrepancy,float(np.linalg.norm(ref.frame(inside)-ref.frame(endpoint)))+
                        F1*abs(inside-endpoint))
    scale=max(1.,jmax,np.max(np.abs(ref.coordinates)),r)
    padding=512*np.finfo(float).eps*scale*(1+jmax/jmin)+r*gain*nn*discrepancy
    roots=[];pending=[(left,right)]
    reasons=dict(amplitude=0,clipping=0,derivative_margin=0)
    def sample(x):return _front_sample(rule,high,low,U,cell,x,al,ar,origin,normal)
    while pending:
        if budget['visited']>=budget['limit']:
            error=MemoryError('curved axial front interval budget exhausted')
            error.integration_diagnostics=dict(isolation_visited=budget['visited'],
                unresolved_reasons=reasons,full_radial_first=full_radial1,
                projected_radial_first=r*projected1,full_radial_second=full_radial2,
                projected_radial_second=r*projected2,cell=cell,pending=len(pending))
            raise error
        xl,xr=pending.pop();budget['visited']+=1
        xm=(xl+xr)/2;half=(xr-xl)/2
        if xm in (xl,xr):raise ValueError('curved axial front parameter resolution exhausted')
        field,values,derivatives,amplitude=sample(xm)
        F=lambda v:Fraction(float(v))
        chord=np.array([float(F(high[cell+1,i])+F(low[cell+1,i])-F(high[cell,i])-F(low[cell,i]))
                        for i in range(3)])
        center1=max(abs(_projected_center_derivative(ref,high,low,U,cell,x,normal)) for x in (xl,xr))
        unclipped=_reference_unclipped_interval(ref,cell,xl,xr)
        radial1,radial2=(r*projected1,r*projected2) if unclipped else (full_radial1,full_radial2)
        L1=center1+radial1
        amin=amplitude-radial1*half-padding
        unresolved=False;isolated=[]
        vlo=sample(xl)[1];vhi=sample(xr)[1]
        for feature in range(4):
            if abs(values[feature])>L1*half+padding:continue
            M2=center2+radial2
            if feature>=2:
                if amin<=0:reasons['amplitude']+=1;unresolved=True;break
                M2+=radial1*radial1/amin
            if unclipped and abs(values[feature])>abs(derivatives[feature])*half+.5*M2*half*half+padding:
                continue
            margin=abs(derivatives[feature])-M2*half-128*np.finfo(float).eps*max(1.,L1)
            if not unclipped or not np.isfinite(margin) or margin<=0:
                reasons['clipping' if not unclipped else 'derivative_margin']+=1
                unresolved=True;break
            if vlo[feature]>padding and vhi[feature]>padding:continue
            if vlo[feature]<-padding and vhi[feature]<-padding:continue
            isolated.append((feature,margin,M2))
        if unresolved:
            pending.extend(((xl,xm),(xm,xr)));continue
        for feature,margin,M2 in isolated:
            lo,hi=xl,xr;flo=vlo[feature];fhi=vhi[feature]
            x=xm
            for _ in range(12):
                _,v,d,_=sample(x);fx=v[feature]
                uncertainty=(abs(fx)+padding)/margin
                if abs(fx)<=padding:
                    # The first bracket guarantees root containment. A local
                    # derivative enclosure can tighten that same bracket;
                    # this changes no force law or numerical tolerance.
                    local_margin=abs(d[feature])-M2*uncertainty-128*np.finfo(float).eps*max(1.,L1)
                    if local_margin>margin:
                        uncertainty=(abs(fx)+padding)/local_margin
                    lo=max(lo,float(np.nextafter(x-uncertainty,-np.inf)))
                    hi=min(hi,float(np.nextafter(x+uncertainty,np.inf)))
                    break
                if (fx>0)==(flo>0):lo=x;flo=fx
                else:hi=x;fhi=fx
                proposed=x-fx/d[feature]
                x=proposed if lo<proposed<hi else (lo+hi)/2
                if x in (lo,hi):
                    break
            else:raise ValueError('curved axial front root precision exhausted')
            if not left<=lo<hi<=right:
                raise ValueError('curved axial front bracket ownership unresolved')
            roots.append((lo,hi,(feature,)))
    # Overlapping brackets retain all feature identities and their entire union.
    # Integration below accounts for its work uncertainty rather than dropping it.
    unions=[]
    for lo,hi,features in sorted(roots):
        if unions and lo<=unions[-1][1]:
            old=unions.pop();unions.append((old[0],max(hi,old[1]),old[2]+features))
        else:unions.append((lo,hi,features))
    bands=[];start=left
    for lo,hi,features in unions:
        if lo>start:bands.append((start,lo,False,()))
        bands.append((lo,hi,True,features));start=hi
    if start<right:bands.append((start,right,False,()))
    return bands,padding


def _negative_arc_width_bound(field,al,ar):
    """Outward arc measure; sub-resolution endpoint/extremum roots keep full arc."""
    with localcontext() as ctx:
        ctx.prec=PRECISION
        gaps=_critical_gaps(field,al,ar)
        resolution=Decimal(float(64*np.finfo(float).eps))*(
            abs(field.offset)+abs(field.cosine)+abs(field.sine))
        if any(abs(g)<=resolution for g in gaps):return ar-al
        if min(gaps)>0:return 0.
        if max(gaps)<0:return ar-al
        try:
            breaks=field.breakpoints(al,ar)
            width=sum((b-a for a,b in zip(breaks[:-1],breaks[1:]) if field.value((a+b)/2)<0),Decimal(0))
        except (ValueError,ArithmeticError):return ar-al
        return min(ar-al,float(np.nextafter(float(width),np.inf)))


def _root_band_measure_bound(field,lower,al,ar):
    # Unsplit fallback nodes can put a full Gauss weight in a tiny pocket.
    # Cover both the true union and that approximate positive-weight measure.
    union=_negative_arc_width_bound(lower,al,ar)
    if not _angular_rule_grazing(field,al,ar):return union,False
    masses=[sum(w for _,w,_ in _active_angular_rule(field,al,ar,*leggauss(order),root_band=True))
            for order in (8,16)]
    return max(union,*(float(np.nextafter(m,np.inf)) for m in masses)),True


def _root_band_station(rule,high,low,U,cell,left,right,al,ar,origin,normal):
    """Prefer a resolved interior angular rule inside the retained bracket.

    The root band remains integrated and budgeted. Its representative station
    need not coincide with an unresolved angular grazing extremum.
    """
    ref=rule.samples[0].station.reference;radius=rule.definition['radius']
    middle=None
    for fraction in (.5,.25,.75):
        xi=float(left+(right-left)*fraction)
        if not left<xi<right:continue
        t=xi+1 if cell==0 else xi
        field=_station_field(high,low,cell,t,ref.half_cell_lift(cell,t),ref.frame(xi),
                             U[cell],radius,origin,normal)
        if not _angular_rule_grazing(field,al,ar):return xi,field
        if fraction==.5:middle=(xi,field)
    if middle is None:raise ValueError('curved axial root-band parameter resolution exhausted')
    return middle


def _root_band_active_width(rule,high,low,U,cell,left,right,al,ar,origin,normal,pad):
    """Angular union containing every possible active point in a root band.

    g(x,angle)>=g(mid,angle)-h*L-pad for the complete band and declared arc.
    Integrating the negative arcs of that lower circular field bounds the
    union measure, including endpoint arcs and any broad existing contact.
    A degenerate/unresolved angular enclosure retains the full arc.
    """
    ref=rule.samples[0].station.reference;radius=rule.definition['radius']
    station,field=_root_band_station(rule,high,low,U,cell,left,right,al,ar,origin,normal)
    F1=_reference_norm_bounds(ref,cell,actual_roll=True)[2]
    center=max(abs(_projected_center_derivative(ref,high,low,U,cell,x,normal)) for x in (left,right))
    projected=_reference_plane_derivative_bounds(ref,cell,U[cell],normal)[0]
    radial=projected if _reference_unclipped_interval(ref,cell,left,right) else (
        _rotation_gain(U[cell])*F1*float(np.linalg.norm(normal)))
    rate=center+radius*radial
    delta=float(np.nextafter(max(station-left,right-station)*rate+pad,np.inf))
    with localcontext() as ctx:
        ctx.prec=PRECISION
        lower=PreciseNormalAngles(field.offset-Decimal(delta),0.,0.,field.cosine,field.sine)
        measure,fallback=_root_band_measure_bound(field,lower,al,ar)
        return measure,fallback,station


def _root_band_work_bound(rule,high,low,U,cell,width,angular_width,geometry,settings,stiffness,normal,pad,*,fallback=False):
    """Bound both true and approximate work on retained root-bracket bands.

    No represented band is deleted. Energy/residual are Lipschitz in xi across
    activation, so interior-station integration has a width² variation bound, plus
    representation allowances and separately measured angular error. Tangent
    keeps twice the full positive-weight integrand norm; large bounds reject.
    """
    ref=rule.samples[0].station.reference;r=rule.definition['radius']
    _,jmax,F1,_=_reference_norm_bounds(ref,cell,actual_roll=True)
    F2=_reference_second_bound(ref,cell)
    fg=_reference_radial_gain(ref);gain=_rotation_gain(U[cell])
    B=_reference_curvature_norm(ref)
    arm=gain*(B/8+r*fg)
    nn=float(np.linalg.norm(normal))
    length=settings.characteristic_length_m
    G=nn*sqrt(1+(arm/length)**2)
    depth=max(0.,-geometry.gap_lower_m)
    area=r*fg*(jmax+r*F1)
    area_prime=r*fg*(B+r*F2)+(jmax+r*F1)*r*F1
    F=lambda v:Fraction(float(v))
    chord=np.array([float(F(high[cell+1,i])+F(low[cell+1,i])-F(high[cell,i])-F(low[cell,i]))
                    for i in range(3)])
    arm_prime=gain*(B/2+r*F1)
    gap_prime=nn*(np.linalg.norm(chord)+arm_prime)
    Gprime=nn*sqrt(2+(arm_prime/length)**2)
    energy_lipschitz=stiffness*(depth*gap_prime*area+.5*depth**2*area_prime)
    force_lipschitz=stiffness*((gap_prime*G+depth*Gprime)*area+depth*G*area_prime)
    energy_error=angular_width*(width**2*energy_lipschitz/2+
                                 width*stiffness*depth*pad*area)
    force_error=angular_width*(width**2*force_lipschitz/2+
                                width*stiffness*pad*G*area)
    if fallback:
        # An unsplit approximate angular rule is not an exact negative-arc
        # integral. Bound both its work and the true work directly instead
        # of treating 8/16 agreement as a quadrature certificate.
        energy_error=max(energy_error,stiffness*area*width*angular_width*depth**2)
        force_error=max(force_error,2*stiffness*area*width*angular_width*depth*G)
    # g has two translational weights and one rotational arm. ||g||²<=G².
    # The 3x3 rotational Hessian has Frobenius norm<=sqrt(3)*arm*nn/L²;
    # once-spatial .5*skew(moment) adds <=arm*nn/(sqrt(2)*L²).
    # Their sum is <3*arm*nn/L². Nine nonzero local coordinates give
    # ||K||_inf<=3*||K||_F. Keep twice the full positive integrand bound.
    tangent_error=6*stiffness*area*width*angular_width*(
        G*G+3*depth*arm*nn/length**2)
    return np.array((energy_error,force_error,tangent_error))*(1+8192*np.finfo(float).eps)


@dataclass(frozen=True)
class CurvedMaterialPlaneResult:
    normal_storage_j: float
    residual: np.ndarray
    tangent: np.ndarray
    source_force: np.ndarray
    target_force: np.ndarray
    source_moment: np.ndarray
    target_moment: np.ndarray
    geometry: object
    intervals: int
    accepted_intervals: int
    integrand_evaluations: int
    estimated_errors: tuple
    axial_root_brackets: tuple=()
    axial_isolation_intervals: int=0

    def spatial_tangent(self):
        return self.tangent


class NativeCurvedMaterialPlaneWork:
    """Fixed-plane local kernel; no solve/restart/history or generic insertion."""
    def __init__(self,rule,*,plane_origin_m,plane_normal,normal_stiffness_Pa_per_m,
                 integration_settings):
        if type(rule) is not NativeRoundSurfaceRule or type(integration_settings) is not StraightNormalIntegrationSettings:
            raise TypeError('exact material surface rule and existing normal integration settings required')
        origin=np.asarray(plane_origin_m,dtype=float);normal=np.asarray(plane_normal,dtype=float)
        if (origin.shape!=(3,) or normal.shape!=(3,) or not np.all(np.isfinite(origin))
            or not np.all(np.isfinite(normal)) or abs(np.linalg.norm(normal)-1.)>1e-12):
            raise ValueError('finite plane origin and unit normal required')
        if isinstance(normal_stiffness_Pa_per_m,(bool,np.bool_)):
            raise ValueError('dimensioned curved normal stiffness must not be boolean')
        stiffness=float(normal_stiffness_Pa_per_m)
        if not isfinite(stiffness) or stiffness<=0:
            raise ValueError('positive dimensioned curved normal stiffness required')
        self.rule=rule;self.reference=rule.samples[0].station.reference
        self.origin=_readonly(origin);self.normal=_readonly(normal)
        self.stiffness=stiffness;self.settings=integration_settings
        self.identity=sha(self.definition())
        self.reference_geometry=curved_patch_geometry_control(rule,self.reference.coordinates,
            np.zeros((3,3)),np.tile(np.eye(3),(2,1,1)),origin,normal)

    def definition(self):
        return dict(profile=PROFILE,rule=self.rule.definition,
            origin_m=self.origin,normal=self.normal,
            stiffness_N_per_m3=self.stiffness,settings=asdict(self.settings),
            reference=self.reference.canonical_data(),measure='actual_fixed_reference_material_area',
            branch_admission='continuous_orientation_simple_axial_front_brackets_bounded_work',
            numerical='GAUSS_8_16_PROJECTED_CURVED_AXIAL_ARCS_RESOLVED_BAND_JET2_SPATIAL_ONCE_V7')

    @property
    def element_id(self):
        return self.rule.samples[0].station.element_id

    @property
    def radius(self):
        return float(self.rule.definition['radius'])

    def minimum_gap_bound(self,positions,position_low,cell_rotations):
        """Conservative whole-patch bound for trial admission, not a gap sample."""
        if sha(self.definition())!=self.identity:
            raise ValueError('curved material plane definition/settings changed')
        return curved_patch_geometry_control(self.rule,positions,position_low,cell_rotations,
            self.origin,self.normal).gap_lower_m

    def evaluate(self,positions,position_low,cell_rotations):
        if sha(self.definition())!=self.identity:
            raise ValueError('curved material plane definition/settings changed')
        geometry=curved_patch_geometry_control(self.rule,positions,position_low,
            cell_rotations,self.origin,self.normal)
        segmented=geometry.gap_lower_m<0<=geometry.gap_upper_m
        high=np.asarray(positions);low=np.asarray(position_low);U=np.asarray(cell_rotations)
        settings=self.settings;radius=float(self.rule.definition['radius'])
        lo,hi=self.rule.definition['xi_range'];alo,ahi=self.rule.definition['angle_range']
        total_area=(hi-lo)*(ahi-alo)
        limits=(settings.energy_absolute_tolerance,settings.force_absolute_tolerance,
                settings.tangent_absolute_tolerance)
        gauss=(leggauss(8),leggauss(16))
        budget=dict(visited=0,limit=settings.max_intervals)
        prepared=[];root_errors=np.zeros(3);root_brackets=[];root_stations={}
        for cell,left,right in ((0,max(lo,-1.),min(hi,0.)),(1,max(lo,0.),min(hi,1.))):
            if right<=left:continue
            bands,pad=(_axial_front_bands(self.rule,high,low,U,cell,left,right,alo,ahi,
                         self.origin,self.normal,budget) if segmented else ([(left,right,False,())],0.))
            for xl,xr,uncertain,features in bands:
                if uncertain:
                    active_width,fallback,station=_root_band_active_width(self.rule,high,low,U,cell,xl,xr,alo,ahi,
                        self.origin,self.normal,pad)
                    error=self._root_work_bound(high,low,U,cell,xr-xl,active_width,
                        geometry,settings,self.stiffness,self.normal,pad,fallback=fallback,
                        interval=(xl,xr),station=station)
                    root_errors+=error
                    root_stations[cell,xl,xr]=station
                    root_brackets.append(dict(cell=cell,lower=xl,upper=xr,features=features,
                        representation_gap_m=pad,active_angular_width_bound=active_width,
                        angular_fallback=fallback,representative_xi=station,error_bounds=tuple(error)))
            prepared.append((cell,bands))
        total_area=self._integration_budget_area(prepared,segmented,high,low,U,
            alo,ahi,total_area)
        remaining=np.asarray(limits)-root_errors
        if np.all(np.isfinite(root_errors)) and np.any(remaining<=0):
            # A formulation-owned tighter enclosure may resolve a conservative
            # rejection. Reuse the same isolated bands/stations and interval
            # budget; do not change an already passing integration or its saved
            # diagnostics. No roots, quadrature points or tolerances are reset.
            refined=[]
            for row in root_brackets:
                error=self._refined_root_work_bound(high,low,U,row['cell'],
                    row['upper']-row['lower'],row['active_angular_width_bound'],
                    geometry,settings,self.stiffness,self.normal,row['representation_gap_m'],
                    fallback=row['angular_fallback'],interval=(row['lower'],row['upper']),
                    station=row['representative_xi'])
                if error is None:break
                refined.append(np.minimum(row['error_bounds'],error))
            if len(refined)==len(root_brackets) and refined:
                root_errors=np.sum(refined,axis=0)
                for row,error in zip(root_brackets,refined):
                    row['error_bounds']=tuple(error)
                    row['enclosure_refinement']='declared_material_arc'
                remaining=np.asarray(limits)-root_errors
        if np.all(np.isfinite(root_errors)) and np.any(remaining<=0):
            # A rejected whole-bracket enclosure may still be certified by
            # further bounded isolation of the SAME retained brackets. Every
            # sub-band keeps its own representative station, depressed-field
            # angular union and work enclosure; the partition's summed
            # enclosure replaces the rejected one only when it fits the
            # unchanged limits within the same finite interval budget. No
            # root is asserted, no representation pad is released, and no
            # already passing evaluation enters this branch.
            groups,attempt=self._certified_root_subbands(high,low,U,geometry,
                settings,limits,budget,root_brackets,alo,ahi)
            if groups is not None:
                prepared,root_brackets=self._partition_root_subbands(prepared,
                    root_brackets,groups,root_stations)
                root_errors=np.sum([node['error'] for group in groups for node in group],axis=0)
                remaining=np.asarray(limits)-root_errors
            else:
                failure=ValueError('curved axial front bracket work uncertainty exceeds integration budget')
                failure.integration_diagnostics=dict(root_brackets=root_brackets,
                    error_bounds=tuple(root_errors),subband_isolation=attempt)
                raise failure
        if not np.all(np.isfinite(root_errors)) or np.any(remaining<=0):
            failure=ValueError('curved axial front bracket work uncertainty exceeds integration budget')
            failure.integration_diagnostics=dict(root_brackets=root_brackets,error_bounds=tuple(root_errors))
            raise failure
        count=self._coordinate_count()
        global_residual=np.zeros(count);global_hessian=np.zeros((count,count))
        energies=[];errors=[tuple(root_errors)];visited=budget['visited'];accepted=evaluations=0
        last_error=last_rectangle=last_support=None
        scale=self._coordinate_scale(count)
        for cell,bands in prepared:
            mapping=self._local_mapping(cell,count)
            pending=[(xl,xr,alo,ahi,uncertain) for xl,xr,uncertain,_ in bands]
            while pending:
                if visited>=settings.max_intervals:
                    failure=MemoryError('curved normal integration interval budget exhausted')
                    failure.integration_diagnostics=dict(visited=visited,accepted=accepted,
                        evaluations=evaluations,last_errors=last_error,last_rectangle=last_rectangle,
                        last_support=last_support)
                    raise failure
                xl,xr,al,ar,root_band=pending.pop();visited+=1
                xm=(xl+xr)/2;am=(al+ar)/2;dx=(xr-xl)/2;da=(ar-al)/2
                if xm in (xl,xr) or am in (al,ar) or dx==0 or da==0:
                    raise ValueError('curved normal integration parameter resolution exhausted')
                # Complete root isolation proves fixed topology between retained
                # brackets. There is no sampled zero-support certificate here.
                if segmented and not root_band:
                    t=xm+1 if cell==0 else xm
                    field=_station_field(high,low,cell,t,self.reference.half_cell_lift(cell,t),
                        self.reference.frame(xm),U[cell],radius,self.origin,self.normal)
                    support='open' if min(_critical_gaps(field,al,ar))>0 else 'active'
                else:support='root_band' if root_band else 'active'
                last_support=support
                if support=='unresolved':
                    last_error=None;last_rectangle=(xl,xr,al,ar)
                    pending.extend(((xl,xm,al,ar,False),(xm,xr,al,ar,False)))
                    continue
                if support=='open':
                    accepted+=1;energies.append(0.);errors.append((0.,0.,0.))
                    continue
                estimates=[];active_samples=0
                for points,weights in gauss:
                    estimate=Jet2.constant(0.,mapping.shape[0])
                    for qx,wx in zip(points,weights):
                        if segmented and not root_band:
                            # Circular onset gives a square-root angular width.
                            # Quadratic approach to both axial partition ends
                            # regularizes that integrand without changing work.
                            theta=pi*(float(qx)+1)/4
                            xi=float(xl+(xr-xl)*sin(theta)**2)
                            axial_measure=float((xr-xl)*pi/4*sin(2*theta)*wx)
                        else:
                            xi=root_stations[cell,xl,xr] if root_band else float(xm+dx*qx)
                            axial_measure=float(dx*wx)
                        if not xl<xi<xr:
                            raise ValueError('curved normal integration axial station unresolved')
                        t=xi+1 if cell==0 else xi
                        reference_area=_prepare_reference_area(self.reference,xi,radius)
                        frame=reference_area.frame
                        lift=self.reference.half_cell_lift(cell,t)
                        field=(_station_field(high,low,cell,t,lift,frame,U[cell],radius,
                                              self.origin,self.normal) if segmented else None)
                        angular=(_active_angular_rule(field,al,ar,points,weights,root_band=root_band) if segmented
                                 else ((float(am+da*qa),float(da*wa),None)
                                       for qa,wa in zip(points,weights)))
                        for angle,angular_measure,scalar_gap in angular:
                            if not al<angle<ar:
                                raise ValueError('curved normal integration angular station unresolved')
                            factor=float(axial_measure*angular_measure)
                            if not isfinite(factor) or factor==0:
                                raise ValueError('curved normal integration quadrature measure unresolved')
                            radial=radius*(cos(angle)*frame[:,1]+sin(angle)*frame[:,2])
                            arm=U[cell]@(lift+radial)
                            gap=(_projected_gap(high,low,cell,t,arm,self.origin,self.normal)
                                 if scalar_gap is None else scalar_gap)
                            evaluations+=1
                            if gap<0 and geometry.gap_lower_m>=0:
                                raise ValueError('curved material gap disagrees with open enclosure')
                            if gap>=0:
                                if geometry.gap_upper_m<0:
                                    raise ValueError('curved material gap disagrees with compressed enclosure')
                                continue
                            active_samples+=1
                            value=self._gap_work_jet(gap,high,low,cell,t,arm)
                            area=reference_area.area_jacobian(angle)
                            coefficient=factor*area*.5*self.stiffness
                            if not isfinite(coefficient) or coefficient==0:
                                raise ValueError('curved normal integration reference measure/stiffness unresolved')
                            estimate=estimate+coefficient*value*value
                    estimates.append(estimate)
                difference=estimates[1]-estimates[0]
                residual=mapping.T@difference.gradient
                tangent=self._spatial_tangent(residual,mapping.T@difference.hessian@mapping)
                error=np.array((abs(difference.value),np.linalg.norm(scale*residual,np.inf),
                    np.linalg.norm(scale[:,None]*tangent*scale[None,:],np.inf)))
                if not np.all(np.isfinite(error)):
                    raise ValueError('nonfinite curved normal integration error estimate')
                last_error=tuple(map(float,error));last_rectangle=(xl,xr,al,ar)
                local_budget=remaining*(xr-xl)*(ar-al)/total_area
                if segmented and not root_band and active_samples==0:
                    raise ValueError('curved material active support disagrees with angular quadrature')
                if root_band or np.all(error<=local_budget):
                    integral=estimates[1];accepted+=1
                    errors.append((float(error[0]),float(error[1]),0.) if root_band else tuple(map(float,error)))
                    energies.append(integral.value)
                    global_residual+=mapping.T@integral.gradient
                    global_hessian+=mapping.T@integral.hessian@mapping
                else:
                    if segmented or (xr-xl)/(hi-lo)>=(ar-al)/(ahi-alo):
                        pending.extend(((xl,xm,al,ar,False),(xm,xr,al,ar,False)))
                    else:
                        pending.extend(((xl,xr,al,am,False),(xl,xr,am,ar,False)))
        tangent=self._spatial_tangent(global_residual,global_hessian)
        energy=fsum(energies)
        final_errors=tuple(fsum(e[i] for e in errors) for i in range(3))
        if np.any(np.asarray(final_errors)>np.asarray(limits)):
            raise ValueError('curved normal integration combined error budget exhausted')
        if not isfinite(energy) or not np.all(np.isfinite(global_residual)) or not np.all(np.isfinite(tangent)):
            raise ValueError('nonfinite curved normal work')
        forces=-global_residual[:18].reshape(3,6)[:,:3]
        force=forces.sum(axis=0)
        moment=sum((np.cross(high[i],forces[i])+np.cross(low[i],forces[i]) for i in range(3)),start=np.zeros(3))
        moment-=global_residual[18:24].reshape(2,3).sum(axis=0)
        return CurvedMaterialPlaneResult(energy,_readonly(global_residual),_readonly(tangent),
            _readonly(force),_readonly(-force),_readonly(moment),_readonly(-moment),geometry,
            visited,accepted,evaluations,final_errors,
            tuple(root_brackets),budget['visited'])

    def _integration_budget_area(self,prepared,segmented,high,low,U,alo,ahi,total_area):
        """Default allocation preserves the complete reference parameter domain."""
        return total_area

    def _coordinate_count(self):
        return 42

    def _coordinate_scale(self,count):
        scale=np.ones(count)
        for index in (3,9,15,18,21):
            scale[index:index+3]=1/self.settings.characteristic_length_m
        return scale

    def _local_mapping(self,cell,count):
        mapping=np.zeros((9,count))
        mapping[:3,6*cell:6*cell+3]=np.eye(3)
        mapping[3:6,6*(cell+1):6*(cell+1)+3]=np.eye(3)
        mapping[6:9,18+3*cell:21+3*cell]=np.eye(3)
        return mapping

    def _gap_work_jet(self,gap,high,low,cell,t,arm):
        gradient=np.r_[(1-t)*self.normal,t*self.normal,np.cross(arm,self.normal)]
        hessian=np.zeros((9,9))
        hessian[6:9,6:9]=(np.outer(arm,self.normal)+np.outer(self.normal,arm))/2-float(self.normal@arm)*np.eye(3)
        return Jet2(gap,gradient,hessian)

    def _spatial_tangent(self,residual,hessian):
        return spatial_jacobian(residual,hessian)

    def _root_work_bound(self,high,low,U,cell,width,angular_width,geometry,
                         settings,stiffness,normal,pad,*,fallback=False,interval=None,station=None):
        return _root_band_work_bound(self.rule,high,low,U,cell,width,angular_width,
            geometry,settings,stiffness,normal,pad,fallback=fallback)

    def _refined_root_work_bound(self,*args,**kwargs):
        return None

    def _certified_root_subbands(self,high,low,U,geometry,settings,limits,budget,
                                 brackets,alo,ahi):
        """Budget-aware further certified isolation of rejected root brackets.

        Bisection partitions each retained bracket into sub-bands sharing the
        caller's finite interval budget. A sub-band's depressed station field
        bounds its complete active angular union, its work enclosure bounds
        its complete work uncertainty, and the partition sum is valid by
        finite subadditivity. Splitting stops when the summed enclosure fits
        strictly under the unchanged limits, the shared budget or float
        resolution ends refinement, or any sub-band enclosure is unresolved.
        An enclosure that cannot be certified smaller by splitting (grazing or
        representation floor) is explicitly rejected as a refinement target
        while other sub-bands continue; no new cap is introduced. The
        partition total is recomputed from the nonnegative sub-band
        enclosures after every replacement, never by a running subtraction;
        an empty group or a nonfinite/negative enclosure keeps the existing
        non-success.
        The representation pad stays in every sub-band; feature identity is
        inherited; no root is ever asserted. Returns the per-bracket sub-band
        groups (None when certification fails) and attempt diagnostics.
        """
        if not brackets:return None,dict(subband_evaluations=0)
        groups=[]
        for index,row in enumerate(brackets):
            error=np.asarray(row['error_bounds'],float)
            if not np.all(np.isfinite(error)) or np.any(error<0):
                return None,dict(subband_evaluations=0)
            groups.append([dict(parent=index,cell=row['cell'],pad=row['representation_gap_m'],
                features=row['features'],lo=row['lower'],hi=row['upper'],
                station=row['representative_xi'],active=row['active_angular_width_bound'],
                fallback=row['angular_fallback'],error=error,
                refinement=row.get('enclosure_refinement'))])
        evaluations=0
        def subband(node):
            active,fallback,station=_root_band_active_width(self.rule,high,low,U,
                node['cell'],node['lo'],node['hi'],alo,ahi,self.origin,self.normal,node['pad'])
            error=self._refined_root_work_bound(high,low,U,node['cell'],node['hi']-node['lo'],
                active,geometry,settings,self.stiffness,self.normal,node['pad'],
                fallback=fallback,interval=(node['lo'],node['hi']),station=station)
            refined=error is not None
            if error is None:
                error=self._root_work_bound(high,low,U,node['cell'],node['hi']-node['lo'],
                    active,geometry,settings,self.stiffness,self.normal,node['pad'],
                    fallback=fallback,interval=(node['lo'],node['hi']),station=station)
            node.update(active=active,fallback=fallback,station=station,
                error=np.asarray(error,float),
                refinement='declared_material_arc' if refined else None)
        def split(node,ceiling):
            mid=(node['lo']+node['hi'])/2
            if not node['lo']<mid<node['hi']:return False
            # Two sub-band evaluations plus one net new integration rectangle
            # must fit the isolation-side share of the shared budget.
            if budget['visited']+3>ceiling or budget['visited']+3>budget['limit']:return False
            children=[dict(parent=node['parent'],cell=node['cell'],pad=node['pad'],
                features=node['features'],lo=node['lo'],hi=mid),
                dict(parent=node['parent'],cell=node['cell'],pad=node['pad'],
                features=node['features'],lo=mid,hi=node['hi'])]
            for child in children:
                budget['visited']+=1
                nonlocal evaluations;evaluations+=1
                try:subband(child)
                except (ValueError,ArithmeticError):return False
                if not np.all(np.isfinite(child['error'])) or np.any(child['error']<0):
                    return False
            return children
        caps=np.asarray(limits,float)
        def partition_total():
            # Recompute from the sub-band lists after every replacement; a
            # running subtraction update can cancel and understate the sum.
            errors=[node['error'] for group in groups for node in group]
            if not errors:return None
            total=np.sum(errors,axis=0)
            if not np.all(np.isfinite(total)) or np.any(total<0):return None
            return total
        total=partition_total()
        # Root-band certification and quadrature share one tangent/energy/force
        # budget: every unit of certified root uncertainty is a unit the
        # integration rectangles cannot use. Refine therefore past the first
        # fitting partition towards the representation floor, spending at
        # most half of the not-yet-spent shared interval budget; integration
        # rectangles keep the other half.
        ceiling=(budget['visited']+budget['limit'])//2
        while total is not None:
            component=int(np.argmax(total/caps))
            target=None
            for group in groups:
                for node in group:
                    if node.get('irreducible'):continue
                    if target is None or node['error'][component]>target['error'][component]:
                        target=node
            if target is None:break
            children=split(target,ceiling)
            if children is False or (children[0]['error'][component]
                    +children[1]['error'][component]>=target['error'][component]):
                # Splitting cannot be certified to shrink this enclosure
                # (grazing/representation floor, exhausted shared budget or
                # float resolution). Reject it explicitly as a refinement
                # target; other sub-bands keep the same unchanged caps.
                target['irreducible']=True;continue
            group=groups[target['parent']]
            for position,node in enumerate(group):
                if node is target:group[position:position+1]=children;break
            total=partition_total()
        attempt=dict(interval_visits=budget['visited'],interval_limit=budget['limit'],
            subband_evaluations=evaluations,
            summed_error_bounds=None if total is None else tuple(total))
        if total is None or np.any(total>=caps):return None,attempt
        return groups,attempt

    def _partition_root_subbands(self,prepared,brackets,groups,stations):
        """Replace subdivided bracket bands with their certified sub-bands."""
        rebuilt=[];rows=[];position=0
        for cell,bands in prepared:
            fresh=[]
            for band in bands:
                if not band[2]:
                    fresh.append(band);continue
                group=groups[position];row=brackets[position];position+=1
                if len(group)==1:
                    fresh.append(band);rows.append(row);continue
                for node in group:
                    fresh.append((node['lo'],node['hi'],True,node['features']))
                    stations[cell,node['lo'],node['hi']]=node['station']
                    row=dict(cell=cell,lower=node['lo'],upper=node['hi'],
                        features=node['features'],representation_gap_m=node['pad'],
                        active_angular_width_bound=node['active'],
                        angular_fallback=node['fallback'],representative_xi=node['station'],
                        error_bounds=tuple(node['error']))
                    if node['refinement'] is not None:
                        row['enclosure_refinement']=node['refinement']
                    rows.append(row)
            rebuilt.append((cell,fresh))
        return rebuilt,rows

    def certify_path(self,start,translations,angular,*,gap_tolerance_m,
                     maximum_intervals,cancellation_token=None):
        """Disposable whole-patch control; no endpoint state is committed here."""
        if sha(self.definition())!=self.identity:
            raise ValueError('curved material plane definition/settings changed')
        return curved_patch_path_control(self.rule,start,translations,angular,
            self.origin,self.normal,gap_tolerance_m=gap_tolerance_m,
            maximum_intervals=maximum_intervals,cancellation_token=cancellation_token)
