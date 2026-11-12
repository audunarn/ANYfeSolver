"""Development moving-front geometry for straight retained round surfaces.

The unnormalised plane gap is A(t,angle)+B(t)*xi. Its axial zero is
xi=-A/B, and stationary fronts satisfy A'B-AB'=0. These local values do
not certify complete root isolation or admit a quadrature panel by themselves.
"""
from dataclasses import dataclass
from math import cos,sin
import numpy as np
from .native_surface_path import material_path_pose
from .native_surface_support import _support_path,_base,segment_gap_slab_enclosure


@dataclass(frozen=True)
class AxialContactFront:
    a: float
    b: float
    a_rate: float
    b_rate: float
    a_acceleration: float
    b_acceleration: float
    plane_area: float

    def axial_coordinate(self):
        if self.b==0.:raise ValueError('axial front is singular; angular partition required')
        return self._finite(-self.a/self.b)

    @staticmethod
    def _finite(value):
        if not np.isfinite(value):raise ValueError('nonfinite axial front quotient or derivative')
        return value

    @property
    def stationary_numerator(self):return self._finite(self.a_rate*self.b-self.a*self.b_rate)

    @property
    def stationary_rate(self):return self._finite(self.a_acceleration*self.b-self.a*self.b_acceleration)


def axial_contact_front(field,cell,segment_index,time,angle):
    """Actual path coefficient derivatives; no finite differences or node sampling."""
    rule=field.patch.rule;lo,hi=rule.definition['xi_range']
    xis=(max(lo,-1.+cell),min(hi,float(cell)))
    # Exact field/reference/half-cell validation, also used by interval geometry.
    segment_gap_slab_enclosure(field,cell,xis,rule.definition['angle_range'],segment_index)
    if not np.isfinite(angle) or not rule.definition['angle_range'][0]<=angle<=rule.definition['angle_range'][1]:
        raise ValueError('front angle outside material selection')
    start,segments=_support_path(field,segment_index);segment=segments[0]
    pose=material_path_pose(start,segment.translations,segment.angular,segment.shell_increment,time)
    target=field.patch.target;corners=target.adapter.current_positions(pose.shell_displacement)
    target.physical_pose(pose.shell_displacement)
    velocities=np.asarray(target.adapter.work_map@segment.shell_increment).reshape(3,3)
    e1,e2=corners[1]-corners[0],corners[2]-corners[0]
    v1,v2=velocities[1]-velocities[0],velocities[2]-velocities[0]
    cross=np.cross(e1,e2);rate=np.cross(v1,e2)+np.cross(e1,v2);acceleration=2*np.cross(v1,v2)
    frame=pose.cell_rotations[cell]@field.patch.source.reference.nodal_triads[0]
    arm=rule.definition['radius']*(cos(angle)*frame[:,1]+sin(angle)*frame[:,2])
    omega=segment.angular[cell];arm_rate=np.cross(omega,arm);arm_acceleration=np.cross(omega,arm_rate)
    # xi=0 extrapolation is exact for either straight half-cell.
    base=_base(pose,cell,0.,corners[0])
    chord=_base(pose,cell,float(cell),corners[0])-_base(pose,cell,float(cell-1),corners[0])
    t=1. if cell==0 else 0.
    base_rate=(1-t)*segment.translations[cell]+t*segment.translations[cell+1]-velocities[0]
    chord_rate=segment.translations[cell+1]-segment.translations[cell]
    x=base+arm;xd=base_rate+arm_rate
    values=(cross@x,cross@chord,rate@x+cross@xd,rate@chord+cross@chord_rate,
        acceleration@x+2*rate@xd+cross@arm_acceleration,
        acceleration@chord+2*rate@chord_rate,np.linalg.norm(cross))
    if not np.all(np.isfinite(values)):raise ValueError('nonfinite axial front geometry')
    return AxialContactFront(*map(float,values))


def _front_coefficient_roundoff(coordinates,velocities,edge,edge_rate,c,cd,cdd,x,d,xrate,drate,omega,radius,*,components=False):
    """Propagate uniform binary64 input/operation errors before cancellation.

    Edge subtraction -> cross products -> coefficient dot products. This keeps
    mixed coordinate/relative-distance/velocity products instead of replacing
    them by a maximum. Controls are uniform norms over the declared slab.
    """
    gamma=1024*np.finfo(float).eps
    coordinate_error=gamma*coordinates;velocity_error=gamma*velocities
    ee=2*coordinate_error+gamma*edge;ev=2*velocity_error+gamma*edge_rate
    ec=2*edge*ee+ee*ee+gamma*edge*edge
    ecd=2*(edge*ev+edge_rate*ee+ee*ev+gamma*edge*edge_rate)
    ecdd=2*(2*edge_rate*ev+ev*ev+gamma*edge_rate*edge_rate)
    ex=2*coordinate_error+gamma*(x+radius)
    ed=2*coordinate_error+gamma*d
    exp=2*velocity_error+gamma*(xrate+3*omega*radius)
    edp=2*velocity_error+gamma*drate
    xacc=omega*omega*radius;exacc=4*gamma*xacc
    def product(a,ea,b,eb):
        return ea*b+(a+ea)*eb+gamma*(a+ea)*(b+eb)
    errors=(product(c,ec,x,ex),product(c,ec,d,ed),
        product(cd,ecd,x,ex)+product(c,ec,xrate,exp),
        product(cd,ecd,d,ed)+product(c,ec,drate,edp),
        product(cdd,ecdd,x,ex)+2*product(cd,ecd,xrate,exp)+product(c,ec,xacc,exacc),
        product(cdd,ecdd,d,ed)+2*product(cd,ecd,drate,edp))
    if not np.all(np.isfinite(errors)):raise ValueError('front coefficient roundoff overflow')
    if components:return tuple(float(np.nextafter(e,np.inf)) for e in errors)
    result=max(errors)
    return float(np.nextafter(result,np.inf))


def stationary_front_slab(field,cell,segment_index,angle,left,right,*,include_a=False,angle_bounds=None,_return_error_controls=False):
    """Padded binary64 Taylor bounds for B, N and N' on one time slab.

    C is quadratic, the chord is affine, and the material arm has constant
    angular speed. Product-rule norm bounds give |A'''| and |B'''|. They imply
    |N''| <= |A'''B|+|A''B'|+|A'B''|+|AB'''|. No sampled extrema are used.
    """
    if not 0<=left<=right<=1:raise ValueError('ordered front time slab required')
    angular_width=0.
    if angle_bounds is not None:
        lo,hi=map(float,angle_bounds);domain=field.patch.rule.definition['angle_range']
        if not np.isfinite(lo+hi) or not domain[0]<=lo<=angle<=hi<=domain[1]:
            raise ValueError('stationary angular slab outside material selection')
        angular_width=max(angle-lo,hi-angle)
    start,segments=_support_path(field,segment_index);segment=segments[0]
    target=field.patch.target;middle=(left+right)/2;h=(right-left)/2
    value=axial_contact_front(field,cell,segment_index,middle,angle)
    poses=[material_path_pose(start,segment.translations,segment.angular,segment.shell_increment,t) for t in (left,right)]
    corners=[target.adapter.current_positions(p.shell_displacement) for p in poses]
    velocities=np.asarray(target.adapter.work_map@segment.shell_increment).reshape(3,3)
    da,db=velocities[1]-velocities[0],velocities[2]-velocities[0]
    edges=[(y[1]-y[0],y[2]-y[0]) for y in corners]
    a0,b0=edges[0];a1,b1=edges[1]
    controls=(np.cross(a0,b0),.5*(np.cross(a0,b1)+np.cross(a1,b0)),np.cross(a1,b1))
    reference=target.adapter.reference_positions
    refcross=np.cross(reference[1]-reference[0],reference[2]-reference[0]);refnormal=refcross/np.linalg.norm(refcross)
    c=max(np.linalg.norm(v) for v in controls)
    if min(float(refnormal@v) for v in controls)<=512*np.finfo(float).eps*max(1.,c):
        raise ValueError('stationary front target area unresolved')
    cd=max(np.linalg.norm(np.cross(da,b)+np.cross(a,db)) for a,b in edges)
    cdd=2*np.linalg.norm(np.cross(da,db))
    t=1. if cell==0 else 0.
    velocity=(1-t)*segment.translations[cell]+t*segment.translations[cell+1]-velocities[0]
    chord_velocity=segment.translations[cell+1]-segment.translations[cell]
    omega=np.linalg.norm(segment.angular[cell]);radius=field.patch.rule.definition['radius']
    a3=3*cdd*(np.linalg.norm(velocity)+omega*radius)+3*cd*omega**2*radius+c*omega**3*radius
    b3=3*cdd*np.linalg.norm(chord_velocity)
    xnorm=max(np.linalg.norm(_base(p,cell,0.,y[0])) for p,y in zip(poses,corners))+radius
    dnorm=max(np.linalg.norm(_base(p,cell,float(cell),y[0])-_base(p,cell,float(cell-1),y[0])) for p,y in zip(poses,corners))
    xrate=np.linalg.norm(velocity)+omega*radius;drate=np.linalg.norm(chord_velocity)
    coefficient_scale=max(1.,c*xnorm,c*dnorm,cd*xnorm+c*xrate,cd*dnorm+c*drate,
        cdd*xnorm+2*cd*xrate+c*omega**2*radius,cdd*dnorm+2*cd*drate,
        *(float(np.max(np.abs(y))) for y in corners))
    coefficient_error=512*np.finfo(float).eps*coefficient_scale
    a3+=coefficient_error;b3+=coefficient_error
    a2=abs(value.a_acceleration)+a3*h;b2=abs(value.b_acceleration)+b3*h
    a1bound=abs(value.a_rate)+a2*h;b1bound=abs(value.b_rate)+b2*h
    a_bound=abs(value.a)+a1bound*h;b_bound=abs(value.b)+b1bound*h
    n2=a3*b_bound+a2*b1bound+a1bound*b2+a_bound*b3
    scale=max(1.,c,cd,cdd,a_bound,b_bound,a1bound,b1bound,a2,b2,n2,
        *(float(np.max(np.abs(y))) for y in corners))
    pad=max(coefficient_error,512*np.finfo(float).eps*scale)
    # Propagate coefficient uncertainty and product-rounding BEFORE cancellation.
    nerror=pad*(a1bound+b_bound+a_bound+b1bound)+2*pad*pad
    nerror+=512*np.finfo(float).eps*(a1bound*b_bound+a_bound*b1bound)
    nderror=pad*(a2+b_bound+a_bound+b2)+2*pad*pad
    nderror+=512*np.finfo(float).eps*(a2*b_bound+a_bound*b2)
    n=value.stationary_numerator;nd=value.stationary_rate
    bwidth=abs(value.b_rate)*h+.5*b2*h*h+pad
    nwidth=(abs(nd)+nderror)*h+.5*n2*h*h+nerror
    ndwidth=n2*h+nderror
    # B has no angular dependence. Differentiate only the round-section arm;
    # the following norm bounds hold over the entire time and angular slab.
    a_theta=c*radius
    arate_theta=(cd+c*omega)*radius
    aacc_theta=(cdd+2*cd*omega+c*omega**2)*radius
    nwidth+=(arate_theta*(b_bound+pad)+a_theta*(b1bound+pad))*angular_width
    ndwidth+=(aacc_theta*(b_bound+pad)+a_theta*(b2+pad))*angular_width
    result=(value.b-bwidth,value.b+bwidth,n-nwidth,n+nwidth,nd-ndwidth,nd+ndwidth)
    if not np.all(np.isfinite(result)):raise ValueError('nonfinite stationary front bounds')
    if include_a:
        awidth=abs(value.a_rate)*h+.5*a2*h*h+pad+a_theta*angular_width
        result=(*result,value.a-awidth,value.a+awidth)
    if _return_error_controls:
        # Uniform point-evaluation roundoff controls, not midpoint residuals.
        # Norm bounds cover the complete time slab and every material angle.
        edge_magnitude=max(1.,*(np.linalg.norm(e) for pair in edges for e in pair))
        edge_speed=max(np.linalg.norm(da),np.linalg.norm(db))
        coordinate_magnitude=max(1.,
            *(float(np.max(np.abs(p.positions)))+float(np.max(np.abs(p.position_low))) for p in poses),
            *(float(np.max(np.abs(y))) for y in corners),
            float(np.max(np.abs(target.adapter.reference_positions))),
            float(np.max(np.abs(start.positions)))+float(np.max(np.abs(start.position_low))),
            float(np.max(np.abs(segment.translations))),
            *(float(np.max(abs(target.adapter.work_map)@np.abs(p.shell_displacement))) for p in poses))
        velocity_magnitude=max(float(np.max(np.abs(velocities))),
            float(np.max(np.abs(segment.translations))),
            float(np.max(abs(target.adapter.work_map)@np.abs(segment.shell_increment))))
        ea,eb,eap,ebp,eapp,ebpp=_front_coefficient_roundoff(coordinate_magnitude,velocity_magnitude,
            edge_magnitude,edge_speed,c,cd,cdd,xnorm,dnorm,xrate,drate,omega,radius,components=True)
        # Derivative arithmetic uses actual rate norms. A displacement-scale
        # uncertainty is not an additive uncertainty on every time derivative.
        # It still propagates through C', C'', and all product-rule terms above.
        uniform_error=max(ea,eb)
        def product_error(a,ae,b,be):
            return ae*b+(a+ae)*be+1024*np.finfo(float).eps*(a+ae)*(b+be)
        amag=c*xnorm;bmag=c*dnorm
        apmag=cd*xnorm+c*xrate;bpmag=cd*dnorm+c*drate
        a2mag=cdd*xnorm+2*cd*xrate+c*omega**2*radius
        b2mag=cdd*dnorm+2*cd*drate
        numerator_error=product_error(apmag,eap,bmag,eb)+product_error(amag,ea,bpmag,ebp)
        derivative_error=product_error(a2mag,eapp,bmag,eb)+product_error(amag,ea,b2mag,ebpp)
        if not np.all(np.isfinite((numerator_error,derivative_error,uniform_error))):
            raise ValueError('front point error bound overflow')
        # Rebuild private Taylor controls with separate coefficient errors.
        # The historical no-controls route above remains unchanged.
        a2=abs(value.a_acceleration)+eapp+a3*h;b2=abs(value.b_acceleration)+ebpp+b3*h
        a1bound=abs(value.a_rate)+eap+a2*h;b1bound=abs(value.b_rate)+ebp+b2*h
        a_bound=abs(value.a)+ea+a1bound*h;b_bound=abs(value.b)+eb+b1bound*h
        n2=a3*b_bound+a2*b1bound+a1bound*b2+a_bound*b3
        bwidth=(abs(value.b_rate)+ebp)*h+.5*b2*h*h+eb
        nwidth=(abs(nd)+derivative_error)*h+.5*n2*h*h+numerator_error
        ndwidth=n2*h+derivative_error
        nwidth+=(arate_theta*b_bound+a_theta*b1bound)*angular_width
        ndwidth+=(aacc_theta*b_bound+a_theta*b2)*angular_width
        result=(value.b-bwidth,value.b+bwidth,n-nwidth,n+nwidth,nd-ndwidth,nd+ndwidth)
        angular_a_interval=None
        if angle_bounds is not None:
            # Retain the cancellation in the exact midpoint angular polynomial.
            # Separate product norms can otherwise leave a fixed angular floor
            # which temporal refinement cannot shrink. Uniform Taylor bounds
            # still cover the entire time slab and all selected material angles.
            from .native_surface_front_boundary import stationary_angular_coefficients
            from .native_surface_support import _sinusoidal_extrema
            acoef,apcoef,appcoef,v=stationary_angular_coefficients(field,cell,segment_index,middle,angle)
            q=tuple(p*v.b-x*v.b_rate for x,p in zip(acoef,apcoef))
            qp=tuple(p*v.b-x*v.b_acceleration for x,p in zip(acoef,appcoef))
            qrange=_sinusoidal_extrema((q[0],),q[1],q[2],angle_bounds)
            qprange=_sinusoidal_extrema((qp[0],),qp[1],qp[2],angle_bounds)
            # Three coefficient errors plus arithmetic before cancellation.
            qe=3*numerator_error+2048*np.finfo(float).eps*sum(
                abs(p*v.b)+abs(x*v.b_rate) for x,p in zip(acoef,apcoef))
            qpe=3*derivative_error+2048*np.finfo(float).eps*sum(
                abs(p*v.b)+abs(x*v.b_acceleration) for x,p in zip(acoef,appcoef))
            magnitude=lambda co:abs(co[0])+float(np.hypot(co[1],co[2]))
            aa2=magnitude(appcoef)+3*eapp+a3*h
            aa1=magnitude(apcoef)+3*eap+aa2*h
            aa=magnitude(acoef)+3*ea+aa1*h
            nn2=a3*b_bound+aa2*b1bound+aa1*b2+aa*b3
            arange=_sinusoidal_extrema((acoef[0],),acoef[1],acoef[2],angle_bounds)
            adelta=(magnitude(apcoef)+3*eap)*h+.5*aa2*h*h+3*ea
            angular_a_interval=(float(np.nextafter(arange[0]-adelta,-np.inf)),
                                float(np.nextafter(arange[1]+adelta,np.inf)))
            delta=max(abs(qprange[0]),abs(qprange[1]))*h+qpe*h+.5*nn2*h*h+qe
            ddelta=nn2*h+qpe
            result=(result[0],result[1],
                max(result[2],float(np.nextafter(qrange[0]-delta,-np.inf))),
                min(result[3],float(np.nextafter(qrange[1]+delta,np.inf))),
                max(result[4],float(np.nextafter(qprange[0]-ddelta,-np.inf))),
                min(result[5],float(np.nextafter(qprange[1]+ddelta,np.inf))))
            if result[2]>result[3] or result[4]>result[5]:
                raise ValueError('inconsistent stationary angular Taylor bounds')
        if include_a:
            awidth=(abs(value.a_rate)+eap)*h+.5*a2*h*h+ea+a_theta*angular_width
            lower,upper=value.a-awidth,value.a+awidth
            if angular_a_interval is not None:
                lower=max(lower,angular_a_interval[0]);upper=min(upper,angular_a_interval[1])
                if lower>upper:raise ValueError('inconsistent stationary angular gap bounds')
            result=(*result,lower,upper)
        return result,(float(uniform_error),float(numerator_error))
    return result


@dataclass(frozen=True)
class StationaryAngularCover:
    field_identity: str
    cell: int
    segment_index: int
    angle_bounds: tuple
    branches: tuple
    xi_bounds: tuple|None=None
    open_time_slabs: tuple=()
    front_components: tuple=()
    front_time_slabs: tuple=()
    branch_bounds: tuple=()
    simple_time_slabs: tuple=()
    variable_branches: tuple=()

    def __iter__(self):return iter(self.branches)
    def __len__(self):return len(self.branches)
    def __bool__(self):return bool(self.branches)


class StationaryAngularSubdivision(ValueError):
    """The fixed angular enclosure needs children; it certifies no support."""


def stationary_front_angular_cover(field,cell,segment_index,angle_bounds,*,budget,xi_bounds=None,
        endpoint_partition=None,_variable_endpoint_branches=False,_allow_angular_subdivision=False):
    """Certify stationary-root branches over an entire angular interval.

    Each returned time bracket contains exactly one simple stationary root for
    every angle. All remaining time slabs exclude roots. Angular bands where
    roots enter/leave a time endpoint remain explicit non-success and require
    boundary partitioning. Optional xi_bounds exclude only whole-domain strictly
    open slabs before requiring B!=0. Such covers count contact-relevant roots;
    their recorded open and H-valid slabs cover the complete time interval.
    This does not itself admit friction integration or replace orientation checks.
    """
    from .native_surface_support import SupportIntervalBudget
    if type(budget) is not SupportIntervalBudget:raise TypeError('shared support budget required')
    lo,hi=map(float,angle_bounds);angle=(lo+hi)/2
    if endpoint_partition is not None:
        if xi_bounds is None:raise ValueError('endpoint partition requires bound axial domain')
        from .native_surface_front_boundary import endpoint_numerator_range
        for time in (0.,1.):endpoint_numerator_range(endpoint_partition,field,cell,segment_index,time,(lo,hi))
    if _variable_endpoint_branches and endpoint_partition is None:
        raise ValueError('variable endpoint branches require bound endpoint angular partition')
    if xi_bounds is not None:
        xi_bounds=tuple(map(float,xi_bounds));domain=field.patch.rule.definition['xi_range']
        if (len(xi_bounds)!=2 or not all(np.isfinite(xi_bounds)) or
            not max(domain[0],cell-1)<=xi_bounds[0]<xi_bounds[1]<=min(domain[1],cell)):
            raise ValueError('stationary cover axial domain outside selected half-cell')
    pending=[(0.,1.)];branches=[];cache={};open_slabs=[];front_slabs=[];simple_slabs=[]
    def enclosure(left,right):
        if (left,right) not in cache:
            if not budget.consume():raise MemoryError('stationary angular cover interval cap exhausted')
            if xi_bounds is None:
                cache[left,right]=stationary_front_slab(field,cell,segment_index,angle,left,right,
                    angle_bounds=(lo,hi))
            else:
                cache[left,right]=stationary_front_slab(field,cell,segment_index,angle,left,right,
                    angle_bounds=(lo,hi),include_a=True,_return_error_controls=True)[0]
            if endpoint_partition is not None and left==right and left in (0.,1.):
                # A captured endpoint has exact sine/cosine dependence on the
                # material angle. Reuse its complete analytic range, rather
                # than a norm-only angular Lipschitz allowance. This never
                # resolves an actual endpoint zero or omits its guard band.
                from .native_surface_front_boundary import endpoint_numerator_range
                nl,nu=endpoint_numerator_range(endpoint_partition,field,cell,segment_index,left,(lo,hi))
                bounds=cache[left,right]
                cache[left,right]=(*bounds[:2],nl,nu,*bounds[4:])
        return cache[left,right]
    while pending:
        left,right=pending.pop();bounds=enclosure(left,right);bl,bu,nl,nu,dl,du=bounds[:6]
        if xi_bounds is not None:
            products=[float(np.nextafter(b*x,-np.inf)) for b in (bl,bu) for x in xi_bounds]
            lower=float(np.nextafter(bounds[6]+min(products),-np.inf))
            if lower>0:
                open_slabs.append((left,right,lower))
                continue
        if bl>0 or bu<0:
            if xi_bounds is not None and (dl>0 or du<0):
                # A root may cross an arbitrary subdivision endpoint as the
                # material angle varies. Classify the union of adjacent strict
                # N' slabs below, rather than dropping or rejecting that root.
                front_slabs.append((left,right,bl,bu))
                simple_slabs.append((left,right,bl,bu,dl,du))
                continue
            if nl>0 or nu<0:
                front_slabs.append((left,right,bl,bu))
                continue
            if dl>0 or du<0:
                endpoint=[enclosure(t,t)[2:4] for t in (left,right)]
                signs=[1 if low>0 else -1 if high<0 else 0 for low,high in endpoint]
                if 0 in signs:raise ValueError('stationary root angular endpoint transition unresolved')
                if signs[0]!=signs[1]:branches.append((left,right))
                front_slabs.append((left,right,bl,bu))
                continue
        middle=(left+right)/2
        if _allow_angular_subdivision and xi_bounds is not None:
            # A fixed-angle enclosure can resolve N or N' at this time, while
            # the whole band cannot. Time refinement cannot reduce that angular
            # contribution at the same time. Yield only a refinement request,
            # never support: the caller must represent both angular children.
            angular=enclosure(middle,middle)
            products=[float(np.nextafter(b*x,-np.inf))
                      for b in angular[:2] for x in xi_bounds]
            open_lower=float(np.nextafter(angular[6]+min(products),-np.inf))
            if (open_lower<=0. and angular[2]<=0.<=angular[3]
                    and angular[4]<=0.<=angular[5]):
                if not budget.consume():
                    raise MemoryError('stationary angular cover interval cap exhausted')
                point=stationary_front_slab(field,cell,segment_index,angle,middle,middle,
                    include_a=True,_return_error_controls=True)[0]
                if ((point[0]>0. or point[1]<0.) and
                        (point[2]>0. or point[3]<0. or point[4]>0. or point[5]<0.)):
                    raise StationaryAngularSubdivision(
                        'stationary angular cover requires angular subdivision')
        if not left<middle<right:raise ValueError('stationary angular cover temporal resolution exhausted')
        pending.extend(((middle,right),(left,middle)))
    if xi_bounds is None:
        return StationaryAngularCover(field.identity,cell,segment_index,(lo,hi),tuple(sorted(branches)))
    simple_components=[]
    for left,right,bl,bu,dl,du in sorted(simple_slabs):
        if (simple_components and simple_components[-1][1]==left and
            (simple_components[-1][2]>0)==(bl>0) and (simple_components[-1][4]>0)==(dl>0)):
            old=simple_components.pop()
            simple_components.append((old[0],right,min(old[2],bl),max(old[3],bu),
                min(old[4],dl),max(old[5],du)))
        else:simple_components.append((left,right,bl,bu,dl,du))
    branch_bounds=[];variable_branches=[]
    for left,right,*bounds in simple_components:
        endpoint=[enclosure(t,t)[2:4] for t in (left,right)]
        signs=[1 if low>0 else -1 if high<0 else 0 for low,high in endpoint]
        if 0 in signs:
            if not _variable_endpoint_branches:
                raise ValueError('stationary root angular endpoint transition unresolved')
            # Strict N' certifies at most one root at every material angle.
            # Keep the potential branch even when its root enters/leaves the
            # actual time interval; prepared queries bound its clamped extremum.
            branches.append((left,right));branch_bounds.append(tuple(bounds));variable_branches.append(True)
            continue
        if signs[0]!=signs[1]:
            branches.append((left,right));branch_bounds.append(tuple(bounds));variable_branches.append(False)
    cursor=0.
    for left,right in sorted([(a,b) for a,b,_ in open_slabs]+[(a,b) for a,b,_,_ in front_slabs]):
        if left!=cursor or not left<right:raise ValueError('incomplete stationary temporal cover')
        cursor=right
    if cursor!=1.:raise ValueError('incomplete stationary temporal cover')
    components=[]
    for left,right,bl,bu in sorted(front_slabs):
        if components and components[-1][1]==left and (components[-1][2]>0)==(bl>0):
            old=components.pop();components.append((old[0],right,min(old[2],bl),max(old[3],bu)))
        else:components.append((left,right,bl,bu))
    return StationaryAngularCover(field.identity,cell,segment_index,(lo,hi),tuple(sorted(branches)),
        xi_bounds,tuple(sorted(open_slabs)),tuple(components),tuple(sorted(front_slabs)),
        tuple(branch_bounds),tuple(sorted(simple_slabs)),tuple(variable_branches))


def _front_components(cover):
    return ((0.,1.,None,None),) if cover.xi_bounds is None else cover.front_components


@dataclass(frozen=True)
class IsolatedStationaryFront:
    fraction: float
    axial_coordinate: float
    time_bracket: tuple
    axial_interval: tuple
    numerator_residual: float
    kind: str='stationary'


def stationary_front_partition(field,cell,segment_index,angle,*,maximum_intervals=128,maximum_axial_uncertainty=None,budget=None,angular_cover=None):
    """Complete bounded time isolation, returning endpoint and stationary fronts.

    Singular B or unresolved multiple stationary roots cause explicit non-success.
    This partitions one fixed angle only; angular integration remains separate.
    """
    from scipy.optimize import brentq
    if type(maximum_intervals) is not int or maximum_intervals<1:
        raise ValueError('positive stationary-front interval cap required')
    if maximum_axial_uncertainty is not None and (not np.isfinite(maximum_axial_uncertainty) or maximum_axial_uncertainty<=0):
        raise ValueError('positive dimensionless axial uncertainty required')
    if budget is not None:
        from .native_surface_support import SupportIntervalBudget
        if type(budget) is not SupportIntervalBudget:raise TypeError('shared support budget required')
    cache={}
    def at(t):
        if t not in cache:cache[t]=axial_contact_front(field,cell,segment_index,t,angle)
        return cache[t]
    if angular_cover is not None:
        if (type(angular_cover) is not StationaryAngularCover or angular_cover.field_identity!=field.identity
                or angular_cover.cell!=cell or angular_cover.segment_index!=segment_index
                or not angular_cover.angle_bounds[0]<=angle<=angular_cover.angle_bounds[1]):
            raise ValueError('stationary angular cover identity mismatch')
        if angular_cover.xi_bounds is not None:
            raise ValueError('domain-bound temporal cover requires prepared front query')
        pending=list(angular_cover.branches)
    else:pending=[(0.,1.)]
    roots={};count=0;slabs={}
    def enclose(left,right):
        nonlocal count
        key=(left,right)
        if key not in slabs:
            if count>=maximum_intervals:raise MemoryError('stationary-front interval cap exhausted')
            if budget is not None and not budget.consume():raise MemoryError('stationary-front shared interval cap exhausted')
            count+=1
            slabs[key]=stationary_front_slab(field,cell,segment_index,angle,left,right,include_a=True)
        return slabs[key]
    while pending:
        left,right=pending.pop()
        bl,bu,nl,nu,dl,du,_,_=enclose(left,right)
        if bl>0. or bu<0.:
            if nl>0. or nu<0.:continue
            if dl>0. or du<0.:
                ends=[enclose(t,t)[2:4] for t in (left,right)]
                signs=[1 if low>0 else -1 if high<0 else 0 for low,high in ends]
                if 0 in signs:
                    raise ValueError('stationary front endpoint sign unresolved')
                if signs[0]!=signs[1]:
                    root=brentq(lambda t:at(t).stationary_numerator,left,right,
                        xtol=4*np.finfo(float).eps,rtol=4*np.finfo(float).eps,maxiter=64)
                    value=at(root)
                    root_bounds=enclose(root,root)
                    rlo,rhi=root_bounds[2:4]
                    time_error=max(abs(rlo),abs(rhi))/min(abs(dl),abs(du))
                    lower=max(left,float(np.nextafter(root-time_error,-np.inf)))
                    upper=min(right,float(np.nextafter(root+time_error,np.inf)))
                    bounds=enclose(lower,upper)
                    denominator=bounds[:2]
                    if denominator[0]<=0<=denominator[1]:
                        raise ValueError('stationary front denominator uncertainty unresolved')
                    # Correlated front motion: H'=-N/B^2. Separately enclosing
                    # A/B over time destroys cancellation near a stationary front.
                    point_b=(max(root_bounds[0],denominator[0],bl),min(root_bounds[1],denominator[1],bu))
                    if point_b[0]>point_b[1] or point_b[0]<=0<=point_b[1]:
                        raise ValueError('stationary front denominator intersection unresolved')
                    ratios=[-a/b for a in root_bounds[6:] for b in point_b]
                    motion_error=max(abs(bounds[2]),abs(bounds[3]))/min(abs(bl),abs(bu))**2*max(root-lower,upper-root)
                    interval=(float(np.nextafter(min(ratios)-motion_error,-np.inf)),
                        float(np.nextafter(max(ratios)+motion_error,np.inf)))
                    if not np.all(np.isfinite(interval)):
                        raise ValueError('stationary front coordinate uncertainty overflow')
                    if maximum_axial_uncertainty is not None and interval[1]-interval[0]>maximum_axial_uncertainty:
                        raise ValueError('stationary front axial uncertainty exceeds budget')
                    roots[float(root)]=IsolatedStationaryFront(float(root),value.axial_coordinate(),
                        (lower,upper),interval,value.stationary_numerator)
                # Strict monotonicity plus certified equal endpoint signs excludes roots.
                continue
        middle=(left+right)/2
        if not left<middle<right:raise ValueError('stationary-front time resolution exhausted')
        pending.extend(((middle,right),(left,middle)))
    fractions=tuple(sorted({0.,1.,*roots}))
    endpoints=[]
    for t in (0.,1.):
        bounds=enclose(t,t)
        if bounds[0]<=0<=bounds[1]:raise ValueError('endpoint front denominator unresolved')
        ratios=[-a/b for a in bounds[6:] for b in bounds[:2]]
        interval=(float(np.nextafter(min(ratios),-np.inf)),float(np.nextafter(max(ratios),np.inf)))
        if not np.all(np.isfinite(interval)):raise ValueError('endpoint front uncertainty overflow')
        if maximum_axial_uncertainty is not None and interval[1]-interval[0]>maximum_axial_uncertainty:
            raise ValueError('endpoint front axial uncertainty exceeds budget')
        endpoints.append(IsolatedStationaryFront(t,at(t).axial_coordinate(),(t,t),interval,at(t).stationary_numerator))
    return tuple((t,at(t).axial_coordinate()) for t in fractions),count,tuple(roots[t] for t in sorted(roots)),tuple(endpoints)


def stationary_front_angular_envelopes(field,cover,*,maximum_axial_uncertainty,budget,prepared=None,
        _include_endpoints=False):
    """Enclose every stationary uncertainty strip over a certified angular band.

    On a simple root branch H_t=0, so dH*/dangle=-A_angle/B. The normal
    cross product and round radius bound A_angle; B is angle independent.
    Enlarge by the full declared fixed-angle uncertainty to include strips,
    not just exact roots. Returned intervals are not contact history states.
    """
    if type(cover) is not StationaryAngularCover or cover.field_identity!=field.identity:
        raise ValueError('bound stationary angular cover required')
    lo,hi=cover.angle_bounds;middle=(lo+hi)/2
    if cover.xi_bounds is None:
        _,_,roots,endpoints=stationary_front_partition(field,cover.cell,cover.segment_index,middle,
            maximum_intervals=field.maximum_intervals,maximum_axial_uncertainty=maximum_axial_uncertainty,
            budget=budget,angular_cover=cover)
    else:
        if prepared is None:prepared=prepare_stationary_band(field,cover,budget=budget)
        if prepared.cover!=cover:raise ValueError('stationary envelope prepared cover mismatch')
        _,_,roots,endpoints=stationary_front_partition_prepared(field,prepared,middle,
            maximum_axial_uncertainty=maximum_axial_uncertainty)
    if len(roots)!=len(cover):raise ValueError('stationary branch envelope count mismatch')
    if _include_endpoints and cover.xi_bounds is None:
        raise ValueError('complete history envelopes require a bound prepared domain')
    start,segments=_support_path(field,cover.segment_index);motion=segments[0]
    radius=field.patch.rule.definition['radius'];intervals=[]
    entries=[(root,left,right,prepared.branches[i][2] if cover.xi_bounds is not None else None)
        for i,(root,(left,right)) in enumerate(zip(roots,cover.branches))]
    if _include_endpoints:
        entries.extend((root,time,time,bounds) for root,(time,_,bounds,_) in zip(endpoints,prepared.endpoints))
    for root,left,right,bounds in entries:
        if not left<=root.fraction<=right:raise ValueError('stationary branch envelope ownership mismatch')
        if not budget.consume():raise MemoryError('stationary envelope support interval cap exhausted')
        if bounds is None:bounds=stationary_front_slab(field,cover.cell,cover.segment_index,middle,left,right)
        if bounds[0]<=0<=bounds[1]:raise ValueError('stationary envelope denominator unresolved')
        denominator=min(abs(bounds[0]),abs(bounds[1]))
        poses=[material_path_pose(start,motion.translations,motion.angular,motion.shell_increment,t)
            for t in (left,right)]
        corners=[field.patch.target.adapter.current_positions(p.shell_displacement) for p in poses]
        (a,b),(c,d)=[(y[1]-y[0],y[2]-y[0]) for y in corners]
        controls=(np.cross(a,b),.5*(np.cross(a,d)+np.cross(c,b)),np.cross(c,d))
        coordinate_scale=max(1.,*(float(np.max(np.abs(y))) for y in corners))
        edge_scale=max(1.,*(np.linalg.norm(e) for e in (a,b,c,d)))
        coordinate_error=1024*np.finfo(float).eps*coordinate_scale
        area_error=1024*np.finfo(float).eps*edge_scale**2+8*coordinate_error*edge_scale+16*coordinate_error**2
        cross_upper=max(np.linalg.norm(v) for v in controls)+area_error
        speed=cross_upper*radius/denominator
        pad=1024*np.finfo(float).eps*max(1.,speed,abs(root.axial_interval[0]),abs(root.axial_interval[1]))
        width=(speed+pad)*(hi-lo)/2+maximum_axial_uncertainty+pad
        interval=(float(np.nextafter(root.axial_interval[0]-width,-np.inf)),
                  float(np.nextafter(root.axial_interval[1]+width,np.inf)))
        if not np.all(np.isfinite(interval)):raise ValueError('stationary angular envelope overflow')
        intervals.append(interval)
    return tuple(intervals)


@dataclass(frozen=True)
class PreparedStationaryBand:
    cover: StationaryAngularCover
    branches: tuple
    endpoints: tuple
    prepared_intervals: int


def prepare_stationary_band(field,cover,*,budget,_reuse_endpoints=None):
    """Prepare uniform geometric/error bounds once for a certified angle band.

    A containing band's prepared endpoint controls may be reused verbatim:
    B has no angular dependence and endpoint coordinate intervals intersect
    the exact point values, so wider angular padding cannot widen them.
    Branch controls are always recomputed for this band's angular width.
    """
    from .native_surface_support import SupportIntervalBudget
    if type(budget) is not SupportIntervalBudget:raise TypeError('shared support budget required')
    if type(cover) is not StationaryAngularCover or cover.field_identity!=field.identity:
        raise ValueError('bound stationary angular cover required')
    middle=sum(cover.angle_bounds)/2;before=budget.intervals
    def controls(left,right,component_bounds=None,branch_bounds=None):
        if not budget.consume():raise MemoryError('prepared stationary band interval cap exhausted')
        bounds,errors=stationary_front_slab(field,cover.cell,cover.segment_index,middle,left,right,
            angle_bounds=cover.angle_bounds,_return_error_controls=True)
        if component_bounds is not None:
            bounds=(max(bounds[0],component_bounds[0]),min(bounds[1],component_bounds[1]),*bounds[2:])
            if bounds[0]>bounds[1]:raise ValueError('prepared stationary component bound mismatch')
        if branch_bounds is not None:
            bounds=(*bounds[:4],max(bounds[4],branch_bounds[2]),min(bounds[5],branch_bounds[3]))
            if bounds[4]>bounds[5]:raise ValueError('prepared stationary derivative bound mismatch')
        if bounds[0]<=0<=bounds[1]:raise ValueError('prepared stationary denominator unresolved')
        return (float(left),float(right),tuple(bounds),tuple(errors))
    if cover.branch_bounds and len(cover.branch_bounds)!=len(cover.branches):
        raise ValueError('prepared stationary branch bound count mismatch')
    branches=tuple(controls(a,b,None if not cover.branch_bounds else cover.branch_bounds[i][:2],
        None if not cover.branch_bounds else cover.branch_bounds[i]) for i,(a,b) in enumerate(cover.branches))
    for _,_,bounds,_ in branches:
        if bounds[4]<=0<=bounds[5]:raise ValueError('prepared stationary simple branch unresolved')
    expected=tuple((float(t),float(t))
        for left,right,_,_ in _front_components(cover) for t in (left,right))
    if _reuse_endpoints is not None and (len(_reuse_endpoints)!=len(expected)
            or tuple((entry[0],entry[1]) for entry in _reuse_endpoints)!=expected):
        _reuse_endpoints=None
    endpoints=(tuple(_reuse_endpoints) if _reuse_endpoints is not None else
        tuple(controls(t,t,None if bl is None else (bl,bu))
            for left,right,bl,bu in _front_components(cover) for t in (left,right)))
    return PreparedStationaryBand(cover,branches,endpoints,budget.intervals-before)


def stationary_front_partition_prepared(field,prepared,angle,*,maximum_axial_uncertainty):
    """Query a proved band without recomputing geometric slabs.

    Actual scalar coefficients and roots are still evaluated. Only uniform
    time/angle bounds are reused; no newly computed interval is hidden from
    the shared budget. Root and endpoint coordinate uncertainties still gate.
    """
    from scipy.optimize import brentq
    from anyfesolver._ge_beam3_p5_seeded.core import sha
    if (type(prepared) is not PreparedStationaryBand or
        prepared.cover.field_identity!=field.identity or sha(field._definition())!=field.identity or
        not prepared.cover.angle_bounds[0]<=angle<=prepared.cover.angle_bounds[1]):
        raise ValueError('prepared stationary band identity/angle mismatch')
    if not np.isfinite(maximum_axial_uncertainty) or maximum_axial_uncertainty<=0:
        raise ValueError('positive dimensionless axial uncertainty required')
    cover=prepared.cover;cache={}
    def at(t):
        if t not in cache:cache[t]=axial_contact_front(field,cover.cell,cover.segment_index,t,angle)
        return cache[t]
    def coordinate(value,bounds,error,motion_error=0.):
        denominator=(max(bounds[0],value.b-error),min(bounds[1],value.b+error))
        if denominator[0]>denominator[1] or denominator[0]<=0<=denominator[1]:
            raise ValueError('prepared front point denominator unresolved')
        ratios=[-a/b for a in (value.a-error,value.a+error) for b in denominator]
        interval=(float(np.nextafter(min(ratios)-motion_error,-np.inf)),
                  float(np.nextafter(max(ratios)+motion_error,np.inf)))
        if not np.all(np.isfinite(interval)) or interval[1]-interval[0]>maximum_axial_uncertainty:
            raise ValueError('prepared stationary axial uncertainty exceeds budget')
        return interval
    roots=[]
    if cover.variable_branches and len(cover.variable_branches)!=len(prepared.branches):
        raise ValueError('prepared variable branch count mismatch')
    for index,(left,right,bounds,(error,nerror)) in enumerate(prepared.branches):
        nl,nr=at(left).stationary_numerator,at(right).stationary_numerator
        variable=bool(cover.variable_branches and cover.variable_branches[index])
        if variable and (abs(nl)<=nerror or abs(nr)<=nerror or np.sign(nl)==np.sign(nr)):
            uncertain=abs(nl)<=nerror or abs(nr)<=nerror
            if uncertain:root=left if abs(nl)<=abs(nr) else right
            else:
                # Reliable endpoint and N' signs locate the external zero.
                # Comparing rounded |N| magnitudes can reverse their order
                # within the supplied error even when both signs are strict.
                root=left if (nl>0)==(bounds[4]>0) else right
            value=at(root)
            minimum_rate=min(abs(bounds[4]),abs(bounds[5]));maximum_rate=max(abs(bounds[4]),abs(bounds[5]))
            residual=abs(value.stationary_numerator)+nerror if uncertain else 0.
            time_error=residual/minimum_rate
            bracket=(max(left,float(np.nextafter(root-time_error,-np.inf))),
                min(right,float(np.nextafter(root+time_error,np.inf)))) if uncertain else (root,root)
            minimum_b=min(abs(bounds[0]),abs(bounds[1]))
            motion=(residual*time_error+.5*maximum_rate*time_error**2)/minimum_b**2
            interval=coordinate(value,bounds,error,motion)
            roots.append(IsolatedStationaryFront(root,value.axial_coordinate(),bracket,interval,
                value.stationary_numerator,'boundary_uncertain' if uncertain else 'clamped_endpoint'))
            continue
        if abs(nl)<=nerror or abs(nr)<=nerror or np.sign(nl)==np.sign(nr):
            raise ValueError('prepared stationary endpoint sign unresolved')
        root=brentq(lambda t:at(t).stationary_numerator,left,right,
            xtol=4*np.finfo(float).eps,rtol=4*np.finfo(float).eps,maxiter=64)
        if not np.isfinite(root) or not left<=root<=right:
            raise ValueError('prepared stationary root outside certified bracket')
        value=at(root);residual=abs(value.stationary_numerator)+nerror
        minimum_rate=min(abs(bounds[4]),abs(bounds[5]));maximum_rate=max(abs(bounds[4]),abs(bounds[5]))
        time_error=residual/minimum_rate
        bracket=(max(left,float(np.nextafter(root-time_error,-np.inf))),
                 min(right,float(np.nextafter(root+time_error,np.inf))))
        minimum_b=min(abs(bounds[0]),abs(bounds[1]))
        motion=(residual*time_error+.5*maximum_rate*time_error**2)/minimum_b**2
        interval=coordinate(value,bounds,error,motion)
        roots.append(IsolatedStationaryFront(float(root),value.axial_coordinate(),bracket,interval,value.stationary_numerator))
    endpoints=[]
    for time,_,bounds,(error,_) in prepared.endpoints:
        value=at(time)
        endpoints.append(IsolatedStationaryFront(time,value.axial_coordinate(),(time,time),
            coordinate(value,bounds,error),value.stationary_numerator))
    fractions=tuple(sorted({*(e.fraction for e in endpoints),*(r.fraction for r in roots)}))
    return tuple((t,at(t).axial_coordinate()) for t in fractions),0,tuple(roots),tuple(endpoints)


def prepared_front_open_interval(prepared,partition,left,right):
    """Whole-time open certificate for the interior of an axial interval.

    The complete band cover excludes B=0. H extrema are exactly its time
    endpoints and certified stationary branches; use their outward intervals.
    No sampled gap or endpoint-only contact assumption enters this test.
    """
    if type(prepared) is not PreparedStationaryBand or not left<right:
        raise ValueError('prepared front and ordered axial interval required')
    cover=prepared.cover
    if cover.xi_bounds is not None and not cover.xi_bounds[0]<=left<right<=cover.xi_bounds[1]:
        raise ValueError('prepared open query outside bound axial domain')
    components=_front_components(cover)
    if len(partition[2])!=len(prepared.branches) or len(partition[3])!=2*len(components):
        raise ValueError('complete prepared front extrema required')
    # Quadrature uses strictly interior material points. Equality at an
    # interval edge may be zero gap but cannot produce interior contact work.
    for i,(start,end,_,_) in enumerate(components):
        endpoints=partition[3][2*i:2*i+2]
        if (endpoints[0].fraction,endpoints[1].fraction)!=(start,end):
            raise ValueError('prepared component endpoint ownership mismatch')
        extrema=(*endpoints,*(r for r in partition[2] if start<=r.fraction<=end))
        lower=min(e.axial_interval[0] for e in extrema);upper=max(e.axial_interval[1] for e in extrema)
        b0=prepared.endpoints[2*i][2][:2]
        if not (left>=upper if b0[0]>0 else right<=lower if b0[1]<0 else False):return False
    return True
