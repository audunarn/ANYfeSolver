"""Local curved material-patch work against a wholly containing physical S3.

Reuses curved axial/angular integration. All moving-normal derivatives and
authoritative offset columns remain. The separately profiled retained owner
uses whole-path controls; finite overlap, rims and friction are not represented.
"""
from math import fsum,sqrt,sin,cos,pi
from fractions import Fraction
import numpy as np
from scipy import sparse

from anyfesolver._ge_beam3_mixed_ad import Jet2,dot,sqrt as jet_sqrt
from anyfesolver._ge_beam3_deterministic_reduction import deterministic_dot, deterministic_norm
from .native_replay_arithmetic import gap_matrix_product, gap_vector_product
from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver._ge_beam3_p5_seeded.core import sha
from .native_curved_plane_work import NativeCurvedMaterialPlaneWork,_station_field,_critical_gaps
from .native_curved_geometry import (curved_patch_geometry_control,
    _reference_norm_bounds,_reference_second_bound,_reference_curvature_norm)
from .native_surface_bounds import _reference_radial_gain,_rotation_gain
from .native_surface_normal import _cross
from .native_material_s3 import (NativeRetainedMaterialS3Work,
    NativeMaterialS3PhysicalResult,NativeMaterialS3WorkResult)
from .native_retained_station import _readonly
from .native_curved_geometry import curved_patch_path_control,CurvedPatchPathControl

PROFILE='NATIVE_RETAINED_CURVED_MATERIAL_S3_LOCAL_WORK_DEV_V1'


def _declared_arc_gradient_bound(base,circular,angle_range):
    """Enclose ||base+B(cos a,sin a)|| on the declared material arc.

    For |a-m|<=delta<=pi/2, rotation about the arc midpoint gives
    B u (cos(a-m)-1)+B v sin(a-m). The triangle inequality bounds
    both terms. Wider arcs keep the existing whole-circle enclosure.
    Add roundoff using uncancelled magnitudes, including when base+B u
    nearly cancels. This bound does not infer an arc from active measure.
    """
    base=np.asarray(base);circular=np.asarray(circular)
    magnitude=np.linalg.norm(base)+np.linalg.norm(circular,2)
    lo,hi=angle_range;delta=(hi-lo)/2
    if delta>pi/2:return float(magnitude)
    midpoint=lo+delta
    u=np.array((cos(midpoint),sin(midpoint)));v=np.array((-u[1],u[0]))
    radial=circular@u;tangential=circular@v
    # 2 sin(delta/2)^2 avoids loss of the small 1-cos(delta) term.
    arc=(np.linalg.norm(base+radial)+sin(delta)*np.linalg.norm(tangential)
         +2*sin(delta/2)**2*np.linalg.norm(radial))
    roundoff=8192*np.finfo(float).eps*(np.linalg.norm(base)
        +np.linalg.norm(circular[:,0])+np.linalg.norm(circular[:,1]))
    return float(np.nextafter(min(magnitude,arc+roundoff),np.inf))


def _psd_root_tangent_difference_bound(leading,remainder,active):
    """Infinity bound for two PSD leading integrals plus general remainders.

    If 0<=A,B<=leading*I, their difference has spectral norm<=leading.
    Each indefinite/nonsymmetric remainder has norm<=remainder, so their
    difference still needs 2*remainder. sqrt(active) converts to infinity
    norm on the complete supported coordinate space. Caller adds roundoff.
    """
    return sqrt(active)*(leading+2*remainder)


def _component_root_leading_difference_bound(factor,bounds):
    """Two positive integrals, each with mass<=factor and |g_i|<=bounds_i."""
    bounds=np.asarray(bounds)
    return float(np.nextafter(2*factor*np.max(bounds)*fsum(bounds),np.inf))


def _anchored_root_leading_difference_bound(factor,anchor,variation):
    """Common fixed gradient plus whole-support variation; both masses<=factor."""
    vi=float(np.max(np.abs(anchor)));v1=fsum(np.abs(anchor))
    ei=float(np.max(variation));e1=fsum(variation)
    bound=factor*(vi*v1+2*(vi*e1+ei*v1+ei*e1))
    return float(np.nextafter(bound*(1+8192*np.finfo(float).eps),np.inf))


def _root_component_gradient_bounds(gradients,angle_range,axial_extension,supported,*,anchor=None):
    base=gradients[0]
    circular=np.column_stack((gradients[1]-base,gradients[2]-base))
    bounds=np.zeros(len(base))
    for i in np.flatnonzero(supported):
        centre=base[i:i+1] if anchor is None else base[i:i+1]-anchor[i:i+1]
        angular=_declared_arc_gradient_bound(centre,circular[i:i+1],angle_range)
        rounding=8192*np.finfo(float).eps*fsum(abs(g[i]) for g in gradients)
        if anchor is not None:rounding+=8192*np.finfo(float).eps*abs(anchor[i])
        bounds[i]=np.nextafter(angular+axial_extension+rounding,np.inf)
    return bounds


def _normal_derivatives(corners):
    q=[Jet2.variable(0.,i,9) for i in range(9)]
    edges=[[Jet2.constant(corners[a,i]-corners[0,i],9)+q[3*a+i]-q[i]
            for i in range(3)] for a in (1,2)]
    raw=_cross(*edges);length=jet_sqrt(dot(raw,raw))
    normal=[v/length for v in raw]
    return (np.array([v.value for v in normal]),
            np.array([v.gradient for v in normal]),
            np.array([v.hessian for v in normal]))


def _relative_center(high,low,cell,t,origin,arm=None):
    # Preserve represented low parts before moving-normal work is formed.
    f=lambda v:Fraction(float(v));weight=f(t)
    arm=np.zeros(3) if arm is None else arm
    return np.array([float((1-weight)*(f(high[cell,i])+f(low[cell,i])-f(origin[i]))+
        weight*(f(high[cell+1,i])+f(low[cell+1,i])-f(origin[i]))+f(arm[i])) for i in range(3)])


class _S3CandidateIntegration(NativeCurvedMaterialPlaneWork):
    """Disposable target data; no mutable candidate cache on the public adapter."""
    def __init__(self,work,corners,*,replay_norms=False):
        self._replay_norms = replay_norms
        normal,dn,hn=_normal_derivatives(corners)
        # A represented corner is on the represented facet. A rounded centroid
        # need not lie on its plane at a large common origin.
        origin=corners[0]
        super().__init__(work.contact.rule,plane_origin_m=origin,plane_normal=normal,
            normal_stiffness_Pa_per_m=work.contact.stiffness,
            integration_settings=work.contact.settings)
        self.shell_map=work.shell.work_map.toarray()
        self.dn=dn;self.hn=hn
        self.origin_map=np.zeros((3,9));self.origin_map[:,:3]=np.eye(3)
        # Normal derivatives and the corner-origin map are fixed for this
        # disposable candidate. Reuse their exact contractions at every
        # quadrature station; the next candidate constructs them afresh.
        self._origin_gradient=gap_vector_product(self.origin_map.T,self.normal)
        self._origin_hessian_left=gap_matrix_product(dn.T,self.origin_map)
        self._origin_hessian_right=gap_matrix_product(self.origin_map.T,dn)
        self._axial_centres=None
        self.integration_budget_policy=work.integration_budget_policy
        self.root_tangent_enclosure_policy=work.root_tangent_enclosure_policy

    def _integration_budget_area(self,prepared,segmented,high,low,U,alo,ahi,total_area):
        if self.integration_budget_policy=='full_reference' or not segmented:
            return total_area
        areas=[]
        for cell,bands in prepared:
            for xl,xr,uncertain,_ in bands:
                # Only the existing complete axial root partition permits this
                # classification. Uncertain bands always retain their share.
                if not uncertain:
                    xm=(xl+xr)/2;t=xm+1 if cell==0 else xm
                    field=_station_field(high,low,cell,t,self.reference.half_cell_lift(cell,t),
                        self.reference.frame(xm),U[cell],self.radius,self.origin,self.normal)
                    if min(_critical_gaps(field,alo,ahi))>0:continue
                areas.append((xr-xl)*(ahi-alo))
        area=fsum(areas)
        # An entirely open partition performs no quadrature. Keep a positive
        # denominator without altering its zero contribution.
        return area if area>0 else total_area

    def evaluate(self,positions,position_low,cell_rotations):
        # Exact axial data belongs to this disposable evaluation only. Keep
        # rational centre+arm addition before the final float conversion.
        if self._axial_centres is not None:
            raise RuntimeError('curved S3 candidate evaluation already active')
        self._axial_centres={}
        try:return super().evaluate(positions,position_low,cell_rotations)
        finally:self._axial_centres=None

    def _candidate_relative_center(self,high,low,cell,t,arm):
        cache=self._axial_centres
        if cache is None:return _relative_center(high,low,cell,t,self.origin,arm)
        key=(id(high),id(low),cell,float(t))
        base=cache.get(key)
        if base is None:
            f=lambda v:Fraction(float(v));weight=f(t)
            base=tuple((1-weight)*(f(high[cell,i])+f(low[cell,i])-f(self.origin[i]))+
                weight*(f(high[cell+1,i])+f(low[cell+1,i])-f(self.origin[i])) for i in range(3))
            cache[key]=base
        return np.array([float(base[i]+Fraction(float(arm[i]))) for i in range(3)])

    def _coordinate_count(self):return 42+self.shell_map.shape[1]

    def _coordinate_scale(self,count):
        scale=super()._coordinate_scale(count)
        for first in range(45,count,6):
            scale[first:first+3]=1/self.settings.characteristic_length_m
        return scale

    def _local_mapping(self,cell,count):
        mapping=np.zeros((18,count))
        mapping[:9]=super()._local_mapping(cell,count)
        mapping[9:,42:]=self.shell_map
        return mapping

    def _gap_work_jet(self,gap,high,low,cell,t,arm):
        n=self.normal;dn=self.dn;hn=self.hn
        d=self._candidate_relative_center(high,low,cell,t,arm)
        # DX maps left spatial section increments; shell corner motion is affine.
        a,b,c=arm
        identity=np.eye(3)
        dx=np.column_stack(((1-t)*identity,t*identity,
                            np.array(((0,c,-b),(-c,0,a),(b,-a,0)))))
        gradient=np.r_[gap_vector_product(dx.T,n),gap_vector_product(dn.T,d)-self._origin_gradient]
        hessian=np.zeros((18,18))
        hessian[6:9,6:9]=(np.outer(arm,n)+np.outer(n,arm))/2-deterministic_dot(n,arm)*identity
        hessian[:9,9:]=gap_matrix_product(dx.T,dn)
        hessian[9:,:9]=hessian[:9,9:].T
        hessian[9:,9:]=(np.einsum('i,ijk->jk',d,hn)
            -self._origin_hessian_left-self._origin_hessian_right)
        return Jet2(gap,gradient,hessian)

    def _spatial_tangent(self,residual,hessian):
        result=hessian.copy()
        result[:42,:42]=spatial_jacobian(residual[:42],hessian[:42,:42])
        return result

    def _bound_norm(self, values):
        if self._replay_norms:
            # Sum binary64 squares in a fixed order before taking the root.
            # The archived enclosure uses the Euclidean (or Frobenius) norm.
            return sqrt(fsum(float(x) * float(x) for x in values.flat))
        return np.linalg.norm(values)

    def _root_work_bound(self,high,low,U,cell,width,angular_width,geometry,
                         settings,stiffness,normal,pad,*,fallback=False,interval=None,station=None,
                         declared_arc=False):
        # At this candidate n/Dn/DÂ²n are constant in xi. Bound their COMPLETE
        # scaled authoritative maps, not just the native nine-coordinate block.
        ref=self.reference;r=self.radius;L=settings.characteristic_length_m
        _,jmax,f1,_=_reference_norm_bounds(ref,cell,actual_roll=True)
        f2=_reference_second_bound(ref,cell);fg=_reference_radial_gain(ref)
        b=_reference_curvature_norm(ref);gain=_rotation_gain(U[cell])
        arm=gain*(b/8+r*fg);ap=gain*(b/2+r*f1)
        scale=self._coordinate_scale(self._coordinate_count())[42:]
        w=self.shell_map*scale
        dn=self.dn@w;o=self.origin_map@w
        hn=np.einsum('ijk,ja,kb->iab',self.hn,w,w)
        n1=np.linalg.norm(dn);n2=self._bound_norm(hn);o1=np.linalg.norm(o,2)
        center=[_relative_center(high,low,cell,t,self.origin) for t in (0.,1.)]
        distance=max(deterministic_norm(v) for v in center)+arm
        chord=np.array([fsum((high[cell+1,i],low[cell+1,i],-high[cell,i],-low[cell,i]))
                        for i in range(3)])
        xp=np.linalg.norm(chord)+ap
        x1=1+arm/L
        g=x1+n1*distance+o1
        gx=sqrt(2)+ap/L+n1*xp
        mapping=self._local_mapping(cell,self._coordinate_count())
        full_scale=self._coordinate_scale(self._coordinate_count())
        components=anchored=None
        station_empty=False
        if interval is not None and station is not None:
            # Complete mapped gradient is affine in cos(alpha),sin(alpha) at
            # one xi. Keep the historical circle bound for passing evaluations;
            # a budget failure may use the declared arc, then its change over the
            # WHOLE band by the complete axial derivative bound above.
            t=station+1 if cell==0 else station
            frame=ref.frame(station);base=U[cell]@ref.half_cell_lift(cell,t)
            gradients=[full_scale*(mapping.T@self._gap_work_jet(0.,high,low,cell,t,arm).gradient)
                       for arm in (base,base+U[cell]@(r*frame[:,1]),base+U[cell]@(r*frame[:,2]))]
            circular=np.column_stack((gradients[1]-gradients[0],gradients[2]-gradients[0]))
            angular=(_declared_arc_gradient_bound(gradients[0],circular,self.rule.definition['angle_range'])
                     if declared_arc else self._bound_norm(gradients[0])+np.linalg.norm(circular,2))
            tight=angular+max(abs(station-x) for x in interval)*gx
            g=min(g,tight*(1+8192*np.finfo(float).eps))
            if declared_arc:
                # Exact zero-depth certification at the retained station: the
                # projected scalar normal gap is positive at every declared-arc
                # critical angle (exact bounded-precision values), so the
                # station's active angular set is empty and the production
                # station quadrature contributes exactly zero. Both angular
                # rule paths yield only negative scalars, grazing included.
                sfield=_station_field(high,low,cell,t,ref.half_cell_lift(cell,t),
                    ref.frame(station),U[cell],r,self.origin,self.normal)
                station_empty=min(_critical_gaps(sfield,*self.rule.definition['angle_range']))>0
            if declared_arc and self.root_tangent_enclosure_policy in ('componentwise','anchored'):
                components=_root_component_gradient_bounds(gradients,self.rule.definition['angle_range'],
                    max(abs(station-x) for x in interval)*gx,np.any(mapping!=0.,axis=0))
                if self.root_tangent_enclosure_policy=='anchored':
                    midpoint=sum(self.rule.definition['angle_range'])/2
                    anchor=gradients[0]+circular@np.array((cos(midpoint),sin(midpoint)))
                    variation=_root_component_gradient_bounds(gradients,self.rule.definition['angle_range'],
                        max(abs(station-x) for x in interval)*gx,np.any(mapping!=0.,axis=0),anchor=anchor)
                    anchored=anchor,variation
        # Includes the once-spatial native correction, mixed and shell blocks.
        h=3*arm/L**2+2*x1*n1+distance*n2+2*n1*o1
        depth=max(0.,-geometry.gap_lower_m)
        area=r*fg*(jmax+r*f1)
        area_prime=r*fg*(b+r*f2)+(jmax+r*f1)*r*f1
        le=stiffness*(depth*xp*area+.5*depth**2*area_prime)
        lf=stiffness*((xp*g+depth*gx)*area+depth*g*area_prime)
        ee=angular_width*(width**2*le/2+width*stiffness*depth*pad*area)
        fe=angular_width*(width**2*lf/2+width*stiffness*pad*g*area)
        if fallback:
            ee=max(ee,stiffness*area*width*angular_width*depth**2)
            fe=max(fe,2*stiffness*area*width*angular_width*depth*g)
        active=np.count_nonzero(np.any(mapping!=0.,axis=0))
        factor=stiffness*area*width*angular_width
        # Positive area and Gauss weights make both exact and approximate
        # mapped leading g*g^T integrals PSD, bounded by factor*gÂ². The
        # measure covers the true union AND grazing fallback quadrature mass.
        # Only the non-PSD Hessian/once-spatial remainder needs a sum bound.
        te=(_psd_root_tangent_difference_bound(factor*g*g,factor*depth*h,active)
            if declared_arc else 2*sqrt(active)*stiffness*area*width*angular_width*(g*g+depth*h))
        if components is not None:
            leading=min(sqrt(active)*factor*g*g,
                _component_root_leading_difference_bound(factor,components))
            if anchored is not None:
                leading=min(leading,_anchored_root_leading_difference_bound(factor,*anchored))
            te=leading+2*sqrt(active)*factor*depth*h
        if station_empty:
            bounds=components if components is not None else _root_component_gradient_bounds(
                gradients,self.rule.definition['angle_range'],
                max(abs(station-x) for x in interval)*gx,np.any(mapping!=0.,axis=0))
            # One-sided enclosure: the station integral is exactly zero,
            # so |A_ij|<=factor*b_i*b_j over the true region (measure
            # <=width*angular_width, Jacobian<=area) bounds the leading
            # infinity norm by factor*max(b)*sum(b); the gap*Hessian
            # remainder keeps one spectral count with unchanged depth,h.
            single=float(np.nextafter(factor*np.max(bounds)*fsum(bounds),np.inf))
            te=min(te,single+sqrt(active)*factor*depth*h)
        return np.array((ee,fe,te))*(1+8192*np.finfo(float).eps)

    def _refined_root_work_bound(self,*args,**kwargs):
        return self._root_work_bound(*args,**kwargs,declared_arc=True)


class NativeCurvedMaterialS3Work(NativeRetainedMaterialS3Work):
    """Local whole-contained curved patch; exact inherited skin/identity guards."""
    def __init__(self,contact,shell,*,integration_budget_policy='full_reference',
                 root_tangent_enclosure_policy='spectral'):
        if integration_budget_policy not in ('full_reference','certified_active'):
            raise ValueError('unknown curved S3 integration budget policy')
        self.integration_budget_policy=integration_budget_policy
        if root_tangent_enclosure_policy not in ('spectral','componentwise','anchored'):
            raise ValueError('unknown curved S3 root tangent enclosure policy')
        self.root_tangent_enclosure_policy=root_tangent_enclosure_policy
        super().__init__(contact,shell)

    def _fingerprint(self):
        definition=dict(profile=PROFILE,base=super()._fingerprint(),
            numerical='CURVED_SHARED_GAUSS8_16_V7_COMPLETE_MOVING_NORMAL_WORK_V2',
            path='affine_skin_left_exp_complete_bernstein_controls_v1')
        if self.integration_budget_policy!='full_reference':
            definition['integration_budget_policy']=self.integration_budget_policy
        if self.root_tangent_enclosure_policy!='spectral':
            definition['root_tangent_enclosure_policy']=(self.root_tangent_enclosure_policy,'whole_band_v1')
        return sha(definition)

    def _geometry(self,pose):
        k=self.contact;corners=self.shell.current_positions(pose.shell_displacement)
        physical=self.target.physical_pose(pose.shell_displacement)
        normal=np.asarray(physical.material_frame)[:,2]
        edges=[corners[(i+1)%3]-corners[i] for i in range(3)]
        if (min(np.linalg.norm(e) for e in edges)<=k.geometry_tolerance or
            np.linalg.norm(np.cross(edges[0],-edges[2]))<=k.geometry_tolerance**2):
            raise ValueError('curved material S3 has unresolved target geometry')
        control=lambda origin,n:curved_patch_geometry_control(k.rule,pose.positions,
            pose.position_low,pose.cell_rotations,origin,n)
        extent=control(corners[0],normal)
        clearance=min(control(corners[i],np.cross(normal,edges[i]/np.linalg.norm(edges[i]))).gap_lower_m
                      for i in range(3))
        if clearance<=k.rim_clearance:
            raise ValueError('curved material S3 patch approaches unadmitted physical rim')
        return extent,clearance

    def geometry_gap(self,positions,position_low,rotations,u):
        extent,clearance=self._geometry(self._configuration(positions,position_low,rotations,u))
        return extent.gap_lower_m,clearance

    def certify_orientation_path(self,*args,**kwargs):
        raise NotImplementedError('curved moving-S3 requires the complete certify_path whole-path protocol')

    def certify_path(self,start,translations,angular,shell_increment,*,
                     gap_tolerance_m,maximum_intervals,cancellation_token=None):
        """Affine skin / left Exp path with whole-patch orientation, gap and rim.

        Geometric gap bounds use Bernstein source controls. Their rotating
        middle control has twice the physical lift; all rates below include it.
        Endpoint or midpoint sampling alone never establishes path admission.
        """
        from anyfesolver.control import cancellation_safe_point
        from .native_surface_s3_friction import NativeS3MaterialConfiguration
        from .native_surface_path import material_path_pose
        pose=self._configuration(start.positions,start.position_low,start.cell_rotations,start.shell_displacement)
        move=np.asarray(translations,dtype=float);spin=np.asarray(angular,dtype=float)
        du=np.asarray(shell_increment,dtype=float)
        if (move.shape!=(3,3) or spin.shape!=(2,3) or du.shape!=pose.shell_displacement.shape
            or not all(np.all(np.isfinite(v)) for v in (move,spin,du))):
            raise ValueError('finite authoritative curved/S3 path increments required')
        if (type(maximum_intervals) is not int or maximum_intervals<1 or
            type(gap_tolerance_m) not in (int,float) or not np.isfinite(gap_tolerance_m) or gap_tolerance_m<=0):
            raise ValueError('positive dimensional gap tolerance and finite path cap required')
        identity=sha(dict(profile=PROFILE,work=self.identity,start=pose,move=move,spin=spin,
            shell_increment=du,gap_tolerance_m=gap_tolerance_m,maximum_intervals=maximum_intervals,
            path='affine_skin_left_exp_complete_bernstein_controls_v1'))
        visited=0;gap=np.inf;orientation=None
        def result(ok,reason):
            return CurvedPatchPathControl(ok,reason,visited,orientation,
                float(gap) if np.isfinite(gap) else None,identity)
        first=self.shell.current_positions(pose.shell_displacement)
        last=self.shell.current_positions(pose.shell_displacement+du)
        a0,b0=first[1]-first[0],first[2]-first[0]
        a1,b1=last[1]-last[0],last[2]-last[0]
        c0,c2=np.cross(a0,b0),np.cross(a1,b1)
        cm=np.cross((a0+a1)/2,(b0+b1)/2);c1=2*cm-(c0+c2)/2
        ref=self.shell.reference_positions
        nr=np.cross(ref[1]-ref[0],ref[2]-ref[0]);nr/=np.linalg.norm(nr)
        eps=np.finfo(float).eps
        # Cancellation of long near-parallel edges must not shrink arithmetic
        # allowance to the resulting (possibly tiny) cross-product magnitude.
        product_scale=max(np.linalg.norm(a) for a in (a0,a1))*max(np.linalg.norm(b) for b in (b0,b1))
        area_pad=32768*eps*max(1.,product_scale)
        area_lower=min(nr@c for c in (c0,c1,c2))-area_pad
        if area_lower<=self.contact.geometry_tolerance**2:
            return result(False,'whole target orientation/area unresolved')
        normal_speed=(np.linalg.norm(a1-a0)*max(np.linalg.norm(b0),np.linalg.norm(b1))+
                      np.linalg.norm(b1-b0)*max(np.linalg.norm(a0),np.linalg.norm(a1)))/area_lower*(1+8192*eps)
        physical=self.target.physical_pose(pose.shell_displacement)
        native=NativeS3MaterialConfiguration(pose.positions,pose.position_low,pose.cell_rotations,np.zeros(0))
        checked=curved_patch_path_control(self.contact.rule,native,move,spin,
            physical.position,np.asarray(physical.material_frame)[:,2],
            gap_tolerance_m=gap_tolerance_m,maximum_intervals=maximum_intervals,
            cancellation_token=cancellation_token,orientation_only=True)
        visited=checked.intervals;orientation=checked.orientation_lower
        if not checked.certified:return result(False,checked.reason)
        # Reference and current source control distances, not merely penetration.
        fg=_reference_radial_gain(self.contact.beam_reference)
        arm=_reference_curvature_norm(self.contact.beam_reference)/4+self.contact.radius*fg
        gains=[_rotation_gain(v)*(1+2048*eps) for v in pose.cell_rotations]
        gain=max(gains)
        source_speed=max(np.linalg.norm(v) for v in move)+max(
            np.linalg.norm(spin[i])*gains[i]*arm for i in (0,1))
        points=np.array([_relative_center(pose.positions,pose.position_low,0,0.,np.zeros(3)),
            _relative_center(pose.positions,pose.position_low,0,1.,np.zeros(3)),
            _relative_center(pose.positions,pose.position_low,1,1.,np.zeros(3))])
        ends=points+move
        distance=lambda x0,x1:max(np.linalg.norm(p-x) for pp,xx in ((points,x0),(ends,x1))
                                 for p in pp for x in np.atleast_2d(xx))+gain*arm
        center0,center1=first[0],last[0]
        gap_speed=source_speed+np.linalg.norm(center1-center0)+normal_speed*distance(center0,center1)
        rim_speed=0.
        for i in range(3):
            j=(i+1)%3;e0=first[j]-first[i];de=last[j]-last[i]-e0
            den=de@de;t=0. if den==0 else np.clip(-(e0@de)/den,0.,1.)
            length_pad=32768*eps*max(1.,np.linalg.norm(e0),np.linalg.norm(e0+de))
            edge_lower=np.linalg.norm(e0+t*de)-length_pad
            if edge_lower<=self.contact.geometry_tolerance:
                return result(False,'whole target edge length unresolved')
            inward_speed=normal_speed+np.linalg.norm(de)/edge_lower
            rim_speed=max(rim_speed,source_speed+np.linalg.norm(last[i]-first[i])+
                          inward_speed*distance(first[i],last[i]))
        scale=max(1.,*(float(np.max(np.abs(v))) for v in (first,last,points,ends)))
        pad=32768*eps*scale*(1+fg)*max(1.,arm)
        pending=[(0.,1.)]
        while pending:
            cancellation_safe_point(cancellation_token,'curved moving-S3 whole path')
            if visited>=maximum_intervals:return result(False,'whole moving-S3 path interval budget exhausted')
            left,right=pending.pop();mid=(left+right)/2;half=(right-left)/2;visited+=1
            if mid in (left,right):return result(False,'whole moving-S3 path resolution exhausted')
            current=material_path_pose(pose,move,spin,du,mid)
            try:extent,rim=self._geometry(current)
            except ValueError:return result(False,'whole moving-S3 geometry unresolved')
            lower_gap=extent.gap_lower_m-half*gap_speed-pad
            lower_rim=rim-half*rim_speed-pad
            if lower_gap>=-gap_tolerance_m and lower_rim>self.contact.rim_clearance:
                gap=min(gap,lower_gap);continue
            pending.extend(((left,mid),(mid,right)))
        return result(True,'whole curved source and moving-S3 gap/rim/orientation bounded')

    def evaluate(self,positions,position_low,rotations,u,*,replay_norms=False):
        pose=self._configuration(positions,position_low,rotations,u)
        extent,clearance=self._geometry(pose)
        corners=self.shell.current_positions(u)
        integral=_S3CandidateIntegration(self,corners,replay_norms=replay_norms).evaluate(positions,position_low,rotations)
        residual=integral.residual
        bf=np.array([-residual[6*i:6*i+3] for i in range(3)])
        sf=-residual[42:][self.shell_translation_dofs]
        bm=sum((np.cross(pose.positions[i],bf[i])+np.cross(pose.position_low[i],bf[i])
                for i in range(3)),start=np.zeros(3))-residual[18:24].reshape(2,3).sum(axis=0)
        sm=np.cross(corners,sf).sum(axis=0)
        physical=NativeMaterialS3PhysicalResult(integral.normal_storage_j,
            _readonly(bf.sum(axis=0)),_readonly(sf.sum(axis=0)),_readonly(bm),_readonly(sm),
            _readonly(sf),extent.gap_lower_m,clearance,None,integral.accepted_intervals,
            integral.integrand_evaluations,*integral.estimated_errors,None)
        return NativeMaterialS3WorkResult(physical,_readonly(residual),
            sparse.csr_matrix(integral.tangent),PROFILE)
