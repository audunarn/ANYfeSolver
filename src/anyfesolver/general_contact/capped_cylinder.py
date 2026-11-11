"""Explicit finite cylinder geometry and classical round-line static contact.

The cylinder has flat physical caps, not hemispherical obstacle ends. The
centreline gap is its signed Euclidean distance minus the declared tube radius.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from fractions import Fraction
import heapq
import math

import numpy as np

from .core import RigidContactBody, ContactSolverSettings, ContactValidationReport
from .beam_plane import _BeamCell
from .beam_plane import _beam_types
from .beam_end_topology import physical_beam_ends
from scipy import sparse
from .beam_radial import (_BeamRadialManager, validate_beam_radial_contact,
                          _curve_coefficients, _real_interval_roots, _GAUSS_8, _GAUSS_16, _RadialResults)
from .beam_temporal_bounds import _fraction, _outward, _split_curve

PROFILE = 'CLASSICAL_B2_B3_ROUND_TUBE_CAPPED_CYLINDER_LINE_PENALTY_STATIC_DEV_V0'
PROFILE_END_CAPS = 'CLASSICAL_B2_B3_ROUND_TUBE_CAPPED_CYLINDER_HEMISPHERE_ENDS_STATIC_DEV_V1'


@dataclass(frozen=True)
class RigidCappedCylinder:
    """Additional immutable descriptor; existing rigid-body schemas stay valid."""
    name: str
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float
    half_length: float
    path_translation: tuple[float, float, float] | None = None
    kind: str = field(default='capped_cylinder', init=False)

    def __post_init__(self):
        axis=np.asarray(self.normal,dtype=float)
        if axis.shape!=(3,) or not np.all(np.isfinite(axis)):
            raise ValueError('capped cylinder axis must be finite and nonzero')
        length=math.hypot(*axis)
        if length==0 or not math.isfinite(length):
            raise ValueError('capped cylinder axis must be finite and nonzero')
        body = RigidContactBody(self.name, 'cylinder', self.point, tuple(axis/length),
                                self.radius, self.path_translation)
        for key in ('point','normal','radius','path_translation'):
            object.__setattr__(self,key,getattr(body,key))
        length = float(self.half_length)
        if not np.isfinite(length) or length <= 0.:
            raise ValueError('capped cylinder half_length must be finite and positive')
        object.__setattr__(self,'half_length',length)


def _frame(axis):
    axis = np.asarray(axis,dtype=float)
    seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first = np.cross(axis,seed); first /= np.linalg.norm(first)
    return np.array([first,np.cross(axis,first),axis])


def signed_distance_work(point, center, frame, radius, half_length):
    """Distance, positional gradient and Hessian on one smooth feature branch."""
    local = frame @ (np.asarray(point)-center)
    rho = float(np.linalg.norm(local[:2])); z = float(local[2])
    radial, axial = rho-radius, abs(z)-half_length
    outside = radial > 0. and axial > 0.
    if not outside and radial == axial and radial <= 0.:
        raise ValueError('capped cylinder interior equal-distance seam is nonsmooth')
    if not outside and axial > radial:
        if z == 0.:
            raise ValueError('capped cylinder cap-controlled midplane is nonsmooth')
        gradient = np.array([0.,0.,math.copysign(1.,z)])
        return axial, frame.T @ gradient, np.zeros((3,3))
    if rho == 0.:
        raise ValueError('capped cylinder radial axis has undefined normal')
    er = np.array([local[0]/rho,local[1]/rho,0.])
    hr = (np.diag([1.,1.,0.])-np.outer(er,er))/rho
    if not outside:
        return radial, frame.T @ er, frame.T @ hr @ frame
    ez = np.array([0.,0.,math.copysign(1.,z)])
    distance = math.hypot(radial,axial)
    normal = (radial*er+axial*ez)/distance
    hessian = radial/distance*hr + (np.outer(er,er)+np.outer(ez,ez)-np.outer(normal,normal))/distance
    return distance, frame.T @ normal, frame.T @ hessian @ frame


def signed_distance_value(point,center,frame,radius,half_length):
    """The geometric value remains defined on medial sets; derivatives do not."""
    local=frame@(np.asarray(point)-center)
    radial=float(np.linalg.norm(local[:2]))-radius
    axial=abs(float(local[2]))-half_length
    if max(radial,axial)<=0: return max(radial,axial)
    return math.hypot(max(radial,0.),max(axial,0.))


def _sqrt_bound(value: Fraction,upper=False):
    if value==0: return Fraction(0)
    exponent=value.numerator.bit_length()-value.denominator.bit_length()
    even=exponent-exponent%2
    scaled=value/(2**even if even>=0 else Fraction(1,2**(-even)))
    result=math.ldexp(math.sqrt(float(scaled)),even//2)
    direction=math.inf if upper else -math.inf
    while (_fraction(result)**2<value if upper else _fraction(result)**2>value):
        result=math.nextafter(result,direction)
    return _fraction(result)


def _box_bound(controls, radius, half_length, beam_radius,upper=False):
    """Exact rational control boxes and outward scalar square roots."""
    intervals = [(min(lane),max(lane)) for lane in controls]
    near = ([max(abs(lo),abs(hi)) for lo,hi in intervals] if upper else
            [Fraction(0) if lo <= 0 <= hi else min(abs(lo),abs(hi)) for lo,hi in intervals])
    radial = _sqrt_bound(near[0]**2+near[1]**2,upper)-_fraction(radius)
    axial = near[2]-_fraction(half_length)
    if max(radial,axial) <= 0: distance = max(radial,axial)
    else: distance = _sqrt_bound(max(radial,Fraction(0))**2+max(axial,Fraction(0))**2,upper)
    return _outward(distance-_fraction(beam_radius),math.inf if upper else -math.inf)


def _box_lower(controls,radius,half_length,beam_radius):
    return _box_bound(controls,radius,half_length,beam_radius)


def _controls(positions,center,frame):
    lanes = []
    for direction in frame:
        values = tuple(sum((_fraction(x)-_fraction(c))*_fraction(d)
                           for x,c,d in zip(point,center,direction)) for point in positions)
        if len(values)==3: values = (values[0],2*values[1]-(values[0]+values[2])/2,values[2])
        lanes.append(values)
    return tuple(lanes)


def _restrict(controls,left,right):
    """Exact Bernstein restriction using the degree-one/two blossom."""
    l,r=_fraction(left),_fraction(right)
    result=[]
    for values in controls:
        if len(values)==2:
            result.append((values[0]+l*(values[1]-values[0]),values[0]+r*(values[1]-values[0])))
        else:
            a,b,c=values
            result.append(((1-l)**2*a+2*l*(1-l)*b+l*l*c,
                (1-l)*(1-r)*a+(l*(1-r)+r*(1-l))*b+l*r*c,
                (1-r)**2*a+2*r*(1-r)*b+r*r*c))
    return tuple(result)


def _product(first,second):
    """Exact one-dimensional Bernstein polynomial multiplication."""
    n,m=len(first)-1,len(second)-1
    result=[Fraction(0)]*(n+m+1)
    for i,a in enumerate(first):
        for j,b in enumerate(second):
            result[i+j]+=a*b*Fraction(math.comb(n,i)*math.comb(m,j),math.comb(n+m,i+j))
    return tuple(result)


def _elevate(values,degree):
    current=tuple(values)
    while len(current)-1<degree:
        n=len(current)
        current=(current[0],*(Fraction(i,n)*current[i-1]+Fraction(n-i,n)*current[i]
                              for i in range(1,n)),current[-1])
    return current


def _inactive(controls,radius,half_length,beam_radius):
    """Sufficient exact gap>=0 proof retaining radial/axial correlation.

    In a rim quadrant A=q+R²+w²-r²>=0 and A²-4R²q>=0 imply
    (sqrt(q)-R)²+w²>=r². Every branch inequality is also enclosed.
    No sampled sign or squared-root extraneous branch establishes inactivity.
    """
    R,h,r=map(_fraction,(radius,half_length,beam_radius))
    xx,yy=_product(controls[0],controls[0]),_product(controls[1],controls[1])
    q=tuple(a+b for a,b in zip(xx,yy))
    if min(q)>=(R+r)**2: return True
    z=controls[2]
    if min(z)>=h+r or max(z)<=-h-r: return True
    if min(q)<R*R: return False
    for sign in (-1,1):
        w=tuple(sign*a-h for a in z)
        if min(w)<0: continue
        ww=_product(w,w)
        A=tuple(a+b+R*R-r*r for a,b in zip(q,ww))
        if min(A)<0: continue
        square=_product(A,A)
        elevated=_elevate(q,len(square)-1)
        if min(a-4*R*R*b for a,b in zip(square,elevated))>=0: return True
    return False


def _point_gap_sign(controls,radius,half_length,beam_radius):
    """Exact algebraic sign of d-r at one Bernstein point (no sqrt cancellation)."""
    x,y,z=(lane[0] for lane in controls)
    R,h,r=map(_fraction,(radius,half_length,beam_radius))
    q=x*x+y*y; w=abs(z)-h
    if w<=0: value=q-(R+r)**2
    elif q<=R*R: value=w-r
    else:
        A=q+R*R+w*w-r*r
        value=A if A<0 else A*A-4*R*R*q
    return (value>0)-(value<0)


def _point_gap(controls,radius,half_length,beam_radius):
    """Exterior gap with rationalized subtraction near activation."""
    x,y,z=(lane[0] for lane in controls)
    R,h,r=map(_fraction,(radius,half_length,beam_radius))
    q=x*x+y*y; w=abs(z)-h
    rho=float(_sqrt_bound(q))
    if w<=0: return float(q-(R+r)**2)/(rho+float(R+r))
    if q<=R*R: return float(w-r)
    distance=math.hypot(rho-radius,float(w))
    A=q+R*R+w*w-r*r
    if A<0: return distance-beam_radius
    F=A*A-4*R*R*q
    return float(F)/((float(A)+2*radius*rho)*(distance+beam_radius))


def curve_gap_lower(positions,center,frame,radius,half_length,beam_radius,*,accuracy,maximum_depth,admit_above=None):
    """Conservative minimum lower bound over the entire represented B2/B3 curve.

    Accuracy controls enclosure width, never permission to increase penetration.
    Exhausted enclosure budget raises instead of returning a sampled minimum.
    """
    controls = _controls(positions,center,frame)
    return _controls_gap_lower(controls,radius,half_length,beam_radius,accuracy,maximum_depth,admit_above)


def _controls_gap_lower(controls,radius,half_length,beam_radius,accuracy,maximum_depth,admit_above=None):
    def gap(s): return _box_bound(_restrict(controls,s,s),radius,half_length,beam_radius,True)
    upper = min(gap(0.),gap(.5),gap(1.))
    serial=0
    heap=[(_box_lower(controls,radius,half_length,beam_radius),serial,0,0.,1.,controls)]
    while upper-heap[0][0] > accuracy and (admit_above is None or heap[0][0] < admit_above):
        _,_,depth,left,right,lanes = heapq.heappop(heap)
        if depth >= maximum_depth:
            raise MemoryError('capped cylinder curve gap enclosure budget exhausted')
        splits = [_split_curve(lane) for lane in lanes]
        middle=(left+right)/2
        for side,lo,hi in ((0,left,middle),(1,middle,right)):
            child=tuple(pair[side] for pair in splits)
            serial+=1
            heapq.heappush(heap,(_box_lower(child,radius,half_length,beam_radius),serial,depth+1,lo,hi,child))
            upper=min(upper,gap((lo+hi)/2))
    return heap[0][0]


def _cuts(positions,center,frame,radius,half_length,beam_radius):
    """Include side/cap/rim activation, medial and geometric branch equations."""
    original_controls=_controls(positions,center,frame)
    # An exact whole-curve cap certificate removes the spurious squared-root
    # duplicates and float-bisection failure near a physical end. This is the
    # actual affine/quadratic cap gap, not a chord replacement.
    q=tuple(a+b for a,b in zip(_product(original_controls[0],original_controls[0]),
                              _product(original_controls[1],original_controls[1])))
    if max(q)<=_fraction(radius)**2:
        for axial_sign in (-1,1):
            z=tuple(axial_sign*v-_fraction(half_length) for v in original_controls[2])
            if min(z)<0:continue
            from .beam_chain_consistent import _roots
            gap=tuple(v-_fraction(beam_radius) for v in z)
            coefficients=((Fraction(0),gap[1]-gap[0],gap[0]) if len(gap)==2 else
                (gap[0]-2*gap[1]+gap[2],2*(gap[1]-gap[0]),gap[0]))
            exact=_roots(coefficients,Fraction(0),Fraction(1));cuts=[]
            for root in exact:
                value=float(root);a,b,c=coefficients;slope=2*a*root+b
                if slope==0:
                    cuts.append(value);continue
                for _ in range(8):
                    sign=_point_gap_sign(_restrict(original_controls,value,value),radius,half_length,beam_radius)
                    if sign>=0:break
                    value=math.nextafter(value,math.inf if slope>0 else -math.inf)
                else:raise ArithmeticError('exact cap activation lacks a representable open-side cut')
                if not 0<value<1:raise ArithmeticError('exact cap activation cut overlaps a domain end')
                cuts.append(value)
            if len(set(cuts))!=len(cuts):raise ArithmeticError('exact cap activation cuts collapse at work precision')
            return (0.,*sorted(cuts),1.)
    a,b,c = _curve_coefficients(positions)
    local = np.array([frame@(a-center),frame@b,frame@c])
    P=np.polynomial.Polynomial
    x,y,z=(P(local[:,i]) for i in range(3))
    q=x*x+y*y
    polynomials=[z,z-half_length,z+half_length,q-radius**2,q-(radius+beam_radius)**2,
                 z-half_length-beam_radius,z+half_length+beam_radius]
    for sign in (-1.,1.):
        w=sign*z-half_length
        polynomials.extend([q-(radius+w)**2,
            (q+radius**2+w*w-beam_radius**2)**2-4*radius**2*q])
    # Extraneous squared roots are harmless additional integration partitions;
    # physical activation always evaluates the unsquared signed distance.
    roots={root for polynomial in polynomials for root in _real_interval_roots(polynomial.coef)}
    raw=(0.,*sorted(roots),1.)
    def sign(s): return _point_gap_sign(_restrict(original_controls,s,s),radius,half_length,beam_radius)
    polished=[]
    for i,root in enumerate(raw[1:-1],1):
        left,right=(raw[i-1]+root)/2,(root+raw[i+1])/2
        gl,gr=sign(left),sign(right)
        if gl*gr<0:
            # Retain an exact open-side endpoint of a sign bracket. Floating
            # distance subtraction can round a narrow pocket to zero; it is
            # never a root or inactivity certificate here.
            for _ in range(64):
                mid=(left+right)/2
                if mid in (left,right): break
                gm=sign(mid)
                if gm==0:
                    left=right=mid; gl=gr=0; break
                if gm==gl: left,gl=mid,gm
                else: right,gr=mid,gm
            else: raise MemoryError('capped cylinder activation root cannot be isolated within its budget')
            root=left if gl>=0 else right
            if not _inactive(_restrict(original_controls,root,root),radius,half_length,beam_radius):
                raise ArithmeticError('capped cylinder activation root lacks an exact open-side certificate')
        polished.append(root)
    return (0.,*sorted(set(polished)),1.)


def validate_capped_cylinder_beam(model,definition,settings=None):
    controls=ContactSolverSettings() if settings is None else settings
    if (len(definition.rigid_bodies)!=1 or type(definition.rigid_bodies[0]) is not RigidCappedCylinder):
        return ContactValidationReport('invalid',('one exact RigidCappedCylinder required',),PROFILE)
    body=definition.rigid_bodies[0]
    if (len(definition.surfaces)!=1 or definition.surfaces[0].beam_radius is None
            or len(definition.interactions)!=1):
        return ContactValidationReport('invalid',('one round beam surface and line interaction required',),PROFILE)
    if definition.interactions[0].gap_tolerance>=definition.surfaces[0].beam_radius:
        return ContactValidationReport('invalid',('capped cylinder exterior profile needs gap_tolerance below beam radius',),PROFILE)
    radial=RigidContactBody(body.name,'cylinder',body.point,body.normal,body.radius,body.path_translation)
    surface,law=definition.surfaces[0],definition.interactions[0]
    caps=surface.beam_end_caps=='hemisphere'
    if caps != (law.endpoint_stiffness is not None):
        return ContactValidationReport('invalid',('hemisphere beam caps require separate endpoint_stiffness in N/m',),PROFILE_END_CAPS if caps else PROFILE)
    if caps:
        definition=replace(definition,surfaces=(replace(surface,beam_end_caps='none'),),
            interactions=(replace(law,endpoint_stiffness=None),))
    # Reuse only the classical section/line/selection admission contract.
    report=validate_beam_radial_contact(model,replace(definition,rigid_bodies=(radial,)),controls)
    return ContactValidationReport(report.status,report.issues,PROFILE_END_CAPS if caps else PROFILE)


def _interval_product(first,second):
    values=[a*b for a in first for b in second]
    return min(values),max(values)


def _cap_orientation_range(point,tangent,radius,half_length):
    """Exact sign enclosure of distance-gradient dot outward endpoint tangent.

    Outside the convex cylinder, its sign is that of
    max(1-R/rho,0)*(x vx+y vy)+(z-clip(z,-h,h))*vz.
    No unit-normal division or sampled ownership establishes this sign.
    """
    p=[(min(lane),max(lane)) for lane in point]
    v=[(min(lane),max(lane)) for lane in tangent]
    near=[Fraction(0) if lo<=0<=hi else min(abs(lo),abs(hi)) for lo,hi in p[:2]]
    far=[max(abs(lo),abs(hi)) for lo,hi in p[:2]]
    lower=_sqrt_bound(sum(x*x for x in near));upper=_sqrt_bound(sum(x*x for x in far),True)
    R,h=_fraction(radius),_fraction(half_length)
    alpha=(max(1-R/lower,Fraction(0)) if lower else Fraction(0),
           max(1-R/upper,Fraction(0)) if upper else Fraction(0))
    radial=[_interval_product(a,b) for a,b in zip(p[:2],v[:2])]
    radial=(sum(a for a,b in radial),sum(b for a,b in radial))
    axial=tuple(z-max(-h,min(h,z)) for z in p[2])
    first=_interval_product(alpha,radial);second=_interval_product(axial,v[2])
    return first[0]+second[0],first[1]+second[1]


def _end_geometry_controls(positions,center,frame,local,outward):
    controls=_controls(positions,center,frame);degree=len(positions)-1
    index=0 if local==0 else -1
    point=tuple((lane[index],) for lane in controls)
    tangent=tuple((outward*degree*(lane[1]-lane[0] if local==0 else lane[-1]-lane[-2]),)
                  for lane in controls)
    return point,tangent


@dataclass(frozen=True)
class _CappedResults(_RadialResults):
    gap_enclosure_accuracy_m: float

    def to_dict(self):
        return {**super().to_dict(),'minimum_gap_kind':'conservative_full_curve_lower_bound',
            'gap_enclosure_accuracy_m':self.gap_enclosure_accuracy_m,
            'geometry_kinematics':'fixed_frame_signed_capped_cylinder_affine_trial_v1'}


@dataclass(frozen=True)
class _HemisphereCappedResults(_CappedResults):
    endpoint_contact_energy: float
    candidate_endpoints: int
    active_endpoints: int
    endpoint_stiffness: float
    endpoint_fields: tuple

    def to_dict(self):
        return {**super().to_dict(),'endpoint_contact_energy':self.endpoint_contact_energy,
            'algorithmic_contact_potential':self.line_contact_energy+self.endpoint_contact_energy,
            'candidate_endpoints':self.candidate_endpoints,'active_endpoints':self.active_endpoints,
            'endpoint_stiffness_N_per_m':self.endpoint_stiffness,
            'endpoint_fields':[dict(field) for field in self.endpoint_fields],
            'moment_origin':[0.,0.,0.],'moment_units':'N*m',
            'beam_end_cap_geometry':'outward_hemisphere_point_penalty',
            'endpoint_ownership':'physical_material_ends; active hemisphere switches reject'}


class CappedCylinderBeamManager(_BeamRadialManager):
    def __init__(self,model,definition,settings=None,initial_displacement=None):
        self.settings=ContactSolverSettings() if settings is None else settings
        report=validate_capped_cylinder_beam(model,definition,self.settings)
        if not report.valid: raise ValueError('unsupported capped-cylinder beam contact: '+'; '.join(report.issues))
        self.model,self.definition,self.profile=model,definition,report.profile
        body=definition.rigid_bodies[0]; surface=definition.surfaces[0]; law=definition.interactions[0]
        self.kind=body.kind; self.reference_point=np.array(body.point); self.point=self.reference_point.copy()
        self.frame=_frame(body.normal); self.obstacle_radius=body.radius; self.half_length=body.half_length
        self.beam_radius=surface.beam_radius; self.radius=body.radius+surface.beam_radius
        self.stiffness=law.line_stiffness; self.gap_tolerance=law.gap_tolerance
        self.end_caps=surface.beam_end_caps;self.endpoint_stiffness=law.endpoint_stiffness
        self.swept_trial_cutbacks=0; self.path_factor=0.
        self.path_translation=np.zeros(3) if body.path_translation is None else np.array(body.path_translation)
        self.cells=[]
        for element_id in surface.element_ids:
            element=model.mesh.elements[element_id]
            reference=np.asarray(element.get_node_coordinates(model.mesh),dtype=float)
            dofs=np.array([model.mesh.dof_manager.get_node_dofs(node)[:3] for node in element.node_ids])
            self.cells.append(_BeamCell(element_id,tuple(element.node_ids),reference,dofs,
                                        float(np.linalg.norm(reference[-1]-reference[0]))))
        self.endpoints=physical_beam_ends(model,self.cells,_beam_types()) if self.end_caps=='hemisphere' else {}
        initial=np.zeros(model.mesh.dof_manager.total_dofs) if initial_displacement is None else np.asarray(initial_displacement,dtype=float)
        if initial.shape!=(model.mesh.dof_manager.total_dofs,) or not np.all(np.isfinite(initial)):
            raise ValueError('capped cylinder initial displacement is invalid')
        scale=max(self.radius,*(cell.length for cell in self.cells))
        for cell in self.cells:
            lower=curve_gap_lower(self._positions(cell,initial),self.point,self.frame,self.obstacle_radius,
                self.half_length,self.beam_radius,accuracy=256*np.finfo(float).eps*scale,
                maximum_depth=self.settings.maximum_swept_subdivisions,admit_above=-256*np.finfo(float).eps*scale)
            if lower < -256*np.finfo(float).eps*scale: raise ValueError('capped cylinder unresolved initial penetration')

    def _gap_pose(self,positions,point,accuracy=None):
        return curve_gap_lower(positions,point,self.frame,self.obstacle_radius,self.half_length,self.beam_radius,
            accuracy=self.gap_tolerance*.1 if accuracy is None else accuracy,
            maximum_depth=self.settings.maximum_swept_subdivisions)

    def _minimum_gap(self,u):
        return min(self._gap_pose(self._positions(cell,u),self.point) for cell in self.cells)

    def evaluate(self,displacement,*,tangent=True):
        if self.end_caps!='hemisphere':
            residual,block,result=super().evaluate(displacement,tangent=tangent)
            return residual,block,_CappedResults(**asdict(result),gap_enclosure_accuracy_m=.1*self.gap_tolerance)
        u=np.asarray(displacement,dtype=float).reshape(-1)
        if u.size!=self.model.mesh.dof_manager.total_dofs or not np.all(np.isfinite(u)):
            raise ValueError('radial beam displacement is invalid')
        end_force=np.zeros_like(u);rows=[];cols=[];values=[]
        energy=0.;active=0;physical=np.zeros(3);moment=np.zeros(3);fields=[]
        for node,(cell,local,outward) in sorted(self.endpoints.items()):
            positions=self._positions(cell,u)
            point,direction=_end_geometry_controls(positions,self.point,self.frame,local,outward)
            gap=_point_gap(point,self.obstacle_radius,self.half_length,self.beam_radius)
            field={'node_id':node,'element_id':cell.element_id,'local_endpoint':local,
                'center_offset_gap_m':gap,'state':'open','physical_force_N':(0.,0.,0.)}
            fields.append(field)
            if _point_gap_sign(point,self.obstacle_radius,self.half_length,self.beam_radius)>=0:continue
            lower,upper=_cap_orientation_range(point,direction,self.obstacle_radius,self.half_length)
            if lower>0:
                field['state']='side_owned';continue
            if upper>=0:raise ValueError('active beam end cap has unresolved cap-to-side ownership')
            # Exterior distance is C1, but its Hessian has two limits at these
            # physical feature seams. Atomic endpoint work cannot ignore them.
            radial_square=point[0][0]**2+point[1][0]**2
            axial=abs(point[2][0])-_fraction(self.half_length)
            local_point=self.frame@(positions[local]-self.point)
            represented_radial=float(np.linalg.norm(local_point[:2]))-self.obstacle_radius
            represented_axial=abs(float(local_point[2]))-self.half_length
            if ((radial_square==_fraction(self.obstacle_radius)**2 and axial>0)
                    or (axial==0 and radial_square>_fraction(self.obstacle_radius)**2)
                    or (represented_radial==0. and represented_axial>0.)
                    or (represented_axial==0. and represented_radial>0.)):
                raise ValueError('active beam end cap lies on an unresolved cylinder feature seam')
            _,normal,H=signed_distance_work(positions[local],self.point,self.frame,self.obstacle_radius,self.half_length)
            force=self.endpoint_stiffness*gap*normal;dofs=cell.dofs[local]
            np.add.at(end_force,dofs,force);physical-=force;moment+=np.cross(positions[local],-force)
            energy+=.5*self.endpoint_stiffness*gap*gap;active+=1
            field['state']='owned_active';field['physical_force_N']=tuple(-force)
            if tangent:
                local_block=self.endpoint_stiffness*(np.outer(normal,normal)+gap*H)
                r,c=np.meshgrid(dofs,dofs,indexing='ij')
                rows.extend(r.reshape(-1));cols.extend(c.reshape(-1));values.extend(local_block.reshape(-1))
        residual,block,result=super().evaluate(displacement,tangent=tangent)
        result=_CappedResults(**asdict(result),gap_enclosure_accuracy_m=.1*self.gap_tolerance)
        residual+=end_force
        if tangent:block=block+sparse.coo_matrix((values,(rows,cols)),shape=block.shape).tocsr()
        combined_force=np.asarray(result.beam_force)+physical;combined_moment=np.asarray(result.beam_moment)+moment
        result=_HemisphereCappedResults(**{**asdict(result),'beam_force':tuple(combined_force),
            'rigid_force':tuple(-combined_force),'beam_moment':tuple(combined_moment),
            'rigid_moment':tuple(-combined_moment)},endpoint_contact_energy=energy,
            candidate_endpoints=len(self.endpoints),active_endpoints=active,endpoint_stiffness=self.endpoint_stiffness,
            endpoint_fields=tuple(fields))
        return residual,block,result

    def _integrate_cell(self,cell,positions):
        if self._gap_pose(positions,self.point)<=-self.beam_radius:
            raise ValueError('capped cylinder exterior profile cannot certify the whole centreline outside the solid')
        count=len(cell.node_ids); size=3*count
        points=0; maximum_error=0.; active_length=0.; active_intervals=0
        original_controls=_controls(positions,self.point,self.frame)
        def quadrature(left,right,rule):
            energy=0.; force=np.zeros(size); tangent=np.zeros((size,size)); active=0.
            for node,weight in zip(*rule):
                s=(left+right)/2+(right-left)/2*float(node)
                point_controls=_restrict(original_controls,s,s)
                if _point_gap_sign(point_controls,self.obstacle_radius,self.half_length,self.beam_radius)>=0:
                    continue
                shape=cell.shape(2*s-1)
                distance,normal,curvature=signed_distance_work(shape@positions,self.point,self.frame,
                    self.obstacle_radius,self.half_length)
                gap=_point_gap(point_controls,self.obstacle_radius,self.half_length,self.beam_radius)
                measure=(right-left)/2*cell.length*float(weight)
                gradient=np.kron(shape,normal)
                energy+=.5*self.stiffness*gap*gap*measure
                force+=self.stiffness*gap*gradient*measure
                tangent+=self.stiffness*(np.outer(gradient,gradient)+gap*np.kron(np.outer(shape,shape),curvature))*measure
                active+=measure
            return energy,force,tangent,active
        def refine(left,right,depth):
            nonlocal points,maximum_error
            restricted=_restrict(original_controls,left,right)
            if _inactive(restricted,self.obstacle_radius,self.half_length,self.beam_radius):
                return 0.,np.zeros(size),np.zeros((size,size)),0.
            low=quadrature(left,right,_GAUSS_8); high=quadrature(left,right,_GAUSS_16); points+=24
            scales=(max(abs(high[0]),1e-14*self.stiffness*self.radius**2*cell.length),
                max(np.linalg.norm(high[1]),1e-14*self.stiffness*self.radius*cell.length),
                max(np.linalg.norm(high[2]),1e-14*self.stiffness*cell.length))
            error=max(float(np.linalg.norm(np.asarray(h)-np.asarray(l)))/scale for h,l,scale in zip(high[:3],low[:3],scales))
            # An all-zero quadrature panel needs an exact inactivity proof.
            # Mixed panels still use the adaptive Gauss error estimate; this
            # guard alone is not a rigorous general integration-error bound.
            mid=(left+right)/2
            hidden=high[3]==0 and not _inactive(restricted,self.obstacle_radius,self.half_length,self.beam_radius)
            if error <= self.settings.line_integration_relative_tolerance and not hidden:
                maximum_error=max(maximum_error,error); return high
            if depth>=self.settings.maximum_line_subdivisions:
                raise MemoryError('capped cylinder line integration budget exhausted')
            first=refine(left,mid,depth+1); second=refine(mid,right,depth+1)
            return tuple(a+b for a,b in zip(first,second))
        knots=_cuts(positions,self.point,self.frame,self.obstacle_radius,self.half_length,self.beam_radius)
        total=(0.,np.zeros(size),np.zeros((size,size)),0.)
        for left,right in zip(knots[:-1],knots[1:]):
            if right<=left: continue
            result=refine(left,right,0)
            active_intervals+=result[3]>0
            total=tuple(a+b for a,b in zip(total,result))
        return (*total[:3],total[3],active_intervals,points,maximum_error)

    def _swept_admissible(self,start,end,start_factor=None,end_factor=None):
        first_point=self.point_at(self.path_factor if start_factor is None else start_factor)
        last_point=self.point_at(self.path_factor if end_factor is None else end_factor)
        for cell in self.cells:
            first=self._positions(cell,start); last=self._positions(cell,end)
            fc,lc=_controls(first,first_point,self.frame),_controls(last,last_point,self.frame)
            # Exact relative Bernstein controls in cylinder coordinates bound
            # speed along every material point of the affine solver path.
            delta=tuple(tuple(b-a for a,b in zip(f,l)) for f,l in zip(fc,lc))
            speed=_outward(max(_sqrt_bound(sum(lane[i]**2 for lane in delta),True)
                               for i in range(len(delta[0]))),math.inf)
            def gap(t):
                time=_fraction(t)
                pose=tuple(tuple((1-time)*a+time*b for a,b in zip(f,l)) for f,l in zip(fc,lc))
                return _controls_gap_lower(pose,self.obstacle_radius,self.half_length,self.beam_radius,
                    self.gap_tolerance*.1,self.settings.maximum_swept_subdivisions)
            def certify(left,right,gl,gr,depth):
                if min(gl,gr)-speed*(right-left)/2 >= -self.gap_tolerance: return True
                if depth>=self.settings.maximum_swept_subdivisions: return False
                mid=(left+right)/2; gm=gap(mid)
                if gm < -self.gap_tolerance: return False
                return certify(left,mid,gl,gm,depth+1) and certify(mid,right,gm,gr,depth+1)
            gl,gr=gap(0.),gap(1.)
            if gl < -self.gap_tolerance: raise ValueError('committed capped-cylinder pose exceeds gap tolerance')
            if gr < -self.gap_tolerance or not certify(0.,1.,gl,gr,0): return False
        return (self._end_cap_swept_admissible(start,end,first_point,last_point)
                if self.end_caps=='hemisphere' else True)

    def _end_cap_swept_admissible(self,start,end,first_point,last_point):
        """Every affine interval is open or has strictly fixed cap ownership."""
        for cell,local,outward in self.endpoints.values():
            first=_end_geometry_controls(self._positions(cell,start),first_point,self.frame,local,outward)
            last=_end_geometry_controls(self._positions(cell,end),last_point,self.frame,local,outward)
            point=tuple((a[0],b[0]) for a,b in zip(first[0],last[0]))
            direction=tuple((a[0],b[0]) for a,b in zip(first[1],last[1]))
            def certify(left,right,depth):
                pose=_restrict(point,left,right);tangent=_restrict(direction,left,right)
                if _inactive(pose,self.obstacle_radius,self.half_length,self.beam_radius):return True
                lower,upper=_cap_orientation_range(pose,tangent,self.obstacle_radius,self.half_length)
                if upper<0 or lower>0:return True
                middle=(left+right)/2
                mid_pose=_restrict(point,middle,middle);mid_tangent=_restrict(direction,middle,middle)
                lo,hi=_cap_orientation_range(mid_pose,mid_tangent,self.obstacle_radius,self.half_length)
                if (lo<=0<=hi and _point_gap_sign(mid_pose,self.obstacle_radius,
                        self.half_length,self.beam_radius)<0):return False
                if depth>=self.settings.maximum_swept_subdivisions:return False
                return certify(left,middle,depth+1) and certify(middle,right,depth+1)
            if not certify(0.,1.,0):return False
        return True

    def restart_contract(self):
        contract=super().restart_contract()
        contract['body']=asdict(self.definition.rigid_bodies[0])
        contract['geometry_revision']='signed_capped_cylinder_frame_bernstein_v1'
        if self.end_caps=='hemisphere':
            contract['interaction']['endpoint_stiffness']=self.endpoint_stiffness
            contract['end_cap_geometry_revision']='outward_hemisphere_exact_affine_ownership_v1'
            contract['physical_endpoints']=[{'node':node,'element':cell.element_id,'local':local}
                for node,(cell,local,sign) in sorted(self.endpoints.items())]
        return contract
