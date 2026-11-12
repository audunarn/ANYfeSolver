"""Actual hemisphere/cylinder endpoint work and exact-axis line geometry.

This explicit profile leaves historical literal-projector line routes intact.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
import numpy as np
from .core import ContactValidationReport, validate_general_contact
from .beam_hemisphere_sphere import HemisphereSphereManager, HemisphereSphereResults
from .beam_hemisphere_cylinder_work import hemisphere_cylinder_work
from .beam_end_radial_geometry import curve_gap_lower, relative_speed_upper, _fraction
from .capped_cylinder import _sqrt_bound

PROFILE='CLASSICAL_B2_B3_HEMISPHERE_SUPPORT_CYLINDER_STATIC_DEV_V1'


def validate_hemisphere_cylinder_contact(model,definition,settings,*,analysis_control='force'):
    if tuple(map(len,(definition.surfaces,definition.rigid_bodies,definition.interactions)))!=(1,1,1):
        return ContactValidationReport('invalid',('hemisphere cylinder requires one explicit beam/cylinder pair',),PROFILE)
    surface,body,law=definition.surfaces[0],definition.rigid_bodies[0],definition.interactions[0]
    issues=[]
    if surface.beam_radius is None or surface.beam_end_caps!='hemisphere' or law.endpoint_stiffness is None:
        issues.append('hemisphere cylinder requires declared radius/caps and endpoint stiffness in N/m')
    if body.kind!='cylinder' or body.path_translation is not None:
        issues.append('hemisphere cylinder support requires a fixed infinite cylinder')
    if surface.beam_radius is not None and body.radius is not None and law.gap_tolerance>=min(surface.beam_radius,body.radius):
        issues.append('hemisphere cylinder exterior gap budget must be below both radii')
    report=validate_general_contact(model,definition,replace(settings,beam_end_cap_work='owned_point'),analysis_control=analysis_control)
    issues.extend(report.issues)
    return ContactValidationReport('invalid' if issues else 'ok',tuple(issues),PROFILE)


@dataclass(frozen=True)
class HemisphereCylinderResults(HemisphereSphereResults):
    def to_dict(self):
        result=super().to_dict()
        result.update(beam_end_cap_geometry='actual_tangent_hemisphere_minimum_cylinder_axis_distance',
            endpoint_geometry='actual_tangent_hemisphere_minimum_cylinder_axis_distance',
            geometry_kinematics='exact_literal_axis_projection_affine_static_v1')
        return result


def exact_axis_projector(axis):
    a=tuple(_fraction(v) for v in axis); aa=sum(v*v for v in a)
    if aa==0:raise ValueError('cylinder bound axis is zero')
    return tuple(tuple(Fraction(int(i==j))-a[i]*a[j]/aa for j in range(3)) for i in range(3))


def exact_cylinder_controls(positions,center,axis):
    """Exact Bernstein projection perpendicular to the literal nonzero axis."""
    projection=exact_axis_projector(axis)
    relative=tuple(tuple(_fraction(v)-_fraction(c) for v,c in zip(point,center)) for point in positions)
    lanes=[]
    for direction in projection:
        values=tuple(sum(v*d for v,d in zip(point,direction)) for point in relative)
        if len(values)==3:values=(values[0],2*values[1]-(values[0]+values[2])/2,values[2])
        lanes.append(values)
    return tuple(lanes)


class HemisphereCylinderManager(HemisphereSphereManager):
    profile=PROFILE
    validator=staticmethod(validate_hemisphere_cylinder_contact)
    results_type=HemisphereCylinderResults

    def _axis_controls(self,positions,center):
        return exact_cylinder_controls(positions,center,self.definition.rigid_bodies[0].normal)

    def _geometry(self,cell,positions,point=None):
        center=self.point if point is None else point
        # Fractions from an affine tool path have value identity, not object
        # array pointer identity. Fixed float centres use the same exact key.
        key=(cell.element_id,positions.tobytes(),tuple(_fraction(v) for v in center))
        if self._geometry_cache is not None and key in self._geometry_cache:return self._geometry_cache[key]
        lanes=self._axis_controls(positions,center)
        scale=max(self.radius,cell.length)
        accuracy=256*np.finfo(float).eps*scale if self._initializing else self.gap_tolerance*.1
        # Initialization needs a nonpenetration certificate, not a roundoff-
        # resolved minimum of an already open interior closest approach.
        # Only a conservative nonnegative bound may stop refinement early;
        # unresolved or penetrating curves retain the original accuracy/budget.
        lower=curve_gap_lower(lanes,getattr(self,'full_envelope_radius',self.radius),accuracy,
            self.settings.maximum_swept_subdivisions,admit_above=0. if self._initializing else None)
        result=lanes,lower
        if self._geometry_cache is not None:self._geometry_cache[key]=result
        return result

    def _radial_work(self,point):
        values=tuple(lane[0] for lane in point); squared=sum(v*v for v in values)
        if squared<=0:raise ValueError('cylinder line normal is unresolved')
        rho=_sqrt_bound(squared); R=self.full_envelope_radius
        gap=float((squared-R*R)/(rho+R)); distance=float(rho)
        normal=np.array([float(v/rho) for v in values])
        projector=np.array([[float(v) for v in row] for row in exact_axis_projector(self.definition.rigid_bodies[0].normal)])
        H=(projector-np.outer(normal,normal))/distance
        return gap,normal,H

    def _cap_work(self,u):
        result=[]; body=self.definition.rigid_bodies[0]
        for node,(cell,local,sign) in sorted(self.endpoints.items()):
            positions=self._positions(cell,u)
            state=hemisphere_cylinder_work(positions,local,sign,self.beam_radius,self.point,body.normal,body.radius)
            result.append((node,cell,positions,state))
        return result

    def _swept_admissible(self,start,end,start_factor=None,end_factor=None):
        if not self._regular(start,end):return False
        center0=self.point_at(self.path_factor if start_factor is None else start_factor)
        center1=self.point_at(self.path_factor if end_factor is None else end_factor)
        for cell in self.cells:
            a=self._axis_controls(self._positions(cell,start),center0)
            b=self._axis_controls(self._positions(cell,end),center1)
            speed=relative_speed_upper(a,b)
            def gap(t):
                lanes=tuple(tuple(x+t*(y-x) for x,y in zip(first,last)) for first,last in zip(a,b))
                return _fraction(curve_gap_lower(lanes,self.full_envelope_radius,self.gap_tolerance*.1,
                    self.settings.maximum_swept_subdivisions))
            def certify(left,right,gl,gr,depth):
                if min(gl,gr)-speed*(right-left)/2>=-_fraction(self.gap_tolerance):return True
                if depth>=self.settings.maximum_swept_subdivisions:return False
                mid=(left+right)/2; gm=gap(mid)
                if gm<-_fraction(self.gap_tolerance):return False
                return certify(left,mid,gl,gm,depth+1) and certify(mid,right,gm,gr,depth+1)
            gl,gr=gap(Fraction(0)),gap(Fraction(1))
            if gl<-_fraction(self.gap_tolerance):raise ValueError('committed cylinder cap pose exceeds gap tolerance')
            if gr<-_fraction(self.gap_tolerance) or not certify(Fraction(0),Fraction(1),gl,gr,0):return False
        return True

    def restart_contract(self):
        result=super().restart_contract()
        result.update(geometry_revision='actual_tangent_hemisphere_cylinder_implicit_rim_v1',
            full_envelope_revision='exact_literal_axis_line_and_cap_bernstein_v1',
            stationary_revision='quartic_one_root_intervals_and_implicit_hessian_v1')
        return result
