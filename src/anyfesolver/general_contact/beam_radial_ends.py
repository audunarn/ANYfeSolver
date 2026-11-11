"""Round classical-beam physical ends against prescribed radial obstacles."""
from dataclasses import dataclass,asdict,replace
from fractions import Fraction
import math

import numpy as np
from scipy import sparse
from .beam_radial import (_BeamRadialManager,_RadialResults,validate_beam_radial_contact,
    _real_interval_roots,_distance_polynomial,_GAUSS_8,_GAUSS_16)
from .beam_plane import _beam_types
from .beam_end_topology import physical_beam_ends
from .beam_end_radial_geometry import (squared_controls,radial_inactive,point_sign,
    radial_point_work,end_controls,orientation_controls,ownership_path_admissible,
    curve_gap_lower,relative_speed_upper,radial_cuts,regular_curve_path,_controls,_restrict,_fraction)
from .core import ContactSolverSettings,ContactValidationReport

PROFILE='CLASSICAL_B2_B3_ROUND_TUBE_RADIAL_HEMISPHERE_ENDS_STATIC_DEV_V1'


def _line_definition(definition):
    return replace(definition,surfaces=(replace(definition.surfaces[0],beam_end_caps='none'),),
        interactions=(replace(definition.interactions[0],endpoint_stiffness=None),))


def validate_radial_beam_ends(model,definition,settings=None):
    if tuple(map(len,(definition.surfaces,definition.rigid_bodies,definition.interactions)))!=(1,1,1):
        return ContactValidationReport('invalid',('one explicit beam surface and radial obstacle required',),PROFILE)
    surface,law=definition.surfaces[0],definition.interactions[0]
    if surface.beam_end_caps!='hemisphere' or law.endpoint_stiffness is None:
        return ContactValidationReport('invalid',('hemisphere ends need explicit endpoint_stiffness in N/m',),PROFILE)
    if surface.beam_radius is None or law.gap_tolerance>=surface.beam_radius:
        return ContactValidationReport('invalid',('radial end exterior profile needs gap_tolerance below beam radius',),PROFILE)
    report=validate_beam_radial_contact(model,_line_definition(definition),settings)
    return ContactValidationReport(report.status,report.issues,PROFILE)


@dataclass(frozen=True)
class _EndRadialResults(_RadialResults):
    endpoint_contact_energy: float
    candidate_endpoints: int
    active_endpoints: int
    endpoint_stiffness: float
    endpoint_fields: tuple
    gap_enclosure_accuracy_m: float

    def to_dict(self):
        return {**super().to_dict(),'endpoint_contact_energy':self.endpoint_contact_energy,
            'algorithmic_contact_potential':self.line_contact_energy+self.endpoint_contact_energy,
            'candidate_endpoints':self.candidate_endpoints,'active_endpoints':self.active_endpoints,
            'endpoint_stiffness_N_per_m':self.endpoint_stiffness,
            'endpoint_fields':[dict(field) for field in self.endpoint_fields],
            'minimum_gap_kind':'conservative_full_curve_lower_bound',
            'gap_enclosure_accuracy_m':self.gap_enclosure_accuracy_m,
            'moment_origin':[0.,0.,0.],'moment_units':'N*m',
            'beam_end_cap_geometry':'outward_hemisphere_point_penalty',
            'geometry_kinematics':'literal_fixed_radial_map_affine_static_v1'}


class BeamRadialEndManager(_BeamRadialManager):
    def __init__(self,model,definition,settings=None,initial_displacement=None):
        controls=ContactSolverSettings() if settings is None else settings
        report=validate_radial_beam_ends(model,definition,controls)
        if not report.valid:raise ValueError('unsupported radial beam ends: '+'; '.join(report.issues))
        self._initializing=True;self._geometry_cache=None
        super().__init__(model,_line_definition(definition),controls,initial_displacement)
        self._initializing=False
        self.definition=definition;self.profile=PROFILE;self.end_caps='hemisphere'
        self.endpoint_stiffness=definition.interactions[0].endpoint_stiffness
        self.beam_radius=definition.surfaces[0].beam_radius
        self.endpoints=physical_beam_ends(model,self.cells,_beam_types())

    def _geometry(self,cell,positions,point=None):
        center=self.point if point is None else point
        key=(cell.element_id,positions.tobytes(),center.tobytes())
        if self._geometry_cache is not None and key in self._geometry_cache:return self._geometry_cache[key]
        lanes=_controls(positions,center,self.projector)
        scale=max(self.radius,cell.length)
        accuracy=(256*np.finfo(float).eps*scale if self._initializing else self.gap_tolerance*.1)
        lower=curve_gap_lower(lanes,getattr(self,'full_envelope_radius',self.radius),
            accuracy,self.settings.maximum_swept_subdivisions)
        result=(lanes,lower)
        if self._geometry_cache is not None:self._geometry_cache[key]=result
        return result

    def _minimum_gap(self,u):
        return min(self._geometry(cell,self._positions(cell,u))[1] for cell in self.cells)

    def _radial_work(self,point):
        return radial_point_work(point,self.projector,self.radius)

    def _integrate_cell(self,cell,positions):
        lanes,minimum=self._geometry(cell,positions)
        if minimum<=-self.beam_radius:raise ValueError('radial end profile cannot certify centreline exterior to obstacle')
        size=3*len(cell.node_ids);points=0;maximum_error=0.;intervals=0
        # Actual-hemisphere profiles share the exact literal radius sum with
        # their caps/bounds. Historical owned-point routes retain their law.
        radius=getattr(self,'full_envelope_radius',self.radius)
        # Floating roots are subdivision proposals, never inactivity certificates.
        polynomial=_distance_polynomial(positions,np.asarray(self.point,dtype=float),self.projector.T@self.projector,self.radius)
        cuts=radial_cuts(lanes,radius,_real_interval_roots(polynomial))
        def quadrature(left,right,rule):
            nonlocal points
            E=0.;f=np.zeros(size);K=np.zeros((size,size));active=0.
            for node,weight in zip(*rule):
                s=(left+right)/2+(right-left)/2*float(node)
                if isinstance(radius,Fraction) and s in (left,right):
                    raise ArithmeticError('radial end quadrature station is unresolved')
                p=_restrict(lanes,s,s)
                if point_sign(p,radius)>=0:continue
                gap,n,H=self._radial_work(p)
                shape=cell.shape(2*s-1);gradient=np.outer(shape,n).reshape(-1)
                measure=cell.length*(right-left)/2*float(weight)
                E+=.5*self.stiffness*gap*gap*measure
                f+=self.stiffness*gap*measure*gradient
                hessian=np.kron(np.outer(shape,shape),H)
                K+=self.stiffness*measure*(np.outer(gradient,gradient)+gap*hessian)
                active+=measure
            points+=len(rule[0]);return E,f,K,active
        def refine(left,right,depth):
            nonlocal maximum_error,intervals
            panel=_restrict(lanes,left,right)
            if radial_inactive(panel,radius):return 0.,np.zeros(size),np.zeros((size,size)),0.
            signs=[(q>0)-(q<0) for q in (v-_fraction(radius)**2 for v in squared_controls(panel)) if q]
            variations=sum(a!=b for a,b in zip(signs,signs[1:]))
            low=quadrature(left,right,_GAUSS_8);high=quadrature(left,right,_GAUSS_16)
            error=max(abs(low[0]-high[0])/max(abs(high[0]),1e-30),
                np.linalg.norm(low[1]-high[1])/max(np.linalg.norm(high[1]),1e-30),
                np.linalg.norm(low[2]-high[2])/max(np.linalg.norm(high[2]),1e-30))
            # More than one possible front must be separated even if some
            # quadrature points detect a different pocket in the same panel.
            hidden=high[3]==0 or variations>1
            if error<=self.settings.line_integration_relative_tolerance and not hidden:
                maximum_error=max(maximum_error,error);intervals+=1;return high
            if depth>=self.settings.maximum_line_subdivisions:
                raise MemoryError('radial end line integration budget exhausted')
            middle=(left+right)/2
            if middle in (left,right):raise ArithmeticError('radial end integration panel is unresolved')
            a=refine(left,middle,depth+1);b=refine(middle,right,depth+1)
            return tuple(x+y for x,y in zip(a,b))
        total=(0.,np.zeros(size),np.zeros((size,size)),0.)
        for left,right in zip(cuts[:-1],cuts[1:]):
            part=refine(left,right,0);total=tuple(x+y for x,y in zip(total,part))
        return (*total,intervals,points,maximum_error)

    def evaluate(self,displacement,*,tangent=True):
        u=np.asarray(displacement,dtype=float).reshape(-1)
        if u.size!=self.model.mesh.dof_manager.total_dofs or not np.all(np.isfinite(u)):
            raise ValueError('radial beam end displacement is invalid')
        self._geometry_cache={}
        try:
            extra=np.zeros_like(u);rows=[];cols=[];values=[];fields=[]
            energy=0.;active=0;physical=np.zeros(3);moment=np.zeros(3)
            for node,(cell,local,outward) in sorted(self.endpoints.items()):
                positions=self._positions(cell,u)
                point,derivative=end_controls(positions,self.point,self.projector,local,outward)
                gap,n,H=radial_point_work(point,self.projector,self.radius)
                field={'node_id':node,'element_id':cell.element_id,'local_endpoint':local,
                    'center_offset_gap_m':gap,'state':'open','physical_force_N':(0.,0.,0.)}
                fields.append(field)
                if point_sign(point,self.radius)>=0:continue
                sign=orientation_controls(point,derivative)[0]
                if sign>0:field['state']='side_owned';continue
                if sign==0:raise ValueError('active radial beam end has unresolved cap-to-side ownership')
                force=self.endpoint_stiffness*gap*n;dofs=cell.dofs[local]
                np.add.at(extra,dofs,force);physical-=force;moment+=np.cross(positions[local],-force)
                energy+=.5*self.endpoint_stiffness*gap*gap;active+=1
                field['state']='owned_active';field['physical_force_N']=tuple(-force)
                if tangent:
                    block=self.endpoint_stiffness*(np.outer(n,n)+gap*H)
                    r,c=np.meshgrid(dofs,dofs,indexing='ij')
                    rows.extend(r.reshape(-1));cols.extend(c.reshape(-1));values.extend(block.reshape(-1))
            residual,K,result=super().evaluate(u,tangent=tangent)
            residual+=extra
            if tangent:K=K+sparse.coo_matrix((values,(rows,cols)),shape=K.shape).tocsr()
            F=np.asarray(result.beam_force)+physical;M=np.asarray(result.beam_moment)+moment
            result=_EndRadialResults(**{**asdict(result),'beam_force':tuple(F),'rigid_force':tuple(-F),
                'beam_moment':tuple(M),'rigid_moment':tuple(-M)},endpoint_contact_energy=energy,
                candidate_endpoints=len(self.endpoints),active_endpoints=active,endpoint_stiffness=self.endpoint_stiffness,
                endpoint_fields=tuple(fields),gap_enclosure_accuracy_m=.1*self.gap_tolerance)
            return residual,K,result
        finally:self._geometry_cache=None

    def _swept_full_envelope_admissible(self,start,end,start_factor=None,end_factor=None,*,resolve_gap_bounds=False):
        first_center=self.point_at(self.path_factor if start_factor is None else start_factor)
        last_center=self.point_at(self.path_factor if end_factor is None else end_factor)
        for cell in self.cells:
            first=self._positions(cell,start);last=self._positions(cell,end)
            if not regular_curve_path(first,last,self.settings.maximum_swept_subdivisions):return False
            a=_controls(first,first_center,self.projector);b=_controls(last,last_center,self.projector)
            speed=relative_speed_upper(a,b)
            def gap(t):
                lanes=tuple(tuple(x+_fraction(t)*(y-x) for x,y in zip(l,m)) for l,m in zip(a,b))
                return _fraction(curve_gap_lower(lanes,getattr(self,'full_envelope_radius',self.radius),self.gap_tolerance*.1,
                    self.settings.maximum_swept_subdivisions,
                    admit_above=None if resolve_gap_bounds else -self.gap_tolerance))
            def certify(left,right,gl,gr,depth):
                if min(gl,gr)-speed*(right-left)/2>=-_fraction(self.gap_tolerance):return True
                if depth>=self.settings.maximum_swept_subdivisions:return False
                middle=(left+right)/2;gm=gap(middle)
                if gm<-_fraction(self.gap_tolerance):return False
                return certify(left,middle,gl,gm,depth+1) and certify(middle,right,gm,gr,depth+1)
            gl,gr=gap(Fraction(0)),gap(Fraction(1))
            if gl<-_fraction(self.gap_tolerance):raise ValueError('committed radial end pose exceeds gap tolerance')
            if gr<-_fraction(self.gap_tolerance) or not certify(Fraction(0),Fraction(1),gl,gr,0):return False
        return True

    def _swept_admissible(self,start,end,start_factor=None,end_factor=None):
        if not self._swept_full_envelope_admissible(start,end,start_factor,end_factor):return False
        first_center=self.point_at(self.path_factor if start_factor is None else start_factor)
        last_center=self.point_at(self.path_factor if end_factor is None else end_factor)
        for cell,local,outward in self.endpoints.values():
            a=end_controls(self._positions(cell,start),first_center,self.projector,local,outward)
            b=end_controls(self._positions(cell,end),last_center,self.projector,local,outward)
            if not ownership_path_admissible(a,b,self.radius,self.settings.maximum_swept_subdivisions):return False
        return True

    def restart_contract(self):
        contract=super().restart_contract()
        contract['interaction']['endpoint_stiffness']=self.endpoint_stiffness
        contract['geometry_revision']='literal_radial_map_bernstein_hemisphere_v1'
        contract['physical_endpoints']=[{'node':node,'element':cell.element_id,'local':local}
            for node,(cell,local,sign) in sorted(self.endpoints.items())]
        return contract
