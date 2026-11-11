"""Opt-in continuous actual-hemisphere point springs against a fixed sphere."""
from dataclasses import dataclass, asdict, replace
from fractions import Fraction
import numpy as np
from scipy import sparse
from .core import ContactValidationReport, validate_general_contact
from .beam_radial import _BeamRadialManager
from .beam_radial_ends import BeamRadialEndManager, _EndRadialResults
from .beam_end_radial_geometry import regular_curve_path
from .beam_hemisphere_sphere_work import hemisphere_sphere_work

PROFILE='CLASSICAL_B2_B3_HEMISPHERE_SUPPORT_SPHERE_STATIC_DEV_V1'


def validate_hemisphere_sphere_contact(model,definition,settings,*,analysis_control='force'):
    if tuple(map(len,(definition.surfaces,definition.rigid_bodies,definition.interactions)))!=(1,1,1):
        return ContactValidationReport('invalid',('hemisphere sphere requires one explicit beam/sphere pair',),PROFILE)
    surface,body,law=definition.surfaces[0],definition.rigid_bodies[0],definition.interactions[0]
    issues=[]
    if surface.beam_radius is None or surface.beam_end_caps!='hemisphere' or law.endpoint_stiffness is None:
        issues.append('hemisphere sphere requires declared radius, caps and endpoint_stiffness in N/m')
    if body.kind!='sphere' or body.path_translation is not None:
        issues.append('hemisphere sphere support requires a fixed analytic sphere')
    if (surface.beam_radius is not None and body.radius is not None
            and law.gap_tolerance>=min(surface.beam_radius,body.radius)):
        issues.append('hemisphere sphere exterior gap budget must be below both radii')
    # Keep generic scalar/material/selector/control gates; no legacy geometry
    # or endpoint potential is evaluated by this validation.
    report=validate_general_contact(model,definition,replace(settings,beam_end_cap_work='owned_point'),
        analysis_control=analysis_control)
    issues.extend(report.issues)
    return ContactValidationReport('invalid' if issues else 'ok',tuple(issues),PROFILE)


@dataclass(frozen=True)
class HemisphereSphereResults(_EndRadialResults):
    def to_dict(self):
        result=super().to_dict()
        result.update(beam_end_cap_work='hemisphere_support_dev',
            beam_end_cap_geometry='actual_tangent_hemisphere_minimum_sphere_distance',
            endpoint_geometry='actual_tangent_hemisphere_minimum_sphere_distance',
            contact_integration_measure='reference_line_length_and_distinct_endpoint_point_springs',
            endpoint_integration_rule='one_minimum_support_point_per_physical_end')
        return result


class HemisphereSphereManager(BeamRadialEndManager):
    profile=PROFILE
    validator=staticmethod(validate_hemisphere_sphere_contact)
    results_type=HemisphereSphereResults

    def __init__(self,model,definition,settings,initial_displacement=None):
        report=self.validator(model,definition,settings)
        if not report.valid:raise ValueError('unsupported hemisphere sphere: '+'; '.join(report.issues))
        # Enclose the actual sum of the two literal radii even when their
        # ordinary floating addition rounds down. Caps, distributed line work
        # and all activation/bound controls must use the same physical sum.
        self.full_envelope_radius=Fraction.from_float(definition.surfaces[0].beam_radius)+Fraction.from_float(definition.rigid_bodies[0].radius)
        super().__init__(model,definition,replace(settings,beam_end_cap_work='owned_point'),initial_displacement)
        self.settings=settings;self.profile=type(self).profile
        initial=np.zeros(model.mesh.dof_manager.total_dofs) if initial_displacement is None else np.asarray(initial_displacement)
        if not self._regular(initial,initial):raise ValueError('sphere cap requires regular physical beam curves')
        self._cap_work(initial)

    def _radial_work(self,point):
        from .beam_end_radial_geometry import radial_point_work
        return radial_point_work(point,self.projector,self.full_envelope_radius)

    def _regular(self,start,end):
        return all(regular_curve_path(self._positions(cell,start),self._positions(cell,end),
            self.settings.maximum_swept_subdivisions) for cell in self.cells)

    def _cap_work(self,u):
        result=[]
        for node,(cell,local,sign) in sorted(self.endpoints.items()):
            positions=self._positions(cell,u)
            state=hemisphere_sphere_work(positions,local,sign,self.beam_radius,
                self.point,self.definition.rigid_bodies[0].radius)
            result.append((node,cell,positions,state))
        return result

    def evaluate(self,displacement,*,tangent=True):
        u=np.asarray(displacement,dtype=float).reshape(-1)
        if u.size!=self.model.mesh.dof_manager.total_dofs or not np.all(np.isfinite(u)):
            raise ValueError('sphere cap displacement is invalid')
        if not self._regular(u,u):raise ValueError('sphere cap physical curve regularity unresolved')
        self._geometry_cache={}
        try:
            # Invoke only distributed line work, with the retained exact/adaptive
            # integration overrides. Never call the legacy endpoint potential.
            residual,K,line=_BeamRadialManager.evaluate(self,u,tangent=tangent)
            energy=0.;active=0;fields=[];F=np.array(line.beam_force);M=np.array(line.beam_moment)
            rows=[];cols=[];values=[]
            for node,cell,positions,state in self._cap_work(u):
                dofs=cell.dofs.reshape(-1);gradient=state.gradient[:-3]
                if state.gap<0:
                    active+=1;energy+=.5*self.endpoint_stiffness*state.gap**2
                    local=self.endpoint_stiffness*state.gap*gradient
                    np.add.at(residual,dofs,local)
                    physical=-local.reshape(-1,3);F+=physical.sum(axis=0)
                    M+=np.cross(positions,physical).sum(axis=0)
                    if tangent:
                        block=self.endpoint_stiffness*(np.outer(gradient,gradient)+state.gap*state.hessian[:-3,:-3])
                        rows.extend(np.repeat(dofs,len(dofs)));cols.extend(np.tile(dofs,len(dofs)));values.extend(block.reshape(-1))
                fields.append({'node':node,'element':cell.element_id,'gap':state.gap,
                    'active':state.gap<0,'branch':state.branch,
                    'skin_point':None if state.skin_point is None else state.skin_point.tolist(),
                    'rigid_point':None if state.rigid_point is None else state.rigid_point.tolist()})
            if tangent and values:K=K+sparse.coo_matrix((values,(rows,cols)),shape=K.shape).tocsr()
            result=self.results_type(**{**asdict(line),'profile':self.profile,'beam_force':tuple(F),
                'rigid_force':tuple(-F),'beam_moment':tuple(M),'rigid_moment':tuple(-M)},
                endpoint_contact_energy=float(energy),candidate_endpoints=len(fields),active_endpoints=active,
                endpoint_stiffness=self.endpoint_stiffness,endpoint_fields=tuple(fields),
                gap_enclosure_accuracy_m=.1*self.gap_tolerance)
            return residual,K,result
        finally:self._geometry_cache=None

    def _swept_admissible(self,start,end,start_factor=None,end_factor=None):
        # Every physical hemisphere is inside its endpoint radius envelope.
        # Full curve/radius certification and tangent regularity contain both
        # support branches, including an active C1 seam crossing.
        # Resolve spatial accuracy before subtracting the temporal speed bound;
        # early stopping at -gap_tolerance can leave no temporal margin.
        return self._swept_full_envelope_admissible(start,end,start_factor,end_factor,resolve_gap_bounds=True)

    def restart_contract(self):
        result=super().restart_contract()
        result.update(geometry_revision='actual_tangent_hemisphere_sphere_rim_work_v1',
            beam_end_cap_work='hemisphere_support_dev',
            full_envelope_revision='literal_radial_map_resolved_bernstein_full_radius_v1',
            full_envelope_radius_addition='exact_binary_radii_sum',
            line_radius_work_revision='exact_literal_sum_activation_and_work_v2',
            end_topology_revision='all_classical_material_node_incidence_v1')
        result['settings']['beam_end_cap_work']='hemisphere_support_dev'
        return result
