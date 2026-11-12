"""Opt-in actual hemisphere support on a deforming interior S3 physical skin.

All beam and skin positions participate in the same local potential. The
existing skin work map pulls that work back to authoritative coordinates.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse

from .core import ContactValidationReport, validate_general_contact
from .deformable_beam_s3_static import DeformableBeamS3Manager, DeformableBeamS3Results
from .beam_end_topology import physical_beam_ends
from .beam_plane import _BeamCell, _beam_types
from .beam_end_radial_geometry import regular_curve_path

PROFILE = 'JET2_B2_B3_HEMISPHERE_SUPPORT_DEFORMABLE_FIXED_DIRECTOR_S3_STATIC_DEV_V1'


def _line_definition(definition):
    return replace(definition,
        surfaces=tuple(replace(s, beam_end_caps='none') if s.beam_radius is not None else s
                       for s in definition.surfaces),
        interactions=tuple(replace(i, endpoint_stiffness=None) for i in definition.interactions))


def validate_deformable_beam_s3_caps(model, definition, settings, *, analysis_control='force'):
    issues = []
    beams = [s for s in definition.surfaces if s.beam_radius is not None]
    if (definition.rigid_bodies or len(definition.surfaces) != 2 or len(beams) != 1
            or len(definition.interactions) != 1):
        return ContactValidationReport('invalid', ('hemisphere S3 support requires one explicit beam/S3 pair',), PROFILE)
    beam, law = beams[0], definition.interactions[0]
    if beam.beam_end_caps != 'hemisphere' or law.endpoint_stiffness is None:
        issues.append('hemisphere S3 support requires declared caps and endpoint_stiffness in N/m')
    if settings.beam_shell_geometry != 'jet2_deformable_s3_dev':
        issues.append('hemisphere S3 support requires the deformable Jet2 S3 work map')
    if law.gap_tolerance >= beam.beam_radius:
        issues.append('hemisphere S3 exterior support requires gap_tolerance < beam_radius')
    report = validate_general_contact(model, _line_definition(definition),
        replace(settings, beam_end_cap_work='owned_point'), analysis_control=analysis_control)
    issues.extend(report.issues)
    return ContactValidationReport('invalid' if issues else 'ok', tuple(issues), PROFILE)


@dataclass(frozen=True)
class DeformableBeamS3CapResults:
    base: object
    endpoint_energy: float
    endpoint_stiffness: float
    endpoint_fields: tuple

    def __getattr__(self, name):
        return getattr(self.base, name)

    def to_dict(self):
        result = self.base.to_dict()
        result.update(beam_end_cap_work='hemisphere_support_dev',
            endpoint_contact_energy=self.endpoint_energy,
            algorithmic_contact_potential=self.base.line_contact_energy+self.endpoint_energy,
            endpoint_stiffness_N_per_m=self.endpoint_stiffness,
            active_endpoints=sum(row['active'] for row in self.endpoint_fields),
            candidate_endpoints=len(self.endpoint_fields),endpoint_fields=list(self.endpoint_fields),
            endpoint_geometry='actual_tangent_hemisphere_minimum_moving_skin_plane_gap',
            contact_integration_measure='reference_line_length_and_distinct_endpoint_point_springs',
            line_integration_rule='three_gauss_points_per_exact_active_interval',
            endpoint_integration_rule='one_minimum_support_point_per_physical_end')
        return result


class DeformableBeamS3CapManager(DeformableBeamS3Manager):
    def __init__(self, model, definition, settings, initial_displacement=None):
        report = validate_deformable_beam_s3_caps(model, definition, settings)
        if not report.valid:
            raise ValueError('unsupported deformable hemisphere S3: ' + '; '.join(report.issues))
        super().__init__(model, _line_definition(definition),
                         replace(settings, beam_end_cap_work='owned_point'), initial_displacement)
        self.definition, self.settings, self.profile = definition, settings, PROFILE
        self.beam_surface = next(s for s in definition.surfaces if s.beam_radius is not None)
        self.end_caps = 'hemisphere'
        self.endpoint_stiffness = definition.interactions[0].endpoint_stiffness
        cell = _BeamCell(self.beam_surface.element_ids[0], self.beam_ids,
                        self.beam_reference,self.beam_dofs[:,:3],self.kernel.length)
        self.cap_endpoints = physical_beam_ends(model,(cell,),_beam_types())
        u = np.zeros(self.work_map.shape[1]) if initial_displacement is None else initial_displacement
        beam, skin = self._positions(u)
        if not regular_curve_path(beam,beam,settings.maximum_swept_subdivisions):
            raise ValueError('hemisphere S3 requires a regular physical beam curve')
        self._endpoint_work(beam,skin)

    def _endpoint_work(self, beam, skin):
        from .beam_hemisphere_s3_support import hemisphere_s3_work
        size = 3*(len(beam)+3)
        energy = 0.; gradient = np.zeros(size); hessian = np.zeros((size,size)); fields=[]
        for node,(_,local,sign) in sorted(self.cap_endpoints.items()):
            state = hemisphere_s3_work(beam,skin,local,sign,self.kernel.radius,self.kernel.orientation)
            if state.gap < 0:
                energy += .5*self.endpoint_stiffness*state.gap**2
                gradient += self.endpoint_stiffness*state.gap*state.gradient
                hessian += self.endpoint_stiffness*(np.outer(state.gradient,state.gradient)+state.gap*state.hessian)
            fields.append({'node':node,'element':self.beam_surface.element_ids[0],
                'gap':state.gap,'active':state.gap<0,'branch':state.branch,
                'skin_point':None if state.skin_point is None else state.skin_point.tolist()})
        return float(energy),gradient,hessian,tuple(fields)

    def evaluate(self, displacement, *, tangent=True):
        beam, skin = self._positions(displacement)
        if not regular_curve_path(beam,beam,self.settings.maximum_swept_subdivisions):
            raise ValueError('hemisphere S3 candidate has unresolved physical curve regularity')
        line = self.kernel.evaluate(beam,skin)
        energy, gradient, hessian, fields = self._endpoint_work(beam,skin)
        gradient += line.gradient; hessian += line.hessian
        residual = np.asarray(self.work_map.T@gradient).reshape(-1)
        K = (self.work_map.T@sparse.csr_matrix(hessian)@self.work_map).tocsr() if tangent else None
        physical = -gradient.reshape(-1,3); nb=len(beam)
        bf,sf = physical[:nb].sum(axis=0),physical[nb:].sum(axis=0)
        bm,sm = np.cross(beam,physical[:nb]).sum(axis=0),np.cross(skin,physical[nb:]).sum(axis=0)
        base = DeformableBeamS3Results(PROFILE,line.minimum_gap,line.potential,
            line.active_line_length,len(line.active_intervals),line.integration_points,
            0.,line.minimum_rim_clearance,tuple(bf),tuple(sf),tuple(bm),tuple(sm),
            line.minimum_gap >= -self.gap_tolerance,self.line_stiffness,
            self.definition.interactions[0].name,self.beam_surface.name,self.shell_surface.name,
            self.beam_surface.element_ids[0],self.shell_surface.element_ids[0])
        return residual,K,DeformableBeamS3CapResults(base,energy,self.endpoint_stiffness,fields)

    def _swept(self, start, end):
        b0,s0=self._positions(start);b1,s1=self._positions(end)
        if not regular_curve_path(b0,b1,self.settings.maximum_swept_subdivisions):
            return False
        # The full radius envelope bounds every hemisphere, independently of
        # its changing tangent. Existing moving-normal/rim bounds cover skins.
        return self.kernel.swept_admissible(b0,s0,b1,s1,gap_tolerance=self.gap_tolerance,
            maximum_intervals=self.settings.beam_shell_maximum_swept_intervals)

    def restart_contract(self):
        result = super().restart_contract()
        result.update(profile=PROFILE,revision='complete_moving_normal_hemisphere_reference_line_v1',
            beam_end_cap_work='hemisphere_support_dev',
            physical_endpoints=[{'node':node,'element':self.beam_surface.element_ids[0],'local':local}
                for node,(_,local,sign) in sorted(self.cap_endpoints.items())],
            end_topology_revision='all_classical_material_node_incidence_v1')
        result['interaction']['endpoint_stiffness'] = self.endpoint_stiffness
        return result
