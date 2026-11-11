"""Opt-in fixed-plane hemisphere support spring with continuous rim transition.

This is a different endpoint potential from the legacy owned-point law. The
distributed line law retains its reference length and N/m² regularization;
the actual hemisphere minimum-gap spring has its own N/m regularization.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math

import numpy as np
from scipy import sparse

from .core import ContactValidationReport, ContactSolverSettings, validate_general_contact
from .beam_plane import _BeamPlaneManager, _beam_types, validate_beam_plane_contact
from .beam_end_topology import physical_beam_ends
from .beam_end_radial_geometry import regular_curve_path
from .beam_hemisphere_support import hemisphere_plane_work
from .capped_cylinder import _sqrt_bound

PROFILE = 'CLASSICAL_B2_B3_HEMISPHERE_SUPPORT_PLANE_STATIC_DEV_V1'


def _line_definition(definition):
    return replace(definition,
        surfaces=tuple(replace(s, beam_end_caps='none') for s in definition.surfaces),
        interactions=tuple(replace(i, endpoint_stiffness=None) for i in definition.interactions))


def validate_hemisphere_plane_contact(model, definition, settings=None, *, analysis_control='force'):
    controls = ContactSolverSettings() if settings is None else settings
    issues = []
    if (len(definition.surfaces), len(definition.rigid_bodies),
            len(definition.interactions)) != (1, 1, 1):
        return ContactValidationReport('invalid',
            ('hemisphere support requires one classical beam selection and one fixed plane',), PROFILE)
    surface, body, law = (definition.surfaces[0], definition.rigid_bodies[0],
                          definition.interactions[0])
    if surface.beam_radius is None or surface.beam_end_caps != 'hemisphere':
        issues.append('hemisphere support requires declared round radius and hemisphere caps')
    if law.endpoint_stiffness is None:
        issues.append('hemisphere support requires endpoint_stiffness in N/m')
    if body.kind != 'plane' or body.path_translation is not None:
        issues.append('hemisphere support requires one fixed analytic plane')
    if surface.beam_radius is not None and law.gap_tolerance >= surface.beam_radius:
        issues.append('hemisphere support exterior profile requires gap_tolerance < beam_radius')
    if controls.beam_chain_tangent != 'finite_difference':
        issues.append('hemisphere plane does not use the beam-pair chain tangent selection')
    # Retain every generic compatibility gate by selecting the legacy law for
    # validation only. Its geometry is not used by this contributor.
    report = validate_general_contact(model, definition,
        replace(controls, beam_end_cap_work='owned_point'), analysis_control=analysis_control)
    issues.extend(report.issues)
    return ContactValidationReport('invalid' if issues else 'ok', tuple(issues), PROFILE)


@dataclass(frozen=True)
class HemispherePlaneResults:
    base: object
    endpoint_fields: tuple[dict, ...]

    def to_dict(self):
        result = self.base.to_dict()
        result['beam_end_cap_work'] = 'hemisphere_support_dev'
        result['endpoint_geometry'] = 'actual_tangent_hemisphere_minimum_plane_gap'
        result['endpoint_fields'] = list(self.endpoint_fields)
        return result

    def __getattr__(self, name):
        return getattr(self.base, name)


class HemispherePlaneManager(_BeamPlaneManager):
    def __init__(self, model, definition, settings=None, initial_displacement=None):
        controls = ContactSolverSettings() if settings is None else settings
        report = validate_hemisphere_plane_contact(model, definition, controls)
        if not report.valid:
            raise ValueError('unsupported hemisphere-plane contact: ' + '; '.join(report.issues))
        self.cap_normal = np.asarray(definition.rigid_bodies[0].normal, dtype=float)
        super().__init__(model, _line_definition(definition), controls, initial_displacement)
        self.definition = definition
        self.profile = PROFILE
        self.endpoint_stiffness = definition.interactions[0].endpoint_stiffness
        self.cap_endpoints = physical_beam_ends(model, self.cells, _beam_types())
        # Parent remains a line-only contributor. Unit normal and plane height
        # use the same literal-normal convention as the hemisphere helper.
        nn = sum(Fraction.from_float(float(n))**2 for n in self.cap_normal)
        norm = float(_sqrt_bound(nn))
        self.normal = self.cap_normal / norm
        initial = (np.zeros(model.mesh.dof_manager.total_dofs) if initial_displacement is None
                   else np.asarray(initial_displacement, dtype=float).reshape(-1))
        if not self._end_ownership_admissible(initial, initial):
            raise ValueError('hemisphere plane requires a regular physical beam curve')

    def _minimum_gap(self, u):
        # Exact polynomial minimum of literal plane height, divided outward by
        # the exact normal norm. The complete radius envelope contains caps.
        normal = tuple(Fraction.from_float(float(n)) for n in self.cap_normal)
        point = tuple(Fraction.from_float(float(p)) for p in self.point)
        nn = sum(n*n for n in normal)
        minimum = None
        for cell in self.cells:
            positions = cell.reference + u[cell.dofs]
            h = [sum((Fraction.from_float(float(x))-p)*n for x,p,n in zip(row,point,normal))
                 for row in positions]
            value = min(h[0], h[-1])
            if len(h) == 3:
                a, b = 2*h[0]-4*h[1]+2*h[2], -3*h[0]+4*h[1]-h[2]
                if a > 0 and 0 < -b/(2*a) < 1:
                    value = min(value, h[0]-b*b/(4*a))
            height = value/_sqrt_bound(nn, value >= 0)
            gap = height-Fraction.from_float(self.radius)
            minimum = gap if minimum is None else min(minimum, gap)
        value = float(minimum)
        return math.nextafter(value, -math.inf) if Fraction.from_float(value) > minimum else value

    def _end_ownership_admissible(self, start, end):
        # No ownership switch: both support branches belong to the same cap.
        return all(regular_curve_path(cell.reference+start[cell.dofs],
            cell.reference+end[cell.dofs], self.settings.maximum_swept_subdivisions)
            for cell in self.cells)

    def evaluate(self, displacement, *, tangent=True):
        u = np.asarray(displacement, dtype=float).reshape(-1)
        if u.size != self.model.mesh.dof_manager.total_dofs or not np.all(np.isfinite(u)):
            raise ValueError('hemisphere plane displacement is invalid')
        if not self._end_ownership_admissible(u, u):
            raise ValueError('hemisphere plane candidate has unresolved physical curve regularity')
        force, matrix, result = super().evaluate(u, tangent=tangent)
        energy = 0.; active = 0; fields = []
        rows = []; cols = []; values = []
        wrench = np.zeros(3); moment = np.zeros(3)
        for node, (cell, local, sign) in sorted(self.cap_endpoints.items()):
            positions = cell.reference+u[cell.dofs]
            state = hemisphere_plane_work(positions, local, sign, self.radius,
                                           self.point, self.cap_normal)
            if state.gap < 0:
                active += 1
                energy += .5*self.endpoint_stiffness*state.gap**2
                local_force = self.endpoint_stiffness*state.gap*state.gradient
                dofs = cell.dofs.reshape(-1)
                np.add.at(force, dofs, local_force)
                physical = -local_force.reshape(-1,3)
                wrench += physical.sum(axis=0)
                moment += np.cross(positions, physical).sum(axis=0)
                if tangent:
                    block = self.endpoint_stiffness*(np.outer(state.gradient, state.gradient)
                                                     +state.gap*state.hessian)
                    rows.extend(np.repeat(dofs,len(dofs))); cols.extend(np.tile(dofs,len(dofs)))
                    values.extend(block.reshape(-1))
            fields.append({'node':node, 'element':cell.element_id, 'branch':state.branch,
                'gap':state.gap, 'active':state.gap < 0,
                'skin_point':None if state.skin_point is None else state.skin_point.tolist()})
        if tangent and values:
            matrix = matrix + sparse.coo_matrix((values,(rows,cols)), shape=matrix.shape).tocsr()
        wrench += np.asarray(result.beam_force); moment += np.asarray(result.beam_moment)
        result = replace(result, profile=PROFILE, endpoint_contact_energy=float(energy),
            active_endpoints=active, endpoint_candidates=len(self.cap_endpoints),
            endpoint_stiffness=self.endpoint_stiffness,
            beam_force=tuple(wrench),rigid_force=tuple(-wrench),
            beam_moment=tuple(moment),rigid_moment=tuple(-moment))
        return force, matrix, HemispherePlaneResults(result, tuple(fields))

    def restart_contract(self):
        result = super().restart_contract()
        result.update(beam_end_cap_work='hemisphere_support_dev',
            hemisphere_support_revision='actual_tangent_plane_sphere_rim_work_v1',
            full_envelope_revision='exact_quadratic_plane_height_v1',
            end_topology_revision='all_classical_material_node_incidence_v1',
            physical_endpoints=[{'node':node,'element':cell.element_id,'local':local}
                for node,(cell,local,sign) in sorted(self.cap_endpoints.items())])
        result['settings']['maximum_swept_subdivisions'] = self.settings.maximum_swept_subdivisions
        return result
