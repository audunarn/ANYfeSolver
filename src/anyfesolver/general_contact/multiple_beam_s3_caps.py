"""Explicit composition of actual-cap beam/S3 and fixed-rigid work.

Each S3 interaction keeps its whole-beam interior-face domain. This is not a
piecewise target-patch union, rim kernel, or new normal law.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse

from .core import ContactDefinition, ContactValidationReport, _contact_settings_contract

PROFILE = 'MULTIPLE_EXPLICIT_B2_B3_HEMISPHERE_DEFORMABLE_S3_FIXED_RIGID_STATIC_DEV_V1'


def _parts(model, definition, settings, analysis_control):
    from .core import validate_general_contact
    if analysis_control not in {'force', 'displacement'}:
        raise ValueError('multiple beam/S3 support requires force or displacement control')
    if (settings.beam_end_cap_work != 'hemisphere_support_dev'
            or settings.beam_shell_geometry != 'jet2_deformable_s3_dev'
            or settings.enforcement_solver != 'nested_augmented'):
        raise ValueError('multiple beam/S3 support requires explicit actual-cap Jet2 line-penalty settings')
    if len(definition.interactions) < 2:
        raise ValueError('multiple beam/S3 support requires at least two explicit pairs')
    surfaces = {s.name: s for s in definition.surfaces}
    bodies = {b.name: b for b in definition.rigid_bodies}
    used_surfaces, used_bodies, owned, geometries, parts = set(), set(), set(), {}, []
    has_shell = False
    for interaction in sorted(definition.interactions, key=lambda i: i.name):
        if len(parts) >= settings.maximum_candidate_pairs:
            raise MemoryError('multiple beam/S3 candidate limit exceeded; no contact dropped')
        source = surfaces.get(interaction.surface)
        if source is None or source.beam_radius is None or len(source.element_ids) != 1:
            raise ValueError('each source must select one actual classical B2/B3 cell')
        element = source.element_ids[0]
        geometry = (source.beam_radius, source.beam_end_caps)
        if element in geometries and geometries[element] != geometry:
            raise ValueError('shared physical beam has conflicting radius or cap geometry')
        geometries[element] = geometry
        used_surfaces.add(source.name)
        if interaction.target in surfaces:
            target = surfaces[interaction.target]
            if target.beam_radius is not None or len(target.element_ids) != 1:
                raise ValueError('deformable target must select one physical S3 skin')
            key = (element, 'shell', target.element_ids[0], target.side)
            part = ContactDefinition((source, target), (), (interaction,))
            child_settings = settings
            kind = 'shell'
            used_surfaces.add(target.name)
            has_shell = True
        elif interaction.target in bodies:
            target = bodies[interaction.target]
            if target.kind not in {'plane', 'sphere'} or target.path_translation is not None:
                raise ValueError('rigid child requires a fixed plane or sphere')
            key = (element, 'rigid', target.name)
            part = ContactDefinition((source,), (target,), (interaction,))
            # The S3 work-map selector applies to deformable children only.
            # The distinct rigid setting is explicit in the restart contract.
            child_settings = replace(settings, beam_shell_geometry='translated_planar')
            kind = 'rigid'
            used_bodies.add(target.name)
        else:
            raise ValueError('interaction target is not a declared skin or rigid body')
        if key in owned:
            raise ValueError('duplicate physical beam/target feature pair')
        owned.add(key)
        report = validate_general_contact(model, part, child_settings,
                                          analysis_control=analysis_control)
        if not report.valid:
            raise ValueError(interaction.name + ': ' + '; '.join(report.issues))
        parts.append((interaction.name, part, child_settings, kind))
    if not has_shell:
        raise ValueError('multiple beam/S3 composition requires a deformable target')
    if used_surfaces != set(surfaces) or used_bodies != set(bodies):
        raise ValueError('multiple beam/S3 definition contains an unreferenced participant')
    return tuple(parts)


def validate_multiple_beam_s3_caps(model, definition, settings, *, analysis_control='force'):
    try:
        _parts(model, definition, settings, analysis_control)
    except (ValueError, KeyError, MemoryError) as exc:
        return ContactValidationReport('invalid', (str(exc),), PROFILE)
    return ContactValidationReport('ok', (), PROFILE)


@dataclass(frozen=True)
class MultipleBeamS3CapResults:
    pairs: tuple
    profile: str = PROFILE

    @property
    def gap_tolerance_met(self):
        return all(result.gap_tolerance_met for _, _, _, result in self.pairs)

    @property
    def minimum_gap(self):
        return min(result.minimum_gap for _, _, _, result in self.pairs)

    def to_dict(self):
        pairs = []; beams = {}; shells = {}; bodies = {}
        for name, part, kind, result in self.pairs:
            field = result.to_dict()
            pairs.append({'interaction': name, 'contact': field})
            element = part.surfaces[0].element_ids[0]
            force, moment = beams.setdefault(element, (np.zeros(3), np.zeros(3)))
            force += np.asarray(field['beam_force']); moment += np.asarray(field['beam_moment'])
            if kind == 'shell':
                element = part.surfaces[1].element_ids[0]
                force, moment = shells.setdefault(element, (np.zeros(3), np.zeros(3)))
                force += np.asarray(field['shell_force']); moment += np.asarray(field['shell_moment'])
            else:
                name = part.rigid_bodies[0].name
                force, moment = bodies.setdefault(name, (np.zeros(3), np.zeros(3)))
                force += np.asarray(field['rigid_force']); moment += np.asarray(field['rigid_moment'])
        fields = [p['contact'] for p in pairs]
        return {'profile': self.profile, 'beam_end_cap_work': 'hemisphere_support_dev',
            'minimum_gap': self.minimum_gap, 'gap_tolerance_met': self.gap_tolerance_met,
            'algorithmic_contact_potential': sum(f['algorithmic_contact_potential'] for f in fields),
            'line_contact_energy': sum(f['line_contact_energy'] for f in fields),
            'endpoint_contact_energy': sum(f['endpoint_contact_energy'] for f in fields),
            'active_line_length': sum(f['active_line_length'] for f in fields),
            'integration_points': sum(f['integration_points'] for f in fields),
            'maximum_integration_relative_error': max(f.get('maximum_integration_relative_error',
                f.get('integration_relative_error', 0.)) for f in fields),
            'active_endpoints': sum(f['active_endpoints'] for f in fields),
            'candidate_endpoints': sum(f.get('candidate_endpoints', f.get('endpoint_candidates', 0))
                                      for f in fields),
            'candidate_pairs': len(pairs), 'pairs': pairs,
            'work_convention': 'sum_of_explicit_pair_work_same_authoritative_coordinates',
            'moment_origin': [0., 0., 0.], 'moment_units': 'N*m',
            'beam_features': [{'element': e, 'force': w[0].tolist(), 'moment': w[1].tolist()}
                              for e, w in sorted(beams.items())],
            'shell_features': [{'element': e, 'force': w[0].tolist(), 'moment': w[1].tolist()}
                               for e, w in sorted(shells.items())],
            'rigid_bodies': [{'name': n, 'rigid_force': w[0].tolist(), 'rigid_moment': w[1].tolist()}
                            for n, w in sorted(bodies.items())]}


class MultipleBeamS3CapManager:
    profile = PROFILE

    def __init__(self, model, definition, settings, initial_displacement=None, *, analysis_control='force'):
        from .deformable_beam_s3_caps import DeformableBeamS3CapManager
        from .beam_hemisphere_plane import HemispherePlaneManager
        from .beam_hemisphere_sphere import HemisphereSphereManager
        self.definition, self.settings = definition, settings
        self.n_dof = model.mesh.dof_manager.total_dofs
        self.swept_trial_cutbacks = 0
        children = []
        for name, part, child_settings, kind in _parts(model, definition, settings, analysis_control):
            owner = (DeformableBeamS3CapManager if kind == 'shell' else
                     HemispherePlaneManager if part.rigid_bodies[0].kind == 'plane' else HemisphereSphereManager)
            children.append((name, part, kind, owner(model, part, child_settings, initial_displacement)))
        self.pairs = tuple(children)

    def evaluate(self, displacement, *, tangent=True):
        residual = np.zeros(self.n_dof)
        stiffness = sparse.csr_matrix((self.n_dof, self.n_dof)) if tangent else None
        results = []
        for name, part, kind, manager in self.pairs:
            force, block, result = manager.evaluate(displacement, tangent=tangent)
            residual += force
            if tangent:
                stiffness = stiffness + block
            results.append((name, part, kind, result))
        return residual, stiffness, MultipleBeamS3CapResults(tuple(results))

    def swept_increment_admissible(self, start, end, start_factor, end_factor):
        admitted = [m.swept_increment_admissible(start, end, start_factor, end_factor)
                    for _, _, _, m in self.pairs]
        self.swept_trial_cutbacks += int(not all(admitted))
        return all(admitted)

    def admissible_step_fraction(self, start, end):
        first, last = np.asarray(start), np.asarray(end)
        fraction = min(m.admissible_step_fraction(first, last) for _, _, _, m in self.pairs)
        for _ in range(self.settings.maximum_swept_subdivisions + 1):
            candidate = first + fraction*(last-first)
            if self.swept_increment_admissible(first, candidate, 0., 0.):
                return fraction
            fraction *= .5
        return 0.

    def restart_contract(self):
        return {'profile': self.profile, 'composition_revision': 'explicit_complete_s3_fixed_rigid_cap_work_v1',
            'settings': _contact_settings_contract(self.settings),
            'pairs': [{'interaction': name, 'kind': kind, 'effective_settings': _contact_settings_contract(m.settings),
                       'contract': m.restart_contract()}
                      for name, _, kind, m in self.pairs]}

    def summary_settings(self):
        return {'pair_enforcement': [{'interaction': name,
            'line_stiffness_N_per_m2': part.interactions[0].line_stiffness,
            'endpoint_stiffness_N_per_m': part.interactions[0].endpoint_stiffness,
            'gap_tolerance_m': part.interactions[0].gap_tolerance}
            for name, part, _, _ in self.pairs],
            'maximum_candidate_pairs': self.settings.maximum_candidate_pairs}
