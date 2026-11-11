"""Development composition of explicit qualified-Q4/fixed-plane penalties.

Every pair retains the existing reference-area nodal-pressure work, physical
skin gap gate, law and dimensional tolerance. No pair discovery, pressure-row
merging, new differentiation, moving body or aggregate penalty law is added.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .core import (
    ContactDefinition, ContactInteraction, ContactSolverSettings, ContactSurface,
    ContactValidationReport, RigidContactBody,
)

PROFILE = 'MULTIPLE_EXPLICIT_Q4_FIXED_PLANES_STATIC_DEV_V1'
_COMPOSITION_REVISION = 'explicit_q4_plane_work_sum_v1'


def _parts(model, definition, settings, analysis_control):
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    from .core import validate_general_contact

    if type(definition) is not ContactDefinition or type(settings) is not ContactSolverSettings:
        raise ValueError('exact ContactDefinition and ContactSolverSettings required')
    if analysis_control not in {'force', 'displacement'}:
        raise ValueError('multiple shell planes require force or displacement control')
    if (settings.pair_motion != 'guided' or settings.warped_q4_plane
            or settings.normal_discretization != 'primal_mortar'
            or settings.enforcement_solver != 'nested_augmented'):
        raise ValueError('multiple shell planes require guided planar primal-mortar penalties')
    if len(definition.interactions) < 2 or not definition.rigid_bodies:
        raise ValueError('multiple shell planes require at least two explicit interactions')
    if (any(type(value) is not ContactSurface for value in definition.surfaces)
            or any(type(value) is not RigidContactBody for value in definition.rigid_bodies)
            or any(type(value) is not ContactInteraction for value in definition.interactions)):
        raise ValueError('exact shell-plane definition components required')
    surfaces = {surface.name: surface for surface in definition.surfaces}
    bodies = {body.name: body for body in definition.rigid_bodies}
    used_surfaces, used_bodies, owned, parts = set(), set(), set(), []
    candidates = pressure_modes = 0
    for interaction in sorted(definition.interactions, key=lambda item: item.name):
        if interaction.surface not in surfaces or interaction.target not in bodies:
            raise ValueError('each shell-plane interaction must name a declared shell and rigid plane')
        source, body = surfaces[interaction.surface], bodies[interaction.target]
        if source.beam_radius is not None or source.beam_end_caps != 'none':
            raise ValueError('multiple shell planes do not admit beam features')
        if body.kind != 'plane' or body.path_translation is not None:
            raise ValueError('multiple shell planes require fixed planes')
        if (interaction.normal_law != 'penalty' or interaction.normal_stiffness is None
                or interaction.line_stiffness is not None or interaction.endpoint_stiffness is not None):
            raise ValueError('multiple shell planes require dimensioned surface penalty laws')
        nodes = set()
        for element_id in source.element_ids:
            element = model.mesh.elements.get(element_id)
            if type(element) is not QualifiedE4PLShellElement:
                raise ValueError('multiple shell planes require exact qualified Q4 surface elements')
            nodes.update(int(node) for node in element.node_ids)
            feature = (element_id, source.side, body.name)
            if feature in owned:
                raise ValueError('duplicate physical shell element/side/rigid-body interaction')
            owned.add(feature)
        candidates += len(source.element_ids)
        pressure_modes += len(nodes)
        if candidates > settings.maximum_candidate_pairs:
            raise MemoryError('multiple shell planes exceed aggregate candidate limit; no contacts dropped')
        if pressure_modes > settings.maximum_patch_normal_dofs:
            raise MemoryError('multiple shell planes exceed aggregate pressure-mode limit; no supports dropped')
        part = ContactDefinition(surfaces=(source,), rigid_bodies=(body,), interactions=(interaction,))
        report = validate_general_contact(model, part, settings, analysis_control=analysis_control)
        if not report.valid:
            raise ValueError(f'{interaction.name}: '+ '; '.join(report.issues))
        parts.append((interaction.name, part, tuple(sorted(nodes))))
        used_surfaces.add(source.name)
        used_bodies.add(body.name)
    if used_surfaces != set(surfaces) or used_bodies != set(bodies):
        raise ValueError('multiple shell-plane definition contains an unreferenced surface or body')
    return tuple(parts), candidates, pressure_modes


def validate_multiple_shell_planes(model, definition, settings=None, *, analysis_control='force'):
    controls = ContactSolverSettings() if settings is None else settings
    try:
        _parts(model, definition, controls, analysis_control)
    except (ValueError, MemoryError, KeyError) as exc:
        return ContactValidationReport('invalid', (str(exc),), PROFILE)
    return ContactValidationReport('ok', (), PROFILE)


@dataclass(frozen=True)
class MultipleShellPlaneResults:
    profile: str
    pairs: tuple

    @property
    def minimum_gap(self):
        return min(result.minimum_gap for _, result in self.pairs)

    @property
    def gap_tolerance_met(self):
        return all(result.gap_tolerance_met for _, result in self.pairs)

    @property
    def penalty_energy(self):
        return float(sum(result.penalty_energy for _, result in self.pairs))

    @property
    def active_pressure_dofs(self):
        return sum(result.active_pressure_dofs for _, result in self.pairs)

    def _wrench(self, name):
        return tuple(float(value) for value in np.sum(
            [getattr(result, name) for _, result in self.pairs], axis=0))

    shell_force = property(lambda self: self._wrench('shell_force'))
    rigid_force = property(lambda self: self._wrench('rigid_force'))
    shell_moment = property(lambda self: self._wrench('shell_moment'))
    rigid_moment = property(lambda self: self._wrench('rigid_moment'))

    def to_dict(self):
        return dict(profile=self.profile, minimum_gap=self.minimum_gap,
            gap_tolerance_met=self.gap_tolerance_met, penalty_energy=self.penalty_energy,
            active_pressure_dofs=self.active_pressure_dofs,
            shell_force=list(self.shell_force), rigid_force=list(self.rigid_force),
            shell_moment=list(self.shell_moment), rigid_moment=list(self.rigid_moment),
            work_convention='sum_of_explicit_pair_potentials_same_authoritative_coordinates',
            normal_law='penalty',
            pairs=[dict(interaction=name, contact=result.to_dict()) for name, result in self.pairs])


class MultipleShellPlaneManager:
    profile = PROFILE
    augmented = False

    def __init__(self, model, definition, settings=None, initial_displacement=None,
                 *, analysis_control='force'):
        from .core import _PlanarQ4PlaneManager

        self.settings = ContactSolverSettings() if settings is None else settings
        parts, self.candidate_count, self.pressure_mode_count = _parts(
            model, definition, self.settings, analysis_control)
        self.model, self.definition = model, definition
        self.n_dof = model.mesh.dof_manager.total_dofs
        pairs, pressure_features = [], []
        for name, part, nodes in parts:
            child = _PlanarQ4PlaneManager(model, part, self.settings, initial_displacement)
            row_nodes = {tuple(model.mesh.dof_manager.get_node_dofs(node)[:6]): node for node in nodes}
            if len(row_nodes) != len(nodes) or len(child.output_nodes) != child.support.size:
                raise ValueError('shell-plane pressure rows have ambiguous node identity')
            for _, dofs in child.output_nodes:
                key = tuple(int(dof) for dof in dofs)
                if key not in row_nodes:
                    raise ValueError('shell-plane pressure row has no declared node identity')
                pressure_features.append((name, row_nodes[key]))
            pairs.append((name, child))
        self.pairs = tuple(pairs)
        self.pressure_features = tuple(pressure_features)
        self.support = np.concatenate([child.support for _, child in self.pairs])
        self.reference_weighted_gap = np.concatenate(
            [child.reference_weighted_gap for _, child in self.pairs])
        self.gap_gradient = sparse.vstack([child.gap_gradient for _, child in self.pairs], format='csr')
        if (len(self.pressure_features) != self.pressure_mode_count
                or self.support.size != self.pressure_mode_count or np.any(self.support <= 0.)):
            raise ValueError('shell-plane composition must retain every positive pressure support')
        self.swept_trial_cutbacks = 0

    def _coordinates(self, displacement):
        value = np.asarray(displacement, dtype=float).reshape(-1)
        if value.size != self.n_dof or not np.all(np.isfinite(value)):
            raise ValueError('multiple shell-plane displacement has invalid size or nonfinite values')
        return value

    def evaluate(self, displacement, *, tangent=True):
        displacement = self._coordinates(displacement)
        force = np.zeros(self.n_dof)
        stiffness = sparse.csr_matrix((self.n_dof, self.n_dof)) if tangent else None
        results = []
        for name, child in self.pairs:
            residual, block, result = child.evaluate(displacement, tangent=tangent)
            force += residual
            if tangent:
                stiffness = stiffness+block
            results.append((name, result))
        return force, stiffness, MultipleShellPlaneResults(self.profile, tuple(results))

    def swept_increment_admissible(self, start, end, start_factor, end_factor):
        """All fixed-plane physical corner gates on the full affine increment."""
        first, last = self._coordinates(start), self._coordinates(end)
        checks = []
        for name, child in self.pairs:
            initial = child.evaluate(first, tangent=False)[2]
            terminal = child.evaluate(last, tangent=False)[2]
            if not initial.gap_tolerance_met:
                raise ValueError(f'{name}: committed fixed-plane pose exceeds its gap tolerance')
            checks.append(terminal.gap_tolerance_met)
        admitted = all(checks)
        if not admitted:
            self.swept_trial_cutbacks += 1
        return admitted

    def admissible_step_fraction(self, start, end):
        first, last = self._coordinates(start), self._coordinates(end)
        self.swept_increment_admissible(first, first, 0., 0.)
        fraction = min(child.admissible_step_fraction(first, last) for _, child in self.pairs)
        for _ in range(self.settings.maximum_swept_subdivisions+1):
            if self.swept_increment_admissible(first, first+fraction*(last-first), 0., 1.):
                return fraction
            fraction *= .5
        return 0.

    def restart_contract(self):
        return dict(profile=self.profile, composition_revision=_COMPOSITION_REVISION,
            pairs=[dict(interaction=name, contract=child.restart_contract()) for name, child in self.pairs],
            pressure_features=[[name, node] for name, node in self.pressure_features],
            candidate_count=self.candidate_count, pressure_mode_count=self.pressure_mode_count,
            resource_controls=dict(maximum_candidate_pairs=self.settings.maximum_candidate_pairs,
                maximum_patch_normal_dofs=self.settings.maximum_patch_normal_dofs,
                maximum_swept_subdivisions=self.settings.maximum_swept_subdivisions))

    def summary_settings(self):
        return dict(normal_law='penalty', pair_enforcement=[
            dict(interaction=name, surface_stiffness_N_per_m3=child.stiffness,
                 gap_tolerance_m=child.gap_tolerance) for name, child in self.pairs],
            candidate_count=self.candidate_count, pressure_mode_count=self.pressure_mode_count)
