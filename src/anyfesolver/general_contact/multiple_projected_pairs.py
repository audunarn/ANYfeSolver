"""Compose explicit finite-pressure shell pairs without another solve driver.

Each interaction retains its own pressure support, gap gate and feature identity.
This is not automatic pairing, self-contact or augmented multiplier composition.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .core import ContactDefinition, ContactValidationReport, validate_general_contact

PROFILE = "MULTIPLE_EXPLICIT_PROJECTED_SHELL_PAIRS_STATIC_DEV_V0"
PATH_PROFILE = "MULTIPLE_EXPLICIT_PROJECTED_SHELL_PAIRS_PATH_STATIC_DEV_V0"


def _plain_result(value):
    """Keep child certificate numbers lossless and standard-JSON compatible."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain_result(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_result(item) for item in value]
    return value


def _parts(model, definition, settings):
    if definition.rigid_bodies or len(definition.interactions) < 2:
        raise ValueError("multiple projected pairs need at least two shell interactions and no rigid bodies")
    surfaces = {surface.name: surface for surface in definition.surfaces}
    used, owned, parts = set(), set(), []
    candidate_count = pressure_count = 0
    for interaction in sorted(definition.interactions, key=lambda item: item.name):
        if interaction.normal_law != "projected_penalty":
            raise ValueError("multiple projected pairs require the same finite normal-law family")
        if (interaction.surface not in surfaces or interaction.target not in surfaces
                or interaction.surface == interaction.target):
            raise ValueError("multiple projected pair must name distinct declared shell surfaces")
        source, target = surfaces[interaction.surface], surfaces[interaction.target]
        if source.beam_radius is not None or target.beam_radius is not None:
            raise ValueError("multiple projected pairs currently admit shells only")
        used.update((source.name, target.name))
        source_nodes = {node for eid in source.element_ids
                        for node in model.mesh.elements[eid].node_ids}
        target_nodes = {node for eid in target.element_ids
                        for node in model.mesh.elements[eid].node_ids}
        if source_nodes & target_nodes:
            raise ValueError("shared material features require a separately admitted self-contact owner")
        candidate_count += len(source.element_ids)*len(target.element_ids)
        pressure_count += len(source_nodes)
        if candidate_count > settings.maximum_candidate_pairs:
            raise MemoryError("multiple projected pairs exceed aggregate candidate limit")
        if pressure_count > settings.maximum_patch_normal_dofs:
            raise MemoryError("multiple projected pairs exceed aggregate pressure-mode limit")
        for first in source.element_ids:
            for second in target.element_ids:
                features = tuple(sorted(((first, source.side), (second, target.side))))
                if features in owned:
                    raise ValueError("duplicate physical feature pair in explicit interactions")
                owned.add(features)
        parts.append((interaction.name, ContactDefinition(
            surfaces=(source, target), rigid_bodies=(), interactions=(interaction,))))
    if used != set(surfaces):
        raise ValueError("multiple projected definition contains an unreferenced surface")
    return tuple(parts)


def validate_multiple_projected_pairs(model, definition, settings, *, analysis_control="force"):
    issues = []
    if settings is None:
        return ContactValidationReport("invalid", ("multiple pairs require explicit settings",), PROFILE)
    try:
        parts = _parts(model, definition, settings)
        for name, part in parts:
            report = validate_general_contact(model, part, settings, analysis_control=analysis_control)
            issues.extend(f"{name}: {issue}" for issue in report.issues)
    except (ValueError, MemoryError, KeyError) as exc:
        issues.append(str(exc))
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), PROFILE)


@dataclass(frozen=True)
class MultiplePairResults:
    profile: str
    pairs: tuple

    @property
    def minimum_gap(self):
        return min(result.minimum_gap for _, result in self.pairs)

    @property
    def gap_tolerance_met(self):
        return all(result.gap_tolerance_met for _, result in self.pairs)

    @property
    def active_pressure_dofs(self):
        return sum(result.active_pressure_dofs for _, result in self.pairs)

    def _wrench(self, name):
        return tuple(np.sum([getattr(result, name) for _, result in self.pairs], axis=0))

    integration_force = property(lambda self: self._wrench("integration_force"))
    target_force = property(lambda self: self._wrench("target_force"))
    integration_moment = property(lambda self: self._wrench("integration_moment"))
    target_moment = property(lambda self: self._wrench("target_moment"))

    @property
    def pair_geometry_evaluations(self):
        return sum(result.pair_geometry_evaluations for _, result in self.pairs)

    @property
    def regularization_storage_j(self):
        return sum(result.regularization_storage_j for _, result in self.pairs)

    def to_dict(self):
        return _plain_result({"profile": self.profile, "minimum_gap": self.minimum_gap,
                "gap_tolerance_met": self.gap_tolerance_met,
                "active_pressure_dofs": self.active_pressure_dofs,
                **{name: list(getattr(self, name)) for name in (
                    "integration_force", "target_force", "integration_moment", "target_moment")},
                "regularization_storage_j": self.regularization_storage_j,
                "pair_geometry_evaluations": self.pair_geometry_evaluations,
                "work_convention": "sum_of_explicit_pair_work_same_authoritative_coordinates",
                "pairs": [{"interaction": name, "contact": result.to_dict()}
                          for name, result in self.pairs]})


class MultipleProjectedPairManager:
    augmented = False
    projected_work = True
    profile = PROFILE

    def __init__(self, model, definition, settings, initial_displacement=None, *, analysis_control="force"):
        report = validate_multiple_projected_pairs(
            model, definition, settings, analysis_control=analysis_control)
        if not report.valid:
            raise ValueError("unsupported multiple projected pairs: " + "; ".join(report.issues))
        from .s3_patch_static import GeneralS3PatchManager
        self.definition, self.settings = definition, settings
        self.n_dof = model.mesh.dof_manager.total_dofs
        self.pairs = tuple((name, GeneralS3PatchManager(
            model, part, settings, initial_displacement))
            for name, part in _parts(model, definition, settings))
        self.swept_trial_cutbacks = 0

    def restrict_tangent_to_reduced_variations(self, transform):
        for _, manager in self.pairs:
            manager.restrict_tangent_to_reduced_variations(transform)

    def evaluate(self, displacement, *, tangent=True):
        residual = np.zeros(self.n_dof)
        stiffness = sparse.csr_matrix((self.n_dof, self.n_dof)) if tangent else None
        results = []
        for name, manager in self.pairs:
            force, block, result = manager.evaluate(displacement, tangent=tangent)
            residual += force
            if tangent:
                stiffness = stiffness + block
            results.append((name, result))
        return residual, stiffness, MultiplePairResults(self.profile, tuple(results))

    def swept_increment_admissible(self, start, end, start_factor, end_factor):
        # Evaluate every participating gate; never replace differing dimensional
        # budgets by a global minimum-gap/single-tolerance decision.
        admitted = [manager.swept_increment_admissible(start, end, start_factor, end_factor)
                    for _, manager in self.pairs]
        if not all(admitted):
            self.swept_trial_cutbacks += 1
        return all(admitted)

    def admissible_step_fraction(self, start, end):
        first, last = np.asarray(start), np.asarray(end)
        fraction = min(manager.admissible_step_fraction(first, last)
                       for _, manager in self.pairs)
        for _ in range(self.settings.maximum_swept_subdivisions + 1):
            if all(manager._swept_admissible(first, first+fraction*(last-first))
                   for _, manager in self.pairs):
                return fraction
            fraction *= .5
        return 0.0

    def restart_contract(self):
        return {"profile": self.profile, "composition_revision": "explicit_surface_work_sum_v1",
                "pairs": [{"interaction": name, "contract": manager.restart_contract()}
                          for name, manager in self.pairs],
                "aggregate_candidate_limit": self.settings.maximum_candidate_pairs,
                "aggregate_pressure_mode_limit": self.settings.maximum_patch_normal_dofs}

    def summary_settings(self):
        # Ordinary result publication must not invent a common stiffness or
        # tolerance when the interacting pairs have different settings.
        return {"normal_law": "projected_penalty", "pair_enforcement": [
            {"interaction": name, "normal_stiffness_pa_per_m": child.stiffness,
             "gap_tolerance_m": child.gap_tolerance}
            for name, child in self.pairs]}
