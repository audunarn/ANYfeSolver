"""Opt-in B3/B3 distributed round-centreline contact in nonlinear static analysis."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .beam_curve_pair import (
    B3PairPotential, B3PairPotentialState, _bezier_controls, certify_b3_pair_gap,
)
from .beam_pair import _BeamPairResults
from .core import ContactDefinition, ContactSolverSettings, ContactValidationReport


PROFILE = "CLASSICAL_B3_B3_ROUND_TUBE_DISTRIBUTED_PAIR_STATIC_DEV_V1"
MIXED_PROFILE = "CLASSICAL_B2_B3_ROUND_TUBE_DISTRIBUTED_PAIR_STATIC_DEV_V0"


def _virtual_b3(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Embed a straight B2 centreline in Q2 coordinates without adding DOFs."""
    if reference.shape == (3, 3):
        return reference, np.eye(9)
    if reference.shape != (2, 3):
        raise ValueError("beam pair needs a B2 or B3 centreline")
    positions = np.vstack((reference[0], 0.5 * (reference[0] + reference[1]),
                           reference[1]))
    embedding = np.zeros((9, 6))
    for station, weights in enumerate(((1.0, 0.0), (0.5, 0.5), (0.0, 1.0))):
        for node, weight in enumerate(weights):
            embedding[3*station:3*station+3, 3*node:3*node+3] = weight*np.eye(3)
    return positions, embedding


def validate_b3_pair_contact(model, definition: ContactDefinition,
                             settings: ContactSolverSettings | None = None
                             ) -> ContactValidationReport:
    issues = []
    if tuple(map(len, (definition.surfaces, definition.rigid_bodies,
                       definition.interactions))) != (2, 0, 1):
        return ContactValidationReport("invalid", (
            "B3 pair needs two beam surfaces, no rigid body and one interaction",
        ), PROFILE)
    controls = ContactSolverSettings() if settings is None else settings
    if controls.pair_motion != "general":
        issues.append("B3 pair needs pair_motion='general'")
    if controls.line_integration_relative_tolerance is None:
        issues.append("B3 pair needs explicit line_integration_relative_tolerance")
    for name in ("beam_pair_gap_accuracy", "maximum_beam_pair_gap_cells",
                 "maximum_beam_pair_gap_depth"):
        if getattr(controls, name) is None:
            issues.append(f"B3 pair needs explicit {name}")
    first, second = definition.surfaces
    interaction = definition.interactions[0]
    if (interaction.surface, interaction.target) != (first.name, second.name):
        issues.append("B3 pair interaction must name the surfaces in order")
    if (interaction.normal_law != "penalty"
            or interaction.normal_stiffness is not None
            or interaction.line_stiffness is None
            or interaction.endpoint_stiffness is not None):
        issues.append("B3 pair needs penalty line stiffness only")
    if (first.beam_radius is not None and second.beam_radius is not None
            and interaction.gap_tolerance >= first.beam_radius + second.beam_radius):
        issues.append("B3 pair gap tolerance must be smaller than combined radius")
    from anyfesolver.elements import BeamElement, QuadraticBeamElement
    from anymaterial import IsotropicMaterial

    selected_nodes = []
    element_types = []
    for surface in definition.surfaces:
        if (surface.beam_radius is None or surface.side != "midsurface"
                or surface.beam_end_caps != "none"
                or len(surface.element_ids) != 1):
            issues.append(f"surface {surface.name} needs one round B3 cell without end caps")
            selected_nodes.append(set())
            continue
        element_id = surface.element_ids[0]
        element = model.mesh.elements.get(element_id)
        if type(element) not in (BeamElement, QuadraticBeamElement):
            issues.append(f"element {element_id} is not an admitted classical B2/B3")
            selected_nodes.append(set())
            continue
        element_types.append(type(element))
        selected_nodes.append(set(map(int, element.node_ids)))
        if (element.generalized_section is not None
                or element._fiber_plasticity is not None
                or getattr(element, "_geometric_nonlinearity", None) != "von_karman"
                or np.any(getattr(element, "eccentricity", np.zeros(3)))):
            issues.append(f"element {element_id} needs a concentric elastic scalar von Karman section")
        material = model.materials.get(element.material_name)
        if (type(material) is not IsotropicMaterial
                or material.yield_stress != 0.0
                or material.hardening_curve is not None):
            issues.append(f"element {element_id} needs isotropic elastic material")
        try:
            reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
            virtual_reference, _ = _virtual_b3(reference)
            B3PairPotential(
                virtual_reference, virtual_reference + np.array((0.0, 1.0, 0.0)),
                surface.beam_radius, 1.0,
                integration_relative_tolerance=1.0,
                maximum_subdivisions=1,
            )
        except ValueError as exc:
            issues.append(f"element {element_id}: {exc}")
    if selected_nodes[0] & selected_nodes[1]:
        issues.append("B3 pair does not admit shared material nodes")
    if controls.maximum_candidate_pairs < 1:
        issues.append("B3 pair candidate budget is exhausted")
    profile = (MIXED_PROFILE if BeamElement in element_types else PROFILE)
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), profile)


class B3PairManager:
    def __init__(self, model, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 initial_displacement: np.ndarray | None = None):
        self.settings = ContactSolverSettings() if settings is None else settings
        report = validate_b3_pair_contact(model, definition, self.settings)
        if not report.valid:
            raise ValueError("unsupported B3 pair contact: " + "; ".join(report.issues))
        self.model = model
        self.definition = definition
        self.profile = report.profile
        self.stiffness = definition.interactions[0].line_stiffness
        self.gap_tolerance = definition.interactions[0].gap_tolerance
        self.radius = sum(surface.beam_radius for surface in definition.surfaces)
        self.end_caps = "none"
        self.swept_trial_cutbacks = 0
        self.reference = []
        self.dofs = []
        self.embeddings = []
        for surface in definition.surfaces:
            element = model.mesh.elements[surface.element_ids[0]]
            reference, embedding = _virtual_b3(np.asarray(
                element.get_node_coordinates(model.mesh), dtype=float,
            ))
            self.reference.append(reference)
            self.embeddings.append(embedding)
            self.dofs.append(np.asarray([
                model.mesh.dof_manager.get_node_dofs(node)[:3]
                for node in element.node_ids
            ], dtype=np.intp))
        self.assembly_map = sparse.block_diag(self.embeddings).toarray()
        self.n_dof = model.mesh.dof_manager.total_dofs
        self.law = B3PairPotential(
            self.reference[0], self.reference[1], self.radius, self.stiffness,
            integration_relative_tolerance=self.settings.line_integration_relative_tolerance,
            maximum_subdivisions=self.settings.maximum_line_subdivisions,
        )
        initial = (np.zeros(self.n_dof) if initial_displacement is None
                   else np.asarray(initial_displacement, dtype=float).reshape(-1))
        gap = self._certificate(initial)
        roundoff = 256.0 * np.finfo(float).eps * max(
            self.radius, self.law.length_a, self.law.length_b,
        )
        if gap.conservative_lower_gap < -roundoff:
            raise ValueError(
                f"B3 pair initial penetration is unresolved: "
                f"lower bound {gap.conservative_lower_gap:.6g} m"
            )

    def _positions(self, displacement):
        u = np.asarray(displacement, dtype=float).reshape(-1)
        if u.size != self.n_dof or not np.all(np.isfinite(u)):
            raise ValueError("B3 pair displacement has invalid size or values")
        return tuple(reference + (embedding @ u[dofs.reshape(-1)]).reshape(3, 3)
                     for reference, embedding, dofs in zip(
                         self.reference, self.embeddings, self.dofs))

    def _certificate(self, displacement):
        a, b = self._positions(displacement)
        return certify_b3_pair_gap(
            a, b, self.radius,
            accuracy=self.settings.beam_pair_gap_accuracy,
            maximum_cells=self.settings.maximum_beam_pair_gap_cells,
            maximum_depth=self.settings.maximum_beam_pair_gap_depth,
        )

    def evaluate(self, displacement, *, tangent=True):
        a, b = self._positions(displacement)
        certificate = self._certificate(displacement)
        self.last_gap_certificate = certificate
        state = (B3PairPotentialState(
            0.0, np.zeros(18), np.zeros((18, 18)), 0, 0.0, 0, 0,
        ) if certificate.conservative_lower_gap >= 0.0
            else self.law.evaluate(a, b))
        self.last_projection_evaluations = state.projection_evaluations
        dofs = np.r_[self.dofs[0].reshape(-1), self.dofs[1].reshape(-1)]
        local_residual = self.assembly_map.T @ state.residual
        force = np.zeros(self.n_dof)
        np.add.at(force, dofs, local_residual)
        stiffness = None
        if tangent:
            rows, cols = np.meshgrid(dofs, dofs, indexing="ij")
            local_tangent = self.assembly_map.T @ state.tangent @ self.assembly_map
            stiffness = sparse.coo_matrix(
                (local_tangent.reshape(-1),
                 (rows.reshape(-1), cols.reshape(-1))),
                shape=(self.n_dof, self.n_dof),
            ).tocsr()
        physical_a = -state.residual[:9].reshape(3, 3)
        physical_b = -state.residual[9:].reshape(3, 3)
        force_a, force_b = np.sum(physical_a, axis=0), np.sum(physical_b, axis=0)
        moment_a = np.sum(np.cross(a, physical_a), axis=0)
        moment_b = np.sum(np.cross(b, physical_b), axis=0)
        results = _BeamPairResults(
            self.profile, certificate.conservative_lower_gap,
            state.energy, state.integrated_intervals,
            state.integration_points, state.maximum_relative_integration_error,
            tuple(map(float, force_a)), tuple(map(float, force_b)),
            tuple(map(float, moment_a)), tuple(map(float, moment_b)),
            certificate.conservative_lower_gap >= -self.gap_tolerance,
            self.stiffness,
        )
        return force, stiffness, results

    def _swept_admissible(self, start, end):
        first = self._positions(start)
        last = self._positions(end)
        velocity_a = _bezier_controls(last[0] - first[0], 0.0, 1.0)
        velocity_b = _bezier_controls(last[1] - first[1], 0.0, 1.0)
        relative_speed = max(float(np.linalg.norm(a - b))
                             for a in velocity_a for b in velocity_b)

        def lower_at(fraction):
            pose = tuple(a + fraction * (b - a)
                         for a, b in zip(first, last))
            return certify_b3_pair_gap(
                pose[0], pose[1], self.radius,
                accuracy=self.settings.beam_pair_gap_accuracy,
                maximum_cells=self.settings.maximum_beam_pair_gap_cells,
                maximum_depth=self.settings.maximum_beam_pair_gap_depth,
            ).conservative_lower_gap

        first_gap, last_gap = lower_at(0.0), lower_at(1.0)
        if first_gap < -self.gap_tolerance:
            raise ValueError("committed B3 pair gap exceeds tolerance")
        if last_gap < -self.gap_tolerance:
            return False

        def certify(left, right, gap_left, gap_right, depth):
            if (min(gap_left, gap_right)
                    - 0.5 * relative_speed * (right - left) >= -self.gap_tolerance):
                return True
            if depth >= self.settings.maximum_swept_subdivisions:
                return False
            middle = 0.5 * (left + right)
            gap_middle = lower_at(middle)
            if gap_middle < -self.gap_tolerance:
                return False
            return (certify(left, middle, gap_left, gap_middle, depth + 1)
                    and certify(middle, right, gap_middle, gap_right, depth + 1))

        return certify(0.0, 1.0, first_gap, last_gap, 0)

    def swept_increment_admissible(self, start, end,
                                   _start_factor, _end_factor):
        admissible = self._swept_admissible(start, end)
        if not admissible:
            self.swept_trial_cutbacks += 1
        return admissible

    def admissible_step_fraction(self, start, end):
        first = np.asarray(start, dtype=float).reshape(-1)
        last = np.asarray(end, dtype=float).reshape(-1)
        if self._swept_admissible(first, last):
            return 1.0
        self.swept_trial_cutbacks += 1
        fraction = 0.5
        for _ in range(self.settings.maximum_swept_subdivisions):
            if self._swept_admissible(first, first + fraction * (last - first)):
                return fraction
            fraction *= 0.5
        return 0.0

    def restart_contract(self):
        interaction = self.definition.interactions[0]
        return {
            "profile": self.profile,
            "surfaces": [
                {"name": surface.name, "element_ids": list(surface.element_ids),
                 "beam_radius": surface.beam_radius,
                 "beam_end_caps": surface.beam_end_caps}
                for surface in self.definition.surfaces
            ],
            "interaction": {
                "name": interaction.name, "surface": interaction.surface,
                "target": interaction.target,
                "line_stiffness": interaction.line_stiffness,
                "gap_tolerance": interaction.gap_tolerance,
                "normal_law": interaction.normal_law,
            },
            "settings": {
                "pair_motion": self.settings.pair_motion,
                "maximum_candidate_pairs": self.settings.maximum_candidate_pairs,
                "line_integration_relative_tolerance": self.settings.line_integration_relative_tolerance,
                "maximum_line_subdivisions": self.settings.maximum_line_subdivisions,
                "maximum_swept_subdivisions": self.settings.maximum_swept_subdivisions,
                "beam_pair_gap_accuracy": self.settings.beam_pair_gap_accuracy,
                "maximum_beam_pair_gap_cells": self.settings.maximum_beam_pair_gap_cells,
                "maximum_beam_pair_gap_depth": self.settings.maximum_beam_pair_gap_depth,
            },
        }
