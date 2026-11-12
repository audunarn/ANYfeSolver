"""Declared round classical beam against a fixed analytic plane.

The line law uses reference centreline length and the actual B2/B3 deformed
nodal curve. A separate endpoint spring represents each owned hemispherical
cap when the plane approaches its outward half-space. This is a narrow static
development profile; its two regularizations have different dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import sparse

from .core import (
    ContactDefinition, ContactSolverSettings, ContactValidationReport,
)


PROFILE = "CLASSICAL_B2_B3_ROUND_TUBE_FIXED_PLANE_LINE_PENALTY_STATIC_DEV_V0"
_GAUSS_XI, _GAUSS_WEIGHT = np.polynomial.legendre.leggauss(3)


def _beam_types():
    from anyfesolver.elements import BeamElement, QuadraticBeamElement

    return BeamElement, QuadraticBeamElement


def validate_beam_plane_contact(model, definition: ContactDefinition,
                                settings: ContactSolverSettings | None = None
                                ) -> ContactValidationReport:
    issues: list[str] = []
    if tuple(map(len, (definition.surfaces, definition.rigid_bodies,
                       definition.interactions))) != (1, 1, 1):
        issues.append("beam-plane profile requires one beam surface, one plane and one interaction")
        return ContactValidationReport("invalid", tuple(issues), PROFILE)
    surface = definition.surfaces[0]
    body = definition.rigid_bodies[0]
    interaction = definition.interactions[0]
    controls = ContactSolverSettings() if settings is None else settings
    if controls.pair_motion != "general":
        issues.append("round beam-plane contact requires pair_motion='general'")
    if surface.beam_radius is None or surface.side != "midsurface":
        issues.append("beam contact requires explicit round radius and no shell side")
    if body.kind != "plane" or body.path_translation is not None:
        issues.append("beam-plane profile requires a fixed analytic plane")
    if interaction.surface != surface.name or interaction.target != body.name:
        issues.append("beam-plane interaction names do not match its participants")
    if interaction.normal_law != "penalty":
        issues.append("beam-plane profile currently admits penalty enforcement only")
    if interaction.normal_stiffness is not None or interaction.line_stiffness is None:
        issues.append("beam-plane contact requires line_stiffness in N/m² and no surface normal_stiffness")
    if (surface.beam_end_caps == "hemisphere") != (interaction.endpoint_stiffness is not None):
        issues.append("hemispherical caps require separate endpoint_stiffness in N/m")
    if len(surface.element_ids) > controls.maximum_candidate_pairs:
        issues.append("beam-plane candidate limit exceeded; no element is discarded")
    b2, b3 = _beam_types()
    from anymaterial import IsotropicMaterial

    for element_id in surface.element_ids:
        element = model.mesh.elements.get(element_id)
        if type(element) not in {b2, b3}:
            issues.append(f"element {element_id} is not an admitted classical B2/B3 beam")
            continue
        if (element.generalized_section is not None
                or element._fiber_plasticity is not None
                or getattr(element, "_geometric_nonlinearity", None) != "von_karman"
                or np.any(getattr(element, "eccentricity", np.zeros(3)))):
            issues.append(f"element {element_id} needs a concentric elastic scalar von Karman section")
        material = model.materials.get(element.material_name)
        if (type(material) is not IsotropicMaterial
                or material.yield_stress != 0.0
                or material.hardening_curve is not None):
            issues.append(f"element {element_id} needs an isotropic elastic material")
        try:
            reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
            if not np.all(np.isfinite(reference)):
                raise ValueError("nonfinite reference coordinates")
            length = float(np.linalg.norm(reference[-1] - reference[0]))
            if length <= 0.0:
                raise ValueError("zero reference length")
        except (ValueError, KeyError) as exc:
            issues.append(f"element {element_id}: {exc}")
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), PROFILE)


@dataclass(frozen=True)
class _BeamCell:
    element_id: int
    node_ids: tuple[int, ...]
    reference: np.ndarray
    dofs: np.ndarray
    length: float

    def shape(self, xi: float) -> np.ndarray:
        if len(self.node_ids) == 2:
            return np.array([(1.0 - xi) * 0.5, (1.0 + xi) * 0.5])
        return np.array([0.5 * xi * (xi - 1.0), 1.0 - xi * xi,
                         0.5 * xi * (xi + 1.0)])


@dataclass(frozen=True)
class _BeamPlaneResults:
    profile: str
    minimum_gap: float
    line_contact_energy: float
    endpoint_contact_energy: float
    active_line_length: float
    active_line_cells: int
    active_endpoints: int
    candidate_cells: int
    active_intervals: int
    integration_points: int
    endpoint_candidates: int
    beam_force: tuple[float, float, float]
    rigid_force: tuple[float, float, float]
    beam_moment: tuple[float, float, float]
    rigid_moment: tuple[float, float, float]
    gap_tolerance_met: bool
    line_stiffness: float
    endpoint_stiffness: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "minimum_gap": self.minimum_gap,
            "line_contact_energy": self.line_contact_energy,
            "endpoint_contact_energy": self.endpoint_contact_energy,
            "algorithmic_contact_potential": (
                self.line_contact_energy + self.endpoint_contact_energy
            ),
            "active_line_length": self.active_line_length,
            "active_line_cells": self.active_line_cells,
            "active_endpoints": self.active_endpoints,
            "candidate_cells": self.candidate_cells,
            "active_intervals": self.active_intervals,
            "integration_points": self.integration_points,
            "endpoint_candidates": self.endpoint_candidates,
            "beam_force": list(self.beam_force),
            "rigid_force": list(self.rigid_force),
            "beam_moment": list(self.beam_moment),
            "rigid_moment": list(self.rigid_moment),
            "gap_tolerance_met": self.gap_tolerance_met,
            "normal_law": "penalty",
            "line_stiffness_N_per_m2": self.line_stiffness,
            "endpoint_stiffness_N_per_m": self.endpoint_stiffness,
        }


def _gap_coefficients(heights: np.ndarray) -> tuple[float, float, float]:
    if len(heights) == 2:
        return 0.0, float(0.5 * (heights[1] - heights[0])), float(
            0.5 * (heights[1] + heights[0])
        )
    return (float(0.5 * (heights[0] + heights[2]) - heights[1]),
            float(0.5 * (heights[2] - heights[0])), float(heights[1]))


def _minimum_polynomial(coefficients: tuple[float, float, float]) -> float:
    a, b, c = coefficients
    values = [a - b + c, a + b + c]
    if a > 0.0:
        stationary = -b / (2.0 * a)
        if -1.0 < stationary < 1.0:
            values.append((a * stationary + b) * stationary + c)
    return float(min(values))


def _active_intervals(coefficients: tuple[float, float, float]) -> tuple[tuple[float, float], ...]:
    a, b, c = coefficients
    scale = max(abs(a), abs(b), abs(c))
    if scale == 0.0:
        return ((-1.0, 1.0),)
    normalized = np.asarray([a, b, c], dtype=float) / scale
    roots = np.roots(normalized if abs(normalized[0]) > 64.0 * np.finfo(float).eps
                     else normalized[1:])
    cuts = [-1.0]
    for root in roots:
        if abs(float(np.imag(root))) <= 1.0e-12:
            value = float(np.real(root))
            if -1.0 < value < 1.0:
                cuts.append(value)
    cuts.append(1.0)
    cuts = sorted(set(cuts))
    intervals = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        middle = 0.5 * (left + right)
        if ((a * middle + b) * middle + c) <= 0.0:
            intervals.append((left, right))
    return tuple(intervals)


class _BeamPlaneManager:
    def __init__(self, model, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 initial_displacement: np.ndarray | None = None):
        self.settings = ContactSolverSettings() if settings is None else settings
        report = validate_beam_plane_contact(model, definition, self.settings)
        if not report.valid:
            raise ValueError("unsupported beam-plane contact: " + "; ".join(report.issues))
        self.model = model
        self.definition = definition
        self.profile = report.profile
        surface = definition.surfaces[0]
        body = definition.rigid_bodies[0]
        interaction = definition.interactions[0]
        self.radius = float(surface.beam_radius)
        self.end_caps = surface.beam_end_caps
        self.stiffness = interaction.line_stiffness
        self.endpoint_stiffness = interaction.endpoint_stiffness
        self.gap_tolerance = interaction.gap_tolerance
        self.swept_trial_cutbacks = 0
        self.point = np.asarray(body.point, dtype=float)
        self.normal = np.asarray(body.normal, dtype=float)
        self.cells = []
        for element_id in surface.element_ids:
            element = model.mesh.elements[element_id]
            reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
            dofs = np.asarray([model.mesh.dof_manager.get_node_dofs(node)[:3]
                               for node in element.node_ids], dtype=np.intp)
            self.cells.append(_BeamCell(
                element_id, tuple(int(node) for node in element.node_ids),
                reference, dofs, float(np.linalg.norm(reference[-1] - reference[0])),
            ))
        self.endpoints: dict[int, tuple[_BeamCell, int, float]] = {}
        if self.end_caps == "hemisphere":
            from .beam_end_topology import physical_beam_ends
            self.endpoints = physical_beam_ends(model,self.cells,_beam_types())
        n_dof = model.mesh.dof_manager.total_dofs
        initial = np.zeros(n_dof) if initial_displacement is None else np.asarray(
            initial_displacement, dtype=float,
        ).reshape(-1)
        if initial.size != n_dof or not np.all(np.isfinite(initial)):
            raise ValueError("beam contact initial displacement is invalid")
        scale = max(self.radius, *(cell.length for cell in self.cells))
        if self._minimum_gap(initial) < -256.0 * np.finfo(float).eps * scale:
            raise ValueError("beam-plane contact has unresolved initial penetration")

    def _heights(self, cell: _BeamCell, u: np.ndarray) -> np.ndarray:
        positions = cell.reference + u[cell.dofs]
        return (positions - self.point) @ self.normal - self.radius

    def _minimum_gap(self, u: np.ndarray) -> float:
        return min(_minimum_polynomial(_gap_coefficients(self._heights(cell, u)))
                   for cell in self.cells)

    def _end_plane_coordinates(self,cell,local,outward,u):
        """Exact represented end gap and actual outward derivative projection."""
        positions=cell.reference+u[cell.dofs]
        heights=[sum((Fraction.from_float(float(x))-Fraction.from_float(float(p)))
            *Fraction.from_float(float(n)) for x,p,n in zip(position,self.point,self.normal))
            for position in positions]
        gap=heights[local]-Fraction.from_float(self.radius)
        if len(heights)==2:derivative=heights[1]-heights[0]
        elif local==0:derivative=(-3*heights[0]+4*heights[1]-heights[2])/2
        else:derivative=(heights[0]-4*heights[1]+3*heights[2])/2
        return gap,int(outward)*derivative

    def _end_ownership_admissible(self,start,end):
        for cell,local,outward in self.endpoints.values():
            g0,q0=self._end_plane_coordinates(cell,local,outward,start)
            g1,q1=self._end_plane_coordinates(cell,local,outward,end)
            if min(g0,g1)>=0:continue  # The entire affine end path is open.
            if (q0<0 and q1<0) or (q0>0 and q1>0):continue
            if q0==q1:return False  # Closed, tangent ownership throughout.
            root=-q0/(q1-q0)
            if 0<=root<=1 and g0+root*(g1-g0)<0:return False
        return True

    def evaluate(self, displacement: np.ndarray, *, tangent: bool = True):
        n_dof = self.model.mesh.dof_manager.total_dofs
        u = np.asarray(displacement, dtype=float).reshape(-1)
        if u.size != n_dof or not np.all(np.isfinite(u)):
            raise ValueError("beam-plane displacement is invalid")
        force = np.zeros(n_dof)
        entries: dict[tuple[int, int], float] = {}
        line_energy = endpoint_energy = active_length = 0.0
        active_cells = active_endpoints = 0
        active_intervals = integration_points = 0
        for cell in self.cells:
            coefficients = _gap_coefficients(self._heights(cell, u))
            intervals = _active_intervals(coefficients)
            if intervals:
                active_cells += 1
            active_intervals += len(intervals)
            for left, right in intervals:
                half = 0.5 * (right - left)
                mid = 0.5 * (right + left)
                active_length += half * cell.length
                for gauss, weight in zip(_GAUSS_XI, _GAUSS_WEIGHT):
                    integration_points += 1
                    xi = mid + half * gauss
                    gap = (coefficients[0] * xi + coefficients[1]) * xi + coefficients[2]
                    shape = cell.shape(float(xi))
                    measure = half * 0.5 * cell.length * float(weight)
                    line_energy += 0.5 * self.stiffness * gap * gap * measure
                    for i, node_dofs in enumerate(cell.dofs):
                        local_gradient = shape[i] * self.normal
                        force[node_dofs] += self.stiffness * gap * measure * local_gradient
                        if tangent:
                            for j, other_dofs in enumerate(cell.dofs):
                                block = (self.stiffness * measure * shape[i] * shape[j]
                                         * np.outer(self.normal, self.normal))
                                for row in range(3):
                                    for col in range(3):
                                        key = (int(node_dofs[row]), int(other_dofs[col]))
                                        entries[key] = entries.get(key, 0.0) + float(block[row, col])
        if self.end_caps == "hemisphere":
            for node, (cell, local, outward_sign) in self.endpoints.items():
                exact_gap,orientation=self._end_plane_coordinates(cell,local,outward_sign,u)
                if exact_gap>=0:continue
                if orientation==0:
                    raise ValueError('active beam-plane end has unresolved cap-to-side ownership')
                if orientation>0:continue
                gap=float(exact_gap)
                active_endpoints += 1
                endpoint_energy += 0.5 * self.endpoint_stiffness * gap * gap
                dofs = cell.dofs[local]
                force[dofs] += self.endpoint_stiffness * gap * self.normal
                if tangent:
                    block = self.endpoint_stiffness * np.outer(self.normal, self.normal)
                    for row in range(3):
                        for col in range(3):
                            key = (int(dofs[row]), int(dofs[col]))
                            entries[key] = entries.get(key, 0.0) + float(block[row, col])
        tangent_matrix = (sparse.coo_matrix(
            ([value for value in entries.values()],
             ([key[0] for key in entries], [key[1] for key in entries])),
            shape=(n_dof, n_dof),
        ).tocsr() if tangent else None)
        beam_force = np.zeros(3)
        beam_moment = np.zeros(3)
        unique_nodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for cell in self.cells:
            for node, reference, dofs in zip(cell.node_ids, cell.reference, cell.dofs):
                unique_nodes[node] = (reference, dofs)
        for reference, dofs in unique_nodes.values():
            physical = -force[dofs]
            beam_force += physical
            beam_moment += np.cross(reference + u[dofs], physical)
        minimum = self._minimum_gap(u)
        results = _BeamPlaneResults(
            self.profile, minimum, line_energy, endpoint_energy,
            active_length, active_cells, active_endpoints,
            len(self.cells), active_intervals, integration_points,
            len(self.endpoints),
            tuple(float(x) for x in beam_force),
            tuple(float(x) for x in -beam_force),
            tuple(float(x) for x in beam_moment),
            tuple(float(x) for x in -beam_moment),
            minimum >= -self.gap_tolerance,
            self.stiffness, self.endpoint_stiffness,
        )
        return force, tangent_matrix, results

    def admissible_step_fraction(self, start: np.ndarray, end: np.ndarray) -> float:
        start=np.asarray(start,dtype=float).reshape(-1)
        end=np.asarray(end,dtype=float).reshape(-1)
        first = self._minimum_gap(start)
        last = self._minimum_gap(end)
        if first < -self.gap_tolerance:
            raise ValueError("committed beam-plane pose exceeds the gap tolerance")
        if last >= -self.gap_tolerance and self._end_ownership_admissible(start,end):
            return 1.0
        self.swept_trial_cutbacks += 1
        denominator = first - last
        fraction=1.0
        if last < -self.gap_tolerance:
            # Preserve the historical no-cap arithmetic and operation order.
            fraction=(first+self.gap_tolerance)/denominator if denominator>0. else 0.0
            fraction=max(0.0,min(1.0,0.98*fraction))
        for _ in range(self.settings.maximum_swept_subdivisions):
            if self._end_ownership_admissible(start,start+fraction*(end-start)):
                return fraction
            fraction*=.5
        return 0.0

    def swept_increment_admissible(self, start: np.ndarray, end: np.ndarray,
                                   _start_factor: float,
                                   _end_factor: float) -> bool:
        # For a fixed plane, every material centreline gap is affine along a
        # linear structural increment. Endpoint certification bounds the sweep.
        start=np.asarray(start,dtype=float).reshape(-1)
        end=np.asarray(end,dtype=float).reshape(-1)
        first = self._minimum_gap(start)
        last = self._minimum_gap(end)
        if first < -self.gap_tolerance:
            raise ValueError("committed beam-plane pose exceeds the gap tolerance")
        if last < -self.gap_tolerance or not self._end_ownership_admissible(start,end):
            self.swept_trial_cutbacks += 1
            return False
        return True

    def restart_contract(self) -> dict[str, object]:
        surface = self.definition.surfaces[0]
        body = self.definition.rigid_bodies[0]
        interaction = self.definition.interactions[0]
        contract = {
            "profile": self.profile,
            "surface": {"name": surface.name, "element_ids": list(surface.element_ids),
                        "beam_radius": surface.beam_radius,
                        "beam_end_caps": surface.beam_end_caps},
            "body": {"name": body.name, "kind": body.kind,
                     "point": list(body.point), "normal": list(body.normal)},
            "interaction": {"name": interaction.name,
                            "normal_stiffness": interaction.normal_stiffness,
                            "line_stiffness": interaction.line_stiffness,
                            "endpoint_stiffness": interaction.endpoint_stiffness,
                            "gap_tolerance": interaction.gap_tolerance,
                            "normal_law": interaction.normal_law},
            "settings": {"pair_motion": self.settings.pair_motion,
                         "maximum_candidate_pairs": self.settings.maximum_candidate_pairs},
        }
        if self.end_caps == "hemisphere":
            contract["end_topology_revision"] = "all_classical_material_node_incidence_v1"
            contract["hemisphere_ownership_revision"] = "fixed_plane_exact_affine_ownership_v1"
            contract["settings"]["maximum_swept_subdivisions"] = self.settings.maximum_swept_subdivisions
            contract["physical_endpoints"] = [
                {"node": node, "element": cell.element_id, "local": local}
                for node, (cell, local, sign) in sorted(self.endpoints.items())
            ]
        return contract
