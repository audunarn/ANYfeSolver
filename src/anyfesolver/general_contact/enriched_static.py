"""Explicit mixed static owner for the separately versioned enriched profile.

Experimental reference-linear elastic structure, physical SO(3) director
offsets and current-frame bubble geometry; no finite-rotation structural
objectivity or D1 acceptance is claimed.  This separate API never changes an
existing profile, pressure space, checkpoint, or default solver route.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import Jet2, matvec, so3_exp
from anyfesolver.control import SolveCancelled
from anyfesolver.e4_pl_enriched import (
    FORMULATION, EnrichedProfileError, compile_cell, readonly,
)
from .enriched_overlap import QuadraturePolicy, moments


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


def check_cancel(cancel):
    if cancel():
        raise SolveCancelled("enriched static transaction cancelled")


@dataclass(frozen=True)
class PairDefinition:
    source_element: int
    target_element: int
    source_offset: float = 0.
    target_offset: float = 0.


@dataclass(frozen=True)
class MixedCriteria:
    """Explicit, source-bound criteria; callers cannot infer them from results."""
    force_scale_n: float
    equilibrium_relative: float
    complementarity_relative: float

    def validate(self):
        if not all(math.isfinite(x) and x > 0 for x in
                   (self.force_scale_n, self.equilibrium_relative, self.complementarity_relative)):
            raise EnrichedProfileError("positive finite mixed acceptance criteria required")


@dataclass(frozen=True)
class IterationControl:
    maximum_iterations: int
    maximum_halvings: int

    def validate(self):
        if (type(self.maximum_iterations) is not int or self.maximum_iterations < 1
                or type(self.maximum_halvings) is not int or self.maximum_halvings < 0):
            raise EnrichedProfileError("finite integer iteration/backtrack limits required")


@dataclass(frozen=True)
class CompiledPair:
    definition: PairDefinition
    indices: tuple
    reference_corners: np.ndarray
    reference_directors: np.ndarray
    reference_source: np.ndarray
    mode_indices: tuple

    def map(self, u):
        """Actual 46 ->37 map, including BOTH bodies' physical offsets.

        Director = exp(theta) n0.  All first/second offset derivatives are
        retained in the chain rule.  Four beta coordinates remain explicit.
        """
        local = np.asarray(u)[np.asarray(self.indices)]
        jets = [Jet2.variable(x, i, 46) for i, x in enumerate(local)]
        points = []
        for node in range(7):
            theta = jets[6*node+3:6*node+6]
            if np.linalg.norm(local[6*node+3:6*node+6]) >= math.pi:
                raise EnrichedProfileError("director rotation outside principal chart")
            offset = self.definition.source_offset if node < 4 else self.definition.target_offset
            arm = matvec(so3_exp(theta), [Jet2.constant(offset*x, 46)
                                        for x in self.reference_directors[node]])
            points.extend(self.reference_corners[node, k]+jets[6*node+k]+arm[k]
                          for k in range(3))
        points.extend(jets[42:46])
        points.extend(self.reference_corners[node, k]+jets[6*node+k]
                      for node in range(4) for k in range(3))
        return (np.array([x.value for x in points]),
                np.array([x.gradient for x in points]),
                np.array([x.hessian for x in points]))


@dataclass(frozen=True)
class ContactState:
    support: np.ndarray
    weighted_gap: np.ndarray
    normalized_gap: np.ndarray
    pressure: np.ndarray
    gap_gradient: np.ndarray
    force: np.ndarray
    tangent: np.ndarray
    pressure_gram: np.ndarray
    branches: tuple
    quadrature_triangles: int


@dataclass(frozen=True)
class AcceptedState:
    coordinates: np.ndarray
    reduced: np.ndarray
    nodal_forces: np.ndarray
    load: np.ndarray
    generation: int
    criteria: tuple


class EnrichedStaticSystem:
    """Immutable compiled definition plus atomically published accepted state.

    Coordinates [original physical DOFs, four beta per source Q4].  Constraints
    act on the original space with the exact original transformation, then
    an identity block on beta.  All Q1 pressure hats of every source remain.
    No structural-only displacement condensation, pressure floor, or mode
    elimination is performed.
    """

    def __setattr__(self, name, value):
        if hasattr(self, name) and name not in {"_accepted", "_busy"}:
            raise EnrichedProfileError("compiled enriched definition is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, stiffness, transform, offset, physical_dofs, cells,
                 cell_indices, pairs, pressure_ids, reference_support,
                 normal_stiffness, quadrature, element_formulations):
        self.stiffness = readonly(stiffness)
        n = self.stiffness.shape[0]
        if self.stiffness.shape != (n, n) or not np.array_equal(self.stiffness, self.stiffness.T):
            raise EnrichedProfileError("symmetric complete structural energy required")
        self.transform = readonly(transform)
        if self.transform.ndim != 2 or self.transform.shape[0] != n:
            raise EnrichedProfileError("exact constraint/explicit-coordinate layout required")
        self.offset = readonly(offset, (n,))
        self.physical_dofs = int(physical_dofs)
        self.cells = tuple(cells)
        self.cell_indices = tuple(tuple(x) for x in cell_indices)
        self.pairs = tuple(pairs)
        self.pressure_ids = tuple(pressure_ids)
        self.reference_support = readonly(reference_support, (len(self.pressure_ids),))
        if not math.isfinite(normal_stiffness) or normal_stiffness <= 0 or np.any(self.reference_support <= 0):
            raise EnrichedProfileError("physical full-reference support and positive NCP scale required")
        self.scales = readonly(normal_stiffness*self.reference_support)
        quadrature.validate()
        self.quadrature = quadrature
        descriptor = dict(profile=FORMULATION, structural_kinematics="REFERENCE_LINEAR",
                          element_formulations=element_formulations,
                          physical_dofs=self.physical_dofs, total_dofs=n,
                          pressure_ids=self.pressure_ids,
                          stiffness=array_digest(self.stiffness), transform=array_digest(self.transform),
                          offset=array_digest(self.offset), cells=[c.identity for c in self.cells],
                          cell_indices=self.cell_indices, reference_support=array_digest(self.reference_support),
                          scales=array_digest(self.scales), quadrature=vars(quadrature),
                          pairs=[dict(definition=vars(p.definition), indices=p.indices,
                                      reference=array_digest(p.reference_corners),
                                      directors=array_digest(p.reference_directors),
                                      skin=array_digest(p.reference_source), modes=p.mode_indices)
                                 for p in self.pairs])
        self.identity = digest(descriptor)
        self._definition_json = json.dumps(descriptor, sort_keys=True, allow_nan=False)
        self._accepted = None
        self._busy = False

    @property
    def definition(self):
        return json.loads(self._definition_json)

    def coordinate_map(self):
        """Old174/300 physical slots are retained, not renamed or normalized.

        This is only a layout mapping; a new enriched response must satisfy
        the original physical/reference predicates AND tests of every beta.
        It does not reuse a historical source/case acceptance.
        """
        return dict(original_physical_range=(0, self.physical_dofs),
                    complete_range=(0, self.offset.size),
                    beta_by_element={cell.element_id: list(indices[24:])
                                     for cell, indices in zip(self.cells, self.cell_indices)},
                    pressure_ids=self.pressure_ids, pressure_modes_removed=0,
                    original_work_gap_force_reference_gates="REQUIRED_UNRUN",
                    additional_beta_work_tangent_energy_gates="REQUIRED_UNRUN")

    @classmethod
    def from_model(cls, model, definitions, *, normal_stiffness, quadrature):
        from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
        from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
        from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
        from anyfesolver.matrix_assembly import assemble_stiffness_matrix
        from anyfesolver.assembly import build_constraint_transformation
        definitions = tuple(definitions)
        if (not definitions or any(type(d) is not PairDefinition for d in definitions)
                or len({(d.source_element, d.target_element) for d in definitions}) != len(definitions)):
            raise EnrichedProfileError("nonduplicate explicit Q4/S3 pair definitions required")
        if any(type(e) not in (QualifiedE4PLShellElement, QualifiedE4PLS3ShellElement,
                              NativeParityE4PLS3V2DShellElement)
               for e in model.mesh.elements.values()):
            raise EnrichedProfileError("reference-linear elastic qualified Q4/S3 model only")
        for element in model.mesh.elements.values():
            material = model.materials[element.material_name]
            if (getattr(material, "yield_stress", 0.) != 0.
                    or getattr(material, "j2_curve", None) is not None
                    or getattr(material, "hardening_curve", None) is not None):
                raise EnrichedProfileError("every assembled source/target material must be elastic")
            if type(element) is NativeParityE4PLS3V2DShellElement:
                if (element.shell_section is not None
                        or type(element.director_polarity) is not int
                        or element.director_polarity != 1
                        or element.reference_surface_offset != 0.
                        or not NativeParityE4PLS3V2DShellElement._is_v2c_elastic_overlap(material)):
                    raise EnrichedProfileError("V2D elastic overlap domain required")
        source_ids = sorted({d.source_element for d in definitions})
        cells = [compile_cell(model.mesh.elements[i], model.mesh,
                              model.materials[model.mesh.elements[i].material_name]) for i in source_ids]
        n = model.mesh.dof_manager.total_dofs
        old = assemble_stiffness_matrix(model)[0]
        _, _, old_t, old_offset, _, _ = build_constraint_transformation(old, np.zeros(n), model)
        size = n+4*len(cells)
        stiffness = np.zeros((size, size))
        stiffness[:n, :n] = old.toarray()
        cell_indices = []
        for ordinal, cell in enumerate(cells):
            element = model.mesh.elements[cell.element_id]
            indices = tuple(d for node in cell.node_ids for d in model.mesh.dof_manager.get_node_dofs(node))
            indices += tuple(range(n+4*ordinal, n+4*ordinal+4))
            delta = np.array(cell.stiffness)
            delta[:24, :24] -= element.compute_stiffness_components(
                model.mesh, model.materials[element.material_name])["total"]
            stiffness[np.ix_(indices, indices)] += delta
            cell_indices.append(indices)
        stiffness = (stiffness+stiffness.T)/2
        transform = np.zeros((size, old_t.shape[1]+4*len(cells)))
        transform[:n, :old_t.shape[1]] = old_t.toarray()
        transform[n:, old_t.shape[1]:] = np.eye(4*len(cells))
        offset = np.concatenate((old_offset, np.zeros(4*len(cells))))
        pressure_ids = tuple(sorted({node for cell in cells for node in cell.node_ids}))
        pressure_index = {node: i for i, node in enumerate(pressure_ids)}
        reference_support = np.zeros(len(pressure_ids))
        for cell in cells:
            # Affine rectangle, full uncut reference Q1 support = area/4.
            area = np.linalg.norm(np.cross(cell.reference[1]-cell.reference[0],
                                           cell.reference[3]-cell.reference[0]))
            for node in cell.node_ids:
                reference_support[pressure_index[node]] += area/4
        pairs = []
        for d in definitions:
            if not all(math.isfinite(x) for x in (d.source_offset, d.target_offset)):
                raise EnrichedProfileError("finite physical offsets required")
            cell = cells[source_ids.index(d.source_element)]
            target = model.mesh.elements[d.target_element]
            if type(target) not in (QualifiedE4PLS3ShellElement, NativeParityE4PLS3V2DShellElement):
                raise EnrichedProfileError("physical target must be qualified S3")
            target_ref = np.asarray(target.get_node_coordinates(model.mesh), dtype=float)
            target_normal = np.cross(target_ref[1]-target_ref[0], target_ref[2]-target_ref[0])
            target_normal /= np.linalg.norm(target_normal)
            directors = np.array([cell.frame[:, 2]]*4+[target_normal]*3)
            corners = np.vstack((cell.reference, target_ref))
            indices = tuple(dof for node in (*cell.node_ids, *target.node_ids)
                            for dof in model.mesh.dof_manager.get_node_dofs(node))
            if len(set((*cell.node_ids, *target.node_ids))) != 7:
                raise EnrichedProfileError("source and target nodes must be disjoint")
            indices += cell_indices[source_ids.index(d.source_element)][24:]
            skin_ref = cell.reference+d.source_offset*cell.frame[:, 2]
            pairs.append(CompiledPair(d, indices, readonly(corners, (7, 3)),
                                      readonly(directors, (7, 3)), readonly(skin_ref, (4, 3)),
                                      tuple(pressure_index[node] for node in cell.node_ids)))
        return cls(stiffness=stiffness, transform=transform, offset=offset,
                   physical_dofs=n, cells=cells, cell_indices=cell_indices, pairs=pairs,
                   pressure_ids=pressure_ids, reference_support=reference_support,
                   normal_stiffness=normal_stiffness, quadrature=quadrature,
                   element_formulations=[dict(element_id=int(key),
                       class_module=type(element).__module__, class_name=type(element).__qualname__,
                       formulation_id=element.formulation_id)
                       for key, element in sorted(model.mesh.elements.items())])

    def linearize(self, coordinates, nodal_forces, *, cancel=lambda: False):
        u = readonly(coordinates, self.offset.shape)
        z = readonly(nodal_forces, self.reference_support.shape)
        n, m = u.size, z.size
        support, b = np.zeros(m), np.zeros(m)
        dm, db = np.zeros((m, n)), np.zeros((m, n))
        hm, hb = np.zeros((m, n, n)), np.zeros((m, n, n))
        gram = np.zeros((m, m))
        branches, count = [], 0
        for pair in self.pairs:
            check_cancel(cancel)
            x, a, h = pair.map(u)
            data = moments(pair.reference_source, x[:12].reshape(4, 3),
                           x[12:21].reshape(3, 3), x[21:25], pair.reference_directors[0],
                           self.quadrature, current_midsurface=x[25:37].reshape(4, 3),
                           cancel=cancel)
            count += data.triangles
            branches.append(data.branch)
            ix = np.asarray(pair.indices)
            modes = np.asarray(pair.mode_indices)
            gram[np.ix_(modes, modes)] += data.pressure_gram
            for local, mode in enumerate(modes):
                support[mode] += data.support[local]
                b[mode] += data.weighted_gap[local]
                for g, hh, gradient, hessian in (
                        (dm, hm, data.support_gradient[local], data.support_hessian[local]),
                        (db, hb, data.gap_integral_gradient[local], data.gap_integral_hessian[local])):
                    g[mode, ix] += gradient @ a
                    hh[mode][np.ix_(ix, ix)] += a.T @ hessian @ a+np.einsum("i,ijk->jk", gradient, h)
        supported = support > 0
        gap, pressure = np.zeros(m), np.zeros(m)
        gap[supported] = b[supported]/support[supported]
        pressure[supported] = z[supported]/support[supported]
        g, hg = np.zeros((m, n)), np.zeros((m, n, n))
        for i in np.flatnonzero(supported):
            g[i] = (db[i]-gap[i]*dm[i])/support[i]
            hg[i] = (hb[i]-gap[i]*hm[i]-np.outer(g[i], dm[i])-np.outer(dm[i], g[i]))/support[i]
        force = -g.T @ z
        tangent = -np.einsum("i,ijk->jk", z, hg)
        check_cancel(cancel)
        return ContactState(*[readonly(x) for x in (support, b, gap, pressure, g, force, tangent, gram)],
                            tuple(branches), count)

    def equations(self, reduced, z, load, *, cancel=lambda: False):
        reduced = readonly(reduced, (self.transform.shape[1],))
        load = readonly(load, self.offset.shape)
        z = readonly(z, self.reference_support.shape)
        u = self.offset+self.transform @ reduced
        raw = self.linearize(u, z, cancel=cancel)
        active = (raw.support > 0) & (z-self.scales*raw.normalized_gap > 0)
        complementarity = np.where(active, self.scales*raw.normalized_gap, z)
        residual = self.transform.T @ (self.stiffness @ u+raw.force-load)
        g = raw.gap_gradient @ self.transform
        lower = np.zeros_like(g)
        lower[active] = self.scales[active, None]*g[active]
        jacobian = np.block([[self.transform.T @ (self.stiffness+raw.tangent) @ self.transform, -g.T],
                             [lower, np.diag((~active).astype(float))]])
        return u, raw, residual, complementarity, jacobian

    @staticmethod
    def metrics(residual, complementarity, criteria):
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(complementarity)):
            raise EnrichedProfileError("nonfinite mixed residual")
        values = (float(np.linalg.norm(residual)/criteria.force_scale_n),
                  float(np.linalg.norm(complementarity, 1)/criteria.force_scale_n))
        if not all(math.isfinite(x) for x in values):
            raise EnrichedProfileError("nonfinite mixed residual metric")
        return values

    def _mixed_direction(self,reduced,z,raw,residual,complementarity,jacobian,
                         *,iteration,counts,cancel):
        """Default direction is the original mixed Newton solve."""
        delta=np.linalg.solve(jacobian,-np.concatenate((residual,complementarity)))
        counts["linear_solves"]+=1
        return delta

    def solve(self, load, *, criteria, control, cancel=lambda: False):
        """Development NCP solve; exact physical observer, no global acceptance.

        Iterates and rejected trials are local.  Only a fully converged NCP
        candidate is published; cancellation/refusal leaves accepted identity
        and checkpoint bytes intact.  Physical-gap extrema/reference/pressure
        stability qualification remains an independent, mandatory open gate.
        """
        if self._busy:
            raise EnrichedProfileError("reentrant enriched transaction refused")
        criteria.validate()
        control.validate()
        load = readonly(load, self.offset.shape)
        self._busy = True
        try:
            check_cancel(cancel)
            if self._accepted is None:
                reduced = np.zeros(self.transform.shape[1])
                z = np.zeros(len(self.pressure_ids))
            else:
                if self._accepted.criteria != tuple(vars(criteria).values()):
                    raise EnrichedProfileError("accepted criteria identity changed")
                reduced = np.array(self._accepted.reduced)
                z = np.array(self._accepted.nodal_forces)
            counts = dict(evaluations=0, linear_solves=0, rejected_trials=0)
            for iteration in range(control.maximum_iterations+1):
                u, raw, r, c, j = self.equations(reduced, z, load, cancel=cancel)
                counts["evaluations"] += 1
                eq, ncp = self.metrics(r, c, criteria)
                if eq <= criteria.equilibrium_relative and ncp <= criteria.complementarity_relative:
                    if np.any(z < 0):
                        raise EnrichedProfileError("negative converged nodal force")
                    check_cancel(cancel)
                    generation = 1 if self._accepted is None else self._accepted.generation+1
                    self._accepted = AcceptedState(readonly(u), readonly(reduced), readonly(z),
                                                   load, generation, tuple(vars(criteria).values()))
                    return dict(status="DEVELOPMENT_NCP_CONVERGED", criteria=vars(criteria),
                                equilibrium=eq, complementarity=ncp, counts=counts,
                                generation=generation, physical_gap_extrema="UNVERIFIED",
                                pressure_stability="UNPROVEN", full_D1_acceptance=False)
                if iteration == control.maximum_iterations:
                    break
                check_cancel(cancel)
                delta = self._mixed_direction(reduced,z,raw,r,c,j,
                    iteration=iteration,counts=counts,cancel=cancel)
                if not np.all(np.isfinite(delta)):
                    raise EnrichedProfileError("nonfinite mixed Newton correction")
                merit = max(eq/criteria.equilibrium_relative, ncp/criteria.complementarity_relative)
                for half in range(control.maximum_halvings+1):
                    check_cancel(cancel)
                    alpha = 2.**(-half)
                    rq = reduced+alpha*delta[:reduced.size]
                    rz = z+alpha*delta[reduced.size:]
                    _, _, rr, cc, _ = self.equations(rq, rz, load, cancel=cancel)
                    counts["evaluations"] += 1
                    ee, nn = self.metrics(rr, cc, criteria)
                    if max(ee/criteria.equilibrium_relative, nn/criteria.complementarity_relative) < merit:
                        reduced, z = rq, rz
                        break
                    counts["rejected_trials"] += 1
                else:
                    raise EnrichedProfileError("mixed backtracking exhausted")
            raise EnrichedProfileError("mixed iteration budget exhausted")
        finally:
            self._busy = False

    def checkpoint(self):
        if self._accepted is None:
            raise EnrichedProfileError("no accepted enriched state")
        a = self._accepted
        payload = dict(profile=FORMULATION, definition=self.identity,
                       coordinates=a.coordinates.tolist(), nodal_forces=a.nodal_forces.tolist(),
                       reduced=a.reduced.tolist(), criteria=list(a.criteria),
                       load=a.load.tolist(), generation=a.generation)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def restore(self, checkpoint, *, criteria, cancel=lambda: False):
        """Typed profile/definition and actual residual admission BEFORE publish."""
        if self._busy or type(checkpoint) is not bytes or len(checkpoint) > 2*1024*1024:
            raise EnrichedProfileError("invalid checkpoint/transaction envelope")
        criteria.validate()
        try:
            payload = json.loads(checkpoint)
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()
        except (ValueError, UnicodeError) as exc:
            raise EnrichedProfileError("malformed enriched checkpoint") from exc
        if (type(payload) is not dict or set(payload) != {"profile", "definition", "coordinates",
                "reduced", "criteria", "nodal_forces", "load", "generation"} or payload["profile"] != FORMULATION
                or payload["definition"] != self.identity or type(payload["generation"]) is not int
                or payload["generation"] < 1 or payload["criteria"] != list(vars(criteria).values())
                or checkpoint != canonical):
            raise EnrichedProfileError("wrong enriched checkpoint profile/definition/layout")
        u = readonly(payload["coordinates"], self.offset.shape)
        z = readonly(payload["nodal_forces"], self.reference_support.shape)
        load = readonly(payload["load"], self.offset.shape)
        reduced = readonly(payload["reduced"], (self.transform.shape[1],))
        if not np.array_equal(self.offset+self.transform @ reduced, u):
            raise EnrichedProfileError("checkpoint violates original constraints")
        self._busy = True
        try:
            _, _, r, c, _ = self.equations(reduced, z, load, cancel=cancel)
            eq, ncp = self.metrics(r, c, criteria)
            if eq > criteria.equilibrium_relative or ncp > criteria.complementarity_relative or np.any(z < 0):
                raise EnrichedProfileError("checkpoint physical residual admission failed")
            check_cancel(cancel)
            self._accepted = AcceptedState(u, reduced, z, load, payload["generation"],
                                           tuple(vars(criteria).values()))
        finally:
            self._busy = False
