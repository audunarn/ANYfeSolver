"""Bounded multi-element E4 material-trial seam for F03 pose diagnostics.

This evaluates Q4 or S3-V2D elements from one frozen elastic predecessor and
one complete total-coordinate vector. No state is accepted or committed, and
the small dense gather is an independent G0 oracle, not production assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from anyfesolver.corotational import corotational_element_response
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_p5_seeded.core import canonical

from scripts._b3_e4_f03_shell_finite_pose_candidate import (
    pose_from_e4_total, reference_nodes)
from scripts._b3_e4_f03_shell_trace import ShellPanel


@dataclass(frozen=True)
class TrialPanelResponse:
    residual: np.ndarray
    tangent: np.ndarray
    position: np.ndarray
    frame: np.ndarray
    candidate_states: tuple[bytes, ...]


class FlatE4TrialPanel:
    """Frozen flat Q4/S3-V2D fixture with one unchanged material predecessor."""

    def __init__(self, panel: ShellPanel):
        if (not isinstance(panel, ShellPanel) or panel.topology not in
                ("Q4", "S3-V2D") or not 1 <= panel.n_s <= 8):
            raise ValueError("bounded Q4/S3 F03 trial panel required")
        self.panel = panel
        self.model = FEModel("f03-flat-e4-material-trial")
        self.model.add_material("steel", 210e9, .3, density=7850.)
        for node, coordinates in enumerate(reference_nodes(panel), 1):
            self.model.add_node(node, *coordinates)
        nx = 2*panel.n_s
        self.elements = []
        for j in range(panel.n_y):
            for i in range(nx):
                a = 1+j*(nx+1)+i
                b, d = a+1, a+nx+1
                c = d+1
                patches = ((a, b, c, d),) if panel.topology == "Q4" else (
                    (a, b, c), (a, c, d))
                for nodes in patches:
                    eid = len(self.elements)+1
                    element = create_shell_element(
                        eid, list(nodes), "steel",
                        formulation=("e4-pl" if panel.topology == "Q4"
                                     else "e4-pl-s3-v2d"),
                        thickness=panel.thickness,
                        reference_normal=(0., 0., 1.))
                    self.model.add_element(eid, element)
                    self.elements.append((eid, nodes, element))
        material = self.model.get_material("steel")
        self._material = material
        self._material_keys = frozenset(vars(material))
        self.origins = tuple(
            element.init_nonlinear_state(3) if panel.topology == "Q4" else
            element.init_model_bound_nonlinear_state(self.model.mesh, material, 3)
            for _, _, element in self.elements)
        self._origin_bytes = tuple(canonical(origin) for origin in self.origins)
        self.identity = sha256(canonical(self._descriptor())).hexdigest()
        self.guard()

    def _descriptor(self):
        material = self.model.get_material("steel")
        return dict(
            policy="F03_BOUNDED_FLAT_E4_TRIAL_PANEL_V1",
            topology=self.panel.topology, n_s=self.panel.n_s,
            **({"n_y": self.panel.n_y} if self.panel.n_y != 2 else {}),
            thickness=self.panel.thickness,
            node_count=self.panel.node_count,
            element_count=self.panel.element_count,
            nodes=[(node, self.model.mesh.nodes[node].id,
                    self.model.mesh.nodes[node].coords())
                   for node in sorted(self.model.mesh.nodes)],
            elements=[(eid, nodes, type(element).__name__,
                       element.formulation_id,
                       {key: value for key, value in vars(element).items()
                        if not key.startswith("_")})
                      for eid, nodes, element in self.elements],
            material=vars(material),
            layers=3, production_qualified=False)

    def guard(self):
        if (len(self.model.mesh.elements) != len(self.elements) or
                self.model.get_material("steel") is not self._material or
                frozenset(vars(self._material)) != self._material_keys or
                self._material.hardening_curve is not None or
                any(self.model.mesh.elements.get(eid) is not element
                    for eid, _, element in self.elements) or
                tuple(canonical(origin) for origin in self.origins) !=
                self._origin_bytes or
                sha256(canonical(self._descriptor())).hexdigest() != self.identity):
            raise ValueError("F03 E4 trial definition or predecessor changed")

    def evaluate(self, total_u: np.ndarray, station: float) -> TrialPanelResponse:
        """Evaluate every local trial from the same frozen predecessor."""
        self.guard()
        values = np.asarray(total_u, dtype=float)
        size = 6*self.panel.node_count
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError("finite complete flat E4 total coordinates required")
        nodal = values.reshape(-1, 6)
        residual = np.zeros(size)
        tangent = np.zeros((size, size))
        candidates = []
        for (eid, nodes, element), origin in zip(self.elements, self.origins):
            local = nodal[np.asarray(nodes)-1].ravel()
            force, stiffness, candidate = corotational_element_response(
                self.model, eid, element, local, True,
                committed_state=origin, num_layers=3,
                tangent_mode="consistent")
            if force is None or stiffness is None or candidate is None:
                raise ValueError("E4 corotational material trial unavailable")
            dofs = np.array([6*(node-1)+axis
                             for node in nodes for axis in range(6)])
            np.add.at(residual, dofs, np.asarray(force, dtype=float))
            tangent[np.ix_(dofs, dofs)] += np.asarray(stiffness, dtype=float)
            candidates.append(canonical(candidate))
        position, frame = pose_from_e4_total(self.panel, station, values)
        self.guard()
        if not np.isfinite(residual).all() or not np.isfinite(tangent).all():
            raise ValueError("nonfinite multi-element E4 material trial")
        return TrialPanelResponse(residual, tangent, position, frame,
                                  tuple(candidates))
