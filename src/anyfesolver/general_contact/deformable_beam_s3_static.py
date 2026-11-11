"""Opt-in small-motion B2/B3 contact against a deforming, fixed-director S3.

The physical skin work map owns offsets and force couples. Nodal translations
may differ; all target rotation coordinates must remain zero. Reference line
measure and exact active-interval polynomial quadrature define the penalty law.
This development profile does not admit rims, multiple faces or finite rotations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .beam_shell import _BeamShellResults, validate_beam_shell_contact
from .deformable_beam_s3_line import DeformableBeamS3Line
from .s3_skin_work import S3SkinWorkAdapter

PROFILE = "JET2_ROUND_BEAM_DEFORMABLE_FIXED_DIRECTOR_S3_STATIC_DEV_V0"


@dataclass(frozen=True)
class DeformableBeamS3Results(_BeamShellResults):
    pair_name: str
    beam_surface: str
    shell_surface: str
    beam_element: int
    shell_element: int

    def to_dict(self):
        payload = super().to_dict()
        payload.update(pair_identity=self.pair_name,
                       physical_features={"beam": {"surface":self.beam_surface,
                                                   "element_id":self.beam_element},
                                          "shell": {"surface":self.shell_surface,
                                                    "element_id":self.shell_element}},
                       moment_origin_m=[0.,0.,0.],
                       contact_integration_measure="reference_line_length",
                       integration_rule="three_gauss_points_per_exact_active_interval",
                       integration_error_semantics="polynomial_truncation_only")
        return payload


class DeformableBeamS3Manager:
    def __init__(self, model, definition, settings, initial_displacement=None):
        report = validate_beam_shell_contact(model, definition, settings)
        if not report.valid or report.profile != PROFILE:
            raise ValueError("unsupported deformable beam-S3: " + "; ".join(report.issues))
        self.model, self.definition, self.settings = model, definition, settings
        self.profile, self.swept_trial_cutbacks = PROFILE, 0
        self.beam_surface = next(s for s in definition.surfaces if s.beam_radius is not None)
        self.end_caps = self.beam_surface.beam_end_caps
        self.shell_surface = next(s for s in definition.surfaces if s.beam_radius is None)
        beam = model.mesh.elements[self.beam_surface.element_ids[0]]
        self.beam_ids = tuple(beam.node_ids)
        self.beam_reference = np.asarray(beam.get_node_coordinates(model.mesh), float)
        self.beam_dofs = np.array([model.mesh.dof_manager.get_node_dofs(n)[:6]
                                   for n in self.beam_ids], dtype=int)
        self.shell = S3SkinWorkAdapter.from_model(model, 2,
                        self.shell_surface.element_ids[0], self.shell_surface.side)
        self.shell_dofs = np.array([model.mesh.dof_manager.get_node_dofs(n)[:6]
                                    for n in self.shell.node_ids], dtype=int)
        total = model.mesh.dof_manager.total_dofs
        beam_map = sparse.coo_matrix((np.ones(3*len(self.beam_ids)),
                        (np.arange(3*len(self.beam_ids)), self.beam_dofs[:, :3].ravel())),
                        shape=(3*len(self.beam_ids), total)).tocsr()
        self.work_map = sparse.vstack((beam_map, self.shell.work_map)).tocsr()
        interaction = definition.interactions[0]
        self.gap_tolerance, self.line_stiffness = interaction.gap_tolerance, interaction.line_stiffness
        side = 1. if self.shell_surface.side == "top" else -1.
        self.kernel = DeformableBeamS3Line(self.beam_reference, self.shell.reference_positions,
                        radius=self.beam_surface.beam_radius, line_stiffness=self.line_stiffness,
                        side_normal=side*self.shell.director,
                        rim_clearance=settings.beam_shell_rim_clearance)
        u = np.zeros(total) if initial_displacement is None else np.asarray(initial_displacement, float)
        b, s = self._positions(u)
        gap = self.kernel.evaluate(b, s).minimum_gap
        numerical = 512*np.finfo(float).eps*max(np.max(np.abs(b)),np.max(np.abs(s)),self.kernel.radius)
        if gap < -numerical:
            raise ValueError("deformable beam-S3 has unresolved initial penetration")

    def _positions(self, displacement):
        u = np.asarray(displacement, float).reshape(-1)
        if u.size != self.work_map.shape[1] or not np.isfinite(u).all():
            raise ValueError("deformable beam-S3 displacement invalid")
        if np.any(u[self.shell_dofs[:, 3:]] != 0.):
            raise ValueError("deformable beam-S3 requires zero target rotational coordinates")
        return (self.beam_reference+u[self.beam_dofs[:, :3]],
                self.shell.current_positions(u))

    def evaluate(self, displacement, *, tangent=True):
        b, s = self._positions(displacement)
        state = self.kernel.evaluate(b, s)
        residual = np.asarray(self.work_map.T@state.gradient).reshape(-1)
        K = (self.work_map.T@sparse.csr_matrix(state.hessian)@self.work_map).tocsr() if tangent else None
        forces = -state.gradient.reshape(-1,3)
        nb = len(b)
        bf, sf = forces[:nb].sum(axis=0), forces[nb:].sum(axis=0)
        bm, sm = np.cross(b, forces[:nb]).sum(axis=0), np.cross(s, forces[nb:]).sum(axis=0)
        result = DeformableBeamS3Results(PROFILE, state.minimum_gap, state.potential,
                    state.active_line_length, len(state.active_intervals), state.integration_points,
                    0., state.minimum_rim_clearance, tuple(bf), tuple(sf), tuple(bm), tuple(sm),
                    state.minimum_gap >= -self.gap_tolerance, self.line_stiffness,
                    self.definition.interactions[0].name, self.beam_surface.name,
                    self.shell_surface.name, self.beam_surface.element_ids[0],
                    self.shell_surface.element_ids[0])
        return residual, K, result

    def _swept(self, start, end):
        b0,s0 = self._positions(start); b1,s1 = self._positions(end)
        return self.kernel.swept_admissible(b0,s0,b1,s1,gap_tolerance=self.gap_tolerance,
                            maximum_intervals=self.settings.beam_shell_maximum_swept_intervals)

    def admissible_step_fraction(self, start, end):
        first, last = np.asarray(start,float),np.asarray(end,float)
        # Bounded halving permits activation inside the explicit gap budget.
        for cut in range(self.settings.maximum_swept_subdivisions+1):
            fraction = 2.**(-cut)
            try:
                admissible = self._swept(first,first+fraction*(last-first))
            except ValueError as exc:
                if "rotational" in str(exc) or "displacement invalid" in str(exc):
                    raise
                admissible = False
            if admissible:
                self.swept_trial_cutbacks += cut
                return fraction
        self.swept_trial_cutbacks += self.settings.maximum_swept_subdivisions+1
        return 0.

    def swept_increment_admissible(self,start,end,_start_factor,_end_factor):
        try:
            good = self._swept(start,end)
        except ValueError as exc:
            if "rotational" in str(exc) or "displacement invalid" in str(exc):
                raise
            good = False
        self.swept_trial_cutbacks += int(not good)
        return good

    def restart_contract(self):
        return {"profile": PROFILE, "revision": "complete_moving_normal_reference_line_v1",
                "surfaces": [{"name":s.name,"element_ids":list(s.element_ids),"side":s.side,
                               "radius":s.beam_radius,"end_caps":s.beam_end_caps}
                              for s in self.definition.surfaces],
                "interaction":{"name":self.definition.interactions[0].name,
                               "normal_law":"penalty","line_stiffness":self.line_stiffness,
                               "gap_tolerance":self.gap_tolerance},
                "settings":{"geometry":self.settings.beam_shell_geometry,
                            "rim_clearance":self.settings.beam_shell_rim_clearance,
                            "maximum_swept_intervals":self.settings.beam_shell_maximum_swept_intervals,
                            "maximum_swept_subdivisions":self.settings.maximum_swept_subdivisions}}
