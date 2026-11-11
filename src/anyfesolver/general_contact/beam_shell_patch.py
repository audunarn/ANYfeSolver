"""B2/B3 line contact on a conforming convex patch of planar shell faces.

Material node IDs own seams. The only clearance boundary is the patch exterior;
the contact line is partitioned into disjoint intervals before face integration.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .beam_shell import (
    PROFILE_PATCH, InteriorBeamShellLine, InteriorBeamShellState,
    _BeamShellManager, _beam_shape, _bezier_controls, validate_beam_shell_contact,
)
from .beam_plane import _gap_coefficients
from .core import ContactDefinition, ContactSolverSettings, ContactValidationReport
from .overlap import PlanarFace, reference_shell_face


def _patch_windows(faces: tuple[PlanarFace, ...],
                   face_node_ids: tuple[tuple[int, ...], ...],
                   beam: np.ndarray, rim_clearance: float,
                   ) -> tuple[tuple[tuple[int, float, float], ...], float]:
    """Certify a convex patch and partition a B2/B3 projected curve."""
    first = faces[0].corners
    origin = first[0]
    tangent = first[1] - origin
    tangent /= np.linalg.norm(tangent)
    normal = np.cross(first[1] - origin, first[-1] - origin)
    normal /= np.linalg.norm(normal)
    binormal = np.cross(normal, tangent)
    scale = max(float(np.max(np.linalg.norm(face.corners - origin, axis=1)))
                for face in faces)
    tolerance = 1.0e-10 * scale
    points: dict[int, np.ndarray] = {}
    face_xy = []
    directed_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    area_sum = 0.0
    for index, (face, node_ids) in enumerate(zip(faces, face_node_ids)):
        corners = face.corners
        geometric = np.cross(corners[1] - corners[0], corners[-1] - corners[0])
        geometric /= np.linalg.norm(geometric)
        if float(geometric @ normal) < 1.0 - 1.0e-10:
            raise ValueError("beam-shell patch faces have inconsistent physical orientation")
        if float(np.max(np.abs((corners - origin) @ normal))) > tolerance:
            raise ValueError("beam-shell patch faces are not coplanar")
        xy = np.column_stack(((corners - origin) @ tangent,
                              (corners - origin) @ binormal))
        face_xy.append(xy)
        area_sum += 0.5 * sum(float(np.linalg.det(np.stack((xy[i], xy[(i + 1) % len(xy)]))))
                              for i in range(len(xy)))
        for node_id, point in zip(node_ids, xy):
            if node_id in points and np.linalg.norm(points[node_id] - point) > tolerance:
                raise ValueError("beam-shell patch has discontinuous physical seam geometry")
            points[node_id] = point
        for left, right in zip(node_ids, node_ids[1:] + node_ids[:1]):
            key = tuple(sorted((left, right)))
            directed_edges.setdefault(key, []).append((left, right))
    boundary = []
    for owners in directed_edges.values():
        if len(owners) == 1:
            boundary.append(owners[0])
        elif len(owners) != 2 or owners[0] != owners[1][::-1]:
            raise ValueError("beam-shell patch has overlapping or nonmanifold face edges")
    if not boundary:
        raise ValueError("beam-shell patch has no physical exterior rim")
    successor = {}
    predecessor = {}
    for left, right in boundary:
        if left in successor or right in predecessor:
            raise ValueError("beam-shell patch has a branched exterior rim")
        successor[left] = right
        predecessor[right] = left
    cursor = boundary[0][0]
    visited = set()
    while cursor in successor and cursor not in visited:
        visited.add(cursor)
        cursor = successor[cursor]
    if cursor != boundary[0][0] or len(visited) != len(boundary):
        raise ValueError("beam-shell patch must have one connected exterior rim")
    boundary_area = 0.5 * sum(float(np.linalg.det(np.stack((points[a], points[b]))))
                              for a, b in boundary)
    if abs(area_sum - boundary_area) > 1.0e-9 * scale * scale:
        raise ValueError("beam-shell patch faces overlap or leave an uncovered region")
    projected_beam = np.column_stack(((_bezier_controls(beam) - origin) @ tangent,
                                      (_bezier_controls(beam) - origin) @ binormal))
    minimum_rim = float("inf")
    for left, right in boundary:
        corner = points[left]
        edge = points[right] - corner
        length = float(np.linalg.norm(edge))
        for point in points.values():
            inward = float(np.linalg.det(np.stack((edge, point - corner))) / length)
            if inward < -tolerance:
                raise ValueError("beam-shell patch exterior is not convex")
        for point in projected_beam:
            inward = float(np.linalg.det(np.stack((edge, point - corner))) / length)
            minimum_rim = min(minimum_rim, inward)
    if minimum_rim < rim_clearance - tolerance:
        raise ValueError("beam-shell curve approaches an unadmitted physical rim")

    if len(beam) == 3:
        beam_xy = np.column_stack(((beam - origin) @ tangent,
                                   (beam - origin) @ binormal))
        cuts = [-1.0, 1.0]
        for xy in face_xy:
            for corner, nxt in zip(xy, np.roll(xy, -1, axis=0)):
                edge = nxt - corner
                signed = tuple(float(np.linalg.det(np.stack((edge, point - corner))))
                               for point in beam_xy)
                a, b, c = _gap_coefficients(signed)
                coefficient_scale = max(abs(a), abs(b), abs(c))
                if coefficient_scale == 0.0:
                    continue
                if abs(a) > 64.0 * np.finfo(float).eps * coefficient_scale:
                    roots = np.roots((a, b, c))
                elif abs(b) > 64.0 * np.finfo(float).eps * coefficient_scale:
                    roots = np.roots((b, c))
                else:
                    continue
                for root in roots:
                    if abs(float(np.imag(root))) <= 1.0e-12:
                        xi = float(np.real(root))
                        if -1.0 + 1.0e-12 < xi < 1.0 - 1.0e-12:
                            cuts.append(xi)
        cuts.sort()
        unique_cuts = [cuts[0]]
        for cut in cuts[1:]:
            if cut - unique_cuts[-1] > 1.0e-12:
                unique_cuts.append(cut)
        windows = []
        for left, right in zip(unique_cuts[:-1], unique_cuts[1:]):
            middle = 0.5 * (left + right)
            point = _beam_shape(middle, 3) @ beam_xy
            owners = []
            for index, xy in enumerate(face_xy):
                if all(float(np.linalg.det(np.stack((nxt - corner, point - corner))))
                       >= -tolerance * float(np.linalg.norm(nxt - corner))
                       for corner, nxt in zip(xy, np.roll(xy, -1, axis=0))):
                    owners.append(index)
            if not owners:
                raise ValueError("beam-shell patch curve projection is not fully covered")
            owner = min(owners, key=lambda i: faces[i].feature_id)
            if windows and windows[-1][0] == owner and abs(windows[-1][2] - left) < 1.0e-12:
                windows[-1] = (owner, windows[-1][1], right)
            else:
                windows.append((owner, left, right))
        return tuple(windows), minimum_rim

    raw = []
    for index, xy in enumerate(face_xy):
        left, right = 0.0, 1.0
        for corner, nxt in zip(xy, np.roll(xy, -1, axis=0)):
            edge = nxt - corner
            h0 = float(np.linalg.det(np.stack((edge, projected_beam[0] - corner))))
            h1 = float(np.linalg.det(np.stack((edge, projected_beam[1] - corner))))
            if h0 < -tolerance and h1 < -tolerance:
                left, right = 1.0, 0.0
                break
            difference = h1 - h0
            if abs(difference) <= 1.0e-15 * scale * scale:
                continue
            crossing = -h0 / difference
            if difference > 0.0:
                left = max(left, crossing)
            else:
                right = min(right, crossing)
        left, right = max(0.0, left), min(1.0, right)
        if right - left > 1.0e-12:
            raw.append((index, left, right))
    cuts = sorted({0.0, 1.0, *(left for _, left, _ in raw),
                   *(right for _, _, right in raw)})
    windows = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        if right - left <= 1.0e-12:
            continue
        middle = 0.5 * (left + right)
        owners = [index for index, start, stop in raw if start <= middle <= stop]
        if not owners:
            raise ValueError("beam-shell patch line projection is not fully covered")
        # An exactly seam-aligned line has two mathematical owners. Assign it
        # once by stable element order, without doubling its distributed work.
        owner = min(owners, key=lambda i: faces[i].feature_id)
        if windows and windows[-1][0] == owner and abs(windows[-1][2] - left) < 1.0e-12:
            windows[-1] = (owner, windows[-1][1], right)
        else:
            windows.append((owner, left, right))
    if not windows or windows[0][1] > 1.0e-12 or windows[-1][2] < 1.0 - 1.0e-12:
        raise ValueError("beam-shell patch line projection is not fully covered")
    return tuple((i, 2.0 * left - 1.0, 2.0 * right - 1.0)
                 for i, left, right in windows), minimum_rim


def validate_beam_shell_patch_contact(model, definition: ContactDefinition,
                                      settings: ContactSolverSettings,
                                      ) -> ContactValidationReport:
    beams = [item for item in definition.surfaces if item.beam_radius is not None]
    shells = [item for item in definition.surfaces if item.beam_radius is None]
    if len(beams) != 1 or len(shells) != 1:
        return ContactValidationReport("invalid", ("patch needs one beam and one shell surface",),
                                       PROFILE_PATCH)
    beam, shell = beams[0], shells[0]
    issues = []
    if len(shell.element_ids) < 2 or len(shell.element_ids) > 64:
        issues.append("patch profile admits 2 to 64 shell faces")
    if len(beam.element_ids) != 1:
        issues.append("patch profile admits one classical B2/B3 cell")
    else:
        from anyfesolver.elements import BeamElement, QuadraticBeamElement

        if type(model.mesh.elements.get(beam.element_ids[0])) not in {
            BeamElement, QuadraticBeamElement,
        }:
            issues.append("patch profile admits an exact classical B2/B3 element")
    for element_id in shell.element_ids:
        single = replace(shell, element_ids=(element_id,))
        report = validate_beam_shell_contact(
            model, replace(definition, surfaces=(beam, single)), settings,
        )
        issues.extend(f"face {element_id}: {issue}" for issue in report.issues)
    if not issues:
        try:
            faces = tuple(reference_shell_face(model, 2, eid, shell.side)
                          for eid in shell.element_ids)
            node_ids = tuple(tuple(int(n) for n in model.mesh.elements[eid].node_ids)
                             for eid in shell.element_ids)
            beam_element = model.mesh.elements[beam.element_ids[0]]
            beam_reference = np.asarray(beam_element.get_node_coordinates(model.mesh), dtype=float)
            _patch_windows(faces, node_ids, beam_reference,
                           settings.beam_shell_rim_clearance)
            levers = []
            side_normals = []
            from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id

            for eid in shell.element_ids:
                element = model.mesh.elements[eid]
                director = element.physical_reference_director
                if director is None:
                    corners = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
                    director = np.cross(corners[1] - corners[0], corners[-1] - corners[0])
                    director /= np.linalg.norm(director)
                sign = 1.0 if shell.side == "top" else -1.0
                side_normals.append(sign * np.asarray(director, dtype=float))
                offset = (element.material_surface_offset_from_reference(sign)
                          if element.formulation_id == s3_id else sign * element.thickness / 2.0)
                levers.append(float(offset) * np.asarray(director, dtype=float))
            if any(float(normal @ side_normals[0]) < 1.0 - 1.0e-10
                   for normal in side_normals[1:]):
                issues.append("patch faces need one continuous physical side orientation")
            if any(np.linalg.norm(lever - levers[0]) > 1.0e-10 * max(1.0, np.linalg.norm(levers[0]))
                   for lever in levers[1:]):
                issues.append("patch faces need one common physical skin lever")
        except (KeyError, ValueError, TypeError) as exc:
            issues.append(str(exc))
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), PROFILE_PATCH)


class _BeamShellPatchManager(_BeamShellManager):
    """Use the single-face transaction and result route with patch-wide ownership."""

    def __init__(self, model, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 initial_displacement: np.ndarray | None = None):
        self.settings = ContactSolverSettings() if settings is None else settings
        report = validate_beam_shell_patch_contact(model, definition, self.settings)
        if not report.valid:
            raise ValueError("unsupported beam-shell patch: " + "; ".join(report.issues))
        self.model = model
        self.definition = definition
        self.profile = report.profile
        self.beam_surface = next(s for s in definition.surfaces if s.beam_radius is not None)
        self.shell_surface = next(s for s in definition.surfaces if s.beam_radius is None)
        interaction = definition.interactions[0]
        self.gap_tolerance = interaction.gap_tolerance
        self.stiffness = interaction.line_stiffness
        self.difference_step = self.settings.beam_shell_difference_step
        self.swept_trial_cutbacks = 0
        self.shell_common_translation_only = True
        beam_element = model.mesh.elements[self.beam_surface.element_ids[0]]
        self.beam_ids = tuple(int(n) for n in beam_element.node_ids)
        self.beam_reference = np.asarray(beam_element.get_node_coordinates(model.mesh), dtype=float)
        self.beam_dofs = np.asarray(
            [model.mesh.dof_manager.get_node_dofs(n)[:6] for n in self.beam_ids], dtype=np.intp,
        )
        self.face_node_ids = tuple(
            tuple(int(n) for n in model.mesh.elements[eid].node_ids)
            for eid in self.shell_surface.element_ids
        )
        self.shell_ids = tuple(sorted({n for face in self.face_node_ids for n in face}))
        indices = {n: i for i, n in enumerate(self.shell_ids)}
        self.face_indices = tuple(tuple(indices[n] for n in face) for face in self.face_node_ids)
        self.shell_dofs = np.asarray(
            [model.mesh.dof_manager.get_node_dofs(n)[:6] for n in self.shell_ids], dtype=np.intp,
        )
        self.shell_node_reference = np.asarray(
            [model.mesh.get_node(n).coords() for n in self.shell_ids], dtype=float,
        )
        self.faces_reference = tuple(
            reference_shell_face(model, 2, eid, self.shell_surface.side)
            for eid in self.shell_surface.element_ids
        )
        first_element = model.mesh.elements[self.shell_surface.element_ids[0]]
        director = first_element.physical_reference_director
        if director is None:
            corners = np.asarray(first_element.get_node_coordinates(model.mesh), dtype=float)
            director = np.cross(corners[1] - corners[0], corners[-1] - corners[0])
            director /= np.linalg.norm(director)
        sign = 1.0 if self.shell_surface.side == "top" else -1.0
        from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id

        offset = (first_element.material_surface_offset_from_reference(sign)
                  if first_element.formulation_id == s3_id else sign * first_element.thickness / 2.0)
        self.shell_lever = float(offset) * np.asarray(director, dtype=float)
        self.kernels = tuple(
            InteriorBeamShellLine(
                self.beam_reference, face, sign * np.asarray(director, dtype=float),
                self.beam_surface.beam_radius, self.stiffness,
                self.settings.beam_shell_rim_clearance,
                self.settings.line_integration_relative_tolerance,
                self.settings.maximum_line_subdivisions,
            ) for face in self.faces_reference
        )
        self.kernel = self.kernels[0]
        self.motion_tolerance = (256.0 * np.finfo(float).eps
                                 * max(self.kernel.reference_length, self.kernel.radius))
        self.gap_roundoff = max(1.0e-10 * self.kernel.reference_length,
                                16.0 * self.motion_tolerance)
        total = model.mesh.dof_manager.total_dofs
        initial = (np.zeros(total) if initial_displacement is None
                   else np.asarray(initial_displacement, dtype=float).reshape(-1))
        if initial.size != total or not np.all(np.isfinite(initial)):
            raise ValueError("beam-shell patch initial displacement is invalid")
        initial_state = self._assemble(initial, enforce_motion=True)[1]
        if initial_state.minimum_gap < -self.motion_tolerance:
            raise ValueError("beam-shell patch has unresolved initial penetration")

    def _positions(self, u: np.ndarray, *, enforce_motion: bool):
        if u.size != self.model.mesh.dof_manager.total_dofs or not np.all(np.isfinite(u)):
            raise ValueError("beam-shell patch trial displacement is invalid")
        shell_translations = u[self.shell_dofs[:, :3]]
        shell_rotations = u[self.shell_dofs[:, 3:]]
        if enforce_motion and (
            float(np.max(np.linalg.norm(shell_translations - shell_translations[0], axis=1)))
            > self.motion_tolerance
            or float(np.max(np.linalg.norm(shell_rotations, axis=1))) > self.motion_tolerance
        ):
            raise ValueError("beam-shell patch requires common shell translation and fixed rotations")
        beam = self.beam_reference + u[self.beam_dofs[:, :3]]
        faces = tuple(
            reference.corners + shell_translations[list(indices)]
            + np.cross(shell_rotations[list(indices)], self.shell_lever)
            for reference, indices in zip(self.faces_reference, self.face_indices)
        )
        return beam, faces

    def _assemble(self, u: np.ndarray, *, enforce_motion: bool):
        beam, face_points = self._positions(u, enforce_motion=enforce_motion)
        faces = tuple(PlanarFace(reference.feature_id, points)
                      for reference, points in zip(self.faces_reference, face_points))
        windows, rim = _patch_windows(
            faces, self.face_node_ids, beam, self.settings.beam_shell_rim_clearance,
        )
        total = self.model.mesh.dof_manager.total_dofs
        force = np.zeros(total)
        beam_residual = np.zeros_like(beam)
        shell_residual = np.zeros((len(self.shell_ids), 3))
        energy = active_length = maximum_error = 0.0
        active_intervals = integration_points = 0
        minimum_gap = float("inf")
        for index, left, right in windows:
            state = self.kernels[index].evaluate(
                beam, face_points[index], integration_window=(left, right), check_rim=False,
            )
            beam_residual += state.beam_residual
            for local, global_index in zip(state.shell_residual, self.face_indices[index]):
                shell_residual[global_index] += local
            energy += state.energy
            active_length += state.active_reference_length
            active_intervals += state.active_intervals
            integration_points += state.integration_points
            maximum_error = max(maximum_error, state.maximum_relative_integration_error)
            minimum_gap = min(minimum_gap, state.minimum_gap)
        for dofs, local in zip(self.beam_dofs, beam_residual):
            force[dofs[:3]] += local
        for dofs, local in zip(self.shell_dofs, shell_residual):
            force[dofs[:3]] += local
            force[dofs[3:]] += np.cross(self.shell_lever, local)
        result = InteriorBeamShellState(
            energy, beam_residual, shell_residual, minimum_gap,
            active_length, active_intervals, integration_points, maximum_error,
            self.kernel.side_normal, rim,
        )
        return force, result
