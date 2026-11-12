"""Development-only face-local curved shell skin geometry and work maps.

This module maps the physical exterior skin of curved shell bodies whose
neighbouring elements carry *distinct* per-element reference directors (for
example the faceted elastic sphere, where each S3 triangle owns its outward
facet normal).  Because a shared structural node then has different physical
skin corners in each incident element, points are keyed face-locally by
``(body_id, element_id, local_corner)`` instead of by node, and the sparse
work maps accumulate forces and moments back onto the shared authoritative
structural DOFs of the model.

The kinematics reuse the existing corotational nodal director field exactly:
at each face-local corner the physical point is

    X + u + h * Exp([theta]x) * d_element

with the node reference position ``X``, the nodal displacement and additive
rotation ``theta``, the element physical offset ``h`` and the element physical
reference director ``d``.  Values, chart Jacobians, spatial spin maps, error
enclosures and the fixed-spatial-force pullback derivative are the existing
``finite_rotation_skin`` primitives; no new rotation algebra or contact law
is introduced.  Triangles stay linear and quadrilaterals bilinear: the
face interpolation and normal evaluation below reuse exactly those original
shape functions.

This is an explicitly development-only geometry service.  It does not admit a
contact solve, does not change solver admission, and makes no claim that the
offset facets form a watertight exterior or an actual contact gap/force
history.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .finite_rotation_skin import (
    FiniteSkinPose,
    _skew,
    _up,
    director_jets,
    director_value_error,
)
from .q4_skin_work import Q4SkinWorkAdapter
from .s3_skin_work import S3SkinWorkAdapter

# Existing finite-shell development admission bound (total nodal rotation).
TOTAL_ROTATION_BOUND_RAD = 1.2

_EPS = np.finfo(float).eps


class CurvedShellSkinError(ValueError):
    """Typed refusal for invalid requests, poses or bound geometry."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class CurvedSkinFace:
    """One admitted physical skin face with its element-local authority."""

    body_id: int
    element_id: int
    side: str
    formulation_id: str
    node_ids: tuple[int, ...]
    director: np.ndarray
    physical_offset: float
    reference_corners: np.ndarray


def _face_adapter(model, body_id: int, element_id: int, side: str):
    """Reuse the authoritative per-element S3/Q4 skin adapters unchanged."""
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    element = model.mesh.elements.get(int(element_id))
    if element is None:
        raise CurvedShellSkinError(
            "missing_element", f"skin element {element_id} is not in the bound model")
    try:
        if type(element) is NativeParityE4PLS3V2DShellElement:
            return S3SkinWorkAdapter.from_model(model, int(body_id), int(element_id), side)
        if type(element) is QualifiedE4PLShellElement:
            return Q4SkinWorkAdapter.from_model(model, int(body_id), int(element_id), side)
    except ValueError as error:
        raise CurvedShellSkinError("unsupported_face", str(error)) from error
    raise CurvedShellSkinError(
        "unsupported_face",
        f"element {element_id} has no admitted curved skin formulation")


def _element_identity(element):
    """Authoritative director/offset-determining state of one bound element."""
    director = element.physical_reference_director
    return (
        type(element),
        float(element.thickness),
        # Copy: a mutable element director property must not alias the bound
        # snapshot, or a later in-place mutation would evade comparison.
        None if director is None else np.array(director, dtype=float, copy=True),
        float(getattr(element, "reference_surface_offset", 0.0)),
        float(getattr(element, "director_polarity", 1.0)),
    )


def _same_element_identity(current, bound) -> bool:
    if current[0] is not bound[0]:
        return False
    if (current[1] != bound[1] or current[3] != bound[3]
            or current[4] != bound[4]):
        return False
    if current[2] is None or bound[2] is None:
        return current[2] is None and bound[2] is None
    return bool(np.array_equal(current[2], bound[2]))


class CurvedShellSkinMap:
    """Face-local physical skin points with sparse work pullback maps.

    Records are keyed by ``(body_id, element_id, local_corner)`` so that a
    shared structural node keeps one distinct physical corner per incident
    element, including its own element director and offset.  The sparse
    ``coordinate_map`` (additive-rotation chart) and ``spin_map`` (spatial
    virtual spin) pull work back onto the shared authoritative structural
    DOFs, accumulating over all incident faces.
    """

    def __init__(self, model, faces):
        declarations: list[tuple[int, int, str]] = []
        for face in tuple(faces):
            if len(face) != 3:
                raise CurvedShellSkinError(
                    "invalid_request",
                    "skin faces must be (body_id, element_id, side) declarations")
            body_id, element_id, side = (int(face[0]), int(face[1]), str(face[2]))
            if (body_id, element_id) in {(b, e) for b, e, _ in declarations}:
                raise CurvedShellSkinError(
                    "invalid_request",
                    f"duplicate skin declaration for body {body_id} element {element_id}")
            declarations.append((body_id, element_id, side))
        if not declarations:
            raise CurvedShellSkinError("invalid_request", "skin needs at least one face")

        self._model = model
        manager = model.mesh.dof_manager
        self.n_dof = int(manager.total_dofs)
        face_records: list[CurvedSkinFace] = []
        keys: list[tuple[int, int, int]] = []
        references: list[np.ndarray] = []
        directors: list[np.ndarray] = []
        offsets: list[float] = []
        dofs: list[np.ndarray] = []
        for body_id, element_id, side in declarations:
            adapter = _face_adapter(model, body_id, element_id, side)
            element = model.mesh.elements[int(element_id)]
            corners = np.asarray(adapter.reference_positions, dtype=float)
            node_ids = tuple(int(node) for node in adapter.node_ids)
            for local, node_id in enumerate(node_ids):
                nodal = np.asarray(manager.get_node_dofs(node_id), dtype=np.intp)
                if nodal.size < 6:
                    raise CurvedShellSkinError(
                        "unsupported_face",
                        f"skin node {node_id} lacks six structural DOFs")
                keys.append((body_id, element_id, local))
                references.append(
                    np.asarray(model.mesh.get_node(int(node_id)).coords(),
                               dtype=float))
                directors.append(np.asarray(adapter.director, dtype=float))
                offsets.append(float(adapter.physical_offset))
                dofs.append(nodal[:6])
            face_records.append(CurvedSkinFace(
                int(body_id), int(element_id), side,
                str(getattr(element, "formulation_id", "unknown")),
                node_ids,
                np.asarray(adapter.director, dtype=float).copy(),
                float(adapter.physical_offset),
                corners.copy()))
        self.faces = tuple(face_records)
        self.corner_keys = tuple(keys)
        self.corner_nodes = tuple(
            node_id for face in face_records for node_id in face.node_ids)
        # Original nodal reference positions X taken from the model itself
        # (the zero-pose skin is X + h * d, matching the authoritative
        # adapters without reconstructing X from offset corners).
        self.reference = np.array(references, dtype=float)
        self.directors = np.array(directors, dtype=float)
        self.offsets = np.array(offsets, dtype=float)
        self.dofs = np.array(dofs, dtype=np.intp)
        for array in (self.reference, self.directors, self.offsets, self.dofs):
            array.setflags(write=False)
        self.face_slices = tuple(
            (start, start + len(face.node_ids))
            for face, start in zip(
                face_records,
                [sum(len(f.node_ids) for f in face_records[:i])
                 for i in range(len(face_records))]))
        # Bind the exact geometry this map was constructed from.
        self._bound_nodes = tuple(sorted({node for node in self.corner_nodes}))
        self._bound_coordinates = {
            int(node): np.asarray(model.mesh.get_node(int(node)).coords(),
                                  dtype=float).copy()
            for node in self._bound_nodes}
        self._bound_node_dofs = {
            int(node): np.asarray(manager.get_node_dofs(int(node))[:6],
                                  dtype=np.intp).copy()
            for node in self._bound_nodes}
        self._bound_elements = {
            int(face.element_id): tuple(
                int(node) for node in
                model.mesh.elements[int(face.element_id)].node_ids)
            for face in self.faces}
        self._bound_element_identity = {
            int(face.element_id): _element_identity(
                model.mesh.elements[int(face.element_id)])
            for face in self.faces}
        self._bound_reference_corners = tuple(
            face.reference_corners.copy() for face in self.faces)

    # -- guards -----------------------------------------------------------

    def _require_bound_geometry(self, model) -> None:
        if int(model.mesh.dof_manager.total_dofs) != self.n_dof:
            raise CurvedShellSkinError(
                "mutated_bound_geometry",
                "bound model DOF count changed after skin construction")
        for node, coordinates in self._bound_coordinates.items():
            current = np.asarray(model.mesh.get_node(int(node)).coords(), dtype=float)
            if not np.array_equal(current, coordinates):
                raise CurvedShellSkinError(
                    "mutated_bound_geometry",
                    f"bound node {node} coordinates changed after skin construction")
        for node, bound_dofs in self._bound_node_dofs.items():
            current = np.asarray(
                model.mesh.dof_manager.get_node_dofs(int(node))[:6],
                dtype=np.intp)
            if not np.array_equal(current, bound_dofs):
                raise CurvedShellSkinError(
                    "mutated_bound_geometry",
                    f"bound node {node} DOF ordering changed after skin "
                    f"construction")
        for element_id, bound_identity in self._bound_element_identity.items():
            element = model.mesh.elements.get(int(element_id))
            if element is None or not _same_element_identity(
                    _element_identity(element), bound_identity):
                raise CurvedShellSkinError(
                    "mutated_bound_geometry",
                    f"bound element {element_id} type, director or physical "
                    f"offset changed after skin construction")
        for element_id, node_ids in self._bound_elements.items():
            element = model.mesh.elements.get(int(element_id))
            if element is None or tuple(int(n) for n in element.node_ids) != node_ids:
                raise CurvedShellSkinError(
                    "mutated_bound_geometry",
                    f"bound element {element_id} connectivity changed after "
                    "skin construction")

    def _require_face_orientation(self, positions: np.ndarray) -> None:
        for face, (start, end) in zip(self.faces, self.face_slices):
            reference = face.reference_corners
            normal = np.cross(reference[1] - reference[0], reference[-1] - reference[0])
            current = positions[start:end]
            for index in range(len(current)):
                cross = np.cross(current[(index + 1) % len(current)] - current[index],
                                 current[(index - 1) % len(current)] - current[index])
                if not np.all(np.isfinite(cross)) or not float(cross @ normal) > 0.:
                    raise CurvedShellSkinError(
                        "invalid_geometry",
                        f"skin face (body {face.body_id}, element "
                        f"{face.element_id}) is degenerate or inverted")

    # -- pose and work maps -------------------------------------------------

    def _require_owned_pose(self, pose: FiniteSkinPose) -> None:
        """Refuse poses produced by another map or body (mismatched identity).

        A saved pose must also refuse stale bound geometry or DOF mapping
        mutated after that pose was created, so every consumer re-checks the
        bound model state here.
        """
        if getattr(pose, "_curved_skin_map", None) is not self:
            raise CurvedShellSkinError(
                "invalid_request",
                "pose was not produced by this curved skin map")
        self._require_bound_geometry(self._model)

    def pose(self, displacement) -> FiniteSkinPose:
        """Physical face-local skin pose with both work maps and errors."""
        self._require_bound_geometry(self._model)
        if not isinstance(displacement, np.ndarray) or displacement.dtype.kind not in "iuf":
            raise CurvedShellSkinError(
                "invalid_displacement", "finite real displacement vector required")
        if displacement.ndim != 1:
            raise CurvedShellSkinError(
                "invalid_displacement",
                "skin displacement must be a one-dimensional DOF vector")
        q = np.asarray(displacement, dtype=float)
        if q.size != self.n_dof or not np.isfinite(q).all():
            raise CurvedShellSkinError(
                "invalid_displacement",
                "skin displacement has invalid size or values")
        nodal = q[self.dofs]
        rotation_norm = np.linalg.norm(nodal[:, 3:], axis=1)
        if float(np.max(rotation_norm)) > TOTAL_ROTATION_BOUND_RAD:
            raise CurvedShellSkinError(
                "rotation_bound_exceeded",
                f"skin nodal rotation {float(np.max(rotation_norm))!r} rad "
                f"exceeds the existing {TOTAL_ROTATION_BOUND_RAD} rad bound")
        d, dg, _ = director_jets(nodal[:, 3:], self.directors)
        positions = self.reference + nodal[:, :3] + self.offsets[:, None] * d
        self._require_face_orientation(positions)
        rows: list[int] = []
        cols: list[int] = []
        chart_values: list[float] = []
        spin_values: list[float] = []
        for i, dofs in enumerate(self.dofs):
            spin = -self.offsets[i] * _skew(d[i])
            chart = self.offsets[i] * dg[i]
            for a in range(3):
                for b in range(6):
                    rows.append(3 * i + a)
                    cols.append(int(dofs[b]))
                    chart_values.append(float(a == b) if b < 3 else chart[a, b - 3])
                    spin_values.append(float(a == b) if b < 3 else spin[a, b - 3])
        shape = (3 * len(d), self.n_dof)
        coordinate = sparse.coo_matrix(
            (chart_values, (rows, cols)), shape=shape).tocsr()
        spin_map = sparse.coo_matrix(
            (spin_values, (rows, cols)), shape=shape).tocsr()
        err = director_value_error(nodal[:, 3:], self.directors)
        magnitude = _up(_up(np.abs(self.reference) + np.abs(nodal[:, :3]))
                        + _up(np.abs(self.offsets[:, None] * d)))
        gamma = np.nextafter(3 * _EPS / (1. - 3 * _EPS), np.inf)
        errors = _up(_up(np.abs(self.offsets[:, None]) * err[:, None])
                     + _up(gamma * magnitude))
        result = FiniteSkinPose(positions, d, dg, coordinate, spin_map, errors)
        object.__setattr__(result, "_curved_skin_map", self)
        return result

    # -- face interpolation (geometry only) ---------------------------------

    def face_shape_weights(self, face_index: int, local_coords) -> np.ndarray:
        """Original S3 barycentric / Q4 bilinear corner weights.

        ``local_coords`` are the original element natural coordinates: three
        barycentric weights, each in ``[0, 1]`` and summing to one, for a
        triangle (no extrapolation), and ``(xi, eta)`` in ``[-1, 1]`` for a
        bilinear quadrilateral.  The returned weights follow the face's
        ``node_ids`` order.  This is geometry-only interpolation of the
        admitted faces, not a contact law or projection.
        """
        index = int(face_index)
        if not 0 <= index < len(self.faces):
            raise CurvedShellSkinError(
                "invalid_request", "skin face index out of range")
        face = self.faces[index]
        coords = np.asarray(local_coords, dtype=float)
        count = len(face.node_ids)
        if count == 3:
            if coords.shape != (3,) or not np.isfinite(coords).all():
                raise CurvedShellSkinError(
                    "invalid_request",
                    "S3 face coordinates are three finite barycentric weights")
            if abs(float(coords.sum()) - 1.0) > 1.0e-12:
                raise CurvedShellSkinError(
                    "invalid_request", "S3 barycentric weights must sum to one")
            if bool(np.any(coords < 0.0)) or bool(np.any(coords > 1.0)):
                raise CurvedShellSkinError(
                    "invalid_request",
                    "S3 barycentric weights must each lie in [0, 1] "
                    "(no extrapolation)")
            return coords
        if count == 4:
            if (coords.shape != (2,) or not np.isfinite(coords).all()
                    or bool(np.any(np.abs(coords) > 1.0))):
                raise CurvedShellSkinError(
                    "invalid_request",
                    "Q4 face coordinates are finite (xi, eta) in [-1, 1]")
            xi, eta = (float(coords[0]), float(coords[1]))
            return 0.25 * np.array([
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ])
        raise CurvedShellSkinError(
            "unsupported_face",
            f"skin face {index} has no admitted interpolation")

    def face_point(self, pose: FiniteSkinPose, face_index: int,
                   local_coords) -> np.ndarray:
        """Interpolated physical point on one face's current skin."""
        self._require_owned_pose(pose)
        weights = self.face_shape_weights(face_index, local_coords)
        start, end = self.face_slices[int(face_index)]
        return weights @ pose.positions[start:end]

    def face_normal(self, pose: FiniteSkinPose, face_index: int,
                    local_coords) -> np.ndarray:
        """Current geometric normal of one face at the natural coordinates.

        This is the normal of the interpolated current skin geometry (the
        original linear/bilinear face), not the rotated element director:
        on a warped or deformed quadrilateral it varies over the face.
        """
        self._require_owned_pose(pose)
        self.face_shape_weights(face_index, local_coords)
        return self._face_normal(pose, face_index, local_coords)

    def face_centre_normals(self, pose: FiniteSkinPose) -> np.ndarray:
        """All face-centre normals with model/pose validation around the batch.

        No cached admission survives the call. Normal arithmetic is identical
        to ``face_normal``; bound geometry is checked again before returning.
        """
        self._require_owned_pose(pose)
        result = np.asarray([
            self._face_normal(pose, i, np.full(3, 1.0 / 3.0)
                              if len(face.node_ids) == 3 else np.zeros(2))
            for i, face in enumerate(self.faces)
        ])
        self._require_owned_pose(pose)
        return result

    def _face_normal(self, pose, face_index, local_coords):
        """Normal arithmetic after the public caller has validated ownership."""
        index = int(face_index)
        face = self.faces[index]
        start, end = self.face_slices[index]
        corners = pose.positions[start:end]
        if len(face.node_ids) == 3:
            normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        else:
            xi, eta = np.asarray(local_coords, dtype=float)
            d_xi = 0.25 * np.array(
                [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
            d_eta = 0.25 * np.array(
                [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])
            normal = np.cross(d_xi @ corners, d_eta @ corners)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 0.0:
            raise CurvedShellSkinError(
                "invalid_geometry",
                f"skin face (body {face.body_id}, element {face.element_id}) "
                f"has a degenerate normal at the requested coordinates")
        return normal / norm

    def face_force_pullback(self, pose: FiniteSkinPose, face_index: int,
                            local_coords, force) -> np.ndarray:
        """Pull a force applied at one interpolated face point to the DOFs.

        The point force is distributed over the face's corner rows with the
        original shape-function weights and mapped through the same sparse
        spin rows as corner forces; this is geometry-only work mapping, not
        a contact law.
        """
        self._require_owned_pose(pose)
        weights = self.face_shape_weights(face_index, local_coords)
        f = np.asarray(force, dtype=float)
        if f.shape != (3,) or not np.isfinite(f).all():
            raise CurvedShellSkinError(
                "invalid_request", "skin point force needs a finite (3,) shape")
        start, end = self.face_slices[int(face_index)]
        forces = np.zeros((len(self.dofs), 3))
        forces[start:end] = weights[:, None] * f
        return self.spin_pullback(pose, forces)

    def spin_pullback(self, pose: FiniteSkinPose, forces: np.ndarray) -> np.ndarray:
        """Pull face-local physical forces to authoritative DOFs (spatial spin).

        The transpose of the spatial spin map accumulates, at each shared
        structural node, the corner forces on the translation DOFs and the
        moments ``h * (d_element x f)`` on the rotation DOFs.
        """
        self._require_owned_pose(pose)
        f = np.asarray(forces, dtype=float)
        if f.shape != (len(self.dofs), 3) or not np.isfinite(f).all():
            raise CurvedShellSkinError(
                "invalid_request", "skin forces need finite (n_corners, 3) shape")
        return np.asarray(pose.spin_map.T @ f.ravel()).reshape(-1)

    def coordinate_pullback(self, pose: FiniteSkinPose, forces: np.ndarray) -> np.ndarray:
        """Pull face-local physical forces back through the additive chart map."""
        self._require_owned_pose(pose)
        f = np.asarray(forces, dtype=float)
        if f.shape != (len(self.dofs), 3) or not np.isfinite(f).all():
            raise CurvedShellSkinError(
                "invalid_request", "skin forces need finite (n_corners, 3) shape")
        return np.asarray(pose.coordinate_map.T @ f.ravel()).reshape(-1)

    def spin_force_derivative(self, pose: FiniteSkinPose, local_force: np.ndarray):
        """Derivative of the fixed-spatial-force spin pullback in q charts.

        Same contract as the existing finite skin owner: the force at each
        face-local corner is held fixed in space, and the derivative of
        ``h * d(theta) x f`` with respect to the additive rotation coordinates
        is accumulated over all corners sharing a node's rotation DOFs.
        """
        self._require_owned_pose(pose)
        force = np.asarray(local_force, dtype=float)
        if force.shape != (len(self.dofs), 3) or not np.isfinite(force).all():
            raise CurvedShellSkinError(
                "invalid_request", "skin forces need finite (n_corners, 3) shape")
        result = sparse.lil_matrix((self.n_dof, self.n_dof))
        for i, dofs in enumerate(self.dofs):
            block = -self.offsets[i] * _skew(force[i]) @ pose.director_gradient[i]
            result[np.ix_(dofs[3:], dofs[3:])] += block
        return result.tocsr()


__all__ = [
    "CurvedShellSkinError",
    "CurvedSkinFace",
    "CurvedShellSkinMap",
    "TOTAL_ROTATION_BOUND_RAD",
]
