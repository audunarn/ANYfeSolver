"""Reference geometry for planar Q4/S3 projected shell overlap.

This module prepares a common polygon for two *physical* faces.  It never
triangulates a Q4 for interpolation or turns clipping diagonals into contact
features.  It is an internal C1 geometry kernel, not a contact law or an
admitted paired-shell solver profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanarFace:
    feature_id: tuple[int, int]
    corners: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.corners, dtype=float)
        if points.shape not in {(3, 3), (4, 3)} or not np.all(np.isfinite(points)):
            raise ValueError("a planar contact face needs three or four finite 3D corners")
        points = points.copy()
        scale = float(np.max(np.linalg.norm(points - points[0], axis=1)))
        if scale <= 0.0:
            raise ValueError("contact face is degenerate")
        normal = np.cross(points[1] - points[0], points[-1] - points[0])
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 1.0e-12 * scale * scale:
            raise ValueError("contact face is degenerate")
        normal /= magnitude
        if float(np.max(np.abs((points - points[0]) @ normal))) > 1.0e-10 * scale:
            raise ValueError("this overlap kernel requires a planar face")
        edge_turns = [
            float(np.cross(points[(i + 1) % len(points)] - points[i],
                           points[(i + 2) % len(points)] - points[(i + 1) % len(points)])
                  @ normal)
            for i in range(len(points))
        ]
        if min(edge_turns) <= 1.0e-12 * scale * scale:
            raise ValueError("contact face must be strictly convex and consistently ordered")
        points.setflags(write=False)
        object.__setattr__(self, "corners", points)


@dataclass(frozen=True)
class ProjectedOverlap:
    """Convex common polygon in the integration face's reference plane."""

    integration_feature: tuple[int, int]
    target_feature: tuple[int, int]
    polygon: np.ndarray
    area: float
    origin: np.ndarray
    tangent: np.ndarray
    binormal: np.ndarray
    normal: np.ndarray

    def shape_values(self, face: PlanarFace, point: np.ndarray) -> np.ndarray:
        """Interpolate on the original Q4 or S3 from a projected point."""

        return self.shape_values_and_gradient(face, point)[0]

    def prepared_shape_values(self, face: PlanarFace):
        """Reuse one face's projected corners during overlap quadrature."""

        xy = _to_frame(face.corners, self.origin, self.tangent, self.binormal)

        def values(point: np.ndarray) -> np.ndarray:
            target = _to_frame(np.asarray(point, dtype=float).reshape(1, 3),
                               self.origin, self.tangent, self.binormal)[0]
            shape, _derivatives = self._shape_from_xy(xy, target)
            return shape

        return values

    def shape_values_and_gradient(self, face: PlanarFace,
                                  point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return physical shape values and their projected spatial gradients."""

        shape, derivatives, xy = self._shape_data(face, point)
        jacobian = derivatives @ xy
        gradient_xy = np.linalg.solve(jacobian, derivatives).T
        gradient = (gradient_xy[:, 0, None] * self.tangent
                    + gradient_xy[:, 1, None] * self.binormal)
        return shape, gradient

    def _shape_data(self, face: PlanarFace,
                    point: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Invert the projected interpolation and retain its original guards."""

        xy = _to_frame(face.corners, self.origin, self.tangent, self.binormal)
        target = _to_frame(np.asarray(point, dtype=float).reshape(1, 3),
                           self.origin, self.tangent, self.binormal)[0]
        shape, derivatives = self._shape_from_xy(xy, target)
        return shape, derivatives, xy

    @staticmethod
    def _shape_from_xy(xy: np.ndarray,
                       target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(xy) == 3:
            edge_matrix = np.column_stack((xy[1] - xy[0], xy[2] - xy[0]))
            bary = np.linalg.solve(edge_matrix, target - xy[0])
            shape = np.array((1.0 - sum(bary), bary[0], bary[1]), dtype=float)
            derivatives = np.array(((-1.0, 1.0, 0.0),
                                    (-1.0, 0.0, 1.0)))
        else:
            natural = np.zeros(2, dtype=float)
            scale = float(np.max(np.linalg.norm(xy - xy[0], axis=1)))
            for _ in range(30):
                shape, derivatives = _q4_shape(natural)
                residual = shape @ xy - target
                if float(np.linalg.norm(residual)) <= 1.0e-12 * scale:
                    break
                jacobian = derivatives @ xy
                natural -= np.linalg.solve(jacobian.T, residual)
            else:
                raise ValueError("Q4 projected point did not invert")
            shape, derivatives = _q4_shape(natural)
        if (not np.all(np.isfinite(shape))
                or float(np.min(shape)) < -1.0e-8
                or abs(float(np.sum(shape)) - 1.0) > 1.0e-9):
            raise ValueError("projected point is outside the physical contact face")
        return shape, derivatives


def _q4_shape(natural: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, s = map(float, natural)
    shape = 0.25 * np.array((
        (1 - r) * (1 - s), (1 + r) * (1 - s),
        (1 + r) * (1 + s), (1 - r) * (1 + s),
    ))
    derivatives = 0.25 * np.array((
        (-(1 - s), (1 - s), (1 + s), -(1 + s)),
        (-(1 - r), -(1 + r), (1 + r), (1 - r)),
    ))
    return shape, derivatives


def _to_frame(points: np.ndarray, origin: np.ndarray,
              tangent: np.ndarray, binormal: np.ndarray) -> np.ndarray:
    relative = points - origin
    return np.column_stack((relative @ tangent, relative @ binormal))


def _signed_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    # The overlap can be a tiny triangle far from the frame origin. A global
    # shoelace sum subtracts O(face_size**2) terms to recover its small area.
    # Fan triangles about one polygon vertex retain the local length scale.
    edges = polygon[1:] - polygon[0]
    return 0.5 * float(np.sum(edges[:-1, 0]*edges[1:, 1]
                              - edges[:-1, 1]*edges[1:, 0]))


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _clip_convex(subject: np.ndarray, clip: np.ndarray,
                 area_tolerance: float, length_tolerance: float) -> np.ndarray:
    polygon = [point.copy() for point in subject]
    for index, edge_start in enumerate(clip):
        if not polygon:
            break
        edge_end = clip[(index + 1) % len(clip)]
        edge = edge_end - edge_start
        previous = polygon[-1]
        previous_distance = _cross2(edge, previous - edge_start)
        output: list[np.ndarray] = []
        for current in polygon:
            current_distance = _cross2(edge, current - edge_start)
            previous_inside = previous_distance >= -area_tolerance
            current_inside = current_distance >= -area_tolerance
            if previous_inside != current_inside:
                denominator = previous_distance - current_distance
                if abs(denominator) > 0.0:
                    fraction = min(1.0, max(0.0, previous_distance / denominator))
                    output.append(previous + fraction * (current - previous))
            if current_inside:
                output.append(current)
            previous, previous_distance = current, current_distance
        polygon = output
    if not polygon:
        return np.empty((0, 2), dtype=float)
    unique = [polygon[0]]
    for point in polygon[1:]:
        if float(np.linalg.norm(point - unique[-1])) > length_tolerance:
            unique.append(point)
    if len(unique) > 1 and float(np.linalg.norm(unique[-1] - unique[0])) <= length_tolerance:
        unique.pop()
    return np.asarray(unique, dtype=float).reshape(-1, 2)


def planar_projected_overlap(integration: PlanarFace, target: PlanarFace,
                             *, relative_tolerance: float = 1.0e-12,
                             exact_clipping: bool = False
                             ) -> ProjectedOverlap | None:
    """Return the projected common polygon for two planar convex faces.

    Physical normal gap and contact-side ownership are decided by the calling
    formulation. An empty or zero-area overlap returns ``None``. Internal
    ``exact_clipping`` uses zero clipping tolerances in floating-point arithmetic;
    it neither pads the inside region nor merges distinct nearby vertices.
    """

    if type(exact_clipping) is not bool:
        raise ValueError("exact_clipping must be boolean")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    corners = integration.corners
    normal = np.cross(corners[1] - corners[0], corners[-1] - corners[0])
    normal /= np.linalg.norm(normal)
    tangent = corners[1] - corners[0]
    tangent /= np.linalg.norm(tangent)
    binormal = np.cross(normal, tangent)
    origin = corners[0]
    integration_xy = _to_frame(corners, origin, tangent, binormal)
    target_xy = _to_frame(target.corners, origin, tangent, binormal)
    if _signed_area(target_xy) < 0.0:
        target_xy = target_xy[::-1].copy()
    span = max(
        float(np.max(np.linalg.norm(integration_xy - integration_xy[0], axis=1))),
        float(np.max(np.linalg.norm(target_xy - target_xy[0], axis=1))),
    )
    length_tolerance = relative_tolerance * span
    clipped = _clip_convex(integration_xy, target_xy,
                           0.0 if exact_clipping else length_tolerance * span,
                           0.0 if exact_clipping else length_tolerance)
    area = abs(_signed_area(clipped))
    if len(clipped) < 3 or area <= length_tolerance * length_tolerance:
        return None
    polygon = origin + clipped[:, 0, None] * tangent + clipped[:, 1, None] * binormal
    return ProjectedOverlap(
        integration.feature_id, target.feature_id, polygon, area,
        origin.copy(), tangent, binormal, normal,
    )


def planar_overlap_translation_gradient(integration: PlanarFace,
                                        target: PlanarFace) -> np.ndarray:
    """Differentiate overlap area for a tangential translation of one face.

    This is the smooth-branch polygon boundary derivative, expressed in global
    coordinates. At zero area or a topology transition the derivative is not
    unique; this routine returns zero at zero area. It is an internal geometry
    ingredient, not a complete contact residual or tangent.
    """

    return planar_overlap_translation_derivatives(integration, target)[0]


def planar_overlap_translation_derivatives(integration: PlanarFace,
                                           target: PlanarFace
                                           ) -> tuple[np.ndarray, np.ndarray]:
    """Return area gradient and Hessian for a smooth tangential translation.

    Branch ownership is fixed while differentiating each clipped edge
    interval. A caller must use one-sided checks at overlap-topology changes.
    """

    overlap = planar_projected_overlap(integration, target)
    if overlap is None:
        return np.zeros(3, dtype=float), np.zeros((3, 3), dtype=float)
    source = _to_frame(integration.corners, overlap.origin,
                       overlap.tangent, overlap.binormal)
    clip = _to_frame(target.corners, overlap.origin,
                     overlap.tangent, overlap.binormal)
    if _signed_area(clip) < 0.0:
        clip = clip[::-1].copy()
    span = max(float(np.max(np.linalg.norm(source - source[0], axis=1))),
               float(np.max(np.linalg.norm(clip - clip[0], axis=1))))
    area_tolerance = 1.0e-12 * span * span
    gradient = np.zeros(2, dtype=float)
    hessian = np.zeros((2, 2), dtype=float)
    for index, start in enumerate(source):
        edge = source[(index + 1) % len(source)] - start
        lower, upper = 0.0, 1.0
        lower_derivative = np.zeros(2, dtype=float)
        upper_derivative = np.zeros(2, dtype=float)
        for clip_index, clip_start in enumerate(clip):
            clip_edge = clip[(clip_index + 1) % len(clip)] - clip_start
            distance = _cross2(clip_edge, start - clip_start)
            slope = _cross2(clip_edge, edge)
            if abs(slope) <= area_tolerance:
                if distance < -area_tolerance:
                    lower, upper = 1.0, 0.0
                    break
                continue
            crossing = -distance / slope
            crossing_derivative = np.array((clip_edge[1], -clip_edge[0])) / slope
            if slope > 0.0:
                if crossing > lower:
                    lower = crossing
                    lower_derivative = crossing_derivative
            else:
                if crossing < upper:
                    upper = crossing
                    upper_derivative = crossing_derivative
            if lower >= upper:
                break
        if upper > lower:
            outward = np.array((edge[1], -edge[0]))
            gradient += (upper - lower) * outward
            hessian += np.outer(outward, upper_derivative - lower_derivative)
    basis = np.column_stack((overlap.tangent, overlap.binormal))
    return basis @ gradient, basis @ hessian @ basis.T


def swept_planar_overlap_interval(
        source: PlanarFace, target: PlanarFace,
        source_start: np.ndarray, source_end: np.ndarray,
        target_start: np.ndarray, target_end: np.ndarray,
        ) -> tuple[float, float] | None:
    """Bound every projected overlap time along two linear translations.

    Convex separating-axis inequalities are affine along this path. The
    closed interval includes edge-touch events; callers decide whether its
    interior has positive area and how to subdivide the mechanical step.
    """

    motions = tuple(np.asarray(value, dtype=float) for value in
                    (source_start, source_end, target_start, target_end))
    if any(value.shape != (3,) or not np.all(np.isfinite(value))
           for value in motions):
        raise ValueError("swept face translations must be finite 3-vectors")
    normal = np.cross(source.corners[1] - source.corners[0],
                      source.corners[-1] - source.corners[0])
    normal /= np.linalg.norm(normal)
    target_normal = np.cross(target.corners[1] - target.corners[0],
                             target.corners[-1] - target.corners[0])
    target_normal /= np.linalg.norm(target_normal)
    if abs(float(normal @ target_normal)) < 1.0 - 1.0e-10:
        raise ValueError("swept planar faces must be parallel")
    tangent = source.corners[1] - source.corners[0]
    tangent /= np.linalg.norm(tangent)
    binormal = np.cross(normal, tangent)
    origin = source.corners[0]
    source_xy = _to_frame(source.corners + motions[0], origin, tangent, binormal)
    target_xy = _to_frame(target.corners + motions[2], origin, tangent, binormal)
    relative_velocity = (motions[1] - motions[0]) - (motions[3] - motions[2])
    velocity_xy = np.array((relative_velocity @ tangent,
                            relative_velocity @ binormal))
    lower, upper = 0.0, 1.0
    for polygon in (source_xy, target_xy):
        for index, start in enumerate(polygon):
            edge = polygon[(index + 1) % len(polygon)] - start
            axis = np.array((edge[1], -edge[0]))
            first = source_xy @ axis
            second = target_xy @ axis
            motion = float(velocity_xy @ axis)
            inequalities = (
                (float(np.max(first) - np.min(second)), motion),
                (float(np.max(second) - np.min(first)), -motion),
            )
            for value, slope in inequalities:
                if abs(slope) <= np.finfo(float).eps * max(1.0, abs(value)):
                    if value < 0.0:
                        return None
                    continue
                crossing = -value / slope
                if slope > 0.0:
                    lower = max(lower, crossing)
                else:
                    upper = min(upper, crossing)
                if lower > upper:
                    return None
    return max(0.0, lower), min(1.0, upper)


def reference_shell_face(model: object, body_id: int, element_id: int,
                         side: str) -> PlanarFace:
    """Read the qualified formulation's physical reference skin.

    S3 V2D's mesh is a reference surface that can differ from its material
    midsurface.  The formulation owns that offset and its director polarity.
    """

    from anyfesolver.e4_pl_element import FORMULATION_ID as q4_id
    from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id

    coordinate = {"bottom": -1.0, "midsurface": 0.0, "top": 1.0}.get(side)
    if coordinate is None:
        raise ValueError("shell side must be bottom, midsurface, or top")
    element = model.mesh.elements.get(element_id)
    if element is None:
        raise ValueError(f"missing contact element {element_id}")
    formulation = getattr(element, "formulation_id", None)
    if formulation not in {q4_id, s3_id}:
        raise ValueError(f"element {element_id} has no admitted shell contact adapter")
    reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
    director = element.physical_reference_director
    if director is None:
        director = np.cross(reference[1] - reference[0],
                            reference[-1] - reference[0])
        director /= np.linalg.norm(director)
    director = np.asarray(director, dtype=float)
    if director.shape != (3,) or not np.all(np.isfinite(director)):
        raise ValueError(f"element {element_id} has invalid physical director")
    if formulation == s3_id:
        offset = float(element.material_surface_offset_from_reference(coordinate))
    else:
        offset = 0.5 * float(element.thickness) * coordinate
    return PlanarFace((int(body_id), int(element_id)), reference + offset * director)


def current_planar_shell_face(model: object, body_id: int, element_id: int,
                              side: str, displacement: np.ndarray) -> PlanarFace:
    """Map a trial small-rotation shell skin to its current planar polygon.

    This internal geometry adapter keeps the original Q4/S3 interpolation
    identity. A warped current Q4 is rejected by ``PlanarFace`` until the
    separate varying-frame contact route is admitted.
    """

    from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id

    reference_face = reference_shell_face(model, body_id, element_id, side)
    element = model.mesh.elements[element_id]
    u = np.asarray(displacement, dtype=float).reshape(-1)
    if u.size != model.mesh.dof_manager.total_dofs or not np.all(np.isfinite(u)):
        raise ValueError("contact trial displacement has invalid size or nonfinite values")
    dofs = np.asarray(
        [model.mesh.dof_manager.get_node_dofs(int(node_id))[:6]
         for node_id in element.node_ids], dtype=np.intp,
    )
    director = element.physical_reference_director
    if director is None:
        reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
        director = np.cross(reference[1] - reference[0],
                            reference[-1] - reference[0])
        director /= np.linalg.norm(director)
    coordinate = {"bottom": -1.0, "midsurface": 0.0, "top": 1.0}[side]
    if element.formulation_id == s3_id:
        offset = float(element.material_surface_offset_from_reference(coordinate))
    else:
        offset = 0.5 * coordinate * float(element.thickness)
    current = (reference_face.corners + u[dofs[:, :3]]
               + offset * np.cross(u[dofs[:, 3:]], director))
    return PlanarFace(reference_face.feature_id, current)
