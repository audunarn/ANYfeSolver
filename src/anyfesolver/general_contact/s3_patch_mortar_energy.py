"""Bounded moving S3 patch mortar penalty with shared material-node pressure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moving import _has_shared_edge_transition
from .overlap import PlanarFace, planar_projected_overlap
from .q4_source_mortar_energy import MovingQ4PrimalMortarPenalty
from .q4_source_overlap import projected_q4_source_face
from .triangle_mortar_energy import MovingS3PrimalMortarPenalty


@dataclass(frozen=True)
class S3PatchFace:
    node_ids: tuple[int, int, int]
    reference_positions: np.ndarray

    def __post_init__(self):
        nodes = tuple(self.node_ids)
        if (len(nodes) not in {3, 4} or len(set(nodes)) != len(nodes)
                or any(type(node) is not int or node <= 0 for node in nodes)):
            raise ValueError("shell patch face needs three or four distinct material nodes")
        face = PlanarFace((1, 1), self.reference_positions)
        if len(face.corners) != len(nodes):
            raise ValueError("shell patch face topology and geometry disagree")
        object.__setattr__(self, "node_ids", nodes)
        object.__setattr__(self, "reference_positions", face.corners)


@dataclass(frozen=True)
class MovingS3PatchMortarState:
    energy: float
    reference_support: np.ndarray
    reference_weighted_gap: np.ndarray
    normalized_gap: np.ndarray
    pressure: np.ndarray
    pressure_gram: np.ndarray
    overlap_area: float
    reference_overlap_area: float
    minimum_physical_gap: float | None
    overlap_signature: tuple[int, ...]
    overlapping_pairs: int
    weighted_gap_integration_error: float = 0.0
    integration_triangles: int = 0
    area_integration_error: float = 0.0
    # Disposable same-pose float evaluations. These remain independent of
    # the Jet2 integration used to check the differentiated work values.
    pair_states: tuple[object, ...] | None = None

    @property
    def active_pressure_dofs(self):
        return int(np.count_nonzero(self.pressure))


@dataclass(frozen=True)
class MovingS3PatchMortarWork:
    state: MovingS3PatchMortarState
    residual: np.ndarray
    tangent: np.ndarray | None
    energy_evaluations: int
    tie_resolution_relative_error: float | None = None


class MovingS3PatchPrimalMortarPenalty:
    """Aggregate pair integrals before normalizing shared source P1 gaps."""

    def __init__(self, source_faces: tuple[S3PatchFace, ...],
                 target_faces: tuple[S3PatchFace, ...], stiffness: float,
                 *, difference_step: float,
                 maximum_energy_evaluations: int = 1500,
                 q4_weighted_gap_tolerance: float | None = None,
                 maximum_q4_integration_triangles: int | None = None,
                 q4_source_area_tolerance: float | None = None,
                 q4_source_weighted_gap_tolerance: float | None = None,
                 maximum_q4_source_integration_triangles: int | None = None,
                 q4_source_derivative_tolerances: tuple[float | None, ...] | None = None,
                 differentiation: str = "global",
                 maximum_candidate_pairs: int = 4,
                 pair_broadphase: bool = True,
                 _full_face_monolithic_s3: bool = False):
        self.source_faces = tuple(source_faces)
        self.target_faces = tuple(target_faces)
        if differentiation not in {"global", "pair_local", "jet2_s3", "jet2_q4_s3", "jet2_s3_q4", "jet2_q4_q4"}:
            raise ValueError("S3 patch differentiation mode is invalid")
        self.differentiation = differentiation
        if type(pair_broadphase) is not bool:
            raise ValueError("patch pair broad phase selection must be boolean")
        self.pair_broadphase = pair_broadphase
        if (type(maximum_candidate_pairs) is not int or maximum_candidate_pairs < 1
                or len(self.source_faces)*len(self.target_faces) > maximum_candidate_pairs):
            raise MemoryError("S3 patch exceeds declared candidate-pair budget")
        q4_pair = (len(self.source_faces) == len(self.target_faces) == 1
                   and len(self.source_faces[0].node_ids) == 4
                   and len(self.target_faces[0].node_ids) == 4)
        bounded = (1 <= len(self.source_faces) <= 2
                   and 1 <= len(self.target_faces) <= 2
                   and (len(self.source_faces)+len(self.target_faces) >= 3
                        or q4_pair))
        refined_s3 = (differentiation in {"pair_local", "jet2_s3"}
                      and len(self.source_faces)+len(self.target_faces) >= 3
                      and all(len(face.node_ids) == 3
                              for face in (*self.source_faces, *self.target_faces)))
        if type(_full_face_monolithic_s3) is not bool:
            raise TypeError("full-face S3 internal admission must be boolean")
        if _full_face_monolithic_s3:
            if (differentiation != "jet2_s3"
                    or len(self.source_faces) != 1 or len(self.target_faces) != 1
                    or any(len(face.node_ids) != 3 for face in (*self.source_faces,*self.target_faces))):
                raise ValueError("internal monolithic S3 admission requires one S3 face per side")
            refined_s3=True
        refined_q4 = (differentiation in {"pair_local", "jet2_q4_s3"}
                      and len(self.source_faces) >= 2
                      and all(len(face.node_ids) == 4 for face in self.source_faces)
                      and all(len(face.node_ids) == 3 for face in self.target_faces))
        refined_s3_q4 = (differentiation == "jet2_s3_q4"
                          and len(self.source_faces) == 2 and len(self.target_faces) == 1
                          and all(len(face.node_ids) == 3 for face in self.source_faces)
                          and len(self.target_faces[0].node_ids) == 4)
        refined_q4_q4 = (differentiation == "jet2_q4_q4"
                          and len(self.source_faces) >= 1 and len(self.target_faces) >= 1
                          and len(self.source_faces)+len(self.target_faces) >= 3
                          and all(len(face.node_ids) == 4 for face in (*self.source_faces, *self.target_faces)))
        refined = refined_s3 or refined_q4 or refined_s3_q4 or refined_q4_q4
        if differentiation == "jet2_q4_q4" and not refined_q4_q4:
            raise ValueError("Q4/Q4 Jet2 needs original Q4 faces and at least three faces")
        if not (bounded or refined):
            raise ValueError("development shell patch needs three faces or one Q4/Q4 pair")
        if differentiation in {"pair_local", "jet2_s3", "jet2_q4_s3", "jet2_s3_q4", "jet2_q4_q4"} and not refined:
            raise ValueError("local differentiation needs S3 faces or Q4 source/S3 target faces")
        if differentiation == "jet2_s3" and not refined_s3:
            raise ValueError("Jet2 overlap derivatives need at least three S3 faces")
        if differentiation == "jet2_q4_s3" and not refined_q4:
            raise ValueError("Jet2 Q4/S3 overlap derivatives need a refined Q4/S3 patch")
        if differentiation in {"jet2_q4_s3", "jet2_s3_q4", "jet2_q4_q4"} and (
            q4_source_derivative_tolerances is None
            or len(q4_source_derivative_tolerances) != 4
            or any(value is None or not np.isfinite(value) or value <= 0.0
                   for value in q4_source_derivative_tolerances)
        ):
            raise ValueError("Jet2 Q4/S3 overlap derivatives need four positive derivative tolerances")
        if differentiation == "jet2_s3_q4" and not refined_s3_q4:
            raise ValueError("S3/Q4 Jet2 needs two source S3 and one original target Q4")
        if differentiation == "jet2_s3_q4" and any(value is None or value <= 0.
                for value in (q4_source_area_tolerance, q4_source_weighted_gap_tolerance,
                              maximum_q4_source_integration_triangles)):
            raise ValueError("S3/Q4 Jet2 needs explicit integration controls")
        source_q4 = all(len(face.node_ids) == 4 for face in self.source_faces)
        if not source_q4 and any(len(face.node_ids) != 3 for face in self.source_faces):
            raise ValueError("development patch pressure side needs S3 triangles or Q4 faces")
        self.source_node_ids = tuple(sorted({node for face in self.source_faces
                                             for node in face.node_ids}))
        self.target_node_ids = tuple(sorted({node for face in self.target_faces
                                             for node in face.node_ids}))
        if set(self.source_node_ids) & set(self.target_node_ids):
            raise ValueError("S3 patch sides must have disjoint material nodes")
        self.reference_source = self._unique_reference(
            self.source_faces, self.source_node_ids,
        )
        self.reference_target = self._unique_reference(
            self.target_faces, self.target_node_ids,
        )
        self.source_face_indices = tuple(np.asarray([
            self.source_node_ids.index(node) for node in face.node_ids
        ], dtype=np.intp) for face in self.source_faces)
        self.target_face_indices = tuple(np.asarray([
            self.target_node_ids.index(node) for node in face.node_ids
        ], dtype=np.intp) for face in self.target_faces)
        self.stiffness = float(stiffness)
        self.difference_step = float(difference_step)
        self.q4_source_area_tolerance = q4_source_area_tolerance
        self.q4_source_weighted_gap_tolerance = q4_source_weighted_gap_tolerance
        self.maximum_q4_source_integration_triangles = maximum_q4_source_integration_triangles
        self.q4_source_derivative_tolerances = q4_source_derivative_tolerances
        if (type(maximum_energy_evaluations) is not int
                or maximum_energy_evaluations < 1):
            raise ValueError("S3 patch energy evaluation budget must be positive")
        self.maximum_energy_evaluations = maximum_energy_evaluations
        self.n_coordinates = 3*(len(self.source_node_ids)+len(self.target_node_ids))
        self.required_hessian_evaluations = (
            1+2*self.n_coordinates**2 if differentiation == "global" else
            3*self._candidate_evaluation_count(self._candidate_pair_indices(
                self.reference_source, self.reference_target))
            if differentiation in {"jet2_s3", "jet2_q4_s3", "jet2_s3_q4", "jet2_q4_q4"} else
            sum(2+2*(3*(len(source.node_ids)+len(target.node_ids)))**2
                for source in self.source_faces for target in self.target_faces)
        )
        if self.required_hessian_evaluations > self.maximum_energy_evaluations:
            raise MemoryError(
                "S3 patch Hessian exceeds declared energy evaluation budget"
            )
        self.pair_laws = tuple(
            MovingQ4PrimalMortarPenalty(
                source.reference_positions, target.reference_positions,
                self.stiffness, difference_step=self.difference_step,
                maximum_energy_evaluations=self.maximum_energy_evaluations,
                area_tolerance=q4_source_area_tolerance,
                weighted_gap_tolerance=q4_source_weighted_gap_tolerance,
                maximum_integration_triangles=maximum_q4_source_integration_triangles,
            ) if len(source.node_ids) == 4 else
            MovingS3PrimalMortarPenalty(
                source.reference_positions, target.reference_positions,
                self.stiffness, difference_step=self.difference_step,
                maximum_energy_evaluations=self.maximum_energy_evaluations,
                q4_weighted_gap_tolerance=q4_weighted_gap_tolerance,
                maximum_q4_integration_triangles=maximum_q4_integration_triangles,
            )
            for source in self.source_faces for target in self.target_faces
        )
        self._check_same_side_overlap(self.reference_source, self.source_face_indices)
        self._check_same_side_overlap(self.reference_target, self.target_face_indices)

    def _candidate_evaluation_count(self, candidate_indices: set[int] | None) -> int:
        return (len(self.source_faces)*len(self.target_faces)
                if candidate_indices is None else len(candidate_indices))

    def _check_energy_evaluation_budget(self, evaluations: int) -> None:
        if evaluations > self.maximum_energy_evaluations:
            raise MemoryError("S3 patch differentiation budget exhausted")

    def _preflight_jet2_candidates(self, candidate_indices: set[int] | None) -> None:
        if self.differentiation in {"jet2_s3", "jet2_q4_s3", "jet2_s3_q4", "jet2_q4_q4"}:
            # Reserve a base float pass, a fallback float pass and one AD
            # pass for every conservative candidate. Tiny-wedge verification
            # consumes the remaining budget under its own runtime guard.
            self._check_energy_evaluation_budget(
                3*self._candidate_evaluation_count(candidate_indices))

    @staticmethod
    def _unique_reference(faces: tuple[S3PatchFace, ...],
                          node_ids: tuple[int, ...]) -> np.ndarray:
        positions: dict[int, np.ndarray] = {}
        scale = max(1.0, *(float(np.max(np.abs(face.reference_positions)))
                           for face in faces))
        for face in faces:
            for node, position in zip(face.node_ids, face.reference_positions):
                if node in positions and (float(np.linalg.norm(positions[node]-position))
                                          > 1e-12*scale):
                    raise ValueError("shared S3 patch node has inconsistent physical skin position")
                positions[node] = position
        return np.asarray([positions[node] for node in node_ids], dtype=float)

    @staticmethod
    def _check_same_side_overlap(positions: np.ndarray,
                                 indices: tuple[np.ndarray, ...]) -> None:
        scale = max(1.0, float(np.max(np.abs(positions))))
        checked = []
        for face_index, nodes in enumerate(indices):
            face_positions = positions[nodes]
            # Validate every face, including those later excluded by the
            # conservative projected bound.
            if len(face_positions) == 4:
                face, normal = projected_q4_source_face(face_positions)
            else:
                face = PlanarFace((1, face_index+1), face_positions)
                normal = np.cross(face_positions[1]-face_positions[0],
                                  face_positions[2]-face_positions[0])
                normal /= np.linalg.norm(normal)
            checked.append((face, normal))
        if len(indices) < 2:
            return
        margin = 64.0*np.finfo(float).eps*scale
        # A projected overlap can occur at any separation along the first
        # face normal. Query a conservative world-space box around that whole
        # normal prism, then retain the original exact projected check.
        from .contact_search import ContactAABBIndex

        face_boxes = np.asarray([
            np.r_[np.min(positions[nodes], axis=0),
                  np.max(positions[nodes], axis=0)]
            for nodes in indices
        ])
        search = ContactAABBIndex(
            tuple(("same_side_face", i) for i in range(len(indices))),
            face_boxes,
        )
        scene_min = np.min(positions, axis=0)
        scene_max = np.max(positions, axis=0)
        scene_corners = np.asarray([
            (x, y, z) for x in (scene_min[0], scene_max[0])
            for y in (scene_min[1], scene_max[1])
            for z in (scene_min[2], scene_max[2])
        ])
        for first_index, (first, normal) in enumerate(checked):
            tangent = first.corners[1]-first.corners[0]
            tangent /= np.linalg.norm(tangent)
            binormal = np.cross(normal, tangent)
            origin = first.corners[0]
            first_relative = positions[indices[first_index]]-origin
            first_xy = np.column_stack((first_relative@tangent,
                                        first_relative@binormal))
            first_min = np.min(first_xy, axis=0)
            first_max = np.max(first_xy, axis=0)
            normal_coordinates = (scene_corners-origin) @ normal
            normal_limits = (np.min(normal_coordinates)-margin,
                             np.max(normal_coordinates)+margin)
            prism_corners = np.asarray([
                origin + a*tangent + b*binormal + c*normal
                for a in (first_min[0]-margin, first_max[0]+margin)
                for b in (first_min[1]-margin, first_max[1]+margin)
                for c in normal_limits
            ])
            prism_box = np.r_[np.min(prism_corners, axis=0),
                              np.max(prism_corners, axis=0)]
            candidate_indices = sorted(
                key[1] for key in search.query(prism_box, padding=4*margin)
                if key[1] > first_index
            )
            for second_index in candidate_indices:
                second_positions = positions[indices[second_index]]
                second_relative = second_positions-origin
                second_xy = np.column_stack((second_relative@tangent,
                                             second_relative@binormal))
                if np.any(np.max(second_xy, axis=0) < first_min-margin) or np.any(
                        np.min(second_xy, axis=0) > first_max+margin):
                    continue
                projected = second_positions - (
                    second_relative @ normal
                )[:, None]*normal
                second = PlanarFace((1, second_index+1), projected)
                overlap = planar_projected_overlap(first, second)
                if overlap is not None and overlap.area > 1e-12*scale**2:
                    raise ValueError("S3 patch has overlapping faces on one side")

    def evaluate(self, current_source: np.ndarray,
                 current_target: np.ndarray) -> MovingS3PatchMortarState:
        source = np.asarray(current_source, dtype=float)
        target = np.asarray(current_target, dtype=float)
        if (source.shape != self.reference_source.shape
                or target.shape != self.reference_target.shape
                or not np.all(np.isfinite(source))
                or not np.all(np.isfinite(target))):
            raise ValueError("S3 patch current coordinates have invalid size or values")
        self._check_same_side_overlap(source, self.source_face_indices)
        self._check_same_side_overlap(target, self.target_face_indices)
        candidate_indices = self._candidate_pair_indices(source, target)
        self._preflight_jet2_candidates(candidate_indices)
        support = np.zeros(len(self.source_node_ids))
        weighted_gap = np.zeros_like(support)
        gram = np.zeros((len(support), len(support)))
        overlap_area = reference_overlap_area = 0.0
        minimum = None
        signature: list[int] = []
        overlapping_pairs = 0
        weighted_error = 0.0
        integration_triangles = 0
        area_error = 0.0
        pair_states = []
        evaluations = 0
        for index, law in enumerate(self.pair_laws):
            if candidate_indices is not None and index not in candidate_indices:
                pair_states.append(None)
                signature.append(0)
                continue
            source_face = index // len(self.target_faces)
            target_face = index % len(self.target_faces)
            evaluations += 1
            if self.differentiation in {"jet2_s3", "jet2_q4_s3", "jet2_s3_q4", "jet2_q4_q4"}:
                self._check_energy_evaluation_budget(evaluations)
            state = law.evaluate(source[self.source_face_indices[source_face]],
                                 target[self.target_face_indices[target_face]])
            pair_states.append(state)
            signature.append(state.overlap_vertices)
            if state.overlap_vertices == 0:
                continue
            overlapping_pairs += 1
            weighted_error += state.weighted_gap_integration_error
            integration_triangles += state.integration_triangles
            area_error += getattr(state, "area_integration_error", 0.0)
            dofs = self.source_face_indices[source_face]
            support[dofs] += state.reference_support
            weighted_gap[dofs] += state.reference_weighted_gap
            gram[np.ix_(dofs, dofs)] += state.pressure_gram
            overlap_area += state.overlap_area
            reference_overlap_area += state.reference_overlap_area
            sampled = (state.minimum_sampled_gap if len(self.source_face_indices[source_face]) == 4
                       else state.minimum_physical_gap)
            minimum = (sampled if minimum is None else min(minimum, sampled))
        normalized = np.zeros_like(support)
        supported = support > 0.0
        normalized[supported] = weighted_gap[supported]/support[supported]
        pressure = np.zeros_like(support)
        pressure[supported] = self.stiffness*np.maximum(-normalized[supported], 0.0)
        energy = 0.5*self.stiffness*float(
            support @ np.minimum(normalized, 0.0)**2
        )
        return MovingS3PatchMortarState(
            energy, support, weighted_gap, normalized, pressure, gram,
            overlap_area, reference_overlap_area, minimum,
            tuple(signature), overlapping_pairs,
            weighted_error, integration_triangles, area_error,
            tuple(pair_states),
        )

    def _candidate_pair_indices(self, source: np.ndarray,
                                target: np.ndarray) -> set[int] | None:
        """Conservatively bound target faces projected along each source normal.

        The source Q4 projection is the same four-corner plane used by the
        exact mortar kernel. Its bilinear interior stays in the convex hull
        of those corners. A target point whose projection can enter that
        hull lies in the full target-depth prism used for the tree query.
        Inclusive padded boxes retain zero-area touching branches.
        """
        if (not self.pair_broadphase
                or len(self.source_faces)*len(self.target_faces) <= 64):
            return None
        from .contact_search import ContactAABBIndex

        target_boxes = np.asarray([
            np.r_[np.min(target[nodes], axis=0),
                  np.max(target[nodes], axis=0)]
            for nodes in self.target_face_indices
        ])
        search = ContactAABBIndex(
            tuple(("target_face", i) for i in range(len(target_boxes))),
            target_boxes,
        )
        scene_min = np.min(target, axis=0)
        scene_max = np.max(target, axis=0)
        scene_corners = np.asarray([
            (x, y, z) for x in (scene_min[0], scene_max[0])
            for y in (scene_min[1], scene_max[1])
            for z in (scene_min[2], scene_max[2])
        ])
        scale = max(1.0, float(np.max(np.abs(source))),
                    float(np.max(np.abs(target))))
        margin = 256.0*np.finfo(float).eps*scale
        candidate_indices: set[int] = set()
        for source_index, nodes in enumerate(self.source_face_indices):
            positions = source[nodes]
            if len(nodes) == 4:
                face, normal = projected_q4_source_face(positions)
            else:
                face = PlanarFace((1, source_index+1), positions)
                normal = np.cross(positions[1]-positions[0],
                                  positions[2]-positions[0])
                normal /= np.linalg.norm(normal)
            tangent = face.corners[1]-face.corners[0]
            tangent /= np.linalg.norm(tangent)
            binormal = np.cross(normal, tangent)
            origin = face.corners[0]
            source_relative = face.corners-origin
            source_xy = np.column_stack((source_relative@tangent,
                                         source_relative@binormal))
            low = np.min(source_xy, axis=0)-margin
            high = np.max(source_xy, axis=0)+margin
            depth = (scene_corners-origin)@normal
            prism = np.asarray([
                origin+a*tangent+b*binormal+c*normal
                for a in (low[0], high[0])
                for b in (low[1], high[1])
                for c in (np.min(depth)-margin, np.max(depth)+margin)
            ])
            box = np.r_[np.min(prism, axis=0), np.max(prism, axis=0)]
            candidate_indices.update(
                source_index*len(self.target_faces)+key[1]
                for key in search.query(box, padding=4*margin)
            )
        return candidate_indices

    def work(self, current_source: np.ndarray, current_target: np.ndarray,
             *, tangent: bool = True) -> MovingS3PatchMortarWork:
        if self.differentiation != "global":
            raise ValueError("pair-local geometry requires projected normal work")
        coordinates = np.concatenate((np.asarray(current_source, dtype=float),
                                      np.asarray(current_target, dtype=float))).reshape(-1)
        if coordinates.size != self.n_coordinates:
            raise ValueError("S3 patch work has invalid physical coordinate count")
        source_count = 3*len(self.source_node_ids)
        def state_at(values):
            return self.evaluate(values[:source_count].reshape(-1, 3),
                                 values[source_count:].reshape(-1, 3))
        base = state_at(coordinates)
        if base.energy == 0.0:
            return MovingS3PatchMortarWork(
                base, np.zeros(self.n_coordinates),
                np.zeros((self.n_coordinates, self.n_coordinates)) if tangent else None,
                1,
            )
        for index, law in enumerate(self.pair_laws):
            source_face = index // len(self.target_faces)
            target_face = index % len(self.target_faces)
            source_positions = np.asarray(current_source)[self.source_face_indices[source_face]]
            target_positions = np.asarray(current_target)[self.target_face_indices[target_face]]
            if len(source_positions) == 4:
                first, normal = projected_q4_source_face(source_positions)
            else:
                first = PlanarFace((1, 1), source_positions)
                normal = np.cross(first.corners[1]-first.corners[0],
                                  first.corners[2]-first.corners[0])
                normal /= np.linalg.norm(normal)
            projected_target = target_positions - (
                (target_positions-first.corners[0])@normal
            )[:, None]*normal
            second = PlanarFace((2, 1), projected_target)
            overlap = planar_projected_overlap(first, second)
            if (overlap is not None
                    and _has_shared_edge_transition(first, second, overlap)):
                raise ValueError("S3 patch shared-edge derivative is nonunique")
        evaluations = 1
        cache: dict[tuple[tuple[int, int], ...], float] = {}
        def energy_at(steps):
            nonlocal evaluations
            key = tuple(sorted(steps))
            if key in cache:
                return cache[key]
            evaluations += 1
            if evaluations > self.maximum_energy_evaluations:
                raise MemoryError("S3 patch differentiation budget exhausted")
            trial = coordinates.copy()
            for coordinate, sign in steps:
                trial[coordinate] += sign*self.difference_step
            result = state_at(trial)
            # Internal clipping diagonals can switch polygon combinatorics while
            # the shared-node patch potential remains smooth. Differentiate the
            # aggregate energy; same-side overlap and true shared-edge contacts
            # are checked independently above.
            cache[key] = result.energy
            return result.energy
        positive = np.empty(self.n_coordinates)
        negative = np.empty(self.n_coordinates)
        residual = np.empty(self.n_coordinates)
        for coordinate in range(self.n_coordinates):
            positive[coordinate] = energy_at(((coordinate, 1),))
            negative[coordinate] = energy_at(((coordinate, -1),))
            residual[coordinate] = ((positive[coordinate]-negative[coordinate])
                                    /(2.0*self.difference_step))
        hessian = None
        if tangent:
            hessian = np.empty((self.n_coordinates, self.n_coordinates))
            for first in range(self.n_coordinates):
                hessian[first, first] = (
                    positive[first]-2.0*base.energy+negative[first]
                )/self.difference_step**2
                for second in range(first+1, self.n_coordinates):
                    hessian[first, second] = hessian[second, first] = (
                        energy_at(((first, 1), (second, 1)))
                        - energy_at(((first, 1), (second, -1)))
                        - energy_at(((first, -1), (second, 1)))
                        + energy_at(((first, -1), (second, -1)))
                    )/(4.0*self.difference_step**2)
        return MovingS3PatchMortarWork(base, residual, hessian, evaluations)
