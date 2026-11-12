"""Development-only multi-face S3/Q4 overlap and integrated-gap work probe.

This is prescribed-pose geometry, not an admitted contact law or solver.
Each S3 physical facet is integrated independently over the original Q4
targets. Distinct facet corners retain their own offset directors. The
1 Pa probe differentiates the reference-area integrated signed gap; its
generalized loads include moving-normal/overlap terms and are not a
predicted contact reaction or a pointwise uniform-pressure traction field.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .curved_shell_skin import CurvedShellSkinMap
from .overlap import PlanarFace, planar_projected_overlap
from .patch_broadphase import PatchAABBTree, projected_face_boxes
from .s3_q4_overlap_derivatives import _s3_q4_first_order_work


# Unchanged controls from test_general_contact_s3_q4_overlap_derivatives.
INTEGRATION_CONTROLS = dict(
    support_tolerance=1e-12, weighted_gap_tolerance=1e-13,
    support_gradient_tolerance=1e-10, weighted_gap_gradient_tolerance=1e-11,
    support_hessian_tolerance=1e-8, weighted_gap_hessian_tolerance=1e-9,
    maximum_triangles=200,
)
PROBE_PRESSURE_PA = 1.0


class CurvedContactWorkError(ValueError):
    """Invalid ownership/geometry or unsupported overlap; never a solve status."""


@dataclass(frozen=True)
class CurvedContactWork:
    pairs: tuple[dict, ...]
    coverage: tuple[dict, ...]
    empty_pairs: int
    source_coordinate_work: np.ndarray
    target_coordinate_work: np.ndarray
    source_spin_work: np.ndarray
    target_spin_work: np.ndarray
    source_corner_work: np.ndarray
    target_corner_work: np.ndarray
    integrated_gap_m3: float


def _normal(corners):
    value = np.cross(corners[1] - corners[0], corners[-1] - corners[0])
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= 0:
        raise CurvedContactWorkError("degenerate physical face")
    return value / length


def _strictly_separated(first, second):
    """Conservative separating-edge test, in the existing projection chart.

    Only a separation exceeding the binary64 operand envelope is empty.
    Touching and unresolved slivers go to the existing refusing kernel.
    This is a rejection-only broad phase, not a replacement clipper.
    """
    for polygon, other in ((first, second), (second, first)):
        for i, start in enumerate(polygon):
            edge = polygon[(i + 1) % len(polygon)] - start
            delta = other - start
            products = np.abs(edge[0] * delta[:, 1]) + np.abs(edge[1] * delta[:, 0])
            span = max(float(np.max(np.abs(polygon))), float(np.max(np.abs(other))), 1.)
            error = 64 * np.finfo(float).eps * (products + span * span)
            cross = edge[0] * delta[:, 1] - edge[1] * delta[:, 0]
            if np.all(cross < -error):
                return True
    return False


class CurvedS3Q4WorkProbe:
    """Prescribed physical top-skin S3 source and bilinear Q4 target query.

    Models and DOF spaces remain separate. Positive signed gap means
    separation: the source chart normal is opposite the outward sphere
    normal. The chart permutation also permutes corner work and identities.
    No pressure modes are joined across sphere facets in this component.
    """

    def __init__(self, source_model, source_skin, target_model, target_skin):
        if (not isinstance(source_skin, CurvedShellSkinMap)
                or not isinstance(target_skin, CurvedShellSkinMap)
                or source_skin._model is not source_model
                or target_skin._model is not target_model
                or source_model is target_model):
            raise CurvedContactWorkError("distinct models and their exact owned skin maps required")
        if (any(len(face.node_ids) != 3 or face.side != "top" for face in source_skin.faces)
                or any(len(face.node_ids) != 4 or face.side != "top" for face in target_skin.faces)):
            raise CurvedContactWorkError("probe requires S3 physical top sources and Q4 physical top targets")
        if set(f.body_id for f in source_skin.faces) & set(f.body_id for f in target_skin.faces):
            raise CurvedContactWorkError("source and target body identities must be disjoint")
        self.source_skin, self.target_skin = source_skin, target_skin

    def evaluate(self, source_pose, target_pose, *, source_faces=None):
        """Return overlaps, per-facet coverage and additive/spin work vectors.

        ``source_faces`` selects a declared diagnostic region by map indices;
        all target faces are searched. No distance/normal-gap cutoff is used.
        Unsupported derivatives propagate; no exception is converted to empty.
        """
        source, target = self.source_skin, self.target_skin
        source._require_owned_pose(source_pose)
        target._require_owned_pose(target_pose)
        chosen = tuple(range(len(source.faces))) if source_faces is None else tuple(source_faces)
        if (not chosen or any(type(i) is not int or not 0 <= i < len(source.faces) for i in chosen)
                or len(set(chosen)) != len(chosen)):
            raise CurvedContactWorkError("source region needs unique valid integer face indices")
        if (not np.isfinite(source_pose.positions).all()
                or not np.isfinite(target_pose.positions).all()):
            raise CurvedContactWorkError("nonfinite physical pose")
        source._require_face_orientation(source_pose.positions)
        target._require_face_orientation(target_pose.positions)
        sf = np.zeros_like(source_pose.positions)
        tf = np.zeros_like(target_pose.positions)
        pairs, coverage = [], []
        empty, integrated = 0, 0.
        for index in chosen:
            face = source.faces[index]
            start, end = source.face_slices[index]
            # Reverse the source chart, never the actual physical director.
            order = np.array((0, 2, 1), dtype=int)
            s = source_pose.positions[start:end][order]
            reference = face.reference_corners[order]
            normal = _normal(s)
            origin = s[0]
            tangent = (s[1] - s[0]) / np.linalg.norm(s[1] - s[0])
            binormal = np.cross(normal, tangent)
            chart = np.column_stack((tangent, binormal))
            source_xy = (s - origin) @ chart
            physical_targets = []
            for j, (a, b) in enumerate(target.face_slices):
                t = target_pose.positions[a:b]
                physical_targets.append(PlanarFace((target.faces[j].body_id,
                                                    target.faces[j].element_id),
                                                   t - ((t - origin) @ normal)[:, None] * normal))
            # Reuse the stable-index broadphase; original Q4 positions below
            # remain authoritative for the physical gap and bilinear inverse.
            boxes, t_axis, b_axis = projected_face_boxes(
                [SimpleNamespace(geometry=face) for face in physical_targets], normal)
            frame = np.column_stack((t_axis, b_axis))
            projected_source = s @ frame
            query = np.r_[projected_source.min(axis=0), projected_source.max(axis=0)]
            padding = 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(query))),
                                                     float(np.max(np.abs(boxes))))
            query[:2] -= padding
            query[2:] += padding
            candidates = PatchAABBTree(boxes).query(query)
            empty += len(target.faces) - len(candidates)
            touched, local_pairs = [], []
            support = np.zeros(3)
            weighted = np.zeros(3)
            resolution, area = 0., 0.
            for j in candidates:
                a, b = target.face_slices[j]
                t = target_pose.positions[a:b]
                if float(_normal(s) @ _normal(t)) <= 0:
                    raise CurvedContactWorkError("candidate physical normals are not opposed")
                target_xy = (t - origin) @ chart
                if _strictly_separated(source_xy, target_xy):
                    empty += 1
                    continue
                # No catch: zero/uncertain branch is not evidence of emptiness.
                result = _s3_q4_first_order_work(reference, s, t, **INTEGRATION_CONTROLS)
                # Reject duplicate projected target support using the existing
                # polygon owner, including cells that are not topological neighbours.
                for previous in touched:
                    if planar_projected_overlap(physical_targets[previous], physical_targets[j]) is not None:
                        raise CurvedContactWorkError("projected panel faces overlap; partition is ambiguous")
                touched.append(j)
                support += result.reference_support
                weighted += result.reference_weighted_gap
                area += result.reference_overlap_area
                resolution += result.overlap_area_resolution_m2
                gradient = PROBE_PRESSURE_PA * np.sum(result.weighted_gap_gradient, axis=0).reshape(7, 3)
                sf[start + order] += gradient[:3]
                tf[a:b] += gradient[3:]
                local_pairs.append(len(pairs))
                corner_gaps = []
                scalar_overlap = planar_projected_overlap(
                    PlanarFace((face.body_id, face.element_id), s), physical_targets[j])
                if scalar_overlap is None:
                    raise CurvedContactWorkError("positive derivative support has no scalar overlap")
                for point, xy in zip(s, source_xy):
                    edges = np.roll(target_xy, -1, axis=0) - target_xy
                    delta = xy - target_xy
                    distances = edges[:, 0] * delta[:, 1] - edges[:, 1] * delta[:, 0]
                    products = np.abs(edges[:, 0] * delta[:, 1]) + np.abs(edges[:, 1] * delta[:, 0])
                    span = max(1., float(np.max(np.abs(target_xy))), float(np.max(np.abs(xy))))
                    errors = 64 * np.finfo(float).eps * (products + span * span)
                    if np.all(distances >= -errors):
                        weights = scalar_overlap.shape_values(physical_targets[j], point)
                        corner_gaps.append(float((point - weights @ t) @ normal))
                pairs.append(dict(
                    source_face_index=index, target_face_index=j,
                    source_identity=[face.body_id, face.element_id],
                    target_identity=[target.faces[j].body_id, target.faces[j].element_id],
                    source_chart_corner_order=order.tolist(),
                    source_chart_nodes=[face.node_ids[k] for k in order],
                    target_chart_nodes=list(target.faces[j].node_ids),
                    reference_support_m2=result.reference_support.tolist(),
                    reference_weighted_gap_m3=result.reference_weighted_gap.tolist(),
                    normalized_gap_m=(result.reference_weighted_gap / result.reference_support).tolist(),
                    source_corner_gaps_on_target_m=corner_gaps,
                    overlap_area_m2=result.overlap_area,
                    reference_overlap_area_m2=result.reference_overlap_area,
                    overlap_area_resolution_m2=result.overlap_area_resolution_m2,
                    measured_errors=list(result.measured_errors),
                    projection_residual_m=result.projection_residual_m,
                    integration_triangles=result.integration_triangles,
                    branch_signature=list(result.branch_signature),
                    derivative_order=result.derivative_order,
                    hessian_available=False,
                    seam_events=list(result.seam_events),
                ))
            facet_area = float(np.linalg.norm(np.cross(reference[1] - reference[0], reference[2] - reference[0])) / 2)
            coverage.append(dict(
                source_face_index=index, source_identity=[face.body_id, face.element_id],
                target_face_indices=list(touched), pair_indices=local_pairs,
                reference_facet_area_m2=facet_area, reference_covered_area_m2=area,
                coverage_fraction=area / facet_area,
                support_m2=support.tolist(), weighted_gap_m3=weighted.tolist(),
                overlap_area_resolution_m2=resolution,
            ))
            integrated += float(np.sum(weighted))
        return CurvedContactWork(
            tuple(pairs), tuple(coverage), empty,
            source.coordinate_pullback(source_pose, sf), target.coordinate_pullback(target_pose, tf),
            source.spin_pullback(source_pose, sf), target.spin_pullback(target_pose, tf),
            sf, tf, integrated,
        )
