"""Internal S3 V2D physical material point and frame for friction development."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import sparse

from .s3_skin_work import S3SkinWorkAdapter


@lru_cache(maxsize=128)
def _physical_skin_frame(corner_bytes,reference_bytes):
    """Bounded immutable frame geometry; exact reference and current bytes bind checks."""
    corners=np.frombuffer(corner_bytes,dtype=np.float64).reshape(3,3)
    reference=np.frombuffer(reference_bytes,dtype=np.float64).reshape(3,3)
    first = corners[1] - corners[0]
    second = corners[2] - corners[0]
    reference_cross = np.cross(reference[1] - reference[0],
                               reference[2] - reference[0])
    current_cross = np.cross(first, second)
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    area_scale = float(np.linalg.norm(current_cross))
    reference_area_scale = float(np.linalg.norm(reference_cross))
    if (first_length == 0.0 or second_length == 0.0
            or area_scale <= 1.0e-12 * max(
                reference_area_scale, first_length * second_length,
            )):
        raise ValueError("S3 material skin is degenerate at the candidate pose")
    if float(current_cross @ reference_cross) <= 0.0:
        raise ValueError("S3 material skin is inverted at the candidate pose")
    tangent_1 = first / first_length
    normal = current_cross / area_scale
    tangent_2 = np.cross(normal, tangent_1)
    frame = np.column_stack((tangent_1, tangent_2, normal))
    if not np.all(np.isfinite(frame)):
        raise ValueError("S3 material skin pose exceeded finite range")
    frame.setflags(write=False)
    return frame


@dataclass(frozen=True)
class S3SkinPointPose:
    position: tuple[float, float, float]
    material_frame: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class S3SkinMaterialPoint:
    """One persistent barycentric point on an admitted physical S3 skin."""

    adapter: S3SkinWorkAdapter
    barycentric: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, S3SkinWorkAdapter):
            raise TypeError("adapter must be an S3SkinWorkAdapter")
        weights = np.asarray(self.barycentric, dtype=float)
        if (weights.shape != (3,) or not np.all(np.isfinite(weights))
                or np.any(weights < 0.0)
                or abs(float(np.sum(weights)) - 1.0) > 1.0e-12):
            raise ValueError("S3 material coordinates need three nonnegative weights summing to one")
        object.__setattr__(self, "barycentric",
                           tuple(float(value) for value in weights))

    @classmethod
    def from_model(cls, model, body_id: int, element_id: int,
                   side: str, barycentric) -> S3SkinMaterialPoint:
        return cls(
            S3SkinWorkAdapter.from_model(model, body_id, element_id, side),
            tuple(barycentric),
        )

    @property
    def position_map(self) -> sparse.csr_matrix:
        weights = sparse.csr_matrix(np.asarray(self.barycentric).reshape(1, 3))
        interpolation = sparse.kron(weights, sparse.eye(3, format="csr"),
                                    format="csr")
        return (interpolation @ self.adapter.work_map).tocsr()

    def physical_pose(self, displacement: np.ndarray) -> S3SkinPointPose:
        corners = self.adapter.current_positions(displacement)
        reference = self.adapter.reference_positions
        frame=_physical_skin_frame(np.asarray(corners,dtype=np.float64).tobytes(),
            np.asarray(reference,dtype=np.float64).tobytes())
        position = np.asarray(self.barycentric) @ corners
        if not np.all(np.isfinite(frame)) or not np.all(np.isfinite(position)):
            raise ValueError("S3 material skin pose exceeded finite range")
        return S3SkinPointPose(
            tuple(float(value) for value in position),
            tuple(tuple(float(value) for value in row) for row in frame),
        )

    def frame_directional_variation(
            self, displacement: np.ndarray,
            direction: np.ndarray) -> np.ndarray:
        """Exact smooth-pose variation of the material frame per FE direction."""

        self.physical_pose(displacement)  # Shared geometry and polarity checks.
        variation = np.asarray(direction, dtype=float).reshape(-1)
        if (variation.size != self.adapter.work_map.shape[1]
                or not np.all(np.isfinite(variation))):
            raise ValueError("S3 frame variation needs a finite full FE direction")
        corners = self.adapter.current_positions(displacement)
        moved = np.asarray(self.adapter.work_map @ variation).reshape(3, 3)
        first = corners[1] - corners[0]
        second = corners[2] - corners[0]
        d_first = moved[1] - moved[0]
        d_second = moved[2] - moved[0]
        length = float(np.linalg.norm(first))
        raw_normal = np.cross(first, second)
        area_scale = float(np.linalg.norm(raw_normal))
        tangent_1 = first / length
        normal = raw_normal / area_scale
        d_raw_normal = (np.cross(d_first, second)
                        + np.cross(first, d_second))
        d_tangent_1 = ((np.eye(3) - np.outer(tangent_1, tangent_1))
                       @ d_first / length)
        d_normal = ((np.eye(3) - np.outer(normal, normal))
                    @ d_raw_normal / area_scale)
        d_tangent_2 = (np.cross(d_normal, tangent_1)
                       + np.cross(normal, d_tangent_1))
        derivative = np.column_stack((d_tangent_1, d_tangent_2, d_normal))
        if not np.all(np.isfinite(derivative)):
            raise ValueError("S3 frame variation exceeded finite range")
        return derivative
