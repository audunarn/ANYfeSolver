"""Shared geometry and element fixtures for the S3 minimum-angle investigation.

The helpers here are deliberately independent of the solver's own triangle
quality code: angles come from the law of cosines on edge lengths (the solver
uses normalized dot products), and the normalized area uses the textbook
``q = 4 sqrt(3) A / (l1^2 + l2^2 + l3^2)`` definition.  They are shared by the
fast regression tests and by ``scripts/qualify_s3_angles.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


ANGLE_BANDS_DEG = (30.0, 25.0, 20.0, 17.5, 15.0)
DIAGNOSTIC_BANDS_DEG = (14.0, 12.0, 10.0)
THICKNESS_RATIOS = (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4)
E = 210.0e9
NU = 0.3
DENSITY = 7850.0

# Candidate joint envelopes investigated by this plan.  They are test inputs,
# not production policy.  Maximum angle, edge ratio and scaled Jacobian keep
# the historical values; only the angle floor and normalized-area floor move.
CANDIDATE_ENVELOPES = {
    "historical_30": {
        "minimum_angle_deg": 30.0,
        "maximum_angle_deg": 150.0,
        "maximum_edge_ratio": 4.0,
        "minimum_scaled_jacobian": 0.20,
        "minimum_normalized_area": 0.60,
    },
    "candidate_20": {
        "minimum_angle_deg": 20.0,
        "maximum_angle_deg": 150.0,
        "maximum_edge_ratio": 4.0,
        "minimum_scaled_jacobian": 0.20,
        "minimum_normalized_area": 0.40,
    },
    "candidate_15": {
        "minimum_angle_deg": 15.0,
        "maximum_angle_deg": 150.0,
        "maximum_edge_ratio": 4.0,
        "minimum_scaled_jacobian": 0.20,
        "minimum_normalized_area": 0.30,
    },
}


def triangle_from_angles(
    alpha_deg: float, beta_deg: float, length: float = 1.0
) -> np.ndarray:
    """Return a counter-clockwise +z triangle with interior angles at nodes 0, 1.

    Node 0 is at the origin, node 1 at ``(length, 0, 0)``; the third angle is
    ``180 - alpha - beta``.
    """

    gamma_deg = 180.0 - alpha_deg - beta_deg
    if min(alpha_deg, beta_deg, gamma_deg) <= 0.0:
        raise ValueError("triangle angles must be positive and sum to 180")
    alpha = math.radians(alpha_deg)
    side = length * math.sin(math.radians(beta_deg)) / math.sin(math.radians(gamma_deg))
    return np.asarray(
        (
            (0.0, 0.0, 0.0),
            (length, 0.0, 0.0),
            (side * math.cos(alpha), side * math.sin(alpha), 0.0),
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class TriangleQuality:
    angles_deg: tuple[float, float, float]
    area: float
    edge_ratio: float
    minimum_scaled_jacobian: float
    normalized_area: float

    @property
    def minimum_angle_deg(self) -> float:
        return min(self.angles_deg)

    @property
    def maximum_angle_deg(self) -> float:
        return max(self.angles_deg)


def independent_quality(coordinates: Sequence[Sequence[float]]) -> TriangleQuality:
    """Recompute quality from edge lengths (law of cosines), not dot products."""

    nodes = np.asarray(coordinates, dtype=np.float64).reshape(3, 3)
    # a_i is the edge opposite node i.
    a = np.asarray(
        [
            np.linalg.norm(nodes[2] - nodes[1]),
            np.linalg.norm(nodes[0] - nodes[2]),
            np.linalg.norm(nodes[1] - nodes[0]),
        ]
    )
    angles = []
    for index in range(3):
        opposite = a[index]
        left, right = a[(index + 1) % 3], a[(index + 2) % 3]
        cosine = (left * left + right * right - opposite * opposite) / (2.0 * left * right)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    area = 0.5 * float(np.linalg.norm(np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])))
    return TriangleQuality(
        angles_deg=tuple(angles),  # type: ignore[arg-type]
        area=area,
        edge_ratio=float(np.max(a) / np.min(a)),
        minimum_scaled_jacobian=float(min(math.sin(math.radians(v)) for v in angles)),
        normalized_area=float(4.0 * math.sqrt(3.0) * area / float(a @ a)),
    )


def envelope_violations(quality: TriangleQuality, envelope: dict[str, float]) -> list[str]:
    tolerance = 1.0e-12
    failures = []
    if quality.minimum_angle_deg < envelope["minimum_angle_deg"] - tolerance:
        failures.append("minimum_angle")
    if quality.maximum_angle_deg > envelope["maximum_angle_deg"] + tolerance:
        failures.append("maximum_angle")
    if quality.edge_ratio > envelope["maximum_edge_ratio"] + tolerance:
        failures.append("edge_ratio")
    if quality.minimum_scaled_jacobian < envelope["minimum_scaled_jacobian"] - tolerance:
        failures.append("scaled_jacobian")
    if quality.normalized_area < envelope["minimum_normalized_area"] - tolerance:
        failures.append("normalized_area")
    return failures


def shape_family(minimum_angle_deg: float) -> dict[str, tuple[float, float]]:
    """Representative shapes whose smallest interior angle is ``minimum_angle_deg``.

    Values are ``(alpha, beta)`` for :func:`triangle_from_angles`.
    """

    a = float(minimum_angle_deg)
    shapes = {
        "acute_isosceles": (a, 0.5 * (180.0 - a)),
        "right": (a, 90.0),
        # third angle 90 - 0.65 a: all angles acute, none equal.
        "asymmetric_acute": (a, 90.0 - 0.35 * a),
        # third angle 180 - 2 a - 0.3 (180 - 3 a): obtuse for a <= 30.
        "asymmetric_obtuse": (a, a + 0.3 * (180.0 - 3.0 * a)),
        "obtuse_isosceles": (a, a),
    }
    # Keep only shapes whose smallest angle is exactly the band angle.
    return {
        name: angles
        for name, angles in shapes.items()
        if min(angles[0], angles[1], 180.0 - sum(angles)) >= a - 1.0e-12
    }


def rotation_matrix(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    axis_array = axis_array / np.linalg.norm(axis_array)
    x, y, z = axis_array
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    C = 1.0 - c
    return np.asarray(
        (
            (c + x * x * C, x * y * C - z * s, x * z * C + y * s),
            (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
            (z * x * C - y * s, z * y * C + x * s, c + z * z * C),
        )
    )


def seeded_rotations(seed: int, count: int) -> Iterator[np.ndarray]:
    generator = np.random.default_rng(seed)
    for _ in range(count):
        yield rotation_matrix(generator.normal(size=3), generator.uniform(0.1, 3.0))


# --------------------------------------------------------------------------
# Element construction on the public current-policy S3 route.
# --------------------------------------------------------------------------


def build_default_s3(
    coordinates: np.ndarray,
    *,
    thickness: float,
    normal: Sequence[float] = (0.0, 0.0, 1.0),
    node_ids: Sequence[int] = (1, 2, 3),
):
    """Build a one-element mesh and the *default* three-node shell element."""

    from anyfesolver.elements import create_shell_element
    from anyfesolver.fe_core import FEMesh, Material

    mesh = FEMesh()
    for node_id, xyz in zip((1, 2, 3), coordinates):
        mesh.add_node(node_id, *map(float, xyz))
    element = create_shell_element(
        1,
        list(node_ids),
        "steel",
        thickness=float(thickness),
        reference_normal=tuple(float(v) for v in normal),
    )
    material = Material("steel", E, NU, density=DENSITY)
    return mesh, element, material


def rigid_body_basis(coordinates: np.ndarray, length: float) -> np.ndarray:
    """Six unit-scaled rigid modes in the element's external DOF order."""

    centroid = np.mean(coordinates, axis=0)
    basis = np.zeros((18, 6))
    for mode in range(3):
        basis[mode::6, mode] = 1.0
    for axis in range(3):
        omega = np.zeros(3)
        omega[axis] = 1.0 / length
        for node, xyz in enumerate(coordinates):
            basis[6 * node : 6 * node + 3, 3 + axis] = np.cross(omega, xyz - centroid)
            basis[6 * node + 3 : 6 * node + 6, 3 + axis] = omega
    return basis


def dimensionless_stiffness(K: np.ndarray, length: float) -> np.ndarray:
    """Scale rotations by ``length`` and the result by the largest diagonal."""

    scale = np.ones(18)
    for node in range(3):
        scale[6 * node + 3 : 6 * node + 6] = 1.0 / length
    scaled = (scale[:, None] * K) * scale[None, :]
    return scaled / float(np.max(np.abs(np.diag(scaled))))


def elastic_spectrum(K: np.ndarray, coordinates: np.ndarray, length: float) -> np.ndarray:
    """Eigenvalues of the dimensionless stiffness on the rigid-body complement."""

    Ks = dimensionless_stiffness(K, length)
    scale = np.ones(18)
    for node in range(3):
        scale[6 * node + 3 : 6 * node + 6] = 1.0 / length
    rigid = rigid_body_basis(coordinates, length) / scale[:, None]
    q, _ = np.linalg.qr(rigid)
    full, _ = np.linalg.qr(np.hstack((q, np.eye(18))))
    complement = full[:, 6:]
    return np.linalg.eigvalsh(complement.T @ (0.5 * (Ks + Ks.T)) @ complement)


def isotropic_plane_stress(E_: float = E, nu: float = NU) -> np.ndarray:
    return E_ / (1.0 - nu * nu) * np.asarray(
        ((1.0, nu, 0.0), (nu, 1.0, 0.0), (0.0, 0.0, 0.5 * (1.0 - nu)))
    )


def membrane_patch_field(coordinates: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Linear in-plane field ``u = G x`` with the matching drill rotation."""

    u = np.zeros(18)
    drill = 0.5 * (gradient[1, 0] - gradient[0, 1])
    for node, xyz in enumerate(coordinates):
        u[6 * node : 6 * node + 2] = gradient @ xyz[:2]
        u[6 * node + 5] = drill
    return u


def bending_patch_field(coordinates: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    """Quadratic Kirchhoff field ``w = x^T H x / 2`` with ``theta = grad-w``."""

    u = np.zeros(18)
    for node, xyz in enumerate(coordinates):
        x = xyz[:2]
        grad = hessian @ x
        u[6 * node + 2] = 0.5 * float(x @ hessian @ x)
        # rotation vector: w = theta_x y - theta_y x  (u = theta x r)
        u[6 * node + 3] = grad[1]
        u[6 * node + 4] = -grad[0]
    return u
