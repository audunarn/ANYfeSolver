"""Clean-room scalar operators for the full 2 x 2 MITC4+/D element.

Equation anchors are Ko, Bathe, and Zhang (2025), Eqs. 1-25, including the
literal Eq. 21 drill-tensor transformation and Eqs. 24-25 QRS membrane field.
The explicitly named 2017 Eq. 27 operator is retained only as a non-default
comparison oracle.  See ``docs/S4_IMPROVED_FORMULATION.md`` for the complete
notation and sign mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .protocol import Q4QualityMetrics, Q4ReferenceData
from .q4_common import (
    EDGE_MIDPOINTS,
    GAUSS_2,
    SURFACE_GAUSS_POINTS,
    covariant_to_local_engineering,
    cross_matrix,
    deterministic_signature,
    plane_tensor_covariant_transform,
    plane_dual_basis,
    q4_assumed_midside_derivatives,
    q4_midside_shapes,
    q4_shape,
)


FloatArray = NDArray[np.float64]


class _ReferenceLike(Protocol):
    coordinates: FloatArray
    directors: FloatArray
    thickness: FloatArray
    drill_direction: FloatArray
    center_covariant: FloatArray
    center_dual: FloatArray
    distortion_vector: FloatArray
    distortion_scalars: FloatArray
    mitc4_plus_coefficients: FloatArray
    mitc4_plus_qrs_coefficients: FloatArray
    drill_edge_coefficients: FloatArray


@dataclass(slots=True)
class _ReferenceGeometry:
    coordinates: FloatArray
    directors: FloatArray
    thickness: FloatArray
    drill_direction: FloatArray
    center_covariant: FloatArray
    center_dual: FloatArray
    distortion_vector: FloatArray
    distortion_scalars: FloatArray
    mitc4_plus_coefficients: FloatArray
    mitc4_plus_qrs_coefficients: FloatArray
    drill_edge_coefficients: FloatArray


def _relative_coordinates(reference: _ReferenceLike) -> FloatArray:
    coordinates = np.asarray(reference.coordinates, dtype=np.float64)
    return coordinates - np.mean(coordinates, axis=0)


def _reference_scales(reference: _ReferenceLike) -> tuple[float, float, float]:
    coordinates = np.asarray(reference.coordinates, dtype=np.float64)
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    length = float(np.max(np.linalg.norm(differences, axis=2)))
    thickness = float(np.mean(reference.thickness))
    epsilon = np.finfo(np.float64).eps
    vector_tolerance = 128.0 * epsilon * max(length, thickness, np.finfo(float).tiny)
    jacobian_tolerance = (
        256.0
        * epsilon
        * max(length * length * thickness, np.finfo(float).tiny)
    )
    return length, vector_tolerance, jacobian_tolerance


def _geometry_bases(
    reference: _ReferenceLike, r: float, s: float, zeta: float
) -> FloatArray:
    values, derivatives = q4_shape(r, s)
    coordinates = _relative_coordinates(reference)
    director_offsets = 0.5 * reference.thickness[:, None] * reference.directors
    g_r = derivatives[:, 0] @ coordinates + zeta * (derivatives[:, 0] @ director_offsets)
    g_s = derivatives[:, 1] @ coordinates + zeta * (derivatives[:, 1] @ director_offsets)
    g_zeta = values @ director_offsets
    return np.vstack((g_r, g_s, g_zeta))


def _positive_jacobian(
    reference: _ReferenceLike, r: float, s: float, zeta: float
) -> float:
    _, _, tolerance = _reference_scales(reference)
    determinant = float(np.linalg.det(_geometry_bases(reference, r, s, zeta).T))
    if determinant <= tolerance:
        raise ValueError(
            "Q4 continuum Jacobian must be positive; "
            f"determinant={determinant:.6e}, tolerance={tolerance:.6e}, "
            f"point=({r:.6g},{s:.6g},{zeta:.6g})"
        )
    return determinant


def _continuum_displacement_parts(
    reference: _ReferenceLike, r: float, s: float, zeta: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return continuum H and its r/s/zeta derivatives (without `/D`)."""

    values, derivatives = q4_shape(r, s)
    displacement = np.zeros((3, 24), dtype=np.float64)
    derivative_r = np.zeros((3, 24), dtype=np.float64)
    derivative_s = np.zeros((3, 24), dtype=np.float64)
    derivative_zeta = np.zeros((3, 24), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)

    for node in range(4):
        translation = slice(6 * node, 6 * node + 3)
        rotation = slice(6 * node + 3, 6 * node + 6)
        director_rotation = -cross_matrix(reference.directors[node])
        half_thickness = 0.5 * reference.thickness[node]

        displacement[:, translation] = values[node] * identity
        derivative_r[:, translation] = derivatives[node, 0] * identity
        derivative_s[:, translation] = derivatives[node, 1] * identity

        displacement[:, rotation] = (
            zeta * half_thickness * values[node] * director_rotation
        )
        derivative_r[:, rotation] = (
            zeta * half_thickness * derivatives[node, 0] * director_rotation
        )
        derivative_s[:, rotation] = (
            zeta * half_thickness * derivatives[node, 1] * director_rotation
        )
        derivative_zeta[:, rotation] = (
            half_thickness * values[node] * director_rotation
        )

    return displacement, derivative_r, derivative_s, derivative_zeta


def _raw_covariant_operator(
    reference: _ReferenceLike, r: float, s: float, zeta: float
) -> FloatArray:
    """Displacement-based covariant tensor operator before MITC projection."""

    _, derivative_r, derivative_s, derivative_zeta = _continuum_displacement_parts(
        reference, r, s, zeta
    )
    g_r, g_s, g_zeta = _geometry_bases(reference, r, s, zeta)
    operator = np.empty((5, 24), dtype=np.float64)
    operator[0] = g_r @ derivative_r
    operator[1] = g_s @ derivative_s
    operator[2] = 0.5 * (g_r @ derivative_s + g_s @ derivative_r)
    operator[3] = 0.5 * (g_r @ derivative_zeta + g_zeta @ derivative_r)
    operator[4] = 0.5 * (g_s @ derivative_zeta + g_zeta @ derivative_s)
    return operator


def _mid_membrane_operator(reference: _ReferenceLike, r: float, s: float) -> FloatArray:
    """Displacement-based midsurface membrane tensor operator."""

    return _raw_covariant_operator(reference, r, s, 0.0)[:3]


def _assumed_mitc4_plus_membrane_2017_eq27_reference(
    reference: _ReferenceLike, r: float, s: float
) -> FloatArray:
    """Non-default comparison field from Ko-Lee-Bathe (2017), Eq. 27."""

    tying_a = _mid_membrane_operator(reference, 0.0, 1.0)[0]
    tying_b = _mid_membrane_operator(reference, 0.0, -1.0)[0]
    tying_c = _mid_membrane_operator(reference, 1.0, 0.0)[1]
    tying_d = _mid_membrane_operator(reference, -1.0, 0.0)[1]
    tying_e = _mid_membrane_operator(reference, 0.0, 0.0)[2]
    a_a, a_b, a_c, a_d, a_e = reference.mitc4_plus_coefficients

    result = np.empty((3, 24), dtype=np.float64)
    result[0] = (
        0.5 * (1.0 - 2.0 * a_a + s + 2.0 * a_a * s * s) * tying_a
        + 0.5 * (1.0 - 2.0 * a_b - s + 2.0 * a_b * s * s) * tying_b
        + a_c * (-1.0 + s * s) * tying_c
        + a_d * (-1.0 + s * s) * tying_d
        + a_e * (-1.0 + s * s) * tying_e
    )
    result[1] = (
        a_a * (-1.0 + r * r) * tying_a
        + a_b * (-1.0 + r * r) * tying_b
        + 0.5 * (1.0 - 2.0 * a_c + r + 2.0 * a_c * r * r) * tying_c
        + 0.5 * (1.0 - 2.0 * a_d - r + 2.0 * a_d * r * r) * tying_d
        + a_e * (-1.0 + r * r) * tying_e
    )
    result[2] = (
        0.25 * (r + 4.0 * a_a * r * s) * tying_a
        + 0.25 * (-r + 4.0 * a_b * r * s) * tying_b
        + 0.25 * (s + 4.0 * a_c * r * s) * tying_c
        + 0.25 * (-s + 4.0 * a_d * r * s) * tying_d
        + (1.0 + a_e * r * s) * tying_e
    )
    return result


def _mitc4_plus_2025_qrs_map(
    reference: _ReferenceLike, r: float, s: float
) -> FloatArray:
    """Literal tensor-component map ``Q(r,s) R(r,s) S`` from 2025 Eq. 25."""

    r = float(r)
    s = float(s)
    if not np.all(np.isfinite((r, s))):
        raise ValueError("natural coordinates must be finite")

    c_r, c_s, denominator = reference.distortion_scalars
    a_a, a_b, a_c, a_d, a_e = reference.mitc4_plus_coefficients
    n_1, n_2, n_3, n_4, n_5, m_1, m_2, m_3, m_4, m_5 = (
        reference.mitc4_plus_qrs_coefficients
    )

    center_jacobian = _positive_jacobian(reference, 0.0, 0.0, 0.0)
    point_jacobian = _positive_jacobian(reference, r, s, 0.0)
    lambda_ratio = center_jacobian / point_jacobian
    inverse_lambda = 1.0 / lambda_ratio
    root_three = np.sqrt(3.0)

    q_map = np.array(
        [
            [
                (1.0 + c_r * s) ** 2,
                (c_s * s) ** 2,
                2.0 * c_s * s * (1.0 + c_r * s),
            ],
            [
                (c_r * r) ** 2,
                (1.0 + c_s * r) ** 2,
                2.0 * c_r * r * (1.0 + c_s * r),
            ],
            [
                c_r * r * (1.0 + c_r * s),
                c_s * s * (1.0 + c_s * r),
                c_r * c_s * r * s
                + (1.0 + c_r * s) * (1.0 + c_s * r),
            ],
        ],
        dtype=np.float64,
    )
    r_map = lambda_ratio * np.array(
        [
            [
                inverse_lambda + root_three * n_1 * s,
                root_three * n_2 * s,
                2.0 * root_three * n_5 * s,
                n_3 * s,
                n_4 * s,
                n_1 * s / root_three,
            ],
            [
                root_three * m_2 * r,
                inverse_lambda + root_three * m_1 * r,
                2.0 * root_three * m_5 * r,
                m_4 * r,
                m_3 * r,
                m_1 * r / root_three,
            ],
            [0.0, 0.0, inverse_lambda, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    s_map = np.array(
        [
            [0.5 - a_a, 0.5 - a_b, -a_c, -a_d, -a_e],
            [-a_a, -a_b, 0.5 - a_c, 0.5 - a_d, -a_e],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.5, -0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, -0.5, 0.0],
            [a_a, a_b, a_c, a_d, a_e],
        ],
        dtype=np.float64,
    )
    if denominator >= 0.0:
        # The coefficients are finite away from d=0, but the literal Q4+
        # construction is defined for the valid convex-map branch d < 0.
        raise ValueError(
            "MITC4+ distortion denominator must be negative for Eq. 25; "
            f"d={denominator:.6e}"
        )
    return q_map @ r_map @ s_map


def _assumed_mitc4_plus_membrane_2025_eq25(
    reference: _ReferenceLike, r: float, s: float
) -> FloatArray:
    """Selected literal MITC4+ membrane field from 2025 Eqs. 24-25."""

    tying_operators = np.vstack(
        (
            _mid_membrane_operator(reference, 0.0, 1.0)[0],
            _mid_membrane_operator(reference, 0.0, -1.0)[0],
            _mid_membrane_operator(reference, 1.0, 0.0)[1],
            _mid_membrane_operator(reference, -1.0, 0.0)[1],
            _mid_membrane_operator(reference, 0.0, 0.0)[2],
        )
    )
    return _mitc4_plus_2025_qrs_map(reference, r, s) @ tying_operators


def _drill_membrane_operator(
    reference: _ReferenceLike, r: float, s: float
) -> FloatArray:
    """Physical assumed drill-membrane field, Ko et al. (2025), Eqs. 17-21."""

    derivatives = q4_assumed_midside_derivatives(r, s)
    center_jacobian = _positive_jacobian(reference, 0.0, 0.0, 0.0)
    point_jacobian = _positive_jacobian(reference, r, s, 0.0)
    scale = center_jacobian / point_jacobian
    fixed_center = np.zeros((3, 24), dtype=np.float64)
    drill_direction = reference.drill_direction

    for edge in range(4):
        node_i = edge
        node_j = (edge + 1) % 4
        derivative_r, derivative_s = derivatives[edge]
        coefficient_r, coefficient_s = reference.drill_edge_coefficients[edge]
        scalar_coefficients = np.array(
            [
                scale * derivative_r * coefficient_r,
                -scale * derivative_s * coefficient_s,
                0.5
                * scale
                * (
                    derivative_s * coefficient_r
                    - derivative_r * coefficient_s
                ),
            ],
            dtype=np.float64,
        )
        rotation_i = slice(6 * node_i + 3, 6 * node_i + 6)
        rotation_j = slice(6 * node_j + 3, 6 * node_j + 6)
        fixed_center[:, rotation_i] -= (
            scalar_coefficients[:, None] * drill_direction
        )
        fixed_center[:, rotation_j] += (
            scalar_coefficients[:, None] * drill_direction
        )

    c_r, c_s, _ = reference.distortion_scalars
    fixed_to_natural = np.array(
        [[1.0 + s * c_r, s * c_s], [r * c_r, 1.0 + r * c_s]],
        dtype=np.float64,
    )
    return plane_tensor_covariant_transform(fixed_to_natural) @ fixed_center


def _assumed_mitc4_shear(
    reference: _ReferenceLike, r: float, s: float, zeta: float
) -> FloatArray:
    """Classical MITC4 transverse shear, Ko et al. (2025), Eq. 20."""

    tying_a = _raw_covariant_operator(reference, 0.0, 1.0, zeta)[3]
    tying_b = _raw_covariant_operator(reference, 0.0, -1.0, zeta)[3]
    tying_c = _raw_covariant_operator(reference, 1.0, 0.0, zeta)[4]
    tying_d = _raw_covariant_operator(reference, -1.0, 0.0, zeta)[4]
    result = np.empty((2, 24), dtype=np.float64)
    result[0] = 0.5 * (1.0 + s) * tying_a + 0.5 * (1.0 - s) * tying_b
    result[1] = 0.5 * (1.0 + r) * tying_c + 0.5 * (1.0 - r) * tying_d
    return result


def covariant_strain_operator(
    reference: Q4ReferenceData | _ReferenceGeometry,
    r: float,
    s: float,
    zeta: float,
) -> FloatArray:
    """Return the full MITC4+/D covariant tensor operator, shape ``(5,24)``.

    Component order is ``[e_rr,e_ss,e_rs,e_rzeta,e_szeta]``.  Covariant shear
    components are tensor shear; use :func:`local_strain_operator` for the
    engineering component contract.
    """

    r = float(r)
    s = float(s)
    zeta = float(zeta)
    if not np.all(np.isfinite((r, s, zeta))):
        raise ValueError("natural coordinates must be finite")

    raw = _raw_covariant_operator(reference, r, s, zeta)
    raw[:3] += (
        _assumed_mitc4_plus_membrane_2025_eq25(reference, r, s)
        + _drill_membrane_operator(reference, r, s)
        - _mid_membrane_operator(reference, r, s)
    )
    raw[3:] = _assumed_mitc4_shear(reference, r, s, zeta)
    return raw


def _drill_displacement_enrichment(
    reference: _ReferenceLike, r: float, s: float
) -> FloatArray:
    """Midsurface `/D` displacement used for consistent inertia (2025 Eq. 15)."""

    midside_values, _ = q4_midside_shapes(r, s)
    enrichment = np.zeros((3, 24), dtype=np.float64)
    dual_r, dual_s = reference.center_dual
    drill_direction = reference.drill_direction

    for edge in range(4):
        node_i = edge
        node_j = (edge + 1) % 4
        coefficient_r, coefficient_s = reference.drill_edge_coefficients[edge]
        physical_direction = dual_r * coefficient_r - dual_s * coefficient_s
        block = midside_values[edge] * np.outer(physical_direction, drill_direction)
        rotation_i = slice(6 * node_i + 3, 6 * node_i + 6)
        rotation_j = slice(6 * node_j + 3, 6 * node_j + 6)
        enrichment[:, rotation_i] -= block
        enrichment[:, rotation_j] += block
    return enrichment


def displacement_operator(
    reference: Q4ReferenceData | _ReferenceGeometry,
    r: float,
    s: float,
    zeta: float,
    *,
    include_drill: bool = True,
) -> FloatArray:
    """Return ``H`` such that ``u(r,s,zeta) = H @ q_e``, shape ``(3,24)``."""

    displacement, _, _, _ = _continuum_displacement_parts(reference, r, s, zeta)
    if include_drill:
        displacement += _drill_displacement_enrichment(reference, r, s)
    return displacement


def local_strain_operator(
    reference: Q4ReferenceData | _ReferenceGeometry,
    r: float,
    s: float,
    zeta: float,
) -> tuple[FloatArray, float]:
    """Return local engineering ``B`` and the positive volume Jacobian.

    Local order is ``[eps11,eps22,gamma12,gamma13,gamma23]``.
    """

    _, _, tolerance = _reference_scales(reference)
    transform, determinant = covariant_to_local_engineering(
        _geometry_bases(reference, r, s, zeta), tolerance
    )
    return transform @ covariant_strain_operator(reference, r, s, zeta), determinant


def generalized_strain_operators(
    reference: Q4ReferenceData | _ReferenceGeometry,
    r: float,
    s: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return stable generalized ``(Bm, Bb, Bs)`` operators.

    Orders are ``[eps11,eps22,gamma12]``,
    ``[kappa11,kappa22,kappa12]``, and ``[gamma13,gamma23]``.  The curvature
    sign is positive when strain grows along the positive director side.
    """

    _, _, tolerance = _reference_scales(reference)
    covariant_mid = covariant_strain_operator(reference, r, s, 0.0)
    transform_mid, _ = covariant_to_local_engineering(
        _geometry_bases(reference, r, s, 0.0), tolerance
    )
    local_mid = transform_mid @ covariant_mid
    covariant_plus = covariant_strain_operator(reference, r, s, 1.0)
    covariant_minus = covariant_strain_operator(reference, r, s, -1.0)
    derivative_zeta = 0.5 * (covariant_plus - covariant_minus)
    half_thickness = float(np.linalg.norm(_geometry_bases(reference, r, s, 0.0)[2]))
    if half_thickness <= tolerance:
        raise ValueError("interpolated shell thickness is singular")
    bending = (transform_mid[:3] @ derivative_zeta) / half_thickness
    return local_mid[:3], bending, local_mid[3:]


def _midsurface_reciprocal_basis(
    reference: _ReferenceLike, r: float, s: float
) -> FloatArray:
    """Return the direct full-3D reciprocal rows at ``(r,s,0)`` (B.1).

    The inversion deliberately includes the interpolated ``g_zeta`` vector;
    this is the paper's continuum reciprocal definition, including for warped
    elements, rather than a center-plane two-vector simplification.
    """

    _, _, tolerance = _reference_scales(reference)
    bases = _geometry_bases(reference, r, s, 0.0)
    determinant = float(np.linalg.det(bases.T))
    if determinant <= tolerance:
        raise ValueError(
            "Q4 midsurface reciprocal basis requires a positive Jacobian; "
            f"determinant={determinant:.6e}, tolerance={tolerance:.6e}, "
            f"point=({r:.6g},{s:.6g},0)"
        )
    return np.linalg.inv(bases.T)


def _appendix_b_qrs_coefficients(reference: _ReferenceLike) -> FloatArray:
    """Compute ``n1..n5,m1..m5`` from the direct 2025 Appendix-B.1 bases."""

    gauss = float(GAUSS_2[1])
    x_r, x_s = reference.center_covariant

    reciprocal_a = _midsurface_reciprocal_basis(reference, 0.0, gauss)
    reciprocal_b = _midsurface_reciprocal_basis(reference, 0.0, -gauss)
    reciprocal_c = _midsurface_reciprocal_basis(reference, gauss, 0.0)
    reciprocal_d = _midsurface_reciprocal_basis(reference, -gauss, 0.0)

    a_r = float(np.dot(x_r, reciprocal_a[0]))
    a_s = float(np.dot(x_r, reciprocal_a[1]))
    b_r = float(np.dot(x_r, reciprocal_b[0]))
    b_s = float(np.dot(x_r, reciprocal_b[1]))
    c_r = float(np.dot(x_s, reciprocal_c[0]))
    c_s = float(np.dot(x_s, reciprocal_c[1]))
    d_r = float(np.dot(x_s, reciprocal_d[0]))
    d_s = float(np.dot(x_s, reciprocal_d[1]))

    return np.array(
        [
            0.5 * (a_r * a_r - b_r * b_r),
            0.5 * (a_s * a_s - b_s * b_s),
            0.5 * (a_r * a_r + b_r * b_r),
            0.5 * (a_r * a_s + b_r * b_s),
            0.5 * (a_r * a_s - b_r * b_s),
            0.5 * (c_s * c_s - d_s * d_s),
            0.5 * (c_r * c_r - d_r * d_r),
            0.5 * (c_s * c_s + d_s * d_s),
            0.5 * (c_r * c_s + d_r * d_s),
            0.5 * (c_r * c_s - d_r * d_s),
        ],
        dtype=np.float64,
    )


def _validate_input(
    coordinates: ArrayLike,
    directors: ArrayLike,
    thickness: float | ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, float, float, float]:
    coordinate_array = np.array(coordinates, dtype=np.float64, order="C", copy=True)
    director_array = np.array(directors, dtype=np.float64, order="C", copy=True)
    if coordinate_array.shape != (4, 3):
        raise ValueError(f"coordinates must have shape (4, 3), got {coordinate_array.shape}")
    if director_array.shape != (4, 3):
        raise ValueError(f"directors must have shape (4, 3), got {director_array.shape}")
    if not np.all(np.isfinite(coordinate_array)):
        raise ValueError("coordinates must contain only finite values")
    if not np.all(np.isfinite(director_array)):
        raise ValueError("directors must contain only finite values")

    thickness_array = np.asarray(thickness, dtype=np.float64)
    if thickness_array.ndim == 0:
        thickness_array = np.full(4, float(thickness_array), dtype=np.float64)
    else:
        thickness_array = np.array(thickness_array, dtype=np.float64, order="C", copy=True)
    if thickness_array.shape != (4,):
        raise ValueError(f"thickness must be scalar or shape (4,), got {thickness_array.shape}")
    if not np.all(np.isfinite(thickness_array)) or np.any(thickness_array <= 0.0):
        raise ValueError("thickness must contain finite positive values")

    differences = coordinate_array[:, None, :] - coordinate_array[None, :, :]
    length = float(np.max(np.linalg.norm(differences, axis=2)))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("Q4 coordinates have zero extent")
    vector_tolerance = 128.0 * np.finfo(np.float64).eps * max(
        length, float(np.max(thickness_array))
    )
    jacobian_tolerance = (
        256.0
        * np.finfo(np.float64).eps
        * length
        * length
        * float(np.mean(thickness_array))
    )

    norms = np.linalg.norm(director_array, axis=1)
    normalization_tolerance = 1.0e-10
    if np.max(np.abs(norms - 1.0)) > normalization_tolerance:
        raise ValueError(
            "reference directors must be normalized; "
            f"maximum norm error={np.max(np.abs(norms - 1.0)):.6e}"
        )
    director_array /= norms[:, None]
    return (
        coordinate_array,
        director_array,
        thickness_array,
        length,
        vector_tolerance,
        jacobian_tolerance,
    )


def build_reference_data(
    coordinates: ArrayLike,
    directors: ArrayLike,
    thickness: float | ArrayLike,
    *,
    jacobian_tol: float | None = None,
) -> Q4ReferenceData:
    """Validate and build immutable full MITC4+/D reference operators.

    Parameters are numeric FE inputs only.  Geometry documents and live source
    geometry objects are intentionally outside this boundary.
    """

    (
        coordinate_array,
        director_array,
        thickness_array,
        length,
        vector_tolerance,
        default_jacobian_tolerance,
    ) = _validate_input(coordinates, directors, thickness)
    if jacobian_tol is not None:
        jacobian_tol = float(jacobian_tol)
        if not np.isfinite(jacobian_tol) or jacobian_tol <= 0.0:
            raise ValueError("jacobian_tol must be finite and positive")
        if jacobian_tol < default_jacobian_tolerance:
            raise ValueError(
                "jacobian_tol may not be smaller than the scale-aware machine floor "
                f"{default_jacobian_tolerance:.6e}"
            )

    relative = coordinate_array - np.mean(coordinate_array, axis=0)
    corner_r = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
    corner_s = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64)
    center_r = 0.25 * (corner_r @ relative)
    center_s = 0.25 * (corner_s @ relative)
    distortion = 0.25 * ((corner_r * corner_s) @ relative)
    drill_cross = np.cross(center_r, center_s)
    drill_cross_norm = float(np.linalg.norm(drill_cross))
    if drill_cross_norm <= vector_tolerance * max(length, 1.0):
        raise ValueError("Q4 center plane is singular")
    drill_direction = drill_cross / drill_cross_norm

    dual_r, dual_s, center_condition = plane_dual_basis(
        center_r, center_s, vector_tolerance
    )
    director_alignment = director_array @ drill_direction
    if np.min(director_alignment) <= 1.0e-8:
        raise ValueError(
            "reference directors reverse or become orthogonal to the Q4 center-plane "
            f"normal; minimum alignment={np.min(director_alignment):.6e}"
        )

    distortion_r = float(np.dot(distortion, dual_r))
    distortion_s = float(np.dot(distortion, dual_s))
    distortion_denominator = distortion_r**2 + distortion_s**2 - 1.0
    denominator_tolerance = 512.0 * np.finfo(np.float64).eps * max(
        1.0, distortion_r**2 + distortion_s**2
    )
    if distortion_denominator >= -denominator_tolerance:
        raise ValueError(
            "MITC4+ distortion denominator must be negative and separated "
            f"from zero; d={distortion_denominator:.6e}, "
            f"tolerance={denominator_tolerance:.6e}"
        )
    coefficients = np.array(
        [
            distortion_r * (distortion_r - 1.0) / (2.0 * distortion_denominator),
            distortion_r * (distortion_r + 1.0) / (2.0 * distortion_denominator),
            distortion_s * (distortion_s - 1.0) / (2.0 * distortion_denominator),
            distortion_s * (distortion_s + 1.0) / (2.0 * distortion_denominator),
            2.0 * distortion_r * distortion_s / distortion_denominator,
        ],
        dtype=np.float64,
    )

    provisional = _ReferenceGeometry(
        coordinates=coordinate_array,
        directors=director_array,
        thickness=thickness_array,
        drill_direction=drill_direction,
        center_covariant=np.vstack((center_r, center_s)),
        center_dual=np.vstack((dual_r, dual_s)),
        distortion_vector=distortion,
        distortion_scalars=np.array(
            [distortion_r, distortion_s, distortion_denominator], dtype=np.float64
        ),
        mitc4_plus_coefficients=coefficients,
        mitc4_plus_qrs_coefficients=np.empty(10, dtype=np.float64),
        drill_edge_coefficients=np.empty((4, 2), dtype=np.float64),
    )

    qrs_coefficients = _appendix_b_qrs_coefficients(provisional)
    provisional.mitc4_plus_qrs_coefficients = qrs_coefficients

    edge_coefficients = np.empty((4, 2), dtype=np.float64)
    for edge, (r_edge, s_edge) in enumerate(EDGE_MIDPOINTS):
        node_i = edge
        node_j = (edge + 1) % 4
        x_mid = (relative[node_i] - relative[node_j]) / 8.0
        g_r, g_s, _ = _geometry_bases(provisional, r_edge, s_edge, 0.0)
        edge_coefficients[edge, 0] = np.dot(
            x_mid, -np.cross(g_r, drill_direction)
        )
        edge_coefficients[edge, 1] = np.dot(
            x_mid, np.cross(g_s, drill_direction)
        )
    provisional.drill_edge_coefficients = edge_coefficients

    # Validate all tying locations as well as the stored quadrature locations.
    validation_surface_points = np.vstack(
        (SURFACE_GAUSS_POINTS, EDGE_MIDPOINTS, np.zeros((1, 2), dtype=np.float64))
    )
    for r_point, s_point in validation_surface_points:
        for zeta in (-1.0, 0.0, 1.0):
            determinant = float(
                np.linalg.det(_geometry_bases(provisional, r_point, s_point, zeta).T)
            )
            tolerance = (
                default_jacobian_tolerance if jacobian_tol is None else jacobian_tol
            )
            if determinant <= tolerance:
                raise ValueError(
                    "Q4 continuum Jacobian must be positive at all tying and "
                    "quadrature locations; "
                    f"determinant={determinant:.6e}, tolerance={tolerance:.6e}, "
                    f"point=({r_point:.6g},{s_point:.6g},{zeta:.6g})"
                )

    membrane_operators = np.empty((4, 3, 24), dtype=np.float64)
    bending_operators = np.empty((4, 3, 24), dtype=np.float64)
    shear_operators = np.empty((4, 2, 24), dtype=np.float64)
    surface_weights = np.empty(4, dtype=np.float64)
    volume_points = np.empty((4, 2, 3), dtype=np.float64)
    volume_weights = np.empty((4, 2), dtype=np.float64)
    volume_strain = np.empty((4, 2, 5, 24), dtype=np.float64)
    volume_displacement = np.empty((4, 2, 3, 24), dtype=np.float64)

    for surface_index, (r_point, s_point) in enumerate(SURFACE_GAUSS_POINTS):
        membrane, bending, shear = generalized_strain_operators(
            provisional, r_point, s_point
        )
        membrane_operators[surface_index] = membrane
        bending_operators[surface_index] = bending
        shear_operators[surface_index] = shear
        midsurface_bases = _geometry_bases(provisional, r_point, s_point, 0.0)
        surface_weights[surface_index] = np.linalg.norm(
            np.cross(midsurface_bases[0], midsurface_bases[1])
        )

        for thickness_index, zeta in enumerate(GAUSS_2):
            operator, determinant = local_strain_operator(
                provisional, r_point, s_point, zeta
            )
            volume_points[surface_index, thickness_index] = (r_point, s_point, zeta)
            volume_weights[surface_index, thickness_index] = determinant
            volume_strain[surface_index, thickness_index] = operator
            volume_displacement[surface_index, thickness_index] = displacement_operator(
                provisional, r_point, s_point, zeta
            )

    minimum_volume = float(np.min(volume_weights))
    maximum_volume = float(np.max(volume_weights))
    minimum_surface = float(np.min(surface_weights))
    maximum_surface = float(np.max(surface_weights))
    director_dots = np.clip(director_array @ director_array.T, -1.0, 1.0)
    maximum_director_angle = float(np.degrees(np.max(np.arccos(director_dots))))
    distortion_norm = float(
        np.linalg.norm(distortion)
        / max(np.sqrt(np.linalg.norm(center_r) * np.linalg.norm(center_s)), vector_tolerance)
    )
    quality = Q4QualityMetrics(
        minimum_volume_jacobian=minimum_volume,
        maximum_volume_jacobian=maximum_volume,
        volume_jacobian_ratio=maximum_volume / minimum_volume,
        minimum_surface_jacobian=minimum_surface,
        maximum_surface_jacobian=maximum_surface,
        surface_jacobian_ratio=maximum_surface / minimum_surface,
        center_plane_condition=center_condition,
        distortion_norm=distortion_norm,
        maximum_director_angle_degrees=maximum_director_angle,
    )
    signature = deterministic_signature(
        "mitc4_plus_d_published_2025_eq21_eq25_reference_v2",
        (coordinate_array, director_array, thickness_array),
    )

    return Q4ReferenceData(
        coordinates=coordinate_array,
        directors=director_array,
        thickness=thickness_array,
        drill_direction=drill_direction,
        center_covariant=np.vstack((center_r, center_s)),
        center_dual=np.vstack((dual_r, dual_s)),
        distortion_vector=distortion,
        distortion_scalars=np.array(
            [distortion_r, distortion_s, distortion_denominator], dtype=np.float64
        ),
        mitc4_plus_coefficients=coefficients,
        mitc4_plus_qrs_coefficients=qrs_coefficients,
        drill_edge_coefficients=edge_coefficients,
        surface_points=SURFACE_GAUSS_POINTS,
        surface_weights=surface_weights,
        generalized_membrane_operators=membrane_operators,
        generalized_bending_operators=bending_operators,
        generalized_shear_operators=shear_operators,
        volume_points=volume_points,
        volume_weights=volume_weights,
        volume_strain_operators=volume_strain,
        volume_displacement_operators=volume_displacement,
        quality=quality,
        signature=signature,
    )


def quality_metrics(reference: Q4ReferenceData) -> Q4QualityMetrics:
    """Return precomputed deterministic Q4 quality metrics."""

    return reference.quality


def batch_eligibility(reference: Q4ReferenceData) -> bool:
    """Return whether the immutable numeric reference satisfies batch shape rules."""

    return bool(
        reference.volume_strain_operators.flags.c_contiguous
        and reference.volume_displacement_operators.flags.c_contiguous
        and reference.volume_weights.flags.c_contiguous
        and np.all(reference.volume_weights > 0.0)
    )
