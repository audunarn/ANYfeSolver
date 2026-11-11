"""Small deterministic operators shared by four-node shell formulations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]

NATURAL_CORNERS = np.array(
    [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    dtype=np.float64,
)
NATURAL_CORNERS.setflags(write=False)

_GAUSS = 1.0 / np.sqrt(3.0)
GAUSS_2 = np.array([-_GAUSS, _GAUSS], dtype=np.float64)
GAUSS_2.setflags(write=False)

# Counter-clockwise order keeps adjacent surface stations contiguous.
SURFACE_GAUSS_POINTS = np.array(
    [[-_GAUSS, -_GAUSS], [_GAUSS, -_GAUSS], [_GAUSS, _GAUSS], [-_GAUSS, _GAUSS]],
    dtype=np.float64,
)
SURFACE_GAUSS_POINTS.setflags(write=False)

EDGE_MIDPOINTS = np.array(
    [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
    dtype=np.float64,
)
EDGE_MIDPOINTS.setflags(write=False)


def readonly_f64(value: ArrayLike, *, copy: bool = True) -> FloatArray:
    """Return a finite, C-contiguous, read-only float64 array."""

    array = np.array(value, dtype=np.float64, order="C", copy=copy)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric reference data must be finite")
    array.setflags(write=False)
    return array


def q4_shape(r: float, s: float) -> tuple[FloatArray, FloatArray]:
    """Bilinear Q4 values and natural derivatives.

    The derivative array has shape ``(4, 2)`` with columns ``d/dr`` and
    ``d/ds``.  Numbering follows :data:`NATURAL_CORNERS`.
    """

    r = float(r)
    s = float(s)
    if not np.isfinite(r) or not np.isfinite(s):
        raise ValueError("natural coordinates must be finite")
    corner_r = NATURAL_CORNERS[:, 0]
    corner_s = NATURAL_CORNERS[:, 1]
    values = 0.25 * (1.0 + corner_r * r) * (1.0 + corner_s * s)
    derivatives = np.empty((4, 2), dtype=np.float64)
    derivatives[:, 0] = 0.25 * corner_r * (1.0 + corner_s * s)
    derivatives[:, 1] = 0.25 * corner_s * (1.0 + corner_r * r)
    return values, derivatives


def q4_midside_shapes(r: float, s: float) -> tuple[FloatArray, FloatArray]:
    """Quadratic fictitious midside shapes and their full derivatives.

    Edge order is bottom, right, top, left, matching the oriented node pairs
    ``(0,1), (1,2), (2,3), (3,0)``.
    """

    r = float(r)
    s = float(s)
    values = 0.5 * np.array(
        [
            (1.0 - r * r) * (1.0 - s),
            (1.0 - s * s) * (1.0 + r),
            (1.0 - r * r) * (1.0 + s),
            (1.0 - s * s) * (1.0 - r),
        ],
        dtype=np.float64,
    )
    derivatives = np.array(
        [
            [-r * (1.0 - s), -0.5 * (1.0 - r * r)],
            [0.5 * (1.0 - s * s), -s * (1.0 + r)],
            [-r * (1.0 + s), 0.5 * (1.0 - r * r)],
            [-0.5 * (1.0 - s * s), -s * (1.0 - r)],
        ],
        dtype=np.float64,
    )
    return values, derivatives


def q4_assumed_midside_derivatives(r: float, s: float) -> FloatArray:
    """Curl-marked `/D` midside derivatives from Ko et al. (2025), Eq. 11.

    Only the edge-tangential derivative is retained: ``d/dr`` on bottom/top
    and ``d/ds`` on right/left.  This is an assumed strain operator, not the
    derivative of a replacement displacement field.
    """

    r = float(r)
    s = float(s)
    if not np.isfinite(r) or not np.isfinite(s):
        raise ValueError("natural coordinates must be finite")

    # Eq. (11) is the curl-marked field formed by zeroing the cross-edge
    # derivatives of Eq. (10).  In paper order (right, top, left, bottom),
    # its nonzero entries are
    #   h5,s = -s(1+r), h6,r = -r(1+s),
    #   h7,s = -s(1-r), h8,r = -r(1-s).
    # Reorder those four entries to this module's bottom/right/top/left cycle.
    assumed = np.zeros((4, 2), dtype=np.float64)
    assumed[0, 0] = -r * (1.0 - s)
    assumed[1, 1] = -s * (1.0 + r)
    assumed[2, 0] = -r * (1.0 + s)
    assumed[3, 1] = -s * (1.0 - r)
    return assumed


def cross_matrix(vector: ArrayLike) -> FloatArray:
    """Matrix ``S(v)`` such that ``S(v) @ w == cross(v, w)``."""

    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def normalized(vector: ArrayLike, tolerance: float, name: str) -> FloatArray:
    """Return a normalized vector or fail deterministically."""

    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(array))
    if norm <= tolerance:
        raise ValueError(f"{name} is singular (norm={norm:.6e})")
    return array / norm


def plane_dual_basis(
    first: ArrayLike, second: ArrayLike, tolerance: float
) -> tuple[FloatArray, FloatArray, float]:
    """Return reciprocal vectors of two independent vectors in their plane."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    gram = np.array(
        [
            [np.dot(first_array, first_array), np.dot(first_array, second_array)],
            [np.dot(second_array, first_array), np.dot(second_array, second_array)],
        ],
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(gram)
    if float(eigenvalues[0]) <= tolerance * tolerance:
        raise ValueError("Q4 center-plane covariant basis is singular")
    coefficients = np.linalg.inv(gram)
    dual_first = coefficients[0, 0] * first_array + coefficients[0, 1] * second_array
    dual_second = coefficients[1, 0] * first_array + coefficients[1, 1] * second_array
    condition = float(eigenvalues[-1] / eigenvalues[0])
    return dual_first, dual_second, condition


def plane_tensor_covariant_transform(coordinate_map: ArrayLike) -> FloatArray:
    """Return the tensor-component map for a two-dimensional basis change.

    ``coordinate_map[i, k]`` is ``g_i . gbar^k``.  The returned matrix maps
    symmetric tensor components ``[e_11, e_22, e_12]`` in the fixed barred
    basis to ``[e_rr, e_ss, e_rs]`` in the current covariant basis.  Both
    input and output shear entries are tensor shear; no engineering-shear
    factor is introduced here.
    """

    mapping = np.asarray(coordinate_map, dtype=np.float64)
    if mapping.shape != (2, 2) or not np.all(np.isfinite(mapping)):
        raise ValueError("coordinate_map must be a finite (2, 2) array")
    a, b = mapping[0]
    c, d = mapping[1]
    return np.array(
        [
            [a * a, b * b, 2.0 * a * b],
            [c * c, d * d, 2.0 * c * d],
            [a * c, b * d, a * d + b * c],
        ],
        dtype=np.float64,
    )


def local_frame_and_reciprocal(
    covariant_bases: ArrayLike, tolerance: float
) -> tuple[FloatArray, FloatArray, float]:
    """Build the paper's shell-aligned frame and reciprocal bases.

    ``covariant_bases`` has rows ``g_r, g_s, g_zeta``.  The local frame follows
    Ko, Lee, and Bathe (2017), Eq. 17: ``L3 || g_zeta``,
    ``L1 || g_s x L3``, and ``L2 = L3 x L1``.
    """

    covariant = np.asarray(covariant_bases, dtype=np.float64)
    if covariant.shape != (3, 3) or not np.all(np.isfinite(covariant)):
        raise ValueError("covariant_bases must be a finite (3, 3) array")
    jacobian_columns = covariant.T
    determinant = float(np.linalg.det(jacobian_columns))
    if determinant <= tolerance:
        raise ValueError(
            "Q4 continuum Jacobian must be positive; "
            f"determinant={determinant:.6e}, tolerance={tolerance:.6e}"
        )
    reciprocal = np.linalg.inv(jacobian_columns)
    local_3 = normalized(covariant[2], tolerance, "thickness covariant vector")
    local_1 = normalized(np.cross(covariant[1], local_3), tolerance, "local L1")
    local_2 = np.cross(local_3, local_1)
    local = np.vstack((local_1, local_2, local_3))
    return local, reciprocal, determinant


def covariant_to_local_engineering(
    covariant_bases: ArrayLike, tolerance: float
) -> tuple[FloatArray, float]:
    """Map covariant tensor components to five local engineering components.

    Input component order is ``[e_rr,e_ss,e_rs,e_rzeta,e_szeta]`` with tensor
    shear.  Output order is ``[e_11,e_22,gamma_12,gamma_13,gamma_23]``.
    """

    local, reciprocal, determinant = local_frame_and_reciprocal(
        covariant_bases, tolerance
    )
    # A[i,k] = L_i . g^k; reciprocal rows are the reciprocal vectors g^k.
    projection = local @ reciprocal.T

    def tensor_row(i: int, j: int, engineering_factor: float) -> FloatArray:
        ai = projection[i]
        aj = projection[j]
        return engineering_factor * np.array(
            [
                ai[0] * aj[0],
                ai[1] * aj[1],
                ai[0] * aj[1] + ai[1] * aj[0],
                ai[0] * aj[2] + ai[2] * aj[0],
                ai[1] * aj[2] + ai[2] * aj[1],
            ],
            dtype=np.float64,
        )

    transform = np.vstack(
        (
            tensor_row(0, 0, 1.0),
            tensor_row(1, 1, 1.0),
            tensor_row(0, 1, 2.0),
            tensor_row(0, 2, 2.0),
            tensor_row(1, 2, 2.0),
        )
    )
    return transform, determinant


def deterministic_signature(label: str, arrays: Iterable[ArrayLike]) -> str:
    """Hash a formulation label plus array shape/dtype/content deterministically."""

    digest = hashlib.sha256(label.encode("ascii"))
    for value in arrays:
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
