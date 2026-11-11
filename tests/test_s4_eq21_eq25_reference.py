"""Independent focused gates for literal 2025 Eq. 21 and Eqs. 24-25."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from numpy.typing import NDArray

from anyfesolver.shell_formulations.mitc4_plus_d_reference import (
    _assumed_mitc4_plus_membrane_2017_eq27_reference,
    _assumed_mitc4_plus_membrane_2025_eq25,
    _drill_membrane_operator,
    _mitc4_plus_2025_qrs_map,
    build_reference_data,
    covariant_strain_operator,
    displacement_operator,
    generalized_strain_operators,
    local_strain_operator,
)
from anyfesolver.shell_formulations.mitc4_plus_d_scalar import (
    consistent_mass,
    generalized_strain_at,
    isotropic_plane_stress_constitutive,
    linear_stiffness,
    strain_at,
)
from anyfesolver.shell_formulations.protocol import Q4ReferenceData
from anyfesolver.shell_formulations.q4_common import deterministic_signature


FloatArray = NDArray[np.float64]
_XI = np.array([-1.0, 1.0, 1.0, -1.0])
_ETA = np.array([-1.0, -1.0, 1.0, 1.0])
_G = 1.0 / math.sqrt(3.0)
_POINTS = (
    (0.0, 0.0),
    (_G, _G),
    (_G, -_G),
    (-_G, _G),
    (-_G, -_G),
    (-0.77, 0.64),
    (0.61, -0.57),
    (0.91, -0.83),
)


def _nodes(
    x_0: FloatArray | tuple[float, float, float],
    x_r: FloatArray | tuple[float, float, float],
    x_s: FloatArray | tuple[float, float, float],
    x_d: FloatArray | tuple[float, float, float],
) -> FloatArray:
    return np.array(
        [
            np.asarray(x_0)
            + _XI[node] * np.asarray(x_r)
            + _ETA[node] * np.asarray(x_s)
            + _XI[node] * _ETA[node] * np.asarray(x_d)
            for node in range(4)
        ],
        dtype=np.float64,
    )


def _case_coordinates() -> dict[str, FloatArray]:
    x_0 = np.array([0.2, -0.1, 0.3])
    x_r = np.array([1.1, 0.1, 0.0])
    x_s = np.array([0.1, 0.95, 0.0])
    x_rd = np.array([1.05, 0.18, 0.0])
    x_sd = np.array([0.26, 0.92, 0.0])
    x_rw = np.array([1.15, 0.12, 0.08])
    x_sw = np.array([0.18, 0.97, -0.11])
    normal = np.cross(x_rw, x_sw)
    normal /= np.linalg.norm(normal)
    return {
        "square": _nodes((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        "affine": _nodes(x_0, (1.2, 0.3, 0.0), (-0.2, 0.8, 0.0), (0.0, 0.0, 0.0)),
        "skew": _nodes(x_0, (1.2, 0.1, 0.0), (0.55, 0.9, 0.0), (0.0, 0.0, 0.0)),
        "tapered": _nodes(x_0, x_r, x_s, 0.28 * x_r),
        "distorted": _nodes(x_0, x_rd, x_sd, -0.31 * x_rd + 0.26 * x_sd),
        "warped": _nodes(x_0, x_rw, x_sw, 0.18 * x_rw - 0.12 * x_sw + 0.42 * normal),
    }


def _directors(coordinates: FloatArray) -> FloatArray:
    relative = coordinates - np.mean(coordinates, axis=0)
    x_r = 0.25 * (_XI @ relative)
    x_s = 0.25 * (_ETA @ relative)
    normal = np.cross(x_r, x_s)
    normal /= np.linalg.norm(normal)
    return np.tile(normal, (4, 1))


def _reference(coordinates: FloatArray) -> Q4ReferenceData:
    return build_reference_data(coordinates, _directors(coordinates), 0.2)


def _shape(r: float, s: float) -> tuple[FloatArray, FloatArray]:
    values = 0.25 * (1.0 + _XI * r) * (1.0 + _ETA * s)
    derivatives = np.column_stack(
        (0.25 * _XI * (1.0 + _ETA * s), 0.25 * _ETA * (1.0 + _XI * r))
    )
    return values, derivatives


def _bases(reference: Q4ReferenceData, r: float, s: float, zeta: float = 0.0) -> FloatArray:
    values, derivatives = _shape(r, s)
    relative = reference.coordinates - np.mean(reference.coordinates, axis=0)
    offsets = 0.5 * reference.thickness[:, None] * reference.directors
    return np.vstack(
        (
            derivatives[:, 0] @ relative + zeta * (derivatives[:, 0] @ offsets),
            derivatives[:, 1] @ relative + zeta * (derivatives[:, 1] @ offsets),
            values @ offsets,
        )
    )


def _characteristics(reference: Q4ReferenceData) -> tuple[FloatArray, FloatArray, FloatArray, float, float, float]:
    relative = reference.coordinates - np.mean(reference.coordinates, axis=0)
    x_r = 0.25 * (_XI @ relative)
    x_s = 0.25 * (_ETA @ relative)
    x_d = 0.25 * ((_XI * _ETA) @ relative)
    gram = np.array([[x_r @ x_r, x_r @ x_s], [x_s @ x_r, x_s @ x_s]])
    dual_coefficients = np.linalg.inv(gram)
    dual_r = dual_coefficients[0, 0] * x_r + dual_coefficients[0, 1] * x_s
    dual_s = dual_coefficients[1, 0] * x_r + dual_coefficients[1, 1] * x_s
    c_r = float(x_d @ dual_r)
    c_s = float(x_d @ dual_s)
    return x_r, x_s, x_d, c_r, c_s, c_r * c_r + c_s * c_s - 1.0


def _reciprocal(reference: Q4ReferenceData, r: float, s: float) -> FloatArray:
    return np.linalg.inv(_bases(reference, r, s, 0.0).T)


def _appendix_b_coefficients(reference: Q4ReferenceData) -> FloatArray:
    x_r, x_s, _, _, _, _ = _characteristics(reference)
    a = _reciprocal(reference, 0.0, _G)
    b = _reciprocal(reference, 0.0, -_G)
    c = _reciprocal(reference, _G, 0.0)
    d = _reciprocal(reference, -_G, 0.0)
    ar, ass = float(x_r @ a[0]), float(x_r @ a[1])
    br, bss = float(x_r @ b[0]), float(x_r @ b[1])
    cr, css = float(x_s @ c[0]), float(x_s @ c[1])
    dr, dss = float(x_s @ d[0]), float(x_s @ d[1])
    return np.array(
        [
            0.5 * (ar * ar - br * br),
            0.5 * (ass * ass - bss * bss),
            0.5 * (ar * ar + br * br),
            0.5 * (ar * ass + br * bss),
            0.5 * (ar * ass - br * bss),
            0.5 * (css * css - dss * dss),
            0.5 * (cr * cr - dr * dr),
            0.5 * (css * css + dss * dss),
            0.5 * (cr * css + dr * dss),
            0.5 * (cr * css - dr * dss),
        ]
    )


def _qrs_oracle(reference: Q4ReferenceData, r: float, s: float) -> FloatArray:
    _, _, _, c_r, c_s, denominator = _characteristics(reference)
    aa = c_r * (c_r - 1.0) / (2.0 * denominator)
    ab = c_r * (c_r + 1.0) / (2.0 * denominator)
    ac = c_s * (c_s - 1.0) / (2.0 * denominator)
    ad = c_s * (c_s + 1.0) / (2.0 * denominator)
    ae = 2.0 * c_r * c_s / denominator
    n1, n2, n3, n4, n5, m1, m2, m3, m4, m5 = _appendix_b_coefficients(reference)
    j0 = float(np.linalg.det(_bases(reference, 0.0, 0.0).T))
    point_j = float(np.linalg.det(_bases(reference, r, s).T))
    lam = j0 / point_j
    root_three = math.sqrt(3.0)
    q = np.array(
        [
            [(1 + c_r * s) ** 2, (c_s * s) ** 2, 2 * c_s * s * (1 + c_r * s)],
            [(c_r * r) ** 2, (1 + c_s * r) ** 2, 2 * c_r * r * (1 + c_s * r)],
            [c_r * r * (1 + c_r * s), c_s * s * (1 + c_s * r), c_r * c_s * r * s + (1 + c_r * s) * (1 + c_s * r)],
        ]
    )
    rr = lam * np.array(
        [
            [1 / lam + root_three * n1 * s, root_three * n2 * s, 2 * root_three * n5 * s, n3 * s, n4 * s, n1 * s / root_three],
            [root_three * m2 * r, 1 / lam + root_three * m1 * r, 2 * root_three * m5 * r, m4 * r, m3 * r, m1 * r / root_three],
            [0.0, 0.0, 1 / lam, 0.0, 0.0, 0.0],
        ]
    )
    ss = np.array(
        [
            [0.5 - aa, 0.5 - ab, -ac, -ad, -ae],
            [-aa, -ab, 0.5 - ac, 0.5 - ad, -ae],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.5, -0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, -0.5, 0.0],
            [aa, ab, ac, ad, ae],
        ]
    )
    return q @ rr @ ss


def _raw_membrane(reference: Q4ReferenceData, r: float, s: float) -> FloatArray:
    _, derivatives = _shape(r, s)
    g_r, g_s, _ = _bases(reference, r, s)
    operator = np.zeros((3, 24))
    identity = np.eye(3)
    for node in range(4):
        translation = slice(6 * node, 6 * node + 3)
        operator[0, translation] = derivatives[node, 0] * (g_r @ identity)
        operator[1, translation] = derivatives[node, 1] * (g_s @ identity)
        operator[2, translation] = 0.5 * (
            derivatives[node, 1] * (g_r @ identity)
            + derivatives[node, 0] * (g_s @ identity)
        )
    return operator


def _outer_ties(reference: Q4ReferenceData) -> FloatArray:
    return np.vstack(
        (
            _raw_membrane(reference, 0.0, 1.0)[0],
            _raw_membrane(reference, 0.0, -1.0)[0],
            _raw_membrane(reference, 1.0, 0.0)[1],
            _raw_membrane(reference, -1.0, 0.0)[1],
            _raw_membrane(reference, 0.0, 0.0)[2],
        )
    )


def _fixed_drill_oracle(reference: Q4ReferenceData, r: float, s: float) -> FloatArray:
    derivatives = np.zeros((4, 2))
    derivatives[0, 0] = -r * (1.0 - s)
    derivatives[1, 1] = -s * (1.0 + r)
    derivatives[2, 0] = -r * (1.0 + s)
    derivatives[3, 1] = -s * (1.0 - r)
    j0 = float(np.linalg.det(_bases(reference, 0.0, 0.0).T))
    point_j = float(np.linalg.det(_bases(reference, r, s).T))
    fixed = np.zeros((3, 24))
    relative = reference.coordinates - np.mean(reference.coordinates, axis=0)
    direction = reference.drill_direction
    edge_points = ((0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
    for edge, (r_edge, s_edge) in enumerate(edge_points):
        node_j = (edge + 1) % 4
        x_mid = (relative[edge] - relative[node_j]) / 8.0
        g_r, g_s, _ = _bases(reference, r_edge, s_edge)
        c_r = float(x_mid @ (-np.cross(g_r, direction)))
        c_s = float(x_mid @ np.cross(g_s, direction))
        h_r, h_s = derivatives[edge]
        values = (j0 / point_j) * np.array(
            [h_r * c_r, -h_s * c_s, 0.5 * (h_s * c_r - h_r * c_s)]
        )
        block = values[:, None] * direction
        fixed[:, 6 * edge + 3 : 6 * edge + 6] -= block
        fixed[:, 6 * node_j + 3 : 6 * node_j + 6] += block
    return fixed


def _eq21_oracle(reference: Q4ReferenceData, r: float, s: float) -> FloatArray:
    fixed = _fixed_drill_oracle(reference, r, s)
    _, _, _, c_r, c_s, _ = _characteristics(reference)
    a, b = 1.0 + s * c_r, s * c_s
    c, d = r * c_r, 1.0 + r * c_s
    tensor_map = np.array(
        [[a * a, b * b, 2 * a * b], [c * c, d * d, 2 * c * d], [a * c, b * d, a * d + b * c]]
    )
    return tensor_map @ fixed


def _local_engineering_membrane_oracle(
    reference: Q4ReferenceData,
    r: float,
    s: float,
    covariant: FloatArray,
) -> FloatArray:
    """Independently map ``[e_rr,e_ss,e_rs]`` to local engineering strain."""

    bases = _bases(reference, r, s, 0.0)
    reciprocal = np.linalg.inv(bases.T)
    local_3 = bases[2] / np.linalg.norm(bases[2])
    local_1 = np.cross(bases[1], local_3)
    local_1 /= np.linalg.norm(local_1)
    local_2 = np.cross(local_3, local_1)
    projection = np.vstack((local_1, local_2, local_3)) @ reciprocal.T

    e_rr, e_ss, e_rs = covariant
    p_1 = projection[0, :2]
    p_2 = projection[1, :2]
    return np.array(
        [
            p_1[0] ** 2 * e_rr
            + p_1[1] ** 2 * e_ss
            + 2.0 * p_1[0] * p_1[1] * e_rs,
            p_2[0] ** 2 * e_rr
            + p_2[1] ** 2 * e_ss
            + 2.0 * p_2[0] * p_2[1] * e_rs,
            2.0
            * (
                p_1[0] * p_2[0] * e_rr
                + p_1[1] * p_2[1] * e_ss
                + (p_1[0] * p_2[1] + p_1[1] * p_2[0]) * e_rs
            ),
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize("case", tuple(_case_coordinates()))
def test_eq25_matches_independent_qrs_and_full_3x24_columns(case: str) -> None:
    reference = _reference(_case_coordinates()[case])
    np.testing.assert_allclose(reference.mitc4_plus_qrs_coefficients, _appendix_b_coefficients(reference), rtol=2e-13, atol=2e-14)
    ties = _outer_ties(reference)
    for r, s in _POINTS:
        expected_map = _qrs_oracle(reference, r, s)
        np.testing.assert_allclose(_mitc4_plus_2025_qrs_map(reference, r, s), expected_map, rtol=3e-13, atol=4e-14)
        expected = expected_map @ ties
        np.testing.assert_allclose(_assumed_mitc4_plus_membrane_2025_eq25(reference, r, s), expected, rtol=4e-13, atol=5e-14)
        np.testing.assert_allclose(covariant_strain_operator(reference, r, s, 0.0)[:3], expected + _eq21_oracle(reference, r, s), rtol=5e-13, atol=7e-14)


@pytest.mark.parametrize("case", ("distorted", "warped"))
def test_eq21_off_center_double_covariant_map_all_drill_columns(case: str) -> None:
    reference = _reference(_case_coordinates()[case])
    rotation_columns = np.array([column for node in range(4) for column in range(6 * node + 3, 6 * node + 6)])
    translation_columns = np.array([column for node in range(4) for column in range(6 * node, 6 * node + 3)])
    for r, s in ((-0.77, 0.64), (0.61, -0.57), (0.91, -0.83)):
        actual = _drill_membrane_operator(reference, r, s)
        expected = _eq21_oracle(reference, r, s)
        np.testing.assert_allclose(actual[:, rotation_columns], expected[:, rotation_columns], rtol=4e-13, atol=5e-14)
        np.testing.assert_array_equal(actual[:, translation_columns], 0.0)
        assert np.linalg.norm(actual - _fixed_drill_oracle(reference, r, s)) > 1.0e-6


def test_selected_2025_field_is_not_2017_eq27_and_identity_is_v2() -> None:
    reference = _reference(_case_coordinates()["square"])
    actual = _assumed_mitc4_plus_membrane_2025_eq25(reference, _G, _G)
    legacy = _assumed_mitc4_plus_membrane_2017_eq27_reference(reference, _G, _G)
    assert np.max(np.abs(actual - legacy)) > 1.0e-2
    expected_signature = deterministic_signature(
        "mitc4_plus_d_published_2025_eq21_eq25_reference_v2",
        (reference.coordinates, reference.directors, reference.thickness),
    )
    assert reference.signature == expected_signature
    assert _reference(_case_coordinates()["square"]).signature == reference.signature
    assert _reference(_case_coordinates()["distorted"]).signature != reference.signature
    with pytest.raises(ValueError, match="shape"):
        Q4ReferenceData(**{**{field: getattr(reference, field) for field in reference.__dataclass_fields__}, "mitc4_plus_qrs_coefficients": np.zeros(9)})


def test_qrs_cyclic_reversal_and_global_rotation_covariance() -> None:
    coordinates = _case_coordinates()["distorted"]
    reference = _reference(coordinates)
    cyclic = _reference(coordinates[[1, 2, 3, 0]])
    reversal = _reference(coordinates[[0, 3, 2, 1]])
    p_cyclic = np.zeros((5, 5)); p_cyclic[0, 3] = p_cyclic[1, 2] = p_cyclic[2, 0] = p_cyclic[3, 1] = 1.0; p_cyclic[4, 4] = -1.0
    t_cyclic = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    p_reverse = np.zeros((5, 5)); p_reverse[0, 2] = p_reverse[1, 3] = p_reverse[2, 0] = p_reverse[3, 1] = p_reverse[4, 4] = 1.0
    t_reverse = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    dof_cyclic = np.zeros((24, 24))
    dof_reverse = np.zeros((24, 24))
    for new_node, old_node in enumerate((1, 2, 3, 0)):
        dof_cyclic[
            6 * new_node : 6 * new_node + 6,
            6 * old_node : 6 * old_node + 6,
        ] = np.eye(6)
    for new_node, old_node in enumerate((0, 3, 2, 1)):
        dof_reverse[
            6 * new_node : 6 * new_node + 6,
            6 * old_node : 6 * old_node + 6,
        ] = np.eye(6)
    for r, s in _POINTS:
        np.testing.assert_allclose(_mitc4_plus_2025_qrs_map(cyclic, r, s) @ p_cyclic, t_cyclic @ _mitc4_plus_2025_qrs_map(reference, -s, r), rtol=5e-13, atol=6e-14)
        np.testing.assert_allclose(_mitc4_plus_2025_qrs_map(reversal, r, s) @ p_reverse, t_reverse @ _mitc4_plus_2025_qrs_map(reference, s, r), rtol=5e-13, atol=6e-14)
        np.testing.assert_allclose(
            _drill_membrane_operator(cyclic, r, s) @ dof_cyclic,
            t_cyclic @ _drill_membrane_operator(reference, -s, r),
            rtol=6e-13,
            atol=8e-14,
        )
        np.testing.assert_allclose(
            _drill_membrane_operator(reversal, r, s) @ dof_reverse,
            t_reverse @ _drill_membrane_operator(reference, s, r),
            rtol=6e-13,
            atol=8e-14,
        )
    axis = np.array([0.3, -0.4, 0.5]); axis /= np.linalg.norm(axis); angle = 0.73
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rotation = np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    rotated = _reference(coordinates @ rotation.T)
    dof_rotation = np.zeros((24, 24))
    for node in range(4):
        dof_rotation[6 * node : 6 * node + 3, 6 * node : 6 * node + 3] = rotation
        dof_rotation[6 * node + 3 : 6 * node + 6, 6 * node + 3 : 6 * node + 6] = rotation
    for r, s in _POINTS:
        np.testing.assert_allclose(covariant_strain_operator(rotated, r, s, 0.0) @ dof_rotation, covariant_strain_operator(reference, r, s, 0.0), rtol=8e-13, atol=1e-13)


def test_patch_objectivity_scalar_generalized_and_engineering_shear() -> None:
    coordinates = _case_coordinates()["affine"]
    reference = _reference(coordinates)
    displacement = np.zeros(24)
    gradient = np.array([[0.013, 0.021, 0.0], [-0.007, -0.009, 0.0], [0.0, 0.0, 0.0]])
    for node in range(4):
        displacement[6 * node : 6 * node + 3] = gradient @ coordinates[node]
    strain_tensor = 0.5 * (gradient + gradient.T)
    normal = reference.drill_direction
    local_1 = np.cross(reference.center_covariant[1], normal)
    local_1 /= np.linalg.norm(local_1)
    local_2 = np.cross(normal, local_1)
    expected = np.array(
        [
            local_1 @ strain_tensor @ local_1,
            local_2 @ strain_tensor @ local_2,
            2.0 * (local_1 @ strain_tensor @ local_2),
            0.0,
            0.0,
        ]
    )
    for r, s in _POINTS:
        operator, _ = local_strain_operator(reference, r, s, 0.0)
        np.testing.assert_allclose(operator @ displacement, expected, rtol=3e-13, atol=3e-14)
        np.testing.assert_allclose(strain_at(reference, displacement, r, s, 0.0), expected, rtol=3e-13, atol=3e-14)
        membrane, bending, shear = generalized_strain_operators(reference, r, s)
        scalar_values = generalized_strain_at(reference, displacement, r, s)
        for actual, expected_operator in zip(scalar_values, (membrane, bending, shear), strict=True):
            np.testing.assert_allclose(actual, expected_operator @ displacement, rtol=2e-14, atol=2e-15)
    omega = np.array([0.17, -0.11, 0.23])
    rigid = np.zeros(24)
    for node in range(4):
        rigid[6 * node : 6 * node + 3] = np.cross(omega, coordinates[node])
        rigid[6 * node + 3 : 6 * node + 6] = omega
    for r, s in _POINTS:
        assert np.linalg.norm(covariant_strain_operator(reference, r, s, 0.0) @ rigid) < 2.0e-15

    square = _reference(_case_coordinates()["square"])
    bending_dofs = np.zeros(24)
    shear_dofs = np.zeros(24)
    for node in range(4):
        x = square.coordinates[node, 0]
        bending_dofs[6 * node + 4] = 0.3 * x
        shear_dofs[6 * node + 2] = 0.4 * x
    for r, s in _POINTS:
        membrane, bending, shear = generalized_strain_operators(square, r, s)
        np.testing.assert_allclose(membrane @ bending_dofs, 0.0, atol=2e-15)
        np.testing.assert_allclose(bending @ bending_dofs, [0.3, 0.0, 0.0], rtol=2e-14, atol=2e-15)
        np.testing.assert_allclose(shear @ bending_dofs, 0.0, atol=2e-15)
        np.testing.assert_allclose(membrane @ shear_dofs, 0.0, atol=2e-15)
        np.testing.assert_allclose(bending @ shear_dofs, 0.0, atol=2e-15)
        np.testing.assert_allclose(shear @ shear_dofs, [0.4, 0.0], rtol=2e-14, atol=2e-15)


def _displacement_oracle(reference: Q4ReferenceData, r: float, s: float, zeta: float) -> FloatArray:
    values, _ = _shape(r, s)
    result = np.zeros((3, 24)); identity = np.eye(3)
    for node in range(4):
        result[:, 6 * node : 6 * node + 3] = values[node] * identity
        director = reference.directors[node]
        cross = np.array([[0.0, -director[2], director[1]], [director[2], 0.0, -director[0]], [-director[1], director[0], 0.0]])
        result[:, 6 * node + 3 : 6 * node + 6] = -zeta * 0.5 * reference.thickness[node] * values[node] * cross
    midside = 0.5 * np.array([(1-r*r)*(1-s), (1-s*s)*(1+r), (1-r*r)*(1+s), (1-s*s)*(1-r)])
    dual_r, dual_s = reference.center_dual
    for edge in range(4):
        node_j = (edge + 1) % 4
        c_r, c_s = reference.drill_edge_coefficients[edge]
        block = midside[edge] * np.outer(dual_r * c_r - dual_s * c_s, reference.drill_direction)
        result[:, 6 * edge + 3 : 6 * edge + 6] -= block
        result[:, 6 * node_j + 3 : 6 * node_j + 6] += block
    return result


def test_eq21_does_not_change_displacement_or_consistent_mass_path() -> None:
    reference = _reference(_case_coordinates()["warped"])
    expected_mass = np.zeros((24, 24))
    for surface in range(4):
        for thickness in range(2):
            r, s, zeta = reference.volume_points[surface, thickness]
            expected = _displacement_oracle(reference, float(r), float(s), float(zeta))
            np.testing.assert_allclose(displacement_operator(reference, r, s, zeta), expected, rtol=3e-13, atol=4e-14)
            np.testing.assert_allclose(reference.volume_displacement_operators[surface, thickness], expected, rtol=3e-13, atol=4e-14)
            expected_mass += reference.volume_weights[surface, thickness] * (expected.T @ expected)
    np.testing.assert_allclose(consistent_mass(reference, 1.0), expected_mass, rtol=3e-13, atol=4e-14)


def test_eq21_planar_distorted_pure_drill_maps_to_local_engineering() -> None:
    reference = _reference(_case_coordinates()["distorted"])
    r, s = 0.61, -0.57
    displacement = np.zeros(24)
    drill_values = (0.2, -0.5, 0.7, -0.1)
    for node, value in enumerate(drill_values):
        displacement[6 * node + 3 : 6 * node + 6] = (
            value * reference.drill_direction
        )

    covariant = _eq21_oracle(reference, r, s) @ displacement
    assert np.linalg.norm(covariant) > 1.0e-4
    expected_membrane = _local_engineering_membrane_oracle(
        reference, r, s, covariant
    )
    operator, _ = local_strain_operator(reference, r, s, 0.0)
    actual = operator @ displacement
    np.testing.assert_allclose(
        actual[:3], expected_membrane, rtol=5e-13, atol=7e-14
    )
    np.testing.assert_allclose(actual[3:], 0.0, atol=3e-15)


@pytest.mark.parametrize("case", tuple(_case_coordinates()))
def test_six_family_stiffness_is_symmetric_without_negative_modes(case: str) -> None:
    reference = _reference(_case_coordinates()[case])
    constitutive = isotropic_plane_stress_constitutive(210.0e9, 0.3)
    stiffness = linear_stiffness(reference, constitutive)
    symmetry_error = np.linalg.norm(stiffness - stiffness.T) / np.linalg.norm(
        stiffness
    )
    assert symmetry_error <= 5.0e-14

    eigenvalues = np.linalg.eigvalsh(0.5 * (stiffness + stiffness.T))
    eigenvalue_tolerance = 1.0e-10 * float(np.max(np.abs(eigenvalues)))
    assert not np.any(eigenvalues < -eigenvalue_tolerance)


def test_invalid_mapping_fails_closed_and_sources_have_no_geometry_dependency() -> None:
    coordinates = _case_coordinates()["distorted"]
    directors = _directors(coordinates)
    with pytest.raises(ValueError, match="reverse|positive"):
        build_reference_data(coordinates[[0, 3, 2, 1]], directors, 0.2)
    import anyfesolver.shell_formulations.mitc4_plus_d_reference as reference_module
    import anyfesolver.shell_formulations.mitc4_plus_d_scalar as scalar_module
    for module in (reference_module, scalar_module):
        source = inspect.getsource(module).lower()
        assert "import anygeometry" not in source
        assert "from anygeometry" not in source
        assert "geometrydocument" not in source
