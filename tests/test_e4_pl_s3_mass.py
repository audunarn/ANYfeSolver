from __future__ import annotations

import itertools
from fractions import Fraction
from math import factorial

import numpy as np
import pytest

from anyfesolver.e4_pl_s3_element import (
    MASS_MOMENT_ID,
    TRIANGLE_QUADRATURE,
    QualifiedE4PLS3ShellElement,
    _analytic_mass_moments,
)
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver.shell_sections import GeneralizedShellSection


def _mesh(nodes: np.ndarray) -> FEMesh:
    mesh = FEMesh()
    for identifier, coordinate in enumerate(nodes, start=1):
        mesh.add_node(identifier, *coordinate)
    return mesh


def _material(*, density: float = 7850.0) -> Material:
    return Material("steel", 210.0e9, 0.3, density=density)


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right, ord=np.inf)
        / max(np.linalg.norm(right, ord=np.inf), 1.0)
    )


def _barycentric_monomial(
    area: Fraction, powers: tuple[int, int, int]
) -> Fraction:
    """Integrate a barycentric monomial without using production quadrature."""

    degree = sum(powers)
    numerator = 2 * area
    for power in powers:
        numerator *= factorial(power)
    return numerator / factorial(degree + 2)


def _mass_moment_oracle(
    area: Fraction,
) -> tuple[list[list[Fraction]], list[Fraction], Fraction]:
    """Independent exact moments for ``(L1, L2, L3, 27 L1 L2 L3)``."""

    corner: list[list[Fraction]] = []
    for left in range(3):
        row = []
        for right in range(3):
            powers = [0, 0, 0]
            powers[left] += 1
            powers[right] += 1
            row.append(_barycentric_monomial(area, tuple(powers)))
        corner.append(row)
    corner_bubble = []
    for corner_index in range(3):
        powers = [1, 1, 1]
        powers[corner_index] += 1
        corner_bubble.append(27 * _barycentric_monomial(area, tuple(powers)))
    bubble = 27**2 * _barycentric_monomial(area, (2, 2, 2))
    return corner, corner_bubble, bubble


def _section(*, mass_per_area: float, rotary_inertia_per_area: float) -> GeneralizedShellSection:
    return GeneralizedShellSection(
        A=np.asarray(((120.0, 12.0, 4.0), (12.0, 95.0, -3.0), (4.0, -3.0, 42.0))),
        B=np.asarray(((1.2, 0.1, 0.0), (0.1, -0.8, 0.1), (0.0, 0.1, 0.3))),
        D=np.asarray(((15.0, 1.0, 0.2), (1.0, 11.0, -0.1), (0.2, -0.1, 5.0))),
        As=np.asarray(((25.0, 2.0), (2.0, 20.0))),
        mass_per_area=mass_per_area,
        rotary_inertia_per_area=rotary_inertia_per_area,
    )


def test_degree_six_bubble_moments_are_closed_form() -> None:
    exact_area = Fraction(29, 4)
    exact_corner, exact_corner_bubble, exact_bubble = _mass_moment_oracle(exact_area)
    area = float(exact_area)
    corner, corner_bubble, bubble = _analytic_mass_moments(area)

    np.testing.assert_array_equal(
        corner,
        np.asarray(exact_corner, dtype=float),
    )
    np.testing.assert_array_equal(
        corner_bubble, np.asarray(exact_corner_bubble, dtype=float)
    )
    assert bubble == float(exact_bubble)
    assert exact_corner[0][0] == exact_area / 6
    assert exact_corner[0][1] == exact_area / 12
    assert exact_corner_bubble == [3 * exact_area / 20] * 3
    assert exact_bubble == 81 * exact_area / 280
    assert float(np.sum(corner_bubble)) == pytest.approx(9.0 * area / 20.0)
    with pytest.raises(ValueError, match="positive finite area"):
        _analytic_mass_moments(0.0)


def test_degree_five_stiffness_rule_is_insufficient_for_bubble_squared() -> None:
    # The published seven-point rule integrates every barycentric monomial
    # through degree five, but b**2 has degree six and must not use this rule.
    for degree in range(6):
        for left in range(degree + 1):
            for middle in range(degree - left + 1):
                right = degree - left - middle
                approximate = 2.0 * sum(
                    weight
                    * (1.0 - r - s) ** left
                    * r**middle
                    * s**right
                    for r, s, weight in TRIANGLE_QUADRATURE
                )
                exact = float(
                    _barycentric_monomial(Fraction(1), (left, middle, right))
                )
                assert approximate == pytest.approx(exact, rel=0.0, abs=5.0e-15)

    quadrature_bubble = 2.0 * sum(
        weight * (27.0 * (1.0 - r - s) * r * s) ** 2
        for r, s, weight in TRIANGLE_QUADRATURE
    )
    exact_bubble = float(_mass_moment_oracle(Fraction(1))[2])
    assert quadrature_bubble == pytest.approx(float(Fraction(72, 245)), abs=8.0e-15)
    assert quadrature_bubble > exact_bubble
    assert quadrature_bubble / exact_bubble == pytest.approx(
        float(Fraction(64, 63)), rel=0.0, abs=3.0e-14
    )
    assert (quadrature_bubble - exact_bubble) / exact_bubble == pytest.approx(
        1.0 / 63.0, rel=0.0, abs=3.0e-14
    )


def test_full_nodal_bubble_mass_and_guyan_reduction() -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.5, 1.5, 0.0)))
    material = _material()
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.2,
        reference_normal=np.asarray((0.0, 0.0, 1.0)),
    )

    made = element.compute_mass_components(_mesh(nodes), material)

    assert made["mass_moment_id"] == MASS_MOMENT_ID
    assert made["full_local"].shape == (20, 20)
    assert made["guyan"].shape == (20, 18)
    assert made["condensed_local"].shape == (18, 18)
    assert made["global"].shape == (18, 18)
    assert made["full_rank"] == 17
    assert made["condensed_rank"] == 15
    np.testing.assert_allclose(
        made["condensed_local"],
        made["guyan"].T @ made["full_local"] @ made["guyan"],
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(made["global"], made["global"].T, rtol=0.0, atol=0.0)

    drill = np.asarray((5, 11, 17), dtype=np.intp)
    np.testing.assert_array_equal(made["full_local"][drill], np.zeros((3, 20)))
    np.testing.assert_array_equal(made["full_local"][:, drill], np.zeros((20, 3)))
    np.testing.assert_array_equal(made["condensed_local"][drill], np.zeros((3, 18)))
    np.testing.assert_array_equal(made["condensed_local"][:, drill], np.zeros((18, 3)))

    expected_total = made["mass_per_area"] * made["area"]
    for component in range(3):
        indices = np.asarray([6 * node + component for node in range(3)])
        assert float(np.sum(made["condensed_local"][np.ix_(indices, indices)])) == pytest.approx(
            expected_total,
            rel=2.0e-15,
        )
    eigenvalues = np.linalg.eigvalsh(made["condensed_local"])
    assert eigenvalues[3] > 0.0
    assert eigenvalues[0] >= -2.0e-12 * eigenvalues[-1]


def test_mass_uses_generalized_section_areal_properties() -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.2, 1.0, 0.0)))
    section = GeneralizedShellSection(
        A=np.diag((120.0, 110.0, 45.0)),
        B=np.zeros((3, 3)),
        D=np.diag((14.0, 12.0, 5.0)),
        As=np.diag((30.0, 25.0)),
        mass_per_area=12.5,
        rotary_inertia_per_area=0.75,
    )
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.2,
        shell_section=section,
        material_direction=np.asarray((1.0, 0.0, 0.0)),
        reference_normal=np.asarray((0.0, 0.0, 1.0)),
    )

    made = element.compute_mass_components(_mesh(nodes), _material(density=1.0))

    assert made["mass_per_area"] == 12.5
    assert made["rotary_inertia_per_area"] == 0.75
    translations = np.asarray([6 * node for node in range(3)])
    assert float(
        np.sum(made["condensed_local"][np.ix_(translations, translations)])
    ) == pytest.approx(12.5 * made["area"], rel=2.0e-15)


def test_zero_areal_properties_produce_zero_mass_and_zero_ranks() -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)))
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.2,
        shell_section=_section(mass_per_area=0.0, rotary_inertia_per_area=0.0),
        material_direction=np.asarray((1.0, 0.2, 0.0)),
        reference_normal=np.asarray((0.0, 0.0, 1.0)),
    )

    made = element.compute_mass_components(_mesh(nodes), _material())

    assert made["mass_per_area"] == 0.0
    assert made["rotary_inertia_per_area"] == 0.0
    assert made["full_rank"] == 0
    assert made["condensed_rank"] == 0
    np.testing.assert_array_equal(made["full_local"], np.zeros((20, 20)))
    np.testing.assert_array_equal(made["condensed_local"], np.zeros((18, 18)))
    np.testing.assert_array_equal(made["global"], np.zeros((18, 18)))
    np.testing.assert_array_equal(element.compute_mass_matrix(_mesh(nodes), _material()), 0.0)


def test_rigid_translation_and_rotation_have_exact_consistent_mass_work() -> None:
    nodes = np.asarray(
        ((0.1, 0.2, 0.3), (1.2, 0.1, 0.5), (0.3, 1.0, 0.4)), dtype=float
    )
    owner = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
    normal = owner / np.linalg.norm(owner)
    section = _section(mass_per_area=12.5, rotary_inertia_per_area=0.75)
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.2,
        shell_section=section,
        material_direction=np.asarray((1.0, 0.2, 0.1)),
        reference_normal=owner,
    )
    made = element.compute_mass_components(_mesh(nodes), _material(density=1.0))
    corner = np.asarray(_mass_moment_oracle(Fraction(1))[0], dtype=float) * made["area"]

    motions = (
        (np.asarray((0.7, -0.2, 0.4)), np.zeros(3)),
        (np.zeros(3), 0.37 * normal),
        (np.asarray((-0.1, 0.3, 0.2)), 0.23 * (nodes[1] - nodes[0])),
    )
    for translation, omega in motions:
        velocities = translation + np.cross(np.broadcast_to(omega, nodes.shape), nodes)
        coordinates = np.zeros(18, dtype=float)
        for node in range(3):
            coordinates[6 * node : 6 * node + 3] = velocities[node]
            coordinates[6 * node + 3 : 6 * node + 6] = omega

        translational_work = made["mass_per_area"] * float(
            np.einsum("ia,ij,ja->", velocities, corner, velocities)
        )
        tangent_omega = omega - float(np.dot(omega, normal)) * normal
        rotary_work = (
            made["rotary_inertia_per_area"]
            * made["area"]
            * float(np.dot(tangent_omega, tangent_omega))
        )
        actual = float(coordinates @ made["global"] @ coordinates)
        assert actual == pytest.approx(
            translational_work + rotary_work, rel=2.0e-13, abs=2.0e-12
        )


def test_all_six_d3_numberings_transport_condensed_mass() -> None:
    nodes = np.asarray(
        ((0.1, 0.2, 0.3), (1.2, 0.1, 0.5), (0.3, 1.0, 0.4)), dtype=float
    )
    owner = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
    material = _material()

    def mass(numbered: np.ndarray) -> np.ndarray:
        element = QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.2,
            reference_normal=owner,
        )
        return element._compute_mass_components(
            _mesh(numbered),
            material,
            enforce_positive_winding=False,
        )["global"]

    baseline = mass(nodes)
    for permutation in itertools.permutations(range(3)):
        actual = mass(nodes[list(permutation)])
        dofs = [6 * node + component for node in permutation for component in range(6)]
        expected = baseline[np.ix_(dofs, dofs)]
        assert _relative(actual, expected) < 4.0e-14, permutation


def test_full_mass_and_guyan_map_are_d3_covariant() -> None:
    nodes = np.asarray(
        ((0.1, 0.2, 0.3), (1.2, 0.1, 0.5), (0.3, 1.0, 0.4)), dtype=float
    )
    owner = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
    section = _section(mass_per_area=12.5, rotary_inertia_per_area=0.75)

    def components(numbered: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
        element = QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.2,
            shell_section=section,
            material_direction=np.asarray((1.0, 0.2, 0.1)),
            reference_normal=owner,
        )
        made = element._compute_mass_components(
            _mesh(numbered), _material(density=1.0), enforce_positive_winding=False
        )
        stiffness = element._compute_stiffness_components(
            _mesh(numbered), _material(density=1.0), enforce_positive_winding=False
        )
        return made, np.asarray(stiffness["frame"])

    baseline, baseline_frame = components(nodes)
    for permutation in itertools.permutations(range(3)):
        actual, actual_frame = components(nodes[list(permutation)])
        frame_change = actual_frame.T @ baseline_frame
        external_change = np.zeros((18, 18), dtype=float)
        for numbered_node, baseline_node in enumerate(permutation):
            for field_offset in (0, 3):
                rows = slice(6 * numbered_node + field_offset, 6 * numbered_node + field_offset + 3)
                columns = slice(
                    6 * baseline_node + field_offset,
                    6 * baseline_node + field_offset + 3,
                )
                external_change[rows, columns] = frame_change
        full_change = np.zeros((20, 20), dtype=float)
        full_change[:18, :18] = external_change
        full_change[18:, 18:] = frame_change[:2, :2]

        expected_full = full_change @ baseline["full_local"] @ full_change.T
        expected_guyan = full_change @ baseline["guyan"] @ external_change.T
        assert _relative(actual["full_local"], expected_full) < 5.0e-14, permutation
        assert _relative(actual["guyan"], expected_guyan) < 3.0e-12, permutation


def test_mass_rebuilds_after_geometry_revision() -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)))
    mesh = _mesh(nodes)
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.1,
        reference_normal=np.asarray((0.0, 0.0, 1.0)),
    )
    material = _material()
    before = element.compute_mass_matrix(mesh, material).copy()

    mesh.set_node_coordinates(3, 0.2, 1.1, 0.0)
    after = element.compute_mass_matrix(mesh, material)

    assert not np.array_equal(after, before)
    assert float(np.sum(after[0::6, 0::6])) > float(np.sum(before[0::6, 0::6]))
