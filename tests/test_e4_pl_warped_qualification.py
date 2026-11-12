from __future__ import annotations

import numpy as np
import pytest

import anyfesolver.e4_pl_element as e4_pl_module
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement, equation7_frame
from anyfesolver.elements import ShellElement
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver.materials import OrthotropicMaterial
from anyfesolver.shell_sections import GeneralizedShellSection


OPERATIONS = (
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (1, 0, 3, 2),
    (3, 2, 1, 0),
    (0, 3, 2, 1),
    (2, 1, 0, 3),
)

WARPED_GEOMETRIES = (
    np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.08), (1.0, 1.0, -0.04), (0.0, 1.0, 0.03))),
    np.asarray(((0.0, 0.0, 0.0), (1.4, -0.1, -0.06), (1.1, 0.9, 0.11), (-0.2, 0.7, -0.03))),
    np.asarray(((-0.7, -0.4, 0.12), (0.9, -0.5, -0.08), (0.6, 0.8, 0.16), (-0.8, 0.6, -0.11))),
)


def _mesh(nodes: np.ndarray) -> FEMesh:
    mesh = FEMesh()
    for identifier, coordinates in enumerate(nodes, start=1):
        mesh.add_node(identifier, *coordinates)
    return mesh


def _material() -> Material:
    return Material("warped", 210.0e9, 0.29, density=7850.0)


def _element(identifier: int = 1) -> QualifiedE4PLShellElement:
    return QualifiedE4PLShellElement(
        identifier,
        [1, 2, 3, 4],
        "warped",
        thickness=0.025,
        warped_formulation="varying_frame",
    )


def _relative_residual(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(right, ord=np.inf)), 1.0)
    return float(np.linalg.norm(left - right, ord=np.inf) / scale)


def _owner_normal(nodes: np.ndarray) -> np.ndarray:
    normal = np.cross(nodes[1] - nodes[0], nodes[3] - nodes[0])
    return normal / np.linalg.norm(normal)


def _warped_generalized_section() -> GeneralizedShellSection:
    return GeneralizedShellSection(
        A=np.asarray(((120.0, 12.0, 4.0), (12.0, 95.0, -3.0), (4.0, -3.0, 42.0))),
        B=np.asarray(((1.2, 0.1, -0.2), (0.05, -0.8, 0.1), (0.3, -0.15, 0.4))),
        D=np.asarray(((15.0, 1.0, 0.2), (1.0, 11.0, -0.1), (0.2, -0.1, 5.0))),
        As=np.asarray(((25.0, 1.5), (1.5, 20.0))),
        mass_per_area=8.0,
        rotary_inertia_per_area=0.02,
    )


def _station_positions(element: QualifiedE4PLShellElement, nodes: np.ndarray) -> np.ndarray:
    return np.asarray(
        [element.compute_shape_functions(*point)[0] @ nodes for point in element.gauss_points],
        dtype=float,
    )


def _physical_station_order(reference: np.ndarray, actual: np.ndarray) -> list[int]:
    order = [int(np.argmin(np.linalg.norm(actual - point, axis=1))) for point in reference]
    assert len(set(order)) == len(order)
    return order


@pytest.mark.parametrize("nodes", WARPED_GEOMETRIES)
def test_warped_varying_frame_has_exactly_six_rigid_modes_and_positive_quotient(
    nodes: np.ndarray,
) -> None:
    mesh = _mesh(nodes)
    element = _element()
    components = element.compute_stiffness_components(mesh, _material())
    stiffness = np.asarray(components["total"])
    assert components["warped_direct"] is True
    assert components["legacy_fallback"] is False
    assert components["warped_formulation"] == "varying_frame"
    assert np.linalg.norm(stiffness - stiffness.T, ord=np.inf) <= 2.0e-12 * max(
        np.linalg.norm(stiffness, ord=np.inf), 1.0
    )

    rigid = element._rigid_body_mode_matrix(nodes)
    assert rigid.shape == (24, 6)
    scale = max(float(np.linalg.norm(stiffness, ord=2)), 1.0)
    assert np.linalg.norm(stiffness @ rigid, ord=np.inf) <= 2.0e-12 * scale
    basis, _ = np.linalg.qr(rigid, mode="complete")
    quotient = basis[:, 6:].T @ stiffness @ basis[:, 6:]
    quotient = 0.5 * (quotient + quotient.T)
    eigenvalues = np.linalg.eigvalsh(quotient)
    assert eigenvalues[0] > 1.0e-9 * max(float(eigenvalues[-1]), 1.0)


@pytest.mark.parametrize("components_first", (True, False))
def test_warped_component_and_matrix_cache_orders_preserve_public_contract(
    components_first: bool,
) -> None:
    mesh = _mesh(WARPED_GEOMETRIES[1])
    material = _material()
    element = _element()

    if components_first:
        components = element.compute_stiffness_components(mesh, material)
        stiffness = element.compute_stiffness_matrix(mesh, material)
    else:
        stiffness = element.compute_stiffness_matrix(mesh, material)
        components = element.compute_stiffness_components(mesh, material)

    total = components["total"]
    assert element.compute_stiffness_components(mesh, material) is components
    np.testing.assert_array_equal(stiffness, total)
    assert not stiffness.flags.writeable
    assert not total.flags.writeable
    assert np.linalg.norm(stiffness - stiffness.T, ord=np.inf) <= 2.0e-12 * max(
        np.linalg.norm(stiffness, ord=np.inf), 1.0
    )

    repeated = element.compute_stiffness_matrix(mesh, material)
    assert repeated is not stiffness
    assert not repeated.flags.writeable
    np.testing.assert_array_equal(repeated, stiffness)

    assembly_fast = e4_pl_module._try_q4_fast_assembly_cached_stiffness(
        element,
        mesh,
        material,
    )
    if np.array_equal(total, total.T):
        assert type(assembly_fast) is bytes
    else:
        # The total-only fast path requires bitwise symmetry.  A legitimate
        # varying-frame total that differs only by binary64 accumulation
        # roundoff remains on the fully guarded component route.
        assert assembly_fast is None


@pytest.mark.parametrize("nodes", WARPED_GEOMETRIES)
def test_warped_d4_numberings_are_operator_congruent(nodes: np.ndarray) -> None:
    material = _material()
    base = _element().compute_stiffness_matrix(_mesh(nodes), material)
    for operation in OPERATIONS:
        numbered = nodes[list(operation)]
        actual = _element().compute_stiffness_matrix(_mesh(numbered), material)
        dofs = [6 * node + component for node in operation for component in range(6)]
        expected = base[np.ix_(dofs, dofs)]
        assert _relative_residual(actual, expected) < 5.0e-13


@pytest.mark.parametrize("nodes", WARPED_GEOMETRIES)
def test_warped_proper_global_rotation_is_covariant(nodes: np.ndarray) -> None:
    angle = 0.71
    axis = np.asarray((0.3, -0.4, 0.7), dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    material = _material()
    base = _element().compute_stiffness_matrix(_mesh(nodes), material)
    transformed_nodes = nodes @ rotation.T + np.asarray((2.3, -1.1, 0.8))
    actual = _element().compute_stiffness_matrix(_mesh(transformed_nodes), material)
    transform = np.zeros((24, 24))
    for node in range(4):
        transform[6 * node : 6 * node + 3, 6 * node : 6 * node + 3] = rotation
        transform[6 * node + 3 : 6 * node + 6, 6 * node + 3 : 6 * node + 6] = rotation
    expected = transform @ base @ transform.T
    assert _relative_residual(actual, expected) < 2.0e-12


@pytest.mark.parametrize("nodes", WARPED_GEOMETRIES)
def test_warped_direct_kernel_retains_established_shell_response(nodes: np.ndarray) -> None:
    mesh = _mesh(nodes)
    material = _material()
    candidate = _element()
    established = ShellElement(2, [1, 2, 3, 4], "warped", thickness=0.025)
    candidate_stiffness = candidate.compute_stiffness_matrix(mesh, material)
    established_stiffness = established.compute_stiffness_matrix(mesh, material)
    assert np.array_equal(candidate_stiffness, established_stiffness)

    displacement = np.linspace(-3.0e-4, 2.0e-4, 24)
    candidate_force, candidate_tangent, _ = candidate.compute_nonlinear_response(
        mesh, material, displacement, tangent=True
    )
    established_force, established_tangent, _ = established.compute_nonlinear_response(
        mesh, material, displacement, tangent=True
    )
    np.testing.assert_array_equal(candidate_force, established_force)
    np.testing.assert_array_equal(candidate_tangent, established_tangent)
    for return_global in (False, True):
        candidate_recovery = candidate.compute_stresses(
            mesh,
            displacement,
            material,
            return_global=return_global,
        )
        established_recovery = established.compute_stresses(
            mesh,
            displacement,
            material,
            return_global=return_global,
        )
        assert candidate_recovery.keys() == established_recovery.keys()
        for name, values in candidate_recovery.items():
            if isinstance(values, np.ndarray):
                np.testing.assert_array_equal(values, established_recovery[name])
            else:
                assert values == established_recovery[name]


def test_warped_direct_kernel_covers_orthotropic_and_generalized_sections() -> None:
    nodes = WARPED_GEOMETRIES[1]
    mesh = _mesh(nodes)
    orthotropic = OrthotropicMaterial(
        name="orthotropic",
        elastic_modulus_1=150.0e9,
        elastic_modulus_2=10.0e9,
        elastic_modulus_3=8.0e9,
        poisson_ratio_12=0.25,
        poisson_ratio_13=0.20,
        poisson_ratio_23=0.30,
        shear_modulus_12=5.0e9,
        shear_modulus_13=4.0e9,
        shear_modulus_23=3.0e9,
        density=1600.0,
    )
    direction = np.asarray((1.0, 0.3, -0.1))
    candidate = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "orthotropic",
        thickness=0.018,
        material_direction=direction,
    )
    established = ShellElement(
        2,
        [1, 2, 3, 4],
        "orthotropic",
        thickness=0.018,
        material_direction=direction,
    )
    np.testing.assert_array_equal(
        candidate.compute_stiffness_matrix(mesh, orthotropic),
        established.compute_stiffness_matrix(mesh, orthotropic),
    )

    section = GeneralizedShellSection(
        A=np.asarray(((120.0, 12.0, 4.0), (12.0, 95.0, -3.0), (4.0, -3.0, 42.0))),
        B=np.asarray(((1.2, 0.1, 0.0), (0.05, -0.8, 0.1), (0.0, 0.05, 0.3))),
        D=np.asarray(((15.0, 1.0, 0.2), (1.0, 11.0, -0.1), (0.2, -0.1, 5.0))),
        As=np.diag((25.0, 20.0)),
        mass_per_area=8.0,
        rotary_inertia_per_area=0.02,
    )
    candidate_section = QualifiedE4PLShellElement(
        3,
        [1, 2, 3, 4],
        "warped",
        shell_section=section,
        reference_normal=np.asarray((0.0, 0.0, 1.0)),
        drilling_stabilization=0.0,
    )
    established_section = ShellElement(
        4,
        [1, 2, 3, 4],
        "warped",
        shell_section=section,
        drilling_stabilization=0.0,
    )
    material = _material()
    np.testing.assert_array_equal(
        candidate_section.compute_stiffness_matrix(mesh, material),
        established_section.compute_stiffness_matrix(mesh, material),
    )
    np.testing.assert_array_equal(
        candidate_section.compute_mass_matrix(mesh, material),
        established_section.compute_mass_matrix(mesh, material),
    )


@pytest.mark.parametrize("director_polarity", (-1, 1))
def test_warped_b_coupled_section_is_all_d4_physically_covariant(
    director_polarity: int,
) -> None:
    nodes = WARPED_GEOMETRIES[1]
    authority = _owner_normal(nodes)
    material_direction = nodes[1] - nodes[0]
    section = _warped_generalized_section()
    material = _material()
    displacement = np.linspace(-2.0e-4, 3.0e-4, 24)

    def make() -> QualifiedE4PLShellElement:
        return QualifiedE4PLShellElement(
            11,
            [1, 2, 3, 4],
            "warped",
            shell_section=section,
            material_direction=material_direction,
            reference_normal=authority,
            director_polarity=director_polarity,
        )

    baseline_element = make()
    baseline_mesh = _mesh(nodes)
    baseline_stiffness = baseline_element.compute_stiffness_matrix(
        baseline_mesh, material
    )
    baseline_mass = baseline_element.compute_mass_matrix(baseline_mesh, material)
    baseline_force, baseline_tangent, _state = (
        baseline_element.compute_nonlinear_response(
            baseline_mesh,
            material,
            displacement,
            tangent=True,
        )
    )
    assert baseline_tangent is not None
    baseline_recovery = baseline_element.compute_stresses(
        baseline_mesh,
        displacement,
        material,
        return_global=True,
    )
    baseline_positions = _station_positions(baseline_element, nodes)

    for operation in OPERATIONS:
        slots = np.asarray(operation, dtype=int)
        numbered_nodes = nodes[slots]
        numbered_displacement = displacement.reshape(4, 6)[slots].reshape(24)
        element = make()
        mesh = _mesh(numbered_nodes)
        dofs = np.asarray(
            [6 * int(slot) + component for slot in slots for component in range(6)],
            dtype=int,
        )
        np.testing.assert_allclose(
            element.compute_stiffness_matrix(mesh, material),
            baseline_stiffness[np.ix_(dofs, dofs)],
            rtol=0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            element.compute_mass_matrix(mesh, material),
            baseline_mass[np.ix_(dofs, dofs)],
            rtol=0.0,
            atol=2.0e-14,
        )
        force, tangent, _state = element.compute_nonlinear_response(
            mesh,
            material,
            numbered_displacement,
            tangent=True,
        )
        assert tangent is not None
        np.testing.assert_allclose(
            force,
            baseline_force[dofs],
            rtol=0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            tangent,
            baseline_tangent[np.ix_(dofs, dofs)],
            rtol=0.0,
            atol=2.0e-12,
        )
        recovery = element.compute_stresses(
            mesh,
            numbered_displacement,
            material,
            return_global=True,
        )
        order = _physical_station_order(
            baseline_positions,
            _station_positions(element, numbered_nodes),
        )
        for key in (
            "global_membrane_resultant_tensors",
            "global_bending_resultant_tensors",
            "global_transverse_shear_resultants",
            "physical_directors",
        ):
            np.testing.assert_allclose(
                recovery[key][order],
                baseline_recovery[key],
                rtol=0.0,
                atol=2.0e-12,
            )
        assert recovery["physical_director_authoritative"] is True
        assert recovery["warped_direct"] is True


def test_warped_director_reversal_changes_b_coupling_but_not_mass() -> None:
    nodes = WARPED_GEOMETRIES[1]
    authority = _owner_normal(nodes)
    section = _warped_generalized_section()
    material = _material()

    def make(polarity: int) -> QualifiedE4PLShellElement:
        return QualifiedE4PLShellElement(
            12,
            [1, 2, 3, 4],
            "warped",
            shell_section=section,
            material_direction=nodes[1] - nodes[0],
            reference_normal=authority,
            director_polarity=polarity,
        )

    mesh = _mesh(nodes)
    positive = make(1)
    negative = make(-1)
    np.testing.assert_array_equal(
        positive.compute_mass_matrix(mesh, material),
        negative.compute_mass_matrix(mesh, material),
    )
    assert not np.allclose(
        positive.compute_stiffness_matrix(mesh, material),
        negative.compute_stiffness_matrix(mesh, material),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    positive_recovery = positive.compute_stresses(
        mesh, np.zeros(24), material, return_global=True
    )
    negative_recovery = negative.compute_stresses(
        mesh, np.zeros(24), material, return_global=True
    )
    np.testing.assert_allclose(
        negative_recovery["physical_directors"],
        -positive_recovery["physical_directors"],
        rtol=0.0,
        atol=2.0e-15,
    )


@pytest.mark.parametrize("length", (1.0e-6, 1.0, 1.0e6))
@pytest.mark.parametrize("warped", (False, True))
def test_qualified_q4_geometry_guards_are_scale_invariant_publicly(
    length: float,
    warped: bool,
) -> None:
    unit_nodes = (
        WARPED_GEOMETRIES[0]
        if warped
        else np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    )
    nodes = length * unit_nodes
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(
        21,
        [1, 2, 3, 4],
        "warped",
        thickness=0.025 * length,
    )
    _frame, _local, warpage = equation7_frame(nodes)
    expected_warpage = equation7_frame(unit_nodes)[2]
    assert warpage == pytest.approx(expected_warpage, rel=2.0e-14, abs=2.0e-16)
    stiffness = element.compute_stiffness_matrix(mesh, material)
    mass = element.compute_mass_matrix(mesh, material)
    recovery = element.compute_stresses(
        mesh,
        np.zeros(24),
        material,
        return_global=True,
    )
    force, tangent, _state = element.compute_nonlinear_response(
        mesh,
        material,
        np.zeros(24),
        tangent=True,
    )
    assert tangent is not None
    for values in (stiffness, mass, force, tangent):
        assert np.all(np.isfinite(values))
    for values in recovery.values():
        if isinstance(values, np.ndarray):
            assert np.all(np.isfinite(values))
    np.testing.assert_array_equal(force, np.zeros(24))
    if not warped:
        np.testing.assert_allclose(tangent, stiffness, rtol=2.0e-12, atol=2.0e-12)


@pytest.mark.parametrize("length", (1.0e-6, 1.0, 1.0e6))
def test_near_collapsed_warped_q4_is_rejected_at_every_scale_and_public_route(
    length: float,
) -> None:
    nodes = length * np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0e-14),
            (1.0, 4.0e-14, 0.0),
            (0.0, 4.0e-14, -1.0e-14),
        ),
        dtype=float,
    )
    mesh = _mesh(nodes)
    material = _material()

    def make() -> QualifiedE4PLShellElement:
        return QualifiedE4PLShellElement(
            31,
            [1, 2, 3, 4],
            "warped",
            thickness=0.01 * length,
            planar_tolerance=0.0,
        )

    routes = (
        lambda: make().compute_stiffness_matrix(mesh, material),
        lambda: make().compute_mass_matrix(mesh, material),
        lambda: make().compute_stresses(mesh, np.zeros(24), material),
        lambda: make().compute_nonlinear_response(
            mesh,
            material,
            np.zeros(24),
            tangent=True,
        ),
    )
    for route in routes:
        with pytest.raises(ValueError, match="dimensionless|nonpositive local Jacobian"):
            route()
