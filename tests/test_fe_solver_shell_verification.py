"""Shell element verification tests.

These tests are intended to catch formulation regressions before geometric
stiffness and nonlinear buckling are added.  They deliberately focus on
invariants and simple patch behaviour rather than full external-reference
comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import assemble_stiffness_matrix, generate_simple_panel_mesh, shell_element_patch_summary


@pytest.mark.parametrize("use_8node_elements", [False, True])
def test_shell_shape_functions_partition_unity_and_zero_derivative_sum(use_8node_elements: bool) -> None:
    model = generate_simple_panel_mesh(
        1.0,
        1.0,
        0.01,
        num_divisions_x=1,
        num_divisions_y=1,
        use_8node_elements=use_8node_elements,
    )
    element = model.mesh.get_element(1)
    assert element is not None

    sample_points = [
        (-0.75, -0.50),
        (0.0, 0.0),
        (0.60, -0.20),
        (0.25, 0.80),
    ]
    for xi, eta in sample_points:
        N, dN_dxi, dN_deta = element.compute_shape_functions(xi, eta)
        assert np.isclose(np.sum(N), 1.0, atol=1.0e-12)
        assert np.isclose(np.sum(dN_dxi), 0.0, atol=1.0e-12)
        assert np.isclose(np.sum(dN_deta), 0.0, atol=1.0e-12)


@pytest.mark.parametrize("use_8node_elements", [False, True])
def test_shell_stiffness_is_symmetric_and_finite(use_8node_elements: bool) -> None:
    model = generate_simple_panel_mesh(
        1.0,
        0.75,
        0.012,
        num_divisions_x=1,
        num_divisions_y=1,
        use_8node_elements=use_8node_elements,
    )
    element = model.mesh.get_element(1)
    assert element is not None
    material = model.get_material("steel")

    K = element.compute_stiffness_matrix(model.mesh, material)

    assert K.shape == (element.total_dofs, element.total_dofs)
    assert np.all(np.isfinite(K))
    assert np.allclose(K, K.T, rtol=1.0e-10, atol=1.0e-5)


def _element_rigid_translation(element, direction: int, value: float = 1.0) -> np.ndarray:
    u = np.zeros(element.total_dofs, dtype=float)
    for local_node_index in range(element.num_nodes):
        u[local_node_index * 6 + direction] = value
    return u


def _element_rigid_rotation(element, model, axis: int, omega: float = 1.0e-3) -> np.ndarray:
    coords = element.get_node_coordinates(model.mesh)
    centroid = np.mean(coords, axis=0)
    rotation = np.zeros(3, dtype=float)
    rotation[int(axis)] = omega
    u = np.zeros(element.total_dofs, dtype=float)
    for local_node_index, coord in enumerate(coords):
        base = local_node_index * 6
        u[base:base + 3] = np.cross(rotation, coord - centroid)
        u[base + 3:base + 6] = rotation
    return u


@pytest.mark.parametrize("use_8node_elements", [False, True])
def test_shell_rigid_body_modes_have_near_zero_strain_energy(use_8node_elements: bool) -> None:
    model = generate_simple_panel_mesh(
        1.0,
        0.8,
        0.01,
        num_divisions_x=1,
        num_divisions_y=1,
        use_8node_elements=use_8node_elements,
    )
    element = model.mesh.get_element(1)
    assert element is not None
    material = model.get_material("steel")
    K = element.compute_stiffness_matrix(model.mesh, material)

    # Scale tolerance by a representative diagonal stiffness because exact zero
    # is not expected after numerical integration and drilling stabilization.
    scale = max(float(np.max(np.abs(np.diag(K)))), 1.0)
    modes = [
        _element_rigid_translation(element, 0),
        _element_rigid_translation(element, 1),
        _element_rigid_translation(element, 2),
        _element_rigid_rotation(element, model, 0),
        _element_rigid_rotation(element, model, 1),
        _element_rigid_rotation(element, model, 2),
    ]
    for mode in modes:
        energy = float(mode @ K @ mode)
        assert abs(energy) < 1.0e-10 * scale


def test_shell_constant_membrane_strain_patch_gives_constant_stress() -> None:
    model = generate_simple_panel_mesh(
        1.0,
        1.0,
        0.01,
        num_divisions_x=1,
        num_divisions_y=1,
        use_8node_elements=False,
    )
    element = model.mesh.get_element(1)
    assert element is not None
    material = model.get_material("steel")

    eps_x = 1.2e-4
    eps_y = -0.4e-4
    gamma_xy = 0.7e-4

    u = np.zeros(element.total_dofs, dtype=float)
    coords = element.get_node_coordinates(model.mesh)
    for local_node_index, coord in enumerate(coords):
        x, y, _z = coord
        base = local_node_index * 6
        u[base + 0] = eps_x * x
        u[base + 1] = eps_y * y + gamma_xy * x

    stresses = element.compute_stresses(model.mesh, u, material)

    E = material.elastic_modulus
    nu = material.poisson_ratio
    expected_sigma_x = E / (1.0 - nu**2) * (eps_x + nu * eps_y)
    expected_sigma_y = E / (1.0 - nu**2) * (eps_y + nu * eps_x)
    expected_tau_xy = E / (2.0 * (1.0 + nu)) * gamma_xy

    assert np.allclose(stresses["membrane_xx"], expected_sigma_x, rtol=1.0e-10, atol=1.0e-5)
    assert np.allclose(stresses["membrane_yy"], expected_sigma_y, rtol=1.0e-10, atol=1.0e-5)
    assert np.allclose(stresses["membrane_xy"], expected_tau_xy, rtol=1.0e-10, atol=1.0e-5)
    assert np.allclose(stresses["bending_xx"], 0.0, atol=1.0e-8)
    assert np.allclose(stresses["bending_yy"], 0.0, atol=1.0e-8)
    assert np.allclose(stresses["bending_xy"], 0.0, atol=1.0e-8)


@pytest.mark.parametrize("use_8node_elements", [False, True])
def test_shell_linear_curvature_patch_gives_constant_bending_stress(use_8node_elements: bool) -> None:
    model = generate_simple_panel_mesh(
        1.0,
        1.0,
        0.01,
        num_divisions_x=1,
        num_divisions_y=1,
        use_8node_elements=use_8node_elements,
    )
    element = model.mesh.get_element(1)
    assert element is not None
    material = model.get_material("steel")

    kappa_x = 1.1e-3
    u = np.zeros(element.total_dofs, dtype=float)
    coords = element.get_node_coordinates(model.mesh)
    for local_node_index, coord in enumerate(coords):
        x, _y, _z = coord
        base = local_node_index * 6
        u[base + 4] = kappa_x * x

    stresses = element.compute_stresses(model.mesh, u, material)
    E = material.elastic_modulus
    nu = material.poisson_ratio
    h = element.thickness
    expected_sigma_b_x = E * h / (2.0 * (1.0 - nu**2)) * kappa_x

    assert np.allclose(stresses["membrane_xx"], 0.0, atol=1.0e-8)
    assert np.allclose(stresses["membrane_yy"], 0.0, atol=1.0e-8)
    assert np.allclose(stresses["membrane_xy"], 0.0, atol=1.0e-8)
    assert np.allclose(stresses["bending_xx"], expected_sigma_b_x, rtol=1.0e-10, atol=1.0e-5)
    assert np.max(stresses["bending_xx"]) - np.min(stresses["bending_xx"]) < 1.0e-8


@pytest.mark.parametrize("use_8node_elements", [False, True])
def test_assembled_shell_rigid_translation_has_near_zero_energy(use_8node_elements: bool) -> None:
    model = generate_simple_panel_mesh(
        2.0,
        1.0,
        0.01,
        num_divisions_x=2,
        num_divisions_y=2,
        use_8node_elements=use_8node_elements,
    )
    K, _info = assemble_stiffness_matrix(model)
    scale = max(float(np.max(np.abs(K.diagonal()))), 1.0)

    for direction in (0, 1, 2):
        u = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
        for node in model.mesh.nodes.values():
            u[node.dofs[direction]] = 1.0
        energy = float(u @ K @ u)
        assert abs(energy) < 1.0e-10 * scale


@pytest.mark.parametrize("use_8node_elements", [False, True])
def test_assembled_shell_constant_membrane_strain_patch(use_8node_elements: bool) -> None:
    model = generate_simple_panel_mesh(
        2.0,
        1.0,
        0.01,
        num_divisions_x=2,
        num_divisions_y=2,
        use_8node_elements=use_8node_elements,
    )
    material = model.get_material("steel")

    eps_x = 1.2e-4
    eps_y = -0.4e-4
    gamma_xy = 0.7e-4
    u_global = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        x, y, _z = node.coords()
        u_global[node.dofs[0]] = eps_x * x
        u_global[node.dofs[1]] = eps_y * y + gamma_xy * x

    E = material.elastic_modulus
    nu = material.poisson_ratio
    expected_sigma_x = E / (1.0 - nu**2) * (eps_x + nu * eps_y)
    expected_sigma_y = E / (1.0 - nu**2) * (eps_y + nu * eps_x)
    expected_tau_xy = E / (2.0 * (1.0 + nu)) * gamma_xy

    for element in model.mesh.elements.values():
        stresses = element.compute_stresses(model.mesh, u_global, material)
        assert np.allclose(stresses["membrane_xx"], expected_sigma_x, rtol=1.0e-10, atol=1.0e-5)
        assert np.allclose(stresses["membrane_yy"], expected_sigma_y, rtol=1.0e-10, atol=1.0e-5)
        assert np.allclose(stresses["membrane_xy"], expected_tau_xy, rtol=1.0e-10, atol=1.0e-5)
        assert np.allclose(stresses["bending_xx"], 0.0, atol=1.0e-8)
        assert np.allclose(stresses["bending_yy"], 0.0, atol=1.0e-8)
        assert np.allclose(stresses["bending_xy"], 0.0, atol=1.0e-8)

        dofs = element.get_dof_mapping(model.mesh)
        summary = shell_element_patch_summary(model, element.element_id, u_global[dofs])
        assert summary.max_membrane_spread < 1.0e-6
        assert summary.max_bending_spread < 1.0e-8
