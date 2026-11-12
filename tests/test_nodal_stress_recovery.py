"""Gauss-to-node stress recovery: exactness and coarse-mesh improvement."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.assembly import solve_linear
from anyfesolver.boundary import LoadCase
from anyfesolver.mesh_gen import generate_simple_panel_mesh
from anyfesolver.results import (
    _cached_gauss_to_node_extrapolation,
    _gauss_to_node_extrapolation,
    recover_nodal_stresses,
)
from anyfesolver.s4_validity import _membrane_displacement, _single_s4_model


def test_extrapolation_operator_reproduces_matching_polynomials_exactly() -> None:
    class _FakeElement:
        def __init__(self, num_nodes, gauss_points):
            self.num_nodes = num_nodes
            self.gauss_points = gauss_points

    g = 1.0 / np.sqrt(3.0)
    gauss_2x2 = np.array([[-g, -g], [g, -g], [g, g], [-g, g]])
    op4 = _gauss_to_node_extrapolation(_FakeElement(4, gauss_2x2))
    assert op4.shape == (4, 4)

    def bilinear(p):
        return 1.5 - 2.0 * p[:, 0] + 0.7 * p[:, 1] + 0.3 * p[:, 0] * p[:, 1]

    nodes4 = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(op4 @ bilinear(gauss_2x2), bilinear(nodes4), rtol=1.0e-12)

    r = np.sqrt(0.6)
    gauss_3x3 = np.array([[x, y] for y in (-r, 0.0, r) for x in (-r, 0.0, r)])
    op8 = _gauss_to_node_extrapolation(_FakeElement(8, gauss_3x3))
    assert op8.shape == (8, 9)

    def biquadratic(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 + x - 0.4 * y + 0.2 * x * y + 0.8 * x**2 - 0.3 * y**2 + 0.1 * x**2 * y

    nodes8 = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float)
    np.testing.assert_allclose(op8 @ biquadratic(gauss_3x3), biquadratic(nodes8), rtol=1.0e-11)


def test_extrapolation_operator_is_cached_by_topology_and_gauss_rule() -> None:
    class _FakeElement:
        num_nodes = 4

        def __init__(self, gauss_points):
            self.gauss_points = gauss_points

    g = 1.0 / np.sqrt(3.0)
    gauss_points = np.array([[-g, -g], [g, -g], [g, g], [-g, g]])
    _cached_gauss_to_node_extrapolation.cache_clear()

    first = _gauss_to_node_extrapolation(_FakeElement(gauss_points.copy()))
    second = _gauss_to_node_extrapolation(_FakeElement(gauss_points.copy()))
    cache_info = _cached_gauss_to_node_extrapolation.cache_info()

    assert first is second
    assert cache_info.misses == 1
    assert cache_info.hits == 1
    assert not first.flags.writeable


def test_recovered_nodal_stresses_exact_for_constant_field_on_skew_element() -> None:
    coords = ((0.0, 0.0, 0.0), (1.35, 0.08, 0.0), (1.08, 0.95, 0.0), (-0.18, 1.12, 0.0))
    model = _single_s4_model(coords)
    material = model.get_material("steel")
    eps = (1.2e-4, -0.4e-4, 0.7e-4)
    u = _membrane_displacement(model, *eps)
    E, nu = material.elastic_modulus, material.poisson_ratio
    expected_xx = E / (1.0 - nu**2) * (eps[0] + nu * eps[1])

    recovered = recover_nodal_stresses(model, u)
    assert recovered["method"] == "gauss_extrapolation_nodal_average"
    for node_values in recovered["nodal"].values():
        assert node_values["global_xx_top"] == pytest.approx(expected_xx, rel=1.0e-10)
        assert node_values["global_xx_bot"] == pytest.approx(expected_xx, rel=1.0e-10)


def _panel_case(division: int):
    model = generate_simple_panel_mesh(1.0, 1.0, 0.01, num_divisions_x=division, num_divisions_y=division, use_8node_elements=False)
    load_case = LoadCase("p")
    for element_id in model.mesh.elements:
        load_case.add_pressure_load(element_id, 1000.0)
    displacements, _info = solve_linear(model, load_case)
    raw = 0.0
    for element in model.mesh.elements.values():
        stresses = element.compute_stresses(model.mesh, displacements, model.get_material(element.material_name))
        raw = max(raw, float(np.max(np.abs(stresses["von_mises"]))))
    recovered = recover_nodal_stresses(model, displacements)["max_von_mises"]
    return raw, recovered


def test_recovery_and_native_q4_stresses_converge_toward_fine_mesh_value() -> None:
    _ref_raw, reference = _panel_case(16)
    raw_coarse, recovered_coarse = _panel_case(4)
    raw_medium, recovered_medium = _panel_case(8)

    # Gauss and nodal maxima sample different physical locations. Each sequence
    # converges toward the resolved peak. Pointwise precision is checked separately
    # against manufactured stresses and cyclic permutations of the same physical nodes.
    assert abs(raw_medium - reference) < abs(raw_coarse - reference)
    assert abs(recovered_medium - reference) < abs(recovered_coarse - reference)
    assert recovered_medium == pytest.approx(reference, rel=0.075)
