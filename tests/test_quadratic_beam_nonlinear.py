"""QuadraticBeamElement nonlinear response: consistency, P-delta, fiber plasticity."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import dnv_c208_steel_curve
from anyfesolver.nonlinear_static import solve_static_nonlinear

E_STEEL = 2.1e11
LENGTH = 1.0
INERTIA = 1.0e-8


def _single_element_model(fiber: bool) -> tuple:
    model = FEModel("q_beam")
    model.add_material("steel", E_STEEL, 0.3, density=7850.0)
    if fiber:
        model.materials["steel"].hardening_curve = dnv_c208_steel_curve("S355", 0.01)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 0.6, 0.1, 0.0)
    model.add_node(3, 1.2, 0.2, 0.0)
    section = {"area": 1e-3, "Iy": 1e-6, "Iz": 2e-6, "J": 1e-6}
    if fiber:
        section["fiber_plasticity"] = True
    element = QuadraticBeamElement(1, [1, 2, 3], "steel", section)
    model.add_element(1, element)
    return model, element


def test_quadratic_beam_rejects_curved_midside_geometry() -> None:
    model = FEModel("curved_q_beam")
    model.add_material("steel", E_STEEL, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 0.5, 0.1, 0.0)
    model.add_node(3, 1.0, 0.0, 0.0)
    element = QuadraticBeamElement(
        1,
        [1, 2, 3],
        "steel",
        {"area": 1.0e-3, "Iy": 1.0e-6, "Iz": 2.0e-6, "J": 1.0e-6},
    )
    model.add_element(1, element)

    with pytest.raises(ValueError, match="straight-sided"):
        element.compute_stiffness_matrix(model.mesh, model.get_material("steel"))


@pytest.mark.parametrize("fiber", [False, True])
def test_quadratic_beam_tangent_is_consistent_with_internal_force(fiber: bool) -> None:
    model, element = _single_element_model(fiber)
    material = model.get_material("steel")
    rng = np.random.default_rng(4)
    u = 2.0e-3 * rng.standard_normal(18)
    _F0, K0, _state = element.compute_nonlinear_response(model.mesh, material, u, None, 5, True)
    step = 1.0e-7
    K_fd = np.zeros((18, 18))
    for j in range(18):
        up, um = u.copy(), u.copy()
        up[j] += step
        um[j] -= step
        Fp, _, _ = element.compute_nonlinear_response(model.mesh, material, up, None, 5, False)
        Fm, _, _ = element.compute_nonlinear_response(model.mesh, material, um, None, 5, False)
        K_fd[:, j] = (Fp - Fm) / (2.0 * step)
    scale = float(np.max(np.abs(K_fd)))
    assert np.max(np.abs(K0 - K_fd)) < 1.0e-6 * scale
    assert np.max(np.abs(K0 - K0.T)) < 1.0e-12 * scale


def _column(quadratic: bool, n: int = 6) -> tuple:
    model = FEModel("column")
    model.add_material("steel", E_STEEL, 0.3, density=7850.0)
    grid = 2 * n if quadratic else n
    for i in range(grid + 1):
        model.add_node(i + 1, LENGTH * i / grid, 0.0, 0.0)
    section = {"area": 1e-3, "Iy": INERTIA, "Iz": INERTIA, "J": 1e-8}
    for e in range(n):
        if quadratic:
            model.add_element(e + 1, QuadraticBeamElement(e + 1, [2 * e + 1, 2 * e + 2, 2 * e + 3], "steel", section))
        else:
            model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", section))
    end = grid + 1
    model.add_boundary_condition(BoundaryCondition("a", [1], {"ux": 0, "uy": 0, "uz": 0, "rx": 0}))
    model.add_boundary_condition(BoundaryCondition("b", [end], {"uy": 0, "uz": 0, "rx": 0}))
    return model, end, grid


def _amplification(quadratic: bool) -> float:
    P_euler = np.pi**2 * E_STEEL * INERTIA / LENGTH**2
    model, end, grid = _column(quadratic)
    load_case = LoadCase("pq")
    load_case.add_nodal_load(end, forces=np.array([-0.8 * P_euler, 0.0, 0.0]))
    load_case.add_nodal_load(grid // 2 + 1, forces=np.array([0.0, 1.0, 0.0]))
    result = solve_static_nonlinear(model, load_case, num_steps=10, max_iterations=40, tolerance=1.0e-8)
    assert result.status == "completed"
    mid = model.mesh.get_node(grid // 2 + 1)
    linear_deflection = 1.0 * LENGTH**3 / (48.0 * E_STEEL * INERTIA)
    return float(result.displacements[mid.dofs[1]] / linear_deflection)


def test_quadratic_beam_p_delta_amplification_matches_2node_and_theory() -> None:
    amp_linear = _amplification(quadratic=False)
    amp_quadratic = _amplification(quadratic=True)
    # secant-type amplification at P = 0.8 P_euler is ~5; both formulations must
    # agree with each other and sit in the physically correct range.
    assert 4.0 < amp_quadratic < 5.5
    assert amp_quadratic == pytest.approx(amp_linear, rel=0.05)


def _plastic_cantilever_tip(quadratic: bool) -> tuple:
    model = FEModel("cantilever")
    model.add_material("steel", E_STEEL, 0.3, density=7850.0)
    model.materials["steel"].hardening_curve = dnv_c208_steel_curve("S355", 0.01)
    n = 4
    grid = 2 * n if quadratic else n
    section = {"area": 1e-3, "Iy": 1e-7, "Iz": 1e-7, "J": 1e-7, "fiber_plasticity": True}
    for i in range(grid + 1):
        model.add_node(i + 1, LENGTH * i / grid, 0.0, 0.0)
    for e in range(n):
        if quadratic:
            model.add_element(e + 1, QuadraticBeamElement(e + 1, [2 * e + 1, 2 * e + 2, 2 * e + 3], "steel", section))
        else:
            model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", section))
    model.add_boundary_condition(BoundaryCondition("clamp", [1], {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0}))
    load_case = LoadCase("f")
    load_case.add_nodal_load(grid + 1, forces=np.array([0.0, 3200.0, 0.0]))
    result = solve_static_nonlinear(model, load_case, num_steps=10, max_iterations=40, tolerance=1.0e-6)
    assert result.status == "completed"
    tip = model.mesh.get_node(grid + 1)
    max_plastic = float((result.info or {}).get("strain_summary", {}).get("max_equivalent_plastic_strain", 0.0) or 0.0)
    return float(result.displacements[tip.dofs[1]]), max_plastic


def test_quadratic_beam_fiber_plasticity_matches_2node_reference() -> None:
    elastic_tip = 3200.0 * LENGTH**3 / (3.0 * E_STEEL * 1e-7)
    tip_linear, plastic_linear = _plastic_cantilever_tip(quadratic=False)
    tip_quadratic, plastic_quadratic = _plastic_cantilever_tip(quadratic=True)

    # clearly past first yield: softer than the elastic solution with committed
    # plastic strain, and the two beam formulations agree.
    assert tip_quadratic > 1.3 * elastic_tip
    assert plastic_quadratic > 1.0e-4
    assert plastic_linear > 1.0e-4
    assert tip_quadratic == pytest.approx(tip_linear, rel=0.05)
