"""Regression tests for beam bending-axis conventions and section orientation.

These tests use an asymmetric cross-section (Iy != Iz) so that any Iy/Iz swap
or uncontrolled section orientation produces an order-of-magnitude error
instead of passing silently.  Closed-form Timoshenko references:

    cantilever tip load P in local z:  w = P L^3 / (3 E Iy) + P L / (kz G A)
    cantilever torque T:               theta = T L / (G J)
    pinned-pinned Euler column:        Pcr = pi^2 E I / L^2
"""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase, PinnedSupport
from anyfesolver.assembly import solve_linear
from anyfesolver.buckling import solve_eigenvalue_buckling

E = 210.0e9
NU = 0.3
G = E / (2.0 * (1.0 + NU))

L = 2.0
A = 0.01
IY = 8.0e-6  # strong axis: bending about local y, deflection in local z
IZ = 1.0e-6  # weak axis
KS = 5.0 / 6.0
P = 1.0e4

CROSS_SECTION = {
    "area": A,
    "Iy": IY,
    "Iz": IZ,
    "J": 1.0e-6,
    "shear_factor_y": KS,
    "shear_factor_z": KS,
    "orientation": (0.0, 0.0, 1.0),  # local z = global Z
}

TIP_W_REF = P * L**3 / (3.0 * E * IY) + P * L / (KS * G * A)
TIP_V_REF = P * L**3 / (3.0 * E * IZ) + P * L / (KS * G * A)


def _build_cantilever(element_cls, axis: str, cross_section, n_elem: int = 20):
    """Cantilever along the given global axis with the section web in +Z."""
    model = FEModel(name=f"cantilever_{element_cls.__name__}_{axis}")
    model.add_material("steel", E, NU)
    if element_cls is BeamElement:
        num_nodes = n_elem + 1
    else:
        num_nodes = 2 * n_elem + 1
    for i in range(num_nodes):
        s = L * i / (num_nodes - 1)
        x = s if axis == "X" else 0.0
        y = s if axis == "Y" else 0.0
        model.add_node(i + 1, x, y, 0.0)
    if element_cls is BeamElement:
        for e in range(n_elem):
            model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", cross_section))
    else:
        for e in range(n_elem):
            base = 2 * e + 1
            model.add_element(
                e + 1, QuadraticBeamElement(e + 1, [base, base + 1, base + 2], "steel", cross_section)
            )
    model.add_boundary_condition(FixedSupport("fix", [1]))
    return model, num_nodes


@pytest.mark.parametrize("element_cls", [BeamElement, QuadraticBeamElement])
@pytest.mark.parametrize("axis", ["X", "Y"])
def test_cantilever_strong_axis_deflection_matches_Iy(element_cls, axis):
    """Tip load along the web (global Z) must engage Iy regardless of heading."""
    model, tip = _build_cantilever(element_cls, axis, CROSS_SECTION)
    load_case = LoadCase(name="tip")
    load_case.add_nodal_load(tip, load_vector=[0.0, 0.0, P, 0.0, 0.0, 0.0])
    displacements, _ = solve_linear(model, load_case)
    w_tip = displacements[model.mesh.get_node(tip).dofs[2]]
    assert w_tip == pytest.approx(TIP_W_REF, rel=2.0e-3)


@pytest.mark.parametrize("element_cls", [BeamElement, QuadraticBeamElement])
def test_cantilever_weak_axis_deflection_matches_Iz(element_cls):
    """Tip load perpendicular to the web (global Y) must engage Iz."""
    model, tip = _build_cantilever(element_cls, "X", CROSS_SECTION)
    load_case = LoadCase(name="tip")
    load_case.add_nodal_load(tip, load_vector=[0.0, P, 0.0, 0.0, 0.0, 0.0])
    displacements, _ = solve_linear(model, load_case)
    v_tip = displacements[model.mesh.get_node(tip).dofs[1]]
    assert v_tip == pytest.approx(TIP_V_REF, rel=2.0e-3)


def test_default_frame_without_orientation_matches_web_up_for_x_member():
    """Heuristic frame for a member along X has local z = global Z."""
    section = {k: v for k, v in CROSS_SECTION.items() if k != "orientation"}
    model, tip = _build_cantilever(BeamElement, "X", section)
    load_case = LoadCase(name="tip")
    load_case.add_nodal_load(tip, load_vector=[0.0, 0.0, P, 0.0, 0.0, 0.0])
    displacements, _ = solve_linear(model, load_case)
    w_tip = displacements[model.mesh.get_node(tip).dofs[2]]
    assert w_tip == pytest.approx(TIP_W_REF, rel=2.0e-3)


def test_cantilever_torsion():
    model, tip = _build_cantilever(BeamElement, "X", CROSS_SECTION, n_elem=10)
    torque = 100.0
    load_case = LoadCase(name="twist")
    load_case.add_nodal_load(tip, load_vector=[0.0, 0.0, 0.0, torque, 0.0, 0.0])
    displacements, _ = solve_linear(model, load_case)
    theta = displacements[model.mesh.get_node(tip).dofs[3]]
    assert theta == pytest.approx(torque * L / (G * CROSS_SECTION["J"]), rel=1.0e-6)


def _euler_column(element_cls, n_elem: int = 10):
    length = 3.0
    inertia = 1.0e-6
    section = {"area": A, "Iy": inertia, "Iz": inertia, "J": 1.0e-6}
    model = FEModel(name=f"euler_{element_cls.__name__}")
    model.add_material("steel", E, NU)
    if element_cls is BeamElement:
        num_nodes = n_elem + 1
    else:
        num_nodes = 2 * n_elem + 1
    for i in range(num_nodes):
        model.add_node(i + 1, length * i / (num_nodes - 1), 0.0, 0.0)
    if element_cls is BeamElement:
        for e in range(n_elem):
            model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", section))
    else:
        for e in range(n_elem):
            base = 2 * e + 1
            model.add_element(
                e + 1, QuadraticBeamElement(e + 1, [base, base + 1, base + 2], "steel", section)
            )
    model.add_boundary_condition(PinnedSupport("p1", [1]))
    model.add_boundary_condition(BoundaryCondition("rx1", [1], {"rx": 0.0}))
    model.add_boundary_condition(BoundaryCondition("p2", [num_nodes], {"uy": 0.0, "uz": 0.0}))
    states = {e + 1: {"axial_compression": 1.0} for e in range(n_elem)}
    result = solve_eigenvalue_buckling(model, element_states=states, num_modes=2)
    return result.critical_load_factor, np.pi**2 * E * inertia / length**2


def test_euler_column_linear_beam():
    pcr_fe, pcr_ref = _euler_column(BeamElement)
    assert pcr_fe == pytest.approx(pcr_ref, rel=1.0e-2)


def test_euler_column_quadratic_beam():
    """3-node beam KG (lateral-gradient theory) must reproduce the Euler load."""
    pcr_fe, pcr_ref = _euler_column(QuadraticBeamElement)
    assert pcr_fe == pytest.approx(pcr_ref, rel=2.0e-2)
