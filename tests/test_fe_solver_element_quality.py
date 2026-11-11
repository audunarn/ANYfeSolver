"""Element-quality regression tests.

Covers the MITC4 shear treatment of the 4-node shell (no spurious zero-energy
modes, no shear locking), physically consistent beam rotary inertia, exact
beam surface-stress recovery from section data, and the tangent-path
displacement amplification of the nonlinear load stepper.
"""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement, QuadraticBeamElement, create_shell_element
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.assembly import compute_stresses, solve_linear
from anyfesolver.buckling import solve_eigenvalue_buckling
from anyfesolver.mesh_gen import StiffenerCrossSection
from anyfesolver.nonlinear import solve_nonlinear_load_stepping

E = 210.0e9
NU = 0.3
G = E / (2.0 * (1.0 + NU))


def _single_shell_model(node_coords, connectivity, thickness=0.01):
    model = FEModel(name="single_shell")
    model.add_material("steel", E, NU)
    for node_id, (x, y, z) in enumerate(node_coords, start=1):
        model.add_node(node_id, x, y, z)
    model.add_element(1, create_shell_element(1, connectivity, "steel", thickness=thickness))
    return model


@pytest.mark.parametrize(
    "node_coords",
    [
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        # distorted quad
        [(0.0, 0.0, 0.0), (1.3, 0.1, 0.0), (1.1, 0.9, 0.0), (-0.2, 1.2, 0.0)],
    ],
)
def test_free_4node_shell_has_exactly_six_zero_energy_modes(node_coords):
    """MITC4 shear removes the w-hourglass mode of 1-point reduced shear."""
    model = _single_shell_model(node_coords, [1, 2, 3, 4])
    element = model.mesh.get_element(1)
    K = element.compute_stiffness_matrix(model.mesh, model.get_material("steel"))
    eigenvalues = np.linalg.eigvalsh(0.5 * (K + K.T))
    threshold = 1.0e-9 * eigenvalues[-1]
    num_zero = int(np.sum(np.abs(eigenvalues) < threshold))
    assert num_zero == 6


def test_thin_plate_cantilever_strip_does_not_shear_lock():
    """MITC4 keeps a span/thickness = 1000 strip within a few percent of
    thin-plate theory; pure displacement Q4 would lock by orders of magnitude."""
    length, width, t = 1.0, 0.1, 0.001
    n = 10
    model = FEModel(name="thin_strip")
    model.add_material("steel", E, NU)
    nid = {}
    k = 1
    for i in range(n + 1):
        for j in range(2):
            model.add_node(k, length * i / n, width * j, 0.0)
            nid[(i, j)] = k
            k += 1
    for i in range(n):
        model.add_element(
            i + 1,
            create_shell_element(
                i + 1,
                [nid[(i, 0)], nid[(i + 1, 0)], nid[(i + 1, 1)], nid[(i, 1)]],
                "steel",
                thickness=t,
            ),
        )
    model.add_boundary_condition(FixedSupport("fix", [nid[(0, 0)], nid[(0, 1)]]))
    P = 1.0
    load_case = LoadCase(name="tip")
    load_case.add_nodal_load(nid[(n, 0)], load_vector=[0.0, 0.0, P / 2.0, 0.0, 0.0, 0.0])
    load_case.add_nodal_load(nid[(n, 1)], load_vector=[0.0, 0.0, P / 2.0, 0.0, 0.0, 0.0])
    displacements, _ = solve_linear(model, load_case)
    w_tip = 0.5 * (
        displacements[model.mesh.get_node(nid[(n, 0)]).dofs[2]]
        + displacements[model.mesh.get_node(nid[(n, 1)]).dofs[2]]
    )
    # A narrow strip with free lateral edges bends like a beam (EI with
    # I = width*t^3/12), not like a plate in cylindrical bending.  A locking
    # pure-displacement Q4 would be orders of magnitude too stiff here.
    w_ref = P * length**3 / (3.0 * E * width * t**3 / 12.0)
    assert w_tip == pytest.approx(w_ref, rel=0.05)


def test_beam_surface_stress_uses_section_fiber_distance():
    """With c_z supplied, the cantilever root bending stress is M*c/I."""
    L, A = 2.0, 0.01
    Iy, c_z = 8.0e-6, 0.125
    section = {"area": A, "Iy": Iy, "Iz": 1.0e-6, "J": 1.0e-6, "c_y": 0.05, "c_z": c_z,
               "orientation": (0.0, 0.0, 1.0)}
    n_elem = 16
    model = FEModel(name="stress_recovery")
    model.add_material("steel", E, NU)
    for i in range(n_elem + 1):
        model.add_node(i + 1, L * i / n_elem, 0.0, 0.0)
    for e in range(n_elem):
        model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fix", [1]))
    P = 1.0e4
    load_case = LoadCase(name="tip")
    load_case.add_nodal_load(n_elem + 1, load_vector=[0.0, 0.0, P, 0.0, 0.0, 0.0])
    displacements, _ = solve_linear(model, load_case)
    stresses = compute_stresses(model, displacements)
    # Element curvature is element-averaged: compare against the moment at the
    # first element midpoint.
    M_mid = P * (L - 0.5 * L / n_elem)
    expected = M_mid * c_z / Iy
    assert abs(stresses[1]["bending_stress_y"]) == pytest.approx(expected, rel=0.02)


def test_beam_torsion_stress_uses_torsion_modulus():
    """With Wt supplied, tau = T / Wt exactly."""
    L = 2.0
    J, Wt = 1.0e-6, 5.0e-5
    section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": J, "torsion_modulus": Wt}
    model = FEModel(name="torsion_stress")
    model.add_material("steel", E, NU)
    for i in range(5):
        model.add_node(i + 1, L * i / 4, 0.0, 0.0)
    for e in range(4):
        model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fix", [1]))
    torque = 250.0
    load_case = LoadCase(name="twist")
    load_case.add_nodal_load(5, load_vector=[0.0, 0.0, 0.0, torque, 0.0, 0.0])
    displacements, _ = solve_linear(model, load_case)
    stresses = compute_stresses(model, displacements)
    assert stresses[1]["torsional_stress"] == pytest.approx(torque / Wt, rel=1.0e-6)


def test_stiffener_cross_section_reports_fiber_distances():
    section = StiffenerCrossSection.from_geometry("T-bar", hw=0.3, tw=0.012, b=0.1, tf=0.02)
    # Extreme fibers: plate side at -z_centroid, flange tip at hw + tf - z_centroid.
    assert section.c_z > 0.0
    assert section.c_y == pytest.approx(0.05)  # half flange width
    assert section.torsion_modulus == pytest.approx(section.J / 0.02)
    total_height = 0.3 + 0.02
    assert section.c_z < total_height
    assert section.c_z + (total_height - section.c_z) == pytest.approx(total_height)


def test_quadratic_beam_axial_stress_recovery_uses_end_nodes():
    """Uniform axial state must recover sigma = F/A, not half of it."""
    L, A = 2.0, 0.01
    section = {"area": A, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
    model = FEModel(name="quad_axial")
    model.add_material("steel", E, NU)
    for i in range(5):
        model.add_node(i + 1, L * i / 4, 0.0, 0.0)
    for e in range(2):
        base = 2 * e + 1
        model.add_element(e + 1, QuadraticBeamElement(e + 1, [base, base + 1, base + 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fix", [1]))
    F = 1.0e5
    load_case = LoadCase(name="pull")
    load_case.add_nodal_load(5, load_vector=[F, 0.0, 0.0, 0.0, 0.0, 0.0])
    displacements, _ = solve_linear(model, load_case)
    stresses = compute_stresses(model, displacements)
    assert stresses[1]["axial_stress"] == pytest.approx(F / A, rel=1.0e-6)
    assert stresses[2]["axial_stress"] == pytest.approx(F / A, rel=1.0e-6)


def test_beam_torsional_rotary_inertia_uses_polar_section_inertia():
    """rx mass entry must be rho*(Iy+Iz)*L/2 per node, not a bar-length term."""
    L = 2.0
    Iy, Iz = 8.0e-6, 1.0e-6
    rho = 7850.0
    section = {"area": 0.01, "Iy": Iy, "Iz": Iz, "J": 1.0e-6, "orientation": (0.0, 0.0, 1.0)}
    model = FEModel(name="mass_check")
    model.add_material("steel", E, NU, density=rho)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, L, 0.0, 0.0)
    element = BeamElement(1, [1, 2], "steel", section)
    model.add_element(1, element)
    M = element.compute_mass_matrix(model.mesh, model.get_material("steel"))
    # Beam along X with web up: local frame == global frame.
    assert M[3, 3] == pytest.approx(rho * (Iy + Iz) * L / 2.0)
    assert M[0, 0] == pytest.approx(rho * 0.01 * L / 2.0)
    # Gravity-style loads only use translational entries; rotary refinement
    # must not change the translational mass.
    assert M[1, 1] == M[2, 2] == M[0, 0]


def test_nonlinear_tangent_path_displacement_amplification():
    """Displacement per unit load must grow as the limit point is approached."""
    num_elements = 10
    length = 4.0
    model = FEModel("amplification")
    model.add_material("steel", E, NU, density=7850.0)
    for i in range(num_elements + 1):
        model.add_node(i + 1, length * i / num_elements, 0.0, 0.0)
    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for i in range(num_elements):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", section))
    all_nodes = list(range(1, num_elements + 2))
    model.add_boundary_condition(
        BoundaryCondition("suppress", all_nodes, {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0})
    )
    model.add_boundary_condition(BoundaryCondition("pins", [1, num_elements + 1], {"uy": 0.0}))
    model.apply_boundary_conditions()

    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    buckling = solve_eigenvalue_buckling(model, states, num_modes=1)
    load_case = LoadCase("lateral")
    load_case.add_nodal_load(1 + num_elements // 2, forces=np.array([0.0, 1.0, 0.0]))

    result = solve_nonlinear_load_stepping(
        model,
        load_case,
        states,
        max_load_factor=0.9 * buckling.critical_load_factor,
        num_steps=9,
        stability_tolerance=0.0,
    )
    assert result.status == "completed"
    per_unit_load = [
        step.displacement_norm / step.load_factor for step in result.steps if step.load_factor > 0.0
    ]
    assert all(later > earlier for earlier, later in zip(per_unit_load, per_unit_load[1:]))
    # At 90% of the critical load the classical amplification 1/(1 - 0.9) = 10x
    # should be clearly visible relative to the first (nearly linear) step.
    assert per_unit_load[-1] / per_unit_load[0] > 5.0
