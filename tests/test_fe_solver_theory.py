import numpy as np
import pytest
from scipy import sparse

from anyfesolver.assembly import (
    _orthonormalize_columns,
    assemble_mass_matrix,
    assemble_system,
    build_constraint_transformation,
    compute_constraint_force_diagnostics,
    reconstruct_full_solution,
    solve_linear,
)
from anyfesolver.boundary import FixedSupport, LoadCase
from anyfesolver.elements import (
    BeamElement,
    CoupledBeamShellElement,
    ShellElement,
    create_shell_element,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.mesh_gen import StiffenerCrossSection, generate_beam_mesh


def _steel_model(name="model"):
    model = FEModel(name)
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.current_material = "steel"
    return model


def _rigid_mode_vector(model, mode):
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in model.mesh.nodes.values():
        x, y, z = node.coords()
        ux, uy, uz, rx, ry, rz = node.dofs

        if mode == 0:
            u[ux] = 1.0
        elif mode == 1:
            u[uy] = 1.0
        elif mode == 2:
            u[uz] = 1.0
        elif mode == 3:
            u[uy] = -z
            u[uz] = y
            u[rx] = 1.0
        elif mode == 4:
            u[ux] = z
            u[uz] = -x
            u[ry] = 1.0
        elif mode == 5:
            u[ux] = -y
            u[uy] = x
            u[rz] = 1.0
        else:
            raise ValueError(mode)
    return u


@pytest.mark.parametrize("node_ids", [[1, 2, 3, 4], list(range(1, 9))])
def test_shell_shape_functions_partition_unity(node_ids):
    element = (
        create_shell_element(1, node_ids, thickness=0.01)
        if len(node_ids) == 4
        else ShellElement(1, node_ids, thickness=0.01)
    )

    for xi, eta in [(-0.3, 0.7), (0.0, 0.0), (0.85, -0.6)]:
        N, dN_dxi, dN_deta = element.compute_shape_functions(xi, eta)

        assert np.sum(N) == pytest.approx(1.0)
        assert np.sum(dN_dxi) == pytest.approx(0.0, abs=1.0e-14)
        assert np.sum(dN_deta) == pytest.approx(0.0, abs=1.0e-14)


def test_beam_element_has_no_rigid_body_stiffness_or_stress():
    model = _steel_model("beam_rigid_body")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 2.0, 0.0, 0.0)
    element = BeamElement(
        1,
        [1, 2],
        "steel",
        {"area": 0.01, "Iy": 2.0e-6, "Iz": 3.0e-6, "J": 4.0e-6},
    )
    model.add_element(1, element)

    K, _, _ = assemble_system(model)
    stiffness_scale = max(float(sparse.linalg.norm(K)), 1.0)

    for mode in range(6):
        u = _rigid_mode_vector(model, mode)
        residual = np.asarray(K @ u).reshape(-1)
        energy = float(u @ residual)

        assert np.linalg.norm(residual) <= stiffness_scale * np.linalg.norm(u) * 1.0e-12
        assert abs(energy) <= stiffness_scale * np.linalg.norm(u) ** 2 * 1.0e-12

        stresses = element.compute_stresses(model.mesh, u, model.get_material("steel"))
        assert stresses["von_mises"] == pytest.approx(0.0, abs=1.0e-12)


def test_timoshenko_cantilever_tip_deflection_matches_bending_plus_shear():
    length = 2.0
    load = 1000.0
    area = 0.01
    inertia_y = 1.0e-6
    shear_factor_y = 5.0 / 6.0
    num_elements = 20
    model = generate_beam_mesh(
        length=length,
        num_divisions=num_elements,
        cross_section={
            "area": area,
            "Iy": inertia_y,
            "Iz": inertia_y,
            "J": 1.0e-8,
            "shear_factor_y": shear_factor_y,
        },
    )

    load_case = LoadCase("tip_load")
    load_case.add_nodal_load(num_elements + 1, forces=np.array([0.0, -load, 0.0]))

    displacements, solver_info = solve_linear(model, load_case)

    material = model.get_material("steel")
    expected = -(
        load * length**3 / (3.0 * material.elastic_modulus * inertia_y)
        + load * length / (shear_factor_y * material.shear_modulus * area)
    )
    tip_node = model.mesh.get_node(num_elements + 1)
    assert solver_info["convergence_info"]["status"] == "converged"
    assert displacements[tip_node.dofs[1]] == pytest.approx(expected, rel=1.0e-8)


def test_beam_torsion_uses_local_x_rotation_dof():
    length = 1.0
    torque = 100.0
    torsion_constant = 1.0e-6
    num_elements = 10
    model = generate_beam_mesh(
        length=length,
        num_divisions=num_elements,
        cross_section={"area": 0.01, "Iy": 1.0e-8, "Iz": 1.0e-8, "J": torsion_constant},
    )

    load_case = LoadCase("tip_torque")
    load_case.add_nodal_load(num_elements + 1, forces=np.zeros(3), moments=np.array([torque, 0.0, 0.0]))

    displacements, solver_info = solve_linear(model, load_case)

    material = model.get_material("steel")
    expected_twist = torque * length / (material.shear_modulus * torsion_constant)
    tip_node = model.mesh.get_node(num_elements + 1)
    assert solver_info["convergence_info"]["status"] == "converged"
    assert displacements[tip_node.dofs[3]] == pytest.approx(expected_twist, rel=1.0e-10)
    assert displacements[tip_node.dofs[5]] == pytest.approx(0.0, abs=1.0e-12)


def test_shell_element_has_no_rigid_body_stiffness():
    model = _steel_model("shell_rigid_body")
    for node_id, (x, y, z) in {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        3: (2.0, 1.0, 0.0),
        4: (0.0, 1.0, 0.0),
    }.items():
        model.add_node(node_id, x, y, z)
    model.add_element(
        1, create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.1)
    )

    K, _, _ = assemble_system(model)
    stiffness_scale = max(float(sparse.linalg.norm(K)), 1.0)

    for mode in range(6):
        u = _rigid_mode_vector(model, mode)
        residual = np.asarray(K @ u).reshape(-1)
        energy = float(u @ residual)

        assert np.linalg.norm(residual) <= stiffness_scale * np.linalg.norm(u) * 1.0e-12
        assert abs(energy) <= stiffness_scale * np.linalg.norm(u) ** 2 * 1.0e-12


def test_shell_constant_membrane_strain_patch_recovers_plane_stress():
    model = _steel_model("shell_membrane_patch")
    for node_id, (x, y, z) in {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        3: (2.0, 1.0, 0.0),
        4: (0.0, 1.0, 0.0),
    }.items():
        model.add_node(node_id, x, y, z)
    element = create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.04)
    model.add_element(1, element)

    eps_x = 1.0e-5
    eps_y = -2.0e-5
    gamma_xy = 3.0e-5
    u_elem = np.zeros(element.total_dofs)
    for local_i, node_id in enumerate(element.node_ids):
        node = model.mesh.get_node(node_id)
        base = local_i * 6
        u_elem[base + 0] = eps_x * node.x + 0.5 * gamma_xy * node.y
        u_elem[base + 1] = eps_y * node.y + 0.5 * gamma_xy * node.x

    material = model.get_material("steel")
    D = material.elastic_modulus / (1.0 - material.poisson_ratio**2) * np.array(
        [
            [1.0, material.poisson_ratio, 0.0],
            [material.poisson_ratio, 1.0, 0.0],
            [0.0, 0.0, (1.0 - material.poisson_ratio) / 2.0],
        ]
    )
    expected = D @ np.array([eps_x, eps_y, gamma_xy])

    stresses = element.compute_stresses(model.mesh, u_elem, material)

    np.testing.assert_allclose(stresses["membrane_xx"], expected[0], rtol=1.0e-12, atol=1.0e-8)
    np.testing.assert_allclose(stresses["membrane_yy"], expected[1], rtol=1.0e-12, atol=1.0e-8)
    np.testing.assert_allclose(stresses["membrane_xy"], expected[2], rtol=1.0e-12, atol=1.0e-8)
    np.testing.assert_allclose(stresses["bending_xx"], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(stresses["shear_xz"], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(stresses["shear_yz"], 0.0, atol=1.0e-10)


def test_consistent_shell_pressure_load_has_correct_resultant():
    model = _steel_model("pressure_resultant")
    for node_id, (x, y, z) in {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        3: (2.0, 1.0, 0.0),
        4: (0.0, 1.0, 0.0),
    }.items():
        model.add_node(node_id, x, y, z)
    model.add_element(
        1, create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.1)
    )

    load_case = LoadCase("pressure")
    load_case.add_pressure_load(1, pressure=5.0)
    F = load_case.get_load_vector(model.mesh, model.mesh.dof_manager)
    nodal_forces = F.reshape(-1, 6)[:, :3]

    np.testing.assert_allclose(np.sum(nodal_forces, axis=0), [0.0, 0.0, 10.0], atol=1.0e-12)
    np.testing.assert_allclose(nodal_forces, np.tile([0.0, 0.0, 2.5], (4, 1)), atol=1.0e-12)


def test_eccentric_beam_shell_mpc_transformation_enforces_cross_product_kinematics():
    model = _steel_model("mpc_kinematics")
    model.add_node(1, 0.2, -0.3, 0.5)
    model.add_node(2, 0.0, 0.0, 0.0)
    model.add_element(1, CoupledBeamShellElement(1, beam_node_id=1, shell_node_id=2, material_name="steel"))

    total_dofs = model.mesh.dof_manager.total_dofs
    K = sparse.eye(total_dofs, format="csr")
    F = np.zeros(total_dofs)
    _, _, T, u0, independent_dofs, info = build_constraint_transformation(K, F, model)

    q = np.linspace(-0.4, 0.7, len(independent_dofs))
    u = reconstruct_full_solution(T, q, u0)

    beam = model.mesh.get_node(1)
    shell = model.mesh.get_node(2)
    r = beam.coords() - shell.coords()
    shell_u = u[shell.dofs[:3]]
    shell_theta = u[shell.dofs[3:6]]
    beam_u = u[beam.dofs[:3]]
    beam_theta = u[beam.dofs[3:6]]

    assert info["num_mpc_slave_dofs"] == 6
    np.testing.assert_allclose(beam_u, shell_u + np.cross(shell_theta, r), atol=1.0e-14)
    np.testing.assert_allclose(beam_theta, shell_theta, atol=1.0e-14)


def test_mpc_force_diagnostics_separate_support_slave_and_master_equivalent_forces():
    model = _steel_model("mpc_force_diagnostics")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 0.0, 0.0, 0.0)
    model.add_element(1, CoupledBeamShellElement(1, beam_node_id=2, shell_node_id=1, material_name="steel"))
    model.add_boundary_condition(FixedSupport("fixed_shell_master", [1]))

    load_vector = np.array([100.0, -50.0, 20.0, 3.0, -4.0, 5.0])
    load = LoadCase("slave_load")
    load.add_nodal_load(2, load_vector)
    displacements = np.zeros(model.mesh.dof_manager.total_dofs)

    diagnostics = compute_constraint_force_diagnostics(model, displacements, load)

    np.testing.assert_allclose(diagnostics["mpc_slave_forces"][2], -load_vector)
    np.testing.assert_allclose(diagnostics["mpc_master_equivalent_forces"][1], -load_vector)
    assert diagnostics["support_reactions"] == {}
    assert len(diagnostics["mpc_constraint_forces"]) == 6


def test_mesh_node_coordinates_are_compact_for_sparse_node_ids():
    model = _steel_model("sparse_node_ids")
    model.add_node(10, 1.0, 2.0, 3.0)
    model.add_node(10000, 4.0, 5.0, 6.0)

    coords = model.mesh.get_node_coordinates()

    assert coords.shape == (2, 3)
    np.testing.assert_allclose(coords, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_gravity_load_uses_element_mass_and_material_density():
    model = FEModel("gravity_shell")
    model.add_material("light", 210.0e9, 0.3, density=10.0)
    model.current_material = "light"
    for node_id, (x, y, z) in {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        3: (2.0, 1.0, 0.0),
        4: (0.0, 1.0, 0.0),
    }.items():
        model.add_node(node_id, x, y, z)
    model.add_element(
        1, create_shell_element(1, [1, 2, 3, 4], "light", thickness=0.2)
    )

    load_case = LoadCase("gravity")
    load_case.set_gravity(0.0, 0.0, -9.81)
    _, F, _ = assemble_system(model, load_case)

    nodal_forces = F.reshape(-1, 6)[:, :3]
    mass = 10.0 * 0.2 * 2.0
    np.testing.assert_allclose(np.sum(nodal_forces, axis=0), [0.0, 0.0, -mass * 9.81])


def test_mass_matrix_is_not_added_to_stiffness_when_requested():
    model = _steel_model("separate_mass")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(
        1,
        BeamElement(
            1,
            [1, 2],
            "steel",
            {"area": 0.01, "Iy": 2.0e-6, "Iz": 3.0e-6, "J": 4.0e-6},
        ),
    )

    K_without_mass, _, _ = assemble_system(model)
    K_with_mass_flag, _, info = assemble_system(model, include_mass=True)
    M, mass_info = assemble_mass_matrix(model)

    np.testing.assert_allclose(K_with_mass_flag.toarray(), K_without_mass.toarray())
    np.testing.assert_allclose(info["mass_matrix"].toarray(), M.toarray())
    assert M.nnz > 0
    assert mass_info["total_dofs"] == model.mesh.dof_manager.total_dofs


def test_angle_cross_section_uses_composite_centroid_and_parallel_axis_terms():
    hw = 0.10
    tw = 0.01
    b = 0.08
    tf = 0.012

    section = StiffenerCrossSection.from_geometry("Angle", hw, tw, b, tf)

    web_area = hw * tw
    flange_area = b * tf
    area = web_area + flange_area
    y_c = (web_area * (tw / 2.0) + flange_area * (b / 2.0)) / area
    z_c = (web_area * (hw / 2.0) + flange_area * (hw + tf / 2.0)) / area
    expected_iy = (
        tw * hw**3 / 12.0
        + web_area * (hw / 2.0 - z_c) ** 2
        + b * tf**3 / 12.0
        + flange_area * (hw + tf / 2.0 - z_c) ** 2
    )
    expected_iz = (
        hw * tw**3 / 12.0
        + web_area * (tw / 2.0 - y_c) ** 2
        + tf * b**3 / 12.0
        + flange_area * (b / 2.0 - y_c) ** 2
    )

    assert section.area == pytest.approx(area)
    assert section.Iy == pytest.approx(expected_iy)
    assert section.Iz == pytest.approx(expected_iz)


def test_orthonormalize_columns_reorthogonalizes_nearly_dependent_modes():
    matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0e-8, 1.0],
            [0.0, 0.0, 1.0e-8],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    q, kept = _orthonormalize_columns(matrix, tolerance=1.0e-14)

    assert kept.tolist() == [0, 1, 2]
    np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1.0e-12)


def test_acceleration_and_added_node_mass_load():
    """Acceleration body load plus inertial load from added edge/node masses."""
    model = FEModel("accel")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    for node_id, (x, y) in {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0), 4: (0.0, 1.0)}.items():
        model.add_node(node_id, x, y, 0.0)
    model.add_element(
        1, create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.01)
    )
    structural_mass = 7850.0 * 0.01 * 1.0

    # acceleration in +x produces a body load m * a
    lc = LoadCase("accel_x")
    lc.set_acceleration(3.0, 0.0, 0.0)
    _, F, _ = assemble_system(model, lc)
    fx = np.sum(F.reshape(-1, 6)[:, 0])
    assert fx == pytest.approx(structural_mass * 3.0)

    # added edge mass (200 kg over nodes 1,2) adds inertial load under -z accel
    lc2 = LoadCase("edge_mass")
    lc2.set_acceleration(0.0, 0.0, -9.81)
    lc2.add_distributed_edge_mass([1, 2], 200.0)
    _, F2, _ = assemble_system(model, lc2)
    fz = np.sum(F2.reshape(-1, 6)[:, 2])
    assert fz == pytest.approx((structural_mass + 200.0) * -9.81)
    # the 200 kg splits 100/100 onto nodes 1 and 2 only
    node1_extra = F2[model.mesh.get_node(1).dofs[2]] - F2[model.mesh.get_node(4).dofs[2]]
    assert node1_extra == pytest.approx(100.0 * -9.81)

    # a single node mass
    lc3 = LoadCase("node_mass")
    lc3.set_acceleration(0.0, 0.0, -9.81)
    lc3.add_node_mass(3, 50.0)
    _, F3, _ = assemble_system(model, lc3)
    assert F3[model.mesh.get_node(3).dofs[2]] - F3[model.mesh.get_node(1).dofs[2]] == pytest.approx(50.0 * -9.81)
