"""Architecture-level verification tests for the modular FE solver."""

from __future__ import annotations

import numpy as np

from anyfesolver import (
    LoadCase,
    MeshConfig,
    PanelGeometry,
    assemble_load_vector,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    assemble_system,
    create_fe_result,
    dof_order_signature,
    generate_beam_mesh,
    generate_simple_panel_mesh,
    generate_stiffened_panel_mesh,
    load_case_resultant,
    max_abs,
    mpc_constraint_residuals,
    nullspace_diagnostics,
    solve_linear,
)


def test_dof_order_is_fixed() -> None:
    model = generate_simple_panel_mesh(1.0, 0.5, 0.01, num_divisions_x=1, num_divisions_y=1)
    signature = dof_order_signature(model)
    first_node = signature[1]
    assert [name for _dof, name in first_node] == ["ux", "uy", "uz", "rx", "ry", "rz"]
    assert [dof for dof, _name in first_node] == [0, 1, 2, 3, 4, 5]


def test_explicit_stiffness_mass_and_load_apis_are_separate() -> None:
    cross_section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
    model = generate_beam_mesh(1.0, num_divisions=1, cross_section=cross_section)
    model.materials["steel"].density = 7850.0

    load_case = LoadCase("nodal")
    load_case.add_nodal_load(2, [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    K, stiffness_info = assemble_stiffness_matrix(model)
    M, mass_info = assemble_mass_matrix(model)
    F, load_info = assemble_load_vector(model, load_case)

    assert K.shape == M.shape
    assert F.shape == (model.mesh.dof_manager.total_dofs,)
    assert stiffness_info["matrix_type"] == "stiffness"
    assert mass_info["matrix_type"] == "mass"
    assert load_info["vector_type"] == "load"
    assert np.linalg.norm(F) > 0.0
    assert K.nnz > 0
    assert M.nnz > 0


def test_assemble_system_wrapper_matches_explicit_apis() -> None:
    cross_section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
    model = generate_beam_mesh(1.0, num_divisions=1, cross_section=cross_section)
    model.materials["steel"].density = 7850.0

    load_case = LoadCase("nodal")
    load_case.add_nodal_load(2, [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    K_explicit, _stiffness_info = assemble_stiffness_matrix(model)
    M_explicit, _mass_info = assemble_mass_matrix(model)
    F_explicit, _load_info = assemble_load_vector(model, load_case)

    K, F, info = assemble_system(model, load_case, include_mass=True)

    assert (K - K_explicit).nnz == 0
    assert np.allclose(F, F_explicit)
    assert "mass_matrix" in info
    assert (info["mass_matrix"] - M_explicit).nnz == 0
    assert info["includes_mass_matrix"] is True
    assert info["stiffness"]["matrix_type"] == "stiffness"
    assert info["mass"]["matrix_type"] == "mass"
    assert info["load"]["vector_type"] == "load"


def test_pressure_resultant_matches_plate_area() -> None:
    length = 2.0
    width = 1.5
    pressure = 12_345.0
    model = generate_simple_panel_mesh(length, width, 0.01, num_divisions_x=2, num_divisions_y=2)
    load_case = LoadCase("pressure")
    for elem_id in model.mesh.elements:
        load_case.add_pressure_load(elem_id, pressure=pressure)

    resultant = load_case_resultant(model, load_case)

    expected_force_z = pressure * length * width
    assert np.isclose(resultant.force[0], 0.0, atol=1.0e-8)
    assert np.isclose(resultant.force[1], 0.0, atol=1.0e-8)
    assert np.isclose(resultant.force[2], expected_force_z, rtol=1.0e-10, atol=1.0e-8)


def test_free_free_self_equilibrated_load_uses_nullspace() -> None:
    # A two-node beam has the expected six rigid-body modes and no extra shell
    # hourglass/mechanism modes, so it is a clean architecture test for the
    # nullspace solver path.
    cross_section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
    model = generate_beam_mesh(1.0, num_divisions=1, cross_section=cross_section)
    model.boundary_conditions.clear()

    load_case = LoadCase("self_equilibrated")
    load_case.add_nodal_load(1, [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_case.add_nodal_load(2, [-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    _u, solver_info = solve_linear(model, load_case, constraint_mode="auto")
    diagnostics = nullspace_diagnostics(solver_info)

    assert diagnostics["status"] == "converged"
    assert "nullspace" in diagnostics["constraint_method"]
    assert diagnostics["rank"] == 6
    assert diagnostics["relative_rigid_body_load_imbalance"] < 1.0e-8


def test_interpolated_beam_shell_mpc_is_satisfied_after_solve() -> None:
    panel = PanelGeometry(
        length=2.0,
        width=1.0,
        plate_thickness=0.01,
        stiffener_spacing=0.5,
        stiffener_height=0.15,
        stiffener_web_thickness=0.01,
        stiffener_flange_width=0.08,
        stiffener_flange_thickness=0.01,
        num_stiffeners=1,
        in_plane_support="Integrated",
        rotational_support="SS",
    )
    config = MeshConfig(shell_num_divisions_x=2, shell_num_divisions_y=2, beam_num_divisions=2, use_coupling_elements=True)
    model = generate_stiffened_panel_mesh(panel, config)
    load_case = LoadCase("small_load")
    load_case.add_nodal_load(10002, [0.0, 0.0, -100.0, 0.0, 0.0, 0.0])

    displacements, solver_info = solve_linear(model, load_case, constraint_mode="auto")
    residuals = mpc_constraint_residuals(model, displacements)

    assert solver_info["convergence_info"].get("status") == "converged"
    assert residuals
    assert max_abs(residuals.values()) < 1.0e-9

    result = create_fe_result(model, displacements, solver_info)
    assert result.element_stresses
