"""Tests and benchmarks for vectorized stiffness and JIT nonlinear response integrations."""

from __future__ import annotations

import time

import numpy as np
import pytest

from anyfesolver.elements import LegacyShellElement, create_element
from anyfesolver.fe_core import FEModel
from anyfesolver.jit_compiler import JIT_DISABLED_REASON, JIT_ENABLED
from anyfesolver.kernel_warmup import warm_fe_solver_kernels
from anyfesolver.matrix_assembly import (
    assemble_geometric_stiffness_matrix,
    assemble_stiffness_matrix,
)
from anyfesolver.mesh_gen import generate_simple_panel_mesh
from anyfesolver.vectorized_stiffness import compute_shell_stiffness_matrices_jit
import anyfesolver.kernel_warmup as kernel_warmup


def test_batched_s4_geometric_stiffness_matches_warped_scalar_and_reuses_geometry():
    model = FEModel("batched_kg_qualification")
    model.add_material("steel", 210.0e9, 0.3)
    coordinates = (
        (0.0, 0.0, 0.0),
        (1.2, 0.1, 0.15),
        (1.1, 1.0, 0.25),
        (-0.1, 0.9, -0.05),
        (2.0, -0.2, 0.3),
        (2.8, 0.2, 0.8),
        (2.6, 1.1, 1.0),
        (1.8, 0.8, 0.45),
    )
    for node_id, point in enumerate(coordinates, start=1):
        model.add_node(node_id, *point)
    first = create_element("shell", 1, [1, 2, 3, 4], "steel", thickness=0.018)
    second = create_element("shell", 2, [5, 6, 7, 8], "steel", thickness=0.031)
    model.add_element(1, first)
    model.add_element(2, second)
    states = {
        1: {
            "membrane_compression_at_gauss": np.array(
                [[12.0, 7.0, 1.0], [11.0, 8.0, -0.5], [9.0, 6.0, 0.3], [13.0, 5.0, 0.8]]
            ),
            "bending_compression": [0.7, -0.2, 0.1],
        },
        2: {
            "membrane_forces": [-16.0, -4.0, 2.0],
            "stress_second_moment": [0.04, 0.03, -0.01],
        },
    }
    expected = np.zeros((model.mesh.dof_manager.total_dofs,) * 2)
    for element_id, element in model.mesh.elements.items():
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        local = element.compute_geometric_stiffness_matrix(
            model.mesh,
            model.get_material(element.material_name),
            states[element_id],
        )
        expected[np.ix_(dofs, dofs)] += local

    actual, first_info = assemble_geometric_stiffness_matrix(model, states)
    repeated, second_info = assemble_geometric_stiffness_matrix(model, states)
    np.testing.assert_allclose(actual.toarray(), expected, rtol=2.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(repeated.toarray(), expected, rtol=2.0e-12, atol=1.0e-12)
    first_diag = first_info["diagnostics"]["vectorized_s4_geometric_stiffness"]
    second_diag = second_info["diagnostics"]["vectorized_s4_geometric_stiffness"]
    assert first_diag["element_count"] == 2
    assert first_diag["geometry_cache_hits"] == 0
    assert second_diag["geometry_cache_hits"] == 1
    assert second_info["diagnostics"]["scalar_element_count"] == 0


def test_vectorized_stiffness_matches_sequential():
    """Verify that vectorized JIT-compiled shell stiffness matches sequential output."""
    model = FEModel("stiffness_test")
    model.add_material("steel", 210.0e9, 0.3)

    # 4-node shell element nodes
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.2, 0.0, 0.0)
    model.add_node(3, 1.2, 1.1, 0.0)
    model.add_node(4, 0.0, 1.1, 0.0)

    # 8-node shell element nodes
    model.add_node(5, 2.0, 0.0, 0.0)
    model.add_node(6, 3.5, 0.0, 0.0)
    model.add_node(7, 3.5, 1.5, 0.0)
    model.add_node(8, 2.0, 1.5, 0.0)
    # mid-side nodes
    model.add_node(9, 2.75, 0.0, 0.0)
    model.add_node(10, 3.5, 0.75, 0.0)
    model.add_node(11, 2.75, 1.5, 0.0)
    model.add_node(12, 2.0, 0.75, 0.0)

    # Add elements
    el1 = LegacyShellElement(1, [1, 2, 3, 4], "steel", thickness=0.02)
    el2 = LegacyShellElement(
        2, [5, 6, 7, 8, 9, 10, 11, 12], "steel", thickness=0.02
    )
    model.add_element(1, el1)
    model.add_element(2, el2)

    mesh = model.mesh
    material = model.get_material("steel")

    # Compute stiffness matrices sequentially
    K1_seq = el1.compute_stiffness_matrix(mesh, material)
    K2_seq = el2.compute_stiffness_matrix(mesh, material)

    # Vectorized computation for Q4
    coords_all_q4 = np.array([el1.get_node_coordinates(mesh)])
    K_all_q4 = compute_shell_stiffness_matrices_jit(
        coords_all_q4,
        is_4node=True,
        thickness=0.02,
        drilling_stabilization=1.0e-3,
        E=float(material.elastic_modulus),
        nu=float(material.poisson_ratio),
        G=float(material.shear_modulus),
        gauss_points=el1.gauss_points,
        gauss_weights=el1.gauss_weights,
        shear_points=np.empty((0, 2)),
        shear_weights=np.empty(0),
    )

    # Vectorized computation for Q8
    coords_all_q8 = np.array([el2.get_node_coordinates(mesh)])
    K_all_q8 = compute_shell_stiffness_matrices_jit(
        coords_all_q8,
        is_4node=False,
        thickness=0.02,
        drilling_stabilization=1.0e-3,
        E=float(material.elastic_modulus),
        nu=float(material.poisson_ratio),
        G=float(material.shear_modulus),
        gauss_points=el2.gauss_points,
        gauss_weights=el2.gauss_weights,
        shear_points=el2.shear_gauss_points,
        shear_weights=el2.shear_gauss_weights,
    )

    # Verify outputs match exactly
    assert np.allclose(K_all_q4[0], K1_seq, rtol=1.0e-10, atol=1.0e-3)
    assert np.allclose(K_all_q8[0], K2_seq, rtol=1.0e-10, atol=1.0e-3)


@pytest.mark.parametrize("use_8node_elements", [False, True])
def test_parallel_legacy_shell_stiffness_batch_matches_legacy_reference(
    use_8node_elements: bool,
) -> None:
    model = generate_simple_panel_mesh(
        2.0,
        1.0,
        0.012,
        num_divisions_x=2,
        num_divisions_y=1,
        use_8node_elements=use_8node_elements,
    )
    material = model.get_material("steel")
    elements = list(model.mesh.elements.values())
    first = elements[0]
    coords_all = np.asarray([element.get_node_coordinates(model.mesh) for element in elements], dtype=float)
    K_all = compute_shell_stiffness_matrices_jit(
        coords_all,
        is_4node=bool(getattr(first, "_is_4node", False)),
        thickness=float(first.thickness),
        drilling_stabilization=float(first.drilling_stabilization),
        E=float(material.elastic_modulus),
        nu=float(material.poisson_ratio),
        G=float(material.shear_modulus),
        gauss_points=first.gauss_points,
        gauss_weights=first.gauss_weights,
        shear_points=np.empty((0, 2)) if getattr(first, "_is_4node", False) else first.shear_gauss_points,
        shear_weights=np.empty(0) if getattr(first, "_is_4node", False) else first.shear_gauss_weights,
    )
    for index, element in enumerate(elements):
        # ``compute_shell_stiffness_matrices_jit`` is the retained legacy
        # kernel.  Q4 panel builders now select the qualified E4-PL element,
        # so compare this low-level compatibility kernel with an explicit
        # legacy rollback element rather than the production Q4 default.
        reference_element = element
        if bool(getattr(element, "_is_4node", False)):
            reference_element = LegacyShellElement(
                int(element.element_id),
                list(element.node_ids),
                str(element.material_name),
                thickness=float(element.thickness),
                drilling_stabilization=float(element.drilling_stabilization),
                reduced_integration=bool(element.reduced_integration),
                hourglass_stabilization=float(element.hourglass_stabilization),
            )
        reference = reference_element.compute_stiffness_matrix(model.mesh, material)
        np.testing.assert_allclose(K_all[index], reference, rtol=1.0e-10, atol=1.0e-3)


def test_warm_fe_solver_kernels_reports_shell_orders_and_consistent_matrices() -> None:
    report = warm_fe_solver_kernels(("S4", "Q8R"))
    assert report["status"] == "completed"
    assert {"S4", "Q8R"} <= set(report["shell_orders"])
    for item in report["shell_orders"].values():
        assert item["cold_assembly_seconds"] >= 0.0
        assert item["warm_assembly_seconds"] >= 0.0
        assert item["matrix_difference_norm"] < 1.0e-12
        assert "jit_enabled" in item


def test_warm_fe_solver_kernels_reports_jit_fallback_state(monkeypatch) -> None:
    monkeypatch.setattr(kernel_warmup, "JIT_ENABLED", False)
    monkeypatch.setattr(kernel_warmup, "JIT_DISABLED_REASON", "unit_test_fallback")
    report = kernel_warmup.warm_fe_solver_kernels(("S4",))
    item = report["shell_orders"]["S4"]
    assert item["jit_enabled"] is False
    assert item["jit_disabled_reason"] == "unit_test_fallback"


def test_warm_fe_solver_kernels_can_precompile_nonlinear_q4_batch() -> None:
    report = warm_fe_solver_kernels(("S4",), include_nonlinear_static=True)

    nonlinear_static = report["nonlinear_static"]
    assert nonlinear_static is not None
    assert nonlinear_static["status"] == "completed"
    assert nonlinear_static["element_count"] == 16
    assert nonlinear_static["nonlinear_layers"] == 5
    assert nonlinear_static["seconds"] >= 0.0
    assert nonlinear_static["tangent_nnz"] > 0
    assert nonlinear_static["internal_force_inf"] < 1.0e-12


def test_nonlinear_response_jit_matches_sequential():
    """Verify JIT-compiled nonlinear integration matches a sequential Python implementation."""
    model = FEModel("nonlin_test")
    model.add_material("steel", 210.0e9, 0.3)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.0, 1.0, 0.0)
    model.add_node(4, 0.0, 1.0, 0.0)
    el = LegacyShellElement(1, [1, 2, 3, 4], "steel", thickness=0.02)
    model.add_element(1, el)

    mesh = model.mesh
    material = model.get_material("steel")

    # Arbitrary element displacements
    u_elem = np.zeros(24)
    u_elem[2::6] = 0.01  # z deflection
    u_elem[3::6] = 0.005  # rot_x
    u_elem[4::6] = -0.002  # rot_y

    # JIT implementation output
    F_jit, K_jit, _ = el.compute_nonlinear_response(mesh, material, u_elem, tangent=True)

    # Sequential reference calculations
    cache = el._nonlinear_geometry(mesh)
    T0 = cache["T0"]
    u_loc = T0 @ u_elem

    E = material.elastic_modulus
    nu = material.poisson_ratio
    G_mod = material.shear_modulus
    h = el.thickness
    C_el = E / (1.0 - nu**2) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
    )
    D_shear = G_mod * (5.0 / 6.0) * h * np.eye(2)
    drilling_stiffness = G_mod * h * 1.0e-3

    n_gp = len(cache["gp"])

    memb_strain = np.zeros((n_gp, 3))
    curvature = np.zeros((n_gp, 3))
    B_eff_list = []
    for g, gp in enumerate(cache["gp"]):
        theta = gp["Gw"] @ u_loc
        B_nl = np.vstack(
            [
                theta[0] * gp["Gw"][0],
                theta[1] * gp["Gw"][1],
                theta[0] * gp["Gw"][1] + theta[1] * gp["Gw"][0],
            ]
        )
        B_eff = gp["B_m"] + B_nl
        memb_strain[g] = gp["B_m"] @ u_loc + np.array(
            [0.5 * theta[0] ** 2, 0.5 * theta[1] ** 2, theta[0] * theta[1]]
        )
        curvature[g] = gp["B_b"] @ u_loc
        B_eff_list.append(B_eff)

    N_res = memb_strain @ (h * C_el).T
    M_res = curvature @ (h**3 / 12.0 * C_el).T
    C0 = np.broadcast_to(h * C_el, (n_gp, 3, 3))
    C2 = np.broadcast_to(h**3 / 12.0 * C_el, (n_gp, 3, 3))

    n_dof = 24
    F_seq = np.zeros(n_dof)
    K_seq = np.zeros((n_dof, n_dof))
    for g, gp in enumerate(cache["gp"]):
        detw = gp["detw"]
        B_eff = B_eff_list[g]
        B_b = gp["B_b"]
        B_d = gp["B_d"]
        F_seq += (B_eff.T @ N_res[g] + B_b.T @ M_res[g]) * detw
        F_seq += B_d.T @ (drilling_stiffness * (B_d @ u_loc)) * detw

        K_seq += (B_eff.T @ C0[g] @ B_eff + B_b.T @ C2[g] @ B_b) * detw
        N_mat = np.array([[N_res[g, 0], N_res[g, 2]], [N_res[g, 2], N_res[g, 1]]])
        K_seq += gp["Gw"].T @ N_mat @ gp["Gw"] * detw
        K_seq += B_d.T @ (drilling_stiffness * B_d) * detw

    for sh in cache["shear"]:
        B_s = sh["B_s"]
        K_s = B_s.T @ D_shear @ B_s * sh["detw"]
        F_seq += K_s @ u_loc
        K_seq += K_s

    F_global_seq = T0.T @ F_seq
    K_global_seq = T0.T @ K_seq @ T0

    # Compare JIT against sequential values
    assert np.allclose(F_jit, F_global_seq, rtol=1.0e-10, atol=1.0e-3)
    assert np.allclose(K_jit, K_global_seq, rtol=1.0e-10, atol=1.0e-3)


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason=f"compiled-kernel timing requires Numba ({JIT_DISABLED_REASON})",
)
def test_vectorized_stiffness_performance():
    """Benchmark warm compiled assembly without treating Python fallback as JIT."""
    model = FEModel("benchmark")
    model.add_material("steel", 210.0e9, 0.3)

    # Create a 20x20 grid of shell nodes and elements (400 elements)
    n_div = 20
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            model.add_node(i * (n_div + 1) + j + 1, float(i), float(j), 0.0)

    elem_id = 1
    for i in range(n_div):
        for j in range(n_div):
            n1 = i * (n_div + 1) + j + 1
            n2 = n1 + 1
            n4 = n1 + (n_div + 1)
            n3 = n4 + 1
            el = LegacyShellElement(
                elem_id, [n1, n2, n3, n4], "steel", thickness=0.01
            )
            model.add_element(elem_id, el)
            elem_id += 1

    # Warm the vectorized/JIT path first. Cold compile time is a separate
    # performance metric and should not be mixed into steady-state assembly.
    t0_cold = time.time()
    assemble_stiffness_matrix(model)
    t1_cold = time.time()
    cold_time = t1_cold - t0_cold

    # Measure warm global stiffness assembly time
    t0 = time.time()
    assemble_stiffness_matrix(model)
    t1 = time.time()
    vectorized_time = t1 - t0

    # Compute sequential element-by-element times
    mesh = model.mesh
    material = model.get_material("steel")
    t0_seq = time.time()
    for _, el in mesh.elements.items():
        el.compute_stiffness_matrix(mesh, material)
    t1_seq = time.time()
    sequential_time = t1_seq - t0_seq

    print(f"\nCold vectorized assembly time: {cold_time:.6f} s")
    print(f"Warm vectorized assembly time: {vectorized_time:.6f} s")
    print(f"Sequential assembly time: {sequential_time:.6f} s")
    print(f"Speedup: {sequential_time / max(vectorized_time, 1e-9):.2f}x")

    assert vectorized_time < sequential_time or vectorized_time < 0.2


def test_vectorized_mass_matches_sequential_and_assembly_uses_batch():
    """Batched shell mass kernel matches element.compute_mass_matrix and drives assemble_mass_matrix."""
    from scipy import sparse

    from anyfesolver.matrix_assembly import assemble_mass_matrix
    from anyfesolver.vectorized_stiffness import compute_shell_mass_matrices_jit

    model = FEModel("mass_test")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)

    # distorted 4-node shell
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.35, 0.08, 0.02)
    model.add_node(3, 1.08, 0.95, -0.03)
    model.add_node(4, -0.18, 1.12, 0.01)
    # distorted 8-node shell
    model.add_node(5, 2.0, 0.0, 0.0)
    model.add_node(6, 3.5, 0.1, 0.02)
    model.add_node(7, 3.4, 1.5, 0.0)
    model.add_node(8, 2.0, 1.4, -0.02)
    model.add_node(9, 2.8, 0.02, 0.01)
    model.add_node(10, 3.5, 0.8, 0.0)
    model.add_node(11, 2.7, 1.5, 0.01)
    model.add_node(12, 1.95, 0.7, 0.0)

    el1 = create_element("shell", 1, [1, 2, 3, 4], "steel", thickness=0.02)
    el2 = LegacyShellElement(
        2, [5, 6, 7, 8, 9, 10, 11, 12], "steel", thickness=0.02
    )
    model.add_element(1, el1)
    model.add_element(2, el2)

    mesh = model.mesh
    material = model.get_material("steel")

    for element, is_4node in ((el1, True), (el2, False)):
        M_seq = element.compute_mass_matrix(mesh, material)
        M_batch = compute_shell_mass_matrices_jit(
            np.array([element.get_node_coordinates(mesh)], dtype=float),
            is_4node=is_4node,
            thickness=float(element.thickness),
            rho=float(material.density),
            gauss_points=element.gauss_points,
            gauss_weights=element.gauss_weights,
        )[0]
        scale = float(np.max(np.abs(M_seq)))
        assert np.max(np.abs(M_batch - M_seq)) < 1.0e-12 * scale

    M_fast, info = assemble_mass_matrix(model)
    assert info["diagnostics"]["vectorized_shell_element_count"] == 2
    assert all(group["kernel"] == "compute_shell_mass_matrices_jit" for group in info["diagnostics"]["vectorized_shell_groups"])

    total = mesh.dof_manager.total_dofs
    rows, cols, data = [], [], []
    for element in (el1, el2):
        dofs = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        m = element.compute_mass_matrix(mesh, material)
        rows.append(np.repeat(dofs, dofs.size))
        cols.append(np.tile(dofs, dofs.size))
        data.append(m.ravel())
    M_ref = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(total, total),
    ).tocsr()
    assert abs(M_fast - M_ref).max() < 1.0e-12 * abs(M_ref).max()


def test_q8r_mass_assembly_keeps_scalar_lumped_path():
    """Reduced-integration S8R shells stay on the scalar lumped mass path."""
    from anyfesolver.matrix_assembly import assemble_mass_matrix

    model = FEModel("q8r_mass_test")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.5, 0.0, 0.0)
    model.add_node(3, 1.5, 1.5, 0.0)
    model.add_node(4, 0.0, 1.5, 0.0)
    model.add_node(5, 0.75, 0.0, 0.0)
    model.add_node(6, 1.5, 0.75, 0.0)
    model.add_node(7, 0.75, 1.5, 0.0)
    model.add_node(8, 0.0, 0.75, 0.0)
    element = LegacyShellElement(
        1,
        [1, 2, 3, 4, 5, 6, 7, 8],
        "steel",
        thickness=0.02,
        reduced_integration=True,
    )
    model.add_element(1, element)

    M, info = assemble_mass_matrix(model)
    assert info["diagnostics"]["vectorized_shell_element_count"] == 0
    reference = element.compute_mass_matrix(model.mesh, model.get_material("steel"))
    dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    assert np.allclose(M.toarray()[np.ix_(dofs, dofs)], reference)
