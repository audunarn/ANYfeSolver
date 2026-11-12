"""Follower-pressure and shell initial-stress qualification tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.buckling import solve_eigenvalue_buckling
from anyfesolver.cylinder_benchmarks import (
    CylinderBenchmarkConfig,
    build_cylindrical_shell_benchmark_model,
)
from anyfesolver.elements import (
    CoupledBeamShellElement,
    LegacyQ4DeprecationWarning,
    LegacyShellElement,
    ShellElement,
    create_shell_element,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.matrix_assembly import (
    assemble_external_load_tangent,
    assemble_geometric_stiffness_matrix,
    assemble_load_vector,
)
from anyfesolver.mesh_gen import generate_simple_panel_mesh
from anyfesolver.nonlinear_static import solve_static_nonlinear
from anyfesolver.validation import validate_production_model


def _single_shell(node_count: int = 4) -> tuple[FEModel, ShellElement]:
    model = FEModel(f"pressure_shell_{node_count}")
    model.add_material("steel", 210.0e9, 0.3)
    if node_count == 3:
        coordinates = [(0.0, 0.0, 0.0), (1.7, 0.1, 0.0), (0.2, 1.2, 0.0)]
    elif node_count == 6:
        corners = np.array([(0.0, 0.0, 0.0), (1.7, 0.1, 0.0), (0.2, 1.2, 0.0)])
        coordinates = [
            tuple(corners[0]),
            tuple(corners[1]),
            tuple(corners[2]),
            tuple(0.5 * (corners[0] + corners[1])),
            tuple(0.5 * (corners[1] + corners[2])),
            tuple(0.5 * (corners[2] + corners[0])),
        ]
    elif node_count == 4:
        coordinates = [
            (0.0, 0.0, 0.0),
            (1.7, 0.1, 0.0),
            (1.8, 1.2, 0.0),
            (0.2, 1.1, 0.0),
        ]
    elif node_count == 8:
        corners = np.array(
            [(0.0, 0.0, 0.0), (1.7, 0.1, 0.0), (1.8, 1.2, 0.0), (0.2, 1.1, 0.0)]
        )
        coordinates = [tuple(item) for item in corners] + [
            tuple(0.5 * (corners[0] + corners[1])),
            tuple(0.5 * (corners[1] + corners[2])),
            tuple(0.5 * (corners[2] + corners[3])),
            tuple(0.5 * (corners[3] + corners[0])),
        ]
    else:
        raise ValueError(node_count)
    for node_id, xyz in enumerate(coordinates, start=1):
        model.add_node(node_id, *xyz)
    node_ids = list(range(1, node_count + 1))
    element = (
        create_shell_element(1, node_ids, "steel", thickness=0.01)
        if node_count == 4
        else ShellElement(1, node_ids, "steel", thickness=0.01)
    )
    model.add_element(1, element)
    return model, element


def _legacy_q4_shell(
    element_id: int,
    node_ids: list[int],
    material_name: str,
    *,
    thickness: float,
) -> LegacyShellElement:
    """Construct an explicit rollback element for registered legacy fixtures."""

    with pytest.warns(LegacyQ4DeprecationWarning, match="temporary rollback"):
        return LegacyShellElement(
            element_id,
            node_ids,
            material_name,
            thickness=thickness,
        )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


@pytest.mark.parametrize("node_count", [3, 4, 6, 8])
def test_follower_pressure_tangent_matches_central_difference(node_count: int) -> None:
    model, element = _single_shell(node_count)
    load = LoadCase("follower", follower_pressure=True)
    load.add_pressure_load(1, 1730.0)
    rng = np.random.default_rng(731 + node_count)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in element.node_ids:
        node = model.mesh.get_node(node_id)
        displacement[node.dofs[:3]] = 0.08 * rng.standard_normal(3)
        displacement[node.dofs[3:]] = 0.2 * rng.standard_normal(3)

    tangent, info = assemble_external_load_tangent(model, load, displacement)
    analytical = tangent.toarray()
    step = 2.0e-7
    for node_id in element.node_ids:
        node = model.mesh.get_node(node_id)
        for dof in node.dofs[:3]:
            plus = displacement.copy()
            minus = displacement.copy()
            plus[dof] += step
            minus[dof] -= step
            f_plus, _ = assemble_load_vector(model, load, plus)
            f_minus, _ = assemble_load_vector(model, load, minus)
            numerical = (f_plus - f_minus) / (2.0 * step)
            np.testing.assert_allclose(analytical[:, dof], numerical, rtol=2.0e-8, atol=2.0e-6)

    rotational_dofs = np.concatenate(
        [np.asarray(model.mesh.get_node(node_id).dofs[3:], dtype=int) for node_id in element.node_ids]
    )
    np.testing.assert_allclose(analytical[:, rotational_dofs], 0.0, atol=0.0)
    assert info["pressure_configuration"] == "current"
    assert info["num_pressure_elements"] == 1


def test_follower_pressure_force_is_objective_under_rigid_rotation() -> None:
    model, element = _single_shell(4)
    load = LoadCase("follower", follower_pressure=True)
    load.add_pressure_load(1, 2200.0)
    reference_force, _ = assemble_load_vector(model, load)
    rotation = _axis_angle(np.array([0.3, -0.4, 0.8]), 0.73)
    translation = np.array([1.2, -0.7, 0.4])
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in element.node_ids:
        node = model.mesh.get_node(node_id)
        xyz = node.coords()
        displacement[node.dofs[:3]] = rotation @ xyz + translation - xyz

    rotated_force, _ = assemble_load_vector(model, load, displacement)
    for node_id in element.node_ids:
        dofs = model.mesh.get_node(node_id).dofs
        np.testing.assert_allclose(
            rotated_force[dofs[:3]],
            rotation @ reference_force[dofs[:3]],
            rtol=2.0e-13,
            atol=2.0e-10,
        )
        np.testing.assert_allclose(rotated_force[dofs[3:]], 0.0, atol=0.0)


def _clamped_pressure_plate(*, legacy_q4: bool = False) -> tuple[FEModel, int]:
    model = generate_simple_panel_mesh(1.0, 1.0, 0.01, 2, 2)
    if legacy_q4:
        for element_id, element in tuple(model.mesh.elements.items()):
            with pytest.warns(
                LegacyQ4DeprecationWarning, match="temporary rollback"
            ):
                rollback = LegacyShellElement(
                    int(element_id),
                    list(element.node_ids),
                    str(element.material_name),
                    thickness=float(element.thickness),
                    drilling_stabilization=float(element.drilling_stabilization),
                )
            model.add_element(int(element_id), rollback)
    model.clear_boundary_conditions()
    edge_nodes = []
    centre_node = -1
    for node_id, node in model.mesh.nodes.items():
        x, y, _ = node.coords()
        if min(x, y, 1.0 - x, 1.0 - y) <= 1.0e-12:
            edge_nodes.append(int(node_id))
        elif abs(x - 0.5) <= 1.0e-12 and abs(y - 0.5) <= 1.0e-12:
            centre_node = int(node_id)
    model.add_boundary_condition(
        BoundaryCondition("clamped_translations", edge_nodes, {"ux": 0.0, "uy": 0.0, "uz": 0.0})
    )
    return model, centre_node


def test_nonlinear_static_uses_follower_pressure_effective_tangent() -> None:
    model, centre_node = _clamped_pressure_plate()
    load = LoadCase("follower", follower_pressure=True)
    for element_id in model.mesh.elements:
        load.add_pressure_load(int(element_id), 2.0e5)

    result = solve_static_nonlinear(
        model,
        load,
        num_steps=5,
        max_iterations=30,
        tolerance=1.0e-7,
    )

    assert result.status == "completed"
    assert result.info["follower_pressure"] is True
    assert result.info["equilibrium_tangent"] == "K_internal-K_external"
    assert result.displacements[model.mesh.get_node(centre_node).dofs[2]] == pytest.approx(
        0.013869955023377319,
        rel=2.0e-5,
    )

    # The explicit rollback remains numerically frozen at the legacy response;
    # activating E4-PL must not silently remove that compatibility route.
    legacy_model, legacy_centre = _clamped_pressure_plate(legacy_q4=True)
    legacy_load = LoadCase("legacy_follower", follower_pressure=True)
    for element_id in legacy_model.mesh.elements:
        legacy_load.add_pressure_load(int(element_id), 2.0e5)
    legacy_result = solve_static_nonlinear(
        legacy_model,
        legacy_load,
        num_steps=5,
        max_iterations=30,
        tolerance=1.0e-7,
    )
    assert legacy_result.status == "completed"
    assert legacy_result.displacements[
        legacy_model.mesh.get_node(legacy_centre).dofs[2]
    ] == pytest.approx(0.013745211987271156, rel=2.0e-5)


def test_armijo_accepts_nonsymmetric_follower_pressure_tangent() -> None:
    model, centre_node = _clamped_pressure_plate()
    load = LoadCase("armijo_follower", follower_pressure=True)
    for element_id in model.mesh.elements:
        load.add_pressure_load(int(element_id), 2.0e4)

    result = solve_static_nonlinear(
        model,
        load,
        num_steps=2,
        max_iterations=25,
        tolerance=1.0e-7,
        convergence_settings={
            "profile": "legacy",
            "line_search": "armijo",
        },
    )

    assert result.status == "completed"
    assert result.info["equilibrium_tangent"] == "K_internal-K_external"
    assert result.info["armijo"]["enabled"] is True
    assert result.info["external_load_reduction"]["preprojected"] is False
    assert result.displacements[model.mesh.get_node(centre_node).dofs[2]] > 0.0


def test_corotational_follower_pressure_selects_consistent_tangent() -> None:
    model, centre_node = _clamped_pressure_plate()
    load = LoadCase("follower", follower_pressure=True)
    for element_id in model.mesh.elements:
        load.add_pressure_load(int(element_id), 2.0e4)

    result = solve_static_nonlinear(
        model,
        load,
        num_steps=3,
        max_iterations=30,
        tolerance=1.0e-6,
        kinematics="corotational",
    )

    assert result.status == "completed"
    assert result.info["corotational_tangent_requested"] == "auto"
    assert result.info["corotational_tangent"] == "consistent"
    assert result.displacements[
        model.mesh.get_node(centre_node).dofs[2]
    ] > 0.0

    with pytest.raises(NotImplementedError, match="consistent"):
        solve_static_nonlinear(
            _clamped_pressure_plate()[0],
            load,
            kinematics="corotational",
            corotational_tangent="rotated",
        )


def test_arc_length_accepts_current_area_follower_pressure() -> None:
    model, _centre_node = _clamped_pressure_plate()
    load = LoadCase("follower", follower_pressure=True)
    for element_id in model.mesh.elements:
        load.add_pressure_load(int(element_id), 2.0e4)
    result = solve_static_arc_length(
        model,
        load,
        control=ArcLengthControl(
            initial_load_increment=0.02,
            maximum_load_increment=0.05,
            maximum_absolute_load_factor=0.08,
            max_steps=8,
        ),
        max_iterations=25,
    )
    assert result.status == "load_factor_limit_reached"
    assert result.info["follower_pressure"] is True
    assert result.info["equilibrium_tangent"] == "K_internal-K_external"
    assert result.load_factor >= 0.08


def test_arc_length_accepts_corotational_follower_pressure() -> None:
    model, _centre_node = _clamped_pressure_plate()
    load = LoadCase("follower", follower_pressure=True)
    for element_id in model.mesh.elements:
        load.add_pressure_load(int(element_id), 2.0e4)

    result = solve_static_arc_length(
        model,
        load,
        kinematics="corotational",
        control=ArcLengthControl(
            initial_load_increment=0.02,
            maximum_load_increment=0.05,
            maximum_absolute_load_factor=0.06,
            max_steps=6,
        ),
        max_iterations=25,
    )

    assert result.status == "load_factor_limit_reached"
    assert result.info["corotational_tangent"] == "consistent"
    assert result.load_factor >= 0.06


def test_production_validation_scopes_follower_pressure_to_supported_analysis() -> None:
    model, _ = _single_shell(4)
    load = LoadCase("follower", follower_pressure=True)
    load.add_pressure_load(1, 1000.0)

    nonlinear = validate_production_model(
        model,
        [load],
        analysis_type="nonlinear_static",
        allow_free_mechanisms=True,
    )
    linear = validate_production_model(
        model,
        [load],
        analysis_type="linear_static",
        allow_free_mechanisms=True,
    )
    corotational = validate_production_model(
        model,
        [load],
        analysis_type="nonlinear_static",
        kinematics="corotational",
        allow_free_mechanisms=True,
    )
    inconsistent_corotational = validate_production_model(
        model,
        [load],
        analysis_type="nonlinear_static",
        kinematics="corotational",
        corotational_tangent="rotated",
        allow_free_mechanisms=True,
    )

    assert nonlinear.status == "ok"
    assert "LOAD001" not in {issue.code for issue in nonlinear.issues}
    assert "LOAD001" in {issue.code for issue in linear.issues}
    assert "LOAD001" not in {issue.code for issue in corotational.issues}
    assert "LOAD001" in {
        issue.code for issue in inconsistent_corotational.issues
    }


def test_buckling_rejects_open_nonsymmetric_follower_pressure_pencil() -> None:
    model, _ = _single_shell(4)
    load = LoadCase("open_patch", follower_pressure=True)
    load.add_pressure_load(1, 1.0)

    result = solve_eigenvalue_buckling(
        model,
        {},
        num_modes=1,
        reference_load_case=load,
        allow_dense_fallback=True,
        allow_free_mechanisms=True,
    )

    assert result.solver_status == "unsupported_nonsymmetric_follower_pencil"
    assert result.modes == []
    assert result.diagnostics["follower_tangent_symmetry_error"] > 1.0e-3
    assert "complex nonconservative eigenanalysis" in result.diagnostics["reason"]


def test_legacy_buckling_preserves_registered_closed_surface_follower_stiffness() -> None:
    """Preserve the pre-activation closed-cube eigenvalue as rollback evidence."""

    model = FEModel("closed_pressure_cube")
    model.add_material("steel", 2.0e5, 0.3)
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    for node_id, xyz in enumerate(points, start=1):
        model.add_node(node_id, *xyz)
    outward_faces = [
        [1, 4, 3, 2],
        [5, 6, 7, 8],
        [1, 2, 6, 5],
        [4, 8, 7, 3],
        [1, 5, 8, 4],
        [2, 3, 7, 6],
    ]
    for element_id, nodes in enumerate(outward_faces, start=1):
        model.add_element(
            element_id,
            _legacy_q4_shell(element_id, nodes, "steel", thickness=0.02),
        )
    model.add_boundary_condition(FixedSupport("fixed_bottom", [1, 2, 3, 4]))
    load = LoadCase("closed_pressure", follower_pressure=True)
    for element_id in model.mesh.elements:
        load.add_pressure_load(int(element_id), -1.0)

    result = solve_eigenvalue_buckling(
        model,
        {},
        num_modes=2,
        reference_load_case=load,
        allow_dense_fallback=True,
    )

    assert result.solver_status == "ok"
    assert result.num_modes_returned == 2
    assert result.critical_load_factor == pytest.approx(13292.45710889684, rel=2.0e-10)
    assert result.diagnostics["follower_load_stiffness_included"] is True
    assert result.diagnostics["follower_tangent_symmetry_error"] < 1.0e-14
    assert result.diagnostics["max_residual_norm"] < 1.0e-12


@pytest.mark.parametrize("circumferential_elements,max_error", [(24, 0.06), (32, 0.035)])
def test_thin_ring_follower_pressure_converges_to_analytical_buckling(
    circumferential_elements: int,
    max_error: float,
) -> None:
    """Plane ring guard: q_cr=3*E*I/R^3 per unit axial width."""
    config = CylinderBenchmarkConfig(
        radius=1.0,
        height=1.0,
        thickness=0.02,
        pressure=1.0,
        num_circumferential=circumferential_elements,
        num_height=1,
        closed_end_axial_load=False,
    )
    model, pressure = build_cylindrical_shell_benchmark_model(config)
    pressure.follower_pressure = True
    # Enforce the plane-ring kinematics by tying the upper edge to the lower
    # edge with zero-offset exact MPCs.  This removes axial variation without
    # adding penalty stiffness.
    for index in range(circumferential_elements):
        model.add_element(
            1000 + index,
            CoupledBeamShellElement(
                1000 + index,
                beam_node_id=circumferential_elements + 1 + index,
                shell_node_id=1 + index,
                material_name="steel",
                eccentricity=np.zeros(3),
            ),
        )
    polygon_membrane_compression = (
        config.pressure
        * config.radius
        * np.cos(np.pi / circumferential_elements)
    )
    unit_pressure_prestress = {
        int(element_id): {
            "membrane_compression_x": polygon_membrane_compression,
            "membrane_compression_y": 0.0,
            "membrane_compression_xy": 0.0,
        }
        for element_id, element in model.mesh.elements.items()
        if isinstance(element, ShellElement)
    }
    follower = solve_eigenvalue_buckling(
        model,
        unit_pressure_prestress,
        num_modes=2,
        reference_load_case=pressure,
        dense_size_limit=10_000,
        allow_dense_fallback=True,
    )
    dead = solve_eigenvalue_buckling(
        model,
        unit_pressure_prestress,
        num_modes=2,
        dense_size_limit=10_000,
        allow_dense_fallback=True,
    )
    bending_stiffness = (
        config.elastic_modulus
        * config.thickness**3
        / (12.0 * (1.0 - config.poisson_ratio**2))
    )
    analytical_pressure = 3.0 * bending_stiffness / config.radius**3

    assert follower.solver_status == "ok", (
        follower.diagnostics.get("reason"),
        {
            key: follower.diagnostics.get("rigid_projection", {}).get(key)
            for key in (
                "elastic_null_residual",
                "geometric_null_residual",
                "elastic_cross_residual",
                "geometric_cross_residual",
            )
        },
    )
    assert follower.critical_load_factor is not None
    relative_error = abs(float(follower.critical_load_factor) / analytical_pressure - 1.0)
    assert relative_error < max_error
    assert follower.diagnostics["solver"] == "dense_scipy_eigh_rigid_quotient"
    assert follower.diagnostics["rigid_body_handling"] == "projected"
    follower_projection = follower.diagnostics["rigid_projection"]
    assert follower_projection["applied"] is True
    assert follower_projection["metric_version"] == "dimensionless_full_dof_bbox_v1"
    # The upper-to-lower plane-ring MPCs preserve four compatible rigid modes;
    # rotations about the two in-plane axes are intentionally excluded.
    assert follower_projection["rigid_rank"] == follower.diagnostics["nullspace_rank"] == 4
    follower_tolerance = (
        4096.0
        * follower_projection["original_dofs"]
        * 2.220446049250313e-16
    )
    assert follower_projection["elastic_null_residual"] <= follower_tolerance
    assert follower_projection["geometric_null_residual"] <= follower_tolerance
    assert follower_projection["elastic_cross_residual"] <= follower_tolerance
    assert follower_projection["geometric_cross_residual"] <= follower_tolerance
    assert all(
        item["positive_definite"]
        for item in follower_projection["spd_sensitivity"].values()
    )
    assert follower.diagnostics["follower_load_stiffness_included"] is True
    assert follower.diagnostics["follower_tangent_symmetry_error"] < 1.0e-12
    # Without the follower load tangent, the imposed dead-prestress operator
    # does not descend to the free ring's rigid quotient.  An eigenvalue would
    # depend on the chosen rigid representative, so the solver must fail closed.
    assert dead.solver_status == "invalid_rigid_quotient"
    assert dead.modes == []
    assert dead.diagnostics["solver"] == "dense_scipy_eigh_rigid_quotient"
    assert "does not descend" in dead.diagnostics["reason"]
    dead_projection = dead.diagnostics["rigid_projection"]
    dead_tolerance = 4096.0 * dead_projection["original_dofs"] * 2.220446049250313e-16
    assert dead_projection["applied"] is True
    assert dead_projection["geometric_null_residual"] > dead_tolerance
    assert dead_projection["geometric_cross_residual"] > dead_tolerance


def test_shell_initial_stress_acts_on_all_translations_and_supports_gauss_fields() -> None:
    model, element = _single_shell(4)
    uniform = {"membrane_compression": [120.0, 45.0, 17.0]}
    sampled = {
        "membrane_compression_at_gauss": np.repeat(
            np.array([[120.0, 45.0, 17.0]]),
            len(element.gauss_points),
            axis=0,
        )
    }
    K_uniform, info = assemble_geometric_stiffness_matrix(model, {1: uniform})
    K_sampled, _ = assemble_geometric_stiffness_matrix(model, {1: sampled})
    dense = K_uniform.toarray()

    np.testing.assert_allclose(K_sampled.toarray(), dense, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(dense, dense.T, rtol=0.0, atol=1.0e-12)
    for component in range(3):
        dofs = np.arange(component, element.total_dofs, 6)
        assert np.linalg.norm(dense[np.ix_(dofs, dofs)]) > 0.0
    translation_x = np.arange(0, element.total_dofs, 6)
    rotation_x = np.arange(3, element.total_dofs, 6)
    rotation_y = np.arange(4, element.total_dofs, 6)
    drilling = np.arange(5, element.total_dofs, 6)
    np.testing.assert_allclose(
        dense[np.ix_(rotation_x, rotation_x)],
        element.thickness**2 / 12.0 * dense[np.ix_(translation_x, translation_x)],
        rtol=2.0e-13,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        dense[np.ix_(rotation_y, rotation_y)],
        element.thickness**2 / 12.0 * dense[np.ix_(translation_x, translation_x)],
        rtol=2.0e-13,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(dense[drilling, :], 0.0, atol=0.0)
    assert (
        info["diagnostics"]["shell_initial_stress_scope"]
        == "mindlin_translations_and_director_gradients; no_drilling_or_transverse_normal_stress_terms"
    )


def test_shell_initial_stress_energy_matches_through_thickness_quadrature() -> None:
    model = FEModel("initial_stress_energy")
    model.add_material("steel", 210.0e9, 0.3)
    for node_id, xyz in enumerate(
        [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (1.4, 0.9, 0.0), (0.0, 0.9, 0.0)],
        start=1,
    ):
        model.add_node(node_id, *xyz)
    element = create_shell_element(
        1, [1, 2, 3, 4], "steel", thickness=0.12
    )
    model.add_element(1, element)
    membrane = np.array([80.0, 31.0, -9.0])
    bending = np.array([1.8, -0.7, 0.35])
    state = {
        1: {
            "membrane_compression": membrane,
            "bending_compression": bending,
        }
    }
    K, _ = assemble_geometric_stiffness_matrix(model, state)
    rng = np.random.default_rng(20260726)
    displacement = rng.normal(scale=0.03, size=element.total_dofs)

    coords = element.get_node_coordinates(model.mesh)
    direct = 0.0
    z_points, z_weights = np.polynomial.legendre.leggauss(5)
    half_thickness = 0.5 * element.thickness
    for (xi, eta), surface_weight in zip(element.gauss_points, element.gauss_weights):
        N, dN_dxi, dN_deta = element.compute_shape_functions(float(xi), float(eta))
        R, dN_dx, dN_dy, det_j = element._local_frame_and_derivatives(
            coords,
            dN_dxi,
            dN_deta,
        )
        local = element._local_dof_transform(R) @ displacement
        del N
        for zeta, z_weight in zip(z_points, z_weights):
            z = half_thickness * float(zeta)
            stress = membrane / element.thickness + 12.0 * bending * z / element.thickness**3
            stress_matrix = np.array(
                [[stress[0], stress[2]], [stress[2], stress[1]]],
                dtype=float,
            )
            fields = (
                local[0::6] + z * local[4::6],
                local[1::6] - z * local[3::6],
                local[2::6],
            )
            density = 0.0
            for field in fields:
                gradient = np.array([dN_dx @ field, dN_dy @ field])
                density += float(gradient @ stress_matrix @ gradient)
            direct += density * det_j * float(surface_weight) * half_thickness * float(z_weight)

    matrix_energy = float(displacement @ (K @ displacement))
    assert matrix_energy == pytest.approx(direct, rel=2.0e-13, abs=2.0e-12)


def test_shell_initial_stress_matrix_is_objective_under_coordinate_rotation() -> None:
    base, base_element = _single_shell(4)
    rotation = _axis_angle(np.array([0.2, 0.7, -0.4]), 0.61)
    rotated = FEModel("rotated_initial_stress")
    rotated.add_material("steel", 210.0e9, 0.3)
    for node_id, node in base.mesh.nodes.items():
        xyz = rotation @ node.coords() + np.array([0.8, -0.2, 1.1])
        rotated.add_node(node_id, *xyz)
    rotated.add_element(
        1,
        create_shell_element(
            1, list(base_element.node_ids), "steel", thickness=0.01
        ),
    )
    state = {1: {"membrane_compression": [91.0, 37.0, -12.0]}}

    K_base, _ = assemble_geometric_stiffness_matrix(base, state)
    K_rotated, _ = assemble_geometric_stiffness_matrix(rotated, state)
    transform = sparse.lil_matrix(K_base.shape, dtype=float)
    for node in base.mesh.nodes.values():
        transform[np.ix_(node.dofs[:3], node.dofs[:3])] = rotation
        transform[np.ix_(node.dofs[3:], node.dofs[3:])] = rotation
    transform = transform.tocsr()
    expected = transform @ K_base @ transform.T
    error = sparse.linalg.norm(K_rotated - expected) / max(sparse.linalg.norm(K_base), 1.0)
    assert error < 2.0e-12
