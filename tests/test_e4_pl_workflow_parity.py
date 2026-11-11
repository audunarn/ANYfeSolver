from __future__ import annotations

import numpy as np

from anyfesolver import (
    BoundaryCondition,
    ContributionPolicy,
    CoupledBeamShellElement,
    CylinderBenchmarkConfig,
    ElementActivity,
    ElementActivityPolicy,
    FEModel,
    FixedSupport,
    LoadCase,
    TransientConfig,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    build_cylindrical_shell_benchmark_model,
    generate_simple_panel_mesh,
    recover_stress_result,
    solve_eigenvalue_buckling,
    solve_linear,
    solve_transient_newmark,
)
from anyfesolver.contact import (
    RigidSphereImpact,
    SphereContactConfig,
    assemble_sphere_contact_load_vector,
)
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.nonlinear_static import ShellInitialField, solve_static_nonlinear
from anyfesolver.vectorized_nonlinear import shell_nonlinear_batch_eligible


def _candidate_model(*, constrained: bool = False) -> tuple[FEModel, QualifiedE4PLShellElement]:
    model = FEModel("q1j_workflow")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    element = QualifiedE4PLShellElement(
        1, [1, 2, 3, 4], "soft", thickness=0.05
    )
    model.add_element(1, element)
    if constrained:
        model.add_boundary_condition(FixedSupport("fixed", [1, 2, 4]))
        model.add_boundary_condition(
            BoundaryCondition(
                "single_transverse_dof",
                [3],
                {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
            )
        )
    return model, element


def _all_activity_policy() -> ElementActivityPolicy:
    return ElementActivityPolicy(
        stiffness=ContributionPolicy.ACTIVITY,
        mass=ContributionPolicy.ACTIVITY,
        damping=ContributionPolicy.ACTIVITY,
        load=ContributionPolicy.ACTIVITY,
        contact=ContributionPolicy.ACTIVITY,
    )


def test_global_assembly_uses_candidate_scalar_kernel_and_activity_lifecycle() -> None:
    model, element = _candidate_model()
    material = model.get_material("soft")
    direct = element.compute_stiffness_matrix(model.mesh, material).copy()
    assembled, info = assemble_stiffness_matrix(model)
    np.testing.assert_allclose(assembled.toarray(), direct, rtol=1.0e-12, atol=1.0e-10)
    assert shell_nonlinear_batch_eligible(element) is True
    assert element.legacy_stiffness_batch_eligible is False

    baseline_mass, _ = assemble_mass_matrix(model)
    activity = ElementActivity([1], policy=_all_activity_policy())
    model.set_element_activity(activity)
    activity.set_activity([1], 0.25, reason="q1j-parity")
    scaled_stiffness, scaled_info = assemble_stiffness_matrix(model)
    scaled_mass, _ = assemble_mass_matrix(model)
    np.testing.assert_allclose(scaled_stiffness.toarray(), 0.25 * direct)
    np.testing.assert_allclose(scaled_mass.toarray(), 0.25 * baseline_mass.toarray())
    assert scaled_info["diagnostics"]["element_activity"]["scaled_element_count"] == 1
    activity.hard_delete([1], reason="q1j-parity")
    deleted, _ = assemble_stiffness_matrix(model)
    assert deleted.nnz == 0
    assert tuple(model.mesh.elements) == (1,)


def test_structured_mesh_cold_assembly_reuses_translation_equivalent_geometry(
) -> None:
    model = generate_simple_panel_mesh(
        2.0,
        1.0,
        0.02,
        num_divisions_x=2,
        num_divisions_y=1,
    )
    for element_id, legacy in list(model.mesh.elements.items()):
        model.mesh.elements[element_id] = QualifiedE4PLShellElement(
            element_id,
            list(legacy.node_ids),
            legacy.material_name,
            thickness=legacy.thickness,
        )
    assert all(
        element._qualified_components is None
        for element in model.mesh.elements.values()
    )
    assembled, info = assemble_stiffness_matrix(model)
    assert assembled.nnz > 0
    diagnostic = info["diagnostics"]["qualified_e4_pl_stiffness"]
    assert diagnostic == {
        "path": "shared_geometry_cache",
        "element_count": 2,
        "unique_geometry_count": 1,
    }
    components = [
        element._qualified_components for element in model.mesh.elements.values()
    ]
    assert all(value is not None for value in components)
    assert components[0] is not components[1]
    assert components[0]["total"] is not components[1]["total"]
    np.testing.assert_array_equal(components[0]["total"], components[1]["total"])
    assert (
        model.mesh.elements[1]._qualified_cache_key
        == model.mesh.elements[2]._qualified_cache_key
    )
    cold_component_ids = tuple(id(value) for value in components)
    cold_total_ids = tuple(id(value["total"]) for value in components)

    warm, warm_info = assemble_stiffness_matrix(model)
    np.testing.assert_array_equal(warm.toarray(), assembled.toarray())
    assert warm_info["diagnostics"]["qualified_e4_pl_stiffness"] == diagnostic
    warm_components = tuple(
        element._qualified_components for element in model.mesh.elements.values()
    )
    assert tuple(id(value) for value in warm_components) == cold_component_ids
    assert tuple(id(value["total"]) for value in warm_components) == cold_total_ids


def test_candidate_runs_transient_and_buckling_solver_workflows() -> None:
    model, _element = _candidate_model(constrained=True)
    load = LoadCase("step")
    load.add_nodal_load(3, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    transient = solve_transient_newmark(
        model,
        TransientConfig(dt=1.0e-3, t_end=5.0e-3),
        base_load_case=load,
    )
    transverse_dof = model.mesh.get_node(3).dofs[2]
    assert transient.status == "completed"
    assert np.isfinite(transient.displacements[-1, transverse_dof])
    assert abs(transient.displacements[-1, transverse_dof]) > 0.0

    buckling = solve_eigenvalue_buckling(
        model,
        {1: {"membrane_compression_x": 1.0, "membrane_compression_y": 1.0}},
        num_modes=1,
    )
    assert buckling.solver_status == "ok"
    assert buckling.critical_load_factor is not None
    assert np.isfinite(buckling.critical_load_factor)
    assert buckling.critical_load_factor > 0.0


def test_candidate_participates_in_contact_coupling_and_recovery() -> None:
    model, element = _candidate_model()
    sphere = RigidSphereImpact(
        "unit",
        radius=0.2,
        mass=1.0,
        start_point=(0.5, 0.5, 0.1),
        travel_direction=(0.0, 0.0, -1.0),
        speed=0.0,
    )
    vector, sphere_force, records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        SphereContactConfig(penalty_stiffness=1000.0),
        sphere_position=np.asarray((0.5, 0.5, 0.1)),
        sphere_velocity=np.zeros(3),
    )
    assert len(records) == 1
    np.testing.assert_allclose(sphere_force, (0.0, 0.0, 100.0), atol=1.0e-10)
    assert np.linalg.norm(vector) > 0.0

    model.add_node(5, 0.0, 0.0, 0.2)
    coupling = CoupledBeamShellElement(
        2, beam_node_id=5, shell_node_id=1, material_name="soft"
    )
    constraints = coupling.get_mpc_constraints(model.mesh)
    assert constraints
    assert {row["slave"] for row in constraints} == set(model.mesh.get_node(5).dofs)

    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    displacement[model.mesh.get_node(2).dofs[0]] = 1.0e-4
    displacement[model.mesh.get_node(3).dofs[0]] = 1.0e-4
    recovered = recover_stress_result(model, displacement)
    assert 1 in recovered.element_stresses
    stress = recovered.element_stresses[1]
    assert stress["membrane_xx"].shape == (len(element.gauss_points),)
    assert np.all(np.isfinite(stress["equivalent_stress"]))


def test_faceted_cylindrical_shell_uses_direct_candidate_and_converges() -> None:
    config = CylinderBenchmarkConfig(
        radius=1.5,
        height=2.0,
        thickness=0.02,
        pressure=2.0e4,
        num_circumferential=8,
        num_height=2,
        use_8node_elements=False,
    )
    model, load_case = build_cylindrical_shell_benchmark_model(config)
    for element_id, legacy in list(model.mesh.elements.items()):
        model.mesh.elements[element_id] = QualifiedE4PLShellElement(
            element_id,
            list(legacy.node_ids),
            legacy.material_name,
            thickness=legacy.thickness,
        )
    displacement, info = solve_linear(model, load_case, constraint_mode="auto")
    assert info["convergence_info"]["status"] == "converged"
    assert np.all(np.isfinite(displacement))
    assert np.max(np.abs(displacement)) > 0.0
    material = model.get_material("steel")
    assert all(
        not element.compute_stiffness_components(model.mesh, material)["legacy_fallback"]
        for element in model.mesh.elements.values()
    )


def _plastic_restart_model(target: float) -> FEModel:
    model = FEModel("q1j_restart")
    model.add_material(
        "steel",
        210.0e9,
        0.3,
        hardening_curve=DNVC208MaterialCurve(
            sigma_prop=320.0e6,
            sigma_yield=357.0e6,
            sigma_yield_2=363.3e6,
            eps_p_y1=0.004,
            eps_p_y2=0.015,
            K=740.0e6,
            n=0.166,
        ),
    )
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.2, 0.0), (0.0, 0.2, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1, [1, 2, 3, 4], "steel", thickness=0.01
        ),
    )
    model.add_boundary_condition(BoundaryCondition("left_x", [1, 4], {"ux": 0.0}))
    model.add_boundary_condition(BoundaryCondition("origin_y", [1], {"uy": 0.0}))
    model.add_boundary_condition(
        BoundaryCondition(
            "plane",
            [1, 2, 3, 4],
            {"uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(
        BoundaryCondition("right_x", [2, 3], {"ux": float(target)})
    )
    return model


def test_candidate_plastic_state_survives_solver_restart() -> None:
    target = 3.0e-3
    preload = solve_static_nonlinear(
        _plastic_restart_model(target),
        num_steps=2,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-8,
    )
    assert preload.status == "completed"
    assert np.max(preload.element_states[1]["alpha"]) > 0.0

    load = LoadCase("restart_shear")
    load.add_nodal_load(3, [0.0, 1.0e3, 0.0, 0.0, 0.0, 0.0])
    restarted_model = _plastic_restart_model(target)
    restarted = solve_static_nonlinear(
        restarted_model,
        load,
        max_load_factor=0.1,
        num_steps=2,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-8,
        initial_element_states=preload.element_states,
        initial_displacements=preload.displacements,
        equilibrate_initial_state=False,
    )
    assert restarted.status == "completed"
    assert restarted.info["prescribed_displacement_path"]["mode"] == "restart_fixed_affine_state"
    assert np.max(restarted.element_states[1]["alpha"]) >= np.max(
        preload.element_states[1]["alpha"]
    )
    for node_id in (2, 3):
        dof = restarted_model.mesh.get_node(node_id).dofs[0]
        assert np.isclose(restarted.displacements[dof], target, rtol=0.0, atol=1.0e-15)


def test_candidate_initial_stress_equilibrates_on_qualified_baseline() -> None:
    model, _element = _candidate_model()
    model.add_boundary_condition(BoundaryCondition("left_x", [1, 4], {"ux": 0.0}))
    model.add_boundary_condition(BoundaryCondition("origin_y", [1], {"uy": 0.0}))
    model.add_boundary_condition(
        BoundaryCondition(
            "plane",
            [1, 2, 3, 4],
            {"uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    stress = 50.0e3
    result = solve_static_nonlinear(
        model,
        initial_fields={
            1: ShellInitialField(
                membrane_stress=[stress, 0.0, 0.0],
                source="q1j-initial-field-parity",
            )
        },
        num_steps=1,
        num_layers=3,
        tolerance=1.0e-9,
    )
    assert result.status == "completed"
    expected = -stress / model.get_material("soft").elastic_modulus
    right = np.mean(
        [
            result.displacements[model.mesh.get_node(node_id).dofs[0]]
            for node_id in (2, 3)
        ]
    )
    assert np.isclose(right, expected, rtol=2.0e-5, atol=1.0e-12)
    assert np.max(np.abs(result.element_states[1]["layer_stress"])) < 1.0e-7
