"""Qualification for orthotropic/generalized S4 production batches."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from anyfesolver import (
    FEModel,
    GeneralizedShellSection,
    Material,
    OrthotropicMaterial,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    create_element,
)
from anyfesolver.nonlinear_performance_bootstrap import (
    clear_nonlinear_assembly_cache,
    get_nonlinear_assembly_plan,
    install_nonlinear_performance_optimizations,
)
from anyfesolver.nonlinear_performance_batch_c import (
    assemble_reduced_system,
    build_reduced_assembly_plan,
)


_QUADS = (
    np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    ),
    np.array(
        [[0.1, 0.0, 0.2], [1.0, 0.2, 0.6], [0.9, 1.1, 0.9], [0.0, 0.8, 0.5]]
    ),
    np.array(
        [[0.0, 0.0, 0.0], [1.3, 0.1, 0.0], [1.0, 1.2, 0.08], [-0.2, 0.9, -0.03]]
    ),
    np.array(
        [[0.0, 0.0, 0.1], [0.9, -0.1, -0.05], [1.2, 0.8, 0.12], [0.1, 1.1, -0.08]]
    ),
)


def _orthotropic() -> OrthotropicMaterial:
    return OrthotropicMaterial(
        name="ortho",
        elastic_modulus_1=145.0e9,
        elastic_modulus_2=11.0e9,
        elastic_modulus_3=8.5e9,
        poisson_ratio_12=0.24,
        poisson_ratio_13=0.19,
        poisson_ratio_23=0.28,
        shear_modulus_12=5.2e9,
        shear_modulus_13=4.1e9,
        shear_modulus_23=3.2e9,
        density=1580.0,
    )


def _generalized_section(*, with_mass: bool = False) -> GeneralizedShellSection:
    # B is intentionally nonsymmetric.  Its transpose is the lower coupling
    # block in the exact section operator.
    return GeneralizedShellSection(
        A=np.array(
            [[1.20e8, 1.8e7, 2.0e6], [1.8e7, 8.5e7, -1.5e6], [2.0e6, -1.5e6, 3.4e7]]
        ),
        B=np.array(
            [[2.0e3, -0.9e3, 0.3e3], [0.2e3, -1.2e3, 0.5e3], [-0.1e3, 0.4e3, 0.8e3]]
        ),
        D=np.array(
            [[1.4e4, 1.1e3, 0.3e3], [1.1e3, 1.0e4, -0.2e3], [0.3e3, -0.2e3, 4.2e3]]
        ),
        As=np.array([[2.8e7, 1.7e6], [1.7e6, 2.1e7]]),
        mass_per_area=17.5 if with_mass else None,
        rotary_inertia_per_area=0.062 if with_mass else None,
    )


def _shell_model(kind: str, *, with_mass: bool = False) -> FEModel:
    model = FEModel(f"{kind}_s4_batch")
    if kind == "orthotropic":
        material = _orthotropic()
        section = None
    else:
        material = Material("carrier", 70.0e9, 0.3, density=2700.0)
        section = _generalized_section(with_mass=with_mass)
    model.register_material(material)
    next_node = 1
    for element_index, quad in enumerate(_QUADS, start=1):
        node_ids = []
        offset = np.array([3.0 * (element_index - 1), 0.0, 0.0])
        for point in quad + offset:
            model.add_node(next_node, *point)
            node_ids.append(next_node)
            next_node += 1
        model.add_element(
            element_index,
            create_element(
                "shell",
                element_index,
                node_ids,
                material.name,
                thickness=0.018 + 0.001 * element_index,
                material_direction=np.array([1.0, 0.35, 0.22]),
                material_angle_deg=15.0 * element_index,
                shell_section=section,
                reference_normal=(
                    np.array([0.0, 0.0, 1.0])
                    if section is not None
                    else None
                ),
            ),
        )
    return model


def _scalar_global_matrix(model: FEModel, matrix_type: str) -> np.ndarray:
    result = np.zeros(
        (model.mesh.dof_manager.total_dofs, model.mesh.dof_manager.total_dofs),
        dtype=float,
    )
    for element in model.mesh.elements.values():
        material = model.get_material(element.material_name)
        local = (
            element.compute_stiffness_matrix(model.mesh, material)
            if matrix_type == "stiffness"
            else element.compute_mass_matrix(model.mesh, material)
        )
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        result[np.ix_(dofs, dofs)] += local
    return result


def _scalar_nonlinear(model: FEModel, displacement: np.ndarray):
    force = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    tangent = np.zeros((force.size, force.size), dtype=float)
    states = {}
    for element_id, element in model.mesh.elements.items():
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        local_force, local_tangent, state = element.compute_nonlinear_response(
            model.mesh,
            model.get_material(element.material_name),
            displacement[dofs],
            tangent=True,
        )
        force[dofs] += local_force
        tangent[np.ix_(dofs, dofs)] += local_tangent
        states[int(element_id)] = state
    return force, tangent, states


def test_linear_orthotropic_s4_batch_matches_scalar_on_geometry_matrix() -> None:
    model = _shell_model("orthotropic")
    expected = _scalar_global_matrix(model, "stiffness")
    actual, info = assemble_stiffness_matrix(model)

    np.testing.assert_allclose(actual.toarray(), expected, rtol=3.0e-12, atol=2.0e-5)
    assert info["diagnostics"]["qualified_e4_pl_stiffness"] == {
        "path": "shared_geometry_cache",
        "element_count": 4,
        "unique_geometry_count": 4,
    }


def test_linear_generalized_s4_batch_preserves_nonsymmetric_B() -> None:
    model = _shell_model("generalized")
    expected = _scalar_global_matrix(model, "stiffness")
    actual, info = assemble_stiffness_matrix(model)

    np.testing.assert_allclose(actual.toarray(), expected, rtol=3.0e-12, atol=2.0e-8)
    np.testing.assert_allclose(actual.toarray(), actual.toarray().T, rtol=0.0, atol=2.0e-8)
    assert info["diagnostics"]["qualified_e4_pl_stiffness"] == {
        "path": "shared_geometry_cache",
        "element_count": 4,
        "unique_geometry_count": 4,
    }


def test_generalized_section_mass_batch_matches_scalar() -> None:
    model = _shell_model("generalized", with_mass=True)
    expected = _scalar_global_matrix(model, "mass")
    actual, info = assemble_mass_matrix(model)

    np.testing.assert_allclose(actual.toarray(), expected, rtol=3.0e-12, atol=2.0e-13)
    assert info["diagnostics"]["generalized_s4_section_mass"] == {
        "path": "compiled_batch",
        "element_count": 4,
    }


def test_nonlinear_orthotropic_s4_batch_matches_scalar_force_and_tangent() -> None:
    model = _shell_model("orthotropic")
    rng = np.random.default_rng(4815)
    displacement = rng.normal(scale=8.0e-4, size=model.mesh.dof_manager.total_dofs)
    expected_force, expected_tangent, _states = _scalar_nonlinear(
        model, displacement
    )

    install_nonlinear_performance_optimizations()
    clear_nonlinear_assembly_cache(model)
    plan = get_nonlinear_assembly_plan(model, 5)
    force, tangent, _states = plan.assemble(displacement, {}, tangent=True)

    np.testing.assert_allclose(force, expected_force, rtol=3.0e-12, atol=2.0e-5)
    np.testing.assert_allclose(
        tangent.toarray(), expected_tangent, rtol=3.0e-12, atol=3.0e-5
    )
    assert plan.diagnostics()["orthotropic_elastic_fast_path_element_count"] == 4


def test_nonlinear_generalized_s4_batch_matches_scalar_and_state() -> None:
    model = _shell_model("generalized")
    rng = np.random.default_rng(9821)
    displacement = rng.normal(scale=1.5e-3, size=model.mesh.dof_manager.total_dofs)
    expected_force, expected_tangent, expected_states = _scalar_nonlinear(
        model, displacement
    )

    install_nonlinear_performance_optimizations()
    clear_nonlinear_assembly_cache(model)
    plan = get_nonlinear_assembly_plan(model, 5)
    force, tangent, states = plan.assemble(displacement, {}, tangent=True)

    np.testing.assert_allclose(force, expected_force, rtol=3.0e-12, atol=2.0e-8)
    np.testing.assert_allclose(
        tangent.toarray(), expected_tangent, rtol=3.0e-12, atol=3.0e-8
    )
    for element_id in expected_states:
        for key in (
            "membrane_strain",
            "curvature",
            "transverse_shear_strain",
            "membrane_resultants",
            "bending_resultants",
            "transverse_shear_resultants",
        ):
            np.testing.assert_allclose(
                states[element_id][key],
                expected_states[element_id][key],
                rtol=3.0e-13,
                atol=3.0e-10,
            )
        assert states[element_id]["recovery_scope"] == "section_resultants_only"
    assert plan.diagnostics()["generalized_elastic_fast_path_element_count"] == 4


def test_generalized_s4_batch_supports_direct_weighted_reduced_scatter() -> None:
    model = _shell_model("generalized")
    rng = np.random.default_rng(2884)
    displacement = rng.normal(scale=8.0e-4, size=model.mesh.dof_manager.total_dofs)
    install_nonlinear_performance_optimizations()
    clear_nonlinear_assembly_cache(model)
    plan = get_nonlinear_assembly_plan(model, 5)
    full_force, full_tangent, full_states = plan.assemble(
        displacement, {}, tangent=True
    )

    full_size = displacement.size
    transformation = sparse.eye(full_size, full_size - 1, format="lil")
    transformation[full_size - 1, 0] = 0.35
    transformation[full_size - 1, 1] = -0.2
    transformation = transformation.tocsr()
    reduced_plan = build_reduced_assembly_plan(plan, transformation)
    reduced_force, reduced_tangent, reduced_states = assemble_reduced_system(
        plan,
        reduced_plan,
        displacement,
        {},
        tangent=True,
    )

    np.testing.assert_allclose(
        reduced_force,
        np.asarray(transformation.T @ full_force).reshape(-1),
        rtol=3.0e-13,
        atol=3.0e-10,
    )
    np.testing.assert_allclose(
        reduced_tangent.toarray(),
        (transformation.T @ full_tangent @ transformation).toarray(),
        rtol=3.0e-13,
        atol=3.0e-9,
    )
    assert reduced_plan.mapping_kind == "weighted_mpc"
    for element_id in full_states:
        np.testing.assert_allclose(
            reduced_states[element_id]["membrane_resultants"],
            full_states[element_id]["membrane_resultants"],
        )


def test_unsupported_generalized_triangle_reports_scalar_fallback() -> None:
    model = FEModel("generalized_s3_fallback")
    material = Material("carrier", 70.0e9, 0.3, density=2700.0)
    model.register_material(material)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 0.1, 0.9, 0.0)
    element = create_element(
        "shell",
        1,
        [1, 2, 3],
        material.name,
        shell_section=_generalized_section(with_mass=True),
    )
    model.add_element(1, element)

    expected_stiffness = _scalar_global_matrix(model, "stiffness")
    stiffness, stiffness_info = assemble_stiffness_matrix(model)
    np.testing.assert_allclose(stiffness.toarray(), expected_stiffness)
    assert stiffness_info["diagnostics"][
        "generalized_shell_section_fallback"
    ] == {
        "path": "general_element",
        "reason": "preintegrated_generalized_shell_section",
        "element_ids": [1],
    }

    expected_mass = _scalar_global_matrix(model, "mass")
    mass, mass_info = assemble_mass_matrix(model)
    np.testing.assert_allclose(mass.toarray(), expected_mass)
    assert mass_info["diagnostics"][
        "generalized_shell_section_mass_fallback"
    ] == {
        "path": "general_element",
        "reason": "unsupported_shell_topology",
        "element_ids": [1],
    }

    install_nonlinear_performance_optimizations()
    clear_nonlinear_assembly_cache(model)
    plan = get_nonlinear_assembly_plan(model, 5)
    assert plan.diagnostics()["shell_element_count"] == 0
    assert plan.diagnostics()["constitutive_fallback"] == {
        "path": "general_element",
        "reason": "generalized_shell_section",
        "element_ids": [1],
    }
