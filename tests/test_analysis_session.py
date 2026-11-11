from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import (
    AnalysisSession,
    BoundaryCondition,
    FixedSupport,
    LoadCase,
    solve_eigenvalue_buckling,
    solve_free_vibration,
    solve_linear,
    solve_linear_many,
)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.mesh_gen import generate_beam_mesh


def _loads() -> tuple[LoadCase, LoadCase]:
    axial = LoadCase("axial")
    axial.add_nodal_load(3, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    bending = LoadCase("bending")
    bending.add_nodal_load(3, [0.0, 0.0, -100.0, 0.0, 0.0, 0.0])
    return axial, bending


def _modal_bar() -> FEModel:
    model = FEModel("session_modal_bar")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(
        1,
        BeamElement(
            1,
            [1, 2],
            "steel",
            {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
        ),
    )
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "slider",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def _buckling_column(num_elements: int = 6) -> tuple[FEModel, dict[int, dict[str, float]]]:
    model = FEModel("session_column")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for index in range(num_elements + 1):
        model.add_node(index + 1, 4.0 * index / num_elements, 0.0, 0.0)
    for index in range(num_elements):
        model.add_element(
            index + 1,
            BeamElement(index + 1, [index + 1, index + 2], "steel", section),
        )
    all_nodes = list(range(1, num_elements + 2))
    model.add_boundary_condition(
        BoundaryCondition(
            "suppress",
            all_nodes,
            {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0},
        )
    )
    model.add_boundary_condition(
        BoundaryCondition("pins", [1, num_elements + 1], {"uy": 0.0})
    )
    return model, {
        element_id: {"axial_compression": 1.0}
        for element_id in model.mesh.elements
    }


def test_repeated_static_session_matches_legacy_and_reuses_plans() -> None:
    model = generate_beam_mesh(
        1.0,
        num_divisions=2,
        cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
    )
    axial, bending = _loads()
    expected_axial, _ = solve_linear(model, axial)
    expected_bending, _ = solve_linear(model, bending)

    with AnalysisSession(model) as session:
        actual_axial, first_info = solve_linear(model, axial, session=session)
        actual_bending, second_info = solve_linear(model, bending, session=session)
        many, many_info = solve_linear_many(
            model,
            [axial, bending],
            session=session,
        )
        diagnostics = session.diagnostics()

    np.testing.assert_allclose(actual_axial, expected_axial, rtol=1.0e-11, atol=1.0e-13)
    np.testing.assert_allclose(actual_bending, expected_bending, rtol=1.0e-11, atol=1.0e-13)
    np.testing.assert_allclose(many[:, 0], expected_axial, rtol=1.0e-11, atol=1.0e-13)
    np.testing.assert_allclose(many[:, 1], expected_bending, rtol=1.0e-11, atol=1.0e-13)
    assert diagnostics["counters"]["stiffness_builds"] == 1
    assert diagnostics["counters"]["constraint_builds"] == 1
    assert diagnostics["counters"]["stiffness_hits"] >= 2
    assert diagnostics["counters"]["constraint_hits"] >= 2
    assert diagnostics["factorization_cache"]["misses"] == 1
    assert diagnostics["factorization_cache"]["hits"] >= 2
    assert first_info["analysis_session"]["plan_builds"] >= 2
    assert second_info["analysis_session"]["plan_reused"] is True
    assert many_info["analysis_session"]["plan_reused"] is True


def test_session_reuses_exact_proportional_dead_load_vector() -> None:
    model = generate_beam_mesh(
        1.0,
        num_divisions=2,
        cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
    )
    first = LoadCase("first")
    first.add_nodal_load(3, [100.0, 20.0, 0.0, 0.0, 0.0, 0.0])
    second = LoadCase("second")
    second.add_nodal_load(3, [250.0, 50.0, 0.0, 0.0, 0.0, 0.0])

    expected, _ = solve_linear(model, second)
    with AnalysisSession(model) as session:
        solve_linear(model, first, session=session)
        actual, info = solve_linear(model, second, session=session)
        diagnostics = session.diagnostics()

    np.testing.assert_allclose(actual, expected, rtol=1.0e-11, atol=1.0e-13)
    assert diagnostics["counters"]["load_vector_builds"] == 1
    assert diagnostics["counters"]["load_vector_hits"] == 1
    assert info["assembly"]["load"]["prepared_load_vector_reused"] is True


def test_prescribed_value_refresh_reuses_structure_and_factorization() -> None:
    model = _modal_bar()
    load = LoadCase("zero")
    session = AnalysisSession(model)

    first, _ = solve_linear(model, load, session=session)
    first_plan = session.constraint_plan()
    fixed = model.boundary_conditions[0]
    fixed.dof_constraints["ux"] = 2.5e-4  # Direct public-object edit, no revision bump.
    second, _ = solve_linear(model, load, session=session)
    second_plan = session.constraint_plan()

    assert second_plan.T is first_plan.T
    assert second_plan.K_red is first_plan.K_red
    assert second[model.mesh.get_node(1).dofs[0]] == pytest.approx(2.5e-4)
    assert not np.array_equal(first_plan.u0, second_plan.u0)
    diagnostics = session.diagnostics()
    assert diagnostics["counters"]["constraint_value_refreshes"] == 1
    assert diagnostics["factorization_cache"]["misses"] == 1
    assert diagnostics["factorization_cache"]["hits"] == 1


def test_stale_constraint_plan_is_rejected_after_prescribed_value_refresh() -> None:
    model = _modal_bar()
    session = AnalysisSession(model)
    constrained_ux = np.asarray([model.mesh.get_node(1).dofs[0]], dtype=np.intp)
    old_constraint = session.constraint_plan()
    old_output = session.output_selection_plan(constrained_ux, old_constraint)

    fixed = model.boundary_conditions[0]
    fixed.dof_constraints["ux"] = 3.5e-4  # No revision bump: fingerprint only.

    stale_consumers = (
        lambda: session.output_selection_plan(constrained_ux, old_constraint),
        lambda: session.reduced_mass(old_constraint),
        lambda: session.rigid_body_modes(old_constraint),
        lambda: session.factorization_signature("test", old_constraint),
    )
    for consume in stale_consumers:
        with pytest.raises(ValueError, match="stale or foreign ConstraintPlan"):
            consume()

    current_constraint = session.constraint_plan()
    current_output = session.output_selection_plan(constrained_ux, current_constraint)
    reduced_zero = np.zeros(current_constraint.K_red.shape[0], dtype=float)

    assert current_constraint is not old_constraint
    assert current_output is not old_output
    assert current_output.value_key == current_constraint.value_key
    assert old_output.value_key != current_output.value_key
    np.testing.assert_allclose(current_output.reconstruct(reduced_zero), [3.5e-4])
    diagnostics = session.diagnostics()
    assert diagnostics["counters"]["constraint_value_refreshes"] == 1
    assert diagnostics["counters"]["constraint_plan_rejections"] == 4


def test_foreign_plans_are_rejected_even_when_revision_keys_match() -> None:
    model = _modal_bar()
    first = AnalysisSession(model)
    second = AnalysisSession(model)
    first_stiffness = first.stiffness_plan()
    first_constraint = first.constraint_plan(first_stiffness)
    second_stiffness = second.stiffness_plan()
    second_constraint = second.constraint_plan(second_stiffness)

    assert first_stiffness.revision_key == second_stiffness.revision_key
    assert first_constraint.structure_key == second_constraint.structure_key
    assert first_constraint.value_key == second_constraint.value_key
    with pytest.raises(ValueError, match="stale or foreign StructuralMatrixPlan"):
        first.constraint_plan(second_stiffness)
    with pytest.raises(ValueError, match="stale or foreign ConstraintPlan"):
        first.output_selection_plan(np.asarray([0]), second_constraint)


def test_revision_categories_invalidate_only_dependent_session_data() -> None:
    model = _modal_bar()
    session = AnalysisSession(model)
    stiffness_1 = session.stiffness_plan()
    mass_1 = session.mass_plan()
    constraint_1 = session.constraint_plan(stiffness_1)
    reduced_mass_1, _ = session.reduced_mass(constraint_1)

    model.bump_revision("load")
    assert session.stiffness_plan() is stiffness_1
    assert session.mass_plan() is mass_1
    assert session.constraint_plan(stiffness_1) is constraint_1

    model.add_point_mass(2, 0.5)
    mass_2 = session.mass_plan()
    assert mass_2 is not mass_1
    assert session.stiffness_plan() is stiffness_1
    reduced_mass_2, _ = session.reduced_mass(constraint_1)
    assert reduced_mass_2 is not reduced_mass_1

    model.get_material("steel").elastic_modulus *= 1.1
    model.bump_revision("material")
    stiffness_2 = session.stiffness_plan()
    constraint_2 = session.constraint_plan(stiffness_2)
    assert stiffness_2 is not stiffness_1
    assert constraint_2.K_red is not constraint_1.K_red
    reasons = session.diagnostics()["invalidation_reasons"]
    assert reasons["mass_revision"] >= 1
    assert reasons["stiffness_revision"] >= 1


def test_topology_change_invalidates_structural_and_constraint_plans() -> None:
    model = _modal_bar()
    session = AnalysisSession(model)
    stiffness_1 = session.stiffness_plan()
    constraint_1 = session.constraint_plan(stiffness_1)

    model.add_node(3, 2.0, 0.0, 0.0)
    model.add_element(
        2,
        BeamElement(
            2,
            [2, 3],
            "steel",
            {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
        ),
    )
    stiffness_2 = session.stiffness_plan()
    constraint_2 = session.constraint_plan(stiffness_2)

    assert stiffness_2 is not stiffness_1
    assert stiffness_2.matrix.shape == (18, 18)
    assert constraint_2 is not constraint_1
    assert constraint_2.T.shape[0] == 18
    assert session.diagnostics()["invalidation_reasons"]["stiffness_revision"] == 1


def test_geometry_change_invalidates_structural_and_reduced_plans() -> None:
    model = _modal_bar()
    session = AnalysisSession(model)
    stiffness_1 = session.stiffness_plan()
    constraint_1 = session.constraint_plan(stiffness_1)

    model.set_node_coordinates(2, 1.25, 0.0, 0.0)
    stiffness_2 = session.stiffness_plan()
    constraint_2 = session.constraint_plan(stiffness_2)

    assert stiffness_2 is not stiffness_1
    assert constraint_2 is not constraint_1
    assert constraint_2.K_red is not constraint_1.K_red
    assert not np.array_equal(stiffness_2.matrix.data, stiffness_1.matrix.data)
    assert session.diagnostics()["invalidation_reasons"]["stiffness_revision"] == 1


def test_old_structural_and_constraint_plans_fail_after_revision_refresh() -> None:
    model = _modal_bar()
    session = AnalysisSession(model)
    old_stiffness = session.stiffness_plan()
    old_constraint = session.constraint_plan(old_stiffness)

    model.set_node_coordinates(2, 1.25, 0.0, 0.0)

    with pytest.raises(ValueError, match="stale or foreign StructuralMatrixPlan"):
        session.constraint_plan(old_stiffness)
    with pytest.raises(ValueError, match="stale or foreign ConstraintPlan"):
        session.output_selection_plan(np.asarray([0]), old_constraint)

    current_stiffness = session.stiffness_plan()
    current_constraint = session.constraint_plan(current_stiffness)
    assert current_stiffness is not old_stiffness
    assert current_constraint is not old_constraint
    assert current_constraint.stiffness_key == current_stiffness.revision_key
    diagnostics = session.diagnostics()
    assert diagnostics["counters"]["stiffness_plan_rejections"] == 1
    assert diagnostics["counters"]["constraint_plan_rejections"] == 1


def test_support_structure_change_invalidates_constraint_and_output_plans() -> None:
    model = _modal_bar()
    session = AnalysisSession(model)
    stiffness = session.stiffness_plan()
    constraint_1 = session.constraint_plan(stiffness)
    tip_ux = np.asarray([model.mesh.get_node(2).dofs[0]], dtype=np.intp)
    output_1 = session.output_selection_plan(tip_ux, constraint_1)

    model.add_boundary_condition(BoundaryCondition("tip_x", [2], {"ux": 0.0}))
    constraint_2 = session.constraint_plan(stiffness)
    output_2 = session.output_selection_plan(tip_ux, constraint_2)

    assert session.stiffness_plan() is stiffness
    assert constraint_2 is not constraint_1
    assert constraint_2.T.shape[1] == constraint_1.T.shape[1] - 1
    assert output_2 is not output_1
    diagnostics = session.diagnostics()
    assert diagnostics["invalidation_reasons"]["constraint_structure"] == 1
    assert diagnostics["counters"]["output_plan_builds"] == 2


def test_mpc_structure_change_invalidates_constraint_and_factorization_plans() -> None:
    model = _modal_bar()
    session = AnalysisSession(model)
    load = LoadCase("unit")
    load.add_nodal_load(2, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    solve_linear(model, load, session=session)
    stiffness = session.stiffness_plan()
    constraint_1 = session.constraint_plan(stiffness)
    node_1_ux = model.mesh.get_node(1).dofs[0]
    node_2_ux = model.mesh.get_node(2).dofs[0]

    model.add_constraint_equation(
        terms=((node_2_ux, 1.0), (node_1_ux, -1.0)),
        source_id="tip-link",
        dependent_dof=node_2_ux,
    )
    model.bump_revision("mpc")
    constraint_2 = session.constraint_plan(stiffness)

    assert session.stiffness_plan() is stiffness
    assert constraint_2 is not constraint_1
    assert constraint_2.T.shape[1] == constraint_1.T.shape[1] - 1
    diagnostics = session.diagnostics()
    assert diagnostics["invalidation_reasons"]["constraint_structure"] == 1
    assert diagnostics["factorization_cache"]["entries"] == 0


def test_output_selection_is_bounded_and_close_releases_memory() -> None:
    model = _modal_bar()
    session = AnalysisSession(model, max_output_plans=2)
    constraint = session.constraint_plan()
    for dof in range(4):
        plan = session.output_selection_plan(np.asarray([dof]), constraint)
        reduced = np.zeros(constraint.K_red.shape[0], dtype=float)
        np.testing.assert_allclose(plan.reconstruct(reduced), constraint.u0[[dof]])
    diagnostics = session.diagnostics()
    assert diagnostics["output_plan_count"] == 2
    assert diagnostics["counters"]["output_plan_evictions"] == 2
    assert diagnostics["estimated_retained_bytes"] > 0

    other = _modal_bar()
    with pytest.raises(ValueError, match="different FEModel"):
        session.stiffness_plan(other)
    session.close()
    assert session.diagnostics()["estimated_retained_bytes"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        session.stiffness_plan()


def test_modal_and_buckling_session_parity() -> None:
    modal_model = _modal_bar()
    expected_modal = solve_free_vibration(modal_model, num_modes=1)
    with AnalysisSession(modal_model) as modal_session:
        actual_modal = solve_free_vibration(
            modal_model,
            num_modes=1,
            session=modal_session,
        )
    np.testing.assert_allclose(
        actual_modal.frequencies_hz,
        expected_modal.frequencies_hz,
        rtol=1.0e-12,
        atol=1.0e-14,
    )

    buckling_model, states = _buckling_column()
    expected_buckling = solve_eigenvalue_buckling(
        buckling_model,
        states,
        num_modes=2,
    )
    with AnalysisSession(buckling_model) as buckling_session:
        actual_buckling = solve_eigenvalue_buckling(
            buckling_model,
            states,
            num_modes=2,
            session=buckling_session,
        )
    assert actual_buckling.solver_status == expected_buckling.solver_status == "ok"
    np.testing.assert_allclose(
        [mode.load_factor for mode in actual_buckling.modes],
        [mode.load_factor for mode in expected_buckling.modes],
        rtol=1.0e-10,
        atol=1.0e-8,
    )
