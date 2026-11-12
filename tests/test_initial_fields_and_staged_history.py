"""Qualification of residual fields and committed staged material history."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.elements import BeamElement, create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.imperfections import ImperfectionField
from anyfesolver.material_curves import DNVC208MaterialCurve, FiberSectionPlasticityConfig
from anyfesolver.nonlinear_static import (
    BeamInitialField,
    DisplacementControl,
    NonlinearLoadProgram,
    NonlinearLoadStage,
    ShellInitialField,
    solve_static_nonlinear,
)


E = 210.0e9
NU = 0.3
S355_CURVE = DNVC208MaterialCurve(
    sigma_prop=320.0e6,
    sigma_yield=357.0e6,
    sigma_yield_2=363.3e6,
    eps_p_y1=0.004,
    eps_p_y2=0.015,
    K=740.0e6,
    n=0.166,
)


def _membrane_patch() -> FEModel:
    model = FEModel(name="initial_shell_field")
    model.add_material("steel", E, NU)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.0, 1.0, 0.0)
    model.add_node(4, 0.0, 1.0, 0.0)
    model.add_element(
        1, create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.01)
    )
    model.add_boundary_condition(BoundaryCondition("left_x", [1, 4], {"ux": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pin_y", [1], {"uy": 0.0}))
    model.add_boundary_condition(
        BoundaryCondition(
            "plane",
            [1, 2, 3, 4],
            {"uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def _fiber_beam(curve: DNVC208MaterialCurve = S355_CURVE) -> FEModel:
    model = FEModel(name="initial_beam_field")
    model.add_material("steel", E, NU, hardening_curve=curve)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    section = {
        "area": 0.01,
        "Iy": 1.0e-5,
        "Iz": 1.0e-5,
        "J": 1.0e-5,
        "fiber_plasticity": FiberSectionPlasticityConfig(5, 5),
    }
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "guide",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def test_shell_initial_stress_is_equilibrated_before_external_loading():
    model = _membrane_patch()
    sigma_x = 105.0e6

    result = solve_static_nonlinear(
        model,
        initial_fields={
            1: ShellInitialField(
                membrane_stress=[sigma_x, 0.0, 0.0],
                source="weld-residual-map-v1",
            )
        },
        num_steps=1,
        num_layers=3,
        tolerance=1.0e-9,
    )

    assert result.status == "completed"
    assert result.info["initial_state_equilibration"]["converged"]
    assert result.info["initial_state_equilibration"]["iterations"] >= 2
    right_ux = np.mean(
        [
            result.displacements[model.mesh.get_node(node_id).dofs[0]]
            for node_id in (2, 3)
        ]
    )
    assert right_ux == pytest.approx(-sigma_x / E, rel=2.0e-5)
    state = result.element_states[1]
    np.testing.assert_allclose(state["initial_membrane_stress"], [sigma_x, 0.0, 0.0])
    assert np.max(np.abs(state["layer_stress"])) < 1.0
    provenance = result.info["initial_condition_provenance"]
    assert provenance["geometric_imperfection"] == []
    assert provenance["residual_stress_or_prestrain"][0]["source"] == "weld-residual-map-v1"
    assert provenance["coordinate_system"]["residual_stress_or_prestrain"] == (
        "element-local reference coordinates"
    )


def test_failed_initial_equilibration_returns_last_committed_pair():
    model = _membrane_patch()
    result = solve_static_nonlinear(
        model,
        initial_fields={
            1: ShellInitialField(membrane_stress=[100.0e6, 0.0, 0.0])
        },
        num_steps=1,
        num_layers=3,
        max_iterations=1,
        tolerance=1.0e-12,
    )

    assert result.status == "diverged"
    assert result.info["failure_reason"] == "maximum_initial_state_iterations_reached"
    np.testing.assert_allclose(result.displacements, 0.0)
    assert "layer_stress" not in result.element_states[1]
    np.testing.assert_allclose(
        result.element_states[1]["initial_membrane_stress"],
        [100.0e6, 0.0, 0.0],
    )


def test_shell_bending_stress_and_prestrain_use_documented_layer_convention():
    model = _membrane_patch()
    element = model.mesh.get_element(1)
    material = model.get_material("steel")
    from anyfesolver.nonlinear_static import _prepare_initial_states

    states, _ = _prepare_initial_states(
        model,
        None,
        {
            1: ShellInitialField(
                bending_stress=[42.0e6, 0.0, 0.0],
                membrane_prestrain=[1.0e-5, 0.0, 0.0],
                curvature_prestrain=[2.0e-3, 0.0, 0.0],
            )
        },
        3,
    )
    _force, _tangent, trial = element.compute_nonlinear_response(
        model.mesh,
        material,
        np.zeros(element.total_dofs),
        states[1],
        num_layers=3,
        tangent=True,
    )
    stress = trial["layer_stress"].reshape(len(element.gauss_points), 3, 3)
    # sigma(z)=sigma_membrane+(2z/t)*sigma_bending_surface, combined with
    # epsilon(z)=epsilon_membrane+z*kappa as an independently recoverable
    # eigenstrain contribution.
    assert np.all(stress[:, 0, 0] < stress[:, 2, 0])
    assert trial["kinematic_layer_strain"].shape == trial["layer_strain"].shape
    np.testing.assert_allclose(trial["kinematic_layer_strain"], 0.0)


def test_self_equilibrated_beam_fiber_stress_persists_and_has_separate_provenance():
    model = _fiber_beam()
    grid = np.linspace(-1.0, 1.0, 5)
    yy, zz = np.meshgrid(grid, grid, indexing="ij")
    fiber_stress = 100.0e6 * (yy * zz).reshape(-1)
    imperfection = ImperfectionField(
        {2: (0.0, 0.001, 0.0)},
        metadata={"source": "survey"},
    )

    result = solve_static_nonlinear(
        model,
        initial_fields={
            1: BeamInitialField(
                fiber_stress=fiber_stress,
                source="weld-fiber-map-v2",
            )
        },
        imperfection=imperfection,
        num_steps=1,
        tolerance=1.0e-9,
    )

    assert result.status == "completed"
    np.testing.assert_allclose(
        result.element_states[1]["fiber_stress"],
        fiber_stress,
        rtol=1.0e-12,
        atol=1.0e-4,
    )
    provenance = result.info["initial_condition_provenance"]
    assert provenance["geometric_imperfection"]
    assert provenance["residual_stress_or_prestrain"][0]["source"] == "weld-fiber-map-v2"
    assert provenance["coordinate_system"]["geometric_imperfection"] == "global nodal coordinates"


def test_beam_scalar_alpha_restart_broadcasts_to_all_fibers():
    model = _fiber_beam()
    element = model.mesh.get_element(1)
    material = model.get_material("steel")
    fiber_count = 25
    state = {
        "plastic_strain": np.zeros(fiber_count, dtype=float),
        "alpha": np.zeros(1, dtype=float),
        "initial_fiber_stress": np.zeros(fiber_count, dtype=float),
        "initial_field_provenance": {
            "kind": "beam",
            "source": "scalar-alpha-restart",
            "components": ["initial_fiber_stress"],
        },
    }
    displacement = np.zeros(element.total_dofs, dtype=float)
    displacement[6] = 0.003

    _force, _tangent, trial = element.compute_nonlinear_response(
        model.mesh,
        material,
        displacement,
        state,
        tangent=True,
    )

    assert trial["alpha"].shape == (fiber_count,)
    assert trial["plastic_strain"].shape == (fiber_count,)
    assert np.max(trial["alpha"]) > 0.0


def test_initial_stress_outside_flow_surface_is_rejected():
    model = _fiber_beam()
    with pytest.raises(ValueError, match="outside the supplied hardening-state flow surface"):
        solve_static_nonlinear(
            model,
            initial_fields={1: BeamInitialField(fiber_stress=400.0e6)},
            num_steps=1,
        )


def test_beam_section_curve_cannot_bypass_initial_stress_admissibility():
    model = FEModel(name="section_curve_initial_beam_field")
    model.add_material("steel", E, NU)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    section = {
        "area": 0.01,
        "Iy": 1.0e-5,
        "Iz": 1.0e-5,
        "J": 1.0e-5,
        "fiber_plasticity": FiberSectionPlasticityConfig(
            5,
            5,
            material_curve=S355_CURVE,
        ),
    }
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))

    with pytest.raises(ValueError, match="outside the supplied hardening-state flow surface"):
        solve_static_nonlinear(
            model,
            initial_fields={1: BeamInitialField(fiber_stress=400.0e6)},
            num_steps=1,
        )


def test_fully_constrained_model_retains_prepared_initial_field_state():
    model = _membrane_patch()
    model.add_boundary_condition(FixedSupport("fully_fixed", [1, 2, 3, 4]))

    result = solve_static_nonlinear(
        model,
        initial_fields={
            1: ShellInitialField(
                membrane_stress=[50.0e6, 0.0, 0.0],
                source="fully-constrained-map",
            )
        },
        num_steps=1,
        num_layers=3,
    )

    assert result.status == "empty_reduced_system"
    assert 1 in result.element_states
    np.testing.assert_allclose(
        result.element_states[1]["initial_membrane_stress"],
        [50.0e6, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        result.element_states[1]["layer_stress"][:, 0],
        50.0e6,
        rtol=1.0e-12,
    )
    assert result.info["initial_state_equilibration"]["fully_constrained"]
    assert (
        result.info["initial_condition_provenance"]["residual_stress_or_prestrain"][0][
            "source"
        ]
        == "fully-constrained-map"
    )


def test_fully_constrained_model_validates_initial_displacements():
    model = _membrane_patch()
    model.add_boundary_condition(FixedSupport("fully_fixed", [1, 2, 3, 4]))
    size = model.mesh.dof_manager.total_dofs

    with pytest.raises(ValueError, match="initial_displacements has"):
        solve_static_nonlinear(
            model,
            initial_displacements=np.zeros(size - 1),
            num_steps=1,
        )
    nonfinite = np.zeros(size, dtype=float)
    nonfinite[0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        solve_static_nonlinear(
            model,
            initial_displacements=nonfinite,
            num_steps=1,
        )
    incompatible = np.zeros(size, dtype=float)
    incompatible[model.mesh.get_node(2).dofs[0]] = 1.0e-3
    with pytest.raises(ValueError, match="fully constrained state"):
        solve_static_nonlinear(
            model,
            initial_displacements=incompatible,
            num_steps=1,
        )


def test_fully_constrained_force_restart_retains_nonunit_affine_state():
    def fully_constrained_patch(prescribed_ux: float) -> FEModel:
        model = _membrane_patch()
        model.add_boundary_condition(
            BoundaryCondition("right_x", [2, 3], {"ux": prescribed_ux})
        )
        model.add_boundary_condition(
            BoundaryCondition("remaining_y", [2, 3, 4], {"uy": 0.0})
        )
        return model

    restart_target = 2.0e-3
    restart_scale = 0.5
    model = fully_constrained_patch(restart_target)
    initial_displacements = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node_id in (2, 3):
        initial_displacements[model.mesh.get_node(node_id).dofs[0]] = (
            restart_scale * restart_target
        )

    result = solve_static_nonlinear(
        model,
        initial_fields={
            1: ShellInitialField(membrane_prestrain=[0.0, 0.0, 0.0])
        },
        initial_displacements=initial_displacements,
        equilibrate_initial_state=False,
        num_steps=1,
        num_layers=3,
    )

    reference = fully_constrained_patch(restart_scale * restart_target)
    reference_displacements = np.zeros(
        reference.mesh.dof_manager.total_dofs,
        dtype=float,
    )
    for node_id in (2, 3):
        reference_displacements[reference.mesh.get_node(node_id).dofs[0]] = (
            restart_scale * restart_target
        )
    reference_result = solve_static_nonlinear(
        reference,
        initial_fields={
            1: ShellInitialField(membrane_prestrain=[0.0, 0.0, 0.0])
        },
        initial_displacements=reference_displacements,
        equilibrate_initial_state=False,
        num_steps=1,
        num_layers=3,
    )

    assert result.status == "empty_reduced_system"
    np.testing.assert_allclose(result.displacements, initial_displacements)
    np.testing.assert_allclose(
        result.element_states[1]["layer_strain"],
        reference_result.element_states[1]["layer_strain"],
    )
    assert result.info["initial_state_equilibration"][
        "constrained_internal_force_norm"
    ] == pytest.approx(
        reference_result.info["initial_state_equilibration"][
            "constrained_internal_force_norm"
        ]
    )


def test_fully_constrained_prescribed_model_without_restart_keeps_target_state():
    model = _membrane_patch()
    target = 2.0e-3
    model.add_boundary_condition(
        BoundaryCondition("right_x", [2, 3], {"ux": target})
    )
    model.add_boundary_condition(
        BoundaryCondition("remaining_y", [2, 3, 4], {"uy": 0.0})
    )

    result = solve_static_nonlinear(model, num_steps=1)

    assert result.status == "empty_reduced_system"
    for node_id in (2, 3):
        assert result.displacements[model.mesh.get_node(node_id).dofs[0]] == pytest.approx(
            target
        )


def test_field_bearing_restart_requires_matching_displacements():
    model = _membrane_patch()
    from anyfesolver.nonlinear_static import _prepare_initial_states

    states, _ = _prepare_initial_states(
        model,
        None,
        {1: ShellInitialField(membrane_stress=[50.0e6, 0.0, 0.0])},
        3,
    )

    with pytest.raises(
        ValueError,
        match="require matching initial_displacements",
    ):
        solve_static_nonlinear(
            model,
            initial_element_states=states,
            num_steps=1,
            num_layers=3,
        )


def test_initial_displacements_reject_nonfinite_values():
    model = _membrane_patch()
    size = model.mesh.dof_manager.total_dofs
    displacements = np.zeros(size, dtype=float)
    displacements[model.mesh.get_node(2).dofs[0]] = np.nan

    with pytest.raises(ValueError, match="must contain only finite values"):
        solve_static_nonlinear(
            model,
            initial_displacements=displacements,
            num_steps=1,
        )


def test_replacing_initial_field_clears_old_components_and_rejects_alpha_history():
    model = _membrane_patch()
    from anyfesolver.nonlinear_static import _prepare_initial_states

    states, _ = _prepare_initial_states(
        model,
        None,
        {
            1: ShellInitialField(
                membrane_stress=[50.0e6, 0.0, 0.0],
                source="old-source",
            )
        },
        3,
    )
    replaced, provenance = _prepare_initial_states(
        model,
        states,
        {
            1: ShellInitialField(
                membrane_prestrain=[1.0e-5, 0.0, 0.0],
                source="new-source",
            )
        },
        3,
    )
    assert "initial_membrane_stress" not in replaced[1]
    assert "initial_membrane_prestrain" in replaced[1]
    assert provenance == [
        {
            "element_id": 1,
            "kind": "shell",
            "source": "new-source",
            "components": {
                "membrane_prestrain": {
                    "shape": [3],
                    "minimum": 0.0,
                    "maximum": 1.0e-5,
                }
            },
        }
    ]

    states[1]["plastic_strain"][...] = 0.0
    states[1]["alpha"][...] = 1.0e-3
    with pytest.raises(ValueError, match="nonzero plastic history"):
        _prepare_initial_states(
            model,
            states,
            {1: ShellInitialField(membrane_prestrain=[1.0e-5, 0.0, 0.0])},
            3,
        )


def test_new_initial_field_and_tangent_qualification_symbols_are_root_exports():
    import anyfesolver

    for name in (
        "BeamInitialField",
        "ShellInitialField",
        "algorithmic_tangent_path_metrics",
        "algorithmic_tangent_performance_metrics",
        "global_newton_tangent_benchmark_metrics",
    ):
        assert name in anyfesolver.__all__
        assert getattr(anyfesolver, name) is not None


def test_shell_initial_field_survives_plastic_commit_and_unloading():
    model = _membrane_patch()
    model.materials["steel"].hardening_curve = S355_CURVE
    element = model.mesh.get_element(1)
    material = model.get_material("steel")
    from anyfesolver.nonlinear_static import _prepare_initial_states

    states, _ = _prepare_initial_states(
        model,
        None,
        {1: ShellInitialField(membrane_stress=[100.0e6, 0.0, 0.0])},
        3,
    )
    loading = np.zeros(element.total_dofs)
    loading[6] = loading[12] = 0.002
    # Owner decision 2026-10-23: plastic qualified Q4 outside the assumed-strain
    # route scope is refused (the compatible-strain plastic route with the
    # qualified correction is unstable once layers yield).
    from anyfesolver._q4_assumed_strain_plasticity import UnsupportedQ4PlasticRouteError
    with pytest.raises(UnsupportedQ4PlasticRouteError, match="active initial"):
        element.compute_nonlinear_response(
            model.mesh, material, loading, states[1], num_layers=3, tangent=True
        )


def test_adaptive_force_control_commits_every_stage_boundary():
    model = _fiber_beam()
    stages = []
    for index, (name, factor) in enumerate(
        (("permanent-a", 0.2), ("permanent-b", 0.3), ("environmental", 0.5)),
        start=1,
    ):
        load = LoadCase(name=name)
        load.add_nodal_load(2, load_vector=[1000.0 * index, 0.0, 0.0, 0.0, 0.0, 0.0])
        stages.append(NonlinearLoadStage(name, load, target_factor=factor))

    result = solve_static_nonlinear(
        model,
        load_program=NonlinearLoadProgram(stages),
        num_steps=1,
        convergence_settings="fast",
    )

    assert result.status == "completed"
    committed = [
        entry["load_factor"]
        for entry in result.info["force_displacement_history"]
        if entry["stage_endpoint_committed"]
    ]
    np.testing.assert_allclose(committed, [0.2, 0.5, 1.0], atol=1.0e-12)


def test_load_program_rejects_ambiguous_names_and_nonfinite_factors():
    load = LoadCase(name="stage")
    with pytest.raises(ValueError, match="must not be empty"):
        NonlinearLoadStage("  ", load)
    for factor in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            NonlinearLoadStage("stage", load, target_factor=factor)
    with pytest.raises(ValueError, match="stage names must be unique"):
        NonlinearLoadProgram(
            [
                NonlinearLoadStage("permanent", load),
                NonlinearLoadStage(" permanent ", load),
            ]
        )


def test_multistage_displacement_control_reuses_preload_plastic_history():
    model = _fiber_beam()
    permanent = LoadCase(name="permanent")
    permanent.add_nodal_load(2, load_vector=[3.40e6, 0.0, 0.0, 0.0, 0.0, 0.0])
    environmental = LoadCase(name="environmental")
    environmental.add_nodal_load(2, load_vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    program = NonlinearLoadProgram(
        [
            NonlinearLoadStage("permanent", permanent),
            NonlinearLoadStage("environmental", environmental),
        ]
    )

    result = solve_static_nonlinear(
        model,
        load_program=program,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=2,
            dof="ux",
            target_displacement=0.006,
        ),
        num_steps=4,
        max_iterations=35,
        tolerance=1.0e-8,
    )

    assert result.status == "completed"
    assert result.info["stop_reason"] == "target_displacement_reached"
    assert result.info["status_category"] == "converged"
    assert result.info["material_history_reused_from_preload"] is True
    preload_alpha = result.info["load_program_preload"]["strain_summary"][
        "max_equivalent_plastic_strain"
    ]
    assert preload_alpha > 0.0
    assert result.info["strain_summary"]["max_equivalent_plastic_strain"] >= preload_alpha
    initial_control = result.info["displacement_control_initial_value"]
    assert initial_control > 0.0
    first_control = result.info["force_displacement_history"][0]["target_displacement"]
    assert first_control == pytest.approx(initial_control + (0.006 - initial_control) / 4.0)
    assert result.steps[-1].control_value == pytest.approx(0.006)
    assert result.info["load_program_stage_factors"]["permanent"] == pytest.approx(1.0)


def test_failed_displacement_control_restores_last_committed_checkpoint():
    model = _fiber_beam()
    load = LoadCase(name="unit")
    load.add_nodal_load(2, load_vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = solve_static_nonlinear(
        model,
        load_case=load,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=2,
            dof="ux",
            target_displacement=0.05,
        ),
        num_steps=1,
        max_iterations=1,
        tolerance=1.0e-12,
    )

    assert result.status == "diverged"
    assert result.steps == []
    assert result.load_factor == pytest.approx(0.0)
    np.testing.assert_allclose(result.displacements, 0.0)
    assert result.info["stop_reason"] == "maximum_iterations_reached"
    assert result.info["status_category"] == "iteration_failure"
    assert result.info["last_converged_load_factor"] == pytest.approx(0.0)
    assert result.info["strain_summary"]["max_equivalent_plastic_strain"] == pytest.approx(0.0)
