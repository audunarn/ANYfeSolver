from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition,
    FixedSupport,
    FractureConfig,
    Hill48Yield,
    LoadCase,
    NonlinearLoadProgram,
    NonlinearLoadStage,
    solve_eigenvalue_buckling,
    solve_free_vibration,
    solve_linear,
    solve_static_nonlinear,
    solve_transient_newmark,
    create_shell_element,
)
from anyfesolver.anystructure_fem_mode import build_fe_model_from_generated_geometry
from anyfesolver.contact import (
    RigidSphereImpact,
    SphereContactConfig,
    _require_user_capacity_for_orthotropic_linear_damage,
    material_characteristic_modulus,
    solve_transient_sphere_impact,
)
from anyfesolver.dynamics import TransientConfig
from anyfesolver.elements import BeamElement, ShellElement
from anyfesolver.external_references import write_calculix_input_deck
from anyfesolver.fe_core import FEModel
from anyfesolver.fracture import ImpactDamageConfig, state_rtcl_increment
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.runtime import _apply_material_curve_to_model


ORTHOTROPIC_CONSTANTS = {
    "elastic_modulus_1": 150.0e9,
    "elastic_modulus_2": 12.0e9,
    "elastic_modulus_3": 10.0e9,
    "poisson_ratio_12": 0.25,
    "poisson_ratio_13": 0.20,
    "poisson_ratio_23": 0.30,
    "shear_modulus_12": 5.0e9,
    "shear_modulus_13": 4.0e9,
    "shear_modulus_23": 3.8e9,
}


def _isotropic_limit_hill(strength: float = 100.0e6) -> Hill48Yield:
    shear_strength = strength / math.sqrt(3.0)
    return Hill48Yield(
        strength,
        strength,
        strength,
        shear_strength,
        shear_strength,
        shear_strength,
    )


def _hill_hardening_curve() -> DNVC208MaterialCurve:
    """Small deterministic hardening range for force-controlled workflow tests."""

    return DNVC208MaterialCurve(
        sigma_prop=100.0e6,
        sigma_yield=105.0e6,
        sigma_yield_2=110.0e6,
        eps_p_y1=0.005,
        eps_p_y2=0.010,
        K=400.0e6,
        n=0.20,
    )


def _orthotropic_model(
    *,
    beam: bool = False,
    hill: bool = False,
    hill_hardening: bool = False,
    fiber_plasticity: bool = False,
) -> FEModel:
    model = FEModel("orthotropic")
    model.add_orthotropic_material(
        "lamina",
        density=1600.0,
        hill_yield=_isotropic_limit_hill() if hill else None,
        hardening_curve=_hill_hardening_curve() if hill_hardening else None,
        **ORTHOTROPIC_CONSTANTS,
    )
    if beam:
        model.add_node(1, 0.0, 0.0, 0.0)
        model.add_node(2, 1.0, 0.0, 0.0)
        section = {
            "area": 0.01,
            "Iy": 1.0e-6,
            "Iz": 1.0e-6,
            "J": 1.0e-6,
            "torsional_rigidity": 2.5e4,
        }
        if fiber_plasticity:
            section["fiber_plasticity"] = True
        model.add_element(
            1,
            BeamElement(
                1,
                [1, 2],
                "lamina",
                section,
            ),
        )
    else:
        for node_id, coords in enumerate(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            start=1,
        ):
            model.add_node(node_id, *coords)
        model.add_element(
            1,
            create_shell_element(
                1,
                [1, 2, 3, 4],
                "lamina",
                thickness=0.02,
                material_direction=(1.0, 0.0, 1.0),
                material_angle_deg=90.0,
                # Hill48 plasticity has no stable qualified Q4 route (typed
                # refusal); Hill shell physics runs on the legacy element.
                **({"formulation": "legacy"} if hill else {}),
            ),
        )
    return model


def _orthotropic_axial_beam(
    *,
    hill: bool = False,
    hill_hardening: bool = False,
    fiber_plasticity: bool = False,
) -> FEModel:
    model = _orthotropic_model(
        beam=True,
        hill=hill,
        hill_hardening=hill_hardening,
        fiber_plasticity=fiber_plasticity,
    )
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "axial_guide",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def _orthotropic_cantilever() -> FEModel:
    model = _orthotropic_model(beam=True)
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    return model


def _orthotropic_column(num_elements: int = 2) -> tuple[FEModel, float, dict[str, float]]:
    length = 2.0
    model = FEModel("orthotropic_column")
    model.add_orthotropic_material(
        "lamina",
        density=1600.0,
        **ORTHOTROPIC_CONSTANTS,
    )
    for index in range(num_elements + 1):
        model.add_node(index + 1, length * index / num_elements, 0.0, 0.0)
    section = {
        "area": 0.01,
        "Iy": 1.0e-6,
        "Iz": 1.5e-6,
        "J": 1.0e-6,
        "torsional_rigidity": 2.5e4,
    }
    for index in range(num_elements):
        model.add_element(
            index + 1,
            BeamElement(
                index + 1,
                [index + 1, index + 2],
                "lamina",
                dict(section),
            ),
        )
    all_nodes = list(range(1, num_elements + 2))
    model.add_boundary_condition(
        BoundaryCondition(
            "suppress_unrelated",
            all_nodes,
            {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0},
        )
    )
    model.add_boundary_condition(
        BoundaryCondition("pinned_ends", [1, num_elements + 1], {"uy": 0.0})
    )
    return model, length, section


def _constrain_shell_membrane(model: FEModel) -> None:
    model.add_boundary_condition(BoundaryCondition("left_x", [1, 4], {"ux": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pin_y", [1], {"uy": 0.0}))
    model.add_boundary_condition(
        BoundaryCondition(
            "in_plane",
            [1, 2, 3, 4],
            {"uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )


def _axial_load(name: str, force: float) -> LoadCase:
    load = LoadCase(name)
    load.add_nodal_load(2, [force, 0.0, 0.0, 0.0, 0.0, 0.0])
    return load


def test_generated_geometry_parses_orthotropic_hill_and_shell_orientation() -> None:
    geometry = {
        "nodes": [
            {"id": 1, "coords": [0.0, 0.0, 0.0]},
            {"id": 2, "coords": [1.0, 0.0, 0.0]},
            {"id": 3, "coords": [1.0, 1.0, 0.0]},
            {"id": 4, "coords": [0.0, 1.0, 0.0]},
        ],
        "materials": [
            {
                "name": "lamina",
                "elastic_symmetry": "orthotropic",
                "engineering_constants": {
                    "E1": ORTHOTROPIC_CONSTANTS["elastic_modulus_1"],
                    "E2": ORTHOTROPIC_CONSTANTS["elastic_modulus_2"],
                    "E3": ORTHOTROPIC_CONSTANTS["elastic_modulus_3"],
                    "nu12": ORTHOTROPIC_CONSTANTS["poisson_ratio_12"],
                    "nu13": ORTHOTROPIC_CONSTANTS["poisson_ratio_13"],
                    "nu23": ORTHOTROPIC_CONSTANTS["poisson_ratio_23"],
                    "G12": ORTHOTROPIC_CONSTANTS["shear_modulus_12"],
                    "G13": ORTHOTROPIC_CONSTANTS["shear_modulus_13"],
                    "G23": ORTHOTROPIC_CONSTANTS["shear_modulus_23"],
                },
                "density": 1600.0,
                "hill48": {
                    "X": 500.0e6,
                    "Y": 300.0e6,
                    "Z": 300.0e6,
                    "S12": 200.0e6,
                    "S13": 180.0e6,
                    "S23": 160.0e6,
                },
                "hardening": {
                    "sigma_prop": 320.0e6,
                    "sigma_yield": 357.0e6,
                    "sigma_yield_2": 363.3e6,
                    "eps_p_y1": 0.004,
                    "eps_p_y2": 0.015,
                    "K": 740.0e6,
                    "n": 0.166,
                },
            }
        ],
        "shells": [
            {
                "id": 1,
                "node_ids": [1, 2, 3, 4],
                "material": "lamina",
                "thickness": 0.02,
                "material_direction": [1.0, 1.0, 0.5],
                "material_angle_deg": 12.5,
            }
        ],
    }

    model = build_fe_model_from_generated_geometry(geometry)
    material = model.get_material("lamina")
    element = model.mesh.get_element(1)

    assert material.elastic_symmetry == "orthotropic"
    assert material.elastic_modulus_1 == pytest.approx(150.0e9)
    assert material.hill_yield.X == pytest.approx(500.0e6)
    assert material.hardening_curve.flow_stress(np.array([0.0]))[0] == pytest.approx(
        320.0e6
    )
    assert isinstance(element, ShellElement)
    np.testing.assert_allclose(element.material_direction, [1.0, 1.0, 0.5])
    assert element.material_angle_deg == pytest.approx(12.5)


def test_generated_geometry_rejects_unknown_elastic_symmetry() -> None:
    geometry = {
        "nodes": [{"id": 1, "coords": [0.0, 0.0, 0.0]}],
        "materials": [
            {
                "name": "future_law",
                "elastic_symmetry": "triclinic",
            }
        ],
    }

    with pytest.raises(NotImplementedError, match="unsupported elastic symmetry"):
        build_fe_model_from_generated_geometry(geometry)


def test_calculix_exports_engineering_constants_and_resolved_shell_orientation(tmp_path: Path) -> None:
    model = _orthotropic_model()
    case = write_calculix_input_deck(model, None, tmp_path / "orthotropic_shell.inp")
    lines = case.inp_path.read_text(encoding="utf-8").splitlines()

    assert "*ELASTIC, TYPE=ENGINEERING CONSTANTS" in lines
    elastic_index = lines.index("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
    first_constants = [float(value) for value in lines[elastic_index + 1].split(",")]
    assert first_constants == pytest.approx(
        [
            150.0e9,
            12.0e9,
            10.0e9,
            0.25,
            0.20,
            0.30,
            5.0e9,
            4.0e9,
        ]
    )
    assert float(lines[elastic_index + 2]) == pytest.approx(3.8e9)
    orientation_index = next(index for index, line in enumerate(lines) if line.startswith("*ORIENTATION, NAME="))
    orientation = np.asarray([float(value) for value in lines[orientation_index + 1].split(",")])
    np.testing.assert_allclose(orientation[:3], [0.0, 1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(orientation[3:], [-1.0, 0.0, 0.0], atol=1.0e-12)
    assert any(
        line.startswith("*SHELL SECTION") and "ORIENTATION=ORI_" in line
        for line in lines
    )


def test_calculix_rejects_orthotropic_beam_reference_mapping(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="analytical orthotropic beam validation"):
        write_calculix_input_deck(_orthotropic_model(beam=True), None, tmp_path / "orthotropic_beam.inp")


def test_rtcl_prefers_stored_physical_stress_and_uses_conventional_von_mises() -> None:
    state = {
        "alpha": np.array([0.10]),
        # Equibiaxial plane stress: eta=2/3 and conventional VM=100.
        "layer_stress": np.array([[100.0e6, 100.0e6, 0.0]]),
        # Deliberately inconsistent legacy fields prove they are not used.
        "layer_strain": np.array([[-0.01, -0.01, 0.0]]),
        "plastic_strain": np.zeros((1, 3)),
    }

    alpha, weighted = state_rtcl_increment(
        state,
        None,
        150.0e9,
        0.25,
        elastic_symmetry="orthotropic",
    )

    assert alpha == pytest.approx([0.10])
    assert weighted == pytest.approx([0.10 * math.exp(0.5)])
    assert (
        state_rtcl_increment(
            {"alpha": np.array([0.1]), "layer_strain": np.ones((1, 3)), "plastic_strain": np.zeros((1, 3))},
            None,
            150.0e9,
            0.25,
            elastic_symmetry="orthotropic",
        )
        is None
    )


def test_contact_scaling_and_linear_damage_capacity_guard_for_orthotropy() -> None:
    model = _orthotropic_model()
    material = model.get_material("lamina")
    assert material_characteristic_modulus(material, "shell") == pytest.approx(150.0e9)
    assert material_characteristic_modulus(material, "beam") == pytest.approx(150.0e9)
    scale_material = model.add_orthotropic_material(
        "scale_test",
        elastic_modulus_1=10.0e9,
        elastic_modulus_2=20.0e9,
        elastic_modulus_3=8.0e9,
        poisson_ratio_12=0.20,
        poisson_ratio_13=0.15,
        poisson_ratio_23=0.15,
        shear_modulus_12=4.0e9,
        shear_modulus_13=3.0e9,
        shear_modulus_23=2.5e9,
    )
    assert material_characteristic_modulus(scale_material, "shell") == pytest.approx(20.0e9)
    assert material_characteristic_modulus(scale_material, "beam") == pytest.approx(10.0e9)

    non_user = ImpactDamageConfig(capacity_basis="yield")
    with pytest.raises(ValueError, match="capacity_basis='user'"):
        _require_user_capacity_for_orthotropic_linear_damage(model, non_user)

    sphere = RigidSphereImpact(
        "guard",
        radius=0.1,
        mass=1.0,
        start_point=(0.5, 0.5, 0.2),
        travel_direction=(0.0, 0.0, -1.0),
        speed=1.0,
    )
    with pytest.raises(ValueError, match="capacity_basis='user'"):
        solve_transient_sphere_impact(
            model,
            TransientConfig(dt=0.001, t_end=0.002),
            sphere,
            SphereContactConfig(penalty_stiffness=1000.0),
            damage_config=non_user,
        )

    _require_user_capacity_for_orthotropic_linear_damage(
        model,
        ImpactDamageConfig(capacity_basis="user", user_capacity=500.0e6),
    )


def test_runtime_isotropic_curve_injection_preserves_orthotropic_material_data() -> None:
    model = _orthotropic_model(
        beam=True,
        hill=True,
        hill_hardening=True,
    )
    material = model.get_material("lamina")
    element = model.mesh.get_element(1)
    original_curve = material.hardening_curve

    _apply_material_curve_to_model(
        model,
        object(),
        {"E_pa": 210.0e9, "sigma_yield": 355.0e6},
    )

    assert material.hardening_curve is original_curve
    assert material.elastic_modulus_1 == pytest.approx(
        ORTHOTROPIC_CONSTANTS["elastic_modulus_1"]
    )
    assert element._fiber_plasticity is None
    assert "fiber_plasticity" not in element.cross_section


def test_orthotropic_shell_linear_static_uses_rotated_engineering_modulus() -> None:
    model = _orthotropic_model()
    _constrain_shell_membrane(model)
    applied_stress = 5.0e6
    total_force = applied_stress * 1.0 * 0.02
    load = LoadCase("x_tension")
    load.add_nodal_load(2, [0.5 * total_force, 0.0, 0.0, 0.0, 0.0, 0.0])
    load.add_nodal_load(3, [0.5 * total_force, 0.0, 0.0, 0.0, 0.0, 0.0])

    displacements, solver_info = solve_linear(model, load)

    right_ux = np.mean(
        [displacements[model.mesh.get_node(node_id).dofs[0]] for node_id in (2, 3)]
    )
    assert solver_info["convergence_info"]["status"] == "converged"
    assert right_ux == pytest.approx(
        applied_stress / ORTHOTROPIC_CONSTANTS["elastic_modulus_2"],
        rel=2.0e-10,
    )
    assert solver_info["assembly"]["stiffness"]["diagnostics"][
        "qualified_e4_pl_stiffness"
    ] == {
        "path": "shared_geometry_cache",
        "element_count": 1,
        "unique_geometry_count": 1,
    }


def test_orthotropic_beam_modal_workflow_uses_e1_and_material_density() -> None:
    model = _orthotropic_axial_beam()

    result = solve_free_vibration(model, num_modes=1)

    expected = math.sqrt(
        2.0 * ORTHOTROPIC_CONSTANTS["elastic_modulus_1"] / 1600.0
    ) / (2.0 * math.pi)
    assert result.solver_status == "ok"
    assert result.frequencies_hz[0] == pytest.approx(expected, rel=2.0e-12)
    assert result.modes[0].modal_mass == pytest.approx(1.0)


def test_orthotropic_beam_eigenvalue_buckling_uses_e1() -> None:
    model, length, section = _orthotropic_column(num_elements=2)
    states = {
        element_id: {"axial_compression": 1.0}
        for element_id in model.mesh.elements
    }

    result = solve_eigenvalue_buckling(model, states, num_modes=1)

    expected = (
        math.pi**2
        * ORTHOTROPIC_CONSTANTS["elastic_modulus_1"]
        * section["Iz"]
        / length**2
    )
    assert result.solver_status == "ok"
    assert result.critical_load_factor == pytest.approx(expected, rel=0.02)
    assert np.all(np.isfinite(result.modes[0].mode_shape))


@pytest.mark.parametrize("kinematics", ["von_karman", "corotational"])
def test_orthotropic_beam_nonlinear_static_workflows(kinematics: str) -> None:
    model = _orthotropic_cantilever()
    load = LoadCase(f"{kinematics}_tip")
    load.add_nodal_load(2, [0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
    linear, linear_info = solve_linear(model, load)

    result = solve_static_nonlinear(
        model,
        load,
        num_steps=2,
        max_iterations=20,
        tolerance=1.0e-7,
        kinematics=kinematics,
    )

    tip_dof = model.mesh.get_node(2).dofs[1]
    assert linear_info["convergence_info"]["status"] == "converged"
    assert result.status == "completed"
    assert result.info["kinematics"] == kinematics
    assert result.displacements[tip_dof] == pytest.approx(
        linear[tip_dof],
        rel=0.02,
    )
    assert np.all(np.isfinite(result.displacements))


@pytest.mark.parametrize("kinematics", ["von_karman", "corotational"])
@pytest.mark.parametrize("element_kind", ["shell", "beam"])
def test_orthotropic_hill_plasticity_commits_in_both_nonlinear_kinematics(
    kinematics: str,
    element_kind: str,
) -> None:
    if element_kind == "beam":
        model = _orthotropic_axial_beam(
            hill=True,
            hill_hardening=True,
            fiber_plasticity=True,
        )
        load = _axial_load(f"{kinematics}_plastic_beam", 1.05e6)
    else:
        model = _orthotropic_model(hill=True, hill_hardening=True)
        _constrain_shell_membrane(model)
        load = LoadCase(f"{kinematics}_plastic_shell")
        load.add_nodal_load(2, [1.05e6, 0.0, 0.0, 0.0, 0.0, 0.0])
        load.add_nodal_load(3, [1.05e6, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = solve_static_nonlinear(
        model,
        load,
        num_steps=4,
        max_iterations=25,
        tolerance=1.0e-8,
        kinematics=kinematics,
    )

    assert result.status == "completed"
    assert result.info["kinematics"] == kinematics
    assert np.max(result.element_states[1]["alpha"]) > 0.0
    assert result.element_states[1]["equivalent_stress_measure"] == "hill48"


def test_orthotropic_beam_transient_workflow_recovers_physical_stress() -> None:
    model = _orthotropic_axial_beam()
    load = _axial_load("step", 1.0e5)

    result = solve_transient_newmark(
        model,
        TransientConfig(
            dt=1.0e-5,
            t_end=3.0e-5,
            output_nodes=[2],
            output_elements=[1],
            include_stress_history=True,
        ),
        base_load_case=load,
    )

    ux_dof = model.mesh.get_node(2).dofs[0]
    assert result.status == "completed"
    assert np.max(np.abs(result.displacements[:, ux_dof])) > 0.0
    assert result.peak_von_mises_stress > 0.0
    assert result.stress_history is not None
    assert result.stress_history[-1][1]["equivalent_stress_measure"] == "von_mises"
    assert result.diagnostics["factorization_reused"] is True


def test_orthotropic_hill_staged_history_and_restart_match_single_load_path() -> None:
    staged_model = _orthotropic_axial_beam(
        hill=True,
        hill_hardening=True,
        fiber_plasticity=True,
    )
    permanent = _axial_load("permanent", 1.03e6)
    environmental = _axial_load("environmental", 0.04e6)
    program = NonlinearLoadProgram(
        [
            NonlinearLoadStage("permanent", permanent),
            NonlinearLoadStage("environmental", environmental),
        ]
    )
    staged = solve_static_nonlinear(
        staged_model,
        load_program=program,
        num_steps=2,
        max_iterations=12,
        tolerance=1.0e-9,
    )

    restart_model = _orthotropic_axial_beam(
        hill=True,
        hill_hardening=True,
        fiber_plasticity=True,
    )
    preload = solve_static_nonlinear(
        restart_model,
        permanent,
        num_steps=1,
        max_iterations=12,
        tolerance=1.0e-9,
    )
    restarted = solve_static_nonlinear(
        restart_model,
        environmental,
        constant_load_case=permanent,
        num_steps=1,
        max_iterations=12,
        tolerance=1.0e-9,
        initial_element_states=preload.element_states,
        initial_displacements=preload.displacements,
    )

    assert staged.status == preload.status == restarted.status == "completed"
    assert staged.info["load_program_stage_factors"] == {
        "permanent": 1.0,
        "environmental": 1.0,
    }
    np.testing.assert_allclose(
        restarted.displacements,
        staged.displacements,
        rtol=2.0e-10,
        atol=1.0e-14,
    )
    assert restarted.element_states[1]["equivalent_stress_measure"] == "hill48"
    assert np.max(restarted.element_states[1]["alpha"]) > 0.0
    np.testing.assert_allclose(
        restarted.element_states[1]["alpha"],
        staged.element_states[1]["alpha"],
        atol=1.0e-14,
    )


def test_orthotropic_hill_alpha_drives_fixed_strain_erosion() -> None:
    model = _orthotropic_model(hill=True)
    _constrain_shell_membrane(model)
    element = model.mesh.get_element(1)
    state = element.init_nonlinear_state(num_layers=3)
    state["alpha"][:] = 0.01
    load = LoadCase("small_tension")
    load.add_nodal_load(2, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load.add_nodal_load(3, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        num_layers=3,
        max_iterations=12,
        initial_element_states={1: state},
        initial_displacements=np.zeros(model.mesh.dof_manager.total_dofs),
        fracture_config=FractureConfig(
            threshold=0.001,
            max_deleted_fraction=0.5,
        ),
    )

    summary = result.info["fracture_summary"]
    assert result.status == "stopped_at_limit"
    assert result.failure_reason == "max_deleted_fraction_reached"
    assert summary["deleted_element_ids"] == [1]
    assert summary["records"][0]["trigger_name"] == "max_equivalent_plastic_strain"
    assert result.element_states[1]["equivalent_stress_measure"] == "hill48"
