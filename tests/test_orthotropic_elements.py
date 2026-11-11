"""Element-level qualification for orthotropic shell and beam behavior."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import (
    BeamElement,
    DNVC208MaterialCurve,
    FEModel,
    Hill48Yield,
    Material,
    OrthotropicMaterial,
    QuadraticBeamElement,
    ShellElement,
    assemble_stiffness_matrix,
    create_shell_element,
    recover_stress_result,
    validate_production_model,
)
from anyfesolver.elements import (
    _shell_material_matrices,
    validate_initial_field_state,
)
from anyfesolver.nonlinear_static import states_von_mises_map


def _orthotropic(
    name: str = "ortho",
    *,
    plastic: bool = False,
) -> OrthotropicMaterial:
    hill = None
    curve = None
    if plastic:
        yield_stress = 100.0e6
        shear_yield = yield_stress / np.sqrt(3.0)
        hill = Hill48Yield(
            yield_stress,
            yield_stress,
            yield_stress,
            shear_yield,
            shear_yield,
            shear_yield,
        )
        curve = DNVC208MaterialCurve(
            sigma_prop=yield_stress,
            sigma_yield=110.0e6,
            sigma_yield_2=120.0e6,
            eps_p_y1=0.002,
            eps_p_y2=0.01,
            K=500.0e6,
            n=0.2,
        )
    return OrthotropicMaterial(
        name=name,
        elastic_modulus_1=150.0e9,
        elastic_modulus_2=10.0e9,
        elastic_modulus_3=8.0e9,
        poisson_ratio_12=0.25,
        poisson_ratio_13=0.20,
        poisson_ratio_23=0.30,
        shear_modulus_12=5.0e9,
        shear_modulus_13=4.0e9,
        shear_modulus_23=3.0e9,
        density=1600.0,
        hill_yield=hill,
        hardening_curve=curve,
    )


def _hill_only_orthotropic(
    name: str = "hill_only",
    *,
    directional: bool = False,
) -> OrthotropicMaterial:
    material = _orthotropic(name)
    if directional:
        material.hill_yield = Hill48Yield(
            100.0e6,
            200.0e6,
            100.0e6,
            60.0e6,
            60.0e6,
            60.0e6,
        )
    else:
        yield_stress = 100.0e6
        shear_yield = yield_stress / np.sqrt(3.0)
        material.hill_yield = Hill48Yield(
            yield_stress,
            yield_stress,
            yield_stress,
            shear_yield,
            shear_yield,
            shear_yield,
        )
    material.hardening_curve = None
    return material


def _square_shell(
    material: OrthotropicMaterial,
    *,
    angle: float = 0.0,
    direction=None,
) -> tuple[FEModel, ShellElement]:
    model = FEModel("orthotropic_shell")
    model.register_material(material)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.0, 1.0, 0.0)
    model.add_node(4, 0.0, 1.0, 0.0)
    element = create_shell_element(
        1,
        [1, 2, 3, 4],
        material.name,
        thickness=0.01,
        material_direction=direction,
        material_angle_deg=angle,
        # Hill48 plasticity has no stable qualified Q4 route (typed refusal);
        # the Hill shell physics is exercised on the legacy compatible element.
        **({"formulation": "legacy"} if material.hill_yield is not None else {}),
    )
    model.add_element(1, element)
    return model, element


def _constant_x_strain_displacements(strain: float) -> np.ndarray:
    values = np.zeros(24, dtype=float)
    values[6] = strain
    values[12] = strain
    return values


def _shell_topology(
    material: OrthotropicMaterial,
    node_count: int,
    *,
    angle: float,
) -> tuple[FEModel, ShellElement, np.ndarray]:
    coordinates = {
        3: ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        4: (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        6: (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.0),
        ),
        8: (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.0, 0.0),
            (1.0, 0.5, 0.0),
            (0.5, 1.0, 0.0),
            (0.0, 0.5, 0.0),
        ),
    }[node_count]
    model = FEModel(f"orthotropic_s{node_count}")
    model.register_material(material)
    for node_id, coordinate in enumerate(coordinates, start=1):
        model.add_node(node_id, *coordinate)
    node_ids = list(range(1, node_count + 1))
    if node_count == 4:
        element = create_shell_element(
            1,
            node_ids,
            material.name,
            thickness=0.01,
            material_angle_deg=angle,
        )
    else:
        element = ShellElement(
            1,
            node_ids,
            material.name,
            thickness=0.01,
            material_angle_deg=angle,
        )
    model.add_element(1, element)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        displacement[node.dofs[0]] = 1.0e-4 * node.x
    return model, element, displacement


@pytest.mark.parametrize("angle", [0.0, 45.0, 90.0])
def test_shell_stress_matches_rotated_constitutive_matrix(angle: float) -> None:
    material = _orthotropic()
    model, element = _square_shell(material, angle=angle)
    strain = 1.0e-4

    stresses = element.compute_stresses(
        model.mesh,
        _constant_x_strain_displacements(strain),
        material,
    )
    expected = _shell_material_matrices(material, np.deg2rad(angle))[0] @ np.array(
        [strain, 0.0, 0.0],
    )

    assert np.allclose(
        [
            stresses["membrane_xx"],
            stresses["membrane_yy"],
            stresses["membrane_xy"],
        ],
        expected[:, None],
        rtol=2.0e-10,
        atol=1.0e-6,
    )
    assert stresses["equivalent_stress_measure"] == "von_mises"


def test_shell_global_direction_and_angle_compose_deterministically() -> None:
    material = _orthotropic()
    model_angle, element_angle = _square_shell(material, angle=90.0)
    model_direction, element_direction = _square_shell(
        _orthotropic("other"),
        direction=(0.0, 1.0, 0.0),
    )
    displacement = _constant_x_strain_displacements(2.0e-4)

    angle_stress = element_angle.compute_stresses(
        model_angle.mesh,
        displacement,
        material,
    )
    direction_stress = element_direction.compute_stresses(
        model_direction.mesh,
        displacement,
        model_direction.get_material("other"),
    )

    for key in ("membrane_xx", "membrane_yy", "membrane_xy"):
        assert np.allclose(angle_stress[key], direction_stress[key])

    combined_model, combined = _square_shell(
        _orthotropic("combined"),
        direction=(0.0, 1.0, 0.0),
        angle=45.0,
    )
    reference_model, reference = _square_shell(
        _orthotropic("reference"),
        angle=135.0,
    )
    combined_stress = combined.compute_stresses(
        combined_model.mesh,
        displacement,
        combined_model.get_material("combined"),
    )
    reference_stress = reference.compute_stresses(
        reference_model.mesh,
        displacement,
        reference_model.get_material("reference"),
    )
    for key in ("membrane_xx", "membrane_yy", "membrane_xy"):
        assert np.allclose(combined_stress[key], reference_stress[key])


@pytest.mark.parametrize("node_count", [3, 4, 6, 8])
def test_off_axis_membrane_patch_is_consistent_for_every_shell_topology(
    node_count: int,
) -> None:
    material = _orthotropic(f"s{node_count}")
    model, element, displacement = _shell_topology(
        material,
        node_count,
        angle=45.0,
    )
    expected = _shell_material_matrices(material, np.deg2rad(45.0))[0] @ np.array(
        [1.0e-4, 0.0, 0.0],
    )

    stiffness = element.compute_stiffness_matrix(model.mesh, material)
    stresses = element.compute_stresses(model.mesh, displacement, material)

    assert np.all(np.isfinite(stiffness))
    assert np.allclose(stiffness, stiffness.T, rtol=1.0e-12, atol=1.0e-5)
    assert np.allclose(stresses["membrane_xx"], expected[0], rtol=2.0e-9)
    assert np.allclose(stresses["membrane_yy"], expected[1], rtol=2.0e-9)
    assert np.allclose(stresses["membrane_xy"], expected[2], rtol=2.0e-9)
    assert abs(expected[2]) > 1.0e5


@pytest.mark.parametrize("node_count", [3, 4, 6, 8])
def test_orthotropic_shell_bending_shear_and_rigid_body_patches(
    node_count: int,
) -> None:
    material = _orthotropic(f"patch_s{node_count}")
    model, element, _membrane = _shell_topology(
        material,
        node_count,
        angle=35.0,
    )
    stiffness = element.compute_stiffness_matrix(model.mesh, material)
    Q_local, G_local, _strain_transform, _stress_transform = (
        _shell_material_matrices(material, np.deg2rad(35.0))
    )

    curvature = 1.1e-3
    bending = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        bending[node.dofs[4]] = curvature * node.x
    bending_stress = element.compute_stresses(model.mesh, bending, material)
    expected_bending = (
        0.5
        * element.thickness
        * Q_local
        @ np.array([curvature, 0.0, 0.0])
    )
    assert np.allclose(
        bending_stress["bending_xx"],
        expected_bending[0],
        rtol=3.0e-9,
        atol=1.0e-5,
    )
    assert np.allclose(
        bending_stress["bending_xy"],
        expected_bending[2],
        rtol=3.0e-9,
        atol=1.0e-5,
    )

    gamma_xz = 2.0e-4
    shear = np.zeros_like(bending)
    for node in model.mesh.nodes.values():
        shear[node.dofs[2]] = gamma_xz * node.x
    shear_stress = element.compute_stresses(model.mesh, shear, material)
    expected_shear = (5.0 / 6.0) * G_local @ np.array(
        [gamma_xz, 0.0]
    )
    assert np.allclose(
        shear_stress["shear_xz"],
        expected_shear[0],
        rtol=3.0e-9,
        atol=1.0e-5,
    )
    assert np.allclose(
        shear_stress["shear_yz"],
        expected_shear[1],
        rtol=3.0e-9,
        atol=1.0e-5,
    )

    scale = max(float(np.max(np.abs(np.diag(stiffness)))), 1.0)
    coordinates = element.get_node_coordinates(model.mesh)
    centroid = np.mean(coordinates, axis=0)
    rigid_modes = []
    for direction in range(3):
        mode = np.zeros(element.total_dofs, dtype=float)
        mode[direction::6] = 1.0
        rigid_modes.append(mode)
    for axis in range(3):
        rotation = np.zeros(3, dtype=float)
        rotation[axis] = 1.0e-3
        mode = np.zeros(element.total_dofs, dtype=float)
        for index, coordinate in enumerate(coordinates):
            mode[6 * index : 6 * index + 3] = np.cross(
                rotation,
                coordinate - centroid,
            )
            mode[6 * index + 3 : 6 * index + 6] = rotation
        rigid_modes.append(mode)
    for mode in rigid_modes:
        assert abs(float(mode @ stiffness @ mode)) < 1.0e-10 * scale


def test_shell_isotropic_limit_and_orthotropic_q4_default_path() -> None:
    E = 70.0e9
    nu = 0.25
    G = E / (2.0 * (1.0 + nu))
    isotropic = Material("iso", E, nu)
    orthotropic = OrthotropicMaterial(
        "ortho",
        E,
        E,
        E,
        nu,
        nu,
        nu,
        G,
        G,
        G,
    )
    iso_model, iso_element = _square_shell(
        OrthotropicMaterial(
            "temporary",
            E,
            E,
            E,
            nu,
            nu,
            nu,
            G,
            G,
            G,
        )
    )
    iso_model.materials["temporary"] = isotropic
    ortho_model, ortho_element = _square_shell(orthotropic)

    K_iso = iso_element.compute_stiffness_matrix(iso_model.mesh, isotropic)
    K_ortho = ortho_element.compute_stiffness_matrix(ortho_model.mesh, orthotropic)
    assert np.allclose(K_iso, K_ortho, rtol=2.0e-12, atol=1.0e-5)

    _matrix, info = assemble_stiffness_matrix(ortho_model)
    assert info["diagnostics"]["qualified_e4_pl_stiffness"] == {
        "path": "shared_geometry_cache",
        "element_count": 1,
        "unique_geometry_count": 1,
    }


def test_orthotropic_shell_hill_state_is_material_axis_and_physical_stress_is_stored() -> None:
    material = _orthotropic(plastic=True)
    model, element = _square_shell(material, angle=30.0)

    force, tangent, state = element.compute_nonlinear_response(
        model.mesh,
        material,
        _constant_x_strain_displacements(0.02),
        num_layers=5,
        tangent=True,
    )

    assert np.all(np.isfinite(force))
    assert tangent is not None and np.all(np.isfinite(tangent))
    assert np.linalg.norm(tangent - tangent.T) <= 1.0e-10 * np.linalg.norm(tangent)
    assert np.max(state["alpha"]) > 0.0
    assert state["plastic_strain"].shape == (20, 3)
    assert state["layer_stress"].shape == (20, 3)
    assert state["layer_stress_material"].shape == (20, 3)
    assert state["equivalent_stress_measure"] == "hill48"


def test_hill_only_shell_is_elastic_perfectly_plastic() -> None:
    material = _hill_only_orthotropic()
    model, element = _square_shell(material, angle=30.0)

    _force, tangent, state = element.compute_nonlinear_response(
        model.mesh,
        material,
        _constant_x_strain_displacements(0.02),
        num_layers=5,
        tangent=True,
    )

    assert tangent is not None and np.all(np.isfinite(tangent))
    assert np.max(state["alpha"]) > 0.0
    assert np.max(np.abs(state["layer_stress_material"])) > 0.0
    assert state["equivalent_stress_measure"] == "hill48"


def test_orthotropic_beam_uses_directional_shear_and_explicit_torsion() -> None:
    material = _orthotropic()
    model = FEModel("orthotropic_beam")
    model.register_material(material)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 2.0, 0.0, 0.0)
    section = {
        "area": 0.01,
        "Iy": 2.0e-6,
        "Iz": 3.0e-6,
        "J": 1.0e-6,
        "torsional_rigidity": 9.0e3,
        "orientation": (0.0, 0.0, 1.0),
    }
    element = BeamElement(1, [1, 2], material.name, section)
    model.add_element(1, element)

    local = element._local_linear_stiffness(2.0, material)
    assert local[0, 0] == pytest.approx(material.elastic_modulus_1 * section["area"] / 2.0)
    assert local[3, 3] == pytest.approx(section["torsional_rigidity"] / 2.0)

    displacement = np.zeros(12)
    displacement[7] = 0.01
    displacement[8] = 0.01
    stresses = element.compute_stresses(model.mesh, displacement, material)
    assert abs(stresses["shear_stress_y"]) > abs(stresses["shear_stress_z"])


def test_quadratic_orthotropic_beam_matches_directional_energy_terms() -> None:
    material = _orthotropic()
    model = FEModel("orthotropic_beam3")
    model.register_material(material)
    for node_id, x in enumerate((0.0, 1.0, 2.0), start=1):
        model.add_node(node_id, x, 0.0, 0.0)
    section = {
        "area": 0.01,
        "Iy": 2.0e-6,
        "Iz": 3.0e-6,
        "J": 1.0e-6,
        "torsional_rigidity": 9.0e3,
    }
    element = QuadraticBeamElement(1, [1, 2, 3], material.name, section)
    model.add_element(1, element)
    stiffness = element.compute_stiffness_matrix(model.mesh, material)
    length = 2.0

    def mode(component: int, gradient: float) -> np.ndarray:
        displacement = np.zeros(18, dtype=float)
        for index, x in enumerate((0.0, 1.0, 2.0)):
            displacement[6 * index + component] = gradient * x
        return displacement

    cases = (
        (mode(0, 2.0e-4), material.elastic_modulus_1 * section["area"]),
        (
            mode(1, 3.0e-4),
            material.shear_modulus_12
            * section["area"]
            * (5.0 / 6.0),
        ),
        (
            mode(2, 4.0e-4),
            material.shear_modulus_13
            * section["area"]
            * (5.0 / 6.0),
        ),
        (mode(3, 5.0e-4), section["torsional_rigidity"]),
    )
    for displacement, rigidity in cases:
        nonzero = np.flatnonzero(displacement)
        applied_gradient = displacement[nonzero[-1]] / 2.0
        assert displacement @ stiffness @ displacement == pytest.approx(
            rigidity * applied_gradient**2 * length,
            rel=2.0e-12,
        )


@pytest.mark.parametrize(
    ("element_type", "node_positions"),
    [
        (BeamElement, (0.0, 2.0)),
        (QuadraticBeamElement, (0.0, 1.0, 2.0)),
    ],
)
@pytest.mark.parametrize("with_hardening", [False, True])
def test_orthotropic_beam_fiber_plasticity_uses_x_strength(
    element_type,
    node_positions,
    with_hardening: bool,
) -> None:
    material = (
        _orthotropic(plastic=True)
        if with_hardening
        else _hill_only_orthotropic()
    )
    model = FEModel("orthotropic_beam_fibers")
    model.register_material(material)
    for node_id, x in enumerate(node_positions, start=1):
        model.add_node(node_id, x, 0.0, 0.0)
    section = {
        "area": 0.01,
        "Iy": 2.0e-6,
        "Iz": 3.0e-6,
        "J": 1.0e-6,
        "torsional_rigidity": 9.0e3,
        "fiber_plasticity": True,
    }
    element = element_type(
        1,
        list(range(1, len(node_positions) + 1)),
        material.name,
        section,
    )
    model.add_element(1, element)
    displacement = np.zeros(6 * len(node_positions), dtype=float)
    for index, x in enumerate(node_positions):
        displacement[6 * index] = 0.02 * x

    force, tangent, state = element.compute_nonlinear_response(
        model.mesh,
        material,
        displacement,
        tangent=True,
    )

    assert np.all(np.isfinite(force))
    assert tangent is not None and np.all(np.isfinite(tangent))
    assert np.max(state["alpha"]) > 0.0
    assert np.max(np.abs(state["fiber_stress"])) >= 0.99 * material.hill_yield.X
    assert state["equivalent_stress_measure"] == "hill48"


def test_beam_hill_equivalent_includes_xz_shear_and_torsion() -> None:
    material = _hill_only_orthotropic(directional=True)
    model = FEModel("orthotropic_beam_equivalent")
    model.register_material(material)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 2.0, 0.0, 0.0)
    section = {
        "area": 0.01,
        "Iy": 2.0e-6,
        "Iz": 3.0e-6,
        "J": 1.0e-6,
        "torsional_rigidity": 9.0e3,
    }
    element = BeamElement(1, [1, 2], material.name, section)
    model.add_element(1, element)
    displacement = np.zeros(12, dtype=float)
    displacement[8] = 0.01
    displacement[9] = 0.02

    stresses = element.compute_stresses(model.mesh, displacement, material)

    assert abs(stresses["shear_stress_z"]) > 0.0
    assert abs(stresses["torsional_stress"]) > 0.0
    assert stresses["equivalent_stress"] > 0.0
    assert stresses["hill_utilization"] > 0.0
    assert "torsion" in stresses["equivalent_stress_scope"]


def test_orthotropic_initial_fields_use_rotated_hill_surface_and_x_scaling() -> None:
    material = _hill_only_orthotropic(directional=True)
    accepted_model, accepted = _square_shell(material, angle=90.0)
    state = {"initial_membrane_stress": np.array([150.0e6, 0.0, 0.0])}

    validate_initial_field_state(
        accepted,
        material,
        state,
        3,
        accepted_model.mesh,
    )

    rejected_model, rejected = _square_shell(
        _hill_only_orthotropic("rejected", directional=True),
        angle=0.0,
    )
    with pytest.raises(ValueError, match="outside the supplied"):
        validate_initial_field_state(
            rejected,
            rejected_model.get_material("rejected"),
            state,
            3,
            rejected_model.mesh,
        )

    beam_model = FEModel("beam_initial_hill")
    beam_material = _hill_only_orthotropic("beam_hill")
    beam_model.register_material(beam_material)
    beam_model.add_node(1, 0.0, 0.0, 0.0)
    beam_model.add_node(2, 1.0, 0.0, 0.0)
    beam = BeamElement(
        1,
        [1, 2],
        beam_material.name,
        {
            "area": 0.01,
            "Iy": 1.0e-6,
            "Iz": 1.0e-6,
            "J": 1.0e-6,
            "torsional_rigidity": 1.0e4,
            "fiber_plasticity": True,
        },
    )
    validate_initial_field_state(
        beam,
        beam_material,
        {"initial_fiber_stress": np.array([90.0e6])},
        3,
        beam_model.mesh,
    )
    with pytest.raises(ValueError, match="outside the supplied"):
        validate_initial_field_state(
            beam,
            beam_material,
            {"initial_fiber_stress": np.array([110.0e6])},
            3,
            beam_model.mesh,
        )


def test_orthotropic_recovery_reports_physical_vm_and_current_hill_utilization() -> None:
    material = _orthotropic(plastic=True)
    model, element = _square_shell(material, angle=25.0)
    displacement = _constant_x_strain_displacements(0.02)
    _force, _tangent, state = element.compute_nonlinear_response(
        model.mesh,
        material,
        displacement,
        num_layers=5,
        tangent=True,
    )

    recovered = recover_stress_result(
        model,
        displacement,
        element_states={1: state},
    ).element_stresses[1]

    assert recovered["equivalent_stress_measure"] == "hill48"
    assert np.max(recovered["hill_utilization"]) <= 1.0 + 5.0e-7
    assert recovered["equivalent_stress_scope"].startswith("return-mapped")

    shear_displacement = np.zeros(24, dtype=float)
    shear_displacement[8] = 0.01
    shear_displacement[14] = 0.01
    zero_force, _zero_tangent, zero_state = element.compute_nonlinear_response(
        model.mesh,
        material,
        np.zeros(24, dtype=float),
        num_layers=5,
        tangent=True,
    )
    assert np.all(np.isfinite(zero_force))
    shear_recovered = recover_stress_result(
        model,
        shear_displacement,
        element_states={1: zero_state},
    ).element_stresses[1]
    assert np.max(shear_recovered["von_mises"]) > np.max(
        shear_recovered["in_plane_von_mises"]
    )


def test_orthotropic_beam_and_shell_orientation_fail_closed() -> None:
    material = _orthotropic()
    model = FEModel("invalid_orthotropic")
    model.register_material(material)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    beam = BeamElement(
        1,
        [1, 2],
        material.name,
        {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
    )
    model.add_element(1, beam)

    with pytest.raises(ValueError, match="torsional_rigidity"):
        beam.compute_stiffness_matrix(model.mesh, material)
    report = validate_production_model(model, allow_free_mechanisms=True)
    assert any(issue.code == "BEAM004" for issue in report.errors)
    beam.cross_section["torsional_rigidity"] = "invalid"
    report = validate_production_model(model, allow_free_mechanisms=True)
    assert any(issue.code == "BEAM004" for issue in report.errors)
    with pytest.raises(ValueError, match="torsional_rigidity"):
        beam.compute_stiffness_matrix(model.mesh, material)

    shell_model, shell = _square_shell(
        _orthotropic("normal"),
        direction=(0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="parallel to the shell normal"):
        shell.compute_stiffness_matrix(
            shell_model.mesh,
            shell_model.get_material("normal"),
        )


def test_incomplete_orthotropic_restart_state_never_uses_isotropic_reconstruction() -> None:
    material = _orthotropic()
    model, _element = _square_shell(material)
    incomplete_state = {
        "layer_strain": np.ones((4, 3), dtype=float),
        "plastic_strain": np.zeros((4, 3), dtype=float),
        "alpha": np.zeros(4, dtype=float),
    }

    assert states_von_mises_map(model, {1: incomplete_state}) == {}

    hill_material = _hill_only_orthotropic()
    hill_model, _hill_element = _square_shell(hill_material)
    local_only_state = {
        "layer_strain": np.zeros((12, 3), dtype=float),
        "layer_stress": np.zeros((12, 3), dtype=float),
        "plastic_strain": np.zeros((12, 3), dtype=float),
        "alpha": np.zeros(12, dtype=float),
    }
    recovered = recover_stress_result(
        hill_model,
        np.zeros(hill_model.mesh.dof_manager.total_dofs),
        element_states={1: local_only_state},
    )
    assert recovered.provenance.per_element_source[1] == (
        "elastic_displacement_reconstruction"
    )
    assert "material-axis layer_stress_material" in (
        recovered.provenance.fallback_reasons[1]
    )
