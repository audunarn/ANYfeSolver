"""Coupled generalized beam-section contract and element qualification."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.beam_sections import (
    GENERALIZED_BEAM_RESULTANT_ORDER,
    GENERALIZED_BEAM_STRAIN_ORDER,
    GeneralizedBeamSection,
    GeneralizedBeamSectionContract,
    coerce_generalized_beam_section,
)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel


def _stiffness() -> np.ndarray:
    diagonal = np.array(
        [2.0e8, 3.0e7, 4.0e7, 5.0e6, 6.0e6, 8.0e6],
        dtype=float,
    )
    stiffness = np.diag(diagonal)
    stiffness[0, 3] = stiffness[3, 0] = 8.0e6
    stiffness[0, 5] = stiffness[5, 0] = -1.0e7
    stiffness[1, 4] = stiffness[4, 1] = 5.0e6
    stiffness[2, 5] = stiffness[5, 2] = -7.0e6
    stiffness[4, 5] = stiffness[5, 4] = 1.0e6
    return stiffness


def _model(
    element_type=BeamElement,
    *,
    section=None,
    cross_section=None,
):
    model = FEModel("generalized_beam")
    model.add_material("dummy", 210.0e9, 0.3, density=7850.0)
    positions = (0.0, 2.0) if element_type is BeamElement else (0.0, 1.0, 2.0)
    for node_id, x in enumerate(positions, start=1):
        model.add_node(node_id, x, 0.0, 0.0)
    element = element_type(
        1,
        list(range(1, len(positions) + 1)),
        "dummy",
        cross_section or {},
        section=section,
    )
    model.add_element(1, element)
    return model, element


def _beam2_constant_strain(strain: np.ndarray, length: float = 2.0) -> np.ndarray:
    eps, gamma_y, gamma_z, twist, kappa_y, kappa_z = strain
    displacement = np.zeros(12, dtype=float)
    displacement[6] = eps * length
    displacement[7] = gamma_y * length
    displacement[8] = gamma_z * length
    displacement[3] = -0.5 * twist * length
    displacement[9] = 0.5 * twist * length
    displacement[4] = -0.5 * kappa_y * length
    displacement[10] = 0.5 * kappa_y * length
    displacement[5] = -0.5 * kappa_z * length
    displacement[11] = 0.5 * kappa_z * length
    return displacement


def _beam3_constant_strain(strain: np.ndarray, length: float = 2.0) -> np.ndarray:
    eps, gamma_y, gamma_z, twist, kappa_y, kappa_z = strain
    displacement = np.zeros(18, dtype=float)
    for index, x in enumerate((0.0, 0.5 * length, length)):
        centered = x - 0.5 * length
        base = 6 * index
        displacement[base + 0] = eps * x
        displacement[base + 1] = (
            gamma_y * x + 0.5 * kappa_z * centered**2
        )
        displacement[base + 2] = (
            gamma_z * x - 0.5 * kappa_y * centered**2
        )
        displacement[base + 3] = twist * x
        displacement[base + 4] = kappa_y * centered
        displacement[base + 5] = kappa_z * centered
    return displacement


def _beam2_reference(
    stiffness: np.ndarray,
    displacement: np.ndarray,
    length: float = 2.0,
):
    """Independent force-based reference for a constant coupled section."""

    compliance = np.linalg.inv(stiffness)
    flexibility = np.zeros((6, 6), dtype=float)
    gauss_points, gauss_weights = np.polynomial.legendre.leggauss(3)

    def force_interpolation(xi):
        x = 0.5 * (float(xi) + 1.0)
        matrix = np.zeros((6, 6), dtype=float)
        matrix[0, 0] = matrix[3, 1] = 1.0
        matrix[2, 2:4] = 1.0 / length
        matrix[4, 2:4] = (-1.0 + x, x)
        matrix[1, 4:6] = -1.0 / length
        matrix[5, 4:6] = (-1.0 + x, x)
        return matrix

    for xi, weight in zip(gauss_points, gauss_weights):
        force_shape = force_interpolation(xi)
        flexibility += (
            force_shape.T
            @ compliance
            @ force_shape
            * (length / 2.0 * weight)
        )
    basic_stiffness = np.linalg.inv(flexibility)
    transform = np.zeros((6, 12), dtype=float)
    transform[0, (0, 6)] = (-1.0, 1.0)
    transform[1, (3, 9)] = (-1.0, 1.0)
    transform[2, (2, 8, 4)] = (-1.0 / length, 1.0 / length, 1.0)
    transform[3, (2, 8, 10)] = (-1.0 / length, 1.0 / length, 1.0)
    transform[4, (1, 7, 5)] = (1.0 / length, -1.0 / length, 1.0)
    transform[5, (1, 7, 11)] = (1.0 / length, -1.0 / length, 1.0)
    element_stiffness = transform.T @ basic_stiffness @ transform
    basic_force = basic_stiffness @ transform @ displacement
    stations = (-1.0, 0.0, 1.0)
    resultants = np.asarray(
        [force_interpolation(xi) @ basic_force for xi in stations]
    )
    strains = resultants @ compliance.T
    return element_stiffness, strains, resultants


def test_generalized_section_contract_and_validation() -> None:
    section = GeneralizedBeamSection(_stiffness(), name="coupled")

    assert isinstance(section, GeneralizedBeamSectionContract)
    assert section.name == "coupled"
    assert GENERALIZED_BEAM_STRAIN_ORDER == (
        "eps_x",
        "gamma_xy",
        "gamma_xz",
        "kappa_x",
        "kappa_y",
        "kappa_z",
    )
    assert GENERALIZED_BEAM_RESULTANT_ORDER == (
        "N",
        "V_y",
        "V_z",
        "T",
        "M_y",
        "M_z",
    )

    asymmetric = _stiffness()
    asymmetric[0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        GeneralizedBeamSection(asymmetric)
    indefinite = _stiffness()
    indefinite[0, 3] = indefinite[3, 0] = 4.0e7
    with pytest.raises(ValueError, match="positive definite"):
        GeneralizedBeamSection(indefinite)
    with pytest.raises(ValueError, match=r"shape \(6, 6\)"):
        GeneralizedBeamSection(np.eye(5))


def test_external_section_protocol_and_mapping_coercion() -> None:
    class ExternalSection:
        name = "external"

        def generalized_stiffness_matrix(self):
            return _stiffness()

    external = coerce_generalized_beam_section(ExternalSection())
    assert external.name == "external"
    mapped = coerce_generalized_beam_section(
        {"name": "mapped", "generalized_stiffness": _stiffness()}
    )
    assert mapped.name == "mapped"


@pytest.mark.parametrize(
    ("element_type", "displacement_factory"),
    [
        (BeamElement, _beam2_constant_strain),
        (QuadraticBeamElement, _beam3_constant_strain),
    ],
)
def test_coupled_linear_energy_and_resultant_recovery(
    element_type,
    displacement_factory,
) -> None:
    stiffness = _stiffness()
    section = GeneralizedBeamSection(stiffness, name="coupled")
    model, element = _model(element_type, section=section)
    material = model.get_material("dummy")
    strain = np.array(
        [2.0e-4, -3.0e-4, 4.0e-4, 5.0e-4, -6.0e-4, 7.0e-4]
    )
    displacement = displacement_factory(strain)

    element_stiffness = element.compute_stiffness_matrix(model.mesh, material)
    recovered = element.compute_stresses(
        model.mesh,
        displacement,
        material,
    )
    if element_type is BeamElement:
        reference_stiffness, expected_strain, expected_resultant = (
            _beam2_reference(stiffness, displacement)
        )
        np.testing.assert_allclose(
            element_stiffness,
            reference_stiffness,
            rtol=3.0e-13,
            atol=2.0e-8,
        )
    else:
        assert displacement @ element_stiffness @ displacement == pytest.approx(
            2.0 * float(strain @ stiffness @ strain),
            rel=5.0e-13,
            abs=1.0e-12,
        )
        expected_strain = np.broadcast_to(strain, (3, 6))
        expected_resultant = np.broadcast_to(
            stiffness @ strain,
            (3, 6),
        )
    np.testing.assert_allclose(
        recovered["generalized_strain"],
        expected_strain,
        rtol=2.0e-13,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        recovered["generalized_resultant"],
        expected_resultant,
        rtol=2.0e-13,
        atol=1.0e-10,
    )
    assert recovered["physical_stress_available"] is False
    assert recovered["recovery_scope"] == "section_resultants_only"
    assert "von_mises" not in recovered


def test_cross_section_opt_in_and_ambiguous_inputs_fail_closed() -> None:
    model, element = _model(
        BeamElement,
        cross_section={"generalized_stiffness": _stiffness()},
    )
    assert element.generalized_section is not None
    assert element.compute_stiffness_matrix(
        model.mesh,
        model.get_material("dummy"),
    ).shape == (12, 12)

    with pytest.raises(ValueError, match="either with section"):
        BeamElement(
            2,
            [1, 2],
            "dummy",
            {"generalized_stiffness": _stiffness()},
            section=GeneralizedBeamSection(_stiffness()),
        )
    with pytest.raises(ValueError, match="fiber_plasticity"):
        BeamElement(
            3,
            [1, 2],
            "dummy",
            {"fiber_plasticity": True},
            section=GeneralizedBeamSection(_stiffness()),
        )
    with pytest.raises(ValueError, match="does not support corotational"):
        QuadraticBeamElement(
            4,
            [1, 2, 3],
            "dummy",
            {"geometric_nonlinearity": "corotational"},
            section=GeneralizedBeamSection(_stiffness()),
        )


def test_beam2_diagonal_generalized_section_matches_legacy_timoshenko() -> None:
    model = FEModel("diagonal_limit")
    elastic_modulus = 210.0e9
    poisson_ratio = 0.3
    model.add_material("steel", elastic_modulus, poisson_ratio)
    material = model.get_material("steel")
    scalar_section = {
        "area": 0.012,
        "Iy": 2.3e-6,
        "Iz": 3.7e-6,
        "J": 8.0e-7,
        "shear_factor_y": 0.81,
        "shear_factor_z": 0.76,
    }
    shear_modulus = elastic_modulus / (2.0 * (1.0 + poisson_ratio))
    generalized = GeneralizedBeamSection(
        np.diag(
            (
                elastic_modulus * scalar_section["area"],
                shear_modulus
                * scalar_section["area"]
                * scalar_section["shear_factor_y"],
                shear_modulus
                * scalar_section["area"]
                * scalar_section["shear_factor_z"],
                shear_modulus * scalar_section["J"],
                elastic_modulus * scalar_section["Iy"],
                elastic_modulus * scalar_section["Iz"],
            )
        )
    )
    legacy = BeamElement(1, [1, 2], "steel", scalar_section)
    coupled = BeamElement(2, [1, 2], "steel", section=generalized)

    np.testing.assert_allclose(
        coupled._local_linear_stiffness(2.4, material),
        legacy._local_linear_stiffness(2.4, material),
        rtol=8.0e-16,
        atol=2.0e-8,
    )


@pytest.mark.parametrize("element_type", [BeamElement, QuadraticBeamElement])
def test_generalized_section_mass_matrix_overrides_material_geometry(
    element_type,
) -> None:
    section_mass = np.diag((10.0, 11.0, 12.0, 2.0, 3.0, 4.0))
    section_mass[0, 5] = section_mass[5, 0] = 0.5
    section = GeneralizedBeamSection(
        _stiffness(),
        mass_matrix=section_mass,
    )
    model, element = _model(element_type, section=section)
    mass = element.compute_mass_matrix(model.mesh, model.get_material("dummy"))
    velocity = np.zeros(mass.shape[0], dtype=float)
    for node_index in range(element.num_nodes):
        velocity[6 * node_index : 6 * node_index + 6] = (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
        )
    local_velocity = np.array((1.0, 0.0, 0.0, 0.0, 0.0, 2.0))
    assert velocity @ mass @ velocity == pytest.approx(
        2.0 * float(local_velocity @ section_mass @ local_velocity),
        rel=2.0e-13,
    )


@pytest.mark.parametrize("element_type", [BeamElement, QuadraticBeamElement])
def test_generalized_von_karman_tangent_matches_central_difference(
    element_type,
) -> None:
    model, element = _model(
        element_type,
        section=GeneralizedBeamSection(_stiffness()),
    )
    material = model.get_material("dummy")
    displacement = np.linspace(-2.0e-3, 3.0e-3, element.total_dofs)
    force, tangent, state = element.compute_nonlinear_response(
        model.mesh,
        material,
        displacement,
        tangent=True,
    )
    assert tangent is not None
    numerical = np.zeros_like(tangent)
    step = 2.0e-7
    for column in range(element.total_dofs):
        plus = displacement.copy()
        minus = displacement.copy()
        plus[column] += step
        minus[column] -= step
        force_plus = element.compute_nonlinear_response(
            model.mesh,
            material,
            plus,
            tangent=False,
        )[0]
        force_minus = element.compute_nonlinear_response(
            model.mesh,
            material,
            minus,
            tangent=False,
        )[0]
        numerical[:, column] = (force_plus - force_minus) / (2.0 * step)

    np.testing.assert_allclose(tangent, tangent.T, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(tangent, numerical, rtol=3.0e-7, atol=2.0e-2)
    assert np.all(np.isfinite(force))
    assert state["recovery_scope"] == "section_resultants_only"


def test_generalized_beam2_corotational_response_routes_section_coupling() -> None:
    section_data = _stiffness()
    model, element = _model(
        BeamElement,
        section=GeneralizedBeamSection(section_data),
        cross_section={"geometric_nonlinearity": "corotational"},
    )
    displacement = np.zeros(12, dtype=float)
    displacement[6] = 2.0e-3
    force, tangent, state = element.compute_nonlinear_response(
        model.mesh,
        model.get_material("dummy"),
        displacement,
        tangent=True,
    )
    assert np.asarray(state["generalized_resultant"]).shape == (3, 6)
    assert np.max(np.abs(state["generalized_resultant"][:, 3])) > 0.0
    assert abs(force[3]) > 0.0
    assert tangent is not None and np.all(np.isfinite(tangent))


@pytest.mark.parametrize("element_type", [BeamElement, QuadraticBeamElement])
def test_generalized_geometric_stiffness_ignores_legacy_section_geometry(
    element_type,
) -> None:
    section = GeneralizedBeamSection(_stiffness(), name="same-section")
    model_a, element_a = _model(
        element_type,
        section=section,
        cross_section={
            "area": 2.0e-4,
            "Iy": 3.0e-10,
            "Iz": 7.0e-10,
        },
    )
    model_b, element_b = _model(
        element_type,
        section=section,
        cross_section={
            "area": 9.0,
            "Iy": 4.0,
            "Iz": 5.0,
        },
    )
    state = {"axial_compression": 2.5e5}

    geometric_a = element_a.compute_geometric_stiffness_matrix(
        model_a.mesh,
        model_a.get_material("dummy"),
        state,
    )
    geometric_b = element_b.compute_geometric_stiffness_matrix(
        model_b.mesh,
        model_b.get_material("dummy"),
        state,
    )

    np.testing.assert_allclose(geometric_a, geometric_b, rtol=0.0, atol=0.0)
    assert np.max(np.abs(geometric_a)) > 0.0
    rotation_x = (
        (3, 9)
        if element_type is BeamElement
        else (3, 9, 15)
    )
    np.testing.assert_allclose(
        geometric_a[np.ix_(rotation_x, rotation_x)],
        0.0,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("element_type", [BeamElement, QuadraticBeamElement])
def test_generalized_wagner_term_requires_explicit_geometric_metadata(
    element_type,
) -> None:
    model, element = _model(
        element_type,
        section=GeneralizedBeamSection(_stiffness()),
        cross_section={"geometric_polar_radius_squared": 2.5e-4},
    )
    geometric = element.compute_geometric_stiffness_matrix(
        model.mesh,
        model.get_material("dummy"),
        {"axial_compression": 3.0e5},
    )
    rotation_x = (
        (3, 9)
        if element_type is BeamElement
        else (3, 9, 15)
    )
    assert np.max(
        np.abs(geometric[np.ix_(rotation_x, rotation_x)])
    ) > 0.0

    element.cross_section["geometric_polar_radius_squared"] = -1.0
    with pytest.raises(ValueError, match="finite non-negative"):
        element.compute_geometric_stiffness_matrix(
            model.mesh,
            model.get_material("dummy"),
            {"axial_compression": 3.0e5},
        )
