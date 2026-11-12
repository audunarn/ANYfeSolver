from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import (
    GENERALIZED_BEAM_RESULTANT_ORDER,
    GENERALIZED_BEAM_STRAIN_ORDER,
    GeneralizedBeamSection,
    GeneralizedBeamSectionContract,
    GeneralizedShellSection,
    GeneralizedShellSectionProtocol,
    build_fe_model_from_generated_geometry,
    recover_prestress_from_static_result,
    validate_generalized_beam_section,
    validate_generalized_shell_section,
)
from anyfesolver.nonlinear_performance_bootstrap import (
    clear_nonlinear_assembly_cache,
    get_nonlinear_assembly_plan,
    install_nonlinear_performance_optimizations,
)
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_v2d_element import (
    FORMULATION_ID as S3_V2D_FORMULATION_ID,
    NativeParityE4PLS3V2DShellElement,
)
from anyfesolver.elements import LegacyShellElement
from anyfesolver.vectorized_nonlinear import shell_nonlinear_batch_eligible


def _shell_section(name: str = "laminate") -> GeneralizedShellSection:
    return GeneralizedShellSection(
        A=np.diag([1.2e8, 8.0e7, 3.0e7]),
        B=np.diag([2.0e4, 1.0e4, 5.0e3]),
        D=np.diag([1.2e5, 8.0e4, 3.0e4]),
        As=np.diag([2.0e7, 1.5e7]),
        name=name,
        mass_per_area=12.0,
        rotary_inertia_per_area=0.03,
    )


def _beam_stiffness() -> np.ndarray:
    stiffness = np.diag([2.0e8, 4.0e7, 3.0e7, 1.0e6, 2.0e6, 3.0e6])
    stiffness[0, 3] = stiffness[3, 0] = 2.0e5
    stiffness[4, 5] = stiffness[5, 4] = 1.0e5
    return stiffness


class _ExternalShellSection:
    name = "external_shell"
    A = np.diag([4.0, 3.0, 2.0])
    B = np.zeros((3, 3))
    D = np.diag([0.4, 0.3, 0.2])
    As = np.diag([1.0, 0.8])
    mass_per_area = None
    rotary_inertia_per_area = None


class _ExternalBeamSection:
    name = "external_beam"

    def generalized_stiffness_matrix(self) -> np.ndarray:
        return _beam_stiffness()

    def generalized_mass_matrix_per_length(self) -> np.ndarray:
        return np.diag([10.0, 10.0, 10.0, 0.2, 0.3, 0.4])


def _generated_geometry() -> dict:
    return {
        "nodes": [
            {"id": 1, "coords": [0.0, 0.0, 0.0]},
            {"id": 2, "coords": [1.0, 0.0, 0.0]},
            {"id": 3, "coords": [1.0, 1.0, 0.0]},
            {"id": 4, "coords": [0.0, 1.0, 0.0]},
            {"id": 10, "coords": [0.0, 0.0, 1.0]},
            {"id": 11, "coords": [1.0, 0.0, 1.0]},
        ],
        "shell_sections": {"laminate": _shell_section().to_dict()},
        "beam_sections": {
            "coupled": {
                "stiffness": _beam_stiffness().tolist(),
                "mass_per_length": np.diag(
                    [10.0, 10.0, 10.0, 0.2, 0.3, 0.4]
                ).tolist(),
            }
        },
        "shells": [
            {
                "id": 1,
                "node_ids": [1, 2, 3, 4],
                "thickness": 0.02,
                "shell_section": "laminate",
                # Authored surface authority; never inferred from connectivity.
                "reference_normal": [0.0, 0.0, 1.0],
            }
        ],
        "beams": [
            {
                "id": 2,
                "node_ids": [10, 11],
                "cross_section": {
                    "area": 0.01,
                    "Iy": 1.0e-6,
                    "Iz": 1.0e-6,
                    "J": 1.0e-6,
                },
                "generalized_section": "coupled",
            }
        ],
    }


def test_public_section_protocols_accept_external_objects_structurally() -> None:
    shell = _ExternalShellSection()
    beam = _ExternalBeamSection()

    assert isinstance(shell, GeneralizedShellSectionProtocol)
    assert isinstance(beam, GeneralizedBeamSectionContract)
    assert validate_generalized_shell_section(shell).name == "external_shell"
    assert validate_generalized_beam_section(beam) is beam
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


def test_generated_geometry_resolves_named_shell_and_beam_sections() -> None:
    model = build_fe_model_from_generated_geometry(_generated_geometry())
    shell = model.mesh.get_element(1)
    beam = model.mesh.get_element(2)

    assert shell.shell_section.name == "laminate"
    assert shell.shell_section.mass_per_area == pytest.approx(12.0)
    assert type(shell) is QualifiedE4PLShellElement
    np.testing.assert_array_equal(shell.reference_normal, (0.0, 0.0, 1.0))
    assert beam.generalized_section.name == "coupled"
    np.testing.assert_allclose(
        beam.generalized_section.generalized_stiffness_matrix(),
        _beam_stiffness(),
    )


def test_generated_geometry_routes_explicit_qualified_s3_with_director_authority() -> None:
    geometry = _generated_geometry()
    geometry["shells"] = [
        {
            "id": 1,
            "node_ids": [1, 2, 4],
            "thickness": 0.02,
            "shell_section": "laminate",
            "shell_formulation": "e4-pl-s3-v2d",
            "formulation_id": S3_V2D_FORMULATION_ID,
            "reference_normal": [0.0, 0.0, 1.0],
            "owner_normal_authority": "PHYSICAL_SURFACE_OWNER_NORMAL_V2D_V1",
        }
    ]

    model = build_fe_model_from_generated_geometry(geometry)
    shell = model.mesh.get_element(1)
    assert type(shell) is NativeParityE4PLS3V2DShellElement
    assert shell.shell_section.name == "laminate"
    np.testing.assert_array_equal(shell.reference_normal, (0.0, 0.0, 1.0))


def test_generated_geometry_requires_v2d_authority_and_preserves_explicit_legacy_s3() -> None:
    geometry = _generated_geometry()
    geometry["shells"] = [
        {
            "id": 1,
            "node_ids": [1, 2, 4],
            "thickness": 0.02,
        }
    ]
    with pytest.raises(
        ValueError,
        match="qualified-s3-generated-record-requires-formulation-id",
    ):
        build_fe_model_from_generated_geometry(geometry)

    geometry["shells"][0]["formulation"] = "legacy-s3"
    model = build_fe_model_from_generated_geometry(geometry)
    assert type(model.mesh.get_element(1)) is LegacyShellElement

    geometry["shells"][0].update(
        {
            "formulation": "legacy-s3",
            "reference_normal": [0.0, 0.0, 1.0],
        }
    )
    with pytest.raises(
        ValueError,
        match="shell-director-authority-requires-qualified-shell-formulation",
    ):
        build_fe_model_from_generated_geometry(geometry)


@pytest.mark.parametrize(
    ("element_key", "section_key", "message"),
    [
        ("shells", "shell_section", "unknown-generalized-shell-section"),
        ("beams", "generalized_section", "unknown-generalized-beam-section"),
    ],
)
def test_generated_geometry_rejects_unknown_section_references(
    element_key: str,
    section_key: str,
    message: str,
) -> None:
    geometry = _generated_geometry()
    geometry[element_key][0][section_key] = "missing"

    with pytest.raises(ValueError, match=message):
        build_fe_model_from_generated_geometry(geometry)


def test_generalized_shell_uses_installed_extended_batch_diagnostic() -> None:
    model = build_fe_model_from_generated_geometry(_generated_geometry())
    shell = model.mesh.get_element(1)

    assert shell_nonlinear_batch_eligible(shell) is False
    install_nonlinear_performance_optimizations()
    clear_nonlinear_assembly_cache(model)
    diagnostic = get_nonlinear_assembly_plan(model, num_layers=3).diagnostics()
    assert diagnostic["constitutive_fallback"] is None
    assert diagnostic["generalized_elastic_fast_path_element_count"] == 1


def test_generated_prestress_uses_exact_section_resultants() -> None:
    model = build_fe_model_from_generated_geometry(_generated_geometry())
    shell = model.mesh.get_element(1)
    beam = model.mesh.get_element(2)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in model.mesh.nodes.values():
        displacement[node.dofs[0]] = 1.0e-4 * node.x

    shell_local = displacement[np.asarray(shell.get_dof_mapping(model.mesh))]
    shell_recovery = shell.compute_stresses(
        model.mesh,
        shell_local,
        model.get_material(shell.material_name),
    )
    beam_local = displacement[np.asarray(beam.get_dof_mapping(model.mesh))]
    beam_recovery = beam.compute_stresses(
        model.mesh,
        beam_local,
        model.get_material(beam.material_name),
    )
    states, summary = recover_prestress_from_static_result(model, displacement)

    np.testing.assert_allclose(
        states[1]["membrane_forces_at_gauss"],
        shell_recovery["membrane_resultants"],
    )
    np.testing.assert_allclose(
        states[1]["bending_moments_at_gauss"],
        shell_recovery["bending_resultants"],
    )
    beam_resultants = np.asarray(beam_recovery["generalized_resultant"])
    assert states[2]["axial_force"] == pytest.approx(
        float(
            beam_resultants[0]
            if beam_resultants.ndim == 1
            else np.mean(beam_resultants[:, 0])
        )
    )
    assert summary["shell_elements"] == 1
    assert summary["beam_elements"] == 1
