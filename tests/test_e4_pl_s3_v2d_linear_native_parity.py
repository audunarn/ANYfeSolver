from __future__ import annotations

import numpy as np
import pytest

import anyfesolver
from anyfesolver.e4_pl_s3_v2c_element import (
    StrictFlatLinearE4PLS3V2CShellElement,
)
from anyfesolver.e4_pl_s3_v2d_element import (
    FORMULATION_ID,
    NativeParityCapabilityError,
    NativeParityE4PLS3V2DShellElement,
)
from anyfesolver.elements import (
    DEFAULT_Q4_FORMULATION,
    DEFAULT_S3_FORMULATION,
    create_shell_element,
    shell_element_from_dict,
    shell_formulation_diagnostics,
)
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver.shell_sections import GeneralizedShellSection


COORDINATES = np.asarray(
    ((0.2, -0.1, 0.0), (2.3, 0.2, 0.0), (0.4, 1.4, 0.0)),
    dtype=np.float64,
)
NORMAL = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
THICKNESS = 0.07
E = 210.0e9
NU = 0.3


def _mesh() -> FEMesh:
    mesh = FEMesh()
    for node_id, coordinate in enumerate(COORDINATES, start=1):
        mesh.add_node(node_id, *coordinate)
    return mesh


def _material() -> Material:
    return Material("steel", E, NU, density=7850.0)


def _v2d(**kwargs: object) -> NativeParityE4PLS3V2DShellElement:
    return create_shell_element(
        1,
        [1, 2, 3],
        "steel",
        formulation="e4-pl-s3-v2d",
        thickness=THICKNESS,
        reference_normal=NORMAL,
        **kwargs,
    )


def test_v2d_elastic_surface_stress_is_available_through_default_public_recovery() -> None:
    from anyfesolver import FEModel, recover_stress_result

    model = FEModel("mixed-q4-s3-surface-stress")
    model.add_material("steel", E, NU, density=7850.0)
    coordinates = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (1.0, 1.0, 0.0),
        4: (0.0, 1.0, 0.0),
        5: (2.0, 0.0, 0.0),
        6: (2.0, 1.0, 0.0),
    }
    for node_id, coordinate in coordinates.items():
        model.add_node(node_id, *coordinate)
    model.add_element(
        1,
        create_shell_element(1, [1, 2, 3, 4], "steel", thickness=THICKNESS),
    )
    model.add_element(
        2,
        create_shell_element(
            2,
            [2, 5, 6],
            "steel",
            formulation="e4-pl-s3-v2d",
            thickness=THICKNESS,
            reference_normal=NORMAL,
        ),
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    axial_strain = 1.0e-4
    for node_id, (x, _y, _z) in coordinates.items():
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        displacement[dofs[0]] = axial_strain * x

    recovered = recover_stress_result(model, displacement).element_stresses
    expected_xx = E * axial_strain / (1.0 - NU**2)
    for element_id, expected_stations in ((1, 4), (2, 3)):
        values = recovered[element_id]
        assert values["physical_stress_available"] is True
        assert values["equivalent_stress_measure"] == "von_mises"
        assert values["membrane_xx"].shape == (expected_stations,)
        np.testing.assert_allclose(values["membrane_xx"], expected_xx, rtol=1.0e-11)
        np.testing.assert_allclose(values["membrane_yy"], NU * expected_xx, rtol=1.0e-11)
        np.testing.assert_allclose(values["von_mises"], values["equivalent_stress"])
        np.testing.assert_allclose(
            values["von_mises"],
            expected_xx * np.sqrt(1.0 - NU + NU**2),
            rtol=1.0e-11,
        )


def test_v2d_surface_recovery_reuses_local_stress_and_refuses_section_stress() -> None:
    mesh = _mesh()
    material = _material()
    displacement = np.arange(18, dtype=np.float64) / 4096.0
    element = _v2d()
    local = element.compute_stresses(mesh, displacement, material)
    global_values = element.compute_stresses(
        mesh, displacement, material, return_global=True
    )
    for component in (
        "membrane_xx",
        "membrane_yy",
        "membrane_xy",
        "bending_xx",
        "bending_yy",
        "bending_xy",
        "shear_xz",
        "shear_yz",
        "von_mises",
        "equivalent_stress",
    ):
        np.testing.assert_array_equal(local[component], global_values[component])
    np.testing.assert_allclose(
        global_values["local_xx_top"],
        local["membrane_xx"] + local["bending_xx"],
    )
    assert np.max(np.abs(local["bending_xx"])) > 0.0
    assert np.max(np.abs(local["shear_xz"])) > 0.0
    assert np.max(np.abs(global_values["global_xx_top"] - global_values["local_xx_top"])) > 0.0
    surface_mises = []
    for surface in ("top", "bot"):
        xx = global_values[f"local_xx_{surface}"]
        yy = global_values[f"local_yy_{surface}"]
        xy = global_values[f"local_xy_{surface}"]
        xz = global_values[f"local_xz_{surface}"]
        yz = global_values[f"local_yz_{surface}"]
        surface_mises.append(
            np.sqrt(xx**2 - xx * yy + yy**2 + 3.0 * (xy**2 + xz**2 + yz**2))
        )
    np.testing.assert_allclose(local["von_mises"], np.maximum(*surface_mises))
    local["membrane_xx"][0] = -1.0
    assert global_values["membrane_xx"][0] != -1.0
    assert local["membrane_resultants"][0, 0] != -THICKNESS

    generalized = _v2d(shell_section=_homogeneous_section())
    for return_global in (False, True):
        result = generalized.compute_stresses(
            mesh, displacement, material, return_global=return_global
        )
        assert result["physical_stress_available"] is False
        assert result["physical_layer_recovery_available"] is False
        assert "von_mises" not in result
        assert "membrane_xx" not in result


def _homogeneous_section(*, coupling: np.ndarray | None = None) -> GeneralizedShellSection:
    scale = E / (1.0 - NU**2)
    plane = scale * np.asarray(
        ((1.0, NU, 0.0), (NU, 1.0, 0.0), (0.0, 0.0, 0.5 * (1.0 - NU)))
    )
    shear = (5.0 / 6.0) * E * THICKNESS / (2.0 * (1.0 + NU))
    return GeneralizedShellSection(
        A=THICKNESS * plane,
        B=np.zeros((3, 3)) if coupling is None else coupling,
        D=THICKNESS**3 / 12.0 * plane,
        As=shear * np.eye(2),
        name="native-v2d-test",
        mass_per_area=7850.0 * THICKNESS,
    )


def test_v2d_is_the_s3_default_and_preserves_the_q4_default() -> None:
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "e4-pl-s3-v2d"
    assert type(
        create_shell_element(
            1,
            [1, 2, 3],
            "steel",
            thickness=THICKNESS,
            reference_normal=NORMAL,
        )
    ) is NativeParityE4PLS3V2DShellElement
    assert type(_v2d()) is NativeParityE4PLS3V2DShellElement
    assert type(
        create_shell_element(
            2,
            [1, 2, 3],
            "steel",
            formulation="qualified-s3-v2d",
            thickness=THICKNESS,
            reference_normal=NORMAL,
        )
    ) is NativeParityE4PLS3V2DShellElement
    assert anyfesolver.S3_V2D_FORMULATION_ID == FORMULATION_ID
    assert shell_formulation_diagnostics(
        node_count=3, formulation="e4-pl-s3-v2d"
    )["topology_policy"] == "NATIVE_PARITY_E4_PL_S3_V2D_DEFAULT"
    for spelling in ("e4_pl_s3_v2d", "qualified_s3_v2d"):
        with pytest.raises(ValueError, match="canonical"):
            create_shell_element(
                3,
                [1, 2, 3],
                "steel",
                formulation=spelling,
                thickness=THICKNESS,
                reference_normal=NORMAL,
            )


def test_v2d_preserves_v2c_elastic_components_byte_for_byte() -> None:
    mesh = _mesh()
    material = _material()
    v2c = StrictFlatLinearE4PLS3V2CShellElement(
        1,
        (1, 2, 3),
        "steel",
        thickness=THICKNESS,
        reference_normal=NORMAL,
    )
    v2d = _v2d()
    accepted = v2c.compute_stiffness_components(mesh, material)
    successor = v2d.compute_stiffness_components(mesh, material)
    for name in ("membrane", "bending", "shear", "physical", "pl", "total"):
        np.testing.assert_array_equal(successor[name], accepted[name])
    assert successor["phi_squared"] == accepted["phi_squared"]
    np.testing.assert_array_equal(
        v2d.compute_dead_transverse_pressure_load(mesh, 12.5),
        v2c.compute_dead_transverse_pressure_load(mesh, 12.5),
    )
    np.testing.assert_array_equal(
        v2d.compute_mass_matrix(mesh, material),
        v2c.compute_mass_matrix(mesh, material),
    )


def test_native_homogeneous_generalized_section_matches_elastic_operator() -> None:
    mesh = _mesh()
    material = _material()
    elastic = _v2d().compute_stiffness_components(mesh, material)
    generalized = _v2d(shell_section=_homogeneous_section()).compute_stiffness_components(
        mesh, material
    )
    for name in ("membrane", "bending", "shear", "physical", "pl", "total"):
        np.testing.assert_allclose(
            generalized[name], elastic[name], rtol=5.0e-15, atol=2.0e-7
        )
    assert generalized["native_section_policy_id"].startswith("S3_V2D_NATIVE")
    assert np.count_nonzero(generalized["membrane_bending_coupling"]) == 0


def test_native_coupled_section_has_symmetric_work_conjugate_operator() -> None:
    membrane_scale = E * THICKNESS / (1.0 - NU**2)
    coupling = 2.0e-4 * membrane_scale * np.asarray(
        ((0.7, 0.1, 0.0), (0.1, 0.6, 0.0), (0.0, 0.0, 0.2))
    )
    element = _v2d(shell_section=_homogeneous_section(coupling=coupling))
    mesh = _mesh()
    material = _material()
    components = element.compute_stiffness_components(mesh, material)
    np.testing.assert_allclose(
        components["total"], components["total"].T, rtol=0.0, atol=0.0
    )
    assert np.linalg.norm(components["membrane_bending_coupling"], ord=np.inf) > 0.0
    displacement = np.arange(18, dtype=np.float64) / 4096.0
    recovered = element.compute_variational_resultants(mesh, displacement, material)
    station_work = np.sum(
        recovered["membrane_strain"] * recovered["membrane_resultants"], axis=1
    )
    station_work += np.sum(
        recovered["curvature"] * recovered["bending_resultants"], axis=1
    )
    station_work += np.sum(
        recovered["transverse_shear_strain"]
        * recovered["transverse_shear_resultants"],
        axis=1,
    )
    integrated = float(recovered["physical_weights"] @ station_work)
    physical_work = float(displacement @ components["physical"] @ displacement)
    np.testing.assert_allclose(integrated, physical_work, rtol=2.0e-14, atol=1.0e-6)


def test_native_section_operator_is_d3_covariant_with_physical_material_direction() -> None:
    from itertools import permutations

    coupling = np.asarray(
        ((2.1e5, 0.4e5, 0.2e5), (0.1e5, 1.8e5, -0.1e5), (0.3e5, 0.2e5, 0.7e5))
    )
    section = _homogeneous_section(coupling=coupling)
    material = _material()
    mesh = _mesh()
    base = create_shell_element(
        1,
        [1, 2, 3],
        "steel",
        formulation="e4-pl-s3-v2d",
        thickness=THICKNESS,
        reference_normal=NORMAL,
        material_direction=(1.0, 0.35, 0.0),
        shell_section=section,
    ).compute_stiffness_matrix(mesh, material)
    eye = np.eye(6)
    for order in permutations((0, 1, 2)):
        element = create_shell_element(
            1,
            [1 + index for index in order],
            "steel",
            formulation="e4-pl-s3-v2d",
            thickness=THICKNESS,
            reference_normal=NORMAL,
            material_direction=(1.0, 0.35, 0.0),
            shell_section=section,
        )
        made = element.compute_stiffness_matrix(mesh, material)
        permutation = np.zeros((18, 18))
        for external, canonical in enumerate(order):
            permutation[6 * external : 6 * external + 6, 6 * canonical : 6 * canonical + 6] = eye
        restored = permutation.T @ made @ permutation
        np.testing.assert_allclose(restored, base, rtol=2.0e-14, atol=2.0e-6)


def test_v2d_stateless_identity_roundtrip_and_remaining_successor_gaps() -> None:
    import subprocess

    element = _v2d(shell_section=_homogeneous_section())
    payload = element.to_dict()
    restored = shell_element_from_dict(payload)
    assert type(restored) is NativeParityE4PLS3V2DShellElement
    assert restored.to_dict() == payload
    changed = dict(payload, source_selection_sha256="0" * 64)
    with pytest.raises(ValueError, match="fingerprint"):
        shell_element_from_dict(changed)
    v6a_source = subprocess.run(
        [
            "git",
            "show",
            "dfe3d31424fdc97bd770a2657f19bb009bab7989:src/anyfesolver/e4_pl_s3_v2d_element.py",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    assert 'self._unsupported("nonlinear_geometry")' in v6a_source
    with pytest.raises(NativeParityCapabilityError, match="restart"):
        element.__getstate__()
    v6b_source = subprocess.run(
        [
            "git",
            "show",
            "afe691759a611278e341be7d58ffaae4f71bde89:src/anyfesolver/e4_pl_s3_v2d_element.py",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    assert "director reversal is pending" in v6b_source
    assert "reference_surface_offset is pending" in v6b_source
