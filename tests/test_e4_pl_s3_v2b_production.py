from __future__ import annotations

import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

import anyfesolver
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_v2b_element import (
    CBMIN3,
    FORMULATION_ID,
    IMPLEMENTATION_ID,
    RELAXATION_AUTHORITY_SHA256,
    SELECTOR,
    StrictFlatLinearCapabilityError,
    StrictFlatLinearE4PLS3V2BShellElement,
)
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import (
    DEFAULT_Q4_FORMULATION,
    DEFAULT_S3_FORMULATION,
    ShellElement,
    create_element,
    create_shell_element,
    shell_formulation_diagnostics,
)
from anyfesolver.fe_core import FEMesh, Material


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

import e4_pl_s3_v5b_relaxed_screen_producer as accepted_reference


COORDINATES = np.asarray(
    ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.3, 1.1, 0.0)),
    dtype=np.float64,
)
NORMAL = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
THICKNESS = 0.08


def _mesh() -> FEMesh:
    mesh = FEMesh()
    for node_id, coordinate in enumerate(COORDINATES, start=1):
        mesh.add_node(node_id, *coordinate)
    return mesh


def _material() -> Material:
    return Material("steel", 210.0e9, 0.3, density=7850.0)


def _element(
    order: tuple[int, int, int] = (0, 1, 2),
    *,
    normal: np.ndarray = NORMAL,
) -> StrictFlatLinearE4PLS3V2BShellElement:
    return StrictFlatLinearE4PLS3V2BShellElement(
        1,
        tuple(index + 1 for index in order),
        "steel",
        thickness=THICKNESS,
        reference_normal=normal,
    )


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right), ord=np.inf)
        / max(float(np.linalg.norm(np.asarray(right), ord=np.inf)), 1.0)
    )


def test_v2b_is_additive_opt_in_and_defaults_are_unchanged() -> None:
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "legacy-s3"
    assert type(create_shell_element(1, [1, 2, 3], "steel")) is ShellElement
    assert type(create_shell_element(2, [1, 2, 3, 4], "steel")) is QualifiedE4PLShellElement

    made = create_shell_element(
        3,
        [1, 2, 3],
        "steel",
        formulation="e4-pl-s3-v2b",
        thickness=THICKNESS,
        reference_normal=NORMAL,
    )
    alias = create_element(
        "qualified-s3-v2b",
        4,
        [1, 2, 3],
        "steel",
        thickness=THICKNESS,
        reference_normal=NORMAL,
    )
    assert type(made) is type(alias) is StrictFlatLinearE4PLS3V2BShellElement
    diagnostic = shell_formulation_diagnostics(node_count=3, formulation="e4-pl-s3-v2b")
    assert diagnostic["selected_formulation"] == "e4-pl-s3-v2b"
    assert diagnostic["production_default"] is False
    assert diagnostic["topology_policy"] == "STRICT_FLAT_LINEAR_E4_PL_S3_V2B_OPT_IN"
    for spelling in ("e4_pl_s3_v2b", "qualified_s3_v2b"):
        with pytest.raises(ValueError, match="canonical"):
            create_shell_element(
                5,
                [1, 2, 3],
                "steel",
                formulation=spelling,
                reference_normal=NORMAL,
            )


def test_v2b_authority_identity_is_explicit() -> None:
    assert anyfesolver.StrictFlatLinearE4PLS3V2BShellElement is StrictFlatLinearE4PLS3V2BShellElement
    assert anyfesolver.STRICT_FLAT_S3_V2B_FORMULATION_ID == FORMULATION_ID
    assert anyfesolver.STRICT_FLAT_S3_V2B_IMPLEMENTATION_ID == IMPLEMENTATION_ID
    assert anyfesolver.STRICT_FLAT_S3_V2B_SELECTOR == SELECTOR
    assert FORMULATION_ID == "CANDIDATE_E4_PL_S3_V2B_FLAT_LINEAR_V1"
    assert SELECTOR == "e4-pl-s3-v2b"
    assert CBMIN3 == 2.0
    assert RELAXATION_AUTHORITY_SHA256 == (
        "0AE9DAA05B63A43D456423BCDC676E7421AB3583F152EE5DB3D0E36FE60A17A0"
    )


def test_components_match_accepted_v5b_reference_exactly() -> None:
    mesh = _mesh()
    element = _element()
    made = element.compute_stiffness_components(mesh, _material())
    expected = accepted_reference.min3_components(COORDINATES, thickness=THICKNESS)
    for name in ("membrane", "bending", "shear", "physical", "pl", "total"):
        assert _relative(made[name], expected[name]) <= 3.0e-13
    assert made["phi_squared"] == expected["phi_squared"]
    assert made["relaxation_authority_sha256"] == RELAXATION_AUTHORITY_SHA256
    assert 0.0 < made["phi_squared"] <= 1.0


def test_all_d3_numberings_match_the_accepted_reference() -> None:
    material = _material()
    mesh = _mesh()
    for order in itertools.permutations(range(3)):
        element = _element(order)
        made = element.compute_stiffness_components(mesh, material)
        expected = accepted_reference.min3_components(
            COORDINATES[np.asarray(order)],
            thickness=THICKNESS,
        )
        assert _relative(made["total"], expected["total"]) <= 3.0e-13
        assert made["phi_squared"] == pytest.approx(
            expected["phi_squared"], rel=3.0e-15, abs=0.0
        )


def test_relaxed_resultants_are_work_conjugate() -> None:
    mesh = _mesh()
    material = _material()
    element = _element()
    displacement = np.arange(1.0, 19.0, dtype=np.float64) / 8192.0
    components = element.compute_stiffness_components(mesh, material)
    resultants = element.compute_variational_resultants(mesh, displacement, material)
    shear_energy = float(displacement @ components["shear"] @ displacement)
    recovered_energy = float(
        np.sum(
            np.asarray(resultants["physical_weights"])[:, None]
            * np.asarray(resultants["transverse_shear_strain"])
            * np.asarray(resultants["transverse_shear_resultants"])
        )
    )
    assert recovered_energy == pytest.approx(shear_energy, rel=3.0e-13, abs=1.0e-8)
    assert resultants["min3_relaxation_phi_squared"] == components["phi_squared"]
    np.testing.assert_allclose(
        element.compute_internal_forces(mesh, displacement, material),
        components["total"] @ displacement,
        rtol=0.0,
        atol=0.0,
    )


def test_uniform_pressure_matches_independent_source_work() -> None:
    mesh = _mesh()
    actual = _element().compute_dead_transverse_pressure_load(mesh, 1000.0)
    expected = accepted_reference.v5a._pressure_load(COORDINATES, 1000.0)
    assert _relative(actual, expected) <= 3.0e-15


def test_v2b_remains_fail_closed_outside_flat_linear_scope() -> None:
    mesh = _mesh()
    element = _element()
    with pytest.raises(StrictFlatLinearCapabilityError, match="consistent_mass"):
        element.compute_mass_matrix(mesh, _material())
    with pytest.raises(StrictFlatLinearCapabilityError, match="serialization"):
        element.to_dict()
    with pytest.raises(ElementCapabilityError, match="legacy ShellElement"):
        ShellElement.compute_stiffness_matrix(element, mesh, _material())
