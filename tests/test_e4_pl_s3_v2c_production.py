from __future__ import annotations

import numpy as np
import pytest

import anyfesolver
from anyfesolver.e4_pl_s3_v2b_element import StrictFlatLinearE4PLS3V2BShellElement
from anyfesolver.e4_pl_s3_v2c_element import (
    FORMULATION_ID,
    GEOMETRIC_STIFFNESS_POLICY_ID,
    MASS_POLICY_ID,
    SERIALIZATION_POLICY_ID,
    StrictFlatLinearCapabilityError,
    StrictFlatLinearE4PLS3V2CShellElement,
)
from anyfesolver.elements import (
    DEFAULT_Q4_FORMULATION,
    DEFAULT_S3_FORMULATION,
    create_shell_element,
    shell_element_from_dict,
)
from anyfesolver.fe_core import FEModel, FEMesh, Material
from anyfesolver.boundary import BoundaryCondition, FixedSupport
from anyfesolver.modal import solve_free_vibration


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


def _candidate() -> StrictFlatLinearE4PLS3V2CShellElement:
    return create_shell_element(
        1,
        [1, 2, 3],
        "steel",
        formulation="e4-pl-s3-v2c",
        thickness=THICKNESS,
        reference_normal=NORMAL,
    )


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, ord=np.inf) / max(float(np.linalg.norm(right, ord=np.inf)), 1.0))


def test_v2c_is_additive_and_preserves_both_defaults() -> None:
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "legacy-s3"
    assert type(_candidate()) is StrictFlatLinearE4PLS3V2CShellElement
    assert type(
        create_shell_element(
            2,
            [1, 2, 3],
            "steel",
            formulation="qualified-s3-v2c",
            thickness=THICKNESS,
            reference_normal=NORMAL,
        )
    ) is StrictFlatLinearE4PLS3V2CShellElement
    assert anyfesolver.STRICT_FLAT_S3_V2C_FORMULATION_ID == FORMULATION_ID
    for spelling in ("e4_pl_s3_v2c", "qualified_s3_v2c"):
        with pytest.raises(ValueError, match="canonical"):
            create_shell_element(3, [1, 2, 3], "steel", formulation=spelling)
    with pytest.raises(StrictFlatLinearCapabilityError, match="director_polarity=1"):
        create_shell_element(
            4,
            [1, 2, 3],
            "steel",
            formulation="e4-pl-s3-v2c",
            thickness=THICKNESS,
            reference_normal=NORMAL,
            director_polarity=-1,
        )


def test_v2c_static_stiffness_is_identical_to_v2b() -> None:
    mesh = _mesh()
    material = _material()
    v2b = StrictFlatLinearE4PLS3V2BShellElement(
        1,
        (1, 2, 3),
        "steel",
        thickness=THICKNESS,
        reference_normal=NORMAL,
    )
    v2c = _candidate()
    first = v2b.compute_stiffness_components(mesh, material)
    second = v2c.compute_stiffness_components(mesh, material)
    for name in ("membrane", "bending", "shear", "physical", "pl", "total"):
        np.testing.assert_array_equal(first[name], second[name])
    assert first["phi_squared"] == second["phi_squared"]


def test_source_selected_lumped_mass_has_only_translational_inertia() -> None:
    mesh = _mesh()
    mass = _candidate().compute_mass_matrix(mesh, _material())
    area = 1.1
    expected = 7850.0 * THICKNESS * area / 3.0
    for node in range(3):
        np.testing.assert_allclose(
            np.diag(mass)[6 * node : 6 * node + 3],
            expected,
            rtol=3.0e-16,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            np.diag(mass)[6 * node + 3 : 6 * node + 6],
            np.zeros(3),
        )
    assert np.linalg.matrix_rank(mass) == 9
    assert MASS_POLICY_ID == "MYSTRAN_TRIA3_LUMPED_TRANSLATIONAL_MASS_V1"


def test_v2c_mass_components_drive_descriptor_modal_without_invented_inertia() -> None:
    model = FEModel("v2c-descriptor-modal")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(COORDINATES, start=1):
        model.add_node(node_id, *coordinate)
    candidate = _candidate()
    model.add_element(1, candidate)
    model.add_boundary_condition(FixedSupport("fixed-edge", [1, 2]))
    model.add_boundary_condition(
        BoundaryCondition("node-3-in-plane", [3], {"ux": 0.0, "uy": 0.0})
    )
    material = model.get_material("steel")
    components = candidate.compute_mass_components(model.mesh, material)
    directions = candidate.dynamic_algebraic_directions(model.mesh, material)
    assert components["condensed_rank"] == 9
    assert components["rotary_inertia_per_area"] == 0.0
    assert components["zero_rotational_inertia"] is True
    assert directions.shape == (18, 9)
    np.testing.assert_array_equal(
        components["global"] @ directions,
        np.zeros((18, 9)),
    )
    result = solve_free_vibration(model, num_modes=1)
    assert result.solver_status == "ok"
    assert result.diagnostics["descriptor_modal"] is True
    assert result.diagnostics["declared_algebraic_element_ids"] == [1]
    assert result.diagnostics["declared_algebraic_mass_certificate"][
        "compatible_global_nullity"
    ] == 3


def test_source_selected_membrane_stress_stiffness() -> None:
    mesh = _mesh()
    candidate = _candidate()
    state = {
        "membrane_compression": [2.0, 3.0, 0.25],
        "bending_compression": [0.0, 0.0, 0.0],
        "stress_second_moment": [0.0, 0.0, 0.0],
    }
    made = candidate.compute_geometric_stiffness_matrix(mesh, _material(), state)
    geometry = candidate._geometry(mesh)
    gradients = np.asarray(geometry["shape_gradients"])
    stress = np.asarray(((2.0, 0.25), (0.25, 3.0)))
    scalar = geometry["area"] * gradients @ stress @ gradients.T
    expected = np.zeros((18, 18))
    for first in range(3):
        for second in range(3):
            for axis in range(3):
                expected[6 * first + axis, 6 * second + axis] = scalar[first, second]
    assert _relative(made, expected) <= 3.0e-15
    assert np.linalg.matrix_rank(made) == 6
    assert GEOMETRIC_STIFFNESS_POLICY_ID.startswith("CST_MEMBRANE")
    for unsupported in ("bending_compression", "stress_second_moment"):
        changed = dict(state)
        changed[unsupported] = [1.0, 0.0, 0.0]
        with pytest.raises(StrictFlatLinearCapabilityError, match=unsupported):
            candidate.compute_geometric_stiffness_matrix(
                mesh,
                _material(),
                changed,
            )


def test_recovery_and_stateless_serialization_are_formulation_native() -> None:
    mesh = _mesh()
    candidate = _candidate()
    displacement = np.arange(18, dtype=np.float64) / 8192.0
    recovered = candidate.compute_stresses(mesh, displacement, _material())
    assert recovered["formulation_id"] == FORMULATION_ID
    assert recovered["recovery_scope"] == "PHYSICAL_LOCAL_RESULTANTS_ONLY"
    assert recovered["numerical_fields_excluded"] is True
    payload = candidate.to_dict()
    assert payload["serialization_policy_id"] == SERIALIZATION_POLICY_ID
    restored = shell_element_from_dict(payload)
    assert type(restored) is StrictFlatLinearE4PLS3V2CShellElement
    assert restored.to_dict() == payload
    changed = dict(payload)
    changed["mass_policy_id"] = "BAD"
    with pytest.raises(ValueError, match="fingerprint"):
        shell_element_from_dict(changed)
    with pytest.raises(StrictFlatLinearCapabilityError, match="restart"):
        candidate.__getstate__()
