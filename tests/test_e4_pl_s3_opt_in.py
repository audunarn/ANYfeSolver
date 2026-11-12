from __future__ import annotations

from collections.abc import Mapping
import itertools
import json

import numpy as np
import pytest

import anyfesolver.e4_pl_s3_element as s3_element_module
from anyfesolver import (
    DEFAULT_Q4_FORMULATION,
    DEFAULT_S3_FORMULATION,
    ElementCapabilityError,
    FEModel,
    ImperfectionField,
    LegacyS3MigrationWarning,
    LegacyShellElement,
    NativeParityE4PLS3V2DShellElement,
    QualifiedE4PLS3ShellElement,
    QualifiedE4PLShellElement,
    QualifiedS3MigrationWarning,
    apply_imperfection,
    create_element,
    create_shell_element,
    shell_element_from_dict,
)
from anyfesolver.e4_pl_s3_element import (
    BUBBLE_OFFSET_D,
    CAPABILITY_GAPS,
    FORMULATION_ID,
    MITC3_PLUS_SOURCE_BYTES,
    MITC3_PLUS_SOURCE_SHA256,
    PHYSICAL_EXTERNAL_INDICES,
    S3_QUADRATURE_AUTHORITY_ID,
    TRIANGLE_QUADRATURE,
    TYING_POINTS,
    invariant_drilling_scale,
)
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver.shell_sections import GeneralizedShellSection


def _mesh(nodes: np.ndarray) -> FEMesh:
    mesh = FEMesh()
    for identifier, coordinate in enumerate(nodes, start=1):
        mesh.add_node(identifier, *coordinate)
    return mesh


def _material() -> Material:
    return Material("steel", 210.0e9, 0.3, density=7850.0)


OWNER_NORMAL = np.asarray((0.0, 0.0, 1.0))


def _rigid_modes(nodes: np.ndarray) -> np.ndarray:
    modes = np.zeros((18, 6), dtype=float)
    centre = np.mean(nodes, axis=0)
    for node, coordinate in enumerate(nodes):
        base = 6 * node
        modes[base : base + 3, :3] = np.eye(3)
        for axis in range(3):
            modes[base : base + 3, 3 + axis] = np.cross(
                np.eye(3)[axis], coordinate - centre
            )
            modes[base + 3 : base + 6, 3 + axis] = np.eye(3)[axis]
    return modes


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right, ord=np.inf)
        / max(np.linalg.norm(right, ord=np.inf), 1.0)
    )


def test_published_identity_and_seven_point_rule_are_frozen() -> None:
    assert MITC3_PLUS_SOURCE_BYTES == 1_146_142
    assert MITC3_PLUS_SOURCE_SHA256 == (
        "182F52217277B55E17627B8C41A3A4626ED91ED5378399088E0EA1748AD93EF0"
    )
    assert BUBBLE_OFFSET_D == 1.0e-4
    assert set(TYING_POINTS) == {"A", "B", "C", "D", "E", "F"}
    assert len(TRIANGLE_QUADRATURE) == 7
    assert sum(weight for _r, _s, weight in TRIANGLE_QUADRATURE) == pytest.approx(0.5)


def test_qualified_aliases_and_s3_defaults_dispatch_to_v2d() -> None:
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "e4-pl-s3-v2d"
    assert type(
        create_shell_element(
            1, [1, 2, 3], "steel", reference_normal=OWNER_NORMAL
        )
    ) is NativeParityE4PLS3V2DShellElement
    assert type(
        create_element(
            "s3", 2, [1, 2, 3], "steel", reference_normal=OWNER_NORMAL
        )
    ) is NativeParityE4PLS3V2DShellElement
    assert type(
        create_element(
            "tria3", 3, [1, 2, 3], "steel", reference_normal=OWNER_NORMAL
        )
    ) is NativeParityE4PLS3V2DShellElement
    assert type(
        create_element(
            "tria3",
            31,
            [1, 2, 3],
            "steel",
            formulation="qualified-s3",
            reference_normal=OWNER_NORMAL,
        )
    ) is NativeParityE4PLS3V2DShellElement
    assert type(create_shell_element(4, [1, 2, 3, 4], "steel")) is QualifiedE4PLShellElement
    assert type(
        create_shell_element(
            5,
            [1, 2, 3],
            "steel",
            formulation="e4-pl-s3",
            reference_normal=OWNER_NORMAL,
        )
    ) is NativeParityE4PLS3V2DShellElement
    assert type(
        create_element(
            "qualified-s3", 6, [1, 2, 3], "steel", reference_normal=OWNER_NORMAL
        )
    ) is NativeParityE4PLS3V2DShellElement
    assert type(
        create_shell_element(
            61, [1, 2, 3], "steel", formulation="legacy-s3"
        )
    ) is LegacyShellElement
    with pytest.raises(ValueError, match="only for three-node"):
        create_element("legacy-s3", 7, [1, 2, 3, 4], "steel")
    with pytest.raises(ValueError, match="only for three-node"):
        create_element("e4-pl-s3", 8, [1, 2, 3, 4], "steel")


def test_local_condensation_pl_completion_and_rigid_kernel() -> None:
    nodes = np.asarray(
        ((0.1, 0.2, 0.3), (1.2, 0.1, 0.5), (0.3, 1.0, 0.4)), dtype=float
    )
    mesh = _mesh(nodes)
    element = QualifiedE4PLS3ShellElement(
        1, [1, 2, 3], "steel", thickness=0.2, reference_normal=np.cross(
            nodes[1] - nodes[0], nodes[2] - nodes[0]
        )
    )
    components = element.compute_stiffness_components(mesh, _material())
    assert components["ranks"] == {
        "uncondensed_physical_rank": 11,
        "bubble_rank": 2,
        "condensed_physical_rank": 9,
        "embedded_physical_rank": 9,
        "pl_rank": 3,
        "total_rank": 12,
        "saddle_rank": 17,
        "saddle_inertia": (14, 3, 6),
    }
    assert np.linalg.eigvalsh(components["bubble_block"])[0] > 0.0
    assert np.array_equal(components["hourglass"], np.zeros((18, 18)))
    assert np.array_equal(components["numerical"], components["pl"])
    stiffness = components["total"]
    rigid = _rigid_modes(nodes)
    scale = max(np.linalg.norm(stiffness, ord=np.inf), 1.0)
    assert np.linalg.norm(stiffness @ rigid, ord=np.inf) / scale < 2.0e-14
    assert np.linalg.norm(components["pl"] @ rigid, ord=np.inf) / scale < 2.0e-14

    constraint = components["pl_constraint"]
    gram = components["pl_multiplier_gram"]
    expected = components["k_d"] * constraint.T @ gram @ constraint
    transform = element._local_dof_transform(components["frame"])
    assert _relative(components["pl"], transform.T @ expected @ transform) < 2.0e-15


def test_structural_rank_certificate_is_unit_and_thickness_scaled() -> None:
    material = _material()
    expected = {
        "uncondensed_physical_rank": 11,
        "bubble_rank": 2,
        "condensed_physical_rank": 9,
        "embedded_physical_rank": 9,
        "pl_rank": 3,
        "total_rank": 12,
        "saddle_rank": 17,
        "saddle_inertia": (14, 3, 6),
    }
    shape = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)))
    for length in (1.0e-6, 1.0, 1.0e6):
        element = QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=length * 1.0e-6,
            reference_normal=OWNER_NORMAL,
        )
        components = element.compute_stiffness_components(_mesh(length * shape), material)
        assert components["rank_certificate"] == "DIMENSIONLESS_KINEMATIC_SUBSPACE_V1"
        assert components["ranks"] == expected


@pytest.mark.parametrize("thickness", (0.2, 1.0e-6), ids=("ordinary", "ultrathin"))
def test_all_d3_numberings_are_operator_covariant(thickness: float) -> None:
    nodes = np.asarray(
        ((0.1, 0.2, 0.3), (1.2, 0.1, 0.5), (0.3, 1.0, 0.4)), dtype=float
    )
    material = _material()

    def stiffness(numbered: np.ndarray) -> np.ndarray:
        element = QualifiedE4PLS3ShellElement(
            1, [1, 2, 3], "steel", thickness=thickness, reference_normal=np.cross(
                nodes[1] - nodes[0], nodes[2] - nodes[0]
            )
        )
        return element._compute_stiffness_components(
            _mesh(numbered), material, enforce_positive_winding=False
        )["total"]

    baseline = stiffness(nodes)
    for permutation in itertools.permutations(range(3)):
        actual = stiffness(nodes[list(permutation)])
        dofs = [6 * node + component for node in permutation for component in range(6)]
        expected = baseline[np.ix_(dofs, dofs)]
        assert _relative(actual, expected) < 2.0e-14, permutation


@pytest.mark.parametrize("thickness", (0.2, 1.0e-6), ids=("ordinary", "ultrathin"))
def test_all_d3_numberings_transport_uncondensed_bubble_pl_and_saddle_blocks(
    thickness: float,
) -> None:
    nodes = np.asarray(
        ((0.1, 0.2, 0.3), (1.2, 0.1, 0.5), (0.3, 1.0, 0.4)), dtype=float
    )
    owner = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
    material = _material()

    def components(numbered: np.ndarray):
        element = QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=thickness,
            reference_normal=owner,
        )
        made = element._compute_stiffness_components(
            _mesh(numbered), material, enforce_positive_winding=False
        )
        return element, made

    baseline_element, baseline = components(nodes)
    baseline_transform = baseline_element._local_dof_transform(baseline["frame"])
    physical = PHYSICAL_EXTERNAL_INDICES
    for permutation in itertools.permutations(range(3)):
        numbered_element, numbered = components(nodes[list(permutation)])
        numbered_transform = numbered_element._local_dof_transform(numbered["frame"])
        node_map = np.zeros((18, 18), dtype=float)
        multiplier_map = np.zeros((3, 3), dtype=float)
        for new_node, old_node in enumerate(permutation):
            node_map[6 * new_node : 6 * new_node + 6, 6 * old_node : 6 * old_node + 6] = np.eye(6)
            multiplier_map[new_node, old_node] = 1.0
        local_external_map = numbered_transform @ node_map @ baseline_transform.T
        physical_map = local_external_map[np.ix_(physical, physical)]
        bubble_map = numbered["frame"][:, :2].T @ baseline["frame"][:, :2]

        transport_17 = np.zeros((17, 17), dtype=float)
        transport_17[:15, :15] = physical_map
        transport_17[15:, 15:] = bubble_map
        assert _relative(
            transport_17.T @ numbered["uncondensed_physical"] @ transport_17,
            baseline["uncondensed_physical"],
        ) < 2.0e-14
        assert np.linalg.norm(
            numbered["bubble_map"] @ physical_map
            - bubble_map @ baseline["bubble_map"],
            ord=np.inf,
        ) < 1.0e-13
        assert np.linalg.norm(
            numbered["pl_constraint"] @ local_external_map
            - multiplier_map @ baseline["pl_constraint"],
            ord=np.inf,
        ) < 2.0e-14

        transport_20 = np.zeros((20, 20), dtype=float)
        transport_20[:18, :18] = local_external_map
        transport_20[18:, 18:] = bubble_map
        transport_23 = np.zeros((23, 23), dtype=float)
        transport_23[:20, :20] = transport_20
        transport_23[20:, 20:] = multiplier_map
        assert _relative(
            transport_23.T @ numbered["full_saddle"] @ transport_23,
            baseline["full_saddle"],
        ) < 2.0e-14


def test_all_d3_numberings_share_the_anisotropic_bubble_accuracy_disposition() -> None:
    """The fail-closed solve certificate is a physical, not basis, decision."""

    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (0.2, 0.9, 0.0)),
        dtype=float,
    )
    material = Material("carrier", 1.0, 0.2, density=1.0)

    def section(shear_ratio: float) -> GeneralizedShellSection:
        return GeneralizedShellSection(
            name=f"D3-near-bubble-bound-{shear_ratio:g}",
            A=np.eye(3),
            B=np.zeros((3, 3)),
            D=1.0e-12 * np.eye(3),
            As=np.diag((1.0, shear_ratio)),
            mass_per_area=1.0,
            rotary_inertia_per_area=1.0,
        )

    passing_conditions = []
    passing_bounds = []
    for permutation in itertools.permutations(range(3)):
        numbered = nodes[list(permutation)]
        element = QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "carrier",
            thickness=1.0,
            shell_section=section(600_000.0),
            material_direction=(1.0, 0.0, 0.0),
            reference_normal=(0.0, 0.0, 1.0),
        )
        diagnostics = element._compute_stiffness_components(
            _mesh(numbered), material, enforce_positive_winding=False
        )["floating_matrix_diagnostics"]
        passing_conditions.append(diagnostics["bubble_condition_2"])
        passing_bounds.append(diagnostics["bubble_forward_error_bound"])

    np.testing.assert_allclose(
        passing_conditions,
        np.full(6, passing_conditions[0]),
        rtol=2.0e-10,
        atol=0.0,
    )
    assert max(passing_bounds) < 1.0e-8

    for permutation in itertools.permutations(range(3)):
        numbered = nodes[list(permutation)]
        element = QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "carrier",
            thickness=1.0,
            shell_section=section(800_000.0),
            material_direction=(1.0, 0.0, 0.0),
            reference_normal=(0.0, 0.0, 1.0),
        )
        with pytest.raises(ValueError, match="uncertified forward accuracy"):
            element._compute_stiffness_components(
                _mesh(numbered), material, enforce_positive_winding=False
            )


def test_b_coupled_section_requires_and_uses_authoritative_normal() -> None:
    nodes = np.asarray(
        ((0.1, 0.2, 0.3), (1.2, 0.1, 0.5), (0.3, 1.0, 0.4)), dtype=float
    )
    normal = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
    normal /= np.linalg.norm(normal)
    section = GeneralizedShellSection(
        A=np.asarray(((120.0, 12.0, 4.0), (12.0, 95.0, -3.0), (4.0, -3.0, 42.0))),
        B=np.asarray(((1.2, 0.1, 0.0), (0.1, -0.8, 0.1), (0.0, 0.1, 0.3))),
        D=np.asarray(((15.0, 1.0, 0.2), (1.0, 11.0, -0.1), (0.2, -0.1, 5.0))),
        As=np.asarray(((25.0, 2.0), (2.0, 20.0))),
    )
    with pytest.raises(ValueError, match="authoritative reference_normal"):
        QualifiedE4PLS3ShellElement(1, [1, 2, 3], "steel", shell_section=section)

    def stiffness(numbered: np.ndarray) -> np.ndarray:
        element = QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            shell_section=section,
            material_direction=np.asarray((1.0, 0.2, 0.1)),
            reference_normal=normal,
        )
        return element._compute_stiffness_components(
            _mesh(numbered), _material(), enforce_positive_winding=False
        )["total"]

    baseline = stiffness(nodes)
    for permutation in itertools.permutations(range(3)):
        actual = stiffness(nodes[list(permutation)])
        dofs = [6 * node + component for node in permutation for component in range(6)]
        assert _relative(actual, baseline[np.ix_(dofs, dofs)]) < 2.0e-14


def test_invariant_drilling_scale_matches_isotropy_and_rotated_section() -> None:
    young = 210.0
    poisson = 0.3
    thickness = 0.4
    q11 = young / (1.0 - poisson * poisson)
    q66 = young / (2.0 * (1.0 + poisson))
    membrane = thickness * np.asarray(
        ((q11, poisson * q11, 0.0), (poisson * q11, q11, 0.0), (0.0, 0.0, q66))
    )
    assert invariant_drilling_scale(membrane) == pytest.approx(thickness * q66)

    section = GeneralizedShellSection(
        A=np.asarray(((120.0, 12.0, 4.0), (12.0, 95.0, -3.0), (4.0, -3.0, 42.0))),
        B=np.zeros((3, 3)),
        D=np.diag((15.0, 11.0, 5.0)),
        As=np.diag((25.0, 20.0)),
    )
    baseline = invariant_drilling_scale(section.A)
    for angle in (0.1, 0.37, 1.1):
        assert invariant_drilling_scale(section.rotated(angle).A) == pytest.approx(
            baseline, rel=2.0e-14
        )


def test_native_linear_recovery_excludes_pl_fields() -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)))
    mesh = _mesh(nodes)
    element = QualifiedE4PLS3ShellElement(
        1, [1, 2, 3], "steel", thickness=0.1, reference_normal=OWNER_NORMAL
    )
    displacement = np.linspace(-1.0e-4, 2.0e-4, 18)
    recovery = element.compute_stresses(mesh, displacement, _material())
    assert recovery["recovery_scope"] == "qualified_s3_local_physical_only"
    assert recovery["numerical_fields_excluded"] is True
    assert recovery["membrane_strain"].shape == (7, 3)
    assert recovery["curvature"].shape == (7, 3)
    assert recovery["transverse_shear_strain"].shape == (7, 2)
    top = np.column_stack(
        (
            recovery["membrane_xx"] + recovery["bending_xx"],
            recovery["membrane_yy"] + recovery["bending_yy"],
            recovery["membrane_xy"] + recovery["bending_xy"],
        )
    )
    bottom = np.column_stack(
        (
            recovery["membrane_xx"] - recovery["bending_xx"],
            recovery["membrane_yy"] - recovery["bending_yy"],
            recovery["membrane_xy"] - recovery["bending_xy"],
        )
    )
    shear_squared = recovery["shear_xz"] ** 2 + recovery["shear_yz"] ** 2
    expected_top = np.sqrt(
        top[:, 0] ** 2
        - top[:, 0] * top[:, 1]
        + top[:, 1] ** 2
        + 3.0 * (top[:, 2] ** 2 + shear_squared)
    )
    expected_bottom = np.sqrt(
        bottom[:, 0] ** 2
        - bottom[:, 0] * bottom[:, 1]
        + bottom[:, 1] ** 2
        + 3.0 * (bottom[:, 2] ** 2 + shear_squared)
    )
    assert np.allclose(
        recovery["equivalent_stress"], np.maximum(expected_top, expected_bottom)
    )
    numerical = element.numerical_internal_force(displacement)
    assert np.array_equal(numerical["hourglass"], np.zeros(18))
    assert np.array_equal(numerical["numerical"], numerical["pl"])
    global_recovery = element.compute_stresses(
        mesh,
        displacement,
        _material(),
        return_global=True,
    )
    assert global_recovery["recovery_scope"] == (
        "qualified_s3_local_and_global_physical"
    )
    assert global_recovery["global_membrane_resultant_tensors"].shape == (7, 3, 3)
    assert global_recovery["global_bending_resultant_tensors"].shape == (7, 3, 3)
    assert global_recovery["global_transverse_shear_resultants"].shape == (7, 3)
    for surface in ("top", "bot"):
        for component in ("xx", "yy", "zz", "xy", "yz", "xz"):
            assert global_recovery[f"global_{component}_{surface}"].shape == (7,)


def test_serialization_is_identity_bound_and_legacy_missing_id_stays_legacy() -> None:
    original = QualifiedE4PLS3ShellElement(
        7,
        [1, 2, 3],
        "steel",
        thickness=0.07,
        material_direction=np.asarray((1.0, 0.2, 0.0)),
        material_angle_deg=11.0,
        reference_normal=np.asarray((0.0, 0.0, 1.0)),
    )
    payload = json.loads(json.dumps(original.to_dict()))
    assert payload["formulation_id"] == FORMULATION_ID
    assert payload["quadrature_authority_id"] == S3_QUADRATURE_AUTHORITY_ID
    rebuilt = shell_element_from_dict(payload)
    assert type(rebuilt) is QualifiedE4PLS3ShellElement
    assert rebuilt.to_dict() == payload

    historical = dict(payload)
    for key in (
        "formulation_id",
        "formulation_schema",
        "bubble_convention",
        "quadrature_id",
        "quadrature_authority_id",
        "dynamic_reduction_policy",
        "algebraic_coordinate_policy",
        "geometric_stiffness_policy",
        "mass_moment_id",
        "recovery_policy_id",
        "state_layout_id",
        "director_polarity",
        "director_polarity_policy_id",
        "director_reversal_transform_id",
        "reference_normal",
        "reference_surface_offset",
        "reference_surface_offset_policy_id",
        "reference_surface_strain_transform_id",
        "reference_surface_mass_shift_id",
    ):
        historical.pop(key)
    historical["type"] = "ShellElement"
    with pytest.warns(
        LegacyS3MigrationWarning,
        match="LEGACY_S3_MISSING_FORMULATION_ID",
    ):
        migrated = shell_element_from_dict(historical)
    assert type(migrated) is LegacyShellElement
    for marker in (
        "algebraic_coordinate_policy",
        "bubble_convention",
        "dynamic_reduction_policy",
        "formulation_schema",
        "geometric_stiffness_policy",
        "mass_moment_id",
        "quadrature_id",
        "quadrature_authority_id",
        "reference_surface_mass_shift_id",
        "reference_surface_offset",
        "reference_surface_offset_policy_id",
        "reference_surface_strain_transform_id",
        "recovery_policy_id",
        "state_layout_id",
    ):
        with pytest.raises(
            ValueError,
            match="retains qualified S3 fingerprint markers.*missing formulation_id",
        ):
            shell_element_from_dict(dict(historical, **{marker: "retained"}))

    mutated = dict(payload, formulation_id="E4_PL_QUALIFIED_S3_UNKNOWN")
    with pytest.raises(ValueError, match="unknown serialized"):
        shell_element_from_dict(mutated)
    missing = dict(payload)
    missing.pop("formulation_id")
    with pytest.raises(ValueError, match="missing formulation_id"):
        shell_element_from_dict(missing)
    generic_with_fingerprint = dict(missing, type="ShellElement")
    with pytest.raises(
        ValueError,
        match="retains qualified S3 fingerprint markers.*missing formulation_id",
    ):
        shell_element_from_dict(generic_with_fingerprint)
    missing_mass_identity = dict(payload)
    missing_mass_identity.pop("mass_moment_id")
    with pytest.raises(ValueError, match="mass moment identity"):
        shell_element_from_dict(missing_mass_identity)
    pre_quadrature_authority = dict(payload)
    pre_quadrature_authority.pop("quadrature_authority_id")
    with pytest.warns(
        QualifiedS3MigrationWarning,
        match="immutable exact quadrature authority",
    ):
        migrated_authority = shell_element_from_dict(pre_quadrature_authority)
    assert migrated_authority.to_dict() == payload
    mutated_quadrature_authority = dict(
        payload,
        quadrature_authority_id="MUTATED_TRIANGLE_QUADRATURE",
    )
    with pytest.raises(ValueError, match="quadrature authority"):
        shell_element_from_dict(mutated_quadrature_authority)
    mutated_descriptor = dict(payload, algebraic_coordinate_policy="INVENTED_DRILL_MASS")
    with pytest.raises(ValueError, match="algebraic coordinate policy"):
        shell_element_from_dict(mutated_descriptor)
    mutated_mass = dict(payload, mass_moment_id="UNDERINTEGRATED_DEGREE5")
    with pytest.raises(ValueError, match="mass moment identity"):
        shell_element_from_dict(mutated_mass)
    mutated_geometric = dict(
        payload, geometric_stiffness_policy="NODE_ONLY_INITIAL_STRESS"
    )
    with pytest.raises(ValueError, match="geometric stiffness policy"):
        shell_element_from_dict(mutated_geometric)
    missing_recovery_policy = dict(payload)
    missing_recovery_policy.pop("recovery_policy_id")
    with pytest.raises(ValueError, match="recovery policy"):
        shell_element_from_dict(missing_recovery_policy)
    mutated_recovery_policy = dict(
        payload,
        recovery_policy_id="LEGACY_TRI3_SURFACE_RECOVERY",
    )
    with pytest.raises(ValueError, match="recovery policy"):
        shell_element_from_dict(mutated_recovery_policy)


def test_qualified_s3_connectivity_and_current_serialization_are_exact() -> None:
    with pytest.raises(TypeError, match="exact non-boolean integers"):
        QualifiedE4PLS3ShellElement(
            1,
            [1.9, 2, 3],
            "steel",
            reference_normal=(0.0, 0.0, 1.0),
        )
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        reference_normal=(0.0, 0.0, 1.0),
    )
    with pytest.raises(TypeError, match="exact non-boolean integers"):
        element.node_ids = (1, 2, False)
    with pytest.raises(ValueError, match="material_direction"):
        element.material_direction = (True, 0.0, 0.0)
    with pytest.raises(ValueError, match="reference_normal"):
        element.reference_normal = (0.0, 0.0, 0.0)
    element.reference_normal = (0.0, 0.0, 2.0)
    assert np.array_equal(element.reference_normal, np.asarray((0.0, 0.0, 1.0)))
    for value in (True, "0.02"):
        with pytest.raises(TypeError):
            QualifiedE4PLS3ShellElement(
                1,
                [1, 2, 3],
                "steel",
                thickness=value,
                reference_normal=(0.0, 0.0, 1.0),
            )
    for value in (True, "12.0"):
        with pytest.raises(TypeError, match="material_angle_deg"):
            QualifiedE4PLS3ShellElement(
                1,
                [1, 2, 3],
                "steel",
                material_direction=(1.0, 0.0, 0.0),
                material_angle_deg=value,
                reference_normal=(0.0, 0.0, 1.0),
            )
    with pytest.raises(ValueError, match="material_direction"):
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            material_direction=(True, 0.0, 0.0),
            reference_normal=(0.0, 0.0, 1.0),
        )

    payload = json.loads(json.dumps(element.to_dict()))
    malformed = (
        dict(payload, element_id=1.9),
        dict(payload, node_ids=[1.0, 2, 3]),
        dict(payload, material_name=123),
        dict(payload, thickness=True),
        dict(payload, thickness="0.01"),
        dict(payload, thickness=-0.01),
        dict(payload, material_angle_deg=True),
        dict(payload, director_polarity=1.9),
        dict(payload, reference_surface_offset=0),
        dict(payload, implementation_id="FOREIGN_Q4"),
        dict(payload, pl_stabilization=99.0),
    )
    for record in malformed:
        with pytest.raises(ValueError):
            shell_element_from_dict(record)

    for key in (
        "material_name",
        "thickness",
        "material_direction",
        "material_angle_deg",
    ):
        incomplete = dict(payload)
        incomplete.pop(key)
        with pytest.raises(ValueError, match="keys are incompatible"):
            shell_element_from_dict(incomplete)
    for key, value, message in (
        ("drilling_stabilization", 0.01, "no user drilling coefficient"),
        ("hourglass_stabilization", 0.01, "no hourglass coefficient"),
        ("reduced_integration", True, "seven-point"),
    ):
        with pytest.raises(ValueError, match=message):
            shell_element_from_dict(dict(payload, **{key: value}))

    with pytest.raises(ValueError, match="authoritative reference_normal"):
        element.reference_normal = None
    with pytest.raises(ValueError, match="requires material_direction"):
        element.material_angle_deg = 15.0
    directional = QualifiedE4PLS3ShellElement(
        2,
        [1, 2, 3],
        "steel",
        material_direction=(1.0, 0.0, 0.0),
        material_angle_deg=15.0,
        reference_normal=(0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="requires material_direction"):
        directional.material_direction = None
    with pytest.raises(ValueError, match="noncanonical types"):
        shell_element_from_dict(dict(payload, reference_normal=[0.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="noncanonical types"):
        shell_element_from_dict(
            dict(payload, material_direction=None, material_angle_deg=1.0)
        )


def test_s3_deserialization_and_vector_callbacks_recheck_runtime_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        reference_normal=(0.0, 0.0, 1.0),
    ).to_dict()
    original_asarray = np.asarray
    reached: list[str] = []

    def changed_asarray(*args: object, **kwargs: object) -> np.ndarray:
        reached.append("asarray")
        return original_asarray(*args, **kwargs)

    class MutatingMapping(Mapping[str, object]):
        def __iter__(self):
            monkeypatch.setattr(np, "asarray", changed_asarray)
            return iter(payload)

        def __len__(self) -> int:
            return len(payload)

        def __getitem__(self, key: str) -> object:
            return payload[key]

    with pytest.raises(ValueError, match="numpy.asarray"):
        QualifiedE4PLS3ShellElement.from_dict(MutatingMapping())
    assert reached == []

    monkeypatch.setattr(np, "asarray", original_asarray)
    original_all = np.all

    def changed_all(*args: object, **kwargs: object) -> object:
        reached.append("all")
        return original_all(*args, **kwargs)

    class MutatingVector:
        def __array__(self, dtype=None, copy=None):
            monkeypatch.setattr(np, "all", changed_all)
            return original_asarray((0.0, 0.0, 1.0), dtype=dtype)

    with pytest.raises(ValueError, match="numpy.all"):
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            reference_normal=MutatingVector(),
        )
    assert reached == []


def test_s3_direct_mechanics_rejects_changed_formulation_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        reference_normal=(0.0, 0.0, 1.0),
    )
    reached: list[str] = []

    def changed_frame(*_args: object, **_kwargs: object) -> object:
        reached.append("triangle_frame")
        raise AssertionError("changed triangle frame reached mechanics")

    monkeypatch.setattr(s3_element_module, "triangle_frame", changed_frame)
    with pytest.raises(ValueError, match="triangle_frame"):
        element.compute_stiffness_matrix(
            _mesh(
                np.asarray(
                    (
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        (0.2, 0.9, 0.0),
                    )
                )
            ),
            _material(),
        )
    assert reached == []


def test_s3_direct_callbacks_recheck_authority_before_cached_force_or_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _mesh(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.2, 0.9, 0.0),
            )
        )
    )
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        reference_normal=(0.0, 0.0, 1.0),
    )
    material = _material()
    element.compute_stiffness_matrix(mesh, material)
    original_asarray = np.asarray
    reached: list[str] = []

    def changed_asarray(*args: object, **kwargs: object) -> np.ndarray:
        reached.append("asarray")
        return original_asarray(*args, **kwargs)

    class MutatingDisplacement:
        def __array__(self, dtype=None, copy=None):
            monkeypatch.setattr(np, "asarray", changed_asarray)
            return original_asarray(np.zeros(18), dtype=dtype)

    with pytest.raises(ValueError, match="numpy.asarray"):
        element.numerical_internal_force(MutatingDisplacement())
    assert reached == []

    monkeypatch.setattr(np, "asarray", original_asarray)
    with pytest.raises(ValueError, match="numpy.asarray"):
        element.compute_stresses(mesh, MutatingDisplacement(), material)
    assert reached == []

    monkeypatch.setattr(np, "asarray", original_asarray)
    with pytest.raises(ValueError, match="numpy.asarray"):
        element.compute_nonlinear_response(
            mesh,
            material,
            MutatingDisplacement(),
            native_rotation_trial=None,
        )
    assert reached == []

    monkeypatch.setattr(np, "asarray", original_asarray)
    original_get_node = mesh.get_node

    def changed_frame(*_args: object, **_kwargs: object) -> object:
        reached.append("triangle_frame")
        raise AssertionError("changed triangle frame reached mechanics")

    def mutating_get_node(node_id: int):
        monkeypatch.setattr(s3_element_module, "triangle_frame", changed_frame)
        return original_get_node(node_id)

    monkeypatch.setattr(mesh, "get_node", mutating_get_node)
    with pytest.raises(ValueError, match="triangle_frame"):
        element.compute_stiffness_matrix(mesh, material)
    assert reached == []

def test_legacy_controls_quality_and_unqualified_capabilities_fail_closed() -> None:
    with pytest.raises(ValueError, match="physical material_direction"):
        QualifiedE4PLS3ShellElement(
            1, [1, 2, 3], "steel", material_angle_deg=15.0
        )
    with pytest.raises(ValueError, match="no user drilling coefficient"):
        QualifiedE4PLS3ShellElement(
            1, [1, 2, 3], "steel", drilling_stabilization=1.0e-3
        )
    with pytest.raises(ValueError, match="no hourglass coefficient"):
        QualifiedE4PLS3ShellElement(
            1, [1, 2, 3], "steel", hourglass_stabilization=1.0e-8
        )
    with pytest.raises(ValueError, match="seven-point"):
        QualifiedE4PLS3ShellElement(
            1, [1, 2, 3], "steel", reduced_integration=True
        )

    bad_nodes = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.99, 0.01, 0.0)))
    bad = QualifiedE4PLS3ShellElement(
        1, [1, 2, 3], "steel", reference_normal=OWNER_NORMAL
    )
    with pytest.raises(ValueError, match="quality admission failed"):
        bad.compute_stiffness_matrix(_mesh(bad_nodes), _material())

    reversed_nodes = np.asarray(
        ((0.0, 0.0, 0.0), (0.2, 0.9, 0.0), (1.0, 0.0, 0.0))
    )
    with pytest.raises(ValueError, match="connectivity winding opposes"):
        QualifiedE4PLS3ShellElement(
            2, [1, 2, 3], "steel", reference_normal=OWNER_NORMAL
        ).compute_stiffness_matrix(_mesh(reversed_nodes), _material())

    nodes = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)))
    element = QualifiedE4PLS3ShellElement(
        1, [1, 2, 3], "steel", reference_normal=OWNER_NORMAL
    )
    closed_layered_capabilities = {
        "initial_fields",
        "material_nonlinearity",
        "nonlinear_geometry",
        "patch_recovery",
        "physical_director_reversal",
        "committed_state_recovery",
        "contact_state",
        "static_restart_history",
        "arc_length_restart_history",
    }
    assert element.capability_gaps == CAPABILITY_GAPS - closed_layered_capabilities
    assert all(
        element.capability_matrix()[name] == "PARITY_GAP"
        for name in element.capability_gaps
    )
    assert all(
        element.capability_matrix()[name] == "PARITY_REPLACED"
        for name in closed_layered_capabilities
    )
    assert element.capability_gaps == frozenset()
    assert element.capability_matrix()["current_state_buckling_s3"] == (
        "PARITY_REPLACED"
    )
    assert element.capability_matrix()["mixed_current_state_buckling"] == (
        "PARITY_REPLACED"
    )
    assert element.capability_matrix()["mixed_current_state_modal"] == (
        "PARITY_REPLACED"
    )
    assert element.capability_matrix()["restart_history"] == (
        "STATIC_AND_ARC_LENGTH_CHECKPOINTS_ONLY"
    )
    assert element.capability_matrix()["linearized_limit_point"] == (
        "UNSUPPORTED_OUTSIDE_ADMITTED_PROFILE"
    )
    mass = element.compute_mass_matrix(_mesh(nodes), _material())
    assert mass.shape == (18, 18)
    assert np.all(np.isfinite(mass))
    np.testing.assert_array_equal(
        element.compute_geometric_stiffness_matrix(_mesh(nodes), _material()),
        np.zeros((18, 18)),
    )
    with pytest.raises(TypeError, match="native_rotation_trial"):
        element.compute_nonlinear_response(_mesh(nodes), _material(), np.zeros(18))


def test_layered_s3_accepts_a_model_imperfection_without_legacy_mechanics() -> None:
    model = FEModel("qualified-s3-imperfection")
    for node_id, coordinate in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)), start=1
    ):
        model.add_node(node_id, *coordinate)
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1, [1, 2, 3], "default", reference_normal=OWNER_NORMAL
        ),
    )
    before = model.mesh.get_node(3).coords().copy()

    returned = apply_imperfection(
        model,
        ImperfectionField({3: (0.0, 0.0, 0.1)}),
        copy_model=False,
    )

    assert returned is model
    np.testing.assert_array_equal(
        model.mesh.get_node(3).coords(),
        before + np.asarray((0.0, 0.0, 0.1)),
    )
