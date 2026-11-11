from __future__ import annotations

from collections.abc import Mapping
import json
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

import anyfesolver.e4_pl_element as q4_element_module
from anyfesolver.e4_pl_element import FORMULATION_ID, QualifiedE4PLShellElement, equation7_frame
from anyfesolver.elements import ShellElement, create_element
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.materials import Hill48Yield, OrthotropicMaterial
from anyfesolver.shell_sections import GeneralizedShellSection


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "docs" / "reference_cases"
sys.path.insert(0, str(CASES))
import e4_pl_q1b_assembled_producer as q1b  # noqa: E402


def _mesh(nodes: list[list[float]]) -> FEMesh:
    mesh = FEMesh()
    for identifier, coordinate in enumerate(nodes, start=1):
        mesh.add_node(identifier, *coordinate)
    return mesh


def _material() -> Material:
    return Material("q1", 15.0, 0.25, density=7.85)


def _geometry_rows() -> list[dict[str, object]]:
    value = json.loads((CASES / "e4_pl_q1r_geometry_contract.json").read_text(encoding="utf-8"))
    return value["geometries"]


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, ord=np.inf) / max(np.linalg.norm(right, ord=np.inf), 1.0))


def _rigid(nodes: np.ndarray) -> np.ndarray:
    result = np.zeros((24, 6), dtype=float)
    for node, (x, y, _z) in enumerate(nodes):
        base = 6 * node
        result[base, 0] = 1.0
        result[base + 1, 1] = 1.0
        result[base + 2, 2] = 1.0
        result[base + 2, 3] = y
        result[base + 3, 3] = 1.0
        result[base + 2, 4] = -x
        result[base + 4, 4] = 1.0
        result[base, 5] = -y
        result[base + 1, 5] = x
        result[base + 5, 5] = 1.0
    return result


def test_dormant_kernel_matches_frozen_q1b_components_for_registered_geometries() -> None:
    for row in _geometry_rows():
        nodes = [[float(Fraction(value)) for value in coordinate] for coordinate in row["nodes"]]
        mesh = _mesh(nodes)
        element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "q1", thickness=2.0 / 3.0)
        actual = element.compute_stiffness_components(mesh, _material())
        expected = q1b.local_components(nodes, thickness=2.0 / 3.0)
        assert actual["legacy_fallback"] is False, row["id"]
        assert actual["mixed_condensed"] is True
        assert _relative(actual["core"], expected["core"]) < 2.0e-12, row["id"]
        assert _relative(actual["pl"], expected["pl"]) < 2.0e-12, row["id"]
        assert _relative(actual["hourglass"], expected["hg"]) < 2.0e-12, row["id"]
        assert _relative(actual["total"], expected["total"]) < 2.0e-12, row["id"]
        assert np.linalg.norm(actual["total"] - actual["total"].T, ord=np.inf) < 1.0e-10
        residual = actual["total"] @ _rigid(np.asarray(nodes, dtype=float))
        assert np.linalg.norm(residual, ord=np.inf) / max(np.linalg.norm(actual["total"], ord=np.inf), 1.0) < 2.0e-11


def test_d4_numbering_covariance_and_equation7_frame() -> None:
    nodes = np.asarray(_geometry_rows()[3]["nodes"], dtype=float)
    base_mesh = _mesh(nodes.tolist())
    base = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "q1", thickness=2.0 / 3.0)
    base_matrix = base.compute_stiffness_matrix(base_mesh, _material())
    operations = (
        (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2),
        (1, 0, 3, 2), (3, 2, 1, 0), (0, 3, 2, 1), (2, 1, 0, 3),
    )
    for permutation in operations:
        numbered = nodes[list(permutation)]
        frame, local, warpage = equation7_frame(numbered)
        assert np.linalg.norm(frame.T @ frame - np.eye(3), ord=np.inf) < 1.0e-12
        assert np.linalg.det(frame) > 1.0 - 1.0e-12
        assert warpage < 1.0e-14
        assert local.shape == (4, 2)
        mesh = _mesh(numbered.tolist())
        element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "q1", thickness=2.0 / 3.0)
        actual = element.compute_stiffness_matrix(mesh, _material())
        dofs = [6 * node + component for node in permutation for component in range(6)]
        expected = base_matrix[np.ix_(dofs, dofs)]
        assert _relative(actual, expected) < 3.0e-12


def test_numerical_terms_are_separate_from_physical_recovery() -> None:
    nodes = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.0, 0.8, 0.0], [-0.1, 0.7, 0.0]]
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "q1", thickness=2.0 / 3.0)
    components = element.compute_stiffness_components(mesh, material)
    displacement = np.linspace(-0.02, 0.03, 24)
    numerical = element.numerical_internal_force(displacement)
    assert np.allclose(numerical["numerical"], numerical["pl"] + numerical["hourglass"])
    assert np.allclose(components["total"] @ displacement, components["physical"] @ displacement + numerical["numerical"])
    legacy = ShellElement(2, [1, 2, 3, 4], "q1", thickness=2.0 / 3.0)
    candidate_stress = element.compute_stresses(mesh, displacement, material)
    legacy_stress = legacy.compute_stresses(mesh, displacement, material)
    assert candidate_stress["numerical_fields_excluded"] is True
    assert candidate_stress["recovery_scope"] == "qualified_q4_local_physical_only"
    inherited_resultants = np.column_stack(
        (
            np.column_stack(
                tuple(legacy_stress[key] for key in ("membrane_xx", "membrane_yy", "membrane_xy"))
            )
            * element.thickness,
            np.column_stack(
                tuple(legacy_stress[key] for key in ("bending_xx", "bending_yy", "bending_xy"))
            )
            * element.thickness**2
            / 6.0,
            np.column_stack(tuple(legacy_stress[key] for key in ("shear_xz", "shear_yz")))
            * element.thickness,
        )
    )
    stationary_resultants = np.column_stack(
        (
            candidate_stress["membrane_resultants"],
            candidate_stress["bending_resultants"],
            candidate_stress["transverse_shear_resultants"],
        )
    )
    assert not np.allclose(stationary_resultants, inherited_resultants)


def test_inherited_mass_geometric_state_and_direct_warped_parity() -> None:
    nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.02], [1.0, 1.0, -0.01], [0.0, 1.0, 0.01]]
    mesh = _mesh(nodes)
    material = _material()
    candidate = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "q1", thickness=0.02)
    legacy = ShellElement(2, [1, 2, 3, 4], "q1", thickness=0.02)
    components = candidate.compute_stiffness_components(mesh, material)
    assert components["legacy_fallback"] is False
    assert components["warped_direct"] is True
    assert components["warped_formulation"] == "varying_frame"
    assert np.array_equal(components["total"], legacy.compute_stiffness_matrix(mesh, material))
    assert np.array_equal(candidate.compute_mass_matrix(mesh, material), legacy.compute_mass_matrix(mesh, material))
    state = {"membrane_compression": [2.0, 1.0, 0.25]}
    assert np.array_equal(
        candidate.compute_geometric_stiffness_matrix(mesh, material, state),
        legacy.compute_geometric_stiffness_matrix(mesh, material, state),
    )
    with pytest.raises(ValueError, match="warped"):
        QualifiedE4PLShellElement(
            3,
            [1, 2, 3, 4],
            "q1",
            thickness=0.02,
            warped_formulation="reject",
        ).compute_stiffness_matrix(mesh, material)


def test_internal_force_and_nonlinear_tangent_use_the_qualified_baseline() -> None:
    nodes = [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.0, 0.9, 0.0], [-0.1, 0.8, 0.0]]
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "q1", thickness=0.025)
    displacement = np.linspace(-2.0e-4, 3.0e-4, 24)
    qualified = element.compute_stiffness_matrix(mesh, material).copy()
    assert np.allclose(
        element.compute_internal_forces(mesh, displacement, material),
        qualified @ displacement,
    )

    zero = np.zeros(24)
    zero_force, zero_tangent, state = element.compute_nonlinear_response(
        mesh, material, zero, tangent=True
    )
    assert np.array_equal(zero_force, zero)
    assert zero_tangent is not None
    assert _relative(zero_tangent, qualified) < 5.0e-13
    assert isinstance(state, dict)

    force, tangent, _state = element.compute_nonlinear_response(
        mesh, material, displacement, tangent=True
    )
    assert tangent is not None
    direction = np.linspace(0.7, -0.4, 24)
    step = 2.0e-7
    plus = element.compute_nonlinear_response(
        mesh, material, displacement + step * direction, tangent=False
    )[0]
    minus = element.compute_nonlinear_response(
        mesh, material, displacement - step * direction, tangent=False
    )[0]
    finite_difference = (plus - minus) / (2.0 * step)
    assert _relative(finite_difference, tangent @ direction) < 2.0e-7
    assert np.all(np.isfinite(force))


def test_generalized_section_and_warped_nonlinear_paths_retain_parity() -> None:
    nodes = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.1, 0.8, 0.0], [-0.1, 0.7, 0.0]]
    mesh = _mesh(nodes)
    material = _material()
    section = GeneralizedShellSection(
        A=np.asarray(((120.0, 12.0, 4.0), (12.0, 95.0, -3.0), (4.0, -3.0, 42.0))),
        B=np.asarray(((1.2, 0.1, 0.0), (0.05, -0.8, 0.1), (0.0, 0.05, 0.3))),
        D=np.asarray(((15.0, 1.0, 0.2), (1.0, 11.0, -0.1), (0.2, -0.1, 5.0))),
        As=np.diag((25.0, 20.0)),
        mass_per_area=8.0,
        rotary_inertia_per_area=0.02,
    )
    element = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "q1",
        shell_section=section,
        reference_normal=np.asarray((0.0, 0.0, -1.0)),
    )
    qualified = element.compute_stiffness_matrix(mesh, material).copy()
    force, tangent, state = element.compute_nonlinear_response(
        mesh, material, np.zeros(24), tangent=True
    )
    assert np.array_equal(force, np.zeros(24))
    assert tangent is not None and _relative(tangent, qualified) < 5.0e-13
    assert state["generalized_section"] is True
    assert state["recovery_scope"] == "section_resultants_only"
    assert np.all(np.linalg.eigvalsh(element.compute_mass_matrix(mesh, material)) >= -1.0e-12)

    warped_nodes = np.asarray(nodes, dtype=float)
    warped_nodes[1, 2] = 0.03
    warped_nodes[2, 2] = -0.02
    warped_mesh = _mesh(warped_nodes.tolist())
    candidate = QualifiedE4PLShellElement(2, [1, 2, 3, 4], "q1", thickness=0.025)
    legacy = ShellElement(3, [1, 2, 3, 4], "q1", thickness=0.025)
    displacement = np.linspace(-1.0e-4, 2.0e-4, 24)
    candidate_response = candidate.compute_nonlinear_response(
        warped_mesh, material, displacement, tangent=True
    )
    legacy_response = legacy.compute_nonlinear_response(
        warped_mesh, material, displacement, tangent=True
    )
    assert np.allclose(candidate_response[0], legacy_response[0], rtol=0.0, atol=2.0e-13)
    assert np.allclose(candidate_response[1], legacy_response[1], rtol=0.0, atol=2.0e-13)


def test_element_is_serializable_default_and_has_explicit_legacy_rollback() -> None:
    element = QualifiedE4PLShellElement(
        7,
        [1, 2, 3, 4],
        "q1",
        thickness=0.02,
        drilling_stabilization=2.0e-3,
        hourglass_stabilization=3.0e-3,
        material_direction=np.asarray((1.0, 0.2, 0.0)),
        material_angle_deg=17.0,
        pl_stabilization=0.8,
        planar_tolerance=2.0e-9,
        warped_formulation="reject",
    )
    payload = element.to_dict()
    assert payload["formulation_id"] == FORMULATION_ID
    assert payload["type"] == "QualifiedE4PLShellElement"
    assert payload["warped_formulation"] == "reject"
    rebuilt = QualifiedE4PLShellElement.from_dict(json.loads(json.dumps(payload)))
    assert rebuilt.to_dict() == payload
    default = create_element("shell", 1, [1, 2, 3, 4], "q1", thickness=0.02)
    rollback = create_element("legacy-shell", 2, [1, 2, 3, 4], "q1", thickness=0.02)
    assert type(default) is QualifiedE4PLShellElement
    assert type(rollback) is ShellElement
    opt_in = create_element("e4-pl", 1, [1, 2, 3, 4], "q1", thickness=0.02)
    assert type(opt_in) is QualifiedE4PLShellElement
    with pytest.raises(ValueError, match="exactly four"):
        QualifiedE4PLShellElement(1, [1, 2, 3], "q1")


def test_qualified_q4_connectivity_and_current_serialization_are_exact() -> None:
    with pytest.raises(TypeError, match="exact non-boolean integers"):
        QualifiedE4PLShellElement(1, [1.9, 2, 3, 4], "q1")
    element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "q1")
    with pytest.raises(TypeError, match="exact non-boolean integers"):
        element.node_ids = (1, 2, 3, True)
    for name, value in (
        ("material_direction", (True, 0.0, 0.0)),
        ("reference_normal", (0.0, 0.0, 0.0)),
        ("hourglass_stabilization", -1.0),
        ("pl_stabilization", -1.0),
        ("planar_tolerance", -1.0),
    ):
        with pytest.raises((TypeError, ValueError)):
            setattr(element, name, value)
    for value in (True, "0.02"):
        with pytest.raises(TypeError):
            QualifiedE4PLShellElement(1, [1, 2, 3, 4], "q1", thickness=value)
    for keyword in (
        "drilling_stabilization",
        "hourglass_stabilization",
        "material_angle_deg",
        "pl_stabilization",
        "planar_tolerance",
    ):
        with pytest.raises(TypeError):
            QualifiedE4PLShellElement(
                1,
                [1, 2, 3, 4],
                "q1",
                **{keyword: True},
            )
    with pytest.raises(TypeError, match="legacy_warped_fallback"):
        QualifiedE4PLShellElement(
            1,
            [1, 2, 3, 4],
            "q1",
            legacy_warped_fallback="false",
        )

    payload = json.loads(json.dumps(element.to_dict()))
    malformed = (
        dict(payload, element_id=1.9),
        dict(payload, node_ids=[1.0, 2, 3, 4]),
        dict(payload, material_name=123),
        dict(payload, thickness=True),
        dict(payload, thickness="0.01"),
        dict(payload, thickness=-0.01),
        dict(payload, reduced_integration="false"),
        dict(payload, director_polarity=True),
        dict(payload, director_polarity=1.9),
        dict(payload, warped_formulation="REJECT"),
        dict(payload, formulation_schema="foreign-family-marker"),
        dict(payload, legacy_warped_fallback=True),
    )
    for record in malformed:
        with pytest.raises(ValueError):
            QualifiedE4PLShellElement.from_dict(record)

    for key in (
        "material_name",
        "thickness",
        "drilling_stabilization",
        "reduced_integration",
        "hourglass_stabilization",
        "material_direction",
        "material_angle_deg",
        "pl_stabilization",
        "planar_tolerance",
        "warped_formulation",
    ):
        incomplete = dict(payload)
        incomplete.pop(key)
        with pytest.raises(ValueError, match="keys are incompatible"):
            QualifiedE4PLShellElement.from_dict(incomplete)

    with pytest.raises(ValueError, match="requires reference_normal"):
        element.director_polarity = -1
    element.reference_normal = (0.0, 0.0, 1.0)
    element.director_polarity = -1
    with pytest.raises(ValueError, match="requires reference_normal"):
        element.reference_normal = None
    directed_payload = json.loads(json.dumps(element.to_dict()))
    with pytest.raises(ValueError, match="noncanonical types"):
        QualifiedE4PLShellElement.from_dict(
            dict(directed_payload, reference_normal=[0.0, 0.0, 2.0])
        )
    with pytest.raises(ValueError, match="formulation_id"):
        QualifiedE4PLShellElement.from_dict(
            dict(
                directed_payload,
                formulation_id=q4_element_module._PLANAR_FORMULATION_ID,
            )
        )

    pre_policy = dict(payload)
    for key in (
        "implementation_id",
        "recovery_policy_id",
        "stationary_solve_policy_id",
        "director_polarity_policy_id",
        "director_reversal_transform_id",
        "current_state_binding_schema_id",
        "current_state_algorithmic_origin_schema_id",
        "current_state_tangent_decomposition_policy_id",
        "current_state_projection_policy_id",
        "activity_disposition_schema_id",
        "deleted_frozen_policy_id",
        "failed_state_policy_id",
        "quadrature_authority_id",
        "reference_normal",
        "director_polarity",
    ):
        pre_policy.pop(key)
    for changed in (
        dict(pre_policy, thickness="0.01"),
        dict(pre_policy, element_id=1.9),
        dict(pre_policy, reduced_integration="false"),
        dict(pre_policy, formulation_schema="foreign-family-marker"),
    ):
        with pytest.raises(ValueError):
            QualifiedE4PLShellElement.from_dict(changed)
    missing_pre_policy = dict(pre_policy)
    missing_pre_policy.pop("pl_stabilization")
    with pytest.raises(ValueError, match="keys are incompatible"):
        QualifiedE4PLShellElement.from_dict(missing_pre_policy)


def test_q4_deserialization_and_vector_callbacks_recheck_runtime_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = QualifiedE4PLShellElement(
        1, [1, 2, 3, 4], "q1"
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
        QualifiedE4PLShellElement.from_dict(MutatingMapping())
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
        QualifiedE4PLShellElement(
            1,
            [1, 2, 3, 4],
            "q1",
            reference_normal=MutatingVector(),
        )
    assert reached == []


def test_q4_direct_mechanics_rejects_changed_formulation_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "q1")
    reached: list[str] = []

    def changed_stationary(*_args: object, **_kwargs: object) -> object:
        reached.append("stationary")
        raise AssertionError("changed stationary helper reached mechanics")

    monkeypatch.setattr(
        q4_element_module,
        "_stationary_blocks",
        changed_stationary,
    )
    with pytest.raises(ValueError, match="_stationary_blocks"):
        element.compute_stiffness_matrix(
            _mesh(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            _material(),
        )
    assert reached == []


def test_q4_direct_callbacks_recheck_authority_before_cached_force_or_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _mesh(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "q1")
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
            return original_asarray(np.zeros(24), dtype=dtype)

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
        )
    assert reached == []

    monkeypatch.setattr(np, "asarray", original_asarray)
    with pytest.raises(ValueError, match="numpy.asarray"):
        element.compute_committed_current_tangent_components(
            mesh,
            material,
            MutatingDisplacement(),
            {},
        )
    assert reached == []

    monkeypatch.setattr(np, "asarray", original_asarray)
    original_get_node = mesh.get_node

    def changed_stationary(*_args: object, **_kwargs: object) -> object:
        reached.append("stationary")
        raise AssertionError("changed stationary helper reached mechanics")

    def mutating_get_node(node_id: int):
        monkeypatch.setattr(
            q4_element_module,
            "_stationary_blocks",
            changed_stationary,
        )
        return original_get_node(node_id)

    monkeypatch.setattr(mesh, "get_node", mutating_get_node)
    with pytest.raises(ValueError, match="_stationary_blocks"):
        element.compute_stiffness_matrix(mesh, material)
    assert reached == []

def test_geometry_and_material_revision_clear_all_candidate_caches() -> None:
    nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "q1")
    mesh.add_element(1, element)
    element.compute_stiffness_matrix(mesh, material)
    assert element._stiffness_matrix is not None
    assert element._qualified_components is not None
    assert element._qualified_cache_key is not None
    mesh.bump_revision("geometry")
    assert element._stiffness_matrix is None
    assert element._qualified_components is None
    assert element._qualified_cache_key is None


def test_warm_stiffness_reuses_components_until_authoritative_revision() -> None:
    nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "q1")
    mesh.add_element(1, element)
    first = element.compute_stiffness_matrix(mesh, material)
    first_components = element._qualified_components
    second = element.compute_stiffness_matrix(mesh, material)
    assert second is not first
    assert first.flags.writeable is False
    assert second.flags.writeable is False
    np.testing.assert_array_equal(second, first)
    assert element._qualified_components is first_components
    mesh.bump_revision("material")
    third = element.compute_stiffness_matrix(mesh, material)
    assert third is not first
    assert element._qualified_components is not first_components


def test_orthotropic_elastic_and_plastic_state_paths_keep_qualified_tangent() -> None:
    yield_stress = 100.0e6
    shear_yield = yield_stress / np.sqrt(3.0)
    material = OrthotropicMaterial(
        name="lamina",
        elastic_modulus_1=150.0e9,
        elastic_modulus_2=12.0e9,
        elastic_modulus_3=10.0e9,
        poisson_ratio_12=0.25,
        poisson_ratio_13=0.20,
        poisson_ratio_23=0.30,
        shear_modulus_12=5.0e9,
        shear_modulus_13=4.0e9,
        shear_modulus_23=3.8e9,
        density=1600.0,
        hill_yield=Hill48Yield(
            yield_stress,
            yield_stress,
            yield_stress,
            shear_yield,
            shear_yield,
            shear_yield,
        ),
        hardening_curve=DNVC208MaterialCurve(
            sigma_prop=yield_stress,
            sigma_yield=105.0e6,
            sigma_yield_2=110.0e6,
            eps_p_y1=0.005,
            eps_p_y2=0.010,
            K=400.0e6,
            n=0.20,
        ),
    )
    nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    mesh = _mesh(nodes)
    candidate = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "lamina",
        thickness=0.01,
        material_direction=np.asarray((1.0, 0.3, 0.0)),
    )
    legacy = ShellElement(
        2,
        [1, 2, 3, 4],
        "lamina",
        thickness=0.01,
        material_direction=np.asarray((1.0, 0.3, 0.0)),
    )
    stiffness = candidate.compute_stiffness_matrix(mesh, material)
    assert np.all(np.isfinite(stiffness))
    assert np.linalg.norm(stiffness - stiffness.T, ord=np.inf) < 1.0e-5
    displacement = np.zeros(24)
    displacement[6] = 0.004
    displacement[12] = 0.004
    # Owner decision 2026-10-23: plastic qualified Q4 outside the assumed-strain
    # route scope is refused (the compatible-strain plastic route with the
    # qualified correction is unstable once layers yield).
    from anyfesolver._q4_assumed_strain_plasticity import UnsupportedQ4PlasticRouteError
    with pytest.raises(UnsupportedQ4PlasticRouteError, match="orthotropic/Hill48"):
        candidate.compute_nonlinear_response(
            mesh, material, displacement, num_layers=3, tangent=True
        )
    legacy_force, legacy_tangent, legacy_state = legacy.compute_nonlinear_response(
        mesh, material, displacement, num_layers=3, tangent=True
    )
    assert legacy_tangent is not None and np.all(np.isfinite(legacy_force))
    assert np.max(legacy_state["alpha"]) > 0.0


