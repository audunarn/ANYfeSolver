from __future__ import annotations

import itertools

import numpy as np
import pytest

from anyfesolver import FEModel, QualifiedE4PLS3ShellElement
from anyfesolver.buckling import solve_eigenvalue_buckling
from anyfesolver.e4_pl_s3_element import (
    GEOMETRIC_STIFFNESS_POLICY_ID,
    HOMOGENEOUS_ELASTIC_STRESS_PROFILE_ID,
    PHYSICAL_EXTERNAL_INDICES,
    REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
    TRIANGLE_QUADRATURE,
)
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver.matrix_assembly import assemble_geometric_stiffness_matrix
from anyfesolver.nonlinear import solve_nonlinear_load_stepping
from anyfesolver.shell_sections import GeneralizedShellSection


OWNER_NORMAL = np.asarray((0.0, 0.0, 1.0))


def _mesh(nodes: np.ndarray) -> FEMesh:
    mesh = FEMesh()
    for identifier, coordinate in enumerate(nodes, start=1):
        mesh.add_node(identifier, *coordinate)
    return mesh


def _material() -> Material:
    return Material("steel", 210.0e9, 0.3, density=7850.0)


def _element() -> QualifiedE4PLS3ShellElement:
    return QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.12,
        reference_normal=OWNER_NORMAL,
    )


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right, ord=np.inf)
        / max(np.linalg.norm(right, ord=np.inf), 1.0)
    )


def _state(scale: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "bubble_linearization_policy": REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
        "membrane_compression": scale * np.asarray((80.0, 31.0, -9.0)),
        "bending_compression": scale * np.asarray((1.8, -0.7, 0.35)),
        "stress_second_moment": scale * np.asarray((0.041, 0.028, -0.006)),
    }


def _barycentric_derivatives(
    local_nodes: np.ndarray,
    r: float,
    s: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Independent literal derivative oracle for L_i and 27 L1 L2 L3."""

    jacobian = np.asarray(
        (
            (local_nodes[1, 0] - local_nodes[0, 0], local_nodes[1, 1] - local_nodes[0, 1]),
            (local_nodes[2, 0] - local_nodes[0, 0], local_nodes[2, 1] - local_nodes[0, 1]),
        )
    )
    inverse = np.linalg.inv(jacobian)
    derivative_r = np.asarray((-1.0, 1.0, 0.0))
    derivative_s = np.asarray((-1.0, 0.0, 1.0))
    bubble_r = 27.0 * s * (1.0 - 2.0 * r - s)
    bubble_s = 27.0 * r * (1.0 - r - 2.0 * s)
    derivative_x = inverse[0, 0] * derivative_r + inverse[0, 1] * derivative_s
    derivative_y = inverse[1, 0] * derivative_r + inverse[1, 1] * derivative_s
    bubble_x = inverse[0, 0] * bubble_r + inverse[0, 1] * bubble_s
    bubble_y = inverse[1, 0] * bubble_r + inverse[1, 1] * bubble_s
    return derivative_x, derivative_y, float(bubble_x), float(bubble_y)


def test_zero_state_is_exactly_zero_and_only_the_element_operator_is_closed() -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)))
    element = _element()

    made = element.compute_geometric_stiffness_components(
        _mesh(nodes), _material(), None
    )

    assert made["policy_id"] == GEOMETRIC_STIFFNESS_POLICY_ID
    assert made["numerical_fields_excluded"] is True
    np.testing.assert_array_equal(made["full_local"], np.zeros((20, 20)))
    np.testing.assert_array_equal(made["condensed_local"], np.zeros((18, 18)))
    np.testing.assert_array_equal(made["global"], np.zeros((18, 18)))
    assert element.capability_matrix()["geometric_stiffness"] == "PARITY_REPLACED"
    assert element.capability_matrix()["buckling"] == (
        "EXPLICIT_REFERENCE_ELASTIC_OR_CURRENT_STATE_AUTHORITY_REQUIRED"
    )


def test_full_mindlin_energy_and_bubble_reduction_have_an_independent_oracle() -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)))
    element = _element()
    made = element.compute_geometric_stiffness_components(
        _mesh(nodes), _material(), _state()
    )
    local_nodes = nodes[:, :2] - nodes[0, :2]
    determinant = float(
        np.linalg.det(
            np.asarray(
                (
                    (local_nodes[1, 0], local_nodes[1, 1]),
                    (local_nodes[2, 0], local_nodes[2, 1]),
                )
            )
        )
    )
    rng = np.random.default_rng(20260825)
    external = rng.normal(size=18)
    full = np.asarray(made["bubble_schur_map"]) @ external

    membrane = np.asarray(_state()["membrane_compression"])
    bending = np.asarray(_state()["bending_compression"])
    second = np.asarray(_state()["stress_second_moment"])
    membrane_matrix = np.asarray(
        ((membrane[0], membrane[2]), (membrane[2], membrane[1]))
    )
    bending_matrix = np.asarray(
        ((bending[0], bending[2]), (bending[2], bending[1]))
    )
    second_matrix = np.asarray(
        ((second[0], second[2]), (second[2], second[1]))
    )

    direct = 0.0
    for r, s, weight in TRIANGLE_QUADRATURE:
        derivative_x, derivative_y, bubble_x, bubble_y = (
            _barycentric_derivatives(local_nodes, r, s)
        )

        gradients = []
        for component in range(3):
            values = full[component:18:6]
            gradients.append(
                np.asarray((derivative_x @ values, derivative_y @ values))
            )
        rotation_x = full[3:18:6]
        rotation_y = full[4:18:6]
        rotation_x_gradient = np.asarray(
            (
                derivative_x @ rotation_x + bubble_x * full[18],
                derivative_y @ rotation_x + bubble_y * full[18],
            )
        )
        rotation_y_gradient = np.asarray(
            (
                derivative_x @ rotation_y + bubble_x * full[19],
                derivative_y @ rotation_y + bubble_y * full[19],
            )
        )
        density = sum(
            float(gradient @ membrane_matrix @ gradient)
            for gradient in gradients
        )
        density += float(rotation_x_gradient @ second_matrix @ rotation_x_gradient)
        density += float(rotation_y_gradient @ second_matrix @ rotation_y_gradient)
        density += 2.0 * float(
            gradients[0] @ bending_matrix @ rotation_y_gradient
        )
        density -= 2.0 * float(
            gradients[1] @ bending_matrix @ rotation_x_gradient
        )
        direct += abs(determinant) * float(weight) * density

    matrix_energy = float(external @ made["condensed_local"] @ external)
    full_energy = float(full @ made["full_local"] @ full)
    assert matrix_energy == pytest.approx(direct, rel=3.0e-13, abs=3.0e-12)
    assert full_energy == pytest.approx(direct, rel=3.0e-13, abs=3.0e-12)
    np.testing.assert_allclose(
        made["condensed_local"],
        made["bubble_schur_map"].T
        @ made["full_local"]
        @ made["bubble_schur_map"],
        rtol=0.0,
        atol=2.0e-13,
    )


def test_condensed_operator_is_the_exact_material_schur_derivative() -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)))
    element = _element()
    mesh = _mesh(nodes)
    material = _material()
    stiffness = element.compute_stiffness_components(mesh, material)
    geometric = element.compute_geometric_stiffness_components(
        mesh, material, _state(1.0e8)
    )
    physical_20 = np.concatenate(
        (PHYSICAL_EXTERNAL_INDICES, np.asarray((18, 19), dtype=np.intp))
    )
    material_17 = np.asarray(stiffness["uncondensed_physical"], dtype=float)
    geometric_17 = np.asarray(geometric["full_local"])[
        np.ix_(physical_20, physical_20)
    ]

    def schur(matrix: np.ndarray) -> np.ndarray:
        external = matrix[:15, :15]
        coupling = matrix[:15, 15:]
        internal = matrix[15:, 15:]
        return external - coupling @ np.linalg.solve(internal, coupling.T)

    step = 1.0e-20
    derivative = np.imag(
        schur(material_17.astype(complex) + 1j * step * geometric_17)
    ) / step
    expected = np.asarray(geometric["condensed_local"])[
        np.ix_(PHYSICAL_EXTERNAL_INDICES, PHYSICAL_EXTERNAL_INDICES)
    ]
    assert _relative(derivative, expected) < 2.0e-12


def test_state_scaling_symmetry_and_numerical_coordinate_exclusion() -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)))
    element = _element()
    baseline = element.compute_geometric_stiffness_components(
        _mesh(nodes), _material(), _state()
    )
    scaled = element.compute_geometric_stiffness_components(
        _mesh(nodes), _material(), _state(3.25)
    )

    np.testing.assert_allclose(
        scaled["global"], 3.25 * baseline["global"], rtol=2.0e-15, atol=2.0e-13
    )
    np.testing.assert_array_equal(baseline["global"], baseline["global"].T)
    drill = np.asarray((5, 11, 17), dtype=np.intp)
    np.testing.assert_array_equal(baseline["condensed_local"][drill], 0.0)
    np.testing.assert_array_equal(baseline["condensed_local"][:, drill], 0.0)
    assert np.linalg.norm(baseline["full_local"][18:, :]) > 0.0
    replayed = element.compute_geometric_stiffness_components(
        _mesh(nodes), _material(), baseline
    )
    np.testing.assert_array_equal(replayed["global"], baseline["global"])


def test_uniform_compression_and_tension_aliases_share_the_homogeneous_default() -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)))
    compression = {
        "bubble_linearization_policy": REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
        "membrane_compression": [80.0, 31.0, -9.0],
        "bending_compression": [1.8, -0.7, 0.35],
        "through_thickness_stress_profile": (
            HOMOGENEOUS_ELASTIC_STRESS_PROFILE_ID
        ),
    }
    tension = {
        "bubble_linearization_policy": REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
        "membrane_forces": [-80.0, -31.0, 9.0],
        "bending_moments": [-1.8, 0.7, -0.35],
        "through_thickness_stress_profile": (
            HOMOGENEOUS_ELASTIC_STRESS_PROFILE_ID
        ),
    }
    made_compression = _element().compute_geometric_stiffness_components(
        _mesh(nodes), _material(), compression
    )
    made_tension = _element().compute_geometric_stiffness_components(
        _mesh(nodes), _material(), tension
    )

    np.testing.assert_array_equal(made_tension["global"], made_compression["global"])
    np.testing.assert_array_equal(
        made_compression["stress_second_moment"],
        made_compression["membrane_compression"] * 0.12**2 / 12.0,
    )
    assert made_tension["second_moment_authority"] == (
        HOMOGENEOUS_ELASTIC_STRESS_PROFILE_ID
    )


def test_all_six_numberings_transport_nonuniform_physical_resultant_fields() -> None:
    nodes = np.asarray(
        ((0.1, 0.2, 0.3), (1.2, 0.1, 0.5), (0.3, 1.0, 0.4)), dtype=float
    )
    owner = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
    material = _material()
    normal = owner / np.linalg.norm(owner)
    baseline_e1 = (nodes[1] - nodes[0]) / np.linalg.norm(nodes[1] - nodes[0])
    baseline_e2 = np.cross(normal, baseline_e1)
    baseline_frame = np.column_stack((baseline_e1, baseline_e2, normal))
    origin = np.mean(nodes, axis=0)

    def numbered_frame(numbered: np.ndarray) -> np.ndarray:
        e1 = (numbered[1] - numbered[0]) / np.linalg.norm(
            numbered[1] - numbered[0]
        )
        e2 = np.cross(normal, e1)
        e1 = np.cross(e2, normal)
        return np.column_stack((e1, e2, normal))

    def physical_state(numbered: np.ndarray, frame: np.ndarray) -> dict[str, np.ndarray]:
        result: dict[str, list[np.ndarray]] = {
            "membrane_compression_at_gauss": [],
            "bending_compression_at_gauss": [],
            "stress_second_moment_at_gauss": [],
        }
        for r, s, _weight in TRIANGLE_QUADRATURE:
            shape = np.asarray((1.0 - r - s, r, s))
            point = shape @ numbered
            x, y = baseline_frame[:, :2].T @ (point - origin)
            tensors = (
                np.asarray(((47.0 + 3.0 * x - 2.0 * y, 9.0 + 0.4 * x),
                            (9.0 + 0.4 * x, 23.0 - 1.5 * x + 0.8 * y))),
                np.asarray(((0.8 + 0.1 * y, -0.17 + 0.03 * x),
                            (-0.17 + 0.03 * x, -0.35 + 0.04 * y))),
                np.asarray(((0.041 + 0.003 * x, -0.006 + 0.001 * y),
                            (-0.006 + 0.001 * y, 0.028 - 0.002 * x))),
            )
            for key, tensor in zip(result, tensors):
                global_tensor = baseline_frame[:, :2] @ tensor @ baseline_frame[:, :2].T
                local_tensor = frame[:, :2].T @ global_tensor @ frame[:, :2]
                result[key].append(
                    np.asarray((local_tensor[0, 0], local_tensor[1, 1], local_tensor[0, 1]))
                )
        made = {key: np.asarray(values) for key, values in result.items()}
        made["bubble_linearization_policy"] = (
            REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID
        )
        return made

    def geometric(numbered: np.ndarray) -> np.ndarray:
        element = QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.12,
            reference_normal=owner,
        )
        return element._compute_geometric_stiffness_components(
            _mesh(numbered),
            material,
            physical_state(numbered, numbered_frame(numbered)),
            enforce_positive_winding=False,
        )["global"]

    baseline = geometric(nodes)
    for permutation in itertools.permutations(range(3)):
        actual = geometric(nodes[list(permutation)])
        dofs = [6 * node + component for node in permutation for component in range(6)]
        expected = baseline[np.ix_(dofs, dofs)]
        assert _relative(actual, expected) < 7.0e-14, permutation


def test_assembly_uses_the_formulation_native_operator_without_claiming_buckling() -> None:
    model = FEModel("qualified-s3-initial-stress")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)), start=1
    ):
        model.add_node(node_id, *coordinates)
    element = _element()
    model.add_element(1, element)

    assembled, info = assemble_geometric_stiffness_matrix(model, {1: _state()})
    expected = element.compute_geometric_stiffness_matrix(
        model.mesh, model.materials["steel"], _state()
    )

    np.testing.assert_allclose(assembled.toarray(), expected, rtol=0.0, atol=0.0)
    assert info["matrix_type"] == "geometric_stiffness"
    assert element.capability_matrix()["buckling"] == (
        "EXPLICIT_REFERENCE_ELASTIC_OR_CURRENT_STATE_AUTHORITY_REQUIRED"
    )


def test_buckling_gap_is_checked_before_boundary_or_matrix_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FEModel("qualified-s3-buckling-guard")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)), start=1
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(1, _element())

    def forbidden() -> None:
        raise AssertionError("buckling evaluated the model before its capability guard")

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(ElementCapabilityError, match="buckling"):
        solve_eigenvalue_buckling(model, {1: _state()}, num_modes=1)


def test_limit_point_gap_is_checked_before_boundary_or_matrix_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FEModel("qualified-s3-limit-point-guard")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)), start=1
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(1, _element())

    def forbidden() -> None:
        raise AssertionError("limit-point solve evaluated the model before its guard")

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(
        ElementCapabilityError,
        match="linearized_limit_point=UNSUPPORTED_OUTSIDE_ADMITTED_PROFILE",
    ):
        solve_nonlinear_load_stepping(
            model,
            element_states={1: _state()},
            num_steps=1,
        )


def test_generalized_and_history_resultants_require_an_explicit_second_moment() -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)))
    section = GeneralizedShellSection(
        A=np.asarray(((120.0, 12.0, 0.0), (12.0, 100.0, 0.0), (0.0, 0.0, 44.0))),
        B=np.zeros((3, 3)),
        D=np.asarray(((14.0, 1.0, 0.0), (1.0, 11.0, 0.0), (0.0, 0.0, 5.0))),
        As=np.asarray(((28.0, 0.0), (0.0, 24.0))),
    )
    generalized = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.12,
        shell_section=section,
        material_direction=np.asarray((1.0, 0.0, 0.0)),
        reference_normal=OWNER_NORMAL,
    )
    with pytest.raises(ValueError, match="explicit stress_second_moment"):
        generalized.compute_geometric_stiffness_matrix(
            _mesh(nodes), _material(), {"membrane_compression": [3.0, 2.0, 0.1]}
        )
    made = generalized.compute_geometric_stiffness_components(
        _mesh(nodes),
        _material(),
        {
            "bubble_linearization_policy": REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
            "membrane_compression": [3.0, 2.0, 0.1],
            "stress_second_moment": [0.02, 0.015, 0.001],
        },
    )
    assert np.linalg.norm(made["global"]) > 0.0

    tension = np.repeat(np.asarray(((17.0, -3.0, 2.0),)), 7, axis=0)
    recovery_shaped = {
        "membrane_forces_at_gauss": tension,
        "membrane_force_x": float(np.mean(tension[:, 0])),
        "membrane_force_y": float(np.mean(tension[:, 1])),
        "membrane_force_xy": float(np.mean(tension[:, 2])),
    }
    with pytest.raises(ValueError, match="explicit stress_second_moment"):
        _element().compute_geometric_stiffness_matrix(
            _mesh(nodes), _material(), recovery_shaped
        )
    authorized_homogeneous = dict(
        recovery_shaped,
        bubble_linearization_policy=(
            REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID
        ),
        through_thickness_stress_profile=(
            HOMOGENEOUS_ELASTIC_STRESS_PROFILE_ID
        ),
    )
    homogeneous = _element().compute_geometric_stiffness_components(
        _mesh(nodes), _material(), authorized_homogeneous
    )
    np.testing.assert_array_equal(
        homogeneous["stress_second_moment"],
        homogeneous["membrane_compression"] * 0.12**2 / 12.0,
    )
    recovery_shaped["stress_second_moment_at_gauss"] = np.repeat(
        np.asarray(((0.03, 0.02, 0.001),)), 7, axis=0
    )
    recovery_shaped["bubble_linearization_policy"] = (
        REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID
    )
    recovered = _element().compute_geometric_stiffness_components(
        _mesh(nodes), _material(), recovery_shaped
    )
    np.testing.assert_array_equal(recovered["membrane_compression"], -tension)


@pytest.mark.parametrize(
    "state",
    (
        {"membrane_compression": [1.0, 2.0]},
        {"membrane_compression_at_gauss": [1.0, 2.0, 3.0]},
        {
            "membrane_compression": [1.0, 2.0, 0.0],
            "membrane_forces": [-1.0, -2.0, 0.0],
        },
        {
            "bending_compression": [1.0, 2.0, 0.0],
            "bending_moments": [-1.0, -2.0, 0.0],
        },
        {
            "stress_second_moment": [1.0, 2.0, 0.0],
            "membrane_compression_second_moment": [1.0, 2.0, 0.0],
        },
        {
            "membrane_compression_at_gauss": np.ones((7, 3)),
        },
        {
            "membrane_compression": [1.0, 2.0, 0.0],
            "through_thickness_stress_profile": "UNBOUND_PROFILE",
        },
        {
            "membrane_compression": [1.0, 2.0, 0.0],
            "stress_second_moment": [0.01, 0.01, 0.0],
        },
        {
            "bubble_linearization_policy": "CURRENT_PLASTIC_TANGENT",
            "membrane_compression": [1.0, 2.0, 0.0],
            "stress_second_moment": [0.01, 0.01, 0.0],
        },
        {
            "resultant_summary_policy": "UNWEIGHTED_STATION_MEAN",
        },
        {
            "membrane_forces_at_gauss": np.repeat(
                np.asarray(((1.0, 2.0, 3.0),)), 7, axis=0
            ),
            "membrane_force_x": 9.0,
            "stress_second_moment": [0.01, 0.01, 0.0],
            "bubble_linearization_policy": (
                REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID
            ),
        },
    ),
)
def test_malformed_or_ambiguous_recognized_resultants_fail_closed(state) -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)))
    with pytest.raises(
        ValueError,
        match="finite|ambiguous|explicit|incompatible|bubble_linearization|inconsistent",
    ):
        _element().compute_geometric_stiffness_matrix(
            _mesh(nodes), _material(), state
        )


def test_state_contract_is_validated_before_stiffness_evaluation(
) -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)))
    element = _element()
    derived_names = (
        "_qualified_components",
        "_qualified_cache_key",
        "_qualified_component_guard",
        "_stiffness_matrix",
        "_internal_forces",
    )
    assert all(getattr(element, name) is None for name in derived_names)
    plan_revision = element._qualified_plan_state_revision
    with pytest.raises(ValueError, match="finite"):
        element.compute_geometric_stiffness_matrix(
            _mesh(nodes),
            _material(),
            {"membrane_compression": [1.0, 2.0]},
        )
    assert all(getattr(element, name) is None for name in derived_names)
    assert element._qualified_plan_state_revision == plan_revision


@pytest.mark.parametrize(
    "state",
    (
        {"membrane_compression": [1.0, float("nan"), 0.0]},
        {"bending_compression": [1.0, 2.0, float("inf")]},
        {"stress_second_moment": [1.0, 2.0, float("-inf")]},
    ),
)
def test_nonfinite_initial_stress_is_rejected_before_matrix_return(state) -> None:
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.3, 1.1, 0.0)))
    with pytest.raises(ValueError, match="finite"):
        _element().compute_geometric_stiffness_matrix(
            _mesh(nodes), _material(), state
        )
