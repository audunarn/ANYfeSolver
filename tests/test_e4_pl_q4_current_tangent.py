from __future__ import annotations

import copy

import numpy as np
import pytest

from anyfesolver.e4_pl_element import (
    Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID,
    Q4_CURRENT_STATE_BINDING_SCHEMA_ID,
    Q4_CURRENT_STATE_PROJECTION_POLICY_ID,
    Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID,
    QualifiedE4PLShellElement,
    _q4_compact_replay_last_place_equivalent,
)
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from anyfesolver.elements import ShellElement
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.materials import Hill48Yield, OrthotropicMaterial
from anyfesolver.nonlinear_state import ShellStateBatch, ShellStateLayout
from anyfesolver.shell_sections import GeneralizedShellSection
from anyfesolver.vectorized_nonlinear import batch_shell_nonlinear_response
from anyfesolver import _q4_assumed_strain_plasticity as _q4_assumed_strain


def _production_q4_kernel(element, mesh, material):
    """Batch kernel the assembly plans use for this element and material."""

    if not _q4_assumed_strain.supported(element, mesh, material):
        return batch_shell_nonlinear_response
    offsets = _q4_assumed_strain.offset_operator(element, mesh, material)

    def kernel(*arguments):
        count = np.asarray(arguments[0]).shape[0]
        return _q4_assumed_strain.kernel_batch(
            *arguments, np.broadcast_to(offsets, (count,) + offsets.shape)
        )[:5]

    return kernel


D4 = (
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (1, 0, 3, 2),
    (3, 2, 1, 0),
    (0, 3, 2, 1),
    (2, 1, 0, 3),
)


def test_compact_q4_replay_accepts_only_the_frozen_last_place_envelope() -> None:
    expected = np.asarray((1.0, 0.5, 2.0), dtype=np.float64)
    accepted_bits = expected.view(np.uint64).copy()
    accepted_bits[0] += np.uint64(262144)
    accepted = accepted_bits.view(np.float64)
    assert _q4_compact_replay_last_place_equivalent(accepted, expected)

    rejected_bits = expected.view(np.uint64).copy()
    rejected_bits[0] += np.uint64(262145)
    rejected = rejected_bits.view(np.float64)
    assert not _q4_compact_replay_last_place_equivalent(rejected, expected)
    assert not _q4_compact_replay_last_place_equivalent(
        np.asarray((np.nan,)), np.asarray((np.nan,))
    )
    assert not _q4_compact_replay_last_place_equivalent(
        np.asarray((1.0,)), np.asarray((1.0, 2.0))
    )


def _mesh(nodes: np.ndarray) -> FEMesh:
    mesh = FEMesh()
    for node_id, coordinates in enumerate(np.asarray(nodes, dtype=float), start=1):
        mesh.add_node(node_id, *coordinates)
    return mesh


def _material(*, plastic: bool = False) -> Material:
    curve = None
    if plastic:
        curve = DNVC208MaterialCurve(
            sigma_prop=100.0e6,
            sigma_yield=105.0e6,
            sigma_yield_2=110.0e6,
            eps_p_y1=0.005,
            eps_p_y2=0.010,
            K=400.0e6,
            n=0.20,
        )
    return Material(
        "q4-current",
        210.0e9,
        0.3,
        density=7850.0,
        yield_stress=100.0e6 if plastic else 0.0,
        hardening_curve=curve,
    )


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(
        float(np.linalg.norm(left, ord="fro")),
        float(np.linalg.norm(right, ord="fro")),
        1.0,
    )
    return float(np.linalg.norm(left - right, ord="fro") / scale)


def _response_and_seal(
    element: QualifiedE4PLShellElement,
    mesh: FEMesh,
    material: Material,
    displacement: np.ndarray,
    *,
    num_layers: int = 3,
) -> dict[str, object]:
    _force, _tangent, state = element.compute_nonlinear_response(
        mesh,
        material,
        displacement,
        None,
        num_layers,
        True,
    )
    assert isinstance(state, dict)
    return element.seal_committed_current_tangent_state(
        mesh,
        material,
        displacement,
        state,
        num_layers,
    )


def test_layered_q4_components_are_independent_read_only_and_newton_exact() -> None:
    nodes = np.asarray(
        ((-0.2, -0.1, 0.0), (1.3, 0.0, 0.0), (1.1, 0.9, 0.0), (-0.1, 0.8, 0.0)),
        dtype=float,
    )
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(
        17,
        [1, 2, 3, 4],
        material.name,
        thickness=0.025,
        reference_normal=(0.0, 0.0, 1.0),
    )
    displacement = np.asarray(
        (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -2.0e-4, 0.5e-4, 2.0e-4, 0.0, 1.0e-4, 0.0,
            -2.5e-4, 1.2e-4, 5.0e-4, -0.8e-4, 1.3e-4, 0.0,
            0.2e-4, 0.8e-4, -1.0e-4, -0.4e-4, 0.0, 0.0,
        ),
        dtype=np.float64,
    )
    state = _response_and_seal(element, mesh, material, displacement)
    frozen = canonical_json_bytes(state)
    components = element.compute_committed_current_tangent_components(
        mesh, material, displacement, state, 3
    )
    ordinary_force, ordinary_total, _state = element.compute_nonlinear_response(
        mesh, material, displacement, state, 3, True
    )
    assert ordinary_total is not None
    np.testing.assert_array_equal(components["force"], ordinary_force)
    np.testing.assert_array_equal(components["total"], ordinary_total)
    np.testing.assert_allclose(
        np.asarray(components["material"]) + np.asarray(components["geometric"]),
        ordinary_total,
        rtol=4.0e-15,
        atol=2.0e-7,
    )
    assert np.linalg.norm(np.asarray(components["material"]), ord="fro") > 0.0
    assert np.linalg.norm(np.asarray(components["geometric"]), ord="fro") > 0.0
    assert components["state_binding_verified"] is True
    assert components["decomposition_policy_id"] == (
        Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
    )
    assert components["projection_policy_id"] == (
        Q4_CURRENT_STATE_PROJECTION_POLICY_ID
    )
    assert state["qualified_q4_committed_binding"]["schema_id"] == (
        Q4_CURRENT_STATE_BINDING_SCHEMA_ID
    )
    assert canonical_json_bytes(state) == frozen
    for name in ("force", "material", "geometric", "total"):
        assert not np.asarray(components[name]).flags.writeable

    direction = np.cos(np.linspace(0.1, 2.0, 24))
    direction /= np.linalg.norm(direction)
    step = 2.0e-8
    plus = element.compute_nonlinear_response(
        mesh, material, displacement + step * direction, state, 3, False
    )[0]
    minus = element.compute_nonlinear_response(
        mesh, material, displacement - step * direction, state, 3, False
    )[0]
    np.testing.assert_allclose(
        np.asarray(components["total"]) @ direction,
        (plus - minus) / (2.0 * step),
        rtol=2.0e-6,
        atol=2.0,
    )


def test_zero_state_has_exact_zero_stress_hessian_and_correction_is_material_only() -> None:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.1, 0.0, 0.0), (1.0, 0.9, 0.0), (-0.1, 0.8, 0.0)),
        dtype=float,
    )
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(
        18, [1, 2, 3, 4], material.name, thickness=0.025
    )
    displacement = np.zeros(24, dtype=np.float64)
    state = _response_and_seal(element, mesh, material, displacement)
    components = element.compute_committed_current_tangent_components(
        mesh, material, displacement, state, 3
    )
    element._hourglass_stiffness_matrix = None
    inherited_force, inherited_total, _trial, inherited = (
        ShellElement.compute_nonlinear_response(
            element,
            mesh,
            material,
            displacement,
            state,
            3,
            True,
            _return_tangent_components=True,
        )
    )
    correction = element._qualified_linear_correction(mesh, material, 3)
    np.testing.assert_array_equal(inherited_force, np.zeros(24))
    np.testing.assert_array_equal(components["geometric"], np.zeros((24, 24)))
    np.testing.assert_allclose(
        components["material"],
        np.asarray(inherited["material"]) + correction,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        components["qualified_linear_material_correction"], correction
    )
    np.testing.assert_array_equal(components["total"], inherited_total + correction)
    np.testing.assert_allclose(
        components["total"],
        element.compute_stiffness_matrix(mesh, material),
        rtol=2.0e-15,
        atol=1.0e-7,
    )


def _direct_membrane_stress_hessian(
    element: QualifiedE4PLShellElement,
    mesh: FEMesh,
    membrane_resultants: np.ndarray,
) -> np.ndarray:
    cache = element._nonlinear_geometry(mesh)
    local = np.zeros((24, 24), dtype=np.float64)
    for resultant, gradient, weight in zip(
        np.asarray(membrane_resultants, dtype=np.float64),
        np.asarray(cache["Gw_all"], dtype=np.float64),
        np.asarray(cache["detw_all"], dtype=np.float64),
    ):
        stress = np.asarray(
            ((resultant[0], resultant[2]), (resultant[2], resultant[1])),
            dtype=np.float64,
        )
        local += gradient.T @ stress @ gradient * float(weight)
    transform = np.asarray(cache["T0"], dtype=np.float64)
    return transform.T @ local @ transform


def test_q4_stress_hessian_matches_direct_gw_n_gw_and_reverses_with_stress() -> None:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (1.1, 0.8, 0.0), (-0.1, 0.7, 0.0)),
        dtype=float,
    )
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(
        19,
        [1, 2, 3, 4],
        material.name,
        shell_section=_generalized_section(),
        reference_normal=(0.0, 0.0, 1.0),
    )

    def evaluate(sign: float) -> tuple[dict[str, object], dict[str, object]]:
        displacement = np.zeros(24, dtype=np.float64)
        for node, (x, y, _z) in enumerate(nodes):
            displacement[6 * node] = sign * 2.0e-4 * x
            displacement[6 * node + 1] = sign * 0.7e-4 * y
        state = _response_and_seal(
            element, mesh, material, displacement, num_layers=3
        )
        components = dict(
            element.compute_committed_current_tangent_components(
                mesh, material, displacement, state, 3
            )
        )
        return state, components

    tension_state, tension = evaluate(1.0)
    compression_state, compression = evaluate(-1.0)
    oracle = _direct_membrane_stress_hessian(
        element,
        mesh,
        np.asarray(tension_state["membrane_resultants"], dtype=np.float64),
    )
    np.testing.assert_allclose(
        tension["geometric"], oracle, rtol=3.0e-15, atol=2.0e-12
    )
    np.testing.assert_allclose(
        compression_state["membrane_resultants"],
        -np.asarray(tension_state["membrane_resultants"]),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        compression["geometric"],
        -np.asarray(tension["geometric"]),
        rtol=2.0e-14,
        atol=2.0e-12,
    )


def test_q4_binding_rejects_every_configuration_and_state_mismatch() -> None:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (1.1, 0.8, 0.0), (-0.1, 0.7, 0.0)),
        dtype=float,
    )
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(
        3, [1, 2, 3, 4], material.name, thickness=0.02
    )
    displacement = np.linspace(-2.0e-5, 3.0e-5, 24)
    state = _response_and_seal(element, mesh, material, displacement)
    digest = element.validate_committed_current_tangent_state(
        mesh, material, displacement, state, 3
    )
    assert digest == state["state_integrity_sha256"]

    bad_u = displacement.copy()
    bad_u[7] += 1.0e-12
    with pytest.raises(ValueError, match="displacement"):
        element.validate_committed_current_tangent_state(
            mesh, material, bad_u, state, 3
        )
    bad_state = copy.deepcopy(state)
    bad_state["plastic_strain"][0, 0] += 1.0e-12
    with pytest.raises(ValueError, match="binding"):
        element.validate_committed_current_tangent_state(
            mesh, material, displacement, bad_state, 3
        )
    bad_hash = copy.deepcopy(state)
    bad_hash["state_integrity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="integrity|binding"):
        element.validate_committed_current_tangent_state(
            mesh, material, displacement, bad_hash, 3
        )
    changed_material = Material(material.name, 205.0e9, 0.3, density=7850.0)
    with pytest.raises(ValueError, match="binding"):
        element.validate_committed_current_tangent_state(
            mesh, changed_material, displacement, state, 3
        )
    changed_element = QualifiedE4PLShellElement(
        3, [1, 2, 3, 4], material.name, thickness=0.021
    )
    with pytest.raises(ValueError, match="binding"):
        changed_element.validate_committed_current_tangent_state(
            mesh, material, displacement, state, 3
        )
    moved_mesh = _mesh(nodes + np.asarray((0.0, 0.0, 1.0e-7)))
    with pytest.raises(ValueError, match="binding"):
        element.validate_committed_current_tangent_state(
            moved_mesh, material, displacement, state, 3
        )
    with pytest.raises(TypeError, match="does not accept"):
        element.compute_committed_current_tangent_components(
            mesh,
            material,
            displacement,
            state,
            3,
            native_rotation_trial=object(),
        )


def test_plastic_q4_origin_is_required_and_reproduces_the_accepted_core() -> None:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    mesh = _mesh(nodes)
    material = _material(plastic=True)
    element = QualifiedE4PLShellElement(
        8, [1, 2, 3, 4], material.name, thickness=0.02
    )
    displacement = np.zeros(24, dtype=np.float64)
    displacement[6] = displacement[12] = 0.004
    _force, _tangent, trial = element.compute_nonlinear_response(
        mesh, material, displacement, None, 3, True
    )
    assert isinstance(trial, dict)
    assert trial["qualified_q4_algorithmic_origin"]["schema_id"] == (
        Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
    )

    missing = copy.deepcopy(trial)
    missing.pop("qualified_q4_algorithmic_origin")
    with pytest.raises(ValueError, match="lacks the accepted algorithmic"):
        element.seal_committed_current_tangent_state(
            mesh, material, displacement, missing, 3
        )

    mutated = copy.deepcopy(trial)
    mutated["qualified_q4_algorithmic_origin"][
        "parent_plastic_strain"
    ][0][0] += 1.0e-10
    with pytest.raises(ValueError, match="does not reproduce committed"):
        element.seal_committed_current_tangent_state(
            mesh, material, displacement, mutated, 3
        )

    stale_recovery = copy.deepcopy(trial)
    stale_recovery["layer_stress"][0, 0] += 1.0
    with pytest.raises(ValueError, match="layer_stress"):
        element.seal_committed_current_tangent_state(
            mesh, material, displacement, stale_recovery, 3
        )

    sealed = element.seal_committed_current_tangent_state(
        mesh, material, displacement, trial, 3
    )
    changed_after_seal = copy.deepcopy(sealed)
    changed_after_seal["qualified_q4_algorithmic_origin"][
        "parent_alpha"
    ][0] += 1.0e-12
    with pytest.raises(ValueError, match="binding"):
        element.compute_committed_current_tangent_components(
            mesh, material, displacement, changed_after_seal, 3
        )


@pytest.mark.parametrize(
    "invalid_layers",
    (True, 0, 1, 2, 4, 6, 8, 10, 12, 3.0),
)
def test_q4_layered_state_boundaries_reject_unsupported_lobatto_counts(
    invalid_layers: object,
) -> None:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(
        9, [1, 2, 3, 4], material.name, thickness=0.02
    )
    displacement = np.zeros(24, dtype=np.float64)
    state = _response_and_seal(element, mesh, material, displacement)
    with pytest.raises(ValueError, match=r"\[3, 5, 7, 9, 11\]"):
        element.init_nonlinear_state(invalid_layers)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\[3, 5, 7, 9, 11\]"):
        element.compute_nonlinear_response(
            mesh,
            material,
            displacement,
            None,
            invalid_layers,  # type: ignore[arg-type]
            True,
        )
    with pytest.raises(ValueError, match=r"\[3, 5, 7, 9, 11\]"):
        element.seal_committed_current_tangent_state(
            mesh,
            material,
            displacement,
            state,
            invalid_layers,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r"\[3, 5, 7, 9, 11\]"):
        element.compute_committed_current_tangent_components(
            mesh,
            material,
            displacement,
            state,
            invalid_layers,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("layers", (3, 5, 7, 9, 11))
def test_q4_admits_exact_supported_lobatto_layer_set(layers: int) -> None:
    material = _material(plastic=True)
    element = QualifiedE4PLShellElement(
        10, [1, 2, 3, 4], material.name, thickness=0.02
    )
    state = element.init_nonlinear_state(layers)
    assert np.asarray(state["plastic_strain"]).shape == (4 * layers, 3)
    assert np.asarray(state["alpha"]).shape == (4 * layers,)


def _generalized_section() -> GeneralizedShellSection:
    return GeneralizedShellSection(
        A=np.asarray(((160.0, 17.0, 5.0), (17.0, 105.0, -4.0), (5.0, -4.0, 48.0))),
        B=np.asarray(((1.4, 0.2, -0.1), (0.2, -0.9, 0.15), (-0.1, 0.15, 0.35))),
        D=np.asarray(((18.0, 1.2, 0.3), (1.2, 12.0, -0.2), (0.3, -0.2, 5.5))),
        As=np.asarray(((28.0, 2.0), (2.0, 21.0))),
        mass_per_area=8.0,
        rotary_inertia_per_area=0.02,
    )


@pytest.mark.parametrize("director_polarity", (-1, 1))
def test_generalized_b_coupled_q4_component_split_is_all_d4_covariant(
    director_polarity: int,
) -> None:
    nodes = np.asarray(
        ((-0.1, 0.0, 0.02), (1.2, -0.1, -0.01), (1.1, 0.9, 0.04), (0.0, 0.8, -0.02)),
        dtype=float,
    )
    material = _material()
    section = _generalized_section()
    normal = np.cross(nodes[1] - nodes[0], nodes[3] - nodes[0])
    normal /= np.linalg.norm(normal)
    material_direction = nodes[1] - nodes[0]
    displacement = 7.0e-5 * np.sin(np.linspace(0.2, 3.4, 24))

    def evaluate(slots: tuple[int, ...]) -> dict[str, object]:
        numbered_nodes = nodes[np.asarray(slots, dtype=int)]
        numbered_u = displacement.reshape(4, 6)[np.asarray(slots, dtype=int)].reshape(24)
        mesh = _mesh(numbered_nodes)
        element = QualifiedE4PLShellElement(
            9,
            [1, 2, 3, 4],
            material.name,
            shell_section=section,
            material_direction=material_direction,
            reference_normal=normal,
            director_polarity=director_polarity,
        )
        state = _response_and_seal(
            element, mesh, material, numbered_u, num_layers=3
        )
        return dict(
            element.compute_committed_current_tangent_components(
                mesh, material, numbered_u, state, 3
            )
        )

    baseline = evaluate(D4[0])
    for operation in D4:
        actual = evaluate(operation)
        dofs = np.asarray(
            [6 * node + component for node in operation for component in range(6)],
            dtype=int,
        )
        for name in ("material", "geometric", "total"):
            expected = np.asarray(baseline[name])[np.ix_(dofs, dofs)]
            assert _relative(np.asarray(actual[name]), expected) <= 2.0e-10
        np.testing.assert_allclose(
            actual["force"],
            np.asarray(baseline["force"])[dofs],
            rtol=2.0e-10,
            atol=3.0e-11,
        )


def test_layered_plastic_q4_component_split_uses_algorithmic_material_tangent() -> None:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=float,
    )
    mesh = _mesh(nodes)
    material = _material(plastic=True)
    element = QualifiedE4PLShellElement(
        21, [1, 2, 3, 4], material.name, thickness=0.02
    )
    displacement = np.zeros(24, dtype=np.float64)
    displacement[6] = 0.004
    displacement[12] = 0.004
    displacement[8] = 4.0e-4
    displacement[14] = 7.0e-4
    accepted_force, accepted_tangent, trial = element.compute_nonlinear_response(
        mesh, material, displacement, None, 3, True
    )
    assert accepted_tangent is not None
    assert isinstance(trial, dict)
    assert np.max(np.asarray(trial["alpha"], dtype=float)) > 0.0
    state = element.seal_committed_current_tangent_state(
        mesh, material, displacement, trial, 3
    )
    frozen = canonical_json_bytes(state)
    components = element.compute_committed_current_tangent_components(
        mesh, material, displacement, state, 3
    )
    np.testing.assert_array_equal(components["force"], accepted_force)
    np.testing.assert_array_equal(components["total"], accepted_tangent)
    assert components["algorithmic_origin_verified"] is True
    assert components["algorithmic_origin_schema_id"] == (
        Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
    )
    assert np.linalg.norm(np.asarray(components["geometric"]), ord="fro") > 0.0
    assert components["relative_decomposition_error"] <= (
        512.0 * np.finfo(np.float64).eps
    )
    direction = np.zeros(24, dtype=np.float64)
    direction[6] = 1.0
    direction[12] = 1.0
    direction /= np.linalg.norm(direction)
    step = 2.0e-8
    origin = state["qualified_q4_algorithmic_origin"]
    parent = {
        "plastic_strain": np.asarray(
            origin["parent_plastic_strain"], dtype=np.float64
        ).copy(),
        "alpha": np.asarray(origin["parent_alpha"], dtype=np.float64).copy(),
    }
    loading_minus_force = element.compute_nonlinear_response(
        mesh,
        material,
        displacement - step * direction,
        parent,
        3,
        False,
    )[0]
    np.testing.assert_allclose(
        np.asarray(components["total"]) @ direction,
        (np.asarray(components["force"]) - loading_minus_force) / step,
        rtol=5.0e-5,
        atol=2.0e3,
    )
    # A subsequent negative increment starts from the now-committed plastic
    # state and selects the elastic unloading branch.  It is intentionally not
    # the prior accepted loading tangent retained for current-state buckling.
    unloading_force, unloading_tangent, _unloading_state = (
        element.compute_nonlinear_response(
            mesh,
            material,
            displacement,
            state,
            3,
            True,
        )
    )
    assert unloading_tangent is not None
    unloading_minus_force = element.compute_nonlinear_response(
        mesh,
        material,
        displacement - step * direction,
        state,
        3,
        False,
    )[0]
    np.testing.assert_allclose(
        np.asarray(unloading_tangent) @ direction,
        (np.asarray(unloading_force) - unloading_minus_force) / step,
        rtol=5.0e-5,
        atol=2.0e3,
    )
    assert _relative(
        np.asarray(components["material"]),
        np.asarray(unloading_tangent)
        - np.asarray(components["geometric"]),
    ) > 1.0e-3
    assert canonical_json_bytes(state) == frozen


def _orthotropic() -> OrthotropicMaterial:
    return OrthotropicMaterial(
        name="q4-orthotropic",
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
    )


def _orthotropic_plastic() -> OrthotropicMaterial:
    yield_stress = 100.0e6
    shear_yield = yield_stress / np.sqrt(3.0)
    return OrthotropicMaterial(
        name="q4-hill-current",
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


@pytest.mark.parametrize("initial_field", (False, True))
def test_isotropic_initial_field_and_orthotropic_hill_plastic_updates_are_refused(
    initial_field: bool,
) -> None:
    """Owner decision 2026-10-23: no silent compatible-strain plastic route.

    Active initial fields and Hill48/orthotropic plasticity have no stable
    qualified Q4 plastic route (assumed-strain route scope), so the element
    refuses them with a typed error instead of the historical unstable form.
    """

    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    mesh = _mesh(nodes)
    material = _material(plastic=True) if initial_field else _orthotropic_plastic()
    element = QualifiedE4PLShellElement(
        25,
        [1, 2, 3, 4],
        material.name,
        thickness=0.02,
        material_direction=(1.0, 0.25, 0.0),
        reference_normal=(0.0, 0.0, 1.0),
    )
    parent = element.init_nonlinear_state(3)
    if initial_field:
        parent["initial_membrane_prestrain"] = np.broadcast_to(
            np.asarray((1.0e-4, -0.5e-4, 0.25e-4)), (4, 3)
        ).copy()
    displacement = np.zeros(24, dtype=np.float64)
    displacement[6] = displacement[12] = 0.004
    match = "active initial" if initial_field else "orthotropic/Hill48"
    with pytest.raises(_q4_assumed_strain.UnsupportedQ4PlasticRouteError, match=match):
        element.compute_nonlinear_response(mesh, material, displacement, parent, 3, True)

def test_residual_only_plastic_candidate_cannot_masquerade_as_accepted_tangent() -> None:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    mesh = _mesh(nodes)
    material = _material(plastic=True)
    element = QualifiedE4PLShellElement(
        26, [1, 2, 3, 4], material.name, thickness=0.02
    )
    displacement = np.zeros(24, dtype=np.float64)
    displacement[6] = displacement[12] = 0.004
    _force, no_tangent, residual_state = element.compute_nonlinear_response(
        mesh, material, displacement, None, 3, False
    )
    assert no_tangent is None
    assert "qualified_q4_algorithmic_origin" not in residual_state
    with pytest.raises(ValueError, match="lacks the accepted algorithmic"):
        element.seal_committed_current_tangent_state(
            mesh, material, displacement, residual_state, 3
        )

    _force, tangent, accepted_state = element.compute_nonlinear_response(
        mesh, material, displacement, None, 3, True
    )
    assert tangent is not None
    sealed = element.seal_committed_current_tangent_state(
        mesh, material, displacement, accepted_state, 3
    )
    assert "qualified_q4_algorithmic_origin" in sealed


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    cross = np.asarray(
        (
            (0.0, -unit[2], unit[1]),
            (unit[2], 0.0, -unit[0]),
            (-unit[1], unit[0], 0.0),
        ),
        dtype=np.float64,
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(unit, unit)
        + np.sin(angle) * cross
    )


def test_orthotropic_q4_components_are_proper_global_covariant() -> None:
    nodes = np.asarray(
        ((-0.2, -0.1, 0.0), (1.3, 0.0, 0.0), (1.1, 0.9, 0.0), (-0.1, 0.8, 0.0)),
        dtype=np.float64,
    )
    material = _orthotropic()
    direction = np.asarray((1.0, 0.37, 0.0), dtype=np.float64)
    normal = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    displacement = 4.0e-5 * np.sin(np.linspace(0.1, 3.2, 24))

    def evaluate(
        coordinates: np.ndarray,
        material_direction: np.ndarray,
        owner_normal: np.ndarray,
        values: np.ndarray,
    ) -> dict[str, object]:
        mesh = _mesh(coordinates)
        element = QualifiedE4PLShellElement(
            27,
            [1, 2, 3, 4],
            material.name,
            thickness=0.018,
            material_direction=material_direction,
            reference_normal=owner_normal,
            director_polarity=-1,
        )
        state = _response_and_seal(
            element, mesh, material, values, num_layers=3
        )
        return dict(
            element.compute_committed_current_tangent_components(
                mesh, material, values, state, 3
            )
        )

    baseline = evaluate(nodes, direction, normal, displacement)
    rotation = _axis_rotation(np.asarray((0.3, -0.5, 0.8)), 0.63)
    rotated_nodes = nodes @ rotation.T
    rotated_direction = rotation @ direction
    rotated_normal = rotation @ normal
    rotated_u = displacement.reshape(4, 6).copy()
    rotated_u[:, :3] = rotated_u[:, :3] @ rotation.T
    rotated_u[:, 3:] = rotated_u[:, 3:] @ rotation.T
    actual = evaluate(
        rotated_nodes,
        rotated_direction,
        rotated_normal,
        rotated_u.reshape(24),
    )
    transport = np.zeros((24, 24), dtype=np.float64)
    for node in range(4):
        transport[6 * node : 6 * node + 3, 6 * node : 6 * node + 3] = rotation
        transport[6 * node + 3 : 6 * node + 6, 6 * node + 3 : 6 * node + 6] = rotation
    for name in ("material", "geometric", "total"):
        expected = transport @ np.asarray(baseline[name]) @ transport.T
        assert _relative(np.asarray(actual[name]), expected) <= 4.0e-10
    np.testing.assert_allclose(
        actual["force"],
        transport @ np.asarray(baseline["force"]),
        rtol=4.0e-10,
        atol=2.0e-5,
    )


def test_q4_seal_rejects_stored_generalized_kinematics_from_another_state() -> None:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (1.1, 0.8, 0.0), (-0.1, 0.7, 0.0)),
        dtype=float,
    )
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(
        31,
        [1, 2, 3, 4],
        material.name,
        shell_section=_generalized_section(),
        reference_normal=(0.0, 0.0, 1.0),
    )
    displacement = np.linspace(-1.0e-5, 2.0e-5, 24)
    _force, _tangent, state = element.compute_nonlinear_response(
        mesh, material, displacement, None, 3, True
    )
    assert isinstance(state, dict)
    state["membrane_strain"][0, 0] += 1.0e-12
    with pytest.raises(ValueError, match="kinematics"):
        element.seal_committed_current_tangent_state(
            mesh, material, displacement, state, 3
        )


def test_packed_scalar_q4_origin_survives_partial_commit_discard_and_delete() -> None:
    layout = ShellStateLayout.from_dimensions((1, 2), 4, 3)
    points = layout.points_per_element

    def state(value: float, parent: float) -> dict[str, object]:
        return {
            "plastic_strain": np.full((points, 3), value),
            "alpha": np.full(points, value),
            "layer_strain": np.full((points, 3), 2.0 * value),
            "qualified_q4_algorithmic_origin": {
                "schema_id": Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID,
                "kind": "LAYERED_DISCRETE_RETURN_MAP_PARENT_STATE",
                "num_layers": 3,
                "parent_plastic_strain": np.full((points, 3), parent),
                "parent_alpha": np.full(points, parent),
            },
        }

    packed = ShellStateBatch(layout, {1: state(1.0, 0.1), 2: state(2.0, 0.2)})
    assert packed.all_packed
    original_second = canonical_json_bytes(packed[2])

    partial = packed.begin_trial()
    packed.set_trial_state(partial, 1, state(3.0, 0.3))
    packed.commit(partial)
    assert canonical_json_bytes(packed[1]) == canonical_json_bytes(state(3.0, 0.3))
    assert canonical_json_bytes(packed[2]) == original_second

    committed_first = canonical_json_bytes(packed[1])
    rejected = packed.begin_trial()
    packed.set_trial_state(rejected, 1, state(4.0, 0.4))
    packed.discard_trial(rejected)
    assert canonical_json_bytes(packed[1]) == committed_first

    packed.freeze_deleted((1,))
    deletion = packed.begin_trial()
    packed.set_trial_state(deletion, 1, state(5.0, 0.5))
    packed.set_trial_state(deletion, 2, state(6.0, 0.6))
    packed.commit(deletion)
    assert canonical_json_bytes(packed[1]) == committed_first
    assert canonical_json_bytes(packed[2]) == canonical_json_bytes(state(6.0, 0.6))


def test_vectorized_q4_accepted_tangent_matches_sealed_scalar_replay_exactly() -> None:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    mesh = _mesh(nodes)
    material = _material(plastic=True)
    element = QualifiedE4PLShellElement(
        33, [1, 2, 3, 4], material.name, thickness=0.02
    )
    displacement = np.zeros(24, dtype=np.float64)
    displacement[6] = displacement[12] = 0.004
    displacement[8] = 4.0e-4
    displacement[14] = 7.0e-4
    cache = element._nonlinear_geometry(mesh)
    parent = element.init_nonlinear_state(3)
    force, tangent, plastic, alpha, layer_strain = (
        _production_q4_kernel(element, mesh, material)(
            displacement[None, :],
            np.asarray(cache["T0"])[None, :, :],
            np.asarray(cache["B_m_all"])[None, :, :, :],
            np.asarray(cache["B_b_all"])[None, :, :, :],
            np.asarray(cache["B_d_all"])[None, :, :, :],
            np.asarray(cache["Gw_all"])[None, :, :, :],
            np.asarray(cache["detw_all"])[None, :],
            np.asarray(cache["B_s_all"])[None, :, :, :],
            np.asarray(cache["detw_shear_all"])[None, :],
            float(material.elastic_modulus),
            float(material.poisson_ratio),
            float(material.shear_modulus),
            float(element.thickness),
            float(element.drilling_stabilization),
            True,
            material.hardening_curve,
            np.asarray(parent["plastic_strain"])[None, :, :],
            np.asarray(parent["alpha"])[None, :],
            3,
        )
    )
    trial = element.attach_current_tangent_algorithmic_origin(
        material,
        parent,
        {
            "plastic_strain": plastic[0],
            "alpha": alpha[0],
            "layer_strain": layer_strain.copy(),
        },
        3,
        tangent_evaluated=True,
    )
    canonical_plastic = np.asarray(plastic[0]).copy()
    perturbed = copy.deepcopy(trial)
    positive = int(np.flatnonzero(canonical_plastic > 0.0)[0])
    changed = canonical_plastic.ravel().copy()
    changed[positive] += 568728.0 * np.spacing(abs(changed[positive]))
    perturbed["plastic_strain"] = changed.reshape(canonical_plastic.shape)
    with pytest.raises(ValueError, match="does not reproduce committed"):
        element.seal_committed_current_tangent_state(
            mesh, material, displacement, perturbed, 3
        )
    sealed = element._seal_solver_finalized_current_tangent_state(
        mesh, material, displacement, perturbed, 3
    )
    np.testing.assert_array_equal(sealed["plastic_strain"], canonical_plastic)
    components = element.compute_committed_current_tangent_components(
        mesh, material, displacement, sealed, 3
    )
    correction = element._qualified_linear_correction(mesh, material, 3)
    np.testing.assert_array_equal(
        components["force"], force[0] + correction @ displacement
    )
    np.testing.assert_array_equal(components["total"], tangent[0] + correction)


def test_distorted_vectorized_q4_plastic_history_replays_after_multiple_increments() -> None:
    """Regression for final-state replay of the production batch path."""

    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (0.47, 0.01, 0.0), (0.45, 0.39, 0.0), (-0.02, 0.36, 0.0)),
        dtype=np.float64,
    )
    mesh = _mesh(nodes)
    material = _material(plastic=True)
    element = QualifiedE4PLShellElement(
        34,
        [1, 2, 3, 4],
        material.name,
        thickness=0.012,
        reference_normal=(0.0, 0.0, 1.0),
    )
    cache = element._nonlinear_geometry(mesh)
    batch_size = 16
    target = 11
    committed = [element.init_nonlinear_state(3) for _ in range(batch_size)]

    for scale in (0.35, 0.7, 1.0):
        displacement = np.zeros((batch_size, 24), dtype=np.float64)
        element_scales = np.linspace(0.91, 1.09, batch_size) * scale
        displacement[:, 6] = 0.004 * element_scales
        displacement[:, 12] = 0.0043 * element_scales
        displacement[:, 8] = 4.0e-4 * element_scales
        displacement[:, 14] = 7.0e-4 * element_scales
        production_kernel = _production_q4_kernel(element, mesh, material)
        force, tangent, plastic, alpha, layer_strain = production_kernel(
            displacement,
            np.broadcast_to(np.asarray(cache["T0"]), (batch_size, 24, 24)),
            np.broadcast_to(np.asarray(cache["B_m_all"]), (batch_size, 4, 3, 24)),
            np.broadcast_to(np.asarray(cache["B_b_all"]), (batch_size, 4, 3, 24)),
            np.broadcast_to(np.asarray(cache["B_d_all"]), (batch_size, 4, 1, 24)),
            np.broadcast_to(np.asarray(cache["Gw_all"]), (batch_size, 4, 2, 24)),
            np.broadcast_to(np.asarray(cache["detw_all"]), (batch_size, 4)),
            np.broadcast_to(np.asarray(cache["B_s_all"]), (batch_size, 4, 2, 24)),
            np.broadcast_to(np.asarray(cache["detw_shear_all"]), (batch_size, 4)),
            float(material.elastic_modulus),
            float(material.poisson_ratio),
            float(material.shear_modulus),
            float(element.thickness),
            float(element.drilling_stabilization),
            True,
            material.hardening_curve,
            np.asarray([state["plastic_strain"] for state in committed]),
            np.asarray([state["alpha"] for state in committed]),
            3,
        )
        single = production_kernel(
            displacement[target : target + 1],
            np.asarray(cache["T0"])[None, :, :],
            np.asarray(cache["B_m_all"])[None, :, :, :],
            np.asarray(cache["B_b_all"])[None, :, :, :],
            np.asarray(cache["B_d_all"])[None, :, :, :],
            np.asarray(cache["Gw_all"])[None, :, :, :],
            np.asarray(cache["detw_all"])[None, :],
            np.asarray(cache["B_s_all"])[None, :, :, :],
            np.asarray(cache["detw_shear_all"])[None, :],
            float(material.elastic_modulus),
            float(material.poisson_ratio),
            float(material.shear_modulus),
            float(element.thickness),
            float(element.drilling_stabilization),
            True,
            material.hardening_curve,
            np.asarray(committed[target]["plastic_strain"])[None, :, :],
            np.asarray(committed[target]["alpha"])[None, :],
            3,
        )
        np.testing.assert_array_equal(single[0][0], force[target])
        np.testing.assert_array_equal(single[1][0], tangent[target])
        np.testing.assert_array_equal(single[2][0], plastic[target])
        np.testing.assert_array_equal(single[3][0], alpha[target])
        np.testing.assert_array_equal(
            single[4], layer_strain[target * 12 : (target + 1) * 12]
        )
        next_committed = []
        points = 4 * 3
        for index in range(batch_size):
            trial = element.attach_current_tangent_algorithmic_origin(
                material,
                committed[index],
                {
                    "plastic_strain": plastic[index],
                    "alpha": alpha[index],
                    "layer_strain": layer_strain[
                        index * points : (index + 1) * points
                    ].copy(),
                },
                3,
                tangent_evaluated=True,
            )
            if index == target:
                trial = element.seal_committed_current_tangent_state(
                    mesh, material, displacement[index], trial, 3
                )
            next_committed.append(trial)
        committed = next_committed
        assert tangent.shape == (batch_size, 24, 24)
        assert force.shape == (batch_size, 24)
