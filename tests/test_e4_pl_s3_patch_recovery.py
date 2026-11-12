"""Focused qualification evidence for the native S3 recovery slice."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from anyfesolver import FEModel, PatchRecoveryConfig
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.corotational import rotation_matrix_from_vector
from anyfesolver.e4_pl_s3_element import (
    QualifiedE4PLS3ShellElement,
    triangle_frame,
)
from anyfesolver.e4_pl_s3_state import (
    GENERALIZED_NONLINEAR_STATE_SCHEMA,
    seal_committed_s3_state,
)
from anyfesolver.elements import ShellElement
from anyfesolver.plasticity import lobatto_layers
from anyfesolver.recovery import (
    _qualified_s3_current_physical_frame,
    _recover_qualified_s3_committed_state,
    recover_shell_patch_stresses,
    recover_stress_result,
)
from anyfesolver.results import _gauss_to_node_extrapolation
from anyfesolver.shell_sections import GeneralizedShellSection


_SURFACE_COMPONENTS = ("xx", "yy", "zz", "xy", "yz", "xz")
_SURFACE_KEYS = tuple(
    f"global_{component}_{surface}"
    for surface in ("top", "bot")
    for component in _SURFACE_COMPONENTS
)


def _single_s3(
    *,
    shell_section: GeneralizedShellSection | None = None,
    element_type: type[QualifiedE4PLS3ShellElement] = QualifiedE4PLS3ShellElement,
) -> tuple[FEModel, QualifiedE4PLS3ShellElement]:
    model = FEModel("qualified_s3_recovery")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        ((0.0, 0.0, 0.0), (1.2, 0.1, 0.0), (0.2, 0.9, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    element = element_type(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.2,
        shell_section=shell_section,
        material_direction=(1.0, 0.0, 0.0) if shell_section is not None else None,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)
    return model, element


def _state_with_station_fields(
    model: FEModel,
    element: QualifiedE4PLS3ShellElement,
) -> dict[str, object]:
    material = model.get_material("steel")
    state = element.init_model_bound_nonlinear_state(model.mesh, material, 3)
    z, weights = lobatto_layers(3, float(element.thickness))
    stresses = np.zeros((7, 3, 3), dtype=float)
    resultants = np.zeros((7, 8), dtype=float)
    for station, (r, s) in enumerate(np.asarray(element.gauss_points, dtype=float)):
        base = np.asarray(
            (10.0 + 3.0 * r - 2.0 * s, -4.0 + r + s, 2.0 - r),
            dtype=float,
        )
        slope = np.asarray((8.0 - r, -3.0 + s, 1.5 + r - s), dtype=float)
        stresses[station] = base[None, :] + z[:, None] * slope[None, :]
        resultants[station, :3] = np.einsum(
            "l,li->i", weights, stresses[station]
        )
        resultants[station, 3:6] = np.einsum(
            "l,l,li->i", weights, z, stresses[station]
        )
        resultants[station, 6:] = (0.7 + 0.2 * r, -0.4 + 0.1 * s)
    mutable = dict(state)
    mutable["layer_stress"] = stresses.reshape(-1, 3)
    mutable["layer_stress_material"] = stresses.reshape(-1, 3).copy()
    mutable["station_generalized_resultant"] = resultants
    return seal_committed_s3_state(mutable)


def _linear_global_surface_stresses(element: ShellElement, model: FEModel) -> dict[str, np.ndarray]:
    coords = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
    positions = []
    for r, s in np.asarray(element.gauss_points, dtype=float):
        shape, _dr, _ds = element.compute_shape_functions(float(r), float(s))
        positions.append(np.asarray(shape, dtype=float) @ coords)
    positions = np.asarray(positions, dtype=float)
    linear = 13.0 + 2.5 * positions[:, 0] - 1.75 * positions[:, 1]
    return {
        key: (
            linear.copy()
            if key.startswith("global_xx_")
            else np.zeros(linear.shape, dtype=float)
        )
        for key in _SURFACE_KEYS
    }


def _mixed_patch_model(
    *,
    s3_nodes: tuple[int, int, int] = (2, 5, 3),
    second_material: bool = False,
) -> tuple[FEModel, QualifiedE4PLShellElement, QualifiedE4PLS3ShellElement]:
    model = FEModel("mixed_q4_s3_patch")
    model.add_material("steel", 210.0e9, 0.3)
    if second_material:
        model.add_material("other", 195.0e9, 0.28)
    coordinates = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (1.0, 1.0, 0.0),
        4: (0.0, 1.0, 0.0),
        5: (2.0, 0.0, 0.0),
    }
    for node_id, coordinate in coordinates.items():
        model.add_node(node_id, *coordinate)
    q4 = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "steel", thickness=0.1)
    s3 = QualifiedE4PLS3ShellElement(
        2,
        list(s3_nodes),
        "other" if second_material else "steel",
        thickness=0.1,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, q4)
    model.add_element(2, s3)
    return model, q4, s3


def test_native_seven_to_three_operator_reproduces_barycentric_linear_fields() -> None:
    model, element = _single_s3()
    del model
    operator = _gauss_to_node_extrapolation(element)
    assert operator is not None
    assert operator.shape == (3, 7)
    points = np.asarray(element.gauss_points, dtype=float)
    station_barycentric = np.column_stack(
        (1.0 - points[:, 0] - points[:, 1], points[:, 0], points[:, 1])
    )
    np.testing.assert_allclose(
        operator @ np.ones(7),
        np.ones(3),
        rtol=0.0,
        atol=8.0 * np.finfo(float).eps,
    )
    np.testing.assert_allclose(
        operator @ station_barycentric,
        np.eye(3),
        rtol=0.0,
        atol=32.0 * np.finfo(float).eps,
    )
    nodal_field = np.asarray((3.5, -2.0, 8.25), dtype=float)
    for permutation in itertools.permutations(range(3)):
        permuted = nodal_field[np.asarray(permutation, dtype=int)]
        recovered = operator @ (station_barycentric @ permuted)
        np.testing.assert_allclose(
            recovered,
            permuted,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps,
        )


def test_native_operator_is_never_selected_for_legacy_tri3() -> None:
    legacy = ShellElement(1, [1, 2, 3], "steel")
    assert _gauss_to_node_extrapolation(legacy) is None


def test_current_physical_frame_and_tensor_transport_are_full_d3_covariant() -> None:
    reference = np.asarray(
        ((0.1, 0.2, 0.3), (1.2, 0.1, 0.4), (0.2, 1.0, 0.35)),
        dtype=float,
    )
    owner_normal = np.cross(reference[1] - reference[0], reference[2] - reference[0])
    owner_normal /= np.linalg.norm(owner_normal)
    rotation = rotation_matrix_from_vector(np.asarray((0.3, -0.2, 0.4)))
    physical_tensor = np.asarray(
        ((12.0, 3.0, 0.0), (3.0, -4.0, 0.0), (0.0, 0.0, 0.0)),
        dtype=float,
    )
    expected_global = rotation @ physical_tensor @ rotation.T

    for permutation in itertools.permutations(range(3)):
        permuted = reference[np.asarray(permutation, dtype=int)]
        reference_frame, _local, _quality = triangle_frame(
            permuted, owner_normal
        )
        current = (rotation @ permuted.T).T
        current_frame = _qualified_s3_current_physical_frame(
            permuted,
            reference_frame,
            current,
            np.repeat((rotation @ owner_normal)[None, :], 3, axis=0),
        )
        np.testing.assert_allclose(
            current_frame,
            rotation @ reference_frame,
            rtol=0.0,
            atol=4.0e-15,
        )
        local_tensor = reference_frame.T @ physical_tensor @ reference_frame
        transported = current_frame @ local_tensor @ current_frame.T
        np.testing.assert_allclose(
            transported,
            expected_global,
            rtol=0.0,
            atol=2.0e-14,
        )


def test_committed_native_history_uses_station_resultants_and_outer_layers() -> None:
    model, element = _single_s3()
    state = _state_with_station_fields(model, element)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)

    recovered, reason, sources = _recover_qualified_s3_committed_state(
        model,
        1,
        element,
        state,
        displacements=displacement,
        return_global=True,
    )

    assert reason == ""
    assert recovered is not None
    expected_resultants = np.asarray(state["station_generalized_resultant"])
    assert np.array_equal(recovered["membrane_resultants"], expected_resultants[:, :3])
    assert np.array_equal(recovered["bending_resultants"], expected_resultants[:, 3:6])
    assert np.array_equal(
        recovered["transverse_shear_resultants"], expected_resultants[:, 6:]
    )
    layers = np.asarray(state["layer_stress"]).reshape(7, 3, 3)
    np.testing.assert_allclose(recovered["local_xx_bot"], layers[:, 0, 0])
    np.testing.assert_allclose(recovered["local_xx_top"], layers[:, -1, 0])
    assert recovered["numerical_fields_excluded"] is True
    assert "committed_pl_internal_force" not in recovered
    assert "committed_internal_force" not in recovered
    assert sources["numerical_pl"] == "excluded_from_physical_recovery"
    assert sources["stress_frame"] == (
        "committed_s3_objective_current_physical_frame"
    )


def test_unified_recovery_dispatch_uses_native_history_when_gates_are_isolated() -> None:
    model, element = _single_s3()
    state = _state_with_station_fields(model, element)
    result = recover_stress_result(
        model,
        np.zeros(model.mesh.dof_manager.total_dofs, dtype=float),
        element_states={1: state},
        return_global=True,
        patch_config=PatchRecoveryConfig(include_error_indicator=True),
    )

    assert result.provenance.mode == "material_history"
    assert result.provenance.per_element_source[1] == (
        "committed_qualified_s3_native_state"
    )
    assert result.element_stresses[1]["recovery_history_source"] == (
        "committed_s3_station7_state"
    )
    assert result.nodal_stresses is not None
    assert result.nodal_stresses["discontinuous_node_ids"] == []
    assert result.nodal_stresses["error_indicator"]["status"] == "available"


def test_generalized_v3_state_exposes_resultants_without_fabricating_stress() -> None:
    section = GeneralizedShellSection(
        A=np.diag((1.0e8, 8.0e7, 3.0e7)),
        B=np.zeros((3, 3)),
        D=np.diag((1.0e5, 8.0e4, 3.0e4)),
        As=np.diag((2.0e7, 1.5e7)),
    )
    model, element = _single_s3(shell_section=section)
    state = element.init_model_bound_nonlinear_state(
        model.mesh,
        model.get_material("steel"),
        3,
    )
    assert "material_nonlinearity" not in element.capability_gaps
    assert element.capability_matrix()["material_nonlinearity"] == (
        "NOT_APPLICABLE_STATELESS_FIXED_SECTION"
    )
    assert "patch_recovery" not in element.capability_gaps
    assert element.capability_matrix()["patch_recovery"] == (
        "NOT_APPLICABLE_NO_PHYSICAL_LAYER_FIELD"
    )
    assert "nonlinear_geometry" not in element.capability_gaps
    assert element.capability_matrix()[
        "stateless_generalized_section_nonlinear_geometry"
    ] == "PARITY_REPLACED"
    assert state["state_schema"] == GENERALIZED_NONLINEAR_STATE_SCHEMA
    assert "layer_stress" not in state
    assert state["physical_layer_recovery_available"] is False

    recovered, reason, sources = _recover_qualified_s3_committed_state(
        model,
        1,
        element,
        state,
        displacements=np.zeros(model.mesh.dof_manager.total_dofs),
        return_global=True,
    )

    assert reason == ""
    assert recovered is not None
    assert recovered["recovery_scope"] == "section_resultants_only"
    assert recovered["physical_stress_available"] is False
    assert recovered["generalized_stress_scope"] == "section_resultants_only"
    np.testing.assert_array_equal(
        recovered["membrane_resultants"], np.zeros((7, 3), dtype=float)
    )
    assert "global_membrane_resultant_tensors" in recovered
    assert not any(key.startswith("global_xx_top") for key in recovered)
    assert sources["physical_stress"] == "unavailable_from_preintegrated_section"


def test_mixed_qualified_q4_s3_patch_is_deterministic_and_linear_exact() -> None:
    model, q4, s3 = _mixed_patch_model()
    stresses = {
        1: _linear_global_surface_stresses(q4, model),
        2: _linear_global_surface_stresses(s3, model),
    }
    settings = PatchRecoveryConfig(include_error_indicator=True)

    first = recover_shell_patch_stresses(model, stresses, config=settings)
    second = recover_shell_patch_stresses(model, stresses, config=settings)

    assert first["method"] == second["method"]
    assert first["qualified_node_ids"] == second["qualified_node_ids"]
    assert first["fallback_node_ids"] == second["fallback_node_ids"]
    assert first["discontinuous_node_ids"] == []
    for node_id, node in model.mesh.nodes.items():
        expected = 13.0 + 2.5 * node.x - 1.75 * node.y
        assert first["nodal"][node_id]["global_xx_top"] == pytest.approx(
            expected, rel=0.0, abs=2.0e-13
        )
        assert second["nodal"][node_id]["global_xx_top"] == pytest.approx(
            first["nodal"][node_id]["global_xx_top"], rel=0.0, abs=0.0
        )
    assert first["error_indicator"]["status"] == "available"
    assert first["error_indicator"]["is_energy_norm_estimate"] is False
    assert first["error_indicator"]["relative"] == pytest.approx(
        0.0, abs=2.0e-15
    )


def test_mixed_patch_d3_reexpression_and_material_discontinuity_are_guarded() -> None:
    reference_model, q4, s3 = _mixed_patch_model(s3_nodes=(2, 5, 3))
    reference = recover_shell_patch_stresses(
        reference_model,
        {
            1: _linear_global_surface_stresses(q4, reference_model),
            2: _linear_global_surface_stresses(s3, reference_model),
        },
    )
    for connectivity in itertools.permutations((2, 5, 3)):
        if connectivity == (2, 5, 3):
            continue
        model, q4_reexpressed, s3_reexpressed = _mixed_patch_model(
            s3_nodes=connectivity
        )
        recovered = recover_shell_patch_stresses(
            model,
            {
                1: _linear_global_surface_stresses(q4_reexpressed, model),
                2: _linear_global_surface_stresses(s3_reexpressed, model),
            },
        )
        for node_id in reference["nodal"]:
            assert recovered["nodal"][node_id]["global_xx_top"] == pytest.approx(
                reference["nodal"][node_id]["global_xx_top"],
                rel=0.0,
                abs=3.0e-13,
            )

    split_model, split_q4, split_s3 = _mixed_patch_model(second_material=True)
    split = recover_shell_patch_stresses(
        split_model,
        {
            1: _linear_global_surface_stresses(split_q4, split_model),
            2: _linear_global_surface_stresses(split_s3, split_model),
        },
    )
    for shared_node in (2, 3):
        assert shared_node in split["discontinuous_node_ids"]
        assert split["node_diagnostics"][shared_node]["reason"] == (
            "material_discontinuity"
        )
        assert len(split["nodal_regions"][shared_node]) == 2
