"""Committed-state recovery for nonlinear generalized sections."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver import (
    BeamElement,
    FEModel,
    GeneralizedBeamSection,
    GeneralizedShellSection,
    PatchRecoveryConfig,
    ShellElement,
    create_shell_element,
    recover_prestress_from_static_result,
    recover_stress_result,
)


def _shell_model() -> tuple[FEModel, ShellElement]:
    model = FEModel("generalized_shell_state")
    model.add_material("dummy", 70.0e9, 0.25, density=2700.0)
    for node_id, coordinates in enumerate(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    section = GeneralizedShellSection(
        A=np.array(
            [
                [120.0, 18.0, 4.0],
                [18.0, 90.0, -3.0],
                [4.0, -3.0, 35.0],
            ]
        ),
        B=np.array(
            [
                [0.8, 0.1, 0.0],
                [0.05, -0.4, 0.08],
                [0.02, 0.0, 0.25],
            ]
        ),
        D=np.array(
            [
                [10.0, 0.8, 0.1],
                [0.8, 8.0, -0.1],
                [0.1, -0.1, 3.0],
            ]
        ),
        As=np.array([[20.0, 2.0], [2.0, 15.0]]),
        name="laminate_abd",
    )
    element = create_shell_element(
        1,
        [1, 2, 3, 4],
        "dummy",
        thickness=0.02,
        shell_section=section,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)
    return model, element


def _beam_model(*, corotational: bool = False) -> tuple[FEModel, BeamElement]:
    model = FEModel("generalized_beam_state")
    model.add_material("dummy", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 2.0, 0.0, 0.0)
    section = GeneralizedBeamSection(
        np.diag((2.0e8, 3.0e7, 4.0e7, 5.0e6, 6.0e6, 8.0e6)),
        name="coupled_section",
    )
    element = BeamElement(
        1,
        [1, 2],
        "dummy",
        (
            {"geometric_nonlinearity": "corotational"}
            if corotational
            else {}
        ),
        section=section,
    )
    model.add_element(1, element)
    return model, element


def _nonlinear_result(
    displacements: np.ndarray,
    state: dict,
    kinematics: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        displacements=np.asarray(displacements, dtype=float).copy(),
        element_states={1: state},
        info={"kinematics": kinematics},
        status="converged",
        load_factor=1.0,
    )


def test_generalized_shell_recovery_uses_exact_committed_von_karman_state() -> None:
    model, element = _shell_model()
    material = model.get_material("dummy")
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    # A transverse slope creates von Karman membrane strain even though the
    # linear membrane displacement field is zero.
    for node_id in element.node_ids:
        node = model.mesh.nodes[node_id]
        x, y, _z = node.coords()
        displacement[node.dofs[2]] = 0.06 * x - 0.025 * y
        displacement[node.dofs[3]] = 0.004 * y
        displacement[node.dofs[4]] = -0.003 * x
    mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    _force, _tangent, state = element.compute_nonlinear_response(
        model.mesh,
        material,
        displacement[mapping],
        tangent=True,
    )
    assert state is not None
    linear = element.compute_stresses(model.mesh, displacement, material)
    assert np.max(
        np.abs(
            np.asarray(state["membrane_strain"])
            - np.asarray(linear["membrane_strain"])
        )
    ) > 1.0e-4

    nonlinear = _nonlinear_result(displacement, state, "von_karman")
    result = recover_stress_result(model, nonlinear_result=nonlinear)
    recovered = result.element_stresses[1]
    for key in (
        "membrane_strain",
        "curvature",
        "transverse_shear_strain",
        "membrane_resultants",
        "bending_resultants",
        "transverse_shear_resultants",
    ):
        np.testing.assert_array_equal(recovered[key], state[key])
    assert recovered["physical_stress_available"] is False
    assert recovered["recovery_scope"] == "section_resultants_only"
    assert "von_mises" not in recovered
    assert "global_xx_top" not in recovered
    assert result.provenance.per_element_source[1] == (
        "committed_generalized_shell_section_state"
    )
    assert result.execution_report is not None
    assert result.execution_report.item_count == 0

    # The committed snapshot remains a valid restart/recovery input and is a
    # defensive copy of the nonlinear result.
    restarted = recover_stress_result(
        model,
        displacement,
        element_states=result.committed_element_states,
        kinematics="von_karman",
    )
    np.testing.assert_array_equal(
        restarted.element_stresses[1]["membrane_resultants"],
        state["membrane_resultants"],
    )


def test_generalized_shell_state_feeds_nonlinear_buckling_prestress() -> None:
    model, element = _shell_model()
    material = model.get_material("dummy")
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node_id in element.node_ids:
        node = model.mesh.nodes[node_id]
        x, y, _z = node.coords()
        displacement[node.dofs[0]] = -8.0e-4 * x
        displacement[node.dofs[2]] = 0.04 * x + 0.015 * y
    mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    state = element.compute_nonlinear_response(
        model.mesh,
        material,
        displacement[mapping],
        tangent=False,
    )[2]
    assert state is not None
    nonlinear = _nonlinear_result(displacement, state, "von_karman")

    prestress, summary = recover_prestress_from_static_result(
        model,
        displacement,
        nonlinear_result=nonlinear,
    )
    np.testing.assert_array_equal(
        np.asarray(prestress[1]["membrane_forces_at_gauss"]),
        state["membrane_resultants"],
    )
    np.testing.assert_array_equal(
        np.asarray(prestress[1]["bending_moments_at_gauss"]),
        state["bending_resultants"],
    )
    assert summary["stress_recovery"]["per_element_source"][1] == (
        "committed_generalized_shell_section_state"
    )


def test_generalized_beam_recovery_and_prestress_use_committed_von_karman_state() -> None:
    model, element = _beam_model()
    material = model.get_material("dummy")
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    displacement[model.mesh.nodes[2].dofs[0]] = -1.5e-3
    displacement[model.mesh.nodes[2].dofs[1]] = 0.08
    mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    state = element.compute_nonlinear_response(
        model.mesh,
        material,
        displacement[mapping],
        tangent=False,
    )[2]
    assert state is not None
    linear = element.compute_stresses(model.mesh, displacement, material)
    assert not np.array_equal(
        np.asarray(linear["generalized_resultant"]),
        np.asarray(state["generalized_resultant"]),
    )

    nonlinear = _nonlinear_result(displacement, state, "von_karman")
    result = recover_stress_result(model, nonlinear_result=nonlinear)
    recovered = result.element_stresses[1]
    np.testing.assert_array_equal(
        recovered["generalized_strain"],
        state["generalized_strain"],
    )
    np.testing.assert_array_equal(
        recovered["generalized_resultant"],
        state["generalized_resultant"],
    )
    assert recovered["physical_stress_available"] is False
    assert "von_mises" not in recovered
    assert result.provenance.per_element_source[1] == (
        "committed_generalized_beam_section_state"
    )

    prestress, summary = recover_prestress_from_static_result(
        model,
        displacement,
        nonlinear_result=nonlinear,
    )
    assert prestress[1]["axial_force"] == pytest.approx(
        float(np.mean(np.asarray(state["generalized_resultant"])[:, 0])),
        rel=0.0,
        abs=0.0,
    )
    assert summary["stress_recovery"]["per_element_source"][1] == (
        "committed_generalized_beam_section_state"
    )


def test_generalized_beam2_corotational_recovery_uses_committed_basic_forces() -> None:
    model, element = _beam_model(corotational=True)
    material = model.get_material("dummy")
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    displacement[model.mesh.nodes[2].dofs[0]] = 0.012
    displacement[model.mesh.nodes[2].dofs[1]] = 0.16
    displacement[model.mesh.nodes[1].dofs[5]] = 0.025
    displacement[model.mesh.nodes[2].dofs[5]] = 0.065
    mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    state = element.compute_nonlinear_response(
        model.mesh,
        material,
        displacement[mapping],
        tangent=False,
    )[2]
    assert state is not None
    nonlinear = _nonlinear_result(displacement, state, "corotational")

    result = recover_stress_result(model, nonlinear_result=nonlinear)
    np.testing.assert_array_equal(
        result.element_stresses[1]["generalized_strain"],
        state["generalized_strain"],
    )
    np.testing.assert_array_equal(
        result.element_stresses[1]["generalized_resultant"],
        state["generalized_resultant"],
    )
    assert result.provenance.analysis_context["kinematics"] == "corotational"
    assert result.provenance.per_element_component_sources[1][
        "stress_frame"
    ] == "current_corotated_frame"


def test_generalized_shell_patch_recovery_is_rejected_without_surface_stress() -> None:
    model, element = _shell_model()
    material = model.get_material("dummy")
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    state = element.compute_nonlinear_response(
        model.mesh,
        material,
        displacement[mapping],
        tangent=False,
    )[2]
    assert state is not None
    with pytest.raises(ValueError, match="physical top/bottom surface stresses"):
        recover_stress_result(
            model,
            nonlinear_result=_nonlinear_result(
                displacement,
                state,
                "von_karman",
            ),
            patch_config=PatchRecoveryConfig(),
        )
