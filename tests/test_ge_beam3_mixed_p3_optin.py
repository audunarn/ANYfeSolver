"""P3 opt-in integration checks for the qualified mixed GE-Beam3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest

import anyfesolver
from anyfesolver._native_rotation_state import create_native_rotation_state_store
from anyfesolver.assembly import solve_linear
from anyfesolver.beam_sections import GeneralizedBeamSection
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.elements import (
    ELEMENT_TYPES,
    CoupledBeamShellElement,
    QuadraticBeamElement,
    create_element,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.ge_beam3_element import (
    GE_BEAM3_QUALIFIED_FORMULATION_ID,
    GeBeam3IntegrationError,
    GeometricallyExactBeam3D3NElement,
    deserialize_ge_beam3_element,
    serialize_ge_beam3_element,
)
from anyfesolver.ge_beam3_mixed_element import (
    GeBeam3MixedStateError,
    GeometricallyExactBeam3D3NElement as CandidateGeBeam3,
)
from anyfesolver.ge_beam3_mixed_state import (
    serialize_ge_beam3_mixed_state,
)
from anyfesolver.ge_beam3_state import (
    STATE_LAYOUT_ID,
    STATE_SCHEMA,
    STATE_VERSION,
    GeBeam3CommittedStateError,
    deserialize_ge_beam3_state,
    serialize_ge_beam3_state,
)
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.modal import solve_free_vibration
from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement
from anyfesolver.nonlinear_state import create_model_native_rotation_store
from anyfesolver.nonlinear_static import solve_static_nonlinear


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs" / "reference_cases" / "ge_beam3_mixed_p3_baseline.json"
NEGATIVE_SELECTORS = (
    "ge_beam3",
    "beam3",
    "b3",
    "gebeam3",
    "ge-beam-3",
    "quadratic_beam",
    "beam",
)


def _section() -> GeneralizedBeamSection:
    stiffness = np.diag((1.20e7, 3.10e6, 2.90e6, 8.00e5, 6.00e5, 4.00e5))
    stiffness[0, 3] = stiffness[3, 0] = 2.00e4
    mass = np.diag((12.0, 12.0, 12.0, 0.20, 0.30, 0.40))
    mass[0, 5] = mass[5, 0] = -0.05
    return GeneralizedBeamSection(
        stiffness=stiffness,
        mass_matrix=mass,
        name="P3_COUPLED_LINEAR",
    )


def _model() -> FEModel:
    model = FEModel("ge-beam3-p3-optin")
    model.add_material("mat", 210.0e9, 0.3, density=7850.0)
    model.add_node(11, 0.0, 0.0, 0.0)
    model.add_node(12, 1.0, 0.0, 0.0)
    model.add_node(13, 2.0, 0.0, 0.0)
    return model


def _element(*, via_factory: bool = False) -> GeometricallyExactBeam3D3NElement:
    arguments: dict[str, Any] = {
        "section": _section(),
        "reference_orientation": (0.0, 1.0, 0.0),
        "reference_axis_direction": (1.0, 0.0, 0.0),
    }
    if via_factory:
        made = create_element("ge-beam3", 7, [11, 12, 13], "mat", **arguments)
    else:
        made = GeometricallyExactBeam3D3NElement(
            7, [11, 12, 13], "mat", **arguments
        )
    assert type(made) is GeometricallyExactBeam3D3NElement
    return made


def _git_blob_sha256(relative: str) -> str:
    raw = subprocess.check_output(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
    )
    return hashlib.sha256(raw).hexdigest().upper()


def test_exact_selector_and_public_exports_are_additive() -> None:
    selected = _element(via_factory=True)
    assert type(selected) is GeometricallyExactBeam3D3NElement
    assert selected.formulation_id == GE_BEAM3_QUALIFIED_FORMULATION_ID
    assert anyfesolver.GeometricallyExactBeam3D3NElement is GeometricallyExactBeam3D3NElement
    assert (
        anyfesolver.GE_BEAM3_QUALIFIED_FORMULATION_ID
        == GE_BEAM3_QUALIFIED_FORMULATION_ID
    )
    assert "GE_BEAM3_QUALIFIED_FORMULATION_ID" in anyfesolver.__all__
    assert "GeometricallyExactBeam3D3NElement" in anyfesolver.__all__
    assert "ge-beam3" not in ELEMENT_TYPES
    assert QuadraticBeamElement not in GeometricallyExactBeam3D3NElement.__mro__

    uppercase = create_element(
        "GE-BEAM3",
        8,
        [11, 12, 13],
        "mat",
        section=_section(),
        reference_orientation=(0.0, 1.0, 0.0),
        reference_axis_direction=(1.0, 0.0, 0.0),
    )
    assert type(uppercase) is GeometricallyExactBeam3D3NElement


@pytest.mark.parametrize("selector", NEGATIVE_SELECTORS)
def test_forbidden_aliases_never_select_qualified_ge_beam3(selector: str) -> None:
    kwargs: dict[str, Any] = {
        "section": _section(),
        "reference_orientation": (0.0, 1.0, 0.0),
        "reference_axis_direction": (1.0, 0.0, 0.0),
    }
    try:
        made = create_element(selector, 9, [11, 12, 13], "mat", **kwargs)
    except (TypeError, ValueError):
        return
    assert type(made) is not GeometricallyExactBeam3D3NElement


@pytest.mark.parametrize("node_ids", ([11, 12], [11, 12, 13, 14], [11, 11, 13]))
def test_selector_rejects_wrong_or_duplicate_connectivity(node_ids: list[int]) -> None:
    with pytest.raises(ValueError, match="three distinct nodes"):
        create_element(
            "ge-beam3",
            10,
            node_ids,
            "mat",
            section=_section(),
            reference_orientation=(0.0, 1.0, 0.0),
            reference_axis_direction=(1.0, 0.0, 0.0),
        )


def test_standard_linear_routes_are_exact_p2_operator_routes() -> None:
    model = _model()
    element = _element()
    public = element.compute_stiffness_matrix(model.mesh, model.get_material("mat"))
    private = element.compute_candidate_stiffness_matrix(model.mesh)
    assert public.dtype == private.dtype
    assert public.tobytes(order="C") == private.tobytes(order="C")

    displacement = np.linspace(-2.0e-4, 3.0e-4, 18)
    force = element.compute_internal_forces(
        model.mesh, displacement, model.get_material("mat")
    )
    np.testing.assert_array_equal(force, public @ displacement)
    with pytest.raises(GeBeam3IntegrationError, match="18 finite"):
        element.compute_internal_forces(model.mesh, np.zeros(17), None)
    bad = np.zeros(18)
    bad[4] = np.nan
    with pytest.raises(GeBeam3IntegrationError, match="18 finite"):
        element.compute_internal_forces(model.mesh, bad, None)


def test_standard_assembly_and_constrained_solve_use_public_operator() -> None:
    model = _model()
    element = _element(via_factory=True)
    model.add_element(element.element_id, element)
    assembled, info = assemble_stiffness_matrix(model)
    dense = np.asarray(assembled.toarray(), dtype=float)
    expected = element.compute_stiffness_matrix(
        model.mesh, model.get_material("mat")
    )
    np.testing.assert_array_equal(dense, expected)
    assert int(info["num_elements"]) == 1

    model.add_boundary_condition(
        BoundaryCondition(
            "fixed-end",
            [11],
            {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
        )
    )
    load_case = LoadCase("public-linear-route")
    load_case.add_nodal_load(13, forces=np.asarray((100.0, 0.0, 0.0)))
    solution, solve_info = solve_linear(model, load_case)
    assert solve_info["convergence_info"]["status"] == "converged"
    assert (
        solve_info["result_case"]["analysis_case"]["analysis_type"]
        == "linear_static"
    )

    free = np.arange(6, 18)
    rhs = np.zeros(len(free), dtype=np.float64)
    rhs[6] = 100.0
    np.testing.assert_allclose(
        dense[np.ix_(free, free)] @ solution[free],
        rhs,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_element_v2_serialization_is_strict_and_lossless() -> None:
    model = _model()
    element = _element()
    raw = serialize_ge_beam3_element(element, mesh=model.mesh)
    restored = deserialize_ge_beam3_element(raw)
    assert type(restored) is GeometricallyExactBeam3D3NElement
    assert restored.to_bytes() == raw
    np.testing.assert_array_equal(
        restored.compute_stiffness_matrix(model.mesh, None),
        element.compute_stiffness_matrix(model.mesh, None),
    )

    unknown = dict(element.to_dict())
    unknown["unregistered"] = False
    with pytest.raises(GeBeam3IntegrationError, match="keys mismatch"):
        GeometricallyExactBeam3D3NElement.from_dict(unknown)
    with pytest.raises(GeBeam3CommittedStateError, match="nonfinite"):
        deserialize_ge_beam3_element(raw.replace(b'"element_id":7', b'"element_id":NaN'))


def test_qualified_state_roundtrip_and_candidate_state_rejection() -> None:
    model = _model()
    element = _element()
    state = element.init_model_bound_nonlinear_state(model.mesh, None, 1)
    assert state["formulation_id"] == GE_BEAM3_QUALIFIED_FORMULATION_ID
    assert state["state_schema"] == STATE_SCHEMA
    assert state["state_version"] == STATE_VERSION
    assert state["state_layout_id"] == STATE_LAYOUT_ID
    raw = serialize_ge_beam3_state(state)
    restored = deserialize_ge_beam3_state(raw)
    validated = element.validate_model_bound_nonlinear_state(
        model.mesh, None, restored, 1
    )
    assert serialize_ge_beam3_state(validated) == raw

    candidate = CandidateGeBeam3(
        7,
        [11, 12, 13],
        "mat",
        section=_section(),
        reference_orientation=(0.0, 1.0, 0.0),
        reference_axis_direction=(1.0, 0.0, 0.0),
    )
    candidate_state = candidate.init_model_bound_nonlinear_state(
        model.mesh, None, 1
    )
    candidate_raw = serialize_ge_beam3_mixed_state(candidate_state)
    with pytest.raises(GeBeam3CommittedStateError, match="non-migratable"):
        deserialize_ge_beam3_state(candidate_raw)
    before = candidate_raw
    with pytest.raises(GeBeam3CommittedStateError, match="non-migratable"):
        element.validate_model_bound_nonlinear_state(
            model.mesh, None, candidate_state, 1
        )
    assert serialize_ge_beam3_mixed_state(candidate_state) == before


def test_qualified_state_nonzero_restart_continuation_is_bitwise_identical() -> None:
    model = _model()
    element = _element()
    model.add_element(element.element_id, element)
    nodes = tuple(element.node_ids)
    reference = element.get_node_coordinates(model.mesh)
    zero = np.zeros(18, dtype=np.float64)
    identity = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 3, axis=0)

    def store(total: np.ndarray, rotations: np.ndarray):
        made = create_native_rotation_state_store(
            nodes,
            rotational_dofs={
                node: (6 * row + 3, 6 * row + 4, 6 * row + 5)
                for row, node in enumerate(nodes)
            },
            coordinate_rows={node: row for row, node in enumerate(nodes)},
            committed_full_displacement=total,
            committed_full_coordinates=reference + total.reshape(3, 6)[:, :3],
            committed_rotation_matrices=rotations,
            coordinate_node_ids=nodes,
        )
        assert made is not None
        return made

    def evaluate(
        active_store: Any,
        committed_state: dict[str, Any],
        total: np.ndarray,
        *,
        commit: bool,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        coordinates = reference + total.reshape(3, 6)[:, :3]
        with active_store.candidate(total, coordinates) as candidate:
            view = candidate.element_view(
                element.element_id,
                nodes,
                element.native_reference_directors(model.mesh),
            )
            force, tangent, next_state = element.compute_nonlinear_response(
                model.mesh,
                model.get_material("mat"),
                total,
                committed_state,
                1,
                True,
                native_rotation_trial=view,
            )
            assert tangent is not None
            if commit:
                candidate.commit(total, coordinates)
        return np.asarray(force), np.asarray(tangent), next_state

    first = np.asarray(
        (
            0.0, 0.0, 0.0, 0.012, -0.009, 0.007,
            0.001, -0.002, 0.004, -0.006, 0.011, 0.005,
            0.003, 0.001, 0.006, 0.008, -0.004, 0.013,
        ),
        dtype=np.float64,
    )
    second = first + np.asarray(
        (
            0.0002, -0.0001, 0.0003, -0.004, 0.003, 0.002,
            0.0004, 0.0002, -0.0001, 0.003, -0.002, 0.004,
            -0.0001, 0.0003, 0.0002, 0.002, 0.004, -0.003,
        ),
        dtype=np.float64,
    )
    initial = element.init_model_bound_nonlinear_state(model.mesh, None, 1)
    continuous_store = store(zero, identity)
    _first_force, _first_tangent, first_state = evaluate(
        continuous_store, initial, first, commit=True
    )
    checkpoint = serialize_ge_beam3_state(first_state)
    restored_state = deserialize_ge_beam3_state(checkpoint)
    assert serialize_ge_beam3_state(restored_state) == checkpoint

    continuous_force, continuous_tangent, continuous_state = evaluate(
        continuous_store, first_state, second, commit=False
    )
    restarted_store = create_model_native_rotation_store(
        model,
        {element.element_id: restored_state},
        np.asarray(restored_state["committed_total_u"], dtype=np.float64),
    )
    assert restarted_store is not None
    restarted_force, restarted_tangent, restarted_state = evaluate(
        restarted_store, restored_state, second, commit=False
    )
    np.testing.assert_array_equal(restarted_force, continuous_force)
    np.testing.assert_array_equal(restarted_tangent, continuous_tangent)
    assert serialize_ge_beam3_state(restarted_state) == serialize_ge_beam3_state(
        continuous_state
    )


def test_public_default_nonlinear_solver_and_checkpoint_restart_are_reachable() -> None:
    def problem() -> tuple[FEModel, LoadCase]:
        model = _model()
        element = _element(via_factory=True)
        model.add_element(element.element_id, element)
        model.add_boundary_condition(
            BoundaryCondition(
                "fixed-end",
                [11],
                {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
            )
        )
        load = LoadCase("axial")
        load.add_nodal_load(13, forces=np.asarray((100.0, 0.0, 0.0)))
        return model, load

    reference_model, reference_load = problem()
    reference = solve_static_nonlinear(
        reference_model,
        reference_load,
        num_steps=2,
        max_iterations=12,
        tolerance=1.0e-8,
        emit_restart_checkpoint=True,
    )
    assert reference.status == "completed"
    assert reference.restart_checkpoint is not None
    assert reference.element_states[7]["state_schema"] == STATE_SCHEMA

    split_model, split_load = problem()
    first = solve_static_nonlinear(
        split_model,
        split_load,
        max_load_factor=0.5,
        num_steps=1,
        max_iterations=12,
        tolerance=1.0e-8,
        emit_restart_checkpoint=True,
    )
    assert first.status == "completed"
    assert first.restart_checkpoint is not None
    resumed = solve_static_nonlinear(
        split_model,
        split_load,
        max_load_factor=1.0,
        num_steps=2,
        max_iterations=12,
        tolerance=1.0e-8,
        restart_checkpoint=first.restart_checkpoint,
    )
    assert resumed.status == "completed"
    np.testing.assert_array_equal(resumed.displacements, reference.displacements)
    assert serialize_ge_beam3_state(resumed.element_states[7]) == serialize_ge_beam3_state(
        reference.element_states[7]
    )


def test_existing_additive_beam_shell_joint_fails_before_ge_beam3_mechanics() -> None:
    model = _model()
    model.add_node(14, 2.0, 1.0, 0.0)
    element = _element(via_factory=True)
    model.add_element(element.element_id, element)
    model.add_element(
        99,
        CoupledBeamShellElement(
            99,
            beam_node_id=13,
            shell_node_id=14,
            material_name="mat",
        ),
    )
    with pytest.raises(GeBeam3IntegrationError, match="beam-shell connection"):
        assemble_stiffness_matrix(model)

    interpolated = _model()
    interpolated.add_node(14, 2.0, 1.0, 0.0)
    interpolated_element = _element(via_factory=True)
    interpolated.add_element(interpolated_element.element_id, interpolated_element)
    interpolated.add_element(
        100,
        InterpolatedBeamShellMPCElement(
            100,
            beam_node_id=13,
            shell_node_ids=[14],
            shape_weights=np.asarray((1.0,)),
            eccentricity=np.zeros(3),
            material_name="mat",
        ),
    )
    with pytest.raises(GeBeam3IntegrationError, match="beam-shell connection"):
        assemble_stiffness_matrix(interpolated)


def test_native_recovery_mass_modal_and_reference_buckling_are_reachable() -> None:
    model = _model()
    element = _element()
    rotations = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    recovery = element.recover_native_fields(
        model.mesh, np.zeros(18), rotation_matrices=rotations
    )
    assert recovery["formulation_id"] == GE_BEAM3_QUALIFIED_FORMULATION_ID
    assert tuple(recovery["station_order"]) == (
        "XI_MINUS_1",
        "XI_ZERO_LEFT",
        "XI_ZERO_RIGHT",
        "XI_PLUS_1",
    )
    assert np.asarray(recovery["local_generalized_strain"]).shape == (4, 6)
    assert np.asarray(recovery["local_generalized_resultant"]).shape == (4, 6)

    stiffness = element.compute_stiffness_matrix(model.mesh, None)
    mass = element.compute_mass_matrix(model.mesh, None)
    assert stiffness.shape == mass.shape == (18, 18)
    assert np.linalg.eigvalsh(mass)[0] > 0.0
    eigenvalues = np.linalg.eigvalsh(
        np.linalg.solve(np.linalg.cholesky(mass), stiffness)
        @ np.linalg.inv(np.linalg.cholesky(mass).T)
    )
    assert int(np.count_nonzero(np.abs(eigenvalues) < 1.0e-7)) >= 6
    model.add_element(element.element_id, element)
    modal = solve_free_vibration(model, num_modes=6)
    assert modal.solver_status == "ok"
    assert len(modal.modes) == 6
    assert modal.assembly_info["stiffness"]["num_elements"] == 1

    compression = element.compute_reference_geometric_stiffness(
        model.mesh, axial_compression=2.5
    )
    force_form = element.compute_reference_geometric_stiffness(
        model.mesh, axial_force=-2.5
    )
    tension = element.compute_reference_geometric_stiffness(
        model.mesh, axial_force=2.5
    )
    np.testing.assert_array_equal(force_form, compression)
    np.testing.assert_array_equal(tension, -compression)


def test_unqualified_routes_fail_closed() -> None:
    model = _model()
    element = _element()
    with pytest.raises(GeBeam3IntegrationError, match="fibre stress"):
        element.compute_stresses(model.mesh, np.zeros(18), None)
    with pytest.raises(
        GeBeam3IntegrationError,
        match="current-state buckling|reference.*axial|buckling",
    ):
        element.compute_geometric_stiffness_matrix(model.mesh, None, {})
    with pytest.raises(GeBeam3IntegrationError, match="NativeElementRotationView"):
        element.compute_nonlinear_response(model.mesh, None, np.zeros(18))
    with pytest.raises(GeBeam3MixedStateError, match="spatial-dead|material-dead"):
        element.compute_reference_line_load(
            model.mesh,
            {
                "classification": "CONSERVATIVE_FOLLOWER",
                "force_per_reference_length_at_nodes": (0.0, 1.0, 0.0),
                "couple_per_reference_length_at_nodes": (0.0, 0.0, 0.0),
            },
        )
    assert {
        "beam_shell_connection",
        "conservative_follower_loads",
        "current_state_modal_and_buckling",
        "curved_reference",
        "finite_rotation_transient_dynamics",
        "gyroscopic_terms",
        "history_bearing_sections",
        "nonconservative_follower_loads",
    } <= element.capability_gaps
    assert "linear_transient_dynamics" not in element.capability_gaps
    assert "transient_algebraic_dynamics" not in element.capability_gaps

    curved = _model()
    curved.mesh.set_node_coordinates(12, 1.0, 0.1, 0.0)
    with pytest.raises(ValueError, match="straight-reference midpoint"):
        element.compute_stiffness_matrix(curved.mesh, None)


def test_frozen_mechanics_defaults_and_package_metadata_are_unchanged() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    frozen = {
        record["path"]: record["sha256"]
        for record in baseline["production_snapshot"]
        if record["path"]
        in {
            "src/anyfesolver/e4_pl_element.py",
            "src/anyfesolver/e4_pl_s3_v2d_element.py",
            "src/anyfesolver/ge_beam3_mixed_element.py",
            "src/anyfesolver/ge_beam3_mixed_state.py",
            "pyproject.toml",
        }
    }
    assert frozen
    for relative, expected in frozen.items():
        assert _git_blob_sha256(relative) == expected
