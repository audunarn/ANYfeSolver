from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest

import anyfesolver.e4_pl_s3_element as s3

from anyfesolver._native_rotation_state import (
    NativeElementRotationView,
    create_native_rotation_state_store,
    rotation_exponential,
)
from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.e4_pl_s3_state import (
    GENERALIZED_NONLINEAR_STATE_SCHEMA,
    GENERALIZED_STATE_MODE,
    S3CommittedStateError,
    canonical_json_bytes,
    seal_committed_s3_state,
    strict_canonical_json_loads,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear_static import solve_static_nonlinear
from anyfesolver.shell_sections import GeneralizedShellSection
from _e4_pl_s3_native_trial import native_trial_for_increment


_REFERENCE = np.asarray(
    ((0.0, 0.0, 0.0), (1.1, 0.0, 0.0), (0.24, 0.94, 0.0)),
    dtype=np.float64,
)


def _section(*, scale: float = 1.0) -> GeneralizedShellSection:
    return GeneralizedShellSection(
        name="B-coupled-anisotropic",
        A=scale
        * np.asarray(
            (
                (2.4e8, 0.42e8, 0.13e8),
                (0.42e8, 1.35e8, -0.08e8),
                (0.13e8, -0.08e8, 0.71e8),
            )
        ),
        B=scale
        * np.asarray(
            (
                (2.1e4, -0.7e4, 0.3e4),
                (0.4e4, 1.6e4, -0.2e4),
                (-0.5e4, 0.1e4, 0.8e4),
            )
        ),
        D=scale
        * np.asarray(
            (
                (3.2e4, 0.55e4, 0.12e4),
                (0.55e4, 2.1e4, -0.09e4),
                (0.12e4, -0.09e4, 1.15e4),
            )
        ),
        As=scale * np.asarray(((0.82e8, 0.11e8), (0.11e8, 0.59e8))),
        mass_per_area=41.0,
        rotary_inertia_per_area=0.014,
    )


def _model(
    *,
    section: GeneralizedShellSection | None = None,
    element_type=QualifiedE4PLS3ShellElement,
) -> tuple[FEModel, QualifiedE4PLS3ShellElement]:
    model = FEModel("qualified-s3-generalized-native")
    model.add_material("carrier", 70.0e9, 0.31, density=2700.0)
    for node_id, coordinates in enumerate(_REFERENCE, start=1):
        model.add_node(node_id, *coordinates)
    element = element_type(
        1,
        [1, 2, 3],
        "carrier",
        thickness=0.027,
        material_direction=(0.91, 0.37, 0.0),
        material_angle_deg=13.0,
        shell_section=_section() if section is None else section,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)
    return model, element


def _view(
    model: FEModel,
    element: QualifiedE4PLS3ShellElement,
    total_u: np.ndarray,
    committed_state: dict[str, object] | None,
) -> NativeElementRotationView:
    trial_u = np.asarray(total_u, dtype=np.float64).reshape(18)
    if committed_state is None:
        committed_u = np.zeros(18, dtype=np.float64)
        committed_q = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    else:
        committed_u = np.asarray(
            committed_state["committed_total_u"], dtype=np.float64
        ).reshape(18)
        committed_q = np.asarray(
            committed_state["committed_nodal_rotation_matrices"],
            dtype=np.float64,
        ).reshape(3, 3, 3)
    node_ids = tuple(element.node_ids)
    store = create_native_rotation_state_store(
        node_ids,
        rotational_dofs={
            node_id: (6 * row + 3, 6 * row + 4, 6 * row + 5)
            for row, node_id in enumerate(node_ids)
        },
        coordinate_rows={node_id: row for row, node_id in enumerate(node_ids)},
        coordinate_node_ids=node_ids,
        committed_full_displacement=committed_u,
        committed_full_coordinates=(
            _REFERENCE + committed_u.reshape(3, 6)[:, :3]
        ),
        committed_rotation_matrices={
            node_id: committed_q[row] for row, node_id in enumerate(node_ids)
        },
    )
    assert store is not None
    token = store.begin_trial(
        trial_u,
        _REFERENCE + trial_u.reshape(3, 6)[:, :3],
    )
    return store.element_view(
        element.element_id,
        node_ids,
        element.native_reference_directors(model.mesh),
        trial_token=token,
    )


def _response(
    model: FEModel,
    element: QualifiedE4PLS3ShellElement,
    total_u: np.ndarray,
    state: dict[str, object] | None = None,
    *,
    tangent: bool = True,
):
    view = _view(model, element, total_u, state)
    force, matrix, trial = element.compute_nonlinear_response(
        model.mesh,
        model.get_material("carrier"),
        total_u,
        state,
        3,
        tangent,
        native_rotation_trial=view,
    )
    return force, matrix, trial, view


def test_generalized_zero_response_has_distinct_state_and_no_layer_fabrication() -> None:
    model, element = _model()
    zero = np.zeros(18)
    force, tangent, state, _view_value = _response(model, element, zero)

    np.testing.assert_allclose(force, np.zeros(18), rtol=0.0, atol=5.0e-9)
    assert tangent is not None
    np.testing.assert_allclose(
        tangent,
        element.compute_stiffness_matrix(model.mesh, model.get_material("carrier")),
        rtol=5.0e-13,
        atol=1.0e-7,
    )
    assert state["state_schema"] == GENERALIZED_NONLINEAR_STATE_SCHEMA
    assert state["state_mode"] == GENERALIZED_STATE_MODE
    assert state["recovery_scope"] == "section_resultants_only"
    assert state["physical_layer_recovery_available"] is False
    for forbidden in (
        "plastic_strain",
        "alpha",
        "layer_strain",
        "layer_stress",
        "layer_stress_material",
    ):
        assert forbidden not in state
    assert "nonlinear_geometry" not in element.capability_gaps
    assert "material_nonlinearity" not in element.capability_gaps
    assert element.capability_matrix()["material_nonlinearity"] == (
        "NOT_APPLICABLE_STATELESS_FIXED_SECTION"
    )
    assert "patch_recovery" not in element.capability_gaps
    assert element.capability_matrix()["patch_recovery"] == (
        "NOT_APPLICABLE_NO_PHYSICAL_LAYER_FIELD"
    )
    assert "committed_state_recovery" not in element.capability_gaps
    assert "initial_fields" not in element.capability_gaps
    assert "restart_history" not in element.capability_gaps
    assert element.capability_matrix()["restart_history"] == (
        "STATIC_AND_ARC_LENGTH_CHECKPOINTS_ONLY"
    )
    assert "static_restart_history" not in element.capability_gaps
    assert "arc_length_restart_history" not in element.capability_gaps
    assert "buckling" not in element.capability_gaps
    assert element.capability_matrix()["current_state_buckling_s3"] == (
        "PARITY_REPLACED"
    )
    assert element.capability_matrix()["mixed_current_state_buckling"] == (
        "PARITY_REPLACED"
    )
    assert element.capability_matrix()["mixed_current_state_modal"] == (
        "PARITY_REPLACED"
    )

    repeated_force, no_tangent, repeated_state, _repeated_view = _response(
        model, element, zero, state, tangent=False
    )
    np.testing.assert_array_equal(repeated_force, force)
    assert no_tangent is None
    assert canonical_json_bytes(repeated_state) == canonical_json_bytes(state)


def test_generalized_total_tangent_fd_rigid_objectivity_and_noncommuting_rebase() -> None:
    model, element = _model()
    rng = np.random.default_rng(20260825)
    centre = 1.5e-5 * rng.standard_normal(18)
    direction = rng.standard_normal(18)
    direction /= np.linalg.norm(direction)
    step = 8.0e-8
    _force, tangent, _state, _trial_view = _response(model, element, centre)
    plus = _response(model, element, centre + step * direction)[0]
    minus = _response(model, element, centre - step * direction)[0]
    assert tangent is not None
    np.testing.assert_allclose(
        tangent @ direction,
        (plus - minus) / (2.0 * step),
        rtol=4.0e-6,
        atol=0.1,
    )

    rotation_vector = np.asarray((0.19, -0.14, 0.21))
    rotation = rotation_exponential(rotation_vector)
    rigid = np.zeros(18)
    rigid.reshape(3, 6)[:, :3] = (rotation @ _REFERENCE.T).T - _REFERENCE
    rigid.reshape(3, 6)[:, 3:6] = rotation_vector
    rigid_force, _rigid_tangent, rigid_state, rigid_view = _response(
        model, element, rigid
    )
    np.testing.assert_allclose(rigid_force, np.zeros(18), rtol=0.0, atol=2.0e-6)
    np.testing.assert_allclose(
        rigid_state["station_generalized_strain"],
        np.zeros((7, 8)),
        rtol=0.0,
        atol=3.0e-15,
    )
    np.testing.assert_array_equal(
        rigid_state["committed_nodal_rotation_matrices"],
        rigid_view.trial_rotation_matrices,
    )

    first_vector = np.asarray((0.17, -0.06, 0.04))
    first = np.zeros(18)
    first.reshape(3, 6)[:, 3:6] = first_vector
    _f1, _k1, state1, _v1 = _response(model, element, first)
    second_vector = np.asarray((-0.05, 0.15, 0.08))
    second = np.asarray(state1["committed_total_u"]).copy()
    second.reshape(3, 6)[:, 3:6] += second_vector
    _f2, _k2, state2, view2 = _response(model, element, second, state1)
    exact_second = np.asarray(view2.rotation_coordinate_increment[0])
    expected_q = rotation_exponential(exact_second) @ rotation_exponential(
        first_vector
    )
    np.testing.assert_array_equal(
        state2["committed_nodal_rotation_matrices"][0], expected_q
    )
    assert not np.allclose(
        expected_q,
        rotation_exponential(first_vector + second_vector),
        rtol=0.0,
        atol=1.0e-5,
    )


def test_generalized_state_round_trip_binds_section_and_rejects_mutation() -> None:
    model, element = _model()
    total = np.zeros(18)
    total[6] = 2.0e-4
    total[12] = -0.7e-4
    total[4] = 1.5e-4
    _force, _tangent, state, _view_value = _response(model, element, total)

    raw = canonical_json_bytes(state)
    decoded = strict_canonical_json_loads(raw)
    validated = element.validate_model_bound_nonlinear_state(
        model.mesh,
        model.get_material("carrier"),
        decoded,
        11,
        expected_committed_total_u=total,
    )
    assert canonical_json_bytes(validated) == raw

    changed_resultant = copy.deepcopy(state)
    changed_resultant["station_generalized_resultant"][0, 0] += 1.0
    changed_resultant = seal_committed_s3_state(changed_resultant)
    with pytest.raises(S3CommittedStateError, match="resultants contradict"):
        element.validate_model_bound_nonlinear_state(
            model.mesh,
            model.get_material("carrier"),
            changed_resultant,
            3,
        )

    other_model, other_element = _model(section=_section(scale=1.0001))
    with pytest.raises(S3CommittedStateError, match="identity"):
        other_element.validate_model_bound_nonlinear_state(
            other_model.mesh,
            other_model.get_material("carrier"),
            state,
            3,
        )

    stale_digest = copy.deepcopy(state)
    stale_digest["committed_internal_force"][0] += 1.0
    with pytest.raises(S3CommittedStateError, match="integrity"):
        element.validate_model_bound_nonlinear_state(
            model.mesh,
            model.get_material("carrier"),
            stale_digest,
            3,
        )

    fabricated_layer = copy.deepcopy(state)
    fabricated_layer["layer_stress"] = np.zeros((21, 3))
    fabricated_layer = seal_committed_s3_state(fabricated_layer)
    with pytest.raises(S3CommittedStateError, match="unknown=.*layer_stress"):
        element.validate_model_bound_nonlinear_state(
            model.mesh,
            model.get_material("carrier"),
            fabricated_layer,
            3,
        )

    inconsistent_pl = copy.deepcopy(state)
    inconsistent_pl["committed_pl_twist"][0] += 1.0e-6
    inconsistent_pl["committed_pl_turn_count"][0] = 0
    inconsistent_pl = seal_committed_s3_state(inconsistent_pl)
    with pytest.raises(S3CommittedStateError, match="PL multiplier"):
        element.validate_model_bound_nonlinear_state(
            model.mesh,
            model.get_material("carrier"),
            inconsistent_pl,
            3,
        )


def _numbered_frame(nodes: np.ndarray, normal: np.ndarray) -> np.ndarray:
    first = np.asarray(nodes[1] - nodes[0], dtype=float)
    first -= float(first @ normal) * normal
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    return np.column_stack((first, second, normal))


def _constitutive_in_frame(
    section: GeneralizedShellSection,
    frame: np.ndarray,
    physical_direction: np.ndarray,
) -> np.ndarray:
    components = frame[:, :2].T @ physical_direction
    rotated = section.rotated(float(np.arctan2(components[1], components[0])))
    result = np.zeros((8, 8))
    result[:3, :3] = rotated.A
    result[:3, 3:6] = rotated.B
    result[3:6, :3] = rotated.B.T
    result[3:6, 3:6] = rotated.D
    result[6:, 6:] = rotated.As
    return result


def _permutation_matrix(permutation: tuple[int, int, int]) -> np.ndarray:
    result = np.zeros((18, 18))
    for new_node, source_node in enumerate(permutation):
        result[
            6 * new_node : 6 * new_node + 6,
            6 * source_node : 6 * source_node + 6,
        ] = np.eye(6)
    return result


def _private_generalized_response(
    nodes: np.ndarray,
    normal: np.ndarray,
    physical_direction: np.ndarray,
    section: GeneralizedShellSection,
    external: np.ndarray,
):
    frame = _numbered_frame(nodes, normal)
    constitutive = _constitutive_in_frame(section, frame, physical_direction)
    reference_triad = np.eye(3)
    triads = np.repeat(reference_triad[None, :, :], 4, axis=0)

    def builder(increment: np.ndarray):
        view, exact, _store = native_trial_for_increment(nodes, triads, increment)
        return s3._native_generalized_uncondensed_response(
            nodes,
            triads,
            exact,
            nodes,
            frame,
            constitutive,
            np.zeros((7, 8)),
            native_rotation_trial=view,
        )

    return s3._solve_native_bubble_equilibrium(
        external, np.zeros(2), builder
    )


def test_b_coupled_anisotropic_generalized_response_is_full_d3_covariant() -> None:
    section = _section()
    normal = np.asarray((0.0, 0.0, 1.0))
    direction = np.asarray((0.91, 0.37, 0.0))
    direction /= np.linalg.norm(direction)
    external = np.asarray(
        (
            1.2e-4, -0.7e-4, 0.4e-4, 0.8e-4, -0.5e-4, 0.3e-4,
            -0.6e-4, 1.1e-4, -0.9e-4, -0.4e-4, 0.7e-4, -0.2e-4,
            0.3e-4, -0.8e-4, 0.6e-4, 0.5e-4, 0.2e-4, -0.6e-4,
        )
    )
    baseline_force, baseline_tangent, _state, _meta = (
        _private_generalized_response(
            _REFERENCE, normal, direction, section, external
        )
    )
    probe = np.linspace(-0.7, 0.9, 18)
    baseline_work = float(probe @ baseline_force)

    for permutation in itertools.permutations(range(3)):
        made = tuple(int(value) for value in permutation)
        operator = _permutation_matrix(made)
        force, tangent, state, _metadata = _private_generalized_response(
            _REFERENCE[np.asarray(made)],
            normal,
            direction,
            section,
            operator @ external,
        )
        np.testing.assert_allclose(
            force, operator @ baseline_force, rtol=2.0e-11, atol=2.0e-6
        )
        np.testing.assert_allclose(
            tangent,
            operator @ baseline_tangent @ operator.T,
            rtol=2.0e-10,
            atol=2.0e-4,
        )
        assert float((operator @ probe) @ force) == pytest.approx(
            baseline_work, rel=2.0e-11, abs=2.0e-8
        )
        assert state["generalized_section"] is True
        assert state["recovery_scope"] == "section_resultants_only"

def test_generalized_section_owned_history_fails_closed() -> None:
    payload = _section().to_dict()
    payload["state_schema"] = "external.section.history.v1"
    with pytest.raises(S3CommittedStateError, match="stateless.*history markers"):
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "carrier",
            shell_section=payload,
            reference_normal=(0.0, 0.0, 1.0),
        )


def _constrained_model() -> tuple[FEModel, LoadCase]:
    model, _element = _model(
        element_type=QualifiedE4PLS3ShellElement
    )
    model.add_boundary_condition(FixedSupport("node-1", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "node-2-one-coordinate",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(FixedSupport("node-3", [3]))
    load = LoadCase("pull")
    load.add_nodal_load(2, [25.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load


def test_generalized_section_runs_static_and_arc_through_native_transaction() -> None:
    static_model, static_load = _constrained_model()
    static = solve_static_nonlinear(
        static_model,
        static_load,
        num_steps=2,
        num_layers=5,
    )
    assert static.status == "completed"
    assert static.element_states[1]["state_schema"] == (
        GENERALIZED_NONLINEAR_STATE_SCHEMA
    )
    assert static.info["nonlinear_state_storage"]["native_rotation_activated"]

    arc_model, arc_load = _constrained_model()
    arc = solve_static_arc_length(
        arc_model,
        arc_load,
        control=ArcLengthControl(
            initial_load_increment=0.05,
            minimum_load_increment=0.001,
            maximum_load_increment=0.10,
            maximum_absolute_load_factor=0.10,
            max_steps=2,
        ),
        num_layers=7,
    )
    assert arc.status == "load_factor_limit_reached"
    assert arc.element_states[1]["state_schema"] == (
        GENERALIZED_NONLINEAR_STATE_SCHEMA
    )
    assert arc.info["nonlinear_state_storage"]["native_rotation_activated"]


def test_generalized_static_and_arc_checkpoints_match_uninterrupted_paths() -> None:
    static_full_model, static_full_load = _constrained_model()
    static_full = solve_static_nonlinear(
        static_full_model,
        static_full_load,
        max_load_factor=1.0,
        num_steps=4,
        num_layers=5,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    static_first_model, static_first_load = _constrained_model()
    static_first = solve_static_nonlinear(
        static_first_model,
        static_first_load,
        max_load_factor=0.5,
        num_steps=2,
        num_layers=5,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    static_resume_model, static_resume_load = _constrained_model()
    static_resumed = solve_static_nonlinear(
        static_resume_model,
        static_resume_load,
        max_load_factor=1.0,
        num_steps=2,
        num_layers=5,
        convergence_settings="legacy",
        restart_checkpoint=static_first.restart_checkpoint_bytes(),
    )
    np.testing.assert_array_equal(static_full.displacements, static_resumed.displacements)
    assert static_full.restart_checkpoint_bytes() == static_resumed.restart_checkpoint_bytes()

    arc_common = {
        "initial_load_increment": 0.05,
        "minimum_load_increment": 0.05,
        "maximum_load_increment": 0.05,
        "growth_factor": 1.0,
        "stop_after_peak_steps": 20,
    }
    arc_full_model, arc_full_load = _constrained_model()
    arc_full = solve_static_arc_length(
        arc_full_model,
        arc_full_load,
        control=ArcLengthControl(max_steps=4, **arc_common),
        num_layers=7,
        emit_restart_checkpoint=True,
    )
    arc_first_model, arc_first_load = _constrained_model()
    arc_first = solve_static_arc_length(
        arc_first_model,
        arc_first_load,
        control=ArcLengthControl(max_steps=2, **arc_common),
        num_layers=7,
        emit_restart_checkpoint=True,
    )
    arc_resume_model, arc_resume_load = _constrained_model()
    arc_resumed = solve_static_arc_length(
        arc_resume_model,
        arc_resume_load,
        control=ArcLengthControl(max_steps=2, **arc_common),
        num_layers=7,
        restart_checkpoint=arc_first.restart_checkpoint_bytes(),
    )
    np.testing.assert_array_equal(arc_full.displacements, arc_resumed.displacements)
    assert arc_full.restart_checkpoint_bytes() == arc_resumed.restart_checkpoint_bytes()
