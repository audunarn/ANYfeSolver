from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

import anyfesolver.arc_length as arc_length_module
import anyfesolver.e4_pl_element as q4_element_module
import anyfesolver.nonlinear_static as nonlinear_static
from anyfesolver.activity import ElementActivity
from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.control import CancellationToken
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import ShellElement
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.matrix_assembly import AssemblyError
from anyfesolver.nonlinear_restart import (
    NonlinearCheckpointError,
    canonical_checkpoint_json_bytes,
    create_nonlinear_checkpoint,
)
from anyfesolver.nonlinear_static import ShellInitialField
from anyfesolver.shell_sections import GeneralizedShellSection


def _cache_tree_snapshot(value: Any) -> tuple[Any, ...]:
    """Capture container shape and exact binary64 payload without mutation."""

    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return (
            "ndarray",
            array.dtype.str,
            tuple(int(size) for size in array.shape),
            tuple(int(stride) for stride in array.strides),
            bool(array.flags.writeable),
            array.tobytes(order="A"),
        )
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        return ("numpy_scalar", scalar.dtype.str, scalar.tobytes())
    if isinstance(value, Mapping):
        return (
            "mapping",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (_cache_tree_snapshot(key), _cache_tree_snapshot(item))
                for key, item in value.items()
            ),
        )
    if isinstance(value, (tuple, list)):
        return (
            type(value).__name__,
            tuple(_cache_tree_snapshot(item) for item in value),
        )
    if isinstance(value, bytes):
        return ("bytes", value)
    if value is None or isinstance(value, (bool, int, str)):
        return (type(value).__name__, value)
    if isinstance(value, float):
        return ("float64", np.asarray(value, dtype=np.float64).tobytes())
    return ("object", type(value).__module__, type(value).__qualname__, id(value))


def _loaded_q4_model() -> tuple[FEModel, LoadCase]:
    model = FEModel("q4-committed-state-lifecycle")
    model.add_material(
        "steel",
        210.0e9,
        0.3,
        density=7850.0,
        hardening_curve=DNVC208MaterialCurve(
            sigma_prop=320.0e6,
            sigma_yield=357.0e6,
            sigma_yield_2=363.3e6,
            eps_p_y1=0.004,
            eps_p_y2=0.015,
            K=740.0e6,
            n=0.166,
        ),
    )
    for node_id, coordinates in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.2, 0.0),
            (0.0, 0.2, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1,
            [1, 2, 3, 4],
            "steel",
            thickness=0.01,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    model.add_boundary_condition(
        BoundaryCondition("left-x", [1, 4], {"ux": 0.0})
    )
    model.add_boundary_condition(
        BoundaryCondition("origin-y", [1], {"uy": 0.0})
    )
    model.add_boundary_condition(
        BoundaryCondition(
            "planar",
            [1, 2, 3, 4],
            {"uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    load = LoadCase("q4-axial")
    load.add_nodal_load(2, [2.0e5, 0.0, 0.0, 0.0, 0.0, 0.0])
    load.add_nodal_load(3, [2.0e5, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load


def _solve_loaded(
    *,
    maximum: float = 1.0,
    steps: int = 1,
    emit_checkpoint: bool = False,
):
    model, load = _loaded_q4_model()
    result = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        max_load_factor=maximum,
        num_steps=steps,
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        convergence_settings="legacy",
        emit_restart_checkpoint=emit_checkpoint,
    )
    assert result.status == "completed"
    return result


def _assert_q4_state_is_exactly_bound(
    model: FEModel,
    result,
) -> str:
    state = result.element_states[1]
    element = model.mesh.elements[1]
    dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    local_u = np.asarray(result.displacements, dtype=np.float64)[dofs]
    digest = element.validate_committed_current_tangent_state(
        model.mesh,
        model.get_material(element.material_name),
        local_u,
        state,
        3,
    )
    np.testing.assert_array_equal(
        state["qualified_q4_committed_binding"]["committed_total_u"],
        local_u,
    )
    assert state["state_integrity_sha256"] == digest
    return digest


def test_solver_seals_scalar_and_batched_layered_q4_results_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonlinear_static._ensure_nonlinear_acceleration()
    from anyfesolver import nonlinear_performance

    batched_model, batched_load = _loaded_q4_model()
    batched = nonlinear_static.solve_static_nonlinear(
        batched_model,
        batched_load,
        num_steps=1,
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        convergence_settings="legacy",
    )
    assert batched.status == "completed"
    assert nonlinear_performance._ORIGINAL_ASSEMBLER is not None

    scalar_model, scalar_load = _loaded_q4_model()
    monkeypatch.setattr(
        nonlinear_static,
        "_assemble_nonlinear_system",
        nonlinear_performance._ORIGINAL_ASSEMBLER,
    )
    scalar = nonlinear_static.solve_static_nonlinear(
        scalar_model,
        scalar_load,
        num_steps=1,
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        convergence_settings="legacy",
    )
    assert scalar.status == "completed"
    np.testing.assert_array_equal(batched.displacements, scalar.displacements)
    assert canonical_json_bytes(batched.element_states[1]) == canonical_json_bytes(
        scalar.element_states[1]
    )
    assert _assert_q4_state_is_exactly_bound(batched_model, batched) == (
        _assert_q4_state_is_exactly_bound(scalar_model, scalar)
    )
    assert batched.info["qualified_q4_committed_state_lifecycle"][
        "sealed_final_element_ids"
    ] == [1]


def test_forced_line_search_cut_commits_only_the_tangent_reevaluated_q4_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonlinear_static._ensure_nonlinear_acceleration()
    model, load = _loaded_q4_model()
    # Enter the nonlinear part of the material curve so the deliberately
    # rejected full Newton candidate cannot coincide with the final state.
    load.add_nodal_load(2, [4.0e5, 0.0, 0.0, 0.0, 0.0, 0.0])
    load.add_nodal_load(3, [4.0e5, 0.0, 0.0, 0.0, 0.0, 0.0])

    original = nonlinear_static._assemble_nonlinear_system
    tangent_count = 0
    forced_rejected: dict[str, Any] = {}
    residual_only_trials: list[dict[str, Any]] = []
    accepted_pair: dict[str, Any] = {}
    tangent_records: list[dict[str, Any]] = []

    def state_core(states) -> dict[str, Any]:
        state = states[1]
        names = (
            "plastic_strain",
            "alpha",
            "layer_strain",
            "qualified_q4_algorithmic_origin",
        )
        return {
            name: copy.deepcopy(state[name])
            for name in names
            if name in state
        }

    def observed(*args, **kwargs):
        nonlocal tangent_count
        force, tangent_matrix, states = original(*args, **kwargs)
        displacement = np.asarray(args[1], dtype=np.float64).copy()
        with_tangent = bool(kwargs.get("tangent", True))
        core = state_core(states)
        if with_tangent:
            tangent_count += 1
            tangent_records.append(
                {"displacement": displacement, "state": core}
            )
            for residual_trial in reversed(residual_only_trials):
                if np.array_equal(
                    displacement, residual_trial["displacement"]
                ):
                    accepted_pair.update(
                        {"residual": residual_trial, "tangent": core}
                    )
                    break
            if tangent_count == 2:
                forced_rejected.update(
                    {"displacement": displacement, "state": core}
                )
                made_force = np.asarray(force, dtype=np.float64).copy()
                for node_id in (2, 3):
                    made_force[model.mesh.get_node(node_id).dofs[0]] += 1.0e12
                return made_force, tangent_matrix, states
        else:
            residual_only_trials.append(
                {"displacement": displacement, "state": core}
            )
        return force, tangent_matrix, states

    monkeypatch.setattr(
        nonlinear_static,
        "_assemble_nonlinear_system",
        observed,
    )
    result = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        num_layers=3,
        max_iterations=30,
        tolerance=1.0e-9,
        convergence_settings={
            "profile": "legacy",
            "line_search": "always",
            "line_search_reduction": 0.5,
        },
    )
    assert result.status == "completed"
    assert forced_rejected and accepted_pair
    assert "qualified_q4_algorithmic_origin" not in accepted_pair["residual"][
        "state"
    ]
    assert "qualified_q4_algorithmic_origin" in accepted_pair["tangent"]
    for name in ("plastic_strain", "alpha", "layer_strain"):
        np.testing.assert_array_equal(
            accepted_pair["residual"]["state"][name],
            accepted_pair["tangent"][name],
        )

    final_tangent = next(
        record["state"]
        for record in reversed(tangent_records)
        if np.array_equal(record["displacement"], result.displacements)
    )
    final_state = result.element_states[1]
    for name in (
        "plastic_strain",
        "alpha",
        "layer_strain",
        "qualified_q4_algorithmic_origin",
    ):
        assert canonical_json_bytes(final_state[name]) == canonical_json_bytes(
            final_tangent[name]
        )
    assert not np.array_equal(
        forced_rejected["displacement"], result.displacements
    )
    assert canonical_json_bytes(
        forced_rejected["state"]["layer_strain"]
    ) != canonical_json_bytes(final_state["layer_strain"])
    _assert_q4_state_is_exactly_bound(model, result)


def test_fully_constrained_generalized_q4_is_materialized_and_sealed() -> None:
    model = FEModel("q4-generalized-fully-constrained")
    model.add_material("carrier", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    section = GeneralizedShellSection(
        A=np.asarray(((160.0, 17.0, 5.0), (17.0, 105.0, -4.0), (5.0, -4.0, 48.0))),
        B=np.asarray(((1.4, 0.2, -0.1), (0.2, -0.9, 0.15), (-0.1, 0.15, 0.35))),
        D=np.asarray(((18.0, 1.2, 0.3), (1.2, 12.0, -0.2), (0.3, -0.2, 5.5))),
        As=np.asarray(((28.0, 2.0), (2.0, 21.0))),
        mass_per_area=8.0,
        rotary_inertia_per_area=0.02,
    )
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1,
            [1, 2, 3, 4],
            "carrier",
            shell_section=section,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    model.add_boundary_condition(FixedSupport("all", [1, 2, 3, 4]))
    result = nonlinear_static.solve_static_nonlinear(
        model,
        num_steps=1,
        num_layers=3,
    )
    assert result.status == "empty_reduced_system"
    assert set(result.element_states[1]) >= {
        "membrane_strain",
        "curvature",
        "transverse_shear_strain",
        "qualified_q4_committed_binding",
        "state_integrity_sha256",
    }
    _assert_q4_state_is_exactly_bound(model, result)


def test_complete_q4_restart_binding_is_validated_stripped_and_resealed_without_mutation() -> None:
    preload = _solve_loaded(maximum=0.5)
    states = copy.deepcopy(preload.element_states)
    before = canonical_json_bytes(states[1])
    model, load = _loaded_q4_model()
    resumed = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        max_load_factor=0.25,
        num_steps=1,
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        convergence_settings="legacy",
        initial_element_states=states,
        initial_displacements=preload.displacements.copy(),
        equilibrate_initial_state=False,
    )
    assert resumed.status == "completed"
    lifecycle = resumed.info["qualified_q4_committed_state_lifecycle"]
    assert lifecycle["validated_bound_element_ids"] == [1]
    assert lifecycle["stripped_internal_binding_element_ids"] == [1]
    assert lifecycle["migrated_historical_unbound_element_ids"] == []
    assert resumed.info["nonlinear_state_storage"]["eligible_batch_count"] == 1
    assert resumed.info["nonlinear_state_storage"][
        "dictionary_fallback_element_count"
    ] == 0
    assert canonical_json_bytes(states[1]) == before
    _assert_q4_state_is_exactly_bound(model, resumed)


def test_partial_and_mismatched_q4_bindings_reject_before_nonlinear_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preload = _solve_loaded(maximum=0.5)
    model, load = _loaded_q4_model()
    original = QualifiedE4PLShellElement.compute_nonlinear_response
    calls = 0

    def observed(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        QualifiedE4PLShellElement,
        "compute_nonlinear_response",
        observed,
    )
    partial = copy.deepcopy(preload.element_states)
    partial[1].pop("state_integrity_sha256")
    partial_bytes = canonical_json_bytes(partial[1])
    with pytest.raises(ElementCapabilityError, match="CRITICAL_API_MISMATCH"):
        nonlinear_static.solve_static_nonlinear(
            model,
            load,
            max_load_factor=0.1,
            num_steps=1,
            num_layers=3,
            initial_element_states=partial,
            initial_displacements=preload.displacements.copy(),
            equilibrate_initial_state=False,
        )
    assert calls == 0
    monkeypatch.setattr(
        QualifiedE4PLShellElement,
        "compute_nonlinear_response",
        original,
    )
    with pytest.raises(ValueError, match="partial committed-state binding"):
        nonlinear_static.solve_static_nonlinear(
            model,
            load,
            max_load_factor=0.1,
            num_steps=1,
            num_layers=3,
            initial_element_states=partial,
            initial_displacements=preload.displacements.copy(),
            equilibrate_initial_state=False,
        )
    assert calls == 0
    assert canonical_json_bytes(partial[1]) == partial_bytes

    mismatch = copy.deepcopy(preload.element_states)
    mismatch_bytes = canonical_json_bytes(mismatch[1])
    incompatible_u = preload.displacements.copy()
    incompatible_u[model.mesh.get_node(2).dofs[0]] += 1.0e-12
    model, load = _loaded_q4_model()
    with pytest.raises(ValueError, match="displacement"):
        nonlinear_static.solve_static_nonlinear(
            model,
            load,
            max_load_factor=0.1,
            num_steps=1,
            num_layers=3,
            initial_element_states=mismatch,
            initial_displacements=incompatible_u,
            equilibrate_initial_state=False,
        )
    assert calls == 0
    assert canonical_json_bytes(mismatch[1]) == mismatch_bytes


def test_historical_unbound_q4_restart_migrates_only_with_exact_restart_displacement() -> None:
    preload = _solve_loaded(maximum=0.5)
    historical = copy.deepcopy(preload.element_states)
    historical[1].pop("qualified_q4_committed_binding")
    historical[1].pop("state_integrity_sha256")
    historical[1].pop("qualified_q4_algorithmic_origin", None)
    historical_bytes = canonical_json_bytes(historical[1])
    model, load = _loaded_q4_model()
    migrated = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        max_load_factor=0.1,
        num_steps=1,
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        convergence_settings="legacy",
        initial_element_states=historical,
        initial_displacements=preload.displacements.copy(),
        equilibrate_initial_state=False,
    )
    assert migrated.status == "completed"
    assert migrated.info["qualified_q4_committed_state_lifecycle"][
        "migrated_historical_unbound_element_ids"
    ] == [1]
    assert canonical_json_bytes(historical[1]) == historical_bytes
    _assert_q4_state_is_exactly_bound(model, migrated)

    model, load = _loaded_q4_model()
    with pytest.raises(ValueError, match="historical unbound qualified Q4"):
        nonlinear_static.solve_static_nonlinear(
            model,
            load,
            max_load_factor=0.1,
            num_steps=1,
            num_layers=3,
            initial_element_states=historical,
        )


def test_q4_checkpoint_round_trip_retains_and_revalidates_closed_binding() -> None:
    model, load = _loaded_q4_model()
    first = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        max_load_factor=0.5,
        num_steps=1,
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    raw = first.restart_checkpoint_bytes()
    assert canonical_checkpoint_json_bytes(first.restart_checkpoint) == raw
    assert first.restart_checkpoint["element_states"][0]["state"][
        "state_integrity_sha256"
    ] == first.element_states[1]["state_integrity_sha256"]

    resumed_model, resumed_load = _loaded_q4_model()
    resumed = nonlinear_static.solve_static_nonlinear(
        resumed_model,
        resumed_load,
        max_load_factor=1.0,
        num_steps=1,
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        convergence_settings="legacy",
        restart_checkpoint=raw,
    )
    assert resumed.status == "completed"
    assert resumed.info["qualified_q4_committed_state_lifecycle"][
        "validated_bound_element_ids"
    ] == [1]
    assert first.restart_checkpoint_bytes() == raw
    _assert_q4_state_is_exactly_bound(resumed_model, resumed)
    assert resumed.restart_checkpoint["element_states"][0]["state"][
        "state_integrity_sha256"
    ] == resumed.element_states[1]["state_integrity_sha256"]


def _arc_control(steps: int) -> ArcLengthControl:
    return ArcLengthControl(
        initial_load_increment=0.05,
        minimum_load_increment=0.05,
        maximum_load_increment=0.05,
        growth_factor=1.0,
        stop_after_peak_steps=20,
        max_steps=steps,
    )


def test_q4_arc_length_stopped_boundary_and_checkpoint_restart_are_closed() -> None:
    model, load = _loaded_q4_model()
    first = solve_static_arc_length(
        model,
        load,
        control=_arc_control(1),
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
        emit_restart_checkpoint=True,
    )
    assert first.status == "maximum_steps_reached"
    first_digest = _assert_q4_state_is_exactly_bound(model, first)
    assert first.restart_checkpoint["element_states"][0]["state"][
        "state_integrity_sha256"
    ] == first_digest
    raw = first.restart_checkpoint_bytes()

    resumed_model, resumed_load = _loaded_q4_model()
    resumed = solve_static_arc_length(
        resumed_model,
        resumed_load,
        control=_arc_control(1),
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
        restart_checkpoint=raw,
    )
    assert resumed.status == "maximum_steps_reached"
    lifecycle = resumed.info["qualified_q4_committed_state_lifecycle"]
    assert lifecycle["validated_bound_element_ids"] == [1]
    assert lifecycle["stripped_internal_binding_element_ids"] == [1]
    assert first.restart_checkpoint_bytes() == raw
    _assert_q4_state_is_exactly_bound(resumed_model, resumed)
    assert resumed.restart_checkpoint["element_states"][0]["state"][
        "state_integrity_sha256"
    ] == resumed.element_states[1]["state_integrity_sha256"]


def test_arc_q4_binding_rejects_partial_mismatch_and_unbound_history_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero_model, zero_load = _loaded_q4_model()
    zero = solve_static_arc_length(
        zero_model,
        zero_load,
        control=_arc_control(1),
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
    )
    model, load = _loaded_q4_model()
    original = QualifiedE4PLShellElement.compute_nonlinear_response
    calls = 0

    def observed(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        QualifiedE4PLShellElement,
        "compute_nonlinear_response",
        observed,
    )

    partial = copy.deepcopy(zero.element_states)
    partial[1].pop("state_integrity_sha256")
    partial_bytes = canonical_json_bytes(partial[1])
    with pytest.raises(ElementCapabilityError, match="CRITICAL_API_MISMATCH"):
        solve_static_arc_length(
            model,
            load,
            control=_arc_control(1),
            num_layers=3,
            initial_element_states=partial,
        )
    assert calls == 0
    monkeypatch.setattr(
        QualifiedE4PLShellElement,
        "compute_nonlinear_response",
        original,
    )
    with pytest.raises(ValueError, match="partial committed-state binding"):
        solve_static_arc_length(
            model,
            load,
            control=_arc_control(1),
            num_layers=3,
            initial_element_states=partial,
        )
    assert calls == 0
    assert canonical_json_bytes(partial[1]) == partial_bytes

    # The previous arc state is nonzero and cannot be attached to the new
    # zero-displacement base merely because its constitutive arrays are valid.
    mismatch = copy.deepcopy(zero.element_states)
    mismatch_bytes = canonical_json_bytes(mismatch[1])
    model, load = _loaded_q4_model()
    with pytest.raises(ValueError, match="displacement"):
        solve_static_arc_length(
            model,
            load,
            control=_arc_control(1),
            num_layers=3,
            initial_element_states=mismatch,
        )
    assert calls == 0
    assert canonical_json_bytes(mismatch[1]) == mismatch_bytes

    historical = copy.deepcopy(zero.element_states)
    historical[1].pop("qualified_q4_committed_binding")
    historical[1].pop("state_integrity_sha256")
    historical_bytes = canonical_json_bytes(historical[1])
    model, load = _loaded_q4_model()
    with pytest.raises(ValueError, match="historical unbound qualified Q4"):
        solve_static_arc_length(
            model,
            load,
            control=_arc_control(1),
            num_layers=3,
            initial_element_states=historical,
        )
    assert calls == 0
    assert canonical_json_bytes(historical[1]) == historical_bytes


def test_fully_constrained_arc_q4_is_materialized_and_exactly_sealed() -> None:
    model = FEModel("q4-arc-fully-constrained")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1,
            [1, 2, 3, 4],
            "steel",
            thickness=0.01,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    model.add_boundary_condition(FixedSupport("all", [1, 2, 3, 4]))
    result = solve_static_arc_length(
        model,
        None,
        control=_arc_control(1),
        num_layers=3,
    )
    assert result.status == "empty_reduced_system"
    _assert_q4_state_is_exactly_bound(model, result)


def test_fully_constrained_plastic_q4_records_a_virgin_algorithmic_origin() -> None:
    def model() -> FEModel:
        made = FEModel("q4-plastic-fully-constrained")
        made.add_material(
            "steel",
            210.0e9,
            0.3,
            density=7850.0,
            hardening_curve=DNVC208MaterialCurve(
                sigma_prop=320.0e6,
                sigma_yield=357.0e6,
                sigma_yield_2=363.3e6,
                eps_p_y1=0.004,
                eps_p_y2=0.015,
                K=740.0e6,
                n=0.166,
            ),
        )
        for node_id, coordinates in enumerate(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            start=1,
        ):
            made.add_node(node_id, *coordinates)
        made.add_element(
            1,
            QualifiedE4PLShellElement(
                1,
                [1, 2, 3, 4],
                "steel",
                thickness=0.01,
                reference_normal=(0.0, 0.0, 1.0),
            ),
        )
        made.add_boundary_condition(FixedSupport("all", [1, 2, 3, 4]))
        return made

    static_model = model()
    static = nonlinear_static.solve_static_nonlinear(
        static_model,
        num_steps=1,
        num_layers=3,
    )
    assert static.status == "empty_reduced_system"
    assert "qualified_q4_algorithmic_origin" in static.element_states[1]
    _assert_q4_state_is_exactly_bound(static_model, static)

    arc_model = model()
    arc = solve_static_arc_length(
        arc_model,
        None,
        control=_arc_control(1),
        num_layers=3,
    )
    assert arc.status == "empty_reduced_system"
    assert "qualified_q4_algorithmic_origin" in arc.element_states[1]
    _assert_q4_state_is_exactly_bound(arc_model, arc)

    seeded_arc_model = model()
    seed = seeded_arc_model.mesh.elements[1].init_nonlinear_state(3)
    seed["alpha"][:] = 1.0e-6
    before = canonical_json_bytes(seed)
    seeded_arc = solve_static_arc_length(
        seeded_arc_model,
        None,
        control=_arc_control(1),
        num_layers=3,
        initial_element_states={1: seed},
    )
    assert seeded_arc.status == "empty_reduced_system"
    assert "qualified_q4_algorithmic_origin" in seeded_arc.element_states[1]
    assert seeded_arc.info["qualified_q4_committed_state_lifecycle"][
        "origin_materialized_from_unevaluated_parent_element_ids"
    ] == [1]
    assert canonical_json_bytes(seed) == before
    _assert_q4_state_is_exactly_bound(seeded_arc_model, seeded_arc)


def test_q4_arc_constant_preload_offset_is_persisted_for_exact_checkpoint_resume() -> None:
    def cases() -> tuple[FEModel, LoadCase, LoadCase]:
        model, proportional = _loaded_q4_model()
        constant = LoadCase("q4-constant-preload")
        constant.add_nodal_load(2, [5.0e4, 0.0, 0.0, 0.0, 0.0, 0.0])
        constant.add_nodal_load(3, [5.0e4, 0.0, 0.0, 0.0, 0.0, 0.0])
        return model, proportional, constant

    full_model, full_load, full_constant = cases()
    full = solve_static_arc_length(
        full_model,
        full_load,
        constant_load_case=full_constant,
        control=_arc_control(2),
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
        emit_restart_checkpoint=True,
    )
    first_model, first_load, first_constant = cases()
    first = solve_static_arc_length(
        first_model,
        first_load,
        constant_load_case=first_constant,
        control=_arc_control(1),
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
        emit_restart_checkpoint=True,
    )
    offset = np.asarray(
        first.restart_checkpoint["path_state"]["exact_base_offset"],
        dtype=np.float64,
    )
    assert offset.shape == first.displacements.shape
    assert np.all(np.isfinite(offset))

    resumed_model, resumed_load, resumed_constant = cases()
    resumed = solve_static_arc_length(
        resumed_model,
        resumed_load,
        constant_load_case=resumed_constant,
        control=_arc_control(1),
        num_layers=3,
        max_iterations=15,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
        restart_checkpoint=first.restart_checkpoint_bytes(),
    )
    assert full.status == first.status == resumed.status == "maximum_steps_reached"
    np.testing.assert_array_equal(full.displacements, resumed.displacements)
    assert canonical_json_bytes(full.element_states[1]) == canonical_json_bytes(
        resumed.element_states[1]
    )
    assert full.restart_checkpoint_bytes() == resumed.restart_checkpoint_bytes()


def test_explicit_virgin_q4_material_seed_is_not_misclassified_as_restart() -> None:
    model, load = _loaded_q4_model()
    element = model.mesh.elements[1]
    seed = element.init_nonlinear_state(3)
    seed["alpha"][:] = 1.0e-6
    before = canonical_json_bytes(seed)
    result = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        max_load_factor=0.1,
        num_steps=1,
        num_layers=3,
        initial_element_states={1: seed},
        convergence_settings="legacy",
    )
    assert result.status == "completed"
    lifecycle = result.info["qualified_q4_committed_state_lifecycle"]
    assert lifecycle["explicit_initial_material_state_element_ids"] == [1]
    assert lifecycle["migrated_historical_unbound_element_ids"] == []
    assert canonical_json_bytes(seed) == before
    _assert_q4_state_is_exactly_bound(model, result)


def test_initial_equilibration_failure_returns_owned_nonauthoritative_q4_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, load = _loaded_q4_model()

    def forced_failure(**kwargs):
        return (
            np.asarray(kwargs["initial_reduced_displacements"], dtype=float).copy(),
            kwargs["committed_states"],
            [{"iteration": 1, "residual_norm": 1.0}],
            "forced_initial_state_failure",
        )

    monkeypatch.setattr(
        nonlinear_static,
        "_equilibrate_initial_fields",
        forced_failure,
    )
    result = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        num_layers=3,
        initial_fields={
            1: ShellInitialField(membrane_stress=[5.0e7, 0.0, 0.0])
        },
    )
    assert result.status == "diverged"
    assert isinstance(result.element_states, dict)
    state = result.element_states[1]
    disposition = state["qualified_q4_activity_disposition"]
    assert disposition["status"] == "FAILED_NONAUTHORITATIVE"
    assert "qualified_q4_committed_binding" not in state
    element = model.mesh.elements[1]
    element.validate_noncurrent_failed_state(
        model.mesh,
        model.get_material(element.material_name),
        state,
        3,
    )
    hybrid = copy.deepcopy(state)
    hybrid["qualified_q4_committed_binding"] = {"forbidden": True}
    hybrid["state_integrity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="must not carry an ACTIVE"):
        element.validate_noncurrent_failed_state(
            model.mesh,
            model.get_material(element.material_name),
            hybrid,
            3,
        )
    dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    with pytest.raises(ValueError, match="noncurrent"):
        element.validate_committed_current_tangent_binding(
            model.mesh,
            model.get_material(element.material_name),
            result.displacements[dofs],
            state,
            3,
        )
    assert result.info["qualified_q4_committed_state_lifecycle"][
        "final_state_policy"
    ] == "FAILED_NONAUTHORITATIVE_NO_ACTIVE_SEAL"


def test_arc_initial_equilibrium_failure_is_owned_but_never_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, load = _loaded_q4_model()
    original = arc_length_module._assemble_nonlinear_system

    def forced_residual(*args, **kwargs):
        force, tangent, states = original(*args, **kwargs)
        made = np.asarray(force, dtype=np.float64).copy()
        made[model.mesh.get_node(2).dofs[0]] += 1.0e12
        return made, tangent, states

    monkeypatch.setattr(
        arc_length_module,
        "_assemble_nonlinear_system",
        forced_residual,
    )
    result = solve_static_arc_length(
        model,
        load,
        control=_arc_control(1),
        num_layers=3,
    )
    assert result.status == "initial_equilibrium_failed"
    state = result.element_states[1]
    assert state["qualified_q4_activity_disposition"]["status"] == (
        "FAILED_NONAUTHORITATIVE"
    )
    assert "qualified_q4_committed_binding" not in state
    element = model.mesh.elements[1]
    element.validate_noncurrent_failed_state(
        model.mesh,
        model.get_material(element.material_name),
        state,
        3,
    )


def test_fully_constrained_plastic_q4_with_initial_field_keeps_true_origin() -> None:
    model, _load = _loaded_q4_model()
    model.add_boundary_condition(FixedSupport("all-extra", [1, 2, 3, 4]))
    # Owner decision 2026-10-23: plastic qualified Q4 outside the assumed-strain
    # route scope is refused (the compatible-strain plastic route with the
    # qualified correction is unstable once layers yield).
    from anyfesolver._q4_assumed_strain_plasticity import UnsupportedQ4PlasticRouteError
    with pytest.raises(UnsupportedQ4PlasticRouteError, match="active initial"):
        nonlinear_static.solve_static_nonlinear(
            model,
            num_steps=1,
            num_layers=3,
            initial_fields={
                1: ShellInitialField(membrane_stress=[5.0e7, 0.0, 0.0])
            },
        )


def test_q4_static_fixed_affine_restart_checkpoint_restores_exact_base() -> None:
    def cases() -> tuple[FEModel, LoadCase]:
        model, load = _loaded_q4_model()
        model.add_boundary_condition(
            BoundaryCondition("prescribed-right-x", [2, 3], {"ux": 2.0e-4})
        )
        return model, load

    preload_model, preload_load = cases()
    preload = nonlinear_static.solve_static_nonlinear(
        preload_model,
        preload_load,
        max_load_factor=0.25,
        num_steps=1,
        num_layers=3,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    assert preload.status == "completed"

    full_model, full_load = cases()
    full = nonlinear_static.solve_static_nonlinear(
        full_model,
        full_load,
        max_load_factor=0.75,
        num_steps=3,
        num_layers=3,
        initial_displacements=preload.displacements.copy(),
        initial_element_states=copy.deepcopy(preload.element_states),
        equilibrate_initial_state=False,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    first_model, first_load = cases()
    first = nonlinear_static.solve_static_nonlinear(
        first_model,
        first_load,
        max_load_factor=0.5,
        num_steps=2,
        num_layers=3,
        initial_displacements=preload.displacements.copy(),
        initial_element_states=copy.deepcopy(preload.element_states),
        equilibrate_initial_state=False,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    path = first.restart_checkpoint["path_state"]
    assert path["affine_path_mode"] == "FIXED_RESTART_AFFINE_STATE"
    exact_base = np.asarray(path["exact_base_offset"], dtype=np.float64)
    assert exact_base.shape == first.displacements.shape
    raw = first.restart_checkpoint_bytes()

    resumed_model, resumed_load = cases()
    resumed = nonlinear_static.solve_static_nonlinear(
        resumed_model,
        resumed_load,
        max_load_factor=0.75,
        num_steps=1,
        num_layers=3,
        convergence_settings="legacy",
        restart_checkpoint=raw,
    )
    assert full.status == first.status == resumed.status == "completed"
    np.testing.assert_array_equal(full.displacements, resumed.displacements)
    assert canonical_json_bytes(full.element_states[1]) == canonical_json_bytes(
        resumed.element_states[1]
    )
    assert first.restart_checkpoint_bytes() == raw
    assert full.restart_checkpoint_bytes() == resumed.restart_checkpoint_bytes()

    legacy = copy.deepcopy(first.restart_checkpoint)
    legacy["path_state"].pop("affine_path_mode")
    legacy["path_state"].pop("exact_base_offset")
    legacy.pop("checkpoint_sha256")
    legacy["checkpoint_sha256"] = hashlib.sha256(
        canonical_checkpoint_json_bytes(legacy)
    ).hexdigest().upper()
    legacy_raw = canonical_checkpoint_json_bytes(legacy)
    legacy_model, legacy_load = cases()
    migrated = nonlinear_static.solve_static_nonlinear(
        legacy_model,
        legacy_load,
        max_load_factor=0.75,
        num_steps=1,
        num_layers=3,
        convergence_settings="legacy",
        restart_checkpoint=legacy_raw,
    )
    np.testing.assert_array_equal(full.displacements, migrated.displacements)
    assert canonical_json_bytes(full.element_states[1]) == canonical_json_bytes(
        migrated.element_states[1]
    )
    assert migrated.info["prescribed_displacement_path"][
        "restart_schema_disposition"
    ] == "MIGRATED_FROM_BOUND_COMMITTED_DISPLACEMENT"


def test_q4_current_state_replay_restores_cache_presence_and_identity() -> None:
    model, load = _loaded_q4_model()
    result = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        max_load_factor=0.5,
        num_steps=1,
        num_layers=3,
        convergence_settings="legacy",
    )
    element = model.mesh.elements[1]
    material = model.get_material(element.material_name)
    state = result.element_states[1]
    dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    local_u = np.asarray(result.displacements, dtype=np.float64)[dofs]

    # Exercise both retained-object and absent-attribute restoration.  Replay
    # may populate all of these internally, but none is committed evidence.
    cache_names = (
        "_nl_cache",
        "_nl_cache_key",
        "_qualified_component_guard",
        "_qualified_components",
        "_qualified_cache_key",
        "_hourglass_stiffness_matrix",
        "_stiffness_matrix",
    )
    for name in ("_hourglass_stiffness_matrix", "_stiffness_matrix"):
        if hasattr(element, name):
            delattr(element, name)
    retained = {
        name: getattr(element, name)
        for name in cache_names
        if hasattr(element, name)
    }
    retained_snapshots = {
        name: _cache_tree_snapshot(value) for name, value in retained.items()
    }

    def assert_exactly_restored() -> None:
        for name in cache_names:
            if name not in retained:
                assert not hasattr(element, name)
                continue
            value = getattr(element, name)
            assert value is retained[name]
            assert _cache_tree_snapshot(value) == retained_snapshots[name]

    element.validate_committed_current_tangent_semantics(
        model.mesh, material, local_u, state, 3
    )
    element.compute_committed_current_tangent_components(
        model.mesh, material, local_u, state, 3
    )
    assert_exactly_restored()

    invalid = copy.deepcopy(state)
    invalid.pop("qualified_q4_committed_binding")
    invalid.pop("state_integrity_sha256")
    invalid["plastic_strain"] = np.asarray(
        invalid["plastic_strain"], dtype=np.float64
    ).copy()
    invalid["plastic_strain"].flat[0] += 1.0e-9
    with pytest.raises(ValueError, match="does not reproduce committed"):
        element.seal_committed_current_tangent_state(
            model.mesh, material, local_u, invalid, 3
        )
    assert_exactly_restored()


def test_q4_checkpoint_rejects_failed_unsealed_and_false_deleted_states_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_model, source_load = _loaded_q4_model()
    source = nonlinear_static.solve_static_nonlinear(
        source_model,
        source_load,
        max_load_factor=0.5,
        num_steps=1,
        num_layers=3,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    assert source.status == "completed"
    element = source_model.mesh.elements[1]
    material = source_model.get_material(element.material_name)
    dofs = np.asarray(element.get_dof_mapping(source_model.mesh), dtype=np.intp)
    local_u = np.asarray(source.displacements, dtype=np.float64)[dofs]
    failed = element.mark_noncurrent_failed_state(
        source_model.mesh,
        material,
        local_u,
        source.element_states[1],
        3,
        failure_reason="checkpoint-negative-fixture",
    )
    element.validate_noncurrent_failed_state(
        source_model.mesh,
        material,
        failed,
        3,
    )
    foreign_active = copy.deepcopy(source.element_states[1])
    foreign_active["qualified_s3_activity_disposition"] = {
        "status": "FAILED_NONAUTHORITATIVE"
    }
    with pytest.raises(ValueError, match="foreign S3"):
        element.validate_committed_current_tangent_state(
            source_model.mesh,
            material,
            local_u,
            foreign_active,
            3,
        )

    variants: list[dict[str, object]] = []

    failed_checkpoint = copy.deepcopy(source.restart_checkpoint)
    failed_checkpoint["element_states"][0]["state"] = failed
    variants.append(failed_checkpoint)

    unsealed_checkpoint = copy.deepcopy(source.restart_checkpoint)
    unsealed_checkpoint["element_states"][0]["state"][
        "state_integrity_sha256"
    ] = "0" * 64
    variants.append(unsealed_checkpoint)

    false_deleted_checkpoint = copy.deepcopy(source.restart_checkpoint)
    false_deleted_checkpoint["deleted_element_ids"] = [1]
    variants.append(false_deleted_checkpoint)

    activity_mismatch_checkpoint = copy.deepcopy(source.restart_checkpoint)
    activity = ElementActivity([1])
    activity.hard_delete([1], step=1, reason="checkpoint-negative-fixture")
    activity_mismatch_checkpoint["activity_state"] = activity.to_restart(
        include_history=True
    )
    variants.append(activity_mismatch_checkpoint)

    foreign_family_checkpoint = copy.deepcopy(source.restart_checkpoint)
    foreign_family_checkpoint["element_states"][0]["state"][
        "qualified_s3_activity_disposition"
    ] = {"status": "DELETED_FROZEN_NONCURRENT"}
    variants.append(foreign_family_checkpoint)

    missing_state_checkpoint = copy.deepcopy(source.restart_checkpoint)
    missing_state_checkpoint["element_states"] = []
    variants.append(missing_state_checkpoint)

    for checkpoint in variants:
        body = {
            key: value
            for key, value in checkpoint.items()
            if key != "checkpoint_sha256"
        }
        checkpoint["checkpoint_sha256"] = hashlib.sha256(
            canonical_checkpoint_json_bytes(body)
        ).hexdigest().upper()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("checkpoint lifecycle guard reached mechanics")

    monkeypatch.setattr(
        nonlinear_static,
        "_assemble_nonlinear_system",
        forbidden,
    )
    for checkpoint in variants:
        resumed_model, resumed_load = _loaded_q4_model()
        with pytest.raises(
            NonlinearCheckpointError, match="incompatible|disagree|has no"
        ):
            nonlinear_static.solve_static_nonlinear(
                resumed_model,
                resumed_load,
                max_load_factor=0.75,
                num_steps=1,
                num_layers=3,
                convergence_settings="legacy",
                restart_checkpoint=canonical_checkpoint_json_bytes(checkpoint),
            )


def test_q4_checkpoint_producer_cannot_emit_missing_qualified_history() -> None:
    model, _load = _loaded_q4_model()
    with pytest.raises(NonlinearCheckpointError, match="has no committed state"):
        create_nonlinear_checkpoint(
            analysis_kind="static",
            model=model,
            analysis_contract={"num_layers": 3},
            displacements=np.zeros(
                model.mesh.dof_manager.total_dofs, dtype=np.float64
            ),
            element_states={},
            path_state={},
        )


def test_q4_checkpoint_nested_mapping_observation_is_guarded_before_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_model, source_load = _loaded_q4_model()
    source = nonlinear_static.solve_static_nonlinear(
        source_model,
        source_load,
        max_load_factor=0.5,
        num_steps=1,
        num_layers=3,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    checkpoint = source.to_restart_checkpoint()
    plain_path_state = checkpoint["path_state"]
    reached: list[str] = []
    original_numpy_all = np.all

    def forbidden_numeric(*_args: object, **_kwargs: object) -> bool:
        reached.append("numeric")
        raise AssertionError("changed numerical operation was invoked")

    class ObservedPath(dict[str, object]):
        def items(self):  # type: ignore[override]
            reached.append("mapping")
            monkeypatch.setattr(np, "all", forbidden_numeric)
            return super().items()

    checkpoint["path_state"] = ObservedPath(checkpoint["path_state"])
    resumed_model, resumed_load = _loaded_q4_model()
    with pytest.raises(AssemblyError, match="qualified shell authority") as caught:
        nonlinear_static.solve_static_nonlinear(
            resumed_model,
            resumed_load,
            max_load_factor=0.75,
            num_steps=1,
            num_layers=3,
            convergence_settings="legacy",
            restart_checkpoint=checkpoint,
        )
    assert reached == ["mapping"]
    assert isinstance(caught.value.__cause__, NonlinearCheckpointError)
    assert "authority changed during input observation" in str(
        caught.value.__cause__
    )
    element = resumed_model.mesh.elements[1]
    for name in (
        "_hourglass_stiffness_matrix",
        "_internal_forces",
        "_mass_matrix",
        "_nl_cache",
        "_nl_cache_key",
        "_qualified_cache_key",
        "_qualified_component_guard",
        "_qualified_components",
        "_stiffness_matrix",
    ):
        assert getattr(element, name, None) is None

    monkeypatch.setattr(np, "all", original_numpy_all)
    checkpoint["path_state"] = plain_path_state
    clean = nonlinear_static.solve_static_nonlinear(
        resumed_model,
        resumed_load,
        max_load_factor=0.75,
        num_steps=1,
        num_layers=3,
        convergence_settings="legacy",
        restart_checkpoint=checkpoint,
    )
    assert clean.status == "completed"


@pytest.mark.parametrize("callback_kind", ("cancellation", "status"))
def test_q4_nonlinear_callbacks_recheck_authority_before_newton_work(
    callback_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, load = _loaded_q4_model()
    reached: list[str] = []

    def forbidden_numeric(*_args: object, **_kwargs: object) -> np.ndarray:
        reached.append("numeric")
        raise AssertionError("changed numerical operation was invoked")

    def change_runtime() -> None:
        reached.append(callback_kind)
        monkeypatch.setattr(np, "asarray", forbidden_numeric)

    class ObservedToken(CancellationToken):
        def raise_if_cancelled(self, stage: str = "") -> None:
            if stage == "nonlinear_static.force.step:0.start":
                change_runtime()

    def status_callback(_message: str) -> None:
        change_runtime()

    kwargs: dict[str, object] = (
        {"cancellation_token": ObservedToken()}
        if callback_kind == "cancellation"
        else {"status_callback": status_callback}
    )
    with pytest.raises(ElementCapabilityError, match="NUMERICAL_AUTHORITY_MISMATCH"):
        nonlinear_static.solve_static_nonlinear(
            model,
            load,
            max_load_factor=0.1,
            num_steps=1,
            num_layers=3,
            convergence_settings="legacy",
            **kwargs,
        )
    assert reached == [callback_kind]


def test_q4_checkpoint_producer_guards_module_helpers_before_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _load = _loaded_q4_model()
    reached: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        reached.append("stationary")
        raise AssertionError("checkpoint producer reached mutated helper")

    monkeypatch.setattr(q4_element_module, "_stationary_blocks", forbidden)
    with pytest.raises(
        NonlinearCheckpointError, match="qualified element API authority"
    ):
        create_nonlinear_checkpoint(
            analysis_kind="static",
            model=model,
            analysis_contract={"num_layers": 3},
            displacements=np.zeros(model.mesh.dof_manager.total_dofs),
            element_states={},
            path_state={},
        )
    assert reached == []


def test_unknown_id_q4_descendant_cannot_downgrade_checkpoint_authority() -> None:
    model, _load = _loaded_q4_model()
    reached: list[str] = []

    class ForeignQ4(QualifiedE4PLShellElement):
        formulation_id = "FOREIGN_Q4_DOWNGRADE"

        def __getattribute__(self, name: str) -> object:
            reached.append(name)
            return super().__getattribute__(name)

        def to_dict(self) -> dict[str, object]:
            reached.append("to_dict")
            return ShellElement.to_dict(self)

    original = model.mesh.elements[1]
    foreign = QualifiedE4PLShellElement(
        1,
        list(original.node_ids),
        original.material_name,
        thickness=original.thickness,
        reference_normal=(0.0, 0.0, 1.0),
    )
    # Exact qualified construction rejects descendants.  Create a valid
    # instance first, then inject the invalid descendant identity solely for
    # this consumer-side checkpoint-authority negative test.
    object.__setattr__(foreign, "__class__", ForeignQ4)
    model.mesh.elements[1] = foreign
    reached.clear()

    with pytest.raises(
        NonlinearCheckpointError, match="qualified element API authority"
    ):
        create_nonlinear_checkpoint(
            analysis_kind="static",
            model=model,
            analysis_contract={"num_layers": 3},
            displacements=np.zeros(model.mesh.dof_manager.total_dofs),
            element_states={},
            path_state={},
        )
    assert reached == []


@pytest.mark.parametrize(
    "helper_name",
    (
        "_local_frame_and_derivatives",
        "_mitc4_shear_b_matrix",
        "to_dict",
    ),
)
def test_q4_checkpoint_rejects_spoofed_base_kernel_before_fingerprint_or_replay(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    source_model, source_load = _loaded_q4_model()
    source = nonlinear_static.solve_static_nonlinear(
        source_model,
        source_load,
        max_load_factor=0.5,
        num_steps=1,
        num_layers=3,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    assert source.status == "completed"
    resumed_model, resumed_load = _loaded_q4_model()
    reached: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        reached.append(helper_name)
        raise AssertionError("spoofed Q4 base kernel reached checkpoint mechanics")

    monkeypatch.setattr(ShellElement, helper_name, forbidden)
    monkeypatch.setattr(
        nonlinear_static,
        "_assemble_nonlinear_system",
        forbidden,
    )

    with pytest.raises(
        (NonlinearCheckpointError, ElementCapabilityError),
        match=(
            "qualified element API authority|DEPENDENCY_AUTHORITY_MISMATCH|"
            "CLASS_NAMESPACE_MISMATCH|BASE_CRITICAL_API_MISMATCH"
        ),
    ):
        nonlinear_static.solve_static_nonlinear(
            resumed_model,
            resumed_load,
            max_load_factor=0.75,
            num_steps=1,
            num_layers=3,
            convergence_settings="legacy",
            restart_checkpoint=source.restart_checkpoint_bytes(),
        )
    assert reached == []
