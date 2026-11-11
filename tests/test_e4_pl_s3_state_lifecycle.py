from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np
import pytest

import anyfesolver.e4_pl_s3_element as s3_module
import anyfesolver.nonlinear_performance as nonlinear_performance
import anyfesolver.nonlinear_static as nonlinear_static
from anyfesolver import (
    FEModel,
    QualifiedE4PLS3ShellElement,
    QualifiedE4PLShellElement,
    solve_eigenvalue_buckling,
    solve_free_vibration,
    solve_static_nonlinear,
)
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.e4_pl_s3_state import canonical_json_bytes, seal_committed_s3_state
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.fracture import FractureConfig
from anyfesolver.nonlinear_restart import (
    NonlinearCheckpointError,
    canonical_checkpoint_json_bytes,
    create_nonlinear_checkpoint,
)
from anyfesolver.nonlinear_static import ShellInitialField


def _pure_s3_model() -> tuple[FEModel, LoadCase, int]:
    model = FEModel("qualified-s3-lifecycle")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.01,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    model.add_boundary_condition(FixedSupport("node-1", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "node-2-axial-only",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(FixedSupport("node-3", [3]))
    load = LoadCase("qualified-s3-pull")
    load.add_nodal_load(2, [1.0e3, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load, 1


def _mixed_s3_q4_model() -> tuple[FEModel, LoadCase, int]:
    model = FEModel("qualified-s3-q4-mixed-lifecycle")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.5, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    # S3 is deliberately the lower element ID so its guard is exercised
    # before the independently ACTIVE Q4 in mixed current-state routes.
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1,
            [2, 5, 3],
            "steel",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    model.add_element(
        2,
        QualifiedE4PLShellElement(
            2,
            [1, 2, 3, 4],
            "steel",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    model.add_boundary_condition(FixedSupport("left", [1, 4]))
    load = LoadCase("qualified-mixed-pull")
    load.add_nodal_load(2, [-1.0e3, 0.0, 0.0, 0.0, 0.0, 0.0])
    load.add_nodal_load(3, [-1.0e3, 0.0, 0.0, 0.0, 0.0, 0.0])
    load.add_nodal_load(5, [-0.5e3, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load, 1


def _fracture_seed(
    model: FEModel,
    element_id: int,
) -> dict[str, Any]:
    element = model.mesh.elements[element_id]
    state = element.init_model_bound_nonlinear_state(
        model.mesh,
        model.get_material(element.material_name),
        3,
    )
    state["alpha"][:] = 0.01
    state = seal_committed_s3_state(state)
    element.validate_model_bound_nonlinear_state(
        model.mesh,
        model.get_material(element.material_name),
        state,
        3,
        expected_committed_total_u=np.zeros(18, dtype=np.float64),
    )
    return state


def _fracture_config() -> FractureConfig:
    return FractureConfig(
        threshold=0.001,
        residual_stiffness_fraction=0.1,
        max_deleted_fraction=1.0,
    )


def test_s3_active_state_rejects_foreign_q4_lifecycle_marker() -> None:
    model, _load, element_id = _pure_s3_model()
    element = model.mesh.elements[element_id]
    state = _fracture_seed(model, element_id)
    state["qualified_q4_activity_disposition"] = {
        "status": "FAILED_NONAUTHORITATIVE"
    }
    with pytest.raises(ValueError, match="foreign Q4"):
        element.validate_model_bound_nonlinear_state(
            model.mesh,
            model.get_material(element.material_name),
            state,
            3,
            expected_committed_total_u=np.zeros(18, dtype=np.float64),
        )


@pytest.mark.parametrize("builder", (_pure_s3_model, _mixed_s3_q4_model))
def test_real_s3_deletion_is_frozen_noncurrent_and_checkpoint_exact(
    builder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, load, s3_id = builder()
    seed = _fracture_seed(model, s3_id)
    seed_bytes = canonical_json_bytes(seed)
    first = solve_static_nonlinear(
        model,
        load,
        max_load_factor=1.0,
        num_steps=2,
        num_layers=3,
        initial_element_states={s3_id: seed},
        fracture_config=_fracture_config(),
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    assert first.status == "completed"
    assert canonical_json_bytes(seed) == seed_bytes
    assert first.info["fracture_summary"]["deleted_element_ids"] == [s3_id]
    state = first.element_states[s3_id]
    marker = state["qualified_s3_activity_disposition"]
    assert marker["status"] == "DELETED_FROZEN_NONCURRENT"
    assert marker["quadrature_authority_id"] == (
        "MITC3_PLUS_SEVEN_POINT_STIFFNESS_AND_SHEAR_IMMUTABLE_EXACT_V1"
    )
    assert marker["operator_semantics"] == (
        "CONSTITUTIVE_HISTORY_FROZEN;"
        "FORCE_AND_TANGENT_REEVALUATED_AT_CURRENT_U_THEN_SCALED"
    )
    deletion_u = np.asarray(marker["accepted_local_u"], dtype=np.float64)
    np.testing.assert_array_equal(state["committed_total_u"], deletion_u)
    element = model.mesh.elements[s3_id]
    dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    assert not np.array_equal(first.displacements[dofs], deletion_u)
    record = first.info["fracture_summary"]["records"][0]
    element.validate_noncurrent_deleted_state(
        model.mesh,
        model.get_material(element.material_name),
        state,
        3,
        expected_deletion_step_index=record["step_index"],
        expected_deletion_load_factor=record["load_factor"],
        expected_residual_stiffness_fraction=0.1,
        expected_trigger_name=record["trigger_name"],
    )

    original_state_bytes = canonical_json_bytes(state)
    raw_checkpoint = first.restart_checkpoint_bytes()
    resumed_model, resumed_load, resumed_s3_id = builder()
    resumed = solve_static_nonlinear(
        resumed_model,
        resumed_load,
        max_load_factor=1.5,
        num_steps=1,
        num_layers=3,
        fracture_config=_fracture_config(),
        convergence_settings="legacy",
        restart_checkpoint=raw_checkpoint,
        emit_restart_checkpoint=True,
    )
    assert resumed.status == "completed"
    resumed_state = resumed.element_states[resumed_s3_id]
    assert canonical_json_bytes(resumed_state) == original_state_bytes
    np.testing.assert_array_equal(
        resumed_state["committed_total_u"], deletion_u
    )
    resumed_dofs = np.asarray(
        resumed_model.mesh.elements[resumed_s3_id].get_dof_mapping(
            resumed_model.mesh
        ),
        dtype=np.intp,
    )
    assert not np.array_equal(resumed.displacements[resumed_dofs], deletion_u)
    assert resumed.info["qualified_s3_committed_state_lifecycle"][
        "restored_deleted_frozen_element_ids"
    ] == [resumed_s3_id]
    assert first.restart_checkpoint_bytes() == raw_checkpoint

    mismatched = first.to_restart_checkpoint()
    mismatched["path_state"]["deletion_records"][0][
        "trigger_name"
    ] = "mismatched-checkpoint-trigger"
    body = {
        key: value
        for key, value in mismatched.items()
        if key != "checkpoint_sha256"
    }
    mismatched["checkpoint_sha256"] = hashlib.sha256(
        canonical_checkpoint_json_bytes(body)
    ).hexdigest().upper()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("deletion metadata guard reached mechanics")

    monkeypatch.setattr(
        nonlinear_static,
        "assemble_stiffness_matrix",
        forbidden,
    )
    rejected_model, rejected_load, _rejected_id = builder()
    with pytest.raises(NonlinearCheckpointError, match="incompatible"):
        solve_static_nonlinear(
            rejected_model,
            rejected_load,
            max_load_factor=1.5,
            num_steps=1,
            num_layers=3,
            fracture_config=_fracture_config(),
            convergence_settings="legacy",
            restart_checkpoint=canonical_checkpoint_json_bytes(mismatched),
        )

    for name, replacement in (
        ("policy_id", "WRONG"),
        ("status", "ACTIVE"),
        ("quadrature_authority_id", "WRONG"),
        ("trigger_name", "wrong-trigger"),
    ):
        mutated = copy.deepcopy(state)
        mutated["qualified_s3_activity_disposition"][name] = replacement
        with pytest.raises(ValueError, match="disposition|trigger"):
            element.validate_noncurrent_deleted_state(
                model.mesh,
                model.get_material(element.material_name),
                mutated,
                3,
                expected_trigger_name=record["trigger_name"],
            )


def test_s3_checkpoint_rejects_spoofed_lifecycle_helper_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_model, source_load, s3_id = _pure_s3_model()
    source = solve_static_nonlinear(
        source_model,
        source_load,
        max_load_factor=1.0,
        num_steps=2,
        num_layers=3,
        initial_element_states={s3_id: _fracture_seed(source_model, s3_id)},
        fracture_config=_fracture_config(),
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    assert source.status == "completed"
    resumed_model, resumed_load, _resumed_s3_id = _pure_s3_model()
    reached: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        reached.append("lifecycle")
        raise AssertionError("spoofed S3 lifecycle helper reached checkpoint replay")

    monkeypatch.setattr(
        QualifiedE4PLS3ShellElement,
        "_validate_model_bound_nonlinear_state_core",
        forbidden,
    )
    monkeypatch.setattr(
        nonlinear_static,
        "_assemble_nonlinear_system",
        forbidden,
    )

    with pytest.raises(
        (NonlinearCheckpointError, ElementCapabilityError),
        match=(
            "qualified element API authority|CLASS_NAMESPACE_MISMATCH|"
            "CRITICAL_API_MISMATCH"
        ),
    ):
        solve_static_nonlinear(
            resumed_model,
            resumed_load,
            max_load_factor=1.5,
            num_steps=1,
            num_layers=3,
            fracture_config=_fracture_config(),
            convergence_settings="legacy",
            restart_checkpoint=source.restart_checkpoint_bytes(),
        )
    assert reached == []

@pytest.mark.parametrize("builder", (_pure_s3_model, _mixed_s3_q4_model))
def test_real_s3_failure_is_nonauthoritative_and_eigen_routes_reject_pre_mechanics(
    builder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, load, s3_id = builder()

    def forced_failure(**kwargs):
        return (
            np.asarray(
                kwargs["initial_reduced_displacements"], dtype=np.float64
            ).copy(),
            kwargs["committed_states"],
            [{"iteration": 1, "residual_norm": 1.0}],
            "forced_s3_initial_state_failure",
        )

    monkeypatch.setattr(
        nonlinear_static,
        "_equilibrate_initial_fields",
        forced_failure,
    )
    result = solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        num_layers=3,
        initial_fields={
            s3_id: ShellInitialField(
                membrane_stress=[5.0e7, 0.0, 0.0]
            )
        },
    )
    assert result.status == "diverged"
    state = result.element_states[s3_id]
    marker = state["qualified_s3_activity_disposition"]
    assert marker["status"] == "FAILED_NONAUTHORITATIVE"
    assert marker["semantics"] == (
        "MATERIALIZED_RESULT_ONLY_NOT_ACCEPTED_CURRENT_STATE_EVIDENCE"
    )
    element = model.mesh.elements[s3_id]
    element.validate_noncurrent_failed_state(
        model.mesh,
        model.get_material(element.material_name),
        state,
        3,
    )
    assert result.info["qualified_s3_committed_state_lifecycle"][
        "final_state_policy"
    ] == "FAILED_NONAUTHORITATIVE_NO_ACTIVE_SEAL"

    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("noncurrent S3 reached current mechanics")

    original_native_response = (
        s3_module._native_layered_uncondensed_response_components
    )
    monkeypatch.setattr(
        s3_module,
        "_native_layered_uncondensed_response_components",
        forbidden,
    )
    for solver in (solve_free_vibration, solve_eigenvalue_buckling):
        with pytest.raises(
            (ValueError, ElementCapabilityError),
            match=(
                "noncurrent|ACTIVE|sealed qualified Q4|"
                "MODULE_HELPER_MISMATCH|authority"
            ),
        ):
            solver(
                model,
                current_state_displacements=result.displacements,
                current_state_element_states=result.element_states,
                current_state_num_layers=3,
            )
    assert calls == 0
    monkeypatch.setattr(
        s3_module,
        "_native_layered_uncondensed_response_components",
        original_native_response,
    )

    mutated = copy.deepcopy(state)
    mutated["qualified_s3_activity_disposition"][
        "quadrature_authority_id"
    ] = "WRONG"
    with pytest.raises(ValueError, match="failed activity disposition"):
        element.validate_noncurrent_failed_state(
            model.mesh,
            model.get_material(element.material_name),
            mutated,
            3,
        )


def test_deleted_s3_is_rejected_by_mixed_modal_and_buckling_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, load, s3_id = _mixed_s3_q4_model()
    result = solve_static_nonlinear(
        model,
        load,
        max_load_factor=1.0,
        num_steps=2,
        num_layers=3,
        initial_element_states={s3_id: _fracture_seed(model, s3_id)},
        fracture_config=_fracture_config(),
        convergence_settings="legacy",
    )
    assert result.element_states[s3_id][
        "qualified_s3_activity_disposition"
    ]["status"] == "DELETED_FROZEN_NONCURRENT"
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("deleted S3 reached current mechanics")

    monkeypatch.setattr(
        s3_module,
        "_native_layered_uncondensed_response_components",
        forbidden,
    )
    for solver in (solve_free_vibration, solve_eigenvalue_buckling):
        with pytest.raises(
            (ValueError, ElementCapabilityError),
            match=(
                "displacement/state pairing|committed displacement|"
                "noncurrent|MODULE_HELPER_MISMATCH|authority"
            ),
        ):
            solver(
                model,
                current_state_displacements=result.displacements,
                current_state_element_states=result.element_states,
                current_state_num_layers=3,
            )
    assert calls == 0


def test_active_mixed_route_still_selects_optimized_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _load, s3_id = _mixed_s3_q4_model()
    states = {s3_id: _fracture_seed(model, s3_id)}
    sentinel = (np.zeros(1), None, {"optimized": True})
    calls: list[str] = []

    class Plan:
        def assemble(self, *_args, **_kwargs):
            calls.append("optimized")
            return sentinel

    monkeypatch.setattr(
        nonlinear_performance,
        "get_nonlinear_assembly_plan",
        lambda *_args, **_kwargs: Plan(),
    )
    monkeypatch.setattr(
        nonlinear_performance,
        "record_nonlinear_assembly_execution",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        nonlinear_performance,
        "_ORIGINAL_ASSEMBLER",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ordinary mixed path fell back to reference assembly")
        ),
    )
    result = nonlinear_performance._optimized_assemble_nonlinear_system(
        model,
        np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64),
        states,
        3,
        deleted_element_ids=(),
    )
    assert result is sentinel
    assert calls == ["optimized"]


def test_deleted_s3_checkpoint_binds_every_deletion_record_field_before_k0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, load, element_id = _pure_s3_model()
    source = solve_static_nonlinear(
        model,
        load,
        max_load_factor=1.0,
        num_steps=2,
        num_layers=3,
        initial_element_states={element_id: _fracture_seed(model, element_id)},
        fracture_config=_fracture_config(),
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    assert source.status == "completed"
    original = source.to_restart_checkpoint()
    record = original["path_state"]["deletion_records"][0]
    replacements = {
        "element_type": "beam",
        "step_index": int(record["step_index"]) + 1,
        "load_factor": float(record["load_factor"]) + 0.125,
        "trigger_name": "foreign_trigger",
        "trigger_value": float(record["trigger_value"]) + 0.125,
        "threshold": float(record["threshold"]) * 123.0,
        "location": "alpha[999]",
        "measure": float(record["measure"]) + 0.125,
    }
    reached: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        reached.append("k0")
        raise AssertionError("deletion record mutation reached K0 mechanics")

    monkeypatch.setattr(
        nonlinear_static, "assemble_stiffness_matrix", forbidden
    )
    for name, replacement in replacements.items():
        mutated = copy.deepcopy(original)
        mutated["path_state"]["deletion_records"][0][name] = replacement
        body = {
            key: value
            for key, value in mutated.items()
            if key != "checkpoint_sha256"
        }
        mutated["checkpoint_sha256"] = hashlib.sha256(
            canonical_checkpoint_json_bytes(body)
        ).hexdigest().upper()
        resumed_model, resumed_load, _ = _pure_s3_model()
        with pytest.raises(NonlinearCheckpointError, match="incompatible"):
            solve_static_nonlinear(
                resumed_model,
                resumed_load,
                max_load_factor=1.5,
                num_steps=1,
                num_layers=3,
                fracture_config=_fracture_config(),
                convergence_settings="legacy",
                restart_checkpoint=canonical_checkpoint_json_bytes(mutated),
            )
    foreign_family = copy.deepcopy(original)
    foreign_family["element_states"][0]["state"][
        "qualified_q4_activity_disposition"
    ] = {"status": "DELETED_FROZEN_NONCURRENT"}
    body = {
        key: value
        for key, value in foreign_family.items()
        if key != "checkpoint_sha256"
    }
    foreign_family["checkpoint_sha256"] = hashlib.sha256(
        canonical_checkpoint_json_bytes(body)
    ).hexdigest().upper()
    resumed_model, resumed_load, _ = _pure_s3_model()
    with pytest.raises(NonlinearCheckpointError, match="incompatible"):
        solve_static_nonlinear(
            resumed_model,
            resumed_load,
            max_load_factor=1.5,
            num_steps=1,
            num_layers=3,
            fracture_config=_fracture_config(),
            convergence_settings="legacy",
            restart_checkpoint=canonical_checkpoint_json_bytes(foreign_family),
        )
    assert reached == []


def test_s3_checkpoint_producer_guards_quadrature_helper_before_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _load, _element_id = _pure_s3_model()
    reached: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        reached.append("quadrature")
        raise AssertionError("checkpoint producer reached mutated quadrature helper")

    monkeypatch.setattr(
        s3_module, "_validate_s3_quadrature_values", forbidden
    )
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
