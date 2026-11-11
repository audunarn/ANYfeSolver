from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

import anyfesolver.nonlinear_static as nonlinear_static_module
from anyfesolver.activity import ElementActivity
from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.elements import Element
from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear_restart import (
    NonlinearCheckpointError,
    canonical_checkpoint_json_bytes,
    create_nonlinear_checkpoint,
    load_nonlinear_checkpoint,
    validate_nonlinear_checkpoint,
)
from anyfesolver.nonlinear_static import DisplacementControl, solve_static_nonlinear


class _SofteningSpring(Element):
    """One physical DOF with an exact, bounded nonlinear continuation path."""

    def __init__(self, element_id: int, node_id: int, *, k: float = 2.0, c: float = 0.0):
        super().__init__(element_id, [node_id], "default")
        self.k = float(k)
        self.c = float(c)

    @property
    def num_nodes(self) -> int:
        return 1

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh):
        return np.asarray([mesh.get_node(self.node_ids[0]).coords()], dtype=float)

    def compute_stiffness_matrix(self, mesh, material):
        matrix = np.eye(6, dtype=float)
        matrix[0, 0] = self.k
        return matrix

    def compute_nonlinear_response(
        self,
        mesh,
        material,
        u_elem,
        state=None,
        num_layers: int = 5,
        tangent: bool = True,
    ):
        displacement = np.asarray(u_elem, dtype=float)
        value = float(displacement[0])
        force = displacement.copy()
        force[0] = self.k * value - self.c * value**3
        stiffness = None
        if tangent:
            stiffness = np.eye(6, dtype=float)
            stiffness[0, 0] = self.k - 3.0 * self.c * value**2
        return force, stiffness, {"spring_displacement": value}


def _model(*, activity: float | None = None) -> tuple[FEModel, LoadCase]:
    model = FEModel("nonlinear-restart-spring")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_element(1, _SofteningSpring(1, 1))
    model.add_boundary_condition(
        BoundaryCondition(
            "one-dof",
            [1],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    if activity is not None:
        manager = ElementActivity([1])
        manager.set_activity([1], [float(activity)], step=0, reason="restart-fixture")
        model.set_element_activity(manager)
    load = LoadCase("unit-reference")
    load.add_nodal_load(1, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load


def _static_kwargs() -> dict[str, object]:
    return {
        "max_iterations": 12,
        "tolerance": 1.0e-12,
        "convergence_settings": "legacy",
    }


def test_generic_staged_hard_delete_is_not_fabricated_as_fracture_history() -> None:
    model, _load = _model()
    activity = ElementActivity([1])
    activity.hard_delete([1], step=1, reason="staged")
    model.set_element_activity(activity)

    checkpoint = create_nonlinear_checkpoint(
        analysis_kind="static",
        model=model,
        analysis_contract={"num_layers": 5},
        displacements=np.zeros(model.mesh.dof_manager.total_dofs),
        element_states={},
        path_state={},
        deleted_element_ids=(),
    )

    assert checkpoint["deleted_element_ids"] == []
    assert checkpoint["activity_state"]["hard_deleted"] == [True]

    incomplete = copy.deepcopy(checkpoint)
    incomplete["activity_state"]["history"] = []
    body = {
        key: value
        for key, value in incomplete.items()
        if key != "checkpoint_sha256"
    }
    incomplete["checkpoint_sha256"] = hashlib.sha256(
        canonical_checkpoint_json_bytes(body)
    ).hexdigest().upper()
    with pytest.raises(NonlinearCheckpointError, match="activity state is invalid"):
        validate_nonlinear_checkpoint(
            incomplete,
            analysis_kind="static",
            model=model,
            analysis_contract={"num_layers": 5},
            num_layers=5,
        )


def test_recorded_fracture_deletion_must_be_hard_deleted_in_activity() -> None:
    model, _load = _model()
    model.set_element_activity(ElementActivity([1]))
    record = {
        "element_id": 1,
        "element_type": "shell",
        "step_index": 1,
        "load_factor": 0.5,
        "trigger_name": "max_equivalent_plastic_strain",
        "trigger_value": 0.01,
        "threshold": 0.001,
        "location": "alpha[0]",
        "measure": 1.0,
    }

    with pytest.raises(NonlinearCheckpointError, match="not hard-deleted"):
        create_nonlinear_checkpoint(
            analysis_kind="static",
            model=model,
            analysis_contract={
                "num_layers": 5,
                "fracture_config": {
                    "threshold": 0.001,
                    "residual_stiffness_fraction": 0.1,
                },
            },
            displacements=np.zeros(model.mesh.dof_manager.total_dofs),
            element_states={1: {"alpha": [0.01]}},
            path_state={"deletion_records": [record]},
            deleted_element_ids=(1,),
        )


def test_force_static_checkpoint_round_trip_and_exact_split_continuation() -> None:
    uninterrupted_model, uninterrupted_load = _model(activity=0.5)
    uninterrupted = solve_static_nonlinear(
        uninterrupted_model,
        uninterrupted_load,
        max_load_factor=0.30,
        num_steps=4,
        emit_restart_checkpoint=True,
        **_static_kwargs(),
    )

    first_model, first_load = _model(activity=0.5)
    first = solve_static_nonlinear(
        first_model,
        first_load,
        max_load_factor=0.15,
        num_steps=2,
        emit_restart_checkpoint=True,
        **_static_kwargs(),
    )
    raw = first.restart_checkpoint_bytes()
    assert canonical_checkpoint_json_bytes(load_nonlinear_checkpoint(raw)) == raw

    resumed_model, resumed_load = _model(activity=1.0)
    resumed = solve_static_nonlinear(
        resumed_model,
        resumed_load,
        max_load_factor=0.30,
        num_steps=2,
        restart_checkpoint=raw,
        **_static_kwargs(),
    )

    assert uninterrupted.status == resumed.status == "completed"
    np.testing.assert_array_equal(resumed.displacements, uninterrupted.displacements)
    assert [step.to_dict() for step in resumed.steps] == [
        step.to_dict() for step in uninterrupted.steps
    ]
    assert resumed.restart_checkpoint_bytes() == uninterrupted.restart_checkpoint_bytes()
    np.testing.assert_array_equal(
        resumed_model.mesh.element_activity.activity_for([1]),
        np.array([0.5]),
    )


def test_armijo_checkpoint_binds_the_resolved_characteristic_length() -> None:
    settings = {
        "profile": "legacy",
        "line_search": "armijo",
        "characteristic_length": 1.0,
    }
    first_model, first_load = _model()
    first = solve_static_nonlinear(
        first_model,
        first_load,
        max_load_factor=0.15,
        num_steps=2,
        max_iterations=12,
        tolerance=1.0e-12,
        convergence_settings=settings,
        emit_restart_checkpoint=True,
    )
    assert first.status == "completed"

    resumed_model, resumed_load = _model()
    resumed = solve_static_nonlinear(
        resumed_model,
        resumed_load,
        max_load_factor=0.30,
        num_steps=2,
        max_iterations=12,
        tolerance=1.0e-12,
        convergence_settings=settings,
        restart_checkpoint=first.restart_checkpoint_bytes(),
    )
    assert resumed.status == "completed"
    assert resumed.info["convergence_settings"]["characteristic_length"] == 1.0

    changed_model, changed_load = _model()
    with pytest.raises(
        NonlinearCheckpointError,
        match="analysis contract is incompatible",
    ):
        solve_static_nonlinear(
            changed_model,
            changed_load,
            max_load_factor=0.30,
            num_steps=2,
            max_iterations=12,
            tolerance=1.0e-12,
            convergence_settings={
                **settings,
                "characteristic_length": 2.0,
            },
            restart_checkpoint=first.restart_checkpoint_bytes(),
        )


def test_legacy_checkpoint_contract_omits_unused_armijo_scaling() -> None:
    model, load = _model()
    contract = nonlinear_static_module._static_restart_analysis_contract(
        model=model,
        load_case=load,
        constant_load_case=None,
        load_program=None,
        control_name="force",
        displacement_control=None,
        num_layers=5,
        max_iterations=12,
        tolerance=1.0e-12,
        effective_min_step_fraction=1.0 / 1024.0,
        settings=nonlinear_static_module.NonlinearConvergenceSettings.for_profile(
            "legacy"
        ),
        fracture_config=None,
        resource_config=None,
        kinematics="von_karman",
        resolved_corotational_tangent="not_applicable",
    )

    convergence_contract = contract["convergence_settings"]
    assert "characteristic_length" not in convergence_contract


def test_legacy_checkpoint_contract_preserves_valid_setting_casing() -> None:
    model, load = _model()
    settings = nonlinear_static_module.NonlinearConvergenceSettings(
        profile="LEGACY",
        line_search="RESCUE",
    )
    contract = nonlinear_static_module._static_restart_analysis_contract(
        model=model,
        load_case=load,
        constant_load_case=None,
        load_program=None,
        control_name="force",
        displacement_control=None,
        num_layers=5,
        max_iterations=12,
        tolerance=1.0e-12,
        effective_min_step_fraction=1.0 / 1024.0,
        settings=settings,
        fracture_config=None,
        resource_config=None,
        kinematics="von_karman",
        resolved_corotational_tangent="not_applicable",
    )

    convergence_contract = contract["convergence_settings"]
    assert convergence_contract["profile"] == "LEGACY"
    assert convergence_contract["line_search"] == "RESCUE"
    assert "characteristic_length" not in convergence_contract

    first = solve_static_nonlinear(
        model,
        load,
        max_load_factor=0.15,
        num_steps=2,
        max_iterations=12,
        tolerance=1.0e-12,
        convergence_settings=settings,
        emit_restart_checkpoint=True,
    )
    resumed_model, resumed_load = _model()
    resumed = solve_static_nonlinear(
        resumed_model,
        resumed_load,
        max_load_factor=0.30,
        num_steps=2,
        max_iterations=12,
        tolerance=1.0e-12,
        convergence_settings=settings,
        restart_checkpoint=first.restart_checkpoint_bytes(),
    )

    assert first.status == resumed.status == "completed"


@pytest.mark.parametrize("line_search", ["AUTO", "RESCUE"])
def test_corotational_uppercase_checkpoint_contract_preserves_v1_fingerprint(
    line_search: str,
) -> None:
    model, load = _model()
    original = nonlinear_static_module.NonlinearConvergenceSettings(
        profile="LEGACY",
        line_search=line_search,
    )
    legacy_contract = nonlinear_static_module._static_restart_analysis_contract(
        model=model,
        load_case=load,
        constant_load_case=None,
        load_program=None,
        control_name="force",
        displacement_control=None,
        num_layers=5,
        max_iterations=12,
        tolerance=1.0e-12,
        effective_min_step_fraction=1.0 / 1024.0,
        settings=original,
        fracture_config=None,
        resource_config=None,
        kinematics="corotational",
        resolved_corotational_tangent="consistent",
    )
    resolved, runtime_override = (
        nonlinear_static_module._resolved_corotational_line_search_settings(
            original,
            "corotational",
        )
    )
    current_contract = nonlinear_static_module._static_restart_analysis_contract(
        model=model,
        load_case=load,
        constant_load_case=None,
        load_program=None,
        control_name="force",
        displacement_control=None,
        num_layers=5,
        max_iterations=12,
        tolerance=1.0e-12,
        effective_min_step_fraction=1.0 / 1024.0,
        settings=resolved,
        fracture_config=None,
        resource_config=None,
        kinematics="corotational",
        resolved_corotational_tangent="consistent",
    )

    assert runtime_override == "never"
    assert resolved is original
    assert current_contract == legacy_contract
    assert current_contract["convergence_settings"]["line_search"] == line_search


def test_displacement_static_checkpoint_exact_split_continuation() -> None:
    full_control = DisplacementControl(node_id=1, dof="ux", target_displacement=0.20)
    half_control = DisplacementControl(node_id=1, dof="ux", target_displacement=0.10)

    uninterrupted_model, uninterrupted_load = _model()
    uninterrupted = solve_static_nonlinear(
        uninterrupted_model,
        uninterrupted_load,
        control="displacement",
        displacement_control=full_control,
        num_steps=4,
        emit_restart_checkpoint=True,
        **_static_kwargs(),
    )

    first_model, first_load = _model()
    first = solve_static_nonlinear(
        first_model,
        first_load,
        control="displacement",
        displacement_control=half_control,
        num_steps=2,
        emit_restart_checkpoint=True,
        **_static_kwargs(),
    )
    resumed_model, resumed_load = _model()
    resumed = solve_static_nonlinear(
        resumed_model,
        resumed_load,
        control="displacement",
        displacement_control=full_control,
        num_steps=2,
        restart_checkpoint=first.restart_checkpoint_bytes(),
        **_static_kwargs(),
    )

    assert uninterrupted.status == resumed.status == "completed"
    np.testing.assert_array_equal(resumed.displacements, uninterrupted.displacements)
    assert [step.to_dict() for step in resumed.steps] == [
        step.to_dict() for step in uninterrupted.steps
    ]
    assert resumed.restart_checkpoint_bytes() == uninterrupted.restart_checkpoint_bytes()


def test_accelerated_displacement_dispatch_delegates_checkpoint_semantics(monkeypatch) -> None:
    from anyfesolver import nonlinear_performance

    nonlinear_performance.install_nonlinear_performance_optimizations()
    original = nonlinear_performance._ORIGINAL_DISPLACEMENT_SOLVER
    assert original is not None
    calls: list[bool] = []

    def observed_reference(**kwargs):
        calls.append(kwargs["restart_analysis_contract"] is not None)
        return original(**kwargs)

    monkeypatch.setattr(
        nonlinear_performance,
        "_ORIGINAL_DISPLACEMENT_SOLVER",
        observed_reference,
    )
    model, load = _model()
    result = solve_static_nonlinear(
        model,
        load,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=1,
            dof="ux",
            target_displacement=0.05,
        ),
        num_steps=1,
        emit_restart_checkpoint=True,
        **_static_kwargs(),
    )

    assert calls == [True]
    assert result.restart_checkpoint is not None
    assert result.info.get("displacement_control_linearization") != "block_elimination"


def test_arc_length_checkpoint_exact_nonzero_split_continuation() -> None:
    common = {
        "initial_load_increment": 0.05,
        "minimum_load_increment": 0.05,
        "maximum_load_increment": 0.05,
        "growth_factor": 1.0,
        "stop_after_peak_steps": 20,
    }
    full_control = ArcLengthControl(max_steps=4, **common)
    split_control = ArcLengthControl(max_steps=2, **common)

    uninterrupted_model, uninterrupted_load = _model()
    uninterrupted = solve_static_arc_length(
        uninterrupted_model,
        uninterrupted_load,
        control=full_control,
        max_iterations=10,
        tolerance=1.0e-12,
        arc_tolerance=1.0e-12,
        emit_restart_checkpoint=True,
    )

    first_model, first_load = _model()
    first = solve_static_arc_length(
        first_model,
        first_load,
        control=split_control,
        max_iterations=10,
        tolerance=1.0e-12,
        arc_tolerance=1.0e-12,
        emit_restart_checkpoint=True,
    )
    assert first.load_factor > 0.0
    resumed_model, resumed_load = _model()
    resumed = solve_static_arc_length(
        resumed_model,
        resumed_load,
        control=split_control,
        max_iterations=10,
        tolerance=1.0e-12,
        arc_tolerance=1.0e-12,
        restart_checkpoint=first.restart_checkpoint_bytes(),
    )

    assert uninterrupted.status == resumed.status == "maximum_steps_reached"
    np.testing.assert_array_equal(resumed.displacements, uninterrupted.displacements)
    assert [step.to_dict() for step in resumed.steps] == [
        step.to_dict() for step in uninterrupted.steps
    ]
    assert resumed.restart_checkpoint_bytes() == uninterrupted.restart_checkpoint_bytes()


def test_checkpoint_rejects_duplicate_nonfinite_hash_and_model_mutation_before_assembly(
    monkeypatch,
) -> None:
    model, load = _model()
    result = solve_static_nonlinear(
        model,
        load,
        max_load_factor=0.10,
        num_steps=1,
        emit_restart_checkpoint=True,
        **_static_kwargs(),
    )
    raw = result.restart_checkpoint_bytes()
    duplicate = raw.replace(b'"schema":', b'"schema":"duplicate","schema":', 1)
    with pytest.raises(NonlinearCheckpointError, match="duplicate"):
        load_nonlinear_checkpoint(duplicate)
    with pytest.raises(NonlinearCheckpointError, match="nonfinite"):
        load_nonlinear_checkpoint(b'{"x":NaN}\n')
    mutated = result.to_restart_checkpoint()
    mutated["path_state"]["load_factor"] += 0.01
    with pytest.raises(NonlinearCheckpointError, match="SHA-256"):
        load_nonlinear_checkpoint(mutated)

    changed_model, changed_load = _model()
    changed_model.mesh.set_node_coordinates(1, 0.01, 0.0, 0.0)
    monkeypatch.setattr(
        nonlinear_static_module,
        "assemble_stiffness_matrix",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("assembly must not run")
        ),
    )
    with pytest.raises(NonlinearCheckpointError, match="model fingerprint"):
        solve_static_nonlinear(
            changed_model,
            changed_load,
            max_load_factor=0.20,
            num_steps=1,
            restart_checkpoint=raw,
            **_static_kwargs(),
        )

    load_model, _unchanged_load = _model()
    changed_load = LoadCase("unit-reference")
    changed_load.add_nodal_load(1, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(NonlinearCheckpointError, match="analysis contract"):
        solve_static_nonlinear(
            load_model,
            changed_load,
            max_load_factor=0.20,
            num_steps=1,
            restart_checkpoint=raw,
            **_static_kwargs(),
        )

    duplicate_states = result.to_restart_checkpoint()
    duplicate_states["element_states"].append(
        copy.deepcopy(duplicate_states["element_states"][0])
    )
    body = {
        key: value
        for key, value in duplicate_states.items()
        if key != "checkpoint_sha256"
    }
    duplicate_states["checkpoint_sha256"] = hashlib.sha256(
        canonical_checkpoint_json_bytes(body)
    ).hexdigest().upper()
    duplicate_model, duplicate_load = _model()
    with pytest.raises(NonlinearCheckpointError, match="duplicate element state ID"):
        solve_static_nonlinear(
            duplicate_model,
            duplicate_load,
            max_load_factor=0.20,
            num_steps=1,
            restart_checkpoint=duplicate_states,
            **_static_kwargs(),
        )


def test_failed_static_trial_checkpoint_contains_only_last_committed_state(
    monkeypatch,
) -> None:
    model, load = _model()
    result = solve_static_nonlinear(
        model,
        load,
        max_load_factor=0.20,
        num_steps=2,
        max_iterations=1,
        tolerance=1.0e-30,
        min_step_fraction=0.75,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    checkpoint = result.to_restart_checkpoint()
    assert checkpoint["path_state"]["load_factor"] == result.load_factor
    np.testing.assert_array_equal(checkpoint["displacements"], result.displacements)
    assert checkpoint["path_state"]["step_index"] == len(result.steps)
    assert checkpoint["element_states"] == [
        {"element_id": int(element_id), "state": state}
        for element_id, state in sorted(result.element_states.items())
    ]
    resumed_model, resumed_load = _model()
    monkeypatch.setattr(
        nonlinear_static_module,
        "assemble_stiffness_matrix",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("assembly must not run")
        ),
    )
    with pytest.raises(ValueError, match="continuable completed static target"):
        solve_static_nonlinear(
            resumed_model,
            resumed_load,
            max_load_factor=0.20,
            num_steps=1,
            max_iterations=1,
            tolerance=1.0e-30,
            min_step_fraction=0.75,
            convergence_settings="legacy",
            restart_checkpoint=result.restart_checkpoint_bytes(),
        )
