from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.fracture import (
    FractureConfig,
    deleted_pressure_load_resultant,
    filtered_load_case_for_deleted_elements,
)
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.matrix_assembly import assemble_load_vector
from anyfesolver.nonlinear_static import _assemble_nonlinear_system, solve_static_nonlinear
from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes


E = 210.0e9
NU = 0.3

CURVE = DNVC208MaterialCurve(
    sigma_prop=320.0e6,
    sigma_yield=357.0e6,
    sigma_yield_2=363.3e6,
    eps_p_y1=0.004,
    eps_p_y2=0.015,
    K=740.0e6,
    n=0.166,
)


def _one_shell_model(curve=CURVE, *, formulation: str | None = None) -> FEModel:
    model = FEModel("fracture_panel")
    model.add_material("steel", E, NU, hardening_curve=curve)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.0, 0.2, 0.0)
    model.add_node(4, 0.0, 0.2, 0.0)
    kwargs = {} if formulation is None else {"formulation": formulation}
    model.add_element(
        1,
        create_shell_element(
            1,
            [1, 2, 3, 4],
            "steel",
            thickness=0.01,
            **kwargs,
        ),
    )
    model.add_boundary_condition(BoundaryCondition("left_x", [1, 4], {"ux": 0.0}))
    model.add_boundary_condition(BoundaryCondition("plane", [1, 2, 3, 4], {"uz": 0.0, "rx": 0.0, "ry": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pin_y", [1], {"uy": 0.0}))
    return model


def _tension_load(model: FEModel, stress: float = 340.0e6) -> LoadCase:
    load = LoadCase("pull")
    total = stress * 0.2 * 0.01
    load.add_nodal_load(2, [0.5 * total, 0, 0, 0, 0, 0])
    load.add_nodal_load(3, [0.5 * total, 0, 0, 0, 0, 0])
    return load


def test_fracture_config_validates_inputs() -> None:
    with pytest.raises(ValueError):
        FractureConfig(threshold=0.0)
    with pytest.raises(ValueError):
        FractureConfig(threshold=1.0e-3, residual_stiffness_fraction=-1.0)
    with pytest.raises(ValueError):
        FractureConfig(threshold=1.0e-3, max_deleted_fraction=1.5)
    with pytest.raises(ValueError):
        FractureConfig(threshold=1.0e-3, element_scope=("solid",))


def test_deleted_element_assembly_uses_residual_stiffness_without_state_update() -> None:
    model = _one_shell_model()
    displacement = np.linspace(0.0, 1.0e-4, model.mesh.dof_manager.total_dofs)
    active_force, active_tangent, active_states = _assemble_nonlinear_system(model, displacement, {}, 3, tangent=True)
    deleted_force, deleted_tangent, deleted_states = _assemble_nonlinear_system(
        model,
        displacement,
        active_states,
        3,
        tangent=True,
        deleted_element_ids=(1,),
        residual_stiffness_fraction=0.2,
    )
    assert np.linalg.norm(deleted_force) == pytest.approx(0.2 * np.linalg.norm(active_force), rel=1.0e-12)
    assert np.linalg.norm(deleted_tangent.toarray()) == pytest.approx(
        0.2 * np.linalg.norm(active_tangent.toarray()),
        rel=1.0e-12,
    )
    assert deleted_states[1] is active_states[1]


def test_deleted_pressure_loads_are_filtered_from_subsequent_load_vectors() -> None:
    model = _one_shell_model(curve=None)
    load = LoadCase("pressure")
    load.add_pressure_load(1, 11.0)
    full, _ = assemble_load_vector(model, load)
    filtered = filtered_load_case_for_deleted_elements(load, (1,))
    reduced, _ = assemble_load_vector(model, filtered)
    removed = deleted_pressure_load_resultant(model, load, (1,))

    assert np.linalg.norm(full) > 0.0
    assert np.linalg.norm(reduced) == pytest.approx(0.0)
    assert removed[2] == pytest.approx(11.0 * 1.0 * 0.2)


def test_plastic_strain_threshold_deletes_after_converged_increment() -> None:
    model = _one_shell_model()
    load = _tension_load(model)
    result = solve_static_nonlinear(
        model,
        load,
        num_steps=4,
        num_layers=3,
        fracture_config=FractureConfig(threshold=1.0e-5, max_deleted_fraction=1.0),
    )

    summary = result.info["fracture_summary"]
    assert summary["deleted_count"] == 1
    assert summary["deleted_element_ids"] == [1]
    assert summary["records"][0]["trigger_name"] == "max_equivalent_plastic_strain"
    assert summary["records"][0]["load_factor"] > 0.0
    assert any(step.deleted_element_count == 1 for step in result.steps)


def test_high_threshold_produces_no_deletions() -> None:
    model = _one_shell_model()
    result = solve_static_nonlinear(
        model,
        _tension_load(model),
        num_steps=4,
        num_layers=3,
        fracture_config=FractureConfig(threshold=1.0, max_deleted_fraction=1.0),
    )
    assert result.info["fracture_summary"]["deleted_count"] == 0
    assert all(step.deleted_element_count == 0 for step in result.steps)


def test_max_deleted_fraction_stops_after_converged_deletion() -> None:
    model = _one_shell_model()
    state = model.mesh.get_element(1).init_nonlinear_state(3)
    state["alpha"][:] = 0.01
    result = solve_static_nonlinear(
        model,
        _tension_load(model, stress=100.0e6),
        num_steps=2,
        num_layers=3,
        initial_element_states={1: state},
        fracture_config=FractureConfig(threshold=0.001, max_deleted_fraction=0.5),
    )
    assert result.status == "stopped_at_limit"
    assert result.failure_reason == "max_deleted_fraction_reached"
    assert result.info["fracture_summary"]["deleted_count"] == 1


def test_fracture_displacement_control_is_explicitly_unsupported() -> None:
    from anyfesolver.nonlinear_static import DisplacementControl

    model = _one_shell_model()
    with pytest.raises(ValueError, match="force control"):
        solve_static_nonlinear(
            model,
            _tension_load(model),
            control="displacement",
            displacement_control=DisplacementControl(node_id=2, dof="ux", target_displacement=1.0e-3),
            fracture_config=FractureConfig(threshold=1.0e-4),
        )


def test_fracture_deletion_history_survives_exact_checkpoint_continuation() -> None:
    config = FractureConfig(
        threshold=1.0e-5,
        residual_stiffness_fraction=0.1,
        max_deleted_fraction=1.0,
    )
    uninterrupted_model = _one_shell_model()
    uninterrupted = solve_static_nonlinear(
        uninterrupted_model,
        _tension_load(uninterrupted_model),
        max_load_factor=1.25,
        num_steps=5,
        num_layers=3,
        fracture_config=config,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    first_model = _one_shell_model()
    first = solve_static_nonlinear(
        first_model,
        _tension_load(first_model),
        max_load_factor=1.0,
        num_steps=4,
        num_layers=3,
        fracture_config=config,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    assert first.restart_checkpoint["deleted_element_ids"] == [1]

    resumed_model = _one_shell_model()
    resumed = solve_static_nonlinear(
        resumed_model,
        _tension_load(resumed_model),
        max_load_factor=1.25,
        num_steps=1,
        num_layers=3,
        fracture_config=config,
        convergence_settings="legacy",
        restart_checkpoint=first.restart_checkpoint_bytes(),
    )

    assert uninterrupted.status == resumed.status
    np.testing.assert_array_equal(uninterrupted.displacements, resumed.displacements)
    assert uninterrupted.info["fracture_summary"] == resumed.info["fracture_summary"]
    assert uninterrupted.restart_checkpoint_bytes() == resumed.restart_checkpoint_bytes()

    state = resumed.element_states[1]
    marker = state["qualified_q4_activity_disposition"]
    assert marker["status"] == "DELETED_FROZEN_NONCURRENT"
    assert marker["operator_semantics"] == (
        "CONSTITUTIVE_HISTORY_FROZEN;"
        "FORCE_AND_TANGENT_REEVALUATED_AT_CURRENT_U_THEN_SCALED"
    )
    deletion_u = np.asarray(marker["accepted_local_u"], dtype=np.float64)
    np.testing.assert_array_equal(
        state["qualified_q4_committed_binding"]["committed_total_u"],
        deletion_u,
    )
    element = resumed_model.mesh.elements[1]
    dofs = np.asarray(element.get_dof_mapping(resumed_model.mesh), dtype=np.intp)
    assert not np.array_equal(deletion_u, resumed.displacements[dofs])
    record = resumed.info["fracture_summary"]["records"][0]
    element.validate_noncurrent_deleted_state(
        resumed_model.mesh,
        resumed_model.get_material(element.material_name),
        state,
        3,
        expected_deletion_step_index=record["step_index"],
        expected_deletion_load_factor=record["load_factor"],
        expected_residual_stiffness_fraction=(
            config.residual_stiffness_fraction
        ),
        expected_trigger_name=record["trigger_name"],
    )
    with pytest.raises(ValueError, match="noncurrent"):
        element.compute_committed_current_tangent_components(
            resumed_model.mesh,
            resumed_model.get_material(element.material_name),
            deletion_u,
            state,
            3,
        )

    for name, replacement in (
        ("status", "ACTIVE"),
        ("policy_id", "WRONG"),
        ("residual_stiffness_fraction", 0.2),
        ("trigger_name", "wrong-trigger"),
    ):
        mutated = copy.deepcopy(state)
        mutated["qualified_q4_activity_disposition"][name] = replacement
        with pytest.raises(ValueError, match="disposition|residual|trigger"):
            element.validate_noncurrent_deleted_state(
                resumed_model.mesh,
                resumed_model.get_material(element.material_name),
                mutated,
                3,
                expected_deletion_step_index=record["step_index"],
                expected_deletion_load_factor=record["load_factor"],
                expected_residual_stiffness_fraction=(
                    config.residual_stiffness_fraction
                ),
                expected_trigger_name=record["trigger_name"],
            )

    mutated_u = copy.deepcopy(state)
    mutated_u["qualified_q4_activity_disposition"]["accepted_local_u"][0] += (
        1.0e-12
    )
    with pytest.raises(ValueError, match="disposition"):
        element.validate_noncurrent_deleted_state(
            resumed_model.mesh,
            resumed_model.get_material(element.material_name),
            mutated_u,
            3,
        )

    # A direct ordinary restart has no deletion-record authority and therefore
    # cannot silently reactivate a frozen marker.  Only the closed checkpoint
    # path above may restore it.
    rejected_model = _one_shell_model()
    with pytest.raises(ValueError, match="disagrees with checkpoint deletion state"):
        solve_static_nonlinear(
            rejected_model,
            _tension_load(rejected_model),
            max_load_factor=0.1,
            num_steps=1,
            num_layers=3,
            fracture_config=config,
            initial_element_states=copy.deepcopy(resumed.element_states),
            initial_displacements=resumed.displacements.copy(),
            equilibrate_initial_state=False,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("step_index", 999, "deletion step"),
        ("load_factor", 999.0, "deletion load factor"),
    ),
)
def test_legacy_fracture_checkpoint_binds_deletion_step_history_before_assembly(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    message: str,
) -> None:
    config = FractureConfig(
        threshold=0.001,
        residual_stiffness_fraction=0.1,
        max_deleted_fraction=1.0,
    )
    model = _one_shell_model(formulation="legacy")
    element = model.mesh.elements[1]
    state = element.init_nonlinear_state(3)
    state["alpha"][:] = 0.01
    result = solve_static_nonlinear(
        model,
        _tension_load(model, stress=100.0e6),
        num_steps=2,
        num_layers=3,
        initial_element_states={1: state},
        fracture_config=config,
        emit_restart_checkpoint=True,
    )
    assert result.restart_checkpoint is not None
    checkpoint = copy.deepcopy(result.restart_checkpoint)
    checkpoint["path_state"]["deletion_records"][0][field] = replacement
    body = {
        key: value
        for key, value in checkpoint.items()
        if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        canonical_checkpoint_json_bytes(body)
    ).hexdigest().upper()

    resumed_model = _one_shell_model(formulation="legacy")
    monkeypatch.setattr(
        resumed_model,
        "apply_boundary_conditions",
        lambda: (_ for _ in ()).throw(
            AssertionError("checkpoint mutation reached assembly")
        ),
    )
    with pytest.raises(ValueError, match=message):
        solve_static_nonlinear(
            resumed_model,
            _tension_load(resumed_model, stress=100.0e6),
            num_steps=1,
            num_layers=3,
            fracture_config=config,
            restart_checkpoint=checkpoint,
        )
