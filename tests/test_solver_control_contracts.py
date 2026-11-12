from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import (
    AnyStructureFEMResult,
    CancellationToken,
    ConstraintEquation,
    ProgressEvent,
    ImperfectionCalibrationResult,
    QuantityUnavailableError,
    ReactionFrame,
    SolveCancelled,
    SolveDisposition,
    SolveOutcome,
    audit_constraints,
    describe_result_quantities,
    registered_result_quantity_ids,
    resolve_result_quantity,
    solve_outcome,
    solve_nonlinear_load_stepping,
)
from anyfesolver.assembly import (
    build_constraint_transformation,
    build_reduced_rigid_body_modes,
    reconstruct_full_solution,
)
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.buckling import BucklingMode, BucklingResult
from anyfesolver.modal import ModalMode, ModalResult
from anyfesolver.nonlinear_static import NonlinearStaticResult, solve_static_nonlinear
from anyfesolver.recovery import StressRecoveryProvenance, StressRecoveryResult
from anyfesolver.results import FEResult


def _elastic_cantilever() -> tuple[FEModel, LoadCase]:
    model = FEModel("solver-control-cantilever")
    model.add_material("steel", 210.0e9, 0.3)
    section = {
        "area": 0.01,
        "Iy": 1.0e-6,
        "Iz": 1.0e-6,
        "J": 1.0e-6,
        "orientation": (0.0, 0.0, 1.0),
    }
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    load = LoadCase("tip")
    load.add_nodal_load(2, [0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    return model, load


def test_force_control_ramps_nonzero_prescribed_displacement_by_increment() -> None:
    model, _load = _elastic_cantilever()
    target = 4.0e-3
    model.add_boundary_condition(
        BoundaryCondition("prescribed-tip-x", [2], {"ux": target})
    )
    progress = []
    result = solve_static_nonlinear(
        model,
        load_case=None,
        num_steps=4,
        record_increment_snapshots=True,
        progress_callback=progress.append,
    )

    assert result.converged
    assert len(result.snapshots) >= 2
    ux = model.mesh.nodes[2].dofs[0]
    imposed = [snapshot.displacements[ux] for snapshot in result.snapshots]
    factors = [snapshot.load_factor for snapshot in result.snapshots]
    assert imposed == pytest.approx([factor * target for factor in factors])
    assert imposed[0] < imposed[-1]
    assert result.displacements[ux] == pytest.approx(target)
    assert result.info["prescribed_displacement_path"]["mode"] == (
        "proportional_to_load_factor"
    )
    assert result.steps[-1].support_reactions["fixed"][0] == pytest.approx(
        -result.steps[-1].support_reactions["prescribed-tip-x"][0]
    )
    assert abs(result.steps[-1].support_reactions["fixed"][0]) > 0.0
    assert progress[-1]["support_reactions"]["fixed"][0] == pytest.approx(
        result.steps[-1].support_reactions["fixed"][0]
    )
    reaction_history = resolve_result_quantity(result, "reaction_history")
    assert reaction_history.descriptor.data_path == "steps[].support_reactions"
    assert reaction_history.descriptor.location == "support"
    assert reaction_history.descriptor.metadata["nodal_reactions_available"] is False
    assert len(reaction_history.data) == len(result.steps)
    assert reaction_history.data[-1].support_resultants == (
        result.steps[-1].support_reactions
    )


def test_force_control_restart_holds_supplied_prescribed_state() -> None:
    model, _load = _elastic_cantilever()
    target = 4.0e-3
    model.add_boundary_condition(
        BoundaryCondition("prescribed-tip-x", [2], {"ux": target})
    )
    preload = solve_static_nonlinear(
        model,
        load_case=None,
        num_steps=2,
        record_increment_snapshots=True,
    )
    ux = model.mesh.nodes[2].dofs[0]
    assert preload.displacements[ux] == pytest.approx(target)

    restarted = solve_static_nonlinear(
        model,
        load_case=None,
        max_load_factor=0.1,
        num_steps=2,
        initial_element_states=preload.element_states,
        initial_displacements=preload.displacements,
        equilibrate_initial_state=False,
        record_increment_snapshots=True,
    )

    assert restarted.converged
    assert restarted.info["prescribed_displacement_path"] == {
        "mode": "restart_fixed_affine_state",
        "schema_mode": "FIXED_RESTART_AFFINE_STATE",
        "restart_schema_disposition": "DIRECT_EXACT_INITIAL_DISPLACEMENT",
        "initial_affine_scale": pytest.approx(1.0),
        "affine_scale_slope": 0.0,
        "target_max_abs": pytest.approx(target),
    }
    assert [snapshot.displacements[ux] for snapshot in restarted.snapshots] == (
        pytest.approx([target] * len(restarted.snapshots))
    )
    assert restarted.displacements[ux] == pytest.approx(target)
    assert restarted.info["constraint_postcheck"]["status"] == "passed"


def test_cancellation_token_is_one_way_and_reports_safe_point() -> None:
    token = CancellationToken()
    assert not token.is_cancelled
    assert token.cancel("operator request")
    assert not token.cancel("later reason")
    assert token.reason == "operator request"
    with pytest.raises(SolveCancelled, match="nonlinear_limit.start") as caught:
        solve_nonlinear_load_stepping(FEModel("cancelled"), cancellation_token=token)
    assert caught.value.reason == "operator request"
    assert caught.value.stage == "nonlinear_limit.start"


def test_progress_event_is_typed_and_legacy_mapping_compatible() -> None:
    event = ProgressEvent(
        "nonlinear_static_step",
        "nonlinear_static.force",
        completed=2,
        total=4,
        iteration=3,
        metadata={"load_factor": 0.5},
    )
    assert event.type == "nonlinear_static_step"
    assert event.fraction == pytest.approx(0.5)
    assert event["load_factor"] == pytest.approx(0.5)
    assert event.get("iteration") == 3
    assert dict(event)["stage"] == "nonlinear_static.force"


def test_general_constraint_equation_uses_common_affine_transformation() -> None:
    model = FEModel("general-equation")
    node = model.add_node(1, 0.0, 0.0, 0.0)
    ux, uy = node.dofs[:2]
    model.add_boundary_condition(BoundaryCondition("prescribed-y", [1], {"uy": 1.0}))
    equation = model.add_constraint_equation(
        terms=((ux, 1.0), (uy, 1.0)),
        rhs=3.0,
        source_id="local-x",
    )
    assert isinstance(equation, ConstraintEquation)

    report = audit_constraints(model)
    assert report.feasible
    assert report.origin_counts["equation"] == 1
    assert report.equations[-1].source_id == "equation:local-x"

    K = sparse.eye(6, format="csr")
    K_red, F_red, T, u0, _independent, info = build_constraint_transformation(
        K,
        np.zeros(6),
        model,
    )
    assert info["num_generalized_constraint_equations"] == 1
    q = np.zeros(K_red.shape[0], dtype=float)
    displacement = reconstruct_full_solution(T, q, u0)
    assert displacement[uy] == pytest.approx(1.0)
    assert displacement[ux] == pytest.approx(2.0)
    assert F_red.shape == (K_red.shape[0],)


def test_constraint_equation_retains_legacy_constructor_aliases() -> None:
    equation = ConstraintEquation(2, ((2, 1.0), (3, -0.5)), 4.0, "legacy", "mpc")
    assert equation.dependent_dof == 2
    assert equation.coefficients == equation.terms
    assert equation.value == equation.rhs == pytest.approx(4.0)
    assert equation.origin == equation.source_id == "legacy"


def test_independent_rotated_rows_are_sparse_row_reduced_without_cycle() -> None:
    model = FEModel("rotated-equations")
    node = model.add_node(1, 0.0, 0.0, 0.0)
    ux, uy = node.dofs[:2]
    c = float(np.sqrt(0.5))
    model.add_constraint_equation(
        terms=((ux, c), (uy, c)),
        rhs=1.0,
        source_id="local-x",
    )
    model.add_constraint_equation(
        terms=((uy, c), (ux, -c)),
        rhs=2.0,
        source_id="local-y",
    )

    report = audit_constraints(model)
    assert report.feasible
    assert report.max_dependency_depth == 2
    K_red, _F_red, T, u0, independent, _info = build_constraint_transformation(
        sparse.eye(6, format="csr"),
        np.zeros(6),
        model,
    )
    displacement = reconstruct_full_solution(T, np.zeros(K_red.shape[0]), u0)
    expected = np.linalg.solve(
        np.array([[c, c], [-c, c]], dtype=float),
        np.array([1.0, 2.0], dtype=float),
    )
    assert displacement[[ux, uy]] == pytest.approx(expected)
    rigid_modes, rigid_info = build_reduced_rigid_body_modes(
        model,
        independent,
        6,
        transformation=T,
    )
    assert rigid_modes.shape[1] == 4
    assert rigid_info["constraint_compatibility_method"] == "affine_transformation_intersection"


def test_nonlinear_increment_snapshots_are_opt_in_and_committed() -> None:
    model, load = _elastic_cantilever()

    progress = []
    status = []
    result = solve_static_nonlinear(
        model,
        load,
        num_steps=2,
        record_increment_snapshots=True,
        progress_callback=progress.append,
        status_callback=status.append,
    )

    assert result.status == "completed"
    assert len(result.snapshots) == len(result.steps) == 2
    assert np.array_equal(result.snapshots[-1].displacements, result.displacements)
    assert not result.snapshots[-1].displacements.flags.writeable
    assert result.snapshots[-1].element_states is not result.element_states
    assert all(isinstance(event, ProgressEvent) for event in progress)
    assert all(event["nominal_increment_count"] == 2 for event in progress)
    assert all(event["load_increment"] > 0.0 for event in progress)
    assert status
    assert all("/2" not in message for message in status)
    assert "Increment trial 1" in status[0]
    assert "load factor" in status[0]
    assert "Newton iteration" in status[0]
    assert "residual" in status[0]
    quantities = describe_result_quantities(result)
    assert {quantity.quantity_id for quantity in quantities} >= {"displacement", "load_factor"}


def test_progress_observer_can_request_cooperative_cancellation() -> None:
    model, load = _elastic_cantilever()
    token = CancellationToken()

    def cancel_after_first_step(event: ProgressEvent) -> None:
        assert event["step_index"] == 1
        token.cancel("stop after preview")

    with pytest.raises(SolveCancelled, match="stop after preview"):
        solve_static_nonlinear(
            model,
            load,
            num_steps=4,
            cancellation_token=token,
            progress_callback=cancel_after_first_step,
        )


def test_result_quantity_resolver_is_canonical_and_fail_closed() -> None:
    snapshots = (
        SimpleNamespace(
            step_index=1,
            load_factor=0.25,
            control_value=None,
            element_states={5: {"alpha": [0.0, 0.01]}},
        ),
        SimpleNamespace(
            step_index=2,
            load_factor=0.5,
            control_value=None,
            element_states={5: {"layer_strain": [0.02]}},
        ),
        SimpleNamespace(
            step_index=3,
            load_factor=0.75,
            control_value=None,
            element_states={5: {"alpha": [0.02, 0.03]}},
        ),
    )
    result = SimpleNamespace(
        displacements=np.arange(12, dtype=float),
        element_states={5: {"alpha": np.asarray([0.02, 0.03])}},
        snapshots=snapshots,
        times=np.asarray([0.0, 0.1]),
        reaction_history=(
            ReactionFrame(0, 0.0, "time", {7: np.ones(6)}, {"fixed": np.ones(6)}),
            ReactionFrame(1, 0.1, "time", {7: 2.0 * np.ones(6)}, {"fixed": 2.0 * np.ones(6)}),
        ),
        diagnostics={
            "strain_energy_measure": "internal_work_proxy",
            "kinetic_energy": [3.0, 2.0],
            "strain_energy": [0.0, 1.0],
            "sphere_kinetic_energy": [4.0],
        },
    )

    assert registered_result_quantity_ids()[0] == "displacement"
    assert resolve_result_quantity(result, "displacement").data is result.displacements
    assert resolve_result_quantity(
        result, "equivalent_plastic_strain"
    ).data == {5: pytest.approx(0.03)}
    history = resolve_result_quantity(
        result, "equivalent_plastic_strain_history"
    )
    assert history.data == ({5: pytest.approx(0.01)}, {5: pytest.approx(0.03)})
    assert history.descriptor.metadata["frame_indices"] == [1, 3]
    assert history.descriptor.data_path == "snapshots[].element_states"
    assert history.descriptor.metadata["source_keys_by_frame"] == [
        {"5": "alpha"},
        {"5": "alpha"},
    ]
    assert resolve_result_quantity(result, "reaction_history").data is result.reaction_history
    assert resolve_result_quantity(result, "internal_work").data == [0.0, 1.0]
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(result, "strain_energy")
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(result, "impactor_kinetic_energy")


def test_real_result_diagnostic_shapes_resolve_without_inventing_fields() -> None:
    impact = SimpleNamespace(
        diagnostics={
            "method": "nonlinear_newmark_sphere_penalty_contact",
            "element_states": {11: {"alpha": np.array([0.0, 0.025])}},
            "kinetic_energy": [2.0, 1.5],
            "strain_energy": [0.0, 0.4],
            "sphere_kinetic_energy": [3.0, 2.2],
        },
        times=np.array([0.0, 0.1]),
    )

    peeq = resolve_result_quantity(impact, "equivalent_plastic_strain")
    assert peeq.data == {11: 0.025}
    assert peeq.descriptor.data_path == "diagnostics.element_states"
    internal_work = resolve_result_quantity(impact, "internal_work")
    kinetic = resolve_result_quantity(impact, "kinetic_energy")
    impactor_kinetic = resolve_result_quantity(impact, "impactor_kinetic_energy")
    assert internal_work.data == [0.0, 0.4]
    assert internal_work.descriptor.metadata["measure"] == (
        "committed_internal_work_proxy"
    )
    assert kinetic.data == [2.0, 1.5]
    assert kinetic.descriptor.metadata["measure"] == "structural_kinetic_energy"
    assert impactor_kinetic.data == [3.0, 2.2]
    assert impactor_kinetic.descriptor.metadata["measure"] == (
        "rigid_impactor_kinetic_energy"
    )
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(impact, "strain_energy")

    transient = SimpleNamespace(
        diagnostics={
            "method": "hht_alpha",
            "kinetic_energy": [1.0, 0.5],
            "strain_energy": [0.0, 0.5],
        },
        times=np.array([0.0, 0.1]),
    )
    assert resolve_result_quantity(transient, "strain_energy").data == [0.0, 0.5]
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(transient, "internal_work")

    recovery = StressRecoveryResult(
        element_stresses={
            4: {
                "membrane_strain_xx": np.array([1.0e-4]),
                "membrane_xx": np.array([21.0e6]),
                "local_xx_top": np.array([22.0e6]),
                "global_xx_top": np.array([20.0e6]),
                "von_mises": np.array([23.0e6]),
                "diagnostic_von_mises_ratio": np.array([0.1]),
                "equivalent_stress_measure": "von_mises",
            },
            5: {
                "generalized_stress_scope": "section_resultants_only",
                "recovery_scope": "section_resultants_only",
                "physical_stress_available": False,
                "membrane_strain": np.zeros((1, 3)),
                "membrane_resultants": np.ones((1, 3)),
            },
        },
        provenance=StressRecoveryProvenance(
            mode="material_history",
            state_source="committed",
        ),
        committed_element_states={4: {"alpha": np.array([0.0, 0.125])}},
    )
    fe_result = FEResult(
        model_name="recovered-peeq",
        displacements=np.zeros(6),
        element_stresses=recovery.element_stresses,
        stress_recovery=recovery,
    )
    recovered_peeq = resolve_result_quantity(
        fe_result, "equivalent_plastic_strain"
    )
    assert recovered_peeq.data == {4: pytest.approx(0.125)}
    assert recovered_peeq.descriptor.data_path == "committed_element_states"

    recovered_stress = resolve_result_quantity(fe_result, "stress")
    assert set(recovered_stress.data) == {4}
    assert set(recovered_stress.data[4]) == {
        "membrane_xx",
        "local_xx_top",
        "global_xx_top",
        "von_mises",
    }
    assert recovered_stress.descriptor.unit == "Pa"
    assert recovered_stress.descriptor.basis == "component_specific"
    assert recovered_stress.descriptor.metadata["component_basis"] == {
        "global_xx_top": "global",
        "local_xx_top": "element_local",
        "membrane_xx": "element_local",
        "von_mises": "invariant",
    }
    assert recovered_stress.descriptor.metadata[
        "excluded_nonphysical_element_ids"
    ] == (5,)
    assert "membrane_strain_xx" not in recovered_stress.data[4]
    assert "diagnostic_von_mises_ratio" not in recovered_stress.data[4]

    stress_history = resolve_result_quantity(
        SimpleNamespace(
            stress_history=(
                recovery.element_stresses,
                recovery.element_stresses,
            ),
            times=np.array([0.0, 1.0]),
        ),
        "stress_history",
    )
    assert len(stress_history.data) == 2
    assert all(set(frame) == {4} for frame in stress_history.data)
    assert stress_history.descriptor.basis == "component_specific"
    assert stress_history.descriptor.metadata[
        "excluded_nonphysical_element_ids_by_frame"
    ] == ((5,), (5,))


def test_result_quantity_resolver_rejects_malformed_history_and_energy() -> None:
    malformed_history = SimpleNamespace(
        reaction_history=(
            ReactionFrame(0, 0.0, "time", {1: np.zeros(5)}, {}),
        )
    )
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(malformed_history, "reaction_history")
    for malformed_history in (
        SimpleNamespace(
            reaction_history=(
                ReactionFrame(0, "0.0", "time", {1: np.zeros(6)}, {}),
            )
        ),
        SimpleNamespace(
            reaction_history=(
                ReactionFrame(0, 0.0, True, {1: np.zeros(6)}, {}),
            )
        ),
        SimpleNamespace(
            reaction_history=(
                ReactionFrame(0, 0.0, "time", {1: ["0"] * 6}, {}),
            )
        ),
    ):
        with pytest.raises(QuantityUnavailableError):
            resolve_result_quantity(malformed_history, "reaction_history")

    malformed_energy = SimpleNamespace(
        diagnostics={"method": "newmark", "kinetic_energy": ["not-a-number"]},
        times=np.array([0.0]),
    )
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(malformed_energy, "kinetic_energy")

    for malformed in (
        SimpleNamespace(
            diagnostics={"method": "newmark", "kinetic_energy": [1.0]},
            times=["bad"],
        ),
        SimpleNamespace(
            diagnostics={"method": "newmark", "kinetic_energy": [1.0]},
            times=[np.nan],
        ),
        SimpleNamespace(
            diagnostics={"method": "newmark", "kinetic_energy": [[1.0, 2.0]]},
            times=[0.0, 0.1],
        ),
    ):
        with pytest.raises(QuantityUnavailableError):
            resolve_result_quantity(malformed, "kinetic_energy")

    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(SimpleNamespace(load_impulse=["bad"]), "load_impulse")
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(SimpleNamespace(load_impulse=[1.0, 2.0]), "load_impulse")
    malformed_displacement = SimpleNamespace(displacements=["bad"])
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(malformed_displacement, "displacement")
    assert not describe_result_quantities(malformed_displacement)

    for malformed_nodal, quantity_id in (
        (SimpleNamespace(displacements=np.zeros(5)), "displacement"),
        (SimpleNamespace(velocities=np.zeros((2, 5))), "velocity"),
        (
            SimpleNamespace(velocities=np.zeros(6), times=np.array([0.0])),
            "velocity",
        ),
        (
            SimpleNamespace(accelerations=np.zeros(6), times=np.array([0.0])),
            "acceleration",
        ),
        (
            SimpleNamespace(displacement_envelope=np.zeros((1, 6))),
            "displacement_envelope",
        ),
        (
            SimpleNamespace(displacements=np.zeros((2, 6)), times=np.array([0.0])),
            "displacement",
        ),
        (
            SimpleNamespace(
                contact_force_history=np.zeros((2, 3)), times=np.array([0.0])
            ),
            "contact_force",
        ),
        (SimpleNamespace(times=np.array([0.0, 2.0, 1.0])), "time"),
        (SimpleNamespace(displacements=[[0.0], [0.0, 1.0]]), "displacement"),
    ):
        assert quantity_id not in {
            item.quantity_id for item in describe_result_quantities(malformed_nodal)
        }

    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(
            SimpleNamespace(element_states={3: {"alpha": [-0.2]}}),
            "equivalent_plastic_strain",
        )
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(
            SimpleNamespace(
                element_states={
                    1: {"alpha": [0.2]},
                    2: {"alpha": [-0.3]},
                }
            ),
            "equivalent_plastic_strain",
        )
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(
            SimpleNamespace(
                snapshots=(
                    SimpleNamespace(
                        step_index=1,
                        element_states={1: {"alpha": [0.2]}},
                    ),
                )
            ),
            "equivalent_plastic_strain_history",
        )

    for reactions in (
        {1: np.zeros(5)},
        {1: np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.nan])},
        {1.5: np.zeros(6)},
        {1: np.zeros(6), "1": np.ones(6)},
        {"not-a-node": np.zeros(6)},
    ):
        malformed_reaction = SimpleNamespace(reactions=reactions)
        assert "reaction" not in {
            item.quantity_id for item in describe_result_quantities(malformed_reaction)
        }
        with pytest.raises(QuantityUnavailableError):
            resolve_result_quantity(malformed_reaction, "reaction")

    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(
            SimpleNamespace(
                element_states={1: {"alpha": [0.2]}, "1": {"alpha": [0.3]}}
            ),
            "equivalent_plastic_strain",
        )


def test_snapshot_displacement_descriptor_and_data_are_aligned() -> None:
    snapshots = (
        SimpleNamespace(displacements=np.arange(6, dtype=float)),
        SimpleNamespace(displacements=np.arange(6, dtype=float) + 10.0),
    )
    result = SimpleNamespace(
        displacements=np.arange(6, dtype=float) + 99.0,
        snapshots=snapshots,
    )

    resolved = resolve_result_quantity(result, "displacement")

    assert resolved.descriptor.data_path == "snapshots[].displacements"
    assert resolved.descriptor.frame_count == 2
    assert isinstance(resolved.data, tuple)
    np.testing.assert_array_equal(resolved.data[0], snapshots[0].displacements)
    np.testing.assert_array_equal(resolved.data[1], snapshots[1].displacements)

    mismatched = SimpleNamespace(
        displacements=np.arange(12, dtype=float),
        snapshots=snapshots,
    )
    final = resolve_result_quantity(mismatched, "displacement")
    assert final.descriptor.data_path == "displacements"
    assert final.data is mismatched.displacements


def test_solve_outcome_round_trip_is_public_and_fail_closed() -> None:
    outcome = SolveOutcome.stopped(
        "minimum_increment_reached",
        requested_control=1.0,
        achieved_control=0.8,
        last_converged_frame=4,
    )

    assert outcome.disposition is SolveDisposition.PARTIAL
    assert outcome.partial and outcome.has_results and not outcome.target_reached
    assert SolveOutcome.from_dict(outcome.to_dict()) == outcome
    assert solve_outcome(SimpleNamespace(outcome=outcome)) is outcome
    with pytest.raises(TypeError):
        solve_outcome(SimpleNamespace(status="completed"))
    with pytest.raises(TypeError):
        SolveOutcome.from_dict(
            {
                "status": "partial",
                "termination": "minimum_increment_reached",
                "target_reached": False,
                "converged": True,
                "has_results": "false",
            }
        )
    with pytest.raises(ValueError):
        SolveOutcome.from_dict(
            {
                **outcome.to_dict(),
                "unexpected": "not-authoritative",
            }
        )


def test_real_nonlinear_result_is_adapted_without_false_completion() -> None:
    result = NonlinearStaticResult(
        steps=[],
        status="diverged",
        displacements=np.zeros(6),
        load_factor=0.0,
        info={"failure_reason": "nonconvergence"},
    )

    outcome = solve_outcome(result)

    assert outcome.failed
    assert not outcome.completed
    assert not outcome.has_results
    assert outcome.termination == "nonconvergence"


def test_public_workflow_result_families_have_exact_type_outcomes() -> None:
    generated = AnyStructureFEMResult(
        valid=False,
        status="invalid",
        invalid_reason="invalid generated geometry",
    )
    calibration = ImperfectionCalibrationResult(
        amplitude=0.0,
        capacity=0.0,
        iterations=0,
        converged=False,
        history=(),
        result=None,
    )

    assert solve_outcome(generated).failed
    assert solve_outcome(generated).termination == "invalid generated geometry"
    assert solve_outcome(calibration).failed

    spoof_type = type(
        "NonlinearStaticResult",
        (),
        {"__module__": "anyfesolver.nonlinear_static"},
    )
    with pytest.raises(TypeError):
        solve_outcome(spoof_type())


def test_mode_outcomes_require_the_requested_mode_count() -> None:
    modal_mode = ModalMode(
        mode_number=1,
        eigenvalue=1.0,
        angular_frequency=1.0,
        frequency_hz=1.0 / (2.0 * np.pi),
        period=2.0 * np.pi,
        mode_shape=np.zeros(6),
        reduced_mode_shape=np.zeros(1),
        modal_mass=1.0,
        modal_stiffness=1.0,
        residual_norm=0.0,
        rigid_body_correlation=0.0,
        is_rigid_body=False,
    )
    modal = ModalResult([modal_mode], 2, "ok", {}, {}, {}, {})
    buckling_mode = BucklingMode(
        mode_number=1,
        load_factor=1.0,
        eigenvalue=-1.0,
        mode_shape=np.zeros(6),
        reduced_mode_shape=np.zeros(1),
        modal_stiffness=1.0,
        modal_geometric_stiffness=-1.0,
    )
    buckling = BucklingResult([buckling_mode], 2, "ok", {}, {})

    assert solve_outcome(modal).partial
    assert solve_outcome(modal).termination == "partial_modes_extracted"
    assert solve_outcome(buckling).partial
    assert solve_outcome(buckling).termination == (
        "partial_buckling_modes_extracted"
    )
    modal.num_modes_requested = 1
    assert solve_outcome(modal).completed
    modal.num_modes_requested = 0
    with pytest.raises(TypeError):
        solve_outcome(modal)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "status": "completed",
            "termination": "nonconvergence",
            "target_reached": False,
            "converged": False,
            "has_results": False,
        },
        {
            "status": "partial",
            "termination": "limit",
            "target_reached": True,
            "converged": True,
            "has_results": True,
        },
        {
            "status": "failed",
            "termination": "nonconvergence",
            "target_reached": False,
            "converged": False,
            "has_results": True,
        },
    ],
)
def test_solve_outcome_rejects_inconsistent_disposition_flags(payload) -> None:
    with pytest.raises(ValueError):
        SolveOutcome.from_dict(payload)
