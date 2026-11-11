"""Mixed nodal-force and displacement static contact-path contracts."""

import os
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (
    ContactPathControl, ContactPathNodalForce, ContactPathProgram,
    solve_static_nonlinear,
)
from examples.general_contact_refined_q4_s3_mixed_path import (
    GAP_BUDGET_M, make_case, solve_case,
)
from anyfesolver.general_contact.core import contact_scope
from scripts.general_contact_d1_mixed_jet2_schur_probe import make_candidate


def test_jet2_schur_patch_uniform_slide_matches_independent_overlap_force():
    model, definition, settings, _program, _capability = make_candidate()
    source_nodes = range(1, 7)
    normal_dofs = [model.mesh.get_node(node).dofs[2]
                   for node in source_nodes]
    with contact_scope(model, definition, settings,
                       allow_augmented=True) as manager:
        for shift in (0.0, 0.05):
            displacement = np.zeros(model.mesh.dof_manager.total_dofs)
            for node in source_nodes:
                dofs = model.mesh.get_node(node).dofs
                displacement[dofs[0]] = shift
                displacement[dofs[2]] = -50.0e-6
            _force, tangent, contact = manager.evaluate(
                displacement, tangent=True,
            )
            # The translated 1 m square source intersects the fixed target
            # over x=[0.1+shift,1] and y=[0.1,1]. Both physical skins have
            # a uniform 10 um compression, so epsilon*area*compression is
            # the exact resultant of this bounded penalty specialization.
            expected_area = (0.9-shift)*0.9
            expected_force = 10000.0*expected_area*10.0e-6
            assert contact.current_overlap_area == pytest.approx(
                expected_area, abs=1.0e-10,
            )
            assert contact.integration_force[2] == pytest.approx(
                expected_force, rel=1.0e-6,
            )
            assert contact.minimum_gap == pytest.approx(-10.0e-6, abs=1.0e-9)
            assert contact.active_pressure_dofs == 6
            assert contact.gap_tolerance_met
            assert (float(tangent[np.ix_(normal_dofs, normal_dofs)].sum())
                    == pytest.approx(10000.0*expected_area, rel=1.0e-5))
            np.testing.assert_allclose(
                np.asarray(contact.integration_force)
                + np.asarray(contact.target_force), 0.0, atol=1.0e-10,
            )


def test_contact_path_force_interpolation_and_legacy_contract():
    controls = (ContactPathControl.node(1, "ux"),
                ContactPathControl.node(2, "uz"))
    old = ContactPathProgram(controls, ((0.0, 0.0), (0.01, 0.0)), 2)
    assert old.contract() == {
        "controls": [{"terms": [[1, "ux", 1.0]]},
                     {"terms": [[2, "uz", 1.0]]}],
        "waypoints": [[0.0, 0.0], [0.01, 0.0]],
        "steps_per_segment": 2,
    }
    assert old.force_values_at(1).size == 0

    program = ContactPathProgram(
        controls, ((0.0, 0.0), (0.0, 0.0), (0.05, 0.0),
                   (0.05, 0.0), (0.05, 0.0)), 2,
        (ContactPathNodalForce(3, "uz", (0.0, -0.1, -0.1, 0.0, -0.1)),),
    )
    assert program.total_steps == 8
    assert program.force_values_at(0).tolist() == [0.0]
    assert program.force_values_at(1).tolist() == [-0.05]
    assert program.force_values_at(2).tolist() == [-0.1]
    assert program.force_values_at(5).tolist() == [-0.05]
    assert program.force_values_at(8).tolist() == [-0.1]
    assert program.target_at(3).tolist() == [0.025, 0.0]
    assert program.contract()["nodal_forces"] == [
        {"node_id": 3, "dof": "uz",
         "values_n": [0.0, -0.1, -0.1, 0.0, -0.1]},
    ]


def test_contact_path_force_contract_rejects_ambiguous_histories():
    controls = (ContactPathControl.node(1, "ux"),
                ContactPathControl.node(2, "uz"))
    points = ((0.0, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError, match="zero-change"):
        ContactPathProgram(controls, points)
    with pytest.raises(ValueError, match="start at zero"):
        ContactPathProgram(
            controls, points, nodal_forces=(
                ContactPathNodalForce(3, "uz", (1.0, 0.0)),
            ),
        )
    with pytest.raises(ValueError, match="unique DOFs"):
        ContactPathProgram(
            controls, points, nodal_forces=(
                ContactPathNodalForce(3, "uz", (0.0, -1.0)),
                ContactPathNodalForce(3, "uz", (0.0, -2.0)),
            ),
        )
    with pytest.raises(ValueError, match="match waypoints"):
        ContactPathProgram(
            controls, points, nodal_forces=(
                ContactPathNodalForce(3, "uz", (0.0, -1.0, -2.0)),
            ),
        )


@pytest.mark.skipif(
    os.environ.get("ANY_RUN_C1_MIXED_PATH") != "1",
    reason="opt-in coupled force/displacement four-phase path and restart",
)
def test_mixed_path_force_slide_unload_reload_and_split_restart():
    full = solve_case()
    assert full.status == "completed", full.failure_reason
    history = full.info["force_displacement_history"]
    assert len(history) == 8
    assert [history[i]["contact"]["active_pressure_dofs"] > 0
            for i in (1, 3, 5, 7)] == [True, True, False, True]
    assert all(row["contact"]["gap_tolerance_met"] for row in history)
    assert min(row["contact"]["minimum_gap"] for row in history) >= -GAP_BUDGET_M
    assert history[5]["contact"]["integration_force"] == [0.0, 0.0, 0.0]
    assert history[5]["contact"]["minimum_gap"] > 0.0
    assert full.restart_checkpoint["version"] == 2
    for row, step in zip(history, full.steps):
        contact = row["contact"]
        np.testing.assert_allclose(
            np.asarray(contact["integration_force"]) + contact["target_force"],
            0.0, atol=1.0e-6,
        )
        np.testing.assert_allclose(
            np.asarray(contact["integration_moment"]) + contact["target_moment"],
            0.0, atol=1.0e-6,
        )
        support_z = sum(value[2] for value in step.support_reactions.values())
        assert support_z + sum(row["nodal_forces_n"]) == pytest.approx(
            0.0, abs=1.0e-6,
        )
    assert history[1]["external_nodal_work_trapezoid_j"] > 0.0
    assert history[5]["external_nodal_work_trapezoid_j"] < 0.0

    prefix = solve_case(max_path_fraction=0.5)
    assert prefix.status == "completed", prefix.failure_reason
    assert len(prefix.steps) == 4
    assert prefix.info["general_contact"]["active_pressure_dofs"] > 0
    resumed = solve_case(restart_checkpoint=prefix.to_restart_checkpoint())
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=0.0, atol=1.0e-10)
    assert resumed.to_restart_checkpoint() == full.to_restart_checkpoint()

    model, definition, settings, program = make_case()
    changed_force = replace(program.nodal_forces[0], values_n=(
        0.0, -0.04, -0.04, 0.0, -0.04,
    ))
    changed_program = replace(
        program, nodal_forces=(changed_force, program.nodal_forces[1]),
    )
    with pytest.raises(ValueError, match="incompatible|contract|restart"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=changed_program,
            restart_checkpoint=prefix.to_restart_checkpoint(),
            max_iterations=18,
        )


@pytest.mark.skipif(
    os.environ.get("ANY_RUN_C1_ACTIVE_SECANT") != "1",
    reason="opt-in strict projected augmentation cycle and split restart",
)
def test_strict_active_secant_path_and_split_restart():
    def solve(*, fraction=1.0, checkpoint=None, acceleration="active_secant"):
        model, definition, settings, program = make_case()
        definition = replace(definition, interactions=(replace(
            definition.interactions[0], normal_stiffness=10000.0,
        ),))
        settings = replace(settings, pressure_tolerance=0.0002,
                           augmentation_acceleration=acceleration)
        return solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            max_load_factor=fraction, restart_checkpoint=checkpoint,
            emit_restart_checkpoint=True,
        )

    full = solve()
    assert full.status == "completed", full.failure_reason
    history = full.info["force_displacement_history"]
    assert len(history) == 8
    assert [history[i]["contact"]["active_pressure_dofs"] > 0
            for i in (1, 3, 5, 7)] == [True, True, False, True]
    assert all(row["normal_multiplier_change"] <= 0.0002
               and row["contact"]["gap_tolerance_met"] for row in history)
    assert history[1]["augmentation_acceleration_applied"] > 0
    assert history[7]["augmentation_acceleration_applied"] > 0
    assert full.restart_checkpoint["version"] == 2

    prefix = solve(fraction=0.5)
    assert prefix.status == "completed", prefix.failure_reason
    resumed = solve(checkpoint=prefix.to_restart_checkpoint())
    assert resumed.status == "completed", resumed.failure_reason
    resumed_history = resumed.info["force_displacement_history"]
    for actual, expected in zip(resumed_history, history, strict=True):
        assert actual["step_index"] == expected["step_index"]
        assert (actual["augmentation_iterations"]
                == expected["augmentation_iterations"])
        assert (actual["contact"]["active_pressure_dofs"]
                == expected["contact"]["active_pressure_dofs"])
        np.testing.assert_allclose(
            actual["contact"]["integration_force"],
            expected["contact"]["integration_force"],
            rtol=1.0e-10, atol=1.0e-12,
        )
        assert actual["contact"]["minimum_gap"] == pytest.approx(
            expected["contact"]["minimum_gap"], abs=1.0e-12,
        )
    with pytest.raises(ValueError, match="incompatible|contract|restart"):
        solve(checkpoint=prefix.to_restart_checkpoint(), acceleration="none")
