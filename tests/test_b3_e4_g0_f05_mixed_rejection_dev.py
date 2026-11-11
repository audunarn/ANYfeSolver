"""F05 development screen for coincident yielded shell/beam state boundaries.

The pair below is a test-local composition of existing owner APIs.  It is not
the production mixed material adapter or an atomic mixed-model coordinator.
"""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from anyfesolver._ge_beam3_retained_plastic_program import ForceProgram, solve_force_program
from anyfesolver.e4_pl_s3_v2d_state import canonical_json_bytes
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve
from test_ge_beam3_retained_plastic_force_probe import model_with_plasticity


def _shell():
    model = FEModel("g0-f05-mixed-rejection-dev")
    material = model.add_material(
        "steel", 210e9, .3, density=7850.,
        hardening_curve=DNVC208MaterialCurve(
            sigma_prop=80e6, sigma_yield=85e6, sigma_yield_2=100e6,
            eps_p_y1=.002, eps_p_y2=.02, K=280e6, n=.18,
        ),
    )
    for index, point in enumerate(((0., 0., 0.), (1.2, .05, 0.), (.2, .95, 0.)), 1):
        model.add_node(index, *point)
    element = create_shell_element(
        1, [1, 2, 3], "steel", formulation="e4-pl-s3-v2d",
        thickness=.018, reference_normal=(0., 0., 1.),
    )
    model.add_element(1, element)
    virgin = element.init_model_bound_nonlinear_state(model.mesh, material, 5)
    accepted_u = np.zeros(18)
    accepted_u[6], accepted_u[12] = .008, -.002
    _, _, accepted = element.compute_nonlinear_response(
        model.mesh, material, accepted_u, virgin, 5, True,
    )
    return model, material, element, accepted_u, accepted


def test_yielded_pair_rejects_and_replays_before_separate_acceptance():
    shell_model, material, shell, accepted_u, accepted_shell = _shell()
    shell_before = shell.serialize_nonlinear_state(
        shell_model.mesh, material, accepted_shell, 5,
    )
    assert np.max(accepted_shell["alpha"]) > 0.
    assert canonical_json_bytes(accepted_shell) == shell_before

    program = ForceProgram(
        (.25, .5, 1., .5, 0., -.5, 0.), ((5, .4, -.005, 0.),),
        max_iterations=24,
    )
    beam_model = model_with_plasticity(1.)
    beam_before_result = solve_force_program(beam_model, program, stop_after=3)
    assert beam_before_result.status == "paused", beam_before_result.failure
    assert beam_before_result.completed_targets == 3
    assert any(np.any(history[:, 2] > 0.) for history in beam_before_result.state.histories)
    beam_before = beam_before_result.checkpoint

    trial_u = accepted_u.copy()
    trial_u[6] += .0002
    trial_u[12] -= .00005
    force, tangent, shell_trial = shell.compute_nonlinear_response(
        shell_model.mesh, material, trial_u, accepted_shell, 5, True,
    )
    assert tangent is not None and np.isfinite(force).all() and np.isfinite(tangent).all()
    assert np.any(np.asarray(shell_trial["alpha"]) > np.asarray(accepted_shell["alpha"]))
    shell_candidate = shell.serialize_nonlinear_state(
        shell_model.mesh, material, shell_trial, 5,
    )
    assert shell_candidate != shell_before

    def reject_before_beam_commit(event):
        if event["stage"] == "paired-plastic.before_commit":
            raise RuntimeError("F05 injected mixed-trial rejection")

    rejected_beam = solve_force_program(
        model_with_plasticity(1.), program, checkpoint=beam_before,
        stop_after=4, progress=reject_before_beam_commit,
    )
    assert rejected_beam.status == "failed"
    assert rejected_beam.completed_targets == 3
    assert "F05 injected mixed-trial rejection" in rejected_beam.failure
    assert rejected_beam.checkpoint == beam_before
    assert shell.serialize_nonlinear_state(
        shell_model.mesh, material, accepted_shell, 5,
    ) == shell_before

    # The rejected pair is replayed from the same accepted parents.  Only
    # after both owner operations succeed are their two artifacts selected.
    _, _, replay_shell = shell.compute_nonlinear_response(
        shell_model.mesh, material, trial_u, accepted_shell, 5, True,
    )
    assert shell.serialize_nonlinear_state(
        shell_model.mesh, material, replay_shell, 5,
    ) == shell_candidate
    accepted_beam = solve_force_program(
        model_with_plasticity(1.), program, checkpoint=beam_before,
        stop_after=4,
    )
    assert accepted_beam.status == "paused", accepted_beam.failure
    assert accepted_beam.completed_targets == 4
    assert accepted_beam.checkpoint != beam_before

    # Both trial owners feed an actual eccentric, nonzero-multiplier pose
    # action before this test-local pair is selected as accepted.
    shell_ref = np.array((1.2, .05, 0.))
    beam_node_id = accepted_beam.state.mechanical.positions.shape[0] - 1
    beam_ref_node = beam_model.mesh.nodes[beam_node_id + 1]
    beam_ref = np.array((beam_ref_node.x, beam_ref_node.y, beam_ref_node.z))
    frames = np.array((np.eye(3), np.eye(3)))
    joint = RigidPoseJoint(np.array((shell_ref, beam_ref)), frames)
    port_positions = np.array((
        shell_ref + trial_u[6:9],
        accepted_beam.state.mechanical.positions[beam_node_id],
    ))
    port_frames = np.array((
        np.eye(3),
        accepted_beam.state.mechanical.nodal_frames[beam_node_id],
    ))
    action = joint.evaluate_ports(
        port_positions, port_frames,
        np.array((2e6, -1e6, .5e6, 2e4, -3e4, 1e4)),
        rotation_coordinates=np.array((trial_u[9:12], np.zeros(3))),
        charted_ports=(True, False),
    )
    assert np.isfinite(action.residual).all() and np.isfinite(action.tangent).all()
    assert np.linalg.norm(action.tangent[:6, 6:12]) > 0.

    restored_shell = shell.deserialize_nonlinear_state(
        shell_model.mesh, material, shell_candidate, 5,
    )
    assert canonical_json_bytes(restored_shell) == shell_candidate
    restored_beam = solve_force_program(
        model_with_plasticity(1.), program,
        checkpoint=accepted_beam.checkpoint,
        expected_checkpoint_sha256=sha256(accepted_beam.checkpoint).hexdigest(),
        stop_after=4,
    )
    assert restored_beam.status == "paused", restored_beam.failure
    assert restored_beam.checkpoint == accepted_beam.checkpoint
    assert canonical_json_bytes(accepted_shell) == shell_before
