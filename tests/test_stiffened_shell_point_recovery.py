"""Physical point actions, with explicit reference/current work provenance."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver import b3_ge, stiffened_shell
from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from anyfesolver import _stiffened_shell_ge_experimental as ge_experimental
from anyfesolver._stiffened_shell_point_recovery import point_record
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_ge_experimental import _panel


def _balance(record):
    s, b = np.asarray(record.shell_action), np.asarray(record.beam_action)
    distance = ((np.asarray(record.beam_position_m)-record.shell_position_m) +
                (np.asarray(record.beam_position_low_m)-record.shell_position_low_m))
    np.testing.assert_allclose(s[:3]+b[:3], 0., atol=1e-9)
    np.testing.assert_allclose(s[3:]+b[3:]+np.cross(distance, b[:3]), 0., atol=1e-8)


@pytest.mark.parametrize("sign", (-1., 1.))
def test_rotated_point_actions_are_spatial_and_work_conjugate(sign):
    offset = np.array((.03, -.07, sign*.1))
    joint = RigidPoseJoint(np.array((np.zeros(3), offset)), np.tile(np.eye(3), (2, 1, 1)))
    theta = np.array((.3, -.2, .4))
    frame = Rotation.from_rotvec(theta).as_matrix()
    positions = np.array(((1., 2., 3.), np.array((1., 2., 3.))+frame@offset))
    lam = np.array((10., -13., 21., 2., 5., -7.))
    trial = joint.evaluate_ports(positions, np.array((np.eye(3), frame)), lam,
        rotation_coordinates=np.array((theta, np.zeros(3))), charted_ports=(True, False))
    owner = SimpleNamespace(_point_incidence=(
        stiffened_shell.PointConnection(1, 101),))
    record = point_record(owner, 0, positions, np.zeros((2, 3)),
        trial.pose.residual[:12], trial.pose.constraints, "current")
    force_b = -frame@lam[:3]
    moment_b = -frame@lam[3:]
    np.testing.assert_allclose(record.beam_action, np.r_[force_b, moment_b], atol=1e-12)
    np.testing.assert_allclose(record.shell_action,
        np.r_[-force_b, -np.cross(frame@offset, force_b)-moment_b], atol=1e-12)
    assert not np.allclose(record.beam_action[3:], -lam[3:])
    assert not np.allclose(trial.residual[3:6], trial.pose.residual[3:6])
    _balance(record)
    # Independent constraint work under the ACTUAL spatial left updates.
    variation = np.array((.1, -.2, .3, .2, -.4, .1, -.3, .1, .2, -.1, .2, .3))
    def work(amount):
        d = amount*variation.reshape(2, 6)
        x = positions+d[:, :3]
        qs, qb = (Rotation.from_rotvec(d[i, 3:]).as_matrix()@frame for i in range(2))
        gap = np.r_[qs.T@(x[1]-x[0])-offset,
                    Rotation.from_matrix(qs.T@qb).as_rotvec()]
        return float(lam@gap)
    h = 1e-6
    derivative = (work(h)-work(-h))/(2*h)
    np.testing.assert_allclose(-np.r_[record.shell_action, record.beam_action]@variation,
                               derivative, rtol=1e-8, atol=1e-8)
    positions[:] = 0.
    assert record.shell_position_m == (1., 2., 3.)
    with pytest.raises(FrozenInstanceError):
        record.point_index = 2


def _offset_panel(sign):
    definition = _panel()
    e = 210e9; g = e/2.6
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((e*.01, g*.008, g*.008, g*1.6666666666666667e-5,
                 e*8.333333333333333e-6, e*8.333333333333333e-6)),
        np.eye(6), 1e20, 1e9)
    beams = tuple(b3_ge.define_beam("b3-ge", i+1,
        (101+2*i, 102+2*i, 103+2*i),
        [(i+j/2, -.5, sign*.1) for j in range(3)],
        np.tile(np.eye(3), (3, 1, 1)), section, np.eye(6)) for i in range(2))
    return replace(definition, beams=beams)


@pytest.mark.parametrize("sign,axis", ((-1., 1), (1., 1), (-1., 2), (1., 2)))
def test_accepted_actions_reference_and_finite_restart(sign, axis):
    definition = _offset_panel(sign)
    wrench = [0.]*6; wrench[axis] = 100.
    actions = (stiffened_shell.NodalAction("beam", 105, tuple(wrench)),)
    linear = stiffened_shell.create_analysis(definition=definition)
    result = linear.solve_linear(actions)
    q = np.r_[result.shell_displacement, result.beam_coordinates]
    assert len(result.point_wrenches) == 3
    for i, record in enumerate(result.point_wrenches):
        assert type(record) is stiffened_shell.PointTransferWrench
        assert record.configuration == "reference_linear"
        assert (record.point_index, record.shell_node_id, record.beam_node_id) == (i, i+1, 101+2*i)
        _balance(record)
        rows = linear._C[6*i:6*i+6]
        expected = -(rows[:, linear._joint_slots[i]].T@result.multipliers[6*i:6*i+6])
        np.testing.assert_array_equal(np.r_[record.shell_action, record.beam_action], expected)
        np.testing.assert_array_equal(record.gap, rows@q)
    restored_linear = stiffened_shell.create_analysis(definition=definition).restore(linear.checkpoint())
    assert restored_linear.point_wrenches == result.point_wrenches

    finite = stiffened_shell.create_analysis(definition=definition)
    current = finite.solve(actions, (1.,))
    assert any(np.linalg.norm(p.beam_action) > 0. for p in current.point_wrenches)
    for record in current.point_wrenches:
        assert record.configuration == "current"
        _balance(record)
    resumed = stiffened_shell.create_analysis(definition=definition)
    restored = resumed.restore(finite.checkpoint())
    assert restored.point_wrenches == current.point_wrenches
    rebased = resumed.rebase_rotations(restored.generation)
    for before, after in zip(restored.point_wrenches, rebased.point_wrenches):
        np.testing.assert_allclose(after.shell_action, before.shell_action, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(after.beam_action, before.beam_action, rtol=1e-10, atol=1e-9)
        _balance(after)
    assert rebased.generation == restored.generation+1


def test_recovery_is_cached_and_rejection_preserves_accepted_actions(monkeypatch):
    owner = stiffened_shell.create_analysis(definition=_panel())
    actions = (stiffened_shell.NodalAction("beam", 105, (0., 0., 100., 0., 0., 0.)),)
    accepted = owner.solve(actions, (1.,))
    driver = owner._finite_driver
    with pytest.raises(ValueError, match="stale finite point recovery"):
        driver._point_records(replace(driver._state))
    original = type(driver)._assemble
    def reject(self, state, factor, *args, **kwargs):
        result = original(self, state, factor, *args, **kwargs)
        if state is not self._state:
            raise RuntimeError("discard evaluated point trial")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(type(driver), "_assemble", reject)
        with pytest.raises(RuntimeError, match="discard evaluated point trial"):
            owner.solve(actions, (2.,))
    assert owner.recover() is accepted
    token = CancellationToken(); token.cancel("point recovery test")
    with pytest.raises(SolveCancelled):
        owner.solve(actions, (2.,), cancellation_token=token)
    assert owner.recover() is accepted
    with monkeypatch.context() as patch:
        patch.setattr(type(driver), "_assemble", lambda *a, **k: pytest.fail("recovery reevaluated mechanics"))
        patch.setattr(RigidPoseJoint, "evaluate_ports", lambda *a, **k: pytest.fail("recovery reevaluated joints"))
        assert owner.recover().point_wrenches is accepted.point_wrenches


def test_cancel_during_linear_point_capture_preserves_predecessor(monkeypatch):
    owner = stiffened_shell.create_analysis(definition=_panel())
    actions = (stiffened_shell.NodalAction("beam", 105, (0., 0., 100., 0., 0., 0.)),)
    accepted = owner.solve_linear(actions)
    checkpoint = owner.checkpoint()
    token = CancellationToken()
    original = ge_experimental.reference_point_records
    def capture(*args):
        records = original(*args)
        token.cancel("cancel after point capture")
        return records
    monkeypatch.setattr(ge_experimental, "reference_point_records", capture)
    with pytest.raises(SolveCancelled):
        owner.solve_linear(actions, cancellation_token=token)
    assert owner.recover() is accepted
    assert owner.checkpoint() == checkpoint


def test_low_position_parts_remain_owned_and_carry_lever_arm():
    joint = RigidPoseJoint(np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)))
    positions = np.array(((1., 0., 0.), (1., 0., 0.)))
    low = np.array(((0., 0., 0.), (2.**-54, 0., 0.)))
    trial = joint.evaluate(positions, np.tile(np.eye(3), (2, 1, 1)),
                           np.array((0., 1e16, 0., 0., 0., 0.)), position_low=low)
    owner = SimpleNamespace(_point_incidence=(
        stiffened_shell.PointConnection(1, 101),))
    record = point_record(owner, 0, positions, low, trial.residual[:12],
                          trial.constraints, "current")
    assert record.beam_position_m == record.shell_position_m
    assert record.beam_position_low_m == (2.**-54, 0., 0.)
    assert abs(record.shell_action[5]) > .5
    _balance(record)
    low[:] = 0.
    assert record.beam_position_low_m == (2.**-54, 0., 0.)
