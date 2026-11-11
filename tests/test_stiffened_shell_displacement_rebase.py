"""Displacement chart events preserve accepted history and pending control."""
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot, POLICY, CHART_POLICY, EVENT_POLICY, encode_snapshot
from test_stiffened_shell_bl5_coupled_material import coupled
from test_stiffened_shell_bl5_material import make
from test_stiffened_shell_bl5_finite import _action


@pytest.mark.parametrize("policy",(None,ss.GENERALIZED_BRANCH_EVENT_TRIAL))
def test_chart_event_pending_adaptive_restart(request, monkeypatch,policy):
    owner=make(request,coupled())
    programme=ss.StagedDeadLoadProgramme(owner.preflight()["identity"],(_action(500.),))
    seed=owner.solve(programme,(.5,1.),max_iterations=20,max_backtracks=12)
    initial=float(seed.beam_positions[-1,2]-.0675)
    targets=(initial*1.01,initial*1.02)
    options=dict(displacement_control=ss.DisplacementControl("beam",105,2,trial_policy=policy),
                 increment_control=ss.AdaptiveIncrementControl(2,8),max_iterations=20,max_backtracks=12)
    token=CancellationToken()
    with pytest.raises(SolveCancelled):
        owner.solve(_action(50.),targets,**options,cancellation_token=token,
                    on_accepted=lambda result:token.cancel("prefix"))
    before=owner.recover();saved=owner.checkpoint()
    unrebased=make(request,coupled());unrebased.restore(saved)
    m,a=decode_snapshot(owner,owner._displacement_snapshot[0]);assert m["policy"]==(POLICY if policy is None else EVENT_POLICY)
    with pytest.raises(ValueError,match="generation"):
        owner.rebase_rotations(before.generation-1)
    import anyfesolver._stiffened_shell_displacement_control as control
    original=control.cancellation_safe_point
    def cancel_before_commit(token,point):
        if point=="stiffened-shell.displacement.publish":token.cancel("precommit")
        original(token,point)
    with monkeypatch.context() as patch:
        patch.setattr(control,"cancellation_safe_point",cancel_before_commit)
        with pytest.raises(SolveCancelled):
            owner.rebase_rotations(before.generation,cancellation_token=CancellationToken())
    assert owner.checkpoint()==saved
    after=owner.rebase_rotations(before.generation)
    new_m,new_a=decode_snapshot(owner,owner._displacement_snapshot[0])
    assert new_m["policy"]==(CHART_POLICY if policy is None else EVENT_POLICY) and new_m["chart_events"]==1
    assert after.event_cell_evaluations==before.event_cell_evaluations
    assert after.trial_policy==before.trial_policy==policy
    assert new_m["generation"]==m["generation"]+1
    assert after.history==before.history and after.increment_attempts==before.increment_attempts
    assert after.amplitude==before.amplitude and after.target==before.target
    for name in ("shell_plastic","shell_alpha","beam_plastic","beam_accumulated",
                 "shell_origin_plastic","shell_origin_alpha","beam_origin_plastic",
                 "beam_origin_accumulated","history","applied_load","shell_frames"):
        np.testing.assert_array_equal(new_a[name],a[name])
    np.testing.assert_allclose(after.shell_frames,before.shell_frames,rtol=0,atol=1e-12)
    np.testing.assert_allclose(after.support_reactions,before.support_reactions,rtol=1e-8,atol=1e-8)
    np.testing.assert_array_equal(after.beam_station_accumulated,before.beam_station_accumulated)
    np.testing.assert_array_equal(after.shell_layer_alpha,before.shell_layer_alpha)
    assert np.count_nonzero(after.shell_chart_coordinates.reshape(-1,6)[:,3:])==0
    with pytest.raises(ValueError,match="no nonzero"):
        owner.rebase_rotations(after.generation)
    with pytest.raises(ValueError,match="generation"):
        encode_snapshot(owner,generation=new_m["generation"],seed_generation=m["seed_generation"],
            seed_sha256=m["seed_sha256"],control_dof=m["control_dof"],arrays=new_a,chart_events=2)
    rebased=owner.checkpoint();restored=make(request,coupled());restored.restore(rebased)
    assert restored.checkpoint()==rebased
    final=owner.solve(_action(50.),targets,**options)
    repeated=restored.solve(_action(50.),targets,**options)
    assert final.status==repeated.status=="completed"
    assert owner.checkpoint()==restored.checkpoint()
    final_m,_=decode_snapshot(owner,owner._displacement_snapshot[0])
    assert final_m["chart_events"]==1 and final_m["steps"]==2

    baseline=unrebased.solve(_action(50.),targets,**options)
    np.testing.assert_allclose(final.beam_positions,baseline.beam_positions,rtol=1e-9,atol=1e-12)
    np.testing.assert_allclose(final.shell_frames,baseline.shell_frames,rtol=1e-9,atol=1e-12)
    np.testing.assert_allclose(final.support_reactions,baseline.support_reactions,rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(final.shell_layer_alpha,baseline.shell_layer_alpha,rtol=1e-9,atol=1e-14)
    np.testing.assert_allclose(final.beam_station_accumulated,baseline.beam_station_accumulated,rtol=1e-9,atol=1e-14)
    from anyfesolver._stiffened_shell_ge_finite_experimental import _material_arrays, _shell_material_arrays
    _,accepted=decode_snapshot(owner,owner._displacement_snapshot[0])
    for actual,expected in zip(_material_arrays(owner._finite_driver._beam_origins),
                               (accepted["beam_plastic"],accepted["beam_accumulated"])):
        np.testing.assert_array_equal(actual,expected)
    for actual,expected in zip(_shell_material_arrays(owner._finite_driver._shell_origins),
                               (accepted["shell_plastic"],accepted["shell_alpha"])):
        np.testing.assert_array_equal(actual,expected)
