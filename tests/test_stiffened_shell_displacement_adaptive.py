"""Adaptive signed displacement: rejected work and accepted-prefix restart."""
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from test_stiffened_shell_bl5_coupled_material import coupled
from test_stiffened_shell_bl5_material import make
from test_stiffened_shell_bl5_finite import _action
from test_stiffened_shell_displacement_increment import assert_single_cutback_history


@pytest.mark.parametrize("policy",(None,ss.GENERALIZED_BRANCH_EVENT_TRIAL))
def test_cutback_cancel_restart_preserves_attempts(request,monkeypatch,policy):
    owner=make(request,coupled())
    programme=ss.StagedDeadLoadProgramme(owner.preflight()["identity"],(_action(500.),))
    owner.solve(programme,(.5,1.),max_iterations=20,max_backtracks=12)
    seed_snapshot=owner.checkpoint()
    dof=owner._shell_count+6*owner._beam_index[105]+2
    initial=float(owner._finite_driver._state.q[dof]);targets=(initial*1.01,initial*1.02)
    control=ss.DisplacementControl("beam",105,2,trial_policy=policy);limits=ss.AdaptiveIncrementControl(2,8)
    kw=dict(displacement_control=control,increment_control=limits,max_iterations=20,max_backtracks=12)
    import anyfesolver._stiffened_shell_displacement_control as module
    original=module.correct;first=True
    def fail_once(driver,**args):
        nonlocal first
        if first:
            first=False
            detail=ss.DisplacementFailureDetail(driver.owner._generation,args["target"],0.,1,1,1,0,1.,0.,.01,event_cell_evaluations=2 if policy else 0)
            raise ss.DisplacementCorrectorFailure("injected bounded cutback",detail)
        return original(driver,**args)
    token=CancellationToken()
    with monkeypatch.context() as patch:
        patch.setattr(module,"correct",fail_once)
        with pytest.raises(SolveCancelled):
            owner.solve(_action(50.),targets,**kw,cancellation_token=token,
                        on_accepted=lambda r:token.cancel("accepted midpoint"))
    prefix=owner.recover();assert prefix.status=="accepted_prefix"
    assert len(prefix.increment_attempts)==2
    assert [r.accepted for r in prefix.increment_attempts]==[False,True]
    assert prefix.increment_attempts[0].event_cell_evaluations==(2 if policy else 0)
    saved=owner.checkpoint();restored=make(request,coupled());restored.restore(saved)
    assert restored.checkpoint()==saved
    with pytest.raises(ValueError,match="must match"):
        restored.solve(_action(50.),targets,**{**kw,"increment_control":ss.AdaptiveIncrementControl(3,8)})
    with pytest.raises(ValueError,match="must match"):
        restored.solve(_action(50.),(targets[-1],),displacement_control=control)
    a=owner.solve(_action(50.),targets,**kw)
    b=restored.solve(_action(50.),targets,**kw)
    assert_single_cutback_history(a,b,prefix,initial,targets,limits)
    assert a.free_equilibrium_ratio<=1e-8 and a.constraint_max<=1e-9
    assert owner.checkpoint()==restored.checkpoint()

    # Exhaustion before any displacement acceptance still persists spent budget.
    exhausted=make(request,coupled());exhausted.restore(seed_snapshot)
    def reject(driver,**args):
        detail=ss.DisplacementFailureDetail(driver.owner._generation,args["target"],0.,1,1,1,13,1.,0.,.01,event_cell_evaluations=2 if policy else 0)
        raise ss.DisplacementCorrectorFailure("injected domain rejection",detail)
    limited={**kw,"increment_control":ss.AdaptiveIncrementControl(0,1)}
    with monkeypatch.context() as patch:
        patch.setattr(module,"correct",reject)
        with pytest.raises(ss.DisplacementIncrementLimit):
            exhausted.solve(_action(50.),targets,**limited)
    empty_prefix=exhausted.checkpoint()
    again=make(request,coupled());again.restore(empty_prefix)
    assert again.checkpoint()==empty_prefix
    with pytest.raises(ss.DisplacementIncrementLimit):again.solve(_action(50.),targets,**limited)
    with pytest.raises(ValueError,match="explicit displacement"):again.solve(_action(50.),(1.,))
    # An observer exception with the numerical exception type must propagate,
    # preserving the already accepted controller, not fabricate a rejection.
    observed=make(request,coupled());observed.restore(seed_snapshot)
    def observer(result):
        detail=ss.DisplacementFailureDetail(result.generation,result.target,result.amplitude,0,1,0,0,0.,0.,0.)
        raise ss.DisplacementCorrectorFailure("observer error",detail)
    with pytest.raises(ss.DisplacementCorrectorFailure,match="observer error"):
        observed.solve(_action(50.),targets,**kw,on_accepted=observer)
    assert len(observed.recover().increment_attempts)==1
    assert observed.recover().increment_attempts[0].accepted
    verified=make(request,coupled());verified.restore(observed.checkpoint())
    assert verified.checkpoint()==observed.checkpoint()
