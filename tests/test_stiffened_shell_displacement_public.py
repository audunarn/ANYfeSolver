"""Experimental displacement workflow, including accepted-prefix persistence."""
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from test_stiffened_shell_bl5_coupled_material import coupled
from test_stiffened_shell_bl5_material import make
from test_stiffened_shell_bl5_finite import _action


@pytest.mark.parametrize("policy",(None,ss.GENERALIZED_BRANCH_EVENT_TRIAL))
def test_public_displacement_cancel_restart_and_atomic_publication(request,monkeypatch,policy):
    owner=make(request,coupled())
    programme=ss.StagedDeadLoadProgramme(owner.preflight()["identity"],(_action(500.),))
    seed=owner.solve(programme,(.5,1.),max_iterations=20,max_backtracks=12)
    seed=owner.rebase_rotations(seed.generation)
    dof=owner._shell_count+6*owner._beam_index[105]+2
    initial=owner._finite_driver._state.q[dof]
    targets=(initial*1.01,initial*1.02)
    control=ss.DisplacementControl("beam",105,2,trial_policy=policy)
    token=CancellationToken()
    def observe(result):
        assert isinstance(result,ss.DisplacementResult)
        assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
        assert owner.recover().generation==result.generation
        token.cancel("accepted displacement prefix")
    kw=dict(displacement_control=control,max_iterations=20,max_backtracks=12)
    with pytest.raises(SolveCancelled):
        owner.solve(_action(50.),targets,**kw,cancellation_token=token,on_accepted=observe)
    saved=owner.checkpoint();prefix=owner.recover()
    assert prefix.trial_policy==policy
    assert prefix.event_cell_evaluations>=0
    changed=ss.DisplacementControl("beam",105,2,trial_policy=
        ss.GENERALIZED_BRANCH_EVENT_TRIAL if policy is None else None)
    with pytest.raises(ValueError,match="match restart"):
        owner.solve(_action(50.),(targets[1],),**{**kw,"displacement_control":changed})
    assert owner.checkpoint()==saved
    from scipy.spatial.transform import Rotation
    np.testing.assert_allclose(prefix.shell_displacement.reshape(-1,6)[:,3:],
        Rotation.from_matrix(prefix.shell_frames).as_rotvec(),rtol=0.,atol=1e-14)
    assert np.linalg.norm(prefix.shell_displacement-prefix.shell_chart_coordinates)>1e-8
    assert prefix.target==targets[0] and len(prefix.history)==1
    restored=make(request,coupled());restored.restore(saved)
    assert restored.checkpoint()==saved
    np.testing.assert_array_equal(restored.recover().shell_displacement,prefix.shell_displacement)
    import anyfesolver._stiffened_shell_displacement_control as module
    def broken(*args,**kwargs):raise RuntimeError("injected prepublication failure")
    with monkeypatch.context() as patch:
        patch.setattr(module,"prepare",broken)
        with pytest.raises(RuntimeError,match="prepublication"):
            owner.solve(_action(50.),(targets[1],),**kw)
    assert owner.checkpoint()==saved
    with pytest.raises(ValueError,match="explicit displacement"):
        owner.solve(_action(50.),(1.,))
    with pytest.raises(ValueError,match="match restart"):
        owner.solve(_action(51.),(targets[1],),**kw)
    import anyfesolver._stiffened_shell_displacement_corrector as corrector
    def no_step(matrix,*args):return np.zeros(matrix.shape[0]),0.
    with monkeypatch.context() as patch:
        patch.setattr(corrector,"bordered_correction",no_step)
        with pytest.raises(ss.DisplacementCorrectorFailure) as failure:
            owner.solve(_action(50.),(targets[1],),**kw)
    detail=failure.value.detail
    assert detail.accepted_generation==prefix.generation
    assert detail.rejected_trials==13 and detail.linear_solves==2
    assert detail.evaluations==15 and detail.scaled_control_error>1e-9
    assert owner.checkpoint()==saved
    final=owner.solve(_action(50.),(targets[1],),**kw)
    other=restored.solve(_action(50.),(targets[1],),**kw)
    assert len(final.history)==2 and final.generation==seed.generation+2
    assert owner.checkpoint()==restored.checkpoint()
    np.testing.assert_array_equal(final.shell_displacement,other.shell_displacement)
