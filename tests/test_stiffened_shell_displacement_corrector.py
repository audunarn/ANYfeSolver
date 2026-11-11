"""Private bordered correction: singular and nonsymmetric equilibrium operators."""
import numpy as np
import pytest
from scipy import sparse
from anyfesolver._stiffened_shell_displacement_corrector import bordered_correction


@pytest.mark.parametrize("coupling",(0.,3.))
def test_displacement_control_resolves_loaded_mechanism(coupling):
    # First coordinate has zero tangent stiffness; second has stiffness two.
    # A load-amplitude unknown and first-coordinate control resolve equilibrium.
    matrix=sparse.csc_matrix([[0.,coupling],[0.,2.]])
    step,amplitude=bordered_correction(matrix,np.array([-1.,4.]),
        np.array([1.,0.]),0,-.3,np.array([0,1]))
    np.testing.assert_allclose(step,[.3,-2.],rtol=0.,atol=1e-15)
    assert amplitude==pytest.approx(-1.-2.*coupling)
    np.testing.assert_allclose(matrix@step-amplitude*np.array([1.,0.]),[1.,-4.])


def test_fixed_control_rejected():
    with pytest.raises(ValueError,match="free coordinate"):
        bordered_correction(sparse.eye(2,format="csc"),np.zeros(2),
                            np.ones(2),0,0.,np.array([1]))


def test_coupled_displacement_trial_preserves_owner_and_predecessor(request,monkeypatch):
    from anyfesolver import stiffened_shell as ss
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver._stiffened_shell_displacement_corrector import correct
    from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
    from test_stiffened_shell_bl5_coupled_material import coupled
    from test_stiffened_shell_bl5_material import make
    from test_stiffened_shell_bl5_finite import _action
    owner=make(request,coupled())
    programme=ss.StagedDeadLoadProgramme(owner.preflight()["identity"],(_action(500.),))
    owner.solve(programme,(.5,1.),max_iterations=20,max_backtracks=12)
    driver=owner._finite_driver;saved=owner.checkpoint()
    base=driver._programme.load_at(1.).copy();direction=base*.1
    dof=int(np.flatnonzero(direction)[0]);target=driver._state.q[dof]*1.01
    args=dict(base_load=base,load_direction=direction,amplitude=0.,
              control_dof=dof,target=target)
    with pytest.raises(ValueError,match="accepted endpoint"):
        correct(driver,**{**args,"base_load":base*1.001})
    trial=correct(driver,**args)
    assert trial.evaluation[2]<=1e-8 and trial.evaluation[3]<=1e-9
    assert abs(trial.state.q[dof]-target)/owner._reference_length<=1e-9
    assert trial.predecessor_state is driver._state
    assert trial.shell_predecessor is driver._shell_origins
    assert trial.beam_predecessor is driver._beam_origins
    assert np.max([np.max(s["alpha"]) for s in trial.evaluation[4]])>0.
    assert owner.checkpoint()==saved
    from hashlib import sha256
    from anyfesolver._stiffened_shell_displacement_checkpoint import encode_snapshot,validate_endpoint
    from anyfesolver._stiffened_shell_ge_finite_experimental import _material_arrays,_shell_material_arrays
    from anyfesolver._stiffened_shell_rotation_charts import physical_frames
    state=trial.state
    anchors=(np.tile(np.eye(3),(len(owner._shell_ids),1,1))
             if state.shell_anchors is None else state.shell_anchors)
    arrays={name:getattr(state,name) for name in ("q","frames","cells","resultants","multipliers")}
    arrays.update(shell_anchors=anchors,
        shell_frames=physical_frames(state.q[:owner._shell_count],anchors),
        base_load=trial.base_load,load_direction=trial.load_direction,
        applied_load=trial.base_load+trial.amplitude*trial.load_direction,
        seed=np.array([driver._state.q[dof],0.]),history=np.array([[target,trial.amplitude]]))
    arrays["beam_plastic"],arrays["beam_accumulated"]=_material_arrays(trial.evaluation[5])
    arrays["beam_origin_plastic"],arrays["beam_origin_accumulated"]=_material_arrays(trial.beam_predecessor)
    arrays["shell_plastic"],arrays["shell_alpha"]=_shell_material_arrays(trial.evaluation[4])
    arrays["shell_origin_plastic"],arrays["shell_origin_alpha"]=_shell_material_arrays(trial.shell_predecessor)
    metadata=dict(generation=owner._generation+1,seed_generation=owner._generation,
                  seed_sha256=sha256(saved).hexdigest(),control_dof=dof)
    raw=encode_snapshot(owner,**metadata,arrays=arrays)
    restored=validate_endpoint(owner,raw)
    assert restored[4][2]<=1e-8 and restored[4][3]<=1e-9
    np.testing.assert_array_equal(restored[3].q,state.q)
    assert restored[2].owner is not owner
    for fault in ("shell_alpha","frames","applied_load"):
        bad={k:np.array(v,copy=True) for k,v in arrays.items()}
        if fault=="shell_alpha":bad[fault].flat[0]+=1e-4
        elif fault=="frames":bad[fault].flat[0]+=1e-3
        else:
            bad["base_load"][dof]+=1.
            bad["applied_load"][dof]+=1.
        forged=encode_snapshot(owner,**metadata,arrays=bad)
        with pytest.raises(ValueError):validate_endpoint(owner,forged)
        assert owner.checkpoint()==saved
    base[:]=0.;direction[:]=0.
    assert np.any(trial.base_load) and np.any(trial.load_direction)
    assert not trial.base_load.flags.writeable and not trial.load_direction.flags.writeable
    token=CancellationToken();token.cancel("private trial")
    args.update(base_load=trial.base_load,load_direction=trial.load_direction)
    with pytest.raises(SolveCancelled):correct(driver,**args,cancellation_token=token)
    assert owner.checkpoint()==saved
    original=FinitePointDriver._assemble;origin=driver._shell_origins
    def invalidate(self,*a,**kw):
        value=original(self,*a,**kw)
        self._shell_origins=tuple(list(origin))
        return value
    try:
        with monkeypatch.context() as patch:
            patch.setattr(FinitePointDriver,"_assemble",invalidate)
            with pytest.raises(ValueError,match="stale displacement"):
                correct(driver,**args)
    finally:
        driver._shell_origins=origin
    assert owner.checkpoint()==saved
