"""Versioned local progress: analytic equilibria through the actual arc selector."""
from threading import RLock
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy import sparse
from anyfesolver._stiffened_shell_arc_metric import PhysicalChordMetric
from anyfesolver._stiffened_shell_arc_trial import correct,select_trial,ArcTrialFailure,ArcSelectionFailure
from anyfesolver._stiffened_shell_arc_checkpoint import identities


def planar_driver(monkeypatch,endpoint,orientation=1,legacy=False):
    # q(s)=(-orientation*s**2,s), with both coordinates in one physical node.
    # This exact oracle snaps only roundoff close to the known endpoint. The
    # residual, tangent, plane, Newton/backtracking and cutbacks remain actual.
    import anyfesolver._stiffened_shell_affine_endpoint as affine
    import anyfesolver._stiffened_shell_arc_checkpoint as checkpoint
    monkeypatch.setattr(affine,'is_native_point_material',lambda owner:True)
    monkeypatch.setattr(checkpoint,'require_native_seed',lambda owner,driver:None)
    a=.25;parent_q=np.zeros(6);parent_q[:2]=(-orientation*a*a,-a)
    parent=NS(q=parent_q,multipliers=np.empty(0),shell_anchors=None,frames=np.empty((0,3,3)))
    owner=NS(_shell_count=6,_beam_ids=(),_K=sparse.eye(6,format='csc'),
        _free=np.array([0,1]),_coordinate_scale=np.ones(6),_constraint_scale=np.empty(0),
        _bl5_shell_material=False,_generation=1,_commit_lock=RLock(),
        _definition=NS(shell=NS(s3_elements=(),j2_curve=None)))
    pattern=np.array([0.,1.,0.,0.,0.,0.])
    driver=NS(owner=owner,_state=parent,_shell_origins=(),_beam_origins=(),
        _accepted=object(),_external=None,_programme=None,_load_factor=-a,_pattern=pattern)
    owner._finite_driver=driver
    if legacy:owner._arc_snapshot=(b'',{'policy':identities(owner,native_version=1)[1]},None)
    def assemble(state,factor,shell,beam,*,applied_load):
        x,y=state.q[:2];residual=np.zeros(6)
        residual[:2]=(x+orientation*y*y,y-applied_load[1])
        tangent=sparse.eye(6,format='lil');tangent[0,1]=2*orientation*y
        return residual,tangent.tocsc(),float(np.linalg.norm(residual)),0.
    def advance(state,step,amount):
        q=state.q+amount*step
        for target in (endpoint,-endpoint):
            if abs(q[1]-target)<1e-8:
                q=q.copy();q[:2]=(-orientation*target**2,target)
                break
        return NS(q=q,multipliers=state.multipliers,shell_anchors=None,frames=state.frames)
    driver._assemble=assemble;driver._advance=advance;driver._increment_origin=lambda state:state
    # Radius is the exact forward frozen tangent projection at the endpoint.
    radius=(2*a*(a*a-endpoint**2)+endpoint+a)/np.sqrt(4*a*a+1)
    args=dict(load_direction=pattern,metric=PhysicalChordMetric([1.],1.),
              radius=radius,predictor_dof=0,orientation=orientation)
    return driver,args


@pytest.mark.parametrize('orientation',(-1,1))
def test_forward_corrected_coordinate_accepts(monkeypatch,orientation):
    driver,args=planar_driver(monkeypatch,.1,orientation)
    parent=driver._state
    selection=select_trial(driver,**args,max_cutbacks=0,max_attempts=1)
    assert selection.attempts[0].accepted
    assert orientation*(selection.trial.state.q[0]-parent.q[0])>0.
    assert driver._state is parent and driver.owner._generation==1


@pytest.mark.parametrize('endpoint',(.25,.3),ids=('zero','negative'))
@pytest.mark.parametrize('orientation',(-1,1))
def test_nonprogress_actual_corrector_cutback_and_exhaustion(monkeypatch,endpoint,orientation):
    driver,args=planar_driver(monkeypatch,endpoint,orientation)
    parent=driver._state;before=parent.q.copy()
    with pytest.raises(ArcTrialFailure,match='no oriented progress') as failed:
        correct(driver,**args)
    trial=failed.value.trial
    assert orientation*(trial.state.q[0]-parent.q[0])<=0.
    assert trial.evaluation[2]<=1e-8 and abs(trial.path_error)<=1e-9
    assert orientation*trial.normal_translations[0,0]>0.
    with pytest.raises(ArcSelectionFailure) as exhausted:
        select_trial(driver,**args,max_cutbacks=0,max_attempts=1)
    assert len(exhausted.value.attempts)==1 and not exhausted.value.attempts[0].accepted
    assert 'no oriented progress' in exhausted.value.attempts[0].failure
    assert driver._state is parent and driver.owner._generation==1
    np.testing.assert_array_equal(parent.q,before)
    selection=select_trial(driver,**args,max_cutbacks=1,max_attempts=2)
    assert [row.accepted for row in selection.attempts]==[False,True]
    assert selection.trial.radius==args['radius']/2
    assert orientation*(selection.trial.state.q[0]-parent.q[0])>0.
    assert driver._state is parent


def test_historical_v1_retains_nonprogress_continuation(monkeypatch):
    driver,args=planar_driver(monkeypatch,.3,legacy=True)
    trial=select_trial(driver,**args,max_cutbacks=0,max_attempts=1).trial
    assert trial.state.q[0]<driver._state.q[0]


def test_v1_two_cycle_has_valid_frozen_planes(monkeypatch):
    driver,args=planar_driver(monkeypatch,.25,legacy=True)
    start=driver._state.q.copy()
    first=correct(driver,**args)
    assert first.state.q[0]==start[0] and first.state.q[1]==-start[1]
    driver._state=first.state;driver._load_factor=.25
    second=correct(driver,**args)
    np.testing.assert_array_equal(second.state.q,start)
    assert abs(first.path_error)<=1e-9 and abs(second.path_error)<=1e-9
    assert first.normal_translations[0,0]>0. and second.normal_translations[0,0]>0.
