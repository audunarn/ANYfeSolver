"""Adaptive selection ownership, failure accounting and cancellation."""
from threading import RLock
from types import SimpleNamespace
import numpy as np
import pytest
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver import _stiffened_shell_arc_trial as arc
from anyfesolver._stiffened_shell_arc_metric import PhysicalChordMetric


def context():
    owner=SimpleNamespace(_generation=4,_commit_lock=RLock())
    driver=SimpleNamespace(owner=owner,_state=object(),_shell_origins=object(),_beam_origins=object())
    owner._finite_driver=driver
    trial=SimpleNamespace(evaluations=3,linear_solves=2,rejected_trials=1,event_cell_evaluations=2)
    return owner,driver,trial


def test_attempt_limit_preserves_failed_work_and_does_not_retry_programming_errors(monkeypatch):
    owner,driver,trial=context();calls=[]
    def fail(driver,**kwargs):
        calls.append(kwargs['radius']);raise arc.ArcTrialFailure('no convergence',trial)
    monkeypatch.setattr(arc,'correct',fail)
    with pytest.raises(arc.ArcSelectionFailure) as caught:
        arc.select_trial(driver,radius=.1,max_attempts=2,load_direction=np.ones(1))
    assert calls==[.1,.05]
    assert sum(a.evaluations for a in caught.value.attempts)==6
    assert not any(a.accepted for a in caught.value.attempts)
    def broken(*args,**kwargs):raise MemoryError('allocation')
    monkeypatch.setattr(arc,'correct',broken)
    with pytest.raises(MemoryError):arc.select_trial(driver,radius=.1,load_direction=np.ones(1))


@pytest.mark.parametrize('change',('cancel','generation','driver'))
def test_return_rechecks_after_attempt_observation(monkeypatch,change):
    owner,driver,trial=context();token=CancellationToken()
    monkeypatch.setattr(arc,'correct',lambda *a,**kw:trial)
    class Records(list):
        def append(self,row):
            super().append(row)
            if change=='cancel':token.cancel('during selection')
            elif change=='generation':owner._generation+=1
            else:owner._finite_driver=object()
    with pytest.raises(SolveCancelled if change=='cancel' else ValueError):
        arc.select_trial(driver,radius=.1,load_direction=np.ones(1),
            cancellation_token=token,attempt_records=Records())


def test_unit_physical_normal_has_unit_origin_derivative():
    from anyfesolver._ge_beam3_p5.algebra import rotation
    metric=PhysicalChordMetric([.3,.7],2.)
    translations=np.array([[.3,.1,.2],[-.2,.4,.1]])
    spins=np.array([[.2,-.1,.3],[.1,.2,-.4]])
    norm=np.sqrt(np.sum(metric.weights*(np.sum((translations/metric.length)**2,axis=1)+np.sum(spins*spins,axis=1))))
    translations/=norm;spins/=norm
    reference=np.array([rotation([.2,.1,-.1]),rotation([-.1,.3,.2])])
    value,gt,gr=metric.plane(np.zeros((2,3)),reference,reference,translations,spins)
    assert value==0.
    assert np.sum(gt*translations)+np.sum(gr*spins)==pytest.approx(1.,rel=1e-14)
    for length in (1e-200,1e200):
        with pytest.raises(ValueError,match='squared metric length'):PhysicalChordMetric([1.],length)
