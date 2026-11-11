"""Actual mixed update chart, including unmeasured algebraic variables."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver, _State
from anyfesolver._stiffened_shell_arc_trial import path_row
from anyfesolver._stiffened_shell_arc_metric import PhysicalChordMetric
from anyfesolver._ge_beam3_p5.algebra import rotation


@pytest.mark.parametrize('plane',(False,True))
def test_actual_shell_anchor_and_beam_multiplicative_row(plane):
    owner=SimpleNamespace(_shell_count=6,_beam_ids=(1,),_beam_parts=(None,),
        _coordinate_support=None)
    driver=object.__new__(FinitePointDriver);driver.owner=owner;driver._beam_nodal=6
    driver._compensated=False
    q=np.zeros(36);q[3:6]=[.3,-.2,.1]
    anchors=np.array([rotation([-.2,.1,.4])])
    parent=_State(q,np.array([rotation([.2,.3,-.1])]),
        np.tile(np.eye(3),(1,2,1,1)),np.zeros((1,18)),np.zeros(2),anchors)
    step=np.linspace(-.05,.06,38)
    state=driver._advance(parent,step,1.)
    metric=PhysicalChordMetric([.4,.6],2.)
    normal=(np.array([[.2,.3,-.1],[.1,-.4,.2]]),np.array([[.3,-.1,.2],[-.2,.1,.3]])) if plane else None
    value,row=path_row(driver,state,parent,metric,normal)
    direction=np.linspace(.3,-.2,38)
    h=1e-5
    plus=path_row(driver,driver._advance(state,direction,h),parent,metric,normal)[0]
    minus=path_row(driver,driver._advance(state,direction,-h),parent,metric,normal)[0]
    assert (plus-minus)/(2*h)==pytest.approx(row@direction,rel=1e-8)
    assert np.count_nonzero(row[12:])==0
    # A genuine shell chart rebase preserves physical distance, not q rotations.
    from anyfesolver._stiffened_shell_rotation_charts import physical_frames
    rebased=state.q.copy();rebased[3:6]=0.
    state2=replace(state,q=rebased,shell_anchors=physical_frames(state.q[:6],anchors))
    assert path_row(driver,state2,parent,metric,normal)[0]==pytest.approx(value,rel=1e-13)


def test_stale_owner_and_unrepresentable_radius_rejected():
    from threading import RLock
    from anyfesolver._stiffened_shell_arc_trial import correct
    owner=SimpleNamespace(_K=sparse.eye(12),_shell_count=6,_beam_ids=(1,),
        _bl5_shell_material=True,_free=np.arange(12),_commit_lock=RLock(),
        _finite_driver=object())
    driver=SimpleNamespace(owner=owner,_state=SimpleNamespace(multipliers=[]),
        _accepted=object(),_external=None)
    args=dict(load_direction=np.r_[1.,np.zeros(11)],metric=PhysicalChordMetric([.5,.5],1.),
        radius=1e-5,predictor_dof=0,orientation=1)
    with pytest.raises(ValueError,match='stale continuation driver'):correct(driver,**args)
    for radius in (1e-200,1e200):
        with pytest.raises(ValueError,match='squared radius'):
            correct(driver,**{**args,'radius':radius})
