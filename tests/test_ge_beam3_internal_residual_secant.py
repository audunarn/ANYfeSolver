"""Numerical-globalization tests, without changing constitutive mechanics."""
import numpy as np
import pytest
from anyfesolver import _ge_beam3_generalized_static_boundary as boundary
from anyfesolver._ge_beam3_p5_seeded.core import canonical
import test_ge_beam3_native_spectral_prestress as preload
from test_ge_beam3_schur_line_program import save


@pytest.mark.parametrize('scale',(1.,1e-200,1e200))
def test_affine_plastic_compliance_secant(scale,tmp_path):
    base=np.zeros(24);base[-1]=-scale
    fraction=1/256
    rejected=base.copy();rejected[-1]=scale*(-1+1001*fraction)
    answer=boundary._interior_residual_secant(base,rejected,fraction)
    assert abs(answer*1001-1)<1e-14
    save(tmp_path/'secant.json',dict(scale=scale,fraction=answer))


@pytest.mark.parametrize('kind',('zero','constant','away','beyond','nan','shape','fraction'))
def test_secant_rejects_invalid_or_noninterior(kind):
    base=np.ones(24);other=-np.ones(24);fraction=.5
    if kind=='zero':base[:]=0;other[:]=0
    if kind=='constant':other=base.copy()
    if kind=='away':other=2*base
    if kind=='beyond':other=.5*base
    if kind=='nan':other[0]=np.nan
    if kind=='shape':base=base[:23]
    if kind=='fraction':fraction=0.
    with pytest.raises(ValueError):boundary._interior_residual_secant(base,other,fraction)


def test_actual_plastic_input_uses_verified_secant(monkeypatch,tmp_path):
    original=boundary._interior_residual_secant;calls=[]
    def recorded(base,rejected,fraction):
        result=original(base,rejected,fraction)
        calls.append(dict(fraction=fraction,proposal=result))
        return result
    monkeypatch.setattr(boundary,'_interior_residual_secant',recorded)
    m,states,_,_,_,store,error=preload.axial_state(.001,1,yield_force=.125)
    assert calls and all(0<r['proposal']<r['fraction'] for r in calls)
    assert sum(sum(h.accumulated)>0 for s in states.values() for h in s['response'].history.stations)==8
    assert store.generation==1 and not store.has_active_trial
    save(tmp_path/'plastic.json',dict(calls=calls,error=error,states=states))


@pytest.mark.parametrize('bad_fraction',(0.,1.))
def test_unverified_secant_cannot_commit(monkeypatch,bad_fraction,tmp_path):
    captured=[];original=preload.make_store
    def capture(*args):
        store=original(*args);captured.append((store,canonical(store.materialize()),store.generation))
        return store
    monkeypatch.setattr(preload,'make_store',capture)
    monkeypatch.setattr(boundary,'_interior_residual_secant',lambda *args:bad_fraction)
    with pytest.raises(ValueError,match='secant failed actual decrease'):
        preload.axial_state(.001,1,yield_force=.125)
    assert len(captured)==1
    store,before,generation=captured[0]
    assert canonical(store.materialize())==before and store.generation==generation
    assert not store.has_active_trial
    save(tmp_path/'failure.json',dict(proposal=bad_fraction,history_unchanged=True,trial_discarded=True))


def test_existing_elastic_path_never_calls_fallback(monkeypatch,tmp_path):
    def forbidden(*args):raise AssertionError('elastic path called fallback')
    monkeypatch.setattr(boundary,'_interior_residual_secant',forbidden)
    _,states,_,_,_,_,error=preload.axial_state(.001,1)
    save(tmp_path/'elastic.json',dict(error=error,states=states))
