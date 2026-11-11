"""Explicit private dense budgets, not relaxed mechanical acceptance."""
import numpy as np
import pytest
from anyfesolver import _native_exact_shift_inertia as exact
from anyfesolver import _ge_beam3_controlled_fibre_modes as native


def test_exact_96_coordinate_inertia_and_unchanged_default():
    d=np.array([1.]*32+[-2.]*32+[0.]*32)
    h=np.diag(d); m=np.zeros_like(h)
    with pytest.raises(ValueError,match='bounded'): exact.exact_inertia(h,m,0.)
    assert exact.exact_inertia(h,m,0.,dimension_limit=96)==(32,32,32)
    # Exactly represented unimodular integer congruence with all sign types.
    t=np.eye(96); t[np.arange(95),np.arange(1,96)]=1.
    assert exact.exact_inertia(t.T@h@t,m,0.,dimension_limit=96)==(32,32,32)


@pytest.mark.parametrize('bad',[True,96.,0,65,128,None])
def test_unregistered_arithmetic_budget_rejects(bad):
    with pytest.raises(ValueError,match='dimension limit'):
        exact.exact_inertia(np.eye(1),np.eye(1),0.,dimension_limit=bad)


@pytest.mark.parametrize('bad',[True,128.,0,81,257,None])
def test_unregistered_native_budget_rejects_before_replay(bad,monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('replay after invalid budget')
    monkeypatch.setattr(native,'Context',forbidden)
    with pytest.raises(ValueError,match='coordinate limit'):
        native.prepare(None,None,b'',{},material_policy=native.FROZEN,coordinate_limit=bad)


def test_chain_propagates_explicit_budget_before_numerics(monkeypatch):
    from anyfesolver import _native_exact_shift_chain_modes as chain
    def forbidden(*args): raise AssertionError('numerics after invalid allocation')
    monkeypatch.setattr(chain,'_owned',forbidden)
    with pytest.raises(ValueError,match='dimension limit'):
        chain.solve_factor_chain_modes(None,None,None,None,(),(),bounds=(-1.,1.),exact_dimension_limit=128)


def test_budget_reaches_exact_fallback(monkeypatch):
    from anyfesolver import _native_exact_shift_chain_modes as chain
    def unresolved(*args): raise ValueError('unresolved shifted sign')
    monkeypatch.setattr(chain,'floating_inertia',unresolved)
    h=np.diag(np.arange(69,dtype=float)-35.25); m=np.zeros_like(h)
    with pytest.raises(ValueError,match='bounded'): chain.inertia(h,m,0.)
    assert chain.inertia(h,m,0.,dimension_limit=96)==36


def test_explicit_160_dimension_signed_congruence():
    d=np.array([1.]*64+[-2.]*64+[0.]*32)
    t=np.eye(160); t[np.arange(159),np.arange(1,160)]=1.
    h=t.T@np.diag(d)@t; m=np.zeros_like(h)
    with pytest.raises(ValueError,match='bounded'): exact.exact_inertia(h,m,0.,dimension_limit=96)
    assert exact.exact_inertia(h,m,0.,dimension_limit=160)==(64,64,32)


@pytest.mark.parametrize('bad',[True,512.,128,1024,None])
def test_invalid_retained_replay_budget_rejected_before_context(bad,monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('context after invalid replay budget')
    monkeypatch.setattr(native,'Context',forbidden)
    with pytest.raises(ValueError,match='retained coordinate budget'):
        native.prepare(None,None,b'',{},material_policy=native.FROZEN,max_coordinates=bad)
