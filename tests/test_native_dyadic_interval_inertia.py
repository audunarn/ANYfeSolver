"""Integer endpoint inclusion checked with a separate Fraction oracle."""
from fractions import Fraction
import numpy as np
import pytest
from anyfesolver import _native_dyadic_interval_inertia as dyadic
from anyfesolver import _native_exact_shift_inertia as exact
from test_native_fraction_free_inertia import rational_inertia


@pytest.mark.parametrize('seed',range(8))
def test_outward_endpoint_operations(seed):
    rng=np.random.default_rng(seed); scale=1<<8
    for _ in range(100):
        a=tuple(sorted(map(int,rng.integers(-1000,1000,size=2))))
        b=tuple(sorted(map(int,rng.integers(-1000,1000,size=2))))
        corners=[Fraction(x*y,scale*scale) for x in a for y in b]
        lo,hi=dyadic._multiply(a,b,scale)
        assert Fraction(lo,scale)<=min(corners)<=max(corners)<=Fraction(hi,scale)
        if not b[0]<=0<=b[1]:
            corners=[Fraction(x,y) for x in a for y in b]
            lo,hi=dyadic._divide(a,b,scale)
            assert Fraction(lo,scale)<=min(corners)<=max(corners)<=Fraction(hi,scale)
        else:
            with pytest.raises(ZeroDivisionError): dyadic._divide(a,b,scale)


@pytest.mark.parametrize('seed',range(12))
def test_filter_answers_match_independent_rational_schur(seed):
    rng=np.random.default_rng(seed)
    for n in (1,2,3,6,10):
        h=rng.normal(size=(n,n)); h=h+h.T
        m=rng.normal(size=(n,n)); m=m+m.T
        shift=float(rng.normal())
        oracle=rational_inertia(h,m,shift)
        for precision in (256,512):
            result=dyadic.interval_inertia(h,m,shift,precision=precision)
            assert result is not None and result==oracle


def test_uncertain_zero_and_hyperbolic_cases_do_not_guess():
    h=np.array([[0.,1.],[1.,0.]])
    assert dyadic.interval_inertia(h,np.zeros_like(h),0.) is None
    assert exact.exact_inertia(h,np.zeros_like(h),0.)==(1,1,0)
    h=np.array([[1.,1.],[1.,1.]])
    assert dyadic.interval_inertia(h,np.zeros_like(h),0.)==(1,0,1)
    # A smallest subnormal cannot receive a sign from a 256/512-bit absolute
    # grid. It must fall through to the unchanged exact integer algorithm.
    h=np.diag([np.nextafter(0.,1.),-np.nextafter(0.,1.)])
    assert dyadic.interval_inertia(h,np.zeros_like(h),0.) is None
    assert exact.exact_inertia(h,np.zeros_like(h),0.)==(1,1,0)


def test_exact_represented_cancellation_not_binary64_subtraction():
    h=np.ones((1,1)); m=np.array([[1.-2.**-52]]); shift=1.+2.**-52
    assert h[0,0]-shift*m[0,0]==0.
    assert dyadic.interval_inertia(h,m,shift)==(1,0,0)


def test_precision_escalation_and_unresolved_fallback(monkeypatch):
    calls=[]
    def unresolved(*args,**kwargs): calls.append(kwargs['precision']); return None
    monkeypatch.setattr(exact,'interval_inertia',unresolved)
    assert exact.exact_inertia(np.diag([2.,-1.,0.]),np.zeros((3,3)),0.)==(1,1,1)
    assert calls==[256,512]


def test_invalid_inputs_cancellation_and_deadline(monkeypatch):
    with pytest.raises(ValueError,match='precision'): dyadic.interval_inertia(np.eye(1),np.eye(1),0.,precision=256.)
    with pytest.raises(ValueError,match='symmetric'): dyadic.interval_inertia(np.array([[1.,1.],[0.,1.]]),np.eye(2),0.)
    def stop(): raise RuntimeError('cancel dyadic')
    with pytest.raises(RuntimeError,match='cancel dyadic'): dyadic.interval_inertia(np.eye(1),np.eye(1),0.,stop)
    ticks=iter([0.,0.,31.]); monkeypatch.setattr(dyadic,'monotonic',lambda:next(ticks))
    with pytest.raises(TimeoutError,match='dyadic shifted inertia deadline'):
        dyadic.interval_inertia(np.eye(1),np.eye(1),0.)


def test_unchanged_binary64_inputs():
    h=np.array([[2.,.3],[.3,-1.]]); m=np.eye(2); before=h.tobytes(),m.tobytes()
    dyadic.interval_inertia(h,m,.2)
    assert (h.tobytes(),m.tobytes())==before


def test_saved_six_benchmark_wiring(tmp_path,monkeypatch):
    import json
    from fixtures.reference_cases import ge_beam3_exact_inertia_benchmark as bench
    if not bench.SIX_INPUT.exists(): pytest.skip('preserved native packet not available')
    monkeypatch.setattr(bench,'guard',lambda _:None)
    for key,value in bench.THREAD_ENVIRONMENT.items(): monkeypatch.setenv(key,value)
    calls=[]
    def stub(h,m,shift,*,dimension_limit):
        assert h.shape==m.shape==(69,69) and dimension_limit==96
        calls.append(shift); return (64,5,0)
    monkeypatch.setattr(exact,'exact_inertia',stub); monkeypatch.setattr(exact,'integer_inertia',stub)
    bench.worker('0'*40,tmp_path,'six')
    value=json.loads((tmp_path/'benchmark.pending.json').read_bytes())
    assert len(calls)==24 and value['dimension']==69 and value['equal']
    assert set(value['timings'])=={'integer','filtered'} and not value['production_qualified']
