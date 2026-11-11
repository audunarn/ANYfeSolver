"""Differential arithmetic tests against rational Schur complements.

The oracle retains the earlier rational algorithm, not integer elimination.
This establishes represented-pencil equality, not continuum certification.
"""
from fractions import Fraction
import numpy as np
import pytest
from anyfesolver import _native_exact_shift_inertia as implementation


def rational_inertia(h, mass, shift):
    s = Fraction(float(shift))
    a = [[Fraction(float(x))-s*Fraction(float(y)) for x, y in zip(row, other)]
         for row, other in zip(h, mass)]
    p = q = z = 0
    while a:
        n = len(a); pivot = max(range(n), key=lambda i: abs(a[i][i]))
        if a[pivot][pivot]:
            order = [pivot]+[i for i in range(n) if i != pivot]
            a = [[a[i][j] for j in order] for i in order]; d = a[0][0]
            p += int(d > 0); q += int(d < 0)
            a = [[a[i][j]-a[i][0]*a[0][j]/d for j in range(1,n)] for i in range(1,n)]
        else:
            pair = next(((i,j) for i in range(n) for j in range(i+1,n) if a[i][j]), None)
            if pair is None: z += n; break
            order = list(pair)+[i for i in range(n) if i not in pair]
            a = [[a[i][j] for j in order] for i in order]; b = a[0][1]
            p += 1; q += 1
            a = [[a[i][j]-(a[i][0]*a[1][j]+a[i][1]*a[0][j])/b
                  for j in range(2,n)] for i in range(2,n)]
    return p,q,z


@pytest.mark.parametrize('seed', range(12))
def test_rational_oracle_general_shifted_pencils(seed):
    rng = np.random.default_rng(seed)
    for n in (1,2,3,5,8,12):
        h = rng.normal(size=(n,n)); h = h+h.T
        m = rng.normal(size=(n,n)); m = m+m.T  # no SPD assumption
        for shift in (0., -.125, float(rng.normal())):
            assert implementation.exact_inertia(h,m,shift) == rational_inertia(h,m,shift)


@pytest.mark.parametrize('seed', range(8))
def test_integer_congruences_singular_and_hyperbolic(seed):
    rng = np.random.default_rng(seed)
    t = np.eye(12, dtype=np.int64)
    for _ in range(12):
        i,j = rng.choice(12,2,replace=False); t[:,i] += t[:,j]
    d = np.diag([0,0,0,0,0,0,0,0,2,-3,0,0])
    for i in (0,2,4,6): d[i,i+1] = d[i+1,i] = (-1)**i*(i+1)
    h = np.asarray(t.T@d@t, dtype=float)
    assert implementation.exact_inertia(h,np.zeros_like(h),0.) == (5,5,2)
    assert rational_inertia(h,np.zeros_like(h),0.) == (5,5,2)


def test_exact_subtraction_not_rounded_subtraction():
    # Rounded binary64 H - s*M is zero, but exact represented value is +2^-104.
    m = np.array([[1.-2.**-52]]); h = np.array([[1.]])
    s = 1.+2.**-52
    assert float(h[0,0]-s*m[0,0]) == 0.
    assert implementation.exact_inertia(h,m,s) == (1,0,0)


def test_subnormals_and_extreme_negative_previous_pivots():
    h = np.diag([-1e300, np.nextafter(0.,1.), -1e-300, 2., 0.])
    assert implementation.exact_inertia(h,np.zeros_like(h),0.) == (2,2,1)


def test_deadline_and_cancellation_inside_elimination(monkeypatch):
    ticks = iter([0.,0.,0.,0.,31.])
    monkeypatch.setattr(implementation,'monotonic',lambda: next(ticks))
    with pytest.raises(TimeoutError,match='exact shifted inertia deadline'):
        implementation.exact_inertia(np.eye(3),np.eye(3),0.)
    monkeypatch.setattr(implementation,'monotonic',lambda: 0.)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 8: raise RuntimeError('cancel mid elimination')
    with pytest.raises(RuntimeError,match='cancel mid elimination'):
        implementation.exact_inertia(np.eye(4),np.eye(4),0.,cancel)


def test_inputs_remain_unchanged():
    h = np.array([[0.,3.,0.],[3.,0.,2.],[0.,2.,0.]])
    m = np.eye(3); before_h=h.copy(); before_m=m.copy()
    implementation.exact_inertia(h,m,.1)
    assert np.array_equal(h,before_h) and np.array_equal(m,before_m)


def test_benchmark_guard_accepts_reduction_stage(monkeypatch):
    from functools import partial
    from docs.reference_cases import ge_beam3_exact_inertia_benchmark as bench
    monkeypatch.setattr(bench.time,'monotonic',lambda: 10.)
    check=partial(bench.deadline_check,0.)
    check(); check('reduction.start')
    monkeypatch.setattr(bench.time,'monotonic',lambda: 121.)
    with pytest.raises(TimeoutError,match='arithmetic comparison deadline'):
        check('reduction.start')


def test_benchmark_complete_saved_packet_wiring(tmp_path,monkeypatch):
    import json
    import runpy
    from docs.reference_cases import ge_beam3_exact_inertia_benchmark as bench
    if not bench.INPUT.exists(): pytest.skip('external saved packet not present')
    monkeypatch.setattr(bench,'guard',lambda _:None)
    for key,value in bench.THREAD_ENVIRONMENT.items(): monkeypatch.setenv(key,value)
    calls=[]
    def stub(h,m,shift):
        assert h.shape==m.shape==(45,45) and type(shift) is float
        calls.append(shift); return (45,0,0)
    monkeypatch.setattr(implementation,'exact_inertia',stub)
    monkeypatch.setattr(implementation,'integer_inertia',stub)
    monkeypatch.setattr(runpy,'run_path',lambda _:dict(rational_inertia=stub))
    bench.worker('0'*40,tmp_path)
    result=json.loads((tmp_path/'benchmark.pending.json').read_bytes())
    assert len(calls)==24 and result['equal'] and result['timed_pairs']==11
    assert result['production_qualified'] is False
