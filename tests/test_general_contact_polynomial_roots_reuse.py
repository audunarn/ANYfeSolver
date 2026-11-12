"""Exact-result reuse controls; the independent root oracles remain unchanged."""

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import gc
import math
from threading import Barrier, Lock
import weakref

import pytest

import anyfesolver.general_contact.polynomial_roots as roots


@pytest.fixture
def isolated_cache(monkeypatch):
    cache = roots._RootResultCache()
    monkeypatch.setattr(roots, '_RESULT_CACHE', cache)
    return cache


def _bits(result):
    return tuple(value.hex() for value in result)


def _snapshot(cache):
    with cache._lock:
        return (tuple(cache._entries),
                cache._payload_bytes + roots.getsizeof(cache._entries))


def _count_misses(monkeypatch):
    original = roots._near_real_roots_uncached
    calls = []

    def counted(p, tolerance):
        calls.append((p, tolerance))
        return original(p, tolerance)

    monkeypatch.setattr(roots, '_near_real_roots_uncached', counted)
    return calls


@pytest.mark.parametrize(('coefficients', 'tolerance'), [
    ([0.], 0.), ([3.], 1.e-10),
    ([-.25, 1., 0., 0.], 1.e-10),
    ([.25, -1., 1.], 1.e-10),
    ([-.125, .75, -1.5, 1.], 1.e-10),
    ([0., .5, -1.5, 1.], 1.e-10),
    ([-2., 0., 0., 1.], 1.e-10),
    ([1., 0., 1.], 1.),
    ([-1., 1., -1., 1.], .5),
    ([-2.**-80, .75 * 2.**-80, -1., .75], 1.e-10),
    ([-math.ulp(0.), 2.], 0.),
    ([math.ulp(0.), 2.], 0.),
    ([-3 * math.ulp(0.), 2.], 0.),
])
def test_cold_warm_and_evicted_outputs_match_uncached_bits(
        coefficients, tolerance, isolated_cache, monkeypatch):
    expected = roots._near_real_roots_uncached(roots._primitive(coefficients), tolerance)
    calls = _count_misses(monkeypatch)
    cold = roots.near_real_roots(coefficients, tolerance)
    warm = roots.near_real_roots(tuple(coefficients), tolerance)
    assert _bits(cold) == _bits(warm) == _bits(expected)
    assert warm is cold and len(calls) == 1
    monkeypatch.setattr(roots, '_RESULT_CACHE', roots._RootResultCache(max_entries=1))
    roots.near_real_roots(coefficients, tolerance)
    roots.near_real_roots([-123., 1.], tolerance)
    evicted = roots.near_real_roots(coefficients, tolerance)
    assert _bits(evicted) == _bits(expected) and len(calls) == 4


@pytest.mark.parametrize('degree', [2, 3])
def test_warm_near_imaginary_cutoff_uses_exact_tolerance_bits(
        degree, isolated_cache, monkeypatch):
    threshold = 1.e-10
    midpoint = (Fraction(threshold) + Fraction(math.nextafter(threshold, math.inf))) / 2
    square = midpoint * midpoint
    near = float(square)
    calls = _count_misses(monkeypatch)
    for value, admitted in ((math.nextafter(near, -math.inf), True),
                            (math.nextafter(near, math.inf), False)):
        coefficients = [value, 0., 1.] if degree == 2 else [-value, value, -1., 1.]
        expected = ((0.,) if admitted else ()) if degree == 2 else ((0., 1.) if admitted else (1.,))
        for _ in range(2):
            assert roots.near_real_roots(coefficients, threshold) == expected
    assert len(calls) == 2
    for _ in range(2):
        roots.near_real_roots(coefficients, math.nextafter(threshold, math.inf))
    assert len(calls) == 3


def test_key_keeps_length_signed_zero_scale_and_tolerance(isolated_cache, monkeypatch):
    calls = _count_misses(monkeypatch)
    inputs = [([0., 1.], 0.), ([-0., 1.], 0.), ([0., 1., 0.], 0.),
              ([0., 1., -0.], 0.), ([0., 2.], 0.), ([0., 1.], -0.),
              ([0., 1.], math.ulp(0.))]
    for coefficients, tolerance in inputs:
        roots.near_real_roots(coefficients, tolerance)
        roots.near_real_roots(coefficients, tolerance)
    assert len(calls) == len(inputs)
    assert len(_snapshot(isolated_cache)[0]) == len(inputs)


def test_mutation_and_one_shot_iterable_do_not_alias(isolated_cache, monkeypatch):
    calls = _count_misses(monkeypatch)
    values = [-.25, 1.]
    assert roots.near_real_roots(values) == (.25,)
    values[0] = -.5
    assert roots.near_real_roots(values) == (.5,)
    assert roots.near_real_roots(iter([-.25, 1.])) == (.25,)
    assert len(calls) == 2


def test_warm_lookup_repeats_conversion_primitive_and_tolerance_in_order(
        isolated_cache, monkeypatch):
    roots.near_real_roots([-.25, 1.], .1)
    events = []
    primitive = roots._primitive

    class Converted:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def __float__(self):
            events.append(self.name)
            return self.value

    def checked(values):
        events.append('primitive')
        return primitive(values)

    monkeypatch.setattr(roots, '_primitive', checked)
    assert roots.near_real_roots([Converted('c0', -.25), Converted('c1', 1.)],
                                 Converted('tolerance', .1)) == (.25,)
    assert events == ['c0', 'c1', 'primitive', 'tolerance']


def test_refusal_order_and_failed_misses_do_not_poison(isolated_cache, monkeypatch):
    roots.near_real_roots([0., 1.], 0.)
    before = _snapshot(isolated_cache)

    class ForbiddenTolerance:
        def __float__(self):
            raise AssertionError('tolerance evaluated before coefficient refusal')

    for coefficients in ([], [math.nan, 1.], [math.inf], [1., 0., 0., 0., 1.]):
        with pytest.raises(ValueError):
            roots.near_real_roots(coefficients, ForbiddenTolerance())
    for tolerance in (math.nan, math.inf, -1.):
        with pytest.raises(ValueError):
            roots.near_real_roots([0., 1.], tolerance)
    with pytest.raises(TypeError):
        roots.near_real_roots([object(), 1.], ForbiddenTolerance())
    assert _snapshot(isolated_cache) == before
    original = roots._near_real_roots_uncached

    def failed(p, tolerance):
        raise RuntimeError('injected algorithm failure')

    monkeypatch.setattr(roots, '_near_real_roots_uncached', failed)
    with pytest.raises(RuntimeError, match='injected algorithm failure'):
        roots.near_real_roots([-.75, 1.])
    assert _snapshot(isolated_cache) == before
    monkeypatch.setattr(roots, '_near_real_roots_uncached', original)
    assert roots.near_real_roots([-.75, 1.]) == (.75,)


def test_oversized_valid_sequence_bypasses_without_skipping_checks(isolated_cache, monkeypatch):
    calls = _count_misses(monkeypatch)
    values = [-.25, 1.] + [0.] * (roots._MAX_CACHE_KEY_BYTES // 8)
    assert roots.near_real_roots(values) == roots.near_real_roots(values) == (.25,)
    assert len(calls) == 2 and not _snapshot(isolated_cache)[0]
    values[-1] = 1.
    with pytest.raises(ValueError, match='degree exceeds three'):
        roots.near_real_roots(values)


def test_entry_limit_lru_and_byte_limit_change_only_reuse(monkeypatch):
    cache = roots._RootResultCache(max_entries=2)
    monkeypatch.setattr(roots, '_RESULT_CACHE', cache)
    calls = _count_misses(monkeypatch)
    for value in (1., 2., 1., 3., 1., 2.):
        assert roots.near_real_roots([-value, 1.]) == (value,)
        assert len(_snapshot(cache)[0]) <= 2
    assert len(calls) == 4  # The warm first root survived, the second was evicted.
    cache = roots._RootResultCache(max_entries=64, max_bytes=2048)
    monkeypatch.setattr(roots, '_RESULT_CACHE', cache)
    for value in range(40):
        assert roots.near_real_roots([-float(value), 1.]) == (float(value),)
        keys, retained_bytes = _snapshot(cache)
        assert retained_bytes <= 2048 and len(keys) <= 64
    assert len(keys) < 40


def test_inputs_and_mutable_root_isolators_are_not_retained(isolated_cache, monkeypatch):
    class Coefficients(list):
        pass

    values = Coefficients([-2., 0., 0., 1.])
    input_ref = weakref.ref(values)
    original = roots._Root
    isolators = []

    def tracked(*args):
        result = original(*args)
        isolators.append(weakref.ref(result))
        return result

    monkeypatch.setattr(roots, '_Root', tracked)
    result = roots.near_real_roots(values)
    del values
    gc.collect()
    assert input_ref() is None and isolators and all(r() is None for r in isolators)
    with pytest.raises(TypeError):
        result[0] = 0.
    assert roots.near_real_roots([-2., 0., 0., 1.]) is result


def test_concurrent_miss_publication_and_eviction_are_consistent(monkeypatch):
    cache = roots._RootResultCache(max_entries=2, max_bytes=4096)
    monkeypatch.setattr(roots, '_RESULT_CACHE', cache)
    original = roots._near_real_roots_uncached
    barrier = Barrier(4, timeout=10.)
    mutex = Lock()
    arrivals = 0

    def simultaneous(p, tolerance):
        nonlocal arrivals
        with mutex:
            arrivals += 1
            number = arrivals
        if number <= 4:
            barrier.wait()
        return original(p, tolerance)

    monkeypatch.setattr(roots, '_near_real_roots_uncached', simultaneous)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(roots.near_real_roots, [-.25, 1.]) for _ in range(4)]
        assert [future.result(timeout=15.) for future in futures] == [(.25,)] * 4
        values = [float(i % 7) for i in range(56)]
        results = list(pool.map(lambda value: roots.near_real_roots([-value, 1.]), values))
    assert results == [(value,) for value in values]
    keys, retained_bytes = _snapshot(cache)
    assert len(keys) <= 2 and retained_bytes <= 4096
    with cache._lock:
        assert cache._payload_bytes == sum(cache._size(k, v) for k, v in cache._entries.items())
