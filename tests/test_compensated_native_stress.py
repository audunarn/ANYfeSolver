"""Large randomized bitwise/exception comparison of the compiled kernels vs the scalar policy."""
import numpy as np
import pytest

from anyfesolver import _compensated_contractions as fast
from test_compensated_contractions_vectorized import (
    dot as scalar_dot, reference_product, reference_sum_arrays, reference_weighted_sum)

pytestmark = pytest.mark.skipif(not fast._native.AVAILABLE, reason="numba not installed")

SPECIAL = np.array([0.0, -0.0, 1.0, -1.0, 2.0**-1074, -2.0**-1074, 2.0**-1022, 1.7976931348623157e308,
                    -1.7976931348623157e308, 1e200, 1e-200, 1e154, np.inf, -np.inf, np.nan, 3.0, 0.1])


def draw(rng, shape):
    mode = rng.integers(0, 6)
    if mode == 0:
        return rng.standard_normal(shape)
    if mode == 1:
        return rng.standard_normal(shape) * 10.0 ** rng.integers(-300, 300, shape)
    if mode == 2:
        return rng.choice(SPECIAL, shape)
    if mode == 3:
        return np.where(rng.random(shape) < .5, rng.choice(SPECIAL[:8], shape), rng.standard_normal(shape))
    if mode == 4:
        base = rng.standard_normal(shape)
        return np.where(rng.random(shape) < .5, base, -base) * 2.0 ** rng.integers(-60, 60, shape)
    return rng.standard_normal(shape) * 2.0 ** rng.integers(-1074, -950, shape)


def outcome(fn, *args):
    try:
        return "ok", np.asarray(fn(*args)).tobytes()
    except Exception as exc:  # noqa: BLE001 - comparing exception classes is the point
        return "err", type(exc)


def test_kernels_match_scalar_policy_including_refusals(monkeypatch):
    monkeypatch.setattr(fast, "_USE_NATIVE", True)
    rng = np.random.default_rng(2026)
    for _ in range(4000):
        m, k, n = rng.integers(1, 7, 3)
        a, b = draw(rng, (m, k)), draw(rng, (k, n))
        assert outcome(fast.product, a, b) == outcome(reference_product, a, b)
        t = int(rng.integers(1, 6))
        shape = tuple(rng.integers(1, 4, rng.integers(0, 3)))
        terms = [draw(rng, shape) for _ in range(t)]
        weights = list(draw(rng, (t,)))
        assert outcome(fast.sum_arrays, terms) == outcome(reference_sum_arrays, terms)
        assert outcome(fast.weighted_sum, terms, weights) == outcome(reference_weighted_sum, terms, weights)
