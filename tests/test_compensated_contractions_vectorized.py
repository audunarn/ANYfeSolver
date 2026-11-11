"""Bitwise differential test: vectorised local contractions vs the scalar policy."""
import numpy as np
import pytest

from anyfesolver import _compensated_contractions as fast

@pytest.fixture(autouse=True, params=["native", "numpy"])
def backend(request, monkeypatch):
    """Every comparison runs on the compiled kernels and on the numpy path."""
    if request.param == "native":
        if not fast._native.AVAILABLE:
            pytest.skip("numba not installed")
        monkeypatch.setattr(fast, "_USE_NATIVE", True)
    else:
        monkeypatch.setattr(fast, "_USE_NATIVE", False)
    return request.param


# --- frozen scalar reference (the original implementation, verbatim) ---
import numpy as np
from math import frexp as _frexp, fsum as _fsum, isfinite as _isfinite, ldexp as _ldexp



def _product_parts(a, b):
    if not _isfinite(a) or not _isfinite(b):
        raise ValueError("compensated contraction requires finite inputs")
    product = a * b
    if not _isfinite(product):
        raise OverflowError("compensated product overflow")
    if a == 0.0 or b == 0.0:
        return product, 0.0
    x, ex = _frexp(a)
    y, ey = _frexp(b)
    split = 134217729.0
    cx, cy = split * x, split * y
    xh, yh = cx - (cx - x), cy - (cy - y)
    xl, yl = x - xh, y - yh
    high = x * y
    low = ((xh * yh - high) + xh * yl + xl * yh) + xl * yl
    # In the normal range power-of-two scaling is exact. In the subnormal
    # range the rounded product need not equal a doubly rounded scaled high;
    # the contract allows that lost half-ulp and scaled-low rounding (<= one
    # minimum subnormal together), rather than claiming an exact expansion.
    return product, _ldexp(low, ex + ey)


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("compensated dot shape mismatch")
    parts = []
    for a, b in zip(left, right):
        parts.extend(_product_parts(float(a), float(b)))
    result = _fsum(parts)
    if not _isfinite(result):
        raise OverflowError("compensated sum overflow")
    return result


def reference_product(left, right):
    """Matrix/matrix or matrix/vector product; each dot uses the same policy."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.ndim != 2 or b.ndim not in (1, 2) or a.shape[1] != b.shape[0]:
        raise ValueError("compensated product shape mismatch")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("compensated product requires finite inputs")
    vector = b.ndim == 1
    columns = b.reshape((-1, 1)) if vector else b
    result = np.empty((a.shape[0], columns.shape[1]), dtype=float)
    for i in range(a.shape[0]):
        for j in range(columns.shape[1]):
            result[i, j] = dot(a[i], columns[:, j])
    return result[:, 0] if vector else result


def reference_sum_arrays(terms):
    """Fixed-order componentwise sum of already represented equal-shape terms."""
    arrays = [np.asarray(term, dtype=float) for term in terms]
    if not arrays or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("compensated array sum requires equal nonempty shapes")
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("compensated array sum requires finite inputs")
    result = np.empty_like(arrays[0])
    for index in np.ndindex(result.shape):
        result[index] = _fsum(float(a[index]) for a in arrays)
    if not np.all(np.isfinite(result)):
        raise OverflowError("compensated array sum overflow")
    return result


def reference_weighted_sum(terms, weights):
    """Quadrature reduction including compensation of each weight product."""
    arrays = [np.asarray(term, dtype=float) for term in terms]
    if not arrays or len(arrays) != len(weights) or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("compensated weighted sum shape mismatch")
    result = np.empty_like(arrays[0])
    for index in np.ndindex(result.shape):
        result[index] = dot([a[index] for a in arrays], weights)
    return result


# --- end reference ---


def same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and x.tobytes() == y.tobytes()


def rand(rng, shape, kind):
    if kind == "normal":
        return rng.standard_normal(shape)
    if kind == "wide":
        return rng.standard_normal(shape) * 10.0 ** rng.integers(-150, 150, shape)
    if kind == "sparse":
        return rng.standard_normal(shape) * (rng.random(shape) < 0.35)
    if kind == "cancel":
        base = rng.standard_normal(shape)
        return np.where(rng.random(shape) < .5, base, -base) * 2.0 ** rng.integers(-30, 30, shape)
    if kind == "tiny":
        return rng.standard_normal(shape) * 2.0 ** rng.integers(-1074, -900, shape)
    if kind == "signed-zero":
        return np.where(rng.random(shape) < .6, rng.choice([0.0, -0.0], shape), rng.standard_normal(shape))
    raise AssertionError(kind)


KINDS = ["normal", "wide", "sparse", "cancel", "tiny", "signed-zero"]


@pytest.mark.parametrize("kind", KINDS)
def test_product_bitwise(kind):
    rng = np.random.default_rng(11)
    for m, k, n in [(1, 1, 1), (3, 5, 2), (6, 24, 6), (24, 3, 24), (4, 7, 1)]:
        for other in KINDS:
            a, b = rand(rng, (m, k), kind), rand(rng, (k, n), other)
            assert same(fast.product(a, b), reference_product(a, b))
            v = rand(rng, (k,), other)
            assert same(fast.product(a, v), reference_product(a, v))


@pytest.mark.parametrize("kind", KINDS)
def test_sum_and_weighted_sum_bitwise(kind):
    rng = np.random.default_rng(5)
    for shape in [(), (3,), (4, 5), (24, 24)]:
        terms = [rand(rng, shape, kind) for _ in range(6)]
        weights = rand(rng, (6,), "normal")
        assert same(fast.sum_arrays(terms), reference_sum_arrays(terms))
        assert same(fast.weighted_sum(terms, weights), reference_weighted_sum(terms, weights))


@pytest.mark.parametrize("kind", KINDS)
def test_dot_bitwise(kind):
    rng = np.random.default_rng(3)
    for n in (0, 1, 2, 9, 40):
        a, b = rand(rng, (n,), kind), rand(rng, (n,), "wide")
        assert np.float64(fast.dot(a, b)).tobytes() == np.float64(dot(a, b)).tobytes()


@pytest.mark.parametrize("fn,args", [
    ("product", (np.array([[1e200, 1.0]]), np.array([[1e200], [1.0]]))),
    ("product", (np.array([[np.nan]]), np.array([[1.0]]))),
    ("dot", ([1e200, 1.0], [1e200, 1.0])),
    ("dot", ([1.0, np.inf], [1.0, 1.0])),
    ("sum_arrays", ([np.array([1.7e308]), np.array([1.7e308])],)),
    ("weighted_sum", ([np.array([1e200])], [1e200])),
])
def test_refusals_match(fn, args):
    ref = {"product": reference_product, "dot": dot, "sum_arrays": reference_sum_arrays,
           "weighted_sum": reference_weighted_sum}[fn]
    with pytest.raises(Exception) as expected:
        ref(*args)
    with pytest.raises(expected.type):
        getattr(fast, fn)(*args)


def test_weighted_sum_first_failure_type_matches_scalar_policy():
    terms = [np.array([1.797e308, 0.0]), np.array([-0.0, 1e-310]),
             np.array([-1.797e308, 2.2e-162]), np.array([3.0, np.nan])]
    weights = [0.3148, -1.0, -1.0, -0.0]
    with pytest.raises(Exception) as expected:
        reference_weighted_sum(terms, weights)
    with pytest.raises(expected.type):
        fast.weighted_sum(terms, weights)


def test_prepared_q4_authority_covers_native_kernels(backend):
    if backend != "native":
        pytest.skip("the authority guard deliberately refuses a flipped backend flag")
    from anyfesolver import prepared_static_q4 as q
    assert {row[0] for row in q._NATIVE_KERNEL_FACTS} >= {"contract_product", "contract_sum", "contract_weighted"}
    q.require_q4_kernel_authority()
    kernels = fast._native
    assert not kernels.POW2.flags.writeable
    kernels.POW2.flags.writeable = True
    try:
        with pytest.raises(Exception, match="kernel authority"):
            q.require_q4_kernel_authority()
    finally:
        kernels.POW2.flags.writeable = False
    original = kernels.POW2[1100]
    kernels.POW2.flags.writeable = True
    try:
        kernels.POW2[1100] = 2.0
        with pytest.raises(Exception, match="kernel authority"):
            q.require_q4_kernel_authority()
    finally:
        kernels.POW2[1100] = original
        kernels.POW2.flags.writeable = False
    q.require_q4_kernel_authority()
