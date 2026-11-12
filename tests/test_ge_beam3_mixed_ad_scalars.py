"""Scalar acceleration against the independent historical Jet2 formulas."""
import warnings

import numpy as np
import pytest

from anyfesolver._ge_beam3_mixed_ad import Jet2


def _generic_product(left, right):
    return Jet2(left.value*right.value,
        left.gradient*right.value + right.gradient*left.value,
        left.hessian*right.value + right.hessian*left.value
        + np.outer(left.gradient, right.gradient)
        + np.outer(right.gradient, left.gradient))


def _generic_reciprocal(jet):
    if jet.value == 0.:
        raise ZeroDivisionError("jet reciprocal is singular")
    inverse = 1./jet.value
    first, second = -(inverse**2), 2.*inverse**3
    return Jet2(inverse, first*jet.gradient,
                first*jet.hessian + second*np.outer(jet.gradient, jet.gradient))


def _oracle(jet, scalar, divide):
    made = Jet2.constant(float(scalar), jet.gradient.size)
    return _generic_product(jet, _generic_reciprocal(made) if divide else made)


def _same(first, second):
    for a, b in ((np.array(first.value), np.array(second.value)),
                 (first.gradient, second.gradient), (first.hessian, second.hessian)):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
        # Signed zeros/infinities and every finite value must also match bits.
        valid = ~np.isnan(a)
        np.testing.assert_array_equal(a.view(np.uint64)[valid], b.view(np.uint64)[valid])


@pytest.mark.parametrize("divide", (False, True))
@pytest.mark.parametrize("scalar", (0., -0., 1., -1., 2.25, -3.,
    np.float64(.5), np.float32(-2.), np.int64(3), True, False, "4.5",
    1e-100, -1e-100, 1e-103, 1e-200, 1e-320, 1e308,
    float("inf"), -float("inf"), float("nan")))
def test_scalar_arithmetic_matches_full_formula_and_exceptions(scalar, divide):
    g = np.array((0., -0., 2., -3.))
    h = np.array(((0., -0., 2., -3.), (-0., 0., -.5, 4.),
                  (2., -.5, 0., -0.), (-3., 4., -0., 0.)))
    for value in (0., -0., 1.25, -2., float("inf"), float("nan")):
        jet = Jet2(value, g.copy(), h.copy())
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                expected = _oracle(jet, scalar, divide)
            except (ZeroDivisionError, OverflowError, FloatingPointError) as error:
                with pytest.raises(type(error)) as actual_error:
                    (jet/scalar) if divide else (jet*scalar)
                assert error.args == actual_error.value.args
                continue
            actual = jet/scalar if divide else jet*scalar
            _same(expected, actual)
            if not divide:
                _same(expected, scalar*jet)
        assert not np.shares_memory(jet.gradient, actual.gradient)
        assert not np.shares_memory(jet.hessian, actual.hessian)


@pytest.mark.parametrize("divide", (False, True))
@pytest.mark.parametrize("policy", ("ignore", "raise"))
def test_nonfinite_derivatives_preserve_array_errors(divide, policy):
    rng = np.random.default_rng(6)
    for gradient_bad, hessian_bad in ((np.inf, 0.), (np.nan, 0.), (0., np.inf),
                                    (0., np.nan), (np.inf, np.inf)):
        for scalar in (0., -0., 2., -2., 1e308):
            g = rng.normal(size=3);h = rng.normal(size=(3, 3))
            g[0], h[1, 2] = gradient_bad, hessian_bad
            jet = Jet2(-1.25, g, h)
            with warnings.catch_warnings(), np.errstate(all=policy):
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    expected = _oracle(jet, scalar, divide)
                except (FloatingPointError, ZeroDivisionError) as error:
                    with pytest.raises(type(error)) as actual:
                        jet/scalar if divide else jet*scalar
                    assert str(error) == str(actual.value)
                else:
                    _same(expected, jet/scalar if divide else jet*scalar)


def test_owned_noncontiguous_and_readonly_derivative_arrays():
    g = np.arange(8., dtype=float)[::2]
    h = np.arange(32., dtype=float).reshape(4, 8)[:, ::2]
    g.setflags(write=False);h.setflags(write=False)
    jet = Jet2(-1.3, g, h)
    for scalar in (-3., 2.):
        for divide in (False, True):
            expected = _oracle(jet, scalar, divide)
            actual = jet/scalar if divide else jet*scalar
            _same(expected, actual)
            actual.gradient[0] = 997.;actual.hessian[0, 0] = 996.
            assert jet.gradient[0] == 0. and jet.hessian[0, 0] == 0.


def test_conversion_once_and_malformed_layout_fallback():
    class Scalar:
        calls = 0
        def __float__(self):
            self.calls += 1
            return 2.
    for dtype in (np.float64, np.float32):
        for divide in (False, True):
            jet = Jet2(3., np.ones(3, dtype=dtype), np.eye(3, dtype=dtype))
            scalar = Scalar();actual = jet/scalar if divide else jet*scalar
            assert scalar.calls == 1
            _same(actual, _oracle(jet, 2., divide))
    for bad in (None, "bad", complex(2., 1.)):
        jet = Jet2(2., np.ones(3), np.eye(3))
        for divide in (False, True):
            with pytest.raises((TypeError, ValueError)):
                jet/bad if divide else jet*bad
    jet = Jet2(2., np.ones((2, 2)), np.ones((4, 4)))
    for divide in (False, True):
        with pytest.raises(ValueError):
            jet/2. if divide else jet*2.


def test_subclass_coercion_and_jet_products_remain_generic():
    class Derived(Jet2):
        def _coerce(self, other):
            return other if isinstance(other, Jet2) else Jet2.constant(float(other)+1., self.gradient.size)
    jet = Derived(2., np.ones(3), np.eye(3))
    _same(jet*3., _generic_product(jet, Jet2.constant(4., 3)))
    _same(jet/3., _generic_product(jet, _generic_reciprocal(Jet2.constant(4., 3))))
    left = Jet2(2., np.arange(3.), np.arange(9.).reshape(3, 3))
    right = Jet2(3., np.arange(3.)+1, np.eye(3))
    _same(left*right, _generic_product(left, right))
    _same(left/right, _generic_product(left, _generic_reciprocal(right)))
