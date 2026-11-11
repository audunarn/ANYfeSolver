"""Independent exact-binary oracle for the single local contraction policy."""
from fractions import Fraction
import math
import copy
import inspect
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

from anyfesolver import _compensated_contractions as arithmetic
from anyfesolver import elements, vectorized_nonlinear
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from test_general_contact_c5_elastic_increment import case, pose


CASES = [
    ([], []),
    ([0.0, -0.0], [-1.0, 1.0]),
    ([1.0e16, 1.0, -1.0e16], [1.0, 1.0, 1.0]),
    ([1.0 + 2.0**-27, -1.0], [1.0 - 2.0**-27, 1.0]),
    ([2.0**500, 2.0**-500, -2.0**500], [1.0, 1.0, 1.0]),
    ([2.0**-1022, -2.0**-1022], [.5, .25]),
    ([2.0**-1074, 2.0**-1074], [.5, .5]),
    ([2.0**-1074, -2.0**-1074], [1.0, 1.0]),
    ([1.0e150, -1.0e150, 3.0], [1.0e150, 1.0e150, .25]),
    ([-3.25, 7.125, -2.5], [4.5, -.25, -8.0]),
]


def exact_dot(left, right):
    return sum((Fraction.from_float(float(a)) * Fraction.from_float(float(b))
                for a, b in zip(left, right)), Fraction(0))


def require_error_bound(actual, left, right):
    exact = exact_dot(left, right)
    # One output ulp covers documented fsum double rounding; one minimum
    # subnormal per product covers both high/residual scaling loss.
    bound = Fraction.from_float(math.ulp(actual)) + len(left) * Fraction(1, 2**1074)
    assert abs(Fraction.from_float(actual) - exact) <= bound


@pytest.mark.parametrize("left,right", CASES, ids=[
    "empty", "signed-zero", "sum-cancellation", "product-cancellation",
    "exponent-spread", "normal-to-subnormal", "lost-half-subnormals",
    "subnormal-cancellation", "large-finite", "mixed-signs"])
def test_exact_binary_dot_error_contract(left, right):
    result = arithmetic.dot(left, right)
    require_error_bound(result, left, right)
    if exact_dot(left, right) == 0:
        assert math.copysign(1.0, result) == 1.0


def test_product_residual_and_sum_compensation_are_active():
    assert arithmetic.dot(*CASES[2]) == 1.0
    assert arithmetic.dot(*CASES[3]) == -2.0**-54


@pytest.mark.parametrize("vector", [False, True], ids=["matrix", "vector"])
def test_matrix_product_uses_independent_binary_oracle(vector):
    a = np.array([[1e16, 1., -1e16], [1. + 2.**-27, -1., 0.]])
    b = np.array([[1., 1. - 2.**-27], [1., 1.], [1., 0.]])
    if vector:
        b = b[:, 0]
    result = arithmetic.product(a, b)
    for i in range(2):
        if vector:
            require_error_bound(float(result[i]), a[i], b)
        else:
            for j in range(2):
                require_error_bound(float(result[i, j]), a[i], b[:, j])


@pytest.mark.parametrize("left,right,error", [
    ([math.nan], [1.], ValueError), ([math.inf], [0.], ValueError),
    ([1e308], [2.], OverflowError), ([1e308, 1e308], [1., 1.], OverflowError),
    ([1.], [], ValueError),
], ids=["nan", "infinite-zero", "product-overflow", "sum-overflow", "shape"])
def test_dot_refuses_unsupported_arithmetic(left, right, error):
    with pytest.raises(error):
        arithmetic.dot(left, right)


def test_component_sum_cancellation_and_shape_refusal():
    np.testing.assert_array_equal(arithmetic.sum_arrays((
        np.array([1e16, -0.]), np.array([1., 0.]), np.array([-1e16, 0.]))), [1., 0.])
    with pytest.raises(ValueError):
        arithmetic.sum_arrays((np.zeros(2), np.zeros(3)))
    terms = [np.array([v]) for v in CASES[3][0]]
    assert arithmetic.weighted_sum(terms, CASES[3][1])[0] == -2.0**-54


def test_independent_baselines_and_current_batch_force_are_connected():
    model, element, material = case()
    geometry = element._nonlinear_geometry(model.mesh)
    scalar = elements._compensated_elastic_zero_tangent(element, model.mesh, material)
    batch = vectorized_nonlinear.compensated_elastic_batch_zero(
        geometry, material.elastic_modulus, material.poisson_ratio,
        material.shear_modulus, element.thickness, element.drilling_stabilization)
    qualified = element.compute_stiffness_matrix(model.mesh, material)
    expected_baseline = arithmetic.sum_arrays((qualified, arithmetic.sum_arrays((batch, -scalar))))
    increment = vectorized_nonlinear.elastic_shell_nonlinear_increment(
        pose(), geometry, material.elastic_modulus, material.poisson_ratio, element.thickness)
    actual = element._elastic_increment_response(
        model.mesh, material, pose(), None, True, batch_baseline=True)
    np.testing.assert_array_equal(actual[0], arithmetic.sum_arrays((
        arithmetic.product(expected_baseline, pose()), increment[0])))
    with pytest.raises(ValueError, match="ambiguous"):
        element._elastic_increment_response(model.mesh, material, pose(), None, True,
                                            batch_baseline=True, batch_linear=batch)


@pytest.mark.parametrize("dependency", ["primitive", "scalar"], ids=["primitive", "scalar"])
def test_changed_compensated_dependency_refuses(monkeypatch, dependency):
    model, element, material = case()
    element._elastic_increment_response(model.mesh, material, pose(), None, True,
                                        batch_baseline=True)
    module, name = ((arithmetic, "dot") if dependency == "primitive" else
                    (elements, "_compensated_elastic_zero_tangent"))
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
        element._elastic_increment_response(model.mesh, material, pose(), None, True,
                                            batch_baseline=True)


@pytest.mark.parametrize("boundary", ["batch", "scalar"], ids=["batch", "scalar"])
def test_original_snapshot_spans_both_baseline_preparations(boundary):
    raw = inspect.getclosurevars(QualifiedE4PLShellElement._elastic_increment_response).nonlocals["method"]
    generation = [0]
    captured = []
    def capture(*args):
        captured.append(generation[0])
        return generation[0]
    def baseline(which):
        if which == boundary:
            generation[0] += 1
        return np.eye(24)
    namespace = dict(raw.__globals__,
        _capture_q4_fast_input_snapshot=capture,
        _elements_module=SimpleNamespace(_compensated_elastic_zero_tangent=lambda *a: baseline("scalar")),
        _elastic_increment_module=SimpleNamespace(
            compensated_elastic_batch_zero=lambda *a: baseline("batch"),
            elastic_shell_nonlinear_increment=lambda *a, **k: (np.zeros(24), np.zeros((24, 24)), np.zeros((24, 24)))))
    body = FunctionType(raw.__code__, namespace, raw.__name__, raw.__defaults__, raw.__closure__)
    body.__kwdefaults__ = raw.__kwdefaults__
    fake = SimpleNamespace(_elastic_increment_admitted=lambda *a: True,
        _defining_class_input_snapshot_matches=lambda *args: args[-1] == generation[0],
        _nonlinear_geometry=lambda *a: {}, compute_stiffness_matrix=lambda *a: np.eye(24),
        thickness=.05, drilling_stabilization=.001)
    material = SimpleNamespace(elastic_modulus=1e5, poisson_ratio=.3, shear_modulus=1.)
    with pytest.raises(RuntimeError, match="inputs changed"):
        body(fake, object(), material, pose(), None, True, batch_baseline=True)
    assert captured == [0]


def test_committed_current_split_matches_direct_response_without_state_change():
    model, element, material = case()
    force, tangent, trial = element.compute_nonlinear_response(model.mesh, material, pose())
    sealed = element.seal_committed_current_tangent_state(model.mesh, material, pose(), trial, 5)
    saved = copy.deepcopy(sealed)
    actual = element.compute_committed_current_tangent_components(
        model.mesh, material, pose(), sealed, 5)
    np.testing.assert_array_equal(actual["force"], force)
    np.testing.assert_array_equal(actual["total"], tangent)
    # Canonical serialization handles all nested state provenance, not just arrays.
    from anyfesolver.e4_pl_element import canonical_json_bytes
    assert canonical_json_bytes(sealed) == canonical_json_bytes(saved)


def test_cancelled_vector_observation_does_not_commit_state():
    model, element, material = case()
    state = element.init_nonlinear_state(5)
    saved = copy.deepcopy(state)
    class Cancelled(BaseException):
        pass
    class CancelledVector:
        def __array__(self, dtype=None, copy=None):
            raise Cancelled()
    with pytest.raises(Cancelled):
        element._elastic_increment_response(model.mesh, material, CancelledVector(), state,
                                            True, batch_baseline=True)
    for key in state:
        np.testing.assert_array_equal(state[key], saved[key])
    assert element.compute_nonlinear_response(model.mesh, material, pose(), state)[1].shape == (24, 24)


@pytest.mark.parametrize("dependency", ["primitive", "scalar"], ids=["primitive", "scalar"])
def test_changed_compensated_dependency_refuses_single_law_route(monkeypatch, dependency):
    """Single elastic law (owner decision 2026-10-23): the public production
    route of a planar qualified Q4 must refuse the same dependency substitution
    as the retained direct helper above (fail closed)."""

    model, element, material = case()
    element.compute_nonlinear_response(model.mesh, material, pose(), None, 5, True)
    module, name = ((arithmetic, "dot") if dependency == "primitive" else
                    (elements, "_compensated_elastic_zero_tangent"))
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
        element.compute_nonlinear_response(model.mesh, material, pose(), None, 5, True)
