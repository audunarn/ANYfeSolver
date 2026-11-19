"""S0: dense certificate, exact residual and forward-error decision of the coupled route."""

from decimal import Decimal, getcontext
from fractions import Fraction

import mpmath
import numpy as np
import pytest
from scipy import sparse

from anyfesolver import _coupled_certificate as cc
from anyfesolver._coupled_certificate import CoupledSolveError

EPS = np.finfo(float).eps
LIMIT = 1e-6 / EPS


def _csr(rows):
    matrix = sparse.csr_matrix(np.array(rows, dtype=float))
    matrix.sum_duplicates()
    return matrix


def _refusal(call):
    with pytest.raises(CoupledSolveError) as info:
        call()
    return info.value


# --- Ported counterexamples (round 4 and round 5), verbatim --------------------------------------------

def test_svd_rounding_cannot_admit_a_system_beyond_the_condition_limit():
    """Codex round-4: zero sweeps, computed kappa 4503599107.2 <= limit, exact 4503600287.7 > limit."""
    getcontext().prec = 70
    a = float.fromhex('0x1.a12456c948ee8p-1')
    b = float.fromhex('0x1.a12456c62d4bfp-1')
    assert (Decimal(a) + Decimal(b)) / (Decimal(a) - Decimal(b)) > Decimal(LIMIT)
    S = _csr([[a, b, 0., 0.], [b, a, 0., 0.], [0., 0., 1., 1.], [0., 0., 1., 0.]])
    certificate = cc._certify_solve_matrix(S)
    assert certificate['accepted'] is False
    assert certificate['equilibration_sweeps'] == 0
    assert certificate['condition_number'] <= LIMIT < certificate['condition_upper_bound']
    error = _refusal(lambda: cc.certified_solve(S, np.ones(4), raw_load_zero=False))
    assert (error.code, error.reason) == ('COUPLED_UNCERTIFIED', 'condition_above_limit')


def test_condition_upper_bound_brackets_the_exact_kappa_of_an_accepted_system():
    getcontext().prec = 70
    a, b = 1.0, 1.0 - 5e-10
    exact = (Decimal(a) + Decimal(b)) / (Decimal(a) - Decimal(b))
    S = _csr([[a, b, 0., 0.], [b, a, 0., 0.], [0., 0., 1., 1.], [0., 0., 1., 0.]])
    certificate = cc._certify_solve_matrix(S)
    assert certificate['accepted'] is True
    assert Decimal(certificate['condition_upper_bound']) >= exact
    assert certificate['condition_upper_bound'] <= certificate['condition_limit']


def _codex_round5_counterexample():
    getcontext().prec = 80
    size = 3000
    alpha = Decimal(max(1000, size) * size) * Decimal(2) ** -52
    limit = Decimal(LIMIT)
    t = 1 / (limit * (1 + Decimal('1e-9')))
    return size, alpha, limit, t


def test_condition_bound_refuses_the_codex_round5_counterexample():
    size, alpha, limit, t = _codex_round5_counterexample()
    assert 1 / t > limit
    old = (1 - alpha + alpha * (1 - alpha) + size * Decimal(2) ** -1074) / (
        t + alpha - alpha * (1 - alpha) - size * Decimal(2) ** -1074) * (1 + 4 * Decimal(2) ** -52)
    assert old <= limit
    bound = cc._condition_upper_bound(float(1 - alpha), float(t + alpha), size)
    assert bound['accepted'] is False
    assert bound['reason'] == 'condition_above_limit'
    assert Decimal(bound['condition_upper_bound']) >= 1 / t


def test_condition_bound_is_an_upper_bound_of_the_worst_case_in_exact_arithmetic():
    getcontext().prec = 80
    rng = np.random.default_rng(66)
    cases = [(1.0, 1.0, 1), (2.0, 1e-3, 191), (1.9, 4.4e-10, 3000), (1.5, 3.3e-10, 4), (1.2, 1e-8, 5000)]
    cases += [(float(rng.uniform(.5, 4.)), float(10 ** rng.uniform(-9.6, 0.)), int(rng.integers(1, 5001)))
              for _ in range(300)]
    for s1, sn, size in cases:
        sn = min(sn, s1)
        bound = cc._condition_upper_bound(s1, sn, size)
        alpha = Decimal(max(1000, size) * size) * Decimal(2) ** -52
        tau = size * Decimal(2) ** -1074
        top = Decimal(s1) / (1 - alpha) + tau
        bottom = Decimal(sn) - alpha * Decimal(s1) / (1 - alpha) - tau
        if bottom <= 0:
            assert bound['accepted'] is False
            continue
        if np.isfinite(bound['condition_upper_bound']):
            assert Decimal(bound['condition_upper_bound']) >= top / bottom
        assert bound['accepted'] == (bound['condition_upper_bound'] <= LIMIT)
        # The exact sigma^- of section 2.2 is the same bottom, so it is positive here.
        assert cc._sigma_lower(s1, sn, size) > 0


def test_condition_bound_refuses_when_alpha_reaches_one_half_or_inputs_are_invalid():
    size = int(np.ceil(np.sqrt(0.5 / EPS)))
    assert cc._condition_upper_bound(1.0, 1.0, size)['reason'] == 'svd_error_bound_invalid'
    for args in [(np.nan, 1., 3), (1., 2., 3), (0., 0., 3), (1., -1., 3), (np.inf, 1., 3)]:
        assert cc._condition_upper_bound(*args)['accepted'] is False


def test_sigma_n_equal_to_delta_is_refused_singular_within_svd_error():
    s1, size = 1.5, 5000
    probe = cc._condition_upper_bound(s1, s1, size)
    delta = probe['svd_error_bound']
    bound = cc._condition_upper_bound(s1, delta, size)
    assert (bound['accepted'], bound['reason']) == (False, 'singular_within_svd_error')
    # The exact sigma^- of section 2.2 is <= 0 at every s_N <= exact delta, and the float gate (rounded
    # outward) refuses there and a little above: the exact bound never admits what the gate refuses.
    alpha = Fraction(max(1000, size) * size) * Fraction(EPS)
    exact_delta = alpha / (1 - alpha) * Fraction(s1) + size * Fraction(1, 2 ** 1074)
    below = float(exact_delta)
    if Fraction(below) > exact_delta:
        below = float(np.nextafter(below, 0.))
    assert Fraction(below) <= exact_delta < Fraction(delta)
    assert cc._sigma_lower(s1, below, size) <= 0
    assert cc._condition_upper_bound(s1, below, size)['reason'] == 'singular_within_svd_error'
    above = float(np.nextafter(below, 1.))
    assert cc._sigma_lower(s1, above, size) > 0
    assert cc._condition_upper_bound(s1, above, size)['accepted'] is False


def test_injected_singular_values_at_delta_refuse_the_solve(monkeypatch):
    S = _csr(np.eye(3))
    delta = cc._condition_upper_bound(1.0, 1.0, 3)['svd_error_bound']
    monkeypatch.setattr(cc, '_svd_values', lambda A: np.array([1.0, 1.0, delta]))
    error = _refusal(lambda: cc.certified_solve(S, np.ones(3), raw_load_zero=False))
    assert (error.code, error.reason) == ('COUPLED_UNCERTIFIED', 'singular_within_svd_error')


# --- Cap and memory ------------------------------------------------------------------------------------

def test_cap_admits_5000_and_refuses_5001_before_any_dense_allocation(monkeypatch):
    calls = []
    original = sparse.csr_matrix.toarray

    def spy(self, *args, **kwargs):
        calls.append(self.shape)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(sparse.csr_matrix, 'toarray', spy)
    monkeypatch.setattr(cc, '_svd_values', lambda A: np.ones(A.shape[0]))  # no 45 s SVD in a unit test
    accepted = cc._certify_solve_matrix(sparse.identity(5000, format='csr'))
    assert accepted['accepted'] is True and accepted['size'] == 5000
    assert calls == [(5000, 5000)]
    calls.clear()
    error = _refusal(lambda: cc._certify_solve_matrix(sparse.identity(5001, format='csr')))
    assert (error.code, error.reason) == ('COUPLED_TOO_LARGE', 'equation_cap')
    assert calls == []


def test_empty_system_is_no_free_dofs():
    certificate = cc._certify_solve_matrix(sparse.csr_matrix((0, 0)))
    assert certificate['method'] == 'no_free_dofs' and certificate['accepted'] is True
    result = cc.certified_solve(sparse.csr_matrix((0, 0)), np.zeros(0), raw_load_zero=True)
    assert result.x.shape == (0,) and result.forward_error == 0


def test_memory_limit_refuses_before_densifying(monkeypatch):
    monkeypatch.setattr(sparse.csr_matrix, 'toarray', lambda self, *a, **k: pytest.fail('densified'))
    error = _refusal(lambda: cc._certify_solve_matrix(sparse.identity(100, format='csr'),
                                                      memory_limit_bytes=4 * 100 * 100 * 8 - 1))
    assert (error.code, error.reason) == ('COUPLED_TOO_LARGE', 'memory_limit')


@pytest.mark.parametrize('where', ['svd', 'candidate'])
def test_injected_memory_error_is_coupled_too_large(monkeypatch, where):
    def boom(*args, **kwargs):
        raise MemoryError
    if where == 'svd':
        monkeypatch.setattr(cc, '_svd_values', boom)
    else:
        monkeypatch.setattr(cc.sparse_linalg, 'splu', boom)
    error = _refusal(lambda: cc.certified_solve(_csr(np.eye(2)), np.ones(2), raw_load_zero=False))
    assert error.code == 'COUPLED_TOO_LARGE'


def test_exactly_singular_candidate_is_coded():
    error = _refusal(lambda: cc._candidate(_csr([[1., 1.], [1., 1.]]), np.ones(2)))
    assert (error.code, error.reason) == ('COUPLED_UNCERTIFIED', 'candidate_failed')


# --- Exact residual -------------------------------------------------------------------------------------

def test_exact_residual_matches_60_digit_mpmath_on_200_random_rows():
    mpmath.mp.dps = 60
    rng = np.random.default_rng(2026)
    n, rows = 40, 200
    dense = rng.standard_normal((rows, n)) * np.ldexp(1.0, rng.integers(-20, 21, size=(rows, n)))
    dense[rng.random((rows, n)) < 0.6] = 0.0
    S = sparse.csr_matrix(dense)
    x = rng.standard_normal(n) * np.ldexp(1.0, rng.integers(-20, 21, size=n))
    b = np.asarray(S @ x).ravel() * (1 + 1e-12 * rng.standard_normal(rows))
    exact = cc._exact_rows(S, x, b)
    differences = 0
    for i in range(rows):
        reference = mpmath.mpf(b[i]) - mpmath.fsum(mpmath.mpf(S[i, j]) * mpmath.mpf(x[j]) for j in range(n))
        ours = exact.residual_fraction(i)
        if float(ours) != float(reference):
            differences += 1
        assert abs(ours.numerator * mpmath.mpf(1) / ours.denominator - reference) <= abs(reference) * mpmath.mpf(10) ** -50
    assert differences == 0
    assert np.array_equal(exact.residual_float(), np.array([float(exact.residual_fraction(i)) for i in range(rows)]))


def test_cancellation_row_and_smallest_subnormal_rhs_are_exact():
    row = [1.0] + [2.0 ** -60] * 1023 + [-1.0]
    exact = cc._exact_rows(sparse.csr_matrix([row]), np.ones(len(row)), np.zeros(1))
    assert exact.residual_fraction(0) == Fraction(-1023, 2 ** 60)
    naive = 0.0
    for value in row:  # left to right in float: 1 + 2^-60 rounds back to 1 every time
        naive += value
    assert naive == 0.0
    tiny = cc._exact_rows(sparse.csr_matrix((1, 1)), np.ones(1), np.array([2.0 ** -1074]))
    assert tiny.residual_fraction(0) == Fraction(1, 2 ** 1074)
    assert tiny.residual_float()[0] == 2.0 ** -1074


def test_sqrt_hi_is_an_upper_bound_and_exact_on_squares():
    assert cc._sqrt_hi(16, 4) == 16
    assert cc._sqrt_hi(9, -2) == Fraction(3, 2)
    for value, exponent in [(2, 0), (3, -7), (10 ** 40 + 1, -300), (5, 1)]:
        root = cc._sqrt_hi(value, exponent)
        target = Fraction(value) * Fraction(2) ** exponent
        assert root * root >= target
        assert root * root <= target * (1 + Fraction(1, 2 ** 20)) or value < 2 ** 40


# --- Offset case, exemption, forward-error threshold ---------------------------------------------------

def test_offset_case_gives_the_exact_solution_with_zero_error():
    d = 2.0 ** -3
    S = _csr([[1., 0., 1.], [0., 1., -1.], [1., -1., 0.]])
    result = cc.certified_solve(S, np.array([1., 1., -d]), raw_load_zero=False)
    assert result.x.tolist() == [1 - d / 2, 1 + d / 2, d / 2]
    assert result.forward_error == 0


def test_exempt_rows_and_a_stored_zero_against_a_nonzero_x():
    # f = (1, -1), lambda = 1 (K = I, B = [1, 1]): accepted.
    S = _csr([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.]])
    result = cc.certified_solve(S, np.array([1., -1., 0.]), raw_load_zero=False)
    assert result.x.tolist() == [1., -1., 0.] or np.allclose(result.x, [1., -1., 0.])
    # A row whose every product is zero and b = 0 is exempt.
    exempt = cc._backward_check(cc._exact_rows(_csr([[0., 1.], [1., 0.]]), np.array([0., 3.]), np.array([3., 0.])))
    assert exempt['accepted'] and exempt['exempt_rows'] == 1
    # A stored zero b_i against nonzero products is not exempt and must meet the tolerance.
    rows = cc._exact_rows(_csr([[1., 1.]]), np.array([1., -1. + 1e-6]), np.zeros(1))
    check = cc._backward_check(rows)
    assert check['exempt_rows'] == 0 and check['accepted'] is False


def test_forward_error_exactly_at_the_threshold_is_accepted_and_the_next_dyadic_refused():
    x = np.array([3.0, -0.75])
    shifts = np.array([2, -1])
    scale = max(Fraction(3) / 4, Fraction(3, 4) * 2)
    threshold = cc.FORWARD_TOLERANCE * scale
    assert cc._forward_scale(x, shifts) == scale
    assert cc._forward_error_accepts(threshold, x, shifts) is True
    k = 2 ** 80
    above = Fraction(int(threshold * k) + 1, k)  # the next dyadic of that grid above the threshold
    assert above > threshold
    assert cc._forward_error_accepts(above, x, shifts) is False


def test_forward_error_above_the_limit_refuses_the_solve(monkeypatch):
    S = _csr([[2., 0.], [0., 3.]])
    monkeypatch.setattr(cc, '_candidate', lambda system, b: np.array([0.5 * (1 + 1e-3), 1. / 3.]))
    monkeypatch.setattr(cc, 'BACKWARD_TOLERANCE', Fraction(1))
    error = _refusal(lambda: cc.certified_solve(S, np.array([1., 1.]), raw_load_zero=False))
    assert (error.code, error.reason) == ('COUPLED_UNCERTIFIED', 'forward_error_above_limit')


def test_accepted_bound_contains_the_exact_solution():
    rng = np.random.default_rng(5)
    n = 30
    M = rng.standard_normal((n, n))
    K = M @ M.T + n * np.eye(n)
    S = sparse.csr_matrix(K)
    b = rng.standard_normal(n)
    result = cc.certified_solve(S, b, raw_load_zero=False)
    mpmath.mp.dps = 50
    exact = mpmath.lu_solve(mpmath.matrix(K.tolist()), mpmath.matrix(b.tolist()))
    for i in range(n):
        bound = result.bound(i)
        assert abs(mpmath.mpf(result.x[i]) - exact[i]) <= mpmath.mpf(bound.numerator) / bound.denominator + \
            mpmath.mpf(10) ** -45


# --- Zero-load path -------------------------------------------------------------------------------------

def test_zero_load_nonsingular_gives_zero_and_a_mechanism_refuses(monkeypatch):
    monkeypatch.setattr(cc.sparse_linalg, 'splu', lambda *a: pytest.fail('candidate on the zero path'))
    result = cc.certified_solve(_csr(np.diag([1., 2.])), np.zeros(2), raw_load_zero=True)
    assert result.x.tolist() == [0., 0.] and result.forward_error == 0 and result.zero_load
    mechanism = _refusal(lambda: cc.certified_solve(_csr([[1., -1.], [-1., 1.]]), np.zeros(2),
                                                    raw_load_zero=True))
    assert mechanism.code == 'COUPLED_UNCERTIFIED'
    assert mechanism.reason in ('singular_within_svd_error', 'condition_above_limit')
    zero_row = _refusal(lambda: cc.certified_solve(_csr([[1., 0.], [0., 0.]]), np.zeros(2), raw_load_zero=True))
    assert zero_row.reason == 'zero_row'


def test_raw_load_stored_as_zero_is_formation_loss():
    error = _refusal(lambda: cc.certified_solve(_csr(np.eye(2)), np.zeros(2), raw_load_zero=False))
    assert (error.code, error.reason) == ('COUPLED_FORMATION_LOSS', 'load_stored_as_zero')
    inconsistent = _refusal(lambda: cc.certified_solve(_csr(np.eye(2)), np.ones(2), raw_load_zero=True))
    assert inconsistent.code == 'COUPLED_UNCERTIFIED'


# --- Input type, errstate -------------------------------------------------------------------------------

@pytest.mark.parametrize('system', [np.eye(2), sparse.csr_matrix(np.eye(2, dtype=np.float32)),
                                    sparse.csc_matrix(np.eye(2)), sparse.csr_array(np.eye(2))])
def test_system_type_is_exact(system):
    assert _refusal(lambda: cc._certify_solve_matrix(system)).code == 'COUPLED_UNCERTIFIED'


@pytest.mark.parametrize('mode', ['raise', 'ignore'])
def test_certified_solve_is_independent_of_the_caller_errstate(mode):
    S = _csr([[1e10, 1e-306, 0.], [1e-306, 1., 0.], [0., 0., 1.]])
    b = np.array([1., 2., 3.])
    reference = cc.certified_solve(S, b, raw_load_zero=False)
    with np.errstate(all=mode):
        again = cc.certified_solve(S, b, raw_load_zero=False)
    assert again.x.tobytes() == reference.x.tobytes() and again.forward_error == reference.forward_error


def test_module_is_not_exported():
    import anyfesolver
    assert not hasattr(anyfesolver, 'certified_solve')
    assert 'certified_solve' not in getattr(anyfesolver, '__all__', ())
