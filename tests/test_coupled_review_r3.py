"""Coupled route: regression tests for Codex code review round 3 (1 P1 + 2 P2, owner decision OD11).
Each test names its finding; the fails-on-c80fefc0b evidence is in the commit message."""

from __future__ import annotations

import contextlib
import inspect
import linecache
from fractions import Fraction

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('skfem')
pytest.importorskip('anysolid')

from anyfesolver import _coupled_certificate as cc  # noqa: E402
from anyfesolver.boundary import LoadCase  # noqa: E402
from test_coupled_review_r1 import refused, small, solve  # noqa: E402
from test_coupled_review_r2 import rounding  # noqa: E402


# --- P1 (moot under OD13, the SVD runs in a child process): an outer scope neither deadlocks nor leaks ---

def test_r3_p1_svd_inside_an_outer_scope_on_the_same_thread():
    """Same-thread nesting stays reentrant: an outer native_thread_scope(4) or default scope on the
    calling thread neither deadlocks nor leaks the one-thread limit."""
    from threadpoolctl import threadpool_info
    from anyfesolver.threading_policy import native_thread_scope
    model, grid, removed, prepared, case, box = small()
    reference = solve(model, prepared, removed, box, case)
    for outer in (4, None):
        with native_thread_scope(outer, phase='r3-outer'):
            before = [p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas']
            result = solve(model, prepared, removed, box, case)
            after = [p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas']
        assert before == after
        assert np.array_equal(result.shell_displacements.values, reference.shell_displacements.values)


# --- P2: the FP environment after every caller callback -----------------------------------------------------

def _caller_frame():
    frame = inspect.currentframe().f_back.f_back
    while frame is not None and frame.f_code.co_name == 'tick':      # the route's checked-clock wrapper
        frame = frame.f_back
    return frame


@pytest.mark.parametrize('function, next_line', [('_solve', 'prepared.assemble_linear()'),
                                                  ('candidate', '_held_candidate(system, b, held)')],
                         ids=['formation', 'candidate'])
def test_r3_p2_clock_switches_rounding_for_one_stage_refuses(function, next_line):
    """The reviewer's schedule: the clock selects downward rounding immediately before a stage
    (solid assembly and formation, or the candidate) and restores nearest-even at the next call."""
    stack = contextlib.ExitStack()
    state = {'armed': False, 'fired': False}

    def clock():
        frame = _caller_frame()
        if state['armed']:
            stack.close()
            state['armed'] = False
        elif not state['fired'] and frame.f_code.co_name == function and \
                next_line in linecache.getline(frame.f_code.co_filename, frame.f_lineno + 1):
            stack.enter_context(rounding(0x100))
            state.update(armed=True, fired=True)
        return 0.0
    model, grid, removed, prepared, case, box = small()
    try:
        refused('COUPLED_UNCERTIFIED', solve, model, prepared, removed, box, case, clock=clock, reason='rounding_mode')
    finally:
        stack.close()
    assert state['fired'], 'the stage was not reached'


@pytest.mark.parametrize('value', [np.float64(0.0), Fraction(0), True, None], ids=['float64', 'fraction', 'bool', 'none'])
def test_r3_p2_clock_value_must_be_an_exact_float_or_int(value):
    model, grid, removed, prepared, case, box = small()
    refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case, clock=lambda: value,
            reason='clock_invalid')


def test_r3_p2_integer_clock_is_admitted():
    model, grid, removed, prepared, case, box = small()
    reference = solve(model, prepared, removed, box, case)
    ticks = iter(range(10 ** 6))
    result = solve(model, prepared, removed, box, case, clock=lambda: next(ticks))
    assert np.array_equal(result.shell_displacements.values, reference.shell_displacements.values)


def test_r3_p2_resources_callback_switching_rounding_refuses():
    """Control (also refused on c80fefc0b, at the next poll): the resources attribute is caller code."""
    stack = contextlib.ExitStack()

    class Resources:
        @property
        def memory_limit_bytes(self):
            stack.enter_context(rounding(0x100))
            return None
    model, grid, removed, prepared, case, box = small()
    try:
        refused('COUPLED_UNCERTIFIED', solve, model, prepared, removed, box, case, resources=Resources(),
                reason='rounding_mode')
    finally:
        stack.close()


def test_r3_p2_removed_ids_iteration_switching_rounding_refuses():
    """Control (also refused on c80fefc0b, at the next poll): the removed-id iteration is caller code."""
    stack = contextlib.ExitStack()
    model, grid, removed, prepared, case, box = small()

    def ids():
        yield from removed
        stack.enter_context(rounding(0x100))
    try:
        refused('COUPLED_UNCERTIFIED', solve, model, prepared, ids(), box, case, reason='rounding_mode')
    finally:
        stack.close()


# --- P2: the S2 oracle uses high precision and the exact cutoff 1e7 -------------------------------------------

def test_r3_oracle_codex_counterexample_is_acceptance_required():
    """S = [[1, r], [r, 1]], r = float(9999499/9999501), b = S[:, 0]: Ruiz leaves S unchanged, x = (1, 0)
    has an exactly zero residual and kappa = (1 + r)/(1 - r) = 9999500.0016 < 1e7 (the eigenvalues
    are exactly 1 +- r). The oracle must require acceptance, and the route accepts."""
    import mpmath
    import test_coupled_r11_classes as r11
    r = float(9_999_499 / 9_999_501)
    S = sparse.csr_matrix(np.array([[1.0, r], [r, 1.0]]))
    b = np.ascontiguousarray(S.toarray()[:, 0])
    production = cc._certify_solve_matrix(S)
    assert not production['shifts'].any()
    exact = (1 + Fraction(r)) / (1 - Fraction(r))
    assert 9_999_500 < exact < 10 ** 7
    warranted, x = r11.acceptance_warranted(S, b)
    assert warranted
    assert np.all(np.abs(x - [1.0, 0.0]) <= 1e-30)
    kappa = r11.independent_condition(S)
    with mpmath.workdps(50):
        assert abs(mpmath.mpf(kappa) - mpmath.mpf(exact.numerator) / exact.denominator) <= mpmath.mpf(10) ** -20 * exact
    solved = cc.certified_solve(S, b, raw_load_zero=False)
    assert np.all(np.abs(solved.x - [1.0, 0.0]) <= [float(solved.bound(i)) for i in range(2)])


def test_r3_oracle_exact_cutoff_on_both_sides():
    """kappa just above 1e7 is not acceptance-required; just below it is (no margin)."""
    import test_coupled_r11_classes as r11
    for target, expected in ((10 ** 7 * (1 - 1e-8), True), (10 ** 7 * (1 + 1e-8), False)):
        r = (target - 1) / (target + 1)
        S = sparse.csr_matrix(np.array([[1.0, r], [r, 1.0]]))
        exact = (1 + Fraction(r)) / (1 - Fraction(r))
        assert (exact <= 10 ** 7) is expected
        b = np.ascontiguousarray(S.toarray()[:, 0])                  # x = (1, 0) exactly: the margin holds
        assert r11.acceptance_warranted(S, b)[0] is expected


def test_r3_oracle_matches_on_the_fixture():
    """On the S2 acceptance fixture, the high-precision kappa agrees with binary64 to the SVD's accuracy."""
    import test_coupled_r11_classes as r11
    from anyfesolver import solid_coupling as sc
    captured = {}
    original = sc.certified_solve

    def spy(system, rhs, **kwargs):
        captured.update(S=system.copy())
        return original(system, rhs, **kwargs)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sc, 'certified_solve', spy)
        model, grid, removed, prepared, case, box = small()
        solve(model, prepared, removed, box, case)
    kappa = r11.independent_condition(captured['S'])
    production = cc._certify_solve_matrix(captured['S'])
    assert abs(float(kappa) / production['condition_number'] - 1) < 1e-6
    assert kappa <= 10 ** 7
