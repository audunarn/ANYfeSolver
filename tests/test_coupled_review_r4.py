"""Coupled route: regression tests for Codex code review round 4 (1 P1 + 4 P2, owner decision OD12;
the SVD-window findings are moot under OD13).
Each test names its finding; the fails-on-6ac863fa0 evidence is in the commit message."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('skfem')
pytest.importorskip('anysolid')

from anyfesolver import _coupled_certificate as cc  # noqa: E402
from test_coupled_review_r1 import small, solve  # noqa: E402
from test_coupled_review_r2 import rounding  # noqa: E402

# The r4 P1 (seal) and P2 (bounded coordinator wait, coordinator errors) tests were removed with those
# mechanisms by owner decision OD13: the certificate SVD runs in a child process (test_coupled_review_r5.py).


# --- P2: no truthiness (or other implicit protocol) call on the caller's clock --------------------------------

class _ProtocolClock:
    """A valid clock whose implicit protocol methods record (and switch the rounding mode, as in the
    reviewer's schedule) if they are ever called."""

    def __init__(self, stack):
        self.stack, self.protocol, self.calls = stack, [], 0

    def __call__(self):
        self.calls += 1
        return 0.0

    def _record(self, name):
        self.protocol.append(name)
        self.stack.enter_context(rounding(0x100))

    def __bool__(self):
        self._record('__bool__')
        return False

    def __len__(self):
        self._record('__len__')
        return 0

    def __eq__(self, other):
        self._record('__eq__')
        return False

    __hash__ = object.__hash__

    def __index__(self):
        self._record('__index__')
        return 0


def test_r4_p2_route_never_calls_the_clock_truthiness():
    stack = contextlib.ExitStack()
    clock = _ProtocolClock(stack)
    model, grid, removed, prepared, case, box = small()
    try:
        result = solve(model, prepared, removed, box, case, clock=clock)
    finally:
        stack.close()
    assert clock.protocol == []
    assert clock.calls > 0, 'the caller clock was not used'
    assert result.timings['total'] == 0.0


def test_r4_p2_certified_solve_never_calls_the_clock_truthiness():
    stack = contextlib.ExitStack()
    clock = _ProtocolClock(stack)
    S = sparse.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    try:
        solved = cc.certified_solve(S, np.array([1.0, 2.0]), raw_load_zero=False, clock=clock)
    finally:
        stack.close()
    assert clock.protocol == []
    assert clock.calls > 0
    assert set(solved.timings.values()) == {0.0}


@pytest.mark.parametrize('flag', [np.True_, 1, None], ids=['numpy_bool', 'int', 'none'])
def test_r4_p2_certified_solve_raw_load_zero_must_be_a_bool(flag):
    """Class search: certified_solve's raw_load_zero is decided by truthiness; only an exact bool is admitted."""
    S = sparse.csr_matrix(np.eye(2))
    with pytest.raises(cc.CoupledSolveError) as info:
        cc.certified_solve(S, np.zeros(2), raw_load_zero=flag)
    assert (info.value.code, info.value.reason) == ('COUPLED_UNCERTIFIED', 'invalid_arguments')


# --- P2: the oracle at exactly kappa = 1e7 ---------------------------------------------------------------

def test_r4_oracle_raises_inside_the_accuracy_band_at_the_cutoff():
    """S = [[10000001, 9999999], [9999999, 10000001]], b = S[:, 0]: both Ruiz shifts are equal, kappa = 1e7
    exactly, x = (1, 0) has zero residual. The computed kappa lies within the oracle's accuracy of the
    cutoff, so the oracle must raise (undecided), not return False."""
    from fractions import Fraction
    import test_coupled_r11_classes as r11
    S = sparse.csr_matrix(np.array([[10000001.0, 9999999.0], [9999999.0, 10000001.0]]))
    b = np.ascontiguousarray(S.toarray()[:, 0])
    shifts = cc._certify_solve_matrix(S)['shifts']
    assert shifts[0] == shifts[1]
    assert (Fraction(10000001) + 9999999) / (Fraction(10000001) - 9999999) == 10 ** 7
    with pytest.raises(AssertionError, match='accuracy band'):
        r11.acceptance_warranted(S, b)
