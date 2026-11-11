"""Focused exact-rational checks for the shared Hermite subdivision kernel.

The accelerated _split must be pointwise identical to an independent
Fraction-arithmetic de Casteljau oracle on every coefficient class the
sweep can produce: arbitrary rational denominators (including the 3 from
Hermite endpoint interpolation), high-bit dyadic values, shared
denominators, zeros and negatives. Refusal, singleton, empty and
non-Fraction behavior must stay unchanged.
"""
import random

import numpy as np
import pytest

from fractions import Fraction

from anyfesolver.general_contact.shell_pair_hermite_sweep import (
    _split, shell_pair_hermite_lower)


def oracle_split(coefficients):
    """Independent original Fraction midpoint de Casteljau subdivision."""
    left, right = [coefficients[0]], [coefficients[-1]]
    row = coefficients
    while len(row) > 1:
        row = tuple((x+y)/2 for x, y in zip(row, row[1:]))
        left.append(row[0])
        right.append(row[-1])
    return tuple(left), tuple(reversed(right))


def evaluate(coefficients, t):
    row = list(coefficients)
    while len(row) > 1:
        row = [(1-t)*x + t*y for x, y in zip(row, row[1:])]
    return row[0]


def coefficient_cases():
    random.seed(20261012)
    return [
        # Arbitrary rational denominators, mixed signs, degree 12.
        tuple(Fraction(random.randint(-10**6, 10**6), random.randint(1, 10**6))
              for _ in range(13)),
        # High-bit dyadic numerators and denominators.
        tuple(Fraction(random.randint(-2**200, 2**200), 2**random.randint(0, 300))
              for _ in range(10)),
        # Smallest positive dyadic and multiples (denominator 2**1074).
        tuple(Fraction(2**-1074)*k for k in range(9)),
        # Shared odd denominators.
        tuple(Fraction(1, 7)*k for k in range(1, 12)),
        # Shared denominators including the Hermite factor 3.
        tuple(Fraction(k, 60) for k in range(15)),
        # Powers reduced modulo another rational.
        tuple(Fraction(22, 7)**k % Fraction(99, 13) for k in range(7)),
        # All zeros and a lone zero.
        (Fraction(0),)*9,
        (Fraction(0),),
        # Integers only (denominator 1 fast path).
        tuple(Fraction(k**3-7, 1) for k in range(11)),
        # Linear pair.
        (Fraction(-1, 3), Fraction(5, 6)),
    ]


@pytest.mark.parametrize('coefficients', coefficient_cases())
def test_split_matches_independent_oracle_exactly(coefficients):
    assert _split(coefficients) == oracle_split(coefficients)
    left, right = _split(coefficients)
    for value in left + right:
        assert type(value) is Fraction
        assert value.denominator > 0


def test_split_recursive_tree_matches_oracle():
    """Full subdivision tree parity on a realistic sweep workload."""
    random.seed(4)
    a0 = np.array([[0., 0., 0.], [1., 0., 0.03], [0., 1., -0.02]])
    a1 = a0 + np.array([[.01, .02, .05], [.02, -.01, .08], [.03, .01, 0.]])
    b0 = np.array([[.2, .2, .5], [1.2, .2, .5], [.2, 1.2, .5], [1.2, 1.2, .5]])
    b1 = b0 + np.array([[.05, .1, -.05], [-.05, .05, 0.], [.1, -.05, .05], [-.1, -.1, -.1]])
    data = dict(a0=a0, a1=a1, b0=b0, b1=b1,
                source_v0=np.full((3, 3), .01), source_v1=np.full((3, 3), -.01),
                target_v0=np.full((4, 3), .02), target_v1=np.full((4, 3), -.02),
                dt=0.013)
    # Build the same flat polynomial tuple the sweep subdivides.
    from anyfesolver.general_contact import shell_pair_hermite_sweep as module
    coords = lambda v: tuple(tuple(Fraction(float(x)) for x in row) for row in v)
    a = module._nodes(coords(a0), coords(a1), coords(data['source_v0']),
                      coords(data['source_v1']), Fraction(0.013))
    b = module._nodes(coords(b0), coords(b1), coords(data['target_v0']),
                     coords(data['target_v1']), Fraction(0.013))
    normal = module._normal(a, 0)
    constraints = [module._dot(normal, module._normal(
        tuple(tuple((v,) for v in p) for p in coords(a0)), 0))]
    for corner in range(4):
        target_normal = module._normal(b, corner)
        constraints.extend((module._dot(target_normal, module._normal(
            tuple(tuple((v,) for v in p) for p in coords(b0)), corner)),
            module._dot(normal, target_normal)))
    numerators = tuple(module._dot(
        tuple(module._sub(x, y) for x, y in zip(ai, bj)), normal)
        for ai in a for bj in b)
    polys = tuple(constraints) + tuple(normal) + numerators
    assert sorted({len(p) for p in polys}) == [7, 10, 13]
    level = [polys]
    for _ in range(6):
        nxt = []
        for flat in level:
            pairs = [_split(p) for p in flat]
            oracle_pairs = [oracle_split(p) for p in flat]
            assert pairs == oracle_pairs
            nxt.append(tuple(q[0] for q in pairs))
            nxt.append(tuple(q[1] for q in pairs))
        level = nxt


def test_split_reversal_identity():
    coefficients = coefficient_cases()[0]
    left, right = _split(coefficients)
    reversed_left, reversed_right = _split(tuple(reversed(coefficients)))
    assert left == tuple(reversed(reversed_right))
    assert right == tuple(reversed(reversed_left))


@pytest.mark.parametrize('t', [Fraction(0), Fraction(1, 8), Fraction(1, 4),
                              Fraction(1, 2), Fraction(3, 4), Fraction(1)])
def test_split_subdivision_identity(t):
    """Children reproduce the parent polynomial on their subranges exactly."""
    coefficients = coefficient_cases()[4]
    left, right = _split(coefficients)
    if t <= Fraction(1, 2):
        assert evaluate(left, 2*t) == evaluate(coefficients, t)
    if t >= Fraction(1, 2):
        assert evaluate(right, 2*t-1) == evaluate(coefficients, t)


def test_split_empty_refusal_is_unchanged():
    with pytest.raises(IndexError):
        _split(())


def test_split_singleton_is_identity():
    value = Fraction(5, 3)
    assert _split((value,)) == ((value,), (value,))


def test_split_non_fraction_inputs_keep_generic_arithmetic():
    integers = (1, 2, 3, 4)
    assert _split(integers) == oracle_split(integers)
    floats = (0.5, 1.5, 2.5)
    assert _split(floats) == oracle_split(floats)
    mixed = (Fraction(1, 3), 2, Fraction(1, 5))
    assert _split(mixed) == oracle_split(mixed)


def subdividing_path():
    # g(t)=1-3.6t+3.6t²: positive minimum .1, negative whole-panel
    # Bernstein controls require subdivision to certify opening.
    triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    target = triangle.copy()
    target[:, 2] -= 1.
    data = dict(a0=triangle.copy(), a1=triangle.copy(), b0=target.copy(),
                b1=target.copy(), source_v0=np.zeros((3, 3)),
                source_v1=np.zeros((3, 3)), target_v0=np.zeros((3, 3)),
                target_v1=np.zeros((3, 3)), dt=1.)
    data['target_v0'][:, 2] = 3.6
    data['target_v1'][:, 2] = -3.6
    return data


def test_sweep_result_is_unchanged_on_subdividing_case(monkeypatch):
    """Compare the complete caller certificate with original arithmetic."""
    import anyfesolver.general_contact.shell_pair_hermite_sweep as sweep
    data = subdividing_path()
    result = sweep.shell_pair_hermite_lower(gap_tolerance_m=0., **data)
    assert result.success and result.visited_panels > 1
    assert 0. < result.lower_gap <= .1
    monkeypatch.setattr(sweep, '_split', oracle_split)
    assert sweep.shell_pair_hermite_lower(gap_tolerance_m=0., **data) == result


def test_sweep_cancellation_and_budget_refusals_are_unchanged(monkeypatch):
    import anyfesolver.general_contact.shell_pair_hermite_sweep as sweep
    data = subdividing_path()
    cancelled = sweep.shell_pair_hermite_lower(cancel=lambda: True, **data)
    assert cancelled.status == 'cancelled' and not cancelled.success
    exhausted = sweep.shell_pair_hermite_lower(
        maximum_panels=1, gap_tolerance_m=0., **data)
    assert exhausted.status == 'panel_budget_exhausted'
    assert not exhausted.success
    monkeypatch.setattr(sweep, '_split', oracle_split)
    assert sweep.shell_pair_hermite_lower(cancel=lambda: True, **data) == cancelled
    assert sweep.shell_pair_hermite_lower(
        maximum_panels=1, gap_tolerance_m=0., **data) == exhausted


def test_split_unsupported_iterator_refuses_before_consumption():
    touched = []
    def coefficients():
        touched.append(True)
        yield Fraction(1, 3)
    with pytest.raises(TypeError):
        _split(coefficients())
    assert not touched
