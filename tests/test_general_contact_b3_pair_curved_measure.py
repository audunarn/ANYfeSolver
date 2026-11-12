"""Opt-in curved reference arc measure for distributed Q2 pair integration."""

import numpy as np
import pytest

from anyfesolver.general_contact.beam_curve_pair import B3PairPotential


def _reference(y_shift):
    return np.array(((-1.0, y_shift, 0.0),
                     (0.0, 0.2+y_shift, 0.0),
                     (1.0, y_shift, 0.0)))


def _current(y_shift):
    return np.array(((-1.0, y_shift, 0.0),
                     (0.0, y_shift, 0.0),
                     (1.0, y_shift, 0.0)))


def test_parallel_constant_gap_uses_independent_curved_arc_length():
    with pytest.raises(ValueError, match="midpoint must be straight"):
        B3PairPotential(_reference(0.0), _reference(0.15), 0.2, 20.0,
                        integration_relative_tolerance=1e-7,
                        maximum_subdivisions=8)
    law = B3PairPotential(
        _reference(0.0), _reference(0.15), 0.2, 20.0,
        integration_relative_tolerance=1e-7,
        maximum_subdivisions=8,
        reference_arc_measure="quadratic",
    )
    state = law.evaluate(_current(0.0), _current(0.15))
    points, weights = np.polynomial.legendre.leggauss(128)
    t = 0.5*(points+1.0)
    length = 0.5*np.sum(weights*np.sqrt(4.0+(0.8-1.6*t)**2))
    expected = 0.5*20.0*0.05**2*length
    assert state.energy == pytest.approx(expected, rel=1e-8)
    direction = np.zeros(18)
    direction[1] = direction[4] = direction[7] = 1.0
    step = 1e-6
    moved = _current(0.0)
    moved[:, 1] += step
    upper = law.evaluate(moved, _current(0.15)).energy
    moved[:, 1] -= 2.0*step
    lower = law.evaluate(moved, _current(0.15)).energy
    assert (upper-lower)/(2.0*step) == pytest.approx(
        state.residual@direction, rel=1e-6,
    )
