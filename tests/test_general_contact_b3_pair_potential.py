"""Symmetric distributed B3/B3 local potential checks."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from anyfesolver.general_contact.beam_curve_pair import (
    B3PairPotential, certify_b3_pair_gap,
)


_TARGET = np.array(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0),
                    (1.0, 0.0, 0.0)))
_SOURCE = _TARGET + np.array((0.0, 0.08, 0.0))


def test_b3_pair_parallel_lines_match_independent_energy_and_balanced_wrench():
    law = B3PairPotential(
        _SOURCE, _TARGET, 0.1, 1000.0,
        integration_relative_tolerance=1.0e-6,
        maximum_subdivisions=8,
    )
    state = law.evaluate(_SOURCE, _TARGET)
    # Independent constant-gap integral: 1/2*k*(R-d)^2*1 m.
    assert state.energy == pytest.approx(0.2, rel=1e-12)
    force_a = -np.sum(state.residual.reshape(6, 3)[:3], axis=0)
    force_b = -np.sum(state.residual.reshape(6, 3)[3:], axis=0)
    np.testing.assert_allclose(force_a, (0.0, 20.0, 0.0), atol=1e-10)
    np.testing.assert_allclose(force_b, -force_a, atol=1e-10)
    positions = np.vstack((_SOURCE, _TARGET))
    gradient_rows = state.residual.reshape(6, 3)
    assert np.linalg.norm(np.sum(np.cross(positions, gradient_rows), axis=0)) < 1e-10
    assert state.maximum_relative_integration_error <= 1e-6
    assert state.integrated_intervals > 0


def test_b3_pair_curved_potential_residual_and_tangent_are_consistent():
    source = _SOURCE.copy()
    source[1, 1] = 0.073
    target = _TARGET.copy()
    target[1, 2] = 0.006
    law = B3PairPotential(
        _SOURCE, _TARGET, 0.1, 1000.0,
        integration_relative_tolerance=2.0e-5,
        maximum_subdivisions=8,
    )
    state = law.evaluate(source, target)
    assert state.energy > 0.0

    # Independent high-accuracy scalar quadrature and bounded closest-point
    # minimization; this oracle does not call the production projection law.
    def point(nodes, station):
        shape = np.array(((1.0 - station) * (1.0 - 2.0 * station),
                          4.0 * station * (1.0 - station),
                          station * (2.0 * station - 1.0)))
        return shape @ nodes

    def direction_energy(first, second):
        def density(station):
            position = point(first, station)
            projection = minimize_scalar(
                lambda target_station: float(np.sum(
                    (position - point(second, target_station))**2
                )), bounds=(0.0, 1.0), method="bounded",
                options={"xatol": 1.0e-13},
            )
            gap = np.sqrt(projection.fun) - 0.1
            return 0.5 * 1000.0 * min(gap, 0.0)**2
        return quad(density, 0.0, 1.0, epsabs=1e-10,
                    epsrel=1e-10)[0]

    oracle_energy = 0.5 * (direction_energy(source, target)
                           + direction_energy(target, source))
    assert state.energy == pytest.approx(oracle_energy, rel=1e-5, abs=1e-8)
    direction = np.array((
        (0.01, -0.02, 0.03), (-0.015, 0.018, 0.011),
        (0.02, -0.01, -0.012), (0.008, 0.012, -0.017),
        (-0.014, 0.006, 0.018), (0.012, -0.009, 0.014),
    ))
    h = 1.0e-6
    plus = law.evaluate(source + h * direction[:3],
                        target + h * direction[3:])
    minus = law.evaluate(source - h * direction[:3],
                         target - h * direction[3:])
    assert (plus.energy - minus.energy) / (2 * h) == pytest.approx(
        float(state.residual @ direction.reshape(18)), rel=2e-5, abs=2e-7,
    )
    np.testing.assert_allclose(
        (plus.residual - minus.residual) / (2 * h),
        state.tangent @ direction.reshape(18), rtol=2e-4, atol=2e-3,
    )
    np.testing.assert_allclose(state.tangent, state.tangent.T,
                               rtol=0, atol=1e-8)
    exchanged = B3PairPotential(
        _TARGET, _SOURCE, 0.1, 1000.0,
        integration_relative_tolerance=2.0e-5,
        maximum_subdivisions=8,
    ).evaluate(target, source)
    assert exchanged.energy == pytest.approx(state.energy, rel=1e-12)
    np.testing.assert_allclose(
        exchanged.residual, np.r_[state.residual[9:], state.residual[:9]],
        rtol=1e-12, atol=1e-10,
    )


def test_b3_pair_open_state_and_explicit_integration_limit():
    law = B3PairPotential(
        _SOURCE, _TARGET, 0.1, 1000.0,
        integration_relative_tolerance=1.0e-6,
        maximum_subdivisions=8,
    )
    separated = _SOURCE + np.array((0.0, 0.3, 0.0))
    state = law.evaluate(separated, _TARGET)
    assert state.energy == 0.0
    np.testing.assert_array_equal(state.residual, 0.0)
    with pytest.raises(ValueError, match="midpoint"):
        B3PairPotential(
            _SOURCE + np.array(((0.0, 0.0, 0.0),
                                (0.0, 0.01, 0.0),
                                (0.0, 0.0, 0.0))), _TARGET,
            0.1, 1000.0, integration_relative_tolerance=1e-6,
            maximum_subdivisions=8,
        )


def test_b3_pair_gap_certificate_contains_independent_curve_grid():
    source = _SOURCE.copy()
    source[1, 1] = 0.073
    target = _TARGET.copy()
    target[1, 2] = 0.006
    certificate = certify_b3_pair_gap(
        source, target, 0.1, accuracy=1e-4,
        maximum_cells=10000, maximum_depth=25,
    )
    parameters = np.linspace(0.0, 1.0, 201)
    shape = np.column_stack(((1 - parameters) * (1 - 2 * parameters),
                             4 * parameters * (1 - parameters),
                             parameters * (2 * parameters - 1)))
    first = shape @ source
    second = shape @ target
    independent = float(np.min(np.linalg.norm(
        first[:, None, :] - second[None, :, :], axis=2,
    )) - 0.1)
    assert certificate.conservative_lower_gap <= independent + 1e-12
    assert certificate.minimum_sampled_gap >= independent - 1e-5
    assert (certificate.minimum_sampled_gap
            - certificate.conservative_lower_gap) <= 1e-4 + 1e-10
    with pytest.raises(MemoryError, match="gap .* budget"):
        certify_b3_pair_gap(
            source, target, 0.1, accuracy=1e-12,
            maximum_cells=1, maximum_depth=0,
        )


def test_b3_pair_narrow_double_pocket_independent_energy_gate():
    # A quadratic parabola crosses a 5 mm tube in two narrow station pockets.
    # The first coarse scan cell contains three gap roots. The original scan
    # stopped after the first and missed the second pocket. Keep this frozen
    # oracle as a regression gate for the straight-target root partition.
    centre = 0.5014

    def position(station):
        u = station - centre
        return (u, 5000.0*u*u - 0.01, 0.0)

    source = np.array([position(s) for s in (0.0, 0.5, 1.0)])
    target = np.array(((0.0, 0.0, -0.1), (0.0, 0.0, 0.0),
                       (0.0, 0.0, 0.1)))
    reference = np.array(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0),
                          (1.0, 0.0, 0.0)))
    law = B3PairPotential(
        reference, target, 0.005, 1.0,
        integration_relative_tolerance=1e-3,
        maximum_subdivisions=16,
    )
    measured = law._direction(source, target, 1.0, True)[0]

    def density(station):
        u = station - centre
        distance = np.hypot(u, 5000.0*u*u - 0.01)
        return 0.5 * min(distance - 0.005, 0.0)**2

    independent = sum(quad(
        density, left, right, epsabs=1e-18, epsrel=1e-10,
        points=[0.5, 0.5028], limit=100,
    )[0] for left, right in ((0.49, centre), (centre, 0.52)))
    assert measured == pytest.approx(independent, rel=1e-4, abs=1e-15)


def test_b3_pair_straight_target_endpoint_to_interior_transition():
    source = np.array(((-0.2, 0.08, 0.0), (0.3, 0.08, 0.0),
                       (0.8, 0.08, 0.0)))
    target = _TARGET.copy()
    law = B3PairPotential(
        source, target, 0.1, 1000.0,
        integration_relative_tolerance=1e-6,
        maximum_subdivisions=8,
    )
    measured = law._direction(source, target, 1.0, True)[0]

    def endpoint_density(station):
        distance = np.hypot(station - 0.2, 0.08)
        return 500.0 * min(distance - 0.1, 0.0)**2

    independent = (quad(endpoint_density, 0.14, 0.2, epsabs=1e-13)[0]
                   + 500.0 * 0.02**2 * 0.8)
    assert measured == pytest.approx(independent, rel=1e-8)
