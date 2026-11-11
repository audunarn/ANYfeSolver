"""Independent checks for arbitrary nodal motion of two S3 skin faces."""

import numpy as np
import pytest

from anyfesolver.general_contact.triangle_pair_energy import MovingS3PairPenalty


_SOURCE = np.array(((0.4, 0.2, 0.02), (0.9, 0.2, 0.02),
                    (0.4, 0.7, 0.02)))
_TARGET = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0)))


def _law(*, difference_step=1e-5, budget=1500):
    return MovingS3PairPenalty(
        _SOURCE, _TARGET, 1000.0,
        difference_step=difference_step,
        maximum_energy_evaluations=budget,
    )


def _independent_constant_gap_energy(source):
    """Analytic one-corner overlap against x+y<=1, no production clipping."""

    first, second, third = source[:, :2]
    signed = 1.0 - np.sum(source[:, :2], axis=1)
    assert signed[0] > 0.0 and signed[1] < 0.0 and signed[2] < 0.0
    along_second = first + signed[0] / (signed[0] - signed[1]) * (second - first)
    along_third = first + signed[0] / (signed[0] - signed[2]) * (third - first)
    overlap = 0.5 * abs(np.linalg.det(np.stack((along_second - first,
                                                along_third - first))))
    current = 0.5 * abs(np.linalg.det(np.stack((second - first,
                                                third - first))))
    reference = 0.125
    gap = float(source[0, 2])
    return 0.5 * 1000.0 * reference / current * overlap * gap * gap


def test_s3_pair_exact_reference_measure_opening_and_recontact():
    law = _law()
    closed = _SOURCE.copy()
    closed[:, 2] = -0.01
    state = law.evaluate(closed, _TARGET)
    assert state.overlap_area == pytest.approx(0.08, abs=1e-13)
    assert state.active_area == pytest.approx(0.08, abs=1e-13)
    assert state.minimum_gap == pytest.approx(-0.01, abs=1e-13)
    assert state.energy == pytest.approx(0.004, rel=1e-12)
    opened = _SOURCE.copy()
    assert law.evaluate(opened, _TARGET).energy == 0.0
    assert law.evaluate(closed, _TARGET).energy == pytest.approx(state.energy)


def test_s3_pair_partial_active_triangle_matches_analytic_affine_gap():
    reference = np.array(((0.2, 0.2, 0.02), (0.6, 0.2, 0.02),
                          (0.2, 0.6, 0.02)))
    law = MovingS3PairPenalty(
        reference, _TARGET, 1000.0, difference_step=1e-5,
    )
    source = reference.copy()
    source[:, 2] = (-0.01, 0.01, 0.01)
    state = law.evaluate(source, _TARGET)
    normal = np.cross(source[1] - source[0], source[2] - source[0])
    normal /= np.linalg.norm(normal)
    closure = 0.01 / normal[2]
    expected = 1000.0 * 0.08 * 0.25 * closure**2 / 12.0
    assert state.overlap_area == pytest.approx(
        0.08 / normal[2], rel=1e-12,
    )
    assert state.active_area == pytest.approx(
        state.overlap_area * 0.25, rel=1e-12,
    )
    assert state.energy == pytest.approx(expected, rel=1e-12)
    assert state.minimum_gap == pytest.approx(-closure, rel=1e-12)


def test_s3_pair_independent_nodal_slide_energy_work_and_rigid_balance():
    law = _law()
    source = _SOURCE.copy()
    source[1, 0] += 0.01
    source[2, 1] -= 0.015
    source[:, 2] = -0.01
    work = law.work(source, _TARGET, tangent=False)
    expected = _independent_constant_gap_energy(source)
    assert work.energy.energy == pytest.approx(expected, rel=1e-12)
    step = 2e-6
    plus, minus = source.copy(), source.copy()
    plus[1, 0] += step
    minus[1, 0] -= step
    independent_direction = (
        _independent_constant_gap_energy(plus)
        - _independent_constant_gap_energy(minus)
    ) / (2.0 * step)
    assert work.residual[3] == pytest.approx(
        independent_direction, rel=1e-5, abs=1e-7,
    )
    rows = work.residual.reshape(6, 3)
    np.testing.assert_allclose(np.sum(rows, axis=0), 0.0, atol=1e-8)
    positions = np.vstack((source, _TARGET))
    np.testing.assert_allclose(np.sum(np.cross(positions, rows), axis=0),
                               0.0, atol=1e-7)
    common = np.array((0.07, -0.03, 0.02))
    assert law.evaluate(source + common, _TARGET + common).energy == pytest.approx(
        work.energy.energy, rel=1e-12,
    )


def test_s3_pair_smooth_tangent_and_explicit_differentiation_budget():
    law = _law(difference_step=2e-5)
    source = _SOURCE.copy()
    source[1, 0] += 0.01
    source[2, 1] -= 0.015
    source[:, 2] = (-0.009, -0.012, -0.008)
    target = _TARGET.copy()
    target[1, 0] += 0.013
    work = law.work(source, target)
    assert work.tangent is not None
    np.testing.assert_allclose(work.tangent, work.tangent.T, atol=0.0)
    direction = np.array((
        (0.2, -0.3, 0.1), (-0.1, 0.2, -0.15),
        (0.15, -0.1, 0.2), (-0.2, 0.1, 0.07),
        (0.05, -0.15, -0.11), (-0.1, 0.2, 0.13),
    ))
    trial_step = 3e-5
    plus = np.vstack((source, target)) + trial_step * direction
    minus = np.vstack((source, target)) - trial_step * direction
    plus_work = law.work(plus[:3], plus[3:], tangent=False)
    minus_work = law.work(minus[:3], minus[3:], tangent=False)
    measured = (plus_work.residual - minus_work.residual) / (2*trial_step)
    relative_error = float(np.linalg.norm(
        measured - work.tangent @ direction.reshape(18)
    ) / np.linalg.norm(measured))
    assert relative_error <= 1e-5
    with pytest.raises(MemoryError, match="budget exhausted"):
        _law(budget=20).work(source, target, tangent=True)


def test_s3_pair_refuses_contact_and_coincident_edge_derivative_kinks():
    with pytest.raises(ValueError, match="contact branch"):
        _law().work(_SOURCE - np.array((0.0, 0.0, 0.02)), _TARGET,
                    tangent=False)
    reference = _TARGET + np.array((0.0, 0.0, 0.02))
    law = MovingS3PairPenalty(
        reference, _TARGET, 1000.0, difference_step=1e-5,
    )
    closed = _TARGET - np.array((0.0, 0.0, 0.01))
    with pytest.raises(ValueError, match="shared-edge"):
        law.work(closed, _TARGET, tangent=False)
