"""First-order isolated seam admission; strict Hessian profiles stay strict."""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.general_contact.curved_s3_q4_work import INTEGRATION_CONTROLS
from anyfesolver.general_contact.overlap import PlanarFace
from anyfesolver.general_contact.q4_target_overlap import q4_target_overlap_integrals
from anyfesolver.general_contact.q4_s3_overlap_derivatives import UnsupportedQ4S3Branch
from anyfesolver.general_contact.s3_q4_overlap_derivatives import (
    s3_q4_overlap_derivatives, _s3_q4_first_order_work,
)


def _pair(name):
    if name == "actual":
        record = json.loads((Path(__file__).parent / "fixtures/s9_seam_pair.json").read_text())
        return tuple(np.array(record[key], dtype=float) for key in ("reference", "source", "target"))
    source = np.array(((0., 0., .01), (.6, -.4, .01), (.6, .4, .01)))
    if name == "positive_offset":
        source[0, 1] = 1e-12
    elif name == "negative_offset":
        source[0, 1] = -1e-12
    target = np.array(((-1., 0., 0.), (1., 0., 0.), (1., 1., 0.), (-1., 1., 0.)))
    return source.copy(), source, target


@pytest.mark.parametrize("name", ("exact", "positive_offset", "negative_offset"))
def test_scalar_and_work_values_against_rational_triangle_moments(name):
    """One fixed oracle: explicit clipped triangle/complement P1 moments."""
    reference, source, target = _pair(name)
    length, height, delta = map(Fraction.from_float,
        (float(source[1, 0]), float(source[2, 1]), float(source[0, 1])))
    half = Fraction(1, 2)
    area = length*height
    if delta >= 0:
        denominator = height+delta
        outside = length*height*height/(2*denominator)
        moments = (area/3-outside*(height/denominator)/3,
                   area/3-outside*(3*half+delta/denominator)/3,
                   area/3-outside*half/3)
    else:
        denominator = height-delta
        inside = length*height*height/(2*denominator)
        moments = (inside*(height/denominator)/3, inside*half/3,
                   inside*(3*half-delta/denominator)/3)
    support = np.array(list(map(float, moments)))
    gap = Fraction.from_float(float(source[0, 2]))
    weighted = np.array([float(value*gap) for value in moments])
    scalar = q4_target_overlap_integrals(PlanarFace((1, 1), source), target,
        reference_area_ratio=1., weighted_gap_tolerance=1e-13, maximum_triangles=200)
    work = _s3_q4_first_order_work(reference, source, target, **INTEGRATION_CONTROLS)
    for actual_support, actual_weighted in ((scalar.support, scalar.weighted_gap),
            (work.reference_support, work.reference_weighted_gap)):
        np.testing.assert_allclose(actual_support, support, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(actual_weighted, weighted, rtol=1e-11, atol=1e-16)


@pytest.mark.parametrize("name", ("exact", "positive_offset", "negative_offset", "actual"))
def test_value_gradient_and_strict_hessian_contract(name):
    reference, source, target = _pair(name)
    with pytest.raises(UnsupportedQ4S3Branch, match="clipping boundary"):
        s3_q4_overlap_derivatives(reference, source, target, **INTEGRATION_CONTROLS)
    result = _s3_q4_first_order_work(reference, source, target, **INTEGRATION_CONTROLS)
    assert result.derivative_order == 1 and result.seam_events
    assert not hasattr(result, "support_hessian")
    assert not hasattr(result, "weighted_gap_hessian")
    assert all(event["hessian_unique"] is False for event in result.seam_events)
    assert 0 < result.overlap_area_resolution_m2 < result.overlap_area
    assert all(error <= INTEGRATION_CONTROLS[key] for error, key in zip(result.measured_errors,
        ("support_tolerance", "weighted_gap_tolerance", "support_gradient_tolerance",
         "weighted_gap_gradient_tolerance", "support_hessian_tolerance", "weighted_gap_hessian_tolerance")))
    ratio = np.linalg.norm(np.cross(reference[1]-reference[0], reference[2]-reference[0])) / np.linalg.norm(np.cross(source[1]-source[0], source[2]-source[0]))
    scalar = q4_target_overlap_integrals(PlanarFace((1, 1), source), target,
        reference_area_ratio=ratio, weighted_gap_tolerance=1e-13, maximum_triangles=200)
    np.testing.assert_allclose(result.reference_support, scalar.support, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result.reference_weighted_gap, scalar.weighted_gap, rtol=1e-11, atol=1e-16)
    direction = np.random.default_rng(910).normal(size=21)
    direction /= np.linalg.norm(direction)
    coordinates = np.concatenate((source, target)).reshape(-1)
    for step in (1e-6, 3e-7):
        plus = (coordinates + step * direction).reshape(7, 3)
        minus = (coordinates - step * direction).reshape(7, 3)
        a = _s3_q4_first_order_work(reference, plus[:3], plus[3:], **INTEGRATION_CONTROLS)
        b = _s3_q4_first_order_work(reference, minus[:3], minus[3:], **INTEGRATION_CONTROLS)
        for value, gradient in (("reference_support", "support_gradient"),
                                ("reference_weighted_gap", "weighted_gap_gradient")):
            analytic = getattr(result, gradient) @ direction
            finite = (getattr(a, value) - getattr(b, value)) / (2 * step)
            assert np.linalg.norm(finite - analytic) / max(np.linalg.norm(analytic), 1e-12) <= 1e-5


@pytest.mark.parametrize("kind", ("collinear", "corner", "outside"))
def test_non_isolated_or_empty_branch_remains_refused(kind):
    reference, source, target = _pair("exact")
    if kind == "collinear":
        source = np.array(((0., 0., .01), (.6, 0., .01), (.2, .6, .01)))
    elif kind == "corner":
        source[:, 0] -= 1.
    else:
        source = np.array(((0., 0., .01), (-.6, -.4, .01), (.6, -.4, .01)))
    with pytest.raises(UnsupportedQ4S3Branch):
        _s3_q4_first_order_work(source.copy(), source, target, **INTEGRATION_CONTROLS)
