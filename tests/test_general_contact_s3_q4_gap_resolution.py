"""Independent scalar value controls; no contact solve or quadrature campaign."""
from decimal import Decimal, localcontext

import numpy as np
import pytest

from anyfesolver.general_contact.s3_overlap_derivatives import UnsupportedOverlapBranch
from anyfesolver.general_contact.s3_q4_gap_resolution import s3_q4_gap_operand_allowance


SOURCE = np.array(((.1, .1, .04), (.9, .1, .04), (.1, .9, .04)))
TARGET = np.array(((0., 0., .04000001), (1.1, .05, .04000002),
                   (1., 1.1, .040000035), (-.03, 1., .040000015)))
SUPPORT = np.array((.07, .11, .13))


def _shape(r, s):
    return .25*np.array(((1-r)*(1-s), (1+r)*(1-s),
                         (1+r)*(1+s), (1-r)*(1+s)))


def _decimal_gap(source, target, r, s):
    """80-digit original bilinear height and exact source cross product.

    Float inputs are converted exactly, not replaced by clean decimal geometry.
    This does not call production geometry, inverse, or contact work.
    """
    with localcontext() as context:
        context.prec = 80
        src = [[Decimal.from_float(float(x)) for x in row] for row in source]
        tgt = [[Decimal.from_float(float(x)) for x in row] for row in target]
        a = [src[1][i]-src[0][i] for i in range(3)]
        b = [src[2][i]-src[0][i] for i in range(3)]
        cross = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
        norm = sum(x*x for x in cross).sqrt()
        n = [x/norm for x in cross]
        r, s = Decimal.from_float(float(r)), Decimal.from_float(float(s))
        one, four = Decimal(1), Decimal(4)
        hats = [(one-r)*(one-s)/four, (one+r)*(one-s)/four,
                (one+r)*(one+s)/four, (one-r)*(one+s)/four]
        point = [sum(hats[j]*tgt[j][i] for j in range(4)) for i in range(3)]
        return sum((src[0][i]-point[i])*n[i] for i in range(3))


def _float_frame(source):
    edge = source[1]-source[0]
    normal = np.cross(edge, source[2]-source[0])
    normal /= np.linalg.norm(normal)
    tangent = edge/np.linalg.norm(edge)
    return normal, np.vstack((tangent, np.cross(normal, tangent)))


def _allow(source=SOURCE, target=TARGET, support=SUPPORT, **kwargs):
    inputs = dict(scalar_projection_residual_m=0., jet_projection_residual_m=0.,
                  natural_coordinate_bound=1.)
    inputs.update(kwargs)
    return s3_q4_gap_operand_allowance(source, target, support, **inputs)


def test_nonaffine_decimal_nearzero_and_measured_inverse_residual(record_property):
    normal, frame = _float_frame(SOURCE)
    xy = (TARGET-SOURCE[0]) @ frame.T
    discrepancies = []
    residual_max = 0.
    for r, s in ((-.6, -.4), (.15, -.55), (-.45, .2)):
        true_shape = _shape(r, s)
        point_xy = true_shape @ xy
        # Deliberately unequal returned roots, with measured forward errors.
        scalar_shape = _shape(r+2e-13, s-3e-13)
        jet_shape = _shape(r-1e-13, s+4e-13)
        scalar_residual = float(np.linalg.norm(scalar_shape @ xy-point_xy))
        jet_residual = float(np.linalg.norm(jet_shape @ xy-point_xy))
        source_point = SOURCE[0]+point_xy @ frame
        scalar_gap = float((source_point-scalar_shape @ TARGET) @ normal)
        jet_gap = float((SOURCE[0]-jet_shape @ TARGET) @ normal)
        exact_gap = _decimal_gap(SOURCE, TARGET, r, s)
        allowance = _allow(scalar_projection_residual_m=scalar_residual,
                           jet_projection_residual_m=jet_residual)
        error = max(abs(Decimal.from_float(scalar_gap)-exact_gap),
                    abs(Decimal.from_float(jet_gap)-exact_gap))
        assert -4e-8 < float(exact_gap) < -1e-8
        assert float(error)*SUPPORT.max() < allowance.min()
        assert np.all(abs(scalar_gap-jet_gap)*SUPPORT <= allowance)
        discrepancies.append(float(error))
        residual_max = max(residual_max, scalar_residual, jet_residual)
    record_property("decimal_max_gap_error_m", max(discrepancies))
    record_property("measured_max_projection_residual_m", residual_max)
    record_property("maximum_allowance_over_support_m", float(np.max(allowance/SUPPORT)))


def test_original_bilinear_height_and_residual_sensitivity(record_property):
    # Nonaffine projected map AND varying original normal height are retained.
    target = TARGET.copy()
    target[:, 2] += np.array((0., .002, -.003, .001))
    zero = _allow(target=target)
    measured = _allow(target=target, scalar_projection_residual_m=2e-12,
                      jet_projection_residual_m=3e-12)
    assert np.all(measured > zero)
    normal, frame = _float_frame(SOURCE)
    xy = (target-SOURCE[0]) @ frame.T
    r, s = -.3, -.2
    root2 = (r+3e-12, s-2e-12)
    residual = float(np.linalg.norm((_shape(*root2)-_shape(r, s)) @ xy))
    difference = abs(float((_shape(*root2)-_shape(r, s)) @ target @ normal))
    allowance = _allow(target=target, scalar_projection_residual_m=residual)
    assert np.all(SUPPORT*difference < allowance)
    record_property("bilinear_height_gap_difference_m", difference)


def test_tilted_translated_decimal_frame_value(record_property):
    a, b = .27, -.19
    rx = np.array(((1., 0., 0.), (0., np.cos(a), -np.sin(a)),
                   (0., np.sin(a), np.cos(a))))
    ry = np.array(((np.cos(b), 0., np.sin(b)), (0., 1., 0.),
                   (-np.sin(b), 0., np.cos(b))))
    rotation = ry @ rx
    source = SOURCE @ rotation.T + (.3, -.2, .5)
    target = TARGET @ rotation.T + (.3, -.2, .5)
    normal, frame = _float_frame(source)
    r, s = -.45, -.3
    shape = _shape(r, s)
    xy = (target-source[0]) @ frame.T
    point = source[0]+(shape @ xy) @ frame
    scalar = float((point-shape @ target) @ normal)
    jet_value = float((source[0]-shape @ target) @ normal)
    exact = _decimal_gap(source, target, r, s)
    allowance = _allow(source=source, target=target)
    error = max(abs(Decimal.from_float(scalar)-exact),
                abs(Decimal.from_float(jet_value)-exact))
    assert np.all(SUPPORT*float(error) < allowance)
    record_property("tilted_decimal_gap_error_m", float(error))


def test_absolute_support_tiny_positive_rows_and_material_error_rejection(record_property):
    support = np.array((1e-25, .11, .13))
    absolute = 2*support
    allowance = _allow(support=support, absolute_support_scale=absolute)
    ordinary = _allow(support=support)
    assert allowance.shape == (3,) and np.all(allowance > 0)
    np.testing.assert_allclose(allowance, 2*ordinary, rtol=8*np.finfo(float).eps, atol=0.)
    assert not allowance.flags.writeable
    # The helper is a value budget, not an error corrector. A material injected
    # weighted-gap mismatch fails the literal caller-style comparison.
    injected = 1e-7*support
    assert np.all(injected > allowance)
    assert np.all(allowance/support < 1e-9*.8)
    record_property("all_three_normalized_allowances_m", str((allowance/support).tolist()))


@pytest.mark.parametrize("kind", ("far_origin", "target_thin", "source_thin", "reversed"))
def test_constant_height_unresolved_geometry_rejects(kind):
    source, target = SOURCE.copy(), TARGET.copy()
    target[:, 2] = .04000001  # Height-gradient zero does not waive map resolution.
    if kind == "far_origin":
        source += 1e10
        target += 1e10
    elif kind == "target_thin":
        target[:, 1] *= 1e-9
    elif kind == "source_thin":
        source[2] = source[0]+np.array((.4, 1e-10, 0.))
    else:
        target = target[[0, 3, 2, 1]]
    with pytest.raises(UnsupportedOverlapBranch):
        _allow(source=source, target=target)


@pytest.mark.parametrize("kwargs", (
    {"scalar_projection_residual_m": -1.},
    {"jet_projection_residual_m": float("nan")},
    {"natural_coordinate_bound": .9},
    {"natural_coordinate_bound": 1e300},
    {"absolute_support_scale": SUPPORT*.5},
    {"support": np.array((0., .1, .1))},
))
def test_invalid_resolution_metadata_rejects(kwargs):
    with pytest.raises(UnsupportedOverlapBranch):
        _allow(**kwargs)


def test_directed_enclosure_boundary_comparison():
    from anyfesolver.general_contact.s3_q4_gap_resolution import _enclosure_fits
    # Exact binary64 boundary must be rejected after directed comparison.
    assert not _enclosure_fits(1., 1., 2., 1.)
    assert _enclosure_fits(1., .5, 2., 1.)
    assert not _enclosure_fits(float("inf"), 1., 2., 1.)


def test_directed_normalized_cap_boundary_comparison():
    from anyfesolver.general_contact.s3_q4_gap_resolution import _within_cap
    support = np.ones(3)
    assert not _within_cap(np.full(3, 1e-9), support, 1.)
    assert _within_cap(np.full(3, .5e-9), support, 1.)
    assert not _within_cap(np.array((1e-9, .5e-9, .5e-9)), support, 1.)
