"""Focused smooth-branch comparisons for Hessian-free Q4/S3 evaluation."""

import numpy as np
import pytest

import anyfesolver.general_contact.q4_s3_overlap_derivatives as derivative_module
from anyfesolver.general_contact.q4_s3_overlap_derivatives import (
    Q4S3OverlapFirstDerivatives,
    UnsupportedQ4S3Branch,
    q4_s3_overlap_derivatives,
    q4_s3_overlap_first_derivatives,
)


REFERENCE = np.array([[0., 0., 0.], [2., 0., 0.],
                      [2., 2., 0.], [0., 2., 0.]])
SOURCE = np.array([[0., 0., 0.], [2., 0., .1],
                   [2., 2., 0.], [0., 2., -.1]])
TARGET = np.array([[.31, .19, -.11], [2.21, .37, -.03],
                   [.63, 1.77, -.15]])
OPTIONS = dict(area_tolerance=1e-9, weighted_gap_tolerance=1e-9,
               maximum_triangles=32, relative_branch_tolerance=1e-10)


@pytest.mark.parametrize("warped", [False, True])
def test_first_order_matches_full_value_and_gradient_on_smooth_pose(warped):
    source = SOURCE.copy()
    if warped:
        source[2, 2] += .12
    options = dict(OPTIONS, maximum_triangles=256) if warped else OPTIONS
    first_options = dict(allow_warped_source=warped)
    full_options = dict(allow_warped_source=warped)
    if warped:
        first_options.update(support_gradient_tolerance=1e-4,
                             weighted_gap_gradient_tolerance=1e-4)
        full_options.update(first_options,
                            support_hessian_tolerance=1e-4,
                            weighted_gap_hessian_tolerance=1e-4)
    first = q4_s3_overlap_first_derivatives(
        REFERENCE, source, TARGET, **options, **first_options)
    full = q4_s3_overlap_derivatives(
        REFERENCE, source, TARGET, **options, **full_options)
    assert isinstance(first, Q4S3OverlapFirstDerivatives)
    assert not hasattr(first, "support_hessian")
    assert not hasattr(first, "weighted_gap_hessian")
    assert first.integration_triangles <= options["maximum_triangles"]
    assert first.integration_tolerances == (
        options["area_tolerance"], options["weighted_gap_tolerance"],
        1e-4 if warped else None, 1e-4 if warped else None)
    assert first.affine_relative_gradient_tolerance == (None if warped else 1e-10)
    for name in ("reference_support", "reference_weighted_gap",
                 "support_gradient", "weighted_gap_gradient"):
        np.testing.assert_allclose(getattr(first, name), getattr(full, name),
                                   rtol=1e-5, atol=2e-7)
    np.testing.assert_allclose(first.gradient_quadrature_differences,
                               full.derivative_quadrature_differences[[0, 2]],
                               rtol=1e-5, atol=1e-12)
    assert first.branch_signature == full.branch_signature


def test_first_order_rejects_clipping_and_affine_boundaries():
    warped = SOURCE.copy()
    warped[2, 2] += .02
    with pytest.raises(UnsupportedQ4S3Branch, match="warped or non-affine"):
        q4_s3_overlap_first_derivatives(REFERENCE, warped, TARGET, **OPTIONS)
    with pytest.raises(UnsupportedQ4S3Branch, match="clipping boundary"):
        q4_s3_overlap_first_derivatives(
            REFERENCE, SOURCE, SOURCE[[0, 1, 3]], **OPTIONS)
    with pytest.raises(UnsupportedQ4S3Branch, match="triangle budget"):
        q4_s3_overlap_first_derivatives(
            REFERENCE, SOURCE, TARGET, **dict(OPTIONS, maximum_triangles=1))


def test_first_order_warped_requires_gradient_tolerances():
    warped = SOURCE.copy()
    warped[2, 2] += .12
    with pytest.raises(ValueError, match="two positive gradient tolerances"):
        q4_s3_overlap_first_derivatives(
            REFERENCE, warped, TARGET, **OPTIONS, allow_warped_source=True)


def test_first_order_does_not_construct_second_order_jets(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Jet2 construction in first-order path")

    monkeypatch.setattr(derivative_module.Jet2, "variable", forbidden)
    result = q4_s3_overlap_first_derivatives(
        REFERENCE, SOURCE, TARGET, **OPTIONS)
    assert np.all(np.isfinite(result.support_gradient))
