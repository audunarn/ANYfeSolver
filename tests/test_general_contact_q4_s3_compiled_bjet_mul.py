"""Parity for the opt-in raw-array batched Jet2 multiplication."""

import numpy as np

import anyfesolver.general_contact.q4_s3_overlap_derivatives as derivatives


REFERENCE = np.array([[0., 0., 0.], [2., 0., 0.],
                      [2., 2., 0.], [0., 2., 0.]])
SOURCE = np.array([[0., 0., 0.], [2., 0., .1],
                   [2., 2., 0.], [0., 2., -.1]])
TARGET = np.array([[.31, .19, -.11], [2.21, .37, -.03],
                   [.63, 1.77, -.15]])
OPTIONS = dict(area_tolerance=1e-9, weighted_gap_tolerance=1e-9,
               maximum_triangles=256, relative_branch_tolerance=1e-10)
WARPED_OPTIONS = dict(
    allow_warped_source=True,
    support_gradient_tolerance=1e-4,
    support_hessian_tolerance=1e-4,
    weighted_gap_gradient_tolerance=1e-4,
    weighted_gap_hessian_tolerance=1e-4,
)
FIELDS = (
    "reference_support", "reference_weighted_gap",
    "support_gradient", "support_hessian",
    "weighted_gap_gradient", "weighted_gap_hessian",
    "overlap_area", "overlap_area_resolution",
    "reference_overlap_area", "derivative_quadrature_differences",
)


def test_compiled_bjet_mul_matches_numpy_on_twelve_smooth_poses(monkeypatch):
    if not derivatives._enable_compiled_bjet_mul():
        assert not derivatives._COMPILED_BJET_MUL_ENABLED
        return
    rng = np.random.default_rng(20260928)
    for pose in range(12):
        source = SOURCE.copy()
        target = TARGET.copy()
        if pose:
            offset = rng.normal(size=(7, 3))*0.005
            offset[2] = offset[1]+offset[3]-offset[0]
            source += offset[:4]
            target += offset[4:]
        options = dict(OPTIONS)
        if pose in (10, 11):
            source[2, 2] += .12
            options.update(WARPED_OPTIONS)

        monkeypatch.setattr(derivatives, "_COMPILED_BJET_MUL_ENABLED", False)
        baseline = derivatives.q4_s3_overlap_derivatives(
            REFERENCE, source, target, **options,
        )
        monkeypatch.setattr(derivatives, "_COMPILED_BJET_MUL_ENABLED", True)
        compiled = derivatives.q4_s3_overlap_derivatives(
            REFERENCE, source, target, **options,
        )

        assert compiled.branch_signature == baseline.branch_signature
        assert compiled.integration_triangles == baseline.integration_triangles
        for field in FIELDS:
            left = np.asarray(getattr(baseline, field))
            right = np.asarray(getattr(compiled, field))
            scale = np.maximum(1., np.maximum(np.abs(left), np.abs(right)))
            assert np.max(np.abs(left-right)/scale) <= 1e-13, (pose, field)
