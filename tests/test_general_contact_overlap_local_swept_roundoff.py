"""Independent algebraic controls for the affine Q4/S3 swept gap bound."""

from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.general_contact.q4_overlap_local_bound import (
    _normal_controls_interval, overlap_local_swept_lower,
)
from anyfesolver.general_contact.s3_patch_static import (
    GeneralS3PatchManager, _affine_midpoint_with_error,
)
from scripts.general_contact_d1_interior_press_dev import build_interior_press
from dataclasses import replace


@pytest.mark.parametrize("triangle", (False, True))
def test_quadratic_normal_enclosure_contains_exact_rational_path(triangle):
    start = np.asarray(((1e8, 1e8, 0.), (1e8+.25, 1e8, .0001),
                        (1e8+.25, 1e8+.25, 0.), (1e8, 1e8+.25, 0.)))
    end = start + np.asarray(((0., 0., 0.), (.002, .001, .0003),
                              (0., 0., 0.), (-.001, .002, -.0002)))
    if triangle:
        start, end = start[[0, 1, 3]], end[[0, 1, 3]]
    low, high = _normal_controls_interval(start, end)
    # Fraction.from_float uses the exact stored input, without any production
    # cross product or midpoint rule in this oracle.
    for t in (Fraction(0), Fraction(1, 7), Fraction(1, 2), Fraction(1)):
        corners = [[(1-t)*Fraction.from_float(float(a))
                    + t*Fraction.from_float(float(b))
                    for a, b in zip(first, last)]
                   for first, last in zip(start, end)]
        e = [a-b for a, b in zip(corners[1], corners[0])]
        f = [a-b for a, b in zip(corners[-1], corners[0])]
        exact = [e[j]*f[k]-e[k]*f[j]
                 for j, k in ((1, 2), (2, 0), (0, 1))]
        for component, value in enumerate(exact):
            assert Fraction.from_float(float(np.min(low[:, component]))) <= value
            assert value <= Fraction.from_float(float(np.max(high[:, component])))


def test_planar_swept_lower_bounds_analytical_minimum_gap():
    source = np.asarray(((0., 0., .01), (1., 0., .01),
                         (1., 1., .01), (0., 1., .01)))
    end = source.copy()
    end[:, 2] = -.02
    target = np.asarray(((-2., -2., 0.), (4., -2., 0.), (-2., 4., 0.)))
    lower, _ = overlap_local_swept_lower(source, end, target, target, 2)
    assert lower <= -.02
    assert lower > -.02000000001


def test_endpoint_valid_but_collapsing_midpath_normal_is_not_certified():
    source = np.asarray(((0., 0., .01), (1., 0., .01),
                         (1., 1., .01), (0., 1., .01)))
    end = source.copy()
    end[:, :2] = 1.-end[:, :2]
    target = np.asarray(((-2., -2., 0.), (4., -2., 0.), (-2., 4., 0.)))
    lower, _ = overlap_local_swept_lower(source, end, target, target, 2)
    assert lower == -np.inf


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_nonfinite_target_endpoint_cannot_return_zero_gap(bad):
    source = np.asarray(((0., 0., .01), (1., 0., .01),
                         (1., 1., .01), (0., 1., .01)))
    target = np.asarray(((-2., -2., 0.), (4., -2., 0.), (-2., 4., 0.)))
    last = target.copy()
    last[0, 2] = bad
    with pytest.raises(ValueError, match="endpoints"):
        overlap_local_swept_lower(source, source, target, last, 2)


def test_subdivision_encloses_original_path_with_large_common_translation():
    left = (np.asarray([[1e16, .1, -.2]]),)
    right = (np.asarray([[1e16+2., .3, .6]]),)
    zeros = (np.zeros_like(left[0]),)
    middle, error = _affine_midpoint_with_error(left, right, zeros, zeros)
    for index in range(3):
        exact = (Fraction.from_float(float(left[0][0, index]))
                 + Fraction.from_float(float(right[0][0, index])))/2
        assert Fraction.from_float(float(middle[0][0, index]-error[0][0, index])) <= exact
        assert exact <= Fraction.from_float(float(middle[0][0, index]+error[0][0, index]))
    assert error[0][0, 0] >= 1.0  # the exact midpoint is not representable
    quarter, quarter_error = _affine_midpoint_with_error(
        left, middle, zeros, error)
    exact_quarter = (3*Fraction.from_float(float(left[0][0, 0]))
                     + Fraction.from_float(float(right[0][0, 0])))/4
    assert Fraction.from_float(float(quarter[0][0, 0]-quarter_error[0][0, 0])) <= exact_quarter
    assert exact_quarter <= Fraction.from_float(float(quarter[0][0, 0]+quarter_error[0][0, 0]))


def test_endpoint_enclosure_includes_both_body_gap_errors():
    source = np.asarray(((0., 0., .01), (1., 0., .01),
                         (1., 1., .01), (0., 1., .01)))
    end = source.copy()
    end[:, 2] = -.02
    target = np.asarray(((-2., -2., 0.), (4., -2., 0.), (-2., 4., 0.)))
    lower, _ = overlap_local_swept_lower(
        source, end, target, target, 2,
        endpoint_errors=(0., 1e-5, 0., 1e-5))
    # A permitted exact end has source z=-.02001 and target z=.00001.
    assert lower <= -.02002


def test_conditioned_bound_admits_saved_deformable_press_path():
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / "docs/d1_review_support/"
                       "d1_pressure_family_source4_coupled_preflight_20260929.json"
                       ).read_text())
    model, definition, settings, _ = build_interior_press(
        force_n=.04, reference_slope=1e-4,
        source_support="perimeter", source_columns=4, source_rows=4,
        target_columns=4, target_rows=4, target_young_multiplier=5.,
        target_reference_full_left_clamp=True, load_shape="physical_bump",
        slide_body="target")
    manager = GeneralS3PatchManager(
        model, definition,
        replace(settings, q4_source_gap_certificate="overlap_local_dev"))
    assert manager.swept_increment_admissible(
        np.zeros(model.mesh.dof_manager.total_dofs),
        np.asarray(saved["displacements"]), 0., 1.)


def test_invalid_proposal_cuts_back_without_hiding_invalid_committed_pose(monkeypatch):
    model, definition, settings, _ = build_interior_press(
        source_support="perimeter", source_rows=4,
        load_shape="interior_distributed", force_n=.04)
    manager = GeneralS3PatchManager(model, definition, settings)
    first = np.zeros(model.mesh.dof_manager.total_dofs)
    last = first.copy()
    last[model.mesh.dof_manager.get_node_dofs(7)[2]] = .1
    original = manager._gap_lower
    def admissible_domain(positions):
        if np.max(positions[0][:, 2]) > .09:
            raise ValueError("inverted trial projection")
        return original(positions)
    monkeypatch.setattr(manager, "_gap_lower", admissible_domain)
    assert manager._swept_admissible(first, last) is False
    with pytest.raises(ValueError, match="inverted trial projection"):
        manager._swept_admissible(last, first)
