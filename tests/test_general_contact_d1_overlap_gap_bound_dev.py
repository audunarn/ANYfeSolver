"""Independent planar controls for the saved-pose overlap-bound screen."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from anyfesolver import validate_general_contact
from anyfesolver.general_contact.q4_source_overlap import (
    q4_source_planar_overlap_gap_lower,
    q4_source_projected_gap_lower,
    q4_source_projected_swept_gap_lower,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.q4_overlap_local_bound import (
    overlap_local_lower,
    overlap_local_swept_lower,
)
from scripts.general_contact_d1_2d_target_press_dev import build_case
from scripts.general_contact_d1_interior_press_dev import build_interior_press
from examples.general_contact_d1_interior_controls_dev import (
    _independent_projected_gap_bound,
)


def test_partial_planar_overlap_bound_is_conservative_and_tightens():
    source = np.asarray((
        (0.0, 0.0, 0.0), (1.0, 0.0, -0.1),
        (1.0, 1.0, -0.1), (0.0, 1.0, 0.0),
    ))
    target = np.asarray((
        (0.01, 0.01, 0.0), (0.21, 0.01, 0.0),
        (0.01, 0.21, 0.0),
    ))
    independent = q4_source_planar_overlap_gap_lower(source, target)
    assert independent is not None
    broad = q4_source_projected_gap_lower(source, target)
    local0, _ = overlap_local_lower(source, target, 0)
    local4, _ = overlap_local_lower(source, target, 4)
    assert local0 < broad + 1.0e-11
    assert local0 <= local4 < independent + 1.0e-11
    assert local4 - local0 > 0.05
    assert max(broad, local4) <= independent + 1.0e-11


def test_disjoint_projected_faces_have_no_overlap_contribution():
    source = np.asarray((
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    ))
    target = np.asarray((
        (1.1, 0.1, 0.0), (1.2, 0.1, 0.0),
        (1.1, 0.2, 0.0),
    ))
    assert q4_source_planar_overlap_gap_lower(source, target) is None
    lower, visited = overlap_local_lower(source, target, 4)
    assert lower == float("inf")
    assert visited == 1


def test_linear_sweep_bound_covers_independent_planar_overlap_samples():
    source = np.asarray((
        (0.0, 0.0, 0.0), (1.0, 0.0, -0.1),
        (1.0, 1.0, -0.1), (0.0, 1.0, 0.0),
    ))
    target = np.asarray((
        (0.01, 0.01, 0.0), (0.21, 0.01, 0.0),
        (0.01, 0.21, 0.0),
    ))
    source_end = source + np.array((0.01, 0.0, -0.01))
    target_end = target + np.array((-0.01, 0.0, 0.005))
    swept, visited = overlap_local_swept_lower(
        source, source_end, target, target_end, 4,
    )
    samples = [q4_source_planar_overlap_gap_lower(
        (1-f)*source+f*source_end,
        (1-f)*target+f*target_end,
    ) for f in np.linspace(0.0, 1.0, 65)]
    assert visited > 1
    assert all(item is not None for item in samples)
    assert swept <= min(samples) + 1.0e-11
    zero, _ = overlap_local_swept_lower(source, source, target, target, 4)
    assert zero <= samples[0] + 1.0e-11
    reversed_lower, _ = overlap_local_swept_lower(
        source_end, source, target_end, target, 4)
    assert reversed_lower <= min(samples) + 1.0e-11
    angle = 0.37
    rotation = np.asarray(((np.cos(angle), -np.sin(angle), 0.0),
                           (np.sin(angle), np.cos(angle), 0.0),
                           (0.0, 0.0, 1.0)))
    shift = np.asarray((0.15, -0.21, 0.07))
    transformed, _ = overlap_local_swept_lower(
        source@rotation.T+shift, source_end@rotation.T+shift,
        target@rotation.T+shift, target_end@rotation.T+shift, 4)
    assert abs(transformed-swept) < 1.0e-9


def test_sweep_cannot_prune_a_midpath_crossing():
    source = np.asarray((
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    ))
    target = np.asarray((
        (1.1, 0.1, 0.01), (1.2, 0.1, 0.01),
        (1.1, 0.2, 0.01),
    ))
    target_end = target + np.array((-1.3, 0.0, 0.0))
    assert q4_source_planar_overlap_gap_lower(source, target) is None
    assert q4_source_planar_overlap_gap_lower(source, target_end) is None
    assert q4_source_planar_overlap_gap_lower(
        source, 0.5*(target+target_end),
    ) is not None
    swept, _ = overlap_local_swept_lower(
        source, source, target, target_end, 3,
    )
    assert np.isfinite(swept)
    assert swept <= -0.01 + 1.0e-11


def test_swept_bound_covers_independent_warped_surface_samples():
    """Check moving projection and physical normals without the bound kernel."""
    rng = np.random.default_rng(20260929)
    reference = np.asarray((
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    ))
    target_reference = np.asarray((
        (0.2, 0.15, 0.015), (0.8, 0.2, 0.015),
        (0.25, 0.85, 0.015),
    ))
    tested = 0
    for _ in range(8):
        source0 = reference + rng.uniform(-0.005, 0.005, size=(4, 3))
        source1 = source0 + rng.uniform(-0.015, 0.015, size=(4, 3))
        target0 = target_reference + rng.uniform(-0.005, 0.005, size=(3, 3))
        target1 = target0 + rng.uniform(-0.015, 0.015, size=(3, 3))
        lower, _ = overlap_local_swept_lower(
            source0, source1, target0, target1, 3,
        )
        for time in np.linspace(0.0, 1.0, 11):
            source = (1-time)*source0+time*source1
            target = (1-time)*target0+time*target1
            projection = np.cross(source[1]-source[0],
                                  source[3]-source[0])
            projection /= np.linalg.norm(projection)
            target_normal = np.cross(target[1]-target[0],
                                     target[2]-target[0])
            denominator = float(target_normal @ projection)
            assert denominator > 0.0
            for u in np.linspace(0.0, 1.0, 9):
                for v in np.linspace(0.0, 1.0, 9):
                    weights = np.asarray(((1-u)*(1-v), u*(1-v),
                                          u*v, (1-u)*v))
                    point = weights @ source
                    distance = float(target_normal @ (point-target[0]))/denominator
                    projected = point-distance*projection
                    bary = np.linalg.lstsq(
                        np.stack((target[1]-target[0],
                                  target[2]-target[0]), axis=1),
                        projected-target[0], rcond=None,
                    )[0]
                    if (min(bary) < -1.0e-11
                            or sum(bary) > 1.0+1.0e-11):
                        continue
                    du = ((1-v)*(source[1]-source[0])
                          + v*(source[2]-source[3]))
                    dv = ((1-u)*(source[3]-source[0])
                          + u*(source[2]-source[1]))
                    physical = np.cross(du, dv)
                    physical /= np.linalg.norm(physical)
                    sampled_gap = float(physical @ (point-projected))
                    assert lower <= sampled_gap + 1.0e-10
                    tested += 1
    assert tested > 100


def test_warped_source_bound_uses_signed_physical_normal_work():
    """An independently computed warped-corner gap stays above each bound."""
    source = np.asarray((
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (1.0, 1.0, -0.5), (0.0, 1.0, 0.0),
    ))
    target = np.asarray((
        (-1.0, -1.0, 0.0), (4.0, -1.0, 0.0),
        (-1.0, 4.0, 0.0),
    ))
    # The source corner (1, 1) projects inside the target triangle. Its
    # physical normal follows directly from the two bilinear tangents.
    du = source[2]-source[3]
    dv = source[2]-source[1]
    normal = np.cross(du, dv)
    normal /= np.linalg.norm(normal)
    physical_gap = float(normal @ (source[2]-np.array((1.0, 1.0, 0.0))))
    assert np.isclose(physical_gap, -1.0/np.sqrt(6.0), atol=1e-15)

    bounds = (
        q4_source_projected_gap_lower(source, target),
        q4_source_projected_swept_gap_lower(
            source, source, target, target, np.array((0.0, 0.0, 1.0)),
        ),
        overlap_local_swept_lower(source, source, target, target, 0)[0],
    )
    assert all(lower <= physical_gap for lower in bounds)


def test_saved_first_step_profile_is_explicit_and_binds_restart_identity():
    saved = Path(__file__).resolve().parents[1] / (
        "docs/d1_review_support/d1_2d_gap_proposal_20260929.json"
    )
    pose = json.loads(saved.read_text(encoding="utf-8"))
    s0, s1, t0, t1 = (np.asarray(pose[key], dtype=float) for key in (
        "current_source_positions_m", "proposed_source_positions_m",
        "current_target_positions_m", "proposed_target_positions_m",
    ))
    model, definition, old_settings, _ = build_case()
    new_settings = replace(old_settings,
                           q4_source_gap_certificate="overlap_local_dev")
    old = GeneralS3PatchManager(model, definition, old_settings)
    new = GeneralS3PatchManager(model, definition, new_settings)
    assert validate_general_contact(model, definition, new_settings).valid
    assert old.profile != new.profile
    assert old.restart_contract() != new.restart_contract()
    old_lower = min(q4_source_projected_swept_gap_lower(
        s0[si], s1[si], t0[ti], t1[ti], old.reference_normals[0],
    ) for si in old.law.source_face_indices
        for ti in old.law.target_face_indices)
    new_lower = min(overlap_local_swept_lower(
        s0[si], s1[si], t0[ti], t1[ti], 3,
    )[0] for si in new.law.source_face_indices
        for ti in new.law.target_face_indices)
    assert old_lower < -new.gap_tolerance
    assert new_lower >= -new.gap_tolerance


def test_saved_eight_s3_rejected_proposal_has_admissible_local_sweep():
    saved = Path(__file__).resolve().parents[1] / (
        "docs/d1_review_support/d1_target8s3_swept_capture_20260929.json"
    )
    capture = json.loads(saved.read_text(encoding="utf-8"))[
        "swept_failure_discriminator"]
    model, definition, settings, _program = build_interior_press(
        force_n=0.04, reference_slope=1.0e-4,
        source_support="perimeter", source_columns=8, source_rows=8,
        target_columns=2, target_rows=2,
        target_reference_full_left_clamp=True,
        load_shape="physical_bump", slide_body="target",
    )
    old = GeneralS3PatchManager(model, definition, settings)
    new = GeneralS3PatchManager(
        model, definition,
        replace(settings, q4_source_gap_certificate="overlap_local_dev"),
    )
    start = np.asarray(capture["captured_start_displacement"])
    proposed = np.asarray(capture["captured_proposed_displacement"])
    assert not old._swept_admissible(start, proposed)
    assert new._swept_admissible(start, proposed)
    source, target = new._positions(proposed)
    assert new.require_d1_coverage(source, target) is None
    independent = _independent_projected_gap_bound(
        model, definition, proposed,
        definition.interactions[0].gap_tolerance)
    assert independent >= -definition.interactions[0].gap_tolerance
