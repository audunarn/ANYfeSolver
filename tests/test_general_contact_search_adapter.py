"""Independent small-scene controls for owner-backed contact search."""

import numpy as np
import pytest

from anyfesolver.general_contact.contact_search import ContactAABBIndex


def _exhaustive(keys, boxes, window, padding=0.0):
    lower = window[:3] - padding
    upper = window[3:] + padding
    return tuple(sorted(
        key for key, box in zip(keys, boxes, strict=True)
        if np.all(box[:3] <= upper) and np.all(lower <= box[3:])
    ))


def test_owner_backed_search_matches_exhaustive_3d_and_refits():
    rng = np.random.default_rng(7321)
    centers = rng.uniform(-2.0, 2.0, (120, 3))
    half_widths = rng.uniform(0.001, 0.2, (120, 3))
    boxes = np.column_stack((centers-half_widths, centers+half_widths))
    keys = tuple(("shell_face", int(i)) for i in rng.permutation(len(boxes)))
    index = ContactAABBIndex(keys, boxes)
    for shift in (0.0, 0.35):
        moved = boxes.copy()
        moved[:, (0, 3)] += shift
        index.refit(moved)
        for _ in range(30):
            center = rng.uniform(-2.0, 2.0, 3)
            radius = rng.uniform(0.0, 0.3, 3)
            window = np.r_[center-radius, center+radius]
            for padding in (0.0, 1.0e-8):
                assert index.query(window, padding=padding) == _exhaustive(
                    keys, moved, window, padding)


def test_owner_backed_search_keeps_identity_and_fails_before_invalid_refit():
    keys = (("beam_cell", 8), ("beam_cell", 2))
    boxes = np.array(((0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                      (4.0, 4.0, 4.0, 5.0, 5.0, 5.0)))
    index = ContactAABBIndex(keys, boxes)
    first_window = np.array((1.0, 0.0, 0.0, 2.0, 1.0, 1.0))
    assert index.query(first_window) == (("beam_cell", 8),)
    result = index.query_with_diagnostics(first_window)
    assert result.keys == (("beam_cell", 8),)
    assert result.diagnostics.candidate_count == 1
    assert index.diagnostics.size == 2
    invalid = boxes.copy()
    invalid[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite ordered"):
        index.refit(invalid)
    assert index.query(first_window) == (("beam_cell", 8),)
    with pytest.raises(ValueError, match="unique"):
        ContactAABBIndex((keys[0], keys[0]), boxes)
