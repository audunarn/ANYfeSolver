"""Padded projected-box rejection against the retained exhaustive predicate."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver.general_contact.overlap import PlanarFace
from anyfesolver.general_contact.s3_patch_augmented import _touching_projected_boundaries


def _old_touch():
    # Extract only the exact prepatch predicate; do not import another kernel.
    path = Path(__file__).parents[1] / "docs/d1_review_support/d1_touch_bounds_prepatch_20260930.py.txt"
    raw = path.read_bytes()
    # Byte-identical to 590d7a42^:src/anyfesolver/general_contact/s3_patch_augmented.py.
    # The former digest did not identify this committed prepatch source.
    assert hashlib.sha256(raw).hexdigest() == "e5e885bfdc10469d476f86db60376ed9ef165d3cae7b351f76fee2715fa2d4e3"
    node = next(n for n in ast.parse(raw).body
                if isinstance(n, ast.FunctionDef) and n.name == "_touching_projected_boundaries")
    namespace = dict(np=np, PlanarFace=PlanarFace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def _quad(width=1.):
    return np.array(((0., 0., 0.), (width, 0., 0.), (width, 1., 0.), (0., 1., 0.)))


@pytest.mark.parametrize("gap", [0., 4e-12, 2e-11, 4e-11, 1., 1e8])
@pytest.mark.parametrize("rotated", [False, True])
def test_old_loop_parity_near_contact_far_and_rotated(gap, rotated):
    first, second = _quad(), np.array(((1.+gap, .2, 12.), (2.+gap, .2, 12.), (1.+gap, .8, 12.)))
    if rotated:
        # Rigid 3D frame change preserves projection, including normal depth.
        axis = np.array((1., 2., 3.)); axis /= np.linalg.norm(axis)
        cross = np.array(((0., -axis[2], axis[1]), (axis[2], 0., -axis[0]), (-axis[1], axis[0], 0.)))
        rotation = np.eye(3) + np.sin(.73)*cross + (1.-np.cos(.73))*(cross@cross)
        first, second = first@rotation.T, second@rotation.T
    faces = PlanarFace((1, 0), first), PlanarFace((2, 0), second)
    assert _touching_projected_boundaries(*faces) == _old_touch()(*faces)


def test_edge_vertex_and_interior_overlap_parity():
    first = PlanarFace((1, 0), _quad())
    for points in (np.array(((1., 1., 0.), (2., 1., 0.), (1., 2., 0.))),
                   _quad()+np.array((1., 0., 3.)), _quad()+np.array((.2, .2, 0.))):
        second = PlanarFace((2, 0), points)
        assert _touching_projected_boundaries(first, second) == _old_touch()(first, second)


def test_far_rejection_avoids_segments_but_near_rounding_uses_old_loop(monkeypatch):
    calls = []
    original = np.clip
    def clip(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(np, "clip", clip)
    first = PlanarFace((1, 0), _quad())
    second = PlanarFace((2, 0), _quad()+np.array((4., 0., 0.)))
    assert not _touching_projected_boundaries(first, second)
    assert not calls
    first = PlanarFace((1, 0), _quad()*1e12)
    second = PlanarFace((2, 0), _quad()+np.array((1e12+.01, 0., 0.)))
    assert not _touching_projected_boundaries(first, second, relative_tolerance=0.)
    assert calls  # Padding includes projected magnitude, not only face size.


def test_large_coordinate_translation_matches_exhaustive_loop():
    offset = np.array((1e12, -2e12, 3e12))
    for gap in (0., .001, 1., 1e5):
        first = PlanarFace((1, 0), _quad()+offset)
        second = PlanarFace((2, 0), _quad()+offset+np.array((1.+gap, 0., 7.)))
        assert _touching_projected_boundaries(first, second) == _old_touch()(first, second)


@pytest.mark.parametrize("small_edge", [0., 1e-180])
def test_invalid_segment_arithmetic_retains_original_failure(small_edge):
    first = PlanarFace((1, 0), _quad())
    # Deliberately bypass PlanarFace validation to test the helper's fallback.
    second = SimpleNamespace(corners=np.array(((10., 0., 0.), (10., small_edge, 0.), (11., 1., 0.))))
    for predicate in (_old_touch(), _touching_projected_boundaries):
        with np.errstate(invalid="raise", divide="raise", under="ignore"):
            with pytest.raises(FloatingPointError):
                predicate(first, second)


def test_subnormal_segment_length_uses_original_loop(monkeypatch):
    calls = []
    original = np.clip
    monkeypatch.setattr(np, "clip", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    first = PlanarFace((1, 0), _quad())
    second = SimpleNamespace(corners=np.array(((10., 0., 0.), (10., 1e-160, 0.), (11., 1., 0.))))
    with np.errstate(all="ignore"):
        assert _touching_projected_boundaries(first, second) == _old_touch()(first, second)
    assert calls


def test_extreme_projected_magnitude_keeps_original_arithmetic(monkeypatch):
    calls = []
    original = np.clip
    monkeypatch.setattr(np, "clip", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    first = PlanarFace((1, 0), _quad())
    # A remote grazing projection; finite projected edges but distance squares
    # may overflow. The box shortcut must not bypass the original arithmetic.
    second = SimpleNamespace(corners=np.array(((1e200, 0., 0.), (1e200, 1., 0.),
                                               (1e200, 2., 1.), (1e200, 3., 1.))))
    with np.errstate(all="ignore"):
        assert _touching_projected_boundaries(first, second) == _old_touch()(first, second)
    assert calls


def test_nonfinite_and_nonfinite_tolerance_use_segment_fallback(monkeypatch):
    calls = []
    original = np.clip
    monkeypatch.setattr(np, "clip", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    first = PlanarFace((1, 0), _quad())
    bad = SimpleNamespace(corners=np.array(((np.nan, 0., 0.), (20., 0., 0.), (20., 1., 0.))))
    with np.errstate(all="ignore"):
        assert _touching_projected_boundaries(first, bad) == _old_touch()(first, bad)
    assert calls
    calls.clear()
    second = PlanarFace((2, 0), _quad()+np.array((10., 0., 0.)))
    assert _touching_projected_boundaries(first, second, relative_tolerance=np.inf)
    assert calls


def test_strict_underflow_policy_retains_original_segment_exception():
    # Normal denominators do not preclude subnormal projection fractions.
    s = np.nextafter(0., 1.)
    first = PlanarFace((1, 0), _quad())
    second = PlanarFace((2, 0), np.array(((10., -s, 0.), (11., -s, 0.),
                                        (11., 1.5, 0.), (10., 1.5, 0.))))
    for predicate in (_old_touch(), _touching_projected_boundaries):
        with np.errstate(under="raise"):
            with pytest.raises(FloatingPointError):
                predicate(first, second)
