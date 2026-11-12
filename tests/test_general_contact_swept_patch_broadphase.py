"""Independent swept-overlap completeness and explicit fallback controls."""

import numpy as np
import pytest

from anyfesolver.general_contact.overlap import planar_projected_overlap, PlanarFace
from anyfesolver.general_contact.q4_overlap_local_bound import overlap_local_swept_lower
from anyfesolver.general_contact.q4_source_overlap import projected_q4_source_face
from anyfesolver.general_contact.swept_patch_broadphase import conservative_swept_pair_indices


Q4 = np.array(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))
S3 = np.array(((.2, .2, 0.), (.6, .2, 0.), (.2, .6, 0.)))


def _search(a0, a1, b0, b1, **kw):
    return conservative_swept_pair_indices(a0, a1, b0, b1, (range(4),), (range(3),), **kw)


def _overlap(a, b):
    face, normal = projected_q4_source_face(a)
    target = b-((b-face.corners[0]) @ normal)[:, None]*normal
    return planar_projected_overlap(face, PlanarFace((2, 1), target)) is not None


def test_middle_only_crossing_not_endpoint_union():
    b0, b1 = S3 + (-2., 0., -.1), S3 + (2., 0., -.1)
    assert not _overlap(Q4, b0) and not _overlap(Q4, b1)
    assert _overlap(Q4, (b0+b1)/2)
    assert _search(Q4, Q4, b0, b1) == {0}
    assert overlap_local_swept_lower(Q4, Q4, b0, b1, 2)[0] < 0.0


def test_rotating_normal_matched_projection_large_separation():
    # The target lies far outside both source world AABBs, but shares the
    # middle source projection after a long displacement along n0(1/2).
    a0, a1 = Q4.copy(), Q4.copy()
    a0[:, 2] = -.15*a0[:, 0]
    a1[:, 2] = .35*a1[:, 0]
    middle = (a0+a1)/2
    _, normal = projected_q4_source_face(middle)
    target = S3.copy()
    target[:, 2] = .1*target[:, 0]
    target += 1000.*normal
    assert target[:, 0].max() < a0[:, 0].min()
    assert _overlap(middle, target)
    assert _search(a0, a1, target, target) == {0}


@pytest.mark.parametrize("height", [-1e8, 1e8])
def test_normal_depth_does_not_exclude_flat_overlap(height):
    b = S3 + (0., 0., height)
    assert _search(Q4, Q4, b, b) == {0}


def test_warped_q4_actual_projection_is_retained():
    a0 = Q4.copy()
    a1 = Q4.copy()
    a0[:, 2] = (0., .08, .13, -.03)
    a1[:, 2] = (.01, -.04, .12, .06)
    b0, b1 = S3+(0., 0., -.02), S3+(.1, 0., -.03)
    assert _search(a0, a1, b0, b1) == {0}
    for fraction in np.linspace(0., 1., 9):
        assert _overlap((1-fraction)*a0+fraction*a1,
                        (1-fraction)*b0+fraction*b1)


def test_endpoint_error_enclosure_keeps_possible_overlap():
    b = S3+(1., 0., 0.)
    assert _search(Q4, Q4, b, b) == set()
    error = np.zeros_like(b)
    error[:, 0] = .25
    assert _search(Q4, Q4, b, b, endpoint_errors=(0., 0., error, error)) == {0}
    assert _overlap(Q4, b+(-.25, 0., 0.))


def test_source_endpoint_enclosure_and_uncertain_cone():
    b = S3+(0.80001, 0., 0.)
    error = np.zeros_like(Q4)
    error[:, 0] = 2e-5
    assert _search(Q4, Q4, b, b) == set()
    assert _search(Q4, Q4, b, b, endpoint_errors=(error, error, 0., 0.)) == {0}
    assert _overlap(Q4+(2e-5, 0., 0.), b)
    # If endpoint uncertainty makes a cone unresolved, fallback preserves
    # the old kernel failure rather than treating the nominal path as safe.
    assert _search(Q4, Q4, b, b, endpoint_errors=(.5, .5, 0., 0.)) is None


@pytest.mark.parametrize("permutation", [(2, 0, 1), (1, 2, 0)])
def test_normal_component_selection_in_other_world_axes(permutation):
    a = Q4[:, permutation]
    b = (S3+(0., 0., 1e4))[:, permutation]
    assert _search(a, a, b, b) == {0}
    far = (S3+(5., 0., 1e4))[:, permutation]
    assert _search(a, a, far, far) == set()


def test_large_coordinates_and_one_ulp_touching_retained():
    shift = np.array((1e9, -1e9, 1e9))
    a = Q4+shift
    b = S3+shift+(0.8, 0., 0.)
    b[:, 0] = np.nextafter(b[:, 0], np.inf)
    assert _search(a, a, b, b) == {0}


def test_remote_singleton_degenerate_target_rejected():
    bad = np.array(((50., 0., 0.), (51., 0., 0.), (52., 0., 0.)))
    with pytest.raises(ValueError, match="degenerate"):
        _search(Q4, Q4, bad, bad)


def test_remote_folded_source_rejected():
    bad = Q4.copy()
    bad[2] = (.2, -.2, 0.)
    with pytest.raises(ValueError, match="inverted|folded|convex"):
        _search(bad, bad, S3+(50., 0., 0.), S3+(50., 0., 0.))


def test_remote_grazing_pair_requires_full_fallback():
    b = np.array(((50., 0., 0.), (50., 1., 0.), (50., 0., 1.)))
    assert _search(Q4, Q4, b, b) is None
    assert overlap_local_swept_lower(Q4, Q4, b, b, 1)[0] == -np.inf


def test_all_physical_corner_cones_guard_even_remote_pair():
    # Parent normal is vertical, while the opposite physical corner exceeds
    # the kernel's cone threshold. A remote target must not hide this branch.
    a = Q4.copy()
    a[2, 2] = 1.
    b = S3+(50., 0., 0.)
    assert _search(a, a, b, b) is None
    assert overlap_local_swept_lower(a, a, b, b, 1)[0] == -np.inf


def test_parent_orientation_reversal_requests_fallback():
    a1 = Q4*np.array((-1., -1., 1.))
    assert _search(Q4, a1, S3, S3) is None


def test_exhaustive_small_scene_bound_and_sampled_overlap_parity():
    sources0 = np.vstack([Q4+(3.*i, 0., 0.) for i in range(4)])
    sources1 = sources0+(0.2, 0., 0.)
    targets0 = np.vstack([S3+(3.*i-.3, 0., -.04) for i in range(4)])
    targets1 = targets0+(0.5, 0., .01)
    si = tuple(np.arange(4*i, 4*i+4) for i in range(4))
    ti = tuple(np.arange(3*i, 3*i+3) for i in range(4))
    candidates = conservative_swept_pair_indices(
        sources0, sources1, targets0, targets1, si, ti)
    assert candidates == {0, 5, 10, 15}
    full, retained = [], []
    for i, nodes in enumerate(si):
        for j, target in enumerate(ti):
            lower, _ = overlap_local_swept_lower(
                sources0[nodes], sources1[nodes], targets0[target], targets1[target], 2)
            full.append(lower)
            if 4*i+j in candidates:
                retained.append(lower)
            else:
                assert lower == np.inf
            for fraction in np.linspace(0., 1., 13):
                a = (1-fraction)*sources0[nodes]+fraction*sources1[nodes]
                b = (1-fraction)*targets0[target]+fraction*targets1[target]
                if _overlap(a, b):
                    assert 4*i+j in candidates
    assert min(full) == min(retained)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.])
def test_invalid_enclosures_rejected(bad):
    with pytest.raises(ValueError, match="enclosure"):
        _search(Q4, Q4, S3, S3, endpoint_errors=(bad, 0., 0., 0.))


def test_invalid_topology_rejected_and_unsupported_shape_falls_back():
    with pytest.raises(ValueError, match="indices"):
        conservative_swept_pair_indices(Q4, Q4, S3, S3, ((0, 1, 2, 4),), (range(3),))
    assert conservative_swept_pair_indices(S3, S3, S3, S3, (range(3),), (range(3),)) is None


def test_nonfinite_remote_endpoint_rejected():
    b = S3+(50., 0., 0.)
    b[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _search(Q4, Q4, b, b)
