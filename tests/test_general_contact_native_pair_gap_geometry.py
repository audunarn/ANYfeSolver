"""Geometry certificates for the complete retained pair, without contact work."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver.general_contact import native_retained_whole_pair as pair_module
from anyfesolver.general_contact.native_retained_whole_pair import (
    NativeRetainedWholeCurvePair,
)


def _reference(y, xs=(0.0, 1.0, 2.0)):
    return CenteredCurvedBeam3ReferenceGeometry(
        np.array([(x, y, 0.0) for x in xs]),
        np.tile(np.eye(3), (3, 1, 1)),
    )


def _pose(reference):
    return [reference.coordinates.copy(), np.zeros((3, 3)),
            np.tile(np.eye(3), (2, 1, 1))]


def _kernel(a, b, *, certify=True, depth=20, cells=10000,
            parameter_tolerance=1e-2):
    return NativeRetainedWholeCurvePair(
        11, a, (1, 2, 3), 0.05, 12, b, (4, 5, 6), 0.05,
        line_stiffness_n_per_m2=100.0,
        distance_tolerance_m=1e-10, feature_tolerance_m=1e-9,
        search_parameter_tolerance=parameter_tolerance,
        maximum_search_depth=depth, maximum_search_cells=cells,
        energy_tolerance_j=1e-8, force_tolerance_n=1e-7,
        tangent_tolerance_n_per_m=1e-6,
        maximum_integration_intervals=1000,
        certify_open_intervals=certify,
    )


@pytest.mark.parametrize("clearance", (0.08, 0.3))
@pytest.mark.parametrize("certify", (False, True))
def test_straight_parallel_lower_is_conservative_and_matches_full(clearance,
                                                                certify):
    a, b = _reference(0.0), _reference(clearance)
    kernel = _kernel(a, b, certify=certify)
    identity = kernel.identity
    lower = kernel.geometry_gap_lower(*_pose(a), *_pose(b))
    # Independent finite parallel segment distance: equal spans overlap fully.
    true_gap = clearance-0.1
    assert np.isfinite(lower)
    assert lower <= true_gap
    if clearance > 0.1:
        assert lower > 0.0
    result = kernel.evaluate(*_pose(a), *_pose(b))
    assert lower == pytest.approx(result.conservative_lower_gap_m, abs=1e-13)
    assert lower <= result.minimum_sampled_gap_m
    assert kernel.identity == identity == kernel._fingerprint()


@pytest.mark.parametrize("cell", (0, 1))
def test_internal_rotation_keeps_interior_pocket_with_open_chord(monkeypatch,
                                                               cell):
    a = _reference(0.12, (0.0, 1.0, 2.24))
    b = _reference(0.0, (0.0, 1.12, 2.24))
    kernel = _kernel(a, b)
    pose = _pose(a)

    def no_contact_work(*args, **kwargs):
        raise AssertionError("geometry query evaluated a contact distance jet")

    monkeypatch.setattr(pair_module, "b3_distance_jet", no_contact_work)
    open_lower = kernel.geometry_gap_lower(*pose, *_pose(b))
    assert open_lower > 0.0
    fixed_positions = pose[0].copy()
    pose[2][cell] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    lower = kernel.geometry_gap_lower(*pose, *_pose(b))
    np.testing.assert_array_equal(pose[0], fixed_positions)
    # Analytic rotated lift: y(t)=0.12-0.12*t*(1-t), with its
    # minimum 0.09 at t=1/2. Endpoints and their chord have gap +0.02.
    assert np.all(pose[0][:, 1]-0.1 > 0.0)
    assert lower <= 0.09-0.1
    assert lower < open_lower
    exchanged = _kernel(b, a).geometry_gap_lower(*_pose(b), *pose)
    assert exchanged == pytest.approx(lower, abs=1e-13)


def test_equal_remote_minima_have_gap_certificate_but_no_force_admission():
    a = _reference(0.0, (-0.5, 0.5, 1.5))
    b = _reference(0.2, (-1.0, 0.0, 1.0))
    kernel = _kernel(a, b)
    pose_b = _pose(b)
    pose_b[0][1, 1] = 1.0
    lower = kernel.geometry_gap_lower(*_pose(a), *pose_b)
    assert np.isfinite(lower)
    assert lower > 0.0
    with pytest.raises(ValueError, match="competing remote target minima"):
        kernel.evaluate(*_pose(a), *pose_b)


def test_unresolved_intervals_are_retained_and_search_budget_is_enforced():
    a, b = _reference(0.0), _reference(0.08)
    kernel = _kernel(a, b, certify=False, depth=1,
                     parameter_tolerance=0.6)
    lower = kernel.geometry_gap_lower(*_pose(a), *_pose(b))
    # At this coarse stopping tolerance every leaf overlaps activation;
    # unresolved leaves must still contribute a conservative finite lower gap.
    assert np.isfinite(lower)
    assert lower <= -0.02
    with pytest.raises(MemoryError, match="search cell budget exhausted"):
        _kernel(a, b, certify=False, cells=1).geometry_gap_lower(
            *_pose(a), *_pose(b),
        )
    with pytest.raises(MemoryError, match="search depth budget exhausted"):
        _kernel(a, b, certify=False, depth=0).geometry_gap_lower(
            *_pose(a), *_pose(b),
        )


@pytest.mark.parametrize("malformation", ("nan", "shape", "rotation", "collapsed"))
def test_malformed_physical_pose_is_rejected(malformation):
    a, b = _reference(0.0), _reference(0.3)
    pose = _pose(a)
    if malformation == "nan":
        pose[0][0, 0] = np.nan
    elif malformation == "shape":
        pose[1] = np.zeros((2, 3))
    elif malformation == "rotation":
        pose[2][0, 0, 0] = 2.0
    else:
        pose[0][1] = pose[0][0]
    with pytest.raises(ValueError):
        _kernel(a, b).geometry_gap_lower(*pose, *_pose(b))


def test_mutated_model_and_topology_are_rejected():
    a, b = _reference(0.0), _reference(0.3)
    kernel = _kernel(a, b)
    kernel.node_ids_b = (4, 5, 4)
    with pytest.raises(ValueError, match="whole pair definition changed"):
        kernel.geometry_gap_lower(*_pose(a), *_pose(b))
    kernel = _kernel(a, b)
    kernel.cells_a[0].reference = _reference(0.01)
    with pytest.raises(ValueError, match="whole pair definition changed"):
        kernel.geometry_gap_lower(*_pose(a), *_pose(b))


def test_requested_minimum_uses_complete_bound_without_scan(monkeypatch):
    a, b = _reference(0.0), _reference(0.3)
    kernel = _kernel(a, b, certify=False, depth=0, cells=1)

    def no_scan(self):
        raise AssertionError("certified global gap unnecessarily scanned")

    monkeypatch.setattr(pair_module.NativeTwoCellUnionSearch, "scan", no_scan)
    lower = kernel.geometry_gap_lower(*_pose(a), *_pose(b), minimum_gap_m=0.19)
    assert 0.19 <= lower <= 0.3-0.1


def test_requested_minimum_accepts_ambiguous_remote_distance(monkeypatch):
    a = _reference(0.0, (-0.5, 0.5, 1.5))
    b = _reference(0.2, (-1.0, 0.0, 1.0))
    pose_b = _pose(b)
    pose_b[0][1, 1] = 1.0

    def no_scan(self):
        raise AssertionError("certified ambiguous distance unnecessarily scanned")

    monkeypatch.setattr(pair_module.NativeTwoCellUnionSearch, "scan", no_scan)
    lower = _kernel(a, b).geometry_gap_lower(*_pose(a), *pose_b,
                                           minimum_gap_m=0.0)
    # Closest endpoint (1, .2) lies above A's second segment: true gap .1.
    assert 0.0 <= lower <= 0.1


def test_requested_minimum_falls_back_to_complete_scan(monkeypatch):
    a, b = _reference(0.0), _reference(0.08)
    original_scan = pair_module.NativeTwoCellUnionSearch.scan
    calls = []

    def record_scan(self):
        calls.append(self)
        return original_scan(self)

    monkeypatch.setattr(pair_module.NativeTwoCellUnionSearch, "scan", record_scan)
    lower = _kernel(a, b).geometry_gap_lower(*_pose(a), *_pose(b),
                                           minimum_gap_m=-0.019)
    assert len(calls) == 4
    assert lower <= 0.08-0.1


@pytest.mark.parametrize("cell", (0, 1))
def test_requested_minimum_keeps_rotated_interior_pocket(cell, monkeypatch):
    a = _reference(0.12, (0.0, 1.0, 2.24))
    b = _reference(0.0, (0.0, 1.12, 2.24))
    pose = _pose(a)
    pose[2][cell] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()

    def no_scan(self):
        raise AssertionError("complete pocket bound unnecessarily scanned")

    monkeypatch.setattr(pair_module.NativeTwoCellUnionSearch, "scan", no_scan)
    lower = _kernel(a, b).geometry_gap_lower(*pose, *_pose(b),
                                           minimum_gap_m=-0.04)
    assert -0.04 <= lower <= 0.09-0.1


@pytest.mark.parametrize("minimum", (True, "0", np.nan, np.inf, -np.inf))
def test_requested_minimum_must_be_finite_scalar(minimum):
    a, b = _reference(0.0), _reference(0.3)
    with pytest.raises(ValueError, match="minimum gap must be finite"):
        _kernel(a, b).geometry_gap_lower(*_pose(a), *_pose(b),
                                        minimum_gap_m=minimum)


def test_requested_minimum_still_rejects_collapsed_physical_geometry():
    a, b = _reference(0.0), _reference(0.3)
    pose = _pose(a)
    pose[0][1] = pose[0][0]
    with pytest.raises(ValueError, match="endpoints must be distinct"):
        _kernel(a, b).geometry_gap_lower(*pose, *_pose(b), minimum_gap_m=-1.0)
