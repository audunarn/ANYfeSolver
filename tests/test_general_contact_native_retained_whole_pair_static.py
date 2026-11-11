"""Distinct whole-curve profile through the retained two-native-beam owner."""

import os

import numpy as np
import pytest
from scipy.integrate import quad

from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver._ge_beam3_generalized_ellipsoid_section import (
    EllipsoidalGeneralizedSection,
)
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_element import (
    NativeGeneralizedStaticElement,
)
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_nodal_loading import NodalDeadForces, Program
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.general_contact.native_retained_pair_static import (
    NativeRetainedRoundPairAnalysis, WHOLE_PROFILE, WHOLE_CHECKPOINT_SCHEMA,
)
from anyfesolver.general_contact.native_retained_whole_pair import (
    NativeRetainedWholeCurvePair,
)


def _reference(y):
    return CenteredCurvedBeam3ReferenceGeometry(
        np.array(((0.0, y, 0.0), (1.0, y, 0.0), (2.0, y, 0.0))),
        np.tile(np.eye(3), (3, 1, 1)),
    )


def _owner(*, separation=0.3, force=0.0, node_ids_b=(4, 5, 6),
           targets=(0.0, 1.0), maximum_swept_intervals=128):
    a, b = _reference(0.0), _reference(separation)
    section = EllipsoidalGeneralizedSection(
        np.diag((1000.0, 400.0, 350.0, 0.8, 1.0, 1.2)),
        np.eye(6), 1e6, 1.0,
    )
    inertia = np.diag((2.0, 2.0, 2.0, 0.07, 0.09, 0.11))
    first = NativeGeneralizedStaticElement(
        11, (1, 2, 3), a, section, order=4,
    )
    second = NativeGeneralizedStaticElement(
        12, (4, 5, 6), b, section, order=4,
    )
    definitions = (NativeBeamDefinition.capture(first, inertia),
                   NativeBeamDefinition.capture(second, inertia))
    axes = {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")}
    boundaries = (BoundaryCondition("first-root", [1], axes),
                  BoundaryCondition("second-root", [4], axes))
    program = Program(
        targets, DistributedPattern(LinePattern(()), ()),
        NodalDeadForces(((6, 0.0, force, 0.0),)),
        max_iterations=24,
    )
    contact = NativeRetainedWholeCurvePair(
        11, a, (1, 2, 3), 0.05,
        12, b, node_ids_b, 0.05,
        line_stiffness_n_per_m2=100.0,
        distance_tolerance_m=1e-10,
        feature_tolerance_m=1e-9,
        search_parameter_tolerance=1e-3,
        maximum_search_depth=20,
        maximum_search_cells=10000,
        energy_tolerance_j=1e-8,
        force_tolerance_n=1e-7,
        tangent_tolerance_n_per_m=1e-6,
        maximum_integration_intervals=1000,
    )
    return NativeRetainedRoundPairAnalysis(
        definitions, boundaries, program, contact,
        gap_tolerance_m=1e-3, equilibrium_tolerance=1e-8,
        maximum_swept_intervals=maximum_swept_intervals,
    )


def test_whole_pair_open_static_route_and_checkpoint_restart():
    owner = _owner()
    full = owner.solve()
    assert full.status == "completed", full.failure
    assert full.profile == WHOLE_PROFILE
    assert full.accepted.contact.potential_j == 0.0
    assert full.accepted.completed_targets == 2
    part = _owner().solve(stop_after=1)
    assert part.status == "paused", part.failure
    assert WHOLE_CHECKPOINT_SCHEMA.encode() in part.checkpoint
    resumed = _owner().solve(checkpoint=part.checkpoint,
                             expected_sha256=part.checkpoint_sha256)
    assert resumed.status == "completed", resumed.failure
    assert resumed.checkpoint == full.checkpoint


@pytest.mark.skipif(
    os.environ.get("ANYSOLVER_LONG_CONTACT") != "1",
    reason="opt-in long-running active native whole-pair development route",
)
def test_whole_pair_active_static_path_restart_and_wrench():
    settings = dict(separation=0.13, force=-0.05,
                    targets=(0.0, 0.1, 0.2, 0.3))
    first = _owner(**settings).solve(stop_after=3)
    assert first.status == "paused", first.failure
    assert first.accepted.completed_targets == 3
    resumed = _owner(**settings).solve(
        checkpoint=first.checkpoint,
        expected_sha256=first.checkpoint_sha256,
    )
    assert resumed.status == "completed", resumed.failure
    contact = resumed.accepted.contact
    assert contact.potential_j > 0.0
    assert contact.conservative_lower_gap_m >= -1e-3
    np.testing.assert_allclose(
        contact.beam_a_force_n + contact.beam_b_force_n,
        0.0, rtol=0.0, atol=1e-8,
    )
    np.testing.assert_allclose(
        contact.beam_a_moment_nm + contact.beam_b_moment_nm,
        0.0, rtol=0.0, atol=1e-8,
    )
    full = _owner(**settings).solve()
    assert full.status == "completed", full.failure
    assert resumed.checkpoint == full.checkpoint


def test_whole_pair_requires_model_bound_node_identities():
    with pytest.raises(ValueError, match="material node identities"):
        _owner(node_ids_b=(4, 7, 6))


def test_swept_endpoint_certificate_skips_safe_midpoint_work(monkeypatch):
    owner = _owner(separation=0.3)
    start = owner.initial.mechanical
    increment = np.zeros(owner.context.count)
    increment[6*owner.context.index[6]+1] = -0.01
    end = owner.context.advance(start, increment)
    start_gap = owner.initial.contact.conservative_lower_gap_m
    end_gap = owner._contact(end).conservative_lower_gap_m
    original = owner._contact
    calls = []

    def counted(state):
        calls.append(state)
        return original(state)

    monkeypatch.setattr(owner, "_contact", counted)
    assert owner._swept_admissible(start, increment, end,
                                  start_gap, end_gap)
    assert calls == []


def test_swept_endpoint_certificate_still_rejects_penetrating_end():
    owner = _owner(separation=0.13)
    start = owner.initial.mechanical
    increment = np.zeros(owner.context.count)
    increment[6*owner.context.index[5]+1] = -0.025
    increment[6*owner.context.index[6]+1] = -0.05
    end = owner.context.advance(start, increment)
    end_gap = owner._contact(end).conservative_lower_gap_m
    assert end_gap < -owner.gap_tolerance
    assert not owner._swept_admissible(
        start, increment, end,
        owner.initial.contact.conservative_lower_gap_m, end_gap,
    )


def test_kinked_whole_pair_near_contact_load_path_completes():
    run = _owner(separation=0.102, force=-0.002,
                 targets=(0.0, 1.0)).solve()
    assert run.status == "completed", run.failure
    assert run.accepted.completed_targets == 2
    contact = run.accepted.contact
    assert contact.potential_j > 0.0
    assert contact.conservative_lower_gap_m >= -1e-3
    np.testing.assert_allclose(
        contact.beam_a_force_n+contact.beam_b_force_n,
        0.0, rtol=0.0, atol=1e-8,
    )
    np.testing.assert_allclose(
        contact.beam_a_moment_nm+contact.beam_b_moment_nm,
        0.0, rtol=0.0, atol=1e-8,
    )


def test_active_affine_pose_assembles_both_retained_beams_and_tangent():
    owner = _owner(separation=0.13)
    increment = np.zeros(owner.context.count)
    increment[6*owner.context.index[5]+1] = -0.025
    increment[6*owner.context.index[6]+1] = -0.05
    state = owner.context.advance(owner.initial.mechanical, increment)
    residual, tangent, _, _, contact = owner._assemble(
        state, 0.0, owner.initial.histories,
    )
    assert contact.potential_j > 0.0
    assert np.linalg.norm(contact.residual[:42]) > 0.0
    assert np.linalg.norm(contact.residual[42:]) > 0.0
    assert np.linalg.norm(contact.tangent[:42, 42:]) > 0.0
    np.testing.assert_allclose(contact.beam_a_force_n+contact.beam_b_force_n,
                               0.0, atol=1e-8)
    direction = np.zeros(owner.context.count)
    direction[6*owner.context.index[5]+1] = -0.001
    direction[6*owner.context.index[6]+1] = -0.002
    step = 1e-3
    upper_state = owner.context.advance(state, step*direction)
    lower_state = owner.context.advance(state, -step*direction)
    upper = owner._assemble(upper_state, 0.0, owner.initial.histories)[0]
    lower = owner._assemble(lower_state, 0.0, owner.initial.histories)[0]
    observed = (upper-lower)/(2.0*step)
    predicted = tangent@direction
    assert np.linalg.norm(observed-predicted)/max(
        1.0, np.linalg.norm(predicted),
    ) < 1e-5


def test_unresolved_curved_support_keeps_accepted_target_prefix():
    owner = _owner(separation=0.13, force=-5.0)
    failed = owner.solve()
    assert failed.status == "failed"
    assert "native whole pair line-search/gap limit" in failed.failure
    assert "rejected=" in failed.failure
    assert failed.accepted.completed_targets == 1
    prefix = _owner(separation=0.13, force=-5.0).solve(stop_after=1)
    assert prefix.status == "paused", prefix.failure
    assert failed.checkpoint == prefix.checkpoint


def test_bent_partial_contact_matches_independent_polyline_energy():
    owner = _owner(separation=0.13)
    increment = np.zeros(owner.context.count)
    increment[6*owner.context.index[6]+1] = -0.04
    state = owner.context.advance(owner.initial.mechanical, increment)
    contact = owner._contact(state)
    assert contact.potential_j > 0.0
    assert contact.integration_cells_checked > 0

    positions_a = state.positions[owner.context.nodes[owner.indices[0]]]
    positions_b = state.positions[owner.context.nodes[owner.indices[1]]]
    segments_a = ((positions_a[0], positions_a[1]),
                  (positions_a[1], positions_a[2]))
    segments_b = ((positions_b[0], positions_b[1]),
                  (positions_b[1], positions_b[2]))

    def distance_to_segment(point, segment):
        start, end = segment
        direction = end-start
        station = np.clip(float((point-start)@direction/(direction@direction)),
                          0.0, 1.0)
        return float(np.linalg.norm(point-start-station*direction))

    def directional_energy(source, target):
        total = 0.0
        for start, end in source:
            def density(station):
                point = start+station*(end-start)
                gap = min(distance_to_segment(point, cell)
                          for cell in target)-0.1
                return min(gap, 0.0)**2
            value, _ = quad(density, 0.0, 1.0,
                            epsabs=1e-11, limit=200)
            total += value
        return total

    reference = 25.0*(directional_energy(segments_a, segments_b)
                      + directional_energy(segments_b, segments_a))
    assert contact.potential_j == pytest.approx(reference,
                                                rel=1e-4, abs=1e-8)
    np.testing.assert_allclose(contact.beam_a_force_n+contact.beam_b_force_n,
                               0.0, atol=1e-8)
