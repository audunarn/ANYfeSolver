"""Two native B3-GE beams solved with retained distributed pair contact."""

from hashlib import sha256
import json

import numpy as np
import pytest

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
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.general_contact.native_retained_cell_pair import (
    NativeRetainedSelectedCellPair,
)
from anyfesolver.general_contact.native_retained_pair_static import (
    NativeRetainedRoundPairAnalysis,
)
from test_general_contact_native_retained_cell_pair import _references


def _owner(*, force=-0.02, iterations=24, targets=(0.5, 1.0),
           offset=0.13, line_stiffness=1000.0, gap_tolerance=1e-3):
    first_ref, second_source = _references()
    second_points = second_source.coordinates.copy()
    second_points[:, 2] = offset
    second_ref = CenteredCurvedBeam3ReferenceGeometry(
        second_points, second_source.nodal_triads,
    )
    section = EllipsoidalGeneralizedSection(
        np.diag((1000.0, 400.0, 350.0, 0.8, 1.0, 1.2)),
        np.eye(6), 1e6, 1.0,
    )
    inertia = np.diag((2.0, 2.0, 2.0, 0.07, 0.09, 0.11))
    first = NativeGeneralizedStaticElement(
        11, (1, 2, 3), first_ref, section, order=4,
    )
    second = NativeGeneralizedStaticElement(
        12, (4, 5, 6), second_ref, section, order=4,
    )
    definitions = (NativeBeamDefinition.capture(first, inertia),
                   NativeBeamDefinition.capture(second, inertia))
    axes = {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")}
    boundaries = (BoundaryCondition("first-root", [1], axes),
                  BoundaryCondition("second-root", [4], axes))
    program = Program(
        targets, DistributedPattern(LinePattern(()), ()),
        NodalDeadForces(((6, 0.0, 0.0, force),)),
        max_iterations=iterations,
    )
    contact = NativeRetainedSelectedCellPair(
        11, first_ref, 0, 0.05, 12, second_ref, 1, 0.05,
        line_stiffness_n_per_m2=line_stiffness,
        integration_relative_tolerance=1e-5,
        maximum_subdivisions=10,
        gap_accuracy_m=1e-4,
        maximum_gap_cells=10000,
        maximum_gap_depth=28,
    )
    return NativeRetainedRoundPairAnalysis(
        definitions, boundaries, program, contact,
        gap_tolerance_m=gap_tolerance, equilibrium_tolerance=1e-8,
        maximum_swept_intervals=128,
    )


def test_two_native_beams_couple_through_retained_contact():
    owner = _owner()
    result = owner.solve()
    assert result.status == "completed", result.failure
    assert result.accepted.completed_targets == 2
    assert result.accepted.contact.potential_j > 0.0
    assert result.accepted.contact.conservative_lower_gap_m >= -owner.gap_tolerance
    assert np.linalg.norm(result.accepted.contact.residual[18:21]) > 0.0
    assert np.linalg.norm(result.accepted.contact.tangent[:42, 42:]) > 0.0
    np.testing.assert_allclose(
        result.accepted.contact.beam_a_force_n
        + result.accepted.contact.beam_b_force_n,
        0.0, atol=1e-8,
    )
    residual, tangent, metrics, _, _ = owner._assemble(
        result.accepted.mechanical, 1.0, result.accepted.histories,
    )
    assert max(metrics) < owner.equilibrium_tolerance
    assert np.linalg.norm(residual[owner.context.free]) < 1e-8
    direction = np.zeros(owner.context.count)
    direction[14] = 0.003
    direction[owner.context.nodal_count] = -0.002
    direction[owner.context.nodal_count+24+3] = 0.002
    step = 1e-5
    plus = owner.context.advance(result.accepted.mechanical, step*direction)
    minus = owner.context.advance(result.accepted.mechanical, -step*direction)
    upper = owner._assemble(plus, 1.0, result.accepted.histories)[0]
    lower = owner._assemble(minus, 1.0, result.accepted.histories)[0]
    observed = (upper-lower)/(2*step)
    predicted = tangent@direction
    assert np.linalg.norm(observed-predicted)/max(1.0, np.linalg.norm(predicted)) < 1e-5


def test_initial_interference_is_rejected():
    with pytest.raises(ValueError, match="initial native pair gap not certified"):
        _owner(offset=0.08)


def test_failed_target_preserves_accepted_prefix():
    owner = _owner(targets=(0.0, 1.0), iterations=1)
    result = owner.solve()
    assert result.status == "failed"
    assert result.accepted.completed_targets == 1
    np.testing.assert_array_equal(
        result.accepted.mechanical.positions,
        owner.initial.mechanical.positions,
    )
    paused = _owner(targets=(0.0, 1.0), iterations=1).solve(stop_after=1)
    assert paused.status == "paused"
    assert result.checkpoint == paused.checkpoint


def test_two_beam_checkpoint_split_restart_matches_full_run():
    full = _owner().solve()
    assert full.status == "completed", full.failure
    part = _owner().solve(stop_after=1)
    assert part.status == "paused", part.failure
    resumed = _owner().solve(
        checkpoint=part.checkpoint,
        expected_sha256=part.checkpoint_sha256,
    )
    assert resumed.status == "completed", resumed.failure
    assert resumed.checkpoint == full.checkpoint
    np.testing.assert_array_equal(
        resumed.accepted.mechanical.positions,
        full.accepted.mechanical.positions,
    )


def test_two_beam_checkpoint_rejects_changed_definition_and_forgery():
    part = _owner().solve(stop_after=1)
    assert part.status == "paused", part.failure
    with pytest.raises(ValueError, match="definition/settings binding"):
        _owner(line_stiffness=1001.0).solve(
            checkpoint=part.checkpoint,
            expected_sha256=part.checkpoint_sha256,
        )
    with pytest.raises(ValueError, match="external checkpoint hash"):
        _owner().solve(checkpoint=part.checkpoint,
                       expected_sha256="0"*64)
    body = json.loads(part.checkpoint)
    body["records"][0]["contact"]["sampled_gap_m"] += 0.001
    body["checkpoint_sha256"] = sha({
        key: value for key, value in body.items()
        if key != "checkpoint_sha256"
    })
    forged = canonical(body)
    with pytest.raises(ValueError, match="physical replay mismatch"):
        _owner().solve(checkpoint=forged,
                       expected_sha256=sha256(forged).hexdigest())


def test_live_pair_definition_mutation_fails_before_solve():
    owner = _owner()
    owner.contact.law.stiffness += 1.0
    with pytest.raises(ValueError, match="contact definition changed"):
        owner.solve()
