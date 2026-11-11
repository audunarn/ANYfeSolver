"""Bounded F06 chart derivatives of both native B3 section families."""

from hashlib import sha256

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_p5.chart import exp_chart_terms
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_retained_fibre_program import solve_force_program
from anyfesolver._ge_beam3_retained_fibre_state import Context as FibreContext
from anyfesolver._ge_beam3_retained_generalized_state import (
    Context as GeneralizedContext, Program as GeneralizedProgram,
)
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._stiffened_shell_b3_chart import pullback_sparse_trial
from anyfesolver._stiffened_shell_b3_local_adapter import evaluate_sparse_trial
from scripts._b3_e4_f03_native_fixture import build
from scripts.b3_e4_f03_square_section_control import square_torsion_constant
from test_ge_beam3_retained_fibre import make_model


def _rotation_blocks(context):
    blocks = [6 * i + 3 for i in range(len(context.node_ids))]
    for i in range(len(context.elements)):
        blocks.extend((context.nodal_count + 24 * i,
                       context.nodal_count + 24 * i + 3))
    return blocks


def _chart_map(context, q, residual):
    chart = np.eye(context.count)
    curvature = np.zeros((context.count, context.count))
    for start in _rotation_blocks(context):
        block = slice(start, start + 3)
        a, da = exp_chart_terms(q[block])
        chart[block, block] = a
        for k in range(3):
            curvature[block, start + k] = da[:, :, k].T @ residual[block]
    return chart, curvature


def _advance_from_chart(context, base_state, base_q, target_q):
    step = target_q - base_q
    for start in _rotation_blocks(context):
        block = slice(start, start + 3)
        relative = (Rotation.from_rotvec(target_q[block])
                    * Rotation.from_rotvec(base_q[block]).inv())
        step[block] = relative.as_rotvec()
    return context.advance(base_state, step)


@pytest.mark.parametrize("distributed_load", (False, True))
def test_native_generalized_chart_tangent_includes_residual_curvature(
        distributed_load):
    fixture = build(4, z_offset=.0675,
                    torsion_constant=square_torsion_constant())
    context = fixture.context
    if distributed_load:
        load = DistributedPattern(
            LinePattern(((1, .09, -.03, .02),)),
            ((1, .07, -.04, .05),))
        context = GeneralizedContext(
            fixture.model, GeneralizedProgram((1.,), load))
    state, origin = context.initial.mechanical, context.initial.origins
    q = np.zeros(context.count)
    free_node = context.nodal_count - 6
    q[free_node:free_node + 6] = (.0004, -.0003, .0002, .16, -.11, .13)
    q[context.nodal_count:context.nodal_count + 7] = (
        .09, -.07, .06, -.05, .08, .04, .0001)
    q[np.asarray(context.fixed, dtype=int)] = 0.
    state = context.advance(state, q)

    direction = np.zeros(context.count)
    direction[free_node:free_node + 6] = (.4, -.3, .2, .15, -.1, .12)
    direction[context.nodal_count:context.nodal_count + 7] = (
        .11, -.09, .07, .05, -.04, .03, .015)
    direction[np.asarray(context.fixed, dtype=int)] = 0.

    base = evaluate_sparse_trial(context, state, origin, .25,
                                 family="generalized")
    chart, curvature = _chart_map(context, q, base.residual)
    transformed = pullback_sparse_trial(context, base, q)
    np.testing.assert_allclose(transformed.residual, chart.T@base.residual,
                               rtol=2e-14, atol=2e-12)
    np.testing.assert_allclose(
        transformed.tangent.toarray(),
        chart.T@base.tangent.toarray()@chart+curvature,
        rtol=2e-13, atol=2e-11)
    assert transformed.candidate_histories is base.candidate_histories
    assert not np.shares_memory(transformed.residual, base.residual)
    assert not np.shares_memory(transformed.tangent.data, base.tangent.data)
    uncorrected = chart.T @ (base.tangent @ (chart @ direction))
    expected = transformed.tangent @ direction
    assert np.linalg.norm(curvature @ direction) > 1e-7
    errors = []
    for h in (1e-5, 3e-6, 1e-6):
        trial = []
        for sign in (1., -1.):
            target_q = q + sign * h * direction
            target_state = _advance_from_chart(context, state, q, target_q)
            response = evaluate_sparse_trial(context, target_state, origin, .25,
                                             family="generalized")
            trial.append(pullback_sparse_trial(
                context, response, target_q).residual)
        actual = (trial[0] - trial[1]) / (2 * h)
        errors.append(np.linalg.norm(actual - expected)
                      / max(1., np.linalg.norm(expected)))
    assert sum(error <= 1e-6 for error in errors) >= 2, errors
    uncorrected_error = (np.linalg.norm(actual - uncorrected)
                         / max(1., np.linalg.norm(expected)))
    assert uncorrected_error > 10 * errors[-1]
    with pytest.raises(ValueError, match="finite native B3 chart"):
        pullback_sparse_trial(context, base, np.full(context.count, np.nan))
    outside_chart = q.copy()
    outside_chart[free_node+3] = .9*np.pi
    with pytest.raises(ValueError, match="cutback"):
        pullback_sparse_trial(context, base, outside_chart)


def test_native_physical_fibre_elastic_chart_tangent_uses_same_sparse_contract():
    context = FibreContext(
        make_model(contrast=1.),
        ForceProgram((.25,), ((5, .4, -.005, 0.),), max_iterations=24))
    layout = context.layout
    state, origin = context.initial.mechanical, context.initial.origins
    direction = np.zeros(layout.count)
    free_node = layout.nodal_count-6
    direction[free_node:free_node+6] = (.4, -.3, .2, .15, -.1, .12)
    direction[layout.nodal_count:layout.nodal_count+7] = (
        .11, -.09, .07, .05, -.04, .03, .015)
    direction[np.asarray(layout.fixed, dtype=int)] = 0.
    q = 1e-3*direction
    state = layout.advance(state, q)
    base = evaluate_sparse_trial(context, state, origin, .25,
                                 family="physical_fibre")
    # The dense fibre owner returns an Exp-chart energy Hessian.  At a new
    # zero-coordinate chart over this nonzero-moment state, the sparse
    # spatial-Jacobian pullback must recover that Hessian exactly.
    dense_hessian = context.assemble(state, .25, origin)[1]
    zero_chart = pullback_sparse_trial(layout, base, np.zeros(layout.count))
    starts = _rotation_blocks(layout)
    start = max(starts, key=lambda s: np.linalg.norm(base.residual[s:s+3]))
    block = slice(start, start+3)
    moment = base.residual[block]
    assert np.linalg.norm(moment) > 1e-6
    assert np.linalg.norm(base.tangent[block, block].toarray()
                          -dense_hessian[block, block]) > 1e-6
    np.testing.assert_allclose(
        zero_chart.tangent[block, block].toarray(),
        dense_hessian[block, block], rtol=2e-13, atol=2e-11)
    focused = np.zeros(layout.count)
    focused[start+int(np.argmin(abs(moment)))] = 1.
    focused_errors = []
    for h in (1e-5, 3e-6, 1e-6):
        residuals = []
        for sign in (1., -1.):
            target_q = sign*h*focused
            target_state = layout.advance(state, target_q)
            trial = evaluate_sparse_trial(context, target_state, origin, .25,
                                          family="physical_fibre")
            residuals.append(pullback_sparse_trial(
                layout, trial, target_q).residual[block])
        actual = (residuals[0]-residuals[1])/(2*h)
        expected = (zero_chart.tangent@focused)[block]
        focused_errors.append(np.linalg.norm(actual-expected)
                              / max(1., np.linalg.norm(expected)))
    assert sum(error <= 1e-6 for error in focused_errors) >= 2, focused_errors
    chart, curvature = _chart_map(layout, q, base.residual)
    pulled = pullback_sparse_trial(layout, base, q)
    np.testing.assert_allclose(pulled.residual, chart.T@base.residual,
                               rtol=2e-14, atol=2e-12)
    np.testing.assert_allclose(
        pulled.tangent.toarray(),
        chart.T@base.tangent.toarray()@chart+curvature,
        rtol=2e-13, atol=2e-11)
    expected = pulled.tangent@direction
    errors = []
    for h in (1e-5, 3e-6, 1e-6):
        residuals = []
        for sign in (1., -1.):
            target_q = q+sign*h*direction
            target_state = _advance_from_chart(layout, state, q, target_q)
            trial = evaluate_sparse_trial(context, target_state, origin, .25,
                                          family="physical_fibre")
            residuals.append(pullback_sparse_trial(
                layout, trial, target_q).residual)
        actual = (residuals[0]-residuals[1])/(2*h)
        errors.append(np.linalg.norm(actual-expected)
                      / max(1., np.linalg.norm(expected)))
    assert sum(error <= 1e-6 for error in errors) >= 2, errors


def test_native_physical_fibre_smooth_active_chart_keeps_accepted_predecessor():
    program = ForceProgram((.25, .5, 1., .5, 0., -.5, 0.),
                           ((5, .4, -.005, 0.),), max_iterations=24)
    accepted = solve_force_program(make_model(contrast=1.), program,
                                   stop_after=3)
    assert accepted.status == "paused" and accepted.completed_targets == 3
    checksum = sha256(accepted.checkpoint).hexdigest()
    context = FibreContext(make_model(contrast=1.), program)
    predecessor, _ = context.restore(
        accepted.checkpoint, expected_sha256=checksum)
    origin = predecessor.histories
    before = canonical(origin)
    state_before = canonical(predecessor.mechanical.descriptor())
    layout = context.layout
    direction = np.zeros(layout.count)
    free_node = layout.nodal_count-6
    direction[free_node:free_node+6] = (.4, -.3, .2, .15, -.1, .12)
    direction[layout.nodal_count:layout.nodal_count+7] = (
        .11, -.09, .07, .05, -.04, .03, .015)
    direction[np.asarray(layout.fixed, dtype=int)] = 0.
    q = 1e-4*direction
    active_state = layout.advance(predecessor.mechanical, q)

    def active_rows(candidate):
        return frozenset(
            (cell_i, station_i, row_i)
            for cell_i, (made_cell, old_cell) in enumerate(zip(candidate, origin))
            for station_i, (made_station, old_station) in enumerate(zip(
                made_cell.stations, old_cell.stations))
            for row_i, (made_row, old_row) in enumerate(zip(
                made_station.rows, old_station.rows))
            if made_row[2]-old_row[2] > 1e-12)

    base = evaluate_sparse_trial(context, active_state, origin, 1.,
                                 family="physical_fibre")
    active_set = active_rows(base.candidate_histories)
    assert active_set
    pulled = pullback_sparse_trial(layout, base, q)
    expected = pulled.tangent@direction
    errors = []
    for h in (1e-5, 3e-6, 1e-6):
        residuals = []
        for sign in (1., -1.):
            target_q = q+sign*h*direction
            target_state = _advance_from_chart(layout, active_state, q,
                                               target_q)
            trial = evaluate_sparse_trial(context, target_state, origin, 1.,
                                          family="physical_fibre")
            assert active_rows(trial.candidate_histories) == active_set
            residuals.append(pullback_sparse_trial(
                layout, trial, target_q).residual)
        actual = (residuals[0]-residuals[1])/(2*h)
        errors.append(np.linalg.norm(actual-expected)
                      / max(1., np.linalg.norm(expected)))
    assert sum(error <= 1e-5 for error in errors) >= 2, errors
    assert canonical(origin) == before
    assert canonical(predecessor.mechanical.descriptor()) == state_before
    assert sha256(accepted.checkpoint).hexdigest() == checksum
