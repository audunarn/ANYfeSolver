"""Independent probe parity for the private, unqualified L1 local operator."""

import numpy as np
import pytest
from dataclasses import replace

from anyfesolver._stiffened_shell_l1_local_experimental import (
    L1LocalOperator, NativeHalfCellPort,
)
from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts.b3_e4_f03_l1_native_connection_probe import L1LogConnection


def _pair():
    beam = build(4, z_offset=.0675, assemble=False)
    element, cell, weight, _ = beam._locate(.95)
    reference = np.array(((.9, 0., 0.), (1., 0., 0.)))
    state = beam.context.initial.mechanical
    old_port = NativeHalfCellSpatialPort(beam.context, element, cell, weight)
    native_reference = beam.context.elements[element][1].operator.reference
    new_port = NativeHalfCellPort.from_reference(
        native_reference, beam.context.nodes[element], element, cell, weight,
        beam.context.nodal_count, beam.context.count)
    old = L1LogConnection.at_reference(
        reference, np.zeros((2, 6)), .035, old_port, state)
    new = L1LocalOperator.at_reference(
        reference, np.zeros((2, 6)), .035, new_port, state)
    return beam, old, new, reference


def test_reference_and_finite_local_work_match_independent_probe():
    beam, old, new, _reference = _pair()
    rng = np.random.default_rng(517)
    beam_step = rng.normal(size=beam.context.count)*.002
    beam_step[:6] = 0.
    states = (beam.context.initial.mechanical,
              beam.context.advance(beam.context.initial.mechanical,
                                   beam_step))
    totals = (np.zeros((2, 6)),
              np.array(((.002, -.001, .0003, .2, -.1, .04),
                        (.003, .002, -.0004, .13, .09, -.03))))
    multipliers = (np.zeros(6), np.array((120., -85., 150., 5., -7., 11.)))
    for state, total, multiplier in zip(states, totals, multipliers):
        value, gap, residual, hessian = old.work_tangent(
            total, state, multiplier)
        local = new.evaluate(total, state, multiplier)
        slots = np.r_[np.arange(12), 12+local.beam_slots.astype(int)]
        np.testing.assert_allclose(local.gap, gap, atol=2e-13)
        np.testing.assert_allclose(local.value, value, atol=2e-11)
        np.testing.assert_allclose(local.gradient, residual[slots], atol=2e-11)
        np.testing.assert_allclose(local.hessian, hessian[np.ix_(slots, slots)],
                                   atol=2e-9)
        assert local.hessian.shape == (21, 21)
        assert local.gap_jacobian.shape == (6, 21)
        assert not np.any(residual[np.setdiff1d(np.arange(residual.size), slots)])
        np.testing.assert_allclose(local.hessian, local.hessian.T, atol=2e-12)


def test_local_jacobian_is_the_exact_multiplier_work_gradient():
    beam, _old, new, _reference = _pair()
    state = beam.context.initial.mechanical
    total = np.zeros((2, 6))
    multiplier = np.array((3., -7., 11., 2., 5., -13.))
    local = new.evaluate(total, state, multiplier)
    np.testing.assert_allclose(local.gap, 0., atol=2e-15)
    np.testing.assert_allclose(local.gradient,
                               local.gap_jacobian.T@multiplier,
                               rtol=0., atol=2e-13)
    direction = np.random.default_rng(19).normal(size=21)
    step = 1e-6
    def gap_at(signed_step):
        advanced = np.zeros(beam.context.count)
        advanced[local.beam_slots.astype(int)] = signed_step*direction[12:]
        trial_state = beam.context.advance(state, advanced)
        return new.evaluate(total+signed_step*direction[:12].reshape(2, 6),
                            trial_state, np.zeros(6)).gap
    numerical = (gap_at(step)-gap_at(-step))/(2*step)
    np.testing.assert_allclose(numerical, local.gap_jacobian@direction,
                               rtol=2e-6, atol=1e-8)


def test_local_operator_owns_reference_and_rejects_degenerate_trace():
    beam, _old, new, reference = _pair()
    original = new.evaluate(np.zeros((2, 6)), beam.context.initial.mechanical,
                            np.ones(6)).gap.copy()
    reference[:] = 100.
    np.testing.assert_array_equal(new.evaluate(
        np.zeros((2, 6)), beam.context.initial.mechanical, np.ones(6)).gap,
        original)
    with pytest.raises(ValueError, match="degenerate projected E4"):
        L1LocalOperator.at_reference(
            np.array(((0., 0., 0.), (0., 0., 1.))), np.zeros((2, 6)),
            .035, new.port, beam.context.initial.mechanical)


def test_native_station_preserves_large_origin_low_part_without_context_runtime():
    beam, old, new, _reference = _pair()
    context = beam.context
    initial = context.initial.mechanical
    left, right = new.port.nodes
    positions = initial.positions.copy()
    positions[left, 0] = 1e16
    positions[right, 0] = 1e16+2.
    lows = initial.position_low.copy()
    lows[right, 0] = .125
    state = context.make(replace(initial, positions=positions,
                                 position_low=lows).descriptor())
    expected_high, expected_low, expected_frame = old.port.pose_pair(state)
    actual_high, actual_low, actual_frame = new.port.pose_pair(state)
    np.testing.assert_array_equal(actual_high, expected_high)
    np.testing.assert_allclose(actual_low, expected_low, rtol=0., atol=1e-16)
    np.testing.assert_array_equal(actual_frame, expected_frame)
    # The local operator does not call the bounded development Context again.
    context.guard = lambda: (_ for _ in ()).throw(RuntimeError("expired"))
    np.testing.assert_array_equal(new.port.pose_pair(state)[0], expected_high)


def test_large_origin_native_low_part_changes_local_gap_and_work():
    beam, _old, new, _reference = _pair()
    context = beam.context
    initial = context.initial.mechanical
    left, right = new.port.nodes
    positions = initial.positions.copy()
    positions[left, 0] = 1e16
    positions[right, 0] = 1e16+2.
    base = context.make(replace(initial, positions=positions).descriptor())
    shell_reference = np.array(((1e16, 0., 0.), (1e16+2., 0., 0.)))
    operator = L1LocalOperator.at_reference(
        shell_reference, np.zeros((2, 6)), .035, new.port, base)
    dual = np.array((10., 0., 0., 0., 0., 0.))
    first = operator.evaluate(np.zeros((2, 6)), base, dual)
    np.testing.assert_allclose(first.gap, 0., atol=2e-15)
    lows = base.position_low.copy()
    lows[right, 0] = .125
    changed = context.make(replace(base, position_low=lows).descriptor())
    second = operator.evaluate(np.zeros((2, 6)), changed, dual)
    expected = new.port.weight*.125
    np.testing.assert_allclose(second.gap[0], expected, rtol=0., atol=3e-15)
    np.testing.assert_allclose(second.value-first.value, 10.*expected,
                               rtol=0., atol=3e-14)
