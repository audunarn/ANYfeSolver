"""Cheap continuous fixture reference: analytic modes and independent controls."""

import ast
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import pytest

from reference_oblique_q4_plateau import (
    B, D, G, H, MASS, STATE_SCALES, TAU_Y,
    ReferenceResolutionError, _check_grouped_residuals, continuous_reference,
)


def combined(result):
    return np.column_stack((result.q, result.v, result.p, result.alpha))


def test_reference_source_has_no_solver_material_or_builder_imports():
    source = Path(__file__).with_name('reference_oblique_q4_plateau.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names]
    assert all(not name.startswith(('anyfesolver', 'anymaterial', 'tests', 'examples')) for name in modules)


@pytest.mark.parametrize('method', ['exponential', 'dop853'])
def test_free_mass_motion_is_analytic_and_has_no_material_reset(method):
    times = np.linspace(0., .0001, 11)
    result = continuous_reference(times, method=method, contact_enabled=False)
    np.testing.assert_allclose(result.q[:, 1], -times, rtol=0., atol=2.e-18)
    np.testing.assert_array_equal(result.q[:, 0], 0.)
    np.testing.assert_array_equal(result.v, np.tile([0., -1.], (len(times), 1)))
    np.testing.assert_array_equal(result.p, 0.)
    np.testing.assert_array_equal(result.alpha, 0.)
    assert not result.events
    assert result.diagnostics['maximum_energy_balance_error_j'] == 0.
    with pytest.raises(ValueError):
        result.q[0, 0] = 1.


@pytest.mark.parametrize('method', ['exponential', 'dop853'])
def test_elastic_membrane_matches_closed_harmonic_mode(method):
    times = np.linspace(0., .0004, 31)
    amplitude = 1.e-11
    frequency = np.sqrt(4*G*H/MASS[0])
    result = continuous_reference(times, method=method, contact_enabled=False,
                                  initial_q=(amplitude, 0.), initial_v=(0., 0.))
    np.testing.assert_allclose(result.q[:, 0], amplitude*np.cos(frequency*times), rtol=0., atol=2.e-23)
    np.testing.assert_allclose(result.v[:, 0], -amplitude*frequency*np.sin(frequency*times), rtol=0., atol=2.e-20)
    np.testing.assert_array_equal(result.p, 0.)
    np.testing.assert_array_equal(result.alpha, 0.)
    assert not result.events
    assert result.diagnostics['maximum_energy_balance_error_j'] < 1.e-25


@pytest.mark.parametrize('method', ['exponential', 'dop853'])
def test_plateau_loading_then_unloading_matches_piecewise_analytic_solution(method):
    times = np.linspace(0., .002, 51)
    e0, v0 = TAU_Y/(2*G), 1.e-7
    deceleration = 2*H*TAU_Y/MASS[0]
    release = v0/deceleration
    maximum_e = e0+v0*release-.5*deceleration*release**2
    accepted_p = maximum_e-e0
    frequency = np.sqrt(4*G*H/MASS[0])
    loading = times <= release
    expected_e = np.where(loading, e0+v0*times-.5*deceleration*times**2,
                          accepted_p+e0*np.cos(frequency*(times-release)))
    expected_v = np.where(loading, v0-deceleration*times,
                          -e0*frequency*np.sin(frequency*(times-release)))
    expected_p = np.where(loading, expected_e-e0, accepted_p)
    result = continuous_reference(times, method=method, contact_enabled=False,
                                  initial_q=(e0, 0.), initial_v=(v0, 0.))
    np.testing.assert_allclose(result.q[:, 0], expected_e, rtol=0., atol=5.e-23)
    np.testing.assert_allclose(result.v[:, 0], expected_v, rtol=0., atol=1.e-18)
    np.testing.assert_allclose(result.p, expected_p, rtol=0., atol=5.e-23)
    np.testing.assert_allclose(result.alpha, 2*expected_p/np.sqrt(3.), rtol=0., atol=5.e-23)
    assert [event.kind for event in result.events] == ['unloading']
    assert abs(result.events[0].time-release) <= 2.e-14
    assert result.diagnostics['zero_duration_events'] == 0
    assert result.diagnostics['maximum_energy_balance_error_j'] < 1.e-24


@pytest.fixture(scope='module')
def reference_control_study():
    times = np.linspace(0., .003, 101)
    ordinary = continuous_reference(times)
    tighter = continuous_reference(times, maximum_step=.5e-6,
                                   event_time_tolerance=2.e-15)
    independent = continuous_reference(times, method='dop853',
                                       relative_tolerance=2.e-13, absolute_tolerance=2.e-15)
    errors = {
        'tighter_exponential_scaled_maxima': np.max(np.abs(combined(ordinary)-combined(tighter))/STATE_SCALES, axis=0),
        'independent_dop853_scaled_maxima': np.max(np.abs(combined(ordinary)-combined(independent))/STATE_SCALES, axis=0),
    }
    record = dict(scope='development reference control study; not engineering acceptance',
        scaled_component_order=['e', 'w', 've', 'vw', 'p', 'alpha'],
        scales=STATE_SCALES.tolist(), times=times.tolist(),
        errors={key:value.tolist() for key,value in errors.items()},
        methods={name:dict(result.diagnostics) for name,result in (
            ('ordinary', ordinary), ('tighter', tighter), ('independent', independent))},
        events={name:[asdict(event) for event in result.events] for name,result in (
            ('ordinary', ordinary), ('tighter', tighter), ('independent', independent))})
    folder = Path(os.environ.get('RUNNER_TEMP', str(Path(__file__).parent)))
    (folder/'oblique_continuous_reference_controls.json').write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    return ordinary, tighter, independent, errors


def test_frozen_fixture_reference_tighter_and_independent_controls_agree(reference_control_study):
    ordinary, tighter, independent, errors = reference_control_study
    for value in errors.values():
        assert np.max(value) <= 1.e-8
    assert [event.kind for event in ordinary.events] == [event.kind for event in tighter.events]
    assert [event.kind for event in ordinary.events] == [event.kind for event in independent.events]
    for reference in (tighter, independent):
        np.testing.assert_allclose([event.time for event in ordinary.events],
            [event.time for event in reference.events], rtol=0., atol=2.e-13)


def test_contact_closure_and_contact_driven_yield_have_no_hidden_reset(reference_control_study):
    result, _, _, _ = reference_control_study
    first = result.events[0]
    assert first.kind == 'contact_closure' and first.indices == (0,)
    analytic_first_closure = B[0]/D[0, 1]
    assert abs(first.time-analytic_first_closure) <= 2.e-14
    assert any(event.kind.startswith('yield') for event in result.events)
    assert any(event.kind == 'contact_release' for event in result.events)
    assert np.max(result.alpha) < .005 and result.alpha[-1] > 1.e-9
    yielded = np.flatnonzero(result.alpha > 1.e-15)[0]
    assert np.any(result.pressure_active[yielded])
    assert result.p[-1] < 0.
    assert result.diagnostics['maximum_energy_balance_error_j'] < 2.e-8
    assert result.diagnostics['maximum_yield_event_residual_pa'] <= 1.e-10
    assert result.diagnostics['zero_duration_events'] <= 2
    assert all(right.time >= left.time for left,right in zip(result.events, result.events[1:]))


@pytest.mark.parametrize('changes', [
    {'maximum_step': 0.}, {'maximum_step': 1.e-3},
    {'method': 'unknown'}, {'initial_q': (0.,)},
    {'initial_p': 1.e-5}, {'initial_alpha': .005},
    {'maximum_intervals': True}, {'contact_enabled': 1},
])
def test_invalid_reference_domain_is_explicit(changes):
    with pytest.raises(ValueError):
        continuous_reference([0., .0001], **changes)


def test_finite_reference_and_root_budgets_fail_without_partial_result():
    with pytest.raises(ReferenceResolutionError, match='interval budget'):
        continuous_reference([0., .001], maximum_intervals=1)
    with pytest.raises(ReferenceResolutionError, match='root budget'):
        continuous_reference([0., .001], maximum_event_iterations=1)


def test_stationary_yield_touch_departs_elastically_without_zero_time_toggle():
    times = np.linspace(0., .0002, 11)
    result = continuous_reference(times, contact_enabled=False,
        initial_q=(TAU_Y/(2*G), 0.), initial_v=(0., 0.))
    np.testing.assert_array_equal(result.p, 0.)
    np.testing.assert_array_equal(result.alpha, 0.)
    assert not result.events
    assert result.v[-1, 0] < 0.
    assert result.diagnostics['zero_duration_events'] == 0


def test_leaving_plateau_is_rejected_instead_of_clipped_or_reset():
    with pytest.raises(ReferenceResolutionError, match='first plateau'):
        continuous_reference([0., .0001], contact_enabled=False,
            initial_q=(TAU_Y/(2*G), 0.), initial_v=(100., 0.))


@pytest.mark.parametrize('method', ['exponential', 'dop853'])
def test_exact_first_closure_output_retains_incoming_tolerance_phase(method):
    closure = B[0]/D[0, 1]
    result = continuous_reference([0., closure, closure+1.e-7], method=method)
    assert not np.any(result.pressure_active[1])
    np.testing.assert_allclose(result.q[1], [0., -closure], rtol=0., atol=2.e-17)
    assert result.p[1] == result.alpha[1] == 0.
    assert result.pressure_active[2, 0]
    assert result.events[0].kind == 'contact_closure'
    assert abs(result.events[0].time-closure) <= 2.e-14
    assert 'incoming' in result.diagnostics['output_phase_convention']


def test_high_rate_time_near_event_must_meet_own_residual_at_group_time():
    first, second = .1, .1+5.e-15
    rate = 1.e9
    # Both proposals lie within a 1e-14 time window. Their individual roots
    # satisfy their dimensional tolerances, but the fast row is not near zero
    # at the earlier root. Its separate floating allowance is explicit.
    proposals = [(first, np.array([-1., first]), 'contact_closure', 0, 1.e-14, 1.e-17),
                 (second, np.array([-rate, rate*second]), 'contact_closure', 1, 1.e-10, 5.e-8)]
    assert second-first < 1.e-14
    assert _check_grouped_residuals(proposals, np.array([second, 1.])) <= 1.
    with pytest.raises(ReferenceResolutionError, match='grouped continuous event residual'):
        _check_grouped_residuals(proposals, np.array([first, 1.]))
