"""Independent raw-operator references and bounded-native refusal contracts."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from anysolver.floating_transient import (
    FloatingTransientSpec, FloatingTransientError, BoundedNativeAdmissionError,
    solve_floating_transient, numeric_array_digest, bounded_native_admission,
)


def spec(*, skew=1.e-4, duration=2., damping=.2):
    times = np.arange(round(duration / .01) + 1) * .01
    A = np.eye(6) * .2
    A[0, 1], A[1, 0] = skew, -skew
    kernel = np.zeros((11, 6, 6))
    return FloatingTransientSpec(times=times, load_components={'external': np.zeros((len(times), 6))},
        mass_dry=np.eye(6), added_mass_infinite=A, radiation_kernel=kernel, kernel_horizon=.1,
        stiffness=np.diag(np.arange(1., 7.)), damping=np.eye(6) * damping,
        initial_displacement=np.array([.1, 0., 0., 0., 0., 0.]), hydro_admission='bounded_native',
        admission_omegas=np.array([.1, 1., 10.]),
        native_source_binding=dict(kernel_digest=numeric_array_digest(kernel),
                                   ainf_digest=numeric_array_digest(A), kernel_dt=.01))


def rebind(s, **changes):
    updated = replace(s, **changes)
    return replace(updated, native_source_binding=dict(
        kernel_digest=numeric_array_digest(updated.radiation_kernel),
        ainf_digest=numeric_array_digest(updated.added_mass_infinite), kernel_dt=.01))


def test_raw_mass_matches_independent_adaptive_ode_without_symmetrizing():
    s = spec()
    result = solve_floating_transient(s)
    assert result.complete, result.message
    M = s.mass_dry + s.added_mass_infinite
    def rhs(t, y):
        return np.r_[y[6:], np.linalg.solve(M, -s.damping @ y[6:] - s.stiffness @ y[:6])]
    oracle = solve_ivp(rhs, (0., 2.), np.r_[s.initial_displacement, np.zeros(6)],
                       method='DOP853', rtol=1.e-12, atol=1.e-14, t_eval=s.times)
    assert oracle.success
    assert np.max(np.abs(result.displacements - oracle.y[:6].T)) / .1 < 1.e-3
    assert np.max(np.abs(result.displacements[:, 1])) > 1.e-7
    assert result.max_scaled_residual <= 1.e-10
    assert result.hydrodynamic_admission_report['mass']['rho'] < 1.e-3
    assert len(result.hydrodynamic_work['reference_energy']) == len(s.times)


def test_strict_default_keeps_its_hydrodynamic_symmetry_refusal():
    with pytest.raises(FloatingTransientError, match='symmetric'):
        solve_floating_transient(replace(spec(), hydro_admission='strict'))


def test_binding_and_fixed_mass_skew_ceiling_cannot_be_bypassed():
    with pytest.raises(BoundedNativeAdmissionError, match='skew'):
        solve_floating_transient(spec(skew=.01))
    s = spec()
    with pytest.raises(BoundedNativeAdmissionError, match='binding'):
        solve_floating_transient(replace(s, native_source_binding=dict(s.native_source_binding, kernel_dt=.02)))
    with pytest.raises(FloatingTransientError, match='positive definite'):
        solve_floating_transient(rebind(s, added_mass_infinite=-2. * np.eye(6)))


def test_kernel_skew_is_bounded_against_a_global_scale():
    s = spec()
    K = np.repeat(np.eye(6)[None], 11, axis=0)
    K[:-1, 0, 1], K[:-1, 1, 0] = .01, -.01
    K[-1] = 0.
    with pytest.raises(BoundedNativeAdmissionError, match='kernel skew'):
        solve_floating_transient(rebind(s, radiation_kernel=K))


def test_near_degenerate_flutter_counterexample_is_refused():
    s = spec(skew=1.e-4, damping=0.)
    K = np.diag([1., 1.00001, 3., 4., 5., 6.])
    with pytest.raises(BoundedNativeAdmissionError, match='pencil') as exc:
        solve_floating_transient(replace(s, stiffness=K))
    assert exc.value.report['pencil']['status'] == 'refused'


def test_cumulative_absolute_skew_work_refuses_and_excludes_the_failed_row():
    s = spec(skew=.001, duration=20., damping=0.)
    s = replace(s, stiffness=np.diag([1., 4., 6., 8., 10., 12.]),
                initial_displacement=np.array([1., 0., 0., 0., 0., 0.]),
                initial_velocity=np.array([0., 1., 0., 0., 0., 0.]))
    result = solve_floating_transient(s)
    assert result.status == 'native_hydrodynamic_work_exceeded'
    assert result.n_accepted == result.failed_step < len(s.times)
    assert result.failed_row_diagnostics['native_hydrodynamics']['skew_fraction'] > 1.e-3
    assert len(result.hydrodynamic_work['absolute_skew_work']) == result.n_accepted


def test_cumulative_negative_radiation_work_is_not_hidden_by_a_small_band_margin():
    s = spec(skew=0., duration=20., damping=0.)
    K = np.zeros((11, 6, 6)); K[0] = -.01 * np.eye(6)
    s = rebind(s, radiation_kernel=K, stiffness=np.zeros((6, 6)),
               initial_displacement=np.zeros(6), initial_velocity=np.array([1., 0., 0., 0., 0., 0.]))
    result = solve_floating_transient(s)
    assert result.status == 'native_hydrodynamic_work_exceeded'
    assert result.failed_row_diagnostics['native_hydrodynamics']['negative_radiation_fraction'] > 1.e-3


def test_negative_aerodynamic_damping_shared_by_reference_is_physical():
    s = spec(duration=1., damping=.1)
    s = replace(s, stiffness=np.diag([10., 20., 30., 40., 50., 60.]),
        initial_displacement=np.array([.001, 0., 0., 0., 0., 0.]), wind_times=s.times,
        wind_velocity=np.tile([10., 0., 0.], (len(s.times), 1)), Q0=np.eye(3),
        hub_offset_body=np.zeros(3), rotor_axis_body=np.array([1., 0., 0.]),
        thrust_speeds=np.array([0., 20.]), thrust_values=np.array([5., 1.]), mean_thrust=3.)
    result = solve_floating_transient(s)
    assert result.complete, result.message
    assert result.negative_damping_steps == len(s.times)


def test_native_cancellation_and_zero_load_energy_scales():
    s = replace(spec(), initial_displacement=np.zeros(6))
    accepted = []
    result = solve_floating_transient(s, progress_callback=lambda i, n: accepted.append(i),
                                    cancellation_token=lambda: len(accepted) >= 4)
    assert result.cancelled and result.n_accepted == 4
    assert np.array_equal(result.hydrodynamic_work['reference_energy'], np.zeros(4))
    assert np.array_equal(result.hydrodynamic_work['skew_fraction'], np.zeros(4))


def test_native_mode_preserves_sample_caps():
    s = spec()
    with pytest.raises(FloatingTransientError, match='1001'):
        solve_floating_transient(rebind(s, radiation_kernel=np.zeros((1002, 6, 6))))
    with pytest.raises(FloatingTransientError, match='513'):
        solve_floating_transient(replace(s, admission_omegas=np.arange(1., 515.)))


def test_mass_whitened_admission_is_invariant_under_coordinate_units():
    s = spec()
    def report(M, A, K, C):
        return bounded_native_admission(mass_dry=M, added_mass_infinite=A, radiation_kernel=s.radiation_kernel,
            dt=.01, stiffness=K, damping=C, admission_omegas=s.admission_omegas,
            native_source_binding=dict(s.native_source_binding, ainf_digest=numeric_array_digest(A)))
    a = report(s.mass_dry, s.added_mass_infinite, s.stiffness, s.damping)
    D = np.diag([1.e-3] * 3 + [100.] * 3)
    b = report(D @ s.mass_dry @ D, D @ s.added_mass_infinite @ D,
               D @ s.stiffness @ D, D @ s.damping @ D)
    assert b['mass']['rho'] == pytest.approx(a['mass']['rho'], rel=1.e-12)
    assert a['pencil']['status'] == b['pencil']['status'] == 'admitted'
