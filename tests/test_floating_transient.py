"""Reference-equation tests for the floating transient solver (wind thrust feedback, windowed memory)."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from anysolver.floating_transient import (
    FloatingTransientError,
    FloatingTransientResourceConfig,
    FloatingTransientSpec,
    solve_floating_transient,
    waterline_guard,
)
from anysolver.marine_linear import MarineLinearSpec, solve_marine_linear


def _grid(dt, duration):
    n = int(round(duration / dt)) + 1
    return np.arange(n) * dt


def _zero_kernel(dt):
    return np.zeros((2, 6, 6)), dt        # horizon = dt: exact zero endpoint, no memory


def _spec(dt=0.01, duration=2.0, mass=1000.0, added=0.0, k=None, c=None, loads=None, **kw):
    t = _grid(dt, duration)
    ker, horizon = _zero_kernel(dt)
    base = dict(
        times=t,
        load_components={"wave": np.zeros((t.size, 6)) if loads is None else loads},
        mass_dry=mass * np.eye(6), added_mass_infinite=added * np.eye(6),
        stiffness=None if k is None else k * np.eye(6),
        damping=None if c is None else c * np.eye(6),
        radiation_kernel=ker, kernel_horizon=horizon,
    )
    base.update(kw)
    return FloatingTransientSpec(**base)


def _free_decay(t, m, c, k, x0=1.0):
    sigma = c / (2.0 * m)
    wd = np.sqrt(k / m - sigma**2)
    return x0 * np.exp(-sigma * t) * (np.cos(wd * t) + sigma / wd * np.sin(wd * t))


# ----------------------------------------------------------------------------- linear core

def test_free_decay_matches_analytic_and_is_second_order():
    m, a, c, k = 1000.0, 250.0, 300.0, 5000.0
    errors = []
    for dt in (0.02, 0.01, 0.005):
        spec = _spec(dt=dt, duration=10.0, mass=m, added=a, k=k, c=c,
                     initial_displacement=[0, 0, 1.0, 0, 0, 0])
        res = solve_floating_transient(spec)
        assert res.status == "complete" and res.complete
        exact = _free_decay(res.times, m + a, c, k)
        errors.append(np.max(np.abs(res.displacements[:, 2] - exact)))
        # the other, uncoupled DOFs stay at rest
        assert np.max(np.abs(res.displacements[:, [0, 1, 3, 4, 5]])) == 0.0
    assert errors[-1] < 1e-3
    assert 3.4 < errors[0] / errors[1] < 4.6 and 3.4 < errors[1] / errors[2] < 4.6


def test_regression_against_marine_linear_without_feedback():
    rng = np.random.default_rng(11)
    dt, n = 0.02, 301
    t = np.arange(n) * dt

    def spd(scale):
        x = rng.standard_normal((6, 6))
        return scale * (x @ x.T / 6.0 + np.eye(6))

    mdry, ainf, K, C = spd(1.0e3), spd(2.0e2), spd(2.0e4), spd(5.0e2)
    kern = np.exp(-0.7 * t)[:, None, None] * spd(4.0e2)[None]
    forces = 3.0e2 * np.sin(1.3 * t)[:, None] * rng.standard_normal(6)[None]
    ref = solve_marine_linear(MarineLinearSpec(
        labels=[f"d{i}" for i in range(6)], dt=dt, times=t, forces=forces, mass_dry=mdry,
        added_mass_infinite=ainf, damping=C, stiffness=K, radiation_kernel=kern, kernel_times=t))
    new = solve_floating_transient(FloatingTransientSpec(
        times=t, load_components={"wave": forces}, mass_dry=mdry, added_mass_infinite=ainf,
        damping=C, stiffness=K, radiation_kernel=kern))
    assert new.status == "complete"
    scale = np.max(np.abs(ref.displacements))
    assert np.max(np.abs(new.displacements - ref.displacements)) < 1e-12 * scale
    assert np.max(np.abs(new.velocities - ref.velocities)) < 1e-12 * np.max(np.abs(ref.velocities))
    assert np.max(np.abs(new.memory_forces - ref.memory_forces)) < 1e-10 * np.max(np.abs(ref.memory_forces))


def test_exponential_memory_against_independent_augmented_ode():
    m, a, k, cr, lam = 1000.0, 200.0, 4000.0, 300.0, 0.8
    dt, dur = 0.005, 12.0
    t = _grid(dt, dur)
    kern = np.zeros((t.size, 6, 6))
    kern[:, 2, 2] = cr * np.exp(-lam * t)
    spec = _spec(dt=dt, duration=dur, mass=m, added=a, k=k, radiation_kernel=kern, kernel_horizon=None,
                 initial_displacement=[0, 0, 1.0, 0, 0, 0])
    res = solve_floating_transient(spec)
    assert res.status == "complete"

    def rhs(_, y):
        q, v, z = y   # z = int exp(-lam(t-s)) v ds
        return [v, -(k * q + cr * z) / (m + a), v - lam * z]

    sol = solve_ivp(rhs, (0, t[-1]), [1.0, 0.0, 0.0], t_eval=t, rtol=1e-11, atol=1e-13, method="DOP853")
    assert np.max(np.abs(res.displacements[:, 2] - sol.y[0])) < 5e-4
    assert np.max(np.abs(res.memory_forces[:, 2] - cr * sol.y[2])) < 1e-3 * cr * np.max(np.abs(sol.y[2]))


def _scalar_oracle(m_tot, k, kfun, dt, n):
    """Independent scalar Newmark + full-lag trapezoid memory with an analytic (windowed) kernel."""
    q = np.zeros(n); v = np.zeros(n); a = np.zeros(n)
    q[0] = 1.0
    a[0] = -k * q[0] / m_tot
    for i in range(1, n):
        qp = q[i - 1] + dt * v[i - 1] + 0.25 * dt * dt * a[i - 1]
        vp = v[i - 1] + 0.5 * dt * a[i - 1]
        past = 0.5 * kfun(i * dt) * v[0] + sum(kfun((i - j) * dt) * v[j] for j in range(1, i))
        past *= dt
        k0 = kfun(0.0)
        # (m + 0.5 dt (0.5 dt k0) + 0.25 dt^2 k) a = -past - 0.5 dt k0 vp - k qp
        eff = m_tot + 0.5 * dt * 0.5 * dt * k0 + 0.25 * dt * dt * k
        a[i] = (-past - 0.5 * dt * k0 * vp - k * qp) / eff
        v[i] = vp + 0.5 * dt * a[i]
        q[i] = qp + 0.25 * dt * dt * a[i]
    return q


def test_windowed_kernel_on_record_longer_than_horizon_matches_finite_memory_oracle():
    m, a, k, cr, lam, H = 1000.0, 200.0, 4000.0, 300.0, 0.3, 4.0
    dt = 0.05
    n_h = int(round(H / dt))
    n = 401   # 20 s >> H
    t = np.arange(n) * dt

    def kfun(tau):
        return cr * np.exp(-lam * tau) * max(0.0, 1.0 - tau / H)

    idx = np.arange(n_h + 1)
    kern = np.zeros((n_h + 1, 6, 6))
    kern[:, 2, 2] = cr * np.exp(-lam * idx * dt) * np.clip(1.0 - idx / n_h, 0.0, None)
    assert kern[-1, 2, 2] == 0.0
    spec = _spec(dt=dt, duration=(n - 1) * dt, mass=m, added=a, k=k, radiation_kernel=kern,
                 kernel_horizon=H, initial_displacement=[0, 0, 1.0, 0, 0, 0])
    res = solve_floating_transient(spec)
    assert res.status == "complete" and res.n_accepted == n
    ref = _scalar_oracle(m + a, k, kfun, dt, n)
    assert np.max(np.abs(res.displacements[:, 2] - ref)) < 1e-11
    # the memory is genuinely truncated: an unwindowed exponential differs on this record
    kern_full = np.zeros((n, 6, 6)); kern_full[:, 2, 2] = cr * np.exp(-lam * t)
    full = solve_floating_transient(_spec(dt=dt, duration=(n - 1) * dt, mass=m, added=a, k=k,
                                          radiation_kernel=kern_full, kernel_horizon=None,
                                          initial_displacement=[0, 0, 1.0, 0, 0, 0]))
    assert np.max(np.abs(full.displacements[:, 2] - ref)) > 1e-3


def test_kernel_admission_refusals():
    t = _grid(0.1, 5.0)
    short = np.zeros((10, 6, 6))
    with pytest.raises(FloatingTransientError, match="cover"):
        solve_floating_transient(_spec(dt=0.1, duration=5.0, radiation_kernel=short, kernel_horizon=None))
    nonzero_end = np.ones((11, 6, 6))
    with pytest.raises(FloatingTransientError, match="exactly zero"):
        solve_floating_transient(_spec(dt=0.1, duration=5.0, radiation_kernel=nonzero_end, kernel_horizon=1.0))
    with pytest.raises(FloatingTransientError, match="kernel_horizon"):
        solve_floating_transient(_spec(dt=0.1, duration=5.0, radiation_kernel=np.zeros((11, 6, 6)), kernel_horizon=0.7))
    assert t.size == 51


def test_symmetry_gates_and_general_external_tangent_with_reported_skew():
    nonsym = 1000.0 * np.eye(6); nonsym[0, 1] = 5.0
    with pytest.raises(FloatingTransientError, match="not symmetric"):
        solve_floating_transient(_spec(stiffness=nonsym))
    # external (documented-attachment) stiffness may be non-symmetric: compare with a dense ODE
    Kx = 4000.0 * np.eye(6)
    Kx[0, 1], Kx[1, 0] = 800.0, -300.0
    Cx = 200.0 * np.eye(6)
    dt, dur, m = 0.005, 6.0, 1000.0
    res = solve_floating_transient(_spec(dt=dt, duration=dur, mass=m, external_stiffness=Kx, external_damping=Cx,
                                         initial_displacement=[1.0, 0.5, 0, 0, 0, 0]))
    A = np.block([[np.zeros((6, 6)), np.eye(6)], [-Kx / m, -Cx / m]])
    sol = solve_ivp(lambda _, y: A @ y, (0, res.times[-1]), np.r_[[1.0, 0.5, 0, 0, 0, 0], np.zeros(6)],
                    t_eval=res.times, rtol=1e-11, atol=1e-13, method="DOP853")
    assert np.max(np.abs(res.displacements - sol.y[:6].T)) < 1e-3
    assert res.external_stiffness_skew_norm == pytest.approx(np.linalg.norm([550.0, 550.0]) , rel=1e-12)
    assert res.external_stiffness_symmetric_norm > 0.0


def test_mass_decomposition_consistency_and_component_forces():
    hull = 800.0 * np.eye(6); rna = 150.0 * np.eye(6); tower = 50.0 * np.eye(6)
    with pytest.raises(FloatingTransientError, match="mass_components"):
        solve_floating_transient(_spec(mass_components={"hull": hull, "rna": rna, "tower": tower}, mass=999.0))
    spec = _spec(mass=1000.0, mass_components={"hull": hull, "rna": rna, "tower": tower},
                 loads=np.tile([10.0, 0, 0, 0, 0, 0], (201, 1)))
    res = solve_floating_transient(spec)
    total = sum(res.dry_inertia_components.values())
    assert np.allclose(total, 1000.0 * res.accelerations, rtol=1e-12, atol=1e-12)
    assert np.allclose(res.dry_inertia_components["rna"], 150.0 * res.accelerations)
    assert res.accelerations[0, 0] == pytest.approx(10.0 / 1000.0)


# ----------------------------------------------------------------------------- turbine feedback

def _wind(U, dur=40.0):
    return dict(wind_times=[0.0, dur], wind_velocity=[[U, 0, 0], [U, 0, 0]])


def _turbine(curve_v, curve_t, Tm, U=10.0, hub=(0.0, 0.0, 0.0), axis=(1.0, 0, 0), dur=40.0, **kw):
    d = dict(Q0=np.eye(3), hub_offset_body=hub, rotor_axis_body=axis, thrust_speeds=curve_v,
             thrust_values=curve_t, mean_thrust=Tm)
    d.update(_wind(U, dur))
    d.update(kw)
    return d


def test_linear_curve_gives_analytic_aerodynamic_damping_oscillator():
    # T(V) = T0 + s (V - U);  Vrel = U - xdot  =>  m xddot + (c + s) xdot + k x = 0
    m, c, k, s, U, T0 = 1000.0, 100.0, 4000.0, 400.0, 10.0, 8.0e4
    dt, dur = 0.005, 10.0
    v = np.array([0.0, 20.0]); T = T0 + s * (v - U)
    spec = _spec(dt=dt, duration=dur, mass=m, k=k, c=c, initial_displacement=[1.0, 0, 0, 0, 0, 0],
                 **_turbine(v, T, T0, U=U, dur=dur))
    res = solve_floating_transient(spec)
    assert res.status == "complete"
    exact = _free_decay(res.times, m, c + s, k)
    assert np.max(np.abs(res.displacements[:, 0] - exact)) < 1e-3
    assert np.allclose(res.aero_damping_slope, s) and res.negative_damping_steps == 0
    assert np.all(res.root_candidates[1:] == 1)
    # kinematic residual and force balance at the solver tolerance
    assert res.max_scaled_residual < 1e-10 and np.max(res.kinematic_residual) < 1e-10
    # the turbine force is b [T - Tm] and equals s (-xdot) in this linear case
    assert np.allclose(res.turbine_forces[:, 0], -s * res.velocities[:, 0], atol=1e-8)


def test_negative_slope_is_reported_and_drives_growth_not_clamped():
    m, c, k, s, U, T0 = 1000.0, 100.0, 4000.0, -300.0, 10.0, 8.0e4   # c + s < 0: negative net damping
    dt, dur = 0.005, 10.0
    v = np.array([0.0, 20.0]); T = T0 + s * (v - U)
    T = np.maximum(T, 0.0)
    res = solve_floating_transient(_spec(dt=dt, duration=dur, mass=m, k=k, c=c, initial_displacement=[0.01, 0, 0, 0, 0, 0],
                                         **_turbine(v, T, T0, U=U, dur=dur)))
    assert res.status == "complete"
    assert res.negative_damping_steps == res.n_accepted
    exact = _free_decay(res.times, m, c + s, k, x0=0.01)
    assert np.max(np.abs(res.displacements[:, 0] - exact)) < 1e-3 * np.max(np.abs(exact))
    assert np.max(np.abs(res.displacements[:, 0])) > 0.01 * 1.5   # growth


M_K, C_K, K_K, U_K = 1000.0, 50.0, 4000.0, 10.0


def _knot_case(sp, th, dt, dur, load_amp=0.0, load_omega=1.5):
    """Solver and independent (clamped-free) ODE for a surge oscillator with a piecewise-linear curve.

    The ODE evaluates the curve with np.interp, which clamps outside the domain; the solver never does, so
    ``domain_exit`` (from the ODE at fine sampling) says whether the solver is required to refuse.
    """
    Tm = float(np.interp(U_K, sp, th))
    t = _grid(dt, dur)
    loads = np.zeros((t.size, 6)); loads[:, 0] = load_amp * np.sin(load_omega * t)
    res = solve_floating_transient(_spec(dt=dt, duration=dur, mass=M_K, k=K_K, c=C_K, loads=loads,
                                         initial_displacement=[0.8, 0, 0, 0, 0, 0],
                                         **_turbine(sp, th, Tm, U=U_K, dur=dur)))

    def rhs(tt, y):
        x, xd = y
        return [xd, (-K_K * x - C_K * xd + load_amp * np.sin(load_omega * tt)
                     + np.interp(U_K - xd, sp, th) - Tm) / M_K]

    def below(tt, y):          # V - V_min: crossing downward leaves the curve domain at the low end
        return U_K - y[1] - sp[0]
    below.terminal, below.direction = True, -1

    def above(tt, y):          # V_max - V: crossing downward leaves the domain at the high end
        return sp[-1] - (U_K - y[1])
    above.terminal, above.direction = True, -1

    dense = np.linspace(0.0, t[-1], 4 * t.size)
    sol = solve_ivp(rhs, (0, t[-1]), [0.8, 0.0], t_eval=dense, rtol=1e-11, atol=1e-13, method="DOP853",
                    max_step=0.01, events=[below, above])
    if sol.status not in (0, 1):
        pytest.fail(f"independent ODE reference is unresolved: {sol.message}")
    vrel = U_K - sol.y[1]
    exit_margin = max(sp[0] - vrel.min(), vrel.max() - sp[-1])
    t_exit = min((float(te[0]) for te in sol.t_events if len(te)), default=None)
    if t_exit is None and abs(exit_margin) < 1e-3:
        pytest.fail("independent ODE reference grazes the curve domain boundary: unresolved")
    return res, sol, t, exit_margin, t_exit


def test_negative_slope_knot_curve_domain_behaviour_agrees_with_independent_ode():
    """Preserved failed-run fixture (negative aerodynamic damping beyond V = 10).

    The solver must refuse where the independent ODE (terminal events at V = 0 / V = 20) leaves the curve domain,
    and otherwise meet the unchanged 1e-3 gate; it never extrapolates or clips.  An unresolved ODE or a grazing
    boundary fails explicitly (no neutral band).
    """
    # ORIGINAL recorded failed-run inputs (checks/transient-initial): slopes 5000, 20000, -10000, -5000 N s/m
    sp = np.array([0.0, 8.0, 10.0, 12.0, 20.0])
    th = np.array([1.0e4, 5.0e4, 9.0e4, 7.0e4, 3.0e4])
    res, sol, t, margin, t_exit = _knot_case(sp, th, 0.004, 12.0)
    if t_exit is not None:
        # the independent ODE physically reaches V = 0 or V = 20 (event stop): the solver must refuse there
        assert res.status == "no_root" and not res.complete and res.negative_damping_steps >= 1
        assert "domain" in res.message and "[0.0, 20.0]" in res.message
        assert res.n_accepted < t.size
        assert abs(float(res.times[-1]) - t_exit) <= 0.05, (float(res.times[-1]), t_exit)
    else:
        assert margin < -1e-3
        assert res.status == "complete"
        assert np.max(np.abs(res.displacements[:, 0] - np.interp(res.times, sol.t, sol.y[0]))) < 1e-3 * 0.8


def test_stable_knot_crossing_nonlinear_curve_matches_independent_ode():
    """Admitted reference: every slope is positive (500..1500 N s/m), so aerodynamic damping is positive on every
    branch, |xdot| <= omega x0 stays near 1.6 m/s and V = 10 - xdot stays inside [0, 20] by a wide margin; a
    400 N harmonic load keeps the motion crossing the V = 10 knot.  Gate unchanged: 1e-3 x amplitude."""
    sp = np.array([0.0, 8.0, 10.0, 12.0, 20.0])
    th = np.array([1.0e3, 5.0e3, 8.0e3, 9.0e3, 9.9e3])        # slopes 1000, 1500, 500, 112.5
    res, sol, t, margin, t_exit = _knot_case(sp, th, 0.004, 12.0, load_amp=400.0)
    assert t_exit is None and margin < -5.0, "reference must stay well inside the admitted domain"
    assert res.status == "complete" and res.negative_damping_steps == 0
    v_rel = res.relative_axial_speed
    assert np.sum(np.diff(np.sign(v_rel - 10.0)) != 0) >= 4          # repeatedly crosses the V = 10 knot
    assert np.min(v_rel) < 10.0 < np.max(v_rel)
    assert np.all(res.root_candidates[1:] == 1)
    ref = np.interp(res.times, sol.t, sol.y[0])
    assert np.max(np.abs(res.displacements[:, 0] - ref)) < 1e-3 * 0.8
    assert res.max_scaled_residual < 1e-10


def test_surge_and_pitch_hub_kinematics_at_the_initial_state():
    U, h = 12.0, 90.0
    v0 = [2.0, 0.0, 0.0, 0.0, 0.05, 0.0]   # surge 2 m/s, pitch rate 0.05 rad/s about +y
    sp = np.array([0.0, 40.0]); th = np.array([0.0, 4.0e4])
    res = solve_floating_transient(_spec(dt=0.01, duration=0.05, mass=1.0e5, initial_velocity=v0,
                                         **_turbine(sp, th, 1.0e4, U=U, hub=(0, 0, h), dur=0.05)))
    # omega x r = (0.05 * h, 0, 0): both reduce the relative wind
    assert res.relative_axial_speed[0] == pytest.approx(U - 2.0 - 0.05 * h, abs=1e-12)
    assert res.thrust[0] == pytest.approx(np.interp(U - 2.0 - h * 0.05, sp, th))
    # the hub moment follows b = [n; r x n] = [1,0,0, 0,h,0]
    assert res.turbine_forces[0, 4] == pytest.approx(h * (res.thrust[0] - 1.0e4))


def test_all_steps_satisfy_direct_cross_product_relative_wind_with_tilt_and_nonuniform_wind():
    tilt = np.deg2rad(7.0)
    axis = np.array([np.cos(tilt), 0.0, np.sin(tilt)])
    hub = np.array([-4.0, 1.5, 95.0])
    Q0 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    wt = np.linspace(0.0, 6.0, 7)
    wind = np.stack([2.0 * np.sin(wt), 11.0 + np.cos(wt), 0.3 * wt], axis=1)
    sp = np.array([0.0, 6.0, 12.0, 25.0]); th = np.array([2.0e4, 6.0e4, 1.2e5, 9.0e4])
    t = _grid(0.02, 6.0)
    rng = np.random.default_rng(5)
    loads = np.zeros((t.size, 6)); loads[:, 3] = 2.0e5 * np.sin(0.8 * t); loads[:, 1] = 5.0e4 * np.cos(0.5 * t)
    spec = FloatingTransientSpec(
        times=t, load_components={"wave": loads}, mass_dry=3.0e6 * np.eye(6), added_mass_infinite=5.0e5 * np.eye(6),
        stiffness=np.diag([2e4, 2e4, 3e6, 5e8, 5e8, 1e7]), damping=2.0e4 * np.eye(6),
        radiation_kernel=np.zeros((2, 6, 6)), kernel_horizon=0.02, wind_times=wt, wind_velocity=wind, Q0=Q0,
        hub_offset_body=hub, rotor_axis_body=axis, thrust_speeds=sp, thrust_values=th, mean_thrust=6.0e4,
        initial_displacement=rng.uniform(-1e-3, 1e-3, 6))
    res = solve_floating_transient(spec)
    assert res.status == "complete", res.message
    for k in range(res.n_accepted):
        U = np.array([np.interp(res.times[k], wt, wind[:, i]) for i in range(3)])
        Ub = Q0.T @ U
        Ub = Ub - np.cross(res.displacements[k, 3:], Ub)
        vh = res.velocities[k, :3] + np.cross(res.velocities[k, 3:], hub)
        assert res.relative_axial_speed[k] == pytest.approx(float((Ub - vh) @ axis), abs=1e-9)
        assert res.thrust[k] == pytest.approx(np.interp(res.relative_axial_speed[k], sp, th), abs=1e-6)


def test_multiple_roots_are_refused_with_partial_history():
    # g(V) = V - 15 + 0.5 T(V) has roots 5, 11.25 and 15 for this curve (hand-derived)
    sp = np.array([0.0, 10.0, 12.0, 30.0]); th = np.array([20.0, 20.0, 0.0, 0.0])
    res = solve_floating_transient(_spec(dt=1.0, duration=3.0, mass=1.0, **_turbine(sp, th, 20.0, U=5.0, dur=3.0),
                                         force_scale=1.0, moment_scale=1.0))
    assert res.status == "multiple_roots" and not res.complete
    assert res.n_accepted == 1 and res.failed_step == 1
    assert "3 roots" in res.message
    with pytest.raises(FloatingTransientError):
        res.raise_for_status()


def test_degenerate_interval_of_roots_is_refused():
    sp = np.array([0.0, 10.0, 20.0]); th = np.array([20.0, 20.0, 0.0])    # slope -2, kappa = 1/2: 1 + kappa s = 0
    res = solve_floating_transient(_spec(dt=1.0, duration=3.0, mass=1.0, **_turbine(sp, th, 20.0, U=10.0, dur=3.0),
                                         force_scale=1.0, moment_scale=1.0))
    assert res.status == "degenerate_interval" and res.n_accepted == 1


def test_curve_domain_exit_is_a_hard_non_success_with_time_and_domain():
    sp = np.array([0.0, 10.0]); th = np.array([5.0, 5.0])
    spec = _spec(dt=0.5, duration=3.0, mass=100.0, force_scale=10.0, moment_scale=10.0,
                 wind_times=[0.0, 1.0, 2.0, 3.0], wind_velocity=[[5, 0, 0], [5, 0, 0], [50, 0, 0], [50, 0, 0]],
                 Q0=np.eye(3), hub_offset_body=[0, 0, 0], rotor_axis_body=[1, 0, 0], thrust_speeds=sp,
                 thrust_values=th, mean_thrust=5.0)
    res = solve_floating_transient(spec)
    assert res.status == "no_root" and not res.complete
    assert res.failed_step is not None and res.n_accepted == res.failed_step
    assert res.times[-1] < 2.0
    assert "domain" in res.message and "[0.0, 10.0]" in res.message


def test_curve_abscissa_and_partial_turbine_inputs_are_refused():
    sp = np.array([0.0, 10.0]); th = np.array([5.0, 5.0])
    with pytest.raises(FloatingTransientError, match="rotor_axial_speed"):
        solve_floating_transient(_spec(**_turbine(sp, th, 5.0), thrust_abscissa="horizontal_hub_speed"))
    bad = _turbine(sp, th, 5.0)
    bad.pop("rotor_axis_body")
    with pytest.raises(FloatingTransientError, match="together"):
        solve_floating_transient(_spec(**bad))
    with pytest.raises(FloatingTransientError, match="cover"):
        solve_floating_transient(_spec(duration=10.0, **_turbine(sp, th, 5.0, dur=5.0)))


# ----------------------------------------------------------------------------- guards and lifecycle

def test_rotation_envelope_is_a_non_success_that_keeps_diagnostics():
    res = solve_floating_transient(_spec(initial_displacement=[0, 0, 0, np.deg2rad(6.0), 0, 0], k=1.0e4))
    assert res.status == "rotation_envelope_exceeded" and not res.complete
    # the failed row is NOT part of the accepted history; its information lives in the diagnostics
    assert res.n_accepted == 0 and res.failed_step == 0 and res.displacements.shape == (0, 6)
    assert res.failed_row_diagnostics["rotation"] == pytest.approx(np.deg2rad(6.0))
    assert res.failed_row_diagnostics["row"] == 0 and "excluded" in res.message


def test_cancellation_and_progress_report_exact_accepted_history():
    seen = []

    class Token:
        def __init__(self): self.polls = 0
        @property
        def is_cancelled(self):
            self.polls += 1
            return self.polls > 5

    res = solve_floating_transient(_spec(dt=0.01, duration=2.0, k=100.0), cancellation_token=Token(),
                                   progress_callback=lambda i, n: seen.append((i, n)))
    assert res.status == "cancelled" and res.cancelled and not res.complete
    assert res.n_accepted == res.times.size == res.displacements.shape[0]
    assert seen[0] == (1, 201) and seen == sorted(seen) and seen[-1][0] == res.n_accepted


def test_resource_guard_refuses_before_integration_and_result_is_read_only():
    with pytest.raises(FloatingTransientError, match="resource"):
        solve_floating_transient(_spec(duration=2.0), resource_config=FloatingTransientResourceConfig(max_steps=10))
    with pytest.raises(FloatingTransientError, match="working memory"):
        solve_floating_transient(_spec(duration=2.0), resource_config={"max_working_bytes": 1000})
    res = solve_floating_transient(_spec(duration=0.2, identity={"body": "B1", "mesh": "m1"}))
    assert res.identity == {"body": "B1", "mesh": "m1"}
    with pytest.raises(ValueError):
        res.displacements[0, 0] = 1.0
    with pytest.raises(ValueError):
        res.load_components["wave"][0, 0] = 1.0


def test_gates_only_tighten_and_direct_construction_validates():
    from dataclasses import replace
    base = _spec(duration=0.1)
    for change in (dict(residual_tolerance=1e-9), dict(rotation_limit=np.deg2rad(6.0)), dict(force_scale=np.inf),
                   dict(moment_scale=float("nan")), dict(force_scale=-1.0), dict(residual_tolerance=0.0),
                   dict(mean_thrust=float("inf")), dict(thrust_abscissa="horizontal_hub_speed")):
        with pytest.raises(FloatingTransientError):
            replace(base, **change)
    tighter = replace(base, residual_tolerance=1e-12, rotation_limit=np.deg2rad(2.0))
    assert solve_floating_transient(tighter).status == "complete"


def test_result_arrays_are_bytes_backed_and_identity_is_deep_detached():
    ident = {"wind": {"seed": 4, "tags": ["a", "b"]}, "arr": np.arange(3)}
    res = solve_floating_transient(_spec(duration=0.1, identity=ident))
    ident["wind"]["seed"] = 99
    ident["wind"]["tags"].append("c")
    ident["arr"][0] = 7
    assert res.identity["wind"]["seed"] == 4 and res.identity["wind"]["tags"] == ("a", "b")
    assert res.identity["arr"][0] == 0
    with pytest.raises(ValueError):
        res.displacements.setflags(write=True)
    with pytest.raises(ValueError):
        res.load_components["wave"].setflags(write=True)
    with pytest.raises(TypeError):
        res.identity["wind"]["seed"] = 1


def test_distinct_nearby_interior_roots_are_not_merged_but_a_shared_knot_root_is_counted_once():
    sp = np.array([0.0, 10.0 - 3e-9, 10.0 - 1e-9, 10.0, 10.0 + 1e-9, 10.0 + 3e-9, 20.0])
    th = 510.0 - sp + np.array([-1.0, -1e-9, 1e-9, 0.0, -1e-9, 1e-9, 1.0])
    res = solve_floating_transient(_spec(dt=2.0, duration=2.0, mass=1.0, force_scale=1.0, moment_scale=1.0,
                                         **_turbine(sp, th, 500.0, U=10.0, dur=2.0)))
    assert res.status == "multiple_roots" and not res.complete
    # a single curve whose only root sits exactly on a knot is one root
    sp1 = np.array([0.0, 10.0, 20.0]); th1 = 510.0 - sp1 + np.array([-1.0, 0.0, 1.0])
    res1 = solve_floating_transient(_spec(dt=2.0, duration=2.0, mass=1.0, force_scale=1.0, moment_scale=1.0,
                                          **_turbine(sp1, th1, 500.0, U=10.0, dur=2.0)))
    assert res1.status == "complete" and res1.root_candidates[1] == 1
    assert res1.relative_axial_speed[1] == pytest.approx(10.0, abs=1e-12)


def test_failed_late_row_is_excluded_and_keeps_its_original_index():
    # a heavily forced roll drives the rotation beyond 5 degrees partway through the run
    t = _grid(0.01, 3.0)
    loads = np.zeros((t.size, 6)); loads[:, 3] = 3.0e4
    res = solve_floating_transient(_spec(dt=0.01, duration=3.0, mass=1.0e3, k=1.0e3, loads=loads))
    assert res.status == "rotation_envelope_exceeded"
    k = res.failed_step
    assert k is not None and 0 < k < t.size and res.n_accepted == k and res.times.size == k
    assert np.max(np.linalg.norm(res.displacements[:, 3:], axis=1)) <= np.deg2rad(5.0)
    assert res.failed_row_diagnostics["row"] == k and res.failed_row_diagnostics["rotation"] > np.deg2rad(5.0)
    assert res.max_rotation <= np.deg2rad(5.0)


def test_frame_tag_mean_thrust_and_reference_wind():
    from dataclasses import replace
    sp = np.array([0.0, 20.0]); th = np.array([0.0, 2.0e4])
    base = _spec(duration=0.05, **_turbine(sp, th, 1.0e4, U=10.0, dur=0.05))
    assert solve_floating_transient(base).frame == "body_attached_first_order"
    with pytest.raises(FloatingTransientError, match="body_attached_first_order"):
        replace(base, frame="fixed_mean_frame")
    no_mean = {k: v for k, v in _turbine(sp, th, 1.0e4, U=10.0, dur=0.05).items() if k != "mean_thrust"}
    with pytest.raises(FloatingTransientError, match="mean_thrust"):
        solve_floating_transient(_spec(duration=0.05, **no_mean))
    assert solve_floating_transient(_spec(duration=0.05)).mean_reference is None      # no turbine, no mean thrust needed
    unverified = solve_floating_transient(base)
    assert unverified.mean_reference["status"] == "unverified" and unverified.mean_equilibrium_status == "not_verified"
    checked = solve_floating_transient(replace(base, reference_wind_velocity=[10.0, 0.0, 0.0]))
    assert checked.mean_reference["residual"] == pytest.approx(1.0e4 - 1.0e4)       # T(10) = 1e4 = mean_thrust
    off = solve_floating_transient(replace(base, reference_wind_velocity=[12.0, 0.0, 0.0]))
    assert off.mean_reference["residual"] == pytest.approx(1.2e4 - 1.0e4)
    with pytest.raises(FloatingTransientError, match="reference wind"):
        solve_floating_transient(replace(base, reference_wind_velocity=[40.0, 0.0, 0.0]))


def test_non_unit_rotor_axis_is_rejected_not_normalised():
    sp = np.array([0.0, 20.0]); th = np.array([0.0, 2.0e4])
    with pytest.raises(FloatingTransientError, match="unit"):
        solve_floating_transient(_spec(duration=0.05, **_turbine(sp, th, 1.0e4, axis=(2.0, 0, 0), dur=0.05)))
    with pytest.raises(FloatingTransientError, match="unit"):
        solve_floating_transient(_spec(duration=0.05, **_turbine(sp, th, 1.0e4, axis=(1.0, 1e-4, 0), dur=0.05)))


def test_scales_are_natural_zero_scale_is_explicit_and_user_scales_only_tighten():
    # no force at all: every term is zero, numerator and scale are zero => residual 0, not a floor-hidden value
    res = solve_floating_transient(_spec(duration=0.1))
    assert np.all(res.force_scale_used == 0.0) and res.max_scaled_residual == 0.0
    # a real balance: used scales equal the largest contributing term, with no 1 N floor
    loads = np.zeros((11, 6)); loads[:, 0] = 1.0e-3
    res = solve_floating_transient(_spec(dt=0.01, duration=0.1, mass=1.0, loads=loads))
    assert res.force_scale_used[0] == pytest.approx(1.0e-3) and res.moment_scale_used[0] == 0.0
    assert res.max_scaled_residual < 1e-12
    # a larger user scale cannot weaken the gate: the natural scale is still used
    big = solve_floating_transient(_spec(dt=0.01, duration=0.1, mass=1.0, loads=loads, force_scale=1.0e6))
    assert big.force_scale_used[0] == pytest.approx(1.0e-3)
    small = solve_floating_transient(_spec(dt=0.01, duration=0.1, mass=1.0, loads=loads, force_scale=1.0e-9))
    assert small.force_scale_used[0] == pytest.approx(1.0e-9) or small.status != "complete"


def test_cancellation_token_shape_is_validated_before_any_step():
    seen = []
    with pytest.raises(FloatingTransientError, match="cancellation_token"):
        solve_floating_transient(_spec(duration=0.1), cancellation_token=42, progress_callback=lambda i, n: seen.append(i))
    assert seen == []
    assert solve_floating_transient(_spec(duration=0.1), cancellation_token=lambda: False).complete


def test_oversize_tangent_is_refused_from_metadata_before_payload_copies():
    n = 1_000_000
    g = np.broadcast_to(np.zeros(1), (n, 6, 6))            # 288 MB virtual view, nothing allocated
    t = np.arange(n) * 0.01
    spec = FloatingTransientSpec(times=t, load_components={"w": np.broadcast_to(np.zeros(1), (n, 6))},
                                 mass_dry=np.eye(6), added_mass_infinite=np.zeros((6, 6)),
                                 radiation_kernel=np.zeros((2, 6, 6)), kernel_horizon=0.01, external_load_tangent=g)
    with pytest.raises(FloatingTransientError, match="working memory"):
        solve_floating_transient(spec)
    with pytest.raises(FloatingTransientError, match="resource_config"):
        solve_floating_transient(_spec(duration=0.1), resource_config={"max_working_bytes": 512 * 1024**2})
    with pytest.raises(FloatingTransientError, match="resource_config"):
        solve_floating_transient(_spec(duration=0.1), resource_config={"max_working_bytes": -1})


def test_both_adjacent_slopes_are_reported_at_a_curve_knot():
    # V = 10 sits exactly on the knot between slopes +1000 (left) and -500 (right): damping may be negative
    sp = np.array([0.0, 10.0, 20.0]); th = np.array([0.0, 1.0e4, 5.0e3])
    res = solve_floating_transient(_spec(dt=0.01, duration=0.01, mass=1.0e3, **_turbine(sp, th, 1.0e4, U=10.0, dur=0.01)))
    assert res.aero_damping_slope_left[0] == pytest.approx(1000.0) and res.aero_damping_slope_right[0] == pytest.approx(-500.0)
    assert res.aero_damping_slope[0] == pytest.approx(-500.0) and res.negative_damping_steps >= 1
    sp2 = np.array([0.0, 10.0, 20.0]); th2 = np.array([0.0, 1.0e4, 2.0e4])
    r2 = solve_floating_transient(_spec(dt=0.01, duration=0.01, mass=1.0e3, **_turbine(sp2, th2, 1.0e4, U=10.0, dur=0.01)))
    assert r2.negative_damping_steps == 0


def test_result_carries_digests_of_the_actual_kernel_and_ainf():
    from anysolver.floating_transient import numeric_array_digest
    kern = np.zeros((2, 6, 6)); kern[0] = 5.0 * np.eye(6)
    res = solve_floating_transient(_spec(duration=0.1, added=7.0, radiation_kernel=kern, kernel_horizon=0.01))
    assert res.kernel_digest == numeric_array_digest(kern) and res.ainf_digest == numeric_array_digest(7.0 * np.eye(6))
    other = solve_floating_transient(_spec(duration=0.1, added=8.0, radiation_kernel=kern, kernel_horizon=0.01))
    assert other.ainf_digest != res.ainf_digest and other.kernel_digest == res.kernel_digest
    assert res.dt == pytest.approx(0.01)


def _active_kernel_spec(mode, c=1.0e-3):
    """Kernel K(t) = -c exp(-t) I (an ACTIVE, non-passive memory): Herm Z(w) = -c/(1+w^2) I, so the normalised
    non-passivity at w = 0.1 is ~ c/(1.01 * 0.1 * M) >> 1e-3 for M = 1."""
    from anysolver.floating_transient import numeric_array_digest
    dt, n_k = 0.01, 101
    tk = np.arange(n_k) * dt
    w = np.clip(1.0 - tk / tk[-1], 0.0, None)                      # exactly zero endpoint (windowed)
    kern = (-c * np.exp(-tk) * w)[:, None, None] * np.eye(6)[None]
    A = 0.5 * np.eye(6)
    binding = dict(kernel_digest=numeric_array_digest(kern), ainf_digest=numeric_array_digest(A), kernel_dt=dt)
    t = np.arange(21) * dt
    loads = np.zeros((t.size, 6)); loads[:, 2] = np.sin(3.0 * t)
    return FloatingTransientSpec(times=t, load_components={"wave": loads}, mass_dry=np.eye(6), added_mass_infinite=A,
                                 radiation_kernel=kern, kernel_horizon=tk[-1], stiffness=np.eye(6),
                                 hydro_admission=mode, admission_omegas=np.array([0.1, 1.0]),
                                 native_source_binding=binding)


def test_development_unverified_retains_the_failed_passivity_gate_instead_of_refusing():
    from anysolver.floating_transient import BoundedNativeAdmissionError
    with pytest.raises(BoundedNativeAdmissionError, match="non-passivity"):
        solve_floating_transient(_active_kernel_spec("bounded_native"))           # bounded mode unchanged: refused
    res = solve_floating_transient(_active_kernel_spec("development_unverified"))
    rep = res.hydrodynamic_admission_report
    assert rep["status"] == "admitted_development_unverified" and rep["hydrodynamic_verification"] == "unverified"
    assert rep["impedance"]["status"] == "failed_retained_development" and rep["impedance"]["worst"] > 1e-3
    assert "hydro_impedance_nonpassivity" in rep["unpassed_gates"]
    assert res.n_accepted >= 1


def test_development_mode_keeps_every_other_gate():
    from anysolver.floating_transient import BoundedNativeAdmissionError
    spec = _active_kernel_spec("development_unverified")
    bad = dict(spec.native_source_binding); bad["kernel_digest"] = "0" * 64
    from dataclasses import replace
    with pytest.raises(BoundedNativeAdmissionError, match="binding"):
        solve_floating_transient(replace(spec, native_source_binding=bad))
    A = spec.added_mass_infinite.copy(); A[0, 1] += 0.1; A[1, 0] -= 0.1          # large mass skew: rho >> 1e-3
    from anysolver.floating_transient import numeric_array_digest
    b2 = dict(spec.native_source_binding); b2["ainf_digest"] = numeric_array_digest(A)
    with pytest.raises(BoundedNativeAdmissionError, match="rho"):
        solve_floating_transient(replace(spec, added_mass_infinite=A, native_source_binding=b2))


def test_complex_inputs_are_refused_before_any_float_cast():
    import warnings
    from dataclasses import replace
    from anysolver.floating_transient import numeric_array_digest
    cx = lambda x: np.asarray(x) * (1.0 + 0j)          # zero imaginary part is still complex
    base = _spec(duration=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # a ComplexWarning must not be the "refusal"
        for change in (dict(load_components={"f": np.ones((11, 6), complex) * (1 + 2j)}),
                       dict(load_components={"f": cx(np.zeros((11, 6)))}),
                       dict(times=cx(base.times)), dict(mass_dry=cx(np.eye(6))), dict(added_mass_infinite=cx(np.zeros((6, 6)))),
                       dict(stiffness=cx(np.eye(6))), dict(external_stiffness=cx(np.eye(6))),
                       dict(radiation_kernel=cx(np.zeros((2, 6, 6)))), dict(initial_displacement=cx(np.zeros(6))),
                       dict(initial_velocity=np.array([1j, 0, 0, 0, 0, 0])), dict(external_load_tangent=cx(np.zeros((11, 6, 6))))):
            with pytest.raises(FloatingTransientError, match="complex"):
                solve_floating_transient(replace(base, **change))
        sp = np.array([0.0, 20.0]); th = np.array([0.0, 2.0e4])
        turb = _turbine(sp, th, 1.0e4, U=10.0, dur=0.1)
        for key, val in (("wind_velocity", cx([[10.0, 0, 0], [10.0, 0, 0]])), ("Q0", cx(np.eye(3))),
                         ("hub_offset_body", cx([0, 0, 1.0])), ("rotor_axis_body", cx([1.0, 0, 0])),
                         ("thrust_speeds", cx(sp)), ("thrust_values", cx(th)), ("wind_times", cx([0.0, 1.0]))):
            with pytest.raises(FloatingTransientError, match="complex"):
                solve_floating_transient(_spec(duration=0.1, **{**turb, key: val}))
        with pytest.raises(FloatingTransientError):
            replace(base, mean_thrust=1.0 + 2j)
        with pytest.raises(FloatingTransientError, match="complex"):
            solve_floating_transient(_spec(duration=0.1, **{**turb, "reference_wind_velocity": cx([10.0, 0, 0])}))
        with pytest.raises(FloatingTransientError, match="complex"):
            numeric_array_digest(cx(np.zeros(3)))
        with pytest.raises(FloatingTransientError, match="complex"):
            waterline_guard(cx(np.zeros((2, 6))), np.zeros((1, 3)), np.zeros((2, 1)), 1.0)
    assert solve_floating_transient(base).complete          # real input is unchanged


def test_digest_is_incremental_and_equals_the_reference_protocol():
    import hashlib
    from anysolver.floating_transient import numeric_array_digest
    rng = np.random.default_rng(1)
    for shape in ((5,), (2, 3), (3, 40000), (1, 6, 6)):               # one shape exceeds a 65536-element chunk
        x = rng.standard_normal(shape)
        h = hashlib.sha256(b"numeric-array-v1|" + repr(tuple(int(s) for s in shape)).encode() + b"|")
        h.update(np.ascontiguousarray(x, dtype="<f8").tobytes())
        assert numeric_array_digest(x) == h.hexdigest()
    x = rng.standard_normal((4, 3))
    assert numeric_array_digest(x.T) == numeric_array_digest(np.ascontiguousarray(x.T))     # non-contiguous input
    assert numeric_array_digest(x.astype(np.float32)) == numeric_array_digest(x.astype(np.float32).astype(float))


def test_digest_of_a_transposed_strided_array_has_no_full_copy_and_matches_the_protocol():
    import hashlib
    import tracemalloc
    from anysolver.floating_transient import numeric_array_digest
    base = np.random.default_rng(6).standard_normal((1200, 1000))            # 9.6 MB
    view = base.T                                                              # strided view, not C-contiguous
    h = hashlib.sha256(b"numeric-array-v1|(1000, 1200)|")
    h.update(np.ascontiguousarray(view, dtype="<f8").tobytes())
    expected = h.hexdigest()
    tracemalloc.start()
    try:
        got = numeric_array_digest(view)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert got == expected and peak < 3 * 1024 * 1024 < base.nbytes
    assert numeric_array_digest(base.astype(">f8")) == numeric_array_digest(base)
    assert numeric_array_digest(np.zeros((0, 3))) == hashlib.sha256(b"numeric-array-v1|(0, 3)|").hexdigest()


def test_admission_shape_handling_never_converts_lists_and_rejects_unknown_types():
    from anysolver.floating_transient import _shape_of
    assert _shape_of([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) == (2, 3) and _shape_of((1.0, 2.0)) == (2,)
    assert _shape_of(np.zeros((4, 2))) == (4, 2) and _shape_of([]) == (0,)
    assert _shape_of(np.broadcast_to(np.zeros(1), (10**9, 6, 6))) == (10**9, 6, 6)        # metadata only
    for ragged in ([[1.0, 2.0], [3.0]], [[1.0, 2.0], 3.0], [1.0, [2.0]], [[1.0], [2.0, 3.0], [4.0]],
                   [[[1.0, 2.0]], [[3.0]]], [[1.0, 2.0], [3.0, "x"]]):
        with pytest.raises(FloatingTransientError, match="ragged"):
            _shape_of(ragged)
    assert _shape_of([[[1.0, 2.0], [3.0, 4.0]]]) == (1, 2, 2) and _shape_of([[], []]) == (2, 0)
    with pytest.raises(FloatingTransientError, match="ragged"):             # also through the public entry point
        solve_floating_transient(_spec(duration=0.05, times=[0.0, 0.01, [0.02], 0.03, 0.04, 0.05],
                                       load_components={"w": np.zeros((6, 6))}))
    with pytest.raises(FloatingTransientError, match="ndarray"):
        _shape_of(iter([1.0, 2.0]))
    with pytest.raises(FloatingTransientError, match="ndarray"):
        _shape_of({"a": 1})
    # list-valued times are still accepted (public compatibility) and a huge list-of-lists tangent is judged by shape
    assert solve_floating_transient(_spec(duration=0.05, times=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                                          load_components={"w": np.zeros((6, 6))})).complete


def test_reference_wind_status_is_a_comparison_not_an_acceptance():
    from dataclasses import replace
    sp = np.array([0.0, 20.0]); th = np.array([0.0, 2.0e4])
    base = _spec(duration=0.05, **_turbine(sp, th, 1.0e4, U=10.0, dur=0.05))
    off = solve_floating_transient(replace(base, reference_wind_velocity=[12.0, 0.0, 0.0]))
    assert off.mean_reference["status"] == "compared_against_reference_wind"
    assert off.mean_reference["residual"] != 0.0 and off.mean_equilibrium_status == "not_verified"


def test_times_must_be_a_fixed_step_grid():
    spec = _spec()
    bad = np.array(spec.times); bad[5] += 0.3 * 0.01
    with pytest.raises(FloatingTransientError, match="fixed-step"):
        solve_floating_transient(_spec(times=bad))


def test_waterline_guard_uses_every_sample_and_rigid_vertical_displacement():
    q = np.zeros((4, 6)); q[:, 2] = [0.0, 0.5, 1.0, 0.2]; q[2, 4] = 0.1    # pitch 0.1 at sample 2
    pts = np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
    eta = np.zeros((4, 2))
    out = waterline_guard(q, pts, eta, limit=1.5)
    # w_z = q_z + (theta x r)_z = q_z - theta_y x ; sample 2: 1.0 -+ 1.0 -> 0 at x=+10, 2.0 at x=-10
    assert out["max_relative_excursion"] == pytest.approx(2.0)
    assert out["violated"] and out["first_violation_index"] == 2 and out["n_samples_checked"] == 4
    assert not waterline_guard(q, pts, eta, limit=2.5)["violated"]
