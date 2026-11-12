"""Development tests for the experimental marine linear route.

Independent reference: the exponential radiation kernel ``C_r exp(-lambda t)``
is represented exactly by the augmented ODE ``zdot = -lambda z + C_r v``
solved with ``scipy.solve_ivp`` (tight tolerances), never by a duplicate
Newmark implementation. For that kernel the harmonic limits are
``B(omega) = C_r lambda/(lambda^2 + omega^2)`` and
``A(omega) = A_inf - C_r/(lambda^2 + omega^2)`` with the ``exp(-i omega t)``
convention.

These are development evidence for the first linear slice, not scientific
qualification.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from anyfesolver.control import CancellationToken
from anyfesolver.marine_linear import (
    NEWMARK_BETA,
    NEWMARK_GAMMA,
    MarineLinearInputError,
    MarineLinearResult,
    MarineLinearSpec,
    exponential_radiation_kernel,
    solve_marine_linear,
)

LABELS = ("bodyA::surge", "bodyA::heave", "bodyB::surge", "bodyB::heave")
LAM = 1.3


def _matrices() -> dict[str, np.ndarray]:
    mass_dry = np.diag([2.0, 3.0, 1.5, 2.5])
    added = np.array([
        [0.8, 0.15, 0.25, 0.05],
        [0.15, 0.9, 0.05, 0.2],
        [0.25, 0.05, 0.7, 0.1],
        [0.05, 0.2, 0.1, 0.6],
    ])
    damping = np.diag([0.4, 0.5, 0.3, 0.4]) + 0.05 * np.ones((4, 4))
    damping = 0.5 * (damping + damping.T)
    stiffness = np.diag([0.3, 2.0, 0.2, 1.8])
    stiffness[0, 2] = stiffness[2, 0] = -0.8
    stiffness[0, 0] += 0.8
    stiffness[2, 2] += 0.8
    cr = np.array([
        [0.6, 0.1, 0.2, 0.05],
        [0.1, 0.5, 0.05, 0.15],
        [0.2, 0.05, 0.4, 0.1],
        [0.05, 0.15, 0.1, 0.3],
    ])
    return {
        "mass_dry": mass_dry,
        "added": added,
        "damping": damping,
        "stiffness": stiffness,
        "cr": cr,
        "total_mass": mass_dry + added,
    }


def _spec(dt: float, n_steps: int, *, forces=None, q0=None, v0=None,
          lam: float = LAM, labels=LABELS, cr=None, zero_kernel: bool = False,
          **overrides) -> MarineLinearSpec:
    m = _matrices()
    times = np.arange(n_steps) * dt
    if forces is None:
        forces = np.zeros((n_steps, len(labels)))
    if cr is None:
        cr = m["cr"]
    kernel = (np.zeros((n_steps, len(labels), len(labels))) if zero_kernel
               else exponential_radiation_kernel(cr, lam, dt, n_steps))
    fields = dict(
        labels=labels, dt=dt, times=times, forces=forces,
        mass_dry=m["mass_dry"], added_mass_infinite=m["added"],
        damping=m["damping"], stiffness=m["stiffness"],
        radiation_kernel=kernel, kernel_times=np.arange(n_steps) * dt,
        initial_displacement=q0, initial_velocity=v0,
    )
    fields.update(overrides)
    return MarineLinearSpec(**fields)


def _augmented_reference(m: dict[str, np.ndarray], times: np.ndarray,
                         force_fn, q0: np.ndarray, v0: np.ndarray,
                         lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact exponential-kernel reference via the augmented ODE."""
    mass = m["total_mass"]
    damping = m["damping"]
    stiffness = m["stiffness"]
    cr = m["cr"]
    ndof = mass.shape[0]
    mass_inv = np.linalg.inv(mass)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        q = y[:ndof]
        v = y[ndof:2 * ndof]
        z = y[2 * ndof:]
        acc = mass_inv @ (np.asarray(force_fn(t), dtype=float) - damping @ v
                          - stiffness @ q - z)
        return np.concatenate([v, acc, -lam * z + cr @ v])

    y0 = np.concatenate([np.asarray(q0, dtype=float),
                         np.asarray(v0, dtype=float), np.zeros(ndof)])
    sol = solve_ivp(rhs, (times[0], times[-1]), y0, t_eval=times,
                    method="DOP853", rtol=1.0e-11, atol=1.0e-13)
    assert sol.success, sol.message
    return sol.y[:ndof].T, sol.y[ndof:2 * ndof].T


def _coupled_force(t: float) -> np.ndarray:
    return np.array([np.sin(0.8 * t), 0.0, 0.0, 0.3])


def _coupled_case(dt: float, n_steps: int) -> tuple[MarineLinearSpec, np.ndarray]:
    times = np.arange(n_steps) * dt
    forces = np.array([_coupled_force(t) for t in times])
    q0 = np.array([0.05, -0.02, 0.0, 0.01])
    v0 = np.array([0.1, 0.0, 0.0, 0.05])
    spec = _spec(dt, n_steps, forces=forces, q0=q0, v0=v0)
    return spec, times


def test_newmark_constants_are_average_acceleration() -> None:
    assert NEWMARK_BETA == 0.25
    assert NEWMARK_GAMMA == 0.5


def test_no_memory_oscillator_matches_analytic_solutions() -> None:
    omega = 2.0
    dt = 0.005
    n_steps = 1001  # t in [0, 5]
    q0 = np.array([1.0])
    v0 = np.array([0.5])
    spec = _spec(dt, n_steps, labels=("osc::x",), q0=q0, v0=v0,
                 zero_kernel=True,
                 mass_dry=np.array([[1.0]]),
                 added_mass_infinite=np.array([[0.0]]),
                 damping=np.array([[0.0]]),
                 stiffness=np.array([[omega ** 2]]))
    result = solve_marine_linear(spec)
    assert result.complete and not result.cancelled
    times = result.times
    analytic = np.cos(omega * times) + (v0[0] / omega) * np.sin(omega * times)
    np.testing.assert_allclose(result.displacements[:, 0], analytic,
                               rtol=1.0e-3, atol=2.0e-4)

    # Constant step load on a stiff SDOF oscillator.
    k = 100.0
    dt_step = 5.0e-4
    n_step = 101  # t in [0, 0.05]
    forces = np.ones((n_step, 1))
    spec_step = _spec(dt_step, n_step, forces=forces, labels=("rod::ux",),
                     zero_kernel=True, mass_dry=np.array([[1.0]]),
                     added_mass_infinite=np.array([[0.0]]),
                     damping=np.array([[0.0]]),
                     stiffness=np.array([[k]]))
    result_step = solve_marine_linear(spec_step)
    expected = (1.0 / k) * (1.0 - np.cos(10.0 * result_step.times[-1]))
    assert result_step.displacements[-1, 0] == pytest.approx(expected, rel=1.0e-3)


def test_temporal_convergence_against_augmented_ode() -> None:
    m = _matrices()
    errors = []
    for dt in (0.02, 0.01, 0.005):
        n_steps = int(round(4.0 / dt)) + 1
        spec, times = _coupled_case(dt, n_steps)
        result = solve_marine_linear(spec)
        assert result.complete
        q_ref, v_ref = _augmented_reference(m, times, _coupled_force,
                                            spec.initial_displacement,
                                            spec.initial_velocity, LAM)
        errors.append(float(np.max(np.abs(result.displacements[-1] - q_ref[-1]))))
    coarse, mid, fine = errors
    assert fine < mid < coarse, f"displacement error not decreasing: {errors}"
    halving_ratios = (coarse / mid, mid / fine)
    assert all(1.5 <= ratio <= 10.0 for ratio in halving_ratios), (
        f"not second order against the augmented ODE: {errors}, "
        f"ratios {halving_ratios}"
    )
    assert coarse / fine >= 4.0
    assert fine < 5.0e-4, f"finest-grid error too large: {fine}"


def test_off_diagonal_coupled_response_matches_reference() -> None:
    m = _matrices()
    dt = 0.005
    n_steps = 801  # t in [0, 4]
    times = np.arange(n_steps) * dt

    def surge_only(t: float) -> np.ndarray:
        return np.array([np.sin(0.8 * t), 0.0, 0.0, 0.0])

    forces = np.array([surge_only(t) for t in times])
    q0 = np.zeros(4)
    v0 = np.array([0.1, 0.0, 0.0, 0.05])
    spec = _spec(dt, n_steps, forces=forces, q0=q0, v0=v0)
    result = solve_marine_linear(spec)
    assert result.complete

    # Full cross-DOF coupling: every DOF, including both bodyB DOFs,
    # carries a real response from the single excited DOF.
    for index in range(4):
        assert float(np.max(np.abs(result.displacements[:, index]))) > 1.0e-6

    q_ref, v_ref = _augmented_reference(m, times, surge_only, q0, v0, LAM)
    scale_q = max(1.0, float(np.max(np.abs(q_ref[-1]))))
    scale_v = max(1.0, float(np.max(np.abs(v_ref[-1]))))
    np.testing.assert_allclose(result.displacements[-1], q_ref[-1],
                               rtol=1.0e-3, atol=1.0e-3 * scale_q)
    np.testing.assert_allclose(result.velocities[-1], v_ref[-1],
                               rtol=1.0e-3, atol=1.0e-3 * scale_v)


def test_result_contract_and_force_balance() -> None:
    dt = 0.01
    n_steps = 401
    spec, _ = _coupled_case(dt, n_steps)
    m = _matrices()
    result = solve_marine_linear(spec)

    assert isinstance(result, MarineLinearResult)
    assert result.labels == LABELS
    assert result.dt == dt
    assert result.complete is True
    assert result.cancelled is False
    assert result.n_steps == n_steps
    assert result.n_accepted == n_steps
    np.testing.assert_array_equal(result.times, np.arange(n_steps) * dt)

    for name in ("displacements", "velocities", "accelerations",
                 "external_forces", "dry_inertia_forces",
                 "added_inertia_forces", "memory_forces", "damping_forces",
                 "stiffness_forces", "residual_forces", "times"):
        array = getattr(result, name)
        assert array.shape[0] == n_steps
        assert not array.flags.writeable, f"{name} must be read-only"
        with pytest.raises(ValueError):
            array[0, ...] = 0.0 if array.ndim > 1 else 0.0

    np.testing.assert_allclose(result.external_forces, spec.forces, atol=0.0)
    np.testing.assert_allclose(result.dry_inertia_forces,
                               result.accelerations @ m["mass_dry"].T, atol=1.0e-12)
    np.testing.assert_allclose(result.added_inertia_forces,
                               result.accelerations @ m["added"].T, atol=1.0e-12)
    np.testing.assert_allclose(result.damping_forces,
                               result.velocities @ m["damping"].T, atol=1.0e-12)
    np.testing.assert_allclose(result.stiffness_forces,
                               result.displacements @ m["stiffness"].T, atol=1.0e-12)
    total = (result.dry_inertia_forces + result.added_inertia_forces
             + result.damping_forces + result.stiffness_forces
             + result.memory_forces)
    np.testing.assert_allclose(result.residual_forces,
                               result.external_forces - total, atol=1.0e-12)
    assert result.balance_relative_max < 1.0e-10


def test_caller_arrays_are_not_mutated() -> None:
    dt = 0.01
    n_steps = 201
    spec, _ = _coupled_case(dt, n_steps)
    before = {name: np.array(getattr(spec, name), copy=True)
              for name in ("times", "forces", "mass_dry",
                           "added_mass_infinite", "damping", "stiffness",
                           "radiation_kernel", "kernel_times",
                           "initial_displacement", "initial_velocity")}
    result = solve_marine_linear(spec)
    for name, snapshot in before.items():
        np.testing.assert_array_equal(getattr(spec, name), snapshot,
                                       err_msg=f"caller array {name} mutated")
    assert not np.shares_memory(result.external_forces, spec.forces)
    assert not np.shares_memory(result.displacements, spec.initial_displacement)


def _valid_two_dof_spec() -> MarineLinearSpec:
    labels = ("bodyA::surge", "bodyA::heave")
    n_steps = 20
    dt = 0.05
    return _spec(dt, n_steps, labels=labels,
                 mass_dry=np.diag([2.0, 3.0]),
                 added_mass_infinite=np.array([[0.8, 0.15], [0.15, 0.9]]),
                 damping=np.diag([0.4, 0.5]),
                 stiffness=np.diag([0.3, 2.0]),
                 cr=np.array([[0.6, 0.1], [0.1, 0.5]]))


def _invalid_specs() -> list[tuple[dict, str]]:
    base = dict(
        labels=("bodyA::surge", "bodyA::heave"),
        dt=0.05,
        times=np.arange(20) * 0.05,
        forces=np.ones((20, 2)),
        mass_dry=np.diag([2.0, 3.0]),
        added_mass_infinite=np.array([[0.8, 0.15], [0.15, 0.9]]),
        damping=np.diag([0.4, 0.5]),
        stiffness=np.diag([0.3, 2.0]),
        radiation_kernel=exponential_radiation_kernel(
            np.array([[0.6, 0.1], [0.1, 0.5]]), 1.3, 0.05, 20),
        kernel_times=np.arange(20) * 0.05,
        initial_displacement=None,
        initial_velocity=None,
    )
    cases = []

    def case(**overrides):
        match = overrides.pop("_match")
        fields = dict(base)
        fields.update(overrides)
        cases.append((fields, match))

    case(labels=("a::x", "a::x"), _match="labels.*unique")
    case(labels=("a::x", 3), _match="labels.*non-empty string")
    case(labels=(), _match="labels.*at least one")
    case(dt=0.0, _match="dt.*strictly positive")
    case(dt=float("nan"), _match="dt.*finite")
    case(dt=-0.1, _match="dt.*strictly positive")
    times = np.arange(20) * 0.05
    times[7] += 1.0e-3
    case(times=times, _match="times.*fixed grid")
    forces = np.ones((20, 2))
    forces[3, 1] = np.nan
    case(forces=forces, _match="forces.*non-finite")
    case(forces=np.ones((19, 2)), _match="forces.*expected shape")
    case(mass_dry=np.diag([1.0, 2.0, 3.0]), _match="mass_dry.*expected shape")
    damping = np.diag([0.4, 0.5])
    damping[0, 1] = 0.2
    case(damping=damping, _match="damping.*not symmetric")
    stiffness = np.diag([0.3, 2.0])
    stiffness[1, 1] = np.inf
    case(stiffness=stiffness, _match="stiffness.*non-finite")
    case(radiation_kernel=exponential_radiation_kernel(
        np.array([[0.6, 0.1], [0.1, 0.5]]), 1.3, 0.05, 10),
        kernel_times=np.arange(10) * 0.05,
        _match="radiation_kernel.*cover every requested lag")
    kernel_times = np.arange(20) * 0.05
    kernel_times[4] = 0.25
    case(kernel_times=kernel_times, _match="kernel_times.*fixed grid")
    kernel = exponential_radiation_kernel(
        np.array([[0.6, 0.1], [0.1, 0.5]]), 1.3, 0.05, 20)
    kernel[5, 0, 1] += 0.3
    case(radiation_kernel=kernel, _match="radiation_kernel.*not symmetric")
    case(added_mass_infinite=np.array([[-2.0, 0.0], [0.0, 0.5]]),
         _match="total mass.*not positive")
    case(mass_dry=np.zeros((2, 2)),
         added_mass_infinite=np.zeros((2, 2)),
         _match="total mass.*not positive")
    case(initial_velocity=np.array([0.1, np.nan]),
         _match="initial_velocity.*non-finite")
    case(initial_displacement=np.array([0.1, 0.2, 0.3]),
         _match="initial_displacement.*expected shape")
    case(radiation_kernel=np.zeros((20, 2, 3)),
         _match="radiation_kernel.*expected shape")
    return cases


@pytest.mark.parametrize("fields,match", _invalid_specs())
def test_invalid_inputs_fail_admission(fields, match) -> None:
    with pytest.raises(MarineLinearInputError, match=match):
        solve_marine_linear(MarineLinearSpec(**fields))


def test_cancellation_returns_incomplete_accepted_history() -> None:
    dt = 0.01
    n_steps = 60
    spec, _ = _coupled_case(dt, n_steps)
    full = solve_marine_linear(spec)
    assert full.complete

    polls = {"count": 0}

    def cancel_after_three() -> bool:
        polls["count"] += 1
        return polls["count"] >= 3

    partial = solve_marine_linear(spec, cancel=cancel_after_three)
    assert partial.cancelled is True
    assert partial.complete is False
    assert partial.n_accepted == 3
    assert partial.n_steps == n_steps
    assert polls["count"] == 3  # polled at accepted-step boundaries only
    np.testing.assert_array_equal(partial.times, full.times[:3])
    for name in ("displacements", "velocities", "accelerations",
                 "external_forces", "memory_forces", "residual_forces"):
        assert getattr(partial, name).shape == (3, 4)
        np.testing.assert_array_equal(getattr(partial, name)[:3],
                                      getattr(full, name)[:3])

    # Immediate cancellation still returns the initial accepted state.
    immediate = solve_marine_linear(spec, cancel=lambda: True)
    assert immediate.cancelled is True
    assert immediate.complete is False
    assert immediate.n_accepted == 1
    np.testing.assert_array_equal(immediate.times, full.times[:1])

    # anyfesolver.control.CancellationToken is accepted directly.
    token = CancellationToken()
    token.cancel("demo stop")
    by_token = solve_marine_linear(spec, cancel=token)
    assert by_token.cancelled is True
    assert by_token.complete is False
    assert by_token.n_accepted == 1

    # Token cancelled from inside a boundary callback mid-run.
    token2 = CancellationToken()
    state = {"count": 0}

    def cancel_via_token() -> bool:
        state["count"] += 1
        if state["count"] >= 5:
            token2.cancel("mid-run stop")
        return token2.is_cancelled

    mid = solve_marine_linear(spec, cancel=cancel_via_token)
    assert mid.cancelled is True
    assert mid.complete is False
    assert mid.n_accepted == 5
    np.testing.assert_array_equal(mid.displacements, full.displacements[:5])


def test_free_decay_work_balance_against_reference() -> None:
    m = _matrices()
    dt = 0.002
    n_steps = 1501  # t in [0, 3]
    times = np.arange(n_steps) * dt
    q0 = np.array([0.05, -0.02, 0.0, 0.01])
    spec = _spec(dt, n_steps, q0=q0, v0=np.zeros(4))
    result = solve_marine_linear(spec)
    assert result.complete

    q_ref, _ = _augmented_reference(m, times, lambda t: np.zeros(4), q0,
                                    np.zeros(4), LAM)
    np.testing.assert_allclose(result.displacements[-1], q_ref[-1],
                               rtol=1.0e-4, atol=1.0e-4)

    # Work/energy balance: E0 - E(T) = dissipated by damping + memory.
    ke = 0.5 * np.einsum("ti,ij,tj->t", result.velocities, m["total_mass"],
                         result.velocities)
    pe = 0.5 * np.einsum("ti,ij,tj->t", result.displacements, m["stiffness"],
                         result.displacements)
    diss_damp = np.trapezoid(
        np.einsum("ti,ti->t", result.velocities, result.damping_forces),
        times)
    diss_mem = np.trapezoid(
        np.einsum("ti,ti->t", result.velocities, result.memory_forces), times)
    closure = (ke[0] + pe[0]) - (ke[-1] + pe[-1]) - diss_damp - diss_mem
    scale = ke[0] + pe[0]
    assert abs(closure) / scale < 5.0e-3, (
        f"work balance not closed: closure={closure}, scale={scale}")


def test_harmonic_steady_state_matches_frequency_domain_reference() -> None:
    # 2-DOF coupled body; independent frequency-domain check of the memory
    # convention using B(omega) and A(omega) of the exponential kernel.
    lam = 1.1
    omega = 1.7
    dt = 0.01
    n_steps = 4001  # t in [0, 40]
    times = np.arange(n_steps) * dt
    labels = ("bodyA::surge", "bodyA::heave")
    mass_dry = np.diag([2.0, 3.0])
    added = np.array([[0.8, 0.15], [0.15, 0.9]])
    damping = np.diag([0.5, 0.6])
    stiffness = np.diag([0.3, 2.0])
    cr = np.array([[0.6, 0.1], [0.1, 0.5]])
    f0 = np.array([1.0, 0.0])
    forces = np.zeros((n_steps, 2))
    forces[:, 0] = f0[0] * np.cos(omega * times)
    spec = _spec(dt, n_steps, labels=labels, forces=forces,
                 mass_dry=mass_dry, added_mass_infinite=added,
                 damping=damping, stiffness=stiffness, cr=cr, lam=lam)
    result = solve_marine_linear(spec)
    assert result.complete

    # Extract the complex steady-state amplitude over the last periods.
    period = 2.0 * np.pi / omega
    window = times >= times[-1] - 4.0 * period
    tw = times[window]
    integrand = result.displacements[window] * np.exp(1j * omega * tw)[:, None]
    qhat_num = 2.0 * np.trapezoid(integrand, tw, axis=0) / (tw[-1] - tw[0])

    denom = lam ** 2 + omega ** 2
    a_omega = added - cr / denom
    b_omega = cr * lam / denom
    system = (-omega ** 2 * (mass_dry + a_omega)
              - 1j * omega * (damping + b_omega) + stiffness)
    qhat_ref = np.linalg.solve(system, f0)

    relative = float(np.linalg.norm(qhat_num - qhat_ref)
                     / np.linalg.norm(qhat_ref))
    assert relative < 1.0e-2, (
        f"steady-state response deviates from A(omega)/B(omega) reference: "
        f"relative={relative}, num={qhat_num}, ref={qhat_ref}")


def test_exponential_kernel_builder_values_and_refusal() -> None:
    cr = np.array([[0.6, 0.1], [0.1, 0.5]])
    kernel = exponential_radiation_kernel(cr, 1.3, 0.05, 5)
    assert kernel.shape == (5, 2, 2)
    expected = np.array([np.exp(-1.3 * k * 0.05) for k in range(5)])
    np.testing.assert_allclose(kernel[:, 0, 0], 0.6 * expected, atol=1.0e-15)
    np.testing.assert_allclose(kernel[:, 1, 0], 0.1 * expected, atol=1.0e-15)
    with pytest.raises(MarineLinearInputError, match="decay.*strictly positive"):
        exponential_radiation_kernel(cr, 0.0, 0.05, 5)
    with pytest.raises(MarineLinearInputError, match="decay.*finite"):
        exponential_radiation_kernel(cr, float("nan"), 0.05, 5)
    with pytest.raises(MarineLinearInputError, match="coefficient_matrix"):
        exponential_radiation_kernel(np.ones((2, 3)), 1.3, 0.05, 5)
    with pytest.raises(MarineLinearInputError, match="count.*>= 1"):
        exponential_radiation_kernel(cr, 1.3, 0.05, 0)
    with pytest.raises(MarineLinearInputError, match="count.*integer"):
        exponential_radiation_kernel(cr, 1.3, 0.05, 20.5)


def test_cancel_contract_rejects_unsupported_callback_types() -> None:
    spec = _valid_two_dof_spec()
    with pytest.raises(MarineLinearInputError, match="cancel.*callable"):
        solve_marine_linear(spec, cancel=42)
