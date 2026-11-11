"""Experimental fixed-grid linear marine radiation-dynamics route.

Development slice for the ANYmarine roadmap's first linear two-body
problem, restricted to constant equilibrium-frame matrices, stable unique
DOF labels and fixed-step Newmark integration (beta = 1/4, gamma = 1/2)
with a direct causal trapezoidal radiation-memory convolution including
the current-step tangent term.

Equation of motion (SI units, common equilibrium frame):

    (M_dry + A_inf) a + C v + K q + memory = F

    memory_n = dt*(0.5*K_r[n] v_0 + sum(j=1..n-1, K_r[n-j] v_j)
                   + 0.5*K_r[0] v_n),  n >= 1,  memory_0 = 0

``K`` is the separately assembled linear hydrostatic plus mechanical link
stiffness, ``A_inf`` is only the instantaneous radiation added mass, and
``K_r`` is the retardation kernel sampled on the same fixed grid. Full
cross-DOF dense matrices are retained.

Scope boundary (fail closed): no finite-rotation mechanics, Morison
loads, lines, contact, adaptive stepping, restart or memory
initialization is supported, and none may be inferred from this module.
This module imports no Capytaine, ANYhydro or Qt code and does not use the
shared structural integrator. See ``docs/MARINE_LINEAR_DEVELOPMENT.md``
for the full contract, admission rules and limitations. Tests of this
module are development evidence, not scientific qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve

__all__ = [
    "NEWMARK_BETA",
    "NEWMARK_GAMMA",
    "MarineLinearError",
    "MarineLinearInputError",
    "MarineLinearResult",
    "MarineLinearSolveError",
    "MarineLinearSpec",
    "exponential_radiation_kernel",
    "solve_marine_linear",
]

NEWMARK_BETA = 0.25
NEWMARK_GAMMA = 0.5

# Relative tolerance for admitting near-symmetric physical matrices and
# near-uniform time/kernel grids. Matrices are used exactly as given; the
# tolerance only bounds admission, it never silently repairs input.
_SYMMETRY_RELATIVE_TOLERANCE = 1.0e-10
_GRID_RELATIVE_TOLERANCE = 1.0e-9


class MarineLinearError(ValueError):
    """Base class for marine linear route failures."""


class MarineLinearInputError(MarineLinearError):
    """Actionable admission failure for marine linear inputs."""


class MarineLinearSolveError(MarineLinearError):
    """Numerical failure inside the fixed-step integration."""


@dataclass(frozen=True)
class MarineLinearSpec:
    """Validated-at-solve input contract for the linear marine route.

    All matrices are constant coefficients in the common equilibrium frame
    of the linearized model. ``mass_dry`` is the structural (dry) inertia
    matrix and ``added_mass_infinite`` is the instantaneous radiation added
    mass ``A_inf``; they are kept separate and both are required to be
    symmetric with a positive-definite sum. ``radiation_kernel`` holds the
    retardation matrices ``K_r[k] = K_r(k*dt)`` on ``kernel_times``.
    """

    labels: Sequence[str]
    dt: float
    times: np.ndarray
    forces: np.ndarray
    mass_dry: np.ndarray
    added_mass_infinite: np.ndarray
    damping: np.ndarray
    stiffness: np.ndarray
    radiation_kernel: np.ndarray
    kernel_times: np.ndarray
    initial_displacement: np.ndarray | None = None
    initial_velocity: np.ndarray | None = None


@dataclass(frozen=True)
class MarineLinearResult:
    """Read-only result of a (possibly incomplete) marine linear run.

    ``displacements``/``velocities``/``accelerations`` are q/v/a on the
    accepted rows of the fixed grid. Force components decompose the
    equation of motion per accepted step: dry inertia ``M_dry a``,
    instantaneous added inertia ``A_inf a``, radiation memory, linear
    damping ``C v`` and stiffness ``K q``. ``residual_forces`` is
    ``external - (sum of components)`` and ``balance_relative_max`` is the
    largest per-step residual norm relative to the largest force-term norm
    at that step (floor 1.0). A cancelled run returns ``cancelled=True``
    and ``complete=False`` with exactly the accepted history.
    """

    labels: tuple[str, ...]
    dt: float
    times: np.ndarray
    displacements: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    external_forces: np.ndarray
    dry_inertia_forces: np.ndarray
    added_inertia_forces: np.ndarray
    memory_forces: np.ndarray
    damping_forces: np.ndarray
    stiffness_forces: np.ndarray
    residual_forces: np.ndarray
    balance_relative_max: float
    n_steps: int
    n_accepted: int
    complete: bool
    cancelled: bool


def _owned_float_array(name: str, value: Any, *, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MarineLinearInputError(
            f"{name}: cannot be read as a float array ({exc}); supply numeric "
            "array-like input"
        ) from exc
    if array.ndim != ndim:
        raise MarineLinearInputError(
            f"{name}: expected a {ndim}-D array, received shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise MarineLinearInputError(
            f"{name}: {bad} non-finite (NaN/inf) value(s); finite input is "
            "required and none are admitted"
        )
    return np.array(array, dtype=float, copy=True, order="C")


def _validated_labels(labels: Any) -> tuple[str, ...]:
    try:
        items = tuple(labels)
    except TypeError as exc:
        raise MarineLinearInputError(
            f"labels: not iterable ({exc}); supply a sequence of DOF label strings"
        ) from exc
    if not items:
        raise MarineLinearInputError("labels: at least one DOF label is required")
    for item in items:
        if not isinstance(item, str) or not item:
            raise MarineLinearInputError(
                "labels: every DOF label must be a non-empty string; received "
                f"{item!r}"
            )
    if len(set(items)) != len(items):
        raise MarineLinearInputError(
            "labels: DOF labels must be unique; received duplicates"
        )
    return items


def _validated_dt(dt: Any) -> float:
    try:
        value = float(dt)
    except (TypeError, ValueError) as exc:
        raise MarineLinearInputError(
            f"dt: not a number ({exc}); supply a positive fixed time step"
        ) from exc
    if not np.isfinite(value) or value <= 0.0:
        raise MarineLinearInputError(
            f"dt: must be finite and strictly positive, received {value!r}"
        )
    return value


def _grid_tolerance(reference: float) -> float:
    return _GRID_RELATIVE_TOLERANCE * max(1.0, abs(reference))


def _validated_uniform_grid(
    name: str, values: np.ndarray, dt: float, *, start: float, count: int
) -> None:
    expected = start + np.arange(count, dtype=float) * dt
    if not np.allclose(values, expected, rtol=0.0, atol=_grid_tolerance(max(abs(start), 1.0)) + 1.0e-12):
        first = int(np.argmax(~np.isclose(values, expected, rtol=0.0,
                                         atol=_grid_tolerance(max(abs(start), 1.0)) + 1.0e-12)))
        raise MarineLinearInputError(
            f"{name}: must match the fixed grid {name}[k] = "
            f"{start!r} + k*dt with dt={dt!r}; first mismatch at index {first} "
            f"({values[first]!r} != {expected[first]!r})"
        )


def _validated_matrix(name: str, value: Any, ndof: int) -> np.ndarray:
    array = _owned_float_array(name, value, ndim=2)
    if array.shape != (ndof, ndof):
        raise MarineLinearInputError(
            f"{name}: expected shape ({ndof}, {ndof}) matching the DOF labels, "
            f"received {array.shape}"
        )
    asymmetry = float(np.max(np.abs(array - array.T)))
    scale = max(1.0, float(np.max(np.abs(array))))
    if asymmetry > _SYMMETRY_RELATIVE_TOLERANCE * scale:
        raise MarineLinearInputError(
            f"{name}: not symmetric (max |M - M^T| = {asymmetry:.3e} exceeds "
            f"{_SYMMETRY_RELATIVE_TOLERANCE:.1e} relative); supply a symmetric "
            "physical matrix"
        )
    return array


def _validated_vector(name: str, value: Any, ndof: int) -> np.ndarray:
    array = _owned_float_array(name, value, ndim=1)
    if array.shape != (ndof,):
        raise MarineLinearInputError(
            f"{name}: expected shape ({ndof},) matching the DOF labels, "
            f"received {array.shape}"
        )
    return array


class _Validated:
    """Owned copies of every admitted input; caller arrays are untouched."""

    def __init__(self, spec: MarineLinearSpec) -> None:
        self.labels = _validated_labels(spec.labels)
        self.ndof = len(self.labels)
        self.dt = _validated_dt(spec.dt)

        self.times = _owned_float_array("times", spec.times, ndim=1)
        self.n_steps = int(self.times.shape[0])
        if self.n_steps < 1:
            raise MarineLinearInputError("times: at least the initial time t0 is required")
        _validated_uniform_grid("times", self.times, self.dt,
                                start=float(self.times[0]), count=self.n_steps)

        self.forces = _owned_float_array("forces", spec.forces, ndim=2)
        if self.forces.shape != (self.n_steps, self.ndof):
            raise MarineLinearInputError(
                f"forces: expected shape ({self.n_steps}, {self.ndof}) matching "
                f"times and DOF labels, received {self.forces.shape}"
            )

        self.mass_dry = _validated_matrix("mass_dry", spec.mass_dry, self.ndof)
        self.added_mass_infinite = _validated_matrix(
            "added_mass_infinite", spec.added_mass_infinite, self.ndof
        )
        self.damping = _validated_matrix("damping", spec.damping, self.ndof)
        self.stiffness = _validated_matrix("stiffness", spec.stiffness, self.ndof)

        self.kernel = _owned_float_array("radiation_kernel", spec.radiation_kernel, ndim=3)
        self.kernel_times = _owned_float_array("kernel_times", spec.kernel_times, ndim=1)
        if self.kernel_times.shape[0] != self.kernel.shape[0]:
            raise MarineLinearInputError(
                "radiation_kernel/kernel_times: kernel stack length "
                f"{self.kernel.shape[0]} must match kernel_times length "
                f"{self.kernel_times.shape[0]}"
            )
        if self.kernel.shape[1:] != (self.ndof, self.ndof):
            raise MarineLinearInputError(
                f"radiation_kernel: expected shape (n_kernel, {self.ndof}, "
                f"{self.ndof}), received {self.kernel.shape}"
            )
        if self.kernel.shape[0] < self.n_steps:
            raise MarineLinearInputError(
                f"radiation_kernel: {self.kernel.shape[0]} samples cover only "
                f"{self.kernel.shape[0]} lags but {self.n_steps} steps are "
                "requested; the kernel grid must cover every requested lag"
            )
        _validated_uniform_grid("kernel_times", self.kernel_times, self.dt,
                                start=0.0, count=self.kernel.shape[0])
        for index in range(self.kernel.shape[0]):
            slice_asymmetry = float(np.max(np.abs(self.kernel[index] - self.kernel[index].T)))
            slice_scale = max(1.0, float(np.max(np.abs(self.kernel[index]))))
            if slice_asymmetry > _SYMMETRY_RELATIVE_TOLERANCE * slice_scale:
                raise MarineLinearInputError(
                    f"radiation_kernel: sample {index} is not symmetric "
                    f"(max asymmetry {slice_asymmetry:.3e}); supply symmetric "
                    "retardation matrices"
                )

        if spec.initial_displacement is None:
            self.q0 = np.zeros(self.ndof)
        else:
            self.q0 = _validated_vector("initial_displacement", spec.initial_displacement, self.ndof)
        if spec.initial_velocity is None:
            self.v0 = np.zeros(self.ndof)
        else:
            self.v0 = _validated_vector("initial_velocity", spec.initial_velocity, self.ndof)

        self.total_mass = self.mass_dry + self.added_mass_infinite
        try:
            # Documented numeric admission check: Cholesky factorization of
            # the total mass. No eigenvalue is projected or clamped.
            self.mass_factor = cho_factor(self.total_mass, lower=True)
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise MarineLinearInputError(
                "mass_dry + added_mass_infinite: total mass is not positive "
                f"definite (Cholesky failed: {exc}); correct the physical mass "
                "data - the admission check is not adjusted to pass"
            ) from exc


def _cancel_requested(cancel: Any) -> bool:
    if cancel is None:
        return False
    is_cancelled = getattr(cancel, "is_cancelled", None)
    if is_cancelled is not None:
        # anyfesolver.control.CancellationToken exposes a bool property; other
        # token types may expose a callable instead.
        return bool(is_cancelled() if callable(is_cancelled) else is_cancelled)
    if callable(cancel):
        return bool(cancel())
    raise MarineLinearInputError(
        "cancel: must be None, a callable returning a truthy value, or an "
        "anyfesolver.control.CancellationToken; received "
        f"{type(cancel).__name__}"
    )


def _readonly(array: np.ndarray) -> np.ndarray:
    owned = np.array(array, copy=True)
    owned.flags.writeable = False
    return owned


def _memory_past(kernel: np.ndarray, velocities: np.ndarray, n: int, dt: float) -> np.ndarray:
    """Trapezoidal radiation memory at step n excluding the current-step term.

    memory_past_n = dt*(0.5*K_r[n] v_0 + sum(j=1..n-1, K_r[n-j] v_j))
    """
    ndof = velocities.shape[1]
    if n == 0:
        return np.zeros(ndof)
    lag_sum = np.zeros(ndof)
    if n > 1:
        lag_sum = np.einsum("lij,lj->i", kernel[1:n], velocities[n - 1:0:-1])
    return dt * (0.5 * (kernel[n] @ velocities[0]) + lag_sum)


def solve_marine_linear(
    spec: MarineLinearSpec, *, cancel: Any = None
) -> MarineLinearResult:
    """Integrate the linear marine equation of motion on the fixed grid.

    ``cancel`` is polled at accepted-step boundaries only. On cancellation
    the function returns normally (never an exception) with a result whose
    ``cancelled`` is True, ``complete`` is False and whose arrays hold
    exactly the accepted history (always at least the initial state at
    ``t0``). Caller arrays are never mutated; result arrays are fresh
    read-only copies.
    """
    data = _Validated(spec)
    ndof = data.ndof
    dt = data.dt
    n_steps = data.n_steps
    kernel = data.kernel
    kernel0 = kernel[0]

    effective = (
        data.total_mass
        + NEWMARK_GAMMA * dt * (data.damping + 0.5 * dt * kernel0)
        + NEWMARK_BETA * dt * dt * data.stiffness
    )
    try:
        effective_factor = lu_factor(effective)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise MarineLinearInputError(
            "effective Newmark system matrix [M + gamma*dt*(C + dt*K_r[0]/2) "
            "+ beta*dt^2*K] is singular; the admitted matrices do not define "
            f"a solvable fixed-step problem ({exc})"
        ) from exc

    q = np.empty((n_steps, ndof))
    v = np.empty((n_steps, ndof))
    a = np.empty((n_steps, ndof))
    f_dry = np.empty((n_steps, ndof))
    f_added = np.empty((n_steps, ndof))
    f_memory = np.empty((n_steps, ndof))
    f_damping = np.empty((n_steps, ndof))
    f_stiffness = np.empty((n_steps, ndof))
    residual = np.empty((n_steps, ndof))

    def record(n: int, q_n: np.ndarray, v_n: np.ndarray, a_n: np.ndarray,
               memory_n: np.ndarray) -> None:
        f_dry[n] = data.mass_dry @ a_n
        f_added[n] = data.added_mass_infinite @ a_n
        f_damping[n] = data.damping @ v_n
        f_stiffness[n] = data.stiffness @ q_n
        f_memory[n] = memory_n
        residual[n] = data.forces[n] - (
            f_dry[n] + f_added[n] + f_damping[n] + f_stiffness[n] + f_memory[n]
        )

    # Initial state: memory starts at rest at t0 (documented convention).
    a0 = cho_solve(data.mass_factor, data.forces[0] - data.damping @ data.v0
                   - data.stiffness @ data.q0)
    if not np.all(np.isfinite(a0)):
        raise MarineLinearSolveError(
            "non-finite initial acceleration at t0; the total mass is "
            "ill-conditioned against the initial load - no non-finite state "
            "is accepted"
        )
    q[0] = data.q0
    v[0] = data.v0
    a[0] = a0
    record(0, data.q0, data.v0, a0, np.zeros(ndof))

    n_accepted = 1
    cancelled = False
    if _cancel_requested(cancel):
        cancelled = True
    else:
        for n in range(1, n_steps):
            q_pred = q[n - 1] + dt * v[n - 1] + dt * dt * (0.5 - NEWMARK_BETA) * a[n - 1]
            v_pred = v[n - 1] + dt * (1.0 - NEWMARK_GAMMA) * a[n - 1]
            memory_past = _memory_past(kernel, v, n, dt)
            rhs = (
                data.forces[n]
                - memory_past
                - (data.damping + 0.5 * dt * kernel0) @ v_pred
                - data.stiffness @ q_pred
            )
            a_n = lu_solve(effective_factor, rhs)
            if not np.all(np.isfinite(a_n)):
                raise MarineLinearSolveError(
                    f"non-finite acceleration at step {n} "
                    f"(t={float(data.times[n]):.6g}); the effective system "
                    "matrix is ill-conditioned or the admitted input excites "
                    "an unstable mode - no non-finite state is accepted"
                )
            v_n = v_pred + NEWMARK_GAMMA * dt * a_n
            q_n = q_pred + NEWMARK_BETA * dt * dt * a_n
            memory_n = memory_past + 0.5 * dt * (kernel0 @ v_n)
            q[n] = q_n
            v[n] = v_n
            a[n] = a_n
            record(n, q_n, v_n, a_n, memory_n)
            n_accepted = n + 1
            if _cancel_requested(cancel):
                cancelled = True
                break

    balance_relative_max = 0.0
    for n in range(n_accepted):
        scale = max(
            1.0,
            float(np.linalg.norm(data.forces[n])),
            float(np.linalg.norm(f_dry[n] + f_added[n])),
            float(np.linalg.norm(f_damping[n])),
            float(np.linalg.norm(f_stiffness[n])),
            float(np.linalg.norm(f_memory[n])),
        )
        balance_relative_max = max(
            balance_relative_max, float(np.linalg.norm(residual[n])) / scale
        )

    return MarineLinearResult(
        labels=data.labels,
        dt=dt,
        times=_readonly(data.times[:n_accepted]),
        displacements=_readonly(q[:n_accepted]),
        velocities=_readonly(v[:n_accepted]),
        accelerations=_readonly(a[:n_accepted]),
        external_forces=_readonly(data.forces[:n_accepted]),
        dry_inertia_forces=_readonly(f_dry[:n_accepted]),
        added_inertia_forces=_readonly(f_added[:n_accepted]),
        memory_forces=_readonly(f_memory[:n_accepted]),
        damping_forces=_readonly(f_damping[:n_accepted]),
        stiffness_forces=_readonly(f_stiffness[:n_accepted]),
        residual_forces=_readonly(residual[:n_accepted]),
        balance_relative_max=balance_relative_max,
        n_steps=n_steps,
        n_accepted=n_accepted,
        complete=(not cancelled) and n_accepted == n_steps,
        cancelled=cancelled,
    )


def exponential_radiation_kernel(
    coefficient_matrix: Any, decay: Any, dt: Any, count: Any
) -> np.ndarray:
    """Build a retardation kernel stack ``K_r[k] = cr * exp(-decay*k*dt)``.

    Linear-scope convenience for exponential (rank-one-retardation)
    radiation kernels; it admits no physics beyond the supplied coefficient
    matrix and decay rate. ``decay`` must be finite and strictly positive.
    """
    try:
        ndim = int(np.asarray(coefficient_matrix).ndim)
    except (TypeError, ValueError) as exc:
        raise MarineLinearInputError(
            f"coefficient_matrix: cannot be read as an array ({exc}); supply "
            "a square symmetric matrix"
        ) from exc
    if ndim != 2:
        raise MarineLinearInputError(
            f"coefficient_matrix: expected a 2-D square matrix, received "
            f"{ndim} dimensions"
        )
    cr = _validated_matrix(
        "coefficient_matrix", coefficient_matrix,
        int(np.asarray(coefficient_matrix).shape[0]),
    )
    try:
        decay_value = float(decay)
    except (TypeError, ValueError) as exc:
        raise MarineLinearInputError(
            f"decay: not a number ({exc}); supply a positive decay rate"
        ) from exc
    if not np.isfinite(decay_value) or decay_value <= 0.0:
        raise MarineLinearInputError(
            f"decay: must be finite and strictly positive, received {decay_value!r}"
        )
    dt_value = _validated_dt(dt)
    try:
        count_value = int(count)
    except (TypeError, ValueError) as exc:
        raise MarineLinearInputError(
            f"count: not an integer ({exc}); supply the number of kernel samples"
        ) from exc
    if count_value < 1:
        raise MarineLinearInputError(
            f"count: must be >= 1, received {count_value}"
        )
    try:
        integral = float(count) == count_value
    except (TypeError, ValueError):
        integral = True
    if not integral:
        raise MarineLinearInputError(
            f"count: must be an integer, received {count!r}"
        )
    factors = np.exp(-decay_value * np.arange(count_value, dtype=float) * dt_value)
    return np.array(cr[None, :, :] * factors[:, None, None], copy=True)
