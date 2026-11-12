"""Deterministic two-body 4-DOF synthetic development example (NOT Capytaine).

Runs the experimental ``anyfesolver.marine_linear`` route on a fixed synthetic
linear two-body model and prints the response summary, the force
decomposition and a cancellation demonstration. This is a development
example with hand-written synthetic matrices: it uses no Capytaine, ANYhydro
or Qt data and makes no qualification claim.

Run with the repository interpreter, for example:

    ./.venv/Scripts/python.exe scripts/marine_linear_demo.py
"""

from __future__ import annotations

import numpy as np

from anyfesolver.control import CancellationToken
from anyfesolver.marine_linear import (
    MarineLinearSpec,
    exponential_radiation_kernel,
    solve_marine_linear,
)

LABELS = ("bodyA::surge", "bodyA::heave", "bodyB::surge", "bodyB::heave")


def build_spec(dt: float, n_steps: int) -> MarineLinearSpec:
    """Fixed synthetic two-body linear model (development data only)."""
    times = np.arange(n_steps) * dt
    forces = np.zeros((n_steps, 4))
    forces[:, 0] = 0.8 * np.sin(0.8 * times)          # wave-frequency surge load
    forces[:, 3] = 0.3 * np.exp(-((times - 2.0) / 0.5) ** 2)  # heave pulse
    mass_dry = np.diag([2.0, 3.0, 1.5, 2.5])
    added_mass_infinite = np.array([
        [0.8, 0.15, 0.25, 0.05],
        [0.15, 0.9, 0.05, 0.2],
        [0.25, 0.05, 0.7, 0.1],
        [0.05, 0.2, 0.1, 0.6],
    ])
    damping = np.diag([0.4, 0.5, 0.3, 0.4]) + 0.05 * np.ones((4, 4))
    damping = 0.5 * (damping + damping.T)
    stiffness = np.diag([0.3, 2.0, 0.2, 1.8])  # hydrostatic + link stiffness
    stiffness[0, 2] = stiffness[2, 0] = -0.8
    stiffness[0, 0] += 0.8
    stiffness[2, 2] += 0.8
    cr = np.array([
        [0.6, 0.1, 0.2, 0.05],
        [0.1, 0.5, 0.05, 0.15],
        [0.2, 0.05, 0.4, 0.1],
        [0.05, 0.15, 0.1, 0.3],
    ])
    kernel = exponential_radiation_kernel(cr, 1.3, dt, n_steps)
    return MarineLinearSpec(
        labels=LABELS,
        dt=dt,
        times=times,
        forces=forces,
        mass_dry=mass_dry,
        added_mass_infinite=added_mass_infinite,
        damping=damping,
        stiffness=stiffness,
        radiation_kernel=kernel,
        kernel_times=np.arange(n_steps) * dt,
        initial_displacement=np.array([0.05, -0.02, 0.0, 0.01]),
        initial_velocity=np.array([0.1, 0.0, 0.0, 0.05]),
    )


def main() -> int:
    dt = 0.005
    n_steps = 1601  # t in [0, 8]
    spec = build_spec(dt, n_steps)
    result = solve_marine_linear(spec)

    print("anyfesolver.marine_linear synthetic two-body development example")
    print("development evidence only - no Capytaine data, no qualification")
    print(f"labels          : {list(result.labels)}")
    print(f"dt / steps      : {result.dt} / {result.n_steps}")
    print(f"complete        : {result.complete}  cancelled: {result.cancelled}")
    print(f"accepted steps  : {result.n_accepted}")
    print(f"balance (rel)   : {result.balance_relative_max:.3e}")
    print()
    print("final state (t = {:.3f} s):".format(result.times[-1]))
    for index, label in enumerate(result.labels):
        print(f"  {label:>16s}  q={result.displacements[-1, index]: .6e}  "
              f"v={result.velocities[-1, index]: .6e}  "
              f"a={result.accelerations[-1, index]: .6e}")
    print()
    print("peak |q| per DOF:")
    for index, label in enumerate(result.labels):
        print(f"  {label:>16s}  {np.max(np.abs(result.displacements[:, index])):.6e}")
    print()
    print("force decomposition at the final step (L2 norms):")
    for name in ("dry_inertia_forces", "added_inertia_forces", "memory_forces",
                 "damping_forces", "stiffness_forces", "external_forces"):
        print(f"  {name:>22s}  {np.linalg.norm(getattr(result, name)[-1]):.6e}")
    print(f"  {'residual_forces':>22s}  "
          f"{np.linalg.norm(result.residual_forces[-1]):.6e}")

    # Cancellation demonstration: stop at an accepted-step boundary.
    polls = {"count": 0}

    def cancel_after_quarter() -> bool:
        polls["count"] += 1
        return polls["count"] >= n_steps // 4

    partial = solve_marine_linear(spec, cancel=cancel_after_quarter)
    print()
    print("cancellation demo (callback observed at accepted-step boundaries):")
    print(f"  cancelled={partial.cancelled}  complete={partial.complete}  "
          f"accepted steps={partial.n_accepted} of {partial.n_steps}")
    print(f"  partial final time = {partial.times[-1]:.3f} s "
          f"(explicitly incomplete history, never a complete run)")

    token = CancellationToken()
    token.cancel("demo stop before the run")
    stopped = solve_marine_linear(spec, cancel=token)
    print(f"  pre-cancelled CancellationToken: accepted steps="
          f"{stopped.n_accepted}, complete={stopped.complete}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
