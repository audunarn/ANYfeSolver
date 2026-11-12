# Arc-length continuation

`anyfesolver.arc_length` provides bounded Crisfield-style spherical arc-length
continuation for nonlinear static capacity checks. The four public
continuation symbols are also exported at the package root.

See [`THEORY.md`](THEORY.md) for the formulation context and the project
[`README.md`](../README.md) for production limits.

## Intended use

The implementation is intended to:

1. start from an imperfect, restrained FE model;
2. apply an optional constant preload;
3. scale one proportional reference load;
4. cross the first limit point;
5. continue for a small number of converged descending-branch steps; and
6. report the maximum converged load factor as the capacity estimate.

It reuses the nonlinear shell/beam tangent and internal-force routines, layered
shell plasticity, constraint transformation and committed/trial material-state
handling from `nonlinear_static.py`. Dead pressure remains the default.
Setting `LoadCase.follower_pressure = True` evaluates shell pressure on the
current midsurface and includes its exact external-load tangent in the
continuation equations.

Direct `initial_fields` setup, zero-load equilibration, and a
field-bearing restart with matching nonzero displacements are currently
provided only by `solve_static_nonlinear`. Arc-length accepts committed element
state but has no initial-displacement restart parameter; it must therefore not
be used to continue a residual-field state that equilibrated to nonzero
displacement. Geometric-imperfection provenance remains distinct from
residual-stress/prestrain provenance.

## Example

```python
from anyfesolver import ArcLengthControl, solve_static_arc_length

control = ArcLengthControl(
    initial_load_increment=0.025,
    minimum_load_increment=1.0e-4,
    maximum_load_increment=0.10,
    stop_after_peak_steps=4,
    max_steps=100,
)

result = solve_static_arc_length(
    imperfect_model,
    environmental_load,
    constant_load_case=permanent_load,
    control=control,
    max_iterations=30,
    tolerance=1.0e-6,
    num_layers=7,
    kinematics="von_karman",
)

capacity_factor = result.peak_load_factor
```

## Numerical formulation

The reduced equilibrium residual, including configuration-dependent loads, is

```text
R(q, lambda) =
    F_constant(q) + lambda F_reference(q) - F_internal(q)
```

and its effective tangent is

```text
K_effective =
    K_internal - K_external,constant - lambda K_external,reference
```

and the spherical constraint is

```text
dq.T W dq + alpha^2 dlambda^2 = ds^2
```

The correction uses block elimination (all quantities are evaluated at the
current trial state):

```text
K_effective a = R
K_effective b = F_reference(q)

dlambda = (-g - 2 dq.T W a) /
          (2 dq.T W b + 2 alpha^2 Delta_lambda)

dq_correction = a + b dlambda
```

This avoids assembling a bordered matrix. The ordinary dead-load path uses the
symmetric-indefinite matrix class. Current-area follower pressure and the
consistent corotational tangent are generally nonsymmetric and therefore use
the general matrix class. Rotational DOFs are converted to equivalent
translation through a characteristic-length metric before calculating the arc
norm.

Follower pressure uses

```text
f_i(q) = p integral N_i (a_xi x a_eta) dxi deta
```

on each supported shell, with the exact derivative of the current area vector.
There are no independent pressure moments because pressure virtual work is
conjugate to midsurface translations.

For `kinematics="corotational"`,
`corotational_tangent="auto"` keeps the inexpensive rotated tangent for
ordinary loads and selects `consistent` when follower pressure is active. The
consistent mode applies the full pull-back/frame/rotate-forward chain rule,
uses centered differences only for rigid-frame sensitivity, and is generally
nonsymmetric. Explicitly requesting `rotated` with follower pressure fails
closed.

## Current limits

The first production version intentionally does not provide:

- nonlinear free-free/nullspace continuation;
- automatic branch switching at a perfect bifurcation;
- contact or general-purpose path following;
- unrestricted deep post-buckling analysis.

Follower semantics apply to shell pressure loads only; nodal, user-provided
element, gravity, and acceleration loads retain their prescribed directions.
Linear buckling rejects an open nonsymmetric follower-pressure eigenpencil
rather than silently symmetrizing it. The corotational consistent tangent is
more expensive than the rotated approximation because its frame sensitivity
is evaluated numerically.

Use a nonzero eigenmode or standard fabrication imperfection for shell-buckling
capacity work. The principal acceptance result is a confirmed peak followed by
converged descending-branch points (`status == "peak_confirmed"`).

## Post-buckling continuation

Bounded post-buckling tracing is available through two `ArcLengthControl`
fields:

- `post_peak_load_fraction` (in `(0, 1)`): continue past the limit point and
  stop automatically once the load factor has fallen to this fraction of the
  recorded peak (`status == "post_buckling_traced"`).  Combine with a large
  `stop_after_peak_steps` so the descending branch is not cut off by the
  limit-point confirmation counter.
- `max_translation` (metres): absolute guard on the largest nodal translation
  (`status == "displacement_limit_reached"`), protecting against runaway
  post-collapse paths.

Both stop statuses report `converged == True`; the equilibrium path up to the
stop is in `steps`, and a `progress_callback` receives one structured
`nonlinear_static_step` dict per converged step for live plotting.  Seed an
imperfection (eigenmode or fabrication shape) so the buckling mode is present
in the path; post-buckling branches remain imperfection- and mesh-sensitive,
so the same qualification-gate comparisons below apply before design use.

## Qualification gates before design use

The mathematical regression test uses a cubic softening spring with the exact
limit load

```text
lambda_max = 2 / (3 sqrt(3))
```

`NLG-008` additionally checks the follower-pressure force/tangent contract and
a thin-ring pressure-buckling case. These are implementation and analytical
qualification gates, not a substitute for project-specific nonlinear
validation.

Before treating shell results as design values, also compare representative
imperfect plate, stiffened-panel and cylindrical-shell cases against a trusted
nonlinear solver using the same mesh, material curve, imperfection and load
history.
