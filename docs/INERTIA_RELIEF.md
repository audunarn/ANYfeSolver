# Linear inertia relief

`solve_inertia_relief` computes the constant rigid acceleration of a free body,
balances its applied load with mass-derived inertial loads, and solves for the
elastic deformation in the accelerating frame. It supports one connected
free-free linear beam/shell assembly, including homogeneous internal MPCs.

```python
from anyfesolver import solve_inertia_relief

result = solve_inertia_relief(model, load_case)
if result.converged:
    print(result.acceleration_at_center_of_mass)  # global m/s²
    print(result.angular_acceleration)           # global rad/s²
    elastic = result.recover()                  # existing stress/strain recovery
else:
    print(result.status, result.error, result.diagnostics)
```

The full signature is:

```python
solve_inertia_relief(
    model, load_case=None, *, reference_point=None, session=None,
    resource_config=None, cancellation_token=None, progress_callback=None,
) -> InertiaReliefResult
```

## Physical equations and coordinates

The six physical columns of `R` are ordered `Tx, Ty, Tz, Rx, Ry, Rz`. At reference
point `O`, a nodal acceleration is `a_O + alpha × (x - O)`, with nodal angular
acceleration `alpha`. Using the canonical assembled mass matrix:

```text
G = R.T M R
G a = R.T F
F_inertia = -M R a
F_effective = F + F_inertia
```

The reference defaults to the centre of mass. An explicit finite global
`reference_point=(x,y,z)` changes the reported translational acceleration and
moment reference; it preserves physical acceleration fields, inertial loads and
elastic strains. The solve uses a centred generalized-coordinate congruence of
the full coupled rigid-inertia equation, avoiding large global-origin moments.
Section rotary inertia, generalized-section mass coupling, point masses and the
model's consistent/lumped selection all enter the same `M`. In particular, the
default two-node beam uses its existing lumped formulation unless its section
selects `consistent_mass=True`. Inertia relief does not change that selection.

After homogeneous MPC reduction `u=Tq`, the existing augmented nullspace solve
enforces `Q.T q=0`. `Q` is an orthonormal gauge basis, and its coefficients are
not physical accelerations. Reported displacements describe elastic deformation
in this gauge; they do not include an integrated free-flight trajectory.

## Admission, status and recovery

Physical supports, nonzero affine offsets, constraints removing rigid directions,
disconnected/isolated nodes, solid models, native nonlinear configurations,
follower pressure and load-case-only `added_node_masses` are rejected with
`InertiaReliefError`. Attach physical point masses through `model.add_point_mass`.
The native-element exclusion does not exclude a reference-linear profile merely
because it shares a shell-family name. The current public S3 factory selects the
linear V2D profile; supply its authoritative `reference_normal`. Manufactured
shell acceptance covers that S3 profile and E4-PL Q4 elements. Dormant native
total-Lagrangian prototypes retain their exclusion.
Both declared and directly assigned point masses must be finite, nonnegative and
reference existing nodes. Total mass must be positive and all six rigid-inertia
directions numerically resolved. Semidefinite full mass matrices are permitted;
the algorithm never inverts the full `M`.

Numerical failure returns a nonconverged result whose `displacements` is `None`.
For example, `unresolved_elastic_stiffness` identifies an additional/negative/unresolved
elastic stiffness mode, or an unconverged bounded Lanczos classification. No
artificial support or physical stiffness is introduced to make a case pass.
The rank-six lift used by the classifier is solely a numerical spectral device.
The scaled elastic lower margin must exceed `1e-12`, with eigenpair residual at
most `1e-13`; a more ill-conditioned valid structure can therefore be refused.
The restarted Lanczos workspace is bounded to 256 columns (or the system size
when smaller), with at most 2000 ARPACK iterations. Actual operator-application
counts and the workspace dimension are reported; there is no fixed wall-time
promise, and every operator application observes cancellation.

`SolveCancelled` propagates from existing cancellation safe points, including
Lanczos iterations. Sparse third-party factorizations remain synchronous; a
pending cancellation is observed immediately after control returns.

`result.recover()` uses the unchanged source model and existing elastic recovery.
It refuses failed or stale results. There are no physical support reactions in
this profile. Internal MPC forces are available separately in
`result.constraint_forces`, computed against the effective load. The full
`K u - F_effective` may contain legitimate MPC forces; its reduced `T.T`
projection measures elastic equilibrium. Gauge multipliers and inertial loads
are not reported as support reactions.

## Numerical reporting

The result holds full read-only arrays for displacements, rigid accelerations,
applied loads, inertial loads and effective loads. `to_dict()` produces summary
metadata, including a result case with `analysis_type="inertia_relief"`, physical
accelerations, mass/inertia, reference point, resultants, diagnostics and backend.
The result case also records model revisions and deterministic point-mass and
constraint identity tokens, including direct point-mass dictionary changes.

Force and moment balance are reported separately about the stated reference,
with absolute errors and cancellation-resistant scales from the individual
nodal virtual-work contributions. Dimensionless balance, rigid compatibility,
stiffness-nullspace, acceleration backward error and displacement-gauge gates
are `1e-10`.

For physical-data elastic backward error, translations are scaled with the
radius-of-gyration length and rotations with one. With this diagonal `S`, the
reported error is

```text
||S (Kr q - T.T F_effective)||inf
--------------------------------------------------------------
||S Kr S||inf ||q/S||inf + ||S T.T F||inf + ||S T.T F_inertia||inf
```

Its gate is `1e-10`. The effective-load-only backward error and absolute reduced
equilibrium residual are also reported. When applied and inertial loads cancel
to rounding size (for example uniform free fall), dividing by `F_effective`
alone can give a large relative error despite a roundoff-sized absolute
residual. No load projection or artificial balancing correction is used.

`elastic_load_cancellation_ratio` gives the scaled effective load divided by the
sum of the applied and inertial load scales. A small ratio means a small elastic
signal is being obtained by subtracting large loads. The accepted backward error
is relative to those total input loads; it is not a guarantee of small relative
error in that small elastic signal or in recovered stresses.
No separate warning status is emitted for cancellation alone; callers inspect
the reported ratio when interpreting elastic stresses.

These diagnostics establish numerical solution accuracy for the supplied FE
operators. Mesh-discretization error, constitutive modeling accuracy and the
validity of a quasistatic accelerating-frame approximation require their own
engineering assessment. No centrifugal, gyroscopic, transient, contact,
nonlinear or prestressed equilibrium terms are included in this linear route.

## Repeated solves and scope of evidence

An `AnalysisSession` reuses canonical matrices, reduction, physical rigid-inertia
factorization, mechanism certification and elastic factorization. Acceleration
and inertial loads are recomputed for every load vector. Changing the reporting
reference point reuses the same physical preparation at the centre of mass.
Point-mass dictionary contents participate in mass-cache identity. Other direct
model edits retain the existing revision contract: use model mutation methods
or explicitly advance the corresponding revision after a direct edit.

Element activity retains its separate mass/load contribution policies. If those
policies differ, gravity loads and participating inertia can scale differently;
the acceleration reflects the actual assembled operators. No implicit policy
change makes them equal. The source implementation is covered by focused
analytical, independent dense-reference, lifecycle and refusal tests; it does not
broaden historical element qualification or authorize publication.

See `examples/inertia_relief.py` for a complete free bar example.

## Static batch fast path

Both static batch interfaces accept explicit inertia relief:

```python
from anyfesolver import AnalysisSession, StaticCase, iter_linear_batches, solve_linear_many

with AnalysisSession(model) as session:
    for block in iter_linear_batches(
        model, (StaticCase(str(i), load) for i, load in enumerate(loads)),
        batch_size=32, output_dofs=selected_dofs, session=session,
        inertia_relief=True, reference_point=None,
    ):
        selected_displacements = block.displacements
        for item in block.case_results:
            summary = item["inertia_relief"]
            acceleration = summary["acceleration_at_reference"]
            balance = summary["diagnostics"]

    # Collect all DOFs/columns using the same prepared block engine and session.
    displacements, info = solve_linear_many(model, loads, session=session,
                                             inertia_relief=True)
```

The engine prepares mass, rigid inertia, reduction, mechanism certification and
the augmented nullspace operator once per unchanged session identity. Each block
uses one six-by-many acceleration solve and one elastic matrix-RHS solve; it does
not call the public scalar solver per case. Normal `constraint_mode="auto"` also
selects this nullspace batch route for free-free models in `solve_linear_many`.
An ordinary unbalanced free-free load still fails unless the existing explicit
`allow_unbalanced_free_free=True` option is used. That option cannot be combined
with inertia relief, and inertia relief refuses `constraint_mode="transformation"`.
`reference_point` is valid only when inertia relief is enabled.

Each accepted case has `analysis_type="inertia_relief"` and a JSON-safe
`inertia_relief` summary in both its result and `solver_info`. The summary contains
accelerations at the reference and centre of mass, angular acceleration, mass,
centre of mass, inertia, reference point, force/moment resultants, conditioning,
mass/formulation provenance, balance, mechanism, backward-error and gauge
diagnostics, and units. It omits DOF-sized applied/inertial/effective load and rigid
acceleration vectors; use `solve_inertia_relief` when those full fields are needed.
Selected output alone does not reconstruct full displacements. Explicit recovery
requests reconstruct only the active block and use effective loads for constraint
forces. `reactions` remains empty for inertia relief; internal MPC forces are kept
in `constraint_forces`, separate from gauge multipliers and physical acceleration.
`backend_gauge_error` and `backend_augmented_backward_error` report the raw solve
before displacement gauge cleanup; the existing accepted elastic/gauge gates
describe the final field. Mutating returned metadata cannot change cached plans.

Every column must pass the scalar numerical gates before its block is yielded.
Failure raises `StaticBatchError` without accepted block data, with case identity
where attributable; earlier yielded blocks remain valid. Cancellation propagates
as `SolveCancelled`. Memory preflight includes inertia preparation/load blocks.
It also conservatively reserves Lanczos Krylov/work arrays even when a session
may already hold a certificate. A model-wide preparation failure identifies the
first case as context and reports `scope="model"` for spectral non-success.
Input/callback/consumer model mutations terminate the stream. Caller-owned
sessions remain open; internally owned sessions close on exhaustion, failure or
generator close. Direct point-mass changes between separate calls rebuild the
physical plan; ordinary model changes retain the revision requirements above.

Retain an `AnalysisSession` to reuse preparation and factorizations across calls.
A raw caller-supplied `FactorizationCache` without a retained session preserves
the existing caller-owned-cache behavior: entries are session-specific and cannot
be hit by a newly created session. The caller controls that cache's capacity and
cleanup. Existing session invalidation can clear its cache, so use a dedicated
cache per session when isolation from other analyses is required.
