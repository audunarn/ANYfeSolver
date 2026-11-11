# Nonlinear solver performance layer

The package has an optional performance layer for nonlinear solves. Activation
is lazy on the first `solve_static_nonlinear()` call, so ordinary imports and
linear workflows do not pay the PyPardiso/nonlinear-bootstrap startup cost. The
optimized paths preserve the scalar element formulations, discrete
constitutive update, and convergence criteria.

The ownership and invalidation relationships are summarized in
[Performance architecture](PERFORMANCE_ARCHITECTURE.md).

## Sol Ultra scope in `performance_2`

The Sol Ultra campaign promotes only paths that have an exact reference path,
an observable eligibility decision, and revision-safe retained data. The final
measured evidence and promotion decisions belong in the eight
`reports/performance/sol_ultra_*` artifacts; this guide describes behavior and
does not repeat workstation-specific timing claims.

The promoted implementation scope is:

- solver-owned persistent shell plastic state for qualified von Karman static
  and arc-length lifecycles;
- compiled Hill-48 batches for canonical ANYmaterial hardening curves;
- elastic orthotropic and pre-integrated generalized S4 force, tangent, and
  mass batches;
- blockwise force/tangent rotations and revision-cached reference data for
  rotated corotational shell and beam response;
- conservative nonlinear-impact tangent/factorization reuse, compact contact
  work buffers, elastic direct reduced impact assembly, and gated incremental
  damage-matrix updates;
- an optional caller-owned `AnalysisSession` for repeated linear, modal,
  buckling, transient, and capacity analyses;
- preprojected transient load bases and selected-output transform rows; and
- retained, formulation-grouped recovery plans with a compiled isotropic S4
  path for sufficiently large selections.

Every path retains a qualified fallback:

| Fast path | Explicit fallback or exclusion |
| --- | --- |
| Persistent state arrays | Legacy owned dictionaries for incompatible layouts, immutable initial-field overrides, unbatched elements, non-von-Karman kinematics, disabled acceleration, setup failure, and the nonlinear-impact lifecycle. |
| Compiled Hill-48 | The safeguarded scalar return map for custom curve protocols; guarded numerical-tangent behavior remains available for pathological rows. |
| Orthotropic/generalized S4 batches | Scalar element evaluation for unsupported formulations or sections, triangles, Q8/Q8R, initialized states requiring an exact override, and any batch rejected by eligibility checks. |
| Corotational block transforms | The dense transformation is retained for the consistent-tangent chain-rule oracle; unsupported corotational element scope still fails closed. |
| Impact tangent reuse | `tangent_reuse_iterations=0` is the full-Newton oracle. Enabled reuse refreshes on each new substep and on contact, damage, deletion, plastic-state, convergence, line-search, or reuse-budget changes. |
| Direct reduced impact assembly | Full-coordinate nonlinear assembly followed by projection whenever the transformation, kinematics, plastic/fiber history, damage scope, setup cost, memory estimate, or JIT availability is ineligible. |
| Compact contact work storage | Public `SphereContactRecord` objects are materialized at result/API boundaries; the linear impact path keeps eager public records where its fracture semantics require them. |
| Incremental damage matrices | Exact cached-term K/M rebuild until projected future updates amortize setup, when retained memory exceeds the allowance, after plan invalidation, or for unsupported input. Point-mass terms remain unscaled. |
| Recovery batches | The exact legacy recovery path for selections below 100 elements, unsupported formulations, disabled JIT, or a failed compiled batch. |
| `AnalysisSession` | Omitting `session=` preserves the one-shot assembly/factorization behavior. A closed session or a session for another model raises instead of silently falling back. |

Fallback decisions and counts are part of result/status diagnostics. Timing is
never the only way to infer whether acceleration was active.

## Implemented

### Linear-time model construction and coupling lookup

Topology and MPC revision updates invalidate global sparsity/signature caches
without scanning every existing element or deleting unaffected element-local
reference matrices. Adding a genuinely new node advances the model-wide
geometry revision without clearing elements that cannot reference that node.
An incoming element has only its own mesh/material-dependent caches cleared,
which also makes precomputed element insertion safe. These rules remove the
former quadratic construction behavior from both element-at-a-time and
interleaved node/element construction.

Structured panel coupling now builds the shell-cell lookup once per coupling
generation and reuses it for every beam node. The lookup maps actual grid cells
to element IDs and does not rely on element-number sequencing. This removes a
second repeated full-grid reconstruction from stiffened-panel generation.

The 2026-07-23 diagnostic sweep measured the topology construction portion of
representative cylinder models as follows on the audit workstation:

| Elements | Before | After | Speedup |
| ---: | ---: | ---: | ---: |
| 2,792 | 1.627 s | 0.051 s | 32x |
| 11,038 | 25.877 s | 0.239 s | 108x |

These timings are diagnostic observations, not release gates; benchmark
comparisons must use the same machine, environment, and model.

### Batched S4 geometric stiffness

Linear buckling assembly batches the S4 initial-stress operator in a compiled
kernel. Membrane resultants, bending resultants, stress second moments and
per-Gauss local transformations retain the scalar element conventions.
Reference derivatives, integration weights and frames are cached by topology
and geometry revision; state sampling remains per assembly. Distorted, rotated
and warped S4 qualification compares the assembled matrix with the scalar
element oracle. Triangles, Q8, beams and unsupported element formulations stay
on the general scalar path, and assembly diagnostics expose both counts and
geometry/kernel timings.

### Persistent nonlinear assembly plan

A `NonlinearAssemblyPlan` is built once for each `(model, num_layers)` pair and
is invalidated when topology, geometry, material or MPC revisions change. A
live model retains at most eight layer-count variants; least-recently-used
variants are evicted and the eviction count is observable in performance
status. It retains:

- shell grouping by element type, thickness, stabilization and material;
- stacked reference transforms and shell B matrices;
- stacked transforms and strain rows for eligible elastic Beam3 elements;
- element DOF mappings;
- reusable displacement and material-state work arrays;
- flat local force/tangent storage;
- the unique global CSR tangent pattern;
- local-entry-to-CSR scatter indices.

Each Newton iteration therefore updates only displacements, constitutive state,
internal force values and tangent values.

### Direct CSR value assembly

The global tangent sparsity pattern is created once. Local dense tangent values
are accumulated directly into the unique CSR data index space with a
Numba-compiled scatter loop. Repeated Python lists, COO construction, duplicate
sorting and COO-to-CSR conversion are removed from the Newton loop.

### Persistent nonlinear state transactions

Qualified plastic shell batches use a solve-owned `NonlinearStateStore` rather
than rebuilding nested public dictionaries at every trial evaluation. Each
homogeneous `ShellStateBatch` owns two contiguous buffers containing plastic
strain, accumulated plastic strain, and total layer strain. A trial begins with
a generation-checked token. A full accepted update swaps committed/trial
buffers; a rejected Newton candidate or arc-length retry discards the inactive
buffer without changing committed history. Partial updates copy only fields
that the candidate did not overwrite.

The store is created once at the start of a nonlinear static or arc-length
solver lifecycle and is not placed in a model-global cache. It preserves
unbatched states and unsupported state shapes in an owned dictionary sidecar.
Initial stress/prestrain and provenance fields are immutable sidecars and force
an exact element override where required. Public dictionaries are materialized
only for an explicit save/restart/recovery request or the final result. A
stale, foreign, or already-committed trial token raises rather than exposing a
partially updated state.

Set `FE_SOLVER_DISABLE_PERSISTENT_STATE=1` before the solve to retain dictionary
state throughout the lifecycle. `nonlinear_state_storage` diagnostics report
activation, batch eligibility, `state_point_count`, retained buffer bytes,
commit/discard counts and timings, materialization count, and dictionary
fallback elements/reasons.

The qualified transaction wiring is deliberately limited to nonlinear static
and arc-length. Nonlinear impact still owns committed/trial dictionaries so
its substep checkpoint and cutback behavior remains the established oracle.
Impact state-array transactions are deferred until their rollback, deletion,
restart, and saved-history materialization boundaries are qualified together.

### Compiled Hill-48 constitutive batches

Material-axis Hill-48 shell rows use a flattened compiled return-map kernel
when their hardening law can be represented by a canonical curve pack. The
qualified packs cover perfect plasticity, `None`, linear, piecewise-linear
(including knot and tail behavior), power-law, and DNV-RP-C208 curves. Packed
curve data are immutable and shared by every point in that constitutive call;
committed/trial values are read from the persistent state arrays when that path
is active.

A third-party or otherwise unsupported hardening protocol stays on
`hill48_plane_stress_return_map`, which is the scientific oracle. Guarded
invalid-row behavior retains the established scalar/numerical-tangent
semantics. `hill48_vectorized_diagnostics()` reports compiled and scalar point
counts, row fallbacks, the last curve/path, and reason counts.

### Revision-based sparsity caching

The general matrix-assembly sparsity cache uses mesh topology/MPC revision
counters. A deterministic full-topology SHA signature is still retained for
collision-resistant cache identity, but it is computed once per matrix type and
topology/MPC revision instead of serializing and hashing every element on every
lookup. Assembly timing includes final sparse conversion, checks, and signature
work rather than stopping before post-assembly processing.

### Displacement-control block elimination

The augmented `(n+1) x (n+1)` sparse matrix has been replaced with two right-hand
sides on the ordinary structural tangent:

```text
K a = R
K b = Fref

dlambda = (constraint - c^T a) / (c^T b)
dq      = a + b dlambda
```

The two right-hand sides share one numerical factorization.

### Batch B: elastic shell fast path

When Numba is active, geometrically nonlinear elastic shell batches use a
dedicated in-place kernel that writes directly into the persistent force and
tangent buffers. It removes per-Newton plastic/layer state arrays, constitutive
batch tensors and temporary batch result matrices. The local-to-global shell
transformation is applied in place by 3x3 node-DOF blocks.

When Numba is unavailable, Batch B is not installed; the existing NumPy batch
path remains active to avoid a slow Python-loop fallback.

The general elastic batch records actual layer strains during assembly. The
direct Batch B kernel omits those displacement-derived arrays inside Newton
iterations, then reconstructs the real layer strains once at the final
converged displacement. Returned element states, stress envelopes, and strain
summaries therefore retain the same recovery contract without paying that cost
per iteration. Plastic batches call the same plane-stress return map and its
analytical consistent tangent as scalar assembly. The central-difference
tangent remains an explicit qualification oracle and guarded invalid-row
fallback. A condition number alone does not discard a finite exact analytical
row, and the oracle uses representable, stiffness-scaled perturbations.
Shells carrying initial stress or prestrain fields no longer disable batching
for the whole model. For elastic S4 batches, immutable stress and prestrain are
integrated into exact membrane and bending resultant offsets inside the JIT
kernel; complete layer/provenance state is reconstructed once at convergence.
Initialized plastic shells retain exact scalar element evaluation, while
ordinary shells in the same model remain accelerated. Diagnostics separate
accelerated initialized elements from scalar overrides. A 96-S4 case with
fields on every element measured 2.03 ms accelerated versus 21.82 ms scalar
after warm-up (10.7x). Reduced-integration Q8R is deliberately ineligible for
shell batching because its experimental hourglass stiffness is not represented
by the accelerated local kernel.

The promoted S4 scope also includes homogeneous orthotropic materials and
pre-integrated generalized `A/B/D/As` sections. Their reference geometry,
material-frame operators, section matrices, areal inertia, and scatter maps are
packed once per revision-valid batch. The generalized kernel does not assume a
symmetric membrane/bending coupling block: it preserves the scalar `B` and
`B.T` placement, transverse shear terms, state resultants, and consistent mass.
Diagnostics separate `orthotropic_s4` and `generalized_s4` batch/element counts.
Triangles, Q8/Q8R, unsupported section laws, and state combinations outside the
qualified kernel remain on exact scalar element calls.

### Elastic Beam3 batch

Straight elastic `QuadraticBeamElement` members use a dedicated compiled batch
for von Karman internal force and consistent tangent assembly. Reference
transforms, Gauss-point strain rows and scatter positions are retained in the
nonlinear plan. Generalized sections, fiber plasticity, deleted elements and
corotational analyses retain their qualified scalar behavior. On the audit
workstation, 72 rotated Beam3 elements measured 0.164 ms batched versus
17.43 ms scalar after warm-up (106x); force and tangent qualification uses the
scalar element implementation as the oracle.

### Corotational block transforms

Corotational shell and beam response retains reference topology, coordinates,
DOF maps, and element/category metadata on the mesh's topology/geometry
revision. The ordinary rotated-tangent path applies force and tangent rotations
as direct 3x3 node blocks, avoiding construction and multiplication of a large
dense block-diagonal transform. The dense transform is deliberately retained
only inside the consistent corotational tangent chain rule, where frame
sensitivity terms require it and provide the qualification oracle. Material or
state updates do not invalidate geometric reference data; topology or geometry
updates do. Unsupported element categories continue to fail closed through
`validate_corotational_scope()`.

### Batch C: direct reduced-coordinate assembly

When constraints reduce the system, element-local values are now scattered
directly into the independent coordinates:

```text
F_int,r = T^T F_int
K_r     = T^T K T
```

The reduced CSR pattern, local-entry scatter maps, force buffer, tangent data
buffer and sparse matrix object are retained for the complete nonlinear solve.
The repeated sparse products `T.T @ K @ T` and `T.T @ F_int` are therefore
removed from force control, displacement control and arc-length correction.

Two setup paths are used:

- `selector` for ordinary fixed supports and unit-coefficient slave mappings;
- `weighted_mpc` for eccentric beam-shell coupling and general MPC coefficients.

The weighted map is expanded once. Set `FE_SOLVER_BATCH_C_MAX_MAP_MB` before
Python starts to change its retained-map memory limit, which defaults to 512 MiB.
If the estimated map exceeds the limit, the solver automatically retains the
full-coordinate assembly and projection path.

Direct reduction also has a solve-cost gate. The representative weighted-MPC
model broke even after about 72 tangent evaluations, so the default threshold
is 144 estimated useful assemblies (a 2x safety margin). The estimate uses up
to four useful Newton evaluations per requested increment instead of the full
failure limit. Short solves keep full-coordinate persistent assembly and avoid
the reduced-plan setup. Set
`FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES` to override the threshold; status
diagnostics record the estimate, threshold, decision and skip count.

Batch C is installed only when Numba is active. Python/NumPy fallback runs keep
the existing sparse projection path.

#### Nonlinear impact extension

The nonlinear rigid-sphere impact loop can use the same retained reduced
scatter plan for its internal force and tangent. Activation is deliberately
narrower than static Batch C: it requires Numba, von Karman kinematics, a
non-identity transformation, an exactly zero affine offset, no plastic-impact
damage/softening/deletion configuration, no material hardening/Hill history or
beam-fiber plasticity marker, a plan below the retained-map memory limit, and
an estimated assembly count above the Batch C cost gate. Every exclusion keeps
the full-coordinate impact oracle and is reported through
`impact_reduced_assembly.fallback_reason` and `exclusion_reasons`. Plastic
impact remains qualified with tangent reuse on the full-coordinate path.

For eligible HHT-alpha runs, the accepted internal-force history is retained
in reduced coordinates. Because eligibility requires `u0 == 0`, both the HHT
residual and the nonlinear internal-work measure remain algebraically exact:
`u.T @ F_int == q.T @ (T.T @ F_int)`. Public displacement, contact and element
state histories are still materialized through the existing impact result
path. Diagnostics expose direct-reduced and full-coordinate assembly counts,
the selector/weighted-MPC mapping kind, setup memory and timing.

Eligibility and retained-plan ownership live in
`impact_reduced_assembly.py`; the transient solver consumes only its controller
interface. New contact, damage, affine-constraint, or ANYfem-facing state
semantics must remain on the full-coordinate fallback until their reduced
force/tangent and committed-state behavior is explicitly qualified.

### Nonlinear-impact tangent reuse

`NonlinearTransientConfig.tangent_reuse_iterations` enables a bounded
modified-Newton policy. Zero preserves the legacy full-Newton behavior. A
positive budget may reuse the current effective-tangent factorization only
inside one time substep. It refreshes on the first iteration, a changed time
step, damage scale or deletion set, active contact element/classification,
meaningful plastic-state change, residual stall, aggressive line search, or an
exhausted reuse budget. The controller owns policy and counters, not finite
element matrices or material state; the factorization handle is solve-local.

Impact diagnostics expose `tangent_assembly_count`, `tangent_reuse_count`,
`factorization_count`, `factorization_reuse_count`, active-contact changes, and
`refresh_reason_counts`. These counters are also emitted when the reuse budget
is zero, making the oracle path observable.

### Contact work buffers and damage-matrix updates

The nonlinear sphere-contact loop retains a solve-local structure-of-arrays
`ContactWorkBuffer`. Candidate geometry, contact forces, compact nodal scatter
entries, the full load vector, and the sphere resultant are overwritten in
place. Active-contact reduction stays deterministic and preserves sticky
contact preference. Public `SphereContactRecord` instances and nodal-force
dictionaries are created only for saved/result boundaries. The standalone
public contact assembly helper uses thread-local buffers so concurrent calls do
not share mutable work arrays.

When impact damage changes element stiffness or mass scales, a
`DamageMatrixPlan` retains the CSR patterns and each element's value positions.
Only elements whose scale changed are added into the existing K/M data; omitted
previous scales return to one. The plan preserves point masses independently
of element scaling and resets exact all-one/all-zero edge states to avoid
accumulated subtraction roundoff. It is created lazily inside one impact solve
and is invalidated by topology, geometry, material, or mass revision. An
invalid plan or non-finite scale invokes the exact scalar matrix rebuild.

Plan construction also has production cost and memory gates. At least two
events must be observed, and their measured event density must project at
least 11 *future* matrix updates in the remaining nominal steps. The retained
array estimate must fit both a 256 MiB default cap and any smaller
`ResourceConfig.memory_limit_bytes` headroom remaining after the solve
preflight estimate. This limit is applied to the peak combined retained
footprint: the cached legacy element-term arrays remain alive for exact
fallback while the plan owns its CSR arrays, contribution maps, and object
overhead. Until both gates pass, cached element terms drive the exact legacy
rebuild. A model revision refreshes those terms and disables the plan for the
rest of that solve.

`contact_work_buffer` diagnostics report assembly/scatter/materialization and
growth counts. `damage_matrix_plan_selection` reports the event projection,
break-even and memory decision, cached-term, plan-only, combined estimated,
combined actual, and conservatively accounted retained bytes, legacy/plan
update counts, and selection reason. `damage_matrix_plan` reports setup/update
time, retained bytes, changed/no-change counts, active scaled elements,
invalidation and fallback counts, and K/M nonzero counts when the plan is
selected.

### Batch D: multicore and sparse backend tuning

When Numba is active, the Batch B elastic shell kernel now uses `prange` over
the element dimension.  `solve_static_nonlinear(...,
resource_config=ResourceConfig(assembly_threads=N))` temporarily sets the Numba
thread count for nonlinear assembly and restores the previous count after the
solve.

`ResourceConfig.solver_threads` is also enforced for every public solver entry
point through a scoped `threadpoolctl` limit. Factorization and reusable-handle
solve diagnostics retain requested and observed pool data. When
`assembly_threads > 1` activates parallel Numba assembly, native BLAS/OpenMP
pools are nested at one thread and both Numba and native limits are restored on
normal return or exception. A missing limiter is reported explicitly; an
omitted resource policy leaves backend defaults unchanged.

Because native pool limits are process-global, explicit limit scopes are
serialized across Python threads. Unlimited/default solver scopes may overlap;
an explicit phase waits until other unlimited solves have left their scope so
its limit cannot leak into them. Same-thread nesting and default-to-explicit
upgrades remain reentrant, and ownership checks prevent a copied `ContextVar`
from inheriting another thread's active limit.

`ResourceConfig.recovery_threads` controls coarse formulation-homogeneous
recovery work. A compiled-only isotropic S4 selection scopes Numba to the
selected worker count. Mixed/scalar chunks use a thread pool and constrain
nested native pools to one thread. `solver_threads`, `assembly_threads`, and
`recovery_threads` are independent explicit choices: the solver does not derive
a default from logical CPU count, and all temporary Numba/native limits are
restored after success or failure. The final campaign report, rather than this
guide, records the selected measured thread policy.

The separate `experimental_csr_assembly` module is benchmark-only. It caches a
CSR topology and local-entry scatter map, but the qualified linear K/M/KG path
continues to use COO. Promotion requires at least 20% complete assembly gain,
plus at least 5% end-to-end gain or 15% peak-memory reduction, matrix error no
greater than `1e-12`, and no representative case regression above 5%.

The sparse backend policy is tuneable through:

```text
FE_SOLVER_PYPARDISO_MIN_DIMENSION
FE_SOLVER_PYPARDISO_MIN_NNZ
FE_SOLVER_PYPARDISO_WARM_MIN_DIMENSION
FE_SOLVER_PYPARDISO_WARM_MIN_NNZ
FE_SOLVER_PYPARDISO_MAX_PATTERN_SLOTS
```

The cold defaults (`10,000` equations and `250,000` nonzeros) avoid paying MKL
startup cost on small one-off solves. Lower warm defaults (`1,000` and
`25,000`) apply only when a retained PARDISO slot has the same prepared
sparsity pattern and a matrix type compatible with the incoming matrix class.
Merely having initialized PyPardiso for an unrelated matrix does not activate
the warm policy. Both thresholds must be met.

The auto backend records the active thresholds, prior initialization state,
compatible-pattern decision, retained-slot count, and selected backend in
solver diagnostics. The PyPardiso module and solver object are constructed only
after the policy selects that backend. The PARDISO backend checks standard MKL
library directories without a recursive environment scan, resolves `mkl_rt`
once per process (`PYPARDISO_MKL_RT`), factorizes symmetric matrix classes as
upper triangles with symmetric mtypes (2 / -2, general fallback on failure),
reuses symbolic analysis (phase 22) when the sparsity pattern is unchanged
through a small LRU of pattern slots, and releases MKL internal memory when
slots are evicted or handles are garbage collected. Handle diagnostics report
`pardiso_mtype` and `pardiso_symbolic_reused`.

Linear static reduced stiffness matrices are declared symmetric indefinite
rather than general, allowing a selected symmetric backend to use its native
matrix class while retaining SuperLU fallback.

### Optional analysis sessions and reduced transient data

`AnalysisSession` is an explicit, caller-owned cache for repeated analyses of
one live `FEModel`. Passing `session=` is optional on `solve_linear()`,
`solve_linear_many()`, `solve_free_vibration()`,
`solve_eigenvalue_buckling()`, `solve_transient_newmark()`, and the capacity
workflow. Omitting it preserves one-shot behavior.

```python
from anyfesolver import AnalysisSession, solve_linear, solve_free_vibration

with AnalysisSession(model) as session:
    displacement, static_info = solve_linear(
        model, load_case, session=session
    )
    modal_result = solve_free_vibration(
        model, num_modes=10, session=session
    )
```

The session weakly references, but does not own, its model. It retains bounded
K/M/T/Kred/Mred, rigid-mode, selected-output-row, and factorization plans.
Structural matrix keys use model revisions; constraint keys additionally use
canonical equation structure/value fingerprints so direct edits cannot reuse a
stale affine transform. Load values are deliberately absent from structural
keys. A prescribed-value-only change refreshes `u0` while retaining the
unchanged reduced stiffness/factorization. Structural constraint changes or
matrix revisions invalidate dependent plans. Output plans and factorization
handles have bounded entry counts. `close()`/`release()` frees retained
matrices and handles; context-manager exit closes automatically.

The linear transient path projects its base load and every pressure-patch basis
through `T.T` once. Selected history mode retains only requested rows of `T`
and `u0`; it reconstructs full displacement/velocity/acceleration vectors only
when full/envelope histories or stress recovery require them. A session can
also supply revision-valid K, M, T, Kred, and Mred. Diagnostics include
`preprojected_load_basis_count`, full/selected reconstruction counts, history
storage mode, factorization counts, and nested `analysis_session` counters.

### Batched stress recovery

Recovery preserves the exact scalar element routine as its oracle. A selection
smaller than 100 elements takes the legacy path immediately, avoiding retained
plan, scheduler, and native-thread setup. Larger selections build a mesh-owned
`RecoveryBatchPlan`, group work by formulation, and retain DOF maps and the
reference data for eligible isotropic S4/S4R elements. The compiled S4 kernel
preserves warped/rotated geometry, local or global output, and top/bottom stress
ordering. Unsupported formulations continue in deterministic coarse scalar
chunks, so a mixed request may use a hybrid compiled/scalar backend.

The plan is valid only for matching topology, geometry, and material revisions
and can be cleared with `clear_recovery_batch_plan(model)`. The execution report
records `recovery_backend`, formulation/batch counts, eligible/fallback element
counts, fallback reasons, setup reuse/time, retained bytes, compiled time, and
native/Numba thread policy.

## Ownership and cache lifetime

| Retained object | Owner and lifetime | Invalidation/release |
| --- | --- | --- |
| Numba machine code | Process-local and lazily initialized. | Process exit; unavailable JIT keeps the oracle path. |
| Nonlinear assembly/reduced scatter plans | Weakly associated with a model; reusable across qualified solves, with at most eight LRU layer-count variants per live model. | Topology, geometry, material, or MPC revision; LRU eviction; explicit `clear_nonlinear_assembly_cache()`. |
| Nonlinear committed/trial state | One static/arc-length solver lifecycle. | Accepted candidate swaps/commits; rejected candidate discards; final public materialization ends the optimized lifecycle. |
| Corotational reference and contact geometry | Mesh-owned immutable reference data. | Relevant topology/geometry revision. |
| Recovery batch plan | Mesh-owned and reused by large recovery calls. | Topology/geometry/material revision; explicit clear. |
| `AnalysisSession` plans and factorizations | Caller-owned, one model, bounded entries. | Model/constraint fingerprints, LRU eviction, `close()`/`release()`. |
| Impact tangent handle, contact work, reduced assembly, and damage matrices | One impact solve (the public helper's contact buffer is thread-local). | New substep/policy refresh, model revision, fallback, or solve completion. |

No mutable constitutive state is stored in a model-global performance cache.
Revision caching therefore cannot commit a rejected Newton/arc candidate or
leak an impact state into a later solve.

## Performance diagnostics

The stable result/status diagnostics include the activation evidence needed by
the campaign. Depending on analysis type, inspect:

- `plan_setup_seconds`, `plan_reused`, eligible/fallback element counts and
  `fallback_reasons`;
- `nonlinear_state_storage` for state points, buffer bytes, transaction counts,
  materializations, and dictionary fallbacks;
- result-local `nonlinear_performance` for assembly/direct-reduction paths,
  nested solve counts, plan reuse, Hill compiled/scalar points, and
  corotational block/dense fallback activation;
- `hill48_vectorized_diagnostics()` for the additional process-wide compiled/
  scalar point and row-reason view;
- nonlinear-plan formulation counts, including orthotropic/generalized S4;
- `impact_reduced_assembly`, `tangent_reuse`, `contact_work_buffer`, and
  `damage_matrix_plan`;
- `factorization_count`, reuse counts, selected sparse backend, and backend
  handle diagnostics;
- transient reconstruction/load-basis counters and `analysis_session`; and
- recovery backend/batch/fallback and thread-policy metadata.

Counters describe work performed, not just configured intent. Final numerical,
cold/warm, memory, thread-scaling, and regression evidence is recorded in the
versioned Sol Ultra reports.

## Activation and A/B testing

Importing `anyfesolver` does not install the nonlinear performance layer.
`solve_static_nonlinear()` performs a one-time lazy activation immediately
before a nonlinear solve. Existing public solver calls are unchanged, and an
optional acceleration import/initialization failure leaves the scalar path
available.

Set the environment variable below before the first nonlinear solve to retain
the scalar/legacy assembly path:

```text
FE_SOLVER_DISABLE_FAST_NL=1
```

Runtime diagnostics are available from:

```python
from anyfesolver.jit_compiler import jit_diagnostics
from anyfesolver.nonlinear_performance_bootstrap import nonlinear_performance_status

print(jit_diagnostics())
print(nonlinear_performance_status())
```

Each nonlinear result also reports analysis-local work in
`result.info["nonlinear_performance"]`.  The `solver` section counts Newton
factorizations/solves, rejected full steps, backtracks, residual-to-tangent
promotions, recoverable trial failures, dead-load projection reuse, and
accepted-force reaction reuse or fallback reassembly.  These counts include
failed work and are safe to interpret without subtracting process-wide
counters.

## Optional Armijo globalization

Force-controlled static beam and shell analyses can select a scaled residual
merit line search explicitly:

```python
result = solve_static_nonlinear(
    model,
    load,
    convergence_settings={
        "profile": "legacy",
        "line_search": "armijo",
        # Optional. The reference-node RMS radius is used when omitted.
        "characteristic_length": 2.5,
    },
)
```

Translational residual rows retain force units. Rotational residual rows are
divided by the frozen characteristic length before the merit norm is formed.
If the independent constraint basis contains no rotational row, a degenerate
reference-node radius is harmless and the solver records a unit placeholder;
models with independent rotations still require an explicit positive length
when that radius is degenerate.
Physical convergence still uses the existing unscaled equilibrium residual and
tolerance. Armijo applies sufficient decrease with coefficient `1e-4`, starts
at the full Newton step, and uses the configured `line_search_reduction` and
`max_line_search_cuts` for bounded backtracking.

This mode is opt-in. It currently excludes displacement control, fracture and
private GE beam programs. Corotational use requires the consistent tangent;
`corotational_tangent="auto"` selects it automatically. Arc-length continuation
keeps its existing globalization and controls.

The force-control solver reuses the converged internal-force payload for
support reactions when its generation remains valid. It falls back to exact
full-coordinate reassembly when the payload is stale or unavailable. Fixed
dead-load vectors are projected through the affine constraint transformation
once per analysis; follower, deletion-dependent and native
configuration-dependent loads retain current-state evaluation.

Before the first nonlinear solve, the status reports that activation has not
yet been attempted. Calling the status helper directly does not activate the
layer.

The plan can be cleared explicitly after non-standard in-place model changes:

```python
from anyfesolver.nonlinear_performance_bootstrap import clear_nonlinear_assembly_cache

clear_nonlinear_assembly_cache(model)
```

Ordinary model methods update revision counters and invalidate the plan
automatically.

## Validation requirements

After changing the nonlinear formulation, constraint mapping, JIT kernels, or
sparse backend, run:

```powershell
python -m pytest tests -q
python scripts/benchmark_nonlinear_assembly.py --nx 20 --ny 10 --repeats 10
python scripts/benchmark_nonlinear_assembly.py --nx 20 --ny 10 --repeats 10 --weighted-mpc-rows 12
python scripts/benchmark_sol_ultra_performance.py --suite full --repeats 3
```

Release qualification additionally captures the independent numerical suite
at the immutable baseline and final candidate SHAs, then runs
`verify_sol_ultra_numerics.py compare` to produce the versioned numerical JSON
and Markdown report. Baseline and candidate captures must not come from the
same checkout state.

The dedicated tests compare optimized and legacy internal-force/tangent assembly,
including tilted shells, selector constraints and weighted MPC transformations.
They also verify cache invalidation, residual-only assembly, CSR uniqueness,
persistent buffer reuse, analytical/numerical plastic-tangent parity, elastic
layer state, mixed initial-field exact overrides, Beam3 scalar-oracle parity,
Q8R scalar fallback, persistent state transaction rollback, canonical and
custom Hill-48 paths, analysis-session invalidation, impact reuse/reduction,
contact materialization, damage updates, recovery gating, lazy nonlinear-solver
activation, and legacy restoration. Qualification records
warmed constitutive and representative global Newton timings for the
analytical and numerical tangent paths; timing is evidence rather than a
wall-clock pass gate.

## Deferred until measured

The following changes remain deferred until profiling identifies the
next dominant phase:

- persistent constitutive-state transactions in nonlinear impact;
- batched corotational frame extraction/local element response beyond the
  promoted direct 3x3 force/tangent block rotations;
- upper-triangle-only element tangent integration;
- retained arc-length predictor factorization;
- automatic Numba thread selection (high logical-core counts can oversubscribe
  memory bandwidth, so `assembly_threads` remains an explicit measured choice);
- Q8R batch acceleration, pending a qualified stabilization formulation.

Symbolic sparse-factorization reuse (PARDISO phase 22 with pattern slots) is
implemented in the PyPardiso backend. Historical Batch B/C/D notes were folded
into this guide; this file is the maintained performance documentation.
