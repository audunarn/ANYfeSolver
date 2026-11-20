# Coupled shell–solid route (private, dense certificate)

`anyfesolver.solid_coupling.solve_shell_solid_coupled` solves a Q4 shell, with a rectangular set of elements
removed, together with a floating P2 solid detail (ANYsolid `PreparedSolid`) that fills the hole. The two are tied
weakly along the four cut sides. The result is either a `CoupledSolution` with certified bounds, or a
`CoupledSolveError` with a `COUPLED_*` code. Every `Exception` raised inside the route becomes a
`CoupledSolveError`. `KeyboardInterrupt`, `SystemExit` and the other `BaseException` subclasses that are not
`Exception` are not translated: they propagate unchanged, also when a callback raises them (OD9).

The module is private. `import anyfesolver` neither imports nor exports it. ANYfem is its only intended caller.
The certificate lives in `anyfesolver._coupled_certificate`. The spec is
`SOLVER-PR67-DENSE-SPEC-20261119.md`, which applies owner decisions OD1–OD13. `threadpoolctl` is a core dependency of
ANYfeSolver and is also listed in the `solids` extra (OD11).

## Trusted-construction boundary (OD9)

The route certifies only untouched objects built by ANYfem's pipeline:
- the `PreparedSolid` from `anysolid.prepare`;
- the `FEModel` from ANYfem's builder;
- the cut (the removed elements and the box faces) from `cut_solid_submodel`.

Objects modified after they are built are out of scope: for example a relabelled `order`, a DOF list swapped behind
the DOF manager, a replaced connector or native output, or a monkeypatched library function. The claims below are
not made for them, even where a check happens to refuse them. The route still keeps cheap structural checks that
also catch some such changes: the removed set is derived from the footprint geometry, the solid coefficients must
have the P2 layout, every nonzero test reads value bits, raw load scalars are type-checked before assembly, and
CSR structure, connector dimensions and native `K`/`F` are checked before use.

```python
from anyfesolver import solid_coupling as sc
result = sc.solve_shell_solid_coupled(
    fe_model, prepared, removed_element_ids=removed, load_case=case,
    box=sc.BoxFaces(divisions=(nx, ny, nz), skin='end', sides=('side_0', 'side_1', 'side_2', 'side_3')),
    cancel=None, resources=None)
```

## Formulation

The shell is the FEModel minus `removed_element_ids` and minus any orphan nodes. Unchanged native code gives:
- `assemble_system` → `(K, F)`;
- `build_constraint_transformation` → `K_s = TᵀKT`, `f_s = Tᵀ(F − K u0)`, `T` and `u0`.

The solid uses `K_v = prepared.assemble_linear()`, with no supports. Its load `f_v` is the skin traction
`prepared.traction(skin, p·n)`.

Each side gets a weak connector, built with `ShellEdgeTrace` and `anysolid.coupling.build_weak_connector`. This gives
`C_shell` and `C_solid`. From them, `B_s = C_shell T` and `g = −C_shell u0`, both formed exactly. The stored system is

```
S z = b:  [[K_s, 0, B_sᵀ], [0, K_v, −B_vᵀ], [B_s, −B_v, 0]] [x_s; x_v; λ] = [f_s; f_v; g],   N = n_s + n_v + m ≤ 5000
```

No row is normalised. Equilibration uses exact powers of two only, so the certified matrix is exactly `S`.

## Contract statements

- **C1, certificate scope.** The certificate covers the stored float64 system `S z = b`. It does not cover
  formation error: native assembly, `T`, `u0` and the connector matrices. Formation fails closed in three ways.
  - Inputs are checked for exact types, float64, canonical CSR and finiteness.
  - Every nonzero of `K`, `F`, `K_s`, `f_s`, `T`, `u0`, `C_shell`, `C_solid`, `K_v` and `f_v` must lie in
    `W_raw = [2^-400, 2^64]`. Otherwise the solve refuses `COUPLED_FORMATION_RANGE`.
  - `B_s` and `g` are exact `Fraction` sums. An exactly nonzero entry whose float falls outside `W_raw`, zero
    included, refuses `COUPLED_FORMATION_LOSS`.
- **C2, C3, arithmetic.** There is no enclosure kernel. Every certified number is one of two things:
  - an exact rational, from Python integers or `Fraction`;
  - a float rounded outward from one, with `_out_up`.
  The exact steps do not depend on the floating-point environment. The SVD and SuperLU calls are the only
  floating-point library calls. The SVD is covered by A1/A2 below, which assume round-to-nearest with gradual
  underflow on the thread that runs it; the route verifies that environment and the single BLAS thread instead
  of assuming them (see "Floating-point environment"). SuperLU supplies a candidate only.
- **C4, pressure provenance and the ledger.**
  - The skin and the four sides must be faces of an affine structured box lattice. This is checked in exact
    arithmetic: the plane to `1e-12·d`, the lattice positions to `1e-9·h_min`, and positively oriented split cells.
  - Every load record goes through the native `LoadCase._load_contributions` traversal. Any component that the
    traversal skips refuses `COUPLED_LOAD_UNCOVERED`.
  - The transfer gates compare the removed native load with the skin traction. The force must match to
    `1e-12‖F_ref‖` and the moment about the footprint centre to `1e-12(‖F_ref‖d + ‖M_ref‖)`. Both must be exactly
    zero when the reference is.
- **C5, claims.** Accuracy is claimed only for the outputs listed under "Claims" below. Reactions, strains,
  stresses and energies carry no claim.

## Certificate (the port's `_certify_solve_matrix`, cap 5000)

The certificate runs these steps:
1. Refuse `COUPLED_TOO_LARGE` when `N > 5000`, or when the estimate `4·N²·8 B` exceeds the memory limit. Both
   checks run before any dense allocation.
2. Ruiz-equilibrate by powers of two: `A = ldexp(S, t_i + t_j)`.
3. Compute `s_1 ≥ … ≥ s_N = svd(A, compute_uv=False)`.
4. With `α = cNε`, `c = max(1000, N)`, `β = α/(1−α)` and `δ = β s_1 + N·2^-1074`, accept only when
   `κ_up = (s_1+δ)/(s_N−δ) ≤ 1e-6/ε`, rounded outward.

The bound rests on two assumptions:
- **(A1)** The computed `s_i` are the exact singular values of `A + E`, with `‖E‖₂ ≤ α‖A‖₂`.
- **(A2)** Each entry of `A` is `2^(t_i+t_j)S_ij` rounded once, and that rounding is inexact only for a subnormal
  result. Then `A* = DSD` satisfies `‖A − A*‖₂ < N·2^-1074`.

Weyl's inequality gives `σ_1(A*) ≤ s_1 + δ` and `σ_N(A*) ≥ s_N − δ`. The full derivation, the reason for
`c = max(1000, N)`, and the Codex round-4 and round-5 counterexamples are in the port's `docs/SOLID_ADAPTER.md`
(b94c9179f, lines 119–182). The counterexamples are kept verbatim in `tests/test_coupled_certificate.py`.

### Inverse-norm bound (spec §2.2)
Whenever `s_N − δ > 0`, `‖A*⁻¹‖₂ ≤ 1/σ⁻` with `σ⁻ = s_N − (β s_1 + N·2^-1074)`. This is evaluated in exact
`Fraction` and adds no premise.

### Exact residual and forward error (spec §2.3, OD6)
`r = b − Sx` is computed exactly. Each product `S_ij x_j` and `b_i` becomes an integer mantissa and an exponent.
These are shifted to the row's minimum exponent and summed as Python integers. The result does not depend on
summation order or the FP environment.

- **V6, the authority** (with the κ gate, OD1). Let `E = sqrt_hi(Σ(d_i r_i)²)/σ⁻`, with `d_i = 2^{t_i}`. The solve
  is accepted only when `E ≤ 1e-6·max_i|x_i/d_i|`, decided exactly. Otherwise it refuses `COUPLED_UNCERTIFIED`
  (`forward_error_above_limit`). The claim is `|x_i − x*_i| ≤ d_i·E`, where `x*` is the exact solution of the stored
  system. It rests on A1 and A2 only.
- **V1–V3, backward error per row** (OD7). Accept only when `|r_i| ≤ 1e-10·ref_i` for every row, with
  `ref_i = Σ_j|S_ij|·‖x‖∞ + |b_i|`, exact. A row is exempt only when `ref_i = 0`. Then `r_i = 0`, which the code
  asserts.

### Zero-load path (spec §2.4)
The path is taken when native `F = 0`, `u0 = 0` and every consumed raw input is exactly zero (review r1 #4). The
zero test reads the original values of the case and the model, never a float image of them (review r2 (d)), and
counts only what native assembly and reduction consume (review r2 (c)):
- every pressure, kept and transferred;
- the first `min(L, 6)` components of every nodal record of length `L` on a kept node;
- the prescribed values of every support that names at least one node (a support with no nodes prescribes
  nothing);
- every MPC right-hand side.

Raw scalars are type-checked before native assembly: a nodal record is a 1-D float64 ndarray, or a list or tuple of
exact `float`, `np.float64` or `int`; support values and MPC terms are exact `float`, `np.float64` or `int`. Anything
else (a `Fraction`, a `bool`, a string, a float32 array) refuses `COUPLED_LOAD_UNSUPPORTED` (`raw_scalar_type`).
The path is otherwise unchanged. Then `x = 0`, and the full certificate must still accept. A nonzero raw input whose stored `b` is all
zero refuses `COUPLED_FORMATION_LOSS` (`load_stored_as_zero`), for example a kept pressure of `2^-1074` whose native
contributions round to zero.

### Floating-point environment (review r1 #1, r2 (a))
Every exact step reads and writes binary64 values by IEEE bit extraction (`_decompose`, `_exact`, `_to_float`,
`_out_up`, the nonzero masks and the raw window), so its result does not depend on FTZ/DAZ or the rounding mode. The
remaining float steps (equilibration, the SVD, outward rounding, comparisons, native assembly) assume IEEE
round-to-nearest-even with gradual underflow, as A1 and A2 do.
- **Probe.** `_require_fp_environment` checks, for Python floats and NumPy vector loops, that subnormal operands
  and results survive (no FTZ/DAZ) and that the rounding is to nearest-even (ties `1 + 2^-53 → 1` and
  `(1 + 2^-52) + 2^-53 → 1 + 2^-51`, and their negations, which exclude rounding up, down and toward zero). It
  refuses `COUPLED_UNCERTIFIED` with `subnormal_flush` or `rounding_mode`. The probe itself never raises, so an
  exception from inside it was raised by caller code running there (a trace, profile or audit hook), including a
  forged `CoupledSolveError`; it refuses `COUPLED_UNCERTIFIED` (`fp_probe`, review r5 #3).
- **Where.** At entry, after every cancellation poll, at the start of the certificate (before the power-of-two
  equilibration), after the outward-rounded κ bound, and immediately before and after the exact residual. The SVD
  worker runs the same probe in its own process before and after the SVD (see below).
- **After every caller callback (review r3).** Caller code runs only in these places, and each is followed by the
  probe before any float step of the route runs:
  - every `clock()` call, in the route and inside the certified solve (the checked clock; a clock that selects
    another rounding mode before a stage and restores it at the next timing call refuses at the first call);
  - the cancellation check: the `is_set` lookup, then each flag read (every poll);
  - the `resources.memory_limit_bytes` lookup and its single integer conversion;
  - the iteration and integer conversion of `removed_element_ids` (materialised once) and of the box divisions.
  Before these callbacks the route tests the caller's arguments by identity only (`clock is None`, `cancel is
  None`; review r4): no `__bool__`, `__len__`, `__eq__` or `__index__` of a caller object runs outside the places
  listed. The clock value must be an exact `float` or `int` (`COUPLED_NATIVE_INVALID`, `clock_invalid`), so the route's
  own timing arithmetic runs no caller code. A callback that changes the environment and restores it before it
  returns leaves no trace and affects no route step.
- **Containers (review r5 #4).** Right after the exact-type checks of the model and the load case, every container
  the route reads is checked by exact built-in type before any protocol method of a caller object can run
  (`__len__`, `__bool__`, `__iter__`, `items`, `__contains__`, `__hash__`, `__eq__`, `__deepcopy__`): the
  LoadCase `nodal_loads`, `element_loads`, `pressure_loads` and `added_node_masses` (exact `dict`, integer keys
  of type `int` or a NumPy integer, mass values `float`, `np.float64` or `int`) and `follower_pressure` (exact
  `bool`); the FEModel `mesh` (`FEMesh`), its `nodes` and `elements` (the mesh's own mappings, integer keys),
  every node (`Node`, `dofs` an exact `list` of integers), every element's `node_ids` (`tuple` or `list` of
  integers), `point_masses` and `concentrated_masses` (`dict`), `materials` (`dict`, `str` keys),
  `boundary_conditions` (`list` of the anyfesolver support classes, each with `node_ids` a `list` or `tuple` of
  integers and `dof_constraints` a `dict` with `str` keys) and `constraint_equations` (`list` of
  `ConstraintEquation`, integer `dependent_dof`, `coefficients` a `tuple` of `(dof, value)` pairs). Anything
  else refuses `COUPLED_NATIVE_INVALID` (`container_type`). Value types are checked as before (`raw_scalar_type`).
- **Process-wide state.** The certificate premise A1 depends on the SVD's environment only, and that runs in its
  own process (below). The FP environment of the solving thread is probed as above. NumPy's `errstate` is per
  context. Other BLAS or LAPACK calls in the solving process (`matrix_rank` in the rigid-rank check, a small Gram
  solve, SuperLU) carry no premise: a different thread count or a hook changes their bits only, and their
  outcomes are refusals or a candidate that the exact checks judge.

### Certificate SVD process (owner decision OD13, 2026-11-20)
Review r5 showed that an in-process single-thread window cannot be closed against same-process caller code: a
profile hook or another thread can change the process-wide BLAS pool limits inside it (directly through
threadpoolctl, bypassing any coordinator) and restore them before a post-check. The certificate SVD therefore
runs in a short-lived child process; the solver's thread coordinator (`threading_policy`) is not used by the
route and is unchanged from the base.

- **Start.** `subprocess.Popen` (a new interpreter, never a fork) of `sys.executable -I -S
  anyfesolver/_coupled_svd_worker.py <NumPy package dir> <path>...`. `-I` ignores every `PYTHON*` variable and the
  user site and keeps the script and working directories off `sys.path`; `-S` skips `site`, so no `.pth` file,
  `sitecustomize` or `usercustomize` runs. `sys.path` is the standard library plus the directory that holds NumPy
  (and threadpoolctl's, if it is loaded from elsewhere). The environment is exactly `OPENBLAS_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and, on Windows, `SYSTEMROOT` (the interpreter needs it to start). The
  working directory is the anyfesolver package directory. The worker module imports nothing from anyfesolver and
  receives no callable, so no caller code runs in the child. (A Windows venv's `python.exe` is a launcher that
  starts the base interpreter in a job object that ends with the launcher; terminating the launcher ends both,
  verified by PID in the tests.)
- **Transfer.** The equilibrated matrix goes to the worker's stdin as raw little-endian float64 bytes in C order
  after an 8-byte magic and the size N (`uint64`); three threads feed stdin and drain stdout and stderr, so the
  solving thread stays free. The worker returns only what the certificate uses: the singular values (it uses
  `σ_1` and `σ_N`; no `U` or `V`). Equilibration, the κ bound, δ, `σ⁻` and the exact integer residual stay in the
  solving process.
- **The worker's own checks.** After importing NumPy (which must be the parent's: the package directory is
  compared) and before reading the matrix, immediately before the SVD and immediately after it, the worker runs
  the route's FP probe (round-to-nearest-even, gradual underflow) and the thread probe: threadpoolctl must be
  importable, NumPy's own BLAS (a library under NumPy's package directory or `numpy.libs`) must be among the pools
  and every BLAS pool must report exactly one thread. A failure ends the worker with an `EN` record and refuses
  `COUPLED_UNCERTIFIED` with the worker's reason (`blas_threads`, `subnormal_flush`, `rounding_mode`). With one
  thread the BLAS runs inline on the worker's main thread, whose environment is the one probed. The parent's own
  FP environment, profile and trace hooks and thread limits do not reach the worker, which starts with the
  process defaults.
- **Validation in the parent.** An `OK` record must have exactly `10 + 8N` bytes, echo N, and hold N float64
  values that are all finite, non-negative and in descending order, decided on the bit patterns (a `-0.0` is
  taken as `+0.0`). Anything else refuses `COUPLED_UNCERTIFIED` (`svd_worker`).
- **Failures.** Every worker failure refuses `COUPLED_UNCERTIFIED` with reason `svd_worker`; the diagnostic
  `worker_failure` names it: `start_failed` (the process could not be created, for example an audit policy that
  denies `subprocess.Popen`), `io` (the transfer failed), `timeout` (no result within 600 s), `exit` (a crash,
  a kill or any other nonzero exit), `bad_output` (a malformed record or a zero exit without one) and
  `environment` (an unknown environment record, or another NumPy). A `MemoryError` in the worker (`MM` record)
  refuses `COUPLED_TOO_LARGE` (`memory_error`). A LAPACK non-convergence (`LA` record) is the certificate reason
  `numerical_failure:LinAlgError`, as before.
- **Cancellation.** While the worker runs, the parent polls the caller's cancellation (then its own FP
  environment) every 50 ms. On a cancellation, a callback error or any other exception it terminates its own
  worker by the handle it created (never by image name), waits for it, closes the pipes and re-raises: a
  cancellation refuses `COUPLED_CANCELLED`.
- **Guarantee.** Under A1 the singular values are those of LAPACK's `gesdd` path in NumPy, run on one verified
  BLAS thread with verified round-to-nearest and gradual underflow, in a process in which no caller code ran.
  Not covered: an attacker who can replace the interpreter, NumPy, OpenBLAS or the worker file on disk, or who
  can write into another process's memory; those are outside the trusted-construction boundary (OD9).
- **Measured overhead.** At N = 5000 (one thread, shared machine at 73 % load): median 28.2 s for the whole worker run against 27.6–28.0 s for
  the worker's own SVD; the overhead is about 0.26 s per run (start-up to NumPy imported 0.10 s, reading the 200 MB
  matrix 0.04 s, writing it, the result and the exit). Peak commit at N = 4969: parent 830 MiB above post-import
  (unchanged; reached in the equilibration), worker 431 MiB (the matrix it read and LAPACK's copy). The
  `COUPLED_TOO_LARGE` memory estimate 4·N²·8 bytes still bounds the dense copies held at once (two in the parent
  during the equilibration; one in the parent and two in the worker during the SVD).
  `evidence/coupled_specimen/RESULT.md` has the runs.

### Candidate (spec §2.5)
`splu(csc(S))` is called directly, with up to five refinement steps on the exact residual (owner decision OD12,
2026-11-20; two before, which left the h = 0.05 specimen refusing `forward_error_above_limit`). The candidate carries no
claim. A `MemoryError` maps to `COUPLED_TOO_LARGE`. Any other failure, including an exactly singular factor, maps to
`COUPLED_UNCERTIFIED` (`candidate_failed`).

## Diagnostics V4 and V5 (they can only add refusals)

All references take the normwise form of OD7 and OD8. Each is sized by the full magnitude of the operands on a row
or a side, never by one component's own terms. OD8 inventoried every a posteriori check:

| Check | Reference | Status |
|---|---|---|
| V1–V3 backward error, per row | `Σ_j|S_ij|·‖x‖∞ + |b_i|` | OD7 (it was `Σ_j|S_ij x_j| + |b_i|`) |
| V4 rigid compatibility `C_shell R_s = C_solid R_v`, per side | `1e-12·max(‖|C_s||R_s|‖∞, ‖|C_v||R_v|‖∞)` | already normwise, unchanged |
| V4 interface balance, per side and per rigid component `r` | `1e-10·Σ_i max_r(|C_sR_s|+|C_vR_v|)_ir·|λ_i|` | OD8 (it was each component against its own `Σ_i|C_sR_s λ|_ir+|C_vR_v λ|_ir`) |
| V5 global balance: kept loads, reactions (with the MPC master forces `−T_Dᵀ r_D`, review r1 #5) and skin load | `1e-9·Σ_{i,r}|v_i R_ir|` over every component | already normwise, unchanged |
| Transfer force and moment gates | `1e-12‖F_ref‖`, `1e-12(‖F_ref‖d + ‖M_ref‖)` (vector norms) | not per component, unchanged |
| V6 forward error | `1e-6·max_i|x_i/d_i|` | normwise, the authority |

`R_s` and `R_v` come from one routine, `_rigid_basis`. It builds the unscaled rigid motions about the footprint
centre: `e` on the translations, and `e × r` with `r = x − x_c` exact. It is never centred or normalised. The stored
centre `x_c` is the corner mean rounded once to binary64; the exported `footprint_centre` is exactly that point, so
every moment, V4 and the transfer gate refer to the exported value (review r1 #6). The exported `forward_error` is
`E` rounded up.

## Claims (C5)

| Output | Bound (exact rationals, rounded up once) |
|---|---|
| `solid_coefficients`, `multipliers` | `d_i·E` |
| `shell_displacements` (`u = T x_s + u0`) | `|float(u_i) − u_i| + sqrt_hi(Σ_j (T_ij d_j)²)·E` |
| `shell_interface_forces` (`C_shellᵀλ`), `solid_interface_forces` (`C_solidᵀλ`) | the same form, with `Cᵀ D_λ` |
| `side_resultants`: force, and moment about the footprint centre | `|float − exact| + sqrt_hi(Σ_k (Q_rk d_k)²)·E` |
| `reactions` | none; a plain array of the dependent (supported or MPC-slave) DOFs only |

`reactions` holds `r_D = (K u + C_shellᵀλ − F)_D` on the dependent DOFs and zero elsewhere. An MPC slave's
constraint force also acts on its masters as `−T_Dᵀ r_D` (V5 includes it, review r1 #5), and that master force is
not part of `reactions`: a consumer that sums `reactions` alone sees the support reactions plus the slave-side MPC
forces, not a balance with the applied load when MPCs are present.

`Q_side = −(C_shell R_s)ᵀ`, so `Q_side λ` is the resultant the solid exerts on the shell, couples included.

## Load route and partition

The coupled solve requires all of the following:
- `element_activity is None`;
- no follower pressure;
- empty `element_loads` and empty `gravity`.

Otherwise it refuses `COUPLED_LOAD_UNSUPPORTED`. The partition rules are carried unchanged from rev2 §2.5:
- the exact removed set and its boundary edges;
- orphans;
- active targets;
- every second edge node is a knot;
- each segment meets at least 2 lattice columns;
- `rank(C_solid R_v) = 6`.

Added by review r1:
- **Geometry, not membership (#3, r2 P1 (a), r2 (b)).** Let `P = {c0 + s a + t b + h n : 0 < s < 1, 0 < t < 1,
  |h| < t/2}` be the open prism over the corner rectangle, with no tolerance. Every test is exact.
  - The rule is **hull exclusion**: an element meets `P` iff the convex hull of its nodes does (exact
    separating-axis test). A warped kept element whose hull crosses `P` refuses, even when its bilinear surface
    clears the solid. So does a kept element that enters `P` by any positive amount, including a cut-edge node
    displaced inward by rounding on a rotated plane.
  - **Accepted v1 limit (owner decision OD11, 2026-11-20).** The overlap tolerance is zero. On a footprint whose
    plane is rotated against the axes, binary64 coordinates of cut-edge nodes can round inward by an ulp, and such a
    model refuses `COUPLED_CUT_MISMATCH` (`kept_inside_footprint`) although it is geometrically conforming. This
    is fail-closed and stays as the v1 behaviour; axis-aligned footprints are not affected.
  - The removed set is derived from the geometry: every removed element must have all its nodes in the closed
    prism and meet `P`, and every element that meets `P` must be removed. A caller-supplied set that differs
    refuses `COUPLED_CUT_MISMATCH` (`removed_set_geometry`); a kept element or kept node meeting `P` refuses
    `kept_inside_footprint`.
  - The prefilter is geometric: an element whose node bounding box misses the bounding box of `P` (its 8
    vertices, exact, rounded outward to binary64) cannot meet `P` and is skipped. No float solve decides anything.
- **Contract properties (#2, r2 P1 (b)).** `prepared.order == 2`, `geometry_order == 1`, units `'m'`, integer
  `ndof` and consistent float64/int64 coefficient arrays, float64 `(n_v, 6k)` rigid modes, and node DOFs equal to
  the DOF manager's mapping; otherwise `COUPLED_NATIVE_INVALID` (`prepared_order`, `prepared_mesh`, `prepared_ndof`,
  `prepared_coefficients`, `prepared_rigid_modes`, `dof_map`). The coefficient layout must be that of vector P2
  on the mesh: `ndof = 3(V + E)` with `V` vertices and `E` unique cell edges, `V + E` coefficients per component,
  and three coefficients at every vertex (`prepared_layout`).
- **Nonzero tests read bits (r2 P1 (c)).** Every "is nonzero" or "has entries" decision reads value bits, never
  stored structure: a `T` row holding only an explicit `0.0` is an inactive DOF for the connector-column rule.
- **C1 structure (r2).** Each CSR input has valid structure before any use (integer 1-D `indptr`/`indices`,
  `indptr` from 0 to `nnz` and nondecreasing, every column index in range; `csr_structure`). Each side's connector
  blocks have `rows ≥ 1`, shapes `(rows, n_shell)` and `(rows, n_v)`, and an affine vector of shape `(rows,)`.
  The native `K` and `F` are type-, structure-, shape-, finiteness- and window-checked before native reduction
  consumes them.
- **Ledger inventory (r2).** Besides the bitwise replay, each record's components are checked against the raw
  record: components `0..min(L, 6)−1` for a nodal record of length `L`, one per element DOF for a pressure record,
  each exactly once (`COUPLED_LOAD_UNCOVERED`, `ledger_inventory`).
- **Types before coercion (#7).** The skin traction, each side's connector blocks, the native `F`, `T`, `u0` and the
  independent list, the cancellation flag (an exact `bool`) and `memory_limit_bytes` (an exact non-negative
  integer, converted once) are type-checked before any conversion. Each `clock()` value must be an exact `float`
  or `int` (`clock_invalid`, review r3).
- **Ledger (#10).** Every nodal and pressure record of the case has a row, also when it yields no component.
- **Cancellation (#8, #9, r2 (d)).** The flag is polled after recovery, after each side's V4, every 50 ms while the
  SVD worker runs (OD13), inside the certified solve after verification, and last after the result object is
  built, so a cancellation during result construction refuses. Translating an exception never runs user code: the
  class name is read from the type slot and only exact `str` arguments are shown (anything else is `<not shown>`);
  no `__str__`, `__repr__` or `__format__` is called.
- **Hooks during translation (review r5 #3).** Each refusal built from a caught exception (the top-level
  translation, native assembly, the candidate, the FP probe) is built in a bounded retry (4 attempts). The
  interpreter unsets a trace or profile function that raises, so a hook that raises while the refusal is built (on
  the call or return of a helper) costs one attempt, and the refusal keeps the code and reason of the original
  exception. Not covered: a local line-trace function that raises on the final `raise` statement itself, and
  `sys.monitoring` callbacks (Python 3.12+), which the interpreter does not unset.

## Refusal codes

| Code | Meaning |
|---|---|
| `COUPLED_TOO_LARGE` | `N > 5000`, the memory limit, or a `MemoryError` anywhere |
| `COUPLED_UNCERTIFIED` | The certificate or a check failed. Reasons include `zero_row`, `equilibration_not_converged`, `nonfinite_entry`, `singular_within_svd_error`, `condition_above_limit`, `svd_error_bound_invalid`, `numerical_failure:*`, `candidate_failed`, `backward_error_above_limit`, `forward_error_above_limit`, `asymmetric_stiffness`, `v4_rigid_compatibility`, `v4_interface_balance`, `v5_global_balance`, `subnormal_flush`, `rounding_mode`, `fp_probe`, `blas_threads` and `svd_worker` (diagnostic `worker_failure`: `start_failed`, `io`, `timeout`, `exit`, `bad_output`, `environment`) |
| `COUPLED_FORMATION_RANGE` | An input has the wrong type, or a raw operand lies outside `W_raw` |
| `COUPLED_FORMATION_LOSS` | An exactly nonzero `B_s` or `g` entry rounds outside `W_raw`, or a raw load is stored as zero |
| `COUPLED_MESH_PROVENANCE` | The plane, lattice, facet-count or skin checks failed |
| `COUPLED_LOAD_UNSUPPORTED` | The load route is not admitted, a raw scalar has an unsupported type (`raw_scalar_type`), or the transfer gates failed (`transfer_resultant`) |
| `COUPLED_LOAD_UNCOVERED` | The ledger skipped a load component, or a record's components differ from its raw inventory (`ledger_inventory`) |
| `COUPLED_CUT_MISMATCH` | The removed set is invalid or not the geometric one (`removed_set_geometry`), a kept element or node meets the footprint (`kept_inside_footprint`), the cut is not one rectangular loop, or the shell faces differ in director or thickness |
| `COUPLED_FREE_NODE` | A support, MPC or nodal load names a node that the cut freed |
| `COUPLED_TARGET_REMOVED` | A load or mass target was removed or is inactive, or a DOF has no rigid motion |
| `COUPLED_MECHANISM` | A zero row, or `solid_rigid_mode_free` |
| `COUPLED_CONNECTOR_PARTITION` / `_GROUP` / `_RANK` | The connector prechecks failed |
| `COUPLED_CONNECTOR_INVALID` | Any other connector `ValueError` |
| `COUPLED_NATIVE_INVALID` | The native assembly or reduction failed, a contract property of the solid or the model does not hold, an argument has the wrong type (including `clock_invalid`), a container has the wrong type (`container_type`), or an unexpected exception was caught |
| `COUPLED_CANCELLED` | The `cancel` callable or event fired |

## Evidence

`evidence/coupled_specimen/RESULT.md` holds the S3 specimen, the references R0–R2 and the timing against the OD2
ceilings. The tests are:
- `tests/test_coupled_certificate.py`;
- `tests/test_coupled_formation.py`;
- `tests/test_coupled_r11_classes.py`, which includes the OD7 and OD8 tests and the ν = 0 patch test; its oracle
  uses an independent implementation of the specified power-of-two Ruiz scaling, an mpmath reference solution,
  and a high-precision κ (binary64 SVD triplets refined by Newton steps on mpmath residuals) judged against the
  exact cutoff 1e7 with no margin (review r3), in private mpmath contexts (review r5 #5);
- `tests/test_coupled_review_r1.py` to `tests/test_coupled_review_r5.py` (the review rounds; r5 holds the OD13
  worker tests: parent hooks, thread limits and FP environment do not reach the worker, isolation, cancellation and
  timeout by PID, every malformed worker record);
- `tests/test_solid_trace_extensions.py`;
- `tests/test_load_contributions.py`.

They run in the ANYsolid venv, because the pinned CI venv has no scikit-fem and skips them.
