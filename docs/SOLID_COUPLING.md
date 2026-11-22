# Coupled shell–solid route (private, dense certificate)

`anyfesolver.solid_coupling.solve_shell_solid_coupled` solves a Q4 shell, with a rectangular set of elements
removed, together with a floating P2 solid detail (ANYsolid `PreparedSolid`) that fills the hole. The two are tied
weakly along the four cut sides. The result is either a `CoupledSolution` with certified bounds, or a
`CoupledSolveError` with a `COUPLED_*` code. Every `Exception` raised inside the route becomes a
`CoupledSolveError`. `KeyboardInterrupt`, `SystemExit` and the other `BaseException` subclasses that are not
`Exception` are not translated: they propagate unchanged, also when a callback raises them (OD9).

The module is private. `import anyfesolver` neither imports nor exports it. ANYfem is its only intended caller.
The certificate lives in `anyfesolver._coupled_certificate`. The spec is
`SOLVER-PR67-DENSE-SPEC-20261119.md`, which applies owner decisions OD1–OD16. `threadpoolctl` is a core dependency of
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

## Caller code in the solving process (OD15)

Owner decision OD15 (2026-11-21, "A"): like tampering with built objects (OD9), code that the caller has installed
in its own process and that runs during the call is **out of scope**:
- audit hooks (`sys.addaudithook`), including audit policies that deny operations;
- trace and profile hooks (`sys.settrace`, `sys.setprofile`, `threading.settrace`, `threading.setprofile` and
  the like);
- caller-defined subclasses of `str`, `int` and other built-ins, which the ANYfem builders do not produce.

Behaviour with such code installed is unsupported: the claims below are not made for it. The defences that already
fail closed against hooks stay (the certificate in the child process, the bounded refusal retries, the identity
type checks); no further hook defences are added. Findings that need no such code (genuine ANYfem-built inputs or
legal environment state, for example a file on the import path) remain in scope.

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
- **Where.** In the solving process: at entry, after every cancellation poll and every callback (below), and
  before the certificate request is sent. The certificate's own float steps (equilibration, SVD, outward
  rounding) run in the certificate process, which runs the same probe at each stage (OD14, see below).
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
- **Containers (review r5 #4, r6 P2).** Right after the exact-type checks of the model and the load case, every container
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
  Review r6: the type tests compare by identity only (`_type_in`), never by `in` on a set or tuple, which would
  run a metaclass `__hash__` or `__eq__` of the caller's class. The FEModel and the LoadCase must hold exactly
  their dataclass attributes (exact `str` keys), with `name` an exact `str` (and the model's `current_material`;
  `load_cases` an exact `list`, never read). The route copies no caller object through a copy protocol: the
  LoadCase copies are exact-type rebuilds (`_case_copy`), and the FEModel copies (`_model_copy`) replace
  `load_cases` (and, for the holed model, the MPC list it rebuilds) by new empty lists and share the materials
  instead of copying them; the mesh is copied by `FEMesh.__deepcopy__`, which copies its own fields only. Element
  objects and the support and MPC objects are copied as before; they are anyfesolver classes admitted above and
  lie inside the trusted-construction boundary (OD9).
- **Process-wide state.** The certificate premises A1 and A2 depend on the certificate process's environment only
  (below). The FP environment of the solving thread is probed as above. NumPy's `errstate` is per
  context. Other BLAS or LAPACK calls in the solving process (`matrix_rank` in the rigid-rank check, a small Gram
  solve, SuperLU) carry no premise: a different thread count or a hook changes their bits only, and their
  outcomes are refusals or a candidate that the exact checks judge.

### Certificate process (owner decisions OD13, 2026-11-20, and OD14, 2026-11-21)
Review r5 showed that an in-process single-thread window cannot be closed against same-process caller code, and
review r6 that the same holds for every float step of the certificate: a trace hook that switched FTZ/DAZ on for
the parent's `np.ldexp` line only (and off before the next probe) flushed subnormal entries during the
equilibration, and the worker then faithfully certified the wrong matrix. Under OD14 the **whole certificate runs
in a short-lived child process**. Parent hooks can then affect only formation and the candidate solve, and the
certificate covers their stored output. The solver's thread coordinator (`threading_policy`) is not used by the
route and is unchanged from the base.

**What runs where.**

| Step | Process |
|---|---|
| Formation of `S`, `b` and the recovery maps (incl. `T`, `u0`, `C`, the rigid basis and `C R`) | parent |
| SuperLU candidate and its refinement on the exact residual (no claim) | parent |
| Cap, memory limit, canonical CSR, finiteness, zero-load consistency (re-decided on the child's copy) | parent and child |
| Power-of-two equilibration, the dense SVD, δ, κ_up and the κ gate | child |
| `σ⁻` and its sign, the exact integer residual, V1–V3, `E`, V6 and `E` rounded up | child |
| Every bounded recovery value and its bound (displacements, coefficients, multipliers, interface forces, side resultants) | child |
| Decoding and validating the reply; raising the refusal | parent |
| V4 rigid compatibility and interface balance, V5 global balance, reactions (no claim) | parent |

The parent does no certificate arithmetic: it decodes integers, bit patterns and hexadecimal fractions and checks
them against the schema (below). The result's `x` is the child's copy of the candidate it verified, and every
`Bounded` value and bound, the shifts, `E`, `σ⁻`, κ_up and the backward record come from the reply.

- **Start.** `subprocess.Popen` (a new interpreter, never a fork) of `sys.executable -I -S
  anyfesolver/_coupled_svd_worker.py <NumPy package dir> <NumPy version> <threadpoolctl SHA-256>`. `-I` ignores
  every `PYTHON*` variable and the user site and keeps the script and working directories off `sys.path`; `-S`
  skips `site`, so no `.pth` file, `sitecustomize` or `usercustomize` runs. `sys.path` stays the standard library
  only. A finder appended to `sys.meta_path` (after the built-in, frozen and path finders) resolves exactly one
  top-level name, `numpy`, from the directory that holds the given package; the loaded package must be that
  directory and report the parent's version, else the child refuses `numpy_mismatch`. `threadpoolctl` is executed
  from the bytes the parent sends in the request (below), after their SHA-256 is checked against the argument
  (`threadpoolctl_mismatch`); the child opens no threadpoolctl file, and bytes that fail to execute leave it
  unimportable (`blas_threads`). Review r6 P1: before, those directories were prepended to `sys.path`, so a
  `numbers.py` next to threadpoolctl shadowed the standard library; review r7 P1: a same-name `threadpoolctl/`
  package next to the file took precedence over it; review r8 P1: the parent hashed whatever `threadpoolctl` it had
  imported, so a same-name module from an earlier `sys.path` entry was run by the child and trusted. The arithmetic
  module
  `_coupled_certificate_math.py` is loaded by its path next to the worker; nothing else from anyfesolver is
  imported. The environment given to the child is exactly `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1` and, on Windows, `SYSTEMROOT`. On Linux CPython itself may then add an `LC_CTYPE` (C-locale
  coercion, PEP 538: `C.UTF-8` where the interpreter build and the host's locales support it); it is not copied from
  the parent, and the isolation test accepts no other value. The working directory is the anyfesolver package directory. The child receives no
  callable, so no caller code runs in it. The `Popen` object is created first and initialised second, so an
  exception after the OS created the process (for example a hook raising on return from `Popen.__init__`) still
  kills and reaps it (review r6 P2). (A Windows venv's `python.exe` is a launcher that starts the base interpreter
  in a job object that ends with the launcher; terminating the launcher ends both, verified by PID in the tests.)
- **Modules the child trusts (review r8 P1), and how each is selected.** None is selected through `sys.modules`
  or the parent's `sys.path` (`_coupled_certificate._trusted_runtime`, run on a lane inside the deadline):

  | Module | Selected by | Checked |
  |---|---|---|
  | NumPy | the package the parent runs, which must be the one installed `numpy` distribution of its own directory (`importlib.metadata`, dist-info in that directory) | parent: same version as the dist-info, `numpy/__init__.py` is that very file and matches its RECORD SHA-256 (`numpy_mismatch`); child: imports only from that directory, same directory and version (`numpy_mismatch`) |
  | threadpoolctl | the one installed `threadpoolctl` distribution in NumPy's directory, file `threadpoolctl.py` from its RECORD | parent: read once, SHA-256 equal to the RECORD's (`threadpoolctl_mismatch`, also when it is missing or unreadable), and exactly those bytes are sent; child: runs the bytes received after checking their SHA-256 |
  | standard library | `sys.executable` with `-I -S` | the interpreter installation (OD9) |
  | `_coupled_svd_worker.py`, `_coupled_certificate_math.py` | this package's own files, by path (the files the parent itself imported) | files on disk (OD9) |

  A provenance refusal (`numpy_mismatch`, `threadpoolctl_mismatch`, code `COUPLED_UNCERTIFIED`) comes from the
  parent before any child starts. NumPy's other files (extension modules, the bundled OpenBLAS) are loaded from
  disk by the child; their content is the installation (OD9); hashing every file of NumPy's RECORD would cost
  about 0.45 s per solve (measured, 42 MB) and adds no selection guarantee beyond the directory and version.
- **Overlap.** The child is started before the candidate: it certifies `S` while the parent computes the candidate,
  then reads the candidate and verifies it. Refusal precedence is unchanged: a certificate refusal (κ, σ⁻) wins
  over a candidate failure (`candidate_failed`, a candidate `MemoryError`), which wins over V1–V3 and V6. On the
  zero-load path no candidate is sent and the child uses `x = 0`.
- **The child's own checks.** After importing NumPy (which must be the parent's: the package directory is
  compared), the child runs the route's FP probe (round-to-nearest-even, gradual underflow) and the thread probe
  (threadpoolctl importable, NumPy's own BLAS among the pools, every BLAS pool at one thread) at start, after
  reading the request, between the equilibration and the SVD, after the SVD, after reading the candidate, after the
  exact residual and before replying. A failure refuses `COUPLED_UNCERTIFIED` with the child's reason
  (`blas_threads`, `subnormal_flush`, `rounding_mode`); a `MemoryError` anywhere in the child, including the
  thread probe (review r6 P2), refuses `COUPLED_TOO_LARGE` (`memory_error`). The parent's FP environment,
  profile and trace hooks and thread limits do not reach the child, which starts with the process defaults.
- **Wire format (little endian).** Request on stdin: `ANYCRT03`, a `uint64` length, a JSON header, the
  threadpoolctl bytes (`threadpoolctl_bytes` in the header, review r8), then raw arrays
  in header order, each with its exact shape and dtype: `indptr` (`int64`, N+1), `indices` (`int64`, nnz), `data`
  (`float64`, nnz), `b` (`float64`, N, solve mode), and per recovery map `indptr` (`int64`), `columns` (`int64`)
  and either `float64` coefficients or a comma-separated text of exact fractions (`[-]hex/hex`, reduced), then the
  optional `float64` offsets. The header names the mode (`certificate` or `solve`), N, nnz, `raw_load_zero`, the
  candidate (`none`, `zero` or `follows`), the memory limit, the maps' sizes and the constants (cap 5000, the
  κ limit's bit pattern, the V1–V3 and V6 tolerances, the sweep limit, the SVD constant, the dense-copy count,
  the root precision); the child refuses a request whose constants differ from its own in value or in JSON type.
  With `follows`, the parent later sends `XV` and N `float64` (the candidate) or `XF` (no candidate). The parent
  closes stdin after its last byte; the child refuses (exit 2) a header with a float or constant literal or a
  repeated key, and any byte after the request (review r7). Review r8: the child reads and validates the whole
  request, the candidate and the end of the stream included, before it writes **any** verdict, on every path: an
  early refusal (`zero_row`, `memory_limit`, `load_stored_as_zero`, ...), an environment verdict and a
  `MemoryError` first read (and discard) the rest of the request, so a trailing byte is a protocol error (exit 2)
  everywhere. The one exception is a NumPy that cannot be loaded or is not the parent's (`numpy_mismatch`): the
  request cannot be parsed without NumPy, and the parent's RECORD check comes first. N = 0 is a valid request
  (below). A rational map with no entries has an empty text (zero bytes), accepted on both sides.
- **Request limits (review r8).** One set of constants in `_coupled_certificate_math` (`MAX_HEADER_BYTES` 16 MiB,
  `MAX_EXPORTS` 64 maps, `MAX_EXPORT_ENTRIES` 2^26 rows and entries per map, `MAX_EXPORT_TEXT_BYTES`,
  `MAX_THREADPOOLCTL_BYTES` 4 MiB). The parent checks them before it starts the child and refuses
  `COUPLED_UNCERTIFIED` (`export_limit`); the child checks the same constants before any allocation (exit 2).
  Before, 65 maps reached the child and surfaced as `svd_worker`/`exit`. The route sends 5 + (number of box sides)
  maps.
- **Recovery maps (review r7 P1).** A map row is bounded by `sqrt_hi(Σ_k (M_rk d_j(k))²)·E`, the 2-norm over the
  row's entries; that is the row norm only when no column repeats in the row (two unit entries on one column are
  the coefficient 2, norm 2, not √2: an accepted bound fell below the actual error). So every row must have
  distinct columns in `[0, N)`, in any order (`_coupled_certificate_math.export_structure_reason`): the parent
  refuses a map that breaks this (`invalid_arguments`), the child refuses it on its own copy (exit 2), and
  `bounded_rows` refuses it again (`invalid_export`). The route's maps (identities, `T`, `Cᵀ` from canonical CSR,
  the side resultants with one entry per multiplier) have distinct columns. The stored system itself must be
  canonical CSR (sorted, distinct columns), checked on both sides. Row pointers (review r8 P2) are checked by
  `row_pointer_reason`: 0 first, nnz last, nondecreasing, decided by element comparisons and Python integers. Before,
  `np.diff` of `[0, 2^62, −2^63, −2^62, 0]` overflowed to four positive steps, the map passed and `bounded_rows`
  raised `IndexError`. Inventory of overflow-prone integer operations on caller-derived or wire integers: the row
  pointers of the parent's export maps, of the child's stored system and maps (`canonical_csr_reason`,
  `export_structure_reason`) and of formation's `_check_csr_structure` all use it now; the `np.diff` and `np.repeat`
  that follow it run on a checked pointer (steps in `[0, nnz]`); column ranges are `int()` of min and max; the
  shifts are checked by two comparisons (review r7); exponents, mantissas and the residual are Python integers or
  bounded `int64` (|exponent| < 2200); header sizes are Python integers.
- **N = 0 (review r7).** The 0 × 0 system goes through the child like any other: its certificate is the fixed
  `no_free_dofs` record, `x = ()`, `E = 0`, `σ⁻ = 1`, and the child evaluates the maps (offset-only rows return
  their offset with bound 0). Before, the parent returned empty arrays for every map, so an offset-only row lost
  its value. Reply on stdout: `ANYCRR02`, a
  `uint64` length, a JSON reply and the binary sections it lists (`shifts` `int64`·N; `x` `float64`·N;
  `values<k>`, `bounds<k>` `float64` per map). Floats in the JSON are their unsigned 64-bit patterns, exact
  rationals hexadecimal fractions; the JSON holds no float literal. Exit 0 for the verdicts `accept`, `refuse`,
  `certificate` and `no_candidate`, 3 for `environment`, 4 for `memory`, 2 for a malformed request (no reply).
- **Validation in the parent.** The magic, the lengths, every key and type of the reply and of the certificate and
  backward records (no unknown, missing or repeated key; exact `int`, `bool` and `str`; no JSON float), the exit
  code against the verdict, the section list against the expected names and byte lengths (exactly; no trailing
  byte), finite values, non-negative bounds and shifts within ±2^20 (two comparisons: `abs(−2^63)` overflowed and
  passed). A refusal is checked against the table of refusals the child can send in that mode, with exactly the
  diagnostics that refusal carries and its certificate record accepted or not (review r7: certificate mode admitted
  solve-only reasons). An accepted certificate must carry the full record with every field non-null, finite and
  non-negative, and a κ_up bit pattern at most the limit's; `forward_error_up` and the backward `max_ratio` must be
  finite and non-negative. Anything else refuses `COUPLED_UNCERTIFIED` (`svd_worker`).
- **Failures.** Every worker failure refuses `COUPLED_UNCERTIFIED` with reason `svd_worker` (the name is kept from
  OD13; the worker now runs the whole certificate); the diagnostic `worker_failure` names it: `start_failed`,
  `io` (the transfer failed or a pipe stayed open past the bounded wait), `timeout` (the child still ran at the
  deadline, 600 s after the run began),
  `exit`, `bad_output` and `environment` (an unknown environment reason, or another NumPy). A `MemoryError` in a
  transfer thread of the parent also refuses `COUPLED_TOO_LARGE` (`memory_error`, review r6 P2), never a timeout
  (state machine below). Coded refusals
  from the child keep their code and reason (for example `condition_above_limit`, `forward_error_above_limit`,
  `bound_not_finite`).

#### Certificate process: states (review r8)
One explicit protocol for the run (`_coupled_certificate._Run`, `_start_worker`), instead of point fixes. **One
deadline** `D` = (monotonic time when `_run_worker` begins) + 600 s, set before any child-related I/O, the
provenance read included, and carried through the candidate to the verdict. Every blocking operation of the
calling thread is a bounded wait: on a lane's lock, the process handle or the candidate lock, each with at most
the time left to `D`; reading files, creating the process, writing and reading pipes happen on lanes
(`_thread.start_new_thread`, no wait on the thread registry), never in the calling thread. Two fixed bounds do
not depend on `D`: after the child has exited its lanes get at most 10 s (an exited child's unread output is at
most one pipe buffer per stream), and TEARDOWN gets 30 s for each of reaping and the lanes.

| State | What happens | Next |
|---|---|---|
| PREPARE (launch lane) | `_trusted_runtime` (RECORD reads, may block on a hung file system), argv, request bytes | provenance refusal → TEARDOWN, refusal; lane not done by `D` → TEARDOWN, `timeout`; else LAUNCH |
| LAUNCH (launch lane) | `Popen`; the process is handed to the run under a lock | creation fails → TEARDOWN, `start_failed`; run already abandoned → the lane kills and reaps its own process; else START |
| START | the writer, stdout and stderr lanes, each started only before `D` | a lane cannot start (thread limit, `MemoryError`) → TEARDOWN at once, `io` / `memory_error`; else RUNNING |
| RUNNING | the calling thread computes the candidate (caller code) and posts it; then it polls every 50 ms: cancellation, the FP probe, lane failures, `poll()` | cancellation or a callback error → TEARDOWN, re-raise; lane failure → TEARDOWN, `memory_error` / `io`; child exited → COLLECT; `D` passed with the child running → EXPIRED |
| COLLECT | the lanes finish (≤ 10 s), then lane failures are checked again (after the final read), then the reply is decoded | lane failure → `memory_error` / `io`; lanes not finished → `io`; else the decoded verdict, **whatever the clock says now** |
| EXPIRED | TEARDOWN first, then lane failures are checked | lane failure → `memory_error`; else `timeout` |
| TEARDOWN (every path, idempotent) | abandon the run (a launch still in progress reaps its own process), release the writer waiting for the candidate, kill and reap the child by its own handle (never by image name), wait for the lanes, close only the pipe ends whose lane finished or never started | — |

**Verdict precedence.** A lane failure (a `MemoryError` is `COUPLED_TOO_LARGE`) wins over everything, and is
checked at every poll, after the final read and after an expiry: a reader that fails just after a poll, while the
child blocks on a full pipe, is never reported as a timeout (review r7 item 5, r8). Otherwise an exited child's
complete reply decides: the parent looks at what it received, not at the wall clock when it looks, so a parent
suspended past `D` (for example in a cancellation callback or by the OS) keeps a timely refusal or acceptance
(review r8). `timeout` is reported only when the child was observed still running at or after `D`. The observation is made by a poll that follows the clock read: when the clock says `D` has passed, the parent polls once more, and a child that exited in the window between the first poll and the clock read decides by its reply (OD17; `tests/test_coupled_od17_race.py`). Every transition above is check-then-act on a child that can change state meanwhile, and each one either re-checks (RUNNING → EXPIRED) or is idempotent and tolerant of the change (TEARDOWN's poll-then-kill, retried; the candidate post and the writer release, now under one lock). Since the
child reads the whole request before any verdict, a certificate refusal is sent only after the candidate (or
`XF`) arrives; a parent that delivers the candidate after `D` therefore gets `timeout`, which is correct: no
verdict existed before `D`. Each transition has a test in `tests/test_coupled_review_r8.py` (`test_r8_state_*`;
RUNNING → TEARDOWN on cancellation and on a lane failure at a poll are `test_od13_cancel_*` and
`test_r7_reader_memory_error_on_a_large_reply_is_too_large`, COLLECT with a pipe held open is
`test_r6_p2_teardown_is_bounded_when_a_pipe_stays_open`).
- **Cancellation.** While the child runs, the parent polls the caller's cancellation (then its own FP environment)
  every 50 ms. On a cancellation, a callback error or any other exception it terminates its own child by the handle
  it created (never by image name) and re-raises: a cancellation refuses `COUPLED_CANCELLED`.
- **What parent hooks can and cannot affect.** In-process hooks are out of scope (OD15, above); this records what
  the design already gives. Caller code in the parent (trace, profile and audit hooks, other
  threads, the FP environment) can change formation, the recovery maps and the candidate. The child certifies
  exactly the bytes it received, so the certificate claim (κ_up, `|x_i − x*_i| ≤ d_i·E` for the exact solution
  `x*` of the received `S` and `b`, and every exported bound) holds for those bytes whatever the parent did. A
  hook cannot make the child accept a matrix whose κ_up exceeds the limit, a candidate that fails V1–V3 or V6, or
  a bound below the exact one; it can make the parent refuse, form a different system (formation: the OD9
  trusted-construction boundary), or suppress the parent-side diagnostics V4 and V5, which only add refusals and
  carry no claim. It also cannot change the returned values after the child computed them except by mutating the
  returned object, which no in-process design prevents.
- **Guarantee.** Under A1 the singular values are those of LAPACK's `gesdd` path in NumPy, run on one verified
  BLAS thread with verified round-to-nearest and gradual underflow, in a process in which no caller code ran, on
  the matrix the child equilibrated itself from the received stored system. Not covered: an attacker who can
  replace the interpreter, NumPy, OpenBLAS, threadpoolctl or the worker files on disk, or who can write into
  another process's memory; those are outside the trusted-construction boundary (OD9). The parent's record checks
  establish the reply's format, not which program wrote it.
- **Measured.** See `evidence/coupled_specimen/RESULT.md` (OD14 section): the whole certificate at N = 5000
  including the child, the specimen time and the peak commit of parent and child.

### Candidate (spec §2.5)
`splu(csc(S))` is called directly, with up to five refinement steps on the exact residual (owner decision OD12,
2026-11-20; two before, which left the h = 0.05 specimen refusing `forward_error_above_limit`). The candidate carries no
claim. A `MemoryError` maps to `COUPLED_TOO_LARGE`. Any other failure, including an exactly singular factor, maps to
`COUPLED_UNCERTIFIED` (`candidate_failed`).

## Diagnostics V4 and V5 (they can only add refusals)

V4 and V5 run in the solving process on the values the certificate process returned (OD14). They check formation
(rigid compatibility of the connector, interface and global balance) and can only add refusals; a parent hook that
suppresses one changes no certified number (see "What parent hooks can and cannot affect").

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

## Minimum resolution for coupled mode (OD17)
The coupled route is not guaranteed to be at least as accurate as all-shell on a coarse shell mesh. The rule: **at
least 8 shell elements across the footprint's shorter side** (the specimen's footprint is 0.4 m wide, so h = 0.05
meets it and h = 0.1, 4 elements, does not). Below it the solid's 3-D field, read on the mid-plane, differs from the
shell model by more than the discretisation error of the shell, and the coupled error can exceed the all-shell
error: on the S3 specimen the coupled centre error is 1.54 times the all-shell error at h = 0.1 (5.97e-7 against
3.88e-7), against 1.15 (centre) and 0.96 (quarter point) at the resolved h = 0.05. The S3 R1 gate (OD17) therefore
applies the error-ratio limit 1.2 at h = 0.05 only, reports the h = 0.1 ratio, and tests the observed order
one-sided (`p_coupled >= p_shell - 0.2`; 0.957 against 0.539). ANYfem's coupled mode will warn below this
resolution, in the later ANYfem PR; the solver itself does not check it.

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
- `tests/test_coupled_review_r1.py` to `tests/test_coupled_review_r8.py` and `tests/test_coupled_od17_race.py`
  (the review rounds; r5 holds the OD13
  worker tests: parent hooks, thread limits and FP environment do not reach the worker, isolation, cancellation and
  timeout by PID, every malformed worker record);
- `tests/test_solid_trace_extensions.py`;
- `tests/test_load_contributions.py`.

Only one portable CI shard installs scikit-fem (Windows, index 0, subshard 0), and these modules are not assigned
to it, so the portable run skips most of them. `.github/workflows/coupled-solid.yml` installs
the solid stack and runs all of them on Windows and Linux. On Linux the tests that switch denormal flushing or
the rounding mode through the MSVC runtime, or inspect processes through the Windows Toolhelp API, are skipped,
so those behaviours are tested on Windows only.
