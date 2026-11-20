# Coupled shell–solid route (private, dense certificate)

`anyfesolver.solid_coupling.solve_shell_solid_coupled` solves a Q4 shell, with a rectangular set of elements
removed, together with a floating P2 solid detail (ANYsolid `PreparedSolid`) that fills the hole. The two are tied
weakly along the four cut sides. The result is either a `CoupledSolution` with certified bounds, or a
`CoupledSolveError` with a `COUPLED_*` code. No other exception escapes.

The module is private. `import anyfesolver` neither imports nor exports it. ANYfem is its only intended caller.
The certificate lives in `anyfesolver._coupled_certificate`. The spec is
`SOLVER-PR67-DENSE-SPEC-20261119.md`, which applies owner decisions OD1–OD8.

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
  So the floating-point environment adds no premise. The SVD and SuperLU calls are the only floating-point library
  calls. The SVD is covered by A1/A2 below. SuperLU supplies a candidate only.
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
The path is taken when native `F = 0`, `u0 = 0` and every consumed raw input is exactly zero: every pressure, kept
and transferred, every consumed nodal component, every prescribed support value and every MPC right-hand side
(review r1 #4). Then `x = 0`, and the full certificate must still accept. A nonzero raw input whose stored `b` is all
zero refuses `COUPLED_FORMATION_LOSS` (`load_stored_as_zero`), for example a kept pressure of `2^-1074` whose native
contributions round to zero.

### Floating-point environment (review r1 #1)
Every exact step reads and writes binary64 values by IEEE bit extraction (`_decompose`, `_exact`, `_to_float`,
`_out_up`, the nonzero masks and the raw window), so its result does not depend on FTZ/DAZ. The remaining float steps
(equilibration, the SVD, outward rounding, comparisons, native assembly) assume gradual underflow, as A2 does. The
route therefore probes the thread at entry, after every cancellation poll and around the certified solve, and refuses
`COUPLED_UNCERTIFIED` (`subnormal_flush`) when subnormal operands or results are flushed.

### Candidate (spec §2.5)
`splu(csc(S))` is called directly, with up to two refinement steps on the exact residual. The candidate carries no
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
| `reactions` | none; a plain array |

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
- **Geometry, not membership (#3).** No kept element (its convex hull, by an exact separating-axis test) and no kept
  node may meet the open footprint prism `{c0 + s a + t b + h n : s, t ∈ (1e-9, 1 − 1e-9), |h| < t/2}`
  (`COUPLED_CUT_MISMATCH`, `kept_inside_footprint`).
- **Contract properties (#2).** `prepared.order == 2`, `geometry_order == 1`, units `'m'`, integer `ndof` and
  consistent float64/int64 coefficient arrays, float64 `(n_v, 6k)` rigid modes, and node DOFs equal to the DOF
  manager's mapping; otherwise `COUPLED_NATIVE_INVALID` (`prepared_order`, `prepared_mesh`, `prepared_ndof`,
  `prepared_coefficients`, `prepared_rigid_modes`, `dof_map`).
- **Types before coercion (#7).** The skin traction, each side's connector blocks, the native `F`, `T`, `u0` and the
  independent list, the cancellation flag (an exact `bool`) and `memory_limit_bytes` (an exact non-negative
  integer) are type-checked before any conversion.
- **Ledger (#10).** Every nodal and pressure record of the case has a row, also when it yields no component.
- **Cancellation (#8, #9).** The flag is polled after recovery, after each side's V4 and before return, and inside the
  certified solve after verification; no exception's text or name can make the translation itself raise.

## Refusal codes

| Code | Meaning |
|---|---|
| `COUPLED_TOO_LARGE` | `N > 5000`, the memory limit, or a `MemoryError` anywhere |
| `COUPLED_UNCERTIFIED` | The certificate or a check failed. Reasons include `zero_row`, `equilibration_not_converged`, `nonfinite_entry`, `singular_within_svd_error`, `condition_above_limit`, `svd_error_bound_invalid`, `numerical_failure:*`, `candidate_failed`, `backward_error_above_limit`, `forward_error_above_limit`, `asymmetric_stiffness`, `v4_rigid_compatibility`, `v4_interface_balance`, `v5_global_balance` and `subnormal_flush` |
| `COUPLED_FORMATION_RANGE` | An input has the wrong type, or a raw operand lies outside `W_raw` |
| `COUPLED_FORMATION_LOSS` | An exactly nonzero `B_s` or `g` entry rounds outside `W_raw`, or a raw load is stored as zero |
| `COUPLED_MESH_PROVENANCE` | The plane, lattice, facet-count or skin checks failed |
| `COUPLED_LOAD_UNSUPPORTED` | The load route is not admitted, or the transfer gates failed (`transfer_resultant`) |
| `COUPLED_LOAD_UNCOVERED` | The ledger skipped a load component |
| `COUPLED_CUT_MISMATCH` | The removed set is invalid, the cut is not one rectangular loop, or the shell faces differ in director or thickness |
| `COUPLED_FREE_NODE` | A support, MPC or nodal load names a node that the cut freed |
| `COUPLED_TARGET_REMOVED` | A load or mass target was removed or is inactive, or a DOF has no rigid motion |
| `COUPLED_MECHANISM` | A zero row, or `solid_rigid_mode_free` |
| `COUPLED_CONNECTOR_PARTITION` / `_GROUP` / `_RANK` | The connector prechecks failed |
| `COUPLED_CONNECTOR_INVALID` | Any other connector `ValueError` |
| `COUPLED_NATIVE_INVALID` | The native assembly or reduction failed, a contract property of the solid or the model does not hold, an argument has the wrong type, or an unexpected exception was caught |
| `COUPLED_CANCELLED` | The `cancel` callable or event fired |

## Evidence

`evidence/coupled_specimen/RESULT.md` holds the S3 specimen, the references R0–R2 and the timing against the OD2
ceilings. The tests are:
- `tests/test_coupled_certificate.py`;
- `tests/test_coupled_formation.py`;
- `tests/test_coupled_r11_classes.py`, which includes the OD7 and OD8 tests and the ν = 0 patch test;
- `tests/test_solid_trace_extensions.py`;
- `tests/test_load_contributions.py`.

They run in the ANYsolid venv, because the pinned CI venv has no scikit-fem and skips them.
