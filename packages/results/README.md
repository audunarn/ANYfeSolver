# anyfesolver-results

**Alpha preview.** This release introduces the renamed Python packages. The solver has open shell equilibrium and checkpoint replay failures. Use it for migration testing. [Verification results](https://github.com/audunarn/ANYfeSolver/actions/runs/43105722933).

NumPy-only offline runtime for ANYfeSolver recovery capsules
(`RUNTIME_ID = "anysolver-results/2"`, also supporting the legacy runtime1 contract).

The package reconstructs solver response fields from the plain-data
recovery capsules exported by ANYfeSolver's `export_recovery_capsule`
(contract `anysolver.recovery_capsule`, versions 1 and 2) without importing
anyfesolver, scipy, ANYfem or ANYpost:

- `shell_surface_component(record, surface, component)` returns the requested
  local `xx`, `yy` or `xy` top/bottom stress using only its membrane/bending
  pair. It preserves float64 addition/subtraction, native sample shapes and
  missing values. This compatibility helper does not certify a full tensor or
  a new load combination.

- `reconstruct_response(responses, coefficients, *, active_indices=None)`
  returns one scalar spatial field using the solver's admitted
  baseline/active-pattern accumulation order.  Active indices are
  one-based response column indices; the runtime coefficients contain
  only those active weights in the original order.
- `prepare_response(responses, *, active_indices=None)` validates the same
  response matrix and active indices and returns a `PreparedResponse`
  owning immutable float64 copies of the baseline column and the
  original-order selected matrix (same memory layout as the ordinary
  expression; empty active order retains the baseline only).  Its
  `nbytes` is the exact retained baseline+selected byte count (Python
  int) for archive cache accounting; the archive owner owns all cache
  and transient-allocation accounting.
- `reconstruct_prepared_response(prepared, packed_coefficients)` replays a
  prepared response through the same shared arithmetic
  (`baseline + selected @ packed_coefficients`), bitwise identical to the
  ordinary reconstruction; only `prepare_response` results are admitted
  and per-request weights are validated for shape and finiteness.
- `derive_field(specification, inputs)` evaluates a closed derived
  operation (shell/beam von Mises, equivalent stress, shell principal
  stress ranks) from already reconstructed input fields.
- `validate_capsule(capsule)` performs the lightweight structural
  admission checks defined by the capsule contract.
- `OPERATIONS` is the closed operation registry; errors are typed
  (`CapsuleValidationError`, `ReconstructionError`, `DerivationError`).

The only runtime dependency is NumPy.  The full ANYfeSolver distribution
delegates its shared beam/shell/principal formulas to this package and
depends on it as `ANYfeSolver-results>=2,<3`; this standalone wheel
carries no ANYfeSolver payload.

The capsule format and exporter are documented in the ANYfeSolver
repository (`docs/RESULT_CAPSULE.md`).

Public reconstruction/preparation defaults retain `combination='legacy_blas/1'`.
Capsule2 consumers explicitly pass `combination='ordered_float64/1'` to both
`reconstruct_response` and `prepare_response`. This performs separate binary64
multiply/add operations in the original active order and adds the baseline last;
empty active order copies the baseline bytes. Prepared objects retain a read-only
algorithm tag; `reconstruct_prepared_response(..., combination=...)` refuses an
explicit mismatch. Ordered replay refuses nonfinite results with a typed error.
Capsule2 source exports default to this arithmetic and store exact source
corrections for primitive, displacement and derived differences. Readers must
honor per-case portability/unavailability and prefer those corrections. Principal
ranks remain explicit until portable spectral replay is separately demonstrated.
See `docs/DETERMINISTIC_CAPSULE_REPLAY.md` for operating conditions and legacy limits.

Storage grammar capability: version2.1 exposes STORAGE_PREDICTION_API=1 and validate_storage_prediction. This optional role is compression-only; it cannot reconstruct missing engineering stored rows or certify cases. Scientific runtime2 and legacy1 APIs remain unchanged.
