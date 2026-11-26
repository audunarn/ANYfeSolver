# Batched qualified Q4 planar formation and recovery (L4)

Owner decision L4 (relayed by the CI-cleanup coordinator, 2026-11-26): batch the formation and
default recovery numerics of the exact qualified Q4 planar route inside the existing authority
lease. Only L4 is approved; L1, L2, L3 and L5 of the Q4 speed study are not implemented.
Results are bit-identical. Source binding:
`docs/reference_cases/qualified_q4_planar_batch_successor.json` (builder
`scripts/build_q4_planar_batch_successor.py`), the single recorded successor of the guard
cadence successor.

## What changes

- `src/anyfesolver/_q4_planar_batch.py` holds numerics only, on stacked arrays of elements
  that have already been admitted. `form` reproduces the planar branch of
  `QualifiedE4PLShellElement.compute_stiffness_components` and `recover` reproduces the default
  `compute_stresses` (four Gauss points, local resultants). Both use the same compensated
  contraction policy for each dot product as the per-element code. The module keeps every
  certification guard of the per-element code (Jacobian, symmetry, Ruiz scaling, backward
  error, finiteness, margins). It refuses invalid or non-finite inputs with the private
  `PlanarBatchFallback`.
- Two small BLAS sums in the per-element code have an order that depends on the CPU dispatch:
  the 4x4 coefficient reduction and the residual-mode sum. The module emulates both. A runtime
  known-answer self-check, run once per process, confirms that this BLAS/CPU build sums in the
  emulated order. Otherwise `available()` is False and the per-element path runs.
- `e4_pl_element.py` (sealed) adds three entry points:
  - `_qualified_planar_batch_formation_expected`
  - `_qualified_planar_batch_stiffness`
  - `_qualified_planar_batch_stresses`

  They also add a private hook for the runtime-authority generation bracket (D1). Admission
  stays per element: exact type, quadrature authority, cache key, elastic increment admission,
  warpage, `equation7_frame`, constitutive and drilling stiffness, and recovery admission. The
  batch result is bound through the existing component cache and guard bindings, exactly as
  `compute_stiffness_matrix` binds it.
- `matrix_assembly.py` and `assembly.py` call the batch for runs of at least 96 formation misses
  or 128 recovery elements, inside the existing lease. Below these sizes, or when no batch
  fires, the original per-element loops run unchanged.
- Any refusal reruns the whole batch through the per-element path, so that path reports its
  own error. Refusals include the self-check, input validation, certification and a binding
  failure.

## Trust boundary and contract notes

- NumPy, Numba, SciPy, Python builtins, the compensated contraction modules and the ANYfeSolver
  classes and guard machinery are trusted not to be rebound at run time.
- Hostile data, instance attributes and subclasses stay in scope. They are refused, or sent to
  the per-element path.
- Mutation in the middle of a request is out of contract.
- D2 (accepted): the batched formation observes the mesh `revision_signature` once per element,
  through the cache key the caller has just computed. It then checks that the coordinates it
  reads reproduce the key's relative-coordinate entry. It does not recompute the key inside the
  formation. A foreign mutate-and-restore during that single observation is outside the
  contract. Persistent mutations are still refused at the lease boundaries.
- A runtime-authority generation move during the shared batch kernel belongs to no single
  element. It discards the batch, and every element reruns under the new generation.
- In error cases, cache-hit elements that come before a failing element may already be bound
  when the error is raised.
- Both differences are visible only outside a lease. The caller sees the same error and state.

## Tests

- `tests/test_q4_planar_batch.py`: kernel identity against the per-element code, refusals,
  no aliasing and the forced fallback hook.
- `tests/test_q4_batch_wiring.py`: fault injection, using `sys.monitoring` on Python 3.12+. It
  covers collection-time precedence, bind-failure restore, generation moves inside each
  element's span and inside the shared kernel, and the D1 bracket.
- `tests/test_q4_batch_paths.py`: CI coverage of both paths. Plate, Scordelis-Lo, pinched,
  irregular, mixed Q4/S3 plate and shell, and beam-stiffened cases must give bit-identical
  displacements and complete `compute_stresses` results on the fast path and under
  `_forced_fallback()`.
- `tests/test_q4_planar_batch_successor.py`: the source-binding record and its chain.

## Measurements

Pending the final A/B CPU run on the G2 base (d4cf5011f).
