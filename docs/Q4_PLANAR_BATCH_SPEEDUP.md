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
- Owner decision A (2026-11-26): in a failing assembly, elements after the failing one may already
  have had their cache-hit steps applied: stale caches cleared, guards rebound. This happens because
  the batched route collects every element's cache-hit steps before it forms the misses, while the
  per-element loop stops at the failing element. The error and every result are identical, and a
  following successful solve ends in the same state on both paths
  (`tests/test_q4_batch_wiring.py::test_failing_assembly_with_later_stale_cache_converges`).
  Further members of this class are documented, not repaired.

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

Five pairs of separate processes, alternating between the G2 base (d4cf5011f) and L4. The
figures are the median CPU time (`process_time`) for `solve_linear` plus `compute_stresses`,
in seconds. Python 3.14 with native contractions, Windows. The machine was loaded (about 50% CPU),
so wall-clock time is not used.

| case | base solve | base recover | base total | L4 solve | L4 recover | L4 total | total | L4 faster |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| irregular plate 64x64 | 13.97 | 8.75 | 22.67 | 11.03 | 4.48 | 15.45 | 1.47x | 5/5 |
| plate 64x64 | 5.28 | 8.19 | 13.47 | 5.34 | 4.23 | 9.58 | 1.41x | 4/5 |
| pinched cylinder 32x32 | 1.25 | 2.06 | 3.30 | 1.27 | 1.14 | 2.41 | 1.37x | 5/5 |
| Scordelis-Lo 32x32 | 1.86 | 2.06 | 3.92 | 1.91 | 1.12 | 3.02 | 1.30x | 5/5 |
| stiffened plate 32x32 | 1.30 | 2.16 | 3.45 | 1.30 | 1.23 | 2.48 | 1.39x | 4/5 |
| mixed Q4/S3 plate 32x32 | 1.66 | 2.58 | 4.22 | 1.64 | 2.08 | 3.72 | 1.13x | 4/5 |
| mixed Q4/S3 Scordelis-Lo 32x32 | 2.83 | 2.55 | 5.38 | 2.83 | 2.05 | 4.88 | 1.10x | 4/5 |

- **Recovery** is 1.75-1.95x faster on pure Q4 meshes and 1.24x on mixed meshes, where only
  the Q4 half is batched.
- **Solve** is unchanged (0.98-1.01x) wherever the shared-geometry cache already forms only one
  element per geometry. It is 1.27x faster on the irregular plate, where every element has a
  distinct geometry.
- **16x16 meshes:** the total CPU time of six repetitions is 1.12-1.50x lower on every case.
- **8x8 meshes:** below both minimum batch sizes, so no batch runs. The only added work is the
  formation predictor and the recovery prepass's element count. Measured directly, that is about
  5 µs per solve, about 0.002% of a 280-445 ms solve. The A/B spread of up to ±2% on these cases
  is timing noise on the loaded machine.
- **Nonlinear runs:** the eight guard-cadence hash cases are bit-identical. The speed effect on
  nonlinear runs was not measured; earlier profiling put the batched numerics at about 6% of a
  nonlinear Q4 run.
- **Plate 64 solve:** the earlier unexplained gain does not reproduce on the G2 base (0.99x) and is
  not claimed.
