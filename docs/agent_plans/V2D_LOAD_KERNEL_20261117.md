# Owned load kernel for S3 V2D pressure and acceleration (2026-11-17)

Replaces the guard-batching approach (PREP_COST_20261116). No batching code is used.

## What changed
* `boundary._run_foreign_code_free_load_vector(..., v2d_authority=False)`: exact
  `NativeParityE4PLS3V2DShellElement` is admitted (pressure, element loads, acceleration) only when the
  caller attests that `stage_guard` also re-validates V2D authority. `_builtin_acceleration_inputs` admits V2D
  (same instance checks as the other mass kinds) and the zero-mass `InterpolatedBeamShellMPCElement`
  (added to `_EXACT_MASS_CLASSES`, same snapshot pattern as `CoupledBeamShellElement`).
* Pressure entries on exact two-node `BeamElement`/`CoupledBeamShellElement` are admitted by
  `_two_node_element_takes_exact_zero_pressure` (see decision below).
* `matrix_assembly._capture_v2d_load_guard` captures the EXISTING
  `current_state_tangent._capture_qualified_v2d_trusted_authority` (nonlinear statics' V2D guard) once per
  assembly; the lease's trusted state guard is wrapped so every stage runs lease-trusted + V2D check.
  `_qualified_s3_pressure_surface_records(v2d_guard=...)` runs the same check next to the trusted input guard.
* `static_batch._model_state` ignores `_qualified_v2d_lease_epochs` (weakref bookkeeping the capture appends
  to V2D elements); without it prepare fails "model changed during prepared static snapshot capture".

## Trust assumptions (what each new branch reuses)
1. V2D: complete lifecycle guard over all V2D elements once (capture), then per stage: V2D class epoch unchanged,
   V2D write counter unchanged, formulation-level guard on one representative. Instances must hold only
   immutable values (`_v2d_trusted_instance_eligible`); otherwise capture returns None and the old public route runs.
2. The kernel never runs V2D code without `v2d_authority`; a subclass/forged class is refused by exact type.
3. Beam/coupling/MPC: `_mass_classes_are_untouched()` class-dict snapshots + plain instance state, as the
   existing acceleration route. Beam/coupling pressure: fallback returns exactly zero for 2-node elements.
4. Model-mapping/mesh reads: unchanged proofs (`_load_model_access_is_builtin`, builtin mapping reads).

## Decision (owner, option A) and test updates
* Exact `BeamElement`/`CoupledBeamShellElement` pressure entries are admitted with exactly zero contribution.
  `test_generic_element_pressure_keeps_the_complete_closing_guard` now asserts both halves: exact beam admitted
  (zero, bit-identical to the public route) and a generic (subclassed) element still refused with the complete
  closing guard (>= 3 complete lifecycle guards).
* Hardening found by the adversarial test: type membership in the kernel used `in` over tuples, which calls a
  forged metaclass `__eq__`; all kernel class checks (loaded types, acceleration kinds, zero-pressure kinds) are now
  identity (`_type_is_one_of`). Test `test_metaclass_equality_and_class_spoof_are_refused` fails without it.
* `test_e4_pl_s3_v6n_stage4a_profile.py::test_pressure_surface_traversal_uses_captured_constant_time_guard` was a
  source-text assertion broken since e593afe02. Rewritten behaviourally (original intent kept in a comment): the
  number of complete model-wide lifecycle scans is independent of the S3 V2D pressure-record count (6 vs 24
  records). Not weakened; it fails on per-record complete guards.

## Measurements (serial, one process, Python 3.13 CI venv, ANYfem bracketed_barge_fatigue, mesh build ~46 s excluded)
* Full default `prepare_linear_static_evaluator`: 76.3 s (old 2,192 s; batching branch 125-131 s).
* Pressure assembly, 1,404 V2D: 4.0 s (old ~2,100 s extrapolated; 50 elements old 56.8 s here, new 1.9 s).
* Gravity + four node masses, full barge: 10.9 s (old > 30 min, stopped).
* Hashes (sha256 of vector bytes, first 16 hex): 50-element pressure probe (first 50 V2D ids, p = 1000):
  old route (unmodified HEAD) vector `28641c0fe4510b17`, new `28641c0fe4510b17`; records `f945cd604b2defed` both.
  The coordinator's `9c2f69cab8210b2d`/`b7832eb6d5b8f2ea` could not be reproduced (selection/hash method unknown;
  searched sorted/dict-order/last 50, several pressures, sha256/md5/sha1/blake2b: no match).
* 1,404-element pressure vector `316eda3ddf369689` and gravity+masses vector `8c704b393b5bc227` equal an
  arithmetic oracle (`LoadCase._get_load_vector_core` with a no-op guard, the old route's arithmetic); the old
  route cannot finish on the barge. On the scaled mixed Q4+V2D+beam+coupling fixture the new route equals the
  public route byte for byte (pressure, gravity+masses, both).
