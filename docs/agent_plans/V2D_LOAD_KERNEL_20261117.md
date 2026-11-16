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

## DECISION NEEDED (owner)
`tests/test_load_vector_guard_amortization.py::test_generic_element_pressure_keeps_the_complete_closing_guard`
pins that a pressure entry on a plain `BeamElement` is refused by the kernel. The barge's prepare pressure probe
puts pressure entries on all 7,792 elements (339 beams, 54 couplings), so without the zero-pressure admission
the kernel is refused for the probe and prepare keeps the per-V2D-record complete guard. With the admission that
test fails (policy change, not weakened by me; test left as is). Options: (A) accept admission and update that
test to the new documented behaviour; (B) drop the admission, prepare stays slow; (C) make the probe skip
elements whose pressure map is exactly zero (touches prepared_static_loads).
Also failing at base, unrelated: `test_e4_pl_s3_v6n_stage4a_profile.py::test_pressure_surface_traversal_uses_captured_constant_time_guard`
(source-text test, broken since e593afe02).
