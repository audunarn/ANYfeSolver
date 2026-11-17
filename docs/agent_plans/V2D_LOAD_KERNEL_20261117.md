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
* Full default `prepare_linear_static_evaluator`: 76.3 s (82.2 s after review round 1) (old 2,192 s; batching branch 125-131 s).
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

## Review round 1 (Codex P1) and closure by enumeration
P1: an unloaded generic element in `mesh.elements` stayed eligible; the V2D scope check
(`e4_pl_s3_v2d_element._validate_model_scope`: `isinstance(element, ShellElement)`) consults an instance
`__class__` and ran a foreign property mid-span. Reproduced (`test_unloaded_generic_element_with_class_property_is_refused[pressure]`
fails on 715818f8a). Class fix: with `v2d_authority`, EVERY mesh element must be an exact known class
(`type(e) is cls` for Q4/S3/V2D/Beam/Coupled/MPC), and the beam-family class snapshot must hold; otherwise the
public route runs. Acceleration already required exact mass kinds for all elements.

Sites on the new path and verdicts (V2D = traversal run when `v2d_authority`):
* boundary kernel: `_type_is_one_of` (identity); all-element exact-class loop; `dict.get(elements, int)` (keys
  verified `type(k) is int`, mapping reads builtin); `object.__getattribute__(e, "__dict__")` (no instance
  `__getattr__`); `_plain_mass_input`/`_plain_numeric_array` (`type(x) is`); `key in class_names`/`in _NATIVE_PRESSURE_HOOKS`
  (str keys, verified). `mass_class_names[type(element)]`: dict keyed by exact classes already identity-verified. Safe.
* `matrix_assembly._capture_v2d_load_guard`: container type was `in (dict, X)` (metaclass `__eq__`); now identity. `type(e) is V2D`;
  `dict.items/get` on verified containers. Safe.
* `_qualified_s3_pressure_surface_records`: `getattr(element, formulation_id/...)` only for elements in the pressure
  set, which the kernel proved exact owned/beam classes; `mesh.get_element` is class-snapshotted. Safe.
* element modules (unchanged): V2D `_validate_model_scope` (`isinstance`, `type(e) not in allowed`) and V2C scope
  (`type(e) not in allowed_types`, `list(elements.items())`) run over all mesh elements; with only exact known classes
  in the mesh no foreign `__class__`/`__eq__` exists to run. V2C-only meshes are not admitted (refused -> public route).
* Not changed, pre-existing: Q4/S3-only kernel admission still tolerates unloaded generic elements (no
  mesh-wide traversal in the Q4/S3 pressure path; outside this change).

## Review items not reported by Codex (what was checked)
* Capture returning None / `__slots__` / descriptors: `_v2d_trusted_instance_eligible` reads `object.__getattribute__(e,"__dict__")`
  (AttributeError -> ineligible) and accepts only immutable values, so an instance cannot carry a descriptor/callable;
  class-level descriptors are caught by the class-epoch/representative guard. Ineligible instance -> `None` -> public route
  (`test_ineligible_v2d_instance_keeps_the_public_route`).
* Zero-pressure admission: subclass, metaclass `__eq__`, `__class__` spoof, hostile/non-plain state, wrong node-id types
  tested; bad entries (NaN/inf, negative, unknown/negative ids, tuple on beam) compared with the public route
  (`test_bad_pressure_entries_on_beams_match_the_public_route`): same vector or same exception type.
* Numerics/error precedence: the owned route runs the identical core arithmetic; errors raised inside the core propagate
  from the same code in both routes; guards differ only in being constant-time trusted checks.
* `_model_state` masking: now masks `_qualified_v2d_lease_epochs` ONLY when it is a list of `weakref.ref`; a real V2D
  mutation still changes the witness and any other value under that name is witnessed
  (`test_witness_masks_only_weakref_bookkeeping`). The list holds no model state (consumed only by the epoch counter).
* Weakref bookkeeping: each capture prunes dead refs under the module lock before appending; 25 assemblies leave <= 2 live
  entries (`test_lease_epoch_subscriptions_stay_bounded`).
