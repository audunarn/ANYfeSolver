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

CORRECTION (round 2): the round-1 enumeration claimed identity checks everywhere; that was wrong. `type(x) in (float, int)`,
`type(v) in (...)` in `_plain_mass_input`, `__mro__ !=` tuple comparison, the truth test of `concentrated_masses` and
the frozenset membership of a type in the V2D eligibility check all run a foreign metaclass `__eq__`/`__hash__`/`__bool__`.
All were found by grepping every `in (`, `==`, `!=`, truthiness, `len`, iteration and key lookup in the owned path and are
fixed below (round 2 commit). Rule now: classify a value or class ONLY by `type(x) is C` or `_type_is_one_of` (identity);
validate that a container is exactly `dict`/`list`/`tuple` before any truth test, `len`, iteration or lookup.

Enumeration (boundary.py kernel, helpers; matrix_assembly V2D capture; current_state_tangent V2D eligibility; static_batch):
* `_plain_mass_input`, `_plain_coupled_element_state`, `_two_node_element_takes_exact_zero_pressure`, kernel pressure/node/dof
  scalar classification, node mass classification, `type(type(element))`, warnings-filter `type(message/module/buffer)`:
  were `type(x) in (...)` (FIXED -> `_type_is_one_of`).
* `__mro__ != mro` at four class-authority sites (mass, routing, warning-context, numpy-route): tuple compare calls `__eq__` on
  differing members (FIXED -> `_identities_differ`, element-wise `is`).
* `mesh_state.get("concentrated_masses")` truth test (FIXED: exact `dict` then `dict.__len__`; other types refuse).
* `_v2d_trusted_immutable_value`: `kind in frozenset(types)` hashes/compares a foreign type (FIXED: identity scan).
* `_v2d_trusted_instance_eligible`: bookkeeping values were skipped unvalidated (FIXED: list of exact `weakref.ref` whose live
  referent is exactly `_V2DLeaseEpoch`; anything else declines the capture -> public route).
* Left as is, verified safe: `key in set/frozenset/tuple` where `key` is an exact `str` (hash/eq of exact str);
  `mass_class_names[type(element)]` (key is an exact package class already identity-checked); `dict.get/values/items/keys`
  on mappings proven exactly `dict` or `_QualifiedStateMapping` with the builtin `get`; iteration of exact `list`/`tuple`;
  `object.__getattribute__(x, "__dict__")`/`type.__getattribute__` (no instance/class hooks); `_callable_authority(_matches)` on
  class-dict snapshot members only (callers compare `is` first); `cell.cell_contents is not value`; ints compared with `!=`;
  `getattr(element, ...)` in `_qualified_s3_pressure_surface_records` only for elements the kernel proved exact package classes
  (all mesh elements when `v2d_authority`); `element_modules` V2D/V2C scope checks see only exact known classes.
* Pre-existing and not on the model/load-case value path: process-global numpy/warnings state read by
  `_numpy_error_routes_are_builtin` (identity checks now used for the type tests above).

## Round 2 findings
* P1 foreign values: tests `test_forged_pressure_scalar_is_refused_without_running_it`, `test_forged_container_truthiness_is_never_evaluated`,
  `test_forged_mass_input_values_are_refused`, `test_class_authority_compares_mro_by_identity` fail on 99820deff, pass now.
* P2 concurrent write: `V2D.__setattr__/__delattr__` counted BEFORE assigning; a capture taken while a writer sat between the two
  steps missed the write. Now the write completes first and is counted after: a capture either scans the new value or observes
  the late count and rejects (conservative). `test_capture_during_an_in_flight_write_never_misses_the_write` (writer thread paused by
  a trace hook at the end of the counting helper) fails on 99820deff. Hot path cost unchanged: 0.256 us per V2D attribute write
  before and after (200k-write loop, best of 5).
* P2 malformed bookkeeping (`[1.0]`): capture declines, public route runs (`test_malformed_lease_bookkeeping_declines_the_capture`).
  Not changed: an ordinary V2D write on an element carrying such a list still raises inside the pre-existing counting helper.
* P2 profile test: the per-record assertion is now independent of the scaling assertion (every record crosses
  `internal_input_guard` once); mutating the production code to skip the per-record guard makes it fail while the scaling assertion
  still passes.

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

## Round 2 measurements and tests
Barge (serial): prepare 77.0 s; pressure 1,404 V2D 4.07 s; gravity+masses 11.09 s; hashes unchanged
(`28641c0fe4510b17`/`f945cd604b2defed` vs old route, `316eda3ddf369689`, `8c704b393b5bc227` vs oracle).
Targeted 11 files: 247 passed. Extra V2D files run: 91 passed, 1 failed
(`test_e4_pl_s3_v2d_linear_native_parity.py::test_v2d_stateless_identity_roundtrip_and_remaining_successor_gaps` needs git object
`dfe3d314...` that this clone lacks; it fails identically on 99820deff).

## Round 3: write barrier (trust assumptions)
Chosen: count BEFORE and AFTER every V2D `__setattr__`/`__delattr__` (not a protected non-replaceable attribute: that would need a
new class-protected entry in the epoch machinery; two calls to the existing helper is the smaller change).
* Before: subscriptions are read while their storage still exists, so assigning, replacing or deleting
  `_qualified_v2d_lease_epochs` itself advances every existing counter first (P1 detach). It also means an existing lease already
  reads a nonzero counter before any assigned value can be observed (P2 paused writer).
* After: a capture that subscribed while a writer sat between the two steps still sees a count (round-2 case, test unchanged).
* Pause points covered deterministically (trace hook, existing lease): call/return of the first and second counting call;
  invariant: whenever the new value is observable the guard raises (`test_existing_lease_rejects_a_visible_write_at_every_pause_point`),
  detach by assign-empty/delete/assign-other (`test_detaching_the_subscriptions_cannot_bypass_invalidation`) and through
  `assemble_load_vector` (`test_detached_subscription_with_method_override_is_not_run_by_assembly`). 7 of the new tests fail on 80ad94d16.
* Not covered (documented module contract): `object.__setattr__`, direct `__dict__` writes and class-level `__setattr__` replacement
  bypass the counted surface; class replacement is covered by the V2D class epoch, instance `__dict__` bypass is not.
* Cost: 0.375 us per V2D attribute write (was 0.249). Writes to V2D elements are rare on the load path.
* `test_v2d_write_counts_are_atomic_under_contention` now expects exactly two counts per write (intent unchanged: no lost increment).
Barge (serial): prepare 76.5 s; pressure 1,404 V2D 4.09 s (one earlier 6.8 s run was machine noise, repeated at 4.09 s);
gravity+masses 10.99 s; hashes unchanged.

## Trust contract and known limits
Assumption (same as the `_capture_qualified_v2d_trusted_authority` docstring): while an assembly or capture runs, no foreign code
runs and no other thread mutates V2D elements.
* Covered: single-threaded replace/delete/set of `_qualified_v2d_lease_epochs`; a writer paused anywhere in the counted write while
  an existing lease observes (value visible implies counter already advanced); a capture taken during a write.
* Accepted limit (owner decision, no lock added): a second thread capturing between a writer's pre- and post-advance while that
  writer replaces or deletes the bookkeeping attribute can lose its subscription. Needs two threads mutating V2D elements during
  a capture, which the contract above excludes.

* Accepted limit (owner decision, round 4): deliberate tampering with the exposed bookkeeping state is outside the trust contract,
  in the same category as `object.__setattr__`, direct `__dict__` writes and `vars().update`: in-place `list.clear()` / slice
  deletion of `_qualified_v2d_lease_epochs`, resetting an exposed epoch's `value`, plus the bypasses listed under "Not covered"
  above. Such tampering needs the same control as foreign code. The storage was deliberately not restructured.

* Guard-shortcut contract (owner decision 2026-11-24, triangle speed fix, branch `claude/tri-shells-s2-20261124`): a
  constant-time or census shortcut must give the reference (complete-guard / element-by-element) answer in supported use.
  Rewriting this package's classes, `ABCMeta` or the element mapping class at run time, custom element metaclasses and
  `__class__` reassignment are outside supported use ("no monkeypatching, no mid-request surface changes"). The shortcuts
  in `_require_v2d_mixed_shell_scope` (type census restricted to package-owned classes whose MRO, `__getattribute__`,
  `__getattr__`, `__class__` and metaclass match an import-time snapshot, classes compared by identity),
  `_qualified_s3_pressure_surface_records` (exact V2D fields read without running foreign code; dropped for the remaining
  records once any record takes the guarded route) and `_build_trusted_v2d_recovery_require` (lease trusted-state check
  plus captured V2D authority plus provider identity; a complete guard after converting non-plain recovery inputs) make
  the common forms of such rewriting fall back to the reference path, but are not an exhaustive defence against them.
  Addition (owner trust boundary, 2026-11-26, recovery-guard follow-up): writes into a package object's instance namespace
  that bypass attribute assignment (`object.__dict__` / `vars()` edits, non-str keys) are outside supported use, like
  run-time class rewriting; the shortcuts make their common forms fall back but are not an exhaustive defence. Hostile
  caller DATA stays in scope: objects passed as arguments (element ids, flags, displacement arrays and their dtypes,
  including custom metaclasses on those argument objects) and caller element subclasses with ordinary metaclasses must
  never get a different answer or skip a refusal. Custom metaclasses on element classes placed in the model remain
  outside supported use (contract above).

## static_batch enumeration (round 4; all now identity-classified)
Inside the core trusted span: none (the span is the kernel, the V2D guard and the load-vector assembly). Upstream/outside the span but
on the load-input and preparation-witness path: all sites below.
* `_snapshot_plain_load.clone` (`type(v) in (int, float, ...)`, `in (list, tuple)`): FIXED to identity; a forged scalar/container makes
  the snapshot refuse. Ordinary loads clone exactly as before (`test_snapshot_of_ordinary_loads_is_unchanged`).
* `_plain_state` (`in (str, int, ...)`, `in (list, tuple)`): FIXED to identity; forged objects fall through to the opaque
  `(type, id, namespace)` form, never returned as plain.
* `_plain_state` `getattr(value, "__dict__", None)`: FIXED -> `_namespace_of` (static MRO inspection; only the interpreter's
  `__dict__` slot descriptor is followed, a foreign property/`__getattr__` is not run; such objects get namespace `None`).
* `_class_state` `type(value) in (staticmethod, classmethod)`: FIXED to identity.
* `_model_state` `dict.fromkeys(type(e) ...)` (hash/eq of foreign classes): FIXED (identity de-duplication).
* `_model_state` element namespace rebuild `{k: v ...}` (hashes keys) and `k not in derived`: FIXED (non-str keys stay in the witness,
  never hashed; the namespace dict is rebuilt only when every key is an exact str).
* Unchanged, verified: `revision_signature().items()` and `k not in {"load", "result_state"}` (package-owned str keys);
  `_callable_state` (`type(function) is`, ids only); `node.x/y/z` and `tuple(node.dofs)` on mesh nodes (reads on package `Node`s
  whose class authority the lease already proves for the load path; witness-only elsewhere).
Tests: `tests/test_static_batch_identity_classification.py` (5 of 6 fail on 89086dd7e).

## Round 4 measurements
Tests: 22 files (11 targeted + kernel + static_batch + V2D families): 436 passed, 1 skipped (pypardiso), 1 failed (the missing-git-object
test above, identical on 99820deff). Barge was machine-load dependent this round: this commit prepare 104.8/106.8 s, pressure 1,404
5.8-5.9 s, gravity 14.7-15.2 s; the PREVIOUS commit 89086dd7e measured back to back in the same load: 126.3 s, 6.97 s, 18.15 s
(so no regression from round 4; quiet-machine numbers were 76.5 s, 4.09 s, 10.99 s in round 3). Hashes unchanged
(`28641c0fe4510b17`/`f945cd604b2defed`, `316eda3ddf369689`, `8c704b393b5bc227`). The double advance cost 0.126 us per V2D write.

## Round 5: metaclass descriptors in static_batch
Every inspection of an unvalidated object's class in `static_batch.py` now reads the interpreter slots
(`type.__dict__["__mro__"].__get__`, `type.__dict__["__dict__"].__get__`) instead of `type.__getattribute__`/`cls.__mro__`/`vars(cls)`,
which run a metaclass property:
* `_namespace_of` (MRO walk and per-base `__dict__` lookup): FIXED (`_class_mro`, `_class_namespace`); result must be an exact `dict`.
* `_class_state` (`cls.__mro__`, `vars(base)`): FIXED (same helpers).
* `_model_state` element namespace (`vars(element)` ran an element's `__dict__` property): FIXED (`_namespace_of`; unreadable -> opaque
  `(type, id)` entry).
* Unchanged, verified: `type(x)` one-argument calls (run no code); `getattr(mesh, ...)` on the model's own `FEMesh`;
  `getattr(iterator, "close")` (the caller's own iterator, deliberately invoked, not a classification).
Tests: `test_metaclass_mro_and_dict_properties_are_never_run` (goes through `_plain_state`, `_snapshot_plain_load`, `_class_state`,
`_model_state`; fails on d27928b93), `test_ordinary_class_inspection_is_unchanged`.
