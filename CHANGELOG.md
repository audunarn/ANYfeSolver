# Changelog

## Unreleased

## 0.5.0 - release candidate

- Require ANYfileio 0.3.3 or later in the 0.3 line for SESTRA shell stress
  recovery from relabelled descriptors and reported rotation frames.
- Rename the solver distribution and import to `ANYfeSolver` and `anyfesolver`.
  The separately installable results runtime is `ANYfeSolver-results` 2.2.0,
  imported as `anyfesolver_results`. See [the migration guide](MIGRATION.md)
  for consumer upgrades, supported historical archives and restart files.

- Remove 22,552 historical development/qualification files from the checkout
  after verified external archival; retain regression inputs, authority records,
  source, tests and tools. See `docs/DEVELOPMENT_ARCHIVE.md` for recovery.
- Reconcile the documentation entry points and local development status after
  the October 27–28 integrations. Retain dated logs and qualification records
  with their original scope; see `docs/STATUS.md`.
- Add linear inertia relief and prepared static-batch integration for the
  admitted free-free beam/shell domain; see `docs/INERTIA_RELIEF.md`.
- Add neutral-input floating-body frequency response, fixed-step transient
  wind-thrust feedback and physical pressure-region loading. These remain
  bounded development routes; see `docs/FLOATING_RESPONSE.md`.
- Add concentrated rigid mass with centre-of-gravity offset and inertia,
  including linear mass/gravity and inertia-relief integration; see
  `docs/concentrated_mass.md`.

- Qualified Q4 plastic shells evaluate layered plasticity at the hybrid assumed
  strain (`_q4_assumed_strain_plasticity`). The historical compatible-strain
  route with the constant qualified correction had an indefinite tangent once
  layers yielded and converged to spurious mechanisms on a clamped plastic plate
  (+203 mm at n = 8, wrong sign at n = 16). The new route has an exact symmetric
  tangent and completes at n = 4..32, within 2.2 % of the legacy element at
  n >= 8 and moving toward CalculiX (-0.7 % at n = 32). Elastic materials without
  a hardening curve are bit-identical in linear analyses.
  Hill48/orthotropic plasticity (with or without a curve), generalized sections
  with hardening, warped facets and active initial fields are refused with
  `UnsupportedQ4PlasticRouteError` instead of the unstable form. Plastic
  `layer_strain`/`layer_stress` now hold assumed-strain values. See
  `docs/Q4_ASSUMED_STRAIN_PLASTICITY_20261023.md`.
  Accepted exceptions (owner approval 2026-10-23):
  - Before yield, plastic-capable elements change through the von-Karman cross
    terms: 1e-10..2e-9 relative for small rotations, and up to about 1e-3 in the
    centre deflection (about 5e-3 in-plane max |du|) on large-deflection thin
    plates. There is no growth under refinement, and the values move slightly
    toward the elastic CalculiX answer.
  - Single elastic law: qualified Q4 elements without a hardening curve run the
    same kernel (return map skipped), so von Karman and corotational nonlinear
    elastic results of planar isotropic qualified Q4 change. Owner waived
    bit-identity. C1 elastic n = 8/16: von Karman -9.2e-4 / -9.6e-5 (toward
    CalculiX), corotational +6.7e-4 / +1.5e-4. Mixed curve/no-curve meshes now
    equal the pure meshes to 1e-9. The compensated direct elastic route and the
    batch-B elastic fast path are retired for these elements. Linear analyses and
    non-qualified-Q4 elements are bit-identical.
  Known limitations:
  - Historical compatible-law plastic histories restored through restarts, seeds
    or a mixed persistent store are not detected. Codex rates this a
    high-severity restart correctness risk; a law-version tag is the proper fix.
    R4's own unsealed restart states also reach the migration point, so a refusal
    there was withdrawn. Open HIGH follow-up.
  Qualification: TARGET 5c34dc2e4, 68/68 PASS with exceptions W1 (2.5 % n8 in-plane allowance), C4 (legacy
  rule), A8 (inventory), H1 (historical-state limitation); A9 2.0x legacy / 1.25x base. Open follow-ups:
  public stress recovery review (option A accepted), plastic-law version tag (HIGH), governed manifest
  successor record, descriptor-shear gate follow-up for S3 impact. Merged to local main 2026-10-23.

- Refresh the private Windows Python 3.12 beam-shell bundle from `6fb907a1c`,
  including both completed-prefix S355 checkpoint workflows. Verify 77 installed
  tests, two pre-yield and six fresh-process plastic full/prefix/resume CLIs;
  check all 727 installed solver files and wheel/sdist contents against the
  committed source. Preserve the earlier artifact under its original hash.
  The private version remains `0.4.7+beam.shell.candidate.20261020.s3`; explicit
  local wheel paths and bundle hashes distinguish the builds. Repository release
  version, scientific gates and publication status remain unchanged.

- Deliver the source S355 beam-shell completed-prefix checkpoint workflow for
  direct and interpolated default coupling: 11/6/5 initial increments to
  -11/60 m via -0.10 m, with unchanged material, iteration controls and tolerances.
  Bind corotational Q4 checkpoint records at their constitutive reference
  deformation in emission, validation and restart preflight. Retain existing
  schema, integrity checks, failed-terminal refusal and `.converged` semantics.
  Cancellation and rejected trials preserve caller inputs and recovery files.
  The previously pinned installed artifact remains pre-yield only.
  Twelve checkpoint and 113 shared/compatibility tests pass. The four remaining
  default regressions also pass, using a retained completed comparison fixture
  with fresh production prefixes/resumes and unchanged assertions. The earlier
  timed-out selection remains preserved; no full-suite or scientific-gate claim.

- Deliver a private Windows Python 3.12 beam-shell pre-yield candidate, version
  `0.4.7+beam.shell.candidate.20261020.s3`, with wheel, sdist, pinned dependencies
  and recorded installed verification (65 focused tests and both CLI modes).
  That artifact omits serialized plastic checkpoints, which were blocked at its
  source identity; the failed candidate remains preserved. The newer source
  workflow above is not included in that unchanged artifact.
  Director continuation and analytical accuracy remain open.

- Add the standalone public example `examples/beam_shell_coupling.py` for the
  registered pre-yield S355 beam-shell case with direct or interpolated
  (default) rotating coupling. Its CLI prints a truthful JSON summary
  (`status`, `stop_reason`, `requested_load_factor`, `achieved_load_factor`,
  `probe_displacement`, `accepted_increments`, `constraint_postcheck`,
  `completed`) and exits 0 only when the solve actually completed to the
  requested load factor within 1e-8 with passing existing constraints; a
  cancelled solve reports its accepted increments without claiming success.
  New focused integration tests cover both couplings, an independent
  true-elastic control, cancellation with empty and non-empty accepted
  history, and the CLI JSON/exit-status contract. No solver default,
  tolerance or material change.
- Correct the beam-shell connection guide's stale completion claim: the
  default rotating route has a demonstrated completed S355 displacement-control
  run to the revised -11/60 m endpoint including the public initial-state
  restart, while the original -0.20 m target, the director-integrated trace
  (stopped at -1/6 m) and the analytical accuracy gate remain open. The guide
  now also explains reading achieved displacements from
  `result.displacements`, `steps[-1].control_value` under displacement control,
  and that `result.converged` alone is not completion. No default, tolerance
  or material change.
- Speed up nonlinear static solves of models that combine qualified Q4 shells
  with exact S3 V2D triangles. The solver's own cancellation token and the V2D
  elements now use the constant-time trusted lifecycle guard inside the Newton
  loop, and every Numba kernel is cached on disk by default (`cache=True` is
  applied by `anysolver.jit_compiler`; pass `cache=False` to opt out).
  Displacements, load factors and step histories are bit-identical to 0.4.7.
  On the representative plate/cylinder case a warm solve drops from 22.5 s to
  11.7 s and the first solve in a fresh process from 38.5 s to 12.0 s.
- A nonlinear static solve now stops with an authority error when an S3 V2D
  element it owns is modified (any ordinary attribute write or deletion) while
  it runs. Writes before or after a solve behave as before.
- A cancellation token is trusted only if it is an unmodified
  `CancellationToken` both before and after its checkpoint code runs; every
  other token keeps the complete lifecycle scan.
- Report which guard path a nonlinear static solve used through the
  `qualified_guard_trusted`, `qualified_guard_complete_scan`,
  `qualified_guard_untrusted_token` and `qualified_v2d_trusted_unavailable`
  events in `info["nonlinear_performance"]["solver"]["event_counts"]`.
- Speed up the exact qualified lifecycle guard for Q4 and S3 shells: class-level
  facts and one element-independent serialization module guard are read once per
  validation call instead of once per element. One complete scan drops from about
  70 to about 33 microseconds per shell; a warm nonlinear solve of a stiffened
  synthetic model is about 20 % faster and the GUI-style plate/cylinder run about
  8 %. Instance-level checks, the element-specific validators and S3 V2D elements
  are unchanged. Semantics: a change to a base class made after the first element
  of a class was checked is reported by the next call rather than by the remaining
  elements of the same call (a change to the Q4 or S3 class itself is still
  reported by them).
- Keep qualified element equations, physical tolerances and defaults
  unchanged.

## 0.4.7 - 2026-09-21

- Speed up representative static analysis and retained-load solves through
  qualified preparation and reuse; the recorded complete-route medians improve
  3.98x and 19.85x respectively on the declared static workload.
- Reduce repeated spectral validation work while preserving cancellation and
  mutation guards. Retained medium-panel buckling improves 11.52% in the
  seven-pair gate, with identical modes and residuals.
- Improve nonlinear static work accounting, accepted-force reaction recovery,
  fixed dead-load projection, and trial-tangent promotion checks. The frozen
  representative nonlinear gate records a 12.44% median improvement across
  non-easy cases with passing physical comparisons.
- Provide an opt-in Armijo residual line search for supported force-controlled
  beam/shell analyses. Its separate convergence promotion gate was NO-GO: it
  did not add difficult solves or reduce failed work, so no convergence benefit
  is claimed; existing defaults remain unchanged.
- Preserve runtime constraint and nonlinear failure details in the generated-
  geometry facade when forwarding locally edited models.
- Keep qualified element equations, physical tolerances, defaults, and
  coordinated ecosystem dependency bounds unchanged.

## 0.4.6 - 2026-09-16

- Accept the independently published ANYmesher 0.5 line through the exact
  dependency bounds `ANYmesher>=0.4,<0.6` and `ANYfileio>=0.3.2,<0.4`.
- Bind the compatibility release to production ANYmaterial 0.2.0,
  ANYgeometry 0.4.3, ANYmesher 0.5.0, and ANYfileio 0.3.2 artifacts.
- Preserve all solver mechanics, B3-GE opt-in behavior, legacy B3 default,
  and qualified Q4/S3 defaults unchanged.

## 0.4.5 - 2026-09-16

- Publish the bounded B3-GE opt-in release after correcting the isolated
  installed-wheel gate to install the wheel's declared runtime dependencies.
- Preserve the failed v0.4.4 GitHub publication attempt as process evidence;
  no v0.4.4 artifacts reached PyPI.
- Keep legacy B3 as the default and leave qualified Q4/S3 mechanics and
  defaults unchanged.

## 0.4.4 - 2026-09-16

- Publish the accepted `B3-GE` model-owned geometrically exact beam workflow
  as an explicit `b3-ge` opt-in. Legacy B3 remains the default for every
  existing three-node beam route and historical record.
- Bind the minor release to the accepted G7 production-opt-in evidence and a
  bounded installed-wheel gate. Qualified Q4/S3 mechanics and defaults are
  unchanged.
- Remove unreferenced development notes from the active tree and exclude
  research plans, raw evidence, tests, and build scripts from release source
  archives. The complete pre-cleanup tree remains available at the archive tag
  `archive/anysolver-pre-0.4.4-development-20260916`.

## 0.4.3 - 2026-09-09

- Add the explicitly selected `ge-beam3` straight mixed beam and the reviewed
  `ge-beam3-native` model-owned straight/regular-curved beam workflows, with
  declared generalized and physical-fibre sections, authenticated history,
  recovery, restart, and admitted reference/current-rest spectral analysis.
- Add the reviewed objective eccentric generalized-beam connection to one
  owned elastic qualified Q4 or S3 V2D through the native workflow API.
- Preserve the full development/qualification history and its scoped
  independent acceptance. This is not qualification of generic mixed FEModel
  routing, plastic/fibre-shell coupling, finite-velocity rotational dynamics,
  arbitrary sections, or stable postbuckling paths from rest.
- Keep existing B2/B3, qualified Q4/S3 mechanics and all defaults unchanged.
  See `docs/GE_BEAM3_NATIVE_WORKFLOWS.md` for the supported API and limitations.
- Separate the explicit current GE runtime CI inventory from preserved extended
  experiments. Historical evidence and the unpublished qualification wheel
  remain immutable; the versioned release wheel is verified separately.

## 0.4.2 - 2026-09-03

- Canonicalize solver-owned compact qualified-Q4 plastic state once at the
  accepted Newton finalization boundary. Public sealing remains fail-closed,
  while restart and subsequent current-tangent replay become byte-exact.
- Preserve qualified Q4/S3 mechanics, coefficients, recovery, formulation
  identities, and defaults.

## 0.4.1 - 2026-09-03

- Correct the coordinated ecosystem dependency bands to require ANYmaterial
  0.2.x, ANYmesher 0.4.x, and ANYfileio 0.3.1 or newer. Version 0.4.0 carried the
  pre-activation bands and cannot be resolved together with ANYfem 0.4.0.
- Add a release gate that resolves the built wheel with the exact coordinated
  sibling releases before trusted publication.
- Admit only a fixed binary64 last-place envelope when scalar final-state
  replay validates compact qualified-Q4 plastic states produced by vectorized
  assembly. This fixes sealing after valid plastic increments while preserving
  exact origin/configuration hashes, exact rich-state replay, and strict
  rejection of material history differences. Qualified Q4/S3 mechanics and
  formulation defaults are unchanged.

## 0.4.0 - 2026-09-03

- Relicense ANYsolver source releases from GPL-3.0-or-later to MPL-2.0.
  Earlier published versions retain the licence terms that applied to them.
- Publish a machine-readable direct-dependency licence inventory and preserve
  third-party notices in source and wheel distributions. The extracted ANY
  dependencies remain separately licensed and are not bundled or relicensed
  by ANYsolver.
- Add release-time checks for repository metadata, notices, dependency
  inventory, and licence files in built distributions.
- Document the qualified-S3 admission boundary for automatically imprinted
  four-way shell intersections. These junctions continue to fail closed rather
  than bypassing qualification or silently selecting legacy S3.
- Activate the V6W-qualified native-parity S3 V2D companion as the default
  three-node shell for new/current-policy models. Historical records without
  formulation authority remain explicit `legacy-s3`, direct `ShellElement`
  construction remains legacy, and qualified S3 creation fails closed without
  an authoritative physical owner normal. Qualified Q4 mechanics are unchanged.
- Mark the coordinated breaking activation boundary as ANYsolver 0.4.0 and
  admit the compatible ANYmesher 0.3 line while retaining the strict 0.4 cap.

- Add the explicitly selected `e4-pl-s3`/`qualified-s3` linear MITC3+
  companion candidate with frozen source identity, two bubble rotations,
  basis-invariant three-mode PL completion, stable serialization identity,
  authoritative-normal admission, formulation-native local recovery, and
  analytic nodal-plus-bubble mass with the frozen Guyan reduction. The modal
  solver excludes exact zero-inertia drill coordinates without inventing mass,
  using coordinate-invariant static condensation for bounded systems and a
  fail-closed swapped descriptor pencil for larger systems. The candidate now
  also supplies formulation-native total-Lagrangian layered response,
  stateless generalized-section geometry with analytically integrated initial
  stress resultants and generalized eigenstrains, contact virtual work,
  committed recovery without fabricated layer data, physical-director
  reversal, current-state modal/buckling operators, and solver-owned
  static/arc-length checkpoints. The original candidate remained opt-in until
  the independent V6W qualification and this successor activation boundary.
- Keep qualified E4-PL as the production Q4 default and begin the explicit
  legacy-Q4 retirement window. Legacy Q4 selectors now emit a dedicated
  deprecation warning, diagnostics expose the selected formulation and
  rollback lifecycle, and migration guidance freezes availability through
  0.4.x with removal no earlier than 0.5.0 after two clean release gates.
  TRI3, TRI6, Q8, and Q8R remain unchanged.

## 0.3.0 - 2026-08-23

- Preserve the corrected published-2025 S4 reference, nullspace proof, geometry
  handoff, and qualification scaffold as dormant release evidence. The legacy
  S4 remains the production default and improved-S4 activation remains
  unavailable under the recorded restrictions; no gauge constraint, invented
  stiffness, dispatch route, or activity-policy change is introduced.
- Widen the declared ANYfileio consumer range to `>=0.1,<0.3`, add pinned
  source compatibility cells for the 0.1 and 0.2 public contracts, and align
  the publication dependency gate. Installed-wheel, index-resolver, and
  publication qualification remain separate release gates.
- Widen the declared ANYmesher consumer range to `>=0.1,<0.3`, add CI coverage
  for pinned 0.1.0 and 0.2.1 endpoints, preserve the public neutral panel, mesh,
  quality, and section contract, retain the strict 0.3 cap, and align CI and
  publication dependency gates with the declared metadata.
- Complete the Sol Ultra `performance_2` campaign with qualified static/arc
  persistent nonlinear state transactions, compiled canonical-curve Hill-48 batches,
  orthotropic/generalized S4 kernels, revision-cached corotational block
  transforms, large-selection isotropic S4 recovery, and direct reduced
  elastic nonlinear-impact assembly. Unsupported formulations, curves, state
  layouts, impact plastic/fiber history, small recovery selections, affine
  constraints, and damage/state scopes retain explicit scalar or
  full-coordinate fallbacks with reason diagnostics.
- Add conservative opt-in impact tangent/factorization reuse, compact lazy
  contact records, and revision-guarded incremental damage K/M updates. A zero
  reuse budget remains the full-Newton oracle; contact, convergence, plastic,
  damage, deletion, time-step, and line-search changes force a refresh. Damage
  plan setup is gated by projected future events and retained-memory headroom;
  invalidation returns to the exact rebuild and never scales point masses.
- Add the optional caller-owned `AnalysisSession` for bounded K/M/constraint,
  reduced-matrix, output-row, and factorization reuse across repeated linear,
  modal, buckling, transient, and capacity work. Linear transient loads are
  preprojected once and selected history output avoids full-vector
  reconstruction when recovery does not require it. Explicit solver,
  assembly, and recovery thread scopes remain independently controlled and are
  restored after each call.
- Release 0.2.0 extraction boundary: material behavior now comes from
  `ANYmaterial`, neutral geometry/meshing from `ANYmesher`, and SESAM/CalculiX
  parsing and writing from `ANYfileio`. ANYsolver retains solver-specific
  FEModel adapters, MPC elements, boundary interpretation, external execution,
  comparisons, and the 0.2.x legacy import facades. Deterministic FE baselines
  remain unchanged.
- Add dependency-free generalized section contracts: pre-integrated shell
  `A/B/D/As` with membrane-bending coupling and optional areal inertia, plus a
  coupled beam 6x6 stiffness and optional mass-per-length law. Generated
  geometry accepts inline or named definitions; recovery exposes exact
  generalized strains/resultants, nonlinear shell acceleration falls back
  deterministically, and unsupported CalculiX/fiber/plastic mappings fail
  closed.
- Add a dependency-free structural material contract and homogeneous
  orthotropic engineering materials for shells and beams. This includes
  projected shell material directions/angles, explicit orthotropic beam
  torsional rigidity, generated-geometry and CalculiX shell interchange, and
  deterministic fallback from isotropic accelerated kernels to general
  assembly.
- Add material-axis Hill-48 shell plasticity with directional-strength
  hardening, consistent tangent plus numerical oracle/fallback, stored physical
  stress recovery, labelled Hill equivalent stress/utilization, orthotropic
  beam-fiber longitudinal yielding, and damage/contact integration. General
  anisotropy, laminates, ply failure, and shear/torsion plastic interaction
  remain outside scope.
- Add an exact constant-stress analytical/CalculiX orthotropic S4 reference
  deck, perfect-plastic Hill behavior when no hardening curve is supplied, and
  current-strength Hill utilization with physical-stress recovery provenance.

## 0.1.3 - 2026-07-28

- Complete the generated-geometry runtime bridge: apply configured shear force
  and torsional moment, pass current-area follower pressure only to supported
  nonlinear static/arc-length paths, retain the selected corotational
  kinematics in arc length, and fail closed for incompatible follower paths.
  Runtime prestress recovery now consumes matching committed nonlinear
  histories and exposes Gauss-point membrane/bending resultants plus recovery
  provenance.
- Correct nonlinear display recovery so plastic-history von Mises values use
  only the return-mapped shell in-plane or beam-fiber components, while purely
  elastic nonlinear states retain their full equivalent stress. Preserve
  combinations with elastically reconstructed transverse shear/torsion as a
  separately labelled model-scope diagnostic.
- Add the public `resolve_runtime_analysis()` contract so application GUIs can
  reflect normalized nonlinear/material/control/kinematics selections without
  importing private runtime helpers.

## 0.1.2 - 2026-07-28

- Add opt-in executable CalculiX reference validation with isolated case runs,
  stale-output protection, timeouts, solver version/hash provenance, ASCII
  FRD/DAT parsing, tolerance-controlled comparisons, and preserved evidence
  semantics. Deck-only generation remains explicitly `not_executed`; stale
  non-executed legacy reports are regenerated deterministically, while invalid
  executed evidence is preserved and rejected for diagnosis.
- Add current-area follower pressure and its exact, generally nonsymmetric
  external-load tangent to nonlinear static and arc-length solves. Extend
  shell initial-stress stiffness to the Mindlin translation/director field
  using membrane, bending, and second stress moments, and add the qualified
  thin-ring pressure-buckling gate.
- Add an opt-in consistent corotational tangent through the full
  pull-back/frame/rotate-forward chain rule. `auto` retains the lower-cost
  rotated tangent for ordinary loads and selects the consistent tangent for
  follower pressure.
- Unify nonlinear stress recovery around committed shell-layer and beam-fiber
  histories, matching solution displacements, objective corotational frames,
  component-level provenance, and explicit elastic fallbacks. Add guarded
  full-integration Q4/Q8 patch recovery that separates discontinuities and
  labels its optional surface-stress L2 indicator as non-energy-norm.
- Add shell membrane/bending stress and prestrain fields plus beam-fiber
  initial fields, with zero-load equilibration, immutable field provenance,
  exact load-stage boundary commits, and persistent plastic history through
  multi-stage displacement control.
- Replace the default finite-difference plane-stress algorithmic tangent with
  the analytical consistent derivative of the discrete return map. Retain the
  representable-step numerical tangent as a qualification oracle and guarded
  invalid-row fallback, add a safeguarded local consistency solve with
  fail-closed residual checks, and add constitutive and full-Newton
  parity/performance evidence.
## 0.1.1 - 2026-07-25

- Correct the layered-shell membrane/bending coupling tangent and keep the
  accelerated plastic shell path consistent with the scalar return mapping.
- Fail closed for curved B3 members, experimental Q8R qualification, unsupported
  DNV-RP-C208 thickness rows, and path-dependent staged displacement control.
- Include point masses in reported mass properties, make buckling constraints
  call-order independent, and correct CalculiX gravity export.
- Remove quadratic model-construction and beam/shell coupling lookup scaling,
  safely reset precomputed incoming elements, cache recovery operators and
  topology signatures, defer optional acceleration imports, and use lower
  PyPardiso thresholds only for compatible retained patterns.
- Preserve straight-sided B3 midpoint geometry through adaptive cylinder
  refinement and reconstruct complete elastic layer states after accelerated
  nonlinear solves.
- Add explicit runtime exports, generic generated-geometry API aliases, and
  package-root arc-length exports.
- Adopt the pure-solver test files from ANYstructure (triangular shell
  backend, local patch transition, geometry panel).

## 0.1.0 - 2026-07-22

- Transfer the qualified beam/shell solver from ANYintelligent into its own package.
- Add the headless runtime meshing and analysis facade formerly embedded in ANYstructure.
- Publish the Python import namespace as `anysolver` without a legacy `fe_solver` shim.
- Preserve SESAM interchange, verification, qualification, benchmark, and baseline workflows.
