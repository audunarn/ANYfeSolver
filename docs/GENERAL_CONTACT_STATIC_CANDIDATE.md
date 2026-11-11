# General-contact static candidate: supported entrypoints and run guide

Updated 2026-10-20. This page describes the **measured opt-in guided static
contact candidate** delivered on 2026-10-19 and its supported public
entrypoints, examples and restart/archive workflows. It is a usability guide,
not a qualification claim: full C0-C9/D0-D8 qualification, capability
admission and released/admitted flags remain unchanged and false. Start with
[the durable status/restart handoff](GENERAL_CONTACT_CANDIDATE_HANDOFF_20261019.md)
for the measured evidence, artifact hashes and consumed allowances.

## Scope

Supported, measured opt-in routes (the W1-W10 delivery workflows):

- elastic S3 normal contact against rigid tools: the S3 plane and finite
  planar triangulated rigid tools (round beam profiles remain a separate
  route and are not part of this measured claim);
- guided deformable shell-pair normal contact with explicit physical sides,
  including the automatic physical-side pair;
- material-anchor Coulomb stick/slip friction with reversal, press/open/
  recontact, checkpoint restart and rollback;
- detached public static result archives with integrity checks and corruption
  refusal.

Not supported outside that guided scope: general/automatic shell-shell,
body-body, rolling or self-contact, target bending in the friction example,
finite-rotation contact, swept friction, and any admission claim. The
rigid-sphere frictionless impact route documented on the README impact/contact
line is a distinct route and remains frictionless.

## Public entrypoints

All entrypoints live in the `anyfesolver.general_contact` package and load
lazily; ordinary no-contact solves never import the contact kernel.

```python
from anyfesolver.general_contact import (
    ContactDefinition, ContactInteraction, ContactSurface,
    ContactSolverSettings, ContactPathControl, ContactPathProgram,
    S3SlidingFrictionSettings,
    solve_static_contact_path,
    read_static_contact_results,
)
```

`S3SlidingFrictionSettings` is the base friction settings class (its
specialized subclasses remain exported too). Construct it with keywords:

```python
friction = S3SlidingFrictionSettings(
    friction_coefficient=0.3,
    tangential_stiffness_Pa_per_m=1000.0,
    steps_by_dof=(2e-7,) * model.mesh.dof_manager.total_dofs,
    maximum_candidate_evaluations=145,
)
```

`steps_by_dof` needs exactly one positive finite-difference step per model
DOF; take the vector length from `model.mesh.dof_manager.total_dofs`. The
measured candidate used the 2e-7 probe step and the 145 evaluation cap shown
above; changing them is allowed by validation but departs from the measured
settings.

## Installation

Two supported routes:

1. **Current candidate bundle** (recommended for evaluation):
   `C:/ANYworkspaceWork/contact-incremental-delivery-20261020/slice5/ANYfeSolver-contact-candidate-20261020-s5.zip`
   (expected slice-5 bundle; its paired-host CI and final artifact identity are
   recorded in the slice-5 delivery receipt once published).
   The extracted bundle ROOT contains `examples/` and a `wheelhouse/` with
   the six pure-Python private wheels (ANYfeSolver, ANYresults, ANYmaterial,
   ANYgeometry, ANYmesher, ANYfileio). From the bundle ROOT, in a fresh
   virtual environment:

   ```text
   python -m pip install --find-links wheelhouse -r requirements-tested.txt "ANYfeSolver[general-contact]==0.4.7+contact.candidate.20261020.s5" "ANYfeSolver-results==2.1.1" "ANYmaterial==0.2.0" "ANYgeometry[planar]==0.4.3" "ANYmesher==0.5.0" "ANYfileio==0.3.2"
   ```

   Numeric and Shapely dependencies resolve normally from their usual
   indexes; the bundle is not offline unless separately provided. The
   original 2026-10-19 bundle remains an immutable historical artifact whose
   claims apply to it alone; for evaluation it is superseded by the bundle
   above.
2. **Repository source** (development): install this repository's source
   package with its declared dependency closure and run the `examples/`
   scripts below from the repository root.

## Use the measured runtime

After activating the Python 3.12 environment, set these values before running
any example. The verified profile uses JIT off and one numerical thread.

PowerShell:

```powershell
$env:NUMBA_DISABLE_JIT = "1"
$env:NUMBA_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:ANYMESHER_DISABLE_NATIVE = "1"
$env:ANYMESHER_REQUIRE_NATIVE = "0"
```

Ubuntu shell:

```bash
export NUMBA_DISABLE_JIT=1 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ANYMESHER_DISABLE_NATIVE=1 ANYMESHER_REQUIRE_NATIVE=0
```

Other runtime configurations are outside this candidate's cross-host measurement.

## Run guide

From the repository root, or the extracted bundle ROOT containing
`examples/`:

- Normal rigid-tool contact:
  `python -m examples.general_contact_s3_plane`,
  `python -m examples.general_contact_triangulated_rigid`.
- Round beam profile (separate route, not part of the measured S3
  rigid-tool claim): `python -m examples.general_contact_round_beam_plane`.
- Automatic physical-side shell pair:
  `python -m examples.general_contact_automatic_s3_pair` and
  `python -m examples.general_contact_s3_physical_orientation`.
- Friction with restart and rollback. A completed partial run writes a restart
  checkpoint inside its output JSON and, with `--checkpoint-output`, the same
  canonical checkpoint bytes directly to a separate file; the restart recipe
  is (each command on its own line):

  ```text
  python -m examples.general_contact_s3_friction_physical_path --through-step 3 --output partial.json --checkpoint-output checkpoint.json
  python -m examples.general_contact_s3_friction_physical_path --through-step 7 --restart checkpoint.json --output completed.json
  ```

  A run that does not complete writes its result JSON with `checkpoint: null`
  and the actual `failure_reason`, prints one concise status/reason/path
  summary and exits nonzero; it never writes a direct checkpoint file. The
  restart input, result output and checkpoint output paths must be distinct
  files; the input checkpoint is never overwritten, and a failed write never
  leaves a partially written destination file.
- Static result archive (export and detached read):
  `python -m examples.general_contact_static_results_archive`.
- Elastic sphere stiffened-panel initial state (separate bounded case, not
  part of the measured S3 rigid-tool claim):
  `python -m examples.elastic_sphere_stiffened_panel --output sphere-initial-state.json`.
  It writes the verified initial state
  (`status=initial_state_prepared`, `impact_status=unsupported`) and exits
  zero; `--impact` is refused with exit 3 and no output file. See
  `docs/ELASTIC_SPHERE_STIFFENED_PANEL.md`.

## Failure, retained state and restart

Use `result.status == "completed"` to decide whether the requested target
finished. Gap rejection reports `contact_gap_exceeded`; exhausted stepping
after acceptance reports `stopped_at_limit`. Failure and iteration-level
reasons describe the actual cause. A retained step may have `converged=True`
while the overall target remains incomplete.

Rejected trials retain the last accepted displacement and material history.
Cancellation discards trial state and permits a fresh solve on the same
model. Checkpoint continuation requires a **completed prefix target**: keep
that checkpoint before further loading. A checkpoint from a terminally failed
solve preserves state for inspection but is refused for direct continuation.
This existing restriction is unchanged; the commands above use a completed prefix.

## Measured settings strings

The candidate was measured with exactly these settings strings and values:
`general_s3_differentiation='jet2_s3_dev'` with difference step 2e-5,
`path_predictor='control_projection_dev'`,
`shell_pair_orientation='physical_sides_dev'`, projected-penalty normal law
with penalty 1000. and gap limit 5e-5, and the friction settings above.
These historical development-profile names are used by the selected measured
examples. The suffix alone establishes neither measured coverage nor
qualification; only the listed workflows have the stated evidence.

## Limitations

- Agreement claims are for the declared finite-penalty law and chosen
  geometry, not continuum or mesh/model error estimates.
- The earlier s2 Windows and Ubuntu Python 3.12 cold installs of the same six wheels passed on 2026-10-20: exact package origins, 20 selected assertions per host, normal/friction/archive commands, transferred active/open restart and all 29 cross-host comparison entries. See the handoff for dimensional bounds and measured differences. Full qualification and unrelated full-CI failures remain open.
- For the current s5 candidate, consult the slice-5 [delivery receipt](C:/ANYworkspaceWork/contact-incremental-delivery-20261020/slice5/DELIVERY.json) for the installed local check, paired-host CI outcome and exact artifact identity once published; until then the s5 changes are implemented with local checks only.
- The slice-4 bundle and its [delivery receipt](C:/ANYworkspaceWork/contact-incremental-delivery-20261020/slice4/DELIVERY.json) remain historical evidence. Earlier measurements do not automatically bind new wheel bytes.
- The 2026-10-19 measurements remain same-host Windows historical evidence.
- Capability/released/admitted flags are unchanged; nothing on this page
  admits the candidate into qualified production scope.


## Slice-7 development-only skin geometry addition

The s7 candidate (`0.4.7+contact.candidate.20261020.s7`) appends only the
development-only curved shell skin geometry service, its thirteen
focused selectors and the `elastic_sphere_panel_contact_skins` example to
the installed workflow and bundle. The new selectors run as a separate
named `curved-skins` pytest batch appended after the unchanged original
35-selector base batch (the Windows base batch already fills most of the
unchanged per-case seconds cap), with the same origins, per-case timeout,
RSS cap and budget accounting and no new runner or harness. It admits no contact solve, changes no
solver behavior, criteria, scales or cross-host quantities, and claims no
contact response for the sphere-panel case; the previously measured s2/s6
evidence is unchanged. See the elastic sphere-panel guide for the geometry
service contract and its honest limitations.

CI scheduling note: the first s7 Windows run completed all thirteen new skin
checks and the new installed CLI, but the combined older 35-assertion batch
hit its 120-second process limit. Those same assertions are now partitioned
into 26 static and nine initial-state selectors. The existing named-batch
runner retains every assertion and the unchanged 120-second case, 600-second
host and 953 MiB limits. Failed run 39435678124 is preserved in the s7 evidence.
