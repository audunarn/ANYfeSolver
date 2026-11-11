# FE solver verification and quality control

This is the maintained verification guide for `anyfesolver`. Source code and
tests are authoritative; generated reports are evidence for the commit and
environment recorded in them.

## Evidence hierarchy

Use evidence in this order:

1. Current source and tests at the reported Git commit.
2. Executed tests appropriate to the change; full portable CI at integration.
3. The analytical/convergence/patch/support/performance QC suite.
4. `scripts/run_fe_verification.py`, which records commands, dependencies,
   warnings, return codes, and family reports.
5. Manifest-driven release gates and the production capability matrix.
6. Frozen local baselines.
7. Executed external-solver reports with solver identity, parsed observables,
   and tolerance-controlled comparisons.

Generated CalculiX-style input decks without matching executed results are
reproducible handoff artifacts, not numerical validation.

## Current verification policy

Since version 0.2, ANYfeSolver delegates material behavior, neutral meshing,
and interchange syntax to ANYmaterial, ANYmesher, and ANYfileio. A qualifying
ecosystem run
therefore installs those sibling packages, runs their own tests, and then runs
the ANYfeSolver suite and ecosystem-wiring tests. Compatibility tests must cover
both the historical ANYfeSolver imports and the canonical package imports.

Pass counts and timings are deliberately not embedded here: they become stale
as tests move to their owning repositories. The generated report under
`reports/verification/` is the evidence for a particular commit. Historical
release evidence remains available in Git history and `CHANGELOG.md`.

## Commands

Run from the repository root.

### Development loop

For an ordinary implementation edit, run the changed-behavior tests and one
affected integrated route. Select exact files or node IDs; inspect their scope.
Do not collect the complete suite, rebuild/install a wheel, hash the entire
package graph or replay a qualification campaign after every edit.

```powershell
python scripts/run_portable_ci.py --development `
  tests/test_changed_behavior.py::test_rejection `
  tests/test_affected_route.py --timeout-seconds 180 --memory-limit-mib 1024
```

Replace the example selectors with real tests. Focused mode runs exactly the
requested selectors in one source process, without CI deselections or automatic
dependency inference. Missing files/node IDs, empty selection, collection errors
and test failures fail. Skips/xfails remain visible and are not passes. The printed
temporary directory contains selection, effective runtime/coverage and resource
records. Keep those records with the relevant diff/commands; no separate
qualification package is required for routine development.

Default focused settings are one numerical thread, JIT off, plugin autoload off,
and a 600-second timeout. Use `--jit on` for changed compiled execution; JIT-off
checks cannot replace required compiled parity. Imported ANYfeSolver modules must
come from this checkout. Observed pools/Numba settings are recorded once after
the run, checked when loaded, and otherwise reported unavailable. The optional
memory ceiling requires `psutil` and samples the owned child process tree;
stricter task-specific time/memory/thread limits always take precedence. Ordinary
CI mode and its default timeout, inventory and matrix are unchanged.

### Coherent integration and promotion

Build and install once per coherent source candidate when its impact warrants
installed verification. Run affected lifecycle/public-consumer checks and one
representative example against that artifact. Prevent source `pythonpath` from
shadowing the installed package and verify actual import locations. Bind results
to the candidate and dependencies; changing artifact bytes changes its identity.
Reuse previous passing campaigns only for unchanged relevant inputs, with an
explicit impact explanation. Do not relabel old execution as a new artifact run.

Broaden testing for changed shared assembly, state, constraints, constitutive
updates, acceleration, public entry points or imports. Include required guard/
preimport and no-feature compatibility checks when affected. An isolated-plugin
development pass does not dismiss a real application/preimport failure.
The known application ABC metadata-cache mismatch remains an integration issue
until its responsible maintainer resolves it.

Full portable CI and all named scientific, external, installed-platform,
performance and release gates remain mandatory at their existing boundaries.
These tiers change execution frequency, not tolerances or supported claims.
Focused mode grants no execution authority, replays no consumed authority and
does not qualify or release a capability. Preserve failures. Do not repeat a
passed campaign without changed relevant inputs, a failure or an unresolved issue.

### Full verification commands

```powershell
python scripts/run_portable_ci.py --workers 2
python run_qc.py --no-save
python scripts/run_fe_verification.py --output-dir reports/verification
```

The portable runner is the current merge inventory. Raw full-directory pytest
also collects historical machine/authority-bound tests; use those only within
their registered contracts. The comprehensive scientific runner retains its
declared scope and authority requirements.

The comprehensive runner covers imports, focused FE tests, full pytest, QC,
baseline generation/comparison, benchmarks, element/member qualification,
modal/mass and buckling validity, plasticity, recovery/resource policy,
capacity workflow, external-reference artifacts, manifest gates, and
mesh/load/boundary-condition verification.

Use `--quick`, `--skip-full`, or a family flag only for development diagnosis.
A partial run must not replace full release evidence.

To execute the external reference family as part of the comprehensive run:

```powershell
python scripts/run_fe_verification.py `
  --execute-calculix `
  --calculix "C:\path\to\ccx.exe" `
  --output-dir reports/verification
```

The executable can instead be discovered from `PATH` or
`ANYSOLVER_CALCULIX_EXECUTABLE`. Each case runs in an isolated directory. An
executed case passes only when the process succeeds, recognized results are
parsed, and every declared comparison meets tolerance.

## Release-scope artifacts

```powershell
python scripts/run_beam_shell_verification.py `
  --output reports/beam_shell_verification/beam_shell_verification_report.json `
  --markdown reports/beam_shell_verification/beam_shell_verification_report.md `
  --external-reference-report reports/external_references/external_reference_report.json

python scripts/write_production_readiness.py `
  --output-dir reports/production_readiness/current
```

The release manifest consumes the canonical artifacts under
`reports/beam_shell_verification/`, `reports/production_readiness/current/`,
`reports/verification/`, and `reports/external_references/`. Do not replace a
valid executed external report with a deck-only artifact.

## Interpreting results

- `PASS` means the implemented metric met its stated acceptance criterion.
- `XFAIL` records a known unsupported feature, unavailable reference data, or
  external execution still required; it is not a pass.
- External `not_executed` means inputs were generated but no numerical
  comparison was performed.
- A generated report applies only to its recorded commit, dependencies,
  options, and environment.
- Performance values are machine-dependent. Correctness equivalence and
  fallback behavior are gates; a historical speedup number is not.
- Production qualification is narrower than the union of callable APIs. See
  [`README.md`](../README.md) and the generated capability matrix.

Historical numbered or ad-hoc verification directories are not maintained
documentation; rerun the appropriate canonical command instead.
