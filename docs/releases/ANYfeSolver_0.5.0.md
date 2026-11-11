# ANYfeSolver 0.5.0 and ANYfeSolver-results 2.2.0

This release renames the distributions and import packages and packages the
current `main`. It is **not** a rename-only delta: `main` is 2620 commits beyond
the `v0.4.7` tag, so the solver code differs from 0.4.7 by far more than names.
The changes since 0.4.7 are listed in `CHANGELOG.md`; this note does not
describe them and makes no claim that element equations, tolerances, default
formulations or performance match 0.4.7.

No fresh scientific, accuracy or performance claim is made for 0.5.0. The only
checks described here are packaging checks (below).

## Names

| Before | From this release |
| --- | --- |
| PyPI `ANYsolver` 0.4.7 | PyPI `ANYfeSolver` 0.5.0 |
| `anysolver_results` runtime, development/source-only, never published on PyPI | PyPI `ANYfeSolver-results` 2.2.0 |
| `import anysolver` | `import anyfesolver` |
| `import anysolver_results` | `import anyfesolver_results` |

There are **no compatibility aliases**: `import anysolver` and
`import anysolver_results` are not provided by either new distribution, and the
verifier fails if an old import tree is shipped or resolves after a clean
install. Existing code must change its imports. The published `ANYsolver`
0.4.7 project is not modified by this release.

The supported historical wire formats (recovery capsule contracts v1 and v2)
are intended to remain readable by the renamed results runtime. This packaging
change does not itself test that; it is covered by the results tests run in CI.

`ANYfeSolver` 0.5.0 requires `ANYfeSolver-results>=2.2,<3` and
`ANYfileio>=0.3.3,<0.4`. The SESTRA reader uses the stress component order
and rotation frame reported in the file. The remaining solver dependency
bands and optional extras follow 0.4.7.
`ANYfeSolver-results` depends only on `numpy>=2.0`. Project URLs point to
`https://github.com/audunarn/ANYfeSolver`.

## Packaging

* Solver source: `src/anyfesolver`; results source:
  `packages/results/src/anyfesolver_results`. Both are built from the same
  commit; the results distribution is packaged and tested separately, and the
  pytest path covers both source trees.
* The optional CPython 3.11 Windows guard builds as `anyfesolver._native_guard`
  from the shipped C++ source, which keeps its existing
  `ANYSOLVER_GUARD_SOURCE_SHA` macro. `ANYFESOLVER_BUILD_NATIVE_GUARD=0`
  disables the build; the previous `ANYSOLVER_BUILD_NATIVE_GUARD` is honoured
  only when the new variable is absent. Other hosts keep the pure-Python
  package.
* Both distributions declare `MPL-2.0` and ship `LICENSE`, `COPYRIGHT` and
  `THIRD_PARTY_NOTICES.md`.

## Release gate

`scripts/verify_release_050.py` records, for one candidate commit:

* clean tracked tree and exact commit/tree identity;
* both wheel/sdist pairs equal the source trees byte for byte, with file
  inventories and SHA-256 values, license files and metadata, and dependency
  bands;
* each sdist is extracted in a fresh directory and rebuilt into a wheel whose
  content matches the shipped wheel, and the rebuilt pair installs and imports;
* a clean `--target` install of the shipped wheels imports only the new names and
  keeps the B3-GE opt-in, legacy B3 default and opt-in-only Armijo boundaries.

The frozen 0.4.7 evidence files (G7 and performance) are re-hashed by the
unmodified `scripts/verify_release_047.py` and cited only as evidence bound to
the 0.4.7 identities. The 0.4.7 terminal is not reused. No scientific
requalification, broad numerical run or performance campaign of 0.5.0 is
performed or implied. The portable test suite is covered separately by CI, which
must succeed on the release commit.

Terminal on success:

`ANYFESOLVER_0_5_0_PACKAGING_GATE_PASSED_NOT_A_PUBLICATION_DECISION`

It records packaging identity only; it neither approves nor publishes anything.

## Publication

`.github/workflows/publish.yml` runs only for tag `v0.5.0` (a dispatch on that
tag, or a published non-draft, non-prerelease GitHub release) on a commit
reachable from `main`, and requires a successful `push`/dispatch CI run on that
exact SHA. Pull-request runs are never used as evidence, and the only artifact
taken from another run is the validated stage artifact described below.

### Stage, then promote the same bytes

* **Stage** is a `workflow_dispatch` run on the tag. It has no production input.
  It builds once, runs the gate, uploads to TestPyPI (`testpypi` environment)
  and keeps the verified files as the run artifact `release-050-distributions`.
* **Promote** is a published GitHub release on the same tag. It **does not
  rebuild**: source distributions are not byte-reproducible (two real builds
  produced matching wheels but different sdists), so a rebuilt sdist would
  conflict with the digest TestPyPI already holds. Instead the `stage` job
  retrieves the artifact of the stage run and the protected `pypi` environment
  uploads those exact bytes.
* The stage run is accepted only if it is a run of
  `.github/workflows/publish.yml` in this same repository (not a fork, no pull
  request), event `workflow_dispatch`, completed with success including its
  TestPyPI job, for the same validated release SHA and on the same tag ref, and
  its artifact is the single unexpired `release-050-distributions` of that run.
  The files are re-hashed against `SHA256SUMS`, and the manifest must carry the
  same commit, versions, terminal and wheel digests.
  Release-time CI, licensing, dependency and tag/version checks still run.
* Uploads to either index are planned per file against the index's SHA-256:
  absent files are uploaded, files present with the same digest are left alone,
  and a differing digest or any foreign file for that version fails the run.
  Nothing is overwritten and nothing is skipped by filename.
  `ANYfeSolver-results` goes before `ANYfeSolver`. After uploading, each index is
  checked up to five times, 30 seconds apart, then the run fails.
* The pure-Python gate runs on the Linux staging runner. The optional Windows
  CPython 3.11 native guard is built and tested separately in CI; the staging
  gate neither builds nor compares native binaries.

### Retention, absent stage and re-staging

The artifact is kept for 30 days (`retention-days: 30`). If the stage run is
absent, failed, for another SHA or ref, or its artifact has expired, promotion
fails closed and uploads nothing.

* A transient failure in the stage run's TestPyPI job: use "re-run failed jobs"
  on that same run. The retained, already verified bytes are reused; the build is
  not repeated. Do not re-run the whole workflow or dispatch again after any file
  has reached TestPyPI: that rebuilds, the new sdist digests differ, and the
  per-file check fails by design.
* A stage that never uploaded anything, or an expired artifact before any upload,
  can be staged afresh with a new dispatch on the tag. Once TestPyPI holds files
  of a version, those names cannot be replaced, so an expired artifact after a
  partial or full TestPyPI upload requires a new version and tag.
* A failure in the promotion run before or during the PyPI upload can be
  re-run: files already on PyPI with the matching digest are left alone and only
  missing ones are uploaded.

The environment protections, the trusted-publisher registrations for both new
project names on both indexes and the `CHANGELOG.md` 0.5.0 section are
repository settings/content outside this file and must be in place before a
release is attempted. Workflow files were not executed here.
