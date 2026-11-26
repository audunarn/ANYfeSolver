# Q4 R4 + single elastic law: merge procedure into local main (prepared, NOT executed)

Execute only after the qualification verdict is READY and the owner approves. No push.

## State (2026-10-23)
* Candidate: `claude/plastic-repair-20261023` at TARGET_SHA 5c34dc2e4 (src frozen). Later doc-only commits on
  this line may sit on top; they change no product digest.
* Local main: `git -C C:\Github\ANYsolver rev-parse main` = 7455222f9 (unchanged since the branch was cut).
  The R4 line descends from 7455222f9, so a fast-forward is possible while main stays there.

## Commands (main unchanged)
```powershell
git -C C:\Github\ANYsolver status --short                       # main checkout must be clean (else STOP, ask owner)
git -C C:\Github\ANYsolver rev-parse main                       # expect 7455222f9...
git -C C:\Github\ANYsolver merge-base --is-ancestor main claude/plastic-repair-20261023; $LASTEXITCODE   # 0
git -C C:\Github\ANYsolver checkout main
git -C C:\Github\ANYsolver merge --ff-only claude/plastic-repair-20261023
git -C C:\Github\ANYsolver log --oneline -1                     # tip = candidate tip
```
If the owner prefers a merge commit: `git merge --no-ff claude/plastic-repair-20261023 -m "Merge Q4 R4 assumed-strain plasticity and single elastic law"`.

## If main has moved (after 7455222f9)
```powershell
git -C C:\Github\ANYsolver log --oneline 7455222f9..main         # list the new main commits
git -C C:\Github\ANYsolver diff --stat 7455222f9 main -- src/anysolver/elements.py src/anysolver/e4_pl_element.py `
    src/anysolver/nonlinear_static.py src/anysolver/nonlinear_performance.py src/anysolver/nonlinear_performance_batch_b.py `
    src/anysolver/current_state_tangent.py src/anysolver/vectorized_nonlinear.py
git -C C:\Github\ANYsolver checkout -b claude/plastic-repair-merge main
git -C C:\Github\ANYsolver merge --no-ff claude/plastic-repair-20261023
```
Expected conflict locations (R4 hunks), resolve by keeping both sides' intent:
* `e4_pl_element.py`:
  * module imports (`_q4_assumed_strain`);
  * `_assumed_strain_kernel` (new method before `_replay_accepted_algorithmic_response`);
  * `_nonlinear_response_core` (coordinate guard, assumed-strain kernel call, `assumed_route`, direct route skipped);
  * committed-components direct route (`elastic_route` skip);
  * the dependency-module tuple (`_q4_assumed_strain`).
* `nonlinear_static.py`: import; reference-batch group key (`route`); `batch_kernel`; direct-builder skip
  (`not assumed_strain_group`).
* `nonlinear_performance.py`:
  * import;
  * `_Q4_ASSUMED_STRAIN_SHELL_KIND`;
  * `_ShellBatchPlan._shell_kernel` and plan build (`plan = cls(...)`, offsets);
  * the two `self._shell_kernel()(` calls;
  * shell-kind selection.
* `nonlinear_performance_batch_b.py`: batch-build exclusion of the assumed-strain kind (one line).
* `current_state_tangent.py`: import and `_Q4_DEPENDENCY_MODULE_AUTHORITY` entry.
* `elements.py`: R4 does not edit it. A conflict there means main changed the inherited kernel. Re-run the tier-0
  tests and the element FD tests before accepting.
After a non-fast-forward merge the product digest changes: the qualification numerical cells must be re-run on
the merged digest (plan runbook section 3) before main is called qualified.

## Source-binding records
Active test-checked bindings that read the changed sources (run these tests after the merge; rebind only with
owner approval, following the precedent f75aed6a4, which rewrote the pinned sha fields of the record in one
commit with an owner-approved note):
* `docs/reference_cases/e4_pl_q4_source_binding_20261019.json` (and its predecessor `..._20261016.json`, kept as
  history), checked by `tests/test_e4_pl_s3_mixed_mesh_manifest.py`. Already red at base. EXCEPTION granted by
  the coordinator 2026-10-23. A successor record (new file, `previous_source_binding` -> 20261019) is owed with
  the next full rebinding. It needs the new `e4_pl_element.py` sha, mechanics symbols and aggregate, and the two
  retargeted C5 test-file hashes.
* Tests that read the changed source files and may pin their hashes (verify after merge, list any red one):
  `test_e4_pl_s3_mixed_eigen_performance.py`, `test_e4_pl_s3_q4_burnin_authority.py`,
  `test_e4_pl_s3_v2_stage4a_authority.py`, `test_e4_pl_s3_v5h_closeout.py`,
  `test_e4_pl_s3_v6c_offset_load_restart_contract.py`, `test_ge_beam3_mixed_p3_optin.py`,
  `test_ge_beam3_q4_affine_numerical_recovery.py`, `test_ge_beam3_q4_affine_recovery.py`,
  `test_ge_beam3_q4_recovery_coefficient_audit.py`. Compare with the 168-file inventory base/candidate run: a
  binding test that is green at base and red after the merge needs an owner-approved rebinding commit.
* Historical evidence JSON under `docs/d1_review_support/`, `docs/contact-candidate-20261019/`,
  `docs/agent_plans/` records source hashes of past runs. These are history; do NOT rebind.

## CHANGELOG / docs at merge
* `CHANGELOG.md` "Unreleased" already carries the entry. At merge, add the qualification verdict line (ledger
  path, TARGET_SHA, psd) under it.
* `docs/Q4_ASSUMED_STRAIN_PLASTICITY_20261023.md`: add a "Qualification" paragraph (verdict, ledger path,
  owner decisions OD-Q4-1..5 from `C:\ANYworkspaceWork\qualification-plan\DECISIONS_LOG.md`).
* `docs/Q4_SINGLE_ELASTIC_LAW_PLAN_20261023.md`: mark HANDOVER as closed (merged SHA).
* Keep open items: plastic-law version tag (OPEN-HIGH); governed manifest successor record (EXCEPTION).

## Post-merge checks (tier 0/1, minutes)
`pytest tests/test_q4_assumed_strain_plasticity.py tests/test_e4_pl_q4_current_tangent.py
tests/test_general_contact_c5_elastic_increment.py tests/test_nonlinear_performance_batch_b.py`. Then the five
elastic hashes (`hashes.py`): the elastic hashes now change by the waiver, so compare with the qualification
ledger values, not with base.

## Scope of the Q4 source binding (owner decision A, 2026-11-25)
Applies to the whole chain `e4_pl_source_binding_20261111.json` -> `..._20261124.json` -> `..._20261125.json`
(and their builders and readers in `tests/test_e4_pl_s3_mixed_mesh_manifest.py`). The 2026-11-11 and 2026-11-24
records are pinned and stay byte-identical; this section is their documentation of scope. The 2026-11-25 record
states it as `binding_scope`.
* Covered: the bytes of the tracked source files in this checkout, and same-directory import-form uniqueness
  (`_require_sole_import_form`: the bound `.py` file is the only importable form of its module in its directory).
* Out of scope: runtime import identity. Examples: `sys.path` order (another checkout's `src`, or a checkout-root
  package, ahead of this `src`); bytecode caches (`__pycache__`, including a timestamp cache that survives an
  edit-and-restore within the same second); installed copies. A separate runtime check may follow later.
