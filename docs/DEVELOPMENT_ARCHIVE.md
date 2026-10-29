# Archived development history

On 2026-10-29, the user explicitly requested removal of unnecessary development
files from the production checkout. **22,552 tracked files** and **18,738
inventoried ignored local files** were removed or relocated outside this repository.
The baseline's existing tracked-file count fell from **35,812 to 13,260**;
new maintenance guides/tests are additional to that baseline count.

No active package source or current test module was removed in this slice, and
no scientific tolerance or qualification decision changed. Historical source
snapshots, including old solver copies, were archived. Authority records,
reference implementations and fixtures needed by the audited consumers remain
at their original paths. Historical campaign replay
may require restoring its archived inputs first; restoration grants no new
execution authority or qualification credit.

## What left the checkout

| Tree | Tracked files removed | Disposition |
|---|---:|---|
| `.development/` | 8,179 | Old nonlinear-coupling attempts/snapshots and completed integration outputs; current inertia-relief task record retained. |
| `.qualification/` | 12,372 | Historical run output, snapshots and qualification support; 37 exact regression inputs retained. |
| `docs/agent_plans/` | 2,001 | Historical outputs: 1,453 under `development/`, 548 elsewhere; directly referenced evidence, authority records and Python reference sources retained. |

All **2,244 retained files** across the audited trees were hashed before and after
removal, with no mismatch. No unexpected tracked deletion or remaining selected
file was found. Tests and selection policy were not weakened or changed.

## Verified archive of tracked history

External archive:

```text
C:/ANYworkspaceWork/anysolver-cleanup-20261029/removal/development-history-4bd22badd-20261029.zip
```

- Files: **22,552**; original bytes: **1,455,874,326**.
- Archive bytes: **670,018,306**.
- SHA-256: `12d27b99b695d00ded00680d0ee433a7283477b0846c7c2ee6a49d959077721d`.
- Every archived member's size and SHA-256 were verified before deletion.
- `_archive_manifest.json` inside the ZIP records each original relative path,
  size and SHA-256. The external `manifest.json` has SHA-256
  `8228e103358bb7a26865a5afc58b1b334b1eafe900693714d904a839759699a7`.
- Original Git baseline: `4bd22baddad4aa415df18e3d486292aabd2de087`.
  Git history remains intact; it is a second recovery source for tracked files.

## Intact local evidence relocation

Additional ignored evidence is preserved at:

```text
C:/ANYworkspaceWork/anysolver-cleanup-20261029/removal/local-evidence/.qualification/
C:/ANYworkspaceWork/anysolver-cleanup-20261029/removal/local-evidence/.development/nonlinear-master-slave-coupling-20261017/
```

These were same-volume directory moves, with unchanged directory file identities.
Some old installed snapshots deny file access even outside the sandbox; their
contents and permissions were preserved without rewriting or deleting descendants.
Consequently **18,738 is a lower-bound file inventory**, not a complete count of
inaccessible descendants, and no per-file hash claim is made for those unreadable
files. The 37 required qualification inputs were copied back and hash-verified.
The previously absent detached-port optional-fixture prefix remains absent, so
this operation does not add a skip for that fixture.

The external `local-relocation-receipt.json` records both directory identities,
source/destination paths and restoration outcome. `post-removal-check.json`
confirms all inventoried moved paths are absent from the checkout and every
retained-file hash matches. The dependency audit covers observed static and
composed-path consumers; arbitrary dynamically generated paths are not proven.
Failed attempts and failed scientific evidence were
preserved, including original local outputs that were never committed to Git.

## Recovering a historical input

Verify the ZIP digest above, then extract into a **new directory outside the
checkout**, for example:

```powershell
Get-FileHash -Algorithm SHA256 C:/ANYworkspaceWork/anysolver-cleanup-20261029/removal/development-history-4bd22badd-20261029.zip
python -m zipfile -e C:/ANYworkspaceWork/anysolver-cleanup-20261029/removal/development-history-4bd22badd-20261029.zip C:/ANYworkspaceWork/ANYsolver-history-restored
```

Use `_archive_manifest.json` to locate and verify the exact input. Copy only the
required historical paths back when deliberately preparing a replay, preserving
current work. Historical documentation can still name those original paths;
those references now address archived history rather than installed solver data.
For tracked inputs, Git at the recorded baseline also retains their original
content. For ignored local inputs, use the intact external directory trees;
Git cannot recover them.

New disposable development outputs should live outside the source checkout.
The ignore rules prevent accidentally adding bulk output back to the archived
trees; existing tracked fixtures and authority files remain versioned.

## Verification scope

See [the cleanup record](CLEANUP_20261029.md) for exact commands, review, baseline
failures and post-removal results. This archive is preservation evidence, not a
new scientific acceptance result or release. Historical source-bound records and
consumed budgets retain their original meaning.
