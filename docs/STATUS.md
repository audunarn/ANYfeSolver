# Current development status

Reconciled **2026-10-29**, against local main
`4bd22baddad4aa415df18e3d486292aabd2de087` before the maintenance cleanup.
Package metadata declares **0.4.7**; this is not confirmation of a published
release. Remote publication and CI were not checked in this reconciliation.

This is the entry point for current status. Keep detailed historical evidence at
its existing path. Update this page when integration changes the source or closes
an open item, with the relevant commit and evidence; do not roll old pass counts
forward to a new candidate.

## Integrated source

| Area | State and evidence |
|---|---|
| Structural analyses | Linear/nonlinear statics, modal/buckling, continuation and transient/impact routes remain available within the [README production scope](../README.md#production-scope). |
| Q4 and S3 laws | The October 23–24 changes are integrated. See [Q4 plasticity](Q4_ASSUMED_STRAIN_PLASTICITY_20261023.md), [S3 definition](S3_VON_KARMAN_ELEMENT_DEFINITION.md), and the final dated entries in the [trust ledger](TRUST_LEDGER_20261023.md). Earlier ledger entries describe intermediate states. |
| Inertia relief | Linear free-free inertia relief and prepared static-batch integration landed October 27 (`c13990cb3`, `4f71f7848`, merged through `491026cad`). See [the admitted domain and refusal paths](INERTIA_RELIEF.md). |
| Floating response | Six-DOF frequency response, fixed-step transient wind-thrust feedback and physical pressure-region loading landed October 28 (`414e9e407`, merged through `4bd22badd`). These are [development routes](FLOATING_RESPONSE.md), not a general marine qualification claim. |
| Concentrated mass | Rigid attached mass, offset/inertia coupling and compatibility fixes landed October 28 (`8c990e212`, `d38676977`). See [the linear scope](concentrated_mass.md). |
| Impact reference and CI dependency | October 28 commits `5cc825716` and `38e9abc17` align the impact reference with the adopted membrane law and update the SESAM-owner CI pin. A commit is not evidence that the current CI run passed. |
| General contact | The [measured opt-in guided static candidate](GENERAL_CONTACT_STATIC_CANDIDATE.md) has public workflows, restart and detached archives. Full C0–C9/D0–D8 qualification and general admission remain incomplete. |

## Open limits and follow-ups

- On October 29, the older V6 source-authority check reported a mismatch for
  `src/anysolver/e4_pl_s3_v2c_element.py`. The same failure occurred before and
  after development-file removal. Its historical binding was not rewritten;
  see the [cleanup evidence](CLEANUP_20261029.md).
- The [Q4 changelog entry](../CHANGELOG.md#unreleased) records the unresolved
  historical plastic-law restart-state identification risk, public stress
  recovery review, and a governed manifest successor. This cleanup does not
  close those engineering items.
- The [final October 24 ledger entry](TRUST_LEDGER_20261023.md) retains the S3
  impact descriptor-shear follow-up and verification coverage gaps. Its earlier
  g3c Q4 channel and lifted-path blockers were subsequently addressed in that
  same ledger; do not treat its earlier snapshot as the final state.
- A single large slide of the documented pressed S3 patch can refuse under the
  adopted von Karman law. The [known-limit discussion](S3_VON_KARMAN_ELEMENT_DEFINITION.md)
  records the two-increment workflow and distinguishes endpoint displacement
  agreement from path-dependent dissipation.
- General-contact candidate evidence has a finite measured domain. Native
  dynamics, impact, arbitrary geometry/material combinations and application
  admission do not inherit qualification from a static example.
- Older source snapshots and duplicate-looking formulation variants remain
  where compatibility or exact scientific identity requires them. Removing or
  changing them needs its own impact assessment.

These are carried-forward limitations, not a fresh exhaustive defect audit.
Source and targeted current checks decide whether a reported issue still applies.

## Maintenance verification

The October 29 cleanup removes 22,552 tracked historical development files from
the checkout after exact archive verification. Audited required fixtures and
authority records remain in place; active package source and current tests are retained.
Another 18,738 inventoried ignored files were relocated in intact evidence trees.
Use the [archive guide](DEVELOPMENT_ARCHIVE.md) to recover a historical campaign.

The [October 29 cleanup record](CLEANUP_20261029.md) contains the actual changed
scope, commands, counts, review findings and evidence locations for this cleanup.
It does not renew consumed qualification allowances or establish a new release.
Use [quality control](QUALITY_CONTROL.md) for the required checks at each
development, integration and release boundary.
