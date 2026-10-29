# Documentation guide

Start with the [solver overview](../README.md) and [current development status](STATUS.md).
The status page distinguishes source on local main, dated verification evidence,
and remaining qualification work. A dated handoff or a log is not a current
capability declaration.

## Using the solver

| Topic | Guide |
|---|---|
| Package ownership and analysis flow | [Architecture](ARCHITECTURE.md) |
| Formulations and validity limits | [Theory](THEORY.md) |
| Continuation | [Arc length](ARC_LENGTH.md) |
| Beam-shell connections | [Connection options](BEAM_SHELL_CONNECTION.md) |
| Free-body linear statics | [Inertia relief](INERTIA_RELIEF.md) |
| Attached mass and inertia | [Concentrated mass](concentrated_mass.md) |
| Floating-body response and wind feedback | [Floating response](FLOATING_RESPONSE.md) |
| General contact candidate | [Static candidate guide](GENERAL_CONTACT_STATIC_CANDIDATE.md) and [capability ledger](GENERAL_CONTACT_CAPABILITIES.md) |
| Results and recovery | [Result capsule](RESULT_CAPSULE.md) |
| Prepared static workflows | [Prepared static response](PREPARED_STATIC_RESPONSE.md) |
| Import and ownership changes | [Migration](../MIGRATION.md) |

## Formulation and performance details

- [Q4 assumed-strain plasticity](Q4_ASSUMED_STRAIN_PLASTICITY_20261023.md) and
  [S3 von Karman definition](S3_VON_KARMAN_ELEMENT_DEFINITION.md) describe the
  adopted laws and their limitations. Historical results using different laws
  are not interchangeable.
- [Native beam workflows](GE_BEAM3_NATIVE_WORKFLOWS.md),
  [B3 production opt-in](B3_GE_PRODUCTION_OPT_IN.md), and
  [native source authority](NATIVE_SOURCE_AUTHORITY.md) retain the exact
  admission and source-identity boundaries.
- [Nonlinear performance](NONLINEAR_PERFORMANCE.md) and
  [performance architecture](PERFORMANCE_ARCHITECTURE.md) describe runtime paths.
  Dated measurements apply to their recorded candidates and environments.
- Marine development routes have separate contracts: [linear radiation](MARINE_LINEAR_DEVELOPMENT.md),
  [rigid mechanics](MARINE_RIGID_DEVELOPMENT.md), [potential flow](MARINE_POTENTIAL_FLOW_DEVELOPMENT.md),
  [Morison loading](MARINE_MORISON_DEVELOPMENT.md), and [mooring equilibrium](MARINE_MOORING_EQUILIBRIUM.md).

## Verification and development

- [Quality control](QUALITY_CONTROL.md): focused development checks, integration
  checks, and release requirements.
- [Reference cases](reference_cases/README.md): analytical/external comparison
  inputs and their execution status.
- [Changelog](../CHANGELOG.md) and [release records](releases/): version history.
- [Repository instructions](../AGENTS.md) and [ecosystem guide](../ECOSYSTEM_GUIDE.md):
  current engineering procedure and worker policy. Locally ignored workflow
  notes are not distributed repository policy.

## Historical records and logs

Required records remain available because tests, scripts, scientific reviews or
source-identity records depend on their exact paths and contents. Bulk historical
outputs were removed from the checkout on 2026-10-29; see the
[development archive and restoration guide](DEVELOPMENT_ARCHIVE.md).

| Location | Purpose and interpretation |
|---|---|
| [agent_plans/](agent_plans/) | Dated plans, registrations, candidate evidence and consumed execution allowances; not an automatic work queue. |
| [d1_review_support/](d1_review_support/) | Contact review inputs, reference cases and retained failures. |
| [.development/](../.development/) | Retained integration task record; archived attempts and source snapshots are held outside the checkout. |
| [.qualification/](../.qualification/) | Retained regression inputs. Bulk historical qualification outputs and failed attempts are preserved in the external archive. |
| [reports/](../reports/) | Generated measurements and reports. Read their command, source identity and environment before relying on a result. |
| Root `GENERAL_CONTACT_*` files | Contact programme decisions, history and execution authority. Use the static candidate guide for the supported user workflow. |
| Dated `*_HANDOFF`, `*_STATUS`, `*_PLAN` and trust-ledger files | Historical checkpoints. Later dated entries may close an earlier open item; consult [current status](STATUS.md) first. |

Do not rewrite old logs to make them look current. Record a new result with its
actual command and source identity. Do not delete a file merely because it is
old, ignored by Git, or absent from a simple import search: dynamic imports,
compatibility routes and source-bound certificates are part of this repository.
