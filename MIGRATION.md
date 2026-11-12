# Move to ANYfeSolver 0.5

Install the solver with `python -m pip install ANYfeSolver==0.5.0`. This installs the separate `ANYfeSolver-results` runtime. For offline recovery alone, install `ANYfeSolver-results==2.2.0`.

| Previous Python name | Current Python name |
| --- | --- |
| `anysolver` | `anyfesolver` |
| `anysolver_results` | `anyfesolver_results` |
| `python -m anysolver.sesam_fem` | `python -m anyfesolver.sesam_fem` |

Update application imports and dependency declarations together. For existing environments, upgrade the solver consumers first (ANYfem 0.4.2 and ANYstructure 6.4.2 for the released lines). Run `python -m pip check`, confirm your applications use the new imports, and remove the old distributions when no installed consumer needs them. A fresh environment installs just the new names.

Existing recovery capsules and detached result archives retain their format identifiers. Supported nonlinear restart files retain their original model fingerprints. Exact-build and unsupported-state refusals still apply.

# Migration to ANYsolver

ANYsolver is a curated snapshot, not a filtered-history import.

## Provenance

- Numerical solver source: `audunarn/ANYintelligent` starting at `96ec92e00e977415c5fed71eee39d1484f6dd068`, finalized at `9e4c59f` by the annotated tag `pre-anysolver-transfer-2026-07-22`.
- Headless runtime source: `audunarn/ANYstructure` at `738004fb1ac1cf033e71099e2a4057c5319ab878`.
- Target seed: `audunarn/ANYsolver` at `f9e80c7113be8b3aa217f36302eb075fe845459a`.

## Import changes

| Previous import | Replacement |
| --- | --- |
| `fe_solver` | `anysolver` |
| `anystruct.fe_solver_backend` | `anysolver` |
| `anystruct.fe_solver` | `anysolver.runtime` |
| `anystruct.fe_runtime_solver` | `anystruct.fem_integration` |

There is intentionally no `fe_solver` compatibility package.

## 0.2 extraction boundary

Version 0.2 removes the in-repository copies of material, neutral meshing, and
file-format implementations. Their canonical homes are now:

| Former ANYsolver surface | Canonical implementation |
| --- | --- |
| `anysolver.materials`, material curves | `anymaterial` |
| neutral parts of `anysolver.mesh_gen` | `anymesher` |
| SESAM records/documents/SIF and CalculiX readers/writer | `anyfileio` |

The old ANYsolver imports remain as thin compatibility facades through the
0.2.x line. Solver-owned behavior stays here: FEModel construction, exact MPC
constraints, support-label interpretation, CalculiX execution/provenance and
comparison evaluation. The extraction provenance and import replacements are
recorded here; each extracted package carries its own detailed migration note
and parity tests.

## Included

- Numerical model, element, assembly, linear, modal, buckling, nonlinear, continuation, dynamic, contact, damage, recovery, and validation modules.
- Generated-geometry and headless runtime analysis facades.
- Neutral SESAM-to-`FEModel` adapters and 0.2 compatibility imports over
  `ANYfileio`; SESAM syntax, documents, SIF parsing, and writing live in that
  package.
- FE-only tests, deterministic baseline, benchmarks, qualification scripts, and current solver documentation.

## Excluded

- PULS, machine-learning, training-data, and predictor artifacts.
- Generated reports and quality-control output directories.
- Python bytecode, IDE files, historical result dumps, and obsolete standalone cylinder/demo prototypes.
