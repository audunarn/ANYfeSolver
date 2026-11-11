"""Build the slice-2 installed static-contact candidate wheel bundle.

Ordinary packaging only: stage the current checkout solver sources with the
candidate stage version, build via setuptools.build_meta, and assemble one
bundle (wheelhouse, module-form examples, guide, requirements, SHA manifest).
The five non-solver private wheels are reused byte-identical from an existing
wheelhouse when available, otherwise built from the pinned sibling checkouts.
No solver product changes: the only source edit is the staged version string.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Packaging-only 0.5.0 successor; the frozen predecessor
# scripts/contact_candidate_slice2.json is retained unchanged.
CONFIG_PATH = ROOT / "scripts" / "contact_candidate_050.json"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
SOLVER_TOP_FILES = ("pyproject.toml", "README.md", "LICENSE", "COPYRIGHT",
                    "THIRD_PARTY_NOTICES.md")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_wheel(project_dir: Path, wheelhouse: Path) -> str:
    """Build one wheel from a project directory with setuptools.build_meta."""
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"
    os.environ["ANYMESHER_DISABLE_NATIVE"] = "1"
    os.environ["ANYMESHER_REQUIRE_NATIVE"] = "0"
    import _distutils_hack
    _distutils_hack.add_shim()
    from setuptools.build_meta import build_wheel
    previous = os.getcwd()
    os.chdir(project_dir)
    try:
        return build_wheel(str(wheelhouse))
    finally:
        os.chdir(previous)


def stage_solver(checkout: Path, stage: Path, candidate_version: str) -> dict:
    """Stage a copy of the solver sources with only the version string changed."""
    (stage / "src").mkdir(parents=True)
    shutil.copytree(checkout / "src" / "anyfesolver", stage / "src" / "anyfesolver",
                    ignore=IGNORE)
    for name in SOLVER_TOP_FILES:
        shutil.copy2(checkout / name, stage / name)
    pyproject = (stage / "pyproject.toml").read_text(encoding="utf-8")
    repo_version = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["repo_version"]
    old = f'version = "{repo_version}"'
    if pyproject.count(old) != 1:
        raise SystemExit(f"expected exactly one repo version line, found {pyproject.count(old)}")
    (stage / "pyproject.toml").write_text(
        pyproject.replace(old, f'version = "{candidate_version}"'), encoding="utf-8")
    files = {}
    for path in sorted((stage / "src" / "anyfesolver").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files[str(path.relative_to(stage))] = sha256_file(path)
    for name in SOLVER_TOP_FILES:
        files[name] = sha256_file(stage / name)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(staged_file_count=len(files), staged_source_digest=digest,
                repo_version_unchanged=True, version_change_only=True)


def write_guide(bundle: Path, config: dict) -> None:
    version = config["candidate_version"]
    bundle.joinpath("README.md").write_text(f"""# ANYfeSolver static-contact candidate

Private candidate wheels for the supported static-contact workflows.
Version: `{version}` (stage version change only; solver sources identical to
the checkout). Numerical runtime: see `requirements-tested.txt`.

## Install (fresh venv, cold resolver)

Use Python 3.12, create `python -m venv .venv` outside the source checkout,
then activate it (`.venv\\Scripts\\Activate.ps1` on Windows PowerShell or
`source .venv/bin/activate` on Ubuntu). From the extracted bundle root:

```
python -m pip install --no-cache-dir -r requirements-tested.txt
python -m pip install --no-cache-dir --no-index --find-links wheelhouse "ANYfeSolver[general-contact]=={version}" anyfesolver-results==2.2.0 ANYmaterial==0.2.0 "ANYgeometry[planar]==0.4.3" ANYmesher==0.5.0 ANYfileio==0.3.2
python -m pip check
```

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

## Run the public module-form examples from this bundle root

```
python -m examples.general_contact_s3_physical_orientation
python -m examples.general_contact_s3_friction_physical_path --output friction.json
python -m examples.general_contact_static_results_archive
python -m examples.elastic_sphere_stiffened_panel --output sphere-initial-state.json
python -m examples.elastic_sphere_panel_contact_skins --output sphere-contact-skins.json
```

The elastic sphere panel contact-skins example maps the actual 320 S3 sphere
and 100 Q4 panel top exterior skins of the delivered initial state
(status=skin_geometry_prepared, contact_response=unsupported). It is a
development-only geometry report with clearly labelled prescribed kinematic
probes; no contact solve, gap history or force history is claimed, and the
per-element offset facets are not claimed to form a watertight exterior.

The elastic sphere stiffened-panel example writes the verified initial state
(status=initial_state_prepared, impact_status=unsupported) and exits zero.
Requesting the impact with `--impact` is refused: it prints the refusal
document, writes no output file and exits 3; no impact history, transient
checkpoint or restart exists or is implied.

Save and resume a completed prefix directly:

```
python -m examples.general_contact_s3_friction_physical_path --through-step 3 --output partial.json --checkpoint-output checkpoint.json
python -m examples.general_contact_s3_friction_physical_path --through-step 7 --restart checkpoint.json --output completed.json
```

Completion exits zero. A noncompleted solve writes its actual status/reason with
`checkpoint: null` and exits nonzero; it cannot produce a restart checkpoint.
Keep completed-prefix checkpoints for recovery. Restart input and output paths
must differ. The embedded checkpoint in completed result JSON is retained.
Scope and limitations: see docs/GENERAL_CONTACT_STATIC_CANDIDATE.md in the
repository. This is a development candidate, not a qualified release.
""", encoding="utf-8")


def copy_delivered_examples(checkout: Path, bundle: Path, config: dict) -> list[str]:
    """Copy exactly the delivered public examples into the bundle.

    Only the delivered example modules named by the recipe's bundle_examples
    (plus the examples package marker when present) enter the bundle; no
    experimental checkout examples are delivered.
    """
    source, target = checkout / "examples", bundle / "examples"
    target.mkdir()
    marker = source / "__init__.py"
    if marker.is_file():
        shutil.copy2(marker, target / "__init__.py")
    else:
        (target / "__init__.py").write_text("", encoding="utf-8")
    for name in config["bundle_examples"]:
        module = source / f"{name}.py"
        if not module.is_file():
            raise SystemExit(f"delivered example missing from checkout: {module}")
        shutil.copy2(module, target / f"{name}.py")
    return list(config["bundle_examples"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="bundle output directory (created fresh)")
    parser.add_argument("--checkout", type=Path, default=ROOT)
    parser.add_argument("--reuse-wheelhouse", type=Path,
                        help="existing wheelhouse with byte-identical private wheels")
    parser.add_argument("--results-root", type=Path, default=None,
                        help="anyfesolver-results project root (default: checkout/packages/results)")
    parser.add_argument("--sibling-root", type=Path, default=None,
                        help="directory holding pinned sibling checkouts (default: checkout/.ecosystem)")
    args = parser.parse_args(argv)
    config = load_config()
    checkout = args.checkout.resolve()
    bundle = args.output.resolve()
    if bundle.exists():
        raise SystemExit("bundle output must not exist; fresh artifact only")
    wheelhouse = bundle / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    results_root = (args.results_root or checkout / "packages" / "results").resolve()
    sibling_root = (args.sibling_root or checkout / ".ecosystem").resolve()
    wheels = []
    for package in ("anyfesolver-results", "ANYmaterial", "ANYgeometry",
                    "ANYmesher", "ANYfileio"):
        expected = config["wheel_files"][package]
        source = (results_root if package == "anyfesolver-results"
                  else sibling_root / config["sibling_dirs"][package])
        reuse = args.reuse_wheelhouse / expected if args.reuse_wheelhouse else None
        if reuse and reuse.is_file():
            shutil.copy2(reuse, wheelhouse / expected)
            origin = f"reused:{reuse}"
        else:
            if not source.is_dir():
                raise SystemExit(f"missing source project for {package}: {source}")
            built = build_wheel(source, wheelhouse)
            if built != expected:
                raise SystemExit(f"{package}: built {built}, expected {expected}")
            origin = f"built:{source}"
        wheels.append(dict(package=package, file=expected,
                           sha256=sha256_file(wheelhouse / expected), origin=origin))
    import tempfile
    with tempfile.TemporaryDirectory(prefix="candidate-s2-stage-") as tmp:
        stage = Path(tmp) / "ANYfeSolver"
        identity = stage_solver(checkout, stage, config["candidate_version"])
        built = build_wheel(stage, wheelhouse)
        expected = config["wheel_files"]["ANYfeSolver"]
        if built != expected:
            raise SystemExit(f"ANYfeSolver: built {built}, expected {expected}")
        wheels.append(dict(package="ANYfeSolver", file=expected,
                           sha256=sha256_file(wheelhouse / expected),
                           origin="built:current-checkout-staged"))
    delivered_examples = copy_delivered_examples(checkout, bundle, config)
    (bundle / "requirements-tested.txt").write_text(
        "# Prospective slice-2 runtime; historical requirements remain unchanged.\n"
        + "\n".join(config["numerical_pins"]) + "\n", encoding="utf-8")
    write_guide(bundle, config)
    manifest = dict(status="BUILT", candidate_version=config["candidate_version"],
                    wheels=sorted(wheels, key=lambda w: w["package"]),
                    delivered_examples=delivered_examples,
                    solver_source_identity=identity,
                    config_sha256=sha256_file(CONFIG_PATH),
                    requirements_sha256=sha256_file(bundle / "requirements-tested.txt"))
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="BUILT", bundle=str(bundle),
                          wheels=[w["file"] for w in wheels])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
