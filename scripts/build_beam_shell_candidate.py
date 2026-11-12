"""Build the private beam-shell bundle from a committed Git source identity.

Reuses the existing candidate's setuptools staging/wheel helpers and immutable
private dependency wheels. Uncommitted files are never packaged.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VERSION = "0.4.7+beam.shell.candidate.20261020.s3"
PINS = ["numpy==2.5.4", "scipy==1.18.1", "numba==0.68.0", "llvmlite==0.50.0",
        "threadpoolctl==3.7.0", "shapely==2.2.0", "pytest==9.1.1", "psutil==7.2.2"]
PRIVATE = ["anyfileio-0.3.2-py3-none-any.whl", "anygeometry-0.4.3-py3-none-any.whl",
           "anymaterial-0.2.0-py3-none-any.whl", "anymesher-0.5.0-py3-none-any.whl",
           "anyfesolver_results-2.1.1-py3-none-any.whl"]
TESTS = ["test_beam_shell_connection_options.py", "test_director_integrated_trace.py",
         "test_beam_shell_workflow.py", "test_beam_shell_checkpoint_workflow.py",
         "test_nonlinear_beam_shell_slice3_continuation.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dependency-wheelhouse", required=True, type=Path)
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--installed-test-guard", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", args.source_ref + "^{commit}"],
                                     cwd=ROOT, text=True).strip()
    selected = ["src/anyfesolver", "pyproject.toml", "README.md", "LICENSE", "COPYRIGHT",
                "THIRD_PARTY_NOTICES.md", "examples/beam_shell_coupling.py",
                "docs/BEAM_SHELL_CONNECTION.md", "CHANGELOG.md",
                "scripts/build_static_contact_candidate.py", "scripts/contact_candidate_slice2.json"]
    for name in TESTS:
        path = "tests/" + name
        if subprocess.run(["git", "cat-file", "-e", commit+":"+path], cwd=ROOT,
                          capture_output=True).returncode == 0:
            selected.append(path)
    archive = subprocess.check_output(["git", "archive", "--format=zip", commit,
                                       "--", *selected], cwd=ROOT)
    bundle = output / "bundle"
    wheelhouse = bundle / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    sdist_dir = bundle / "source-distribution"
    sdist_dir.mkdir()
    with tempfile.TemporaryDirectory(prefix="beam-shell-committed-") as temporary:
        committed = Path(temporary) / "committed"
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            zipped.extractall(committed)
        stage = Path(temporary) / "stage"
        # Freeze the reused build helper too: other main work may be dirty.
        helper_path = committed / "scripts/build_static_contact_candidate.py"
        spec = importlib.util.spec_from_file_location("committed_candidate_packaging", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        identity = helper.stage_solver(committed, stage, VERSION)
        for directory in ("examples", "docs", "tests"):
            shutil.copytree(committed / directory, stage / directory)
        shutil.copyfile(committed / "CHANGELOG.md", stage / "CHANGELOG.md")
        (stage / "MANIFEST.in").write_text(
            "recursive-include examples *.py\nrecursive-include docs *.md\n"
            "recursive-include tests *.py\ninclude CHANGELOG.md\n", encoding="utf-8")
        wheel = helper.build_wheel(stage, wheelhouse)
        helper_sha = sha(helper_path)
        from setuptools.build_meta import build_sdist
        previous = Path.cwd()
        try:
            os.chdir(stage)
            sdist = build_sdist(str(sdist_dir))
        finally:
            os.chdir(previous)
        sources = {str(p.relative_to(stage / "src")): sha(p)
                   for p in sorted((stage / "src" / "anyfesolver").rglob("*"))
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        for directory in ("examples", "docs", "tests"):
            shutil.copytree(committed / directory, bundle / directory)
        shutil.copyfile(committed / "CHANGELOG.md", bundle / "CHANGELOG.md")
    checkpoint_workflow = (bundle / "tests/test_beam_shell_checkpoint_workflow.py").exists()
    if args.installed_test_guard:
        shutil.copyfile(args.installed_test_guard, bundle / "tests/conftest.py")
    for filename in PRIVATE:
        shutil.copyfile(args.dependency_wheelhouse / filename, wheelhouse / filename)
    (bundle / "requirements-tested.txt").write_text("\n".join(PINS) + "\n", encoding="utf-8")
    recipe = 'python -I -m pip install -r requirements-tested.txt ' + ' '.join(
        '"wheelhouse/' + name + '"' for name in [wheel, *PRIVATE])
    environment = dict(NUMBA_DISABLE_JIT="1", NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", ANYMESHER_DISABLE_NATIVE="1", ANYMESHER_REQUIRE_NATIVE="0",
        ANYSOLVER_RUN_RETIRED_PROGRAMME_TESTS="0")
    powershell = "\n".join(f'$env:{key} = "{value}"' for key, value in environment.items())
    (bundle / "README.md").write_text(f'''# Beam-shell development candidate

Version: `{VERSION}`. Committed solver/example source: `{commit}`.
Use a fresh Windows Python3.12 environment, then from this bundle root:

```text
{recipe}
```

Before starting Python, use the measured profile:

```powershell
{powershell}
```

Run both pre-yield modes from this bundle root:

```text
python -m examples.beam_shell_coupling --coupling direct
python -m examples.beam_shell_coupling --coupling interpolated
```

''' + ('''The completed-prefix S355 workflow is included. Run each command in a
separate process, using the same coupling for all three:

```text
python -m examples.beam_shell_coupling --case plastic --coupling interpolated --checkpoint-out full.json
python -m examples.beam_shell_coupling --case plastic --coupling interpolated --stage prefix --checkpoint-out prefix.json
python -m examples.beam_shell_coupling --case plastic --coupling interpolated --restart prefix.json --checkpoint-out resumed.json
```

Repeat with `--coupling direct`. Full/prefix/resume retain 11/6/5 initial
increments, maximum 40 Newton iterations and tolerance 1e-8. Targets are
-11/60 m and -0.10 m. JSON reports actual completion, target error and constraints.
Only completed targets replace checkpoint output files. Rejected trials,
cancellation and refused failed-terminal restarts preserve recovery inputs.
''' if checkpoint_workflow else '''Serialized plastic checkpoint delivery is
omitted for this source identity; only the pre-yield workflow is included.
''') + f'''
See `docs/BEAM_SHELL_CONNECTION.md` for registered scope and limits. The package
release version in the repository is unchanged. This bundle is not offline: numerical wheels
resolve from the configured package index.

This is a bounded development workflow, not a qualification or release.
Director continuation and analytical accuracy remain open. The historical
beam-shell checkpoint collected568 tests:565passed/3failed; those three
director failures remain ordinary tests in the repository. Focused delivery
checks do not establish a green full suite or accuracy outside the stated case.
`converged` alone does not establish target completion. Failed-terminal
checkpoints remain refused for continuation.
The earlier pre-yield-only artifact has the same private version but a different
source/hash. Install the explicit local wheel above, not a version-only index
selection. This artifact is identified by source `{commit}` and its SHA manifest.
''', encoding="utf-8")
    files = {str(p.relative_to(bundle)): sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    manifest = dict(status="BUILT_NOT_YET_INSTALLED_VERIFIED", version=VERSION,
                    source_commit=commit, stage_identity=identity, solver_wheel=wheel,
                    sdist=sdist, source_files=sources, files=files,
                    build_script_sha256=sha(Path(__file__)), build_helper_sha256=helper_sha,
                    delivered_features=["direct/interpolated preyield public workflow", "truthful completion/cancellation reporting"] +
                        (["S355 completed-prefix serialized checkpoint/restart, both default couplings"] if checkpoint_workflow else []),
                    omitted_features=[] if checkpoint_workflow else ["serialized plastic prefix/restart workflow"],
                    deferred_scientific_gates=["original -0.20 m target", "director continuation", "analytical accuracy"],
                    installed_test_guard_sha256=sha(args.installed_test_guard) if args.installed_test_guard else None,
                    runtime_environment=environment,
                    install_command=recipe)
    (bundle / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = Path(shutil.make_archive(str(output / ("ANYfeSolver-beam-shell-"+VERSION)),
                                       "zip", root_dir=bundle))
    (output / "BUILD.json").write_text(json.dumps(dict(source_commit=commit,
        bundle=str(bundle), zip=str(zip_path), sha256=sha(zip_path)), indent=2))
    print(json.dumps(dict(bundle=str(bundle), zip=str(zip_path), source_commit=commit)))


if __name__ == "__main__":
    main()
