"""Build frozen G6 wheel and probe solver plus opt-in consumers."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def fingerprint(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def write_new(path, value):
    raw = value if type(value) is bytes else canonical(value)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def _terminate(process):
    if process.poll() is None:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False, timeout=30)


def bounded(command, *, cwd, stdout, stderr, seconds=600):
    env = dict(os.environ, **{name: "1" for name in THREADS})
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    started = time.monotonic()
    with Path(stdout).open("xb") as out, Path(stderr).open("xb") as err:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            _terminate(process); raise RuntimeError("bounded G6 package child timeout")
    if code:
        raise RuntimeError("bounded G6 package child failed: " + " ".join(command))
    return {"returncode": 0, "elapsed_seconds": time.monotonic() - started,
            "stdout": fingerprint(stdout), "stderr": fingerprint(stderr)}


def safe_extract(archive, destination):
    destination = Path(destination).resolve(); destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            if not (destination / item.filename).resolve().is_relative_to(destination):
                raise ValueError("unsafe source archive member")
        source.extractall(destination)


def _installed(target):
    target = Path(target).resolve(); root = ROOT.resolve()
    sys.path[:] = [str(target), *[value for value in sys.path if value
        and not Path(value).resolve().is_relative_to(root)]]
    return target


def probe(target, output):
    target = _installed(target)
    import numpy as np
    import anyfesolver
    from anyfesolver.beam_sections import GeneralizedBeamSection
    from anyfesolver.elements import BeamElement, QuadraticBeamElement, create_element
    from anyfesolver.ge_beam3_element import GeometricallyExactBeam3D3NElement
    from anyfesolver.ge_beam3_native import ConsumerPolicy
    if not Path(anyfesolver.__file__).resolve().is_relative_to(target):
        raise RuntimeError("installed package origin escaped target")
    section = GeneralizedBeamSection(np.diag((12., 10., 9., 4., 5., 6.)),
        mass_matrix=np.diag((2., 2., 2., .1, .2, .3)), name="G6_INSTALLED")
    element = create_element("ge-beam3", 1, [1, 2, 3], "mat", section=section,
        cross_section={"contact_radius": .04}, reference_orientation=(0., 1., 0.))
    if (type(element) is not GeometricallyExactBeam3D3NElement
            or ConsumerPolicy.ge_beam3().selector != "ge-beam3"
            or type(create_element("beam", 2, [1, 3], "mat")) is not BeamElement
            or type(create_element("quadratic_beam", 3, [1, 2, 3], "mat")) is not QuadraticBeamElement):
        raise RuntimeError("installed G6 route mismatch")
    write_new(output, {"schema": "GE_BEAM3_G6_INSTALLED_PROBE_V1",
        "package_origin_below_target": True, "formulation_id": element.formulation_id,
        "explicit_selector_only": True, "legacy_b2_unchanged": True,
        "legacy_b3_unchanged": True, "production_default_qualified": False})


def consumer_probe(target, source, kind, output):
    _installed(target); sys.path.insert(0, str(Path(source).resolve()))
    import numpy as np
    if kind == "anyfem":
        from anyfem.solve.ge_beam3 import GeBeam3OptIn
        definition = GeBeam3OptIn(np.diag((10., 11., 12., 13., 14., 15.)),
            np.diag((2., 2., 2., .1, .2, .3)), (0., 1., 0.))
        element = definition.build(1, (1, 2, 3), "steel"); record = definition.to_dict()
    elif kind == "anystructure":
        from anystruct.ge_beam3_optin import GeBeam3RuntimeDefinition, runtime_status
        definition = GeBeam3RuntimeDefinition(np.diag((10., 11., 12., 13., 14., 15.)),
            np.diag((2., 2., 2., .1, .2, .3)), (0., 1., 0.))
        element = definition.build(1, (1, 2, 3), "steel"); record = runtime_status(definition)
    else:
        raise ValueError("registered consumer kind required")
    if type(element).__name__ != "GeometricallyExactBeam3D3NElement":
        raise RuntimeError("consumer did not construct exact GE-B3")
    write_new(output, {"schema": "GE_BEAM3_G6_CONSUMER_PROBE_V1", "consumer": kind,
        "formulation_id": element.formulation_id, "record": record,
        "explicit_opt_in": True, "default_changed": False})


def _head(path):
    safe = f"safe.directory={path.as_posix()}"
    dirty = subprocess.check_output(["git", "-c", safe, "status", "--porcelain",
        "--untracked-files=all"], cwd=path, text=True).strip()
    if dirty:
        raise ValueError(f"clean frozen sibling required: {path}")
    return subprocess.check_output(["git", "-c", safe, "rev-parse", "HEAD"],
                                   cwd=path, text=True).strip()


def run(revision, output, anyfem, anyfem_revision, anystructure, anystructure_revision):
    output = Path(output).resolve(); anyfem = Path(anyfem).resolve(); anystructure = Path(anystructure).resolve()
    if output.is_relative_to(ROOT.resolve()) or output.exists():
        raise ValueError("new external package output required")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                    cwd=ROOT, text=True).strip()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if dirty or revision != actual or _head(anyfem) != anyfem_revision or _head(anystructure) != anystructure_revision:
        raise ValueError("exact clean frozen candidate graph required")
    output.parent.mkdir(parents=True, exist_ok=True); output.mkdir(); receipts = []
    archive = output / "archive"; archive.mkdir()
    receipts.append({"name": "archive", **bounded(["git", "archive", "--format=zip",
        "--output=" + str(output / "source.zip"), revision], cwd=ROOT,
        stdout=archive / "stdout.log", stderr=archive / "stderr.log")})
    safe_extract(output / "source.zip", output / "source")
    build = output / "build"; build.mkdir()
    receipts.append({"name": "build", **bounded([sys.executable, "-B", "-m", "build", "--wheel",
        "--no-isolation", "--outdir", str(output / "wheels"), str(output / "source")], cwd=output,
        stdout=build / "stdout.log", stderr=build / "stderr.log")})
    wheels = list((output / "wheels").glob("*.whl"))
    if len(wheels) != 1 or not wheels[0].is_file() or wheels[0].is_symlink():
        raise ValueError("exact regular wheel required")
    wheel = wheels[0]; install = output / "install"; install.mkdir()
    receipts.append({"name": "install", **bounded([sys.executable, "-B", "-m", "pip", "install", "--no-index",
        "--no-deps", "--no-compile", "--target", str(output / "target"), str(wheel)], cwd=output,
        stdout=install / "stdout.log", stderr=install / "stderr.log")})
    for name in ("probe-a", "probe-b"):
        directory = output / name; directory.mkdir()
        receipts.append({"name": name, **bounded([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
            "--probe", str(output / "target"), str(directory / "science.json")], cwd=output,
            stdout=directory / "stdout.log", stderr=directory / "stderr.log")})
    left = (output / "probe-a/science.json").read_bytes()
    if left != (output / "probe-b/science.json").read_bytes():
        raise ValueError("installed probe replicas disagree")
    consumer_hashes = {}
    for kind, source in (("anyfem", anyfem / "src"), ("anystructure", anystructure)):
        directory = output / kind; directory.mkdir()
        receipts.append({"name": kind, **bounded([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
            "--consumer-probe", str(output / "target"), str(source), kind,
            str(directory / "science.json")], cwd=output,
            stdout=directory / "stdout.log", stderr=directory / "stderr.log")})
        consumer_hashes[kind + "_probe_sha256"] = fingerprint(directory / "science.json")["sha256"]
    write_new(output / "complete.json", {"schema": "GE_BEAM3_G6_INSTALLED_ARTIFACT_V1",
        "revision": revision, "anyfem_revision": anyfem_revision,
        "anystructure_revision": anystructure_revision,
        "installed_artifact": {"wheel_sha256": fingerprint(wheel)["sha256"],
            "probe_sha256": sha256(left).hexdigest(), "replicas_byte_identical": True,
            **consumer_hashes}, "wheel": {"filename": wheel.name, **fingerprint(wheel)},
        "receipts": receipts, "all_children_terminal": True,
        "production_default_qualified": False})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--revision")
    parser.add_argument("--output", type=Path); parser.add_argument("--anyfem", type=Path)
    parser.add_argument("--anyfem-revision"); parser.add_argument("--anystructure", type=Path)
    parser.add_argument("--anystructure-revision"); parser.add_argument("--probe", nargs=2)
    parser.add_argument("--consumer-probe", nargs=4)
    args = parser.parse_args()
    if args.probe: probe(*args.probe); return 0
    if args.consumer_probe: consumer_probe(*args.consumer_probe); return 0
    required = (args.revision, args.output, args.anyfem, args.anyfem_revision,
                args.anystructure, args.anystructure_revision)
    if any(value is None for value in required): parser.error("candidate graph and output required")
    run(args.revision, args.output, args.anyfem, args.anyfem_revision,
        args.anystructure, args.anystructure_revision); return 0


if __name__ == "__main__":
    raise SystemExit(main())
