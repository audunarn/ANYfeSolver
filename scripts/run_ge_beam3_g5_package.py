"""Build one frozen G5 wheel and run two isolated installed probes."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS")


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def fingerprint(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def write_new(path, value):
    raw = value if type(value) is bytes else canonical(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def _terminate_tree(process):
    if process.poll() is not None:
        return
    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=False, timeout=30)


def bounded(command, *, cwd, stdout, stderr, seconds=600):
    env = dict(os.environ, **{name: "1" for name in THREADS})
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    started = time.monotonic()
    with Path(stdout).open("xb") as out, Path(stderr).open("xb") as err:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out,
                                   stderr=err, creationflags=flags)
        try:
            returncode = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            _terminate_tree(process)
            raise RuntimeError("bounded package child timeout")
    elapsed = time.monotonic() - started
    if returncode:
        raise RuntimeError("bounded package child failed: " + " ".join(command))
    return {"returncode": returncode, "elapsed_seconds": elapsed,
            "stdout": fingerprint(stdout), "stderr": fingerprint(stderr)}


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("unsafe source archive member")
        source.extractall(destination)


def probe(target, output):
    target = Path(target).resolve(); output = Path(output).resolve()
    root = ROOT.resolve()
    sys.path[:] = [str(target), *[
        value for value in sys.path
        if value and not Path(value).resolve().is_relative_to(root)
    ]]
    import numpy as np
    import anyfesolver
    from anyfesolver.beam_sections import GeneralizedBeamSection
    from anyfesolver.elements import BeamElement, QuadraticBeamElement, create_element
    from anyfesolver.ge_beam3_element import GeometricallyExactBeam3D3NElement

    package = Path(anyfesolver.__file__).resolve()
    module = Path(sys.modules[GeometricallyExactBeam3D3NElement.__module__].__file__).resolve()
    if not package.is_relative_to(target) or not module.is_relative_to(target):
        raise RuntimeError("installed package origin escaped target")
    model = anyfesolver.FEModel("g5-installed")
    model.add_material("mat", 2.1e5, .3, density=2.)
    for node, x in ((1, 0.), (2, 1.), (3, 2.)):
        model.add_node(node, x, 0., 0.)
    section = GeneralizedBeamSection(
        np.diag((1.2e5, 4.e4, 3.5e4, 2.e3, 4.5e3, 5.e3)),
        mass_matrix=np.diag((2., 2., 2., .04, .05, .06)), name="G5_INSTALLED")
    element = create_element("ge-beam3", 1, [1, 2, 3], "mat", section=section,
        reference_orientation=(0., 1., 0.), reference_axis_direction=(1., 0., 0.))
    model.add_element(1, element)
    model.add_boundary_condition(anyfesolver.FixedSupport("fixed", [1]))
    model.add_boundary_condition(anyfesolver.BoundaryCondition("axial", [2, 3],
        {name: 0. for name in ("uy", "uz", "rx", "ry", "rz")}))
    stiffness, _ = anyfesolver.assemble_stiffness_matrix(model)
    mass, _ = anyfesolver.assemble_mass_matrix(model)
    load = anyfesolver.LoadCase("installed")
    load.add_nodal_load(3, forces=(1., 0., 0.))
    modal = anyfesolver.solve_free_vibration(model, num_modes=1)
    transient = anyfesolver.solve_transient_newmark(model,
        anyfesolver.TransientConfig(dt=2.e-4, t_end=6.e-4), base_load_case=load)
    if (type(element) is not GeometricallyExactBeam3D3NElement
            or element.formulation_id != "GE_BEAM3_DC_MIXED_K1_MACRO_V2"
            or type(create_element("beam", 2, [1, 3], "mat")) is not BeamElement
            or type(create_element("quadratic_beam", 3, [1, 2, 3], "mat")) is not QuadraticBeamElement
            or modal.solver_status != "ok" or transient.status != "completed"
            or not np.all(np.isfinite(stiffness.data))
            or not np.all(np.isfinite(mass.data))):
        raise RuntimeError("installed G5 route mismatch")
    write_new(output, {"schema": "GE_BEAM3_G5_INSTALLED_PROBE_V1",
        "package_origin_below_target": True, "class_origin_below_target": True,
        "formulation_id": element.formulation_id, "explicit_selector_only": True,
        "legacy_b2_unchanged": True, "legacy_b3_unchanged": True,
        "reference_stiffness": True, "consistent_mass": True,
        "reference_modal": True, "linear_transient": True,
        "production_qualified": False})


def run(revision, output):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError("external package output required")
    if output.exists():
        raise FileExistsError("exclusive package output required")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                               cwd=ROOT, text=True).strip():
        raise ValueError("clean frozen candidate required")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    if revision != actual:
        raise ValueError("exact HEAD revision required")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    receipts = []
    commands = (
        ("archive", ["git", "archive", "--format=zip",
                     "--output=" + str(output / "source.zip"), revision], ROOT),
    )
    for name, command, cwd in commands:
        directory = output / name; directory.mkdir()
        receipts.append({"name": name, **bounded(command, cwd=cwd,
            stdout=directory / "stdout.log", stderr=directory / "stderr.log")})
    safe_extract(output / "source.zip", output / "source")
    build = output / "build"; build.mkdir()
    receipts.append({"name": "build", **bounded(
        [sys.executable, "-B", "-m", "build", "--wheel", "--no-isolation",
         "--outdir", str(output / "wheels"), str(output / "source")],
        cwd=output, stdout=build / "stdout.log", stderr=build / "stderr.log")})
    wheels = list((output / "wheels").glob("*.whl"))
    if len(wheels) != 1 or not wheels[0].is_file() or wheels[0].is_symlink():
        raise ValueError("exact regular wheel required")
    wheel = wheels[0]; wheel_info = {"filename": wheel.name, **fingerprint(wheel)}
    install = output / "install"; install.mkdir()
    receipts.append({"name": "install", **bounded(
        [sys.executable, "-B", "-m", "pip", "install", "--no-index",
         "--no-deps", "--no-compile", "--target", str(output / "target"), str(wheel)],
        cwd=output, stdout=install / "stdout.log", stderr=install / "stderr.log")})
    for name in ("probe-a", "probe-b"):
        directory = output / name; directory.mkdir()
        receipts.append({"name": name, **bounded(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--probe",
             str(output / "target"), str(directory / "science.json")],
            cwd=output, stdout=directory / "stdout.log", stderr=directory / "stderr.log")})
    left = (output / "probe-a" / "science.json").read_bytes()
    right = (output / "probe-b" / "science.json").read_bytes()
    if left != right:
        raise ValueError("installed probe replicas disagree")
    write_new(output / "complete.json", {
        "schema": "GE_BEAM3_G5_INSTALLED_ARTIFACT_V1", "revision": revision,
        "wheel": wheel_info, "probe": {"bytes": len(left),
            "sha256": sha256(left).hexdigest(), "replicas_byte_identical": True},
        "receipts": receipts, "all_children_terminal": True,
        "production_qualified": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe", nargs=2, metavar=("TARGET", "OUTPUT"))
    args = parser.parse_args()
    if args.probe:
        probe(*args.probe); return 0
    if not args.revision or args.output is None:
        parser.error("--revision and --output are required")
    run(args.revision, args.output); return 0


if __name__ == "__main__":
    raise SystemExit(main())
