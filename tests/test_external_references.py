"""Executable CalculiX reference validation without requiring a ccx install."""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

import anyfesolver
from anyfesolver.external_references import (
    generate_external_reference_cases,
    generate_external_reference_report,
    run_calculix_reference_case,
)
from scripts.run_fe_verification import _commands


def _scratch() -> Path:
    root = Path.cwd() / ".pytest_local_tmp"
    root.mkdir(exist_ok=True)
    path = root / f"external_references_{uuid.uuid4().hex}"
    path.mkdir()
    return path


def _deck_records(path: Path) -> tuple[dict[int, tuple[float, float, float]], list[list[str]], list[list[str]]]:
    nodes: dict[int, tuple[float, float, float]] = {}
    boundaries: list[list[str]] = []
    loads: list[list[str]] = []
    section = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        upper = line.upper()
        if upper == "*NODE":
            section = "node"
            continue
        if upper.startswith("*BOUNDARY"):
            section = "boundary"
            continue
        if upper == "*CLOAD":
            section = "cload"
            continue
        if upper.startswith("*"):
            section = ""
            continue
        if not line or line.startswith("**"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if section == "node" and len(fields) >= 4:
            nodes[int(fields[0])] = tuple(float(value) for value in fields[1:4])
        elif section == "boundary":
            boundaries.append(fields)
        elif section == "cload":
            loads.append(fields)
    return nodes, boundaries, loads


def test_external_reference_execution_api_is_available_from_package_root() -> None:
    names = {
        "CALCULIX_EXECUTABLE_ENV",
        "DEFAULT_CALCULIX_RUN_PATH",
        "CalculixParsedResults",
        "calculix_solver_provenance",
        "evaluate_calculix_comparisons",
        "merge_calculix_results",
        "parse_calculix_dat",
        "parse_calculix_frd",
        "resolve_calculix_executable",
        "run_calculix_reference_case",
    }
    assert names <= set(anyfesolver.__all__)
    assert all(hasattr(anyfesolver, name) for name in names)


def test_verification_command_only_executes_calculix_when_explicitly_requested(tmp_path: Path) -> None:
    deck_only = dict(_commands(tmp_path, False, True, True, ("reference",)))["external_reference_report"]
    assert "--execute" not in deck_only
    assert "--calculix" not in deck_only

    executable = tmp_path / "ccx.exe"
    executed = dict(
        _commands(
            tmp_path,
            False,
            True,
            True,
            ("reference",),
            execute_calculix=True,
            calculix_executable=executable,
            calculix_args=("--wrapper-option",),
            calculix_timeout=42.0,
        )
    )["external_reference_report"]
    assert executed[executed.index("--calculix") + 1] == str(executable)
    assert executed[executed.index("--timeout") + 1] == "42.0"
    assert executed[executed.index("--calculix-arg") + 1] == "--wrapper-option"
    assert "--execute" in executed

    beam_shell_pipeline = _commands(
        tmp_path,
        False,
        True,
        True,
        ("beam_shell",),
        execute_calculix=True,
        calculix_executable=executable,
    )
    command_names = [name for name, _command in beam_shell_pipeline]
    assert command_names.index("external_reference_report") < command_names.index("beam_shell_verification_report")
    beam_shell_command = dict(beam_shell_pipeline)["beam_shell_verification_report"]
    report_argument = beam_shell_command[beam_shell_command.index("--external-reference-report") + 1]
    assert report_argument == str(tmp_path / "external_reference_report.json")


def test_generated_decks_have_active_reference_physics_and_analytical_observables() -> None:
    root = _scratch()
    try:
        cases = {case.name: case for case in generate_external_reference_cases(root / "decks")}
        plate = cases["pressure_plate_s4"]
        plate_nodes, plate_boundaries, _plate_loads = _deck_records(plate.inp_path)
        fixed_uz = {int(row[0]) for row in plate_boundaries if int(row[1]) <= 3 <= int(row[2])}
        center_node = next(
            node_id
            for node_id, (x, y, z) in plate_nodes.items()
            if x == pytest.approx(1.0) and y == pytest.approx(0.5) and z == pytest.approx(0.0)
        )
        assert len(plate_nodes) == 325
        assert center_node not in fixed_uz
        assert set(plate_nodes) - fixed_uz

        column = cases["beam_column_buckling"]
        _column_nodes, column_boundaries, column_loads = _deck_records(column.inp_path)
        fixed_ux = {int(row[0]) for row in column_boundaries if int(row[1]) <= 1 <= int(row[2])}
        assert fixed_ux == {1}
        assert ["11", "1", "-1"] in column_loads
        assert 11 not in fixed_ux

        cylinder = cases["cylinder_s4_pressure"]
        assert cylinder.model_summary["nodes"] == 288
        orthotropic = cases["orthotropic_membrane_s4"]
        orthotropic_text = orthotropic.inp_path.read_text(encoding="utf-8")
        assert "*ELASTIC, TYPE=ENGINEERING CONSTANTS" in orthotropic_text
        assert "*ORIENTATION, NAME=" in orthotropic_text
        assert any(
            item["name"] == "orthotropic_max_abs_ux"
            for item in orthotropic.comparisons
        )
        for case in cases.values():
            assert case.comparisons
            assert all(item["reference"]["kind"] == "analytical" for item in case.comparisons)
    finally:
        shutil.rmtree(root, ignore_errors=True)


_FAKE_CALCULIX = r'''
from __future__ import annotations
import json
import sys
from pathlib import Path

if "--version" in sys.argv:
    print("CalculiX CrunchiX version 2.23-fake")
    raise SystemExit(0)

job = sys.argv[sys.argv.index("-i") + 1]
metadata = json.loads(Path(job + ".json").read_text(encoding="utf-8"))
expected = {item["name"]: float(item["expected"]) for item in metadata["comparisons"]}
mismatch = "--mismatch" in sys.argv

def dataset(lines, name, components, rows):
    lines.append(f" -4  {name} {len(components)} 1")
    for index, component in enumerate(components, start=1):
        lines.append(f" -5  {component} 1 2 {index} 0")
    for node, values in rows:
        lines.append(" -1 %d %s" % (node, " ".join(f"{value:.12E}" for value in values)))
    lines.append(" -3")

if job == "beam_column_buckling":
    factor = expected["column_first_buckling_factor"]
    if mismatch:
        factor *= 2.0
    Path(job + ".dat").write_text(
        " B U C K L I N G   F A C T O R   O U T P U T\n"
        " MODE NO       BUCKLING FACTOR\n"
        f"      1          {factor:.12E}\n",
        encoding="utf-8",
    )
    Path(job + ".frd").write_text("    1CFAKE\n", encoding="utf-8")
else:
    lines = ["    1CFAKE", "    2C"]
    if job == "pressure_plate_s4":
        displacement = expected["plate_max_abs_uz"] * (2.0 if mismatch else 1.0)
        stress = expected["plate_max_von_mises"]
        _balance = expected["plate_nodal_force_balance_z"]
        coordinates = [(1, (0.0, 0.0, 0.0)), (2, (1.0, 0.5, 0.0))]
        displacements = [(1, (0.0, 0.0, 0.0)), (2, (0.0, 0.0, -displacement))]
        stresses = [(1, (stress, 0.0, 0.0, 0.0, 0.0, 0.0)), (2, (stress, 0.0, 0.0, 0.0, 0.0, 0.0))]
        reactions = [(1, (0.0, 0.0, 12.5)), (2, (0.0, 0.0, -12.5))]
    elif job == "cylinder_s4_pressure":
        displacement = expected["cylinder_mean_abs_radial_displacement"] * (2.0 if mismatch else 1.0)
        stress = expected["cylinder_median_von_mises"]
        coordinates = [(1, (1.0, 0.0, 1.0)), (2, (-1.0, 0.0, 1.0))]
        displacements = [(1, (displacement, 0.0, 0.0)), (2, (-displacement, 0.0, 0.0))]
        stresses = [(1, (stress, 0.0, 0.0, 0.0, 0.0, 0.0)), (2, (stress, 0.0, 0.0, 0.0, 0.0, 0.0))]
        reactions = []
    else:
        ux = expected["orthotropic_max_abs_ux"] * (2.0 if mismatch else 1.0)
        uy = expected["orthotropic_max_abs_uy"] * (2.0 if mismatch else 1.0)
        stress = expected["orthotropic_max_von_mises"] * (2.0 if mismatch else 1.0)
        coordinates = [(1, (0.0, 0.0, 0.0)), (2, (2.0, 1.0, 0.0))]
        displacements = [(1, (0.0, 0.0, 0.0)), (2, (ux, -uy, 0.0))]
        stresses = [(1, (stress, 0.0, 0.0, 0.0, 0.0, 0.0)), (2, (stress, 0.0, 0.0, 0.0, 0.0, 0.0))]
        reactions = [(1, (-100000.0, 0.0, 0.0)), (2, (100000.0, 0.0, 0.0))]
    for node, values in coordinates:
        lines.append(" -1 %d %s" % (node, " ".join(f"{value:.12E}" for value in values)))
    lines.append(" -3")
    dataset(lines, "DISP", ("D1", "D2", "D3"), displacements)
    dataset(lines, "STRESS", ("SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"), stresses)
    if reactions:
        dataset(lines, "FORC", ("F1", "F2", "F3"), reactions)
    lines.append(" 9999")
    Path(job + ".frd").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(job + ".dat").write_text("CalculiX fake static results\n", encoding="utf-8")
print("CalculiX CrunchiX version 2.23-fake")
'''


def _fake_calculix(root: Path) -> Path:
    script = root / "fake_ccx.py"
    script.write_text(_FAKE_CALCULIX, encoding="utf-8")
    return script


def test_executed_report_requires_parsed_tolerance_controlled_results() -> None:
    root = _scratch()
    try:
        fake = _fake_calculix(root)
        report = generate_external_reference_report(
            root / "decks",
            execute=True,
            calculix_executable=sys.executable,
            calculix_args=(str(fake),),
            run_dir=root / "runs",
            timeout_seconds=10.0,
        )
        assert report["status"] == "passed"
        assert report["validation_performed"] is True
        assert report["solver"]["version"] == "2.23"
        assert [case["validation"]["status"] for case in report["cases"]] == [
            "passed",
            "passed",
            "passed",
            "passed",
        ]
        assert all(
            comparison["status"] == "passed"
            for case in report["cases"]
            for comparison in case["validation"]["comparisons"]
        )
        assert all(Path(case["validation"]["working_directory"]).is_dir() for case in report["cases"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_executed_report_fails_when_an_observable_exceeds_tolerance() -> None:
    root = _scratch()
    try:
        fake = _fake_calculix(root)
        report = generate_external_reference_report(
            root / "decks",
            execute=True,
            calculix_executable=sys.executable,
            calculix_args=(str(fake), "--mismatch"),
            run_dir=root / "runs",
            timeout_seconds=10.0,
        )
        assert report["status"] == "failed"
        statuses = {
            case["name"]: case["validation"]["status"]
            for case in report["cases"]
        }
        assert statuses["pressure_plate_s4"] == "failed"
        assert statuses["beam_column_buckling"] == "failed"
        assert statuses["cylinder_s4_pressure"] == "failed"
        assert statuses["orthotropic_membrane_s4"] == "failed"
        assert any(
            comparison["status"] == "failed"
            for case in report["cases"]
            for comparison in case["validation"]["comparisons"]
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_timeout_is_deterministic_and_cannot_reuse_stale_results() -> None:
    root = _scratch()
    try:
        slow = root / "slow_ccx.py"
        slow.write_text(
            "import time\n"
            "time.sleep(2.0)\n",
            encoding="utf-8",
        )
        case = generate_external_reference_cases(root / "decks")[0]
        stale_dir = root / "runs" / case.name
        stale_dir.mkdir(parents=True)
        (stale_dir / f"{case.name}.frd").write_text("stale", encoding="utf-8")
        validation = run_calculix_reference_case(
            case,
            executable=sys.executable,
            executable_args=(str(slow),),
            run_root=root / "runs",
            timeout_seconds=0.05,
        )
        assert validation["status"] == "execution_failed"
        assert validation["timed_out"] is True
        assert not (stale_dir / f"{case.name}.frd").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_missing_explicit_executable_is_reported_without_execution() -> None:
    root = _scratch()
    try:
        missing = root / "definitely_missing_ccx.exe"
        report = generate_external_reference_report(
            root / "decks",
            execute=True,
            calculix_executable=missing,
            run_dir=root / "runs",
        )
        assert report["status"] == "solver_unavailable"
        assert report["validation_performed"] is False
        assert all(case["validation"]["status"] == "solver_unavailable" for case in report["cases"])
    finally:
        shutil.rmtree(root, ignore_errors=True)
