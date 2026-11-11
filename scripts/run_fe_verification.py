"""Run the local FE solver verification gate and write JSON/Markdown reports."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _git_sha() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _dependency_versions(names: Iterable[str]) -> Dict[str, str | None]:
    versions: Dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _run_command(name: str, command: List[str]) -> Dict[str, Any]:
    start = time.perf_counter()
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(SRC), existing_pythonpath) if value
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    stdout = completed.stdout[-8000:]
    stderr = completed.stderr[-8000:]
    return {
        "name": name,
        "command": command,
        "returncode": int(completed.returncode),
        "status": "passed" if completed.returncode == 0 else "failed",
        "elapsed_seconds": float(elapsed),
        "stdout_tail": stdout,
        "stderr_tail": stderr,
    }


def _pytest_command(*paths: str) -> List[str]:
    return [sys.executable, "-m", "pytest", *paths, "-q", "-p", "no:cacheprovider"]


def _external_reference_command(
    output_dir: Path,
    *,
    execute_calculix: bool = False,
    calculix_executable: Path | None = None,
    calculix_args: Sequence[str] = (),
    calculix_timeout: float = 300.0,
) -> List[str]:
    command = [
        sys.executable,
        "scripts/run_external_references.py",
        "--output",
        str(output_dir / "external_reference_report.json"),
        "--deck-dir",
        str(output_dir / "external_reference_decks"),
        "--markdown",
        str(output_dir / "external_reference_report.md"),
    ]
    if execute_calculix:
        command.extend(
            [
                "--execute",
                "--run-dir",
                str(output_dir / "external_reference_runs"),
                "--timeout",
                str(float(calculix_timeout)),
            ]
        )
        if calculix_executable is not None:
            command.extend(["--calculix", str(calculix_executable)])
        for argument in calculix_args:
            command.extend(["--calculix-arg", str(argument)])
    return command


def _beam_shell_verification_command(output_dir: Path) -> List[str]:
    return [
        sys.executable,
        "scripts/run_beam_shell_verification.py",
        "--output",
        str(output_dir / "beam_shell_verification_report.json"),
        "--markdown",
        str(output_dir / "beam_shell_verification_report.md"),
        "--external-reference-report",
        str(output_dir / "external_reference_report.json"),
    ]


def _family_commands(
    output_dir: Path,
    families: Sequence[str],
    *,
    execute_calculix: bool = False,
    calculix_executable: Path | None = None,
    calculix_args: Sequence[str] = (),
    calculix_timeout: float = 300.0,
) -> List[tuple[str, List[str]]]:
    py = sys.executable
    commands: List[tuple[str, List[str]]] = []
    for family in families:
        if family == "static":
            commands.append(("static_tests", _pytest_command("tests/test_fe_solver_architecture.py", "tests/test_fe_solver_theory.py")))
        elif family == "buckling":
            commands.append(("buckling_tests", _pytest_command("tests/test_fe_solver_buckling.py", "tests/test_beam_axis_conventions.py")))
            commands.append(
                (
                    "buckling_validity_report",
                    [
                        py,
                        "scripts/run_buckling_validity.py",
                        "--output",
                        str(output_dir / "buckling_validity_report.json"),
                        "--markdown",
                        str(output_dir / "buckling_validity_report.md"),
                    ],
                )
            )
        elif family == "nonlinear":
            commands.append(
                (
                    "nonlinear_tests",
                    _pytest_command(
                        "tests/test_fe_solver_nonlinear_static.py",
                        "tests/test_fe_solver_nonlinear_dnv.py",
                        "tests/test_fe_solver_nonlinear_limit_point.py",
                    ),
                )
            )
        elif family == "transient":
            commands.append(("transient_tests", _pytest_command("tests/test_fe_solver_dynamics.py", "tests/test_fe_solver_contact.py")))
        elif family == "performance":
            commands.append(("performance_tests", _pytest_command("tests/test_performance_improvements.py")))
            commands.append(
                (
                    "benchmark_smoke",
                    [
                        py,
                        "scripts/run_fe_benchmarks.py",
                        "--output",
                        str(output_dir / "fe_infrastructure_benchmarks.json"),
                        "--markdown",
                        str(output_dir / "fe_infrastructure_benchmarks.md"),
                    ],
                )
            )
        elif family == "s4":
            commands.append(("s4_validity_tests", _pytest_command("tests/test_fe_solver_shell_verification.py", "tests/test_fe_solver_element_quality.py", "tests/test_fe_solver_s4_validity.py")))
            commands.append(
                (
                    "s4_validity_report",
                    [
                        py,
                        "scripts/run_s4_validity.py",
                        "--output",
                        str(output_dir / "s4_validity_report.json"),
                        "--markdown",
                        str(output_dir / "s4_validity_report.md"),
                    ],
                )
            )
        elif family == "capacity":
            commands.append(("capacity_workflow_tests", _pytest_command("tests/test_fe_solver_capacity_workflow.py")))
            commands.append(
                (
                    "capacity_workflow_smoke",
                    [
                        py,
                        "scripts/run_capacity_workflow.py",
                        "--output",
                        str(output_dir / "capacity_workflow_report.json"),
                        "--steps",
                        "4",
                    ],
                )
            )
        elif family == "beam":
            commands.append(("beam_geometric_tests", _pytest_command("tests/test_beam_axis_conventions.py", "tests/test_fe_solver_corotational_beam.py")))
            commands.append(
                (
                    "beam_validity_report",
                    [
                        py,
                        "scripts/run_beam_validity.py",
                        "--output",
                        str(output_dir / "beam_validity_report.json"),
                        "--markdown",
                        str(output_dir / "beam_validity_report.md"),
                    ],
                )
            )
        elif family == "modal":
            commands.append(("mass_modal_tests", _pytest_command("tests/test_fe_solver_mass_modal.py")))
            commands.append(
                (
                    "mass_modal_validity_report",
                    [
                        py,
                        "scripts/run_mass_modal_validity.py",
                        "--output",
                        str(output_dir / "mass_modal_validity_report.json"),
                        "--markdown",
                        str(output_dir / "mass_modal_validity_report.md"),
                    ],
                )
            )
        elif family == "element":
            commands.append(
                (
                    "element_qualification_tests",
                    _pytest_command(
                        "tests/test_fe_solver_element_quality.py",
                        "tests/test_fe_solver_element_qualification.py",
                        "tests/test_beam_axis_conventions.py",
                    ),
                )
            )
            commands.append(
                (
                    "element_qualification_report",
                    [
                        py,
                        "scripts/run_element_qualification.py",
                        "--output",
                        str(output_dir / "element_qualification_report.json"),
                        "--markdown",
                        str(output_dir / "element_qualification_report.md"),
                    ],
                )
            )
        elif family == "plasticity":
            commands.append(
                (
                    "plasticity_tangent_tests",
                    _pytest_command(
                        "tests/test_fe_solver_nonlinear_static.py",
                        "tests/test_fe_solver_nonlinear_dnv.py",
                        "tests/test_fe_solver_plasticity_qualification.py",
                    ),
                )
            )
            commands.append(
                (
                    "plasticity_qualification_report",
                    [
                        py,
                        "scripts/run_plasticity_qualification.py",
                        "--output",
                        str(output_dir / "plasticity_qualification_report.json"),
                        "--markdown",
                        str(output_dir / "plasticity_qualification_report.md"),
                    ],
                )
            )
        elif family == "recovery":
            commands.append(("recovery_policy_tests", _pytest_command("tests/test_fe_solver_recovery_policy.py")))
            commands.append(
                (
                    "recovery_policy_report",
                    [
                        py,
                        "scripts/run_recovery_policy.py",
                        "--output",
                        str(output_dir / "recovery_policy_report.json"),
                        "--markdown",
                        str(output_dir / "recovery_policy_report.md"),
                    ],
                )
            )
        elif family == "reference":
            commands.append(("reference_case_tests", _pytest_command("tests/test_fe_solver_reference_cases.py")))
            commands.append(
                (
                    "external_reference_report",
                    _external_reference_command(
                        output_dir,
                        execute_calculix=execute_calculix,
                        calculix_executable=calculix_executable,
                        calculix_args=calculix_args,
                        calculix_timeout=calculix_timeout,
                    ),
                )
            )
        elif family == "beam_shell":
            commands.append(("beam_shell_verification_tests", _pytest_command("tests/test_beam_shell_verification.py")))
            commands.append(
                (
                    "beam_shell_verification_report",
                    _beam_shell_verification_command(output_dir),
                )
            )
        elif family == "mesh_load_bc":
            commands.append(("mesh_load_bc_verification_tests", _pytest_command("tests/test_mesh_load_bc_verification.py")))
            commands.append(
                (
                    "mesh_load_bc_verification_report",
                    [
                        py,
                        "scripts/run_mesh_load_bc_verification.py",
                        "--output",
                        str(output_dir / "mesh_load_bc_verification_report.json"),
                        "--markdown",
                        str(output_dir / "mesh_load_bc_verification_report.md"),
                    ],
                )
            )
    return commands


def _commands(
    output_dir: Path,
    quick: bool,
    skip_full: bool,
    skip_qc: bool,
    families: Sequence[str],
    *,
    execute_calculix: bool = False,
    calculix_executable: Path | None = None,
    calculix_args: Sequence[str] = (),
    calculix_timeout: float = 300.0,
) -> List[tuple[str, List[str]]]:
    py = sys.executable
    import_check = [
        py,
        "-c",
        "import anyfesolver; missing=[n for n in anyfesolver.__all__ if not hasattr(anyfesolver,n)]; "
        "assert not missing, missing; print(f'imported {len(anyfesolver.__all__)} public FE symbols')",
    ]
    commands: List[tuple[str, List[str]]] = [("public_imports", import_check)]
    if quick:
        commands.append(("baseline_generation_smoke", [py, "scripts/generate_fe_baselines.py", "--output", str(output_dir / "quick_baseline.json"), "--no-timing"]))
        if execute_calculix:
            commands.append(
                (
                    "external_reference_report",
                    _external_reference_command(
                        output_dir,
                        execute_calculix=True,
                        calculix_executable=calculix_executable,
                        calculix_args=calculix_args,
                        calculix_timeout=calculix_timeout,
                    ),
                )
            )
        return commands
    if families:
        if execute_calculix and "reference" not in families:
            commands.append(
                (
                    "external_reference_report",
                    _external_reference_command(
                        output_dir,
                        execute_calculix=True,
                        calculix_executable=calculix_executable,
                        calculix_args=calculix_args,
                        calculix_timeout=calculix_timeout,
                    ),
                )
            )
        commands.extend(
            _family_commands(
                output_dir,
                families,
                execute_calculix=execute_calculix,
                calculix_executable=calculix_executable,
                calculix_args=calculix_args,
                calculix_timeout=calculix_timeout,
            )
        )
        return commands
    commands.append(
        (
            "focused_fe_tests",
            [
                py,
                "-m",
                "pytest",
                "tests/test_fe_solver_architecture.py",
                "tests/test_fe_solver_dynamics.py",
                "tests/test_fe_solver_contact.py",
                "tests/test_fe_solver_buckling.py",
                "tests/test_fe_solver_infrastructure.py",
                "tests/test_fe_solver_s4_validity.py",
                "tests/test_fe_solver_capacity_workflow.py",
                "tests/test_fe_solver_corotational_beam.py",
                "tests/test_fe_solver_mass_modal.py",
                "tests/test_fe_solver_element_qualification.py",
                "tests/test_fe_solver_plasticity_qualification.py",
                "tests/test_fe_solver_recovery_policy.py",
                "tests/test_fe_solver_reference_cases.py",
                "tests/test_beam_shell_verification.py",
                "tests/test_mesh_load_bc_verification.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
        )
    )
    if not skip_full:
        commands.append(("full_pytest", [py, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"]))
    if not skip_qc:
        commands.append(("qc_smoke", [py, "run_qc.py", "--test-cases", "--no-save"]))
    commands.append(("baseline_generation_smoke", [py, "scripts/generate_fe_baselines.py", "--output", str(output_dir / "generated_baseline.json")]))
    commands.append(("baseline_comparison_smoke", [py, "scripts/compare_fe_baselines.py", "--report", str(output_dir / "baseline_compare.json")]))
    commands.append(
        (
            "benchmark_smoke",
            [
                py,
                "scripts/run_fe_benchmarks.py",
                "--output",
                str(output_dir / "fe_infrastructure_benchmarks.json"),
                "--markdown",
                str(output_dir / "fe_infrastructure_benchmarks.md"),
            ],
        )
    )
    commands.append(
        (
            "s4_validity_report",
            [
                py,
                "scripts/run_s4_validity.py",
                "--output",
                str(output_dir / "s4_validity_report.json"),
                "--markdown",
                str(output_dir / "s4_validity_report.md"),
            ],
        )
    )
    commands.append(
        (
            "capacity_workflow_smoke",
            [
                py,
                "scripts/run_capacity_workflow.py",
                "--output",
                str(output_dir / "capacity_workflow_report.json"),
                "--steps",
                "4",
            ],
        )
    )
    commands.append(
        (
            "beam_validity_report",
            [
                py,
                "scripts/run_beam_validity.py",
                "--output",
                str(output_dir / "beam_validity_report.json"),
                "--markdown",
                str(output_dir / "beam_validity_report.md"),
            ],
        )
    )
    commands.append(
        (
            "mass_modal_validity_report",
            [
                py,
                "scripts/run_mass_modal_validity.py",
                "--output",
                str(output_dir / "mass_modal_validity_report.json"),
                "--markdown",
                str(output_dir / "mass_modal_validity_report.md"),
            ],
        )
    )
    commands.append(
        (
            "buckling_validity_report",
            [
                py,
                "scripts/run_buckling_validity.py",
                "--output",
                str(output_dir / "buckling_validity_report.json"),
                "--markdown",
                str(output_dir / "buckling_validity_report.md"),
            ],
        )
    )
    commands.append(
        (
            "element_qualification_report",
            [
                py,
                "scripts/run_element_qualification.py",
                "--output",
                str(output_dir / "element_qualification_report.json"),
                "--markdown",
                str(output_dir / "element_qualification_report.md"),
            ],
        )
    )
    commands.append(
        (
            "plasticity_qualification_report",
            [
                py,
                "scripts/run_plasticity_qualification.py",
                "--output",
                str(output_dir / "plasticity_qualification_report.json"),
                "--markdown",
                str(output_dir / "plasticity_qualification_report.md"),
            ],
        )
    )
    commands.append(
        (
            "recovery_policy_report",
            [
                py,
                "scripts/run_recovery_policy.py",
                "--output",
                str(output_dir / "recovery_policy_report.json"),
                "--markdown",
                str(output_dir / "recovery_policy_report.md"),
            ],
        )
    )
    commands.append(
        (
            "external_reference_report",
            _external_reference_command(
                output_dir,
                execute_calculix=execute_calculix,
                calculix_executable=calculix_executable,
                calculix_args=calculix_args,
                calculix_timeout=calculix_timeout,
            ),
        )
    )
    commands.append(
        (
            "beam_shell_verification_report",
            _beam_shell_verification_command(output_dir),
        )
    )
    commands.append(
        (
            "mesh_load_bc_verification_report",
            [
                py,
                "scripts/run_mesh_load_bc_verification.py",
                "--output",
                str(output_dir / "mesh_load_bc_verification_report.json"),
                "--markdown",
                str(output_dir / "mesh_load_bc_verification_report.md"),
            ],
        )
    )
    return commands


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# FE Solver Verification Report",
        "",
        f"- Date: {report['date']}",
        f"- Commit: {report.get('commit') or 'unknown'}",
        f"- Platform: {report['environment']['platform']}",
        f"- Python: {report['environment']['python']}",
        f"- Overall status: **{report['status']}**",
        "",
        "## Dependencies",
        "",
    ]
    for name, version in sorted(report["environment"]["dependencies"].items()):
        lines.append(f"- {name}: {version or 'not installed'}")
    lines.extend(["", "## Commands", ""])
    for command in report["commands"]:
        lines.append(f"### {command['name']}")
        lines.append(f"- Status: {command['status']}")
        lines.append(f"- Return code: {command['returncode']}")
        lines.append(f"- Time: {command['elapsed_seconds']:.3f} s")
        lines.append(f"- Command: `{' '.join(command['command'])}`")
        if command.get("stderr_tail"):
            lines.append("")
            lines.append("```text")
            lines.append(command["stderr_tail"].strip())
            lines.append("```")
        lines.append("")
    lines.extend(["## Warnings And Known Limitations", ""])
    for item in report["warnings"]:
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/verification"), help="Report output directory.")
    parser.add_argument("--quick", action="store_true", help="Run import and baseline smoke checks only.")
    parser.add_argument("--skip-full", action="store_true", help="Skip full pytest run.")
    parser.add_argument("--skip-qc", action="store_true", help="Skip run_qc.py smoke test.")
    parser.add_argument("--static", action="store_true", help="Run only static-analysis verification family plus import checks.")
    parser.add_argument("--buckling", action="store_true", help="Run only buckling verification family plus import checks.")
    parser.add_argument("--nonlinear", action="store_true", help="Run only nonlinear verification family plus import checks.")
    parser.add_argument("--transient", action="store_true", help="Run only transient verification family plus import checks.")
    parser.add_argument("--performance", action="store_true", help="Run only performance/benchmark verification family plus import checks.")
    parser.add_argument("--s4", action="store_true", help="Run only S4 shell validity verification family plus import checks.")
    parser.add_argument("--capacity", action="store_true", help="Run only nonlinear capacity workflow verification family plus import checks.")
    parser.add_argument("--beam", action="store_true", help="Run only beam/member geometry verification family plus import checks.")
    parser.add_argument("--modal", action="store_true", help="Run only mass and modal verification family plus import checks.")
    parser.add_argument("--element", action="store_true", help="Run only Q8, beam, and mass element qualification checks plus import checks.")
    parser.add_argument("--plasticity", action="store_true", help="Run only plasticity and nonlinear tangent qualification checks plus import checks.")
    parser.add_argument("--recovery", action="store_true", help="Run only selective recovery/resource-policy checks plus import checks.")
    parser.add_argument("--reference", action="store_true", help="Run only external reference-case/deck checks plus import checks.")
    parser.add_argument("--beam-shell", action="store_true", help="Run only manifest-driven beam/shell verification checks plus import checks.")
    parser.add_argument("--mesh-load-bc", action="store_true", help="Run only mesh/load/boundary-condition verification checks plus import checks.")
    parser.add_argument(
        "--execute-calculix",
        action="store_true",
        help="Execute CalculiX reference cases and require parsed tolerance-controlled comparisons.",
    )
    parser.add_argument(
        "--calculix",
        type=Path,
        default=None,
        help="Explicit ccx executable path (otherwise ANYSOLVER_CALCULIX_EXECUTABLE/PATH discovery is used).",
    )
    parser.add_argument(
        "--calculix-arg",
        action="append",
        default=[],
        help="Argument inserted before '-i JOB' for a CalculiX wrapper; repeat as needed.",
    )
    parser.add_argument("--calculix-timeout", type=float, default=300.0, help="Per-case CalculiX timeout in seconds.")
    args = parser.parse_args()
    if not args.execute_calculix and (args.calculix is not None or args.calculix_arg):
        parser.error("--calculix and --calculix-arg require --execute-calculix")
    if args.calculix_timeout <= 0.0:
        parser.error("--calculix-timeout must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    families = [
        name
        for name, enabled in (
            ("static", args.static),
            ("buckling", args.buckling),
            ("nonlinear", args.nonlinear),
            ("transient", args.transient),
            ("performance", args.performance),
            ("s4", args.s4),
            ("capacity", args.capacity),
            ("beam", args.beam),
            ("modal", args.modal),
            ("element", args.element),
            ("plasticity", args.plasticity),
            ("recovery", args.recovery),
            ("reference", args.reference),
            ("beam_shell", args.beam_shell),
            ("mesh_load_bc", args.mesh_load_bc),
        )
        if enabled
    ]
    command_results = [
        _run_command(name, command)
        for name, command in _commands(
            output_dir,
            args.quick,
            args.skip_full,
            args.skip_qc,
            families,
            execute_calculix=args.execute_calculix,
            calculix_executable=args.calculix,
            calculix_args=args.calculix_arg,
            calculix_timeout=args.calculix_timeout,
        )
    ]
    status = "passed" if all(item["returncode"] == 0 for item in command_results) else "failed"
    report = {
        "date": started,
        "commit": _git_sha(),
        "status": status,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "dependencies": _dependency_versions(["numpy", "scipy", "pytest", "pypardiso"]),
        },
        "commands": command_results,
        "selected_families": families,
        "external_reference_mode": "calculix" if args.execute_calculix else "deck_only",
        "warnings": [
            (
                "CalculiX execution was not requested; generated external-reference decks are handoff artifacts, not numerical validation."
                if not args.execute_calculix
                else "CalculiX results are external-solver comparisons; DNV qualification remains a separate engineering-assurance activity."
            ),
            "Baseline timing fields are informational and not used for numeric comparison.",
            (
                "Follower pressure, consistent corotational tangents, shell initial-stress stiffness and "
                "history-aware recovery are qualified only by their documented verification cases and limits."
            ),
        ],
    }
    json_path = output_dir / "fe_verification_report.json"
    md_path = output_dir / "fe_verification_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
