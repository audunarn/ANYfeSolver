"""Reference case discovery tests for CalculiX/PrePoMax benchmarks."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from anyfesolver import (
    FEModel,
    classify_reference_case_from_nodes,
    compare_shell_benchmark_to_reference,
    discover_calculix_reference_cases,
    discover_calculix_shell_convergence_tables,
    generate_external_reference_cases,
    generate_external_reference_report,
    parse_calculix_shell_convergence_file,
    run_simple_supported_shell_benchmark,
    run_simple_supported_shell_convergence,
    write_calculix_input_deck,
    write_internal_shell_convergence_table,
    upstream_calculix_reference_manifest,
    upstream_calculix_shell_reference_values,
    write_external_reference_report,
)
from anyfesolver.boundary import LoadCase
from anyfesolver.elements import BeamElement


def _make_reference_temp_dir() -> Path:
    """Create disposable scratch outside the repository using the temp policy.

    A runner can set TMP/TEMP to an accessible external evidence root on
    Windows. Existing repository-local evidence is never reused or removed.
    """
    return Path(tempfile.mkdtemp(prefix="anyfesolver-reference-cases-"))


def test_reference_case_discovery_finds_local_inp_frd_pair() -> None:
    repo_root = _make_reference_temp_dir()
    try:
        case_dir = repo_root / "tests" / "reference_cases" / "flat_plate"
        case_dir.mkdir(parents=True)
        (case_dir / "flat_plate.inp").write_text(
            """
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
*ELEMENT, TYPE=S4, ELSET=Eall
1, 1, 2, 3, 4
""".strip(),
            encoding="utf-8",
        )
        (case_dir / "flat_plate.frd").write_text("    1C\n", encoding="utf-8")
        (case_dir / "flat_plate.json").write_text(
            json.dumps({"name": "flat plate smoke reference"}),
            encoding="utf-8",
        )

        cases = discover_calculix_reference_cases(repo_root=repo_root)

        assert len(cases) == 1
        case = cases[0]
        assert case.name == "flat plate smoke reference"
        assert case.kind == "flat_plate"
        assert case.node_count == 4
        assert case.element_count == 1
        assert case.has_results is True
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_reference_case_discovery_can_report_input_only_cases() -> None:
    repo_root = _make_reference_temp_dir()
    try:
        case_dir = repo_root / "reference_cases"
        case_dir.mkdir(parents=True)
        (case_dir / "input_only.inp").write_text(
            """
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
*ELEMENT, TYPE=B31, ELSET=Eall
1, 1, 2
""".strip(),
            encoding="utf-8",
        )

        assert discover_calculix_reference_cases(repo_root=repo_root, require_frd=True) == []
        cases = discover_calculix_reference_cases(repo_root=repo_root, require_frd=False)

        assert len(cases) == 1
        assert cases[0].name == "input_only"
        assert cases[0].has_results is False
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_generated_external_reference_decks_are_discoverable_input_cases() -> None:
    repo_root = _make_reference_temp_dir()
    try:
        deck_dir = repo_root / "reference_cases"
        cases = generate_external_reference_cases(deck_dir)
        assert [case.name for case in cases] == [
            "pressure_plate_s4",
            "beam_column_buckling",
            "cylinder_s4_pressure",
            "orthotropic_membrane_s4",
        ]
        assert all(case.inp_path.exists() for case in cases)
        assert all(case.metadata_path.exists() for case in cases)

        discovered = discover_calculix_reference_cases(roots=[deck_dir], repo_root=repo_root, require_frd=False)
        assert len(discovered) == 4
        kinds = {case.name: case.kind for case in discovered}
        assert kinds["pressure_plate_s4"] == "flat_plate"
        assert kinds["cylinder_s4_pressure"] == "cylinder"
        assert kinds["orthotropic_membrane_s4"] == "flat_plate"
        assert all(case.element_count > 0 for case in discovered)
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_calculix_gravity_uses_magnitude_unit_direction_and_defined_all_set() -> None:
    repo_root = _make_reference_temp_dir()
    try:
        model = FEModel("gravity_export")
        model.add_material("steel", 210.0e9, 0.3, density=7850.0)
        for node_id, x in enumerate((0.0, 1.0, 2.0), start=1):
            model.add_node(node_id, x, 0.0, 0.0)
        section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
        model.add_element(7, BeamElement(7, [1, 2], "steel", section))
        model.add_element(42, BeamElement(42, [2, 3], "steel", section))
        gravity = LoadCase("inclined_gravity")
        gravity.set_gravity(3.0, 4.0, 0.0)

        deck_path = repo_root / "gravity.inp"
        write_calculix_input_deck(model, gravity, deck_path)
        lines = deck_path.read_text(encoding="utf-8").splitlines()

        all_set_index = lines.index("*ELSET, ELSET=ALL")
        assert lines[all_set_index + 1] == "7, 42"
        assert "ALL, GRAV, 5, 0.6, 0.8, 0" in lines
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_external_reference_report_cli_artifacts_are_written() -> None:
    repo_root = _make_reference_temp_dir()
    try:
        output = repo_root / "external_reference_report.json"
        markdown = repo_root / "external_reference_report.md"
        deck_dir = repo_root / "decks"

        report = write_external_reference_report(output, deck_dir=deck_dir, markdown=markdown)

        assert report["status"] == "not_executed"
        assert report["validation_performed"] is False
        assert len(report["cases"]) == 4
        assert all(case["validation"]["status"] == "not_executed" for case in report["cases"])
        assert output.exists()
        assert markdown.exists()
        assert (deck_dir / "pressure_plate_s4.inp").exists()
        assert "not numerical validation" in report["known_limitations"][0]
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_reference_case_classification_detects_cylinder_nodes() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    z = np.array([0.0, 1.0])
    nodes = []
    for zi in z:
        for angle in theta:
            nodes.append((np.cos(angle), np.sin(angle), zi))

    assert classify_reference_case_from_nodes(np.asarray(nodes, dtype=float)) == "cylinder"


def test_upstream_calculix_shell_reference_manifest_is_available() -> None:
    manifest = upstream_calculix_reference_manifest()
    names = {entry["name"] for entry in manifest}

    assert "calculix_examples_shell_convergence" in names
    shell_case = next(entry for entry in manifest if entry["name"] == "calculix_examples_shell_convergence")
    assert shell_case["repository"] == "calculix/CalculiX-Examples"
    assert shell_case["directory"] == "Elements/Shell"
    assert shell_case["requires_generated_includes"] is True
    assert "U" in shell_case["expected_outputs"]
    assert "S" in shell_case["expected_outputs"]
    assert shell_case["reference_values"] == {"sref": 1.848, "wref": 0.0587}


def test_shell_convergence_result_file_is_parsed_and_normalized() -> None:
    repo_root = _make_reference_temp_dir()
    try:
        result_file = repo_root / "S4.txt"
        result_file.write_text(
            """
# size NoN smax umax
100 12 0.616489 0.013121
50 20 1.2899 0.027729
""".strip(),
            encoding="utf-8",
        )

        table = parse_calculix_shell_convergence_file(result_file)
        references = upstream_calculix_shell_reference_values()

        assert table.element_type == "S4"
        assert table.stress_reference == references["sref"]
        assert table.displacement_reference == references["wref"]
        assert len(table.points) == 2
        assert table.finest_point is not None
        assert table.finest_point.size == 50.0
        assert np.isclose(table.points[0].stress_normalized, 0.616489 / references["sref"])
        assert np.isclose(table.points[0].displacement_normalized, 0.013121 / references["wref"])
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_shell_convergence_table_discovery_reads_known_element_files() -> None:
    repo_root = _make_reference_temp_dir()
    try:
        (repo_root / "S4.txt").write_text("# size NoN smax umax\n100 12 0.616489 0.013121\n", encoding="utf-8")
        (repo_root / "S8.txt").write_text("# size NoN smax umax\n50 40 1.500000 0.050000\n", encoding="utf-8")
        (repo_root / "notes.txt").write_text("ignored\n", encoding="utf-8")

        tables = discover_calculix_shell_convergence_tables(repo_root)

        assert [table.element_type for table in tables] == ["S4", "S8"]
        assert [len(table.points) for table in tables] == [1, 1]
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_internal_shell_benchmark_returns_normalized_result() -> None:
    result = run_simple_supported_shell_benchmark(
        length=1.0,
        width=1.0,
        thickness=0.01,
        divisions_x=1,
        divisions_y=1,
        pressure=100.0,
        stress_reference=1.0,
        displacement_reference=1.0,
    )

    assert result.element_type == "S4"
    assert result.node_count == 4
    assert result.element_count == 1
    assert result.solver_status == "converged"
    assert np.isfinite(result.max_out_of_plane_displacement)
    assert np.isfinite(result.max_von_mises_stress)
    assert result.displacement_normalized == result.max_out_of_plane_displacement
    assert result.stress_normalized == result.max_von_mises_stress
    assert result.to_dict()["element_type"] == "S4"


def test_internal_shell_convergence_runner_returns_ordered_points() -> None:
    results = run_simple_supported_shell_convergence(
        divisions=(1, 2),
        length=1.0,
        width=1.0,
        thickness=0.01,
        pressure=100.0,
        stress_reference=1.0,
        displacement_reference=1.0,
    )

    assert [result.divisions_x for result in results] == [1, 2]
    assert [result.element_type for result in results] == ["S4", "S4"]
    assert all(result.solver_status == "converged" for result in results)
    assert results[1].node_count > results[0].node_count


def test_internal_shell_table_writer_and_loose_reference_comparison() -> None:
    repo_root = _make_reference_temp_dir()
    try:
        internal_results = run_simple_supported_shell_convergence(
            divisions=(1, 2),
            length=1.0,
            width=1.0,
            thickness=0.01,
            pressure=100.0,
            stress_reference=1.0,
            displacement_reference=1.0,
        )
        internal_path = write_internal_shell_convergence_table(internal_results, repo_root / "S4_internal.txt")

        lines = internal_path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "# size NoN smax umax s_norm u_norm"
        assert all(len(line.split()) == 6 for line in lines[1:])

        external_path = repo_root / "S4.txt"
        external_path.write_text(
            """
# size NoN smax umax
1.0 4 0.25 0.10
0.5 9 0.50 0.20
""".strip(),
            encoding="utf-8",
        )
        external_table = parse_calculix_shell_convergence_file(
            external_path,
            stress_reference=1.0,
            displacement_reference=1.0,
        )
        comparison = compare_shell_benchmark_to_reference(external_table, internal_results)

        assert comparison.external_element_type == "S4"
        assert comparison.internal_element_type == "S4"
        assert len(comparison.points) == 2
        assert np.isfinite(comparison.max_abs_stress_normalized_delta)
        assert np.isfinite(comparison.max_abs_displacement_normalized_delta)
        assert "Informational only" in comparison.notes[0]
        assert comparison.to_dict()["points"]
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)
