from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
GITHUB_ROOT = Path(os.environ["ANY_ECOSYSTEM_ROOT"]) if "ANY_ECOSYSTEM_ROOT" in os.environ else next(
    (parent for parent in ROOT.parents if (parent / "ANYgeometry" / "src").is_dir()),
    ROOT.parent,
)
RUNNER_PATH = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_s3_mixed_mesh_qualification_runner.py"
)
INPUT_PATH = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_s3_mixed_mesh_smoke_input.json"
)
SCHEMA_PATH = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_s3_mixed_mesh_runner_schema.json"
)


def _source_paths() -> list[Path]:
    candidates = [
        ROOT / "src",
        GITHUB_ROOT / "ANYgeometry" / "src",
        GITHUB_ROOT / "ANYmaterial" / "src",
        GITHUB_ROOT / "ANYmesh" / "src",
        GITHUB_ROOT / "ANYfileIO" / "src",
    ]
    return [path for path in candidates if path.is_dir()]


for _path in reversed(_source_paths()):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load_runner() -> Any:
    module_name = "e4_pl_s3_mixed_mesh_qualification_runner_under_test"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


@pytest.fixture(scope="module")
def authorities() -> Any:
    return RUNNER.load_authorities(INPUT_PATH)


@pytest.fixture(scope="module")
def built_cases(authorities: Any) -> list[Any]:
    return [
        RUNNER.build_case_model(authorities, case)
        for case in authorities.input_payload["cases"]
    ]


def _independent_topology_sha256(built: Any) -> str:
    level = int(built.record["level"])
    digest = hashlib.sha256(f"level:{level}\n".encode("ascii"))
    for element_id, element in built.model.mesh.elements.items():
        kind = "Q4" if len(element.node_ids) == 4 else "S3"
        nodes = ",".join(str(int(node_id)) for node_id in element.node_ids)
        digest.update(f"{int(element_id)}:{kind}:{nodes}\n".encode("ascii"))
    return digest.hexdigest().upper()


def _polygon_area_xy(coordinates: np.ndarray) -> float:
    origin = coordinates[0, :2]
    area = 0.0
    for index in range(1, len(coordinates) - 1):
        first = coordinates[index, :2] - origin
        second = coordinates[index + 1, :2] - origin
        area += 0.5 * abs(float(first[0] * second[1] - first[1] * second[0]))
    return area


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH", "")
    pieces = [str(path) for path in _source_paths()]
    if inherited:
        pieces.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(pieces)
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    return environment


@pytest.fixture(scope="module")
def deterministic_smoke_outputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, bytes, dict[str, Any], dict[str, Any]]:
    cycle_directories = {
        cycle: tmp_path_factory.mktemp(f"mixed_runner_cycle_{cycle}")
        for cycle in (1, 2)
    }

    def execute_cycle(
        cycle: int,
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        directory = cycle_directories[cycle]
        result = directory / "result.json"
        provenance = directory / "provenance.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER_PATH),
                "--input",
                str(INPUT_PATH),
                "--result",
                str(result),
                "--provenance",
                str(provenance),
            ],
            cwd=ROOT,
            env=_subprocess_environment(),
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        return completed, result, provenance

    outcomes: list[
        tuple[
            int,
            subprocess.CompletedProcess[str] | None,
            Path | None,
            Path | None,
            Exception | None,
        ]
    ] = []
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="mixed-smoke-cycle",
    ) as executor:
        futures = {
            cycle: executor.submit(execute_cycle, cycle)
            for cycle in (1, 2)
        }
        for cycle in (1, 2):
            try:
                completed, result, provenance = futures[cycle].result()
            except Exception as exc:  # collect both cycle terminals before failing
                outcomes.append((cycle, None, None, None, exc))
            else:
                outcomes.append((cycle, completed, result, provenance, None))

    payloads: list[tuple[bytes, bytes]] = []
    for cycle, completed, result, provenance, error in outcomes:
        assert error is None, f"mixed smoke cycle {cycle} raised {error!r}"
        assert completed is not None
        assert result is not None
        assert provenance is not None
        assert completed.returncode == 0, completed.stderr
        payloads.append((result.read_bytes(), provenance.read_bytes()))
    assert payloads[0] == payloads[1]
    result_payload = json.loads(payloads[0][0])
    provenance_payload = json.loads(payloads[0][1])
    return payloads[0][0], payloads[0][1], result_payload, provenance_payload


def test_input_is_strict_canonical_and_authority_bound(authorities: Any) -> None:
    raw = INPUT_PATH.read_bytes()
    assert raw == RUNNER._pretty_canonical_bytes(authorities.input_payload)
    assert authorities.input_path == INPUT_PATH.resolve()
    assert hashlib.sha256(authorities.manifest_raw).hexdigest().upper() == (
        authorities.input_payload["authority"]["connectivity_manifest"]["sha256"]
    )
    assert hashlib.sha256(authorities.contract_raw).hexdigest().upper() == (
        authorities.input_payload["authority"]["qualification_contract"]["sha256"]
    )
    assert len(authorities.manifest["records"]) == 252
    assert len(authorities.manifest["research_control"]["records"]) == 12


def test_schema_has_fail_closed_future_extension_slots() -> None:
    schema = json.loads(SCHEMA_PATH.read_bytes())
    assert schema["$id"] == RUNNER.INPUT_SCHEMA
    levels = schema["properties"]["cases"]["items"]["properties"]["topology"]["properties"]["level"]["enum"]
    assert levels == [20, 40, 80, 160]
    executors = schema["properties"]["execution"]["properties"]["executor"]["enum"]
    assert executors == [RUNNER.EXECUTOR_ID, *RUNNER.FUTURE_EXECUTOR_IDS]
    assert set(RUNNER.SUPPORTED_SMOKE_LEVELS) == {20, 40}


def test_constructed_topologies_independently_rehash_and_match_exact_counts(
    built_cases: list[Any],
) -> None:
    expected = {
        "SMOKE_N20_COMPACT_CLUSTER_5PCT_ALTERNATING": (20, 441, 380, 40, 5),
        "SMOKE_N40_DISPERSED_1PCT_ALTERNATING": (40, 1681, 1584, 32, 1),
    }
    for built in built_cases:
        case_id = built.case_spec["case_id"]
        level, nodes, q4_count, s3_count, fraction = expected[case_id]
        assert _independent_topology_sha256(built) == built.record["connectivity_sha256"]
        assert built.topology_sha256 == built.record["connectivity_sha256"]
        assert built.model.mesh.num_nodes == nodes == (level + 1) ** 2
        actual_q4 = sum(len(element.node_ids) == 4 for element in built.model.mesh.elements.values())
        actual_s3 = sum(len(element.node_ids) == 3 for element in built.model.mesh.elements.values())
        assert (actual_q4, actual_s3) == (q4_count, s3_count)
        assert built.record["s3_area_fraction_percent"] == fraction


def test_coordinates_areas_winding_and_owner_normals_are_exact(
    built_cases: list[Any],
) -> None:
    for built in built_cases:
        level = int(built.record["level"])
        coordinate_roundoff = 2.0 * np.finfo(float).eps / level
        triangle_area = 0.0
        sheet_area = 0.0
        for node_id, node in built.model.mesh.nodes.items():
            ordinal = int(node_id) - 1
            i = ordinal % (level + 1)
            j = ordinal // (level + 1)
            np.testing.assert_array_equal(node.coords(), (i / level, j / level, 0.0))
        for element in built.model.mesh.elements.values():
            coordinates = np.asarray(element.get_node_coordinates(built.model.mesh), dtype=float)
            area = _polygon_area_xy(coordinates)
            sheet_area += area
            cross = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
            assert cross[2] > 0.0
            if len(element.node_ids) == 3:
                triangle_area += area
                assert area == pytest.approx(
                    0.5 / level**2,
                    rel=0.0,
                    abs=coordinate_roundoff,
                )
                np.testing.assert_array_equal(element.reference_normal, (0.0, 0.0, 1.0))
                np.testing.assert_array_equal(element.physical_reference_director, (0.0, 0.0, 1.0))
            else:
                assert area == pytest.approx(
                    1.0 / level**2,
                    rel=0.0,
                    abs=coordinate_roundoff,
                )
        assert sheet_area == pytest.approx(1.0, rel=0.0, abs=2.0e-14)
        assert 100.0 * triangle_area == pytest.approx(
            built.record["s3_area_fraction_percent"],
            rel=0.0,
            abs=2.0e-13,
        )


def test_full_model_inputs_and_hashes_are_deterministic(
    authorities: Any,
    built_cases: list[Any],
) -> None:
    for case_spec, built in zip(authorities.input_payload["cases"], built_cases):
        descriptor = built.model_input_descriptor
        assert len(descriptor["coordinates"]) == built.record["node_count"]
        assert len(descriptor["elements"]) == built.record["element_count"]
        assert len(descriptor["boundary_conditions"]) == 3
        assert built.load_case is built.model.load_cases[0]
        applied = np.asarray(list(built.load_case.nodal_loads.values()), dtype=float).sum(axis=0)
        np.testing.assert_allclose(applied, (1000.0, 0.0, 0.0, 0.0, 0.0, 0.0), rtol=0.0, atol=1.0e-13)
        rebuilt = RUNNER.build_case_model(authorities, case_spec)
        assert rebuilt.model_input_sha256 == built.model_input_sha256
        assert RUNNER._sha256(RUNNER._canonical_bytes(descriptor)) == built.model_input_sha256


def test_factories_are_real_qualified_classes_and_s3_remains_opt_in(
    built_cases: list[Any],
) -> None:
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
    from anyfesolver.elements import (
        DEFAULT_Q4_FORMULATION,
        DEFAULT_S3_FORMULATION,
        ShellElement,
        create_shell_element,
    )

    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "legacy-s3"
    for built in built_cases:
        q4 = next(
            element for element in built.model.mesh.elements.values() if len(element.node_ids) == 4
        )
        s3 = next(
            element for element in built.model.mesh.elements.values() if len(element.node_ids) == 3
        )
        assert type(q4) is QualifiedE4PLShellElement
        assert q4.formulation_id == RUNNER.Q4_FORMULATION_ID
        assert type(s3) is QualifiedE4PLS3ShellElement
        assert s3.formulation_id == RUNNER.S3_FORMULATION_ID
    implicit_s3 = create_shell_element(999999, [1, 2, 3], "smoke_steel")
    assert type(implicit_s3) is ShellElement
    assert not isinstance(implicit_s3, QualifiedE4PLS3ShellElement)


def test_real_n20_n40_smoke_is_byte_identical_and_never_claims_a_gate(
    authorities: Any,
    deterministic_smoke_outputs: tuple[bytes, bytes, dict[str, Any], dict[str, Any]],
) -> None:
    result_raw, provenance_raw, result, provenance = deterministic_smoke_outputs
    assert result_raw == RUNNER._canonical_bytes(result)
    assert provenance_raw == RUNNER._canonical_bytes(provenance)
    assert result["classification"] == RUNNER.CLASSIFICATION
    assert result["qualification_claim"] == "NONE"
    assert result["qualification_decision"] == "NO_QUALIFICATION_OR_DEFAULT_ACTIVATION"
    assert result["all_launched_cases_terminal"] is True
    assert result["case_count"] == 2
    assert result["smoke_terminal"] == "SMOKE_OBSERVATIONS_RECORDED"
    assert result["smoke_failure_case_ids"] == []
    gate_names = sorted(authorities.contract["acceptance_gates"])
    assert result["unexecuted_contract_gates"] == gate_names
    assert result["formal_gate_status"] == {name: "UNEXECUTED" for name in gate_names}
    assert provenance["classification"] == RUNNER.CLASSIFICATION
    assert provenance["default_activation"] == {
        "q4_default": "e4-pl",
        "s3_default": "legacy-s3",
        "s3_qualified_default_activated": False,
    }

    assert [case["level"] for case in result["cases"]] == [20, 40]
    model_input = authorities.input_payload["model"]
    shear_modulus = model_input["material"]["elastic_modulus"] / (
        2.0 * (1.0 + model_input["material"]["poisson_ratio"])
    )
    line_resultant = (
        shear_modulus
        * model_input["patches"]["membrane"]["gamma_xy"]
        * model_input["section"]["thickness"]
    )
    for case in result["cases"]:
        assert case["terminal_status"] == RUNNER.TERMINAL_RECORDED
        assert case["static_probe"]["solver_status"] == "converged"
        assert case["assembly"]["assembled_element_count"] == sum(case["element_counts"].values())
        assert case["assembly"]["matrix_nnz"] > 0
        assert case["symmetry_relative_frobenius_residual"] < 1.0e-12
        assert case["covariance"]["relative_frobenius_residual"] < 1.0e-12
        assert case["patches"]["membrane"]["patch_residual"] < 1.0e-10
        assert case["patches"]["bending"]["patch_residual"] < 1.0e-8
        assert case["patches"]["shear"]["patch_residual"] > 1.0e-3
        assert case["patches"]["shear"]["maximum_s3_bubble_rotation_norm"] > 0.0
        assert case["patches"]["shear"]["interpretation"] == (
            "NONCLASSIFYING_AFFINE_TRANSVERSE_SHEAR_TRACE_DIAGNOSTIC"
        )
        assert case["patches"]["shear"]["diagnostic_classification"] == (
            "NONCLASSIFYING_NOT_THE_PUBLISHED_FORCE_LOADED_PATCH"
        )
        coupling = case["patches"]["shear"]["s3_bubble_coupling"]
        assert coupling["cause"].startswith("HIERARCHICAL_BUBBLE_SCHUR_RELAXATION_")
        assert coupling["maximum_force_coupling_ratio"] == pytest.approx(
            0.5,
            rel=0.0,
            abs=1.0e-13,
        )
        assert coupling["maximum_mean_shear_operator_frobenius"] == pytest.approx(
            np.sqrt(0.5),
            rel=0.0,
            abs=1.0e-13,
        )
        assert coupling["maximum_equilibrated_bubble_residual"] < 1.0e-12
        assert coupling["uncondensed_trace_stress_residual"] < 1.0e-12
        assert coupling["kinematic_decomposition_bubble_residual"] < 1.0e-12
        assert coupling["kinematic_decomposition_stress_residual"] < 1.0e-12
        assert coupling["recovery_vs_condensed_stiffness_energy_residual"] < 1.0e-12
        assert coupling["relaxation_energy_j"] > 0.0

        force_patch = case["force_loaded_in_plane_shear_patch"]
        assert force_patch["classification"] == "SMOKE_DIAGNOSTIC_NOT_FORMAL_GATE"
        assert force_patch["interpretation"] == (
            "VALID_FORCE_LOADED_CONSTANT_IN_PLANE_SHEAR_PATCH"
        )
        assert force_patch["solver_status"] == "converged"
        assert force_patch["support_dof_count"] == 6
        assert force_patch["support_node_id"] == 1
        assert force_patch["boundary_loaded_node_count"] == 4 * case["level"]
        expected_edge_resultants = {
            "bottom": (-line_resultant, 0.0, 0.0),
            "left": (0.0, -line_resultant, 0.0),
            "right": (0.0, line_resultant, 0.0),
            "top": (line_resultant, 0.0, 0.0),
        }
        for edge, expected in expected_edge_resultants.items():
            np.testing.assert_allclose(
                force_patch["boundary_traction_resultants"][edge],
                expected,
                rtol=0.0,
                atol=5.0e-11,
            )
        assert force_patch["patch_residual"] < 1.0e-10
        assert max(force_patch["topology_patch_residual"].values()) < 1.0e-10
        assert force_patch["force_residual"] < 1.0e-10
        assert force_patch["action_reaction_residual"] < 1.0e-10
        assert force_patch["edge_work_residual"] < 1.0e-10
        assert force_patch["work"]["continuum_work_residual"] < 1.0e-10
        assert force_patch["exact_affine_solution_relative_residual"] < 1.0e-8
        assert set(force_patch["pl_participation"]) == {
            "Q4_PL",
            "Q4_RESIDUAL_HOURGLASS",
            "S3_PL",
        }
        for patch in case["patches"].values():
            assert set(patch["pl_participation"]) == {
                "Q4_PL",
                "Q4_RESIDUAL_HOURGLASS",
                "S3_PL",
            }
            assert patch["work"]["assembled_internal_work_j"] != 0.0
            assert patch["work"]["element_internal_work_j"] != 0.0
    assert result["mechanics_scope"]["force_loaded_in_plane_shear_patch"].startswith(
        "EXECUTED_VALID_CONSTANT_STRESS_PATCH_"
    )
    assert result["mechanics_scope"]["transverse_shear_affine_trace"].startswith(
        "EXECUTED_NONCLASSIFYING_DIAGNOSTIC:"
    )
    assert result["mechanics_scope"]["transverse_shear_force_loaded_patch"].startswith(
        "UNEXECUTED_"
    )


def test_duplicate_noncanonical_and_connectivity_mutations_fail_before_output(
    tmp_path: Path,
) -> None:
    original = INPUT_PATH.read_bytes()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(
        original.replace(
            b"{\n",
            b'{\n  "schema": "duplicate-must-fail",\n',
            1,
        )
    )
    with pytest.raises(RUNNER.CampaignInputError, match="duplicate key 'schema'"):
        RUNNER.load_authorities(duplicate)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(original + b"\n")
    with pytest.raises(RUNNER.CampaignInputError, match="not canonical pretty JSON"):
        RUNNER.load_authorities(noncanonical)

    payload = copy.deepcopy(json.loads(original))
    payload["cases"][0]["topology"]["connectivity_sha256"] = "0" * 64
    mutated = tmp_path / "mutated.json"
    mutated.write_bytes(RUNNER._pretty_canonical_bytes(payload))
    result = tmp_path / "must-not-exist-result.json"
    provenance = tmp_path / "must-not-exist-provenance.json"
    with pytest.raises(RUNNER.CampaignInputError, match="connectivity digest disagrees"):
        RUNNER.execute_to_paths(mutated, result, provenance)
    assert not result.exists()
    assert not provenance.exists()


def test_mechanics_failure_waits_for_every_terminal_before_canonical_emission(
    authorities: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "terminal-result.json"
    provenance_path = tmp_path / "terminal-provenance.json"
    launched: list[str] = []

    def fake_run(_authorities: Any, case: dict[str, Any]) -> dict[str, Any]:
        launched.append(case["case_id"])
        assert not result_path.exists()
        assert not provenance_path.exists()
        if len(launched) == 1:
            raise RuntimeError("deterministic injected mechanics failure")
        return {
            "case_id": case["case_id"],
            "classification": RUNNER.CLASSIFICATION,
            "terminal_status": RUNNER.TERMINAL_RECORDED,
        }

    monkeypatch.setattr(RUNNER, "_run_case", fake_run)
    return_code = RUNNER.execute_to_paths(INPUT_PATH, result_path, provenance_path)
    assert launched == [case["case_id"] for case in authorities.input_payload["cases"]]
    assert return_code == 1
    result_raw = result_path.read_bytes()
    provenance_raw = provenance_path.read_bytes()
    result = json.loads(result_raw)
    provenance = json.loads(provenance_raw)
    assert result_raw == RUNNER._canonical_bytes(result)
    assert provenance_raw == RUNNER._canonical_bytes(provenance)
    assert result["all_launched_cases_terminal"] is True
    assert result["smoke_failure_case_ids"] == [launched[0]]
    assert [case["terminal_status"] for case in result["cases"]] == [
        RUNNER.TERMINAL_FAILED,
        RUNNER.TERMINAL_RECORDED,
    ]
