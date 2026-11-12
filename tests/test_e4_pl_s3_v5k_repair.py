from __future__ import annotations

import importlib.util
import ast
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

from anyfesolver.e4_pl_s3_v2c_element import StrictFlatLinearE4PLS3V2CShellElement
from anyfesolver.fe_core import FEModel
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.s3_v2c_fast_assembly import get_v2c_stiffness_plan


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "docs/reference_cases/e4_pl_s3_v5k_producer.py"
CHECKER = ROOT / "docs/reference_cases/e4_pl_s3_v5k_checker.py"


def _single() -> tuple[FEModel, StrictFlatLinearE4PLS3V2CShellElement]:
    model = FEModel("v5k-single")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.3, 1.1, 0.0)), start=1):
        model.add_node(node_id, *coordinate)
    element = StrictFlatLinearE4PLS3V2CShellElement(
        1,
        (1, 2, 3),
        "steel",
        thickness=0.08,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)
    return model, element


def _csr_identical(left: sparse.csr_matrix, right: sparse.csr_matrix) -> bool:
    return bool(
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data, right.data)
    )


def _scalar_assembly(model: FEModel) -> sparse.csr_matrix:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for element_id, element in model.mesh.elements.items():
        del element_id
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        material = model.get_material(element.material_name)
        matrix = np.asarray(element.compute_stiffness_matrix(model.mesh, material), dtype=np.float64)
        rows.append(np.repeat(dofs, dofs.size))
        cols.append(np.tile(dofs, dofs.size))
        data.append(matrix.ravel())
    return sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(model.mesh.dof_manager.total_dofs, model.mesh.dof_manager.total_dofs),
    ).tocsr()


def test_v2c_plan_is_exact_readonly_and_reused() -> None:
    model, element = _single()
    first, first_reused = get_v2c_stiffness_plan(model, ((1, element),))
    second, second_reused = get_v2c_stiffness_plan(model, ((1, element),))
    assert first_reused is False
    assert second_reused is True
    assert first is second
    assert first.matrices[1].flags.writeable is False
    np.testing.assert_array_equal(
        first.matrices[1],
        element.compute_stiffness_matrix(model.mesh, model.get_material("steel")),
    )


def test_global_plan_is_byte_identical_to_scalar_and_warm() -> None:
    model, _element = _single()
    scalar = _scalar_assembly(model)
    cold, cold_info = assemble_stiffness_matrix(model)
    warm, warm_info = assemble_stiffness_matrix(model)
    assert _csr_identical(scalar, cold)
    assert _csr_identical(cold, warm)
    assert cold_info["diagnostics"]["s3_v2c_exact_stiffness"]["plan_reused"] is False
    assert warm_info["diagnostics"]["s3_v2c_exact_stiffness"]["plan_reused"] is True
    assert warm_info["diagnostics"]["scalar_shell_element_count"] == 0


def test_supported_node_and_material_mutations_invalidate() -> None:
    model, _element = _single()
    first, _first_info = assemble_stiffness_matrix(model)
    assemble_stiffness_matrix(model)
    model.mesh.nodes[3].x = 0.35
    changed_geometry, geometry_info = assemble_stiffness_matrix(model)
    assert not _csr_identical(first, changed_geometry)
    assert geometry_info["diagnostics"]["s3_v2c_exact_stiffness"]["plan_reused"] is False
    model.get_material("steel").elastic_modulus = 190.0e9
    changed_material, material_info = assemble_stiffness_matrix(model)
    assert not _csr_identical(changed_geometry, changed_material)
    assert material_info["diagnostics"]["s3_v2c_exact_stiffness"]["plan_reused"] is False


def test_registered_mixed_route_has_no_scalar_v2c() -> None:
    path = ROOT / "docs/reference_cases/e4_pl_s3_v5i_stage4b.py"
    spec = importlib.util.spec_from_file_location("_v5k_route_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    payload = json.loads((ROOT / "docs/reference_cases/e4_pl_s3_v5i_stage4b_input.json").read_text(encoding="ascii"))
    lane, _runner, authorities = module._lane_and_model(payload)
    built = lane._build_case(authorities, 10, auxiliary=False)
    _cold, _cold_info = assemble_stiffness_matrix(built.model)
    _warm, warm_info = assemble_stiffness_matrix(built.model)
    assert warm_info["diagnostics"]["vectorized_shell_element_count"] == 440
    assert warm_info["diagnostics"]["scalar_shell_element_count"] == 0
    assert any(
        group["kernel"] == "s3_v2c_exact_revision_bound_matrix_plan"
        and group["num_elements"] == 80
        for group in warm_info["diagnostics"]["vectorized_shell_groups"]
    )


def test_spectral_clusters_use_both_acceptance_intervals() -> None:
    spec = importlib.util.spec_from_file_location("_v5k_producer_cluster_test", PRODUCER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    reference = [1.0, 1.2, 2.0, 3.0, 3.001, 4.0, 5.0, 6.0]
    candidate = [1.0, 1.2, 2.0, 3.0, 3.002, 4.0, 5.0, 6.0]
    assert module.spectral_clusters(reference, candidate) == [[0], [1], [2], [3, 4], [5], [6], [7]]


def test_independent_checker_imports_no_production_or_producer() -> None:
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imports)
    assert not any("v5k_producer" in name for name in imports)


def test_assembly_proof_has_byte_identical_checker_replicas(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("_v5k_producer_assembly_test", PRODUCER)
    assert spec is not None and spec.loader is not None
    producer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = producer
    spec.loader.exec_module(producer)
    proof, _diagnostic = producer.produce("ASSEMBLY_10")
    checker_spec = importlib.util.spec_from_file_location("_v5k_checker_test", CHECKER)
    assert checker_spec is not None and checker_spec.loader is not None
    checker = importlib.util.module_from_spec(checker_spec)
    sys.modules[checker_spec.name] = checker
    checker_spec.loader.exec_module(checker)
    first = checker.canonical_bytes(checker.verify(proof))
    second = checker.canonical_bytes(checker.verify(json.loads(producer.canonical_bytes(proof))))
    assert first == second
    assert proof["gate_status"] == "PASS_MEASURED_REGISTERED_SCOPE"
