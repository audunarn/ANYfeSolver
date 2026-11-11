from __future__ import annotations

import numpy as np
from scipy import sparse

from anyfesolver.activity import ElementActivity
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.fe_core import FEModel
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.s3_v2d_fast_assembly import V2D_GLOBAL_ASSEMBLY_POLICY_ID


def _model(count: int = 4) -> FEModel:
    model = FEModel("s3-v2d-v6t-global-cache")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    node_id = 1
    for index in range(count):
        x = 2.0 * index
        ids = [node_id, node_id + 1, node_id + 2]
        for current, point in zip(ids, ((x, 0, 0), (x + 1, 0, 0), (x, 1, 0))):
            model.add_node(current, *point)
        model.add_element(
            index + 1,
            NativeParityE4PLS3V2DShellElement(
                index + 1,
                ids,
                "steel",
                thickness=0.02,
                reference_normal=(0.0, 0.0, 1.0),
            ),
        )
        node_id += 3
    return model


def _scalar(model: FEModel) -> sparse.csr_matrix:
    rows, cols, data = [], [], []
    for element in model.mesh.elements.values():
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        matrix = np.asarray(
            element.compute_stiffness_matrix(
                model.mesh, model.get_material(element.material_name)
            ),
            dtype=np.float64,
        )
        rows.append(np.repeat(dofs, dofs.size))
        cols.append(np.tile(dofs, dofs.size))
        data.append(matrix.ravel())
    return sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(model.mesh.dof_manager.total_dofs,) * 2,
    ).tocsr()


def _csr_bytes(matrix: sparse.csr_matrix) -> bytes:
    return matrix.data.tobytes() + matrix.indices.tobytes() + matrix.indptr.tobytes()


def test_global_plan_is_scalar_exact_and_reused() -> None:
    model = _model()
    scalar = _scalar(model)
    cold, cold_info = assemble_stiffness_matrix(model)
    warm, warm_info = assemble_stiffness_matrix(model)
    assert _csr_bytes(cold) == _csr_bytes(scalar)
    assert _csr_bytes(warm) == _csr_bytes(cold)
    assert "s3_v2d_global_stiffness" not in cold_info["diagnostics"]
    assert warm_info["diagnostics"]["s3_v2d_global_stiffness"] == {
        "plan_reused": True,
        "policy_id": V2D_GLOBAL_ASSEMBLY_POLICY_ID,
    }
    assert warm_info["diagnostics"]["s3_v2d_exact_stiffness"]["plan_reused"] is True


def test_material_mutation_invalidates_global_and_element_plans() -> None:
    model = _model()
    first, _first_info = assemble_stiffness_matrix(model)
    cached, cached_info = assemble_stiffness_matrix(model)
    assert cached_info["diagnostics"]["s3_v2d_global_stiffness"]["plan_reused"] is True
    model.materials["steel"].elastic_modulus *= 1.01
    changed, changed_info = assemble_stiffness_matrix(model)
    assert "s3_v2d_global_stiffness" not in changed_info["diagnostics"]
    assert changed_info["diagnostics"]["s3_v2d_exact_stiffness"]["plan_reused"] is False
    assert _csr_bytes(first) != _csr_bytes(changed)


def test_nonzero_activity_bypasses_global_plan() -> None:
    model = _model()
    assemble_stiffness_matrix(model)
    assemble_stiffness_matrix(model)
    model.set_element_activity(
        ElementActivity(list(model.mesh.elements), [0.5] * len(model.mesh.elements))
    )
    active, info = assemble_stiffness_matrix(model)
    assert active.shape[0] == model.mesh.dof_manager.total_dofs
    assert "s3_v2d_global_stiffness" not in info["diagnostics"]
