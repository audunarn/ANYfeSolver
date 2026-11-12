"""Compare advanced S4 production batches with the retained scalar oracle."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver import (  # noqa: E402
    FEModel,
    GeneralizedShellSection,
    Material,
    OrthotropicMaterial,
    assemble_stiffness_matrix,
    create_element,
)
from anyfesolver.nonlinear_performance_bootstrap import (  # noqa: E402
    clear_nonlinear_assembly_cache,
    get_nonlinear_assembly_plan,
    install_nonlinear_performance_optimizations,
)


def _section() -> GeneralizedShellSection:
    return GeneralizedShellSection(
        A=np.array(
            [[1.20e8, 1.8e7, 2.0e6], [1.8e7, 8.5e7, -1.5e6], [2.0e6, -1.5e6, 3.4e7]]
        ),
        B=np.array(
            [[2.0e3, -0.9e3, 0.3e3], [0.2e3, -1.2e3, 0.5e3], [-0.1e3, 0.4e3, 0.8e3]]
        ),
        D=np.array(
            [[1.4e4, 1.1e3, 0.3e3], [1.1e3, 1.0e4, -0.2e3], [0.3e3, -0.2e3, 4.2e3]]
        ),
        As=np.array([[2.8e7, 1.7e6], [1.7e6, 2.1e7]]),
    )


def _build_model(kind: str, nx: int, ny: int) -> FEModel:
    model = FEModel(f"benchmark_{kind}")
    if kind == "orthotropic":
        material = OrthotropicMaterial(
            name="ortho",
            elastic_modulus_1=145.0e9,
            elastic_modulus_2=11.0e9,
            elastic_modulus_3=8.5e9,
            poisson_ratio_12=0.24,
            poisson_ratio_13=0.19,
            poisson_ratio_23=0.28,
            shear_modulus_12=5.2e9,
            shear_modulus_13=4.1e9,
            shear_modulus_23=3.2e9,
            density=1580.0,
        )
        section = None
    else:
        material = Material("carrier", 70.0e9, 0.3, density=2700.0)
        section = _section()
    model.register_material(material)
    node_id = 1
    node_ids = np.empty((ny + 1, nx + 1), dtype=np.int64)
    for row in range(ny + 1):
        for column in range(nx + 1):
            x = float(column)
            y = float(row)
            z = 0.035 * np.sin(0.23 * column) * np.cos(0.31 * row)
            model.add_node(node_id, x, y, z)
            node_ids[row, column] = node_id
            node_id += 1
    element_id = 1
    for row in range(ny):
        for column in range(nx):
            model.add_element(
                element_id,
                create_element(
                    "shell",
                    element_id,
                    [
                        int(node_ids[row, column]),
                        int(node_ids[row, column + 1]),
                        int(node_ids[row + 1, column + 1]),
                        int(node_ids[row + 1, column]),
                    ],
                    material.name,
                    thickness=0.018,
                    material_direction=np.array([1.0, 0.25, 0.08]),
                    material_angle_deg=float((row + 2 * column) % 4) * 15.0,
                    shell_section=section,
                ),
            )
            element_id += 1
    return model


def _scalar_linear(model: FEModel) -> sparse.csr_matrix:
    rows = []
    columns = []
    values = []
    for element in model.mesh.elements.values():
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        matrix = element.compute_stiffness_matrix(
            model.mesh, model.get_material(element.material_name)
        )
        rows.append(np.repeat(dofs, dofs.size))
        columns.append(np.tile(dofs, dofs.size))
        values.append(matrix.reshape(-1))
    return sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(model.mesh.dof_manager.total_dofs,) * 2,
    ).tocsr()


def _scalar_nonlinear(model: FEModel, displacement: np.ndarray):
    total_dofs = model.mesh.dof_manager.total_dofs
    force = np.zeros(total_dofs, dtype=float)
    rows = []
    columns = []
    values = []
    for element in model.mesh.elements.values():
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        local_force, local_tangent, _state = element.compute_nonlinear_response(
            model.mesh,
            model.get_material(element.material_name),
            displacement[dofs],
            tangent=True,
        )
        np.add.at(force, dofs, local_force)
        rows.append(np.repeat(dofs, dofs.size))
        columns.append(np.tile(dofs, dofs.size))
        values.append(local_tangent.reshape(-1))
    tangent = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(total_dofs, total_dofs),
    ).tocsr()
    return force, tangent


def _median(callable_, repeats: int) -> tuple[float, object]:
    timings = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        timings.append(time.perf_counter() - start)
    return float(statistics.median(timings)), result


def _case(kind: str, nx: int, ny: int, repeats: int) -> dict[str, object]:
    model = _build_model(kind, nx, ny)
    install_nonlinear_performance_optimizations()
    clear_nonlinear_assembly_cache(model)
    nonlinear_plan = get_nonlinear_assembly_plan(model, 5)
    rng = np.random.default_rng(5279)
    displacement = rng.normal(
        scale=6.0e-4, size=model.mesh.dof_manager.total_dofs
    )

    # Warm all JIT kernels and persistent geometry outside measurements.
    assemble_stiffness_matrix(model)
    assemble_stiffness_matrix(model)
    nonlinear_plan.assemble(displacement, {}, tangent=True)
    nonlinear_plan.assemble(displacement, {}, tangent=True)
    _scalar_linear(model)
    _scalar_nonlinear(model, displacement)

    scalar_linear_seconds, scalar_linear = _median(
        lambda: _scalar_linear(model), repeats
    )
    batch_linear_seconds, batch_linear_result = _median(
        lambda: assemble_stiffness_matrix(model), repeats
    )
    batch_linear = batch_linear_result[0]

    scalar_nonlinear_seconds, scalar_nonlinear = _median(
        lambda: _scalar_nonlinear(model, displacement), repeats
    )
    batch_nonlinear_seconds, batch_nonlinear = _median(
        lambda: nonlinear_plan.assemble(displacement, {}, tangent=True), repeats
    )
    scalar_force, scalar_tangent = scalar_nonlinear
    batch_force, batch_tangent, _states = batch_nonlinear
    return {
        "kind": kind,
        "elements": int(nx * ny),
        "linear": {
            "scalar_seconds": scalar_linear_seconds,
            "batch_seconds": batch_linear_seconds,
            "speedup": scalar_linear_seconds / batch_linear_seconds,
            "maximum_absolute_error": float(
                np.max(np.abs((batch_linear - scalar_linear).data))
                if (batch_linear - scalar_linear).nnz
                else 0.0
            ),
        },
        "nonlinear": {
            "scalar_seconds": scalar_nonlinear_seconds,
            "batch_seconds": batch_nonlinear_seconds,
            "speedup": scalar_nonlinear_seconds / batch_nonlinear_seconds,
            "maximum_force_error": float(np.max(np.abs(batch_force - scalar_force))),
            "maximum_tangent_error": float(
                np.max(np.abs((batch_tangent - scalar_tangent).data))
                if (batch_tangent - scalar_tangent).nnz
                else 0.0
            ),
        },
        "diagnostics": nonlinear_plan.diagnostics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    result = {
        "schema": "anyfesolver.advanced_s4_batch_benchmark",
        "nx": int(args.nx),
        "ny": int(args.ny),
        "repeats": int(args.repeats),
        "cases": [
            _case("orthotropic", args.nx, args.ny, args.repeats),
            _case("generalized", args.nx, args.ny, args.repeats),
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
