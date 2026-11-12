"""Compare guided Q4 transverse local work with independent scalar shear work."""

import json

import numpy as np

from anyfesolver import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.matrix_assembly import assemble_stiffness_matrix


def scalar_work(dx: float, dy: float, rigidity: float) -> np.ndarray:
    result = np.zeros((4, 4))
    points = (-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0))
    for xi in points:
        for eta in points:
            derivative_xi = np.array((
                -(1-eta), 1-eta, 1+eta, -(1+eta),
            ))/4.0
            derivative_eta = np.array((
                -(1-xi), -(1+xi), 1+xi, 1-xi,
            ))/4.0
            gradient_x = derivative_xi*2.0/dx
            gradient_y = derivative_eta*2.0/dy
            result += rigidity*(
                np.outer(gradient_x, gradient_x)
                + np.outer(gradient_y, gradient_y)
            )*dx*dy/4.0
    return result


def check(nx: int, ny: int) -> dict:
    dx = 1.0/nx
    dy = 1.0/ny
    model = FEModel(f"guided_scalar_work_{nx}_{ny}")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node, x, y in (
        (1, 0.0, 0.0), (2, dx, 0.0),
        (3, dx, dy), (4, 0.0, dy),
    ):
        model.add_node(node, x, y, 0.04004)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.05,
        reference_normal=(0.0, 0.0, 1.0),
    ))
    stiffness, _ = assemble_stiffness_matrix(model)
    z = [model.mesh.dof_manager.get_node_dofs(node)[2]
         for node in (1, 2, 3, 4)]
    actual = stiffness[np.ix_(z, z)].toarray()
    rigidity = (5.0/6.0)*1.0e5*0.05/(2.0*(1.0+0.3))
    expected = scalar_work(dx, dy, rigidity)
    return {
        "nx": nx, "ny": ny,
        "relative_frobenius_error": float(
            np.linalg.norm(actual-expected)/np.linalg.norm(expected)
        ),
        "maximum_absolute_error_n_per_m": float(np.max(np.abs(actual-expected))),
        "actual_uniform_translation_work_n_m": float(np.ones(4) @ actual @ np.ones(4)),
        "expected_uniform_translation_work_n_m": float(np.ones(4) @ expected @ np.ones(4)),
    }


if __name__ == "__main__":
    print(json.dumps([check(nx, ny) for nx, ny in (
        (16, 1), (64, 1), (128, 1), (128, 2), (128, 4),
    )], indent=2))
