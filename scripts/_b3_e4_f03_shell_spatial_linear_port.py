"""Bounded six-channel flat-reference E4-PL shell station diagnostic.

This is a reference *linearization*, not a finite SO(3) shell pose. The
out-of-plane director tilts are independent of midsurface slopes, while
normal-axis rotation comes from the physical in-plane displacement field.
No numerical drilling coordinate participates in this port.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEMesh, Material

from scripts._b3_e4_f03_shell_trace import ShellPanel


def reference_spatial_trace(panel: ShellPanel, station: float) -> np.ndarray:
    """Return [centroid translation, director tilts, fitted yaw] variation.

    Columns follow the actual six-DOF shell-node order. The four off-grid
    sample hosts and yaw normalization are those of the existing F03 planar
    diagnostic. The reference panel is flat in global XY with +Z director.
    """
    if not isinstance(panel, ShellPanel):
        raise TypeError("F03 shell panel required")
    weights = panel.sample_weights(station)
    mean = weights.mean(axis=0)
    planar = panel.trace(station)
    result = np.zeros((6, 6*panel.node_count))
    for axis in range(3):
        result[axis, axis::6] = mean
    result[3, 3::6] = mean
    result[4, 4::6] = mean
    result[5, 0::6] = planar[2, 0::2]
    result[5, 1::6] = planar[2, 1::2]
    return result


def reference_wrench_pullback(panel: ShellPanel, station: float,
                              wrench: np.ndarray) -> np.ndarray:
    """Map a six-component physical reference wrench to shell coordinates."""
    value = np.asarray(wrench, dtype=float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("finite six-component spatial wrench required")
    return reference_spatial_trace(panel, station).T@value


def physical_shell_stiffness(panel: ShellPanel) -> sparse.csr_matrix:
    """Assemble actual Q4/S3 physical blocks with all six nodal columns.

    This small diagnostic keeps the numerical PL/hourglass blocks separate.
    The physical matrix can be singular on drilling coordinates and must not
    be inverted as a positive-definite global production operator.
    """
    if (not isinstance(panel, ShellPanel) or panel.topology not in
            ("Q4", "S3-V2D") or not 1 <= panel.n_s <= 8 or
            panel.node_count != (panel.n_y+1)*(2*panel.n_s+1) or
            panel.element_count != (2 if panel.topology == "Q4" else 4)*panel.n_s*panel.n_y):
        raise ValueError("bounded Q4/S3 F03 shell panel required")
    nx = 2*panel.n_s
    mesh = FEMesh()
    for j, y in enumerate(np.linspace(-.5, .5, panel.n_y+1)):
        for i in range(nx+1):
            mesh.add_node(1+j*(nx+1)+i, -.5+i/panel.n_s, y, 0.)
    material = Material("steel", 210e9, .3, density=7850.)
    rows, columns, values = [], [], []
    element_id = 0
    for j in range(panel.n_y):
        for i in range(nx):
            a = 1+j*(nx+1)+i
            b, d = a+1, a+nx+1
            c = d+1
            patches = ((a, b, c, d),) if panel.topology == "Q4" else (
                (a, b, c), (a, c, d))
            for nodes in patches:
                element_id += 1
                element = create_shell_element(
                    element_id, list(nodes), "steel",
                    formulation=("e4-pl" if panel.topology == "Q4"
                                 else "e4-pl-s3-v2d"),
                    thickness=panel.thickness,
                    reference_normal=(0., 0., 1.))
                physical = np.asarray(
                    element.compute_stiffness_components(mesh, material)["physical"])
                dofs = np.array([6*(node-1)+axis
                                 for node in nodes for axis in range(6)])
                ii, jj = np.meshgrid(dofs, dofs, indexing="ij")
                rows.extend(ii.ravel())
                columns.extend(jj.ravel())
                values.extend(physical.ravel())
    size = 6*panel.node_count
    matrix = sparse.coo_matrix((values, (rows, columns)),
                               shape=(size, size)).tocsr()
    matrix.sum_duplicates()
    return matrix
