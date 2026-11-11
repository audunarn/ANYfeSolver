"""Source-only Q4/y-row pressure discriminator against scalar continuum.

The shell stiffness is the production elastic tangent. The clipped Q1
pressure work, scalar quadratic problem and continuum reference are assembled
outside the general-contact kernel. This is development evidence only.
"""

import argparse
import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from anyfesolver import BoundaryCondition, create_shell_element
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.fe_core import FEModel
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from scripts.general_contact_d1_continuum_obstacle_reference import solve
from scripts.general_contact_d1_front_cluster_screen import _cluster_knots
from scripts.general_contact_d1_linearized_front_reference import _nonnegative_quadratic


def _q1_mass(cells: int, right: float) -> sparse.csr_matrix:
    if cells < 1 or right <= 0.0 or right > 1.0:
        raise ValueError("invalid Q1 interval")
    matrix = sparse.lil_matrix((cells+1, cells+1))
    h = 1.0/cells
    points = (0.5*(1.0-1.0/np.sqrt(3.0)),
              0.5*(1.0+1.0/np.sqrt(3.0)))
    for cell in range(cells):
        left = cell*h
        end = min((cell+1)*h, right)
        if end <= left:
            continue
        interval = end-left
        for fraction in points:
            coordinate = left+fraction*interval
            local = (coordinate-left)/h
            shape = np.array((1.0-local, local))
            matrix[cell:cell+2, cell:cell+2] += (
                0.5*interval*np.outer(shape, shape)
            )
    return matrix.tocsr()


def _q1_mass_knots(knots: np.ndarray, right: float) -> sparse.csr_matrix:
    """Exact clipped P1 mass on a declared nonuniform physical grid."""
    coordinates = np.asarray(knots, dtype=float)
    if (coordinates.ndim != 1 or len(coordinates) < 2
            or not np.all(np.isfinite(coordinates))
            or coordinates[0] != 0.0 or coordinates[-1] != 1.0
            or np.min(np.diff(coordinates)) <= 0.0
            or right <= 0.0 or right > 1.0):
        raise ValueError("invalid nonuniform Q1 interval")
    cells = len(coordinates)-1
    matrix = sparse.lil_matrix((cells+1, cells+1))
    points = (0.5*(1.0-1.0/np.sqrt(3.0)),
              0.5*(1.0+1.0/np.sqrt(3.0)))
    for cell in range(cells):
        left, cell_right = coordinates[cell:cell+2]
        end = min(cell_right, right)
        if end <= left:
            continue
        interval = end-left
        for fraction in points:
            coordinate = left+fraction*interval
            local = (coordinate-left)/(cell_right-left)
            shape = np.array((1.0-local, local))
            matrix[cell:cell+2, cell:cell+2] += (
                0.5*interval*np.outer(shape, shape)
            )
    return matrix.tocsr()


def _aligned_q1_cross_mass(structural_cells: int,
                           pressure_cells: int) -> sparse.csr_matrix:
    """Exact pressure P1 × structural Q1 work on nested x grids."""
    if (pressure_cells < 1 or pressure_cells > structural_cells
            or structural_cells % pressure_cells):
        raise ValueError("pressure x knots must align to structural x nodes")
    return _aligned_q1_cross_knots(
        structural_cells, np.linspace(0.0, 1.0, pressure_cells+1),
    )


def _aligned_q1_cross_knots(structural_cells: int,
                            pressure_knots: np.ndarray) -> sparse.csr_matrix:
    """Exact Q1 cross-work for pressure knots on structural nodes."""
    knots = np.asarray(pressure_knots, dtype=float)
    if (knots.ndim != 1 or len(knots) < 2
            or not np.all(np.isfinite(knots))
            or knots[0] != 0.0 or knots[-1] != 1.0
            or np.min(np.diff(knots)) <= 0.0
            or not np.allclose(knots*structural_cells,
                               np.rint(knots*structural_cells),
                               rtol=0.0, atol=1.0e-12)):
        raise ValueError("pressure x knots must align with structural nodes")
    pressure_cells = len(knots)-1
    matrix = sparse.lil_matrix((pressure_cells+1, structural_cells+1))
    h = 1.0/structural_cells
    points = (0.5*(1.0-1.0/np.sqrt(3.0)),
              0.5*(1.0+1.0/np.sqrt(3.0)))
    for cell in range(structural_cells):
        for fraction in points:
            coordinate = (cell+fraction)*h
            pressure_cell = min(pressure_cells-1,
                                int(np.searchsorted(knots, coordinate, side="right")-1))
            structural_shape = np.array((1.0-fraction, fraction))
            pressure_local = ((coordinate-knots[pressure_cell])
                              /(knots[pressure_cell+1]-knots[pressure_cell]))
            pressure_shape = np.array((1.0-pressure_local, pressure_local))
            matrix[pressure_cell:pressure_cell+2, cell:cell+2] += (
                0.5*h*np.outer(pressure_shape, structural_shape)
            )
    if abs(float(matrix.sum())-1.0) > 1.0e-13:
        raise AssertionError("aligned Q1 cross-work partition is invalid")
    return matrix.tocsr()


def _model(nx: int, ny: int,
           source_y_knots: np.ndarray | None = None) -> FEModel:
    model = FEModel(f"source_y_resolution_{nx}_{ny}")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    stride = nx+1
    for row in range(ny+1):
        for column in range(nx+1):
            node = 1+row*stride+column
            y = row/ny if source_y_knots is None else source_y_knots[row]
            model.add_node(node, 0.1+column/nx, 0.1+y, 0.04004)
    for row in range(ny):
        for column in range(nx):
            bottom = 1+row*stride+column
            element = 1+row*nx+column
            model.add_element(element, create_shell_element(
                element,
                (bottom, bottom+1, bottom+stride+1, bottom+stride),
                "elastic", thickness=0.05,
                reference_normal=(0.0, 0.0, 1.0),
            ))
    nodes = tuple(range(1, stride*(ny+1)+1))
    left = tuple(1+row*stride for row in range(ny+1))
    model.add_boundary_condition(BoundaryCondition(
        "guide", nodes, {name: 0.0 for name in
                         ("uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "left_support", left, {"uz": 0.0, "ux": 0.0},
    ))
    return model


def run(nx: int, ny: int, pressure_nx: int | None = None,
        comparison_nx: int | None = None,
        pressure_knots: tuple[float, ...] | None = None,
        obstacle_y_right: float = 0.9,
        source_y_knots: tuple[float, ...] | None = None) -> dict:
    if obstacle_y_right not in (0.9, 1.0):
        raise ValueError("development obstacle must end at y=0.9 or y=1.0")
    if pressure_knots is not None and pressure_nx is not None:
        raise ValueError("choose uniform pressure cells or selected knots")
    y_knots = (None if source_y_knots is None
               else np.asarray(source_y_knots, dtype=float))
    if y_knots is not None and (
            y_knots.shape != (ny+1,) or not np.all(np.isfinite(y_knots))
            or y_knots[0] != 0.0 or y_knots[-1] != 1.0
            or np.min(np.diff(y_knots)) <= 0.0):
        raise ValueError("source y knots must increase from zero to one")
    knots = (np.asarray(pressure_knots, dtype=float)
             if pressure_knots is not None else None)
    pressure_nx = (len(knots)-1 if knots is not None
                   else nx if pressure_nx is None else pressure_nx)
    comparison_nx = nx if comparison_nx is None else comparison_nx
    pressure_coordinates = (knots if knots is not None
                            else np.linspace(0.0, 1.0, pressure_nx+1))
    if (comparison_nx < pressure_nx or comparison_nx > 256
            or 256 % comparison_nx
            or not np.allclose(pressure_coordinates*comparison_nx,
                               np.rint(pressure_coordinates*comparison_nx),
                               rtol=0.0, atol=1.0e-12)):
        raise ValueError("comparison x bins must be a nested pressure grid dividing 256")
    model = _model(nx, ny, y_knots)
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    stride = nx+1
    node_z = np.asarray([
        model.mesh.dof_manager.get_node_dofs(node)[2]
        for node in range(1, stride*(ny+1)+1)
    ], dtype=int)
    load = np.zeros(stiffness.shape[0])
    for row in range(ny):
        for column in range(nx):
            nodes = (
                row*stride+column, row*stride+column+1,
                (row+1)*stride+column+1, (row+1)*stride+column,
            )
            cell_height = (1.0/ny if y_knots is None
                           else y_knots[row+1]-y_knots[row])
            load[node_z[list(nodes)]] -= 0.3*cell_height/(4.0*nx)
    if abs(np.sum(load)+0.3) > 1.0e-13:
        raise AssertionError("distributed source load is invalid")
    reduced_k, reduced_load, transform, offset, _, _ = (
        build_constraint_transformation(stiffness, load, model)
    )
    if np.linalg.norm(offset) > 0.0:
        raise ValueError("source y reference requires zero prescribed pose")

    mx = (_aligned_q1_cross_knots(nx, knots) if knots is not None
          else _q1_mass(nx, 1.0) if pressure_nx == nx
          else _aligned_q1_cross_mass(nx, pressure_nx))
    my = (_q1_mass(ny, obstacle_y_right) if y_knots is None
          else _q1_mass_knots(y_knots, obstacle_y_right))
    pressure_work_on_z = sparse.kron(my, mx, format="coo")
    work = sparse.coo_matrix((
        pressure_work_on_z.data,
        (pressure_work_on_z.row, node_z[pressure_work_on_z.col]),
    ), shape=(pressure_work_on_z.shape[0], stiffness.shape[0])).tocsr()
    full_support = np.asarray(pressure_work_on_z.tocsr() @ np.ones(len(node_z)))
    if (np.min(full_support) < -1.0e-14
            or abs(np.sum(full_support)-obstacle_y_right) > 1.0e-12):
        raise ValueError("clipped Q1 pressure support is invalid")
    supported = full_support > 1.0e-14
    support = full_support[supported]
    work = work[supported]
    reduced_work = (work @ transform).toarray()
    factor = splu(reduced_k.tocsc())
    free = factor.solve(reduced_load)
    response = factor.solve(reduced_work.T)
    compliance = reduced_work @ response
    compliance = 0.5*(compliance+compliance.T)
    initial_gap = 4.0e-5
    linear = support*initial_gap+reduced_work @ free
    qp_diagnostics = {}
    pressure = _nonnegative_quadratic(
        compliance, linear, support,
        accept_iteration_limit_with_certificate=True,
        diagnostics=qp_diagnostics,
    )
    active = pressure > 1.0e-8
    active_compliance = compliance[np.ix_(active, active)]
    active_diagonal = np.sqrt(np.diag(active_compliance))
    active_scaled = active_compliance / np.outer(active_diagonal, active_diagonal)
    active_eigenvalues = np.linalg.eigvalsh(active_scaled)
    if (not np.all(np.isfinite(active_eigenvalues))
            or active_eigenvalues[0] <= 0.0):
        raise RuntimeError("active pressure compliance is not positive definite")
    reduced_displacement = free+response @ pressure
    equilibrium = reduced_k @ reduced_displacement-reduced_load-reduced_work.T @ pressure
    weighted_gap = (linear+compliance @ pressure)/support
    if (np.linalg.norm(equilibrium) > 1.0e-7
            or np.min(weighted_gap) < -1.0e-11):
        raise RuntimeError("source y-resolution equilibrium or weak gap failed")
    full = np.asarray(transform @ reduced_displacement).reshape(-1)
    vertical = full[node_z].reshape(ny+1, nx+1)
    full_pressure = np.zeros(pressure_work_on_z.shape[0])
    full_pressure[supported] = pressure
    pressure_grid = full_pressure.reshape(ny+1, pressure_nx+1)
    y_support = np.asarray(my @ np.ones(ny+1)).reshape(-1)
    x_line = y_support @ pressure_grid
    # Pressure knots lie on the declared comparison grid. Interpolate the P1
    # field there, then integrate every compared case on identical x bins.
    shared_line = np.interp(
        np.linspace(0.0, 1.0, comparison_nx+1),
        pressure_coordinates, x_line,
    )
    x_force = (shared_line[:-1]+shared_line[1:])/(2.0*comparison_nx)
    if abs(float(np.sum(x_force)-support @ pressure)) > 1.0e-12:
        raise AssertionError("common-bin pressure force differs from Q1 support integral")
    boundary = obstacle_y_right*ny
    lower = (int(np.floor(boundary)) if y_knots is None
             else int(np.searchsorted(y_knots, obstacle_y_right, side="right")-1))
    if lower == ny:
        boundary_vertical = vertical[-1]
    elif (abs(boundary-lower) < 1.0e-14 if y_knots is None
          else abs(y_knots[lower]-obstacle_y_right) < 1.0e-14):
        boundary_vertical = vertical[lower]
    else:
        fraction = (boundary-lower if y_knots is None else
                    (obstacle_y_right-y_knots[lower])
                    /(y_knots[lower+1]-y_knots[lower]))
        boundary_vertical = ((1.0-fraction)*vertical[lower]
                             +fraction*vertical[lower+1])
    represented_gap = min(
        float(np.min(initial_gap+vertical[:lower+1])),
        float(np.min(initial_gap+boundary_vertical)),
    )
    if represented_gap < -50.0e-6:
        raise RuntimeError("source y-resolution case exceeds its 50 µm represented-gap budget")
    return {
        "nx": nx, "ny": ny, "pressure_nx": pressure_nx,
        "obstacle_y_right": obstacle_y_right,
        "source_y_knots_m": (y_knots.tolist() if y_knots is not None else None),
        "pressure_knots": (knots.tolist() if knots is not None else None),
        "comparison_x_bins": comparison_nx,
        "pressure_modes": len(pressure),
        "unsupported_pressure_modes": int(np.sum(~supported)),
        "active_modes": int(np.sum(active)),
        "active_scaled_compliance_minimum_eigenvalue": float(active_eigenvalues[0]),
        "active_scaled_compliance_condition": float(
            active_eigenvalues[-1]/active_eigenvalues[0]
        ),
        "integrated_reaction_n": float(np.sum(x_force)),
        "minimum_weak_gap_m": float(np.min(weighted_gap)),
        "minimum_represented_gap_m": represented_gap,
        "equilibrium_residual_n": float(np.linalg.norm(equilibrium)),
        "pressure_qp": qp_diagnostics,
        "pressure_grid_pa": pressure_grid.tolist(),
        "line_pressure_n_per_m": x_line.tolist(),
        "x_force_n": x_force.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, nargs="+", default=(16, 32))
    parser.add_argument("--ny", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--pressure-nx", type=int, nargs="+", default=None)
    parser.add_argument("--comparison-nx", type=int, default=None)
    parser.add_argument("--front-cluster-subdivisions", type=int,
                        choices=(8,), default=None)
    args = parser.parse_args()
    if (not args.nx or not args.ny or max(args.nx) > 128
            or max(args.ny) > 16 or min(args.nx) < 8 or min(args.ny) < 1
            or any(256 % nx for nx in args.nx)):
        raise ValueError("development source grids require nx=8..128 dividing 256, ny=1..16")
    if args.pressure_nx is not None and (
            not args.pressure_nx or min(args.pressure_nx) < 1
            or any(pressure_nx > nx or nx % pressure_nx
                   or 256 % pressure_nx
                   for nx in args.nx for pressure_nx in args.pressure_nx)):
        raise ValueError("pressure x grids must divide the structural and 256-cell reference grids")
    selected_knots = (None if args.front_cluster_subdivisions is None
                      else _cluster_knots(args.front_cluster_subdivisions))
    if selected_knots is not None and (
            args.pressure_nx is not None or args.nx != [128]
            or args.ny != [16]):
        raise ValueError("the selected front-knot screen is bounded to one 128x16 source")
    if args.comparison_nx is not None and (
            args.comparison_nx < 1 or args.comparison_nx > 256
            or 256 % args.comparison_nx
            or (selected_knots is not None and not np.allclose(
                np.asarray(selected_knots)*args.comparison_nx,
                np.rint(np.asarray(selected_knots)*args.comparison_nx),
                rtol=0.0, atol=1.0e-12,
            ))
            or (selected_knots is None and any(
                args.comparison_nx % pressure_nx
                for nx in args.nx
                for pressure_nx in (args.pressure_nx or (nx,))
            ))):
        raise ValueError("comparison x bins must contain every pressure knot and divide 256")
    continuum = solve(256, 320, 0.9)
    continuum_x = np.sum(continuum.pressure_force, axis=0)
    cases = [
        run(nx, ny, pressure_nx, args.comparison_nx,
            pressure_knots=selected_knots)
        for nx in args.nx for ny in args.ny
        for pressure_nx in ((None,) if selected_knots is not None
                            else (args.pressure_nx or (nx,)))
    ]
    for case in cases:
        bins = case["comparison_x_bins"]
        reference = continuum_x.reshape(bins, 256//bins).sum(axis=1)
        source = np.asarray(case.pop("x_force_n"))
        difference = source-reference
        case["force_difference_n"] = float(np.sum(difference))
        case["x_force_l1_relative"] = float(
            np.sum(np.abs(difference))/np.sum(reference)
        )
        case["x_force_l2_relative"] = float(
            np.linalg.norm(difference)/np.linalg.norm(reference)
        )
        case["cumulative_force_max_relative"] = float(
            np.max(np.abs(np.cumsum(difference)))/np.sum(reference)
        )
    print(json.dumps({
        "method": "source_q4_y_resolution_against_independent_scalar_obstacle_dev",
        "continuum_grid": [256, 320],
        "continuum_contact_reaction_n": continuum.contact_reaction_n,
        "cases": cases,
    }, indent=2))


if __name__ == "__main__":
    main()
