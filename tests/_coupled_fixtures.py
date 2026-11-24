"""Fixtures for the coupled shell-solid route: a Q4 plate with a rectangular hole on element edges
and the structured Kuhn-split box lattice of the footprint (the anymesher structured_box layout,
numbered v = (k (ny+1) + j)(nx+1) + i). Not a test module."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from anyfesolver import BoundaryCondition
from anyfesolver.boundary import LoadCase
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel

E, NU = 210.0e9, 0.3


def plate(n=6, length=1.0, thickness=0.02, *, hole=(2, 4, 2, 4), supports='simple', nu=NU, modulus=E):
    """An n x n Q4 plate on [0, L]^2 at z = 0; ``hole = (i0, i1, j0, j1)`` in element indices.
    Returns (model, grid of node ids, removed element ids, footprint (x0, x1, y0, y1))."""
    model = FEModel('coupled-plate')
    model.add_material('steel', modulus, nu)
    h = length / n
    grid = {}
    for j in range(n + 1):
        for i in range(n + 1):
            nid = j * (n + 1) + i + 1
            model.add_node(nid, i * h, j * h, 0.0)
            grid[i, j] = nid
    removed = []
    i0, i1, j0, j1 = hole
    for j in range(n):
        for i in range(n):
            eid = j * n + i + 1
            ids = [grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]]
            model.add_element(eid, create_shell_element(eid, ids, 'steel', thickness=thickness))
            if i0 <= i < i1 and j0 <= j < j1:
                removed.append(eid)
    edge = sorted({grid[i, j] for (i, j) in grid if i in (0, n) or j in (0, n)})
    if supports == 'simple':
        model.add_boundary_condition(BoundaryCondition(name='ss', node_ids=edge, dof_constraints={'uz': 0.0}))
        corner = [grid[0, 0]]
        model.add_boundary_condition(BoundaryCondition(name='pin', node_ids=corner,
                                                       dof_constraints={'ux': 0.0, 'uy': 0.0, 'rz': 0.0}))
        model.add_boundary_condition(BoundaryCondition(name='roller', node_ids=[grid[n, 0]],
                                                       dof_constraints={'uy': 0.0}))
    elif supports == 'clamped':
        model.add_boundary_condition(BoundaryCondition(
            name='clamp', node_ids=edge, dof_constraints={k: 0.0 for k in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    footprint = (i0 * h, i1 * h, j0 * h, j1 * h)
    return model, grid, removed, footprint


def pressure_case(removed, model, p=1000.0, *, everywhere=True):
    case = LoadCase(name='p')
    for eid in model.mesh.elements:
        if everywhere or eid in removed:
            case.pressure_loads[eid] = float(p)
    return case


def lattice_coordinates(footprint, thickness, nx, ny, nz, *, offset=0.0):
    x0, x1, y0, y1 = footprint
    coordinates = np.empty(((nz + 1), (ny + 1), (nx + 1), 3))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                coordinates[k, j, i] = (x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny,
                                        offset - thickness / 2 + thickness * k / nz)
    return coordinates.reshape(-1, 3)


def kuhn_cells(nx, ny, nz):
    sx, sy = nx + 1, ny + 1
    offsets = (1, sx, sx * sy)
    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v0 = (k * sy + j) * sx + i
                v7 = v0 + sum(offsets)
                for p in permutations(range(3)):
                    o1 = offsets[p[0]]
                    o2 = o1 + offsets[p[1]]
                    cells.append((v0, v0 + o1, v0 + o2, v7))
    return np.asarray(cells, dtype=np.int64)


def box_mesh(footprint, thickness, nx, ny, nz, *, coordinates=None, drop_facet=None, group_names=None):
    """SolidMesh of the lattice with groups start/end/side_0..side_3 (side_k between corners k, k+1)."""
    from anysolid.mesh import SolidMesh
    x = lattice_coordinates(footprint, thickness, nx, ny, nz) if coordinates is None else coordinates
    cells = kuhn_cells(nx, ny, nz)
    # positive orientation
    d = x[cells[:, 1:]] - x[cells[:, :1]]
    flip = np.linalg.det(d) < 0
    cells[flip, 2], cells[flip, 3] = cells[flip, 3].copy(), cells[flip, 2].copy()
    first = SolidMesh(x, cells)
    lattice = lattice_coordinates(footprint, thickness, nx, ny, nz)
    ijk = np.stack(np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing='ij'),
                   axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)
    names = group_names or {}
    groups = {}
    for facet, bid in zip(first.boundary_facets, first.boundary_ids):
        idx = ijk[facet]
        if (idx[:, 2] == 0).all():
            name = 'start'
        elif (idx[:, 2] == nz).all():
            name = 'end'
        elif (idx[:, 1] == 0).all():
            name = 'side_0'
        elif (idx[:, 0] == nx).all():
            name = 'side_1'
        elif (idx[:, 1] == ny).all():
            name = 'side_2'
        else:
            name = 'side_3'
        name = names.get(name, name)
        if drop_facet is not None and name == drop_facet[0] and len(groups.get(name, [])) == drop_facet[1]:
            continue
        groups.setdefault(name, []).append(int(bid))
    del lattice
    return SolidMesh(first.coordinates, first.cells, boundary_groups=groups)


def prepare(mesh, *, nu=NU, modulus=E, order=2):
    import anysolid
    return anysolid.prepare(mesh, {'elastic_modulus': modulus, 'poisson_ratio': nu}, order=order)
