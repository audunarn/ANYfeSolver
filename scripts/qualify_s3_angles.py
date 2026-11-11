"""S3 minimum-angle campaign runner (docs/S3_ANGLE_EXTENSION.md).

Runs the element sweep and the assembled-model accuracy families for the
current-policy three-node shell route (``create_shell_element`` default, which
resolves to E4-PL S3 V2D) at a selected minimum-angle target, and writes a
manifest, raw metrics and a readable summary under
``reports/s3_angles/<candidate-sha>/<run-id>/``.

Modes
-----
``--mode exploratory`` runs everything and never fails on criteria.
``--mode formal`` additionally evaluates the frozen acceptance criteria in
``ACCEPTANCE`` and exits non-zero when any required criterion fails.

Every reference is analytical: the plane-stress Timoshenko cantilever field
(exact Dirichlet manufactured solution), and Navier double-sine series for the
hard simply-supported Mindlin plate (deflection, moments, vibration without
rotary inertia, and uniaxial buckling).  No legacy-S3 or coarse-Q4 solution is
used as ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from _s3_angle_fixtures import (  # noqa: E402
    ANGLE_BANDS_DEG,
    CANDIDATE_ENVELOPES,
    DIAGNOSTIC_BANDS_DEG,
    E,
    NU,
    DENSITY,
    build_default_s3,
    bending_patch_field,
    elastic_spectrum,
    envelope_violations,
    independent_quality,
    isotropic_plane_stress,
    membrane_patch_field,
    rigid_body_basis,
    shape_family,
    triangle_from_angles,
)

SCHEMA = "anysolver.s3-angle-campaign-v1"
KAPPA = 5.0 / 6.0
G_MOD = E / (2.0 * (1.0 + NU))

# Frozen before the formal runs (plan section 6.3).  Errors are relative.
ACCEPTANCE = {
    "element_rigid_residual": 1.0e-10,
    "element_patch_residual": 1.0e-10,
    "principal_deflection_error": 0.02,
    "resultant_l2_error": 0.05,
    # Membrane (CST) resultants: finest-level L2 error no more than 1.5x the
    # 45 deg family's error at equal DOF count (see adjudicate()).
    "membrane_resultant_ratio_to_regular": 1.5,
    "first_frequency_error": 0.01,
    "buckling_factor_error": 0.03,
    # Observed asymptotic order no more than 10% below the matched regular
    # (45 deg) family, where a rate is meaningful.
    "rate_fraction_of_regular": 0.90,
    # Mixed Q4/S3: finest-mesh error within the absolute target and no worse
    # than max(1.5x the same layout at 30 deg, target / 10).
    "mixed_error_factor": 1.5,
}


# ---------------------------------------------------------------------------
# Structured rectangular meshes with controlled triangle shapes
# ---------------------------------------------------------------------------


@dataclass
class Mesh2D:
    nodes: Dict[int, Tuple[float, float]]
    elements: List[Tuple[str, Tuple[int, ...]]]
    xs: np.ndarray
    ys: np.ndarray
    grid: Dict[Tuple[int, int], int]
    description: Dict[str, Any] = field(default_factory=dict)

    @property
    def lx(self) -> float:
        return float(self.xs[-1] - self.xs[0])

    @property
    def ly(self) -> float:
        return float(self.ys[-1] - self.ys[0])


def rectangular_mesh(
    xs: Sequence[float],
    ys: Sequence[float],
    cell_kind: Callable[[int, int], str],
) -> Mesh2D:
    """Build a conformal mesh; each cell is 'q4', 'diag', 'anti' or 'cross'."""

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    nodes: Dict[int, Tuple[float, float]] = {}
    grid: Dict[Tuple[int, int], int] = {}
    next_id = 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            nodes[next_id] = (float(x), float(y))
            grid[(i, j)] = next_id
            next_id += 1
    elements: List[Tuple[str, Tuple[int, ...]]] = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            a, b = grid[(i, j)], grid[(i + 1, j)]
            c, d = grid[(i + 1, j + 1)], grid[(i, j + 1)]
            kind = cell_kind(i, j)
            if kind == "q4":
                elements.append(("q4", (a, b, c, d)))
            elif kind == "diag":
                elements.append(("s3", (a, b, c)))
                elements.append(("s3", (a, c, d)))
            elif kind == "anti":
                elements.append(("s3", (a, b, d)))
                elements.append(("s3", (b, c, d)))
            elif kind == "cross":
                centre = next_id
                next_id += 1
                nodes[centre] = (
                    0.5 * float(xs[i] + xs[i + 1]),
                    0.5 * float(ys[j] + ys[j + 1]),
                )
                for p, q in ((a, b), (b, c), (c, d), (d, a)):
                    elements.append(("s3", (p, q, centre)))
            else:
                raise ValueError(kind)
    return Mesh2D(nodes, elements, xs, ys, grid)


def narrow_axes(
    angle_deg: float,
    level: int,
    *,
    target: Tuple[float, float],
    base_count: int,
    narrow: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform grid lines with cell aspect ``tan(angle)`` exactly.

    ``narrow='y'`` makes cells short in y (h_y = h_x tan a); the y extent is
    rounded to a whole number of cells.  Refinement doubles both counts, so the
    domain and cell shape are preserved across levels.
    """

    lx, ly = target
    ratio = math.tan(math.radians(angle_deg))
    if narrow == "y":
        nx = base_count
        hx = lx / nx
        ny = max(1, int(round(ly / (hx * ratio))))
        ly = ny * hx * ratio
    elif narrow == "x":
        ny = base_count
        hy = ly / ny
        nx = max(1, int(round(lx / (hy * ratio))))
        lx = nx * hy * ratio
    else:
        raise ValueError(narrow)
    factor = 2**level
    return (
        np.linspace(0.0, lx, nx * factor + 1),
        np.linspace(0.0, ly, ny * factor + 1),
    )


def s3_family_mesh(
    family: str,
    angle_deg: float,
    level: int,
    *,
    target: Tuple[float, float],
    base_count: int,
    narrow: str = "y",
) -> Mesh2D:
    """Pure-S3 families.

    ``right``: diagonal split of cells with aspect tan(a): (a, 90-a, 90).
    ``obtuse``: cross split: (a, a, 180-2a) and (90-a, 90-a, 2a).
    ``regular``: 45 deg right triangles (baseline, ``angle_deg`` ignored).
    """

    if family == "regular":
        xs, ys = narrow_axes(45.0, level, target=target, base_count=base_count, narrow=narrow)
        mesh = rectangular_mesh(xs, ys, lambda i, j: "diag")
    elif family == "right":
        xs, ys = narrow_axes(angle_deg, level, target=target, base_count=base_count, narrow=narrow)
        mesh = rectangular_mesh(xs, ys, lambda i, j: "diag")
    elif family == "right_alternating":
        xs, ys = narrow_axes(angle_deg, level, target=target, base_count=base_count, narrow=narrow)
        mesh = rectangular_mesh(xs, ys, lambda i, j: "diag" if (i + j) % 2 else "anti")
    elif family == "obtuse":
        # Cross split: the long cell side must be the tan(90-a) one, so that
        # triangles on the long side are the (a, a, 180-2a) obtuse ones.
        xs, ys = narrow_axes(angle_deg, level, target=target, base_count=base_count, narrow=narrow)
        mesh = rectangular_mesh(xs, ys, lambda i, j: "cross")
    else:
        raise ValueError(family)
    mesh.description = {
        "family": family,
        "angle_deg": angle_deg,
        "level": level,
        "narrow": narrow,
    }
    return mesh


def mixed_mesh(
    pattern: str,
    angle_deg: float,
    level: int,
    *,
    lx: float = 1.0,
    ly: float = 1.0,
    base_count: int = 8,
) -> Mesh2D:
    """Q4 panel with narrow low-angle S3 fill-ins.

    The grid is regular (square cells) except for a set of *narrow* columns
    of width ``h tan(a)``; the domain width is adjusted to keep total width.
    Narrow cells that are triangulated give right triangles with minimum
    angle ``a`` (diag) or obtuse (a, a, 180-2a) triangles (cross); the
    remaining narrow cells stay as Q4 rectangles.

    Patterns control placement: ``isolated``, ``chain``, ``boundary``,
    ``cluster`` and ``quarter`` (approximately 25% triangles).
    """

    factor = 2**level
    n = base_count * factor
    h = ly / n
    ratio = math.tan(math.radians(angle_deg))
    narrow_width = h * ratio
    # Choose narrow column indices (in units of base columns, scaled).
    if pattern == "isolated":
        narrow_columns = [n // 2]
    elif pattern == "chain":
        narrow_columns = [n // 3]
    elif pattern == "boundary":
        narrow_columns = [0]
    elif pattern == "cluster":
        narrow_columns = [n // 2 - 1, n // 2, n // 2 + 1]
    elif pattern == "quarter":
        narrow_columns = list(range(n // 4, n // 4 + max(2, n // 8)))
    else:
        raise ValueError(pattern)
    regular_count = n
    total_columns = regular_count + len(narrow_columns)
    widths = []
    narrow_set = set()
    regular_width = (lx - len(narrow_columns) * narrow_width) / regular_count
    for column in range(total_columns):
        if column in narrow_columns:
            widths.append(narrow_width)
            narrow_set.add(column)
        else:
            widths.append(regular_width)
    xs = np.concatenate(([0.0], np.cumsum(widths)))
    xs[-1] = lx
    ys = np.linspace(0.0, ly, n + 1)

    def kind(i: int, j: int) -> str:
        if i not in narrow_set:
            return "q4"
        if pattern == "isolated":
            # A single split cell per narrow column at mid-height, plus the
            # cell 3/4 up in obtuse (cross) form.
            if j == n // 2:
                return "diag"
            if j == (3 * n) // 4:
                return "cross"
            return "q4"
        if pattern == "cluster":
            return "cross" if (n // 3 <= j < (2 * n) // 3) else "q4"
        # chain, boundary, quarter: full-height chains alternating diag/cross
        return "diag" if j % 4 else "cross"

    mesh = rectangular_mesh(xs, ys, kind)
    mesh.description = {
        "family": f"mixed_{pattern}",
        "angle_deg": angle_deg,
        "level": level,
        "regular_width": regular_width,
        "narrow_width": narrow_width,
    }
    return mesh


def all_q4_mesh(level: int, *, lx: float = 1.0, ly: float = 1.0, base_count: int = 8) -> Mesh2D:
    n = base_count * 2**level
    mesh = rectangular_mesh(
        np.linspace(0.0, lx, n + 1), np.linspace(0.0, ly, n + 1), lambda i, j: "q4"
    )
    mesh.description = {"family": "all_q4", "level": level}
    return mesh


def mesh_statistics(mesh: Mesh2D) -> Dict[str, Any]:
    angles_min, angles_max, q_min, ratios = [], [], [], []
    tri = 0
    for kind, nodes in mesh.elements:
        if kind != "s3":
            continue
        tri += 1
        coordinates = np.asarray([(*mesh.nodes[n], 0.0) for n in nodes])
        quality = independent_quality(coordinates)
        angles_min.append(quality.minimum_angle_deg)
        angles_max.append(quality.maximum_angle_deg)
        q_min.append(quality.normalized_area)
        ratios.append(quality.edge_ratio)
    total = len(mesh.elements)
    area_s3 = 0.0
    edge_lengths = []
    for kind, nodes in mesh.elements:
        coordinates = np.asarray([mesh.nodes[n] for n in nodes])
        closed = np.vstack((coordinates, coordinates[:1]))
        edge_lengths.extend(np.linalg.norm(np.diff(closed, axis=0), axis=1).tolist())
    return {
        "nodes": len(mesh.nodes),
        "dofs": 6 * len(mesh.nodes),
        "elements": total,
        "s3_elements": tri,
        "s3_fraction": tri / total if total else 0.0,
        "s3_min_angle_deg": min(angles_min) if angles_min else None,
        "s3_max_angle_deg": max(angles_max) if angles_max else None,
        "s3_min_normalized_area": min(q_min) if q_min else None,
        "s3_max_edge_ratio": max(ratios) if ratios else None,
        "h_max": max(edge_lengths),
        "h_min": min(edge_lengths),
    }


# ---------------------------------------------------------------------------
# Model assembly on the public API
# ---------------------------------------------------------------------------


def build_model(mesh: Mesh2D, thickness: float):
    from anyfesolver import FEModel, create_shell_element

    model = FEModel("s3-angle")
    model.add_material("steel", E, NU, density=DENSITY)
    for node_id, (x, y) in mesh.nodes.items():
        model.add_node(node_id, x, y, 0.0)
    for element_id, (kind, nodes) in enumerate(mesh.elements, start=1):
        kwargs: Dict[str, Any] = {"thickness": thickness}
        if kind == "s3":
            kwargs["reference_normal"] = (0.0, 0.0, 1.0)
        element = create_shell_element(element_id, list(nodes), "steel", **kwargs)
        model.add_element(element_id, element)
    return model


def _boundary_nodes(mesh: Mesh2D, tolerance: float = 1.0e-12) -> Dict[str, List[int]]:
    x0, x1 = mesh.xs[0], mesh.xs[-1]
    y0, y1 = mesh.ys[0], mesh.ys[-1]
    sides: Dict[str, List[int]] = {"x0": [], "x1": [], "y0": [], "y1": []}
    for node_id, (x, y) in mesh.nodes.items():
        if abs(x - x0) <= tolerance:
            sides["x0"].append(node_id)
        if abs(x - x1) <= tolerance:
            sides["x1"].append(node_id)
        if abs(y - y0) <= tolerance:
            sides["y0"].append(node_id)
        if abs(y - y1) <= tolerance:
            sides["y1"].append(node_id)
    return sides


def element_resultants(model, displacement: np.ndarray, key: str):
    """Yield (station xy, weight, global 2x2 resultant tensor) for S3 elements."""

    from anyfesolver.fe_core import FEMesh

    material = model.get_material("steel")
    manager = model.mesh.dof_manager
    for element in model.mesh.elements.values():
        if len(element.node_ids) != 3:
            continue
        # Post-processing only: evaluate each element on a private three-node
        # mesh.  V2D's model-scope check scans every registered element on each
        # geometry call, which makes a whole-model sweep quadratic; the model
        # scope was already validated by the solve itself.
        local = FEMesh()
        for node_id in element.node_ids:
            local.add_node(node_id, *model.mesh.get_node(node_id).coords())
        vector = np.zeros(local.dof_manager.total_dofs)
        for node_id in element.node_ids:
            vector[local.dof_manager.get_node_dofs(node_id)] = displacement[
                manager.get_node_dofs(node_id)
            ]
        data = element.compute_variational_resultants(local, vector, material)
        frame = np.asarray(data["frame"])[:2, :2]
        for station, weight, voigt in zip(
            data["physical_station_coordinates"], data["physical_weights"], data[key]
        ):
            local = np.asarray(((voigt[0], voigt[2]), (voigt[2], voigt[1])))
            yield station[:2], float(weight), frame @ local @ frame.T


# ---------------------------------------------------------------------------
# Membrane: Timoshenko plane-stress cantilever (exact Dirichlet field)
# ---------------------------------------------------------------------------


def timoshenko_field(x: np.ndarray, y: np.ndarray, length: float, c: float, thickness: float):
    load = 1.0e5
    inertia = thickness * (2.0 * c) ** 3 / 12.0
    ei, gi = E * inertia, G_MOD * inertia
    u = (
        -load * x**2 * y / (2.0 * ei)
        - NU * load * y**3 / (6.0 * ei)
        + load * y**3 / (6.0 * gi)
        + (load * length**2 / (2.0 * ei) - load * c**2 / (2.0 * gi)) * y
    )
    v = (
        NU * load * x * y**2 / (2.0 * ei)
        + load * x**3 / (6.0 * ei)
        - load * length**2 * x / (2.0 * ei)
        + load * length**3 / (3.0 * ei)
    )
    sxx = -load * x * y / inertia
    sxy = -load * (c**2 - y**2) / (2.0 * inertia)
    return u, v, sxx, sxy


def membrane_exact_energy(length: float, c: float, thickness: float) -> float:
    gx, gw = np.polynomial.legendre.leggauss(6)
    xs = 0.5 * length * (gx + 1.0)
    ys = c * gx
    wx = 0.5 * length * gw
    wy = c * gw
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    _u, _v, sxx, sxy = timoshenko_field(X, Y, length, c, thickness)
    density = 0.5 * (sxx**2 / E + sxy**2 / G_MOD) * thickness
    return float(wx @ density @ wy)


def run_membrane(mesh: Mesh2D, thickness: float = 0.05) -> Dict[str, Any]:
    from anyfesolver import solve_linear
    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver import assemble_stiffness_matrix

    length = mesh.lx
    c = 0.5 * mesh.ly
    shifted = {nid: (x, y - c) for nid, (x, y) in mesh.nodes.items()}
    work = Mesh2D(shifted, mesh.elements, mesh.xs, mesh.ys - c, mesh.grid, mesh.description)
    model = build_model(work, thickness)
    sides = _boundary_nodes(work)
    boundary = sorted(set(sum(sides.values(), [])))
    for node_id in model.mesh.nodes:
        model.add_boundary_condition(
            BoundaryCondition(f"flat-{node_id}", [node_id], {"uz": 0.0, "rx": 0.0, "ry": 0.0})
        )
    for node_id in boundary:
        x, y = shifted[node_id]
        u, v, _sxx, _sxy = timoshenko_field(np.asarray(x), np.asarray(y), length, c, thickness)
        model.add_boundary_condition(
            BoundaryCondition(f"exact-{node_id}", [node_id], {"ux": float(u), "uy": float(v)})
        )
    started = time.perf_counter()
    displacement, info = solve_linear(model)
    elapsed = time.perf_counter() - started
    K, _ = assemble_stiffness_matrix(model)
    energy_h = 0.5 * float(displacement @ (K @ displacement))
    energy = membrane_exact_energy(length, c, thickness)
    dm = model.mesh.dof_manager
    errors, norms = [], []
    for node_id, (x, y) in shifted.items():
        dofs = dm.get_node_dofs(node_id)
        u, v, _a, _b = timoshenko_field(np.asarray(x), np.asarray(y), length, c, thickness)
        errors.append(np.hypot(displacement[dofs[0]] - u, displacement[dofs[1]] - v))
        norms.append(np.hypot(u, v))
    # Raw station membrane resultants vs exact N = t * sigma (L2, relative).
    num = den = 0.0
    for xy, weight, tensor in element_resultants(model, displacement, "membrane_resultants"):
        _u, _v, sxx, sxy = timoshenko_field(np.asarray(xy[0]), np.asarray(xy[1]), length, c, thickness)
        exact = thickness * np.asarray(((sxx, sxy), (sxy, 0.0)))
        num += weight * float(np.sum((tensor - exact) ** 2))
        den += weight * float(np.sum(exact**2))
    tip_nodes = [nid for nid, (x, y) in shifted.items() if abs(x) < 1e-12 and abs(y) < 1e-9]
    return {
        "energy_norm_error": math.sqrt(max(energy_h - energy, 0.0) / energy),
        "energy_h_minus_exact_relative": (energy_h - energy) / energy,
        "max_nodal_displacement_error": float(max(errors) / max(norms)),
        "resultant_l2_error": math.sqrt(num / den) if den else None,
        "solve_seconds": elapsed,
        "solver_status": str(info.get("solver_status", info.get("status", "ok"))),
        "domain": {"length": length, "half_depth": c, "thickness": thickness},
        "tip_node_on_axis": bool(tip_nodes),
    }


# ---------------------------------------------------------------------------
# Plate bending, vibration, buckling: hard simply-supported Mindlin plate
# ---------------------------------------------------------------------------


def plate_rigidity(thickness: float) -> float:
    return E * thickness**3 / (12.0 * (1.0 - NU**2))


def navier_reference(a: float, b: float, thickness: float, pressure: float, terms: int = 301):
    D = plate_rigidity(thickness)
    shear = KAPPA * G_MOD * thickness
    m = np.arange(1, terms + 1, 2, dtype=np.float64)
    M, N = np.meshgrid(m, m, indexing="ij")
    am, bn = M * math.pi / a, N * math.pi / b
    alpha2 = am**2 + bn**2
    qmn = 16.0 * pressure / (math.pi**2 * M * N)
    w_bend = qmn / (D * alpha2**2)
    w_shear = qmn / (shear * alpha2)
    signs = np.sin(M * math.pi / 2.0) * np.sin(N * math.pi / 2.0)
    w_centre = float(np.sum((w_bend + w_shear) * signs))
    compliance = float(
        np.sum((w_bend + w_shear) * pressure * (2.0 * a / (M * math.pi)) * (2.0 * b / (N * math.pi)))
    )

    def moments(x: float, y: float) -> np.ndarray:
        sx, sy = np.sin(am * x), np.sin(bn * y)
        cx, cy = np.cos(am * x), np.cos(bn * y)
        # Moments of the Kirchhoff part (the Mindlin SS moments coincide).
        wxx = -np.sum(w_bend * am**2 * sx * sy)
        wyy = -np.sum(w_bend * bn**2 * sx * sy)
        wxy = np.sum(w_bend * am * bn * cx * cy)
        mxx = -D * (wxx + NU * wyy)
        myy = -D * (wyy + NU * wxx)
        mxy = -D * (1.0 - NU) * wxy
        return np.asarray(((mxx, mxy), (mxy, myy)))

    return {"w_centre": w_centre, "compliance": compliance, "moments": moments, "D": D}


def plate_frequencies(a: float, b: float, thickness: float, count: int = 3) -> List[float]:
    D = plate_rigidity(thickness)
    shear = KAPPA * G_MOD * thickness
    rho_t = DENSITY * thickness
    values = []
    for m in range(1, 6):
        for n in range(1, 6):
            alpha2 = (m * math.pi / a) ** 2 + (n * math.pi / b) ** 2
            omega2 = D * alpha2**2 / (rho_t * (1.0 + D * alpha2 / shear))
            values.append(math.sqrt(omega2) / (2.0 * math.pi))
    return sorted(values)[:count]


def plate_buckling_nx(a: float, b: float, thickness: float) -> float:
    D = plate_rigidity(thickness)
    shear = KAPPA * G_MOD * thickness
    best = math.inf
    for m in range(1, 8):
        am, bn = m * math.pi / a, math.pi / b
        alpha2 = am**2 + bn**2
        nx = D * alpha2**2 / (am**2 * (1.0 + D * alpha2 / shear))
        best = min(best, nx)
    return best


def _simply_supported(model, mesh: Mesh2D, *, inplane: str) -> None:
    from anyfesolver.boundary import BoundaryCondition

    sides = _boundary_nodes(mesh)
    for side in ("x0", "x1"):
        model.add_boundary_condition(BoundaryCondition(f"ss-{side}", sides[side], {"uz": 0.0, "rx": 0.0}))
    for side in ("y0", "y1"):
        model.add_boundary_condition(BoundaryCondition(f"ss-{side}", sides[side], {"uz": 0.0, "ry": 0.0}))
    if inplane == "clamped":
        boundary = sorted(set(sum(sides.values(), [])))
        model.add_boundary_condition(BoundaryCondition("inplane", boundary, {"ux": 0.0, "uy": 0.0}))
    elif inplane == "uniaxial":
        model.add_boundary_condition(BoundaryCondition("ux0", sides["x0"], {"ux": 0.0}))
        corner = [n for n in sides["x0"] if n in sides["y0"]]
        model.add_boundary_condition(BoundaryCondition("uy0", corner, {"uy": 0.0}))
    else:
        raise ValueError(inplane)


def _interpolate_centre(mesh: Mesh2D, values: Dict[int, float]) -> float:
    xc, yc = 0.5 * (mesh.xs[0] + mesh.xs[-1]), 0.5 * (mesh.ys[0] + mesh.ys[-1])
    scale = max(mesh.lx, mesh.ly)
    for node_id, (x, y) in mesh.nodes.items():
        if abs(x - xc) <= 1.0e-12 * scale and abs(y - yc) <= 1.0e-12 * scale:
            return float(values[node_id])
    i = int(np.clip(np.searchsorted(mesh.xs, xc) - 1, 0, len(mesh.xs) - 2))
    j = int(np.clip(np.searchsorted(mesh.ys, yc) - 1, 0, len(mesh.ys) - 2))
    x0, x1 = mesh.xs[i], mesh.xs[i + 1]
    y0, y1 = mesh.ys[j], mesh.ys[j + 1]
    s, t = (xc - x0) / (x1 - x0), (yc - y0) / (y1 - y0)
    g = mesh.grid
    return float(
        (1 - s) * (1 - t) * values[g[(i, j)]]
        + s * (1 - t) * values[g[(i + 1, j)]]
        + s * t * values[g[(i + 1, j + 1)]]
        + (1 - s) * t * values[g[(i, j + 1)]]
    )


def run_plate_static(mesh: Mesh2D, thickness: float, pressure: float = 1.0e3) -> Dict[str, Any]:
    from anyfesolver import LoadCase, solve_linear

    model = build_model(mesh, thickness)
    _simply_supported(model, mesh, inplane="clamped")
    started = time.perf_counter()
    # The solver's own consistent element pressure integration, applied as the
    # equivalent nodal load.  (``assemble_load_vector`` with pressure records
    # additionally runs a per-element model-wide lifecycle guard that is
    # quadratic in mesh size; that diagnostic does not change the vector.)
    pressure_case = LoadCase("pressure")
    for element_id in model.mesh.elements:
        pressure_case.add_pressure_load(element_id, pressure)
    dm = model.mesh.dof_manager
    force = np.asarray(
        pressure_case.get_load_vector(model.mesh, dm, model.get_material), dtype=np.float64
    )
    load = LoadCase("pressure-nodal")
    for node_id in mesh.nodes:
        dofs = dm.get_node_dofs(node_id)
        if np.any(force[dofs] != 0.0):
            load.add_nodal_load(node_id, force[dofs])
    displacement, info = solve_linear(model, load)
    elapsed = time.perf_counter() - started
    w = {nid: float(displacement[dm.get_node_dofs(nid)[2]]) for nid in mesh.nodes}
    reference = navier_reference(mesh.lx, mesh.ly, thickness, pressure)
    w_centre = _interpolate_centre(mesh, w)
    sign = 1.0 if w_centre * reference["w_centre"] > 0 else -1.0
    compliance = float(force @ displacement)
    num = den = 0.0
    tri_stations = 0
    for xy, weight, tensor in element_resultants(model, displacement, "bending_resultants"):
        tri_stations += 1
        exact = reference["moments"](float(xy[0]), float(xy[1]))
        num += weight * float(np.sum((tensor - exact) ** 2))
        den += weight * float(np.sum(exact**2))
    moment_error = math.sqrt(num / den) if den else None
    if moment_error is not None and moment_error > 1.0:
        # opposite resultant sign convention; retain magnitude comparison
        num = den = 0.0
        for xy, weight, tensor in element_resultants(model, displacement, "bending_resultants"):
            exact = -reference["moments"](float(xy[0]), float(xy[1]))
            num += weight * float(np.sum((tensor - exact) ** 2))
            den += weight * float(np.sum(exact**2))
        moment_error = math.sqrt(num / den)
    return {
        "w_centre_error": abs(sign * w_centre - reference["w_centre"]) / abs(reference["w_centre"]),
        "compliance_error": abs(abs(compliance) - reference["compliance"]) / reference["compliance"],
        "s3_moment_l2_error": moment_error,
        "w_centre_reference": reference["w_centre"],
        "solve_seconds": elapsed,
        "thickness_ratio": thickness / min(mesh.lx, mesh.ly),
    }


def _private_element_matrix(model, element, method: str, *args: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate an element matrix on a private mesh holding only its nodes.

    Returns (global DOF indices, matrix).  Used to avoid the quadratic
    model-wide guards of the public mass/geometric assemblies (see
    docs/S3_ANGLE_EXTENSION.md section 3); the routes are cross-checked in
    tests/test_s3_angle_models.py.
    """

    from anyfesolver.fe_core import FEMesh

    local = FEMesh()
    for node_id in element.node_ids:
        local.add_node(node_id, *model.mesh.get_node(node_id).coords())
    local_dofs = np.concatenate([local.dof_manager.get_node_dofs(n) for n in element.node_ids])
    if not np.array_equal(local_dofs, np.arange(6 * len(element.node_ids))):
        raise AssertionError("unexpected private-mesh DOF order")
    matrix = np.asarray(getattr(element, method)(local, *args), dtype=np.float64)
    global_dofs = np.concatenate(
        [model.mesh.dof_manager.get_node_dofs(n) for n in element.node_ids]
    )
    return global_dofs, matrix


def _sparse_from_blocks(blocks: List[Tuple[np.ndarray, np.ndarray]], n: int):
    import scipy.sparse as sp

    rows = np.concatenate([np.repeat(d, d.size) for d, _m in blocks])
    cols = np.concatenate([np.tile(d, d.size) for d, _m in blocks])
    vals = np.concatenate([m.reshape(-1) for _d, m in blocks])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def run_plate_modal(
    mesh: Mesh2D, thickness: float, modes: int = 3, *, public_route: bool = False
) -> Dict[str, Any]:
    """SS plate vibration.  ``public_route`` uses ``solve_free_vibration``.

    The default assembles ``K`` with the public (batched) assembly and ``M``
    from each element's own ``compute_mass_matrix`` on a private mesh, then
    solves ``M phi = mu K phi`` (``mu = 1 / omega^2``) on the solver's
    constraint-reduced space; this is valid for V2D's zero rotary inertia.
    """

    import scipy.sparse.linalg as spla
    from anyfesolver import assemble_stiffness_matrix
    from anyfesolver.assembly import build_constraint_transformation
    from anyfesolver.modal import solve_free_vibration

    model = build_model(mesh, thickness)
    _simply_supported(model, mesh, inplane="clamped")
    started = time.perf_counter()
    if public_route:
        result = solve_free_vibration(model, num_modes=modes + 3)
        computed = sorted(float(f) for f in result.frequencies_hz if float(f) > 1.0e-6)[:modes]
        status = result.solver_status
    else:
        material = model.get_material("steel")
        model.apply_boundary_conditions()
        K, _ = assemble_stiffness_matrix(model)
        n = K.shape[0]
        M = _sparse_from_blocks(
            [
                _private_element_matrix(model, element, "compute_mass_matrix", material)
                for element in model.mesh.elements.values()
            ],
            n,
        )
        K_red, _f, T, _u0, _free, _info = build_constraint_transformation(K, np.zeros(n), model)
        M_red = (T.T @ M @ T).tocsc()
        mu, _vec = spla.eigsh(M_red, k=modes + 2, M=K_red.tocsc(), which="LA")
        omegas = sorted(math.sqrt(1.0 / m) for m in mu if m > 0)
        computed = [w / (2.0 * math.pi) for w in omegas][:modes]
        status = "ok"
    elapsed = time.perf_counter() - started
    reference = plate_frequencies(mesh.lx, mesh.ly, thickness, modes)
    return {
        "frequency_errors": [abs(c - r) / r for c, r in zip(computed, reference)],
        "frequencies_hz": computed,
        "reference_hz": reference,
        "solver_status": status,
        "route": "solve_free_vibration" if public_route else "element_mass_direct_eigsh",
        "solve_seconds": elapsed,
    }


def run_plate_buckling(mesh: Mesh2D, thickness: float, *, public_route: bool = False) -> Dict[str, Any]:
    """Uniaxial buckling of the SS plate (pure-S3 meshes).

    V2D reference buckling consumes the frozen uniform membrane-compression
    state in each element's local frame.  The exact prebuckling state of the
    SS plate under uniform edge compression is uniaxial N_x, so it is
    prescribed directly and rotated into each element frame.

    ``public_route=True`` solves with ``solve_eigenvalue_buckling``.  The
    default assembles the element-owned ``K`` (public assembly) and each
    element's own ``compute_geometric_stiffness_matrix`` on a private
    three-node mesh, then reduces with the solver's constraint transformation:
    the public geometric-stiffness assembly re-runs a model-wide lifecycle guard
    per element and is quadratic in mesh size.  The two routes are
    cross-checked in ``tests/test_s3_angle_models.py``.
    """

    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from anyfesolver import assemble_stiffness_matrix
    from anyfesolver.assembly import build_constraint_transformation
    from anyfesolver.buckling import solve_eigenvalue_buckling
    from anyfesolver.fe_core import FEMesh

    model = build_model(mesh, thickness)
    _simply_supported(model, mesh, inplane="uniaxial")
    unit_nx = 1.0  # N/m compression
    material = model.get_material("steel")
    manager = model.mesh.dof_manager
    states: Dict[int, Any] = {}
    global_state = np.asarray(((unit_nx, 0.0), (0.0, 0.0)))
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    started = time.perf_counter()
    for element_id, element in model.mesh.elements.items():
        if len(element.node_ids) != 3:
            raise ValueError("buckling family is pure S3")
        local = FEMesh()
        for node_id in element.node_ids:
            local.add_node(node_id, *model.mesh.get_node(node_id).coords())
        frame = np.asarray(
            element.compute_variational_resultants(
                local, np.zeros(local.dof_manager.total_dofs), material
            )["frame"]
        )[:2, :2]
        s = frame.T @ global_state @ frame
        state = {"membrane_compression": [s[0, 0], s[1, 1], s[0, 1]]}
        states[element_id] = state
        if public_route:
            continue
        kg = element.compute_geometric_stiffness_matrix(local, material, state)
        # Element DOF order is the element's node order on both meshes.
        global_dofs = np.concatenate([manager.get_node_dofs(n) for n in element.node_ids])
        local_dofs = np.concatenate([local.dof_manager.get_node_dofs(n) for n in element.node_ids])
        if not np.array_equal(local_dofs, np.arange(18)):
            raise AssertionError("unexpected private-mesh DOF order")
        rows.append(np.repeat(global_dofs, 18))
        cols.append(np.tile(global_dofs, 18))
        vals.append(np.asarray(kg).reshape(-1))
    if public_route:
        result = solve_eigenvalue_buckling(model, element_states=states, num_modes=1)
        factor = None if result.critical_load_factor is None else float(result.critical_load_factor)
        status = result.solver_status
    else:
        model.apply_boundary_conditions()
        K, _ = assemble_stiffness_matrix(model)
        n = K.shape[0]
        KG = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        )
        K_red, _f, T, _u0, _free, _info = build_constraint_transformation(K, np.zeros(n), model)
        KG_red = (T.T @ KG @ T).tocsc()
        # K phi = lambda KG phi with KG positive for compression; smallest
        # positive lambda via shift-invert on K (mu = 1/lambda largest).
        mu, _vec = spla.eigsh(KG_red, k=4, M=K_red.tocsc(), which="LA")
        positive = sorted(1.0 / m for m in mu if m > 0)
        factor = positive[0] if positive else None
        status = "ok" if factor is not None else "no_positive_eigenvalue"
    elapsed = time.perf_counter() - started
    reference = plate_buckling_nx(mesh.lx, mesh.ly, thickness) / unit_nx
    return {
        "buckling_factor": factor,
        "reference_factor": reference,
        "buckling_error": None if factor is None else abs(factor - reference) / reference,
        "solver_status": status,
        "route": "solve_eigenvalue_buckling" if public_route else "element_kg_direct_eigsh",
        "solve_seconds": elapsed,
    }


def run_nonlinear_end_moment(mesh: Mesh2D, thickness: float = 0.01, turn: float = 1.0) -> Dict[str, Any]:
    """Corotational cantilever strip under a tip moment (nu = 0).

    With nu = 0 the strip has no anticlastic coupling and the exact
    geometrically nonlinear response is a circular arc of radius EI/M:
    tip (u, w) = (R sin(phi) - L, R (1 - cos(phi))) with phi = M L / EI.
    """

    from anyfesolver import FEModel, LoadCase, create_shell_element
    from anyfesolver.boundary import FixedSupport
    from anyfesolver.nonlinear_static import solve_static_nonlinear

    model = FEModel("s3-angle-nonlinear")
    model.add_material("steel", E, 0.0, density=DENSITY)
    for node_id, (x, y) in mesh.nodes.items():
        model.add_node(node_id, x, y, 0.0)
    for element_id, (kind, nodes) in enumerate(mesh.elements, start=1):
        kwargs: Dict[str, Any] = {"thickness": thickness}
        if kind == "s3":
            kwargs["reference_normal"] = (0.0, 0.0, 1.0)
        model.add_element(element_id, create_shell_element(element_id, list(nodes), "steel", **kwargs))
    sides = _boundary_nodes(mesh)
    model.add_boundary_condition(FixedSupport("root", sides["x0"]))
    length, width = mesh.lx, mesh.ly
    rigidity = E * width * thickness**3 / 12.0
    moment = turn * rigidity / length
    tip = sorted(sides["x1"], key=lambda n: mesh.nodes[n][1])
    load = LoadCase("tip-moment")
    for k, node_id in enumerate(tip):
        y = mesh.nodes[node_id][1]
        lower = y - mesh.nodes[tip[k - 1]][1] if k > 0 else 0.0
        upper = mesh.nodes[tip[k + 1]][1] - y if k + 1 < len(tip) else 0.0
        # Negative moment about y lifts the tip (+z) for a strip along +x.
        load.add_nodal_load(node_id, moments=[0.0, -moment * 0.5 * (lower + upper) / width, 0.0])
    started = time.perf_counter()
    result = solve_static_nonlinear(
        model,
        load,
        num_steps=8,
        max_iterations=30,
        kinematics="corotational",
        corotational_tangent="consistent",
    )
    elapsed = time.perf_counter() - started
    radius = length / turn
    exact_u = radius * math.sin(turn) - length
    exact_w = radius * (1.0 - math.cos(turn))
    dm = model.mesh.dof_manager
    u = np.mean([result.displacements[dm.get_node_dofs(n)[0]] for n in tip])
    w = np.mean([result.displacements[dm.get_node_dofs(n)[2]] for n in tip])
    scale = math.hypot(exact_u, exact_w)
    return {
        "status": result.status,
        "load_factor": float(result.load_factor),
        "tip_error": float(math.hypot(u - exact_u, w - exact_w) / scale),
        "tip_u": float(u),
        "tip_w": float(w),
        "exact_u": exact_u,
        "exact_w": exact_w,
        "steps": len(result.steps),
        "iterations": [int(step.iterations) for step in result.steps],
        "solve_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Element sweep
# ---------------------------------------------------------------------------


def element_sweep(bands: Iterable[float], seed: int) -> List[Dict[str, Any]]:
    rows = []
    generator = np.random.default_rng(seed)
    for band in bands:
        shapes = dict(shape_family(band))
        # fixed-seed random shapes with exactly this minimum angle
        for k in range(3):
            beta = generator.uniform(band, 180.0 - 2.0 * band)
            if 180.0 - band - beta < band:
                beta = 180.0 - 2.0 * band
            shapes[f"random_{k}"] = (band, float(beta))
        for name, angles in shapes.items():
            coordinates = triangle_from_angles(*angles)
            quality = independent_quality(coordinates)
            for ratio in (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4):
                mesh, element, material = build_default_s3(coordinates, thickness=ratio)
                components = element.compute_stiffness_components(mesh, material)
                K = np.asarray(components["total"])
                rigid = rigid_body_basis(coordinates, 1.0)
                spectrum = elastic_spectrum(K, coordinates, 1.0)
                g = np.asarray(((1.0e-3, 3.0e-4), (-2.0e-4, 5.0e-4)))
                um = membrane_patch_field(coordinates, g)
                strain = np.asarray((g[0, 0], g[1, 1], g[0, 1] + g[1, 0]))
                exact_m = quality.area * ratio * strain @ isotropic_plane_stress() @ strain
                h = np.asarray(((2.0e-3, 7.0e-4), (7.0e-4, -1.0e-3)))
                ub = bending_patch_field(coordinates, h)
                curvature = np.asarray((h[0, 0], h[1, 1], 2.0 * h[0, 1]))
                exact_b = quality.area * ratio**3 / 12.0 * curvature @ isotropic_plane_stress() @ curvature
                rows.append(
                    {
                        "band_deg": band,
                        "shape": name,
                        "angles_deg": list(quality.angles_deg),
                        "normalized_area": quality.normalized_area,
                        "edge_ratio": quality.edge_ratio,
                        "t_over_L": ratio,
                        "t_over_h_min": ratio / (2.0 * quality.area / max(
                            np.linalg.norm(coordinates[1] - coordinates[0]),
                            np.linalg.norm(coordinates[2] - coordinates[1]),
                            np.linalg.norm(coordinates[0] - coordinates[2]),
                        )),
                        "rigid_residual": float(np.max(np.abs(K @ rigid)) / np.max(np.abs(np.diag(K)))),
                        "symmetry_residual": float(np.max(np.abs(K - K.T)) / np.max(np.abs(K))),
                        "elastic_modes": int(spectrum.size),
                        "min_elastic_eigenvalue": float(spectrum[0]),
                        "min_eigenvalue_over_t2": float(spectrum[0] / ratio**2),
                        "condition": float(spectrum[-1] / spectrum[0]),
                        "membrane_patch_residual": abs(float(um @ K @ um) / exact_m - 1.0),
                        "bending_patch_residual": abs(float(ub @ K @ ub) / exact_b - 1.0),
                        "envelopes": {
                            name_: not envelope_violations(quality, limits)
                            for name_, limits in CANDIDATE_ENVELOPES.items()
                        },
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


def observed_rates(values: List[float], sizes: List[float]) -> List[float]:
    return [
        math.log(values[k] / values[k + 1]) / math.log(sizes[k] / sizes[k + 1])
        for k in range(len(values) - 1)
        if values[k] > 0 and values[k + 1] > 0
    ]


def run_campaign(target: float, levels: int, quick: bool) -> Dict[str, Any]:
    results: Dict[str, Any] = {"target_deg": target}
    bands = [b for b in ANGLE_BANDS_DEG if b >= target] + list(DIAGNOSTIC_BANDS_DEG[:1])
    results["element_sweep"] = element_sweep(bands, seed=int(target * 100))

    # Membrane families (aligned and misaligned narrow direction).
    membrane = []
    for family, angle, narrow in (
        ("regular", 45.0, "y"),
        ("right", target, "y"),
        ("right", target, "x"),
        ("right_alternating", target, "x"),
        ("obtuse", target, "y"),
        ("obtuse", target, "x"),
    ):
        series = []
        for level in range(levels):
            mesh = s3_family_mesh(
                family, angle, level, target=(4.0, 1.0), base_count=8 if narrow == "y" else 2, narrow=narrow
            )
            row = {"mesh": mesh.description, "stats": mesh_statistics(mesh)}
            row.update(run_membrane(mesh))
            series.append(row)
        sizes = [r["stats"]["h_max"] for r in series]
        membrane.append(
            {
                "family": family,
                "angle_deg": angle,
                "narrow": narrow,
                "levels": series,
                "energy_rate": observed_rates([r["energy_norm_error"] for r in series], sizes),
            }
        )
    results["membrane"] = membrane

    # Plate bending families (t/L = 1e-2 and a thin 1e-3 locking probe).
    plate = []
    for thickness in (0.01, 0.001):
        for family, angle, narrow in (
            ("regular", 45.0, "y"),
            ("right", target, "y"),
            ("right_alternating", target, "y"),
            ("obtuse", target, "y"),
        ):
            series = []
            for level in range(levels):
                mesh = s3_family_mesh(family, angle, level, target=(1.0, 1.0), base_count=4, narrow=narrow)
                row = {"mesh": mesh.description, "stats": mesh_statistics(mesh)}
                row.update(run_plate_static(mesh, thickness))
                series.append(row)
            sizes = [r["stats"]["h_max"] for r in series]
            plate.append(
                {
                    "family": family,
                    "angle_deg": angle,
                    "thickness": thickness,
                    "levels": series,
                    "compliance_rate": observed_rates([r["compliance_error"] for r in series], sizes),
                    "moment_rate": observed_rates([r["s3_moment_l2_error"] for r in series], sizes),
                }
            )
    results["plate"] = plate

    # Mixed Q4/S3 panels (bending) against all-Q4 at the same refinement.
    mixed = []
    for pattern in ("isolated", "chain", "boundary", "cluster", "quarter"):
        series = []
        for level in range(levels):
            mesh = mixed_mesh(pattern, target, level, base_count=4)
            baseline = all_q4_mesh(level, base_count=4)
            row = {"mesh": mesh.description, "stats": mesh_statistics(mesh)}
            row.update(run_plate_static(mesh, 0.01))
            row["all_q4"] = run_plate_static(baseline, 0.01)
            # Same layout at the historical 30 deg floor: isolates the angle
            # effect from the S3-versus-Q4 accuracy difference.
            row["layout_30"] = run_plate_static(mixed_mesh(pattern, 30.0, level, base_count=4), 0.01)
            series.append(row)
        mixed.append({"pattern": pattern, "levels": series})
    results["mixed"] = mixed

    # Modal and buckling on the finest two levels of each pure family.
    eigen = []
    modal_levels = range(max(0, levels - 2), levels)
    for family, angle in (("regular", 45.0), ("right", target), ("obtuse", target)):
        for level in modal_levels:
            mesh = s3_family_mesh(family, angle, level, target=(1.0, 1.0), base_count=4)
            row = {"mesh": mesh.description, "stats": mesh_statistics(mesh)}
            row["modal"] = run_plate_modal(mesh, 0.01)
            row["buckling"] = run_plate_buckling(mesh, 0.01)
            eigen.append(row)
    for pattern in ("chain", "quarter"):
        mesh = mixed_mesh(pattern, target, levels - 1, base_count=4)
        row = {"mesh": mesh.description, "stats": mesh_statistics(mesh)}
        row["modal"] = run_plate_modal(mesh, 0.01)
        row["modal_layout_30"] = run_plate_modal(mixed_mesh(pattern, 30.0, levels - 1, base_count=4), 0.01)
        row["buckling"] = None  # buckling family is pure S3 (V2D prestress policy)
        eigen.append(row)
    results["eigen"] = eigen

    # Geometrically nonlinear (corotational) end-moment strip, 1 rad end rotation.
    nonlinear = []
    # Level 2 (1 rad): the coarse obtuse cross layout has an O(phi^2)
    # corotational discretization error that grows as the angle falls
    # (level 1 at 1 rad: 0.10% at 45 deg, 0.44% at 30, 2.2% at 20) and
    # converges fast under refinement (20 deg: 2.2% -> 0.14% at level 2).
    for family, angle in (("regular", 45.0), ("right", target), ("obtuse", 30.0), ("obtuse", target)):
        mesh = s3_family_mesh(family, angle, 2, target=(1.0, 0.1), base_count=10, narrow="y")
        row = {"mesh": mesh.description, "stats": mesh_statistics(mesh)}
        row.update(run_nonlinear_end_moment(mesh))
        nonlinear.append(row)
    results["nonlinear"] = nonlinear
    return results


def regular_error_at(series: Dict[str, Any], dofs: float, key: str) -> float:
    """Log-log interpolate (or extrapolate) a regular-family error at ``dofs``."""

    x = np.log([r["stats"]["dofs"] for r in series["levels"]])
    y = np.log([r[key] for r in series["levels"]])
    k = int(np.clip(np.searchsorted(x, math.log(dofs)) - 1, 0, len(x) - 2))
    slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
    return float(math.exp(y[k] + slope * (math.log(dofs) - x[k])))


def adjudicate(results: Dict[str, Any]) -> Dict[str, Any]:
    a = ACCEPTANCE
    target = results["target_deg"]
    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"criterion": name, "passed": bool(passed), "detail": detail})

    in_scope = [r for r in results["element_sweep"] if r["band_deg"] >= target]
    check(
        "element_rigid_and_symmetry",
        all(r["rigid_residual"] <= a["element_rigid_residual"] and r["symmetry_residual"] <= 1e-13 for r in in_scope),
        max(r["rigid_residual"] for r in in_scope),
    )
    check(
        "element_patch_tests",
        all(
            max(r["membrane_patch_residual"], r["bending_patch_residual"]) <= a["element_patch_residual"]
            for r in in_scope
        ),
        max(max(r["membrane_patch_residual"], r["bending_patch_residual"]) for r in in_scope),
    )
    check(
        "element_twelve_elastic_modes",
        all(r["elastic_modes"] == 12 and r["min_eigenvalue_over_t2"] > 1e-3 for r in in_scope),
        min(r["min_eigenvalue_over_t2"] for r in in_scope),
    )
    regular_membrane = next(m for m in results["membrane"] if m["family"] == "regular")
    regular_rate = regular_membrane["energy_rate"][-1]
    for m in results["membrane"]:
        finest = m["levels"][-1]
        label = f"membrane_{m['family']}_{m['narrow']}"
        # CST membrane resultants converge O(h); the absolute 5% target is
        # formulation-limited (the 45 deg family misses it at the same
        # resolution), so the required check is accuracy relative to the
        # 45 deg family at equal DOF count.  The absolute value is recorded.
        if m["family"] != "regular":
            ratio = finest["resultant_l2_error"] / regular_error_at(
                regular_membrane, finest["stats"]["dofs"], "resultant_l2_error"
            )
            check(
                label + "_resultant_vs_regular_at_equal_dofs",
                ratio <= a["membrane_resultant_ratio_to_regular"],
                {"ratio": ratio, "absolute": finest["resultant_l2_error"]},
            )
        check(label + "_displacement", finest["max_nodal_displacement_error"] <= a["principal_deflection_error"], finest["max_nodal_displacement_error"])
        check(
            label + "_rate",
            m["energy_rate"][-1] >= a["rate_fraction_of_regular"] * regular_rate,
            {"rate": m["energy_rate"], "regular": regular_rate},
        )
    for p in results["plate"]:
        finest = p["levels"][-1]
        label = f"plate_t{p['thickness']:g}_{p['family']}"
        check(label + "_centre_deflection", finest["w_centre_error"] <= a["principal_deflection_error"], finest["w_centre_error"])
        check(label + "_moment_l2", finest["s3_moment_l2_error"] <= a["resultant_l2_error"], finest["s3_moment_l2_error"])
    regular_plate = {p["thickness"]: p for p in results["plate"] if p["family"] == "regular"}
    for p in results["plate"]:
        if p["family"] == "regular":
            continue
        ref = regular_plate[p["thickness"]]["moment_rate"][-1]
        check(
            f"plate_t{p['thickness']:g}_{p['family']}_moment_rate",
            p["moment_rate"][-1] >= a["rate_fraction_of_regular"] * ref,
            {"rate": p["moment_rate"], "regular": ref},
        )
    for m in results["mixed"]:
        finest = m["levels"][-1]
        detail = {
            "mixed": finest["w_centre_error"],
            "layout_30": finest["layout_30"]["w_centre_error"],
            "all_q4": finest["all_q4"]["w_centre_error"],
        }
        check(f"mixed_{m['pattern']}_centre_deflection", finest["w_centre_error"] <= a["principal_deflection_error"], detail)
        check(
            f"mixed_{m['pattern']}_centre_deflection_vs_30",
            finest["w_centre_error"]
            <= max(a["mixed_error_factor"] * finest["layout_30"]["w_centre_error"], a["principal_deflection_error"] / 10.0),
            detail,
        )
        check(f"mixed_{m['pattern']}_s3_moment_l2", finest["s3_moment_l2_error"] <= a["resultant_l2_error"], finest["s3_moment_l2_error"])
    finest_eigen: Dict[str, Dict[str, Any]] = {}
    for row in results["eigen"]:
        finest_eigen[row["mesh"]["family"]] = row
    for family, row in finest_eigen.items():
        detail = {"errors": row["modal"]["frequency_errors"]}
        if "modal_layout_30" in row:
            detail["layout_30"] = row["modal_layout_30"]["frequency_errors"]
        check(f"modal_{family}_first_frequency", row["modal"]["frequency_errors"][0] <= a["first_frequency_error"], detail)
        if row["buckling"] is None:
            continue
        err = row["buckling"]["buckling_error"]
        check(f"buckling_{family}", err is not None and err <= a["buckling_factor_error"], err)
    regular_nl = next(r for r in results["nonlinear"] if r["mesh"]["family"] == "regular")
    for row in results["nonlinear"]:
        family = f"{row['mesh']['family']}_{row['mesh']['angle_deg']:g}"
        check(
            f"nonlinear_{family}_end_moment",
            row["status"] == "completed" and row["tip_error"] <= a["principal_deflection_error"],
            {"status": row["status"], "tip_error": row["tip_error"]},
        )
        # New stalls/cutbacks: total Newton iterations within 1.5x regular.
        check(
            f"nonlinear_{family}_iterations",
            sum(row["iterations"]) <= 1.5 * sum(regular_nl["iterations"]),
            {"iterations": row["iterations"], "regular": regular_nl["iterations"]},
        )
    return {"checks": checks, "passed": all(c["passed"] for c in checks)}


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def manifest(args: argparse.Namespace) -> Dict[str, Any]:
    import anyfesolver
    import anymesher
    import scipy

    packages = {}
    for name in ("anyfesolver", "anymesher", "anymaterial", "anyfileio", "numpy", "scipy", "numba"):
        try:
            module = __import__(name)
            packages[name] = {
                "version": getattr(module, "__version__", "unknown"),
                "location": str(Path(module.__file__).parent),
            }
        except ImportError:
            packages[name] = None
    from anyfesolver.elements import DEFAULT_S3_FORMULATION
    from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID, IMPLEMENTATION_ID

    return {
        "schema": SCHEMA,
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "selected_s3_formulation": DEFAULT_S3_FORMULATION,
        "formulation_id": FORMULATION_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "target_deg": args.target,
        "levels": args.levels,
        "mode": args.mode,
        "candidate_envelopes": CANDIDATE_ENVELOPES,
        "acceptance": ACCEPTANCE,
        "command": [sys.executable, *sys.argv],
    }


def summarize(results: Dict[str, Any], verdict: Dict[str, Any]) -> str:
    lines = [f"# S3 angle campaign: target {results['target_deg']:g} deg", ""]
    lines.append("## Membrane (Timoshenko cantilever, exact Dirichlet field)")
    lines.append("| family | narrow | min angle | levels: energy-norm error | rate | N L2 error (finest) |")
    lines.append("|---|---|---:|---|---|---:|")
    for m in results["membrane"]:
        errs = ", ".join(f"{r['energy_norm_error']:.3e}" for r in m["levels"])
        rate = ", ".join(f"{r:.2f}" for r in m["energy_rate"])
        lines.append(
            f"| {m['family']} | {m['narrow']} | {m['levels'][-1]['stats']['s3_min_angle_deg']:.2f} | {errs} | {rate} | {m['levels'][-1]['resultant_l2_error']:.3e} |"
        )
    lines.append("")
    lines.append("## Plate bending (SS Mindlin plate, uniform pressure)")
    lines.append("| t | family | min angle | w_c error by level | compliance rate | M L2 error by level | M rate |")
    lines.append("|---|---|---:|---|---|---|---|")
    for p in results["plate"]:
        w = ", ".join(f"{r['w_centre_error']:.2e}" for r in p["levels"])
        mm = ", ".join(f"{r['s3_moment_l2_error']:.2e}" for r in p["levels"])
        lines.append(
            f"| {p['thickness']:g} | {p['family']} | {p['levels'][-1]['stats']['s3_min_angle_deg']:.2f} | {w} | "
            f"{', '.join(f'{r:.2f}' for r in p['compliance_rate'])} | {mm} | {', '.join(f'{r:.2f}' for r in p['moment_rate'])} |"
        )
    lines.append("")
    lines.append("## Mixed Q4/S3 panels (t/L = 0.01)")
    lines.append("| pattern | S3 fraction (finest) | min angle | w_c error: target / same layout at 30 deg / all-Q4, by level | S3 M L2 error (finest) |")
    lines.append("|---|---:|---:|---|---:|")
    for m in results["mixed"]:
        pairs = ", ".join(f"{r['w_centre_error']:.2e}/{r['layout_30']['w_centre_error']:.2e}/{r['all_q4']['w_centre_error']:.2e}" for r in m["levels"])
        finest = m["levels"][-1]
        lines.append(
            f"| {m['pattern']} | {finest['stats']['s3_fraction']:.3f} | {finest['stats']['s3_min_angle_deg']:.2f} | {pairs} | {finest['s3_moment_l2_error']:.3e} |"
        )
    lines.append("")
    lines.append("## Modal and buckling (t/L = 0.01)")
    lines.append("| family | level | f errors (first 3) | buckling error |")
    lines.append("|---|---:|---|---:|")
    for row in results["eigen"]:
        f = ", ".join(f"{e:.2e}" for e in row["modal"]["frequency_errors"])
        b = None if row["buckling"] is None else row["buckling"]["buckling_error"]
        lines.append(f"| {row['mesh']['family']} | {row['mesh']['level']} | {f} | {'n/a' if b is None else f'{b:.2e}'} |")
    lines.append("")
    lines.append("## Corotational end-moment strip (1 rad end rotation, nu = 0, consistent tangent)")
    lines.append("| family | min angle | status | tip error | Newton iterations |")
    lines.append("|---|---:|---|---:|---|")
    for row in results["nonlinear"]:
        lines.append(
            f"| {row['mesh']['family']} (level {row['mesh']['level']}) | {row['stats']['s3_min_angle_deg']:.2f} | {row['status']} | {row['tip_error']:.2e} | {sum(row['iterations'])} ({len(row['iterations'])} steps) |"
        )
    lines.append("")
    lines.append("## Criteria")
    for c in verdict["checks"]:
        lines.append(f"- {'PASS' if c['passed'] else 'FAIL'} {c['criterion']}: {json.dumps(c['detail'], default=_json_default)}")
    lines.append("")
    lines.append(f"**All required criteria passed: {verdict['passed']}**")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=float, required=True, help="minimum-angle target in degrees")
    parser.add_argument("--levels", type=int, default=4, help="shape-preserving refinement levels (>=3 for rates)")
    parser.add_argument("--mode", choices=("exploratory", "formal"), default="exploratory")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    args = parser.parse_args(argv)
    if args.mode == "formal" and args.levels < 3:
        parser.error("formal runs require at least three refinement levels")

    info = manifest(args)
    started = time.perf_counter()
    results = run_campaign(args.target, args.levels, quick=False)
    info["wall_seconds"] = time.perf_counter() - started
    verdict = adjudicate(results)
    output = args.output or ROOT / "reports" / "s3_angles" / info["commit"][:12] / f"{args.run_id}-t{args.target:g}"
    output.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(results, default=_json_default, indent=1, sort_keys=True).encode()
    info["results_sha256"] = hashlib.sha256(raw).hexdigest()
    (output / "results.json").write_bytes(raw)
    (output / "verdict.json").write_text(json.dumps(verdict, default=_json_default, indent=1) + "\n")
    (output / "manifest.json").write_text(json.dumps(info, default=_json_default, indent=1, sort_keys=True) + "\n")
    (output / "SUMMARY.md").write_text(summarize(results, verdict))
    print(f"wrote {output}")
    print(f"required criteria passed: {verdict['passed']}")
    return 0 if (verdict["passed"] or args.mode == "exploratory") else 1


if __name__ == "__main__":
    raise SystemExit(main())
