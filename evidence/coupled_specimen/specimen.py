"""Coupled shell-solid specimen, S3 of the dense coupled route (SOLVER-PR67-DENSE-SPEC section 7, 8 S3).

Measurement script, not a test. One command per fresh process; prints one JSON object.

    python specimen.py [--threads 1|default] coupled N RUNS     coupled solve at h = 1/N (warm-up + RUNS)
    python specimen.py [--threads ...] shell N                  native all-shell plate at h = 1/N
    python specimen.py [--threads ...] solid N LAYERS           all-solid P2 plate (ANYsolid backend), h_s = 1/N
    python specimen.py [--threads ...] oneway N RUNS            native shell + one-way solid on the same plate
    python specimen.py [--threads ...] svd RUNS                 dense SVD at N = 5000 (warm-up + RUNS)
    python specimen.py [--threads ...] peak N                   peak commit of one coupled solve (fresh process)

Needs PYTHONPATH = <repo>/src;<repo>/packages/results/src;<repo>/tests (and mpmath for the tests only).
"""

from __future__ import annotations

import json
import os
import sys

ARGS = sys.argv[1:]
THREADS = 'default'
if ARGS[:1] == ['--threads']:
    THREADS = ARGS[1]
    ARGS = ARGS[2:]
if THREADS == '1':
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
# DIAGNOSTIC ONLY (never a gate): --diagnostic-refinement K sets the candidate's refinement-step count
# in this measurement process, to show what the widened specimen gives if the candidate were refined
# further. The production route uses its own count (2); see RESULT.md, finding R1-3.
DIAGNOSTIC_REFINEMENT = None
if ARGS[:1] == ['--diagnostic-refinement']:
    DIAGNOSTIC_REFINEMENT = int(ARGS[1])
    ARGS = ARGS[2:]

import ctypes  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from ctypes import wintypes  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse import linalg as splinalg  # noqa: E402

import _coupled_fixtures as fx  # noqa: E402
from anyfesolver import solid_coupling as sc  # noqa: E402
from anyfesolver.assembly import assemble_system, solve_linear  # noqa: E402
from anyfesolver.solid_submodel import solve_shell_driven_solid  # noqa: E402

L, T, P = 1.0, 0.02, 1000.0
SIDES = ('side_0', 'side_1', 'side_2', 'side_3')
# Footprint (OD11, 2026-11-20, review r3 / OD10 R1 (a)): widened from [0.4, 0.6]^2 to [0.3, 0.7]^2 so that
# h = 0.1 has 4 x 4 lattice cells across the solid. Two layers would give N = 6703 > 5000 at h = 0.05
# (measured), so the lattice has one P2 layer through the thickness.
X0, X1 = 0.3, 0.7
A = X1 - X0                               # footprint side, centred
LAYERS = 1
SAMPLE_POINT = (X0 + 0.25 * A, 0.5)       # 0.25 a inside the footprint edge, on the centre line
PROTOCOL_RADIUS = 0.05 * A                # rev2 section 5.2
DIAGNOSTIC_RADIUS = 0.25 * A              # non-gating diagnostic (the protocol set is empty, see RESULT.md)


def peak_pagefile() -> int:
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32 = ctypes.WinDLL('kernel32')
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError('GetProcessMemoryInfo failed')
    return int(counters.PeakPagefileUsage)


POST_IMPORT_PEAK = peak_pagefile() if platform.system() == 'Windows' else None


def environment():
    blas = [v for k, v in np.show_config(mode='dicts')['Build Dependencies'].items() if k in ('blas', 'lapack')]
    import anysolid
    import skfem
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'skfem': skfem.__version__, 'anysolid': getattr(anysolid, '__version__', 'unknown'),
            'blas': [f"{b.get('name')} {b.get('version')}" for b in blas], 'threads': THREADS,
            'machine': platform.machine(), 'cpu_count': os.cpu_count()}


def hole(n):
    k0, k1 = round(X0 * n), round(X1 * n)
    assert abs(k0 / n - X0) < 1e-12 and abs(k1 / n - X1) < 1e-12, 'the footprint must lie on element edges'
    return (k0, k1, k0, k1)


def shell_points(n):
    """Centre and quarter point (0.25, 0.5); at n = 10 the quarter point is no node: x = 0.2 and 0.3."""
    xs = [0.25] if n % 4 == 0 else [0.2, 0.3]
    return {'centre': (0.5, 0.5), **{f'quarter_x{x:g}': (x, 0.5) for x in xs}}


def node_at(grid, n, x, y):
    i, j = round(x * n), round(y * n)
    assert abs(i / n - x) < 1e-12 and abs(j / n - y) < 1e-12
    return grid[i, j]


def coefficient_at(prepared, point, component):
    coords = np.asarray(prepared.coefficient_coordinates)
    comps = np.asarray(prepared.coefficient_components)
    hit = np.flatnonzero((np.abs(coords - np.asarray(point)).max(axis=1) < 1e-12) & (comps == component))
    assert len(hit) == 1, (point, hit)
    return int(hit[0])


def kappa_xx(prepared, q, radius):
    """kappa_xx = (mean eps_xx on 'end' - mean eps_xx on 'start') / t, area-weighted at the centroids of
    the skin triangles whose in-plane centroid lies within ``radius`` of SAMPLE_POINT. eps_xx = du_x/dx by
    a central difference along x inside the owning face (exact for the quadratic P2 field)."""
    mesh = prepared.mesh
    coords = np.asarray(mesh.coordinates, dtype=float)
    out = {}
    for name in ('end', 'start'):
        facets = np.asarray(mesh.boundary_facets)[np.isin(mesh.boundary_ids, mesh.boundary_groups[name])]
        tri = coords[facets]
        centroid = tri.mean(axis=1)
        pick = np.hypot(centroid[:, 0] - SAMPLE_POINT[0], centroid[:, 1] - SAMPLE_POINT[1]) <= radius
        if pick.sum() < 2:
            return {'count_end': int(pick.sum()) if name == 'end' else out.get('count_end'),
                    'failed': f'fewer than 2 {name!r} triangles within {radius:g}', 'kappa_xx': None}
        c = centroid[pick]
        area = 0.5 * np.linalg.norm(np.cross(tri[pick, 1] - tri[pick, 0], tri[pick, 2] - tri[pick, 0]), axis=1)
        delta = 1e-6 * np.sqrt(area.min())
        plus, minus = c.copy(), c.copy()
        plus[:, 0] += delta
        minus[:, 0] -= delta
        up = sparse.csr_matrix(prepared.trace(plus)) @ q
        um = sparse.csr_matrix(prepared.trace(minus)) @ q
        eps = (up[0::3] - um[0::3]) / (2 * delta)
        out[name] = float(np.sum(area * eps) / np.sum(area))
        out[f'count_{name}'] = int(pick.sum())
    out['kappa_xx'] = (out['end'] - out['start']) / T
    return out


def coupled(n, runs):
    k0, k1, _, _ = hole(n)
    nx = k1 - k0
    model, grid, removed, footprint = fx.plate(n, thickness=T, hole=hole(n))
    case = fx.pressure_case(removed, model, P)
    prepared = fx.prepare(fx.box_mesh(footprint, T, nx, nx, LAYERS))
    box = sc.BoxFaces((nx, nx, LAYERS), 'end', SIDES)
    if DIAGNOSTIC_REFINEMENT is not None:
        from anyfesolver import _coupled_certificate as cc
        cc._REFINEMENT_STEPS = DIAGNOSTIC_REFINEMENT
    timings, result = [], None
    for _ in range(runs + 1):
        t0 = time.perf_counter()
        try:
            result = sc.solve_shell_solid_coupled(model, prepared, removed_element_ids=removed, box=box,
                                                  load_case=case)
        except sc.CoupledSolveError as exc:          # a refusal is a result: record it, never retry
            wall = time.perf_counter() - t0
            d = exc.diagnostics
            return {'cmd': 'coupled', 'n': n, 'h': 1 / n, 'refused': {'code': exc.code, 'reason': exc.reason},
                    'forward_error': d.get('forward_error'),
                    'certificate': {k: d.get('certificate', {}).get(k) for k in
                                    ('accepted', 'condition_upper_bound', 'sigma_min', 'sigma_max', 'size')},
                    'wall_to_refusal': wall, 'diagnostic_refinement': DIAGNOSTIC_REFINEMENT}
        wall = time.perf_counter() - t0
        timings.append({**{k: float(v) for k, v in result.timings.items()}, 'wall': wall})
    measured = timings[1:] if runs else timings
    median = {k: statistics.median(t[k] for t in measured) for k in measured[0]}
    u = result.shell_displacements
    index = {nid: k for k, nid in enumerate(result.shell_node_ids)}
    obs = {}
    for name, (x, y) in shell_points(n).items():
        if name == 'centre':
            i = coefficient_at(prepared, (x, y, 0.0), 2)
            obs[name] = {'w': float(result.solid_coefficients.values[i]),
                         'bound': float(result.solid_coefficients.bounds[i]), 'source': 'solid (x, y, 0)'}
        else:
            k = 6 * index[node_at(grid, n, x, y)] + 2
            obs[name] = {'w': float(u.values[k]), 'bound': float(u.bounds[k]), 'source': 'shell node'}
    reaction = float(result.reactions.reshape(-1, 6)[:, 2].sum())
    q = result.solid_coefficients.values
    return {'cmd': 'coupled', 'n': n, 'h': 1 / n, 'sizes': result.sizes, 'runs': runs,
            'diagnostic_refinement': DIAGNOSTIC_REFINEMENT,
            'certificate': {k: result.certificate.get(k) for k in ('accepted', 'condition_upper_bound')},
            'verification': {k: v for k, v in result.verification.items() if k != 'backward'},
            'backward_max_ratio': result.verification['backward']['max_ratio'], 'traction': result.echo['traction'],
            'observables': obs, 'reaction_z': reaction,
            'kappa_protocol': kappa_xx(prepared, q, PROTOCOL_RADIUS),
            'kappa_diagnostic': kappa_xx(prepared, q, DIAGNOSTIC_RADIUS),
            'timings_all': timings, 'timings_median': median}


def shell(n):
    model, grid, removed, _ = fx.plate(n, thickness=T, hole=(0, 0, 0, 0))
    case = fx.pressure_case(removed, model, P)
    t0 = time.perf_counter()
    u, _info = solve_linear(model, case)
    wall = time.perf_counter() - t0
    K, F, _ = assemble_system(model, case)
    r = K @ u - F
    uz = [model.mesh.nodes[nid].dofs[2] for nid in model.mesh.nodes]
    obs = {name: {'w': float(u[model.mesh.nodes[node_at(grid, n, x, y)].dofs[2]])}
           for name, (x, y) in shell_points(n).items()}
    return {'cmd': 'shell', 'n': n, 'h': 1 / n, 'dofs': int(len(u)), 'observables': obs,
            'reaction_z': float(r[uz].sum()), 'load_z': float(F[uz].sum()), 'wall': wall}


def traction_vector():
    model, grid, removed, footprint = fx.plate(10, thickness=T, hole=hole(10))
    nx = hole(10)[1] - hole(10)[0]
    return np.asarray(sc.solve_shell_solid_coupled(
        model, fx.prepare(fx.box_mesh(footprint, T, nx, nx, LAYERS)), removed_element_ids=removed,
        box=sc.BoxFaces((nx, nx, LAYERS), 'end', SIDES), load_case=fx.pressure_case(removed, model, P)).echo['traction'])


def solid(n, layers):
    q_vec = traction_vector()
    t0 = time.perf_counter()
    prepared = fx.prepare(fx.box_mesh((0.0, L, 0.0, L), T, n, n, layers))
    K = prepared.assemble_linear().tocsr()
    f = np.asarray(prepared.traction('end', q_vec), dtype=float)
    x = np.asarray(prepared.coefficient_coordinates)
    c = np.asarray(prepared.coefficient_components)
    on_edge = (np.abs(x[:, 2]) < 1e-12) & ((np.abs(x[:, 0]) < 1e-12) | (np.abs(x[:, 0] - L) < 1e-12)
                                          | (np.abs(x[:, 1]) < 1e-12) | (np.abs(x[:, 1] - L) < 1e-12))
    fixed = np.flatnonzero(on_edge & (c == 2))       # w = 0 on the mid-surface edge lines
    fixed = np.concatenate([fixed, [coefficient_at(prepared, (0, 0, 0), 0), coefficient_at(prepared, (0, 0, 0), 1),
                                    coefficient_at(prepared, (L, 0, 0), 1)]])
    free = np.setdiff1d(np.arange(prepared.ndof), fixed)
    assembled = time.perf_counter() - t0
    lu = splinalg.splu(K[free][:, free].tocsc())
    u = np.zeros(prepared.ndof)
    u[free] = lu.solve(f[free])
    K_ff = K[free][:, free]
    for _ in range(2):                                  # iterative refinement (the reference carries no claim)
        u[free] += lu.solve(f[free] - K_ff @ u[free])
    residual = K[free] @ u - f[free]
    wall = time.perf_counter() - t0
    reaction = K @ u - f
    obs = {name: {'w': float(u[coefficient_at(prepared, (px, py, 0.0), 2)])}
           for name, (px, py) in {'centre': (0.5, 0.5), 'quarter_x0.25': (0.25, 0.5)}.items()
           if abs(px * n - round(px * n)) < 1e-12}
    return {'cmd': 'solid', 'n': n, 'h_s': 1 / n, 'layers': layers, 'ndof': int(prepared.ndof),
            'observables': obs, 'reaction_z': float(reaction[fixed][c[fixed] == 2].sum()),
            'load_z': float(f[c == 2].sum()), 'relative_residual': float(np.abs(residual).max() / np.abs(f).max()),
            'kappa_protocol': kappa_xx(prepared, u, PROTOCOL_RADIUS),
            'kappa_diagnostic': kappa_xx(prepared, u, DIAGNOSTIC_RADIUS), 'assembly': assembled, 'wall': wall}


def oneway(n, runs):
    k0, k1, _, _ = hole(n)
    nx = k1 - k0
    q_vec = traction_vector()
    model, grid, _, _ = fx.plate(n, thickness=T, hole=(0, 0, 0, 0))
    case = fx.pressure_case([], model, P)
    footprint = (k0 / n, k1 / n, k0 / n, k1 / n)
    prepared = fx.prepare(fx.box_mesh(footprint, T, nx, nx, LAYERS))
    h = L / n

    def locate(point):
        i = min(int(np.floor(point[0] / h + 1e-9)), n - 1)
        j = min(int(np.floor(point[1] / h + 1e-9)), n - 1)
        return [(j * n + i + 1, (2 * (point[0] / h - i) - 1, 2 * (point[1] / h - j) - 1))]
    rows = []
    for _ in range(runs + 1):
        t0 = time.perf_counter()
        u, _info = solve_linear(model, case)
        t1 = time.perf_counter()
        solve_shell_driven_solid(model, u, prepared, cut_groups=SIDES, locate=locate, skin_tractions={'end': q_vec})
        t2 = time.perf_counter()
        rows.append({'native_shell': t1 - t0, 'one_way_solid': t2 - t1, 'total': t2 - t0})
    measured = rows[1:] if runs else rows
    return {'cmd': 'oneway', 'n': n, 'runs': runs, 'timings_all': rows,
            'timings_median': {k: statistics.median(r[k] for r in measured) for k in measured[0]}}


def svd(runs):
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((5000, 5000))
    matrix = matrix + matrix.T
    from anyfesolver import _coupled_certificate as cc
    times = []
    for _ in range(runs + 1):
        t0 = time.perf_counter()
        cc._svd_values(matrix)      # the production call: one verified BLAS thread (review r2 (a))
        times.append(time.perf_counter() - t0)
    return {'cmd': 'svd', 'N': 5000, 'runs': runs, 'times': times,
            'median': statistics.median(times[1:] if runs else times)}


def peak(n):
    base = POST_IMPORT_PEAK
    out = coupled(n, 0)
    return {'cmd': 'peak', 'n': n, 'N': out['sizes']['N'] if 'sizes' in out else out['certificate']['size'],
            'refused': out.get('refused'), 'post_import_peak_bytes': base,
            'peak_commit_bytes': peak_pagefile(), 'peak_commit_minus_post_import_mib': (peak_pagefile() - base) / 2 ** 20}


def main():
    cmd, rest = ARGS[0], [int(a) for a in ARGS[1:]]
    out = {'coupled': coupled, 'shell': shell, 'solid': solid, 'oneway': oneway, 'svd': svd, 'peak': peak}[cmd](*rest)
    out['environment'] = environment()
    out['cpu_load_note'] = 'machine shared with other sessions; see RESULT.md'
    print(json.dumps(out, default=float))


if __name__ == '__main__':
    main()
