"""Local discontinuous dual/P0 planar mortar falsification, not F03 acceptance.

Translation tests are elementwise biorthogonal to native nodal P1 traces;
yaw tests are cellwise P0. Reassemble all constraint and mass diagnostics.
"""

from __future__ import annotations

from dataclasses import replace
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

if not os.environ.get('NUMBA_CACHE_DIR'):
    raise RuntimeError('set an explicit writable NUMBA_CACHE_DIR before import')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import anyfesolver
import numpy as np
import scipy
import numba
from scipy import linalg, sparse

from _b3_e4_f03_native_fixture import build
from _b3_e4_f03_shell_trace import make_shell
from probe_b3_ge_e4_pl_g0_f03_kkt import (
    ELL, CandidateSingular, PrimalNotAdmissible, assemble,
    candidate_rank_context, physical_support_balance, scales, solve_exact,
)
from probe_b3_ge_e4_pl_g0_f03_k1_rank import candidate_spaces
from probe_b3_ge_e4_pl_g0_f03_k2p import fixed_load, physical_stations, U


CASES = (('Q4', 8, 1), ('Q4', 8, 64),
         ('S3-V2D', 8, 1), ('S3-V2D', 8, 64))


def test_matrix(n: int, cell: int, xi: float) -> np.ndarray:
    """Rows: x/y at free nodes 1..n; yaw on cells 0..n-1."""
    out = np.zeros((3*n, 3))
    left, right = 2*(1-xi)-xi, 2*xi-(1-xi)
    if cell:
        out[cell-1, 0] = left
        out[n+cell-1, 1] = left
    out[cell, 0] = right
    out[n+cell, 1] = right
    out[2*n+cell, 2] = 1.
    return out


def integrated(beam, shell, overlay, order):
    n = beam.n_half_cells
    roots, weights = np.polynomial.legendre.leggauss(order)
    free = shell.free_count
    ncols = free + len(beam.physical) + len(beam.internal)
    c = np.zeros((3*n, ncols))
    mass = np.zeros((3*n, 3*n))
    full = np.zeros((3*n, shell.dof_count+beam.context.count))
    for a, b in zip(overlay[:-1], overlay[1:]):
        cell = min(int((a+b)*.5*n), n-1)
        cell_a = cell/n
        for root, weight in zip(roots, weights):
            s = float(.5*(a+b)+.5*(b-a)*root)
            xi = (s-cell_a)*n
            tests = test_matrix(n, cell, xi)
            st = shell.trace(s)
            bt = beam.planar_trace_jacobian(s)
            trace = np.diag((1., 1., ELL)) @ np.hstack(
                (-st[:, shell.free], bt[:, beam.physical], bt[:, beam.internal]))
            trace_full = np.diag((1., 1., ELL)) @ np.hstack((-st, bt))
            w = .5*(b-a)*weight
            c += w*tests@trace
            full += w*tests@trace_full
            mass += w*tests@tests.T
    return c, mass, full


def analyze(topology: str, n_beam: int, n_shell: int):
    beam, shell = build(n_beam), make_shell(n_shell, topology)
    spaces = {s.name: s for s in candidate_spaces(beam, shell)}
    p1 = assemble(beam, shell, spaces['p1-beam'])
    p0 = assemble(beam, shell, spaces['p0-beam'])
    # K4's blockwise selected rows are the historical comparison.
    x = [3*i for i in range(1, n_beam+1)]
    y = [3*i+1 for i in range(1, n_beam+1)]
    yaw = [3*i+2 for i in range(n_beam)]
    k4 = np.vstack((p1.matrix[x].toarray(), p1.matrix[y].toarray(),
                    p0.matrix[yaw].toarray()))
    c, mass, full = integrated(beam, shell, p1.integration_knots, 4)
    c2, m2, _ = integrated(beam, shell, p1.integration_knots, 2)
    sq, _ = scales(beam, shell, len(c))
    rel = lambda a, b: float(np.linalg.norm(a-b)/np.linalg.norm(a))
    if not np.all(np.isfinite(c)) or linalg.eigvalsh(mass)[0] <= 0:
        raise AssertionError('invalid local dual rows or mass')
    # Beam P1 translation coordinate moment pairing; root is fixed.
    pairs = np.zeros((n_beam, n_beam+1))
    roots, weights = np.polynomial.legendre.leggauss(4)
    for cell in range(n_beam):
        for root, weight in zip(roots, weights):
            xi = (root+1)/2
            t = test_matrix(n_beam, cell, xi)
            pairs[:,cell] += weight/(2*n_beam)*t[:n_beam,0]*(1-xi)
            pairs[:,cell+1] += weight/(2*n_beam)*t[:n_beam,0]*xi
    expected = np.zeros_like(pairs)
    for i in range(1, n_beam+1):
        expected[i-1,i] = (1/n_beam if i<n_beam else 1/(2*n_beam))
    if not np.allclose(pairs, expected, rtol=0, atol=2e-15):
        raise AssertionError('native P1 biorthogonality failed')
    whiten = linalg.solve_triangular(linalg.cholesky(mass, lower=True),
                                      c*sq[None,:],lower=True)
    def rank(a):
        sv = linalg.svdvals(a)
        return int(np.count_nonzero(sv > 1e-10*sv[0]))
    native_rank = rank(whiten[:,shell.free_count:])
    full_rank = rank(whiten)
    combined_rows = np.vstack((c*sq[None,:], k4*sq[None,:]))
    combined_values = linalg.svdvals(combined_rows)
    combined = rank(combined_rows)
    k4_rank = rank(k4*sq[None,:])
    incidence = int(np.count_nonzero(np.abs(c) > 1e-15, axis=1).max())
    coupling = replace(p1, matrix=sparse.csr_matrix(c), mass=mass,
                       full_constraint=full,
                       max_row_incidence=incidence,
                       quadrature_error={'C':rel(c*sq[None,:],c2*sq[None,:]),
                                         'M':rel(mass,m2),
                                         'G_D':p1.quadrature_error['G_D']})
    result = {'topology':topology, 'n_beam':n_beam, 'n_shell':n_shell,
              'rows':len(c), 'native_rank':native_rank, 'full_rank':full_rank,
              'shell_only_restrictions':full_rank-native_rank,
              'k4_rank':k4_rank, 'combined_k4_rank':combined,
              'k4_span_additional_rows':combined-len(c),
              'combined_rank_cutoff':float(1e-10*combined_values[0]),
              'combined_smallest_singular':float(combined_values[-1]),
              'combined_last_included_singular':float(combined_values[combined-1]),
              'quadrature_error':coupling.quadrature_error,
              'maximum_row_incidence':incidence,
              'native_pairing_max_error':float(np.max(np.abs(pairs-expected))),
              'loads':{}, 'status':'EXPLORATORY_FALSIFICATION_NOT_F03_ACCEPTANCE'}
    if full_rank != len(c) or native_rank != len(c):
        result['status'] = 'RANK_FAILURE_NOT_F03_ACCEPTANCE'
        return result
    for kind in ('smooth','sine'):
        load, full_load = fixed_load(beam, shell, coupling, kind, 8)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', linalg.LinAlgWarning)
                measure, solution = solve_exact(beam, shell, coupling, load)
            stations = physical_stations(beam, shell, solution)
            balance = physical_support_balance(
                beam, shell, replace(coupling, full_load_smooth=full_load),
                solution, 'smooth')
            result['loads'][kind] = {
                'max_sampled_gap_over_u':float(max(np.linalg.norm(s['gap_m'])
                                                    for s in stations)/U),
                'shell_energy_j':balance['active_physical_shell_energy_j'],
                'normalized_force_balance':balance['normalized_force_balance'],
                'normalized_moment_balance':balance['normalized_moment_balance'],
                'scaled_solution_parity':measure['scaled_solution_parity'],
                'scaled_free_equilibrium':measure['scaled_free_equilibrium'],
                'normalized_weak_residual':measure['normalized_weak_residual'],
                'scaled_kkt_singular_ratio':measure['scaled_kkt_singular_ratio']}
        except (linalg.LinAlgError, linalg.LinAlgWarning,
                CandidateSingular, PrimalNotAdmissible) as exc:
            result['loads'][kind] = {'failure_type':type(exc).__name__,
                                     'failure':str(exc)}
            result['status'] = 'NUMERICAL_RESPONSE_FAILURE_NOT_F03_ACCEPTANCE'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    source = {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
              for name in (Path(__file__).name, 'probe_b3_ge_e4_pl_g0_f03_kkt.py',
                           '_b3_e4_f03_native_fixture.py', '_b3_e4_f03_shell_trace.py',
                           'probe_b3_ge_e4_pl_g0_f03_k1_rank.py',
                           'probe_b3_ge_e4_pl_g0_f03_k2p.py')}
    (args.root/'manifest.json').write_text(json.dumps(
        {'purpose':'UNREGISTERED_LOCAL_DUAL_FALSIFICATION', 'cases':CASES,
         'source_sha256':source, 'anyfesolver_import':str(Path(anyfesolver.__file__).resolve()),
         'python':sys.version, 'numpy':np.__version__, 'scipy':scipy.__version__,
         'numba':numba.__version__,
         'numba_cache_directory':os.environ['NUMBA_CACHE_DIR']},
        indent=2, sort_keys=True)+'\n', encoding='utf-8')
    failed = False
    with (args.root/'results.jsonl').open('x',encoding='utf-8') as output:
        for case in CASES:
            try:
                result = analyze(*case)
            except Exception as exc:
                result = {'case':case, 'failure_type':type(exc).__name__,
                          'failure':str(exc), 'status':'EXPLORATORY_FAILURE'}
                output.write(json.dumps(result,sort_keys=True)+'\n')
                output.flush()
                raise
            output.write(json.dumps(result,sort_keys=True)+'\n')
            output.flush()
            failed |= (result['status'] !=
                       'EXPLORATORY_FALSIFICATION_NOT_F03_ACCEPTANCE')
            print(json.dumps(result,sort_keys=True), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
