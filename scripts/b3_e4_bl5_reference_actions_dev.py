"""Physical BL5 work into independently enriched positive reference norms.

One bounded zero-lift, straight/flat P0 fixture; no native beta or F03 execution.
Reference loads integrate the declared continuous density at reference grid
boundaries. Reusing native station atoms would change the reference functional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import anyfesolver
import numba
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info

from scripts.b3_e4_bl5_coherent_aggregation_dev import model
from scripts.b3_e4_bl5_riesz_controls_dev import beam_norm, shell_scalar


ELL = .4
LENGTH, WIDTH, BEAM_WIDTH = 1., .2, .1
NS, NT, NB, C_LAMBDA = 8, 2, 16, 1


def spatial_action(density, frame, director, basis, arm_b, arm_s):
    """Exerted loads; minus constraint-work gradient on each domain."""
    fb = frame@density[:3]
    couple = np.cross(director, frame@basis.T@density[3:])
    return fb, np.cross(arm_b, fb)+couple, -fb, np.cross(arm_s, -fb)-couple


def integrated_p1(knots, values, support_knots):
    """Exact piecewise constant density work, split at both partitions."""
    out = np.zeros((len(knots), values.shape[1]))
    partitions = np.unique(np.r_[knots, support_knots])
    for a, b in zip(partitions[:-1], partitions[1:]):
        x = (a+b)/2
        e = min(np.searchsorted(knots, x)-1, len(knots)-2)
        j = min(np.searchsorted(support_knots, x)-1, len(values)-1)
        t = (x-knots[e])/(knots[e+1]-knots[e])
        out[e] += (b-a)*(1-t)*values[j]
        out[e+1] += (b-a)*t*values[j]
    return out


def physical_loads(densities, frame, director, basis, arm_b, arm_s,
                   beam_knots, shell_x, ny, cells, span):
    support_knots = np.linspace(0., LENGTH, len(densities)+1)
    wrenches = [spatial_action(d, frame, director, basis, arm_b, arm_s) for d in densities]
    fb, mb, fs, ms = (np.array([w[i] for w in wrenches]) for i in range(4))
    beam_u = integrated_p1(beam_knots, fb, support_knots)
    cell_m = np.array([LENGTH/cells*mb[j//span] for j in range(cells)])
    shell_f = integrated_p1(shell_x, fs, support_knots)
    shell_m = integrated_p1(shell_x, ms[:, :2], support_knots)
    # Shell root five fields have Dirichlet traces; other boundaries natural.
    shell = np.zeros((5, (ny+1)*len(shell_x)))
    shell[:3, :len(shell_x)] = shell_f.T
    shell[3:, :len(shell_x)] = shell_m.T
    return np.r_[beam_u.reshape(-1), cell_m.reshape(-1)], shell.reshape(-1), (fb, mb, fs, ms)


def native_slots(owner, shell_active, beam_active):
    # Shell bulk fields in five-block order, beam U interleaved then cell Theta.
    scalar_count = len(owner._shell_ids)
    shell_slots = np.r_[*[6*shell_active+a for a in range(5)]]
    beam_u = np.r_[*[owner._shell_count+6*i+np.arange(3) for i in range(len(owner._beam_ids))]]
    retained = owner._shell_count+6*len(owner._beam_ids)
    rotations = np.r_[*[retained+24*ei+3*cell+np.arange(3)
                         for ei, cell in owner._definition.lines[0].beam_halfcells]]
    physical = np.r_[shell_slots, np.r_[beam_u, rotations][beam_active]]
    all_fields = np.r_[*[6*np.arange(scalar_count)+a for a in range(5)], beam_u, rotations]
    absent = np.setdiff1d(np.arange(owner._K.shape[0]), all_fields)
    return physical, absent, all_fields


def reference_norm(refinement, cells, arm_b):
    bk, hb, ba, _ = beam_norm(LENGTH, ELL, cells, refinement, arm_b, BEAM_WIDTH)
    sx, _, hs, sa = shell_scalar(LENGTH, WIDTH, ELL, NS*refinement, NT*refinement)
    hs5 = sparse.block_diag((hs, hs, hs, ELL**2*hs, ELL**2*hs), format='csc')
    shell_active5 = np.r_[*[i*((NT*refinement+1)*len(sx))+sa for i in range(5)]]
    return bk, sx, hb, hs5, ba, shell_active5


def solve_columns(h, b):
    z = splu(h).solve(b)
    residuals = np.linalg.norm(h@z-b, axis=0)/np.maximum(np.linalg.norm(b, axis=0), 1e-30)
    energies = np.einsum('ij,ij->j', b, z)
    if np.max(residuals) > 1e-10 or np.any(energies <= 0):
        raise ArithmeticError('reference positive inverse action failed')
    return energies, residuals


def evaluate():
    pools = threadpool_info()
    if not pools or any(p['num_threads'] != 1 for p in pools) or not numba.config.DISABLE_JIT:
        raise RuntimeError('effective one-thread/JIT-off settings required')
    started = time.perf_counter()
    definition, policy = model(NS, NT, NB, C_LAMBDA)
    owner = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    line = definition.lines[0]
    if line.basis != 'p0' or len(definition.lines) != 1 or definition.points:
        raise ValueError('bounded P0 physical correspondence required')
    local = owner._line_samples[0][0]
    frame = local.port.frame
    director = frame@local.reference_director
    arm_b = frame@local.reference_arm
    arm_s = local.thickness*director/2
    for sample in owner._line_samples:
        op = sample[0]
        if (np.linalg.norm(op.port.lift) > 1e-13 or
                not np.allclose(op.port.frame, frame, atol=1e-13, rtol=0.) or
                not np.allclose(frame@op.reference_arm, arm_b, atol=1e-13, rtol=0.)):
            raise ValueError('zero lift/constant frame and arm contract required')
    np.testing.assert_allclose(director, (0., 0., 1.), atol=1e-13)
    np.testing.assert_allclose(arm_b[:2], 0., atol=1e-13)
    span, supports = line.multiplier_span, NB//line.multiplier_span
    qscale = np.array((1., 1., 1., ELL, ELL))
    rng = np.random.default_rng(31465)
    mu = [np.tile(np.eye(5)[j], (supports, 1)) for j in range(5)]
    mu.append(rng.normal(size=(supports, 5)))
    # Nonzero continuous force surviving exact cancellation of cell moments.
    force = np.array((1., 0., 0.))
    couple_map = np.column_stack([np.cross(director, frame@local.director_basis[j]) for j in range(2)])
    direct, _, _, _ = np.linalg.lstsq(couple_map, -np.cross(arm_b, force), rcond=None)
    cancellation = np.r_[frame.T@force, direct]/qscale
    mu.append(np.tile(cancellation, (supports, 1)))
    names = ('force_x', 'force_y', 'force_z', 'director_0', 'director_1', 'mixed', 'arm_cancellation')
    densities = [m*qscale for m in mu]
    lambdas = np.column_stack([d.T.reshape(-1) for d in densities])
    labels = ['N/m']*3+['N']*2
    # Native-grid physical loads independently integrate the density, not C rows.
    bk, hb, ba, _ = beam_norm(LENGTH, ELL, NB, 1, arm_b, BEAM_WIDTH)
    sx, _, hs, sa = shell_scalar(LENGTH, WIDTH, ELL, NS, NT)
    sa5 = np.r_[*[j*len(owner._shell_ids)+sa for j in range(5)]]
    slots, absent, all_fields = native_slots(owner, sa, ba)
    gauge_entries = owner._C[:, absent].data
    gauge_max = float(np.max(np.abs(gauge_entries))) if len(gauge_entries) else 0.
    assert gauge_max < 1e-13
    loads = [physical_loads(d, frame, director, local.director_basis, arm_b, arm_s,
                             bk, sx, NT, NB, span) for d in densities]
    physical_rhs = np.column_stack([np.r_[s[sa5], b[ba]] for b, s, _ in loads])
    native_rhs = -np.asarray(owner._C[:, slots].T@lambdas)
    work_error = float(np.max(np.abs(physical_rhs-native_rhs)))
    np.testing.assert_allclose(physical_rhs, native_rhs, atol=1e-12, rtol=1e-11)
    # Deterministic parity for every density basis, including supported rows.
    density_basis = np.diag(np.repeat(qscale, supports))
    full_basis_loads = [physical_loads(col.reshape(5, supports).T,
                          frame, director, local.director_basis, arm_b, arm_s,
                          bk, sx, NT, NB, span) for col in density_basis.T]
    complete_physical = np.column_stack([np.r_[s, b] for b, s, _ in full_basis_loads])
    complete_native = -np.asarray(owner._C[:, all_fields].T@density_basis)
    complete_error = float(np.max(np.abs(complete_physical-complete_native)))
    np.testing.assert_allclose(complete_physical, complete_native, atol=1e-12, rtol=1e-11)
    # Q work pairing and reciprocal multiplier-basis transformation.
    virtual = rng.normal(size=len(slots))
    bmap = owner._C[:, slots]
    for m, lam, force_rhs in zip(mu, lambdas.T, physical_rhs.T):
        lhs = float(lam@(bmap@virtual))
        rhs = float(m.T.reshape(-1)@(np.repeat(qscale, supports)*(bmap@virtual)))
        assert abs(lhs-rhs) < 1e-12*max(abs(lhs), 1.)
        assert abs(float(force_rhs@virtual)+lhs) < 1e-12*max(abs(lhs), 1.)
    diagonal = np.geomspace(.1, 10., len(lambdas))
    np.testing.assert_allclose((sparse.diags(diagonal)@bmap).T@(lambdas/diagonal[:, None]),
                               bmap.T@lambdas, atol=1e-12, rtol=1e-12)
    assert np.max(np.abs(loads[-1][2][1])) < 1e-13
    levels = []
    for refinement in (2, 4, 8):
        bk, sx, hb, hs5, ba, sa5 = reference_norm(refinement, NB, arm_b)
        reference_loads = [physical_loads(d, frame, director, local.director_basis,
                              arm_b, arm_s, bk, sx, NT*refinement, NB, span) for d in densities]
        bbeam = np.column_stack([b[ba] for b, _, _ in reference_loads])
        bshell = np.column_stack([s[sa5] for _, s, _ in reference_loads])
        eb, rb = solve_columns(hb, bbeam)
        es, rs = solve_columns(hs5, bshell)
        levels.append(dict(refinement=refinement, beam_rows=hb.shape[0], shell_rows=hs5.shape[0],
                           beam_dual=eb.tolist(), shell_dual=es.tolist(), total_dual=(eb+es).tolist(),
                           beam_inverse_residual=rb.tolist(), shell_inverse_residual=rs.tolist()))
    previous = np.array(levels[-2]['total_dual']); final = np.array(levels[-1]['total_dual'])
    for older, newer in zip(levels[:-1], levels[1:]):
        assert np.all(np.array(newer['total_dual']) >= np.array(older['total_dual'])*(1-1e-12))
    owner.close()
    return dict(status='COMPLETED_DEVELOPMENT_ONLY', imported_anyfesolver=anyfesolver.__file__, threads=pools,
                elapsed_seconds=time.perf_counter()-started, policy=policy,
                ell_m=ELL, beam_norm_width_m=BEAM_WIDTH, shell_width_m=WIDTH, action_names=names,
                density_units=labels, force_equivalent_mu=[m.tolist() for m in mu],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                native_physical_pairing_max_abs=work_error, omitted_kinematic_slots=len(absent),
                complete_basis_columns=len(density_basis), complete_supported_pairing_max_abs=complete_error,
                omitted_slot_C_max_abs=gauge_max, native_physical_field_slots=len(slots),
                reference_levels=levels, final_relative_increment=((final-previous)/final).tolist(),
                limitation='reference work/gauge controls only; absent kinematic slots are not removed from actual mechanics; no candidate beta, floor, mixed kernel, F03 or M2 acceptance')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = evaluate()
    with a.output.open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2)
    print(json.dumps(dict(status=result['status'], final_relative_increment=result['final_relative_increment'],
                         elapsed_seconds=result['elapsed_seconds'])))
