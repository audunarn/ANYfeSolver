"""Bounded actual native mixed-system oracle; never production condensation.

One fixed supported16x4/32-halfcell/c2 fixture. Preserve all physical/cell
rotations; eliminate only checked negative stationary resultants locally.
Dense QR/eigenanalysis is an independent small-model oracle (physical<=1000),
not a scalable solver route, an SPD claim for native K, or F03 acceptance.
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
from scipy import linalg, sparse
from threadpoolctl import threadpool_info

from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_ge_experimental import _linear_equilibrium, _scaled_constraint_gap
from scripts.b3_e4_bl5_coherent_aggregation_dev import model, LOADS, physical_transfer


def relative(a, b, floor=1e-14):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), floor))


def norm2(a):
    return float(linalg.norm(a, 2))


def lower_completion_difference(a):
    """Measure the triangle completion implicit in a symmetric numerical oracle.

    This is an uncertainty quantity; the actual tangent is never overwritten,
    averaged or repaired. All symmetric routines below explicitly use lower.
    """
    return norm2(a-(np.tril(a)+np.tril(a, -1).T))


def evaluate(output, *, basis="p0"):
    start = time.monotonic()
    definition, policy = model(16, 4, 32, 2, basis=basis)
    owner = stiffened_shell.create_analysis(definition=definition)
    report = dict(classification='bounded mechanical development only; no F03/locking/accuracy acceptance',
                  policy=policy, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        k, c, free = owner._K, owner._C, owner._free
        base = owner._shell_count+6*len(owner._beam_ids)
        members = len(owner._beam_parts)
        assert k.shape == (base+24*members, base+24*members)
        blocks = [base+24*i+np.arange(6, 24) for i in range(members)]
        pp = np.concatenate(blocks)
        x = np.setdiff1d(free, pp)
        if len(x) > 1000 or len(free)+c.shape[0] > 1500:
            raise ValueError('bounded dense mechanical oracle size exceeded')
        assert np.isin(pp, free).all() and c[:, pp].nnz == 0
        independent = k[pp, :][:, pp]-sparse.block_diag([k[p, :][:, p] for p in blocks])
        assert not np.max(np.abs(independent.data), initial=0.)
        full_skew = float(sparse.linalg.norm(k-k.T)/sparse.linalg.norm(k))
        if full_skew > 1e-9:
            raise ArithmeticError('actual reference tangent is not conservative/symmetric')
        s = k[x, :][:, x].tocsr()
        # Sparse lift maps supported physical x back to every native free slot.
        free_map = {int(slot): i for i, slot in enumerate(free)}
        rows = [free_map[int(slot)] for slot in x]
        columns = list(range(len(x))); data = [1.]*len(x)
        maps, local_info = [], []
        for p in blocks:
            d = k[p, :][:, p].toarray()
            if relative(d, d.T) > 1e-9:
                raise ArithmeticError('stationary resultant block is nonsymmetric')
            factor = linalg.cho_factor(-d, lower=True)
            chol = np.tril(factor[0])
            factor_error = d+chol@chol.T
            reconstruction = norm2(factor_error)
            local_values = linalg.eigvalsh(d, lower=True)
            local_budget = reconstruction+64*np.finfo(float).eps*len(d)*norm2(d)
            if -local_values[-1] <= local_budget:
                raise ArithmeticError('local negative resultant block unresolved')
            coupling = k[p, :][:, x]
            neighbours = np.unique(coupling.nonzero()[1])
            if len(neighbours) > 24:
                raise ValueError('local resultant coupling exceeds native element layout')
            px = coupling[:, neighbours].toarray()
            lift = linalg.cho_solve(factor, px)  # -D^-1 Kpx
            residual = d@lift+px
            backward = float(np.linalg.norm(residual)/max(
                np.linalg.norm(d)*np.linalg.norm(lift)+np.linalg.norm(px), 1e-30))
            if backward > 1e-10:
                raise ArithmeticError('local stationary lift inaccurate')
            update = k[x[neighbours], :][:, p].toarray()@lift
            rr = np.repeat(neighbours, len(neighbours)); cc = np.tile(neighbours, len(neighbours))
            s += sparse.coo_matrix((update.ravel(), (rr, cc)), shape=s.shape).tocsr()
            rows.extend(np.repeat([free_map[int(slot)] for slot in p], len(neighbours)))
            columns.extend(np.tile(neighbours, len(p))); data.extend(lift.ravel())
            maps.append((p, neighbours, lift, factor, residual, factor_error))
            local_info.append(dict(negative_eigenvalues=local_values.tolist(),
                cholesky_reconstruction_error=reconstruction, negative_block_budget=local_budget,
                stationary_backward_residual=backward))
        lift_all = sparse.coo_matrix((data, (rows, columns)), shape=(len(free), len(x))).tocsr()
        kfree = k[free, :][:, free]
        cx = c[:, x].toarray()
        work_pullback = lift_all.T@kfree@lift_all
        work_parity = float(sparse.linalg.norm(work_pullback-s)/max(sparse.linalg.norm(s), 1e-30))
        constraint_parity = float(np.linalg.norm(c[:, free]@lift_all-cx)/max(np.linalg.norm(cx), 1e-30))
        if max(work_parity, constraint_parity) > 1e-10:
            raise ArithmeticError('stationary work/constraint pullback parity failed')
        # Invertible recorded congruence, including EVERY nodal and cell rotation.
        scale = owner._coordinate_scale[x].copy()
        diagonal = np.abs(s.diagonal()*scale**2)
        scale /= np.sqrt(np.where(diagonal > 0., diagonal, np.max(diagonal)))
        sd = sparse.diags(scale)
        scaled = (sd@s@sd).toarray()
        symmetry = relative(scaled, scaled.T)
        if symmetry > 1e-9:
            raise ArithmeticError('physical stationary tangent skew exceeds budget')
        cscaled = cx*scale
        row_norm = np.linalg.norm(cscaled, axis=1)
        if np.any(row_norm == 0.):
            raise ArithmeticError('supported physical constraint row vanishes')
        cscaled /= row_norm[:, None]
        singular = linalg.svdvals(cscaled)
        eps = np.finfo(float).eps
        rank_budget = 64*eps*max(cscaled.shape)*singular[0]
        if singular[-1] <= rank_budget:
            raise ArithmeticError('constraint rank unresolved; do not discard multiplier rows')
        qbasis, _ = linalg.qr(cscaled.T, mode='full')
        z = qbasis[:, c.shape[0]:]
        restricted = z.T@scaled@z
        values, vectors = linalg.eigh(restricted)
        snorm = float(np.max(np.abs(linalg.eigvalsh(scaled))))
        orthogonality = norm2(z.T@z-np.eye(z.shape[1]))
        leakage = norm2(cscaled@z)
        eta = leakage/singular[-1]
        eig_residual = norm2(restricted@vectors-vectors*values)
        eliminated_error = norm2((sd@(work_pullback-s)@sd).toarray())
        # Error to the exact stationary lift, using its local negative blocks.
        for _, neighbours, lift, factor, residual, factor_error in maps:
            r = residual*scale[neighbours]
            eliminated_error += norm2(r.T@linalg.cho_solve(factor, r))
            scaled_lift = lift*scale[neighbours]
            eliminated_error += norm2(scaled_lift.T@factor_error@scaled_lift)
        terms = dict(rounding=64*eps*len(x)*snorm,
                     elimination_and_reconstruction=eliminated_error,
                     kernel_leakage=(2*eta+eta**2)*snorm,
                     kernel_orthogonality=(2*orthogonality+orthogonality**2)*snorm,
                     physical_triangle_completion=lower_completion_difference(scaled),
                     eigen_residual=eig_residual)
        uncertainty = sum(terms.values())
        kernel_positive = bool(values[0] > uncertainty)
        saddle = np.block([[scaled, cscaled.T],
                           [cscaled, np.zeros((c.shape[0], c.shape[0]))]])
        saddle_values = linalg.eigvalsh(saddle)
        saddle_completion = lower_completion_difference(saddle)
        saddle_budget = 64*eps*len(saddle)*float(np.max(np.abs(saddle_values)))+uncertainty+saddle_completion
        inertia = dict(positive=int(np.sum(saddle_values > saddle_budget)),
                       negative=int(np.sum(saddle_values < -saddle_budget)),
                       unresolved=int(np.sum(np.abs(saddle_values) <= saddle_budget)))
        expected = dict(positive=len(x), negative=c.shape[0], unresolved=0)
        report.update(physical_order=len(x), native_supported_order=len(free), constraint_rows=c.shape[0],
            full_relative_skew=full_skew, physical_relative_skew=symmetry, local_blocks=local_info,
            work_pullback_relative=work_parity, constraint_pullback_relative=constraint_parity,
            constraint_singular_values=singular.tolist(), constraint_rank_budget=float(rank_budget),
            kernel_orthogonality=orthogonality, kernel_leakage=leakage, eta=eta,
            kernel_eigenvalues=values.tolist(), spectral_uncertainty_terms=terms,
            spectral_uncertainty=uncertainty, kernel_positive_resolved=kernel_positive,
            reduced_saddle_eigenvalues=saddle_values.tolist(), reduced_saddle_inertia=inertia,
            expected_reduced_saddle_inertia=expected, saddle_budget=saddle_budget,
            saddle_triangle_completion=saddle_completion,
            expected_native_inertia=dict(positive=len(x), negative=len(pp)+c.shape[0], unresolved=0),
            implied_native_inertia=(dict(positive=len(x), negative=len(pp)+c.shape[0], unresolved=0)
                                   if kernel_positive and inertia==expected else None))
        # Retain all modes; an unresolved smallest mode is never removed.
        np.savez_compressed(output.with_suffix('.npz'), physical_slots=x, resultant_slots=pp,
            physical_scale=scale, constraint_row_scale=row_norm, kernel_basis=z,
            kernel_vectors=vectors, kernel_values=values, saddle_values=saddle_values)
        loads = {}
        for name, wrench in LOADS.items():
            result = owner.solve_linear((stiffened_shell.NodalAction('beam', 133, wrench),))
            if result.status != 'completed':
                raise ArithmeticError('production native solve did not complete')
            force = owner._accepted[1]
            assert np.max(np.abs(force[pp]), initial=0.) == 0.
            rhs = np.r_[force[x]*scale, np.zeros(c.shape[0])]
            solved = linalg.solve(saddle, rhs, assume_a='sym', lower=True)
            physical = solved[:len(x)]*scale
            multipliers = solved[len(x):]/row_norm
            native = np.zeros(k.shape[0]); native[free] = lift_all@physical
            original = np.r_[result.shell_displacement, result.beam_coordinates]
            residual = k@native+c.T@multipliers-force
            support = np.zeros_like(residual); support[owner._fixed] = residual[owner._fixed]
            groups = dict(shell=x[x<owner._shell_count],
                          beam_nodes=x[(x>=owner._shell_count)&(x<base)],
                          retained_rotations=x[x>=base], resultants=pp)
            parity = {key:relative(native[slots]/owner._coordinate_scale[slots],
                                  original[slots]/owner._coordinate_scale[slots])
                      for key, slots in groups.items()}
            parity['multipliers'] = relative(multipliers*owner._constraint_scale,
                                             result.multipliers*owner._constraint_scale, 1e-12)
            parity['support'] = relative(support*owner._coordinate_scale,
                                        result.reactions*owner._coordinate_scale, 1e-12)
            equilibrium = _linear_equilibrium(residual[free], force[free], owner._coordinate_scale[free])
            gap = _scaled_constraint_gap(c@native, owner._constraint_scale)
            energy = relative(np.array(native@(k@native)), np.array(native@force), 1e-12)
            loads[name] = dict(parity=parity, equilibrium=equilibrium, weak_constraint=gap,
                energy_work_relative=energy, applied_work_j=float(native@force),
                transfer=physical_transfer(owner, result))
        report.update(loads=loads, factorization_count=owner.preflight()['reference_linear_factorizations'],
            elapsed_seconds=time.monotonic()-start)
        predicates = dict(kernel_positive=kernel_positive, saddle_inertia=inertia==expected,
                          solution_parity=all(max(v['parity'].values()) <= 1e-10 for v in loads.values()),
                          equilibrium=all(v['equilibrium'] <= 1e-8 for v in loads.values()),
                          weak_constraints=all(v['weak_constraint'] <= 1e-9 for v in loads.values()),
                          energy_work=all(v['energy_work_relative'] <= 1e-8 for v in loads.values()),
                          reference_factorization_reused=report['factorization_count']==1)
        report.update(predicates=predicates, status='COMPLETED_DEVELOPMENT_ONLY' if
                      all(predicates.values()) else 'UNRESOLVED_DEVELOPMENT_CHECK')
        return report
    finally:
        owner.close()
        if 'status' not in report:
            report['status'] = 'ABORTED_DEVELOPMENT_CHECK'
            output.with_name('partial-result.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--basis', choices=('p0', 'discontinuous_p1'), default='p0')
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        raise FileExistsError(args.output)
    pools = threadpool_info()
    assert pools and all(p['num_threads']==1 for p in pools) and numba.config.DISABLE_JIT
    result = evaluate(args.output, basis=args.basis)
    result.update(threads=pools, jit_disabled=True, imported_anyfesolver=anyfesolver.__file__)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key:result[key] for key in ('status','physical_order','kernel_positive_resolved','elapsed_seconds')}))
    raise SystemExit(0 if all(result['predicates'].values()) else 1)
