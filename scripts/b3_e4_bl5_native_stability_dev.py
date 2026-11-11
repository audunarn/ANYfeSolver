"""One installed native BL5 case against frozen independent reference criteria.

Development only. No formal F03 authority, mechanical KKT acceptance, hidden
small-eigenvalue removal, or multiplier selection. Run family cases serially.
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
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info

from scripts.b3_e4_bl5_coherent_aggregation_dev import model
from scripts.b3_e4_bl5_reference_actions_dev import native_slots, physical_loads
from scripts.b3_e4_bl5_reference_accuracy_dev import beam_bracket, AuxiliaryInverseAccuracyError
from scripts.b3_e4_bl5_modal_reference_dev import p0_shell_bracket
from scripts.b3_e4_bl5_riesz_controls_dev import beam_norm, shell_scalar


def inverse_gram(h, action):
    z = splu(h).solve(action)
    residual = np.linalg.norm(h@z-action)/np.linalg.norm(action)
    cross = action.T@z
    gram = cross+cross.T-z.T@(h@z)
    return gram, float(residual)


def evaluate(nx, ny, cells, frozen, reference_start=8, preparation_path=None):
    start = time.monotonic()
    scalar_table = json.loads(frozen.read_text())
    entry = [r for r in scalar_table['cases'] if
             (r['nx'], r['ny'], r['beam_halfcells']) == (nx, ny, cells)]
    if len(entry) != 1:
        raise ValueError('case outside frozen c2 finite family')
    scalar_case = entry[0]
    definition, policy = model(nx, ny, cells, 2)
    owner = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    try:
        local = owner._line_samples[0][0]
        frame = local.port.frame
        director = frame@local.reference_director
        arm_b = frame@local.reference_arm
        arm_s = local.thickness*director/2
        np.testing.assert_allclose(director, (0., 0., 1.), atol=1e-13, rtol=0.)
        np.testing.assert_allclose(arm_b, (0., 0., -.05), atol=1e-13, rtol=0.)
        np.testing.assert_allclose(arm_s, (0., 0., .0175), atol=1e-13, rtol=0.)
        for sample in owner._line_samples:
            op = sample[0]
            if (np.linalg.norm(op.port.lift) > 1e-13 or
                    not np.allclose(op.port.frame, frame, atol=1e-13, rtol=0.) or
                    not np.allclose(frame@op.reference_arm, arm_b, atol=1e-13, rtol=0.)):
                raise ValueError('unadmitted variable frame/arm/lift')
        span, supports = definition.lines[0].multiplier_span, cells//definition.lines[0].multiplier_span
        assert span == scalar_case['span'] and supports == scalar_case['supports']
        bk, hb, ba, _ = beam_norm(1., .4, cells, 1, arm_b, .1)
        sx, _, hs, sa = shell_scalar(1., .2, .4, nx, ny)
        slots, absent, all_fields = native_slots(owner, sa, ba)
        physical_node_x = np.array([owner._beam_nodes[n][0][0] for n in owner._beam_ids])
        np.testing.assert_allclose(physical_node_x, bk, atol=1e-13, rtol=0.)
        q = np.array((1., 1., 1., .4, .4))
        densities = np.diag(np.repeat(q, supports))
        loads = [physical_loads(col.reshape(5, supports).T, frame, director,
                local.director_basis, arm_b, arm_s, bk, sx, ny, cells, span)
                for col in densities.T]
        complete = np.column_stack([np.r_[s, b] for b, s, _ in loads])
        native_complete = -np.asarray(owner._C[:, all_fields].T@densities)
        parity = float(np.max(np.abs(complete-native_complete)))
        np.testing.assert_allclose(complete, native_complete, atol=1e-12, rtol=1e-11)
        absent_entries = owner._C[:, absent].data
        omitted_max = float(np.max(np.abs(absent_entries))) if len(absent_entries) else 0.
        if omitted_max > 1e-13:
            raise ValueError('norm quotient omits nonzero native action')
        h5 = sparse.block_diag((hs, hs, hs, .4**2*hs, .4**2*hs), format='csc')
        h = sparse.block_diag((h5, hb), format='csc')
        action = -np.asarray(owner._C[:, slots].T@densities)
        native_gram, native_residual = inverse_gram(h, action)
        scalar_lo, scalar_hi, shell_info = p0_shell_bracket(np.linspace(0., 1., supports+1), 1., .2, .4, 4096)
        # Complete independently evaluated physical force/moment coefficient maps.
        fs = np.stack([row[2][2] for row in loads], axis=2)
        ms = np.stack([row[2][3] for row in loads], axis=2)
        shell_lo = np.zeros_like(native_gram); shell_hi = np.zeros_like(native_gram)
        for channel in range(5):
            coefficient = fs[:, channel, :] if channel < 3 else ms[:, channel-3, :]/.4
            shell_lo += coefficient.T@scalar_lo@coefficient
            shell_hi += coefficient.T@scalar_hi@coefficient
        force = np.repeat(np.stack([row[2][0] for row in loads], axis=2), span, axis=0)
        moments = np.repeat(np.stack([row[2][1] for row in loads], axis=2), span, axis=0)/cells
        refinements = []
        schedule = tuple(n for n in (1, 2, 4, 8, 16, 32) if n >= reference_start)
        error = None
        for subdivision in schedule:
            try:
                beam_lo, beam_hi, info = beam_bracket(force, moments, length=1., ell=.4,
                    arm=arm_b, width=.1, subdivisions=subdivision)
            except AuxiliaryInverseAccuracyError as exc:
                refinements.append(dict(subdivisions=subdivision,
                    status='UNRESOLVED_INVERSE_ACCURACY', inverse_residual_history=exc.history))
                if preparation_path is not None:
                    preparation_path.write_text(json.dumps(refinements, indent=2))
                if reference_start == 8:
                    raise  # Original protocol stops at an unresolved inverse.
                continue  # Only this identified numerical eligibility failure.
            lower, upper = shell_lo+beam_lo, shell_hi+beam_hi
            error = float(linalg.eigvalsh(upper-lower, lower)[-1])
            refinements.append(dict(beam=info, total_relative_loewner_error=error,
                status='ELIGIBLE' if error <= 1e-3 else 'UNRESOLVED_REFERENCE_ERROR'))
            if preparation_path is not None:
                preparation_path.write_text(json.dumps(refinements, indent=2))
            if error <= 1e-3:
                break
        else:
            raise ArithmeticError('bounded reference preparation exhausted before native beta')
        skew = np.linalg.norm(native_gram-native_gram.T)/np.linalg.norm(native_gram)
        # Symmetric inverse-functional Gram roundoff only; mechanical K untouched.
        values, vectors = linalg.eigh((native_gram+native_gram.T)/2, upper)
        beta = float(np.sqrt(values[0])) if values[0] > 0 else None
        eigen_residual = float(np.linalg.norm(native_gram@vectors-(upper@vectors)*values[None, :])/
            np.linalg.norm(native_gram@vectors))
        conditions = dict(native=float(np.linalg.cond(native_gram)), reference=float(np.linalg.cond(upper)))
        safe_floor = scalar_case['prospective_product_floor']/np.sqrt(1.001)-1e-6
        predicates = dict(reference_error=error <= 1e-3, inverse_residual=native_residual <= 1e-10,
            eigen_residual=eigen_residual <= 1e-10, gram_symmetry=skew <= 1e-10,
            conditioning=max(conditions.values()) <= 1e4, all_eigenvalues_positive=bool(np.all(values > 0)),
            case_floor=beta is not None and beta >= safe_floor,
            absolute_floor=beta is not None and beta >= .43,
            variational_ceiling=bool(values[-1] <= (1+1e-6)**2))
        predicates = {name:bool(value) for name,value in predicates.items()}
        return dict(status='COMPLETED_DEVELOPMENT_ONLY' if all(predicates.values()) else 'UNRESOLVED_DEVELOPMENT_CHECK',
            predicates=predicates, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            nx=nx, ny=ny, beam_halfcells=cells, policy=policy, rows=owner._K.shape[0],
            constraint_rows=owner._C.shape[0], norm_rows=h.shape[0], native_physical_pairing_max_abs=parity,
            omitted_C_max_abs=omitted_max, omitted_slots=len(absent), native_inverse_residual=native_residual,
            gram_relative_skew=float(skew), eigen_relative_residual=eigen_residual,
            conditions=conditions, reference_refinements=refinements, shell_reference=shell_info,
            reference_subdivision_schedule=schedule,
            all_generalized_eigenvalues=values.tolist(), beta=beta, safe_case_floor=float(safe_floor),
            frozen_scalar_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
            imported_anyfesolver=anyfesolver.__file__, elapsed_seconds=time.monotonic()-start,
            limitation='finite reference inf-sup development only; actual indefinite KKT, locking, physical accuracy and F03/M2 acceptance remain open')
    finally:
        owner.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scalar', type=Path, required=True)
    parser.add_argument('--nx', type=int, required=True)
    parser.add_argument('--ny', type=int, required=True)
    parser.add_argument('--halfcells', type=int, required=True)
    parser.add_argument('--reference-subdivision-start', type=int, choices=(1, 2, 4, 8), default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pools = threadpool_info()
    assert pools and all(p['num_threads'] == 1 for p in pools) and numba.config.DISABLE_JIT
    preparation = args.output.parent/'reference-preparation.json'
    if preparation.exists():
        raise FileExistsError(preparation)
    result = evaluate(args.nx, args.ny, args.halfcells, args.scalar,
                      args.reference_subdivision_start, preparation)
    result.update(threads=pools, jit_disabled=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
    print(json.dumps({key:result[key] for key in ('status', 'beta', 'safe_case_floor', 'elapsed_seconds')}))
    raise SystemExit(0 if all(result['predicates'].values()) else 1)
