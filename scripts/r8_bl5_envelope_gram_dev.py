"""Additive saved-envelope preparation and fixed-target Gram assessment.

Frozen common sources remain unchanged. No product solve or mechanical change.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info

from scripts import r8_bl5_projected_native_beta_dev as native
from scripts import r8_bl5_sparse_kernel_saved_dev as bound
from scripts.r8_bl5_sparse_envelope_kernel_dev import (
    ROOT, RAW, INPUT_BUDGET, GRAPH_BUDGET, LOCAL_COLUMN_LIMIT, PROOF_BUDGET as OLD_PROOF_BUDGET,
    require, sha, partition, stationary, support_pivots, stationary_incidence_error, store_sparse,
)
from scripts._b3_e4_f03_sparse_certificate import check_arithmetic
from scripts.r8_bl5_sparse_kernel_interval_dev import IntervalBudget, down, up, sum_up
from scripts.r8_bl5_sparse_gram_margin_dev import certify_gram_margin
from scripts.r8_bl5_sparse_factor_residual_dev import _own, _ResourceRefusal

IDENTITY = 'BL5_SAVED_ENVELOPE_PREPARED_FIXED_SHIFT_GRAM_DEV_V1'
PROOF_BUDGET = replace(OLD_PROOF_BUDGET, max_workspace_bytes=2*1024**3)


def prepare(case,owner,output,*,wheel,site):
    stationary_bound,precision='incidence',50
    check_arithmetic();started=time.monotonic();output=Path(output);owner=Path(owner)
    require(stationary_bound in ('sum','incidence') and type(precision) is int and
        precision in (50,100), 'explicit stationary bound and precision required')
    require(not output.exists() and not output.with_suffix('.npz').exists() and output.suffix=='.json', 'fresh paired outputs required')
    require(output.parent.is_dir(), 'existing owned output directory required')
    pools=threadpool_info()
    require(pools and all(pool['num_threads']==1 for pool in pools), 'effective numerical pools1 required')
    require(not any(name.split('.')[0]=='anyfesolver' for name in sys.modules), 'saved assessment must be product-free')
    arrays={};report=dict(identity=IDENTITY,status='STARTED',fixture_case=case.fixture_case,
        stationary_bound=stationary_bound,decimal_precision=precision,
        fixture=case.fixture,counts=case.counts,formal_f03_authority_consumed=False,
        source_sha256=sha(__file__),dependencies={name:sha(ROOT/'scripts'/name) for name in (
            'r8_bl5_sparse_kernel_saved_dev.py','r8_bl5_sparse_kernel_interval_dev.py',
            '_b3_e4_f03_sparse_certificate.py','r8_bl5_projected_envelope_norm_dev.py',
            'r8_bl5_projected_envelope_owner_dev.py',
            'r8_bl5_sparse_envelope_kernel_dev.py','r8_bl5_projected_native_beta_dev.py',
            'r8_bl5_sparse_gram_margin_dev.py','r8_bl5_sparse_factor_residual_dev.py')},effective_pools=pools,
        budgets=dict(input=asdict(INPUT_BUDGET),graph=asdict(GRAPH_BUDGET),proof=asdict(PROOF_BUDGET),local_columns=LOCAL_COLUMN_LIMIT),
        limitation='Exact saved work-form kernel only; no native inf-sup, fullK SPD, uniform/physical qualification or formalF03.')
    def save(stage):
        report.update(stage=stage,elapsed_seconds=time.monotonic()-started)
        np.savez_compressed(output.with_suffix('.npz'),**arrays)
        report['arrays_sha256']=sha(output.with_suffix('.npz'))
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if stage != 'FAILED' and report['elapsed_seconds'] > 600:
            raise TimeoutError('complete sparse assessment600s ceiling exceeded')
    try:
        summary,report['input_binding']=native.frozen_owner(case,owner)
        native.installed_graph(site,wheel,summary);save('BOUND_SAVED_INSTALLED_OWNER')
        metadata=json.loads((owner/'native/inputs.json').read_text())
        with np.load(owner/'native/operators.npz',allow_pickle=False) as data:
            k=bound.saved_csr(data,'K',sparse_budget=INPUT_BUDGET)
            c=bound.saved_csr(data,'C',square=False,sparse_budget=INPUT_BUDGET)
            free,fixed,coordinate=[data[name].copy() for name in ('free','fixed','coordinate_scale')]
        require(k.shape==(case.counts['full_q'],)*2 and c.shape==(case.counts['multipliers'],case.counts['full_q']), 'actual sparse shapes differ')
        require(coordinate.shape==(case.counts['full_q'],) and np.isfinite(coordinate).all() and np.all(coordinate>0), 'finite positive complete coordinate scales required')
        x,pp,blocks=partition(case,metadata,free,fixed)
        arrays.update(physical_slots=x,resultant_slots=pp)
        s=stationary(k,c,x,blocks,report);store_sparse(arrays,'S',s);save('COMPLETE_LOCAL_STATIONARY_PULLBACK')
        scale=coordinate[x].copy();diagonal=np.abs(s.diagonal()*scale**2)
        require(np.isfinite(diagonal).all() and diagonal.max()>0, 'stationary dimensional scale unresolved')
        scale /= np.sqrt(np.where(diagonal>0,diagonal,diagonal.max()))
        require(np.isfinite(scale).all() and np.all(scale>0), 'positive finite physical scales required')
        row_norm=np.sqrt(np.asarray(c[:,x].multiply(scale).power(2).sum(axis=1)).ravel())
        require(np.isfinite(row_norm).all() and np.all(row_norm>0), 'all actual constraint rows must be nonzero')
        shat,_,_=bound.scaled_data(s,scale,scale)
        chat,clo,chi=bound.scaled_data(c[:,x],row_norm,scale,divide_left=True)
        groups,pivots=support_pivots(case,metadata,x)
        z,ez,graph=bound.graph_kernel(chat,clo,chi,groups,pivots,
            sparse_budget=GRAPH_BUDGET,local_column_limit=LOCAL_COLUMN_LIMIT)
        if stationary_bound=='incidence':
            es,local,incidence=stationary_incidence_error(k,shat,scale,x,blocks)
            arrays['stationary_incidence_weights']=incidence.pop('incidence_weights')
            arrays['stationary_incidence_counts']=incidence.pop('incidence_counts')
            report.update(stationary_incidence=incidence,stationary_operator_error_induced2_upper=es)
        else:
            es,local=bound.stationary_error(k,shat,scale,x,blocks)
            report['stationary_operator_error_fro_upper']=es
        report['stationary_local_proofs']=local
        save('ENCLOSED_STATIONARY_WORK')
        preliminary=z.T@(shat@z);diag=preliminary.diagonal()
        require(np.isfinite(diag).all() and np.all(diag>0), 'positive kernel graph diagonal unresolved')
        t=1./np.sqrt(diag)
        require(np.isfinite(t).all() and np.all(t>0), 'finite positive graph congruence required')
        znew,zl,zh=bound.scaled_data(z,np.ones(z.shape[0]),t)
        scaling_error=bound.fro_upper(bound.box_abs(down(zl.data-znew.data),up(zh.data-znew.data)))
        ez=sum_up([float(up(ez*float(t.max()))),scaling_error]);z=znew
        zn,sn=bound.norm_upper(z),bound.norm_upper(shat)
        first,e1=bound.sparse_product(shat,z);ahat,e2=bound.sparse_product(z.T.tocsr(),first)
        lower=sparse.tril(ahat,format='csr')
        representative=(lower+sparse.tril(ahat,-1,format='csr').T).tocsr()
        triangle=bound.difference_upper(ahat,representative);true_zn=float(up(zn+ez))
        graph_error=float(up(up(up(2*zn*ez)+up(ez*ez))*sn))
        operator_error=float(up(up(true_zn*true_zn)*es))
        epsilon=sum_up([graph_error,operator_error,float(up(zn*e1)),e2,triangle])
        shift=float(up(2*epsilon));metric=float(up(true_zn*true_zn))
        report.update(physical_order=len(x),kernel_order=z.shape[1],constraint_rows=c.shape[0],
            positive_target_norm='stored dimensionless physical u, q_phys=diag(physical_scale) u; not unscaled nodal Euclidean norm',
            graph_local_proofs=graph,kernel_error_upper=epsilon,
            graph_error_fro_upper=ez,
            error_terms=dict(graph=graph_error,stationary=operator_error,first_product=float(up(zn*e1)),second_product=e2,triangle=triangle),
            fixed_shift=shift,graph_metric_upper=metric,
            graph_column_congruence='fixed stored inverse-square-root positive preliminary diagonal; error and metric retained')
        arrays.update(physical_scale=scale,constraint_row_scale=row_norm,graph_column_scale=t)
        store_sparse(arrays,'Z',z);store_sparse(arrays,'A_representative',representative)
        report['status']='PREPARED_MECHANICAL_KERNEL'
        save('COMPLETE_MECHANICAL_ERROR_BOUND')
        return report
    except BaseException as error:
        report.update(status='FAILED_SPARSE_ENVELOPE_PREPARATION',failure=dict(type=type(error).__name__,message=str(error)))
        try:save('FAILED')
        except BaseException as recording_error:error.add_note('Recording also failed: '+repr(recording_error))
        raise


def certify_target(matrix, *, epsilon, metric, budget=PROOF_BUDGET, checkpoint=None):
    """One deterministic producer, complete residual proof, unchanged floor.

    The factor signs/order only admit a real producer candidate; they do not
    accept mechanics. No automatic shift/order/precision/factorization retry.
    """
    require(type(budget) is IntervalBudget, 'explicit IntervalBudget required')
    require(checkpoint is None or callable(checkpoint), 'callable evidence checkpoint required')
    require(type(epsilon) is float and np.isfinite(epsilon) and epsilon > 0.,
            'positive finite stored error required')
    require(type(metric) is float and np.isfinite(metric) and metric > 0.,
            'positive finite physical graph metric required')
    sigma = float(up(2*epsilon))
    tau = float(up(sigma+epsilon))
    report = dict(status='HOLD_MECHANICAL_KERNEL', fixed_shift=sigma,
                  producer_shift=tau, epsilon=epsilon, graph_metric_upper=metric,
                  physical_kernel_lower_bound=None, formal_f03_authority_consumed=False)
    arrays = {}
    try:
        a = _own(matrix, budget)
    except _ResourceRefusal as error:
        report.update(hold_kind='RESOURCE_REFUSAL', hold_reason=str(error))
        return report, arrays
    require(not np.any((a-a.T).data != 0.), 'exact symmetric representative required')
    bhat = a.copy()
    bhat.setdiag(bhat.diagonal()-tau)
    bhat.sort_indices()
    backend = sparse.linalg.splu(bhat.tocsc(), permc_spec='MMD_AT_PLUS_A',
        diag_pivot_thresh=0., options={'SymmetricMode':True, 'Equil':False})
    diagonal = backend.U.diagonal().copy()
    arrays.update(producer_diagonal=diagonal, producer_perm_r=backend.perm_r.copy(),
                  producer_perm_c=backend.perm_c.copy())
    report['producer'] = dict(L_nnz=backend.L.nnz, U_nnz=backend.U.nnz,
        sign_limitation='Signs only construct a real Gram candidate; never a mechanical certificate.')
    if checkpoint is not None:
        checkpoint('SAVED_PRODUCER', report, arrays)
    if not np.array_equal(backend.perm_r, backend.perm_c):
        report.update(hold_kind='PRODUCER_PERMUTATION', hold_reason='symmetric producer ordering required')
        return report, arrays
    if not np.isfinite(diagonal).all() or not np.all(diagonal > 0.):
        report.update(hold_kind='PRODUCER_DIAGONAL', hold_reason='finite positive producer diagonal unavailable')
        return report, arrays
    order = np.argsort(backend.perm_r).tolist()
    arrays['permutation'] = np.asarray(order, dtype=np.int64)
    f = sparse.csr_matrix(backend.L.multiply(np.sqrt(diagonal)))
    f.sum_duplicates(); f.sort_indices()
    store_sparse(arrays, 'candidate_F', f)
    del backend, bhat
    if checkpoint is not None:
        checkpoint('SAVED_CANDIDATE_FACTOR', report, arrays)
    proof = certify_gram_margin(a, f, order, target_shift=sigma, producer_shift=tau, budget=budget)
    for name, value in list(proof.items()):
        if isinstance(value, np.ndarray):
            arrays['proof_'+name] = proof.pop(name)
    report['gram_margin_proof'] = proof
    floor = float(down(float(down(sigma-epsilon))/metric))
    if proof['status']=='POSITIVE_FIXED_SHIFT_GRAM_MARGIN' and proof['positive'] and floor > 0.:
        report.update(status='POSITIVE_ENVELOPE_MECHANICAL_KERNEL_DEVELOPMENT_ONLY',
                      physical_kernel_lower_bound=floor)
    return report, arrays
