"""Complete sparse mechanical-kernel assessment of saved envelope operators.

Development only. Native resultant coordinates remain in the product owner;
this independent stationary pullback never changes its tangent or state. No
global dense inverse/nullspace/eigenanalysis, invented inertia or fullK SPD.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import linalg, sparse
from threadpoolctl import threadpool_info

from scripts import r8_bl5_sparse_kernel_saved_dev as bound
from scripts._b3_e4_f03_sparse_certificate import SparseBudget, validate_sparse, check_arithmetic
from scripts.r8_bl5_sparse_kernel_interval_dev import (
    IntervalBudget, down, up, sum_up, factor_pattern, complete_symbolic_pattern,
    certify_shifted_lower, certify_decimal_lower,
)
from scripts.r8_bl5_projected_envelope_norm_dev import case_from_id, all_cases
from scripts.r8_bl5_projected_envelope_owner_dev import registration_identity, WHEEL_SHA

IDENTITY = 'BL5_SAVED_ENVELOPE_SPARSE_MECHANICAL_KERNEL_DEV_V1'
MANIFEST_SHA = '4c6ffe7a94e0553b83b67a4f420d11599ed05c1e89b99e5093388a0ad4b0f73a'
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'.qualification/b3-ge-e4-pl/r8-bl5-projected-assessment-dev-20261003-v1'
# Explicit new sparse caller admission, not a replacement for old default guards.
INPUT_BUDGET = SparseBudget(25068, 4_000_000, 1, 768*1024**2)
GRAPH_BUDGET = SparseBudget(15745, 2_000_000, 1, 768*1024**2)
PROOF_BUDGET = IntervalBudget(15585, 1_500_000, 250_000, 50_000_000, 768*1024**2, 450)
LOCAL_COLUMN_LIMIT = 384


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def partition(case, metadata, free, fixed):
    """Derive every retained role from its actual typed/member inventory."""
    counts = case.counts; layout = metadata['layout']
    names = ('shell_nodes','beam_nodes','physical_dofs','members','retained_cell_rotations',
             'retained_resultants','retained_dofs','full_q','multipliers')
    require(layout['counts']=={name:counts[name] for name in names} and
        metadata['fixture']==case.fixture, 'actual case/layout counts differ')
    nfull, nodal, ns = counts['full_q'], counts['physical_dofs'], counts['shell_nodes']
    require(free.ndim==fixed.ndim==1 and free.dtype.kind in 'iu' and fixed.dtype.kind in 'iu' and
        len(fixed)==case.fixed_count and len(free)+len(fixed)==nfull and
        np.array_equal(np.sort(np.r_[free,fixed]),np.arange(nfull)), 'complete actual free/fixed partition required')
    expected_fixed={6*j*(case.nx+1)+axis for j in range(case.ny+1) for axis in range(6)}
    expected_fixed |= set(range(6*ns,6*ns+5))
    require(set(map(int,fixed))==expected_fixed, 'actual fixed root slots differ')
    typed=layout['typed_physical_nodes']
    require([(r['domain'],r['native_node'],r['q_offset']) for r in typed]==
        [('shell',j+1,6*j) for j in range(ns)]+
        [('beam',101+j,6*(ns+j)) for j in range(case.halfcells+1)], 'actual typed physical slots differ')
    retained=layout['retained_members']
    require(len(retained)==counts['members'], 'complete member inventory required')
    blocks=[]
    for member,record in enumerate(retained):
        start=nodal+24*member
        require(record['member_index']==member and record['retained_24_offset']==start and
            record['cell_rotation_slots']==list(range(start,start+6)) and
            record['resultant_slots']==list(range(start+6,start+24)), 'native retained roles differ')
        blocks.append(np.arange(start+6,start+24,dtype=np.int64))
    pp=np.concatenate(blocks);x=np.setdiff1d(free,pp)
    require(np.isin(pp,free).all() and len(x)==counts['physical_free'] and
        len(x)-counts['multipliers']==counts['kernel'] and
        len(free)+counts['multipliers']==case.supported_kkt_count, 'physical/resultant dimensions differ')
    return x, pp, blocks


def support_pivots(case, metadata, physical):
    mapping={int(slot):i for i,slot in enumerate(physical)}
    supports=metadata['supports'];groups=[];pivots=[]
    require(len(supports)==case.supports, 'complete actual support inventory required')
    ns,nodal=case.counts['shell_nodes'],case.counts['physical_dofs']
    for index,support in enumerate(supports):
        cells=support['native_halfcells'];begin=index*case.span
        rows=support['multiplier_rows']
        require(cells==list(range(begin,begin+case.span)) and
            len(rows)==10 and all(type(row) is int for row in rows), 'actual support cells/rows differ')
        a,b=begin+case.span//4,begin+3*case.span//4
        slots=[6*(ns+node)+axis for node in (a,b) for axis in range(3)]
        slots += [nodal+24*(cell//2)+3*(cell%2)+axis for cell in (a,b) for axis in (0,1)]
        require(all(slot in mapping for slot in slots), 'quarter pivots must be physical free coordinates')
        groups.append(np.array(rows,dtype=np.int64))
        pivots.append(np.array([mapping[slot] for slot in slots],dtype=np.int64))
    return groups,pivots


def negative_block_proof(d):
    """Bound local negative work without treating the full matrix as SPD.

    The explicit lower completion is a proof representative. Its error to the
    saved D bounds the symmetric work-form error; no solver matrix is repaired.
    """
    require(d.shape==(18,18) and np.isfinite(d).all(), 'finite native18-resultant block required')
    actual=sparse.csr_matrix(-d);lower=sparse.tril(actual,format='csr')
    representative=(lower+sparse.tril(actual,-1,format='csr').T).tocsr()
    error=bound.difference_upper(actual,representative)
    shift=float(up(2*error))
    budget=IntervalBudget(18,324,171,3000,2*1024**2,10)
    # A bounded complete local pattern, not a model-wide dense object.
    pattern=sparse.csr_matrix(np.tril(np.ones((18,18))))
    proof=certify_decimal_lower(representative,pattern,list(range(18)),shift=shift,budget=budget)
    proof['pivot_lower']=proof['pivot_lower'].tolist()
    proof['pivot_upper']=proof['pivot_upper'].tolist()
    return dict(error_upper=error,shift=shift,proof=proof,
        conclusion='negative saved symmetric work form' if proof['status']=='POSITIVE_SHIFTED_LOWER_COMPLETION' else 'UNRESOLVED_LOCAL_WORK')


def stationary(k,c,x,blocks,report):
    require(c[:,np.concatenate(blocks)].nnz==0, 'resultant variables occur in actual constraints')
    pp=np.concatenate(blocks)
    offblock=k[pp][:,pp]-sparse.block_diag([k[p][:,p] for p in blocks],format='csr')
    require(not np.any(offblock.data), 'nonlocal resultant coupling cannot be eliminated locally')
    full_skew=bound.fro_upper(k-k.T)/max(bound.fro_upper(k),1e-30)
    require(full_skew<=1e-9, 'conservative savedK skew exceeds unchanged criterion')
    report['full_K_skew_relative']=full_skew
    s=k[x][:,x].tocsr();cache={};local=[]
    # Attach live records before their acceptance predicates, including failure.
    report.update(local_negative_work=cache,local_stationary=local)
    for member,p in enumerate(blocks):
        d=k[p][:,p].toarray();key=hashlib.sha256(d.tobytes()).hexdigest()
        if key not in cache:cache[key]=negative_block_proof(d)
        require(cache[key]['proof']['status']=='POSITIVE_SHIFTED_LOWER_COMPLETION',
            'local negative work certificate unresolved: '+str(member))
        coupling=k[p][:,x]
        neighbours=np.union1d(coupling.nonzero()[1],k[x][:,p].nonzero()[0])
        require(len(neighbours)<=24, 'native local stationary neighbourhood exceeds24')
        px=coupling[:,neighbours].toarray();lift=linalg.solve(d,-px,assume_a='gen')
        residual=d@lift+px
        backward=float(linalg.norm(residual,'fro')/max(linalg.norm(d,'fro')*linalg.norm(lift,'fro')+linalg.norm(px,'fro'),1e-30))
        local.append(dict(member=member,block_sha256=key,neighbours=len(neighbours),backward_residual=backward))
        require(backward<=1e-10 and np.isfinite(lift).all(), 'local stationary response unresolved')
        update=k[x[neighbours]][:,p].toarray()@lift
        rows,cols=np.repeat(neighbours,len(neighbours)),np.tile(neighbours,len(neighbours))
        s += sparse.coo_matrix((update.ravel(),(rows,cols)),shape=s.shape).tocsr()
    report.update(local_negative_work=cache,local_stationary=local)
    return s


def bind(case,owner):
    manifest=ROOT/'docs/agent_plans/development/r8-bl5-projected-assessment-20261003/raw-manifest-v8.json'
    require(sha(manifest)==MANIFEST_SHA, 'immutable input campaign manifest differs')
    records={}
    # v8 freezes its own files and the hash chain to the three reused owners.
    for version in (8,7,6,5):
        record=json.loads(manifest.read_text())
        for row in record['files']:
            previous=records.setdefault(row['path'],row['sha256'])
            require(previous==row['sha256'], 'conflicting immutable file bindings')
        if version>5:
            manifest=manifest.with_name(f'raw-manifest-v{version-1}.json')
            require(sha(manifest)==previous_hash(record,version), 'immutable campaign hash chain differs')
    aggregate=RAW/'envelope/aggregate-v1.json'
    require(sha(aggregate)==records[aggregate.relative_to(ROOT).as_posix()], 'aggregate binding differs')
    row=json.loads(aggregate.read_text())['cases'][case.fixture_case]
    summary=json.loads((owner/'summary.json').read_text())
    require(sha(owner/'summary.json')==row['owner_summary_sha256'] and
        summary['status']=='COMPLETED_INSTALLED_DEVELOPMENT_ONLY' and summary['fixture']==case.fixture and
        summary['wheel_sha256']==WHEEL_SHA and summary['matching_modules']==417 and
        summary['factorization_count']==1 and summary['accurate_refinements']==6, 'actual installed owner binding/status differs')
    for name,digest in summary['native_input_sha256'].items():
        require(sha(owner/'native'/name)==digest, 'saved actual native input differs')
    require(len(summary['module_sha256'])==417 and all(
        sha(ROOT/'src'/name)==digest for name,digest in summary['module_sha256'].items()), 'product source graph changed')
    version=2 if case.fixture_case in ('ns64_b4','ns64_b8') else 1
    registration=RAW/'envelope/registrations'/case.fixture_case/f'registration-v{version}.json'
    require(sha(registration)==row['scalar_sha256'], 'independent scalar record differs')
    _,scalar_binding=registration_identity(case,registration,ROOT)
    state_bindings={}
    reused={'ns16_b1_2':RAW/'ratios/whole_half/owner-v1',
        'ns16_b2':RAW/'ratios/whole_double/owner-v1','ns32_b1':RAW/'joint32/installed-v2/owner'}
    canonical=reused.get(case.fixture_case,RAW/'envelope/owners'/case.fixture_case/'owner-v1')
    for load in ('fy','mx','mz'):
        folder=owner/'native'/load;state=json.loads((folder/'raw-state.json').read_text())
        for name in ('raw-state.json','raw-state.npz','physical-actions.json'):
            frozen_file(folder/name,canonical/'native'/load/name,records)
        require(state['status']=='completed' and sha(folder/'raw-state.npz')==state['arrays_sha256'], 'saved complete state differs')
        require(json.loads((folder/'physical-actions.json').read_text())==summary['cases'][load], 'physical action binding differs')
        state_bindings[load]={name:sha(folder/name) for name in ('raw-state.json','raw-state.npz','physical-actions.json')}
    return summary,dict(owner=str(owner.resolve()),summary_sha256=sha(owner/'summary.json'),
        inputs_sha256=summary['native_input_sha256'],states=state_bindings,
        scalar=scalar_binding,manifest_sha256=MANIFEST_SHA,wheel_sha256=WHEEL_SHA)


def frozen_file(actual,canonical,records):
    expected=records[canonical.relative_to(ROOT).as_posix()]
    require(sha(actual)==expected, 'immutable saved state/action file differs')


def previous_hash(record,version):
    if version==7:
        require(record['schema']=='BL5_DEVELOPMENT_RAW_EXTENSION_V7' and
            record['previous_manifest']=='raw-manifest-v6.json', 'immutable v7 manifest schema differs')
        return record['previous_sha256']
    require(version in (8,6), 'explicit immutable manifest predecessor required')
    return record['previous_manifest_sha256']


def incidence_bound(order, supports, errors):
    """Induced2 bound for sum of local errors on their actual two-sided support.

    For ||E_j||2<=delta_j supported on I_j x I_j, weighted Cauchy gives
    |u.T sum(E_j) v| <= max_i sum(j:i in I_j) delta_j ||u||2 ||v||2.
    No symmetry or cancellation assumption is used. Count original supports.
    """
    require(type(order) is int and 0<order<=GRAPH_BUDGET.max_order and
        len(supports)==len(errors), 'bounded matching incidence inventory required')
    weights=np.zeros(order);counts=np.zeros(order,dtype=np.int64)
    for support,error in zip(supports,errors,strict=True):
        indices=np.asarray(support)
        require(indices.ndim==1 and indices.dtype.kind in 'iu' and len(indices)<=24 and
            np.array_equal(indices,np.unique(indices)) and np.all(indices>=0) and
            np.all(indices<order), 'unique bounded actual local support required')
        require(np.isfinite(error) and error>=0, 'finite nonnegative local norm bound required')
        weights[indices]=up(weights[indices]+error);counts[indices]+=1
    return float(weights.max(initial=0.)),weights,counts


def stationary_incidence_error(k,s_hat,scale,x,blocks):
    """Same Schur target and local enclosures; local sum uses induced2 incidence.

    Base scaling, original whole-block-count assembly gamma, and reconstruction
    difference remain unchanged. This is not a reduced scientific tolerance.
    Both Kpx and Kxp determine support even when a stored update cancels to zero.
    """
    base,bl,bh=bound.scaled_data(k[x][:,x],scale,scale)
    reconstructed=base.copy();absolute=abs(base)
    base_error=bound.fro_upper(bound.box_abs(down(bl.data-base.data),up(bh.data-base.data)))
    details=[];supports=[];errors=[]
    for member,block in enumerate(blocks):
        d=k[block][:,block].toarray();_,inv,proof=bound.inverse_bound(d,d)
        coupling=k[block][:,x]
        neighbours=np.union1d(coupling.nonzero()[1],k[x][:,block].nonzero()[0])
        require(len(neighbours)<=24,'native local stationary neighbourhood exceeds24')
        px=coupling[:,neighbours].toarray()*scale[neighbours]
        pxlo,pxhi=down(px),up(px);lift=linalg.solve(d,-px,assume_a='gen')
        rl,rh=bound.box_dot(d,d,lift)
        lift_error=float(up(inv*bound.fro_upper(bound.box_abs(down(rl+pxlo),up(rh+pxhi)))))
        xp=k[x[neighbours]][:,block].toarray()*scale[neighbours,None]
        xplo,xphi=down(xp),up(xp);ul,uh=bound.box_dot(xplo,xphi,lift);update=xp@lift
        arithmetic_error=bound.fro_upper(bound.box_abs(down(ul-update),up(uh-update)))
        local_error=sum_up([arithmetic_error,float(up(bound.fro_upper(bound.box_abs(xplo,xphi))*lift_error))])
        supports.append(neighbours);errors.append(local_error)
        rr,cc=np.repeat(neighbours,len(neighbours)),np.tile(neighbours,len(neighbours))
        u=sparse.coo_matrix((update.ravel(),(rr,cc)),shape=base.shape).tocsr()
        reconstructed+=u;absolute+=abs(u)
        details.append(dict(**proof,member=member,physical_neighbours=neighbours.tolist(),
            lift_error_fro_upper=lift_error,schur_update_error_fro_upper=local_error))
    local_operator,weights,counts=incidence_bound(len(x),supports,errors)
    # Keep the original conservative assembly gamma: no new backend assumption.
    g=bound._gamma(2*(len(blocks)+1))
    assembly_error=float(up(up(g/down(1.-g))*bound.fro_upper(absolute)))
    difference=bound.difference_upper(reconstructed,s_hat)
    total=sum_up([base_error,local_operator,assembly_error,difference])
    return total,details,dict(identity='LOCAL_TWO_SIDED_INCIDENCE_INDUCED2_V1',
        base_scaling_fro_upper=base_error,local_operator_induced2_upper=local_operator,
        original_local_sum_fro_upper=sum_up(errors),assembly_fro_upper=assembly_error,
        assembly_gamma_original_block_count=len(blocks),reconstruction_difference_fro_upper=difference,
        incidence_weights=weights,incidence_counts=counts)


def store_sparse(arrays,name,matrix):
    arrays.update({name+'_data':matrix.data.copy(),name+'_indices':matrix.indices.copy(),
        name+'_indptr':matrix.indptr.copy(),name+'_shape':np.asarray(matrix.shape)})


def assess(case,owner,output,*,stationary_bound='sum',precision=50):
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
            'r8_bl5_projected_envelope_owner_dev.py')},effective_pools=pools,
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
        summary,report['input_binding']=bind(case,owner);save('BOUND_SAVED_INSTALLED_OWNER')
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
        store_sparse(arrays,'Z',z);store_sparse(arrays,'A_representative',representative);save('COMPLETE_MECHANICAL_ERROR_BOUND')
        backend,order=factor_pattern(representative,budget=PROOF_BUDGET)
        pattern=complete_symbolic_pattern(representative,order,budget=PROOF_BUDGET)
        arrays['permutation']=np.asarray(order);store_sparse(arrays,'complete_pattern',pattern)
        report.update(backend_pattern_nnz=backend.nnz,complete_pattern_nnz=pattern.nnz);save('COMPLETE_SYMBOLIC_PATTERN')
        proof=certify_shifted_lower(representative,pattern,order,shift=shift,budget=PROOF_BUDGET)
        arrays['binary64_pivot_lower']=proof.pop('pivot_lower');arrays['binary64_pivot_upper']=proof.pop('pivot_upper')
        report['binary64_interval_proof']=dict(proof);save('BINARY64_INTERVAL_RESULT')
        if proof['status']=='HOLD_INTERVAL_PIVOT':
            proof=certify_decimal_lower(representative,pattern,order,shift=shift,budget=PROOF_BUDGET,precision=precision)
            arrays['pivot_lower']=proof.pop('pivot_lower');arrays['pivot_upper']=proof.pop('pivot_upper')
        report['interval_proof']=proof
        margin=float(down(down(shift-epsilon)/metric))
        passed=proof['status']=='POSITIVE_SHIFTED_LOWER_COMPLETION' and margin>0
        report.update(status='POSITIVE_ENVELOPE_MECHANICAL_KERNEL_DEVELOPMENT_ONLY' if passed else 'HOLD_MECHANICAL_KERNEL',
            physical_kernel_lower_bound=margin if passed else None)
        save('TERMINAL')
        return report
    except BaseException as error:
        report.update(status='FAILED_SPARSE_ENVELOPE_ASSESSMENT',failure=dict(type=type(error).__name__,message=str(error)))
        try:save('FAILED')
        except BaseException as recording_error:error.add_note('Recording also failed: '+repr(recording_error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=[case.fixture_case for case in all_cases()],required=True)
    parser.add_argument('--owner',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stationary-bound',choices=('sum','incidence'),default='sum')
    parser.add_argument('--precision',choices=(50,100),type=int,default=50)
    args=parser.parse_args();result=assess(case_from_id(args.case),args.owner,args.output,
        stationary_bound=args.stationary_bound,precision=args.precision)
    print(json.dumps({key:result[key] for key in ('status','fixture_case','physical_order','kernel_order','kernel_error_upper','physical_kernel_lower_bound','elapsed_seconds')}))
