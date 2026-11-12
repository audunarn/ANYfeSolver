"""Current c2/P0 sparse signed-work diagnostic. Positive probes are UNRESOLVED.

No owner construction, physical solve or F03 gate occurs on import. Stored K
is never averaged. The unadopted exact-congruence proposal is task evidence.
"""
from dataclasses import dataclass
from fractions import Fraction
from copy import deepcopy
import hashlib
import math
import numpy as np
from scipy import sparse,linalg
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse.linalg import LinearOperator,splu

FAMILY={16:(16,4,16),32:(32,8,32),64:(64,16,64)}
class Refused(ValueError):pass
class Unresolved(RuntimeError):pass

def require(ok,message):
    if not ok:raise Refused(message)

def case_spec(level):
    """Literal family construction/load inputs, not execution authority."""
    require(type(level) is int and level in FAMILY,'builtin literal current family level required')
    nx,ny,halfcells=FAMILY[level]
    return dict(level=level,model_arguments=(nx,ny,halfcells,2),span_halfcells=2,
                multiplier_space='BL5_P0_span_2',action_domain='beam',action=(0.,100.,0.,0.,0.,0.))

def declared_endpoint(owner,halfcells):
    """Bind the complete actual compiled native node inventory before using its end."""
    ids=tuple(owner._beam_ids)
    require(ids==tuple(range(101,102+halfcells)),'literal declared native node IDs mismatch')
    return ids,ids[-1]

def sha_matrix(matrix):
    a=matrix.tocsr(copy=True)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.indptr.tobytes()+a.indices.tobytes()+a.data.tobytes()).hexdigest()

def matrices(k,c):
    require(sparse.issparse(k) and sparse.issparse(c),'sparse K/C required')
    require(k.shape[0]==k.shape[1] and c.shape[1]==k.shape[0],'K/C shape mismatch')
    require(k.dtype==np.dtype('float64') and c.dtype==np.dtype('float64'),'binary64 K/C required')
    require(np.isfinite(k.data).all() and np.isfinite(c.data).all(),'nonfinite K/C')
    require(k.has_canonical_format and c.has_canonical_format,'canonical sparse storage required')
    norm=float(sparse.linalg.norm(k));require(norm>0,'zero mechanical operator')
    skew=float(sparse.linalg.norm(k-k.T)/norm)
    require(skew<=1e-9,'stored skew exceeds unchanged signed-check admission')
    return skew

def exact_symmetry_label(k):
    difference=k-k.T;difference.eliminate_zeros()
    return difference.nnz==0

@dataclass
class Snapshot:
    k:object
    c:object
    free:object
    fixed:object
    blocks:tuple
    roles:dict
    coordinate_scale:object
    constraint_scale:object
    identity:dict

def extract_owner(owner,level,policy):
    """Detached compiled snapshot; no owner is constructed or solved."""
    require(level in FAMILY,'literal parity-family level required')
    nx,ny,halfcells=FAMILY[level]
    require((policy.get('beam_halfcells'),policy.get('span_halfcells'),policy.get('c_lambda'))==
            (halfcells,2,2),'current span2 parity family mismatch; doubled beam/span4 is distinct')
    require(policy.get('shell_trace_segments')==nx,'shell trace identity mismatch')
    require(len(owner._definition.lines)==1 and owner._definition.lines[0].multiplier_space=='BL5_P0_span_2',
            'actual declared multiplier space is not BL5 P0/span2')
    k,c=owner._K.copy(),owner._C.copy();free=np.array(owner._free,copy=True);fixed=np.array(owner._fixed,copy=True)
    base=owner._shell_count+6*len(owner._beam_ids);parts=len(owner._beam_parts)
    require(owner._shell_count==(nx+1)*(ny+1)*6 and parts==halfcells//2 and
            k.shape==(base+24*parts,base+24*parts) and c.shape==(5*halfcells//2,k.shape[0]),'actual physical role inventory mismatch')
    require(sorted(map(int,np.r_[free,fixed]))==list(range(k.shape[0])),'support/free coverage mismatch')
    assembled=sparse.bmat([[k[free][:,free],c[:,free].T],[c[:,free],None]],format='csc')
    require(np.max(np.abs((assembled-owner._A).data),initial=0.)==0.,'owner KKT differs from actual K/C/supports')
    blocks=tuple(tuple(base+24*i+j for j in range(6,24)) for i in range(parts))
    roles=dict(shell=np.arange(owner._shell_count),beam_nodes=np.arange(owner._shell_count,base),
        retained_rotations=np.array([base+24*i+j for i in range(parts) for j in range(6)]),resultants=np.array([v for b in blocks for v in b]))
    skew=matrices(k,c)
    identity=dict(family=(nx,ny,halfcells,2),owner_identity=str(owner._identity),k_sha256=sha_matrix(k),c_sha256=sha_matrix(c),
        actual_multiplier_space=owner._definition.lines[0].multiplier_space,stored_skew=skew,
        exact_stored_symmetry=exact_symmetry_label(k),
        operator_scope='exact symmetric' if exact_symmetry_label(k) else 'bounded-skew unsymmetrized action; inertia unresolved',qualification=False)
    return Snapshot(k,c,free,fixed,blocks,roles,np.array(owner._coordinate_scale,copy=True),np.array(owner._constraint_scale,copy=True),identity)

def exact_work_witness(k,c,q,*,resultant_indices=(),fixed_indices=(),term_cap=200000,bit_cap=4096):
    """Specific stored-input negative-work witness, not exact factorization.

    Rounded small gaps/lift residuals do not prove a mechanical kernel direction.
    Exact Cq=0, K[resultant,:]q=0 and negative qTKq are required, excluding the
    deliberately negative independent stress modes. No nonsymmetric inertia.
    """
    require(type(term_cap)is int and term_cap>0 and type(bit_cap)is int and bit_cap>0,'positive integer witness cost limits required')
    q=np.asarray(q);require(q.shape==(k.shape[0],) and np.isfinite(q).all() and np.any(q),'finite nonzero complete witness required')
    observed=float(q@(k@q))
    if not math.isfinite(observed) or observed>=0:return dict(status='UNRESOLVED',blocker='no finite observed negative work; no exact arithmetic attempted',qualification=False)
    fixed=tuple(fixed_indices)
    require(all(type(i) in (int,np.int64,np.int32) and 0<=i<k.shape[0] for i in fixed),'integer fixed witness rows required')
    if any(q[i]!=0. for i in fixed):return dict(status='UNRESOLVED',blocker='nonzero fixed-coordinate witness',qualification=False)
    if k.nnz+c.nnz>term_cap:return dict(status='UNRESOLVED',blocker='witness sparse term cost limit')
    values=[Fraction.from_float(float(x)) for x in q]
    indices=tuple(resultant_indices)
    require(all(type(i) in (int,np.int64,np.int32) and 0<=i<k.shape[0] for i in indices) and len(set(indices))==len(indices),'unique integer resultant witness rows required')
    def bounded(x):
        if max(x.numerator.bit_length(),x.denominator.bit_length())>bit_cap:raise Unresolved('witness rational-bit cost limit')
        return x
    try:
        gaps=[Fraction() for _ in range(c.shape[0])]
        coo=c.tocoo(copy=False)
        for i,j,v in zip(coo.row,coo.col,coo.data):gaps[i]=bounded(gaps[i]+bounded(Fraction.from_float(float(v))*values[j]))
        coo=k.tocoo(copy=False);work=Fraction();stationarity={int(i):Fraction() for i in indices}
        for i,j,v in zip(coo.row,coo.col,coo.data):
            action=bounded(Fraction.from_float(float(v))*values[j])
            if int(i) in stationarity:stationarity[int(i)]=bounded(stationarity[int(i)]+action)
            work=bounded(work+bounded(values[i]*action))
        stationary=all(x==0 for x in stationarity.values());constraint_zero=all(x==0 for x in gaps);admissible=constraint_zero and stationary
        return dict(status='NEGATIVE_STORED_WORK_WITNESS' if admissible and work<0 else 'UNRESOLVED',exact_constraint_zero=constraint_zero,
                    work_sign=(work>0)-(work<0),work_numerator=str(work.numerator),work_denominator=str(work.denominator),
                    exact_resultant_stationarity=stationary,exact_fixed_zero=True,
                    blocker=None if admissible and work<0 else ('nonstationary resultant witness' if not stationary else
                        ('constraint witness is not exactly admissible' if not constraint_zero else 'exact work is nonnegative')),
                    whole_kernel_positive=None,inertia=None,qualification=False)
    except Unresolved as exc:return dict(status='UNRESOLVED',blocker=str(exc),qualification=False)

class SparseSignedPreparation:
    """Local stationary lift plus implicit sparse Z/ZT. No global dense array."""
    def __init__(self,snapshot):
        snapshot=Snapshot(snapshot.k.copy(),snapshot.c.copy(),np.array(snapshot.free,copy=True),np.array(snapshot.fixed,copy=True),
            tuple(tuple(b) for b in snapshot.blocks),{n:np.array(v,copy=True) for n,v in snapshot.roles.items()},
            np.array(snapshot.coordinate_scale,copy=True),np.array(snapshot.constraint_scale,copy=True),deepcopy(snapshot.identity))
        self.snapshot=snapshot;self.skew=matrices(snapshot.k,snapshot.c)
        k,c=snapshot.k,snapshot.c;free=np.asarray(snapshot.free)
        require(free.ndim==1 and free.dtype.kind in 'iu' and len(np.unique(free))==len(free),'unique integer free inventory required')
        require(np.all(free>=0) and np.all(free<k.shape[0]),'free coordinate out of range')
        self.p=np.array([v for b in snapshot.blocks for v in b],dtype=int)
        require(len(np.unique(self.p))==len(self.p) and set(map(int,self.p))<=set(map(int,free)),'overlapping or supported resultant block')
        require(c[:,self.p].nnz==0,'physical constraints contain resultants')
        self.x=np.setdiff1d(free,self.p);self.local=[];self.local_evidence=[]
        for block in snapshot.blocks:
            block=np.array(block);other=np.setdiff1d(self.p,block)
            require(k[block][:,other].nnz==0,'resultant blocks are coupled')
            d=k[block][:,block].toarray() # only bounded local native blocks
            require(0<len(d)<=18,'local D block size unsupported')
            if not np.array_equal(d,d.T):raise Unresolved('local operator exact symmetry unproved; not a physical-defect determination')
            try:factor=linalg.cho_factor(-d,lower=True)
            except linalg.LinAlgError as exc:raise Unresolved('local D is not numerically negative definite') from exc
            chol=np.tril(factor[0]);ferr=d+chol@chol.T
            budget=float(linalg.norm(ferr,2)+64*np.finfo(float).eps*len(d)*linalg.norm(d,2))
            upper=float(linalg.eigvalsh(d)[-1]);require(-upper>budget,'local negative block unresolved')
            self.local.append((block,factor,k[block][:,self.x].tocsr()))
            self.local_evidence.append(dict(max_eigenvalue=upper,numerical_uncertainty=budget))
        self.cx=c[:,self.x].tocsr()
        require(0<c.shape[0]<len(self.x),'nonempty constraint/kernel dimensions required')
        pattern=self.cx.copy();pattern.data=np.ones_like(pattern.data)
        self.pivot=maximum_bipartite_matching(pattern,perm_type='column')
        require(len(self.pivot)==c.shape[0] and np.all(self.pivot>=0) and len(np.unique(self.pivot))==len(self.pivot),'no complete sparse structural matching')
        self.rest=np.setdiff1d(np.arange(len(self.x)),self.pivot)
        self.cp=self.cx[:,self.pivot].tocsc();self.cr=self.cx[:,self.rest].tocsr()
        try:self.factor=splu(self.cp)
        except RuntimeError as exc:raise Unresolved('sparse constraint pivot factorization failed') from exc
        self.operator=LinearOperator((len(self.rest),len(self.rest)),matvec=self.action,rmatvec=self.transpose_action,dtype=float)

    def solve_pivot(self,rhs,transpose=False):
        trans='T' if transpose else 'N';a=self.cp.T if transpose else self.cp
        out=self.factor.solve(np.asarray(rhs),trans=trans)
        for _ in range(3):out+=self.factor.solve(np.asarray(rhs)-a@out,trans=trans)
        residual=np.linalg.norm(a@out-rhs);scale=np.linalg.norm(rhs)
        require(np.isfinite(out).all(),'nonfinite constraint pivot solve')
        require(residual==0. if scale==0. else residual<=1e-10*scale,'constraint pivot backward residual exceeds unchanged1e-10')
        return out

    def z(self,v):
        v=np.asarray(v);require(v.shape==(len(self.rest),) and np.isfinite(v).all(),'invalid kernel parameter')
        out=np.zeros(len(self.x));out[self.rest]=v;out[self.pivot]=-self.solve_pivot(self.cr@v)
        return out

    def zt(self,v):
        v=np.asarray(v);require(v.shape==(len(self.x),),'invalid kernel dual')
        return v[self.rest]-self.cr.T@self.solve_pivot(v[self.pivot],True)

    def lift(self,x):
        q=np.zeros(self.snapshot.k.shape[0]);q[self.x]=x
        for block,factor,coupling in self.local:q[block]=linalg.cho_solve(factor,coupling@x)
        return q

    def action(self,v):
        return self.zt((self.snapshot.k@self.lift(self.z(v)))[self.x])

    def transpose_action(self,v):
        x=self.z(v);q=np.zeros(self.snapshot.k.shape[0]);q[self.x]=x;raw=self.snapshot.k.T@q
        for block,factor,coupling in self.local:raw[self.x]+=coupling.T@linalg.cho_solve(factor,raw[block])
        return self.zt(raw[self.x])

    def probe(self,v):
        q=self.lift(self.z(v));gap=self.snapshot.c@q;require(np.any(q),'zero signed-work probe')
        work=float(q@(self.snapshot.k@q));pull=float(np.asarray(v)@self.action(v));denominator=max(abs(work),abs(pull))
        require(work==pull if denominator==0 else abs(work-pull)<=1e-10*denominator,'stationary work transpose mismatch')
        witness=(exact_work_witness(self.snapshot.k,self.snapshot.c,q,resultant_indices=tuple(map(int,self.p)),fixed_indices=tuple(map(int,self.snapshot.fixed)))
            if math.isfinite(work) and work<0 else dict(status='UNRESOLVED',blocker='positive/zero probe cannot certify whole kernel',qualification=False))
        return dict(status=witness['status'],qualification=False,full_work=work,reduced_work=pull,weak_gap=float(np.max(np.abs(gap),initial=0.)),
            scaled_gap=float(np.max(np.abs(gap)/self.snapshot.constraint_scale,initial=0.)),witness=witness,
            kernel_positive=None,inertia=None,stored_skew=self.skew,local_blocks=self.local_evidence,
            scope='sampled stored signed work; positive samples never certify whole kernel')

def role_parity(snapshot,a,b):
    values={}
    for name,slots in snapshot.roles.items():
        slots=np.asarray(slots,dtype=int);x=np.asarray(a)[slots]/snapshot.coordinate_scale[slots];y=np.asarray(b)[slots]/snapshot.coordinate_scale[slots]
        norm=max(np.linalg.norm(x),np.linalg.norm(y))
        values[name]=float(np.linalg.norm(x-y)/norm) if norm else (0. if np.array_equal(x,y) else math.inf)
    return values

def prepare_existing_owner(owner,level,policy):
    return SparseSignedPreparation(extract_owner(owner,level,policy))

def stationary_system(prepared):
    """Sparse S/Cx; only each bounded native neighbor block is materialized."""
    k=prepared.snapshot.k;s=k[prepared.x][:,prepared.x].tocsr()
    for block,factor,coupling in prepared.local:
        left=k[prepared.x][:,block]
        neighbours=np.union1d(coupling.nonzero()[1],left.nonzero()[0]);require(len(neighbours)<=24,'local native coupling closure exceeds prior bound24')
        local=coupling[:,neighbours].toarray()
        update=left[neighbours].toarray()@linalg.cho_solve(factor,local)
        s=s+sparse.coo_matrix((update.ravel(),(np.repeat(neighbours,len(neighbours)),np.tile(neighbours,len(neighbours)))),shape=s.shape).tocsr()
    return s,prepared.cx

def reduced_state(prepared,load):
    from anyfesolver._b2_physical_linear_batch_dev import _accurate_sparse_residual
    load=np.asarray(load);require(load.shape==(prepared.snapshot.k.shape[0],) and np.isfinite(load).all(),'complete finite applied load required')
    require(np.max(np.abs(load[prepared.p]),initial=0.)==0.,'loaded resultant coordinates are outside stationary zero-RHS scope')
    s,c=stationary_system(prepared);a=sparse.bmat([[s,c.T],[c,None]],format='csc')
    rhs=np.r_[load[prepared.x],np.zeros(c.shape[0])]
    try:factor=splu(a)
    except RuntimeError as exc:raise Unresolved('stationary saddle factorization failed') from exc
    value=factor.solve(rhs)
    for _ in range(3):value+=factor.solve(-_accurate_sparse_residual(a,value[:,None],rhs[:,None])[:,0])
    require(np.isfinite(value).all(),'nonfinite reduced state')
    return prepared.lift(value[:len(prepared.x)]),value[len(prepared.x):]

def state_metrics(snapshot,q,lm,load):
    from anyfesolver._stiffened_shell_ge_experimental import _linear_equilibrium,_scaled_constraint_gap
    residual=snapshot.k@q+snapshot.c.T@lm-load;work=float(q@load)
    require(np.isfinite(residual).all() and math.isfinite(work) and work!=0.,'nonzero finite physical work denominator required')
    reactions=np.zeros_like(q);reactions[snapshot.fixed]=residual[snapshot.fixed]
    return dict(equilibrium=float(_linear_equilibrium(residual[snapshot.free],load[snapshot.free],snapshot.coordinate_scale[snapshot.free])),
        weak_gap=float(_scaled_constraint_gap(snapshot.c@q,snapshot.constraint_scale)),
        applied_work=work,energy_work_relative=abs(float(q@(snapshot.k@q))-work)/abs(work),
        compatible_work_relative=abs(float(lm@(snapshot.c@q)))/abs(work),
        ground_work_relative=abs(float(q@reactions))/abs(work))

def case_disposition(physical,parity_ok,probes):
    """A certified negative stationary witness stops progression, never hidden."""
    if any(p.get('status')=='NEGATIVE_STORED_WORK_WITNESS' for p in probes):return 'NEGATIVE_STORED_WORK_WITNESS'
    return 'UNRESOLVED_KERNEL' if physical and parity_ok else 'FAILED_PHYSICAL_OR_PARITY'

def construction_provenance(model):
    """Bind actual source producers, including the top-level fixture import."""
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    fixture=model.__globals__.get('_bl5_joint_mesh_definition')
    require(callable(fixture),'actual model fixture producer is missing')
    require(model.__module__=='scripts.b3_e4_bl5_coherent_aggregation_dev' and
            fixture.__module__=='test_stiffened_shell_l1_assembly_experimental','case/fixture producer module identity mismatch')
    producers=dict(candidate=Path(__file__).resolve(),case=Path(sys.modules[model.__module__].__file__).resolve(),
                   fixture=Path(sys.modules[fixture.__module__].__file__).resolve())
    expected=dict(candidate=root/'scripts/r8_f03_sparse_signed_family_dev.py',case=root/'scripts/b3_e4_bl5_coherent_aggregation_dev.py',
                  fixture=root/'tests/test_stiffened_shell_l1_assembly_experimental.py')
    require(all(p==expected[n].resolve() for n,p in producers.items()),'actual fixture/case producer escaped assigned source path')
    return {n:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in producers.items()}

def family_case(level,output):
    """Future authorized ONE case caller. Not invoked by source/toy tests."""
    import json,sys
    from pathlib import Path
    spec=case_spec(level)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    nx,ny,halfcells,_=spec['model_arguments']
    owner=None
    record=dict(status='FAILED',qualification=False,family=spec['model_arguments'],requested_case=spec,stage='imports')
    try:
        from anyfesolver import stiffened_shell
        from scripts.b3_e4_bl5_coherent_aggregation_dev import model
        record['source_producers']=construction_provenance(model)
        record['stage']='model-construction'
        definition,policy=model(*spec['model_arguments'])
        record['stage']='owner-construction'
        owner=stiffened_shell.create_analysis(definition=definition)
        record['stage']='compiled-extraction'
        snapshot=extract_owner(owner,level,policy)
        ids,endpoint=declared_endpoint(owner,halfcells)
        resultant_count=sum(len(b) for b in snapshot.blocks)
        record.update(endpoint_node=endpoint,beam_node_ids=ids,shell_node_ids=tuple(definition.shell.node_ids),
            input_policy=policy,operator_identity=snapshot.identity,roles={n:np.asarray(v).tolist() for n,v in snapshot.roles.items()},
            literal_inputs=dict(shell_x_intervals=nx,shell_y_intervals=ny,native_halfcells=halfcells,c_lambda=2,span_halfcells=2,
                action_domain=spec['action_domain'],action_node=endpoint,action=spec['action']),
            shape=dict(full=snapshot.k.shape[0],free=len(snapshot.free),local_resultants=resultant_count,
                physical=len(snapshot.free)-resultant_count,constraints=snapshot.c.shape[0],kernel=len(snapshot.free)-resultant_count-snapshot.c.shape[0]))
        # Preserve the actual failed operator/layout before any local preparation.
        sparse.save_npz(out/'K.npz',snapshot.k);sparse.save_npz(out/'C.npz',snapshot.c)
        np.savez(out/'compiled-layout.npz',free=snapshot.free,fixed=snapshot.fixed,coordinate_scale=snapshot.coordinate_scale,
            constraint_scale=snapshot.constraint_scale,**snapshot.roles)
        (out/'inputs.json').write_text(json.dumps(record,indent=2,allow_nan=False))
        record['stage']='native-equilibrium'
        result=owner.solve_linear((stiffened_shell.NodalAction(spec['action_domain'],endpoint,spec['action']),))
        record['public_result']=dict(status=str(result.status),failure=repr(getattr(result,'failure',None)),message=str(getattr(result,'message','')))
        if result.status!='completed':
            record['status']='FAILED_PUBLIC_SOLVE'
            raise Refused('public solve_linear returned noncompleted result: '+str(result.status))
        q=np.r_[result.shell_displacement,result.beam_coordinates];lm=np.asarray(result.multipliers);load=owner._accepted[1].copy()
        np.savez(out/'owner-state.npz',owner_q=q,owner_lm=lm,load=load)
        # Preserve the native accepted prefix even if local-D/Cp preparation refuses.
        record['stage']='stationary-preparation'
        prep=SparseSignedPreparation(snapshot)
        record['stage']='stationary-reduced-equilibrium'
        rq,rlm=reduced_state(prep,load)
        np.savez(out/'states.npz',owner_q=q,owner_lm=lm,reduced_q=rq,reduced_lm=rlm,load=load,free=snapshot.free,fixed=snapshot.fixed,
            x=prep.x,pivot=prep.pivot,rest=prep.rest,coordinate_scale=snapshot.coordinate_scale,constraint_scale=snapshot.constraint_scale)
        record['stage']='physical-and-parity-assessment'
        parity=role_parity(snapshot,q,rq)
        denominator=max(np.linalg.norm(lm*snapshot.constraint_scale),np.linalg.norm(rlm*snapshot.constraint_scale))
        require(denominator>0,'unresolved zero multiplier parity denominator')
        parity['multipliers']=float(np.linalg.norm((lm-rlm)*snapshot.constraint_scale)/denominator)
        owner_metrics=state_metrics(snapshot,q,lm,load);reduced_metrics=state_metrics(snapshot,rq,rlm,load)
        reactions=[]
        for state,multiplier in ((q,lm),(rq,rlm)):
            residual=snapshot.k@state+snapshot.c.T@multiplier-load
            reaction=np.zeros_like(state);reaction[snapshot.fixed]=residual[snapshot.fixed];reactions.append(reaction*snapshot.coordinate_scale)
        norm=max(np.linalg.norm(v) for v in reactions)
        parity['reactions']=float(np.linalg.norm(reactions[0]-reactions[1])/norm) if norm else 0.
        probes=[prep.probe(q[prep.x][prep.rest]),prep.probe(np.ones(len(prep.rest)))]
        physical=all(v['equilibrium']<=1e-8 and v['weak_gap']<=1e-9 and
            max(v['energy_work_relative'],v['compatible_work_relative'],v['ground_work_relative'])<=1e-8 for v in (owner_metrics,reduced_metrics))
        parity_ok=all(v<=1e-10 for v in parity.values())
        modules={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items() if n.startswith(('anyfesolver','scripts.')) and getattr(m,'__file__',None)}
        root=Path(__file__).resolve().parents[1]
        require(all(Path(p).is_relative_to(root/'src' if n.startswith('anyfesolver') else root/'scripts') for n,p in modules.items()),'import escaped assigned source')
        record.update(status=case_disposition(physical,parity_ok,probes),stop_family_progression=True,
            stage='completed',physical_parity_pass=bool(physical and parity_ok),
            owner_metrics=owner_metrics,reduced_metrics=reduced_metrics,parity=parity,probes=probes,kernel_positive=None,inertia=None,
            provenance={n:dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for n,p in modules.items()},
            admission='all positive sampled values remain unresolved; no whole-kernel certificate')
        return record
    except BaseException as exc:
        import traceback
        record.update(exception=repr(exc),traceback=traceback.format_exc())
        if isinstance(exc,Unresolved):record['status']='UNRESOLVED_LOCAL_OPERATOR'
        raise
    finally:
        (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False))
        if owner is not None:owner.close()

if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description='Only execute after separate actual-case authorization; positive samples are unresolved')
    parser.add_argument('--level',type=int,choices=tuple(FAMILY),required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(family_case(args.level,args.output),indent=2))
