"""Private G3c finite mixed-graph development owner; NOT qualification.

One immutable graph generation; actual native material transactions are confined
to disposable sandboxes. No public factory or checkpoint-import admission.
"""
from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
import sys
import threading
from time import monotonic
import numpy as np
from . import _ge_beam3_g1_element as native_module
from . import _ge_beam3_g1_operator as operator_module
from . import _ge_beam3_g3c_local_beam as beam_module
from . import _ge_beam3_g3c_local_shell as shell_module
from . import _ge_beam3_pose_joint as joint_module
from . import nonlinear_state as state_module
from . import elements as legacy_module
from . import _native_rotation_state as rotation_module
from . import _ge_beam3_g3c_definition as definition_module
from ._ge_beam3_g3c_definition import expand, command, CONTRACT_SHA, U, SHIFT
from ._ge_beam3_g1_analysis import runtime_digest
from ._ge_beam3_g1_elastic import ElasticSection, canonical, owned, sha, solve
from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from ._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, so3_exp, so3_log, rotation_log
from .fe_core import FEModel

POLICY='GE_BEAM3_G3C_MIXED_ELASTIC_OWNER_V1'
SCHEMA='GE_BEAM3_G3C_MIXED_ELASTIC_STATE_V1'
ENVIRONMENT='2ce154b866565e1d03019651b1ae7fdd070e5554f38d72adf21d8080558b6756'

def progress(stage):
    print('G3C CHECKPOINT '+stage,flush=True)

def dispatch():
    modules=(native_module,operator_module,beam_module,shell_module,joint_module,
             state_module,legacy_module,rotation_module,sys.modules[__name__])
    found=[]
    def functions(name,value):
        if isinstance(value,property):
            return [(name+'.'+k,fn) for k,fn in (('get',value.fget),('set',value.fset),('del',value.fdel)) if fn is not None]
        if isinstance(value,(staticmethod,classmethod)): value=value.__func__
        return [(name,value)] if inspect.isfunction(value) else []
    for module in modules:
        for name,value in sorted(vars(module).items()):
            made=functions(name,value)
            if inspect.isclass(value):
                # Include imported classes and inherited methods/properties,
                # notably Reference and ElasticSection, not just local classes.
                for base in value.__mro__:
                    for method,fn in sorted(vars(base).items()):
                        made.extend(functions(name+'.'+base.__module__+'.'+base.__qualname__+'.'+method,fn))
            for name,fn in made:
                # Code objects are immutable; replacing __code__ changes this
                # identity. marshal encodings are not an identity primitive.
                found.append((module.__name__,name,fn,id(fn.__code__)))
    return tuple(found)

def close(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.norm(a-b)>1e-11*max(1.,np.linalg.norm(a),np.linalg.norm(b)):
        raise ValueError(label)

def native_payload(body):
    # Only called on bytes already owned by this generation, never an import.
    value=json.loads(canonical(body))
    value['history']=(); value['response']['history']=()
    return value

@dataclass(frozen=True)
class Generation:
    state: bytes
    history: bytes
    state_sha256: str
    history_sha256: str

def generation(state,history):
    raw=canonical(state); journal=canonical(history)
    if len(raw)+len(journal)>8388608: raise ValueError('generation byte bound')
    return Generation(raw,journal,sha256(raw).hexdigest(),sha256(journal).hexdigest())

class MixedGraphOwner:
    __slots__=('_definition','_expanded','_programs','_seal','_runtime','_dispatch',
               '_published','_initial','_lock','_owned_lock','_active','_serial','_poisoned')
    def __setattr__(self,name,value): raise AttributeError('owned graph attributes are immutable')
    def __delattr__(self,name): raise AttributeError('owned graph attributes are immutable')

    def __init__(self,fixture_id='J_B2_PAIR',variant='BASE',common_motion='NONE'):
        if type(self) is not MixedGraphOwner: raise ValueError('exact owner required')
        definition,expanded,programs=expand(fixture_id,variant,common_motion)
        for name,value in dict(_definition=canonical(definition),_expanded=canonical(expanded),
            _programs=canonical(programs),_runtime=sha(dict(source=runtime_digest(),environment=ENVIRONMENT)),
            _dispatch=dispatch(),_lock=threading.Lock(),_active=None,_serial=0,_poisoned=False).items():
            object.__setattr__(self,name,value)
        object.__setattr__(self,'_owned_lock',self._lock)
        object.__setattr__(self,'_seal',sha((self._definition.hex(),self._expanded.hex(),self._programs.hex())))
        progress('initialization')
        g=expanded['graph']; n=len(g['nodes']); zero=np.zeros(6*n); q=np.tile(np.eye(3),(n,1,1))
        model,elements,states=self._native_model(None)
        rows=self._adapter_rows(zero,q,q,epoch=0,previous=None,check=lambda stage:None)[2]
        initial=dict(schema=SCHEMA,epoch=0,definition_sha256=sha(definition),previous_sha256=None,
            total_u=zero,rotations=q,native_rows=[dict(element_id=e.element_id,payload=states[e.element_id]) for e in elements],
            adapter_rows=rows,multipliers=np.zeros(6*(len(g['fixed_nodes'])+len(g['joints']))))
        made=generation(initial,[])
        object.__setattr__(self,'_published',made); object.__setattr__(self,'_initial',made.state_sha256)
        self._guard(full=True)
        # Analytical component rigid spaces, not a stiffness eigenvalue cutoff.
        rows,H=self._constraints(zero,q,np.zeros(len(initial['multipliers'])),None)
        J=H[len(zero):,:len(zero)]
        components=[]
        for e in g['elements']:
            touching=[c for c in components if set(e['nodes']) & c]
            merged=set(e['nodes'])
            for c in touching: merged.update(c); components.remove(c)
            components.append(merged)
        ids=[r[0] for r in g['nodes']]; X=np.array([r[1] for r in g['nodes']],dtype=float)
        R=np.zeros((6*n,6*len(components)))
        for c,nodes in enumerate(components):
            origin=X[ids.index(min(nodes))]
            for node in nodes:
                i=ids.index(node); R[6*i:6*i+3,6*c:6*c+3]=np.eye(3)
                R[6*i:6*i+3,6*c+3:6*c+6]=np.column_stack([np.cross(a,X[i]-origin) for a in np.eye(3)])
                R[6*i+3:6*i+6,6*c+3:6*c+6]=np.eye(3)
        JR=J@R; singular=np.linalg.svd(JR,compute_uv=False)
        floor=64*np.finfo(float).eps*max(JR.shape)*max(1.,singular[0])
        if len(singular)<R.shape[1] or singular[-1]<=floor: raise ValueError('unsupported component rigid space')

    def _guard(self,origin=None,*,full=False,lock=None):
        try:
            current=dispatch()
            if current!=self._dispatch:
                before={(m,n):(f,c) for m,n,f,c in self._dispatch}
                after={(m,n):(f,c) for m,n,f,c in current}
                names=sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))
                raise ValueError('runtime dispatch changed: '+repr(names[:5]))
            if (type(self) is not MixedGraphOwner or self._poisoned or
                self._lock is not self._owned_lock or
                (lock is not None and self._lock is not lock) or
                U!=((0,-1,0),(1,0,0),(0,0,1)) or SHIFT!=(2,-3,1) or
                definition_module.U!=U or definition_module.SHIFT!=SHIFT or
                sha((self._definition.hex(),self._expanded.hex(),self._programs.hex()))!=self._seal or
                (origin is not None and self._published is not origin) or
                sha256(self._published.state).hexdigest()!=self._published.state_sha256 or
                sha256(self._published.history).hexdigest()!=self._published.history_sha256):
                raise ValueError('captured graph/runtime/generation changed')
            if full:
                definition,expanded,programs=expand(**{k:json.loads(self._definition)[k] for k in ('fixture_id','variant','common_motion')})
                if (canonical(definition)!=self._definition or canonical(expanded)!=self._expanded or
                    canonical(programs)!=self._programs or sha(dict(source=runtime_digest(),environment=ENVIRONMENT))!=self._runtime):
                    raise ValueError('frozen source or definition changed')
        except BaseException:
            object.__setattr__(self,'_poisoned',True)
            raise

    @property
    def size(self): return 6*len(json.loads(self._expanded)['graph']['nodes'])

    def snapshot_bytes(self):
        # Diagnostic snapshot only; no import/restart API until full preflight gate.
        lock=self._owned_lock
        if not lock.acquire(False): raise RuntimeError('owner in use')
        try:
            self._guard(full=True,lock=lock)
            return canonical(dict(kind='UNQUALIFIED_G3C_GENERATION_DIAGNOSTIC',
                state=json.loads(self._published.state),history=json.loads(self._published.history),
                state_sha256=self._published.state_sha256,history_sha256=self._published.history_sha256))
        finally: lock.release()

    def _native_model(self,state):
        data=json.loads(self._expanded); g=data['graph']; ids=[r[0] for r in g['nodes']]
        points={n:np.asarray(x,dtype=float) for n,x in g['nodes']}; model=FEModel('private G3c sandbox')
        for n in ids: model.add_node(n,*points[n])
        elements=[]
        for row in g['elements']:
            if row['family']!='NATIVE': continue
            x=np.array([points[n] for n in row['nodes']]); axis=x[2]-x[0]; axis/=np.linalg.norm(axis)
            direction=np.array(row['orientation'],dtype=float); direction-=axis*(axis@direction); direction/=np.linalg.norm(direction)
            frame=np.column_stack((axis,direction,np.cross(axis,direction)))
            if row['roll_radians']!=0: raise ValueError('unregistered fixture roll')
            e=native_module.ElasticElement(row['id'],tuple(row['nodes']),Reference(x,np.tile(frame,(3,1,1))),
                ElasticSection.isotropic(**data['definitions']['native']),order=4)
            model.add_element(e.element_id,e); model.materials[e.material_name]=e.section; elements.append(e)
        from .matrix_assembly import _get_cached_sparsity_pattern
        _get_cached_sparsity_pattern(model.mesh,'tangent_stiffness')
        if model.mesh.dof_manager.total_dofs!=6*len(ids): raise ValueError('global DOF allocation changed')
        states=({e.element_id:e.init_model_bound_nonlinear_state(model.mesh,e.section,1) for e in elements}
            if state is None else {r['element_id']:native_payload(r['payload']) for r in state['native_rows']})
        return model,tuple(elements),states

    def _adapter_rows(self,total,qa,q,*,epoch,previous,check):
        data=json.loads(self._expanded); g=data['graph']; ids=[r[0] for r in g['nodes']]
        points={n:np.array(x,dtype=float) for n,x in g['nodes']}; size=6*len(ids)
        force=np.zeros(size); H=np.zeros((size,size)); envelopes=[]; diagnostics={}
        previous_rows={} if previous is None else {r['element_id']:r for r in previous['adapter_rows']}
        anchors={r['element_id']:r['anchor_node'] for r in data['anchors']}
        origin=None if previous is None else sha(previous)
        for row in g['elements']:
            if row['family']=='NATIVE': continue
            indices=[ids.index(n) for n in row['nodes']]; dofs=np.array([6*i+j for i in indices for j in range(6)])
            x=np.array([points[n] for n in row['nodes']]); u=total[dofs].copy()
            if previous is not None:
                u.reshape(-1,6)[:,3:]-=np.asarray(previous['total_u'])[dofs].reshape(-1,6)[:,3:]
            if row['family'] in ('B2','B3'):
                d=data['definitions']['legacy']
                section={k:d[k] for k in ('area','Iy','Iz','J','shear_factor_y','shear_factor_z','orientation')}
                local=beam_module.LocalBeam(row['family'],tuple(row['nodes']),x,anchors[row['id']],d['E'],d['nu'],section)
                trial=local.evaluate(u,owned(qa[indices])); diagnostic_hash=sha(None)
            else:
                direction=np.array([1.,0.,0.])
                if json.loads(self._definition)['variant']=='PROPER_GLOBAL_TRANSFORM': direction=np.asarray(U)@direction
                local=shell_module.LocalShell('Q4' if row['family']=='Q4' else 'S3-V2D',tuple(row['nodes']),x,
                    np.array(data['definitions']['shell']['reference_normal'],dtype=float),direction)
                trial=local.evaluate(u,owned(qa[indices])); diagnostic_hash=sha256(trial.candidate).hexdigest()
            close(trial.kinematics.rotations,q[indices],'local physical pose mismatch')
            force[dofs]+=trial.chart_force; H[np.ix_(dofs,dofs)]+=trial.chart_hessian
            envelopes.append(dict(element_id=row['id'],policy=data['local_policies'][row['family']],
                definition_sha256=trial.definition_sha256,epoch=epoch,
                previous_sha256=None if previous is None else sha(previous_rows[row['id']]),
                origin_sha256=origin,pose_sha256=sha(dict(node_ids=row['nodes'],
                    total_translations=total[dofs].reshape(-1,6)[:,:3],rotations=q[indices])),
                deformation=trial.kinematics.deformation,diagnostic_sha256=diagnostic_hash,
                source_material_committed=False,physical_recovery_complete=False))
            diagnostics[row['id']]=trial
            check('family:'+str(row['id']))
        return owned(force),owned(H),envelopes,diagnostics

    def _targets(self,cmd):
        definition=json.loads(self._definition); programs=json.loads(self._programs)
        motion=definition['common_motion']; W=np.eye(3); t=np.zeros(3)
        if motion!='NONE' and cmd is not None:
            s=cmd['step']/4 if cmd['kind']=='PREPARE_COMMON_MOTION' else 1.
            W=rotation_module.rotation_exponential(s*np.array(programs['common_rotation_vectors'][int(motion[-1])]))
            t=s*np.array(SHIFT,dtype=float)
            if definition['variant']=='PROPER_GLOBAL_TRANSFORM':
                transform=np.array(U,dtype=float); W=transform@W@transform.T
                t=transform@t+np.array(SHIFT)-W@np.array(SHIFT)
        root=np.eye(3); force=np.zeros(3)
        if cmd is not None and cmd['kind']=='LOAD_STAGE':
            v1,v2=np.array(programs['root_increment_vectors'],dtype=float)
            if definition['variant']=='PROPER_GLOBAL_TRANSFORM': v1=np.asarray(U)@v1; v2=np.asarray(U)@v2
            r1=rotation_module.rotation_exponential(v1); r2=rotation_module.rotation_exponential(v2)
            root=[np.eye(3),r1,r2@r1,r1,np.eye(3)][cmd['root_stage']]
            force=cmd['load_factor']*cmd['force_scale']*np.asarray(programs['force'])
            if definition['variant']=='PROPER_GLOBAL_TRANSFORM': force=np.asarray(U)@force
            force=W@force
        return W,t,root,force

    def _constraints(self,total,qa,mu,cmd):
        g=json.loads(self._expanded)['graph']; ids=[r[0] for r in g['nodes']]; X=np.array([r[1] for r in g['nodes']],dtype=float)
        size=6*len(ids); count=6*(len(g['fixed_nodes'])+len(g['joints']))
        residual=np.zeros(size+count); tangent=np.zeros((size+count,size+count))
        previous=np.asarray(json.loads(self._published.state)['total_u'])
        eta=(total-previous).reshape(-1,6)[:,3:]; W,t,root,_=self._targets(cmd)
        for k,node in enumerate(g['fixed_nodes']):
            i=ids.index(node); target=W@(root if k==0 else np.eye(3)); xtarget=W@X[i]+t
            z=[Jet2.variable(float(total[6*i+j]),j,6) for j in range(6)]
            dq=[z[j+3]-float(previous[6*i+j+3]) for j in range(3)]
            made=matmul(constant_matrix(target.T,6),matmul(so3_exp(dq),constant_matrix(qa[i],6)))
            rows=[z[j]+float(X[i,j])-float(xtarget[j]) for j in range(3)]+so3_log(made)
            J=np.array([v.gradient for v in rows]); h=np.einsum('i,ijk->jk',mu[6*k:6*k+6],np.array([v.hessian for v in rows]))
            idx=np.arange(6*i,6*i+6); m=np.arange(size+6*k,size+6*k+6)
            residual[idx]+=J.T@mu[6*k:6*k+6]; residual[m]=[v.value for v in rows]
            tangent[np.ix_(idx,idx)]+=h; tangent[np.ix_(idx,m)]+=J.T; tangent[np.ix_(m,idx)]+=J
        for j,row in enumerate(g['joints']):
            ij=[ids.index(row['master']),ids.index(row['slave'])]; k=len(g['fixed_nodes'])+j
            d0=np.array([row['master_frame'],row['slave_frame']],dtype=float)
            joint=joint_module.RigidPoseJoint(X[ij],d0)
            result=joint.evaluate_ports(X[ij]+total.reshape(-1,6)[ij,:3],qa[ij]@d0,
                owned(mu[6*k:6*k+6]),rotation_coordinates=owned(eta[ij]),charted_ports=(True,True))
            idx=np.array([6*i+l for i in ij for l in range(6)]+list(range(size+6*k,size+6*k+6)))
            residual[idx]+=result.residual; tangent[np.ix_(idx,idx)]+=result.tangent
        return owned(residual),owned(tangent)

    def _evaluate(self,total,mu,cmd,origin,check):
        self._guard(origin); state=json.loads(origin.state); total=owned(total,(self.size,))
        qa=owned(state['rotations']); previous=owned(state['total_u']); delta=(total-previous).reshape(-1,6)[:,3:]
        if np.any(np.linalg.norm(delta,axis=1)>=.9*np.pi): raise ValueError('trial chart requires cutback')
        q=owned([rotation_module.rotation_exponential(v)@a for v,a in zip(delta,qa)])
        model,elements,states=self._native_model(state); g=json.loads(self._expanded)['graph']; ids=[r[0] for r in g['nodes']]
        X=np.array([r[1] for r in g['nodes']],dtype=float); coordinates=X+total.reshape(-1,6)[:,:3]
        store=state_module.NonlinearStateStore.from_shell_layouts((),states)
        store.attach_native_rotation_store(state_module.create_model_native_rotation_store(model,states,previous))
        token=store.begin_trial(full_displacement=total,full_coordinates=coordinates)
        force=np.zeros(self.size); H=np.zeros((self.size,self.size))
        try:
            for e in elements:
                mapping=e.get_dof_mapping(model.mesh); indices=[ids.index(n) for n in e.node_ids]
                e._load=(store,owned(np.zeros(3)),owned(np.zeros(3)))
                def guard(e=e):
                    self._guard(origin)
                    if e._load is None or e._load[0] is not store or np.any(e._load[1]) or np.any(e._load[2]):
                        raise ValueError('sandbox native load authority')
                e._owner_guard=guard
                view=store.native_element_rotation_view(token,e.element_id,e.node_ids,e.native_reference_directors(model.mesh))
                if not np.array_equal(view.trial_rotation_matrices,q[indices]): raise ValueError('native subset pose differs')
                f,K,candidate=e.compute_nonlinear_response(model.mesh,e.section,total[mapping],states[e.element_id],1,True,
                    native_rotation_trial=view,native_material_context=store.native_material_context(token,e.element_id))
                if candidate['epoch']!=state['epoch']+1: raise ValueError('material epoch reset')
                store.set_trial_state(token,e.element_id,candidate)
                force[mapping]+=f; H[np.ix_(mapping,mapping)]+=K; check('native:'+str(e.element_id))
            f,K,rows,diagnostics=self._adapter_rows(total,qa,q,epoch=state['epoch']+1,previous=state,check=check)
            force+=f; H+=K
            _,_,_,load=self._targets(cmd); i=ids.index(g['load_node']); force[6*i:6*i+3]-=load
            r,A=self._constraints(total,qa,mu,cmd); r=r.copy(); A=A.copy()
            r[:self.size]+=force; A[:self.size,:self.size]+=H
            check('assembly')
            next_state=dict(schema=SCHEMA,epoch=state['epoch']+1,definition_sha256=sha(json.loads(self._definition)),
                previous_sha256=origin.state_sha256,total_u=total,rotations=q,
                native_rows=[dict(element_id=e.element_id,payload=store.materialize(trial_token=token)[e.element_id]) for e in elements],
                adapter_rows=rows,multipliers=mu)
            return owned(r),owned(A),next_state,(store,token,coordinates),diagnostics
        except BaseException:
            if store.has_active_trial: store.discard_trial(token)
            raise

    @staticmethod
    def _discard(sandbox):
        store,token,_=sandbox
        if store.has_active_trial: store.discard_trial(token)

    def trial(self,total,multipliers,cmd,*,hook=None):
        """Nonpublishing diagnostic; no caller origin or candidate-state import."""
        return self._run(cmd,total,multipliers,hook=hook,solve_graph=False)

    def solve(self,cmd,*,hook=None):
        return self._run(cmd,None,None,hook=hook,solve_graph=True)

    def _run(self,cmd,total,mu,*,hook,solve_graph):
        if type(self) is not MixedGraphOwner or (hook is not None and not callable(hook)): raise ValueError('exact owner/callback')
        lock=self._owned_lock
        if not lock.acquire(False): raise RuntimeError('owner already in use')
        sandbox=None; start=monotonic(); origin=self._published
        try:
            self._guard(origin,full=True,lock=lock)
            history=json.loads(origin.history); state=json.loads(origin.state)
            cmd=command(cmd,history,json.loads(self._definition)['common_motion'])
            total=owned(state['total_u'] if total is None else total,(self.size,)).copy()
            count=len(state['multipliers']); mu=owned(np.zeros(count) if mu is None else mu,(count,)).copy()
            nonce=object(); object.__setattr__(self,'_active',nonce); object.__setattr__(self,'_serial',self._serial+1)
            def check(stage):
                if monotonic()-start>=600: raise TimeoutError('graph child deadline')
                if self._active is not nonce: raise ValueError('foreign/stale graph trial capability')
                self._guard(origin,lock=lock)
                if hook is not None:
                    hook(stage)
                    self._guard(origin,full=True,lock=lock)
                    if self._active is not nonce: raise ValueError('changed graph trial capability')
                progress(stage)
            check('pose')
            if solve_graph and cmd['kind']=='PREPARE_COMMON_MOTION':
                W,t,_,_=self._targets(cmd); X=np.asarray([r[1] for r in json.loads(self._expanded)['graph']['nodes']],dtype=float)
                total.reshape(-1,6)[:,:3]=(X@W.T+t)-X
                for i,qa in enumerate(np.asarray(state['rotations'])):
                    total[6*i+3:6*i+6]+=rotation_log(W@qa.T)
            for iteration in range(25):
                r,A,next_state,sandbox,diagnostics=self._evaluate(total,mu,cmd,origin,check)
                if not solve_graph:
                    result=dict(residual=r,tangent=A,candidate=canonical(next_state),
                        diagnostics=diagnostics,production_qualified=False,state_committed=False)
                    return result
                error=float(np.linalg.norm(r))
                if error<=1e-11:
                    check('prepare')
                    pending={r['element_id']:native_payload(r['payload']) for r in next_state['native_rows']}
                    store,token,coordinates=sandbox
                    store.commit(token,accepted_full_displacement=total,accepted_full_coordinates=coordinates)
                    check('native_committed')
                    if canonical(store.materialize())!=canonical(pending): raise ValueError('native commit changed prepared state')
                    entry=dict(epoch=next_state['epoch'],previous_entry_sha256=None if not history else sha(history[-1]),
                        origin_sha256=origin.state_sha256,command=cmd,accepted_state_sha256=sha(next_state))
                    made=generation(next_state,history+[entry])
                    result=dict(total_u=owned(total),rotations=owned(next_state['rotations']),multipliers=owned(mu),
                        residual=r,iterations=iteration,state_sha256=made.state_sha256,epoch=next_state['epoch'],
                        production_qualified=False,physical_recovery_complete=False)
                    check('before_publish'); self._guard(origin,full=True,lock=lock)
                    # All sandbox work is terminal before the sole publication.
                    # The finally path now only clears an owned nonce/releases
                    # the already-held primitive lock; no family dispatch remains.
                    sandbox=None
                    object.__setattr__(self,'_active',None)
                    object.__setattr__(self,'_published',made)
                    return result
                self._discard(sandbox); sandbox=None
                if iteration==24: raise ValueError('graph iteration limit')
                step=solve(A,-r)
                for cut in range(9):
                    check('line_search'); fraction=.5**cut
                    trial=total+fraction*step[:self.size]; multipliers=mu+fraction*step[self.size:]
                    changed,_,_,sandbox,_=self._evaluate(trial,multipliers,cmd,origin,check)
                    trial_error=float(np.linalg.norm(changed)); self._discard(sandbox); sandbox=None
                    if trial_error<error:
                        total=trial; mu=multipliers; break
                else: raise ValueError('graph line search failed')
            raise ValueError('unreachable graph solve')
        finally:
            if sandbox is not None: self._discard(sandbox)
            object.__setattr__(self,'_active',None)
            lock.release()
