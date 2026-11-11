"""Private generalized native translation continuation with signed load factor.

The full bordered Jacobian uses the analytically condensed parameter column.
No element law, old solver route or force-programme bound is changed.
"""
from copy import deepcopy
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic
import numpy as np
from scipy import sparse
from .linalg import factorize,MatrixClass
from .control import cancellation_safe_point,SolveCancelled
from .nonlinear_state import NonlinearStateStore,create_model_native_rotation_store,native_trial_full_coordinates
from ._ge_beam3_native_generalized_program import model_identity
from ._ge_beam3_native_generalized_loading import DistributedPattern,assemble_distributed_trial,_ACTIVE
from ._ge_beam3_native_generalized_combined_couples import _Run,_RUN,external_at
from ._ge_beam3_native_generalized_parameter import parameter_column
from ._ge_beam3_native_line_loading import LinePattern,nodal_force_vector
from ._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from ._ge_beam3_native_generalized_restart import _model,LoadPoint as DistributedPoint
from ._ge_beam3_native_generalized_combined_restart import LoadPoint
from ._ge_beam3_p5_seeded.core import sha
from ._native_reference_modal import _owned
from ._ge_beam3_native_history_profile import require as require_history_profile,binding as history_binding

POLICY='GE_BEAM3_NATIVE_GENERALIZED_TRANSLATION_CONTINUATION_V1'
EMPTY=DistributedPattern(LinePattern(()),())
_PATH=ContextVar('ge_beam3_native_translation',default=None)


def scalar(value):
    if type(value) is not float or not np.isfinite(value):raise ValueError('finite binary64 path scalar required')
    return value


@dataclass(frozen=True)
class TranslationProgram:
    targets: tuple
    control_node: int
    control_component: str
    distributed: DistributedPattern
    nodal_moments: object=None
    max_iterations: int=24
    max_backtracks: int=8
    history_profile: object=None

    def require(self,model):
        require_history_profile(self.history_profile)
        if type(self) is not TranslationProgram or type(self.targets) is not tuple or not 1<=len(self.targets)<=64:
            raise ValueError('bounded explicit translation targets required')
        for target in self.targets:scalar(target)
        if type(self.control_node) is not int or self.control_node not in model.mesh.nodes or type(self.control_component) is not str or self.control_component not in ('ux','uy','uz'):
            raise ValueError('explicit translation control node/component required')
        if type(self.distributed) is not DistributedPattern:raise ValueError('exact translation distributed pattern')
        self.distributed.require(model.mesh)
        if self.nodal_moments is not None:
            if type(self.nodal_moments) is not SpatialNodalMoments:raise ValueError('exact translation moment pattern')
            SpatialNodalMoments(self.nodal_moments.rows)
            if any(row[0] not in model.mesh.nodes for row in self.nodal_moments.rows):raise ValueError('translation moment node absent')
        if not self.distributed.line.rows and not self.distributed.couples and self.nodal_moments is None:
            raise ValueError('nonzero proportional translation load required')
        if type(self.max_iterations) is not int or not 1<=self.max_iterations<=24 or type(self.max_backtracks) is not int or not 0<=self.max_backtracks<=8:
            raise ValueError('bounded translation Newton controls required')

    def descriptor(self):
        return dict(policy=POLICY,targets=self.targets,control_node=self.control_node,control_component=self.control_component,
            distributed=self.distributed,nodal_moments=self.nodal_moments,max_iterations=self.max_iterations,
            max_backtracks=self.max_backtracks,tolerance=1e-12,production_qualified=False,conservative_spectral_authority=False,
            **history_binding(self.history_profile))


def capture(model,program):
    if type(program) is not TranslationProgram:raise ValueError('exact native translation programme required')
    program.require(model);elements,n,free,fixed,identity=_model(model)
    control=model.mesh.dof_manager.get_node_dofs(program.control_node)[('ux','uy','uz').index(program.control_component)]
    free=np.array(sorted(free),dtype=int)
    where=np.flatnonzero(free==control)
    if len(where)!=1:raise ValueError('translation control must be free')
    return elements,n,free,identity,int(control),int(where[0])


def effective(program,parameter):
    scalar(parameter)
    def scaled(rows):
        result=[]
        for eid,*values in rows:
            with np.errstate(over='ignore',invalid='ignore'):v=parameter*np.array(values)
            if not np.isfinite(v).all():raise ValueError('translation effective load range')
            v[v==0.]=0.  # Match the existing effective-load codec's +0 origin.
            if np.any(v):result.append((eid,*map(float,v)))
        return tuple(result)
    pattern=DistributedPattern(LinePattern(scaled(program.distributed.line.rows)),scaled(program.distributed.couples))
    rows=scaled(() if program.nodal_moments is None else program.nodal_moments.rows)
    return pattern,SpatialNodalMoments(rows) if rows else None


def load_point(program,parameter,*,genesis=False):
    pattern,moments=effective(program,parameter)
    # Inner codec validates effective loads and complete history. The outer
    # continuation envelope, not this unit coordinate, owns signed lambda.
    return LoadPoint(DistributedPoint(0. if genesis else 1.,EMPTY,pattern),None,moments)


def physical(model,states,moments):
    net=np.zeros(model.mesh.dof_manager.total_dofs)
    for eid,e in sorted(model.mesh.elements.items()):np.add.at(net,e.get_dof_mapping(model.mesh),states[eid]['response'].residual)
    for node,*moment in (() if moments is None else moments.rows):net[list(model.mesh.dof_manager.get_node_dofs(node)[3:])]-=moment
    if not np.isfinite(net).all() or not np.isfinite(np.linalg.norm(net)):raise ValueError('translation physical residual range')
    return net


def assemble(model,store,u,program,parameter):
    if _RUN.get() is not None or _ACTIVE.get() is not None:raise ValueError('unnested translation assembly required')
    pattern,moments=effective(program,parameter)
    force,matrix,states,external=assemble_distributed_trial(model,u,store,pattern)
    token=_RUN.set(_Run(model,None,moments))
    try:nodal,tangent=external_at(model,store,u,0.,tangent=True)
    finally:_RUN.reset(token)
    scale=float(np.linalg.norm(external+nodal))
    if not np.isfinite(scale):raise ValueError('translation external norm range')
    return force-external-nodal,matrix-tangent,states,physical(model,states,moments),max(1.,scale)


def bordered(matrix,column,free,local_control):
    row=np.zeros(len(free));row[local_control]=1.
    return sparse.bmat([[matrix[free][:,free],sparse.csr_matrix(column[free,None])],
        [sparse.csr_matrix(row[None,:]),sparse.csr_matrix((1,1))]],format='csr')


@dataclass(frozen=True)
class TranslationResult:
    status: str
    completed_targets: int
    parameter: float
    displacements: np.ndarray
    physical_imbalance: np.ndarray
    checkpoint: bytes
    failure: object
    events: tuple
    production_qualified: bool=False


def _solve_translation(model,program,*,checkpoint=None,expected_sha256=None,stop_after=None,progress=None,cancellation_token=None):
    from ._ge_beam3_native_translation_restart import encode_checkpoint,decode_checkpoint
    started=monotonic();elements,n,free,identity,control,local_control=capture(model,program)
    if _RUN.get() is not None or _ACTIVE.get() is not None:raise ValueError('unnested native translation programme required')
    if progress is not None and not callable(progress):raise ValueError('callable translation observer required')
    end=len(program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0<=end<=len(program.targets):raise ValueError('bounded translation stop cursor')
    fingerprint=sha(program.descriptor());events=[]
    def safe(stage):
        cancellation_safe_point(cancellation_token,stage)
        program.require(model)
        if model_identity(model)!=identity or sha(program.descriptor())!=fingerprint:raise ValueError('translation programme authority changed')
        if monotonic()-started>600.:raise RuntimeError('translation programme deadline')
    def observe(stage,index,iteration=0,**kw):
        safe(stage);event=dict(stage=stage,target=index,iteration=iteration,**kw);events.append(event)
        if progress is not None:progress(deepcopy(event))
        safe(stage)
    if checkpoint is None:
        if expected_sha256 is not None:raise ValueError('translation hash without checkpoint')
        total=np.zeros(n);states={i:e.init_model_bound_nonlinear_state(model.mesh,e.section,1) for i,e in elements}
        chain=(dict(load_point=load_point(program,0.,genesis=True),displacements=total.copy(),states=states),);records=()
        capsule=encode_checkpoint(model,program,chain,records)
    else:
        chain,records=decode_checkpoint(model,program,checkpoint,expected_sha256=expected_sha256)
        total=chain[-1]['displacements'].copy();states=chain[-1]['states'];capsule=checkpoint
    cursor=len(records);parameter=0. if not records else records[-1]['parameter']
    if cursor>end:raise ValueError('translation cannot rewind accepted cursor')
    reaction=physical(model,states,effective(program,parameter)[1])
    store=NonlinearStateStore.from_shell_layouts((),states)
    store.attach_native_rotation_store(create_model_native_rotation_store(model,states,total))
    status='completed' if end==len(program.targets) else 'paused';failure=None
    def evaluate(u,p,index,it):
        observe('before_assembly',index,it)
        result=assemble(model,store,u,program,p);safe('after_assembly');return result
    def merit(force,reaction,scale,u,target):
        value=max(float(np.linalg.norm(force[free]))/scale,float(np.linalg.norm(reaction[free]))/scale,
            abs(float(u[control])-target)/max(1.,abs(target)))
        if not np.isfinite(value):raise ValueError('translation merit range')
        return value
    try:
        observe('initialized',cursor)
        for index in range(cursor,end):
            target=program.targets[index];candidate=total.copy();candidate[control]=target;p=parameter;accepted=False
            for iteration in range(program.max_iterations+1):
                force,matrix,trial,imbalance,scale=evaluate(candidate,p,index+1,iteration)
                norm=merit(force,imbalance,scale,candidate,target)
                observe('iteration',index+1,iteration,parameter=p,merit=norm)
                if norm<=1e-12:
                    next_records=records+(dict(index=index+1,target=target,parameter=p,iterations=iteration),)
                    next_chain=chain+(dict(load_point=load_point(program,p),displacements=candidate.copy(),states=deepcopy(dict(trial))),)
                    staged=encode_checkpoint(model,program,next_chain,next_records)
                    observe('before_commit',index+1,iteration)
                    coordinates=native_trial_full_coordinates(store,model,candidate)
                    store.commit(store.active_trial_token(),accepted_full_displacement=candidate,accepted_full_coordinates=coordinates)
                    capsule,chain,records=staged,next_chain,next_records
                    total,reaction,parameter,cursor=candidate.copy(),imbalance.copy(),p,index+1;accepted=True
                    observe('committed',cursor,iteration);break
                if iteration==program.max_iterations:break
                column=parameter_column(model,store,trial,program.distributed,nodal_moments=program.nodal_moments)['column']
                safe('before_factorization')
                handle=factorize(bordered(matrix,column,free,local_control),MatrixClass.GENERAL,
                    signature=f'native_translation:{fingerprint}:{index}:{iteration}')
                delta=np.asarray(handle.solve(-np.r_[force[free],candidate[control]-target]),dtype=float).reshape(-1)
                if delta.shape!=(len(free)+1,) or not np.isfinite(delta).all():raise ValueError('translation Newton increment range')
                for cut in range(program.max_backtracks+1):
                    proposed=candidate.copy();proposed[free]+=(.5**cut)*delta[:-1];proposed[control]=target
                    next_p=float(p+(.5**cut)*delta[-1]);changed,_,_,physical_changed,s=evaluate(proposed,next_p,index+1,iteration)
                    if merit(changed,physical_changed,s,proposed,target)<norm:candidate,p=proposed,next_p;break
                else:raise RuntimeError('translation line search exhausted')
            if not accepted:raise RuntimeError('translation iteration bound exhausted')
    except SolveCancelled as error:status,failure='cancelled',str(error)
    except Exception as error:status,failure='failed',type(error).__name__+': '+str(error)
    finally:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())
    return TranslationResult(status,cursor,parameter,_owned(total),_owned(reaction),capsule,failure,tuple(events))


def solve_translation(model,program,**kwargs):
    if _PATH.get() is not None:raise ValueError('nested native translation programme forbidden')
    token=_PATH.set(object())
    try:return _solve_translation(model,program,**kwargs)
    finally:_PATH.reset(token)
