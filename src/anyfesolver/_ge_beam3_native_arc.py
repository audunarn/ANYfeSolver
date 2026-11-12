"""Private objective generalized native frame-chord continuation.

Predictors live in the spatial left-trivialized tangent coordinates at each
accepted origin. Actual Exp increments, not total rotation vectors interpreted
as orientations, enter the objective chord constraint. No branch switching.
"""
from copy import deepcopy
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from time import monotonic
import numpy as np
from scipy import sparse
from .linalg import factorize,MatrixClass
from .control import cancellation_safe_point,SolveCancelled
from .nonlinear_state import NonlinearStateStore,create_model_native_rotation_store,native_trial_full_coordinates
from ._ge_beam3_native_translation import TranslationProgram,scalar,assemble,effective,load_point,physical,_PATH
from ._ge_beam3_native_generalized_program import model_identity,_PROGRAM
from ._ge_beam3_native_generalized_combined_couples import _RUN
from ._ge_beam3_native_generalized_loading import _ACTIVE
from ._ge_beam3_native_generalized_restart import _model
from ._ge_beam3_native_generalized_parameter import parameter_column
from ._ge_beam3_arc_geometry import constraint,POLICY as GEOMETRY
from ._ge_beam3_p5_seeded.core import sha
from ._native_reference_modal import _owned
from ._ge_beam3_native_history_profile import require as require_history_profile,binding as history_binding

POLICY='GE_BEAM3_NATIVE_GENERALIZED_FRAME_CHORD_ARC_V1'
ORIENTATION='SPATIAL_LEFT_TRIVIALIZED_PREDICTOR_DOT_V1'
_ARC=ContextVar('ge_beam3_native_generalized_arc',default=None)


def inverse_square(value):
    scalar(value)
    if value<=0.:raise ValueError('positive explicit arc scale required')
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        inverse=np.float64(1.)/value;result=float(inverse*inverse)
    if not np.isfinite(result) or result<=0.:raise ValueError('arc metric scale range')
    return result


@dataclass(frozen=True)
class ArcProgram:
    steps: tuple
    length_scale: float
    distributed: object
    nodal_moments: object=None
    parameter_scale: float=1.
    initial_sign: float=1.
    max_iterations: int=24
    max_backtracks: int=8
    history_profile: object=None
    source: object=None

    def require(self,model):
        require_history_profile(self.history_profile)
        if type(self) is not ArcProgram or type(self.steps) is not tuple or not 1<=len(self.steps)<=64:
            raise ValueError('bounded explicit native arc steps required')
        for step in self.steps:
            scalar(step)
            if not 0.<step<=.25:raise ValueError('positive bounded native arc step')
        inverse_square(self.length_scale);inverse_square(self.parameter_scale);scalar(self.initial_sign)
        if self.initial_sign not in (-1.,1.):raise ValueError('explicit native arc orientation sign')
        if not model.mesh.nodes:raise ValueError('native arc model is empty')
        # Reuse load/control field validation only; no dummy target is run.
        TranslationProgram((0.,),min(model.mesh.nodes),'ux',self.distributed,self.nodal_moments,
            self.max_iterations,self.max_backtracks).require(model)
        if self.source is not None:
            from ._ge_beam3_native_arc_source import require_compatible
            require_compatible(model,self)

    def descriptor(self):
        return dict(policy=POLICY,geometry=GEOMETRY,orientation=ORIENTATION,steps=self.steps,length_scale=self.length_scale,
            parameter_scale=self.parameter_scale,initial_sign=self.initial_sign,distributed=self.distributed,
            nodal_moments=self.nodal_moments,max_iterations=self.max_iterations,max_backtracks=self.max_backtracks,
            tolerance=1e-12,production_qualified=False,conservative_spectral_authority=False,**history_binding(self.history_profile),
            **({} if self.source is None else {'source':self.source.descriptor()}))


def capture(model,program):
    if type(program) is not ArcProgram:raise ValueError('exact native arc programme required')
    program.require(model);elements,n,free,_,identity=_model(model)
    maps=np.array([model.mesh.dof_manager.get_node_dofs(i) for i in sorted(model.mesh.nodes)],dtype=int)
    if not 1<=len(maps)<=42:raise ValueError('bounded frame-chord node count')
    metric=np.empty(n+1);metric[maps[:,:3]]=inverse_square(program.length_scale)/len(maps)
    metric[maps[:,3:]]=1./len(maps);metric[-1]=inverse_square(program.parameter_scale)
    if not np.isfinite(metric).all() or np.any(metric<=0.):raise ValueError('finite positive native arc metric')
    initial=np.zeros(n+1);initial[-1]=program.initial_sign*program.parameter_scale
    return elements,n,np.array(sorted(free),dtype=int),identity,maps,_owned(metric),_owned(initial)


def bordered(matrix,column,free,row):
    return sparse.bmat([[matrix[free][:,free],sparse.csr_matrix(column[free,None])],
        [sparse.csr_matrix(row[free][None,:]),sparse.csr_matrix([[row[-1]]])]],format='csr')


def direction(matrix,column,free,previous,metric):
    row=metric*previous;rhs=np.zeros(len(free)+1);rhs[-1]=1.
    handle=factorize(bordered(matrix,column,free,row),MatrixClass.GENERAL)
    reduced=np.asarray(handle.solve(rhs),dtype=float).reshape(-1)
    if reduced.shape!=(len(free)+1,) or not np.isfinite(reduced).all():raise ValueError('native arc predictor range')
    value=np.zeros(len(metric));value[free]=reduced[:-1];value[-1]=reduced[-1]
    norm=float(np.sum(metric*value*value))
    if not np.isfinite(norm) or norm<=0.:raise ValueError('unresolved native arc predictor')
    value/=sqrt(norm)
    if not np.isfinite(value).all() or float(row@value)<=0.:raise ValueError('unresolved native arc orientation')
    return _owned(value)


def arc_row(total,origin,tangent,metric,parameter,old_parameter,step,maps):
    gap,local=constraint((total-origin)[maps],tangent[:-1][maps],metric[:-1][maps],
        float(parameter-old_parameter),float(tangent[-1]),float(metric[-1]),step)
    row=np.empty(len(metric));row[maps.ravel()]=local[:-1];row[-1]=local[-1]
    return gap,row


def make_store(model,states,total):
    store=NonlinearStateStore.from_shell_layouts((),states)
    store.attach_native_rotation_store(create_model_native_rotation_store(model,states,total))
    return store


def clone_model(model):
    """Replay must not replace the original elements' issued validators."""
    from .fe_core import FEModel
    from ._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
    identity=model_identity(model);made=FEModel('private-native-arc-replay')
    for i,node in sorted(model.mesh.nodes.items()):made.add_node(i,*node.coords())
    for i,e in sorted(model.mesh.elements.items()):
        item=NativeGeneralizedStaticElement(i,e.node_ids,e.operator.reference,e.section,order=e.operator.order)
        made.add_element(i,item);made.materials[item.material_name]=item.section
    for bc in model.boundary_conditions:made.add_boundary_condition(deepcopy(bc))
    if model_identity(made)!=identity:raise ValueError('native arc replay model identity mismatch')
    return made


def replay_direction(model,program,snapshot,parameter,previous):
    made=clone_model(model);_,_,free,_,_,metric,_=capture(made,program)
    store=make_store(made,snapshot['states'],snapshot['displacements'])
    try:
        f,k,trial,r,scale=assemble(made,store,snapshot['displacements'],program,parameter)
        if max(np.linalg.norm(f[free]),np.linalg.norm(r[free]))/scale>1e-12:raise ValueError('native arc replay origin not equilibrated')
        column=parameter_column(made,store,trial,program.distributed,nodal_moments=program.nodal_moments)['column']
        return direction(k,column,free,previous,metric)
    finally:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())


@dataclass(frozen=True)
class ArcResult:
    status: str
    completed_steps: int
    parameter: float
    displacements: np.ndarray
    physical_imbalance: np.ndarray
    checkpoint: bytes
    failure: object
    events: tuple
    production_qualified: bool=False


class ArcConvergenceFailure(RuntimeError):
    """Typed local convergence exhaustion; never a state/authority failure."""


def attempt_step(model,program,store,total,parameter,previous,index,step,maps,metric,free,evaluate,observe):
    """One uncommitted native predictor/corrector; caller owns commit/rollback."""
    def residual_norm(f,r,scale):return float(max(np.linalg.norm(f[free]),np.linalg.norm(r[free]))/scale)
    f,k,trial,r,scale=evaluate(total,parameter,index,0)
    if residual_norm(f,r,scale)>1e-12:raise ValueError('native arc origin not equilibrated')
    column=parameter_column(model,store,trial,program.distributed,nodal_moments=program.nodal_moments)['column']
    observe('before_predictor',index);tangent=direction(k,column,free,previous,metric)
    candidate=total+step*tangent[:-1];p=float(parameter+step*tangent[-1])
    for iteration in range(program.max_iterations+1):
        force,matrix,trial,imbalance,scale=evaluate(candidate,p,index,iteration)
        gap,row=arc_row(candidate,total,tangent,metric,p,parameter,step,maps)
        norm=max(residual_norm(force,imbalance,scale),abs(gap))
        if not np.isfinite(norm):raise ValueError('native arc merit range')
        observe('iteration',index,iteration,parameter=p,merit=norm,arc_residual=gap)
        if norm<=1e-12:return candidate,p,trial,imbalance,tangent,iteration,float(abs(gap))
        if iteration==program.max_iterations:break
        column=parameter_column(model,store,trial,program.distributed,nodal_moments=program.nodal_moments)['column']
        observe('before_corrector',index,iteration)
        handle=factorize(bordered(matrix,column,free,row),MatrixClass.GENERAL)
        delta=np.asarray(handle.solve(-np.r_[force[free],gap]),dtype=float).reshape(-1)
        if delta.shape!=(len(free)+1,) or not np.isfinite(delta).all():raise ValueError('native arc corrector range')
        for cut in range(program.max_backtracks+1):
            proposed=candidate.copy();proposed[free]+=(.5**cut)*delta[:-1]
            next_p=float(p+(.5**cut)*delta[-1]);changed,_,_,changed_r,s=evaluate(proposed,next_p,index,iteration)
            changed_gap,_=arc_row(proposed,total,tangent,metric,next_p,parameter,step,maps)
            if max(residual_norm(changed,changed_r,s),abs(changed_gap))<norm:candidate,p=proposed,next_p;break
        else:raise ArcConvergenceFailure('native arc line search exhausted')
    raise ArcConvergenceFailure('native arc iteration bound exhausted')


def _solve_arc(model,program,*,checkpoint=None,expected_sha256=None,stop_after=None,progress=None,cancellation_token=None):
    from ._ge_beam3_native_arc_restart import encode_checkpoint,decode_checkpoint
    started=monotonic();elements,n,free,identity,maps,metric,initial=capture(model,program)
    if progress is not None and not callable(progress):raise ValueError('callable native arc observer required')
    end=len(program.steps) if stop_after is None else stop_after
    if type(end) is not int or not 0<=end<=len(program.steps):raise ValueError('bounded native arc stop cursor')
    fingerprint=sha(program.descriptor());events=[]
    def safe(stage):
        cancellation_safe_point(cancellation_token,stage);program.require(model)
        if model_identity(model)!=identity or sha(program.descriptor())!=fingerprint:raise ValueError('native arc programme authority changed')
        if monotonic()-started>600.:raise RuntimeError('native arc programme deadline')
    def observe(stage,index,iteration=0,**kw):
        safe(stage);event=dict(stage=stage,step=index,iteration=iteration,**kw);events.append(event)
        if progress is not None:progress(deepcopy(event))
        safe(stage)
    source_chain=None;source_parameter=0.
    if program.source is not None:
        if checkpoint is not None and (type(checkpoint) is not bytes or type(expected_sha256) is not str or sha256(checkpoint).hexdigest()!=expected_sha256):
            raise ValueError('native arc external checkpoint SHA-256 mismatch')
        from ._ge_beam3_native_arc_source import load_source,secant_orientation
        source_chain,source_parameter,previous_parameter=load_source(model,program.source)
        if len(source_chain)+len(program.steps)>65:raise ValueError('complete source plus arc history exceeds snapshot bound')
        initial=secant_orientation(source_chain,source_parameter,previous_parameter,metric,program.source.forward_sign)
    if checkpoint is None:
        if expected_sha256 is not None:raise ValueError('native arc hash without checkpoint')
        if source_chain is None:
            total=np.zeros(n);states={i:e.init_model_bound_nonlinear_state(model.mesh,e.section,1) for i,e in elements}
            chain=(dict(load_point=load_point(program,0.,genesis=True),displacements=total.copy(),states=states),)
        else:
            chain=source_chain;total=chain[-1]['displacements'].copy();states=chain[-1]['states']
        records=()
        capsule=encode_checkpoint(model,program,chain,records)
    else:
        chain,records=decode_checkpoint(model,program,checkpoint,expected_sha256=expected_sha256)
        total=chain[-1]['displacements'].copy();states=chain[-1]['states'];capsule=checkpoint
    cursor=len(records);parameter=source_parameter if not records else records[-1]['parameter']
    previous=initial.copy() if not records else records[-1]['direction'].copy()
    if cursor>end:raise ValueError('native arc cannot rewind accepted cursor')
    reaction=physical(model,states,effective(program,parameter)[1]);store=make_store(model,states,total)
    status='completed' if end==len(program.steps) else 'paused';failure=None
    def evaluate(u,p,index,it):
        observe('before_assembly',index,it);result=assemble(model,store,u,program,p);safe('after_assembly');return result
    try:
        observe('initialized',cursor)
        for index in range(cursor,end):
            step=program.steps[index]
            candidate,p,trial,imbalance,tangent,iteration,residual=attempt_step(
                model,program,store,total,parameter,previous,index+1,step,maps,metric,free,evaluate,observe)
            next_records=records+(dict(index=index+1,step_size=step,parameter=p,iterations=iteration,
                direction=tangent.copy(),arc_residual=residual),)
            next_chain=chain+(dict(load_point=load_point(program,p),displacements=candidate.copy(),states=deepcopy(dict(trial))),)
            staged=encode_checkpoint(model,program,next_chain,next_records)
            observe('before_commit',index+1,iteration)
            coordinates=native_trial_full_coordinates(store,model,candidate)
            store.commit(store.active_trial_token(),accepted_full_displacement=candidate,accepted_full_coordinates=coordinates)
            capsule,chain,records=staged,next_chain,next_records
            total,reaction,parameter,cursor=candidate.copy(),imbalance.copy(),p,index+1;previous=tangent.copy()
            observe('committed',cursor,iteration)
    except SolveCancelled as error:status,failure='cancelled',str(error)
    except Exception as error:
        name='RuntimeError' if type(error) is ArcConvergenceFailure else type(error).__name__
        status,failure='failed',name+': '+str(error)
    finally:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())
    return ArcResult(status,cursor,parameter,_owned(total),_owned(reaction),capsule,failure,tuple(events))


def solve_arc(model,program,**kwargs):
    if any(value is not None for value in (_ARC.get(),_PATH.get(),_PROGRAM.get(),_RUN.get(),_ACTIVE.get())):
        raise ValueError('nested native arc programme forbidden')
    token=_ARC.set(object())
    try:return _solve_arc(model,program,**kwargs)
    finally:_ARC.reset(token)
