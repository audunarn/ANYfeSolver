"""Uncommitted local tangent-plane continuation of coupled BL5/J2 equilibria.

Experimental fixed mesh, explicit frozen physical weights and affine
translational dead loads. The public controller owns commit; no capacity claim.
"""
from dataclasses import dataclass
import numpy as np
from .control import cancellation_safe_point
from ._ge_beam3_pose_joint import _exp_terms
from ._stiffened_shell_rotation_charts import physical_frames
from ._stiffened_shell_arc_metric import (PhysicalChordMetric, NativeDirectorChordMetric,
    ChordDomainError, _owned, require_metric_owner, require_director_chart)
from ._stiffened_shell_displacement_corrector import (
    bordered_correction, bordered_path_correction, DisplacementCorrectorFailure)


@dataclass(frozen=True)
class ArcTrial:
    state: object
    amplitude: float
    evaluation: tuple
    accepted_generation: int
    predecessor_state: object
    shell_predecessor: object
    beam_predecessor: object
    radius: float
    path_error: float
    evaluations: int
    linear_solves: int
    rejected_trials: int
    event_cell_evaluations: int
    base_load: np.ndarray
    load_direction: np.ndarray
    normal_translations: np.ndarray
    normal_spins: np.ndarray
    metric: PhysicalChordMetric | NativeDirectorChordMetric
    predictor_dof: int
    orientation: int
    amplitude_origin: float


class ArcTrialFailure(DisplacementCorrectorFailure):
    """Failed uncommitted candidate and work, retained for a caller's cutback."""
    def __init__(self,message,trial):
        super().__init__(message)
        self.trial=trial


@dataclass(frozen=True)
class ArcAttempt:
    radius: float
    accepted: bool
    evaluations: int
    linear_solves: int
    rejected_trials: int
    event_cell_evaluations: int
    failure: str | None


@dataclass(frozen=True)
class ArcSelection:
    """One selected, uncommitted step; not completion of a requested path."""
    requested_radius: float
    trial: ArcTrial
    attempts: tuple
    max_cutbacks: int
    max_attempts: int
    max_iterations: int
    max_backtracks: int


class ArcSelectionFailure(RuntimeError):
    def __init__(self,message,attempts):
        super().__init__(message)
        self.attempts=tuple(attempts)


def select_trial(driver, *, radius, max_cutbacks=4, max_attempts=5,
                 attempt_records=None, **kwargs):
    """Bounded cutback from one unchanged accepted predecessor.

    Retry only the explicitly typed Newton/backtrack failures emitted below.
    Cancellation, stale ownership, programming errors and unclassified numerical
    failures propagate. The caller still owns atomic publication and path control.
    """
    if (type(max_cutbacks) is not int or not 0<=max_cutbacks<=8
            or type(max_attempts) is not int or not 1<=max_attempts<=64
            or not np.isfinite(radius) or radius<=0.):
        raise ValueError("bounded arc selection controls required")
    owner=driver.owner
    with owner._commit_lock:
        generation=owner._generation;parent=driver._state
        shell=driver._shell_origins;beam=driver._beam_origins
    kwargs=dict(kwargs)
    kwargs['load_direction']=np.array(kwargs['load_direction'],dtype=float,copy=True)
    if kwargs.get('load_base') is not None:
        kwargs['load_base']=_owned(kwargs['load_base'])
    attempts=[]
    def current():
        if (owner._finite_driver is not driver or owner._generation!=generation
                or driver._state is not parent or driver._shell_origins is not shell
                or driver._beam_origins is not beam):
            raise ValueError("stale arc selection predecessor")
        cancellation_safe_point(kwargs.get('cancellation_token'),"stiffened-shell.arc.selection")
    for depth in range(min(max_attempts,max_cutbacks+1)):
        current()
        actual=float(radius)*(.5**depth)
        if not np.isfinite(actual*actual) or actual*actual<=0.:
            raise ArcSelectionFailure("arc radius no longer representable",attempts)
        failure=None
        try:
            trial=correct(driver,radius=actual,**kwargs)
        except ArcTrialFailure as error:
            trial=error.trial;failure=str(error)
        with owner._commit_lock:current()
        row=ArcAttempt(actual,failure is None,trial.evaluations,trial.linear_solves,
            trial.rejected_trials,trial.event_cell_evaluations,failure)
        attempts.append(row)
        if attempt_records is not None:attempt_records.append(row)
        if failure is None:
            with owner._commit_lock:
                current()
                return ArcSelection(float(radius),trial,tuple(attempts),max_cutbacks,max_attempts,
                    kwargs.get('max_iterations',20),kwargs.get('max_backtracks',12))
    raise ArcSelectionFailure("arc cutback/attempt limit",attempts)


def path_row(driver,state,parent,metric,normal=None):
    """Squared physical chord and row in the driver's actual update columns."""
    owner=driver.owner
    nodal=owner._shell_count+6*len(owner._beam_ids)
    if type(metric) is NativeDirectorChordMetric:
        require_director_chart(parent.q[:owner._shell_count])
        require_director_chart(state.q[:owner._shell_count])
    reference=np.concatenate((physical_frames(parent.q[:owner._shell_count],parent.shell_anchors),parent.frames))
    frames=np.concatenate((physical_frames(state.q[:owner._shell_count],state.shell_anchors),state.frames))
    displacement=(state.q[:nodal]-parent.q[:nodal]).reshape(-1,6)[:,:3]
    value,translations,spins=(metric.evaluate(displacement,frames,reference) if normal is None
        else metric.plane(displacement,frames,reference,*normal))
    gradient=np.zeros(len(state.q)+len(state.multipliers))
    rows=gradient[:nodal].reshape(-1,6)
    rows[:,:3]=translations;rows[:,3:]=spins
    for i,chart in enumerate(state.q[:owner._shell_count].reshape(-1,6)):
        rows[i,3:]=_exp_terms(chart[3:])[1].T@spins[i]
    return value,gradient


def correct(driver, *, load_direction, metric, radius, predictor_dof,
            orientation, max_iterations=20, max_backtracks=12,
            cancellation_token=None, iteration_records=None,
            load_base=None, amplitude_origin=0.):
    """Return a candidate; all trial material updates share one predecessor."""
    from ._stiffened_shell_ge_finite_experimental import _RotationStepTooLarge
    from ._stiffened_shell_bl5_local_experimental import BL5TrialDomainError
    from ._stiffened_shell_branch_event_trial import trial_fraction
    owner=driver.owner
    n=owner._K.shape[0]; nodal=owner._shell_count+6*len(owner._beam_ids)
    from ._stiffened_shell_affine_endpoint import is_native_point_material,is_native_material_bl5_arc
    native=is_native_point_material(owner) or is_native_material_bl5_arc(owner)
    require_metric_owner(owner,metric)
    if type(metric) is NativeDirectorChordMetric:
        require_director_chart(driver._state.q[:owner._shell_count])
    if native:
        from ._stiffened_shell_arc_checkpoint import require_native_seed
        require_native_seed(owner,driver)
    size=n+len(driver._state.multipliers)
    if (len(metric.weights)!=nodal//6
            or not (owner._bl5_shell_material or getattr(owner,"_curved_reference",False) or native) or driver._accepted is None
            or driver._external is not None):
        raise ValueError("accepted admitted arc owner with fixed physical metric required")
    if (not np.isfinite(radius) or radius<=0. or type(orientation) is not int or orientation not in (-1,1)
            or type(predictor_dof) is not int or predictor_dof not in owner._free
            or not 0<=predictor_dof<nodal or predictor_dof%6>=3
            or type(max_iterations) is not int or not 1<=max_iterations<=24
            or type(max_backtracks) is not int or not 0<=max_backtracks<=12):
        raise ValueError("bounded radius, orientation and physical predictor required")
    radius2=float(radius)*float(radius)
    if not np.isfinite(radius2) or radius2<=0.:
        raise ValueError("representable squared radius required")
    pattern=np.array(load_direction,dtype=float,copy=True)
    allowed=np.zeros(n,dtype=bool)
    for start in range(0,nodal,6):allowed[start:start+3]=True
    if (pattern.shape!=(n,) or not np.isfinite(pattern).all()
            or np.any(pattern[~allowed]) or not np.any(pattern[owner._free])):
        raise ValueError("finite free translational dead-load direction required")
    with owner._commit_lock:
        if owner._finite_driver is not driver:
            raise ValueError("stale continuation driver")
        generation=owner._generation;parent=driver._state
        shell=driver._shell_origins;beam=driver._beam_origins
        accepted_load=getattr(driver,"_arc_applied_load",None)
        if accepted_load is None:accepted_load=getattr(driver,"_displacement_applied_load",None)
        if accepted_load is None:
            accepted_load=driver._applied_load if driver._programme is not None else driver._load_factor*driver._pattern
        base=_owned(accepted_load if load_base is None else load_base)
        if (not isinstance(amplitude_origin,(float,int,np.floating,np.integer))
                or isinstance(amplitude_origin,(bool,np.bool_)) or not np.isfinite(amplitude_origin)
                or base.shape!=(n,) or not np.isfinite(base).all() or np.any(base[~allowed])
                or not np.array_equal(base+float(amplitude_origin)*pattern,accepted_load)):
            raise ValueError("affine arc load must start at the accepted endpoint")
    pattern=_owned(pattern)
    free=np.r_[owner._free,np.arange(n,size)]
    direction=np.r_[pattern,np.zeros(size-n)]
    energy=float(np.max((abs(owner._K)@owner._coordinate_scale)*owner._coordinate_scale))
    if not np.isfinite(energy) or energy<=0.:raise ValueError("positive reference work scale required")
    weights=np.r_[owner._coordinate_scale,energy/owner._constraint_scale]/energy
    evaluations=solves=rejected=0; events=[]

    def current():
        if (owner._finite_driver is not driver or owner._generation!=generation or driver._state is not parent
                or driver._shell_origins is not shell or driver._beam_origins is not beam):
            raise ValueError("stale continuation predecessor")
        cancellation_safe_point(cancellation_token,"stiffened-shell.arc.trial")

    def evaluate(state,amplitude):
        nonlocal evaluations
        current();evaluations+=1
        return driver._assemble(state,0.,shell,beam,applied_load=base+amplitude*pattern)

    def path(state):
        value,gradient=path_row(driver,state,parent,metric,normal)
        return value-radius,gradient

    def merit(state,evaluation):
        gap,_=path(state)
        return float(np.hypot(np.linalg.norm(evaluation[0][free]*weights[free]),gap))

    state=driver._increment_origin(parent);amplitude=float(amplitude_origin);evaluation=evaluate(state,amplitude)
    # Homogeneous equilibrium tangent, oriented by a physical free translation.
    # Retained entries follow that tangent but do not determine its norm.
    step,delta=bordered_correction(evaluation[1],np.zeros(size),direction,
        predictor_dof,-orientation*metric.length,free);solves+=1
    translations=step[:nodal].reshape(-1,6)[:,:3].copy()
    spins=step[:nodal].reshape(-1,6)[:,3:].copy()
    for i,chart in enumerate(state.q[:owner._shell_count].reshape(-1,6)):
        spins[i]=_exp_terms(chart[3:])[1]@spins[i]
    if type(metric) is NativeDirectorChordMetric:
        reference=np.concatenate((physical_frames(parent.q[:owner._shell_count],parent.shell_anchors),parent.frames))
        norm=float(np.sqrt(metric.tangent_norm_squared(reference,translations,spins)))
    else:
        norm=float(np.sqrt(np.sum(metric.weights*(np.sum((translations/metric.length)**2,axis=1)
            +np.sum(spins*spins,axis=1)))))
    if not np.isfinite(norm) or norm<=0.:
        raise DisplacementCorrectorFailure("unmeasured physical continuation tangent")
    normal=(translations/norm,spins/norm)
    step*=radius/norm;delta*=radius/norm
    if (not np.isfinite(normal[0]).all() or not np.isfinite(normal[1]).all()
            or not np.isfinite(step).all() or not np.isfinite(delta)):
        raise DisplacementCorrectorFailure("continuation tangent normalization not representable")
    def trial_snapshot():
        gap,_=path(state)
        return ArcTrial(state,amplitude,evaluation,generation,parent,shell,beam,
            float(radius),float(gap/radius),evaluations,solves,rejected,
            sum(row['cell_evaluations'] for row in events),_owned(base),_owned(pattern),
            _owned(normal[0]),_owned(normal[1]),metric,predictor_dof,orientation,float(amplitude_origin))
    if type(metric) is NativeDirectorChordMetric:
        # Refuse an invalid accepted chart before assembly above. A predictor
        # leaving this metric's local domain is instead a recoverable cutback,
        # with the unchanged origin as its failure snapshot and work inventory.
        try:
            candidate=driver._advance(state,step,1.)
            path_row(driver,candidate,parent,metric,normal)
        except (_RotationStepTooLarge,ChordDomainError) as error:
            rejected+=1
            raise ArcTrialFailure(str(error),trial_snapshot()) from error
        state=candidate;amplitude+=delta
    else:
        state=driver._advance(state,step,1.);amplitude+=delta
    evaluation=evaluate(state,amplitude)
    for iteration in range(max_iterations+1):
        gap,gradient=path(state)
        if iteration_records is not None:
            iteration_records.append(dict(iteration=iteration,amplitude=amplitude,
                equilibrium=float(evaluation[2]),constraint=float(evaluation[3]),
                path_error=float(gap/radius),evaluations=evaluations,
                solves=solves,rejected=rejected,
                events=sum(row['cell_evaluations'] for row in events)))
        if evaluation[2]<=1e-8 and evaluation[3]<=1e-9 and abs(gap)/radius<=1e-9:
            with owner._commit_lock:
                current()
                trial=trial_snapshot()
                from ._stiffened_shell_arc_checkpoint import requires_coordinate_progress
                if requires_coordinate_progress(owner):
                    progress=orientation*(state.q[predictor_dof]-parent.q[predictor_dof])
                    if not progress>0.:
                        raise ArcTrialFailure("native arc selected coordinate made no oriented progress",trial)
                return trial
        if iteration==max_iterations:
            raise ArcTrialFailure("local arc Newton limit",trial_snapshot())
        step,delta=bordered_path_correction(evaluation[1],evaluation[0],direction,gradient,gap,free);solves+=1
        record=dict(cell_evaluations=0);events.append(record)
        initial=1.
        if not native:initial,_=trial_fraction(driver,state,step,beam,current,record)
        old=merit(state,evaluation)
        for backtrack in range(max_backtracks+1):
            amount=initial*.5**backtrack
            try:
                candidate=driver._advance(state,step,amount)
                if type(metric) is NativeDirectorChordMetric:
                    path_row(driver,candidate,parent,metric,normal)
                value=amplitude+amount*delta
                changed=evaluate(candidate,value)
                improved=merit(candidate,changed)<old
            except (_RotationStepTooLarge,BL5TrialDomainError,ChordDomainError):
                rejected+=1;continue
            if improved:
                state,amplitude,evaluation=candidate,value,changed;break
            rejected+=1
        else:raise ArcTrialFailure("local arc backtrack limit",trial_snapshot())
    raise AssertionError("unreachable")
