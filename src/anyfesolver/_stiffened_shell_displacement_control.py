"""Experimental coupled-Q4 displacement controller, separate from load cursors."""
from dataclasses import dataclass,replace,field
from copy import deepcopy
from hashlib import sha256
import numpy as np
from .control import cancellation_safe_point
from ._stiffened_shell_branch_event_trial import POLICY as GENERALIZED_BRANCH_EVENT_TRIAL
from ._stiffened_shell_displacement_corrector import correct,PAIRED_FIXED_WORK_QUADRATIC_V1
from ._stiffened_shell_displacement_checkpoint import encode_snapshot,validate_endpoint


@dataclass(frozen=True)
class DisplacementControl:
    domain: str
    node: int
    component: int
    trial_policy: str | None = None
    force_seed_sha256: str | None = field(default=None,kw_only=True)
    merit_globalization: str | None = field(default=None,kw_only=True)

    def __post_init__(self):
        if self.merit_globalization is not None and (
                self.merit_globalization!=PAIRED_FIXED_WORK_QUADRATIC_V1 or
                self.force_seed_sha256 is None or self.trial_policy is not None):
            raise ValueError("explicit paired fixed-work interpolation policy and seed required")
        if self.force_seed_sha256 is not None and (type(self.force_seed_sha256) is not str or
                len(self.force_seed_sha256)!=64 or any(c not in "0123456789abcdef" for c in self.force_seed_sha256)):
            raise ValueError("canonical force seed digest required")
        if self.trial_policy not in (None,GENERALIZED_BRANCH_EVENT_TRIAL):
            raise ValueError("supported displacement trial policy required")

    def dof(self,owner):
        if self.domain not in ("shell","beam") or type(self.node) is not int or type(self.component) is not int or not 0<=self.component<3:
            raise ValueError("physical translational displacement control required")
        index=owner._shell_index if self.domain=="shell" else owner._beam_index
        if self.node not in index:raise ValueError("unknown displacement control node")
        dof=(0 if self.domain=="shell" else owner._shell_count)+6*index[self.node]+self.component
        if dof not in owner._free:raise ValueError("displacement control is restrained")
        return dof


@dataclass(frozen=True)
class DisplacementResult:
    status: str
    amplitude: float
    target: float
    history: tuple
    generation: int
    shell_displacement: np.ndarray
    beam_positions: np.ndarray
    beam_frames: np.ndarray
    cell_frames: np.ndarray
    cell_resultants: np.ndarray
    multipliers: np.ndarray
    support_reactions: np.ndarray
    applied_dead_load: np.ndarray
    free_equilibrium_ratio: float
    constraint_max: float
    shell_frames: np.ndarray
    shell_chart_coordinates: np.ndarray
    shell_rotation_anchors: np.ndarray
    shell_q4_material: tuple
    beam_station_accumulated: tuple
    shell_layer_alpha: tuple
    beam_shell_line_work: tuple
    point_wrenches: tuple
    production_qualified: bool = False
    increment_attempts: tuple = ()
    trial_policy: str | None = None
    event_cell_evaluations: int = 0
    beam_generalized_material: tuple = ()
    beam_fibre_material: tuple = ()
    beam_fibre_energy: tuple = ()
    shell_s3_material: tuple = ()
    distributing_footprint_wrenches: tuple = ()


def prepare(owner,raw):
    from ._stiffened_shell_ge_finite_experimental import (
        _owned,_accepted_shell_material_origins,_shell_q4_recovery,_material_observables)
    metadata,a,driver,state,evaluation,shell_parent,beam_parent=validate_endpoint(owner,raw)
    residual,_,relative,gap,shell_next,beam_next=evaluation
    from ._stiffened_shell_ge_finite_experimental import _material_histories
    # Replay validates both versions; saved accepted histories remain authoritative.
    # Installing tolerance-close replay values would permit repeated restore drift.
    from ._stiffened_shell_affine_endpoint import has_shell_material,has_native_s3_material
    s3_material=has_native_s3_material(owner)
    q4_count=len(owner._definition.shell.q4_elements)
    if has_shell_material(owner):
        q4_next=tuple({**candidate,"plastic_strain":a["shell_plastic"][i],
                          "alpha":a["shell_alpha"][i]}
                         for i,candidate in enumerate(shell_next[:q4_count] if s3_material else shell_next))
        shell_next=q4_next+driver._snapshot_s3_accepted if s3_material else q4_next
    from ._stiffened_shell_affine_endpoint import uses_native_material_history
    native=uses_native_material_history(owner)
    if native:
        from ._stiffened_shell_beam_history import histories
        beam_next=histories(owner._beam_parts,a,"beam")
    else:
        beam_next=_material_histories(owner,a["beam_plastic"],a["beam_accumulated"])
    history=tuple(tuple(float(v) for v in row) for row in a["history"])
    target,amplitude=history[-1] if history else tuple(a["seed"])
    support=np.zeros(owner._K.shape[0]);support[owner._fixed]=residual[owner._fixed]
    from scipy.spatial.transform import Rotation
    local=_owned(state.q[:owner._shell_count])
    display=local.copy().reshape(-1,6)
    display[:,3:]=Rotation.from_matrix(a["shell_frames"]).as_rotvec()
    result=DisplacementResult("completed",amplitude,target,history,metadata["generation"],
        _owned(display.ravel()),_owned(driver._beam_positions(state)),
        state.frames,state.cells,state.resultants,state.multipliers,_owned(support),
        a["applied_load"],relative,gap,a["shell_frames"],local,a["shell_anchors"],
        (_shell_q4_recovery(owner._shell_parts[:q4_count],shell_next[:q4_count])
         if s3_material else _shell_q4_recovery(owner._shell_parts,shell_next)
         if has_shell_material(owner) else ()),_material_observables(beam_next),
        (tuple(tuple(float(v) for v in item["alpha"]) for item in shell_next) if has_shell_material(owner) else ()),
        driver._line_trial_recovery[1],driver._point_records(state),
        trial_policy=metadata.get("trial_policy"),
        event_cell_evaluations=metadata.get("event_cell_evaluations",0),
        beam_generalized_material=driver._generalized_records(state,beam_parent,beam_next,include_native=native))
    if native:
        from ._stiffened_shell_ge_finite_experimental import _fibre_recovery
        records,energies=_fibre_recovery(owner,state,beam_parent,beam_next)
        result=replace(result,beam_fibre_material=records,beam_fibre_energy=energies)
    if s3_material:
        from ._stiffened_shell_ge_finite_experimental import _shell_s3_recovery
        result=replace(result,shell_s3_material=_shell_s3_recovery(
            owner._shell_parts[q4_count:],shell_next[q4_count:]))
    if getattr(owner,"_finite_distributing",False):
        result=replace(result,distributing_footprint_wrenches=
            driver._distributing_records(state,metadata["generation"]))
    driver._state=state;driver._shell_predecessors=shell_parent
    driver._beam_predecessors=beam_parent
    driver._shell_origins=(shell_next if s3_material else
                          _accepted_shell_material_origins(owner,state,shell_next)
                           if owner._definition.shell.j2_curve is not None else shell_next)
    driver._beam_origins=beam_next;driver._accepted=result
    driver._displacement_applied_load=a["applied_load"]
    return metadata,a,driver,result


def publish(owner,raw,*,expected_driver,expected_generation,cancellation_token=None,adaptive=None):
    metadata,a,driver,result=prepare(owner,raw)
    adaptive_raw=None
    if adaptive is not None:
        from ._stiffened_shell_displacement_adaptive import encode
        from ._stiffened_shell_affine_endpoint import uses_native_material_history,is_paired_displacement
        adaptive_raw=encode(owner._identity,adaptive,raw,native=uses_native_material_history(owner),
                            paired=is_paired_displacement(owner))
        result=replace(result,status="completed" if adaptive.state.completed else "accepted_prefix",
                       increment_attempts=adaptive.state.attempts)
        driver._accepted=result
    cancellation_safe_point(cancellation_token,"stiffened-shell.displacement.publish")
    with owner._commit_lock:
        if owner._generation!=expected_generation or owner._finite_driver is not expected_driver:
            raise ValueError("stale displacement publication")
        driver.owner=owner
        owner._finite_driver=driver;owner._generation=result.generation
        owner._displacement_snapshot=(raw,metadata,a)
        owner._displacement_adaptive=None if adaptive is None else (adaptive,adaptive_raw)
    return result


def solve(owner,actions,targets,control,*,max_iterations,max_backtracks,
          cancellation_token=None,on_accepted=None,_adaptive=None):
    from ._stiffened_shell_ge_finite_experimental import _material_arrays,_shell_material_arrays
    from ._stiffened_shell_rotation_charts import physical_frames
    from ._stiffened_shell_affine_endpoint import uses_native_material_history
    native=uses_native_material_history(owner)
    from ._stiffened_shell_affine_endpoint import is_paired_displacement
    paired=is_paired_displacement(owner)
    if getattr(control,'merit_globalization',None) is not None and (
            not paired or _adaptive is None or
            _adaptive.merit_globalization!=control.merit_globalization):
        raise ValueError("fixed-work interpolation requires matching paired adaptive context")
    if paired and (type(control) is not DisplacementControl or control.domain!="shell" or
            control.force_seed_sha256 is None or control.trial_policy is not None):
        raise ValueError("paired mixed displacement admits explicit force-seed shell translations only")
    if not paired and getattr(control,"force_seed_sha256",None) is not None:
        raise ValueError("force seed transfer requires paired mixed owner")
    distributing=getattr(owner,"_finite_distributing",False)
    if distributing and (type(control) is not DisplacementControl or control.trial_policy is not None or _adaptive is not None):
        raise ValueError("finite distributing displacement admits fixed targets without event/adaptive policies only")
    if native and type(control) is DisplacementControl and control.trial_policy is not None:
        raise ValueError("native material displacement excludes branch event policy")
    if type(control) is not DisplacementControl or not (
            owner._bl5_shell_material or getattr(owner,"_curved_material",False) or native or distributing):
        raise ValueError("coupled BL5 or curved material displacement control required")
    pending=getattr(owner,"_displacement_adaptive",None)
    if _adaptive is None and pending is not None and not pending[0].state.completed:
        raise ValueError("pending adaptive displacement controls must match")
    driver=owner._finite_driver
    if driver is None or driver._accepted is None:
        raise ValueError("displacement control requires an accepted force-controlled seed")
    if distributing and (driver._external is not None or driver._programme is not None or
            driver._adaptive_state is not None or
            (driver._state.shell_anchors is not None and not np.array_equal(
                driver._state.shell_anchors,np.tile(np.eye(3),(len(owner._shell_ids),1,1))))):
        raise ValueError("finite distributing displacement requires an ordinary unrebased fixed-force seed")
    from ._stiffened_shell_affine_endpoint import require_native_displacement_policy
    require_native_displacement_policy(owner,driver)
    if native:
        from ._stiffened_shell_displacement_adaptive import require_native_seed
        require_native_seed(owner,driver,force_seed_sha256=control.force_seed_sha256)
    if type(targets) is not tuple or not targets or any(not isinstance(v,(float,int,np.floating,np.integer)) or isinstance(v,(bool,np.bool_)) or not np.isfinite(v) for v in targets):
        raise ValueError("finite displacement targets required")
    dof=control.dof(owner);direction=driver._load(actions)
    if distributing and owner._distributing_cyclic and not np.array_equal(
            direction,owner._cyclic_reference_pattern):
        raise ValueError("cyclic distributing displacement requires the model-bound reference load direction")
    existing=getattr(owner,"_displacement_snapshot",None)
    if paired and existing is None and not np.array_equal(direction,driver._pattern):
        raise ValueError("paired displacement must retain original force pattern")
    from ._stiffened_shell_displacement_adaptive import inner_checkpoint
    force_seed=(inner_checkpoint(owner) if existing is None else existing[2].get("force_seed")) if paired else None
    if existing is None:
        raw=force_seed if paired else owner.checkpoint();generation=owner._generation
        base=(driver._applied_load if driver._programme is not None else driver._load_factor*driver._pattern)
        metadata=dict(seed_generation=generation,seed_sha256=sha256(raw).hexdigest(),control_dof=dof,
                      trial_policy=control.trial_policy,event_cell_evaluations=0)
        seed=np.array([driver._state.q[dof],0.]);history=();amplitude=0.
        cyclic_seed_factor=driver._load_factor if distributing and owner._distributing_cyclic else None
    else:
        _,m,a=existing
        if (dof!=m["control_dof"] or not np.array_equal(direction,a["load_direction"]) or
                control.trial_policy!=m.get("trial_policy")):
            raise ValueError("displacement control and load direction must match restart")
        metadata={k:m[k] for k in ("seed_generation","seed_sha256","control_dof")}
        metadata["chart_events"]=m.get("chart_events",0)
        metadata["trial_policy"]=m.get("trial_policy")
        metadata["event_cell_evaluations"]=m.get("event_cell_evaluations",0)
        base=a["base_load"];seed=a["seed"];history=tuple(map(tuple,a["history"]))
        amplitude=float(history[-1][1] if history else seed[1])
        cyclic_seed_factor=float(a["cyclic_seed_factor"][0]) if distributing and owner._distributing_cyclic else None
    if any(float(x)==float(y) for x,y in zip((driver._state.q[dof],)+targets,targets)):
        raise ValueError("displacement targets must change")
    for target in targets:
        driver=owner._finite_driver;generation=owner._generation
        trial=correct(driver,base_load=base,load_direction=direction,amplitude=amplitude,
            control_dof=dof,target=float(target),max_iterations=max_iterations,
            max_backtracks=max_backtracks,cancellation_token=cancellation_token,
            trial_policy=control.trial_policy,force_seed_sha256=control.force_seed_sha256,
            **({} if control.merit_globalization is None else {"merit_globalization":control.merit_globalization}))
        state=trial.state
        anchors=(np.tile(np.eye(3),(len(owner._shell_ids),1,1)) if state.shell_anchors is None else state.shell_anchors)
        arrays={k:getattr(state,k) for k in ("q","frames","cells","resultants","multipliers")}
        if paired:
            arrays.update((k,getattr(state,k)) for k in ("q_low","frames_low","cells_low","resultants_low"))
            arrays["force_seed"]=force_seed
        next_history=history+((float(target),trial.amplitude),)
        arrays.update(shell_anchors=anchors,shell_frames=physical_frames(state.q[:owner._shell_count],anchors),
            base_load=base,load_direction=direction,applied_load=base+trial.amplitude*direction,
            seed=seed,history=np.asarray(next_history))
        if cyclic_seed_factor is not None:
            arrays["cyclic_seed_factor"]=np.asarray([cyclic_seed_factor])
        if native:
            from ._stiffened_shell_beam_history import arrays as native_arrays
            arrays.update(native_arrays(owner._beam_parts,trial.evaluation[5],"beam"))
            arrays.update(native_arrays(owner._beam_parts,trial.beam_predecessor,"beam_origin"))
        else:
            arrays["beam_plastic"],arrays["beam_accumulated"]=_material_arrays(trial.evaluation[5])
            arrays["beam_origin_plastic"],arrays["beam_origin_accumulated"]=_material_arrays(trial.beam_predecessor)
        from ._stiffened_shell_affine_endpoint import has_shell_material,has_native_s3_material
        if has_shell_material(owner):
            q4_count=len(owner._definition.shell.q4_elements)
            s3_material=has_native_s3_material(owner)
            arrays["shell_plastic"],arrays["shell_alpha"]=_shell_material_arrays(
                trial.evaluation[4][:q4_count] if s3_material else trial.evaluation[4])
            arrays["shell_origin_plastic"],arrays["shell_origin_alpha"]=_shell_material_arrays(
                trial.shell_predecessor[:q4_count] if s3_material else trial.shell_predecessor)
            if s3_material:
                from ._stiffened_shell_ge_finite_experimental import _accepted_shell_material_origins
                accepted_shell=_accepted_shell_material_origins(owner,state,trial.evaluation[4])
                material=owner._shell_model.get_material("shell")
                arrays["s3_states"]=tuple(element.serialize_nonlinear_state(
                    owner._shell_model.mesh,material,origin,3)
                    for origins in (accepted_shell,trial.shell_predecessor)
                    for (_eid,element,_slots,_),origin in zip(
                        owner._shell_parts[q4_count:],origins[q4_count:]))
                arrays["shell_origin_q"]=trial.predecessor_state.q[:owner._shell_count]
                arrays["shell_origin_anchors"]=(np.tile(np.eye(3),(len(owner._shell_ids),1,1))
                    if trial.predecessor_state.shell_anchors is None else
                    trial.predecessor_state.shell_anchors)
        next_metadata=dict(metadata,event_cell_evaluations=
                           metadata["event_cell_evaluations"]+trial.event_cell_evaluations)
        raw=encode_snapshot(owner,generation=generation+1,**next_metadata,arrays=arrays)
        next_adaptive=None
        if _adaptive is not None:
            next_adaptive=replace(_adaptive,state=_adaptive.state.record(True,
                evaluations=trial.evaluations,linear_solves=trial.linear_solves,
                rejected_trials=trial.rejected_trials,
                event_cell_evaluations=trial.event_cell_evaluations))
        result=publish(owner,raw,expected_driver=driver,expected_generation=generation,
                       cancellation_token=cancellation_token,adaptive=next_adaptive)
        history=next_history;amplitude=trial.amplitude;metadata=next_metadata
        if on_accepted is not None:on_accepted(deepcopy(result))
    return owner._finite_driver._accepted


def rebase(owner,expected_generation,*,cancellation_token=None):
    """Publish a chart event at the same accepted displacement and affine load."""
    if type(expected_generation) is not int or expected_generation!=owner._generation:
        raise ValueError("stale or invalid displacement rebase generation")
    snapshot=getattr(owner,"_displacement_snapshot",None)
    if snapshot is None:
        raise ValueError("displacement rebase requires an accepted displacement step")
    from ._stiffened_shell_nodal_footprint import require_rotation_triples
    require_rotation_triples(owner._fixed,len(owner._shell_ids))
    driver=owner._finite_driver
    _,metadata,saved=snapshot
    cancellation_safe_point(cancellation_token,"stiffened-shell.displacement.rebase.prepare")
    q=saved["q"].copy()
    rotations=q[:owner._shell_count].reshape(-1,6)[:,3:]
    if not np.any(rotations!=0.):
        raise ValueError("no nonzero shell chart coordinates to rebase")
    rotations[:]=0.
    arrays=dict(saved)
    arrays.update(q=q,shell_anchors=saved["shell_frames"])
    raw=encode_snapshot(owner,generation=expected_generation+1,
        seed_generation=metadata["seed_generation"],seed_sha256=metadata["seed_sha256"],
        control_dof=metadata["control_dof"],arrays=arrays,
        chart_events=metadata.get("chart_events",0)+1,
        trial_policy=metadata.get("trial_policy"),
        event_cell_evaluations=metadata.get("event_cell_evaluations",0))
    adaptive=getattr(owner,"_displacement_adaptive",None)
    return publish(owner,raw,expected_driver=driver,expected_generation=expected_generation,
        cancellation_token=cancellation_token,adaptive=None if adaptive is None else adaptive[0])
