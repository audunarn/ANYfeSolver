"""Signed displacement adaptation with bounded durable controller metadata."""
from dataclasses import dataclass,replace,asdict
from hashlib import sha256
import json
import numpy as np
from ._stiffened_shell_displacement_increment import DisplacementIncrementState,MAX_EVENT_WORK
from ._stiffened_shell_increment_control import AdaptiveIncrementControl
from ._stiffened_shell_ge_checkpoint_binary import MAX_CHECKPOINT_BYTES
from ._stiffened_shell_displacement_corrector import DisplacementCorrectorFailure,PAIRED_FIXED_WORK_QUADRATIC_V1

from ._stiffened_shell_branch_event_trial import POLICY as EVENT_POLICY

MAGIC=b"ANYSSDCA1"
EVENT_MAGIC=b"ANYSSDCA2"
NATIVE_MAGIC=b"ANYSSDCN1"
PAIRED_MAGIC=b"ANYSSDCP1"
PAIRED_INTERPOLATION_MAGIC=b"ANYSSDCP2"
MAGICS=(MAGIC,EVENT_MAGIC,NATIVE_MAGIC,PAIRED_MAGIC,PAIRED_INTERPOLATION_MAGIC)
HEADER_LIMIT=256*1024


@dataclass(frozen=True)
class Context:
    state: DisplacementIncrementState
    dof: int
    direction_sha256: str
    history_start: int
    max_iterations: int
    max_backtracks: int
    trial_policy: str | None = None
    event_work_start: int = 0
    force_seed_sha256: str | None = None
    merit_globalization: str | None = None

    def __post_init__(self):
        if self.merit_globalization is not None and (
                self.merit_globalization!=PAIRED_FIXED_WORK_QUADRATIC_V1 or
                self.force_seed_sha256 is None or self.trial_policy is not None):
            raise ValueError("paired fixed-work interpolation context required")
        if self.force_seed_sha256 is not None and (
                type(self.force_seed_sha256) is not str or len(self.force_seed_sha256)!=64 or
                any(c not in "0123456789abcdef" for c in self.force_seed_sha256) or
                self.trial_policy is not None or self.event_work_start):
            raise ValueError("paired adaptive displacement requires an explicit force seed and no event policy")
        if ((self.trial_policy is not None and
                (type(self.trial_policy) is not str or self.trial_policy!=EVENT_POLICY)) or
                type(self.event_work_start) is not int or not 0<=self.event_work_start<=MAX_EVENT_WORK or
                self.event_work_start+sum(row.event_cell_evaluations for row in self.state.attempts)>MAX_EVENT_WORK or
                (self.trial_policy is None and (self.event_work_start or
                    any(row.event_cell_evaluations for row in self.state.attempts)))):
            raise ValueError("invalid adaptive displacement event policy/work")


def direction_hash(direction):
    return sha256(np.asarray(direction,dtype="<f8").tobytes()).hexdigest()


def require_native_seed(owner,driver,*,force_seed_sha256=None):
    """Check native seed admission before controller publication or restoration."""
    from ._stiffened_shell_affine_endpoint import is_paired_displacement
    if is_paired_displacement(owner):
        from hashlib import sha256
        snapshot=getattr(owner,"_displacement_snapshot",None)
        if type(force_seed_sha256) is not str:
            raise ValueError("paired displacement requires explicit original force seed digest")
        if snapshot is None:
            if (driver._programme is not None or driver._external is not None or
                    sha256(inner_checkpoint(owner)).hexdigest()!=force_seed_sha256):
                raise ValueError("paired displacement force seed mismatch")
        elif snapshot[1]["seed_sha256"]!=force_seed_sha256:
            raise ValueError("paired displacement original force seed mismatch")
        return
    if force_seed_sha256 is not None:
        raise ValueError("force seed transfer requires paired mixed Q4-J2 POINT owner")
    from ._stiffened_shell_affine_endpoint import require_native_displacement_policy
    require_native_displacement_policy(owner,driver)
    anchors=driver._state.shell_anchors
    snapshot=getattr(owner,"_displacement_snapshot",None)
    charted=False
    if snapshot is not None:
        from ._stiffened_shell_displacement_checkpoint import identities
        metadata=snapshot[1]
        charted=(metadata.get("chart_events",0)>0 and metadata.get("policy")==identities(owner)[2])
    if (driver._programme is not None or driver._external is not None or
            driver._adaptive_state is not None or
            (snapshot is None and anchors is not None) or
            (anchors is not None and not charted and not np.array_equal(
                anchors,np.tile(np.eye(3),(len(owner._shell_ids),1,1))))):
        raise ValueError("native adaptive displacement requires an ordinary unrebased force seed")
    if getattr(owner,"_displacement_snapshot",None) is None:
        applied=driver._load_factor*driver._pattern
        nodal=owner._shell_count+6*len(owner._beam_ids)
        if np.any(applied[nodal:]) or np.any(applied[:nodal].reshape(-1,6)[:,3:]):
            raise ValueError("native adaptive displacement requires nodal dead translations")


def require_native_inner(inner):
    from ._stiffened_shell_ge_checkpoint_binary import MAGIC as FORCE_MAGIC,S3_MAGIC,NATIVE_SHELL_MAGIC
    from ._stiffened_shell_displacement_checkpoint import MAGIC as FIXED_MAGIC
    if not inner.startswith((FORCE_MAGIC,S3_MAGIC,NATIVE_SHELL_MAGIC,FIXED_MAGIC)):
        raise ValueError("nested or unsupported native adaptive displacement checkpoint")


def require_paired_inner(inner,seed_sha256):
    from ._stiffened_shell_ge_checkpoint_binary import COMPENSATED_MAGIC
    from ._stiffened_shell_displacement_checkpoint import PAIRED_MAGIC as FIXED_PAIRED_MAGIC
    if type(inner) is not bytes or not inner.startswith((COMPENSATED_MAGIC,FIXED_PAIRED_MAGIC)):
        raise ValueError("paired adaptive displacement requires an exact paired snapshot or compensated seed")
    if inner.startswith(COMPENSATED_MAGIC) and sha256(inner).hexdigest()!=seed_sha256:
        raise ValueError("paired adaptive displacement force seed digest mismatch")


def encode(model,context,inner,*,native=False,paired=False):
    if type(paired) is not bool or paired!=(context.force_seed_sha256 is not None):
        raise ValueError("paired adaptive displacement formulation/seed mismatch")
    if type(native) is not bool or (native and context.trial_policy is not None):
        raise ValueError("native adaptive displacement excludes branch event policy")
    value=dict(model=model,state=context.state.payload(),dof=context.dof,
        direction_sha256=context.direction_sha256,history_start=context.history_start,
        max_iterations=context.max_iterations,max_backtracks=context.max_backtracks)
    if context.trial_policy is not None:
        value.update(trial_policy=context.trial_policy,event_work_start=context.event_work_start)
    if paired:value['force_seed_sha256']=context.force_seed_sha256
    if context.merit_globalization is not None:value['merit_globalization']=context.merit_globalization
    magic=PAIRED_MAGIC if paired else NATIVE_MAGIC if native else EVENT_MAGIC if context.trial_policy is not None else MAGIC
    if context.merit_globalization is not None:magic=PAIRED_INTERPOLATION_MAGIC
    header=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")
    if len(header)>HEADER_LIMIT or type(inner) is not bytes or not inner:
        raise ValueError("bounded adaptive displacement metadata required")
    if paired:require_paired_inner(inner,context.force_seed_sha256)
    elif native:require_native_inner(inner)
    content=magic+len(header).to_bytes(4,"little")+header+inner
    if len(content)+32>MAX_CHECKPOINT_BYTES:raise ValueError("adaptive displacement byte limit")
    return content+sha256(content).digest()


def decode(owner,raw):
    from ._stiffened_shell_affine_endpoint import is_native_point_material,is_paired_displacement
    native=is_native_point_material(owner)
    paired=is_paired_displacement(owner)
    if (not (owner._bl5_shell_material or getattr(owner,"_curved_material",False) or native) or
            type(raw) is not bytes or not raw.startswith(MAGICS) or
            not len(MAGIC)+36<len(raw)<=MAX_CHECKPOINT_BYTES):
        raise ValueError("bounded adaptive displacement checkpoint required")
    interpolating=raw.startswith(PAIRED_INTERPOLATION_MAGIC)
    if raw.startswith((PAIRED_MAGIC,PAIRED_INTERPOLATION_MAGIC))!=paired or (not paired and raw.startswith(NATIVE_MAGIC)!=native):
        raise ValueError("adaptive displacement formulation identity mismatch")
    eventful=raw.startswith(EVENT_MAGIC)
    start=len(MAGIC)+4;size=int.from_bytes(raw[len(MAGIC):start],"little")
    if not 0<size<=HEADER_LIMIT or start+size>=len(raw)-32 or sha256(raw[:-32]).digest()!=raw[-32:]:
        raise ValueError("invalid adaptive displacement framing")
    try:
        header=raw[start:start+size];v=json.loads(header)
        if json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")!=header:
            raise ValueError("noncanonical adaptive displacement metadata")
        keys={"model","state","dof","direction_sha256","history_start","max_iterations","max_backtracks"}
        if eventful:keys|={"trial_policy","event_work_start"}
        if paired:keys.add('force_seed_sha256')
        if interpolating:keys.add('merit_globalization')
        if type(v) is not dict or set(v)!=keys or v["model"]!=owner._identity:
            raise ValueError("incompatible adaptive displacement metadata")
        if interpolating and v['merit_globalization']!=PAIRED_FIXED_WORK_QUADRATIC_V1:
            raise ValueError("explicit paired interpolation wrapper policy required")
        state=DisplacementIncrementState.from_payload(v["state"])
        if eventful and v["trial_policy"]!=EVENT_POLICY:
            raise ValueError("explicit adaptive event trial policy required")
        context=Context(state,*(v[k] for k in ("dof","direction_sha256","history_start","max_iterations","max_backtracks")),
                        trial_policy=v.get("trial_policy"),event_work_start=v.get("event_work_start",0),
                        force_seed_sha256=v.get('force_seed_sha256'),merit_globalization=v.get('merit_globalization'))
    except (TypeError,KeyError,UnicodeError,RecursionError) as error:
        raise ValueError("invalid adaptive displacement metadata") from error
    if (any(type(v[k]) is not int for k in ("dof","history_start","max_iterations","max_backtracks")) or
            not 0<=context.dof<owner._shell_count+6*len(owner._beam_ids) or context.dof%6>=3 or
            context.dof not in owner._free or not 0<=context.history_start<=MAX_CHECKPOINT_BYTES//8 or
            not 1<=context.max_iterations<=24 or not 0<=context.max_backtracks<=12 or
            type(context.direction_sha256) is not str or len(context.direction_sha256)!=64 or
            any(c not in "0123456789abcdef" for c in context.direction_sha256)):
        raise ValueError("invalid adaptive displacement controls")
    inner=raw[start+size:-32]
    if inner.startswith(MAGICS):
        raise ValueError("nested adaptive displacement checkpoint")
    if paired:require_paired_inner(inner,context.force_seed_sha256)
    elif native:require_native_inner(inner)
    return context,inner


def inner_checkpoint(owner):
    snapshot=getattr(owner,"_displacement_snapshot",None)
    if snapshot is not None:return snapshot[0]
    from ._stiffened_shell_affine_endpoint import is_paired_displacement
    if is_paired_displacement(owner):
        adaptive=getattr(owner,'_displacement_adaptive',None)
        return decode(owner,adaptive[1])[1] if adaptive is not None else owner.checkpoint()
    return owner._finite_driver.checkpoint()


def validate_progress(owner,context):
    if owner._finite_driver is None or owner._finite_driver._accepted is None:
        raise ValueError("adaptive displacement requires a finite accepted seed")
    from ._stiffened_shell_affine_endpoint import is_native_point_material,is_paired_displacement
    if is_paired_displacement(owner)!=(context.force_seed_sha256 is not None):
        raise ValueError("adaptive displacement formulation/seed mismatch")
    if is_native_point_material(owner):
        if context.trial_policy is not None:
            raise ValueError("native adaptive displacement excludes branch event policy")
        require_native_seed(owner,owner._finite_driver,force_seed_sha256=context.force_seed_sha256)
    snapshot=getattr(owner,"_displacement_snapshot",None)
    rows=tuple(row.displacement for row in context.state.attempts if row.accepted)
    if snapshot is None:
        if rows or context.history_start!=0 or context.event_work_start!=0 or owner._finite_driver._state.q[context.dof]!=context.state.initial:
            raise ValueError("adaptive displacement seed mismatch")
        if (context.force_seed_sha256 is not None and
                context.direction_sha256!=direction_hash(owner._finite_driver._pattern)):
            raise ValueError("adaptive displacement load/control mismatch")
    else:
        _,metadata,a=snapshot;history=a["history"]
        start=context.history_start
        if (start>len(history) or tuple(history[start:,0])!=rows or
                (start==0 and context.event_work_start!=0)):
            raise ValueError("adaptive displacement accepted history mismatch")
        accepted_work=sum(row.event_cell_evaluations for row in context.state.attempts if row.accepted)
        if (metadata.get("trial_policy")!=context.trial_policy or
                metadata.get("event_cell_evaluations",0)!=context.event_work_start+accepted_work):
            raise ValueError("adaptive displacement event policy/work mismatch")
        initial=a["seed"][0] if start==0 else history[start-1,0]
        if (context.state.initial!=initial or context.dof!=metadata["control_dof"] or
                context.direction_sha256!=direction_hash(a["load_direction"])):
            raise ValueError("adaptive displacement load/control mismatch")


def solve(owner,actions,targets,control,limits,*,max_iterations,max_backtracks,
          cancellation_token=None,on_accepted=None):
    from ._stiffened_shell_displacement_control import solve as fixed_solve,DisplacementControl
    if type(limits) is not AdaptiveIncrementControl:
        raise ValueError("typed adaptive displacement limits required")
    driver=owner._finite_driver
    if driver is None:raise ValueError("accepted displacement seed required")
    if type(control) is not DisplacementControl or type(targets) is not tuple or any(
            not isinstance(v,(float,int,np.floating,np.integer)) or isinstance(v,(bool,np.bool_)) or not np.isfinite(v) for v in targets):
        raise ValueError("typed control and finite displacement targets required")
    from ._stiffened_shell_affine_endpoint import is_native_point_material,is_paired_displacement
    native=is_native_point_material(owner)
    paired=is_paired_displacement(owner)
    if control.merit_globalization is not None and not paired:
        raise ValueError("fixed-work interpolation requires paired adaptive owner")
    if native:
        if control.trial_policy is not None:
            raise ValueError("native adaptive displacement excludes branch event policy")
        require_native_seed(owner,driver,force_seed_sha256=control.force_seed_sha256)
    dof=control.dof(owner);direction=direction_hash(driver._load(actions))
    targets=tuple(float(v) for v in targets)
    snapshot=getattr(owner,"_displacement_snapshot",None)
    if snapshot is not None and snapshot[1].get("trial_policy")!=control.trial_policy:
        raise ValueError("displacement trial policy must match restart")
    current=getattr(owner,"_displacement_adaptive",None)
    if current is not None and not current[0].state.completed:
        context=current[0]
        if (context.state.targets!=targets or context.state.limits!=limits or
                context.dof!=dof or context.direction_sha256!=direction or
                context.max_iterations!=max_iterations or context.max_backtracks!=max_backtracks or
                context.trial_policy!=control.trial_policy or
                context.force_seed_sha256!=control.force_seed_sha256 or
                context.merit_globalization!=control.merit_globalization):
            raise ValueError("pending adaptive displacement controls must match")
    else:
        snapshot=getattr(owner,"_displacement_snapshot",None)
        count=0 if snapshot is None else len(snapshot[2]["history"])
        context=Context(DisplacementIncrementState(limits,targets,float(driver._state.q[dof])),
                        dof,direction,count,max_iterations,max_backtracks,control.trial_policy,
                        0 if snapshot is None else snapshot[1].get("event_cell_evaluations",0),
                        control.force_seed_sha256,control.merit_globalization)
    while not context.state.completed:
        target=context.state.next_target()
        predecessor=owner._finite_driver;generation=owner._generation
        try:
            fixed_solve(owner,actions,(target,),control,max_iterations=max_iterations,
                max_backtracks=max_backtracks,cancellation_token=cancellation_token,
                on_accepted=on_accepted,_adaptive=context)
        except DisplacementCorrectorFailure as error:
            if owner._finite_driver is not predecessor or owner._generation!=generation:
                # Observer failure after publication is not a rejected corrector.
                raise
            detail=error.detail
            if detail is None:raise
            state=context.state.record(False,evaluations=detail.evaluations,
                linear_solves=detail.linear_solves,rejected_trials=detail.rejected_trials,
                event_cell_evaluations=detail.event_cell_evaluations)
            context=replace(context,state=state)
            raw=encode(owner._identity,context,inner_checkpoint(owner),native=native,paired=paired)
            result=(replace(owner._finite_driver._accepted,status="accepted_prefix",
                            increment_attempts=state.attempts)
                    if paired or getattr(owner,"_displacement_snapshot",None) is not None else None)
            with owner._commit_lock:
                if owner._finite_driver is not predecessor or owner._generation!=generation:
                    raise ValueError("stale adaptive displacement failure")
                owner._displacement_adaptive=(context,raw)
                if result is not None:owner._finite_driver._accepted=result
        else:context=owner._displacement_adaptive[0]
    return owner.recover()


def restore_paired(owner,raw,*,cancellation_token=None):
    """Replay a paired controller in isolation, then publish one atomic prefix."""
    from copy import copy
    from threading import RLock
    from .control import cancellation_safe_point
    context,inner=decode(owner,raw)
    generation=owner._generation;driver=owner._finite_driver
    if driver is not None or owner._accepted is not None:
        raise RuntimeError("restore requires a fresh experimental GE analysis")
    cancellation_safe_point(cancellation_token,"stiffened-shell.displacement.adaptive.restore.prepare")
    prepared=copy(owner);prepared._commit_lock=RLock();prepared._writing=False
    prepared._displacement_adaptive=None
    result=prepared.restore(inner)
    validate_progress(prepared,context)
    result=replace(result,status="completed" if context.state.completed else "accepted_prefix",
                   increment_attempts=context.state.attempts)
    prepared._finite_driver._accepted=result
    cancellation_safe_point(cancellation_token,"stiffened-shell.displacement.adaptive.restore.publish")
    with owner._commit_lock:
        if owner._generation!=generation or owner._finite_driver is not driver or owner._accepted is not None:
            raise ValueError("stale paired adaptive displacement restore")
        owner._finite_driver=prepared._finite_driver;owner._finite_driver.owner=owner
        owner._generation=prepared._generation
        owner._displacement_snapshot=getattr(prepared,"_displacement_snapshot",None)
        owner._displacement_adaptive=(context,raw)
    return result
