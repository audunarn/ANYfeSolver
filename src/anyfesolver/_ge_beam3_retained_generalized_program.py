"""Private transactional retained generalized distributed force controller."""
from dataclasses import dataclass
import numpy as np
from .control import cancellation_safe_point,SolveCancelled
from ._ge_beam3_retained_generalized_state import Context,Program,State

@dataclass(frozen=True)
class Result:
    status: str
    completed_targets: int
    state: State
    checkpoint: bytes
    failure: str | None
    production_qualified: bool=False

def solve(model,program,*,checkpoint=None,expected_sha256=None,stop_after=None,cancellation_token=None,progress=None):
    return _solve(Context,model,program,checkpoint=checkpoint,expected_sha256=expected_sha256,
        stop_after=stop_after,cancellation_token=cancellation_token,progress=progress)


def _solve(context_type,model,program,*,checkpoint=None,expected_sha256=None,stop_after=None,cancellation_token=None,progress=None):
    cancellation_safe_point(cancellation_token,'retained-generalized.start')
    if progress is not None and not callable(progress):raise ValueError('callable observer required')
    context=context_type(model,program);end=len(program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0<=end<=len(program.targets):raise ValueError('bounded stop target')
    if checkpoint is None:
        if expected_sha256 is not None:raise ValueError('hash requires a checkpoint')
        accepted,records=context.initial,();capsule=context.checkpoint(records)
    else:accepted,records=context.restore(checkpoint,expected_sha256=expected_sha256);capsule=checkpoint
    if accepted.completed_targets>end:raise ValueError('accepted history cannot rewind')
    status='completed' if end==len(program.targets) else 'paused';failure=None
    def safe(stage,target,iteration=0):
        cancellation_safe_point(cancellation_token,stage);context.guard()
        if progress is not None:progress(dict(stage=stage,target=target,iteration=iteration))
        context.guard();cancellation_safe_point(cancellation_token,stage)
    try:
        safe('retained-generalized.initialized',accepted.completed_targets)
        for index in range(accepted.completed_targets,end):
            parameter=program.targets[index];trial=accepted.mechanical;origins=accepted.histories
            for iteration in range(program.max_iterations+1):
                safe('retained-generalized.before_assembly',index+1,iteration)
                r,j,metrics,_,_=context.assemble(trial,parameter,origins)
                safe('retained-generalized.before_factorization',index+1,iteration)
                step,correction=context.step(trial,r,j)
                if max(*metrics,correction)<=1e-11:
                    proposed,record=context.stage(trial,accepted,records,iteration)
                    next_records=(*records,record);staged=context.checkpoint(next_records)
                    safe('retained-generalized.before_commit',index+1,iteration)
                    accepted,records,capsule=proposed,next_records,staged
                    safe('retained-generalized.committed',index+1,iteration);break
                if iteration==program.max_iterations:raise RuntimeError('retained generalized Newton limit')
                angular=[step[6*i+3:6*i+6] for i in range(len(context.node_ids))]
                angular.extend(step[context.nodal_count+24*i+3*c:context.nodal_count+24*i+3*c+3] for i in range(len(context.elements)) for c in (0,1))
                fraction=min(1.,.45*np.pi/max(max(float(np.linalg.norm(v)) for v in angular),np.finfo(float).tiny))
                for cut in range(program.max_backtracks+1):
                    safe('retained-generalized.before_trial',index+1,iteration)
                    candidate=context.advance(trial,step*(fraction*.5**cut))
                    _,_,changed,_,_=context.assemble(candidate,parameter,origins)
                    if max(changed)<max(metrics):trial=candidate;break
                else:raise RuntimeError('retained generalized line search limit')
    except SolveCancelled as exc:status,failure='cancelled',str(exc)
    except Exception as exc:status,failure='failed',type(exc).__name__+': '+str(exc)
    return Result(status,accepted.completed_targets,accepted,capsule,failure)
