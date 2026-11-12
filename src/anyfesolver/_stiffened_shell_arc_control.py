"""Experimental fixed-mesh physical tangent-plane continuation.

The requested count is a count of accepted steps, not a requested total distance.
Cutbacks reduce actual travelled distance and remain visible in history. This
local controller does not establish stability, a limit point or branch uniqueness.
"""
from copy import deepcopy
from dataclasses import dataclass,replace
from hashlib import sha256
import numpy as np
from .control import cancellation_safe_point
from ._stiffened_shell_arc_metric import (PhysicalChordMetric, NativeDirectorChordMetric,
    require_metric_owner, DIRECTOR_METRIC_ID)
from ._stiffened_shell_arc_trial import select_trial
from ._stiffened_shell_arc_checkpoint import from_selection
from ._stiffened_shell_arc_publication import publish
from ._stiffened_shell_displacement_control import DisplacementControl


@dataclass(frozen=True)
class ArcLengthControl:
    """Fixed mesh weights follow compiled shell nodes, then beam nodes.

    The chosen free translation orients the tangent at each step. It must remain
    a usable local branch coordinate; no automatic branch switching is provided.
    Weights and characteristic length are explicit and frozen across restart.
    """
    domain: str
    node: int
    component: int
    metric: PhysicalChordMetric | NativeDirectorChordMetric
    radius: float
    orientation: int = 1
    max_cutbacks: int = 4
    max_attempts: int = 5
    max_iterations: int = 20
    max_backtracks: int = 12

    def __post_init__(self):
        if type(self.metric) not in (PhysicalChordMetric, NativeDirectorChordMetric):
            raise ValueError('explicit physical arc metric required')
        if (type(self.radius) not in (int,float) or not np.isfinite(self.radius)
                or self.radius<=0. or type(self.orientation) is not int or self.orientation not in (-1,1)):
            raise ValueError('positive arc radius and signed orientation required')
        for name,low,high in (('max_cutbacks',0,8),('max_attempts',1,64),
                              ('max_iterations',1,24),('max_backtracks',0,12)):
            if type(getattr(self,name)) is not int or not low<=getattr(self,name)<=high:
                raise ValueError('bounded arc controls required')


def solve(owner,actions,control,*,steps=1,cancellation_token=None,on_accepted=None):
    if not isinstance(control,ArcLengthControl) or type(steps) is not int or not 1<=steps<=1024:
        raise ValueError('arc control and bounded accepted-step count required')
    if on_accepted is not None and not callable(on_accepted):raise ValueError('accepted observer must be callable')
    driver=owner._finite_driver
    require_metric_owner(owner,control.metric)
    from ._stiffened_shell_affine_endpoint import is_native_point_material,is_native_material_bl5_arc
    native=is_native_point_material(owner) or is_native_material_bl5_arc(owner)
    if not (owner._bl5_shell_material or getattr(owner,"_curved_reference",False) or native) or driver is None or driver._accepted is None:
        raise ValueError('arc continuation requires an accepted coupled Q4/J2 BL5 or admitted curved point state')
    if native:
        from ._stiffened_shell_arc_checkpoint import require_native_seed
        require_native_seed(owner,driver)
    if getattr(owner,'_external_context',None) is not None or driver._external is not None:
        raise ValueError('arc continuation admits affine nodal dead translations only')
    adaptive=getattr(owner,'_displacement_adaptive',None)
    if adaptive is not None:
        raise ValueError('finish displacement history and use its endpoint checkpoint before arc continuation')
    dof=DisplacementControl(control.domain,control.node,control.component).dof(owner)
    direction=driver._load(actions)
    saved=getattr(owner,'_arc_snapshot',None)
    knobs={name:getattr(control,name) for name in
           ('max_cutbacks','max_attempts','max_iterations','max_backtracks')}
    if saved is None:
        seed_generation=owner._generation;seed_sha256=sha256(owner.checkpoint()).hexdigest()
        base=None;amplitude=0.;history=();attempt_history=()
    else:
        _,m,a=saved
        director=type(control.metric) is NativeDirectorChordMetric
        if (m.get('metric_kind')!=(DIRECTOR_METRIC_ID if director else None)
                or m.get('metric_shell_nodes')!=(control.metric.shell_nodes if director else None)):
            raise ValueError('arc metric identity differs from accepted history')
        if (m['predictor_dof']!=dof or m['orientation']!=control.orientation
                or m['requested_radius']!=control.radius or m['metric_length']!=control.metric.length
                or not np.array_equal(a['metric_weights'],control.metric.weights)
                or not np.array_equal(a['load_direction'],direction)
                or any(m[name]!=value for name,value in knobs.items())):
            raise ValueError('arc controller/load/metric differs from accepted history')
        seed_generation=m['seed_generation'];seed_sha256=m['seed_sha256']
        base=a['base_load'];history=tuple(map(tuple,a['history']));amplitude=float(history[-1][1])
        attempt_history=tuple(m['attempt_history'])
    if len(history)+steps>1024:raise ValueError('arc accepted history limit')
    for _ in range(steps):
        cancellation_safe_point(cancellation_token,'stiffened-shell.arc.step')
        driver=owner._finite_driver;generation=owner._generation
        selection=select_trial(driver,load_direction=direction,metric=control.metric,
            radius=control.radius,predictor_dof=dof,orientation=control.orientation,
            load_base=base,amplitude_origin=amplitude,cancellation_token=cancellation_token,**knobs)
        raw=from_selection(owner,selection,seed_generation=seed_generation,
            seed_sha256=seed_sha256,history=history,attempt_history=attempt_history)
        result=publish(owner,raw,expected_driver=driver,expected_generation=generation,
                       cancellation_token=cancellation_token)
        _,m,a=owner._arc_snapshot
        base=a['base_load'];history=result.history;amplitude=result.amplitude
        attempt_history=tuple(m['attempt_history'])
        if on_accepted is not None:on_accepted(deepcopy(result))
    # Owner/restore retain an accepted prefix, independently of this call's goal.
    return replace(result,status='completed')
