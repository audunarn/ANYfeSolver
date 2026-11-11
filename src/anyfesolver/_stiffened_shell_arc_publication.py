"""Atomic endpoint publication behind the experimental public arc controller."""
from dataclasses import dataclass,replace
from math import fsum
from hashlib import sha256
import numpy as np
from scipy.spatial.transform import Rotation
from .control import cancellation_safe_point
from ._stiffened_shell_arc_checkpoint import validate_endpoint
from ._stiffened_shell_arc_trial import ArcAttempt
from ._stiffened_shell_affine_endpoint import has_shell_material,uses_native_material_history,has_native_s3_material


@dataclass(frozen=True)
class ArcResult:
    status: str
    amplitude: float
    arc_position: float
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
    beam_generalized_material: tuple
    increment_attempts: tuple
    attempt_history: tuple
    production_qualified: bool = False
    beam_fibre_material: tuple = ()
    beam_fibre_energy: tuple = ()
    shell_s3_material: tuple = ()


def prepare(owner,raw):
    from ._stiffened_shell_ge_finite_experimental import (
        _owned,_accepted_shell_material_origins,_shell_q4_recovery,
        _material_observables,_material_histories)
    m,a,driver,state,evaluation,shell_parent,beam_parent=validate_endpoint(owner,raw)
    residual,_,relative,gap,shell_next,beam_next=evaluation
    native=uses_native_material_history(owner);s3=has_native_s3_material(owner)
    count=len(owner._definition.shell.q4_elements)
    if has_shell_material(owner):
        q4_next=tuple({**candidate,'plastic_strain':a['shell_plastic'][i],
                          'alpha':a['shell_alpha'][i]} for i,candidate in enumerate(shell_next[:count] if s3 else shell_next))
        shell_next=q4_next+driver._snapshot_s3_accepted if s3 else q4_next
    if native:
        from ._stiffened_shell_beam_history import histories
        beam_next=histories(owner._beam_parts,a,'beam')
    else:beam_next=_material_histories(owner,a['beam_plastic'],a['beam_accumulated'])
    history=tuple(tuple(float(x) for x in row) for row in a['history'])
    support=np.zeros(owner._K.shape[0]);support[owner._fixed]=residual[owner._fixed]
    local=_owned(state.q[:owner._shell_count]);display=local.copy().reshape(-1,6)
    display[:,3:]=Rotation.from_matrix(a['shell_frames']).as_rotvec()
    result=ArcResult('accepted_prefix',history[-1][1],fsum(row[0] for row in history),
        history,m['generation'],_owned(display.ravel()),_owned(driver._beam_positions(state)),
        state.frames,state.cells,state.resultants,state.multipliers,_owned(support),
        a['applied_load'],relative,gap,a['shell_frames'],local,a['shell_anchors'],
        (_shell_q4_recovery(owner._shell_parts[:count],shell_next[:count]) if s3 else
         _shell_q4_recovery(owner._shell_parts,shell_next) if has_shell_material(owner) else ()),_material_observables(beam_next),
        (tuple(tuple(float(v) for v in row['alpha']) for row in shell_next) if has_shell_material(owner) else ()),
        driver._line_trial_recovery[1],driver._point_records(state),
        driver._generalized_records(state,beam_parent,beam_next,include_native=native),
        tuple(ArcAttempt(**row) for row in m['attempts']),
        tuple(tuple(ArcAttempt(**row) for row in group['attempts']) for group in m['attempt_history']))
    driver._state=state;driver._shell_predecessors=shell_parent;driver._beam_predecessors=beam_parent
    if native:
        from ._stiffened_shell_ge_finite_experimental import _fibre_recovery
        records,energies=_fibre_recovery(owner,state,beam_parent,beam_next)
        result=replace(result,beam_fibre_material=records,beam_fibre_energy=energies)
    if s3:
        from ._stiffened_shell_ge_finite_experimental import _shell_s3_recovery
        result=replace(result,shell_s3_material=_shell_s3_recovery(owner._shell_parts[count:],shell_next[count:]))
    driver._shell_origins=(shell_next if s3 else _accepted_shell_material_origins(owner,state,shell_next)
                           if has_shell_material(owner) else shell_next)
    driver._beam_origins=beam_next;driver._accepted=result
    driver._arc_applied_load=a['applied_load']
    return m,a,driver,result


def _bind_predecessor(owner,m,a,live,expected_generation):
    from ._stiffened_shell_ge_finite_experimental import _material_arrays,_shell_material_arrays
    from ._stiffened_shell_rotation_charts import physical_frames
    if (live is None or m['generation']!=expected_generation+1
            or m['predecessor_sha256']!=sha256(owner.checkpoint()).hexdigest()):
        raise ValueError('arc endpoint has a different accepted predecessor')
    saved=getattr(owner,'_arc_snapshot',None)
    if saved is not None:
        previous=saved[1]
        if any(m.get(key)!=previous.get(key) for key in ('policy','metric_kind','metric_shell_nodes')):
            raise ValueError('arc publication cannot cross metric history identities')
    nodes=len(owner._shell_ids)+len(owner._beam_ids)
    native=uses_native_material_history(owner);s3=has_native_s3_material(owner)
    count=len(owner._definition.shell.q4_elements)
    shell=(_shell_material_arrays(live._shell_origins[:count] if s3 else live._shell_origins) if has_shell_material(owner) else ())
    frames=np.concatenate((physical_frames(live._state.q[:owner._shell_count],
        live._state.shell_anchors),live._state.frames))
    applied=getattr(live,'_arc_applied_load',None)
    if applied is None:applied=getattr(live,'_displacement_applied_load',None)
    if applied is None:
        applied=live._applied_load if live._programme is not None else live._load_factor*live._pattern
    witnesses=(('origin_translations',live._state.q[:nodes*6].reshape(-1,6)[:,:3]),
        ('origin_frames',frames))
    if native:
        from ._stiffened_shell_beam_history import arrays
        witnesses+=arrays(owner._beam_parts,live._beam_origins,'beam_origin')
    else:
        beam=_material_arrays(live._beam_origins)
        witnesses+=(('beam_origin_plastic',beam[0]),('beam_origin_accumulated',beam[1]))
    if has_shell_material(owner):
        witnesses+=(('shell_origin_plastic',shell[0]),('shell_origin_alpha',shell[1]))
    if s3:
        anchors=live._state.shell_anchors
        if anchors is None:anchors=np.tile(np.eye(3),(len(owner._shell_ids),1,1))
        witnesses+=(('shell_origin_q',live._state.q[:owner._shell_count]),('shell_origin_anchors',anchors))
        material=owner._shell_model.get_material('shell')
        blobs=tuple(element.serialize_nonlinear_state(owner._shell_model.mesh,material,origin,3)
            for (_,element,_,_),origin in zip(owner._shell_parts[count:],live._shell_origins[count:]))
        if a['s3_states'][len(blobs):]!=blobs:raise ValueError('arc S3 predecessor mismatch')
    if (any(not np.array_equal(a[name],value) for name,value in witnesses)
            or not np.array_equal(a['base_load']+m['amplitude_origin']*a['load_direction'],applied)):
        raise ValueError('arc physical/material predecessor mismatch')


def publish(owner,raw,*,expected_driver,expected_generation,cancellation_token=None,restoring=False):
    m,a,driver,result=prepare(owner,raw)
    cancellation_safe_point(cancellation_token,'stiffened-shell.arc.publish')
    with owner._commit_lock:
        owner._require_open()
        if owner._generation!=expected_generation or owner._finite_driver is not expected_driver:
            raise ValueError('stale arc publication')
        if restoring:
            if expected_driver is not None or owner._accepted is not None or expected_generation!=0:
                raise ValueError('arc restore requires a fresh owner')
        else:
            _bind_predecessor(owner,m,a,expected_driver,expected_generation)
        cancellation_safe_point(cancellation_token,'stiffened-shell.arc.commit')
        driver.owner=owner;owner._finite_driver=driver;owner._generation=result.generation
        owner._arc_snapshot=(raw,m,a)
        owner._displacement_snapshot=None;owner._displacement_adaptive=None
    return result
