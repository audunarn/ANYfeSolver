"""Experimental arc endpoint framing, distinct from force/displacement history.

Checksums and controller consistency do not authenticate a loading history.
Mechanical restoration uses the common fixed-predecessor endpoint validator.
"""
import json
from hashlib import sha256
from dataclasses import asdict
import numpy as np
from ._stiffened_shell_ge_checkpoint_binary import encode,decode,encode_s3,decode_s3,MAX_CHECKPOINT_BYTES
from ._stiffened_shell_arc_metric import (PhysicalChordMetric, NativeDirectorChordMetric,
    DIRECTOR_METRIC_ID, require_metric_owner, require_director_chart)
from ._stiffened_shell_rotation_charts import physical_frames

MAGIC=b'ANYSSARC1'
KIND='finite_arc_bl5_coupled'
POLICY='B3_GE_BL5_PHYSICAL_TANGENT_PLANE_V1_DEV'
NATIVE_CHART_POLICY='B3_GE_NATIVE_POINT_ARC_FIXED_SHELL_CHART_V1_DEV'
NATIVE_BL5_CHART_POLICY='B3_GE_NATIVE_FIBRE_BL5_ARC_UNREBASED_SHELL_CHART_V1_DEV'


def _native_owner(owner):
    from ._stiffened_shell_affine_endpoint import is_native_point_material,is_native_material_bl5_arc
    return is_native_point_material(owner) or is_native_material_bl5_arc(owner)


def _chart_policy(owner):
    from ._stiffened_shell_affine_endpoint import is_native_material_bl5_arc
    return NATIVE_BL5_CHART_POLICY if is_native_material_bl5_arc(owner) else NATIVE_CHART_POLICY


def _s3_arc_policies():
    """(legacy V1, legacy V2, stable V2) arc identities for the generalized S3 domain.

    Derived with the exact identities() transform from the displacement policy
    constants. Stable V1 is never admitted: fresh stable arcs are V2 only.
    """
    from ._stiffened_shell_displacement_checkpoint import ELASTIC_S3_POLICY,STABLE_ELASTIC_S3_POLICY
    legacy=ELASTIC_S3_POLICY.replace('DISPLACEMENT_','TANGENT_PLANE_')
    stable=STABLE_ELASTIC_S3_POLICY.replace('DISPLACEMENT_','TANGENT_PLANE_')
    return (legacy,legacy.replace('_V1_DEV','_V2_DEV'),stable.replace('_V1_DEV','_V2_DEV'))


def _replay_s3_stable(owner,policy):
    """Bind arc replay S3 arithmetic to the snapshot's declared policy identity."""
    from ._stiffened_shell_ge_finite_experimental import supports_generalized_stable_s3
    if not supports_generalized_stable_s3(owner):
        return None
    legacy_v1,legacy_v2,stable_v2=_s3_arc_policies()
    if policy in (legacy_v1,legacy_v2):
        return False
    if policy==stable_v2:
        return True
    raise ValueError('unbound native S3 arc numerical policy')


def require_native_seed(owner,driver):
    """Admit ordinary/displacement seeds or this controller's validated prefix."""
    from ._stiffened_shell_affine_endpoint import require_native_displacement_policy
    require_native_displacement_policy(owner,driver)
    from ._stiffened_shell_affine_endpoint import is_native_material_bl5_arc
    bl5=is_native_material_bl5_arc(owner)
    if bl5:
        if driver._programme is not None or driver._external is not None or driver._adaptive_state is not None:
            raise ValueError('native material BL5 arc excludes staged/external/adaptive seeds')
        anchors=driver._state.shell_anchors
        saved=getattr(owner,'_displacement_snapshot',None)
        if (anchors is not None and not np.array_equal(
                anchors,np.tile(np.eye(3),(len(owner._shell_ids),1,1)))) or (
                saved is not None and (saved[1].get('chart_events',0)!=0
                                       or saved[1].get('trial_policy') is not None)):
            raise ValueError('native material BL5 arc requires an unrebased fixed-translation seed')
    snapshot=getattr(owner,'_arc_snapshot',None)
    if snapshot is None:
        if bl5:
            saved=getattr(owner,'_displacement_snapshot',None)
            if saved is not None:
                from ._stiffened_shell_displacement_checkpoint import identities as displacement_identities
                if saved[1]['policy']!=displacement_identities(owner)[1]:
                    raise ValueError('native material BL5 arc fixed-translation seed policy mismatch')
            else:
                applied=driver._load_factor*driver._pattern
                nodal=owner._shell_count+6*len(owner._beam_ids)
                if np.any(applied[nodal:]) or np.any(applied[:nodal].reshape(-1,6)[:,3:]):
                    raise ValueError('native material BL5 arc requires nodal dead translations')
        else:
            from ._stiffened_shell_displacement_adaptive import require_native_seed as displacement_seed
            displacement_seed(owner,driver)
        saved=getattr(owner,'_displacement_snapshot',None)
        if saved is not None and saved[1].get('trial_policy') is not None:
            raise ValueError('native arc excludes branch event seeds')
    else:
        m,a=decode_snapshot(owner,snapshot[0])
        if m['generation']!=owner._generation:
            raise ValueError('native arc seed generation mismatch')
        if not np.array_equal(driver._state.q,a['q']):
            raise ValueError('native arc seed coordinate mismatch')
    if driver._programme is not None or driver._external is not None or driver._adaptive_state is not None:
        raise ValueError('native arc excludes staged/external force seeds')


def identities(owner, *, native_version=2):
    from ._stiffened_shell_affine_endpoint import is_native_point_material,is_native_material_bl5_arc
    if is_native_material_bl5_arc(owner):
        from ._stiffened_shell_displacement_checkpoint import identities as displacement_identities
        kind,policy,_,_=displacement_identities(owner)
        return (kind.replace('finite_displacement_','finite_arc_'),
                policy.replace('DISPLACEMENT_','TANGENT_PLANE_'))
    if is_native_point_material(owner):
        from ._stiffened_shell_displacement_checkpoint import identities as displacement_identities
        kind,policy,_,_=displacement_identities(owner)
        if native_version not in (1,2):raise ValueError('native arc policy version')
        return (kind.replace('finite_displacement_','finite_arc_'),
                policy.replace('DISPLACEMENT_','TANGENT_PLANE_').replace('_V1_DEV',f'_V{native_version}_DEV'))
    if getattr(owner, '_curved_material', False):
        return 'finite_arc_curved_material', 'B3_GE_CURVED_MATERIAL_POINT_TANGENT_PLANE_V1_DEV'
    if getattr(owner, '_curved_reference', False):
        return 'finite_arc_curved_points', 'B3_GE_CURVED_ELASTIC_POINT_TANGENT_PLANE_V1_DEV'
    if owner._bl5_shell_material:
        return KIND, POLICY
    raise ValueError('admitted material or curved elastic point arc owner required')


def director_policy(owner):
    """Distinct outer controller/layout identity; no legacy policy upgrade."""
    from ._stiffened_shell_affine_endpoint import is_native_material_bl5_arc
    if not is_native_material_bl5_arc(owner):
        raise ValueError('director metric requires native material BL5')
    return identities(owner)[1].replace('TANGENT_PLANE_', 'POSITION_DIRECTOR_TANGENT_PLANE_')


def active_policy(owner,metric=None):
    """Fresh POINT seeds select V2; historical POINT prefixes keep their policy."""
    saved=getattr(owner,'_arc_snapshot',None)
    if saved is not None:return saved[1]['policy']
    return director_policy(owner) if type(metric) is NativeDirectorChordMetric else identities(owner)[1]


def requires_coordinate_progress(owner):
    if not _native_owner(owner):return False
    from ._stiffened_shell_affine_endpoint import is_native_material_bl5_arc
    policies=(identities(owner)[1],)
    if is_native_material_bl5_arc(owner):policies+=(director_policy(owner),)
    return active_policy(owner) in policies


def metric_from_snapshot(owner,metadata,arrays):
    """Reconstruct exactly the declared metric; never infer from normal data."""
    if metadata.get('metric_kind')==DIRECTOR_METRIC_ID:
        metric=NativeDirectorChordMetric(arrays['metric_weights'],metadata['metric_length'],
                                        metadata['metric_shell_nodes'])
        if metadata['policy']!=director_policy(owner):
            raise ValueError('arc director metric policy mismatch')
    else:
        metric=PhysicalChordMetric(arrays['metric_weights'],metadata['metric_length'])
    require_metric_owner(owner,metric)
    return metric


CONTROL_LIMITS=(('max_cutbacks',0,8),('max_attempts',1,64),
                ('max_iterations',1,24),('max_backtracks',0,12))
ATTEMPT_GROUP_KEYS={'requested_radius','amplitude_origin','attempts',
                    *(name for name,_,_ in CONTROL_LIMITS)}


def _validate_attempt_group(group):
    if (type(group) is not dict or set(group)!=ATTEMPT_GROUP_KEYS
            or any(type(group[k]) is not int or not low<=group[k]<=high
                   for k,low,high in CONTROL_LIMITS)
            or type(group['amplitude_origin']) not in (float,int)
            or not np.isfinite(group['amplitude_origin'])):
        raise ValueError('arc step controller contract')
    requested=group['requested_radius']
    if (type(requested) not in (float,int) or not np.isfinite(requested)
            or requested<=0. or not np.isfinite(requested*requested) or requested*requested<=0.):
        raise ValueError('arc requested radius')
    attempts=group['attempts']
    keys={'radius','accepted','evaluations','linear_solves','rejected_trials','event_cell_evaluations','failure'}
    if type(attempts) is not list or not 1<=len(attempts)<=min(group['max_cutbacks']+1,group['max_attempts']):
        raise ValueError('bounded arc step attempts required')
    for i,row in enumerate(attempts):
        if (type(row) is not dict or set(row)!=keys or type(row['accepted']) is not bool
                or row['accepted']!=(i==len(attempts)-1)
                or type(row['radius']) not in (float,int) or row['radius']!=requested*.5**i
                or not np.isfinite(row['radius']*row['radius']) or row['radius']*row['radius']<=0.
                or any(type(row[k]) is not int or not 0<=row[k]<=2**63-1 for k in
                       ('evaluations','linear_solves','rejected_trials','event_cell_evaluations'))
                or row['evaluations']<1 or row['linear_solves']<1
                or row['linear_solves']>group['max_iterations']+1
                or row['rejected_trials']>group['max_iterations']*(group['max_backtracks']+1)
                or row['evaluations']>2+group['max_iterations']*(group['max_backtracks']+1)
                or (row['accepted'] and row['failure'] is not None)
                or (not row['accepted'] and (type(row['failure']) is not str or not row['failure'] or len(row['failure'])>256))):
            raise ValueError('inconsistent arc attempt inventory')
    return attempts


def layout(owner,steps):
    # Reuse only the existing mechanical array layout, not its control fields.
    from ._stiffened_shell_displacement_checkpoint import _layout
    from ._stiffened_shell_affine_endpoint import has_shell_material
    mechanical=tuple((name,shape) for name,shape in _layout(owner,steps)
                     if name not in ('seed','history') and
                     (has_shell_material(owner) or name not in
                      ('shell_plastic','shell_alpha','shell_origin_plastic','shell_origin_alpha')))
    nodes=len(owner._shell_ids)+len(owner._beam_ids)
    return mechanical+(("history",(steps,2)),("metric_weights",(nodes,)),
        ("normal_translations",(nodes,3)),("normal_spins",(nodes,3)),
        ("origin_translations",(nodes,3)),("origin_frames",(nodes,3,3)))


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')


def encode_snapshot(owner,metadata,arrays):
    from ._stiffened_shell_affine_endpoint import has_native_s3_material,is_native_point_material
    s3=has_native_s3_material(owner)
    header=canonical(metadata)
    expected={name for name,_ in layout(owner,metadata['steps'])}
    if s3:expected.add('s3_states')
    if len(header)>262144 or set(arrays)!=expected:
        raise ValueError('bounded arc snapshot fields required')
    numeric=tuple((name,arrays[name]) for name,_ in layout(owner,metadata['steps']))
    # The native mechanical payload reuses the existing displacement codec;
    # the outer arc policy binds its distinct controller identity and layout.
    kind=identities(owner)[0]
    if _native_owner(owner):
        from ._stiffened_shell_displacement_checkpoint import identities as displacement_identities
        kind=displacement_identities(owner)[0]
    inner=(encode_s3(owner._identity,metadata['generation'],numeric,arrays['s3_states'],
                    native_material=True,displacement=True) if s3 else
           encode(kind,owner._identity,metadata['generation'],numeric))
    content=MAGIC+len(header).to_bytes(4,'little')+header+inner
    if len(content)+32>MAX_CHECKPOINT_BYTES:raise ValueError('arc snapshot size limit')
    raw=content+sha256(content).digest()
    decode_snapshot(owner,raw)
    return raw


def decode_snapshot(owner,raw):
    from ._stiffened_shell_affine_endpoint import is_native_point_material,has_native_s3_material
    native=_native_owner(owner);point=is_native_point_material(owner);s3=has_native_s3_material(owner)
    identities(owner)
    if (type(raw) is not bytes
            or not raw.startswith(MAGIC) or not len(MAGIC)+36<len(raw)<=MAX_CHECKPOINT_BYTES):
        raise ValueError('bounded coupled arc snapshot required')
    start=len(MAGIC)+4;length=int.from_bytes(raw[len(MAGIC):start],'little')
    if not 0<length<=262144 or start+length>=len(raw)-32:
        raise ValueError('arc metadata size')
    if sha256(memoryview(raw)[:-32]).digest()!=raw[-32:]:raise ValueError('arc snapshot digest')
    header=raw[start:start+length]
    try:m=json.loads(header.decode('ascii'))
    except (ValueError,UnicodeError,RecursionError) as e:raise ValueError('arc metadata') from e
    keys={'policy','model','generation','seed_generation','seed_sha256','steps',
          'metric_length','predictor_dof','orientation','requested_radius','attempts',
          'max_cutbacks','max_attempts','max_iterations','max_backtracks',
          'amplitude_origin','attempt_history','predecessor_sha256'}
    if native:keys.add('native_chart_policy')
    from ._stiffened_shell_affine_endpoint import is_native_material_bl5_arc
    director=(type(m) is dict and is_native_material_bl5_arc(owner)
              and m.get('policy')==director_policy(owner))
    if director:keys.update(('metric_kind','metric_shell_nodes'))
    from ._stiffened_shell_ge_finite_experimental import supports_generalized_stable_s3
    if point and supports_generalized_stable_s3(owner):
        # The generalized stable domain admits its restored legacy V1/V2
        # histories and the new stable V2 identity; the replay arithmetic is
        # bound later from the snapshot's declared policy.
        policies=_s3_arc_policies()
    else:
        policies=((identities(owner,native_version=1)[1],identities(owner)[1])
                  if point else (identities(owner)[1],))
    if is_native_material_bl5_arc(owner):policies+=(director_policy(owner),)
    if (type(m) is not dict or set(m)!=keys or canonical(m)!=header
            or m['policy'] not in policies or m['model']!=owner._identity
            or (native and m['native_chart_policy']!=_chart_policy(owner))):
        raise ValueError('incompatible arc metadata')
    if director and (m['metric_kind']!=DIRECTOR_METRIC_ID
            or type(m['metric_shell_nodes']) is not int or m['metric_shell_nodes']!=len(owner._shell_ids)):
        raise ValueError('incompatible arc director metric layout')
    if (any(type(m[k]) is not int for k in ('generation','seed_generation','steps','predictor_dof','orientation'))
            or not 1<=m['steps']<=1024 or not 1<=m['seed_generation']<m['generation']
            or m['generation']-m['seed_generation']!=m['steps'] or m['orientation'] not in (-1,1)):
        raise ValueError('arc history/generation mismatch')
    dof=m['predictor_dof'];nodes=len(owner._shell_ids)+len(owner._beam_ids)
    if not 0<=dof<6*nodes or dof%6>=3 or dof not in owner._free:
        raise ValueError('arc physical orientation coordinate')
    for name in ('seed_sha256','predecessor_sha256'):
        digest=m[name]
        if type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('arc predecessor/seed identity')
    if m['steps']==1 and m['predecessor_sha256']!=m['seed_sha256']:
        raise ValueError('first arc predecessor must be its seed')
    if type(m['requested_radius']) not in (float,int) or not np.isfinite(m['requested_radius']) or m['requested_radius']<=0.:
        raise ValueError('arc requested radius')
    if (type(m['metric_length']) not in (float,int) or
            any(type(m[k]) is not int or not low<=m[k]<=high for k,low,high in
                (('max_cutbacks',0,8),('max_attempts',1,64),('max_iterations',1,24),('max_backtracks',0,12)))):
        raise ValueError('arc controller/scaling contract')
    def shapes(kind,generation):
        expected=identities(owner)[0]
        if native:
            from ._stiffened_shell_displacement_checkpoint import identities as displacement_identities
            expected=displacement_identities(owner)[0]
        if kind!=expected or generation!=m['generation']:raise ValueError('arc mechanical identity')
        return layout(owner,m['steps'])
    if s3:
        def s3_shapes(generation):
            if generation!=m['generation']:raise ValueError('arc mechanical identity')
            return layout(owner,m['steps'])
        _,a,states=decode_s3(raw[start+length:-32],owner._identity,s3_shapes,
                             len(owner._definition.shell.s3_elements)*2,
                             native_material=True,displacement=True)
        a['s3_states']=states
    else:_,_,a=decode(raw[start+length:-32],owner._identity,shapes)
    metric=metric_from_snapshot(owner,m,a)
    groups=m['attempt_history']
    if type(groups) is not list or len(groups)!=m['steps']:
        raise ValueError('complete arc attempt history required')
    for index,group in enumerate(groups):
        rows=_validate_attempt_group(group)
        if ((index==0 and group['amplitude_origin']!=0.) or
                any(group[key]!=m[key] for key in
                    ('requested_radius','max_cutbacks','max_attempts','max_iterations','max_backtracks'))):
            raise ValueError('arc frozen controller/history mismatch')
        if rows[-1]['radius']!=a['history'][index,0]:
            raise ValueError('arc step radius/history mismatch')
        if index and group['amplitude_origin']!=a['history'][index-1,1]:
            raise ValueError('arc amplitude origin/history mismatch')
    if groups[-1]!={key:m[key] for key in ATTEMPT_GROUP_KEYS}:
        raise ValueError('arc final controls/history mismatch')
    attempts=m['attempts']
    radius,amplitude=a['history'][-1]
    if np.any(a['history'][:,0]<=0.) or radius!=attempts[-1]['radius']:
        raise ValueError('arc accepted radius/history mismatch')
    if not np.array_equal(a['base_load']+amplitude*a['load_direction'],a['applied_load']):
        raise ValueError('arc affine load mismatch')
    translations=a['normal_translations'];spins=a['normal_spins']
    if director:
        require_director_chart(a['q'][:owner._shell_count])
        if 'shell_origin_q' in a:require_director_chart(a['shell_origin_q'])
        norm=metric.tangent_norm_squared(a['origin_frames'],translations,spins)
    else:
        norm=np.sum(metric.weights*(np.sum((translations/metric.length)**2,axis=1)+np.sum(spins*spins,axis=1)))
    if not np.isfinite(norm) or abs(norm-1.)>1e-10:raise ValueError('arc physical normal is not unit')
    frames=np.concatenate((physical_frames(a['q'][:owner._shell_count],a['shell_anchors']),a['frames']))
    delta=a['q'][:nodes*6].reshape(-1,6)[:,:3]-a['origin_translations']
    if native and ((m['policy'] in _s3_arc_policies()[1:]
                    if supports_generalized_stable_s3(owner)
                    else m['policy']==identities(owner)[1]) or director):
        progress=m['orientation']*delta[dof//6,dof%6]
        if not progress>0.:raise ValueError('native arc selected coordinate made no oriented progress')
    distance,_,_=metric.plane(delta,frames,a['origin_frames'],translations,spins)
    if abs(distance-radius)/radius>1e-9:raise ValueError('arc path endpoint mismatch')
    return m,a


def from_selection(owner,selection,*,seed_generation,seed_sha256,history=(),attempt_history=()):
    from ._stiffened_shell_ge_finite_experimental import _material_arrays,_shell_material_arrays
    from ._stiffened_shell_affine_endpoint import has_shell_material,is_native_point_material,has_native_s3_material
    trial=selection.trial;state=trial.state;parent=trial.predecessor_state
    require_metric_owner(owner,trial.metric)
    with owner._commit_lock:
        driver=owner._finite_driver
        if (driver is None or owner._generation!=trial.accepted_generation
                or driver._state is not parent or driver._shell_origins is not trial.shell_predecessor
                or driver._beam_origins is not trial.beam_predecessor):
            raise ValueError('stale arc checkpoint predecessor')
        predecessor_sha256=sha256(owner.checkpoint()).hexdigest()
    nodes=len(owner._shell_ids)+len(owner._beam_ids)
    anchors=np.tile(np.eye(3),(len(owner._shell_ids),1,1)) if state.shell_anchors is None else state.shell_anchors
    arrays={k:getattr(state,k) for k in ('q','frames','cells','resultants','multipliers')}
    arrays.update(shell_anchors=anchors,shell_frames=physical_frames(state.q[:owner._shell_count],anchors),
        base_load=trial.base_load,load_direction=trial.load_direction,
        applied_load=trial.base_load+trial.amplitude*trial.load_direction,
        history=np.asarray(tuple(history)+((trial.radius,trial.amplitude),)),
        metric_weights=trial.metric.weights,normal_translations=trial.normal_translations,
        normal_spins=trial.normal_spins,origin_translations=parent.q[:nodes*6].reshape(-1,6)[:,:3],
        origin_frames=np.concatenate((physical_frames(parent.q[:owner._shell_count],parent.shell_anchors),parent.frames)))
    native=_native_owner(owner);s3=has_native_s3_material(owner)
    if native:
        from ._stiffened_shell_beam_history import arrays as native_arrays
        arrays.update(native_arrays(owner._beam_parts,trial.evaluation[5],'beam'))
        arrays.update(native_arrays(owner._beam_parts,trial.beam_predecessor,'beam_origin'))
    else:
        arrays['beam_plastic'],arrays['beam_accumulated']=_material_arrays(trial.evaluation[5])
        arrays['beam_origin_plastic'],arrays['beam_origin_accumulated']=_material_arrays(trial.beam_predecessor)
    if has_shell_material(owner):
        count=len(owner._definition.shell.q4_elements)
        arrays['shell_plastic'],arrays['shell_alpha']=_shell_material_arrays(trial.evaluation[4][:count] if s3 else trial.evaluation[4])
        arrays['shell_origin_plastic'],arrays['shell_origin_alpha']=_shell_material_arrays(trial.shell_predecessor[:count] if s3 else trial.shell_predecessor)
        if s3:
            from ._stiffened_shell_ge_finite_experimental import _accepted_shell_material_origins
            accepted=_accepted_shell_material_origins(owner,state,trial.evaluation[4])
            material=owner._shell_model.get_material('shell')
            arrays['s3_states']=tuple(element.serialize_nonlinear_state(owner._shell_model.mesh,material,origin,3)
                for origins in (accepted,trial.shell_predecessor)
                for (_,element,_,_),origin in zip(owner._shell_parts[count:],origins[count:]))
            arrays['shell_origin_q']=parent.q[:owner._shell_count]
            arrays['shell_origin_anchors']=(np.tile(np.eye(3),(len(owner._shell_ids),1,1))
                if parent.shell_anchors is None else parent.shell_anchors)
    group=dict(requested_radius=selection.requested_radius,amplitude_origin=trial.amplitude_origin,
        attempts=[asdict(row) for row in selection.attempts],
        max_cutbacks=selection.max_cutbacks,max_attempts=selection.max_attempts,
        max_iterations=selection.max_iterations,max_backtracks=selection.max_backtracks)
    m=dict(policy=active_policy(owner,trial.metric),model=owner._identity,generation=trial.accepted_generation+1,
        seed_generation=seed_generation,seed_sha256=seed_sha256,steps=len(history)+1,
        metric_length=trial.metric.length,predictor_dof=trial.predictor_dof,orientation=trial.orientation,
        predecessor_sha256=predecessor_sha256,
        attempt_history=list(attempt_history)+[group],**group)
    if native:m['native_chart_policy']=_chart_policy(owner)
    if type(trial.metric) is NativeDirectorChordMetric:
        m.update(metric_kind=DIRECTOR_METRIC_ID,metric_shell_nodes=trial.metric.shell_nodes)
    return encode_snapshot(owner,m,arrays)


def validate_endpoint(owner,raw):
    from ._stiffened_shell_affine_endpoint import validate_arrays,is_native_point_material
    metadata,arrays=decode_snapshot(owner,raw)
    return metadata,arrays,*validate_arrays(owner,arrays,
                                            native_chart=is_native_point_material(owner),
                                            s3_stable=_replay_s3_stable(owner,metadata['policy']))
