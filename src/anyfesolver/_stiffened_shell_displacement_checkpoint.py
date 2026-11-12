"""Private displacement snapshot framing; decoding is NOT mechanical acceptance.

No force factors are fabricated. Generation includes chart events; history rows
contain only actual accepted displacement steps (target, amplitude). The seed is
an authentic earlier accepted endpoint identified by its checkpoint digest.
"""
import json
from hashlib import sha256
import numpy as np
from ._stiffened_shell_ge_checkpoint_binary import encode,decode,encode_s3,decode_s3,MAX_CHECKPOINT_BYTES

MAGIC=b"ANYSSDC1"
PAIRED_MAGIC=b"ANYSSPD1"
PAIRED_KIND="finite_displacement_paired_mixed_q4_coupled"
PAIRED_POLICY="GE_PAIRED_MIXED_Q4_J2_POINT_DISPLACEMENT_V1_DEV"
PAIRED_CHART_POLICY="GE_PAIRED_MIXED_Q4_J2_POINT_DISPLACEMENT_CHART_V1_DEV"
KIND="finite_displacement_bl5_coupled"
POLICY="B3_GE_BL5_COUPLED_DISPLACEMENT_CONTROL_V1_DEV"
EVENT_POLICY="B3_GE_BL5_COUPLED_DISPLACEMENT_CONTROL_EVENT_V3_DEV"
from ._stiffened_shell_branch_event_trial import POLICY as EVENT_TRIAL_POLICY
CHART_POLICY="B3_GE_BL5_COUPLED_DISPLACEMENT_CONTROL_CHART_V2_DEV"
DISTRIBUTING_KIND="finite_distributing_displacement"
DISTRIBUTING_POLICY="B3_GE_DISTRIBUTING_DISPLACEMENT_V1_DEV"
ELASTIC_S3_POLICY="B3_GE_NATIVE_POINT_ELASTIC_S3_DISPLACEMENT_V1_DEV"
ELASTIC_S3_CHART_POLICY="B3_GE_NATIVE_POINT_ELASTIC_S3_DISPLACEMENT_CHART_V1_DEV"
STABLE_ELASTIC_S3_POLICY="B3_GE_NATIVE_POINT_ELASTIC_S3_DISPLACEMENT_STABLE_V1_DEV"
STABLE_ELASTIC_S3_CHART_POLICY="B3_GE_NATIVE_POINT_ELASTIC_S3_DISPLACEMENT_STABLE_CHART_V1_DEV"


def generalized_stable_live(owner):
    """True when the owner's live driver runs the new generalized stable S3 arithmetic.

    This is the encoding mode only: a stable-live owner frames new snapshots
    with the new private policy identity, while a restored legacy driver keeps
    emitting the exact old identities and bytes. No live driver is fail-closed
    legacy. Replay arithmetic is bound separately by the snapshot's policy.
    """
    from ._stiffened_shell_ge_finite_experimental import supports_generalized_stable_s3
    driver=getattr(owner,"_finite_driver",None)
    return bool(supports_generalized_stable_s3(owner) and driver is not None
        and type(getattr(driver,"_stable_s3",None)) is bool and driver._stable_s3)


def admitted_s3_policies():
    """(base policy, chart policy, replay stable flag) for the generalized S3 domain.

    The legacy identity keeps its original arithmetic (False); the new stable
    identity binds the generalized stable selection (True). Both are admitted
    for decoding so restored legacy histories replay without upgrade.
    """
    return ((ELASTIC_S3_POLICY,ELASTIC_S3_CHART_POLICY,False),
            (STABLE_ELASTIC_S3_POLICY,STABLE_ELASTIC_S3_CHART_POLICY,True))


def replay_s3_stable(owner,policy):
    """Bind the replay S3 arithmetic to the snapshot's declared policy identity."""
    from ._stiffened_shell_ge_finite_experimental import supports_generalized_stable_s3
    if not supports_generalized_stable_s3(owner):
        return None
    for base,chart,stable in admitted_s3_policies():
        if policy in (base,chart):
            return stable
    raise ValueError("unbound native S3 displacement numerical policy")


def identities(owner):
    """Bind the controller and mechanical payload to their admitted profile."""
    from ._stiffened_shell_affine_endpoint import is_native_point_material,is_paired_displacement
    if is_paired_displacement(owner):
        return PAIRED_KIND,PAIRED_POLICY,PAIRED_CHART_POLICY,None
    if getattr(owner,"_finite_distributing",False):
        return DISTRIBUTING_KIND,DISTRIBUTING_POLICY,None,None
    if getattr(owner,"_bl5_native_mixed_material",False):
        return ("finite_displacement_native_fibre_bl5_mixed_shell",
                "B3_GE_NATIVE_FIBRE_BL5_MIXED_SHELL_J2_DISPLACEMENT_V1_DEV",None,None)
    if getattr(owner,"_bl5_native_material",False):
        return ("finite_displacement_native_fibre_bl5",
                "B3_GE_NATIVE_FIBRE_BL5_DISPLACEMENT_V1_DEV",None,None)
    if is_native_point_material(owner):
        if owner._definition.shell.s3_elements:
            if owner._definition.shell.j2_curve is not None:
                return ("finite_displacement_native_point_plastic_s3",
                        "B3_GE_NATIVE_POINT_SHELL_J2_S3_DISPLACEMENT_V1_DEV","B3_GE_NATIVE_POINT_SHELL_J2_S3_DISPLACEMENT_CHART_V1_DEV",None)
            if owner._has_physical_fibre:
                return ("finite_displacement_native_point_fibre_elastic_s3",
                        "B3_GE_NATIVE_POINT_FIBRE_ELASTIC_S3_DISPLACEMENT_V1_DEV","B3_GE_NATIVE_POINT_FIBRE_ELASTIC_S3_DISPLACEMENT_CHART_V1_DEV",None)
            # The plain elastic S3 route binds its live numerical policy
            # identity: the new generalized stable arithmetic frames new
            # private identities, while a restored legacy driver keeps the
            # exact old strings and bytes.
            if generalized_stable_live(owner):
                return ("finite_displacement_native_point_elastic_s3",
                        STABLE_ELASTIC_S3_POLICY,STABLE_ELASTIC_S3_CHART_POLICY,None)
            return ("finite_displacement_native_point_elastic_s3",
                    ELASTIC_S3_POLICY,ELASTIC_S3_CHART_POLICY,None)
        if owner._definition.shell.j2_curve is not None:
            return ("finite_displacement_native_point_coupled",
                    "B3_GE_NATIVE_POINT_COUPLED_J2_DISPLACEMENT_V1_DEV","B3_GE_NATIVE_POINT_COUPLED_J2_DISPLACEMENT_CHART_V1_DEV",None)
        return ("finite_displacement_native_point_material",
                "B3_GE_NATIVE_POINT_MATERIAL_DISPLACEMENT_V1_DEV","B3_GE_NATIVE_POINT_MATERIAL_DISPLACEMENT_CHART_V1_DEV",None)
    if getattr(owner,"_curved_material",False):
        return ("finite_displacement_curved_material",
                "B3_GE_CURVED_MATERIAL_POINT_DISPLACEMENT_V1_DEV",
                "B3_GE_CURVED_MATERIAL_POINT_DISPLACEMENT_CHART_V1_DEV",
                "B3_GE_CURVED_MATERIAL_POINT_DISPLACEMENT_EVENT_V1_DEV")
    if owner._bl5_shell_material:
        return KIND,POLICY,CHART_POLICY,EVENT_POLICY
    raise ValueError("admitted coupled BL5 or curved material displacement owner required")


def _layout(owner,steps):
    from ._stiffened_shell_affine_endpoint import has_native_s3_material
    n=owner._K.shape[0];members=len(owner._beam_parts)
    shells=(len(owner._definition.shell.q4_elements) if has_native_s3_material(owner)
            else len(owner._shell_parts))
    stations=sum(len(part[0].operator.cell.stations) for part in owner._beam_parts)
    layout=(("q",(n,)),("frames",(len(owner._beam_ids),3,3)),
        ("cells",(members,2,3,3)),("resultants",(members,18)),
        ("multipliers",(owner._C.shape[0],)),
        ("shell_anchors",(len(owner._shell_ids),3,3)),
        ("shell_frames",(len(owner._shell_ids),3,3)),
        ("base_load",(n,)),("load_direction",(n,)),("applied_load",(n,)),
        ("seed",(2,)),("history",(steps,2)),
        ("beam_plastic",(stations,6,2)),("beam_accumulated",(stations,2)),
        ("beam_origin_plastic",(stations,6,2)),("beam_origin_accumulated",(stations,2)),
        ("shell_plastic",(shells,12,3)),("shell_alpha",(shells,12)),
        ("shell_origin_plastic",(shells,12,3)),("shell_origin_alpha",(shells,12)))
    from ._stiffened_shell_affine_endpoint import has_shell_material
    from ._stiffened_shell_affine_endpoint import uses_native_material_history
    if uses_native_material_history(owner):
        from ._stiffened_shell_beam_history import shapes
        native=layout[:12]+shapes(owner._beam_parts,"beam")+shapes(owner._beam_parts,"beam_origin")
        result=native+(layout[-4:] if has_shell_material(owner) else ())
        if has_native_s3_material(owner):
            result += (("shell_origin_q",(owner._shell_count,)),
                       ("shell_origin_anchors",(len(owner._shell_ids),3,3)))
        if getattr(owner,"_finite_distributing",False) and owner._distributing_cyclic:
            result += (("cyclic_seed_factor",(1,)),)
        from ._stiffened_shell_affine_endpoint import is_paired_displacement
        if is_paired_displacement(owner):
            result += (("q_low",(n,)),("frames_low",(len(owner._beam_ids),3,3)),
                       ("cells_low",(members,2,3,3)),("resultants_low",(members,18)))
        return result
    if (getattr(owner,"_curved_material",False) or getattr(owner,"_finite_distributing",False)) and not has_shell_material(owner):
        layout=tuple(item for item in layout if item[0] not in (
            "shell_plastic","shell_alpha","shell_origin_plastic","shell_origin_alpha"))
    if getattr(owner,"_finite_distributing",False) and owner._distributing_cyclic:
        layout += (("cyclic_seed_factor",(1,)),)
    return layout


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")


def encode_snapshot(owner, *, generation, seed_generation, seed_sha256,
                    control_dof, arrays, chart_events=0, trial_policy=None, event_cell_evaluations=0):
    """Frame owned endpoint arrays, without conferring mechanical validity."""
    if type(chart_events) is not int or chart_events < 0:
        raise ValueError("nonnegative displacement chart event count required")
    if (trial_policy not in (None,EVENT_TRIAL_POLICY) or
            type(event_cell_evaluations) is not int or not 0<=event_cell_evaluations<=2**63-1 or
            (trial_policy is None and event_cell_evaluations!=0)):
        raise ValueError("invalid displacement trial policy/work")
    kind,policy,chart_policy,event_policy=identities(owner)
    if (chart_policy is None and chart_events) or (event_policy is None and trial_policy is not None):
        raise ValueError("native point displacement excludes chart/event policies")
    steps=len(arrays["history"])
    metadata=dict(policy=policy,model=owner._identity,generation=generation,
        seed_generation=seed_generation,seed_sha256=seed_sha256,
        control_dof=control_dof,steps=steps)
    if chart_events:
        metadata.update(policy=chart_policy,chart_events=chart_events)
    if trial_policy is not None:
        metadata.update(policy=event_policy,trial_policy=trial_policy,
                        chart_events=chart_events,event_cell_evaluations=event_cell_evaluations)
    header=_canonical(metadata)
    from ._stiffened_shell_affine_endpoint import has_native_s3_material
    s3_material=has_native_s3_material(owner)
    expected={name for name,_ in _layout(owner,steps)}
    if s3_material:expected.add("s3_states")
    paired=kind==PAIRED_KIND
    if paired:expected.add("force_seed")
    if len(header)>1024 or set(arrays)!=expected:
        raise ValueError("invalid displacement snapshot fields")
    numeric=tuple((name,arrays[name]) for name,_ in _layout(owner,steps))
    inner=(encode_s3(owner._identity,generation,numeric,arrays["s3_states"],
                    native_material=True,displacement=True,
                    mixed_shell_material=getattr(owner,"_bl5_native_mixed_material",False)) if s3_material else
           encode(kind,owner._identity,generation,numeric))
    if paired:
        seed=arrays["force_seed"]
        if type(seed) is not bytes or sha256(seed).hexdigest()!=seed_sha256 or not 0<len(seed)<MAX_CHECKPOINT_BYTES:
            raise ValueError("exact bounded paired force seed required")
        inner=len(seed).to_bytes(8,"little")+seed+inner
    content=(PAIRED_MAGIC if paired else MAGIC)+len(header).to_bytes(4,"little")+header+inner
    if len(content)+32>MAX_CHECKPOINT_BYTES:
        raise ValueError("displacement snapshot exceeds resource limit")
    raw=content+sha256(content).digest()
    decode_snapshot(owner,raw)
    return raw


def decode_snapshot(owner,raw):
    """Validate framing/layout/controller consistency; caller must replay mechanics.

    Returned byte-backed arrays are read-only. Caller must additionally validate
    physical frames, supports, material state, equilibrium and endpoint work before
    atomic publication in an isolated restoration owner.
    """
    kind,policy,chart_policy,event_policy=identities(owner)
    base_policies={policy}
    chart_policies={chart_policy} if chart_policy is not None else set()
    from ._stiffened_shell_ge_finite_experimental import supports_generalized_stable_s3
    if supports_generalized_stable_s3(owner):
        # The generalized stable domain admits both its restored legacy
        # arithmetic identity and the new stable identity; the replay
        # arithmetic is bound later from the snapshot's declared policy.
        for base,chart,_ in admitted_s3_policies():
            base_policies.add(base);chart_policies.add(chart)
    magic=PAIRED_MAGIC if kind==PAIRED_KIND else MAGIC
    if (type(raw) is not bytes or
            not raw.startswith(magic) or not len(MAGIC)+4+32<len(raw)<=MAX_CHECKPOINT_BYTES):
        raise ValueError("bounded coupled displacement snapshot required")
    start=len(MAGIC)+4;length=int.from_bytes(raw[len(MAGIC):start],"little")
    if not 0<length<=1024 or start+length>=len(raw)-32:
        raise ValueError("invalid displacement metadata bound")
    if sha256(memoryview(raw)[:-32]).digest()!=raw[-32:]:
        raise ValueError("displacement snapshot digest mismatch")
    try:
        header=raw[start:start+length];m=json.loads(header.decode("ascii"))
        canonical=_canonical(m)
    except (ValueError,UnicodeError,RecursionError) as error:
        raise ValueError("invalid displacement metadata") from error
    base_keys={"policy","model","generation","seed_generation",
               "seed_sha256","control_dof","steps"}
    if type(m) is not dict:
        raise ValueError("incompatible displacement metadata")
    chart_events=m.get("chart_events",0)
    if chart_policy is None and (m.get("policy")!=policy or set(m)!=base_keys):
        raise ValueError("native point displacement excludes chart/event policies")
    if (canonical!=header or m.get("model")!=owner._identity or
            not ((m.get("policy") in base_policies and set(m)==base_keys) or
                 (m.get("policy") in chart_policies and set(m)==base_keys|{"chart_events"}
                  and type(chart_events) is int and chart_events>0) or
                 (event_policy is not None and m.get("policy")==event_policy and
                  set(m)==base_keys|{"chart_events","trial_policy","event_cell_evaluations"} and
                  m["trial_policy"]==EVENT_TRIAL_POLICY and
                  type(chart_events) is int and chart_events>=0 and
                  type(m["event_cell_evaluations"]) is int and
                  0<=m["event_cell_evaluations"]<=2**63-1))):
        raise ValueError("incompatible displacement metadata")
    if (any(type(m[k]) is not int for k in ("generation","seed_generation","control_dof","steps")) or
            not 1<=m["seed_generation"]<=m["generation"] or
            not 1<=m["steps"]==m["generation"]-m["seed_generation"]-chart_events or
            m["generation"]>MAX_CHECKPOINT_BYTES//8):
        raise ValueError("invalid displacement generation/history")
    dof=m["control_dof"];nodal=owner._shell_count+6*len(owner._beam_ids)
    if not 0<=dof<nodal or dof%6>=3 or dof not in owner._free:
        raise ValueError("free physical translation control required")
    digest=m["seed_sha256"]
    if type(digest) is not str or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("canonical seed checkpoint digest required")
    inner=raw[start+length:-32]
    seed=None
    if kind==PAIRED_KIND:
        if dof>=owner._shell_count:
            raise ValueError("paired displacement controls shell translations only")
        seed_length=int.from_bytes(inner[:8],"little")
        if not 0<seed_length<len(inner)-8:
            raise ValueError("invalid paired force seed length")
        seed=inner[8:8+seed_length];inner=inner[8+seed_length:]
        if sha256(seed).hexdigest()!=digest:
            raise ValueError("paired force seed digest mismatch")
        from ._stiffened_shell_ge_checkpoint_binary import COMPENSATED_MAGIC
        if not seed.startswith(COMPENSATED_MAGIC):
            raise ValueError("paired displacement requires original compensated force seed")
    def shapes(decoded_kind,generation):
        if decoded_kind!=kind or generation!=m["generation"]:
            raise ValueError("displacement mechanical identity mismatch")
        return _layout(owner,m["steps"])
    from ._stiffened_shell_affine_endpoint import has_native_s3_material
    if has_native_s3_material(owner):
        def s3_shapes(generation):
            if generation != m["generation"]:
                raise ValueError("displacement mechanical identity mismatch")
            return _layout(owner,m["steps"])
        _,a,states=decode_s3(raw[start+length:-32],owner._identity,s3_shapes,
                            2*len(owner._definition.shell.s3_elements),
                            native_material=True,displacement=True,
                            mixed_shell_material=getattr(owner,"_bl5_native_mixed_material",False))
        a["s3_states"]=states
    else:
        _,_,a=decode(inner,owner._identity,shapes)
    if seed is not None:a["force_seed"]=seed
    if np.any(np.diff(np.r_[a["seed"][0],a["history"][:,0]])==0.):
        raise ValueError("displacement history targets must change")
    target,amplitude=a["history"][-1]
    if not np.array_equal(a["base_load"]+amplitude*a["load_direction"],a["applied_load"]):
        raise ValueError("displacement affine load mismatch")
    if abs(a["q"][dof]-target)/owner._reference_length>1e-9:
        raise ValueError("displacement target mismatch")
    return m,a



def validate_endpoint(owner,raw):
    """Replay an endpoint in an isolated driver; never publish to the caller.

    Returns private prepared mechanics and decoded controller data. This is not
    a public restore operation and does not authenticate the loading path.
    """
    from ._stiffened_shell_affine_endpoint import validate_arrays
    metadata,a=decode_snapshot(owner,raw)
    paired=metadata["policy"] in (PAIRED_POLICY,PAIRED_CHART_POLICY)
    if paired:
        _validate_force_seed(owner,metadata,a)
    return metadata,a,*validate_arrays(owner,a,
        native_chart=paired or metadata.get("chart_events",0)>0,
        s3_stable=replay_s3_stable(owner,metadata["policy"]))


def _validate_force_seed(owner,metadata,a):
    """One strict same-endpoint force restore; never replay its load trajectory."""
    from copy import copy
    from threading import RLock
    from ._stiffened_shell_ge_checkpoint_binary import COMPENSATED_MAGIC
    seed=a["force_seed"]
    if not seed.startswith(COMPENSATED_MAGIC):
        raise ValueError("original paired force checkpoint required")
    prepared=copy(owner);prepared._commit_lock=RLock();prepared._writing=False
    prepared._finite_driver=None;prepared._accepted=None;prepared._generation=0
    prepared._displacement_snapshot=None;prepared._displacement_adaptive=None;prepared._arc_snapshot=None
    # Refuse controller nesting or non-force seed domains before recursive restore.
    from ._stiffened_shell_ge_checkpoint_binary import decode_compensated,MAGIC as FORCE_MAGIC,CHART_MAGIC
    from ._stiffened_shell_increment_control import MAGIC as ADAPTIVE_MAGIC,decode_control
    shapes=(("q_low",(owner._K.shape[0],)),("frames_low",(len(owner._beam_ids),3,3)),
            ("cells_low",(len(owner._beam_parts),2,3,3)),("resultants_low",(len(owner._beam_parts),18)))
    inner,_=decode_compensated(seed,owner._identity,shapes)
    if inner.startswith(ADAPTIVE_MAGIC):_,inner=decode_control(inner,owner._identity)
    if not inner.startswith((FORCE_MAGIC,CHART_MAGIC)):
        raise ValueError("paired displacement seed must be an ordinary or adaptive force endpoint")
    prepared.restore(seed)
    driver=prepared._finite_driver
    if (driver is None or getattr(prepared,"_displacement_snapshot",None) is not None or
            driver._programme is not None or driver._external is not None or
            prepared._generation!=metadata["seed_generation"] or
            not np.array_equal(driver._pattern,a["load_direction"]) or
            not np.array_equal(driver._load_factor*driver._pattern,a["base_load"]) or
            not np.array_equal(np.array([driver._state.q[metadata["control_dof"]],0.]),a["seed"])):
        raise ValueError("paired displacement original force provenance mismatch")
    if metadata["steps"]==1:
        from ._stiffened_shell_beam_history import arrays as history_arrays
        from ._stiffened_shell_ge_finite_experimental import _shell_material_arrays
        expected=list(history_arrays(owner._beam_parts,driver._beam_origins,"beam_origin"))
        plastic,alpha=_shell_material_arrays(driver._shell_origins)
        expected += [("shell_origin_plastic",plastic),("shell_origin_alpha",alpha)]
        if any(not np.array_equal(a[name],value) for name,value in expected):
            raise ValueError("first paired displacement predecessor must equal original accepted force history")
    return driver
