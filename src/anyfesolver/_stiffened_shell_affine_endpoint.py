"""Private fixed-affine endpoint validation for admitted Q4 arc mechanics.

Decoding arrays or checkpoint framing is not mechanical acceptance. This module
replays the decoded endpoint in an isolated owner and driver. The final publisher
owns predecessor validation and the atomic live commit; this validator never
publishes state.
"""
from copy import copy
from threading import RLock
import numpy as np

from ._stiffened_shell_ge_finite_experimental import (
    FinitePointDriver,_State,_owned,_material_histories,_material_arrays,
    _shell_material_states,_shell_material_arrays)
from ._stiffened_shell_rotation_charts import owned_anchors,physical_frames


def is_native_point_material(owner):
    """Fixed-target native point route with admitted Q4/S3 material histories."""
    if getattr(owner,"_definition",None) is None:
        return False
    from ._stiffened_shell_q4_algorithmic_pullback import POLICY,HISTORICAL_POLICY
    definition=owner._definition
    return bool(definition.points and
        (definition.shell.q4_elements or definition.shell.s3_elements) and
        definition.shell_reference_policy=="flat" and
        definition.q4_pullback_policy==(POLICY if definition.shell.q4_elements else HISTORICAL_POLICY) and
        not definition.lines and not definition.footprints and
        not definition.distributing_footprints and not definition.finite_distributing_footprints)


def is_paired_displacement(owner):
    """Exact opted-in flat Q4-J2 mixed native POINT inventory; fixed shell control."""
    from ._stiffened_shell_ge_finite_experimental import supports_mixed_material_adaptive
    return supports_mixed_material_adaptive(owner)


def uses_native_material_history(owner):
    """Native station histories, without broadening POINT profile admission."""
    return is_native_point_material(owner) or bool(getattr(owner,"_bl5_native_material",False)) or bool(getattr(owner,"_distributing_native",False))


def is_native_material_bl5_arc(owner):
    """Explicit constructed native BL5 material domain; not a history predicate."""
    definition=getattr(owner,"_definition",None)
    return bool(definition is not None and getattr(owner,"_bl5_native_material",False)
        and getattr(owner,"_finite_bl5",False) and definition.lines
        and definition.bl5_material_policy in (
            "BL5_NATIVE_FIBRE_MATERIAL_V1_DEV",
            "BL5_NATIVE_FIBRE_MIXED_SHELL_MATERIAL_V1_DEV"))


def require_native_displacement_policy(owner,driver):
    """Never upgrade legacy S3 force-seed arithmetic during replay."""
    if (is_native_point_material(owner) and owner._definition.shell.s3_elements):
        stable=driver._stable_s3
        if type(stable) is not bool:
            raise ValueError("native S3 displacement seed numerical policy mismatch")
        from ._stiffened_shell_ge_finite_experimental import supports_generalized_stable_s3
        # The exact generalized stable domain admits both its restored legacy
        # False seed and the new stable True arithmetic; a restored legacy
        # seed is never upgraded. Every other owner keeps the pinned policy.
        if (not supports_generalized_stable_s3(owner) and
                stable != bool(owner._has_physical_fibre and
                              owner._definition.shell.j2_curve is None)):
            raise ValueError("native S3 displacement seed numerical policy mismatch")


def has_native_s3_material(owner):
    return bool((is_native_point_material(owner) or getattr(owner,"_bl5_native_mixed_material",False)) and owner._definition.shell.s3_elements
                and owner._definition.shell.j2_curve is not None)


def has_shell_material(owner):
    """Only admitted algorithmic shell histories belong in affine endpoints."""
    return owner._bl5_shell_material or bool(getattr(owner,"_distributing_shell_material",False)) or (uses_native_material_history(owner) and
        owner._definition.shell.j2_curve is not None) or (
        getattr(owner, "_curved_material", False)
        and owner._definition.shell.j2_curve is not None)


def validate_arrays(owner,a,*,native_chart=False,s3_stable=None):
    """Replay decoded fixed-affine translation arrays in isolated mechanics.

    The replay S3 seed arithmetic is checkpoint-bound: ``s3_stable`` carries the
    snapshot's declared numerical-policy identity, never the fresh constructor
    selection. A legacy identity (False) replays the original arithmetic; a
    new stable identity (True) refuses atomically unless the constructor
    admitted exactly that policy. ``None`` keeps the constructor selection
    for every route whose metadata does not bind the generalized policy.
    """
    if not (owner._bl5_shell_material or getattr(owner,"_curved_reference",False) or uses_native_material_history(owner) or getattr(owner,"_finite_distributing",False)):
        raise ValueError("bounded coupled affine endpoint required")
    prepared=copy(owner);prepared._commit_lock=RLock();prepared._writing=False
    prepared._finite_driver=None;prepared._accepted=None
    if getattr(owner,"_finite_distributing",False):
        from ._stiffened_shell_ge_distributing_finite import PreparedFiniteDistributingAttachment
        # Bind scratch operators to this isolated owner. Immutable geometry,
        # pose and multiplier row starts are reused; live bindings stay intact.
        generation=prepared._generation
        prepared._generation=0
        try:
            prepared._finite_distributing_bindings=tuple((
                PreparedFiniteDistributingAttachment._from_compiled(
                    prepared,adapter.reference,adapter.pose),start)
                for adapter,start in owner._finite_distributing_bindings)
        finally:
            prepared._generation=generation
    if getattr(owner,"_external_context",None) is not None:
        raise ValueError("displacement snapshot excludes external load contexts")
    driver=FinitePointDriver(prepared)
    if s3_stable is not None:
        # The snapshot's policy identity owns the replay arithmetic. A legacy
        # identity never upgrades to the constructor's new generalized stable
        # selection; a new stable identity must match it exactly or refuse
        # atomically before any replay state exists.
        from ._stiffened_shell_ge_finite_experimental import supports_generalized_stable_s3
        if (type(s3_stable) is not bool or not supports_generalized_stable_s3(owner)
                or type(driver._stable_s3) is not bool
                or (s3_stable and driver._stable_s3 is not True)):
            raise ValueError("native S3 displacement seed numerical policy mismatch")
        if not s3_stable:
            driver._stable_s3=False
    n=owner._shell_count+6*len(owner._beam_ids)
    q=_owned(a["q"])
    if (np.any(q[owner._fixed]!=0.) or np.any(q[n:]!=0.) or
            np.any(q[owner._shell_count:n].reshape(-1,6)[:,3:]!=0.)):
        raise ValueError("invalid displacement coordinate/support layout")
    for name in ("base_load","load_direction","applied_load"):
        if np.any(a[name][n:]) or np.any(a[name][:n].reshape(-1,6)[:,3:]):
            raise ValueError("displacement snapshot requires nodal dead translations")
    if not np.any(a["load_direction"][owner._free]):
        raise ValueError("nonzero free displacement load direction required")
    if (getattr(owner,"_finite_distributing",False) and owner._distributing_cyclic and
            not np.array_equal(a["load_direction"],owner._cyclic_reference_pattern)):
        raise ValueError("cyclic distributing displacement requires the model-bound reference load direction")
    if (getattr(owner,"_finite_distributing",False) and owner._distributing_cyclic and
            not np.array_equal(a["base_load"],float(a["cyclic_seed_factor"][0])*owner._cyclic_reference_pattern)):
        raise ValueError("cyclic distributing displacement base must match its stored seed force factor")
    anchors=owned_anchors(a["shell_anchors"],len(owner._shell_ids))
    if getattr(owner,"_finite_distributing",False) and (native_chart or not np.array_equal(
            anchors,np.tile(np.eye(3),(len(owner._shell_ids),1,1)))):
        raise ValueError("finite distributing displacement excludes rebased shell charts")
    if uses_native_material_history(owner) and not native_chart and not np.array_equal(
            anchors,np.tile(np.eye(3),(len(owner._shell_ids),1,1))):
        raise ValueError("native point displacement excludes rebased shell charts")
    witness=owned_anchors(a["shell_frames"],len(owner._shell_ids))
    if not np.allclose(physical_frames(q[:owner._shell_count],anchors),witness,
                       rtol=0.,atol=1e-11):
        raise ValueError("displacement shell pose witness mismatch")
    fixed=set(owner._fixed)
    for i,anchor in enumerate(anchors):
        if any(6*i+j in fixed for j in (3,4,5)) and np.linalg.norm(anchor-np.eye(3))>1e-12:
            raise ValueError("displacement anchor violates rotational support")
    for name in ("frames","cells"):
        for frame in a[name].reshape(-1,3,3):
            if np.linalg.norm(frame.T@frame-np.eye(3))>1e-9 or abs(np.linalg.det(frame)-1.)>1e-9:
                raise ValueError("nonphysical displacement rotation")
    for name in (("shell_alpha","shell_origin_alpha") if has_shell_material(owner) else ()):
        if np.any(a[name]<0.):raise ValueError("nonphysical shell J2 history")
    shell_parent=(_shell_material_states(a["shell_origin_plastic"],a["shell_origin_alpha"])
                  if has_shell_material(owner) else driver._shell_origins)
    s3_material=has_native_s3_material(owner)
    if s3_material:
        predecessor_anchors=owned_anchors(a["shell_origin_anchors"],len(owner._shell_ids))
        if not native_chart and not np.array_equal(predecessor_anchors,
                              np.tile(np.eye(3),(len(owner._shell_ids),1,1))):
            raise ValueError("native point displacement excludes rebased predecessor charts")
        from ._stiffened_shell_ge_finite_experimental import _restore_native_s3_histories
        accepted_s3, predecessor_s3, origin_q, origin_anchors = _restore_native_s3_histories(
            prepared, q, anchors, a)
        shell_parent += predecessor_s3
        driver._snapshot_s3_accepted = accepted_s3
        driver._shell_predecessor_q = origin_q
        driver._shell_predecessor_anchors = origin_anchors
    native=uses_native_material_history(owner)
    if native:
        from ._stiffened_shell_beam_history import histories
        beam_parent=histories(prepared._beam_parts,a,"beam_origin")
        histories(prepared._beam_parts,a,"beam")
    else:
        beam_parent=_material_histories(prepared,a["beam_origin_plastic"],a["beam_origin_accumulated"])
        # Validate accepted beam history admissibility before replay comparison.
        beam_accepted=_material_histories(prepared,a["beam_plastic"],a["beam_accumulated"])
        if getattr(owner,"_finite_distributing",False) and not owner._distributing_material:
            from ._stiffened_shell_ge_finite_experimental import _require_elastic_footprint_histories
            _require_elastic_footprint_histories(beam_parent,"displacement predecessor",scope="finite distributing footprints")
            _require_elastic_footprint_histories(beam_accepted,"displacement accepted",scope="finite distributing footprints")
    state=_State(q,_owned(a["frames"]),_owned(a["cells"]),
                 _owned(a["resultants"]),_owned(a["multipliers"]),anchors)
    if is_paired_displacement(owner):
        from dataclasses import replace
        from ._stiffened_shell_f06_increment_arithmetic import validate_pairs,validate_frames
        words={}
        for name in ("q_low","frames_low","cells_low","resultants_low"):
            high=getattr(state,name.removesuffix("_low"));low=_owned(a[name],high.shape)
            (validate_frames if name in ("frames_low","cells_low") else validate_pairs)(high,low,name)
            words[name]=low
        translated={owner._shell_count+6*i+j for i in range(len(owner._beam_ids)) for j in range(3)}
        if any(v!=0. for i,v in enumerate(words["q_low"]) if i not in translated or i in owner._fixed):
            raise ValueError("paired displacement only admits free native translation corrections")
        state=replace(state,**words)
    evaluation=driver._assemble(state,0.,shell_parent,beam_parent,applied_load=a["applied_load"])
    if not np.isfinite(evaluation[2]) or evaluation[2]>1e-8 or evaluation[3]>1e-9:
        raise ValueError("mechanically inconsistent displacement equilibrium")
    if native:
        from ._stiffened_shell_beam_history import arrays
        comparisons=list(arrays(prepared._beam_parts,evaluation[5],"beam"))
    else:
        bp,ba=_material_arrays(evaluation[5])
        comparisons=[("beam_plastic",bp),("beam_accumulated",ba)]
    if has_shell_material(owner):
        q4_count = len(owner._definition.shell.q4_elements)
        sp,sa=_shell_material_arrays(evaluation[4][:q4_count] if s3_material else evaluation[4])
        comparisons.extend((("shell_plastic",sp),("shell_alpha",sa)))
        if s3_material:
            from ._stiffened_shell_ge_finite_experimental import _require_native_s3_replay
            saved = _shell_material_states(a["shell_plastic"],a["shell_alpha"]) + accepted_s3
            _require_native_s3_replay(prepared, state, saved, evaluation[4])
    for name,repeated in comparisons:
        saved=a[name]
        if native and not name.startswith("shell_"):
            if not np.array_equal(saved,repeated):
                raise ValueError("displacement native material replay mismatch")
            continue
        scale=max(float(np.max(abs(saved),initial=0.)),
                  float(np.max(abs(repeated),initial=0.)),np.finfo(float).tiny)
        if np.max(abs(saved-repeated),initial=0.)>64*np.finfo(float).eps*scale:
            raise ValueError("displacement material replay mismatch")
    # Scratch recovery belongs only to this prepared driver. The publisher must
    # seal histories, build results/checkpoint and recheck its live predecessor.
    return driver,state,evaluation,shell_parent,beam_parent
