"""Event-policy persistence tests without mechanical evaluations."""
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import json
import numpy as np
import pytest

from anyfesolver._stiffened_shell_displacement_adaptive import (
    Context, MAGIC, EVENT_MAGIC, EVENT_POLICY, encode, decode,
    direction_hash, validate_progress)
from anyfesolver._stiffened_shell_displacement_increment import DisplacementIncrementState
from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl


def state():
    return DisplacementIncrementState(AdaptiveIncrementControl(2,8),(1.,),0.)


def owner():
    return SimpleNamespace(_identity="model",_bl5_shell_material=True,
        _shell_count=6,_beam_ids=(),_free=np.arange(6),
        _finite_driver=SimpleNamespace(_accepted=object(),_state=SimpleNamespace(q=np.zeros(6))))


def context(event=False):
    return Context(state(),0,direction_hash(np.ones(6)),0,20,12,
                   EVENT_POLICY if event else None)


def rewrite(raw,mutate):
    start=len(MAGIC)+4;length=int.from_bytes(raw[len(MAGIC):start],"little")
    value=json.loads(raw[start:start+length]);mutate(value)
    header=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")
    content=raw[:len(MAGIC)]+len(header).to_bytes(4,"little")+header+raw[start+length:-32]
    return content+sha256(content).digest()


def test_envelope_policy_binds_zero_work_and_preserves_legacy_fields():
    for event,magic in ((False,MAGIC),(True,EVENT_MAGIC)):
        c=context(event);raw=encode("model",c,b"force-seed")
        assert raw.startswith(magic)
        restored,inner=decode(owner(),raw)
        assert restored==c and inner==b"force-seed"
        assert encode("model",restored,inner)==raw
        assert restored.state.payload()["policy"]=="DISPLACEMENT_INCREMENT_V1"
        validate_progress(owner(),restored)
    with pytest.raises(ValueError):
        decode(owner(),rewrite(encode("model",context(),b"seed"),lambda v:v.update(event_work_start=0)))
    for mutation in (lambda v:v.update(trial_policy=None),
                     lambda v:v.update(event_work_start=True),
                     lambda v:v.update(event_work_start=-1),
                     lambda v:v.update(event_work_start=2**63)):
        with pytest.raises(ValueError):
            decode(owner(),rewrite(encode("model",context(True),b"seed"),mutation))
    for inner_context in (context(),context(True)):
        with pytest.raises(ValueError,match="nested"):
            decode(owner(),encode("model",context(True),encode("model",inner_context,b"seed")))


def test_attempt_payload_versions_and_failed_work():
    legacy=state().record(False,evaluations=1,linear_solves=1,rejected_trials=13)
    payload=legacy.payload()
    assert payload["policy"]=="DISPLACEMENT_INCREMENT_V1"
    assert "event_cell_evaluations" not in payload["attempts"][0]
    counted=state().record(False,evaluations=1,linear_solves=1,rejected_trials=13,event_cell_evaluations=2)
    assert counted.payload()["policy"]=="DISPLACEMENT_INCREMENT_V2"
    assert DisplacementIncrementState.from_payload(counted.payload())==counted
    payload["attempts"][0]["event_cell_evaluations"]=0
    with pytest.raises(ValueError):DisplacementIncrementState.from_payload(payload)
    for count in (-1,True,2**63):
        with pytest.raises(ValueError):
            state().record(False,evaluations=1,linear_solves=1,rejected_trials=0,event_cell_evaluations=count)


def test_accepted_suffix_event_sum_and_policy_match():
    accepted=state().record(True,evaluations=2,linear_solves=1,rejected_trials=0,event_cell_evaluations=4)
    c=replace(context(True),state=accepted,history_start=1,event_work_start=7)
    o=owner();metadata=dict(control_dof=0,trial_policy=EVENT_POLICY,event_cell_evaluations=11)
    arrays=dict(history=np.array([[0.,.2],[1.,.3]]),seed=np.array([-.1,0.]),load_direction=np.ones(6))
    o._displacement_snapshot=(b"raw",metadata,arrays)
    validate_progress(o,c)
    metadata["event_cell_evaluations"]=4
    with pytest.raises(ValueError,match="policy/work"):validate_progress(o,c)
    metadata.update(event_cell_evaluations=11,trial_policy=None)
    with pytest.raises(ValueError,match="policy/work"):validate_progress(o,c)


def test_no_event_work_before_first_history_row():
    accepted=state().record(True,evaluations=2,linear_solves=1,rejected_trials=0,event_cell_evaluations=4)
    c=replace(context(True),state=accepted,event_work_start=7)
    o=owner();o._displacement_snapshot=(b"raw",dict(control_dof=0,trial_policy=EVENT_POLICY,event_cell_evaluations=11),
        dict(history=np.array([[1.,.3]]),seed=np.array([0.,0.]),load_direction=np.ones(6)))
    with pytest.raises(ValueError,match="history mismatch"):validate_progress(o,c)
