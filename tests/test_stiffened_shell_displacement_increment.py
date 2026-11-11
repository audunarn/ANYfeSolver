"""Signed cutback accounting and durable metadata consistency."""
import json
import math
from copy import deepcopy
import pytest
from anyfesolver._stiffened_shell_displacement_increment import DisplacementIncrementState as State,DisplacementIncrementLimit
from anyfesolver._stiffened_shell_increment_control import AdaptiveIncrementControl


def record(state,accepted):
    return state.record(accepted,evaluations=3,linear_solves=1,rejected_trials=0)


def test_signed_cutback_restart_and_reversal():
    state=State(AdaptiveIncrementControl(2,12),(-2.,1.),-1.)
    assert state.next_target()==-2.
    state=record(state,False);assert state.next_target()==-1.5
    state=record(state,True);assert state.next_target()==-2.
    restored=State.from_payload(json.loads(json.dumps(state.payload())))
    assert restored==state
    state=record(restored,True);assert state.next_target()==1.
    state=record(state,False);assert state.next_target()==-.5
    state=record(state,True);assert state.next_target()==1.
    state=record(state,True);assert state.completed


def test_spent_budget_survives_restart():
    state=record(State(AdaptiveIncrementControl(0,1),(-2.,),-1.),False)
    restored=State.from_payload(state.payload())
    with pytest.raises(DisplacementIncrementLimit) as error:restored.next_target()
    assert error.value.state.attempts==state.attempts


@pytest.mark.parametrize("fault",("position","work","target","fields"))
def test_corrupt_attempt_rejected(fault):
    state=record(State(AdaptiveIncrementControl(),(-2.,),-1.),False)
    payload=deepcopy(state.payload());row=payload["attempts"][0]
    if fault=="position":row["accepted_from"]=0.
    if fault=="work":row["evaluations"]=-1
    if fault=="target":row["requested_target"]=-3.
    if fault=="fields":row["extra"]=1
    with pytest.raises(ValueError):State.from_payload(payload)


def test_domain_rejections_need_not_evaluate_residual():
    state=State(AdaptiveIncrementControl(),(-2.,),-1.)
    state=state.record(False,evaluations=1,linear_solves=1,rejected_trials=13)
    assert State.from_payload(state.payload())==state


def assert_single_cutback_history(result, resumed, prefix, initial, targets, limits):
    """Check the durable history contract, including representable remainders."""
    attempts=result.increment_attempts
    assert result.status==resumed.status=="completed"
    assert attempts==resumed.increment_attempts
    assert attempts[:len(prefix.increment_attempts)]==prefix.increment_attempts
    assert [row for row in attempts if not row.accepted]==[prefix.increment_attempts[0]]
    state=State(limits,targets,initial,attempts)
    assert state.completed
    assert 4<=len(attempts)<=limits.max_attempts
    accepted=tuple(row.displacement for row in attempts if row.accepted)
    assert tuple(row[0] for row in result.history)==accepted
    assert tuple(value for value in accepted if value in targets)==targets
    assert result.target==targets[-1]


def test_one_ulp_cutback_remainder_is_durable():
    # Actual CI seed: two rounded half-steps stop one ULP before the target.
    initial=8.915537828759525e-05
    targets=(initial*1.01,initial*1.02)
    state=State(AdaptiveIncrementControl(2,8),targets,initial)
    state=record(state,False)
    midpoint=state.next_target()
    state=record(state,True)
    assert state.next_target()==math.nextafter(targets[0],initial)
    state=record(state,True)
    state=State.from_payload(json.loads(json.dumps(state.payload())))
    assert state.next_target()==targets[0]
    state=record(state,True)
    assert state.next_target()==targets[1]
    state=record(state,True)
    assert state.completed and len(state.attempts)==5
    assert [row.displacement for row in state.attempts]==[
        targets[0],midpoint,math.nextafter(targets[0],initial),*targets]
