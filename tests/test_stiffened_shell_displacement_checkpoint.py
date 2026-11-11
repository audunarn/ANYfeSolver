"""Controller framing only; physical restoration remains a separate requirement."""
import numpy as np
import pytest
from anyfesolver._stiffened_shell_displacement_checkpoint import (
    _layout,encode_snapshot,decode_snapshot)
from test_stiffened_shell_bl5_coupled_material import coupled
from test_stiffened_shell_bl5_material import make


def fixture(request):
    owner=make(request,coupled());n=owner._K.shape[0]
    dof=next(int(i) for i in owner._free if i%6==2)
    a={name:np.zeros(shape) for name,shape in _layout(owner,2)}
    a["history"][:]=[[.1,-1.],[.2,-2.]]
    a["q"][dof]=.2;a["load_direction"][dof]=1.;a["applied_load"][dof]=-2.
    args=dict(generation=7,seed_generation=5,seed_sha256="a"*64,control_dof=dof,arrays=a)
    return owner,a,args


def test_signed_amplitude_and_chart_generation_are_not_force_history(request):
    owner,a,args=fixture(request);raw=encode_snapshot(owner,**args)
    m,decoded=decode_snapshot(owner,raw)
    # A framing-only snapshot must not enter the existing force restore route.
    with pytest.raises(ValueError):owner.restore(raw)
    assert owner._generation==0 and owner._finite_driver is None
    assert m["generation"]==7 and m["steps"]==2 and m["seed_generation"]==5
    assert decoded["history"][-1,1]==-2.
    assert "accepted_factors" not in decoded
    a["q"][:]=99.
    assert decoded["q"][args["control_dof"]]==.2
    assert all(not v.flags.writeable for v in decoded.values())
    corrupt=bytearray(raw);corrupt[-33]^=1
    with pytest.raises(ValueError,match="digest"):
        decode_snapshot(owner,bytes(corrupt))


@pytest.mark.parametrize("fault",("generation","target","load","shape","control","digest","repeated_target"))
def test_inconsistent_controller_snapshot_rejected(request,fault):
    owner,a,args=fixture(request)
    if fault=="generation":args["seed_generation"]=9
    if fault=="target":a["q"][args["control_dof"]]=0.
    if fault=="load":a["applied_load"][:]=0.
    if fault=="shape":a["shell_alpha"]=np.zeros((1,1))
    if fault=="control":args["control_dof"]=3
    if fault=="repeated_target":a["history"][0,0]=a["seed"][0]
    if fault=="digest":args["seed_sha256"]="z"*64
    with pytest.raises(ValueError):encode_snapshot(owner,**args)


def test_event_snapshot_zero_work_identity_and_strict_fields(request):
    from hashlib import sha256
    import json
    from anyfesolver._stiffened_shell_displacement_checkpoint import MAGIC,EVENT_POLICY
    from anyfesolver._stiffened_shell_branch_event_trial import POLICY as TRIAL
    owner,_,args=fixture(request)
    raw=encode_snapshot(owner,**args,trial_policy=TRIAL)
    m,_=decode_snapshot(owner,raw)
    assert m["policy"]==EVENT_POLICY and m["chart_events"]==m["event_cell_evaluations"]==0
    start=len(MAGIC)+4;length=int.from_bytes(raw[len(MAGIC):start],"little")
    for changes in ({"event_cell_evaluations":True},{"event_cell_evaluations":-1},
                    {"event_cell_evaluations":2**63},{"trial_policy":None},
                    {"chart_events":True},{"extra":0}):
        header=json.dumps(dict(m,**changes),sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")
        content=MAGIC+len(header).to_bytes(4,"little")+header+raw[start+length:-32]
        with pytest.raises(ValueError):decode_snapshot(owner,content+sha256(content).digest())
    for count in (True,-1,2**63):
        with pytest.raises(ValueError):encode_snapshot(owner,**args,trial_policy=TRIAL,event_cell_evaluations=count)
    with pytest.raises(ValueError):encode_snapshot(owner,**args,event_cell_evaluations=1)
