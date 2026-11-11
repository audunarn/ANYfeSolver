"""Successor regressions for preserved G3B-LR-01 and G3B-LR-02."""
from copy import copy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from anyfesolver._ge_beam3_g1_elastic import canonical
from anyfesolver._ge_beam3_g3b_owner import MixedReferenceOwner

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("g3b_owner_baseline",ROOT/"tests/test_ge_beam3_g3b_owner.py")
baseline=importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
LANES=baseline.LANES


def record(request, **observed):
    destination=os.environ.get("G3B_CORRECTION_RECORDS")
    if destination is None: return
    payload=dict(schema="G3B_OWNER_CORRECTION_CASE_V1",node=request.node.nodeid,
                 qualification=False,observed=observed)
    path=Path(destination)/(sha256(request.node.nodeid.encode()).hexdigest()+".json")
    with path.open("xb") as stream: stream.write(canonical(payload))


@pytest.mark.parametrize("lane",LANES)
@pytest.mark.parametrize("boundary",("first","second","final"))
@pytest.mark.parametrize("target",("recover","guard","_identity","cell","section_descriptor","other_recovery"))
def test_recovery_closure_mutation_preserves_accepted_prefix(lane,boundary,target,request):
    module,p,owner,f=baseline.setup(lane)
    owner.solve((f,)); before=owner.checkpoint(); factor=owner._factor
    operator=p.native.operator; cell=operator.cell; changes=[]; hits=[]
    def hook(phase,preview):
        if phase=="native_prepared": hits.append(phase)
        trigger=(phase=="final_prepare" if boundary=="final" else
                 phase=="native_prepared" and len(hits)==(1 if boundary=="first" else 2))
        if not trigger: return
        if target=="cell": obj,name,value=operator,"cell",copy(cell)
        elif target=="section_descriptor":
            obj,name,value=type(cell.section),"descriptor",lambda *a: {}
        elif target=="other_recovery":
            obj=p.legacy if hasattr(p,"legacy") else p.shell
            name,value="compute_stresses",lambda *a,**k: {}
        else: obj,name,value=cell,target,lambda *a,**k: ()
        change=patch.object(obj,name,value); change.start(); changes.append(change)
    try:
        with pytest.raises(ValueError,match="dispatch"):
            owner.solve((-.5*f,f),on_prepare=hook)
        assert changes and owner._bundle.data==before and owner._factor is factor
    finally:
        for change in reversed(changes): change.stop()
    assert owner.checkpoint()==before
    restored=MixedReferenceOwner.restore(module.make(),before,sha256(before).hexdigest())
    assert canonical(restored.solve((-.5*f,)))==canonical(owner.solve((-.5*f,)))
    assert restored.checkpoint()==owner.checkpoint()
    record(request,lane=lane,boundary=boundary,target=target,
        accepted_before=json.loads(before),accepted_after=json.loads(owner.checkpoint()),
        replay_after=json.loads(restored.checkpoint()),factor_reused=owner._factor is factor)


@pytest.mark.parametrize("lane",LANES)
@pytest.mark.parametrize("kind",("missing_u","extra","missing_station","station_extra","station_shape",
    "station_bool","history","recovery_missing","recovery_extra","recovery_bool","policy",
    "qualification","node_bool","multipliers","energy","late_entry"))
def test_complete_journal_schema_precedes_owner_construction(lane,kind,request):
    module,p,owner,f=baseline.setup(lane); owner.solve((f,)); owner.solve((-.5*f,))
    body=json.loads(owner.checkpoint())
    result=body["history"][-1 if kind=="late_entry" else 0]["results"][0]
    station=result["native_stations"][0]
    recovery=result["legacy_recovery" if lane in ("mb2","mb3") else "shell_recovery"]
    if kind in ("missing_u","late_entry"): del result["u"]
    elif kind=="extra": result["extra"]=0
    elif kind=="missing_station": result["native_stations"].pop()
    elif kind=="station_extra": station["extra"]=0
    elif kind=="station_shape": station["strain"].pop()
    elif kind=="station_bool": station["strain"][0]=False
    elif kind=="history": station["history"]=[0]
    elif kind=="recovery_missing": del recovery[next(iter(recovery))]
    elif kind=="recovery_extra": recovery["extra"]=0
    elif kind=="recovery_bool":
        if lane in ("mb2","mb3"): recovery["axial_stress"]=False
        else: recovery["membrane_resultants"][0][0]=False
    elif kind=="policy": result["policy"]="foreign"
    elif kind=="qualification": result["qualification"]=0
    elif kind=="node_bool": result["node_ids"][0]=True
    elif kind=="multipliers": result["multipliers"].pop()
    elif kind=="energy": result["energy"]="0"
    data=canonical(body); fresh=module.make()
    with patch.object(MixedReferenceOwner,"__init__",side_effect=AssertionError("owner constructed")), \
         patch.object(type(fresh),"solve",side_effect=AssertionError("replay started")):
        with pytest.raises(ValueError,match="mixed result"):
            MixedReferenceOwner.restore(fresh,data,sha256(data).hexdigest())
    # No provisional owner/claim was created during inert preflight.
    MixedReferenceOwner(fresh).solve((f,))
    record(request,lane=lane,mutation=kind,accepted=json.loads(owner.checkpoint()),
           rejected_envelope=json.loads(data),rejection_before_owner_construction=True)


@pytest.mark.parametrize("lane",LANES)
@pytest.mark.parametrize("failure",("raising_view","forged_view"))
def test_entry_bookkeeping_rejection_releases_lock(lane,failure,request):
    from anyfesolver._ge_beam3_g3b_owner import _Bundle
    module,p,owner,f=baseline.setup(lane); owner.solve((f,))
    before=owner.checkpoint(); factor=owner._factor
    fake_data=canonical(dict(forged=True)); fake=_Bundle(fake_data,sha256(fake_data).hexdigest())
    def unavailable(self):
        if failure=="raising_view": raise RuntimeError("injected entry publication access failure")
        return fake
    with patch.object(MixedReferenceOwner,"_bundle",property(unavailable)):
        with pytest.raises((RuntimeError,ValueError)):
            owner.solve((-.5*f,))
        assert not owner._lock.locked()
    assert owner.checkpoint()==before and owner._factor is factor
    restored=MixedReferenceOwner.restore(module.make(),before,sha256(before).hexdigest())
    assert canonical(owner.solve((-.5*f,)))==canonical(restored.solve((-.5*f,)))
    assert owner.checkpoint()==restored.checkpoint()
    record(request,lane=lane,boundary="entry",target=failure,
        accepted_before=json.loads(before),accepted_after=json.loads(owner.checkpoint()),
        replay_after=json.loads(restored.checkpoint()),factor_reused=owner._factor is factor)
