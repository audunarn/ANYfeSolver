"""G3b owner development regression inventory; no formal qualification."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from anyfesolver._ge_beam3_g1_elastic import canonical, owned
from anyfesolver._ge_beam3_g3b_owner import MixedReferenceOwner, _Bundle, MAX_BYTES

ROOT=Path(__file__).resolve().parents[1]
LANES=("mb2","mb3","mq4","ms3","weighted")
PARENT="40a1cbdc971116b0b371fa7dd8e82161d249c66e"


def baseline(lane):
    spec=importlib.util.spec_from_file_location("owner_baseline_"+lane,
        ROOT/("tests/test_ge_beam3_g3b_"+lane+".py"))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def setup(lane="mq4"):
    module=baseline(lane); problem=module.make()
    return module,problem,MixedReferenceOwner(problem),module.load(problem)


def test_mq4_atomic_publication_restart_smoke():
    module,p,owner,f=setup()
    original=owner.checkpoint()
    result=owner.solve((f,-.5*f))
    assert owner.checkpoint()!=original
    assert canonical(result)==canonical([p.solve(f),p.solve(-.5*f)])
    data=owner.checkpoint()
    restored=MixedReferenceOwner.restore(module.make(),data,sha256(data).hexdigest())
    assert restored.checkpoint()==data
    assert canonical(restored.solve((f,)))==canonical(owner.solve((f,)))


@pytest.mark.parametrize("lane",LANES)
def test_two_rhs_replay_and_independent_owners(lane):
    module,p,owner,f=setup(lane)
    second=MixedReferenceOwner(module.make()); untouched=second.checkpoint()
    result=owner.solve((f,-.5*f))
    assert canonical(result)==canonical([p.solve(f),p.solve(-.5*f)])
    assert second.checkpoint()==untouched
    data=owner.checkpoint()
    recovered=MixedReferenceOwner.restore(module.make(),data,sha256(data).hexdigest())
    assert recovered.checkpoint()==data
    result[0]["u"][0]+=1
    assert owner.checkpoint()==data  # returned views cannot mutate committed bytes
    for factor in (0.,1.,-.5,0.):
        assert canonical(owner.solve((factor*f,)))==canonical(recovered.solve((factor*f,)))
    assert owner.checkpoint()==recovered.checkpoint()
    assert p.native._mesh is None and p.native._owner_guard is None


@pytest.mark.parametrize("lane",LANES)
@pytest.mark.parametrize("phase",("native_prepared","other_prepared","final_prepare"))
def test_last_preparation_failure_is_atomic(lane,phase):
    _,p,owner,f=setup(lane); owner.solve((f,))
    before=owner.checkpoint(); factor=owner._factor; hits=[]
    def hook(stage,preview):
        assert type(preview) is bytes
        if stage==phase:
            hits.append(stage)
            if phase=="final_prepare" or len(hits)==2: raise RuntimeError("injected last prepare")
    with pytest.raises(RuntimeError,match="injected"):
        owner.solve((-.5*f,f),on_prepare=hook)
    assert owner.checkpoint()==before and owner._factor is factor
    owner.solve((f,))


@pytest.mark.parametrize("stop",range(1,7))
def test_cancellation_rolls_back_complete_batch(stop):
    _,p,owner,f=setup(); before=owner.checkpoint(); calls=[]
    def cancel(): calls.append(True); return len(calls)==stop
    with pytest.raises(InterruptedError): owner.solve((f,-.5*f),cancel=cancel)
    assert len(calls)==stop and owner.checkpoint()==before


@pytest.mark.parametrize("mutation",("factor","bundle","lock","graph","constraint","dispatch","owner_guard"))
def test_final_callback_mutation_rejected(mutation):
    _,p,owner,f=setup(); owner.solve((f,)); before=owner.checkpoint()
    node=p.model.mesh.nodes[201]; x=node.x; guard=MixedReferenceOwner._guard
    old_T=p.T
    def hook(stage,preview):
        if stage!="final_prepare": return
        if mutation=="factor": object.__setattr__(owner,"_factor",owned(owner._factor*2))
        elif mutation=="bundle": object.__setattr__(owner,"_bundle",_Bundle(b"bad","bad"))
        elif mutation=="lock":
            import threading
            object.__setattr__(owner,"_lock",threading.Lock())
        elif mutation=="graph": node.x+=.1
        elif mutation=="constraint": p.T=owned(p.T*.5)
        elif mutation=="dispatch": p.solve=lambda *a,**k: None
        else: MixedReferenceOwner._guard=lambda *a,**k: None
    try:
        with pytest.raises((ValueError,AttributeError)): owner.solve((f,),on_prepare=hook)
        assert owner._bundle.data==before
    finally:
        # Restoring even the same node coordinate advances the mesh epoch.
        # A genuinely mutated graph stays poisoned; do not reseal its epoch.
        if mutation=="graph": node.x=x
        p.T=old_T
        p.__dict__.pop("solve",None)
        MixedReferenceOwner._guard=guard
    if mutation=="graph":
        with pytest.raises(ValueError): owner.checkpoint()
        assert owner._bundle.data==before
    else: assert owner.checkpoint()==before


def test_exclusive_ownership_reentry_and_foreign_preview():
    module,p,owner,f=setup(); before=owner.checkpoint()
    with pytest.raises(ValueError,match="already has"): MixedReferenceOwner(p)
    def hook(stage,preview):
        with pytest.raises(RuntimeError): owner.solve((f,))
        with pytest.raises(RuntimeError): owner.checkpoint()
    owner.solve((f,),on_prepare=hook)
    before=owner.checkpoint()
    with pytest.raises(ValueError,match="return authority"):
        owner.solve((f,),on_prepare=lambda *a: before)
    assert owner.checkpoint()==before


@pytest.mark.parametrize("kind",("finite","rotations","subclass","wrong_type","bad_batch","bool_load"))
def test_admission_rejects_before_helper_solve(kind):
    _,p,owner,f=setup()
    with patch.object(type(p),"solve",side_effect=AssertionError("helper evaluated")):
        with pytest.raises(ValueError):
            if kind=="finite": owner.solve((f,),mode="FINITE")
            elif kind=="rotations": owner.solve((f,),rotation_targets=np.eye(3))
            elif kind=="subclass":
                class Foreign(MixedReferenceOwner): pass
                Foreign(p)
            elif kind=="wrong_type": MixedReferenceOwner(object())
            elif kind=="bad_batch": owner.solve((f,f,f))
            else: owner.solve((np.ones(p.size,dtype=bool),))


@pytest.mark.parametrize("kind",("digest","duplicate","nonfinite","whitespace","extra","missing",
    "foreign","definition","operators","runtime","bool_generation","entry_bound","rhs_shape","size"))
def test_restart_preflight_rejects_before_solve(kind):
    module,p,owner,f=setup(); owner.solve((f,))
    data=owner.checkpoint(); body=json.loads(data); digest=None
    if kind=="digest": digest="0"*64
    elif kind=="duplicate": data=b'{"schema":0,"schema":1}\n'
    elif kind=="nonfinite": data=b'{"x":NaN}\n'
    elif kind=="whitespace": data=b" "+data
    elif kind=="size": data=b" "*(MAX_BYTES+1)
    else:
        if kind=="extra": body["unregistered"]=True
        elif kind=="missing": del body["operators"]
        elif kind=="foreign": body["schema"]="GE_BEAM3_G3_GRAPH_ELASTIC_RESTART_V1"
        elif kind=="definition": body["definition"]["size"]+=6
        elif kind=="operators": body["operators"]="0"*64
        elif kind=="runtime": body["runtime"]="0"*64
        elif kind=="bool_generation": body["generation"]=True
        elif kind=="entry_bound": body["history"]*=129; body["generation"]=129
        elif kind=="rhs_shape": body["history"][0]["loads"][0].append(0)
        data=canonical(body)
    fresh=module.make()
    with patch.object(type(fresh),"solve",side_effect=AssertionError("replay started")):
        with pytest.raises(ValueError): MixedReferenceOwner.restore(fresh,data,digest or sha256(data).hexdigest())


def test_resealed_result_fails_replay_and_releases_claim():
    module,p,owner,f=setup(); owner.solve((f,))
    body=json.loads(owner.checkpoint()); body["history"][0]["results"][0]["u"][0]+=.1
    data=canonical(body); fresh=module.make()
    with pytest.raises(ValueError,match="replay mismatch"):
        MixedReferenceOwner.restore(fresh,data,sha256(data).hexdigest())
    another=MixedReferenceOwner(fresh)
    another.solve((f,))


def test_cache_corruption_between_calls_rejected():
    _,p,owner,f=setup(); before=owner._bundle.data
    factor=owner._factor
    object.__setattr__(owner,"_factor",owned(factor*2))
    with pytest.raises(ValueError): owner.solve((f,))
    assert owner._bundle.data==before
    object.__setattr__(owner,"_factor",factor)
    owner.solve((f,))


def test_stale_and_foreign_publication_cannot_replace_owner_head():
    module,p,owner,f=setup(); stale=owner._bundle
    foreign=MixedReferenceOwner(module.make())
    owner.solve((f,)); accepted=owner.checkpoint()
    for bundle in (stale,foreign._bundle):
        with pytest.raises(AttributeError): object.__setattr__(owner,"_bundle",bundle)
    with pytest.raises(AttributeError): object.__setattr__(owner._bundle,"data",b"bad")
    assert owner.checkpoint()==accepted


def test_owner_deterministic_development_packet(tmp_path):
    rows=[]
    for lane in LANES:
        module,p,owner,f=setup(lane)
        owner.solve((f,-.5*f))
        data=owner.checkpoint()
        fresh=MixedReferenceOwner.restore(module.make(),data,sha256(data).hexdigest())
        assert fresh.checkpoint()==data
        rows.append(dict(lane=lane,checkpoint=json.loads(data)))
    data=canonical(dict(schema="G3B_OWNER_DEVELOPMENT_PACKET_V1",qualification=False,records=rows))
    path=Path(os.environ.get("G3B_OWNER_DIAGNOSTIC_PAYLOAD",str(tmp_path/"owner.json")))
    with path.open("xb") as out: out.write(data)


def test_owner_extent_and_prior_tests_unchanged():
    git=["git","-c","safe.directory="+ROOT.as_posix()]
    allowed={"src/anyfesolver/_ge_beam3_g3b_owner.py":"A","tests/test_ge_beam3_g3b_owner.py":"A",
        "docs/GE_BEAM3_G3B_OWNER_CONTRACT.md":"A",
        "docs/reference_cases/ge_beam3_g3b_owner_development_v1.json":"A",
        "scripts/run_ge_beam3_g3b.py":"M"}
    allowed.update({p:"A" for p in ["docs/GE_BEAM3_G3B_ENTRY_SAFETY_STATUS.md","docs/reference_cases/ge_beam3_g3b_confirmation_inventory_v2.json"]})
    allowed.update({p:"A" for p in ["scripts/confirm_ge_beam3_g3b.py","scripts/ge_beam3_g3b_environment.py","tests/test_ge_beam3_g3b_confirmation_runner.py","docs/GE_BEAM3_G3B_CONFIRMATION_RUNNER_STATUS.md"]})
    allowed.update({p:"A" for p in ["scripts/ge_beam3_g3b_confirmation_authority.py","tests/test_ge_beam3_g3b_confirmation_authority.py","docs/GE_BEAM3_G3B_FORMAL_CONFIRMATION_CONTRACT.md","docs/reference_cases/ge_beam3_g3b_confirmation_inventory_v1.json"]})
    allowed.update({p:"A" for p in ["docs/GE_BEAM3_G3B_LOCAL_REVIEW_STATUS.md","docs/reference_cases/ge_beam3_g3b_local_safety_review_v1.json","src/anyfesolver/_ge_beam3_g3b_result_schema.py","tests/test_ge_beam3_g3b_owner_corrections.py","docs/GE_BEAM3_G3B_OWNER_CORRECTIONS.md"]})
    markers=dict(mb2="test_additive_scope_and_immutable_parent_authority",
        mb3="test_successor_extent_preserves_mechanics_and_historical_records",
        mq4="test_mq4_extent_and_frozen_parent_bindings",ms3="test_ms3_extent_and_frozen_parent_bindings",
        weighted="test_weighted_extent_and_frozen_parent_bindings",transport="test_transport_extent")
    for lane,marker in markers.items():
        path="tests/test_ge_beam3_g3b_"+lane+".py"; allowed[path]="M"
        delimiter=("def "+marker+"():").encode()
        previous=subprocess.check_output(git+["show",PARENT+":"+path],cwd=ROOT,timeout=20)
        assert previous.split(delimiter)[0]==(ROOT/path).read_bytes().replace(b"\r\n",b"\n").split(delimiter)[0]
    for line in subprocess.check_output(git+["diff","--name-status",PARENT,"--"],cwd=ROOT,timeout=20).decode().splitlines():
        change,path=line.split("\t"); assert allowed.get(path)==change
    extra=subprocess.check_output(git+["ls-files","--others","--exclude-standard"],cwd=ROOT,timeout=20).decode().splitlines()
    assert set(extra)<={p for p,s in allowed.items() if s=="A"}
