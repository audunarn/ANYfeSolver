"""Nonmechanical G3b authority infrastructure inventory; no qualification claim."""
import ast
from copy import deepcopy
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("g3b_confirmation_authority",
    ROOT/"scripts/ge_beam3_g3b_confirmation_authority.py")
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


@pytest.mark.parametrize("raw",(b'{"x":0,"x":1}\n',b'{"x":NaN}\n',b'{"x":Infinity}\n',
    b'{"x":1e999}\n',b' {"x":0}\n',b'{"x":0}',b'',b'[]\r\n'))
def test_strict_malformed(raw):
    with pytest.raises(ValueError): c.strict(raw)


def test_frozen_separate_inventory():
    body=c.inventory(c.INVENTORY.read_bytes().replace(b"\r\n",b"\n"))
    assert [(r["lane"],len(r["nodes"])) for r in body["lanes"]]==list(zip(c.LANES,c.COUNTS))
    assert c.strict(c.canonical(body))==body


@pytest.mark.parametrize("kind",("reorder","missing","extra","duplicate","node_family","test_file","hash","scope"))
def test_inventory_mutation(kind):
    body=c.strict(c.INVENTORY.read_bytes().replace(b"\r\n",b"\n")); row=body["lanes"][0]
    if kind=="reorder": body["lanes"].reverse()
    elif kind=="missing": row["nodes"].pop()
    elif kind=="extra": row["nodes"].append(row["nodes"][0])
    elif kind=="duplicate": row["nodes"][1]=row["nodes"][0]
    elif kind=="node_family": row["nodes"][0]="tests/test_ge_beam3_g3a_graph.py::test_old"
    elif kind=="test_file": row["test_file"]="other"
    elif kind=="hash": row["source_stdout_sha256"]="x"*64
    else: body["scope"]="FULL_PARITY"
    with pytest.raises(ValueError): c.inventory(c.canonical(body))


def review_body():
    # Disposable validator fixture only, never an actual reviewer attestation.
    return dict(decision=c.ACCEPTED,findings=[],subject_commit="a"*40,
        reviewer=dict(id="TEST_FIXTURE_NOT_A_REVIEW",independent=True,provenance="external_reviewer"),
        scope=dict(subject_tree="b"*40,scope_id=c.SCOPE,inputs_sha256="c"*64,
                   environment_sha256="d"*64,inventory_sha256="e"*64))


@pytest.mark.parametrize("kind",("valid","hash","subject_commit","subject_tree","scope_id","inputs_sha256",
    "environment_sha256","inventory_sha256","findings","decision","independent","author","provenance","extra"))
def test_review_bindings(kind):
    body=review_body()
    if kind=="subject_commit": body[kind]="b"*40
    elif kind in body["scope"]: body["scope"][kind]="foreign"
    elif kind=="findings": body[kind]=[dict(severity="P2",message="state safety")]
    elif kind=="decision": body[kind]="BLOCKED"
    elif kind=="independent": body["reviewer"][kind]=1
    elif kind=="author": body["reviewer"]["id"]="/root"
    elif kind=="provenance": body["reviewer"][kind]="self_review"
    elif kind=="extra": body["extra"]=0
    raw=c.canonical(body)
    digest="0"*64 if kind=="hash" else sha256(raw).hexdigest()
    kwargs=dict(candidate="a"*40,tree="b"*40,inputs_hash="c"*64,environment_hash="d"*64,inventory_hash="e"*64)
    if kind=="valid": assert c.review(raw,digest,**kwargs)==body
    else:
        with pytest.raises(ValueError): c.review(raw,digest,**kwargs)


@pytest.mark.parametrize("kind",("valid","missing","duplicate","skipped","failed","collection","exit_bool","extra"))
def test_all_test_phases(kind):
    registered=dict(lane="mb2",nodes=["one"])
    body=dict(lane="mb2",collected=["one"],exitcode=0,
        reports=[dict(node="one",phase=p,outcome="passed") for p in ("setup","call","teardown")])
    if kind=="missing": body["reports"].pop()
    elif kind=="duplicate": body["reports"].append(body["reports"][0])
    elif kind in ("skipped","failed"): body["reports"][1]["outcome"]=kind
    elif kind=="collection": body["collected"]=[]
    elif kind=="exit_bool": body["exitcode"]=False
    elif kind=="extra": body["extra"]=0
    if kind=="valid": assert c.lane_result(c.canonical(body),registered)==body
    else:
        with pytest.raises(ValueError): c.lane_result(c.canonical(body),registered)


@pytest.mark.parametrize("kind",("valid","timeout","memory","alive","crash","bool","hash"))
def test_terminal_process_bounds(kind):
    body=dict(status="PASSED",returncode=0,active_processes=0,elapsed_seconds=1.2,
        peak_tree_bytes=1000,stdout_sha256="a"*64,stderr_sha256="b"*64)
    if kind=="timeout": body["elapsed_seconds"]=600
    elif kind=="memory": body["peak_tree_bytes"]=24*1024**3+1
    elif kind=="alive": body["active_processes"]=1
    elif kind=="crash": body["returncode"]=-1
    elif kind=="bool": body["peak_tree_bytes"]=True
    elif kind=="hash": body["stdout_sha256"]="bad"
    if kind=="valid": assert c.process_record(c.canonical(body))==body
    else:
        with pytest.raises(ValueError): c.process_record(c.canonical(body))


@pytest.mark.parametrize("kind",("valid","missing","extra","order","bytes","type"))
def test_cycle_equality_keeps_lane_boundaries(kind):
    first={lane:b'{}\n' for lane in c.LANES}; second=deepcopy(first)
    if kind=="missing": second.pop("owner")
    elif kind=="extra": second["old_g3a"]=b'{}\n'
    elif kind=="order": second=dict(reversed(list(second.items())))
    elif kind=="bytes": second["owner"]=b'{"changed":true}\n'
    elif kind=="type": second["owner"]="{}\n"
    if kind=="valid": c.compare_cycles(first,second)
    else:
        with pytest.raises(ValueError): c.compare_cycles(first,second)


def test_authority_module_is_inert_stdlib_only():
    tree=ast.parse(Path(c.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            assert all(alias.name.split('.')[0] in sys.stdlib_module_names for alias in node.names)
        elif isinstance(node,ast.ImportFrom):
            assert node.level==0 and node.module.split('.')[0] in sys.stdlib_module_names
        elif isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            assert node.func.id not in ("exec","eval","__import__","compile")


@pytest.mark.parametrize("lane",c.LANES)
def test_real_collection_matches_frozen_lane(lane):
    row=next(r for r in c.inventory(c.INVENTORY.read_bytes().replace(b"\r\n",b"\n"))["lanes"] if r["lane"]==lane)
    result=subprocess.run([sys.executable,"-B","-m","pytest","--collect-only","-q","-p","no:cacheprovider",
                           row["test_file"]],cwd=ROOT,capture_output=True,text=True,timeout=30,check=True)
    nodes=[line.strip() for line in result.stdout.splitlines() if line.startswith(row["test_file"]+"::")]
    assert nodes==row["nodes"]


def test_confirmation_has_only_owner_entry_runtime_delta():
    base="f67aed07ed877aec679837e0a44e2db5cb398f74"
    git=["git","-c","safe.directory="+ROOT.as_posix()]
    path="src/anyfesolver/_ge_beam3_g3b_owner.py"
    changed=subprocess.check_output(git+["diff","--name-only",base,"--","src/"],cwd=ROOT,timeout=20).decode().splitlines()
    assert changed==[path]
    old=ast.parse(subprocess.check_output(git+["show",base+":"+path],cwd=ROOT,timeout=20))
    new=ast.parse((ROOT/path).read_text())
    for tree in (old,new):
        owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="MixedReferenceOwner")
        owner.body=[n for n in owner.body if not (isinstance(n,ast.FunctionDef) and n.name=="solve")]
    assert ast.dump(old)==ast.dump(new)
    assert not subprocess.check_output(git+["diff","--name-only",base,"--","pyproject.toml",".github"],cwd=ROOT,timeout=20).strip()
    subprocess.run(git+["diff","--check",base],cwd=ROOT,timeout=20,check=True)
