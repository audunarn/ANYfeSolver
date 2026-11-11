"""Inert G3b confirmation authority checks; never execute mechanics or publish GO.

The future coordinator must authenticate all inputs before importing pytest or
ANYfeSolver. Passing these validators alone is not formal confirmation.
"""
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
INVENTORY=ROOT/"docs/reference_cases/ge_beam3_g3b_confirmation_inventory_v2.json"
SCOPE="G3B_S16_MIXED_S18_REFERENCE_LINEAR_ONLY"
LANES=("mb2","mb3","mq4","ms3","weighted","transport","owner","owner_corrections")
COUNTS=(25,40,42,45,54,39,60,180)
ACCEPTED="ACCEPTED_G3B_IMPLEMENTATION_AND_RUNNER_REVIEW"
REVIEW_KEYS={"decision","findings","reviewer","scope","subject_commit"}


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,
                       allow_nan=False)+"\n").encode("ascii")


def strict(raw,limit=16*1024*1024):
    if type(raw) is not bytes or not 0<len(raw)<=limit:
        raise ValueError("bounded canonical bytes required")
    def pairs(rows):
        made={}
        for key,value in rows:
            if key in made: raise ValueError("duplicate key")
            made[key]=value
        return made
    def bad(value): raise ValueError("nonfinite JSON")
    try:
        body=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
        if canonical(body)!=raw: raise ValueError("noncanonical JSON")
        return body
    except (UnicodeError,RecursionError,TypeError,OverflowError) as exc:
        raise ValueError("invalid canonical JSON") from exc


def hash_id(value,length=64):
    if type(value) is not str or re.fullmatch("[0-9a-f]{"+str(length)+"}",value) is None:
        raise ValueError("invalid exact hash identity")


def inventory(raw):
    body=strict(raw)
    if (type(body) is not dict or set(body)!={"schema","scope","lanes"} or
            body["schema"]!="GE_BEAM3_G3B_CONFIRMATION_INVENTORY_V2" or body["scope"]!=SCOPE or
            type(body["lanes"]) is not list or len(body["lanes"])!=len(LANES)):
        raise ValueError("inventory schema/scope")
    for row,lane,count in zip(body["lanes"],LANES,COUNTS):
        path="tests/test_ge_beam3_g3b_"+lane+".py"
        if (type(row) is not dict or set(row)!={"lane","nodes","test_file","source_stdout_sha256"} or
                row["lane"]!=lane or row["test_file"]!=path or type(row["nodes"]) is not list or
                len(row["nodes"])!=count or any(type(n) is not str or not n.startswith(path+"::test_")
                for n in row["nodes"]) or len(set(row["nodes"]))!=count):
            raise ValueError("ordered separate lane inventory")
        hash_id(row["source_stdout_sha256"])
    return body


def review(raw,expected_hash,*,candidate,tree,inputs_hash,environment_hash,inventory_hash):
    for value,length in ((expected_hash,64),(candidate,40),(tree,40),(inputs_hash,64),
                         (environment_hash,64),(inventory_hash,64)): hash_id(value,length)
    if sha256(raw).hexdigest()!=expected_hash: raise ValueError("review hash mismatch")
    body=strict(raw)
    scope=dict(subject_tree=tree,scope_id=SCOPE,inputs_sha256=inputs_hash,
               environment_sha256=environment_hash,inventory_sha256=inventory_hash)
    if (type(body) is not dict or set(body)!=REVIEW_KEYS or body["decision"]!=ACCEPTED or
            body["findings"]!=[] or body["subject_commit"]!=candidate or body["scope"]!=scope):
        raise ValueError("review authority mismatch")
    who=body["reviewer"]
    if (type(who) is not dict or set(who)!={"id","independent","provenance"} or
            type(who["id"]) is not str or not who["id"].strip() or who["id"]=="/root" or
            who["independent"] is not True or who["provenance"] not in ("separate_reviewer","external_reviewer")):
        raise ValueError("independent reviewer required")
    return body


def lane_result(raw,registered):
    body=strict(raw)
    if (type(body) is not dict or set(body)!={"lane","collected","reports","exitcode"} or
            body["lane"]!=registered["lane"] or type(body["exitcode"]) is not int or body["exitcode"]!=0 or
            body["collected"]!=registered["nodes"]):
        raise ValueError("lane collection/process failure")
    expected=[dict(node=node,phase=phase,outcome="passed") for node in registered["nodes"]
              for phase in ("setup","call","teardown")]
    if body["reports"]!=expected: raise ValueError("missing/failed/skipped/extra test phase")
    return body


def process_record(raw):
    body=strict(raw)
    if (type(body) is not dict or set(body)!={"status","returncode","active_processes","elapsed_seconds",
            "peak_tree_bytes","stdout_sha256","stderr_sha256"} or body["status"]!="PASSED" or
            type(body["returncode"]) is not int or body["returncode"]!=0 or
            type(body["active_processes"]) is not int or body["active_processes"]!=0 or
            type(body["peak_tree_bytes"]) is not int or not 0<=body["peak_tree_bytes"]<=24*1024**3 or
            type(body["elapsed_seconds"]) not in (int,float) or not 0<=body["elapsed_seconds"]<600):
        raise ValueError("process/resource evidence failure")
    hash_id(body["stdout_sha256"]); hash_id(body["stderr_sha256"])
    return body


def compare_cycles(first,second):
    # Call only after validating every lane, process and packet in each cycle.
    # Preserve lane boundaries; never replace eight inventories with a sum.
    if (type(first) is not dict or type(second) is not dict or list(first)!=list(LANES) or
            list(second)!=list(LANES) or any(type(first[k]) is not bytes or type(second[k]) is not bytes
            for k in LANES) or any(first[k]!=second[k] for k in LANES)):
        raise ValueError("incomplete or nondeterministic cycles")
