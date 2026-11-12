"""Pure protocol checks; no mechanics run or owner modifications."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import pytest
from docs.reference_cases import ge_beam3_signed_lifecycle as p

@pytest.mark.parametrize('operation',('negative-seed','negative-resume','cancel-before','cancel-after','resume-cancelled'))
def test_exact_disposition(operation):
    _,_,status,cursor,event=p.recipe(operation)
    rows=[] if event is None else [dict(stage=event,target=2,iteration=4)]
    p.disposition(operation,status,cursor,rows)

@pytest.mark.parametrize('operation',('',None,True,1,'positive','retry'))
def test_unknown_operations(operation):
    with pytest.raises(ValueError):p.recipe(operation)

@pytest.mark.parametrize('kind',('status','cursor','bool','missing','duplicate','stage','target','iteration','extra'))
def test_cancellation_mutations(kind):
    status='cancelled';cursor=1;rows=[dict(stage='elastic-continuation.before_commit',target=2,iteration=4)]
    if kind=='status':status='paused'
    elif kind=='cursor':cursor=2
    elif kind=='bool':cursor=True
    elif kind=='missing':rows=[]
    elif kind=='duplicate':rows+=deepcopy(rows)
    elif kind=='stage':rows[0]['stage']='elastic-continuation.before_assembly'
    elif kind=='target':rows[0]['target']=1
    elif kind=='iteration':rows[0]['iteration']=True
    else:rows[0]['extra']=False
    with pytest.raises(ValueError):p.disposition('cancel-before',status,cursor,rows)

def test_exact_preserved_positive_inputs_and_owner():
    for name,(_,digest) in p.POSITIVES.items():assert sha256(p.positive(name)).hexdigest()==digest
    assert sha256((p.ROOT/'src/anyfesolver/_ge_beam3_elastic_seed_continuation.py').read_bytes()).hexdigest()==p.OWNER_SHA

def test_changed_manifest(tmp_path):
    (tmp_path/'manifest.json').write_bytes(b'{}\n')
    with pytest.raises(ValueError,match='manifest'):p.positive('prefix',tmp_path)
