"""Adversarial byte protocol controls, no physical model or solve."""
import os,json,hashlib
import numpy as np
import pytest
pytestmark=pytest.mark.skipif('F06_CHECKPOINT_REPAIR_OUTPUT' not in os.environ,reason='fresh root repair supervisor only')
MODEL='repair-byte-model'
SHAPES=(('q_low',(1,)),('frames_low',(1,3,3)),('cells_low',(1,2,3,3)),('resultants_low',(1,18)))

def raw_header(payload,tail=b'',declared=None):
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import COMPENSATED_MAGIC
    header=payload if type(payload) is bytes else json.dumps(payload,separators=(',',':')).encode('ascii')
    body=COMPENSATED_MAGIC+(len(header) if declared is None else declared).to_bytes(4,'little')+header+tail
    return body+hashlib.sha256(body).digest()

@pytest.mark.parametrize('root',[None,0,1,1.2,'schema',[],[{}],True,False,{}])
def test_checkpoint_repair_valid_digest_nonobject_and_malformed_root(root):
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated
    with pytest.raises(ValueError,match='model identity'):decode_compensated(raw_header(root),MODEL,SHAPES)

@pytest.mark.parametrize('header',[b'\xff',b'{'])
def test_checkpoint_repair_expected_parse_errors(header):
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated
    with pytest.raises(ValueError,match='invalid compensated metadata'):decode_compensated(raw_header(header),MODEL,SHAPES)

@pytest.mark.parametrize('declared',[0,8193,20])
def test_checkpoint_repair_bounded_metadata_slices(declared):
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated
    with pytest.raises(ValueError,match='bounded compensated metadata'):decode_compensated(raw_header(b'null',declared=declared),MODEL,SHAPES)

def test_checkpoint_repair_bounded_inner_slice():
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated,COMPENSATED_SCHEMA
    metadata=dict(schema=COMPENSATED_SCHEMA,model=MODEL,inner_bytes=10**100,fields=[])
    with pytest.raises(ValueError,match='bounded inner checkpoint'):decode_compensated(raw_header(metadata),MODEL,SHAPES)

def test_checkpoint_repair_memoryerror_propagates(monkeypatch):
    import anyfesolver._stiffened_shell_ge_checkpoint_binary as binary
    raw=raw_header(None);failure=MemoryError('injected parser allocation failure')
    def fail(*args,**kwargs):raise failure
    monkeypatch.setattr(binary.json,'loads',fail)
    with pytest.raises(MemoryError) as caught:binary.decode_compensated(raw,MODEL,SHAPES)
    assert caught.value is failure

def test_checkpoint_repair_exact_compensated_roundtrip():
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import encode_compensated,decode_compensated
    values={name:np.arange(np.prod(shape),dtype=np.float64).reshape(shape)/100 for name,shape in SHAPES}
    raw=encode_compensated(b'owned inner bytes',MODEL,values)
    inner,got=decode_compensated(raw,MODEL,SHAPES)
    assert inner==b'owned inner bytes'
    for name in values:
        np.testing.assert_array_equal(got[name],values[name]);assert not np.shares_memory(got[name],values[name])
    assert encode_compensated(inner,MODEL,got)==raw

def test_checkpoint_repair_affected_legacy_binary_metadata():
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import encode,decode
    shapes=(('q',(2,)),('multipliers',(1,)),('load',(2,)))
    values=tuple((name,np.arange(np.prod(shape),dtype=np.float64).reshape(shape)) for name,shape in shapes)
    raw=encode('linear',MODEL,1,values)
    def expected(kind,generation):
        assert kind=='linear' and generation==1
        return shapes
    kind,generation,got=decode(raw,MODEL,expected)
    assert kind=='linear' and generation==1
    assert encode(kind,MODEL,generation,tuple((n,got[n]) for n,_ in shapes))==raw
    with pytest.raises(ValueError,match='incompatible'):decode(raw,'wrong model',expected)
