"""Data-reader checks; protected native capsule is input, never replayed."""
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

FIXTURE = Path(__file__).parent/'data/general_contact_native_curved_pair_lifecycle_press_v1.checkpoint'


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii')


def _resign(value):
    previous = hashlib.sha256(_canonical(dict(schema=value['schema'], identity=value['identity']))).hexdigest()
    for row in value['records']:
        row['previous_sha256'] = previous
        previous = hashlib.sha256(_canonical(row)).hexdigest()
    unsigned = {k: v for k, v in value.items() if k != 'checkpoint_sha256'}
    value['checkpoint_sha256'] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    return _canonical(value)


def _same(left, right):
    assert type(left) is type(right)
    if type(left) is float:
        assert struct.pack('>d', left) == struct.pack('>d', right)
    elif type(left) is dict:
        assert set(left) == set(right)
        for key in left:
            _same(left[key], right[key])
    elif type(left) is list:
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    else:
        assert left == right


def test_portable_native_producer_public_consumer():
    already_imported=set(sys.modules)
    from anyfesolver.general_contact import read_native_contact_checkpoint_results
    from examples.general_contact_native_checkpoint_results import read_portable
    raw = FIXTURE.read_bytes()
    expected = json.loads(raw)
    result = read_native_contact_checkpoint_results(raw, expected_identity=expected['identity'])
    _same(expected, result.to_dict())
    assert result.to_bytes() == raw
    assert result.completed_targets == 1
    assert result.identity == expected['identity']
    assert read_portable(FIXTURE).to_bytes() == raw
    forbidden = [name for name in sys.modules if name not in already_imported and name in (
        'anyfesolver.general_contact.native_retained_pair_static',
        'anyfesolver.general_contact.native_fibre_context')]
    assert forbidden == []
    output = os.environ.get('CONTACT_APP_RESULT_RECORD')
    if output:
        Path(output).write_text(json.dumps(dict(producer_bytes=len(raw),
            producer_sha256=hashlib.sha256(raw).hexdigest(), exact_fields=True,
            exact_original_bytes=True, completed_targets=result.completed_targets,
            example_public_read=True, native_owner_imports=forbidden,
            scope='DATA_INTEGRITY_ONLY_NO_MECHANICAL_REPLAY'), indent=2)+'\n', encoding='utf-8')


def test_portable_native_consumer_cold_import_excludes_mechanics():
    # Other test modules legitimately import native owners during collection.
    # Cold import isolation is a fresh-process contract, not shared-process
    # test order. Preserve the same real capsule and exact-byte round trip.
    code="""import sys
from pathlib import Path
from anyfesolver.general_contact import read_native_contact_checkpoint_results
from examples.general_contact_native_checkpoint_results import read_portable
raw=Path(sys.argv[1]).read_bytes()
assert read_native_contact_checkpoint_results(raw).to_bytes()==raw
assert read_portable(Path(sys.argv[1])).to_bytes()==raw
for name in ('anyfesolver.general_contact.native_retained_pair_static',
             'anyfesolver.general_contact.native_fibre_context'):
    assert name not in sys.modules, name
"""
    result=subprocess.run([sys.executable,'-c',code,str(FIXTURE)],
        cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=30.)
    assert result.returncode==0,result.stdout+'\n'+result.stderr


def test_native_detachment_and_numeric_bits():
    from anyfesolver.general_contact import read_native_contact_checkpoint_results
    value = json.loads(FIXTURE.read_bytes())
    value['records'][0]['mechanical']['position_low'][0][0] = -0.0
    value['records'][0]['mechanical']['position_low'][0][1] = float.fromhex('0x0.0000000000001p-1022')
    raw = _resign(value)
    result = read_native_contact_checkpoint_results(raw)
    _same(value, result.to_dict())
    view = result.to_dict()
    view['records'][0]['histories'][0]['stations'].clear()
    _same(value, result.to_dict())
    with pytest.raises(FrozenInstanceError):
        result._raw = b'changed'


def test_native_closed_roles_integrity_and_refusals():
    from anyfesolver.general_contact import (read_native_contact_checkpoint_results,
        NativeContactCheckpointResultError)
    raw = FIXTURE.read_bytes()
    for mutate in (
        lambda v: v.update(profile='unknown'),
        lambda v: v.update(extra=True),
        lambda v: v.update(completed_targets=True),
        lambda v: v['records'][0].update(target=True),
        lambda v: v['records'][0]['contact'].update(search_cells=1.0),
        lambda v: v['records'][0]['contact'].update(potential_j=True),
        lambda v: v['records'][0]['contact'].update(beam_a_force_n=['a','b','c']),
        lambda v: v['records'][0]['mechanical'].update(positions=[[0.]*3]),
        lambda v: v['records'][0]['histories'][0]['stations'][0].update(rows=[[0., 0., 'x', 0.]]),
    ):
        value = json.loads(raw)
        mutate(value)
        with pytest.raises(NativeContactCheckpointResultError):
            read_native_contact_checkpoint_results(_resign(value))
    value = json.loads(raw)
    value['records'][0]['mechanical']['positions'][0][0] += .1
    with pytest.raises(NativeContactCheckpointResultError, match='digest'):
        read_native_contact_checkpoint_results(_canonical(value))
    value = json.loads(raw)
    value['records'][0]['previous_sha256'] = '0'*64
    unsigned = {k: v for k, v in value.items() if k != 'checkpoint_sha256'}
    value['checkpoint_sha256'] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    with pytest.raises(NativeContactCheckpointResultError, match='chain'):
        read_native_contact_checkpoint_results(_canonical(value))
    for bad in (b'{"schema":"duplicate",'+raw[1:], raw.replace(b'0.0', b'NaN', 1),
                raw.replace(b'0.0', b'1e9999', 1), b' '+raw):
        with pytest.raises(NativeContactCheckpointResultError):
            read_native_contact_checkpoint_results(bad)
    with pytest.raises(NativeContactCheckpointResultError, match='expected identity'):
        read_native_contact_checkpoint_results(raw, expected_identity='0'*64)


def test_native_preparse_resource_and_type_limits():
    from anyfesolver.general_contact import (read_native_contact_checkpoint_results,
        NativeContactCheckpointResultError)
    raw = FIXTURE.read_bytes()
    for cap in (len(raw)-1, 0, True, 2*1024*1024+1):
        with pytest.raises(NativeContactCheckpointResultError):
            read_native_contact_checkpoint_results(raw, max_bytes=cap)
    for bad in (bytearray(raw), b'['*33+b'0'+b']'*33,
                b'['+b'0,'*200_001+b'0]', b'{}\xff'):
        with pytest.raises(NativeContactCheckpointResultError):
            read_native_contact_checkpoint_results(bad)


def test_public_example_bounded_file_read(tmp_path):
    from anyfesolver.general_contact.native_checkpoint_results import MAX_BYTES
    from anyfesolver.general_contact import NativeContactCheckpointResultError
    from examples.general_contact_native_checkpoint_results import read_portable
    original = FIXTURE.read_bytes()
    assert read_portable(FIXTURE).to_bytes() == original
    oversized = tmp_path/'oversized_native_capsule.bin'
    with oversized.open('wb') as stream:
        stream.truncate(MAX_BYTES+8192)
    with pytest.raises(NativeContactCheckpointResultError, match='limit'):
        read_portable(oversized)
    assert FIXTURE.read_bytes() == original
