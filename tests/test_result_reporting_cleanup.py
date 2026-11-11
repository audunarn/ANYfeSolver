"""Reporting compatibility at public producer/export boundaries."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from anyfesolver.quality_control import QCReport, QCResult
from anyfesolver.general_contact import static_results_archive as static
from anyfesolver.general_contact import arc_results_archive as arc
from test_general_contact_static_results_archive import sample as static_sample
from test_general_contact_arc_results_archive import sample as arc_sample


# Captured from unchanged sources at 4bd22badd; differential comparison verified
# each complete byte stream before these digests became regression fixtures.
WIRE_SHA256 = {
    ('static', 0): 'f3b8dba1a635ff363bf26572f4c0eca0475107caa556f7e0cedb9d80d0556775',
    ('static', 1): 'c76eba2b1b58bcefb78aeb1b5be1f22cb4b0c754726c8d2c90c4e70e603fd497',
    ('static', 2): 'b0bca898732031779977bc9ea76dd5541c53db1d71bec18a35aa7405f5f3b348',
    ('arc', 0): '3788bcfe48089ef36d20cec6dca26d71f5e8e59c2448cc38d41cf569d120b7dd',
    ('arc', 1): '517e284fc6a3e091abdc981445cbf4c74fe86a030c02ce2fdc86b46a16717ea9',
    ('arc', 2): '14f48822483edddebb695b251068161e120be2fe8d750a39d2ee5086e1b9a116',
}


@pytest.mark.parametrize('owner', [QCResult, QCReport])
def test_qc_nested_conversion_and_recursive_override(owner):
    class Printable:
        def __str__(self):
            return 'printable'

    value = {'nested': (np.int64(3), [np.bool_(True), np.float32(-0.)]),
             'array': np.array([[1, 2]], dtype=np.int32), 'none': None,
             'custom': Printable(), 'native': [4, 2.5, 'text', False]}
    instance = owner('test', 'reporting', True) if owner is QCResult else owner()
    converted = instance._make_serializable(value)
    assert converted == {'nested': [3., [True, -0.]], 'array': [[1, 2]],
                         'none': 'None', 'custom': 'printable',
                         'native': [4, 2.5, 'text', False]}
    assert type(converted['nested'][0]) is float
    assert type(converted['nested'][1][0]) is bool
    assert np.signbit(converted['nested'][1][1])
    assert isinstance(value['nested'], tuple)
    marker = object()

    class Customized(owner):
        def _make_serializable(self, obj):
            if obj is marker:
                return 'override'
            return super()._make_serializable(obj)

    customized = Customized('test', 'reporting', True) if owner is QCResult else Customized()
    assert customized._make_serializable({'deep': ([marker],)}) == {'deep': [['override']]}
    # ndarray.tolist historically does not recurse through object-array entries.
    assert instance._make_serializable(np.array([marker], dtype=object))[0] is marker


def test_qc_saved_report_keeps_public_output(tmp_path):
    result = QCResult('sample', 'reporting', np.bool_(True), np.float64(-0.),
                      'reference', np.int64(2), {'values': (np.int32(3), None)}, 'fixed')
    report = QCReport([result], {'enabled': np.bool_(False)}, 'fixed')
    target = tmp_path / 'nested' / 'report.json'
    report.save(str(target))
    assert target.read_text() == json.dumps({
        'summary': {'enabled': False},
        'results': [{'test_name': 'sample', 'test_category': 'reporting', 'passed': True,
                     'error': -0., 'expected': 'reference', 'actual': 2.,
                     'details': {'values': [3., 'None']}, 'timestamp': 'fixed'}],
        'timestamp': 'fixed'}, indent=2)


def archive_case(factory, count):
    value = factory()
    snapshot = value.snapshots[0]
    snapshots = tuple(replace(snapshot, step_index=index + 1) for index in range(count))
    return replace(value, snapshots=snapshots, info={**value.info,
        'nested': [{'integration_force': [np.float32(-0.), 1., 2.],
                    'minimum_gap': float('inf'), 'gap_tolerance_met': np.bool_(True)}]})


@pytest.mark.parametrize('module,factory,kind', [(static, static_sample, 'static'), (arc, arc_sample, 'arc')])
@pytest.mark.parametrize('count', [0, 1, 2])
def test_archive_wire_snapshot_compatibility(module, factory, kind, count):
    value = archive_case(factory, count)
    serialize = getattr(module, f'serialize_{kind}_contact_results')
    read = getattr(module, f'read_{kind}_contact_results')
    raw = serialize(value)
    assert hashlib.sha256(raw).hexdigest() == WIRE_SHA256[kind, count]
    archive = read(raw)
    with pytest.raises(module.codec.TransientContactArchiveError):
        serialize(value, max_bytes=len(raw) - 1)
    with pytest.raises(FrozenInstanceError):
        archive._raw = b''
    if count:
        view = archive.to_dict()
        view['snapshots'][0]['fields']['displacements'][0] = 42.
        assert np.signbit(archive.to_dict()['snapshots'][0]['fields']['displacements'][0])
        assert np.signbit(value.snapshots[0].displacements[0])
    assert archive.to_bytes() == raw
    assert len(value.snapshots) == count
    # The separately saved pre-cleanup sources are used for the differential run.
    legacy_root = os.environ.get('RESULT_REPORTING_LEGACY_ROOT')
    if legacy_root:
        alias = f'anyfesolver.general_contact._cleanup_legacy_{kind}'
        spec = importlib.util.spec_from_file_location(alias, Path(legacy_root) / f'{kind}_results_archive.py')
        legacy = importlib.util.module_from_spec(spec)
        sys.modules[alias] = legacy
        try:
            spec.loader.exec_module(legacy)
            old_serialize = getattr(legacy, f'serialize_{kind}_contact_results')
            assert old_serialize(value) == raw
            cases = [replace(value, snapshots=[]), replace(value, snapshots=(object(),)),
                     replace(value, info={}), replace(value, displacements=np.array([True])),
                     replace(value, info={'general_contact': {'nested': [{'integration_force': [1., 2.]}]}}),
                     replace(value, info={'general_contact': {'gap_tolerance_met': 1}}),
                     replace(value, info={'general_contact': {'minimum_gap': float('nan')}})]
            for bad in cases:
                errors = []
                for writer in (old_serialize, serialize):
                    with pytest.raises(module.codec.TransientContactArchiveError) as caught:
                        writer(bad)
                    errors.append((type(caught.value), str(caught.value)))
                assert errors[0] == errors[1]
            for cap in (128, len(raw) - 1, len(raw)):
                errors = []
                for writer in (old_serialize, serialize):
                    with pytest.raises(module.codec.TransientContactArchiveError) as caught:
                        writer(value, max_bytes=cap)
                    errors.append((type(caught.value), str(caught.value)))
                assert errors[0] == errors[1]
            record = Path(legacy_root).parent / f'wire-{kind}-{count}.json'
            record.write_text(json.dumps({'sha256': hashlib.sha256(raw).hexdigest(),
                                         'bytes': len(raw), 'legacy_equal': True}))
        finally:
            sys.modules.pop(alias, None)
    assert serialize(value) == raw
