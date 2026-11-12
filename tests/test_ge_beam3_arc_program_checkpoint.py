"""Static native arc-program evidence binding; no mechanics rerun."""

import hashlib
import json
from pathlib import Path

import pytest

from docs.reference_cases.ge_beam3_native_load_source_map import canonical


ROOT = Path(__file__).resolve().parents[1]


def source(path): return (ROOT/path).read_text(encoding='utf-8').encode('utf-8')


def check(path,binding):
    raw = source(path)
    assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) == binding


def validate():
    raw = source('docs/reference_cases/ge_beam3_arc_program_development_evidence.json')
    record = json.loads(raw); assert canonical(record) == raw
    for path,binding in record['source_bindings'].items(): check(path,binding)
    for path,digest in record['preserved_records'].items():
        previous = source(path); assert hashlib.sha256(previous).hexdigest() == digest
        data = json.loads(previous)
        for item,binding in data.get('bindings',data.get('source_bindings',{})).items(): check(item,binding)
    return record


def test_arc_implementation_is_not_qualification_or_activation():
    record = validate()
    assert record['source_regression']['passed'] == sum(row['passed'] for row in record['source_regression']['lanes']) == 201
    assert record['source_regression']['failed'] == 0
    assert record['native_arc_driver_implemented'] and record['plastic_split_restart_byte_identical']
    assert record['independent_review_status'] == 'PENDING' and record['remaining']
    assert record['formal_resource_requests'] == []
    assert not any(record[key] for key in ('production_qualified','public_driver_authorized',
        'existing_source_files_changed','mechanics_changed','tolerances_changed','defaults_changed',
        'historical_evidence_reclassified','release_authorized','adaptive_cutback_implemented',
        'installed_arc_wheel_test_executed','small_arch_engineering_or_stability_qualified'))


def test_both_arch_traces_and_capsules_are_bound_identically():
    record = validate(); rows = {r['path']:r for r in record['archive_files']}
    assert len(rows) == record['archive_file_count'] == 6
    assert sum(r['bytes'] for r in rows.values()) == record['archive_total_bytes'] == 204522
    for name in ('diagnostic-checkpoint.json','diagnostic-status.json','diagnostic-progress.json'):
        first,second = rows['cycle-1/'+name],rows['cycle-2/'+name]
        assert first['bytes'] == second['bytes'] and first['sha256'] == second['sha256']


@pytest.mark.parametrize('path',[
    'src/anyfesolver/_ge_beam3_arc_program.py',
    'src/anyfesolver/_ge_beam3_arc_geometry.py',
    'src/anyfesolver/_native_rotation_state.py',
])
def test_new_or_preserved_source_mutation_breaks_binding(monkeypatch,path):
    original = source
    monkeypatch.setattr(__import__(__name__),'source',
        lambda item: original(item)+(b'\n# mutation\n' if item == path else b''))
    with pytest.raises(AssertionError): validate()
