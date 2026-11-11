"""Static displacement development/evidence extent checks."""

import hashlib
import json
from pathlib import Path

import pytest

from docs.reference_cases import ge_beam3_native_load_source_map as native


ROOT = Path(__file__).resolve().parents[1]
RECORD = 'docs/reference_cases/ge_beam3_displacement_program_development_evidence.json'


def source(path): return (ROOT/path).read_text(encoding='utf-8').encode('utf-8')


def check_binding(path,expected):
    raw = source(path)
    assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) == expected


def validate():
    raw = source(RECORD); record = json.loads(raw)
    assert native.canonical(record) == raw
    for path,expected in record['source_bindings'].items(): check_binding(path,expected)
    previous = source('docs/reference_cases/ge_beam3_force_program_development_evidence.json')
    assert hashlib.sha256(previous).hexdigest() == record['preserved_force_evidence_sha256']
    force = json.loads(previous)
    for group in ('source_bindings','preserved_framework'):
        for path,expected in force[group].items(): check_binding(path,expected)
    assert native.canonical(native.build()) == native.source(native.MANIFEST)
    return record


def test_displacement_scope_and_no_production_claim():
    record = validate()
    assert record['source_regression']['passed'] == sum(row['passed'] for row in record['source_regression']['lanes']) == 132
    assert record['source_regression']['failed'] == 0
    assert record['independent_review_status'] == 'PENDING' and record['remaining']
    assert record['formal_resource_requests'] == []
    assert not any(record[key] for key in ('production_qualified','public_driver_authorized',
        'existing_source_files_changed','mechanics_changed','tolerances_changed','defaults_changed',
        'historical_evidence_reclassified','release_authorized','arc_length_implemented',
        'adaptive_cutback_implemented','installed_displacement_wheel_test_executed',
        'small_arch_critical_point_or_stability_qualified'))


def test_genuine_failure_and_matching_corrected_traces_are_retained():
    record = validate(); incident = record['development_incidents'][0]
    assert incident['initial_arch_test']['failed'] == incident['traced_arch_test']['failed'] == 1
    assert not incident['initial_arch_test']['raw_trace_available']
    assert incident['traced_arch_test']['completed_targets'] == 0
    files = {row['path']:row for row in record['archive_files']}
    assert len(files) == record['archive_file_count'] == 9
    assert sum(row['bytes'] for row in files.values()) == record['archive_total_bytes'] == 148004
    for name in ('diagnostic-checkpoint.json','diagnostic-status.json','diagnostic-progress.json'):
        a,b = files['corrected-cycle-1/'+name],files['corrected-cycle-2/'+name]
        assert a['sha256'] == b['sha256'] and a['bytes'] == b['bytes']
        assert files['failed-trace/'+name]['sha256'] != a['sha256']


@pytest.mark.parametrize('path',[
    'src/anyfesolver/_ge_beam3_displacement_program.py',
    'tests/test_ge_beam3_displacement_program.py',
    'src/anyfesolver/_ge_beam3_load_program.py',
])
def test_changed_driver_test_or_preserved_helper_breaks_binding(monkeypatch,path):
    original = source
    monkeypatch.setattr(__import__(__name__),'source',
        lambda item: original(item)+(b'\n# mutation\n' if item == path else b''))
    with pytest.raises(AssertionError): validate()
