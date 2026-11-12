"""Static adaptive-arc source/evidence binding; no mechanics rerun."""

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def source(path): return (ROOT/path).read_text(encoding='utf-8').encode('utf-8')


def validate():
    raw = source('docs/reference_cases/ge_beam3_adaptive_arc_development_evidence.json')
    record = json.loads(raw)
    assert (json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii') == raw
    for path, expected in record['source_bindings'].items():
        data = source(path)
        assert dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) == expected
    for path, digest in record['preserved_records'].items():
        assert hashlib.sha256(source(path)).hexdigest() == digest
    return record


def test_adaptive_scope_is_not_qualification_or_activation():
    record = validate()
    assert record['native_adaptive_arc_implemented'] and record['real_newton_cutback_exercised']
    assert record['independent_review_status'] == 'PENDING' and record['remaining']
    assert record['formal_resource_requests'] == []
    assert not any(record[key] for key in ('production_qualified', 'public_driver_authorized',
        'mechanics_changed', 'tolerances_changed', 'defaults_changed', 'release_authorized',
        'historical_evidence_reclassified', 'installed_adaptive_wheel_test_executed'))


def test_regression_inventories_and_real_cutback_accounting():
    record = validate(); regression = record['source_regression']
    assert regression['failed'] == 0
    assert regression['passed'] == sum(lane['passed'] for lane in regression['lanes'])
    assert record['native_diagnostic'] == dict(original_steps=[0.2], accepted_steps=[0.025]*8,
        numerical_cutbacks=7, attempts=15, newton_limit=1, split_restart_byte_identical=True,
        fixed_schedule_native_capsule_byte_identical=True, engineering_qualified=False)


def test_preserved_failure_and_two_fresh_diagnostics():
    record = validate(); rows = {row['path']: row for row in record['archive_files']}
    assert len(rows) == record['archive_file_count']
    assert sum(row['bytes'] for row in rows.values()) == record['archive_total_bytes']
    assert record['development_incidents'][0]['classification'] == 'DIAGNOSTIC_DID_NOT_EXERCISE_CUTBACK'
    prior = record['prior_archive']
    old_rows = {row['path']: row for row in prior['files']}
    assert len(old_rows) == prior['file_count'] == 9
    assert sum(row['bytes'] for row in old_rows.values()) == prior['total_bytes']
    assert old_rows['initial-no-cutback/diagnostic-checkpoint.json']['bytes'] > 0
    for name in ('diagnostic-checkpoint.json', 'diagnostic-progress.json', 'diagnostic-status.json'):
        left, right = rows['cycle-1/'+name], rows['cycle-2/'+name]
        assert left['bytes'] == right['bytes'] and left['sha256'] == right['sha256']
        assert right['sha256'] == old_rows['cycle-2/'+name]['sha256']


@pytest.mark.parametrize('path', ['src/anyfesolver/_ge_beam3_adaptive_arc_program.py',
    'tests/test_ge_beam3_adaptive_arc_program.py'])
def test_mutated_adaptive_source_is_rejected(monkeypatch, path):
    original = source
    monkeypatch.setattr(__import__(__name__), 'source',
        lambda item: original(item)+(b'\n# mutation\n' if item == path else b''))
    with pytest.raises(AssertionError): validate()
