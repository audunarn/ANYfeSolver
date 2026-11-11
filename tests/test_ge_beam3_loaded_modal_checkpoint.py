"""Static loaded-packet source/evidence binding, without mechanics reruns."""

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def source(path): return (ROOT/path).read_text(encoding='utf-8').encode('utf-8')


def validate():
    raw = source('docs/reference_cases/ge_beam3_loaded_modal_development_evidence.json')
    record = json.loads(raw)
    assert (json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii') == raw
    for path, expected in record['source_bindings'].items():
        data = source(path)
        assert dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) == expected
    for path, digest in record['preserved_records'].items():
        assert hashlib.sha256(source(path)).hexdigest() == digest
    return record


def test_scope_is_loaded_equilibrium_development_only():
    record = validate()
    assert record['loaded_elastic_interior_spectrum_implemented']
    assert record['distributed_load_second_variation_included'] and record['retained_cell_inertia']
    assert record['independent_review_status'] == 'PENDING' and record['remaining']
    assert record['formal_resource_requests'] == []
    assert not any(record[key] for key in ('production_qualified', 'public_driver_authorized',
        'mechanics_changed', 'tolerances_changed', 'defaults_changed', 'release_authorized',
        'historical_evidence_reclassified', 'installed_loaded_modal_wheel_test_executed',
        'buckling_factor_qualified', 'inelastic_vibration_branch_qualified'))


def test_regression_inventories_and_preserved_histories():
    record = validate(); regression = record['source_regression']
    assert regression['failed'] == 0
    assert regression['passed'] == sum(lane['passed'] for lane in regression['lanes'])
    assert record['accepted_history_unchanged'] and record['elastic_unloading_after_plasticity_checked']
    assert record['negative_physical_eigenvalues_retained']
    assert record['source_shared_AD_elastic_check_is_independent_oracle'] is False


def test_two_fresh_diagnostics_match_and_archive_extent_is_exact():
    record = validate(); rows = {row['path']: row for row in record['archive_files']}
    assert len(rows) == record['archive_file_count'] == 4
    assert sum(row['bytes'] for row in rows.values()) == record['archive_total_bytes']
    for name in ('axial-spectra.json', 'translated-pencil.json'):
        left, right = rows['cycle-1/'+name], rows['cycle-2/'+name]
        assert left['bytes'] == right['bytes'] and left['sha256'] == right['sha256']


@pytest.mark.parametrize('path', ['src/anyfesolver/_ge_beam3_loaded_modal.py', 'tests/test_ge_beam3_loaded_modal.py'])
def test_mutated_adapter_or_test_is_rejected(monkeypatch, path):
    original = source
    monkeypatch.setattr(__import__(__name__), 'source',
        lambda item: original(item)+(b'\n# mutation\n' if item == path else b''))
    with pytest.raises(AssertionError): validate()
