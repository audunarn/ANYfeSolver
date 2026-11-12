"""Static arc-geometry binding; no solver/qualification claim."""

import hashlib
import json
from pathlib import Path

import pytest

from docs.reference_cases.ge_beam3_native_load_source_map import canonical


ROOT = Path(__file__).resolve().parents[1]


def source(path): return (ROOT/path).read_text(encoding='utf-8').encode('utf-8')


def validate():
    raw = source('docs/reference_cases/ge_beam3_arc_geometry_development_evidence.json')
    record = json.loads(raw); assert canonical(record) == raw
    for path,binding in record['bindings'].items():
        data = source(path)
        assert dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) == binding
    return record


def test_geometry_only_scope_is_explicit():
    record = validate()
    assert record['source_regression']['passed'] == sum(row['passed'] for row in record['source_regression']['lanes']) == 43
    assert record['source_regression']['failed'] == 0
    assert record['independent_review_status'] == 'PENDING' and record['remaining']
    assert record['formal_resource_requests'] == []
    assert not any(record[key] for key in ('arc_solver_integrated','adaptive_cutback_implemented',
        'installed_wheel_test_executed','production_qualified','public_driver_authorized',
        'mechanics_changed','existing_source_files_changed','defaults_changed',
        'historical_evidence_reclassified','release_authorized'))


@pytest.mark.parametrize('path',[
    'src/anyfesolver/_ge_beam3_arc_geometry.py',
    'tests/test_ge_beam3_arc_geometry.py',
    'docs/reference_cases/ge_beam3_curved_p5_continuation_probe.py',
])
def test_changed_formula_or_bound_source_is_rejected(monkeypatch,path):
    original = source
    monkeypatch.setattr(__import__(__name__),'source',
        lambda item: original(item)+(b'\n# mutation\n' if item == path else b''))
    with pytest.raises(AssertionError): validate()
