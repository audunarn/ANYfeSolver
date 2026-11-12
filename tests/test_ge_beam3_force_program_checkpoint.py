"""Static development binding; does not run or authorize beam mechanics."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from docs.reference_cases import ge_beam3_native_load_source_map as native


ROOT = Path(__file__).resolve().parents[1]
RECORD = 'docs/reference_cases/ge_beam3_force_program_development_evidence.json'


def source(path): return (ROOT/path).read_text(encoding='utf-8').encode('utf-8')


def binding(raw): return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def validate():
    raw = source(RECORD); record = json.loads(raw)
    assert native.canonical(record) == raw
    for group in ('source_bindings','preserved_framework'):
        for path,expected in record[group].items():
            assert binding(source(path)) == expected, path
    assert hashlib.sha256(native.source(native.MANIFEST)).hexdigest() == record['preserved_native_source_map_sha256']
    assert native.canonical(native.build()) == native.source(native.MANIFEST)
    return record


def test_force_program_checkpoint_is_bound_and_nonqualifying():
    record = validate()
    assert record['base_commit'] == '45a3fe533dbb6623f01d8bf10a14a508b8aef3a6'
    assert record['source_regression']['passed'] == 151
    assert record['source_regression']['failed'] == 0
    assert sum(row['passed'] for row in record['source_regression']['lanes']) == 151
    assert record['independent_review_status'] == 'PENDING' and record['remaining']
    assert record['formal_resource_requests'] == []
    assert not any(record[key] for key in ('production_qualified','public_driver_authorized',
        'existing_production_files_changed','mechanics_changed','defaults_changed',
        'historical_evidence_reclassified','release_authorized','installed_wheel_test_executed',
        'fresh_process_program_cycles_executed'))


@pytest.mark.parametrize('path',[
    'src/anyfesolver/_ge_beam3_load_program.py',
    'tests/test_ge_beam3_load_program.py',
    'src/anyfesolver/linalg.py',
])
def test_changed_bound_source_is_rejected(monkeypatch,path):
    original = source
    monkeypatch.setattr(__import__(__name__), 'source',
        lambda item: original(item)+(b'\n# mutation\n' if item == path else b''))
    with pytest.raises(AssertionError): validate()


def test_private_driver_uses_no_research_or_legacy_mechanics_imports():
    tree = ast.parse(source('src/anyfesolver/_ge_beam3_load_program.py'))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): modules.extend(alias.name for alias in node.names)
        elif isinstance(node,ast.ImportFrom): modules.append(node.module or '')
    assert not any(name.split('.')[0] in ('docs','tests') for name in modules)
    assert not any('quadratic_beam' in name or 'corotational' in name for name in modules)
    assert b'_ge_beam3_load_program' not in source('src/anyfesolver/__init__.py')
