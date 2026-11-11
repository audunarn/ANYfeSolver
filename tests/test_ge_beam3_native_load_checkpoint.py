"""Static source/state-schema handoff checks; no new mechanics execution."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from docs.reference_cases import ge_beam3_native_load_source_map as binding


ROOT = Path(__file__).resolve().parents[1]


def test_load_development_record_binds_extent_and_does_not_authorize_public_routes():
    raw = binding.source('docs/reference_cases/ge_beam3_native_load_development_evidence.json')
    evidence = json.loads(raw)
    assert binding.canonical(evidence) == raw
    assert evidence['source_regression']['passed'] == 106 and evidence['source_regression']['failed'] == 0
    assert sum(row['passed'] for row in evidence['source_regression']['lanes']) == 106
    assert evidence['independent_review_status'] == 'PENDING'
    assert not any(evidence[k] for k in ('public_driver_authorized','solver_load_schedule_integration_complete',
        'global_solver_restart_qualified','loaded_modal_or_buckling_qualified','installed_wheel_test_executed',
        'two_formal_cycles_executed','production_qualified','defaults_changed','existing_production_files_changed',
        'historical_evidence_reclassified','release_authorized'))
    assert evidence['formal_resource_requests'] == [] and evidence['remaining']
    source_map = binding.source(binding.MANIFEST)
    assert hashlib.sha256(source_map).hexdigest() == evidence['source_map_sha256']
    assert binding.canonical(binding.build()) == source_map
    assert len(binding.build()['outputs']) == 46 and len(binding.FILES) == 6


@pytest.mark.parametrize('mode',['preserved_framework','new_source','research_import'])
def test_native_load_source_mutations_do_not_keep_the_binding(monkeypatch,mode):
    original = binding.source
    target = ('src/anyfesolver/nonlinear_state.py' if mode == 'preserved_framework' else binding.TARGET+'/state.py')
    def altered(path):
        value = original(path)
        if path == target: value += b'\nimport docs\n' if mode == 'research_import' else b'\n# mutation\n'
        return value
    monkeypatch.setattr(binding,'source',altered)
    if mode == 'new_source':
        assert binding.canonical(binding.build()) != original(binding.MANIFEST)
    else:
        with pytest.raises(ValueError): binding.build()


@pytest.mark.parametrize('name',['_keys','_load'])
def test_strict_parser_and_duplicate_nonfinite_guards_are_preserved_exactly(name):
    def function(path):
        tree = ast.parse(binding.source(path))
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name)
    original = function('src/anyfesolver/_ge_beam3_p5_centered/codec.py')
    successor = function('src/anyfesolver/_ge_beam3_p5_loads/codec.py')
    assert ast.dump(original) == ast.dump(successor)
