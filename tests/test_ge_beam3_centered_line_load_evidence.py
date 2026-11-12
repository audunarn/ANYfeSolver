"""Static binding/adjudication checks; no mechanics or diagnostic reruns."""

import hashlib
import math
from pathlib import Path

from docs.reference_cases import ge_beam3_centered_line_load_diagnostic as diagnostic


ROOT = Path(__file__).resolve().parents[1]


def read():
    import json
    raw = (ROOT/'docs/reference_cases/ge_beam3_centered_line_load_evidence.json').read_text(encoding='utf-8').encode('ascii')
    value = json.loads(raw)
    assert raw == diagnostic.canonical(value)
    return value


def test_line_load_evidence_is_complete_development_only():
    evidence = read()
    assert evidence['base_commit'] == diagnostic.BASE
    assert evidence['schema'] == 'GE_BEAM3_CENTERED_DEAD_LINE_DEVELOPMENT_V1'
    assert evidence['policy'] == 'SPATIAL_DEAD_FORCE_PER_REFERENCE_ARCLENGTH_V1'
    assert not evidence['production_qualified'] and not evidence['native_load_integration_complete']
    assert not evidence['historical_evidence_reclassified']
    assert evidence['independent_review_status'] == 'PENDING'
    assert [(r['geometry'],r['material']) for r in evidence['records']] == [
        ('straight','elastic'),('straight','plastic'),('curved','elastic'),('curved','plastic')]
    for r in evidence['records']:
        assert r['station_count'] == 16 and r['all_translations_byte_identical']
        assert (r['plastic_station_count'] > 0) == (r['material'] == 'plastic')
        correction = float.fromhex(r['internal_load_correction_norm_hex'])
        assert math.isfinite(correction)
        if r['geometry'] == 'straight': assert correction == 0.
        else: assert correction > 1e-5
        for name,raw in r['errors_hex'].items():
            error = float.fromhex(raw)
            assert math.isfinite(error) and 0 <= error <= (1e-7 if name.startswith('parameter') else 1e-11)


def test_line_load_source_bindings_and_preserved_candidate_are_exact():
    evidence = read()
    assert diagnostic.bindings() == evidence['source_bindings']
    for path,expected in evidence['source_bindings'].items():
        raw = (ROOT/path).read_text(encoding='utf-8').encode('utf-8')
        assert len(raw) == expected['bytes']
        assert hashlib.sha256(raw).hexdigest() == expected['sha256']


def test_preserved_source_mutation_fails_before_diagnostic_mechanics(monkeypatch):
    original = diagnostic.binding
    def mutated(path):
        value = original(path)
        if path == 'src/anyfesolver/_ge_beam3_centered_mixed.py': value = {**value,'sha256':'0'*64}
        return value
    monkeypatch.setattr(diagnostic,'binding',mutated)
    import pytest
    with pytest.raises(ValueError,match='preserved source changed'):
        diagnostic.bindings()
