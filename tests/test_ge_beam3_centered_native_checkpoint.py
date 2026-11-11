"""Read-only development bindings and artifact checks; not qualification."""

import hashlib
import json
from pathlib import Path

import pytest

from docs.reference_cases import ge_beam3_centered_native_source_map as source_map


ROOT = Path(__file__).resolve().parents[1]


def record():
    raw = source_map.source('docs/reference_cases/ge_beam3_centered_native_development_evidence.json')
    value = json.loads(raw)
    assert raw == source_map.canonical(value)
    return value


def test_centered_native_source_binding_and_development_boundary():
    evidence = record()
    assert evidence['source_regression']['failed'] == 0
    assert evidence['installed_test']['passed'] == 1 and evidence['installed_test']['failed'] == 0
    assert evidence['independent_review_status'] == 'PENDING'
    assert not any(evidence[k] for k in ('production_qualified', 'public_selector_authorized',
        'release_authorized', 'defaults_changed', 'historical_evidence_reclassified'))
    assert evidence['formal_resource_requests'] == [] and evidence['remaining']
    raw = source_map.source(source_map.MANIFEST)
    assert hashlib.sha256(raw).hexdigest() == evidence['source_map_sha256']
    made = source_map.build()
    assert source_map.canonical(made) == raw
    assert len(made['outputs']) == 36
    assert len(source_map.FILES) == 8


@pytest.mark.parametrize('kind', ['new', 'preserved', 'research_import'])
def test_source_mutations_cannot_keep_the_development_binding(monkeypatch, kind):
    original = source_map.source
    target = (source_map.TARGET+'/positions.py' if kind != 'preserved' else
              'src/anyfesolver/_ge_beam3_p5_coordinates/positions.py')
    def altered(path):
        raw = original(path)
        if path == target:
            raw += b'\nimport docs\n' if kind == 'research_import' else b'\n# mutation\n'
        return raw
    monkeypatch.setattr(source_map, 'source', altered)
    if kind == 'new':
        assert source_map.canonical(source_map.build()) != original(source_map.MANIFEST)
    else:
        with pytest.raises(ValueError): source_map.build()


def test_installed_archive_inventory_binds_identical_results():
    evidence = record(); files = evidence['archive_files']; installed = evidence['installed_test']
    assert len(files) == len({f['name'] for f in files}) == 6
    assert sum(f['bytes'] for f in files) == evidence['archive_bytes']
    cycles = [f for f in files if f['name'].startswith('cycle-')]
    assert len(cycles) == 2
    assert all((f['bytes'], f['sha256']) == (installed['result_bytes'], installed['result_sha256']) for f in cycles)
    assert installed['cycles_identical'] and installed['translated_native_solution_verified']
    wheel = next(f for f in files if f['name'].endswith('.whl'))
    assert (wheel['bytes'], wheel['sha256']) == (installed['wheel_bytes'], installed['wheel_sha256'])


def test_preserved_archive_bytes_and_installed_provenance():
    evidence = record(); root = Path(evidence['archive_root'])
    if not all((root/f['name']).is_file() for f in evidence['archive_files']):
        pytest.skip('development archive is not mounted; artifact verification not executed')
    for item in evidence['archive_files']:
        raw = (root/item['name']).read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    first = (root/'cycle-1.json').read_bytes()
    assert first == (root/'cycle-2.json').read_bytes()
    output = json.loads(first)
    assert output['formulation'] == evidence['formulation_id']
    assert output['reference_evaluation'] == evidence['reference_evaluation']
    assert output['source_map_sha256'] == evidence['source_map_sha256']
    for key in ('imports_isolated', 'research_imports_forbidden', 'restart_exact',
                'sub_ulp_axial_commit_verified', 'translated_native_solution_verified'):
        assert output[key]
    assert not output['production_qualified']
    receipt = json.loads((root/'receipt.json').read_bytes())
    assert receipt['result_sha256'] == evidence['installed_test']['result_sha256']
    assert len(receipt['dependencies']) == 10 and not receipt['release_authorized']
