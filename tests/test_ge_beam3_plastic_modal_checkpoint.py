"""Static explicit-policy spectrum evidence audit; no spectral rerun."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import pytest
from test_ge_beam3_station_resultant_checkpoint import strict, sha

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'docs/reference_cases/ge_beam3_plastic_modal_development_evidence.json'


def test_canonical_record_preserves_material_policy_boundary():
    value = strict(RECORD.read_bytes())
    assert value['production_qualified'] is False and value['independent_review'] == 'PENDING'
    assert value['policies']['frozen'] == 'FROZEN_PLASTIC_COORDINATES_ELASTIC_PERTURBATION'
    assert value['policies']['algorithmic_interpretation'] == 'LAST_INCREMENT_OPERATOR_ONLY_NOT_PHYSICAL_VIBRATION_AUTHORITY'
    assert value['scope']['covariance_tolerance'] == value['scope']['ritz_tolerance'] == '1e-11'
    assert not value['scope']['state_advanced'] and not value['scope']['buckling_factor_authorized']
    assert not value['scope']['certified_intervals'] and len(value['remaining']) == 5


@pytest.mark.parametrize('raw', [b'{"a":0,"a":1}\n', b'{"a":NaN}\n', b'{"a":Infinity}\n', b'{ "a":0}\n'])
def test_evidence_parser_remains_strict(raw):
    with pytest.raises(ValueError): strict(raw)


def test_source_bindings_and_no_runtime_research_import():
    value = strict(RECORD.read_bytes())
    for entry in value['sources']:
        raw = (ROOT/entry['path']).read_bytes().replace(b'\r\n', b'\n')
        assert len(raw) == entry['bytes'] and sha(raw) == entry['sha256']
        if entry['path'].startswith('src/'):
            for node in ast.walk(ast.parse(raw)):
                if isinstance(node, ast.Import): imports = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom): imports = [node.module]
                else: continue
                assert all(name.split('.')[0] in sys.stdlib_module_names | {'anyfesolver', 'numpy', 'scipy'} for name in imports)


def test_exact_archive_and_deterministic_pairs():
    value = strict(RECORD.read_bytes()); root = Path(value['archive']['path'])
    if not root.exists(): pytest.skip('explicit external plastic-modal development archive required')
    raw = (root/'archive-manifest.json').read_bytes()
    assert len(raw) == value['archive']['manifest_bytes'] and sha(raw) == value['archive']['manifest_sha256']
    manifest = json.loads(raw)
    for change in value.get('format_only_changes', []):
        assert change['kind'] == 'REMOVE_ONE_TRAILING_NEWLINE'
        final = (ROOT/change['path']).read_bytes().replace(b'\r\n', b'\n')
        tested = (root/'sources'/change['path']).read_bytes()
        assert tested == final+b'\n'
        assert sha(tested) == change['tested_sha256'] and sha(final) == change['final_sha256']
    assert len(manifest['files']) == 221 and sum(r['bytes'] for r in manifest['files']) == 11150531
    for row in manifest['files']:
        raw = (root/row['path']).read_bytes()
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
    for run in value['inventories']:
        suite = ET.parse(root/run['lane']/'junit.xml').getroot().find('testsuite')
        assert suite is not None
        for key in ('tests', 'errors', 'failures', 'skipped'): assert int(suite.attrib[key]) == run[key]
        assert suite.attrib['time'] == run['seconds']
    assert len(value['deterministic_pairs']) == 52
    for row in value['deterministic_pairs']:
        a = (root/'cycle-a'/row['path']).read_bytes(); b = (root/'cycle-b'/row['path']).read_bytes()
        assert a == b and len(a) == row['bytes'] and sha(a) == row['sha256']


def test_exact_seven_new_paths_and_no_existing_mechanics_delta():
    value = strict(RECORD.read_bytes())
    def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).splitlines()
    assert git('rev-parse', value['base_commit']+'^{tree}') == [value['base_tree']]
    changed = set(git('diff', '--name-only', value['base_commit']))
    changed.update(git('ls-files', '--others', '--exclude-standard', '--', 'docs/', 'src/', 'tests/'))
    expected = {row['path'] for row in value['sources']} | {
        'docs/reference_cases/ge_beam3_plastic_modal_development_evidence.json',
        'docs/agent_plans/GE_BEAM3_PAIRED_PLASTIC_MODAL_DEVELOPMENT_CHECKPOINT.md',
        'tests/test_ge_beam3_plastic_modal_checkpoint.py'}
    assert changed == expected
    assert git('diff', '--name-only', '--diff-filter=MDR', value['base_commit'], '--', 'src/') == []
