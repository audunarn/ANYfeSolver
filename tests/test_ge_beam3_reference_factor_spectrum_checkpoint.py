"""Static source/evidence checks, with no mechanics or archive dependency."""

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def reject(_): raise ValueError('nonfinite')


def pairs(rows):
    result = {}
    for key, value in rows:
        if key in result: raise ValueError('duplicate')
        result[key] = value
    return result


def evidence():
    raw = (ROOT/'docs/reference_cases/ge_beam3_reference_factor_spectrum_evidence.json').read_text(encoding='utf-8')
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    assert json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n' == raw
    return result


def test_exact_source_bindings_and_no_old_source_replacement():
    record = evidence()
    assert record['baseline'] == dict(commit='cc2f50c5d50e3467a26d71e1d7276f06412ac4dc',
        tree='86c3993c075905000c533cac8006668a5597276c')
    assert record['source_normalization'] == 'UTF8_LF' and len(record['sources']) == 6
    for row in record['sources']:
        path = (ROOT/row['path']).resolve(); assert path.is_relative_to(ROOT)
        raw = path.read_text(encoding='utf-8').encode('utf-8')
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    assert {r['path'] for r in record['sources'] if r['path'].startswith('src/')} == {
        'src/anyfesolver/_native_reference_factor_spectrum.py', 'src/anyfesolver/_ge_beam3_virgin_reference_modes.py'}


def test_failed_evidence_preserved_without_fabricated_absent_output():
    record = evidence(); incident = record['incident']
    assert incident == dict(status='DENSE_REFERENCE_SPECTRUM_ACCURACY_DEFECT', tests_failed=2,
        tests_passed=1, slender_10000_accuracy_failed=True, slender_1000000_residual_rejected=True,
        slender_1000000_original_raw_output_absent=True, failed_sources_preserved=True)
    archive = record['archive']; rows = archive['files']
    assert len(rows) == archive['file_count'] == 19
    assert sum(r['bytes'] for r in rows) == archive['bytes'] == 11201
    assert len({r['path'] for r in rows}) == 19
    assert len([r for r in rows if r['path'].startswith('incident/')]) == 4
    assert len(archive['identical_pairs']) == 6
    by_path = {r['path']: r for r in rows}
    for a, b in archive['identical_pairs']:
        assert a != b
        assert by_path[a]['sha256'] == by_path[b]['sha256']
        assert by_path[a]['bytes'] == by_path[b]['bytes']


def test_no_prestress_or_qualification_scope_upgrade():
    record = evidence()
    assert record['scope'] == dict(production_qualified=False, prestressed_tangent_authorized=False,
        independent_review='PENDING', mechanics_changed=False, previous_sources_changed=False,
        defaults_changed=False, resource_requests_consumed=0, extreme_contrast_invariant_precision_unresolved=True,
        installed_wheel_checked=False)
    assert record['checks'] == dict(kernel_tests=12, slender_diagnostic_tests=4,
        native_adapter_tests=10, loaded_modal_regression_tests=23, engineering_regression_tests=2,
        final_suite_passed=True)


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_record_parser(raw):
    with pytest.raises(ValueError): json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
