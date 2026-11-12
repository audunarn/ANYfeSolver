"""M16 negative controls: measure what relative error the frozen assessment detects.

First version (evidence m16-claude-05-controls) assumed a 1e-11 relative error
would be detected in derived measures; it was not (linear 124 detections,
derived 0). That failure is preserved. This version measures the detection
threshold for each criterion instead of assuming it. Criteria are unchanged.
"""
import json
import numpy as np
import prepared_static_response_m16_assessment as assessment
from anyfesolver import RecoveryConfig
from test_prepared_static_evaluator import _shell_model, _shell_case


def _run_with_reference_error(monkeypatch, relative):
    original = assessment._canonical_reference

    def perturbed(record, requested):
        made = original(record, requested)
        return {name: np.asarray(value, dtype=float) * (1.0 + relative) for name, value in made.items()}

    monkeypatch.setattr(assessment, '_canonical_reference', perturbed)
    recovery = RecoveryConfig(include_stresses=True, include_reactions=False, components=assessment.SHELL_COMPONENTS)
    try:
        return assessment.assess('control', _shell_model(), {'p': _shell_case(0, pressure=2.).load_case},
                                 [{'p': 1.3}, {'p': -0.4}], recovery=recovery)
    finally:
        monkeypatch.setattr(assessment, '_canonical_reference', original)


def test_detection_thresholds_of_frozen_criteria(monkeypatch):
    rows = []
    for relative in (1e-15, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7):
        summary = _run_with_reference_error(monkeypatch, relative)
        rows.append(dict(relative=relative, linear_entries=summary['linear_entries'],
                         linear_failures=summary['linear_failures'], derived_entries=summary['derived_entries'],
                         derived_failures=summary['derived_failures']))
    print('M16 control detection:', json.dumps(rows))
    by = {row['relative']: row for row in rows}
    assert by[1e-15]['linear_failures'] == 0 and by[1e-15]['derived_failures'] == 0
    assert by[1e-11]['linear_failures'] > 0
    assert any(row['derived_failures'] > 0 for row in rows)
