"""Two pre-validation captures, not reruns of qualification or accepted modes."""
import sys
import numpy as np
import pytest
from anyfesolver import _ge_beam3_native_generalized_factor_modal as factor
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from docs.reference_cases.ge_beam3_returned_mode_capture import capture_failure, analyze
from test_ge_beam3_generalized_slenderness_diagnostic import make
from test_ge_beam3_schur_line_program import save


@pytest.mark.parametrize('rho,curved,expected', (
    (10000., True, 'original signed bilinear Ritz identity failed'),
    (1000000., False, 'physical modal mass normalization')))
def test_failed_vector_capture(rho, curved, expected, tmp_path):
    model, states, inertias = make(rho, curved, np.eye(3))
    before = canonical(states)
    print(dict(stage='capture-initialization', rho=rho, curved=curved), flush=True)
    data = capture_failure(lambda: factor.solve_modes(model, states, np.zeros(18),
        inertias, np.zeros(18), bounds=(-100., 1e28), num_modes=12))
    assert data['failure'] == expected and canonical(states) == before
    assert sys.gettrace() is None and data['c'].shape[1] == 12
    save(tmp_path/'prevalidation.json', data)
    print(dict(stage='captured', failure=data['failure']), flush=True)
    result = analyze(data, lambda: None)
    save(tmp_path/'arithmetic-diagnostic.json', dict(rho=rho, curved=curved,
        comparisons=result, production_qualified=False, accepted_modes=False,
        independent_review='PENDING'))
    print({key: {k: v for k, v in value.items() if k not in ('energy', 'kinetic_gram')}
           for key, value in result.items()}, flush=True)


def test_trace_restored_on_unrelated_failure():
    def fail():
        raise ValueError('unrelated failure')
    with pytest.raises(ValueError, match='unrelated failure'):
        capture_failure(fail)
    assert sys.gettrace() is None


def test_unexpected_success_rejected():
    with pytest.raises(ValueError, match='did not occur'):
        capture_failure(lambda: None)
    assert sys.gettrace() is None
