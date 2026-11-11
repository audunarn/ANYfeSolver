"""Independent mapping, deficiency and admission controls; no owner execution."""
from fractions import Fraction

import numpy as np
import pytest
from scipy import sparse

from scripts import r8_bl5_projected_native_beta_dev as candidate
from scripts import r8_bl5_projected_envelope_norm_dev as envelope


def test_counts_require_declared_case_and_ceiling():
    case = envelope.EnvelopeCase(16, Fraction(8))
    assert candidate.validate_counts(case)['multipliers'] == 40
    with pytest.raises(ValueError, match='EnvelopeCase'):
        candidate.validate_counts({'nx': 16})


def test_reference_row_ceiling_applies_before_fine_preparation():
    case = envelope.EnvelopeCase(64, Fraction(8))
    assert candidate.validate_reference_rows(case, 16) == 52224
    with pytest.raises(ValueError, match='100k row ceiling'):
        candidate.validate_reference_rows(case, 32)


def test_numerical_predicates_remain_boolean_in_saved_diagnostics():
    import json

    report = dict(predicates={'floor': np.float64(.8) >= np.float64(.4),
                              'failed': np.bool_(False)}, beta=np.float64(.8))
    decoded = json.loads(json.dumps(candidate.json_record(report), allow_nan=False))
    assert decoded == {'predicates': {'floor': True, 'failed': False}, 'beta': .8}
    assert all(type(value) is bool for value in decoded['predicates'].values())


def test_actual_envelope_policy_schema_without_legacy_line_record():
    case = envelope.EnvelopeCase(16, Fraction(8))
    policy = dict(basis='discontinuous_p1_cell_average', span_halfcells=case.span,
                  support_endpoints_m=np.linspace(0., 1., case.supports+1).tolist())
    candidate.validate_policy(case, {'policy': policy})
    for key, value in [('basis', 'p0'), ('span_halfcells', case.span//2)]:
        with pytest.raises(ValueError, match='density policy differs'):
            candidate.validate_policy(case, {'policy': dict(policy, **{key: value})})


def test_complete_historical_graph_rejects_site_and_wheel_mutation(tmp_path):
    import hashlib
    import zipfile

    site = tmp_path/'site'
    package = site/'anyfesolver'
    package.mkdir(parents=True)
    wheel = tmp_path/'historical.whl'
    payload = b'# frozen historical module\n'
    modules = {}
    with zipfile.ZipFile(wheel, 'w') as archive:
        for i in range(417):
            name = f'anyfesolver/module_{i}.py'
            archive.writestr(name, payload)
            (site/name).write_bytes(payload)
            modules[name] = hashlib.sha256(payload).hexdigest()
    summary = dict(wheel_sha256=candidate.envelope.sha(wheel), module_sha256=modules)
    assert candidate.installed_graph(site, wheel, summary) == modules
    changed = package/'module_416.py'
    changed.write_bytes(b'# changed installed module\n')
    with pytest.raises(ValueError, match='module mismatch'):
        candidate.installed_graph(site, wheel, summary)
    changed.write_bytes(payload)
    wheel.write_bytes(wheel.read_bytes()+b'changed archive bytes')
    with pytest.raises(ValueError, match='wheel differs'):
        candidate.installed_graph(site, wheel, summary)


def test_saved_owner_rejects_changed_manifest_before_state_load(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, 'SOURCE', tmp_path)
    folder = tmp_path/'docs/agent_plans/development/r8-bl5-projected-assessment-20261003'
    folder.mkdir(parents=True)
    (folder/'raw-manifest-v8.json').write_text('{}')
    with pytest.raises(ValueError, match='campaign manifest differs'):
        candidate.frozen_owner(envelope.EnvelopeCase(16, Fraction(8)), tmp_path/'unused-owner')


def scalar_fixture():
    case = envelope.EnvelopeCase(16, Fraction(8))
    n = 2*case.supports
    d = np.diag(1/np.sqrt(envelope.density_mass_diagonal(case)))
    raw = dict(projection=envelope.projection(case), density_L2_scale=d,
               scalar_lower=np.eye(n), reference_upper=1.5*np.eye(n))
    rho = np.hypot(.05, .05)/float(envelope.ELL)
    product = max(3., max(2., 1+2*rho*rho)*max(3., 1+rho*rho))
    floor = 1/np.sqrt(product)/np.sqrt(1.001)-envelope.CRITERIA['arithmetic_beta_margin']
    contract = dict(physical_L2_basis_scale=np.diag(d).tolist(), safe_case_floor=floor, cs_squared_upper=1.5)
    return case, raw, contract


def test_floor_is_bound_to_scalar_equation_and_projection():
    case, raw, contract = scalar_fixture()
    assert candidate.scalar_floor(contract, raw, case) == pytest.approx(contract['safe_case_floor'])
    for key in ('safe_case_floor', 'cs_squared_upper'):
        changed = dict(contract, **{key: contract[key]*1.01})
        with pytest.raises(ValueError, match='scalar floor'):
            candidate.scalar_floor(changed, raw, case)
    raw['projection'][0, 1] = 0.
    with pytest.raises(ValueError, match='density differs'):
        candidate.scalar_floor(contract, raw, case)


def test_complete_spectrum_preserves_deficient_duplicate_direction():
    action = np.array([[1., 1.], [2., 2.]])
    values, _, _, _ = candidate.spectrum(action.T@action, np.eye(2), np.ones(2))
    assert values[0] <= 1e-14
    assert not np.all(values > 1e-14)
    assert len(values) == 2


def test_reciprocal_basis_and_common_units_preserve_spectrum():
    g = np.array([[.8, .1], [.1, .5]])
    m = np.array([[2., .2], [.2, 1.]])
    original = candidate.spectrum(g, m, np.ones(2))[0]
    d = np.diag((.02, 50.))
    transformed = candidate.spectrum(d@g@d, d@m@d, (50., .02))[0]
    np.testing.assert_allclose(transformed, original, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(candidate.spectrum(1e6*g, 1e6*m, np.ones(2))[0], original, rtol=1e-13)


def test_native_gram_rejects_nonfinite_and_invalid_sparse_structure():
    with pytest.raises(ValueError, match='finite real'):
        candidate.spectrum(np.diag((np.nan, 1.)), np.eye(2), np.ones(2))
    raw = dict(C_shape=np.array((2, 2)), C_data=np.array((1.,)),
               C_indices=np.array((2,)), C_indptr=np.array((0, 1, 1)))
    with pytest.raises(ValueError, match='invalid CSR'):
        candidate.sparse_input(raw, 'C', (2, 2))


def test_fresh_output_guards_saved_owner_and_existing_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, 'SOURCE', tmp_path)
    root = tmp_path/'saved'; root.mkdir()
    with pytest.raises(ValueError, match='saved owner'):
        candidate.fresh_output(root/'assessment.json', root)
    output = tmp_path/'new.json'; output.with_suffix('.npz').touch()
    with pytest.raises(ValueError, match='fresh'):
        candidate.fresh_output(output, root)
    with pytest.raises(ValueError, match='owned'):
        candidate.fresh_output(tmp_path.parent/'outside.json', root)


def test_work_mapping_keeps_retained_directors_and_rejects_wrong_column(monkeypatch):
    case = envelope.EnvelopeCase(16, Fraction(8))
    ns, cells, cols = case.counts['shell_nodes'], case.halfcells, case.counts['multipliers']
    # Capture independently assembled work by returning its negative as C.T.
    # This exercises mapping, supports, physical units and an isolated mutation.
    original = candidate.require

    def observe(condition, message):
        if message == 'independent all-column physical work mismatch':
            return
        original(condition, message)

    monkeypatch.setattr(candidate, 'require', observe)
    shell_root = np.arange(0, 6*ns, 6*(case.nx+1))
    fixed = np.r_[np.concatenate([i+np.arange(6) for i in shell_root]), 6*ns+np.arange(5)]
    c = sparse.csr_matrix((cols, case.counts['full_q']))
    work, h, _ = candidate.physical_work(case, c, fixed, np.eye(3), np.array(((1., 0., 0.), (0., 1., 0.))),
                                       np.array((0., 0., -.05)), np.array((0., 0., .0175)))
    monkeypatch.setattr(candidate, 'require', original)
    all_fields = work['all_fields']
    rows, columns = np.nonzero(work['full_work'].T)
    values = -work['full_work'].T[rows, columns]/work['scale'][rows]
    c = sparse.coo_matrix((values, (rows, all_fields[columns])), shape=c.shape).tocsr()
    actual, _, error = candidate.physical_work(case, c, fixed, np.eye(3), np.array(((1., 0., 0.), (0., 1., 0.))),
                                             np.array((0., 0., -.05)), np.array((0., 0., .0175)))
    assert error <= 1e-12
    assert h.shape[0] == len(actual['slots'])
    retained = np.array([case.counts['physical_dofs']+24*m+3*half+a
                        for m in range(cells//2) for half in (0, 1) for a in range(3)])
    assert np.all(np.isin(retained, actual['slots']))
    assert np.max(np.abs(actual['full_work'][np.isin(all_fields, retained)])) > 0.
    # Independent constant-density cell integrals: +Fy acting at z=-.05
    # supplies +Mx=.05 per length to the retained beam director. +Fz gives
    # h/2 at the first beam endpoint, and shell Fy work has the opposite sign.
    index = {int(slot): i for i, slot in enumerate(all_fields)}
    assert actual['full_work'][index[retained[0]], 2*case.supports] == pytest.approx(.05/cells)
    assert actual['full_work'][index[6*ns+2], 4*case.supports] == pytest.approx(.5/cells)
    assert actual['full_work'][index[1], 2*case.supports] == pytest.approx(-.5/case.nx)
    changed = c.copy().tolil(); changed[0, retained[0]] += .001
    with pytest.raises(ValueError, match='work mismatch'):
        candidate.physical_work(case, changed.tocsr(), fixed, np.eye(3), np.array(((1., 0., 0.), (0., 1., 0.))),
                                np.array((0., 0., -.05)), np.array((0., 0., .0175)))


def test_installed_graph_rejects_incomplete_historical_modules(tmp_path):
    wheel = tmp_path/'wheel'; wheel.write_bytes(b'frozen')
    summary = dict(wheel_sha256=envelope.sha(wheel), module_sha256={'anyfesolver/a.py': 'wrong'})
    with pytest.raises(ValueError, match='417-module'):
        candidate.installed_graph(tmp_path, wheel, summary)
