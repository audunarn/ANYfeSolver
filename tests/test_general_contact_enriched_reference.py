"""Five fixed V14 discriminators; the final case executes the COMPLETE new assay.

The first four are small independent analytic/admission controls, not retained
qualification or reruns of W13 science. No saved producer is a reference seed.
"""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import general_contact_enriched_reference_geometry as geometry
from scripts.general_contact_enriched_reference_geometry import ContactOracle, ReferenceRefused, skin, bubbles
from scripts.general_contact_enriched_reference_bounds import coefficients, bernstein_range, global_gap_certificate
from scripts import general_contact_enriched_reference as reference
from scripts import general_contact_enriched_reference_run as runner


def analytic_oracle(monkeypatch, *, target_z=-.2, beta=None, order=8, cancel=lambda: False):
    """Synthetic physical surface fixture, never a surrogate matched reference."""
    coords = np.array(((-.5, -.5, 0), (.5, -.5, 0), (.5, .5, 0), (-.5, .5, 0),
        (-2, -2, target_z), (2, -2, target_z), (2, 2, target_z), (-2, 2, target_z)), dtype=float)
    nodes = {i+1: SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for i, p in enumerate(coords)}
    elements = {i: SimpleNamespace(node_ids=ids, physical_reference_director=np.array((0., 0., 1.)))
                for i, ids in ((1, (1, 2, 3, 4)), (2, (5, 6, 8)), (3, (6, 7, 8)))}
    dofs = SimpleNamespace(total_dofs=48, get_node_dofs=lambda n: np.arange(6*(n-1), 6*n))
    model = SimpleNamespace(mesh=SimpleNamespace(nodes=nodes, elements=elements, dof_manager=dofs))
    def identity(m):
        return hashlib.sha256(repr([(n, p.x, p.y, p.z) for n, p in m.mesh.nodes.items()]).encode()).hexdigest()
    monkeypatch.setattr(geometry, '_raw_identity', identity)
    definitions = tuple(SimpleNamespace(source_element=1, target_element=t, source_offset=0., target_offset=0.) for t in (2, 3))
    u = np.zeros(52)
    if beta is not None:
        u[48:] = beta
    return ContactOracle(model, definitions, u, np.array((1., 0., 0., 0.)), order, cancel=cancel)


def test_independent_physical_field_and_positive_pressure_wrench(monkeypatch, record_property):
    oracle = analytic_oracle(monkeypatch)
    base = oracle.base_value
    assert np.max(np.abs(base['support']-.25)) <= 1e-14
    assert np.max(np.abs(base['gap']-.2)) <= 1e-14
    assert abs(base['work']+.2) <= 1e-14
    sf, sm = reference.rigid_wrench(oracle, 'source')
    tf, tm = reference.rigid_wrench(oracle, 'target')
    assert sf[2] > 0 and tf[2] < 0  # Balance alone cannot detect a sign reversal.
    assert np.linalg.norm(sf-np.array((0., 0., 1.))) <= 1e-8
    assert np.linalg.norm(sf+tf) <= 1e-8 and np.linalg.norm(sm+tm) <= 1e-8
    r, s = np.array((-.3, .2)), np.array((.1, -.4))
    q = oracle.base_sources[1][0]
    beta = np.array((.01, -.002, .003, .001))
    p, pr, ps, n = skin(r, s, q, beta, np.array((0., 0., 1.)))
    b, br, bs = bubbles(r, s)
    assert np.array_equal(p[:, 2], b @ beta)
    assert np.array_equal(pr[:, 2], br @ beta) and np.array_equal(ps[:, 2], bs @ beta)
    record_property('positive_pressure_physical_source_force', json.dumps(sf.tolist()))


def test_bicubic_global_gap_certifies_or_refuses_interior_not_corners(monkeypatch, record_property):
    oracle = analytic_oracle(monkeypatch, target_z=0., beta=(1e-5, 2e-6, -3e-6, 1e-6))
    cert = global_gap_certificate(oracle, oracle.base_pose)
    assert cert['lower_m'] >= -50e-6 and not cert['sampled_surrogate']
    bad = oracle.base_pose.copy(); bad[48] = -1e-4
    # Every boundary corner remains exactly at zero; the interior violates.
    with pytest.raises(ReferenceRefused, match='violates|unresolved'):
        global_gap_certificate(oracle, bad)
    corners, beta, normal = oracle.base_sources[1]
    polynomials = coefficients(corners, beta, normal)
    lo, hi = bernstein_range(polynomials[2], (Fraction(-1), Fraction(1), Fraction(-1), Fraction(1)))
    assert lo <= Fraction.from_float(float(beta[0])) <= hi
    original_vertices = oracle.vertices
    def reversed_target_pose(pose):
        source, target = original_vertices(pose)
        return source, {tid: face[[0, 2, 1]] for tid, face in target.items()}
    with monkeypatch.context() as patch:
        patch.setattr(oracle, 'vertices', reversed_target_pose)
        with pytest.raises(ReferenceRefused, match='direction'):
            global_gap_certificate(oracle, oracle.base_pose)
    record_property('global_gap_certificate', json.dumps(cert))


def test_complete238_work_inventory_and_live_mutation_cancellation(monkeypatch):
    oracle = analytic_oracle(monkeypatch)
    # Actual independent-reader mutation/cancellation, not an early unrelated error.
    oracle.model.mesh.nodes[1].z = 1e-3
    with pytest.raises(ReferenceRefused, match='mutated'):
        oracle.evaluate(oracle.base_pose)
    oracle.model.mesh.nodes[1].z = 0.
    from anyfesolver.control import SolveCancelled
    oracle.cancel = lambda: True
    with pytest.raises(SolveCancelled):
        oracle.evaluate(oracle.base_pose)
    # Algebraic normalized-gap fixture gives every coordinate a distinct derivative.
    # This exercises the actual238-loop/+/-/h/order consumer, not mechanics replay.
    weights = np.arange(1, 239, dtype=float)*1e-8
    z = np.arange(1, 26, dtype=float)/325
    class AlgebraicOracle:
        def __init__(self, model, definitions, u, forces, order, **kwargs):
            self.base_value = self.evaluate(u)
        def evaluate(self, u):
            # Separate b and M vary, while b/M has this analytic derivative.
            M = np.arange(1, 26, dtype=float)*(1+1e-3*np.sum(u))
            g = np.full(25, .2+weights @ u)
            return dict(work=float(-z @ g), gap=g, support=M,
                        weighted_gap=M*g, signature=(), union_certificate={'admitted': True})
    monkeypatch.setattr(reference, 'ContactOracle', AlgebraicOracle)
    monkeypatch.setattr(reference, 'coordinate_inventory', lambda model, system: {i: {'coordinate': i} for i in range(238)})
    result = reference.full_work(None, None, (), np.zeros(238), z, -weights)
    assert result['development_complete']
    assert len(result['coordinate_inventory']) == 238 and [r['h'] for r in result['records']] == [1e-5, 3e-6]
    assert all([o['order'] for o in row['orders']] == [4, 8] and
               all(o['gap_gradient'].shape == (25, 238) for o in row['orders']) for row in result['records'])


def test_exact_producer_binding_original_controls_and_no_seed(tmp_path):
    raw = tmp_path/'producer.json'; closure = tmp_path/'closure.json'; baseline = tmp_path/'before.json'
    raw.write_text(json.dumps(dict(status='DEVELOPMENT_NCP_CONVERGED', scientific_acceptance=False,
        coordinates=[0]*238, nodal_forces=[0]*25)))
    closure.write_text(json.dumps(dict(status='PASS_SCOPED_DEVELOPMENT_CLOSED_NOT_QUALIFICATION',
        raw_pytest_exit=0, verification={'active_raw_drift': 0})))
    baseline.write_text(json.dumps({'files': {}}))
    binding = dict(files={str(p): runner.sha(p) for p in (raw, closure, baseline)},
        source_before=str(baseline), producer_raw=str(raw), producer_closure=str(closure))
    assert len(runner.verify_binding(binding)['coordinates']) == 238
    raw.write_text(raw.read_text()+' ')
    with pytest.raises(ReferenceRefused, match='binding'):
        runner.verify_binding(binding)
    c = np.zeros((2, 174)); c[0, 0] = 1; c[1, 6] = 1
    descriptor = dict(control_matrix=c.tolist(), control_values=[0, 0], control_scales=[1e-9, 1e-9])
    assert reference.control_predicate(descriptor, np.zeros(238))['error'] == 0
    u = np.zeros(238); u[0] = 2e-15
    with pytest.raises(ReferenceRefused, match='control'):
        reference.control_predicate(descriptor, u)


def test_relocated_source_map_binds_original_bytes_and_complete_inventory(tmp_path):
    source = tmp_path/'relocated.py'
    source.write_bytes(b'original source')
    original = {'scripts/original.py': runner.sha(source)}
    row = dict(name='scripts/original.py', sha256=runner.sha(source), resolved=source.name)
    assert runner._mapped_paths([row], original, tmp_path) == {'scripts/original.py': source}
    with pytest.raises(ReferenceRefused, match='incomplete'):
        runner._mapped_paths([], original, tmp_path)
    with pytest.raises(ReferenceRefused, match='duplicate'):
        runner._mapped_paths([row, row], original, tmp_path)
    source.write_bytes(b'wrong source')
    with pytest.raises(ReferenceRefused, match='binding'):
        runner._mapped_paths([dict(row, sha256=runner.sha(source))], original, tmp_path)
    with pytest.raises(ValueError, match='exact V15 binding'):
        runner.run(tmp_path/'output', source, resolved_inputs=tmp_path/'unused')


def test_original_matched238_complete_independent_seven_gates(record_property, tmp_path):
    out = Path(os.environ.get('ENRICHED_REFERENCE_OUTPUT', tmp_path))
    binding = Path(os.environ.get('ENRICHED_REFERENCE_BINDING', runner.W/'INPUT_BINDING.json'))
    result = runner.run(out/'reference', binding, resolved_inputs=os.environ.get('ENRICHED_REFERENCE_INPUT_MAP'))
    record_property('assessment_file', str(out/'reference/ASSESSMENT.json'))
    record_property('assessment_sha256', runner.sha(out/'reference/ASSESSMENT.json'))
    record_property('actual_seven', json.dumps(result['gates'], sort_keys=True))
    assert result['status'] == 'PASS'
    assert tuple(result['gates']) == reference.GATE_NAMES
    assert all(v is True for v in result['gates'].values())
    assert result['independent_reference']['initialization'] == 'ZERO_UNLOADED_NO_PRODUCTION_SEED'
    assert result['scientific_acceptance'] is False
