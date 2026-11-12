"""Second G3b development fixture; independent assembly, not formal acceptance."""
from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.fe_core import FEModel, Material
from anyfesolver._ge_beam3_g1_element import ElasticElement
from anyfesolver._ge_beam3_g1_elastic import ElasticSection, canonical
from anyfesolver._ge_beam3_g1_operator import ElasticOperator
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from anyfesolver._ge_beam3_g3b_reference import B3TranslationReferenceProblem, B3_POLICY

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT/'docs/reference_cases/ge_beam3_g3_graph_fixtures_v1.json').read_text())
MB3 = next(g for g in FIXTURE['mixed_graphs'] if g['id'] == 'M_B3')


def invariant(a, b):
    a, b = np.asarray(a), np.asarray(b)
    error = np.linalg.norm(a-b)/max(1., np.linalg.norm(a), np.linalg.norm(b))
    assert np.isfinite(error) and error <= 1e-11


def parts():
    nodes = {n: np.array(x, dtype=float) for n, x in MB3['nodes']}
    d = MB3['native_element']; ids = tuple(d['nodes'])
    native = ElasticElement(d['id'], ids, Reference([nodes[n] for n in ids], np.tile(np.eye(3), (3, 1, 1))),
                            ElasticSection.isotropic(**FIXTURE['materials']['native_isotropic']))
    m = FIXTURE['materials']['legacy_scalar']
    section = {k: deepcopy(v) for k, v in m.items() if k not in ('E', 'G', 'nu')}
    legacy = QuadraticBeamElement(MB3['other_element']['id'], list(MB3['other_element']['nodes']), cross_section=section)
    material = Material('G3b frozen scalar', m['E'], m['nu'])
    assert material.shear_modulus == m['G']
    return native, legacy, material, dict(nodes=nodes, fixed_nodes=tuple(MB3['fixed_nodes']), tie=deepcopy(MB3['tie']))


def make():
    native, legacy, material, kw = parts()
    return B3TranslationReferenceProblem(native, legacy, material, **kw)


def load(p, *, moment=True):
    f = np.zeros(36); i = p.ids.index(MB3['load_node'])*6
    f[i:i+3] = FIXTURE['programs']['mixed_force']
    if moment: f[i+3:i+6] = FIXTURE['programs']['mixed_moment']
    return f


def independent_full_reference():
    # Fresh elements, independent global assembly and explicit multiplier rows.
    # The unchanged family operators are permitted by the frozen G3 contract.
    native, legacy, material, kw = parts(); ids = sorted(kw['nodes'])
    model = FEModel('independent reference')
    for n in ids: model.add_node(n, *kw['nodes'][n])
    model.add_element(native.element_id, native); model.add_element(legacy.element_id, legacy)
    x = native.operator.reference.coordinates
    h = native.operator.evaluate(x, np.zeros((3, 3)), native.operator.reference.nodal_triads,
                                 np.tile(np.eye(3), (2, 1, 1)), np.zeros(18))['hessian']
    H = np.zeros((60, 60))
    native_dofs = [6*ids.index(n)+j for n in native.node_ids for j in range(6)]
    native_all = native_dofs+list(range(36, 60))
    for i, gi in enumerate(native_all):
        for j, gj in enumerate(native_all): H[gi, gj] += h[i, j]
    legacy_dofs = [6*ids.index(n)+j for n in legacy.node_ids for j in range(6)]
    b = legacy.compute_stiffness_matrix(model.mesh, material)
    for i, gi in enumerate(legacy_dofs):
        for j, gj in enumerate(legacy_dofs): H[gi, gj] += b[i, j]
    J = np.zeros((15, 60)); row = 0
    for n in sorted(kw['fixed_nodes']):
        for d in range(6): J[row, 6*ids.index(n)+d] = 1; row += 1
    for d in range(3):
        J[row+d, 6*ids.index(103)+d] = 1
        J[row+d, 6*ids.index(201)+d] = -1
    return H, J, native, legacy, material, model, native_dofs, legacy_dofs


def test_mb3_reference_kkt_smoke():
    p = make(); f = load(p); result = p.solve(f)
    H, J, native, legacy, material, model, nd, ld = independent_full_reference()
    system = np.block([[H, J.T], [J, np.zeros((15, 15))]])
    assert p.size == 36 and p.T.shape == (36, 21) and system.shape == (75, 75)
    ref = np.linalg.solve(system, np.r_[f, np.zeros(24+15)])
    invariant(result['u'], ref[:36]); invariant(result['internal'], ref[36:60])
    invariant(result['multipliers'], ref[60:]); invariant(p.J, J[:, :36])
    invariant(result['energy'], ref[:60] @ H @ ref[:60]/2)
    invariant(result['energy']*2, result['u'] @ f)
    force = result['support_reactions'].reshape(6, 6)+f.reshape(6, 6)
    invariant(force[:, :3].sum(axis=0), np.zeros(3))
    invariant((force[:, 3:]+np.cross(p.positions, force[:, :3])).sum(axis=0), np.zeros(3))
    tie = result['tie_forces'].reshape(6, 6)
    invariant(tie[:, :3].sum(axis=0), np.zeros(3)); assert not np.any(tie[:, 3:])
    invariant(tie[p.ids.index(103)], -tie[p.ids.index(201)])
    assert len(result['native_stations']) == 8
    expected = native.operator.cell.recover(ref[42:60])
    for a, b in zip(result['native_stations'], expected):
        invariant(a['strain'], b['strain']); invariant(a['resultants'], b['resultants'])
    recovery = legacy.compute_stresses(model.mesh, ref[ld], material)
    for key in recovery:
        if isinstance(recovery[key], str): assert result['legacy_recovery'][key] == recovery[key]
        else: invariant(result['legacy_recovery'][key], recovery[key])
    assert p.native._mesh is None and p.native._validator is None
    assert result['qualification'] is False and result['policy'] == B3_POLICY


@pytest.mark.parametrize('case', ['midpoint', 'curved', 'eccentricity', 'generalized', 'b2_type', 'subclass', 'null_policy', 'b2_policy'])
def test_b3_specific_admission_precedes_both_operators(case):
    from anyfesolver.elements import BeamElement
    from anyfesolver._ge_beam3_g3b_reference import POLICY
    native, legacy, material, kw = parts()
    problem = B3TranslationReferenceProblem
    if case == 'midpoint': kw['nodes'][202][0] += 1e-9
    elif case == 'curved': kw['nodes'][202][1] += .01
    elif case == 'eccentricity': legacy.eccentricity[1] = .01
    elif case == 'generalized': legacy.generalized_section = object()
    elif case == 'b2_type': legacy = BeamElement(2, [201, 203], cross_section=legacy.cross_section)
    elif case == 'subclass':
        class Unregistered(B3TranslationReferenceProblem): pass
        problem = Unregistered
    elif case == 'null_policy': kw['policy'] = None
    else: kw['policy'] = POLICY
    with patch.object(ElasticOperator, 'evaluate', side_effect=AssertionError('native mechanics called')):
        with patch.object(QuadraticBeamElement, 'compute_stiffness_matrix', side_effect=AssertionError('B3 mechanics called')):
            with pytest.raises(ValueError): problem(native, legacy, material, **kw)
    assert native._mesh is None and native._validator is None


@pytest.mark.parametrize('case', ['midpoint', 'eccentricity', 'generalized', 'family', 'nullspace', 'constraint_rows'])
def test_b3_specific_capture_mutations(case):
    p = make(); f = load(p)
    if case == 'midpoint': p.model.mesh.nodes[202].x += .01
    elif case == 'eccentricity': p.legacy.eccentricity[0] = .01
    elif case == 'generalized': p.legacy.generalized_section = object()
    elif case == 'family':
        from anyfesolver._ge_beam3_g3b_reference import B2TranslationReferenceProblem
        p.__class__ = B2TranslationReferenceProblem
    elif case == 'nullspace': p.T = p.T*.5
    else: p.J = p.J*.5
    with pytest.raises(ValueError): p.solve(f)


def test_b3_nonzero_rhs_reuse_preserves_reference_capture():
    p = make(); f = load(p); baseline = p.solve(f)
    for factor in FIXTURE['programs']['rhs_factors']:
        result = p.solve(factor*f)
        invariant(result['u'], factor*baseline['u'])
        invariant(result['multipliers'], factor*baseline['multipliers'])
        invariant(result['energy'], factor**2*baseline['energy'])
    assert canonical(p.solve(f)) == canonical(baseline)
    assert p.native._mesh is None and p.native._validator is None


def test_translation_tie_has_no_rotational_work_or_equality():
    p = make(); f = np.zeros(36); a = 6*p.ids.index(103); b = 6*p.ids.index(201)
    f[a+3] = FIXTURE['programs']['mixed_moment'][0]
    r = p.solve(f)
    assert abs(r['u'][a+3]) > 1e-8 and abs(r['u'][b+3]) <= 1e-11
    assert not np.any(p.J[12:, np.array([6*i+j for i in range(6) for j in (3, 4, 5)])])
    delta = p.T @ np.linspace(-.2, .3, 21)
    invariant(delta @ r['tie_forces'], 0.)
    invariant(delta @ (p.K @ r['u']-f), 0.)


@pytest.mark.parametrize('case', ['policy', 'adapter', 'rotations', 'shared_node', 'rotational_tie', 'offset', 'weight', 'history', 'orphan', 'boolean'])
def test_unsupported_mixed_admission_precedes_operator_evaluation(case):
    native, legacy, material, kw = parts()
    if case == 'policy': kw['policy'] = 'FINITE'
    elif case == 'adapter': kw['adapter_allowlist'] = ('UNREGISTERED',)
    elif case == 'rotations': kw['rotation_targets'] = np.eye(3)
    elif case == 'shared_node': legacy.node_ids[0] = 103
    elif case == 'rotational_tie': kw['tie']['components'].append(3)
    elif case == 'offset': kw['tie']['offset'][0] = 1
    elif case == 'weight': kw['tie']['masters'][0][1] = '1/2'
    elif case == 'history': material.hardening_curve = object()
    elif case == 'orphan': kw['nodes'][999] = np.zeros(3)
    elif case == 'boolean': kw['tie']['slave'] = True
    with patch.object(ElasticOperator, 'evaluate', side_effect=AssertionError('mechanics called')):
        with pytest.raises(ValueError): B3TranslationReferenceProblem(native, legacy, material, **kw)
    assert native._mesh is None and native._validator is None


@pytest.mark.parametrize('mutation', ['coordinates', 'material', 'connectivity', 'section', 'operator', 'activity', 'constraints', 'loads'])
def test_frozen_reference_mutations_rejected(mutation):
    p = make(); f = load(p)
    if mutation == 'coordinates': p.model.mesh.nodes[101].x += .1
    elif mutation == 'material': p.material.elastic_modulus *= 2
    elif mutation == 'connectivity': p.legacy.node_ids.reverse()
    elif mutation == 'section': p.legacy._A *= 2
    elif mutation == 'operator': p.K = p.K+np.eye(36)
    elif mutation == 'activity': p.model.mesh.element_activity = {}
    elif mutation == 'constraints': p.model.constraint_equations.append('unregistered tie')
    else: p.model.load_cases.append(object())
    with pytest.raises(ValueError): p.solve(f)
    assert p.native._mesh is None and p.native._validator is None


@pytest.mark.parametrize('phase', [1, 2])
def test_cancel_and_independent_reuse(phase):
    p = make(); other = make(); f = load(p); before = p.solve(f); calls = []
    def cancel(): calls.append(True); return len(calls) == phase
    with pytest.raises(InterruptedError): p.solve(-.5*f, cancel=cancel)
    assert len(calls) == phase and canonical(p.solve(f)) == canonical(before)
    assert canonical(other.solve(f)) == canonical(before)
    invariant(p.solve(-.5*f)['u'], -.5*before['u'])
    with pytest.raises(ValueError): p.checkpoint()
    with patch.object(p, '_guard', side_effect=AssertionError('must reject before state work')):
        with pytest.raises(ValueError): p.solve(f, mode='FINITE')
        with pytest.raises(ValueError): p.solve(f, rotation_targets=np.eye(3))


def test_mb3_deterministic_development_packet(tmp_path):
    a, b = make(), make(); f = load(a)
    packet = canonical(dict(schema='G3B_M_B3_DEVELOPMENT_V1', qualification=False, result=a.solve(f)))
    assert packet == canonical(dict(schema='G3B_M_B3_DEVELOPMENT_V1', qualification=False, result=b.solve(f)))
    path = Path(os.environ.get('G3B_MB3_DIAGNOSTIC_PAYLOAD', str(tmp_path/'mb3.json')))
    with path.open('xb') as stream: stream.write(packet)


def test_legacy_zero_force_stub_is_never_used():
    p = make()
    with patch.object(p.legacy, 'compute_internal_forces', side_effect=AssertionError('legacy placeholder used')):
        result = p.solve(load(p))
    assert np.linalg.norm(result['support_reactions']) > 0


def test_successor_extent_preserves_mechanics_and_historical_records():
    import subprocess
    parent = '5da2a9c6efad7feb6a573196f3791b7872717bbe'
    git = ['git', '-c', 'safe.directory='+ROOT.as_posix()]
    allowed = {
        'src/anyfesolver/_ge_beam3_g3b_reference.py': 'M',
        'scripts/run_ge_beam3_g3b.py': 'M',
        'tests/test_ge_beam3_g3b_mb2.py': 'M',
        'tests/test_ge_beam3_g3b_mb3.py': 'A',
        'docs/GE_BEAM3_G3B_M_B3_DEVELOPMENT.md': 'A',
        'docs/reference_cases/ge_beam3_g3b_mb3_development_v1.json': 'A',
        'src/anyfesolver/_ge_beam3_g3b_q4_reference.py': 'A', 'tests/test_ge_beam3_g3b_mq4.py': 'A',
        'docs/GE_BEAM3_G3B_M_Q4_DEVELOPMENT.md': 'A',
        'docs/reference_cases/ge_beam3_g3b_mq4_development_v1.json': 'A',
        'src/anyfesolver/_ge_beam3_g3b_s3_reference.py': 'A', 'tests/test_ge_beam3_g3b_ms3.py': 'A',
        'docs/GE_BEAM3_G3B_M_S3_DEVELOPMENT.md': 'A',
        'docs/reference_cases/ge_beam3_g3b_ms3_development_v1.json': 'A',
        'src/anyfesolver/_ge_beam3_g3b_weighted_reference.py':'A','tests/test_ge_beam3_g3b_weighted.py':'A',
        'docs/GE_BEAM3_G3B_WEIGHTED_DEVELOPMENT.md':'A','docs/reference_cases/ge_beam3_g3b_weighted_development_v1.json':'A'}
    allowed.update({p:'A' for p in ["tests/test_ge_beam3_g3b_transport.py","docs/GE_BEAM3_G3B_TRANSPORT_DEVELOPMENT.md","docs/reference_cases/ge_beam3_g3b_transport_development_v1.json"]})
    allowed.update({p:'A' for p in ["src/anyfesolver/_ge_beam3_g3b_owner.py","tests/test_ge_beam3_g3b_owner.py","docs/GE_BEAM3_G3B_OWNER_CONTRACT.md","docs/reference_cases/ge_beam3_g3b_owner_development_v1.json"]})
    allowed.update({p:"A" for p in ["docs/GE_BEAM3_G3B_ENTRY_SAFETY_STATUS.md","docs/reference_cases/ge_beam3_g3b_confirmation_inventory_v2.json"]})
    allowed.update({p:"A" for p in ["scripts/confirm_ge_beam3_g3b.py","scripts/ge_beam3_g3b_environment.py","tests/test_ge_beam3_g3b_confirmation_runner.py","docs/GE_BEAM3_G3B_CONFIRMATION_RUNNER_STATUS.md"]})
    allowed.update({p:"A" for p in ["scripts/ge_beam3_g3b_confirmation_authority.py","tests/test_ge_beam3_g3b_confirmation_authority.py","docs/GE_BEAM3_G3B_FORMAL_CONFIRMATION_CONTRACT.md","docs/reference_cases/ge_beam3_g3b_confirmation_inventory_v1.json"]})
    allowed.update({p:"A" for p in ["docs/GE_BEAM3_G3B_LOCAL_REVIEW_STATUS.md","docs/reference_cases/ge_beam3_g3b_local_safety_review_v1.json","src/anyfesolver/_ge_beam3_g3b_result_schema.py","tests/test_ge_beam3_g3b_owner_corrections.py","docs/GE_BEAM3_G3B_OWNER_CORRECTIONS.md"]})
    diff = subprocess.check_output(git+['diff', '--name-status', parent, '--'], cwd=ROOT, timeout=20).decode()
    for line in diff.splitlines():
        status, path = line.split('\t'); assert allowed.get(path) == status
    extra = subprocess.check_output(git+['ls-files', '--others', '--exclude-standard'], cwd=ROOT, timeout=20).decode()
    assert set(extra.splitlines()) <= {p for p, s in allowed.items() if s == 'A'}
    # The preceding M_B2 mechanics regression is unchanged; only its declared
    # additive boundary admits this successor's three research/test paths.
    old = subprocess.check_output(git+['show', parent+':tests/test_ge_beam3_g3b_mb2.py'], cwd=ROOT, timeout=20)
    current = (ROOT/'tests/test_ge_beam3_g3b_mb2.py').read_bytes().replace(b'\r\n', b'\n')
    marker = b'def test_additive_scope_and_immutable_parent_authority():'
    assert old.split(marker)[0] == current.split(marker)[0]
