"""Frozen M_S3 development fixture; independent assembly, not formal acceptance."""
from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.fe_core import FEModel, Material
from anyfesolver._ge_beam3_g1_element import ElasticElement
from anyfesolver._ge_beam3_g1_elastic import ElasticSection, canonical
from anyfesolver._ge_beam3_g1_operator import ElasticOperator
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from anyfesolver._ge_beam3_g3b_s3_reference import S3TranslationReferenceProblem, POLICY

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT/'docs/reference_cases/ge_beam3_g3_graph_fixtures_v1.json').read_text())
MS3 = next(g for g in FIXTURE['mixed_graphs'] if g['id'] == 'M_S3')


def invariant(a, b):
    a, b = np.asarray(a), np.asarray(b)
    error = np.linalg.norm(a-b)/max(1., np.linalg.norm(a), np.linalg.norm(b))
    assert np.isfinite(error) and error <= 1e-11


def compare_recovery(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(b[k], (str, bool)) or (isinstance(b[k], (tuple, list)) and all(isinstance(v,str) for v in b[k])):
            assert a[k] == b[k]
        else: invariant(a[k], b[k])


def parts():
    nodes = {n: np.array(x, dtype=float) for n, x in MS3['nodes']}
    d = MS3['native_element']; ids = tuple(d['nodes'])
    native = ElasticElement(d['id'], ids, Reference([nodes[n] for n in ids], np.tile(np.eye(3), (3, 1, 1))),
                            ElasticSection.isotropic(**FIXTURE['materials']['native_isotropic']))
    m = FIXTURE['materials']['shell']
    shell = NativeParityE4PLS3V2DShellElement(MS3['other_element']['id'], list(MS3['other_element']['nodes']),
                                      thickness=m['thickness'], reference_normal=m['reference_normal'])
    material = Material('G3b frozen shell', m['E'], m['nu'])
    return native, shell, material, dict(nodes=nodes, fixed_nodes=tuple(MS3['fixed_nodes']),
        tie=deepcopy(MS3['tie']), reference_normal=deepcopy(m['reference_normal']))



def make():
    native, shell, material, kw = parts()
    return S3TranslationReferenceProblem(native, shell, material, **kw)


def load(p, *, moment=True):
    f = np.zeros(36); i = p.ids.index(MS3['load_node'])*6
    f[i:i+3] = FIXTURE['programs']['mixed_force']
    if moment: f[i+3:i+6] = FIXTURE['programs']['mixed_moment']
    return f


def independent_full_reference():
    # Fresh elements, independent global assembly and explicit multiplier rows.
    # The unchanged family operators are permitted by the frozen G3 contract.
    native, shell, material, kw = parts(); ids = sorted(kw['nodes'])
    model = FEModel('independent reference')
    for n in ids: model.add_node(n, *kw['nodes'][n])
    model.add_element(native.element_id, native); model.add_element(shell.element_id, shell)
    x = native.operator.reference.coordinates
    h = native.operator.evaluate(x, np.zeros((3, 3)), native.operator.reference.nodal_triads,
                                 np.tile(np.eye(3), (2, 1, 1)), np.zeros(18))['hessian']
    H = np.zeros((60, 60))
    native_dofs = [6*ids.index(n)+j for n in native.node_ids for j in range(6)]
    native_all = native_dofs+list(range(36, 60))
    for i, gi in enumerate(native_all):
        for j, gj in enumerate(native_all): H[gi, gj] += h[i, j]
    shell_dofs = [6*ids.index(n)+j for n in shell.node_ids for j in range(6)]
    b = shell.compute_stiffness_matrix(model.mesh, material)
    for i, gi in enumerate(shell_dofs):
        for j, gj in enumerate(shell_dofs): H[gi, gj] += b[i, j]
    J = np.zeros((21, 60)); row = 0
    for n in sorted(kw['fixed_nodes']):
        for d in range(6): J[row, 6*ids.index(n)+d] = 1; row += 1
    for d in range(3):
        J[row+d, 6*ids.index(103)+d] = 1
        J[row+d, 6*ids.index(201)+d] = -1
    return H, J, native, shell, material, model, native_dofs, shell_dofs


def test_ms3_reference_kkt_smoke():
    p = make(); f = load(p); result = p.solve(f)
    H, J, native, shell, material, model, nd, ld = independent_full_reference()
    system = np.block([[H, J.T], [J, np.zeros((21, 21))]])
    ref = np.linalg.solve(system, np.r_[f, np.zeros(24+21)])
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
    recovery = shell.compute_stresses(model.mesh, ref[ld], material, return_global=True)
    compare_recovery(result['shell_recovery'], recovery)
    assert result['shell_recovery']['numerical_fields_excluded'] is True
    assert result['shell_recovery']['formulation_id'] == p.shell.formulation_id
    invariant(result['shell_recovery']['frame'][:,2], [0, 0, 1])
    assert p.T.shape == (36, 15) and system.shape == (81, 81)
    assert p.native._mesh is None and p.native._validator is None
    assert result['qualification'] is False and result['policy'] == POLICY


def test_translation_tie_has_no_rotational_work_or_equality():
    p = make(); f = np.zeros(36); a = 6*p.ids.index(103); b = 6*p.ids.index(201)
    f[a+3] = FIXTURE['programs']['mixed_moment'][0]
    r = p.solve(f)
    assert abs(r['u'][a+3]) > 1e-8 and abs(r['u'][b+3]) <= 1e-11
    assert not np.any(p.J[18:, np.array([6*i+j for i in range(6) for j in (3, 4, 5)])])
    delta = p.T @ np.linspace(-.2, .3, 15)
    invariant(delta @ r['tie_forces'], 0.)
    invariant(delta @ (p.K @ r['u']-f), 0.)


@pytest.mark.parametrize('case', ['policy', 'adapter', 'rotations', 'shared_node', 'rotational_tie',
    'offset', 'weight', 'history', 'orphan', 'boolean', 'wrong_normal', 'missing_normal',
    'owner_mismatch', 'section', 'geometry', 'subclass', 'wrong_class', 'thickness', 'cached_shell'])
def test_rejection_precedes_both_element_operators(case):
    native, shell, material, kw = parts(); cls = S3TranslationReferenceProblem
    if case == 'policy': kw['policy'] = 'FINITE'
    elif case == 'adapter': kw['adapter_allowlist'] = ('UNREGISTERED',)
    elif case == 'rotations': kw['rotation_targets'] = np.eye(3)
    elif case == 'shared_node': shell.node_ids = (103, 202, 203)
    elif case == 'rotational_tie': kw['tie']['components'].append(3)
    elif case == 'offset': kw['tie']['offset'][0] = 1
    elif case == 'weight': kw['tie']['masters'][0][1] = '1/2'
    elif case == 'history': material.hardening_curve = object()
    elif case == 'orphan': kw['nodes'][999] = np.zeros(3)
    elif case == 'boolean': kw['tie']['slave'] = True
    elif case == 'wrong_normal': shell.reference_normal = [0, 0, -1]; kw['reference_normal'] = [0, 0, -1]
    elif case == 'missing_normal': shell.reference_normal = None
    elif case == 'owner_mismatch': kw['reference_normal'] = [0, 1, 0]
    elif case == 'section': object.__setattr__(shell, 'shell_section', object())
    elif case == 'geometry': kw['nodes'][203][2] = .01
    elif case == 'subclass':
        class Unregistered(S3TranslationReferenceProblem): pass
        cls = Unregistered
    elif case == 'wrong_class':
        from anyfesolver.elements import ShellElement
        shell = ShellElement(2, [201,202,203])
    elif case == 'thickness': shell.thickness = .3
    else: object.__setattr__(shell, '_stiffness_matrix', np.eye(18))
    with patch.object(ElasticOperator, 'evaluate', side_effect=AssertionError('native mechanics called')):
        with patch.object(NativeParityE4PLS3V2DShellElement, 'compute_stiffness_matrix', side_effect=AssertionError('shell mechanics called')):
            with pytest.raises(ValueError): cls(native, shell, material, **kw)
    assert native._mesh is None and native._validator is None


@pytest.mark.parametrize('mutation', ['coordinates', 'material', 'connectivity', 'thickness', 'normal',
    'operator', 'activity', 'constraints', 'loads', 'nullspace', 'constraint_rows', 'state_token',
    'foreign_subscription', 'epoch', 'nonlinear_cache'])
def test_frozen_capture_mutations_rejected(mutation):
    p = make(); f = load(p)
    if mutation == 'coordinates': p.model.mesh.nodes[201].x += .1
    elif mutation == 'material': p.material.elastic_modulus *= 2
    elif mutation == 'connectivity': p.shell.node_ids = tuple(reversed(p.shell.node_ids))
    elif mutation == 'thickness': p.shell.thickness *= 2
    elif mutation == 'normal': p.shell.reference_normal = [0, 0, -1]
    elif mutation == 'operator': p.K = p.K+np.eye(36)
    elif mutation == 'activity': p.model.mesh.element_activity = {}
    elif mutation == 'constraints': p.model.constraint_equations.append('unregistered tie')
    elif mutation == 'loads': p.model.load_cases.append(object())
    elif mutation == 'nullspace': p.T = p.T*.5
    elif mutation == 'constraint_rows': p.J = p.J*.5
    elif mutation == 'state_token': object.__setattr__(p.shell, '_qualified_direct_state_token', object())
    elif mutation == 'foreign_subscription': p.shell._qualified_direct_state_tokens.append([0])
    elif mutation == 'epoch': p.model.mesh._qualified_direct_state_token[0] += 1
    else: object.__setattr__(p.shell, '_nl_cache', {})
    with pytest.raises(ValueError): p.solve(f)
    assert p.native._mesh is None and p.native._validator is None


@pytest.mark.parametrize('phase', [1, 2])
def test_cancel_reuse_and_fail_closed_finite_program(phase):
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


def test_s3_numerical_drill_fields_excluded_from_physical_recovery():
    p = make(); result = p.solve(load(p)); u = result['u'][list(p._dofs[1])].copy()
    u[[5,11,17]] += np.array([.1,-.2,.3])
    recovered = p.shell.compute_stresses(p.model.mesh, u, p.material, return_global=True)
    compare_recovery(recovered, result['shell_recovery'])
    assert recovered['numerical_fields_excluded'] is True


def test_ms3_deterministic_development_packet(tmp_path):
    a, b = make(), make(); f = load(a)
    packet = canonical(dict(schema='G3B_M_S3_DEVELOPMENT_V1', qualification=False, result=a.solve(f)))
    assert packet == canonical(dict(schema='G3B_M_S3_DEVELOPMENT_V1', qualification=False, result=b.solve(f)))
    path = Path(os.environ.get('G3B_MS3_DIAGNOSTIC_PAYLOAD', str(tmp_path/'ms3.json')))
    with path.open('xb') as stream: stream.write(packet)


def test_shuffle_insertion_and_rhs_reuse():
    p = make(); n, s, m, kw = parts(); kw['nodes'] = dict(reversed(list(kw['nodes'].items())))
    kw['fixed_nodes'] = tuple(reversed(kw['fixed_nodes']))
    other = S3TranslationReferenceProblem(n,s,m,**kw)
    f = load(p); baseline = p.solve(f)
    assert canonical(other.solve(f)) == canonical(baseline)
    for factor in FIXTURE['programs']['rhs_factors']:
        r = p.solve(factor*f)
        invariant(r['u'],factor*baseline['u'])
        invariant(r['multipliers'],factor*baseline['multipliers'])
        invariant(r['energy'],factor**2*baseline['energy'])
    assert canonical(p.solve(f)) == canonical(baseline)


@pytest.mark.parametrize('family', ['v2c', 'mitc3plus', 'q4'])
def test_older_candidates_and_q4_are_not_s3_v2d(family):
    from anyfesolver.e4_pl_s3_v2c_element import StrictFlatLinearE4PLS3V2CShellElement
    from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    native, shell, material, kw = parts()
    cls = {'v2c': StrictFlatLinearE4PLS3V2CShellElement, 'mitc3plus': QualifiedE4PLS3ShellElement,
           'q4': QualifiedE4PLShellElement}[family]
    # Type rejection precedes any access to a foreign instance's internals.
    foreign = object.__new__(cls)
    with patch.object(ElasticOperator, 'evaluate', side_effect=AssertionError('native mechanics called')):
        with pytest.raises(ValueError): S3TranslationReferenceProblem(native, foreign, material, **kw)


def test_ms3_extent_and_frozen_parent_bindings():
    import hashlib
    import subprocess
    parent = '10a16c522e338088cba2d4f40af19169f149c8c9'
    git = ['git','-c','safe.directory='+ROOT.as_posix()]
    allowed = {
        'src/anyfesolver/_ge_beam3_g3b_s3_reference.py':'A', 'tests/test_ge_beam3_g3b_ms3.py':'A',
        'docs/GE_BEAM3_G3B_M_S3_DEVELOPMENT.md':'A',
        'docs/reference_cases/ge_beam3_g3b_ms3_development_v1.json':'A',
        'scripts/run_ge_beam3_g3b.py':'M', 'tests/test_ge_beam3_g3b_mb2.py':'M',
        'tests/test_ge_beam3_g3b_mb3.py':'M', 'tests/test_ge_beam3_g3b_mq4.py':'M',
        'src/anyfesolver/_ge_beam3_g3b_weighted_reference.py':'A','tests/test_ge_beam3_g3b_weighted.py':'A',
        'docs/GE_BEAM3_G3B_WEIGHTED_DEVELOPMENT.md':'A','docs/reference_cases/ge_beam3_g3b_weighted_development_v1.json':'A'}
    allowed.update({p:'A' for p in ["tests/test_ge_beam3_g3b_transport.py","docs/GE_BEAM3_G3B_TRANSPORT_DEVELOPMENT.md","docs/reference_cases/ge_beam3_g3b_transport_development_v1.json"]})
    allowed.update({p:'A' for p in ["src/anyfesolver/_ge_beam3_g3b_owner.py","tests/test_ge_beam3_g3b_owner.py","docs/GE_BEAM3_G3B_OWNER_CONTRACT.md","docs/reference_cases/ge_beam3_g3b_owner_development_v1.json"]})
    allowed.update({p:"A" for p in ["docs/GE_BEAM3_G3B_ENTRY_SAFETY_STATUS.md","docs/reference_cases/ge_beam3_g3b_confirmation_inventory_v2.json"]})
    allowed.update({p:"A" for p in ["scripts/confirm_ge_beam3_g3b.py","scripts/ge_beam3_g3b_environment.py","tests/test_ge_beam3_g3b_confirmation_runner.py","docs/GE_BEAM3_G3B_CONFIRMATION_RUNNER_STATUS.md"]})
    allowed.update({p:"A" for p in ["scripts/ge_beam3_g3b_confirmation_authority.py","tests/test_ge_beam3_g3b_confirmation_authority.py","docs/GE_BEAM3_G3B_FORMAL_CONFIRMATION_CONTRACT.md","docs/reference_cases/ge_beam3_g3b_confirmation_inventory_v1.json"]})
    allowed.update({p:"A" for p in ["docs/GE_BEAM3_G3B_LOCAL_REVIEW_STATUS.md","docs/reference_cases/ge_beam3_g3b_local_safety_review_v1.json","src/anyfesolver/_ge_beam3_g3b_result_schema.py","tests/test_ge_beam3_g3b_owner_corrections.py","docs/GE_BEAM3_G3B_OWNER_CORRECTIONS.md"]})
    diff = subprocess.check_output(git+['diff','--name-status',parent,'--'],cwd=ROOT,timeout=20).decode()
    for line in diff.splitlines():
        status,path = line.split('\t'); assert allowed.get(path) == status
    extra = subprocess.check_output(git+['ls-files','--others','--exclude-standard'],cwd=ROOT,timeout=20).decode()
    assert set(extra.splitlines()) <= {p for p,s in allowed.items() if s == 'A'}
    for path, marker in [('tests/test_ge_beam3_g3b_mb2.py',b'def test_additive_scope_and_immutable_parent_authority():'),
                         ('tests/test_ge_beam3_g3b_mb3.py',b'def test_successor_extent_preserves_mechanics_and_historical_records():'),
                         ('tests/test_ge_beam3_g3b_mq4.py',b'def test_mq4_extent_and_frozen_parent_bindings():')]:
        old = subprocess.check_output(git+['show',parent+':'+path],cwd=ROOT,timeout=20)
        assert old.split(marker)[0] == (ROOT/path).read_bytes().replace(b'\r\n',b'\n').split(marker)[0]
    contract = json.loads((ROOT/'docs/reference_cases/ge_beam3_g3_graph_contract_v1.json').read_text())
    for binding in contract['source_bindings']+contract['text_bindings']:
        raw=(ROOT/binding['path']).read_bytes().replace(b'\r\n',b'\n')
        assert len(raw) == binding['bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256']
