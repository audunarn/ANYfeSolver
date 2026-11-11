"""Single small continuum rehearsal against immutable, already solved native states."""
import ast
import copy
from pathlib import Path
import pytest
from docs.reference_cases import ge_beam3_line_modal_comparison as comparison
from docs.reference_cases import ge_beam3_line_modal_probe as probe


@pytest.fixture(scope='module')
def made(tmp_path_factory):
    root=tmp_path_factory.mktemp('comparison'); events=[]
    def save(name,raw):
        with (root/name).open('xb') as stream: stream.write(raw)
    rows=comparison.build(save,events.append)
    packets={name:comparison.parse((root/name).read_bytes()) for name in comparison.ARTIFACTS}
    return root,rows,packets,events


def test_complete_saved_state_comparison_and_deterministic_inspection(made):
    root,rows,packets,events=made
    a=probe.summary(root,'0'*40); b=probe.summary(root,'0'*40)
    assert comparison.canonical(a)==comparison.canonical(b)
    assert a['rows']==rows and [r['macros'] for r in rows]==[1,2,4]
    assert len(a['artifacts'])==12
    assert a['new_native_spectra']==a['new_native_nonlinear_solves']==0
    assert a['new_reference_equilibria']==1 and a['new_reference_spectra']==2
    assert not a['production_qualified'] and not a['buckling_factor_authorized']
    assert not a['clustered_mac_qualification'] and a['independent_review']=='PENDING'
    assert [e['stage'] for e in events].count('CONTINUUM_EQUILIBRIUM')==1
    for row in rows:
        assert row['native_rule_kinetic_identity_error']<1e-11
        assert row['reference_resolution_error']<1e-7
    with (root/'rehearsal-summary.json').open('xb') as stream: stream.write(comparison.canonical(a))


@pytest.mark.parametrize('mutation',['inventory','input','equilibrium','profile','matrix','eigenvalue','mode','overlap'])
def test_mutated_comparison_rejected(made,mutation):
    packets=copy.deepcopy(made[2])
    if mutation=='inventory': packets['extra.json']={}
    if mutation=='input': packets['checkpoint-1.json']['extra']=0
    if mutation=='equilibrium': packets['equilibrium.json']['force'][0]+=.01
    if mutation=='profile': packets['reference-18.json']['quadrature']=64
    if mutation=='matrix': packets['reference-14.json']['stiffness'][0][0]+=.01
    if mutation=='eigenvalue': packets['reference-14.json']['eigenvalues'][0]+=.1
    if mutation=='mode': packets['reference-14.json']['modes'][0][0]+=.01
    if mutation=='overlap': packets['overlap-1.json']['4']['overlap'][0][0]+=.01
    with pytest.raises(ValueError): comparison.recompute(packets)


@pytest.mark.parametrize('raw',[b'{"a":1,"a":1}\n',b'{"a":NaN}\n',b'{ "a":1}\n'])
def test_noncanonical_evidence_rejected(raw):
    with pytest.raises(ValueError): comparison.parse(raw)


def test_bounded_registered_overlap_profiles_and_no_native_imports():
    for m in (1,2,4):
        assert len(comparison.overlap_sites(m,32))==64*m
    with pytest.raises(ValueError): comparison.overlap_sites(8,32)
    with pytest.raises(ValueError): comparison.overlap_sites(1,True)
    tree=ast.parse(Path(comparison.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom): assert not (node.module or '').startswith('anyfesolver')
        if isinstance(node,ast.Import): assert not any(n.name.startswith('anyfesolver') for n in node.names)
    calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
    assert not any('native.solve' in x or 'model(' in x or 'solve_modes' in x for x in calls)


def test_probe_checks_authority_before_output_creation(tmp_path,monkeypatch):
    output=tmp_path/'not-created'
    def reject(_): raise ValueError('frozen clean source required')
    monkeypatch.setattr(probe,'guard',reject)
    with pytest.raises(ValueError,match='frozen'): probe.run('0'*40,output)
    assert not output.exists()
