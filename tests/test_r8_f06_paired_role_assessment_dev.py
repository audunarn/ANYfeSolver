"""Focused controls and one installed assessment; execution requires authority."""
from pathlib import Path
import json
import os
import pytest


def _saved_preparation_bytes_present():
    from scripts import r8_f06_paired_role_assessment_dev as a
    return all((a.PREP/name).exists() for name in
        ('candidate-direction.json','candidate.json','ROOT_DIRECTION_DECISION.json'))


def _skip_without_saved_preparation_bytes():
    if not _saved_preparation_bytes_present():
        pytest.skip('local prepared candidate/preparation/decision bytes absent; '
            'saved-schema dependent controls require the saved preparation')


def _source_binding_seam(monkeypatch):
    """Explicit synthetic unit seam isolating ONLY the saved source binding.

    The saved preparation binds the private finite old source and the current
    source drifted, so the unpatched binding loop must refuse (asserted
    separately in test_source_unpatched_stale_preparation_refuses). The seam
    satisfies exactly the saved source-binding digests inside load_candidate();
    the runner, candidate, preparation, decision and frozen-direction archive
    checks still run unpatched on the real saved bytes. Helper/schema
    semantics only: never physical admission, numerical execution or
    authority renewal.
    """
    from scripts import r8_f06_paired_role_assessment_dev as a
    from scripts import r8_f06_paired_active_tangent_dev as original
    from scripts import r8_f06_paired_branch_domain_dev as domain
    real_sha=a.sha
    prepared=json.loads((a.PREP/'candidate.json').read_text())
    doubly_verified={str(Path(p).resolve()) for p in (original.__file__,
        a.PREP/'candidate-direction.json',a.PREP/'candidate.json',
        a.PREP/'ROOT_DIRECTION_DECISION.json',domain.SAVED/'direction-and-row-map.npz')}
    bindings={str((a.ROOT/rel).resolve()):digest for rel,digest in
        prepared['bindings'].items() if str((a.ROOT/rel).resolve()) not in doubly_verified}
    def seam_sha(path):
        key=str(Path(path).resolve())
        return bindings[key] if key in bindings else real_sha(path)
    monkeypatch.setattr(a,'sha',seam_sha)


def test_source_unpatched_stale_preparation_refuses():
    """Actual unpatched stale-preparation refusal; never forced valid."""
    from scripts import r8_f06_paired_role_assessment_dev as a
    if _saved_preparation_bytes_present():
        with pytest.raises(ValueError,match='changed preparation dependency'):
            a.load_candidate()
    else:
        with pytest.raises((ValueError,OSError)):
            a.load_candidate()


def test_source_adopted_scale_and_complete_nonzeros(monkeypatch):
    _skip_without_saved_preparation_bytes()
    from scripts import r8_f06_paired_role_assessment_dev as a
    _source_binding_seam(monkeypatch)
    candidate,prepared=a.load_candidate()
    original=a.domain.read_vector(a.domain.SAVED/'direction-and-row-map.npz','direction')
    scaled=a.scaled_from_original(original,candidate)
    assert scaled==candidate['direction'] and len(scaled)==132
    assert sum(x!=0 for x in scaled)==120 and prepared['nonzero_after']==120


def test_source_changed_scale_channel_and_metadata_refused(monkeypatch):
    _skip_without_saved_preparation_bytes()
    from copy import deepcopy
    from scripts import r8_f06_paired_role_assessment_dev as a
    _source_binding_seam(monkeypatch)
    candidate,_=a.load_candidate();vector=a.domain.read_vector(a.domain.SAVED/'direction-and-row-map.npz','direction')
    for field,value in [('scale_by_role',{**a.SCALES,'physical':a.SCALE}),('roles',{**a.ROLES,'native_retained':[]}),('h',[1e-4,1e-5,2e-6]),('direction',[0.]*132)]:
        changed=deepcopy(candidate);changed[field]=value
        with pytest.raises(ValueError):a.scaled_from_original(vector,changed)


def test_source_changed_candidate_bytes_refused(tmp_path):
    _skip_without_saved_preparation_bytes()
    from scripts import r8_f06_paired_role_assessment_dev as a
    path=tmp_path/'candidate-direction.json';path.write_bytes((a.PREP/'candidate-direction.json').read_bytes()+b' ')
    with pytest.raises(ValueError,match='changed candidate'):a.load_candidate(path)


def test_source_missing_branch_and_unexercised_work_refused():
    from scripts import r8_f06_paired_role_assessment_dev as a
    good=dict(raw_action_norm=1.,work_action_norm=2.,exercised=True)
    report=dict(base_smooth=True,smooth_assessment_resolved=True,
        evaluations=[dict(h=h,branches_stable=True,groups={g:dict(good) for g in
            ('shell','beam','constraints','fibre_retained','generalized_retained')}) for h in a.original.STEPS],
        local_adjacent_plateau={'0':True},local_derivatives=[dict(metrics=dict(good)) for _ in range(3)])
    assert all(a.complete_metrics(report).values())
    del report['evaluations'][1]['branches_stable']
    assert not a.complete_metrics(report)['branches']
    report['evaluations'][1]['branches_stable']=True
    report['evaluations'][0]['groups']['constraints']['raw_action_norm']=0.
    assert not a.complete_metrics(report)['positive_groups']
    report['local_derivatives'][0]['metrics']['work_action_norm']=0.
    assert not a.complete_metrics(report)['positive_locals']


def test_source_override_refused_outside_installed_child(tmp_path,monkeypatch):
    _skip_without_saved_preparation_bytes()
    from scripts import r8_f06_paired_role_assessment_dev as a
    _source_binding_seam(monkeypatch)
    monkeypatch.delenv('F06_BRANCH_STAGE',raising=False)
    saved=a.original.frozen_direction
    with pytest.raises(ValueError,match='installed-child-only'):a.assess(tmp_path)
    assert a.original.frozen_direction is saved
    # Exercise the actual wrapper caller with a persisted-report writer which
    # deliberately returns None, and with a failure after partial persistence.
    good=dict(raw_action_norm=1.,work_action_norm=2.,exercised=True)
    report=dict(status='SCOPED_PASS',base_smooth=True,smooth_assessment_resolved=True,
        evaluations=[dict(h=h,branches_stable=True,groups={g:dict(good) for g in
            ('shell','beam','constraints','fibre_retained','generalized_retained')}) for h in a.original.STEPS],
        local_adjacent_plateau={'0':True},local_derivatives=[dict(metrics=dict(good)) for _ in range(3)])
    vector=a.domain.read_vector(a.domain.SAVED/'direction-and-row-map.npz','direction')
    candidate,_=a.load_candidate()
    for name in ('atomic','chart_atomic','refusal_atomic','cancellation_atomic','terminal_atomic','chart_terminal_atomic'):
        report[name]={key:True for key in a.ATOMIC_KEYS}
    def fake_direction(owner,residual):return vector
    monkeypatch.setattr(a.original,'frozen_direction',fake_direction)
    monkeypatch.setenv('F06_BRANCH_STAGE','installed-assessment')
    def writer(out):
        assert a.original.frozen_direction is not fake_direction
        assert a.original.frozen_direction(None,None).tolist()==candidate['direction']
        a.original.write(out/'assessment.json',report)
    monkeypatch.setattr(a.original,'assess',writer)
    result=a.assess(tmp_path)
    assert result['status']=='SCOPED_PASS' and all(result['complete_observations'].values())
    persisted=json.loads((tmp_path/'assessment.json').read_text())
    assert persisted['branch_direction_provenance']['scale_by_role']==a.SCALES
    assert all(persisted['complete_atomic_observations'].values())
    assert a.original.frozen_direction is fake_direction
    def refusal(out):
        a.original.write(out/'assessment.json',{**report,'status':'UNRESOLVED_BRANCH','smooth_assessment_resolved':False})
        raise ValueError('actual saved branch refusal')
    monkeypatch.setattr(a.original,'assess',refusal)
    with pytest.raises(ValueError,match='actual saved branch refusal'):a.assess(tmp_path)
    persisted=json.loads((tmp_path/'assessment.json').read_text())
    assert persisted['status']=='UNRESOLVED_BRANCH' and not persisted['complete_observations']['branches']
    assert persisted['branch_direction_provenance']['candidate_sha256']==a.CANDIDATE_SHA
    assert a.original.frozen_direction is fake_direction
    report['atomic']['checkpoint']=False
    monkeypatch.setattr(a.original,'assess',writer)
    with pytest.raises(AssertionError,match='126 atomic'):a.assess(tmp_path)
    persisted=json.loads((tmp_path/'assessment.json').read_text())
    assert persisted['status']=='UNRESOLVED_ATOMIC' and not persisted['complete_atomic_observations']['atomic']
    assert a.original.frozen_direction is fake_direction


def test_source_exact_artifact_delta_adoption_refuses_drift():
    from scripts import r8_f06_paired_role_assessment_dev as a
    row=dict(path=a.B2_DELTA_PATH,old_sha256='a'*64,new_sha256='b'*64)
    decision=dict(status='ARTIFACT_REUSE_ADOPTED',manifest_sha256=a.MANIFEST_SHA,
        wheel_sha256=a.original.WHEEL_SHA,current_source_deltas=[row],independence_justification='frozen installed artifact remains exact; unrelated current B2 diagnostic delta')
    assert a.check_artifact_delta([row],decision)
    with pytest.raises(ValueError,match='drift'):a.check_artifact_delta([{**row,'new_sha256':'c'*64}],decision)
    with pytest.raises(ValueError,match='adoption'):a.check_artifact_delta([row],{**decision,'status':'PROPOSED'})
    altered={**row,'path':'src/anyfesolver/_ge_beam3_fibre_cell.py'}
    with pytest.raises(ValueError,match='B2'):a.check_artifact_delta([altered],{**decision,'current_source_deltas':[altered]})


def test_source_caller_selects_exact_inventory_including_installed_name(tmp_path):
    from scripts import r8_f06_paired_role_assessment_dev as a
    captured=[]
    def main(args):captured.append(args);return 0
    assert a.run_tests(tmp_path,'source-controls',main)==0
    targets=[x for x in captured[-1] if '::' in x]
    assert targets==list(a.source_nodeids()) and len(targets)==17 and len(set(targets))==17
    assert str(a.TEST)+'::test_source_override_refused_outside_installed_child' in targets
    assert not any(x.split('::')[-1]=='test_installed_branch_domain_tangent' for x in targets)
    assert '-k' not in captured[-1]
    assert a.run_tests(tmp_path,'installed-assessment',main)==0
    assert [x for x in captured[-1] if '::' in x]==[str(a.TEST)+'::test_installed_branch_domain_tangent']
    with pytest.raises(ValueError,match='unknown'):a.run_tests(tmp_path,'other',main)


def test_installed_branch_domain_tangent():
    # Actual installed runtime: runs only under the supervisor's existing
    # output binding; it skips only when that binding is absent.
    output=os.environ.get('F06_BRANCH_ASSESSMENT_OUTPUT')
    if output is None:
        pytest.skip('installed assessment output binding absent; supervisor-only runtime')
    from scripts.r8_f06_paired_role_assessment_dev import assess
    result=assess(Path(output))
    assert result['status']=='SCOPED_PASS' and all(result['complete_observations'].values()) and all(result['complete_atomic_observations'].values())
