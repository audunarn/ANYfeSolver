"""Explicit budget policy preserves default identity and certified support."""
import numpy as np
import pytest
from anyfesolver._ge_beam3_p5_seeded.core import sha
import anyfesolver.general_contact.native_curved_s3_work as module
from test_general_contact_native_curved_s3_work import _case


def test_budget_policy_identity_default_and_mutation_guard():
    work,p,low,U,u= _case()
    explicit=module.NativeCurvedMaterialS3Work(work.contact,work.shell,
        integration_budget_policy='full_reference')
    historical=sha(dict(profile=module.PROFILE,
        base=super(module.NativeCurvedMaterialS3Work,work)._fingerprint(),
        numerical='CURVED_SHARED_GAUSS8_16_V7_COMPLETE_MOVING_NORMAL_WORK_V2',
        path='affine_skin_left_exp_complete_bernstein_controls_v1'))
    assert work.identity==explicit.identity==historical
    active=module.NativeCurvedMaterialS3Work(work.contact,work.shell,
        integration_budget_policy='certified_active')
    assert active.identity!=historical
    active.integration_budget_policy='full_reference'
    with pytest.raises(ValueError):active._configuration(p,low,U,u)
    with pytest.raises(ValueError):
        module.NativeCurvedMaterialS3Work(work.contact,work.shell,integration_budget_policy='sampled')


def test_budget_area_keeps_uncertainty_and_requires_certified_partition(monkeypatch):
    work,p,low,U,u=_case()
    work=module.NativeCurvedMaterialS3Work(work.contact,work.shell,
        integration_budget_policy='certified_active')
    engine=module._S3CandidateIntegration(work,work.shell.current_positions(u))
    calls=[]
    def field(*args):calls.append(args[3]);return args[3]
    monkeypatch.setattr(module,'_station_field',field)
    monkeypatch.setattr(module,'_critical_gaps',lambda field,*_: [1. if field<.5 else -1.])
    prepared=[(1,[(0.,.4,False,()),(.4,.6,True,()),(.6,1.,False,())])]
    assert engine._integration_budget_area(prepared,False,p,low,U,0.,2.,2.)==2.
    assert calls==[]
    assert engine._integration_budget_area(prepared,True,p,low,U,0.,2.,2.)==pytest.approx(1.2)
    assert calls==[.2,.8] # uncertain support is never sampled away
    monkeypatch.setattr(module,'_critical_gaps',lambda *_: [1.])
    assert engine._integration_budget_area([(1,[(0.,1.,False,())])],
        True,p,low,U,0.,2.,2.)==2. # entirely certified-open positive fallback
    engine.integration_budget_policy='full_reference'
    assert engine._integration_budget_area(prepared,True,p,low,U,0.,2.,2.)==2.


def test_plane_default_budget_unchanged():
    assert module.NativeCurvedMaterialPlaneWork._integration_budget_area(
        None,[],True,None,None,None,0.,1.,7.)==7.


def test_active_budget_complete_work_independent_oracle_and_mixed_tangent(monkeypatch):
    import test_general_contact_native_curved_s3_work as checks
    work,p,low,U,u=_case()
    active=module.NativeCurvedMaterialS3Work(work.contact,work.shell,
        integration_budget_policy='certified_active')
    baseline=work.evaluate(p,low,U,u);candidate=active.evaluate(p,low,U,u)
    # This smooth fully active fixture must preserve the original subdivisions
    # and complete both-body operators exactly, not merely integrated forces.
    np.testing.assert_array_equal(candidate.residual,baseline.residual)
    np.testing.assert_array_equal(candidate.tangent.toarray(),baseline.tangent.toarray())
    assert candidate.contact.potential_j==baseline.contact.potential_j
    assert candidate.contact.integration_intervals==baseline.contact.integration_intervals
    assert candidate.contact.integration_evaluations==baseline.contact.integration_evaluations
    monkeypatch.setattr(checks,'_case',lambda:(active,p,low,U,u))
    checks.test_curved_moving_s3_independent_work_wrenches_and_full_mixed_tangent()


def test_active_budget_partial_front_independent_work_and_tangent(monkeypatch):
    import test_general_contact_native_curved_s3_work as checks
    work,p,low,U,u=_case()
    active=module.NativeCurvedMaterialS3Work(work.contact,work.shell,
        integration_budget_policy='certified_active')
    monkeypatch.setattr(checks,'_case',lambda:(active,p,low,U,u))
    checks.test_curved_s3_axial_front_complete_target_work_and_independent_partition()


def test_budget_policy_checkpoint_identity_and_default_active_fixture_replay():
    from pathlib import Path
    from hashlib import sha256
    from test_general_contact_native_curved_s3_static import _analysis
    default=_analysis(physical_fibres=True,plastic_fibres=True)
    explicit=_analysis(physical_fibres=True,plastic_fibres=True,integration_budget_policy='full_reference')
    active=_analysis(physical_fibres=True,plastic_fibres=True,integration_budget_policy='certified_active')
    assert default.identity==explicit.identity and active.identity!=default.identity
    assert default._checkpoint(())==explicit._checkpoint(())
    for reader,writer in ((default,active),(active,default)):
        raw=writer._checkpoint(())
        with pytest.raises(ValueError,match='definition/settings binding'):
            reader._restore(raw,sha256(raw).hexdigest())
    raw=active._checkpoint(())
    _,records=active._restore(raw,sha256(raw).hexdigest())
    assert active._checkpoint(records)==raw
    raw=(Path(__file__).parent/'data/general_contact_native_fibre_s3_plastic_active_v1.checkpoint').read_bytes()
    _,records=default._restore(raw,sha256(raw).hexdigest())
    assert default._checkpoint(records)==raw
    with pytest.raises(ValueError,match='definition/settings binding'):
        active._restore(raw,sha256(raw).hexdigest())
