"""Changed threshold decision semantics; no new physical criterion."""
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from anyfesolver.general_contact import s3_patch_static as owner

def fixture(targets=2):
    source=np.array(((0,0,0),(1,0,0),(1,1,0),(0,1,0)),float)
    triangle=np.array(((-2,-2,0),(4,-2,0),(-2,4,0)),float)
    target=np.vstack([triangle+[0,0,index*.01] for index in range(targets)])
    geometry=SimpleNamespace(source_face_indices=(np.arange(4),),
        target_face_indices=tuple(np.arange(3*i,3*i+3) for i in range(targets)),
        pair_laws=tuple(range(targets)),_check_same_side_overlap=Mock(),
        _candidate_pair_indices=Mock(return_value=None))
    instance=SimpleNamespace(settings=SimpleNamespace(d1_interior_profile=True,
        q4_source_gap_certificate='overlap_local_dev',q4_source_overlap_dual_dev=False),projected_work=False,
        law=geometry,gap_tolerance=5e-5)
    return instance,(source,target)

def test_old_safe_bound_skips_local_and_matching(monkeypatch):
    instance,positions=fixture()
    monkeypatch.setattr(owner,'q4_source_projected_gap_lower',Mock(return_value=-1e-6))
    local=Mock(side_effect=AssertionError('safe old bound must not refine'))
    matching=Mock(side_effect=AssertionError('safe old bound must not tighten'))
    monkeypatch.setattr(owner,'overlap_local_lower',local)
    monkeypatch.setattr(owner,'matching_projection_swept_lower',matching)
    assert owner.GeneralS3PatchManager._point_gap_budget_admissible(instance,positions)
    assert instance.law._check_same_side_overlap.call_count==2
    assert not local.called and not matching.called

def test_old_local_safe_bound_skips_matching(monkeypatch):
    instance,positions=fixture()
    monkeypatch.setattr(owner,'q4_source_projected_gap_lower',Mock(return_value=-1e-3))
    monkeypatch.setattr(owner,'overlap_local_lower',Mock(return_value=(-1e-6,1)))
    matching=Mock(side_effect=AssertionError('local safe bound must not tighten'))
    monkeypatch.setattr(owner,'matching_projection_swept_lower',matching)
    assert owner.GeneralS3PatchManager._point_gap_budget_admissible(instance,positions)
    assert not matching.called

def test_decision_refuses_first_pair_but_diagnostic_finishes_all(monkeypatch):
    instance,positions=fixture()
    instance.profile='normal_D1';instance._mesh_physical_ray_owner=None
    monkeypatch.setattr(owner,'q4_source_projected_gap_lower',Mock(return_value=-1e-3))
    monkeypatch.setattr(owner,'overlap_local_lower',Mock(return_value=(-1e-3,1)))
    matching=Mock(return_value=(-2e-4,1))
    monkeypatch.setattr(owner,'matching_projection_swept_lower',matching)
    assert not owner.GeneralS3PatchManager._point_gap_budget_admissible(instance,positions)
    assert matching.call_count==1
    assert matching.call_args.kwargs['gap_budget']==instance.gap_tolerance
    matching.reset_mock()
    lower=owner.GeneralS3PatchManager._gap_lower(instance,positions)
    assert lower==-2e-4 and matching.call_count==2
    assert all('gap_budget' not in c.kwargs
               and c.kwargs['refinement_gap_budget']==instance.gap_tolerance
               and c.kwargs['adaptive_depth']==7
               and c.kwargs['node_budget']==1024 for c in matching.call_args_list)

def test_later_grazing_chart_preserves_exception_before_early_refusal(monkeypatch):
    instance,(source,target)=fixture()
    target[3:]=np.array(((2,2,0),(2,3,0),(2,2,1)))
    monkeypatch.setattr(owner,'q4_source_projected_gap_lower',Mock(return_value=-1e-3))
    matching=Mock(return_value=(-1e-3,1));monkeypatch.setattr(owner,'matching_projection_swept_lower',matching)
    with pytest.raises(ValueError,match='target plane is nearly grazing'):
        owner.GeneralS3PatchManager._point_gap_budget_admissible(instance,(source,target))
    assert not matching.called

@pytest.mark.parametrize('lower',(float('nan'),-float('inf')),ids=('nan','negative_infinity'))
def test_unresolved_matching_certificate_is_not_admitted(monkeypatch,lower):
    instance,positions=fixture()
    monkeypatch.setattr(owner,'q4_source_projected_gap_lower',Mock(return_value=-1e-3))
    monkeypatch.setattr(owner,'overlap_local_lower',Mock(return_value=(-1e-3,1)))
    monkeypatch.setattr(owner,'matching_projection_swept_lower',Mock(return_value=(lower,1)))
    assert not owner.GeneralS3PatchManager._point_gap_budget_admissible(instance,positions)

def test_other_profile_delegates_original_diagnostic():
    instance,positions=fixture();instance.settings.d1_interior_profile=False
    instance._gap_lower=Mock(return_value=-1e-6)
    assert owner.GeneralS3PatchManager._point_gap_budget_admissible(instance,positions)
    instance._gap_lower.assert_called_once_with(positions)
