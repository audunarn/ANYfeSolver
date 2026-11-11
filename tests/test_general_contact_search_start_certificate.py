"""Search-local preparation controls; no external model or acceptance fixture."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from scipy import sparse
from anyfesolver.general_contact.core import ContactSolverSettings
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


def manager():
    m=object.__new__(GeneralS3PatchManager)
    m.settings=SimpleNamespace(d1_interior_profile=True,shell_skin_kinematics='reference_linear',
        q4_source_gap_certificate='overlap_local_dev',q4_source_overlap_dual_dev=True,
        maximum_swept_subdivisions=3)
    face=SimpleNamespace(node_ids=(1,2,3),reference_positions=np.zeros((3,3)))
    g=SimpleNamespace(source_face_indices=(np.array([0,1,2]),),
        target_face_indices=(np.array([0,1,2]),),source_faces=(face,),target_faces=(face,),
        source_node_ids=(1,2,3),target_node_ids=(4,5,6),reference_source=np.zeros((3,3)),
        reference_target=np.zeros((3,3)),pair_broadphase=True,pair_laws=(object(),),
        _check_same_side_overlap=lambda *a:None,_candidate_pair_indices=lambda *a:None)
    m.law=SimpleNamespace(geometry=g);m.projected_work=True
    m.reference=(np.zeros((3,3)),np.zeros((3,3)));m.reference_normals=(np.array([0.,0.,1.]),)*2
    m.map=sparse.eye(18,format='csr');m.n_dof=18;m.source_coordinates=9
    m.profile='test-only';m.finite_skin=None;m.gap_tolerance=50e-6;m.swept_trial_cutbacks=0
    return m


def pose():
    return (np.zeros((3,3)),np.zeros((3,3)))


def test_exact_key_and_nonaliasing():
    m=manager();u=np.zeros(18);p=pose();e=pose()
    m._start_point_search={'entry':None}
    with patch.object(m,'_point_gap_budget_admissible',return_value=True) as point:
        assert m._search_start_point_admissible(u,p,e)
        assert m._search_start_point_admissible(u.copy(),tuple(a.copy() for a in p),e)
        assert point.call_count==1
        p[0][0,0]=np.nextafter(0.,1.)
        assert m._search_start_point_admissible(u,p,e)
        assert point.call_count==2


@pytest.mark.parametrize('dependency',['start','error','reference','map','topology','settings','method'])
def test_actual_dependency_miss(dependency):
    m=manager();u=np.zeros(18);p=pose();e=pose();m._start_point_search={'entry':None}
    with patch.object(m,'_point_gap_budget_admissible',return_value=True) as point:
        assert m._search_start_point_admissible(u,p,e)
        if dependency=='start':u[0]=np.nextafter(0.,1.)
        elif dependency=='error':e[0][0,0]=np.nextafter(0.,1.)
        elif dependency=='reference':m.reference[0][0,0]=1.
        elif dependency=='map':m.map.data[0]=2.
        elif dependency=='topology':m.law.geometry.source_face_indices[0][0]=1
        elif dependency=='settings':m.settings.maximum_swept_subdivisions=4
        else:m.law.geometry._candidate_pair_indices=lambda *a:None
        assert m._search_start_point_admissible(u,p,e)
        assert point.call_count==2


def test_real_search_hits_start_but_not_changed_end_and_restores_context():
    m=manager();u=np.zeros(18);p=pose();e=pose();previous={'entry':('outer',)}
    m._start_point_search=previous;ends=[]
    def sweep(start,end):
        assert m._search_start_point_admissible(start,p,e)
        ends.append(end.copy())
        return len(ends)==3
    with patch.object(m,'_point_gap_budget_admissible',return_value=True) as point,patch.object(m,'_swept_admissible',side_effect=sweep):
        assert m.admissible_step_fraction(u,np.ones(18))==.25
        assert point.call_count==1
    assert m._start_point_search is previous
    assert [a[0] for a in ends]==[1.,.5,.25]
    assert m.swept_trial_cutbacks==1


def test_exception_does_not_publish_and_search_context_restored():
    m=manager();u=np.zeros(18);p=pose();e=pose();previous={'entry':('outer',)}
    m._start_point_search=previous
    def sweep(start,end):return m._search_start_point_admissible(start,p,e)
    with patch.object(m,'_point_gap_budget_admissible',side_effect=ValueError('real refusal')),patch.object(m,'_swept_admissible',side_effect=sweep):
        with pytest.raises(ValueError,match='real refusal'):m.admissible_step_fraction(u,np.ones(18))
    assert m._start_point_search is previous
    m._start_point_search={'entry':None}
    with patch.object(m,'_point_gap_budget_admissible',return_value=False) as point:
        assert not m._search_start_point_admissible(u,p,e)
        assert not m._search_start_point_admissible(u,p,e)
        assert point.call_count==2 and m._start_point_search['entry'] is None


def test_reentrant_search_restores_outer_entry():
    m=manager();u=np.zeros(18);p=pose();e=pose();nested=False
    def sweep(start,end):
        nonlocal nested
        assert m._search_start_point_admissible(start,p,e)
        context=m._start_point_search;key=context['entry']
        if not nested:
            nested=True
            assert m.admissible_step_fraction(start,np.ones(18))==1.
            assert m._start_point_search is context and context['entry']==key
        return True
    with patch.object(m,'_point_gap_budget_admissible',return_value=True) as point,patch.object(m,'_swept_admissible',side_effect=sweep):
        assert m.admissible_step_fraction(u,np.ones(18))==1.
        assert point.call_count==2
    assert not hasattr(m,'_start_point_search')


def test_false_route_preserves_requests_and_unknown_key_bypasses():
    m=manager();u=np.zeros(18);p=pose();e=pose();m._start_point_search={'entry':None}
    with patch.object(m,'_point_gap_budget_admissible',return_value=True) as point:
        m.settings.q4_source_overlap_dual_dev=False
        for _ in range(2):assert m._search_start_point_admissible(u,p,e)
        assert point.call_count==2 and m._start_point_search['entry'] is None
        m.settings.q4_source_overlap_dual_dev=True;m.settings.unknown=object()
        for _ in range(2):assert m._search_start_point_admissible(u,p,e)
        assert point.call_count==4 and m._start_point_search['entry'] is None


def test_same_function_different_bound_receiver_misses():
    from types import MethodType
    m=manager();u=np.zeros(18);p=pose();e=pose();m._start_point_search={'entry':None}
    def candidate(self,*args):return None
    g=m.law.geometry;first=SimpleNamespace(option=1);second=SimpleNamespace(option=2)
    g._candidate_pair_indices=MethodType(candidate,first)
    with patch.object(m,'_point_gap_budget_admissible',return_value=True) as point:
        assert m._search_start_point_admissible(u,p,e)
        g._candidate_pair_indices=MethodType(candidate,second)
        assert m._search_start_point_admissible(u,p,e)
        assert point.call_count==2


@pytest.mark.parametrize('admitted,expected',[(True,1.),(False,0.)])
def test_full_and_zero_return_clear_entry_and_new_search_rechecks(admitted,expected):
    m=manager();u=np.zeros(18);p=pose();e=pose()
    def sweep(start,end):
        assert m._search_start_point_admissible(start,p,e)
        return admitted
    with patch.object(m,'_point_gap_budget_admissible',return_value=True) as point,patch.object(m,'_swept_admissible',side_effect=sweep):
        for _ in range(2):
            assert m.admissible_step_fraction(u,np.ones(18))==expected
            assert not hasattr(m,'_start_point_search')
        assert point.call_count==2


def test_unsupported_finite_dispatch_does_not_enable_search_entry():
    m=manager();m.finite_skin=object();u=np.zeros(18);p=pose();e=pose();calls=[]
    def sweep(start,end):
        assert not hasattr(m,'_start_point_search')
        assert m._search_start_point_admissible(start,p,e)
        calls.append(end.copy());return len(calls)==3
    with patch.object(m,'_point_gap_budget_admissible',return_value=True) as point,patch.object(m,'_swept_admissible',side_effect=sweep):
        assert m.admissible_step_fraction(u,np.ones(18))==.25
        assert point.call_count==3
    assert not hasattr(m,'_start_point_search')
