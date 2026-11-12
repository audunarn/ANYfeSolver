"""Conservative adaptive source-normal certificate proof regressions.

These portable bodies retain installed evidence from the bounded adaptive
certificate development checks. Baseline-module and actual-manager checks
remain in the raw evidence record.
"""
import numpy as np
import pytest
from anyfesolver.general_contact import q4_overlap_local_bound as bounds

BUDGET=5e-5

def source_plane():
    source=np.array(((0,0,-1e-4),(1,0,1e-4),(1,1,1e-4),(0,1,-1e-4)),float)
    target=np.array(((.3,.2,0),(.7,.2,0),(.3,.6,0)),float)
    return source,target

def adaptive(source,target,*,node_budget=1024,errors=None):
    details={}
    lower,visited=bounds.matching_projection_swept_lower(source,source,target,target,3,
        gap_budget=BUDGET,adaptive_depth=7,node_budget=node_budget,
        endpoint_errors=errors,diagnostics=details)
    assert visited==details['visited_nodes'] and 1<=visited<=node_budget
    return lower,details

def test_declared_partial_front_refines_without_changing_gap_measure():
    source,target=source_plane()
    original,_=bounds.matching_projection_swept_lower(source,source,target,target,3,gap_budget=BUDGET)
    lower,details=adaptive(source,target)
    # Target-corner extrema are exact for these parallel planar normals:
    # n_s=(-.0002,0,1)/norm and n_s dot(s_anchor-target).
    normal=np.array((-.0002,0,1.));normal/=np.linalg.norm(normal)
    exact_min=min(float(normal@(source[0]-point)) for point in target)
    assert original < -BUDGET and lower>=-BUDGET
    assert original<=lower<=exact_min
    assert not details['node_budget_exhausted']

def test_node_exhaustion_keeps_complete_parent_bound():
    source,target=source_plane()
    lower,details=adaptive(source,target,node_budget=1)
    parent,_=bounds.matching_projection_swept_lower(source,source,target,target,0,gap_budget=BUDGET)
    assert lower==parent and lower<-BUDGET
    assert details['node_budget_exhausted'] and details['visited_nodes']==1

@pytest.mark.parametrize('completed_children',(1,2,3))
def test_post_initial_partial_children_keep_parent_refusal(completed_children):
    # This triangle strictly covers the flat square, so no initial cell is
    # disjoint/positive/threshold-safe: the complete depth3 tree has 85 nodes.
    source=np.array(((0,0,-1e-4),(1,0,-1e-4),(1,1,-1e-4),(0,1,-1e-4)),float)
    target=np.array(((-4,-4,0),(8,-4,0),(-4,8,0)),float)
    baseline_details={}
    baseline,initial_nodes=bounds.matching_projection_swept_lower(
        source,source,target,target,3,refinement_gap_budget=BUDGET,
        adaptive_depth=3,node_budget=85,diagnostics=baseline_details)
    assert initial_nodes==85 and baseline<-BUDGET
    # One more depth is exactly four terminal children: node86 re-evaluates
    # the selected initial leaf, then 1--3 nodes compute its children. The
    # finite budget prevents the fourth child, so the whole parent must remain.
    details={};cap=86+completed_children
    refused,visited=bounds.matching_projection_swept_lower(
        source,source,target,target,3,refinement_gap_budget=BUDGET,
        adaptive_depth=4,node_budget=cap,diagnostics=details)
    assert visited==cap and details['node_budget_exhausted']
    assert details['depth_exhausted']
    assert np.asarray(refused).view(np.uint64)==np.asarray(baseline).view(np.uint64)
    assert refused<-BUDGET  # missing fourth region cannot be silently accepted

def test_full_diagnostic_retains_complete_nonadaptive_floor():
    source,target=source_plane()
    complete,_=bounds.matching_projection_swept_lower(source,source,target,target,3)
    refined,visited=bounds.matching_projection_swept_lower(source,source,target,target,3,
        refinement_gap_budget=BUDGET,adaptive_depth=7,node_budget=1024)
    assert refined>=complete and visited<=1024
    with pytest.raises(ValueError,match='initial complete partition'):
        bounds.matching_projection_swept_lower(source,source,target,target,3,
            refinement_gap_budget=BUDGET,adaptive_depth=7,node_budget=1)

def test_depth_limit_and_explicit_budget_refuse_actual_penetration():
    source,target=source_plane();source[:,2]-=.001
    lower,details=adaptive(source,target)
    assert lower<-BUDGET
    assert details['depth_exhausted'] or details['node_budget_exhausted']
    with pytest.raises(ValueError):
        bounds.matching_projection_swept_lower(source,source,target,target,3,
            gap_budget=BUDGET,adaptive_depth=8,node_budget=1024)
    with pytest.raises(ValueError):
        bounds.matching_projection_swept_lower(source,source,target,target,3,
            gap_budget=BUDGET,adaptive_depth=7,node_budget=True)

def test_error_chart_and_nonfinite_refusal(monkeypatch):
    source,target=source_plane()
    error=np.zeros((4,3));error[:,2]=.001
    enclosed,_=adaptive(source,target,errors=(error,error,0.,0.))
    assert enclosed<-BUDGET
    refused,details=adaptive(source,target[[0,2,1]])
    assert refused==-np.inf and details['unresolved_chart']
    original=bounds._linear_quadratic_dot_controls
    def broken(*args):
        controls=list(original(*args));lo,hi=controls[-1]
        controls[-1]=(np.full_like(lo,np.nan),hi)
        return tuple(controls)
    monkeypatch.setattr(bounds,'_linear_quadratic_dot_controls',broken)
    refused,_=adaptive(source,target)
    assert refused==-np.inf

def test_hidden_time_penetration_not_pruned_by_refinement():
    source=np.array(((-.01,-.01,0),(.01,-.01,0),(.01,.01,0),(-.01,.01,0)),float)
    first=np.array(((-3,0,1.1),(1,-10,-.9),(1,10,-.9)),float)
    last=np.array(((-1,0,-.9),(3,-10,1.1),(3,10,1.1)),float)
    details={}
    lower,visited=bounds.matching_projection_swept_lower(source,source,first,last,3,
        gap_budget=BUDGET,adaptive_depth=7,node_budget=1024,diagnostics=details)
    assert lower<=-.09 and visited<=1024
