"""Disposable exact preparation: independent controls and stale-input rejection."""
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import runpy
import numpy as np
import pytest
from anyfesolver.general_contact.s3_pressure_hermite_bounds import (
    _PreparationCache,s3_pressure_hermite_bounds)
from anyfesolver.general_contact.s3_overlap_hermite_gap import s3_overlap_hermite_gap_bounds
from anyfesolver.general_contact.normal_s3_skin_path import prepare_normal_s3_skin_path
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


def inputs():
    source=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    target=np.array([[-1.,-1.,-.001],[3.,-1.,-.001],[-1.,3.,-.001]])
    return [source.copy(),target.copy(),((0,1,2),),source.copy(),source.copy(),
            target.copy(),target.copy(),np.zeros((3,3)),np.zeros((3,3)),
            np.zeros((3,3)),np.zeros((3,3)),1.]


def strong(values,cache=None,**controls):
    return s3_overlap_hermite_gap_bounds(*values,
        **dict(gap_tolerance_m=.0001,_preparation_cache=cache,**controls))


def weak(values,cache=None,**controls):
    return s3_pressure_hermite_bounds(*values,
        **{'gap_tolerance_m':1.e-10,'active':np.zeros(3,dtype=bool),
           '_preparation_cache':cache,**controls})


def test_one_payload_is_shared_without_reusing_decisions():
    values=inputs();cache=_PreparationCache()
    actual=strong(values,cache)
    assert actual.success and actual==strong(values)
    assert weak(values,cache)==weak(values)
    assert cache.preparations==1 and cache.hits==1
    closed=weak(values,cache,active=np.ones(3,dtype=bool))
    assert closed==weak(values,active=np.ones(3,dtype=bool))
    assert not closed.success and closed.status=='pressure_phase_violation'
    bounded=strong(values,cache,maximum_spatial_cells=1)
    assert bounded==strong(values,maximum_spatial_cells=1)
    assert not bounded.success and bounded.status=='spatial_budget_exhausted'
    cancelled=strong(values,cache,cancel=lambda:True)
    assert not cancelled.success and cancelled.status=='cancelled'
    assert cache.preparations==1 and cache.hits==3


@pytest.mark.parametrize('index',(0,1,3,4,5,6,7,8,9,10))
def test_each_reference_pose_and_rate_mutation_invalidates_exact_snapshot(index):
    values=inputs();cache=_PreparationCache()
    strong(values,cache)
    original_key=cache._key
    # Mutate the SAME input array: no identity/pointer-based reuse is permitted.
    values[index][0,2]+=1.e-6
    assert strong(values,cache)==strong(values)
    assert cache.preparations==2 and cache.hits==0 and cache._key!=original_key
    # Payload contains original-face tuples/Fractions, never live array aliases.
    assert isinstance(cache._faces[0].polynomials[0][0],Fraction)


def test_duration_and_original_topology_invalidate_preparation():
    values=inputs();cache=_PreparationCache()
    strong(values,cache);values[-1]=2.
    assert strong(values,cache)==strong(values)
    assert cache.preparations==2
    values[2]=((0,2,1),)
    actual=strong(values,cache)
    assert actual==strong(values) and not actual.success
    assert actual.status=='orientation_invalid' and cache.preparations==3


@pytest.mark.parametrize('kind',('boolean_face','nonfinite','complex','shape','phase'))
def test_cached_payload_cannot_hide_invalid_new_inputs(kind):
    values=inputs();cache=_PreparationCache();strong(values,cache)
    if kind=='boolean_face':values[2]=((False,1,2),)
    elif kind=='nonfinite':values[3][0,0]=np.nan
    elif kind=='complex':values[3]=values[3].astype(complex)
    elif kind=='shape':values[3]=values[3][:2]
    with pytest.raises(ValueError):
        if kind=='phase':weak(values,cache,active=np.zeros(4,dtype=bool))
        else:strong(values,cache)
    assert cache.preparations==1 and cache.hits==0


def test_hidden_approach_and_gap_limits_are_not_cached_successes():
    values=inputs();values[7][:,2]=-.012;values[8][:,2]=.012
    cache=_PreparationCache()
    relaxed=s3_overlap_hermite_gap_bounds(*values,gap_tolerance_m=.01,_preparation_cache=cache)
    assert relaxed.success
    strict=s3_overlap_hermite_gap_bounds(*values,gap_tolerance_m=.0001,_preparation_cache=cache)
    assert not strict.success
    hidden=s3_overlap_hermite_gap_bounds(*values,gap_tolerance_m=.01,
        exclude_open_endpoint_approach_tolerance_m=.0001,maximum_depth=0,
        _preparation_cache=cache)
    assert not hidden.success and hidden.status=='hidden_approach_unresolved'
    assert cache.preparations==1 and cache.hits==2


@pytest.mark.parametrize('populated',(False,True))
@pytest.mark.parametrize('index',(0,7))
def test_mixed_boolean_coordinate_lists_preserve_original_rejection(populated,index):
    values=inputs();cache=_PreparationCache()
    if populated:strong(values,cache)
    values[index]=values[index].tolist();values[index][0][0]=False
    with pytest.raises(ValueError):strong(values,cache)
    with pytest.raises(ValueError):strong(values)
    assert cache.preparations==int(populated) and cache.hits==0


def test_failed_preparation_preserves_previous_disposable_payload():
    first=inputs();cache=_PreparationCache();result=strong(first,cache)
    original=cache._faces
    bad=inputs();bad[0][:]=0.
    with pytest.raises(ValueError,match='degenerate'):strong(bad,cache)
    assert cache._faces is original and cache.preparations==1
    assert strong(first,cache)==result and cache.hits==1


def test_owner_shares_one_path_and_replacement_owns_a_new_slot():
    example=runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_deforming_s3_impact_dev.py'))
    model,definition,modes=example['build_model']()
    manager=GeneralS3PatchManager(model,definition,example['_fixture']['SETTINGS'])
    owner=prepare_normal_s3_skin_path(model,manager,modes)
    zero=np.zeros(modes.shape[0])
    _,failure=owner.overlap_sweep(manager,zero,zero,zero,zero,2.5e-5)
    assert failure is None
    proof=owner.pressure_sweep(manager,zero,zero,zero,zero,2.5e-5,
        active=np.zeros(4,dtype=bool),gap_tolerance_m=1.e-10)
    assert proof.success and owner._preparation.preparations==1 and owner._preparation.hits==1
    other=replace(owner)
    assert other._preparation is not owner._preparation
    assert other._preparation.preparations==other._preparation.hits==0
