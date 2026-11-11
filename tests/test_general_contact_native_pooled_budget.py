"""V14 opt-in discriminator; analytic mechanics and exact reserve accounting."""
from fractions import Fraction
import json
import os
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.general_contact.native_pair_error_budget import PooledQuadratureBudget, within_reserve
from anyfesolver.general_contact.native_retained_whole_pair import (
    NativeRetainedWholeCurvePair, _exact_tiny_cell_omission_bound, _private_factored_gap,
    _pooled_connected_junction_partition,
    _pooled_interval_open, _pooled_endpoint_feature_inventory, _vertex_fixed_projection_feature,
)
from anyfesolver.general_contact.native_retained_body_projection import strict_projection_station, charged_junction_omission
from anyfesolver.general_contact.beam_curve_pair import b3_distance_jet, _b3_distance_jet_at_station
from anyfesolver.general_contact.native_curved_pair_partition import physical_point_map
from anyfesolver._ge_beam3_connected_dynamic import _RetainedMidpointAssembly
from examples.general_contact_native_pair_impact_dev import build, config


def test_pooled_budget_exact_reserves_and_parallel_line_oracle():
    ledger = PooledQuadratureBudget((2.,2.,2.))
    ledger.charge((.75,.75,.75)); ledger.charge((.25,.25,.25))
    assert ledger.used == (Fraction(1),)*3 and ledger.remaining() == (0.,)*3
    assert not ledger.admits((np.nextafter(0.,1.),0.,0.))
    with pytest.raises(MemoryError):
        ledger.charge((np.nextafter(0.,1.),0.,0.))
    assert ledger.charges == 2
    # Ordinary addition would discard this charge at the exact half reserve.
    assert 1.+2.**-54 == 1.
    assert not within_reserve((1.,0.,0.),(2.**-54,0.,0.),(2.,2.,2.),half=True)
    assert not within_reserve((2.,0.,0.),(2.**-54,0.,0.),(2.,2.,2.))
    for bad in ((-1.,0.,0.),(np.inf,0.,0.),(np.nan,0.,0.)):
        with pytest.raises(ValueError):
            ledger.admits(bad)
    # Complete junction bulk plus root-H enclosure is charged to B/E once;
    # zero approximation avoids an unresolved rounded midpoint station.
    bound,aggregate,boundary = charged_junction_omission(
        np.array((1e-12,1e-11,1e-10)),1e-11,np.zeros(3),np.zeros(3),np.array((1e-8,1e-7,1e-6)))
    np.testing.assert_array_equal(aggregate,boundary)
    assert bound[2] >= 1.1e-10
    with pytest.raises(MemoryError):
        charged_junction_omission(np.array((2.**-54,0.,0.)),0.,np.zeros(3),np.array((1.,0.,0.)),np.array((2.,2.,2.)))

    owner = build(separation=.08); pair = owner.contact
    a = np.array(((0.,0.,0.),(1.,0.,0.),(2.,0.,0.)))
    b = a+np.array((0.,.08,0.)); low = np.zeros((3,3)); rotations = np.repeat(np.eye(3)[None],2,axis=0)
    poses = (a,low,rotations,b,low,rotations)
    diagnostics = {}
    actual = pair.evaluate(*poses,_diagnostics=diagnostics)
    # Symmetric half-weighted line measure has length 2, k=100, p=.02.
    np.testing.assert_allclose(actual.potential_j,.04,rtol=0.,atol=1e-8)
    np.testing.assert_allclose(actual.beam_a_force_n,(0.,-4.,0.),rtol=0.,atol=1e-7)
    np.testing.assert_allclose(actual.beam_b_force_n,(0.,4.,0.),rtol=0.,atol=1e-7)
    direction = np.zeros(84); direction[[1,7,13]] = 1.
    np.testing.assert_allclose(actual.residual@direction,4.,rtol=0.,atol=1e-7)
    np.testing.assert_allclose(direction@actual.tangent@direction,200.,rtol=0.,atol=1e-6)
    limits = (pair.energy_tolerance,pair.force_tolerance,pair.tangent_tolerance)
    assert within_reserve(diagnostics['boundary_error'],(0.,0.,0.),limits,half=True)
    assert within_reserve(diagnostics['error'],(0.,0.,0.),limits)
    assert all(Fraction(float(v)) <= Fraction(float(t))/2 for v,t in zip(diagnostics['pooled_quadrature']['used'],limits))

    historical = NativeRetainedWholeCurvePair(pair.cells_a[0].element_id,pair.cells_a[0].reference,pair.node_ids_a,pair.radius_a,
        pair.cells_b[0].element_id,pair.cells_b[0].reference,pair.node_ids_b,pair.radius_b,
        line_stiffness_n_per_m2=pair.stiffness,distance_tolerance_m=pair.distance_tolerance,
        feature_tolerance_m=pair.feature_tolerance,search_parameter_tolerance=pair.search_parameter_tolerance,
        maximum_search_depth=pair.maximum_search_depth,maximum_search_cells=pair.maximum_search_cells,
        energy_tolerance_j=pair.energy_tolerance,force_tolerance_n=pair.force_tolerance,
        tangent_tolerance_n_per_m=pair.tangent_tolerance,maximum_integration_intervals=pair.maximum_integration_intervals,
        certify_curved_ownership=True)
    assert not historical.pooled_quadrature_budget and historical.identity != pair.identity
    old = historical.evaluate(*poses)
    np.testing.assert_allclose(actual.residual,old.residual,rtol=0.,atol=1e-7)
    np.testing.assert_allclose(actual.tangent,old.tangent,rtol=0.,atol=1e-6)

    source = pair.cells_a[0].evaluate(a,low,rotations).positions
    target = pair.cells_b[0].evaluate(b,low,rotations).positions
    station = .37; t = strict_projection_station(source,target,station)
    assert abs(t-station) <= 2e-15
    jet = _b3_distance_jet_at_station(source,target,station,pair.radius,t)
    old_jet = b3_distance_jet(source,target,station,pair.radius)
    np.testing.assert_allclose(jet.gradient,old_jet.gradient,rtol=0.,atol=1e-14)
    np.testing.assert_allclose(jet.hessian,old_jet.hessian,rtol=0.,atol=1e-12)
    lower,upper,nominal = _private_factored_gap(source,target,station,pair.radius,t)
    assert lower <= .08-.1 <= upper and abs(nominal-jet.gap) <= 1e-15
    with pytest.raises(ValueError,match='strict convexity'):
        strict_projection_station(source,np.repeat(target[:1],3,axis=0),station)

    # Existing absolute bound is for 18 independent cell coordinates. Embed
    # into the production 36 cell / 30 unique point chart, using its 2/4 bound.
    left,right = station,station+2.**-30
    bound = _exact_tiny_cell_omission_bound(source,target,pair.reference_curves[0],pair.radius,pair.stiffness,left,right)*np.array((1.,2.,4.))
    jacobian = physical_point_map(); gradient = np.zeros(36); hessian = np.zeros((36,36))
    slots = np.r_[np.arange(9),np.arange(18,27)]
    measure = pair.stiffness*(right-left)
    gradient[slots] = measure*jet.gap*jet.gradient
    hessian[np.ix_(slots,slots)] = measure*(np.outer(jet.gradient,jet.gradient)+jet.gap*jet.hessian)
    values = np.array((.5*measure*jet.gap**2,np.max(np.abs(jacobian.T@gradient)),np.max(np.abs(jacobian.T@hessian@jacobian))))
    assert np.all(np.isfinite(bound)) and np.all(bound >= values) and np.all(bound >= 0.)
    report = dict(energy_j=actual.potential_j,force_a_n=actual.beam_a_force_n.tolist(),
        normal_stiffness_n_per_m=float(direction@actual.tangent@direction),
        quadrature=diagnostics['pooled_quadrature'],boundary=diagnostics['boundary_error'].tolist(),
        aggregate=diagnostics['error'].tolist(),strip_bound=bound.tolist(),strip_oracle=values.tolist(),
        kernel_identity=pair.identity,historical_identity=historical.identity,
        station=t,gap_bracket=(lower,upper),integration_visited=actual.integration_cells_checked)
    if os.environ.get('C8_JOB_DIR'):
        (Path(os.environ['C8_JOB_DIR'])/'pooled_oracle.json').write_text(json.dumps(report,indent=2)+'\n')


def test_saved_perturbation_connected_junction_inventory():
    """Reconstruct V3b producer's exact stage chart, without a contact solve.

    The saved failed selector defines dt=.01 and both arbitrary epsilons. Bare
    mechanics evaluation supplies only the stage record; contact projection
    inventory uses its real retained curves, never snapped/interpolated traces.
    """
    owner = build(); accepted = owner.initialize(physical_velocity=config(owner).initial_velocity)
    direction = np.sin(np.arange(owner.size)+.3); direction[list(owner.p)] *= 100.
    evidence = []; certified = 0; one_sided = 0
    evaluate = owner._evaluate
    try:
        owner._evaluate = lambda state,token=None: _RetainedMidpointAssembly._evaluate(owner,state,token)
        for epsilon in (1e-6,1e-7):
            stage = owner._stage(accepted,epsilon*direction,.01,None)[0]
            for index,search in enumerate(owner.contact._geometry_searches(owner._poses(stage))):
                with pytest.raises(MemoryError,match='search cell budget'):
                    _pooled_connected_junction_partition(search,0.,.0009765625,remaining_cells=0)
                partition = _pooled_connected_junction_partition(search,0.,.0009765625,remaining_cells=10000)
                if partition is not None:
                    certified += 1
                    assert partition.pieces[0][0] == 0. and partition.pieces[-1][1] == .0009765625
                    assert all(stop > start and target in (0,1) for start,stop,target in partition.pieces)
                    assert partition.visited <= 10000
                    if len(partition.pieces) == 1 and partition.visited == 1:
                        one_sided += 1
                        assert _pooled_connected_junction_partition(search,0.,.0009765625,remaining_cells=1).visited == 1
                evidence.append(dict(epsilon=epsilon,source=index,
                    pieces=None if partition is None else partition.pieces,
                    visited=None if partition is None else partition.visited))
    finally:
        owner._evaluate = evaluate
    assert certified > 0 and one_sided > 0
    if os.environ.get('C8_JOB_DIR'):
        (Path(os.environ['C8_JOB_DIR'])/'saved_junction_inventory.json').write_text(json.dumps(dict(
            provenance='RECONSTRUCTED_SAVED_V3B_SELECTOR_UNKNOWN_AND_EXISTING_STAGE_CHART_NO_CONTACT_SOLVE',
            certified=certified,inventory=evidence),indent=2)+'\n')


def test_saved_open_stage_complete_interval_certificate():
    # Exact binary64 curve data from c8_impact_diagnostic_v3e's failing call.
    source = np.array(((0.,.00015000000000000001,0.),(.5,.00015000000000000001,0.),(1.,.00015000000000000001,0.)))
    target = np.array(((0.,.10035000000000001,0.),(.5,.10035000000000001,0.),(1.,.10035000000000001,0.)))
    for other in (target,target+np.array((1.,0.,0.))):
        assert _pooled_interval_open(source,other,0.,1.,.1)
    # Independent rational box-distance oracle and one-ulp closed refusal.
    separation = Fraction(float(target[0,1]))-Fraction(float(source[0,1]))
    assert separation*separation > Fraction(.1)**2
    zero_source = source.copy(); zero_source[:,1] = 0.
    near = zero_source.copy(); near[:,1] = np.nextafter(.1,0.)
    assert Fraction(float(near[0,1]))**2 < Fraction(.1)**2
    assert not _pooled_interval_open(zero_source,near,0.,1.,.1)
    assert not _pooled_interval_open(source,source+np.array((0.,.08,0.)),0.,1.,.1)
    bowed = target.copy(); bowed[1,1] = .09
    assert not _pooled_interval_open(source,bowed,0.,1.,.1)
    owner = build(separation=.1002); pair = owner.contact
    a = np.array(((0.,0.,0.),(1.,0.,0.),(2.,0.,0.))); b = a+np.array((0.,.1002,0.))
    low = np.zeros_like(a); rotations = np.repeat(np.eye(3)[None],2,axis=0)
    result = pair.evaluate(a,low,rotations,b,low,rotations)
    assert result.potential_j == 0. and result.integration_cells_checked <= 1000
    np.testing.assert_array_equal(result.residual,np.zeros(84))
    np.testing.assert_array_equal(result.tangent,np.zeros((84,84)))


def test_saved_arbitrary_stage_exact_vertex_kkt_discriminator():
    """Classify both original derivative epsilons/signs without contact solve.

    This passes only exact inventory/provenance checks, not the unmet FD gate.
    A genuine two-interior label explicitly preserves mature fallback needs.
    """
    from anyfesolver.general_contact.native_retained_body_projection import endpoint_projection_bands
    inventory = _oracle_inventory
    owner = build(); accepted = owner.initialize(physical_velocity=config(owner).initial_velocity)
    direction = np.sin(np.arange(owner.size)+.3); direction[list(owner.p)] *= 100.
    evaluate = owner._evaluate; records = []
    try:
        owner._evaluate = lambda state,token=None: _RetainedMidpointAssembly._evaluate(owner,state,token)
        for epsilon in (1e-6,1e-7):
            for sign in (1.,-1.):
                stage = owner._stage(accepted,sign*epsilon*direction,.01,None)[0]
                poses = owner._poses(stage)
                states = tuple(cell.evaluate(*poses[:3]) for cell in owner.contact.cells_a)
                np.testing.assert_array_equal(states[0].positions[2],states[1].positions[0])
                np.testing.assert_array_equal(states[0].geometric_jacobian[6:9],states[1].geometric_jacobian[:3])
                workmap = physical_point_map()
                np.testing.assert_array_equal(workmap[6:9],workmap[9:12])
                np.testing.assert_array_equal(workmap[24:27],workmap[27:30])
                search = owner.contact._geometry_searches(poses)[3]
                with pytest.raises(MemoryError,match='search cell budget'):
                    inventory(search,endpoint_projection_bands,maximum_cells=0)
                row = inventory(search,endpoint_projection_bands,maximum_cells=10000)
                assert row['status'] == 'COMPLETE_EXACT_KKT_INVENTORY' and row['shared_vertex']
                assert row['visited'] <= 10000 and row['root_free']
                partition,used,labels = _pooled_endpoint_feature_inventory(search,0.,1.,remaining_cells=10000,
                    reference=owner.contact.reference_curves[3],radius=owner.contact.radius,
                    stiffness=owner.contact.stiffness,shared_workmap=True)
                assert 1 <= used <= row['visited'] and partition is not None and partition.visited == 0
                assert all(label != 'GENUINE_TWO_INTERIOR' for label in labels)
                assert all(np.all(np.isfinite(bulk)) and np.all(bulk >= 0) and root_h == 0
                           for _,_,bulk,root_h in partition.bands)
                assert _pooled_endpoint_feature_inventory(search,0.,1.,remaining_cells=1,
                    reference=owner.contact.reference_curves[3],radius=owner.contact.radius,
                    stiffness=owner.contact.stiffness,shared_workmap=False)[:2] == (None,1)

                covered = sorted([r['interval'] for r in row['root_free']+row['root_strips']])
                assert covered[0][0] == 0. and covered[-1][1] == 1.
                assert all(a[1] == b[0] for a,b in zip(covered[:-1],covered[1:]))
                records.append(dict(epsilon=epsilon,sign=sign,source=3,dt=.01,
                    native_endpoint_workmap_exact=True,physical30_endpoint_workmap_exact=True,**row))
    finally:
        owner._evaluate = evaluate
    assert len(records) == 4
    if os.environ.get('C8_JOB_DIR'):
        (Path(os.environ['C8_JOB_DIR'])/'exact_vertex_kkt_inventory.json').write_text(json.dumps(dict(
            provenance='ORIGINAL_ARBITRARY_DIRECTION_AND_PUBLIC_ACCEPTED_STATE_EXISTING_STAGE_CHART_NO_CONTACT_SOLVE',
            original_fd_gate='NONPASS_NOT_RECLASSIFIED',inventory=records),indent=2)+'\n')



def test_fixed_winning_projection_feature_requires_complete_endpoint_kkt():
    """Source-only V3k regression oracle; not executed by the V3j pass group."""
    target=np.array(((0.,0.,0.),(.5,0.,0.),(1.,0.,0.)))
    source=lambda xs: np.array([(x,.08,0.) for x in xs])
    assert _vertex_fixed_projection_feature(source((.2,.5,.8)),target,0.,1.) == 'FIXED_INTERIOR'
    assert _vertex_fixed_projection_feature(source((-1.,-.5,-.1)),target,0.,1.) == 'FIXED_START'
    assert _vertex_fixed_projection_feature(source((1.1,1.5,2.)),target,0.,1.) == 'FIXED_END'
    assert _vertex_fixed_projection_feature(source((-.2,.5,1.2)),target,0.,1.) is None
    assert _vertex_fixed_projection_feature(source((0.,.2,.4)),target,0.,1.) is None
    assert _vertex_fixed_projection_feature(source((.6,.8,1.)),target,0.,1.) is None

    # Strict losing-cell vertex KKT gives winner dominance, but the winning
    # horizontal cell changes interior->end at x=1. This must not fast-return.
    from types import SimpleNamespace
    crossing=np.array(((.9,-.08,0.),(1.,-.08,0.),(1.1,-.08,0.)))
    vertical=np.array(((1.,0.,0.),(1.,.5,0.),(1.,1.,0.)))
    assert _vertex_fixed_projection_feature(crossing,target,0.,1.) is None
    partition,visited,labels=_pooled_endpoint_feature_inventory(
        SimpleNamespace(source=crossing,targets=(target,vertical)),0.,1.,
        remaining_cells=10000,reference=crossing,radius=.1,stiffness=100.,shared_workmap=True)
    assert 1 <= visited <= 10000
    assert partition is None or partition.bands
    if partition is not None:
        assert all(np.all(np.isfinite(bulk)) and np.all(bulk >= 0) and root_h == 0
                   for _,_,bulk,root_h in partition.bands)


# Pure test-local exact oracle, preserved from the original frozen diagnostic.
# Ordinary checkout tests require no external raw evidence directory.
def _oracle_rows(curve):
    return tuple(tuple(Fraction(float(value)) for value in row) for row in curve)
def _oracle_dot(a,b):
    return sum(x*y for x,y in zip(a,b))
def _oracle_sub(a,b):
    return tuple(x-y for x,y in zip(a,b))
def _oracle_coefficients(curve):
    p=_oracle_rows(curve)
    return p[0],tuple(-3*x+4*y-z for x,y,z in zip(*p)),tuple(2*x-4*y+2*z for x,y,z in zip(*p))
def _oracle_at(polynomial,s):
    a,b,c=polynomial
    return a+b*s+c*s*s
def _oracle_extrema(polynomial,left,right):
    a,b,c=polynomial; lo,hi=Fraction(float(left)),Fraction(float(right))
    values=[_oracle_at(polynomial,lo),_oracle_at(polynomial,hi)]
    if c and lo < -b/(2*c) < hi:
        values.append(_oracle_at(polynomial,-b/(2*c)))
    return min(values),max(values)
def _oracle_source_hull(curve,left,right):
    a,b,c=_oracle_coefficients(curve); l,r=Fraction(float(left)),Fraction(float(right))
    p0=tuple(x+y*l+z*l*l for x,y,z in zip(a,b,c))
    p1=tuple(x+y*r+z*r*r for x,y,z in zip(a,b,c))
    middle=tuple(x+(r-l)*(y+2*z*l)/2 for x,y,z in zip(p0,b,c))
    return p0,middle,p1

def _oracle_inventory(search,endpoint_bands,*,maximum_cells):
    """Return every root-free/uncertainty interval and exact visited debit.

    Classification uses exact full-interval _oracle_extrema, never midpoint signs.
    The complete target endpoint-root _oracle_inventory is charged before any None.
    This diagnostic does not return a mechanical partition or omit strips.
    """
    if maximum_cells < 1:
        raise MemoryError('endpoint _oracle_inventory search cell budget exhausted')
    targets=search.targets; source=search.source
    shared=_oracle_rows(targets[0])[2]==_oracle_rows(targets[1])[0]
    result=dict(shared_vertex=shared,visited=1,root_free=[],root_strips=[],convexity=[])
    if not shared:
        result['status']='UNSHARED_VERTEX'; return result
    hull=_oracle_source_hull(source,0.,1.)
    for target in targets:
        q,b,c=_oracle_coefficients(target); acc=tuple(2*x for x in c)
        aa=_oracle_dot(acc,acc); va=_oracle_dot(b,acc)
        t=max(Fraction(0),min(Fraction(1),-va/aa)) if aa else Fraction(0)
        bound=min(_oracle_dot(b,b)+_oracle_dot(_oracle_sub(q,p),acc)+3*va*t+Fraction(3,2)*aa*t*t for p in hull)
        result['convexity'].append(str(bound))
        if bound <= 0:
            result['status']='NONCONVEX'; return result
    p,b,c=_oracle_coefficients(source); j=_oracle_rows(targets[0])[2]
    left_b,left_c=_oracle_coefficients(targets[0])[1:]
    left_t=tuple(x+2*y for x,y in zip(left_b,left_c))
    right_t=_oracle_coefficients(targets[1])[1]
    phis=tuple((_oracle_dot(_oracle_sub(j,p),t),-_oracle_dot(b,t),-_oracle_dot(c,t)) for t in (left_t,right_t))
    result['phi_coefficients']=[[str(value) for value in phi] for phi in phis]
    bands=[]
    for target in targets:
        remaining=maximum_cells-result['visited']
        if remaining < 1:
            raise MemoryError('endpoint _oracle_inventory search cell budget exhausted')
        found,visited=endpoint_bands(source,target,0.,1.,maximum_cells=remaining)
        if visited > remaining:
            raise MemoryError('endpoint _oracle_inventory returned excess search cells')
        result['visited']+=visited; bands.extend(found)
    merged=[]
    for lo,hi in sorted(bands):
        if merged and lo <= merged[-1][1]:
            merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
        else:
            merged.append((lo,hi))
    def label(lo,hi):
        (lmin,lmax),(rmin,rmax)=[_oracle_extrema(phi,lo,hi) for phi in phis]
        if lmax < 0 < rmin:
            kind='SHARED_VERTEX_CONE_PLATEAU'
        elif lmin > 0 and rmin > 0:
            kind='LEFT_DOMINATES_RIGHT_VERTEX'
        elif lmax < 0 and rmax < 0:
            kind='RIGHT_DOMINATES_LEFT_VERTEX'
        elif lmin > 0 > rmax:
            kind='GENUINE_TWO_INTERIOR'
        else:
            kind='UNCERTAIN_OR_ZERO_KKT'
        return dict(interval=[lo,hi],label=kind,phi_left_range=[str(lmin),str(lmax)],
            phi_right_range=[str(rmin),str(rmax)],endpoint_transition_only=(lmax<=0 or rmin>=0))
    position=0.
    for lo,hi in merged:
        if lo > position:
            result['root_free'].append(label(position,lo))
        result['root_strips'].append(label(lo,hi)); position=hi
    if position < 1.:
        result['root_free'].append(label(position,1.))
    result['status']='COMPLETE_EXACT_KKT_INVENTORY'
    return result
