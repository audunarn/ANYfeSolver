"""Independent analytic oracle and focused checks for certified sub-band
axial-front refinement.

The oracle is a scalar linear-axial/sinusoidal model g(x,a)=h+s*x+r*sin(a):
the fixture's chord is exactly linear in the reference parameter and the
material circle is exactly sinusoidal in the declared angle, verified against
the independent point helper. The oracle computes analytic angular-union
measures and exact-region works by its own arcsin breakpoints and Gauss
quadrature. It never calls the production bound kernel or any production
root/width/station helper; production is exercised only through evaluate()
and the public result. Focused checks: conditional partition entry, full
root union tiling, representation pad retention, finite cap refusal, and
unchanged passing evaluations. The captured cold 8e7 axial front checks
the one-sided station-empty enclosure: independent analytic arcs and
pad-depressed worst-case tangent integrals stay under the certified bound.
"""
from math import pi
import json,os
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from anyfesolver.general_contact.native_curved_geometry import curved_patch_geometry_control
from anyfesolver.general_contact.native_curved_plane_work import NativeCurvedMaterialPlaneWork
from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
from anyfesolver.general_contact.native_surface_normal import StraightNormalIntegrationSettings
from test_general_contact_native_retained_station import _reference
from test_general_contact_native_surface_bounds import _independent_point
from anyfesolver.general_contact.native_curved_s3_work import _S3CandidateIntegration
from anyfesolver.general_contact.native_surface_quadrature import _prepare_reference_area
from test_general_contact_native_curved_s3_static import _analysis

RADIUS=.03
SLOPE=1e-4
ROOT=-.6999999
HEIGHT=RADIUS-SLOPE*ROOT
LIMITS=(1e-10,1e-8,1e-6)


def _fixture(stiffness,intervals=64,slope=SLOPE,root=ROOT):
    height=RADIUS-slope*root
    ref=_reference(True)
    rule=NativeRoundSurfaceRule(7,ref,RADIUS,xi_range=(-.7,-.6),angle_range=(4.2,5.2),
        axial_order=1,angular_order=1)
    law=NativeCurvedMaterialPlaneWork(rule,plane_origin_m=(0.,0.,0.),
        plane_normal=(0.,0.,1.),normal_stiffness_Pa_per_m=stiffness,
        integration_settings=StraightNormalIntegrationSettings(1e-10,1e-8,1e-6,intervals,1.))
    p=ref.coordinates.copy();p[:,2]=(height-slope,height,height+slope)
    return law,p,np.zeros((3,3)),np.tile(np.eye(3),(2,1,1))


def _gap(x,a,slope=SLOPE,height=HEIGHT):
    return height+slope*x+RADIUS*np.sin(a)


def _area(x,a):
    J=np.sqrt(1+.16*x*x)
    return RADIUS*(J+.4*RADIUS*np.cos(a)/(J*J))


def _sin_breaks(offset,al,ar):
    """Analytic roots of offset+RADIUS*sin(a)=0 inside [al,ar], with ends."""
    if offset<=-RADIUS or offset>=RADIUS:return [al,ar]
    base=np.arcsin(-offset/RADIUS)
    roots={al,ar}
    for candidate in (base,np.pi-base):
        for turn in (-1,0,1):
            angle=candidate+2*np.pi*turn
            if al<angle<ar:roots.add(angle)
    return sorted(roots)


def _arc_measure(offset,al,ar):
    """Oracle measure of {a in [al,ar]: offset+RADIUS*sin(a)<0}, analytic."""
    if offset<=-RADIUS:return ar-al
    if offset>=RADIUS:return 0.
    breaks=_sin_breaks(offset,al,ar)
    return sum((right-left for left,right in zip(breaks[:-1],breaks[1:])
        if offset+RADIUS*np.sin((left+right)/2)<0),0.)


def _union_measure(lo,hi,al,ar,slope=SLOPE,height=HEIGHT):
    """Exact active angular union over x in [lo,hi]; the chord is linear."""
    worst=height+slope*(lo if slope>0 else hi)
    return _arc_measure(worst,al,ar)


def _region_work(law,lo,hi,order=48,slope=SLOPE,height=HEIGHT):
    """Oracle work over the exact active region of x in [lo,hi]."""
    al,ar=law.rule.definition['angle_range'];k=law.stiffness
    nodes,weights=leggauss(order)
    breaks=_sin_breaks(height+slope*lo,al,ar)
    total=0.
    for left,right in zip(breaks[:-1],breaks[1:]):
        for qa,wa in zip(nodes,weights):
            angle=(left+right)/2+(right-left)*qa/2
            stop=(-RADIUS*np.sin(angle)-height)/slope
            upper=min(hi,stop)
            if upper<=lo:continue
            for qx,wx in zip(nodes,weights):
                x=(lo+upper)/2+(upper-lo)*qx/2
                gap=_gap(x,angle,slope,height)
                if gap<0:
                    total+=.5*k*gap*gap*_area(x,angle)*wx*wa*(upper-lo)*(right-left)/4
    return total


def _station_work(law,lo,hi,station,order=48,slope=SLOPE,height=HEIGHT):
    """Oracle work of the production scheme: x frozen at the station."""
    al,ar=law.rule.definition['angle_range'];k=law.stiffness
    nodes,weights=leggauss(order)
    breaks=_sin_breaks(height+slope*station,al,ar)
    total=0.
    for left,right in zip(breaks[:-1],breaks[1:]):
        for qa,wa in zip(nodes,weights):
            angle=(left+right)/2+(right-left)*qa/2
            gap=_gap(station,angle,slope,height)
            if gap<0:
                total+=.5*k*gap*gap*_area(station,angle)*wa*(right-left)/2*(hi-lo)
    return total


def _spy(monkeypatch):
    calls=[]
    original=NativeCurvedMaterialPlaneWork._certified_root_subbands
    def wrapper(self,*args,**kwargs):
        groups,attempt=original(self,*args,**kwargs)
        calls.append(dict(certified=groups is not None,attempt=attempt,
            subbands=[len(group) for group in groups] if groups is not None else None))
        return groups,attempt
    monkeypatch.setattr(NativeCurvedMaterialPlaneWork,'_certified_root_subbands',wrapper)
    return calls


def test_oracle_scalar_model_matches_independent_points():
    law,p,low,U=_fixture(10000.)
    al,ar=law.rule.definition['angle_range'];lo,hi=law.rule.definition['xi_range']
    worst=0.
    for x in np.linspace(lo,hi,41):
        for a in np.linspace(al,ar,41):
            point=_independent_point(x,a,RADIUS,p,low,U,False)
            worst=max(worst,abs(point[2]-_gap(x,a)))
    assert worst<1e-15
    # The analytic root of the angular-minimum feature is inside the domain.
    assert lo<ROOT<hi
    assert _gap(ROOT,np.pi*3/2)==pytest.approx(0.,abs=1e-17)
    assert _gap(ROOT-.5,3*np.pi/2)<0<_gap(ROOT+.5,3*np.pi/2)


def test_conditional_partition_certifies_rejected_bracket_full_union_and_pad(monkeypatch):
    law,p,low,U=_fixture(8e7)
    calls=_spy(monkeypatch)
    value=law.evaluate(p,low,U)
    monkeypatch.undo()
    # Conditional entry: the whole-bracket enclosure was rejected first.
    assert len(calls)==1 and calls[0]['certified'] and calls[0]['subbands']==[16]
    assert calls[0]['attempt']['interval_visits']<=calls[0]['attempt']['interval_limit']
    rows=value.axial_root_brackets
    assert len(rows)>1
    al,ar=law.rule.definition['angle_range']
    pads={row['representation_gap_m'] for row in rows}
    features={tuple(row['features']) for row in rows}
    cells={row['cell'] for row in rows}
    assert len(pads)==1 and min(pads)>0.  # representation pad retained everywhere
    assert len(features)==1 and len(cells)==1  # feature identity inherited
    rows=sorted(rows,key=lambda row:row['lower'])
    assert rows[0]['lower']<ROOT<rows[-1]['upper']  # true analytic root enclosed
    for left,right in zip(rows[:-1],rows[1:]):
        assert left['upper']==right['lower']  # exact tiling, no gap or overlap
    parent_union=_union_measure(rows[0]['lower'],rows[-1]['upper'],al,ar)
    unions=[]
    for row in rows:
        lo,hi=row['lower'],row['upper']
        union=_union_measure(lo,hi,al,ar)
        unions.append(union)
        # Angular-union oracle: certified width dominates the exact union.
        assert row['active_angular_width_bound']>=union
        # Work-bound oracle: certified energy uncertainty dominates the exact
        # true-region vs representative-station work difference.
        true_work=_region_work(law,lo,hi)
        station_work=_station_work(law,lo,hi,row['representative_xi'])
        assert abs(true_work-station_work)<=row['error_bounds'][0]
    # Full root union: the partition's largest sub-band union is the parent's.
    assert max(unions)==pytest.approx(parent_union,abs=1e-15,rel=1e-12)
    assert value.estimated_errors[2]<LIMITS[2]
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path+'-front-refinement.json').write_text(json.dumps(dict(
        subband_rows=len(rows),attempt=calls[0]['attempt'],
        estimated_errors=value.estimated_errors,
        certified_union_domination=float(min(
            row['active_angular_width_bound']-u for row,u in zip(rows,unions)))),indent=2))


def test_passing_path_never_enters_subband_branch_and_replays_unchanged(monkeypatch):
    law,p,low,U=_fixture(10000.)
    calls=_spy(monkeypatch)
    first=law.evaluate(p,low,U)
    second=law.evaluate(p,low,U)
    monkeypatch.undo()
    assert calls==[]  # conditional only on previous rejection
    assert first.normal_storage_j==second.normal_storage_j
    np.testing.assert_array_equal(first.residual,second.residual)
    np.testing.assert_array_equal(first.tangent,second.tangent)
    np.testing.assert_array_equal(first.source_force,second.source_force)
    np.testing.assert_array_equal(first.source_moment,second.source_moment)
    assert first.intervals==second.intervals
    assert first.integrand_evaluations==second.integrand_evaluations
    assert first.estimated_errors==second.estimated_errors
    assert first.axial_root_brackets==second.axial_root_brackets
    assert first.axial_isolation_intervals==second.axial_isolation_intervals
    # Whole brackets only: no sub-band rows, no subdivision marker keys.
    for row in first.axial_root_brackets:
        assert set(row)=={'cell','lower','upper','features','representation_gap_m',
            'active_angular_width_bound','angular_fallback','representative_xi','error_bounds'}


def test_finite_cap_refusal_keeps_explicit_non_success(monkeypatch):
    # Grazing floor above the unchanged tangent limit: even the full shared
    # budget cannot certify, so the existing refusal is raised explicitly,
    # with the attempt recorded and no cap increase.
    law,p,low,U=_fixture(8e7,slope=6e-5)
    calls=_spy(monkeypatch)
    with pytest.raises(ValueError,match='work uncertainty exceeds integration budget') as error:
        law.evaluate(p,low,U)
    monkeypatch.undo()
    diagnostics=error.value.integration_diagnostics
    assert len(calls)==1 and calls[0]['certified'] is False
    attempt=diagnostics['subband_isolation']
    assert attempt['interval_visits']<=attempt['interval_limit']==64
    assert attempt['summed_error_bounds'][2]>=LIMITS[2]
    assert diagnostics['error_bounds'][2]>=LIMITS[2]


def test_certified_root_subbands_direct_contract_guards(monkeypatch):
    law,p,low,U=_fixture(10000.)
    value=law.evaluate(p,low,U)
    rows=[dict(row) for row in value.axial_root_brackets]
    assert rows
    geometry=curved_patch_geometry_control(law.rule,p,low,U,law.origin,law.normal)
    al,ar=law.rule.definition['angle_range']
    # Empty bracket list: existing non-success, no evaluation spent.
    groups,attempt=law._certified_root_subbands(p,low,U,geometry,law.settings,
        LIMITS,dict(visited=1,limit=64),[],al,ar)
    assert groups is None and attempt==dict(subband_evaluations=0)
    # Nonfinite or negative enclosures are invalid, not certifiable.
    for bad in (-1.,np.nan,np.inf):
        broken=[dict(rows[0],error_bounds=tuple(bad*e for e in rows[0]['error_bounds']))]
        groups,attempt=law._certified_root_subbands(p,low,U,geometry,law.settings,
            LIMITS,dict(visited=1,limit=64),broken,al,ar)
        assert groups is None and attempt==dict(subband_evaluations=0)
    # Exhausted shared budget: no split, honest non-success within the cap.
    errors=np.asarray(rows[0]['error_bounds'],float)
    limits=tuple(errors*.75)  # above the representation floor, below the whole bracket
    budget=dict(visited=63,limit=64)
    groups,attempt=law._certified_root_subbands(p,low,U,geometry,law.settings,
        limits,budget,rows,al,ar)
    assert groups is None
    assert attempt['interval_visits']<=64 and attempt['subband_evaluations']==0
    assert np.allclose(attempt['summed_error_bounds'],errors)
    # Fresh budget: the same rejected bracket certifies by sub-band partition.
    budget=dict(visited=1,limit=64)
    groups,attempt=law._certified_root_subbands(p,low,U,geometry,law.settings,
        limits,budget,rows,al,ar)
    assert groups is not None and attempt['subband_evaluations']>0
    assert attempt['interval_visits']<=64
    nodes=[node for group in groups for node in group]
    assert len(nodes)>1
    assert all(np.all(np.isfinite(node['error'])) and np.all(node['error']>=0)
        for node in nodes)
    total=np.sum([node['error'] for node in nodes],axis=0)
    assert np.all(total<np.asarray(limits))
    ordered=sorted(nodes,key=lambda node:node['lo'])
    assert ordered[0]['lo']==rows[0]['lower'] and ordered[-1]['hi']==rows[0]['upper']
    for left,right in zip(ordered[:-1],ordered[1:]):
        assert left['hi']==right['lo']
    assert {node['pad'] for node in nodes}=={rows[0]['representation_gap_m']}
    assert {tuple(node['features']) for node in nodes}=={tuple(rows[0]['features'])}

COLD_HIGH=[[0.0,0.0,0.0],[0.9999999996523736,0.20000000000254345,-3.343695155835758e-05],
    [1.9999999979799468,2.4212776619950985e-10,-0.000107819173844755]]
COLD_LOW=[[0.0,0.0,0.0],[1.2323312853913734e-17,2.950698536392097e-18,9.396771758602394e-22],
    [-5.5357198746523576e-17,-1.6862690710004348e-26,-2.6734477397713854e-21]]
COLD_ROT=[[[0.9999999994187693,1.0289376067095029e-10,3.40948906514366e-05],
    [2.8635384935192563e-11,0.9999999999925588,-3.857737110528041e-06],
    [-3.4094890651579834e-05,3.857737109262124e-06,0.9999999994113281]],
    [[0.9999999975074936,8.31290179711732e-10,7.060462578644909e-05],
    [4.621822578131054e-10,0.99999999983219,-1.831994061869211e-05],
    [-7.06046257898301e-05,1.831994060566174e-05,0.9999999973396836]]]
COLD_CORNERS=[[-5.0,-5.0,-0.030100000000000002],[5.0,-5.0,-0.030100000000000002],
    [0.0,10.0,-0.030100000000000002]]
COLD_CELL=1;COLD_LOWER=0.8993388519106321;COLD_UPPER=0.8993388641284482
COLD_STATION=0.8993388549650861;COLD_PAD=4.722709568835759e-13
COLD_STATION_MINIMUM=2.361354813637968e-13
COLD_ONE_SIDED_TE=9.411214837583546e-07
COLD_WORST_TANGENT=5.957266511270955e-07


def _cold_candidate():
    owner=_analysis(physical_fibres=True,plastic_fibres=True,normal_stiffness_Pa_per_m=8e7,
        integration_budget_policy='certified_active')
    return _S3CandidateIntegration(owner.pairs[0].work,np.array(COLD_CORNERS,float))


def _cold_fit(candidate,x,high,low,U):
    """Independent analytic sinusoid; helper angle = production angle + pi/2."""
    radius=candidate.radius;normal=candidate.normal;origin=candidate.origin
    point=lambda angle:_independent_point(x,angle+pi/2,radius,high,low,U,False)
    p0,pp,ph=point(0.),point(pi),point(pi/2)
    centre=(p0+pp)/2
    return (float(normal@(centre-origin)),float(normal@((p0-pp)/2)),float(normal@(ph-centre)))


def _cold_arcs(A,B,C,threshold,al,ar):
    """Analytic arcs of A+B*cos a+C*sin a<threshold inside [al,ar]."""
    amplitude=float(np.hypot(B,C))
    if amplitude==0.:return [(al,ar)] if A<threshold else []
    m=(threshold-A)/amplitude
    if m>=1.:return [(al,ar)]
    if m<=-1.:return []
    phase=float(np.arctan2(C,B))+pi;half=pi-float(np.arccos(m))
    return [(max(phase-half+2*pi*turn,al),min(phase+half+2*pi*turn,ar))
            for turn in range(-2,3)
            if phase+half+2*pi*turn>al and phase-half+2*pi*turn<ar]


def test_captured_cold_front_evaluates_with_one_sided_station_enclosure():
    # The captured cold 8e7 front: the retained station's projected scalar
    # normal gap is positive on the whole declared arc, so the station
    # quadrature contributes exactly zero and the refined declared-arc
    # enclosure is one-sided; the previously refused pose now evaluates.
    candidate=_cold_candidate()
    result=candidate.evaluate(np.array(COLD_HIGH,float),np.array(COLD_LOW,float),
        np.array(COLD_ROT,float))
    rows=[row for row in result.axial_root_brackets
        if row['cell']==COLD_CELL and abs(row['lower']-COLD_LOWER)<1e-12]
    assert len(rows)==1
    row=rows[0]
    assert abs(row['upper']-COLD_UPPER)<1e-12
    assert row['enclosure_refinement']=='declared_material_arc'
    assert row['error_bounds'][2]==pytest.approx(COLD_ONE_SIDED_TE,rel=1e-9)
    assert row['error_bounds'][2]<LIMITS[2]
    assert result.estimated_errors[2]<LIMITS[2]


def test_captured_cold_front_one_sided_enclosure_dominates_independent_worst_case():
    candidate=_cold_candidate()
    high=np.array(COLD_HIGH,float);low=np.array(COLD_LOW,float);U=np.array(COLD_ROT,float)
    al,ar=candidate.rule.definition['angle_range']
    radius=candidate.radius;cell=COLD_CELL;ref=candidate.reference
    count=candidate._coordinate_count();mapping=candidate._local_mapping(cell,count)
    scale=candidate._coordinate_scale(count)
    # Independent zero-depth certification: the station's analytic angular
    # minimum is positive, so the exact station active set is empty and the
    # production station integral is exactly zero (the one-sided premise).
    A,B,C=_cold_fit(candidate,COLD_STATION,high,low,U)
    minimum=A-float(np.hypot(B,C))
    assert 0.<minimum<1e-12
    assert minimum==pytest.approx(COLD_STATION_MINIMUM,abs=1e-15)
    # Independent scalar geometry/arc integration with production derivative maps over the
    # band: analytic arcs and gaps from independent points. This is not an independent
    # force/tangent derivative oracle; existing directional checks cover those maps. The retained
    # moving-normal jets carry the geometry/normal/circular correlations.
    nodes,weights=leggauss(24)
    integral=np.zeros((2,count,count));xs=np.linspace(COLD_LOWER,COLD_UPPER,161)
    for i,x in enumerate(xs):
        A,B,C=_cold_fit(candidate,x,high,low,U)
        share=(.5 if i in (0,len(xs)-1) else 1.)*(COLD_UPPER-COLD_LOWER)/(len(xs)-1)
        t=x+1 if cell==0 else x
        tangent=np.array((1.,-.4*x,0.));tangent/=np.linalg.norm(tangent)
        frame=np.column_stack((tangent,(-tangent[1],tangent[0],0.),(0.,0.,1.)))
        lift=np.array((0.,.2*t*(1-t),0.))
        for region,threshold in enumerate((0.,COLD_PAD)):
            for left,right in _cold_arcs(A,B,C,threshold,al,ar):
                if right<=left:continue
                for qa,wa in zip(nodes,weights):
                    angle=(left+right)/2+(right-left)*qa/2
                    helper=angle+pi/2
                    arm=U[cell]@(lift+radius*(np.cos(helper)*frame[:,1]
                        +np.sin(helper)*frame[:,2]))
                    gap=A+B*np.cos(angle)+C*np.sin(angle)-threshold
                    jet=candidate._gap_work_jet(float(gap),high,low,cell,t,arm)
                    S=scale*(mapping.T@jet.gradient)
                    Hs=scale[:,None]*(mapping.T@jet.hessian@mapping)*scale[None,:]
                    J=_prepare_reference_area(ref,x,radius).area_jacobian(angle)
                    integral[region]+=candidate.stiffness*J*(np.outer(S,S)+gap*Hs)*wa*(right-left)/2*share
    worst=float(np.linalg.norm(integral[1],np.inf))
    true=float(np.linalg.norm(integral[0],np.inf))
    assert worst==pytest.approx(COLD_WORST_TANGENT,rel=5e-2)
    assert true<worst<=COLD_ONE_SIDED_TE<LIMITS[2]
