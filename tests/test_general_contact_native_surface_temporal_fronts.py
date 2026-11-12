"""Complete time covers with whole-domain open portions, independent controls."""
from math import cos,sin
from dataclasses import replace
import numpy as np
import pytest
from scipy.optimize import brentq
from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment
from anyfesolver.general_contact.native_surface_front import (StationaryAngularCover,PreparedStationaryBand,
    IsolatedStationaryFront,stationary_front_angular_cover,prepare_stationary_band,
    stationary_front_partition_prepared,prepared_front_open_interval)
from anyfesolver.general_contact.native_surface_support import SupportIntervalBudget
from test_general_contact_native_surface_support import _field
from test_general_contact_native_surface_local_fronts import _mixed


def test_parallel_open_prefix_then_analytic_contact_pulse():
    original=_field(height=.031,rotation=-.8,activation_policy='transverse_event_sequence')
    displacement=np.zeros((3,3));displacement[:,2]=.01*original.initial.positions[:,0]
    motion=NativeSurfaceMotionSegment(displacement,np.array(((1.6,0.,0.),(0.,0.,0.))),np.zeros(18))
    field=NativeSurfaceHistoryField(original.patch,original.initial,(motion,))
    angles=field.patch.rule.definition['angle_range'];budget=SupportIntervalBudget(128)
    cover=stationary_front_angular_cover(field,0,0,angles,budget=budget,xi_bounds=(-.8,-.2))
    assert cover.open_time_slabs and cover.open_time_slabs[0][0]==0.
    assert len(cover)==1 and all(left>0 and bl>0 for left,_,bl,_ in cover.front_components)
    intervals=sorted([(a,b) for a,b,_ in cover.open_time_slabs]+[(a,b) for a,b,_,_ in cover.front_time_slabs])
    assert intervals[0][0]==0 and intervals[-1][1]==1
    assert all(x[1]==y[0] for x,y in zip(intervals,intervals[1:]))
    prepared=prepare_stationary_band(field,cover,budget=budget)
    angle=sum(angles)/2;delta=angle-1.5*np.pi
    partition=stationary_front_partition_prepared(field,prepared,angle,maximum_axial_uncertainty=1e-7)
    # Independent plane/circle path: A=.031-.03*cos(-.8+1.6t+delta), B=.01t.
    def n(t):
        phase=-.8+1.6*t+delta
        return .01*t*.03*1.6*sin(phase)-.01*(.031-.03*cos(phase))
    exact_time=brentq(n,.25,.8,xtol=1e-14)
    exact_xi=(.03*cos(-.8+1.6*exact_time+delta)-.031)/(.01*exact_time)
    root,=partition[2]
    assert root.time_bracket[0]<=exact_time<=root.time_bracket[1]
    assert root.axial_interval[0]<=exact_xi<=root.axial_interval[1]
    assert not prepared_front_open_interval(prepared,partition,-.8,-.4)  # Real hidden pulse.
    with pytest.raises(ValueError,match='outside bound axial domain'):
        prepared_front_open_interval(prepared,partition,-.9,-.8)


def _component_predicate_probe():
    # Independent data for H linear within two components, B=+1 then-1.
    # The intervening slab has a separate whole-domain positive-G certificate.
    components=((0.,.25,1.,1.),(.75,1.,-1.,-1.))
    cover=StationaryAngularCover('predicate',0,0,(1.,2.),(),(-.8,-.2),
        ((.25,.75,1.),),components,components)
    times=(0.,.25,.75,1.);hs=(-1.,-.9,-.5,-.4)
    endpoints=tuple(IsolatedStationaryFront(t,h,(t,t),(h,h),0.) for t,h in zip(times,hs))
    controls=tuple((t,t,(b,b,-.4,-.4,0.,0.),(0.,0.)) for t,b in zip(times,(1.,1.,-1.,-1.)))
    prepared=PreparedStationaryBand(cover,(),controls,4)
    return prepared,(tuple(zip(times,hs)),0,(),endpoints)


def test_opposite_sign_components_both_constrain_open_query_and_domain():
    prepared,partition=_component_predicate_probe()
    assert prepared_front_open_interval(prepared,partition,-.8,-.6)
    assert not prepared_front_open_interval(prepared,partition,-.8,-.45)  # Second component contacts.
    endpoints=list(partition[3]);endpoints[0]=replace(endpoints[0],axial_interval=(-.7,-.7))
    assert not prepared_front_open_interval(prepared,(*partition[:3],tuple(endpoints)),-.8,-.6)
    assert prepared_front_open_interval(prepared,partition,-.7,-.65)  # Child reuse inside fixed domain.
    with pytest.raises(ValueError,match='outside bound axial domain'):
        prepared_front_open_interval(prepared,partition,-.9,-.7)
    with pytest.raises(ValueError,match='complete prepared front extrema'):
        prepared_front_open_interval(prepared,(*partition[:3],partition[3][:2]),-.8,-.6)


def test_zero_b_during_nonpositive_gap_remains_resource_non_success():
    displacement=np.zeros((3,3));displacement[:,2]=-.01
    motion=NativeSurfaceMotionSegment(displacement,np.zeros((2,3)),np.zeros(18))
    field=_field(height=.035,segments=(motion,),activation_policy='transverse_event_sequence')
    budget=SupportIntervalBudget(16)
    with pytest.raises(MemoryError,match='interval cap'):
        stationary_front_angular_cover(field,0,0,field.patch.rule.definition['angle_range'],
            budget=budget,xi_bounds=(-.8,-.2))
    assert budget.intervals==16


def test_stationary_root_at_temporal_boundary_remains_explicitly_unresolved():
    field=_mixed(rotation=.8)
    with pytest.raises(ValueError,match='endpoint transition unresolved'):
        stationary_front_angular_cover(field,0,0,field.patch.rule.definition['angle_range'],
            budget=SupportIntervalBudget(128),xi_bounds=(-.8,-.2))


def test_saved_local_front_coefficient_discriminator(tmp_path):
    """Ten bounded coefficient controls; samples diagnose, never certify."""
    import json,os
    from pathlib import Path
    from dataclasses import asdict
    from anyfesolver.general_contact.native_surface_front import axial_contact_front,stationary_front_slab
    from test_general_contact_native_surface_local_fronts import _saved_local_front_field
    field,_,_=_saved_local_front_field();rows=[]
    for index in range(len(field.segments)):
        for left,right in ((0.,0.),(0.,.5),(.5,.5),(.5,1.),(1.,1.)):
            bounds,errors=stationary_front_slab(field,1,index,3.195,left,right,
                angle_bounds=(3.19,3.20),include_a=True,_return_error_controls=True)
            actual=axial_contact_front(field,1,index,(left+right)/2,3.195)
            gap_lower=float(np.nextafter(bounds[6]+min(b*x for b in bounds[:2] for x in (.72,.8)),-np.inf))
            rows.append(dict(segment=index,time=(left,right),bounds=bounds,
                coefficient_error=errors[0],numerator_error=errors[1],gap_lower=gap_lower,
                actual={**asdict(actual),'n':actual.stationary_numerator,'nd':actual.stationary_rate}))
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
    endpoints=[dict(segment=index,time=t,angle=angle,
        n=axial_contact_front(field,1,index,t,angle).stationary_numerator,
        nd=axial_contact_front(field,1,index,t,angle).stationary_rate)
        for index in range(len(field.segments)) for t in (0.,1.) for angle in (3.19,3.195,3.20)]
    (folder/'saved_local_coefficient_discriminator.json').write_text(
        json.dumps(dict(slabs=rows,endpoint_angle_samples=endpoints),indent=2),encoding='utf-8')
    assert len(rows)==10


@pytest.mark.parametrize('rate',[0.,1e-4])
def test_component_roundoff_preserves_rate_scale_and_decimal_product_errors(rate):
    from decimal import Decimal,localcontext
    from anyfesolver.general_contact.native_surface_front import _front_coefficient_roundoff
    # Independent scalar cross/dot analogue with two moving orthogonal edges.
    # Perturb their absolute positions and velocities before evaluating products.
    with localcontext() as context:
        context.prec=80
        gamma=Decimal(1024)*Decimal(2)**-52
        v=Decimal(str(rate));position_delta=gamma*Decimal('1e12')
        velocity_delta=gamma*v
        def coefficients(e,w,x,d,xp,dp):
            c=e*e;cd=2*e*w;cdd=2*w*w
            return (c*x,c*d,cd*x+c*xp,cd*d+c*dp,
                cdd*x+2*cd*xp,cdd*d+2*cd*dp)
        exact=coefficients(Decimal(1),v,Decimal('1e6'),Decimal(1),v,v)
        perturbed=coefficients(1+position_delta,v+velocity_delta,
            Decimal('1e6')+position_delta,1+position_delta,v+velocity_delta,v+velocity_delta)
        differences=tuple(float(abs(a-b)) for a,b in zip(exact,perturbed))
    errors=_front_coefficient_roundoff(1e12,rate,1.,rate,1.,2*rate,2*rate**2,
        1e6,1.,rate,rate,0.,.03,components=True)
    assert all(error>=difference for error,difference in zip(errors,differences))
    if rate==0.:
        assert all(error==np.nextafter(0.,np.inf) for error in errors[2:])
    else:
        assert errors[2]<2*rate*errors[0] and errors[4]<10*rate*errors[2]


def test_endpoint_angular_partition_encloses_analytic_boundary_without_dropping_bands():
    from anyfesolver.general_contact.native_surface_front_boundary import stationary_endpoint_angular_partition
    field=_mixed(rotation=.8);angles=field.patch.rule.definition['angle_range']
    result=stationary_endpoint_angular_partition(field,0,angles,budget=SupportIntervalBudget(16))
    # Independent plane/round-section path: N at final time is a positive
    # constant times sin(angle-3pi/2). Only the exact central angle is a zero.
    assert len(result.guards)==1
    left,right,index,time=result.guards[0]
    assert index==0 and time==1. and left<1.5*np.pi<right
    assert result.intervals_used==2
    assert [band[2] for band in result.bands]==['regular','unresolved','regular']
    assert result.bands[0][0]==angles[0] and result.bands[-1][1]==angles[1]
    assert all(a[1]==b[0] for a,b in zip(result.bands,result.bands[1:]))
    assert sum(b-a for a,b,_,_ in result.bands)==pytest.approx(angles[1]-angles[0],abs=1e-16)


def test_endpoint_angular_partition_retains_degenerate_and_tangent_polynomials():
    from anyfesolver.general_contact.native_surface_front_boundary import _polynomial_guards
    assert _polynomial_guards((0.,0.,0.),0.,0.,1.)==((0.,1.),)
    assert _polynomial_guards((-1.,1.,0.),1e-12,0.,1.)==((0.,1.),)
    assert not _polynomial_guards((2.,1.,0.),1e-12,0.,1.)
    # Two independent cos zeros, one at each selection endpoint. Both guard
    # bands are retained and clipped, rather than omitted as out-of-range.
    guards=_polynomial_guards((0.,1.,0.),1e-12,np.pi/2,1.5*np.pi)
    assert len(guards)==2 and guards[0][0]==np.pi/2 and guards[1][1]==1.5*np.pi


def test_near_tangent_simple_guards_contain_all_bounded_coefficient_perturbations():
    from itertools import product
    from math import acos,atan2
    from anyfesolver.general_contact.native_surface_front_boundary import _polynomial_guards
    coefficients=(-1.+1e-5,1.,0.);error=1e-7
    guards=_polynomial_guards(coefficients,error,-.1,.1)
    assert len(guards)==2 and all(a<b for a,b in guards)
    # Independent exact trig zero formula for27 coefficient perturbations.
    # Their combined uniform function error is at most the supplied allowance.
    for offsets in product((-error/3,0.,error/3),repeat=3):
        q0,qc,qs=(a+b for a,b in zip(coefficients,offsets))
        amplitude=np.hypot(qc,qs);phase=atan2(qs,qc);alpha=acos(-q0/amplitude)
        for root,(left,right) in zip((phase-alpha,phase+alpha),guards):
            assert left<root<right


def test_saved_endpoint_angular_partition_records_both_captured_transitions(tmp_path):
    import json,os
    from pathlib import Path
    from dataclasses import asdict
    from anyfesolver.general_contact.native_surface_front_boundary import stationary_endpoint_angular_partition
    from anyfesolver.general_contact.native_surface_front import axial_contact_front
    from test_general_contact_native_surface_local_fronts import _saved_local_front_field
    field,_,_=_saved_local_front_field();angles=field.patch.rule.definition['angle_range']
    result=stationary_endpoint_angular_partition(field,1,angles,budget=SupportIntervalBudget(16))
    assert len(result.guards)==2
    for index,time,(q0,qc,qs),error in result.endpoint_polynomials:
        for angle in (angles[0],sum(angles)/2,angles[1]):
            represented=q0+qc*cos(angle)+qs*sin(angle)
            actual=axial_contact_front(field,1,index,time,angle).stationary_numerator
            assert abs(represented-actual)<=error
    for left,right,index,time in result.guards:
        assert index==1
        # Representation consistency only: independent mechanics are checked
        # in the analytical fixture above. This is the saved actual geometry.
        root=brentq(lambda angle:axial_contact_front(field,1,index,time,angle).stationary_numerator,
            *angles,xtol=5e-15)
        assert left<root<right
    assert all(a[1]==b[0] for a,b in zip(result.bands,result.bands[1:]))
    assert sum(b-a for a,b,_,_ in result.bands)==pytest.approx(angles[1]-angles[0],abs=1e-16)
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
    (folder/'saved_endpoint_angular_partition.json').write_text(json.dumps(asdict(result),indent=2),encoding='utf-8')
    with pytest.raises(MemoryError,match='interval cap'):
        stationary_endpoint_angular_partition(field,1,angles,budget=SupportIntervalBudget(1))
    with pytest.raises(ValueError,match='outside material selection'):
        stationary_endpoint_angular_partition(field,1,(3.18,3.2),budget=SupportIntervalBudget(16))


def test_saved_regular_angular_bands_have_complete_zero_one_zero_branch_covers(tmp_path):
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_front_boundary import stationary_endpoint_angular_partition
    from test_general_contact_native_surface_local_fronts import _saved_local_front_field
    field,_,_=_saved_local_front_field();angles=field.patch.rule.definition['angle_range']
    budget=SupportIntervalBudget(128)
    partition=stationary_endpoint_angular_partition(field,1,angles,budget=budget)
    rows=[]
    for lower,upper,status,owners in partition.bands:
        if status=='unresolved':
            rows.append(dict(angles=(lower,upper),status=status,owners=owners))
            continue
        covers=[stationary_front_angular_cover(field,1,index,(lower,upper),budget=budget,
            xi_bounds=(.72,.80),endpoint_partition=partition) for index in range(len(field.segments))]
        rows.append(dict(angles=(lower,upper),status=status,branches=[len(c) for c in covers],
            temporal_slabs=[len(c.open_time_slabs)+len(c.front_time_slabs) for c in covers]))
    assert [sum(row['branches']) for row in rows if row['status']=='regular']==[0,1,0]
    assert len([row for row in rows if row['status']=='unresolved'])==2
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
    (folder/'saved_regular_angular_covers.json').write_text(
        json.dumps(dict(rows=rows,support_intervals=budget.intervals),indent=2),encoding='utf-8')


def test_variable_endpoint_branch_prepared_extrema_match_analytic_clamped_motion():
    from anyfesolver.general_contact.native_surface_front_boundary import stationary_endpoint_angular_partition
    field=_mixed(rotation=.8);angles=field.patch.rule.definition['angle_range'];budget=SupportIntervalBudget(128)
    endpoints=stationary_endpoint_angular_partition(field,0,angles,budget=budget)
    cover=stationary_front_angular_cover(field,0,0,angles,budget=budget,
        xi_bounds=(-.8,-.2),endpoint_partition=endpoints,_variable_endpoint_branches=True)
    assert cover.variable_branches==(True,)
    prepared=prepare_stationary_band(field,cover,budget=budget)
    for delta,kind in ((-.01,'clamped_endpoint'),(0.,'boundary_uncertain'),(.01,'stationary')):
        result=stationary_front_partition_prepared(field,prepared,1.5*np.pi+delta,
            maximum_axial_uncertainty=1e-7)
        root,=result[2]
        assert root.kind==kind
        exact_time=min(1.,1.-delta/.8)
        exact_xi=(.03*cos(-.8+.8*exact_time+delta)-.035)/.01
        assert root.time_bracket[0]<=exact_time<=root.time_bracket[1]
        assert root.axial_interval[0]<=exact_xi<=root.axial_interval[1]
        assert prepared_front_open_interval(prepared,result,-.499,-.2)
        assert not prepared_front_open_interval(prepared,result,-.8,-.6)
    # The strict historical route still rejects the true boundary crossing.
    with pytest.raises(ValueError,match='endpoint transition unresolved'):
        stationary_front_angular_cover(field,0,0,angles,budget=SupportIntervalBudget(128),
            xi_bounds=(-.8,-.2),endpoint_partition=endpoints)
    with pytest.raises(ValueError,match='require bound endpoint angular partition'):
        stationary_front_angular_cover(field,0,0,angles,budget=SupportIntervalBudget(128),
            xi_bounds=(-.8,-.2),_variable_endpoint_branches=True)


def test_saved_all_angular_bands_keep_bounded_potential_extrema(tmp_path):
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_front_boundary import stationary_endpoint_angular_partition
    from test_general_contact_native_surface_local_fronts import _saved_local_front_field
    field,_,_=_saved_local_front_field();angles=field.patch.rule.definition['angle_range']
    budget=SupportIntervalBudget(128)
    boundaries=stationary_endpoint_angular_partition(field,1,angles,budget=budget);rows=[]
    for lower,upper,status,_ in boundaries.bands:
        covers=[stationary_front_angular_cover(field,1,index,(lower,upper),budget=budget,
            xi_bounds=(.72,.80),endpoint_partition=boundaries,_variable_endpoint_branches=True)
            for index in range(len(field.segments))]
        prepared=[prepare_stationary_band(field,cover,budget=budget) for cover in covers]
        outputs=[stationary_front_partition_prepared(field,band,(lower+upper)/2,
            maximum_axial_uncertainty=1e-7) for band in prepared]
        rows.append(dict(angles=(lower,upper),status=status,branches=[len(c) for c in covers],
            kinds=[[root.kind for root in out[2]] for out in outputs],
            intervals=[[(r.axial_interval,r.time_bracket) for r in out[2]] for out in outputs]))
    assert [sum(row['branches']) for row in rows]==[0,1,1,1,0]
    assert all(len(row['kinds'][1])==1 for row in rows[1:4])
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
    (folder/'saved_all_angular_extrema.json').write_text(
        json.dumps(dict(rows=rows,support_intervals=budget.intervals,
            scope='geometry only; unresolved bands are not material-integral admission'),indent=2),encoding='utf-8')


def test_strict_clamp_uses_certified_derivative_sign_when_rounded_magnitudes_reverse(monkeypatch):
    import anyfesolver.general_contact.native_surface_front as front
    field=_mixed(rotation=.8);angles=field.patch.rule.definition['angle_range'];e=1e-12
    cover=StationaryAngularCover(field.identity,0,0,angles,((0.,1.),),
        xi_bounds=(-.8,-.2),front_components=((0.,1.,1.,1.),),variable_branches=(True,))
    bounds=(1.,1.,0.,4*e,.1*e,.1*e);errors=(1e-15,e)
    prepared=PreparedStationaryBand(cover,((0.,1.,bounds,errors),),
        ((0.,0.,bounds,errors),(1.,1.,bounds,errors)),3)
    # True N=(2.2+.1t)e is increasing. Rounded endpoint evaluations (3,2.5)e
    # reverse magnitudes while both errors remain <=e and signs stay reliable.
    def coefficient_probe(field,cell,index,t,angle):
        a=(2.2*t+.05*t*t)*e;ap=(3.-.5*t)*e
        return front.AxialContactFront(a,1.,ap,0.,.1*e,0.,1.)
    monkeypatch.setattr(front,'axial_contact_front',coefficient_probe)
    result=stationary_front_partition_prepared(field,prepared,sum(angles)/2,
        maximum_axial_uncertainty=1e-7)
    root,=result[2]
    assert root.kind=='clamped_endpoint' and root.fraction==0.
    assert root.axial_interval[0]<=0.<=root.axial_interval[1]
