"""Reusable whole-band stationary bounds versus scalar and slab controls."""
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver.general_contact.native_surface_front import (
    stationary_front_angular_cover,prepare_stationary_band,stationary_front_partition_prepared,
    stationary_front_partition)
from anyfesolver.general_contact.native_surface_support import SupportIntervalBudget
from test_general_contact_native_surface_local_fronts import _mixed


@pytest.mark.parametrize('height,rotation',[(.035,1.1),(.035,.821),(.03000000001,1.1)])
def test_prepared_fronts_contain_analytic_roots_and_match_slab_oracle(height,rotation):
    field=_mixed(height=height,rotation=rotation);angles=field.patch.rule.definition['angle_range']
    budget=SupportIntervalBudget(128)
    cover=stationary_front_angular_cover(field,0,0,angles,budget=budget)
    prepared=prepare_stationary_band(field,cover,budget=budget)
    before=budget.intervals
    assert prepared.prepared_intervals==len(cover)+2
    for angle in np.linspace(*angles,7):
        result=stationary_front_partition_prepared(field,prepared,float(angle),maximum_axial_uncertainty=1e-7)
        oracle=stationary_front_partition(field,0,0,float(angle),maximum_axial_uncertainty=1e-7,angular_cover=cover)
        roots=result[2];assert len(roots)==len(oracle[2])==1
        exact_time=(.8-(angle-1.5*np.pi))/rotation
        exact_xi=(.03-height)/.01
        assert roots[0].time_bracket[0]<=exact_time<=roots[0].time_bracket[1]
        assert roots[0].axial_interval[0]<=exact_xi<=roots[0].axial_interval[1]
        assert roots[0].fraction==pytest.approx(oracle[2][0].fraction,abs=2e-13)
        assert max(roots[0].axial_interval[0],oracle[2][0].axial_interval[0])<=min(roots[0].axial_interval[1],oracle[2][0].axial_interval[1])
        for endpoint in result[3]:
            theta=-.8+rotation*endpoint.fraction+angle-1.5*np.pi
            exact=(.03*np.cos(theta)-height)/.01
            assert endpoint.axial_interval[0]<=exact<=endpoint.axial_interval[1]
        assert result[1]==0 and budget.intervals==before


def test_prepared_band_identity_resource_and_uncertainty_guards():
    field=_mixed();angles=field.patch.rule.definition['angle_range'];budget=SupportIntervalBudget(128)
    cover=stationary_front_angular_cover(field,0,0,angles,budget=budget)
    with pytest.raises(MemoryError,match='cap'):
        prepare_stationary_band(field,cover,budget=SupportIntervalBudget(1))
    prepared=prepare_stationary_band(field,cover,budget=budget)
    with pytest.raises(ValueError,match='identity/angle'):
        stationary_front_partition_prepared(field,prepared,angles[1]+.001,maximum_axial_uncertainty=1e-7)
    with pytest.raises(ValueError,match='uncertainty'):
        stationary_front_partition_prepared(field,prepared,sum(angles)/2,maximum_axial_uncertainty=1e-16)
    field.maximum_intervals+=1
    with pytest.raises(ValueError,match='identity/angle'):
        stationary_front_partition_prepared(field,prepared,sum(angles)/2,maximum_axial_uncertainty=1e-7)


@pytest.mark.parametrize('height,slope,open_interval,contact_interval',[
    (.035,.01,(-.49,-.2),(-.62,-.55)),(.025,-.01,(-.8,-.51),(-.45,-.38))])
def test_prepared_open_interval_rejects_hidden_pulse(height,slope,open_interval,contact_interval):
    from anyfesolver.general_contact.native_surface_front import prepared_front_open_interval
    field=_mixed(height=height,slope=slope);angles=field.patch.rule.definition['angle_range']
    budget=SupportIntervalBudget(128)
    cover=stationary_front_angular_cover(field,0,0,angles,budget=budget)
    prepared=prepare_stationary_band(field,cover,budget=budget)
    partition=stationary_front_partition_prepared(field,prepared,sum(angles)/2,maximum_axial_uncertainty=1e-7)
    assert prepared_front_open_interval(prepared,partition,*open_interval)
    assert not prepared_front_open_interval(prepared,partition,*contact_interval)
    assert not prepared_front_open_interval(prepared,partition,-.500000001,-.499999999)


def test_prepared_bounds_match_moving_target_slab_control():
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment
    original=_mixed();s=np.zeros(18);s[6:9]=(.02,.01,.001);s[12:15]=(-.01,.02,-.001)
    old=original.segments[0]
    field=NativeSurfaceHistoryField(original.patch,original.initial,
        (NativeSurfaceMotionSegment(old.translations,old.angular,s),))
    angles=field.patch.rule.definition['angle_range'];budget=SupportIntervalBudget(128)
    cover=stationary_front_angular_cover(field,0,0,angles,budget=budget)
    prepared=prepare_stationary_band(field,cover,budget=budget)
    for angle in np.linspace(*angles,5):
        actual=stationary_front_partition_prepared(field,prepared,float(angle),maximum_axial_uncertainty=1e-7)
        reference=stationary_front_partition(field,0,0,float(angle),maximum_axial_uncertainty=1e-7,angular_cover=cover)
        assert len(actual[2])==len(reference[2])==1
        for a,b in zip((*actual[2],*actual[3]),(*reference[2],*reference[3])):
            assert max(a.axial_interval[0],b.axial_interval[0])<=min(a.axial_interval[1],b.axial_interval[1])
            assert a.fraction==pytest.approx(b.fraction,abs=2e-13)


def test_roundoff_propagation_covers_large_coordinate_cross_dot_amplification():
    from decimal import Decimal,localcontext
    from anyfesolver.general_contact.native_surface_front import _front_coefficient_roundoff
    # Admissible edge perturbations of size gamma*X can amplify through C dot x.
    # Unit edges, large shared coordinates, long relative arm: max(X,R) is not X*R.
    with localcontext() as context:
        context.prec=60
        delta=Decimal(1024)*Decimal(2)**-52*Decimal('1e12')
        difference=((Decimal(1)+delta)**2-1)*Decimal('1e6')
    bound=_front_coefficient_roundoff(1e12,1.,1.,1.,1.,2.,2.,1e6,1.,1.,1.,0.,.03)
    assert float(difference)>1e5
    assert bound>=float(difference)


@pytest.mark.parametrize('height,slope',[(.035,.01),(.025,-.01)])
def test_open_interior_starts_at_outward_extremum_edge(height,slope):
    from anyfesolver.general_contact.native_surface_front import prepared_front_open_interval
    field=_mixed(height=height,slope=slope);angles=field.patch.rule.definition['angle_range']
    budget=SupportIntervalBudget(128)
    cover=stationary_front_angular_cover(field,0,0,angles,budget=budget)
    prepared=prepare_stationary_band(field,cover,budget=budget)
    partition=stationary_front_partition_prepared(field,prepared,sum(angles)/2,maximum_axial_uncertainty=1e-7)
    extrema=(*partition[2],*partition[3])
    if slope>0:
        bound=max(e.axial_interval[1] for e in extrema)
        assert prepared_front_open_interval(prepared,partition,bound,-.2)
        xi=(bound-.2)/2
    else:
        bound=min(e.axial_interval[0] for e in extrema)
        assert prepared_front_open_interval(prepared,partition,-.8,bound)
        xi=(-.8+bound)/2
    # Independently, the minimum over the actual full rotation is at theta=0.
    assert height+slope*xi-.03>0.
