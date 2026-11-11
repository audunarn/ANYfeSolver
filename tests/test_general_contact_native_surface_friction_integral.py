"""Pressure-aware replay integration against independent scalar surface work."""
from math import acos,sin

import numpy as np
import pytest

from anyfesolver.general_contact.native_surface_friction_integral import (
    NativeSurfaceFrictionIntegrationSettings,integrate_material_friction)
from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField
from test_general_contact_native_surface_history import _origin,_segment
from test_general_contact_native_surface_patch import _patch


def _field():
    owner,bridge=_patch(activation_policy='transverse_events',gap_check='straight_patch')
    initial=_origin(owner)
    return NativeSurfaceHistoryField(bridge.pair,initial,(_segment(.0001,-.00201),))


def _settings():
    return NativeSurfaceFrictionIntegrationSettings(1e-10,1e-5,1e-12,1.,maximum_queries=2048)


def test_narrow_first_touch_surface_force_matches_independent_integral():
    field=_field();result=integrate_material_friction(field,_settings())
    # Straight cylinder above z=-.051, radius .049, translated down .00201.
    # Stick traction = kt*dx/d * penetration after transverse first touch.
    radius=.049;depth=.00201;initial_gap=.002;slide=.0001
    theta=acos(1-(depth-initial_gap)/radius)
    penetration_integral=2*((depth-initial_gap-radius)*theta+radius*sin(theta))
    reference=field.patch.rule.samples[0].station.reference
    jacobian=radius*np.linalg.norm(reference.coordinates[2]-reference.coordinates[1])
    exact=1000*slide/depth*jacobian*.6*penetration_integral
    force=sum(result.residual[6*i] for i in range(3))
    assert force==pytest.approx(exact,rel=2e-3,abs=1e-12)
    assert result.stored_energy_j>0 and result.cumulative_dissipation_j==0
    assert result.dissipated_increment_j==0 and result.released_storage_j==0
    # The original 2x2 rule misses this narrow contact band completely.
    assert all(abs(p.source.angle-np.pi)>theta for p in field.patch.pairs)
    np.testing.assert_allclose(sum((result.residual[6*i:6*i+3] for i in range(3)),np.zeros(3))+
        sum((result.residual[42+6*i:45+6*i] for i in range(3)),np.zeros(3)),0.,atol=1e-12,rtol=0.)
    assert result.diagnostics['maximum_scaled_error_estimate']<=1.


def test_integral_enforces_aggregate_query_and_memory_caps():
    from dataclasses import replace
    field=_field()
    for settings in (replace(_settings(),maximum_queries=1),replace(_settings(),maximum_cache_bytes=1)):
        with pytest.raises(MemoryError,match='query/cache cap'):
            integrate_material_friction(field,settings)


def test_integral_rejects_hidden_temporal_pulse_before_quadrature():
    from dataclasses import replace
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_friction_integral import NativeFrictionIntegrationLimit
    from anyfesolver._ge_beam3_p5_seeded.core import sha
    from test_general_contact_native_surface_support import _field as support_field
    segment=NativeSurfaceMotionSegment(np.zeros((3,3)),np.array(((1.6,0.,0.),(0.,0.,0.))),np.zeros(18))
    field=support_field(height=.025,rotation=-.8,segments=(segment,))
    # The declared arc opens at both endpoints and closes at midtime.
    # Unresolved support must fail explicitly before any quadrature query.
    assert .025-.03*np.cos(.8-.02)>0.
    assert .025-.03<0.
    initial_definition=sha(field._definition())
    # The complete sequence support proof now resolves this fixture at 128
    # intervals. Preserve the pre-quadrature refusal contract with an explicitly
    # insufficient proof budget; the resolved pulse work is checked below.
    settings=replace(_settings(),maximum_support_intervals=1)
    with pytest.raises(NativeFrictionIntegrationLimit,match='aggregate support interval cap') as caught:
        integrate_material_friction(field,settings)
    diagnostics=caught.value.integration_diagnostics
    assert diagnostics['queries']==0
    assert diagnostics['support_intervals']==settings.maximum_support_intervals
    assert diagnostics['last_unresolved'] is not None
    assert sha(field._definition())==initial_definition


def test_resolved_hidden_temporal_pulse_preserves_independent_release_work():
    from math import cos, fsum
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceMotionSegment
    from test_general_contact_native_surface_support import _field as support_field
    segment=NativeSurfaceMotionSegment(np.zeros((3,3)),np.array(((1.6,0.,0.),(0.,0.,0.))),np.zeros(18))
    field=support_field(height=.025,rotation=-.8,segments=(segment,))
    result=integrate_material_friction(field,_settings())
    # Independently restate the declared 32-substep Coulomb release law.
    theta=acos(.025/.03); elastic=0.; previous=.03*sin(-theta); work=[]
    for index in range(1,32):
        angle=-theta+2*theta*index/32; current=.03*sin(angle)
        trial=elastic+current-previous; cap=.3*1000*(.03*cos(angle)-.025)
        next_elastic=min(trial,cap/1000); work.append(cap*(trial-next_elastic))
        elastic=next_elastic; previous=current
    area=.03*.6*.04
    assert result.dissipated_increment_j==pytest.approx(area*fsum(work),rel=1e-8,abs=1e-14)
    assert result.cumulative_dissipation_j==pytest.approx(result.dissipated_increment_j,rel=1e-12)
    assert result.released_storage_j==pytest.approx(area*.5*1000*elastic**2,rel=1e-8,abs=1e-14)
    assert result.dissipated_increment_j>0
    np.testing.assert_array_equal(result.residual,np.zeros(60))
    assert result.diagnostics['maximum_scaled_error_estimate']<=1.


def test_integral_support_budget_is_shared_across_panels():
    from dataclasses import replace
    with pytest.raises(MemoryError,match='aggregate support interval cap'):
        integrate_material_friction(_field(),replace(_settings(),maximum_support_intervals=1))


def test_integrated_first_touch_tangent_tracks_moving_pressure_front():
    field=_field();value=integrate_material_friction(field,_settings())
    direction=np.zeros(60);direction[[0,6,12]]=1.;direction[[2,8,14]]=.1
    derivative=value.tangent@direction
    for step in (1e-7,2e-8):
        results=[]
        for sign in (-1.,1.):
            path=(_segment(.0001+sign*step,-.00201+sign*step*.1),)
            candidate=NativeSurfaceHistoryField(field.patch,field.initial,path)
            results.append(integrate_material_friction(candidate,_settings()).residual)
        numerical=(results[1]-results[0])/(2*step)
        assert np.linalg.norm(derivative-numerical)/np.linalg.norm(numerical)<1e-5


def test_integrated_release_retains_prior_dissipation_and_storage_release():
    from scipy.integrate import quad
    field=_field()
    slide=.01;depth=.00201;radius=.049;mu=.3;normal=1e4;tangent=1000.;steps=32
    field=NativeSurfaceHistoryField(field.patch,field.initial,(_segment(slide,-depth),_segment(0.,.004)))
    result=integrate_material_friction(field,_settings())
    theta=acos(1-(depth-.002)/radius)
    integral=quad(lambda a:(depth-.002-radius*(1-np.cos(a)))**2,-theta,theta,
                  epsabs=1e-20,epsrel=1e-11)[0]
    reference=field.patch.rule.samples[0].station.reference
    integral*=radius*np.linalg.norm(reference.coordinates[2]-reference.coordinates[1])*.6
    initial_energy=.5*(mu*normal)**2/tangent*integral
    press_dissipation=mu*normal*(slide/depth-mu*normal/tangent)*integral
    # This is the independently evaluated declared32-substep release law,
    # not a claim of converged physical unloading work.
    assert result.dissipated_increment_j==pytest.approx(initial_energy*(1-1/steps),rel=2e-3,abs=1e-14)
    assert result.released_storage_j==pytest.approx(initial_energy/steps**2,rel=2e-3,abs=1e-15)
    assert result.cumulative_dissipation_j==pytest.approx(
        press_dissipation+initial_energy*(1-1/steps),rel=2e-3,abs=1e-14)
    np.testing.assert_array_equal(result.residual,0.)
    np.testing.assert_array_equal(result.tangent,0.)
    assert result.stored_energy_j==0.
