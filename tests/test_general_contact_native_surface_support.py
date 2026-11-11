"""Whole material-panel separation, including hidden temporal contact."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField, NativeSurfaceMotionSegment
from anyfesolver.general_contact.native_surface_patch import NativeS3MaterialSurfacePatch
from anyfesolver.general_contact.native_surface_path import material_path_pose
from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
from anyfesolver.general_contact.native_surface_support import (
    SupportIntervalBudget, history_endpoint_support_certified, panel_open_over_history,
    history_target_ownership_certified, history_endpoint_support_cover_certified,
)
from anyfesolver.general_contact.s3_friction_point import S3SkinMaterialPoint
from test_general_contact_native_surface_s3_friction import _case


def _field(*, height=.05, rotation=0., segments=(), maximum_intervals=256, narrow_rim=False, activation_policy='transverse_events'):
    _, point_pair, initial, _ = _case()
    angles = (1.5*np.pi-.02, 1.5*np.pi+.02)
    rule = NativeRoundSurfaceRule(7, point_pair.source.reference, .03,
        xi_range=(-.8, -.2), angle_range=angles, axial_order=1, angular_order=2)
    positions = initial.positions.copy()
    positions[:, 2] = -.04+height
    if narrow_rim:
        positions[:, 0] = -.019+.004*point_pair.source.reference.coordinates[:, 0]
    rotations = initial.cell_rotations.copy()
    rotations[0] = Rotation.from_rotvec((rotation, 0., 0.)).as_matrix()
    initial = replace(initial, positions=positions, cell_rotations=rotations)
    adapter = point_pair.target.adapter
    corners = adapter.current_positions(initial.shell_displacement)
    edges = np.column_stack((corners[1]-corners[0], corners[2]-corners[0]))
    targets = []
    for sample in rule.samples:
        point = sample.station.evaluate(initial.positions, initial.position_low, initial.cell_rotations).surface_position
        uv = np.linalg.lstsq(edges, point-corners[0], rcond=None)[0]
        targets.append(S3SkinMaterialPoint(adapter, (1-uv.sum(), *uv)))
    patch = NativeS3MaterialSurfacePatch(rule, targets, normal_stiffness_Pa_per_m=1000.,
        friction_coefficient=.3, tangential_stiffness_Pa_per_m=1000.,
        activation_policy=activation_policy, gap_check='straight_patch')
    return NativeSurfaceHistoryField(patch, initial, segments, maximum_intervals=maximum_intervals)


def _certify(field, maximum_intervals=256):
    return panel_open_over_history(field, 0, (-.8, -.2),
                                  field.patch.rule.definition['angle_range'], maximum_intervals=maximum_intervals)


def test_static_whole_panel_and_all_history_segments():
    assert _certify(_field())
    zero = NativeSurfaceMotionSegment(np.zeros((3, 3)), np.zeros((2, 3)), np.zeros(18))
    assert _certify(_field(segments=(zero, zero)), maximum_intervals=2)
    assert not _certify(_field(segments=(zero, zero)), maximum_intervals=1)


def test_hidden_contact_pulse_between_open_rotational_endpoints():
    segment = NativeSurfaceMotionSegment(np.zeros((3, 3)), np.array(((1.6, 0., 0.), (0., 0., 0.))), np.zeros(18))
    field = _field(height=.025, rotation=-.8, segments=(segment,))
    # For the centre angle, g(alpha)=.025-.03*cos(-.8+1.6*alpha).
    # The entire narrow arc is open at both endpoints and closed at midtime.
    assert .025-.03*np.cos(.8-.02) > 0.
    assert .025-.03 < 0.
    assert not _certify(field)
    # An endpoint-only rule would accept; both static endpoint fields do.
    assert _certify(_field(height=.025, rotation=-.8))
    assert _certify(_field(height=.025, rotation=.8))


def test_positive_rotating_source_with_finite_affine_target_deformation():
    shell = np.zeros(18)
    shell[8] = .01
    shell[14] = -.006
    shell[12] = .025
    segment = NativeSurfaceMotionSegment(np.tile((.015, 0., -.002), (3, 1)),
        np.array(((.2, .1, -.05), (0., 0., 0.))), shell)
    field = _field(height=.08, segments=(segment,))
    assert _certify(field)


def test_target_affine_collapse_is_not_certified_from_valid_endpoints():
    field = _field(height=.08)
    corners = field.patch.target.adapter.current_positions(field.initial.shell_displacement)
    shell = np.zeros(18)
    for i in range(3):
        shell[6*i:6*i+2] = -2*corners[i, :2]
    segment = NativeSurfaceMotionSegment(np.zeros((3, 3)), np.zeros((2, 3)), shell)
    field = _field(height=.08, segments=(segment,))
    assert not _certify(field)


def test_support_checks_domain_binding_and_captured_path_cap():
    field = _field()
    with pytest.raises(ValueError, match='half-cell'):
        panel_open_over_history(field, 0, (-.8, .1), (4.7, 4.72), maximum_intervals=2)
    with pytest.raises(ValueError, match='interval cap'):
        _certify(field, maximum_intervals=0)
    segment = NativeSurfaceMotionSegment(np.zeros((3, 3)), np.array(((1., 0., 0.), (0., 0., 0.))), np.zeros(18))
    limited = _field(height=.031, segments=(segment,), maximum_intervals=1)
    assert not _certify(limited, maximum_intervals=256)
    field.patch.pairs[0].source.radius *= 1.01
    with pytest.raises(ValueError, match='definition changed'):
        _certify(field)


def test_shared_support_budget_is_not_reset_between_panels():
    zero = NativeSurfaceMotionSegment(np.zeros((3, 3)), np.zeros((2, 3)), np.zeros(18))
    field = _field(segments=(zero,))
    budget = SupportIntervalBudget(1)
    angles = field.patch.rule.definition['angle_range']
    assert panel_open_over_history(field, 0, (-.8, -.5), angles,
                                  maximum_intervals=256, budget=budget)
    assert budget.intervals == 1
    assert not panel_open_over_history(field, 0, (-.5, -.2), angles,
                                      maximum_intervals=256, budget=budget)
    assert budget.intervals == 1


def test_near_front_affine_slide_needs_only_one_interval():
    segment = NativeSurfaceMotionSegment(np.tile((.01, 0., 0.), (3, 1)),
                                        np.zeros((2, 3)), np.zeros(18))
    field = _field(height=.0300000001, segments=(segment,))
    budget = SupportIntervalBudget(1)
    assert panel_open_over_history(field, 0, (-.8, -.2),
        field.patch.rule.definition['angle_range'], maximum_intervals=1, budget=budget)
    assert budget.intervals == 1


def test_endpoint_support_admits_affine_spatially_varying_slopes():
    translations = np.zeros((3, 3))
    translations[0, 2] = -.03
    translations[1, 2] = .03
    segment = NativeSurfaceMotionSegment(translations, np.zeros((2, 3)), np.zeros(18))
    field = _field(segments=(segment,))
    # Some material points approach the plane and others recede; each has an
    # affine gap, so uniform whole-patch derivative sign is unnecessary.
    assert history_endpoint_support_certified(field)


def test_endpoint_support_admits_uniform_monotone_finite_rotation():
    segment = NativeSurfaceMotionSegment(np.tile((0., 0., -.02), (3, 1)),
        np.array(((.2, 0., 0.), (0., 0., 0.))), np.zeros(18))
    field = _field(height=.08, segments=(segment,))
    assert history_endpoint_support_certified(field)


def test_endpoint_support_rejects_hidden_pulse_and_target_collapse():
    pulse = NativeSurfaceMotionSegment(np.zeros((3, 3)),
        np.array(((1.6, 0., 0.), (0., 0., 0.))), np.zeros(18))
    field = _field(height=.025, rotation=-.8, segments=(pulse,))
    assert not history_endpoint_support_certified(field)
    flat = _field(height=.08)
    corners = flat.patch.target.adapter.current_positions(flat.initial.shell_displacement)
    shell = np.zeros(18)
    for i in range(3):
        shell[6*i:6*i+2] = -2*corners[i, :2]
    collapse = NativeSurfaceMotionSegment(np.zeros((3, 3)), np.zeros((2, 3)), shell)
    assert not history_endpoint_support_certified(_field(height=.08, segments=(collapse,)))


def test_swept_ownership_static_translation_and_shared_budget():
    assert history_target_ownership_certified(_field(), budget=SupportIntervalBudget(1))
    segment = NativeSurfaceMotionSegment(np.tile((.01, 0., 0.), (3, 1)),
                                        np.zeros((2, 3)), np.zeros(18))
    field = _field(segments=(segment,))
    budget = SupportIntervalBudget(1)
    assert history_target_ownership_certified(field, budget=budget)
    assert budget.intervals == 1
    assert not history_target_ownership_certified(field, budget=budget)
    assert budget.intervals == 1


def test_open_rotational_rim_excursion_is_not_endpoint_owned():
    segment = NativeSurfaceMotionSegment(np.zeros((3, 3)),
        np.array(((2., 0., 0.), (0., 0., 0.))), np.zeros(18))
    field = _field(height=.08, rotation=.6, narrow_rim=True, segments=(segment,))
    # Triangle's diagonal is x+y=0. Source y=.03*sin(theta) at the
    # centre angle, while x is in [-.0222,-.0198]. The endpoints are
    # interior, but the midpoint theta=1.6 places points across the rim.
    assert -.0198+.03*np.sin(.6+.02) < 0.
    assert -.0198+.03*np.sin(2.6-.02) < 0.
    assert -.0222+.03*np.sin(1.6) > 0.
    assert _certify(field)  # Gap separation alone is insufficient ownership.
    assert history_target_ownership_certified(_field(height=.08, rotation=.6, narrow_rim=True),
                                            budget=SupportIntervalBudget(1))
    assert history_target_ownership_certified(_field(height=.08, rotation=2.6, narrow_rim=True),
                                            budget=SupportIntervalBudget(1))
    assert not history_target_ownership_certified(field, budget=SupportIntervalBudget(256))


def test_swept_ownership_with_finite_target_deformation():
    shell = np.zeros(18)
    shell[8] = .01
    shell[14] = -.006
    shell[12] = .025
    segment = NativeSurfaceMotionSegment(np.tile((.015, 0., -.002), (3, 1)),
        np.array(((.2, .1, -.05), (0., 0., 0.))), shell)
    field = _field(height=.08, segments=(segment,))
    assert history_target_ownership_certified(field, budget=SupportIntervalBudget(256))

def test_regional_cover_rejects_hidden_pulse_with_bounded_work():
    segment=NativeSurfaceMotionSegment(np.zeros((3,3)),np.array(((1.6,0.,0.),(0.,0.,0.))),np.zeros(18))
    field=_field(height=.025,rotation=-.8,segments=(segment,))
    budget=SupportIntervalBudget(128)
    assert not history_endpoint_support_cover_certified(field,budget=budget)
    assert budget.intervals<=128
    tiny=SupportIntervalBudget(1)
    assert not history_endpoint_support_cover_certified(field,budget=tiny)
    assert tiny.intervals<=1


def test_regional_endpoint_certificate_rejects_outside_or_empty_selection():
    field=_field()
    for bounds in ((-.9,-.2),(-.3,-.3),(float('nan'),-.2)):
        with pytest.raises(ValueError,match='rectangle outside'):
            history_endpoint_support_certified(field,xi_bounds=bounds)
    with pytest.raises(ValueError,match='rectangle outside'):
        history_endpoint_support_certified(field,angle_bounds=(0.,1.))


def test_curvature_certificate_resolves_shallow_rotating_open_arc_in_one_interval():
    # Independent exact minimum: height-radius > 0 for every rotation angle.
    # The first-order speed enclosure R*|omega|/2 exceeds this shallow gap.
    radius=.03;height=radius+1e-5;omega=.04
    assert height-radius>0 and radius*omega/2>height-radius
    segment=NativeSurfaceMotionSegment(np.zeros((3,3)),
        np.array(((omega,0.,0.),(0.,0.,0.))),np.zeros(18))
    field=_field(height=height,rotation=-omega/2,segments=(segment,))
    budget=SupportIntervalBudget(1)
    args=(field,0,(-.8,-.2),field.patch.rule.definition['angle_range'])
    assert panel_open_over_history(*args,maximum_intervals=1,budget=budget)
    assert budget.intervals==1
    assert not panel_open_over_history(*args,maximum_intervals=1,budget=budget)


def test_panel_support_distinguishes_persistent_contact_from_hidden_pulse():
    from anyfesolver.general_contact.native_surface_support import classify_history_panel,panel_closed_over_history
    translations=np.zeros((3,3));translations[:,0]=.0001
    translations[:,2]=(-.05,-.019,-.019)
    press=NativeSurfaceMotionSegment(translations,np.zeros((2,3)),np.zeros(18))
    pulse=NativeSurfaceMotionSegment(np.zeros((3,3)),np.array(((1.6,0.,0.),(0.,0.,0.))),np.zeros(18))
    field=_field(height=.05,rotation=-.8,segments=(press,pulse))
    angles=field.patch.rule.definition['angle_range']
    # g=.05+dz(xi)-.03*cos(-.8+1.6*t+angle_offset).
    # Left material remains closed; right material is open at both pulse ends.
    assert .05-.05*.8-.019*.2-.03*np.cos(.82)<0
    assert .05-.05*.2-.019*.8-.03*np.cos(.78)>0
    assert .05-.05*.2-.019*.8-.03<0
    persistent=classify_history_panel(field,0,(-.8,-.7),angles,budget=SupportIntervalBudget(128))
    assert persistent.status=='resolved'
    hidden=classify_history_panel(field,0,(-.25,-.2),angles,budget=SupportIntervalBudget(128))
    # The complete transverse time cover may admit this pulse, but must
    # label it nonzero/event-bearing rather than empty or persistently closed.
    assert hidden.status=='resolved' and hidden.event_covered
    assert not panel_closed_over_history(field,0,(-.25,-.2),angles,
        maximum_intervals=128,budget=SupportIntervalBudget(128),segment_index=1)
    mixed=classify_history_panel(field,0,(-.8,-.2),angles,budget=SupportIntervalBudget(128))
    assert mixed.status=='resolved' and mixed.event_covered


def test_unresolved_nonzero_panel_cannot_be_accepted(monkeypatch):
    from anyfesolver.general_contact.native_surface_support import HistoryPanelSupport
    import anyfesolver.general_contact.native_surface_support as support
    from anyfesolver.general_contact.native_surface_friction_integral import integrate_material_friction
    from test_general_contact_native_surface_friction_integral import _field as integral_field,_settings
    field=integral_field();calls=[]
    def unresolved(field,cell,xis,angles,*,budget):
        calls.append((cell,*xis,*angles))
        return HistoryPanelSupport('unresolved',calls[-1],'unsampled historical pulse')
    monkeypatch.setattr(support,'classify_history_panel',unresolved)
    def forbidden_query(*args,**kwargs):
        raise AssertionError('unresolved panel must refine before using quadrature agreement')
    monkeypatch.setattr(field,'query',forbidden_query)
    with pytest.raises(MemoryError,match='panel cap'):
        integrate_material_friction(field,replace(_settings(),maximum_panels=4))
    assert len(calls)==4
    # Independent counterexample: both Gauss rules integrate background1 but
    # miss a positive pulse strictly between their nearest nodes about zero.
    from numpy.polynomial.legendre import leggauss
    estimates=[]
    for order in (4,8):
        nodes,weights=leggauss(order)
        estimates.append(float(weights@(1.+(np.abs(nodes)<.01))))
    assert estimates[0]==pytest.approx(estimates[1],abs=1e-14)
    assert 2.02-estimates[0]>.019


@pytest.mark.parametrize('cell,xis,angles',[(1,(-.8,-.2),(4.70,4.72)),
    (0,(-.8,.2),(4.70,4.72)),(0,(-.8,-.2),(float('nan'),4.72)),
    (2,(-.8,-.2),(4.70,4.72)),(0,(-.2,-.2),(4.70,4.72))])
def test_panel_classifier_rejects_mismatched_cell_rectangle(cell,xis,angles):
    from anyfesolver.general_contact.native_surface_support import classify_history_panel
    with pytest.raises(ValueError,match='half-cell'):
        classify_history_panel(_field(),cell,xis,angles,budget=SupportIntervalBudget(128))


def test_support_exhaustion_preserves_region_and_named_recovery_contract(monkeypatch):
    import anyfesolver.general_contact.native_surface_support as support
    from anyfesolver.general_contact.native_surface_friction_integral import (
        integrate_material_friction, NativeFrictionIntegrationLimit)
    from test_general_contact_native_surface_friction_integral import _field as integral_field,_settings
    field=integral_field();regions=[]
    message='native friction aggregate support interval cap exhausted'
    def classify(field,cell,xis,angles,*,budget):
        regions.append((cell,*xis,*angles))
        if len(regions)==1:
            return support.HistoryPanelSupport('unresolved',regions[-1],'hidden history control')
        raise MemoryError(message)
    monkeypatch.setattr(support,'classify_history_panel',classify)
    def forbidden_query(*args,**kwargs):
        raise AssertionError('support failure must precede material queries')
    monkeypatch.setattr(field,'query',forbidden_query)
    with pytest.raises(NativeFrictionIntegrationLimit) as caught:
        integrate_material_friction(field,_settings())
    assert isinstance(caught.value,MemoryError)
    assert str(caught.value)==message
    diagnostics=caught.value.integration_diagnostics
    assert diagnostics['queries']==0
    assert diagnostics['support_panels_visited']==2
    assert diagnostics['support_region']==regions[1]
    assert diagnostics['quadrature_panel']==regions[0]
    assert diagnostics['last_unresolved']==dict(region=regions[0],reason='hidden history control')


def test_closed_endpoints_allow_transverse_release_and_recontact_cover(monkeypatch):
    # Independent scalar fixture, identical at every material point:
    # g(t) = -t**2+t-3/16. Closed endpoints; simple roots at1/4,3/4.
    # Positive/negative root rates are both necessary in a complete cover.
    from types import SimpleNamespace
    import anyfesolver.general_contact.native_surface_support as module
    def enclosure(left,right):
        values=[-left*left+left-3/16,-right*right+right-3/16]
        if left<=.5<=right:values.append(1/16)
        return SimpleNamespace(gap_lower=min(values),gap_upper=max(values),
            zero_derivative_lower=1-2*right,zero_derivative_upper=1-2*left,
            spatial_derivative_width=0.,temporal_derivative_width=2*(right-left))
    monkeypatch.setattr(module,'segment_gap_slab_enclosure',lambda *args:enclosure)
    # No scalar geometry/point sampling is used to certify the time cover.
    budget=module.SupportIntervalBudget(32)
    field=SimpleNamespace(maximum_intervals=32)
    assert module.panel_transverse_event_cover(field,0,(0.,1.),(0.,1.),0,
        budget=budget)=='events'
    assert 1<budget.intervals<=32


def test_zero_rate_tangency_is_not_admitted_by_transverse_cover(monkeypatch):
    from types import SimpleNamespace
    import anyfesolver.general_contact.native_surface_support as module
    # g(t)=(t-1/2)**2 has a double root; enclosing q includes0 on every
    # root slab. Exact spatial certainty does not certify transverse motion.
    def enclosure(left,right):
        values=[(left-.5)**2,(right-.5)**2]
        lower=0. if left<=.5<=right else min(values)
        return SimpleNamespace(gap_lower=lower,gap_upper=max(values),
            zero_derivative_lower=2*left-1,zero_derivative_upper=2*right-1,
            spatial_derivative_width=0.,temporal_derivative_width=2*(right-left))
    monkeypatch.setattr(module,'segment_gap_slab_enclosure',lambda *args:enclosure)
    budget=module.SupportIntervalBudget(16)
    field=SimpleNamespace(maximum_intervals=16)
    assert module.panel_transverse_event_cover(field,0,(0.,1.),(0.,1.),0,
        budget=budget) is None
    assert budget.intervals<=16
