"""Independent polynomial and scalar mass/spring development oracles."""

from fractions import Fraction
import math

import numpy as np
import pytest

from anyfesolver.general_contact.event_subdivision import EventNeedsCutback
from anyfesolver.general_contact.plane_newmark_events import (
    branch_path_admissible, hermite_branch_extrema, localize_endpoint_event,
)


def extrema(first, last, v0, v1, h=1.):
    return hermite_branch_extrema([first], [last], [v0], [v1], h)


@pytest.mark.parametrize('first,last,v0,v1,h,low,high', [
    (2., 2., 0., 0., 1., 2., 2.),
    (-1., 2., 3., 3., 1., -1., 2.),
    (.1, .1, -1., 1., 1., -.15, .1),
    (0., 0., .5, .5, 1., -math.sqrt(3.)/36., math.sqrt(3.)/36.),
    (-.125, .125, .75, .75, 1., -.125, .125),
    (.25, .25, -.5, .5, 2., 0., .25),
])
def test_independent_linear_quadratic_cubic_extrema(first, last, v0, v1, h, low, high):
    bounds = extrema(first, last, v0, v1, h)
    # Analytical decimal literals need their own rounding allowance; the exact
    # represented cubic is independently checked with Fractions below.
    assert bounds.minimum_gaps[0] <= low+2.e-16
    assert bounds.maximum_gaps[0] >= high-2.e-16
    allowance = 2*np.spacing(max(1., abs(low), abs(high)))
    np.testing.assert_allclose(bounds.minimum_gaps, [low], rtol=0., atol=allowance)
    np.testing.assert_allclose(bounds.maximum_gaps, [high], rtol=0., atol=allowance)


def test_exact_rational_samples_are_inside_outward_enclosures():
    rng = np.random.default_rng(1064)
    first, last, rate0, rate1 = rng.normal(size=(4, 16))
    duration = .37
    bounds = hermite_branch_extrema(first, last, rate0, rate1, duration)
    h = Fraction(duration)
    for index in range(first.size):
        g0, g1, v0, v1 = map(Fraction, (first[index], last[index], rate0[index], rate1[index]))
        for step in range(101):
            t = Fraction(step, 100)
            value = ((2*t**3-3*t**2+1)*g0+(t**3-2*t**2+t)*h*v0
                     +(-2*t**3+3*t**2)*g1+(t**3-t**2)*h*v1)
            assert Fraction(bounds.minimum_gaps[index]) <= value <= Fraction(bounds.maximum_gaps[index])
    with pytest.raises(ValueError):
        bounds.minimum_gaps[0] = 0.


def test_near_double_stationary_root_and_small_gap_remain_enclosed():
    # Nearly monotone cubic; cancellation in a float derivative discriminant
    # must not omit the tiny stationary pair.
    perturbation = 2.**-52
    bounds = extrema(-.125, .125, .75+perturbation, .75+perturbation)
    assert bounds.minimum_gaps[0] < -.125
    assert bounds.maximum_gaps[0] > .125
    tiny = extrema(1.e-300, 1.e-300, -4.e-300, 4.e-300)
    assert tiny.minimum_gaps[0] <= 0. <= tiny.maximum_gaps[0]


def test_orientation_preserves_each_pressure_branch_and_detects_hidden_excursion():
    valid = branch_path_admissible([.2, -.2], [.1, -.1], [-.1, .1], [-.1, .1], 1.,
                                   active=[False, True], gap_tolerance=1.e-12)
    assert valid.admissible
    np.testing.assert_allclose(valid.oriented_minimum_gaps, [.1, .1], atol=1.e-15)
    hidden = branch_path_admissible([.1, -.1], [.1, -.1], [-1., 1.], [1., -1.], 1.,
                                    active=[False, True], gap_tolerance=1.e-12)
    assert not hidden.admissible
    assert np.all(hidden.oriented_minimum_gaps < -.14)
    wrong_phase = branch_path_admissible([.1], [.1], [0.], [0.], 1.,
                                         active=[True], gap_tolerance=1.e-12)
    assert not wrong_phase.admissible


@pytest.mark.parametrize('change', [
    {'start_gaps': []}, {'end_gaps': [1., 2.]}, {'start_gap_rates': [np.nan]},
    {'end_gap_rates': [np.inf]}, {'duration': 0.}, {'duration': True},
    {'duration': np.inf}, {'start_gaps': [[1.]]},
])
def test_invalid_hermite_data_rejected(change):
    arguments = dict(start_gaps=[1.], end_gaps=[2.], start_gap_rates=[0.],
                     end_gap_rates=[0.], duration=1.)
    arguments.update(change)
    with pytest.raises(ValueError):
        hermite_branch_extrema(**arguments)


def test_unrepresentable_finite_outward_bound_is_explicit():
    with pytest.raises(ValueError, match='represented finitely'):
        extrema(np.finfo(float).max, np.finfo(float).max, 0., 0.)


def localize(callback, active=(False,), required=(0,), **overrides):
    arguments = dict(minimum_dt=.01, time_tolerance=1.e-10,
                     gap_tolerance=1.e-10, maximum_evaluations=128)
    arguments.update(overrides)
    return localize_endpoint_event(callback, 1., active, required, **arguments)


def test_actual_scalar_free_mass_closure_uses_only_positive_endpoint_durations():
    evaluated = []
    def callback(h):
        evaluated.append(h)
        return [.3-h]
    event = localize(callback)
    assert abs(event.duration-.3) <= 1.e-10
    assert event.support_indices == (0,)
    assert evaluated[:2] == [.01, 1.]
    assert min(evaluated) >= .01
    assert event.endpoint_evaluations == len(evaluated) <= 128
    assert len(set(evaluated)) == len(evaluated)


def test_batched_front_shares_evaluations_and_retains_earliest_linear_crossing():
    roots=np.linspace(.3,.8,12)
    calls=[]
    def callback(h):
        calls.append(h)
        return roots-h
    with pytest.raises(EventNeedsCutback,match='budget'):
        localize(callback,active=(False,)*12,required=tuple(range(12)),maximum_evaluations=64)
    calls.clear()
    event=localize(callback,active=(False,)*12,required=tuple(range(12)),
                   maximum_evaluations=64,batch_front=True)
    assert abs(event.duration-roots[0])<=1.e-10
    assert event.support_indices==(0,)
    assert event.endpoint_evaluations==len(calls)<40
    assert len(set(calls))==len(calls)


def test_batched_front_keeps_ties_orientation_and_hidden_crossing_rejection():
    root=.3
    event=localize(lambda h:[root-h,h-root,.7-h],active=(False,True,False),
                   required=(0,1,2),batch_front=True)
    assert abs(event.duration-root)<=1.e-10
    assert event.support_indices==(0,1)
    with pytest.raises(EventNeedsCutback,match='no sign bracket'):
        localize(lambda h:[(h-.2)*(h-.8),.5-h],active=(False,False),
                 required=(0,1),batch_front=True)
    with pytest.raises(EventNeedsCutback,match='omit'):
        localize(lambda h:[.2-h,.6-h],active=(False,False),required=(1,),batch_front=True)


def test_batched_front_proposal_still_requires_complete_path_check():
    def polynomial(h):return -(h-.2)*(h-.4)*(h-.8)
    def derivative(h):return -((h-.4)*(h-.8)+(h-.2)*(h-.8)+(h-.2)*(h-.4))
    event=localize(lambda h:[polynomial(h),.75-h],active=(False,False),
                   required=(0,1),batch_front=True)
    bounds=branch_path_admissible([polynomial(0.),.75],
        [polynomial(event.duration),.75-event.duration],
        [derivative(0.),-1.],[derivative(event.duration),-1.],event.duration,
        active=[False,False],gap_tolerance=1.e-10)
    # An endpoint-family front alone may bracket a later root after reentry.
    if event.duration>.4:assert not bounds.admissible
    else:assert abs(event.duration-.2)<=1.e-10


def test_batched_front_selection_is_exact_boolean():
    with pytest.raises(ValueError,match='controls'):
        localize(lambda h:[.3-h],batch_front=1)


def test_independent_active_mass_spring_newmark_release():
    # m=2, k=8, g0=-.2, v0=1: a0=.8. Average acceleration
    # gives g(h)=(-.2+h+.2*h*h)/(1+h*h). Its positive root is exact.
    root = (-1.+math.sqrt(1.16))/.4
    event = localize(lambda h: [(-.2+h+.2*h*h)/(1.+h*h)], active=(True,))
    assert abs(event.duration-root) <= 1.e-10
    assert event.support_indices == (0,)


def test_earliest_candidate_across_all_required_modes():
    event = localize(lambda h: [(.6-h)/(1.+h*h), (.2-h)/(1.+2*h*h)],
                     active=(False, False), required=(0, 1))
    assert abs(event.duration-.2) <= 1.e-10
    assert event.support_indices == (1,)


def test_simultaneous_mixed_active_release_and_inactive_closure():
    event = localize(lambda h: [(.3-h)/(1.+2*h*h), -(.3-h)/(1.+h*h)],
                     active=(False, True), required=(0, 1))
    assert abs(event.duration-.3) <= 1.e-10
    assert event.support_indices == (0, 1)


def test_four_simultaneous_modes_share_actual_endpoint_queries():
    queried = []
    def callback(h):
        queried.append(h)
        return np.array([1., 2., -1., -2.])*(.3-h)
    event = localize(callback, active=(False, False, True, True),
                     required=(0, 1, 2, 3), maximum_evaluations=64)
    assert event.support_indices == (0, 1, 2, 3)
    assert event.endpoint_evaluations == len(queried) < 40
    assert len(set(queried)) == len(queried)


@pytest.mark.parametrize('callback,message', [
    (lambda h: [.005-h], 'minimum_dt'),
    (lambda h: [.01-h], 'minimum_dt'),
    (lambda h: [(h-.4)**2], 'no sign bracket'),
    (lambda h: [(h-.2)*(h-.8)], 'no sign bracket'),
    (lambda h: [1.-h], 'no sign bracket'),
])
def test_below_minimum_grazing_and_hidden_double_root_need_cutback(callback, message):
    with pytest.raises(EventNeedsCutback, match=message) as caught:
        localize(callback)
    assert caught.value.endpoint_evaluations == 2


def test_no_double_root_hiding_behind_another_mode_event():
    # The complete branch cubic already requires both modes to be examined.
    # A hidden first-mode excursion cannot be silently dropped in favor of .5.
    with pytest.raises(EventNeedsCutback, match='no sign bracket'):
        localize(lambda h: [(h-.2)*(h-.8), .5-h],
                 active=(False, False), required=(0, 1))


def test_endpoint_proposal_alone_cannot_hide_an_earlier_crossing_pair():
    # Three roots permit an endpoint sign bracket around a later root. No
    # callback-only bisection can certify the first root of an arbitrary family.
    # The obligatory complete represented cubic check detects the earlier pair.
    def polynomial(h):
        return -(h-.2)*(h-.4)*(h-.8)
    def derivative(h):
        return -((h-.4)*(h-.8)+(h-.2)*(h-.8)+(h-.2)*(h-.4))
    event = localize(lambda h: [polynomial(h)])
    path = branch_path_admissible([polynomial(0.)], [polynomial(event.duration)],
                                  [derivative(0.)], [derivative(event.duration)],
                                  event.duration, active=[False], gap_tolerance=1.e-10)
    if event.duration > .4:
        assert not path.admissible
        assert path.minimum_gaps[0] < -.004
    else:
        assert abs(event.duration-.2) <= 1.e-10


def test_omitted_obvious_sign_crossing_mode_needs_cutback():
    with pytest.raises(EventNeedsCutback, match='omit'):
        localize(lambda h: [.2-h, .6-h], active=(False, False), required=(1,))


def test_earlier_unlisted_mode_below_minimum_needs_cutback():
    with pytest.raises(EventNeedsCutback, match='minimum_dt'):
        localize(lambda h: [.005-h, .6-h], active=(False, False), required=(1,))


def test_bounded_budget_and_failing_nonfinite_endpoint_keep_counts():
    with pytest.raises(EventNeedsCutback, match='budget') as caught:
        localize(lambda h: [.3-h], maximum_evaluations=3)
    assert caught.value.endpoint_evaluations == 3
    with pytest.raises(EventNeedsCutback, match='invalid equilibrated') as caught:
        localize(lambda h: [np.nan])
    assert caught.value.endpoint_evaluations == 1


def test_cancel_before_endpoint_evaluation_is_not_swallowed():
    calls = []
    class Cancelled(RuntimeError):
        pass
    def cancel(stage):
        assert stage == 'plane Newmark endpoint event'
        raise Cancelled('owned cancellation')
    with pytest.raises(Cancelled):
        localize(lambda h: calls.append(h) or [.3-h], check_cancel=cancel)
    assert not calls


@pytest.mark.parametrize('changes', [
    {'minimum_dt': 1.}, {'time_tolerance': 0.}, {'gap_tolerance': 0.},
    {'maximum_evaluations': True}, {'maximum_evaluations': 1},
    {'active': (1,)}, {'required_indices': ()},
    {'required_indices': (0, 0)}, {'required_indices': (True,)},
    {'required_indices': (1,)},
])
def test_invalid_event_controls_rejected(changes):
    arguments = dict(endpoint_gaps=lambda h: [.3-h], duration=1., active=(False,),
                     required_indices=(0,), minimum_dt=.01, time_tolerance=1.e-10,
                     gap_tolerance=1.e-10, maximum_evaluations=128)
    arguments.update(changes)
    with pytest.raises(ValueError):
        localize_endpoint_event(**arguments)


def test_endpoint_candidate_failure_retains_cause_and_actual_query_count():
    def candidate(duration):
        raise ValueError('owned structural candidate failure')
    with pytest.raises(EventNeedsCutback,match='endpoint candidate failed: owned structural candidate failure') as info:
        localize(candidate)
    assert info.value.endpoint_evaluations==1
    assert isinstance(info.value.__cause__,ValueError)


def test_typed_endpoint_cutback_preserves_diagnostics_and_callback_units():
    def candidate(duration):
        raise EventNeedsCutback('nonconverged endpoint',endpoint_evaluations=9,
            diagnostics={'inner_stage_evaluations':20})
    with pytest.raises(EventNeedsCutback,match='nonconverged endpoint') as info:
        localize(candidate)
    assert info.value.endpoint_evaluations==1
    assert info.value.diagnostics=={'inner_stage_evaluations':20}
