"""Second derivative certificates for the actual retained exponential path."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from anyfesolver.control import CancellationToken
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver.general_contact.native_retained_pair_static import _swept_distance_chord_lower
from test_general_contact_native_curved_gap import _independent_gap


def _owner(*, control=True, curved=True, separation=.101):
    build = runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_native_retained_whole_pair.py'))['analysis']
    return build(physical_fibres=True, plastic_fibres=True, targets=(.05,),
                 line_stiffness_n_per_m2=4000., gap_tolerance_m=5e-5,
                 gap_step_control=control, reference_rise_m=.005 if curved else 0.,
                 certify_curved_ownership=curved, separation=separation)


def _arguments(owner, state):
    a,b = owner.indices
    return (state.positions[owner.context.nodes[a]], state.position_low[owner.context.nodes[a]], state.cell_rotations[a],
            state.positions[owner.context.nodes[b]], state.position_low[owner.context.nodes[b]], state.cell_rotations[b])


def test_analytic_norm_chord_encloses_hidden_collision_and_feature_switch():
    # Independent norm oracle; moving translations cross the nearest direction.
    for x0, velocity in ((-.1, .2), (-.01, .02), (-.2, .4)):
        d = lambda t: np.hypot(x0+velocity*t, .10001)
        g0,g1 = d(0)-.1,d(1)-.1
        lower = _swept_distance_chord_lower(g0,g1,1.,abs(velocity),0.,.1,1e-13)
        exact = minimize_scalar(d, bounds=(0,1), method='bounded').fun-.1
        assert lower <= exact
        assert _swept_distance_chord_lower(g0,g1,1.,abs(velocity),0.,.001,1e-13) == -np.inf
    # The infimum of two smooth distances may switch features; both remain
    # enclosed by the common global-endpoint certificate.
    branches = (lambda t: np.hypot(.03*t, .1), lambda t: np.hypot(.03*(1-t), .1))
    lower = _swept_distance_chord_lower(0.,0.,1.,.03,0.,.1,1e-13)
    assert all(lower <= min(f(t) for f in branches)-.1 for t in np.linspace(0,1,41))


def test_internal_rotation_hidden_collision_uses_actual_path():
    owner = _owner(separation=.1023)
    start = owner.initial.mechanical
    frames = start.cell_rotations.copy()
    a = owner.indices[1]
    for cell in (0,1): frames[a,cell] = rotation(np.array([0.,0.,np.pi-.8]))
    start = replace(start, cell_rotations=frames)
    increment = np.zeros(owner.context.count)
    first = owner.context.nodal_count+24*a
    increment[first+2] = increment[first+5] = 1.6
    end = owner.context.advance(start, increment)
    middle = owner.context.advance(start, .5*increment)
    g0,g1,gm = (owner._physical_gap(state) for state in (start,end,middle))
    assert min(g0,g1) > 0 and gm < -owner.gap_tolerance
    assert _independent_gap(owner,_arguments(owner,middle)) < -owner.gap_tolerance
    speed,acceleration = owner._swept_motion_bounds(start,increment)
    assert speed > 0 and acceleration > 0
    assert _swept_distance_chord_lower(g0,g1,1.,speed,acceleration,.1,1e-13) <= gm
    assert not owner._swept_admissible(start,increment,end,g0,g1)


def test_actual_caller_certifies_retained_motion_without_intermediate_queries(monkeypatch):
    owner = _owner()
    start = owner.initial.mechanical
    increment = np.zeros(owner.context.count)
    first = owner.context.nodal_count+24*owner.indices[1]
    increment[first+2] = increment[first+5] = .01
    end = owner.context.advance(start,increment)
    g0,g1 = owner._physical_gap(start),owner._physical_gap(end)
    # Choose a conservative near-tolerance endpoint lower: old Lipschitz
    # certificate needs subdivision, whereas the distance curvature proves it.
    lower = -owner.gap_tolerance+2e-6
    assert lower <= min(g0,g1)
    speed,acceleration = owner._swept_motion_bounds(start,increment)
    assert lower-.5*speed < -owner.gap_tolerance
    calls=[]
    monkeypatch.setattr(owner,'_swept_gap',lambda state: calls.append(state) or owner._physical_gap(state))
    assert owner._swept_admissible(start,increment,end,lower,lower)
    assert not calls
    # Actual physical geometry on the path also remains well clear.
    assert _independent_gap(owner,_arguments(owner,owner.context.advance(start,.5*increment))) > 0


def test_budget_cancellation_mutation_and_policy_identity(monkeypatch):
    owner = _owner(control=False)
    assert owner._identity() == owner.identity
    # All curved owners bind the changed policy, irrespective of secant flag.
    from anyfesolver._ge_beam3_p5_seeded.core import sha
    old = dict(profile=owner.profile,native_model=owner.context.identity,
               native_support_sha256=__import__('hashlib').sha256(owner.context.support_identity).hexdigest(),
               contact=owner.contact.identity,gap_tolerance_m=owner.gap_tolerance,
               equilibrium_tolerance=owner.equilibrium_tolerance,
               maximum_swept_intervals=owner.maximum_swept_intervals)
    assert sha(old) != owner.identity
    raw = owner._checkpoint(())
    assert owner._restore(raw,sha256(raw).hexdigest())[0].completed_targets == 0
    plain = _owner(control=False,curved=False)
    old.update(profile=plain.profile,native_model=plain.context.identity,
               native_support_sha256=__import__('hashlib').sha256(plain.context.support_identity).hexdigest(),
               contact=plain.contact.identity,gap_tolerance_m=plain.gap_tolerance,
               equilibrium_tolerance=plain.equilibrium_tolerance,
               maximum_swept_intervals=plain.maximum_swept_intervals)
    assert sha(old) == plain.identity
    with pytest.raises(ValueError,match='binding'):
        plain._restore(raw,sha256(raw).hexdigest())
    start = owner.initial.mechanical
    step = np.zeros(owner.context.count)
    with pytest.raises(ValueError,match='finite'):
        owner._swept_motion_bounds(start,np.full(owner.context.count,np.nan))
    owner.maximum_swept_intervals = 1
    owner.identity = owner._identity()
    calls=[]
    monkeypatch.setattr(owner,'_swept_gap',lambda state: calls.append(state) or -owner.gap_tolerance)
    moving = step.copy()
    first = owner.context.nodal_count+24*owner.indices[0]
    moving[first+2] = moving[first+5] = 2.
    end=owner.context.advance(start,moving)
    assert not owner._swept_admissible(start,moving,end,-owner.gap_tolerance,-owner.gap_tolerance)
    assert len(calls)==1
    token=CancellationToken()
    with owner.context.operation(token):
        owner.context.activate_cancellation(token);token.cancel()
        with pytest.raises(Exception,match='cancel'):
            owner._swept_admissible(start,step,start,0.,0.)
    owner.maximum_swept_intervals += 1
    with pytest.raises(ValueError,match='changed'):
        owner._swept_admissible(start,step,start,0.,0.)
