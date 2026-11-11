"""Small two-element state/constraint tests; not a refinement campaign."""

import numpy as np
import pytest

from docs.reference_cases.ge_beam3_curved_p5_arch_refinement_case import make_beam
from docs.reference_cases.ge_beam3_curved_p5_displacement_control_probe import (
    DisplacementControlledAssemblyProbe,DisplacementControlError,
)
from docs.reference_cases.ge_beam3_curved_p5_history_path_probe import digest


def driver():
    beam,_=make_beam(2)
    return beam,DisplacementControlledAssemblyProbe(beam,node=2)


def test_target_reaction_full_equilibrium_and_replay():
    beam,control=driver();before=digest(beam.committed)
    trial=control.trial(.095)
    assert trial.assembly.positions[2,1]==.095
    assert trial.assembly.forces[2,1]<0 and trial.assembly.residual_norm<=1e-11
    assert np.count_nonzero(trial.assembly.forces)==1
    assert trial.reaction_derivative>0
    assert digest(beam.committed)==before and control.committed_model.committed.epoch==0
    response=trial.assembly.response
    state=control.commit(trial)
    assert state.epoch==1 and state.positions[2,1]==.095
    assert digest(control.committed_model.replay())==digest(response)
    assert np.array_equal(state.positions[[0,4]],beam.committed.positions[[0,4]])
    assert np.array_equal(state.rotations[[0,4]],beam.committed.rotations[[0,4]])


def test_reaction_derivative_matches_independent_small_target_difference():
    _,control=driver();trial=control.trial(.095);control.commit(trial)
    step=1e-6
    plus=control.trial(.095+step);fplus=plus.assembly.forces[2,1];control.discard(plus)
    minus=control.trial(.095-step);fminus=minus.assembly.forces[2,1];control.discard(minus)
    derivative=(fplus-fminus)/(2*step)
    assert abs(derivative-trial.reaction_derivative)/max(1.,abs(derivative))<1e-7


def test_discard_failure_and_pending_state_are_explicit():
    _,control=driver();before=digest(control.committed_model.committed)
    trial=control.trial(.095)
    with pytest.raises(DisplacementControlError,match='commit/discard'): control.trial(.09)
    control.discard(trial)
    assert digest(control.committed_model.committed)==before
    with pytest.raises(DisplacementControlError,match='owned'): control.commit(trial)
    with pytest.raises(DisplacementControlError): control.trial(.095,max_mixed_evaluations=0)
    assert digest(control.committed_model.committed)==before and control._pending is None


def test_mutated_trial_cannot_publish():
    _,control=driver();before=digest(control.committed_model.committed)
    trial=control.trial(.095)
    trial.assembly.positions.setflags(write=True);trial.assembly.positions[2,1]+=.001
    with pytest.raises(DisplacementControlError,match='altered'): control.commit(trial)
    assert digest(control.committed_model.committed)==before
    control.discard(trial)


@pytest.mark.parametrize('target,kwargs',[(np.nan,{}),(.05,{}),(.095,{'max_iterations':17}),
                                       (.095,{'max_mixed_evaluations':513}),(True,{})])
def test_invalid_or_unbounded_targets_are_rejected(target,kwargs):
    _,control=driver()
    with pytest.raises(ValueError): control.trial(target,**kwargs)


@pytest.mark.parametrize('node,component',[(0,1),(2,3),(True,1),(2,True),(-1,0)])
def test_only_free_translations_are_controlled(node,component):
    beam,_=make_beam(2)
    with pytest.raises(ValueError): DisplacementControlledAssemblyProbe(beam,node=node,component=component)


def test_small_controlled_reversal_restores_elastic_reference_state():
    _,control=driver()
    for target in (.095,.09,.095,.1):
        trial=control.trial(target);control.commit(trial)
        assert digest(control.committed_model.replay())==digest(trial.assembly.response)
    state=control.committed_model.committed
    assert np.max(np.abs(state.positions-control.committed_model._coordinates))<1e-10
    assert np.max(np.abs(state.rotations-np.eye(3)))<1e-10
    assert abs(state.forces[2,1])<1e-10
    assert all(h.accumulated==0 for group in state.histories for h in group)


def test_coupled_plastic_section_control_replay_and_discard():
    from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
    from docs.reference_cases.ge_beam3_curved_p5_assembly_history_probe import NonlinearAssemblyHistoryProbe
    from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law
    reference=CurvedBeam3ReferenceGeometry([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]],np.tile(np.eye(3),(3,1,1)))
    beam=NonlinearAssemblyHistoryProbe([reference],[(0,1,2)],[law()],order=8)
    control=DisplacementControlledAssemblyProbe(beam,node=2,component=0)
    active=False
    for target in (2.01,2.02,2.01,2.,1.99,2.):
        before=digest(control.committed_model.committed)
        trial=control.trial(target)
        assert digest(control.committed_model.committed)==before
        active |= any(s.response.plastic_active for e in trial.assembly.response.elements for s in e.stations)
        control.commit(trial)
        assert digest(control.committed_model.replay())==digest(trial.assembly.response)
    assert active and any(h.accumulated>0 for hs in control.committed_model.committed.histories for h in hs)
    before=digest(control.committed_model.committed)
    trial=control.trial(2.01);control.discard(trial)
    assert digest(control.committed_model.committed)==before
