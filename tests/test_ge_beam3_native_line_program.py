"""Bounded actual Newton line-program integration, not public qualification."""
from copy import deepcopy
import numpy as np
import pytest
from anyfesolver._ge_beam3_native_line_program import solve_line_static, _PROGRAM, model_identity
from anyfesolver._ge_beam3_native_line_loading import LinePattern, _ACTIVE
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_native_line_loading import problem, independent_work
from test_ge_beam3_schur_line_program import save


@pytest.mark.parametrize('curved,plastic,search,constant_scale', [(False,False,False,0.),(True,False,True,0.),(True,True,True,0.),(True,False,True,.25)])
def test_actual_newton_line_equilibrium(curved,plastic,search,constant_scale,tmp_path):
    model,e = problem(curved,plastic)
    prop = LinePattern(((1,.34,-.014,.007),))
    constant=LinePattern(()) if not constant_scale else LinePattern(((1,*map(float,constant_scale*np.array([.34,-.014,.007]))),))
    result,events = solve_line_static(model,prop,constant=constant,line_search=search)
    save(tmp_path/'result.json',dict(status=result.status,displacements=result.displacements,
        states=result.element_states,events=events,steps=[step.to_dict() for step in result.steps],production_qualified=False))
    assert result.status == 'completed', result.info
    assert len(result.snapshots) == 2
    audits=[]
    for snapshot in result.snapshots:
        state=deepcopy(snapshot.element_states[1]); factor=float(snapshot.load_factor)
        force=constant.force(1)+factor*prop.force(1)
        pattern=LinePattern(((1,*map(float,force)),))
        assert state['load_pattern'] == pattern
        e._validate(model.mesh,state)
        # Stored condensed residual is net of complete distributed work.
        residual=state['response'].residual
        assert np.linalg.norm(residual[6:]) <= 1e-11
        value,gradient,hessian=independent_work(e,state,pattern.force(1))
        reaction=np.array(result.steps[len(audits)].support_reactions['root'])
        reaction_error=float(np.linalg.norm(reaction-residual[:6]))
        assert reaction_error<=1e-11
        force_balance=reaction[:3]+gradient[:18].reshape(3,6)[:,:3].sum(axis=0)
        assert np.linalg.norm(force_balance)<=1e-11
        # Full load-only reconstruction is independently audited by the frozen
        # trial suite; here check global scope and accepted net equilibrium.
        audits.append(dict(factor=factor,free_residual=float(np.linalg.norm(residual[6:])),load_work=value,
            reaction_error=reaction_error,force_balance=force_balance,
            reaction_events=sum(event['reaction'] and event['parameter']==factor for event in events)))
        assert audits[-1]['reaction_events']==1
        assert all(event['pattern_sha256']==pattern.signature for event in events if event['parameter']==factor)
    assert all(bool(row['line_search_used'])==search for row in result.info['force_displacement_history'])
    assert canonical(result.snapshots[1].element_states[1]['origins'])==canonical(result.snapshots[0].element_states[1]['response'].history)
    if plastic: assert any(row[2]>0 for station in result.element_states[1]['response'].history.stations for row in station.rows)
    assert all(event['parameter'] in (.5,1.) for event in events)
    assert _PROGRAM.get() is None and _ACTIVE.get() is None
    save(tmp_path/'audit.json',audits)


def test_failed_newton_preserves_virgin_state(tmp_path):
    model,e=problem(True,True)
    initial=e.init_model_bound_nonlinear_state(model.mesh,e.section,1)
    result,events=solve_line_static(model,LinePattern(((1,.34,-.014,.007),)),steps=1,max_iterations=1)
    save(tmp_path/'failure.json',dict(status=result.status,states=result.element_states,events=events))
    assert result.status!='completed'
    assert canonical(result.element_states[1])==canonical(initial)
    assert _PROGRAM.get() is None and _ACTIVE.get() is None


@pytest.mark.parametrize('kind',['pattern','bounds','model'])
def test_invalid_authority_before_solver(kind,monkeypatch,tmp_path):
    import anyfesolver.nonlinear_static as solver
    model,_=problem(False,False); pattern=LinePattern(((1,.1,0.,0.),))
    kwargs={}
    if kind=='pattern':object.__setattr__(pattern,'signature','0'*64)
    elif kind=='bounds':kwargs['steps']=17
    else:model.boundary_conditions.clear()
    def forbidden(*a,**k):raise AssertionError('solver must not execute')
    monkeypatch.setattr(solver,'solve_static_nonlinear',forbidden)
    with pytest.raises(ValueError):solve_line_static(model,pattern,**kwargs)
    assert _PROGRAM.get() is None and _ACTIVE.get() is None
    save(tmp_path/'rejected.json',dict(kind=kind,rejected=True))


@pytest.mark.parametrize('kind',['signature','resealed-program','resealed-effective'])
def test_mid_assembly_mutation_discards_uncommitted_trial(kind,monkeypatch,tmp_path):
    import anyfesolver.nonlinear_static as solver
    from test_ge_beam3_native_fibre_static_element import store_for,sample
    from anyfesolver._ge_beam3_native_line_program import _Program,assemble_at
    model,e=problem(True,True);store=store_for(model,e);before=canonical(store.materialize())
    pattern=LinePattern(((1,.1,-.01,.005),));events=[]
    program=_Program(model,model_identity(model),pattern,LinePattern(()),events)
    original=solver._assemble_nonlinear_system
    def mutate(*a,**kw):
        result=original(*a,**kw)
        if kind=='signature': object.__setattr__(pattern,'signature','0'*64)
        elif kind=='resealed-program': object.__setattr__(program,'proportional',LinePattern(((1,.2,-.01,.005),)))
        else:
            changed=LinePattern(((1,.2,-.01,.005),)); active=_ACTIVE.get().pattern
            object.__setattr__(active,'rows',changed.rows); object.__setattr__(active,'signature',changed.signature)
        return result
    monkeypatch.setattr(solver,'_assemble_nonlinear_system',mutate)
    token=_PROGRAM.set(program)
    try:
        with pytest.raises(ValueError,match='authority'):assemble_at(.5,model,sample(),store,1)
    finally:_PROGRAM.reset(token)
    assert not store.has_active_trial and canonical(store.materialize())==before
    assert _ACTIVE.get() is None and _PROGRAM.get() is None and not events
    save(tmp_path/'mutation.json',dict(kind=kind,trial_discarded=True,committed_unchanged=True,scope_reset=True))
