"""Simultaneous shell J2/generalized beam history on experimental Q4 BL5."""
from dataclasses import replace
from math import fsum
import json
import numpy as np
import pytest
from anymaterial import DNVC208MaterialCurve
from anyfesolver import stiffened_shell as ss
from anyfesolver._stiffened_shell_beam_history import arrays as native_history_arrays
from test_stiffened_shell_bl5_material import model, make
from test_stiffened_shell_bl5_finite import _action


def coupled(basis="p0", *, simultaneous_yield_fixture=False):
    base=model(basis,hardening=1e8)
    p=1e6
    # Keep saved-checkpoint consumers on their original material identity.
    strains=(.0005,.001) if simultaneous_yield_fixture else (.005,.01)
    curve=DNVC208MaterialCurve(p,1.05*p,1.1*p,*strains,4*p,.2)
    return replace(base,shell=replace(base.shell,j2_curve=curve),
                   bl5_material_policy=ss.BL5_COUPLED_J2_MATERIAL_V1)


def test_yielded_bl5_actual_update_tangent_in_all_row_families(request, record_property):
    """Distinguish an untested coupled derivative from continuation control."""
    definition = coupled("p0", simultaneous_yield_fixture=True)
    owner = make(request, definition)
    programme = ss.StagedDeadLoadProgramme(
        owner.preflight()["identity"], (_action(700.), _action(-500.)))
    result = owner.solve(programme, (.5, 1.), max_iterations=20,
                         max_backtracks=12,
                         increment_control=ss.AdaptiveIncrementControl(4, 64))
    assert result.load_position == 1.
    assert np.max(result.beam_station_accumulated) > 0.
    assert np.max(result.shell_layer_alpha) > 0.
    saved = owner.checkpoint()
    driver = owner._finite_driver

    def evaluate(state):
        return driver._assemble(state, 1., driver._shell_predecessors,
                                driver._beam_predecessors,
                                applied_load=result.applied_dead_load)

    def active(response):
        beam_trial = dict(native_history_arrays(owner._beam_parts, response[5], "trial"))
        beam_origin = dict(native_history_arrays(
            owner._beam_parts, driver._beam_predecessors, "origin"))
        generalized = np.array([fsum((*a, *-b)) > 0. for a, b in zip(
            beam_trial["trial_accumulated"], beam_origin["origin_accumulated"])])
        shell = np.concatenate([
            candidate["alpha"] - origin["alpha"] > 0.
            for candidate, origin in zip(response[4], driver._shell_predecessors)])
        return generalized, shell

    residual, tangent, *_ = evaluate(driver._state)
    reference_mask = active(evaluate(driver._state))
    count = owner._K.shape[0]
    beam_end = owner._shell_count + 6 * len(owner._beam_ids)
    direction = np.zeros(len(residual))
    rng = np.random.default_rng(818)
    direction[:beam_end] = rng.normal(size=beam_end) * 1e-4
    for i in range(len(owner._beam_parts)):
        base = beam_end + 24 * i
        direction[base:base+6] = rng.normal(size=6) * 1e-4
        resultants = direction[base+6:base+24].reshape(3, 6)
        resultants[:, :3] = rng.normal(size=(3, 3)) * 1000.
        resultants[:, 3:] = rng.normal(size=(3, 3)) * 30.
    multipliers = direction[count:].reshape(-1, 5)
    multipliers[:, :3] = rng.normal(size=multipliers[:, :3].shape) * 1000.
    multipliers[:, 3:] = rng.normal(size=multipliers[:, 3:].shape) * 30.
    direction[owner._fixed] = 0.
    weights = np.r_[owner._coordinate_scale, np.ones(owner._C.shape[0])]
    predicted = (tangent @ direction) * weights
    groups = {
        "shell": owner._free[owner._free < owner._shell_count],
        "beam": owner._free[(owner._free >= owner._shell_count) &
                            (owner._free < beam_end)],
        "retained": np.arange(beam_end, count),
        "constraints": np.arange(count, len(residual)),
    }
    norms = {name: float(np.linalg.norm(predicted[rows])) for name, rows in groups.items()}
    assert all(value > 0. for value in norms.values()), norms
    steps = (1e-4, 1e-5, 1e-6)
    errors = {name: [] for name in groups}
    for h in steps:
        plus = evaluate(driver._advance(driver._state, direction, h))
        minus = evaluate(driver._advance(driver._state, direction, -h))
        for response in (plus, minus):
            for expected, actual in zip(reference_mask, active(response)):
                np.testing.assert_array_equal(actual, expected)
        difference = ((plus[0] - minus[0]) / (2*h)) * weights - predicted
        for name, rows in groups.items():
            errors[name].append(float(np.linalg.norm(difference[rows]) / norms[name]))
    record_property("bl5_yielded_tangent", json.dumps(dict(
        steps=steps, relative_errors=errors, predicted_norms=norms,
        active_generalized=int(np.count_nonzero(reference_mask[0])),
        active_shell=int(np.count_nonzero(reference_mask[1])),
        physical_rows=beam_end, retained_rows=count-beam_end,
        multiplier_rows=len(residual)-count), sort_keys=True))
    assert any(all(errors[name][i] <= 1e-5 and errors[name][i+1] <= 1e-5
                   for name in groups) for i in range(len(steps)-1)), errors
    assert owner.checkpoint() == saved


def _coupled_history_roundtrip(request,monkeypatch,tmp_path,basis,*,reversal):
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
    definition=coupled(basis, simultaneous_yield_fixture=True);a=make(request,definition)
    assert "shell J2 plastic" not in a.capabilities["excluded"]
    programme=ss.StagedDeadLoadProgramme(a.preflight()["identity"],
                                       (_action(700.),_action(-500. if reversal else 770.)))
    targets=(.5,1.,1.25,1.75,2.) if reversal else (.5,1.,2.);control=ss.AdaptiveIncrementControl(4,64)
    kwargs=dict(max_iterations=20,max_backtracks=12,increment_control=control)
    token=CancellationToken()
    def observe(result):
        assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
        if result.load_position==1.:token.cancel("coupled accepted prefix")
    with pytest.raises(SolveCancelled):
        a.solve(programme,targets,on_accepted=observe,cancellation_token=token,**kwargs)
    result=a.recover();assert result.load_position==1.
    assert np.max(result.beam_station_accumulated)>0
    assert np.max(result.shell_layer_alpha)>0
    saved=a.checkpoint();b=make(request,definition);b.restore(saved)
    assert b.checkpoint()==saved
    for left,right in zip(result.shell_q4_material,b.recover().shell_q4_material):
        np.testing.assert_array_equal(left.alpha,right.alpha)
        np.testing.assert_array_equal(left.layer_stress,right.layer_stress)
    driver=a._finite_driver;state=driver._state
    def evaluate(trial):
        return driver._assemble(trial,1.,driver._shell_predecessors,
            driver._beam_predecessors,applied_load=result.applied_dead_load)
    residual,tangent,*_=evaluate(state);direction=np.zeros(len(residual))
    direction[:a._shell_count:6]=np.linspace(0.,1e-4,len(a._shell_ids))
    for i in range(len(a._beam_ids)):direction[a._shell_count+6*i+2]=.001*(i+1)
    direction[a._fixed]=0.;predicted=tangent@direction
    weights=np.r_[a._coordinate_scale,np.ones(a._C.shape[0])];errors=[]
    for h in (1e-4,1e-5,1e-6):
        plus=evaluate(driver._advance(state,direction,h))[0]
        minus=evaluate(driver._advance(state,direction,-h))[0]
        errors.append(np.linalg.norm(((plus-minus)/(2*h)-predicted)*weights)/
                      max(np.linalg.norm(predicted*weights),1.))
    assert min(errors)<=1e-5,errors
    original=FinitePointDriver._assemble
    def broken(self,*args,**kw):
        original(self,*args,**kw);raise RuntimeError("injected coupled trial")
    with monkeypatch.context() as patch:
        patch.setattr(FinitePointDriver,"_assemble",broken)
        with pytest.raises(RuntimeError,match="injected coupled"):
            a.solve(programme,targets,**kwargs)
    assert a.checkpoint()==saved
    a.rebase_rotations(result.generation)
    repeated=driver._assemble(driver._state,1.,driver._shell_origins,
        driver._beam_origins,applied_load=result.applied_dead_load)[4]
    for before,after in zip(result.shell_q4_material,repeated):
        np.testing.assert_allclose(before.layer_stress,after["layer_stress"],rtol=1e-10,atol=1e-7)
        np.testing.assert_allclose(before.layer_strain,after["layer_strain"],rtol=1e-10,atol=1e-12)
    b=make(request,definition);b.restore(a.checkpoint())
    from pathlib import Path
    from dataclasses import asdict
    import json
    evidence=tmp_path/"coupled-history"
    evidence.mkdir()
    (evidence/(basis+"-prefix.bin")).write_bytes(a.checkpoint())
    try:
        final=a.solve(programme,targets,**kwargs)
    except ss.AdaptiveIncrementExhausted as error:
        accepted=a.recover()
        (evidence/(basis+"-failure.json")).write_text(json.dumps(dict(
            attempts=[asdict(item) for item in error.attempts],
            accepted_factors=error.accepted_factors,
            load_position=accepted.load_position,
            applied_load_norm=float(np.linalg.norm(accepted.applied_dead_load)),
            tangent_errors=errors),indent=2))
        (evidence/(basis+"-failure-prefix.bin")).write_bytes(a.checkpoint())
        raise
    resumed=b.solve(programme,targets,**kwargs)
    assert final.load_programme_complete
    assert np.max(final.shell_layer_alpha)>np.max(result.shell_layer_alpha)
    assert np.max(final.beam_station_accumulated)>np.max(result.beam_station_accumulated)
    np.testing.assert_array_equal(final.shell_displacement,resumed.shell_displacement)
    assert a.checkpoint()==b.checkpoint()



@pytest.mark.parametrize("basis",("p0","continuous_p1_coarse"))
def test_coupled_bl5_history_roundtrip(request,monkeypatch,tmp_path,basis):
    # Full signed-history requirement remains separate and currently unresolved.
    _coupled_history_roundtrip(request,monkeypatch,tmp_path,basis,reversal=True)


def test_coupled_bl5_monotonic_lifecycle(request,monkeypatch,tmp_path):
    # A bounded lifecycle check, not a substitute for signed-history acceptance.
    _coupled_history_roundtrip(request,monkeypatch,tmp_path,"p0",reversal=False)


def test_coupled_native_retained_force_moment_scales_and_legacy_isolation(request):
    selected=make(request,coupled());legacy=make(request,model(hardening=1e8))
    assert selected.preflight()["identity"]!=legacy.preflight()["identity"]
    start=selected._shell_count+6*len(selected._beam_ids)
    length=selected._reference_length
    for i,definition in enumerate(selected._definition.beams):
        element,_=definition.instantiate()
        force=np.abs(element.section.elastic)[:3]@np.r_[np.ones(3),np.ones(3)/length]
        moment=np.abs(element.section.elastic)[3:]@np.r_[np.ones(3),np.ones(3)/length]
        actual=selected._coordinate_scale[start+24*i+6:start+24*i+24].reshape(6,3)
        np.testing.assert_array_equal(actual[:2],np.tile(force,(2,1)))
        np.testing.assert_array_equal(actual[2:],np.tile(moment,(4,1)))
        old=legacy._coordinate_scale[start+24*i+6:start+24*i+24]
        np.testing.assert_array_equal(old,np.tile(np.r_[force,moment],3))
    for bad in (replace(coupled(),bl5_material_policy=ss.BL5_GENERALIZED_MATERIAL_V1),
                replace(model(),bl5_material_policy=ss.BL5_COUPLED_J2_MATERIAL_V1)):
        with pytest.raises(ValueError):ss.create_analysis(definition=bad)
