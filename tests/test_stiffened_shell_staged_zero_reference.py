"""Explicit zero-load equilibrium and durable plastic history, scoped development."""
from dataclasses import replace
from hashlib import sha256
from math import fsum
import json
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver import _stiffened_shell_staged_load_experimental as staged
from anyfesolver._stiffened_shell_ge_experimental import _linear_equilibrium, PROFILE
from anyfesolver._stiffened_shell_beam_history import arrays as native_history_arrays
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_staged_load_definition_experimental import (
    MODEL, _load, _owner as compiler_owner, _programme, _unframe, _frame, _canonical)
from test_stiffened_shell_ge_physical_fibre_finite import _yielding_definition


def fake_owner():
    owner = compiler_owner()
    owner._definition = SimpleNamespace(profile=PROFILE, points=(object(),),
        lines=(), footprints=(), distributing_footprints=(),
        finite_distributing_footprints=(), shell_reference_policy='flat')
    owner._free = np.arange(12)
    owner._coordinate_scale = np.ones(12)
    return owner


def test_zero_reference_identity_default_bytes_and_strict_checkpoint_policy():
    legacy = _programme()
    expected_core = dict(policy=staged.POLICY, model_identity=MODEL,
        endpoints=legacy.to_payload()['endpoints'])
    expected = _canonical({**expected_core, 'identity':sha256(_canonical(expected_core)).hexdigest()})
    assert legacy.serialize() == expected
    chosen = replace(legacy,zero_load_reference_endpoint=1)
    other = replace(legacy,zero_load_reference_endpoint=2)
    assert len({p.identity for p in (legacy,chosen,other)}) == 3
    assert len({p.augmented_model_identity for p in (legacy,chosen,other)}) == 3
    assert staged.StagedDeadLoadProgramme.deserialize(chosen.serialize()) == chosen
    compiled = chosen.compile(fake_owner(),_load)
    raw = staged.encode_checkpoint(compiled,1.,b'inner')
    assert staged.decode_checkpoint(raw) == (chosen,1.,b'inner')
    header,inner = _unframe(raw)
    header['policy'] = staged.POLICY
    with pytest.raises(ValueError,match='policy mismatch'):
        staged.decode_checkpoint(_frame(header,inner))
    payload = chosen.to_payload()
    payload['zero_load_reference_endpoint'] = 2
    with pytest.raises(ValueError,match='identity'):
        staged.StagedDeadLoadProgramme.from_payload(payload)


@pytest.mark.parametrize('value',(True,0,3,-1,1.,'1'))
def test_invalid_reference_endpoint_rejected(value):
    with pytest.raises(ValueError,match='endpoint'):
        replace(_programme(),zero_load_reference_endpoint=value)


def test_owned_reference_exact_zero_only_and_free_work_admission():
    chosen = replace(_programme(),zero_load_reference_endpoint=1)
    owner = fake_owner()
    buffer = np.zeros(12)
    def reused(actions):
        buffer[:] = _load(actions)
        return buffer
    compiled = chosen.compile(owner,reused)
    buffer[:] = 999.
    zero = np.zeros(12)
    reference = compiled.equilibrium_load(zero)
    assert reference is compiled.vectors[1] and not reference.flags.writeable
    assert not np.any(zero)
    tiny = np.zeros(12); tiny[7] = np.finfo(float).tiny
    assert compiled.equilibrium_load(tiny) is tiny
    assert _linear_equilibrium(-tiny,tiny,np.ones(12)) == 1.
    residual = np.zeros(12); residual[0] = 1e-6
    assert _linear_equilibrium(residual,reference,np.ones(12)) > 1e-8
    residual[:] = 0.; residual[4] = 1e-6
    assert _linear_equilibrium(residual,reference,np.ones(12)) > 1e-8
    owner._free = np.array([7])
    with pytest.raises(ValueError,match='free work'):
        chosen.compile(owner,_load)
    owner = fake_owner(); owner._coordinate_scale[0] = np.finfo(float).max
    with pytest.raises(ValueError,match='free work'):
        chosen.compile(owner,_load)
    tiny_action = ss.NodalAction('shell',1,(np.nextafter(0.,1.),0.,0.,0.,0.,0.))
    tiny_reference = staged.StagedDeadLoadProgramme(MODEL,((tiny_action,),),
        zero_load_reference_endpoint=1)
    owner = fake_owner(); owner._coordinate_scale[0] = .1
    with pytest.raises(ValueError,match='free work'):
        tiny_reference.compile(owner,_load)
    owner = fake_owner(); owner._definition.lines = (object(),)
    with pytest.raises(ValueError,match='point GE'):
        chosen.compile(owner,_load)
    with pytest.raises(ValueError,match='nonzero'):
        staged.StagedDeadLoadProgramme(MODEL,((),),zero_load_reference_endpoint=1)


def test_zero_reference_equivalent_work_units_preserve_small_component_checks():
    # The work map uses translations in length units and rotations dimensionless.
    # Change N/m to kN/mm with covector and coordinate dual transformations.
    force_moment = np.array((1e3,1e-3,0.,2.,0.,0.))
    scales_m = np.array((2.,2.,2.,1.,1.,1.))
    residual = force_moment*np.array((1e-10,2e-9,0.,3e-9,0.,0.))
    ratio_m = _linear_equilibrium(residual,force_moment,scales_m)
    units = np.array((.001,.001,.001,1.,1.,1.))
    scales_mm = scales_m*np.array((1000.,1000.,1000.,1.,1.,1.))
    ratio_mm = _linear_equilibrium(residual*units,force_moment*units,scales_mm)
    assert ratio_m == pytest.approx(ratio_mm,rel=1e-15)
    residual[1] = force_moment[1]
    assert _linear_equilibrium(residual,force_moment,scales_m) >= 1.


def test_first_zero_stage_uses_prepared_reference_without_precommit_mutation(request,monkeypatch):
    from test_stiffened_shell_staged_load_owner_experimental import _owner,_actions
    from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
    owner = _owner(request)
    programme = ss.StagedDeadLoadProgramme(owner._identity,((),_actions()),
        zero_load_reference_endpoint=2)
    original = FinitePointDriver._assemble
    observed = []
    def inspect(driver,state,factor,shell_origins,beam_origins,**kwargs):
        if not observed:
            assert driver._programme is None and owner._generation == 0
            assert kwargs['load_programme'].programme == programme
            direction = np.zeros(owner._K.shape[0]+owner._C.shape[0])
            direction[owner._free[0]] = 1e-9
            perturbed = driver._advance(state,direction,1.)
            response = original(driver,perturbed,factor,shell_origins,beam_origins,**kwargs)
            assert np.isfinite(response[2]) and response[2] > 0.
            old = original(driver,perturbed,factor,shell_origins,beam_origins,
                applied_load=kwargs['applied_load'])
            assert np.isinf(old[2])
            observed.append(response[2])
            assert driver._programme is None and owner._generation == 0
        return original(driver,state,factor,shell_origins,beam_origins,**kwargs)
    monkeypatch.setattr(FinitePointDriver,'_assemble',inspect)
    result = owner.solve(programme,(1.,))
    assert observed and result.load_equilibrium_reference_used
    assert not np.any(result.applied_dead_load) and result.load_position == 1.


def test_point_plastic_zero_unload_restart_and_atomic_failure(request,monkeypatch,record_property,tmp_path):
    model = _yielding_definition(mixed=True)
    def make():
        owner = ss.create_analysis(definition=model)
        request.addfinalizer(owner.close)
        return owner
    control,resumed,failed = make(),make(),make()
    actions = (ss.NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)
    programme = ss.StagedDeadLoadProgramme(control.preflight()['identity'],
        (actions,(),actions),zero_load_reference_endpoint=1)
    options = dict(max_iterations=20,max_backtracks=12)
    loaded = control.solve(programme,(.5,1.),**options)
    assert any(sum(row.incremental_dissipation_j) > 0. for row in loaded.beam_fibre_energy)
    loaded_checkpoint = control.checkpoint()
    def histories(owner):
        return dict(native_history_arrays(owner._beam_parts,
            owner._finite_driver._beam_origins,'accepted'))
    previous = histories(control)
    snapshots = []
    def observe(result):
        nonlocal previous
        current = histories(control)
        for name,columns in (('accepted_fibres',slice(2,None)),
                             ('accepted_accumulated',slice(None))):
            assert current[name].shape == previous[name].shape
            for a,b in zip(current[name][:,columns],previous[name][:,columns]):
                change = fsum((*a,*-b))
                rounding = 64*np.finfo(float).eps*max(abs(fsum(a)),abs(fsum(b)))
                assert change >= -rounding
        assert all(fsum(row.incremental_dissipation_j) >= 0. for row in result.beam_fibre_energy)
        snapshots.append(dict(position=result.load_position,
            accumulated=[fsum(row[2:]) for row in current['accepted_fibres']]))
        (tmp_path/f'accepted-{result.load_position}.checkpoint').write_bytes(control.checkpoint())
        previous = current
    failed.restore(loaded_checkpoint)
    token = CancellationToken(); token.cancel('before zero unloading')
    with pytest.raises(SolveCancelled):
        failed.solve(programme,(1.5,2.),cancellation_token=token,**options)
    assert failed.checkpoint() == loaded_checkpoint
    with monkeypatch.context() as patch:
        from anyfesolver import _stiffened_shell_ge_finite_experimental as finite
        patch.setattr(finite.spla,'spsolve',lambda matrix,rhs:np.zeros_like(rhs))
        with pytest.raises(RuntimeError):
            failed.solve(programme,(2.,),max_iterations=1,max_backtracks=0)
    assert failed.checkpoint() == loaded_checkpoint
    unloaded = control.solve(programme,(1.5,2.),on_accepted=observe,**options)
    assert not np.any(unloaded.applied_dead_load)
    assert unloaded.load_equilibrium_reference_used
    assert unloaded.load_equilibrium_reference_endpoint == 1
    assert unloaded.load_equilibrium_reference_policy == staged.ZERO_REFERENCE_POLICY
    assert unloaded.free_equilibrium_ratio <= 1e-8 and unloaded.constraint_max <= 1e-9
    assert np.max(histories(control)['accepted_fibres'][:,2:].sum(axis=1)) > 0.
    positions = np.vstack((np.asarray(model.shell.coordinates_m)+
        unloaded.shell_displacement.reshape(-1,6)[:,:3],unloaded.beam_positions))
    reactions = unloaded.support_reactions[:6*len(positions)].reshape(-1,6)
    force = reactions[:,:3].sum(axis=0)
    moment = (reactions[:,3:]+np.cross(positions,reactions[:,:3])).sum(axis=0)
    assert np.linalg.norm(force)/1000. <= 1e-8
    assert np.linalg.norm(moment)/2000. <= 1e-8
    for port in unloaded.point_wrenches:
        actions = np.array((port.shell_action,port.beam_action))
        position = np.array((port.shell_position_m,port.beam_position_m))
        low = np.array((port.shell_position_low_m,port.beam_position_low_m))
        assert np.linalg.norm(actions[:,:3].sum(axis=0))/1000. <= 1e-8
        assert np.linalg.norm((actions[:,3:]+np.cross(position,actions[:,:3])+
            np.cross(low,actions[:,:3])).sum(axis=0))/2000. <= 1e-8
    raw = control.checkpoint()
    descriptor,position,inner = staged.decode_checkpoint(raw)
    assert descriptor == programme and position == 2.
    other_reference = replace(programme,zero_load_reference_endpoint=3)
    other = make()
    from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
    forged = staged.encode_checkpoint(other_reference.compile(other,FinitePointDriver(other)._load),2.,inner)
    with pytest.raises(ValueError):
        other.restore(forged)
    assert other._generation == 0 and other._finite_driver is None
    restored = resumed.restore(raw)
    assert resumed.checkpoint() == raw and restored.load_equilibrium_reference_used
    for field in ('beam_positions','cell_resultants','multipliers','support_reactions'):
        np.testing.assert_array_equal(getattr(restored,field),getattr(unloaded,field))
    assert restored.beam_fibre_energy == unloaded.beam_fibre_energy
    a = control.solve(programme,(2.5,3.),on_accepted=observe,**options)
    b = resumed.solve(programme,(2.5,3.),**options)
    assert not a.load_equilibrium_reference_used
    assert control.checkpoint() == resumed.checkpoint()
    assert a.beam_fibre_energy == b.beam_fibre_energy
    assert a.accepted_factors == (.5,1.,1.5,2.,2.5,3.)
    # A descriptor change cannot silently continue the accepted history.
    legacy = replace(programme,zero_load_reference_endpoint=None)
    with pytest.raises(ValueError,match='programme changed'):
        resumed.solve(legacy,(3.5,),**options)
    assert control.checkpoint() == resumed.checkpoint()
    record_property('point_zero_reference',json.dumps(dict(
        positions=a.accepted_factors,zero_position=2.,policy=programme.policy,
        zero_equilibrium=unloaded.free_equilibrium_ratio,gap=unloaded.constraint_max,
        force_N=force.tolist(),moment_Nm=moment.tolist(),restart_exact=True,
        history_non_decreasing=True,snapshots=snapshots)))
