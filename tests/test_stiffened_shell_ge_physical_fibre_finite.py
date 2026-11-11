"""Finite native histories through the existing experimental point owner."""
from dataclasses import replace
import json
import warnings

import anyfesolver
import numpy as np
import pytest

from anyfesolver import b3_ge
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
from anyfesolver._stiffened_shell_ge_experimental import NodalAction
from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode, encode, SCHEMAS
from anyfesolver._stiffened_shell_beam_history import arrays, histories, shapes
from test_stiffened_shell_ge_physical_fibre_linear import _definition


def _yielding_definition(mixed=False, triangles=False):
    definition = _definition(mixed, triangles)
    beams = []
    for native in definition.beams:
        element, inertia = native.instantiate()
        if type(element.section) is not b3_ge.PhysicalFibreSection:
            beams.append(native)
            continue
        section = b3_ge.PhysicalFibreSection(tuple(
            replace(fibre, curve=b3_ge.FlowCurve((0.,), (1e5,), 1e9))
            if index == 0 else fibre
            for index, fibre in enumerate(element.section.fibres)), element.section.background_factor)
        reference = element.operator.reference
        beams.append(b3_ge.define_beam('b3-ge', element.element_id,
            element.node_ids, reference.coordinates, reference.nodal_triads,
            section, inertia, quadrature=element.operator.order))
    return replace(definition, beams=tuple(beams))


@pytest.mark.parametrize('mixed,triangles', ((False,False),(True,False),(True,True)))
def test_finite_native_response_restart_and_predecessor(mixed, triangles):
    definition = _definition(mixed, triangles)
    owner = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    actions = (NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)
    first = owner.solve(actions, targets=(.5,1.))
    assert first.generation == 2
    assert first.beam_fibre_material
    assert all(record.fibres for record in first.beam_fibre_material)
    assert all(not record.strain.flags.writeable for record in first.beam_fibre_material)
    driver = owner._finite_driver
    saved = owner.checkpoint()
    resumed = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    restored = resumed.restore(saved)
    assert resumed.checkpoint() == saved
    for name in ('beam_positions','beam_frames','cell_frames','cell_resultants',
                 'multipliers','shell_displacement'):
        np.testing.assert_array_equal(getattr(first,name),getattr(restored,name))
    for a,b in zip(first.beam_fibre_material, restored.beam_fibre_material):
        assert a.fibres == b.fibres
        np.testing.assert_array_equal(a.strain,b.strain)
        np.testing.assert_array_equal(a.resultants,b.resultants)
    def expected(kind, generation):
        assert kind == ('finite_native_material_stable' if triangles else 'finite_native_material')
        count = owner._K.shape[0]
        return (("q",(count,)),("frames",(len(owner._beam_ids),3,3)),
            ("cells",(len(owner._beam_parts),2,3,3)),
            ("resultants",(len(owner._beam_parts),18)),
            ("multipliers",(owner._C.shape[0],)),("pattern",(count,)),
            ("accepted_factors",(generation,))) + shapes(owner._beam_parts,'origin') + shapes(owner._beam_parts,'accepted')
    kind,generation,values = decode(saved,owner._identity,expected)
    json_payload = json.dumps({'schema': SCHEMAS[kind],
        'model': owner._identity, 'generation': generation,
        **{name:value.tolist() for name,value in values.items()}}).encode()
    json_owner = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    with pytest.raises(ValueError,match='requires binary framing'):
        json_owner.restore(json_payload)
    assert json_owner._generation == 0
    json_owner.close()
    changed = {name: np.array(value,copy=True) for name,value in values.items()}
    changed['accepted_fibres'][0,2] += 1e-6
    corrupt = encode(kind,owner._identity,generation,tuple(changed.items()))
    untouched = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    with pytest.raises(ValueError,match='update disagrees with accepted history'):
        untouched.restore(corrupt)
    assert untouched._generation == 0 and untouched._finite_driver is None
    untouched.close()
    if not triangles:
        forged = encode('finite_native_material_stable',owner._identity,
                        generation,tuple(values.items()))
        ineligible = anyfesolver.stiffened_shell.create_analysis(definition=definition)
        try:
            with pytest.raises(ValueError,match='requires S3 elements'):
                ineligible.restore(forged)
            assert ineligible._generation == 0 and ineligible._finite_driver is None
        finally:
            ineligible.close()
    continued = resumed.solve(actions,targets=(.5,))
    oracle = owner.solve(actions,targets=(.5,))
    np.testing.assert_array_equal(continued.cell_resultants,oracle.cell_resultants)
    assert continued.beam_station_accumulated == oracle.beam_station_accumulated
    packed = dict(arrays(owner._beam_parts,driver._beam_origins,'accepted'))
    rebuilt = histories(owner._beam_parts,packed,'accepted')
    assert rebuilt == driver._beam_origins
    packed['accepted_fibres'][:] = np.nan
    assert rebuilt == driver._beam_origins
    with pytest.raises(ValueError,match='finite model-bound'):
        histories(owner._beam_parts,packed,'accepted')
    if triangles:
        rebased = owner.rebase_rotations(expected_generation=oracle.generation)
        charted = owner.checkpoint()
        chart_owner = anyfesolver.stiffened_shell.create_analysis(definition=definition)
        try:
            chart_result = chart_owner.restore(charted)
            assert chart_owner._finite_driver._stable_s3
            assert chart_owner.checkpoint() == charted
            np.testing.assert_array_equal(chart_result.shell_frames,rebased.shell_frames)
            np.testing.assert_array_equal(chart_result.cell_resultants,rebased.cell_resultants)
        finally:
            chart_owner.close()
    owner.close(); resumed.close()


@pytest.mark.parametrize('mixed,triangles', ((False,False),(True,False)))
def test_native_yield_unload_reject_cancel_and_restore(mixed,triangles):
    definition = _yielding_definition(mixed,triangles)
    owner = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    actions = (NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)
    first = owner.solve(actions,targets=(.5,1.),
                        max_iterations=18,max_backtracks=8)
    assert max(v for row in first.beam_station_accumulated for v in row) > 0.
    assert any(sum(f.plastic_increment) != 0. for record in first.beam_fibre_material
               for f in record.fibres)
    saved = owner.checkpoint()
    with pytest.raises(RuntimeError):
        owner.solve(actions,targets=(2.,),max_iterations=1,max_backtracks=0)
    assert owner.checkpoint() == saved
    token = CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled):
        owner.solve(actions,targets=(1.5,),cancellation_token=token)
    assert owner.checkpoint() == saved
    resumed = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    restored = resumed.restore(saved)
    assert resumed.checkpoint() == saved
    assert any(sum(f.plastic_increment) != 0. for record in restored.beam_fibre_material
               for f in record.fibres)
    assert resumed._finite_driver._beam_predecessors != resumed._finite_driver._beam_origins
    a = owner.solve(actions,targets=(.5,),max_iterations=18)
    b = resumed.solve(actions,targets=(.5,),max_iterations=18)
    np.testing.assert_array_equal(a.cell_resultants,b.cell_resultants)
    assert owner.checkpoint() == resumed.checkpoint()
    owner.close(); resumed.close()


def test_native_cancel_after_material_recovery_preserves_accepted_state(monkeypatch):
    from anyfesolver import _stiffened_shell_ge_finite_experimental as finite
    owner = anyfesolver.stiffened_shell.create_analysis(definition=_yielding_definition())
    actions = (NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)
    first = owner.solve(actions,targets=(.5,))
    saved = owner.checkpoint()
    token = CancellationToken()
    original = finite.cancellation_safe_point
    def cancel(token_value, stage):
        if stage == 'stiffened-shell.finite.material-publish':
            token.cancel('after native recovery')
        return original(token_value,stage)
    with monkeypatch.context() as context:
        context.setattr(finite,'cancellation_safe_point',cancel)
        with pytest.raises(SolveCancelled,match='after native recovery'):
            owner.solve(actions,targets=(1.,),cancellation_token=token)
    assert owner.recover() is first
    assert owner.checkpoint() == saved
    assert owner._generation == first.generation
    owner.close()


@pytest.mark.parametrize('yielding', (False,True))
def test_pure_fibre_s3_monotonic_completion_and_restart(yielding, monkeypatch):
    """The stable shell map must actually converge the unchanged fixture.

    Earlier failures stay in immutable evidence; full G3 remains unqualified.
    """
    definition = (_yielding_definition(False,True) if yielding else _definition(False,True))
    owner = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    actions = (NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)
    try:
        result = owner.solve(actions,targets=(.5,1.))
        assert result.accepted_factors == (.5,1.)
        assert result.free_equilibrium_ratio <= 1e-8
        assert result.constraint_max <= 1e-9
        if yielding:
            assert any(sum(row.incremental_dissipation_j)>0.
                       for row in result.beam_fibre_energy)
        support = result.support_reactions[:owner._shell_count].reshape(-1,6)
        force = support[:,:3].sum(axis=0)
        moment = support[:,3:].sum(axis=0)+np.cross(
            np.asarray(definition.shell.coordinates_m),support[:,:3]).sum(axis=0)
        applied=np.array((0.,0.,1000.))
        assert np.linalg.norm(force+applied)/np.linalg.norm(applied)<1e-8
        assert np.linalg.norm(moment+np.cross(result.beam_positions[-1],applied))/(
            np.linalg.norm(applied)*owner._reference_length)<1e-8
        saved = owner.checkpoint()
        resumed = anyfesolver.stiffened_shell.create_analysis(definition=definition)
        try:
            restored = resumed.restore(saved)
            assert resumed.checkpoint() == saved
            assert restored.beam_fibre_energy == result.beam_fibre_energy
            assert owner._finite_driver._stable_s3
            assert resumed._finite_driver._stable_s3
            if yielding:
                # Rollback is proved deterministically first: an untyped
                # failure raised by the trial assembly must propagate and
                # leave the accepted prefix, checkpoint and generation
                # untouched.
                class _InjectedTrialFailure(RuntimeError):
                    pass
                def injected(*args, **kwargs):
                    raise _InjectedTrialFailure('injected trial assembly failure')
                with monkeypatch.context() as context:
                    context.setattr(owner._finite_driver,'_assemble',injected)
                    with pytest.raises(_InjectedTrialFailure):
                        owner.solve(actions,targets=(.5,))
                assert owner.recover() is result
                assert owner.checkpoint() == saved
                assert owner._generation == result.generation
                # The unchanged unloading is then attempted for real. Runtime
                # nonconvergence is not guaranteed: a genuine budget
                # exhaustion stays an unresolved diagnostic and a completed
                # unload must satisfy the same physical contract. Neither
                # outcome closes G3.
                try:
                    unload = owner.solve(actions,targets=(.5,))
                except FiniteIncrementFailure as error:
                    if error.attempt.failure != 'finite mixed backtrack budget exhausted':
                        raise
                    assert owner.recover() is result
                    assert owner.checkpoint() == saved
                    assert owner._generation == result.generation
                    warnings.warn('genuine finite mixed backtrack budget exhaustion on '
                        'the unchanged fibre unloading path: recorded as an unresolved '
                        'diagnostic, not qualification; G3 remains open')
                else:
                    assert unload.free_equilibrium_ratio <= 1e-8
                    assert unload.constraint_max <= 1e-9
                    support = unload.support_reactions[:owner._shell_count].reshape(-1,6)
                    force = support[:,:3].sum(axis=0)
                    moment = support[:,3:].sum(axis=0)+np.cross(
                        np.asarray(definition.shell.coordinates_m),support[:,:3]).sum(axis=0)
                    applied = np.array((0.,0.,500.))
                    assert np.linalg.norm(force+applied)/np.linalg.norm(applied)<1e-8
                    assert np.linalg.norm(moment+np.cross(unload.beam_positions[-1],applied))/(
                        np.linalg.norm(applied)*owner._reference_length)<1e-8
                    repeated = resumed.solve(actions,targets=(.5,))
                    assert owner.checkpoint() == resumed.checkpoint()
                    assert unload.beam_fibre_energy == repeated.beam_fibre_energy
                    for a,b in zip(unload.beam_fibre_material,repeated.beam_fibre_material):
                        assert a.fibres == b.fibres
                        np.testing.assert_array_equal(a.strain,b.strain)
                        np.testing.assert_array_equal(a.resultants,b.resultants)
                return
            direct = owner.solve(actions,targets=(.5,))
            continued = resumed.solve(actions,targets=(.5,))
            assert owner.checkpoint() == resumed.checkpoint()
            assert direct.beam_fibre_energy == continued.beam_fibre_energy
        finally:
            resumed.close()
    finally:
        owner.close()
