"""Atomic native beam / shell J2 state, not complete G3 qualification."""
from dataclasses import replace

import anyfesolver
from anymaterial import DNVC208MaterialCurve
import numpy as np
import pytest

from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_experimental import NodalAction
from anyfesolver._stiffened_shell_ge_checkpoint_binary import (
    NATIVE_SHELL_MAGIC, decode_s3, encode_s3,
)
from anyfesolver._stiffened_shell_ge_finite_experimental import native_shell_checkpoint_shapes
from anyfesolver.e4_pl_s3_v2d_state import seal_v2d_state
from test_stiffened_shell_ge_physical_fibre_finite import _yielding_definition
from test_stiffened_shell_ge_physical_fibre_linear import _definition


def definition_for(mixed=False, triangles=False):
    base = (_definition(True, True) if triangles else _yielding_definition(mixed))
    pressure = 20_000. if triangles else 1e6
    curve = DNVC208MaterialCurve(pressure, 1.05*pressure, 1.1*pressure,
                                .005, .01, 4.*pressure, .2)
    return replace(base, shell=replace(base.shell, j2_curve=curve))


@pytest.mark.parametrize('mixed,triangles', ((False,False),(True,False),(True,True)))
def test_coupled_material_update_restart_and_rejection(mixed, triangles, monkeypatch):
    definition = definition_for(mixed, triangles)
    owner = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    resumed = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    bad = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    actions = (NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)
    targets = (.05,) if triangles else (.5,1.)
    following = .055 if triangles else .5
    try:
        accepted = owner.solve(actions, targets=targets, max_iterations=20, max_backtracks=12)
        assert accepted.status == 'completed'
        assert max(max(row) for row in accepted.shell_layer_alpha) >= 0.
        if triangles:
            assert max(max(row) for row in accepted.shell_layer_alpha) > 0.
        if not triangles:
            assert any(sum(row.incremental_dissipation_j)>0. for row in accepted.beam_fibre_energy)
        assert accepted.free_equilibrium_ratio <= 1e-8 and accepted.constraint_max <= 1e-9
        saved = owner.checkpoint()
        assert saved.startswith(NATIVE_SHELL_MAGIC)
        generation, arrays, blobs = decode_s3(saved, owner._identity,
            lambda generation: native_shell_checkpoint_shapes(owner,generation),
            2*len(definition.shell.s3_elements), native_material=True)
        assert all(not row.flags.writeable for row in arrays.values())
        restored = resumed.restore(saved)
        assert resumed.checkpoint() == saved
        assert restored.shell_layer_alpha == accepted.shell_layer_alpha
        assert restored.beam_fibre_energy == accepted.beam_fibre_energy
        for name in ('shell_displacement','cell_resultants','support_reactions'):
            np.testing.assert_array_equal(getattr(restored,name), getattr(accepted,name))
        for original, repeated in zip(accepted.shell_q4_material+accepted.shell_s3_material,
                                      restored.shell_q4_material+restored.shell_s3_material):
            for name in ('layer_strain','layer_stress','plastic_strain','alpha'):
                np.testing.assert_array_equal(getattr(original,name),getattr(repeated,name))
        changed = {name:np.array(row,copy=True) for name,row in arrays.items()}
        changed['accepted_fibres'][0,2] += 1e-6
        corrupted = encode_s3(owner._identity, generation, tuple(changed.items()),
                              blobs, native_material=True)
        with pytest.raises(ValueError, match='update disagrees with accepted history'):
            bad.restore(corrupted)
        assert bad._generation == 0 and bad._finite_driver is None
        if triangles:
            driver = owner._finite_driver
            element = owner._shell_parts[0][1]
            forged = dict(driver._shell_origins[0])
            forged['layer_stress'] = np.array(forged['layer_stress'], copy=True)
            forged['layer_stress'][0,0] += 1.
            forged = seal_v2d_state(forged)
            forged_blob = element.serialize_nonlinear_state(owner._shell_model.mesh,
                owner._shell_model.get_material('shell'), forged, 3)
            corrupted = encode_s3(owner._identity,generation,tuple(arrays.items()),
                                  (forged_blob,)+blobs[1:],native_material=True)
            with pytest.raises(ValueError,match='response disagrees with accepted update'):
                bad.restore(corrupted)
            assert bad._generation == 0 and bad._finite_driver is None
        token = CancellationToken(); token.cancel('coupled material prefix')
        with pytest.raises(SolveCancelled):
            owner.solve(actions, targets=(following,), cancellation_token=token)
        assert owner.checkpoint() == saved
        with pytest.raises(RuntimeError):
            owner.solve(actions, targets=(following,), max_iterations=1,max_backtracks=0)
        assert owner.checkpoint() == saved
        from anyfesolver import _stiffened_shell_ge_finite_experimental as finite
        original = finite.cancellation_safe_point
        token = CancellationToken()
        def cancel(token_value, stage):
            if stage == 'stiffened-shell.finite.material-publish':
                token.cancel('after coupled recovery')
            return original(token_value,stage)
        with monkeypatch.context() as context:
            context.setattr(finite,'cancellation_safe_point',cancel)
            with pytest.raises(SolveCancelled,match='after coupled recovery'):
                owner.solve(actions,targets=(following,),max_iterations=20,max_backtracks=12,
                            cancellation_token=token)
        assert owner.checkpoint() == saved and owner.recover() is accepted
        direct = owner.solve(actions, targets=(following,), max_iterations=20,max_backtracks=12)
        continued = resumed.solve(actions, targets=(following,),max_iterations=20,max_backtracks=12)
        assert owner.checkpoint() == resumed.checkpoint()
        assert direct.beam_fibre_energy == continued.beam_fibre_energy
        assert direct.shell_layer_alpha == continued.shell_layer_alpha
        if triangles:
            owner.rebase_rotations(expected_generation=direct.generation)
            chart_saved = owner.checkpoint()
            chart_owner = anyfesolver.stiffened_shell.create_analysis(definition=definition)
            try:
                chart_owner.restore(chart_saved)
                assert chart_owner.checkpoint() == chart_saved
                assert chart_owner.recover().shell_layer_alpha == owner.recover().shell_layer_alpha
            finally:
                chart_owner.close()
    finally:
        owner.close();resumed.close();bad.close()
