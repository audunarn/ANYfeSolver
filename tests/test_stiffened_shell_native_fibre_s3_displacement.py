"""Current stable fibre/S3 displacement, with legacy arithmetic preserved."""
from dataclasses import replace
from math import fsum
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot
from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode, encode
from anyfesolver._stiffened_shell_beam_history import shapes
from test_stiffened_shell_native_point_displacement import (
    model, owner, compare, ACTIONS, DIRECTION, CONTROL, OPTIONS, HISTORICAL_POLICY)
from test_stiffened_shell_native_s3_displacement_admission import compare_generalized


def fibre_model(family='fibre', all_s3=True):
    base = model(family)
    q4 = () if all_s3 else ((1, 2, 5, 4),)
    s3 = ((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5)) if all_s3 else ((2, 3, 6), (2, 6, 5))
    return replace(base, shell=replace(base.shell, q4_elements=q4, s3_elements=s3),
                   q4_pullback_policy=HISTORICAL_POLICY if all_s3 else base.q4_pullback_policy)


@pytest.mark.parametrize('family,all_s3', (('fibre', False), ('fibre', True), ('mixed', True)))
def test_native_fibre_s3_flow_cancel_restore_reject(request, monkeypatch, tmp_path, family, all_s3):
    definition = fibre_model(family, all_s3)
    analysis = owner(request, definition)
    resumed = owner(request, definition)
    assert analysis.capabilities['displacement_control'] == 'experimental_native_point_fibre_elastic_s3_fixed_translation_v1'
    seed = analysis.solve(ACTIONS, (.5, 1.), **OPTIONS)
    assert analysis._finite_driver._stable_s3
    assert any(fsum(row.incremental_dissipation_j) > 0. for row in seed.beam_fibre_energy)
    dof = analysis._shell_count + 6*analysis._beam_index[105] + 2
    initial = analysis._finite_driver._state.q[dof]
    targets = (initial*1.01, initial*1.02)
    token = CancellationToken()
    def accepted(result):
        assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
        assert any(fsum(row.incremental_dissipation_j) > 0. for row in result.beam_fibre_energy)
        token.cancel('accepted fibre S3 displacement prefix')
    kwargs = dict(OPTIONS, displacement_control=CONTROL)
    with pytest.raises(SolveCancelled):
        analysis.solve(DIRECTION, targets, **kwargs, cancellation_token=token, on_accepted=accepted)
    saved = analysis.checkpoint()
    prefix = analysis.recover()
    metadata, _ = decode_snapshot(analysis, saved)
    assert metadata['policy'] == 'B3_GE_NATIVE_POINT_FIBRE_ELASTIC_S3_DISPLACEMENT_V1_DEV'
    restored = resumed.restore(saved)
    assert resumed.checkpoint() == saved and resumed._finite_driver._stable_s3
    compare(prefix, restored)
    if family == 'mixed':
        compare_generalized(prefix.beam_generalized_material, restored.beam_generalized_material)
    (tmp_path/'fibre-s3-prefix.checkpoint').write_bytes(saved)
    import anyfesolver._stiffened_shell_displacement_corrector as corrector
    with monkeypatch.context() as patch:
        patch.setattr(corrector, 'bordered_correction', lambda matrix, *args: (np.zeros(matrix.shape[0]), 0.))
        with pytest.raises(ss.DisplacementCorrectorFailure):
            analysis.solve(DIRECTION, (targets[1],), **kwargs)
    assert analysis.checkpoint() == saved
    final = analysis.solve(DIRECTION, (targets[1],), **kwargs)
    continued = resumed.solve(DIRECTION, (targets[1],), **kwargs)
    assert analysis.checkpoint() == resumed.checkpoint()
    compare(final, continued)
    if family == 'mixed':
        compare_generalized(final.beam_generalized_material, continued.beam_generalized_material)
    assert any(fsum(row.incremental_dissipation_j) > 0. for row in final.beam_fibre_energy)
    assert not final.production_qualified
    (tmp_path/'fibre-s3-completed.checkpoint').write_bytes(analysis.checkpoint())


def test_legacy_fibre_s3_force_seed_does_not_upgrade_arithmetic(request, monkeypatch, tmp_path):
    definition = fibre_model()
    analysis = owner(request, definition)
    legacy = owner(request, definition)
    zero = (ss.NodalAction('beam', 105, (0., 0., 0., 0., 0., 0.)),)
    # Genuine public reference equilibrium: positive factor, exactly zero action.
    accepted = analysis.solve(zero, (1.,), **OPTIONS)
    assert accepted.free_equilibrium_ratio == 0. and accepted.constraint_max <= 1e-9
    raw = analysis.checkpoint()
    def layout(kind, generation):
        assert kind == 'finite_native_material_stable'
        n = analysis._K.shape[0]
        return (('q', (n,)), ('frames', (len(analysis._beam_ids), 3, 3)),
                ('cells', (len(analysis._beam_parts), 2, 3, 3)),
                ('resultants', (len(analysis._beam_parts), 18)),
                ('multipliers', (analysis._C.shape[0],)), ('pattern', (n,)),
                ('accepted_factors', (generation,))) + shapes(analysis._beam_parts, 'origin') + shapes(analysis._beam_parts, 'accepted')
    _, generation, values = decode(raw, analysis._identity, layout)
    historical = encode('finite_native_material', analysis._identity, generation, tuple(values.items()))
    # Reference zero is mechanically equivalent in both old/new arithmetic.
    legacy.restore(historical)
    assert legacy.checkpoint() == historical and not legacy._finite_driver._stable_s3
    before = legacy.recover()
    import anyfesolver._stiffened_shell_displacement_control as control_module
    def forbidden(*args, **kwargs):
        raise AssertionError('incompatible seed reached checkpoint capture or trial')
    with monkeypatch.context() as patch:
        patch.setattr(legacy, 'checkpoint', forbidden)
        patch.setattr(control_module, 'correct', forbidden)
        with pytest.raises(ValueError, match='seed numerical policy mismatch'):
            legacy.solve(DIRECTION, (.001,), displacement_control=CONTROL, **OPTIONS)
    assert legacy.recover() is before
    assert legacy.checkpoint() == historical and legacy._generation == generation
    assert getattr(legacy, '_displacement_snapshot', None) is None
    (tmp_path/'legacy-reference-seed.checkpoint').write_bytes(historical)
