"""Native control-history spectra; small development, not qualification."""
from hashlib import sha256
import json
import numpy as np
import pytest

import anyfesolver._ge_beam3_controlled_fibre_modes as native
import anyfesolver._ge_beam3_seeded_fibre_control as control
from anyfesolver._ge_beam3_retained_fibre_state import Layout
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_retained_fibre_control import make_model, program, arch
from test_ge_beam3_retained_fibre_modes import dense_reference


@pytest.fixture(scope='module', params=[1., 1.e12])
def accepted(request, tmp_path_factory):
    model = make_model(request.param); p = program()
    result = control.solve_translation_program(model, p, stop_after=2)
    root = tmp_path_factory.mktemp('controlled-modal-'+str(request.param))
    (root/'accepted.json').write_bytes(result.checkpoint)
    assert result.status == 'paused', result.failure
    inertias = {eid: np.diag([1., 1., 1., .02, .01, .01]) for eid in model.mesh.elements}
    return request.param, model, p, result, inertias


def test_native_history_spectra_hold_load_not_control_displacement(accepted, tmp_path):
    contrast, model, p, accepted_state, inertias = accepted
    before = accepted_state.checkpoint; results = []
    for label, policy in [('frozen', native.FROZEN), ('algorithmic', native.ALGORITHMIC)]:
        packet, result = native.solve_modes(model, p, before, inertias, material_policy=policy,
            bounds=(-1000., 10000.), expected_checkpoint_sha256=sha256(before).hexdigest())
        (tmp_path/(label+'.json')).write_bytes(canonical(dict(packet=packet, result=result)))
        assert result.equilibrium_load_parameter == accepted_state.state.parameter
        assert result.perturbation_load_policy == 'FIXED_ACCEPTED_SPATIAL_DEAD_LOAD_PATTERN'
        assert not result.continuation_constraint_retained and not result.checkpoint_converted
        assert not result.state_advanced and not result.production_qualified
        assert not result.buckling_factor_authorized and result.negative_eigenvalues_retained
        assert result.original_ritz_residual <= 1e-11 and result.spectral_residual <= 1e-11
        context = control.Context(model, p)
        slots = range(6*context.node, 6*context.node+3)
        assert all(slot in packet.free for slot in slots)
        assert np.any(packet.kinetic[:, list(slots)])
        assert np.all(packet.kinetic[:, packet.algebraic] == 0.)
        if contrast == 1.:
            reference = dense_reference(packet)
            assert np.max(abs(result.eigenvalues-reference)/np.maximum(1., abs(reference))) <= 1e-11
        results.append(result.eigenvalues)
    assert before == accepted_state.checkpoint
    assert np.max(results[0]-results[1]) > 1e-3


def test_spectral_replay_never_advances_or_initializes_history(accepted, monkeypatch):
    _, model, p, value, inertias = accepted
    def forbidden(*args, **kwargs):
        raise AssertionError('spectra cannot advance or initialize the nonlinear path')
    monkeypatch.setattr(control.Context, 'initialize', forbidden)
    monkeypatch.setattr(Layout, 'advance', forbidden)
    packet, guard, parameter = native.prepare(model, p, value.checkpoint, inertias,
        material_policy=native.FROZEN)
    guard()
    assert parameter == value.state.parameter and not packet.production_qualified


@pytest.mark.parametrize('incident', ['hash', 'policy', 'old_controller', 'cancel'])
def test_control_authority_guards(accepted, incident):
    _, model, p, value, inertias = accepted
    expected = sha256(value.checkpoint).hexdigest()
    policy = native.FROZEN; raw = value.checkpoint; token = CancellationToken()
    if incident == 'hash':
        expected = '0'*64
    elif incident == 'policy':
        policy = 'automatic'
    elif incident == 'old_controller':
        from anyfesolver._ge_beam3_retained_fibre_control import Context as Old
        raw = Old(model, p).checkpoint(()); expected = sha256(raw).hexdigest()
    else:
        token.cancel()
    with pytest.raises(SolveCancelled if incident == 'cancel' else ValueError):
        native.prepare(model, p, raw, inertias, material_policy=policy,
            expected_checkpoint_sha256=expected, cancellation_token=token)


@pytest.mark.parametrize('incident', ['cancel', 'model'])
def test_final_guard_after_spectrum_rejects_mutation(accepted, monkeypatch, incident):
    _, model, p, value, inertias = accepted
    original = native.solve_factor_chain_modes; token = CancellationToken()
    position = model.mesh.nodes[5].x
    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        if incident == 'cancel':
            token.cancel()
        else:
            model.mesh.nodes[5].x += .001
        return result
    monkeypatch.setattr(native, 'solve_factor_chain_modes', altered)
    try:
        with pytest.raises(SolveCancelled if incident == 'cancel' else ValueError):
            native.solve_modes(model, p, value.checkpoint, inertias, material_policy=native.FROZEN,
                bounds=(-1000., 10000.), cancellation_token=token)
    finally:
        model.mesh.nodes[5].x = position


def test_fixed_load_arch_spectra_retain_descending_branch_instability(tmp_path):
    model = arch()
    p = control.TranslationProgram((.01, .025, .04, .055, .075, .1, .15, .2),
        3, (0., -1., 0.), ((3, 0., -1., 0.),))
    value = control.solve_translation_program(model, p)
    (tmp_path/'arch.json').write_bytes(value.checkpoint)
    assert value.status == 'completed', value.failure
    context = control.Context(model, p); _, records = context.restore(value.checkpoint)
    inertias = {eid: np.diag([1., 1., 1., .02, .01, .01]) for eid in model.mesh.elements}
    negative = []
    for cursor in (1, 6):
        raw = context.checkpoint(records[:cursor])
        packet, result = native.solve_modes(model, p, raw, inertias, material_policy=native.FROZEN,
            bounds=(-1.e6, 1.e8))
        (tmp_path/('spectrum-'+str(cursor)+'.json')).write_bytes(canonical(dict(packet=packet, result=result)))
        reference = dense_reference(packet)
        assert np.max(abs(reference-result.eigenvalues)/np.maximum(1., abs(reference))) <= 1e-11
        assert result.original_ritz_residual <= 1e-11
        negative.append(int(np.count_nonzero(result.eigenvalues < 0.)))
        assert not result.continuation_constraint_retained and not result.buckling_factor_authorized
    assert negative[0] == 0 and negative[1] > 0
