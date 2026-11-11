"""Native explicit-policy spectral port, provenance, and cancellation guards."""
import numpy as np
import pytest
import json
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._native_factor_chain_modes import solve_factor_chain_modes
import anyfesolver._ge_beam3_retained_plastic_modes as native
from docs.reference_cases.ge_beam3_paired_plastic_modal_probe import prepare, FROZEN, ALGORITHMIC
from test_ge_beam3_paired_plastic_modal_probe import accepted


@pytest.mark.parametrize('policy', [FROZEN, ALGORITHMIC])
def test_native_spectral_packet_and_roots_match_probe(accepted, policy, tmp_path):
    contrast, model, p, value, inertias = accepted; before = value.checkpoint
    reference = prepare(model, p, before, inertias, policy=policy)
    expected = solve_factor_chain_modes(reference.left, reference.right, reference.geometric, reference.kinetic,
        reference.free, reference.algebraic, bounds=(-1000., 10000.), num_modes=6, root_width=1e-10)
    packet, result = native.solve_modes(model, p, before, inertias,
        material_policy=policy, bounds=(-1000., 10000.))
    assert canonical(packet) == canonical(reference)
    for field in ('eigenvalues', 'full_modes', 'numerical_brackets'):
        np.testing.assert_array_equal(getattr(result, field), getattr(expected, field))
        with pytest.raises(ValueError): getattr(result, field).setflags(write=True)
    assert result.material_policy == policy and result.operator_identity == packet.identity
    assert not result.state_advanced and not result.production_qualified
    assert not result.buckling_factor_authorized and not result.certified_intervals
    if policy == ALGORITHMIC: assert result.interpretation == 'LAST_INCREMENT_OPERATOR_ONLY_NOT_PHYSICAL_VIBRATION_AUTHORITY'
    else: assert result.interpretation == 'ELASTIC_PERTURBATIONS_WITH_PLASTIC_COORDINATES_FIXED'
    assert value.checkpoint == before
    (tmp_path/'native-spectrum.json').write_bytes(canonical(dict(contrast=contrast, packet=packet, result=result)))


def test_policy_is_required_and_capture_cancellation_preserves_state(accepted):
    _, model, p, value, inertias = accepted; before = value.checkpoint
    with pytest.raises(TypeError): native.solve_modes(model, p, before, inertias, bounds=(-1000., 10000.))
    token = CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled): native.solve_modes(model, p, before, inertias,
        material_policy=FROZEN, bounds=(-1000., 10000.), cancellation_token=token)
    assert value.checkpoint == before


@pytest.mark.parametrize('incident', ['cancel', 'model_mutation'])
def test_kernel_exit_is_guarded_before_returning_spectrum(accepted, monkeypatch, incident):
    _, model, p, value, inertias = accepted; before = value.checkpoint; x = model.mesh.nodes[5].x
    token = CancellationToken(); original = native.solve_factor_chain_modes
    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        if incident == 'cancel': token.cancel()
        else: model.mesh.nodes[5].x += .001
        return result
    monkeypatch.setattr(native, 'solve_factor_chain_modes', altered)
    try:
        with pytest.raises(SolveCancelled if incident == 'cancel' else ValueError):
            native.solve_modes(model, p, before, inertias, material_policy=FROZEN,
                bounds=(-1000., 10000.), cancellation_token=token)
    finally: model.mesh.nodes[5].x = x
    assert value.checkpoint == before


def test_boolean_inertia_id_is_not_a_model_identity(accepted):
    _, model, p, value, inertias = accepted
    bad = {True:inertias[1], 2:inertias[2]}
    with pytest.raises(ValueError, match='one explicit section inertia'):
        native.solve_modes(model, p, value.checkpoint, bad, material_policy=FROZEN, bounds=(-1000., 10000.))


@pytest.mark.parametrize('mutation', ['yield_boundary', 'compliance'])
def test_algorithmic_policy_rejects_nonunique_or_mismatched_tangent(accepted, mutation):
    _, model, p, value, _ = accepted
    context = native.Context(model, p); state, _ = context.restore(value.checkpoint)
    probe = context.probes[0]; mechanical = state.mechanical; nodes = context.layout.nodes[0]
    response = probe.evaluate(mechanical.positions[nodes], mechanical.position_low[nodes],
        mechanical.nodal_frames[nodes], mechanical.cell_rotations[0], mechanical.resultants[0],
        origins=state.origins[0].tolist())
    data = json.loads(response.material)
    if mutation == 'yield_boundary': data['derivative_kind'] = 'SEMISMOOTH_ELASTIC_SELECTION'
    else: data['hessian'][0][0][0] += .001
    with pytest.raises(ValueError, match='yield-boundary|compliance differs'):
        native.compliance_factor(probe, data, state.histories[0], mechanical.resultants[0],
            policy=ALGORITHMIC, check=context.guard)
