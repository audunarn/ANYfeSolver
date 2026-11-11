"""Small explicit-policy spectra; no plastic vibration/buckling qualification."""
import numpy as np
import pytest
from anyfesolver._ge_beam3_retained_plastic_program import solve_force_program
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._native_factor_chain_modes import solve_factor_chain_modes
from docs.reference_cases.ge_beam3_paired_plastic_modal_probe import prepare, FROZEN, ALGORITHMIC
from test_ge_beam3_retained_plastic_state import program
from test_ge_beam3_retained_plastic_force_probe import model_with_plasticity
from test_ge_beam3_curved_contrast_probe import make_curved, ROTATION
from test_ge_beam3_complementary_section import section
from anyfesolver._ge_beam3_p5_seeded import NativeP5BeamElement
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram


@pytest.fixture(scope='module', params=[1., 1000000000000.])
def accepted(request):
    model = model_with_plasticity(request.param); p = program()
    value = solve_force_program(model, p, stop_after=3)
    assert value.status == 'paused', value.failure
    assert any(np.any(x[:, 2] > 0) for x in value.state.histories)
    inertias = {eid:np.diag([1., 1., 1., .02, .01, .01]) for eid in model.mesh.elements}
    return request.param, model, p, value, inertias


def test_current_state_spectra_have_explicit_material_policies(accepted, tmp_path):
    contrast, model, p, value, inertias = accepted; saved = value.checkpoint
    results = []
    for label, policy in (('frozen', FROZEN), ('algorithmic', ALGORITHMIC)):
        packet = prepare(model, p, saved, inertias, policy=policy)
        (tmp_path/(label+'-packet.json')).write_bytes(canonical(packet))
        try:
            modes = solve_factor_chain_modes(packet.left, packet.right, packet.geometric, packet.kinetic,
                packet.free, packet.algebraic, bounds=(-1000., 10000.), num_modes=6, root_width=1e-10)
        except Exception as error:
            (tmp_path/(label+'-failure.json')).write_bytes(canonical(dict(error=type(error).__name__,
                message=str(error), contrast=contrast, material_policy=policy, production_qualified=False)))
            raise
        (tmp_path/(label+'-modes.json')).write_bytes(canonical(modes))
        assert packet.material_policy == policy and not packet.production_qualified
        assert np.all(modes.eigenvalues > 0.)
        assert modes.original_ritz_residual <= 1e-11 and modes.spectral_residual <= 1e-11
        assert np.all(packet.kinetic[:, packet.algebraic] == 0.)
        assert saved == value.checkpoint
        results.append(modes.eigenvalues)
    # Freezing plastic coordinates cannot be softer than the smooth
    # accepted-increment tangent for this convex hardening model.
    assert np.all(results[1] <= results[0]+1e-10)
    assert np.max(results[0]-results[1]) > 1e-3


def test_spectral_policy_and_inertia_inputs_fail_closed(accepted):
    _, model, p, value, inertias = accepted
    with pytest.raises(ValueError, match='explicit spectral policy'):
        prepare(model, p, value.checkpoint, inertias, policy='automatic')
    with pytest.raises(ValueError, match='one explicit section inertia'):
        prepare(model, p, value.checkpoint, {}, policy=FROZEN)
    bad = {eid:mass.copy() for eid, mass in inertias.items()}; bad[1][0, 0] = -1.
    with pytest.raises(ValueError): prepare(model, p, value.checkpoint, bad, policy=FROZEN)


@pytest.mark.parametrize('transform', [ROTATION, rotation([.4, -.3, .2])], ids=['R90', 'general'])
def test_rebuilt_loaded_spectral_covariance(accepted, transform, tmp_path):
    contrast, original, p, original_value, inertias = accepted
    model, _ = make_curved(100., rotation=transform)
    for eid, old in list(model.mesh.elements.items()):
        new = NativeP5BeamElement(eid, old.node_ids, old.core.reference, section(contrast), line_force=np.zeros(3))
        model.mesh.elements[eid] = new; model.materials[new.material_name] = new.core.section
    force = (transform@np.array([.4, -.005, 0.])).tolist()
    other_program = ForceProgram(p.targets, ((5, *force),), max_iterations=24)
    other_value = solve_force_program(model, other_program, stop_after=3)
    (tmp_path/'rotated-state.json').write_bytes(other_value.checkpoint)
    assert other_value.status == 'paused', other_value.failure
    records = []
    for label, policy in (('frozen', FROZEN), ('algorithmic', ALGORITHMIC)):
        results = []
        for case, current_model, current_program, checkpoint in (
            ('E', original, p, original_value.checkpoint), ('R', model, other_program, other_value.checkpoint)):
            packet = prepare(current_model, current_program, checkpoint, inertias, policy=policy)
            (tmp_path/(label+'-'+case+'-packet.json')).write_bytes(canonical(packet))
            try:
                modes = solve_factor_chain_modes(packet.left, packet.right, packet.geometric, packet.kinetic,
                    packet.free, packet.algebraic, bounds=(-1000., 10000.), num_modes=6, root_width=1e-10)
            except Exception as error:
                (tmp_path/(label+'-'+case+'-failure.json')).write_bytes(canonical(dict(error=type(error).__name__, message=str(error))))
                raise
            (tmp_path/(label+'-'+case+'-modes.json')).write_bytes(canonical(modes))
            results.append(modes.eigenvalues)
        error = float(np.max(np.abs(results[0]-results[1])/np.maximum(1., np.abs(results[0]))))
        records.append(dict(policy=policy, relative_eigenvalue_error=error))
    (tmp_path/'covariance.json').write_bytes(canonical(records))
    assert max(row['relative_eigenvalue_error'] for row in records) <= 1e-11
