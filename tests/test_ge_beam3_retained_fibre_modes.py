"""Small physical-fibre current-state spectra; not independent qualification."""
from decimal import localcontext, Decimal as D
import json
import numpy as np
import pytest
from scipy.linalg import eigh

import anyfesolver._ge_beam3_retained_fibre_modes as native
from anyfesolver._ge_beam3_retained_fibre_program import solve_force_program
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_station_resultant_cell import mul, tr, solve, ident
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_retained_fibre import make_model
from test_ge_beam3_retained_fibre_state import program


@pytest.fixture(scope='module', params=[1., 1.e12])
def accepted(request, tmp_path_factory):
    model = make_model(request.param); p = program()
    value = solve_force_program(model, p, stop_after=3)
    root = tmp_path_factory.mktemp('fibre-modal-'+str(request.param))
    (root/'accepted.json').write_bytes(value.checkpoint)
    assert value.status == 'paused', value.failure
    assert any(row[2] > 0 for c in value.state.histories for s in c.stations for row in s.rows)
    inertias = {eid: np.diag([1., 1., 1., .02, .01, .01]) for eid in model.mesh.elements}
    return request.param, model, p, value, inertias


def dense_reference(packet):
    """Ordinary moderate-contrast generalized pencil, no production eigensolver."""
    f = packet.left@packet.right
    k = f.T@f+packet.geometric; m = packet.kinetic.T@packet.kinetic
    algebraic = list(packet.algebraic); dynamic = [i for i in packet.free if i not in algebraic]
    transform = np.zeros((len(k), len(dynamic))); transform[dynamic, :] = np.eye(len(dynamic))
    transform[algebraic, :] = -np.linalg.solve(k[np.ix_(algebraic, algebraic)], k[np.ix_(algebraic, dynamic)])
    return eigh(transform.T@k@transform, transform.T@m@transform, eigvals_only=True)[:6]


def test_two_explicit_policies_and_current_mass(accepted, tmp_path):
    contrast, model, p, value, inertias = accepted; before = value.checkpoint; results = []
    for label, policy in [('frozen', native.FROZEN), ('algorithmic', native.ALGORITHMIC)]:
        packet, result = native.solve_modes(model, p, before, inertias, material_policy=policy,
            bounds=(-1000., 10000.))
        (tmp_path/(label+'.json')).write_bytes(canonical(dict(packet=packet, result=result)))
        assert result.original_ritz_residual <= 1e-11 and result.spectral_residual <= 1e-11
        assert np.all(result.eigenvalues > 0.)
        assert not result.production_qualified and not result.state_advanced and not result.buckling_factor_authorized
        assert not result.certified_intervals and result.negative_eigenvalues_retained
        assert np.all(packet.kinetic[:, packet.algebraic] == 0.)
        assert np.linalg.matrix_rank(packet.kinetic[:, list(set(packet.free)-set(packet.algebraic))]) == 24
        if policy == native.ALGORITHMIC:
            assert result.interpretation == 'LAST_INCREMENT_OPERATOR_ONLY_NOT_PHYSICAL_VIBRATION_AUTHORITY'
        else: assert result.interpretation == 'ELASTIC_PERTURBATIONS_WITH_PLASTIC_COORDINATES_FIXED'
        if contrast == 1.:
            reference = dense_reference(packet)
            assert np.max(abs(reference-result.eigenvalues)/np.maximum(1., abs(reference))) <= 1e-11
        for array in (packet.left, packet.right, packet.kinetic, result.eigenvalues, result.full_modes):
            with pytest.raises(ValueError): array.setflags(write=True)
        results.append(result.eigenvalues)
    assert value.checkpoint == before
    assert np.all(results[1] <= results[0]+1e-10)
    assert np.max(results[0]-results[1]) > 1e-3


def test_cell_factor_matches_original_compliance(accepted):
    _, model, p, value, _ = accepted
    context = native.Context(model, p); state, _ = context.restore(value.checkpoint)
    probe = context.probes[0]; data = probe.cell.response(state.mechanical.resultants[0].tolist(), state.origins[0])
    for policy in (native.FROZEN, native.ALGORITHMIC):
        factor = native.compliance_factor(probe, data, state.origins[0], state.histories[0],
            state.mechanical.resultants[0], policy=policy, check=context.check)
        with localcontext() as ctx:
            ctx.prec = 80
            f = [[D.from_float(float(h))+D.from_float(float(l)) for h, l in zip(row[:18], row[18:])] for row in factor]
            if policy == native.ALGORITHMIC:
                c = [[D.from_float(h)+D.from_float(l) for h, l in zip(row[0], row[1])] for row in data['hessian']]
            else:
                cell = probe.cell; c = mul(tr(cell.load_map), solve([list(row) for row in cell._elastic], [list(row) for row in cell.load_map], context.check))
            residual = mul(mul(tr(f), f), c)
            assert max(abs(x-D(i == j)) for i, row in enumerate(residual) for j, x in enumerate(row)) < D('1e-18')


@pytest.mark.parametrize('incident', ['boundary', 'compliance', 'gradient', 'history', 'gradient_shape', 'stations'])
def test_material_claim_mutations_fail(accepted, incident):
    _, model, p, value, _ = accepted
    context = native.Context(model, p); state, _ = context.restore(value.checkpoint); probe = context.probes[0]
    data = json.loads(canonical(probe.cell.response(state.mechanical.resultants[0].tolist(), state.origins[0])))
    if incident == 'boundary': data['derivative_kind'] = 'SEMISMOOTH_BRANCH_SELECTION'
    elif incident == 'compliance': data['hessian'][0][0][0] += .01
    elif incident == 'gradient': data['gradient'][0][0] += .01
    elif incident == 'history': data['history']['stations'][0]['rows'][0][0] += .01
    elif incident == 'gradient_shape': data['gradient'][0].pop()
    else: data['stations'].pop()
    with pytest.raises(ValueError): native.compliance_factor(probe, data, state.origins[0], state.histories[0],
        state.mechanical.resultants[0], policy=native.ALGORITHMIC, check=context.check)


@pytest.mark.parametrize('incident', ['policy', 'missing', 'boolean', 'negative', 'nan', 'bytes', 'cancel'])
def test_input_and_cancellation_guards(accepted, incident):
    _, model, p, value, inertias = accepted
    data = {i: x.copy() for i, x in inertias.items()}; policy = native.FROZEN
    checkpoint = value.checkpoint; token = CancellationToken()
    if incident == 'policy': policy = 'automatic'
    elif incident == 'missing': data.pop(1)
    elif incident == 'boolean': data[True] = data.pop(1)
    elif incident == 'negative': data[1][0, 0] = -1.
    elif incident == 'nan': data[1][0, 0] = float('nan')
    elif incident == 'bytes': checkpoint = bytearray(checkpoint)
    else: token.cancel()
    with pytest.raises(SolveCancelled if incident == 'cancel' else ValueError):
        native.solve_modes(model, p, checkpoint, data, material_policy=policy,
            bounds=(-1000., 10000.), cancellation_token=token)


@pytest.mark.parametrize('incident', ['cancel', 'model'])
def test_final_publication_guard(accepted, monkeypatch, incident):
    _, model, p, value, inertias = accepted; before = value.checkpoint
    token = CancellationToken(); original = native.solve_factor_chain_modes; x = model.mesh.nodes[5].x
    def altered(*args, **kwargs):
        modes = original(*args, **kwargs)
        if incident == 'cancel': token.cancel()
        else: model.mesh.nodes[5].x += .001
        return modes
    monkeypatch.setattr(native, 'solve_factor_chain_modes', altered)
    try:
        with pytest.raises(SolveCancelled if incident == 'cancel' else ValueError):
            native.solve_modes(model, p, before, inertias, material_policy=native.FROZEN,
                bounds=(-1000., 10000.), cancellation_token=token)
    finally: model.mesh.nodes[5].x = x
    assert value.checkpoint == before


def test_rebuilt_loaded_general_rotation_covariance(accepted, tmp_path):
    from anyfesolver._ge_beam3_p5.algebra import rotation
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
    from anyfesolver._ge_beam3_retained_fibre_element import NativeRetainedFibreElement
    from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
    contrast, model, p, value, inertias = accepted
    transform = rotation([.4, -.3, .2]); other = make_model(contrast)
    for node in other.mesh.nodes.values():
        node.x, node.y, node.z = map(float, transform@node.coords())
    for eid, element in list(other.mesh.elements.items()):
        ref = element.operator.reference
        rotated = Reference(np.array([other.mesh.nodes[n].coords() for n in element.node_ids]),
            np.einsum('ij,njk->nik', transform, ref.nodal_triads))
        made = NativeRetainedFibreElement(eid, tuple(element.node_ids), rotated, element.section, order=4)
        other.mesh.elements[eid] = made; other.materials[made.material_name] = made.section
    other_program = ForceProgram(p.targets, ((5, *map(float, transform@np.array([.4, -.005, 0.]))),), max_iterations=24)
    other_value = solve_force_program(other, other_program, stop_after=3)
    (tmp_path/'rotated-state.json').write_bytes(other_value.checkpoint)
    assert other_value.status == 'paused', other_value.failure
    errors = []
    for label, policy in [('frozen', native.FROZEN), ('algorithmic', native.ALGORITHMIC)]:
        spectra = []
        for case, model_now, program_now, checkpoint in [('E', model, p, value.checkpoint),
                ('R', other, other_program, other_value.checkpoint)]:
            packet, result = native.solve_modes(model_now, program_now, checkpoint, inertias,
                material_policy=policy, bounds=(-1000., 10000.))
            (tmp_path/(label+'-'+case+'.json')).write_bytes(canonical(dict(packet=packet, result=result)))
            spectra.append(result.eigenvalues)
        error = float(np.max(abs(spectra[0]-spectra[1])/np.maximum(1., abs(spectra[0]))))
        errors.append(dict(policy=policy, relative_error=error))
    (tmp_path/'covariance.json').write_bytes(canonical(errors))
    assert max(row['relative_error'] for row in errors) <= 1e-11
