"""V5 seed correction: preserve V4 defect evidence and actual state safety."""

import json
import numpy as np
import pytest
from anyfesolver._ge_beam3_authoritative_cell_seed import cell_initial_rotations
from anyfesolver._ge_beam3_p5.algebra import rotation, log_rotation
from anyfesolver._ge_beam3_p5_loads import NativeP5BeamElement as V4
from anyfesolver._ge_beam3_p5_seeded import NativeP5BeamElement as V5
from anyfesolver._ge_beam3_p5_loads.core import canonical
from anyfesolver._ge_beam3_signed_loaded_modes import solve_signed_loaded_modes as old_modes
from anyfesolver._ge_beam3_seeded_signed_modes import solve_signed_loaded_modes
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram, solve_force_program
from test_ge_beam3_curved_contrast_probe import make_curved


def test_identity_and_common_authoritative_operators_are_preserved_exactly():
    for q in (np.eye(3), rotation([.2, -.3, .4])):
        operators = np.tile(q, (3, 1, 1)); saved = operators.copy()
        seed = cell_initial_rotations(operators)
        np.testing.assert_array_equal(seed, np.tile(q, (2, 1, 1)))
        np.testing.assert_array_equal(operators, saved)
        with pytest.raises(ValueError): seed.setflags(write=True)


def test_noncommuting_operator_midpoints_retain_published_geodesic_identity():
    q = np.array([rotation([.2, -.1, .3]), rotation([-.1, .2, .05]), rotation([.1, .1, -.2])])
    seed = cell_initial_rotations(q)
    for i in (0, 1):
        np.testing.assert_allclose(log_rotation(q[i].T@seed[i]), log_rotation(seed[i].T@q[i+1]),
            rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('case', ['reflection', 'nonfinite', 'large_relative'])
def test_bad_authoritative_rotations_fail_closed(case):
    q = np.tile(np.eye(3), (3, 1, 1))
    if case == 'reflection': q[0, 0, 0] = -1.
    if case == 'nonfinite': q[0, 0, 0] = np.nan
    if case == 'large_relative': q[1] = rotation([0., 0., .95*np.pi])
    with pytest.raises(ValueError): cell_initial_rotations(q)


@pytest.mark.parametrize('slenderness', [10000., 1000000.])
def test_preserved_v4_reference_defect_is_not_reclassified_as_qualification(slenderness, tmp_path):
    model, inertias = make_curved(slenderness, element_type=V4)
    states = {i:e.init_model_bound_nonlinear_state(model.mesh,e.core.section,1) for i,e in model.mesh.elements.items()}
    before = canonical(states)
    with pytest.raises(ValueError, match='loaded free equilibrium required'):
        old_modes(model, states, np.zeros(30), inertias, np.zeros(30),
            load_parameter=0., bounds=(-10000., 100000000.), num_modes=6)
    from anyfesolver._ge_beam3_loaded_modal import prepare
    packet, _ = prepare(model, states, np.zeros(30), inertias, load_parameter=0.)
    assert np.linalg.norm(packet.net_residual[list(packet.free_dofs)]) > 1e-11
    assert canonical(states) == before
    with (tmp_path/'v4-defect.json').open('xb') as stream:
        stream.write(canonical(dict(slenderness=slenderness, free_residual=packet.net_residual[list(packet.free_dofs)],
            terminal='V4_REFERENCE_SEED_DEFECT_REPRODUCED', production_qualified=False)))


@pytest.mark.parametrize('slenderness', [100., 10000., 1000000.])
def test_v5_reference_is_exactly_stress_free_without_force_clipping(slenderness):
    model, _ = make_curved(slenderness)
    for e in model.mesh.elements.values():
        state = e.init_model_bound_nonlinear_state(model.mesh, e.core.section, 1)
        response = state['material_state']['response']
        assert response.iterations == 0 and response.evaluations == 1
        np.testing.assert_array_equal(response.residual, np.zeros(18))
        np.testing.assert_array_equal(response.local_rotations, np.tile(np.eye(3), (2, 1, 1)))
        np.testing.assert_array_equal(response.moments, np.zeros((2, 2, 3)))
        for station in response.stations:
            np.testing.assert_array_equal(station.response.strain, np.zeros(6))
            np.testing.assert_array_equal(station.response.resultants, np.zeros(6))


def test_v4_v5_hot_state_exchange_is_rejected():
    old, _ = make_curved(100., element_type=V4); new, _ = make_curved(100.)
    for a, b in ((old, new), (new, old)):
        e = a.mesh.elements[1]; other = b.mesh.elements[1]
        state = e.init_model_bound_nonlinear_state(a.mesh, e.core.section, 1)
        encoded = e.serialize_native_material_state(a.mesh, state)
        with pytest.raises(ValueError): other.validate_model_bound_nonlinear_state(b.mesh, other.core.section, encoded, 1)


def test_v5_finite_loaded_equilibrium_and_serialized_replay(tmp_path, monkeypatch):
    model, inertias = make_curved(100.)
    program = ForceProgram((.5, 1.), ((5, .05, -.001, 0.),))
    with (tmp_path/'progress.jsonl').open('x') as log:
        from anyfesolver import _ge_beam3_seeded_load_program as driver
        original = driver._physical_force
        def measured(*args):
            force = original(*args)
            log.write(json.dumps(dict(stage='diagnostic.physical_force',
                free_norm=float(np.linalg.norm(force[6:]))), sort_keys=True)+'\n'); log.flush()
            return force
        monkeypatch.setattr(driver, '_physical_force', measured)
        def progress(row):
            log.write(json.dumps(row, sort_keys=True)+'\n'); log.flush()
        result = solve_force_program(model, program, progress=progress)
    monkeypatch.setattr(driver, '_physical_force', original)
    with (tmp_path/'state.json').open('xb') as stream: stream.write(result.checkpoint)
    with (tmp_path/'outcome.json').open('xb') as stream:
        stream.write(canonical(dict(status=result.status, failure=result.failure,
            completed_targets=result.completed_targets, production_qualified=False)))
    assert result.status == 'completed', result.failure
    states = {r['element_id']:r['state'] for r in json.loads(result.checkpoint)['element_states']}
    saved = canonical(states); force = np.zeros(30); force[24:27] = [.05, -.001, 0.]
    packet, modes = solve_signed_loaded_modes(model, states, result.displacements, inertias, force,
        load_parameter=1., bounds=(-10000., 100000000.), num_modes=6)
    clone, clone_inertias = make_curved(100.)
    again = solve_force_program(clone, program, checkpoint=result.checkpoint)
    assert again.status == 'completed' and again.checkpoint == result.checkpoint
    replay = solve_signed_loaded_modes(clone, states, again.displacements, clone_inertias, force,
        load_parameter=1., bounds=(-10000., 100000000.), num_modes=6)
    assert canonical(replay) == canonical((packet, modes))
    assert canonical(states) == saved
