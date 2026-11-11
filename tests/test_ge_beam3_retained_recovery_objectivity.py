"""Complementary recovery work and fixed-reference rigid-motion checks."""
import numpy as np
import pytest
from anyfesolver._ge_beam3_retained_force_program import ForceProgram, solve_force_program
from anyfesolver._ge_beam3_retained_state import Context
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_curved_contrast_probe import make_curved


@pytest.fixture(scope='module', params=[100., 1000000.])
def finite(request):
    model, _ = make_curved(request.param)
    program = ForceProgram((1., 2., 4., 8., 16.), ((5, .05, -.001, 0.),))
    result = solve_force_program(model, program)
    assert result.status == 'completed', result.failure
    return request.param, Context(model, program), result.state


def test_recovered_resultants_and_energy_match_complementary_work(finite, tmp_path):
    slenderness, context, state = finite
    recovered = context.recover(state); metrics = []
    for i, (probe, row) in enumerate(zip(context.probes, recovered)):
        p = state.resultants[i]; dual = probe.compliance@p
        integrated = np.zeros(18); energy = 0.; force_error = 0.
        c = context.elements[i][1].core.section._elastic
        for station in row['stations']:
            cell = station['cell']; xi = station['xi']; t = xi-cell+1
            weight = station['measure']; ref = probe.reference
            v = ref.frame(xi).T/ref.jacobian(xi)
            strain = station['strain']; stress = station['resultants']
            integrated[3*cell:3*cell+3] += weight*v.T@stress[:3]
            integrated[6+6*cell:9+6*cell] += weight*(1-t)*strain[3:]
            integrated[9+6*cell:12+6*cell] += weight*t*strain[3:]
            energy += .5*weight*(strain@stress)
            force_error = max(force_error, float(np.linalg.norm(c@strain-stress)))
            np.testing.assert_allclose(station['current_frame'],
                state.cell_rotations[i, cell]@station['reference_frame'], rtol=0, atol=0)
            assert 'fibre_stress' not in station and 'plastic_history' not in station
        force_integral_error = float(np.linalg.norm(integrated[:6]-p[:6]))
        curvature_integral_error = float(np.linalg.norm(integrated[6:]-dual[6:]))
        energy_error = float(abs(energy-.5*p@dual))
        metrics.append(dict(element=row['element_id'], force_integral_error=force_integral_error,
            curvature_integral_error=curvature_integral_error, section_law_error=force_error,
            energy_error=energy_error))
    with (tmp_path/'recovery.json').open('xb') as stream:
        stream.write(canonical(dict(slenderness=slenderness, metrics=metrics, recovery=recovered,
            production_qualified=False)))
    print(canonical(metrics).decode(), flush=True)
    assert max(max(row[k] for k in ('force_integral_error', 'curvature_integral_error',
                                   'section_law_error', 'energy_error')) for row in metrics) <= 1e-11


def test_fixed_reference_finite_state_objectivity(finite, tmp_path):
    slenderness, context, state = finite
    q = rotation([2.6, .8, -.3]); translation = np.array([1000., -2000., 3000.])
    x = np.empty_like(state.positions); low = np.empty_like(x)
    for node in range(len(x)):
        for axis in range(3):
            terms = [float(translation[axis])]
            for j in range(3):
                terms.extend((float(q[axis, j]*state.positions[node, j]),
                              float(q[axis, j]*state.position_low[node, j])))
            x[node, axis], low[node, axis] = split_sum(terms)
    transform = np.eye(42); transform[:24, :24] = np.kron(np.eye(8), q)
    errors = []
    for i, probe in enumerate(context.probes):
        nodes = context.nodes[i]; p = state.resultants[i]
        first = probe.evaluate(state.positions[nodes], state.position_low[nodes],
            state.nodal_frames[nodes], state.cell_rotations[i], p)
        second = probe.evaluate(x[nodes], low[nodes], q@state.nodal_frames[nodes], q@state.cell_rotations[i], p)
        k_error = float(np.linalg.norm(first.kinematics-second.kinematics))
        r_error = float(np.linalg.norm(transform@first.residual-second.residual))
        h_error = float(np.linalg.norm(transform@first.hessian@transform.T-second.hessian)/max(1., np.linalg.norm(first.hessian)))
        energy_error = abs(first.potential-second.potential)
        errors.append(dict(kinematics=k_error, residual=r_error, hessian=h_error, energy=energy_error))
        recovered = probe.recover(q@state.cell_rotations[i], p)
        original = probe.recover(state.cell_rotations[i], p)
        for a, b in zip(original, recovered):
            np.testing.assert_array_equal(a['strain'], b['strain'])
            np.testing.assert_array_equal(a['resultants'], b['resultants'])
            np.testing.assert_allclose(b['current_frame'], q@a['current_frame'], rtol=1e-11, atol=1e-11)
    with (tmp_path/'objectivity.json').open('xb') as stream:
        stream.write(canonical(dict(slenderness=slenderness, errors=errors, production_qualified=False)))
    assert max(max(row.values()) for row in errors) <= 1e-11


def test_private_operator_port_preserves_probe_at_nonzero_chart(finite):
    from docs.reference_cases.ge_beam3_full_resultant_legendre_probe import ElasticResultantProbe
    _, context, state = finite
    increment = np.sin(np.arange(42))*.001
    for i, probe in enumerate(context.probes):
        old = ElasticResultantProbe(context.elements[i][1]); nodes = context.nodes[i]
        args = (state.positions[nodes], state.position_low[nodes], state.nodal_frames[nodes],
                state.cell_rotations[i], state.resultants[i])
        first = old.evaluate(*args, increment=increment)
        second = probe.evaluate(*args, increment=increment)
        assert first.potential == second.potential
        for field in ('residual', 'hessian', 'kinematics'):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
