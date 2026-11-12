"""Real shell/beam residual boundary, not accepted coupled-history qualification."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
import numpy as np
import pytest

from anyfesolver._ge_beam3_shell_joint_trial import ShellBeamTrialAssembly
from anyfesolver._ge_beam3_retained_generalized_state import Context, Program
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver.corotational import corotational_element_response
from test_ge_beam3_native_generalized_modal import make


def specimen(topology='Q4', curved=False):
    model, _, _ = make(curved, clamped=True)
    owner = Context(model, Program((1.,), DistributedPattern(LinePattern(()), ())))
    coordinates = np.array([[0., 0., .2], [1., 0., .2], [1., 1., .2], [0., 1., .2]])
    if topology == 'S3-V2D': coordinates = coordinates[[0, 1, 2]]
    made = ShellBeamTrialAssembly(owner, topology=topology, coordinates=coordinates,
        reference_normal=np.array([0., 0., 1.]), thickness=.2, elastic_modulus=1000.,
        poisson_ratio=.3, shell_node=2, beam_node=3)
    return made


def norm_error(actual, expected):
    return float(np.linalg.norm(actual-expected)/max(1., np.linalg.norm(expected)))


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
@pytest.mark.parametrize('curved', (False, True))
def test_real_components_and_spatial_row_additive_column_derivative(topology, curved, tmp_path):
    made = specimen(topology, curved); c = made.beam
    before = canonical(c.initial)
    beam_step = .01*np.sin(np.arange(c.count)+.7); beam_step[:6] = 0.
    mechanical = c.advance(c.initial.mechanical, beam_step)
    u = .02*np.cos(np.arange(made.shell_count)+.4)
    # Nontrivial additive angle makes a mistaken chart-covector pullback visible.
    u.reshape(-1, 6)[:, 3:] += np.array([.7, -.3, .2])
    force = np.array([.1, -.2, .3, .02, .04, -.03])
    result = made.evaluate(c.initial, mechanical, u, force, parameter=0.)
    br, bj, _, _, _ = c.assemble(mechanical, 0., c.initial.histories)
    sr, sj, _ = corotational_element_response(made.model, 1, made.element, u, True,
        committed_state=made.shell_origin, num_layers=3, tangent_mode='consistent')
    r = result.residual.copy(); r[list(made.slots)] -= result.joint_residual
    j = result.tangent.copy(); j[np.ix_(made.slots, made.slots)] -= result.joint_tangent
    assert norm_error(r, np.r_[sr, br, np.zeros(6)]) < 1e-13
    assert norm_error(j[:made.shell_count, :made.shell_count], sj) < 1e-13
    assert norm_error(j[made.shell_count:-6, made.shell_count:-6], bj) < 1e-13
    wrench = result.joint_residual[:12].reshape(2, 6)
    distance = (mechanical.positions[c.index[made.beam_node]]
        -made.coordinates[made.shell_node]-u.reshape(-1, 6)[made.shell_node, :3])
    assert np.linalg.norm(wrench[0, :3]+wrench[1, :3]) < 1e-11
    assert np.linalg.norm(wrench[0, 3:]+wrench[1, 3:]+np.cross(distance, wrench[1, :3])) < 1e-11
    direction = np.sin(np.arange(made.count)+.2)
    direction[made.shell_count:made.shell_count+6] = 0.
    eps = 2e-6
    def probe(sign):
        changed = c.advance(mechanical, sign*eps*direction[made.shell_count:-6])
        return made.evaluate(c.initial, changed, u+sign*eps*direction[:made.shell_count],
            force+sign*eps*direction[-6:], parameter=0.).residual
    derivative = (probe(1)-probe(-1))/(2*eps)
    error = norm_error(result.tangent@direction, derivative)
    assert error < 1e-7, error
    assert canonical(c.initial) == before and canonical(made.shell_origin) == made.origin_bytes
    assert not result.production_qualified and not result.coupled_state_committed
    assert not hasattr(made, 'commit') and not hasattr(made, 'checkpoint')
    assert not result.residual.flags.writeable and not result.tangent.flags.writeable
    saved = asdict(result)
    for key in ('shell_candidate', 'beam_histories'): saved[key] = json.loads(saved[key])
    (tmp_path/'trial.json').write_bytes(canonical(dict(topology=topology, curved=curved,
        result=saved, directional_error=error, predecessor_unchanged=True)))


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
@pytest.mark.parametrize('curved', (False, True))
def test_bounded_first_increment_coupled_newton_diagnostic(topology, curved, tmp_path):
    made = specimen(topology, curved); c = made.beam
    before = canonical(c.initial)
    u = np.zeros(made.shell_count); mechanical = c.initial.mechanical; lm = np.zeros(6)
    # Both submodels have genuine supports; no endpoint displacement is imposed
    # in place of coupling. Load crosses the six-multiplier eccentric joint.
    shell_fixed = set(range(6))
    if topology == 'Q4': shell_fixed.update(range(18, 24))
    free = np.array([i for i in range(made.shell_count) if i not in shell_fixed] +
        [made.shell_count+int(i) for i in c.free] + list(range(made.count-6, made.count)))
    applied = np.zeros(made.count); applied[8] = .0002
    rows = []
    for iteration in range(9):
        result = made.evaluate(c.initial, mechanical, u, lm, parameter=0.)
        residual = result.residual-applied
        error = float(np.linalg.norm(residual[free]))
        rows.append(error)
        if error < 1e-11: break
        assert iteration < 8, rows
        step = np.zeros(made.count)
        step[free] = np.linalg.solve(result.tangent[np.ix_(free, free)], -residual[free])
        mechanical = c.advance(mechanical, step[made.shell_count:-6])
        u += step[:made.shell_count]; lm += step[-6:]
    assert np.linalg.norm(result.constraints) < 1e-11
    assert np.linalg.norm(lm) > 1e-8
    assert np.linalg.norm(mechanical.positions-c.initial.mechanical.positions) > 1e-8
    assert canonical(c.initial) == before and canonical(made.shell_origin) == made.origin_bytes
    # Converged diagnostic data is NOT committed material or restart evidence.
    assert not result.coupled_state_committed
    (tmp_path/'newton.json').write_bytes(canonical(dict(topology=topology, curved=curved,
        residuals=rows, shell_u=u, beam=mechanical, multipliers=lm,
        constraints=result.constraints, coupled_state_committed=False)))


@pytest.mark.parametrize('kind', ('foreign', 'forged', 'origin', 'node', 'thickness',
    'material', 'slots', 'frame', 'extent', 'parameter', 'cancel', 'busy'))
def test_owner_guard_failure_and_cancellation_never_commit(kind):
    made = specimen(); c = made.beam; accepted = c.initial
    before = canonical(accepted); token = None; parameter = 0.
    if kind == 'foreign': accepted = specimen().beam.initial
    elif kind == 'forged': accepted = replace(accepted)
    elif kind == 'origin': made.shell_origin['unregistered'] = 1
    elif kind == 'node': made.model.mesh.nodes[1].x += .01
    elif kind == 'thickness': made.element.thickness *= 2
    elif kind == 'material': made.model.get_material('joint-shell').elastic_modulus *= 2
    elif kind == 'slots': made.slots = tuple(reversed(made.slots))
    elif kind == 'frame': made.frame = made.frame.T
    elif kind == 'extent': made.count += 6
    elif kind == 'parameter': parameter = float('nan')
    elif kind == 'cancel': token = CancellationToken(); token.cancel()
    elif kind == 'busy': made._lock.acquire()
    if kind == 'frame': made.frame = np.diag([-1., -1., 1.])
    try:
        with pytest.raises((ValueError, RuntimeError, SolveCancelled)):
            made.evaluate(accepted, c.initial.mechanical, np.zeros(made.shell_count), np.zeros(6),
                parameter=parameter, cancellation_token=token)
    finally:
        if kind == 'busy': made._lock.release()
    assert not made._lock.locked()
    assert canonical(c.initial) == before


def test_unsupported_shell_and_beam_owner_rejected():
    made = specimen()
    args = dict(topology='Q4', coordinates=made.coordinates, reference_normal=made.normal,
        thickness=.2, elastic_modulus=1000., poisson_ratio=.3, shell_node=2, beam_node=3)
    with pytest.raises(ValueError): ShellBeamTrialAssembly(object(), **args)
    args['topology'] = 'legacy'
    with pytest.raises(ValueError): ShellBeamTrialAssembly(made.beam, **args)


def test_genuine_curved_plastic_beam_predecessor_and_rejected_joint_trial(tmp_path):
    from anyfesolver import _ge_beam3_retained_nodal_loading as nodal
    from test_ge_beam3_native_generalized_restart import make as plastic_model, pattern
    model = plastic_model('curved-plastic')
    p = nodal.Program((.25, .5, 1.), pattern(model), nodal.NodalDeadForces(()))
    solved = nodal.solve(model, p)
    assert solved.status == 'completed', solved.failure
    c = nodal.Context(model, p)
    accepted, records = c.restore(solved.checkpoint, expected_sha256=sha256(solved.checkpoint).hexdigest())
    assert any(sum(station.accumulated) > 0. for cell in accepted.histories for station in cell.stations)
    made = ShellBeamTrialAssembly(c, topology='S3-V2D',
        coordinates=np.array([[0., 0., .2], [1., 0., .2], [1., 1., .2]]),
        reference_normal=np.array([0., 0., 1.]), thickness=.2, elastic_modulus=1000.,
        poisson_ratio=.3, shell_node=2, beam_node=max(c.node_ids))
    before = c.checkpoint(records)
    recovery_before = canonical(c.recover(accepted))
    step = .0001*np.sin(np.arange(c.count)); step[:6] = 0.
    trial = c.advance(accepted.mechanical, step)
    result = made.evaluate(accepted, trial, np.zeros(18), np.zeros(6), parameter=.8)
    assert np.linalg.norm(result.residual) > 1e-8
    assert not result.coupled_state_committed
    assert c.checkpoint(records) == before
    assert canonical(accepted) == canonical(solved.state)
    assert canonical(c.recover(accepted)) == recovery_before
    (tmp_path/'predecessor.json').write_bytes(before)
    (tmp_path/'trial-history.json').write_bytes(result.beam_histories)
