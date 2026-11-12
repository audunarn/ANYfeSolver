"""Two-macro first-Newton-step diagnosis, not nonlinear qualification.

Only the first global step is replaced by a supplied-factor Decimal audit.
The unchanged local solve then decides whether that trial is admissible.
Every trial result is recorded; a passing audit is NOT a completed force gate.
"""
from decimal import Decimal as D, localcontext
import numpy as np
import pytest
from scipy.linalg import block_diag
import anyfesolver._ge_beam3_seeded_load_program as driver
from anyfesolver._ge_beam3_seeded_loaded_modal import prepare
from anyfesolver._ge_beam3_shared_kinematic_split import shared_kinematic_chain
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from docs.reference_cases.ge_beam3_decimal_static_chain_audit import solve_static_chain
from test_ge_beam3_curved_contrast_probe import make_curved


def test_static_chain_retains_weak_terms_before_gram():
    result = solve_static_chain([[1e8, 0.], [0., 1.]],
                               [[1., 1.], [0., 1.]], [0., 1.], [0, 1])
    assert tuple(D(x) for x in result) == (D(-1), D(1))
    with pytest.raises(ValueError, match='finite'):
        solve_static_chain([[float('nan')]], [[1.]], [1.], [0])


@pytest.mark.parametrize('slenderness', [100., 10000., 1000000.])
def test_first_force_step_from_unexpanded_chain(slenderness, tmp_path, monkeypatch):
    model, inertias = make_curved(slenderness)
    states = {i: e.init_model_bound_nonlinear_state(model.mesh, e.core.section, 1)
              for i, e in model.mesh.elements.items()}
    before = canonical(states)
    packet, guard = prepare(model, states, np.zeros(30), inertias, load_parameter=0.)
    lefts = []; rights = []; nodal_k = np.zeros((30, 30))
    for eid, internal in packet.internal_layout:
        e = model.mesh.elements[eid]
        state = e.validate_model_bound_nonlinear_state(model.mesh, e.core.section, states[eid], 1)
        slots = tuple(int(i) for i in e.get_dof_mapping(model.mesh))
        l, j, g = shared_kinematic_chain(e, state['material_state'])
        assert np.count_nonzero(g) == 0
        row = np.zeros((len(j), 42)); row[:, slots+internal] = j
        lefts.append(l); rights.append(row)
        nodal_k[np.ix_(slots, slots)] += state['material_state']['response'].tangent
    left = block_diag(*lefts); right = np.vstack(rights)
    force = np.zeros(42); force[24:27] = .5*np.array([.05, -.001, 0.])
    free = list(packet.free_dofs)
    low = solve_static_chain(left.tolist(), right.tolist(), force.tolist(), free, digits=60)
    high = solve_static_chain(left.tolist(), right.tolist(), force.tolist(), free, digits=90)
    with localcontext() as context:
        context.prec = 100
        assert max(abs(D(a)-D(b)) for a, b in zip(low, high)) < D('1e-40')
    step = np.array(high, dtype=float)
    guard(); assert canonical(states) == before
    original = driver.factorize; capture = {}; assemblies = []
    assemble = driver._assemble_nonlinear_system
    def observe_assembly(*args, **kwargs):
        try:
            output = assemble(*args, **kwargs)
        except Exception as exc:
            assemblies.append(dict(status='failed', error=type(exc).__name__, message=str(exc)))
            raise
        assemblies.append(dict(status='returned'))
        return output
    def replace_first(matrix, kind, **kwargs):
        assert not capture, 'diagnostic must never replace a second Newton step'
        handle = original(matrix, kind, **kwargs)
        class FirstStep:
            def solve(self, rhs):
                raw = np.asarray(handle.solve(rhs))
                capture.update(matrix=matrix.toarray(), rhs=np.array(rhs), raw_step=raw)
                np.testing.assert_array_equal(matrix.toarray(), nodal_k[6:, 6:])
                np.testing.assert_array_equal(rhs, force[6:30])
                return step[6:30].copy()
        return FirstStep()
    monkeypatch.setattr(driver, 'factorize', replace_first)
    monkeypatch.setattr(driver, '_assemble_nonlinear_system', observe_assembly)
    result = driver.solve_force_program(model,
        driver.ForceProgram((.5,), ((5, .05, -.001, 0.),), max_iterations=1))
    assert capture
    with (tmp_path/'first-step-inputs.npz').open('xb') as stream:
        np.savez(stream, left=left, right=right, force=force, free=free,
                 decimal_step=step, **capture)
    record = dict(slenderness=slenderness, diagnostic_only=True, production_qualified=False,
        unexpanded_chain_decimal_step=high,
        condensed_step_relative_difference=float(np.linalg.norm(capture['raw_step']-step[6:30])/np.linalg.norm(step[6:30])),
        trial_status=result.status, trial_failure=result.failure, assemblies=assemblies,
        completed_targets=result.completed_targets)
    with (tmp_path/'first-step-audit.json').open('xb') as stream: stream.write(canonical(record))
    with (tmp_path/'last-accepted-checkpoint.json').open('xb') as stream: stream.write(result.checkpoint)
    print(canonical(record).decode(), flush=True)
    # A failure is preserved, not reclassified as scientific success.
    assert result.completed_targets == 0
    assert np.count_nonzero(result.displacements) == 0
