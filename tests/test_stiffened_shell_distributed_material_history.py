"""Distributed native work with actual coupled material activity."""
import json
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
from examples.stiffened_shell_coupled_plastic_history import definition, activity, OPTIONS, CONTROL


def test_distributed_force_activates_shell_and_physical_fibres(request, record_property):
    owner = ss.create_analysis(definition=definition())
    request.addfinalizer(owner.close)
    loads = ss.GEBeamDistributedLoad(owner.preflight()['identity'], tuple(
        ss.GEBeamLineAction(e.element_id, (0., 0., 3000.)) for e, _, _ in owner._beam_parts))
    history, accepted = [], []
    token = CancellationToken()
    targets = (.5, 1., .25, 1.1)
    def observe(result):
        row = activity(result, accepted[-1] if accepted else None)
        row['factor'] = result.accepted_factors[-1]
        row['shell_stress_abs_max_Pa'] = max(float(np.max(np.abs(r.layer_stress)))
                                           for r in result.shell_q4_material)
        ground = result.support_reactions[:owner._shell_count].reshape(-1, 6)
        positions = owner._shell_xyz+result.shell_displacement.reshape(-1, 6)[:, :3]
        force_error = ground[:, :3].sum(0)+np.asarray(result.distributed_load.applied_force)
        moment_error = (np.cross(positions, ground[:, :3])+ground[:, 3:]).sum(0)+np.asarray(
            result.distributed_load.applied_moment_about_origin)
        # Same fixed physical scaling as the existing coupled-material case:
        # reference total force (4m*3000N/m), and that force times 2m span.
        row['force_balance_ratio'] = float(np.linalg.norm(force_error)/12000.)
        row['moment_balance_ratio'] = float(np.linalg.norm(moment_error)/24000.)
        print(json.dumps({k:v for k,v in row.items() if k != 'active_fibres'}))
        history.append(row)
        accepted.append(result)
        assert row['force_balance_ratio'] <= 1e-8 and row['moment_balance_ratio'] <= 1e-8
        np.testing.assert_allclose(result.distributed_load.applied_force,
                                   (0., 0., 12000.*result.load_factor), atol=1e-10)
    def cancel_at_yielded_prefix(result):
        observe(result)
        if result.load_factor == 1.:
            token.cancel('yielded distributed prefix')
    with pytest.raises(SolveCancelled):
        owner.solve(loads, targets, increment_control=CONTROL,
                    on_accepted=cancel_at_yielded_prefix, cancellation_token=token, **OPTIONS)
    result = owner.recover()
    assert result.accepted_factors == (.5, 1.)
    assert any(row['simultaneous'] for row in history), history
    restored = ss.create_analysis(definition=definition())
    request.addfinalizer(restored.close)
    other = restored.restore(owner.checkpoint())
    np.testing.assert_array_equal(result.shell_layer_alpha, other.shell_layer_alpha)
    assert owner.checkpoint() == restored.checkpoint()
    last = owner.solve(loads, targets, increment_control=CONTROL, on_accepted=observe, **OPTIONS)
    repeated = restored.solve(loads, targets, increment_control=CONTROL, **OPTIONS)
    assert last.status == repeated.status == 'completed'
    assert last.accepted_factors.index(.25) < last.accepted_factors.index(1.1)
    assert owner.checkpoint() == restored.checkpoint()
    assert next(row for row in history if row['factor'] == 1.1)['simultaneous']
    before = owner.checkpoint()
    with pytest.raises(FiniteIncrementFailure):
        owner.solve(loads, (2.,), max_iterations=1)
    assert owner.checkpoint() == before
    recovered = owner.recover()
    assert recovered.generation == last.generation and recovered.load_factor == last.load_factor
    np.testing.assert_array_equal(recovered.shell_layer_alpha, last.shell_layer_alpha)
    record_property('material_activity', history)
