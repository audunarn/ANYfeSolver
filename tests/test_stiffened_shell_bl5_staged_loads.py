"""Experimental line/staged-load integration; no physical BL5 qualification."""
import numpy as np
import pytest
from dataclasses import replace

from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_bl5_finite import _model, _action


@pytest.mark.parametrize('basis', ['p0', 'continuous_p1_coarse'])
def test_bl5_signed_stages_cancel_restore_and_continue(request, basis):
    definition = replace(_model(basis=basis), q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    def owner(model=definition):
        result = ss.create_analysis(definition=model)
        request.addfinalizer(result.close)
        return result
    a = owner()
    programme = ss.StagedDeadLoadProgramme(a.preflight()['identity'],
                                          (_action(100.), _action(-100.)))
    targets = (.5, 1., 1.25, 1.75, 2.)
    control = ss.AdaptiveIncrementControl(2,16) if basis == 'p0' else None
    token = CancellationToken()
    history = []
    def observe(result):
        history.append(result)
        assert result.free_equilibrium_ratio <= 1e-8
        assert result.constraint_max <= 1e-9
        assert len(result.beam_shell_line_work) == len(a._line_samples)
        # Grounded rotations have zero charts here, so their covectors are
        # physical moments. Retained algebraic rows are not physical nodes.
        count = a._shell_count+6*len(a._beam_ids)
        net = (result.support_reactions[:count]+result.applied_dead_load[:count]).reshape(-1,6)
        xyz = np.vstack([np.asarray(definition.shell.coordinates_m)+
                         result.shell_displacement.reshape(-1,6)[:,:3],
                         result.beam_positions])
        assert np.linalg.norm(net[:,:3].sum(axis=0)) <= 1e-6
        assert np.linalg.norm((np.cross(xyz,net[:,:3])+net[:,3:]).sum(axis=0)) <= 2e-6
        if result.load_position == 1.:
            token.cancel('line staged accepted prefix')
    with pytest.raises(SolveCancelled):
        a.solve(programme, targets, cancellation_token=token, on_accepted=observe,
                increment_control=control)
    assert [r.load_position for r in history] == [.5, 1.]
    saved = a.checkpoint()
    other_basis = 'p0' if basis != 'p0' else 'continuous_p1_coarse'
    incompatible = owner(replace(_model(basis=other_basis),
                                 q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1))
    with pytest.raises(ValueError):
        incompatible.restore(saved)
    assert incompatible._finite_driver is None
    historical = owner(_model(basis=basis))
    with pytest.raises(ValueError):
        historical.restore(saved)
    assert historical._finite_driver is None and historical._generation == 0
    if control is not None:
        from scipy.spatial.transform import Rotation
        before = a.recover()
        # This fresh branch has identity anchors; physical frame witness is
        # published explicitly only after the first chart event.
        before_frames = Rotation.from_rotvec(before.shell_displacement.reshape(-1,6)[:,3:].copy()).as_matrix()
        rebased = a.rebase_rotations(before.generation)
        np.testing.assert_allclose(before_frames, rebased.shell_frames, rtol=0., atol=1e-12)
        np.testing.assert_array_equal(before.beam_positions, rebased.beam_positions)
        saved = a.checkpoint()
    b = owner()
    b.restore(saved)
    assert b.checkpoint() == saved
    continuation = targets if control is not None else targets[2:]
    direct = a.solve(programme, continuation, on_accepted=observe, increment_control=control)
    resumed = b.solve(programme, continuation, increment_control=control)
    assert direct.load_programme_complete and direct.load_position == 2.
    np.testing.assert_array_equal(direct.beam_positions, resumed.beam_positions)
    assert a.checkpoint() == b.checkpoint()
    # Independent path through the already supported fixed-force mode at the
    # final elastic signed load; no claim about path-independent plasticity.
    oracle = owner().solve(_action(-100.))
    np.testing.assert_allclose(direct.beam_positions, oracle.beam_positions,
                               rtol=0., atol=1e-9)
    assert a.capabilities['staged_dead_loads']
    assert a.capabilities['status'] == 'EXPERIMENTAL_NOT_QUALIFIED'


def test_staged_line_admission_preserves_historical_and_l1_guards(request):
    from test_stiffened_shell_l1_assembly_experimental import _definition
    for definition in (_model(), _definition()):
        owner = ss.create_analysis(definition=definition)
        request.addfinalizer(owner.close)
        programme = ss.StagedDeadLoadProgramme(owner.preflight()['identity'], (_action(),))
        assert not owner.capabilities['staged_dead_loads']
        with pytest.raises(ValueError):
            owner.solve(programme)
        assert owner._finite_driver is None
