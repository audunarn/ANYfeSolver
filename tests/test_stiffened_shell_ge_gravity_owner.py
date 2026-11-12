"""Experimental native body load: complete owner work and restart contracts."""
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_ge_experimental import _panel
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from anyfesolver._stiffened_shell_ge_body_load import encode_checkpoint, decode_checkpoint


def owner(request):
    result = ss.create_analysis(definition=replace(_panel(), q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1))
    request.addfinalizer(result.close)
    return result


def load(analysis):
    return ss.GEBodyLoad(analysis.preflight()["identity"], (0., 0., -9.81))


def balance(analysis, result):
    reactions = result.support_reactions if hasattr(result, "support_reactions") else result.reactions
    points = np.asarray(analysis._shell_xyz)
    if hasattr(result, "beam_positions"):
        points = points+result.shell_displacement.reshape(-1, 6)[:, :3]
    wrench = reactions[:analysis._shell_count].reshape(-1, 6)
    np.testing.assert_allclose(wrench[:, :3].sum(axis=0), -np.asarray(result.body_load.applied_force),
                               rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose((np.cross(points, wrench[:, :3])+wrench[:, 3:]).sum(axis=0),
                               -np.asarray(result.body_load.applied_moment_about_origin), rtol=1e-7, atol=2e-5)


def test_reference_load_factor_reuse_and_checkpoint_binding(request):
    analysis, restarted = owner(request), owner(request)
    gravity = load(analysis)
    result = analysis.solve_linear(gravity)
    assert result.status == "completed"
    assert result.body_load.total_mass == pytest.approx(1101.)
    balance(analysis, result)
    saved = analysis.checkpoint()
    restored = restarted.restore(saved)
    np.testing.assert_array_equal(restored.shell_displacement, result.shell_displacement)
    assert restarted.checkpoint() == saved
    again = analysis.solve_linear(gravity)
    assert analysis._linear_factorizations == 1
    np.testing.assert_array_equal(again.shell_displacement, result.shell_displacement)
    descriptor, inner = decode_checkpoint(saved)
    assert descriptor.identity == gravity.identity
    wrong = ss.GEBodyLoad(gravity.model_identity, (0., 0., -1.)).compile(owner(request))
    target = owner(request)
    with pytest.raises(ValueError): target.restore(encode_checkpoint(wrong, inner))
    assert target._accepted is None and target._generation == 0
    legacy = analysis.solve_linear((ss.NodalAction("beam", 105, (0., 0., 1., 0., 0., 0.)),))
    assert legacy.body_load is None
    assert analysis._linear_factorizations == 1


def test_finite_body_work_restart_rebase_and_rejection(request, monkeypatch):
    analysis, restarted = owner(request), owner(request)
    gravity = load(analysis)
    first = analysis.solve(gravity, (.1,), max_iterations=15)
    assert first.status == "completed" and first.accepted_factors == (.1,)
    balance(analysis, first)
    before = analysis.checkpoint()
    token = CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled): analysis.solve(gravity, (.2,), cancellation_token=token)
    assert analysis.checkpoint() == before
    with pytest.raises(ValueError):
        analysis.solve(ss.GEBodyLoad(gravity.model_identity, (0., 0., -1.)), (.2,))
    assert analysis.checkpoint() == before
    original = FinitePointDriver._assemble
    def broken(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("injected after body evaluation")
    with monkeypatch.context() as scope:
        scope.setattr(FinitePointDriver, "_assemble", broken)
        with pytest.raises(RuntimeError, match="injected"):
            analysis.solve(gravity, (.2,))
    assert analysis.checkpoint() == before
    restored = restarted.restore(before)
    np.testing.assert_array_equal(restored.beam_positions, first.beam_positions)
    assert restarted.checkpoint() == before
    a = analysis.solve(gravity, (.2,), max_iterations=15)
    b = restarted.solve(gravity, (.2,), max_iterations=15)
    assert a.accepted_factors == b.accepted_factors == (.1, .2)
    np.testing.assert_array_equal(a.beam_positions, b.beam_positions)
    assert analysis.checkpoint() == restarted.checkpoint()
    balance(analysis, a)
    rebased = analysis.rebase_rotations(a.generation)
    np.testing.assert_allclose(rebased.body_load.applied_force, a.body_load.applied_force, rtol=0, atol=0)
    target = owner(request); target.restore(analysis.checkpoint())
    final_a = analysis.solve(gravity, (.3,), max_iterations=15)
    final_b = target.solve(gravity, (.3,), max_iterations=15)
    np.testing.assert_array_equal(final_a.beam_positions, final_b.beam_positions)
    balance(analysis, final_a)


def test_eccentric_native_load_scatter_matches_actual_update_derivative(request):
    from test_stiffened_shell_ge_reference_modal import _definition
    from anyfesolver._ge_beam3_p5.algebra import rotation
    analysis = ss.create_analysis(definition=replace(_definition(), q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1))
    request.addfinalizer(analysis.close)
    driver = FinitePointDriver(analysis)
    pattern, _ = driver._prepare_load(load(analysis), (.1,))
    state = replace(driver._state, cells=np.array([[rotation([.12, -.09, .04]), rotation([-.07, .1, .03])]]))
    zeros = np.zeros(analysis._K.shape[0])
    def residual(trial, factor):
        return driver._assemble(trial, factor, driver._shell_origins, driver._beam_origins, pattern=pattern)
    r1,k1,*_ = residual(state, 1.)
    r0,k0,*_ = residual(state, 0.)
    force,jac,_ = driver._external.evaluate(state)
    np.testing.assert_allclose(r1[:len(zeros)]-r0[:len(zeros)], -force, rtol=1e-9, atol=1e-6)
    assert np.linalg.norm(force[analysis._shell_count+18:analysis._shell_count+24]) > .01
    np.testing.assert_allclose((k1-k0)[:len(zeros),:len(zeros)].toarray(), -jac.toarray(), rtol=1e-6, atol=1e-6)
    direction=np.zeros(len(r1));direction[analysis._shell_count+18:analysis._shell_count+24]=(.3,-.7,.2,-.4,.1,.8)
    step=1e-5
    plus=driver._advance(state,direction,step);minus=driver._advance(state,direction,-step)
    numerical=(driver._external.evaluate(plus)[0]-driver._external.evaluate(minus)[0])/(2*step)
    np.testing.assert_allclose(numerical,jac@direction[:len(zeros)],rtol=1e-7,atol=1e-8)


def test_historical_q4_finite_gravity_rejected_without_state(request):
    analysis = ss.create_analysis(definition=_panel());request.addfinalizer(analysis.close)
    with pytest.raises(ValueError,match="objective Q4"):
        analysis.solve(load(analysis),(.1,))
    assert analysis._finite_driver is None and analysis._generation == 0


@pytest.mark.parametrize("triangles", [False, True])
def test_mixed_and_s3_body_restart(request, triangles):
    base=_panel()
    shell=replace(base.shell,q4_elements=() if triangles else ((1,2,5,4),),
        s3_elements=((1,2,5),(1,5,4),(2,3,6),(2,6,5)) if triangles else ((2,3,6),(2,6,5)))
    definition=replace(base,shell=shell,**({} if triangles else {"q4_pullback_policy":ss.Q4_ALGORITHMIC_PULLBACK_V1}))
    a=ss.create_analysis(definition=definition);b=ss.create_analysis(definition=definition)
    request.addfinalizer(a.close);request.addfinalizer(b.close)
    result=a.solve(load(a),(.01,),max_iterations=15)
    assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
    restored=b.restore(a.checkpoint())
    np.testing.assert_array_equal(restored.beam_positions,result.beam_positions)
    assert b.checkpoint() == a.checkpoint()
    balance(a,result)


def test_adaptive_cancelled_prefix_restore_and_same_load_identity(request):
    analysis, restarted = owner(request), owner(request)
    gravity=load(analysis); token=CancellationToken()
    control=ss.AdaptiveIncrementControl(2,16)
    def cancel(result): token.cancel("complete body prefix")
    with pytest.raises(SolveCancelled):
        analysis.solve(gravity,(.1,.2),increment_control=control,on_accepted=cancel,cancellation_token=token)
    assert analysis.recover().accepted_factors == (.1,)
    saved=analysis.checkpoint();restored=restarted.restore(saved)
    assert restored.status == "accepted_prefix"
    assert restarted.checkpoint() == saved
    a=analysis.solve(gravity,(.1,.2),increment_control=control)
    b=restarted.solve(gravity,(.1,.2),increment_control=control)
    assert a.status == b.status == "completed"
    np.testing.assert_array_equal(a.beam_positions,b.beam_positions)
    assert analysis.checkpoint() == restarted.checkpoint()


def test_body_gravity_native_material_history_restart(request):
    from test_stiffened_shell_ge_finite_experimental import _q4_shell_j2_panel
    definition=replace(_q4_shell_j2_panel(),q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    a=ss.create_analysis(definition=definition);b=ss.create_analysis(definition=definition)
    request.addfinalizer(a.close);request.addfinalizer(b.close)
    gravity=load(a)
    first=a.solve(gravity,(.1,),max_iterations=20,max_backtracks=12)
    saved=a.checkpoint();restored=b.restore(saved)
    np.testing.assert_array_equal(restored.shell_layer_alpha,first.shell_layer_alpha)
    assert b.checkpoint() == saved
    last=a.solve(gravity,(.2,),max_iterations=20,max_backtracks=12)
    resumed=b.solve(gravity,(.2,),max_iterations=20,max_backtracks=12)
    np.testing.assert_array_equal(resumed.shell_layer_alpha,last.shell_layer_alpha)
    np.testing.assert_array_equal(resumed.beam_station_accumulated,last.beam_station_accumulated)
    assert a.checkpoint() == b.checkpoint()
    balance(a,last)
