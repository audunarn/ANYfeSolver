"""Fixed development fixtures, not hydrodynamic source qualification."""
from dataclasses import FrozenInstanceError, replace
from importlib.util import find_spec
import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver.marine_rigid import RigidBody6D, RigidBodyState
from anyfesolver.marine_hydro_coordinates import EquilibriumHydroFrame, evaluate_coupled
from anyfesolver.marine_contributions import (BodyLayout, CommittedHistory, ContributionSet,
    MarineContributionsError, StepInformation)
from anyfesolver.marine_mechanical import (MarineMechanicalModel, MarineMechanicalRunSpec,
    solve_marine_mechanical)
from anyfesolver.marine_potential_flow import (PotentialFlowDevelopmentData,
    PotentialFlowDevelopmentContribution, MarinePotentialFlowError, equilibrium_geometry_hash)


def body(name='a'):
    return RigidBody6D(name, 4., (0., 0., 0.), tuple(map(tuple, 3.*np.eye(3))))


def frame(name='a', complex_geometry=False):
    if not complex_geometry:
        return EquilibriumHydroFrame(name, (0.,)*3, np.eye(3), np.eye(3), (0.,)*3)
    return EquilibriumHydroFrame(name, (.1, -.2, .3),
        Rotation.from_rotvec((-.1, .12, .04)).as_matrix(),
        Rotation.from_rotvec((.2, -.1, .3)).as_matrix(), (.2, -.3, .1))


def fixture(frames=None, linear=False):
    frames = (frame(),) if frames is None else tuple(frames)
    n = 6*len(frames)
    if linear:
        A, C = np.zeros((n, n)), np.zeros((n, n))
        A[0, 0], C[0, 0] = 2., 3.
        kernel = np.zeros((3, n, n))
        kernel[:, 0, 0] = (2., 1., .5)
        baseline, real, imag = np.zeros(n), np.zeros(n), np.zeros(n)
        baseline[0], real[0], imag[0] = .6, .4, .2
    else:
        seed = np.arange(1, n+1)/n
        A = np.diag(1.+seed)+.07*np.outer(seed, seed)
        C = np.diag(.5+seed)+.1*np.outer(seed, seed)
        coupled = np.diag(.2+seed)+.05*np.outer(seed, seed)
        kernel = np.array([2.*coupled, coupled, .5*coupled])
        baseline, real, imag = seed, seed[::-1]/3., seed/4.
    return PotentialFlowDevelopmentData(frames, equilibrium_geometry_hash(frames), .1,
        kernel, A, C, baseline, real, imag, 2., np.tile((4., .8), (len(frames), 1)),
        'fixed synthetic arrays, not physical source evidence')


def states_for(frames, general):
    return {f.body_id:body(f.body_id).state(position=(.15, -.08, .21),
        rotation_vector=(.2, -.1, .15) if general else (0.,)*3,
        anchor=Rotation.from_rotvec((-.06, .08, -.1)).as_matrix(),
        velocity=(.12, -.05, .08, .11, .09, -.13),
        acceleration=(.04, .06, -.09, -.07, .08, .1)) for f in frames}


def perturb(states, body_id, field, j, delta):
    values = dict(states)
    state = states[body_id]
    if field == 'q':
        name, column = ('position', j) if j < 3 else ('rotation_vector', j-3)
    else:
        name, column = field, j
    vector = np.array(getattr(state, name), copy=True)
    vector[column] += delta
    values[body_id] = replace(state, **{name:tuple(vector)})
    return values


@pytest.mark.tangents
@pytest.mark.parametrize('general', (False, True))
def test_full_two_body_total_tangents_cross_blocks_and_work(general):
    frames = (frame('a', True), frame('b', True))
    data = fixture(frames)
    states = states_for(frames, general)
    provider = PotentialFlowDevelopmentContribution(data)
    previous = states_for(frames, False)
    information = StepInformation(1, data.dt, 'step', previous)
    layout = BodyLayout(data.body_ids)
    def evaluate(values):
        return provider.evaluate_trial_with_context(CommittedHistory(), values, layout, information)
    result = evaluate(states)
    assert np.linalg.norm(result.mass[:6, 6:]) > 0.
    epsilon = 1e-5
    for field, tangent in (('q', result.stiffness), ('velocity', result.damping),
                           ('acceleration', result.mass)):
        finite = np.empty((12, 12))
        for i, name in enumerate(data.body_ids):
            for j in range(6):
                plus = evaluate(perturb(states, name, field, j, epsilon)).external_force
                minus = evaluate(perturb(states, name, field, j, -epsilon)).external_force
                finite[:, 6*i+j] = -(plus-minus)/(2*epsilon)
        # Unit-separated body translation/rotation force/moment blocks.
        for rows in (slice(0,3), slice(3,6), slice(6,9), slice(9,12)):
            for cols in (slice(0,3), slice(3,6), slice(6,9), slice(9,12)):
                scale = max(1., np.max(np.abs(finite[rows,cols])))
                np.testing.assert_allclose(tangent[rows,cols], finite[rows,cols],
                                           rtol=1e-8, atol=1e-8*scale)
    np.testing.assert_allclose(sum(result.diagnostics['components'].values()),
                               result.external_force, rtol=1e-10, atol=1e-10)
    chart_velocity = np.concatenate([s.velocity for s in states.values()])
    world_velocity = np.concatenate([s.world_velocity for s in states.values()])
    assert abs(result.external_force@chart_velocity-
               result.diagnostics['world_wrench']@world_velocity) < 1e-10


@pytest.mark.oracle
@pytest.mark.parametrize('n_kernel',(2,3))
def test_fixed_grid_trapezoid_initialization_and_finite_memory_window(n_kernel):
    original = fixture(linear=True)
    data = replace(original,kernel=original.kernel[:n_kernel])
    provider = PotentialFlowDevelopmentContribution(data)
    owner = ContributionSet(('a',), (provider,))
    b = body()
    previous = {'a':b.state(position=(.2,0.,0.), velocity=(.3,0.,0.,0.,0.,0.))}
    step = owner.prepare_step(StepInformation(0,.1,'initialization'))
    initial = owner.evaluate_trial(step, previous)
    assert initial.diagnostics[provider.contribution_id]['components']['radiation'][0] == 0.
    with pytest.raises(MarineContributionsError, match='cannot commit'):
        owner.commit(step, initial)
    owner.abort_step(step)
    velocities = [.3]
    for k, velocity in enumerate((.4, .7, .9), 1):
        current = {'a':b.state(position=(.2,0.,0.), velocity=(velocity,0.,0.,0.,0.,0.))}
        step = owner.prepare_step(StepInformation(k,.1,'step',previous))
        trial = owner.evaluate_trial(step, current)
        velocities.append(velocity)
        length = min(k,n_kernel-1)
        # Independently explicit trapezoidal endpoint/interior formula.
        expected = .1*(2.*velocity/2.+sum((1. if j < length else .5)*
                          (1.,.5)[j-1]*velocities[k-j] for j in range(1,length+1)))
        diag = trial.diagnostics[provider.contribution_id]['components']
        assert abs(diag['radiation'][0]+expected) < 1e-10
        expected_wave = .4*math.cos(2*k*.1)+.2*math.sin(2*k*.1)
        assert abs(diag['excitation'][0]-expected_wave) < 1e-10
        owner.commit(step, trial)
        memory = owner.committed_history[provider.contribution_id]
        assert memory['step_index'] == k and len(memory['velocities']) == min(k+1,n_kernel-1)
        assert memory['velocities'][-1][0] == velocity
        previous = current


@pytest.mark.native
def test_production_newmark_matches_independent_discrete_linear_recurrence():
    b, data = body(), fixture(linear=True)
    provider = PotentialFlowDevelopmentContribution(data)
    result = solve_marine_mechanical(MarineMechanicalModel((b,), contributions=(provider,)),
        {'a':b.state(position=(.2,0.,0.), velocity=(.3,0.,0.,0.,0.,0.), acceleration=(99.,)*6)},
        MarineMechanicalRunSpec(.1,3))
    assert result.status == 'completed', result.logs
    x, v, a = .2, .3, .4/6.
    velocities = [v]
    for k, accepted in enumerate(result.accepted):
        if k:
            xp, vp = x+.1*v+.0025*a, v+.05*a
            length = min(k,2)
            memory = .1*sum((1. if j < length else .5)*(1.,.5)[j-1]*
                            velocities[k-j] for j in range(1,length+1))
            wave = .4*math.cos(.2*k)+.2*math.sin(.2*k)
            a = (.6+wave-3.*xp-.1*vp-memory)/(6.+3.*.0025+.1*.05)
            x, v = xp+.0025*a, vp+.05*a
            velocities.append(v)
        np.testing.assert_allclose((accepted.positions['a'][0], accepted.world_velocities['a'][0],
                                   accepted.world_accelerations['a'][0]), (x,v,a), rtol=1e-10, atol=1e-10)
    assert result.histories[provider.contribution_id]['step_index'] == 3
    assert len(result.histories[provider.contribution_id]['velocities']) == 2


@pytest.mark.history
def test_rejected_cancelled_and_stale_trials_leave_memory_unchanged():
    b, data = body(), fixture(linear=True)
    provider = PotentialFlowDevelopmentContribution(data)
    flag = {'cancel':False}
    owner = ContributionSet(('a',), (provider,), cancel=lambda:flag['cancel'])
    previous = {'a':b.state(velocity=(.3,0.,0.,0.,0.,0.))}
    step = owner.prepare_step(StepInformation(1,.1,'step',previous))
    stale = owner.evaluate_trial(step, previous)
    accepted = owner.evaluate_trial(step, previous)
    with pytest.raises(MarineContributionsError, match='stale'):
        owner.commit(step, stale)
    assert not len(owner.committed_history)
    owner.commit(step, accepted)
    committed = owner.committed_history
    step = owner.prepare_step(StepInformation(2,.1,'step',previous))
    with pytest.raises(MarinePotentialFlowError, match='outside'):
        owner.evaluate_trial(step, {'a':b.state(position=(5.,0.,0.))})
    assert owner.committed_history is committed
    trial = owner.evaluate_trial(step, previous)
    flag['cancel'] = True
    with pytest.raises(MarineContributionsError, match='cancellation'):
        owner.commit(step, trial)
    assert owner.committed_history is committed
    flag['cancel'] = False
    owner.abort_step(step)
    with pytest.raises(MarinePotentialFlowError, match='clock'):
        owner.prepare_step(StepInformation(3,.1,'step',previous))
    assert owner.committed_history is committed


@pytest.mark.history
def test_fixed_equilibrium_memory_survives_real_body_rechart():
    data = fixture((frame('a', True),))
    provider = PotentialFlowDevelopmentContribution(data)
    owner = ContributionSet(('a',),(provider,))
    old = states_for(data.frames,True)['a']
    states = {'a':old}
    step = owner.prepare_step(StepInformation(1,.1,'step',states))
    trial = owner.evaluate_trial(step,states)
    owner.commit(step,trial)
    committed = owner.committed_history
    # Same physical pose, WORLD rates/accelerations, new zero-angle chart.
    recharted = RigidBodyState(position=old.position, anchor=old.rotation,
        rotation_vector=(0.,)*3, velocity=old.world_velocity, acceleration=old.world_acceleration)
    new_states = {'a':recharted}
    information = StepInformation(2,.1,'step',new_states)
    step = owner.prepare_step(information)
    current = owner.evaluate_trial(step,new_states)
    old_trial = provider.evaluate_trial_with_context(committed,states,BodyLayout(('a',)),
        StepInformation(2,.1,'step',states))
    np.testing.assert_allclose(current.diagnostics[provider.contribution_id]['world_wrench'],
                               old_trial.diagnostics['world_wrench'],rtol=1e-10,atol=1e-10)
    assert owner.committed_history is committed
    owner.abort_step(step)


@pytest.mark.refusal
def test_declarations_binding_aliases_clock_order_and_ownership_refused():
    data = fixture()
    for changes in ({'source_kind':'bem'}, {'force_basis':'physical_wrench'},
                    {'hydrostatics_role':'full'}, {'gravity_included':True},
                    {'prehistory':'unknown'}, {'equilibrium_hash':'wrong'},
                    {'equilibrium_hash':np.array(['wrong','bad'])},
                    {'motion_bounds':[[1.,math.pi]]}, {'dt':True},
                    {'omega':math.inf}, {'added_mass_infinite':np.zeros((2,2))}):
        with pytest.raises(MarinePotentialFlowError):
            replace(data, **changes)
    changed = (replace(data.frames[0], equilibrium_position=(.1,0.,0.)),)
    with pytest.raises(MarinePotentialFlowError, match='binding'):
        replace(data, frames=changed)
    original = np.zeros(6)
    copied = replace(data, baseline=original)
    original[:] = 5.
    assert not np.any(copied.baseline)
    with pytest.raises(ValueError):
        copied.baseline.setflags(write=True)
    provider = PotentialFlowDevelopmentContribution(data)
    with pytest.raises(FrozenInstanceError):
        provider.data = copied
    owner = ContributionSet(('a',), (provider,))
    with pytest.raises(MarinePotentialFlowError, match='explicit clock'):
        owner.prepare_step()
    with pytest.raises(MarinePotentialFlowError, match='exact kernel dt'):
        owner.prepare_step(StepInformation(0,.2,'initialization'))
    with pytest.raises(MarineContributionsError, match='duplicate'):
        ContributionSet(('a',), (provider,PotentialFlowDevelopmentContribution(copied)))
    with pytest.raises(MarinePotentialFlowError, match='whole body layout'):
        provider.evaluate_trial_with_context(CommittedHistory(), {'a':body().state()},
            BodyLayout(('wrong',)),StepInformation(0,.1,'initialization'))
    with pytest.raises(MarinePotentialFlowError, match='missing'):
        provider.prepare_step_with_context(CommittedHistory(),StepInformation(2,.1,'step',{'a':body().state()}))
    invalid = CommittedHistory({provider.contribution_id:{'dt':np.array([.1,.2]),
        'step_index':1,'velocities':((0.,)*6,(0.,)*6)}})
    with pytest.raises(MarinePotentialFlowError):
        provider.prepare_step_with_context(invalid,StepInformation(2,.1,'step',{'a':body().state()}))


@pytest.mark.native
def test_native_cancellation_during_second_fluid_trial_preserves_first_step_memory():
    flag = {'cancel':False}
    class CancelOnSecond(PotentialFlowDevelopmentContribution):
        def evaluate_trial_with_context(self,committed,states,layout,information):
            result = super().evaluate_trial_with_context(committed,states,layout,information)
            if information.step_index == 2:
                flag['cancel'] = True
            return result
    b = body()
    provider = CancelOnSecond(fixture(linear=True))
    result = solve_marine_mechanical(MarineMechanicalModel((b,),contributions=(provider,)),
        {'a':b.state(position=(.2,0.,0.),velocity=(.3,0.,0.,0.,0.,0.))},
        MarineMechanicalRunSpec(.1,3),cancel=lambda:flag['cancel'])
    assert result.status == 'cancelled' and [f.step for f in result.accepted] == [0,1]
    memory = result.histories[provider.contribution_id]
    assert memory['step_index'] == 1 and len(memory['velocities']) == 2
    assert abs(memory['velocities'][-1][0]-result.accepted[1].world_velocities['a'][0]) < 1e-10


@pytest.mark.adapter
def test_public_anyhydro_kernel_snapshot_order_and_optional_import_contract():
    if find_spec('anyhydro') is None:
        pytest.importorskip('anyhydro', reason='optional marine-excitation provider is not installed')
    from anyhydro.radiation import RadiationKernel, RadiationSettings, RadiationDiagnostics
    data = fixture()
    diagnostics = RadiationDiagnostics(np.array([1.,2.]), np.zeros((6,6)), np.zeros((6,6)),
        2.,1.,.2,.2,.1,0.,0.,0.,0,0,0.,0)
    artifact = RadiationKernel(RadiationSettings(.1,3,1e-10,1e-10,
        zero_frequency_damping=np.zeros((6,6))), np.array([0.,.1,.2]), data.dof_keys,
        data.body_ids, data.kernel, data.added_mass_infinite, diagnostics,
        {'source':'synthetic public immutable artifact'}, ('development only',))
    arguments = {name:getattr(data,name) for name in ('frames','equilibrium_hash','restoring',
        'baseline','excitation_real','excitation_imag','omega','motion_bounds')}
    copied = PotentialFlowDevelopmentData.from_radiation_kernel(artifact,
        **arguments,development_only=True)
    assert copied.source_identity.endswith(artifact.content_hash())
    assert artifact.qualification_status == 'unqualified'
    np.testing.assert_array_equal(copied.kernel,artifact.kernel)
    np.testing.assert_array_equal(copied.added_mass_infinite,artifact.added_mass_infinite)
    with pytest.raises(MarinePotentialFlowError,match='development_only'):
        PotentialFlowDevelopmentData.from_radiation_kernel(artifact,**arguments,development_only=False)
    with pytest.raises(MarinePotentialFlowError,match='order'):
        PotentialFlowDevelopmentData.from_radiation_kernel(replace(artifact,
            dof_keys=tuple(reversed(data.dof_keys))),**arguments,development_only=True)


@pytest.mark.preservation
def test_nonsymmetric_full_coupled_force_values_and_included_geometric_bias():
    frames = (frame('a',True),frame('b',True))
    original = fixture(frames)
    A, C, K = original.added_mass_infinite.copy(), original.restoring.copy(), original.kernel.copy()
    A[0,8] += .37
    C[7,3] += .24
    K[0,0,8] += .43
    K[1,7,3] += .29
    data = replace(original,added_mass_infinite=A,restoring=C,kernel=K,
                   source_identity='NONPHYSICAL nonsymmetric algebra-preservation fixture')
    provider = PotentialFlowDevelopmentContribution(data)
    current, previous = states_for(frames,True),states_for(frames,False)
    mapped, old = evaluate_coupled(frames,current),evaluate_coupled(frames,previous)
    trial = provider.evaluate_trial_with_context(CommittedHistory(),current,BodyLayout(data.body_ids),
        StepInformation(1,.1,'step',previous))
    D = mapped.jacobian
    components = trial.diagnostics['components']
    expected_radiation = -.05*D.T@(K[0]@mapped.velocity+K[1]@old.velocity)
    expected_restoring = -D.T@C@mapped.coordinates
    expected_inertia = -D.T@A@mapped.acceleration
    expected_wave = D.T@(data.excitation_real*math.cos(.2)+data.excitation_imag*math.sin(.2))
    for actual, expected in ((components['reference_baseline'],D.T@data.baseline),
        (components['incremental_restoring'],expected_restoring),
        (components['radiation'],expected_radiation),(components['added_inertia'],expected_inertia),
        (components['excitation'],expected_wave),
        (trial.external_force,D.T@data.baseline+expected_restoring+expected_radiation+expected_inertia+expected_wave)):
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
    bias = np.concatenate([block.geometric_bias for block in mapped.blocks])
    np.testing.assert_allclose(trial.diagnostics['geometric_inertia_external'],-D.T@A@bias,
                               rtol=1e-10,atol=1e-10)
    assert np.linalg.norm(D.T@A@bias) > 1e-5
