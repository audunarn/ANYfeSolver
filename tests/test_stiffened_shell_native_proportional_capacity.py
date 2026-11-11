"""SOURCE_ONLY_UNRUN: proportional caller contracts and future public diagnostic.

Orchestration doubles below test reporting/ownership, never mechanical acceptance.
All collection, interpreter, model construction and numerical tests are UNRUN.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import os

import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import SolveCancelled
from examples import stiffened_shell_native_capacity as capacity


def node_order():
    shell,beam = capacity.reference_nodes(capacity.Specimen())
    return tuple(('shell',n) for n in shell)+tuple(('beam',n) for n in beam)


@dataclass
class PublicRecord:
    status: str = 'completed'
    generation: int = 3
    load_factor: float = .5
    accepted_factors: tuple = (.1,.25,.5)
    amplitude: float | None = None


def test_known_prefix_validation_rejects_unchecked_inputs(tmp_path):
    with pytest.raises(ValueError): capacity.read_verified_force_prefix(b'wrong')
    with pytest.raises(TypeError): capacity.read_verified_force_prefix(bytearray())
    with pytest.raises(FileNotFoundError): capacity.read_verified_force_prefix(tmp_path/'missing.bin')


@pytest.mark.parametrize('change',(
    dict(status='accepted_prefix'),dict(generation=2),dict(load_factor=.75),
    dict(load_factor=float('nan')),dict(amplitude=0.),dict(accepted_factors=(.1,.5))))
def test_origin_is_bound_to_actual_restored_factor_inventory(change):
    preflight = dict(identity=capacity.VERIFIED_FORCE_PREFIX_MODEL)
    assert capacity.validate_proportional_origin(PublicRecord(),preflight) == .5
    with pytest.raises(ValueError):
        capacity.validate_proportional_origin(replace(PublicRecord(),**change),preflight)
    with pytest.raises(ValueError):
        capacity.validate_proportional_origin(PublicRecord(),dict(identity='different'))
    staged = PublicRecord(); staged.load_programme_identity = 'stage'
    with pytest.raises(ValueError): capacity.validate_proportional_origin(staged,preflight)


def test_full_and_half_load_vectors_use_distinct_origin_conventions():
    s = capacity.Specimen(); e = s.estimate()
    preload,probe = capacity.load_patterns(s,1.1*e['euler_clamped_free_n'],.02*e['euler_clamped_free_n'])
    full = capacity.combine((1.,preload),(1.,probe))
    assert capacity._plain(full) == capacity._plain(capacity.VERIFIED_FORCE_PREFIX_ACTIONS)
    order = node_order(); size = 6*len(order)+48
    full_vector = capacity.declared_load_vector(full,order,size)
    base_vector = capacity.declared_load_vector(capacity.combine((.5,full)),order,size)
    np.testing.assert_array_equal(base_vector,.5*full_vector)
    for alpha in (-.25,0.,.1,.5):
        np.testing.assert_allclose(base_vector+alpha*full_vector,(.5+alpha)*full_vector,
                                   rtol=1e-14,atol=0.)
    assert not np.any(full_vector[6*len(order):])
    origin = dict(generation=3,amplitude=None,proportional_factor=.5)
    arc = [dict(generation=4,amplitude=.1,proportional_factor=.6),
           dict(generation=5,amplitude=.05,proportional_factor=.55)]
    assert capacity.proportional_turning_observations([origin]+arc) == [dict(
        generations=[3,4,5],kind='proportional_factor_direction_change_unvalidated')]
    assert origin['amplitude'] is None


def test_checkpoint_packet_owns_complete_observation_and_writer_errors_propagate():
    result = PublicRecord(); raw = b'contract-only-checkpoint'
    class Owner:
        def recover(self): return result
        def checkpoint(self): return raw
    observation = dict(generation=3,state=dict(beam_positions=[[1.,2.,3.]]))
    packet = capacity._checkpointed_packet(Owner(),result,observation)
    observation['state']['beam_positions'][0][0] = -1.
    assert packet['observation']['state']['beam_positions'][0][0] == 1.
    assert packet['checkpoint_sha256'] == sha256(raw).hexdigest()
    report = dict(observations=[],checkpoint_inventory=[])
    def writer(delivered):
        delivered['observation']['state']['beam_positions'][0][0] = 999.
        raise OSError('checkpoint write failed')
    with pytest.raises(OSError,match='checkpoint write failed'):
        capacity._retain_checkpointed_packet(report,packet,writer)
    assert report['observations'][0]['state']['beam_positions'][0][0] == 1.
    assert report['checkpoint_inventory'][0]['checkpoint_bytes'] == raw


def caller_double(monkeypatch, behavior):
    """No mechanics: isolate the example's public-call/reporting contract."""
    raw = b'caller-contract-prefix'
    order = node_order()
    class Owner:
        def __init__(self): self.result = PublicRecord(); self.closed = False
        def preflight(self):
            return dict(identity=capacity.VERIFIED_FORCE_PREFIX_MODEL,arc_metric_node_order=order)
        def restore(self,received):
            assert received == raw
            return self.result
        def checkpoint(self):
            return raw if self.result.generation == 3 else b'arc'+bytes((self.result.generation,))
        def recover(self): return self.result
        def close(self): self.closed = True
        def solve(self,*args,**kwargs): raise AssertionError('must not repeat force seed')
        def continue_arc_length(self,actions,control,**kwargs):
            assert capacity._plain(actions) == capacity._plain(capacity.VERIFIED_FORCE_PREFIX_ACTIONS)
            assert control.orientation == 1 and control.radius == .01 and kwargs['steps'] == 1
            return behavior(self,kwargs)
    owner = Owner()
    monkeypatch.setattr(capacity,'read_verified_force_prefix',lambda source:raw)
    monkeypatch.setattr(capacity,'definition',lambda specimen:None)
    monkeypatch.setattr(ss,'create_analysis',lambda **kwargs:owner)
    def observation(result,*args,**kwargs):
        assert kwargs['seed_actions'] == kwargs['arc_direction']
        assert capacity._plain(kwargs['arc_base_actions']) == capacity._plain(
            capacity.combine((.5,kwargs['seed_actions'])))
        return dict(generation=result.generation,amplitude=result.amplitude,state=capacity._plain(result),
            material_activity=dict(any_material_history_or_update=False,
                                   elastic_fold_interpretation_eligible=True))
    monkeypatch.setattr(capacity,'observe',observation)
    return owner


def test_each_acceptance_delivers_checkpoint_inside_public_callback(monkeypatch):
    def accepted(owner,kwargs):
        owner.result = PublicRecord(status='accepted_prefix',generation=owner.result.generation+1,
                                    amplitude=.1*(owner.result.generation-2))
        kwargs['on_accepted'](deepcopy(owner.result))
        return replace(owner.result,status='completed')
    owner = caller_double(monkeypatch,accepted)
    streamed = []
    report = capacity.proportional_workflow(b'ignored-by-double',arc_steps=2,on_state=streamed.append)
    assert owner.closed
    assert [packet['generation'] for packet in streamed] == [3,4,5]
    assert report['accepted_arc_steps'] == 2
    assert report['status'] == 'bounded_no_turning_observed'
    assert report['recovered_status'] == 'accepted_prefix'
    assert report['recovered_generation'] == 5
    for packet,row in zip(streamed,report['checkpoint_inventory']):
        assert packet['checkpoint_bytes'] == row['checkpoint_bytes']
        assert packet['checkpoint_sha256'] == row['sha256']


@pytest.mark.parametrize('failure',(
    ss.ArcSelectionFailure('bounded attempts',()),
    ss.FiniteIncrementFailure(ss.IncrementAttempt(.75,.5,.75,'rejected',12,73,0,48,1.,0.,'limit')),
    SolveCancelled('stop at restored origin',stage='stiffened-shell.arc.step')))
def test_exact_typed_failure_retains_actual_prefix_and_status(monkeypatch,failure):
    def reject(owner,kwargs): raise failure
    owner = caller_double(monkeypatch,reject)
    report = capacity.proportional_workflow(b'ignored-by-double',arc_steps=1)
    assert owner.closed
    assert report['accepted_arc_steps'] == 0
    assert report['recovered_generation'] == 3 and report['recovered_status'] == 'completed'
    assert report['failure']['type'] == type(failure).__name__
    assert len(report['observations']) == len(report['checkpoint_inventory']) == 1
    assert report['status'] == ('cancelled' if isinstance(failure,SolveCancelled) else 'typed_failure')


def test_partial_return_does_not_claim_requested_completion(monkeypatch):
    def partial(owner,kwargs): return replace(owner.result,status='accepted_prefix')
    caller_double(monkeypatch,partial)
    report = capacity.proportional_workflow(b'ignored-by-double',arc_steps=2)
    assert report['status'] == 'incomplete_arc_return'
    assert report['returned_status'] == 'accepted_prefix' and report['accepted_arc_steps'] == 0


@pytest.mark.parametrize('error',(ValueError('malformed'),RuntimeError('restart mismatch'),
    ss.ArcSelectionFailure('writer exception must propagate',())))
def test_programming_and_checkpoint_writer_errors_propagate(monkeypatch,error):
    def broken(owner,kwargs): raise error
    caller_double(monkeypatch,broken)
    if isinstance(error,ss.ArcSelectionFailure):
        def writer(packet):
            if packet['generation'] == 3: raise error
        with pytest.raises(type(error)):
            capacity.proportional_workflow(b'ignored-by-double',arc_steps=1,on_state=writer)
    else:
        with pytest.raises(type(error)):
            capacity.proportional_workflow(b'ignored-by-double',arc_steps=1)


def test_future_public_restart_rejects_changed_direction_and_control():
    """UNRUN numerical diagnostic; requires separately granted execution authority."""
    fixture = os.environ.get('ANYSOLVER_NATIVE_CAPACITY_VERIFIED_PREFIX')
    if not fixture:
        pytest.skip('ANYSOLVER_NATIVE_CAPACITY_VERIFIED_PREFIX absent: the separately authorized '
                    'SHA-bound prefix fixture is not granted; UNRUN numerical diagnostic not executed')
    path = Path(fixture)
    if not path.is_absolute():
        pytest.fail('ANYSOLVER_NATIVE_CAPACITY_VERIFIED_PREFIX must be an explicit absolute checkpoint path')
    streamed = []
    report = capacity.proportional_workflow(path,arc_steps=1,on_state=streamed.append)
    assert report['capacity_claim'] is False
    if report['accepted_arc_steps'] == 0:
        pytest.fail('No accepted arc checkpoint: actual restart/refusal contract remains unexercised; '
                    f"status={report['status']}, failure={report['failure']}")
    raw = streamed[-1]['checkpoint_bytes']
    owner = ss.create_analysis(definition=capacity.definition())
    try:
        owner.restore(raw)
        expected_recovery = deepcopy(streamed[-1]['observation']['state'])
        restored_recovery = deepcopy(capacity._plain(owner.recover()))
        assert restored_recovery == expected_recovery
        assert owner.checkpoint() == raw
        order = tuple(owner.preflight()['arc_metric_node_order'])
        control = ss.ArcLengthControl('shell',18,2,capacity.metric_for(order,capacity.Specimen()),.01,orientation=1)
        for actions,changed in ((capacity.combine((-1.,capacity.VERIFIED_FORCE_PREFIX_ACTIONS)),control),
            (capacity.VERIFIED_FORCE_PREFIX_ACTIONS,replace(control,radius=.02))):
            with pytest.raises(ValueError): owner.continue_arc_length(actions,changed)
            assert deepcopy(capacity._plain(owner.recover())) == expected_recovery
            assert owner.checkpoint() == raw
    finally:
        owner.close()
