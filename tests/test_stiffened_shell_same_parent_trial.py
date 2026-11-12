"""Same-parent payload ownership, observation and complete-history checks."""
from dataclasses import replace
import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from anyfesolver._stiffened_shell_same_parent_trial import FixedTrialHandoff
from anyfesolver.control import SolveCancelled
from scripts.experimental_stiffened_ge_multi_member_panel import make_definition, actions_for


@pytest.fixture
def assembled():
    definition = replace(make_definition(2, 2), q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    owner = ss.create_analysis(definition=definition)
    driver = FinitePointDriver(owner)
    owner._finite_driver = driver
    trial = driver._increment_origin(driver._state)
    pattern = driver._load(actions_for(definition))
    before = FixedTrialHandoff.begin(driver, trial, .01, pattern)
    result = driver._assemble(trial, .01, driver._shell_origins, driver._beam_origins, pattern=pattern)
    handoff = FixedTrialHandoff.capture(driver, trial, .01, pattern, result, before=before)
    assert handoff is not None
    yield owner, driver, trial, pattern, result, handoff
    owner.close()


def test_complete_payload_is_single_use_and_keeps_matching_recovery(assembled):
    _owner, driver, trial, pattern, result, handoff = assembled
    points = driver._point_trial_recovery
    assert handoff.take(driver, trial, .01, pattern) is result
    assert driver._point_trial_recovery is points
    assert handoff.take(driver, trial, .01, pattern) is None


@pytest.mark.parametrize('mutation', ('trial', 'parent', 'factor', 'load', 'free', 'slots',
                                     'reference', 'residual', 'tangent', 'recovery', 'generation', 'shape', 'errstate'))
def test_changed_dependency_discards_payload(assembled, mutation):
    owner, driver, trial, pattern, result, handoff = assembled
    factor = .01
    if mutation == 'trial':
        trial.q[0] = 1e-10
    elif mutation == 'parent':
        origin = driver._shell_origins[0]
        origin['alpha'] = origin['alpha'].copy()+1e-10
    elif mutation == 'factor':
        factor = .02
    elif mutation == 'load':
        pattern = pattern.copy()
        pattern[0] += 1.
    elif mutation == 'free':
        owner._free = owner._free.copy()
    elif mutation == 'slots':
        slots = owner._shell_parts[0][2]
        slots.setflags(write=True)
        slots[0] += 1
    elif mutation == 'reference':
        driver._reference_positions[0, 0] += 1e-8
    elif mutation == 'residual':
        result[0][0] += 1.
    elif mutation == 'tangent':
        result[1].data[0] += 1.
    elif mutation == 'recovery':
        driver._point_trial_recovery = (trial, ())
    elif mutation == 'shape':
        owner._K = owner._K[:-1, :-1]
    elif mutation == 'errstate':
        old = np.seterr(under='raise')
        try:
            assert handoff.take(driver, trial, factor, pattern) is None
        finally:
            np.seterr(**old)
        return
    else:
        owner._generation += 1
    assert handoff.take(driver, trial, factor, pattern) is None
    assert handoff.take(driver, trial, factor, pattern) is None


@pytest.mark.parametrize('mutation', ('node', 'material', 'prepared', 'joint', 'beam'))
def test_existing_input_guards_still_reject_corruption(assembled, mutation):
    owner, driver, trial, pattern, _result, handoff = assembled
    _eid, element, _slots, _origin = owner._shell_parts[0]
    if mutation == 'node':
        owner._shell_model.mesh.nodes[element.node_ids[0]].x += .001
    elif mutation == 'material':
        owner._shell_model.get_material(element.material_name).elastic_modulus *= 1.01
    elif mutation == 'prepared':
        object.__setattr__(element, '_qualified_component_guard', None)
    elif mutation == 'joint':
        object.__setattr__(owner._connections[0], 'identity', 'changed')
    else:
        object.__setattr__(owner._beam_parts[0][0].operator, 'arithmetic_policy', 'changed')
    with pytest.raises((ValueError, RuntimeError)):
        handoff.take(driver, trial, .01, pattern)
    assert handoff.take(driver, trial, .01, pattern) is None


def test_full_history_restart_and_actual_assembly_counts_match_oracle(monkeypatch):
    definition = replace(make_definition(2, 2), q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    actions = actions_for(definition)
    owners = [ss.create_analysis(definition=definition) for _ in range(3)]
    counts = {}
    original = FinitePointDriver._assemble
    def counted(self, *args, **kwargs):
        counts[id(self.owner)] = counts.get(id(self.owner), 0)+1
        return original(self, *args, **kwargs)
    monkeypatch.setattr(FinitePointDriver, '_assemble', counted)
    try:
        fast, full, resumed = owners
        result = fast.solve(actions, (.01, .02), max_iterations=8, max_backtracks=4)
        saved = fast.checkpoint()
        with monkeypatch.context() as patch:
            patch.setattr(FixedTrialHandoff, 'capture', classmethod(lambda cls, *args, **kwargs: None))
            oracle = full.solve(actions, (.01, .02), max_iterations=8, max_backtracks=4)
        assert counts[id(fast)] < counts[id(full)]
        assert saved == full.checkpoint()
        resumed.restore(saved)
        assert resumed.checkpoint() == saved
        for owner in owners:
            before = owner.checkpoint()
            with pytest.raises(Exception):
                owner.solve(actions, (.03,), max_iterations=1, max_backtracks=1)
            assert owner.checkpoint() == before
        for owner in owners:
            owner.solve(actions, (.03,), max_iterations=8, max_backtracks=4)
        assert fast.checkpoint() == full.checkpoint() == resumed.checkpoint()
        for field in ('shell_displacement', 'beam_positions', 'beam_frames', 'cell_frames',
                      'cell_resultants', 'multipliers', 'support_reactions', 'point_wrenches'):
            a, b = getattr(result, field), getattr(oracle, field)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                assert a == b
    finally:
        for owner in owners:
            owner.close()


@pytest.mark.parametrize('action', ('cancel', 'mutate'))
def test_next_trial_observes_cancellation_before_handoff_validation(action):
    definition = replace(make_definition(2, 2), q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    owner = ss.create_analysis(definition=definition)
    stages = []
    class Observer:
        def raise_if_cancelled(self, stage):
            stages.append(stage)
            if stage == 'stiffened-shell.finite.trial' and stages.count(stage) == 2:
                if action == 'cancel':
                    raise SolveCancelled('candidate discarded before commit')
                element = owner._shell_parts[0][1]
                owner._shell_model.get_material(element.material_name).elastic_modulus *= 1.01
    try:
        expected = SolveCancelled if action == 'cancel' else (ValueError, RuntimeError)
        with pytest.raises(expected):
            owner.solve(actions_for(definition), (.01,), cancellation_token=Observer())
        assert owner._generation == 0
        # Owner discards an unaccepted newly created driver atomically.
        assert owner._finite_driver is None
    finally:
        owner.close()


@pytest.mark.parametrize('target', ('module', 'code'))
def test_changed_dispatch_invalidates_completed_payload(assembled, monkeypatch, target):
    _owner, driver, trial, pattern, _result, handoff = assembled
    from anyfesolver import _stiffened_shell_ge_finite_experimental as module
    if target == 'module':
        monkeypatch.setattr(module, 'q4_algorithmic_response', lambda *a, **kw: None)
    else:
        def alternate(self):
            return None
        monkeypatch.setattr(type(driver)._point_records, '__code__', alternate.__code__)
    assert handoff.take(driver, trial, .01, pattern) is None


def test_observable_factor_subclass_uses_full_path(assembled):
    _owner, driver, trial, pattern, _result, _handoff = assembled
    class Observed(float):
        def __float__(self):
            raise AssertionError('new conversion observation')
    assert FixedTrialHandoff.begin(driver, trial, Observed(.01), pattern) is None


def test_input_change_during_evaluation_cannot_bless_payload(assembled):
    _owner, driver, trial, pattern, result, _handoff = assembled
    before = FixedTrialHandoff.begin(driver, trial, .01, pattern)
    trial.q[0] += 1e-10
    assert FixedTrialHandoff.capture(driver, trial, .01, pattern, result, before=before) is None


@pytest.mark.parametrize('mode', ('call', 'log', 'print', 'warn'))
def test_arithmetic_observers_remain_on_full_evaluation_path(assembled, mode):
    _owner, driver, trial, pattern, _result, handoff = assembled
    old = np.seterr(under=mode)
    try:
        assert FixedTrialHandoff.begin(driver, trial, .01, pattern) is None
        assert handoff.take(driver, trial, .01, pattern) is None
    finally:
        np.seterr(**old)


@pytest.mark.parametrize('target', ('compensated', 'spatial', 'jet'))
def test_nested_admitted_dispatch_invalidates_payload(assembled, monkeypatch, target):
    _owner, driver, trial, pattern, _result, handoff = assembled
    if target == 'compensated':
        from anyfesolver._ge_beam3_p5 import compensated
        monkeypatch.setattr(compensated, 'sum_jets', lambda *a: None)
    elif target == 'spatial':
        from anyfesolver import _ge_beam3_generalized_static_boundary as module
        monkeypatch.setattr(module, '_array', lambda *a: None)
    else:
        from anyfesolver._ge_beam3_mixed_ad import Jet2
        monkeypatch.setattr(Jet2, '__mul__', lambda *a: None)
    assert handoff.take(driver, trial, .01, pattern) is None



def test_changed_decimal_arithmetic_invalidates_payload(assembled):
    from decimal import localcontext, ROUND_DOWN
    _owner, driver, trial, pattern, _result, handoff = assembled
    with localcontext() as context:
        context.rounding = ROUND_DOWN
        assert handoff.take(driver, trial, .01, pattern) is None
