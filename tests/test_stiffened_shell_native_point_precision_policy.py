"""Checkpoint-bound native point staged-force polishing policy controls.

Every control drives the public staged solve/checkpoint/restore/cancellation
facade of the experimental GE owner; the synthetic gate test only narrows
the binding seam and is not acceptance by itself. Physical limits, the
constitutive law and the comparator are unchanged.
"""
from types import SimpleNamespace

import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver import _stiffened_shell_ge_finite_experimental as finite
from test_stiffened_shell_native_plastic_chart import definition, programme
from test_stiffened_shell_ge_physical_fibre_linear import _definition as _fibre_panel

OPTIONS = dict(max_iterations=6, max_backtracks=3)
CONTROL = ss.AdaptiveIncrementControl(2, 8)
BEAM_FORCE = (ss.NodalAction('beam', 105, (0., 0., 1000., 0., 0., 0.)),)


def _owner(request, model=None, *, selection=None):
    made = ss.create_analysis(definition=model if model is not None else definition())
    request.addfinalizer(made.close)
    if selection is not None:
        finite.select_admissible_polish_candidate(made, selection)
    return made


def _variant_model():
    # A second in-domain model with a different identity.
    from dataclasses import replace
    return replace(_fibre_panel(mixed=True),
                   q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)


def _fresh(owner):
    assert owner._finite_driver is None and owner._accepted is None
    return owner


def test_explicit_selection_remains_binding(request):
    model = definition()
    tagged = _owner(request, model, selection=True)
    tagged.solve(programme(tagged), (.001,), **OPTIONS)
    raw = tagged.checkpoint()
    refused = _fresh(_owner(request, model, selection=False))
    with pytest.raises(ValueError):
        refused.restore(raw)
    assert refused._accepted is None and refused._finite_driver is None
    restored = _owner(request, model, selection=True)
    restored.restore(raw)
    assert restored.checkpoint() == raw
    legacy = _owner(request, model, selection=False)
    legacy.solve(programme(legacy), (.001,), **OPTIONS)
    legacy_raw = legacy.checkpoint()
    receiver = _fresh(_owner(request, model, selection=True))
    with pytest.raises(ValueError):
        receiver.restore(legacy_raw)
    assert receiver._accepted is None and receiver._finite_driver is None


def test_fresh_staged_force_solve_auto_binds_the_candidate(request):
    model = definition()
    auto = _owner(request, model)
    auto.solve(programme(auto), (.001,), increment_control=CONTROL, **OPTIONS)
    assert auto._finite_driver._admissible_polish is True
    raw = auto.checkpoint()
    # The auto policy is the SAME tagged policy as the explicit candidate.
    explicit = _owner(request, model, selection=True)
    explicit.solve(programme(explicit), (.001,), increment_control=CONTROL, **OPTIONS)
    assert explicit.checkpoint() == raw
    # Tagged bytes are refused without the policy: the tag is real.
    refused = _fresh(_owner(request, model, selection=False))
    with pytest.raises(ValueError):
        refused.restore(raw)
    assert refused._accepted is None and refused._finite_driver is None


def test_auto_binding_gate_is_staged_fresh_and_unset(monkeypatch):
    owner = SimpleNamespace(_admissible_polish_candidate=False,
                            _admissible_polish_selection=None)
    driver = object.__new__(finite.FinitePointDriver)
    driver.owner = owner
    driver._admissible_polish = False
    driver._accepted = None
    driver._compensated = False
    monkeypatch.setattr(finite, 'supports_native_chart_polish', lambda o: o is owner)
    driver._bind_staged_force_polish(object())  # unset fresh staged binds
    assert driver._admissible_polish is True
    driver._admissible_polish = False
    driver._bind_staged_force_polish(None)  # plain force never auto-binds
    assert driver._admissible_polish is False
    owner._admissible_polish_selection = False
    driver._bind_staged_force_polish(object())  # explicit False stays binding
    assert driver._admissible_polish is False
    owner._admissible_polish_selection = True
    driver._bind_staged_force_polish(object())  # explicit True never auto-binds
    assert driver._admissible_polish is False
    owner._admissible_polish_selection = None
    driver._accepted = object()
    driver._bind_staged_force_polish(object())  # accepted state never switches
    assert driver._admissible_polish is False
    driver._accepted = None
    driver._compensated = True
    driver._bind_staged_force_polish(object())  # compensated stays outside
    assert driver._admissible_polish is False


def test_plain_force_route_stays_unpolished(request):
    model = definition()
    owner = _owner(request, model)
    owner.solve(BEAM_FORCE, (.5,), **OPTIONS)
    assert owner._finite_driver._admissible_polish is False
    raw = owner.checkpoint()
    selected = _fresh(_owner(request, model, selection=True))
    with pytest.raises(ValueError):
        selected.restore(raw)
    fresh = _owner(request, model)
    fresh.restore(raw)
    assert fresh._finite_driver._admissible_polish is False
    assert fresh.checkpoint() == raw


def test_raw_legacy_staged_checkpoint_exact_restart(request):
    model = definition()
    legacy = _owner(request, model, selection=False)
    legacy.solve(programme(legacy), (.001,), increment_control=CONTROL, **OPTIONS)
    raw = legacy.checkpoint()
    fresh = _owner(request, model)
    result = fresh.restore(raw)
    assert fresh._finite_driver._admissible_polish is False
    assert fresh.checkpoint() == raw
    assert result.load_position == .001
    # Old first-admissible continuation stays byte-identical.
    legacy.solve(programme(legacy), (.002,), increment_control=CONTROL, **OPTIONS)
    fresh.solve(programme(fresh), (.002,), increment_control=CONTROL, **OPTIONS)
    assert fresh.checkpoint() == legacy.checkpoint()


def test_tagged_auto_roundtrip_through_adaptive_and_chart_wrappers(request):
    model = definition()
    control = _owner(request, model)
    control.solve(programme(control), (.001,), increment_control=CONTROL, **OPTIONS)
    raw = control.checkpoint()
    reset = _owner(request, model)
    reset.restore(raw)
    assert reset._finite_driver._admissible_polish is True
    # The inferred policy is checkpoint-bound to the restored driver; the
    # live owner itself never gains an explicit selection.
    assert getattr(reset, '_admissible_polish_candidate', None) is None
    assert getattr(reset, '_admissible_polish_selection', None) is None
    assert reset.checkpoint() == raw
    event = reset.rebase_rotations(reset.recover().generation)
    chart_raw = reset.checkpoint()
    chart = _owner(request, model)
    chart.restore(chart_raw)
    assert chart._finite_driver._admissible_polish is True
    assert chart.checkpoint() == chart_raw
    # Polished continuation is identical through both tagged wrapper layers.
    for branch in (reset, chart):
        branch.solve(programme(branch), (.002,), increment_control=CONTROL, **OPTIONS)
    assert reset.checkpoint() == chart.checkpoint()


def test_refusals_preserve_the_fresh_owner(request):
    model = definition()
    control = _owner(request, model)
    control.solve(programme(control), (.001,), increment_control=CONTROL, **OPTIONS)
    raw = control.checkpoint()
    with pytest.raises(ValueError):
        _fresh(_owner(request, model)).restore(raw[:-1])
    # Forged framing: a readable tagged header with a broken digest infers
    # the mode on the private copy, then the validated decode refuses.
    with pytest.raises(ValueError):
        _fresh(_owner(request, model)).restore(raw[:-32] + b"\x00"*32)
    # Tagged bytes of a different in-domain model never restore.
    variant = _fresh(_owner(request, _variant_model()))
    with pytest.raises(ValueError):
        variant.restore(raw)
    assert variant._accepted is None and variant._finite_driver is None


def test_stale_generation_and_honest_counters(request):
    model = definition()
    control = _owner(request, model)
    result = control.solve(programme(control), (.001,),
        increment_control=CONTROL, **OPTIONS)
    raw = control.checkpoint()
    attempt = control._finite_driver._adaptive_state.attempts[-1]
    assert attempt.status == 'accepted'
    assert attempt.iterations == attempt.linear_solves
    assert attempt.evaluations >= attempt.iterations + 1
    assert attempt.rejected_trials <= (OPTIONS['max_backtracks'] + 1) * attempt.linear_solves
    assert attempt.equilibrium_ratio <= 1e-8 and attempt.constraint_max <= 1e-9
    assert result.generation == 1
    reset = _owner(request, model)
    reset.restore(raw)
    with pytest.raises(ValueError, match='stale'):
        reset.rebase_rotations(result.generation - 1)
    assert reset.checkpoint() == raw
    event = reset.rebase_rotations(result.generation)
    assert event.generation == result.generation + 1


def test_cancellation_discards_uncommitted_candidate_and_keeps_prefix(request):
    model = definition()
    control = _owner(request, model)
    token = CancellationToken()
    seen = []

    def cancel_after_first(result):
        seen.append(result.load_position)
        token.cancel('policy control cancellation')

    with pytest.raises(SolveCancelled):
        control.solve(programme(control), (.001, .002),
            increment_control=CONTROL, on_accepted=cancel_after_first,
            cancellation_token=token, **OPTIONS)
    assert seen == [.001]
    prefix = control.recover()
    assert prefix.load_position == .001 and prefix.generation == 1
    raw = control.checkpoint()
    fresh = _owner(request, model)
    fresh.restore(raw)
    assert fresh._finite_driver._admissible_polish is True
    assert fresh.checkpoint() == raw
