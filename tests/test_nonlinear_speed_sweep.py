"""Configuration ownership must skip only observations that cannot run hooks."""
from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import nonlinear_static as ns
from test_nonlinear_static_armijo import _spring_model, _armijo_settings


def _config(guard, **overrides):
    values = dict(max_load_factor=1.0, num_steps=5, max_iterations=20,
                  tolerance=1e-9, num_layers=5, min_step_fraction=0.01,
                  control="force", kinematics="von_karman",
                  corotational_tangent="auto", equilibrate_initial_state=None,
                  record_increment_snapshots=False, emit_restart_checkpoint=False)
    values.update(overrides)
    return ns._owned_nonlinear_static_operation_config(
        object(), _exact_guard=guard, **values)


def test_plain_configuration_keeps_final_guard_and_matches_reference(monkeypatch):
    calls = []
    guard = lambda model, *, context: calls.append(context)
    candidate = _config(guard)
    assert calls == ["nonlinear static owned configuration"]
    calls.clear()
    monkeypatch.setattr(ns, "_inert_nonlinear_config_scalar", lambda value: False)
    assert _config(guard) == candidate
    assert len(calls) == 12


@pytest.mark.parametrize("field,value", [
    ("tolerance", np.float64(1e-9)), ("num_steps", np.int64(5)),
    ("record_increment_snapshots", np.bool_(False)),
])
def test_numpy_scalar_conversion_retains_immediate_guard(field, value):
    calls = []
    _config(lambda model, *, context: calls.append(context), **{field: value})
    assert calls == [f"nonlinear static {field} conversion",
                     "nonlinear static owned configuration"]


@pytest.mark.parametrize("kind", ["float", "int", "str", "bool"])
def test_hostile_conversion_stops_before_next_observation(kind):
    events = []
    class Float(float):
        def __float__(self):
            events.append("mutation")
            return 1.0
    class Int(int):
        def __int__(self):
            events.append("mutation")
            return 5
    class String(str):
        def __str__(self):
            events.append("mutation")
            return "force"
    class Flag:
        def __bool__(self):
            events.append("mutation")
            return False
    field, value = {"float": ("max_load_factor", Float(1)),
                    "int": ("num_steps", Int(5)),
                    "str": ("control", String("force")),
                    "bool": ("record_increment_snapshots", Flag())}[kind]
    def guard(model, *, context):
        events.append(context)
        if "mutation" in events:
            raise RuntimeError("mutation refused")
    with pytest.raises(RuntimeError, match="mutation refused"):
        _config(guard, **{field: value})
    assert events == ["mutation", f"nonlinear static {field} conversion"]


def test_custom_metaclass_is_not_observed_by_inert_classifier():
    class Meta(type):
        def __eq__(self, other):
            raise AssertionError("type equality executed")
        def __hash__(self):
            raise AssertionError("type hash executed")
    class Foreign(metaclass=Meta):
        pass
    assert not ns._inert_nonlinear_config_scalar(Foreign())


def test_replaced_converter_keeps_immediate_guard(monkeypatch):
    events = []
    def replaced_float(value):
        events.append("mutation")
        return float(value)
    monkeypatch.setattr(ns, "float", replaced_float, raising=False)
    def guard(model, *, context):
        events.append(context)
        raise RuntimeError("mutation refused")
    with pytest.raises(RuntimeError, match="mutation refused"):
        _config(guard)
    assert events == ["mutation", "nonlinear static max_load_factor conversion"]


@pytest.mark.parametrize("field", ["control", "kinematics"])
def test_string_subclass_lower_is_guarded_before_later_conversion(field):
    events = []
    class LowerHook(str):
        def lower(self):
            events.append("lower mutation")
            return "force" if field == "control" else "von_karman"
    class StringInput:
        def __str__(self):
            events.append("string conversion")
            return LowerHook("VALUE")
    class LaterObserver:
        def __str__(self):
            events.append("forbidden later observation")
            return "auto"
    def guard(model, *, context):
        events.append(context)
        if "lower mutation" in events:
            raise RuntimeError("lower mutation refused")
    with pytest.raises(RuntimeError, match="lower mutation refused"):
        _config(guard, **{field: StringInput(),
                         "corotational_tangent": LaterObserver()})
    assert events == ["string conversion", f"nonlinear static {field} conversion",
                      "lower mutation", f"nonlinear static {field} normalization"]


def test_complete_armijo_route_matches_every_conversion_guard(monkeypatch):
    def solve():
        model, load = _spring_model()
        return ns.solve_static_nonlinear(
            model, load, num_steps=1, max_iterations=20, tolerance=1e-12,
            convergence_settings=_armijo_settings())
    candidate = solve()
    monkeypatch.setattr(ns, "_inert_nonlinear_config_scalar", lambda value: False)
    reference = solve()
    assert candidate.status == reference.status == "completed"
    np.testing.assert_array_equal(candidate.displacements, reference.displacements)
    assert candidate.load_factor == reference.load_factor
    assert candidate.element_states == reference.element_states
    assert len(candidate.steps) == len(reference.steps)
    for actual, expected in zip(candidate.steps, reference.steps):
        assert actual.load_factor == expected.load_factor
        assert actual.residual_norm == expected.residual_norm
        assert actual.displacement_norm == expected.displacement_norm
        assert actual.support_reactions.keys() == expected.support_reactions.keys()
        for name in actual.support_reactions:
            np.testing.assert_array_equal(actual.support_reactions[name],
                                          expected.support_reactions[name])
