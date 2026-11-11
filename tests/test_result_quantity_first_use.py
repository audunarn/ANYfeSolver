"""Selected stress resolution keeps the general duck-typed contract."""
from types import SimpleNamespace

import numpy as np
import pytest

import anyfesolver.quantities as quantities
from anyfesolver.quantities import QuantityUnavailableError, resolve_result_quantity
from anyfesolver.recovery import StressRecoveryProvenance, StressRecoveryResult
from anyfesolver.results import FEResult


def _carrier(frame, kind="recovery"):
    if kind == "fe":
        return FEResult("selected stress", np.zeros(6), element_stresses=frame)
    return StressRecoveryResult(frame, StressRecoveryProvenance("elastic", "displacement"))


def _legacy(result):
    return resolve_result_quantity(SimpleNamespace(**vars(result)), "stress")


@pytest.mark.parametrize("kind", ["fe", "recovery"])
def test_selected_stress_preserves_native_routing_and_identity(monkeypatch, kind):
    values = np.array([3., 4.])
    frame = {7: {"von_mises": values, "global_xx_top": np.float64(2.)}}
    result = _carrier(frame, kind)
    expected = _legacy(result)
    calls = []
    original = quantities._physical_stress_frame
    def observed(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(quantities, "_physical_stress_frame", observed)
    actual = resolve_result_quantity(result, "stress")
    assert len(calls) == (5 if kind == "fe" else 1)
    assert actual.descriptor == expected.descriptor
    assert actual.data is frame
    assert actual.data[7]["von_mises"] is values
    values[0] = np.nan
    with pytest.raises(QuantityUnavailableError, match="carries no result quantity 'stress'"):
        resolve_result_quantity(result, "stress")


@pytest.mark.parametrize("frame", [
    {}, {True: {"von_mises": 1.}}, {"01": {"von_mises": 1.}},
    {1: {"von_mises": 1.}, "1": {"von_mises": 2.}},
    {1: {}}, {1: {"von_mises": np.array([np.inf])}},
    {1: {"von_mises": True}}, {1: {"von_mises": np.array(["1."])}},
    {1: {"von_mises": 2., "membrane_xx": np.array([np.nan])}},
    {1: {"membrane_resultants": np.ones(3)}},
])
def test_native_refusals_match_general_route(frame):
    result = _carrier(frame)
    with pytest.raises(QuantityUnavailableError) as expected:
        _legacy(result)
    with pytest.raises(QuantityUnavailableError) as actual:
        resolve_result_quantity(result, "stress")
    # Class name differs across carriers; the diagnostic/refusal category agrees.
    assert str(actual.value).partition(" carries ")[2] == str(expected.value).partition(" carries ")[2]


def test_filtered_bases_exclusions_and_nonphysical_values_match():
    frame = {
        "7": {"von_mises": np.ones(2), "membrane_xx": np.ones(2),
              "global_xx_top": np.ones(2), "membrane_strain_xx": np.array([np.nan])},
        8: {"physical_stress_available": False, "von_mises": np.array([np.nan])},
        9: {"recovery_scope": "section_resultants_only", "membrane_resultants": np.ones(3)},
    }
    result = _carrier(frame)
    actual, expected = resolve_result_quantity(result, "stress"), _legacy(result)
    assert actual.descriptor == expected.descriptor
    assert set(actual.data) == {7}
    assert actual.descriptor.metadata["excluded_nonphysical_element_ids"] == (8, 9)
    assert set(actual.data[7]) == set(expected.data[7])
    for key, value in actual.data[7].items():
        assert value is expected.data[7][key]
    frame["7"]["von_mises"][0] = np.nan
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(result, "stress")


@pytest.mark.parametrize("custom", ["mapping", "array", "result", "injected", "state"])
def test_callback_capable_carriers_keep_general_route(monkeypatch, custom):
    class CustomDict(dict):
        pass
    class CustomArray(np.ndarray):
        pass
    class CustomRecovery(StressRecoveryResult):
        pass
    result = _carrier({1: {"von_mises": np.ones(1)}})
    if custom == "mapping":
        result.element_stresses = CustomDict(result.element_stresses)
    elif custom == "array":
        result.element_stresses[1]["von_mises"] = np.ones(1).view(CustomArray)
    elif custom == "result":
        result = CustomRecovery(result.element_stresses, result.provenance)
    elif custom == "state":
        result.committed_element_states = {1: {"alpha": np.zeros(1)}}
    else:
        result.snapshots = ()
    calls = []
    original = quantities._physical_stress_frame
    def observed(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(quantities, "_physical_stress_frame", observed)
    actual = resolve_result_quantity(result, "stress")
    assert actual.data is result.element_stresses
    assert len(calls) == 5


def test_dynamic_carrier_retains_reread_refusal():
    class Changing:
        reads = 0
        @property
        def element_stresses(self):
            self.reads += 1
            return {1: {"von_mises": np.array([1. if self.reads < 2 else np.nan])}}
    result = Changing()
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(result, "stress")
    assert result.reads == 2


def test_history_alignment_and_mutation_still_fail_closed():
    frame = {1: {"von_mises": np.ones(1)}}
    result = SimpleNamespace(stress_history=[frame, frame], times=np.array([0., 1.]))
    resolved = resolve_result_quantity(result, "stress_history")
    assert resolved.data == (frame, frame)
    assert resolved.descriptor.frame_count == 2
    result.times = np.array([0.])
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(result, "stress_history")
    result.times = np.array([0., 1.])
    frame[1]["von_mises"][0] = np.nan
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(result, "stress_history")


def test_injected_conversion_callback_cannot_leave_stale_admission():
    values = np.ones(1)
    result = _carrier({1: {"von_mises": values}})
    class MutatingDisplacement:
        def __array__(self, dtype=None, copy=None):
            values[0] = np.nan
            return np.zeros(6, dtype=dtype)
    # The native dataclass is mutable; an injected descriptor source can run
    # user code before stress admission in the general descriptor inventory.
    result.displacements = MutatingDisplacement()
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(result, "stress")
    assert np.isnan(values[0])


@pytest.mark.parametrize("custom", ["instance_dictionary", "field_definitions"])
def test_native_admission_does_not_invoke_injected_field_protocols(custom):
    values = np.array([np.nan])
    result = _carrier({1: {"von_mises": values}})
    calls = []
    class MutatingFields(dict):
        def keys(self):
            calls.append("keys")
            values[0] = 1.
            return super().keys()
    class MutatingDefinitions:
        def __iter__(self):
            calls.append("iter")
            values[0] = 1.
            return iter(StressRecoveryResult.__dataclass_fields__)
    if custom == "instance_dictionary":
        result.__dict__ = MutatingFields(vars(result))
    else:
        result.__dataclass_fields__ = MutatingDefinitions()
    # Baseline inventory never consults either injected protocol.
    assert "stress" not in {
        item.quantity_id for item in quantities.describe_result_quantities(result)
    }
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(result, "stress")
    assert not calls
    assert np.isnan(values[0])


@pytest.mark.parametrize("dynamic", ["field_property", "getattribute"])
def test_exact_native_class_dynamic_access_keeps_reread_refusal(monkeypatch, dynamic):
    values = np.ones(1)
    result = _carrier({1: {"von_mises": values}})
    original_fields = vars(result)
    reads = []
    def changed_frame():
        reads.append("frame")
        if len(reads) > 1:
            values[0] = np.nan
        return original_fields["element_stresses"]
    if dynamic == "field_property":
        monkeypatch.setattr(StressRecoveryResult, "element_stresses",
                            property(lambda self: changed_frame()), raising=False)
    elif dynamic == "getattribute":
        def dynamic_getattribute(self, name):
            if name == "element_stresses":
                return changed_frame()
            return object.__getattribute__(self, name)
        monkeypatch.setattr(StressRecoveryResult, "__getattribute__", dynamic_getattribute)
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(result, "stress")
    assert len(reads) == 2


def test_injected_energy_class_property_retains_inventory_refusal(monkeypatch):
    values = np.array([np.nan])
    result = _carrier({1: {"von_mises": values}})
    calls = []
    def dynamic_diagnostics(self):
        calls.append("energy")
        values[0] = 1.
        return {}
    monkeypatch.setattr(StressRecoveryResult, "diagnostics",
                        property(dynamic_diagnostics), raising=False)
    # The original inventory sees the invalid stress before any energy probe.
    # A selected native route must not skip that inventory callback sequence.
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(result, "stress")
    assert calls
    assert values[0] == 1.


@pytest.mark.parametrize("location", ["class_fields", "instance_fields"])
def test_native_layout_does_not_hash_custom_field_names(monkeypatch, location):
    values = np.array([np.nan])
    result = _carrier({1: {"von_mises": values}})
    calls = []
    class CustomName(str):
        def __hash__(self):
            calls.append("hash")
            values[0] = 1.
            return super().__hash__()
    if location == "class_fields":
        fields = dict(StressRecoveryResult.__dataclass_fields__)
        fields[CustomName("extra")] = None
        monkeypatch.setattr(StressRecoveryResult, "__dataclass_fields__", fields)
    else:
        vars(result)[CustomName("extra")] = None
    calls.clear()
    values[0] = np.nan
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(result, "stress")
    assert not calls
    assert np.isnan(values[0])


def test_extended_dataclass_metadata_cannot_admit_dynamic_inputs(monkeypatch):
    values = np.ones(1)
    result = _carrier({1: {"von_mises": values}})
    calls = []
    class DynamicDiagnostics(dict):
        def get(self, key, default=None):
            calls.append(key)
            values[0] = np.nan
            return super().get(key, default)
    definitions = dict(StressRecoveryResult.__dataclass_fields__)
    definitions["diagnostics"] = None
    monkeypatch.setattr(StressRecoveryResult, "__dataclass_fields__", definitions)
    result.diagnostics = DynamicDiagnostics()
    with pytest.raises(QuantityUnavailableError):
        resolve_result_quantity(result, "stress")
    assert calls
    assert np.isnan(values[0])
