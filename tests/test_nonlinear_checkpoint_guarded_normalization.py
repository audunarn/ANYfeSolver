"""Adversarial checkpoint observations retain their authority boundary."""

import pytest
import numpy as np

from anyfesolver.nonlinear_restart import _normalize_json


def test_custom_sequence_iteration_is_guarded_after_observation():
    observed = []

    class ChangingSequence(list):
        def __iter__(self):
            observed.append("iteration")
            return super().__iter__()

    def guard(_model, *, context):
        if observed:
            raise RuntimeError(context)

    with pytest.raises(RuntimeError, match="sequence observation"):
        _normalize_json(ChangingSequence([1.0]), _exact_guard=guard,
                        _guard_model=object())
    assert observed == ["iteration"]


def test_custom_scalar_conversion_is_guarded_after_observation():
    observed = []

    class ChangingInteger(int):
        def __int__(self):
            observed.append("conversion")
            return super().__int__()

    def guard(_model, *, context):
        if observed:
            raise RuntimeError(context)

    with pytest.raises(RuntimeError, match="scalar observation"):
        _normalize_json(ChangingInteger(3), _exact_guard=guard,
                        _guard_model=object())
    assert observed == ["conversion"]


def test_exact_dictionary_avoids_redundant_observation_checks():
    observed = []
    payload = {"nested": {"values": [1, 0.0, -0.0, True, None, "value"]}}
    normalized = _normalize_json(payload, _exact_guard=lambda *a, **k: observed.append(k),
                                 _guard_model=object())
    assert normalized == payload and normalized is not payload
    assert normalized["nested"] is not payload["nested"]
    assert observed == []


@pytest.mark.parametrize("member_kind", ["mapping", "scalar", "sequence"])
def test_exact_dictionary_retains_nested_effectful_member_boundary(member_kind):
    observed = []

    class ChangingMapping(dict):
        def items(self):
            observed.append("mapping")
            return super().items()

    class ChangingInteger(int):
        def __int__(self):
            observed.append("scalar")
            return super().__int__()

    class ChangingSequence(list):
        def __iter__(self):
            observed.append("sequence")
            return super().__iter__()

    members = {"mapping": ChangingMapping(value=1), "scalar": ChangingInteger(3),
               "sequence": ChangingSequence([1.0])}

    def guard(_model, *, context):
        if observed:
            raise RuntimeError(context)

    with pytest.raises(RuntimeError, match=member_kind + " observation"):
        _normalize_json({"plain": {"effectful": members[member_kind]}},
                        _exact_guard=guard, _guard_model=object())
    assert observed == [member_kind]


def test_nonexact_dictionary_key_retains_mapping_check():
    class StringKey(str):
        pass

    observed = []
    _normalize_json({StringKey("value"): 1},
                    _exact_guard=lambda *a, **k: observed.append(k["context"]),
                    _guard_model=object())
    assert len(observed) == 1 and "mapping observation" in observed[0]


def test_exact_array_uses_pure_observation_and_keeps_canonical_values():
    observed = []
    result = _normalize_json(np.array([0.0, -0.0, 1.25]),
        _exact_guard=lambda *a, **k: observed.append(k), _guard_model=object())
    assert result == [0.0, 0.0, 1.25] and observed == []


def test_array_subclass_retains_observation_boundary():
    class Array(np.ndarray):
        pass
    observed = []
    _normalize_json(np.array([1.25]).view(Array),
        _exact_guard=lambda *a, **k: observed.append(k), _guard_model=object())
    assert len(observed) == 1 and "array observation" in observed[0]["context"]


def test_restoring_conversion_cannot_hide_its_observation(monkeypatch):
    original = np.asarray
    changed = []
    def conversion(*a, **k):
        changed.append(True)
        monkeypatch.setattr(np, "asarray", original)
        return original(*a, **k)
    def guard(*a, **k):
        if changed:
            raise RuntimeError("conversion observed")
    monkeypatch.setattr(np, "asarray", conversion)
    with pytest.raises(RuntimeError, match="conversion observed"):
        _normalize_json(np.array([1.25]), _exact_guard=guard, _guard_model=object())


def test_object_array_keeps_effectful_member_boundary():
    changed = []
    class Integer(int):
        def __int__(self):
            changed.append(True)
            return 3
    def guard(*a, **k):
        if changed:
            raise RuntimeError("member observed")
    with pytest.raises(RuntimeError, match="member observed"):
        _normalize_json(np.array([Integer(3)], dtype=object),
                        _exact_guard=guard, _guard_model=object())


def test_explicit_array_cast_is_inert_only_for_exact_native_same_dtype():
    from anyfesolver.nonlinear_restart import _inert_array_conversion
    for dtype in (np.float64, np.int64):
        assert _inert_array_conversion(np.ones(3, dtype=dtype), np.asarray, dtype)
    assert not _inert_array_conversion(np.ones(3, dtype=np.float32), np.asarray, np.float64)
    assert not _inert_array_conversion(np.array([1], dtype=object), np.asarray, np.int64)
    assert not _inert_array_conversion(np.ones(3, dtype=">f8"), np.asarray, np.float64)
    assert not _inert_array_conversion(np.ones(3, dtype=complex), np.asarray, np.float64)
    assert not _inert_array_conversion(np.zeros(3, dtype=[("x", "f8")]),
                                       np.asarray, np.float64)
    assert not _inert_array_conversion([1, 2], np.asarray, np.int64)


def test_custom_numpy_module_lookup_is_not_an_inert_observation(monkeypatch):
    from types import ModuleType
    from anyfesolver import nonlinear_restart as restart
    class CustomModule(ModuleType):
        def __getattribute__(self, name):
            return super().__getattribute__(name)
    proxy = CustomModule("numpy_proxy")
    proxy.ndarray, proxy.asarray = np.ndarray, np.asarray
    monkeypatch.setattr(restart, "np", proxy)
    assert not restart._inert_array_conversion(np.array([1.0]), np.asarray)


def test_restoring_module_getter_cannot_hide_its_observation(monkeypatch):
    from types import ModuleType
    from anyfesolver import nonlinear_restart as restart
    observed = []
    class CustomModule(ModuleType):
        def __getattribute__(self, name):
            value = super().__getattribute__(name)
            if name == "integer":
                observed.append(True)
                self.__class__ = ModuleType
            return value
    proxy = CustomModule("numpy_proxy")
    vars(proxy).update(vars(np))
    monkeypatch.setattr(restart, "np", proxy)
    def guard(*a, **k):
        if observed:
            raise RuntimeError("getter observed")
    with pytest.raises(RuntimeError, match="getter observed"):
        _normalize_json(np.array([1.0]), _exact_guard=guard, _guard_model=object())


@pytest.mark.parametrize("attribute", ["asarray", "integer", "float64"])
def test_restoring_module_property_cannot_hide_its_observation(monkeypatch, attribute):
    from types import ModuleType
    from anyfesolver import nonlinear_restart as restart
    observed = []
    class CustomModule(ModuleType):
        pass
    def descriptor(self):
        observed.append(True)
        delattr(CustomModule, attribute)
        return getattr(np, attribute)
    setattr(CustomModule, attribute, property(descriptor))
    proxy = CustomModule("numpy_proxy")
    vars(proxy).update(vars(np))
    monkeypatch.setattr(restart, "np", proxy)
    assert not restart._numpy_lookup_inert()
    if attribute == "float64":
        conversion, dtype = proxy.asarray, proxy.float64
        assert not restart._inert_array_conversion(np.array([1.0]), conversion, dtype,
                                                   _lookup_inert=False)
        assert observed
    else:
        def guard(*a, **k):
            if observed:
                raise RuntimeError("property observed")
        with pytest.raises(RuntimeError, match="property observed"):
            _normalize_json(np.array([1.0]), _exact_guard=guard, _guard_model=object())


def test_custom_module_metaclass_descriptor_is_rejected_without_observation(monkeypatch):
    from types import ModuleType
    from anyfesolver import nonlinear_restart as restart
    observed = []
    class Meta(type):
        @property
        def __mro__(cls):
            observed.append(True)
            return type.__getattribute__(cls, "__dict__").get("real_mro", ())
    class CustomModule(ModuleType, metaclass=Meta):
        pass
    proxy = CustomModule("numpy_proxy")
    vars(proxy).update(vars(np))
    monkeypatch.setattr(restart, "np", proxy)
    assert not restart._numpy_lookup_inert()
    assert observed == []


def test_module_getattribute_descriptor_is_rejected_without_observation():
    from types import ModuleType
    from anyfesolver import nonlinear_restart as restart
    observed = []
    class Descriptor:
        def __get__(self, instance, owner):
            observed.append(True)
            return ModuleType.__getattribute__
    class CustomModule(ModuleType):
        __getattribute__ = Descriptor()
    proxy = CustomModule("numpy_proxy")
    original = restart.np
    try:
        # Avoid pytest inspecting the deliberately malformed fixture itself.
        restart.np = proxy
        assert not restart._numpy_lookup_inert()
        assert observed == []
    finally:
        restart.np = original
