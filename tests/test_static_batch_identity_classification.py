"""static_batch classifies values and classes by identity, never by ==, hash or getattr.

A forged class whose metaclass claims equality with everything must not be
accepted as a plain scalar or container, and an unvalidated object's __dict__
property or __hash__ must not run while the batch snapshots a load or the
preparation witness fingerprints the model.
"""

from __future__ import annotations

import numpy as np

from anyfesolver.boundary import LoadCase
from anyfesolver import static_batch as sb
from test_load_kernel_v2d_pressure_acceleration import _mixed_model


class _EqMeta(type):
    def __eq__(cls, other):  # noqa: D105
        return True

    __hash__ = type.__hash__


def _forged():
    log: list = []

    class Forged(metaclass=_EqMeta):
        def __float__(self):
            log.append("float")
            return 1.0

    return Forged(), log


def test_snapshot_refuses_a_forged_pressure_value() -> None:
    value, log = _forged()
    case = LoadCase("forged")
    case.pressure_loads = {1: value}
    assert sb._snapshot_plain_load(case) == (False, None)
    nested = LoadCase("forged-nested")
    nested.pressure_loads = {1: (1.0, value)}
    assert sb._snapshot_plain_load(nested) == (False, None)
    assert log == []


def test_snapshot_of_ordinary_loads_is_unchanged() -> None:
    case = LoadCase("ordinary")
    case.pressure_loads = {1: 2.5, 2: (1.0, 2.0)}
    case.nodal_loads = {3: np.arange(6, dtype=float)}
    ok, clone = sb._snapshot_plain_load(case)
    assert ok and clone is not case
    assert clone.pressure_loads == case.pressure_loads
    assert np.array_equal(clone.nodal_loads[3], case.nodal_loads[3])
    assert sb._snapshot_plain_load(None) == (True, None)


def test_plain_state_does_not_pass_a_forged_object_through() -> None:
    value, log = _forged()
    state = sb._plain_state([value, {"k": value}])
    assert state[0] is not value and state[0][0] is type(value)
    assert log == []
    assert sb._plain_state({"a": 1, "b": [1.5, "x", None, True]}) == (
        ("a", 1), ("b", (1.5, "x", None, True)))


def test_plain_state_never_runs_a_foreign_dict_property() -> None:
    log: list = []

    class Sneaky:
        @property
        def __dict__(self):  # noqa: D401
            log.append("dict")
            return {"x": 1}

    state = sb._plain_state(Sneaky())
    assert log == [] and state[0] is Sneaky


def test_class_state_ignores_a_forged_member_type() -> None:
    value, _log = _forged()

    class Holder:
        member = value

    assert sb._class_state(Holder)  # a forged type is not mistaken for staticmethod


def test_model_state_never_hashes_foreign_classes() -> None:
    log: list = []

    class HashMeta(type):
        def __hash__(cls):  # noqa: D105
            log.append("hash")
            return 1

        def __eq__(cls, other):  # noqa: D105
            log.append("eq")
            return cls is other

    class Generic(metaclass=HashMeta):
        element_id = 99
        node_ids = (1, 2)
        material_name = "steel"

    model, q4, v2d = _mixed_model(2)
    model.mesh.elements[99] = Generic()
    log.clear()
    sb._model_state(model)
    assert log == []


def _metaclass_hooks():
    """Instances whose metaclass defines __mro__ / __dict__ properties."""

    hit: list = []

    class Meta(type):
        @property
        def __mro__(cls):  # noqa: D401
            hit.append("mro")
            return (cls, object)

        @property
        def __dict__(cls):  # noqa: D401
            hit.append("dict")
            return {}

    class Hooked(metaclass=Meta):
        pass

    return Hooked, hit


def test_metaclass_mro_and_dict_properties_are_never_run() -> None:
    cls, hit = _metaclass_hooks()
    # _plain_state: the instance is opaque, nothing from the metaclass runs.
    state = sb._plain_state(cls())
    assert hit == [] and state[0] is cls
    # _snapshot_plain_load: a foreign object in a load is refused without inspection.
    case = LoadCase("hooked")
    case.pressure_loads = {1: cls()}
    assert sb._snapshot_plain_load(case) == (False, None)
    assert hit == []
    # class fingerprints and the model witness read slots, not the properties
    assert sb._class_state(cls)
    model, q4, v2d = _mixed_model(2)
    element = cls()
    element.element_id = 99
    model.mesh.elements[99] = element
    sb._model_state(model)
    assert hit == []


def test_ordinary_class_inspection_is_unchanged() -> None:
    class Plain:
        x = 1

    assert sb._class_mro(Plain) == Plain.__mro__
    assert sb._class_namespace(Plain).get("x") == 1
    assert sb._namespace_of(Plain()) == {}
    plain = Plain()
    plain.y = 2
    assert sb._namespace_of(plain) == {"y": 2}
