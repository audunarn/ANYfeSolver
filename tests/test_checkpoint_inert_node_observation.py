"""Built-in node optimization must preserve every callback boundary."""

import numpy as np
import pytest

from anyfesolver.fe_core import FEModel, Node
from anyfesolver import nonlinear_restart as restart


def model_and_node():
    model = FEModel("checkpoint_node_probe")
    node = Node(1, 0.25, 0.5, -0.0, [0, 1, 2])
    model.mesh.nodes[1] = node
    return model, node


def fail_on_observation(changed):
    def guard(_model, *, context):
        if changed:
            raise RuntimeError(context)
    return guard


def test_plain_node_descriptor_has_exact_coordinates_and_keeps_outer_guards():
    model, node = model_and_node()
    assert restart._inert_node_observation(node)
    contexts = []
    result = restart.model_configuration_descriptor(model,
        _exact_guard=lambda _model, **kw: contexts.append(kw["context"]))
    assert result["nodes"] == [{"id": 1, "coordinates": [0.25, 0.5, -0.0], "dofs": [0, 1, 2]}]
    assert "nonlinear checkpoint node mapping observation" in contexts
    assert any("DOF array observation" in context for context in contexts)
    assert not any(context.endswith("coordinate observation")
                   or context.endswith("DOF observation") for context in contexts)


def test_changed_node_method_keeps_its_post_observation_guard(monkeypatch):
    model, _ = model_and_node()
    changed = []
    original = Node.coords
    def coords(self):
        changed.append(True)
        monkeypatch.setattr(Node, "coords", original)
        return original(self)
    monkeypatch.setattr(Node, "coords", coords)
    with pytest.raises(RuntimeError, match="coordinate observation"):
        restart.model_configuration_descriptor(model, _exact_guard=fail_on_observation(changed))


def test_coordinate_property_is_not_invoked_by_proof_and_stays_guarded(monkeypatch):
    model, node = model_and_node()
    changed = []
    def coordinate(self):
        changed.append(True)
        return self.__dict__["x"]
    monkeypatch.setattr(Node, "x", property(coordinate), raising=False)
    assert not restart._inert_node_observation(node) and changed == []
    with pytest.raises(RuntimeError, match="coordinate observation"):
        restart.model_configuration_descriptor(model, _exact_guard=fail_on_observation(changed))


def test_restoring_numpy_array_conversion_remains_guarded(monkeypatch):
    model, _ = model_and_node()
    changed = []
    original = np.array
    def array(*args, **kwargs):
        changed.append(True)
        monkeypatch.setattr(np, "array", original)
        return original(*args, **kwargs)
    monkeypatch.setattr(np, "array", array)
    with pytest.raises(RuntimeError, match="coordinate observation"):
        restart.model_configuration_descriptor(model, _exact_guard=fail_on_observation(changed))


def test_effectful_coordinate_conversion_stays_guarded():
    model, node = model_and_node()
    changed = []
    class Coordinate:
        def __float__(self):
            changed.append(True)
            return 0.25
    node.x = Coordinate()
    assert not restart._inert_node_observation(node)
    with pytest.raises(RuntimeError, match="coordinate array observation"):
        restart.model_configuration_descriptor(model, _exact_guard=fail_on_observation(changed))


def test_effectful_dof_conversion_stays_guarded():
    model, node = model_and_node()
    changed = []
    class Dof:
        def __int__(self):
            changed.append(True)
            return 0
    node.dofs = [Dof(), 1, 2]
    assert restart._inert_node_observation(node)
    with pytest.raises(RuntimeError, match="DOF array observation"):
        restart.model_configuration_descriptor(model, _exact_guard=fail_on_observation(changed))


def test_node_subclasses_instance_overrides_and_large_integers_fall_back():
    _, node = model_and_node()
    class CustomNode(Node):
        pass
    assert not restart._inert_node_observation(CustomNode(1, 0.25, 0.5, 0.0))
    node.coords = lambda: np.zeros(3)
    assert not restart._inert_node_observation(node)
    del node.coords
    node.x = 1 << 100
    assert not restart._inert_node_observation(node)


def test_changed_original_function_code_is_not_inert(monkeypatch):
    _, node = model_and_node()
    def replacement(self):
        return np.zeros(3)
    monkeypatch.setattr(Node.coords, "__code__", replacement.__code__)
    assert not restart._inert_node_observation(node)


def test_class_getter_descriptor_is_not_invoked_by_proof(monkeypatch):
    _, node = model_and_node()
    changed = []
    class Descriptor:
        def __get__(self, instance, owner):
            changed.append(True)
            return object.__getattribute__
    monkeypatch.setattr(Node, "__getattribute__", Descriptor())
    assert not restart._inert_node_observation(node)
    assert changed == []


def test_replaced_dof_getter_remains_guarded(monkeypatch):
    model, _ = model_and_node()
    changed = []
    original = getattr
    def getter(value, name, *defaults):
        result = original(value, name, *defaults)
        if name == "dofs":
            changed.append(True)
            monkeypatch.setattr(restart, "getattr", original)
        return result
    monkeypatch.setattr(restart, "getattr", getter, raising=False)
    with pytest.raises(RuntimeError, match="DOF observation"):
        restart.model_configuration_descriptor(model, _exact_guard=fail_on_observation(changed))


def test_dof_property_retains_its_observation_boundary(monkeypatch):
    model, node = model_and_node()
    changed = []
    def dofs(self):
        changed.append(True)
        return self.__dict__["dofs"]
    monkeypatch.setattr(Node, "dofs", property(dofs), raising=False)
    assert not restart._inert_node_observation(node) and changed == []
    with pytest.raises(RuntimeError, match="DOF observation"):
        restart.model_configuration_descriptor(model, _exact_guard=fail_on_observation(changed))


def test_changed_coordinate_numpy_global_falls_back(monkeypatch):
    from types import SimpleNamespace
    _, node = model_and_node()
    monkeypatch.setitem(Node.coords.__globals__, "np", SimpleNamespace(array=np.array))
    assert not restart._inert_node_observation(node)


def test_custom_node_metaclass_metadata_is_not_observed():
    changed = []
    class Meta(type):
        @property
        def __mro__(cls):
            changed.append(True)
            return ()
    class CustomNode(Node, metaclass=Meta):
        pass
    assert not restart._inert_node_observation(CustomNode(1, 0.25, 0.5, 0.0))
    assert changed == []
