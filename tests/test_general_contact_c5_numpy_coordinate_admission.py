"""Original C5 builder inputs must reach the guarded direct elastic caller."""
import copy
import inspect
import sys
from contextlib import contextmanager

import numpy as np
import pytest

from anyfesolver import e4_pl_element as qualified
from anyfesolver.general_contact.core import suspend_contact
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from scripts.general_contact_d1_free_target_press_probe import build_free_target_case


def production_case():
    model, _definition, _settings, _program = build_free_target_case(
        4, steps_per_segment=2, four_phase=True,
        enforcement_solver="monolithic_active_set")
    model.apply_boundary_conditions()
    elements = [element for element in model.mesh.elements.values()
                if type(element) is qualified.QualifiedE4PLShellElement]
    assert len(elements) == 4
    return model, elements, model.get_material("elastic")


def coordinate_records(model, elements):
    return [(node_id, tuple(vars(model.mesh.nodes[node_id])[axis]
                            for axis in ("x", "y", "z")))
            for node_id in sorted({n for element in elements for n in element.node_ids})]


def unchanged_coordinates(model, before):
    for node_id, values in before:
        actual = vars(model.mesh.nodes[node_id])
        assert all(actual[axis] is value for axis, value in zip(("x", "y", "z"), values))


@contextmanager
def observe_direct_returns():
    """Observe the actual body; never replace a guarded method or evaluator."""
    method = inspect.getclosurevars(
        qualified.QualifiedE4PLShellElement._elastic_increment_response).nonlocals["method"]
    code = method.__code__
    rows = []
    previous = sys.getprofile()
    assert previous is None
    def observe(frame, event, value):
        if frame.f_code is code and event == "return":
            rows.append((id(frame.f_locals["self"]), value is not None,
                         frame.f_locals["batch_baseline"]))
    sys.setprofile(observe)
    try:
        yield rows
    finally:
        sys.setprofile(previous)


@contextmanager
def observe_single_law_and_direct_returns():
    """Observe the single-law kernel and the retired direct body (no replacement)."""
    from anyfesolver import _q4_assumed_strain_plasticity as assumed

    kernel_code = assumed.kernel_batch.__code__
    direct_code = inspect.getclosurevars(
        qualified.QualifiedE4PLShellElement._elastic_increment_response
    ).nonlocals["method"].__code__
    direct_rows, kernel_rows = [], []
    previous = sys.getprofile()
    assert previous is None

    def observe(frame, event, value):
        if event != "return":
            return
        if frame.f_code is kernel_code:
            kernel_rows.append(int(np.asarray(frame.f_locals["u_elem_batch"]).shape[0]))
        elif frame.f_code is direct_code:
            direct_rows.append(id(frame.f_locals["self"]))
    sys.setprofile(observe)
    try:
        yield direct_rows, kernel_rows
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("route", ["scalar", "batch"], ids=["scalar", "batch"])
def test_original_builder_cold_direct_activation(route):
    # Retargeted 2026-10-23 (single elastic law): the original C5 builder inputs
    # now reach the single-law kernel (one batched call for all four elements,
    # one scalar call for one element); the retired direct elastic route must
    # not run for qualified Q4.
    model, elements, material = production_case()
    before = coordinate_records(model, elements)
    assert all(type(values[1]) is np.float64 and type(values[2]) is np.float64
               for _, values in before)
    assert all(vars(e)["_qualified_components"] is None for e in elements)
    displacement = np.linspace(-1e-5, 1e-5, model.mesh.dof_manager.total_dofs)
    with observe_single_law_and_direct_returns() as (rows, kernel_rows):
        if route == "scalar":
            element = elements[0]
            mapping = element.get_dof_mapping(model.mesh)
            force, tangent, _state = element.compute_nonlinear_response(
                model.mesh, material, displacement[mapping], None, 5, True)
        else:
            with suspend_contact():
                force, tangent, _states = _assemble_nonlinear_system(
                    model, displacement, {}, 5, tangent=True,
                    kinematics="von_karman", require_full_coordinates=True)
    assert rows == []
    assert kernel_rows == ([1] if route == "scalar" else [len(elements)])
    assert np.all(np.isfinite(force))
    assert tangent.shape[0] == force.size
    unchanged_coordinates(model, before)


def test_raw_numpy_signed_zero_and_snapshot_identity_are_preserved():
    model, elements, material = production_case()
    element = elements[0]
    node = model.mesh.nodes[element.node_ids[0]]
    raw_zero = np.float64(-0.0)
    node.x = raw_zero
    before = coordinate_records(model, elements)
    snapshot = qualified._capture_q4_fast_input_snapshot(element, model.mesh, material)
    assert snapshot is not None
    assert qualified._q4_fast_input_snapshot_matches(element, model.mesh, material, snapshot)
    assert element._defining_class_input_snapshot_matches(model.mesh, material, snapshot)
    assert node.x is raw_zero and np.signbit(node.x)
    record = next(row for row in snapshot.nodes_snapshot if row[0] == element.node_ids[0])
    assert record[4][0] is raw_zero and record[3][0] is np.float64
    unchanged_coordinates(model, before)


@pytest.mark.parametrize("change", ["replacement", "type", "value", "revision"],
                         ids=["replacement", "type", "value", "revision"])
def test_numpy_coordinate_snapshot_detects_changes(change):
    model, elements, material = production_case()
    element = elements[0]
    node = model.mesh.nodes[element.node_ids[0]]
    snapshot = qualified._capture_q4_fast_input_snapshot(element, model.mesh, material)
    assert snapshot is not None
    old = node.y
    if change == "replacement":
        replacement = np.array([old], dtype=np.float64)[0]
        assert replacement is not old
        node.y = replacement
    elif change == "type":
        node.y = float(old)
    elif change == "value":
        node.y = old + np.float64(.01)
    else:
        object.__setattr__(node, "_coordinate_revision", node._coordinate_revision + 1)
    assert not qualified._q4_fast_input_snapshot_matches(element, model.mesh, material, snapshot)
    assert not element._defining_class_input_snapshot_matches(model.mesh, material, snapshot)


@pytest.mark.parametrize("kind", ["bool", "custom", "float-subclass", "numpy-subclass", "float32"],
                         ids=["bool", "custom", "float-subclass", "numpy-subclass", "float32"])
def test_nonexact_coordinate_types_remain_unadmitted(kind):
    model, elements, material = production_case()
    element = elements[0]
    observed = []
    class Custom:
        def __float__(self):
            observed.append("conversion")
            raise AssertionError("custom conversion must not run")
    class PythonSubclass(float):
        pass
    class NumpySubclass(np.float64):
        pass
    value = {"bool": True, "custom": Custom(), "float-subclass": PythonSubclass(.1),
             "numpy-subclass": NumpySubclass(.1), "float32": np.float32(.1)}[kind]
    model.mesh.nodes[element.node_ids[0]].y = value
    assert qualified._capture_q4_fast_input_snapshot(element, model.mesh, material) is None
    assert observed == []


def test_numpy_node_mutation_during_response_cannot_publish():
    model, elements, material = production_case()
    element = elements[0]
    node = model.mesh.nodes[element.node_ids[0]]
    state = element.init_nonlinear_state(5)
    saved = copy.deepcopy(state)
    class MutatingVector:
        def __array__(self, dtype=None, copy=None):
            node.y = node.y + np.float64(.001)
            return np.zeros(24, dtype=dtype)
    with pytest.raises((RuntimeError, ValueError), match="changed|incompatible"):
        element._elastic_increment_response(
            model.mesh, material, MutatingVector(), state, True, batch_baseline=True)
    for key in saved:
        np.testing.assert_array_equal(state[key], saved[key])


def test_numpy_coordinate_cancelled_observation_keeps_state():
    model, elements, material = production_case()
    element = elements[0]
    before = coordinate_records(model, elements)
    state = element.init_nonlinear_state(5)
    saved = copy.deepcopy(state)
    class Cancelled(BaseException):
        pass
    class CancelledVector:
        def __array__(self, dtype=None, copy=None):
            raise Cancelled()
    with pytest.raises(Cancelled):
        element._elastic_increment_response(
            model.mesh, material, CancelledVector(), state, True, batch_baseline=True)
    for key in saved:
        np.testing.assert_array_equal(state[key], saved[key])
    unchanged_coordinates(model, before)


@pytest.mark.parametrize("binding", ["numpy", "captured"], ids=["numpy", "captured"])
def test_scalar_type_authority_rebinding_refuses_public_response(monkeypatch, binding):
    model, elements, material = production_case()
    element = elements[0]
    vector = np.zeros(24)
    module, name = ((np, "float64") if binding == "numpy" else
                    (qualified, "_Q4_FAST_COORDINATE_FLOAT64_TYPE"))
    with monkeypatch.context() as patch:
        patch.setattr(module, name, float)
        with pytest.raises((RuntimeError, ValueError, TypeError)):
            element.compute_nonlinear_response(model.mesh, material, vector)


def test_defining_class_matcher_refuses_subclass_without_selectable_class():
    model, elements, material = production_case()
    element = elements[0]
    owner = type(element)
    snapshot = qualified._capture_q4_fast_input_snapshot(element, model.mesh, material)
    assert snapshot is not None
    assert list(inspect.signature(owner._defining_class_input_snapshot_matches).parameters) == [
        "self", "mesh", "material", "snapshot"]
    class Derived(owner):
        pass
    foreign = object.__new__(Derived)
    with pytest.raises(ValueError, match="defining class identity"):
        owner._defining_class_input_snapshot_matches(foreign, model.mesh, material, snapshot)
    assert not qualified._q4_fast_input_snapshot_matches(foreign, model.mesh, material, snapshot)


@pytest.mark.parametrize("target", ["global_class", "class_helper"])
def test_defining_class_admission_authority_rebinding_refuses(monkeypatch, target):
    model, elements, material = production_case()
    element = elements[0]
    assert element._elastic_increment_admitted(model.mesh, material, None)
    with monkeypatch.context() as patch:
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            if target == "global_class":
                patch.setattr(qualified, "QualifiedE4PLShellElement", object)
            else:
                patch.setattr(type(element), "_defining_class_input_snapshot_matches", lambda *args: True)
            element._elastic_increment_admitted(model.mesh, material, None)


def test_defining_class_cell_change_refuses_and_restores():
    model, elements, material = production_case()
    element = elements[0]
    snapshot = qualified._capture_q4_fast_input_snapshot(element, model.mesh, material)
    raw = type(element)._defining_class_input_snapshot_matches
    cell = dict(zip(raw.__code__.co_freevars, raw.__closure__ or ()))['__class__']
    original = cell.cell_contents
    try:
        cell.cell_contents = object
        with pytest.raises(ValueError, match="defining class identity"):
            raw(element, model.mesh, material, snapshot)
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            element._elastic_increment_admitted(model.mesh, material, None)
    finally:
        cell.cell_contents = original
    assert raw(element, model.mesh, material, snapshot)

# Single elastic law (owner decision 2026-10-23): the production route for a
# planar qualified Q4 is the assumed-strain kernel; the two lifecycle checks
# above (which exercise the retained, guarded direct helper) are repeated on the
# public production route, which must fail closed in the same way.
def test_numpy_node_mutation_during_single_law_response_cannot_publish():
    model, elements, material = production_case()
    element = elements[0]
    node = model.mesh.nodes[element.node_ids[0]]
    state = element.init_nonlinear_state(5)
    saved = copy.deepcopy(state)
    element.compute_nonlinear_response(model.mesh, material, np.zeros(24), state, 5, True)
    class MutatingVector:
        def __array__(self, dtype=None, copy=None):
            node.y = node.y + np.float64(.001)
            return np.zeros(24, dtype=dtype)
    with pytest.raises((RuntimeError, ValueError), match="changed|incompatible|authority"):
        element.compute_nonlinear_response(model.mesh, material, MutatingVector(), state, 5, True)
    for key in saved:
        np.testing.assert_array_equal(state[key], saved[key])


def test_numpy_coordinate_cancelled_single_law_observation_keeps_state():
    model, elements, material = production_case()
    element = elements[0]
    before = coordinate_records(model, elements)
    state = element.init_nonlinear_state(5)
    saved = copy.deepcopy(state)
    class Cancelled(BaseException):
        pass
    class CancelledVector:
        def __array__(self, dtype=None, copy=None):
            raise Cancelled()
    with pytest.raises(Cancelled):
        element.compute_nonlinear_response(model.mesh, material, CancelledVector(), state, 5, True)
    for key in saved:
        np.testing.assert_array_equal(state[key], saved[key])
    unchanged_coordinates(model, before)
