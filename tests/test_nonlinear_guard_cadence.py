"""Nonlinear guard cadence (owner decisions G2, L1 and G1, 2026-11-24).

G2: the qualified Q4 class/namespace authority (dependency class namespaces,
class identity metadata, class mutable mappings, authority-array metadata) is
validated once per Newton evaluation or lease scan window -- completely on
first use and again when the window closes -- instead of on every
element-level guarded call.  Per-element instance and quadrature checks still
run on every call.

L1: inside one force-control step attempt, solver-internal complete lifecycle
scans run at step entry and before the step commits instead of at every
evaluation.  Caller-controlled observations (status/progress callbacks,
untrusted cancellation tokens) still run the complete scan immediately.

G1: the remaining identity sweeps run as exact C-speed iterator forms with the
same order, messages and outcomes.

Trust boundary: NumPy/SciPy/builtins/ANYmaterial internals and ANYfeSolver's
own classes and guard machinery are not replaced at run time, and concurrent
mutation during a request is out of contract.  Within that contract every
persistent change below is refused with the same exception type as before and
before any result of the affected evaluation or step is accepted.  What
changes (documented by the tests marked "cadence change"): a class change made
and reverted inside one evaluation window, or a lifecycle-API change made and
reverted between solver-internal points of one step, is no longer detected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import FEModel, LoadCase, solve_static_nonlinear
from anyfesolver import nonlinear_static as _nonlinear_static
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import create_shell_element
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.fe_core import FEMesh
from anyfesolver.matrix_assembly import AssemblyError
from anyfesolver import e4_pl_element as _q4


# ----------------------------------------------------------------- fixtures


def _fix_edges(model: FEModel, ids: dict, n: int) -> None:
    for (i, j), node in ids.items():
        if i in (0, n) or j in (0, n):
            model.add_boundary_condition(
                BoundaryCondition(
                    f"bc{node}",
                    [node],
                    {dof: 0.0 for dof in ("ux", "uy", "uz", "rx", "ry", "rz")},
                )
            )


def _plate(n: int = 3, *, triangles: bool = False, plastic: bool = False):
    model = FEModel("guard-cadence")
    model.add_material("mat", 2.1e11, 0.3, density=7850.0)
    if plastic:
        model.materials["mat"].yield_stress = 3.0e7
    ids = {}
    for i in range(n + 1):
        for j in range(n + 1):
            ids[i, j] = len(ids) + 1
            model.add_node(ids[i, j], i / n, j / n, 0.0)
    element_id = 0
    element_ids = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
            if triangles:
                for nodes in ((a, b, c), (a, c, d)):
                    element_id += 1
                    model.add_element(
                        element_id,
                        create_shell_element(
                            element_id, list(nodes), "mat", thickness=0.01,
                            reference_normal=(0.0, 0.0, 1.0),
                        ),
                    )
                    element_ids.append(element_id)
            else:
                element_id += 1
                model.add_element(
                    element_id,
                    create_shell_element(element_id, [a, b, c, d], "mat", thickness=0.01),
                )
                element_ids.append(element_id)
    _fix_edges(model, ids, n)
    load = LoadCase("p")
    if triangles:
        # A point load keeps the V2D model on the solver-internal complete
        # lifecycle path (no constant-time trusted guard is available).
        load.add_nodal_load(ids[n // 2, n // 2], [0.0, 0.0, -2.0e4, 0.0, 0.0, 0.0])
    else:
        for element in element_ids:
            load.add_pressure_load(int(element), 3.0e4 if plastic else 2.0e5)
    return model, load


def _solve(model, load, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_static_nonlinear(model, load, num_steps=2, **kwargs)


def _q4_boundary_closures():
    """Return the bound Q4 boundary guard's runtime/instance closures."""

    holder = dict(
        zip(
            _q4._require_exact_q4_runtime_authority.__code__.co_freevars,
            (cell.cell_contents for cell in _q4._require_exact_q4_runtime_authority.__closure__),
        )
    )
    boundary_guard = holder["authority"][0]
    cells = dict(
        zip(
            boundary_guard.__code__.co_freevars,
            (cell.cell_contents for cell in boundary_guard.__closure__),
        )
    )
    runtime_guard = cells["runtime_guard"]
    runtime_cells = dict(
        zip(
            runtime_guard.__code__.co_freevars,
            (cell.cell_contents for cell in runtime_guard.__closure__),
        )
    )
    return cells["instance_guard"], runtime_cells["formulation_class_guard"]


def _closure_cells(function):
    return dict(
        zip(
            function.__code__.co_freevars,
            (cell.cell_contents for cell in function.__closure__),
        )
    )


class _ClassProbe:
    """Add one persistent entry to a dependency class namespace, then remove it."""

    def __init__(self, owner: type, name: str = "_guard_cadence_probe") -> None:
        self.owner = owner
        self.name = name
        self.active = False

    def set(self) -> None:
        setattr(self.owner, self.name, 1)
        self.active = True

    def clear(self) -> None:
        if self.name in type.__getattribute__(self.owner, "__dict__"):
            delattr(self.owner, self.name)
        self.active = False


@pytest.fixture
def femesh_probe():
    probe = _ClassProbe(FEMesh)
    try:
        yield probe
    finally:
        probe.clear()


@pytest.fixture
def v2d_probe():
    probe = _ClassProbe(NativeParityE4PLS3V2DShellElement)
    try:
        yield probe
    finally:
        probe.clear()


# --------------------------------------------- G2: detection is preserved


def test_g2_results_are_unchanged_on_an_unchanged_model() -> None:
    model, load = _plate()
    result = _solve(model, load)
    assert result.status == "completed"
    assert len(result.steps) >= 2


def test_g2_dependency_class_change_before_solve_is_refused(femesh_probe) -> None:
    model, load = _plate()
    femesh_probe.set()
    with pytest.raises(AssemblyError, match="found incompatible qualified shell authority"):
        _solve(model, load)


def test_g2_dependency_class_change_between_steps_is_refused(femesh_probe) -> None:
    model, load = _plate()
    committed = []

    def progress(event):
        committed.append(event)
        if len(committed) == 1:
            femesh_probe.set()

    with pytest.raises(AssemblyError, match="found incompatible qualified shell authority"):
        _solve(model, load, progress_callback=progress)
    # The change made after step 1 is refused before step 2 commits.
    assert len(committed) == 1


def test_g2_dependency_class_change_by_status_callback_is_refused(femesh_probe) -> None:
    model, load = _plate()
    committed = []

    def status(_message):
        femesh_probe.set()

    with pytest.raises(AssemblyError, match="found incompatible qualified shell authority"):
        _solve(model, load, status_callback=status, progress_callback=committed.append)
    assert committed == []


def test_g2_q4_class_namespace_entry_is_refused() -> None:
    model, load = _plate()
    cls = _q4.QualifiedE4PLShellElement
    try:
        setattr(cls, "_guard_cadence_probe", 1)
        with pytest.raises(ElementCapabilityError):
            _solve(model, load)
    finally:
        if "_guard_cadence_probe" in type.__getattribute__(cls, "__dict__"):
            delattr(cls, "_guard_cadence_probe")


def _authority_array_with_bytes_base():
    _instance_guard, class_guard = _q4_boundary_closures()
    metadata = _closure_cells(class_guard)["authority_array_metadata"]
    for array, _dtype, shape, _strides, _contiguous in metadata:
        if len(shape) == 2 and shape[0] > 1 and shape[1] > 1:
            return array, shape
    pytest.skip("no two-dimensional authority array is captured")


def test_g2_authority_array_metadata_change_is_refused() -> None:
    model, load = _plate()
    array, shape = _authority_array_with_bytes_base()
    try:
        array.shape = (shape[0] * shape[1],)
        with pytest.raises((ElementCapabilityError, ValueError)) as caught:
            _solve(model, load)
        assert "metadata" in str(caught.value) or "incompatible" in str(caught.value)
    finally:
        array.shape = shape


def test_g2_window_rejects_a_persistent_in_window_class_change(femesh_probe) -> None:
    """A persistent change made inside one window is refused when it closes."""

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    with pytest.raises(ValueError, match="qualified Q4 FEMesh class namespace changed"):
        with _q4._q4_class_authority_window():
            _q4._require_exact_q4_runtime_authority(element, context="probe")
            femesh_probe.set()
            _q4._require_exact_q4_runtime_authority(element, context="probe")


def test_g2_cadence_change_reverted_in_window_class_change_is_not_detected(
    femesh_probe,
) -> None:
    """Cadence change: a class change made and reverted inside one window.

    Outside a window the same sequence is refused on the next guarded call.
    """

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    with _q4._q4_class_authority_window():
        _q4._require_exact_q4_runtime_authority(element, context="probe")
        femesh_probe.set()
        _q4._require_exact_q4_runtime_authority(element, context="probe")
        femesh_probe.clear()
    femesh_probe.set()
    with pytest.raises(ValueError, match="qualified Q4 FEMesh class namespace changed"):
        _q4._require_exact_q4_runtime_authority(element, context="probe")


def test_g2_window_token_does_not_outlive_its_window(femesh_probe) -> None:
    import contextvars

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    with _q4._q4_class_authority_window():
        _q4._require_exact_q4_runtime_authority(element, context="probe")
        copied = contextvars.copy_context()
    femesh_probe.set()
    with pytest.raises(ValueError, match="class namespace changed"):
        copied.run(
            _q4._require_exact_q4_runtime_authority, element, context="probe"
        )


# ------------------------------- G2: per-element instance checks unchanged


def _first_q4(model):
    return next(iter(model.mesh.elements.values()))


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("shadow", "class authority has instance shadows"),
        ("callable", "callable authority override"),
        ("quadrature", "gauss_points"),
        ("topology", "fixed topology authority"),
    ),
)
def test_g2_instance_checks_still_run_on_every_call(mutation, expected) -> None:
    model, load = _plate()
    element = _first_q4(model)
    _solve(model, load)
    namespace = object.__getattribute__(element, "__dict__")
    if mutation == "shadow":
        namespace["formulation_id"] = "shadow"
    elif mutation == "callable":
        namespace["_guard_cadence_callable"] = len
    elif mutation == "quadrature":
        namespace["gauss_points"] = np.zeros((4, 2))
    else:
        namespace["_is_4node"] = False
    with _q4._q4_class_authority_window():
        with pytest.raises(ValueError, match=expected):
            _q4._require_exact_q4_runtime_authority(element, context="probe")
    with pytest.raises(ValueError, match=expected):
        _q4._require_exact_q4_runtime_authority(element, context="probe")


def test_g2_instance_shadow_refuses_the_solve() -> None:
    model, load = _plate()
    object.__getattribute__(_first_q4(model), "__dict__")["formulation_id"] = "x"
    with pytest.raises((ElementCapabilityError, ValueError)):
        _solve(model, load)


# ------------------------------------------------------- L1: step cadence


def test_l1_results_and_lifecycle_detection_on_v2d_model(v2d_probe) -> None:
    model, load = _plate(3, triangles=True)
    result = _solve(model, load)
    assert result.status == "completed"
    model, load = _plate(3, triangles=True)
    v2d_probe.set()
    with pytest.raises(ElementCapabilityError):
        _solve(model, load)


def test_l1_lifecycle_change_by_status_callback_is_refused_before_commit(
    v2d_probe,
) -> None:
    model, load = _plate(3, triangles=True)
    committed = []

    def status(_message):
        v2d_probe.set()

    with pytest.raises(ElementCapabilityError):
        _solve(model, load, status_callback=status, progress_callback=committed.append)
    assert committed == []


def test_l1_lifecycle_change_between_steps_is_refused(v2d_probe) -> None:
    model, load = _plate(3, triangles=True)
    committed = []

    def progress(event):
        committed.append(event)
        v2d_probe.set()

    with pytest.raises(ElementCapabilityError):
        _solve(model, load, progress_callback=progress)
    assert len(committed) == 1


def _replace_v2d_lifecycle_api():
    cls = NativeParityE4PLS3V2DShellElement
    original = type.__getattribute__(cls, "__dict__")["compute_nonlinear_response"]

    def replacement(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    setattr(cls, "compute_nonlinear_response", replacement)
    return lambda: setattr(cls, "compute_nonlinear_response", original)


@pytest.mark.parametrize("when", ("before", "status_callback", "between_steps"))
def test_l1_lifecycle_api_replacement_is_refused(when) -> None:
    model, load = _plate(3, triangles=True)
    committed = []
    restore = []

    def replace_once(*_args):
        if not restore:
            restore.append(_replace_v2d_lifecycle_api())

    def progress(event):
        committed.append(event)
        if when == "between_steps":
            replace_once()

    kwargs = {"progress_callback": progress}
    if when == "status_callback":
        kwargs["status_callback"] = replace_once
    try:
        if when == "before":
            replace_once()
        with pytest.raises(ElementCapabilityError, match="lifecycle APIs"):
            _solve(model, load, **kwargs)
    finally:
        for undo in restore:
            undo()
    assert len(committed) == (1 if when == "between_steps" else 0)


def _hook_internal_events(monkeypatch, on_event):
    original = _nonlinear_static.record_nonlinear_solver_event

    def hook(event, **kwargs):
        original(event, **kwargs)
        on_event(event)

    monkeypatch.setattr(_nonlinear_static, "record_nonlinear_solver_event", hook)


def test_l1_persistent_internal_point_change_is_refused_before_commit(
    monkeypatch, v2d_probe
) -> None:
    """Out-of-contract stand-in: a change made at a solver-internal point.

    The status callback arms the hook; the next solver-internal guard event
    makes a persistent change.  It is refused (complete lifecycle scan) before
    the step commits.
    """

    model, load = _plate(3, triangles=True)
    armed = []
    committed = []

    def on_event(event):
        if armed and not v2d_probe.active and event.startswith("qualified_guard_"):
            v2d_probe.set()

    _hook_internal_events(monkeypatch, on_event)
    with pytest.raises((ElementCapabilityError, ValueError)):
        _solve(
            model,
            load,
            status_callback=lambda _message: armed.append(True),
            progress_callback=committed.append,
        )
    assert committed == []


def test_l1_cadence_change_reverted_internal_change_is_not_detected(
    monkeypatch, v2d_probe
) -> None:
    """Cadence change: a lifecycle change made and reverted between two
    solver-internal points of one step is no longer detected (L1)."""

    model, load = _plate(3, triangles=True)
    deferred = []

    def on_event(event):
        if event != "qualified_guard_step_deferred":
            return
        deferred.append(event)
        if len(deferred) == 1:
            v2d_probe.set()
        elif len(deferred) == 2:
            v2d_probe.clear()

    _hook_internal_events(monkeypatch, on_event)
    result = _solve(model, load)
    assert result.status == "completed"
    assert len(deferred) >= 2


def test_l1_q4_fast_path_is_unchanged() -> None:
    """Exact all-Q4 models keep their constant-time internal guard."""

    model, load = _plate()
    events = []
    original = _nonlinear_static.record_nonlinear_solver_event

    def hook(event, **kwargs):
        events.append(event)
        original(event, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_nonlinear_static, "record_nonlinear_solver_event", hook)
        result = _solve(model, load)
    assert result.status == "completed"
    assert "qualified_guard_step_deferred" not in events
    assert "qualified_guard_trusted" in events


# ------------------------------------------------- G1: exact differential


def _old_entries_changed(namespace, expected_items):
    # Verbatim pre-G1 sweep.
    return any(
        name not in namespace or namespace[name] is not expected
        for name, expected in expected_items
    )


def _old_all_exact_str(values):
    return all(type(name) is str for name in values)


class _Counting:
    calls = 0

    def __eq__(self, other):
        type(self).calls += 1
        return NotImplemented

    def __hash__(self):
        type(self).calls += 1
        return 7

    def __bool__(self):
        type(self).calls += 1
        return True


def _sweep_cases():
    shared = object()
    base = {"a": 1, "b": shared, "c": None}
    yield dict(base), tuple(base.items())
    yield {"a": 1, "b": shared}, tuple(base.items())
    yield {"a": 1, "b": object(), "c": None}, tuple(base.items())
    yield {"a": 1, "b": shared, "c": None, "d": 2}, tuple(base.items())
    yield {}, ()
    yield {"a": _Counting()}, (("a", _Counting()),)
    yield {"x": 1}, (("a", 1),)
    yield type.__getattribute__(FEMesh, "__dict__"), tuple(
        type.__getattribute__(FEMesh, "__dict__").items()
    )


@pytest.mark.parametrize("case", list(range(8)))
def test_g1_identity_sweep_matches_the_verbatim_old_sweep(case) -> None:
    namespace, items = list(_sweep_cases())[case]
    names = tuple(name for name, _ in items)
    values = tuple(value for _, value in items)
    _Counting.calls = 0
    old = _old_entries_changed(namespace, items)
    old_calls = _Counting.calls
    _Counting.calls = 0
    new = _q4._q4_identity_entries_changed(namespace, names, values)
    assert new is old
    assert _Counting.calls <= old_calls


@pytest.mark.parametrize(
    "values",
    (
        {"a": 1, "b": 2},
        {},
        {1: 2},
        {"a": 1, b"b": 2},
        ("a", "b"),
        (1, 2, 3, 4),
        (1, True, 3, 4),
        (1, 2, 3.0, 4),
    ),
)
def test_g1_exact_type_sweep_matches_the_verbatim_old_sweep(values) -> None:
    for expected_type in (str, int):
        old = all(type(value) is expected_type for value in values)
        assert _q4._q4_all_exact_type(values, expected_type) is old


def _verbatim_old_formulation_class_guard(cells):
    """The pre-G1 formulation class guard rebuilt from the bound closure data."""

    dependency_namespaces = tuple(
        (owner, dict(zip(names, values)))
        for owner, names, values, _count, _name in cells["namespace_sweeps"]
    )
    dependency_class_metadata = cells["dependency_class_metadata"]
    class_mutable_mappings = tuple(
        (label, mapping, tuple(zip(names, values)))
        for label, mapping, names, values, _count in cells["mapping_sweeps"]
    )
    ignored = cells["ignored_dependency_class_cache_name"]
    metadata_check = _q4._require_authority_array_metadata
    authority_array_metadata = cells["authority_array_metadata"]

    def formulation_class_guard() -> None:
        for (owner, expected_namespace), metadata in zip(
            dependency_namespaces, dependency_class_metadata
        ):
            actual_namespace = type.__getattribute__(owner, "__dict__")
            if (
                len(actual_namespace) - int(ignored in actual_namespace)
                != len(expected_namespace)
                or any(
                    name not in actual_namespace
                    or actual_namespace[name] is not expected
                    for name, expected in expected_namespace.items()
                )
            ):
                raise ValueError(f"qualified Q4 {metadata[2]} class namespace changed")
        for (
            owner,
            expected_type,
            expected_name,
            _expected_qualname,
            _expected_module,
            expected_bases,
        ) in dependency_class_metadata:
            actual_name = type.__getattribute__(owner, "__name__")
            actual_bases = type.__getattribute__(owner, "__bases__")
            if (
                type(owner) is not expected_type
                or type(actual_name) is not str
                or actual_name != expected_name
                or type(actual_bases) is not tuple
                or len(actual_bases) != len(expected_bases)
                or any(
                    actual_base is not expected_base
                    for actual_base, expected_base in zip(actual_bases, expected_bases)
                )
            ):
                raise ValueError(f"qualified Q4 {expected_name} class identity changed")
        for label, mapping, expected_items in class_mutable_mappings:
            if (
                type(mapping) is not dict
                or not all(type(key) is str for key in mapping)
                or len(mapping) != len(expected_items)
                or any(
                    name not in mapping or mapping[name] is not expected
                    for name, expected in expected_items
                )
            ):
                raise ValueError(f"qualified Q4 {label} mutable class authority changed")
        metadata_check(authority_array_metadata, label="qualified Q4 runtime authority")

    return formulation_class_guard


def _outcome(function, *args):
    try:
        function(*args)
    except Exception as exc:  # noqa: BLE001 - differential outcome capture
        return type(exc), str(exc)
    return None


def _class_guard_mutations():
    cells = _closure_cells(_q4_boundary_closures()[1])
    owners = [owner for owner, *_rest in cells["namespace_sweeps"]]
    mappings = [mapping for _label, mapping, *_rest in cells["mapping_sweeps"]]

    def none():
        return lambda: None

    def add(owner):
        type.__setattr__(owner, "_g1_probe", 1)
        return lambda: type.__delattr__(owner, "_g1_probe")

    def slotnames(owner):
        if "__slotnames__" in type.__getattribute__(owner, "__dict__"):
            return lambda: None
        type.__setattr__(owner, "__slotnames__", [])
        return lambda: type.__delattr__(owner, "__slotnames__")

    def replace(owner):
        name, value = next(
            (key, val)
            for key, val in type.__getattribute__(owner, "__dict__").items()
            if not key.startswith("__")
        )
        type.__setattr__(owner, name, object())
        return lambda: type.__setattr__(owner, name, value)

    def mapping_entry(mapping):
        key = next(iter(mapping), None)
        if key is None:
            mapping["_g1_probe"] = 1
            return lambda: mapping.pop("_g1_probe")
        value = mapping[key]
        mapping[key] = object()
        return lambda: mapping.__setitem__(key, value)

    def mapping_key(mapping):
        mapping[1] = 1
        return lambda: mapping.pop(1)

    yield "unchanged", none
    for owner in owners:
        yield f"add {owner.__name__}", lambda owner=owner: add(owner)
        yield f"slotnames {owner.__name__}", lambda owner=owner: slotnames(owner)
        yield f"replace {owner.__name__}", lambda owner=owner: replace(owner)
    for index, mapping in enumerate(mappings):
        yield f"mapping entry {index}", lambda mapping=mapping: mapping_entry(mapping)
        yield f"mapping key {index}", lambda mapping=mapping: mapping_key(mapping)


def test_g1_formulation_class_guard_matches_the_verbatim_old_guard() -> None:
    new_guard = _q4_boundary_closures()[1]
    old_guard = _verbatim_old_formulation_class_guard(_closure_cells(new_guard))
    seen = []
    for label, mutate in _class_guard_mutations():
        restore = mutate()
        try:
            old = _outcome(old_guard)
            new = _outcome(new_guard)
        finally:
            restore()
        assert new == old, label
        seen.append((label, old is None))
    assert ("unchanged", True) in seen
    assert any(not passed for _label, passed in seen)
    assert _outcome(new_guard) is None


def _verbatim_old_instance_guard(cells):
    expected_class = cells["expected_class"]
    class_data_names = cells["class_data_names"]
    expected_formulation_id = cells["expected_formulation_id"]
    static_mro_attribute = _q4._static_mro_attribute

    def instance_guard(self) -> None:
        if type(self) is not expected_class:
            raise ValueError("qualified Q4 requires the exact element class")
        namespace = object.__getattribute__(self, "__dict__")
        if type(namespace) is not dict:
            raise ValueError("qualified Q4 instance namespace is incompatible")
        if not all(type(name) is str for name in namespace):
            raise ValueError("qualified Q4 instance keys must be exact strings")
        shadowed = class_data_names.intersection(namespace)
        if shadowed:
            raise ValueError(
                "qualified Q4 class authority has instance shadows: "
                + ", ".join(sorted(shadowed))
            )
        if any(callable(value) for value in namespace.values()):
            raise ValueError(
                "qualified Q4 instance contains a callable authority override"
            )
        element_id = namespace.get("element_id")
        node_ids = namespace.get("node_ids")
        material_name = namespace.get("material_name")
        if (
            type(element_id) is not int
            or type(node_ids) is not tuple
            or len(node_ids) != 4
            or not all(type(node_id) is int for node_id in node_ids)
            or len(set(node_ids)) != 4
            or type(material_name) is not str
            or static_mro_attribute(type(self), "element_id") is not None
            or static_mro_attribute(type(self), "node_ids") is not None
            or static_mro_attribute(type(self), "material_name") is not None
            or static_mro_attribute(type(self), "formulation_id")
            != expected_formulation_id
        ):
            raise ValueError(
                "qualified Q4 instance connectivity/material authority is incompatible"
            )
        expected_topology = {
            "reduced_integration": False,
            "_is_3node": False,
            "_is_4node": True,
            "_is_6node": False,
            "_is_8node": False,
            "_is_triangular": False,
            "_is_quadrilateral": True,
        }
        if any(
            dict.get(namespace, name) is not expected
            for name, expected in expected_topology.items()
        ):
            raise ValueError("qualified Q4 fixed topology authority is incompatible")
        for name in ("reference_normal", "material_direction"):
            vector = dict.get(namespace, name)
            if vector is None:
                continue
            if (
                type(vector) is not np.ndarray
                or vector.dtype != np.dtype(np.float64)
                or vector.shape != (3,)
                or vector.strides != (8,)
                or not vector.flags.c_contiguous
                or vector.flags.writeable
            ):
                raise ValueError(f"qualified Q4 {name} vector authority is incompatible")
            current = vector
            while type(current) is np.ndarray:
                if current.flags.writeable:
                    raise ValueError(f"qualified Q4 {name} vector base is writeable")
                current = current.base
            if not (
                type(current) is bytes
                or isinstance(current, memoryview) and current.readonly
            ):
                raise ValueError(f"qualified Q4 {name} vector base is incompatible")

    return instance_guard


_INSTANCE_MUTATIONS = (
    ("unchanged", lambda ns: None),
    ("non-str key", lambda ns: ns.__setitem__(1, 1)),
    ("bytes key", lambda ns: ns.__setitem__(b"k", 1)),
    ("shadow", lambda ns: ns.__setitem__("formulation_id", "x")),
    ("callable", lambda ns: ns.__setitem__("_g1_callable", len)),
    ("counting value", lambda ns: ns.__setitem__("_g1_counting", _Counting())),
    ("element id float", lambda ns: ns.__setitem__("element_id", 1.0)),
    ("node ids list", lambda ns: ns.__setitem__("node_ids", list(ns["node_ids"]))),
    ("node ids short", lambda ns: ns.__setitem__("node_ids", tuple(ns["node_ids"])[:3])),
    ("node ids bool", lambda ns: ns.__setitem__("node_ids", (True,) + tuple(ns["node_ids"])[1:])),
    ("node ids duplicate", lambda ns: ns.__setitem__("node_ids", (1, 1, 2, 3))),
    ("material name", lambda ns: ns.__setitem__("material_name", 3)),
    ("topology", lambda ns: ns.__setitem__("_is_4node", False)),
    ("topology missing", lambda ns: ns.pop("_is_quadrilateral")),
    ("topology truthy", lambda ns: ns.__setitem__("reduced_integration", 0)),
    ("normal writeable", lambda ns: ns.__setitem__("reference_normal", np.zeros(3))),
    ("normal list", lambda ns: ns.__setitem__("reference_normal", [0.0, 0.0, 1.0])),
)


@pytest.mark.parametrize(("label", "mutate"), _INSTANCE_MUTATIONS)
def test_g1_instance_guard_matches_the_verbatim_old_guard(label, mutate) -> None:
    new_guard = _q4_boundary_closures()[0]
    old_guard = _verbatim_old_instance_guard(_closure_cells(new_guard))
    model, _load = _plate(2)
    element = _first_q4(model)
    namespace = object.__getattribute__(element, "__dict__")
    mutate(namespace)
    _Counting.calls = 0
    old = _outcome(old_guard, element)
    old_calls = _Counting.calls
    _Counting.calls = 0
    new = _outcome(new_guard, element)
    assert new == old, label
    assert _Counting.calls <= old_calls
    if label == "unchanged":
        assert old is None


# ------------------------------------------------ successor record binding

ROOT = Path(__file__).resolve().parents[1]
GUARD_CADENCE_SUCCESSOR = (
    ROOT / "docs" / "reference_cases" / "qualified_shell_nonlinear_guard_cadence_successor.json"
)
SEALED_CARRIERS = ("e4_pl_element.py", "e4_pl_s3_state.py")
PORTABLE_CARRIERS = frozenset({"e4_pl_s3_state.py"})
OTHER_CHANGED_SOURCES = (
    "e4_pl_s3_v2d_state.py",
    "matrix_assembly.py",
    "nonlinear_performance_batch_c.py",
    "nonlinear_static.py",
)


def _identity(data: bytes) -> dict:
    return {
        "bytes": len(data),
        "git_blob": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest().upper(),
    }


def _strict_object(pairs):
    made = {}
    for key, value in pairs:
        if key in made:
            raise ValueError(f"duplicate key: {key}")
        made[key] = value
    return made


def _load_successor() -> dict:
    raw = GUARD_CADENCE_SUCCESSOR.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_strict_object)
    assert raw == (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return payload


def _assert_successor_bindings(payload: dict) -> None:
    assert payload["schema"] == "anysolver.qualified-shell-nonlinear-guard-cadence-successor-v1"
    assert set(payload["base"]) == {"commit", "tree"}
    assert all(len(value) == 40 for value in payload["base"].values())
    # Base: origin/main before the V1 canonical-JSON and G2/L1/G1 changes.
    assert payload["base"]["commit"].startswith("071910d60")
    assert payload["owner_decisions"]["date"] == "2026-11-24"
    assert payload["owner_decisions"]["approved"] == ["G2", "L1", "G1"]
    assert payload["qualification_boundary"] == {
        "default_policy_changed": False,
        "formulation_ids_changed": False,
        "mass_policy_changed": False,
        "mechanics_coefficients_or_operators_changed": False,
        "mechanics_symbols_changed": False,
        "numerical_tolerances_changed": False,
        "prior_evidence_modified": False,
        "qualified_recovery_changed": False,
    }
    change = payload["change"]
    assert tuple(sorted(change["sealed_carriers"])) == SEALED_CARRIERS
    for name, row in change["sealed_carriers"].items():
        assert row["live_path"] == f"src/anyfesolver/{name}"
        portable = name in PORTABLE_CARRIERS
        assert row["portable_text"] is portable
        live = (ROOT / row["live_path"]).read_bytes()
        canonical = live.replace(b"\r\n", b"\n")
        assert _identity(canonical) == row["current_lf"]
        if not portable:
            assert _identity(live) == row["current_raw"]
        source = canonical if portable else live
        lines = source.decode("utf-8").splitlines(keepends=True)
        for start, end, replacement in reversed(row["reverse_line_patch"]):
            lines[start:end] = replacement
        prior = "".join(lines).encode("utf-8")
        if not portable:
            assert _identity(prior) == row["prior_raw"]
        assert _identity(prior.replace(b"\r\n", b"\n")) == row["prior_lf"]
        assert row["prior_lf"] != row["current_lf"]
    assert tuple(sorted(change["other_changed_sources"])) == OTHER_CHANGED_SOURCES
    for name, row in change["other_changed_sources"].items():
        live = (ROOT / row["live_path"]).read_bytes().replace(b"\r\n", b"\n")
        assert _identity(live)["git_blob"] == row["current_lf_git_blob"]
        assert row["prior_lf_git_blob"] != row["current_lf_git_blob"]
    preserved = payload["preserved_evidence"]
    assert preserved["path"] == "docs/reference_cases/e4_pl_source_binding_20261111.json"
    data = (ROOT / preserved["path"]).read_bytes()
    assert {k: preserved[k] for k in ("bytes", "git_blob", "sha256")} == _identity(data)
    binding = json.loads(data)
    bound_rows = list(binding["carriers"].values()) + list(
        binding["sibling_imports_of_entry_module"]
    )
    for name, row in change["sealed_carriers"].items():
        # The source-binding record keeps describing the prior bytes exactly.
        matches = [r for r in bound_rows if r.get("live_path") == row["live_path"]]
        assert matches
        assert all(r["live_lf"] == row["prior_lf"] for r in matches)
    note = payload["validation"]["performance_note"]
    note_data = (ROOT / note["path"]).read_bytes()
    assert {k: note[k] for k in ("bytes", "git_blob", "sha256")} == _identity(note_data)
    assert payload["validation"]["regression_tests"] == "tests/test_nonlinear_guard_cadence.py"
    assert payload["review"]["outcome"] in {"PENDING", "ACCEPT"}
    if payload["review"]["outcome"] == "PENDING":
        assert payload["terminal"] == "PENDING_INDEPENDENT_REVIEW"


def test_guard_cadence_successor_binds_every_changed_source() -> None:
    _assert_successor_bindings(_load_successor())


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("base", "commit"), "0" * 40),
        (("qualification_boundary", "mechanics_symbols_changed"), True),
        (("change", "sealed_carriers", "e4_pl_element.py", "current_raw", "sha256"), "0"),
        (("change", "sealed_carriers", "e4_pl_element.py", "prior_lf", "git_blob"), "0"),
        (("change", "sealed_carriers", "e4_pl_s3_state.py", "current_lf", "sha256"), "0"),
        (("change", "sealed_carriers", "e4_pl_s3_state.py", "reverse_line_patch"), []),
        (("change", "sealed_carriers", "e4_pl_element.py", "reverse_line_patch"), []),
        (("change", "other_changed_sources", "nonlinear_static.py", "current_lf_git_blob"), "0"),
        (("preserved_evidence", "sha256"), "0"),
        (("owner_decisions", "approved"), ["G2", "L1", "G1", "S2"]),
    ),
)
def test_guard_cadence_successor_rejects_mutations(path, value) -> None:
    mutated = copy.deepcopy(_load_successor())
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError):
        _assert_successor_bindings(mutated)
