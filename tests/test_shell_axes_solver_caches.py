"""shell_axes_outputs after a solve and the ordinary stress recovery (2026-11-27).

The ordinary recovery leaves two solver-owned caches behind: the L4 batch
plan ``mesh._recovery_batch_plan`` (``RecoveryBatchPlan``, #95) and, on V2D
elements of a mixed model, the trusted-capture subscription list
``element._qualified_v2d_lease_epochs`` (callback-free weak references to
``_V2DLeaseEpoch``, never dereferenced by the module).  Before this change the #94 plain-object admission
refused every shell with ``AXES_OBJECT_NOT_PLAIN`` once either was present
(ANYfem D2-1: all 4596 Q4 + 44 V2D of the linear barge).  The module now
admits exactly these two values, bound to their owner type and key; the
axes result is bit-identical to the one on a fresh model without them.

Also the public gate ``admit_shell_axes(element, mesh)``: the very gate
``recover_shell_axes`` runs first (not a copy).
"""

from __future__ import annotations

import sys
import weakref

import numpy as np
import pytest

import anyfesolver.shell_axes_outputs as module
from anyfesolver import create_shell_element, recover_stress_result, solve_linear
from anyfesolver.cylinder_benchmarks import (
    CylinderBenchmarkConfig,
    build_cylindrical_shell_benchmark_model,
)
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement, _V2DLeaseEpoch
from anyfesolver.recovery_batches import RecoveryBatchPlan
from anyfesolver.s3_reference_batch import PreparedReferenceS3Components
from anyfesolver.shell_axes_outputs import ShellAxesRefusal, admit_shell_axes, recover_shell_axes

PLAN = "_recovery_batch_plan"
S3_PLAN = "_qualified_s3_reference_stiffness_plan"
LEASES = "_qualified_v2d_lease_epochs"


def _mixed_cylinder():
    """120 planar e4-pl Q4 + 48 V2D (the first ring split into triangles):
    above the 100-element batch-plan threshold, self-equilibrated pressure."""

    config = CylinderBenchmarkConfig(
        radius=1.5,
        height=2.0,
        thickness=0.02,
        pressure=2.0e4,
        num_circumferential=24,
        num_height=6,
        use_8node_elements=False,
    )
    model, load_case = build_cylindrical_shell_benchmark_model(config)
    mesh = model.mesh
    ids = sorted(mesh.elements)
    next_id = max(ids) + 1
    for index, element_id in enumerate(ids):
        old = mesh.elements[element_id]
        nodes = list(old.node_ids)
        if index >= config.num_circumferential:
            mesh.elements[element_id] = QualifiedE4PLShellElement(
                element_id, nodes, old.material_name, thickness=old.thickness
            )
            continue
        xyz = np.array([[mesh.nodes[n].x, mesh.nodes[n].y, mesh.nodes[n].z] for n in nodes])
        normal = np.cross(xyz[1] - xyz[0], xyz[3] - xyz[0])
        normal = np.asarray(normal / np.linalg.norm(normal), dtype=np.float64)
        del mesh.elements[element_id]
        for tri in ([nodes[0], nodes[1], nodes[2]], [nodes[0], nodes[2], nodes[3]]):
            mesh.elements[next_id] = create_shell_element(
                next_id,
                tri,
                old.material_name,
                formulation="e4-pl-s3-v2d",
                thickness=old.thickness,
                reference_normal=normal,
            )
            next_id += 1
    return model, load_case


def _qs3_cylinder():
    """288 qualified S3 triangles (every Q4 of the 24 x 6 cylinder split)."""

    config = CylinderBenchmarkConfig(
        radius=1.5,
        height=2.0,
        thickness=0.02,
        pressure=2.0e4,
        num_circumferential=24,
        num_height=6,
        use_8node_elements=False,
    )
    model, load_case = build_cylindrical_shell_benchmark_model(config)
    mesh = model.mesh
    ids = sorted(mesh.elements)
    next_id = max(ids) + 1
    for element_id in ids:
        old = mesh.elements[element_id]
        nodes = list(old.node_ids)
        xyz = np.array([[mesh.nodes[n].x, mesh.nodes[n].y, mesh.nodes[n].z] for n in nodes])
        normal = np.cross(xyz[1] - xyz[0], xyz[3] - xyz[0])
        normal = [float(v) for v in normal / np.linalg.norm(normal)]
        del mesh.elements[element_id]
        for tri in ([nodes[0], nodes[1], nodes[2]], [nodes[0], nodes[2], nodes[3]]):
            mesh.elements[next_id] = QualifiedE4PLS3ShellElement(
                next_id, tri, old.material_name, thickness=old.thickness, reference_normal=normal
            )
            next_id += 1
    return model, load_case


def _identical(a, b) -> bool:
    if isinstance(a, dict) or isinstance(b, dict):
        return type(a) is type(b) and a.keys() == b.keys() and all(_identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            type(a) is type(b)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and (a.tobytes() == b.tobytes() if a.dtype != object else _identical(a.tolist(), b.tolist()))
        )
    if isinstance(a, float) and isinstance(b, float):
        return type(a) is type(b) and np.float64(a).tobytes() == np.float64(b).tobytes()
    return type(a) is type(b) and a == b


def _axes_identical(x, y) -> bool:
    a, b = x.axes, y.axes
    return (
        _identical(x.recovery, y.recovery)
        and a.branch == b.branch
        and _identical(a.arrays, b.arrays)
        and a.refusals == b.refusals
        and a.certified == b.certified
        and _identical(a.diagnostics, b.diagnostics)
        and a.conventions == b.conventions
    )


@pytest.fixture(scope="module")
def solved():
    """(model after solve + ordinary recovery, fresh identical model, u, material)."""

    model, load_case = _mixed_cylinder()
    displacement, info = solve_linear(model, load_case, constraint_mode="auto")
    assert info["convergence_info"]["status"] == "converged"
    fresh, _ = _mixed_cylinder()
    recover_stress_result(model, displacement)
    return model, fresh, displacement, model.get_material("steel")


def test_recover_shell_axes_after_solve_and_ordinary_recovery_is_bit_identical(solved):
    """Regression (fails before the fix: every shell AXES_OBJECT_NOT_PLAIN)."""

    model, fresh, displacement, material = solved
    mesh = model.mesh
    assert type(vars(mesh)[PLAN]) is RecoveryBatchPlan
    assert PLAN not in vars(fresh.mesh)
    kinds = {QualifiedE4PLShellElement: 0, NativeParityE4PLS3V2DShellElement: 0}
    with_leases = 0
    for element_id, element in mesh.elements.items():
        kinds[type(element)] += 1
        with_leases += LEASES in vars(element)
        got = recover_shell_axes(element, mesh, displacement, material)
        reference = recover_shell_axes(fresh.mesh.elements[element_id], fresh.mesh, displacement, material)
        assert _axes_identical(got, reference), element_id
    assert kinds == {QualifiedE4PLShellElement: 120, NativeParityE4PLS3V2DShellElement: 48}
    assert with_leases == 48  # every V2D element carries the subscription list
    for element in fresh.mesh.elements.values():
        assert LEASES not in vars(element)


def _q4_v2d(model):
    q4 = next(e for e in model.mesh.elements.values() if type(e) is QualifiedE4PLShellElement)
    v2d = next(e for e in model.mesh.elements.values() if type(e) is NativeParityE4PLS3V2DShellElement)
    return q4, v2d


def _refused(element, mesh, code="AXES_OBJECT_NOT_PLAIN"):
    with pytest.raises(ShellAxesRefusal) as caught:
        admit_shell_axes(element, mesh)
    assert caught.value.code == code


def test_only_the_owned_key_on_the_owner_type_is_admitted():
    """Each cache is admitted only on its owner type, under its own key; the
    same value anywhere else stays refused (the generic plain rule)."""

    model, _ = _mixed_cylinder()
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    recover_stress_result(model, displacement)
    mesh = model.mesh
    q4, v2d = _q4_v2d(model)
    admit_shell_axes(q4, mesh)
    admit_shell_axes(v2d, mesh)
    plan = vars(mesh)[PLAN]
    leases = vars(v2d)[LEASES]

    vars(mesh)["_another_plan"] = plan  # plan under another mesh key
    _refused(q4, mesh)
    del vars(mesh)["_another_plan"]
    vars(q4)[PLAN] = plan  # plan on an element
    _refused(q4, mesh)
    del vars(q4)[PLAN]
    node = mesh.nodes[q4.node_ids[0]]
    vars(node)[PLAN] = plan  # plan on a node
    _refused(q4, mesh)
    del vars(node)[PLAN]
    epoch = _V2DLeaseEpoch()
    vars(q4)[LEASES] = [weakref.ref(epoch)]  # lease list on a Q4 element
    _refused(q4, mesh)
    del vars(q4)[LEASES]
    vars(mesh)[LEASES] = [weakref.ref(epoch)]  # lease list on the mesh
    _refused(v2d, mesh)
    del vars(mesh)[LEASES]
    admit_shell_axes(q4, mesh)
    assert vars(v2d)[LEASES] is leases
    admit_shell_axes(v2d, mesh)


def test_owned_keys_with_other_values_stay_refused():
    model, _ = _mixed_cylinder()
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    recover_stress_result(model, displacement)
    mesh = model.mesh
    q4, v2d = _q4_v2d(model)
    plan = vars(mesh)[PLAN]

    class PlanSubclass(RecoveryBatchPlan):
        pass

    for hostile in (object.__new__(PlanSubclass), object(), lambda: None, weakref.ref(plan)):
        vars(mesh)[PLAN] = hostile
        _refused(q4, mesh)
    vars(mesh)[PLAN] = plan
    admit_shell_axes(q4, mesh)

    class RefSubclass(weakref.ref):
        pass

    class ListSubclass(list):
        pass

    class Other:
        pass

    live = _V2DLeaseEpoch()
    other = Other()
    for hostile in (
        ListSubclass(),
        (weakref.ref(live),),  # tuple, not the solver's list
        [lambda: live],  # a callable, not a weak reference
        [RefSubclass(live)],  # a weak-reference subclass
        [weakref.ref(live, lambda _ref: None)],  # the solver registers no callback
        [weakref.ref(live), object()],
    ):
        vars(v2d)[LEASES] = hostile
        _refused(v2d, mesh)
    # what the solver itself stores: dead and live references to its epochs;
    # the referents are ignored (never dereferenced, never read)
    dead = _V2DLeaseEpoch()
    dead_reference = weakref.ref(dead)
    del dead
    for admitted in (
        [],
        [dead_reference],
        [weakref.ref(live)],
        [dead_reference, weakref.ref(live)],
        [weakref.ref(other)],
    ):
        vars(v2d)[LEASES] = admitted
        admit_shell_axes(v2d, mesh)


def test_lease_admission_takes_no_strong_reference_to_the_referent():
    """Codex r1 P1: dereferencing made the gate the last owner of an epoch
    whose release runs weak-reference callbacks registered elsewhere."""

    model, _ = _mixed_cylinder()
    recover_stress_result(model, np.zeros(model.mesh.dof_manager.total_dofs))
    _, v2d = _q4_v2d(model)
    epoch = _V2DLeaseEpoch()
    vars(v2d)[LEASES] = [weakref.ref(epoch)]
    fired = []
    watcher = weakref.ref(epoch, lambda _ref: fired.append(True))
    before = sys.getrefcount(epoch)
    admit_shell_axes(v2d, model.mesh)
    assert sys.getrefcount(epoch) == before and not fired
    del epoch
    assert fired and watcher() is None
    admit_shell_axes(v2d, model.mesh)  # a dead reference stays admitted


def test_missing_gate_attribute_is_a_typed_refusal(solved):
    """Codex r1 P2: a missing scalar raised AttributeError, not the contract's refusal."""

    model, _, displacement, material = solved
    q4, _ = _q4_v2d(model)
    thickness = vars(q4).pop("thickness")
    try:
        _refused(q4, model.mesh, "AXES_ATTRIBUTE_TYPE")
        with pytest.raises(ShellAxesRefusal) as caught:
            recover_shell_axes(q4, model.mesh, displacement, material)
        assert caught.value.code == "AXES_ATTRIBUTE_TYPE"
    finally:
        vars(q4)["thickness"] = thickness
    admit_shell_axes(q4, model.mesh)


def test_axes_never_read_the_batch_plan(solved):
    """Codex r1 P2: an exact but uninitialised plan (any attribute read raises)
    leaves recovery and axes bit-identical to the fresh model."""

    model, fresh, displacement, material = solved
    blank, _ = _mixed_cylinder()
    vars(blank.mesh)[PLAN] = object.__new__(RecoveryBatchPlan)
    for element_id, element in blank.mesh.elements.items():
        admit_shell_axes(element, blank.mesh)
        got = recover_shell_axes(element, blank.mesh, displacement, material)
        reference = recover_shell_axes(fresh.mesh.elements[element_id], fresh.mesh, displacement, material)
        assert _axes_identical(got, reference), element_id


def test_public_gate_is_the_gate_recover_shell_axes_runs(solved, monkeypatch):
    """``admit_shell_axes`` is the module's pre-recovery gate itself: the
    private name is the same object, and ``recover_shell_axes`` calls the
    public name first (a refusal from it stops the call before any access)."""

    model, _, displacement, material = solved
    q4, v2d = _q4_v2d(model)
    assert "admit_shell_axes" in module.__all__
    assert module._admit is module.admit_shell_axes
    assert admit_shell_axes(q4, model.mesh) is QualifiedE4PLShellElement
    assert admit_shell_axes(v2d, model.mesh) is NativeParityE4PLS3V2DShellElement

    calls = []

    class Sentinel(ShellAxesRefusal):
        pass

    def spy(element, mesh):
        calls.append((element, mesh))
        raise Sentinel("SPY", "gate reached")

    monkeypatch.setattr(module, "admit_shell_axes", spy)
    with pytest.raises(Sentinel):
        recover_shell_axes(q4, model.mesh, displacement, material)
    assert calls == [(q4, model.mesh)]
    with pytest.raises(Sentinel):
        module._derive_shell_axes_from_recovery(q4, model.mesh, displacement, material, {})
    assert len(calls) == 2


def test_public_gate_runs_before_the_recovery(solved, monkeypatch):
    """Codex r1 P2: the public gate is the FIRST step: when it refuses, nothing
    else runs (no coordinates, no recovery).  A gate called only later, inside
    the derivation, fails this test."""

    model, _, displacement, material = solved
    q4, v2d = _q4_v2d(model)
    for element in (q4, v2d):
        events = []
        kind = type(element)

        class Sentinel(ShellAxesRefusal):
            pass

        def spy(element, mesh):
            events.append("gate")
            raise Sentinel("SPY", "gate reached")

        def recorder(what):
            def recorded(*args, **kwargs):
                events.append(what)
                raise AssertionError(f"{what} ran before the gate")

            return recorded

        monkeypatch.setattr(module, "admit_shell_axes", spy)
        # every element access the module makes after the gate (Codex r2:
        # the V2D coordinate lookup reads mesh nodes before the recovery)
        for name in ("compute_stresses", "_coordinates", "get_node_coordinates"):
            if hasattr(kind, name):
                monkeypatch.setattr(kind, name, recorder(name))
        with pytest.raises(Sentinel):
            recover_shell_axes(element, model.mesh, displacement, material)
        monkeypatch.undo()
        assert events == ["gate"], events
    # and unpatched, the same elements recover normally afterwards
    recover_shell_axes(q4, model.mesh, displacement, material)
    recover_shell_axes(v2d, model.mesh, displacement, material)


def test_public_gate_raises_the_same_refusal_as_recover_shell_axes(solved):
    model, _, displacement, material = solved
    q4, _ = _q4_v2d(model)
    vars(q4)["_hostile"] = object()
    try:
        with pytest.raises(ShellAxesRefusal) as gate:
            admit_shell_axes(q4, model.mesh)
        with pytest.raises(ShellAxesRefusal) as full:
            recover_shell_axes(q4, model.mesh, displacement, material)
    finally:
        del vars(q4)["_hostile"]
    assert (gate.value.code, str(gate.value)) == (full.value.code, str(full.value))
    assert gate.value.code == "AXES_OBJECT_NOT_PLAIN"


# ------------------------------------------------- Codex r2 (2026-11-27)
def test_mapping_proxy_over_a_python_mapping_runs_no_code_and_is_refused(solved):
    """The proxy is read through its exact-dict backing only; a proxy over a
    Python mapping is refused without calling the mapping."""

    import collections
    import types

    model, _, _, _ = solved
    q4, _ = _q4_v2d(model)
    calls = []

    class Spy(collections.UserDict):
        def __iter__(self):
            calls.append("iter")
            return super().__iter__()

        def __getitem__(self, key):
            calls.append("getitem")
            return super().__getitem__(key)

        def __len__(self):
            calls.append("len")
            return super().__len__()

    vars(q4)["_proxy"] = types.MappingProxyType(Spy({"a": 1.0}))
    try:
        _refused(q4, model.mesh)
        assert calls == []
        vars(q4)["_proxy"] = types.MappingProxyType({"a": 1.0, "b": (2, 3)})
        admit_shell_axes(q4, model.mesh)
        vars(q4)["_proxy"] = types.MappingProxyType({"a": object()})
        _refused(q4, model.mesh)
    finally:
        del vars(q4)["_proxy"]


def test_refusal_text_runs_no_metaclass_code(solved):
    model, _, _, _ = solved
    q4, _ = _q4_v2d(model)
    hooks = []

    class Meta(type):
        def __getattribute__(cls, name):
            hooks.append(name)
            raise RuntimeError("metaclass hook ran")

    class Hostile(metaclass=Meta):
        pass

    _refused(Hostile(), model.mesh, "AXES_FORMULATION_NOT_QUALIFIED")
    vars(q4)["_hostile"] = Hostile()
    try:
        _refused(q4, model.mesh)
    finally:
        del vars(q4)["_hostile"]
    assert hooks == []


def test_huge_node_ids_are_admitted_when_present_and_refused_typed_when_missing(solved):
    """Codex r2/r3: a missing huge id must not escape as ValueError from the
    refusal text, and a present one (>= 2**63) is an ordinary admitted id."""

    from anyfesolver.fe_core import FEMesh

    model, _, _, _ = solved
    q4, _ = _q4_v2d(model)
    node_ids = vars(q4)["node_ids"]
    for hostile in (10**5000, -(10**5000), 2**63):
        vars(q4)["node_ids"] = type(node_ids)([hostile, *list(node_ids)[1:]])
        try:
            with pytest.raises(ShellAxesRefusal) as caught:
                admit_shell_axes(q4, model.mesh)
            assert caught.value.code == "AXES_OBJECT_NOT_PLAIN"
            assert ("<int of" in str(caught.value))  # every one of these is beyond the 10**18 print bound
        finally:
            vars(q4)["node_ids"] = node_ids
    admit_shell_axes(q4, model.mesh)

    mesh = FEMesh()
    big = [2**63 + k for k in (1, 2, 3, 4)]
    for node_id, xyz in zip(big, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))):
        mesh.add_node(node_id, *xyz)
    element = QualifiedE4PLShellElement(1, big, "steel", thickness=0.02)
    assert admit_shell_axes(element, mesh) is QualifiedE4PLShellElement


def test_refusal_text_runs_no_str_subclass_name_code(solved):
    """Codex r3 P1: a class may carry a str subclass as __name__; its
    __format__ must not run in the refusal text."""

    model, _, _, _ = solved
    q4, _ = _q4_v2d(model)
    hooks = []

    class Name(str):
        def __format__(self, spec):
            hooks.append("format")
            raise RuntimeError("name hook ran")

        def __str__(self):
            hooks.append("str")
            raise RuntimeError("name hook ran")

    class Hostile:
        pass

    Hostile.__name__ = Name("Hostile")
    assert type(Hostile.__name__) is Name
    with pytest.raises(ShellAxesRefusal) as caught:
        admit_shell_axes(Hostile(), model.mesh)
    assert caught.value.code == "AXES_FORMULATION_NOT_QUALIFIED"
    assert "non-str name" in str(caught.value)
    vars(q4)["_hostile"] = Hostile()
    try:
        _refused(q4, model.mesh)
    finally:
        del vars(q4)["_hostile"]
    assert hooks == []


def test_denied_proxy_inspection_is_a_refusal_not_an_exception(solved, monkeypatch):
    """Codex r3 P2: when gc.get_referents is denied (an audit hook raising),
    the proxy is refused instead of the exception escaping the gate."""

    import types

    model, _, _, _ = solved
    q4, _ = _q4_v2d(model)

    def denied(*_objects):
        raise PermissionError("gc.get_referents denied by policy")

    vars(q4)["_proxy"] = types.MappingProxyType({"a": 1})
    try:
        admit_shell_axes(q4, model.mesh)
        monkeypatch.setattr(module.gc, "get_referents", denied)
        _refused(q4, model.mesh)
    finally:
        del vars(q4)["_proxy"]


def test_lease_admitter_never_calls_a_list_item():
    """Codex r3 P2: a transient dereference (call, then release) leaves no
    reference-count trace, so the admitter's code is checked directly: its
    only calls are type(), len() and list.copy()."""

    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(module._admit_v2d_lease_epochs)))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            calls.append(ast.unparse(node.func))
    assert set(calls) <= {"type", "len", "list.copy"}, calls


# ------------------------------- qualified S3 stiffness plan (owner 2026-11-27)
def test_recover_shell_axes_after_repeated_assembly_on_qs3_is_bit_identical():
    """Regression (fails before the fix): the second stiffness assembly on
    qualified S3 shells stores ``mesh._qualified_s3_reference_stiffness_plan``;
    the gate refused every shell afterwards.  The axes after it (and after
    the ordinary recovery) equal the fresh model's bit for bit."""

    model, load_case = _qs3_cylinder()
    first, info = solve_linear(model, load_case, constraint_mode="auto")
    assert info["convergence_info"]["status"] == "converged"
    displacement, info = solve_linear(model, load_case, constraint_mode="auto")
    assert info["convergence_info"]["status"] == "converged"
    assert np.array_equal(first, displacement)
    mesh = model.mesh
    assert type(vars(mesh)[S3_PLAN]) is PreparedReferenceS3Components
    recover_stress_result(model, displacement)
    assert type(vars(mesh)[PLAN]) is RecoveryBatchPlan
    fresh, _ = _qs3_cylinder()
    assert S3_PLAN not in vars(fresh.mesh)
    material = model.get_material("steel")
    count = 0
    for element_id, element in mesh.elements.items():
        assert type(element) is QualifiedE4PLS3ShellElement
        got = recover_shell_axes(element, mesh, displacement, material)
        reference = recover_shell_axes(fresh.mesh.elements[element_id], fresh.mesh, displacement, material)
        assert _axes_identical(got, reference), element_id
        count += 1
    assert count == 288


def test_s3_plan_is_admitted_only_exact_and_only_on_the_mesh_key(solved):
    model, _, _, _ = solved
    q4, v2d = _q4_v2d(model)
    blank = object.__new__(PreparedReferenceS3Components)

    class PlanSubclass(PreparedReferenceS3Components):
        pass

    mesh_dict = vars(model.mesh)
    try:
        mesh_dict[S3_PLAN] = blank
        admit_shell_axes(q4, model.mesh)
        admit_shell_axes(v2d, model.mesh)
        for hostile in (object.__new__(PlanSubclass), object(), vars(model.mesh)[PLAN]):
            mesh_dict[S3_PLAN] = hostile
            _refused(q4, model.mesh)
        del mesh_dict[S3_PLAN]
        mesh_dict["_another_s3_plan"] = blank
        _refused(q4, model.mesh)
        del mesh_dict["_another_s3_plan"]
        vars(q4)[S3_PLAN] = blank
        _refused(q4, model.mesh)
    finally:
        mesh_dict.pop(S3_PLAN, None)
        mesh_dict.pop("_another_s3_plan", None)
        vars(q4).pop(S3_PLAN, None)
    admit_shell_axes(q4, model.mesh)


def test_axes_never_read_either_plan_during_qs3_and_mixed_recovery():
    """Codex r3 P2: exact but uninitialised plans (any attribute read raises)
    on the mesh leave every QS3, Q4 and V2D recovery and axes bit-identical."""

    for build in (_qs3_cylinder, _mixed_cylinder):
        model, load_case = build()
        displacement, info = solve_linear(model, load_case, constraint_mode="auto")
        assert info["convergence_info"]["status"] == "converged"
        material = model.get_material("steel")
        fresh, _ = build()
        blank, _ = build()
        vars(blank.mesh)[PLAN] = object.__new__(RecoveryBatchPlan)
        vars(blank.mesh)[S3_PLAN] = object.__new__(PreparedReferenceS3Components)
        for element_id, element in blank.mesh.elements.items():
            got = recover_shell_axes(element, blank.mesh, displacement, material)
            reference = recover_shell_axes(fresh.mesh.elements[element_id], fresh.mesh, displacement, material)
            assert _axes_identical(got, reference), (build.__name__, element_id)


# ------------------------------------------------- Codex r4 (2026-11-27)
def test_a_cache_cleared_during_admission_is_no_key_error(solved, monkeypatch):
    """The gate reads one snapshot of each instance dict: a cache removed
    while the gate runs (another thread clearing it) is no KeyError."""

    model, _, _, _ = solved
    q4, _ = _q4_v2d(model)
    mesh_dict = vars(model.mesh)
    plan = mesh_dict[PLAN]
    real = module._plain_value
    cleared = []

    first_key, first_value = next(iter(mesh_dict.items()))
    assert first_key != PLAN and type(first_value) not in (int, float, bool, str, type(None))

    def clearing(value):
        # fires while the mesh's own values are checked (after its snapshot)
        if value is first_value and not cleared and PLAN in mesh_dict:
            del mesh_dict[PLAN]  # what clear_recovery_batch_plan does
            cleared.append(True)
        return real(value)

    monkeypatch.setattr(module, "_plain_value", clearing)
    try:
        admit_shell_axes(q4, model.mesh)
        assert cleared
    finally:
        monkeypatch.undo()
        mesh_dict[PLAN] = plan
    admit_shell_axes(q4, model.mesh)


def test_axes_read_no_attribute_of_either_plan(monkeypatch):
    """Codex r4 P2: trace every attribute access on the populated plans while
    the axes run on QS3, Q4 and V2D shells (a guarded getattr would pass a
    bit-identity check)."""

    for build in (_qs3_cylinder, _mixed_cylinder):
        model, load_case = build()
        for _ in range(2):
            displacement, info = solve_linear(model, load_case, constraint_mode="auto")
            assert info["convergence_info"]["status"] == "converged"
        recover_stress_result(model, displacement)
        material = model.get_material("steel")
        plans = [vars(model.mesh)[key] for key in (PLAN, S3_PLAN) if key in vars(model.mesh)]
        assert type(plans[0]) is RecoveryBatchPlan
        if build is _qs3_cylinder:
            assert len(plans) == 2
        reads = []
        for kind in {type(plan) for plan in plans}:
            original = kind.__getattribute__

            def traced(self, name, _original=original):
                if any(self is plan for plan in plans):
                    reads.append((type(self).__name__, name))
                return _original(self, name)

            monkeypatch.setattr(kind, "__getattribute__", traced)
        try:
            for element in model.mesh.elements.values():
                recover_shell_axes(element, model.mesh, displacement, material)
        finally:
            monkeypatch.undo()
        assert reads == [], (build.__name__, reads[:5])
