"""shell_axes_outputs after a solve and the ordinary stress recovery (2026-11-27).

The ordinary recovery leaves two solver-owned caches behind: the L4 batch
plan ``mesh._recovery_batch_plan`` (``RecoveryBatchPlan``, #95) and, on V2D
elements of a mixed model, the trusted-capture subscription list
``element._qualified_v2d_lease_epochs`` (weak references to
``_V2DLeaseEpoch``).  Before this change the #94 plain-object admission
refused every shell with ``AXES_OBJECT_NOT_PLAIN`` once either was present
(ANYfem D2-1: all 4596 Q4 + 44 V2D of the linear barge).  The module now
admits exactly these two values, bound to their owner type and key; the
axes result is bit-identical to the one on a fresh model without them.

Also the public gate ``admit_shell_axes(element, mesh)``: the very gate
``recover_shell_axes`` runs first (not a copy).
"""

from __future__ import annotations

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
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement, _V2DLeaseEpoch
from anyfesolver.recovery_batches import RecoveryBatchPlan
from anyfesolver.shell_axes_outputs import ShellAxesRefusal, admit_shell_axes, recover_shell_axes

PLAN = "_recovery_batch_plan"
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
        [weakref.ref(other)],  # a live reference to a foreign object
        [weakref.ref(live), object()],
    ):
        vars(v2d)[LEASES] = hostile
        _refused(v2d, mesh)
    # what the solver itself stores: dead and live references to its epochs
    dead = _V2DLeaseEpoch()
    dead_reference = weakref.ref(dead)
    del dead
    for admitted in ([], [dead_reference], [weakref.ref(live)], [dead_reference, weakref.ref(live)]):
        vars(v2d)[LEASES] = admitted
        admit_shell_axes(v2d, mesh)


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
