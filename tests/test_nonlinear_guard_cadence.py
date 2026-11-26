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

import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
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


def _result_bytes(value) -> bytes:
    """Exact bytes of a result value (arrays, floats, nested containers)."""

    out = bytearray()

    def feed(item) -> None:
        if item is None or isinstance(item, (bool, int, str)):
            out.extend(repr((type(item).__name__, item)).encode())
        elif isinstance(item, float):
            out.extend(b"f" + np.float64(item).tobytes())
        elif isinstance(item, np.ndarray):
            out.extend(f"a{item.dtype}{item.shape}".encode())
            if item.dtype == object:
                for entry in item.ravel():
                    feed(entry)
            else:
                out.extend(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, np.generic):
            out.extend(b"g" + np.asarray(item).tobytes())
        elif isinstance(item, dict):
            out.extend(b"{")
            for key in sorted(item, key=repr):
                feed(key)
                feed(item[key])
            out.extend(b"}")
        elif isinstance(item, (list, tuple)):
            out.extend(b"[")
            for entry in item:
                feed(entry)
            out.extend(b"]")
        elif hasattr(item, "__dataclass_fields__"):
            out.extend(type(item).__name__.encode())
            feed({name: getattr(item, name) for name in item.__dataclass_fields__})
        elif hasattr(item, "__dict__"):
            out.extend(type(item).__name__.encode())
            feed(dict(vars(item)))
        else:
            out.extend(repr(item).encode())

    feed(value)
    return bytes(out)


@contextlib.contextmanager
def _g2_cadence_off():
    """Reference: every Q4 class check runs completely on every guarded call.

    The module namespaces are pinned by the lifecycle guards, so the reference
    replaces, for the duration of the solve, the window hook captured by each
    bound Q4 runtime guard with the unconditional per-call check (the pre-G2
    behaviour) and restores it afterwards.
    """

    cls = _q4.QualifiedE4PLShellElement
    cells = []
    for value in type.__getattribute__(cls, "__dict__").values():
        function = value.fget if isinstance(value, property) else value
        if not callable(function) or getattr(function, "__closure__", None) is None:
            continue
        guarded = _closure_cells(function)
        runtime_guard = guarded.get("runtime_guard")
        if runtime_guard is None:
            continue
        for name, cell in zip(runtime_guard.__code__.co_freevars, runtime_guard.__closure__):
            if name == "class_window_require" and cell not in [c for c, _ in cells]:
                cells.append((cell, cell.cell_contents))
    assert cells

    calls = []

    def per_call(exact_guard, element=None):
        del element
        calls.append(exact_guard)
        exact_guard()

    try:
        for cell, _original in cells:
            cell.cell_contents = per_call
        yield calls
    finally:
        for cell, original in cells:
            cell.cell_contents = original


@pytest.mark.parametrize("plastic", (False, True))
def test_g2_results_are_unchanged_on_an_unchanged_model(plastic) -> None:
    """G2 is result-neutral: exact bytes equal a cadence-off reference."""

    from anyfesolver import nonlinear_performance_bootstrap as _bootstrap

    model, load = _plate(plastic=plastic)
    _bootstrap.clear_nonlinear_assembly_cache()
    result = _solve(model, load)
    reference_model, reference_load = _plate(plastic=plastic)
    _bootstrap.clear_nonlinear_assembly_cache()
    with _g2_cadence_off() as per_call_checks:
        reference = _solve(reference_model, reference_load)
    assert per_call_checks  # the reference really ran the per-call checks
    assert result.status == reference.status == "completed"
    assert len(result.steps) == len(reference.steps) >= 2
    assert np.asarray(result.displacements).tobytes() == np.asarray(
        reference.displacements
    ).tobytes()
    assert _result_bytes(result.element_states) == _result_bytes(reference.element_states)
    assert _result_bytes(
        [(s.load_factor, s.iterations, s.residual_norm) for s in result.steps]
    ) == _result_bytes(
        [(s.load_factor, s.iterations, s.residual_norm) for s in reference.steps]
    )
    assert result.load_factor == reference.load_factor


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


# ------------------- G2 round 1: every window close validates (P1/P2 repairs)


def test_g2_nested_window_validates_before_returning(femesh_probe) -> None:
    """A nested window may skip only its first-use check, never its close."""

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    with pytest.raises(ValueError, match="qualified Q4 FEMesh class namespace changed"):
        with _q4._q4_class_authority_window():
            _q4._require_exact_q4_runtime_authority(element, context="outer")
            with _q4._q4_class_authority_window():
                _q4._require_exact_q4_runtime_authority(element, context="nested")
            femesh_probe.set()
            returned = []
            with _q4._q4_class_authority_window():
                _q4._require_exact_q4_runtime_authority(element, context="nested")
                returned.append("inside")
            returned.append("escaped")  # pragma: no cover - must not be reached
    assert returned == ["inside"]


def test_g2_swallowed_nested_refusal_is_refused_at_the_enclosing_close(
    femesh_probe,
) -> None:
    """Codex re-chain review P2 #1: a first-use refusal in a nested window that
    code inside the enclosing window catches must not let the enclosing window
    close successfully while the change persists."""

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    swallowed = []
    with pytest.raises(ValueError, match="qualified Q4 FEMesh class namespace changed"):
        with _q4._q4_class_authority_window():
            femesh_probe.set()
            try:
                with _q4._q4_class_authority_window():
                    _q4._require_exact_q4_runtime_authority(element, context="nested")
            except ValueError:
                swallowed.append("caught")
    assert swallowed == ["caught"]


def test_g2_swallowed_nested_refusal_reverted_before_close_is_accepted(
    femesh_probe,
) -> None:
    """Within the accepted cadence contract: the same change reverted before the
    enclosing window closes is not refused there."""

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    with _q4._q4_class_authority_window():
        femesh_probe.set()
        with pytest.raises(ValueError, match="class namespace changed"):
            with _q4._q4_class_authority_window():
                _q4._require_exact_q4_runtime_authority(element, context="nested")
        femesh_probe.clear()


def test_g2_swallowed_nested_close_refusal_is_refused_at_the_enclosing_close(
    femesh_probe,
) -> None:
    """Codex re-chain round 2 lead: a nested window whose own close validation
    refuses (the guard passed on first use there) and whose refusal is caught
    inside the enclosing window must not let the enclosing window close
    successfully while the change persists; a doubly nested close refusal
    caught one level up is owed by every enclosing window too."""

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    with pytest.raises(ValueError, match="qualified Q4 FEMesh class namespace changed"):
        with _q4._q4_class_authority_window():
            try:
                with _q4._q4_class_authority_window():
                    _q4._require_exact_q4_runtime_authority(element, context="nested")
                    femesh_probe.set()
            except ValueError:
                pass
    femesh_probe.clear()
    with pytest.raises(ValueError, match="qualified Q4 FEMesh class namespace changed"):
        with _q4._q4_class_authority_window():
            try:
                with _q4._q4_class_authority_window():
                    with _q4._q4_class_authority_window():
                        _q4._require_exact_q4_runtime_authority(element, context="deep")
                        femesh_probe.set()
            except ValueError:
                pass


def test_g2_enclosing_refusal_drops_caches_of_nested_only_elements(
    femesh_probe,
) -> None:
    """Codex re-chain review P2 #2: an element whose guarded call ran only in a
    nested window that closed successfully still has its derived caches
    dropped when the enclosing window refuses."""

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    with pytest.raises(ValueError, match="qualified Q4 FEMesh class namespace changed"):
        with _q4._q4_class_authority_window():
            with _q4._q4_class_authority_window():
                _q4._require_exact_q4_runtime_authority(element, context="nested")
            object.__getattribute__(element, "__dict__")["_nl_cache"] = {"probe": 1}
            femesh_probe.set()
    assert object.__getattribute__(element, "__dict__")["_nl_cache"] is None


def test_g2_nested_assembly_cannot_return_under_an_active_change(femesh_probe) -> None:
    """Caller code inside an outer window (e.g. a stiffness-scale mapping)
    runs a nested assembly: its result is refused while the change persists."""

    from anyfesolver.matrix_assembly import assemble_stiffness_matrix

    model, _load = _plate()
    assemble_stiffness_matrix(model)
    returned = []
    with pytest.raises((ValueError, AssemblyError)) as caught:
        with _q4._q4_class_authority_window():
            assemble_stiffness_matrix(model)
            femesh_probe.set()
            returned.append(assemble_stiffness_matrix(model))
    assert returned == []
    chain = []
    error = caught.value
    while error is not None:
        chain.append(str(error))
        error = error.__cause__ or error.__context__
    assert any("qualified Q4 FEMesh class namespace changed" in text for text in chain)


def test_g2_exceptional_window_exit_keeps_class_refusal_precedence(femesh_probe) -> None:
    """An invalid call under an active class change raises the class error
    first, as the per-call guard does (the original error is chained)."""

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    femesh_probe.set()
    with pytest.raises(ValueError, match="qualified Q4 FEMesh class namespace changed"):
        element.get_node_coordinates()
    femesh_probe.clear()
    with pytest.raises(ValueError, match="qualified Q4 FEMesh class namespace changed") as caught:
        with _q4._q4_class_authority_window():
            element.get_node_coordinates(model.mesh)
            femesh_probe.set()
            element.get_node_coordinates()
    assert isinstance(caught.value.__cause__, TypeError)


def test_g2_wrapped_class_refusal_keeps_its_wrapper(femesh_probe) -> None:
    """A class refusal wrapped by library code (as the assembly lease wraps it
    into AssemblyError) leaves the enclosing window unchanged."""

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    with pytest.raises(RuntimeError, match="wrapped") as caught:
        with _q4._q4_class_authority_window():
            _q4._require_exact_q4_runtime_authority(element, context="outer")
            femesh_probe.set()
            try:
                with _q4._q4_class_authority_window():
                    _q4._require_exact_q4_runtime_authority(element, context="nested")
            except ValueError as refusal:
                raise RuntimeError("wrapped") from refusal
    assert isinstance(caught.value.__cause__, ValueError)


def test_g2_exceptional_exit_without_class_change_is_unchanged() -> None:
    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    with pytest.raises(TypeError, match="mesh"):
        with _q4._q4_class_authority_window():
            element.get_node_coordinates(model.mesh)
            element.get_node_coordinates()


def test_g2_close_refusal_drops_element_caches(femesh_probe) -> None:
    """Derived Q4 caches written under a refused window are dropped."""

    from anyfesolver.matrix_assembly import assemble_stiffness_matrix

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    with pytest.raises(ValueError, match="FEMesh class namespace changed"):
        with _q4._q4_class_authority_window():
            _q4._require_exact_q4_runtime_authority(element, context="probe")
            object.__setattr__(element, "_stiffness_matrix", np.eye(2))
            femesh_probe.set()
    assert object.__getattribute__(element, "__dict__").get("_stiffness_matrix") is None
    femesh_probe.clear()
    stiffness, _info = assemble_stiffness_matrix(model)
    assert stiffness.shape[0] > 0


def _hook_plan_build(monkeypatch, after_build):
    from anyfesolver import nonlinear_performance as _performance

    built = []
    original = _performance.NonlinearAssemblyPlan.__dict__["build"].__func__

    def build(cls, model, num_layers):
        plan = original(cls, model, num_layers)
        built.append(plan)
        after_build()
        return plan

    monkeypatch.setattr(_performance.NonlinearAssemblyPlan, "build", classmethod(build))
    return built


def test_g2_plan_refused_at_window_close_is_not_reused(monkeypatch, femesh_probe) -> None:
    """Codex round 1 P1 #2: a plan built while the class change became
    active is refused and never served from the reusable cache."""

    from anyfesolver import nonlinear_performance_bootstrap as _bootstrap

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    _bootstrap.clear_nonlinear_assembly_cache(model)
    built = _hook_plan_build(monkeypatch, femesh_probe.set)
    with pytest.raises(ValueError, match="FEMesh class namespace changed"):
        with _q4._q4_class_authority_window():
            _q4._require_exact_q4_runtime_authority(element, context="prime")
            _bootstrap.get_nonlinear_assembly_plan(model, 5)
    assert len(built) == 1
    femesh_probe.clear()
    monkeypatch.undo()
    retry = _bootstrap.get_nonlinear_assembly_plan(model, 5)
    assert retry is not built[0]


def test_g2_published_plan_is_evicted_when_an_enclosing_window_refuses(
    femesh_probe,
) -> None:
    from anyfesolver import nonlinear_performance_bootstrap as _bootstrap

    model, _load = _plate()
    element = next(iter(model.mesh.elements.values()))
    _bootstrap.clear_nonlinear_assembly_cache(model)
    with pytest.raises(ValueError, match="FEMesh class namespace changed"):
        with _q4._q4_class_authority_window():
            _q4._require_exact_q4_runtime_authority(element, context="prime")
            plan = _bootstrap.get_nonlinear_assembly_plan(model, 5)
            femesh_probe.set()
    femesh_probe.clear()
    assert _bootstrap.get_nonlinear_assembly_plan(model, 5) is not plan
    # An accepted window keeps the published plan.
    with _q4._q4_class_authority_window():
        kept = _bootstrap.get_nonlinear_assembly_plan(model, 5)
    assert _bootstrap.get_nonlinear_assembly_plan(model, 5) is kept


def test_l1_complete_scans_are_entry_and_pre_commit_per_step(monkeypatch) -> None:
    """Codex round 1 P2 #4: per committed step exactly one entry and one
    pre-commit internal complete scan; none at tangent assembly or at the
    committed-state / reaction observations (covered by the step's
    unconditional support-reaction scan)."""

    from collections import Counter

    original = _nonlinear_static._EXACT_QUALIFIED_LIFECYCLE_GUARD
    contexts = []

    def counted(observed_model, *, context):
        contexts.append(context)
        return original(observed_model, context=context)

    monkeypatch.setattr(_nonlinear_static, "_EXACT_QUALIFIED_LIFECYCLE_GUARD", counted)
    result = _solve(*_plate(3, triangles=True))
    assert result.status == "completed"
    steps = len(result.steps)
    assert steps == 2
    counts = Counter(contexts)
    assert counts["nonlinear static force step cancellation"] == steps
    assert counts["nonlinear static step pre-commit"] == steps
    assert counts["nonlinear static support reactions"] == steps
    for context in (
        "nonlinear static force tangent assembly",
        "nonlinear static committed-state observation",
        "nonlinear static accepted reaction force",
        "nonlinear static reaction assembly",
        "nonlinear static reaction load",
        "nonlinear static force step window exit",
    ):
        assert counts[context] == 0, context


def test_l1_post_commit_deferral_is_scanned_before_the_step_is_reported(
    monkeypatch, v2d_probe
) -> None:
    """A persistent change at a deferred post-commit observation is refused
    before that step is reported to the caller."""

    model, load = _plate(3, triangles=True)
    committed = []
    original_commit = _nonlinear_static._commit_nonlinear_state_candidate

    def commit(*args, **kwargs):
        out = original_commit(*args, **kwargs)
        if not v2d_probe.active:
            v2d_probe.set()
        return out

    monkeypatch.setattr(_nonlinear_static, "_commit_nonlinear_state_candidate", commit)
    with pytest.raises(ElementCapabilityError):
        _solve(model, load, progress_callback=committed.append)
    assert committed == []


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
    "nonlinear_performance_bootstrap.py",
    "nonlinear_static.py",
)
CHANGE_COMMIT_TOUCHES = "src/anyfesolver/e4_pl_element.py"
GIT_ROOT = ROOT
# Predecessor: the newest link of the Q4 source-binding chain
# 2026-11-11 -> 2026-11-24 -> 2026-11-25 (pin from
# tests/test_e4_pl_s3_mixed_mesh_manifest.py, Q4_SOURCE_BINDING_DELTA2_SHA256).
PREDECESSOR = "docs/reference_cases/e4_pl_source_binding_20261125.json"
PREDECESSOR_SHA256 = "CFBA512F71CF589FC8C2E31E7EE12FA7A58E90F6ADDBAA28EA7E52FD7B3B3032"


def _read_checkout(relative: str) -> bytes:
    """Bytes of *relative* as this checkout has them (patched for LF/CRLF)."""

    return (ROOT / relative).read_bytes()


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(GIT_ROOT), *args], capture_output=True, check=False
    )


SHALLOW_BOUNDARY_SKIP = (
    "commit history is absent at the explicit GitHub shallow boundary "
    "(actions/checkout fetch-depth 1); every byte binding was checked, "
    "ancestry is checked in full clones and by the builder's --check"
)


def _is_explicit_github_shallow_boundary() -> bool:
    """True only in a GitHub Actions checkout that git itself records as
    shallow with HEAD as a shallow boundary.  Anywhere else absent history
    is a refusal, never a skip (same rule as the S3 V2 candidate binding)."""

    if os.environ.get("GITHUB_ACTIONS") != "true":
        return False
    shallow = _git("rev-parse", "--is-shallow-repository")
    if shallow.returncode or shallow.stdout.strip() != b"true":
        return False
    name = _git("rev-parse", "--git-path", "shallow")
    head = _git("rev-parse", "HEAD")
    if name.returncode or head.returncode:
        return False
    path = Path(os.fsdecode(name.stdout.strip()))
    if not path.is_absolute():
        path = Path(GIT_ROOT) / path
    if not path.is_file():
        return False
    return head.stdout.decode("ascii").strip() in path.read_text(encoding="ascii").splitlines()


def _history_present(*revisions: str) -> bool:
    return all(
        _git("cat-file", "-e", f"{revision}^{{commit}}").returncode == 0
        for revision in revisions
    )


def _require_history(*revisions: str) -> None:
    """Skip only at the explicit GitHub shallow boundary; refuse elsewhere."""

    if _history_present(*revisions):
        return
    assert _is_explicit_github_shallow_boundary(), (
        f"recorded commits are not in this repository: {revisions}"
    )
    pytest.skip(SHALLOW_BOUNDARY_SKIP)


def _assert_change_commit(commit) -> bool:
    """Validate change.commit; return False only when its history is absent
    at the explicit GitHub shallow boundary (the caller then skips after its
    byte checks).  Absent history anywhere else fails closed."""

    assert type(commit) is str and len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)
    if not _history_present(commit, f"{commit}^"):
        assert _is_explicit_github_shallow_boundary(), (
            f"change.commit {commit} (or its parent) is not in this repository"
        )
        return False
    assert _git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0
    touched = _git("diff", "--name-only", f"{commit}^", commit)
    assert touched.returncode == 0
    assert CHANGE_COMMIT_TOUCHES in touched.stdout.decode().split()
    return True


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


def _chain_bound_identities(delta2_raw: bytes) -> dict:
    """path -> (portable, identity) for every file the chain bound by the
    2026-11-25 record binds; each earlier link authenticated by exact bytes."""

    delta2 = json.loads(delta2_raw, object_pairs_hook=_strict_object)
    delta_path = delta2["predecessor"]["path"]
    delta_raw = (ROOT / delta_path).read_bytes()
    assert delta2["predecessor"] == {"path": delta_path, **_identity(delta_raw)}
    delta = json.loads(delta_raw, object_pairs_hook=_strict_object)
    origin_path = delta["predecessor"]["path"]
    origin_raw = (ROOT / origin_path).read_bytes()
    assert delta["predecessor"] == {"path": origin_path, **_identity(origin_raw)}
    origin = json.loads(origin_raw, object_pairs_hook=_strict_object)
    bound = {}
    rows = list(origin["carriers"].values()) + list(origin["sibling_imports_of_entry_module"])
    rows += list(origin["historical_gaps_of_current_historical_record"])
    for row in rows:
        entry = (False, row["live_raw"]) if "live_raw" in row else (True, row["live_lf"])
        assert entry[0] is False or row["portable_text"] is True
        assert bound.setdefault(row["live_path"], entry) == entry
    for record in (delta, delta2):
        for path, row in record["files"].items():
            assert bound.get(path, (False, None))[0] is False
            bound[path] = (False, row["live_raw"])
    return bound


def _load_successor() -> dict:
    raw = GUARD_CADENCE_SUCCESSOR.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_strict_object)
    assert raw == (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return payload


def _assert_successor_bindings(payload: dict) -> bool:
    assert payload["schema"] == "anysolver.qualified-shell-nonlinear-guard-cadence-successor-v1"
    assert set(payload["base"]) == {"commit", "tree"}
    assert all(len(value) == 40 for value in payload["base"].values())
    # Base: origin/main after S2 (#90), holding the 2026-11-25-bound bytes.
    assert payload["base"]["commit"].startswith("72faf02b1")
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
    history_verified = _assert_change_commit(change["commit"])
    assert tuple(sorted(change["sealed_carriers"])) == SEALED_CARRIERS
    for name, row in change["sealed_carriers"].items():
        assert row["live_path"] == f"src/anyfesolver/{name}"
        portable = name in PORTABLE_CARRIERS
        assert row["portable_text"] is portable
        live = _read_checkout(row["live_path"])
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
        # Codex P2 (#92 shallow review): a row must bind the file its key names.
        assert set(row) == {"current_lf_git_blob", "live_path", "prior_lf_git_blob"}
        assert row["live_path"] == f"src/anyfesolver/{name}"
        live = _read_checkout(row["live_path"]).replace(b"\r\n", b"\n")
        assert _identity(live)["git_blob"] == row["current_lf_git_blob"]
        assert row["prior_lf_git_blob"] != row["current_lf_git_blob"]
    assert "preserved_evidence" not in payload
    predecessor = payload["predecessor"]
    assert set(predecessor) == {"path", "bytes", "git_blob", "sha256"}
    assert predecessor["path"] == PREDECESSOR
    data = _read_checkout(predecessor["path"])
    assert {k: predecessor[k] for k in ("bytes", "git_blob", "sha256")} == _identity(data)
    assert predecessor["sha256"] == PREDECESSOR_SHA256
    chain = _chain_bound_identities(data)
    for name, row in change["sealed_carriers"].items():
        # The chain keeps describing the prior (base) bytes exactly.
        portable, bound = chain[row["live_path"]]
        assert portable is row["portable_text"]
        assert (row["prior_lf"] if portable else row["prior_raw"]) == bound
    for name, row in change["other_changed_sources"].items():
        assert row["live_path"] not in chain
    note = payload["validation"]["performance_note"]
    # Auto-text note: identified by its LF form on every checkout.
    assert note["text_identity"] == "lf"
    note_data = _read_checkout(note["path"]).replace(b"\r\n", b"\n")
    assert {k: note[k] for k in ("bytes", "git_blob", "sha256")} == _identity(note_data)
    assert payload["validation"]["regression_tests"] == "tests/test_nonlinear_guard_cadence.py"
    assert payload["review"]["outcome"] in {"PENDING", "ACCEPT"}
    if payload["review"]["outcome"] == "PENDING":
        assert payload["terminal"] == "PENDING_INDEPENDENT_REVIEW"
    return history_verified


def test_guard_cadence_successor_binds_every_changed_source() -> None:
    if not _assert_successor_bindings(_load_successor()):
        pytest.skip(SHALLOW_BOUNDARY_SKIP)


def _checkout_attributes(relative: str) -> dict:
    done = _git("check-attr", "text", "eol", "--", relative)
    assert done.returncode == 0
    values = {}
    for line in done.stdout.decode().splitlines():
        _path, attribute, value = line.rsplit(": ", 2)
        values[attribute] = value
    return values


@pytest.mark.parametrize("ending", (b"\n", b"\r\n"))
def test_guard_cadence_successor_is_checkout_independent(monkeypatch, ending) -> None:
    """Codex round 1 P2 #5: the record verifies the bytes an LF (Linux/macOS)
    or a CRLF (Windows autocrlf) checkout produces: auto-text files are read
    with either ending, ``-text`` and ``eol=lf`` files keep their bytes."""

    def checkout(relative: str) -> bytes:
        data = (ROOT / relative).read_bytes()
        attributes = _checkout_attributes(relative)
        if attributes.get("text") == "unset" or attributes.get("eol") == "lf":
            return data
        canonical = data.replace(b"\r\n", b"\n")
        return canonical if ending == b"\n" else canonical.replace(b"\n", b"\r\n")

    monkeypatch.setattr(sys.modules[__name__], "_read_checkout", checkout)
    payload = _load_successor()
    history_verified = _assert_successor_bindings(payload)
    hashed = [row["live_path"] for row in payload["change"]["sealed_carriers"].values()]
    hashed += [payload["predecessor"]["path"], payload["validation"]["performance_note"]["path"]]
    for relative in hashed:
        attributes = _checkout_attributes(relative)
        raw_bound = relative in {
            row["live_path"]
            for row in payload["change"]["sealed_carriers"].values()
            if not row["portable_text"]
        } or relative == payload["predecessor"]["path"]
        if raw_bound:
            # Raw bytes are hashed: the checkout must not convert them.
            assert attributes.get("text") == "unset" or attributes.get("eol") == "lf", relative
    if not history_verified:
        pytest.skip(SHALLOW_BOUNDARY_SKIP)


def test_guard_cadence_change_commit_is_validated() -> None:
    """Codex round 1 P3 #7: change.commit must be an ancestor commit whose
    diff touches the bound entry carrier."""

    payload = _load_successor()
    if not _assert_change_commit(payload["change"]["commit"]):
        pytest.skip(SHALLOW_BOUNDARY_SKIP)
    for bogus in ("0" * 40, "071910d60031beb455ef0fa2328a027d8ba77c3d"):
        with pytest.raises(AssertionError):
            _assert_change_commit(bogus)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "_guard_cadence_builder", ROOT / "scripts" / "build_guard_cadence_successor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_builder_history() -> None:
    payload = _load_successor()
    _require_history(
        payload["base"]["commit"],
        payload["change"]["commit"],
        *(entry.split(" ", 1)[0] for entry in payload["change"]["commits"]),
    )


def test_guard_cadence_builder_validates_every_change_commit() -> None:
    _require_builder_history()
    builder = _load_builder()
    recorded = _load_successor()["change"]["commits"]
    assert builder.validated_change_commits(list(recorded)) == recorded
    first, subject = recorded[0].split(" ", 1)
    for bogus in (
        recorded[:1] + ["0" * 9 + " " + subject],
        ["071910d60 " + subject] + recorded[1:],  # ancestor, touches no sealed carrier
        [first + " another subject"] + recorded[1:],
        [r for r in recorded if not r.startswith(builder.CHANGE_COMMIT[:9])],
        [],
    ):
        with pytest.raises(SystemExit):
            builder.validated_change_commits(bogus)


def test_guard_cadence_builder_refuses_a_patch_that_does_not_rebuild_the_base(monkeypatch) -> None:
    """The round-trip check is explicit (it must survive ``python -O``)."""

    _require_builder_history()
    builder = _load_builder()
    monkeypatch.setattr(builder, "apply_patch", lambda current, hunks: current + b"#")
    with pytest.raises(SystemExit, match="reverse line patch does not rebuild the base bytes"):
        builder.build({"hashes": {}})
    source = (ROOT / "scripts" / "build_guard_cadence_successor.py").read_text(encoding="utf-8")
    assert "assert apply_patch(" not in source


def test_guard_cadence_builder_check_is_fresh() -> None:
    _require_builder_history()
    done = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "build_guard_cadence_successor.py"), "--check"],
        cwd=ROOT, capture_output=True, check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.decode().strip() == "fresh"


def test_original_q4_source_binding_builder_check_still_passes() -> None:
    """Codex round 1 P2 #6: the standalone 20261111 verifier keeps verifying
    its historical pins, on the prior bytes reconstructed through the
    successor, plus the live mechanics closure."""

    revisions = ("e0c5f32ab40e9548ba0ea8981cf5e288ed375a1c",)
    for revision in revisions:
        if _git("cat-file", "-e", f"{revision}^{{commit}}").returncode != 0:
            pytest.skip("historical source-binding revisions are not present")
    done = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "build_q4_source_binding_20261111.py"), "--check"],
        cwd=ROOT, capture_output=True, check=False,
    )
    assert done.returncode == 0, (done.stdout + done.stderr).decode()[-3000:]


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
        (("predecessor", "sha256"), "0"),
        (("predecessor", "path"), "docs/reference_cases/e4_pl_source_binding_20261124.json"),
        (("base", "commit"), "071910d60031beb455ef0fa2328a027d8ba77c3d"),
        (("owner_decisions", "approved"), ["G2", "L1", "G1", "S2"]),
        (("change", "commit"), "0" * 40),
        (("change", "commit"), "071910d60031beb455ef0fa2328a027d8ba77c3d"),
        (("validation", "performance_note", "sha256"), "0"),
        (("validation", "performance_note", "text_identity"), "raw"),
        (("change", "other_changed_sources", "nonlinear_performance_bootstrap.py", "current_lf_git_blob"), "0"),
    ),
)
def test_guard_cadence_successor_rejects_mutations(path, value) -> None:
    if path == ("change", "commit"):
        # A bogus id is indistinguishable from absent history at the shallow boundary.
        _require_history(_load_successor()["change"]["commit"])
    mutated = copy.deepcopy(_load_successor())
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError):
        _assert_successor_bindings(mutated)


def _git_at(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@invalid", "-c", "commit.gpgsign=false",
         "-c", "init.defaultBranch=main", *args],
        cwd=cwd, capture_output=True, check=True,
    )


def test_guard_cadence_successor_defers_only_ancestry_at_the_github_shallow_boundary(
    tmp_path, monkeypatch
) -> None:
    """CI #92: actions/checkout fetch-depth 1 has no change-commit history,
    so ``merge-base --is-ancestor`` exits 128.  At that explicit boundary the
    record still verifies every byte binding and defers only ancestry; a
    missing commit anywhere else (no GitHub Actions, or a full clone) is
    still refused."""

    origin = tmp_path / "origin"
    origin.mkdir()
    _git_at(origin, "init", "-q")
    for message in ("one", "two"):
        _git_at(origin, "commit", "-q", "--allow-empty", "-m", message)
    shallow = tmp_path / "shallow"
    _git_at(tmp_path, "clone", "-q", "--depth", "1", origin.as_uri(), str(shallow))
    full = tmp_path / "full"
    _git_at(tmp_path, "clone", "-q", str(origin), str(full))
    module = sys.modules[__name__]
    payload = _load_successor()

    monkeypatch.setattr(module, "GIT_ROOT", shallow)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with pytest.raises(AssertionError):
        _assert_successor_bindings(payload)

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert _assert_successor_bindings(payload) is False
    for path in (
        ("change", "sealed_carriers", "e4_pl_element.py", "current_raw", "sha256"),
        ("change", "other_changed_sources", "nonlinear_static.py", "current_lf_git_blob"),
        ("predecessor", "sha256"),
    ):
        mutated = copy.deepcopy(payload)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = "0"
        with pytest.raises(AssertionError):
            _assert_successor_bindings(mutated)
    with pytest.raises(AssertionError):
        _assert_change_commit("z" * 40)

    monkeypatch.setattr(module, "GIT_ROOT", full)
    with pytest.raises(AssertionError):
        _assert_successor_bindings(payload)


@pytest.mark.parametrize("name", OTHER_CHANGED_SOURCES)
def test_guard_cadence_successor_rows_bind_the_file_their_key_names(name) -> None:
    """Codex P2 (#92 shallow review): replacing one other_changed_sources row
    with another valid row must be refused, so the keyed file is checked."""

    payload = _load_successor()
    rows = payload["change"]["other_changed_sources"]
    donor = next(other for other in OTHER_CHANGED_SOURCES if other != name)
    mutated = copy.deepcopy(payload)
    mutated["change"]["other_changed_sources"][name] = copy.deepcopy(rows[donor])
    with pytest.raises(AssertionError):
        _assert_successor_bindings(mutated)
