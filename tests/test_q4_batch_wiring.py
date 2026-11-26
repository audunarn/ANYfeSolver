"""Fault-injection tests for the batched qualified Q4 wiring.

Run:  PYTHONPATH=<src> python -m pytest -q test_q4_batch_wiring.py
The expectations are the per-element semantics, so the file also passes on the
unwired base tree (where no batch runs).
"""
import hashlib
import os
import sys
import warnings
from types import MappingProxyType

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4_batch_cases as q4bench  # noqa: E402

from anyfesolver import _q4_planar_batch as B  # noqa: E402
from anyfesolver import e4_pl_element as E4  # noqa: E402
from anyfesolver import matrix_assembly as MA  # noqa: E402
from anyfesolver import solve_linear  # noqa: E402

N = 12  # 144 distinct geometries (rplate): above both batch minimum sizes
WIRED = hasattr(E4, "_Q4_PLANAR_BATCH_FORM_MIN_ROWS")
# The faults are injected from sys.monitoring callbacks (sealed classes refuse
# monkeypatching); CI runs these on its Python 3.12+ shards.
requires_monitoring = pytest.mark.skipif(
    not hasattr(sys, "monitoring"), reason="fault injection uses sys.monitoring (Python 3.12+)"
)


@pytest.fixture(autouse=True)
def quiet():
    warnings.simplefilter("ignore")


def test_fixture_reaches_batch_minimum():
    if not WIRED:
        pytest.skip("unwired base tree")
    assert N * N >= max(E4._Q4_PLANAR_BATCH_FORM_MIN_ROWS, E4._Q4_PLANAR_BATCH_RECOVER_MIN_ROWS)


def canon(v):
    if isinstance(v, np.ndarray):
        return ("nd", str(v.dtype), v.shape, np.ascontiguousarray(v).tobytes())
    if isinstance(v, (dict, MappingProxyType)):
        return (type(v).__name__, tuple((k, canon(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return (type(v).__name__, tuple(canon(x) for x in v))
    if isinstance(v, float):
        return ("float", np.float64(v).tobytes())
    return (type(v).__name__, repr(v))


def invert(model, eid):
    e = model.mesh.elements[eid]
    nodes = [model.mesh.get_node(n) for n in e.node_ids]
    c = np.mean([[n.x, n.y, n.z] for n in nodes], axis=0)
    nodes[0].x, nodes[0].y = 2 * c[0] - nodes[0].x + 0.0, 2 * c[1] - nodes[0].y + 0.0


def outcome(fn):
    try:
        r = fn()
    except Exception as exc:  # noqa: BLE001
        return ("raise", type(exc).__name__, str(exc))
    h = hashlib.sha256()
    q4bench.feed(h, np.asarray(r))
    return ("ok", h.hexdigest())


def element_state(model):
    """Per-element cache state with model-specific identities removed."""
    out = {}
    for eid, e in model.mesh.elements.items():
        ns = object.__getattribute__(e, "__dict__")
        comps = ns.get("_qualified_components")
        key = ns.get("_qualified_cache_key")
        guard = ns.get("_qualified_component_guard")
        try:
            E4._require_q4_component_cache_provenance(e, comps, key, guard)
            provenance = "valid"
        except Exception as exc:  # noqa: BLE001
            provenance = type(exc).__name__
        out[eid] = (
            canon(comps),
            None if key is None else canon(key[2:]),  # drop id(mesh), id(material)
            guard is None,
            None if guard is None else (guard[8] is comps, guard[9] is key),
            canon(ns.get("_stiffness_matrix")),
            canon(ns.get("_hourglass_stiffness_matrix")),
            provenance,
        )
    return out


class _Fault:
    """Raise inside one sealed method for one element id, without mutating any code.

    The sealed classes and modules refuse monkeypatching (authority epochs), so
    the fault is raised from a ``sys.monitoring`` PY_START callback: the
    exception propagates from the first line of the monitored call.
    ``caller`` optionally restricts the fault to calls made directly from a
    function of that name.
    """

    TOOL = 5

    def __init__(self, function, element_id, message, caller=None):
        self.code = function.__code__
        self.element_id = element_id
        self.message = message
        self.caller = caller
        self.hits = 0

    def _callback(self, code, offset):
        if code is not self.code:
            return None
        frame = sys._getframe(1)
        element = frame.f_locals.get("self")
        if getattr(element, "element_id", None) != self.element_id:
            return None
        if self.caller is not None and frame.f_back.f_code.co_name != self.caller:
            return None
        self.hits += 1
        raise RuntimeError(self.message)

    def __enter__(self):
        mon = sys.monitoring
        mon.use_tool_id(self.TOOL, "q4-fault-injection")
        mon.register_callback(self.TOOL, mon.events.PY_START, self._callback)
        mon.set_local_events(self.TOOL, self.code, mon.events.PY_START)
        return self

    def __exit__(self, *exc):
        mon = sys.monitoring
        mon.set_local_events(self.TOOL, self.code, 0)
        mon.register_callback(self.TOOL, mon.events.PY_START, None)
        mon.free_tool_id(self.TOOL)
        return False


Q = E4.QualifiedE4PLShellElement
ASSEMBLY_LOOP = "_assemble_element_matrix_under_lease_impl"


def _collection_fault(element_id):
    # The assembly loop's own cache-key observation (collection time), not the
    # inner one of compute_stiffness_components.
    return _Fault(Q._qualified_stiffness_cache_key, element_id,
                  f"injected collection fault at element {element_id}", caller=ASSEMBLY_LOOP)


@requires_monitoring
def test_collection_fault_after_failing_miss_reports_earlier_error():
    """(i-a) Element 5 fails formation; element 20 fails at collection: 5 wins."""
    model, lc = q4bench.CASES["rplate"](N)
    invert(model, 5)
    with _collection_fault(20) as fault:
        got = outcome(lambda: solve_linear(model, lc)[0])
    # The wired loop reaches element 20's collection step before forming
    # element 5; the per-element loop fails at element 5 first.
    assert fault.hits == (1 if WIRED else 0)
    assert got == ("raise", "ValueError", "E4-PL element 5 has a nonpositive local Jacobian")


@requires_monitoring
def test_collection_fault_runs_preceding_misses_first():
    """(i-b) Only element 20 faults: its error is raised after every preceding miss formed."""
    reference, lc_ref = q4bench.CASES["rplate"](N)
    with B._forced_fallback():
        solve_linear(reference, lc_ref)
    ref_state = element_state(reference)

    model, lc = q4bench.CASES["rplate"](N)
    with _collection_fault(20) as fault:
        got = outcome(lambda: solve_linear(model, lc)[0])
    assert fault.hits == 1
    assert got == ("raise", "RuntimeError", "injected collection fault at element 20")
    state = element_state(model)
    for eid in range(1, 20):
        assert state[eid] == ref_state[eid], f"element {eid} not formed as per element"
    for eid in range(20, N * N + 1):
        assert state[eid][0] == canon(None), f"element {eid} formed after the fault"


def _bind_fault_run(forced, bound_element):
    model, lc = q4bench.CASES["rplate"](N)
    fault = _Fault(Q._bind_qualified_component_guard, bound_element,
                   f"injected bind fault at element {bound_element}")
    with fault:
        if forced:
            with B._forced_fallback():
                first = outcome(lambda: solve_linear(model, lc)[0])
        else:
            first = outcome(lambda: solve_linear(model, lc)[0])
    state = element_state(model)
    second = outcome(lambda: solve_linear(model, lc)[0])
    return first, state, second, element_state(model), fault.hits


@requires_monitoring
@pytest.mark.parametrize("bound_element", [1, 17, N * N])
def test_bind_fault_inside_batch_matches_per_element(bound_element):
    """(ii) A failing guard bind leaves the per-element state and the same next result."""
    reference = _bind_fault_run(True, bound_element)
    candidate = _bind_fault_run(False, bound_element)
    assert candidate[0] == reference[0]
    assert candidate[0][0] == "raise" and "injected bind fault" in candidate[0][2]
    assert candidate[1] == reference[1]  # state right after the failure
    assert candidate[2] == reference[2]  # next solve, fault removed
    assert candidate[3] == reference[3]
    assert reference[4] >= 1
    if WIRED:
        # The fault fired once inside the batch, then again in the per-element rerun.
        assert candidate[4] == reference[4] + 1
    else:
        assert candidate[4] == reference[4]


class _Bump(_Fault):
    """Advance the Q4 runtime-authority generation once (first matching call).

    Models a transient authority mutation inside one element's guarded span:
    nothing is raised by the callback; the guarded wrapper (per element) or
    the batch bracket must reject it.  ``element_id=None`` matches any call.
    """

    TOOL = 4

    def __init__(self, function, element_id):
        super().__init__(function, element_id, "")
        self.callers = []
        self.after = 0  # later calls for higher element ids (work past the rejection)

    def _callback(self, code, offset):
        if code is not self.code:
            return None
        frame = sys._getframe(1)
        if self.hits:
            if self.element_id is not None:
                eid = getattr(frame.f_locals.get("self"), "element_id", None)
                if isinstance(eid, int) and eid > self.element_id:
                    self.after += 1
            return None
        if self.element_id is not None:
            element = frame.f_locals.get("self")
            if getattr(element, "element_id", None) != self.element_id:
                return None
        self.hits += 1
        self.callers.append(frame.f_back.f_code.co_name)
        E4._q4_runtime_epoch_manager._generation += 1
        return None


# The guarded wrapper's ValueError ("... authority changed during guarded call")
# reaches the solve caller translated by the system assembly.
GENERATION_REJECTION = ("raise", "AssemblyError",
                        "system assembly exceptional output found incompatible qualified "
                        "shell authority")


def _solve_bump_run(forced, function, element_id):
    model, lc = q4bench.CASES["rplate"](N)
    bump = _Bump(function, element_id)
    with bump:
        if forced:
            with B._forced_fallback():
                first = outcome(lambda: solve_linear(model, lc)[0])
        else:
            first = outcome(lambda: solve_linear(model, lc)[0])
    state = element_state(model)
    second = outcome(lambda: solve_linear(model, lc)[0])
    return first, state, second, element_state(model), bump.hits, bump.callers, bump.after


@pytest.mark.parametrize("target", ["_constitutive_and_drill_stiffness",
                                    "_bind_qualified_component_guard"])
@requires_monitoring
@pytest.mark.parametrize("element_id", [1, 17, N * N])
def test_generation_change_in_batched_formation_span_rejected_as_per_element(target, element_id):
    """(iii) A generation move inside one batched element's span (front end or
    bind) is that element's rejection, exactly as the guarded wrapper's."""
    function = getattr(Q, target)
    reference = _solve_bump_run(True, function, element_id)
    candidate = _solve_bump_run(False, function, element_id)
    assert reference[0] == GENERATION_REJECTION
    assert candidate[0] == reference[0]
    assert candidate[1] == reference[1]  # cache state right after the rejection
    assert candidate[2] == reference[2]  # next solve
    assert candidate[3] == reference[3]
    assert reference[4] == candidate[4] == 1
    # Rejected at the same element: no later element's span work ran.
    assert reference[6] == 0 and candidate[6] == 0
    if WIRED:
        # The move happened inside the batch helper (its front end or its
        # lazy per-row publication), not in a per-element call.
        expected = "front_end" if target == "_constitutive_and_drill_stiffness" else "publish"
        assert candidate[5] == [expected]


def _stress_outcome(fn):
    try:
        r = fn()
    except Exception as exc:  # noqa: BLE001
        return ("raise", type(exc).__name__, str(exc))
    return ("ok", hashlib.sha256(repr(canon(r)).encode()).hexdigest(), tuple(r))


@requires_monitoring
@pytest.mark.parametrize("element_id", [1, 17, N * N])
def test_generation_change_in_batched_recovery_span_rejected_as_per_element(element_id):
    """(iv) The same for the recovery batch (front end of one row)."""
    from anyfesolver.assembly import compute_stresses

    results = []
    for forced in (True, False):
        model, lc = q4bench.CASES["rplate"](N)
        u, _ = solve_linear(model, lc)
        bump = _Bump(Q._constitutive_and_drill_stiffness, element_id)
        with bump:
            if forced:
                with B._forced_fallback():
                    got = _stress_outcome(lambda: compute_stresses(model, u))
            else:
                got = _stress_outcome(lambda: compute_stresses(model, u))
        state = element_state(model)
        again = _stress_outcome(lambda: compute_stresses(model, u))
        results.append((got, state, again, bump.hits, bump.callers, bump.after))
    reference, candidate = results
    assert reference[0][0] == "raise" or element_id not in reference[0][2]
    assert candidate[0] == reference[0]
    assert candidate[1] == reference[1]
    assert candidate[2] == reference[2]
    assert reference[3] == candidate[3] == 1
    if reference[0][0] == "raise":
        assert reference[5] == 0 and candidate[5] == 0
    else:
        # Not fail-closed: the per-element loop swallows the row's rejection and
        # continues; equal later work on both paths.
        assert candidate[5] == reference[5]
    if WIRED:
        assert candidate[4] == ["front_end"]


def _direct_items(model):
    mesh = model.mesh
    items = []
    for _eid, element in mesh.elements.items():
        material = model.get_material(element.material_name)
        items.append((element, material, element._qualified_stiffness_cache_key(mesh, material)))
    return mesh, items


@pytest.mark.parametrize("target", ["_constitutive_and_drill_stiffness",
                                    "_bind_qualified_component_guard"])
@requires_monitoring
@pytest.mark.parametrize("element_id", [1, 17])
def test_batch_helper_rejection_equals_guarded_wrapper(target, element_id):
    """(vi) Outside any operation lease: the batch helper's rejection, raised by
    the caller in per-element order, equals the guarded public wrapper's
    (exception type and message, rejected element, every element's cache state)."""
    if not WIRED:
        pytest.skip("unwired base tree")
    function = getattr(Q, target)
    model, _ = q4bench.CASES["rplate"](N)
    mesh, items = _direct_items(model)
    ref = None
    with _Bump(function, element_id):
        for element, material, _key in items:
            try:
                element.compute_stiffness_matrix(mesh, material)
            except Exception as exc:  # noqa: BLE001
                ref = (type(exc).__name__, str(exc), element.element_id)
                break
    ref_state = element_state(model)
    assert ref is not None and ref[0] == "ValueError" and "authority changed" in ref[1]

    model, _ = q4bench.CASES["rplate"](N)
    mesh, items = _direct_items(model)
    got = None
    with _Bump(function, element_id):
        result = Q._qualified_planar_batch_stiffness(items, mesh)
        assert result is not None
        publication, rejection = result
        # Drive the result as the assembly caller does: in item order, raise a
        # front-end rejection at its position, otherwise publish each row when
        # reached (a bind-span move is raised by ``get`` at that element).
        for index, (element, material, _key) in enumerate(items):
            try:
                if rejection is not None and index == rejection[0]:
                    E4._reject_q4_batch_generation(element, rejection[1])
                if publication.get(index) is None:
                    element.compute_stiffness_matrix(mesh, material)
            except Exception as exc:  # noqa: BLE001
                got = (type(exc).__name__, str(exc), element.element_id)
                break
    assert got == ref
    assert element_state(model) == ref_state


@requires_monitoring
def test_generation_change_during_batch_kernel_reruns_per_element():
    """(v) A move during the shared kernel span belongs to no element: the batch
    is discarded (no element rejected, no batched value bound); the operation-
    wide assembly lease still rejects the solve, as it rejects a move anywhere
    in a per-element solve.  The next solve equals the per-element one."""
    if not WIRED:
        pytest.skip("unwired base tree")
    reference = _solve_bump_run(True, Q._constitutive_and_drill_stiffness, N * N)
    candidate = _solve_bump_run(False, B.form, None)
    assert candidate[4] == 1
    assert candidate[0] == reference[0] == GENERATION_REJECTION
    assert candidate[2] == reference[2] and candidate[2][0] == "ok"
    assert candidate[3] == reference[3]


def test_wired_batch_actually_runs(monkeypatch):
    """Guard against the tests above silently exercising only the per-element path."""
    if not WIRED:
        pytest.skip("unwired base tree")
    rows = {"form": 0, "recover": 0}
    form, recover = B.form, B.recover

    def counting_form(*args, **kwargs):
        rows["form"] += args[0].shape[0]
        return form(*args, **kwargs)

    def counting_recover(*args, **kwargs):
        rows["recover"] += args[0].shape[0]
        return recover(*args, **kwargs)

    monkeypatch.setattr(B, "form", counting_form)
    monkeypatch.setattr(B, "recover", counting_recover)
    from anyfesolver.assembly import compute_stresses

    model, lc = q4bench.CASES["rplate"](N)
    u, _ = solve_linear(model, lc)
    compute_stresses(model, u)
    assert rows == {"form": N * N, "recover": N * N}


def _rejected_element_run(forced, failing_element):
    """A warped element with ``warped_formulation="reject"`` is not batch
    admitted; the per-element loop raises at it.  Elements after it in the
    assembly order must not have been published by the batch."""

    model, lc = q4bench.CASES["rplate"](N)
    element = model.mesh.elements[failing_element]
    object.__setattr__(element, "warped_formulation", "reject")
    model.mesh.get_node(element.node_ids[2]).z += 0.01
    if forced:
        with B._forced_fallback():
            first = outcome(lambda: solve_linear(model, lc)[0])
    else:
        first = outcome(lambda: solve_linear(model, lc)[0])
    return first, element_state(model)


@pytest.mark.parametrize("failing_element", [1, 17, N * N - N])
def test_failing_element_leaves_later_caches_unpublished(failing_element):
    """Codex L4 round 1 P2: the batch publishes each row only when the
    ordered per-element loop reaches it, so a failing element leaves exactly
    the per-element cache state (and error)."""

    reference = _rejected_element_run(True, failing_element)
    candidate = _rejected_element_run(False, failing_element)
    assert reference[0][0] == "raise"
    assert candidate[0] == reference[0]
    assert candidate[1] == reference[1]


def _later_stale_cache_run(forced):
    """Warm the last element alone, invert element 1, solve (fails), undo the
    inversion, solve again (succeeds)."""

    model, lc = q4bench.CASES["rplate"](N)
    late = model.mesh.elements[N * N]
    late.compute_stiffness_matrix(model.mesh, model.get_material(late.material_name))
    element = model.mesh.elements[1]
    node = model.mesh.get_node(element.node_ids[0])
    original = (node.x, node.y)
    invert(model, 1)

    def solve():
        if forced:
            with B._forced_fallback():
                return solve_linear(model, lc)[0]
        return solve_linear(model, lc)[0]

    failed = outcome(solve)
    late_kept = object.__getattribute__(late, "__dict__").get("_qualified_components") is not None
    node.x, node.y = original
    succeeded = outcome(solve)
    return failed, late_kept, succeeded, element_state(model)


def test_failing_assembly_with_later_stale_cache_converges():
    """Owner decision A (2026-11-26): in a failing assembly the batched route
    may already have applied later elements' cache-hit steps (here: a stale
    cache cleared).  Pinned: the same error, and a following successful solve
    ends in the same result and element state on both paths."""

    reference = _later_stale_cache_run(True)
    candidate = _later_stale_cache_run(False)
    assert reference[0][0] == "raise" and candidate[0] == reference[0]
    assert reference[1] is True
    if WIRED and B.available():
        # The documented difference (not a defect under decision A); with the
        # batch disabled by its self-check both runs keep the later cache.
        assert candidate[1] is False
    else:
        assert candidate[1] is True
    assert reference[2][0] == "ok" and candidate[2] == reference[2]
    assert candidate[3] == reference[3]
