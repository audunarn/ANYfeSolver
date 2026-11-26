"""CI coverage of both batched Q4 paths: the batched fast path and the forced per-element fallback.

Each case is solved and recovered twice in this process, once with the batched
numerics available and once inside ``_forced_fallback()`` (the per-element path).
The displacement vector and the complete ``compute_stresses`` result must be
bit-identical (dtype, shape and bytes), and the batch must have run only in the
fast path.  On Python 3.12+ the per-element stationary solves are counted
(``sys.monitoring``, read-only): every successful batch row must replace exactly
one per-element solve, so a batch that is attempted but not consumed fails.
"""
import hashlib
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4_batch_cases as cases  # noqa: E402

from anyfesolver import _q4_planar_batch as B  # noqa: E402
from anyfesolver import e4_pl_element as E4  # noqa: E402
from anyfesolver import solve_linear  # noqa: E402
from anyfesolver.assembly import compute_stresses  # noqa: E402

# Sizes above both batch minimum row counts (96 formation, 128 recovery Q4 rows).
CASES = [("plate", 12), ("sl", 12), ("pinch", 12), ("rplate", 12),
         ("mplate", 16), ("msl", 16), ("stiff", 12)]


class _CountSolves:
    """Count per-element ``_solve_stationary_system`` calls (no code is changed)."""

    TOOL = 3

    def __init__(self):
        self.calls = 0
        self.code = E4._solve_stationary_system.__code__

    def _callback(self, code, offset):
        if code is self.code:
            self.calls += 1

    def __enter__(self):
        if hasattr(sys, "monitoring"):
            mon = sys.monitoring
            mon.use_tool_id(self.TOOL, "q4-batch-path-count")
            mon.register_callback(self.TOOL, mon.events.PY_START, self._callback)
            mon.set_local_events(self.TOOL, self.code, mon.events.PY_START)
        return self

    def __exit__(self, *exc):
        if hasattr(sys, "monitoring"):
            mon = sys.monitoring
            mon.set_local_events(self.TOOL, self.code, 0)
            mon.register_callback(self.TOOL, mon.events.PY_START, None)
            mon.free_tool_id(self.TOOL)
        return False


def _run(case, n, monkeypatch, forced):
    rows = {"form": 0, "recover": 0}
    form, recover = B.form, B.recover

    def counting_form(*args, **kwargs):
        result = form(*args, **kwargs)
        rows["form"] += args[0].shape[0]  # counted only when the kernel succeeded
        return result

    def counting_recover(*args, **kwargs):
        result = recover(*args, **kwargs)
        rows["recover"] += args[0].shape[0]
        return result

    with monkeypatch.context() as patch:
        patch.setattr(B, "form", counting_form)
        patch.setattr(B, "recover", counting_recover)
        model, load = cases.CASES[case](n)
        with warnings.catch_warnings(), _CountSolves() as solves:
            warnings.simplefilter("ignore")
            if forced:
                with B._forced_fallback():
                    u, _ = solve_linear(model, load)
                    stresses = compute_stresses(model, u)
            else:
                u, _ = solve_linear(model, load)
                stresses = compute_stresses(model, u)
    hu, hs = hashlib.sha256(), hashlib.sha256()
    cases.feed(hu, np.asarray(u))
    cases.feed(hs, stresses)
    return hu.hexdigest(), hs.hexdigest(), rows, solves.calls


def test_batch_available_on_ci():
    assert B.available() is True
    with B._forced_fallback():
        assert B.available() is False
    assert B.available() is True


@pytest.mark.parametrize("case,n", CASES)
def test_fast_path_equals_forced_fallback(case, n, monkeypatch):
    fast = _run(case, n, monkeypatch, forced=False)
    fallback = _run(case, n, monkeypatch, forced=True)
    assert fast[:2] == fallback[:2]
    assert fallback[2] == {"form": 0, "recover": 0}
    assert fast[2]["recover"] > 0
    if hasattr(sys, "monitoring"):
        # Every successful batch row replaced exactly one per-element solve.
        assert fallback[3] > 0
        assert fast[3] == fallback[3] - fast[2]["form"] - fast[2]["recover"]
    if case == "rplate":
        # Distinct geometries: every element misses the shared-geometry cache.
        assert fast[2]["form"] == n * n


def test_path_check_detects_a_batch_that_is_never_consumed(monkeypatch):
    """Codex L4 round 1 P2: a kernel that refuses every call must not pass the
    path check (its rows are not counted and every element solves per element)."""

    def refusing(*args, **kwargs):
        raise B.PlanarBatchFallback("refused for the test")

    monkeypatch.setattr(B, "form", refusing)
    monkeypatch.setattr(B, "recover", refusing)
    fast = _run("rplate", 12, monkeypatch, forced=False)
    monkeypatch.undo()
    fallback = _run("rplate", 12, monkeypatch, forced=True)
    assert fast[:2] == fallback[:2]
    assert fast[2] == {"form": 0, "recover": 0}  # "recover > 0" in the path test fails
    if hasattr(sys, "monitoring"):
        assert fast[3] == fallback[3] > 0
