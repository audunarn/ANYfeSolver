"""Phase/cost probe for the 4x4 driver fixture under the guard environment.

Attempt-01 showed that one real 2x2 macro step does not complete inside a
60 s job under NUMBA_DISABLE_JIT=1. This probe does not change any
production mechanic: it wraps the certificate/report/increment entry points
with call-through timing recorders and only if the half solve actually
finishes asserts the same explicit fixture gates (completed single step
with real active contact). The recorders are installed through a
scope-managed monkeypatch context and the log is closed in a finally, so
an exception from the instrumented solve leaves the production hooks
restored and the log file closed; the recurring faulthandler thread was
removed because the external run_focused supervisor already captures
bounded time/stack artifacts. The log file is the evidence; no per-solve
timing is invented from a timeout.
"""
import time

import numpy as np
import pytest

from test_general_contact_admission_globalization_driver_small_patch import (
    _assert_active_contact,
    _case,
    _enabled,
    _solve,
)
from anyfesolver.general_contact import s3_patch_static, static_path
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


def _hook_targets():
    """The production hooks the probe wraps, as (owner, name, label)."""
    targets = [
        (s3_patch_static, "matching_projection_swept_lower", "matching"),
        (GeneralS3PatchManager, "evaluate", "evaluate"),
        (GeneralS3PatchManager, "admissible_step_fraction", "fraction"),
        (static_path, "solve_semismooth_increment", "increment"),
    ]
    if getattr(s3_patch_static, "overlap_violating_point_witness",
               None) is not None:
        targets.append(
            (s3_patch_static, "overlap_violating_point_witness", "witness"))
    return targets


def _probe(monkeypatch):
    """Install reversible call-through timing recorders.

    The wrappers are applied through the scope-managed monkeypatch
    fixture/context rather than a bare setattr, so the production hooks
    are restored even when the instrumented solve raises.
    """
    counts = {}

    def wrap(owner, name, label):
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                entry = counts.setdefault(
                    label, {"calls": 0, "seconds": 0.0, "last_kwargs": None})
                entry["calls"] += 1
                entry["seconds"] += time.perf_counter() - start
                entry["last_kwargs"] = {
                    key: (value if isinstance(value, (int, float, str, bool,
                                                       type(None)))
                          else repr(value)[:80])
                    for key, value in kwargs.items()}

        monkeypatch.setattr(owner, name, wrapper)

    for owner, name, label in _hook_targets():
        wrap(owner, name, label)
    return counts


def test_small_patch_solve_phase_probe(monkeypatch, tmp_path):
    log = open(tmp_path / "small_patch_probe.log", "w", buffering=1)
    t0 = time.perf_counter()

    def mark(message):
        log.write(f"{time.perf_counter()-t0:9.3f} {message}\n")

    targets = _hook_targets()
    originals = {label: getattr(owner, name)
                 for owner, name, label in targets}
    mark("probe start")
    try:
        model, definition, settings, program = _case()
        mark("fixture built")
        with monkeypatch.context() as mp:
            counts = _probe(mp)
            mark("half solve starts")
            result = _solve(model, definition, _enabled(settings), program,
                            max_load_factor=0.5,
                            emit_restart_checkpoint=True)
            mark("half solve returned")
        for owner, name, label in targets:
            assert getattr(owner, name) is originals[label]
        mark("production hooks restored after instrumented half solve")
        for label, entry in sorted(counts.items()):
            mark(f"count {label} calls={entry['calls']} "
                 f"seconds={entry['seconds']:.3f} "
                 f"last_kwargs={entry['last_kwargs']}")
        assert result.status == "completed" and len(result.steps) == 1
        checkpoint = result.to_restart_checkpoint()
        _assert_active_contact(checkpoint["path_state"]["history"][0])
        mark("fixture gates asserted (completed, active contact)")
    finally:
        log.close()


def test_probe_instrumentation_cleanup_on_solver_exception(monkeypatch,
                                                            tmp_path):
    """Exception-cleanup regression for the current probe instrumentation.

    The expensive solver is mocked to raise, so this node runs no
    mechanical solve: it asserts that the failing instrumented call leaves
    the production hooks restored and the probe log closed.
    """
    targets = _hook_targets()
    originals = {label: getattr(owner, name)
                 for owner, name, label in targets}
    log = open(tmp_path / "small_patch_probe_cleanup.log", "w",
               buffering=1)
    counts = None

    def mocked_expensive_solve():
        raise RuntimeError("simulated expensive solver failure")

    with pytest.raises(RuntimeError,
                       match="simulated expensive solver failure"):
        try:
            with monkeypatch.context() as mp:
                counts = _probe(mp)
                for owner, name, label in targets:
                    assert getattr(owner, name) is not originals[label]
                mocked_expensive_solve()
        finally:
            log.close()

    for owner, name, label in targets:
        assert getattr(owner, name) is originals[label]
    assert log.closed
    assert counts == {}
