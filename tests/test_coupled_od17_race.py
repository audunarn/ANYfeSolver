"""OD17 (2026-11-21): the poll/deadline race in ``_Run.result``.

A parent suspended between ``process.poll()`` (child still running) and the deadline check, while the child
exits and the deadline passes, must take the exited child's reply (docs "Certificate process: states",
verdict precedence), not report ``timeout``. The window is forced deterministically through the module's
``_remaining`` clock function (a test-side patch; production code is not patched)."""

from __future__ import annotations

import sys
import threading

import numpy as np
import pytest

from anyfesolver import _coupled_certificate as cc
from _coupled_worker_patch import patch_worker
from test_coupled_review_r1 import refused
from test_coupled_review_r8 import _diag


def _suspend_between_poll_and_deadline_check(monkeypatch):
    """The first ``_remaining`` call made by ``_Run.result`` on the calling thread (that is, right after its
    ``poll()`` saw the child running) waits until the child has exited and then reports the deadline as passed."""
    captured = {}
    original_adopt = cc._Run.adopt

    def adopt(self, process):
        captured['process'] = process
        return original_adopt(self, process)
    monkeypatch.setattr(cc._Run, 'adopt', adopt)
    original = cc._remaining
    state = {'armed': True, 'fired': False}

    def remaining(deadline):
        caller = sys._getframe(1).f_code.co_name
        if state['armed'] and caller == 'result' and threading.current_thread() is threading.main_thread():
            state['armed'] = False
            captured['process'].wait(timeout=60)      # the child exits while the parent is suspended
            state['fired'] = True
            return -1.0                               # ... and the deadline has passed
        return original(deadline)
    monkeypatch.setattr(cc, '_remaining', remaining)
    return state


def test_od17_child_exiting_in_the_poll_deadline_window_is_not_a_timeout(monkeypatch):
    patch_worker(monkeypatch, cc, 'import time\ntime.sleep(1.5)\n')
    state = _suspend_between_poll_and_deadline_check(monkeypatch)
    result = cc.certified_solve(_diag(4.0, 2.0, 1.0), np.ones(3), raw_load_zero=False)
    assert state['fired']
    assert result.x.tolist() == [0.25, 0.5, 1.0]


def test_od17_the_exited_childs_refusal_wins_in_the_window(monkeypatch):
    patch_worker(monkeypatch, cc, 'import time\ntime.sleep(1.5)\n')
    state = _suspend_between_poll_and_deadline_check(monkeypatch)
    error = refused('COUPLED_UNCERTIFIED', cc.certified_solve, _diag(0.0), np.ones(1), raw_load_zero=False)
    assert state['fired']
    assert error.reason == 'zero_row', (error.reason, error.diagnostics.get('worker_failure'))


def test_od17_a_child_still_running_after_the_deadline_is_a_timeout(monkeypatch):
    """Control: the re-poll does not turn a running child into an acceptance."""
    patch_worker(monkeypatch, cc, 'import time\ntime.sleep(30)\n')
    monkeypatch.setattr(cc, '_SVD_WORKER_TIMEOUT_SECONDS', 2.0)
    error = refused('COUPLED_UNCERTIFIED', cc._certify_solve_matrix, _diag(4.0, 2.0, 1.0), reason='svd_worker')
    assert error.diagnostics['worker_failure'] == 'timeout'
