"""Focused checks for the static batch benchmark driver's measurement contract.

Covers the frozen case histories, the exact per-case selected-response digest,
and the driver-side equivalent backend-operation instrumentation used when the
measured checkout predates ``anyfesolver.linalg.backend_operation_counts``.
"""
import importlib.util
import math
from pathlib import Path
import types

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/benchmark_static_batches.py'
_spec = importlib.util.spec_from_file_location('benchmark_static_batches', SCRIPT)
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)


def test_frozen_histories_are_preserved():
    assert driver.positions('pilot') == list(range(0, 120, 6))
    assert driver.positions('primary') == list(range(0, 600, 6))
    assert driver.positions('holdout') == list(range(2, 600, 3))
    assert len(driver.positions('pilot')) == 20
    assert len(driver.positions('primary')) == 100
    assert len(driver.positions('holdout')) == 200
    primary = set(driver.positions('primary'))
    holdout = set(driver.positions('holdout'))
    assert not primary & holdout
    assert set(driver.positions('pilot')) <= primary


def test_case_digest_is_exact_and_order_independent():
    values = {('u', 3): 1.5, ('stress', 7, 'xx'): -2.0,
              ('support_reactions', 2, 4): 0.0}
    reordered = {('support_reactions', 2, 4): 0.0, ('u', 3): 1.5,
                 ('stress', 7, 'xx'): -2.0}
    assert driver.case_digest(values) == driver.case_digest(reordered)
    changed = dict(values)
    changed[('u', 3)] = math.nextafter(1.5, 2.0)
    assert driver.case_digest(changed) != driver.case_digest(values)
    other_key = dict(values)
    other_key[('u', 4)] = 1.5
    assert driver.case_digest(other_key) != driver.case_digest(values)


def test_case_summary_reports_separate_magnitudes():
    values = {('u', 0): -0.5, ('u', 1): 0.25,
              ('stress', 9, 'xx'): 4.0, ('stress', 9, 'yy'): -1.0,
              ('support_reactions', 2, 0): 0.75}
    summary = driver.case_summary(values)
    assert summary == {'u_abs_max': 0.5, 'stress_abs_max': 4.0,
                       'reaction_abs_max': 0.75}
    assert driver.case_summary({('u', 0): 0.}) == {
        'u_abs_max': 0., 'stress_abs_max': 0., 'reaction_abs_max': 0.}


class StubPardisoSolver:
    def __init__(self):
        self.phase = None

    def set_phase(self, phase):
        self.phase = phase


class StubFactorization:
    def __init__(self):
        self.stale_rebuild_count = 0
        self._private_solver = None

    def _active_solver(self):
        if self._private_solver is None:
            self.stale_rebuild_count += 1
            self._private_solver = object()
        return self._private_solver


def stub_linalg():
    module = types.ModuleType('stub_anyfesolver_linalg')
    module._pardiso_full_factorize = lambda solver, prepared: None
    module._new_pypardiso_solver = lambda *, mtype: StubPardisoSolver()
    module._PardisoFactorization = StubFactorization
    module.splu = lambda csc, permc_spec=None: object()
    return module


def test_native_backend_counts_are_used_without_wrapping():
    module = stub_linalg()
    native = {'scipy_factorization_attempts': 5}
    module.backend_operation_counts = lambda: dict(native)
    counts, instrumentation = driver.install_backend_counters(module)
    assert instrumentation['mode'] == 'native'
    assert instrumentation['wrapped_symbols'] == []
    assert counts() == native


def test_wrapper_counts_equivalent_baseline_operations():
    module = stub_linalg()
    counts, instrumentation = driver.install_backend_counters(module)
    assert instrumentation['mode'] == 'driver-wrapper'
    assert counts() == {'scipy_factorization_attempts': 0,
                        'pardiso_factorization_attempts': 0,
                        'pardiso_stale_rebuilds': 0}
    module._pardiso_full_factorize(None, None)
    assert counts()['pardiso_factorization_attempts'] == 1
    solver = module._new_pypardiso_solver(mtype=2)
    solver.set_phase(12)
    assert counts()['pardiso_factorization_attempts'] == 1
    solver.set_phase(22)
    assert counts()['pardiso_factorization_attempts'] == 2
    solver.set_phase(33)
    assert counts()['pardiso_factorization_attempts'] == 2
    later = module._new_pypardiso_solver(mtype=-2)
    later.set_phase(22)
    assert counts()['pardiso_factorization_attempts'] == 3
    handle = module._PardisoFactorization()
    handle._active_solver()
    assert counts()['pardiso_stale_rebuilds'] == 1
    handle._active_solver()
    assert counts()['pardiso_stale_rebuilds'] == 1
    module.splu(None)
    assert counts()['scipy_factorization_attempts'] == 1
    snapshot = counts()
    snapshot['pardiso_stale_rebuilds'] = 99
    assert counts()['pardiso_stale_rebuilds'] == 1


def test_wrapper_requires_equivalent_baseline_symbols():
    module = stub_linalg()
    del module._pardiso_full_factorize
    with pytest.raises(RuntimeError, match='_pardiso_full_factorize'):
        driver.install_backend_counters(module)


def test_phase_semantics_document_both_routes():
    assert set(driver.PHASE_SEMANTICS['baseline']) >= {
        'load_evaluation', 'solve', 'batch_solve', 'stress_recovery',
        'constraint_diagnostics', 'envelope_output'}
    assert set(driver.PHASE_SEMANTICS['candidate']) == {
        'stream_load_solve_recover', 'envelope_output'}
    assert 'note' in driver.PHASE_SEMANTICS
