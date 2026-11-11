"""Focused map/refusal controls; no assembly, factorization or physical solve."""
import numpy as np
import pytest

from anyfesolver._stiffened_shell_l1_assembly_experimental import compile_supported_shell_work
from anyfesolver._stiffened_shell_l1_supported_work_experimental import SupportedShellWorkMap


def test_effective_bounded_numerical_runtime():
    import os
    if "NUMBA_DISABLE_JIT" not in os.environ:
        pytest.skip("explicit NUMBA_DISABLE_JIT runtime binding absent")
    import numba
    import psutil
    from threadpoolctl import threadpool_info
    requested_jit = os.environ["NUMBA_DISABLE_JIT"]
    assert requested_jit in ("0", "1")
    if requested_jit == "1":
        assert psutil.Process().cpu_affinity() == [0]
    assert all(os.environ.get(key) == "1" for key in
               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"))
    assert int(numba.config.DISABLE_JIT) == int(requested_jit)
    assert numba.get_num_threads() == 1
    pools = threadpool_info()
    assert pools and all(pool["num_threads"] == 1 for pool in pools)


@pytest.mark.parametrize("requested_jit", ("0", "1"))
@pytest.mark.parametrize("defect", (
    None, "unknown_mode", "effective_jit", "numba_threads", "pool_threads",
    "no_pools", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "diagnostic_affinity",
))
def test_bounded_runtime_modes_and_refusals(monkeypatch, requested_jit, defect):
    """Exercise the actual runtime guard, including its diagnostic restrictions."""
    from types import SimpleNamespace
    import numba
    import psutil
    import threadpoolctl

    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        monkeypatch.setenv(key, "2" if defect == key else "1")
    monkeypatch.setenv("NUMBA_DISABLE_JIT",
                       "invalid" if defect == "unknown_mode" else requested_jit)
    effective = 1 - int(requested_jit) if defect == "effective_jit" else int(requested_jit)
    monkeypatch.setattr(numba.config, "DISABLE_JIT", effective)
    monkeypatch.setattr(numba, "get_num_threads",
                        lambda: 2 if defect == "numba_threads" else 1)
    affinity = [2, 3] if requested_jit == "0" or defect == "diagnostic_affinity" else [0]
    monkeypatch.setattr(psutil, "Process",
                        lambda: SimpleNamespace(cpu_affinity=lambda: affinity))
    pools = [] if defect == "no_pools" else [
        {"num_threads": 2 if defect == "pool_threads" else 1}]
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: pools)
    should_refuse = defect is not None and not (
        defect == "diagnostic_affinity" and requested_jit == "0")
    if should_refuse:
        with pytest.raises(AssertionError):
            test_effective_bounded_numerical_runtime()
    else:
        test_effective_bounded_numerical_runtime()


def family(**changes):
    nodes = tuple(range(1, 10))
    xyz = [(x, y, 0.) for y in (0., .1, .2) for x in (0., .5, 1.)]
    cells = ((1, 2, 5, 4), (2, 3, 6, 5), (4, 5, 8, 7), (5, 6, 9, 8))
    inputs = dict(shell_ids=nodes, shell_xyz=xyz, shell_cells=cells,
                  trace_nodes=(1, 2, 3), full_count=78, fixed_slots=tuple(range(6)))
    inputs.update(changes)
    return compile_supported_shell_work(**inputs)


def test_shared_rectangular_maps_keep_every_retained_role_zero():
    native, external = family(), family(full_count=60)
    assert native.evaluation.shape == (78, 6)
    assert np.array_equal(native.evaluation[:60], external.evaluation)
    assert np.count_nonzero(native.evaluation[54:]) == 0
    assert np.count_nonzero(native.evaluation) == 6
    assert np.all(native.evaluation[list(native.zero_slots)] == 0.)
    with pytest.raises(ValueError):
        native.evaluation.setflags(write=True)


def test_transpose_work_and_signed_bilinear_preserve_negative_and_unsymmetric_work():
    mapped = family()
    dual = np.arange(78, dtype=float)
    amplitude = np.array([1., -2., .5, -.3, 4., .7])
    assert np.isclose(mapped.evaluate(amplitude) @ dual, amplitude @ mapped.transpose_work(dual))
    k = np.diag(np.arange(78, dtype=float)-40.)
    k[24, 25] = 3.  # Preserve a raw unsymmetric action, never average it.
    work = mapped.body_work(k)
    assert work[0, 0] < 0.
    assert work[0, 1] != work[1, 0]
    assert np.isclose(amplitude @ work @ amplitude, mapped.evaluate(amplitude) @ k @ mapped.evaluate(amplitude))


def test_exact_constraint_kernel_is_for_every_density_not_a_selected_solve():
    mapped = family()
    c = np.zeros((5, 78))
    c[:, :18] = np.arange(90).reshape((5, 18))
    c[:, 54:] = -2.
    assert np.count_nonzero(mapped.require_constraint_kernel(c)) == 0
    for density in (np.ones(5), np.arange(5), -np.ones(5)):
        assert np.count_nonzero(mapped.transpose_work(c.T @ density)) == 0
    c[0, 24] = 1e-30
    with pytest.raises(ValueError, match="exact trace kernel"):
        mapped.require_constraint_kernel(c)


def test_explicit_physical_permutation_is_evaluation_not_an_invertible_transfer():
    original = family()
    slots = np.roll(np.arange(54).reshape((9, 6)), 1, axis=0)
    permuted = family(node_slots=slots, fixed_slots=())
    for i in range(9):
        assert np.array_equal(permuted.evaluation[slots[i]], original.evaluation[6*i:6*i+6])


@pytest.mark.parametrize("changes, message", [
    ({"trace_nodes": (5,)}, "boundary attachment"),
    ({"shell_cells": ((1, 2, 5, 4),)}, "Q4 cell correspondence"),
    ({"trace_nodes": (999,)}, "node identities"),
    ({"fixed_slots": (24,)}, "zero slots"),
    ({"node_slots": np.zeros((9, 6), dtype=int)}, "six unique"),
    ({"shell_xyz": [(x, y, .01 if i == 4 else 0.) for i, (x, y) in enumerate(
        ([(x, y) for y in (0., .1, .2) for x in (0., .5, 1.)]))]}, "flat Cartesian"),
])
def test_refuse_unmatched_topology_support_and_slots(changes, message):
    with pytest.raises(ValueError, match=message):
        family(**changes)


def test_reject_hidden_retained_value_or_nonfinite_dual():
    mapped = family()
    p = mapped.evaluation.copy()
    p[-1, 0] = 1.
    with pytest.raises(ValueError, match="beam or retained"):
        SupportedShellWorkMap(p, mapped.names, mapped.shell_slots, mapped.zero_slots)
    dual = np.zeros(78)
    dual[70] = np.nan
    with pytest.raises(ValueError, match="physical dual"):
        mapped.transpose_work(dual)


def test_constraint_cancellation_is_not_structural_exclusion():
    mapped = SupportedShellWorkMap(np.array([[1.], [1.]]), ("shared",), (0, 1), ())
    c = np.array([[1., -1.]])
    assert np.array_equal(c @ mapped.evaluation, np.zeros((1, 1)))
    with pytest.raises(ValueError, match="structural exact trace kernel"):
        mapped.require_constraint_kernel(c)
