"""Full-state work diagnostic for the fixed-support physical footprint route."""

import json
import os
from pathlib import Path

import numpy as np
import psutil
import pytest
from threadpoolctl import threadpool_info

from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
from examples.stiffened_shell_b2_physical_rigid_footprint import definition, preload


def test_focused_runtime_is_cpu0_one_thread_no_jit():
    if (os.environ.get("NUMBA_DISABLE_JIT") != "1"
            or psutil.Process().cpu_affinity() != [0]):
        pytest.skip("qualified focused runtime (CPU0 pinning, NUMBA_DISABLE_JIT=1)"
                    " is absent in ordinary portable CI")
    assert psutil.Process().cpu_affinity() == [0]
    assert os.environ["NUMBA_DISABLE_JIT"] == "1"
    pools = threadpool_info()
    assert all(pool["num_threads"] == 1 for pool in pools), pools
    if path := os.environ.get("B2_WORK_IDENTITY_CAPTURE"):
        Path(path).with_name("runtime.json").write_text(json.dumps({
            "cpu_affinity": psutil.Process().cpu_affinity(),
            "threadpools": pools,
            "jit_disabled": os.environ["NUMBA_DISABLE_JIT"],
        }, indent=2))


def test_zero_external_work_uses_registered_absolute_allowance():
    owner = api.create_analysis(definition=definition())
    try:
        assert owner._physical_footprint_law
        assert not owner.point_ids and owner._point_trace_policy is None
        assert not np.any(owner._u0)
        q = np.zeros(owner._T.shape[1])
        q[0] = 1.
        trial_u = owner._T @ q
        stiffness = float(trial_u @ (owner._K @ trial_u))
        assert stiffness > 0.
        q *= np.sqrt(5e-13 / stiffness)
        answer = np.r_[q, np.zeros(owner._C.shape[0])]
        load = np.zeros(owner._u0.shape)
        assert owner._physical_work_ratio(answer, load) == pytest.approx(.5)
        assert owner._physical_work_ratio(np.r_[2 * q, answer[len(q):]], load) == pytest.approx(2.)
    finally:
        owner.close()


def _terms(owner, result, load):
    u = np.asarray(result.displacements)
    lam = np.asarray(result.constraint_multipliers)
    gap = np.asarray(result.constraint_gaps)
    K, C, T = owner._K, owner._C, owner._T
    assert np.count_nonzero(owner._u0) == 0
    np.testing.assert_array_equal(T @ (T.T @ u), u)
    projected_residual = np.asarray(T.T @ (K @ u + C.T @ lam - load))
    work_defect = 2 * result.internal_strain_energy - u @ load
    equilibrium_work = (T.T @ u) @ projected_residual
    constraint_work = lam @ gap
    physical_scale = np.asarray(owner._abs_K @ np.abs(u) +
                                owner._abs_Ct @ np.abs(lam) + np.abs(load))
    local_scale = np.asarray(owner._abs_Tt @ physical_scale)
    by_node = physical_scale.reshape(-1, 6)
    physical_roundoff = np.tile(
        (float(np.max(by_node[:, :3])),) * 3 +
        (float(np.max(by_node[:, 3:])),) * 3, len(by_node))
    local_roundoff = np.asarray(owner._abs_Tt @ physical_roundoff)
    tolerance = 1e-9 * local_scale + len(u) * np.finfo(float).eps * local_roundoff
    potential_equilibrium_work = np.abs(T.T @ u) @ tolerance
    return dict(u=u, lam=lam, gap=gap, K=K, C=C, T=T,
                projected_residual=projected_residual,
                work_defect=work_defect, equilibrium_work=equilibrium_work,
                constraint_work=constraint_work,
                potential_equilibrium_work=potential_equilibrium_work,
                equilibrium_tolerance=tolerance,
                largest_equilibrium_ratio=np.max(np.abs(projected_residual) / tolerance))


@pytest.mark.parametrize("load_scale", (1., 100.))
def test_fixed_support_physical_work_identity_and_batch_parity(load_scale):
    owner = api.create_analysis(definition=definition())
    try:
        load = load_scale * preload()
        direct = owner.solve_linear(load)
        assert direct.status == "completed", direct.failure_reason
        terms = _terms(owner, direct, load)
        np.testing.assert_allclose(
            terms["work_defect"],
            terms["equilibrium_work"] - terms["constraint_work"],
            rtol=0, atol=1e-11 * max(1., abs(direct.internal_strain_energy)))
        assert abs(terms["work_defect"]) <= 1e-8 * abs(terms["u"] @ load)
        batch = owner.solve_linear_many((("same", load),), selected_dofs=tuple(range(len(load))))
        assert batch.status == "completed", batch.failure_reason
        np.testing.assert_array_equal(batch.cases[0].selected_displacements, terms["u"])
        assert batch.cases[0].internal_strain_energy == direct.internal_strain_energy
        if path := os.environ.get("B2_WORK_IDENTITY_CAPTURE"):
            target = Path(path).with_name(f"full-state-{int(load_scale)}.npz")
            target.parent.mkdir(parents=True, exist_ok=True)
            matrices = {}
            for name in ("K", "C", "T"):
                matrix = terms[name].tocsr()
                matrices.update({f"{name}_data": matrix.data,
                                 f"{name}_indices": matrix.indices,
                                 f"{name}_indptr": matrix.indptr,
                                 f"{name}_shape": matrix.shape})
            np.savez_compressed(target, load=load, **matrices,
                                **{key: value for key, value in terms.items()
                                   if key not in ("K", "C", "T")})
            target.with_suffix(".json").write_text(json.dumps({
                "work_defect": float(terms["work_defect"]),
                "equilibrium_work": float(terms["equilibrium_work"]),
                "constraint_work": float(terms["constraint_work"]),
                "external_work": float(terms["u"] @ load),
                "max_gap": float(np.max(np.abs(terms["gap"]))),
                "max_projected_residual": float(np.max(np.abs(terms["projected_residual"]))),
                "potential_equilibrium_work": float(terms["potential_equilibrium_work"]),
                "largest_equilibrium_ratio": float(terms["largest_equilibrium_ratio"]),
                "factorization_count": direct.factorization_count,
                "solve_count": direct.solve_count,
            }, indent=2))
    finally:
        owner.close()


def test_fixed_support_physical_work_batch_cancellation_preserves_completed_case():
    owner = api.create_analysis(definition=definition())
    token = CancellationToken()
    try:
        def cancel_after_first(_case_id):
            token.cancel("after first case")
        result = owner.solve_linear_many(
            (("first", preload()), ("second", 2 * preload())),
            selected_dofs=(12,), cancellation_token=token, on_case=cancel_after_first)
        assert result.status == "cancelled"
        assert len(result.cases) == 1
    finally:
        owner.close()


@pytest.mark.parametrize("size", ("small", "medium"))
def test_large_style_fixed_support_allocates_work_defect(size):
    from scripts.b2_large_scale_development import model, settings

    config = settings(size, 1)
    physical, inventory, shell_id = model(config)
    owner = api.create_analysis(definition=physical)
    try:
        load = np.zeros(inventory["dofs"])
        loaded_dof = 6 * (shell_id(config["nx"], config["ny"] // 2) - 1) + 2
        load[loaded_dof] = -1.
        result = owner.solve_linear(load, recover_members={})
        assert result.status == "completed", result.failure_reason
        terms = _terms(owner, result, load)
        np.testing.assert_allclose(
            terms["work_defect"],
            terms["equilibrium_work"] - terms["constraint_work"],
            rtol=0, atol=1e-8 * max(1., abs(result.internal_strain_energy)))
        assert abs(terms["work_defect"]) <= 1e-8 * abs(terms["u"] @ load)
        if path := os.environ.get("B2_WORK_IDENTITY_CAPTURE"):
            target = Path(path).with_name(f"full-state-large-style-{size}.npz")
            matrices = {}
            for name in ("K", "C", "T"):
                matrix = terms[name].tocsr()
                matrices.update({f"{name}_data": matrix.data,
                                 f"{name}_indices": matrix.indices,
                                 f"{name}_indptr": matrix.indptr,
                                 f"{name}_shape": matrix.shape})
            np.savez_compressed(target, load=load, **matrices,
                                **{key: value for key, value in terms.items()
                                   if key not in ("K", "C", "T")})
            target.with_suffix(".json").write_text(json.dumps({
                "work_defect": float(terms["work_defect"]),
                "equilibrium_work": float(terms["equilibrium_work"]),
                "constraint_work": float(terms["constraint_work"]),
                "external_work": float(terms["u"] @ load),
                "max_gap": float(np.max(np.abs(terms["gap"]))),
                "max_projected_residual": float(np.max(np.abs(terms["projected_residual"]))),
                "potential_equilibrium_work": float(terms["potential_equilibrium_work"]),
                "largest_equilibrium_ratio": float(terms["largest_equilibrium_ratio"]),
                "factorization_count": result.factorization_count,
                "solve_count": result.solve_count,
                "inventory": inventory,
            }, indent=2))
    finally:
        owner.close()


@pytest.mark.parametrize("correction_fraction", (1., .5, 0.))
def test_aggregate_work_triggers_bounded_correction_or_typed_failure(correction_fraction):
    from scripts.b2_large_scale_development import model, settings

    config = settings("medium", 1)
    physical, inventory, shell_id = model(config)
    owner = api.create_analysis(definition=physical)
    try:
        load = np.zeros(inventory["dofs"])
        load[6 * (shell_id(config["nx"], config["ny"] // 2) - 1) + 2] = -1.
        accurate = owner.solve_linear(load, recover_members={})
        assert accurate.status == "completed", accurate.failure_reason
        terms = _terms(owner, accurate, load)
        q = np.asarray(owner._T.T @ terms["u"])
        answer = np.r_[q, terms["lam"]]
        bias = np.r_[0.1 * np.sign(q) * terms["equilibrium_tolerance"],
                     np.zeros(len(terms["lam"]))]
        scaled_bias = owner._factor_scale * bias
        inaccurate = (answer / owner._factor_scale +
                      owner._handle.solve(scaled_bias))
        perturbed = owner._factor_scale * inaccurate
        assert owner._physical_equilibrium_ratio(perturbed, load) <= 1.
        assert owner._physical_work_ratio(perturbed, load) > 1.

        class BiasedFirstSolve:
            def __init__(self, original):
                self.original = original
                self.calls = 0

            def __getattr__(self, name):
                return getattr(self.original, name)

            def solve(self, rhs):
                self.calls += 1
                raw = self.original.solve(rhs)
                if self.calls == 1:
                    return inaccurate.copy()
                return correction_fraction * raw

        wrapper = BiasedFirstSolve(owner._handle)
        owner._handle = wrapper
        result = owner.solve_linear(load, recover_members={})
        assert wrapper.calls == 2
        if correction_fraction == 0.:
            assert result.status == "failed"
            assert "correction did not improve" in result.failure_reason
        elif correction_fraction == .5:
            assert result.status == "failed"
            assert "work consistency failed" in result.failure_reason
        else:
            assert result.status == "completed", result.failure_reason
            assert owner._physical_work_ratio(
                np.r_[owner._T.T @ result.displacements,
                      result.constraint_multipliers], load) <= 1.
    finally:
        owner.close()
