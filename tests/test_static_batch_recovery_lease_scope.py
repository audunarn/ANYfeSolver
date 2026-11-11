"""Focused block recovery lease scope tests for the static batch stream.

These tests pin the borrowed block lease semantics: one captured
qualified recovery lease per block, complete per-operation guards for
every borrowed call, explicit revocation for contexts copied inside a
live scope, and unchanged case failure identities.
"""
from __future__ import annotations

import contextvars
from typing import Any

import numpy as np
import pytest

from anyfesolver import (BoundaryCondition, LoadCase, StaticCase, StaticBatchError,
    iter_linear_batches, RecoveryConfig, SolveCancelled, solve_linear,
    compute_constraint_force_diagnostics)
from anyfesolver.matrix_assembly import AssemblyError
from anyfesolver.recovery import recover_stress_result

import anyfesolver.recovery as recovery_module
import anyfesolver.recovery_batches as recovery_batches_module

from test_static_batch import _bar, _case


def _panel(num_divisions: int = 2):
    from anyfesolver import generate_simple_panel_mesh
    model = generate_simple_panel_mesh(1.0, 1.0, 0.01,
        num_divisions_x=num_divisions, num_divisions_y=num_divisions)
    model.boundary_conditions.clear()
    for node_id, node in model.mesh.nodes.items():
        model.add_boundary_condition(BoundaryCondition(
            "affine" + str(node_id), [node_id],
            {"ux": node.x * 1e-5, "uy": node.y * -2e-6, "uz": 0.0,
             "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    return model


def _counted_capture(monkeypatch: pytest.MonkeyPatch):
    """Count recovery lease captures and full lease checks."""
    original = recovery_batches_module._CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE
    captures: list[str] = []
    checks: list[str] = []

    def counted_capture(*args: Any, **kwargs: Any) -> Any:
        captures.append(kwargs.get("context"))
        lease = original(*args, **kwargs)

        def counted_lease(expected_model: Any, *, context: str, **options: Any) -> None:
            checks.append(context)
            lease(expected_model, context=context, **options)

        vars(counted_lease).update(vars(lease))
        return counted_lease

    monkeypatch.setattr(
        recovery_batches_module,
        "_CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE",
        counted_capture,
    )
    return captures, checks


def test_block_scope_amortizes_lease_captures_across_cases(monkeypatch):
    model = _bar()
    recovery = RecoveryConfig(node_ids=[1, 2], element_ids=[1],
        components=["axial_stress"], include_reactions=True)
    cases = [_case(index, force) for index, force
             in enumerate([1.0, -2.0, 3.0, -1.0])]
    captures, checks = _counted_capture(monkeypatch)

    # The public per-case route still captures one lease per operation.
    u, _ = solve_linear(model, cases[0].load_case)
    recover_stress_result(model, u, recovery)
    assert captures == ["element stress recovery preflight"]

    captures.clear()
    checks.clear()
    blocks = list(iter_linear_batches(model, cases, batch_size=4,
        recovery_config=recovery))
    assert [len(b.case_results) for b in blocks] == [4]
    # One captured lease per block, not per case or per operation.
    assert captures == ["static batch recovery block preflight"]
    # Every borrowed operation still pays its own full preflight/output
    # guards under the same contexts as the ordinary capture path.
    for stage in ("preflight", "output"):
        assert checks.count(f"element stress recovery {stage}") == 4
        assert checks.count(f"static batch prepared reactions {stage}") == 4

    captures.clear()
    checks.clear()
    blocks = list(iter_linear_batches(model, cases, batch_size=2,
        recovery_config=recovery))
    assert [len(b.case_results) for b in blocks] == [2, 2]
    assert captures == ["static batch recovery block preflight"] * 2
    for stage in ("preflight", "output"):
        assert checks.count(f"element stress recovery {stage}") == 4
        assert checks.count(f"static batch prepared reactions {stage}") == 4


def test_borrowed_block_outputs_match_public_recovery_exactly():
    model = _bar()
    cases = [_case(0, 2.0), _case(1, -1.0), _case(2, 0.5)]
    recovery = RecoveryConfig(node_ids=[1, 2], element_ids=[1],
        components=["axial_stress"])
    public = []
    for case in cases:
        u, _ = solve_linear(model, case.load_case)
        recovered = recover_stress_result(model, u, recovery)
        forces = compute_constraint_force_diagnostics(model, u, case.load_case)
        public.append((recovered.element_stresses, forces))
    blocks = list(iter_linear_batches(model, cases, batch_size=3,
        recovery_config=recovery))
    assert len(blocks) == 1
    for j, item in enumerate(blocks[0].case_results):
        stresses, forces = public[j]
        assert set(item["element_stresses"][1]) == set(stresses[1])
        for name, value in stresses[1].items():
            np.testing.assert_array_equal(item["element_stresses"][1][name], value)
        for bucket in ("support_reactions", "mpc_slave_forces",
                        "mpc_master_equivalent_forces"):
            assert set(item["constraint_forces"][bucket]) == set(forces[bucket])
            for node, force in forces[bucket].items():
                np.testing.assert_allclose(
                    item["constraint_forces"][bucket][node], force)


def test_foreign_edit_inside_borrowed_operation_fails_with_case_identity(
    monkeypatch,
):
    import anyfesolver.matrix_assembly as matrix_assembly_module
    model = _panel()
    recovery = RecoveryConfig(include_displacements=False,
        include_reactions=False, components=["von_mises"])
    original = recovery_module.recover_stress_result
    calls: list[int] = []

    def mutating_recover(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        calls.append(len(calls))
        if len(calls) == 1:
            # A foreign authority edit made during one borrowed operation
            # must be rejected by the next case's own guard boundary, with
            # that case's identity, exactly like the ordinary per-case
            # capture path.
            monkeypatch.setattr(matrix_assembly_module, "np", object())
        return result

    monkeypatch.setattr(recovery_module, "recover_stress_result",
        mutating_recover)
    with pytest.raises(StaticBatchError) as error:
        list(iter_linear_batches(model,
            [StaticCase("a", None), StaticCase("b", None)],
            batch_size=2, recovery_config=recovery))
    assert error.value.diagnostics["case_id"] == "b"
    assert error.value.diagnostics["input_index"] == 1
    assert error.value.diagnostics["stage"] == "stress recovery"
    assert isinstance(error.value.__cause__, AssemblyError)
    assert calls == [0]


def test_foreign_state_edit_between_borrowed_calls_still_terminates_stream(
    monkeypatch,
):
    model = _panel()
    recovery = RecoveryConfig(include_displacements=False,
        include_reactions=False, components=["von_mises"])
    original = recovery_module.recover_stress_result
    calls: list[int] = []

    def mutating_recover(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        calls.append(len(calls))
        if len(calls) == 1:
            # A coordinate edit the per-operation lease checks do not
            # cover must still be rejected by the unchanged case scan;
            # the held-open trusted scopes must not mask it.
            model.mesh.nodes[1].x = 2.0
        return result

    monkeypatch.setattr(recovery_module, "recover_stress_result",
        mutating_recover)
    with pytest.raises(StaticBatchError, match="model changed") as error:
        list(iter_linear_batches(model,
            [StaticCase("a", None), StaticCase("b", None)],
            batch_size=2, recovery_config=recovery))
    assert error.value.diagnostics["stage"] == "case recovery"
    assert calls == [0]
    assert recovery_batches_module._ACTIVE_QUALIFIED_RECOVERY_RUNTIME_LEASE_SCOPE.get() is None


def test_copied_context_uses_fresh_lease_after_block_close(monkeypatch):
    model = _panel()
    recovery = RecoveryConfig(include_displacements=False,
        include_reactions=False, components=["von_mises"])
    captures, _checks = _counted_capture(monkeypatch)
    original = recovery_module.recover_stress_result
    saved: dict[str, Any] = {}

    def context_copying_recover(*args: Any, **kwargs: Any) -> Any:
        if "context" not in saved:
            # A foreign callback copies the context while the block lease
            # scope is live and retains it past the stream boundary.
            saved["context"] = contextvars.copy_context()
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery_module, "recover_stress_result",
        context_copying_recover)
    stream = iter_linear_batches(model,
        [StaticCase("a", None), StaticCase("b", None)],
        batch_size=2, recovery_config=recovery)
    block = next(stream)
    assert captures == ["static batch recovery block preflight"]
    stream.close()

    u, _ = solve_linear(model, LoadCase("zero"))
    expected = recover_stress_result(model, u, recovery).element_stresses
    fresh_captures = len(captures)

    def run_in_copied_context():
        return recover_stress_result(model, u, recovery)

    result = saved["context"].run(run_in_copied_context)
    # The closed scope's borrowed lease was not reused: the copied context
    # paid a fresh capture for its own operation lease.
    assert len(captures) == fresh_captures + 1
    assert captures[-1] == "element stress recovery preflight"
    assert set(result.element_stresses) == set(expected)
    for element_id, values in expected.items():
        np.testing.assert_array_equal(
            result.element_stresses[element_id]["von_mises"],
            values["von_mises"])
    assert recovery_batches_module._ACTIVE_QUALIFIED_RECOVERY_RUNTIME_LEASE_SCOPE.get() is None
    del block


def test_scope_borrow_revocation_wrong_model_and_copied_context(monkeypatch):
    model = _bar()
    other = _bar()
    captures, _checks = _counted_capture(monkeypatch)
    scope = recovery_batches_module._qualified_recovery_runtime_lease_scope
    runner = recovery_batches_module._run_with_qualified_recovery_runtime_lease
    saved: dict[str, Any] = {}

    with scope(model, context="probe scope") as borrow:
        saved["borrow"] = borrow
        assert runner(model, context="probe operation",
            operation=lambda guard: "ok") == "ok"
        assert captures == ["probe scope preflight"]
        # A borrowing operation must present the captured model.
        with pytest.raises(AssemblyError, match="model identity"):
            borrow(other, operation_context="wrong model",
                operation=lambda guard: None)
        # Dispatch for another model inside the live scope takes the
        # ordinary fresh capture path instead of the borrowed lease.
        assert runner(other, context="other operation",
            operation=lambda guard: "ok") == "ok"
        assert captures == ["probe scope preflight", "other operation preflight"]
        saved["context"] = contextvars.copy_context()

    # After the scope closes the borrow is revoked even in the live context.
    with pytest.raises(AssemblyError, match="no longer active"):
        saved["borrow"](model, operation_context="stale",
            operation=lambda guard: None)

    # A copied context cannot retain the closed scope: dispatch falls back
    # to a fresh capture with full guards.
    def run_in_copied_context():
        assert runner(model, context="copied operation",
            operation=lambda guard: "copied") == "copied"

    saved["context"].run(run_in_copied_context)
    assert captures == ["probe scope preflight", "other operation preflight",
        "copied operation preflight"]
    assert recovery_batches_module._ACTIVE_QUALIFIED_RECOVERY_RUNTIME_LEASE_SCOPE.get() is None


@pytest.mark.parametrize("cancel", [False, True])
def test_recovery_failure_and_cancel_restore_scope_state(monkeypatch, cancel):
    from anyfesolver.e4_pl_element import _q4_trusted_operation_accepts
    model = _panel()
    recovery = RecoveryConfig(include_displacements=False,
        include_reactions=False, components=["von_mises"])
    original = recovery_module.recover_stress_result
    calls: list[int] = []
    failure = (SolveCancelled("recovery cancelled") if cancel
               else RuntimeError("foreign recovery failure"))

    def failing_recover(*args: Any, **kwargs: Any) -> Any:
        calls.append(len(calls))
        if len(calls) == 2:
            raise failure
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery_module, "recover_stress_result",
        failing_recover)
    with pytest.raises(SolveCancelled if cancel else StaticBatchError) as caught:
        list(iter_linear_batches(model,
            [StaticCase("a", None), StaticCase("b", None)],
            batch_size=2, recovery_config=recovery))
    if not cancel:
        assert caught.value.diagnostics["case_id"] == "b"
        assert caught.value.diagnostics["input_index"] == 1
        assert caught.value.diagnostics["stage"] == "stress recovery"
        assert caught.value.__cause__ is failure
    # The nested qualified trusted scopes closed and the lease scope
    # marker was cleared on the failure path.
    element = next(iter(model.mesh.elements.values()))
    assert not _q4_trusted_operation_accepts(element)
    assert recovery_batches_module._ACTIVE_QUALIFIED_RECOVERY_RUNTIME_LEASE_SCOPE.get() is None
    # A later public recovery still works with its own fresh lease.
    u, _ = solve_linear(model, LoadCase("zero"))
    assert recover_stress_result(model, u, recovery).element_stresses


def test_scope_entry_failure_keeps_case_identity(monkeypatch):
    model = _bar()
    recovery = RecoveryConfig(node_ids=[1, 2], element_ids=[1],
        components=["axial_stress"])

    def broken_collection(model_arg):
        raise RuntimeError("trusted element collection failed")

    monkeypatch.setattr(recovery_batches_module,
        "_qualified_recovery_trusted_elements", broken_collection)
    with pytest.raises(StaticBatchError) as error:
        next(iter_linear_batches(model, [_case(0)],
            recovery_config=recovery))
    assert error.value.diagnostics["stage"] == "recovery lease scope"
    assert error.value.diagnostics["case_id"] == "0"
    assert error.value.diagnostics["input_index"] == 0
    assert isinstance(error.value.__cause__, RuntimeError)
    assert recovery_batches_module._ACTIVE_QUALIFIED_RECOVERY_RUNTIME_LEASE_SCOPE.get() is None


def test_free_free_nullspace_diagnostics_survive_borrowed_block():
    model = _bar(False)
    cases = [_case(0, 1, True), _case(1, -2, True)]
    expected = [solve_linear(model, c.load_case)[0] for c in cases]
    recovery = RecoveryConfig(node_ids=[1, 2], include_stresses=False,
        include_displacements=False)
    blocks = list(iter_linear_batches(model, cases, batch_size=2,
        recovery_config=recovery))
    assert len(blocks) == 1
    for j, item in enumerate(blocks[0].case_results):
        info = item["solver_info"]
        assert info["method"] == "rigid_body_nullspace_augmented"
        assert info["gauge_residual_norm"] < 1e-12
        np.testing.assert_allclose(blocks[0].displacements[:, j],
            expected[j], rtol=1e-11, atol=1e-13)
        for bucket in ("support_reactions", "mpc_slave_forces",
                        "mpc_master_equivalent_forces"):
            assert bucket in item["constraint_forces"]


def test_prescribed_mpc_reactions_match_public_under_borrowed_block():
    model = _bar()
    model.boundary_conditions[0].dof_constraints["ux"] = 0.02
    model.add_node(3, 2.0, 0.0, 0.0)
    model.add_boundary_condition(BoundaryCondition("slave_other", [3],
        {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    model.add_constraint_equation(terms=((model.mesh.nodes[3].dofs[0], 1.0),
        (model.mesh.nodes[2].dofs[0], -1.0)), rhs=0.01, source_id="affine",
        dependent_dof=model.mesh.nodes[3].dofs[0])
    cases = []
    for index, force in enumerate([2.0, -1.5]):
        case = _case(index, force)
        case.load_case.add_nodal_load(3, [3, 0, 0, 0, 0, 0])
        cases.append(case)
    public = []
    for case in cases:
        u, _ = solve_linear(model, case.load_case)
        public.append(compute_constraint_force_diagnostics(
            model, u, case.load_case))
    recovery = RecoveryConfig(node_ids=[1, 3], include_stresses=False,
        include_displacements=False)
    blocks = list(iter_linear_batches(model, cases, batch_size=2,
        recovery_config=recovery))
    assert len(blocks) == 1
    for j, item in enumerate(blocks[0].case_results):
        forces = public[j]
        assert item["reaction_node_ids"] == (1, 3)
        assert "residual" not in item["constraint_forces"]
        for bucket in ("support_reactions", "mpc_slave_forces",
                        "mpc_master_equivalent_forces"):
            # Selected reaction requests filter the physical buckets to the
            # requested nodes; every reported node must match the public
            # per-case route exactly.
            stream_bucket = item["constraint_forces"][bucket]
            assert set(stream_bucket) <= set(forces[bucket])
            for node, force in stream_bucket.items():
                np.testing.assert_allclose(force, forces[bucket][node])


def _s3_triangle():
    """One prescribed-displacement S3 triangle for token lifetime tests."""
    from anyfesolver import FEModel, QualifiedE4PLS3ShellElement
    model = FEModel("s3-operation-token-lifetime")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    coordinates = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.5, 0.5 * np.sqrt(3.0), 0.0))
    for node_id, point in enumerate(coordinates, start=1):
        model.add_node(node_id, *point)
    model.add_element(1, QualifiedE4PLS3ShellElement(1, (1, 2, 3), "steel",
        thickness=0.02, reference_normal=(0.0, 0.0, 1.0)))
    for node_id, node in model.mesh.nodes.items():
        model.add_boundary_condition(BoundaryCondition(str(node_id), [node_id],
            {"ux": node.x * 1e-5, "uy": 0.0, "uz": 0.0,
             "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    return model


def _trusted_operation_case(family):
    """Return (element, scope, accepts) for the Q4 or S3 trusted scope."""
    if family == "q4":
        from anyfesolver.e4_pl_element import (
            _q4_trusted_operation_accepts as accepts,
            _q4_trusted_operation_scope as scope,
        )
        model = _panel(1)
        element = next(iter(model.mesh.elements.values()))
    else:
        from anyfesolver.e4_pl_s3_element import (
            _s3_trusted_operation_accepts as accepts,
            _s3_trusted_operation_scope as scope,
        )
        model = _s3_triangle()
        element = model.mesh.elements[1]
    return element, scope, accepts


def test_copied_context_cannot_retain_q4_operation_token_after_close(
    monkeypatch,
):
    from anyfesolver.e4_pl_element import _q4_trusted_operation_accepts
    model = _panel(1)
    recovery = RecoveryConfig(include_displacements=False,
        include_reactions=False, components=["von_mises"])
    original = recovery_module.recover_stress_result
    saved: dict[str, Any] = {}

    def context_copying_recover(*args: Any, **kwargs: Any) -> Any:
        if "context" not in saved:
            element = next(iter(model.mesh.elements.values()))
            saved["context"] = contextvars.copy_context()
            # The copy is made while the block's trusted operation scope
            # is live, so it really holds the operation token here.
            saved["accepted_while_live"] = saved["context"].run(
                _q4_trusted_operation_accepts, element)
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery_module, "recover_stress_result",
        context_copying_recover)
    stream = iter_linear_batches(model, [StaticCase("a", None)],
        batch_size=1, recovery_config=recovery)
    next(stream)
    stream.close()
    element = next(iter(model.mesh.elements.values()))
    assert saved["accepted_while_live"] is True
    # After the stream closed, the retained token in the copied context
    # is revoked and no longer accepts the element.
    assert saved["context"].run(_q4_trusted_operation_accepts, element) is False


def test_copied_context_cannot_retain_s3_operation_token_after_close(
    monkeypatch,
):
    from anyfesolver.e4_pl_s3_element import _s3_trusted_operation_accepts
    model = _s3_triangle()
    recovery = RecoveryConfig(element_ids=[1], include_displacements=False,
        include_reactions=False)
    original = recovery_module.recover_stress_result
    saved: dict[str, Any] = {}

    def context_copying_recover(*args: Any, **kwargs: Any) -> Any:
        if "context" not in saved:
            saved["context"] = contextvars.copy_context()
            saved["accepted_while_live"] = saved["context"].run(
                _s3_trusted_operation_accepts, model.mesh.elements[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery_module, "recover_stress_result",
        context_copying_recover)
    stream = iter_linear_batches(model, [StaticCase("a", None)],
        batch_size=1, recovery_config=recovery)
    next(stream)
    stream.close()
    assert saved["accepted_while_live"] is True
    assert saved["context"].run(
        _s3_trusted_operation_accepts, model.mesh.elements[1]) is False


@pytest.mark.parametrize("family", ["q4", "s3"])
def test_trusted_operation_token_revoked_on_error_and_cancel(family):
    element, scope, accepts = _trusted_operation_case(family)
    saved: dict[str, Any] = {}

    # An exception inside the scope revokes the token before exit
    # validation, so a context copied inside cannot retain it.
    with pytest.raises(RuntimeError, match="foreign failure"):
        with scope((element,)):
            saved["context"] = contextvars.copy_context()
            assert saved["context"].run(accepts, element) is True
            raise RuntimeError("foreign failure")
    assert saved["context"].run(accepts, element) is False

    # Cancellation (GeneratorExit at the yield point) revokes it as well.
    def run_inside_scope():
        with scope((element,)):
            yield contextvars.copy_context()

    generator = run_inside_scope()
    try:
        cancelled = next(generator)
        assert cancelled.run(accepts, element) is True
    finally:
        generator.close()
    assert cancelled.run(accepts, element) is False
    assert accepts(element) is False


@pytest.mark.parametrize("family", ["q4", "s3"])
def test_nested_trusted_scopes_revoke_only_their_own_token(family):
    element, scope, accepts = _trusted_operation_case(family)
    with scope((element,)):
        outer = contextvars.copy_context()
        assert outer.run(accepts, element) is True
        with scope((element,)):
            inner = contextvars.copy_context()
            assert inner.run(accepts, element) is True
        # The inner scope's exit revoked only the inner token; the outer
        # scope's token stays live until the outer scope exits.
        assert inner.run(accepts, element) is False
        assert outer.run(accepts, element) is True
    assert outer.run(accepts, element) is False
    assert accepts(element) is False
