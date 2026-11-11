from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import (AnalysisSession, BoundaryCondition, FixedSupport, LoadCase,
    StaticCase, StaticBatchError, iter_linear_batches, RecoveryConfig, ResourceConfig,
    CancellationToken, SolveCancelled, solve_linear, compute_constraint_force_diagnostics)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.recovery import ResourcePolicyError
from anyfesolver.linalg import backend_operation_counts


def _bar(supported=True):
    model = FEModel("stream")
    model.add_material("steel", 100.0, 0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel",
        {"area": 1.0, "Iy": 1e-3, "Iz": 1e-3, "J": 1e-3}))
    if supported:
        model.add_boundary_condition(FixedSupport("fixed", [1]))
        model.add_boundary_condition(BoundaryCondition("slide", [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    return model


def _case(index, force=1.0, balanced=False):
    load = LoadCase(str(index))
    load.add_nodal_load(2, [force, 0.0, 0.0, 0.0, 0.0, 0.0])
    if balanced:
        load.add_nodal_load(1, [-force, 0.0, 0.0, 0.0, 0.0, 0.0])
    return StaticCase(str(index), load, float(index)**2)


def test_arbitrary_cases_stream_bounded_and_reuse_actual_operator():
    model = _bar()
    cases = [_case(i, force) for i, force in enumerate([1, -3, 0.2, 5, -1])]
    expected = [solve_linear(model, c.load_case)[0] for c in cases]
    observed = []
    def inputs():
        for case in cases:
            observed.append(case.case_id)
            yield case
    with AnalysisSession(model) as session:
        before = backend_operation_counts()
        stream = iter_linear_batches(model, inputs(), batch_size=2, session=session)
        first = next(stream)
        assert observed == ["0", "1"]
        blocks = [first, *stream]
        assert [len(b.case_ids) for b in blocks] == [2, 2, 1]
        actual = np.column_stack([b.displacements for b in blocks])
        np.testing.assert_allclose(actual, np.column_stack(expected), rtol=1e-12, atol=1e-14)
        assert session.diagnostics()["counters"]["static_operator_builds"] == 1
        assert not session.closed
        assert all(r["solver_info"]["relative_backward_error_disposition"] == "CERTIFIED"
                   for b in blocks for r in b.case_results)
        after = backend_operation_counts()
        assert sum(after.values()) > sum(before.values())


def test_selected_output_and_prepared_reactions_match_ordinary(monkeypatch):
    model = _bar()
    cases = [_case(0, 2), _case(1, -1)]
    expected = [solve_linear(model, c.load_case)[0] for c in cases]
    expected_forces = [compute_constraint_force_diagnostics(model, u, c.load_case)
                       for u, c in zip(expected, cases)]
    dof = model.mesh.nodes[2].dofs[0]
    recovery = RecoveryConfig(node_ids=[1, 2], element_ids=[1], components=["axial_stress"])
    # Prepared reactions must not reassemble the stiffness or load per case.
    import anyfesolver.assembly as assembly
    def refuse(*args, **kwargs):
        raise AssertionError("prepared reaction route reassembled system")
    monkeypatch.setattr(assembly, "assemble_system", refuse)
    blocks = list(iter_linear_batches(model, cases, batch_size=1, output_dofs=[dof], recovery_config=recovery))
    for j, block in enumerate(blocks):
        np.testing.assert_array_equal(block.output_dofs, [dof])
        np.testing.assert_allclose(block.displacements[:, 0], expected[j][[dof]])
        item = block.case_results[0]
        assert set(item["element_stresses"][1]) == {"axial_stress"}
        for key in ("support_reactions", "mpc_slave_forces", "mpc_master_equivalent_forces"):
            assert set(item["constraint_forces"][key]) == set(expected_forces[j][key])
            for node in expected_forces[j][key]:
                np.testing.assert_allclose(item["constraint_forces"][key][node], expected_forces[j][key][node])


def test_free_free_balanced_multiple_rhs_matches_single_nullspace():
    model = _bar(False)
    cases = [_case(0, 1, True), _case(1, -2, True)]
    expected = [solve_linear(model, c.load_case)[0] for c in cases]
    with AnalysisSession(model) as session:
        blocks = list(iter_linear_batches(model, cases, batch_size=1, session=session))
        for j, block in enumerate(blocks):
            np.testing.assert_allclose(block.displacements[:, 0], expected[j], rtol=1e-11, atol=1e-13)
            assert block.case_results[0]["solver_info"]["gauge_residual_norm"] < 1e-12
        assert session.factorization_cache.diagnostics()["misses"] == 1


def test_free_free_incompatible_is_typed_and_explicit_allow_matches_single():
    model = _bar(False)
    case = _case(0)
    with pytest.raises(StaticBatchError) as error:
        list(iter_linear_batches(model, [case]))
    assert error.value.diagnostics["status"] == "incompatible_free_free_load"
    expected, _ = solve_linear(model, case.load_case, allow_unbalanced_free_free=True)
    block = next(iter_linear_batches(model, [case], allow_unbalanced_free_free=True))
    np.testing.assert_allclose(block.displacements[:, 0], expected, rtol=1e-11, atol=1e-13)
    assert block.case_results[0]["solver_info"]["warnings"]


@pytest.mark.parametrize("boundary", ["input", "yield", "callback"])
def test_model_mutation_at_foreign_boundaries_terminates_stream(boundary):
    model = _bar()
    def mutate():
        model.mesh.nodes[2].x = 2.0
    def inputs():
        yield _case(0)
        if boundary == "input":
            mutate()
        yield _case(1)
    callback = (lambda event: mutate()) if boundary == "callback" else None
    stream = iter_linear_batches(model, inputs(), batch_size=1, progress_callback=callback)
    if boundary == "callback":
        with pytest.raises(StaticBatchError, match="model changed"):
            next(stream)
    else:
        next(stream)
        if boundary == "yield":
            mutate()
        with pytest.raises(StaticBatchError, match="model changed"):
            next(stream)


def test_cancellation_and_owned_session_close(monkeypatch):
    model = _bar()
    created = []
    import anyfesolver.static_batch as module
    original = module.AnalysisSession
    def make(model):
        session = original(model)
        created.append(session)
        return session
    monkeypatch.setattr(module, "AnalysisSession", make)
    token = CancellationToken()
    stream = iter_linear_batches(model, [_case(0), _case(1)], batch_size=1, cancellation_token=token)
    next(stream)
    token.cancel("test")
    with pytest.raises(SolveCancelled):
        next(stream)
    assert created[0].closed
    stream = iter_linear_batches(model, [_case(0), _case(1)], batch_size=1)
    next(stream)
    stream.close()
    assert created[1].closed


def test_memory_refuses_one_and_adapts_before_factorization():
    model = _bar()
    before = backend_operation_counts()
    with pytest.raises(ResourcePolicyError):
        next(iter_linear_batches(model, [_case(0)], resource_config=ResourceConfig(memory_limit_bytes=1)))
    assert backend_operation_counts() == before
    probe = next(iter_linear_batches(model, [_case(0)], batch_size=1))
    limit = probe.diagnostics["estimated_block_bytes"]
    blocks = list(iter_linear_batches(model, [_case(0), _case(1)], resource_config=ResourceConfig(memory_limit_bytes=limit)))
    assert [b.diagnostics["effective_batch_size"] for b in blocks] == [1, 1]


@pytest.mark.parametrize("edit", ["cross_section", "cached_area", "material", "node_dofs", "class"])
def test_direct_structural_edits_are_rejected(edit, monkeypatch):
    model = _bar()
    stream = iter_linear_batches(model, [_case(0), _case(1)], batch_size=1)
    next(stream)
    if edit == "cross_section":
        model.mesh.elements[1].cross_section["area"] = 2.0
    elif edit == "cached_area":
        model.mesh.elements[1]._A = 2.0
    elif edit == "material":
        model.materials["steel"] = _bar().materials["steel"]
    elif edit == "node_dofs":
        model.mesh.nodes[2].dofs.reverse()
    else:
        original = BeamElement.compute_stiffness_matrix
        def replacement(self, mesh, material):
            return 2 * original(self, mesh, material)
        monkeypatch.setattr(BeamElement, "compute_stiffness_matrix", replacement)
    with pytest.raises(StaticBatchError, match="model changed"):
        next(stream)


def test_load_only_registration_is_allowed_and_empty_input_is_lazy():
    model = _bar(False)
    before = backend_operation_counts()
    assert list(iter_linear_batches(model, [], resource_config=ResourceConfig(memory_limit_bytes=1))) == []
    assert backend_operation_counts() == before
    model = _bar()
    def cases():
        for index in range(3):
            case = _case(index, index + 1)
            model.add_load_case("registered" + str(index))
            yield case
    blocks = list(iter_linear_batches(model, cases(), batch_size=1))
    assert len(blocks) == 3


def test_bad_load_identity_is_annotated_once_and_primary_failure_survives_close():
    class Bad(LoadCase):
        calls = 0
        def get_load_vector(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("bad physical load")
    bad = Bad("bad")
    class Source:
        position = 0
        closed = False
        def __iter__(self):
            return self
        def __next__(self):
            self.position += 1
            if self.position == 1:
                return _case(0)
            if self.position == 2:
                return StaticCase("bad", bad)
            raise StopIteration
        def close(self):
            self.closed = True
            raise RuntimeError("secondary cleanup failure")
    source = Source()
    with pytest.raises(StaticBatchError) as error:
        list(iter_linear_batches(_bar(), source))
    assert error.value.diagnostics["case_id"] == "bad"
    assert error.value.diagnostics["input_index"] == 1
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "bad physical load"
    assert bad.calls == 1 and source.closed


def test_time_conversion_mutation_is_rejected_and_plain_time_is_retained():
    model = _bar()
    class Time:
        def __float__(self):
            model.mesh.nodes[2].x = 2.0
            return 1.0
    with pytest.raises(StaticBatchError):
        next(iter_linear_batches(model, [StaticCase("time", _case(0).load_case, Time())]))
    block = next(iter_linear_batches(_bar(), [StaticCase("time", _case(0).load_case, np.float64(1.5))]))
    assert type(block.times[0]) is float


def test_single_session_free_free_and_collecting_nullspace_share_augmented_factor():
    from anyfesolver import solve_linear_many
    model = _bar(False)
    cases = [_case(0, 1, True), _case(1, -2, True)]
    with AnalysisSession(model) as session:
        expected = [solve_linear(model, c.load_case, session=session)[0] for c in cases]
        actual, info = solve_linear_many(model, [c.load_case for c in cases], constraint_mode="nullspace", session=session)
        np.testing.assert_allclose(actual, np.column_stack(expected), rtol=1e-12, atol=1e-14)
        assert session.factorization_cache.diagnostics()["misses"] == 1
        assert info["status"] == "converged"
        operator = session.static_operator_plan()
        for array in (operator.scaling, operator.rigid_modes, operator.matrix.data):
            with pytest.raises(ValueError):
                array.setflags(write=True)


def test_affine_mpc_selected_outputs_and_prepared_force_buckets():
    model = _bar()
    model.boundary_conditions[0].dof_constraints["ux"] = 0.02
    model.add_node(3, 2.0, 0.0, 0.0)
    model.add_boundary_condition(BoundaryCondition("slave_other", [3],
        {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    model.add_constraint_equation(terms=((model.mesh.nodes[3].dofs[0], 1.0),
        (model.mesh.nodes[2].dofs[0], -1.0)), rhs=0.01, source_id="affine", dependent_dof=model.mesh.nodes[3].dofs[0])
    case = _case(0, 2)
    case.load_case.add_nodal_load(3, [3, 0, 0, 0, 0, 0])
    expected, _ = solve_linear(model, case.load_case)
    forces = compute_constraint_force_diagnostics(model, expected, case.load_case)
    recovery = RecoveryConfig(node_ids=[1, 3], include_stresses=False, include_displacements=False)
    dofs = [model.mesh.nodes[2].dofs[0], model.mesh.nodes[3].dofs[0]]
    block = next(iter_linear_batches(model, [case], output_dofs=dofs, recovery_config=recovery))
    np.testing.assert_allclose(block.displacements[:, 0], expected[dofs])
    item = block.case_results[0]
    assert item["reaction_node_ids"] == (1, 3)
    assert "residual" not in item["constraint_forces"]
    for bucket in ("support_reactions", "mpc_slave_forces", "mpc_master_equivalent_forces"):
        for node, force in item["constraint_forces"][bucket].items():
            np.testing.assert_allclose(force, forces[bucket][node])


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_reused_mutable_builtin_load_is_owned_at_each_yield(batch_size):
    model = _bar()
    loads = [_case(index, force).load_case for index, force in enumerate([1.0, -2.0, 3.0])]
    for index, load in enumerate(loads):
        load.set_acceleration(index * 0.25, 0.0, 0.0)
    expected = np.column_stack([solve_linear(model, load)[0] for load in loads])
    shared = LoadCase("shared")
    def cases():
        for index, load in enumerate(loads):
            shared.nodal_loads = load.nodal_loads
            shared.gravity = load.gravity
            yield StaticCase(str(index), shared, index)
    actual = np.column_stack([block.displacements for block in iter_linear_batches(model, cases(), batch_size=batch_size)])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_reused_foreign_load_is_assembled_before_next_input():
    class Foreign(LoadCase):
        value = 1.0
        def get_load_vector(self, mesh, manager, *args, **kwargs):
            vector = np.zeros(manager.total_dofs)
            vector[mesh.nodes[2].dofs[0]] = self.value
            return vector
    load = Foreign("foreign")
    def cases():
        for index, value in enumerate([1.0, -2.0, 3.0]):
            load.value = value
            yield StaticCase(str(index), load)
    blocks = list(iter_linear_batches(_bar(), cases(), batch_size=8))
    np.testing.assert_allclose(blocks[0].displacements[6], [0.01, -0.02, 0.03])


@pytest.mark.parametrize("kind", ["q4", "qualified_s3"])
def test_compiled_recovery_is_used_by_stream_and_reused(kind, monkeypatch):
    from anyfesolver import generate_simple_panel_mesh
    from anyfesolver.recovery import _compute_one_element_stress
    import anyfesolver.recovery as module
    import numba
    workers = 2 if module.JIT_ENABLED and numba.config.NUMBA_NUM_THREADS >= 2 else 1
    observed_workers = []
    kernel = module.recover_isotropic_s4
    def observe_kernel(*args, **kwargs):
        observed_workers.append(numba.get_num_threads())
        return kernel(*args, **kwargs)
    monkeypatch.setattr(module, "recover_isotropic_s4", observe_kernel)
    if kind == "q4":
        model = generate_simple_panel_mesh(1.0, 1.0, 0.01, num_divisions_x=10, num_divisions_y=10)
    else:
        from test_e4_pl_s3_reference_batch import _build_model
        model = _build_model(128, include_q4=True)
    # A prescribed affine displacement field exercises the public stream's
    # actual recovery path without introducing a new mechanics benchmark.
    model.boundary_conditions.clear()
    for node_id, node in model.mesh.nodes.items():
        model.add_boundary_condition(BoundaryCondition("affine" + str(node_id), [node_id],
            {"ux": node.x * 1e-5, "uy": node.y * -2e-6, "uz": 0.0,
             "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    expected, _ = solve_linear(model, LoadCase("zero"))
    scalar = {element_id: _compute_one_element_stress(model, expected, element_id, return_global=False)[1]
              for element_id in model.mesh.elements}
    recovery = RecoveryConfig(include_displacements=False, include_reactions=False,
        components=["von_mises"])
    blocks = list(iter_linear_batches(model, [StaticCase("one", None), StaticCase("two", None)],
        batch_size=1, output_dofs=[0], recovery_config=recovery,
        resource_config=ResourceConfig(solver_threads=1, assembly_threads=workers, recovery_threads=workers)))
    for block in blocks:
        for element_id, actual in block.case_results[0]["element_stresses"].items():
            np.testing.assert_allclose(actual["von_mises"], scalar[element_id]["von_mises"], rtol=1e-12, atol=1e-12)
    first = blocks[0].case_results[0]["solver_info"]["recovery"]["metadata"]
    second = blocks[1].case_results[0]["solver_info"]["recovery"]["metadata"]
    assert second["plan_reused"]
    if kind == "q4" and module.JIT_ENABLED:
        assert first["compiled_batch_count"] == 1
        assert observed_workers == [workers, workers]
    if kind == "qualified_s3":
        assert first["qualified_s3_reference_batch_count"] == 1


@pytest.mark.parametrize("fail_load", [False, True])
def test_phase_thread_policy_restores_before_foreign_boundaries(monkeypatch, fail_load):
    import numba
    import anyfesolver.jit_compiler as jit
    import anyfesolver.matrix_assembly as assembly
    from anyfesolver.threading_policy import current_solver_threads
    from threadpoolctl import threadpool_info
    initial = numba.get_num_threads()
    workers = 2 if jit.JIT_ENABLED and numba.config.NUMBA_NUM_THREADS >= 2 else 1
    numba.set_num_threads(1)
    observed = []
    original = assembly._assemble_load_matrix_annotated
    def load(*args, **kwargs):
        observed.append("load")
        if jit.JIT_ENABLED:
            assert numba.get_num_threads() == workers
        assert current_solver_threads() == 1
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
        if fail_load:
            raise RuntimeError("load failure inside scoped phase")
        return original(*args, **kwargs)
    monkeypatch.setattr(assembly, "_assemble_load_matrix_annotated", load)
    def outside():
        assert numba.get_num_threads() == 1
        assert current_solver_threads() is None
    def inputs():
        try:
            outside()
            yield _case(0)
            outside()
            yield _case(1)
        finally:
            outside()
            observed.append("closed")
    try:
        stream = iter_linear_batches(_bar(), inputs(), batch_size=1,
            resource_config=ResourceConfig(solver_threads=1, assembly_threads=workers),
            progress_callback=lambda event: outside())
        if fail_load:
            with pytest.raises(RuntimeError, match="load failure"):
                next(stream)
        else:
            next(stream)
            outside()
            stream.close()
        outside()
        assert observed == ["load", "closed"]
    finally:
        numba.set_num_threads(initial)


def test_uncertified_backend_result_never_yields_success(monkeypatch):
    model = _bar()
    with AnalysisSession(model) as session:
        plan = session.static_operator_plan()
        original = type(plan.handle).solve_many
        def corrupt_second_column(handle, rhs):
            result = original(handle, rhs)
            result[:, 1] = np.nan
            return result
        monkeypatch.setattr(type(plan.handle), "solve_many", corrupt_second_column)
        with pytest.raises(StaticBatchError) as error:
            next(iter_linear_batches(model, [StaticCase("repeat", _case(i).load_case)
                for i in range(2)], session=session))
        assert error.value.diagnostics["relative_backward_error_disposition"] == "NONFINITE_OR_OVERFLOW"
        assert error.value.diagnostics["input_index"] == 1
        assert error.value.diagnostics["case_id"] == "repeat"


def test_collecting_nullspace_honors_external_cache_and_refuses_conflicting_session():
    from anyfesolver import solve_linear_many
    from anyfesolver.linalg import FactorizationCache
    model = _bar(False)
    cache = FactorizationCache(name="caller nullspace", max_entries=2)
    try:
        values, info = solve_linear_many(model, [_case(0, balanced=True).load_case],
            constraint_mode="nullspace", factorization_cache=cache)
        assert values.shape == (12, 1)
        assert cache.diagnostics()["misses"] == 1
        assert cache.diagnostics()["entries"] == 1
        assert info["factorization_cache"]["entries"] == 1
        with AnalysisSession(model, factorization_cache=cache) as session:
            solve_linear_many(model, [_case(1, balanced=True).load_case],
                constraint_mode="nullspace", factorization_cache=cache, session=session)
            with pytest.raises(ValueError, match="match the supplied session"):
                solve_linear_many(model, [], constraint_mode="nullspace", session=session,
                    factorization_cache=FactorizationCache(name="conflict"))
        assert cache.diagnostics()["entries"] > 0
    finally:
        cache.clear()


def test_default_v2d_recovery_preserves_public_scalar_outputs():
    from anyfesolver import create_shell_element
    from anyfesolver.recovery import recover_stress_result
    model = FEModel("v2d_stream")
    model.add_material("steel", 100.0, 0.3)
    for node_id, coords in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0)], 1):
        model.add_node(node_id, *coords)
        model.add_boundary_condition(BoundaryCondition(str(node_id), [node_id],
            {"ux": coords[0] * 1e-5, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0}))
    model.add_element(1, create_shell_element(1, [1, 2, 3], "steel", thickness=0.01, reference_normal=[0, 0, 1]))
    u, _ = solve_linear(model, LoadCase("zero"))
    config = RecoveryConfig(element_ids=[1], components=["von_mises", "xx_top"], include_reactions=False)
    expected = recover_stress_result(model, u, config).element_stresses
    block = next(iter_linear_batches(model, [StaticCase("v2d", None)], recovery_config=config))
    assert set(block.case_results[0]["element_stresses"][1]) == set(expected[1])
    for name, values in expected[1].items():
        np.testing.assert_array_equal(block.case_results[0]["element_stresses"][1][name], values)


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("stage", ["stress", "reaction"])
def test_recovery_failure_identifies_case_and_preserves_cancellation(monkeypatch, cancel, stage):
    import anyfesolver.recovery as recovery
    import anyfesolver.assembly as assembly
    target, name = ((recovery, "recover_stress_result") if stage == "stress" else
                    (assembly, "_compute_constraint_force_diagnostics_under_lease"))
    original = getattr(target, name)
    observed = []
    failure = SolveCancelled("recovery cancelled") if cancel else RuntimeError("foreign recovery failure")
    def recover(*args, **kwargs):
        observed.append(len(observed))
        if len(observed) == 2:
            raise failure
        return original(*args, **kwargs)
    monkeypatch.setattr(target, name, recover)
    with pytest.raises(SolveCancelled if cancel else StaticBatchError) as caught:
        next(iter_linear_batches(_bar(), [_case(0), _case(1)],
            recovery_config=RecoveryConfig(include_stresses=stage == "stress", include_reactions=stage == "reaction")))
    assert observed == [0, 1]
    if cancel:
        assert caught.value is failure
    else:
        assert caught.value.__cause__ is failure
        assert caught.value.diagnostics["case_id"] == "1"
        assert caught.value.diagnostics["input_index"] == 1
        assert caught.value.diagnostics["stage"] == stage + " recovery"


def test_session_cleanup_cannot_replace_primary_failure(monkeypatch):
    original = AnalysisSession.close
    def close(session):
        original(session)
        raise RuntimeError("owned session close failure")
    monkeypatch.setattr(AnalysisSession, "close", close)
    with pytest.raises(StaticBatchError, match="time must be finite") as caught:
        next(iter_linear_batches(_bar(), [StaticCase("invalid", None, float("inf"))]))
    assert any("owned session close failure" in note for note in caught.value.__notes__)


def test_recovery_scope_entries_scale_with_blocks_not_cases(monkeypatch):
    import contextlib
    import numba
    import anyfesolver.assembly as assembly_module
    import anyfesolver.jit_compiler as jit
    import anyfesolver.matrix_assembly as matrix_assembly
    import anyfesolver.recovery as recovery_module
    import anyfesolver.threading_policy as policy
    from anyfesolver.threading_policy import current_solver_threads
    from threadpoolctl import threadpool_info

    workers = 2 if jit.JIT_ENABLED and numba.config.NUMBA_NUM_THREADS >= 2 else 1
    resources = ResourceConfig(solver_threads=1, assembly_threads=workers, recovery_threads=workers)
    recovery = RecoveryConfig(node_ids=[1, 2], element_ids=[1],
        components=["axial_stress"], include_reactions=True)
    state = {"recovery_calls": 0, "recovery_scopes": 0}

    original_scope = policy.solver_thread_scope

    @contextlib.contextmanager
    def counting_scope(resource_config):
        with original_scope(resource_config) as report:
            entered_calls = state["recovery_calls"]
            try:
                yield report
            finally:
                if state["recovery_calls"] > entered_calls:
                    state["recovery_scopes"] += 1

    monkeypatch.setattr(policy, "solver_thread_scope", counting_scope)

    def observe_recovery():
        state["recovery_calls"] += 1
        assert current_solver_threads() == 1
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())

    original_stress = recovery_module.recover_stress_result
    def observed_stress(*args, **kwargs):
        observe_recovery()
        return original_stress(*args, **kwargs)
    monkeypatch.setattr(recovery_module, "recover_stress_result", observed_stress)

    original_forces = assembly_module._compute_constraint_force_diagnostics_under_lease
    def observed_forces(*args, **kwargs):
        observe_recovery()
        return original_forces(*args, **kwargs)
    monkeypatch.setattr(assembly_module, "_compute_constraint_force_diagnostics_under_lease", observed_forces)

    original_assembly = matrix_assembly._assemble_load_matrix_annotated
    def observed_assembly(*args, **kwargs):
        if jit.JIT_ENABLED:
            assert numba.get_num_threads() == workers
        assert current_solver_threads() == 1
        return original_assembly(*args, **kwargs)
    monkeypatch.setattr(matrix_assembly, "_assemble_load_matrix_annotated", observed_assembly)

    model = _bar()

    def run(case_count, batch_size):
        state["recovery_calls"] = 0
        state["recovery_scopes"] = 0
        forces = [1.0, -2.0, 3.0, -1.0, 2.0, -3.0]
        cases = [_case(index, force) for index, force in enumerate(forces[:case_count])]
        blocks = list(iter_linear_batches(model, cases, batch_size=batch_size,
            recovery_config=recovery, resource_config=resources))
        for block in blocks:
            assert current_solver_threads() is None
        return (state["recovery_scopes"], state["recovery_calls"], len(blocks))

    # Solver scopes that contain recovery work follow blocks, not cases: one
    # block of six cases pays one recovery scope, and tripling the block count
    # triples the scopes while per-case recovery calls stay case-proportional.
    assert run(6, 6) == (1, 12, 1)
    assert run(3, 6) == (1, 6, 1)
    assert run(6, 2) == (3, 12, 3)

    # A recovery failure inside the amortized block scope still restores the
    # solver lease before the typed case error reaches the consumer.
    stress_calls = {"count": 0}
    def failing_stress(*args, **kwargs):
        stress_calls["count"] += 1
        if stress_calls["count"] == 2:
            raise RuntimeError("scope restoration failure probe")
        return original_stress(*args, **kwargs)
    monkeypatch.setattr(recovery_module, "recover_stress_result", failing_stress)
    with pytest.raises(StaticBatchError) as error:
        list(iter_linear_batches(model, [_case(0), _case(1)], batch_size=2,
            recovery_config=recovery, resource_config=resources))
    assert error.value.diagnostics["stage"] == "stress recovery"
    assert error.value.diagnostics["input_index"] == 1
    assert current_solver_threads() is None
