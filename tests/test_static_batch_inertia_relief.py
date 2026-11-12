"""Production batch integration: physical balance, reuse and refusal boundaries."""
import json

import numpy as np
import pytest

from anyfesolver import (AnalysisSession, CancellationToken, CoupledBeamShellElement,
    BeamElement, FixedSupport, LoadCase, RecoveryConfig, ResourceConfig,
    SolveCancelled, StaticBatchError, StaticCase, iter_linear_batches,
    solve_inertia_relief, solve_linear)
from anyfesolver.matrix_assembly import assemble_mass_matrix, assemble_stiffness_matrix
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.recovery import ResourcePolicyError
from test_inertia_relief import bar, rigid, load_vector
from test_inertia_relief_integration import panel


def cases(model):
    end = max(model.mesh.nodes)
    result = []
    for i, force in enumerate((1., -2., 0., .5, 3.)):
        load = LoadCase(str(i))
        load.add_nodal_load(end, forces=[force, .2*force, -.1*force],
                            moments=[.03*force, -.05*force, .01*force])
        result.append(StaticCase(str(i), load, float(i*i)))
    return result


def collect(model, inputs, **kwargs):
    return list(iter_linear_batches(model, inputs, inertia_relief=True, **kwargs))


def assert_stress_values(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual: assert_stress_values(actual[key], expected[key])
    elif np.asarray(actual).dtype.kind in "OUS":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("consistent", [False, True])
def test_multiple_rhs_blocks_reuse_preparation_and_match_scalar(consistent, monkeypatch):
    import anyfesolver.inertia_relief as module
    model = bar(consistent=consistent)
    model.mesh.point_masses[2] = .7
    inputs = cases(model)
    expected = [solve_inertia_relief(model, c.load_case) for c in inputs]
    checks = []
    original = module._mechanism_check
    def record(*args, **kwargs):
        checks.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "_mechanism_check", record)
    def no_scalar(*args, **kwargs):
        pytest.fail("batch fell back to the scalar public solve")
    monkeypatch.setattr(module, "solve_inertia_relief", no_scalar)
    with AnalysisSession(model) as session:
        plan = session.static_operator_plan(constraint_mode="nullspace")
        solves = []
        original_solve = type(plan.handle).solve_many
        def solve_many(self, rhs):
            solves.append(rhs.shape)
            return original_solve(self, rhs)
        monkeypatch.setattr(type(plan.handle), "solve_many", solve_many)
        blocks = collect(model, inputs, batch_size=2, session=session, constraint_mode="nullspace")
        assert [len(b.case_ids) for b in blocks] == [2, 2, 1]
        assert [i for b in blocks for i in b.input_indices] == list(range(5))
        actual = np.column_stack([b.displacements for b in blocks])
        np.testing.assert_allclose(actual, np.column_stack([r.displacements for r in expected]), rtol=1e-8, atol=1e-13)
        results = [r for b in blocks for r in b.case_results]
        for item, oracle in zip(results, expected):
            summary = item["inertia_relief"]
            assert item["analysis_type"] == summary["analysis_type"] == "inertia_relief"
            for name in ("acceleration_at_center_of_mass", "angular_acceleration"):
                np.testing.assert_allclose(summary[name], getattr(oracle, name), rtol=1e-10, atol=1e-12)
            for name in ("force_balance_error", "moment_balance_error", "elastic_backward_error", "gauge_error"):
                assert summary["diagnostics"][name] <= 1e-10
            assert item["reactions"] == {}
            json.dumps(summary, allow_nan=False)
        assert len(checks) == 1
        assert [shape[1] for shape in solves] == [2, 2, 1]
        assert session.diagnostics()["counters"]["inertia_relief_builds"] == 1
        assert session.diagnostics()["counters"]["static_operator_builds"] == 1
        assert session.factorization_cache.diagnostics()["misses"] == 1
        assert not session.closed


def test_selected_output_rigid_acceleration_and_zero_columns_are_analytical():
    model = bar()
    M, _ = assemble_mass_matrix(model)
    acceleration = np.array([.3, -.7, .1, .2, .4, -.5])
    load = load_vector(model, M @ (rigid(model, np.array([1., 0., 0.])) @ acceleration))
    inputs = [StaticCase("rigid", load), StaticCase("zero", None)]
    block = collect(model, inputs, output_dofs=[0, 6], batch_size=2)[0]
    assert not block.diagnostics["full_reconstruction"]
    assert block.displacements.shape == (2, 2)
    assert np.linalg.norm(block.displacements) < 1e-13
    np.testing.assert_allclose(block.case_results[0]["inertia_relief"]["angular_acceleration"], acceleration[3:], rtol=1e-10)
    np.testing.assert_array_equal(block.case_results[1]["inertia_relief"]["acceleration_at_reference"], [0., 0., 0.])
    for item in block.case_results:
        assert not any(key in item["inertia_relief"] for key in ("applied_loads", "inertial_loads", "effective_loads", "rigid_acceleration"))


def test_scalar_and_batch_share_plan_mechanism_and_factorization_across_reference_changes(monkeypatch):
    import anyfesolver.inertia_relief as module
    model = bar()
    case = cases(model)[0]
    with AnalysisSession(model) as session:
        initial = solve_inertia_relief(model, case.load_case, session=session)
        assert initial.converged
        def forbidden(*args, **kwargs):
            pytest.fail("unchanged physical operator was re-certified")
        monkeypatch.setattr(module, "_mechanism_check", forbidden)
        point = np.array([3., -2., 1.])
        block = collect(model, [case], session=session, reference_point=point)[0]
        summary = block.case_results[0]["inertia_relief"]
        np.testing.assert_allclose(summary["acceleration_at_reference"], initial.acceleration_at_center_of_mass + np.cross(initial.angular_acceleration, point-initial.center_of_mass), rtol=1e-10)
        np.testing.assert_allclose(block.displacements[:, 0], initial.displacements, rtol=1e-8, atol=1e-13)
        assert session.factorization_cache.diagnostics()["misses"] == 1
        assert session.diagnostics()["counters"]["inertia_relief_builds"] == 1


@pytest.mark.parametrize("eccentric", [False, True])
def test_shell_and_eccentric_slave_load_dense_reference_effective_recovery(eccentric):
    model = panel()
    if eccentric:
        model.add_node(5, 0., 0., .2)
        model.add_node(6, 2., 0., .2)
        section = {"area": .03, "Iy": .0001, "Iz": .0002, "J": .0001, "consistent_mass": True}
        model.add_element(2, BeamElement(2, [5, 6], "steel", section))
        model.add_element(3, CoupledBeamShellElement(3, 5, 1, "steel"))
        model.add_element(4, CoupledBeamShellElement(4, 6, 2, "steel"))
    inputs = cases(model)[:2]
    expected = [solve_inertia_relief(model, c.load_case) for c in inputs]
    config = RecoveryConfig(include_displacements=True, include_stresses=True, include_reactions=True)
    block = collect(model, inputs, recovery_config=config, batch_size=2)[0]
    K, _ = assemble_stiffness_matrix(model)
    M, _ = assemble_mass_matrix(model)
    R = rigid(model, np.zeros(3))
    for j, oracle in enumerate(expected):
        a = np.linalg.solve(R.T @ (M @ R), R.T @ oracle.applied_loads)
        Fe = oracle.applied_loads-M @ (R @ a)
        Kr, Fr, T, _, ind, _ = build_constraint_transformation(K, Fe, model)
        Q, _ = np.linalg.qr(R[ind, :])
        B = np.block([[Kr.toarray(), Q], [Q.T, np.zeros((6, 6))]])
        q = np.linalg.solve(B, np.r_[Fr, np.zeros(6)])[:len(ind)]
        np.testing.assert_allclose(block.displacements[:, j], T @ q, rtol=1e-8, atol=1e-11)
        item = block.case_results[j]
        assert item["element_stresses"]
        stress_oracle = oracle.recover(recovery_config=config).element_stresses
        for eid, fields in item["element_stresses"].items():
            for name, value in fields.items():
                assert_stress_values(value, stress_oracle[eid][name])
        assert item["reactions"] == {}
        for bucket in ("support_reactions", "mpc_slave_forces", "mpc_master_equivalent_forces"):
            assert item["constraint_forces"][bucket].keys() == oracle.constraint_forces[bucket].keys()
            for node in oracle.constraint_forces[bucket]:
                np.testing.assert_allclose(item["constraint_forces"][bucket][node], oracle.constraint_forces[bucket][node], atol=1e-10)
        np.testing.assert_allclose(T.T @ (K @ block.displacements[:, j]-Fe), 0., atol=1e-10)


def test_direct_point_mass_and_revision_changes_invalidate_session_plan():
    model = bar()
    case = cases(model)[0]
    with AnalysisSession(model) as session:
        initial = collect(model, [case], session=session)[0]
        model.mesh.point_masses[2] = 3.
        changed = collect(model, [case], session=session)[0]
        oracle = solve_inertia_relief(model, case.load_case)
        np.testing.assert_allclose(changed.displacements[:, 0], oracle.displacements, rtol=1e-8, atol=1e-12)
        assert changed.case_results[0]["inertia_relief"]["total_mass"] == pytest.approx(5.)
        assert initial.case_results[0]["inertia_relief"]["total_mass"] == pytest.approx(2.)
        assert session.diagnostics()["counters"]["inertia_relief_builds"] == 2
        model.materials["steel"].elastic_modulus *= 2.
        model.bump_revision("material")
        updated = collect(model, [case], session=session)[0]
        fresh = solve_inertia_relief(model, case.load_case)
        np.testing.assert_allclose(updated.displacements[:, 0], fresh.displacements, rtol=1e-8, atol=1e-12)
        assert session.diagnostics()["counters"]["inertia_relief_builds"] == 3


@pytest.mark.parametrize("mutation", ["consumer", "callback", "input"])
def test_stream_mutation_remains_rejected(mutation):
    model = bar()
    inputs = cases(model)
    def change(): model.mesh.point_masses[1] = .4
    def source():
        yield inputs[0]
        if mutation == "input": change()
        yield inputs[1]
    def callback(event):
        if mutation == "callback" and event["stage"] == "static_batch.complete": change()
    stream = iter_linear_batches(model, source(), inertia_relief=True, batch_size=1, progress_callback=callback)
    try:
        if mutation == "callback":
            with pytest.raises(StaticBatchError, match="model changed"): next(stream)
        else:
            next(stream)
            if mutation == "consumer": change()
            with pytest.raises(StaticBatchError, match="model changed"): next(stream)
    finally: stream.close()


@pytest.mark.parametrize("stage", ["static_batch.start", "inertia_relief.mechanism_iteration", "static_batch.before_solve", "static_batch.before_yield"])
def test_cancellation_propagates_and_owned_session_closes(stage, monkeypatch):
    import anyfesolver.static_batch as module
    model = bar()
    made = []
    original = module.AnalysisSession
    def create(*args, **kwargs):
        session = original(*args, **kwargs); made.append(session); return session
    monkeypatch.setattr(module, "AnalysisSession", create)
    class Token(CancellationToken):
        def raise_if_cancelled(self, at=""):
            if at == stage: self.cancel("test boundary")
            super().raise_if_cancelled(at)
    with pytest.raises(SolveCancelled): collect(model, cases(model), cancellation_token=Token())
    assert all(session.closed for session in made)


def test_numerical_non_success_never_yields_and_closes_session(monkeypatch):
    import anyfesolver.inertia_relief as module
    model = bar()
    monkeypatch.setattr(module, "_mechanism_check", lambda *args: {"status": "unresolved", "error": "test nonconvergence"})
    with AnalysisSession(model) as session:
        with pytest.raises(StaticBatchError) as error:
            collect(model, cases(model), session=session)
        assert error.value.diagnostics["status"] == "unresolved_elastic_stiffness"
        assert error.value.diagnostics["scope"] == "model"
        assert not session.closed


def test_corrupt_column_is_refused_with_identity_before_yield(monkeypatch):
    model = bar()
    with AnalysisSession(model) as session:
        plan = session.static_operator_plan(constraint_mode="nullspace")
        original = type(plan.handle).solve_many
        def corrupt(self, rhs):
            solved = original(self, rhs)
            solved[0, 1] += 1.
            return solved
        monkeypatch.setattr(type(plan.handle), "solve_many", corrupt)
        with pytest.raises(StaticBatchError) as error:
            collect(model, cases(model)[:2], session=session)
        assert error.value.diagnostics["case_id"] == "1"
        assert error.value.diagnostics["input_index"] == 1


@pytest.mark.parametrize("kind", ["support", "added_mass", "follower", "custom_load"])
def test_unsupported_inputs_keep_case_identity(kind):
    model = bar()
    inputs = cases(model)
    if kind == "support": model.add_boundary_condition(FixedSupport("fixed", [1]))
    elif kind == "added_mass": inputs[1].load_case.added_node_masses[1] = 1.
    elif kind == "follower": inputs[1].load_case.follower_pressure = True
    else:
        class Custom(LoadCase): pass
        inputs[1] = StaticCase("1", Custom("custom"))
    with pytest.raises(StaticBatchError) as error:
        collect(model, inputs, batch_size=2)
    expected = 0 if kind == "support" else 1
    assert error.value.diagnostics["input_index"] == expected
    assert error.value.diagnostics["status"] == "invalid_input"
    assert error.value.diagnostics["code"] in {"unsupported_constraints", "unsupported_load"}


@pytest.mark.parametrize("kwargs", [{"allow_unbalanced_free_free": True}, {"constraint_mode": "transformation"}, {"reference_point": [1., np.nan, 2.]}])
def test_conflicting_options_fail_closed(kwargs):
    model = bar()
    with pytest.raises(ValueError): collect(model, cases(model), **kwargs)


def test_reference_requires_relief_and_small_memory_limit_is_honored():
    model = bar()
    with pytest.raises(ValueError, match="reference_point"):
        list(iter_linear_batches(model, cases(model), reference_point=[0., 0., 0.]))
    with pytest.raises(ResourcePolicyError):
        collect(model, cases(model), resource_config=ResourceConfig(memory_limit_bytes=1))


def test_nullspace_default_refusal_and_legacy_opt_in_are_unchanged():
    model = bar()
    case = cases(model)[0]
    with pytest.raises(StaticBatchError) as error:
        list(iter_linear_batches(model, [case], constraint_mode="nullspace"))
    assert error.value.diagnostics["status"] == "incompatible_free_free_load"
    expected, _ = solve_linear(model, case.load_case, constraint_mode="nullspace", allow_unbalanced_free_free=True)
    block = next(iter_linear_batches(model, [case], constraint_mode="nullspace", allow_unbalanced_free_free=True))
    np.testing.assert_allclose(block.displacements[:, 0], expected, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("mode", ["auto", "nullspace"])
def test_collecting_fast_path_relief_and_nullspace_auto(mode):
    from anyfesolver import solve_linear_many
    model = bar()
    inputs = cases(model)[:3]
    with AnalysisSession(model) as session:
        expected = collect(model, inputs, session=session)[0]
        actual, info = solve_linear_many(model, [c.load_case for c in inputs],
            session=session, inertia_relief=True, constraint_mode=mode)
        np.testing.assert_allclose(actual, expected.displacements, rtol=1e-8, atol=1e-12)
        assert info["analysis_type"] == "inertia_relief"
        assert info["result_case"]["analysis_case"]["analysis_type"] == "inertia_relief"
        assert session.factorization_cache.diagnostics()["misses"] == 1
        assert all(c["inertia_relief"]["status"] == "converged" for c in info["case_results"])
    K, _ = assemble_stiffness_matrix(model)
    displacement = np.zeros(K.shape[0]); displacement[-6] = 1.e-4
    balanced = load_vector(model, K @ displacement)
    expected, _ = solve_linear(model, balanced)
    actual, info = solve_linear_many(model, [balanced, balanced], constraint_mode=mode)
    np.testing.assert_allclose(actual, np.column_stack([expected, expected]), rtol=1e-8, atol=1e-12)
    with pytest.raises(StaticBatchError): solve_linear_many(model, [inputs[0].load_case], constraint_mode=mode)


@pytest.mark.parametrize("mass_policy,load_policy,ratio", [("delete_only", "activity", .375), ("activity", "delete_only", 1./.375), ("activity", "activity", 1.)])
def test_activity_policies_and_invalidation_are_preserved(mass_policy, load_policy, ratio):
    from anyfesolver.activity import ElementActivity, ElementActivityPolicy
    model = bar(n=1)
    activity = ElementActivity([1], .375, policy=ElementActivityPolicy(mass=mass_policy, load=load_policy))
    model.set_element_activity(activity)
    load = LoadCase("gravity", gravity=np.array([1., -2., 3.]))
    with AnalysisSession(model) as session:
        first = collect(model, [StaticCase("gravity", load)], session=session)[0]
        np.testing.assert_allclose(first.case_results[0]["inertia_relief"]["acceleration_at_center_of_mass"], ratio*load.gravity, rtol=1e-10)
        activity.set_activity([1], .125, reason="batch update")
        changed = collect(model, [StaticCase("gravity", load)], session=session)[0]
        oracle = solve_inertia_relief(model, load)
        np.testing.assert_allclose(changed.displacements[:, 0], oracle.displacements, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(changed.case_results[0]["inertia_relief"]["acceleration_at_center_of_mass"], oracle.acceleration_at_center_of_mass, rtol=1e-10)


@pytest.mark.parametrize("magnitude", [1.e-12, 1.e12])
def test_physical_balance_gates_do_not_gain_an_absolute_unit_floor(magnitude):
    model = bar()
    load = LoadCase("scaled"); load.add_nodal_load(4, forces=[magnitude, 0., 0.])
    block = collect(model, [StaticCase("scaled", load)])[0]
    summary = block.case_results[0]["inertia_relief"]
    assert summary["acceleration_at_center_of_mass"][0] == pytest.approx(magnitude/2., rel=1e-10)
    assert summary["diagnostics"]["force_balance_error"] <= 1e-10


def test_direct_constraint_change_does_not_reuse_physical_plan():
    model = bar()
    case = cases(model)[0]
    with AnalysisSession(model) as session:
        collect(model, [case], session=session)
        model.add_constraint_equation(terms=((model.mesh.nodes[2].dofs[0], 1.),
            (model.mesh.nodes[1].dofs[0], -2.)), rhs=0., source_id="restrains translation")
        with pytest.raises(StaticBatchError): collect(model, [case], session=session)


def test_generator_close_releases_owned_session_and_input(monkeypatch):
    import anyfesolver.static_batch as module
    model = bar()
    closed = []
    made = []
    original = module.AnalysisSession
    def create(*args, **kwargs):
        session = original(*args, **kwargs); made.append(session); return session
    monkeypatch.setattr(module, "AnalysisSession", create)
    def inputs():
        try: yield from cases(model)
        finally: closed.append(True)
    stream = iter_linear_batches(model, inputs(), inertia_relief=True, batch_size=1)
    next(stream); stream.close()
    assert closed == [True]
    assert made[0].closed


def test_collecting_relief_external_cache_and_conflict():
    from anyfesolver import solve_linear_many
    from anyfesolver.linalg import FactorizationCache
    model = bar()
    cache = FactorizationCache(name="relief caller", max_entries=2)
    try:
        u, info = solve_linear_many(model, [c.load_case for c in cases(model)[:2]],
            inertia_relief=True, factorization_cache=cache)
        assert u.shape == (24, 2)
        assert cache.diagnostics()["entries"] == 1
        with AnalysisSession(model, factorization_cache=cache) as session:
            with pytest.raises(ValueError, match="match the supplied session"):
                solve_linear_many(model, [], inertia_relief=True, session=session,
                    factorization_cache=FactorizationCache(name="conflicting"))
    finally: cache.clear()


def test_memory_preflight_accounts_for_krylov_workspace_before_certification(monkeypatch):
    import anyfesolver.inertia_relief as module
    from anyfesolver.recovery import estimate_model_memory
    model = bar(n=20)
    n = model.mesh.dof_manager.total_dofs
    recovery = RecoveryConfig(include_displacements=False, include_stresses=False, include_reactions=False)
    estimate = estimate_model_memory(model, num_rhs=0, recovery_config=recovery)
    previous_base_and_one_case = int(estimate.total_bytes_estimate*6+n*24*8+(n*14+1)*8)
    limit = previous_base_and_one_case+module._mechanism_workspace_bytes(n)//2
    def forbidden(*args, **kwargs): pytest.fail("certification ran before memory refusal")
    monkeypatch.setattr(module, "_mechanism_check", forbidden)
    with pytest.raises(ResourcePolicyError):
        collect(model, [cases(model)[0]], output_dofs=[0], recovery_config=recovery,
                resource_config=ResourceConfig(memory_limit_bytes=limit))


@pytest.mark.parametrize("change", ["mass", "stiffness", "constraint"])
def test_unrelated_session_operations_release_stale_relief_plan(change):
    model = bar()
    with AnalysisSession(model) as session:
        collect(model, [cases(model)[0]], session=session)
        assert session._inertia_relief is not None
        if change == "mass":
            model.mesh.point_masses[1] = .2
            session.mass_plan()
        elif change == "stiffness":
            model.bump_revision("material")
            session.stiffness_plan()
        else:
            model.add_constraint_equation(terms=((model.mesh.nodes[2].dofs[3], 1.),
                (model.mesh.nodes[1].dofs[3], -1.)), rhs=0., source_id="tie rotation")
            session.constraint_plan()
        assert session._inertia_relief is None


def test_scalar_and_batch_metadata_cannot_poison_cached_provenance():
    model = bar()
    load = cases(model)[0].load_case
    with AnalysisSession(model) as session:
        first = solve_inertia_relief(model, load, session=session)
        first.diagnostics["mass_formulations"].append("corrupt")
        first.diagnostics["beam_mass_selection"]["1"] = "corrupt"
        second = solve_inertia_relief(model, load, session=session)
        assert second.diagnostics["mass_formulations"] == ["consistent"]
        assert second.diagnostics["beam_mass_selection"]["1"] == "consistent"
        block = collect(model, [StaticCase("one", load)], session=session)[0]
        block.case_results[0]["solver_info"]["equilibration"]["applied"] = "corrupt"
        again = collect(model, [StaticCase("one", load)], session=session)[0]
        assert again.case_results[0]["solver_info"]["equilibration"]["applied"] is False


def test_raw_backend_gauge_quality_is_reported_before_cleanup(monkeypatch):
    model = bar()
    case = cases(model)[0]
    expected = solve_inertia_relief(model, case.load_case)
    with AnalysisSession(model) as session:
        operator = session.static_operator_plan(constraint_mode="nullspace")
        original = type(operator.handle).solve_many
        def drift(self, rhs):
            solved = original(self, rhs)
            solved[:-6, :] += 1.e-4*operator.rigid_modes[:, [0]]
            return solved
        monkeypatch.setattr(type(operator.handle), "solve_many", drift)
        block = collect(model, [case], session=session)[0]
        diagnostics = block.case_results[0]["inertia_relief"]["diagnostics"]
        assert diagnostics["backend_gauge_error"] > 1e-4
        assert diagnostics["gauge_error"] <= 1e-10
        np.testing.assert_allclose(block.displacements[:, 0], expected.displacements, rtol=1e-8, atol=1e-12)


def test_admitted_mpc_structure_change_rebuilds_and_recertifies(monkeypatch):
    import anyfesolver.inertia_relief as module
    model = bar()
    count = []
    original = module._mechanism_check
    def record(*args, **kwargs): count.append(1); return original(*args, **kwargs)
    monkeypatch.setattr(module, "_mechanism_check", record)
    case = cases(model)[0]
    with AnalysisSession(model) as session:
        collect(model, [case], session=session)
        model.add_constraint_equation(terms=((model.mesh.nodes[2].dofs[3], 1.),
            (model.mesh.nodes[1].dofs[3], -1.)), rhs=0., source_id="tie rotation")
        changed = collect(model, [case], session=session)[0]
        assert len(count) == 2
        assert session.diagnostics()["counters"]["inertia_relief_builds"] == 2
        oracle = solve_inertia_relief(model, case.load_case)
        np.testing.assert_allclose(changed.displacements[:, 0], oracle.displacements, rtol=1e-8, atol=1e-12)


def test_scalar_zero_load_has_zero_acceleration_with_reused_plan():
    model = bar()
    with AnalysisSession(model) as session:
        result = solve_inertia_relief(model, session=session)
        assert result.converged
        np.testing.assert_array_equal(result.displacements, np.zeros(24))
        np.testing.assert_array_equal(result.acceleration_at_center_of_mass, np.zeros(3))
        block = collect(model, [StaticCase("zero", None)], session=session)[0]
        np.testing.assert_array_equal(block.displacements[:, 0], result.displacements)


@pytest.mark.parametrize("kind,code", [("support", "unsupported_constraints"), ("disconnected", "disconnected_body")])
def test_empty_inputs_do_not_skip_model_admission(kind, code):
    from anyfesolver import solve_linear_many
    model = bar()
    if kind == "support": model.add_boundary_condition(FixedSupport("fixed", [1]))
    else: model.add_node(99, 10., 0., 0.)
    for invoke in (lambda: collect(model, []), lambda: solve_linear_many(model, [], inertia_relief=True)):
        with pytest.raises(StaticBatchError) as error: invoke()
        assert error.value.diagnostics["scope"] == "model"
        assert error.value.diagnostics["status"] == "invalid_input"
        assert error.value.diagnostics["code"] == code


def test_non_success_mechanism_details_are_detached_from_session_cache():
    from test_inertia_relief import AxialOnlyBeam
    model = bar(n=1)
    element = model.mesh.elements[1]
    model.mesh.elements[1] = AxialOnlyBeam(1, [1, 2], "steel", dict(element.cross_section))
    with AnalysisSession(model) as session:
        first = solve_inertia_relief(model, session=session)
        assert not first.converged
        rows = first.diagnostics["mechanism"]["zero_stiffness_reduced_dofs"]
        assert rows
        rows.clear()
        again = solve_inertia_relief(model, session=session)
        assert again.diagnostics["mechanism"]["zero_stiffness_reduced_dofs"]


def test_collecting_refuses_solid_protocol_before_downstream_dispatch():
    from anyfesolver import InertiaReliefError, solve_linear_many
    class Solid:
        _solid_model_protocol = True
    with pytest.raises(InertiaReliefError) as error:
        solve_linear_many(Solid(), [], inertia_relief=True)
    assert error.value.code == "unsupported_model"
