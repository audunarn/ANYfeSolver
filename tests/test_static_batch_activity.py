from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import AnalysisSession, BoundaryCondition, FixedSupport, LoadCase, solve_linear
from anyfesolver.activity import ContributionPolicy, ElementActivity, ElementActivityPolicy
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.matrix_assembly import assemble_mass_matrix, assemble_stiffness_matrix


def _bar() -> FEModel:
    model = FEModel("activity_session")
    model.add_material("steel", 100.0, 0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel",
        {"area": 1.0, "Iy": 1e-6, "Iz": 1e-6, "J": 1e-6}))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("slider", [2],
        {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    return model


def _load() -> LoadCase:
    load = LoadCase("axial")
    load.add_nodal_load(2, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return load


def _assert_fresh(model: FEModel, session: AnalysisSession) -> None:
    stiffness, _ = assemble_stiffness_matrix(model)
    mass, _ = assemble_mass_matrix(model)
    np.testing.assert_array_equal(session.stiffness_plan().matrix.toarray(), stiffness.toarray())
    np.testing.assert_array_equal(session.mass_plan().matrix.toarray(), mass.toarray())


def test_attach_replace_detach_invalidates_zero_sequence_providers() -> None:
    model = _bar()
    with AnalysisSession(model) as session:
        initial_k = session.stiffness_plan()
        initial_m = session.mass_plan()
        for provider in (ElementActivity([1], 0.5), ElementActivity([1], 0.25), None):
            old_k, old_m = session.stiffness_plan(), session.mass_plan()
            model.set_element_activity(provider)
            _assert_fresh(model, session)
            assert session.stiffness_plan() is not old_k
            assert session.mass_plan() is not old_m
        np.testing.assert_array_equal(session.stiffness_plan().matrix.toarray(), initial_k.matrix.toarray())
        np.testing.assert_array_equal(session.mass_plan().matrix.toarray(), initial_m.matrix.toarray())


def test_default_softening_reuses_mass_but_refreshes_all_stiffness_dependents() -> None:
    model, load = _bar(), _load()
    activity = ElementActivity([1])
    model.set_element_activity(activity)
    with AnalysisSession(model) as session:
        first, info = solve_linear(model, load, session=session)
        old_k, old_m = session.stiffness_plan(), session.mass_plan()
        old_constraint = session.constraint_plan()
        old_reduced, _ = session.reduced_mass()
        old_q, _ = session.rigid_body_modes()
        old_output = session.output_selection_plan(np.asarray([model.mesh.nodes[2].dofs[0]]))
        assert session.factorization_cache.diagnostics()["entries"] == 1
        activity.set_activity([1], [0.5])
        assert session.stiffness_plan() is not old_k
        assert session.factorization_cache.diagnostics()["entries"] == 0
        assert not session._load_vectors and not session._output_plans
        assert session._constraint is None and session._rigid_modes is None
        assert session.mass_plan() is old_m
        assert session.constraint_plan() is not old_constraint
        assert session.reduced_mass()[0] is not old_reduced
        assert session.rigid_body_modes()[0] is not old_q
        assert session.output_selection_plan(old_output.full_dofs) is not old_output
        with pytest.raises(ValueError, match="stale or foreign"):
            session.constraint_plan(old_k)
        with pytest.raises(ValueError, match="stale or foreign"):
            session.output_selection_plan(old_output.full_dofs, old_constraint)
        second, _ = solve_linear(model, load, session=session)
        expected, _ = solve_linear(model, load)
        np.testing.assert_array_equal(second, expected)
        np.testing.assert_allclose(second, 2 * first)
        _assert_fresh(model, session)
        activity.hard_delete([1])
        assert session.mass_plan() is not old_m
        _assert_fresh(model, session)


@pytest.mark.parametrize("policy, k_changes, m_changes", [
    (ElementActivityPolicy(stiffness="retain", mass="activity"), False, True),
    (ElementActivityPolicy(stiffness="delete_only", mass="retain"), False, False),
    (ElementActivityPolicy(stiffness="activity", mass="activity"), True, True),
])
def test_effective_contribution_policies_control_softening_reuse(policy, k_changes, m_changes) -> None:
    model = _bar()
    activity = ElementActivity([1], policy=policy)
    model.set_element_activity(activity)
    with AnalysisSession(model) as session:
        old_k, old_m = session.stiffness_plan(), session.mass_plan()
        activity.set_activity([1], [0.5])
        assert (session.stiffness_plan() is not old_k) is k_changes
        assert (session.mass_plan() is not old_m) is m_changes
        _assert_fresh(model, session)


def test_load_policy_and_provider_identity_refresh_cached_element_load() -> None:
    model = _bar()
    policy = ElementActivityPolicy(stiffness="retain", mass="retain", load="activity")
    activity = ElementActivity([1], policy=policy)
    model.set_element_activity(activity)
    load = LoadCase("element")
    load.element_loads[1] = np.ones(12)
    with AnalysisSession(model) as session:
        old_k = session.stiffness_plan()
        first = session.linear_system(load)[1]
        activity.set_activity([1], [0.5])
        second = session.linear_system(load)[1]
        assert session.stiffness_plan() is old_k
        np.testing.assert_array_equal(second, 0.5 * first)
        model.set_element_activity(ElementActivity([1], 0.25, policy=policy))
        third = session.linear_system(load)[1]
        np.testing.assert_array_equal(third, 0.25 * first)


def test_unchanged_epoch_does_not_rescan_scales_and_prescribed_values_reuse(monkeypatch) -> None:
    model, load = _bar(), _load()
    activity = ElementActivity([1])
    model.set_element_activity(activity)
    with AnalysisSession(model) as session:
        solve_linear(model, load, session=session)
        old_k, old_m = session.stiffness_plan(), session.mass_plan()
        old_constraint = session.constraint_plan()
        import anyfesolver.matrix_assembly as assembly
        original = assembly._activity_scales
        calls = []
        def observe(model, quantity):
            calls.append(quantity)
            return original(model, quantity)
        monkeypatch.setattr(assembly, "_activity_scales", observe)
        for _ in range(3):
            assert session.stiffness_plan() is old_k
            assert session.mass_plan() is old_m
            session.linear_system(load)
        assert calls == []
        model.boundary_conditions[0].dof_constraints["ux"] = 2.5e-4
        current = session.constraint_plan()
        assert current.T is old_constraint.T and current.K_red is old_constraint.K_red
        assert session.factorization_cache.diagnostics()["entries"] == 1
        activity.set_activity([1], [0.5])
        assert session.mass_plan() is old_m
        assert calls == ["mass"]
        session.mass_plan()
        assert calls == ["mass"]


def test_generic_provider_observes_effective_weights_without_sequence_contract() -> None:
    class Generic(ElementActivity):
        multiplier = 1.0
        def scales(self, quantity, element_ids=None):
            return super().scales(quantity, element_ids) * self.multiplier
    model = _bar()
    provider = Generic([1])
    model.set_element_activity(provider)
    with AnalysisSession(model) as session:
        old = session.stiffness_plan()
        provider.multiplier = 0.25
        current = session.stiffness_plan()
        assert provider.sequence == 0 and current is not old
        np.testing.assert_array_equal(current.matrix.toarray(), 0.25 * old.matrix.toarray())
        session.linear_system(_load())
        _assert_fresh(model, session)


def test_replaced_same_identity_scale_code_cannot_use_builtin_epoch(monkeypatch) -> None:
    model = _bar()
    model.set_element_activity(ElementActivity([1]))
    with AnalysisSession(model) as session:
        old = session.stiffness_plan()
        def replacement(self, quantity, element_ids=None):
            return np.full(len(element_ids), 0.25)
        monkeypatch.setattr(ElementActivity.scales, "__code__", replacement.__code__)
        current = session.stiffness_plan()
        assert current is not old
        np.testing.assert_array_equal(current.matrix.toarray(), 0.25 * old.matrix.toarray())



