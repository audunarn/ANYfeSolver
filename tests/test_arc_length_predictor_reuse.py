"""Whole-route retry retention checks using the exact qualified Q4 lifecycle."""
from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import arc_length as arc
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.fe_core import FEModel


def panel():
    model = FEModel("predictor-retry")
    model.add_material("steel", 210e9, 0.3)
    for nid, xy in enumerate(((0., 0.), (1., 0.), (1., 1.), (0., 1.)), 1):
        model.add_node(nid, *xy, 0.)
    model.add_element(1, QualifiedE4PLShellElement(
        1, [1, 2, 3, 4], "steel", thickness=.01,
        reference_normal=(0., 0., 1.),
    ))
    model.add_boundary_condition(BoundaryCondition(
        "fixed", [1, 4], dict.fromkeys(("ux", "uy", "uz", "rx", "ry", "rz"), 0.),
    ))
    model.add_boundary_condition(BoundaryCondition(
        "axial", [2, 3], dict.fromkeys(("ux", "uy", "rx", "ry", "rz"), 0.),
    ))
    load = LoadCase("axial")
    for nid in (2, 3):
        load.add_nodal_load(nid, load_vector=[0., 0., 1e7, 0., 0., 0.])
    return model, load


def run_case(monkeypatch, *, cached=True, reject=True, token=None, always_reject=False):
    model, load = panel()
    real_factorize = arc.factorize
    rejected = []
    def factorize(*args, **kwargs):
        if (reject and "arc_length.corrector:" in kwargs.get("signature", "")
                and (always_reject or not rejected)):
            rejected.append(kwargs["signature"])
            raise RuntimeError("deterministic rejected corrector for retry parity")
        return real_factorize(*args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(arc, "factorize", factorize)
        if not cached:
            scoped.setattr(arc, "_arc_predictor_reuse_exclusion", lambda **_: "oracle")
        result = arc.solve_static_arc_length(
            model, load, control=arc.ArcLengthControl(
                initial_load_increment=.1, minimum_load_increment=1e-6,
                maximum_load_increment=.1, max_steps=2, max_retries_per_step=2,
            ), max_iterations=12, tolerance=1e-10, arc_tolerance=1e-10,
            cancellation_token=token,
        )
    return result, rejected


def assert_same(left, right):
    assert left.status == right.status
    assert left.info["failure_reason"] == right.info["failure_reason"]
    assert left.info["adaptation_history"] == right.info["adaptation_history"]
    assert left.info["total_retries"] == right.info["total_retries"]
    np.testing.assert_array_equal(left.displacements, right.displacements)
    assert left.load_factor == right.load_factor
    assert len(left.steps) == len(right.steps)
    for a, b in zip(left.steps, right.steps):
        assert a.to_dict() == b.to_dict()
    # Public state includes nested arrays; recurse without discarding metadata.
    def equal(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b
    equal(left.element_states, right.element_states)


@pytest.mark.parametrize("always_reject", [False, True])
def test_retry_matches_uncached_route(monkeypatch, always_reject):
    candidate, rejected = run_case(monkeypatch, always_reject=always_reject)
    oracle, oracle_rejected = run_case(monkeypatch, cached=False, always_reject=always_reject)
    assert rejected == oracle_rejected and rejected
    assert_same(candidate, oracle)
    diagnostics = candidate.info["predictor_reuse"]
    assert diagnostics["exclusion_reason"] is None
    assert diagnostics["reuse_count"] > 0
    assert diagnostics["solve_count"] + diagnostics["reuse_count"] == oracle.info["predictor_reuse"]["solve_count"]
    if not always_reject:
        assert candidate.steps


def test_no_retry_does_not_change_route(monkeypatch):
    candidate, _ = run_case(monkeypatch, reject=False)
    oracle, _ = run_case(monkeypatch, cached=False, reject=False)
    assert_same(candidate, oracle)
    assert candidate.info["predictor_reuse"]["reuse_count"] == 0


def test_cancellation_callback_keeps_legacy_path(monkeypatch):
    class RetryCancellation(CancellationToken):
        def raise_if_cancelled(self, stage=""):
            if stage.endswith("retry:1"):
                self.cancel("retry cancellation")
            super().raise_if_cancelled(stage)
    with pytest.raises(SolveCancelled, match="retry cancellation"):
        run_case(monkeypatch, token=RetryCancellation())
    candidate, _ = run_case(monkeypatch, token=CancellationToken())
    assert candidate.info["predictor_reuse"]["exclusion_reason"] == "cancellation_callback"
    assert candidate.info["predictor_reuse"]["reuse_count"] == 0


def test_unowned_contributors_are_excluded():
    model, load = panel()
    common = dict(model=model, load_cases=(load, None), contact=None,
                  follower_active=False, kinematics="von_karman", cancellation_token=None)
    assert arc._arc_predictor_reuse_exclusion(**common) is None
    assert arc._arc_predictor_reuse_exclusion(**dict(common, contact=object())) == "contact_contributor"
    class CustomLoad(LoadCase):
        pass
    assert arc._arc_predictor_reuse_exclusion(**dict(common, load_cases=(CustomLoad("custom"),))) == "custom_load_case"
    from types import SimpleNamespace
    model.mesh.elements[1] = SimpleNamespace()
    assert arc._arc_predictor_reuse_exclusion(**common) == "unqualified_element_lifecycle"


def test_retry_mutation_is_rejected_before_retention(monkeypatch):
    model, load = panel()
    monkeypatch.setitem(globals(), "panel", lambda: (model, load))
    original_safe_point = arc.cancellation_safe_point
    changed = []
    def mutate(token, stage=""):
        if stage.endswith("retry:1"):
            model.set_node_coordinates(2, 1.01, 0., 0.)
            changed.append(stage)
        original_safe_point(token, stage)
    monkeypatch.setattr(arc, "cancellation_safe_point", mutate)
    with pytest.raises((ValueError, NotImplementedError)):
        run_case(monkeypatch)
    assert changed


def test_singular_predictor_keeps_retry_accounting(monkeypatch):
    real_solve = arc._factorized_solve
    signatures = []
    def singular(matrix, rhs, signature, **kwargs):
        if signature.startswith("arc_length.predictor:"):
            signatures.append(signature)
            raise RuntimeError("singular predictor oracle")
        return real_solve(matrix, rhs, signature, **kwargs)
    monkeypatch.setattr(arc, "_factorized_solve", singular)
    candidate, _ = run_case(monkeypatch, reject=False)
    candidate_calls = signatures[:]
    signatures.clear()
    oracle, _ = run_case(monkeypatch, cached=False, reject=False)
    assert candidate_calls == signatures
    assert len(candidate_calls) == 3
    assert_same(candidate, oracle)
    assert candidate.info["predictor_reuse"]["reuse_count"] == 0
    assert candidate.info["failure_reason"] == "singular_predictor_tangent"


def test_s3_and_plastic_material_keep_uncached_path():
    from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
    from anyfesolver.material_curves import dnv_c208_steel_curve
    model, load = panel()
    common = dict(model=model, load_cases=(load, None), contact=None,
                  follower_active=False, kinematics="von_karman", cancellation_token=None)
    assert arc._arc_predictor_reuse_exclusion(**dict(common, initial_state_supplied=True)) == "initial_material_history"
    model.add_material("plastic", 210e9, .3,
                       hardening_curve=dnv_c208_steel_curve("S355", .01))
    model.mesh.elements[1].material_name = "plastic"
    assert arc._arc_predictor_reuse_exclusion(**common) == "plastic_material"
    model.mesh.elements[1] = QualifiedE4PLS3ShellElement(
        1, [1, 2, 4], "steel", thickness=.01, reference_normal=(0., 0., 1.))
    assert arc._arc_predictor_reuse_exclusion(**common) == "unqualified_element_lifecycle"
