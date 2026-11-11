"""Owner-local Q4 preparation parity and adversarial lifetime checks."""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from anyfesolver._stiffened_shell_q4_algorithmic_pullback import response
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from anyfesolver.e4_pl_element import _lookup_prepared_q4_correction
from scripts.experimental_stiffened_ge_multi_member_panel import make_definition, actions_for
from test_corotational import _single_shell_model
from test_stiffened_shell_q4_algorithmic_pullback import yielded


def _fixture():
    model = _single_shell_model()
    element = model.mesh.elements[1]
    material = model.get_material(element.material_name)
    origin = element.init_nonlinear_state(3)
    preparation = element._prepare_nonlinear_linear_correction(model.mesh, material)
    assert preparation is not None
    return model, element, material, origin, preparation


def test_preparation_matches_full_local_and_actual_chart_maps():
    model, element, material, origin, preparation = _fixture()
    before = canonical_json_bytes(origin)
    rng = np.random.default_rng(398)
    for amplitude in (0., 1e-8, .0002, .003):
        u = amplitude*rng.normal(size=24)
        for tangent in (False, True):
            full = element.compute_nonlinear_response(model.mesh, material, u, origin, 3, tangent)
            fast = element._compute_prepared_nonlinear_response(model.mesh, material, u,
                origin, 3, tangent, prepared=preparation)
            np.testing.assert_array_equal(full[0], fast[0])
            if tangent:
                np.testing.assert_array_equal(full[1], fast[1])
            else:
                assert fast[1] is None
            assert canonical_json_bytes(full[2]) == canonical_json_bytes(fast[2])
        full = response(model, element, u, origin, 3)
        fast = response(model, element, u, origin, 3, prepared_correction=preparation)
        for a, b in zip(full[:2], fast[:2]):
            np.testing.assert_array_equal(a, b)
        assert canonical_json_bytes(full[2]) == canonical_json_bytes(fast[2])
    assert canonical_json_bytes(origin) == before
    matrix = np.frombuffer(_lookup_prepared_q4_correction(preparation)[5], dtype=float)
    with pytest.raises(ValueError):
        matrix.setflags(write=True)
    with pytest.raises(AttributeError):
        preparation.payload = bytes(24*24*8)
    forged = type(preparation)()
    with pytest.raises(RuntimeError, match="stale"):
        element._compute_prepared_nonlinear_response(model.mesh, material,
            np.zeros(24), origin, 3, True, prepared=forged)


@pytest.mark.parametrize("mutation", ("node", "node_aba", "material", "element", "token"))
def test_original_binding_rejects_mutation_even_after_valid_cache_rebuild(mutation):
    model, element, material, origin, preparation = _fixture()
    if mutation.startswith("node"):
        node = model.mesh.nodes[element.node_ids[0]]
        old = node.x
        node.x = old+.001
        if mutation == "node_aba":
            node.x = old
    elif mutation == "material":
        material.elastic_modulus *= 1.001
    elif mutation == "element":
        element.thickness *= 1.001
    else:
        # Rebinding an equal-valued mesh token cannot revive original authority.
        model.mesh._qualified_direct_state_token = list(model.mesh._qualified_direct_state_token)
    element.compute_stiffness_matrix(model.mesh, material)
    with pytest.raises(RuntimeError, match="stale"):
        element._compute_prepared_nonlinear_response(model.mesh, material,
            np.zeros(24), origin, 3, True, prepared=preparation)


def test_observation_mutation_discards_prepared_trial():
    model, element, material, origin, preparation = _fixture()
    class Observed:
        def __array__(self, dtype=None, copy=None):
            model.mesh.nodes[element.node_ids[0]].x += .001
            element.compute_stiffness_matrix(model.mesh, material)
            return np.zeros(24, dtype=dtype)
    with pytest.raises(RuntimeError, match="stale"):
        element._compute_prepared_nonlinear_response(model.mesh, material,
            Observed(), origin, 3, True, prepared=preparation)
    assert np.all(origin["alpha"] == 0.)


def test_record_attributes_cannot_substitute_numerical_payload(monkeypatch):
    model, element, material, origin, preparation = _fixture()
    u = np.random.default_rng(924).normal(size=24)*.002
    expected = element.compute_nonlinear_response(model.mesh, material, u, origin, 3, True)
    original = object.__getattribute__
    def poisoned(self, name):
        if name == "payload":
            return bytes(24*24*8)
        return original(self, name)
    monkeypatch.setattr(type(preparation), "__getattribute__", poisoned)
    actual = element._compute_prepared_nonlinear_response(model.mesh, material,
        u, origin, 3, True, prepared=preparation)
    np.testing.assert_array_equal(expected[0], actual[0])
    np.testing.assert_array_equal(expected[1], actual[1])


def test_prepared_entry_retains_class_and_material_provider_boundaries(monkeypatch):
    model, element, material, origin, preparation = _fixture()
    entry = element._compute_prepared_nonlinear_response
    calls = []
    def attacker(*args, **kwargs):
        calls.append(True)
        return np.zeros((6, 6))
    with monkeypatch.context() as patch:
        patch.setattr(type(material), "elastic_compliance_matrix", attacker)
        with pytest.raises((ValueError, RuntimeError)):
            entry(model.mesh, material, np.zeros(24), origin, 3, True, prepared=preparation)
    assert not calls
    with monkeypatch.context() as patch:
        patch.setattr(type(element), "_compute_prepared_nonlinear_response", attacker)
        with pytest.raises((ValueError, RuntimeError)):
            entry(model.mesh, material, np.zeros(24), origin, 3, True, prepared=preparation)
    assert not calls


def test_yielded_origin_and_hardening_retain_full_kernel(yielded):
    model, material, element, _reference, origin, trial = yielded
    assert element._prepare_nonlinear_linear_correction(model.mesh, material) is None
    before = canonical_json_bytes(origin)
    full = response(model, element, trial, origin, 3)
    fallback = response(model, element, trial, origin, 3, prepared_correction=None)
    np.testing.assert_array_equal(full[0], fallback[0])
    np.testing.assert_array_equal(full[1], fallback[1])
    assert canonical_json_bytes(origin) == before


def test_complete_history_checkpoint_and_rejection_match_full_oracle():
    definition = replace(make_definition(2, 2),
                         q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    actions = actions_for(definition)
    fast, full, resumed = (ss.create_analysis(definition=definition) for _ in range(3))
    try:
        full._finite_driver = FinitePointDriver(full)
        assert full._finite_driver._q4_corrections
        full._finite_driver._q4_corrections.clear()
        for owner in (fast, full):
            result = owner.solve(actions, (.01, .02), max_iterations=8, max_backtracks=4)
            assert result.status == "completed" and result.generation == 2
        saved = fast.checkpoint()
        assert saved == full.checkpoint()
        resumed.restore(saved)
        assert resumed.checkpoint() == saved
        for owner in (fast, full, resumed):
            before = owner.checkpoint()
            with pytest.raises(Exception):
                owner.solve(actions, (.03,), max_iterations=1, max_backtracks=1)
            assert owner.checkpoint() == before
        results = [owner.solve(actions, (.03,), max_iterations=8, max_backtracks=4)
                   for owner in (fast, full, resumed)]
        for other in results[1:]:
            for field in ("shell_displacement", "beam_positions", "beam_frames",
                          "cell_frames", "cell_resultants", "multipliers", "support_reactions"):
                np.testing.assert_array_equal(getattr(results[0], field), getattr(other, field))
        assert fast.checkpoint() == full.checkpoint() == resumed.checkpoint()
    finally:
        for owner in (fast, full, resumed):
            owner.close()
