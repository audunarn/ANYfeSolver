"""Atomic finite-step checks for the experimental native point panel."""

import numpy as np
import pytest
from dataclasses import replace

from anymaterial import DNVC208MaterialCurve
from anyfesolver import b3_ge
from anyfesolver._stiffened_shell_ge_experimental import (
    ExperimentalGEAnalysis, NodalAction,
)
from anyfesolver import _stiffened_shell_ge_finite_experimental as finite_experimental
from anyfesolver._stiffened_shell_ge_checkpoint_binary import (
    MAGIC as BINARY_CHECKPOINT_MAGIC, encode as encode_binary_checkpoint,
    S3_MAGIC as S3_BINARY_CHECKPOINT_MAGIC, decode_s3, encode_s3,
)
from anyfesolver.e4_pl_s3_v2d_state import seal_v2d_state
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_ge_experimental import _panel


def _actions():
    return (NodalAction("beam", 105, (0., 0., 1000., 0., 0., 0.)),)


def test_finite_position_pair_preserves_sub_spacing_motion():
    reference = np.array([[1., 0., 0.]])
    displacement = np.array([[2.**-54, 0., 0.]])
    high, low = finite_experimental._position_pair(reference, displacement)
    assert high[0, 0] == 1.
    assert low[0, 0] == displacement[0, 0]
    reference[0, 0] = 2.
    displacement[0, 0] = 0.
    assert high[0, 0] == 1. and low[0, 0] == 2.**-54


def test_finite_joint_resolves_sub_spacing_beam_motion():
    owner = ExperimentalGEAnalysis(_panel())
    driver = finite_experimental.FinitePointDriver(owner)
    count = owner._K.shape[0]
    pattern = np.zeros(count)
    state = driver._state
    shifted_q = state.q.copy()
    shifted_q[owner._shell_count+6*owner._beam_index[103]] = 2.**-54
    shifted = replace(state, q=shifted_q)
    baseline = driver._assemble(
        state, 0., driver._shell_origins, driver._beam_origins,
        pattern=pattern)[0]
    changed = driver._assemble(
        shifted, 0., driver._shell_origins, driver._beam_origins,
        pattern=pattern)[0]
    assert baseline[count+6] == 0.
    assert changed[count+6] == 2.**-54


def _plastic_panel():
    base = _panel()
    beams = []
    for definition in base.beams:
        element, inertia = definition.instantiate()
        section = b3_ge.EllipsoidalGeneralizedSection(
            element.section.elastic, element.section.metric, 4000., 1e8)
        beams.append(b3_ge.define_beam(
            "b3-ge", element.element_id, element.node_ids,
            element.operator.reference.coordinates,
            element.operator.reference.nodal_triads, section, inertia,
            quadrature=element.operator.order))
    return replace(base, beams=tuple(beams))


def _q4_shell_j2_panel():
    base = _plastic_panel()
    pressure = 1.0e6
    curve = DNVC208MaterialCurve(
        pressure, 1.05*pressure, 1.1*pressure,
        .005, .01, 4.*pressure, .2)
    return replace(base, shell=replace(base.shell, j2_curve=curve))


def test_mixed_q4_s3_small_load_converges_without_shell_pullback_floor():
    base = _panel()
    pressure = 20_000.
    curve = DNVC208MaterialCurve(
        pressure, 1.05*pressure, 1.1*pressure,
        .005, .01, 4.*pressure, .2)
    definition = replace(base, shell=replace(
        base.shell, q4_elements=((1, 2, 5, 4),),
        s3_elements=((2, 3, 6), (2, 6, 5)), j2_curve=curve))
    owner = ExperimentalGEAnalysis(definition)
    result = owner.solve(_actions(), (.01,), max_iterations=20,
                         max_backtracks=12)
    assert result.accepted_factors == (.01,)
    assert result.free_equilibrium_ratio <= 1e-8
    assert result.constraint_max <= 1e-9
    assert len(result.shell_q4_material) == 1
    assert len(result.shell_s3_material) == 2


def test_q4_shell_and_beam_yield_atomically_and_restart(monkeypatch):
    definition = _q4_shell_j2_panel()
    direct_owner = ExperimentalGEAnalysis(definition)
    accepted = direct_owner.solve(
        _actions(), (.5, 1.), max_iterations=20, max_backtracks=12)
    assert accepted.free_equilibrium_ratio <= 1e-8
    assert accepted.constraint_max <= 1e-9
    assert max(max(row) for row in accepted.shell_layer_alpha) > 0.
    assert max(max(row) for row in accepted.beam_station_accumulated) > 0.
    assert tuple(item.element_id for item in accepted.shell_q4_material) == (1, 2)
    e = definition.shell.young_pa
    nu = definition.shell.poisson
    plane_stress = e/(1.-nu*nu)*np.array(
        ((1., nu, 0.), (nu, 1., 0.), (0., 0., (1.-nu)/2.)))
    for index, response in enumerate(accepted.shell_q4_material):
        assert response.layer_stress.shape == (12, 3)
        assert response.layer_strain.shape == (12, 3)
        assert np.isfinite(response.layer_stress).all()
        assert not response.layer_stress.flags.writeable
        assert not response.plastic_strain.flags.writeable
        with pytest.raises(ValueError):
            response.layer_stress.setflags(write=True)
        np.testing.assert_array_equal(response.alpha,
                                      accepted.shell_layer_alpha[index])
        np.testing.assert_allclose(
            response.layer_stress,
            (response.layer_strain-response.plastic_strain)@plane_stress.T,
            rtol=1e-10, atol=1e-6)
    saved = direct_owner.checkpoint()
    assert saved.startswith(BINARY_CHECKPOINT_MAGIC)
    restored_owner = ExperimentalGEAnalysis(definition)
    restored = restored_owner.restore(saved)
    assert restored.shell_layer_alpha == accepted.shell_layer_alpha
    assert restored.beam_station_accumulated == accepted.beam_station_accumulated
    for original, repeated in zip(accepted.shell_q4_material,
                                  restored.shell_q4_material):
        assert original.element_id == repeated.element_id
        for field in ("layer_strain", "layer_stress", "plastic_strain", "alpha"):
            np.testing.assert_allclose(getattr(original, field),
                                       getattr(repeated, field),
                                       rtol=1e-10, atol=1e-8)
    np.testing.assert_array_equal(restored.shell_displacement,
                                  accepted.shell_displacement)

    original = finite_experimental.FinitePointDriver._assemble

    def fail_trial(self, state, factor, *args, **kwargs):
        if factor == 1.1:
            raise RuntimeError("fixture rejected mixed material trial")
        return original(self, state, factor, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(finite_experimental.FinitePointDriver,
                        "_assemble", fail_trial)
        with pytest.raises(RuntimeError, match="rejected mixed material trial"):
            direct_owner.solve(_actions(), (1.1,), max_iterations=20,
                               max_backtracks=12)
    assert direct_owner.checkpoint() == saved
    assert direct_owner.recover() is accepted
    def fail_shell_state(_states):
        raise RuntimeError("fixture shell state preparation failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(finite_experimental, "_shell_material_arrays",
                        fail_shell_state)
        with pytest.raises(RuntimeError, match="shell state preparation failure"):
            direct_owner.solve(_actions(), (1.1,), max_iterations=20,
                               max_backtracks=12)
    assert direct_owner.checkpoint() == saved
    assert direct_owner.recover() is accepted
    token = CancellationToken()
    token.cancel("mixed J2 accepted prefix")
    with pytest.raises(SolveCancelled, match="mixed J2 accepted prefix"):
        direct_owner.solve(_actions(), (1.1,), cancellation_token=token)
    assert direct_owner.checkpoint() == saved

    shell_parent = direct_owner._finite_driver._shell_origins
    beam_parent = direct_owner._finite_driver._beam_origins
    direct = direct_owner.solve(_actions(), (1.1,), max_iterations=20,
                                max_backtracks=12)
    resumed = restored_owner.solve(_actions(), (1.1,), max_iterations=20,
                                    max_backtracks=12)
    assert direct.accepted_factors == resumed.accepted_factors
    assert direct.shell_layer_alpha == resumed.shell_layer_alpha
    assert direct.beam_station_accumulated == resumed.beam_station_accumulated
    for current, restarted in zip(direct.shell_q4_material,
                                  resumed.shell_q4_material):
        np.testing.assert_allclose(current.layer_stress,
                                   restarted.layer_stress,
                                   rtol=1e-10, atol=1e-8)
    np.testing.assert_array_equal(direct.shell_displacement,
                                  resumed.shell_displacement)
    np.testing.assert_array_equal(direct.cell_resultants,
                                  resumed.cell_resultants)

    driver = direct_owner._finite_driver
    state = driver._state
    _, tangent, _, _, shell_trial, _ = driver._assemble(
        state, 1.1, shell_parent, beam_parent, pattern=driver._pattern)
    assert max(float(np.max(new["alpha"]-old["alpha"]))
               for new, old in zip(shell_trial, shell_parent)) > 0.
    rng = np.random.default_rng(20260929)
    count = direct_owner._K.shape[0]
    end_nodal = direct_owner._shell_count+6*len(direct_owner._beam_ids)
    direction = np.zeros(tangent.shape[0])
    direction[:end_nodal] = rng.normal(size=end_nodal)*1e-7
    direction[end_nodal:count] = rng.normal(size=count-end_nodal)*1e-3
    direction[count:] = rng.normal(size=len(direction)-count)*1e-3
    direction[direct_owner._fixed] = 0.
    predicted = tangent@direction
    for step in (.5, .25):
        plus = driver._advance(state, direction, step)
        minus = driver._advance(state, direction, -step)
        positive = driver._assemble(
            plus, 1.1, shell_parent, beam_parent,
            pattern=driver._pattern)[0]
        negative = driver._assemble(
            minus, 1.1, shell_parent, beam_parent,
            pattern=driver._pattern)[0]
        observed = (positive-negative)/(2*step)
        for left, right in ((0, direct_owner._shell_count),
                            (direct_owner._shell_count, count),
                            (count, len(direction))):
            relative = (np.linalg.norm((observed-predicted)[left:right]) /
                        np.linalg.norm(predicted[left:right]))
            assert relative < 1e-5


def test_mixed_q4_s3_j2_and_beam_yield_checkpoint(monkeypatch):
    base = _q4_shell_j2_panel()
    definition = replace(base, shell=replace(
        base.shell, q4_elements=((1, 2, 5, 4),),
        s3_elements=((2, 3, 6), (2, 6, 5))))
    caller_coordinates = np.array(definition.shell.coordinates_m, copy=True)
    caller_q4 = list(definition.shell.q4_elements)
    caller_s3 = list(definition.shell.s3_elements)
    caller_beams = list(definition.beams)
    caller_points = list(definition.points)
    caller_fixed = list(definition.fixed)
    caller_definition = replace(
        definition, shell=replace(
            definition.shell, coordinates_m=caller_coordinates,
            q4_elements=caller_q4, s3_elements=caller_s3),
        beams=caller_beams, points=caller_points, fixed=caller_fixed)
    owner = ExperimentalGEAnalysis(caller_definition)
    original_capabilities = owner.capabilities
    original_identity = owner.preflight()["identity"]
    caller_coordinates[:, :] += 10.
    caller_q4.clear()
    caller_s3.clear()
    caller_beams.clear()
    caller_points.clear()
    caller_fixed.clear()
    assert owner.capabilities == original_capabilities
    assert owner.preflight()["identity"] == original_identity
    accepted = owner.solve(_actions(), (.5, 1.), max_iterations=20,
                           max_backtracks=12)
    assert accepted.status == "completed"
    assert accepted.accepted_factors == (.5, 1.)
    assert accepted.free_equilibrium_ratio <= 1e-8
    assert accepted.constraint_max <= 1e-9
    assert len(accepted.shell_q4_material) == 1
    assert len(accepted.shell_s3_material) == 2
    assert np.max(accepted.shell_q4_material[0].alpha) > 0.
    assert all(np.isfinite(item.layer_stress).all()
               for item in accepted.shell_s3_material)
    assert max(max(row) for row in accepted.beam_station_accumulated) > 0.

    saved = owner.checkpoint()
    driver = owner._finite_driver
    state = driver._state
    args = (1.05, driver._shell_origins, driver._beam_origins)
    residual, tangent, *_ = driver._assemble(
        state, *args, pattern=driver._pattern)
    count = owner._K.shape[0]
    free = np.r_[owner._free, np.arange(count, tangent.shape[0])]
    direction = np.zeros(tangent.shape[0])
    direction[free] = finite_experimental.spla.spsolve(
        tangent[free, :][:, free], -residual[free])
    candidate = driver._advance(state, direction, .5)
    base_eval = driver._assemble(candidate, *args, pattern=driver._pattern)
    predicted = base_eval[1]@direction
    origin_alpha = np.asarray(driver._shell_origins[0]["alpha"])
    active = np.asarray(base_eval[4][0]["alpha"]) > origin_alpha+1e-12
    assert 0 < np.count_nonzero(active) < active.size
    for amount in (1e-2, 1e-3):
        positive = driver._assemble(
            driver._advance(candidate, direction, amount), *args,
            pattern=driver._pattern)
        negative = driver._assemble(
            driver._advance(candidate, direction, -amount), *args,
            pattern=driver._pattern)
        observed = (positive[0]-negative[0])/(2*amount)
        for response in (positive, negative):
            np.testing.assert_array_equal(
                np.asarray(response[4][0]["alpha"]) > origin_alpha+1e-12,
                active)
        for indices in (slice(0, owner._shell_count),
                        slice(owner._shell_count, count), free):
            error = np.linalg.norm((observed-predicted)[indices])
            scale = np.linalg.norm(predicted[indices])
            assert scale > 1.
            assert error/scale < 1e-5
        assert np.linalg.norm((observed-predicted)[count:]) < 1e-12
    assert owner.checkpoint() == saved

    restored_owner = ExperimentalGEAnalysis(definition)
    restored = restored_owner.restore(saved)
    assert restored.shell_layer_alpha == accepted.shell_layer_alpha
    assert restored.beam_station_accumulated == accepted.beam_station_accumulated
    for field in ("shell_displacement", "beam_positions", "beam_frames",
                  "cell_frames", "cell_resultants", "multipliers",
                  "support_reactions"):
        np.testing.assert_allclose(getattr(restored, field),
                                   getattr(accepted, field),
                                   rtol=1e-12, atol=1e-10)
    for name in ("shell_q4_material", "shell_s3_material"):
        before, after = getattr(accepted, name), getattr(restored, name)
        assert len(before) == len(after)
        for original, repeated in zip(before, after):
            for field in ("layer_strain", "layer_stress", "plastic_strain",
                          "alpha"):
                np.testing.assert_allclose(getattr(original, field),
                                           getattr(repeated, field),
                                           rtol=1e-10, atol=1e-8)

    original_assemble = finite_experimental.FinitePointDriver._assemble
    candidate_assemblies = 0

    def reject_advanced_candidate(self, state, factor, *args, **kwargs):
        nonlocal candidate_assemblies
        response = original_assemble(self, state, factor, *args, **kwargs)
        if factor == 1.1:
            candidate_assemblies += 1
            if candidate_assemblies == 2:
                assert not np.array_equal(state.q, self._state.q)
                raise RuntimeError("rejected advanced mixed-material candidate")
        return response

    with monkeypatch.context() as patcher:
        patcher.setattr(finite_experimental.FinitePointDriver, "_assemble",
                        reject_advanced_candidate)
        with pytest.raises(RuntimeError, match="advanced mixed-material"):
            owner.solve(_actions(), (1.1,), max_iterations=20,
                        max_backtracks=12)
    assert candidate_assemblies == 2
    assert owner.checkpoint() == saved
    assert owner.recover() is accepted

    direct_next = owner.solve(_actions(), (1.05,), max_iterations=20,
                              max_backtracks=12)
    resumed_next = restored_owner.solve(_actions(), (1.05,),
                                        max_iterations=20, max_backtracks=12)
    assert direct_next.accepted_factors == resumed_next.accepted_factors
    assert direct_next.shell_layer_alpha == resumed_next.shell_layer_alpha
    assert (direct_next.beam_station_accumulated ==
            resumed_next.beam_station_accumulated)
    for field in ("shell_displacement", "beam_positions", "beam_frames",
                  "cell_frames", "cell_resultants", "multipliers",
                  "support_reactions"):
        np.testing.assert_allclose(getattr(direct_next, field),
                                   getattr(resumed_next, field),
                                   rtol=1e-10, atol=1e-8)
    for name in ("shell_q4_material", "shell_s3_material"):
        before, after = getattr(direct_next, name), getattr(resumed_next, name)
        assert len(before) == len(after)
        for original, repeated in zip(before, after):
            for field in ("layer_strain", "layer_stress", "plastic_strain",
                          "alpha"):
                np.testing.assert_allclose(getattr(original, field),
                                           getattr(repeated, field),
                                           rtol=1e-10, atol=1e-8)


def test_q4_j2_positive_unload_reload_restores_layer_response():
    definition = _q4_shell_j2_panel()
    actions = _actions()
    direct_owner = ExperimentalGEAnalysis(definition)
    loaded = direct_owner.solve(actions, (.5, 1.), max_iterations=20,
                                max_backtracks=12)
    resumed_owner = ExperimentalGEAnalysis(definition)
    resumed_owner.restore(direct_owner.checkpoint())
    loaded_stress = tuple(item.layer_stress.copy()
                          for item in loaded.shell_q4_material)
    for factor in (.7, .4, .2, .1, .5, 1.1):
        direct = direct_owner.solve(actions, (factor,), max_iterations=20,
                                    max_backtracks=12)
        resumed = resumed_owner.solve(actions, (factor,), max_iterations=20,
                                      max_backtracks=12)
        assert direct.free_equilibrium_ratio <= 1e-8
        assert resumed.free_equilibrium_ratio <= 1e-8
        assert direct.shell_layer_alpha == resumed.shell_layer_alpha
        assert direct.beam_station_accumulated == resumed.beam_station_accumulated
        for current, restarted in zip(direct.shell_q4_material,
                                      resumed.shell_q4_material):
            np.testing.assert_allclose(current.layer_stress,
                                       restarted.layer_stress,
                                       rtol=1e-10, atol=1e-8)
            np.testing.assert_allclose(current.plastic_strain,
                                       restarted.plastic_strain,
                                       rtol=1e-10, atol=1e-12)
        if factor == .1:
            assert direct.shell_layer_alpha == loaded.shell_layer_alpha
            assert any(not np.array_equal(item.layer_stress, before)
                       for item, before in zip(direct.shell_q4_material,
                                               loaded_stress))
    assert direct.accepted_factors == resumed.accepted_factors == (
        .5, 1., .7, .4, .2, .1, .5, 1.1)
    assert max(max(row) for row in direct.shell_layer_alpha) > max(
        max(row) for row in loaded.shell_layer_alpha)


def test_q4_j2_checkpoint_rejects_corrupt_shell_history():
    definition = _q4_shell_j2_panel()
    owner = ExperimentalGEAnalysis(definition)
    owner.solve(_actions(), (.5, 1.), max_iterations=20,
                max_backtracks=12)
    saved = bytearray(owner.checkpoint())
    saved[-1] ^= 1
    with pytest.raises(ValueError, match="digest mismatch"):
        ExperimentalGEAnalysis(definition).restore(bytes(saved))

    driver = owner._finite_driver
    state = driver._state
    beam_plastic, beam_accumulated = finite_experimental._material_arrays(
        driver._beam_origins)
    shell_plastic, shell_alpha = finite_experimental._shell_material_arrays(
        driver._shell_origins)
    changed_plastic = shell_plastic.copy()
    changed_plastic[0, 0, 0] += .01
    reencoded = encode_binary_checkpoint(
        "finite_shell_material", owner._identity, owner._generation,
        (("q", state.q), ("frames", state.frames),
         ("cells", state.cells), ("resultants", state.resultants),
         ("multipliers", state.multipliers), ("pattern", driver._pattern),
         ("accepted_factors", np.asarray(owner.recover().accepted_factors)),
         ("beam_plastic", beam_plastic),
         ("beam_accumulated", beam_accumulated),
         ("shell_plastic", changed_plastic), ("shell_alpha", shell_alpha)))
    with pytest.raises(ValueError, match="mechanically inconsistent"):
        ExperimentalGEAnalysis(definition).restore(reencoded)


def test_s3_shell_j2_is_explicitly_experimental():
    base = _q4_shell_j2_panel()
    definition = replace(base, shell=replace(
        base.shell, q4_elements=(),
        s3_elements=((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5))))
    owner = ExperimentalGEAnalysis(definition)
    assert owner.capabilities["status"] == "EXPERIMENTAL_NOT_QUALIFIED"
    assert "experimental_q4_s3_j2_finite_point" in owner.capabilities["analysis"]
    assert "qualified S3 J2 finite-rotation response" in owner.capabilities["excluded"]
    assert owner.preflight()["stress_free_scaled_max"] <= 32.*np.finfo(float).eps


def test_s3_j2_checkpoint_restores_without_beam_yield():
    base = _q4_shell_j2_panel()
    definition = replace(base, beams=_panel().beams, shell=replace(
        base.shell, q4_elements=(),
        s3_elements=((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5))))
    owner = ExperimentalGEAnalysis(definition)
    accepted = owner.solve(_actions(), (.05,), max_iterations=20,
                           max_backtracks=12)
    assert accepted.status == "completed"
    assert len(accepted.shell_s3_material) == 4
    assert max(max(row) for row in accepted.shell_layer_alpha) == 0.
    assert max(max(row) for row in accepted.beam_station_accumulated) == 0.
    saved = owner.checkpoint()
    assert saved.startswith(S3_BINARY_CHECKPOINT_MAGIC)
    driver = owner._finite_driver
    beam_plastic, beam_accumulated = finite_experimental._material_arrays(
        driver._beam_origins)
    shell_plastic, shell_alpha = finite_experimental._shell_material_arrays(())
    expected = (("q", driver._state.q.shape),
                ("frames", driver._state.frames.shape),
                ("cells", driver._state.cells.shape),
                ("resultants", driver._state.resultants.shape),
                ("multipliers", driver._state.multipliers.shape),
                ("pattern", driver._pattern.shape),
                ("accepted_factors", (1,)),
                ("beam_plastic", beam_plastic.shape),
                ("beam_accumulated", beam_accumulated.shape),
                ("shell_plastic", shell_plastic.shape),
                ("shell_alpha", shell_alpha.shape))
    generation, arrays, blobs = decode_s3(
        saved, owner._identity, lambda _: expected, 4)
    _, element, slots, _ = owner._shell_parts[0]
    material = owner._shell_model.get_material("shell")
    wrong_chart = element.seal_solver_integrated_nonlinear_state(
        owner._shell_model.mesh, material, driver._shell_origins[0], 3,
        driver._state.q[slots], kinematics="COROTATIONAL")
    wrong_blob = element.serialize_nonlinear_state(
        owner._shell_model.mesh, material, wrong_chart, 3)
    altered = encode_s3(owner._identity, generation,
                        tuple((name, arrays[name]) for name, _ in expected),
                        (wrong_blob,)+blobs[1:])
    with pytest.raises(ValueError, match="disagrees with mixed owner|committed displacement binding mismatch"):
        ExperimentalGEAnalysis(definition).restore(altered)
    wrong_constitutive = dict(driver._shell_origins[0])
    shifted = np.asarray(wrong_constitutive["committed_constitutive_u"]).copy()
    shifted[0] += 1e-6
    wrong_constitutive["committed_constitutive_u"] = shifted
    wrong_constitutive = seal_v2d_state(wrong_constitutive)
    wrong_blob = element.serialize_nonlinear_state(
        owner._shell_model.mesh, material, wrong_constitutive, 3)
    altered = encode_s3(owner._identity, generation,
                        tuple((name, arrays[name]) for name, _ in expected),
                        (wrong_blob,)+blobs[1:])
    with pytest.raises(ValueError, match="disagrees with mixed owner"):
        ExperimentalGEAnalysis(definition).restore(altered)
    resumed_owner = ExperimentalGEAnalysis(definition)
    restored = resumed_owner.restore(saved)
    assert restored.accepted_factors == accepted.accepted_factors
    np.testing.assert_array_equal(restored.shell_displacement,
                                  accepted.shell_displacement)
    for original, repeated in zip(accepted.shell_s3_material,
                                  restored.shell_s3_material):
        for field in ("layer_strain", "layer_stress", "plastic_strain", "alpha",
                      "station_generalized_resultant"):
            np.testing.assert_allclose(getattr(original, field),
                                       getattr(repeated, field),
                                       rtol=1e-10, atol=1e-8)
    direct_next = owner.solve(_actions(), (.1,), max_iterations=20,
                              max_backtracks=12)
    resumed_next = resumed_owner.solve(_actions(), (.1,), max_iterations=20,
                                      max_backtracks=12)
    assert direct_next.accepted_factors == resumed_next.accepted_factors
    np.testing.assert_allclose(direct_next.shell_displacement,
                               resumed_next.shell_displacement,
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(direct_next.beam_positions,
                               resumed_next.beam_positions,
                               rtol=1e-10, atol=1e-12)
    prefix = resumed_owner.checkpoint()
    token = CancellationToken()
    token.cancel("S3 accepted-prefix check")
    with pytest.raises(SolveCancelled, match="S3 accepted-prefix check"):
        resumed_owner.solve(_actions(), (.15,), cancellation_token=token)
    assert resumed_owner.checkpoint() == prefix


def test_yielded_s3_checkpoint_rejection_and_continuation(monkeypatch):
    base = _panel()
    pressure = 20_000.
    curve = DNVC208MaterialCurve(
        pressure, 1.05*pressure, 1.1*pressure,
        .005, .01, 4.*pressure, .2)
    definition = replace(base, shell=replace(
        base.shell, q4_elements=(),
        s3_elements=((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5)),
        j2_curve=curve))
    owner = ExperimentalGEAnalysis(definition)
    accepted = owner.solve(_actions(), (.05,), max_iterations=20,
                           max_backtracks=12)
    assert max(max(row) for row in accepted.shell_layer_alpha) > 0.
    assert max(max(row) for row in accepted.beam_station_accumulated) == 0.
    assert accepted.free_equilibrium_ratio <= 1e-8
    saved = owner.checkpoint()
    assert saved.startswith(S3_BINARY_CHECKPOINT_MAGIC)
    resumed_owner = ExperimentalGEAnalysis(definition)
    restored = resumed_owner.restore(saved)
    assert restored.shell_layer_alpha == accepted.shell_layer_alpha
    for original, repeated in zip(accepted.shell_s3_material,
                                  restored.shell_s3_material):
        for field in ("layer_stress", "plastic_strain", "alpha"):
            np.testing.assert_allclose(getattr(original, field),
                                       getattr(repeated, field),
                                       rtol=1e-10, atol=1e-8)

    original_assemble = finite_experimental.FinitePointDriver._assemble

    rejected_assemblies = 0

    def reject_next(self, state, factor, *args, **kwargs):
        nonlocal rejected_assemblies
        response = original_assemble(self, state, factor, *args, **kwargs)
        if factor == .06:
            rejected_assemblies += 1
            if rejected_assemblies == 2:
                assert not np.array_equal(state.q, self._state.q)
                raise RuntimeError("fixture rejected yielded S3 trial")
        return response

    with monkeypatch.context() as patcher:
        patcher.setattr(finite_experimental.FinitePointDriver,
                        "_assemble", reject_next)
        with pytest.raises(RuntimeError, match="rejected yielded S3 trial"):
            owner.solve(_actions(), (.06,), max_iterations=20,
                        max_backtracks=12)
    assert rejected_assemblies == 2
    assert owner.checkpoint() == saved
    token = CancellationToken()
    cancelled_assemblies = 0

    def cancel_next(self, state, factor, *args, **kwargs):
        nonlocal cancelled_assemblies
        response = original_assemble(self, state, factor, *args, **kwargs)
        if factor == .06:
            cancelled_assemblies += 1
            if cancelled_assemblies == 2:
                assert not np.array_equal(state.q, self._state.q)
                token.cancel("yielded S3 accepted-prefix cancellation")
        return response

    with monkeypatch.context() as patcher:
        patcher.setattr(finite_experimental.FinitePointDriver,
                        "_assemble", cancel_next)
        with pytest.raises(SolveCancelled, match="yielded S3 accepted-prefix"):
            owner.solve(_actions(), (.06,), cancellation_token=token,
                        max_iterations=20, max_backtracks=12)
    assert cancelled_assemblies == 2
    assert owner.checkpoint() == saved

    direct_next = owner.solve(_actions(), (.06,), max_iterations=20,
                              max_backtracks=12)
    resumed_next = resumed_owner.solve(_actions(), (.06,), max_iterations=20,
                                      max_backtracks=12)
    assert direct_next.accepted_factors == resumed_next.accepted_factors
    assert direct_next.shell_layer_alpha == resumed_next.shell_layer_alpha
    assert (direct_next.beam_station_accumulated ==
            resumed_next.beam_station_accumulated)
    for field in ("shell_displacement", "beam_positions", "beam_frames",
                  "cell_frames", "cell_resultants", "multipliers",
                  "support_reactions"):
        np.testing.assert_allclose(getattr(direct_next, field),
                                   getattr(resumed_next, field),
                                   rtol=1e-10, atol=1e-12)
    for original, repeated in zip(direct_next.shell_s3_material,
                                  resumed_next.shell_s3_material):
        for field in ("layer_stress", "plastic_strain", "alpha"):
            np.testing.assert_allclose(getattr(original, field),
                                       getattr(repeated, field),
                                       rtol=1e-10, atol=1e-8)


def test_s3_shell_yielded_beam_actual_update_tangent():
    base = _q4_shell_j2_panel()
    definition = replace(base, shell=replace(
        base.shell, q4_elements=(),
        s3_elements=((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5))))
    owner = ExperimentalGEAnalysis(definition)
    accepted = owner.solve(_actions(), (.5,), max_iterations=20,
                           max_backtracks=12)
    assert accepted.generation == 1
    assert max(max(row) for row in accepted.beam_station_accumulated) > 0.
    assert max(max(row) for row in accepted.shell_layer_alpha) == 0.

    driver = owner._finite_driver
    state = driver._state
    args = (1., driver._shell_origins, driver._beam_origins)
    residual, tangent, *_ = driver._assemble(
        state, *args, pattern=driver._pattern)
    count = owner._K.shape[0]
    free = np.r_[owner._free, np.arange(count, tangent.shape[0])]
    direction = np.zeros(tangent.shape[0])
    direction[free] = finite_experimental.spla.spsolve(
        tangent[free, :][:, free], -residual[free])
    predicted = tangent@direction
    for magnitude in (1e-6, 1e-7):
        positive = driver._assemble(
            driver._advance(state, direction, magnitude), *args,
            pattern=driver._pattern)[0]
        negative = driver._assemble(
            driver._advance(state, direction, -magnitude), *args,
            pattern=driver._pattern)[0]
        observed = (positive-negative)/(2*magnitude)
        for indices in (slice(0, owner._shell_count),
                        slice(owner._shell_count, count), free):
            relative = (np.linalg.norm((observed-predicted)[indices]) /
                        np.linalg.norm(predicted[indices]))
            assert relative < 1e-5
        assert np.linalg.norm((observed-predicted)[count:]) < 1e-9
    assert owner._generation == 1


def test_generalized_beam_plastic_history_is_atomic_and_restartable(monkeypatch):
    definition = _plastic_panel()
    owner = ExperimentalGEAnalysis(definition)
    accepted = owner.solve(_actions(), (1.,), max_iterations=20,
                           max_backtracks=12)
    assert accepted.generation == 1
    assert max(value for member in accepted.beam_station_accumulated
               for value in member) > 0.
    driver = owner._finite_driver
    shell_origin = tuple(part[3] for part in owner._shell_parts)
    beam_origin = tuple(part[2] for part in owner._beam_parts)
    _, tangent, _, _, _, active = driver._assemble(
        driver._state, 1., shell_origin, beam_origin,
        pattern=driver._pattern)
    assert max(float(sum(station.accumulated)) for history in active
               for station in history.stations) > 0.
    rng = np.random.default_rng(9125)
    count = owner._K.shape[0]
    beam_nodes_end = owner._shell_count+6*len(owner._beam_ids)
    direction = np.zeros(tangent.shape[0])
    direction[:owner._shell_count] = rng.normal(
        size=owner._shell_count)*1e-7
    direction[owner._shell_count:beam_nodes_end] = rng.normal(
        size=beam_nodes_end-owner._shell_count)*1e-7
    direction[beam_nodes_end:count] = rng.normal(
        size=count-beam_nodes_end)*1e-3
    direction[count:] = rng.normal(size=len(direction)-count)*1e-3
    direction[owner._fixed] = 0.
    predicted = tangent@direction
    for step in (.5, .25):
        plus = driver._advance(driver._state, direction, step)
        minus = driver._advance(driver._state, direction, -step)
        positive = driver._assemble(
            plus, 1., shell_origin, beam_origin, pattern=driver._pattern)[0]
        negative = driver._assemble(
            minus, 1., shell_origin, beam_origin, pattern=driver._pattern)[0]
        observed = (positive-negative)/(2*step)
        for left, right in ((0, owner._shell_count),
                            (owner._shell_count, count),
                            (count, len(direction))):
            relative = (np.linalg.norm((observed-predicted)[left:right]) /
                        np.linalg.norm(predicted[left:right]))
            assert relative < 1e-5
    saved = owner.checkpoint()
    assert saved.startswith(BINARY_CHECKPOINT_MAGIC)
    restored_owner = ExperimentalGEAnalysis(definition)
    restored = restored_owner.restore(saved)
    assert restored.beam_station_accumulated == accepted.beam_station_accumulated
    np.testing.assert_array_equal(restored.beam_positions,
                                  accepted.beam_positions)
    token = CancellationToken()
    token.cancel("plastic prefix cancellation")
    with pytest.raises(SolveCancelled, match="plastic prefix cancellation"):
        owner.solve(_actions(), (1.1,), cancellation_token=token)
    assert owner.checkpoint() == saved

    state = driver._state
    plastic, accumulated = finite_experimental._material_arrays(
        driver._beam_origins)
    forged_plastic = plastic.copy()
    forged_plastic[-1, 0] = (0.5, 0.)
    forged = encode_binary_checkpoint(
        "finite_material", owner._identity, accepted.generation,
        (("q", state.q), ("frames", state.frames),
         ("cells", state.cells), ("resultants", state.resultants),
         ("multipliers", state.multipliers), ("pattern", driver._pattern),
         ("accepted_factors", np.asarray(accepted.accepted_factors)),
         ("beam_plastic", forged_plastic),
         ("beam_accumulated", accumulated)))
    with pytest.raises(ValueError, match="mechanically inconsistent"):
        ExperimentalGEAnalysis(definition).restore(forged)

    original_assemble = finite_experimental.FinitePointDriver._assemble

    def fail_trial(self, state, factor, *args, **kwargs):
        if factor == 1.1:
            raise RuntimeError("fixture material trial failure")
        return original_assemble(self, state, factor, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(finite_experimental.FinitePointDriver,
                        "_assemble", fail_trial)
        with pytest.raises(RuntimeError, match="material trial failure"):
            owner.solve(_actions(), (1.1,))
    assert owner.checkpoint() == saved
    assert owner.recover() is accepted

    uninterrupted = owner.solve(_actions(), (1.1,), max_iterations=20,
                                max_backtracks=12)
    continued = restored_owner.solve(_actions(), (1.1,), max_iterations=20,
                                     max_backtracks=12)
    assert uninterrupted.accepted_factors == continued.accepted_factors == (
        1., 1.1)
    assert (uninterrupted.beam_station_accumulated ==
            continued.beam_station_accumulated)
    np.testing.assert_array_equal(uninterrupted.beam_positions,
                                  continued.beam_positions)
    np.testing.assert_array_equal(uninterrupted.cell_resultants,
                                  continued.cell_resultants)


def test_generalized_beam_unload_reload_preserves_history_across_restart():
    definition = _plastic_panel()
    actions = _actions()
    uninterrupted = ExperimentalGEAnalysis(definition)
    loaded = uninterrupted.solve(actions, (1.,), max_iterations=20,
                                  max_backtracks=12)
    restored_owner = ExperimentalGEAnalysis(definition)
    restored_owner.restore(uninterrupted.checkpoint())
    targets = (.7, .4, .2, .1, .5, 1.1)
    for factor in targets:
        try:
            direct = uninterrupted.solve(actions, (factor,), max_iterations=20,
                                         max_backtracks=12)
            resumed = restored_owner.solve(actions, (factor,), max_iterations=20,
                                           max_backtracks=12)
        except RuntimeError as exc:
            pytest.fail(f"unload/reload target {factor}: {exc}")
        assert direct.free_equilibrium_ratio <= 1e-8
        assert resumed.free_equilibrium_ratio <= 1e-8
    assert direct.accepted_factors == resumed.accepted_factors == (1.,)+targets
    assert direct.beam_station_accumulated == resumed.beam_station_accumulated
    assert max(value for member in direct.beam_station_accumulated
               for value in member) >= max(
                   value for member in loaded.beam_station_accumulated
                   for value in member)
    np.testing.assert_array_equal(direct.beam_positions,
                                  resumed.beam_positions)
    np.testing.assert_array_equal(direct.cell_resultants,
                                  resumed.cell_resultants)
    final_checkpoint = uninterrupted.checkpoint()
    reopened = ExperimentalGEAnalysis(definition).restore(final_checkpoint)
    assert reopened.accepted_factors == direct.accepted_factors
    assert reopened.beam_station_accumulated == direct.beam_station_accumulated


def test_elastic_unload_uses_new_checkpoint_without_changing_legacy_format(monkeypatch):
    # Deliberately start the historical arithmetic. Fresh flat-Q4 point solves
    # now use a separate checkpoint-bound stable numerical policy.
    original_init = finite_experimental.FinitePointDriver.__init__
    def legacy_init(self, owner):
        original_init(self, owner)
        self._stable_q4 = False
    monkeypatch.setattr(finite_experimental.FinitePointDriver, "__init__", legacy_init)
    analysis = ExperimentalGEAnalysis(_panel())
    assert "nonpositive finite load factors" in analysis.capabilities["excluded"]
    assert "nonmonotone loads" not in analysis.capabilities["excluded"]
    analysis.solve(_actions(), (.5, 1.))
    legacy = analysis.checkpoint()
    assert not legacy.startswith(BINARY_CHECKPOINT_MAGIC)
    unloaded = analysis.solve(_actions(), (.5,))
    assert unloaded.accepted_factors == (.5, 1., .5)
    saved = analysis.checkpoint()
    assert saved.startswith(BINARY_CHECKPOINT_MAGIC)
    restored_owner = ExperimentalGEAnalysis(_panel())
    restored = restored_owner.restore(saved)
    assert restored.accepted_factors == unloaded.accepted_factors
    assert all(value == 0. for member in restored.beam_station_accumulated
               for value in member)
    np.testing.assert_array_equal(restored.beam_positions,
                                  unloaded.beam_positions)
    with pytest.raises(ValueError, match="positive finite load factors"):
        analysis.solve(_actions(), (0.,))
    assert analysis.recover() is unloaded


def test_finite_tiny_load_cannot_accept_zero_response(monkeypatch):
    analysis = ExperimentalGEAnalysis(_panel())
    tiny = (NodalAction("beam", 105, (1e-12, 0., 0., 0., 0., 0.)),)
    attempts = []

    def zero_step(_matrix, rhs):
        attempts.append(True)
        return np.zeros_like(rhs)

    monkeypatch.setattr(finite_experimental.spla, "spsolve", zero_step)
    with pytest.raises(RuntimeError, match="backtrack budget exhausted"):
        analysis.solve(tiny, (1.,))
    assert attempts
    assert analysis._generation == 0 and analysis._finite_driver is None


def test_finite_point_panel_accepts_two_complete_elastic_increments():
    analysis = ExperimentalGEAnalysis(_panel())
    result = analysis.solve(_actions(), (.5, 1.))
    assert result.status == "completed"
    assert result.generation == 2 and result.load_factor == 1.
    assert result.free_equilibrium_ratio <= 1e-8
    assert result.constraint_max <= 1e-9
    assert result.beam_positions[-1, 2] > 0.
    np.testing.assert_allclose(
        result.support_reactions[:analysis._shell_count].reshape(-1, 6)[:, 2].sum(),
        -1000., rtol=1e-7, atol=1e-5)
    assert not result.beam_frames.flags.writeable
    with pytest.raises(ValueError):
        result.beam_frames.setflags(write=True)
    assert analysis.recover() is result
    saved = analysis.checkpoint()
    resumed = ExperimentalGEAnalysis(_panel())
    restored = resumed.restore(saved)
    np.testing.assert_allclose(restored.shell_displacement,
                               result.shell_displacement, rtol=0., atol=0.)
    np.testing.assert_allclose(restored.beam_positions,
                               result.beam_positions, rtol=0., atol=0.)
    assert restored.accepted_factors == (.5, 1.)
    continued = resumed.solve(_actions(), (2.,))
    assert continued.generation == 3
    assert continued.accepted_factors == (.5, 1., 2.)
    assert continued.free_equilibrium_ratio <= 1e-8

    damaged = bytearray(saved)
    damaged[-20] = ord("9")
    fresh = ExperimentalGEAnalysis(_panel())
    with pytest.raises(ValueError):
        fresh.restore(bytes(damaged))
    assert fresh._generation == 0 and fresh._finite_driver is None


def test_binary_checkpoint_restores_finite_prefix_and_continues(monkeypatch):
    analysis = ExperimentalGEAnalysis(_panel())
    accepted = analysis.solve(_actions(), (.5, 1.))
    monkeypatch.setattr(finite_experimental, "MAX_JSON_CHECKPOINT_VALUES", 0)
    raw = analysis.checkpoint()
    assert raw.startswith(BINARY_CHECKPOINT_MAGIC)
    resumed = ExperimentalGEAnalysis(_panel())
    restored = resumed.restore(raw)
    assert restored.accepted_factors == accepted.accepted_factors
    np.testing.assert_array_equal(restored.shell_displacement,
                                  accepted.shell_displacement)
    np.testing.assert_array_equal(restored.cell_resultants,
                                  accepted.cell_resultants)
    continued = resumed.solve(_actions(), (2.,))
    assert continued.accepted_factors == (.5, 1., 2.)
    assert continued.free_equilibrium_ratio <= 1e-8
    damaged = bytearray(raw)
    damaged[-35] ^= 1
    fresh = ExperimentalGEAnalysis(_panel())
    with pytest.raises(ValueError, match="digest"):
        fresh.restore(bytes(damaged))
    assert fresh._generation == 0 and fresh._finite_driver is None
    driver = analysis._finite_driver
    extreme = driver._pattern.copy()
    extreme[analysis._free[0]] = 1.e308
    state = driver._state
    forged = encode_binary_checkpoint(
        "finite", analysis._identity, accepted.generation,
        (("q", state.q), ("frames", state.frames), ("cells", state.cells),
         ("resultants", state.resultants),
         ("multipliers", state.multipliers), ("pattern", extreme),
         ("accepted_factors", np.asarray(accepted.accepted_factors))))
    with pytest.raises(ValueError, match="mechanically inconsistent"):
        fresh.restore(forged)
    assert fresh._generation == 0 and fresh._finite_driver is None


def test_rejected_and_cancelled_finite_trials_keep_accepted_prefix():
    analysis = ExperimentalGEAnalysis(_panel())
    accepted = analysis.solve(_actions(), (.5,))
    with pytest.raises(RuntimeError, match="iteration limit"):
        analysis.solve(_actions(), (1.,), max_iterations=1)
    assert analysis.recover() is accepted
    assert analysis.recover().generation == 1
    token = CancellationToken()
    token.cancel("fixture cancellation")
    with pytest.raises(SolveCancelled, match="fixture cancellation"):
        analysis.solve(_actions(), (1.,), cancellation_token=token)
    assert analysis.recover() is accepted
    completed = analysis.solve(_actions(), (1.,))
    assert completed.generation == 2


def test_failed_first_finite_attempt_does_not_select_mode_or_load_pattern():
    analysis = ExperimentalGEAnalysis(_panel())
    other = (NodalAction("beam", 105, (1000., 0., 0., 0., 0., 0.)),)
    token = CancellationToken()
    token.cancel("fixture cancellation")
    with pytest.raises(SolveCancelled):
        analysis.solve(other, (1.,), cancellation_token=token)
    assert analysis._generation == 0 and analysis._finite_driver is None
    with pytest.raises(RuntimeError, match="iteration limit"):
        analysis.solve(other, (1.,), max_iterations=1)
    assert analysis._generation == 0 and analysis._finite_driver is None
    accepted = analysis.solve(_actions(), (.5, 1.))
    assert accepted.generation == 2


def test_rotation_chart_limit_uses_backtracking_before_exhaustion(monkeypatch):
    analysis = ExperimentalGEAnalysis(_panel())
    from anyfesolver._stiffened_shell_ge_finite_experimental import (
        FinitePointDriver, _RotationStepTooLarge,
    )
    original = FinitePointDriver._advance
    rejected = []

    def chart_limited(self, state, increment, amount):
        if not rejected and amount == 1.:
            rejected.append(True)
            raise _RotationStepTooLarge("finite beam rotation increment requires cutback")
        return original(self, state, increment, amount)

    monkeypatch.setattr(FinitePointDriver, "_advance", chart_limited)
    result = analysis.solve(_actions(), (.5, 1.))
    assert rejected and result.generation == 2
    assert result.free_equilibrium_ratio <= 1e-8


def test_partial_first_solve_keeps_accepted_prefix(monkeypatch):
    analysis = ExperimentalGEAnalysis(_panel())
    from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
    original = FinitePointDriver._assemble

    def fail_second(self, state, factor, *args, **kwargs):
        if factor == 1.:
            raise RuntimeError("fixture second-step failure")
        return original(self, state, factor, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(FinitePointDriver, "_assemble", fail_second)
        with pytest.raises(RuntimeError, match="second-step failure"):
            analysis.solve(_actions(), (.5, 1.))
    accepted = analysis.recover()
    assert accepted.generation == 1 and accepted.accepted_factors == (.5,)
    resumed = analysis.solve(_actions(), (1.,))
    assert resumed.generation == 2


@pytest.mark.parametrize("q4,s3", [
    (((1, 2, 5, 4),), ((2, 3, 6), (2, 6, 5))),
    ((), ((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5))),
])
def test_finite_mixed_and_s3_panels_restart_and_continue(q4, s3):
    base = _panel()
    definition = replace(base, shell=replace(
        base.shell, q4_elements=q4, s3_elements=s3))
    analysis = ExperimentalGEAnalysis(definition)
    first = analysis.solve(_actions(), (.5, 1.))
    assert first.generation == 2 and first.accepted_factors == (.5, 1.)
    assert first.free_equilibrium_ratio <= 1e-8
    assert first.constraint_max <= 1e-9
    support = first.support_reactions[:analysis._shell_count].reshape(-1, 6)
    np.testing.assert_allclose(support[:, 2].sum(), -1000., atol=1e-5)
    resumed = ExperimentalGEAnalysis(definition)
    restored = resumed.restore(analysis.checkpoint())
    np.testing.assert_array_equal(restored.shell_displacement,
                                  first.shell_displacement)
    np.testing.assert_array_equal(restored.beam_positions,
                                  first.beam_positions)
    continued = resumed.solve(_actions(), (2.,))
    assert continued.generation == 3
    assert continued.free_equilibrium_ratio <= 1e-8
