"""Development controls for the finite-area displacement candidate; unqualified."""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_displacement_checkpoint import (
    decode_snapshot, encode_snapshot, DISTRIBUTING_KIND, DISTRIBUTING_POLICY,
)
from anyfesolver._stiffened_shell_displacement_control import prepare, publish
from test_stiffened_shell_ge_distributing_finite_public import definition, actions


CONTROL = ss.DisplacementControl("beam", 103, 2)
KW = dict(displacement_control=CONTROL, max_iterations=20, max_backtracks=12)


def _initial(owner, component=2):
    return float(owner._finite_driver._state.q[
        owner._shell_count+6*owner._beam_index[103]+component])


@pytest.fixture(scope="module")
def seed():
    model = definition()
    owner = ss.create_analysis(definition=model)
    try:
        owner.solve(actions(), (.25,))
        return model, owner.checkpoint(), _initial(owner)
    finally:
        owner.close()


def _restored(seed):
    model, raw, _ = seed
    owner = ss.create_analysis(definition=model)
    owner.restore(raw)
    return owner


def _assert_physical(owner, result, *, observer_copy=False):
    assert isinstance(result, ss.DisplacementResult)
    assert result.production_qualified is False
    assert result.free_equilibrium_ratio <= 1e-8
    assert result.constraint_max <= 1e-9
    record, = result.distributing_footprint_wrenches
    assert record.generation == result.generation
    assert record.configuration == "current"
    selected, = owner.recover(selection=
        ss.GEReferenceDistributingFootprintSelection((record.footprint_id,)))
    assert selected.generation == record.generation == owner.recover().generation
    for name in ("footprint_id", "model_identity", "compiled_identity", "configuration"):
        assert getattr(selected, name) == getattr(record, name)
    for name in ("centroid_m", "centroid_low_m", "beam_position_m", "beam_position_low_m",
                 "shell_node_positions_m", "shell_node_position_low_m", "shell_node_actions",
                 "shell_action", "beam_action", "gap"):
        np.testing.assert_array_equal(getattr(selected, name), getattr(record, name))
    np.testing.assert_allclose(record.shell_action[:3]+record.beam_action[:3],
                               0., rtol=0., atol=1e-8)
    arm = ((record.beam_position_m-record.centroid_m)+
           (record.beam_position_low_m-record.centroid_low_m))
    np.testing.assert_allclose(record.shell_action[3:]+record.beam_action[3:]+
        np.cross(arm, record.beam_action[:3]), 0., rtol=0., atol=1e-8)
    if not observer_copy:
        for value in (result.shell_displacement, result.applied_dead_load,
                      record.shell_node_actions, record.gap):
            with pytest.raises(ValueError):
                value.setflags(write=True)


def test_signed_restart_continue_and_physical_recovery(seed):
    owner = _restored(seed)
    resumed = ss.create_analysis(definition=seed[0])
    direct = _restored(seed)
    initial = seed[2]
    targets = (initial*1.02, initial*.98, -initial*.2)
    try:
        first = owner.solve(actions(), targets[:2], **KW)
        _assert_physical(owner, first)
        raw = owner.checkpoint()
        metadata, arrays = decode_snapshot(owner, raw)
        assert metadata["policy"] == DISTRIBUTING_POLICY
        assert owner.capabilities["checkpoints"][-1] == DISTRIBUTING_KIND
        assert "shell_plastic" not in arrays
        assert tuple(arrays["history"][:, 0]) == targets[:2]
        np.testing.assert_array_equal(arrays["base_load"],
            .25*owner._finite_driver._load(actions()))
        np.testing.assert_array_equal(first.applied_dead_load,
            arrays["base_load"]+first.amplitude*arrays["load_direction"])
        restored = resumed.restore(raw)
        _assert_physical(resumed, restored)
        assert resumed.checkpoint() == raw
        a = resumed.solve(actions(), targets[2:], **KW)
        b = direct.solve(actions(), targets, **KW)
        _assert_physical(resumed, a)
        for name in ("shell_displacement", "cell_resultants", "multipliers",
                     "support_reactions", "applied_dead_load"):
            np.testing.assert_allclose(getattr(a, name), getattr(b, name),
                                       rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(a.distributing_footprint_wrenches[0].shell_action,
            b.distributing_footprint_wrenches[0].shell_action, rtol=1e-10, atol=1e-9)
    finally:
        owner.close(); resumed.close(); direct.close()


def test_cancelled_prefix_observer_failure_and_stale_publication(seed):
    owner = _restored(seed)
    token = CancellationToken()
    initial = seed[2]
    observed = []
    def observe(result):
        observed.append(owner.checkpoint())
        _assert_physical(owner, result, observer_copy=True)
        accepted = owner.recover()
        accepted_record, = accepted.distributing_footprint_wrenches
        copied_record, = result.distributing_footprint_wrenches
        # The observer receives deepcopy(result), whose NumPy arrays can be
        # writable. Mutating that detached copy must leave live recovery and
        # its complete published checkpoint unchanged.
        for name in ("shell_displacement", "applied_dead_load", "cell_resultants"):
            saved = getattr(accepted, name).copy()
            copied = getattr(result, name)
            assert not np.shares_memory(copied, getattr(accepted, name))
            copied.setflags(write=True)
            copied.flat[0] += 1.
            np.testing.assert_array_equal(getattr(owner.recover(), name), saved)
        for name in ("shell_node_actions", "shell_action", "gap"):
            saved = getattr(accepted_record, name).copy()
            copied = getattr(copied_record, name)
            assert not np.shares_memory(copied, getattr(accepted_record, name))
            copied.setflags(write=True)
            copied.flat[0] += 1.
            selected, = owner.recover(selection=
                ss.GEReferenceDistributingFootprintSelection((accepted_record.footprint_id,)))
            assert selected.generation == accepted.generation
            np.testing.assert_array_equal(getattr(selected, name), saved)
        assert owner.checkpoint() == observed[-1]
        _assert_physical(owner, owner.recover())
        token.cancel("accepted displacement prefix")
    try:
        with pytest.raises(SolveCancelled):
            owner.solve(actions(), (initial*1.01, initial*1.02), **KW,
                        cancellation_token=token, on_accepted=observe)
        assert owner.checkpoint() == observed[0]
        assert len(owner.recover().history) == 1
        raw = owner.checkpoint()
        driver, generation = owner._finite_driver, owner._generation
        def failing(result):
            assert owner.checkpoint() != raw
            with pytest.raises(RuntimeError):
                owner.solve(actions(), (initial*1.03,), **KW)
            raise RuntimeError("observer failure after complete publication")
        with pytest.raises(RuntimeError, match="observer failure"):
            owner.solve(actions(), (initial*1.02,), **KW, on_accepted=failing)
        accepted = owner.checkpoint()
        with pytest.raises(ValueError, match="stale"):
            publish(owner, raw, expected_driver=driver, expected_generation=generation)
        assert owner.checkpoint() == accepted
        assert len(owner.recover().history) == 2
    finally:
        owner.close()


@pytest.mark.parametrize("operation", ("adaptive", "event", "chart", "arc", "moment",
                                         "restrained", "duplicate", "force"))
def test_refusals_preserve_seed_or_prefix(seed, operation):
    owner = _restored(seed)
    try:
        before = owner.checkpoint()
        initial = seed[2]
        with pytest.raises(ValueError):
            if operation == "adaptive":
                owner.solve(actions(), (initial*1.01,), **KW,
                            increment_control=ss.AdaptiveIncrementControl())
            elif operation == "event":
                owner.solve(actions(), (initial*1.01,),
                    displacement_control=replace(CONTROL, trial_policy=ss.GENERALIZED_BRANCH_EVENT_TRIAL))
            elif operation == "chart":
                owner.rebase_rotations(owner._generation)
            elif operation == "arc":
                owner.continue_arc_length(actions(), object())
            elif operation == "moment":
                owner.solve((ss.NodalAction("beam",103,(0.,0.,0.,0.,1.,0.)),),
                            (initial*1.01,), **KW)
            elif operation == "restrained":
                owner.solve(actions(), (initial*1.01,),
                            displacement_control=ss.DisplacementControl("shell",1,2))
            elif operation == "duplicate":
                owner.solve(actions(), (initial,), **KW)
            else:
                owner.solve(actions(), (initial*1.01,), **KW)
                before = owner.checkpoint()
                owner.solve(actions(), (.5,))
        assert owner.checkpoint() == before
    finally:
        owner.close()


def test_no_implicit_seed_and_reference_records_stay_reference_only():
    from test_stiffened_shell_ge_distributing_reference import _definition, _selection
    finite = ss.create_analysis(definition=definition())
    reference = ss.create_analysis(definition=replace(_definition(), points=(),
        distributing_footprints=(_selection(),)))
    try:
        for owner in (finite, reference):
            with pytest.raises(ValueError):
                owner.solve(actions(), (1e-6,), **KW)
            assert owner._generation == 0 and owner._finite_driver is None
    finally:
        finite.close(); reference.close()


def test_canonical_but_inconsistent_endpoint_is_not_published(seed):
    owner = _restored(seed)
    fresh = ss.create_analysis(definition=seed[0])
    try:
        owner.solve(actions(), (seed[2]*1.01,), **KW)
        raw = owner.checkpoint()
        metadata, arrays = decode_snapshot(owner, raw)
        changed = dict(arrays)
        changed["resultants"] = arrays["resultants"].copy()
        changed["resultants"][0, 0] += 1e6
        corrupt = encode_snapshot(owner, generation=metadata["generation"],
            seed_generation=metadata["seed_generation"], seed_sha256=metadata["seed_sha256"],
            control_dof=metadata["control_dof"], arrays=changed)
        with pytest.raises(ValueError, match="equilibrium"):
            fresh.restore(corrupt)
        assert fresh._generation == 0 and fresh._finite_driver is None
        assert owner.checkpoint() == raw
    finally:
        owner.close(); fresh.close()


def test_cancel_before_publication_and_domain_failure_preserve_seed(seed, monkeypatch):
    import anyfesolver._stiffened_shell_displacement_control as controller
    from anyfesolver._stiffened_shell_displacement_corrector import DisplacementCorrectorFailure
    from anyfesolver._stiffened_shell_distributing_pose import DistributingPoseDomainError
    owner = _restored(seed)
    token = CancellationToken()
    original = controller.prepare
    def cancel_after_prepare(*args):
        prepared = original(*args)
        token.cancel("before displacement publication")
        return prepared
    try:
        before = owner.checkpoint()
        with monkeypatch.context() as patch:
            patch.setattr(controller, "prepare", cancel_after_prepare)
            with pytest.raises(SolveCancelled):
                owner.solve(actions(), (seed[2]*1.01,), **KW, cancellation_token=token)
        assert owner.checkpoint() == before
        driver = owner._finite_driver
        def bad_pose(*args):
            raise DistributingPoseDomainError("bounded candidate outside pose domain")
        with monkeypatch.context() as patch:
            patch.setattr(driver, "_advance", bad_pose)
            with pytest.raises(DisplacementCorrectorFailure, match="predictor outside domain"):
                owner.solve(actions(), (seed[2]+.01*owner._reference_length,), **KW)
        assert owner.checkpoint() == before
    finally:
        owner.close()


@pytest.mark.parametrize("profile", ("generalized", "coupled", "native", "native_coupled"))
def test_actual_material_histories_and_restart(profile, monkeypatch):
    if profile.startswith("native"):
        from test_stiffened_shell_distributing_native_material import definition as material
        model = material(coupled=profile.endswith("coupled"), mixed=True, yielding="axial")
        node = 105
        load = model.cyclic_reference_actions
    else:
        from test_stiffened_shell_ge_distributing_material import definition as material, actions as material_actions
        model = material(profile == "coupled")
        node = 103
        load = material_actions()
    owner = ss.create_analysis(definition=model)
    restored = ss.create_analysis(definition=model)
    try:
        owner.solve(load, (.25, .5, .75, 1.), max_iterations=24, max_backtracks=16)
        initial = float(owner._finite_driver._state.q[
            owner._shell_count+6*owner._beam_index[node]])
        control = ss.DisplacementControl("beam", node, 0)
        result = owner.solve(load, (initial*1.01,), displacement_control=control,
                             max_iterations=24, max_backtracks=12)
        assert result.beam_generalized_material
        if "native" in profile:
            assert result.beam_fibre_material and result.beam_fibre_energy
        if "coupled" in profile:
            assert result.shell_q4_material and np.max(result.shell_layer_alpha) > 0.
        raw = owner.checkpoint()
        replay = restored.restore(raw)
        assert restored.checkpoint() == raw
        assert replay.distributing_footprint_wrenches[0].generation == replay.generation
        np.testing.assert_array_equal(replay.cell_resultants, result.cell_resultants)
        np.testing.assert_array_equal(replay.shell_layer_alpha, result.shell_layer_alpha)
        _, arrays = decode_snapshot(owner, raw)
        for name, value in arrays.items():
            with pytest.raises(ValueError):
                value.setflags(write=True)
        if "native" in profile:
            for record in result.beam_fibre_material:
                for value in (record.strain, record.strain_low, record.resultants, record.resultants_low):
                    with pytest.raises(ValueError):
                        value.setflags(write=True)
            wrong = tuple(replace(action, wrench=tuple(2.*v for v in action.wrench)) for action in load)
            with pytest.raises(ValueError, match="model-bound reference load direction"):
                restored.solve(wrong, (initial*1.02,), displacement_control=control)
            assert restored.checkpoint() == raw
            metadata, original_arrays = decode_snapshot(owner, raw)
            changed = dict(original_arrays)
            changed["load_direction"] = 2.*original_arrays["load_direction"]
            amplitude = float(original_arrays["history"][-1, 1])
            changed["base_load"] = original_arrays["applied_load"]-amplitude*changed["load_direction"]
            changed["applied_load"] = changed["base_load"]+amplitude*changed["load_direction"]
            # Cancellation-free preparation must reject a canonically framed
            # endpoint whose affine sum is internally consistent.
            forged = encode_snapshot(owner, generation=metadata["generation"],
                seed_generation=metadata["seed_generation"], seed_sha256=metadata["seed_sha256"],
                control_dof=metadata["control_dof"], arrays=changed)
            with pytest.raises(ValueError, match="model-bound reference load direction"):
                prepare(restored, forged)
            assert restored.checkpoint() == raw
            assert original_arrays["cyclic_seed_factor"][0] == 1.
            changed = dict(original_arrays)
            changed["base_load"] = original_arrays["base_load"].copy()
            restrained = next(int(slot) for slot in owner._fixed if slot % 6 < 3)
            changed["base_load"][restrained] += 1.
            changed["applied_load"] = changed["base_load"]+amplitude*changed["load_direction"]
            forged_base = encode_snapshot(owner, generation=metadata["generation"],
                seed_generation=metadata["seed_generation"], seed_sha256=metadata["seed_sha256"],
                control_dof=metadata["control_dof"], arrays=changed)
            with pytest.raises(ValueError, match="stored seed force factor"):
                prepare(restored, forged_base)
            assert restored.checkpoint() == raw
        import anyfesolver._stiffened_shell_displacement_control as controller
        original_prepare = controller.prepare
        token = CancellationToken()
        def cancel_after_material_prepare(*args):
            prepared = original_prepare(*args)
            token.cancel("material candidate prepared before publication")
            return prepared
        with monkeypatch.context() as patch:
            patch.setattr(controller, "prepare", cancel_after_material_prepare)
            with pytest.raises(SolveCancelled):
                restored.solve(load, (initial*1.02,), displacement_control=control,
                    max_iterations=24, max_backtracks=12, cancellation_token=token)
        assert restored.checkpoint() == raw
        a = restored.solve(load, (initial*1.02,), displacement_control=control,
                           max_iterations=24, max_backtracks=12)
        b = owner.solve(load, (initial*1.02,), displacement_control=control,
                        max_iterations=24, max_backtracks=12)
        np.testing.assert_array_equal(a.cell_resultants, b.cell_resultants)
        np.testing.assert_array_equal(a.shell_layer_alpha, b.shell_layer_alpha)
        assert a.distributing_footprint_wrenches[0].generation == a.generation
        if "native" in profile:
            _, continued = decode_snapshot(restored, restored.checkpoint())
            np.testing.assert_array_equal(continued["cyclic_seed_factor"], arrays["cyclic_seed_factor"])
    finally:
        owner.close(); restored.close()
