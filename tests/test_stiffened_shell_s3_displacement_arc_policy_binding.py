"""D5/arc numerical-policy identity binding for the generalized S3 point owner.

Displacement (D5) and arc snapshots of the plain elastic S3 generalized POINT
owner carry their live S3 arithmetic in a private policy identity: the new
generalized stable selection frames new identities, while restored legacy
histories keep the exact old strings, bytes and arithmetic. Replay binds the
arithmetic from the snapshot's declared policy identity, never from the fresh
constructor selection, so a restored legacy False history can never upgrade;
new-policy false/true mismatches are refused atomically before any state is
published. No new structural checkpoint format: kind, magic, layout, codec
and metadata key sets are unchanged; only the policy identity differs.
"""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import stiffened_shell as public
from anyfesolver import _stiffened_shell_ge_finite_experimental as finite_module
from anyfesolver._stiffened_shell_displacement_checkpoint import (
    ELASTIC_S3_CHART_POLICY, ELASTIC_S3_POLICY,
    STABLE_ELASTIC_S3_CHART_POLICY, STABLE_ELASTIC_S3_POLICY,
    admitted_s3_policies, decode_snapshot as decode_displacement,
    encode_snapshot as encode_displacement, generalized_stable_live,
    identities as displacement_identities, replay_s3_stable,
)
from anyfesolver._stiffened_shell_arc_checkpoint import (
    _replay_s3_stable, _s3_arc_policies, decode_snapshot as decode_arc,
)
from test_stiffened_shell_ge_experimental import _panel
from test_stiffened_shell_ge_physical_fibre_linear import _definition

ACTIONS = (public.NodalAction('beam', 105, (-1000., 0., 100., 0., 0., 0.)),)
MILD_ACTIONS = (public.NodalAction('beam', 105, (0., 0., 100., 0., 0., 0.)),)
ZERO_ACTIONS = (public.NodalAction('beam', 105, (0., 0., 0., 0., 0., 0.)),)
TARGETS = (.5, 1.)
BASE_KEYS = {'policy', 'model', 'generation', 'seed_generation',
             'seed_sha256', 'control_dof', 'steps'}
LEGACY_ARC_V1, LEGACY_ARC_V2, STABLE_ARC_V2 = _s3_arc_policies()
NATIVE_ARC_CHART_POLICY = 'B3_GE_NATIVE_POINT_ARC_FIXED_SHELL_CHART_V1_DEV'


def _s3_generalized_definition():
    """Matching plain S3 generalized point fixture (the accepted_owner1
    ('s3','generalized') model of the mixed-buckling suite)."""
    return replace(_definition(mixed=False, triangles=True), beams=_panel().beams)


def _q4_generalized_definition():
    return replace(_definition(mixed=False, triangles=False), beams=_panel().beams,
                   q4_pullback_policy=public.Q4_ALGORITHMIC_PULLBACK_V1)


def _displacement_control():
    return public.DisplacementControl('beam', 105, 2)


def _arc_control(owner, radius):
    count = len(owner.preflight()['arc_metric_node_order'])
    metric = public.PhysicalChordMetric(np.full(count, 1./count), owner._reference_length)
    return public.ArcLengthControl('beam', 105, 2, metric, radius, 1)


def _stable_owner(actions=ACTIONS):
    owner = public.create_analysis(definition=_s3_generalized_definition())
    result = owner.solve(actions, targets=TARGETS, max_iterations=24, max_backtracks=12)
    assert result.status == 'completed'
    assert owner._finite_driver._stable_s3 is True
    return owner


def _displacement_step(owner, actions, control, delta):
    dof = control.dof(owner)
    target = float(owner._finite_driver._state.q[dof]) + delta*owner._reference_length
    result = owner.solve(actions, (target,), displacement_control=control,
                         max_iterations=24, max_backtracks=12)
    assert result.status == 'completed'
    return target, result


def test_stable_displacement_restore_continuation_binds_new_identity():
    owner = _stable_owner()
    resumed = None
    try:
        control = _displacement_control()
        target, result = _displacement_step(owner, ACTIONS, control, 1e-3)
        raw, metadata, arrays = owner._displacement_snapshot
        assert metadata['policy'] == STABLE_ELASTIC_S3_POLICY
        assert set(metadata) == BASE_KEYS
        m, a = decode_displacement(owner, raw)
        assert m == metadata
        rebuilt = encode_displacement(owner, generation=m['generation'],
            seed_generation=m['seed_generation'], seed_sha256=m['seed_sha256'],
            control_dof=m['control_dof'], arrays=a)
        assert rebuilt == raw
        # Actual public restore-continuation: a fresh owner replays the
        # declared stable arithmetic and continues in the same identity.
        resumed = public.create_analysis(definition=owner._definition)
        repeated = resumed.restore(raw)
        assert resumed._finite_driver._stable_s3 is True
        assert resumed.checkpoint() == raw
        np.testing.assert_array_equal(repeated.cell_resultants, result.cell_resultants)
        continued = _displacement_step(resumed, ACTIONS, control, 1e-3)[1]
        assert resumed._displacement_snapshot[1]['policy'] == STABLE_ELASTIC_S3_POLICY
        assert resumed._finite_driver._stable_s3 is True
        oracle = _displacement_step(owner, ACTIONS, control, 1e-3)[1]
        np.testing.assert_array_equal(continued.cell_resultants, oracle.cell_resultants)
    finally:
        if resumed is not None:
            resumed.close()
        owner.close()


def test_legacy_displacement_identity_bytes_and_arithmetic_are_preserved(monkeypatch):
    definition = _s3_generalized_definition()
    monkeypatch.setattr(finite_module, 'supports_generalized_stable_s3',
                        lambda owner: False)
    owner = public.create_analysis(definition=definition)
    resumed = None
    try:
        seed = owner.solve(ZERO_ACTIONS, targets=TARGETS,
                           max_iterations=24, max_backtracks=12)
        assert seed.status == 'completed'
        assert owner._finite_driver._stable_s3 is False
        control = _displacement_control()
        target, result = _displacement_step(owner, MILD_ACTIONS, control, 1e-3)
        raw, metadata, arrays = owner._displacement_snapshot
        # The legacy identity and metadata contract are exactly the old ones.
        assert metadata['policy'] == ELASTIC_S3_POLICY
        assert set(metadata) == BASE_KEYS
        m, a = decode_displacement(owner, raw)
        rebuilt = encode_displacement(owner, generation=m['generation'],
            seed_generation=m['seed_generation'], seed_sha256=m['seed_sha256'],
            control_dof=m['control_dof'], arrays=a)
        assert rebuilt == raw
        monkeypatch.undo()
        # A fresh new-policy owner restores the legacy snapshot: the replay
        # arithmetic comes from the old policy identity, never from the new
        # constructor selection, and continuation keeps the old identity.
        resumed = public.create_analysis(definition=definition)
        repeated = resumed.restore(raw)
        assert resumed._finite_driver._stable_s3 is False
        assert resumed.checkpoint() == raw
        np.testing.assert_array_equal(repeated.cell_resultants, result.cell_resultants)
        continued = _displacement_step(resumed, MILD_ACTIONS, control, 1e-3)[1]
        assert resumed._displacement_snapshot[1]['policy'] == ELASTIC_S3_POLICY
        assert resumed._finite_driver._stable_s3 is False
        oracle = _displacement_step(owner, MILD_ACTIONS, control, 1e-3)[1]
        np.testing.assert_array_equal(continued.cell_resultants, oracle.cell_resultants)
    finally:
        if resumed is not None:
            resumed.close()
        owner.close()


def test_stable_arc_restore_continuation_binds_new_identity():
    owner = _stable_owner(actions=MILD_ACTIONS)
    resumed = None
    try:
        control = _arc_control(owner, 1e-3*owner._reference_length)
        result = owner.continue_arc_length(MILD_ACTIONS, control, steps=1)
        raw, metadata, arrays = owner._arc_snapshot
        assert metadata['policy'] == STABLE_ARC_V2
        assert metadata['native_chart_policy'] == NATIVE_ARC_CHART_POLICY
        m, a = decode_arc(owner, raw)
        assert m == metadata
        resumed = public.create_analysis(definition=owner._definition)
        repeated = resumed.restore(raw)
        assert resumed._finite_driver._stable_s3 is True
        assert resumed.checkpoint() == raw
        np.testing.assert_array_equal(repeated.cell_resultants, result.cell_resultants)
        continued = resumed.continue_arc_length(MILD_ACTIONS, control, steps=1)
        assert resumed._arc_snapshot[1]['policy'] == STABLE_ARC_V2
        assert resumed._finite_driver._stable_s3 is True
        oracle = owner.continue_arc_length(MILD_ACTIONS, control, steps=1)
        np.testing.assert_array_equal(continued.cell_resultants, oracle.cell_resultants)
    finally:
        if resumed is not None:
            resumed.close()
        owner.close()


def test_legacy_arc_identity_and_arithmetic_are_preserved(monkeypatch):
    definition = _s3_generalized_definition()
    monkeypatch.setattr(finite_module, 'supports_generalized_stable_s3',
                        lambda owner: False)
    owner = public.create_analysis(definition=definition)
    resumed = None
    try:
        seed = owner.solve(ZERO_ACTIONS, targets=TARGETS,
                           max_iterations=24, max_backtracks=12)
        assert seed.status == 'completed'
        assert owner._finite_driver._stable_s3 is False
        control = _arc_control(owner, 1e-3*owner._reference_length)
        result = owner.continue_arc_length(MILD_ACTIONS, control, steps=1)
        raw, metadata, arrays = owner._arc_snapshot
        assert metadata['policy'] == LEGACY_ARC_V2
        assert metadata['native_chart_policy'] == NATIVE_ARC_CHART_POLICY
        monkeypatch.undo()
        resumed = public.create_analysis(definition=definition)
        repeated = resumed.restore(raw)
        assert resumed._finite_driver._stable_s3 is False
        assert resumed.checkpoint() == raw
        np.testing.assert_array_equal(repeated.cell_resultants, result.cell_resultants)
        continued = resumed.continue_arc_length(MILD_ACTIONS, control, steps=1)
        assert resumed._arc_snapshot[1]['policy'] == LEGACY_ARC_V2
        assert resumed._finite_driver._stable_s3 is False
        oracle = owner.continue_arc_length(MILD_ACTIONS, control, steps=1)
        np.testing.assert_array_equal(continued.cell_resultants, oracle.cell_resultants)
    finally:
        if resumed is not None:
            resumed.close()
        owner.close()


def test_new_policy_snapshots_are_refused_atomically_on_foreign_owners():
    displacement_owner = _stable_owner()
    arc_owner = _stable_owner(actions=MILD_ACTIONS)
    foreign = None
    arc_foreign = None
    try:
        control = _displacement_control()
        _displacement_step(displacement_owner, ACTIONS, control, 1e-3)
        displacement_raw = displacement_owner._displacement_snapshot[0]
        arc_control = _arc_control(arc_owner, 1e-3*arc_owner._reference_length)
        arc_owner.continue_arc_length(MILD_ACTIONS, arc_control, steps=1)
        arc_raw = arc_owner._arc_snapshot[0]
        foreign = public.create_analysis(definition=_q4_generalized_definition())
        with pytest.raises(ValueError, match='incompatible displacement metadata'):
            foreign.restore(displacement_raw)
        assert foreign._generation == 0 and foreign._finite_driver is None
        arc_foreign = public.create_analysis(definition=_q4_generalized_definition())
        with pytest.raises(ValueError, match='incompatible arc metadata'):
            arc_foreign.restore(arc_raw)
        assert arc_foreign._generation == 0 and arc_foreign._finite_driver is None
    finally:
        if foreign is not None:
            foreign.close()
        if arc_foreign is not None:
            arc_foreign.close()
        arc_owner.close()
        displacement_owner.close()


def test_replay_binding_guards_and_identity_classification():
    owner = _stable_owner()
    fresh = None
    foreign = None
    try:
        control = _displacement_control()
        _displacement_step(owner, ACTIONS, control, 1e-3)
        raw, metadata, arrays = owner._displacement_snapshot
        # The declared policy identity owns the replay arithmetic.
        assert admitted_s3_policies() == (
            (ELASTIC_S3_POLICY, ELASTIC_S3_CHART_POLICY, False),
            (STABLE_ELASTIC_S3_POLICY, STABLE_ELASTIC_S3_CHART_POLICY, True))
        assert replay_s3_stable(owner, ELASTIC_S3_POLICY) is False
        assert replay_s3_stable(owner, ELASTIC_S3_CHART_POLICY) is False
        assert replay_s3_stable(owner, STABLE_ELASTIC_S3_POLICY) is True
        assert replay_s3_stable(owner, STABLE_ELASTIC_S3_CHART_POLICY) is True
        with pytest.raises(ValueError, match='unbound native S3 displacement'):
            replay_s3_stable(owner, 'FOREIGN_POLICY')
        assert _replay_s3_stable(owner, LEGACY_ARC_V1) is False
        assert _replay_s3_stable(owner, LEGACY_ARC_V2) is False
        assert _replay_s3_stable(owner, STABLE_ARC_V2) is True
        with pytest.raises(ValueError, match='unbound native S3 arc'):
            _replay_s3_stable(owner, 'FOREIGN_POLICY')
        # A fresh owner without a live driver fails closed onto the legacy
        # encoding identity; the constructor selection never decides it.
        fresh = public.create_analysis(definition=owner._definition)
        assert generalized_stable_live(fresh) is False
        assert displacement_identities(fresh)[1] == ELASTIC_S3_POLICY
        # Non-domain owners are never bound by the generalized identities,
        # and false/true mismatches are refused before any replay mechanics.
        foreign = public.create_analysis(definition=_q4_generalized_definition())
        assert replay_s3_stable(foreign, STABLE_ELASTIC_S3_POLICY) is None
        assert _replay_s3_stable(foreign, STABLE_ARC_V2) is None
        from anyfesolver._stiffened_shell_affine_endpoint import validate_arrays
        with pytest.raises(ValueError, match='numerical policy mismatch'):
            validate_arrays(foreign, arrays, s3_stable=True)
        with pytest.raises(ValueError, match='numerical policy mismatch'):
            validate_arrays(owner, arrays, s3_stable=1)
    finally:
        if fresh is not None:
            fresh.close()
        if foreign is not None:
            foreign.close()
        owner.close()


def test_policy_identity_strings_pin_the_exact_byte_contracts():
    assert ELASTIC_S3_POLICY == 'B3_GE_NATIVE_POINT_ELASTIC_S3_DISPLACEMENT_V1_DEV'
    assert ELASTIC_S3_CHART_POLICY == 'B3_GE_NATIVE_POINT_ELASTIC_S3_DISPLACEMENT_CHART_V1_DEV'
    assert STABLE_ELASTIC_S3_POLICY == 'B3_GE_NATIVE_POINT_ELASTIC_S3_DISPLACEMENT_STABLE_V1_DEV'
    assert STABLE_ELASTIC_S3_CHART_POLICY == 'B3_GE_NATIVE_POINT_ELASTIC_S3_DISPLACEMENT_STABLE_CHART_V1_DEV'
    assert (LEGACY_ARC_V1, LEGACY_ARC_V2, STABLE_ARC_V2) == (
        'B3_GE_NATIVE_POINT_ELASTIC_S3_TANGENT_PLANE_V1_DEV',
        'B3_GE_NATIVE_POINT_ELASTIC_S3_TANGENT_PLANE_V2_DEV',
        'B3_GE_NATIVE_POINT_ELASTIC_S3_TANGENT_PLANE_STABLE_V2_DEV')
