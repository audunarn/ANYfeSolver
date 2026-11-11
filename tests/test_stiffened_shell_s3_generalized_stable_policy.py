"""Narrow elastic S3 stable-translation policy for the generalized point owner.

The plain S3 generalized POINT owner reuses the existing
cancellation-resistant elastic S3 deformation: same equations, tolerances and
actual derivative maps, selected privately at construction. Old checkpoints
keep their original arithmetic explicitly; the existing stable native schema,
history packing and shape validation are reused without format changes.
"""
from dataclasses import replace
import json

import numpy as np
import pytest

from anyfesolver import stiffened_shell as public
from anyfesolver import _stiffened_shell_ge_point_buckling as buckling
from anyfesolver import _stiffened_shell_ge_finite_experimental as finite_module
from anyfesolver._stiffened_shell_ge_finite_experimental import (
    FinitePointDriver, supports_generalized_stable_s3,
)
from anyfesolver._stiffened_shell_ge_checkpoint_binary import (
    SCHEMAS, STABLE_NATIVE_SCHEMA, decode, encode,
)
from anyfesolver._stiffened_shell_beam_history import shapes
from anyfesolver._ge_beam3_variational_shell import deformation, response
from test_stiffened_shell_ge_experimental import _panel
from test_stiffened_shell_ge_physical_fibre_linear import _definition
from test_stiffened_shell_ge_mixed_buckling import (
    dense_mixed_oracle,
    test_actual_update_covector_plateau as _plateau_check,
)

ACTIONS = (public.NodalAction('beam', 105, (-1000., 0., 100., 0., 0., 0.)),)
TARGETS = (.5, 1.)
MILD_ACTIONS = (public.NodalAction('beam', 105, (0., 0., 100., 0., 0., 0.)),)
ZERO_ACTIONS = (public.NodalAction('beam', 105, (0., 0., 0., 0., 0., 0.)),)
LEGACY_SCHEMA = "B3_GE_E4_PL_STIFFENED_SHELL_EXPERIMENTAL_FINITE_V1"


def _s3_generalized_definition():
    """Matching plain S3 generalized point fixture (the accepted_owner1
    ('s3','generalized') model of the mixed-buckling suite)."""
    return replace(_definition(mixed=False, triangles=True), beams=_panel().beams)


def _q4_generalized_definition():
    return replace(_definition(mixed=False, triangles=False), beams=_panel().beams,
                   q4_pullback_policy=public.Q4_ALGORITHMIC_PULLBACK_V1)


def _mixed_generalized_definition():
    model = _s3_generalized_definition()
    return replace(model, shell=replace(model.shell, q4_elements=((1, 2, 5, 4),),
                                       s3_elements=((2, 3, 6), (2, 6, 5))),
                   q4_pullback_policy=public.Q4_ALGORITHMIC_PULLBACK_V1)


def _expected_shapes(owner):
    def expected(kind, generation):
        assert kind == 'finite_native_material_stable'
        count = owner._K.shape[0]
        return (("q", (count,)), ("frames", (len(owner._beam_ids), 3, 3)),
                ("cells", (len(owner._beam_parts), 2, 3, 3)),
                ("resultants", (len(owner._beam_parts), 18)),
                ("multipliers", (owner._C.shape[0],)), ("pattern", (count,)),
                ("accepted_factors", (generation,))) + \
            shapes(owner._beam_parts, 'origin') + shapes(owner._beam_parts, 'accepted')
    return expected


@pytest.fixture
def accepted_owner():
    # Independent per-test owner: the public restore/continuation tests
    # advance the accepted load factor, and a shared module owner would make
    # the next test's same-factor continuation be refused.
    owner = public.create_analysis(definition=_s3_generalized_definition())
    result = owner.solve(ACTIONS, targets=TARGETS,
                         max_iterations=24, max_backtracks=12)
    assert result.status == 'completed'
    yield owner
    owner.close()


def test_generalized_point_owner_selects_stable_policy_and_passes_thresholds(accepted_owner):
    owner = accepted_owner
    driver = owner._finite_driver
    assert supports_generalized_stable_s3(owner)
    assert driver._stable_s3 is True
    accepted = driver._accepted
    assert accepted.generation == 2
    assert accepted.free_equilibrium_ratio <= 1e-8
    assert accepted.constraint_max <= 1e-9
    assert owner.mixed_buckling_capabilities['definition_admitted']


def test_policy_domain_is_exactly_the_plain_s3_generalized_point_owner():
    # (definition, domain predicate, constructor stable selection)
    cases = ((_s3_generalized_definition(), True, True),
             (_q4_generalized_definition(), False, False),
             (_mixed_generalized_definition(), False, False),
             (_definition(mixed=False, triangles=True), False, True))
    for definition, domain, stable in cases:
        owner = public.create_analysis(definition=definition)
        try:
            assert supports_generalized_stable_s3(owner) is domain
            assert FinitePointDriver(owner)._stable_s3 is stable
        finally:
            owner.close()


def test_displacement_seed_policy_admits_both_arithmetics_exactly(accepted_owner):
    """The exact generalized stable domain admits legacy False and new True
    seeds; a restored legacy False seed is never upgraded, and nonboolean
    flags are refused."""
    from anyfesolver._stiffened_shell_affine_endpoint import (
        require_native_displacement_policy)
    owner = accepted_owner
    driver = owner._finite_driver
    assert driver._stable_s3 is True
    require_native_displacement_policy(owner, driver)
    saved = driver._stable_s3
    try:
        driver._stable_s3 = False
        require_native_displacement_policy(owner, driver)
        assert driver._stable_s3 is False
        for forged in (np.bool_(True), 1, None):
            driver._stable_s3 = forged
            with pytest.raises(ValueError, match='numerical policy mismatch'):
                require_native_displacement_policy(owner, driver)
    finally:
        driver._stable_s3 = saved
    require_native_displacement_policy(owner, driver)


def test_displacement_seed_policy_keeps_existing_guards_for_other_owners():
    from anyfesolver._stiffened_shell_affine_endpoint import (
        require_native_displacement_policy)
    # (definition, admitted constructor seed, refused forged seed)
    cases = ((_definition(mixed=False, triangles=True), True, False),
             (_mixed_generalized_definition(), False, True))
    for definition, admitted, refused in cases:
        owner = public.create_analysis(definition=definition)
        try:
            driver = FinitePointDriver(owner)
            assert driver._stable_s3 is admitted
            require_native_displacement_policy(owner, driver)
            saved = driver._stable_s3
            driver._stable_s3 = refused
            try:
                with pytest.raises(ValueError, match='numerical policy mismatch'):
                    require_native_displacement_policy(owner, driver)
            finally:
                driver._stable_s3 = saved
        finally:
            owner.close()


def test_stable_checkpoint_restore_continuation_and_byte_roundtrip(accepted_owner):
    # Complete public contract: the reused stable native schema restores
    # through owner.restore, continues stable, and roundtrips byte-exactly.
    # The owner-level framing admission (stable_kind AND the exact
    # generalized stable domain) is applied in the expected_shapes gate.
    source = accepted_owner
    before = source.checkpoint()
    kind, generation, values = decode(before, source._identity, _expected_shapes(source))
    assert kind == 'finite_native_material_stable'
    assert SCHEMAS[kind] == STABLE_NATIVE_SCHEMA
    resumed = public.create_analysis(definition=source._definition)
    try:
        restored = resumed.restore(before)
        assert resumed._finite_driver._stable_s3 is True
        assert resumed.checkpoint() == before
        np.testing.assert_array_equal(restored.cell_resultants,
                                       source._finite_driver._accepted.cell_resultants)
        continued = resumed.solve(ACTIONS, targets=(.5,))
        assert resumed._finite_driver._stable_s3 is True
        oracle = source.solve(ACTIONS, targets=(.5,))
        np.testing.assert_array_equal(continued.cell_resultants, oracle.cell_resultants)
        assert continued.beam_station_accumulated == oracle.beam_station_accumulated
    finally:
        resumed.close()


def test_driver_restore_admits_exactly_the_stable_native_schema(accepted_owner):
    """Driver-level admission of the reused stable schema (owned policy).

    The public owner-level framing admission for this new domain is a
    reserved root decision; this test pins the finite-driver contract that
    any such admission must delegate to.
    """
    source = accepted_owner
    saved = source.checkpoint()
    kind, generation, values = decode(saved, source._identity, _expected_shapes(source))
    value = {"schema": SCHEMAS[kind], "model": source._identity,
             "generation": generation, **values,
             "accepted_factors": values["accepted_factors"].tolist()}
    owner = public.create_analysis(definition=source._definition)
    refused = None
    try:
        driver = FinitePointDriver(owner)
        restored = driver.restore(value)
        owner._finite_driver = driver
        assert driver._stable_s3 is True
        assert owner.checkpoint() == saved
        np.testing.assert_array_equal(restored.cell_resultants,
                                       source._finite_driver._accepted.cell_resultants)
        continued = owner.solve(ACTIONS, targets=(.5,))
        assert owner._finite_driver._stable_s3 is True
        assert continued.status == 'completed'
        forged = dict(value, schema=SCHEMAS['finite_native_material'])
        refused = public.create_analysis(definition=source._definition)
        with pytest.raises(ValueError, match='requires fibre model'):
            FinitePointDriver(refused).restore(forged)
        assert refused._generation == 0 and refused._finite_driver is None
    finally:
        if refused is not None:
            refused.close()
        owner.close()


def test_zero_load_legacy_checkpoint_continues_original_arithmetic_and_roundtrips(monkeypatch):
    """Legacy-history compatibility, exercised ONLY for explicit stress-free
    zero-load accepted predecessors/checkpoints and zero-load continuations:
    the same public constructor produces the old schema and arithmetic, a
    restored legacy False seed is never upgraded, and bytes roundtrip exactly.
    No nonzero legacy-history qualification is claimed: the nonzero 100z mild
    load cannot converge under the old arithmetic (kept as failed evidence
    at windows-policy-09 with the original controls; no threshold is loosened)."""
    definition = _s3_generalized_definition()
    monkeypatch.setattr(finite_module, 'supports_generalized_stable_s3',
                        lambda owner: False)
    owner = public.create_analysis(definition=definition)
    resumed = None
    try:
        result = owner.solve(ZERO_ACTIONS, targets=TARGETS,
                             max_iterations=24, max_backtracks=12)
        assert result.status == 'completed'
        assert owner._finite_driver._stable_s3 is False
        legacy = owner.checkpoint()
        assert json.loads(legacy)['schema'] == LEGACY_SCHEMA
        monkeypatch.undo()
        resumed = public.create_analysis(definition=definition)
        assert resumed._finite_driver is None
        restored = resumed.restore(legacy)
        assert resumed._finite_driver._stable_s3 is False
        assert resumed.checkpoint() == legacy
        np.testing.assert_array_equal(restored.cell_resultants, result.cell_resultants)
        continued = resumed.solve(ZERO_ACTIONS, targets=(.5,))
        assert resumed._finite_driver._stable_s3 is False
        oracle = owner.solve(ZERO_ACTIONS, targets=(.5,))
        np.testing.assert_array_equal(continued.cell_resultants, oracle.cell_resultants)
        assert continued.beam_station_accumulated == oracle.beam_station_accumulated
    finally:
        if resumed is not None:
            resumed.close()
        owner.close()


def test_malformed_and_wrong_schema_checkpoints_are_refused_atomically(accepted_owner):
    owner = accepted_owner
    saved = owner.checkpoint()
    kind, generation, values = decode(saved, owner._identity, _expected_shapes(owner))
    forged = encode('finite_native_material', owner._identity,
                     generation, tuple(values.items()))
    ineligible = public.create_analysis(definition=owner._definition)
    try:
        with pytest.raises(ValueError, match='elastic-shell fibre model'):
            ineligible.restore(forged)
        assert ineligible._generation == 0 and ineligible._finite_driver is None
    finally:
        ineligible.close()
    changed = {name: np.array(value, copy=True) for name, value in values.items()}
    changed['accepted_plastic'][0, 0, 0] += 1e-6
    corrupt = encode(kind, owner._identity, generation, tuple(changed.items()))
    untouched = public.create_analysis(definition=owner._definition)
    try:
        with pytest.raises(ValueError):
            untouched.restore(corrupt)
        assert untouched._generation == 0 and untouched._finite_driver is None
    finally:
        untouched.close()
    corrupted = bytearray(saved)
    corrupted[-3] = ord('9')
    fresh = public.create_analysis(definition=owner._definition)
    try:
        with pytest.raises(ValueError, match='digest mismatch'):
            fresh.restore(bytes(corrupted))
        assert fresh._generation == 0 and fresh._finite_driver is None
    finally:
        fresh.close()
    json_payload = json.dumps({'schema': SCHEMAS[kind], 'model': owner._identity,
        'generation': generation,
        **{name: value.tolist() for name, value in values.items()}}).encode()
    json_owner = public.create_analysis(definition=owner._definition)
    try:
        with pytest.raises(ValueError, match='requires binary framing'):
            json_owner.restore(json_payload)
        assert json_owner._generation == 0 and json_owner._finite_driver is None
    finally:
        json_owner.close()
    assert owner.checkpoint() == saved


def test_actual_derivative_maps_and_subspacing_response(accepted_owner):
    owner = accepted_owner
    _eid, element, _slots, origin = owner._shell_parts[0]
    reference = np.array([owner._shell_model.mesh.nodes[node].coords()
                          for node in element.node_ids])
    epsilon = 1e-14
    u = np.zeros((3, 6))
    u[:, :3] = epsilon*(reference-reference.mean(axis=0))
    local, _, _ = deformation(reference, u.ravel(), stable_translation=True)
    np.testing.assert_allclose(local, u.ravel(), rtol=1e-12, atol=epsilon*1e-14)
    force, tangent, state = response(owner._shell_model, element, u.ravel(), origin, 3,
                                     stable_translation=True)
    material = owner._shell_model.get_material('shell')
    expected = element.compute_stiffness_matrix(owner._shell_model.mesh, material)@u.ravel()
    assert np.linalg.norm(expected) > 0.
    assert np.linalg.norm(force-expected)/np.linalg.norm(expected) < 1e-12
    assert np.isfinite(tangent).all()
    np.testing.assert_array_equal(state['alpha'], np.zeros_like(state['alpha']))
    packet, guard = buckling.capture(owner, owner._generation, mixed=True)
    h, km, g = dense_mixed_oracle(owner, packet)
    for actual, expected_matrix in ((packet.hessian, h), (packet.material, km),
                                   (packet.geometric, g)):
        np.testing.assert_allclose(actual.toarray(), expected_matrix,
                                   rtol=1e-11, atol=2e-6)
    free = list(packet.free)
    np.testing.assert_allclose(
        (packet.material+packet.geometric)[free, :][:, free].toarray(),
        h[np.ix_(free, free)], rtol=1e-11, atol=2e-6)
    guard()
    _plateau_check(owner)


def test_buckling_consumes_stable_policy_and_preserves_checkpoint(accepted_owner):
    owner = accepted_owner
    before = owner.checkpoint()
    result = owner.buckling(public.GEMixedBucklingRequest(owner._generation, count=2))
    assert type(result) is public.GEMixedBucklingResult and result.status == 'completed', result.failure_reason
    assert max(result.equation_residuals) <= 1e-8 and max(result.constraint_residuals) <= 1e-8
    assert result.resultant_modes and result.multiplier_modes
    assert owner.checkpoint() == before
    packet, guard = buckling.capture(owner, owner._generation, mixed=True)
    driver = owner._finite_driver
    stable = driver._stable_s3
    driver._stable_s3 = not stable
    try:
        with pytest.raises(ValueError, match='accepted state changed'):
            guard()
    finally:
        driver._stable_s3 = stable
    guard()
    assert owner.checkpoint() == before
