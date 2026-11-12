"""Explicit finite-pad material admission and complete accepted-state ownership."""
from dataclasses import replace
import json
import numpy as np
import pytest

from anyfesolver import b3_ge, stiffened_shell as ss
from anymaterial import DNVC208MaterialCurve
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_checkpoint_binary import MAGIC, decode, encode
from test_stiffened_shell_ge_distributing_finite_public import definition as elastic_definition


def definition(coupled=True):
    base = elastic_definition()
    element, inertia = base.beams[0].instantiate()
    section = b3_ge.EllipsoidalGeneralizedSection(
        element.section.elastic, element.section.metric, 1e6, 1e11)
    coordinates = element.operator.reference.coordinates.copy()
    coordinates[:, 2] = 0.
    beam = b3_ge.define_beam("b3-ge", element.element_id, element.node_ids,
        coordinates, element.operator.reference.nodal_triads,
        section, inertia, quadrature=element.operator.order,
        arithmetic_policy=element.operator.arithmetic_policy)
    curve = DNVC208MaterialCurve(200e6, 220e6, 230e6, .005, .01, 1e9, .2) if coupled else None
    return replace(base, beams=(beam,), shell=replace(base.shell, j2_curve=curve),
        finite_distributing_material_policy=(ss.GE_FINITE_DISTRIBUTING_COUPLED_J2_MATERIAL_V1
            if coupled else ss.GE_FINITE_DISTRIBUTING_GENERALIZED_MATERIAL_V1))


def actions():
    # A centered axial history separates constitutive ownership from unresolved
    # eccentric bending capacity. The retained 400/500 N bending failures and
    # the 300 N unyielded-shell trial are not replaced or reclassified by this.
    return (ss.NodalAction("beam", 103, (16e6, 0., 0., 0., 0., 0.)),)


@pytest.fixture(scope="module", params=(False, True), ids=("generalized", "coupled"))
def completed(request):
    model = definition(request.param)
    owner = ss.create_analysis(definition=model)
    try:
        try:
            result = owner.solve(actions(), (.25, .5, .75, 1.), max_iterations=24, max_backtracks=16)
        except ss.FiniteIncrementFailure as error:
            pytest.fail(f"material fixture failed: {error.attempt}; accepted generation={owner._generation}")
        yield model, owner, result
    finally:
        owner.close()


def test_simultaneous_material_admission_and_yield(completed):
    model, owner, result = completed
    assert result.status == "completed"
    assert np.max(result.beam_station_accumulated) > 0.
    if model.shell.j2_curve is not None:
        assert np.max(result.shell_layer_alpha) > 0.
        assert result.shell_q4_material
    assert result.beam_generalized_material
    assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
    assert owner.capabilities["status"] == "EXPERIMENTAL_NOT_QUALIFIED"
    assert owner.capabilities["material_policy"] == model.finite_distributing_material_policy


def test_owned_predecessor_restart(completed):
    model, owner, result = completed
    checkpoint = owner.checkpoint()
    restored = ss.create_analysis(definition=model)
    try:
        other = restored.restore(checkpoint)
        assert restored.checkpoint() == checkpoint
        for name in ("shell_displacement", "cell_resultants", "support_reactions",
                     "beam_station_accumulated", "shell_layer_alpha"):
            np.testing.assert_allclose(getattr(other, name), getattr(result, name), rtol=1e-10, atol=1e-12)
        assert other.beam_generalized_material
    finally:
        restored.close()


def test_policy_does_not_reclassify_elastic_or_reference_records():
    base = elastic_definition()
    with pytest.raises(ValueError, match="finite attachment"):
        ss.create_analysis(definition=replace(base, finite_distributing_footprints=(),
            finite_distributing_material_policy=ss.GE_FINITE_DISTRIBUTING_GENERALIZED_MATERIAL_V1))
    with pytest.raises(ValueError, match="matching shell J2"):
        ss.create_analysis(definition=replace(base,
            finite_distributing_material_policy=ss.GE_FINITE_DISTRIBUTING_COUPLED_J2_MATERIAL_V1))
    with pytest.raises(ValueError, match="distributing reference"):
        ss.create_analysis(definition=replace(definition(), finite_distributing_material_policy="elastic"))
    for model, budget in ((definition(), 17), (base, 13)):
        owner = ss.create_analysis(definition=model)
        try:
            with pytest.raises(ValueError, match="bounded finite Newton"):
                owner.solve(actions(), max_backtracks=budget)
            assert owner._generation == 0 and owner._finite_driver is None
        finally:
            owner.close()


def decoded(raw, model):
    # Trusted fixture mutation oracle only; production derives shapes from the
    # compiled owner and never lets these header dimensions allocate its state.
    n = int.from_bytes(raw[len(MAGIC):len(MAGIC)+4], "little")
    header = json.loads(raw[len(MAGIC)+4:len(MAGIC)+4+n])
    return decode(raw, model, lambda kind, generation:
                  tuple((name, tuple(shape)) for name, shape in header["arrays"]))


def test_fixed_predecessor_actual_update_tangent_and_cached_recovery(completed, monkeypatch):
    _, owner, result = completed
    driver = owner._finite_driver
    before = owner.checkpoint()
    def evaluate(state):
        return driver._assemble(state, 1., driver._shell_predecessors,
                                driver._beam_predecessors, pattern=driver._pattern)
    residual, tangent, *rest = evaluate(driver._state)
    repeated = evaluate(driver._state)
    np.testing.assert_array_equal(residual, repeated[0])
    np.testing.assert_array_equal(tangent.toarray(), repeated[1].toarray())
    restored = ss.create_analysis(definition=owner._definition)
    try:
        restored.restore(before)
        other = restored._finite_driver
        recovered = other._assemble(other._state, 1., other._shell_predecessors,
                                    other._beam_predecessors, pattern=other._pattern)
        np.testing.assert_array_equal(residual, recovered[0])
        np.testing.assert_array_equal(tangent.toarray(), recovered[1].toarray())
    finally:
        restored.close()
    direction = np.zeros(len(residual))
    direction[:owner._shell_count:6] = np.linspace(0., 1e-4, len(owner._shell_ids))
    for i in range(len(owner._beam_ids)):
        direction[owner._shell_count+6*i+2] = .001*(i+1)
        direction[owner._shell_count+6*i+4] = .0001*(i+1)
    direction[owner._fixed] = 0.
    weights = np.r_[owner._coordinate_scale, np.ones(owner._C.shape[0])]
    predicted = (tangent @ direction)*weights
    errors = []
    for h in (1e-4, 1e-5, 1e-6):
        plus = evaluate(driver._advance(driver._state, direction, h))[0]
        minus = evaluate(driver._advance(driver._state, direction, -h))[0]
        errors.append(np.linalg.norm((plus-minus)/(2*h)*weights-predicted)/
                      max(np.linalg.norm(predicted), 1.))
    assert min(errors) <= 1e-5, errors
    assert owner.checkpoint() == before
    def forbidden(*args, **kwargs):
        raise AssertionError("accepted recovery repeated constitutive evaluation")
    with monkeypatch.context() as patch:
        patch.setattr(driver, "_assemble", forbidden)
        for element, _, _ in owner._beam_parts:
            patch.setattr(type(element.operator), "evaluate", forbidden)
        recovered = owner.recover()
        np.testing.assert_array_equal(recovered.beam_station_accumulated, result.beam_station_accumulated)
        np.testing.assert_array_equal(recovered.shell_layer_alpha, result.shell_layer_alpha)


def test_predecessor_corruption_and_wrong_kind_reject_before_publication(completed):
    model, owner, result = completed
    kind, generation, arrays = decoded(owner.checkpoint(), owner._identity)
    assert kind == owner.capabilities["checkpoints"][0]
    for field in ("beam_origin_plastic", "beam_origin_accumulated", "shell_origin_plastic", "shell_origin_alpha"):
        if field not in arrays:
            assert model.shell.j2_curve is None and field.startswith("shell_")
            continue
        changed = {name: value.copy() for name, value in arrays.items()}
        changed[field].flat[0] += .01
        raw = encode(kind, owner._identity, generation, tuple(changed.items()))
        fresh = ss.create_analysis(definition=model)
        try:
            with pytest.raises(ValueError):
                fresh.restore(raw)
            assert fresh._generation == 0 and fresh._finite_driver is None
        finally:
            fresh.close()
    fresh = ss.create_analysis(definition=model)
    try:
        raw = encode("finite_distributing", fresh._identity, generation,
            tuple((name, value) for name, value in arrays.items() if not name.startswith(("beam_", "shell_"))))
        with pytest.raises(ValueError, match="kind"):
            fresh.restore(raw)
        assert fresh._generation == 0 and fresh._finite_driver is None
    finally:
        fresh.close()
    for changed in (dict(arrays, accepted_factors=np.array((.5, .25, .75, 1.))),
                    {name: value for name, value in arrays.items() if name != "beam_origin_plastic"}):
        fresh = ss.create_analysis(definition=model)
        try:
            raw = encode(kind, fresh._identity, generation, tuple(changed.items()))
            with pytest.raises(ValueError):
                fresh.restore(raw)
            assert fresh._generation == 0 and fresh._finite_driver is None
        finally:
            fresh.close()


def test_continuation_and_complete_case_cancellation(completed):
    # The 1.01 continuation failed and remains in raw v7/v9 evidence. This
    # smaller prescribed increment tests ownership, not large-step capacity.
    model, accepted, _ = completed
    prefix = accepted.checkpoint()
    resumed = ss.create_analysis(definition=model)
    continuous = ss.create_analysis(definition=model)
    token = CancellationToken()
    try:
        resumed.restore(prefix)
        continuous.solve(actions(), (.25, .5, .75, 1.), max_iterations=24, max_backtracks=16)
        assert continuous.checkpoint() == prefix
        def on_accepted(result):
            assert continuous.recover().generation == result.generation
            token.cancel("complete yielded finite pad prefix")
        with pytest.raises(SolveCancelled):
            continuous.solve(actions(), (1.0001, 1.0002), on_accepted=on_accepted,
                cancellation_token=token, max_iterations=24, max_backtracks=16)
        a = continuous.recover()
        b = resumed.solve(actions(), (1.0001,), max_iterations=24, max_backtracks=16)
        assert a.accepted_factors == b.accepted_factors == (.25, .5, .75, 1., 1.0001)
        assert resumed.checkpoint() == continuous.checkpoint()
        for name in ("shell_displacement", "cell_resultants", "support_reactions",
                     "beam_station_accumulated", "shell_layer_alpha"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    finally:
        resumed.close(); continuous.close()


@pytest.mark.parametrize("failure", (RuntimeError, MemoryError))
def test_candidate_and_checkpoint_failure_preserve_all_domains(completed, monkeypatch, failure):
    from anyfesolver._stiffened_shell_ge_distributing_finite import PreparedFiniteDistributingAttachment
    import anyfesolver._stiffened_shell_ge_finite_experimental as finite
    model, accepted, _ = completed
    owner = ss.create_analysis(definition=model)
    try:
        before = accepted.checkpoint()
        owner.restore(before)
        generation = owner.recover().generation
        q = owner._finite_driver._state.q.copy()
        original = PreparedFiniteDistributingAttachment.evaluate
        def broken(self, actual, state, multipliers, **kwargs):
            if not np.array_equal(state.q, q):
                raise failure("trial after shell and beam candidates")
            return original(self, actual, state, multipliers, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(PreparedFiniteDistributingAttachment, "evaluate", broken)
            with pytest.raises(failure, match="trial after shell and beam candidates"):
                owner.solve(actions(), (1.0001,))
        assert owner.recover().generation == generation and owner.checkpoint() == before
        def failed_encoding(*args, **kwargs):
            raise failure("snapshot before atomic publication")
        with monkeypatch.context() as patch:
            patch.setattr(finite, "_encode_binary_checkpoint", failed_encoding)
            with pytest.raises(failure, match="snapshot before atomic publication"):
                owner.solve(actions(), (1.0001,), max_iterations=24, max_backtracks=16)
        assert owner.recover().generation == generation and owner.checkpoint() == before
    finally:
        owner.close()


def test_cancel_during_prepared_snapshot_preserves_yielded_prefix(completed, monkeypatch):
    import anyfesolver._stiffened_shell_ge_finite_experimental as finite
    model, accepted, _ = completed
    owner = ss.create_analysis(definition=model)
    token = CancellationToken()
    try:
        before = accepted.checkpoint(); owner.restore(before)
        generation = owner._generation
        original = finite._encode_binary_checkpoint
        def cancelled(*args, **kwargs):
            result = original(*args, **kwargs)
            token.cancel("candidate histories prepared, no commit")
            return result
        with monkeypatch.context() as patch:
            patch.setattr(finite, "_encode_binary_checkpoint", cancelled)
            with pytest.raises(SolveCancelled):
                owner.solve(actions(), (1.0001,), cancellation_token=token,
                    max_iterations=24, max_backtracks=16)
        assert owner._generation == generation and owner.checkpoint() == before
    finally:
        owner.close()
