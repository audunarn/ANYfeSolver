"""Descriptor/compiler checks without a mechanical solve or state mutation."""
from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver._stiffened_shell_ge_experimental import NodalAction
from anyfesolver import _stiffened_shell_staged_load_experimental as staged


MODEL = "ab" * 32


def _action(force, domain="shell", node=1):
    return NodalAction(domain, node, tuple(force)+(0., 0., 0.))


def _programme():
    # Absolute endpoints: the second removes shell Fx and introduces beam Fy.
    return staged.StagedDeadLoadProgramme(MODEL, (
        (_action((10., 0., 0.)),),
        (_action((0., 20., 0.), "beam", 2),),
    ))


def _owner():
    return SimpleNamespace(_identity=MODEL, _K=SimpleNamespace(shape=(12, 12)))


def _load(actions):
    vector = np.zeros(12)
    for action in actions:
        key = (action.domain, action.node)
        if key not in (("shell", 1), ("beam", 2)):
            raise ValueError("unknown fixture node")
        start = 0 if action.domain == "shell" else 6
        vector[start:start+6] = action.wrench
    return vector


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _unframe(raw):
    start = len(staged.MAGIC)+4
    size = int.from_bytes(raw[len(staged.MAGIC):start], "little")
    return json.loads(raw[start:start+size]), raw[start+size:-32]


def _frame(header, inner, header_bytes=None):
    metadata = _canonical(header) if header_bytes is None else header_bytes
    content = staged.MAGIC+len(metadata).to_bytes(4, "little")+metadata+inner
    return content+sha256(content).digest()


def test_descriptor_owns_normalized_actions_and_complete_future_identity():
    wrench = np.array([10., 0., 0., 0., 0., 0.])
    programme = staged.StagedDeadLoadProgramme(MODEL.upper(), ((NodalAction("shell", 1, wrench),),))
    original = programme.serialize()
    wrench[0] = 999.
    payload = programme.to_payload()
    payload["endpoints"][0][0]["wrench"][0] = 888.
    assert programme.model_identity == MODEL
    assert programme.endpoints[0][0].wrench == (10., 0., 0., 0., 0., 0.)
    assert programme.serialize() == original
    with pytest.raises(FrozenInstanceError):
        programme.endpoints = ()
    changed_future = staged.StagedDeadLoadProgramme(MODEL, _programme().endpoints+((),))
    assert changed_future.identity != _programme().identity
    assert changed_future.augmented_model_identity != _programme().augmented_model_identity
    assert programme.augmented_model_identity != MODEL


def test_compilation_endpoint_exact_convex_forces_and_owned_write_protection():
    buffer = np.zeros(12)
    def reused_load(actions):
        buffer[:] = _load(actions)
        return buffer
    compiled = _programme().compile(_owner(), reused_load)
    buffer[:] = 999.
    np.testing.assert_array_equal(compiled.load_at(0.), np.zeros(12))
    np.testing.assert_array_equal(compiled.load_at(1.), _load(_programme().endpoints[0]))
    np.testing.assert_array_equal(compiled.load_at(2.), _load(_programme().endpoints[1]))
    expected = np.zeros(12)
    expected[0], expected[7] = 5., 10.
    np.testing.assert_array_equal(compiled.load_at(1.5), expected)
    assert compiled.load_at(1.) is compiled.vectors[1]
    assert compiled.model_identity == compiled.augmented_model_identity
    assert compiled.physical_model_identity == MODEL
    assert compiled.identity == compiled.programme.identity
    for vector in (*compiled.vectors, compiled.load_at(.5), compiled.load_at(1.5)):
        assert not vector.flags.writeable
        with pytest.raises(ValueError):
            vector.setflags(write=True)
        with pytest.raises(ValueError):
            vector[0] = 1.
    assert compiled.progress_at(0.) == (0, 0.)
    assert compiled.progress_at(.5) == (0, .5)
    assert compiled.progress_at(1.) == (1, 0.)
    assert compiled.progress_at(2.) == (2, 0.)


@pytest.mark.parametrize("identity", [None, "", "g"*64, "a"*63, "a"*65])
def test_model_identity_format_rejected(identity):
    with pytest.raises(ValueError):
        staged.StagedDeadLoadProgramme(identity, ((),))


@pytest.mark.parametrize("action", [
    object(), NodalAction("other", 1, (0.,)*6),
    NodalAction("shell", True, (0.,)*6), NodalAction("shell", 0, (0.,)*6),
    NodalAction("shell", 1, (0.,)*5),
    NodalAction("shell", 1, (float("nan"), 0., 0., 0., 0., 0.)),
    NodalAction("shell", 1, (float("inf"), 0., 0., 0., 0., 0.)),
    NodalAction("shell", 1, (True, 0., 0., 0., 0., 0.)),
    NodalAction("shell", 1, (0., 0., 0., 1., 0., 0.)),
])
def test_invalid_actions_and_nontranslation_wrenches_rejected(action):
    with pytest.raises(ValueError):
        staged.StagedDeadLoadProgramme(MODEL, ((action,),))


def test_duplicate_same_node_rejected_before_any_aggregation():
    with pytest.raises(ValueError, match="duplicate"):
        staged.StagedDeadLoadProgramme(MODEL, ((
            _action((1e16, 0., 0.)), _action((-1e16, 0., 0.)),
        ),))
    # Domain identity is part of node identity.
    programme = staged.StagedDeadLoadProgramme(MODEL, ((
        _action((1., 0., 0.)), _action((2., 0., 0.), "beam", 1),
    ),))
    assert len(programme.endpoints[0]) == 2


@pytest.mark.parametrize("endpoints", [(), [], [()], ([],)])
def test_endpoint_container_contract(endpoints):
    with pytest.raises(ValueError):
        staged.StagedDeadLoadProgramme(MODEL, endpoints)


def test_descriptor_and_compiled_resource_caps_before_loader(monkeypatch):
    monkeypatch.setattr(staged, "MAX_STAGES", 1)
    with pytest.raises(ValueError):
        _programme()
    monkeypatch.setattr(staged, "MAX_STAGES", 64)
    monkeypatch.setattr(staged, "MAX_ACTIONS_PER_STAGE", 1)
    with pytest.raises(ValueError):
        staged.StagedDeadLoadProgramme(MODEL, ((
            _action((1., 0., 0.)), _action((2., 0., 0.), "beam", 2),
        ),))
    monkeypatch.setattr(staged, "MAX_ACTIONS", 1)
    with pytest.raises(ValueError):
        _programme()
    monkeypatch.setattr(staged, "MAX_ACTIONS", 16384)
    programme = _programme()
    monkeypatch.setattr(staged, "MAX_COMPILED_VALUES", 35)
    calls = []
    with pytest.raises(ValueError, match="storage"):
        programme.compile(_owner(), lambda a: calls.append(a))
    assert calls == []


def test_compile_binding_and_original_loader_errors():
    programme = _programme()
    owner = _owner()
    owner._identity = "cd"*32
    calls = []
    with pytest.raises(ValueError, match="different model"):
        programme.compile(owner, lambda a: calls.append(a))
    assert calls == []
    with pytest.raises(TypeError):
        programme.compile(_owner(), None)
    for failure in (MemoryError("allocation"), RuntimeError("programming")):
        def broken(actions):
            raise failure
        with pytest.raises(type(failure)) as caught:
            programme.compile(_owner(), broken)
        assert caught.value is failure
    for value in (np.zeros(11), np.full(12, np.nan), np.ones(12)):
        with pytest.raises(ValueError):
            programme.compile(_owner(), lambda a: value)
    foreign_node = staged.StagedDeadLoadProgramme(MODEL, ((_action((1., 0., 0.), node=99),),))
    with pytest.raises(ValueError, match="unknown fixture node"):
        foreign_node.compile(_owner(), _load)


def test_strict_descriptor_roundtrip_and_integrity():
    programme = _programme()
    assert staged.StagedDeadLoadProgramme.deserialize(programme.serialize()) == programme
    assert staged.StagedDeadLoadProgramme.from_payload(programme.to_payload()) == programme
    for field, value in (("policy", "foreign"), ("identity", "00"*32), ("extra", 1)):
        payload = programme.to_payload()
        payload[field] = value
        with pytest.raises(ValueError):
            staged.StagedDeadLoadProgramme.from_payload(payload)
    payload = programme.to_payload()
    payload["endpoints"][1][0]["wrench"][1] = 21.
    with pytest.raises(ValueError, match="identity"):
        staged.StagedDeadLoadProgramme.from_payload(payload)
    for raw in (b" "+programme.serialize(), programme.serialize()[:-1],
                b'{"policy":"x","policy":"y"}', b'{"x":NaN}'):
        with pytest.raises(ValueError):
            staged.StagedDeadLoadProgramme.deserialize(raw)


@pytest.mark.parametrize("accepted,targets", [
    (0., (.25, .5, 1., 1.5, 2.)), (.5, (1., 2.)), (1., (1.5, 2.)),
])
def test_targets_require_strict_progress_and_every_crossed_boundary(accepted, targets):
    compiled = _programme().compile(_owner(), _load)
    assert compiled.validate_targets(accepted, targets) == targets


@pytest.mark.parametrize("accepted,targets", [
    (0., (1.5,)), (0., (2.,)), (.5, (1.5, 2.)), (1., (1.,)),
    (1., (.5,)), (0., (1., 1.)), (0., (3.,)), (0., (float("nan"),)),
    (0., (True,)), (0., [1.]), (0., ()), (2., (2.,)),
])
def test_invalid_target_sequences_rejected(accepted, targets):
    with pytest.raises(ValueError):
        _programme().compile(_owner(), _load).validate_targets(accepted, targets)


def test_interpolation_maximum_finite_forces_exact_cancellation_and_zero_endpoint():
    large = np.finfo(float).max
    programme = staged.StagedDeadLoadProgramme(MODEL, (
        (_action((large, 0., 0.)),), (_action((-large, 0., 0.)),), (),
    ))
    compiled = programme.compile(_owner(), _load)
    assert compiled.load_at(1.)[0] == large
    assert compiled.load_at(2.)[0] == -large
    assert compiled.load_at(1.5)[0] == 0.
    assert compiled.load_at(2.5)[0] == -large/2
    np.testing.assert_array_equal(compiled.load_at(3.), np.zeros(12))
    constant = staged.StagedDeadLoadProgramme(MODEL, (
        (_action((large, 0., 0.)),), (_action((large, 0., 0.)),),
    )).compile(_owner(), _load)
    assert constant.load_at(1.5)[0] == large


def test_interpolation_rejects_underresolved_subnormal_but_keeps_exact_stored_values():
    tiny = np.nextafter(0., 1.)
    for force in (tiny, 3*tiny):
        compiled = staged.StagedDeadLoadProgramme(MODEL, ((_action((force, 0., 0.)),),)).compile(_owner(), _load)
        assert compiled.load_at(1.)[0] == force
        with pytest.raises(staged.StagedLoadResolutionError):
            compiled.load_at(.5)
    compiled = staged.StagedDeadLoadProgramme(MODEL, ((_action((2*tiny, 0., 0.)),),)).compile(_owner(), _load)
    assert compiled.load_at(.5)[0] == tiny


@pytest.mark.parametrize("position", [0., .25, 1., 1.5, 2.])
def test_checkpoint_roundtrip_binds_descriptor_position_and_inner(position):
    compiled = _programme().compile(_owner(), _load)
    inner = b"independent inner checkpoint fixture"
    raw = staged.encode_checkpoint(compiled, position, inner)
    programme, restored_position, restored_inner = staged.decode_checkpoint(raw)
    assert programme == compiled.programme
    assert restored_position == position
    assert restored_inner == inner
    header, _ = _unframe(raw)
    assert (header["cursor"], header["fraction"]) == compiled.progress_at(position)


@pytest.mark.parametrize("key,value", [
    ("policy", "foreign"), ("position", 1), ("cursor", True),
    ("cursor", 0), ("fraction", .25), ("fraction", 0),
    ("inner_sha256", "00"*32), ("extra", 1),
])
def test_checkpoint_strict_metadata_rejects_even_rehashed_tampering(key, value):
    raw = staged.encode_checkpoint(_programme().compile(_owner(), _load), 1.5, b"inner")
    header, inner = _unframe(raw)
    header[key] = value
    with pytest.raises(ValueError):
        staged.decode_checkpoint(_frame(header, inner))


def test_checkpoint_framing_descriptor_and_inner_digests_are_independent_guards(monkeypatch):
    compiled = _programme().compile(_owner(), _load)
    raw = staged.encode_checkpoint(compiled, .5, b"inner")
    for broken in (b"x"+raw[1:], raw[:-1], raw[:-32]+b"0"*32,
                   staged.MAGIC+(2**32-1).to_bytes(4, "little")+raw[len(staged.MAGIC)+4:]):
        with pytest.raises(ValueError):
            staged.decode_checkpoint(broken)
    header, inner = _unframe(raw)
    with pytest.raises(ValueError, match="inner digest"):
        staged.decode_checkpoint(_frame(header, b"changed inner"))
    header["programme"]["endpoints"][1][0]["wrench"][1] = 21.
    with pytest.raises(ValueError, match="identity"):
        staged.decode_checkpoint(_frame(header, inner))
    header, inner = _unframe(raw)
    with pytest.raises(ValueError):
        staged.decode_checkpoint(_frame(header, inner, b" "+_canonical(header)))
    monkeypatch.setattr(staged, "MAX_CHECKPOINT_BYTES", len(raw)-1)
    with pytest.raises(ValueError, match="bound"):
        staged.encode_checkpoint(compiled, .5, b"inner")
    with pytest.raises(ValueError):
        staged.decode_checkpoint(raw)


def test_descriptor_metadata_bound_and_checkpoint_metadata_bound(monkeypatch):
    compiled = _programme().compile(_owner(), _load)
    monkeypatch.setattr(staged, "MAX_METADATA_BYTES", 8)
    with pytest.raises(ValueError, match="metadata"):
        _programme()
    with pytest.raises(ValueError):
        staged.StagedDeadLoadProgramme.deserialize(compiled.programme.serialize())
    with pytest.raises(ValueError, match="metadata"):
        staged.encode_checkpoint(compiled, .5, b"inner")
