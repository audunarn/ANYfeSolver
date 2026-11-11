"""Bounded V5 S3 checkpoint framing and legacy-format separation."""

from hashlib import sha256

import numpy as np
import pytest

from anyfesolver._stiffened_shell_ge_checkpoint_binary import (
    MAGIC, S3_MAGIC, decode_s3, encode_s3,
)


def _shapes(_generation):
    return (("q", (3,)), ("accepted_factors", (2,)))


def _sample():
    arrays = (("q", np.array((1., -2., 3.))),
              ("accepted_factors", np.array((.5, 1.))))
    return encode_s3("model-a", 2, arrays, (b"native-state-a", b"native-state-b"))


def test_s3_checkpoint_round_trip_has_distinct_legacy_identity():
    raw = _sample()
    assert raw.startswith(S3_MAGIC)
    assert not raw.startswith(MAGIC)
    generation, arrays, states = decode_s3(raw, "model-a", _shapes, 2)
    assert generation == 2
    np.testing.assert_array_equal(arrays["q"], (1., -2., 3.))
    np.testing.assert_array_equal(arrays["accepted_factors"], (.5, 1.))
    assert states == (b"native-state-a", b"native-state-b")
    assert not arrays["q"].flags.writeable


def test_s3_checkpoint_rejects_corruption_identity_and_compiled_layout():
    raw = _sample()
    corrupted = bytearray(raw)
    corrupted[-33] ^= 1
    with pytest.raises(ValueError, match="digest mismatch"):
        decode_s3(bytes(corrupted), "model-a", _shapes, 2)
    with pytest.raises(ValueError, match="incompatible"):
        decode_s3(raw, "other-model", _shapes, 2)
    with pytest.raises(ValueError, match="incompatible"):
        decode_s3(raw, "model-a", _shapes, 1)
    with pytest.raises(ValueError, match="layout"):
        decode_s3(raw, "model-a", lambda _: (("q", (4,)),), 2)


def test_s3_checkpoint_rejects_truncated_native_state_with_valid_digest():
    raw = _sample()
    content = raw[:-sha256().digest_size]
    broken = content[:-1]
    broken += sha256(broken).digest()
    with pytest.raises(ValueError, match="state length"):
        decode_s3(broken, "model-a", _shapes, 2)
