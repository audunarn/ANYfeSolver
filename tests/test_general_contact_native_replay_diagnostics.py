"""Focused controls for the bounded native replay first-difference walker.

The replay and binding gates stay exact and unchanged; these tests pin the
diagnostic contract only: canonical-exact leaf comparison (including
negative zero, without coercing floats in product state), refusal of
nonfinite values, bounded tokens/paths/leaf counts, and component digests
whose input bytes remain the exact canonical bytes while the plain display
form drops only the terminal newline.
"""
from hashlib import sha256

import pytest

from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.general_contact.native_replay_diagnostics import (
    LEAF_COUNT_LIMIT,
    PATH_LIMIT,
    PLAIN_COMPONENT_LIMIT,
    TOKEN_LIMIT,
    component_digests,
    first_difference,
    replay_difference_message,
)


def test_negative_zero_is_reported_canonically_exact():
    stored = {'contact': {'beam_a_force_n': [-0.0]}}
    replayed = {'contact': {'beam_a_force_n': [0.0]}}
    # The exact gate fails on the canonical byte difference; the walker must
    # explain it instead of silently reporting no difference.
    assert canonical(stored) != canonical(replayed)
    assert first_difference(stored, replayed) == (
        'contact.beam_a_force_n[0]', '-0.0', '0.0', '1')
    message = replay_difference_message(stored, replayed, 'record 1')
    assert 'path=contact.beam_a_force_n[0] stored=-0.0 replayed=0.0' in message
    assert 'differing leaves=1' in message
    assert first_difference(replayed, replayed) is None
    assert replay_difference_message(replayed, replayed, 'record 1') == ''


def test_nonfinite_product_state_is_refused_without_coercion():
    for value in (float('inf'), -float('inf'), float('nan')):
        with pytest.raises(ValueError):
            first_difference({'gap': value}, {'gap': 0.})
        with pytest.raises(ValueError):
            component_digests({'gap': value})


def test_shape_and_key_differences_are_reported_first():
    keys = first_difference({'a': 1., 'stored_only': 2.},
                            {'a': 1., 'replayed_only': 3.})
    assert keys == ('<root>', "keys=['stored_only']", "keys=['replayed_only']", '1')
    length = first_difference({'records': [1., 2.]}, {'records': [1.]})
    assert length == ('records', 'length=2', 'length=1', '1')


def test_tokens_paths_and_leaf_counts_are_bounded():
    long_stored = 'x' * 200
    long_replayed = 'y' * 200
    token = first_difference({'payload': long_stored}, {'payload': long_replayed})
    # The token is the canonical display text, so a string leaf keeps its
    # JSON quotes and the truncation lands inside them.
    assert token[1] == chr(34) + long_stored[:TOKEN_LIMIT - 4] + '...'
    assert token[2] == chr(34) + long_replayed[:TOKEN_LIMIT - 4] + '...'
    deep_stored = {'leaf': 1.}
    deep_replayed = {'leaf': 2.}
    for _ in range(100):
        deep_stored = {'k': deep_stored}
        deep_replayed = {'k': deep_replayed}
    deep = first_difference(deep_stored, deep_replayed)
    assert len(deep[0]) == PATH_LIMIT and deep[0].endswith('...')
    many_stored = {'i%03d' % i: float(i) for i in range(80)}
    many_replayed = {'i%03d' % i: float(i) + 1. for i in range(80)}
    many = first_difference(many_stored, many_replayed)
    assert many[0] == 'i000'
    assert many[3] == '%d+' % LEAF_COUNT_LIMIT


def test_component_digests_hash_exact_canonical_bytes_and_stay_single_line():
    long_value = 'y' * 50
    assert component_digests({'gap': long_value}) == ('gap=sha256:'
        + sha256(canonical(long_value)).hexdigest())
    over_limit = 'z' * (PLAIN_COMPONENT_LIMIT - 2)
    assert len(canonical(over_limit)) == PLAIN_COMPONENT_LIMIT + 1
    assert component_digests({'gap': over_limit}) == ('gap=sha256:'
        + sha256(canonical(over_limit)).hexdigest())
    short_limit = 'z' * (PLAIN_COMPONENT_LIMIT - 3)
    assert len(canonical(short_limit)) == PLAIN_COMPONENT_LIMIT
    digests = component_digests({'gap': 0.25, 'mode': short_limit})
    assert digests == 'gap=0.25 mode=' + chr(34) + short_limit + chr(34)
    assert chr(10) not in digests
