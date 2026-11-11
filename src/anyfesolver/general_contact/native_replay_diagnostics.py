"""Bounded first-difference context for strict native checkpoint replay gates.

The replay and binding gates compare exact canonical bytes and stay unchanged.
These helpers only append bounded, informative context to an existing failure
message so a remote runner log can identify the first differing recorded
quantity or the divergent settings component. Nothing here alters a gate,
tolerance, identity or recorded byte sequence; the stored checkpoint keeps
only the aggregate identity by schema contract.
"""
from hashlib import sha256
from json import loads

from .._ge_beam3_p5_seeded.core import canonical

TOKEN_LIMIT = 96
PATH_LIMIT = 160
LEAF_COUNT_LIMIT = 64
PLAIN_COMPONENT_LIMIT = 40


def _text(value):
    return canonical(value).decode('ascii')


def _plain(text):
    """Display form of a canonical text: only its terminal newline is stripped."""
    return text[:-1] if text.endswith('\n') else text


def _token(value):
    text = _plain(_text(value))
    return text if len(text) <= TOKEN_LIMIT else text[:TOKEN_LIMIT - 3] + '...'


def _component_digest(name, value):
    text = _text(value)
    if len(text) <= PLAIN_COMPONENT_LIMIT:
        return name + '=' + _plain(text)
    return name + '=sha256:' + sha256(text.encode('ascii')).hexdigest()


def component_digests(components):
    """Bounded per-component digests for settings/definition binding context."""
    return ' '.join(_component_digest(name, components[name]) for name in sorted(components))


def _walk(stored, replayed, path, differences):
    if len(differences) > LEAF_COUNT_LIMIT:
        return
    if type(stored) is dict and type(replayed) is dict:
        only_stored = sorted(set(stored) - set(replayed))
        only_replayed = sorted(set(replayed) - set(stored))
        if only_stored or only_replayed:
            differences.append((path, 'keys=' + repr(only_stored[:8]), 'keys=' + repr(only_replayed[:8])))
        for key in sorted(set(stored) & set(replayed)):
            child = (path + '.' + str(key)) if path else str(key)
            _walk(stored[key], replayed[key], child, differences)
        return
    if type(stored) is list and type(replayed) is list:
        if len(stored) != len(replayed):
            differences.append((path, 'length=' + str(len(stored)), 'length=' + str(len(replayed))))
        for index in range(min(len(stored), len(replayed))):
            _walk(stored[index], replayed[index], path + '[' + str(index) + ']', differences)
        return
    if type(stored) is not type(replayed) or _text(stored) != _text(replayed):
        differences.append((path, _token(stored), _token(replayed)))


def first_difference(stored, replayed):
    """Return the first differing canonical record path and bounded tokens."""
    differences = []
    _walk(loads(_text(stored)), loads(_text(replayed)), '', differences)
    if not differences:
        return None
    path, stored_token, replayed_token = differences[0]
    count = len(differences)
    total = str(count) if count <= LEAF_COUNT_LIMIT else str(LEAF_COUNT_LIMIT) + '+'
    if len(path) > PATH_LIMIT:
        path = path[:PATH_LIMIT - 3] + '...'
    return path or '<root>', stored_token, replayed_token, total


def replay_difference_message(stored, replayed, label):
    """Bounded first-difference context appended to a failed replay gate."""
    difference = first_difference(stored, replayed)
    if difference is None:
        return ''
    path, stored_token, replayed_token, total = difference
    return (' first difference ' + label + ' path=' + path + ' stored=' + stored_token
        + ' replayed=' + replayed_token + ' differing leaves=' + total)
