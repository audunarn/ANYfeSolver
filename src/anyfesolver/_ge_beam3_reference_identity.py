"""Private exact live-input guard for an already captured canonical reference.

Reuse the original SHA only when every canonical input still has the same
binary representation. This is not an object-only cache or a sampled guard.
No reference equations, canonical encoding or fingerprint algorithm changes.
"""
from struct import pack
import numpy as np
from . import ge_beam3_curved_reference as source
from . import _ge_beam3_centered_reference as centered


def _array(value):
    if type(value) is not np.ndarray or value.dtype != np.dtype('float64'):
        raise ValueError('snapshot requires an ordinary binary64 array')
    return value.shape, value.dtype.str, value.tobytes(order='C')


def _number(value):
    # The canonical source casts scalars to float. Binary storage includes
    # signed zero; a changed representation takes the canonical fallback.
    return pack('!d', float(value))


def _methods():
    return (centered.CenteredCurvedBeam3ReferenceGeometry.canonical_data,
        centered.CenteredCurvedBeam3ReferenceGeometry.canonical_bytes,
        centered.CenteredCurvedBeam3ReferenceGeometry.fingerprint,
        source.CurvedBeam3ReferenceGeometry.canonical_data,
        source.CurvedBeam3ReferenceGeometry.canonical_bytes,
        source.CurvedBeam3ReferenceGeometry.fingerprint,
        source.CurvedBeam3RegularityCertificate.canonical_data,
        source._binary64, source._binary64_array, source.canonical_json_bytes)


def _snapshot(reference):
    if type(reference) is not centered.CenteredCurvedBeam3ReferenceGeometry:
        raise ValueError('exact centered reference required')
    regularity = reference._regularity
    if type(regularity) is not source.CurvedBeam3RegularityCertificate:
        raise ValueError('exact canonical regularity certificate required')
    constants = (source.GE_BEAM3_CURVED_REFERENCE_INTERPOLATION_ID,
        source.GE_BEAM3_CURVED_REFERENCE_REVERSAL_ID, source.GE_BEAM3_CURVED_REFERENCE_SCHEMA_ID,
        source.GE_BEAM3_CURVED_REFERENCE_REGULARITY_ID, centered.SCHEMA, centered.EVALUATION_ID)
    if any(type(value) is not str for value in constants):
        raise ValueError('exact canonical reference identifiers required')
    return (constants, getattr(regularity.canonical_data, '__func__', regularity.canonical_data),
        _array(source._NODE_PARAMETERS), _array(reference._coordinates),
        _array(reference._nodal_triads), _array(reference._coefficient_high), _array(reference._coefficient_low),
        tuple(tuple(_number(v) for v in row) for row in reference._half_frame_data),
        tuple(_number(v) for v in regularity.affine_constant),
        tuple(_number(v) for v in regularity.affine_linear),
        _number(regularity.minimizer_xi), _number(regularity.minimum_jacobian_squared),
        _number(regularity.characteristic_length), _number(regularity.minimum_admissible_jacobian),
        _number(reference._frame_tolerance), _number(reference._regularity_relative_tolerance),
        _number(reference._rotation_tolerance))


class CapturedReferenceIdentity:
    """Immutable captured authority; mismatch falls back to the ORIGINAL hash.

The fallback retains the source's canonical equivalences (for example -0/+0
or an identical copy), but never adopts a changed identity or replaces the
captured snapshot. All inputs are read on every call, including array contents.
"""
    __slots__ = ('_reference', '_snapshot', '_identity', '_methods', '_capture', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('captured reference identity is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, reference):
        self._reference = reference; self._methods = _methods(); self._snapshot = _snapshot(reference)
        self._identity = reference.fingerprint()
        if _snapshot(reference) != self._snapshot or _methods() != self._methods:
            raise ValueError('reference changed during identity capture')
        self._capture = (self._reference, self._snapshot, self._identity, self._methods)
        self._sealed = True

    def require(self, reference):
        if (reference is not self._reference or any(a is not b for a,b in zip(
                (self._reference, self._snapshot, self._identity, self._methods), self._capture))
                or _methods() != self._methods):
            raise ValueError('captured reference identity authority changed')
        try:
            unchanged = _snapshot(reference) == self._snapshot
        except (TypeError, ValueError, AttributeError, OverflowError):
            unchanged = False
        if not unchanged and reference.fingerprint() != self._identity:
            raise ValueError('captured retained generalized reference changed')
        return self._identity
