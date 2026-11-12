"""Fixed binary64 contractions for native contact curves and gap differentials.

These are the existing nine-by-three work maps, with their evaluation order
specified explicitly. They do not select a BLAS backend or round a result for
serialization. The first eight vector rows accumulate the first product into
the second; the final row accumulates the second into the first. Both then
add the separately rounded third product. This preserves the archived work
map's vector-block and scalar-tail evaluation order on every host. Q2 curves
and three-coordinate matrix products use ascending fused accumulation.
"""
import ctypes
import math
import os
from types import BuiltinMethodType

import numpy as np

from .._ge_beam3_deterministic_reduction import _fma as _portable_fma


def _load_fma():
    # Python 3.13+ already supplies the native primitive through the shared
    # implementation. Earlier versions can use the same C99 operation from
    # the system runtime, retaining the exact rational implementation when
    # the symbol is unavailable. Never load a workspace DLL on Windows.
    if hasattr(math, 'fma'):
        return _portable_fma
    try:
        runtime = (ctypes.CDLL('ucrtbase.dll', winmode=0x800)
                   if os.name == 'nt' else ctypes.CDLL(None))
        function = runtime.fma
        function.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
        function.restype = ctypes.c_double
        return function
    except (AttributeError, OSError):
        return _portable_fma


_fma = _load_fma()
# Eligibility only: arithmetic below still resolves the live _fma binding.
# Python/fallback hooks retain their original operand-read/callback order.
_native_fma = _fma if isinstance(_fma, ctypes._CFuncPtr) else None
_native_double = ctypes.c_double
_native_from_param = ctypes.c_double.from_param


def _native_preparation_allowed():
    function = _fma
    if function is not _native_fma or function is None:
        return False
    if ctypes.c_double is not _native_double:
        return False
    conversion = ctypes.c_double.from_param
    if type(conversion) is not BuiltinMethodType or conversion != _native_from_param:
        return False
    argtypes = function.argtypes
    return (type(argtypes) is tuple and len(argtypes) == 3
            and all(item is _native_double for item in argtypes)
            and function.restype is _native_double
            and getattr(function, 'errcheck', None) is None)


_S3_ORDER = 18
_S3_TAIL_COLUMNS = (16, 17)


def s3_congruence_product(transform, matrix):
    """Fixed-order fused ``transform.T @ matrix @ transform`` for 18x18 S3 data.

    This reproduces the archived S3 shell-stiffness globalization bits for the
    contact checkpoints whose first product ``transform.T @ matrix`` and second
    product ``first @ transform`` were observed on a SkylakeX OpenBLAS kernel.
    It is a fixed arithmetic specification only; it makes no claim about any
    BLAS kernel or any other operand set.

    First product: every entry is an ascending-index fused accumulation from
    +0. Second product: columns 0..15 use the same single ascending fused
    accumulation; columns 16 and 17 use two independent accumulators, one for
    even and one for odd inner indices, added once at the end. The result is
    not symmetrized or rounded here.
    """
    left = np.asarray(transform, dtype=float)
    right = np.asarray(matrix, dtype=float)
    if left.shape != (_S3_ORDER, _S3_ORDER) or right.shape != left.shape:
        raise ValueError('S3 congruence product requires 18x18 operands')
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError('S3 congruence product requires finite operands')
    t_columns = [list(map(float, column)) for column in left.T]   # rows of T.T
    m_columns = [list(map(float, column)) for column in right.T]
    first = np.empty((_S3_ORDER, _S3_ORDER))
    for i, a_row in enumerate(t_columns):
        for j, b_col in enumerate(m_columns):
            acc = 0.
            for a, b in zip(a_row, b_col):
                acc = _fma(a, b, acc)
            first[i, j] = acc
    rows = [list(map(float, row)) for row in first]
    t_cols = [list(map(float, column)) for column in left.T]
    result = np.empty((_S3_ORDER, _S3_ORDER))
    for i, a_row in enumerate(rows):
        for j, b_col in enumerate(t_cols):
            if j in _S3_TAIL_COLUMNS:
                even = odd = 0.
                for k in range(0, _S3_ORDER, 2):
                    even = _fma(a_row[k], b_col[k], even)
                    odd = _fma(a_row[k+1], b_col[k+1], odd)
                result[i, j] = even + odd
            else:
                acc = 0.
                for a, b in zip(a_row, b_col):
                    acc = _fma(a, b, acc)
                result[i, j] = acc
    return result


def gap_vector_product(matrix, vector):
    """The fixed 9x3 gap map applied to a three-component spatial vector."""
    matrix = np.asarray(matrix, dtype=float)
    vector = np.asarray(vector, dtype=float)
    if matrix.shape != (9, 3) or vector.shape != (3,):
        raise ValueError('native gap vector contraction requires 9x3 and 3')
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        raise ValueError('native gap contraction requires finite operands')
    x, y, z = map(float, vector)
    result = np.empty(9)
    for i, row in enumerate(matrix):
        a, b, c = map(float, row)
        pair = _fma(a, x, b*y) if i < 8 else _fma(b, y, a*x)
        result[i] = pair + c*z
    return result


def gap_matrix_product(left, right):
    """Fixed ascending fused accumulation of three matrix-product terms."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != 3 or right.shape[0] != 3:
        raise ValueError('native gap matrix contraction requires three inner coordinates')
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError('native gap contraction requires finite operands')
    result = np.empty((left.shape[0], right.shape[1]))
    if left.shape == (9, 3) and right.shape == (3, 9) and _native_preparation_allowed():
        # One finite binary64 operand snapshot per call, never retained on a
        # candidate. Conversion preserves IEEE signed zeros. No result cache.
        columns = tuple(tuple(map(float, column)) for column in right.T)
        for i, row in enumerate(left):
            a, b, c = map(float, row)
            for j, (x, y, z) in enumerate(columns):
                result[i, j] = _fma(c, z, _fma(b, y, _fma(a, x, 0.)))
        return result
    for i, row in enumerate(left):
        a, b, c = map(float, row)
        for j, column in enumerate(right.T):
            x, y, z = map(float, column)
            result[i, j] = _fma(c, z, _fma(b, y, _fma(a, x, 0.)))
    return result
