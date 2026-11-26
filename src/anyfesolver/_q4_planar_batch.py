"""Batched numerics of the qualified Q4 (E4-PL) compensated planar elastic route.

Private module.  It evaluates, over a leading element axis, exactly the
numerics of the *compensated planar elastic* branch of
``QualifiedE4PLShellElement.compute_stiffness_components`` (core, pl,
hourglass, numerical, total, jacobian_centre and the varying stationary-solve
diagnostics) and of the default ``compute_stresses`` (four Gauss records,
``return_global=False``, no ``reference_normal``, no Hill data), with
bit-identical results.

Separation of duties
--------------------
This module is numerics only.  The sealed caller in ``e4_pl_element`` performs
every per-element admission (exact element type, quadrature authority, cache
key, ``_elastic_increment_admitted``, warpage, ``equation7_frame``,
``_constitutive_and_drill_stiffness``, recovery admission) and binds the
per-element results through the existing component cache.  Here only the
already-admitted stacked arrays are processed:

* :func:`available` runs a bitwise known-answer self-check once (cached) and
  returns ``False`` (never raises) when this BLAS/CPU build does not
  reproduce the emulated helpers, or the kernels cannot be loaded.
* :func:`form` and :func:`recover` raise :class:`PlanarBatchFallback` on any
  input or certification failure.  The caller then re-runs the whole batch
  through the per-element path, which produces its own reference result or
  error.  Every guard of the per-element code (Jacobian, symmetry, Ruiz,
  backward error including nonfinite norms, finiteness) is kept; guards whose
  batched evaluation is not claimed bit-identical use a fail-closed margin,
  so this module never produces an output where the per-element code refuses.

Trust boundary
--------------
ANYfeSolver and library internals (numpy, numba, the compensated contraction
modules, ``e4_pl_element`` constants) are trusted not to be rebound at run
time.  Hostile *data* are in scope: inputs must be exact ``numpy.ndarray``
float64 C-contiguous arrays of the fixed shapes and finite, and are copied in;
everything else is refused to fallback.  Outputs are fresh arrays that alias
neither the inputs nor each other.  Mutation of inputs from another thread
during a call is out of contract.

Identity strategy (each item verified against the sealed per-element code)
------------------------------------------------------------------------
* Every compensated contraction (``product``/``weighted_sum``/``sum_arrays``)
  calls the same compiled kernel from ``_compensated_native`` once per element
  slice, or, when the sealed code runs its numpy policy (no Numba), the same
  numpy policy functions with a leading element axis.  A compensated dot is
  correctly rounded from its operands, so skipping exact-zero pairs and
  keeping pruned zero columns cannot change a value.
* Element-wise numpy expressions keep the sealed operand order.
* Small BLAS calls of the sealed helpers are emulated: ``_coefficients`` uses a
  4x4 gemv (this OpenBLAS build sums ``(p0 + p2) + (p1 + p3)``) and
  ``_residual_mode`` 4-term ``ddot`` (sequential).  All products there are
  exact, so only the summation order matters.  That order is a property of
  the installed BLAS kernel, not a contract, hence the known-answer
  self-check behind :func:`available`.
* ``_source_fields``/``_compatible`` matmuls have one nonzero term per output
  (or two exact terms) and are reproduced element-wise.
* ``np.linalg.solve`` is called on the stacked systems (the same LAPACK gesv
  per matrix through numpy's gufunc loop).
"""
from __future__ import annotations

import contextlib
import math
import threading

import numpy as np

from . import _compensated_contractions as _cc
from . import _compensated_native as _native
from . import e4_pl_element as _q4

__all__ = ["PlanarBatchFallback", "available", "form", "recover"]


class PlanarBatchFallback(ValueError):
    """The batch is not computed here; re-run it through the per-element path."""


# ---------------------------------------------------------------------------
# Availability (known-answer self-check) and the forced-fallback test hook
# ---------------------------------------------------------------------------

_STATE_LOCK = threading.Lock()
_SELF_CHECK = None  # None: not run; True: passed; str: failure description
_FORCED = 0


def available() -> bool:
    """True when the batched kernels reproduce the sealed helpers on this build."""
    global _SELF_CHECK
    if _FORCED:
        return False
    if _SELF_CHECK is None:
        with _STATE_LOCK:
            if _SELF_CHECK is None:
                try:
                    _self_check()
                except Exception as exc:  # noqa: BLE001 - any failure disables the route
                    _SELF_CHECK = f"{type(exc).__name__}: {exc}"
                else:
                    _SELF_CHECK = True
    return _SELF_CHECK is True and not _FORCED


@contextlib.contextmanager
def _forced_fallback():
    """Private test hook: :func:`available` returns False inside this context."""
    global _FORCED
    with _STATE_LOCK:
        _FORCED += 1
    try:
        yield
    finally:
        with _STATE_LOCK:
            _FORCED -= 1


def _known_unavailable() -> bool:
    """True when :func:`available` is already known to be False (no self-check run).

    Lets a caller skip its per-element batch front end cheaply; an undecided
    self-check stays undecided (lazy) until a batch reaches its minimum size.
    """
    return bool(_FORCED) or (_SELF_CHECK is not None and _SELF_CHECK is not True)


def _require_available():
    if not available():
        raise PlanarBatchFallback("batched Q4 planar numerics are not available")


# ---------------------------------------------------------------------------
# Batched compensated contractions (same compiled kernel per element slice)
# ---------------------------------------------------------------------------

# The kernels are module-level so that Numba's on-disk cache applies (the
# repository convention: ``_compensated_native._kernel`` caches when a cache
# locator exists and compiles uncached otherwise).  They call the same compiled
# contraction kernels of ``_compensated_native``.  Note: Numba invalidates a
# cached kernel by this file's source stamp only, as for the fused kernels of
# ``_compensated_native`` that call its helpers.
_contract_weighted = _native.contract_weighted
_contract_sum = _native.contract_sum
_prep = _native._prep
_residual_of = _native._residual
_fsum_buf = _native._fsum_buf


@_native._kernel
def _batched_weighted(stack, w, out, pow2):
    for e in range(stack.shape[0]):
        if _contract_weighted(stack[e], w[e], out[e], pow2) != 0:
            return e + 1
    return 0


@_native._kernel
def _batched_sum(stack, out):
    for e in range(stack.shape[0]):
        if _contract_sum(stack[e], out[e]) != 0:
            return e + 1
    return 0


@_native._kernel
def _batched_product(a, b, out, pow2):
    """``contract_product`` per slice, skipping pairs with an exact-zero operand.

    A pair with a zero operand contributes (+-0, 0) to the fsum buffer;
    fsum's result is the correctly rounded exact sum of its inputs, so the
    omitted zeros cannot change it (verified bitwise by the self-check).
    Nonfinite input is still refused by ``_prep``.
    """
    count, m, k = a.shape
    n = b.shape[2]
    ax = np.empty((m, k))
    ah = np.empty((m, k))
    al = np.empty((m, k))
    ae = np.empty((m, k), dtype=np.int64)
    bx = np.empty((k, n))
    bh = np.empty((k, n))
    bl = np.empty((k, n))
    be = np.empty((k, n), dtype=np.int64)
    buf = np.empty(2 * k)
    partials = np.empty(2 * k + 1)
    for e in range(count):
        ae_ = a[e]
        be_ = b[e]
        if not (_prep(ae_, ax, ah, al, ae) and _prep(be_, bx, bh, bl, be)):
            return e + 1
        for i in range(m):
            for j in range(n):
                used = 0
                for l in range(k):
                    av = ae_[i, l]
                    bv = be_[l, j]
                    if av == 0.0 or bv == 0.0:
                        continue
                    product = av * bv
                    if not math.isfinite(product):
                        return e + 1
                    buf[used] = product
                    buf[used + 1] = _residual_of(ax[i, l], ah[i, l], al[i, l], ae[i, l],
                                                 bx[l, j], bh[l, j], bl[l, j], be[l, j], pow2)
                    used += 2
                v, st = _fsum_buf(buf, used, partials)
                if st != 0:
                    return e + 1
                out[e, i, j] = v
    return 0


_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    if not (_native.AVAILABLE and _cc._USE_NATIVE):
        _KERNELS = False  # numpy path of the sealed policy (see _numpy_*)
        return _KERNELS
    _KERNELS = (_batched_product, _batched_weighted, _batched_sum)
    return _KERNELS


def _c(a):
    return np.ascontiguousarray(a, dtype=np.float64)


# Without the compiled kernels the sealed code runs the numpy policy
# (``_vector_parts`` + ``_interleave_sum``: math.fsum over interleaved
# product/residual pairs).  The same functions are applied with a leading
# element axis, in chunks to bound memory.  The sealed numpy product prunes
# all-zero operand columns per element; pruned terms are exact zeros and fsum
# is correctly rounded, so keeping them cannot change a value.
_CHUNK = 64


def _numpy_refusal(exc):
    return PlanarBatchFallback(f"compensated contraction refused: {type(exc).__name__}: {exc}")


def _numpy_product(a, b):
    out = np.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=np.float64)
    bt = np.swapaxes(b, 1, 2)
    try:
        for start in range(0, a.shape[0], _CHUNK):
            stop = start + _CHUNK
            prod, low = _cc._vector_parts(a[start:stop, :, None, :], bt[start:stop, None, :, :])
            out[start:stop] = _cc._interleave_sum(prod, low)
    except (ValueError, OverflowError) as exc:
        raise _numpy_refusal(exc) from exc
    return out


def _numpy_weighted(stack, w):
    try:
        prod, low = _cc._vector_parts(np.swapaxes(stack, 1, 2), w[:, None, :])
        return _cc._interleave_sum(prod, low)
    except (ValueError, OverflowError) as exc:
        raise _numpy_refusal(exc) from exc


def _numpy_sum(stack):
    values = np.swapaxes(stack, 1, 2)
    try:
        return _cc._interleave_sum(values, np.zeros_like(values))
    except (ValueError, OverflowError) as exc:
        raise _numpy_refusal(exc) from exc


def _bproduct(left, right):
    """Per-element ``_compensated_contractions.product`` over a leading axis.

    ``left`` is (E, m, k); ``right`` is (E, k, n) or (E, k) (matrix-vector).
    """
    a = _c(left)
    vector = right.ndim == 2
    b = _c(right[:, :, None] if vector else right)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise PlanarBatchFallback("batched compensated product shape mismatch")
    kernels = _kernels()
    if kernels is False:
        out = _numpy_product(a, b)
        return out[:, :, 0] if vector else out
    out = np.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=np.float64)
    status = kernels[0](a, b, out, _native.POW2)
    if status:
        raise PlanarBatchFallback(f"compensated product refused for batch element {status - 1}")
    return out[:, :, 0] if vector else out


def _bweighted(terms, weights):
    """Per-element ``weighted_sum``: ``terms`` list of (E, ...) arrays, weights (E, T)."""
    first = terms[0]
    count = first.shape[0]
    stack = _c(np.stack(terms, axis=1).reshape(count, len(terms), -1))
    w = _c(weights)
    if w.shape != (count, len(terms)):
        raise PlanarBatchFallback("batched weighted sum shape mismatch")
    kernels = _kernels()
    if kernels is False:
        return _numpy_weighted(stack, w).reshape(first.shape)
    out = np.empty((count, stack.shape[2]), dtype=np.float64)
    status = kernels[1](stack, w, out, _native.POW2)
    if status:
        raise PlanarBatchFallback(f"compensated weighted sum refused for batch element {status - 1}")
    return out.reshape(first.shape)


def _bsum(terms):
    """Per-element ``sum_arrays`` of equal-shape (E, ...) terms."""
    first = terms[0]
    count = first.shape[0]
    stack = _c(np.stack(terms, axis=1).reshape(count, len(terms), -1))
    if not np.isfinite(stack).all():
        raise PlanarBatchFallback("compensated array sum requires finite inputs")
    kernels = _kernels()
    if kernels is False:
        return _numpy_sum(stack).reshape(first.shape)
    out = np.empty((count, stack.shape[2]), dtype=np.float64)
    status = kernels[2](stack, out)
    if status:
        raise PlanarBatchFallback(f"compensated array sum refused for batch element {status - 1}")
    return out.reshape(first.shape)


def _bt(a):
    """Batched transpose of the last two axes."""
    return np.swapaxes(a, -1, -2)


# ---------------------------------------------------------------------------
# Batched element-wise replicas of the sealed planar helpers
# ---------------------------------------------------------------------------

_MODAL = np.asarray(((1, 1, 1, 1), (-1, 1, 1, -1), (-1, -1, 1, 1), (1, -1, 1, -1)),
                    dtype=float) / 4.0


def _gemv4(v):
    """Emulate ``_MODAL @ v`` for v (E, 4): returns 4 arrays (E,).

    Products by +-1/4 are exact; the installed 4x4 gemv kernel sums
    ``(p0 + p2) + (p1 + p3)`` (verified by the self-check).
    """
    out = []
    for i in range(4):
        p = _MODAL[i][None, :] * v
        out.append((p[:, 0] + p[:, 2]) + (p[:, 1] + p[:, 3]))
    return out


def _dot4(coefficients, v):
    """Emulate the 4-term ``ddot`` (sequential) for v (E, 4)."""
    p = np.asarray(coefficients, dtype=float)[None, :] * v
    return ((p[:, 0] + p[:, 1]) + p[:, 2]) + p[:, 3]


def _bcoefficients(local):
    x0, xr, xs, xrs = _gemv4(local[:, :, 0])
    y0, yr, ys, yrs = _gemv4(local[:, :, 1])
    return {
        "x0": x0, "xr": xr, "xs": xs, "xrs": xrs,
        "y0": y0, "yr": yr, "ys": ys, "yrs": yrs,
        "jc": xr * ys - xs * yr,
        "jr": xr * yrs - xrs * yr,
        "js": xrs * ys - xs * yrs,
    }


def _bjacobian(c, r, s):
    xr = c["xr"] + c["xrs"] * s
    xs = c["xs"] + c["xrs"] * r
    yr = c["yr"] + c["yrs"] * s
    ys = c["ys"] + c["yrs"] * r
    return xr, xs, yr, ys, xr * ys - xs * yr


def _shape(r, s):
    return np.asarray(((1.0 - r) * (1.0 - s), (1.0 + r) * (1.0 - s),
                       (1.0 + r) * (1.0 + s), (1.0 - r) * (1.0 + s)), dtype=float) / 4.0


def _shape_derivatives(r, s):
    return (np.asarray((-(1.0 - s), 1.0 - s, 1.0 + s, -(1.0 + s)), dtype=float) / 4.0,
            np.asarray((-(1.0 - r), -(1.0 + r), 1.0 + r, 1.0 - r), dtype=float) / 4.0)


def _two_term_dot(values, derivative):
    """``values @ derivative`` where ``derivative`` has at most two nonzero entries.

    Products by +-1/2 or 0 are exact and two nonzero terms sum to the same
    rounded value in any order; refuse any other derivative pattern.
    """
    nonzero = [i for i in range(4) if derivative[i] != 0.0]
    if len(nonzero) > 2 or any(abs(float(derivative[i])) not in (0.5, 0.25, 1.0)
                               for i in nonzero):
        raise PlanarBatchFallback("natural shear derivative pattern is not order independent")
    total = None
    for i in nonzero:
        term = values[:, i] * derivative[i]
        total = term if total is None else total + term
    return total if total is not None else np.zeros(values.shape[0])


def _bnatural_shear(local, r, s, direction):
    shape = _shape(r, s)
    nr, ns = _shape_derivatives(r, s)
    derivative = nr if direction == 0 else ns
    x_direction = _two_term_dot(local[:, :, 0], derivative)
    y_direction = _two_term_dot(local[:, :, 1], derivative)
    row = np.zeros((local.shape[0], 20), dtype=float)
    for index in range(4):
        base = 5 * index
        row[:, base + 2] = derivative[index]
        row[:, base + 3] = -y_direction * shape[index]
        row[:, base + 4] = x_direction * shape[index]
    return row


def _bshear_rows(local):
    return (_bnatural_shear(local, 0.0, -1.0, 0), _bnatural_shear(local, 0.0, 1.0, 0),
            _bnatural_shear(local, 1.0, 0.0, 1), _bnatural_shear(local, -1.0, 0.0, 1))


def _bcompatible(shear_rows, c, r, s):
    count = c["jc"].shape[0]
    nr, ns = _shape_derivatives(r, s)
    xr, xs, yr, ys, determinant = _bjacobian(c, r, s)
    nx = (ys[:, None] * nr[None, :] - yr[:, None] * ns[None, :]) / determinant[:, None]
    ny = (-xs[:, None] * nr[None, :] + xr[:, None] * ns[None, :]) / determinant[:, None]
    result = np.zeros((count, 8, 20), dtype=float)
    for index in range(4):
        base = 5 * index
        result[:, 0, base] = nx[:, index]
        result[:, 1, base + 1] = ny[:, index]
        result[:, 2, base] = ny[:, index]
        result[:, 2, base + 1] = nx[:, index]
        result[:, 3, base + 4] = nx[:, index]
        result[:, 4, base + 3] = -ny[:, index]
        result[:, 5, base + 4] = ny[:, index]
        result[:, 5, base + 3] = -nx[:, index]
    row_r_minus, row_r_plus, row_s_plus, row_s_minus = shear_rows
    row_r = 0.5 * (1.0 - s) * row_r_minus + 0.5 * (1.0 + s) * row_r_plus
    row_s = 0.5 * (1.0 + r) * row_s_plus + 0.5 * (1.0 - r) * row_s_minus
    result[:, 6] = (ys[:, None] * row_r - yr[:, None] * row_s) / determinant[:, None]
    result[:, 7] = (-xs[:, None] * row_r + xr[:, None] * row_s) / determinant[:, None]
    return result


def _btensor_transform(xr, xs, yr, ys, a, b):
    out = np.empty((xr.shape[0], 3, 3), dtype=float)
    out[:, 0, 0] = xr * xr
    out[:, 0, 1] = xs * xs
    out[:, 0, 2] = a * xr * xs
    out[:, 1, 0] = yr * yr
    out[:, 1, 1] = ys * ys
    out[:, 1, 2] = a * yr * ys
    out[:, 2, 0] = b * xr * yr
    out[:, 2, 1] = b * xs * ys
    out[:, 2, 2] = xr * ys + yr * xs
    return out


def _bsource_fields(c, r, s):
    jc, jr, js = c["jc"], c["jr"], c["js"]
    count = jc.shape[0]
    r_bar, s_bar = jr / (3.0 * jc), js / (3.0 * jc)
    stress_transform = _btensor_transform(c["xr"], c["xs"], c["yr"], c["ys"], 2.0, 1.0)
    strain_transform = _btensor_transform(c["xr"], c["xs"], c["yr"], c["ys"], 1.0, 2.0)
    n_sigma = np.zeros((count, 8, 14), dtype=float)
    n_epsilon = np.zeros((count, 8, 21), dtype=float)
    n_sigma[:, :, :8] = np.eye(8)
    n_epsilon[:, :, :8] = np.eye(8)
    seed_r = s - s_bar  # seed[0, 0]
    seed_s = r - r_bar  # seed[1, 1]
    # T @ seed with seed = [[a, 0], [0, b], [0, 0]]: one nonzero term per output.
    stress_vary = np.stack((stress_transform[:, :, 0] * seed_r[:, None],
                            stress_transform[:, :, 1] * seed_s[:, None]), axis=2)
    strain_vary = np.stack((strain_transform[:, :, 0] * seed_r[:, None],
                            strain_transform[:, :, 1] * seed_s[:, None]), axis=2)
    for row, column in ((0, 8), (3, 10)):
        n_sigma[:, row:row + 3, column:column + 2] = stress_vary
        n_epsilon[:, row:row + 3, column:column + 2] = strain_vary
    shear = np.empty((count, 2, 2), dtype=float)
    shear[:, 0, 0] = c["xr"] * seed_r
    shear[:, 0, 1] = c["xs"] * seed_s
    shear[:, 1, 0] = c["yr"] * seed_r
    shear[:, 1, 1] = c["ys"] * seed_s
    n_sigma[:, 6:8, 12:14] = shear
    n_epsilon[:, 6:8, 12:14] = shear
    rs = r * s
    st = strain_transform
    enriched = np.stack((st[:, :, 0] * r, st[:, :, 1] * s, st[:, :, 2] * r, st[:, :, 2] * s,
                         st[:, :, 0] * rs, st[:, :, 1] * rs, st[:, :, 2] * rs), axis=2)
    determinant = _bjacobian(c, r, s)[4]
    n_epsilon[:, :3, 14:21] = (jc / determinant)[:, None, None] * enriched
    return n_sigma, n_epsilon


def _bcentre_taylor(c):
    f0 = np.ones(4, dtype=float) / 4.0
    fr = np.asarray((-1, 1, 1, -1), dtype=float) / 4.0
    fs = np.asarray((-1, -1, 1, 1), dtype=float) / 4.0
    frs = np.asarray((1, -1, 1, -1), dtype=float) / 4.0
    jc, jr, js = c["jc"], c["jr"], c["js"]
    result = np.zeros((jc.shape[0], 3, 24), dtype=float)
    for coordinate in range(24):
        node, component = divmod(coordinate, 6)
        ur = fr[node] if component == 0 else 0.0
        us = fs[node] if component == 0 else 0.0
        urs = frs[node] if component == 0 else 0.0
        vr = fr[node] if component == 1 else 0.0
        vs = fs[node] if component == 1 else 0.0
        vrs = frs[node] if component == 1 else 0.0
        d0 = f0[node] if component == 5 else 0.0
        dr = fr[node] if component == 5 else 0.0
        ds = fs[node] if component == 5 else 0.0
        n0 = -c["xs"] * ur + c["xr"] * us - c["ys"] * vr + c["yr"] * vs
        nr = -c["xrs"] * ur + c["xr"] * urs - c["yrs"] * vr + c["yr"] * vrs
        ns = -c["xs"] * urs + c["xrs"] * us - c["ys"] * vrs + c["yrs"] * vs
        result[:, 0, coordinate] = d0 + n0 / (2.0 * jc)
        result[:, 1, coordinate] = dr + (nr * jc - n0 * jr) / (2.0 * jc * jc)
        result[:, 2, coordinate] = ds + (ns * jc - n0 * js) / (2.0 * jc * jc)
    return result


_XI = (-1.0, 1.0, 1.0, -1.0)
_ETA = (-1.0, -1.0, 1.0, 1.0)
_ALT = (1.0, -1.0, 1.0, -1.0)


def _bresidual_mode(local, c):
    centred_x = local[:, :, 0] - c["x0"][:, None]
    centred_y = local[:, :, 1] - c["y0"][:, None]
    xi = np.asarray(_XI)[None, :]
    eta = np.asarray(_ETA)[None, :]
    alternating = np.asarray(_ALT)[None, :]
    area = (4.0 * c["jc"])[:, None]
    b1 = (_dot4(_ETA, centred_y)[:, None] * xi
          - _dot4(_XI, centred_y)[:, None] * eta) / (4.0 * area)
    b2 = (-_dot4(_ETA, centred_x)[:, None] * xi
          + _dot4(_XI, centred_x)[:, None] * eta) / (4.0 * area)
    return (alternating - _dot4(_ALT, centred_x)[:, None] * b1
            - _dot4(_ALT, centred_y)[:, None] * b2) / 4.0


def _bglobal_transform(frame):
    transform = np.zeros((frame.shape[0], 24, 24), dtype=float)
    for node in range(4):
        transform[:, 6 * node:6 * node + 3, 6 * node:6 * node + 3] = frame
        transform[:, 6 * node + 3:6 * node + 6, 6 * node + 3:6 * node + 6] = frame
    return transform


def _bstationary_blocks(local, c, constitutive, shear_rows):
    """Compensated ``_stationary_blocks`` over the batch axis."""
    count = local.shape[0]
    f_terms, coupling_terms, strain_terms, gram_terms, weights = [], [], [], [], []
    for r, s in _q4._GAUSS:
        determinant = _bjacobian(c, r, s)[4]
        n_sigma, n_epsilon = _bsource_fields(c, r, s)
        compatible = _bcompatible(shear_rows, c, r, s)
        polynomial = np.asarray((1.0, r, s), dtype=float)
        n_epsilon_t = _bt(n_epsilon)
        f_terms.append(-_bproduct(n_epsilon_t, n_sigma))
        coupling_terms.append(_bproduct(_bt(n_sigma), compatible))
        strain_terms.append(_bproduct(_bproduct(n_epsilon_t, constitutive), n_epsilon))
        gram_terms.append(np.broadcast_to(np.outer(polynomial, polynomial), (count, 3, 3)))
        weights.append(determinant)
    w = np.stack(weights, axis=1)
    f_matrix = _bweighted(f_terms, w)
    coupling_20 = _bweighted(coupling_terms, w)
    strain_gram = _bweighted(strain_terms, w)
    gram = _bweighted(gram_terms, w)
    stationary = np.zeros((count, 35, 35), dtype=float)
    stationary[:, :14, 14:] = _bt(f_matrix)
    stationary[:, 14:, :14] = f_matrix
    stationary[:, 14:, 14:] = strain_gram
    coupling = np.zeros((count, 24, 35), dtype=float)
    physical_coupling = np.zeros((count, 20, 35), dtype=float)
    physical_coupling[:, :, :14] = _bt(coupling_20)
    for node in range(4):
        coupling[:, 6 * node:6 * node + 5] = physical_coupling[:, 5 * node:5 * node + 5]
    return stationary, coupling, gram


# Fail-closed margin for guards whose batched evaluation is not claimed to be
# bit-identical (norm-based guards).  A batch element whose guard value lies
# within this relative band of the sealed limit is sent to fallback.
_GUARD_MARGIN = 1.0e-6


def _inf_norm(a):
    """Batched ``np.linalg.norm(matrix, ord=np.inf)`` (max absolute row sum).

    The rows are made C-contiguous first so that each row sum runs over the
    contiguous axis, as numpy reduces a C-ordered 2-D operand.
    """
    return np.ascontiguousarray(np.abs(a)).sum(axis=2).max(axis=1)


def _solve_stationary(stationary, coupling):
    """Batched ``_solve_stationary_system`` (Ruiz congruence + certified solve)."""
    made = stationary
    if not np.all(np.isfinite(coupling)):
        raise PlanarBatchFallback("E4-PL stationary coupling must be finite")
    if not np.all(np.isfinite(made)):
        raise PlanarBatchFallback("E4-PL stationary system must be finite")
    eps = np.finfo(np.float64).eps
    with np.errstate(over="ignore", invalid="ignore"):
        input_norm = _inf_norm(made)
        asymmetry = _inf_norm(made - _bt(made))
    if not (np.all(np.isfinite(input_norm)) and np.all(np.isfinite(asymmetry))):
        raise PlanarBatchFallback("E4-PL stationary system norm is non-finite")
    if np.any(asymmetry > (1.0 - _GUARD_MARGIN) * 64.0 * eps * input_norm):
        raise PlanarBatchFallback("E4-PL stationary system symmetry guard is not resolved")
    equilibrated = 0.5 * (made + _bt(made))
    accumulated = np.ones(made.shape[:2], dtype=np.float64)
    for _iteration in range(_q4._STATIONARY_RUIZ_ITERATIONS):
        row_norms = np.max(np.abs(equilibrated), axis=2)
        if np.any(~np.isfinite(row_norms)) or np.any(row_norms <= 0.0):
            raise PlanarBatchFallback("E4-PL stationary system is singular")
        step = 1.0 / np.sqrt(row_norms)
        accumulated *= step
        equilibrated = step[:, :, None] * equilibrated * step[:, None, :]
    equilibrated = 0.5 * (equilibrated + _bt(equilibrated))
    final_row_norms = np.max(np.abs(equilibrated), axis=2)
    if (np.any(~np.isfinite(accumulated)) or np.any(accumulated <= 0.0)
            or np.any(~np.isfinite(equilibrated)) or np.any(~np.isfinite(final_row_norms))
            or np.any(final_row_norms <= 0.0)):
        raise PlanarBatchFallback("E4-PL stationary equilibration is unresolved")
    row_norm_ratio = np.max(final_row_norms, axis=1) / np.min(final_row_norms, axis=1)
    if (not np.all(np.isfinite(row_norm_ratio))
            or np.any(row_norm_ratio > _q4._STATIONARY_RUIZ_ROW_NORM_RATIO_LIMIT)):
        raise PlanarBatchFallback(
            "E4-PL stationary equilibration exceeded the row-norm ratio limit")
    rhs = _bt(coupling)
    scaled_rhs = accumulated[:, :, None] * rhs
    try:
        equilibrated_solution = np.linalg.solve(equilibrated, scaled_rhs)
    except np.linalg.LinAlgError as exc:
        raise PlanarBatchFallback("E4-PL stationary system is singular") from exc
    solution = accumulated[:, :, None] * equilibrated_solution
    with np.errstate(over="ignore", invalid="ignore"):
        residual = made @ solution - rhs
        matrix_norm = _inf_norm(made)
        solution_norm = _inf_norm(solution)
        # The per-element right-hand side is ``coupling.T``, an F-ordered
        # operand that numpy reduces along its strided axis; that summation
        # order differs from the contiguous row sum of ``_inf_norm``.  Take the
        # same call on the same per-element layout (Codex L4 round 1 P2).
        rhs_norm = np.array(
            [np.linalg.norm(coupling[index].T, ord=np.inf) for index in range(coupling.shape[0])],
            dtype=np.float64,
        )
        residual_norm = _inf_norm(residual)
        denominator = matrix_norm * solution_norm + rhs_norm
        backward = residual_norm / denominator
    norms_finite = all(np.all(np.isfinite(value)) for value in
                       (matrix_norm, solution_norm, rhs_norm, residual_norm, denominator))
    if (not norms_finite or not np.all(np.isfinite(solution)) or not np.all(np.isfinite(backward))
            or np.any(denominator <= 0.0)
            or np.any(backward > (1.0 - _GUARD_MARGIN) * _q4._STATIONARY_BACKWARD_ERROR_LIMIT)):
        raise PlanarBatchFallback("E4-PL stationary solve certification is not resolved")
    diagnostics = {
        "input_asymmetry_relative": asymmetry / np.maximum(input_norm, np.finfo(float).tiny),
        "row_norm_ratio": row_norm_ratio,
        "scale_max": np.max(accumulated, axis=1),
        "scale_min": np.min(accumulated, axis=1),
        "relative_backward_error": backward,
    }
    return solution, diagnostics


def _jacobian_guard(local, c, points):
    """Refuse nonpositive local Jacobians (sealed ``1e-12 * diameter**2`` guard).

    The diameter is evaluated with plain numpy (not claimed identical to the
    sealed FMA-based norm), so elements within ``_GUARD_MARGIN`` of the limit
    are sent to fallback rather than decided.
    """
    diffs = local[:, :, None, :] - local[:, None, :, :]
    length = np.sqrt(np.max(np.sum(diffs * diffs, axis=3), axis=(1, 2)))
    scale = length * length
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise PlanarBatchFallback("E4-PL local Jacobian scale must be positive and finite")
    determinants = [c["jc"], *(_bjacobian(c, r, s)[4] for r, s in points)]
    if np.any(np.min(np.stack(determinants), axis=0)
              <= (1.0 + _GUARD_MARGIN) * 1.0e-12 * scale):
        raise PlanarBatchFallback("E4-PL element has a nonpositive (or near-limit) local Jacobian")


# ---------------------------------------------------------------------------
# Input validation (exact float64 C-contiguous, fixed trailing shapes, finite)
# ---------------------------------------------------------------------------

def _admit(named, count=None):
    """Validate and copy ``(name, array, trailing_shape)`` triples."""
    copies = []
    for name, array, tail in named:
        if type(array) is not np.ndarray:
            raise PlanarBatchFallback(f"{name} must be an exact numpy.ndarray")
        if array.dtype != np.float64 or array.dtype.byteorder not in ("=", "|") \
                or not array.flags.c_contiguous:
            raise PlanarBatchFallback(f"{name} must be native float64 C-contiguous")
        if array.ndim != 1 + len(tail) or tuple(array.shape[1:]) != tail:
            raise PlanarBatchFallback(f"{name} has shape {array.shape}, expected (E, *{tail})")
        if count is None:
            count = array.shape[0]
        if array.shape[0] != count:
            raise PlanarBatchFallback("batched inputs disagree on the element count")
        made = np.array(array, dtype=np.float64, order="C", copy=True)
        if not np.all(np.isfinite(made)):
            raise PlanarBatchFallback(f"{name} must be finite")
        copies.append(made)
    if count is None or count < 1:
        raise PlanarBatchFallback("a batch needs at least one element")
    return copies


def _fresh(value):
    return np.array(value, dtype=np.float64, order="C", copy=True)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def form(frames, locals, constitutive, drill, pl_stabilization, hourglass_stabilization):
    """Batched compensated planar formation.

    Inputs (E = element count): ``frames`` (E, 3, 3), ``locals`` (E, 4, 2),
    ``constitutive`` (E, 8, 8), ``drill``, ``pl_stabilization`` and
    ``hourglass_stabilization`` (E,).  Returns ``core``, ``pl``,
    ``hourglass``, ``numerical``, ``total`` (E, 24, 24), ``jacobian_centre``
    (E,) and ``stationary_solve_diagnostics``: a dict of (E,) arrays with the
    varying entries of the sealed diagnostics (``input_asymmetry_relative``,
    ``row_norm_ratio``, ``scale_max``, ``scale_min``,
    ``relative_backward_error``); the constant entries (id, iterations,
    limits, disposition) are the sealed caller's.
    """
    _require_available()
    frames, local, constitutive, drill, pl_stab, hourglass_stab = _admit((
        ("frames", frames, (3, 3)), ("locals", locals, (4, 2)),
        ("constitutive", constitutive, (8, 8)), ("drill", drill, ()),
        ("pl_stabilization", pl_stabilization, ()),
        ("hourglass_stabilization", hourglass_stabilization, ()),
    ))
    if np.any(drill <= 0.0):
        raise PlanarBatchFallback("drilling stiffness must be positive")
    c = _bcoefficients(local)
    _jacobian_guard(local, c, _q4._GAUSS)
    shear_rows = _bshear_rows(local)
    stationary, coupling, gram = _bstationary_blocks(local, c, constitutive, shear_rows)
    solution, diagnostics = _solve_stationary(stationary, coupling)
    core_local = -_bproduct(coupling, solution)
    core_local = 0.5 * (core_local + _bt(core_local))
    centre = _bcentre_taylor(c)
    pl_core = _bproduct(_bproduct(_bt(centre), gram), centre)
    pl_local = (pl_stab * drill)[:, None, None] * pl_core
    gamma = _bresidual_mode(local, c)
    count = local.shape[0]
    gamma_24 = np.zeros((count, 24), dtype=float)
    gamma_24[:, 5::6] = gamma
    area = 4.0 * c["jc"]
    scale = 2.0 * hourglass_stab * drill * area
    hourglass_local = scale[:, None, None] * (gamma_24[:, :, None] * gamma_24[:, None, :])
    transform = _bglobal_transform(frames)
    transform_t = _bt(transform)
    core = _bproduct(_bproduct(transform, core_local), transform_t)
    pl = _bproduct(_bproduct(transform, pl_local), transform_t)
    hourglass = _bproduct(_bproduct(transform, hourglass_local), transform_t)
    core = 0.5 * (core + _bt(core))
    pl = 0.5 * (pl + _bt(pl))
    hourglass = 0.5 * (hourglass + _bt(hourglass))
    numerical = _bsum((pl, hourglass))
    total = _bsum((core, numerical))
    result = {"core": core, "pl": pl, "hourglass": hourglass, "numerical": numerical,
              "total": total, "jacobian_centre": c["jc"]}
    for name, values in result.items():
        if not np.all(np.isfinite(values)):
            raise PlanarBatchFallback(f"qualified Q4 formation produced non-finite {name!r}")
    result = {name: _fresh(values) for name, values in result.items()}
    result["stationary_solve_diagnostics"] = {
        name: _fresh(values) for name, values in diagnostics.items()}
    return result


def recover(frames, locals, constitutive, thickness, element_displacements):
    """Batched default-point planar recovery (``return_global=False``).

    ``element_displacements`` is (E, 24), the element vectors in DOF-mapping
    order.  Returns a dict of fresh arrays with a leading element axis, keyed
    and shaped per element as the array fields of ``compute_stresses``; the
    constant entries (ids, scopes, flags, orders, natural points) are the
    sealed caller's.  Only for elements admitted without ``reference_normal``
    and without Hill yield data (identity physical maps, von Mises measure).
    """
    _require_available()
    frames, local, constitutive, thickness, u = _admit((
        ("frames", frames, (3, 3)), ("locals", locals, (4, 2)),
        ("constitutive", constitutive, (8, 8)), ("thickness", thickness, ()),
        ("element_displacements", element_displacements, (24,)),
    ))
    if np.any(thickness <= 0.0):
        raise PlanarBatchFallback("qualified Q4 recovery requires positive finite thickness")
    points = _q4._GAUSS
    count = local.shape[0]
    transform = _bglobal_transform(frames)
    local_displacement = _bproduct(_bt(transform), u)
    c = _bcoefficients(local)
    _jacobian_guard(local, c, points)
    shear_rows = _bshear_rows(local)
    stationary, coupling, _gram = _bstationary_blocks(local, c, constitutive, shear_rows)
    solution, _diagnostics = _solve_stationary(stationary, coupling)
    parameters = -_bproduct(solution, local_displacement)
    stress_parameters = parameters[:, :14]
    strain_parameters = parameters[:, 14:]
    residual = _bsum((_bproduct(stationary, parameters),
                      _bproduct(_bt(coupling), local_displacement)))
    physical_displacement = np.concatenate(
        tuple(local_displacement[:, 6 * node:6 * node + 5] for node in range(4)), axis=1)
    npts = len(points)
    compatible = np.zeros((count, npts, 8), dtype=float)
    independent = np.zeros_like(compatible)
    resultants = np.zeros_like(compatible)
    for index, (r, s) in enumerate(points):
        n_sigma, n_epsilon = _bsource_fields(c, float(r), float(s))
        compatible_matrix = _bcompatible(shear_rows, c, float(r), float(s))
        compatible[:, index] = _bproduct(compatible_matrix, physical_displacement)
        independent[:, index] = _bproduct(n_epsilon, strain_parameters)
        resultants[:, index] = _bproduct(n_sigma, stress_parameters)
    for name, values in (("local_displacement", local_displacement),
                         ("stationary_parameters", parameters),
                         ("stationarity_residual", residual), ("compatible", compatible),
                         ("independent", independent), ("resultants", resultants)):
        if not np.all(np.isfinite(values)):
            raise PlanarBatchFallback(
                f"qualified Q4 mixed recovery produced non-finite field {name!r}")
    # Without reference_normal the physical maps are identities.  Compensated
    # results are never -0.0, hence ``x @ I == x`` bitwise; ``+ 0.0`` keeps the
    # +0 convention of a matmul accumulator defensively.
    independent = independent + 0.0
    compatible = compatible + 0.0
    resultants = resultants + 0.0
    t = thickness[:, None]
    membrane_stress = resultants[:, :, :3] / t[:, :, None]
    bending_stress = 6.0 * resultants[:, :, 3:6] / (t * t)[:, :, None]
    transverse = resultants[:, :, 6:] / t[:, :, None]
    top = membrane_stress + bending_stress
    bottom = membrane_stress - bending_stress
    vm_top = np.sqrt(top[..., 0] ** 2 - top[..., 0] * top[..., 1] + top[..., 1] ** 2
                     + 3.0 * (top[..., 2] ** 2 + transverse[..., 0] ** 2
                              + transverse[..., 1] ** 2))
    vm_bottom = np.sqrt(bottom[..., 0] ** 2 - bottom[..., 0] * bottom[..., 1]
                        + bottom[..., 1] ** 2
                        + 3.0 * (bottom[..., 2] ** 2 + transverse[..., 0] ** 2
                                 + transverse[..., 1] ** 2))
    von_mises = np.maximum(vm_top, vm_bottom)
    fields = {
        "membrane_strain": independent[:, :, :3],
        "curvature": independent[:, :, 3:6],
        "transverse_shear_strain": independent[:, :, 6:],
        "compatible_membrane_strain": compatible[:, :, :3],
        "compatible_curvature": compatible[:, :, 3:6],
        "compatible_transverse_shear_strain": compatible[:, :, 6:],
        "membrane_resultants": resultants[:, :, :3],
        "bending_resultants": resultants[:, :, 3:6],
        "transverse_shear_resultants": resultants[:, :, 6:],
        "physical_director": frames[:, :, 2],
        "membrane_xx": membrane_stress[..., 0], "membrane_yy": membrane_stress[..., 1],
        "membrane_xy": membrane_stress[..., 2],
        "bending_xx": bending_stress[..., 0], "bending_yy": bending_stress[..., 1],
        "bending_xy": bending_stress[..., 2],
        "shear_xz": transverse[..., 0], "shear_yz": transverse[..., 1],
        "von_mises": von_mises,
        "hill_utilization": np.zeros((count, npts), dtype=float),
        "equivalent_stress": von_mises,
    }
    for name, values in fields.items():
        if not np.all(np.isfinite(values)):
            raise PlanarBatchFallback(f"qualified Q4 recovery produced non-finite field {name!r}")
    return {name: _fresh(values) for name, values in fields.items()}


# ---------------------------------------------------------------------------
# Known-answer self-check against the sealed helpers
# ---------------------------------------------------------------------------

def _self_check_cases(count=48):
    rng = np.random.default_rng(20261124)
    cases = []
    for _ in range(count):
        scale = 10.0 ** rng.uniform(-4, 4)
        base = np.asarray(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
        local = (base + rng.uniform(-0.3, 0.3, (4, 2))) * scale
        local = local - local.mean(axis=0)
        cases.append(local)
    return np.asarray(cases)


def _self_check():
    """Bitwise known-answer check of every emulated helper; raises on mismatch."""
    _kernels()
    prng = np.random.default_rng(20261125)
    for m, k, n in ((7, 9, 6), (24, 24, 24), (24, 35, 24), (8, 20, 1)):
        a = prng.standard_normal((5, m, k)) * 2.0 ** prng.integers(-40, 40, (5, m, k))
        b = prng.standard_normal((5, k, n)) * 2.0 ** prng.integers(-40, 40, (5, k, n))
        a[prng.random(a.shape) < 0.5] = 0.0
        b[prng.random(b.shape) < 0.3] = -0.0
        a[0, 0, :] = 0.0  # an all-zero dot
        got = _bproduct(a, b)
        for e in range(5):
            if got[e].tobytes() != _cc.product(a[e], b[e]).tobytes():
                raise AssertionError("batched compensated product")
    locals_ = _self_check_cases()
    rng = np.random.default_rng(7)
    constitutive = np.zeros((len(locals_), 8, 8))
    for e in range(len(locals_)):
        a = rng.standard_normal((8, 8))
        constitutive[e] = a @ a.T + 8.0 * np.eye(8)
    c = _bcoefficients(locals_)
    shear_rows = _bshear_rows(locals_)
    stationary, coupling, gram = _bstationary_blocks(locals_, c, constitutive, shear_rows)
    centre = _bcentre_taylor(c)
    gamma = _bresidual_mode(locals_, c)
    for e, local in enumerate(locals_):
        ref_c = _q4._coefficients(local)
        for key, value in ref_c.items():
            if np.float64(value).tobytes() != c[key][e].tobytes():
                raise AssertionError(f"_coefficients[{key}]")
        ce = {key: float(value[e]) for key, value in c.items()}
        for r, s in _q4._GAUSS:
            ns_ref, ne_ref = _q4._source_fields(ref_c, r, s)
            ns, ne = _bsource_fields({k: v[e:e + 1] for k, v in c.items()}, r, s)
            if not (np.array_equal(ns_ref, ns[0]) and np.array_equal(ne_ref, ne[0])):
                raise AssertionError("_source_fields")
            comp = _bcompatible(tuple(row[e:e + 1] for row in shear_rows),
                                {k: v[e:e + 1] for k, v in c.items()}, r, s)
            if not np.array_equal(_q4._compatible(local, ce, r, s), comp[0]):
                raise AssertionError("_compatible")
        st_ref, cp_ref, gr_ref = _q4._stationary_blocks(local, ref_c, constitutive[e],
                                                        compensated=True)
        for ref, got, label in ((st_ref, stationary[e], "stationary"),
                                (cp_ref, coupling[e], "coupling"),
                                (gr_ref, gram[e], "gram"),
                                (_q4._centre_taylor(ref_c), centre[e], "centre"),
                                (_q4._residual_mode(local, ref_c), gamma[e], "residual")):
            if ref.tobytes() != np.ascontiguousarray(got).tobytes():
                raise AssertionError(label)
