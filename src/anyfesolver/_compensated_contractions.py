"""Finite, fixed-order element-local contractions (no BLAS dispatch).

Products are split on normalized binary64 mantissas with Dekker's transform;
both parts are accumulated with ``math.fsum`` in ascending index order.
No exact-rounding claim is made: scaling a residual into the subnormal range
can lose at most one minimum subnormal per product, and fsum has its documented
platform double-rounding limitation. Exact cancellation/empty sums give +0.
Nonfinite inputs and overflowing individual products or sums are refused,
even when a wider exact accumulator could cancel an intermediate overflow.
There is no precision option, persistent result cache or fallback that changes
a value. Optional compiled kernels (``_compensated_native``, Numba) execute the
same arithmetic bit for bit; they are enabled only when Numba is present and a
known-answer self-check at import reproduces the numpy path bitwise (it guards
against fused multiply-add contraction). Any refusal replays the numpy path so
the reference exception is raised.
"""
from math import frexp as _frexp, fsum as _fsum, isfinite as _isfinite, ldexp as _ldexp

import os
from types import MappingProxyType as _MappingProxyType

import numpy as np

from . import _compensated_native as _native

# Compiled kernels run the identical arithmetic (see _compensated_native); a
# nonzero kernel status replays the vectorised path, which raises the reference
# exception. ANYSOLVER_NATIVE_CONTRACTIONS=0 forces the numpy path.
_USE_NATIVE = _native.AVAILABLE and os.environ.get("ANYSOLVER_NATIVE_CONTRACTIONS", "1") != "0"
# The fused element kernels additionally need their own known-answer match
# (checked at import below); ANYSOLVER_NATIVE_CONTRACTIONS=0 disables both.
_USE_FUSED = _USE_NATIVE


def _product_parts(a, b):
    if not _isfinite(a) or not _isfinite(b):
        raise ValueError("compensated contraction requires finite inputs")
    product = a * b
    if not _isfinite(product):
        raise OverflowError("compensated product overflow")
    if a == 0.0 or b == 0.0:
        return product, 0.0
    x, ex = _frexp(a)
    y, ey = _frexp(b)
    split = 134217729.0
    cx, cy = split * x, split * y
    xh, yh = cx - (cx - x), cy - (cy - y)
    xl, yl = x - xh, y - yh
    high = x * y
    low = ((xh * yh - high) + xh * yl + xl * yh) + xl * yl
    # In the normal range power-of-two scaling is exact. In the subnormal
    # range the rounded product need not equal a doubly rounded scaled high;
    # the contract allows that lost half-ulp and scaled-low rounding (<= one
    # minimum subnormal together), rather than claiming an exact expansion.
    return product, _ldexp(low, ex + ey)


_SPLIT = 134217729.0


def _vector_parts(a, b):
    """Elementwise ``_product_parts`` for finite broadcastable arrays.

    Same IEEE operations in the same order as the scalar routine, so the
    (product, residual) pairs are bitwise identical. Refuses nonfinite inputs
    and overflowing products with the scalar routine's exceptions.
    """
    with np.errstate(all="ignore"):
        product = a * b
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                and np.all(np.isfinite(product))):
            # Rare error path: reproduce the scalar first-failure semantics.
            for x, y in zip(np.broadcast_to(a, product.shape).ravel().tolist(),
                            np.broadcast_to(b, product.shape).ravel().tolist()):
                _product_parts(x, y)
        x, ex = np.frexp(a)
        y, ey = np.frexp(b)
        cx, cy = _SPLIT * x, _SPLIT * y
        xh, yh = cx - (cx - x), cy - (cy - y)
        xl, yl = x - xh, y - yh
        high = x * y
        low = ((xh * yh - high) + xh * yl + xl * yh) + xl * yl
        low = np.ldexp(low, ex + ey)
        low = np.where((a == 0.0) | (b == 0.0), 0.0, low)
    return product, low


def _interleave_sum(product, low):
    """fsum of ``[p0, l0, p1, l1, ...]`` along the last axis, as the scalar dot."""
    lead = product.shape[:-1]
    count = product.shape[-1]
    parts = np.empty((product.size // max(count, 1), 2 * count), dtype=float)
    parts[:, 0::2] = product.reshape(-1, count)
    parts[:, 1::2] = low.reshape(-1, count)
    values = np.array(list(map(_fsum, parts.tolist())), dtype=float)
    if not np.all(np.isfinite(values)):
        raise OverflowError("compensated sum overflow")
    return values.reshape(lead)


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("compensated dot shape mismatch")
    a = np.array([float(v) for v in left], dtype=float)
    b = np.array([float(v) for v in right], dtype=float)
    if a.size == 0:
        return 0.0
    product, low = _vector_parts(a, b)
    return float(_interleave_sum(product, low))


def product(left, right):
    """Matrix/matrix or matrix/vector product; each dot uses the same policy."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.ndim != 2 or b.ndim not in (1, 2) or a.shape[1] != b.shape[0]:
        raise ValueError("compensated product shape mismatch")
    vector = b.ndim == 1
    columns = b.reshape((-1, 1)) if vector else b
    if _USE_NATIVE:
        # The kernel refuses nonfinite input itself; any refusal falls through
        # to the checks below, which raise the reference exception.
        native = np.empty((a.shape[0], columns.shape[1]), dtype=float)
        if _native.contract_product(np.ascontiguousarray(a), np.ascontiguousarray(columns),
                                    native, _native.POW2) == 0:
            return native[:, 0] if vector else native
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("compensated product requires finite inputs")
    result = np.zeros((a.shape[0], columns.shape[1]), dtype=float)
    # Terms with an all-zero operand slice are exact zeros in every dot and
    # cannot change an exactly rounded sum.
    keep = np.any(a != 0.0, axis=0) & np.any(columns != 0.0, axis=1)
    # Overflow in a pruned term would still be refused by the scalar policy
    # only if both factors were nonzero, so pruning never hides an error.
    if keep.any():
        a, columns = a[:, keep], columns[keep]
        prod, low = _vector_parts(a[:, None, :], columns.T[None, :, :])
        result = _interleave_sum(prod, low)
    return result[:, 0] if vector else result


def sum_arrays(terms):
    """Fixed-order componentwise sum of already represented equal-shape terms."""
    arrays = [np.asarray(term, dtype=float) for term in terms]
    if not arrays or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("compensated array sum requires equal nonempty shapes")
    shape = arrays[0].shape
    if _USE_NATIVE:
        # contract_sum refuses overflow only; nonfinite input is screened here
        # on the single stacked array instead of once per term.
        stack = np.ascontiguousarray(np.stack(arrays).reshape(len(arrays), -1))
        if np.isfinite(stack).all():
            native = np.empty(stack.shape[1], dtype=float)
            if _native.contract_sum(stack, native) == 0:
                return native.reshape(shape)
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("compensated array sum requires finite inputs")
    rows = np.stack(arrays).reshape(len(arrays), -1).T.tolist()
    result = np.array(list(map(_fsum, rows)), dtype=float).reshape(shape)
    if not np.all(np.isfinite(result)):
        raise OverflowError("compensated array sum overflow")
    return result


def weighted_sum(terms, weights):
    """Quadrature reduction including compensation of each weight product."""
    arrays = [np.asarray(term, dtype=float) for term in terms]
    if not arrays or len(arrays) != len(weights) or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("compensated weighted sum shape mismatch")
    shape = arrays[0].shape
    try:
        if _USE_NATIVE:
            flat = np.ascontiguousarray(np.stack(arrays).reshape(len(arrays), -1))
            w = np.array([float(v) for v in weights], dtype=float)
            native = np.empty(flat.shape[1], dtype=float)
            if _native.contract_weighted(flat, w, native, _native.POW2) == 0:
                return native.reshape(shape)
        stacked = np.stack(arrays).reshape(len(arrays), -1).T
        w = np.array([float(v) for v in weights], dtype=float)
        prod, low = _vector_parts(stacked, w[None, :])
        return _interleave_sum(prod, low).reshape(shape)
    except (ValueError, OverflowError):
        # Refused input: replay index by index so the first failure (and its
        # exception type) is the one the scalar policy reports.
        result = np.empty(shape, dtype=float)
        for index in np.ndindex(shape):
            result[index] = _scalar_dot([a[index] for a in arrays], weights)
        return result


def _scalar_dot(left, right):
    parts = []
    for a, b in zip(left, right):
        parts.extend(_product_parts(float(a), float(b)))
    result = _fsum(parts)
    if not _isfinite(result):
        raise OverflowError("compensated sum overflow")
    return result


def _shell_zero_tangent_reference(geometry, membrane, bending, shear, drill):
    """Contract independently supplied elastic coefficients on frozen geometry."""
    terms, weights = [], []
    for bm, bb, bd, weight in zip(geometry["B_m_all"], geometry["B_b_all"],
                                  geometry["B_d_all"], geometry["detw_all"]):
        terms.append(sum_arrays((product(product(bm.T, membrane), bm),
                                 product(product(bb.T, bending), bb),
                                 product(bd.T, drill * bd))))
        weights.append(weight)
    for bs, weight in zip(geometry["B_s_all"], geometry["detw_shear_all"]):
        terms.append(product(product(bs.T, shear), bs))
        weights.append(weight)
    local = weighted_sum(terms, weights)
    transform = geometry["T0"]
    return product(product(transform.T, local), transform)


def _increment(u_elem, geometry, constitutive_factory, tangent, use_kernel):
    """Elastic von-Karman increment: the reference evaluation order, optionally fused.

    Order (identical for both routes): read ``T0``; the displacement product
    (its result ``u`` is computed once and reused); one coefficient evaluation;
    only then, at the moment of choosing, the numpy error policy is read and the
    compiled kernel is tried with ``u`` and the cached coefficients. A policy
    that is not default-or-ignore, non-conforming input or any kernel refusal
    falls through to the reference loop below with the same ``u``/coefficients.
    """
    transform = geometry["T0"]
    u = product(transform, u_elem)
    constitutive = constitutive_factory()
    if use_kernel and _fused_policy_allows():
        fused = _increment_fused(transform, u, geometry, constitutive, tangent)
        if fused is not None:
            return fused
    force = np.zeros(u.size, dtype=float)
    material = np.zeros((u.size, u.size), dtype=float)
    geometric = np.zeros((u.size, u.size), dtype=float)
    forces, materials, geometrics = [], [], []
    weights = []
    for B, G, weight in zip(geometry["B_m_all"], geometry["Gw_all"],
                            geometry["detw_all"]):
        theta = product(G, u)
        eta = np.array((0.5 * theta[0]**2, 0.5 * theta[1]**2,
                        theta[0] * theta[1]), dtype=float)
        H = np.vstack((theta[0] * G[0], theta[1] * G[1],
                       theta[0] * G[1] + theta[1] * G[0]))
        linear_stress = product(constitutive, product(B, u))
        nonlinear_stress = product(constitutive, eta)
        forces.append(sum_arrays((product(B.T, nonlinear_stress),
                                  product(H.T, linear_stress),
                                  product(H.T, nonlinear_stress))))
        weights.append(weight)
        if tangent:
            materials.append(sum_arrays((product(product(B.T, constitutive), H),
                                         product(product(H.T, constitutive), B),
                                         product(product(H.T, constitutive), H))))
            stress = sum_arrays((linear_stress, nonlinear_stress))
            geometrics.append(sum_arrays((stress[0] * np.outer(G[0], G[0]),
                                          stress[1] * np.outer(G[1], G[1]),
                                          stress[2] * np.outer(G[0], G[1]),
                                          stress[2] * np.outer(G[1], G[0]))))
    if forces:
        force = weighted_sum(forces, weights)
    if tangent and materials:
        material = weighted_sum(materials, weights)
        geometric = weighted_sum(geometrics, weights)
    return (product(transform.T, force), product(product(transform.T, material), transform),
            product(product(transform.T, geometric), transform))


def elastic_increment_reference(u_elem, geometry, constitutive_factory, tangent):
    """Reference (numpy-level) elastic von-Karman increment; see vectorized_nonlinear."""
    return _increment(u_elem, geometry, constitutive_factory, tangent, False)


def elastic_increment(u_elem, geometry, constitutive_factory, tangent):
    """The same increment with the compiled kernel tried after the coefficient evaluation."""
    return _increment(u_elem, geometry, constitutive_factory, tangent, True)


def _fused_array(value, shape_tail):
    """Writable C float64 array of exactly ``1 + len(shape_tail)`` dims, or ``None``."""
    if (type(value) is not np.ndarray or value.dtype != np.float64
            or value.ndim != len(shape_tail) + 1 or value.shape[1:] != shape_tail):
        return None
    if value.flags.c_contiguous and value.flags.writeable:
        return value
    return np.array(value, dtype=np.float64, order="C")


# Exact mapping types the qualified route supplies (frozen geometry is a mappingproxy).
_GEOMETRY_TYPES = (dict, _MappingProxyType)


def _fused_matrix(value, shape):
    """Writable C float64 copy-or-view of an exact ndarray with exactly ``shape``, or ``None``."""
    if type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != shape:
        return None
    if value.flags.c_contiguous and value.flags.writeable:
        return value
    return np.array(value, dtype=np.float64, order="C")


def _fused_scalar(value):
    if type(value) in (float, np.float64):
        return float(value)
    return None


def _fused_policy_allows():
    """Fused kernels run only under numpy's default-or-ignore floating-point policy.

    The reference performs plain numpy multiplies (e.g. ``drill * bd``) for which
    numpy may warn, raise or call back on underflow; the compiled multiplies
    cannot. Overflow/invalid always make the kernel refuse (the nonfinite value
    reaches a compensated product or sum), so the reference replays and reports
    them itself. Any other policy runs the reference.
    """
    if not (_USE_NATIVE and _USE_FUSED):
        return False
    policy = np.geterr()
    return (policy["under"] == "ignore"
            and all(policy[kind] in ("ignore", "warn") for kind in ("over", "divide", "invalid")))


def _zero_tangent_args(geometry, membrane, bending, shear, drill):
    """Kernel arguments for ``shell_zero_tangent``, or ``None`` for any non-conforming input."""
    try:
        if type(geometry) not in _GEOMETRY_TYPES:
            return None
        t0 = geometry["T0"]
        n = t0.shape[0] if type(t0) is np.ndarray and t0.ndim == 2 else -1
        if n <= 0:
            return None
        arrays = (_fused_array(geometry["B_m_all"], (3, n)), _fused_array(geometry["B_b_all"], (3, n)),
                  _fused_array(geometry["B_d_all"], (1, n)), _fused_array(geometry["detw_all"], ()),
                  _fused_array(geometry["B_s_all"], (2, n)), _fused_array(geometry["detw_shear_all"], ()),
                  _fused_matrix(t0, (n, n)), _fused_matrix(membrane, (3, 3)),
                  _fused_matrix(bending, (3, 3)), _fused_matrix(shear, (2, 2)))
        drill_value = _fused_scalar(drill)
        if (drill_value is None or any(a is None for a in arrays)
                or not len(arrays[0]) == len(arrays[1]) == len(arrays[2]) == len(arrays[3])
                or len(arrays[4]) != len(arrays[5]) or len(arrays[0]) + len(arrays[4]) == 0):
            return None
        return (arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5],
                arrays[6], arrays[7], arrays[8], arrays[9], drill_value)
    except Exception:  # noqa: BLE001 - anything unexpected is the reference's to report
        return None


def _zero_tangent_fused(geometry, membrane, bending, shear, drill):
    """Run the compiled kernel; ``None`` when not applicable or refused."""
    args = _zero_tangent_args(geometry, membrane, bending, shear, drill)
    if args is None:
        return None
    out = np.empty((args[6].shape[0],) * 2, dtype=np.float64)
    if _native.fused_shell_zero_tangent(*args, out, _native.POW2) != 0:
        return None
    return out


def shell_zero_tangent(geometry, membrane, bending, shear, drill):
    """Contract independently supplied elastic coefficients on frozen geometry.

    One compiled call per element when the inputs have the expected float64
    layout; anything else, or any refusal, runs the reference implementation.
    """
    if _fused_policy_allows():
        fused = _zero_tangent_fused(geometry, membrane, bending, shear, drill)
        if fused is not None:
            return fused
    return _shell_zero_tangent_reference(geometry, membrane, bending, shear, drill)


def _increment_fused(transform, u, geometry, constitutive, tangent):
    """Run the compiled increment kernel on the already computed ``u``; ``None`` if not applicable."""
    try:
        if type(geometry) not in _GEOMETRY_TYPES or type(tangent) is not bool:
            return None
        n = transform.shape[0] if type(transform) is np.ndarray and transform.ndim == 2 else -1
        if n <= 0:
            return None
        arrays = (_fused_matrix(transform, (n, n)), _fused_matrix(u, (n,)),
                  _fused_matrix(constitutive, (3, 3)), _fused_array(geometry["B_m_all"], (3, n)),
                  _fused_array(geometry["Gw_all"], (2, n)), _fused_array(geometry["detw_all"], ()))
        if (any(a is None for a in arrays)
                or not len(arrays[3]) == len(arrays[4]) == len(arrays[5]) > 0):
            return None
    except Exception:  # noqa: BLE001 - anything unexpected is the reference's to report
        return None
    force = np.empty(n, dtype=np.float64)
    material = np.zeros((n, n), dtype=np.float64)
    geometric = np.zeros((n, n), dtype=np.float64)
    if _native.fused_elastic_increment(
            arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5], tangent,
            force, material, geometric, _native.POW2, 2.0) != 0:
        return None
    return force, material, geometric


def _native_known_answers_match():
    """Disable the compiled kernels unless they reproduce the numpy path bitwise."""
    global _USE_NATIVE
    if not _USE_NATIVE:
        return
    rng = np.random.default_rng(20261022)
    a = rng.standard_normal((7, 9)) * 2.0 ** rng.integers(-40, 40, (7, 9))
    b = rng.standard_normal((9, 6)) * 2.0 ** rng.integers(-40, 40, (9, 6))
    stack = rng.standard_normal((5, 11)) * 2.0 ** rng.integers(-40, 40, (5, 11)) * 1e8
    stack[1] = -stack[0] * (1.0 + 2.0 ** -40)
    weights = rng.standard_normal(5)
    results = []
    for use in (True, False):
        _USE_NATIVE = use
        try:
            results.append((product(a, b).tobytes(), sum_arrays(list(stack)).tobytes(),
                            weighted_sum(list(stack), weights).tobytes()))
        except Exception:  # noqa: BLE001 - any failure disables the kernels
            results.append(None)
    _USE_NATIVE = results[0] is not None and results[0] == results[1]


_native_known_answers_match()


def _fused_cases():
    """Deterministic known-answer inputs (also exercised by the tests)."""
    rng = np.random.default_rng(20261023)
    for case in range(3):
        n, ng, ns = 24, 4, 4
        scale = (1.0, 2.0 ** 20, 2.0 ** -30)[case]

        def rnd(*shape):
            return rng.standard_normal(shape) * 2.0 ** rng.integers(-3, 4, shape) * scale

        geometry = {"B_m_all": rnd(ng, 3, n), "B_b_all": rnd(ng, 3, n), "B_d_all": rnd(ng, 1, n),
                    "Gw_all": rnd(ng, 2, n), "B_s_all": rnd(ns, 2, n), "T0": rnd(n, n),
                    "detw_all": np.abs(rnd(ng)) + 0.1, "detw_shear_all": np.abs(rnd(ns)) + 0.1}
        yield (geometry, rnd(3, 3), rnd(3, 3), rnd(2, 2), float(rnd(1)[0]),
               rnd(n) * 1e-2, rnd(3, 3) * 1e3)
    # Squares where libm pow differs from x * x: for identity-like Gw rows theta
    # equals u exactly, so the kernel's scalar square is exercised on them.
    hits = [float(v) for v in np.random.default_rng(7).uniform(1.0, 2.0, 60000)
            if np.float64(v) ** 2 != v * v][:4]
    n = 6
    geometry = {"B_m_all": np.ones((1, 3, n)), "B_b_all": np.ones((1, 3, n)), "B_d_all": np.ones((1, 1, n)),
                "Gw_all": np.zeros((1, 2, n)), "B_s_all": np.ones((1, 2, n)), "T0": np.eye(n),
                "detw_all": np.ones(1), "detw_shear_all": np.ones(1)}
    geometry["Gw_all"][0, 0, 0] = geometry["Gw_all"][0, 1, 1] = 1.0
    for x in hits or [1.5]:
        u = np.zeros(n)
        u[0], u[1] = x, x
        yield geometry, np.eye(3), np.eye(3), np.eye(2), 0.5, u, np.eye(3) * 1e3


def _fused_check(zero_kernel, increment_kernel):
    """True iff the given compiled kernels reproduce the reference bitwise *and run*.

    Calls the kernels directly (never the fallback-capable wrappers), so a
    kernel that always refuses fails the check.
    """
    try:
        for geometry, membrane, bending, shear, drill, u_elem, constitutive in _fused_cases():
            args = _zero_tangent_args(geometry, membrane, bending, shear, drill)
            if args is None:
                return False
            out = np.empty((args[6].shape[0],) * 2)
            if zero_kernel(*args, out, _native.POW2) != 0:
                return False
            if out.tobytes() != _shell_zero_tangent_reference(
                    geometry, membrane, bending, shear, drill).tobytes():
                return False
            u = product(geometry["T0"], u_elem)
            for tangent in (True, False):
                n = geometry["T0"].shape[0]
                f, m, g = np.empty(n), np.zeros((n, n)), np.zeros((n, n))
                if increment_kernel(geometry["T0"], u, constitutive, geometry["B_m_all"],
                                    geometry["Gw_all"], geometry["detw_all"], tangent,
                                    f, m, g, _native.POW2, 2.0) != 0:
                    return False
                ref = elastic_increment_reference(u_elem, geometry, lambda: constitutive, tangent)
                if any(x.tobytes() != y.tobytes() for x, y in zip((f, m, g), ref)):
                    return False
        return True
    except Exception:  # noqa: BLE001 - any failure disables the fused kernels
        return False


def _fused_known_answers_match():
    """Disable the fused element kernels unless they reproduce the reference bitwise."""
    global _USE_FUSED
    if not _USE_FUSED:
        return
    with np.errstate(all="ignore"):
        _USE_FUSED = _fused_check(_native.fused_shell_zero_tangent, _native.fused_elastic_increment)


_fused_known_answers_match()
