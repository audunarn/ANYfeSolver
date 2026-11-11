"""Optional compiled kernels for the local compensated contractions.

Bitwise the same arithmetic as ``_compensated_contractions``: Dekker product
splitting and CPython ``math.fsum`` (Shewchuk partials with the final half-way
correction), with no fast-math and no fused multiply-add contraction.  A kernel
returns status 1 for nonfinite input, product overflow or fsum overflow; the
caller then replays the scalar-policy path, which raises the reference
exception.  Without Numba the kernels are never used.
"""
import math

import numpy as np

from .jit_compiler import JIT_ENABLED, njit

AVAILABLE = bool(JIT_ENABLED)

POW2 = np.array([math.ldexp(1.0, e) if e <= 1023 else float("inf") for e in range(-1100, 1100)])
POW2.flags.writeable = False


def _kernel(function):
    """njit with the disk cache; without a writable cache location compile uncached."""
    try:
        return njit(function)
    except Exception:  # noqa: BLE001 - numba raises RuntimeError when no cache locator exists
        return njit(cache=False)(function)


@_kernel
def _fsum_buf(buf, n, partials):
    """math.fsum(buf[:n]); status 1 on intermediate overflow."""
    m = 0
    for t in range(n):
        x = buf[t]
        j = 0
        for i in range(m):
            y = partials[i]
            if abs(x) < abs(y):
                tmp = x
                x = y
                y = tmp
            hi = x + y
            if not math.isfinite(hi):
                return 0.0, 1
            yr = hi - x
            lo = y - yr
            if lo != 0.0:
                partials[j] = lo
                j += 1
            x = hi
        m = j
        if x != 0.0:
            partials[m] = x
            m += 1
    hi = 0.0
    if m > 0:
        m -= 1
        hi = partials[m]
        lo = 0.0
        while m > 0:
            x = hi
            m -= 1
            y = partials[m]
            hi = x + y
            yr = hi - x
            lo = y - yr
            if lo != 0.0:
                break
        if m > 0 and ((lo < 0.0 and partials[m - 1] < 0.0) or (lo > 0.0 and partials[m - 1] > 0.0)):
            y = lo * 2.0
            x = hi + y
            yr = x - hi
            if y == yr:
                hi = x
    return hi, 0


@_kernel
def _prep(a, xm, xh, xl, ex):
    """Mantissa, Dekker halves and exponent of every operand; False on nonfinite."""
    split = 134217729.0
    for idx in range(a.size):
        v = a.flat[idx]
        if not math.isfinite(v):
            return False
        if v == 0.0:
            xm.flat[idx] = 0.0
            xh.flat[idx] = 0.0
            xl.flat[idx] = 0.0
            ex.flat[idx] = 0
        else:
            x, e = math.frexp(v)
            c = split * x
            h = c - (c - x)
            xm.flat[idx] = x
            xh.flat[idx] = h
            xl.flat[idx] = x - h
            ex.flat[idx] = e
    return True


@_kernel
def _residual(x, xh, xl, ex, y, yh, yl, ey, pow2):
    high = x * y
    low = ((xh * yh - high) + xh * yl + xl * yh) + xl * yl
    e = ex + ey
    if low == 0.0:
        return low
    if -1100 <= e < 1100:
        scaled = low * pow2[e + 1100]
        if abs(scaled) >= 2.2250738585072014e-308:
            return scaled
    return math.ldexp(low, e)


@_kernel
def contract_product(a, b, out, pow2):
    m, k = a.shape
    n = b.shape[1]
    ax = np.empty((m, k))
    ah = np.empty((m, k))
    al = np.empty((m, k))
    ae = np.empty((m, k), dtype=np.int64)
    bx = np.empty((k, n))
    bh = np.empty((k, n))
    bl = np.empty((k, n))
    be = np.empty((k, n), dtype=np.int64)
    if not (_prep(a, ax, ah, al, ae) and _prep(b, bx, bh, bl, be)):
        return 1
    buf = np.empty(2 * k)
    partials = np.empty(2 * k + 1)
    for i in range(m):
        for j in range(n):
            for l in range(k):
                av = a[i, l]
                bv = b[l, j]
                product = av * bv
                if not math.isfinite(product):
                    return 1
                buf[2 * l] = product
                if av == 0.0 or bv == 0.0:
                    buf[2 * l + 1] = 0.0
                else:
                    buf[2 * l + 1] = _residual(ax[i, l], ah[i, l], al[i, l], ae[i, l],
                                               bx[l, j], bh[l, j], bl[l, j], be[l, j], pow2)
            v, st = _fsum_buf(buf, 2 * k, partials)
            if st != 0:
                return 1
            out[i, j] = v
    return 0


@_kernel
def contract_weighted(stack, w, out, pow2):
    """out[c] = compensated dot(stack[:, c], w): terms (left) times weights (right)."""
    t, n = stack.shape
    sx = np.empty((t, n))
    sh = np.empty((t, n))
    sl = np.empty((t, n))
    se = np.empty((t, n), dtype=np.int64)
    wx = np.empty(t)
    wh = np.empty(t)
    wl = np.empty(t)
    we = np.empty(t, dtype=np.int64)
    if not (_prep(stack, sx, sh, sl, se) and _prep(w, wx, wh, wl, we)):
        return 1
    buf = np.empty(2 * t)
    partials = np.empty(2 * t + 1)
    for c in range(n):
        for l in range(t):
            av = stack[l, c]
            bv = w[l]
            product = av * bv
            if not math.isfinite(product):
                return 1
            buf[2 * l] = product
            if av == 0.0 or bv == 0.0:
                buf[2 * l + 1] = 0.0
            else:
                buf[2 * l + 1] = _residual(sx[l, c], sh[l, c], sl[l, c], se[l, c],
                                           wx[l], wh[l], wl[l], we[l], pow2)
        v, st = _fsum_buf(buf, 2 * t, partials)
        if st != 0:
            return 1
        out[c] = v
    return 0


@_kernel
def contract_sum(stack, out):
    t, n = stack.shape
    buf = np.empty(t)
    partials = np.empty(t + 1)
    for c in range(n):
        for l in range(t):
            buf[l] = stack[l, c]
        v, st = _fsum_buf(buf, t, partials)
        if st != 0:
            return 1
        out[c] = v
    return 0


# ---------------------------------------------------------------------------
# Fused element kernels.  Each runs the *same sequence* of compensated products
# and sums as the numpy-level routines in ``_compensated_contractions`` /
# ``vectorized_nonlinear`` (every step calls ``contract_product`` /
# ``contract_sum`` / ``contract_weighted``), plus the single IEEE multiplies and
# adds that numpy performs as separate array operations.  Any refusal returns
# status 1 and the caller replays the reference implementation, which raises the
# reference exception.  The scalar squares use ``pow`` with a run-time exponent
# because numpy's float64 scalar ``**`` is libm ``pow`` and is not always equal
# to ``x * x``; LLVM would rewrite ``pow(x, 2.0)`` to ``x * x``.
# ---------------------------------------------------------------------------
@_kernel
def _mm(a, b, pow2):
    out = np.empty((a.shape[0], b.shape[1]))
    st = contract_product(a, b, out, pow2)
    return out, st


@_kernel
def _tc(a):
    return np.ascontiguousarray(a.T)


@_kernel
def fused_shell_zero_tangent(bm_all, bb_all, bd_all, detw, bs_all, detw_shear, t0,
                             membrane, bending, shear, drill, out, pow2):
    """Fused ``shell_zero_tangent``: all gauss-point terms, weighted sum and transform."""
    ng = bm_all.shape[0]
    ns = bs_all.shape[0]
    n = bm_all.shape[2]
    nn = n * n
    nt = ng + ns
    terms = np.empty((nt, nn))
    weights = np.empty(nt)
    stack = np.empty((3, nn))
    for g in range(ng):
        bm = bm_all[g]
        bb = bb_all[g]
        bd = bd_all[g]
        x, s = _mm(_tc(bm), membrane, pow2)
        if s != 0:
            return 1
        p1, s = _mm(x, bm, pow2)
        if s != 0:
            return 1
        x, s = _mm(_tc(bb), bending, pow2)
        if s != 0:
            return 1
        p2, s = _mm(x, bb, pow2)
        if s != 0:
            return 1
        scaled = drill * bd
        p3, s = _mm(_tc(bd), scaled, pow2)
        if s != 0:
            return 1
        for i in range(n):
            for j in range(n):
                stack[0, i * n + j] = p1[i, j]
                stack[1, i * n + j] = p2[i, j]
                stack[2, i * n + j] = p3[i, j]
        if contract_sum(stack, terms[g]) != 0:
            return 1
        weights[g] = detw[g]
    for g in range(ns):
        bs = bs_all[g]
        x, s = _mm(_tc(bs), shear, pow2)
        if s != 0:
            return 1
        p1, s = _mm(x, bs, pow2)
        if s != 0:
            return 1
        for i in range(n):
            for j in range(n):
                terms[ng + g, i * n + j] = p1[i, j]
        weights[ng + g] = detw_shear[g]
    local_flat = np.empty(nn)
    if contract_weighted(terms, weights, local_flat, pow2) != 0:
        return 1
    local = local_flat.reshape((n, n))
    x, s = _mm(_tc(t0), local, pow2)
    if s != 0:
        return 1
    res, s = _mm(x, t0, pow2)
    if s != 0:
        return 1
    for i in range(n):
        for j in range(n):
            out[i, j] = res[i, j]
    return 0


@_kernel
def fused_elastic_increment(t0, u_vec, constitutive, bm_all, gw_all, detw, tangent,
                            f_out, m_out, g_out, pow2, two):
    """Fused increment loop on the transformed displacement ``u_vec = T0 @ u_elem`` (>= 1 gauss point)."""
    n = t0.shape[0]
    ng = bm_all.shape[0]
    nn = n * n
    u = np.empty((n, 1))
    for i in range(n):
        u[i, 0] = u_vec[i]
    forces = np.empty((ng, n))
    materials = np.empty((ng, nn))
    geometrics = np.empty((ng, nn))
    weights = np.empty(ng)
    fstack = np.empty((3, n))
    mstack = np.empty((3, nn))
    gstack = np.empty((4, nn))
    sstack = np.empty((2, 3))
    h = np.empty((3, n))
    eta = np.empty((3, 1))
    for g in range(ng):
        b = bm_all[g]
        gg = gw_all[g]
        theta, s = _mm(gg, u, pow2)
        if s != 0:
            return 1
        th0 = theta[0, 0]
        th1 = theta[1, 0]
        eta[0, 0] = 0.5 * (th0 ** two)
        eta[1, 0] = 0.5 * (th1 ** two)
        eta[2, 0] = th0 * th1
        for i in range(n):
            h[0, i] = th0 * gg[0, i]
            h[1, i] = th1 * gg[1, i]
            h[2, i] = th0 * gg[1, i] + th1 * gg[0, i]
        bu, s = _mm(b, u, pow2)
        if s != 0:
            return 1
        linear_stress, s = _mm(constitutive, bu, pow2)
        if s != 0:
            return 1
        nonlinear_stress, s = _mm(constitutive, eta, pow2)
        if s != 0:
            return 1
        bt = _tc(b)
        ht = _tc(h)
        f1, s = _mm(bt, nonlinear_stress, pow2)
        if s != 0:
            return 1
        f2, s = _mm(ht, linear_stress, pow2)
        if s != 0:
            return 1
        f3, s = _mm(ht, nonlinear_stress, pow2)
        if s != 0:
            return 1
        for i in range(n):
            fstack[0, i] = f1[i, 0]
            fstack[1, i] = f2[i, 0]
            fstack[2, i] = f3[i, 0]
        if contract_sum(fstack, forces[g]) != 0:
            return 1
        weights[g] = detw[g]
        if tangent:
            x, s = _mm(bt, constitutive, pow2)
            if s != 0:
                return 1
            m1, s = _mm(x, h, pow2)
            if s != 0:
                return 1
            x, s = _mm(ht, constitutive, pow2)
            if s != 0:
                return 1
            m2, s = _mm(x, b, pow2)
            if s != 0:
                return 1
            m3, s = _mm(x, h, pow2)
            if s != 0:
                return 1
            for i in range(n):
                for j in range(n):
                    mstack[0, i * n + j] = m1[i, j]
                    mstack[1, i * n + j] = m2[i, j]
                    mstack[2, i * n + j] = m3[i, j]
            if contract_sum(mstack, materials[g]) != 0:
                return 1
            for c in range(3):
                sstack[0, c] = linear_stress[c, 0]
                sstack[1, c] = nonlinear_stress[c, 0]
            stress = np.empty(3)
            if contract_sum(sstack, stress) != 0:
                return 1
            for i in range(n):
                for j in range(n):
                    k = i * n + j
                    gstack[0, k] = stress[0] * (gg[0, i] * gg[0, j])
                    gstack[1, k] = stress[1] * (gg[1, i] * gg[1, j])
                    gstack[2, k] = stress[2] * (gg[0, i] * gg[1, j])
                    gstack[3, k] = stress[2] * (gg[1, i] * gg[0, j])
            for k in range(nn):
                for r in range(4):
                    if not math.isfinite(gstack[r, k]):
                        return 1
            if contract_sum(gstack, geometrics[g]) != 0:
                return 1
    force = np.empty(n)
    if contract_weighted(forces, weights, force, pow2) != 0:
        return 1
    fcol = np.empty((n, 1))
    for i in range(n):
        fcol[i, 0] = force[i]
    tt = _tc(t0)
    fres, s = _mm(tt, fcol, pow2)
    if s != 0:
        return 1
    for i in range(n):
        f_out[i] = fres[i, 0]
    if tangent:
        mat_flat = np.empty(nn)
        geo_flat = np.empty(nn)
        if contract_weighted(materials, weights, mat_flat, pow2) != 0:
            return 1
        if contract_weighted(geometrics, weights, geo_flat, pow2) != 0:
            return 1
        mat = mat_flat.reshape((n, n))
        geo = geo_flat.reshape((n, n))
        x, s = _mm(tt, mat, pow2)
        if s != 0:
            return 1
        mres, s = _mm(x, t0, pow2)
        if s != 0:
            return 1
        x, s = _mm(tt, geo, pow2)
        if s != 0:
            return 1
        gres, s = _mm(x, t0, pow2)
        if s != 0:
            return 1
        for i in range(n):
            for j in range(n):
                m_out[i, j] = mres[i, j]
                g_out[i, j] = gres[i, j]
    return 0
