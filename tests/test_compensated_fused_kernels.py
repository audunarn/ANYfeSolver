"""Differential tests: fused shell element kernels against the reference arithmetic.

The fused compiled kernels (``_compensated_native.fused_*``) must reproduce the
numpy-level reference bit for bit (values, signed zeros) and must either succeed
with the reference value or leave the reference to raise its own exception.
``ANYSOLVER_FUSED_CASES`` sets the randomized case count (default is small; the
acceptance run uses >= 100000).
"""
import os
import warnings

import numpy as np
import pytest

from anyfesolver import _compensated_contractions as cc
from anyfesolver import vectorized_nonlinear as vn

# Skipped only when the compiled kernels are really unavailable (or explicitly opted out);
# a fused path that was disabled at import by a failed self-check must make these tests fail.
pytestmark = pytest.mark.skipif(
    not cc._native.AVAILABLE or os.environ.get("ANYSOLVER_NATIVE_CONTRACTIONS", "1") == "0",
    reason="compiled contraction kernels unavailable or opted out")

CASES = int(os.environ.get("ANYSOLVER_FUSED_CASES", "1500"))
TINY = 5e-324


def _rnd(rng, shape, mode):
    base = rng.standard_normal(shape)
    if mode == 0:
        return base
    if mode == 1:
        return base * 2.0 ** rng.integers(-40, 40, shape)
    if mode == 2:  # extreme magnitudes: overflow and subnormal products
        return base * 2.0 ** rng.integers(-520, 520, shape)
    if mode == 3:  # sparse with signed zeros and subnormals
        out = base * 2.0 ** rng.integers(-60, 60, shape)
        pick = rng.integers(0, 6, shape)
        out = np.where(pick == 0, 0.0, out)
        out = np.where(pick == 1, -0.0, out)
        out = np.where(pick == 2, TINY * rng.integers(-5, 6, shape), out)
        return out
    # mode 4: values on a coarse binary grid so that sums cancel exactly or by one ulp
    return rng.integers(-8, 9, shape) * 2.0 ** rng.integers(-3, 4, shape) * (1.0 + (rng.integers(0, 3, shape) == 0) * 2.0 ** -52)


def _geometry(rng, n, ng, ns, mode):
    return {"B_m_all": _rnd(rng, (ng, 3, n), mode), "B_b_all": _rnd(rng, (ng, 3, n), mode),
            "B_d_all": _rnd(rng, (ng, 1, n), mode), "Gw_all": _rnd(rng, (ng, 2, n), mode),
            "B_s_all": _rnd(rng, (ns, 2, n), mode), "T0": _rnd(rng, (n, n), mode),
            "detw_all": np.abs(_rnd(rng, (ng,), min(mode, 1))) + 0.05,
            "detw_shear_all": np.abs(_rnd(rng, (ns,), min(mode, 1))) + 0.05}


def _poison(rng, arrays):
    """Inject a nonfinite value into one randomly chosen array (in place)."""
    target = arrays[rng.integers(0, len(arrays))]
    flat = target.reshape(-1)
    if flat.size:
        flat[rng.integers(0, flat.size)] = (np.nan, np.inf, -np.inf)[rng.integers(0, 3)]


def _run(function, *args):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            result = function(*args)
        except Exception as exc:  # noqa: BLE001 - exception identity is compared
            return ("error", type(exc), str(exc))
    if isinstance(result, tuple):
        return ("ok",) + tuple(np.asarray(r).tobytes() for r in result)
    return ("ok", np.asarray(result).tobytes())


def _count_kernel(monkeypatch, name):
    """Count kernel calls by status; a reference fallback can never move the success count."""
    real = getattr(cc._native, name)
    counts = {0: 0, 1: 0}

    def counted(*args):
        status = real(*args)
        counts[status] += 1
        return status

    monkeypatch.setattr(cc._native, name, counted)
    return counts


def test_randomized_zero_tangent_matches_reference(monkeypatch):
    counts = _count_kernel(monkeypatch, "fused_shell_zero_tangent")
    rng = np.random.default_rng(20261023)
    fused_ok = 0
    for case in range(CASES):
        mode = int(rng.integers(0, 5))
        n = int(rng.choice((2, 3, 6, 8, 12)))
        ng, ns = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        geometry = _geometry(rng, n, ng, ns, mode)
        membrane, bending, shear = (_rnd(rng, (3, 3), mode), _rnd(rng, (3, 3), mode),
                                    _rnd(rng, (2, 2), mode))
        drill = float(_rnd(rng, (1,), mode)[0])
        if case % 9 == 4:
            _poison(rng, [*geometry.values(), membrane, bending, shear])
        want = _run(cc._shell_zero_tangent_reference, geometry, membrane, bending, shear, drill)
        before = counts[0]
        got = _run(cc.shell_zero_tangent, geometry, membrane, bending, shear, drill)
        assert got == want, (case, mode, n, ng, ns)
        served = counts[0] - before
        assert served == (1 if want[0] == "ok" else 0) or (mode == 2 and served == 0), (case, mode)
        fused_ok += served
    if CASES >= 1000:
        assert fused_ok > CASES // 5


def test_randomized_elastic_increment_matches_reference(monkeypatch):
    counts = _count_kernel(monkeypatch, "fused_elastic_increment")
    rng = np.random.default_rng(20261024)
    fused_ok = 0
    for case in range(CASES):
        mode = int(rng.integers(0, 5))
        n = int(rng.choice((2, 3, 6, 8, 12)))
        ng = int(rng.integers(0, 4))
        geometry = _geometry(rng, n, ng, 0, mode)
        u_elem = _rnd(rng, (n,), mode) * (1e-2 if mode < 2 else 1.0)
        constitutive = _rnd(rng, (3, 3), mode)
        tangent = bool(rng.integers(0, 2))
        if case % 9 == 4:
            _poison(rng, [geometry["T0"], geometry["B_m_all"], geometry["Gw_all"], u_elem, constitutive])
        want = _run(cc.elastic_increment_reference, u_elem, geometry, lambda: constitutive, tangent)

        before = counts[0]
        got = _run(cc.elastic_increment, u_elem, geometry, lambda: constitutive, tangent)
        assert got == want, (case, mode, n, ng, tangent)
        served = counts[0] - before
        # Every case the reference can evaluate (with a gauss point) must be served by the
        # kernel itself; only extreme-magnitude cases may be refused where the reference's
        # own native step also refused and its numpy replay succeeded.
        expected = 1 if want[0] == "ok" and ng > 0 else 0
        assert served == expected or (mode == 2 and served == 0), (case, mode)
        fused_ok += served
    if CASES >= 1000:
        assert fused_ok > CASES // 5


def test_public_increment_exception_order_and_full_size_element():
    rng = np.random.default_rng(7)
    geometry = _geometry(rng, 24, 4, 4, 1)
    u = rng.standard_normal(24) * 1e-2
    for E, nu, h in ((2.1e11, 0.3, 0.01), (2.1e11, 1.0, 0.01), (2.1e11, float("nan"), 0.01),
                     (float("inf"), 0.3, 0.01), (2.1e11, 0.3, 0.0)):
        for tangent in (True, False):
            def reference():
                return cc.elastic_increment_reference(
                    u, geometry, lambda: h * vn.plane_stress_elastic_matrix(E, nu), tangent)
            assert (_run(lambda *a: vn.elastic_shell_nonlinear_increment(*a, tangent=tangent), u, geometry, E, nu, h)
                    == _run(reference))
    bad = u.copy()
    bad[3] = np.nan
    assert (_run(vn.elastic_shell_nonlinear_increment, bad, geometry, 2.1e11, 1.0, 0.01)
            == _run(cc.elastic_increment_reference, bad, geometry,
                    lambda: 0.01 * vn.plane_stress_elastic_matrix(2.1e11, 1.0), True))


@pytest.mark.parametrize("dtype,readonly,order", [(np.float32, False, "C"), (np.float64, True, "C"),
                                                  (np.float64, False, "F")])
def test_layout_variants_match_reference(dtype, readonly, order):
    rng = np.random.default_rng(11)
    geometry = _geometry(rng, 6, 2, 2, 1)
    membrane, bending, shear = rng.standard_normal((3, 3)), rng.standard_normal((3, 3)), rng.standard_normal((2, 2))
    geometry = {k: np.asarray(np.asfortranarray(v) if order == "F" else v, dtype=dtype)
                for k, v in geometry.items()}
    if readonly:
        for v in geometry.values():
            v.flags.writeable = False
    assert (_run(cc.shell_zero_tangent, geometry, membrane, bending, shear, 0.3)
            == _run(cc._shell_zero_tangent_reference, geometry, membrane, bending, shear, 0.3))


def test_signed_zero_and_cancellation_results_are_bitwise_equal():
    n = 4
    geometry = {k: np.zeros(s) for k, s in (("B_m_all", (1, 3, n)), ("B_b_all", (1, 3, n)), ("B_d_all", (1, 1, n)),
                                           ("Gw_all", (1, 2, n)), ("B_s_all", (1, 2, n)), ("T0", (n, n)))}
    geometry["T0"][:] = -0.0
    geometry["detw_all"] = np.array([-0.0])
    geometry["detw_shear_all"] = np.array([0.0])
    args = (geometry, np.eye(3), -np.eye(3), np.eye(2), -0.0)
    assert _run(cc.shell_zero_tangent, *args) == _run(cc._shell_zero_tangent_reference, *args)
    # a huge pair cancelling exactly plus a tiny remainder: compensation must survive
    geometry = _geometry(np.random.default_rng(3), n, 2, 2, 4)
    geometry["B_m_all"][1] = -geometry["B_m_all"][0]
    geometry["detw_all"][:] = 1.0
    args = (geometry, np.eye(3) * 1e150, np.eye(3) * 1e-150, np.eye(2), 1.0)
    assert _run(cc.shell_zero_tangent, *args) == _run(cc._shell_zero_tangent_reference, *args)


def test_scalar_square_uses_libm_pow_not_multiply():
    """numpy's float64 scalar ``**2`` is libm pow; the kernel must follow it, not ``x * x``."""
    rng = np.random.default_rng(2)
    n = 3
    base = _geometry(rng, n, 1, 0, 0)
    u = np.zeros(n)
    mismatch = 0
    for _ in range(4000):
        geometry = dict(base)
        geometry["Gw_all"] = rng.uniform(1, 2, (1, 2, n))
        geometry["T0"] = np.eye(n)
        u = rng.uniform(1, 2, n)
        constitutive = np.eye(3)
        want = cc.elastic_increment_reference(u, geometry, lambda: constitutive, True)
        got = cc.elastic_increment(u, geometry, lambda: constitutive, True)
        mismatch += any(a.tobytes() != b.tobytes() for a, b in zip(got, want))
    assert mismatch == 0


def test_fused_backend_flag_is_authority_guarded():
    from anyfesolver import prepared_static_q4 as q
    q.require_q4_kernel_authority()
    names = {row[0] for row in q._NATIVE_KERNEL_FACTS}
    assert {"fused_shell_zero_tangent", "fused_elastic_increment", "_mm", "_tc"} <= names
    assert {row[0] for row in q._CONTRACTION_FACTS} >= {
        "shell_zero_tangent", "elastic_increment", "_increment", "elastic_increment_reference",
        "_shell_zero_tangent_reference", "_fused_array", "_fused_scalar"}
    cc._USE_FUSED = False
    try:
        with pytest.raises(Exception, match="authority"):
            q.require_q4_kernel_authority()
    finally:
        cc._USE_FUSED = True
    q.require_q4_kernel_authority()


# --- review round 1 regressions -------------------------------------------------
def _outcome(function, *args, **kwargs):
    """Result bytes or exception identity, without touching the numpy error policy."""
    try:
        result = function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - exception identity is compared
        return ("error", type(exc), str(exc))
    if isinstance(result, tuple):
        return ("ok",) + tuple(np.asarray(r).tobytes() for r in result)
    return ("ok", np.asarray(result).tobytes())


@pytest.mark.parametrize("kernel", ["fused_elastic_increment", "fused_shell_zero_tangent"])
def test_substituted_native_kernel_is_refused_by_the_e4_lifecycle(monkeypatch, kernel):
    from test_general_contact_c5_elastic_increment import case, pose
    model, element, material = case()
    run = lambda: element._elastic_increment_response(  # noqa: E731
        model.mesh, material, pose(), None, True, batch_baseline=True)
    genuine = run()  # the genuine kernels are admitted
    called = []

    def fabricate(*args):
        called.append(True)
        for array in [a for a in args if isinstance(a, np.ndarray) and a.size and a.flags.writeable][-3:]:
            array[...] = 1.0
        return 0

    monkeypatch.setattr(cc._native, kernel, fabricate)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        run()
    assert not called, "the fabricating kernel must be refused before it can produce output"
    monkeypatch.undo()
    assert all(np.array_equal(x, y) for x, y in zip(run(), genuine))


@pytest.mark.parametrize("name", ["_USE_FUSED", "shell_zero_tangent", "elastic_increment",
                                  "elastic_increment_reference", "_shell_zero_tangent_reference"])
def test_substituted_wrapper_or_flag_is_refused_by_the_e4_lifecycle(monkeypatch, name):
    from test_general_contact_c5_elastic_increment import case, pose
    model, element, material = case()
    monkeypatch.setattr(cc, name, False if name == "_USE_FUSED" else (lambda *a, **k: None))
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        element._elastic_increment_response(model.mesh, material, pose(), None, True,
                                            batch_baseline=True)


def _conforming(rng=None):
    rng = rng or np.random.default_rng(5)
    return _geometry(rng, 6, 2, 2, 1)


@pytest.mark.parametrize("field,value", [
    ("detw_all", np.array(1.0)), ("detw_all", np.ones((2, 1))), ("detw_all", [1.0, 1.0]),
    ("detw_all", np.ones(2, dtype=object)), ("detw_all", np.ones(3)),
    ("B_m_all", np.ones((2, 3, 6), dtype=np.float32)), ("B_m_all", np.ones((2, 3, 5))),
    ("Gw_all", np.ones((2, 6))), ("T0", np.eye(6)[None]), ("T0", np.ones((6, 5))),
    ("detw_shear_all", np.array(2.0)), ("B_d_all", np.ones((2, 6)))])
def test_malformed_geometry_goes_to_the_reference_in_reference_order(field, value):
    geometry = _conforming()
    geometry[field] = value
    nan_u = np.zeros(6)
    nan_u[2] = np.nan
    for u in (nan_u, np.ones(6) * 1e-3):
        assert (_outcome(vn.elastic_shell_nonlinear_increment, u, geometry, 2.1e11, 0.3, 0.01)
                == _outcome(cc.elastic_increment_reference, u, geometry,
                            lambda: 0.01 * vn.plane_stress_elastic_matrix(2.1e11, 0.3), True))
    args = (geometry, np.eye(3), np.eye(3), np.eye(2), 0.3)
    assert _outcome(cc.shell_zero_tangent, *args) == _outcome(cc._shell_zero_tangent_reference, *args)


@pytest.mark.parametrize("under", ["raise", "warn", "print"])
def test_numpy_underflow_policy_is_respected(under):
    geometry = _conforming()
    geometry["B_d_all"] = geometry["B_d_all"] * 0 + 1e-200
    args = (geometry, np.eye(3), np.eye(3), np.eye(2), 1e-200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(under=under):
            want = _outcome(cc._shell_zero_tangent_reference, *args)
            got = _outcome(cc.shell_zero_tangent, *args)
    assert got == want
    if under == "raise":
        assert want[0] == "error" and want[1] is FloatingPointError


def _mutated(kernel, replacements, fastmath=False):
    """Compile a source-mutated copy of a fused kernel in the native module's namespace."""
    import inspect
    import textwrap
    source = textwrap.dedent(inspect.getsource(kernel.py_func))
    source = chr(10).join(line for line in source.splitlines() if not line.startswith("@"))
    for old, new in replacements:
        assert old in source
        source = source.replace(old, new)
    namespace = dict(vars(cc._native))
    exec(compile(source, "<mutated>", "exec"), namespace)  # noqa: S102 - test-only mutation
    function = namespace[kernel.py_func.__name__]
    return cc._native.njit(fastmath=True, cache=False)(function) if fastmath else cc._native.njit(cache=False)(function)


def test_self_check_accepts_the_genuine_kernels():
    assert cc._fused_check(cc._native.fused_shell_zero_tangent, cc._native.fused_elastic_increment)


def test_self_check_fails_for_an_always_refusing_kernel():
    refuse = lambda *args: 1  # noqa: E731
    assert not cc._fused_check(refuse, cc._native.fused_elastic_increment)
    assert not cc._fused_check(cc._native.fused_shell_zero_tangent, refuse)


def test_self_check_fails_when_the_square_is_x_times_x():
    bad = [x for x in np.random.default_rng(7).uniform(1.0, 2.0, 60000) if np.float64(x) ** 2 != x * x]
    if not bad:
        pytest.skip("this libm squares exactly: pow(x, 2) == x * x for all sampled x")
    mutated = _mutated(cc._native.fused_elastic_increment,
                       [("th0 ** two", "th0 * th0"), ("th1 ** two", "th1 * th1")])
    assert not cc._fused_check(cc._native.fused_shell_zero_tangent, mutated)


def test_self_check_fails_for_a_wrong_sign():
    mutated = _mutated(cc._native.fused_elastic_increment,
                       [("th0 * gg[1, i] + th1 * gg[0, i]", "th0 * gg[1, i] - th1 * gg[0, i]")])
    assert not cc._fused_check(cc._native.fused_shell_zero_tangent, mutated)


def test_self_check_flags_a_fast_math_variant_if_it_contracts():
    """fastmath lets LLVM contract multiply-adds into FMA; the check must notice any change."""
    mutated = _mutated(cc._native.fused_elastic_increment, [], fastmath=True)
    plain = _mutated(cc._native.fused_elastic_increment, [])
    assert cc._fused_check(cc._native.fused_shell_zero_tangent, plain)
    fast_ok = cc._fused_check(cc._native.fused_shell_zero_tangent, mutated)
    # Contraction is target dependent: when it happens the check must fail; when the
    # compiler does not contract, the variant is arithmetically identical and passes.
    differs = False
    for geometry, _m, _b, _s, _d, u_elem, constitutive in cc._fused_cases():
        n = geometry["T0"].shape[0]
        outs = []
        for kernel in (plain, mutated):
            f, m, g = np.empty(n), np.zeros((n, n)), np.zeros((n, n))
            kernel(geometry["T0"], cc.product(geometry["T0"], u_elem), constitutive, geometry["B_m_all"], geometry["Gw_all"],
                   geometry["detw_all"], True, f, m, g, cc._native.POW2, 2.0)
            outs.append((f.tobytes(), m.tobytes(), g.tobytes()))
        differs = differs or outs[0] != outs[1]
    assert fast_ok == (not differs)


# --- review round 2 regressions -------------------------------------------------
class _Lying(np.ndarray):
    """ndarray subclass whose indexing returns different (ordinary) data than its buffer."""

    def __getitem__(self, index):
        return np.asarray(np.ndarray.__getitem__(self, index)) * 2.0 + 1.0


class _LyingArray(np.ndarray):
    def __array__(self, dtype=None, copy=None):
        return np.zeros(self.shape)


def _as(subclass, value):
    return np.asarray(value, dtype=float).view(subclass)


@pytest.mark.parametrize("subclass", [_Lying, _LyingArray])
@pytest.mark.parametrize("argument", ["membrane", "bending", "shear", "T0", "B_m_all", "detw_all"])
def test_ndarray_subclasses_never_enter_the_fused_zero_tangent(subclass, argument):
    rng = np.random.default_rng(13)
    geometry = _geometry(rng, 6, 2, 2, 1)
    coefficients = {"membrane": rng.standard_normal((3, 3)), "bending": rng.standard_normal((3, 3)),
                    "shear": rng.standard_normal((2, 2))}
    if argument in coefficients:
        coefficients[argument] = _as(subclass, coefficients[argument])
    else:
        geometry[argument] = _as(subclass, geometry[argument])
    args = (geometry, coefficients["membrane"], coefficients["bending"], coefficients["shear"], 0.3)
    assert _outcome(cc.shell_zero_tangent, *args) == _outcome(cc._shell_zero_tangent_reference, *args)


@pytest.mark.parametrize("subclass", [_Lying, _LyingArray])
@pytest.mark.parametrize("argument", ["constitutive", "u_elem", "T0", "Gw_all"])
def test_ndarray_subclasses_never_enter_the_fused_increment(subclass, argument):
    rng = np.random.default_rng(14)
    geometry = _geometry(rng, 6, 2, 0, 1)
    constitutive, u = rng.standard_normal((3, 3)) * 1e3, rng.standard_normal(6) * 1e-2
    if argument == "constitutive":
        constitutive = _as(subclass, constitutive)
    elif argument == "u_elem":
        u = _as(subclass, u)
    else:
        geometry[argument] = _as(subclass, geometry[argument])
    want = _outcome(cc.elastic_increment_reference, u, geometry, lambda: constitutive, True)
    assert _outcome(cc.elastic_increment, u, geometry, lambda: constitutive, True) == want


def _callback_log(policy):
    log = []
    np.seterrcall(lambda kind, flag: log.append((kind, flag)))
    return log


@pytest.mark.parametrize("policy", [dict(), dict(all="ignore"), dict(all="warn"), dict(all="raise"),
                                    dict(all="call"), dict(over="call"), dict(all="print"), dict(all="log"),
                                    dict(under="warn"), dict(invalid="raise")])
@pytest.mark.parametrize("case", ["nonfinite_u", "overflow_coefficients", "ordinary", "extreme_geometry"])
def test_coefficient_evaluation_order_count_and_side_effects_match_the_reference(policy, case):
    rng = np.random.default_rng(21)
    geometry = _geometry(rng, 6, 2, 0, 1)
    u = rng.standard_normal(6) * 1e-2
    E, nu, h = 2.1e11, 0.3, 0.01
    if case == "nonfinite_u":
        u[2] = np.nan
        E, nu, h = 2.0, 0.0, 1e308
    elif case == "overflow_coefficients":
        E, nu, h = 2.0, 0.0, 1e308
    elif case == "extreme_geometry":
        geometry["B_m_all"] = geometry["B_m_all"] * 2.0 ** 480
        geometry["Gw_all"] = geometry["Gw_all"] * 2.0 ** 480
    evaluations = []

    def factory():
        evaluations.append(1)
        return h * vn.plane_stress_elastic_matrix(E, nu)

    def observe(function):
        evaluations.clear()
        log = []
        np.seterrcall(lambda kind, flag: log.append((kind, flag)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with np.errstate(**policy):
                outcome = _outcome(function)
        return outcome, log, [str(w.message) for w in caught], len(evaluations)

    old_call = np.seterrcall(None)
    try:
        want = observe(lambda: cc.elastic_increment_reference(u, geometry, factory, True))
        got = observe(lambda: cc.elastic_increment(u, geometry, factory, True))
    finally:
        np.seterrcall(old_call)
    assert got == want
    assert want[3] <= 1 and got[3] == want[3]


def test_public_increment_evaluates_the_coefficients_once(monkeypatch):
    calls = []
    real = vn.plane_stress_elastic_matrix
    monkeypatch.setattr(vn, "plane_stress_elastic_matrix", lambda *a: calls.append(a) or real(*a))
    rng = np.random.default_rng(22)
    geometry = _geometry(rng, 6, 2, 0, 2)  # extreme magnitudes: the kernel may refuse
    for _ in range(20):
        calls.clear()
        _outcome(vn.elastic_shell_nonlinear_increment, rng.standard_normal(6), geometry, 2.1e11, 0.3, 0.01)
        assert len(calls) == 1


def test_frozen_geometry_mappingproxy_is_served_by_the_kernel(monkeypatch):
    """The qualified route passes mappingproxy geometry; it must reach the compiled kernel."""
    from types import MappingProxyType
    counts = _count_kernel(monkeypatch, "fused_elastic_increment")
    zero = _count_kernel(monkeypatch, "fused_shell_zero_tangent")
    rng = np.random.default_rng(31)
    geometry = _geometry(rng, 6, 2, 2, 1)
    frozen = MappingProxyType(geometry)
    u, C = rng.standard_normal(6) * 1e-2, rng.standard_normal((3, 3)) * 1e3
    assert (_outcome(cc.elastic_increment, u, frozen, lambda: C, True)
            == _outcome(cc.elastic_increment_reference, u, geometry, lambda: C, True))
    args = (np.eye(3), np.eye(3), np.eye(2), 0.3)
    assert (_outcome(cc.shell_zero_tangent, frozen, *args)
            == _outcome(cc._shell_zero_tangent_reference, geometry, *args))
    assert counts[0] == 1 and zero[0] == 1


# --- review round 3: hooks between the displacement product and the kernel -----------
def _fresh(seed=41, n=6, gauss=2):
    rng = np.random.default_rng(seed)
    geometry = _geometry(rng, n, gauss, 0, 1)
    return geometry, rng.uniform(0.05, 0.2, n), rng.standard_normal((3, 3)) * 1e3


def _hooked_pair(make_hook, geometry_mutation=None, seed=41):
    """Run reference and fused routes on independent identical copies with the same hook."""
    results = []
    for function in (cc.elastic_increment_reference, cc.elastic_increment):
        geometry, u, constitutive = _fresh(seed)
        if geometry_mutation is not None:
            geometry = geometry_mutation(geometry)
        hook = make_hook(geometry, u)

        def factory():
            hook()
            return constitutive

        old_call = np.seterrcall(None)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with np.errstate():
                    outcome = _outcome(function, u, geometry, factory, True)
        finally:
            np.seterrcall(old_call)
        results.append((outcome, [str(w.message) for w in caught], u.tobytes()))
    return results


def test_factory_mutating_the_displacement_cannot_change_the_result():
    want, got = _hooked_pair(lambda g, u: lambda: u.__setitem__(slice(None), 0.9))
    assert got == want and want[0][0] == "ok"


def test_factory_mutating_geometry_in_place_matches_the_reference():
    for key in ("T0", "B_m_all", "Gw_all", "detw_all"):
        want, got = _hooked_pair(lambda g, u, key=key: lambda: g[key].__imul__(1.5))
        assert got == want, key


def test_factory_replacing_geometry_entries_matches_the_reference():
    from types import MappingProxyType

    for key in ("B_m_all", "Gw_all", "detw_all"):
        results = []
        for function in (cc.elastic_increment_reference, cc.elastic_increment):
            geometry, u, constitutive = _fresh()
            proxy = MappingProxyType(geometry)

            def factory():
                geometry[key] = geometry[key] * 0.5
                return constitutive

            results.append(_outcome(function, u, proxy, factory, True))
        assert results[0] == results[1], key


@pytest.mark.parametrize("policy", [dict(under="raise"), dict(under="warn"), dict(over="raise"),
                                    dict(all="raise"), dict(invalid="call"), dict(all="ignore")])
def test_factory_changing_the_error_policy_is_honoured(policy):
    def make(geometry, u):
        geometry["Gw_all"] = geometry["Gw_all"] * 0 + 1e-200
        return lambda: np.seterr(**policy)

    results = []
    for function in (cc.elastic_increment_reference, cc.elastic_increment):
        geometry, u, constitutive = _fresh()
        geometry["Gw_all"] = geometry["Gw_all"] * 0 + 1e-200
        saved = np.geterr()
        try:
            def factory():
                np.seterr(**policy)
                return constitutive
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results.append(_outcome(function, u, geometry, factory, True))
        finally:
            np.seterr(**saved)
    assert results[0] == results[1]
    if policy == dict(under="raise"):
        assert results[0][0] == "error" and results[0][1] is FloatingPointError


class _Counting:
    calls = 0

    def __init__(self, value):
        self.value = value

    def __float__(self):
        type(self).calls += 1
        return self.value


def test_object_dtype_displacement_is_converted_exactly_as_often_as_the_reference():
    counts = []
    outcomes = []
    for function in (cc.elastic_increment_reference, cc.elastic_increment):
        geometry, u, constitutive = _fresh()
        wrapped = np.empty(u.size, dtype=object)
        for i, v in enumerate(u):
            wrapped[i] = _Counting(float(v))
        _Counting.calls = 0
        outcomes.append(_outcome(function, wrapped, geometry, lambda: constitutive, True))
        counts.append(_Counting.calls)
    assert outcomes[0] == outcomes[1] and counts[0] == counts[1] and counts[0] > 0


def test_every_hook_order_is_identical_for_the_user_visible_trace():
    """Record the order of: displacement conversion, factory, warnings, error callbacks."""
    traces = []
    for function in (cc.elastic_increment_reference, cc.elastic_increment):
        geometry, u, constitutive = _fresh()
        trace = []
        wrapped = np.empty(u.size, dtype=object)
        for i, v in enumerate(u):
            wrapped[i] = _Counting(float(v))
        _Counting.calls = 0

        def factory():
            trace.append(("factory", _Counting.calls))
            return constitutive

        _outcome(function, wrapped, geometry, factory, False)
        trace.append(("end", _Counting.calls))
        traces.append(trace)
    assert traces[0] == traces[1]
