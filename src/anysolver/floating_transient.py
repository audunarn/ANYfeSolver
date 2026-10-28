"""Fixed-step six-DOF floating-body transient with wind-thrust feedback (headless, neutral inputs).

Equation of motion in the first-order mean-body frame (SI), generalized coordinates
``q = (translation, small rotation theta)`` through the mean CG:

    (M_dry + A_inf) a + (C + C_ext) v + (K + K_ext - G(t)) q + memory = F(t) + b [T(Vrel) - T_mean]

    memory_n = dt (0.5 K_r[n] v_0 + sum_{j=1..n-1} K_r[n-j] v_j + 0.5 K_r[0] v_n)

This is the Newmark average-acceleration algebra (beta = 1/4, gamma = 1/2) and the direct causal
trapezoidal memory of ``marine_linear``, including the current-step tangent ``0.5 dt K_r[0]``.
``marine_linear`` is not modified; this module only reuses its constants, cancellation polling
and the lag-sum helper for the full-coverage case.  Differences: a kernel may be an explicitly
windowed kernel that is exactly zero at and beyond its declared horizon, external stiffness /
damping / load tangents may be non-symmetric (general LU path; skew parts are reported, never
removed) and a single fixed-body turbine feeds back through a scalar piecewise-linear root.

Turbine (one applicator, rotor axis ``n`` and hub offset ``r`` fixed in the body):

    U_b   = Q0^T U_world - theta x (Q0^T U_world),   v_h = v_t + omega x r
    Vrel  = (U_b - v_h) . n = c0 - d.q - b.v,        b = [n; r x n],  d = [0; (Q0^T U) x n]

With ``q = q_pred + beta dt^2 a``, ``v = v_pred + gamma dt a``, ``a = a_base + H^-1 b T`` the step
reduces to ``g(V) = V - V_base + kappa T(V) = 0`` with
``kappa = (beta dt^2 d + gamma dt b) . H^-1 b``.  Every curve segment is solved exactly, in-domain
roots are kept and shared-knot roots deduplicated; exactly one root is required.  Zero, multiple or
a whole interval of roots is an explicit non-success.  Negative slopes (negative aerodynamic
damping) are reported, never clamped.  No lagged force, extrapolation or timestep change.

No finite rotations are supported: the rotation envelope (default 5 degrees) is enforced on every
step and a violation is an explicit non-success that keeps the diagnostic history.  Tests of this
module are development evidence, not scientific qualification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional
import math

import numpy as np
from scipy.linalg import cho_factor, lu_factor, lu_solve

from .marine_linear import NEWMARK_BETA, NEWMARK_GAMMA, _cancel_requested, _memory_past

__all__ = [
    "FloatingTransientError",
    "FloatingTransientResourceConfig",
    "FloatingTransientResult",
    "FloatingTransientSpec",
    "solve_floating_transient",
    "numeric_array_digest",
    "waterline_guard",
    "FRAME_BODY_FIRST_ORDER",
]

THRUST_ABSCISSA = "rotor_axial_speed"
ROTATION_LIMIT_RAD = float(np.deg2rad(5.0))
RESIDUAL_TOLERANCE = 1.0e-10
_SYMMETRY_TOL = 1.0e-10
_GRID_TOL = 1.0e-9

STATUS_COMPLETE = "complete"
STATUS_CANCELLED = "cancelled"
STATUS_NO_ROOT = "no_root"
STATUS_MULTIPLE_ROOTS = "multiple_roots"
STATUS_DEGENERATE = "degenerate_interval"
STATUS_ROTATION = "rotation_envelope_exceeded"
STATUS_RESIDUAL = "residual_exceeded"
STATUS_NONFINITE = "non_finite"
STATUS_NATIVE_WORK = "native_hydrodynamic_work_exceeded"
STATUS_NATIVE_PENCIL = "native_hydrodynamic_pencil_unresolved"


class FloatingTransientError(ValueError):
    """Input admission or resource refusal (raised before any integration)."""


class BoundedNativeAdmissionError(FloatingTransientError):
    """bounded_native pre-run admission refused; ``report`` holds every computed (serialisable) measure."""

    def __init__(self, message: str, report: Optional[dict] = None):
        super().__init__(message)
        self.report = report or {}


@dataclass(frozen=True)
class FloatingTransientResourceConfig:
    """Preflight limits; exceeding one refuses before allocation."""

    max_working_bytes: int = 256 * 1024 * 1024
    max_steps: int = 2_000_000
    max_operations: float = 5.0e11

    def __post_init__(self) -> None:
        # validated before any allocation; the approved defaults may only tighten
        for name, ceiling in (("max_working_bytes", 256 * 1024 * 1024), ("max_steps", 2_000_000),
                              ("max_operations", 5.0e11)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise FloatingTransientError(f"resource_config.{name}: a number is required")
            if not (np.isfinite(value) and 0 < value <= ceiling):
                raise FloatingTransientError(
                    f"resource_config.{name}={value!r}: must lie in (0, {ceiling:g}]; the approved ceiling can only tighten")


FRAME_BODY_FIRST_ORDER = "body_attached_first_order"

#: hydrodynamic admission modes.  'strict' (DEFAULT, unchanged): A_inf and every kernel sample must be symmetric.
#: 'bounded_native' (opt-in): raw A_inf and raw K(t) are kept in the equation (never symmetrised) and admitted only
#: through the fixed, non-widenable bounds below plus pre-run pencil and runtime energy-work gates.
HYDRO_ADMISSION_STRICT = "strict"
HYDRO_ADMISSION_BOUNDED_NATIVE = "bounded_native"
#: 'development_unverified' (explicit opt-in): the bounded_native gates, pencils and runtime work gates all apply,
#: except that a hydro-impedance NON-PASSIVITY failure is retained in the report (status 'failed_retained_development',
#: 'unpassed_gates') instead of refusing.  Results are labelled unverified hydrodynamics; never an accepted result.
HYDRO_ADMISSION_DEVELOPMENT = "development_unverified"
HYDRO_ADMISSIONS = (HYDRO_ADMISSION_STRICT, HYDRO_ADMISSION_BOUNDED_NATIVE, HYDRO_ADMISSION_DEVELOPMENT)
BOUNDED_MASS_SKEW_CEILING = 1.0e-3        # rho = ||S^-1/2 W S^-1/2||_2, S = sym(M_dry + A_inf), W = skew(A_inf)
BOUNDED_KERNEL_SKEW_CEILING = 1.0e-3      # max_t ||skew(K_w(t))||_2 / max_t ||sym(K_w(t))||_2 (ONE global scale)
BOUNDED_IMPEDANCE_CEILING = 1.0e-3        # -lambda_min(S^-1/2 Herm Z(w) S^-1/2) / w
BOUNDED_WORK_CEILING = 1.0e-3             # cumulative |skew work| and negative memory work vs reference energy
#: numerical classification bands for the raw-vs-symmetric pencils (fractions of the pencil scale times eps):
#: below REAL is numerically real / non-growing, above COMPLEX is complex / growing, between is AMBIGUOUS (non-success).
PENCIL_REAL_BAND = 64.0
PENCIL_COMPLEX_BAND = 4096.0
STATUS_SKEW_WORK = "skew_work_exceeded"
STATUS_MEMORY_WORK = "memory_work_negative"
DIGEST_PROTOCOL = "numeric-array-v1"


def numeric_array_digest(array: Any) -> str:
    """Documented digest protocol shared with ANYhydro: sha256(b'numeric-array-v1|' + repr(shape) + b'|' + '<f8' bytes)."""
    import hashlib
    raw = _real_input("array", array)
    h = hashlib.sha256()
    h.update(DIGEST_PROTOCOL.encode() + b"|" + repr(tuple(int(x) for x in raw.shape)).encode() + b"|")
    # C-order buffered iteration in chunks of at most 65536 elements: strided / transposed / non-native-dtype input is
    # never materialised as a full contiguous copy and only a 512 KiB chunk is converted and hashed at a time
    if raw.size:
        it = np.nditer(raw, flags=["external_loop", "buffered", "zerosize_ok"], op_dtypes=[np.dtype("<f8")],
                       casting="safe", order="C", buffersize=1 << 16)
        for chunk in it:
            h.update(np.ascontiguousarray(chunk, dtype="<f8").tobytes())
    return h.hexdigest()


def _shape_of(x: Any) -> tuple:
    """Shape WITHOUT converting / allocating: ndarray-likes report ``shape``; nested lists / tuples are walked along
    their first elements (regularity is verified later by the real conversion).  Anything else must be passed as
    an ndarray (typed contract) so that admission never allocates a hidden full-size copy."""
    shape = getattr(x, "shape", None)
    if shape is not None:
        return tuple(int(s) for s in shape)
    dims = []
    cur = x
    while isinstance(cur, (list, tuple)):
        dims.append(len(cur))
        if len(dims) > 6:
            raise FloatingTransientError("nested input is too deep; pass an ndarray")
        if not cur:
            break
        cur = cur[0]
    if dims or isinstance(cur, (int, float, complex, np.generic)):
        # Rectangularity is verified without NumPy, with recursion bounded by six dimensions: a ragged or
        # mixed list is rejected here, before any payload conversion, instead of being estimated from its first element
        scalar = (int, float, complex, np.generic)
        def check_rectangular(node, depth):
            if not isinstance(node, (list, tuple)) or len(node) != dims[depth]:
                raise FloatingTransientError(
                    f"ragged input: expected length {dims[depth]} at nesting depth {depth}; pass a rectangular list or an ndarray")
            last = depth == len(dims) - 1
            for item in node:
                if last:
                    if isinstance(item, (list, tuple)) or not isinstance(item, scalar):
                        raise FloatingTransientError("ragged input: expected a scalar at the innermost level")
                else:
                    check_rectangular(item, depth + 1)
        if dims:
            check_rectangular(x, 0)
        return tuple(dims)
    raise FloatingTransientError(
        f"input of type {type(x).__name__} must be an ndarray or a (nested) list / tuple of numbers")


def _prod(shape: tuple) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _preflight(spec: "FloatingTransientSpec", resource: FloatingTransientResourceConfig) -> None:
    """Admission from metadata / shapes ONLY (no payload copy): every own array, result array, immutable
    return copy and peak temporary / hash buffer is counted with Python integers before the first copy."""
    ts = _shape_of(spec.times)
    if len(ts) != 1 or ts[0] < 2:
        raise FloatingTransientError("times: a 1-D array with at least two samples is required")
    n = ts[0]
    if n > resource.max_steps:
        raise FloatingTransientError(f"{n} steps exceed the resource limit {resource.max_steps}")
    comps = dict(spec.load_components or {})
    n_loads = len(comps)
    n_mass = len(dict(spec.mass_components or {})) or 1
    nk = _shape_of(spec.radiation_kernel)[0] if len(_shape_of(spec.radiation_kernel)) == 3 else 0
    f8 = 8
    inputs = n * f8 + n_loads * n * 6 * f8 + n * 6 * f8                    # times, load copies, F_total
    if spec.external_load_tangent is not None:
        inputs += _prod(_shape_of(spec.external_load_tangent)) * f8 * 2     # payload copy + per-step Ks temporaries
    inputs += _prod(_shape_of(spec.radiation_kernel)) * f8
    inputs += 64 * 36 * f8                                                  # small 6x6 matrices
    if spec.wind_velocity is not None:
        wm = _shape_of(spec.wind_velocity)
        inputs += (_prod(wm) + (wm[0] if wm else 0)) * f8 + 3 * n * 3 * f8  # wind copy, Ub, U0, interpolation temps
    # explicit validation temporaries: np.diff / abs(steps - dt) (3 vectors), the bool isfinite masks of the larger
    # payload arrays (1/8 of each) and one wind-interpolation vector; the incremental digest needs only 512 KiB
    inputs += 5 * n * f8 + (inputs // 8) + (1 << 20)
    work = n * f8 * (6 * (7 + n_mass + 3) + 3 + 10 + 3)                     # q,v,a,mem,dmp,stf,tf, dry comps, ainf, res, scales...
    returned = work + n_loads * n * 6 * f8                                  # immutable-return copies
    largest = n * 6 * f8 * 2 if n_loads == 0 else max(n * 6 * f8, 1) * 2   # tobytes + frombuffer temporaries
    total = inputs + work + returned + largest
    native_frequencies = 0
    if spec.hydro_admission == HYDRO_ADMISSION_BOUNDED_NATIVE:
        if nk > 1001:
            raise FloatingTransientError("bounded-native v1 retains the 1001 kernel-sample ceiling")
        shape_w = _shape_of(spec.admission_omegas)
        if len(shape_w) != 1 or not 1 <= shape_w[0] <= 513:
            raise FloatingTransientError("bounded-native admission requires 1..513 source frequencies")
        native_frequencies = shape_w[0]
        total += 4 * nk * 36 * f8 + 14 * n * f8 + native_frequencies * f8 * 4
    if total > resource.max_working_bytes:
        raise FloatingTransientError(
            f"estimated working memory {total} B (inputs {inputs}, work arrays {work}, immutable returns {returned}, "
            f"temporaries {largest}) exceeds the limit {resource.max_working_bytes} B")
    n_h = nk if spec.kernel_horizon is None else max(nk - 1, 0)
    ops = float(n) * min(n, max(n_h, 1)) * 72.0
    if native_frequencies:
        ops += float(n) * 12**3 * 8 + native_frequencies * nk * 144.0
    if ops > resource.max_operations:
        raise FloatingTransientError(f"estimated {ops:.3g} operations exceed the limit {resource.max_operations:.3g}")


def _real_input(name: str, value: Any) -> np.ndarray:
    """Real data as an array; complex input (even with zero imaginary part) is refused BEFORE any float cast."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise FloatingTransientError(f"{name}: not numeric ({exc})") from exc
    if raw.dtype.kind == "c":
        raise FloatingTransientError(
            f"{name}: complex-valued input is refused (real time-domain data required; no imaginary part is dropped)")
    return raw


def _arr(name: str, value: Any, shape: Optional[tuple] = None, ndim: Optional[int] = None) -> np.ndarray:
    try:
        out = np.array(_real_input(name, value), dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise FloatingTransientError(f"{name}: not numeric ({exc})") from exc
    if shape is not None and out.shape != shape:
        raise FloatingTransientError(f"{name}: expected shape {shape}, got {out.shape}")
    if ndim is not None and out.ndim != ndim:
        raise FloatingTransientError(f"{name}: expected {ndim}-D array, got shape {out.shape}")
    if not np.all(np.isfinite(out)):
        raise FloatingTransientError(f"{name}: non-finite values are not admitted")
    return out


def _ro(a: np.ndarray) -> np.ndarray:
    """Detached, bytes-backed array: ``setflags(write=True)`` cannot re-enable writes."""
    arr = np.array(a, copy=True, order="C")
    if arr.size == 0:
        arr.flags.writeable = False
        return arr
    return np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)


def _deep_freeze(value: Any) -> Any:
    """Deep-detached immutable copy of nested identity data (mappings, sequences, arrays, scalars)."""
    import copy
    from types import MappingProxyType
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return _ro(value)
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


def _sym_check(name: str, m: np.ndarray) -> None:
    skew = float(np.max(np.abs(m - m.T)))
    scale = max(1.0, float(np.max(np.abs(m))))
    if skew > _SYMMETRY_TOL * scale:
        raise FloatingTransientError(
            f"{name}: not symmetric (max |M-M^T| = {skew:.3e}); owner-supplied mass / hydrodynamic "
            "matrices are not symmetrised. Route genuine non-symmetric external terms through "
            "external_stiffness / external_damping with documented attachment forces"
        )


def _skew_report(m: np.ndarray) -> tuple:
    return float(np.linalg.norm(0.5 * (m - m.T))), float(np.linalg.norm(0.5 * (m + m.T)))


@dataclass(frozen=True)
class FloatingTransientSpec:
    """Neutral-array input.  ``times`` is the actual accepted uniform time grid (seconds).

    Convention (``frame``, the only admitted value ``"body_attached_first_order"``): q = (translation, small
    rotation) about the mean CG; force, moment and wind components are those of the first-order BODY-ATTACHED
    frame (``U_b = Q0^T U - theta x Q0^T U``).  A fixed mean-frame EOM or finite rotations are rejected.  The
    static weight / buoyancy / gravity-restoring integration and the mean equilibrium are NOT part of this
    solver: ``mean_thrust`` is supplied (and only compared with ``T(V_ref)`` when ``reference_wind_velocity`` is
    given); ``mean_equilibrium_status`` stays ``"not_verified"``.

    Load channels (``load_components``: name -> (n, 6)) are generalized forces already expressed in the
    first-order body frame about the mean CG and are summed; they are recorded separately.
    ``mass_components`` (name -> 6x6; e.g. hull, tower, RNA, ballast) must sum to ``mass_dry`` when
    both are given (relative 1e-10); either may be omitted.  No turbine mass is ever added here.
    """

    times: Any
    load_components: Mapping[str, Any]
    added_mass_infinite: Any
    radiation_kernel: Any
    mass_dry: Any = None
    mass_components: Optional[Mapping[str, Any]] = None
    damping: Any = None
    stiffness: Any = None
    external_damping: Any = None
    external_stiffness: Any = None
    external_load_tangent: Any = None          # (n, 6, 6) G(t): load = F + G q (e.g. world-load pull-back)
    kernel_horizon: Optional[float] = None     # declared finite memory H; kernel exactly zero at H
    initial_displacement: Any = None
    initial_velocity: Any = None
    # wind / turbine (all optional together)
    wind_times: Any = None
    wind_velocity: Any = None                  # (m, 3) world velocity
    Q0: Any = None                             # mean-body -> world
    hub_offset_body: Any = None
    rotor_axis_body: Any = None
    thrust_speeds: Any = None
    thrust_values: Any = None
    thrust_abscissa: str = THRUST_ABSCISSA
    mean_thrust: Optional[float] = None        # REQUIRED (finite) with a turbine; never silently 0
    reference_wind_velocity: Any = None        # optional world (3,) mean wind to check T(V_ref) against mean_thrust
    # scaling / guards
    force_scale: Optional[float] = None        # optional reference scales: they can only TIGHTEN the natural gate
    moment_scale: Optional[float] = None
    residual_tolerance: float = RESIDUAL_TOLERANCE
    rotation_limit: float = ROTATION_LIMIT_RAD
    identity: Mapping[str, Any] = field(default_factory=dict)
    frame: str = FRAME_BODY_FIRST_ORDER        # the only convention: first-order BODY-ATTACHED components
    # ---- opt-in hydrodynamic admission (strict default unchanged; see bounded_native_admission)
    hydro_admission: str = "strict"            # 'strict' | 'bounded_native'
    admission_omegas: Any = None               # bounded_native: explicit verified source frequency grid (rad/s)
    native_source_binding: Optional[Mapping[str, Any]] = None   # bounded_native: kernel_digest, ainf_digest, kernel_dt

    def __post_init__(self) -> None:
        if self.hydro_admission not in HYDRO_ADMISSIONS:
            raise FloatingTransientError(f"hydro_admission must be one of {HYDRO_ADMISSIONS}")
        self._validate_scalars()

    def _validate_scalars(self) -> None:
        # direct construction (and dataclasses.replace) validates the scalar gates; they may only tighten
        def num(name: str, value: Any) -> float:
            try:
                x = float(value)
            except (TypeError, ValueError) as exc:
                raise FloatingTransientError(f"{name}: not a number ({exc})") from exc
            if not np.isfinite(x):
                raise FloatingTransientError(f"{name}: must be finite, got {x!r}")
            return x

        tol = num("residual_tolerance", self.residual_tolerance)
        if not (0.0 < tol <= RESIDUAL_TOLERANCE):
            raise FloatingTransientError(
                f"residual_tolerance {tol!r}: must lie in (0, {RESIDUAL_TOLERANCE:g}]; the approved gate can only tighten")
        lim = num("rotation_limit", self.rotation_limit)
        if not (0.0 < lim <= ROTATION_LIMIT_RAD):
            raise FloatingTransientError(
                f"rotation_limit {lim!r}: must lie in (0, {ROTATION_LIMIT_RAD!r}] rad; the 5 degree envelope can only tighten")
        for name in ("force_scale", "moment_scale"):
            value = getattr(self, name)
            if value is not None and not num(name, value) > 0.0:
                raise FloatingTransientError(f"{name}: must be finite and positive")
        if self.mean_thrust is not None:
            num("mean_thrust", self.mean_thrust)
        if self.frame != FRAME_BODY_FIRST_ORDER:
            raise FloatingTransientError(
                f"frame={self.frame!r}: only {FRAME_BODY_FIRST_ORDER!r} (small rotations, force / wind components in the "
                "first-order body-attached frame; q = translation, rotation about the mean CG) is supported; "
                "fixed mean-frame and finite-rotation conventions are rejected")
        if self.reference_wind_velocity is not None:
            ref = np.asarray(_real_input("reference_wind_velocity", self.reference_wind_velocity), dtype=np.float64)
            if ref.shape != (3,) or not np.all(np.isfinite(ref)):
                raise FloatingTransientError("reference_wind_velocity: a finite world 3-vector is required")
        if self.thrust_abscissa != THRUST_ABSCISSA:
            raise FloatingTransientError(
                f"thrust_abscissa={self.thrust_abscissa!r}: only {THRUST_ABSCISSA!r} is admitted")


@dataclass(frozen=True)
class FloatingTransientResult:
    """Accepted history (possibly partial) with load decomposition and diagnostics."""

    status: str
    message: str
    complete: bool
    cancelled: bool
    failed_step: Optional[int]
    times: np.ndarray
    dt: float
    n_steps: int
    n_accepted: int
    displacements: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    load_components: Mapping[str, np.ndarray]
    dry_inertia_components: Mapping[str, np.ndarray]
    added_inertia_forces: np.ndarray
    memory_forces: np.ndarray
    damping_forces: np.ndarray
    stiffness_forces: np.ndarray
    turbine_forces: np.ndarray                 # b [T - T_mean]
    thrust: np.ndarray
    relative_axial_speed: np.ndarray
    hub_wind_world: np.ndarray
    residual_forces: np.ndarray                # scaled
    kinematic_residual: np.ndarray
    root_candidates: np.ndarray
    aero_damping_slope: np.ndarray             # dT/dV at the accepted root (signed)
    negative_damping_steps: int
    max_rotation: float
    force_scale: float
    moment_scale: float
    max_scaled_residual: float
    external_stiffness_skew_norm: float
    external_stiffness_symmetric_norm: float
    external_damping_skew_norm: float
    effective_matrix_condition: float
    identity: Mapping[str, Any]
    # --- bound identity / convention / diagnostics (all computed from the ACTUAL inputs of this run)
    frame: str = FRAME_BODY_FIRST_ORDER
    kernel_digest: str = ""                    # numeric_array_digest of the sampled radiation kernel actually used
    ainf_digest: str = ""                      # numeric_array_digest of the A_inf actually used
    digest_protocol: str = DIGEST_PROTOCOL
    aero_damping_slope_left: Optional[np.ndarray] = None    # slope of the segment ending at V (== right off the knots)
    aero_damping_slope_right: Optional[np.ndarray] = None   # aero_damping_slope is this right-hand value
    force_scale_used: Optional[np.ndarray] = None           # per-step natural (or tightened) scales actually used
    moment_scale_used: Optional[np.ndarray] = None
    kinematic_scale_used: Optional[np.ndarray] = None
    failed_row_diagnostics: Optional[Mapping[str, Any]] = None   # the excluded rotation / residual row, if any
    mean_reference: Optional[Mapping[str, Any]] = None      # T(V_ref) - mean_thrust, or marked unverified
    mean_equilibrium_status: str = "not_verified"           # mean equilibrium is NOT established by this solver
    hydrodynamic_admission_report: Optional[Mapping[str, Any]] = None
    hydrodynamic_work: Optional[Mapping[str, np.ndarray]] = None

    def raise_for_status(self) -> None:
        if self.status != STATUS_COMPLETE:
            raise FloatingTransientError(f"{self.status}: {self.message}")


def _prepare(spec: FloatingTransientSpec, resource: FloatingTransientResourceConfig) -> dict:
    d: dict = {}
    _preflight(spec, resource)                    # shape / metadata admission BEFORE the first payload copy
    t = _arr("times", spec.times, ndim=1)
    n = t.size
    if n < 2:
        raise FloatingTransientError("times: at least two samples required")
    steps = np.diff(t)
    dt = float(steps[0])
    if dt <= 0.0 or np.max(np.abs(steps - dt)) > _GRID_TOL * dt:
        raise FloatingTransientError("times: a strictly increasing fixed-step grid is required")
    d["times"], d["dt"], d["n"] = t, dt, n
    if n > resource.max_steps:
        raise FloatingTransientError(f"{n} steps exceed the resource limit {resource.max_steps}")

    comps = dict(spec.load_components or {})
    if not comps:
        raise FloatingTransientError("load_components: at least one (n, 6) channel (zeros allowed) is required")
    d["loads"] = {str(k): _arr(f"load_components[{k!r}]", v, shape=(n, 6)) for k, v in comps.items()}

    # mass: dry decomposition
    mc = {}
    if spec.mass_components:
        for k, m in spec.mass_components.items():
            mm = _arr(f"mass_components[{k!r}]", m, shape=(6, 6))
            _sym_check(f"mass_components[{k!r}]", mm)
            mc[str(k)] = mm
    if spec.mass_dry is None and not mc:
        raise FloatingTransientError("mass_dry or mass_components is required")
    total = sum(mc.values()) if mc else None
    if spec.mass_dry is not None:
        md = _arr("mass_dry", spec.mass_dry, shape=(6, 6))
        _sym_check("mass_dry", md)
        if total is not None and np.max(np.abs(md - total)) > 1e-10 * max(1.0, float(np.max(np.abs(md)))):
            raise FloatingTransientError(
                "mass_dry does not equal the sum of mass_components (relative 1e-10 consistency check)")
    else:
        md = total
    if not mc:
        mc = {"dry": md}
    d["mass_dry"], d["mass_components"] = md, mc
    ainf = _arr("added_mass_infinite", spec.added_mass_infinite, shape=(6, 6))
    bounded = spec.hydro_admission in (HYDRO_ADMISSION_BOUNDED_NATIVE, HYDRO_ADMISSION_DEVELOPMENT)
    if not bounded:
        _sym_check("added_mass_infinite", ainf)        # strict default: unchanged
    d["bounded"] = bounded
    d["ainf"] = ainf
    d["M"] = md + ainf
    try:
        cho_factor(d["M"], lower=True)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise FloatingTransientError(f"mass_dry + added_mass_infinite is not positive definite ({exc})") from exc

    zero = np.zeros((6, 6))
    d["C"] = zero if spec.damping is None else _arr("damping", spec.damping, shape=(6, 6))
    d["K"] = zero if spec.stiffness is None else _arr("stiffness", spec.stiffness, shape=(6, 6))
    _sym_check("damping", d["C"])
    _sym_check("stiffness", d["K"])
    d["Cx"] = zero if spec.external_damping is None else _arr("external_damping", spec.external_damping, shape=(6, 6))
    d["Kx"] = zero if spec.external_stiffness is None else _arr("external_stiffness", spec.external_stiffness, shape=(6, 6))
    d["G"] = None if spec.external_load_tangent is None else _arr("external_load_tangent", spec.external_load_tangent, shape=(n, 6, 6))

    # kernel
    ker = _arr("radiation_kernel", spec.radiation_kernel, ndim=3)
    if ker.shape[1:] != (6, 6) or ker.shape[0] < 2:
        raise FloatingTransientError(f"radiation_kernel: expected (n_kernel>=2, 6, 6), got {ker.shape}")
    for i in range(ker.shape[0] if not bounded else 0):        # bounded_native: raw kernel, bounded globally below
        if np.max(np.abs(ker[i] - ker[i].T)) > _SYMMETRY_TOL * max(1.0, float(np.max(np.abs(ker[i])))):
            raise FloatingTransientError(f"radiation_kernel: sample {i} is not symmetric")
    if spec.kernel_horizon is None:
        if ker.shape[0] < n:
            raise FloatingTransientError(
                f"radiation_kernel: {ker.shape[0]} samples do not cover the {n}-step history; supply a kernel "
                "covering every lag or declare an exactly-zero-endpoint windowed kernel via kernel_horizon")
        n_h = ker.shape[0]
    else:
        H = float(spec.kernel_horizon)
        n_h = ker.shape[0] - 1
        if not np.isfinite(H) or abs(n_h * dt - H) > _GRID_TOL * max(dt, H):
            raise FloatingTransientError(
                f"kernel_horizon {H!r} must equal (n_kernel-1)*dt = {n_h * dt!r} on the integration grid")
        if np.any(ker[-1] != 0.0):
            raise FloatingTransientError("windowed kernel must be exactly zero at the horizon endpoint")
    d["kernel"], d["n_h"] = ker, n_h

    d["q0"] = np.zeros(6) if spec.initial_displacement is None else _arr("initial_displacement", spec.initial_displacement, shape=(6,))
    d["v0"] = np.zeros(6) if spec.initial_velocity is None else _arr("initial_velocity", spec.initial_velocity, shape=(6,))

    turbine = [spec.wind_velocity, spec.hub_offset_body, spec.rotor_axis_body, spec.thrust_speeds, spec.thrust_values, spec.Q0]
    if any(x is not None for x in turbine):
        if any(x is None for x in turbine) or spec.wind_times is None:
            raise FloatingTransientError(
                "turbine feedback needs wind_times, wind_velocity, Q0, hub_offset_body, rotor_axis_body, "
                "thrust_speeds and thrust_values together")
        if spec.thrust_abscissa != THRUST_ABSCISSA:
            raise FloatingTransientError(
                f"thrust_abscissa={spec.thrust_abscissa!r}: only {THRUST_ABSCISSA!r} is admitted")
        wt = _arr("wind_times", spec.wind_times, ndim=1)
        wv = _arr("wind_velocity", spec.wind_velocity, shape=(wt.size, 3))
        if wt.size < 2 or np.any(np.diff(wt) <= 0.0):
            raise FloatingTransientError("wind_times must be strictly increasing with >= 2 samples")
        tol = 1e-12 * max(1.0, abs(float(t[-1])))
        if t[0] < wt[0] - tol or t[-1] > wt[-1] + tol:
            raise FloatingTransientError(
                f"wind record [{float(wt[0])!r}, {float(wt[-1])!r}] does not cover the history "
                f"[{float(t[0])!r}, {float(t[-1])!r}]; "
                "no extrapolation or looping")
        Q0 = _arr("Q0", spec.Q0, shape=(3, 3))
        if np.max(np.abs(Q0.T @ Q0 - np.eye(3))) > 1e-10 or np.linalg.det(Q0) <= 0.0:
            raise FloatingTransientError("Q0 must be a proper rotation (body->world)")
        r = _arr("hub_offset_body", spec.hub_offset_body, shape=(3,))
        nax = _arr("rotor_axis_body", spec.rotor_axis_body, shape=(3,))
        nn = float(np.linalg.norm(nax))
        if abs(nn - 1.0) > 1e-10:
            raise FloatingTransientError(
                f"rotor_axis_body must already be a unit vector within 1e-10 (norm {nn!r}); it is never normalised silently")
        if spec.mean_thrust is None:
            raise FloatingTransientError("mean_thrust must be supplied explicitly with a turbine (no silent default of 0)")
        sp = _arr("thrust_speeds", spec.thrust_speeds, ndim=1)
        tv = _arr("thrust_values", spec.thrust_values, shape=sp.shape)
        if sp.size < 2 or np.any(sp < 0.0) or np.any(np.diff(sp) <= 0.0) or np.any(tv < 0.0):
            raise FloatingTransientError("thrust curve: >= 2 strictly increasing nonnegative speeds and nonnegative thrusts")
        Ub = np.stack([np.interp(np.clip(t, wt[0], wt[-1]), wt, wv[:, i]) for i in range(3)], axis=1)
        d["turbine"] = dict(
            wind_world=Ub, U0=Ub @ Q0,  # row-wise Q0^T U
            r=r, n=nax, b=np.concatenate([nax, np.cross(r, nax)]),
            speeds=sp, thrusts=tv, Tm=float(spec.mean_thrust),
        )
        if not np.isfinite(d["turbine"]["Tm"]):
            raise FloatingTransientError("mean_thrust must be finite")
        # optional mean-reference consistency (this is NOT a mean-equilibrium solve)
        if spec.reference_wind_velocity is not None:
            Uref = Q0.T @ _arr("reference_wind_velocity", spec.reference_wind_velocity, shape=(3,))
            v_ref = float(Uref @ nax)
            if not (sp[0] <= v_ref <= sp[-1]):
                raise FloatingTransientError(
                    f"reference wind gives rotor-axial speed {v_ref!r} outside the curve domain "
                    f"[{float(sp[0])!r}, {float(sp[-1])!r}]")
            t_ref = float(np.interp(v_ref, sp, tv))
            # a COMPARISON only: it reports the mismatch and does not accept or establish an equilibrium
            d["mean_reference"] = dict(status="compared_against_reference_wind", reference_axial_speed=v_ref,
                                       curve_thrust=t_ref, mean_thrust=d["turbine"]["Tm"],
                                       residual=t_ref - d["turbine"]["Tm"])
        else:
            d["mean_reference"] = dict(status="unverified", reason="no reference_wind_velocity supplied",
                                       mean_thrust=d["turbine"]["Tm"])
    else:
        d["turbine"] = None
        d["mean_reference"] = None

    # optional user reference scales can only tighten the natural per-step gate (validated positive in the spec)
    d["user_fs"] = None if spec.force_scale is None else float(spec.force_scale)
    d["user_ms"] = None if spec.moment_scale is None else float(spec.moment_scale)
    d["hydro_report"] = bounded_native_admission(
        mass_dry=d["mass_dry"], added_mass_infinite=d["ainf"], radiation_kernel=d["kernel"], dt=dt,
        stiffness=d["K"] + d["Kx"] - (d["G"][0] if d["G"] is not None else 0.0), damping=d["C"] + d["Cx"],
        admission_omegas=spec.admission_omegas, native_source_binding=spec.native_source_binding,
        development=spec.hydro_admission == HYDRO_ADMISSION_DEVELOPMENT) if bounded else None
    return d


def _whitener(S: np.ndarray) -> np.ndarray:
    lam, V = np.linalg.eigh(S)
    return (V / np.sqrt(lam)) @ V.T


def _spectral(m: np.ndarray) -> float:
    return float(np.linalg.norm(m, 2)) if m.size else 0.0


def _ratio(num: float, den: float) -> float:
    if den > 0.0:
        return num / den
    return 0.0 if num == 0.0 else float("inf")


def _pencil_classification(raw_M, sym_M, K, C_raw, C_sym) -> dict:
    """Compare mass-whitened raw/reference modes, with backward-error and ambiguity gates."""
    from scipy.optimize import linear_sum_assignment
    eps = np.finfo(float).eps
    Wh = _whitener(sym_M)
    Mr = Wh @ raw_M @ Wh
    Ms = Wh @ sym_M @ Wh
    Kh = Wh @ K @ Wh
    Cr, Cs = Wh @ C_raw @ Wh, Wh @ C_sym @ Wh
    frequency = max(math.sqrt(_spectral(Kh)), _spectral(Cr), _spectral(Cs))
    if frequency == 0.0:
        return dict(status="admitted", reason="zero mechanical operators: neutral rigid motion", backward_error=0.,
                    frequency_scale=0.)
    Kh = Kh / frequency**2
    Cr, Cs = Cr / frequency, Cs / frequency
    out = dict(frequency_scale=frequency, backward_error=0.)

    def eig(A, B):
        values, vectors = np.linalg.eig(np.linalg.solve(B, A))
        residual = np.linalg.norm(A @ vectors - (B @ vectors) * values[None, :], axis=0)
        scale = (_spectral(A) + np.abs(values) * _spectral(B)) * np.linalg.norm(vectors, axis=0)
        error = max((_ratio(float(r), float(d)) for r, d in zip(residual, scale)), default=0.)
        out["backward_error"] = max(out["backward_error"], error)
        return values

    def compare(raw, reference, stage, units):
        i, j = linear_sum_assignment(np.abs(raw[:, None] - reference[None, :]))
        raw, reference = raw[i], reference[j]
        scale = max(float(np.max(np.abs(raw))), float(np.max(np.abs(reference))))
        lo, hi = PENCIL_REAL_BAND * eps * scale, PENCIL_COMPLEX_BAND * eps * scale
        values_raw = np.abs(raw.imag) if stage == "undamped" else raw.real
        values_ref = np.abs(reference.imag) if stage == "undamped" else reference.real
        report = dict(scale=scale * units, real_band=lo * units, complex_band=hi * units,
                      raw=[[float(z.real * units), float(z.imag * units)] for z in raw],
                      reference=[[float(z.real * units), float(z.imag * units)] for z in reference])
        out[stage] = report
        if any(lo < abs(float(x)) <= hi for x in np.r_[values_raw, values_ref]):
            return "ambiguous", f"{stage} mode classification lies inside the numerical uncertainty band"
        if stage == "undamped":
            changed = (values_raw > hi) != (values_ref > hi)
        else:
            changed = (values_raw > hi) != (values_ref > hi)
        if bool(np.any(changed)):
            return "refused", f"native operators change the {stage} stability classification of a matched mode"
        return "admitted", "matched modes retain their reference stability classification"

    raw, reference = eig(Kh, Mr), eig(Kh, Ms)
    status, reason = compare(raw, reference, "undamped", frequency**2)
    if status == "admitted":
        def state(M, C):
            A = np.zeros((12, 12))
            A[:6, 6:] = np.eye(6)
            A[6:, :6] = -np.linalg.solve(M, Kh)
            A[6:, 6:] = -np.linalg.solve(M, C)
            return A
        raw = eig(state(Mr, Cr), np.eye(12))
        reference = eig(state(Ms, Cs), np.eye(12))
        status, reason = compare(raw, reference, "damped", frequency)
    if not np.isfinite(out["backward_error"]) or out["backward_error"] > RESIDUAL_TOLERANCE:
        status, reason = "ambiguous", "mechanical pencil eigenpair backward error exceeds 1e-10"
    out.update(status=status, reason=reason)
    return out


def bounded_native_admission(*, mass_dry: Any, added_mass_infinite: Any, radiation_kernel: Any, dt: float,
                             stiffness: Any, damping: Any, admission_omegas: Any,
                             native_source_binding: Optional[Mapping[str, Any]], development: bool = False) -> dict:
    """Pre-run admission of RAW native A_inf and K(t) (nothing is symmetrised).  Fixed ceilings, no caller widening:

    * S = sym(M_dry + A_inf) positive definite (Cholesky); dry mass symmetry is checked separately as before.
    * rho = ||S^-1/2 W S^-1/2||_2 <= 1e-3, W = skew(A_inf) (unit invariant); the harmonic Frobenius ratio
      ||W||_F / ||sym A_inf||_F is reported alongside.
    * kernel skew: max_t ||skew K_w(t)||_2 / max_t ||sym K_w(t)||_2 <= 1e-3 with K_w = S^-1/2 K S^-1/2 (ONE global scale).
    * hydro impedance at the explicit grid: Z(w) = -i w A_inf + dt sum_k c_k K_k exp(i w t_k) (trapezoid weights c_k);
      -lambda_min(S^-1/2 Herm Z S^-1/2) / w <= 1e-3 (a normalised non-passivity bound, NOT a passivity proof).
    * binding: numeric_array_digest of the ACTUAL kernel and A_inf and the kernel dt must equal native_source_binding.
    * raw-vs-symmetric pencils (stiffness, damping + 0.5 dt K0 raw / sym) via _pencil_classification.
    Returns a JSON-serialisable report; raises BoundedNativeAdmissionError with that report on refusal."""
    Md = _arr("mass_dry", mass_dry, shape=(6, 6))
    A = _arr("added_mass_infinite", added_mass_infinite, shape=(6, 6))
    Kr = _arr("radiation_kernel", radiation_kernel, ndim=3)
    _sym_check("mass_dry", Md)
    if Kr.shape[1:] != (6, 6) or not 2 <= len(Kr) <= 1001:
        raise FloatingTransientError("bounded-native kernel must be (2..1001, 6, 6)")
    if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, (int, float, np.integer, np.floating)) \
            or not np.isfinite(dt) or dt <= 0.0:
        raise FloatingTransientError("bounded-native dt must be finite and positive")
    report: dict = dict(mode=HYDRO_ADMISSION_BOUNDED_NATIVE, ceilings=dict(
        mass_skew=BOUNDED_MASS_SKEW_CEILING, kernel_skew=BOUNDED_KERNEL_SKEW_CEILING,
        impedance=BOUNDED_IMPEDANCE_CEILING, work=BOUNDED_WORK_CEILING))

    def refuse(msg: str) -> None:
        report["status"] = "refused"
        report["reason"] = msg
        raise BoundedNativeAdmissionError(msg, report)

    # ---- binding of the ACTUAL arrays (no report strings as proof)
    if native_source_binding is not None and not isinstance(native_source_binding, Mapping):
        refuse("native_source_binding must be a mapping")
    b = dict(native_source_binding or {})
    if not b:
        refuse("bounded_native requires native_source_binding (kernel_digest, ainf_digest, kernel_dt)")
    try:
        kdt = float(b.get("kernel_dt"))
    except (TypeError, ValueError):
        kdt = float("nan")
    report["binding"] = dict(kernel_digest=numeric_array_digest(Kr), ainf_digest=numeric_array_digest(A), dt=float(dt))
    if (b.get("kernel_digest") != report["binding"]["kernel_digest"] or b.get("ainf_digest") != report["binding"]["ainf_digest"]
            or not (np.isfinite(kdt) and abs(kdt - dt) <= _GRID_TOL * dt)):
        refuse("the supplied kernel / A_inf / dt do not match the native source binding digests")
    # ---- mass
    S = Md + 0.5 * (A + A.T)
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        refuse("S = sym(M_dry + A_inf) is not positive definite")
    Wh = _whitener(S)
    W = 0.5 * (A - A.T)
    rho = _spectral(Wh @ W @ Wh)
    report["mass"] = dict(rho=rho, harmonic_frobenius_ratio=_ratio(float(np.linalg.norm(W)),
                                                                 float(np.linalg.norm(0.5 * (A + A.T)))))
    if not rho <= BOUNDED_MASS_SKEW_CEILING:
        refuse(f"whitened A_inf skew rho = {rho:.3e} exceeds the fixed ceiling {BOUNDED_MASS_SKEW_CEILING:g}")
    # ---- kernel (one global scale over the whitened kernel)
    Kw = np.einsum("ab,kbc,cd->kad", Wh, Kr, Wh)
    sym_max = max((_spectral(0.5 * (k + k.T)) for k in Kw), default=0.0)
    skew_max = max((_spectral(0.5 * (k - k.T)) for k in Kw), default=0.0)
    kr = _ratio(skew_max, sym_max)
    report["kernel"] = dict(global_sym_scale=sym_max, max_skew=skew_max, ratio=kr)
    if not kr <= BOUNDED_KERNEL_SKEW_CEILING:
        refuse(f"whitened kernel skew ratio {kr:.3e} exceeds the fixed ceiling {BOUNDED_KERNEL_SKEW_CEILING:g}")
    # ---- impedance on the explicit grid
    if admission_omegas is None:
        refuse("bounded_native requires an explicit admission_omegas grid")
    w = np.asarray(_real_input("admission_omegas", admission_omegas), dtype=np.float64)
    if w.ndim != 1 or not 1 <= w.size <= 513 or not np.all(np.isfinite(w)) or np.any(w <= 0.0) or np.any(np.diff(w) <= 0.0):
        refuse("admission_omegas must be a strictly increasing positive finite 1-D grid with 1..513 samples")
    t = np.arange(Kr.shape[0]) * dt
    c = np.ones(Kr.shape[0]); c[0] = 0.5; c[-1] = 0.5
    worst = -np.inf
    rows = []
    for om in w:
        Z = -1j * om * A + dt * np.einsum("k,kij->ij", c * np.exp(1j * om * t), Kr)
        Hz = 0.5 * (Z + Z.conj().T)
        lam = float(np.min(np.linalg.eigvalsh(Wh @ Hz @ Wh)))
        val = -lam / om
        rows.append(val)
        worst = max(worst, val)
    report["impedance"] = dict(omegas=w.tolist(), normalised_nonpassivity=rows, worst=worst,
                               status="passed" if worst <= BOUNDED_IMPEDANCE_CEILING else "failed")
    if not worst <= BOUNDED_IMPEDANCE_CEILING:
        if not development:
            refuse(f"normalised hydro-impedance non-passivity {worst:.3e} exceeds the fixed ceiling {BOUNDED_IMPEDANCE_CEILING:g}")
        # explicit development execution of UNVERIFIED hydrodynamics: the failure is retained verbatim, never relabelled
        report["impedance"]["status"] = "failed_retained_development"
        report.setdefault("unpassed_gates", []).append("hydro_impedance_nonpassivity")
    # ---- pencils (same stiffness / external terms; C + 0.5 dt K0 raw vs sym)
    K = _arr("stiffness", stiffness, shape=(6, 6))
    C = _arr("damping", damping, shape=(6, 6))
    K0 = Kr[0]
    try:
        pencil = _pencil_classification(Md + A, S, K, C + 0.5 * dt * K0, C + 0.5 * dt * 0.5 * (K0 + K0.T))
    except (np.linalg.LinAlgError, FloatingPointError, ValueError, OverflowError) as exc:
        refuse(f"mechanical pencil is numerically unresolved: {exc}")
    report["pencil"] = pencil
    if pencil["status"] != "admitted":
        refuse(f"pencil gate: {pencil['reason']}")
    if development:
        report["mode"] = HYDRO_ADMISSION_DEVELOPMENT
        report["status"] = "admitted_development_unverified"
        report["hydrodynamic_verification"] = "unverified"
        report.setdefault("unpassed_gates", [])
        return report
    report["status"] = "admitted"
    return report


def _windowed_memory_past(kernel: np.ndarray, v: np.ndarray, n: int, dt: float, n_h: int) -> np.ndarray:
    """Trapezoidal memory at step n excluding the current-step term; zero beyond lag n_h-1.

    With K[n_h] = 0 exactly this equals the unwindowed sum over every lag.
    """
    if n == 0:
        return np.zeros(v.shape[1])
    if n_h > n:  # full coverage of the past: identical to marine_linear
        return _memory_past(kernel, v, n, dt)
    m_max = n_h - 1
    acc = np.zeros(v.shape[1])
    if m_max >= 1:
        acc = np.einsum("lij,lj->i", kernel[1:m_max + 1], v[n - m_max:n][::-1])
    return dt * acc


def _segment_roots(V_base: float, kappa: float, speeds: np.ndarray, thrusts: np.ndarray) -> tuple:
    """All roots of g(V) = V - V_base + kappa T(V) on the closed curve domain.

    Returns (roots, degenerate_interval_found, n_segments).
    """
    # g is exactly linear on every segment, so its knot values decide everything: a sign change between two
    # knot values is one interior root, a knot value within the evaluation roundoff is that knot's root
    # (shared by both neighbouring segments, counted once), and two adjacent knot roots mean a whole
    # interval of roots.  No velocity tolerance is used, so distinct nearby interior roots are never merged.
    # Knot signs: clearly separated floating values keep their float sign; a value inside the evaluation
    # roundoff bound is decided by EXACT binary rational arithmetic (Fraction.from_float of the very
    # floats used), never replaced by zero.  Only an exactly zero knot is a (shared) knot root; every
    # exact sign change between knots is its own interior root, counted per segment even if two such
    # roots round to the same float.
    from fractions import Fraction
    eps = np.finfo(float).eps
    gk = speeds - V_base + kappa * thrusts
    gtol = 16.0 * eps * (np.abs(speeds) + abs(V_base) + abs(kappa) * np.abs(thrusts))
    vb, kp = Fraction(float(V_base)), Fraction(float(kappa))

    def exact_g(i: int) -> Fraction:
        return Fraction(float(speeds[i])) - vb + kp * Fraction(float(thrusts[i]))

    signs = [0] * speeds.size
    exact: dict = {}
    for i in range(speeds.size):
        if abs(gk[i]) > gtol[i]:
            signs[i] = 1 if gk[i] > 0.0 else -1
        else:
            exact[i] = exact_g(i)
            signs[i] = (exact[i] > 0) - (exact[i] < 0)
    roots: list = [float(speeds[i]) for i in range(speeds.size) if signs[i] == 0]
    degenerate = False
    for i in range(speeds.size - 1):
        s0, s1 = signs[i], signs[i + 1]
        if s0 == 0 and s1 == 0:
            degenerate = True
        elif s0 != 0 and s1 != 0 and s0 != s1:
            a, b = exact.get(i) or exact_g(i), exact.get(i + 1) or exact_g(i + 1)
            v0, v1 = float(speeds[i]), float(speeds[i + 1])
            V = float(Fraction(v0) + (a / (a - b)) * (Fraction(v1) - Fraction(v0)))
            roots.append(min(max(V, v0), v1))
    roots.sort()
    return roots, degenerate, speeds.size - 1


def solve_floating_transient(
    spec: FloatingTransientSpec,
    *,
    cancellation_token: Any = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    resource_config: Any = None,
) -> FloatingTransientResult:
    """Integrate the transient.  Failures after admission return a partial result with a status."""
    if resource_config is None:
        resource = FloatingTransientResourceConfig()
    elif isinstance(resource_config, FloatingTransientResourceConfig):
        resource = resource_config
    elif isinstance(resource_config, Mapping):
        try:
            resource = FloatingTransientResourceConfig(**dict(resource_config))
        except TypeError as exc:
            raise FloatingTransientError(f"resource_config: {exc}") from exc
    else:
        raise FloatingTransientError("resource_config must be None, a mapping or FloatingTransientResourceConfig")
    _check_token(cancellation_token)              # shape / type of the token validated before any step
    d = _prepare(spec, resource)
    n, dt, t = d["n"], d["dt"], d["times"]
    beta, gamma = NEWMARK_BETA, NEWMARK_GAMMA
    M, kernel, n_h = d["M"], d["kernel"], d["n_h"]
    K0 = kernel[0]
    Cs = d["C"] + d["Cx"] + 0.5 * dt * K0       # includes current-step memory tangent
    C_nomem = d["C"] + d["Cx"]
    Ks_const = d["K"] + d["Kx"]
    turb = d["turbine"]
    loads = d["loads"]
    F_total = sum(loads.values())
    user_fs, user_ms = d["user_fs"], d["user_ms"]

    q = np.zeros((n, 6)); v = np.zeros((n, 6)); a = np.zeros((n, 6))
    mem = np.zeros((n, 6)); dmp = np.zeros((n, 6)); stf = np.zeros((n, 6)); tf = np.zeros((n, 6))
    dry = {k: np.zeros((n, 6)) for k in d["mass_components"]}
    ainf_f = np.zeros((n, 6)); thrust = np.zeros(n); vrel = np.zeros(n)
    hub_wind = np.zeros((n, 3)); res = np.zeros((n, 6)); kin = np.zeros(n)
    cand = np.zeros(n, dtype=int); slope = np.zeros(n); slope_l = np.zeros(n); neg_flag = np.zeros(n, dtype=bool)
    fscale = np.zeros(n); mscale = np.zeros(n); kscale = np.zeros(n)
    max_cond = 0.0

    def ratio(numerator: float, scale: float) -> float:
        """Relative size with explicit zero-scale handling: 0/0 = 0, nonzero/0 = failure (inf)."""
        if scale > 0.0:
            return numerator / scale
        return 0.0 if numerator == 0.0 else float("inf")

    def Ks_at(k: int) -> np.ndarray:
        return Ks_const if d["G"] is None else Ks_const - d["G"][k]

    def curve_T(V: float) -> float:
        return float(np.interp(V, turb["speeds"], turb["thrusts"]))

    def curve_slopes(V: float) -> tuple:
        """(left, right) slopes adjacent to V; equal away from the knots (end knots use their only segment)."""
        sp = turb["speeds"]
        out = []
        for side in ("left", "right"):
            i = max(min(int(np.searchsorted(sp, V, side=side)) - 1, sp.size - 2), 0)
            out.append(float((turb["thrusts"][i + 1] - turb["thrusts"][i]) / (sp[i + 1] - sp[i])))
        return out[0], out[1]

    def in_domain(V: float) -> bool:
        return turb["speeds"][0] <= V <= turb["speeds"][-1]

    def vrel_of(k: int, qk: np.ndarray, vk: np.ndarray) -> float:
        U0 = turb["U0"][k]
        c0 = float(U0 @ turb["n"])
        dvec = np.concatenate([np.zeros(3), np.cross(U0, turb["n"])])
        return c0 - float(dvec @ qk) - float(turb["b"] @ vk)

    def record(k: int, memory_k: np.ndarray, T: float, V: float) -> None:
        for name, mi in d["mass_components"].items():
            dry[name][k] = mi @ a[k]
        ainf_f[k] = d["ainf"] @ a[k]
        mem[k] = memory_k
        dmp[k] = C_nomem @ v[k]
        stf[k] = Ks_at(k) @ q[k]
        if turb is not None:
            tf[k] = turb["b"] * (T - turb["Tm"])
            thrust[k] = T
            vrel[k] = V
            hub_wind[k] = turb["wind_world"][k]
        inertia = M @ a[k]
        r_k = F_total[k] + tf[k] - (inertia + dmp[k] + stf[k] + memory_k)
        # natural per-step force / moment scales: the largest ACTUAL contributing term (loads, actual thrust force
        # b[T - T_mean], inertia, damping, stiffness, memory), no undeclared floors.  A user reference scale can
        # only tighten (min), and 0/0 = 0 while a nonzero residual at zero scale is a failure.
        parts = np.stack([F_total[k], tf[k], inertia, dmp[k], stf[k], memory_k])
        nat_f = float(np.max(np.linalg.norm(parts[:, :3], axis=1)))
        nat_m = float(np.max(np.linalg.norm(parts[:, 3:], axis=1)))
        fscale[k] = nat_f if user_fs is None else min(nat_f, user_fs)
        mscale[k] = nat_m if user_ms is None else min(nat_m, user_ms)
        res[k, :3] = [ratio(abs(float(x)), float(fscale[k])) for x in r_k[:3]]
        res[k, 3:] = [ratio(abs(float(x)), float(mscale[k])) for x in r_k[3:]]
        if turb is not None:
            U0 = turb["U0"][k]
            dvec = np.concatenate([np.zeros(3), np.cross(U0, turb["n"])])
            terms = (abs(float(U0 @ turb["n"])), abs(float(dvec @ q[k])), abs(float(turb["b"] @ v[k])), abs(V))
            kscale[k] = max(terms)
            kin[k] = ratio(abs(vrel_of(k, q[k], v[k]) - V), float(kscale[k]))

    status, message, failed = STATUS_COMPLETE, "all steps accepted", None
    failed_diag = None
    native_report = None if d["hydro_report"] is None else dict(d["hydro_report"])
    native_work = None
    native_failure = None
    if native_report is not None:
        native_work = {name: np.zeros(n) for name in (
            "absolute_skew_work", "radiation_work", "reference_energy", "skew_fraction", "negative_radiation_fraction")}
        mass_skew = 0.5 * (M - M.T)
        kernel_skew = 0.5 * (kernel - kernel.transpose(0, 2, 1))
        sym_mass = 0.5 * (M + M.T)
        sym_stiffness = 0.5 * (Ks_const + Ks_const.T)
        eig_k, vec_k = np.linalg.eigh(sym_stiffness)
        negative_k = (vec_k * np.minimum(eig_k, 0.0)) @ vec_k.T
        absolute_k = (vec_k * np.abs(eig_k)) @ vec_k.T
        eig_c, vec_c = np.linalg.eigh(0.5 * (C_nomem + C_nomem.T))
        negative_c = (vec_c * np.minimum(eig_c, 0.0)) @ vec_c.T
        initial_energy = float(0.5 * d["v0"] @ sym_mass @ d["v0"] + 0.5 * d["q0"] @ absolute_k @ d["q0"])
        initial_negative_potential = float(0.5 * d["q0"] @ negative_k @ d["q0"])
        previous_powers = np.zeros(3)
        external_work = 0.0
        native_report["runtime"] = dict(pencil_checks=0, max_pencil_backward_error=0.0,
                                         max_skew_fraction=0.0, max_negative_radiation_fraction=0.0)

    def native_guard(k):
        nonlocal native_failure, external_work, previous_powers
        if native_work is None:
            return
        # Include the actual local turbine tangent on both sides of a curve knot.
        slopes = [0.0] if turb is None else sorted({float(slope_l[k]), float(slope[k])})
        for local_slope in slopes:
            stiffness = Ks_at(k)
            damping = C_nomem
            if turb is not None:
                dvec = np.r_[np.zeros(3), np.cross(turb["U0"][k], turb["n"])]
                stiffness = stiffness + local_slope * np.outer(turb["b"], dvec)
                damping = damping + local_slope * np.outer(turb["b"], turb["b"])
            try:
                pencil = _pencil_classification(M, sym_mass, stiffness,
                                                damping + 0.5 * dt * K0,
                                                damping + 0.25 * dt * (K0 + K0.T))
            except (np.linalg.LinAlgError, FloatingPointError, ValueError, OverflowError) as exc:
                native_failure = dict(row=k, time=float(t[k]), kind="pencil", reason=str(exc))
                raise _Fail(STATUS_NATIVE_PENCIL, f"native mechanical pencil is numerically unresolved: {exc}",
                            row=k, rotation=float(np.linalg.norm(q[k, 3:]))) from exc
            runtime = native_report["runtime"]
            runtime["pencil_checks"] += 1
            runtime["max_pencil_backward_error"] = max(runtime["max_pencil_backward_error"], pencil["backward_error"])
            if pencil["status"] != "admitted":
                native_failure = dict(row=k, time=float(t[k]), kind="pencil", report=pencil)
                raise _Fail(STATUS_NATIVE_PENCIL, pencil["reason"], row=k, rotation=float(np.linalg.norm(q[k, 3:])))
        skew_memory = np.zeros(6) if k == 0 else (
            _windowed_memory_past(kernel_skew, v, k, dt, n_h) + 0.5 * dt * kernel_skew[0] @ v[k])
        skew_power = abs(float(v[k] @ mass_skew @ a[k])) + abs(float(v[k] @ skew_memory))
        radiation_power = float(v[k] @ mem[k])
        forcing = F_total[k] + tf[k] - 0.5 * (d["Kx"] - d["Kx"].T) @ q[k] - negative_c @ v[k]
        if d["G"] is not None:
            forcing = forcing + d["G"][k] @ q[k]
        supplied_power = abs(float(v[k] @ forcing))
        powers = np.array([skew_power, radiation_power, supplied_power])
        if not np.all(np.isfinite(powers)):
            raise _Fail(STATUS_NONFINITE, "non-finite native energy diagnostic", row=k,
                        rotation=float(np.linalg.norm(q[k, 3:])))
        if k:
            increment = 0.5 * dt * (previous_powers + powers)
            native_work["absolute_skew_work"][k] = native_work["absolute_skew_work"][k - 1] + increment[0]
            native_work["radiation_work"][k] = native_work["radiation_work"][k - 1] + increment[1]
            external_work += increment[2]
        previous_powers = powers
        released = max(0.0, initial_negative_potential - float(0.5 * q[k] @ negative_k @ q[k]))
        reference = initial_energy + external_work + released
        native_work["reference_energy"][k] = reference
        skew_fraction = _ratio(float(native_work["absolute_skew_work"][k]), reference)
        negative_fraction = _ratio(max(0.0, -float(native_work["radiation_work"][k])), reference)
        native_work["skew_fraction"][k], native_work["negative_radiation_fraction"][k] = skew_fraction, negative_fraction
        runtime = native_report["runtime"]
        runtime["max_skew_fraction"] = max(runtime["max_skew_fraction"], skew_fraction)
        runtime["max_negative_radiation_fraction"] = max(runtime["max_negative_radiation_fraction"], negative_fraction)
        if max(skew_fraction, negative_fraction) > BOUNDED_WORK_CEILING:
            native_failure = dict(row=k, time=float(t[k]), kind="work", skew_fraction=skew_fraction,
                                  negative_radiation_fraction=negative_fraction, reference_energy=reference)
            raise _Fail(STATUS_NATIVE_WORK, "native skew or negative radiation work exceeds the fixed 1e-3 energy budget",
                        row=k, rotation=float(np.linalg.norm(q[k, 3:])))
    cancelled = False
    n_acc = 0
    max_rot = 0.0
    tol = float(spec.residual_tolerance)
    lim = float(spec.rotation_limit)

    # ---- step 0: explicit acceleration at the initial state, zero radiation prehistory
    q[0], v[0] = d["q0"], d["v0"]
    T0 = Vr0 = 0.0
    try:
        if turb is not None:
            Vr0 = vrel_of(0, q[0], v[0])
            if not in_domain(Vr0):
                raise _Fail(STATUS_NO_ROOT, f"initial rotor-axial speed {Vr0!r} at t={float(t[0])!r} is outside the curve "
                            f"domain [{float(turb['speeds'][0])!r}, {float(turb['speeds'][-1])!r}]")
            T0 = curve_T(Vr0)
            slope_l[0], slope[0] = curve_slopes(Vr0)
            neg_flag[0] = min(slope_l[0], slope[0]) < 0.0
        rhs0 = F_total[0] - C_nomem @ v[0] - Ks_at(0) @ q[0]
        if turb is not None:
            rhs0 = rhs0 + turb["b"] * (T0 - turb["Tm"])
        a[0] = np.linalg.solve(M, rhs0)
        if not np.all(np.isfinite(a[0])):
            raise _Fail(STATUS_NONFINITE, "non-finite initial acceleration")
        record(0, np.zeros(6), T0, Vr0)
        cand[0] = 0
        rot0 = float(np.linalg.norm(q[0, 3:]))
        bad = rot0 > lim
        if bad or float(np.max(res[0])) > tol or kin[0] > tol:
            raise _Fail(STATUS_ROTATION if bad else STATUS_RESIDUAL,
                        f"row 0 (t={float(t[0])!r}) rejected and excluded from the history: rotation {rot0:.6g} rad "
                        f"(limit {lim:.6g}), scaled residual {float(np.max(res[0])):.3e}, kinematic {float(kin[0]):.3e} "
                        f"(tolerance {tol:.1e})", row=0, rotation=rot0)
        native_guard(0)
        n_acc = 1
        max_rot = rot0
        if progress_callback is not None:
            progress_callback(1, n)
        cancelled = _poll(cancellation_token)

        cached = None
        if not cancelled:
            for k in range(1, n):
                Ks = Ks_at(k)
                H = M + gamma * dt * Cs + beta * dt * dt * Ks
                if d["G"] is not None or cached is None:
                    cached = (lu_factor(H), H)
                    try:
                        max_cond = max(max_cond, float(np.linalg.cond(H)))
                    except np.linalg.LinAlgError:
                        max_cond = float("inf")
                lu = cached[0]
                q_pred = q[k - 1] + dt * v[k - 1] + dt * dt * (0.5 - beta) * a[k - 1]
                v_pred = v[k - 1] + dt * (1.0 - gamma) * a[k - 1]
                mem_past = _windowed_memory_past(kernel, v, k, dt, n_h)
                rhs = F_total[k] - mem_past - Cs @ v_pred - Ks @ q_pred
                T = V = 0.0
                if turb is None:
                    a_k = lu_solve(lu, rhs)
                else:
                    b = turb["b"]
                    Hb = lu_solve(lu, b)
                    a_base = lu_solve(lu, rhs - b * turb["Tm"])
                    U0 = turb["U0"][k]
                    c0 = float(U0 @ turb["n"])
                    dvec = np.concatenate([np.zeros(3), np.cross(U0, turb["n"])])
                    q_b = q_pred + beta * dt * dt * a_base
                    v_b = v_pred + gamma * dt * a_base
                    V_base = c0 - float(dvec @ q_b) - float(b @ v_b)
                    kappa = float((beta * dt * dt * dvec + gamma * dt * b) @ Hb)
                    roots, degenerate, nseg = _segment_roots(V_base, kappa, turb["speeds"], turb["thrusts"])
                    cand[k] = len(roots)
                    if degenerate:
                        raise _Fail(STATUS_DEGENERATE, f"a whole interval of roots at step {k} (t={float(t[k])!r}); "
                                    "the thrust curve and system are degenerate here")
                    if len(roots) == 0:
                        raise _Fail(STATUS_NO_ROOT, f"no in-domain root at step {k} (t={float(t[k])!r}): V_base={V_base!r}, "
                                    f"kappa={kappa!r}, curve domain [{float(turb['speeds'][0])!r}, {float(turb['speeds'][-1])!r}]")
                    if len(roots) > 1:
                        raise _Fail(STATUS_MULTIPLE_ROOTS, f"{len(roots)} roots {roots!r} at step {k} (t={float(t[k])!r}); "
                                    "the step is not uniquely determined (no branch is chosen)")
                    V = roots[0]
                    T = curve_T(V)
                    slope_l[k], slope[k] = curve_slopes(V)
                    neg_flag[k] = min(slope_l[k], slope[k]) < 0.0
                    a_k = a_base + Hb * T
                if not np.all(np.isfinite(a_k)):
                    raise _Fail(STATUS_NONFINITE, f"non-finite acceleration at step {k} (t={float(t[k])!r})")
                a[k] = a_k
                v[k] = v_pred + gamma * dt * a_k
                q[k] = q_pred + beta * dt * dt * a_k
                memory_k = mem_past + 0.5 * dt * (K0 @ v[k])
                record(k, memory_k, T, V)
                rot = float(np.linalg.norm(q[k, 3:]))
                if rot > lim:
                    raise _Fail(STATUS_ROTATION, f"row {k} (t={float(t[k])!r}) rejected and excluded from the history: "
                                f"rotation {rot:.6g} rad exceeds the {lim:.6g} rad envelope; no finite-rotation "
                                "mechanics are supported", row=k, rotation=rot)
                if float(np.max(res[k])) > tol or kin[k] > tol:
                    raise _Fail(STATUS_RESIDUAL, f"row {k} (t={float(t[k])!r}) rejected and excluded from the history: "
                                f"scaled residual {float(np.max(res[k])):.3e} / kinematic {float(kin[k]):.3e} "
                                f"exceed {tol:.1e}", row=k, rotation=rot)
                native_guard(k)
                n_acc = k + 1                      # only a fully admitted row becomes part of the history
                max_rot = max(max_rot, rot)
                if progress_callback is not None:
                    progress_callback(k + 1, n)
                if _poll(cancellation_token):
                    cancelled = True
                    break
    except _Fail as exc:
        status, message = exc.status, str(exc)
        failed = n_acc if exc.row is None else int(exc.row)     # original input index; never an accepted row
        if exc.row is not None:
            k = int(exc.row)
            failed_diag = dict(row=k, time=float(t[k]), rotation=exc.rotation, max_scaled_residual=float(np.max(res[k])),
                               kinematic_residual=float(kin[k]), force_scale=float(fscale[k]), moment_scale=float(mscale[k]))
            if native_failure is not None:
                failed_diag["native_hydrodynamics"] = native_failure
    if cancelled and status == STATUS_COMPLETE:
        status, message = STATUS_CANCELLED, f"cancelled after {n_acc} accepted steps"

    m = n_acc
    sl = slice(0, m)
    neg = int(np.count_nonzero(neg_flag[sl])) if turb is not None else 0
    kx_skew, kx_sym = _skew_report(d["Kx"])
    cx_skew, _ = _skew_report(d["Cx"])
    return FloatingTransientResult(
        status=status, message=message, complete=(status == STATUS_COMPLETE and m == n), cancelled=cancelled,
        failed_step=failed, times=_ro(t[sl]), dt=dt, n_steps=n, n_accepted=m,
        displacements=_ro(q[sl]), velocities=_ro(v[sl]), accelerations=_ro(a[sl]),
        load_components=_mapping({k: x[sl] for k, x in loads.items()}),
        dry_inertia_components=_mapping({k: x[sl] for k, x in dry.items()}),
        added_inertia_forces=_ro(ainf_f[sl]), memory_forces=_ro(mem[sl]), damping_forces=_ro(dmp[sl]),
        stiffness_forces=_ro(stf[sl]), turbine_forces=_ro(tf[sl]), thrust=_ro(thrust[sl]),
        relative_axial_speed=_ro(vrel[sl]), hub_wind_world=_ro(hub_wind[sl]),
        residual_forces=_ro(res[sl]), kinematic_residual=_ro(kin[sl]), root_candidates=_ro(cand[sl]),
        aero_damping_slope=_ro(slope[sl]), negative_damping_steps=neg, max_rotation=max_rot,
        force_scale=float(np.max(fscale[sl])) if m else 0.0, moment_scale=float(np.max(mscale[sl])) if m else 0.0,
        max_scaled_residual=float(np.max(res[sl])) if m else 0.0,
        external_stiffness_skew_norm=kx_skew, external_stiffness_symmetric_norm=kx_sym,
        external_damping_skew_norm=cx_skew, effective_matrix_condition=max_cond,
        identity=_deep_freeze(dict(spec.identity or {})),
        frame=FRAME_BODY_FIRST_ORDER, kernel_digest=numeric_array_digest(kernel), ainf_digest=numeric_array_digest(d["ainf"]),
        aero_damping_slope_left=_ro(slope_l[sl]), aero_damping_slope_right=_ro(slope[sl]),
        force_scale_used=_ro(fscale[sl]), moment_scale_used=_ro(mscale[sl]), kinematic_scale_used=_ro(kscale[sl]),
        failed_row_diagnostics=None if failed_diag is None else _deep_freeze(failed_diag),
        mean_reference=None if d["mean_reference"] is None else _deep_freeze(d["mean_reference"]),
        hydrodynamic_admission_report=None if native_report is None else _deep_freeze(native_report),
        hydrodynamic_work=None if native_work is None else _mapping({key: value[sl] for key, value in native_work.items()}),
    )


class _Fail(Exception):
    def __init__(self, status: str, message: str, row: Optional[int] = None, rotation: float = 0.0):
        super().__init__(message)
        self.status = status
        self.row = row
        self.rotation = rotation


def _check_token(token: Any) -> None:
    """A cancellation token must be None, an object with ``is_cancelled`` (bool or callable) or a callable."""
    # attribute presence is tested without evaluating a property (an evaluation would count as a poll)
    declared = hasattr(type(token), "is_cancelled") or "is_cancelled" in getattr(token, "__dict__", {})
    if token is None or declared or callable(token):
        return
    raise FloatingTransientError(
        "cancellation_token must be None, an object with an is_cancelled attribute or a callable returning a "
        f"truthy value; received {type(token).__name__}")


def _poll(token: Any) -> bool:
    """Poll with the marine semantics but surface only FloatingTransientError."""
    try:
        return _cancel_requested(token)
    except ValueError as exc:   # MarineLinearInputError is a ValueError
        raise FloatingTransientError(f"cancellation_token: {exc}") from exc


def _mapping(d: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    from types import MappingProxyType
    return MappingProxyType({k: _ro(v) for k, v in d.items()})


def waterline_guard(
    displacements: Any, points_body: Any, elevation: Any, limit: Any,
) -> dict:
    """Relative-waterline excursion on EVERY integration sample (independent of output decimation).

    ``displacements`` (n, 6) rigid q; ``points_body`` (n_q, 3) waterline query points relative to the
    CG; ``elevation`` (n, n_q) total free-surface elevation (incident + diffraction + radiation) at the
    points; ``limit`` scalar or (n_q,) allowed relative excursion (draft/freeboard).  The relative
    elevation is ``eta - w_z`` with first-order rigid vertical displacement ``w_z = q_z + (theta x r)_z``.
    """
    q = _arr("displacements", displacements, ndim=2)
    pts = _arr("points_body", points_body, ndim=2)
    eta = _arr("elevation", elevation, shape=(q.shape[0], pts.shape[0]))
    lim = np.broadcast_to(_arr("limit", limit), (pts.shape[0],))
    if q.shape[1] != 6 or pts.shape[1] != 3:
        raise FloatingTransientError("displacements must be (n, 6) and points_body (n_q, 3)")
    if q.shape[0] < 1 or not np.all(lim > 0.0):
        raise FloatingTransientError("at least one sample and strictly positive excursion limits are required")
    wz = q[:, 2:3] + (q[:, 3:4] * pts[None, :, 1] - q[:, 4:5] * pts[None, :, 0])   # (theta x r)_z
    rel = eta - wz
    excess = np.abs(rel) - lim[None, :]
    return {
        "max_relative_excursion": float(np.max(np.abs(rel))),
        "max_ratio_to_limit": float(np.max(np.abs(rel) / lim[None, :])),
        "violated": bool(np.any(excess > 0.0)),
        "first_violation_index": int(np.argmax(np.any(excess > 0.0, axis=1))) if np.any(excess > 0.0) else None,
        "n_samples_checked": int(q.shape[0]),
    }
