"""D2-1 (revision 4, "B2 with cross-check"): shell result-axes outputs.

``anyfesolver.shell_axes_outputs`` derives station positions, tensor-basis
normals, director metadata and (where the element emits none) the three global
family tensors OUTSIDE the pinned element sources, and cross-checks its own
recomputed frames against quantities the solver emitted (design D2 section 1,
rules (1)-(5)).

Checked for every recovery branch (legacy T3/Q4/Q4w/T6/T6c/Q8/Q8c, e4-pl Q4
planar/warped with and without ``reference_normal``, qualified S3, S3 V2D;
homogeneous and generalized), with rigidly rotated fixtures and reversed /
shifted node order for the three- and four-node elements:

(a) array shapes; (b) the tensors re-expressed in the element's native frame
reproduce the native local N, M, Q to 1e-13 of the family max (mixed-magnitude
fixtures: surface-only branches refuse a family the surfaces cannot resolve,
direct-reference branches certify every family); (c) station positions equal
shape-function interpolation; (d) normals are unit, annihilate the membrane and
bending tensors, flat-element normals; (e) the input dict and the element
outputs are unchanged by the call; (f) the design regression fixtures (a)-(e)
with a wrong frame injected through the module's private ``_frame_override``
hook; (g) director polarity -1 conventions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anyfesolver.e4_pl_element import equation7_frame
from anyfesolver.e4_pl_s3_element import (
    TRIANGLE_QUADRATURE,
    QualifiedE4PLS3ShellElement,
    triangle_frame,
)
from anyfesolver.elements import ShellElement, create_shell_element
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver.shell_axes_outputs import (
    AXES_FRAME_RULE_MISMATCH,
    FAMILIES,
    ShellAxesFrameRuleMismatch,
    ShellAxesOutputs,
    ShellAxesRecovery,
    ShellAxesRefusal,
    _derive_shell_axes_from_recovery,
    recover_shell_axes,
)
from anyfesolver.shell_sections import GeneralizedShellSection

EPS = 2.0**-52
H = 0.1
E_MOD, NU = 210.0e9, 0.3
MATERIAL = Material("steel", E_MOD, NU, density=7850.0)

_axis = np.array([1.0, 2.0, -0.5])
_axis /= np.linalg.norm(_axis)
_ang = 0.7
_K = np.array(
    [
        [0, -_axis[2], _axis[1]],
        [_axis[2], 0, -_axis[0]],
        [-_axis[1], _axis[0], 0],
    ]
)
R0 = np.eye(3) + math.sin(_ang) * _K + (1 - math.cos(_ang)) * (_K @ _K)
T0 = np.array([0.5, -0.3, 0.2])
N0 = R0[:, 2]

FAMILY_KEYS = {
    "membrane": "global_membrane_resultant_tensors",
    "bending": "global_bending_resultant_tensors",
    "shear": "global_transverse_shear_resultants",
}
NORMAL_KEYS = {
    "membrane": "membrane_basis_normals",
    "bending": "bending_basis_normals",
    "shear": "shear_basis_normals",
}
SURFACE_KEYS = tuple(
    f"global_{c}_{s}"
    for s in ("top", "bot")
    for c in ("xx", "yy", "zz", "xy", "yz", "xz")
)


def _section() -> GeneralizedShellSection:
    a = E_MOD * H / (1 - NU**2)
    d = E_MOD * H**3 / (12 * (1 - NU**2))
    g = E_MOD / (2 * (1 + NU))
    in_plane = np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
    return GeneralizedShellSection(
        A=a * in_plane,
        B=np.zeros((3, 3)),
        D=d * in_plane,
        As=(5 / 6) * g * H * np.eye(2),
        name="iso",
    )


def _mid(p, q):
    return tuple((np.array(p) + np.array(q)) / 2)


def topology(name: str):
    if name == "t3":
        return [(0, 0, 0), (1.2, 0, 0), (0.3, 0.9, 0)]
    if name == "q4":
        return [(0, 0, 0), (1.2, 0, 0), (1.2, 0.8, 0), (0, 0.8, 0)]
    if name == "q4w":
        return [(0, 0, 0), (1.2, 0, 0), (1.2, 0.8, 0.15), (0, 0.8, 0)]
    corners = (
        [(0, 0, 0), (1.2, 0, 0), (0.3, 0.9, 0)]
        if name.startswith("t6")
        else [(0, 0, 0), (1.2, 0, 0), (1.2, 0.8, 0), (0, 0.8, 0)]
    )
    pts = list(corners)
    for i in range(len(corners)):
        pts.append(_mid(corners[i], corners[(i + 1) % len(corners)]))
    if name.endswith("c"):
        pts = [(x, y, 0.3 * y * y) for x, y, _ in pts]
    return pts


def displacement(loc, bend=1.0, mem=1.0, shear=1.0) -> np.ndarray:
    """Generic mixed field (anisotropic, with twist and transverse shear)."""

    k, k2, e, g, s = 0.02, 0.013, 1.0e-3, 2.0e-3, 1.5e-3
    out = []
    for (x, y, _z) in loc:
        u = np.zeros(3)
        th = np.zeros(3)
        u[2] += bend * (-k * x * x / 2 - k2 * y * y / 2)
        th[1] += bend * (k * x + 0.3 * k * y)
        th[0] += bend * (-k2 * y - 0.2 * k * x)
        u[0] += mem * (e * x + s * y)
        u[1] += mem * (0.4 * e * y + s * x)
        u[2] += shear * (g * x + 0.5 * g * y)
        out.append(np.concatenate((R0 @ u, R0 @ th)))
    return np.concatenate(out)


def plate_field(loc, *, kx=0.0, ky=0.0, ex=0.0, ey=0.0, gx=0.0) -> np.ndarray:
    """Pure curvature (kx, ky), membrane strain (ex, ey), shear gx in loc axes.

    w = -(kx x^2 + ky y^2)/2 + gx x with theta_x = dw/dy, theta_y = -dw/dx
    of the bending part only, so the transverse shear strain is gx.
    """

    out = []
    for (x, y, _z) in loc:
        u = np.array((ex * x, ey * y, -(kx * x * x + ky * y * y) / 2 + gx * x))
        th = np.array((-ky * y, kx * x, 0.0))
        out.append(np.concatenate((R0 @ u, R0 @ th)))
    return np.concatenate(out)


ORDERS = {
    3: {"std": [0, 1, 2], "rev": [0, 2, 1], "shift": [1, 2, 0]},
    4: {"std": [0, 1, 2, 3], "rev": [0, 3, 2, 1], "shift": [2, 3, 0, 1]},
}


def build(kind, topo, sec, *, polarity=1, order="std", ref=True, normal=None, thickness=H):
    loc = topology(topo)
    n = len(loc)
    if n in ORDERS:
        idx = ORDERS[n][order]
    else:
        assert order == "std"
        idx = list(range(n))
    loc = [loc[i] for i in idx]
    mesh = FEMesh()
    for i, p in enumerate(loc, start=1):
        mesh.add_node(i, *(R0 @ np.asarray(p, float) + T0))
    kw = dict(thickness=thickness)
    if sec == "g":
        kw["shell_section"] = _section()
    ids = list(range(1, n + 1))
    nrm = [float(v) for v in (N0 if normal is None else normal)]
    if kind == "legacy":
        el = ShellElement(1, ids, "steel", **kw)
    elif kind == "s3":
        kw.update(reference_normal=nrm, director_polarity=polarity)
        el = QualifiedE4PLS3ShellElement(1, ids, "steel", **kw)
    else:
        if ref:
            kw.update(reference_normal=nrm, director_polarity=polarity)
        form = {"e4pl": "e4-pl", "v2d": "e4-pl-s3-v2d"}[kind]
        el = create_shell_element(1, ids, "steel", formulation=form, **kw)
    return el, mesh, loc


def recover(el, mesh, loc, flag=True, *, field=None, **scales):
    u = displacement(loc, **scales) if field is None else field
    return el.compute_stresses(mesh, u, MATERIAL, return_global=flag), u


def derive(el, mesh, u, res, **kw) -> ShellAxesOutputs:
    # the private core on a given recovery (the public B3 route supplies its
    # own; test_b3_public_route_* shows the two agree bit for bit)
    return _derive_shell_axes_from_recovery(el, mesh, u, MATERIAL, res, **kw)


# --------------------------------------------------------------------- cases
# (id, kind, topo, sec, expected n_p, ref)
BRANCHES = []
for _topo, _label, _np in (
    ("t3", "T3", 3),
    ("q4", "Q4", 4),
    ("q4w", "Q4w", 4),
    ("t6", "T6", 7),
    ("t6c", "T6c", 7),
    ("q8", "Q8", 9),
    ("q8c", "Q8c", 9),
):
    for _sec in ("h", "g"):
        BRANCHES.append((f"legacy-{_label}-{_sec}", "legacy", _topo, _sec, _np, True))
for _topo, _label in (("q4", "Q4flat"), ("q4w", "Q4warp")):
    for _sec in ("h", "g"):
        BRANCHES.append((f"e4pl-{_label}-{_sec}", "e4pl", _topo, _sec, 4, True))
        BRANCHES.append((f"e4pl-{_label}-{_sec}-noref", "e4pl", _topo, _sec, 4, False))
for _sec in ("h", "g"):
    BRANCHES.append((f"s3-{_sec}", "s3", "t3", _sec, 7, True))
    BRANCHES.append((f"v2d-{_sec}", "v2d", "t3", _sec, 3, True))
IDS = [b[0] for b in BRANCHES]

ORDER_CASES = [
    (bid + f"-{order}", kind, topo, sec, n_p, ref, order)
    for (bid, kind, topo, sec, n_p, ref) in BRANCHES
    if topo in ("q4", "q4w", "t3") and kind in ("legacy", "e4pl", "v2d")
    for order in ("std", "rev", "shift")
] + [
    (bid + "-std", kind, topo, sec, n_p, ref, "std")
    for (bid, kind, topo, sec, n_p, ref) in BRANCHES
    if not (topo in ("q4", "q4w", "t3") and kind in ("legacy", "e4pl", "v2d"))
]
ORDER_IDS = [c[0] for c in ORDER_CASES]


def _build_case(case, **kw):
    _bid, kind, topo, sec, _n_p, ref, order = case
    return build(kind, topo, sec, order=order, ref=ref, **kw)


def _is_surface_only(el, kind, topo, sec) -> bool:
    """Homogeneous branches that emit only the surface tensors (F2)."""

    if sec == "g":
        return False
    if kind == "legacy":
        return True
    if kind == "e4pl":
        return topo == "q4w"  # warped homogeneous; with or without reference
    return False


def _s3_polarity(el, kind) -> float:
    return float(el.director_polarity) if kind == "s3" else 1.0


# ------------------------------------------------- independent frame access
def _legacy_station_frames(el, coords):
    out = []
    for xi, eta in el.gauss_points:
        _N, dxi, deta = el.compute_shape_functions(float(xi), float(eta))
        out.append(el._local_frame_and_derivatives(coords, dxi, deta)[0])
    return np.asarray(out)


def _frames(el, mesh, res, kind, topo, n_p):
    """(membrane, bending, shear) native frames, each (n_p,3,3)."""

    coords = el.get_node_coordinates(mesh)
    if kind == "legacy" or (kind == "e4pl" and topo == "q4w" and el.reference_normal is None):
        out = _legacy_station_frames(el, coords)
        return out, out, out
    if kind == "e4pl":
        if "natural_point_coordinates" in res:
            frame = el._physical_director_context(equation7_frame(coords)[0])[0]
            rep = np.repeat(frame[None], n_p, axis=0)
            return rep, rep, rep
        centre = el._physical_director_context(el._center_frame(coords))[0]
        stations = np.asarray(
            [el._physical_director_context(f)[0] for f in _legacy_station_frames(el, coords)]
        )
        return stations, stations, np.repeat(centre[None], n_p, axis=0)
    if kind == "s3":
        frame = triangle_frame(coords, el.reference_normal)[0]
    else:
        frame = np.asarray(res["frame"], float)
    rep = np.repeat(frame[None], n_p, axis=0)
    return rep, rep, rep


def _native(res, el, sec):
    """Native local (N, M, Q) per station: (11,22,12), (11,22,12), (13,23)."""

    if sec == "g" or "membrane_resultants" in res:
        return (
            np.asarray(res["membrane_resultants"], float),
            np.asarray(res["bending_resultants"], float),
            np.asarray(res["transverse_shear_resultants"], float),
        )
    h = float(el.thickness)
    membrane = np.column_stack((res["membrane_xx"], res["membrane_yy"], res["membrane_xy"]))
    bending = np.column_stack((res["bending_xx"], res["bending_yy"], res["bending_xy"]))
    shear = np.column_stack((res["shear_xz"], res["shear_yz"]))
    return h * membrane, bending * max(h * h, 1.0e-12) / 6.0, h * shear


def _tensor(v):
    return np.array([[v[0], v[2], 0.0], [v[2], v[1], 0.0], [0.0, 0.0, 0.0]])


def _expected_resolvable(el, res, kind, topo, sec):
    """Rule (3) prediction per family for surface-only branches (stress form)."""

    if not _is_surface_only(el, kind, topo, sec):
        return {f: True for f in FAMILIES}
    s_surf = max(float(np.max(np.abs(res[k]))) for k in SURFACE_KEYS)
    stress = {
        "membrane": np.column_stack((res["membrane_xx"], res["membrane_yy"], res["membrane_xy"])),
        "bending": np.column_stack((res["bending_xx"], res["bending_yy"], res["bending_xy"])),
        "shear": np.column_stack((res["shear_xz"], res["shear_yz"])),
    }
    out = {}
    for fam, values in stress.items():
        s_f = float(np.max(np.abs(values)))
        a_f = 1.0e-13 * s_f + 64.0 * EPS * s_surf
        # margins: predictions within a factor 4 of the threshold are not asserted
        if a_f <= 0.25e-9 * s_f:
            out[fam] = True
        elif a_f >= 4.0e-9 * s_f:
            out[fam] = False
        else:
            out[fam] = None
    return out


# ------------------------------------------------------------------ (a) shapes
@pytest.mark.parametrize("branch", BRANCHES, ids=IDS)
def test_a_array_shapes_and_all_families_certified(branch):
    bid, kind, topo, sec, n_p, ref = branch
    el, mesh, loc = build(kind, topo, sec, ref=ref)
    res, u = recover(el, mesh, loc)
    out = derive(el, mesh, u, res)
    assert isinstance(out, ShellAxesOutputs)
    assert out.refusals == {}, out.refusals
    assert out.certified == {f: True for f in FAMILIES}
    a = out.arrays
    assert a["recovery_station_positions"].shape == (n_p, 3)
    for fam in FAMILIES:
        assert a[NORMAL_KEYS[fam]].shape == (n_p, 3)
    assert a[FAMILY_KEYS["membrane"]].shape == (n_p, 3, 3)
    assert a[FAMILY_KEYS["bending"]].shape == (n_p, 3, 3)
    assert a[FAMILY_KEYS["shear"]].shape == (n_p, 3)
    assert a["shear_normal_components"].shape == (n_p,)
    for key, value in a.items():
        assert isinstance(value, np.ndarray), key
        assert not value.flags.writeable, key
        assert np.all(np.isfinite(value)), key
    out.require_all()  # no refusal: must not raise
    # passed through unchanged where the element already emits them
    if not _is_surface_only(el, kind, topo, sec):
        for fam in FAMILIES:
            assert np.array_equal(a[FAMILY_KEYS[fam]], res[FAMILY_KEYS[fam]])
            assert a[FAMILY_KEYS[fam]] is not res[FAMILY_KEYS[fam]]
    else:
        for fam in FAMILIES:
            assert FAMILY_KEYS[fam] not in res  # derived by the module (F2)


def test_return_global_false_is_a_typed_refusal():
    el, mesh, loc = build("legacy", "q4", "h")
    res, u = recover(el, mesh, loc, False)
    with pytest.raises(ShellAxesRefusal) as info:
        derive(el, mesh, u, res)
    assert isinstance(info.value, ValueError)
    assert info.value.code == "AXES_RETURN_GLOBAL_REQUIRED"
    el, mesh, loc = build("e4pl", "q4", "g")
    res, u = recover(el, mesh, loc, False)
    with pytest.raises(ShellAxesRefusal):
        derive(el, mesh, u, res)


# --------------------------------------------------- (b) native reproduction
SCALES = (
    dict(bend=1.0, mem=1.0, shear=1.0),
    dict(bend=1.0, mem=1.0e-8, shear=1.0e-8),
    dict(bend=1.0e-8, mem=1.0, shear=1.0e-8),
    dict(bend=1.0e-8, mem=1.0e-8, shear=1.0),
)
SCALE_IDS = ("mixed", "membrane-1e-8", "bending-1e-8", "only-shear")


@pytest.mark.parametrize("scales", SCALES, ids=SCALE_IDS)
@pytest.mark.parametrize("case", ORDER_CASES, ids=ORDER_IDS)
def test_b_tensors_reproduce_native_local_values(case, scales):
    _bid, kind, topo, sec, n_p, _ref, _order = case
    el, mesh, loc = _build_case(case)
    res, u = recover(el, mesh, loc, **scales)
    out = derive(el, mesh, u, res)
    expected = _expected_resolvable(el, res, kind, topo, sec)
    fm, fb, fs = _frames(el, mesh, res, kind, topo, n_p)
    nat_m, nat_b, nat_q = _native(res, el, sec)
    p = _s3_polarity(el, kind)
    for fam, frames, native, sign in (
        ("membrane", fm, nat_m, 1.0),
        ("bending", fb, nat_b, p),
        ("shear", fs, nat_q, p),
    ):
        if expected[fam] is None:
            continue
        if not expected[fam]:
            assert not out.certified[fam]
            assert out.refusals[fam].code == AXES_FRAME_RULE_MISMATCH
            assert out.refusals[fam].reason == "unresolvable_family", out.refusals[fam]
            assert FAMILY_KEYS[fam] not in out.arrays
            continue
        assert out.certified[fam], out.refusals.get(fam)
        stored = out.arrays[FAMILY_KEYS[fam]]
        scale = float(np.max(np.abs(native)))
        assert scale > 0.0
        for index in range(n_p):
            frame = frames[index]
            if fam == "shear":
                local = frame.T @ stored[index]
                assert abs(local[2]) <= 1.0e-13 * scale
                assert np.max(np.abs(local[:2] - sign * native[index])) <= 1.0e-13 * scale
            else:
                local = frame.T @ stored[index] @ frame
                assert np.max(np.abs(local - sign * _tensor(native[index]))) <= 1.0e-13 * scale
    # stations and normals have their own checks and survive family refusals
    assert out.arrays["recovery_station_positions"].shape == (n_p, 3)
    for fam in FAMILIES:
        assert NORMAL_KEYS[fam] in out.arrays
    if any(v is False for v in expected.values()):
        with pytest.raises(ShellAxesFrameRuleMismatch) as info:
            out.require_all()
        assert info.value.code == AXES_FRAME_RULE_MISMATCH


@pytest.mark.parametrize(
    "branch",
    [b for b in BRANCHES if b[3] == "h" and b[1] == "legacy" and b[2] in ("t3", "q4", "t6", "q8")],
    ids=lambda b: b[0],
)
def test_b_surface_only_branches_refuse_the_small_family_and_direct_ones_certify(branch):
    bid, kind, topo, sec, n_p, ref = branch
    el, mesh, loc = build(kind, topo, sec, ref=ref)
    res, u = recover(el, mesh, loc, bend=1.0, mem=1.0e-8, shear=1.0e-8)
    out = derive(el, mesh, u, res)
    assert out.refusals["membrane"].reason == "unresolvable_family"
    assert out.certified["bending"]


@pytest.mark.parametrize("branch", [b for b in BRANCHES if not (b[3] == "h" and (b[1] == "legacy" or b[2] == "q4w" and b[1] == "e4pl"))], ids=lambda b: b[0])
def test_b_direct_reference_branches_certify_mixed_magnitudes(branch):
    bid, kind, topo, sec, n_p, ref = branch
    el, mesh, loc = build(kind, topo, sec, ref=ref)
    for scales in SCALES:
        res, u = recover(el, mesh, loc, **scales)
        out = derive(el, mesh, u, res)
        assert out.refusals == {}, (scales, out.refusals)


def test_b_homogeneous_moment_is_native_below_the_legacy_floor():
    """M = sigma_b * max(h^2, 1e-12) / 6 inverts the recovery floor (h^3 scaling)."""

    thin = 1.0e-7
    el, mesh, loc = build("legacy", "q4", "h")
    u = plate_field(loc, kx=K, ky=-K)
    ref = derive(el, mesh, u, el.compute_stresses(mesh, u, MATERIAL, return_global=True))
    el2, mesh2, loc2 = build("legacy", "q4", "h", thickness=thin)
    u2 = plate_field(loc2, kx=K, ky=-K)
    got = derive(el2, mesh2, u2, el2.compute_stresses(mesh2, u2, MATERIAL, return_global=True))
    assert ref.certified["bending"] and got.certified["bending"], got.refusals
    a = ref.arrays[FAMILY_KEYS["bending"]] * (thin / H) ** 3
    b = got.arrays[FAMILY_KEYS["bending"]]
    assert np.max(np.abs(a - b)) <= 1.0e-9 * np.max(np.abs(a))


def test_b_thin_warped_homogeneous_bending_is_an_honest_unresolvable_refusal():
    """At h = 1e-7 the warped membrane coupling dominates the surfaces; the
    bending family (about 1e-8 of the surface scale) cannot be certified."""

    el, mesh, loc = build("e4pl", "q4w", "h", thickness=1.0e-7)
    u = plate_field(loc, kx=K, ky=-K)
    out = derive(el, mesh, u, el.compute_stresses(mesh, u, MATERIAL, return_global=True))
    assert out.refusals["bending"].reason == "unresolvable_family"
    assert out.certified["membrane"]


# ------------------------------------------------------------ (c) positions
def _natural_points(el, res, kind):
    if kind == "e4pl" and "natural_point_coordinates" in res:
        return np.asarray(res["natural_point_coordinates"], float)
    if kind == "s3":
        return np.asarray([(r, s) for r, s, _w in TRIANGLE_QUADRATURE], float)
    return np.asarray(el.gauss_points, float)


def _shape_values(el, kind, point):
    if kind in ("s3", "v2d") or el.num_nodes == 3:
        r, s = point
        return np.array([1.0 - r - s, r, s])
    xi, eta = point
    if el.num_nodes == 4:
        return 0.25 * np.array(
            [
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta),
            ]
        )
    return np.asarray(el.compute_shape_functions(float(xi), float(eta))[0], float)


@pytest.mark.parametrize("case", ORDER_CASES, ids=ORDER_IDS)
def test_c_station_positions_are_shape_function_interpolation(case):
    _bid, kind, topo, sec, n_p, _ref, _order = case
    el, mesh, loc = _build_case(case)
    res, u = recover(el, mesh, loc)
    out = derive(el, mesh, u, res)
    coords = np.asarray(el.get_node_coordinates(mesh), float)
    if kind == "v2d":
        expected = np.asarray(res["external_barycentric_coordinates"], float) @ coords
    else:
        expected = np.asarray(
            [_shape_values(el, kind, p) @ coords for p in _natural_points(el, res, kind)]
        )
    got = out.arrays["recovery_station_positions"]
    scale = float(np.max(np.abs(coords)))
    assert np.max(np.abs(got - expected)) <= 1.0e-14 * scale
    if kind == "v2d":
        assert np.max(np.abs(got - res["physical_station_coordinates"])) <= 1.0e-14 * scale


def test_c_planar_and_warped_q4_station_orders_differ_and_are_followed():
    el, mesh, loc = build("e4pl", "q4", "h")
    res, u = recover(el, mesh, loc)
    planar = derive(el, mesh, u, res).arrays["recovery_station_positions"]
    elw, meshw, locw = build("e4pl", "q4w", "h")
    resw, uw = recover(elw, meshw, locw)
    warped = derive(elw, meshw, uw, resw).arrays["recovery_station_positions"]
    natural = np.asarray(res["natural_point_coordinates"], float)
    gauss = np.asarray(elw.gauss_points, float)
    assert not np.array_equal(np.sign(natural), np.sign(gauss))
    # index 2 is (+,+) on the planar path but (-,+) on the warped path
    assert tuple(np.sign(natural[2])) == (1.0, 1.0)
    assert tuple(np.sign(gauss[2])) == (-1.0, 1.0)
    coords = el.get_node_coordinates(mesh)
    assert np.allclose(planar[2], _shape_values(el, "e4pl", natural[2]) @ coords, rtol=0, atol=1e-15)
    coordsw = elw.get_node_coordinates(meshw)
    assert np.allclose(warped[2], _shape_values(elw, "e4pl", gauss[2]) @ coordsw, rtol=0, atol=1e-15)


# ----------------------------------------------------------------- (d) normals
def _expected_flat_normal(kind, order, ref, polarity=1):
    if kind == "legacy" or (kind == "e4pl" and not ref):
        return N0 if order in ("std", "shift") else -N0
    if kind == "e4pl":
        return polarity * N0
    return N0


@pytest.mark.parametrize("case", ORDER_CASES, ids=ORDER_IDS)
def test_d_normals_unit_annihilate_tensors_and_flat_normal(case):
    _bid, kind, topo, sec, n_p, ref, order = case
    el, mesh, loc = _build_case(case)
    res, u = recover(el, mesh, loc)
    out = derive(el, mesh, u, res)
    a = out.arrays
    for fam in FAMILIES:
        normals = a[NORMAL_KEYS[fam]]
        assert np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0)) <= 64 * EPS
    for fam in ("membrane", "bending"):
        tensors = a[FAMILY_KEYS[fam]]
        normals = a[NORMAL_KEYS[fam]]
        s_t = float(np.max(np.abs(tensors)))
        assert np.max(np.abs(np.einsum("ni,nij->nj", normals, tensors))) <= 1.0e-13 * s_t
    q_n = np.einsum("ni,ni->n", a[NORMAL_KEYS["membrane"]], a[FAMILY_KEYS["shear"]])
    assert np.allclose(a["shear_normal_components"], q_n, rtol=0, atol=1e-6 * np.max(np.abs(a[FAMILY_KEYS["shear"]])))
    flat = topo in ("t3", "q4", "t6", "q8")
    if flat:
        expected = _expected_flat_normal(kind, order, ref)
        for fam in FAMILIES:
            assert np.allclose(a[NORMAL_KEYS[fam]], expected[None, :], rtol=0, atol=1e-14)
        assert np.max(np.abs(a["shear_normal_components"])) <= 1e-13 * np.max(np.abs(a[FAMILY_KEYS["shear"]]))
    if kind == "e4pl" and topo == "q4w" and ref:
        # warped e4-pl: membrane/bending station normals, shear centre normal
        assert np.allclose(a[NORMAL_KEYS["membrane"]], res["physical_directors"], rtol=0, atol=1e-15)
        assert np.allclose(a[NORMAL_KEYS["bending"]], res["physical_directors"], rtol=0, atol=1e-15)
        centre = np.repeat(np.asarray(res["physical_director"])[None], n_p, axis=0)
        assert np.allclose(a[NORMAL_KEYS["shear"]], centre, rtol=0, atol=1e-15)
        # the centre-frame shear is not tangent to the station planes
        assert np.max(np.abs(a["shear_normal_components"])) > 1e-6 * np.max(np.abs(a[FAMILY_KEYS["shear"]]))
    if kind == "v2d":
        for fam in FAMILIES:
            assert np.allclose(a[NORMAL_KEYS[fam]], np.asarray(res["frame"])[:, 2][None], rtol=0, atol=1e-15)


# ---------------------------------------------------------- (e) no side effects
def _snapshot(d):
    out = {}
    for k, v in d.items():
        out[k] = (v, np.array(v, copy=True)) if isinstance(v, np.ndarray) else (v, v)
    return out


def _same(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("branch", BRANCHES, ids=IDS)
def test_e_input_dict_and_element_outputs_unchanged(branch):
    bid, kind, topo, sec, n_p, ref = branch
    el, mesh, loc = build(kind, topo, sec, ref=ref)
    res, u = recover(el, mesh, loc)
    before = _snapshot(res)
    u_before = u.copy()
    derive(el, mesh, u, res)
    assert set(res) == set(before)
    for key, (obj, copy) in before.items():
        assert res[key] is obj, key
        assert _same(res[key], copy), key
    assert np.array_equal(u, u_before)
    again, _ = recover(el, mesh, loc)
    assert set(again) == set(before)
    for key, (_obj, copy) in before.items():
        assert _same(again[key], copy), key


# ----------------------------------------------- (f) design regression fixtures
def _rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _override(local_rotation, roles=("station", "centre")):
    def hook(role, frames):
        if role not in roles:
            return frames
        return np.asarray(frames) @ local_rotation

    return hook


K = 0.02


def test_f_a_isotropic_bending_with_tiny_membrane_refuses_membrane_by_design():
    el, mesh, loc = build("legacy", "q4", "h")
    e = 1.0e-8 * H * K / 2  # isotropic membrane stress 1e-8 of the bending stress
    u = plate_field(loc, kx=K, ky=K, ex=e, ey=e)
    res = el.compute_stresses(mesh, u, MATERIAL, return_global=True)
    ratio = abs(res["membrane_xx"][0] / res["bending_xx"][0])
    assert 1e-9 < ratio < 1e-7
    for hook in (None, _override(_rz(1.0e-6))):
        out = derive(el, mesh, u, res, _frame_override=hook)
        assert out.refusals["membrane"].reason == "unresolvable_family"
        assert FAMILY_KEYS["membrane"] not in out.arrays
        # isotropic bending is invariant under a rotation about the normal
        assert out.certified["bending"], out.refusals.get("bending")


@pytest.mark.parametrize("branch", ["legacy-Q4-h", "legacy-T3-h", "e4pl-Q4flat-h", "legacy-Q4-g"])
def test_f_b_anisotropic_families_certified_and_wrong_frame_refused(branch):
    kind, topo, sec = {
        "legacy-Q4-h": ("legacy", "q4", "h"),
        "legacy-T3-h": ("legacy", "t3", "h"),
        "e4pl-Q4flat-h": ("e4pl", "q4", "h"),
        "legacy-Q4-g": ("legacy", "q4", "g"),
    }[branch]
    el, mesh, loc = build(kind, topo, sec)
    e = 1.0e-3 * H * K / 2  # anisotropic membrane 1e-3 of the anisotropic bending
    u = plate_field(loc, kx=K, ky=-K, ex=e, ey=-e)
    res = el.compute_stresses(mesh, u, MATERIAL, return_global=True)
    honest = derive(el, mesh, u, res)
    assert honest.certified["membrane"], honest.refusals
    assert honest.certified["bending"], honest.refusals
    wrong = derive(el, mesh, u, res, _frame_override=_override(_rz(1.0e-6)))
    for fam in ("membrane", "bending"):
        assert wrong.refusals[fam].reason == "residual", wrong.refusals.get(fam)
        assert FAMILY_KEYS[fam] not in wrong.arrays
    with pytest.raises(ShellAxesFrameRuleMismatch):
        wrong.require_all()


def test_f_c_tiny_shear_under_a_pi_flip_is_unresolvable():
    el, mesh, loc = build("legacy", "q4", "h")
    u = plate_field(loc, kx=K, ky=-K, gx=1.0e-14 * H * K * 6.0 / 5.0)
    res = el.compute_stresses(mesh, u, MATERIAL, return_global=True)
    honest = derive(el, mesh, u, res)
    flipped = derive(el, mesh, u, res, _frame_override=_override(np.diag((-1.0, -1.0, 1.0))))
    for out in (honest, flipped):
        assert out.refusals["shear"].reason == "unresolvable_family"
        assert out.certified["bending"], out.refusals.get("bending")


@pytest.mark.parametrize("branch", ["legacy-Q4-h", "legacy-T6-h"])
def test_f_e_out_of_plane_tilt_is_refused_by_rule_5(branch):
    topo = {"legacy-Q4-h": "q4", "legacy-T6-h": "t6"}[branch]
    el, mesh, loc = build("legacy", topo, "h")
    u = plate_field(loc, kx=K, ky=-K)  # surfaces +-diag(1, -1, 0) up to roundoff
    res = el.compute_stresses(mesh, u, MATERIAL, return_global=True)
    honest = derive(el, mesh, u, res)
    assert honest.certified["bending"], honest.refusals
    tilted = derive(el, mesh, u, res, _frame_override=_override(_ry(1.0e-7)))
    diag = tilted.diagnostics["bending"]
    assert diag["residual"] <= diag["allowance"]  # in-plane family residual passes
    assert tilted.refusals["bending"].reason == "out_of_plane", tilted.refusals
    assert FAMILY_KEYS["bending"] not in tilted.arrays


@pytest.mark.parametrize(
    "branch",
    ["legacy-Q4-g", "legacy-Q4w-g", "e4pl-Q4flat-g", "e4pl-Q4warp-g", "s3-g", "v2d-g", "legacy-Q8c-g"],
)
def test_f_d_generalized_wrong_frame_of_1e12_is_detected(branch):
    b = dict((x[0], x) for x in BRANCHES)[branch]
    _bid, kind, topo, sec, _n_p, ref = b
    el, mesh, loc = build(kind, topo, sec, ref=ref)
    res, u = recover(el, mesh, loc)
    assert derive(el, mesh, u, res).refusals == {}
    wrong = derive(el, mesh, u, res, _frame_override=_override(_rz(1.0e-12)))
    for fam in FAMILIES:
        assert wrong.refusals[fam].reason == "residual", (fam, wrong.refusals.get(fam))
        d = wrong.diagnostics[fam]
        assert d["allowance"] == 1.0e-13 * d["scale"]
        assert d["residual"] > d["allowance"]


def test_f_warped_centre_frame_is_checked_through_the_shear_family_only():
    el, mesh, loc = build("e4pl", "q4w", "h")
    res, u = recover(el, mesh, loc)
    assert derive(el, mesh, u, res).refusals == {}
    out = derive(el, mesh, u, res, _frame_override=_override(_rz(1.0e-6), roles=("centre",)))
    # rule (1): only the shear family sees the centre frame ...
    assert out.diagnostics["membrane"]["residual"] <= out.diagnostics["membrane"]["allowance"]
    assert out.diagnostics["bending"]["residual"] <= out.diagnostics["bending"]["allowance"]
    assert out.refusals["shear"].reason == "residual"
    # ... but the joint rule (5) surface rebuild contains that shear, and a
    # rule (5) failure refuses every family of the element (fail closed).
    assert out.refusals["membrane"].reason == "out_of_plane"
    assert out.refusals["bending"].reason == "out_of_plane"
    out = derive(el, mesh, u, res, _frame_override=_override(_rz(1.0e-6), roles=("station",)))
    assert out.refusals["membrane"].reason == "residual"
    assert out.refusals["bending"].reason == "residual"


def test_f_nonfinite_and_missing_reference_are_refused():
    el, mesh, loc = build("legacy", "q4", "g")
    res, u = recover(el, mesh, loc)
    broken = dict(res)
    bad = np.array(res[FAMILY_KEYS["bending"]], copy=True)
    bad[0, 0, 0] = np.nan
    broken[FAMILY_KEYS["bending"]] = bad
    del broken[FAMILY_KEYS["shear"]]
    out = derive(el, mesh, u, broken)
    assert out.certified["membrane"]
    assert out.refusals["bending"].reason == "nonfinite"
    assert out.refusals["shear"].reason == "no_reference"
    assert set(FAMILY_KEYS[f] for f in ("bending", "shear")).isdisjoint(out.arrays)
    elh, meshh, loch = build("legacy", "q4", "h")
    resh, uh = recover(elh, meshh, loch)
    brokenh = dict(resh)
    surf = np.array(resh["global_xz_top"], copy=True)
    surf[1] = np.inf
    brokenh["global_xz_top"] = surf
    outh = derive(elh, meshh, uh, brokenh)
    assert {r.reason for r in outh.refusals.values()} == {"nonfinite"}
    assert set(outh.refusals) == set(FAMILIES)


# ------------------------------------------------------------- (g) polarity
@pytest.mark.parametrize("sec", ("h", "g"))
@pytest.mark.parametrize("kind,topo", (("e4pl", "q4"), ("e4pl", "q4w"), ("s3", "t3"), ("v2d", "t3")))
def test_g_negative_polarity_conventions(kind, topo, sec):
    el, mesh, loc = build(kind, topo, sec, polarity=-1)
    res, u = recover(el, mesh, loc)
    out = derive(el, mesh, u, res)
    assert out.refusals == {}, out.refusals
    a = out.arrays
    assert a["director_polarity"].shape == () and int(a["director_polarity"]) == -1
    assert np.array_equal(a["physical_reference_director"], el.physical_reference_director)
    assert np.allclose(a["physical_reference_director"], -N0, rtol=0, atol=1e-15)
    if kind == "e4pl":
        assert out.conventions["bending"] == "physical_director"
        assert out.conventions["shear"] == "physical_director"
        normal = -N0 if topo == "q4" else None
    elif kind == "s3":
        assert out.conventions["bending"] == "owner_normal"
        assert out.conventions["shear"] == "owner_normal"
        normal = N0
    else:
        assert out.conventions["bending"] == "physical_director"
        normal = N0
    if normal is not None:
        for fam in FAMILIES:
            assert np.allclose(a[NORMAL_KEYS[fam]], normal[None], rtol=0, atol=1e-14)
    n_p = a["recovery_station_positions"].shape[0]
    fm, fb, fs = _frames(el, mesh, res, kind, topo, n_p)
    nat_m, nat_b, nat_q = _native(res, el, sec)
    sign = -1.0 if kind == "s3" else 1.0  # S3 global M, Q are owner convention
    scale = float(np.max(np.abs(nat_b)))
    for i in range(n_p):
        got = fb[i].T @ a[FAMILY_KEYS["bending"]][i] @ fb[i]
        assert np.max(np.abs(got - sign * _tensor(nat_b[i]))) <= 1e-13 * scale
    if kind in ("s3", "v2d"):
        for key in (
            "reference_surface_offset",
            "section_origin_offset_from_reference",
            "physical_bottom_offset_from_reference",
            "physical_top_offset_from_reference",
        ):
            assert a[key].shape == ()


@pytest.mark.parametrize("sec", ("h", "g"))
def test_g_v2d_ordinary_case_gets_polarity_and_offsets_from_the_element(sec):
    el, mesh, loc = build("v2d", "t3", sec)
    res, u = recover(el, mesh, loc)
    if sec == "h":
        assert "director_polarity" not in res  # V2C shortcut (D2-0 item 1)
    a = derive(el, mesh, u, res).arrays
    assert int(a["director_polarity"]) == 1
    assert float(a["reference_surface_offset"]) == 0.0
    assert float(a["section_origin_offset_from_reference"]) == el.material_mid_surface_offset_from_reference
    assert float(a["physical_top_offset_from_reference"]) == el.material_surface_offset_from_reference(1.0)
    assert float(a["physical_bottom_offset_from_reference"]) == el.material_surface_offset_from_reference(-1.0)
    assert np.array_equal(a["physical_reference_director"], el.physical_reference_director)


def test_g_legacy_and_unreferenced_e4pl_carry_no_director():
    for kind, topo, ref in (("legacy", "q4", True), ("e4pl", "q4", False), ("e4pl", "q4w", False)):
        el, mesh, loc = build(kind, topo, "h", ref=ref)
        res, u = recover(el, mesh, loc)
        out = derive(el, mesh, u, res)
        assert "director_polarity" not in out.arrays
        assert "physical_reference_director" not in out.arrays
        assert out.conventions["bending"] == "numbered_normal"


@pytest.mark.parametrize("topo", ("q8", "q8c"))
def test_legacy_q8_batched_recovery_frames_path_is_certified(topo):
    """ANYfem recovers with the exact anymaterial isotropic material, which
    switches legacy Q8 to ``native_recovery_frames``; the module follows it."""

    from anymaterial import IsotropicMaterial

    from anyfesolver.recovery_shell_frames import native_recovery_frames

    material = IsotropicMaterial("steel", E_MOD, NU, 7850.0)
    el, mesh, loc = build("legacy", topo, "h")
    assert native_recovery_frames(el, el.get_node_coordinates(mesh), el.gauss_points, material) is not None
    u = displacement(loc)
    res = el.compute_stresses(mesh, u, material, return_global=True)
    out = recover_shell_axes(el, mesh, u, material).axes
    assert out.refusals == {}, out.refusals


def test_nonfinite_derived_resultant_is_refused_not_certified():
    """Codex r1 P2: N = h sigma overflows while the stresses are finite."""

    el, mesh, loc = build("legacy", "q4", "h", thickness=1000.0)
    res, u = recover(el, mesh, loc, field=plate_field(loc, ex=5.0e294))
    out = derive(el, mesh, u, res)
    refused = [f for f, r in out.refusals.items() if r.reason == "nonfinite"]
    assert refused, out.refusals
    for family in refused:
        assert not out.certified[family]
    for key in ("global_membrane_resultant_tensors", "global_bending_resultant_tensors"):
        if key in out.arrays:
            assert np.all(np.isfinite(out.arrays[key]))


def test_nonfinite_positions_are_refused_not_silently_omitted(monkeypatch):
    """Codex r2: a nonfinite station table must not leave every family certified."""

    import anyfesolver.shell_axes_outputs as module

    original = module._shape_positions

    def poisoned(element, coords, points):
        out = np.array(original(element, coords, points))
        out[0, 0] = float("nan")
        return out

    monkeypatch.setattr(module, "_shape_positions", poisoned)
    el, mesh, loc = build("e4pl", "q4", "h")
    res, u = recover(el, mesh, loc)
    out = derive(el, mesh, u, res)
    assert out.refusals["positions"].reason == "nonfinite"
    assert "recovery_station_positions" not in out.arrays
    with pytest.raises(ShellAxesRefusal):
        out.require_all()


def test_nonfinite_metadata_is_refused_not_copied():
    """Codex r2: finite thickness and offset can still give an infinite top offset."""

    el, mesh, loc = build("v2d", "t3", "g")
    res, u = recover(el, mesh, loc)
    el.thickness = 1.7e308
    el.reference_surface_offset = 1.7e308
    out = derive(el, mesh, u, res)
    assert out.refusals["metadata"].reason == "nonfinite"
    assert "physical_bottom_offset_from_reference" not in out.arrays


# ------------------------------------------- Codex r3 (2026-11-26) regressions
def _q4_about_z(angle, sec="h"):
    """Flat legacy Q4 in the global XY plane, rotated about Z (no R0)."""

    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    mesh = FEMesh()
    for i, p in enumerate(topology("q4"), start=1):
        mesh.add_node(i, *(rot @ np.asarray(p, float)))
    kw = dict(thickness=H)
    if sec == "g":
        kw["shell_section"] = _section()
    return ShellElement(1, [1, 2, 3, 4], "steel", **kw), mesh


def test_r3_custom_legacy_stations_are_bound_to_the_request():
    """P1: reversed natural_points with an equal count must not reuse the default order."""

    el, mesh, loc = build("legacy", "q4", "h")
    u = displacement(loc)
    points = np.asarray(el.gauss_points, dtype=float)[::-1].copy()
    res = el.compute_stresses(mesh, u, MATERIAL, return_global=True, natural_points=points)
    out = derive(el, mesh, u, res, natural_points=points)
    coords = el.get_node_coordinates(mesh)
    expected = np.asarray([el.compute_shape_functions(float(x), float(y))[0] @ coords for x, y in points])
    assert np.array_equal(out.arrays["recovery_station_positions"], expected)
    # a generalized section, a warped Q4, S3 and V2D take no custom stations
    for kind, topo, sec in (("legacy", "q4", "g"), ("e4pl", "q4w", "h"), ("s3", "t3", "h"), ("v2d", "t3", "h")):
        el2, mesh2, loc2 = build(kind, topo, sec)
        res2, u2 = recover(el2, mesh2, loc2)
        with pytest.raises(ShellAxesRefusal) as info:
            derive(el2, mesh2, u2, res2, natural_points=points)
        assert info.value.code == "AXES_STATIONS_NOT_QUALIFIED"


@pytest.mark.parametrize("sec", ("h", "g"))
def test_r3_tampered_station_tables_are_refused(sec):
    """P1: a reordered emitted station table must not relocate valid tensors."""

    el, mesh, loc = build("e4pl", "q4", sec)
    res, u = recover(el, mesh, loc)
    broken = dict(res)
    broken["natural_point_coordinates"] = np.asarray(res["natural_point_coordinates"])[::-1].copy()
    with pytest.raises(ShellAxesRefusal) as info:
        derive(el, mesh, u, broken)
    assert info.value.code == "AXES_STATIONS_NOT_QUALIFIED"
    if sec == "h":
        points = np.asarray(res["natural_point_coordinates"])[::-1].copy()
        custom = el.compute_stresses(mesh, u, MATERIAL, return_global=True, natural_points=points)
        with pytest.raises(ShellAxesRefusal):
            derive(el, mesh, u, custom)  # request not passed: table != default rule
        out = derive(el, mesh, u, custom, natural_points=points)
        assert out.refusals == {}, out.refusals
    el, mesh, loc = build("v2d", "t3", sec)
    res, u = recover(el, mesh, loc)
    for key in ("external_barycentric_coordinates", "physical_station_coordinates"):
        broken = dict(res)
        broken[key] = np.asarray(res[key])[::-1].copy()
        with pytest.raises(ShellAxesRefusal) as info:
            derive(el, mesh, u, broken)
        assert info.value.code == "AXES_STATIONS_NOT_QUALIFIED", key


@pytest.mark.parametrize("sec", ("h", "g"))
def test_r3_instance_method_shadows_are_refused(sec):
    """P1: instance attributes shadowing element methods (A'': instance attributes in scope)."""

    el, mesh, loc = build("v2d", "t3", sec)
    res, u = recover(el, mesh, loc)
    original = el.get_node_coordinates
    el.get_node_coordinates = lambda m: 2.0 * np.asarray(original(m))
    with pytest.raises(ShellAxesRefusal) as info:
        derive(el, mesh, u, res)
    assert info.value.code == "AXES_INSTANCE_SHADOW"
    el, mesh, loc = build("v2d", "t3", sec)
    res, u = recover(el, mesh, loc)
    offset = el.material_surface_offset_from_reference
    el.material_surface_offset_from_reference = lambda side: offset(-side)
    with pytest.raises(ShellAxesRefusal) as info:
        derive(el, mesh, u, res)
    assert info.value.code == "AXES_INSTANCE_SHADOW"


def test_r3_nan_in_the_rule_5_rebuild_refuses_every_family():
    """P1: max(0.0, nan) must not hide a nonfinite rule (5) surface rebuild."""

    el, mesh = _q4_about_z(math.pi / 4.0)
    loc = topology("q4")
    u = displacement(loc)
    res = el.compute_stresses(mesh, u, MATERIAL, return_global=True)
    forged = dict(res)
    for key in ("membrane_xx", "membrane_yy", "membrane_xy", "shear_xz", "shear_yz"):
        forged[key] = np.zeros_like(np.asarray(res[key], dtype=float))
    for key in ("bending_xx", "bending_yy", "bending_xy"):
        forged[key] = np.full_like(np.asarray(res[key], dtype=float), 1.4e308)
    for key in SURFACE_KEYS:
        forged[key] = np.zeros_like(np.asarray(res[key], dtype=float))
    out = derive(el, mesh, u, forged)
    assert set(out.refusals) >= set(FAMILIES), out.refusals
    assert not any(out.certified.values())
    assert set(FAMILY_KEYS.values()).isdisjoint(out.arrays)


def _negate_normal(role, frames):
    out = np.array(frames, copy=True)
    out[..., :, 2] *= -1.0
    return out


@pytest.mark.parametrize("branch", ["legacy-Q4-g", "e4pl-Q4flat-g", "e4pl-Q4flat-h", "s3-g", "v2d-g", "legacy-Q4-h"])
def test_r3_left_handed_frame_with_reversed_normal_is_refused(branch):
    """P1: negating only the basis normal leaves every in-plane tensor unchanged."""

    _bid, kind, topo, sec, _n_p, ref = dict((x[0], x) for x in BRANCHES)[branch]
    el, mesh, loc = build(kind, topo, sec, ref=ref)
    res, u = recover(el, mesh, loc)
    assert derive(el, mesh, u, res).refusals == {}
    out = derive(el, mesh, u, res, _frame_override=_negate_normal)
    assert set(out.refusals) >= set(FAMILIES), out.refusals
    assert set(NORMAL_KEYS.values()).isdisjoint(out.arrays)


def test_r3_refused_station_frames_feed_no_returned_value():
    """P1: refused station frames must not leak normals or nu.Q."""

    el, mesh, loc = build("e4pl", "q4w", "g")
    res, u = recover(el, mesh, loc)
    assert derive(el, mesh, u, res).refusals == {}
    out = derive(el, mesh, u, res, _frame_override=_override(_ry(1.0e-6), roles=("station",)))
    assert "membrane" in out.refusals and "bending" in out.refusals, out.refusals
    assert out.certified["shear"], out.refusals
    assert NORMAL_KEYS["membrane"] not in out.arrays
    assert NORMAL_KEYS["bending"] not in out.arrays
    assert "shear_normal_components" not in out.arrays
    assert NORMAL_KEYS["shear"] in out.arrays


def test_r3_malformed_recovery_and_subclasses_are_typed_refusals():
    """P2: malformed input stays inside the typed refusal boundary."""

    el, mesh, loc = build("legacy", "q4", "h")
    res, u = recover(el, mesh, loc)
    for value in (1.0, "text", [[1.0, 2.0]], np.zeros((4, 2))):
        broken = dict(res)
        broken["membrane_xx"] = value
        try:
            out = derive(el, mesh, u, broken)
        except ShellAxesRefusal:
            continue
        assert not any(out.certified.values()), (value, out.refusals)

    class Probe(ShellElement):
        def get_node_coordinates(self, mesh):  # must not run before the type check
            raise RuntimeError("subclass code ran")

    sub = Probe(1, [1, 2, 3, 4], "steel", thickness=H)
    with pytest.raises(ShellAxesRefusal) as info:
        derive(sub, mesh, u, res)
    assert info.value.code == "AXES_FORMULATION_NOT_QUALIFIED"


@pytest.mark.parametrize("polarity", (1, -1))
@pytest.mark.parametrize("sec", ("h", "g"))
@pytest.mark.parametrize("kind,topo", (("e4pl", "q4"), ("s3", "t3"), ("v2d", "t3")))
def test_r3_bending_sign_matches_the_physically_tensioned_side(kind, topo, sec, polarity):
    """P2: oracle independent of the solver's top/bot labels.

    w = -K x^2 / 2 puts the fibre on the +N0 side in tension (eps = z K), so
    e_x . M e_x is positive in the owner-normal (N0) convention and has the
    sign of the polarity in the physical-director convention.
    """

    el, mesh, loc = build(kind, topo, sec, polarity=polarity)
    u = plate_field(loc, kx=K)
    res = el.compute_stresses(mesh, u, MATERIAL, return_global=True)
    out = derive(el, mesh, u, res)
    assert out.certified["bending"], out.refusals
    e_x = R0[:, 0]
    value = np.einsum("i,nij,j->n", e_x, out.arrays[FAMILY_KEYS["bending"]], e_x)
    expected = 1.0 if out.conventions["bending"] == "owner_normal" else float(polarity)
    assert np.all(np.sign(value) == expected), (out.conventions["bending"], value)


# ------------------------------------------- B3 (owner 2026-11-26) + Codex r4
def _identical(a, b) -> bool:
    """Bit-identical deep comparison of recovery values."""

    if isinstance(a, dict) or isinstance(b, dict):
        return type(a) is type(b) and a.keys() == b.keys() and all(_identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            type(a) is type(b)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and (a.tobytes() == b.tobytes() if a.dtype != object else _identical(a.tolist(), b.tolist()))
        )
    if isinstance(a, float) and isinstance(b, float):
        return type(a) is type(b) and np.float64(a).tobytes() == np.float64(b).tobytes()
    return type(a) is type(b) and a == b


@pytest.mark.parametrize("polarity", (1, -1))
@pytest.mark.parametrize("case", ORDER_CASES, ids=ORDER_IDS)
def test_b3_public_route_recovery_and_axes_are_bit_identical(case, polarity):
    """B3: result.recovery == a direct compute_stresses call, result.axes ==
    the core on that direct recovery (arrays, refusals, diagnostics)."""

    if polarity == -1 and case[1] == "legacy":
        pytest.skip("legacy has no director")
    el, mesh, loc = _build_case(case, polarity=polarity) if case[1] != "legacy" else _build_case(case)
    u = displacement(loc)
    direct = el.compute_stresses(mesh, u, MATERIAL, return_global=True)
    result = recover_shell_axes(el, mesh, u, MATERIAL)
    assert isinstance(result, ShellAxesRecovery)
    assert _identical(result.recovery, direct)
    core = derive(el, mesh, u, direct)
    assert result.axes.refusals == core.refusals == {}
    assert result.axes.arrays.keys() == core.arrays.keys()
    for key in core.arrays:
        assert _identical(result.axes.arrays[key], core.arrays[key]), key
    assert result.axes.diagnostics == core.diagnostics


def test_b3_public_route_binds_custom_stations_and_refuses_them_elsewhere():
    el, mesh, loc = build("legacy", "q4", "h")
    u = displacement(loc)
    points = np.asarray(el.gauss_points, dtype=float)[::-1].copy()
    result = recover_shell_axes(el, mesh, u, MATERIAL, natural_points=points)
    direct = el.compute_stresses(mesh, u, MATERIAL, return_global=True, natural_points=points)
    assert _identical(result.recovery, direct)
    coords = el.get_node_coordinates(mesh)
    expected = np.asarray([el.compute_shape_functions(float(x), float(y))[0] @ coords for x, y in points])
    assert np.array_equal(result.axes.arrays["recovery_station_positions"], expected)
    el, mesh, loc = build("v2d", "t3", "h")
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, displacement(loc), MATERIAL, natural_points=points[:3])
    assert info.value.code == "AXES_STATIONS_NOT_QUALIFIED"


def _flip_y_and_normal(role, frames):
    """Right-handed: diag(1, -1, -1) keeps det = +1 and reverses the normal."""

    return np.asarray(frames) @ np.diag((1.0, -1.0, -1.0))


@pytest.mark.parametrize("field", ("zero", "axial_bending"))
@pytest.mark.parametrize(
    "branch", ["legacy-Q4-h", "legacy-Q4-g", "e4pl-Q4flat-g", "e4pl-Q4flat-h", "e4pl-Q4warp-g", "s3-g", "v2d-g", "v2d-h"]
)
def test_r4_right_handed_reversed_normal_is_refused_by_rule_6(branch, field):
    """r4 P1: zero (or xy-free) tensors cannot see diag(1,-1,-1); geometry can."""

    _bid, kind, topo, sec, _n_p, ref = dict((x[0], x) for x in BRANCHES)[branch]
    el, mesh, loc = build(kind, topo, sec, ref=ref)
    u = np.zeros(6 * len(loc)) if field == "zero" else plate_field(loc, kx=K)
    honest = recover_shell_axes(el, mesh, u, MATERIAL).axes
    # surface-only branches honestly refuse the families the bending field
    # leaves at zero (rule (3)); nothing else
    assert {r.reason for r in honest.refusals.values()} <= {"unresolvable_family"}, honest.refusals
    out = recover_shell_axes(el, mesh, u, MATERIAL, _frame_override=_flip_y_and_normal).axes
    assert set(out.refusals) >= set(FAMILIES), out.refusals
    if field == "zero":  # only geometry can see it; with tensors rules (1)-(5) may refuse first
        assert {r.reason for r in out.refusals.values()} == {"normal_geometry"}, out.refusals
    assert set(NORMAL_KEYS.values()).isdisjoint(out.arrays)
    with pytest.raises(ShellAxesFrameRuleMismatch):
        out.require_all()


ADMISSION_CODES = {"AXES_ATTRIBUTE_TYPE", "AXES_OBJECT_NOT_PLAIN", "AXES_INSTANCE_SHADOW"}


class _SquaresBig(float):
    def __pow__(self, power):
        return 4.0 * float(self) ** power


class _EqualsZero(float):
    def __eq__(self, other):
        return other == 0.0 or float.__eq__(self, other)

    __hash__ = float.__hash__


def test_r4_hostile_scalar_attribute_types_are_refused():
    """r4 P1: a float subclass gives the recovery and B2 different meanings."""

    el, mesh, loc = build("legacy", "q4", "h")
    el.thickness = _SquaresBig(H)
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, displacement(loc), MATERIAL)
    assert info.value.code in ADMISSION_CODES  # option P refuses it before the scalar check
    el, mesh, loc = build("v2d", "t3", "g")
    el.reference_surface_offset = _EqualsZero(0.125)
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, displacement(loc), MATERIAL)
    assert info.value.code in ADMISSION_CODES
    for name, value in (("director_polarity", True), ("director_polarity", 2), ("reference_normal", [0.0, 0.0, 1.0])):
        el, mesh, loc = build("e4pl", "q4", "h")
        object.__setattr__(el, name, value)
        with pytest.raises(ShellAxesRefusal) as info:
            recover_shell_axes(el, mesh, displacement(loc), MATERIAL)
        assert info.value.code in ADMISSION_CODES, name


def test_r4_v2d_station_check_is_exact_far_from_the_origin():
    """r4 P1: at x = 1e13 a relative bound let a reversed table and a second mesh pass."""

    el = create_shell_element(
        1, [1, 2, 3], "steel", formulation="e4-pl-s3-v2d", thickness=H, reference_normal=[0.0, 0.0, 1.0], director_polarity=1
    )
    u = np.zeros(18)
    far = FEMesh()
    for i, p in enumerate([(1.0e13, 0.0, 0.0), (1.0e13 + 1.0, 0.0, 0.0), (1.0e13, 1.0, 0.0)], start=1):
        far.add_node(i, *p)
    with pytest.raises(ShellAxesRefusal) as info:  # option 1: outside the declared domain
        recover_shell_axes(el, far, u, MATERIAL)
    assert info.value.code == "AXES_GEOMETRY_CONDITIONING"
    origin = 1.0e5  # inside the domain (G ~ 3e5): the exact table checks still bite
    mesh = FEMesh()
    for i, p in enumerate([(origin, 0.0, 0.0), (origin + 1.0, 0.0, 0.0), (origin, 1.0, 0.0)], start=1):
        mesh.add_node(i, *p)
    res = el.compute_stresses(mesh, u, MATERIAL, return_global=True)
    assert recover_shell_axes(el, mesh, u, MATERIAL).axes.refusals == {}
    broken = dict(res)
    broken["physical_station_coordinates"] = np.asarray(res["physical_station_coordinates"])[::-1].copy()
    with pytest.raises(ShellAxesRefusal) as info:
        derive(el, mesh, u, broken)
    assert info.value.code == "AXES_STATIONS_NOT_QUALIFIED"
    moved = FEMesh()
    for i, p in enumerate([(origin, 0.25, 0.0), (origin + 1.0, 0.25, 0.0), (origin, 1.25, 0.0)], start=1):
        moved.add_node(i, *p)
    with pytest.raises(ShellAxesRefusal) as info:
        derive(el, moved, u, res)
    assert info.value.code == "AXES_STATIONS_NOT_QUALIFIED"


def test_r4_complex_recovery_values_are_refused():
    """r4 P1: float() conversion silently dropped an infinite imaginary part."""

    el, mesh, loc = build("e4pl", "q4", "g")
    res, u = recover(el, mesh, loc)
    broken = dict(res)
    broken[FAMILY_KEYS["membrane"]] = np.asarray(res[FAMILY_KEYS["membrane"]]) + 1j * np.inf
    with pytest.raises(ShellAxesRefusal) as info:
        derive(el, mesh, u, broken)
    assert info.value.code == "AXES_RECOVERY_MALFORMED"


def test_r4_subclass_getattribute_runs_no_code_before_refusal():
    """r4 P2: admission by exact type only; no element attribute is read first."""

    class Hostile(ShellElement):
        def __getattribute__(self, name):
            if name in ("shell_section", "thickness", "__dict__", "get_node_coordinates"):
                raise RuntimeError(f"subclass code ran for {name}")
            return object.__getattribute__(self, name)

    el, mesh, loc = build("legacy", "q4", "h")
    res, u = recover(el, mesh, loc)
    sub = Hostile(1, [1, 2, 3, 4], "steel", thickness=H)
    for call in (lambda: recover_shell_axes(sub, mesh, u, MATERIAL), lambda: derive(sub, mesh, u, res)):
        with pytest.raises(ShellAxesRefusal) as info:
            call()
        assert info.value.code == "AXES_FORMULATION_NOT_QUALIFIED"


@pytest.mark.parametrize("field", ("zero", "mixed"))
@pytest.mark.parametrize("sec", ("h", "g"))
def test_r4_solver_admitted_tilted_v2d_normal_is_not_refused(sec, field):
    """r4 P2: V2D admits |e . n| <= 1e-10 sqrt(edge_scale); the module's
    orthonormality and tangency bounds follow that admission."""

    first_edge = R0 @ np.array([1.2, 0.0, 0.0])
    normal = N0 + 5.0e-11 * first_edge / np.linalg.norm(first_edge)
    normal /= np.linalg.norm(normal)
    el, mesh, loc = build("v2d", "t3", sec, normal=normal)
    u = np.zeros(18) if field == "zero" else displacement(loc)
    out = recover_shell_axes(el, mesh, u, MATERIAL).axes
    assert out.refusals == {}, out.refusals
    # beyond the admission the solver itself refuses (no module decision needed)
    far = N0 + 1.0e-8 * first_edge / np.linalg.norm(first_edge)
    far /= np.linalg.norm(far)
    with pytest.raises(Exception):
        bad, bmesh, bloc = build("v2d", "t3", sec, normal=far)
        bad.compute_stresses(bmesh, np.zeros(18), MATERIAL, return_global=True)


def _nan_in(role_to_poison):
    def hook(role, frames):
        out = np.array(frames, copy=True)
        if role == role_to_poison:
            out[..., 0, 0] = np.nan
        return out

    return hook


def test_r4_nonfinite_frame_refuses_only_the_families_using_it():
    """r4 P2: generalized warped Q4, station vs centre frames are separate."""

    el, mesh, loc = build("e4pl", "q4w", "g")
    u = displacement(loc)
    station = recover_shell_axes(el, mesh, u, MATERIAL, _frame_override=_nan_in("station")).axes
    assert {"membrane", "bending"} <= set(station.refusals)
    assert station.certified["shear"], station.refusals
    assert "shear_normal_components" not in station.arrays
    centre = recover_shell_axes(el, mesh, u, MATERIAL, _frame_override=_nan_in("centre")).axes
    assert centre.certified["membrane"] and centre.certified["bending"], centre.refusals
    assert "shear" in centre.refusals
    assert "shear_normal_components" not in centre.arrays


def _assert_surface_arrays(res, n_stations):
    """r6/r7 P3: the twelve surface components exist, have the station count and are finite."""

    assert n_stations > 0
    for key in SURFACE_KEYS:
        values = np.asarray(res[key], dtype=float)
        assert values.shape == (n_stations,), key
        assert np.all(np.isfinite(values)), key


def _projected_surface_stress(tensor, direction):
    """r8 P3: finite components can still project to +-inf; require finite projections."""

    stress = np.einsum("i,nij,j->n", direction, tensor, direction)
    assert np.all(np.isfinite(stress)), stress
    return stress


def test_r8_surface_oracle_rejects_infinite_projections():
    s_vec = np.sign(R0[:, 0])
    huge = 1.6e308 * np.outer(s_vec, s_vec)
    assert np.all(np.isfinite(huge))
    with pytest.raises(AssertionError):
        _projected_surface_stress(huge[None], R0[:, 0])


def test_r7_surface_oracle_rejects_nonfinite_arrays():
    el, mesh, loc = build("e4pl", "q4", "h")
    result = recover_shell_axes(el, mesh, plate_field(loc, kx=K), MATERIAL)
    res = dict(result.recovery)
    n = result.axes.arrays["recovery_station_positions"].shape[0]
    _assert_surface_arrays(res, n)
    res["global_xx_top"] = np.full(n, np.inf)
    res["global_xx_bot"] = np.full(n, -np.inf)
    with pytest.raises(AssertionError):
        _assert_surface_arrays(res, n)


@pytest.mark.parametrize("polarity", (1, -1))
@pytest.mark.parametrize("kind,topo", (("s3", "t3"), ("v2d", "t3"), ("e4pl", "q4")))
def test_r4_surface_stresses_are_tension_on_the_physically_tensioned_fibre(kind, topo, polarity):
    """r4 P2: oracle on the surfaces themselves, independent of top/bot labels.

    w = -K x^2 / 2 strains the fibre at signed distance z along N0 by z K.
    The 'top' surface lies at physical_top_offset_from_reference along the
    physical reference director, so its distance along N0 is that offset
    times (director . N0); its e_x stress must have that sign, 'bot' the
    opposite one.
    """

    el, mesh, loc = build(kind, topo, "h", polarity=polarity)
    u = plate_field(loc, kx=K)
    result = recover_shell_axes(el, mesh, u, MATERIAL)
    res = result.recovery
    missing = [k for k in SURFACE_KEYS if k not in res]
    assert not missing, (kind, missing)  # r5 P3: never skip when a field disappears
    _assert_surface_arrays(res, result.axes.arrays["recovery_station_positions"].shape[0])
    director = np.asarray(el.physical_reference_director, dtype=float)
    if kind == "e4pl":
        top_offset, bot_offset = 0.5 * H, -0.5 * H
    else:
        top_offset = el.material_surface_offset_from_reference(1.0)
        bot_offset = el.material_surface_offset_from_reference(-1.0)
    e_x = R0[:, 0]
    for surface, offset in (("top", top_offset), ("bot", bot_offset)):
        tensor = np.zeros((len(res[f"global_xx_{surface}"]), 3, 3))
        for label, i, j in (("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("xy", 0, 1), ("yz", 1, 2), ("xz", 0, 2)):
            values = np.asarray(res[f"global_{label}_{surface}"], dtype=float)
            tensor[:, i, j] = values
            tensor[:, j, i] = values
        stress = _projected_surface_stress(tensor, e_x)
        expected = np.sign(offset * float(director @ N0))
        assert np.all(np.sign(stress) == expected), (surface, offset, stress)


# ------------------------------------------------------ Codex r5 new classes
def test_r5_s3_with_a_tilted_owner_normal_certifies():
    """r5 P2: S3 frames use the facet normal oriented by the owner normal."""

    first_edge = R0 @ np.array([1.2, 0.0, 0.0])
    owner = math.cos(0.01) * N0 + math.sin(0.01) * first_edge / np.linalg.norm(first_edge)
    for sec in ("h", "g"):
        el, mesh, loc = build("s3", "t3", sec, normal=owner)
        out = recover_shell_axes(el, mesh, displacement(loc), MATERIAL).axes
        assert out.refusals == {}, out.refusals
        assert np.allclose(out.arrays[NORMAL_KEYS["bending"]], N0[None], rtol=0, atol=1e-14)


@pytest.mark.parametrize("kind", ("e4pl", "legacy"))
def test_r5_large_origin_is_not_a_spurious_rule_6_refusal(kind):
    """r5 P2: cancellation of absolute coordinates must not refuse an admitted facet."""

    e1 = np.array([1.1, 0.4, 0.39])
    e2 = np.array([0.04, 1.05, 0.89])
    origin = np.array([1.0e5, 1.0e5, 1.0e5])  # inside the declared domain (G ~ 1.8e5)
    mesh = FEMesh()
    for i, p in enumerate((origin, origin + e1, origin + e1 + e2, origin + e2), start=1):
        mesh.add_node(i, *p)
    normal = np.cross(e1, e2)
    normal /= np.linalg.norm(normal)
    if kind == "e4pl":
        el = create_shell_element(
            1, [1, 2, 3, 4], "steel", formulation="e4-pl", thickness=H,
            reference_normal=[float(v) for v in normal], director_polarity=1,
        )
    else:
        el = ShellElement(1, [1, 2, 3, 4], "steel", thickness=H)
    for u in (np.zeros(24), np.tile(np.array([1e-4, -2e-4, 3e-4, 1e-3, -2e-3, 0.0]), 4) * np.repeat([0.0, 1.0, 2.0, 1.0], 6)):
        out = recover_shell_axes(el, mesh, u, MATERIAL).axes
        assert {r.reason for r in out.refusals.values()} <= {"unresolvable_family"}, out.refusals


def test_r5_ill_conditioned_warped_centre_shear_is_not_certified():
    """r5 P1: the projected surface residual cannot bound the full centre shear
    on extreme warping; a 1e-8 centre-frame rotation must not certify."""

    mesh = FEMesh()
    for i, (r, s) in enumerate(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)), start=1):
        mesh.add_node(i, r, 1.0e13 * s, 1.0e7 * r * s)
    el = create_shell_element(
        1, [1, 2, 3, 4], "steel", formulation="e4-pl", thickness=H,
        reference_normal=[0.0, 0.0, 1.0], director_polarity=1,
    )
    u = np.tile(np.array([0.0, 0.0, 0.0, -0.001, 0.0, 0.0]), 4)

    def rotate_centre(role, frames):
        return np.asarray(frames) @ _rz(1.0e-8) if role == "centre" else frames

    # option 1: this geometry (aspect 1e13) is outside the declared domain
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, u, MATERIAL, _frame_override=rotate_centre)
    assert info.value.code == "AXES_GEOMETRY_CONDITIONING"


# --------------------------------------- option P (owner 2026-11-26): r5 repeats
class _ReversedRows(np.ndarray):
    def __iter__(self):
        return iter(np.asarray(self)[::-1])


def test_p_instance_shadow_of_class_data_is_refused():
    """r5 #1: an instance GAUSS_POINTS_2x2 (class data read via self) with a reversed iterator."""

    for sec in ("h", "g"):
        el, mesh, loc = build("legacy", "q4", sec)
        el.GAUSS_POINTS_2x2 = np.asarray(ShellElement.GAUSS_POINTS_2x2, dtype=float).view(_ReversedRows)
        with pytest.raises(ShellAxesRefusal) as info:
            recover_shell_axes(el, mesh, displacement(loc), MATERIAL)
        assert info.value.code == "AXES_INSTANCE_SHADOW"


def test_p_mesh_instance_callable_is_refused():
    """r5 #2: an instance mesh.get_node can show the recovery and the module different nodes."""

    el, mesh, loc = build("legacy", "q4", "h")
    original = mesh.get_node
    mesh.get_node = lambda node_id: original(node_id)
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, displacement(loc), MATERIAL)
    assert info.value.code == "AXES_INSTANCE_SHADOW"
    el, mesh, loc = build("legacy", "q4", "h")
    node = mesh.get_node(1)
    node.coords = lambda: np.zeros(3)
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, displacement(loc), MATERIAL)
    assert info.value.code == "AXES_INSTANCE_SHADOW"


class _HidingDict(dict):
    def __iter__(self):
        return iter(())

    def keys(self):
        return ()


def test_p_dict_subclass_instance_dict_is_refused():
    """r5 #3: a dict-subclass __dict__ whose iteration hides a method shadow."""

    el, mesh, loc = build("legacy", "q4", "h")
    rotated = el._local_frame_and_derivatives

    def shadow(coords, d_xi, d_eta):
        frame, *rest = rotated(coords, d_xi, d_eta)
        return (frame @ _rz(0.5), *rest)

    hidden = _HidingDict(vars(el))
    hidden["_local_frame_and_derivatives"] = shadow
    el.__dict__ = hidden
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, displacement(loc), MATERIAL)
    assert info.value.code == "AXES_OBJECT_NOT_PLAIN"


class _FloatLookalike(type):
    def __eq__(cls, other):
        return other is float or type.__eq__(cls, other)

    __hash__ = type.__hash__


class _SpoofedFloat(float, metaclass=_FloatLookalike):
    def __pow__(self, power):
        return 4.0 * float(self) ** power


def test_p_metaclass_spoofed_scalar_type_is_refused():
    """r5 #4: type admission by identity, never by equality."""

    assert _SpoofedFloat == float  # the spoof works against an equality check
    el, mesh, loc = build("legacy", "q4", "h")
    el.thickness = _SpoofedFloat(H)
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, displacement(loc), MATERIAL)
    assert info.value.code in ADMISSION_CODES


def test_p_object_and_ragged_natural_points_are_typed_refusals():
    """r5 #6: object arrays hid complex(0, inf); ragged input must not escape untyped."""

    el, mesh, loc = build("legacy", "q4", "h")
    u = displacement(loc)
    hidden = np.array([[np.complex128(complex(0.0, np.inf)), 0.0]], dtype=object)
    for request in (hidden, [[0.0], [0.0, 0.0]], [[complex(0.0, 1.0), 0.0]], "points", [[True, 0.0]]):
        with pytest.raises(ShellAxesRefusal) as info:
            recover_shell_axes(el, mesh, u, MATERIAL, natural_points=request)
        assert info.value.code == "AXES_STATIONS_NOT_QUALIFIED", request


# ------------------------------- option 1 (owner 2026-11-26): domain, ordinary inputs
from anyfesolver.shell_axes_outputs import GEOMETRY_CONDITIONING_LIMIT, GEOMETRY_ROUNDOFF  # noqa: E402


def _square_e4pl(offset, *, warped=0.0, side=1.0):
    mesh = FEMesh()
    for i, (x, y) in enumerate(((0.0, 0.0), (side, 0.0), (side, side), (0.0, side)), start=1):
        z = warped * (1.0 if i in (1, 3) else -1.0)
        mesh.add_node(i, offset + x, offset + y, offset + z)
    el = create_shell_element(
        1, [1, 2, 3, 4], "steel", formulation="e4-pl", thickness=H, reference_normal=[0.0, 0.0, 1.0], director_polarity=1
    )
    return el, mesh


def _element_conditioning(el, mesh):
    import anyfesolver.shell_axes_outputs as module

    coords = np.asarray(el.get_node_coordinates(mesh), dtype=float)
    values = [module._geometry_conditioning(coords, coords[2] - coords[0], coords[3] - coords[1])]
    for xi, eta in np.asarray(el.gauss_points, dtype=float):
        _n, dxi, deta = el.compute_shape_functions(float(xi), float(eta))
        values.append(module._geometry_conditioning(coords, np.asarray(dxi) @ coords, np.asarray(deta) @ coords))
    return max(values)


def _square_at(fraction, *, warped=0.0):
    """e4-pl unit square translated so that its geometry conditioning is
    ``fraction`` of the declared limit (bisection on the module's own G)."""

    low, high = 0.0, 1.0e8
    for _ in range(200):
        mid = 0.5 * (low + high)
        el, mesh = _square_e4pl(mid, warped=warped)
        if _element_conditioning(el, mesh) < fraction * GEOMETRY_CONDITIONING_LIMIT:
            low = mid
        else:
            high = mid
    el, mesh = _square_e4pl(low, warped=warped)
    assert abs(_element_conditioning(el, mesh) / GEOMETRY_CONDITIONING_LIMIT - fraction) < 1e-3 * fraction
    return el, mesh


def _rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@pytest.mark.parametrize("fraction", (0.8, 0.999))
def test_domain_inside_boundary_reversed_normal_and_tilt_are_refused(fraction):
    """Just inside the domain: ordinary input certifies; a reversed normal and a
    normal tilt of 1e-9 (the resolution) are refused."""

    el, mesh = _square_at(fraction)
    for u in (np.zeros(24), np.tile(np.array([0.0, 0.0, 0.0, -0.001, 0.0, 0.0]), 4)):
        assert recover_shell_axes(el, mesh, u, MATERIAL).axes.refusals == {}
        flipped = recover_shell_axes(el, mesh, u, MATERIAL, _frame_override=_flip_y_and_normal).axes
        assert set(flipped.refusals) >= set(FAMILIES), flipped.refusals
        tilted = recover_shell_axes(el, mesh, u, MATERIAL, _frame_override=_override(_rx(1.0e-9))).axes
        assert set(tilted.refusals) >= set(FAMILIES), tilted.refusals


def test_domain_inside_boundary_warped_centre_tilt_is_refused():
    """r6 P1 #5 inside the domain: a centre-plane tilt is seen by rule (6)."""

    el, mesh = _square_at(0.8, warped=0.02)
    u = np.tile(np.array([0.0, 0.0, 0.0, -0.001, 0.0, 0.0]), 4)
    honest = recover_shell_axes(el, mesh, u, MATERIAL).axes
    # the rotation-only field leaves membrane/bending unresolvable on this
    # surface-only branch (rule (3)); shear is the family under test
    assert honest.certified["shear"], honest.refusals
    assert {r.reason for r in honest.refusals.values()} <= {"unresolvable_family"}, honest.refusals
    out = recover_shell_axes(el, mesh, u, MATERIAL, _frame_override=_override(_rx(1.0e-8), roles=("centre",))).axes
    assert not out.certified["shear"], out.refusals


@pytest.mark.parametrize("z", ("1.25 limit", 2.0e14))
def test_domain_outside_is_a_typed_conditioning_refusal(z):
    """r6 P1 #4: outside the declared domain the module refuses instead of bounding."""

    if z == "1.25 limit":
        el, mesh = _square_at(1.25)
    else:
        el, mesh = _square_e4pl(0.0)
        mesh = FEMesh()
        for i, (x, y) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1)), start=1):
            mesh.add_node(i, float(x), float(y), z)
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, np.zeros(24), MATERIAL)
    assert info.value.code == "AXES_GEOMETRY_CONDITIONING"


def test_domain_nearly_parallel_diagonals_inside_the_domain_certify():
    """r6 P2 #6: the conditioned bound covers nearly parallel diagonals."""

    u_dir = np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0)
    v_dir = np.array([3.0, 0.0, -1.0]) / math.sqrt(10.0)
    normal = np.cross(u_dir, v_dir)
    normal /= np.linalg.norm(normal)
    mesh = FEMesh()
    for i, (r, s) in enumerate(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)), start=1):
        mesh.add_node(i, *(r * u_dir + 1.0e-5 * s * v_dir))
    el = create_shell_element(
        1, [1, 2, 3, 4], "steel", formulation="e4-pl", thickness=H,
        reference_normal=[float(v) for v in normal], director_polarity=1,
    )
    out = recover_shell_axes(el, mesh, np.zeros(24), MATERIAL).axes
    assert out.refusals == {}, out.refusals


def test_ordinary_element_activity_int64_ids_and_float32_coordinates_are_admitted():
    """r6 P2 #8: ordinary solver inputs the legacy recovery accepts."""

    from anyfesolver.activity import ElementActivity

    el, mesh, loc = build("legacy", "q4", "h")
    mesh.element_activity = ElementActivity([1])  # what FEModel.set_element_activity attaches
    assert recover_shell_axes(el, mesh, displacement(loc), MATERIAL).axes.refusals == {}

    el, mesh, loc = build("legacy", "q4", "h")
    wide = ShellElement(1, [np.int64(i) for i in (1, 2, 3, 4)], "steel", thickness=H)
    assert recover_shell_axes(wide, mesh, displacement(loc), MATERIAL).axes.refusals == {}

    el, mesh, loc = build("legacy", "q4", "h")
    for node in mesh.nodes.values():
        node.x, node.y, node.z = (np.float32(node.x), np.float32(node.y), np.float32(node.z))
    out = recover_shell_axes(el, mesh, displacement(loc), MATERIAL).axes
    assert {r.reason for r in out.refusals.values()} <= {"unresolvable_family"}, out.refusals


# ------------------------------------------------------------- Codex r7 fixes
def test_r7_tilt_above_resolution_near_the_old_limit_does_not_certify():
    """r7 P1: G ~ 4.7e5 certified a 1.2e-9 tilt (component check, no margin)."""

    U = np.array([1.0, 1.0, 1.0])
    V = 7.0 * 2.0**-22 * np.array([1.0, -1.0, 0.0])
    mesh = FEMesh()
    for i, p in enumerate((-U - V, U - V, U + V, -U + V), start=1):
        mesh.add_node(i, *p)
    normal = np.cross(U, V)
    normal /= np.linalg.norm(normal)
    el = create_shell_element(
        1, [1, 2, 3, 4], "steel", formulation="e4-pl", thickness=H,
        reference_normal=[float(v) for v in normal], director_polarity=1,
    )

    def tilt(role, frames):
        return np.asarray(frames) @ _ry(1.2e-9)

    try:
        out = recover_shell_axes(el, mesh, np.zeros(24), MATERIAL, _frame_override=tilt).axes
    except ShellAxesRefusal as exc:
        assert exc.code == "AXES_GEOMETRY_CONDITIONING"
        return
    assert not any(out.certified.values()), out.refusals


@pytest.mark.parametrize("axis", ("x", "y"))
def test_r7_euclidean_tilt_check_inside_the_domain(axis):
    """Inside the domain at 0.999 of the limit: a 1e-9 tilt in any direction is refused."""

    el, mesh = _square_at(0.999)
    rotation = _rx(1.0e-9) if axis == "x" else _ry(1.0e-9)
    out = recover_shell_axes(el, mesh, np.zeros(24), MATERIAL, _frame_override=_override(rotation)).axes
    assert set(out.refusals) >= set(FAMILIES), out.refusals


def test_r7_domain_covers_custom_natural_points():
    """r7 P1: G 13 at the default stations, 2.2e6 at natural_points [[0, 1]]."""

    mesh = FEMesh()
    for i, p in enumerate(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.999999, 1.0, 0.0)), start=1):
        mesh.add_node(i, *p)
    el = create_shell_element(
        1, [1, 2, 3, 4], "steel", formulation="e4-pl", thickness=H, reference_normal=[0.0, 0.0, 1.0], director_polarity=1
    )
    assert recover_shell_axes(el, mesh, np.zeros(24), MATERIAL).axes.refusals == {}
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, np.zeros(24), MATERIAL, natural_points=[[0.0, 1.0]])
    assert info.value.code == "AXES_GEOMETRY_CONDITIONING"


def test_r7_domain_refusal_comes_before_the_recovery():
    """r7 P2: an extreme sliver gets the typed refusal, not the solver's ValueError."""

    mesh = FEMesh()
    for i, p in enumerate(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0e-13, 0.0), (0.0, 1.0e-13, 0.0)), start=1):
        mesh.add_node(i, *p)
    el = create_shell_element(
        1, [1, 2, 3, 4], "steel", formulation="e4-pl", thickness=H, reference_normal=[0.0, 0.0, 1.0], director_polarity=1
    )
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, np.zeros(24), MATERIAL)
    assert info.value.code == "AXES_GEOMETRY_CONDITIONING"


def test_r7_exact_limit_is_inside_the_domain():
    """r7 P2: a unit square whose exact G equals the limit is admitted (inclusive).

    For the unit square the station tangents have length 0.5 and are
    orthogonal, so G = max|x| / 0.5 + 1; the diagonals give less."""

    reach = (GEOMETRY_CONDITIONING_LIMIT - 1.0) * 0.5
    el, mesh = _square_e4pl(reach - 1.0)
    assert max(abs(n.x) for n in mesh.nodes.values()) == reach
    assert recover_shell_axes(el, mesh, np.zeros(24), MATERIAL).axes.refusals == {}


def test_r7_numpy_scalars_in_natural_points_rows_are_admitted():
    """r7 P2: list rows of np.float32 / np.int32 as the recovery accepts them."""

    el, mesh, loc = build("legacy", "q4", "h")
    u = displacement(loc)
    request = [(np.float32(0.25), np.int32(0)), (np.float32(-0.5), np.int32(1))]
    result = recover_shell_axes(el, mesh, u, MATERIAL, natural_points=request)
    direct = el.compute_stresses(mesh, u, MATERIAL, return_global=True, natural_points=np.asarray(request, dtype=float))
    assert _identical(result.recovery, direct)


# ------------------------------------------------------------- Codex r8 fixes
def test_r8_domain_includes_the_warped_centre_tangents():
    """r8 P1: diagonals and stations G ~ 1.9e5, the centre tangents ~ 4e5."""

    mesh = FEMesh()
    for i, p in enumerate(((-3.0, -2.0e-5, 0.1), (-1.0, -2.0e-5, -0.1), (3.0, 2.0e-5, 0.1), (1.0, 2.0e-5, -0.1)), start=1):
        mesh.add_node(i, *p)
    el = create_shell_element(
        1, [1, 2, 3, 4], "steel", formulation="e4-pl", thickness=H, reference_normal=[0.0, 0.0, 1.0], director_polarity=1
    )
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, np.zeros(24), MATERIAL)
    assert info.value.code == "AXES_GEOMETRY_CONDITIONING"


def test_r8_huge_finite_geometry_gets_the_conditioning_code():
    """r8 P2: sides 2**270 x 2**250 must not overflow into AXES_GEOMETRY_DEGENERATE."""

    a, b = 2.0**270, 2.0**250
    mesh = FEMesh()
    for i, p in enumerate(((0.0, 0.0, 0.0), (a, 0.0, 0.0), (a, b, 0.0), (0.0, b, 0.0)), start=1):
        mesh.add_node(i, *p)
    el = create_shell_element(
        1, [1, 2, 3, 4], "steel", formulation="e4-pl", thickness=H, reference_normal=[0.0, 0.0, 1.0], director_polarity=1
    )
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, np.zeros(24), MATERIAL)
    assert info.value.code == "AXES_GEOMETRY_CONDITIONING"


def test_r8_natural_points_rows_may_be_exact_1d_arrays():
    """r8 P2: a list of exact (2,) numeric ndarray rows, as the recovery accepts."""

    el, mesh, loc = build("legacy", "q4", "h")
    u = displacement(loc)
    request = [np.array([0.25, 0.0], dtype=np.float32), np.array([-0.5, 1.0])]
    result = recover_shell_axes(el, mesh, u, MATERIAL, natural_points=request)
    direct = el.compute_stresses(mesh, u, MATERIAL, return_global=True, natural_points=np.asarray(request, dtype=float))
    assert _identical(result.recovery, direct)
    with pytest.raises(ShellAxesRefusal):
        recover_shell_axes(el, mesh, u, MATERIAL, natural_points=[np.array([0.25, 0.0, 1.0])])


def test_r8_refusal_text_states_the_actual_limit():
    """r8 P3: the refusal prints 250000, not 2e+05."""

    el, mesh = _square_at(1.25)
    with pytest.raises(ShellAxesRefusal) as info:
        recover_shell_axes(el, mesh, np.zeros(24), MATERIAL)
    assert "250000" in info.value.reason
    import anyfesolver.shell_axes_outputs as module

    assert "2.5e5" in module.__doc__ and "5e5`` (see" not in module.__doc__.replace("2.5e5`` (see", "")


def test_cutoff_in_double_precision_keeps_the_2x_margin():
    """The cutoff is evaluated in double precision; within a few ulp of the
    limit the tilt guarantee keeps its 2x margin below the 1e-9 resolution."""

    import anyfesolver.shell_axes_outputs as module

    for ulps in (-4, 0, 4):
        g = GEOMETRY_CONDITIONING_LIMIT * (1.0 + ulps * 2.0**-52)
        bound = module.GEOMETRY_BOUND + GEOMETRY_ROUNDOFF * g
        assert 2.0 * bound <= module.RESOLUTION, (ulps, bound)
    # and behaviourally: an element at the limit (to 1e-3) refuses a 1e-9 tilt
    el, mesh = _square_at(0.9999)
    out = recover_shell_axes(el, mesh, np.zeros(24), MATERIAL, _frame_override=_override(_ry(1.0e-9))).axes
    assert set(out.refusals) >= set(FAMILIES), out.refusals
