"""D2-1 / O5(a): additive global outputs of shell recovery (return_global only).

For every shell recovery branch the following are checked:

(a) the new keys exist with the right shape under ``return_global=True`` and
    are absent under ``return_global=False``;
(b) every key that existed before is bit-identical with and without
    ``return_global`` (shared keys only);
(c) each stored global family tensor, re-expressed in the frame the family
    was rotated with, reproduces the native local membrane / bending / shear
    values to 1e-13 of the family's maximum (including fixtures where one
    family is 1e-8 of the other, which any top/bottom-surface reconstruction
    would fail);
(d) ``recovery_station_positions`` equals shape-function interpolation of the
    node coordinates;
(e) basis normals are unit vectors, annihilate their family tensor, and for
    planar flat elements equal the element normal.

Fixtures are rigidly rotated and use reversed / cyclically shifted node order
for the four-node elements, so that node order matters.
"""

from __future__ import annotations

import math
import pathlib

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
from anyfesolver.shell_sections import GeneralizedShellSection

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

NEW_FAMILY_KEYS = (
    "global_membrane_resultant_tensors",
    "global_bending_resultant_tensors",
    "global_transverse_shear_resultants",
)
NEW_NORMAL_KEYS = (
    "membrane_basis_normals",
    "bending_basis_normals",
    "shear_basis_normals",
)
NEW_KEYS = NEW_FAMILY_KEYS + ("recovery_station_positions",) + NEW_NORMAL_KEYS


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


def recover(el, mesh, loc, flag, **scales):
    return el.compute_stresses(
        mesh, displacement(loc, **scales), MATERIAL, return_global=flag
    )


# --------------------------------------------------------------------- cases
# (id, kind, topo, sec, expected n_p)
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
        BRANCHES.append((f"legacy-{_label}-{_sec}", "legacy", _topo, _sec, _np))
for _topo, _label in (("q4", "Q4flat"), ("q4w", "Q4warp")):
    for _sec in ("h", "g"):
        BRANCHES.append((f"e4pl-{_label}-{_sec}", "e4pl", _topo, _sec, 4))
for _sec in ("h", "g"):
    BRANCHES.append((f"s3-{_sec}", "s3", "t3", _sec, 7))
    BRANCHES.append((f"v2d-{_sec}", "v2d", "t3", _sec, 3))
IDS = [b[0] for b in BRANCHES]

# four-node elements are also run with reversed and shifted numbering
ORDER_CASES = [
    (bid + f"-{order}", kind, topo, sec, n_p, order)
    for (bid, kind, topo, sec, n_p) in BRANCHES
    if topo in ("q4", "q4w", "t3") and kind in ("legacy", "e4pl", "v2d")
    for order in ("std", "rev", "shift")
]


def _case(request_param):
    bid, kind, topo, sec, n_p, order = request_param
    return build(kind, topo, sec, order=order), kind, topo, sec, n_p


# --------------------------------------------------------------- frame access
def _natural_points(el, res, kind, topo):
    if kind == "e4pl" and "natural_point_coordinates" in res:
        return np.asarray(res["natural_point_coordinates"], float)
    if kind == "s3":
        return np.asarray([(r, s) for r, s, _w in TRIANGLE_QUADRATURE], float)
    return np.asarray(el.gauss_points, float)


def _shape_values(el, kind, point):
    if kind in ("s3", "v2d"):
        r, s = point
        return np.array([1.0 - r - s, r, s])
    xi, eta = point
    if el.num_nodes == 3:
        return np.array([1.0 - xi - eta, xi, eta])
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


def _frames(el, mesh, res, kind, topo, n_p):
    """Return (membrane_frames, bending_frames, shear_frames), each (n_p,3,3)."""

    coords = el.get_node_coordinates(mesh)
    if kind == "legacy":
        out = []
        for xi, eta in el.gauss_points:
            _N, dxi, deta = el.compute_shape_functions(float(xi), float(eta))
            out.append(el._local_frame_and_derivatives(coords, dxi, deta)[0])
        out = np.asarray(out)
        return out, out, out
    if kind == "e4pl":
        planar = "natural_point_coordinates" in res
        if planar:
            frame = el._physical_director_context(equation7_frame(coords)[0])[0]
            rep = np.repeat(frame[None], n_p, axis=0)
            return rep, rep, rep
        if el.reference_normal is None:
            raise AssertionError("unreferenced warped path is legacy")
        centre = el._physical_director_context(el._center_frame(coords))[0]
        stations = []
        for xi, eta in el.gauss_points:
            _N, dxi, deta = el.compute_shape_functions(float(xi), float(eta))
            numbered = el._local_frame_and_derivatives(coords, dxi, deta)[0]
            stations.append(el._physical_director_context(numbered)[0])
        stations = np.asarray(stations)
        return stations, stations, np.repeat(centre[None], n_p, axis=0)
    if kind == "s3":
        frame = triangle_frame(coords, el.reference_normal)[0]
    else:
        frame = np.asarray(res["frame"], float)
    rep = np.repeat(frame[None], n_p, axis=0)
    return rep, rep, rep


def _native(res, sec):
    """Native local (N, M, Q) per station as rows (11,22,12), (11,22,12), (13,23)."""

    if sec == "g":
        return (
            np.asarray(res["membrane_resultants"], float),
            np.asarray(res["bending_resultants"], float),
            np.asarray(res["transverse_shear_resultants"], float),
        )
    membrane = np.column_stack(
        (res["membrane_xx"], res["membrane_yy"], res["membrane_xy"])
    )
    bending = np.column_stack(
        (res["bending_xx"], res["bending_yy"], res["bending_xy"])
    )
    shear = np.column_stack((res["shear_xz"], res["shear_yz"]))
    return H * membrane, bending * H * H / 6.0, H * shear


def _tensor(v):
    return np.array([[v[0], v[2], 0.0], [v[2], v[1], 0.0], [0.0, 0.0, 0.0]])


# ------------------------------------------------------------------- (a)/(b)
@pytest.mark.parametrize("branch", BRANCHES, ids=IDS)
def test_new_keys_only_under_return_global_and_old_keys_identical(branch):
    bid, kind, topo, sec, n_p = branch
    el, mesh, loc = build(kind, topo, sec)
    off = recover(el, mesh, loc, False)
    on = recover(el, mesh, loc, True)
    for key in NEW_KEYS:
        assert key not in off, key
        assert key in on, key
    assert on["recovery_station_positions"].shape == (n_p, 3)
    assert on["global_membrane_resultant_tensors"].shape == (n_p, 3, 3)
    assert on["global_bending_resultant_tensors"].shape == (n_p, 3, 3)
    assert on["global_transverse_shear_resultants"].shape == (n_p, 3)
    for key in NEW_NORMAL_KEYS:
        assert on[key].shape == (n_p, 3), key
    # ``recovery_scope`` is a label that already differed with the flag before
    # D2-1 (it names whether global fields are present).
    for key in (set(off) & set(on)) - {"recovery_scope"}:
        a, b = off[key], on[key]
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), key
        else:
            assert a == b, key


# ---------------------------------------------------------------------- (c)
SCALES = (
    dict(bend=1.0, mem=1.0, shear=1.0),
    dict(bend=1.0, mem=1.0e-8, shear=1.0e-8),
    dict(bend=1.0e-8, mem=1.0, shear=1.0e-8),
    dict(bend=1.0e-8, mem=1.0e-8, shear=1.0),
)
SCALE_IDS = ("mixed", "membrane-1e-8", "bending-1e-8", "only-shear")


@pytest.mark.parametrize("scales", SCALES, ids=SCALE_IDS)
@pytest.mark.parametrize("case", ORDER_CASES, ids=[c[0] for c in ORDER_CASES])
def test_global_tensors_reproduce_native_local_values(case, scales):
    (el, mesh, loc), kind, topo, sec, n_p = _case(case)
    res = recover(el, mesh, loc, True, **scales)
    _check_native(el, mesh, res, kind, topo, sec, n_p)


@pytest.mark.parametrize(
    "branch",
    [b for b in BRANCHES if b[2] not in ("q4", "q4w", "t3")],
    ids=[b[0] for b in BRANCHES if b[2] not in ("q4", "q4w", "t3")],
)
def test_global_tensors_reproduce_native_local_values_higher_order(branch):
    bid, kind, topo, sec, n_p = branch
    el, mesh, loc = build(kind, topo, sec)
    for scales in SCALES:
        res = recover(el, mesh, loc, True, **scales)
        _check_native(el, mesh, res, kind, topo, sec, n_p)


def _check_native(el, mesh, res, kind, topo, sec, n_p):
    fm, fb, fs = _frames(el, mesh, res, kind, topo, n_p)
    nat_m, nat_b, nat_q = _native(res, sec)
    for key, frames, native, is_shear in (
        ("global_membrane_resultant_tensors", fm, nat_m, False),
        ("global_bending_resultant_tensors", fb, nat_b, False),
        ("global_transverse_shear_resultants", fs, nat_q, True),
    ):
        stored = np.asarray(res[key], float)
        scale = float(np.max(np.abs(native)))
        assert scale > 0.0
        for index in range(n_p):
            frame = frames[index]
            if is_shear:
                local = frame.T @ stored[index]
                assert abs(local[2]) <= 1.0e-13 * scale
                assert np.max(np.abs(local[:2] - native[index])) <= 1.0e-13 * scale
            else:
                local = frame.T @ stored[index] @ frame
                expected = _tensor(native[index])
                assert np.max(np.abs(local - expected)) <= 1.0e-13 * scale, (
                    key,
                    index,
                )


# ---------------------------------------------------------------------- (d)
@pytest.mark.parametrize("case", ORDER_CASES, ids=[c[0] for c in ORDER_CASES])
def test_station_positions_are_shape_function_interpolation(case):
    (el, mesh, loc), kind, topo, sec, n_p = _case(case)
    res = recover(el, mesh, loc, True)
    _check_positions(el, mesh, res, kind, topo, n_p)


@pytest.mark.parametrize("branch", BRANCHES, ids=IDS)
def test_station_positions_all_branches(branch):
    bid, kind, topo, sec, n_p = branch
    el, mesh, loc = build(kind, topo, sec)
    res = recover(el, mesh, loc, True)
    _check_positions(el, mesh, res, kind, topo, n_p)


def _check_positions(el, mesh, res, kind, topo, n_p):
    coords = np.asarray(el.get_node_coordinates(mesh), float)
    points = _natural_points(el, res, kind, topo)
    if kind == "v2d":
        # barycentric Hammer points in external node order
        expected = np.asarray(res["external_barycentric_coordinates"], float) @ coords
    else:
        expected = np.asarray(
            [_shape_values(el, kind, p) @ coords for p in points], float
        )
    got = np.asarray(res["recovery_station_positions"], float)
    assert got.shape == (n_p, 3)
    scale = float(np.max(np.abs(coords)))
    assert np.max(np.abs(got - expected)) <= 1.0e-14 * scale
    if kind == "v2d":
        assert np.array_equal(got, res["physical_station_coordinates"])


# ---------------------------------------------------------------------- (e)
def _expected_normal(kind, order, polarity=1):
    if kind == "legacy":
        return N0 if order in ("std", "shift") else -N0
    if kind == "e4pl":
        return polarity * N0
    return N0


@pytest.mark.parametrize("case", ORDER_CASES, ids=[c[0] for c in ORDER_CASES])
def test_basis_normals_unit_annihilate_family_and_match_flat_normal(case):
    (el, mesh, loc), kind, topo, sec, n_p = _case(case)
    order = case[5]
    res = recover(el, mesh, loc, True, bend=1.0, mem=1.0, shear=1.0)
    for key in NEW_NORMAL_KEYS:
        normals = np.asarray(res[key], float)
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=0, atol=1e-14)
    nm, nb, ns = (np.asarray(res[k], float) for k in NEW_NORMAL_KEYS)
    gm = np.asarray(res["global_membrane_resultant_tensors"], float)
    gb = np.asarray(res["global_bending_resultant_tensors"], float)
    gs = np.asarray(res["global_transverse_shear_resultants"], float)
    for i in range(n_p):
        assert np.linalg.norm(gm[i] @ nm[i]) <= 1e-12 * np.max(np.abs(gm))
        assert np.linalg.norm(gb[i] @ nb[i]) <= 1e-12 * np.max(np.abs(gb))
        assert abs(gs[i] @ ns[i]) <= 1e-12 * np.max(np.abs(gs))
    if topo != "q4w":
        expected = _expected_normal(kind, order)
        for normals in (nm, nb, ns):
            assert np.allclose(normals, expected[None, :], rtol=0, atol=1e-14)


@pytest.mark.parametrize("polarity", (1, -1))
@pytest.mark.parametrize("sec", ("h", "g"))
@pytest.mark.parametrize("topo", ("q4", "q4w"))
def test_e4pl_physical_reference_director_and_polarity_normals(topo, sec, polarity):
    el, mesh, loc = build("e4pl", topo, sec, polarity=polarity)
    on = recover(el, mesh, loc, True)
    off = recover(el, mesh, loc, False)
    assert "physical_reference_director" not in off
    director = np.asarray(on["physical_reference_director"], float)
    assert director.shape == (3,)
    assert np.allclose(director, polarity * N0, rtol=0, atol=1e-15)
    assert abs(np.linalg.norm(director) - 1.0) <= 1e-15
    if topo == "q4":
        for key in NEW_NORMAL_KEYS:
            assert np.allclose(on[key], director[None, :], rtol=0, atol=1e-14)
    else:
        assert np.array_equal(on["membrane_basis_normals"], on["physical_directors"])
        assert np.array_equal(on["bending_basis_normals"], on["physical_directors"])
        assert np.array_equal(
            on["shear_basis_normals"],
            np.repeat(np.asarray(on["physical_director"])[None], 4, axis=0),
        )


def test_e4pl_without_reference_normal_has_no_reference_director():
    el, mesh, loc = build("e4pl", "q4", "h", ref=False)
    res = recover(el, mesh, loc, True)
    assert "physical_reference_director" not in res
    assert "recovery_station_positions" in res


def test_e4pl_warped_without_reference_normal_uses_legacy_outputs():
    for sec in ("h", "g"):
        el, mesh, loc = build("e4pl", "q4w", sec, ref=False)
        res = recover(el, mesh, loc, True)
        for key in NEW_KEYS:
            assert key in res
        assert "physical_reference_director" not in res


# ----------------------------------------------------------- S3 / V2D metadata
@pytest.mark.parametrize("polarity", (1, -1))
@pytest.mark.parametrize("sec", ("h", "g"))
def test_qualified_s3_reports_director_polarity_and_owner_normal(sec, polarity):
    el, mesh, loc = build("s3", "t3", sec, polarity=polarity)
    on = recover(el, mesh, loc, True)
    off = recover(el, mesh, loc, False)
    assert "director_polarity" not in off
    assert on["director_polarity"] == polarity
    assert isinstance(on["director_polarity"], int)
    for key in NEW_NORMAL_KEYS:
        assert np.allclose(on[key], N0[None, :], rtol=0, atol=1e-14)


@pytest.mark.parametrize("sec", ("h", "g"))
def test_v2d_ordinary_case_reports_polarity_and_offsets(sec):
    el, mesh, loc = build("v2d", "t3", sec)
    on = recover(el, mesh, loc, True)
    off = recover(el, mesh, loc, False)
    assert on["director_polarity"] == 1
    assert isinstance(on["director_polarity"], int)
    assert on["reference_surface_offset"] == 0.0
    assert on["section_origin_offset_from_reference"] == (
        el.material_mid_surface_offset_from_reference
    )
    assert on["physical_bottom_offset_from_reference"] == (
        el.material_surface_offset_from_reference(-1.0)
    )
    assert on["physical_top_offset_from_reference"] == (
        el.material_surface_offset_from_reference(1.0)
    )
    if sec == "h":
        # ordinary homogeneous case: keys must not appear without the flag
        for key in (
            "director_polarity",
            "reference_surface_offset",
            "section_origin_offset_from_reference",
            "physical_bottom_offset_from_reference",
            "physical_top_offset_from_reference",
        ):
            assert key not in off, key
    for key in NEW_NORMAL_KEYS:
        assert np.allclose(on[key], on["frame"][:, 2][None, :], rtol=0, atol=1e-15)


def test_v2d_negative_polarity_and_offset_metadata_unchanged():
    el, mesh, loc = build("v2d", "t3", "h", polarity=-1)
    on = recover(el, mesh, loc, True)
    off = recover(el, mesh, loc, False)
    assert on["director_polarity"] == -1
    assert off["director_polarity"] == -1
    assert "recovery_station_positions" in on
    assert "recovery_station_positions" not in off


# ---------------------------------------------------- compiled S4 batch kernel
def test_compiled_s4_kernel_new_keys_match_scalar_oracle():
    """The batch kernel carries the new keys like the existing global_* keys.

    The kernel function itself runs here in its pure-Python fallback when
    numba is absent; the production dispatch (JIT_ENABLED) is not exercised.
    """

    from anyfesolver import generate_simple_panel_mesh
    from anyfesolver.recovery import _compute_one_element_stress
    from anyfesolver.recovery_batches import RecoveryBatchPlan
    from anyfesolver.recovery_s4 import recover_isotropic_s4

    model = generate_simple_panel_mesh(
        4.0, 2.0, 0.01, num_divisions_x=3, num_divisions_y=2
    )
    for node_id, node in tuple(model.mesh.nodes.items()):
        model.set_node_coordinates(
            node_id, node.x, node.y, 2.0e-3 * np.sin(0.7 * node.x + 0.4 * node.y)
        )
    rng = np.random.default_rng(7)
    displacement = rng.normal(scale=2.0e-5, size=model.mesh.dof_manager.total_dofs)
    plan = RecoveryBatchPlan.build(model)
    batch = plan.isotropic_s4
    assert batch is not None
    ids = list(model.mesh.elements)
    indices = batch.select_indices(ids)
    off = recover_isotropic_s4(batch, indices, displacement, return_global=False)
    on = recover_isotropic_s4(batch, indices, displacement, return_global=True)
    for element_id in ids:
        scalar = _compute_one_element_stress(
            model, displacement, element_id, return_global=True
        )[1]
        assert on[element_id].keys() == scalar.keys()
        assert not (set(NEW_KEYS) & set(off[element_id]))
        for key in NEW_KEYS:
            expected = np.asarray(scalar[key], float)
            got = np.asarray(on[element_id][key], float)
            assert got.shape == expected.shape
            scale = max(float(np.max(np.abs(expected))), 1.0e-300)
            assert np.max(np.abs(got - expected)) <= 1.0e-12 * scale, key


# ------------------------------------------------------------- golden (old code)
GOLDEN = pathlib.Path(__file__).parent / "data" / "shell_global_outputs_d2_golden.npz"


@pytest.mark.parametrize("branch", BRANCHES, ids=IDS)
@pytest.mark.parametrize("flag", (False, True))
def test_pre_existing_values_bit_identical_to_golden_from_base_code(branch, flag):
    """Golden arrays were written by the code before D2-1 (origin/main 0dea1cc0f)."""

    bid, kind, topo, sec, n_p = branch
    el, mesh, loc = build(kind, topo, sec)
    res = recover(el, mesh, loc, flag)
    golden = np.load(GOLDEN)
    prefix = f"{bid}|{int(flag)}|"
    keys = [k[len(prefix):] for k in golden.files if k.startswith(prefix)]
    assert keys, "golden has no entries for this branch"
    assert set(keys) <= set(res)
    for key in keys:
        # Cross-platform: the golden was written on one BLAS/libm; bit identity of
        # the flag on/off paths is asserted in the same process elsewhere.
        a, g = np.asarray(res[key]), golden[prefix + key]
        if a.dtype.kind in "fc":
            assert np.allclose(a, g, rtol=1.0e-12, atol=1.0e-12 * max(float(np.max(np.abs(g), initial=0.0)), 1.0e-300)), key
        else:
            assert np.array_equal(a, g), key
    if not flag:
        assert not (set(res) & set(NEW_KEYS))


def test_warped_homogeneous_resultants_scale_with_thickness_below_the_legacy_floor():
    """Codex r1 P2: M must be native (h^3 scaling) even for h < 1e-6, where the
    recovered bending stress used 6M/max(h^2, 1e-12)."""

    thin = 1.0e-7
    el, mesh, loc = build("e4pl", "q4w", "h")
    ref = recover(el, mesh, loc, True)
    el2, mesh2, loc2 = build("e4pl", "q4w", "h", thickness=thin)
    got = recover(el2, mesh2, loc2, True)
    ratio = thin / H
    for key, power in (
        ("global_membrane_resultant_tensors", 1),
        ("global_bending_resultant_tensors", 3),
    ):
        a = np.asarray(ref[key]) * ratio**power
        b = np.asarray(got[key])
        assert np.max(np.abs(a - b)) <= 1.0e-9 * np.max(np.abs(a)), key
