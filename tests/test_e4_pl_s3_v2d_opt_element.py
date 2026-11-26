"""Opt-in V2D-OPT triangle: Felippa OPT membrane + V2D plate action, linear route only."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from anyfesolver import (
    FEModel,
    LoadCase,
    NativeParityE4PLS3V2DShellElement,
    create_shell_element,
    shell_formulation_diagnostics,
    solve_linear,
)
from anyfesolver.assembly import compute_stresses
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.e4_pl_s3_v2d_opt_element import NativeParityE4PLS3V2DOptShellElement as OPT
from anyfesolver.element_capabilities import ElementCapabilityError

NORMAL = (0.0, 0.0, 1.0)
E, NU = 1.0e7, 0.3
OPT_KIND = "e4-pl-s3-v2d-opt"


def _make(kind, eid, nodes, **kw):
    """OPT is constructed directly (registry exposure needs a sealed elements.py edit: separate step)."""
    if kind == OPT_KIND:
        return OPT(eid, list(nodes), "mat", **kw)
    return create_shell_element(eid, list(nodes), "mat", **kw)


FIX6 = {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}


def _single(kind, xyz, t=0.1, normal=NORMAL):
    m = FEModel("one")
    m.add_material("mat", E, NU)
    for i, p in enumerate(xyz, 1):
        m.add_node(i, *map(float, p))
    el = _make(kind, 1, [1, 2, 3], thickness=t, reference_normal=normal)
    m.add_element(1, el)
    return m, el


def _strip(kind, n, nw=1, length=6.0, width=0.2, t=0.1, nu=NU):
    m = FEModel("strip")
    m.add_material("mat", E, nu)
    nid = {}
    for i in range(n + 1):
        for j in range(nw + 1):
            nid[i, j] = len(nid) + 1
            m.add_node(nid[i, j], length * i / n, -width / 2 + width * j / nw, 0.0)
    e = c = 0
    for i in range(n):
        for j in range(nw):
            c += 1
            a, b, cc, d = nid[i, j], nid[i + 1, j], nid[i + 1, j + 1], nid[i, j + 1]
            for tri in ([(a, b, cc), (a, cc, d)] if c % 2 else [(a, b, d), (b, cc, d)]):
                e += 1
                m.add_element(e, _make(kind, e, tri, thickness=t, reference_normal=NORMAL))
    for j in range(nw + 1):
        m.add_boundary_condition(BoundaryCondition(f"c{j}", [nid[0, j]], dict(FIX6)))
    for key, node in nid.items():
        m.add_boundary_condition(BoundaryCondition(f"z{key[0]}_{key[1]}", [node], {"uz": 0.0, "rx": 0.0, "ry": 0.0}))
    return m, nid


def _tip_ux_uy(m, nid, n, nw, loads):
    lc = LoadCase("l")
    for node, vec in loads:
        lc.add_nodal_load(node, vec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, info = solve_linear(m, lc)
    assert info["convergence_info"]["status"] == "converged"
    return u, [u[m.mesh.nodes[nid[n, j]].dofs] for j in range(nw + 1)]


def test_round_trip_exact_class_and_default_untouched():
    el = OPT(7, [1, 2, 3], "mat", thickness=0.1, reference_normal=NORMAL)
    assert type(el) is OPT and el.formulation_id == "CANDIDATE_E4_PL_S3_V2D_OPT_V1"
    back = OPT.from_dict(el.to_dict())
    assert type(back) is OPT and back.to_dict() == el.to_dict()
    default = create_shell_element(8, [1, 2, 3], "mat", thickness=0.1, reference_normal=NORMAL)
    assert type(default) is NativeParityE4PLS3V2DShellElement          # default selection unchanged
    assert shell_formulation_diagnostics(node_count=3)["selected_formulation"] == "e4-pl-s3-v2d"
    with pytest.raises(ValueError):  # OPT is not (yet) selectable through the shared factory
        create_shell_element(9, [1, 2, 3], "mat", thickness=0.1, reference_normal=NORMAL, formulation="e4-pl-s3-v2d-opt")
    bad = el.to_dict()
    bad["formulation_id"] = "other"
    with pytest.raises(ValueError):
        OPT.from_dict(bad)
    bad = el.to_dict()
    bad["extra"] = 1
    with pytest.raises(ValueError):
        OPT.from_dict(bad)
    with pytest.raises(ValueError):  # non-unit normal is refused at first use
        OPT(9, [1, 2, 3], "mat", thickness=0.1, reference_normal=(0.0, 0.0, 2.0))._validate_configuration()


def test_stiffness_rank_symmetry_and_rigid_body_modes_on_a_tilted_triangle():
    th = 0.4
    r = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    xyz = (r @ np.array([[0.1, 0.05, 0.0], [1.3, 0.4, 0.0], [0.5, 1.1, 0.0]]).T).T + [2.0, -1.0, 3.0]
    normal = tuple(r @ np.array(NORMAL))
    m, el = _single(OPT_KIND, xyz, normal=normal)
    mat = m.materials["mat"]
    k = el.compute_stiffness_matrix(m.mesh, mat)
    assert np.allclose(k, k.T, rtol=0, atol=1e-9 * np.abs(k).max())
    ev = np.linalg.eigvalsh(k)
    assert int(np.sum(ev < 1e-9 * ev[-1])) == 6 and ev[0] > -1e-9 * ev[-1]
    for axis in np.eye(3):  # translations and rotations about the global axes (small)
        trans = np.concatenate([np.r_[axis * 1e-3, 0, 0, 0] for _ in range(3)])
        rot = np.concatenate([np.r_[np.cross(axis * 1e-3, p), axis * 1e-3] for p in xyz])
        for mode in (trans, rot):
            assert np.linalg.norm(k @ mode) < 1e-9 * ev[-1] * np.linalg.norm(mode)


def test_beta0_and_membrane_block_are_the_opt_kernel_not_cst_plus_drill():
    m, el = _single(OPT_KIND, [[0.1, 0.05, 0], [1.3, 0.4, 0], [0.5, 1.1, 0]])
    comp = el.compute_stiffness_components(m.mesh, m.materials["mat"])
    assert comp["membrane_beta0"] == pytest.approx(0.5 * (1 - 4 * NU * NU))
    assert not np.any(comp["pl"])
    d = create_shell_element(1, [1, 2, 3], "mat", thickness=0.1, reference_normal=NORMAL)
    m2 = FEModel("d")
    m2.add_material("mat", E, NU)
    for i, p in enumerate([[0.1, 0.05, 0], [1.3, 0.4, 0], [0.5, 1.1, 0]], 1):
        m2.add_node(i, *map(float, p))
    m2.add_element(1, d)
    kd = d.compute_stiffness_matrix(m2.mesh, m2.materials["mat"])
    assert np.abs(comp["total"] - kd).max() > 1e-3 * np.abs(kd).max()  # membrane block differs
    # plate action (w, rx, ry) is the V2D one (native path vs the default's V2C wrapper: equal to round-off)
    plate = np.ix_([2, 3, 4, 8, 9, 10, 14, 15, 16], [2, 3, 4, 8, 9, 10, 14, 15, 16])
    assert np.allclose(comp["total"][plate], kd[plate], rtol=1e-12, atol=1e-12 * np.abs(kd).max())


@pytest.mark.parametrize("n,nw,expect_pct,tol", [(6, 1, 6.5, 0.2), (30, 2, 0.9, 0.2)])
def test_macneal_beam_shear_is_the_measured_opt_value_and_default_is_far_off(n, nw, expect_pct, tol):
    ref = 0.1081
    out = {}
    for kind in (OPT_KIND, "e4-pl-s3-v2d"):
        m, nid = _strip(kind, n, nw)
        loads = [(nid[n, j], [0.0, (0.5 if j in (0, nw) else 1.0) / nw, 0, 0, 0, 0]) for j in range(nw + 1)]
        _, tips = _tip_ux_uy(m, nid, n, nw, loads)
        out[kind] = 100.0 * (np.mean([t[1] for t in tips]) / ref - 1.0)
    assert out[OPT_KIND] == pytest.approx(expect_pct, abs=tol)   # measured (beta0 = 0.32 at nu = 0.3)
    assert out["e4-pl-s3-v2d"] < -50.0                                    # default CST+PL control locks


@pytest.mark.parametrize("n,nw", [(6, 1), (12, 2)])
def test_pure_end_moment_is_nearly_exact_at_nu_zero(n, nw):
    length, width, t = 6.0, 0.2, 0.1
    m, nid = _strip(OPT_KIND, n, nw, nu=0.0)
    inertia = t * width**3 / 12
    loads = []
    ys = [-width / 2 + width * j / nw for j in range(nw + 1)]
    f = np.zeros(nw + 1)
    for j in range(nw):
        y1, y2 = ys[j], ys[j + 1]
        s1, s2 = -y1 / inertia * t, -y2 / inertia * t
        f[j] += (y2 - y1) * (2 * s1 + s2) / 6
        f[j + 1] += (y2 - y1) * (s1 + 2 * s2) / 6
    loads = [(nid[n, j], [f[j], 0.0, 0.0, 0, 0, 0]) for j in range(nw + 1)]
    _, tips = _tip_ux_uy(m, nid, n, nw, loads)
    expected = length**2 / (2 * E * inertia)   # tip uy = M L^2 / (2 E I), M = 1
    # consistent nodal loads are not the paper's energy-balanced EBZ lumping: measured residual < 2e-4
    assert np.mean([tip[1] for tip in tips]) == pytest.approx(expected, rel=5e-4)


def test_constant_stress_patch_and_recovery_are_exact_on_an_irregular_patch():
    pts = np.array([[0, 0], [0.24, 0], [0.24, 0.12], [0, 0.12], [0.04, 0.02], [0.18, 0.03], [0.16, 0.08], [0.08, 0.08]])
    quads = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
    m = FEModel("patch")
    m.add_material("mat", E, 0.25)
    for i, (x, y) in enumerate(pts, 1):
        m.add_node(i, float(x), float(y), 0.0)
    e = 0
    for k, (a, b, c, d) in enumerate(quads):
        for tri in ([(a, b, c), (a, c, d)] if k % 2 == 0 else [(a, b, d), (b, c, d)]):
            e += 1
            m.add_element(e, OPT(e, [i + 1 for i in tri], "mat", thickness=0.01, reference_normal=NORMAL))
    exx, eyy, gxy = 1e-3, -3e-4, 6e-4
    prescribed = {}
    for i, (x, y) in enumerate(pts, 1):
        if i <= 4:
            prescribed[i] = {"ux": exx * x + 0.5 * gxy * y, "uy": eyy * y + 0.5 * gxy * x, "uz": 0.0,
                             "rx": 0.0, "ry": 0.0, "rz": 0.5 * 0.0}
    for i in range(1, 9):
        dofs = {"uz": 0.0, "rx": 0.0, "ry": 0.0}
        dofs.update(prescribed.get(i, {}))
        m.add_boundary_condition(BoundaryCondition(f"b{i}", [i], dofs))
    lc = LoadCase("l")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, info = solve_linear(m, lc)
    assert info["convergence_info"]["status"] == "converged"
    for i, (x, y) in enumerate(pts, 1):
        d = u[m.mesh.nodes[i].dofs]
        assert d[0] == pytest.approx(exx * x + 0.5 * gxy * y, abs=1e-12)
        assert d[1] == pytest.approx(eyy * y + 0.5 * gxy * x, abs=1e-12)
    stresses = compute_stresses(m, u, return_global=True)
    ex = E / (1 - 0.25**2) * np.array([exx + 0.25 * eyy, eyy + 0.25 * exx, (1 - 0.25) / 2 * gxy])
    expected = np.array([[ex[0], ex[2], 0.0], [ex[2], ex[1], 0.0], [0.0, 0.0, 0.0]])
    for result in stresses.values():
        tensors = np.asarray(result["global_membrane_resultant_tensors"]) / 0.01
        for tensor in tensors:
            assert np.allclose(tensor, expected, rtol=1e-9, atol=1e-6)
        assert result["formulation_id"] == "CANDIDATE_E4_PL_S3_V2D_OPT_V1"


def test_refusals_fail_closed_with_typed_errors():
    m, el = _single(OPT_KIND, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mat = m.materials["mat"]
    for call in (
        lambda: el.compute_mass_matrix(m.mesh, mat),
        lambda: el.compute_geometric_stiffness_matrix(m.mesh, mat, np.zeros(18)),
        lambda: el.compute_nonlinear_response(m.mesh, mat),
        lambda: el.init_nonlinear_state(1),
    ):
        with pytest.raises(ElementCapabilityError):
            call()
    with pytest.raises(ElementCapabilityError):
        OPT(2, [1, 2, 3], "mat", thickness=0.1, reference_normal=NORMAL, director_polarity=-1)
    with pytest.raises(ElementCapabilityError):
        OPT(2, [1, 2, 3], "mat", thickness=0.1, reference_normal=NORMAL, reference_surface_offset=0.01)
    with pytest.raises(ElementCapabilityError):
        OPT(2, [1, 2, 3], "mat", thickness=0.1, reference_normal=NORMAL, shell_section=object())
    # mixed shell scope: an OPT triangle beside a default V2D triangle is refused (both directions)
    m.add_node(4, 1.0, 1.0, 0.0)
    other = create_shell_element(2, [2, 4, 3], "mat", thickness=0.1, reference_normal=NORMAL)
    m.add_element(2, other)
    with pytest.raises(ElementCapabilityError):
        el.compute_stiffness_matrix(m.mesh, mat)
    with pytest.raises(ElementCapabilityError):
        other.compute_stiffness_matrix(m.mesh, mat)
    # subclasses are refused
    class Sub(OPT):
        pass
    sub = Sub(5, [1, 2, 3], "mat", thickness=0.1, reference_normal=NORMAL)
    with pytest.raises(ElementCapabilityError):
        sub._validate_configuration()


def test_dead_pressure_load_matches_the_default_triangle():
    xyz = [[0.1, 0.05, 0], [1.3, 0.4, 0], [0.5, 1.1, 0]]
    mo, eo = _single(OPT_KIND, xyz)
    md, ed = _single("e4-pl-s3-v2d", xyz)
    assert np.array_equal(eo.compute_dead_transverse_pressure_load(mo.mesh, 3.0),
                          ed.compute_dead_transverse_pressure_load(md.mesh, 3.0))
