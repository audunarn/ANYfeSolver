"""Regression tests for the 2026-11-24 independent linear spot check.

1. MacNeal-Harder twisted beam on warped Q4 shells (drilling_stabilization).
2. BeamElement default local axes along a curved line without ``orientation``
   (characterised; the warning needs the sealed elements.py, owner decision).
3. Linear shell buckling: ``reference_displacements`` carry no prestress.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from anyfesolver import FEModel, LoadCase, recover_prestress_from_static_result, solve_linear
from anyfesolver.boundary import BoundaryCondition, FixedSupport
from anyfesolver.buckling import solve_eigenvalue_buckling
from anyfesolver.elements import BeamElement, create_shell_element

_DOF = ("ux", "uy", "uz", "rx", "ry", "rz")


def _solve(model, load_case):
    u, info = solve_linear(model, load_case)
    status = info["convergence_info"]["status"]
    if status != "converged":  # not an AssertionError: must not satisfy the xfail below
        raise RuntimeError(f"linear solve did not converge: {status}")
    return u


# --------------------------------------------------------------- finding 1
def _twisted_beam_tip(n, nw, dof, **shell_kw):
    """MacNeal-Harder (1985) twisted beam, L=12, w=1.1, t=0.32, 90 deg twist."""
    E, nu, L, W, t = 29.0e6, 0.22, 12.0, 1.1, 0.32
    model = FEModel("twisted_beam")
    model.add_material("mat", E, nu)

    def nid(i, j):
        return i * (nw + 1) + j + 1

    for i in range(n + 1):
        phi = np.pi / 2 * i / n
        for j in range(nw + 1):
            s = -W / 2 + W * j / nw
            model.add_node(nid(i, j), L * i / n, s * np.cos(phi), s * np.sin(phi))
    eid = 0
    for i in range(n):
        for j in range(nw):
            eid += 1
            model.add_element(eid, create_shell_element(
                eid, [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)],
                "mat", thickness=t, **shell_kw))
    model.add_boundary_condition(FixedSupport("root", [nid(0, j) for j in range(nw + 1)]))
    load_case = LoadCase("tip")
    weights = np.ones(nw + 1)
    weights[0] = weights[-1] = 0.5
    weights /= weights.sum()
    for j in range(nw + 1):
        force = [0.0] * 6
        force[dof] = float(weights[j])
        load_case.add_nodal_load(nid(n, j), force)
    u = _solve(model, load_case)
    return float(u[model.mesh.nodes[nid(n, nw // 2)].dofs][dof])


_TWISTED_REFERENCE = {2: 0.005424, 1: 0.001754}  # MacNeal & Harder 1985, Table 3


# Characterisation of the open finding: with the default drilling_stabilization=1e-3
# warped Q4 is +31.4 % (in-plane) / +23.7 % (out-of-plane) too flexible at 24x4 and
# the error does not vanish under refinement.  The contract change is the owner's
# decision; when it lands, replace these bands by the reference tolerance.
_DEFAULT_DRILLING_BAND = {2: (0.28, 0.35), 1: (0.20, 0.28)}


@pytest.mark.parametrize("dof", [2, 1], ids=["in_plane", "out_of_plane"])
def test_twisted_beam_default_drilling_known_discrepancy(dof):
    got = _twisted_beam_tip(24, 4, dof)
    low, high = _DEFAULT_DRILLING_BAND[dof]
    assert low < got / _TWISTED_REFERENCE[dof] - 1.0 < high


@pytest.mark.parametrize("dof", [2, 1], ids=["in_plane", "out_of_plane"])
def test_twisted_beam_with_physical_drilling_scale_matches_reference(dof):
    got = _twisted_beam_tip(24, 4, dof, drilling_stabilization=0.1)
    assert got == pytest.approx(_TWISTED_REFERENCE[dof], rel=0.03)


# --------------------------------------------------------------- finding 2
def _curved_beam(section, n=24):
    """MacNeal-Harder curved beam (R=4.12+0.1, 90 deg arc in the XY plane)."""
    E, nu, R = 1.0e7, 0.25, 4.22
    model = FEModel("curved_beam")
    model.add_material("mat", E, nu)
    for i in range(n + 1):
        s = np.pi / 2 * i / n
        model.add_node(i + 1, R * np.cos(s), R * np.sin(s), 0.0)
    for i in range(n):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "mat", dict(section)))
    model.add_boundary_condition(FixedSupport("root", [1]))
    load_case = LoadCase("tip")
    load_case.add_nodal_load(n + 1, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    u = _solve(model, load_case)
    return model, float(u[model.mesh.nodes[n + 1].dofs][2])


_W, _T = 0.2, 0.1
_STRIP = {"area": _W * _T, "Iy": _W * _T**3 / 12, "Iz": _T * _W**3 / 12, "J": 0.229 * _W * _T**3}
_CURVED_OUT_OF_PLANE_REFERENCE = 0.5022  # MacNeal & Harder 1985


def test_default_beam_axes_switch_by_90_degrees_near_global_y():
    """Characterisation of the documented heuristic (finding 2, owner decision open)."""
    from anyfesolver.elements import _beam_rotation_matrix

    near_y = np.array([-np.sin(0.2), np.cos(0.2), 0.0])  # |e1.Y| = 0.980 > 0.95
    near_x = np.array([-np.sin(1.4), np.cos(1.4), 0.0])  # |e1.Y| = 0.170
    z_near_y = _beam_rotation_matrix(near_y, None)[:, 2]
    z_near_x = _beam_rotation_matrix(near_x, None)[:, 2]
    assert abs(z_near_y[2]) < 1e-12  # local z in the XY plane
    assert abs(abs(z_near_x[2]) - 1.0) < 1e-12  # local z = +-global Z
    oriented = np.array([0.0, 0.0, 1.0])
    for e1 in (near_y, near_x):
        assert np.allclose(_beam_rotation_matrix(e1, oriented)[:, 2], oriented)


def test_curved_beam_needs_orientation_for_reference_result():
    _model, default = _curved_beam(_STRIP)
    _model, oriented = _curved_beam({**_STRIP, "orientation": [0.0, 0.0, 1.0]})
    assert oriented == pytest.approx(_CURVED_OUT_OF_PLANE_REFERENCE, rel=0.03)
    assert abs(default / _CURVED_OUT_OF_PLANE_REFERENCE - 1.0) > 0.10



# --------------------------------------------------------------- finding 3
def _ss_plate(n):
    E, nu, t, a = 210e9, 0.3, 0.01, 1.0
    model = FEModel("ss_plate")
    model.add_material("mat", E, nu)
    nodes = np.arange(1, (n + 1) ** 2 + 1).reshape(n + 1, n + 1)
    xs = np.linspace(0.0, a, n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            model.add_node(int(nodes[i, j]), xs[i], xs[j], 0.0)
    eid = 0
    for i in range(n):
        for j in range(n):
            eid += 1
            ids = [int(nodes[i, j]), int(nodes[i + 1, j]), int(nodes[i + 1, j + 1]), int(nodes[i, j + 1])]
            model.add_element(eid, create_shell_element(eid, ids, "mat", thickness=t))
    fixed = {}
    edge = set(nodes[0]) | set(nodes[-1]) | set(nodes[:, 0]) | set(nodes[:, -1])
    for k in nodes.ravel():
        fixed.setdefault(int(k), {})["rz"] = 0.0
    for k in edge:
        fixed[int(k)]["uz"] = 0.0
    for k in nodes[n // 2, :]:
        fixed[int(k)]["ux"] = 0.0
    for k in nodes[:, n // 2]:
        fixed[int(k)]["uy"] = 0.0
    for k, dofs in fixed.items():
        model.add_boundary_condition(BoundaryCondition(f"bc{k}", [k], dofs))
    load_case = LoadCase("nx")
    w = np.ones(n + 1)
    w[0] = w[-1] = 0.5
    w *= a / n
    for j in range(n + 1):
        load_case.add_nodal_load(int(nodes[0, j]), [w[j], 0, 0, 0, 0, 0])
        load_case.add_nodal_load(int(nodes[-1, j]), [-w[j], 0, 0, 0, 0, 0])
    u = _solve(model, load_case)
    reference = 4.0 * np.pi**2 * (E * t**3 / 12 / (1 - nu**2)) / a**2  # k = 4, Nx = 1 N/m
    return model, load_case, u, reference


def test_buckling_reference_displacements_alone_explain_missing_prestress():
    model, load_case, u, _reference = _ss_plate(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solve_eigenvalue_buckling(
            model, reference_load_case=load_case, reference_displacements=u, num_modes=1)
    assert result.solver_status == "zero_geometric_stiffness"
    hint = result.diagnostics.get("hint", "")
    assert "element_states" in hint and "recover_prestress_from_static_result" in hint


def test_ss_square_plate_uniaxial_buckling_k4_via_recovered_prestress():
    model, _load_case, u, reference = _ss_plate(16)
    states, _info = recover_prestress_from_static_result(model, u)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solve_eigenvalue_buckling(model, element_states=states, num_modes=2)
    assert result.solver_status == "ok"
    assert result.critical_load_factor == pytest.approx(reference, rel=0.01)
