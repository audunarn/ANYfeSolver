"""Warped qualified Q4 facets apply a drilling-stabilization floor of 0.1.

Owner decision (spot-check finding 1, option C): with the default
``drilling_stabilization=1e-3`` the varying-frame kernel used on warped
qualified Q4 facets made the MacNeal-Harder twisted beam +31 %/+24 % too
flexible at 24x4 and did not converge under refinement.  Warped facets now
apply ``max(alpha, 0.1)``; planar facets and every other element apply the
stored value unchanged, and the stored value is kept for serialization.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from anyfesolver import FEModel, LoadCase, create_shell_element, solve_linear
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement, equation7_frame
from anyfesolver.elements import LegacyQ4DeprecationWarning, LegacyShellElement
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.nonlinear_static import solve_static_nonlinear

DOF = {"ux": 0, "uy": 1, "uz": 2, "rx": 3, "ry": 4, "rz": 5}
# MacNeal-Harder (1985) twisted beam, t = 0.32, unit tip loads.
REFERENCE = {"uz": 0.005424, "uy": 0.001754}


def _twisted_beam(n, nw, *, alpha=None, flat_fraction=0.0, legacy=False):
    """Twisted strip; the first ``flat_fraction`` of the length is untwisted."""

    length, width, thickness = 12.0, 1.1, 0.32
    model = FEModel("twisted_beam")
    model.add_material("mat", 29.0e6, 0.22, density=1.0)
    ids = np.arange(1, (n + 1) * (nw + 1) + 1).reshape(n + 1, nw + 1)
    flat = flat_fraction * length
    for i in range(n + 1):
        x = length * i / n
        theta = 0.0 if x <= flat else 0.5 * math.pi * (x - flat) / (length - flat)
        for j in range(nw + 1):
            s = -0.5 * width + width * j / nw
            model.add_node(int(ids[i, j]), x, s * math.cos(theta), s * math.sin(theta))
    kwargs = {} if alpha is None else {"drilling_stabilization": alpha}
    element_id = 1
    for i in range(n):
        for j in range(nw):
            nodes = [int(ids[i, j]), int(ids[i + 1, j]), int(ids[i + 1, j + 1]), int(ids[i, j + 1])]
            if legacy:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", LegacyQ4DeprecationWarning)
                    element = create_shell_element(
                        element_id, nodes, "mat", formulation="legacy", thickness=thickness, **kwargs
                    )
            else:
                element = create_shell_element(element_id, nodes, "mat", thickness=thickness, **kwargs)
            model.add_element(element_id, element)
            element_id += 1
    model.add_boundary_condition(
        BoundaryCondition("root", [int(v) for v in ids[0]], {d: 0.0 for d in DOF})
    )
    return model, ids


def _tip_load(ids, dof, magnitude=1.0):
    load_case = LoadCase(f"tip_{dof}")
    weights = np.ones(ids.shape[1])
    weights[0] = weights[-1] = 0.5
    weights /= weights.sum()
    for node, weight in zip(ids[-1], weights):
        vector = np.zeros(6)
        vector[DOF[dof]] = magnitude * weight
        load_case.add_nodal_load(int(node), vector)
    return load_case


def _tip(model, ids, u, dof):
    node = int(ids[-1, ids.shape[1] // 2])
    return float(u[model.mesh.dof_manager.get_node_dofs(node)[DOF[dof]]])


def _linear_tip(n, nw, dof, *, alpha=None, magnitude=1.0):
    model, ids = _twisted_beam(n, nw, alpha=alpha)
    u, _info = solve_linear(model, _tip_load(ids, dof, magnitude))
    return _tip(model, ids, u, dof)


@pytest.mark.parametrize("dof", ["uz", "uy"])
def test_default_twisted_beam_24x4_is_within_three_percent(dof: str) -> None:
    # Old behaviour: +31.4 % (uz, in-plane) and +23.7 % (uy, out-of-plane).
    value = _linear_tip(24, 4, dof)
    assert abs(value / REFERENCE[dof] - 1.0) < 0.03


def test_explicit_small_alpha_is_floored_and_larger_alpha_is_honoured() -> None:
    model, _ids = _twisted_beam(2, 1, alpha=0.1)
    reference = model.mesh.elements[1].compute_stiffness_matrix(
        model.mesh, model.get_material("mat")
    )
    for alpha in (1.0e-3, 0.0, 0.05):
        other, _ = _twisted_beam(2, 1, alpha=alpha)
        element = other.mesh.elements[1]
        assert element.drilling_stabilization == alpha
        stiffness = element.compute_stiffness_matrix(other.mesh, other.get_material("mat"))
        assert np.array_equal(stiffness, reference), alpha
    honoured, _ = _twisted_beam(2, 1, alpha=1.0)
    element = honoured.mesh.elements[1]
    coordinates = element.get_node_coordinates(honoured.mesh)
    assert element._effective_drilling_stabilization(coordinates) == 1.0
    stiffness = element.compute_stiffness_matrix(honoured.mesh, honoured.get_material("mat"))
    assert not np.array_equal(stiffness, reference)


def test_floor_applies_only_to_warped_qualified_q4() -> None:
    from anyfesolver.elements import _kernel_drilling_stabilization

    model, _ = _twisted_beam(4, 1, flat_fraction=0.5)
    warped = planar = 0
    for element in model.mesh.elements.values():
        assert type(element) is QualifiedE4PLShellElement
        coordinates = element.get_node_coordinates(model.mesh)
        effective = element._effective_drilling_stabilization(coordinates)
        if equation7_frame(coordinates)[2] > element.planar_tolerance:
            warped += 1
            assert effective == 0.1
        else:
            planar += 1
            assert effective is element.drilling_stabilization
        assert _kernel_drilling_stabilization(element, model.mesh) == effective
    assert warped and planar
    legacy, _ = _twisted_beam(2, 1, legacy=True)
    for element in legacy.mesh.elements.values():
        assert type(element) is LegacyShellElement
        assert element._effective_drilling_stabilization(
            element.get_node_coordinates(legacy.mesh)
        ) == 1.0e-3
        assert _kernel_drilling_stabilization(element, legacy.mesh) == 1.0e-3


def test_serialization_keeps_the_raw_value() -> None:
    for alpha in (None, 1.0e-3, 0.0, 1.0):
        model, _ = _twisted_beam(2, 1, alpha=alpha)
        element = model.mesh.elements[1]
        expected = 1.0e-3 if alpha is None else alpha
        assert element.drilling_stabilization == expected
        assert element.to_dict()["drilling_stabilization"] == expected


# Planar-plate displacements recorded on the pre-floor code (b9b4a6e80).  The
# floor must not touch planar facets: any leak of 0.1 would move these far
# beyond the 1e-12 relative band.
# (The planar E4-PL drill is formed from the section, so both values coincide.)
_PLANAR_CORNER = (
    -1.538543785935065e-08, 7.140572467147668e-08, -0.00013610685178639428,
    -0.0001109519465816097, 0.0001940777805857692, 1.1039125949052686e-07,
)
PLANAR_PLATE_BASELINE = {1.0e-3: _PLANAR_CORNER, 0.1: _PLANAR_CORNER}


def _planar_plate(alpha):
    model = FEModel("planar_plate")
    model.add_material("mat", 210.0e9, 0.3, density=1.0)
    n = 4
    ids = np.arange(1, (n + 1) ** 2 + 1).reshape(n + 1, n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            x, y = i / n, j / n
            model.add_node(int(ids[i, j]), x, y + 0.1 * x * y, 0.0)
    element_id = 1
    for i in range(n):
        for j in range(n):
            nodes = [int(ids[i, j]), int(ids[i + 1, j]), int(ids[i + 1, j + 1]), int(ids[i, j + 1])]
            model.add_element(
                element_id,
                create_shell_element(
                    element_id, nodes, "mat", thickness=0.01, drilling_stabilization=alpha
                ),
            )
            element_id += 1
    model.add_boundary_condition(
        BoundaryCondition("edge", [int(v) for v in ids[0]], {d: 0.0 for d in DOF})
    )
    load_case = LoadCase("corner")
    load_case.add_nodal_load(int(ids[-1, -1]), np.array([10.0, 20.0, -5.0, 0.0, 0.0, 0.0]))
    u, _info = solve_linear(model, load_case)
    node = model.mesh.dof_manager.get_node_dofs(int(ids[-1, -1]))
    return model, np.array([u[node[k]] for k in range(6)])


@pytest.mark.parametrize("alpha", [1.0e-3, 0.1])
def test_planar_plate_is_unchanged(alpha: float) -> None:
    model, corner = _planar_plate(alpha)
    for element in model.mesh.elements.values():
        coordinates = element.get_node_coordinates(model.mesh)
        assert equation7_frame(coordinates)[2] <= element.planar_tolerance
    baseline = np.array(PLANAR_PLATE_BASELINE[alpha])
    np.testing.assert_allclose(corner, baseline, rtol=1.0e-12, atol=0.0)


def _zero_state_tangent(model, assembler):
    total = model.mesh.dof_manager.total_dofs
    _force, tangent, _states = assembler(model, np.zeros(total), {}, 5, tangent=True)
    return tangent.toarray() if hasattr(tangent, "toarray") else np.asarray(tangent)


def _element_tangent_sum(model):
    """Scalar element zero-state tangents scattered into the global matrix."""

    total = model.mesh.dof_manager.total_dofs
    result = np.zeros((total, total))
    material = model.get_material("mat")
    for element in model.mesh.elements.values():
        _force, tangent, _state = element.compute_nonlinear_response(
            model.mesh, material, np.zeros(element.total_dofs), None, 5, True
        )
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=int)
        result[np.ix_(dofs, dofs)] += tangent
    return result


# The von Karman route keeps the legacy centre-frame nonlinear kernel on warped
# facets, whose zero-state tangent is NOT the varying-frame linear stiffness
# (a single twisted-beam element differs by about 42 % before and after this
# change).  The floor therefore cannot be checked as "nonlinear tangent ==
# linear stiffness"; instead every nonlinear route must apply exactly the
# coefficient its scalar kernel applies, i.e. default == explicit 0.1 on warped
# facets, unchanged on planar ones.
@pytest.mark.parametrize("route", ["reference", "accelerated"])
def test_nonlinear_routes_apply_the_floored_coefficient(route: str) -> None:
    from anyfesolver import nonlinear_static

    nonlinear_static._ensure_nonlinear_acceleration()
    if route == "reference":
        # The scalar/vectorized grouping of nonlinear_static itself.
        assembler = nonlinear_static._assemble_nonlinear_system_unchecked
    else:
        # The installed performance layer (assembly plan, batch B/C kernels).
        assembler = nonlinear_static._assemble_nonlinear_system

    def tangent_of(model):
        return _zero_state_tangent(model, assembler)

    def assert_close(actual, expected):
        scale = float(np.max(np.abs(expected)))
        assert np.max(np.abs(actual - expected)) <= 1.0e-12 * scale

    # Mixed mesh: planar and warped Q4 with the same raw alpha must be batched
    # by the applied coefficient and agree with the scalar element kernels.
    mixed, _ = _twisted_beam(6, 2, flat_fraction=0.5)
    assert_close(tangent_of(mixed), _element_tangent_sum(mixed))
    # Warped facets: default alpha behaves exactly as an explicit 0.1.
    warped_default, _ = _twisted_beam(4, 1)
    warped_explicit, _ = _twisted_beam(4, 1, alpha=0.1)
    default_tangent = tangent_of(warped_default)
    assert_close(default_tangent, tangent_of(warped_explicit))
    assert_close(_element_tangent_sum(warped_default), _element_tangent_sum(warped_explicit))
    honoured, _ = _twisted_beam(4, 1, alpha=1.0)
    assert not np.allclose(tangent_of(honoured), default_tangent, rtol=1.0e-9, atol=0.0)


@pytest.mark.parametrize("dof", ["uz", "uy"])
def test_small_load_nonlinear_static_uses_the_floored_coefficient(dof: str) -> None:
    def tip(alpha):
        model, ids = _twisted_beam(12, 2, alpha=alpha)
        result = solve_static_nonlinear(
            model, _tip_load(ids, dof, 1.0e-2), num_steps=1, max_iterations=20,
            tolerance=1.0e-10,
        )
        assert result.status == "completed"
        return _tip(model, ids, result.displacements, dof)

    default = tip(None)
    assert default == pytest.approx(tip(0.1), rel=1.0e-9, abs=0.0)
    assert default != pytest.approx(tip(1.0e-3 * 10.0 ** 3), rel=1.0e-6, abs=0.0)
