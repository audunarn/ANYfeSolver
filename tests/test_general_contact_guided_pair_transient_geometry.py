"""Development checks of fixed-overlap motion and represented cubic bounds."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.core import (
    ContactDefinition, ContactInteraction, ContactSolverSettings, ContactSurface,
)
from anyfesolver.general_contact.pair import _PlanarShellPairManager
from anyfesolver.general_contact.guided_pair_transient_geometry import (
    guided_shell_pair_path_minimum_gap, prepare_guided_shell_pair_transient,
)


def _pair(scale=1.0, triangle=False):
    model = FEModel("guided_pair_geometry")
    model.add_material("elastic", 1e5, 0.3, density=1.0)
    square = ((0, 0), (1, 0), (1, 1), (0, 1))
    upper = ((0, 0), (1, 0), (0, 1)) if triangle else square
    for node, (x, y, z) in enumerate(
            (*((x, y, 0) for x, y in square),
             *((x, y, 0.1) for x, y in upper)), 1):
        model.add_node(node, scale*x, scale*y, scale*z)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.02*scale,
        reference_normal=(0, 0, 1)))
    nodes = tuple(range(5, 5+len(upper)))
    kwargs = {"formulation": "e4-pl-s3-v2d"} if triangle else {}
    model.add_element(2, create_shell_element(
        2, nodes, "elastic", thickness=0.02*scale,
        reference_normal=(0, 0, 1), **kwargs))
    definition = ContactDefinition(
        surfaces=(ContactSurface("upper", (2,), "bottom"),
                  ContactSurface("lower", (1,), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction("pair", "upper", "lower", 1e3, 1e-6),))
    manager = _PlanarShellPairManager(model, definition)
    T = np.zeros((model.mesh.dof_manager.total_dofs, 2))
    for column, selected in enumerate((nodes, (1, 2, 3, 4))):
        for node in selected:
            T[manager.node_dofs[node][2], column] = 1.0
    return manager, T


@pytest.mark.parametrize("triangle", (False, True))
@pytest.mark.parametrize("scale", (1e-3, 1.0, 1e3))
def test_whole_subspace_admission_is_coordinate_and_column_scale_independent(triangle, scale):
    manager, T = _pair(scale, triangle)
    features = prepare_guided_shell_pair_transient(manager, sparse.csr_matrix(T))
    assert features == tuple(("pair", node) for node in manager.integration_nodes)
    for factor in (1e-200, 1e200):
        assert prepare_guided_shell_pair_transient(manager, T*factor) == features
    # A common tangent mode is also permitted by the legacy pair motion contract.
    common = np.zeros(T.shape[0])
    for dofs in manager.node_dofs.values():
        common[dofs[0]] = 1.0
    assert prepare_guided_shell_pair_transient(manager, np.column_stack((T, common))) == features


@pytest.mark.parametrize("kind", ("normal_deformation", "rotation", "tangent_difference"))
@pytest.mark.parametrize("factor", (1e-200, 1.0, 1e200))
def test_nonadmitted_constraint_mode_cannot_hide_in_a_small_column(kind, factor):
    manager, T = _pair(scale=1e3)
    bad = np.zeros(T.shape[0])
    dofs = manager.node_dofs[manager.integration_nodes[0]]
    bad[dofs[{"normal_deformation": 2, "rotation": 3, "tangent_difference": 0}[kind]]] = factor
    with pytest.raises(ValueError, match="uniform normal|fixed shell rotations"):
        prepare_guided_shell_pair_transient(manager, np.column_stack((T, bad)))


@pytest.mark.parametrize("invalid", (None, "empty", "zero", "nan", "shape", "complex", "sparse_zero"))
def test_unrepresented_or_nonfinite_constraint_subspace_is_rejected(invalid):
    manager, T = _pair()
    values = {None: None, "empty": T[:, :0], "zero": T*0,
              "nan": T*np.nan, "shape": T[:-1], "complex": T.astype(complex),
              "sparse_zero": sparse.coo_matrix(([1., -1.], ([2, 2], [0, 0])),
                                               shape=(T.shape[0], 1))}
    with pytest.raises(ValueError, match="constraint subspace"):
        prepare_guided_shell_pair_transient(manager, values[invalid])


def test_every_positive_pressure_support_counts_toward_cap_and_stable_identity():
    manager, T = _pair()
    manager.support[-1] = np.nextafter(0.0, 1.0)
    manager.settings = replace(manager.settings, maximum_patch_normal_dofs=4)
    assert prepare_guided_shell_pair_transient(manager, T)[-1] == ("pair", 8)
    manager.settings = replace(manager.settings, maximum_patch_normal_dofs=3)
    with pytest.raises(MemoryError, match="no positive support was discarded"):
        prepare_guided_shell_pair_transient(manager, T)


@pytest.mark.parametrize("triangle", (False, True))
def test_two_moving_surfaces_have_an_interior_crossing_despite_separated_endpoints(triangle):
    manager, T = _pair(triangle=triangle)
    prepare_guided_shell_pair_transient(manager, T)
    u = T @ np.array((0.0, 0.0))
    v = T @ np.array((-1.0, 0.6))
    # Independent scalar path: g(s)=0.08-0.8*s+0.8*s**2 for dt=0.5.
    assert manager._minimum_gap(u) == pytest.approx(0.08)
    minimum = guided_shell_pair_path_minimum_gap(manager, u, u, v, -v, 0.5)
    assert minimum <= -0.12
    assert minimum == pytest.approx(-0.12, abs=1e-14)


@pytest.mark.parametrize("data", (
    ((0.01, -0.02), (0.06, 0.01), (-0.7, 0.2), (0.3, -0.4), 0.3),
    ((0.0, 0.0), (-0.1, 0.05), (0.0, 0.0), (0.0, 0.0), 0.4),
    ((0.0, 0.0), (0.0, 0.0), (-1.0, 0.6), (1.0, -0.6), 0.5),
))
def test_complete_path_bound_matches_independent_scalar_polynomial_extrema(data):
    manager, T = _pair()
    q0, q1, w0, w1, dt = map(np.asarray, data)
    g0, g1 = 0.08+q0[0]-q0[1], 0.08+q1[0]-q1[1]
    r0, r1 = w0[0]-w0[1], w1[0]-w1[1]
    coefficients = np.array((g0, dt*r0, 3*(g1-g0)-dt*(2*r0+r1),
                             2*(g0-g1)+dt*(r0+r1)))
    roots = np.polynomial.polynomial.polyroots(coefficients[1:]*np.arange(1, 4))
    points = [0.0, 1.0] + [float(r.real) for r in roots if abs(r.imag) < 1e-13 and 0 < r.real < 1]
    expected = min(np.polynomial.polynomial.polyval(s, coefficients) for s in points)
    minimum = guided_shell_pair_path_minimum_gap(manager, T@q0, T@q1, T@w0, T@w1, float(dt))
    assert minimum == pytest.approx(expected, abs=2e-14)


@pytest.mark.parametrize("argument", (0, 1, 2, 3))
def test_path_rejects_nonadmitted_endpoint_or_velocity_before_bounding(argument):
    manager, T = _pair()
    values = [T@np.zeros(2) for _ in range(4)]
    values[argument][manager.node_dofs[5][2]] = 1e-30
    with pytest.raises(ValueError, match="uniform normal"):
        guided_shell_pair_path_minimum_gap(manager, *values, 0.5)


@pytest.mark.parametrize("invalid", ("nan", "complex", "shape"))
def test_path_rejects_nonfinite_or_malformed_vectors(invalid):
    manager, T = _pair()
    zero = T@np.zeros(2)
    values = {"nan": zero+np.nan, "complex": zero.astype(complex), "shape": zero[:-1]}
    with pytest.raises(ValueError, match="finite full-DOF"):
        guided_shell_pair_path_minimum_gap(manager, zero, zero, values[invalid], zero, 0.5)


@pytest.mark.parametrize("argument", (0, 2))
def test_generator_cancellation_cannot_hide_an_inadmitted_actual_path_vector(argument):
    manager, T = _pair()
    generator = T[:, 0]
    perturbed = generator.copy()
    perturbed[manager.node_dofs[5][2]] += 1e-14
    represented = np.column_stack((generator, perturbed))
    # Each normalized generator is inside the floating epsilon contract, but
    # their cancellation exposes a nonuniform normal field. Admission of the
    # generators does not exempt any actual displacement or velocity from checks.
    prepare_guided_shell_pair_transient(manager, represented)
    values = [np.zeros(T.shape[0]) for _ in range(4)]
    values[argument] = represented@np.array((-1.0, 1.0))
    with pytest.raises(ValueError, match="uniform normal"):
        guided_shell_pair_path_minimum_gap(manager, *values, 0.5)


def test_physical_samples_are_bounded_even_when_pressure_averages_are_separated():
    # Analytic field fixture independent of the production gap oracle.
    manager = SimpleNamespace(
        model=SimpleNamespace(mesh=SimpleNamespace(dof_manager=SimpleNamespace(total_dofs=12))),
        normal=np.array((0., 0., 1.)),
        motion_faces=[np.arange(6).reshape(1, 6), np.arange(6, 12).reshape(1, 6)],
        support=np.ones(1), pressure_nodes=(1,),
        settings=ContactSolverSettings(), reference_weighted_gap=np.array((1.,)),
        gap_gradient=sparse.csr_matrix(([1., -1.], ([0, 0], [2, 8])), shape=(1, 12)),
        samples=[(0.1, np.ones(1), np.ones(1), np.arange(6).reshape(1, 6),
                  np.arange(6, 12).reshape(1, 6), np.array((0, 0, 1, 0, 0, 0)),
                  np.array((0, 0, 1, 0, 0, 0)))])
    zero = np.zeros(12)
    v = zero.copy()
    v[2], v[8] = -1., 1.
    assert guided_shell_pair_path_minimum_gap(manager, zero, zero, v, -v, 1.) == pytest.approx(-0.4)
    class Cancelled(Exception):
        pass
    def cancel():
        raise Cancelled
    with pytest.raises(Cancelled):
        guided_shell_pair_path_minimum_gap(manager, zero, zero, v, -v, 1., cancel)
