"""Physical overlap-region pressure work: agreement with the owner's reference-region integrator and exact fixtures."""
import numpy as np
import pytest

from anysolver.elements import create_shell_element
from anysolver.fe_core import FEModel
from anysolver.floating_pressure_regions import full_element_vectors, region_nodal_vectors
from anysolver.pressure_regions import PressureRegionError, region_wrench, wet_region_load

RHO_G = 1000.0


def build(points):
    model = FEModel('floating-region')
    model.add_material('m', 210e9, .3)
    for index, point in enumerate(points, 1):
        model.add_node(index, *point)
    normal = np.cross(np.asarray(points[1])-points[0], np.asarray(points[2])-points[0])
    normal /= np.linalg.norm(normal)
    element = create_shell_element(1, tuple(range(1, len(points) + 1)), 'm', thickness=.01,
                                   formulation='e4-pl' if len(points) == 4 else 'legacy',
                                   **({'reference_normal': normal} if len(points) == 4 else {}))
    model.add_element(1, element)
    return model.mesh, element


def fan(poly):
    p = np.asarray(poly, float)
    return np.array([p[[0, k, k + 1]] for k in range(1, len(p) - 1)])


def hydro(pts, ti):
    return RHO_G * (-pts[:, 2])


def test_partial_triangle_matches_owner_reference_region_integrator_all_components():
    corners = [(0., 0., -1.), (1., 0., -1.), (0., 0., 1.)]
    mesh, element = build(corners)
    level = np.array([1.0, 1.0, -1.0]); pressure = RHO_G * np.array([1.0, 1.0, -1.0])
    reference = wet_region_load(element, mesh, level, pressure)
    wet = [(0, 0, -1), (1, 0, -1), (.5, 0, 0), (0, 0, 0)]                      # physical z<=0 part
    mine = region_nodal_vectors(element, mesh, fan(wet), hydro)
    np.testing.assert_allclose(mine.vectors, reference.vectors, rtol=1e-10, atol=1e-9)
    assert mine.area == pytest.approx(0.75, rel=1e-12)
    wrench = region_wrench(reference, np.array(corners))
    assert wrench[3] == pytest.approx(-RHO_G * 7 / 24, rel=1e-10)


def test_sub_region_of_a_trapezoid_quad_is_force_and_moment_exact_without_averaging():
    corners = [(0., 0., 0.), (4., 0., 0.), (3., 0., 2.), (1., 0., 2.)]       # non-affine (trapezoid) quad map, plane y=0
    mesh, element = build(corners)
    full, area = full_element_vectors(element, mesh, 1.0)
    unit = full.sum(axis=0)[:3] / area                                          # force per unit area per unit pressure
    tri = np.array([[(0.5, 0., 0.2), (2.5, 0., 0.4), (1.5, 0., 1.6)]])
    p = 7.0
    mine = region_nodal_vectors(element, mesh, tri, lambda pts, ti: np.full(len(pts), p))
    area_sub = 0.5 * np.linalg.norm(np.cross(tri[0, 1] - tri[0, 0], tri[0, 2] - tri[0, 0]))
    np.testing.assert_allclose(mine.vectors[:, :3].sum(axis=0), p * area_sub * unit, rtol=1e-10, atol=1e-10)
    centroid = tri[0].mean(axis=0)
    wrench = region_wrench(type('L', (), dict(forces=mine.vectors[:, :3], moments=mine.vectors[:, 3:]))(), np.array(corners))
    np.testing.assert_allclose(wrench[3:], np.cross(centroid, p * area_sub * unit), rtol=1e-9, atol=1e-9)
    # the loaded fraction is the region's share of the element, not the whole element
    whole, _ = full_element_vectors(element, mesh, p)
    assert np.linalg.norm(mine.vectors[:, :3].sum(axis=0)) == pytest.approx(
        np.linalg.norm(whole[:, :3].sum(axis=0)) * area_sub / area, rel=1e-10)


def test_regions_tiling_the_element_reproduce_the_whole_element_load():
    corners = [(0., 0., 0.), (2., 0., 0.), (2., 0., 2.), (0., 0., 2.)]
    mesh, element = build(corners)
    tiles = np.array([[(0, 0, 0), (2, 0, 0), (2, 0, 2)], [(0, 0, 0), (2, 0, 2), (0, 0, 2)]], float)
    mine = region_nodal_vectors(element, mesh, tiles, lambda pts, ti: np.full(len(pts), 3.0))
    whole, area = full_element_vectors(element, mesh, 3.0)
    np.testing.assert_allclose(mine.vectors, whole, rtol=1e-10, atol=1e-9)
    assert mine.area == pytest.approx(area)


def test_owner_kernel_equivalence_proof_gates_the_region_integration_and_is_reused_only_for_identical_elements(monkeypatch):
    import anysolver.floating_pressure_regions as module
    corners = [(0., 0., 0.), (2., 0., 0.), (2., 0., 2.), (0., 0., 2.)]
    mesh, element = build(corners)
    tri = np.array([[(0, 0, 0), (1, 0, 0), (0, 0, 1)]], float)
    calls = []
    real = module.element_kernel
    monkeypatch.setattr(module, 'element_kernel', lambda e, m: calls.append(id(e)) or real(e, m))
    one = lambda pts, ti: np.full(len(pts), 1.0)
    first = module.region_nodal_vectors(element, mesh, tri, one)
    module.region_nodal_vectors(element, mesh, tri, one)
    assert len(calls) == 2                                                    # no proof dict: the owner proof runs on every call
    proof = {}
    again = module.region_nodal_vectors(element, mesh, tri, one, kernel_proof=proof)
    module.region_nodal_vectors(element, mesh, tri, one, kernel_proof=proof)
    assert len(calls) == 3 and len(proof) == 1                                 # job-owned dict: proven once for the identical element
    np.testing.assert_array_equal(first.vectors, again.vectors)                # the proof never changes the integrated vectors
    # a changed element (node coordinates) invalidates the stored proof
    mesh.nodes[1].x = 0.01
    module.region_nodal_vectors(element, mesh, tri, one, kernel_proof=proof)
    assert len(calls) == 4


def test_a_failed_owner_equivalence_proof_refuses_the_region_load(monkeypatch):
    import anysolver.floating_pressure_regions as module
    corners = [(0., 0., 0.), (2., 0., 0.), (2., 0., 2.), (0., 0., 2.)]
    mesh, element = build(corners)
    tri = np.array([[(0, 0, 0), (1, 0, 0), (0, 0, 1)]], float)

    def refuse(e, m):
        raise PressureRegionError('shell formulation pressure interpolation differs from its shape-function integral')
    monkeypatch.setattr(module, 'element_kernel', refuse)
    with pytest.raises(PressureRegionError, match='differs'):
        module.region_nodal_vectors(element, mesh, tri, lambda pts, ti: np.ones(len(pts)))


def test_complex_field_and_refusals():
    corners = [(0., 0., 0.), (2., 0., 0.), (2., 0., 2.), (0., 0., 2.)]
    mesh, element = build(corners)
    tri = np.array([[(0, 0, 0), (1, 0, 0), (0, 0, 1)]], float)
    c = region_nodal_vectors(element, mesh, tri, lambda pts, ti: np.full(len(pts), 2.0 - 1.0j))
    r = region_nodal_vectors(element, mesh, tri, lambda pts, ti: np.full(len(pts), 2.0))
    i = region_nodal_vectors(element, mesh, tri, lambda pts, ti: np.full(len(pts), -1.0))
    np.testing.assert_allclose(c.vectors, r.vectors + 1j * i.vectors, atol=1e-12)
    with pytest.raises(PressureRegionError):                                    # region off the element
        region_nodal_vectors(element, mesh, np.array([[(5, 0, 5), (6, 0, 5), (5, 0, 6)]], float), hydro)
    with pytest.raises(PressureRegionError):
        region_nodal_vectors(element, mesh, tri, lambda pts, ti: np.full(len(pts), np.nan))
    with pytest.raises(PressureRegionError):
        region_nodal_vectors(element, mesh, tri, hydro, sign=0.5)
