import numpy as np
import pytest

from anysolver.boundary import LoadCase
from anysolver.elements import create_shell_element
from anysolver.fe_core import FEModel
from anysolver.pressure_regions import PressureRegionError, add_region_load, region_wrench, wet_region_load

RHO_G = 1000.


def build(points, node_count_midside=False):
    model = FEModel('wet-region')
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


def triangle_vertical(order=3):
    corners = [(0., 0., -1.), (1., 0., -1.), (0., 0., 1.)]
    if order == 6:
        mids = [(.5, 0., -1.), (.5, 0., 0.), (0., 0., 0.)]
        corners += mids
    return corners


def rectangle(order=4):
    corners = [(0., 0., -1.), (2., 0., -1.), (2., 0., 1.), (0., 0., 1.)]
    if order == 8:
        corners += [(1., 0., -1.), (2., 0., 0.), (1., 0., 1.), (0., 0., 0.)]
    return corners


def hydrostatic(points, n_corner):
    z = np.array([p[2] for p in points[:n_corner]])
    return -z, RHO_G * (-z)      # level = depth below the surface z=0; p = rho g depth


@pytest.mark.parametrize('order', (3, 6))
def test_triangle_partial_wet_force_and_moment_match_analytic_integrals(order):
    points = triangle_vertical(order)
    mesh, element = build(points)
    level, pressure = hydrostatic(points, 3)
    load = wet_region_load(element, mesh, level, pressure)
    wrench = region_wrench(load, np.array(points))
    assert load.state == 'partial'
    assert load.force[1] == pytest.approx(-RHO_G * 5 / 12, rel=1e-11)
    assert wrench[3] == pytest.approx(-RHO_G * 7 / 24, rel=1e-11)
    assert wrench[5] == pytest.approx(-RHO_G * 17 / 96, rel=1e-11)
    assert np.allclose(wrench[[0, 2, 4]], 0., atol=1e-9)
    # wet area: integral of (1-z)/2 over z in [-1,0]
    assert load.wet_area == pytest.approx(.75, rel=1e-12)
    assert load.wet_fraction == pytest.approx(.75, rel=1e-12)
    assert load.mean_pressure == pytest.approx(RHO_G * 5 / 12 / 1., rel=1e-11)


@pytest.mark.parametrize('order', (4, 8))
def test_rectangle_partial_wet_matches_analytic_force_and_centroid_moment(order):
    points = rectangle(order)
    mesh, element = build(points)
    level, pressure = hydrostatic(points, 4)
    load = wet_region_load(element, mesh, level, pressure)
    wrench = region_wrench(load, np.array(points))
    assert load.force[1] == pytest.approx(-RHO_G * 2 * .5, rel=1e-11)
    # Mx = -z Fy integrated: p=rho g (-z), width 2: sum z*(-z)*2 dz
    assert wrench[3] == pytest.approx(-RHO_G * 2 / 3, rel=1e-11)
    assert wrench[5] == pytest.approx(-RHO_G * 1.0 * 1.0, rel=1e-11)  # centroid x = 1 times |F|
    assert load.wet_fraction == pytest.approx(.5, rel=1e-12)


def test_dry_is_exactly_zero_and_fully_wet_is_whole_element():
    points = rectangle()
    mesh, element = build(points)
    dry = wet_region_load(element, mesh, [-1., -1., -2., -2.], [5., 5., 5., 5.])
    assert dry.state == 'dry' and not dry.forces.any() and dry.wet_fraction == 0.
    wet = wet_region_load(element, mesh, [1., 1., 1., 1.], [7., 7., 7., 7.])
    assert wet.state == 'wet' and wet.wet_fraction == pytest.approx(1.)
    assert wet.force[1] == pytest.approx(-7. * 4., rel=1e-12)


def test_refinement_total_force_is_conservative_for_linear_field():
    # Split the wet part analytically: total over two stacked triangles equals one triangle's analytic result.
    total = 0.
    for corners in ([(0., 0., -1.), (1., 0., -1.), (0., 0., 1.)], [(1., 0., -1.), (1., 0., 1.), (0., 0., 1.)]):
        mesh, element = build(corners)
        level, pressure = hydrostatic(corners, 3)
        total += wet_region_load(element, mesh, level, pressure).force[1]
    # Unit-width strip wet for z in [-1, 0]: integral of rho g (-z) dz = rho g / 2, independent of the split.
    assert total == pytest.approx(-RHO_G * .5, rel=1e-11)


@pytest.mark.parametrize('points,corners', [(triangle_vertical(6), 3), (rectangle(8), 4)])
def test_full_element_kernel_equals_fully_wet_region_load_and_full_area(points, corners):
    from anysolver.pressure_regions import element_kernel
    mesh, element = build(points)
    pressure = np.array([3., 5., 7., 11.][:corners])
    kernel, area, corner_areas = element_kernel(element, mesh)
    assert corner_areas.sum() == pytest.approx(area, rel=1e-12)
    load = wet_region_load(element, mesh, np.ones(corners), pressure, sign=-1.)
    assert np.allclose(-np.einsum('nkc,c->nk', kernel, pressure), load.vectors, rtol=1e-12, atol=1e-12)
    assert area == pytest.approx(load.total_area, rel=1e-12)
    assert area == pytest.approx(load.wet_area, rel=1e-12)


def test_qualified_formulations_match_their_native_dead_pressure_or_are_refused():
    from anysolver.e4_pl_element import QualifiedE4PLShellElement
    from anysolver.e4_pl_element import QualifiedE4PLShellElement
    from anysolver.pressure_regions import element_kernel
    model = FEModel('qualified')
    model.add_material('m', 210e9, .3)
    points = [(0., 0., 0.), (2., 0., 0.), (2., 1., 0.), (0., 1., 0.)]
    for index, point in enumerate(points, 1):
        model.add_node(index, *point)
    element = QualifiedE4PLShellElement(1, (1, 2, 3, 4), 'm', thickness=.01, reference_normal=(0., 0., 1.))
    model.add_element(1, element)
    kernel, full_area, _ = element_kernel(element, model.mesh)      # raises if the interpolation differs from native
    assert full_area == pytest.approx(2., rel=1e-12)
    assert kernel.sum(axis=(0, 2))[2] == pytest.approx(2., rel=1e-12)


def v2d(points, normal):
    from anysolver import create_shell_element
    model = FEModel('v2d')
    model.add_material('m', 210e9, .3)
    for index, point in enumerate(points, 1):
        model.add_node(index, *point)
    element = create_shell_element(1, list(range(1, len(points) + 1)), material_name='m', thickness=.01,
                                   formulation='e4-pl-s3-v2d', reference_normal=normal)
    model.add_element(1, element)
    return model.mesh, element


def test_current_default_v2d_uniform_region_reproduces_all_native_components_including_moments():
    from anysolver.pressure_regions import element_kernel
    tilted = [(0., 0., 0.), (2., .3, .1), (.4, 1.5, -.2)]
    facet = np.cross(np.subtract(tilted[1], tilted[0]), np.subtract(tilted[2], tilted[0]))
    mesh, element = v2d(tilted, tuple(facet / np.linalg.norm(facet)))     # reference normal = the actual facet normal
    native = np.asarray(element.compute_native_dead_pressure_load(mesh, 7.)).reshape(3, 6)
    assert np.abs(native[:, 3:]).max() > 0.                      # the native load really has rotational terms
    kernel, area, _ = element_kernel(element, mesh)
    assert np.allclose(7. * kernel.sum(axis=2), native, rtol=1e-10, atol=1e-10 * np.abs(native).max())
    load = wet_region_load(element, mesh, np.ones(3), np.full(3, 7.), total_area=area)
    assert np.allclose(load.vectors, native, rtol=1e-10, atol=1e-10 * np.abs(native).max())
    case = LoadCase('v2d')
    add_region_load(case, load)
    assert all(len(v) == 6 for v in case.nodal_loads.values())
    assert np.allclose(np.array([case.nodal_loads[i] for i in (1, 2, 3)]), native, rtol=1e-10, atol=1e-10 * np.abs(native).max())


def test_current_default_v2d_partial_region_matches_analytic_force_and_moment_and_orientation_flip():
    points = triangle_vertical()
    mesh, element = v2d(points, (0., -1., 0.))
    level, pressure = hydrostatic(points, 3)
    load = wet_region_load(element, mesh, level, pressure)
    wrench = region_wrench(load, np.array(points))
    assert load.state == 'partial' and load.moments.any()
    assert load.force[1] == pytest.approx(-RHO_G * 5 / 12, rel=1e-10)
    assert wrench[3] == pytest.approx(-RHO_G * 7 / 24, rel=1e-6) and wrench[5] == pytest.approx(-RHO_G * 17 / 96, rel=1e-6)
    flipped_mesh, flipped = v2d(points, (0., 1., 0.))              # opposite reference normal: positive pressure flips
    other = wet_region_load(flipped, flipped_mesh, level, pressure)
    assert np.allclose(other.force, -load.force, rtol=1e-10, atol=1e-8)


def test_native_density_formulation_refuses_higher_order_and_unknown_shapes():
    class Fake:
        node_ids = (1, 2, 3, 4, 5, 6)
        def compute_native_pressure_density(self, mesh, points): raise AssertionError
    with pytest.raises(PressureRegionError):
        wet_region_load(Fake(), None, np.ones(3), np.ones(3))


def test_loads_enter_load_case_through_public_nodal_loads_and_preserve_force():
    points = triangle_vertical()
    mesh, element = build(points)
    level, pressure = hydrostatic(points, 3)
    load = wet_region_load(element, mesh, level, pressure)
    case = LoadCase('wet')
    add_region_load(case, load)
    total = sum(v[:3] for v in case.nodal_loads.values())
    assert np.allclose(total, load.force)


def test_refuses_nonfinite_degenerate_and_unsupported():
    points = triangle_vertical()
    mesh, element = build(points)
    with pytest.raises(PressureRegionError):
        wet_region_load(element, mesh, [1., np.nan, 1.], [1., 1., 1.])
    with pytest.raises(PressureRegionError):
        wet_region_load(element, mesh, [1., 1.], [1., 1.])
    with pytest.raises(PressureRegionError):
        wet_region_load(object(), mesh, [1., 1., 1.], [1., 1., 1.])
    flat, flat_element = build([(0., 0., 0.), (1., 0., 0.), (2., 0., 0.)])
    with pytest.raises(PressureRegionError):
        wet_region_load(flat_element, flat, [1., 1., 1.], [1., 1., 1.])
