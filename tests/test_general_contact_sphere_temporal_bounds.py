"""Independent analytic fixtures for the complete beam/sphere Hermite path."""

from decimal import Decimal, localcontext
from fractions import Fraction
import math
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver.general_contact.beam_temporal_bounds import BeamHermiteGapBoundsError
from anyfesolver.general_contact.sphere_temporal_bounds import beam_sphere_hermite_gap_bounds
from anyfesolver.general_contact.sphere_temporal_bounds import _squared_controls


def _manager(points, radius=.25):
    reference = np.asarray(points, dtype=float)
    count = len(reference)
    cell = SimpleNamespace(reference=reference,
                           dofs=np.arange(count * 3).reshape(count, 3),
                           node_ids=tuple(range(count)))
    return SimpleNamespace(cells=[cell], radius=radius)


def _bounds(manager, **options):
    zeros = np.zeros(int(max(np.max(cell.dofs) for cell in manager.cells)) + 1)
    return beam_sphere_hermite_gap_bounds(
        manager, options.pop("start", zeros), options.pop("end", zeros),
        options.pop("start_velocity", zeros), options.pop("end_velocity", zeros),
        options.pop("dt", 1.),
        sphere_start_point=options.pop("sphere_start_point", np.zeros(3)),
        sphere_end_point=options.pop("sphere_end_point", np.zeros(3)),
        sphere_start_velocity=options.pop("sphere_start_velocity", np.zeros(3)),
        sphere_end_velocity=options.pop("sphere_end_velocity", np.zeros(3)),
        tolerance_m=options.pop("tolerance_m", 1.e-8),
        maximum_subdivisions=options.pop("maximum_subdivisions", 4096), **options)


def _encloses(report, minimum, tolerance=1.e-8):
    assert report.lower_gap <= minimum <= report.upper_gap
    assert report.upper_gap - report.lower_gap <= tolerance


@pytest.mark.parametrize("moving", ["sphere", "beam"])
def test_interior_cubic_crossing_despite_distant_positive_endpoint_gaps(moving):
    # z(s)=16*(s-1/4)^2*(1+s): endpoints are 1 and 18,
    # derivatives are -7 and 57, and the unique temporal zero is s=1/4.
    # A linear endpoint path would miss this complete interior crossing.
    if moving == "sphere":
        manager = _manager([[-1., 0., 0.], [1., 0., 0.]])
        report = _bounds(manager, dt=2., sphere_start_point=[0., 0., 1.],
                         sphere_end_point=[0., 0., 18.],
                         sphere_start_velocity=[0., 0., -3.5],
                         sphere_end_velocity=[0., 0., 28.5])
    else:
        manager = _manager([[-1., 0., 1.], [1., 0., 1.]])
        end, v0, v1 = (np.zeros(6) for _ in range(3))
        end[2::3], v0[2::3], v1[2::3] = 17., -3.5, 28.5
        report = _bounds(manager, end=end, start_velocity=v0, end_velocity=v1, dt=2.)
    _encloses(report, -.25)
    assert report.subdivisions > 0


def test_joint_interior_space_time_minimum_with_moving_sphere():
    # Relative coordinates: x=eta-3/8-(s-5/8)/4,
    # y=(s-5/8)^2*(1+s), z=1/2. The norm is >=1/2,
    # attained exactly at eta=3/8, s=5/8.
    eta, b = np.array([0., .5, 1.]), .625
    manager = _manager(np.column_stack((eta-.375, np.zeros(3), np.full(3, .5))))
    report = _bounds(manager, dt=2.,
                     sphere_start_point=[-.25*b, -b*b, 0.],
                     sphere_end_point=[.25*(1-b), -2*(1-b)**2, 0.],
                     sphere_start_velocity=[.125, (2*b-b*b)/2, 0.],
                     sphere_end_velocity=[.125, -(4*(1-b)+(1-b)**2)/2, 0.])
    _encloses(report, .25)


def test_uniform_b2_cubic_turning_gap_resolves_under_unchanged_512_split_cap():
    # Uniform z(s)=d+a*(3*s-1)^2*(1+s), with an interior minimum
    # at the non-dyadic time s=1/3. This isolates the broad line's spatial
    # support and a small temporal turning excursion, as in the saved coupled
    # dynamics failure. All endpoint data reconstruct this polynomial exactly.
    d, a, radius = .01875, 2.**-20, .02
    manager = _manager([[-.01, 0., d+a], [.01, 0., d+a]], radius=radius)
    end, v0, v1 = (np.zeros(6) for _ in range(3))
    end[2::3], v0[2::3], v1[2::3] = 7*a, -5*a, 28*a
    tolerance = 1.e-10
    report = _bounds(manager, end=end, start_velocity=v0, end_velocity=v1,
                     tolerance_m=tolerance, maximum_subdivisions=512)
    _encloses(report, d-radius, tolerance)
    assert report.subdivisions <= 512


@pytest.mark.parametrize("spatial_degree", [1, 2])
def test_squared_tensor_controls_match_independent_bernstein_evaluation(spatial_degree):
    # Arbitrary signed rational vector coefficients exercise mixed products.
    # Evaluate each original lane directly in the Bernstein basis and compare
    # its norm squared with the derived degree-doubled polynomial.
    lanes = tuple(tuple(tuple(Fraction((component+1)*(i+2)-(j+1)**2, 16)
                                   for j in range(4))
                             for i in range(spatial_degree+1))
                  for component in range(3))
    eta, time = Fraction(2, 7), Fraction(3, 11)
    def evaluate(lane):
        n, m = len(lane)-1, len(lane[0])-1
        return sum(value*math.comb(n, i)*eta**i*(1-eta)**(n-i)
                   *math.comb(m, j)*time**j*(1-time)**(m-j)
                   for i, row in enumerate(lane) for j, value in enumerate(row))
    squared = _squared_controls(lanes)
    assert len(squared) == 2*spatial_degree+1
    assert all(len(row) == 7 for row in squared)
    assert evaluate(squared) == sum(evaluate(lane)**2 for lane in lanes)


def test_b2_spatial_minimum_inside_segment():
    # x=eta-3/4, y=1/2: norm minimum is 1/2 at eta=3/4.
    _encloses(_bounds(_manager([[-.75, .5, 0.], [.25, .5, 0.]])), .25)


@pytest.mark.parametrize("height,minimum", [(0., -.25), (.5, .25)])
def test_b3_curved_spatial_minimum_between_all_lagrange_nodes(height, minimum):
    # x=eta-3/4, y=(eta-3/4)^2. All three nodes are separated
    # from the sphere, but the curved centreline reaches x=y=0.
    eta = np.array([0., .5, 1.])
    points = np.column_stack((eta-.75, (eta-.75)**2, np.full(3, height)))
    _encloses(_bounds(_manager(points)), minimum)


def test_stationary_open_and_endpoint_minima_need_no_subdivisions():
    open_report = _bounds(_manager([[-1., 2., 0.], [1., 2., 0.]]), maximum_subdivisions=0)
    _encloses(open_report, 1.75)
    endpoint_report = _bounds(_manager([[3., 4., 0.], [4., 4., 0.]]), maximum_subdivisions=0)
    _encloses(endpoint_report, 4.75)
    assert open_report.subdivisions == endpoint_report.subdivisions == 0


def test_irrational_distance_enclosure_with_independent_decimal_root():
    report = _bounds(_manager([[0., 1., 1.], [1., 1., 1.]]), maximum_subdivisions=0)
    with localcontext() as context:
        context.prec = 80
        minimum = Decimal(2).sqrt() - Decimal("0.25")
        assert Decimal.from_float(report.lower_gap) <= minimum <= Decimal.from_float(report.upper_gap)
    # Check final float distance inequalities independently and exactly.
    low = Fraction.from_float(report.lower_gap) + Fraction(1, 4)
    high = Fraction.from_float(report.upper_gap) + Fraction(1, 4)
    assert low*low <= 2 <= high*high


def test_common_rigid_translation_and_identical_hermite_motion_cancel():
    points = np.array([[-.75, .5625, .5], [-.25, .0625, .5], [.25, .0625, .5]])
    original = _bounds(_manager(points))
    translation = np.array([2.**40, -2.**30, 2.**20])
    displacement = np.array([1., -2., 3.])
    v0, v1 = np.array([.5, 1., -.5]), np.array([-.25, .75, .25])
    translated = _bounds(_manager(points+translation), end=np.tile(displacement, 3),
                         start_velocity=np.tile(v0, 3), end_velocity=np.tile(v1, 3),
                         sphere_start_point=translation, sphere_end_point=translation+displacement,
                         sphere_start_velocity=v0, sphere_end_velocity=v1)
    assert translated == original
    _encloses(translated, .25)


def test_all_cells_and_unused_full_dof_entries_are_admitted():
    manager = _manager([[2., 0., 0.], [3., 0., 0.]])
    closer = _manager([[-.75, .5, 0.], [.25, .5, 0.]]).cells[0]
    closer.dofs += 6
    manager.cells.append(closer)
    zeros = np.zeros(15)
    _encloses(_bounds(manager, start=zeros, end=zeros,
                      start_velocity=zeros, end_velocity=zeros), .25)
    zeros[-1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _bounds(manager, start=zeros, end=np.zeros(15),
                start_velocity=np.zeros(15), end_velocity=np.zeros(15))


def test_shared_subdivision_budget_fails_explicitly():
    with pytest.raises(BeamHermiteGapBoundsError, match="after 0 subdivisions"):
        _bounds(_manager([[-.75, .5, 0.], [.25, .5, 0.]]), maximum_subdivisions=0)


def test_roundoff_floor_fails_explicitly():
    with pytest.raises(BeamHermiteGapBoundsError, match="roundoff"):
        _bounds(_manager([[-1., 2., 0.], [1., 2., 0.]]), tolerance_m=1.e-30)


def test_squared_distance_overflow_does_not_prevent_finite_gap_enclosure():
    magnitude = 1.e200
    tolerance = 1.e190
    report = _bounds(_manager([[magnitude, 0., 0.], [magnitude, 1., 0.]], radius=0.),
                     tolerance_m=tolerance, maximum_subdivisions=0)
    _encloses(report, magnitude, tolerance)


def test_unrepresentable_gap_is_rejected():
    largest = np.finfo(float).max
    with pytest.raises(BeamHermiteGapBoundsError, match="finite float range"):
        _bounds(_manager([[largest, largest, 0.], [largest, largest, 1.]], radius=0.))


@pytest.mark.parametrize("name,size", [
    ("start", 6), ("end", 6), ("start_velocity", 6), ("end_velocity", 6),
    ("sphere_start_point", 3), ("sphere_end_point", 3),
    ("sphere_start_velocity", 3), ("sphere_end_velocity", 3),
])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_path_data_rejected(name, size, bad):
    vector = np.zeros(size)
    vector[-1] = bad
    with pytest.raises(ValueError, match="finite"):
        _bounds(_manager([[-1., 0., 0.], [1., 0., 0.]]), **{name: vector})


@pytest.mark.parametrize("options", [
    {"dt": 0.}, {"dt": -1.}, {"dt": np.nan},
    {"tolerance_m": 0.}, {"tolerance_m": np.inf},
    {"maximum_subdivisions": -1}, {"maximum_subdivisions": 1.5},
    {"maximum_subdivisions": True}, {"sphere_end_point": [0., 0.]},
    {"sphere_end_velocity": [0., 0.]}, {"end": np.zeros(7)},
    {"start": np.zeros((2, 3))}, {"check_cancel": 1},
])
def test_invalid_admission_and_controls_rejected(options):
    with pytest.raises(ValueError):
        _bounds(_manager([[-1., 0., 0.], [1., 0., 0.]]), **options)


@pytest.mark.parametrize("radius", [-1., np.nan, np.inf])
def test_invalid_combined_radius_rejected(radius):
    with pytest.raises(ValueError):
        _bounds(_manager([[-1., 0., 0.], [1., 0., 0.]], radius=radius))


def test_invalid_geometry_and_dof_mapping_rejected():
    manager = _manager([[-1., 0., 0.], [1., 0., 0.]])
    manager.cells[0].reference[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _bounds(manager)
    manager.cells[0].reference[0, 0] = -1.
    manager.cells[0].dofs = manager.cells[0].dofs.astype(float)
    with pytest.raises(ValueError, match="DOFs"):
        _bounds(manager)


def test_empty_cell_support_rejected():
    zeros = np.zeros(6)
    with pytest.raises(ValueError, match="at least one"):
        beam_sphere_hermite_gap_bounds(SimpleNamespace(cells=[], radius=.25),
            zeros, zeros, zeros, zeros, 1., sphere_start_point=np.zeros(3),
            sphere_end_point=np.zeros(3), sphere_start_velocity=np.zeros(3),
            sphere_end_velocity=np.zeros(3), tolerance_m=1.e-8, maximum_subdivisions=0)


def test_cancellation_is_checked_and_propagated():
    manager = _manager([[-1., 0., 0.], [1., 0., 0.]])
    with pytest.raises(BeamHermiteGapBoundsError, match="cancelled"):
        _bounds(manager, check_cancel=lambda: True)
    def cancel():
        raise InterruptedError("owned stop")
    with pytest.raises(InterruptedError, match="owned stop"):
        _bounds(manager, check_cancel=cancel)
