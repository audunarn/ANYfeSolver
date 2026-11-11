"""Analytic manufactured checks independent of the production contact kernel."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver.general_contact.beam_temporal_bounds import (
    BeamHermiteGapBoundsError,
    beam_hermite_gap_bounds,
)


def _manager(heights, *, radius=0.25):
    count = len(heights)
    reference = np.zeros((count, 3))
    reference[:, 0] = np.linspace(0, 1, count)
    reference[:, 2] = np.asarray(heights) + radius
    cell = SimpleNamespace(reference=reference,
                           dofs=np.arange(count * 3).reshape(count, 3),
                           node_ids=tuple(range(count)))
    return SimpleNamespace(cells=[cell], point=np.zeros(3),
                           normal=np.array([0., 0., 1.]), radius=radius)


def _bounds(manager, start=None, end=None, v0=None, v1=None, **options):
    size = int(np.max(manager.cells[0].dofs)) + 1
    zeros = np.zeros(size)
    return beam_hermite_gap_bounds(
        manager, zeros if start is None else start, zeros if end is None else end,
        zeros if v0 is None else v0, zeros if v1 is None else v1,
        options.pop("dt", 1.), tolerance_m=options.pop("tolerance_m", 1.e-9),
        maximum_subdivisions=options.pop("maximum_subdivisions", 2048), **options)


def _encloses(report, minimum, tolerance=1.e-9):
    assert report.lower_gap <= minimum <= report.upper_gap
    assert report.upper_gap - report.lower_gap <= tolerance


def test_b3_manufactured_interior_curve_and_cubic_time_minimum():
    # g(eta,s) = (eta-3/8)^2 + (s-5/8)^2*(1+s) - 1/8.
    # Both nonnegative terms vanish at an interior point. All constants and
    # nodal values are binary exact, so the analytic minimum is exactly -1/8.
    eta = np.array([0., .5, 1.])
    a, b = .375, .625
    manager = _manager((eta-a)**2 + b*b - .125)
    end, v0, v1 = (np.zeros(9) for _ in range(3))
    end[2::3] = 2*(1-b)**2 - b*b
    v0[2::3] = (-2*b + b*b) / 2
    v1[2::3] = (4*(1-b) + (1-b)**2) / 2
    report = _bounds(manager, end=end, v0=v0, v1=v1, dt=2.)
    _encloses(report, -.125)
    assert report.subdivisions > 0
    with pytest.raises(FrozenInstanceError):
        report.lower_gap = 1.


def test_b3_coupled_space_time_minimum_with_moving_curve_vertex():
    # g=(eta-a-k*(s-b))^2+(s-b)^2*(1+s)-1/8 has the same
    # unique interior minimum, while its spatial vertex moves through time.
    eta = np.array([0., .5, 1.])
    a, b, k = .375, .625, .25
    def value(s):
        return (eta-a-k*(s-b))**2 + (s-b)**2*(1+s) - .125
    def velocity(s):
        return -2*k*(eta-a-k*(s-b)) + 2*(s-b)*(1+s) + (s-b)**2
    manager = _manager(value(0.))
    end, v0, v1 = (np.zeros(9) for _ in range(3))
    end[2::3] = value(1.) - value(0.)
    v0[2::3], v1[2::3] = velocity(0.), velocity(1.)
    _encloses(_bounds(manager, end=end, v0=v0, v1=v1), -.125)


@pytest.mark.parametrize("heights,minimum", [([.5, -.25], -.25),
                                             ([-.25, .5], -.25)])
def test_b2_static_minimum_is_each_endpoint(heights, minimum):
    report = _bounds(_manager(heights), maximum_subdivisions=0)
    _encloses(report, minimum)
    assert report.subdivisions == 0


def test_b2_time_minimum_occurs_between_equal_positive_end_states():
    # The lower endpoint follows (s-1/2)^2 - 1/8, despite positive endpoints.
    manager = _manager([.125, .625])
    v0, v1 = np.zeros(6), np.zeros(6)
    v0[2::3], v1[2::3] = -1., 1.
    _encloses(_bounds(manager, v0=v0, v1=v1), -.125)


def test_b3_negative_shape_weights_keep_off_midpoint_minimum():
    # g(eta)=(eta-3/4)^2-1/32 is positive at all three Lagrange nodes.
    eta = np.array([0., .5, 1.])
    heights = (eta-.75)**2 - .03125
    assert np.all(heights > 0)
    _encloses(_bounds(_manager(heights)), -.03125)


def test_b3_midpoint_extremum_is_included():
    _encloses(_bounds(_manager([.25, -.25, .25])), -.25)


def test_common_plane_and_reference_translation_preserves_gap():
    original = _manager([.53125, .03125, .03125])
    shifted = _manager([.53125, .03125, .03125])
    translation = np.array([2.**40, -2.**30, 2.**20])
    shifted.point += translation
    shifted.cells[0].reference += translation
    first, second = _bounds(original), _bounds(shifted)
    assert first == second
    _encloses(second, -.03125)


def test_every_cell_and_full_dof_index_is_kept():
    manager = _manager([.5, .5])
    deeper = _manager([.53125, .03125, .03125]).cells[0]
    deeper.dofs += 6
    manager.cells.append(deeper)
    zeros = np.zeros(18)  # Unused full-vector entries must still be finite.
    report = beam_hermite_gap_bounds(manager, zeros, zeros, zeros, zeros, 1.,
                                    tolerance_m=1.e-9, maximum_subdivisions=2048)
    _encloses(report, -.03125)
    invalid = zeros.copy()
    invalid[-1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        beam_hermite_gap_bounds(manager, invalid, zeros, zeros, zeros, 1.,
                                tolerance_m=1.e-9, maximum_subdivisions=2048)


def test_insufficient_finite_budget_fails_explicitly():
    with pytest.raises(BeamHermiteGapBoundsError, match="after 0 subdivisions"):
        _bounds(_manager([.53125, .03125, .03125]), maximum_subdivisions=0)


def test_roundoff_floor_does_not_report_false_resolution():
    with pytest.raises(BeamHermiteGapBoundsError, match="roundoff"):
        _bounds(_manager([1., 1.]), tolerance_m=1.e-30)


@pytest.mark.parametrize("name", ["start", "end", "v0", "v1"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_full_path_inputs_rejected(name, bad):
    values = np.zeros(6)
    values[0] = bad
    with pytest.raises(ValueError, match="finite"):
        _bounds(_manager([0., 0.]), **{name: values})


@pytest.mark.parametrize("attribute", ["point", "normal", "radius", "reference"])
def test_nonfinite_geometry_rejected(attribute):
    manager = _manager([0., 0.])
    if attribute == "radius":
        manager.radius = np.nan
    elif attribute == "reference":
        manager.cells[0].reference[0, 0] = np.inf
    else:
        getattr(manager, attribute)[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _bounds(manager)


@pytest.mark.parametrize("options", [
    {"dt": 0.}, {"dt": -1.}, {"dt": np.nan}, {"dt": np.inf},
    {"tolerance_m": 0.}, {"tolerance_m": -1.}, {"tolerance_m": np.nan},
    {"maximum_subdivisions": -1}, {"maximum_subdivisions": 1.5},
    {"maximum_subdivisions": True},
])
def test_invalid_refinement_controls_rejected(options):
    with pytest.raises(ValueError):
        _bounds(_manager([0., 0.]), **options)


def test_invalid_full_vector_shape_and_dof_mapping_rejected():
    manager = _manager([0., 0.])
    with pytest.raises(ValueError, match="shape"):
        _bounds(manager, end=np.zeros(7))
    with pytest.raises(ValueError, match="one-dimensional"):
        _bounds(manager, start=np.zeros((2, 3)))
    manager.cells[0].dofs[0, 0] = -1
    with pytest.raises(ValueError, match="DOFs"):
        _bounds(manager)


def test_cancellation_is_checked_and_propagated():
    with pytest.raises(BeamHermiteGapBoundsError, match="cancelled"):
        _bounds(_manager([0., 0.]), check_cancel=lambda: True)
    def cancel():
        raise InterruptedError("owned stop")
    with pytest.raises(InterruptedError, match="owned stop"):
        _bounds(_manager([0., 0.]), check_cancel=cancel)


def test_finite_inputs_with_unrepresentable_gap_fail_explicitly():
    manager = _manager([0., 0.])
    manager.cells[0].reference[:, 2] = np.finfo(float).max
    displacement = np.zeros(6)
    displacement[2::3] = np.finfo(float).max
    with pytest.raises(BeamHermiteGapBoundsError, match="finite float range"):
        _bounds(manager, start=displacement, end=displacement)
