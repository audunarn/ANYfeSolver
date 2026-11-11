"""Bounded load mechanics checks; no material solve or qualification claim."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from anyfesolver._ge_beam3_fibre_line_work import evaluate as ad_work
from anyfesolver._ge_beam3_p5.algebra import rotation, skew
from anyfesolver._ge_beam3_retained_fibre import RetainedFibreOperator
from anyfesolver._ge_beam3_retained_generalized import RetainedGeneralizedOperator
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._stiffened_shell_native_distributed_load import compile_native_distributed
from test_ge_beam3_fibre_section import section as fibre_section
from test_stiffened_shell_native_gravity import reference, state


def operator(kind='generalized', order=4, curved=True):
    ref = reference(curved)
    if kind == 'fibre':
        return RetainedFibreOperator(ref, fibre_section(), order=order)
    section = EllipsoidalGeneralizedSection(np.diag([4., 6., 8., 2., 3., 5.]), np.eye(6), .25, .6)
    return RetainedGeneralizedOperator(ref, section, order=order)


@pytest.mark.parametrize('kind', ['generalized', 'fibre'])
@pytest.mark.parametrize('order', [4, 8])
def test_reusable_density_matches_fresh_compilation_without_preparation(kind, order, monkeypatch):
    op = operator(kind, order)
    x, low, u, force = state(op.reference)
    kernel = compile_native_distributed(op, force, np.array([.3, -.4, .2]))
    original = kernel.evaluate(x, low, u)
    stages = [(force, [0., 0., 0.]), (-force, [.2, .1, -.4]),
              (np.zeros(3), np.zeros(3)), (np.zeros(3), [-.3, .4, -.2])]
    expected = [compile_native_distributed(op, f, c).evaluate(x, low, u) for f, c in stages]
    def forbidden(*args, **kwargs):
        raise AssertionError('stage evaluation rebuilt quadrature')
    monkeypatch.setattr(np.polynomial.legendre, 'leggauss', forbidden)
    for (f, c), want in zip(stages, expected, strict=True):
        density, couple = np.array(f), np.array(c)
        got = kernel.evaluate_density(x, low, u, density, couple)
        density[:] = 99.; couple[:] = 99.
        for name in ('external_force', 'spatial_jacobian', 'total_force', 'total_moment'):
            np.testing.assert_array_equal(getattr(got, name), getattr(want, name))
            assert not getattr(got, name).flags.writeable
        assert got.force_potential == want.force_potential
        assert got.potential == want.potential
    np.testing.assert_array_equal(kernel.evaluate(x, low, u).external_force, original.external_force)


@pytest.mark.parametrize('bad', [[float('nan'), 0., 0.], [1., 2.], [1j, 0., 0.]])
def test_reusable_density_rejects_invalid_inputs(bad):
    op = operator()
    x, low, u, force = state(op.reference)
    kernel = compile_native_distributed(op, force, np.zeros(3))
    for f, c in ((bad, np.zeros(3)), (np.zeros(3), bad)):
        with pytest.raises(ValueError):
            kernel.evaluate_density(x, low, u, f, c)


def test_reusable_density_preserves_resolution_failures():
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
    op = operator()
    ref = Reference(op.reference.coordinates*1e-3, op.reference.nodal_triads)
    small = RetainedGeneralizedOperator(ref, op.section, order=4)
    kernel = compile_native_distributed(small, np.zeros(3), np.zeros(3))
    rotations = np.tile(np.eye(3), (2, 1, 1))
    tiny = [np.nextafter(0., 1.), 0., 0.]
    for force, couple in ((tiny, np.zeros(3)), (np.zeros(3), tiny)):
        with pytest.raises(ValueError, match='representable'):
            kernel.evaluate_density(ref.coordinates, np.zeros((3, 3)), rotations, force, couple)
    full = compile_native_distributed(op, np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match='representable'):
        full.evaluate_density(op.reference.coordinates, np.zeros((3, 3)), rotations,
                              [np.finfo(float).max, 0., 0.], np.zeros(3))


@pytest.mark.parametrize('kind', ['generalized', 'fibre'])
@pytest.mark.parametrize('order', [4, 8])
@pytest.mark.parametrize('curved', [False, True])
def test_same_rule_ad_force_and_existing_cell_couple(kind, order, curved):
    op = operator(kind, order, curved)
    x, low, u, force = state(op.reference)
    couple = np.array([.3, -.4, .2])
    kernel = compile_native_distributed(op, force, couple)
    actual = kernel.evaluate(x, low, u)
    expected = ad_work(op.reference, x, low, u, force, order=order)
    if kind == 'fibre':
        from anyfesolver._ge_beam3_fibre_distributed_couples import cell_couple_load
    else:
        from anyfesolver._ge_beam3_generalized_static_boundary import cell_couple_load
    applied = cell_couple_load(op, couple)
    np.testing.assert_allclose(actual.external_force, expected.gradient+applied, rtol=4e-14, atol=1e-14)
    spatial = expected.hessian.copy()
    for cell in (0, 1):
        sl = slice(18+3*cell, 21+3*cell)
        spatial[sl, sl] -= .5*skew(expected.gradient[sl])
    np.testing.assert_allclose(actual.spatial_jacobian, spatial, rtol=4e-14, atol=1e-14)
    np.testing.assert_allclose(actual.force_potential, expected.value, rtol=4e-14, atol=1e-14)
    assert actual.potential is None
    assert kernel.order == order
    assert not np.any(actual.external_force[24:])
    assert not np.any(actual.spatial_jacobian[24:])
    assert not np.any(actual.spatial_jacobian[:, 24:])
    for start in (3, 9, 15):
        assert not np.any(actual.external_force[start:start+3])
    conservative = compile_native_distributed(op, force, np.zeros(3)).evaluate(x, low, u)
    assert conservative.potential == conservative.force_potential


def test_actual_left_update_derivative_and_fixed_couple_zero_spatial_derivative():
    op = operator()
    x, low, u, force = state(op.reference)
    couple = np.array([.3, -.4, .2])
    kernel = compile_native_distributed(op, force, couple)
    only_couple = compile_native_distributed(op, np.zeros(3), couple)
    work = kernel.evaluate(x, low, u)
    fixed = only_couple.evaluate(x, low, u)
    assert not np.any(fixed.spatial_jacobian)
    assert fixed.force_potential == 0.
    assert fixed.potential is None
    step = 2e-5
    for cell in (0, 1):
        for axis in range(3):
            delta = np.eye(3)[axis]*step
            plus, minus = u.copy(), u.copy()
            plus[cell] = rotation(delta)@u[cell]
            minus[cell] = rotation(-delta)@u[cell]
            wp = kernel.evaluate(x, low, plus)
            wm = kernel.evaluate(x, low, minus)
            col = 18+3*cell+axis
            np.testing.assert_allclose((wp.external_force-wm.external_force)/(2*step),
                                       work.spatial_jacobian[:, col], rtol=2e-9, atol=2e-10)
            np.testing.assert_array_equal(only_couple.evaluate(x, low, plus).external_force, fixed.external_force)
            np.testing.assert_array_equal(only_couple.evaluate(x, low, minus).external_force, fixed.external_force)


def test_actual_integrated_force_and_global_origin_moment():
    op = operator(order=8)
    x, low, u, force = state(op.reference)
    couple = np.array([.3, -.4, .2])
    kernel = compile_native_distributed(op, force, couple)
    work = kernel.evaluate(x, low, u)
    expected_force, expected_moment = np.zeros(3), np.zeros(3)
    points, weights = np.polynomial.legendre.leggauss(op.order)
    for cell in (0, 1):
        for point, weight in zip(points, weights):
            t = float((point+1)/2)
            measure = float(weight*op.reference.jacobian(cell-1+t)/2)
            position = (1-t)*(x[cell]+low[cell])+t*(x[cell+1]+low[cell+1])
            position += u[cell]@op.reference.half_cell_lift(cell, t)
            expected_force += measure*force
            expected_moment += measure*(np.cross(position, force)+couple)
    np.testing.assert_allclose(work.total_force, expected_force, rtol=4e-15, atol=1e-14)
    np.testing.assert_allclose(work.total_moment, expected_moment, rtol=4e-15, atol=1e-14)
    shift = np.array([4., -2., 1.])
    moved = kernel.evaluate(x+shift, low, u)
    np.testing.assert_allclose(moved.total_moment, work.total_moment+np.cross(shift, work.total_force), atol=3e-14)


def test_owned_capture_is_independent_of_source_and_result_arrays_are_immutable():
    op = operator()
    x, low, u, force = state(op.reference)
    couple = np.array([.3, -.4, .2])
    kernel = compile_native_distributed(op, force, couple)
    expected = kernel.evaluate(x, low, u)
    force[:] = 0.
    couple[:] = 0.
    object.__setattr__(op, 'order', 8)
    object.__setattr__(op, 'reference', reference(False))
    actual = kernel.evaluate(x, low, u)
    np.testing.assert_array_equal(actual.external_force, expected.external_force)
    assert kernel.order == 4
    for value in (kernel, actual):
        for name in value.__dataclass_fields__:
            array = getattr(value, name)
            if isinstance(array, np.ndarray):
                with pytest.raises(ValueError):
                    array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        kernel.order = 8


def test_rejects_wrong_operator_nonfinite_complex_and_unrepresentable_values():
    with pytest.raises(ValueError, match='exact'):
        compile_native_distributed(object(), np.zeros(3), np.zeros(3))
    op = operator()
    for bad in ([np.nan, 0., 0.], [np.inf, 0., 0.], [1j, 0., 0.], [1., 2.]):
        with pytest.raises(ValueError):
            compile_native_distributed(op, bad, np.zeros(3))
        with pytest.raises(ValueError):
            compile_native_distributed(op, np.zeros(3), bad)
    x, low, u, force = state(op.reference)
    kernel = compile_native_distributed(op, force, np.zeros(3))
    with pytest.raises(ValueError):
        kernel.evaluate(np.full((3, 3), np.nan), low, u)
    with pytest.raises(ValueError):
        kernel.evaluate(x, low, np.zeros((2, 3, 3)))
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
    small_ref = Reference(op.reference.coordinates*1e-3, op.reference.nodal_triads)
    small_op = RetainedGeneralizedOperator(small_ref, op.section, order=4)
    # Short reference shape measures cannot represent this nonzero force.
    tiny = compile_native_distributed(small_op, [np.nextafter(0., 1.), 0., 0.], np.zeros(3))
    with pytest.raises(ValueError, match='representable'):
        tiny.evaluate(small_ref.coordinates, low, u)
    huge = compile_native_distributed(op, [np.finfo(float).max, 0., 0.], np.zeros(3))
    with pytest.raises(ValueError, match='representable'):
        huge.evaluate(x, low, u)
