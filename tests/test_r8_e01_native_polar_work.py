"""Actual native half-cell local trace controls; no element/global solves."""
from dataclasses import replace
from types import SimpleNamespace
import math
import json
import os
from pathlib import Path

import numpy as np
import pytest

from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum
from anyfesolver._ge_beam3_retained_generalized_state import Context, Mechanical
from anyfesolver._stiffened_shell_l1_local_experimental import NativeHalfCellPort
from anyfesolver._ge_beam3_mixed_ad import rotation_exponential
from anyfesolver import _ge_beam3_mixed_ad as ad
from scripts import r8_e01_native_polar_work as native
from scripts import r8_e01_objective_polar_trace as polar


XY = np.array(((0., -.5), (2., -.5), (2., .5), (0., .5)))
STATION = (.8, -.2)
LAM = np.array((3., -2., 1., .4, -.6, .8))


def fixture(cell=0, weight=.3, reverse=False):
    coordinates = np.array(((0., -.5, .1), (1., -.5, .14), (2., -.5, .1)))
    triads = np.array([rotation_exponential((0., -math.atan(slope), 0.)) for slope in (.08, 0., -.08)])
    if reverse:
        coordinates = coordinates[::-1].copy()
        triads = np.array([r@np.diag((-1., -1., 1.)) for r in triads[::-1]])
    reference = CenteredCurvedBeam3ReferenceGeometry(coordinates, triads)
    mechanical = Mechanical(coordinates.copy(), np.zeros((3, 3)), triads.copy(),
                             np.tile(np.eye(3), (1, 2, 1, 1)), np.arange(18).reshape(1, 18)*.01)
    port = NativeHalfCellPort.from_reference(reference, (0, 1, 2), 0, cell, weight, 18, 42)
    operator = native.NativePolarWork.at_reference(XY, STATION, port, mechanical,
                                                  np.arange(24), native_base=100)
    shell = native.ShellState(np.column_stack((XY, np.zeros(4))), np.tile((0., 0., 1.), (4, 1)))
    return operator, shell, mechanical


def finite(operator, shell, mechanical):
    delta = np.zeros(33)
    delta[:24].reshape(4, 6)[:, :3] = np.column_stack((.02*XY[:, 0], -.01*XY[:, 1], .03*XY[:, 0]*XY[:, 1]))
    delta[:24].reshape(4, 6)[:, 3:] = ((.03, -.04, .02), (.05, -.02, .01), (.04, -.03, -.01), (.02, -.05, .03))
    delta[24:] = (.01, -.02, .015, -.01, .03, -.005, .08, -.04, .06)
    return native.updated_shell(shell, delta[:24]), native.updated_native(mechanical, operator.port, delta[24:])


def physical_pose(operator, shell, mechanical):
    high, low, frame = operator.port.pose_pair(mechanical)
    position = np.array([math.fsum((float(h), float(l))) for h, l in zip(high, low, strict=True)])
    return polar.PhysicalState(shell.positions, shell.directors, position, frame)


def chain(operator, mechanical):
    matrix = np.zeros((30, 33)); matrix[:24, :24] = np.eye(24)
    matrix[24:27, 24:27] = (1.-operator.port.weight)*np.eye(3)
    matrix[24:27, 27:30] = operator.port.weight*np.eye(3)
    lift = operator.port.parts(mechanical)[-1] @ operator.port.lift
    axes = np.eye(3)
    for a in range(3): matrix[24:27, 30+a] = np.cross(axes[a], lift)
    matrix[27:30, 30:33] = np.eye(3)
    curvature = np.array([[.5*(np.cross(axes[a], np.cross(axes[b], lift))+
                                np.cross(axes[b], np.cross(axes[a], lift))) for b in range(3)] for a in range(3)])
    return matrix, curvature


@pytest.mark.parametrize('cell,weight,reverse', ((0, .3, False), (1, .7, False), (0, .3, True), (1, .7, True)))
def test_authoritative_pose_slots_reference_and_reversed_cells(cell, weight, reverse):
    operator, shell, mechanical = fixture(cell, weight, reverse)
    result = operator.evaluate(shell, mechanical, LAM)
    high, low, frame = operator.port.pose_pair(mechanical)
    np.testing.assert_array_equal(result.beam_position_high, high)
    np.testing.assert_array_equal(result.beam_position_low, low)
    np.testing.assert_array_equal(result.beam_rotation, frame)
    np.testing.assert_array_equal(result.beam_slots, operator.port.active_beam_slots())
    np.testing.assert_array_equal(result.global_slots, np.r_[np.arange(24), 100+operator.port.active_beam_slots()])
    np.testing.assert_allclose(result.local.constraint, 0., rtol=0., atol=2e-14)
    assert result.local.hessian.shape == (33, 33)
    assert result.inactive_native_coordinates == 33
    assert np.linalg.norm(operator.port.lift) > .001


def test_full_chain_gradient_hessian_and_nonzero_lift_curvature():
    operator, shell, mechanical = fixture()
    shell, mechanical = finite(operator, shell, mechanical)
    result = operator.evaluate(shell, mechanical, LAM).local
    independent = polar.evaluate(operator.trace, physical_pose(operator, shell, mechanical), LAM)
    matrix, curvature = chain(operator, mechanical)
    correction = np.zeros((33, 33))
    correction[30:33, 30:33] = np.einsum('i,abi->ab', independent.covector[24:27], curvature)
    np.testing.assert_allclose(result.constraint, independent.constraint, rtol=0., atol=2e-13)
    np.testing.assert_allclose(result.jacobian, independent.jacobian @ matrix, rtol=0., atol=2e-12)
    np.testing.assert_allclose(result.covector, matrix.T @ independent.covector, rtol=0., atol=2e-12)
    np.testing.assert_allclose(result.hessian, matrix.T @ independent.hessian @ matrix+correction,
                               rtol=0., atol=2e-11)
    assert np.linalg.norm(correction) > 1e-4


def test_tiny_rotation_lift_increment_avoids_identity_cancellation():
    angle = 2.**-54
    spin = [ad.Jet2.variable(v, i, 3) for i, v in enumerate((0., 0., angle))]
    lift = [ad.Jet2.constant(v, 3) for v in (2.**54, 2.**54, 0.)]
    result = native._rotated_lift_increment(spin, lift)
    np.testing.assert_allclose([v.value for v in result], (-1., 1., 0.), rtol=0., atol=1e-14)
    # This is a local analytic increment control, not a claim that a fresh
    # native matrix-vector pose can resolve such a huge lift to sub-ULP work.


def test_actual_update_plateau_full_multiplier_work_and_chart():
    operator, shell, mechanical = fixture(1, .7)
    shell, mechanical = finite(operator, shell, mechanical)
    result = operator.evaluate(shell, mechanical, LAM).local
    direction = np.cos(np.arange(33)+.2); direction /= np.linalg.norm(direction)
    errors_j, errors_h = [], []
    for h in (2e-4, 1e-4, 5e-5):
        plus = operator.evaluate(shell, mechanical, LAM, increments=h*direction).local
        minus = operator.evaluate(shell, mechanical, LAM, increments=-h*direction).local
        actual = operator.evaluate(native.updated_shell(shell, h*direction[:24]),
            native.updated_native(mechanical, operator.port, h*direction[24:]), LAM).local
        np.testing.assert_allclose(actual.constraint, plus.constraint, rtol=0., atol=2e-14)
        exact = result.jacobian @ direction
        errors_j.append(np.linalg.norm((plus.constraint-minus.constraint)/(2*h)-exact)/np.linalg.norm(exact))
        exact_h = result.hessian @ direction
        errors_h.append(np.linalg.norm((plus.covector-minus.covector)/(2*h)-exact_h)/np.linalg.norm(exact_h))
    assert max(errors_j) <= 1e-6, errors_j
    assert max(errors_h) <= 1e-6, errors_h
    assert np.linalg.norm(result.hessian-result.hessian.T)/np.linalg.norm(result.hessian) <= 1e-9
    np.testing.assert_allclose(result.hessian, np.einsum('i,ijk->jk', LAM, result.constraint_hessians), atol=1e-14)


def test_wrench_balance_origin_and_native_variable_ownership():
    operator, shell, mechanical = fixture()
    before = {k: v.copy() for k, v in mechanical.descriptor().items()}
    shell, moved = finite(operator, shell, mechanical)
    result = operator.evaluate(shell, moved, LAM)
    for origin in ((0., 0., 0.), (11., -7., 3.)):
        force, moment = native.wrench_balance(shell, moved, operator, result, origin)
        np.testing.assert_allclose(force, 0., rtol=0., atol=1e-12)
        np.testing.assert_allclose(moment, 0., rtol=0., atol=1e-11)
    for k, value in before.items(): np.testing.assert_array_equal(getattr(mechanical, k), value)
    np.testing.assert_array_equal(moved.nodal_frames, mechanical.nodal_frames)
    np.testing.assert_array_equal(moved.resultants, mechanical.resultants)
    np.testing.assert_array_equal(moved.cell_rotations[0, 1], mechanical.cell_rotations[0, 1])
    # A bounded scatter witness preserves all inactive coordinates as zeros.
    full = np.zeros(42); full[result.beam_slots] = result.local.covector[24:]
    inactive = np.setdiff1d(np.arange(42), result.beam_slots)
    np.testing.assert_array_equal(full[inactive], 0.)
    assert not result.global_slots.flags.writeable


def test_common_finite_motion_and_covariant_local_work():
    operator, shell, mechanical = fixture()
    shell, mechanical = finite(operator, shell, mechanical)
    r = rotation_exponential((.7, -.4, .3)); t = np.array((13., -21., 8.))
    positions, lows = np.empty_like(mechanical.positions), np.empty_like(mechanical.position_low)
    for i in range(3):
        for j in range(3):
            positions[i, j], lows[i, j] = split_sum(tuple(float(v) for v in (
                t[j], *(r[j]*mechanical.positions[i]), *(r[j]*mechanical.position_low[i]))))
    moved = Mechanical(positions, lows, np.einsum('ij,njk->nik', r, mechanical.nodal_frames),
                        np.einsum('ij,ncjk->ncik', r, mechanical.cell_rotations), mechanical.resultants.copy())
    moved_shell = native.ShellState(shell.positions@r.T+t, shell.directors@r.T)
    first = operator.evaluate(shell, mechanical, LAM).local
    second = operator.evaluate(moved_shell, moved, LAM).local
    transform = np.kron(np.eye(11), r)
    np.testing.assert_allclose(second.constraint, first.constraint, rtol=0., atol=2e-13)
    np.testing.assert_allclose(second.covector, transform @ first.covector, rtol=0., atol=2e-12)
    np.testing.assert_allclose(second.hessian, transform @ first.hessian @ transform.T, rtol=0., atol=2e-11)


def test_large_common_origin_preserves_low_translation_in_constraint():
    operator, shell, mechanical = fixture()
    position, low = mechanical.positions.copy(), mechanical.position_low.copy()
    for i in range(3): position[i, 0], low[i, 0] = split_sum((float(position[i, 0]), 1e16))
    shifted = replace(mechanical, positions=position, position_low=low)
    shell_position = shell.positions.copy(); shell_position[:, 0] += 1e16
    shifted_shell = native.ShellState(shell_position, shell.directors)
    before = operator.evaluate(shifted_shell, shifted, LAM)
    np.testing.assert_allclose(before.local.constraint, 0., rtol=0., atol=2e-14)
    moved = native.updated_native(shifted, operator.port, (.125, 0., 0., .125, 0., 0., 0., 0., 0.))
    after = operator.evaluate(shifted_shell, moved, LAM)
    assert after.local.constraint[0] == pytest.approx(.125, abs=2e-14)
    assert np.any(after.beam_position_low != 0.)


def test_midpoint_sided_cells_keep_distinct_retained_rotations():
    left, shell, mechanical = fixture(0, 1.)
    right, _, _ = fixture(1, 0.)
    cells = mechanical.cell_rotations.copy()
    cells[0, 0] = rotation_exponential((.05, 0., 0.)); cells[0, 1] = rotation_exponential((0., .07, 0.))
    moved = replace(mechanical, cell_rotations=cells)
    a = left.evaluate(shell, moved, LAM); b = right.evaluate(shell, moved, LAM)
    np.testing.assert_array_equal(a.beam_position_high, b.beam_position_high)
    assert not np.allclose(a.beam_rotation, b.beam_rotation)
    np.testing.assert_array_equal(a.beam_slots[-3:], (18, 19, 20))
    np.testing.assert_array_equal(b.beam_slots[-3:], (21, 22, 23))
    assert np.linalg.norm(a.local.constraint[3:]-b.local.constraint[3:]) > .05


def test_combined_symmetric_shear_director_tilt_declared_analytic_frame():
    operator, shell, mechanical = fixture()
    g, beta = .2, .3
    x = shell.positions.copy(); x[:, 0] += g*XY[:, 1]; x[:, 1] += g*XY[:, 0]
    ry = rotation_exponential((0., beta, 0.))
    directors = np.tile(ry @ (0., 0., 1.), (4, 1))
    phi = math.atan(g*(1.-math.cos(beta))/(1.+math.cos(beta)))
    expected = ry @ rotation_exponential((0., 0., phi))
    result = operator.evaluate(native.ShellState(x, directors), mechanical, LAM).local
    np.testing.assert_allclose(result.shell_frame, expected, rtol=0., atol=2e-14)
    assert phi > .004


def test_explicit_slot_and_type_rejections():
    operator, shell, mechanical = fixture()
    with pytest.raises(polar.TraceDomainError, match='overlap'):
        replace(operator, native_base=0)
    with pytest.raises(polar.TraceDomainError):
        replace(operator, shell_slots=np.arange(24, dtype=float))
    with pytest.raises(TypeError):
        operator.evaluate(shell, object(), LAM)
    bad = replace(mechanical, positions=np.zeros((2, 3)))
    with pytest.raises(ValueError, match='dimensions'):
        operator.evaluate(shell, bad, LAM)
    nonzero = operator.evaluate(shell, mechanical, LAM, increments=np.full(33, .01))
    with pytest.raises(polar.TraceDomainError, match='recenter'):
        native.wrench_balance(shell, mechanical, operator, nonzero)


def test_tiny_spin_large_lift_matches_actual_context_advance_and_pose():
    operator, _, mechanical = fixture()
    # Observable off-diagonal quadratic term: its displacement starts at zero,
    # so this is not a large-origin/sub-ULP loss in the output coordinate.
    port = replace(operator.port, lift=np.array((0., 2.**54, 0.)), frame=np.eye(3))
    mechanical = replace(mechanical, positions=np.zeros((3, 3)))
    delta = np.array((.125, 0., 0., .125, 0., 0., 1e-9, 1e-9, 0.))
    step = np.zeros(42); step[port.active_beam_slots()] = delta
    # Execute the actual update method with only its layout/make boundary;
    # no Context construction, issued owner state, assembly or solver is used.
    boundary = SimpleNamespace(node_ids=(0, 1, 2), elements=(None,),
                               nodal_count=18, make=lambda fields: Mechanical(**fields))
    actual = Context.advance(boundary, mechanical, step)
    copied = native.updated_native(mechanical, port, delta)
    for key, value in actual.descriptor().items():
        np.testing.assert_array_equal(copied.descriptor()[key], value)
    assert copied.cell_rotations[0, 0, 0, 1] == pytest.approx(.5e-18, rel=1e-14)
    high, low, frame = port.pose_pair(copied)
    actual_high, actual_low, actual_frame = port.pose_pair(actual)
    np.testing.assert_array_equal(high, actual_high)
    np.testing.assert_array_equal(low, actual_low)
    np.testing.assert_array_equal(frame, actual_frame)
    assert math.fsum((float(high[0]), float(low[0]), -.125)) == pytest.approx(
        .5e-18*2.**54, rel=1e-14)


def test_original_unsigned_slot_and_native_sum_overflow_rejected():
    operator, _, _ = fixture()
    slots = np.arange(24, dtype=np.uint64); slots[0] = np.uint64(2**63)
    with pytest.raises(polar.TraceDomainError, match='intp'):
        replace(operator, shell_slots=slots)
    maximum = int(np.iinfo(np.intp).max)
    with pytest.raises(polar.TraceDomainError, match='intp'):
        replace(operator, native_base=maximum)


def test_maximum_valid_global_slot_normalized_without_wrap():
    operator, shell, mechanical = fixture()
    maximum = int(np.iinfo(np.intp).max)
    base = maximum-int(operator.port.active_beam_slots().max())
    bounded = replace(operator, shell_slots=np.arange(24, dtype=np.uint64), native_base=base)
    result = bounded.evaluate(shell, mechanical, LAM)
    assert result.global_slots.dtype == np.dtype(np.intp)
    assert int(result.global_slots[-1]) == maximum
    assert np.all(result.global_slots >= 0)
    assert len(np.unique(result.global_slots)) == 33
    np.testing.assert_array_equal(result.global_slots[:24], np.arange(24))


def record_homogeneous(name, cell, result):
    destination = os.environ.get('ANYSOLVER_E01_SHEAR_RECORD')
    if destination:
        with Path(destination).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(case=name, cell=cell, result=result),
                                    default=lambda value: value.tolist())+'\n')


@pytest.mark.parametrize('cell', (0, 1))
@pytest.mark.parametrize('field,gamma', ((name, sign*.2) for name in
                         ('ux_gamma_y', 'uy_gamma_x', 'symmetric') for sign in (-1, 1)))
def test_homogeneous_inplane_frames_impose_distinct_native_shear(field, gamma, cell):
    f = np.eye(3)
    if field in ('ux_gamma_y', 'symmetric'): f[0, 1] = gamma
    if field in ('uy_gamma_x', 'symmetric'): f[1, 0] = gamma
    result = native.homogeneous_frame_comparison(f, cell=cell)
    polar_angle = (-math.atan(gamma/2) if field == 'ux_gamma_y' else
                   math.atan(gamma/2) if field == 'uy_gamma_x' else 0.)
    line_angle = 0. if field == 'ux_gamma_y' else math.atan(gamma)
    expected_polar_shear = gamma if field == 'symmetric' else gamma/math.sqrt(4+gamma**2)
    for name, angle in (('projected_polar', polar_angle), ('material_line', line_angle)):
        expected_frame = np.array(((math.cos(angle), -math.sin(angle), 0.),
                                   (math.sin(angle), math.cos(angle), 0.), (0., 0., 1.)))
        np.testing.assert_allclose(result[name]['frame'], expected_frame, rtol=0., atol=2e-14)
        np.testing.assert_allclose(result[name]['gap'], 0., rtol=0., atol=2e-14)
        # Independent physical strain oracle; raw native entries are integrated
        # half-cell chord changes and are divided by each reference length.
        chord = f @ np.array((1., 0., 0.))
        expected = expected_frame.T @ chord-np.array((1., 0., 0.))
        lengths = np.linalg.norm(np.diff(result['reference']['coordinates'], axis=0), axis=1)
        physical = result[name]['kinematics'][:6].reshape(2, 3)/lengths[:, None]
        np.testing.assert_allclose(physical, np.tile(expected, (2, 1)), rtol=0., atol=2e-14)
        np.testing.assert_allclose(result[name]['kinematics'][6:], 0., rtol=0., atol=2e-14)
        np.testing.assert_array_equal(result[name]['resultants'], np.arange(18).reshape(1, 18)*.01)
    np.testing.assert_allclose(result['projected_polar']['kinematics'][[1, 4]], expected_polar_shear,
                               rtol=0., atol=2e-14)
    np.testing.assert_allclose(result['material_line']['kinematics'][[1, 4]], 0., rtol=0., atol=2e-14)
    np.testing.assert_array_equal(result['reference']['lift'], np.zeros(3))
    assert result['reference']['retained_count'] == 42
    record_homogeneous(field+str(gamma), cell, result)


@pytest.mark.parametrize('cell', (0, 1))
def test_homogeneous_finite_rigid_rotation_reference_and_eccentric_arm(cell):
    rotation = rotation_exponential((.2, -.3, .4))
    result = native.homogeneous_frame_comparison(rotation, cell=cell, director_rotation=rotation,
                                                thickness=.4, eccentricity=.125)
    for name in ('projected_polar', 'material_line'):
        np.testing.assert_allclose(result[name]['frame'], rotation, rtol=0., atol=2e-14)
        np.testing.assert_allclose(result[name]['gap'], 0., rtol=0., atol=2e-14)
        np.testing.assert_allclose(result[name]['kinematics'], 0., rtol=0., atol=2e-14)
        np.testing.assert_allclose(result[name]['positions'],
                                   result['reference']['coordinates']@rotation.T, rtol=0., atol=2e-14)
    record_homogeneous('finite_rigid', cell, result)


@pytest.mark.parametrize('cell', (0, 1))
def test_homogeneous_independent_director_admits_transverse_beam_shear(cell):
    f = np.eye(3); f[2, 0] = .2
    result = native.homogeneous_frame_comparison(f, cell=cell)
    geometric_normal = np.cross(f[:, 0], f[:, 1]); geometric_normal /= np.linalg.norm(geometric_normal)
    assert np.linalg.norm(geometric_normal-np.array((0., 0., 1.))) > .1
    for name in ('projected_polar', 'material_line'):
        np.testing.assert_allclose(result[name]['frame'], np.eye(3), rtol=0., atol=2e-14)
        np.testing.assert_allclose(result[name]['gap'], 0., rtol=0., atol=2e-14)
        np.testing.assert_allclose(result[name]['kinematics'][:6], (0., 0., .2, 0., 0., .2),
                                   rtol=0., atol=2e-14)
        np.testing.assert_allclose(result[name]['kinematics'][6:], 0., rtol=0., atol=2e-14)
    record_homogeneous('independent_director_transverse', cell, result)
