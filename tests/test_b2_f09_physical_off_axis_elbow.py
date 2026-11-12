"""Bounded F09 physical-section graph check; no production admission claim."""

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import math
import runpy

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import diags
from scipy.sparse.linalg import splu

PROBE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts"
                           / "probe_b3_ge_e4_pl_g0_f09_graph_reversal.py"))

from anyfesolver._b2_physical_linear_batch_dev import CompiledB2AffineBatch
from anyfesolver._b2_physical_flexibility_linear import (
    LinearOperator, LinearSection, _flexibility,
)


def _physical_section(width=.040, depth=.012):
    """A physical rectangle with principal axes rotated 31 degrees."""
    young, poisson = 210e9, .3
    shear = young/(2*(1+poisson))
    area = width*depth
    iy, iz = width*depth**3/12, depth*width**3/12
    # Standard thin-rectangle Saint-Venant approximation; this diagnostic
    # tests a declared engineering section, not an exact 3-D torsion solution.
    torsion = width*depth**3*(1/3-.21*depth/width*(1-depth**4/(12*width**4)))
    principal = np.array([young*area, shear*area*5/6,
                          shear*area*5/6, shear*torsion,
                          young*iy, young*iz])
    angle = math.radians(31)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    transform = np.eye(6)
    transform[np.ix_([1, 2], [1, 2])] = rotation
    transform[np.ix_([4, 5], [4, 5])] = rotation
    stiffness = transform@np.diag(principal)@transform.T
    # The analytical section is symmetric; construct both stored triangles
    # from the same rounded value for the definition's exact symmetry contract.
    stiffness = np.triu(stiffness)+np.triu(stiffness, 1).T
    compliance = transform@np.diag(1/principal)@transform.T
    assert np.max(np.abs(stiffness@compliance-np.eye(6))) <= (
        32*np.finfo(float).eps)
    return stiffness, compliance


@pytest.mark.parametrize("width,depth", ((.08, .04), (.04, .012), (.012, .003)))
@pytest.mark.parametrize("length", (2., 1.5))
def test_physical_basic_actions_survive_binary64_storage(width, depth, length):
    stiffness, _ = _physical_section(width, depth)
    section = LinearSection.generalized(stiffness)
    operator = LinearOperator(length, section)
    exact_basic = operator._basic_inverse
    rounded_basic = np.array(exact_basic, dtype=float)
    diagonal = np.array([float(exact_basic[i][i]) for i in range(6)])
    assert np.isfinite(rounded_basic).all()
    assert np.isfinite(diagonal).all() and np.all(diagonal > 0.)
    normalized = rounded_basic/np.sqrt(diagonal)[:, None]/np.sqrt(diagonal)[None, :]
    assert np.linalg.eigvalsh(normalized)[0] > 0.

    # Each isolated basic deformation induces 1000 N or 1000 Nm on its
    # conjugate diagonal; mixed signs probe cancellation in coupled modes.
    basic_scales = [Fraction(1000)/exact_basic[i][i] for i in range(6)]
    actions = [tuple(1 if i == j else 0 for i in range(6))
               for j in range(6)]
    actions += [(1, -1, 1, -1, 1, -1),
                (-1, 1, -1, 1, -1, 1),
                (0, 0, 1, -1, 1, -1)]
    for action in actions:
        deformation = [basic_scales[i]*action[i] for i in range(6)]
        exact_force = [sum(exact_basic[i][j]*deformation[j]
                           for j in range(6)) for i in range(6)]
        stored_deformation = np.array([float(value) for value in deformation])
        stored_force = rounded_basic@stored_deformation
        applied_scale_error = np.max(np.abs(stored_force-np.array(
            [float(value) for value in exact_force]))/1000.)
        exact_work = float(sum(deformation[i]*exact_force[i]
                               for i in range(6)))
        rounded_work = float(stored_deformation@stored_force)
        assert exact_work > 0. and rounded_work > 0.
        assert applied_scale_error < 1e-8
        assert abs(rounded_work-exact_work)/exact_work < 1e-8


def _basic_rounding_response_bound(operator):
    """Exact norm bound for the inverse response of a rounded basic law."""
    exact_k = operator._basic_inverse
    exact_f = _flexibility(operator.length, operator._compliance)
    assert all(sum(exact_k[i][j]*exact_f[j][k] for j in range(6))
               == int(i == k) for i in range(6) for k in range(6))
    # Exact powers of two normalize the declared SI coordinate magnitudes
    # without introducing a second rounding in this bound.
    scales = [Fraction(2)**(-round(math.log2(float(exact_k[i][i]))/2))
              for i in range(6)]
    inverse_norm = max(sum(abs(exact_f[i][j]/(scales[i]*scales[j]))
                           for j in range(6)) for i in range(6))
    perturbation_norm = max(sum(abs(scales[i]*scales[j]*(
        Fraction(float(exact_k[i][j]))-exact_k[i][j]))
                                for j in range(6)) for i in range(6))
    return inverse_norm*perturbation_norm


@pytest.mark.parametrize("width,depth", ((.08, .04), (.04, .012), (.012, .003)))
@pytest.mark.parametrize("length", (2., 1.5))
def test_physical_basic_rounding_inverse_response_bound(width, depth, length):
    stiffness, _ = _physical_section(width, depth)
    operator = LinearOperator(length, LinearSection.generalized(stiffness))
    eta = _basic_rounding_response_bound(operator)
    assert 0 < eta < 1
    assert eta/(1-eta) < Fraction(1, 10**8)


def test_weak_generalized_basic_rounding_inverse_response_bound():
    stiffness = np.diag([12., 2.**-40, 2.**-39, 4., 3., 5.])
    stiffness[0, 4] = stiffness[4, 0] = .5
    stiffness[1, 2] = stiffness[2, 1] = 2.**-42
    stiffness[3, 5] = stiffness[5, 3] = .25
    operator = LinearOperator(2., LinearSection.generalized(stiffness))
    eta = _basic_rounding_response_bound(operator)
    # This local rounded law is certifiable even though the existing
    # two-member binary64 graph fails the per-load physical balance screen.
    assert 0 < eta < 1
    assert eta/(1-eta) < Fraction(1, 10**8)


def _cut_work(members, tip, compliance):
    """Integrate physical cut wrenches using derived sectional compliance."""
    force, moment = np.asarray(tip[:3]), np.asarray(tip[3:])
    tip_position = np.asarray(PROBE["POINTS"][2])
    sign = np.diag([-1., 1., -1., -1., 1., -1.])
    total = 0.
    for member in members:
        start, end = map(np.asarray, member.reference_points_m)
        frame = member.local_frame_rows()
        local_compliance = (sign@compliance@sign if member.node_ids == (2, 1)
                            else compliance)
        density = []
        for fraction in (0., .5, 1.):
            cut = start+fraction*(end-start)
            cut_moment = moment+np.cross(tip_position-cut, force)
            local = np.concatenate((frame@force, frame@cut_moment))
            density.append(float(local@local_compliance@local))
        total += member.length_m*(density[0]+4*density[1]+density[2])/12
    return total


def _cut_tip_response(members, tip, compliance):
    """Differentiate independent cut strain energy by each physical tip load."""
    force, moment = np.asarray(tip[:3]), np.asarray(tip[3:])
    tip_position = np.asarray(PROBE["POINTS"][2])
    sign = np.diag([-1., 1., -1., -1., 1., -1.])
    result = np.zeros(6)
    basis = np.eye(6)
    for member in members:
        start, end = map(np.asarray, member.reference_points_m)
        frame = member.local_frame_rows()
        local_compliance = (sign@compliance@sign if member.node_ids == (2, 1)
                            else compliance)
        for fraction, weight in ((0., 1.), (.5, 4.), (1., 1.)):
            cut = start+fraction*(end-start)
            lever = tip_position-cut
            local = np.concatenate((frame@force,
                                    frame@(moment+np.cross(lever, force))))
            strain = local_compliance@local
            for i, unit in enumerate(basis):
                df, dm = unit[:3], unit[3:]
                dlocal = np.concatenate((frame@df,
                                         frame@(dm+np.cross(lever, df))))
                result[i] += member.length_m*weight/6*float(dlocal@strain)
    return result


def _exact_product_residual(matrix, solution, rhs):
    """Multiply the stored binary64 state by the stored binary64 matrix exactly."""
    dense = matrix.toarray()
    return np.array([float(sum(Fraction(float(a))*Fraction(float(x))
                               for a, x in zip(row, solution))
                           - Fraction(float(target)))
                     for row, target in zip(dense, rhs)])


def _high_precision_rounded_matrix_solve(matrix, rhs):
    """Solve the stored matrix independently at 100 decimal digits."""
    with localcontext() as context:
        context.prec = 100
        rows = [[Decimal(float(value)) for value in row]
                + [Decimal(float(target))]
                for row, target in zip(matrix.toarray(), rhs)]
        n = len(rows)
        for k in range(n):
            pivot = max(range(k, n), key=lambda i: abs(rows[i][k]))
            rows[k], rows[pivot] = rows[pivot], rows[k]
            denominator = rows[k][k]
            assert denominator != 0
            for i in range(k+1, n):
                factor = rows[i][k]/denominator
                for j in range(k+1, n+1):
                    rows[i][j] -= factor*rows[k][j]
                rows[i][k] = Decimal(0)
        answer = [Decimal(0)]*n
        for i in range(n-1, -1, -1):
            answer[i] = (rows[i][-1]-sum(rows[i][j]*answer[j]
                                         for j in range(i+1, n)))/rows[i][i]
        return np.array([float(value) for value in answer])


@pytest.mark.parametrize("width,depth", ((.08, .04), (.04, .012), (.012, .003)))
def test_off_axis_rectangular_section_graph_work_reversal_and_balance(width, depth):
    stiffness, compliance = _physical_section(width, depth)
    assert abs(stiffness[4, 5]) > 0.1*math.sqrt(stiffness[4, 4]*stiffness[5, 5])
    section = LinearSection.generalized(stiffness)
    original = tuple(replace(member, section_id="rotated-rectangle", section=section)
                     for member in PROBE["members"]())
    sign = np.diag([-1., 1., -1., -1., 1., -1.])
    reversed_base = PROBE["members"](True)
    reversed_members = (
        replace(reversed_base[0], section_id="rotated-rectangle", section=section),
        replace(reversed_base[1], section_id="rotated-rectangle-reversed",
                section=LinearSection.generalized(sign@stiffness@sign)))
    matrices = [PROBE["assemble"](members)
                for members in (original, reversed_members)]
    np.testing.assert_allclose(matrices[0].toarray(), matrices[1].toarray(),
                               rtol=1e-10, atol=1e-5)

    free = np.arange(6, 18)
    mixed_tips = ((100., -80., 60., 12., -9., 7.),
                  (-45., 75., -95., -5., 8., -11.))
    pure_tips = tuple(tuple(1000. if i == j else 0. for i in range(6))
                      for j in range(6))
    for tip in mixed_tips + pure_tips:
        load = np.zeros(18)
        load[12:] = tip
        expected = _cut_work(original, tip, compliance)
        expected_tip = _cut_tip_response(original, tip, compliance)
        assert expected > 0.
        assert abs(.5*float(np.asarray(tip)@expected_tip)-expected)/expected < 1e-12
        assert abs(_cut_work(reversed_members, tip, compliance)-expected)/expected < 1e-12
        np.testing.assert_allclose(_cut_tip_response(reversed_members, tip, compliance),
                                   expected_tip, rtol=1e-12, atol=1e-12)
        for members, matrix in zip((original, reversed_members), matrices):
            reduced = matrix[free, :][:, free].tocsc()
            rhs = load[free]
            scale = 1/np.sqrt(reduced.diagonal())
            answer = scale*splu((diags(scale)@reduced@diags(scale)).tocsc()).solve(scale*rhs)
            rounded_reference = _high_precision_rounded_matrix_solve(reduced, rhs)
            # Pure actions may have physically zero cross-components with
            # binary64 solve noise; retain the tighter mixed-case floor.
            response_floor = 1e-12 if tip in mixed_tips else 1e-10
            np.testing.assert_allclose(answer, rounded_reference,
                                       rtol=1e-9, atol=response_floor)
            np.testing.assert_allclose(answer[6:], expected_tip,
                                       rtol=1e-8, atol=1e-10)
            stored_work = .5*float(rhs@answer)
            assert abs(stored_work-expected)/expected < 1e-9
            assert abs(PROBE["physical_work"](members, tip)-expected)/expected < 1e-9
            residual = _exact_product_residual(reduced, answer, rhs).reshape(2, 6)
            # Retain the tighter mixed-load scales; pure cases use declared
            # 1000 N and 1000 Nm floors for their undriven wrench channel.
            force_scale = max(map(abs, tip[:3])) or 1000.
            moment_scale = max(map(abs, tip[3:])) or 1000.
            force_error = np.max(np.abs(residual[:, :3]))/force_scale
            moment_error = np.max(np.abs(residual[:, 3:]))/moment_scale
            assert max(force_error, moment_error) < 1e-8
            full = np.zeros(18)
            full[free] = answer
            for member in members:
                indices = [6*node+j for node in member.node_ids for j in range(6)]
                recovered = member.local_operator().recover_sections(
                    member.local_dof_map()@full[indices], stations=(0., .5, 1.))
                start, end = map(np.asarray, member.reference_points_m)
                frame = member.local_frame_rows()
                local_compliance = (sign@compliance@sign if member.node_ids == (2, 1)
                                    else compliance)
                for i, fraction in enumerate((0., .5, 1.)):
                    cut = start+fraction*(end-start)
                    cut_moment = np.asarray(tip[3:])+np.cross(
                        np.asarray(PROBE["POINTS"][2])-cut, tip[:3])
                    expected_resultant = np.concatenate((frame@tip[:3],
                                                          frame@cut_moment))
                    # The reversed member's positive internal cut is on the
                    # opposite side of the free body from the physical tip.
                    if member.node_ids == (2, 1):
                        expected_resultant = -expected_resultant
                    np.testing.assert_allclose(recovered.resultants[i],
                                               expected_resultant, rtol=1e-8, atol=1e-7)
                    np.testing.assert_allclose(recovered.strains[i],
                                               local_compliance@expected_resultant,
                                               rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("width,depth", ((.08, .04), (.04, .012), (.012, .003)))
def test_off_axis_rectangle_private_batch_reuses_one_factorization(width, depth):
    stiffness, _ = _physical_section(width, depth)
    members = tuple(replace(member, section_id="rotated-rectangle",
                            section=LinearSection.generalized(stiffness))
                    for member in PROBE["members"]())
    transform = sparse.eye(18, format="csr")[:, 6:]
    batch = CompiledB2AffineBatch(
        tuple(member.compile_local() for member in members),
        transform, np.zeros(18), np.arange(12, 18), max_chunk=2)
    tips = ((100., -80., 60., 12., -9., 7.),
            (-45., 75., -95., -5., 8., -11.))
    cases = []
    for i, tip in enumerate(tips):
        load = np.zeros(18)
        load[12:] = tip
        cases.append((f"rect-{i}", load))
    result = batch.solve_many(cases, recover_members={"m0": (0., .5, 1.)})
    assert result.status == "completed"
    assert result.factorization_count == 1
    assert result.maximum_rhs_chunk == 2
    for case, tip in zip(result.cases, tips):
        assert len(case.members) == 1
        reduced = batch.full_stiffness[6:, 6:]
        rhs = np.zeros(12)
        rhs[6:] = tip
        exact = _high_precision_rounded_matrix_solve(reduced, rhs)
        np.testing.assert_allclose(case.values, exact[6:], rtol=1e-9, atol=1e-12)
    batch.close()
