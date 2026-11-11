"""Physical Q4/B2 reference-point solve through the experimental owner."""

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import null_space
from types import SimpleNamespace

from anyfesolver._stiffened_shell_b2_q4_point_experimental import (
    create_experimental_b2_q4_point_analysis,
    _NumericalFailure,
)
from anyfesolver import stiffened_shell
from test_b2_f10_f11_point_attachment import VERTICES, specimen


def test_q4_b2_point_owner_matches_independent_reduced_work_and_reactions():
    member, law, shell_k, beam_k = specimen()
    fixed = set(range(6)) | set(range(18, 24)) | set(range(24, 30))
    fixed.update(6*i+j for i in (1, 2) for j in (3, 4, 5))
    free = np.array([i for i in range(36) if i not in fixed])
    T = sparse.eye(36, format="csr")[:, free]
    u0 = np.zeros(36)
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PointDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            member, VERTICES, 1000., .3, .15, .4, T, u0,
            offset_local_m=(0., .08, -.03)))
    assert analysis.capabilities.status == "experimental"
    assert not analysis.capabilities.production_qualified
    T.data[:] = 0.
    u0[:] = 1.

    full_k = np.zeros((36, 36))
    full_k[:24, :24] = shell_k
    full_k[24:, 24:] = beam_k
    c = law.constraint[:, free]
    z = null_space(c)
    reduced_k = full_k[np.ix_(free, free)]
    for multiplier in (1., 2.):
        load = np.zeros(36)
        load[12] = .15*multiplier
        load[13] = -.09*multiplier
        load[32] = .11*multiplier
        load[33] = .035*multiplier
        result = analysis.solve_linear(load)
        assert result.status == "completed" and result.failure_reason is None
        assert result.factorization_count == 1
        displacement = np.asarray(result.displacements)
        lam = np.asarray(result.multiplier)
        reactions = np.asarray(result.affine_reactions)
        expected_free = z @ np.linalg.solve(
            z.T @ reduced_k @ z, z.T @ load[free])
        expected_lam = np.linalg.lstsq(
            c.T, load[free]-reduced_k @ expected_free, rcond=None)[0]
        np.testing.assert_allclose(displacement[free], expected_free,
                                   rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(lam, expected_lam,
                                   rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(displacement[list(fixed)], 0., atol=1e-13)
        np.testing.assert_allclose(law.constraint @ displacement,
                                   result.point_gap, atol=1e-11)
        np.testing.assert_allclose(reactions[free], 0., atol=1e-10)
        assert np.linalg.norm(reactions[list(fixed)]) > 0.
        assert abs(displacement @ load - 2*result.internal_strain_energy) < 1e-10
        assert len(result.beam_resultants_local) == 3
    analysis.close()


def test_q4_b2_point_solver_failure_is_not_success(monkeypatch):
    member, _, _, _ = specimen()
    fixed = set(range(6)) | set(range(18, 24)) | set(range(24, 30))
    fixed.update(6*i+j for i in (1, 2) for j in (3, 4, 5))
    free = np.array([i for i in range(36) if i not in fixed])
    analysis = create_experimental_b2_q4_point_analysis(
        member, VERTICES, shell_E=1000., shell_nu=.3,
        shell_thickness_m=.15, station=.4,
        offset_local_m=(0., .08, -.03),
        transformation=sparse.eye(36, format="csr")[:, free],
        prescribed_offset=np.zeros(36))
    def numerical_failure(_rhs):
        raise RuntimeError("Sparse solve produced NaN/Inf values")
    monkeypatch.setattr(analysis._handle, "solve", numerical_failure)
    result = analysis.solve_linear(np.zeros(36))
    assert result.status == "failed" and result.displacements == ()
    assert "NaN/Inf" in result.failure_reason
    analysis.close()


@pytest.mark.parametrize("tip_component", (1, 3))
def test_q4_b2_point_small_force_or_moment_retains_relative_equilibrium(
        tip_component):
    member, _, _, _ = specimen()
    fixed = set(range(6)) | set(range(18, 24)) | set(range(24, 30))
    fixed.update(6*i+j for i in (1, 2) for j in (3, 4, 5))
    free = np.array([i for i in range(36) if i not in fixed])
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PointDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            member, VERTICES, 1000., .3, .15, .4,
            sparse.eye(36, format="csr")[:, free], np.zeros(36),
            offset_local_m=(0., .08, -.03)))
    try:
        # Component 1 is physical Fy [N]; component 3 is physical Mx [Nm].
        # The same linear equations must preserve either applied work scale.
        load = np.zeros(36)
        load[30+tip_component] = 1.
        reference = analysis.solve_linear(load)
        small = analysis.solve_linear(load*1e-12)
        assert reference.status == small.status == "completed"
        for name in ("displacements", "multiplier", "affine_reactions"):
            np.testing.assert_allclose(
                getattr(small, name), np.asarray(getattr(reference, name))*1e-12,
                rtol=1e-8, atol=1e-24)

        # A finite zero answer cannot stand in for this nonzero force/moment.
        handle = analysis._handle
        analysis._handle = SimpleNamespace(
            solve=lambda rhs: np.zeros_like(rhs),
            factorization_count=handle.factorization_count,
            solve_count=handle.solve_count)
        failed = analysis.solve_linear(load*1e-12)
        assert failed.status == "failed"
        assert failed.displacements == ()
        assert "equilibrium" in failed.failure_reason
    finally:
        analysis.close()


def test_q4_b2_point_affine_projection_retains_cancelled_applied_work():
    member, _, _, _ = specimen()
    fixed = set(range(6)) | set(range(18, 24)) | set(range(24, 30))
    fixed.update(6*i+j for i in (1, 2) for j in (3, 4, 5))
    free = np.array([i for i in range(36) if i not in fixed])
    transformation = sparse.eye(36, format="lil")[:, free]
    transformation[free[1], 0] = 1.
    transformation[free[2], 0] = 1.
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PointDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            member, VERTICES, 1000., .3, .15, .4,
            transformation.tocsr(), np.zeros(36),
            offset_local_m=(0., .08, -.03)))
    try:
        # Three physical force contributions retain exactly 1 N conjugate
        # to the first stored affine translation, hence its nonzero work.
        load = np.zeros(36)
        load[free[0]] = 1e16
        load[free[1]] = 1.
        load[free[2]] = -1e16
        projected = analysis._project(load)
        assert projected[0] == 1.
        assert np.count_nonzero(projected[1:]) == 2
        handle = analysis._handle
        analysis._handle = SimpleNamespace(
            solve=lambda rhs: np.zeros_like(rhs),
            factorization_count=handle.factorization_count,
            solve_count=handle.solve_count)
        failed = analysis.solve_linear(load)
        assert failed.status == "failed" and failed.displacements == ()
        assert "resolution" in failed.failure_reason or "equilibrium" in failed.failure_reason
    finally:
        analysis.close()


@pytest.mark.parametrize("coefficient", (1., .3))
def test_q4_b2_point_subnormal_affine_work_fails_without_hiding_programming_error(
        coefficient, monkeypatch):
    member, _, _, _ = specimen()
    fixed = set(range(6)) | set(range(18, 24)) | set(range(24, 30))
    fixed.update(6*i+j for i in (1, 2) for j in (3, 4, 5))
    free = np.array([i for i in range(36) if i not in fixed])
    transformation = sparse.eye(36, format="lil")[:, free]
    transformation[free[0], 0] = coefficient
    transformation[free[1], 0] = coefficient
    transformation[free[2], 0] = 1.
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PointDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            member, VERTICES, 1000., .3, .15, .4,
            transformation.tocsr(), np.zeros(36),
            offset_local_m=(0., .08, -.03)))
    try:
        # Unity emits a subnormal; .3 would erase a nonzero stored product.
        # Neither can satisfy the declared relative applied-work resolution.
        load = np.zeros(36)
        load[free[0]] = np.nextafter(0., 1.)
        with pytest.raises(_NumericalFailure, match="affine projection.*not representable"):
            analysis._project(load)
        failed = analysis.solve_linear(load)
        assert failed.status == "failed" and failed.displacements == ()
        assert "affine projection" in failed.failure_reason
        assert "not representable" in failed.failure_reason
        assert failed.solve_count == 0

        def programming_error(_values):
            raise ValueError("injected affine projection programming error")

        monkeypatch.setattr(analysis, "_project", programming_error)
        with pytest.raises(ValueError, match="programming error"):
            analysis.solve_linear(np.zeros(36))
    finally:
        analysis.close()


def test_q4_b2_point_extreme_finite_load_cannot_report_success():
    member, _, _, _ = specimen()
    fixed = set(range(6)) | set(range(18, 24)) | set(range(24, 30))
    fixed.update(6*i+j for i in (1, 2) for j in (3, 4, 5))
    free = np.array([i for i in range(36) if i not in fixed])
    analysis = create_experimental_b2_q4_point_analysis(
        member, VERTICES, shell_E=1000., shell_nu=.3,
        shell_thickness_m=.15, station=.4,
        offset_local_m=(0., .08, -.03),
        transformation=sparse.eye(36, format="csr")[:, free],
        prescribed_offset=np.zeros(36))
    load = np.zeros(36)
    load[33] = np.finfo(float).max
    with np.errstate(over="ignore", invalid="ignore"):
        result = analysis.solve_linear(load)
    assert result.status == "failed"
    assert result.displacements == ()
    analysis.close()
