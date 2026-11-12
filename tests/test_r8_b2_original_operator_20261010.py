"""Prospective small independent original-target and actual caller controls.

No large case/campaign, backend switch, precision change or new acceptance norm.
These tests have not been executed under the source-only preparation authority.
"""
from fractions import Fraction
import numpy as np
import pytest
from scipy import sparse, linalg

from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
from anyfesolver._stiffened_shell_b2_q4_panel_experimental import (
    ExperimentalB2Q4PanelAnalysis, _NumericalFailure,
)
from examples.stiffened_shell_b2_physical_rigid_footprint import definition, preload


class Factor:
    """One tiny reusable factor, recording the actual production solve RHS."""
    def __init__(self, a):
        self.factor = linalg.lu_factor(a)
        self.factorization_count = 1
        self.solve_count = 0
        self.rhs = []
        self.answers = []
        self.fraction = 1.

    def solve(self, rhs):
        self.solve_count += 1
        self.rhs.append(np.array(rhs, copy=True))
        value = self.fraction*linalg.lu_solve(self.factor, rhs)
        self.answers.append(value.copy())
        return value


def toy(*, rounded_change=0., offset=0., affine=False, scales=None):
    """Physical K=2I, C ties q2 to prescribed u11; all units explicit SI.

    The fixture perturbs the stored first equation to expose a represented
    original/stored mismatch. It is not a claim about the large-case magnitude.
    """
    owner = object.__new__(ExperimentalB2Q4PanelAnalysis)
    t = np.eye(12)[:, :11]
    if affine:
        t[1, 0] = .5
    owner._K = sparse.eye(12, format="csr")*2
    owner._C = sparse.csr_matrix(([1., 1.], ([0, 0], [2, 11])), shape=(1, 12))
    owner._T = sparse.csr_matrix(t)
    owner._Tt = owner._T.T.tocsr()
    row_counts = np.diff(owner._Tt.indptr)
    owner._projection_multi = np.flatnonzero(row_counts > 1)
    owner._projection_single = np.flatnonzero(row_counts == 1)
    owner._u0 = np.zeros(12)
    owner._u0[11] = offset
    owner._cu0 = np.asarray(owner._C @ owner._u0).reshape(-1)
    owner._factor_scale = np.ones(12) if scales is None else np.array(scales, dtype=float)
    owner._physical_singletons = ()
    owner._physical_footprint_law = True
    owner._point_trace_policy = owner._H = None
    owner.point_ids = ()
    owner._abs_K, owner._abs_C = abs(owner._K), abs(owner._C)
    owner._abs_Ct, owner._abs_Tt = abs(owner._C.T), abs(owner._Tt)
    a = np.zeros((12, 12))
    a[:11, :11] = t.T @ (2*t)
    a[:11, 11] = a[11, :11] = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    a[0, 0] += rounded_change
    scaled_a = owner._factor_scale[:, None]*a*owner._factor_scale[None, :]
    owner._physical_scaled_kkt = sparse.csr_matrix(scaled_a)
    owner._handle = Factor(scaled_a)
    f = np.zeros(12)
    f[0] = 4.
    b = np.r_[t.T @ (f-2*owner._u0), -offset]
    return owner, f, owner._factor_scale*b


def exact_work_identity(owner, answer, f):
    """Independent exact rational work, not production residual evaluation."""
    t = [[Fraction(float(v)) for v in row] for row in owner._T.toarray()]
    q, lam = [Fraction(float(v)) for v in answer[:11]], Fraction(float(answer[11]))
    u0 = [Fraction(float(v)) for v in owner._u0]
    u = [sum((a*b for a, b in zip(row, q)), Fraction())+v for row, v in zip(t, u0)]
    applied = [Fraction(float(v)) for v in f]
    gap = u[2]+u[11]
    reaction = [2*v-g for v, g in zip(u, applied)]
    reaction[2] += lam
    reaction[11] += lam
    projected = [sum((t[i][j]*reaction[i] for i in range(12)), Fraction()) for j in range(11)]
    defect = sum((2*v*v-v*g for v, g in zip(u, applied)), Fraction())
    prescribed = sum((v*r for v, r in zip(u0, reaction)), Fraction())
    return defect, sum((v*r for v, r in zip(q, projected)), Fraction())-lam*gap+prescribed


def test_original_target_falsifies_zero_stored_residual():
    owner, load, rhs = toy(rounded_change=2e-5)
    z = owner._handle.solve(rhs)
    stored = owner._physical_scaled_kkt @ z-rhs
    assert np.max(abs(stored)) <= 1e-14
    assert owner._physical_work_ratio(z, load) > 1.
    before = owner._physical_original_residual(z, load)
    assert abs(before[0]) > 1e-5
    candidate = owner._correct_physical_equilibrium(z, rhs, load)
    np.testing.assert_array_equal(owner._handle.rhs[-1], -before)
    assert owner._handle.factorization_count == 1 and owner._handle.solve_count == 2
    assert owner._physical_work_ratio(candidate, load) <= 1.
    assert np.max(abs(owner._physical_original_residual(candidate, load))) < np.max(abs(before))


def test_correction_rhs_and_update_use_restored_candidate():
    owner, load, rhs = toy(scales=(.5, 1., 4., 1., 1., 1., 1., 1., 1., 1., 1., 2.))
    z = owner._handle.solve(rhs)
    z[0] = 1.9/owner._factor_scale[0]
    z[2] = 5e-15/owner._factor_scale[2]  # Roundoff-sized physical singleton.
    owner._physical_singletons = ((0, 2, 1.),)
    restore = owner._restore_physical_singletons
    scored = restore(owner._factor_scale*z)
    target = owner._physical_original_residual(scored, load)
    incoming = []

    def capture_before_restoration(value):
        incoming.append(value.copy())
        return restore(value)

    owner._restore_physical_singletons = capture_before_restoration
    made = owner._correct_physical_equilibrium(z, rhs, load)
    np.testing.assert_array_equal(owner._handle.rhs[-1], -owner._factor_scale*target)
    assert owner._handle.rhs[-1][2] == owner._handle.rhs[-1][-1] == 0.
    assert len(incoming) == 2
    expected = scored+owner._factor_scale*owner._handle.answers[-1]
    np.testing.assert_array_equal(incoming[-1], expected)
    assert incoming[-1][2] == 0.  # No final restoration/tolerance can mask this.
    regressed = owner._factor_scale*(z+owner._handle.answers[-1])
    assert regressed[2] == 5e-15
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(regressed, expected)
    np.testing.assert_allclose(made, np.r_[[2., *([0.]*10)], 0.], rtol=0, atol=1e-14)


def test_affine_offset_and_multiplier_work_are_preserved():
    owner, load, rhs = toy(offset=.25, affine=True)
    trial = np.zeros(12)
    trial[0], trial[2], trial[11] = 1., -.125, .5
    # Direct Cartesian equations: u0=1,u1=.5,u2=-.125,u11=.25.
    # Projected first force=(-2)+.5*(1)=-1.5; q2 force=.25; gap=.125.
    target = owner._physical_original_residual(trial, load)
    assert target[0] == -1.5 and target[2] == .25 and target[-1] == .125
    assert exact_work_identity(owner, trial, load)[0] == exact_work_identity(owner, trial, load)[1]
    made = owner._correct_physical_equilibrium(trial, rhs, load)
    np.testing.assert_allclose(owner._physical_original_residual(made, load), 0., rtol=0, atol=1e-14)
    assert exact_work_identity(owner, made, load)[0] == exact_work_identity(owner, made, load)[1]


@pytest.mark.parametrize("foreign", ("compliance", "loaded", "legacy"))
def test_foreign_constraint_contract_refused(foreign):
    owner, load, _ = toy()
    if foreign == "compliance":
        owner._H = sparse.eye(1, format="csr")
    elif foreign == "loaded":
        owner._point_trace_policy = "foreign-nonzero-d"
    else:
        owner._physical_footprint_law = False
    with pytest.raises(_NumericalFailure, match="separate physical footprint"):
        owner._physical_original_residual(np.zeros(12), load)


def test_nonimproving_correction_remains_typed_failure():
    owner, load, rhs = toy()
    trial = owner._handle.solve(rhs)
    trial[0] *= .9
    owner._handle.fraction = 0.
    with pytest.raises(_NumericalFailure, match="did not improve"):
        owner._correct_physical_equilibrium(trial, rhs, load)
    assert owner._handle.factorization_count == 1 and owner._handle.solve_count == 2


@pytest.mark.parametrize("where", ("answer", "load"))
def test_nonfinite_original_target_refused_without_solve(where):
    owner, load, _ = toy()
    answer = np.zeros(12)
    (answer if where == "answer" else load)[0] = np.nan
    with pytest.raises(_NumericalFailure, match="finite original physical"):
        owner._physical_original_residual(answer, load)
    assert owner._handle.solve_count == 0


class BiasedFirstSolve:
    def __init__(self, handle):
        self.handle, self.calls = handle, 0

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def solve(self, rhs):
        self.calls += 1
        value = self.handle.solve(rhs)
        return value*(1.-1e-5) if self.calls == 1 else value


def test_actual_public_batch_preserves_prefix_factor_reuse_and_inputs():
    owner = api.create_analysis(definition=definition())
    token = CancellationToken()
    load = preload()
    before = load.copy()
    try:
        assert owner._H is None and not owner.point_ids and owner._point_trace_policy is None
        owner._handle = BiasedFirstSolve(owner._handle)
        result = owner.solve_linear_many((("corrected", load), ("not-run", 2*load)),
                    selected_dofs=(12, 30, 48), recover_members={}, cancellation_token=token,
                    on_case=lambda *_: token.cancel("preserve corrected completed prefix"))
        assert result.status == "cancelled" and len(result.cases) == 1
        assert result.factorization_count == 1 and result.solve_count == 2
        assert owner._handle.calls == 2
        np.testing.assert_array_equal(load, before)
        later = owner.solve_linear(load, recover_members={})
        assert later.status == "completed", later.failure_reason
        assert later.factorization_count == 1
    finally:
        owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.solve_linear(load)


def test_legacy_point_caller_does_not_enter_physical_correction(monkeypatch):
    from test_stiffened_shell_b2_q4_panel_experimental import _panel
    nodes, shells, members, points, free = _panel()
    plan = api.B2Q4PanelDefinition(api.B2_E4_PL_FAST_LINEAR_V1, nodes, shells, members, points,
                                  sparse.eye(54, format="csr")[:, free], np.zeros(54))
    owner = api.create_analysis(definition=plan)
    try:
        assert not owner._physical_footprint_law
        monkeypatch.setattr(owner, "_physical_original_residual", lambda *_: (_ for _ in ()).throw(AssertionError("foreign correction entered")))
        result = owner.solve_linear(np.zeros(owner._u0.shape), recover_members={})
        assert result.status == "completed", result.failure_reason
    finally:
        owner.close()


def test_loaded_body_affine_route_preserves_nonzero_H_and_d(monkeypatch):
    from test_b2_loaded_point_owner import _definition
    owner = api.create_analysis(definition=_definition(affine=True))
    try:
        assert not owner._physical_footprint_law and owner._H is not None and owner._H.nnz
        monkeypatch.setattr(owner, "_physical_original_residual", lambda *_: (_ for _ in ()).throw(AssertionError("loaded correction entered")))
        load = owner.body_acceleration_load((0., 0., -1.))
        body = np.zeros(owner._C.shape[0])
        for compiled in owner._loaded_members.values():
            values, _ = compiled.body_constraint(load.acceleration_m_s2)
            slots = [6*i+j for i in compiled.point_indices for j in range(6)]
            body[slots] = values
        assert np.count_nonzero(body)
        result = owner.solve_linear(load, recover_members={})
        assert result.status == "completed", result.failure_reason
        u = np.asarray(result.displacements)
        lam = np.concatenate([values for _, values in result.point_multipliers])
        original_gap = owner._C @ u-owner._H @ lam+body
        reported_gap = np.concatenate([values for _, values in result.point_gaps])
        np.testing.assert_allclose(reported_gap, original_gap, rtol=0, atol=1e-12)
    finally:
        owner.close()
