"""Independent exact-oracle and refusal checks for the initial-contact seed.

The oracle here is an exact scalar constrained-contact problem solved by
hand: two free degrees, one control row on the first degree and one
normalized-gap row. Its closed-form constrained response, equilibrated gap
response H and nonnegative pressure force are derived on paper, not from
the production contact kernel, so agreement is an independent witness.

Dimensions are explicit: the pressure force z has N, the normalized
(support-weighted mean) gap has m and the equilibrated gap response
H = G K_constrained^{-1} G.T has m/N. The weighted-mean gap closure is
checked exactly: at the accepted seed the active mode's linearized weighted
gap is exactly zero and its pressure force is the exact hand value.

Refusal cases are exact too: a singular constrained operator, an
over-cap mode count, an indefinite gap response and an asymmetric gap
response each refuse explicitly, and a refused proposal mutates none of
the caller's blocks.
"""
import numpy as np
from scipy import sparse

from anyfesolver.general_contact.initial_contact_seed import (
    InitialContactSeedRefusal,
    MAXIMUM_SEED_PRESSURE_MODES,
    propose_linearized_kkt_seed,
)


def _blocks(k1, k2, gap0, *, f1=0.0, f2=1.0, target=0.0,
            g1=0.5, g2=-1.0, support=(1.0,), gap_rows=None):
    """Hand-solvable scalar constrained blocks in the Newton convention.

    The control constrains free degree 1 (rows = [[1, 0]]); the normalized
    gap row is (g1, g2) so the gap closes as dof 2 moves. Signs follow the
    live residual convention R = R_struct - G.T z.
    """
    kqq = sparse.diags([float(k1), float(k2)], format="csr")
    control_rows = np.asarray([[1.0, 0.0]])
    if gap_rows is None:
        gap_q = np.asarray([[float(g1), float(g2)]])
    else:
        gap_q = np.asarray(gap_rows, dtype=float)
    return dict(
        kqq=kqq,
        kqz=sparse.csr_matrix(-gap_q.T),
        gap_q=gap_q,
        control_reactions=control_rows.T.copy(),
        control_rows=control_rows,
        residual=np.asarray([float(f1), float(f2)]),
        constraint=np.asarray([float(target)]),
        q=np.zeros(2),
        normalized_gap=np.asarray([float(gap0)]*(1 if gap_rows is None
                                               else len(gap_rows))),
        support=np.asarray(support, dtype=float),
    )


def _propose(blocks):
    return propose_linearized_kkt_seed(signature="oracle", **blocks)


def test_exact_scalar_active_contact_oracle():
    """Closed-form active constrained contact: exact z, gap closure, equilibrium.

    Free response: dq1 = t = 0, dmu = k1*t - f1 = 0, dq2 = f2/k2 = 1/3.
    Weighted-mean free gap g_free = gap0 + g2*dq2 = 0.01 - 1/3 < 0, so the
    exact pressure force is z* = -g_free/H with H = g2^2/k2 = 1/3 m/N:
    z* = (1/3 - 0.01)*3 = 0.97 N. The seed closes the active mode's
    linearized weighted gap exactly and holds the linearized equilibrium
    and control constraint exactly.
    """
    blocks = _blocks(2.0, 3.0, 0.01)
    proposal = _propose(blocks)
    assert not proposal.refused
    z_exact = 0.97
    np.testing.assert_allclose(proposal.nodal_forces, [z_exact],
                               rtol=0., atol=1e-12)
    # pressure force z has N; the record carries its exact magnitude
    np.testing.assert_allclose(proposal.diagnostics["seed_pressure_force_n"],
                               z_exact, rtol=0., atol=1e-12)
    # seed pose: free response plus the pressure response of dof 2
    np.testing.assert_allclose(proposal.q, [0.0, 0.01], rtol=0., atol=1e-12)
    # control reaction: dmu = k1*t - f1 - g1*z
    np.testing.assert_allclose(proposal.reactions, [-0.485], rtol=0., atol=1e-12)
    # H has m/N: its single eigenvalue is g2^2/k2 = 1/3
    np.testing.assert_allclose(
        proposal.diagnostics["gap_response_eigenvalue_max"], 1.0/3.0,
        rtol=0., atol=1e-12)
    # normalized gap has m: the free weighted-mean gap is 0.01 - 1/3
    np.testing.assert_allclose(proposal.diagnostics["free_gap_min_m"],
                               0.01 - 1.0/3.0, rtol=0., atol=1e-12)
    # one constrained factorization, free + pressure solves, bounded QP
    assert proposal.diagnostics["constrained_factorizations"] == 1
    assert proposal.diagnostics["constrained_solves"] == 2
    assert proposal.diagnostics["qp_terminal"] == "optimal"
    assert proposal.diagnostics["qp_iterations"] >= 1
    assert proposal.diagnostics["cost_s"] >= 0.0
    # exact linearized equilibrium at the seed:
    # kqq dq - C dmu - G.T z = residual
    dq = proposal.q - blocks["q"]
    equilibrium = (np.asarray(blocks["kqq"] @ dq).reshape(-1)
                   - blocks["control_reactions"] @ proposal.reactions
                   - blocks["gap_q"].T @ proposal.nodal_forces)
    np.testing.assert_allclose(equilibrium, blocks["residual"],
                               rtol=0., atol=1e-12)
    # exact control constraint: rows @ seed_q = target
    np.testing.assert_allclose(blocks["control_rows"] @ proposal.q,
                               blocks["constraint"], rtol=0., atol=1e-12)
    # weighted-mean gap closure: the active mode's linearized gap is exact 0
    linearized_gap = (blocks["normalized_gap"]
                      + blocks["gap_q"] @ dq)
    np.testing.assert_allclose(linearized_gap, [0.0], rtol=0., atol=1e-12)


def test_exact_scalar_open_contact_oracle():
    """Open contact: positive free gap gives exactly zero pressure force."""
    blocks = _blocks(2.0, 3.0, 1.0)
    proposal = _propose(blocks)
    assert not proposal.refused
    np.testing.assert_allclose(proposal.nodal_forces, [0.0], rtol=0., atol=0.0)
    # the seed is exactly the contact-free constrained response
    np.testing.assert_allclose(proposal.q, [0.0, 1.0/3.0], rtol=0., atol=1e-12)
    np.testing.assert_allclose(proposal.reactions, [0.0], rtol=0., atol=1e-12)
    assert proposal.diagnostics["qp_iterations"] == 0
    assert proposal.diagnostics["qp_terminal"] == "optimal"


def test_exact_scalar_control_constrained_oracle():
    """A nonzero control target is met exactly and shifts the exact pressure.

    Free response: dq1 = t = 0.02, dmu = k1*t - f1 = 0.04, dq2 = 1/3.
    g_free = 0.01 + 0.5*0.02 - 1/3, so z* = -g_free/H = (1/3 - 0.02)*3
    = 0.94 N and the seed pose is (0.02, 0.02).
    """
    blocks = _blocks(2.0, 3.0, 0.01, target=0.02)
    proposal = _propose(blocks)
    assert not proposal.refused
    # control row: seed_q[0] = 0.02 exactly; reaction = k1*t - f1 - g1*z
    np.testing.assert_allclose(proposal.q, [0.02, 0.02], rtol=0., atol=1e-12)
    np.testing.assert_allclose(proposal.reactions, [0.04 - 0.47],
                               rtol=0., atol=1e-12)
    np.testing.assert_allclose(proposal.nodal_forces, [0.94], rtol=0., atol=1e-12)


def test_unsupported_mode_is_excluded_and_supported_mode_is_exact():
    """A zero-support mode gets exactly zero force; the supported one is exact.

    Supported mode 2 has row (0.25, -0.5): H = 0.25/3 = 1/12 m/N and
    g_free = 0.01 - 1/6 < 0, so z* = (1/6 - 0.01)*12 = 1.88 N exactly.
    """
    gap_rows = [[0.5, -1.0], [0.25, -0.5]]
    blocks = _blocks(2.0, 3.0, 0.0, support=(0.0, 1.0), gap_rows=gap_rows)
    blocks["normalized_gap"] = np.asarray([5.0, 0.01])
    proposal = _propose(blocks)
    assert not proposal.refused
    assert proposal.nodal_forces[0] == 0.0
    np.testing.assert_allclose(proposal.nodal_forces[1], 1.88,
                               rtol=0., atol=1e-12)
    assert proposal.diagnostics["supported_pressure_modes"] == 1
    assert proposal.diagnostics["total_pressure_modes"] == 2


def test_no_supported_pressure_modes_refuses():
    blocks = _blocks(2.0, 3.0, 0.01, support=(0.0,))
    refusal = _propose(blocks)
    assert isinstance(refusal, InitialContactSeedRefusal)
    assert refusal.reason == "no_supported_pressure_modes"


def test_singular_constrained_operator_refuses_explicitly():
    """A zero stiffness on the uncontrolled degree makes the operator singular."""
    blocks = _blocks(2.0, 0.0, 0.01)
    refusal = _propose(blocks)
    assert isinstance(refusal, InitialContactSeedRefusal)
    assert refusal.reason == "constrained_factorization_failed"


def test_mode_cap_refuses_explicitly():
    """The explicit supported-mode cap is a resource refusal, not a drop."""
    rows = np.zeros((MAXIMUM_SEED_PRESSURE_MODES + 1, 2))
    rows[:, 1] = -1.0
    blocks = _blocks(2.0, 3.0, 0.0, support=(1.0,)*(MAXIMUM_SEED_PRESSURE_MODES + 1),
                     gap_rows=rows.tolist())
    blocks["normalized_gap"] = np.zeros(MAXIMUM_SEED_PRESSURE_MODES + 1)
    refusal = _propose(blocks)
    assert isinstance(refusal, InitialContactSeedRefusal)
    assert refusal.reason == "pressure_mode_cap_exceeded"
    assert refusal.diagnostics["supported_pressure_modes"] == (
        MAXIMUM_SEED_PRESSURE_MODES + 1)


def test_indefinite_gap_response_refuses_explicitly():
    """A negative structural stiffness gives H < 0 (m/N) and a hard refusal."""
    blocks = _blocks(2.0, -3.0, 0.01)
    refusal = _propose(blocks)
    assert isinstance(refusal, InitialContactSeedRefusal)
    assert refusal.reason == "indefinite_gap_response"
    assert refusal.diagnostics["gap_response_eigenvalue_min"] < 0.0


def test_asymmetric_gap_response_refuses_explicitly():
    """An inconsistent pressure block gives an asymmetric H and a refusal."""
    blocks = _blocks(2.0, 3.0, 0.0, support=(1.0, 1.0),
                     gap_rows=[[0.5, -1.0], [0.25, 0.5]])
    blocks["normalized_gap"] = np.asarray([0.01, 0.01])
    # Scale the second pressure column inconsistently: kqz[:, 1] is twice
    # the gap-gradient transpose, so H[0, 1] = 2*H[1, 0] exactly.
    gap_q = blocks["gap_q"]
    kqz = sparse.csr_matrix(-np.vstack([gap_q[0], 2.0*gap_q[1]]).T)
    blocks["kqz"] = kqz
    refusal = _propose(blocks)
    assert isinstance(refusal, InitialContactSeedRefusal)
    assert refusal.reason == "asymmetric_gap_response"


def test_refusal_and_proposal_mutate_no_caller_blocks():
    """Neither a refusal nor a proposal mutates any caller-owned array."""
    blocks = _blocks(2.0, 3.0, 0.01)
    before = {name: np.asarray(value).copy()
              for name, value in blocks.items()
              if isinstance(value, np.ndarray)}
    proposal = _propose(blocks)
    assert not proposal.refused
    for name, value in before.items():
        np.testing.assert_array_equal(np.asarray(blocks[name]), value)
    singular = _blocks(2.0, 0.0, 0.01)
    before = {name: np.asarray(value).copy()
              for name, value in singular.items()
              if isinstance(value, np.ndarray)}
    refusal = _propose(singular)
    assert refusal.refused
    for name, value in before.items():
        np.testing.assert_array_equal(np.asarray(singular[name]), value)
