"""Sparse supported reference modes against an independent bounded reduction."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse, linalg
from scipy.sparse.linalg import ArpackNoConvergence

from anyfesolver import stiffened_shell
from anyfesolver.control import CancellationToken
from anyfesolver._stiffened_shell_physical_q4_mass import physical_mass
from anyfesolver._stiffened_shell_b2_q4_panel_experimental import (
    ExperimentalB2Q4PanelAnalysis, ExperimentalB2Q4PanelError,
)
import anyfesolver._stiffened_shell_b2_q4_panel_experimental as owner
from test_stiffened_shell_b2_q4_panel_experimental import _panel, _independent_dense


def _definition():
    nodes, shells, members, points, free = _panel()
    shells = tuple(replace(s, density_kg_m3=2.) for s in shells)
    mass = stiffened_shell.MassMetadata(
        "centered-physical", (0., 0., 0.),
        np.diag((.3, .3, .3, .003, .001, .002)))
    members = tuple(replace(member, mass=mass) for member in members)
    T = sparse.eye(54, format="csr")[:, free]
    return stiffened_shell.B2Q4PanelDefinition(
        stiffened_shell.B2_E4_PL_FAST_LINEAR_V1, nodes, shells, members, points,
        T, np.zeros(54)), free


def _dense_oracle(definition, free):
    k, c = _independent_dense(definition.shell_nodes_m, definition.shells,
                               definition.members, definition.points)
    mass = np.zeros((54, 54))
    shell_index = {node: index for index, node in enumerate(sorted(definition.shell_nodes_m))}
    beam_index = {node: index+6 for index, node in enumerate((10, 11, 12))}
    for shell in definition.shells:
        slots = np.array([6*shell_index[node]+j for node in shell.node_ids for j in range(6)])
        local = physical_mass([definition.shell_nodes_m[node] for node in shell.node_ids],
                              density_kg_m3=shell.density_kg_m3, thickness_m=shell.thickness_m)
        mass[np.ix_(slots, slots)] += local
    for member in definition.members:
        slots = np.array([6*beam_index[node]+j for node in member.node_ids for j in range(6)])
        frame = member.local_dof_map()
        mass[np.ix_(slots, slots)] += frame.T@member.local_candidate_mass_matrix()@frame
    null = linalg.null_space(c[:, free])
    reduced_k = null.T@k[np.ix_(free, free)]@null
    reduced_m = null.T@mass[np.ix_(free, free)]@null
    lam = linalg.eigh(reduced_k, reduced_m, eigvals_only=True)
    return np.sqrt(lam)/(2.*np.pi), k, c, mass


def test_public_sparse_modes_match_bounded_independent_descriptor():
    definition, free = _definition()
    expected, k, c, mass = _dense_oracle(definition, free)
    analysis = stiffened_shell.create_analysis(definition=definition)
    assert "modes" in analysis.capabilities.operations
    result = analysis.modes(stiffened_shell.B2ReferenceModesRequest(count=3))
    assert result.status == "ok", result.failure_reason
    assert not result.production_qualified
    np.testing.assert_allclose(result.frequencies_hz, expected[:3], rtol=2e-8)
    vectors = np.array(result.physical_modes).T
    np.testing.assert_allclose(vectors.T@mass@vectors, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(vectors[np.setdiff1d(np.arange(54), free)], 0., atol=0.)
    for omega, vector, multiplier in zip(result.frequencies_hz, vectors.T,
                                          result.point_multiplier_modes):
        residual = k@vector+c.T@multiplier-(2.*np.pi*omega)**2*mass@vector
        assert np.linalg.norm(residual[free])/(np.linalg.norm(k@vector)+
                                               np.linalg.norm(c.T@multiplier)) <= 1e-8
        assert np.linalg.norm(c@vector) <= 1e-8
    assert result.factorization_count == 1 and result.solve_count > 0
    repeated = analysis.modes(stiffened_shell.B2ReferenceModesRequest(count=3, shift_hz=.2))
    assert repeated.status == "ok", repeated.failure_reason
    np.testing.assert_allclose(repeated.frequencies_hz, result.frequencies_hz, rtol=2e-8)


def test_modes_require_explicit_mass_and_do_not_change_static_route():
    definition, _ = _definition()
    static = replace(definition, shells=tuple(replace(s, density_kg_m3=None)
                                              for s in definition.shells))
    analysis = stiffened_shell.create_analysis(definition=static)
    assert "modes" not in analysis.capabilities.operations
    with pytest.raises(ValueError, match="explicit shell density"):
        analysis.modes(stiffened_shell.B2ReferenceModesRequest())
    assert analysis.solve_linear(np.zeros(54)).status == "completed"
    analysis.close()
    with pytest.raises(ExperimentalB2Q4PanelError, match="closed"):
        analysis.modes(stiffened_shell.B2ReferenceModesRequest())


def test_cancelled_modes_publish_no_partial_pairs_and_preserve_static_factor():
    definition, _ = _definition()
    analysis = stiffened_shell.create_analysis(definition=definition)
    token = CancellationToken(); token.cancel("before modal work")
    result = analysis.modes(stiffened_shell.B2ReferenceModesRequest(), cancellation=token)
    assert result.status == "cancelled" and result.frequencies_hz == ()
    assert result.factorization_count == 0
    assert analysis.solve_linear(np.zeros(54)).status == "completed"


@pytest.mark.parametrize("kwargs", ({"count": True}, {"count": 0},
                                     {"shift_hz": 0.}, {"shift_hz": float("nan")},
                                     {"max_iterations": 0}))
def test_modes_reject_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        stiffened_shell.B2ReferenceModesRequest(**kwargs)


def test_modal_coordinate_rescaling_preserves_physical_pairs():
    definition, _ = _definition()
    reference = stiffened_shell.create_analysis(definition=definition).modes(
        stiffened_shell.B2ReferenceModesRequest())
    scales = np.geomspace(1e-3, 1e3, definition.transformation.shape[1])
    changed = replace(definition, transformation=definition.transformation@sparse.diags(scales))
    result = stiffened_shell.create_analysis(definition=changed).modes(
        stiffened_shell.B2ReferenceModesRequest())
    assert result.status == "ok", result.failure_reason
    np.testing.assert_allclose(result.frequencies_hz, reference.frequencies_hz, rtol=2e-8)


def test_modes_discard_partial_arpack_and_preserve_static_work(monkeypatch):
    definition, _ = _definition()
    analysis = stiffened_shell.create_analysis(definition=definition)
    def failed(*args, **kwargs):
        raise ArpackNoConvergence("controlled partial eigensolve", np.array((1.,)),
                                       np.ones((42, 1)))
    monkeypatch.setattr(owner, "eigs", failed)
    result = analysis.modes(stiffened_shell.B2ReferenceModesRequest())
    assert result.status == "failed" and result.physical_modes == ()
    assert "partial eigensolve" in result.failure_reason
    assert analysis.solve_linear(np.zeros(54)).status == "completed"


def test_cancellation_inside_modal_action_discards_all_pairs(monkeypatch):
    definition, _ = _definition()
    analysis = stiffened_shell.create_analysis(definition=definition)
    token = CancellationToken()
    def interrupt(operator, **kwargs):
        token.cancel("modal iteration")
        operator@np.ones(operator.shape[0])
        raise AssertionError("cancellation must stop the action")
    monkeypatch.setattr(owner, "eigs", interrupt)
    result = analysis.modes(stiffened_shell.B2ReferenceModesRequest(), cancellation=token)
    assert result.status == "cancelled" and result.physical_modes == ()
    assert result.factorization_count == 1
    assert analysis.solve_linear(np.zeros(54)).status == "completed"


def test_modal_programming_errors_propagate(monkeypatch):
    definition, _ = _definition()
    def bug(*args, **kwargs):
        raise TypeError("controlled programming error")
    monkeypatch.setattr(owner, "eigs", bug)
    with pytest.raises(TypeError, match="programming error"):
        stiffened_shell.create_analysis(definition=definition).modes(
            stiffened_shell.B2ReferenceModesRequest())


@pytest.mark.parametrize("shift", (1e-300, 1e300))
def test_unrepresentable_shift_fails_as_configuration(shift):
    definition, _ = _definition()
    with pytest.raises(ValueError, match="shift is not representable"):
        stiffened_shell.create_analysis(definition=definition).modes(
            stiffened_shell.B2ReferenceModesRequest(shift_hz=shift))


def test_retained_zero_drilling_inertia_is_algebraic_not_invented_mass():
    definition, free = _definition()
    _, k, c, mass = _dense_oracle(definition, free)
    # Retain one shell normal-spin coordinate, which has stiffness completion
    # but exactly zero physical inertia. Its equation is algebraic.
    free = np.sort(np.r_[free, 17])
    definition = replace(definition, transformation=sparse.eye(54, format="csr")[:, free])
    np.testing.assert_array_equal(mass[17], 0.)
    assert np.linalg.norm(k[17]) > 0.
    a = np.block([[k[np.ix_(free, free)], c[:, free].T],
                  [c[:, free], np.zeros((12, 12))]])
    b = linalg.block_diag(mass[np.ix_(free, free)], np.zeros((12, 12)))
    lam = linalg.eigvals(a, b)
    assert np.isinf(lam).any()
    finite = np.sort(lam[np.isfinite(lam)].real)
    result = stiffened_shell.create_analysis(definition=definition).modes(
        stiffened_shell.B2ReferenceModesRequest())
    assert result.status == "ok", result.failure_reason
    np.testing.assert_allclose(result.frequencies_hz, np.sqrt(finite[:3])/(2.*np.pi), rtol=2e-8)
    for frequency, vector, multiplier in zip(result.frequencies_hz, result.physical_modes,
                                             result.point_multiplier_modes):
        balance = k@vector+c.T@multiplier-(2.*np.pi*frequency)**2*mass@vector
        assert abs(balance[17]) <= 1e-8*np.linalg.norm(k@vector)


def test_modal_mixed_affine_columns_are_explicitly_unsupported():
    definition, _ = _definition()
    transformation = definition.transformation.tolil()
    row = transformation[:, 0].nonzero()[0][0]
    transformation[row, 1] = .2
    changed = replace(definition, transformation=transformation.tocsr())
    analysis = stiffened_shell.create_analysis(definition=changed)
    with pytest.raises(ValueError, match="without MPCs"):
        analysis.modes(stiffened_shell.B2ReferenceModesRequest())


def test_known_nonfinite_backend_solve_returns_failed_details(monkeypatch):
    definition, _ = _definition()
    analysis = stiffened_shell.create_analysis(definition=definition)
    def bad_solve(rhs):
        raise RuntimeError("Sparse solve produced NaN/Inf values")
    failed = SimpleNamespace(status="ok", factorization_count=1, solve_count=1,
                             solve=bad_solve)
    monkeypatch.setattr(owner, "factorize_cached", lambda *args, **kwargs: failed)
    result = analysis.modes(stiffened_shell.B2ReferenceModesRequest())
    assert result.status == "failed" and result.physical_modes == ()
    assert "NaN/Inf" in result.failure_reason
    assert result.factorization_count == 1 and result.solve_count == 1


def test_shifted_scaling_failure_is_structured(monkeypatch):
    definition, _ = _definition()
    analysis = stiffened_shell.create_analysis(definition=definition)
    def bad_scaling(*args):
        raise ExperimentalB2Q4PanelError("controlled representability failure")
    monkeypatch.setattr(owner, "_scaled_point_kkt", bad_scaling)
    result = analysis.modes(stiffened_shell.B2ReferenceModesRequest())
    assert result.status == "failed" and result.factorization_count == 0
    assert "controlled representability" in result.failure_reason


def test_nonfinite_modal_pairs_are_not_published(monkeypatch):
    definition, _ = _definition()
    analysis = stiffened_shell.create_analysis(definition=definition)
    monkeypatch.setattr(owner, "eigs", lambda operator, **kwargs:
                        (np.full(3, np.nan), np.ones((operator.shape[0], 3))))
    result = analysis.modes(stiffened_shell.B2ReferenceModesRequest())
    assert result.status == "failed" and result.physical_modes == ()
