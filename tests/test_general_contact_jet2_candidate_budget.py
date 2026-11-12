"""Conservative candidate admission and counted Jet2 pair work."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty, S3PatchFace,
)


def _scene(count=9, q4=False):
    sources, targets = [], []
    next_node = 1
    for i in range(count):
        corners = np.array(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
        if q4:
            corners = np.array(((0., 0., 0.), (1., 0., 0.),
                                (1., 1., 0.), (0., 1., 0.)))
        corners[:, 0] += 1.3*i
        nodes = tuple(range(next_node, next_node+len(corners)))
        next_node += len(corners)
        sources.append(S3PatchFace(nodes, corners))
    for i in range(count):
        corners = np.array(((.2, .1, 0.), (.6, .1, 0.), (.2, .3, 0.)))
        corners[:, 0] += 1.3*i
        nodes = tuple(range(next_node, next_node+3))
        next_node += 3
        targets.append(S3PatchFace(nodes, corners))
    return tuple(sources), tuple(targets)


def _geometry(*, q4=False, count=9, broadphase=True, budget=27,
              differentiation=None, candidate_cap=None):
    sources, targets = _scene(count, q4)
    options = {}
    if q4:
        options = dict(q4_source_area_tolerance=1e-8,
                       q4_source_weighted_gap_tolerance=1e-10,
                       maximum_q4_source_integration_triangles=128,
                       q4_source_derivative_tolerances=(1e-7,)*4)
    return MovingS3PatchPrimalMortarPenalty(
        sources, targets, 1000., difference_step=2e-6,
        differentiation=differentiation or ("jet2_q4_s3" if q4 else "jet2_s3"),
        maximum_candidate_pairs=count*count if candidate_cap is None else candidate_cap,
        maximum_energy_evaluations=budget, pair_broadphase=broadphase, **options,
    )


@pytest.mark.parametrize(("q4", "count", "budget"),
                         ((False, 9, 27), (True, 20, 883)))
def test_sparse_admission_preserves_exhaustive_float_and_ad_work(q4, count, budget):
    sparse = _geometry(q4=q4, count=count, budget=budget)
    exhaustive = _geometry(q4=q4, count=count, broadphase=False,
                           budget=3*count*count)
    assert count*count > 64
    assert 3*count*count > budget
    assert sparse.required_hessian_evaluations == 3*count
    source, target = sparse.reference_source.copy(), sparse.reference_target.copy()
    source[:, 2] -= .01
    candidates = sparse._candidate_pair_indices(source, target)
    assert len(candidates) == count
    state = sparse.evaluate(source, target)
    reference_state = exhaustive.evaluate(source, target)
    assert sum(local is not None for local in state.pair_states) == count
    assert state.overlapping_pairs == reference_state.overlapping_pairs == count
    for field in ("reference_support", "reference_weighted_gap", "pressure_gram"):
        np.testing.assert_allclose(getattr(state, field), getattr(reference_state, field),
                                   rtol=1e-12, atol=1e-15)
    candidate = MovingS3PatchProjectedNormal(sparse).work(
        source, target, np.zeros(len(sparse.source_node_ids)), tangent=True)
    reference = MovingS3PatchProjectedNormal(exhaustive).work(
        source, target, np.zeros(len(sparse.source_node_ids)), tangent=True)
    np.testing.assert_allclose(candidate.residual, reference.residual, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(candidate.tangent, reference.tangent, rtol=1e-12, atol=1e-12)
    assert candidate.energy_evaluations == 2*count <= budget
    assert reference.energy_evaluations == count*count+count


def test_denser_trial_fails_before_pair_integration_without_pruning(monkeypatch):
    geometry = _geometry()
    source, target = geometry.reference_source.copy(), geometry.reference_target.copy()
    target[:, 0] += .7
    candidates = geometry._candidate_pair_indices(source, target)
    assert len(candidates) > 9
    def forbidden(*args, **kwargs):
        pytest.fail("over-budget pose started float pair integration")
    for law in geometry.pair_laws:
        monkeypatch.setattr(law, "evaluate", forbidden)
    with pytest.raises(MemoryError, match="differentiation budget exhausted"):
        geometry.evaluate(source, target)
    assert geometry._candidate_pair_indices(source, target) == candidates


def test_fallback_float_and_ad_counts_reach_exact_admitted_budget():
    geometry = _geometry()
    normal = MovingS3PatchProjectedNormal(geometry)
    source, target = geometry.reference_source.copy(), geometry.reference_target.copy()
    source[:, 2] -= .01
    base = replace(normal.evaluate(source, target, np.zeros(27)), pair_states=None)
    work = normal._jet2_s3_work(source, target, base, tangent=True)
    assert work.energy_evaluations == geometry.maximum_energy_evaluations == 27


@pytest.mark.parametrize("nodal_force", (False, True))
def test_sparse_shared_edge_four_pass_reserve_counts_actual_candidates(nodal_force):
    geometry = _geometry(q4=True, count=20, budget=883)
    normal = MovingS3PatchProjectedNormal(geometry)
    normal.set_shared_edge_tie(3e-8, (1., 0., 0.), 1e-6)
    source, target = geometry.reference_source.copy(), geometry.reference_target.copy()
    # A resolved nonzero slide gap derivative keeps the existing independent
    # directional gate meaningful at nanometre neighbor offsets.
    source[:, 2] -= .01-1e-4*source[:, 0]
    if nodal_force:
        force = 10.*geometry.evaluate(source, target).reference_support
        work = normal._shared_edge_nodal_force_limit(
            source, target, force, tangent=True, recoverable_pair=None)
    else:
        work = normal._shared_edge_limit(
            source, target, np.zeros(80), tangent=True, recoverable_pair=None)
    # The failed-work/exact-state reserve remains four conservative passes;
    # the four smooth neighbors each perform one float and one AD pass.
    assert work.energy_evaluations == 4*20+4*(2*20) == 240
    if not nodal_force:
        assert work.tie_resolution_relative_error <= 1e-6


@pytest.mark.parametrize("differentiation", ("global", "pair_local"))
def test_finite_difference_worst_case_admission_is_unchanged(differentiation):
    with pytest.raises(MemoryError, match="Hessian exceeds declared"):
        _geometry(differentiation=differentiation,
                  count=2 if differentiation == "global" else 9)


def test_cartesian_candidate_storage_cap_is_unchanged():
    with pytest.raises(MemoryError, match="candidate-pair budget"):
        _geometry(candidate_cap=80)


def test_exhaustive_reference_over_energy_cap_is_rejected():
    with pytest.raises(MemoryError, match="Hessian exceeds declared"):
        _geometry(broadphase=False)
