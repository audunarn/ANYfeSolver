"""Same-call preparation reuse and instantaneous projected gap pruning."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver.general_contact import s3_patch_static as owner
from anyfesolver.general_contact.q4_overlap_local_bound import overlap_local_lower
from anyfesolver.general_contact.q4_source_overlap import q4_source_projected_gap_lower
from anyfesolver.general_contact.s3_patch_mortar_energy import MovingS3PatchPrimalMortarPenalty
from test_general_contact_jet2_candidate_budget import _scene
from test_general_contact_q4_s3_refined_patch import _two_q4_two_s3
from test_general_contact_q4_s3_jet2_static_owner import _settings


def test_manager_consumes_exact_validated_geometry_once(monkeypatch):
    model, definition, settings = _two_q4_two_s3()
    settings = _settings(settings)
    original = owner.MovingS3PatchPrimalMortarPenalty
    built = []

    def prepare(*args, **kwargs):
        law = original(*args, **kwargs)
        built.append(law)
        return law

    monkeypatch.setattr(owner, 'MovingS3PatchPrimalMortarPenalty', prepare)
    first = owner.GeneralS3PatchManager(model, definition, settings)
    assert len(built) == 1 and first.law.geometry is built[0]
    assert owner.validate_general_s3_patch(model, definition, settings).valid
    assert len(built) == 2  # Public validation remains independent.
    changed = replace(settings, general_s3_difference_step=3e-5)
    second = owner.GeneralS3PatchManager(model, definition, changed)
    assert len(built) == 3 and second.law.geometry is built[-1]
    assert second.law.geometry is not first.law.geometry
    assert second.law.geometry.difference_step == 3e-5
    with pytest.raises(ValueError, match='unsupported'):
        owner.GeneralS3PatchManager(model, definition,
            replace(settings, q4_source_area_tolerance=None))


def gap_fixture():
    sources, targets = _scene(9, q4=True)
    law = MovingS3PatchPrimalMortarPenalty(
        sources, targets, 1000., difference_step=1e-5,
        maximum_energy_evaluations=3000, maximum_candidate_pairs=81,
        q4_source_area_tolerance=1e-9, q4_source_weighted_gap_tolerance=1e-10,
        maximum_q4_source_integration_triangles=128,
        q4_source_derivative_tolerances=(1e-5,)*4,
        differentiation='jet2_q4_s3')
    manager = object.__new__(owner.GeneralS3PatchManager)
    manager.law = law
    manager.augmented = False
    # Explicit current-manager state for this primal-mortar Q4-source/S3-target
    # branch: the profile a validated manager assigns this scene, no projected
    # wrapper around the primal law, and the reference-linear skin selection.
    manager.projected_work = False
    manager.profile = owner.SOURCE_Q4_PROFILE
    manager.finite_skin = None
    manager.settings = SimpleNamespace(q4_source_gap_certificate='overlap_local_dev',
                                       d1_interior_profile=False)
    return manager, law.reference_source.copy(), law.reference_target.copy()


def exhaustive(manager, source, target):
    return min(max(q4_source_projected_gap_lower(source[si], target[ti]),
                   overlap_local_lower(source[si], target[ti], 3)[0])
               for si in manager.law.source_face_indices
               for ti in manager.law.target_face_indices)


@pytest.mark.parametrize('shift', [0., .95, 1.3, 50.])
def test_warped_instantaneous_bound_matches_exhaustive_and_new_candidates(shift):
    manager, source, target = gap_fixture()
    # The physical Q4 normal varies, and target depth is outside source AABBs.
    for nodes in manager.law.source_face_indices:
        source[nodes[2], 2] = .04
    target[:, 2] = -.02
    target[:, 0] += shift
    actual = manager._gap_lower((source, target))
    expected = exhaustive(manager, source, target)
    assert actual == expected
    manager.law.pair_broadphase = False
    assert manager._gap_lower((source, target)) == expected


def test_remote_malformed_target_still_rejected():
    manager, source, target = gap_fixture()
    nodes = manager.law.target_face_indices[-1]
    target[nodes, 0] += 100.
    target[nodes[2]] = target[nodes[0]]
    with pytest.raises(ValueError, match='degenerate'):
        manager._gap_lower((source, target))


def test_singleton_remote_degenerate_target_is_validated_before_pruning():
    manager, source, target = gap_fixture()
    # A reduced target index set exercises the shared singleton validator.
    manager.law.target_face_indices = manager.law.target_face_indices[:1]
    target[:] += (100., 0., 0.)
    nodes = manager.law.target_face_indices[0]
    target[nodes[2]] = target[nodes[0]]
    with pytest.raises(ValueError, match='degenerate'):
        manager._gap_lower((source, target))
