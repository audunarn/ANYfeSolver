"""The owner must preserve swept rejection, exhaustive fallback and identity."""
import numpy as np
import pytest
from anyfesolver.general_contact import swept_patch_broadphase as search
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_preparation_reuse import gap_fixture


def owner_fixture():
    manager, source, target = gap_fixture()
    manager.settings.d1_interior_profile = False
    manager.settings.maximum_swept_subdivisions = 0
    manager.gap_tolerance = .005
    manager.reference_normals = (np.array((0.,0.,1.)), np.array((0.,0.,1.)))
    manager.swept_trial_cutbacks = 0
    count = len(source)
    manager._positions = lambda values: (np.asarray(values).reshape(-1,3)[:count],
                                         np.asarray(values).reshape(-1,3)[count:])
    # The overridden positions are exact coordinate literals, so their endpoint
    # enclosure is exactly zero; the production certify/bound search stays real.
    manager._position_errors = lambda values: (np.zeros((count,3)),
                                               np.zeros((len(target),3)))
    for index,nodes in enumerate(manager.law.source_face_indices):
        source[nodes,0] += 8.7*index
    for index,nodes in enumerate(manager.law.target_face_indices):
        target[nodes,0] += 8.7*index
    return manager, source, target


@pytest.mark.parametrize('height,expected', [(-.02,True),(.02,False)])
def test_full_path_middle_only_contact_matches_exhaustive_owner(monkeypatch,height,expected):
    manager, source, target = owner_fixture()
    target[:,2] = height
    start=np.vstack((source,target+(-2.,0.,0.))).ravel()
    end=np.vstack((source,target+(2.,0.,0.))).ravel()
    # No instantaneous overlap at either endpoint, but a middle collision
    # must reject when penetration exceeds the declared physical budget.
    assert manager._gap_lower(manager._positions(start))==np.inf
    assert manager._gap_lower(manager._positions(end))==np.inf
    actual=manager.swept_increment_admissible(start,end,0.,1.)
    monkeypatch.setattr(search,'conservative_swept_pair_indices',lambda *a,**k: None)
    assert manager._swept_admissible(start,end)==actual==expected
    assert manager.swept_trial_cutbacks==int(not expected)


def test_search_geometry_exception_is_rejected_not_empty_candidate(monkeypatch):
    manager,source,target=owner_fixture()
    target[:,2]=-.02
    point=np.vstack((source,target)).ravel()
    def invalid(*args,**kwargs):
        raise ValueError('invalid proposed geometry')
    monkeypatch.setattr(search,'conservative_swept_pair_indices',invalid)
    assert not manager.swept_increment_admissible(point,point,0.,1.)
    assert manager.swept_trial_cutbacks==1


def test_prior_swept_revision_checkpoint_rejected_before_continuation():
    import json
    from dataclasses import replace
    from pathlib import Path
    from anyfesolver import solve_static_nonlinear
    from anyfesolver.nonlinear_restart import NonlinearCheckpointError
    from scripts.general_contact_d1_interior_press_dev import build_interior_press
    path=Path(__file__).resolve().parents[1]/'docs/d1_review_support/d1_pressure_family4_moving_edge_cycle_v4_20260929.json'
    checkpoint=json.loads(path.read_text(encoding='utf-8'))['restart_checkpoint']
    model,definition,settings,program=build_interior_press(
        source_columns=4,source_rows=4,target_columns=4,target_rows=4,
        force_n=.04,reference_slope=1e-4,target_young_multiplier=5,
        source_support='perimeter',load_shape='physical_bump',slide_body='target',
        target_reference_full_left_clamp=True,target_left_edge_motion=True,
        four_phase=True,slide_distance_m=.05)
    settings=replace(settings,q4_source_gap_certificate='overlap_local_dev')
    manager=GeneralS3PatchManager(model,definition,settings)
    assert manager.restart_contract()['settings']['d1_swept_certificate_revision']=='affine_union_outward_swept_search_v2'
    with pytest.raises(NonlinearCheckpointError,match='analysis contract is incompatible'):
        solve_static_nonlinear(model,contact=definition,contact_settings=settings,
            contact_path_program=program,restart_checkpoint=checkpoint,max_iterations=18)
