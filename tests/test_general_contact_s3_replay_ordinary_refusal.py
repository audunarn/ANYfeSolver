"""Bounded ordinary-route resource refusal coverage for the S3 replay workflow.

The partial-overlap coupled workflow's ordinary actual-pressure route
intrinsically exceeds the committed maximum_event_stages cap inside
prepare_commit control refinement (linux-29b6-m4/P02: both normal_guide
fixtures refused after 645/657 s). That refusal is preserved exactly; no cap
is raised anywhere. These focused checks exercise the same refusal site and
its atomicity on the same mechanical inputs at a bounded reduced stage cap,
and the explicitly selected prepared partial-pressure route completing the
identical refused operation with unchanged force and work.
"""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver.general_contact.s3_friction_replay import S3ReplayFrictionManager
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_s3_friction_replay import _case
from test_general_contact_s3_friction_static import _mode


def _build(**overrides):
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True,**overrides))
    return owner,model


def _program_poses(model):
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    # The coupled workflow's path program steps as manager-level poses:
    # ((0,0),(0,-.02),(.55,-.02),(.54,-.02),(.54,0),(.54,-.02)).
    return (0*x,-.02*z,.55*x-.02*z,.54*x-.02*z,.54*x,.54*x-.02*z)


def test_ordinary_stage_cap_refusal_in_prepare_commit_control_is_atomic():
    owner,model=_build(prepared_partial_pressure_paths=False,maximum_event_stages=512)
    poses=_program_poses(model)
    owner.commit(owner.prepare_commit(poses[1]))
    owner.commit(owner.prepare_commit(poses[2]))
    # The third pose's base full integration fits the reduced budget; its
    # control refinement does not. This mirrors the coupled fixture's
    # intrinsic refusal inside prepare_commit at the committed 4096 cap.
    owner._integrate(poses[3],full=True)
    assert owner.event_stage_evaluations<owner.friction_settings.maximum_event_stages
    before=owner.checkpoint_state()
    with pytest.raises(MemoryError,match="replay actual-pressure event stage cap exceeded"):
        owner.prepare_commit(poses[3])
    assert owner.event_stage_evaluations==owner.friction_settings.maximum_event_stages
    assert owner.checkpoint_state()==before
    assert len(owner._event_frame_cache)<=owner.friction_settings.maximum_event_stages
    # The refusal is disposable: a fresh public query budget still replays the
    # unchanged committed history exactly as an untouched twin does.
    bary=np.array(((1/3,1/3,1/3),(.2,.3,.5)))
    state=owner.query_material(0,bary)
    twin,_model=_build(prepared_partial_pressure_paths=False)
    twin.commit(twin.prepare_commit(poses[1]))
    twin.commit(twin.prepare_commit(poses[2]))
    reference=twin.query_material(0,bary)
    for key in state:np.testing.assert_array_equal(state[key],reference[key])


def test_explicit_prepared_partial_selection_completes_the_refused_operation():
    # The ordinary twin keeps the committed default cap and supplies the
    # reference physics; the prepared twin works under the same reduced
    # ordinary budget that refuses the ordinary route in the test above.
    ordinary,model=_build(prepared_partial_pressure_paths=False)
    prepared,_model=_build(prepared_partial_pressure_paths=True,maximum_event_stages=512)
    poses=_program_poses(model)
    for owner in (ordinary,prepared):
        owner.commit(owner.prepare_commit(poses[1]))
        owner.commit(owner.prepare_commit(poses[2]))
    reference=ordinary.prepare_commit(poses[3])
    assert ordinary.event_stage_evaluations>0
    completed=prepared.prepare_commit(poses[3])
    # Explicit prepared partial selection never consumes the ordinary
    # actual-pressure budget: every stage frame comes from the prepared path.
    assert prepared.event_stage_evaluations==0
    assert prepared.prepared_partial_pressure_samples>0
    np.testing.assert_array_equal(completed[0],reference[0])
    np.testing.assert_array_equal(completed[1]["local"],reference[1]["local"])
    np.testing.assert_allclose(completed[1]["pressure"],reference[1]["pressure"],
        rtol=1e-11,atol=1e-12)
    for field in ("dissipation_increment_j","storage_j","released_storage_j",
                  "cumulative_dissipation_j","storage_change_j"):
        assert completed[2][field]==pytest.approx(reference[2][field],rel=1e-11,abs=1e-12)
    ordinary.commit(reference);prepared.commit(completed)
    first,second=ordinary.checkpoint_state(),prepared.checkpoint_state()
    assert first["displacement"]==second["displacement"]
    assert first["cumulative_dissipation_j"]==pytest.approx(
        second["cumulative_dissipation_j"],rel=1e-11,abs=1e-12)
    assert first["storage_j"]==pytest.approx(second["storage_j"],rel=1e-11,abs=1e-12)
    for frame_a,frame_b in zip(first["frames"],second["frames"]):
        assert frame_a["local"]==frame_b["local"]
        np.testing.assert_allclose(frame_a["pressure"],frame_b["pressure"],
            rtol=1e-11,atol=1e-12)
