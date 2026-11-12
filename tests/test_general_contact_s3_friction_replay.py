"""Moving common refinement, material replay and independent partial-force work."""
from dataclasses import replace
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from anyfesolver import BoundaryCondition,solve_static_nonlinear
from anyfesolver.general_contact import ContactPathControl,ContactPathProgram
from anyfesolver.general_contact.s3_sliding_friction import S3SlidingFrictionManager

from anyfesolver.general_contact.s3_friction_replay import S3ReplayFrictionSettings,S3ReplayFrictionManager
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_s3_sliding_friction import _manager
from test_general_contact_s3_friction_static import _mode


def _case(level=1):
    model,definition,normal_settings,material,_unused=_manager()
    material=replace(material,activation_history="linear_constant_normal_touch_dev",maximum_candidate_evaluations=145)
    settings=S3ReplayFrictionSettings(material,level,1e-5,1e-5,1e-6)
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal_settings),settings)
    active=[model.mesh.dof_manager.get_node_dofs(n)[a] for n in range(1,8) for a in (0,1)]
    active += [model.mesh.dof_manager.get_node_dofs(n)[2] for n in (1,2,3)]
    T=sparse.coo_matrix((np.ones(len(active)),(active,np.arange(len(active)))),shape=(42,len(active)))
    owner.restrict_tangent_to_reduced_variations(T)
    return model,definition,normal_settings,settings,owner


@pytest.mark.parametrize('event_subdivisions',[0,4])
def test_separated_candidate_work_preserves_force_work_and_commit(event_subdivisions):
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=event_subdivisions,prepared_pressure_paths=bool(event_subdivisions)))
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    for u in (-.02*z,.1*x-.02*z,.1*x):owner.commit(owner.prepare_commit(u))
    before=owner.checkpoint_state()
    points=np.array(((.2,.3,.5),(.4,.5,.1)))
    physical=owner.query_material(0,points)
    assert np.all(physical['dissipation']>0)
    assert owner._source_strictly_separated(0)
    candidate=.11*x-.02*z
    owner._separated_candidate_work_enabled=False
    baseline=owner._integrate(candidate)
    canonical_commit=owner.prepare_commit(candidate)
    owner._separated_candidate_work_enabled=True
    optimized=owner._integrate(candidate)
    np.testing.assert_array_equal(optimized[0],baseline[0])
    for key in ('dissipation_increment_j','storage_j','released_storage_j'):
        np.testing.assert_allclose(optimized[2][key],baseline[2][key],rtol=1e-12,atol=1e-14)
    assert owner.prepare_commit(candidate)[2]==canonical_commit[2]
    assert owner.checkpoint_state()==before
    for key,value in physical.items():np.testing.assert_array_equal(owner.query_material(0,points)[key],value)
    owner.restore(before,owner.committed_displacement)
    assert owner.checkpoint_state()==before


def test_separated_source_certificate_declines_touching_and_penetration():
    model,_d,_n,_s,owner=_case()
    z=_mode(model,(1,2,3),2)
    opened=owner._frame(np.zeros(42),check_path=False)
    owner.frames=(opened,)
    assert owner._source_strictly_separated(0)
    original=opened['source'][0]
    touching=original.copy()
    touching-=np.outer((touching-opened['origin'])@opened['normal'],opened['normal'])
    # Independent per-face decisions, without broad body-level exclusion.
    owner.frames=(dict(opened,source=(original,touching)),)
    assert owner._source_strictly_separated(0)
    assert not owner._source_strictly_separated(1)
    penetrating=original.copy();penetrating[0]-=.1*opened['normal']
    owner.frames=(dict(opened,source=(penetrating,)),)
    assert not owner._source_strictly_separated(0)


def test_material_history_cache_exact_queries_invalidation_and_bounds():
    model,_d,_n,settings,owner=_case()
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    for u in (-.02*z,-.02*z+.55*x):owner.commit(owner.prepare_commit(u))
    bary=np.array(((.2,.7,.1),(.3,.3,.4),(.8,.1,.1),(.3,.3,.4)))
    expected=owner._replay_material(0,bary)
    owner._material_history_cache.clear()
    for indices in ([1,2],[3,0,2,1],list(range(4))):
        actual=owner.query_material(0,bary[indices])
        for key in actual:
            np.testing.assert_allclose(actual[key],expected[key][indices],rtol=1e-12,atol=1e-14)
    actual['slip'][:]=123
    np.testing.assert_allclose(owner.query_material(0,bary)['slip'],expected['slip'],rtol=1e-12,atol=1e-14)
    owner.friction_settings=replace(settings,maximum_integration_points=2)
    owner._material_history_cache.clear()
    owner.query_material(0,bary)
    assert len(owner._material_history_cache)==2
    owner.friction_settings=settings
    # A new immutable physical history must invalidate every old result.
    owner.commit(owner.prepare_commit(.55*x))
    actual=owner.query_material(0,bary)
    fresh=owner._replay_material(0,bary)
    for key in actual:np.testing.assert_array_equal(actual[key],fresh[key])
    assert np.all(actual['element']==-1)


def test_partial_overlap_force_and_boundary_tangent_match_exact_area():
    model,_d,_n,_s,owner=_case()
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    press=-.02*z
    owner.commit(owner.prepare_commit(press))
    u=press+.55*x
    force,K,_certificate=owner.evaluate(u)
    normal,K_normal,_certificate=owner.normal.evaluate(u)
    # x_source=.2+.4*bary1+.55. Inside x<=1 -> bary1<=.625.
    # P(bary1<=a)=2a-aÂ² and reference area=.08; p=5, mu=.3.
    assert x@(force-normal)==pytest.approx(1.5*.08*(2*.625-.625**2),rel=1e-9)
    assert x@((K-K_normal)@x)==pytest.approx(-.225,rel=1e-6)
    for h in (1e-4,2e-5):
        fd=(owner.evaluate(u+h*x,tangent=False)[0]-owner.evaluate(u-h*x,tangent=False)[0])/(2*h)
        assert np.linalg.norm(K@x-fd)/np.linalg.norm(fd)<=1e-5
    assert owner.friction_settings.integration_refinement==1


def test_never_sampled_material_queries_recover_their_own_history():
    model,_d,_n,_s,owner=_case()
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    owner.commit(owner.prepare_commit(-.02*z))
    owner.commit(owner.prepare_commit(-.02*z+.55*x))
    # Arbitrary material coordinates, independent of every moving quadrature list.
    bary=np.array(((.2,.7,.1),(.3,.3,.4),(.8,.1,.1)))
    state=owner.query_material(0,bary)
    assert state["element"][0]==-1  # x=.2+.4*.7+.55=1.03 outside.
    assert np.all(state["element"][1:]>=0)
    np.testing.assert_allclose(state["storage"][1:],.5*1000*.0015**2,rtol=1e-9)
    np.testing.assert_allclose(state["slip"][1:],.55-.0015,rtol=1e-9)
    assert state["slip"][0]==0  # Declared endpoint discrete law, not exact swept work.
    assert all(not f["local"].flags.writeable and not f["pressure"].flags.writeable for f in owner.frames)


def test_replay_matches_existing_full_coverage_history_at_material_queries():
    model,definition,normal,settings,owner=_case()
    fixed=S3SlidingFrictionManager(model,GeneralS3PatchManager(model,definition,normal),settings.material)
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    bary=np.array([p.barycentric for p in fixed.points])
    for slide,height in ((0,-.02),(.1,-.02),(.08,-.02),(.08,0),(.08,-.02)):
        u=slide*x+height*z
        fixed.commit(fixed.prepare_commit(u));owner.commit(owner.prepare_commit(u))
        state=owner.query_material(0,bary)
        for j,history in enumerate(fixed.history):
            assert state["element"][j]==(-1 if history.target_element is None else history.target_element)
            if history.target_element is not None:
                np.testing.assert_allclose(state["bary"][j],history.barycentric,rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose([state[k][j] for k in ("slip","dissipation","storage")],
                (history.cumulative_plastic_slip_m,history.cumulative_dissipation_Pa_m,history.stored_energy_Pa_m),rtol=1e-9,atol=1e-13)


def test_replay_restore_and_failed_resource_admission_are_atomic():
    model,_d,_n,settings,owner=_case()
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    for u in (-.02*z,-.02*z+.55*x):owner.commit(owner.prepare_commit(u))
    state=owner.checkpoint_state()
    resumed=_case()[-1];resumed.restore(state,owner.committed_displacement)
    assert resumed.checkpoint_state()==state
    arbitrary=np.array(((.2,.7,.1),(.3,.3,.4),(.8,.1,.1)))
    for key,value in owner.query_material(0,arbitrary).items():
        np.testing.assert_array_equal(value,resumed.query_material(0,arbitrary)[key])
    for field in ("storage_j","cumulative_dissipation_j"):
        invalid=copy.deepcopy(state);invalid[field]+=1e-4
        with pytest.raises(ValueError,match="work disagrees"):resumed.restore(invalid,owner.committed_displacement)
        assert resumed.checkpoint_state()==state
    invalid=copy.deepcopy(state);invalid["frames"][0]["pressure"][0]+=1
    with pytest.raises(ValueError,match="pressure disagrees"):resumed.restore(invalid,owner.committed_displacement)
    assert resumed.checkpoint_state()==state
    capped=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,_d,_n),replace(settings,maximum_history_frames=1))
    capped.commit(capped.prepare_commit(-.02*z));before=capped.checkpoint_state()
    with pytest.raises(MemoryError,match="history-frame cap"):capped.prepare_commit(-.02*z+.55*x)
    assert capped.checkpoint_state()==before


def test_hidden_patch_crossing_and_nonfinite_work_reject_without_commit():
    model,definition,normal,settings,owner=_case()
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    first=owner._frame(-.02*z-x);last=owner._frame(-.02*z+x)
    before=owner.checkpoint_state()
    with pytest.raises(ValueError,match="hidden contact interval"):
        owner._step(0,np.array(((1/3,1/3,1/3),)),owner._empty(1),first,last)
    assert owner.checkpoint_state()==before
    material=replace(settings.material,friction_coefficient=1e308)
    huge=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),replace(settings,material=material))
    before=huge.checkpoint_state()
    with pytest.raises(ValueError,match="nonfinite replay Coulomb cap"):huge.prepare_commit(-.02*z)
    assert huge.checkpoint_state()==before
    capped=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),replace(settings,maximum_integration_points=1))
    before=capped.checkpoint_state()
    with pytest.raises(MemoryError,match="integration-point cap"):capped.prepare_commit(-.02*z)
    assert capped.checkpoint_state()==before


@pytest.mark.parametrize("normal_guide",(False,True))
def test_partial_overlap_coupled_workflow_and_split_restart(monkeypatch,tmp_path,normal_guide):
    # The ordinary actual-pressure route intrinsically exceeds the committed
    # maximum_event_stages cap in prepare_commit control refinement on this
    # partial-overlap fixture (linux-29b6-m4/P02: both settings, 645/657 s).
    # That refusal is preserved and covered by bounded focused checks in
    # test_general_contact_s3_replay_ordinary_refusal.py. The positive coupled
    # workflow explicitly selects the existing prepared partial-pressure
    # route; mechanical inputs and scientific gates are unchanged.
    _partial_overlap_coupled_workflow(monkeypatch,tmp_path,normal_guide,prepared_partial=True)


def test_partial_overlap_coupled_full_workflow(monkeypatch,tmp_path):
    _partial_overlap_coupled_workflow(monkeypatch,tmp_path,False,split_restart=False,prepared_partial=True)


def test_partial_overlap_saved_reference_split_restart(monkeypatch,tmp_path):
    reference=Path(__file__).resolve().parents[1]/(
        'docs/d1_review_support/contact_c5_s3_partial_replay_20261002/separation_full_v1_vk_s3r1_checkpoint.json')
    _partial_overlap_coupled_workflow(monkeypatch,tmp_path,False,
        prepared_partial=True,reference_checkpoint=reference)


def _partial_overlap_coupled_workflow(monkeypatch,tmp_path,normal_guide,*,split_restart=True,prepared_partial=False,reference_checkpoint=None):
    import time
    original_intervals=S3ReplayFrictionManager._check_contact_intervals
    def capture_intervals(owner,coefficients,**kwargs):
        try:return original_intervals(owner,coefficients,**kwargs)
        except ValueError:
            np.save(tmp_path/"unresolved_intervals.npy",coefficients)
            raise
    monkeypatch.setattr(S3ReplayFrictionManager,"_check_contact_intervals",capture_intervals)
    original=S3ReplayFrictionManager.evaluate
    def capture(owner,u,**kwargs):
        start=time.perf_counter()
        if kwargs.get("tangent",True):
            (tmp_path/"current_tangent_candidate.json").write_text(json.dumps(dict(
                displacement=np.asarray(u).tolist(),state=owner.checkpoint_state(),
                reduced_directions=owner.reduced_directions.tolist())),encoding="utf-8")
        try:return original(owner,u,**kwargs)
        except (ValueError,MemoryError) as error:
            if "tangent probes are unresolved" in str(error) or isinstance(error,MemoryError):
                (tmp_path/"unresolved_candidate.json").write_text(json.dumps(dict(
                    displacement=np.asarray(u).tolist(),state=owner.checkpoint_state(),
                    reduced_directions=owner.reduced_directions.tolist(),error=str(error),
                    event_pressure_evaluations=owner.event_stage_evaluations)),encoding="utf-8")
            raise
        finally:
            with (tmp_path/"candidate_cost.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(dict(seconds=time.perf_counter()-start,
                    frames=len(owner.frames),tangent=kwargs.get("tangent",True),
                    total_evaluations=owner.evaluation_count,
                    cached_frames=len(owner._event_frame_cache)))+"\n")
    monkeypatch.setattr(S3ReplayFrictionManager,"evaluate",capture)
    model,definition,normal,friction,_unused=_case()
    friction=replace(friction,event_subdivisions=4,prepared_pressure_paths=True,
                     prepared_partial_pressure_paths=prepared_partial)
    model.add_boundary_condition(BoundaryCondition("target_plane",(4,5,6,7),{d:0 for d in ("uz","rx","ry","rz")}))
    model.add_boundary_condition(BoundaryCondition("target_origin",(4,),{"ux":0,"uy":0}))
    model.add_boundary_condition(BoundaryCondition("target_orientation",(5,),{"uy":0}))
    model.add_boundary_condition(BoundaryCondition("source_guide",(1,2,3),{d:0 for d in ("uy","rx","ry","rz")}))
    if normal_guide:
        control=model.mesh.dof_manager.get_node_dofs(1)[2]
        for node in (2,3):
            dependent=model.mesh.dof_manager.get_node_dofs(node)[2]
            model.add_constraint_equation(terms=((dependent,1.),(control,-1.)),
                dependent_dof=dependent,source_id=f"declared_normal_guide_{node}")
    # S3 R1 von Karman default: imposing the whole 0.55 m slide in one step drives the pressed
    # source patch into compressive membrane resultants whose geometric (buckling-type) softening
    # makes the reduced tangent indefinite (lowest eigenvalue -48 vs -0.54 pre-VK), Newton stagnates
    # and the solve stops with a typed contact_gap_exceeded refusal. The same load path imposed in two
    # equal increments converges and its final state is path independent (2/4/8 increments agree to
    # 1.3e-11). Everything else (gates, tolerances, the rest of the path) is unchanged.
    program=ContactPathProgram((ContactPathControl.node(1,"ux"),ContactPathControl.node(1,"uz")),
        ((0,0),(0,-.02),(.275,-.02),(.55,-.02),(.54,-.02),(.54,0),(.54,-.02)),1)
    kwargs=dict(model=model,contact=definition,contact_settings=normal,contact_friction=friction,
        contact_path_program=program,max_iterations=18,tolerance=1e-8,
        emit_restart_checkpoint=True,record_increment_snapshots=True)
    if reference_checkpoint is None:
        full=solve_static_nonlinear(**kwargs)
        assert full.status=="completed",full.info
        reference=full.to_restart_checkpoint()
        (tmp_path/"full_checkpoint.json").write_text(json.dumps(reference),encoding="utf-8")
        history=full.info["force_displacement_history"]
        assert all(row["friction"]["dissipation_increment_j"]>=0 for row in history)
        # step index of the (.54,0) full-separation waypoint: 3 before the slide was split in two
        assert history[4]["friction"]["storage_j"]==0
        moved=full.snapshots[1].displacements
        assert max(abs(moved[model.mesh.dof_manager.get_node_dofs(n)[0]]) for n in (5,6,7))>1e-8
    else:
        reference=json.loads(reference_checkpoint.read_text(encoding='utf-8'))
        assert reference['path_state']['accepted_step']==program.total_steps
        assert reference['path_state']['terminal_status']=='completed'
        history=reference['path_state']['history']
    if not split_restart:return
    # .5 of the 6 steps = the end of the slide (formerly .4 of 5 steps): the same physical split point
    half=solve_static_nonlinear(max_load_factor=.5,**kwargs)
    assert half.status=='completed',half.info
    for key in ('model_fingerprint','analysis_fingerprint'):
        assert half.to_restart_checkpoint()[key]==reference[key]
    (tmp_path/"split_checkpoint.json").write_text(json.dumps(half.to_restart_checkpoint()),encoding="utf-8")
    resumed=solve_static_nonlinear(restart_checkpoint=half.to_restart_checkpoint(),**kwargs)
    assert resumed.status=="completed",resumed.info
    (tmp_path/"resumed_checkpoint.json").write_text(json.dumps(resumed.to_restart_checkpoint()),encoding="utf-8")
    np.testing.assert_allclose(resumed.displacements,reference['displacements'],rtol=1e-7,atol=1e-9)
    observed_history=resumed.info["force_displacement_history"]
    if reference_checkpoint is not None:
        _assert_legacy_reference_history(observed_history,history)
    else:
        assert observed_history==history



def _assert_legacy_reference_history(observed,reference):
    """Only the newly explicit world-origin label differs from this old record."""
    compared=copy.deepcopy(observed)
    assert len(compared)==len(reference)
    for actual,legacy in zip(compared,reference):
        if "wrench_origin_m" not in legacy["contact"]:
            assert actual["contact"].pop("wrench_origin_m")==[0.,0.,0.]
    assert compared==reference


def test_legacy_reference_metadata_preserves_every_numerical_and_other_field():
    legacy=[dict(contact=dict(force=1.),step_index=1)]
    actual=[dict(contact=dict(force=1.,wrench_origin_m=[0.,0.,0.]),step_index=1)]
    _assert_legacy_reference_history(actual,legacy)
    assert actual[0]["contact"]["wrench_origin_m"]==[0.,0.,0.]
    for changed in (dict(force=2.,wrench_origin_m=[0.,0.,0.]),
                    dict(force=1.,wrench_origin_m=[1.,0.,0.]),
                    dict(force=1.,wrench_origin_m=[0.,0.,0.],unrelated=0)):
        with pytest.raises(AssertionError):
            _assert_legacy_reference_history([dict(contact=changed,step_index=1)],legacy)

def test_opening_deformation_can_create_a_real_hidden_contact_interval():
    from anyfesolver.general_contact.s3_friction_replay import ReplayContactSubdivisionRequired
    # Captured from the unmodified coupled workflow. The material point enters
    # the moving patch before it opens in the normal direction; both endpoints
    # are inactive. This must require subdivision, not a roundoff exemption.
    _,_,_,_,owner=_case()
    coefficients=np.array((
        (3.1398972193995324e-3,-1.8139897219399535e-2,0.),
        (.30606059181495615,6.424186129431234e-6,0.),
        (.9999939416153832,-2.1273995851292059e-5,5.260730195254534e-11),
        (-5.0834557451544645e-6,3.2415873944142417e-5,-9.033831387417194e-11),
        (.6939325441561803,4.3992482944498704e-7,-8.209544005078355e-11),
    ))
    def inside(t):
        value=coefficients@np.array((1.,t,t*t))
        return np.all(value>=0) and value[0]>0
    assert not inside(0.) and inside(.16) and not inside(1.)
    before=owner.checkpoint_state()
    with pytest.raises(ReplayContactSubdivisionRequired,match="hidden contact interval") as caught:
        owner._check_contact_intervals(coefficients[:,:,None])
    event=caught.value
    assert len(event.intervals)==1
    point,entry,exit=event.intervals[0]
    assert point==0
    # Independent scalar roots of the captured boundary/gap equations.
    expected_entry=min(t.real for t in np.roots(coefficients[3,::-1])
                       if abs(t.imag)<1e-14 and 0<t.real<1)
    expected_exit=-coefficients[0,0]/coefficients[0,1]
    assert entry==pytest.approx(expected_entry,rel=1e-13)
    assert exit==pytest.approx(expected_exit,rel=1e-13)
    assert entry<event.suggested_fraction<exit
    assert inside(event.suggested_fraction)
    assert owner.checkpoint_state()==before


def test_event_replay_retains_hidden_pulse_work_and_refines_without_query_remapping():
    model,definition,normal,settings,_unused=_case()
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    bary=np.array(((1/3,1/3,1/3),(.2,.2,.6)))
    errors=[]
    for subdivisions in (4,16,64):
        owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
            replace(settings,event_subdivisions=subdivisions))
        first=owner._frame(-.02*z-x);last=owner._frame(-.02*z+x)
        before=owner.checkpoint_state()
        response=owner._step(0,bary,owner._empty(2),first,last)
        state=response[0]
        assert np.all(state["element"]==-1)
        assert np.all(state["storage"]==0)
        assert np.all(state["released_storage"]>0)
        assert np.all(response[2]>0)
        # Each point traverses the full one-metre target at p=5Pa. The
        # continuous slider yields cap*(distance-cap/kt), independently of
        # the production contact/event implementation.
        exact=1.5*(1.-1.5/1000.)
        errors.append(float(np.max(abs(response[2]-exact))/exact))
        if subdivisions==16:
            single=owner._step(0,bary[:1],owner._empty(1),first,last)
            for key in state:np.testing.assert_array_equal(state[key][:1],single[0][key])
            assert owner.restart_contract()["history"].endswith("_v2")
        assert owner.checkpoint_state()==before
    assert errors[2]<errors[1]<errors[0]
    assert errors[-1]<.01
    limited=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,maximum_event_stages=1))
    before=limited.checkpoint_state()
    with pytest.raises(MemoryError,match="actual-pressure event stage cap"):
        limited._step(0,bary,limited._empty(2),first,last)
    assert limited.checkpoint_state()==before


def test_event_queries_have_fresh_budgets_and_point_local_boundary_roundoff():
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,maximum_event_stages=6))
    z=_mode(model,(1,2,3),2)
    frame=owner._frame(-.02*z)
    # Disposable retained frame fixture isolates the public query's budget;
    # it does not claim a physically committed/integrated work ledger.
    owner.frames=(frame,)
    bary=np.array(((1/3,1/3,1/3),))
    first=owner.query_material(0,bary)
    owner.event_stage_evaluations=owner.friction_settings.maximum_event_stages
    second=owner.query_material(0,bary)
    for key in first:np.testing.assert_array_equal(first[key],second[key])
    with pytest.raises(ValueError,match="no normal multipliers"):
        owner.normal.set_normal_multipliers(np.ones(3))
    start=np.array(((-1e-9,.3,.01),));end=start+np.array((.001,0.,0.))
    alone=owner._entry(start,end,frame,frame)
    # A distant co-query must not enlarge the near-edge point's roundoff band.
    far=np.array(((1e6,.3,.01),))
    together=owner._entry(np.vstack((start,far)),np.vstack((end,far)),frame,frame)
    np.testing.assert_array_equal(alone,together[:1])
    assert alone[0]>0


def test_physical_commit_summary_does_not_depend_on_disposable_cache_warmth():
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4))
    pose=-.02*_mode(model,(1,2,3),2)
    before=owner.checkpoint_state()
    first=owner.prepare_commit(pose);second=owner.prepare_commit(pose)
    assert first[2]==second[2]
    assert "event_pressure_evaluations" not in first[2]
    assert "event_frame_cache_hits" not in first[2]
    assert owner.event_frame_cache_hits>0
    assert owner.checkpoint_state()==before


@pytest.mark.parametrize("recorded_press",(False,True))
@pytest.mark.parametrize("prepared_paths",(False,True))
def test_event_replay_evaluation_cost_discriminator(tmp_path,recorded_press,prepared_paths):
    import cProfile,pstats,time
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=prepared_paths))
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    if recorded_press:
        path=Path(__file__).resolve().parents[1]/"docs/d1_review_support/contact_c5_s3_partial_replay_20261002/unresolved_opening_candidate.json"
        saved=json.loads(path.read_text(encoding="utf-8"))
        pose=owner._full(np.array(saved["state"]["frames"][0]["local"]))
    else:
        owner.commit(owner.prepare_commit(-.02*z))
        pose=-.02*z+.55*x
    profile=cProfile.Profile();start=time.perf_counter()
    profile.enable();force,_frame,summary=owner._integrate(pose);profile.disable()
    elapsed=time.perf_counter()-start
    profile.dump_stats(str(tmp_path/"event_evaluation.prof"))
    with (tmp_path/"event_profile.txt").open("w",encoding="utf-8") as stream:
        pstats.Stats(profile,stream=stream).strip_dirs().sort_stats("cumulative").print_stats(22)
    (tmp_path/"event_response.json").write_text(json.dumps(dict(
        elapsed_seconds=elapsed,force=force.tolist(),summary=summary),indent=2),encoding="utf-8")
    warmed,_frame,warm_summary=owner._integrate(pose)
    np.testing.assert_array_equal(warmed,force)
    for field in ("dissipation_increment_j","storage_j","released_storage_j"):
        assert warm_summary[field]==summary[field]
    assert warm_summary["event_pressure_evaluations"]==0
    assert warm_summary["event_frame_cache_hits"]>0
    # A changed pose must miss its own stage entries; historical frames may hit.
    _force,_frame,changed=owner._integrate(pose+1e-4*x)
    assert changed["event_pressure_evaluations"]+changed["prepared_pressure_samples"]>0
    assert len(owner._event_frame_cache)<=settings.maximum_event_stages
    assert np.all(np.isfinite(force))
    for axis in range(3):assert abs(force@_mode(model,range(1,8),axis))<1e-10


def test_full_coverage_stage_pressure_matches_unchanged_mortar_and_rejects_partial():
    model,_definition,_normal,_settings,owner=_case()
    rng=np.random.default_rng(20261002)
    for _ in range(24):
        u=np.zeros(42)
        for node in (1,2,3):
            dofs=model.mesh.dof_manager.get_node_dofs(node)
            u[dofs[:2]]=rng.uniform(-.03,.03,2)
            u[dofs[2]]=rng.uniform(-.025,-.01)
        for node in (4,5,6,7):
            dofs=model.mesh.dof_manager.get_node_dofs(node)
            u[dofs[:2]]=rng.uniform(-.02,.02,2)
        corners,n,_basis,origin=owner.template._patch(u)
        source,target=owner.normal._positions(u)
        actual=owner._full_coverage_pressure(source,target,corners,n,origin)
        expected=owner.normal.law.evaluate(source,target,owner.normal.normal_multipliers).pressure
        assert actual is not None
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    u=-.02*z+.55*x
    corners,n,_basis,origin=owner.template._patch(u)
    source,target=owner.normal._positions(u)
    assert owner._full_coverage_pressure(source,target,corners,n,origin) is None


def test_partial_planar_stage_moments_match_ordinary_mortar():
    model,_definition,_normal,_settings,owner=_case()
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    rng=np.random.default_rng(20261003)
    for shift in np.linspace(.35,.85,17):
        pose=-.02*z+shift*x
        for node in (1,2,3):
            dofs=model.mesh.dof_manager.get_node_dofs(node)
            pose[dofs[2]]+=rng.uniform(-.004,.002)
        for node in (4,5,6,7):
            dofs=model.mesh.dof_manager.get_node_dofs(node)
            pose[dofs[:2]]=rng.uniform(-.008,.008,2)
        corners,n,_basis,origin=owner.template._patch(pose)
        source,target=owner.normal._positions(pose)
        actual=owner._partial_planar_pressure(source,target,corners,n,origin)
        expected=owner.normal.law.evaluate(source,target,owner.normal.normal_multipliers).pressure
        assert actual is not None
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-11)
    # Constant traction is independently known, including a small positive
    # overlap at the physical rim. No support is deleted by an area threshold.
    for width in (1e-2,1e-4,1e-6,1e-10,1e-13,1e-15,0.):
        pose=-.02*z+(.8-width)*x
        corners,n,_basis,origin=owner.template._patch(pose)
        source,target=owner.normal._positions(pose)
        actual=owner._partial_planar_pressure(source,target,corners,n,origin)
        if width>=1e-6:np.testing.assert_allclose(actual,5.,rtol=1e-11,atol=1e-11)
        expected=owner.normal.law.evaluate(source,target,owner.normal.normal_multipliers).pressure
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-11)
    warped=target.copy();warped[-1,2]+=1e-5
    assert owner._partial_planar_pressure(source,warped,corners,n,origin) is None


def test_partial_stage_moments_preserve_slightly_concave_triangle_union():
    from types import SimpleNamespace
    from anyfesolver.general_contact.s3_patch_mortar_energy import S3PatchFace,MovingS3PatchPrimalMortarPenalty
    source=np.array(((.4,.4,-.01),(.7,.4,-.009),(.4,.7,-.011)))
    target=np.array(((0.,0.,0.),(1.,0.,0.),(.5,.5-2e-11,0.),(0.,1.,0.)))
    geometry=MovingS3PatchPrimalMortarPenalty(
        (S3PatchFace((1,2,3),source),),
        (S3PatchFace((4,5,6),target[[0,1,2]]),S3PatchFace((4,6,7),target[[0,2,3]])),
        1000.,difference_step=1e-6)
    fixture=SimpleNamespace(normal=SimpleNamespace(law=SimpleNamespace(
        geometry=geometry,pressure_basis=None,stiffness=1000.),normal_multipliers=np.zeros(3)))
    actual=S3ReplayFrictionManager._partial_planar_pressure(fixture,source,target,
        {1:target[[0,1,2]],2:target[[0,2,3]]},np.array((0.,0.,1.)),target[0])
    expected=geometry.evaluate(source,target).pressure
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_partial_stage_moments_preserve_shape_domain_rejection():
    from types import SimpleNamespace
    from anyfesolver.general_contact.s3_patch_mortar_energy import S3PatchFace,MovingS3PatchPrimalMortarPenalty
    h=1e-6
    source=np.array(((-1e-8,h/4,-.01),(h/4,h/4,-.01),(0.,1.,-.01)))
    target=np.array(((0.,0.,0.),(h,0.,0.),(0.,h,0.)))
    remote=target+np.array((2*h,0.,0.))
    geometry=MovingS3PatchPrimalMortarPenalty((S3PatchFace((1,2,3),source),),
        (S3PatchFace((4,5,6),target),S3PatchFace((7,8,9),remote)),1000.,difference_step=1e-6)
    fixture=SimpleNamespace(normal=SimpleNamespace(law=SimpleNamespace(
        geometry=geometry,pressure_basis=None,stiffness=1000.),normal_multipliers=np.zeros(3)))
    with pytest.raises(ValueError,match="outside"):
        geometry.evaluate(source,np.vstack((target,remote)))
    with pytest.raises(ValueError,match="outside"):
        S3ReplayFrictionManager._partial_planar_pressure(fixture,source,np.vstack((target,remote)),
            {1:target,2:remote},np.array((0.,0.,1.)),target[0])


def test_full_coverage_pressure_preserves_shared_node_reference_moments():
    from types import SimpleNamespace
    from anyfesolver.general_contact.s3_patch_mortar_energy import S3PatchFace,MovingS3PatchPrimalMortarPenalty
    reference=np.array(((.2,.2,.02),(.8,.2,.02),(.6,.6,.02),(.2,.7,.02)))
    target=np.array(((-1.,-1.,0.),(3.,-1.,0.),(-1.,3.,0.)))
    geometry=MovingS3PatchPrimalMortarPenalty(
        (S3PatchFace((1,2,3),reference[[0,1,2]]),S3PatchFace((1,3,4),reference[[0,2,3]])),
        (S3PatchFace((5,6,7),target),),1000.,difference_step=1e-6)
    fixture=SimpleNamespace(normal=SimpleNamespace(law=SimpleNamespace(
        geometry=geometry,pressure_basis=None,stiffness=1000.),normal_multipliers=np.zeros(4)),
        template=SimpleNamespace(target_adapters={1:SimpleNamespace(node_ids=(5,6,7))},
            boundary_edges=frozenset(((5,6),(6,7),(5,7)))))
    source=reference.copy();source[:,2]=(-.002,-.009,-.004,-.007)
    actual=S3ReplayFrictionManager._full_coverage_pressure(fixture,source,target,
        {1:target},np.array((0.,0.,1.)),target[0])
    expected=geometry.evaluate(source,target).pressure
    assert actual is not None
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_prepared_pressure_path_matches_actual_mortar_and_certifies_between_endpoints():
    from anyfesolver.general_contact.s3_friction_prepared_path import prepare_s3_plane_pressure_path,_positive
    from fractions import Fraction
    model,_definition,_normal,_settings,owner=_case()
    path=Path(__file__).resolve().parents[1]/"docs/d1_review_support/contact_c5_s3_partial_replay_20261002/unresolved_opening_candidate.json"
    saved=json.loads(path.read_text(encoding="utf-8"))
    pose=owner._full(np.array(saved["state"]["frames"][0]["local"]))
    # Also test a deliberately tilted, nonuniform source gap.
    tilted=pose.copy()
    for node,value in zip((1,2,3),(-.023,-.018,-.027)):
        tilted[model.mesh.dof_manager.get_node_dofs(node)[2]]=value
    for endpoint in (pose,tilted):
        last=owner._frame(endpoint)
        prepared=prepare_s3_plane_pressure_path(owner,owner.initial_frame,last)
        assert prepared is not None
        fractions=np.r_[np.linspace(0,1,25),.013712,.413771,.913889]
        actual=prepared.pressure(fractions)
        expected=[]
        for fraction in fractions:
            source,target=owner.normal._positions(fraction*endpoint)
            expected.append(owner.normal.law.evaluate(source,target,owner.normal.normal_multipliers).pressure)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(prepared.pressure(fractions[:1]),actual[:1])
    assert not _positive([Fraction(1),Fraction(-4),Fraction(4)])
    # Endpoints are the same valid square, but an affine half-turn collapses it
    # internally. Endpoint checks or sampled eligibility may not admit it.
    half_turn=np.zeros(42)
    for node,xy in zip((4,5,6,7),((0,0),(1,0),(0,1),(1,1))):
        half_turn[model.mesh.dof_manager.get_node_dofs(node)[:2]]=1-2*np.asarray(xy)
    last=owner._frame(half_turn,check_path=False)
    assert prepare_s3_plane_pressure_path(owner,owner.initial_frame,last) is None
    partial=-.02*_mode(model,(1,2,3),2)+.55*_mode(model,(1,2,3),0)
    assert prepare_s3_plane_pressure_path(owner,owner.initial_frame,owner._frame(partial)) is None


def test_prepared_event_batches_preserve_material_history_force_and_work():
    model,definition,normal,settings,_unused=_case()
    ordinary=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4))
    prepared=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True,maximum_prepared_stage_batch=2))
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    pose=-.02*z+.08*x
    pose[model.mesh.dof_manager.get_node_dofs(2)[2]]=-.025
    pose[model.mesh.dof_manager.get_node_dofs(3)[2]]=-.018
    bary=np.array(((.2,.3,.5),(.1,.8,.1),(.8,.1,.1)))
    before=prepared.checkpoint_state()
    a=ordinary.query_material(0,bary,candidate=ordinary._frame(pose))
    b=prepared.query_material(0,bary,candidate=prepared._frame(pose))
    for key in a[0]:np.testing.assert_allclose(a[0][key],b[0][key],rtol=1e-11,atol=1e-12)
    for expected,actual in zip(a[1:],b[1:]):
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-12)
    assert prepared.event_stage_evaluations==0 and prepared.prepared_pressure_samples>0
    assert ordinary.event_stage_evaluations>0
    assert prepared.checkpoint_state()==before
    for frame in prepared._event_frame_cache.values():
        assert not frame["pressure"].flags.writeable and not frame["local"].flags.writeable
    # Complete candidate force/work integration includes the same material law.
    force_a,_frame,work_a=ordinary._integrate(pose)
    force_b,_frame,work_b=prepared._integrate(pose)
    np.testing.assert_allclose(force_b,force_a,rtol=1e-11,atol=1e-12)
    for field in ("storage_j","dissipation_increment_j","released_storage_j"):
        assert work_b[field]==pytest.approx(work_a[field],rel=1e-11,abs=1e-12)
    # Prepared batches have their own bounded work; they do not consume the
    # ordinary geometry-construction budget or grow its LRU beyond the cap.
    limited=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True,
                maximum_prepared_stage_batch=2,maximum_event_stages=1))
    c=limited.query_material(0,bary,candidate=limited._frame(pose))
    for key in b[0]:np.testing.assert_array_equal(c[0][key],b[0][key])
    assert len(limited._event_frame_cache)<=1
    assert limited.event_stage_evaluations==0
    partial=-.02*z+.55*x
    before=limited.checkpoint_state()
    with pytest.raises(MemoryError,match="actual-pressure event stage cap"):
        limited.query_material(0,bary,candidate=limited._frame(partial))
    assert limited.checkpoint_state()==before


def test_ordinary_stage_batch_reuses_snapshot_after_lru_eviction():
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,maximum_event_stages=1))
    first=owner.initial_frame
    last=owner._frame(-.02*_mode(model,(1,2,3),2))
    key=(first["local"].tobytes(),last["local"].tobytes())
    retained=list(owner._stage_frames(first,last,(.75,),key,None))[0]
    owner.event_stage_evaluations=0
    owner.event_frame_cache_hits=0
    before=owner.checkpoint_state()
    frames=list(owner._stage_frames(first,last,(.25,.75),key,None))
    assert frames[1] is retained
    assert owner.event_stage_evaluations==1 and owner.event_frame_cache_hits==1
    assert len(owner._event_frame_cache)==1
    assert owner.checkpoint_state()==before


def test_prepared_replay_handles_many_material_times_in_bounded_batches(tmp_path):
    import time
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True,maximum_prepared_stage_batch=16))
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    pose=-.02*z+.08*x
    pose[model.mesh.dof_manager.get_node_dofs(2)[2]]=-.025
    pose[model.mesh.dof_manager.get_node_dofs(3)[2]]=-.018
    bary=np.random.default_rng(20261002).dirichlet(np.ones(3),1296)
    before=owner.checkpoint_state();start=time.perf_counter()
    result=owner.query_material(0,bary,candidate=owner._frame(pose))
    elapsed=time.perf_counter()-start
    samples=owner.prepared_pressure_samples
    (tmp_path/"prepared_batch_cost.json").write_text(json.dumps(dict(
        query_count=len(bary),elapsed_seconds=elapsed,prepared_samples=samples,
        ordinary_evaluations=owner.event_stage_evaluations,resident_cache_frames=len(owner._event_frame_cache),
        maximum_batch=owner.friction_settings.maximum_prepared_stage_batch),indent=2),encoding="utf-8")
    assert samples>4096 and owner.event_stage_evaluations==0
    assert len(owner._event_frame_cache)<=settings.maximum_event_stages
    assert all(np.all(np.isfinite(value)) for value in result[0].values())
    assert np.all(result[0]["dissipation"]>=0)
    single=owner.query_material(0,bary[701:702],candidate=owner._frame(pose))
    for key in result[0]:np.testing.assert_array_equal(single[0][key],result[0][key][701:702])
    assert owner.checkpoint_state()==before


def test_prepared_partial_path_matches_actual_pressure_and_work_maps():
    from anyfesolver.general_contact.s3_friction_prepared_path import prepare_s3_plane_pressure_path
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True,prepared_partial_pressure_paths=True))
    saved=json.loads((Path(__file__).resolve().parents[1]/
        "docs/d1_review_support/contact_c5_s3_partial_replay_20261002/stage_moments_full_v1_last_tangent_candidate.json").read_text())
    first=owner._frame(owner._full(np.array(saved["state"]["frames"][-1]["local"])),check_path=False)
    last=owner._frame(np.asarray(saved["displacement"]),check_path=False)
    assert prepare_s3_plane_pressure_path(owner,first,last) is None
    path=prepare_s3_plane_pressure_path(owner,first,last,allow_partial=True)
    assert path is not None and not path.full_coverage
    times=np.linspace(0.,1.,25)
    for t,frame in zip(times,path.frames(times)):
        pose=owner._full(first["local"]+t*(last["local"]-first["local"]))
        ordinary=owner._frame(pose,check_path=False)
        source,target=owner.normal._positions(pose)
        pressure=owner.normal.law.evaluate(source,target,owner.normal.normal_multipliers).pressure
        np.testing.assert_allclose(frame["pressure"],pressure,rtol=1e-11,atol=1e-11)
        for key in ("local","source","normal","basis","origin"):
            np.testing.assert_allclose(frame[key],ordinary[key],rtol=1e-13,atol=1e-14)
        for e in frame["target"]:
            np.testing.assert_allclose(frame["target"][e],ordinary["target"][e],rtol=1e-13,atol=1e-14)


def test_prepared_partial_replay_preserves_history_and_explicit_budgets():
    model,definition,normal,settings,_unused=_case()
    ordinary=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True))
    settings=replace(settings,event_subdivisions=4,prepared_pressure_paths=True,prepared_partial_pressure_paths=True)
    prepared=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),settings)
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    bary=np.array(((.2,.3,.5),(.1,.8,.1),(.8,.1,.1)))
    for shift,height in ((0,-.02),(.55,-.02),(.54,0),(.54,-.02)):
        pose=shift*x+height*z
        a_frame=ordinary._frame(pose);b_frame=prepared._frame(pose)
        a=ordinary.query_material(0,bary,candidate=a_frame)
        b=prepared.query_material(0,bary,candidate=b_frame)
        for key in a[0]:np.testing.assert_allclose(a[0][key],b[0][key],rtol=1e-11,atol=1e-12)
        for expected,actual in zip(a[1:],b[1:]):np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-12)
        ordinary.frames=(*ordinary.frames,a_frame);prepared.frames=(*prepared.frames,b_frame)
    assert prepared.prepared_partial_pressure_samples>0 and prepared.event_stage_evaluations==0
    limited=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,maximum_prepared_partial_pressure_samples=1))
    before=limited.checkpoint_state()
    with pytest.raises(MemoryError,match="partial-pressure sample cap"):
        limited.query_material(0,bary,candidate=limited._frame(-.02*z+.55*x))
    assert limited.checkpoint_state()==before
    assert prepared.restart_contract()!=ordinary.restart_contract()


def test_prepared_partial_stage_decline_uses_counted_ordinary_fallback(monkeypatch):
    from anyfesolver.general_contact.s3_friction_prepared_path import prepare_s3_plane_pressure_path
    model,definition,normal,settings,_unused=_case()
    settings=replace(settings,event_subdivisions=4,prepared_pressure_paths=True,
        prepared_partial_pressure_paths=True,maximum_event_stages=1)
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),settings)
    first=owner.initial_frame
    pose=-.02*_mode(model,(1,2,3),2)+.55*_mode(model,(1,2,3),0)
    last=owner._frame(pose)
    prepared=prepare_s3_plane_pressure_path(owner,first,last,allow_partial=True)
    assert not prepared.full_coverage
    # Exercise the documented decline contract independently of which floating
    # interpolation first loses bitwise planarity on a particular platform.
    monkeypatch.setattr(prepared,"_uniform_partial_pressure",lambda *args:None)
    monkeypatch.setattr(owner,"_partial_planar_pressure",lambda *args:None)
    key=(first["local"].tobytes(),last["local"].tobytes())
    before=owner.checkpoint_state()
    actual=list(owner._stage_frames(first,last,(.75,),key,prepared))[0]
    expected=owner._frame(.75*pose,check_path=False)
    np.testing.assert_array_equal(actual["pressure"],expected["pressure"])
    assert owner.event_stage_evaluations==1 and owner.prepared_partial_pressure_samples==1
    assert owner.checkpoint_state()==before
    with pytest.raises(MemoryError,match="actual-pressure event stage cap"):
        list(owner._stage_frames(first,last,(.5,),key,prepared))
    assert owner.checkpoint_state()==before


def test_saved_reversal_commit_control_with_prepared_partial_paths(tmp_path):
    import time
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True,prepared_partial_pressure_paths=True))
    saved=json.loads((Path(__file__).resolve().parents[1]/
        "docs/d1_review_support/contact_c5_s3_partial_replay_20261002/stage_moments_full_v1_last_tangent_candidate.json").read_text())
    start=time.perf_counter()
    owner.restore(saved["state"],np.asarray(saved["state"]["displacement"]))
    before=owner.checkpoint_state()
    try:
        result=owner.prepare_commit(np.asarray(saved["displacement"]))
    finally:
        (tmp_path/"commit_control_cost.json").write_text(json.dumps(dict(
            elapsed_seconds=time.perf_counter()-start,ordinary_stages=owner.event_stage_evaluations,
            prepared_samples=owner.prepared_pressure_samples,
            partial_samples=owner.prepared_partial_pressure_samples),indent=2),encoding="utf-8")
    assert owner.checkpoint_state()==before
    assert result[2]["history_frames"]==3
    assert owner.prepared_partial_pressure_samples>4096
    assert owner.event_stage_evaluations==0
    (tmp_path/"commit_control_work.json").write_text(json.dumps(result[2],indent=2),encoding="utf-8")


def test_prepared_partial_replay_directional_tangent_on_retained_reversal():
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True,prepared_partial_pressure_paths=True))
    saved=json.loads((Path(__file__).resolve().parents[1]/
        "docs/d1_review_support/contact_c5_s3_partial_replay_20261002/stage_moments_full_v1_last_tangent_candidate.json").read_text())
    owner.restore(saved["state"],np.asarray(saved["state"]["displacement"]))
    source_x=_mode(model,(1,2,3),0);target_x=_mode(model,(4,5,6,7),0)
    owner.restrict_tangent_to_reduced_variations(sparse.csr_matrix(np.column_stack((source_x,target_x))))
    pose=np.asarray(saved["displacement"])
    before=owner.checkpoint_state()
    _force,tangent,_certificate=owner.evaluate(pose)
    direction=.7*source_x-.2*target_x
    for h in (2e-6,1e-6):
        plus=owner.evaluate(pose+h*direction,tangent=False)[0]
        minus=owner.evaluate(pose-h*direction,tangent=False)[0]
        fd=(plus-minus)/(2*h)
        assert np.linalg.norm(tangent@direction-fd)/max(np.linalg.norm(fd),1e-30)<=1e-5
    assert owner.checkpoint_state()==before


def test_prepared_event_route_directional_tangent_includes_both_bodies():
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True))
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    target_x=_mode(model,(4,5,6,7),0)
    owner.restrict_tangent_to_reduced_variations(sparse.csc_matrix(np.column_stack((x,z,target_x))))
    pose=-.02*z+.01*x
    _force,tangent,_certificate=owner.evaluate(pose)
    direction=x+.1*z-.3*target_x
    for h in (2e-6,1e-6):
        plus=owner.evaluate(pose+h*direction,tangent=False)[0]
        minus=owner.evaluate(pose-h*direction,tangent=False)[0]
        difference=(plus-minus)/(2*h)
        assert np.linalg.norm(tangent@direction-difference)/np.linalg.norm(difference)<=1e-5


def test_reference_partition_keeps_tiny_positive_cells_and_both_body_work():
    from anyfesolver.general_contact.s3_friction_replay import _split,_twice_area,_SOURCE_UV,_area_centroid,_rational_polygon
    from fractions import Fraction
    cells=_split(_SOURCE_UV,np.array((1e-12,0)),np.array((1e-12,1)))
    assert len(cells)==2
    assert sum(_twice_area(c) for c in cells)==Fraction(1)
    assert min(_twice_area(c) for c in cells)>0
    redundant=_rational_polygon(((0,0),(1,0),(1,0),(.5,.5),(0,1)))
    np.testing.assert_array_equal(_area_centroid(redundant),(Fraction(1,3),Fraction(1,3)))
    model,_d,_n,_settings,owner=_case()
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    owner.commit(owner.prepare_commit(-.02*z));u=-.02*z+.55*x
    force=owner._integrate(u)[0]
    for axis in range(3):assert abs(force@_mode(model,range(1,8),axis))<1e-11
    positions=((.75,.2,.045),(1.15,.2,.045),(.75,.6,.045),(0,0,0),(1,0,0),(0,1,0),(1,1,0))
    for omega in np.eye(3):
        spin=np.zeros(42)
        for node,pos in enumerate(positions,1):
            dofs=model.mesh.dof_manager.get_node_dofs(node)
            spin[dofs[:3]]=np.cross(omega,pos);spin[dofs[3:]]=omega
        assert abs(force@spin)<1e-11


def test_coupled_plane_translation_uses_admitted_reduced_tangent_direction():
    model,_d,_n,_s,owner=_case()
    source_z=_mode(model,(1,2,3),2);target_z=_mode(model,(4,5,6,7),2)
    owner.commit(owner.prepare_commit(-.02*source_z))
    T=sparse.csr_matrix(target_z[:,None]);owner.restrict_tangent_to_reduced_variations(T)
    u=-.02*source_z+.1*_mode(model,(1,2,3),0)
    _force,K,_cert=owner.evaluate(u)
    h=2e-5
    fd=(owner.evaluate(u+h*target_z,tangent=False)[0]-owner.evaluate(u-h*target_z,tangent=False)[0])/(2*h)
    assert np.linalg.norm(K@target_z-fd)/np.linalg.norm(fd)<1e-5


def test_partitioned_stage_entry_reuse_matches_complete_substep_checks():
    model,definition,normal,settings,_unused=_case()
    settings=replace(settings,event_subdivisions=4,prepared_pressure_paths=True)
    fast=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),settings)
    oracle=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),settings)
    original=oracle._step_endpoint
    oracle._step_endpoint=lambda *args,**kwargs:original(*args,path_partitioned=False)
    counts=[0,0]
    def counted(index,function):
        def call(*args,**kwargs):
            counts[index]+=1
            return function(*args,**kwargs)
        return call
    fast._entry=counted(0,fast._entry);oracle._entry=counted(1,oracle._entry)
    bary=np.array(((.2,.3,.5),(.1,.8,.1),(.8,.1,.1)))
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    # Manufactured affine material history: moving/straining target, partial
    # overlap, opening and recontact. This is a replay parity check, not a solve.
    for step,(slide,height) in enumerate(((0,-.02),(.55,-.02),(.54,0),(.54,-.02))):
        pose=slide*x+height*z
        for node in (4,5,6,7):
            dofs=model.mesh.dof_manager.get_node_dofs(node)
            pose[dofs[0]]=.001*step+(.001*step if node in (5,7) else 0.)
        a_frame=fast._frame(pose);b_frame=oracle._frame(pose)
        a=fast.query_material(0,bary,candidate=a_frame)
        b=oracle.query_material(0,bary,candidate=b_frame)
        for key in a[0]:np.testing.assert_array_equal(a[0][key],b[0][key])
        for expected,actual in zip(b[1:],a[1:]):np.testing.assert_array_equal(actual,expected)
        fast.frames=(*fast.frames,a_frame);oracle.frames=(*oracle.frames,b_frame)
    assert counts[0]<counts[1]/2


def _clipped_triangle_mixed_fixture(event_subdivisions):
    """Independent reference-area integral with a moving stick/slip front."""
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=event_subdivisions,
            prepared_pressure_paths=bool(event_subdivisions),
            prepared_partial_pressure_paths=bool(event_subdivisions)))
    x,z=(_mode(model,(1,2,3),axis) for axis in (0,2))
    target_x=_mode(model,(4,5,6,7),0)
    owner.restrict_tangent_to_reduced_variations(sparse.csr_matrix(np.column_stack((x,target_x))))
    a=.02;d=.001;delta0=.00599;s0=.8-delta0-(.4+d)*a
    # Reachable physical frames, for a local work check only (no global ledger).
    owner.frames=(owner._frame(.796*x-.02*z,check_path=False),
                  owner._frame(s0*x,check_path=False))
    pose=(s0+delta0)*x-.02*z
    pose[model.mesh.dof_manager.get_node_dofs(2)[0]]+=d
    # alpha=.75, final pressure5Pa. Proportional closure keeps each point on
    # its final stick/slip branch: tau=min(250*(delta0+d*b),1.5).
    # Material area density is2*A0*(1-b); overlap b in[0,a]. The old cut at
    # b=.01 happens to resolve this new stick front but has no physical history.
    c=(1.5/250-delta0)/d
    reference=.16*(1.5*(a-a*a/2)-250*d*(c*c/2-c**3/6))
    derivative=.16*(250*(c-c*c/2)-1.5*(1-a)/(.4+d))
    return owner,pose,x,reference,derivative


@pytest.mark.parametrize('event_subdivisions',[0,4])
def test_clipped_triangle_mixed_friction_independent_integral(tmp_path,event_subdivisions):
    owner,pose,x,reference,derivative=_clipped_triangle_mixed_fixture(event_subdivisions)
    rows=[];errors=[]
    for level in (1,2,3):
        for omit in (False,True):
            owner._separated_partition_enabled=omit
            force,_frame,summary=owner._integrate(pose,level=level)
            actual=float(x@force);error=abs(actual-reference)/abs(reference)
            rows.append(dict(level=level,omit_history=omit,reaction=actual,
                reference=reference,relative_error=error,points=summary['integration_points']))
            if omit:errors.append(error)
    (tmp_path/'clipped_triangle_reference.json').write_text(json.dumps(rows,indent=2))
    # Existing analytical response budget0.1%; integration consumes at most1/10.
    assert errors[-1]<=1e-4
    assert errors[-1]<=errors[0]
    owner.friction_settings=replace(owner.friction_settings,integration_refinement=3)
    before=owner.checkpoint_state();evaluations=owner.evaluation_count
    force,K,_cert=owner.evaluate(pose)
    assert owner._partition_retry_count==1
    assert owner.evaluation_count-evaluations==20
    assert owner._canonical_partition_frames is owner.frames
    assert owner.checkpoint_state()==before
    np.testing.assert_array_equal(owner.evaluate(pose,tangent=False)[0],force)
    normal_force,normal_K,_cert=owner.normal.evaluate(pose)
    actual_derivative=float(x@((K-normal_K)@x))
    (tmp_path/'clipped_triangle_tangent.json').write_text(json.dumps(dict(
        actual=actual_derivative,reference=derivative,
        relative_error=abs(actual_derivative-derivative)/abs(derivative)),indent=2))
    assert abs(actual_derivative-derivative)/abs(derivative)<=1e-3
    np.testing.assert_allclose(x@(force-normal_force),reference,rtol=1e-4,atol=1e-12)


def test_partial_pressure_ramp_work_refines_to_independent_slider(tmp_path):
    # Two points stay within the target throughout the active part of the path.
    # Pressure and tangential displacement grow proportionally from first touch.
    # Continuous slip L=max(D-cap/kt,0), work=cap*L/2, storage=tau**2/(2*kt).
    b=np.array((.005,.015))
    bary=np.column_stack(((1-b)/2,b,(1-b)/2))
    distance=.25*(.00599+.001*b)
    traction=np.minimum(1000*distance,1.5)
    slip=np.maximum(distance-1.5/1000,0)
    work=.5*1.5*slip
    storage=traction**2/2000
    rows=[]
    for subdivisions in (8,32,128):
        owner,pose,_x,_reference,_derivative=_clipped_triangle_mixed_fixture(subdivisions)
        before=owner.checkpoint_state()
        state,t,diss,_gap,_returned,_pressure=owner._step(
            0,bary,owner._empty(2),owner.frames[-1],owner._frame(pose))
        np.testing.assert_allclose(np.linalg.norm(t,axis=1),traction,rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose(state['slip'],slip,rtol=1e-7,atol=1e-12)
        np.testing.assert_allclose(state['storage'],storage,rtol=1e-9,atol=1e-12)
        assert abs(diss[0])<=1e-12
        error=float(abs(diss[1]-work[1])/work[1])
        rows.append(dict(subdivisions=subdivisions,dissipation=diss.tolist(),
                         reference=work.tolist(),relative_error=error))
        assert owner.checkpoint_state()==before
    (tmp_path/'partial_pressure_ramp_work.json').write_text(json.dumps(rows,indent=2))
    assert rows[2]['relative_error']<rows[1]['relative_error']<rows[0]['relative_error']
    assert rows[-1]['relative_error']<=.01


def _material_work_reference_identity(owner,bary,weight):
    """Bind the physical path, sampling and all settings except the varied N."""
    import hashlib
    contract=owner.restart_contract()
    contract['settings'].pop('event_subdivisions')
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True,allow_nan=False).encode())
    def bind(value):
        array=np.asarray(value,dtype='<f8')
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    bind(bary);bind(weight);bind(owner.work_columns)
    for frame in (owner.initial_frame,*owner.frames):
        for key in ('local','pressure','normal','origin'):bind(frame[key])
        for source in frame['source']:bind(source)
        for element,target in sorted(frame['target'].items()):bind((element,));bind(target)
    return digest.hexdigest()


def test_saved_workflow_material_work_refinement_discriminator(tmp_path):
    """Fixed accepted physical paths; no re-equilibration or restart claim."""
    import time
    saved=json.loads((Path(__file__).resolve().parents[1]/
        'docs/d1_review_support/contact_c5_s3_partial_replay_20261002/separation_full_v1_checkpoint.json').read_text())
    frames=saved['path_state']['friction_state']['frames']
    rows=[]
    for subdivisions,level in ((4,0),(4,1),(12,0),(32,0)):
        model,definition,normal,settings,_unused=_case()
        owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
            replace(settings,event_subdivisions=subdivisions,prepared_pressure_paths=True,
                    prepared_partial_pressure_paths=True))
        prepared=[]
        for item in frames:
            frame=owner._frame(owner._full(np.asarray(item['local'])),check_path=False)
            np.testing.assert_allclose(frame['pressure'],item['pressure'],rtol=1e-12,atol=1e-12)
            prepared.append(frame)
        owner.frames=tuple(prepared)
        bary,weight=owner._quadrature(0,owner.frames[-1],level,full=True)
        before=owner.checkpoint_state();start=time.perf_counter()
        state=owner.query_material(0,bary)
        row=dict(subdivisions=subdivisions,integration_level=level,points=len(bary),
            input_identity=_material_work_reference_identity(owner,bary,weight),
            seconds=time.perf_counter()-start,
            dissipation_j=float(weight@state['dissipation']),storage_j=float(weight@state['storage']),
            released_storage_j=float(weight@state['released_storage']),
            prepared_partial_samples=owner.prepared_partial_pressure_samples,
            ordinary_stages=owner.event_stage_evaluations)
        rows.append(row)
        (tmp_path/'saved_workflow_work_refinement.json').write_text(json.dumps(dict(
            scope='fixed-path development discriminator; not coupled or spatial qualification',
            saved_cumulative_dissipation_j=saved['path_state']['friction_state']['cumulative_dissipation_j'],
            rows=rows),indent=2))
        assert owner.checkpoint_state()==before
        assert all(np.isfinite(row[k]) and row[k]>=0 for k in
                   ('dissipation_j','storage_j','released_storage_j'))


def test_saved_workflow_step_partition_preserves_refined_dissipation(tmp_path):
    """Split the same affine path; isolate step partition from spatial sampling."""
    import time
    evidence=Path(__file__).resolve().parents[1]/'docs/d1_review_support/contact_c5_s3_partial_replay_20261002'
    saved=json.loads((evidence/'separation_full_v1_checkpoint.json').read_text())
    reference=json.loads((evidence/'saved_workflow_work_refinement.json').read_text())
    reference=next(row for row in reference['rows'] if row['subdivisions']==32)
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=16,prepared_pressure_paths=True,
                prepared_partial_pressure_paths=True))
    original=tuple(owner._frame(owner._full(np.asarray(item['local'])),check_path=False)
                   for item in saved['path_state']['friction_state']['frames'])
    owner.frames=original
    bary,weight=owner._quadrature(0,original[-1],0,full=True)
    assert len(bary)==reference['points']
    identity=_material_work_reference_identity(owner,bary,weight)
    assert identity==reference['input_identity']
    partitioned=[];first=owner.initial_frame
    for last in original:
        partitioned.extend((owner._frame(owner._full((first['local']+last['local'])/2),
                                        check_path=False),last))
        first=last
    owner.frames=tuple(partitioned)
    before=owner.checkpoint_state();start=time.perf_counter()
    state=owner.query_material(0,bary)
    work=float(weight@state['dissipation'])
    error=abs(work-reference['dissipation_j'])/reference['dissipation_j']
    result=dict(scope='fixed-path step partition; not coupled qualification',
                input_identity=identity,
                points=len(bary),history_frames=len(owner.frames),seconds=time.perf_counter()-start,
                dissipation_j=work,reference_j=reference['dissipation_j'],relative_error=error,
                storage_j=float(weight@state['storage']),
                released_storage_j=float(weight@state['released_storage']),
                partial_samples=owner.prepared_partial_pressure_samples)
    (tmp_path/'saved_workflow_step_partition.json').write_text(json.dumps(result,indent=2))
    assert owner.checkpoint_state()==before
    assert error<=.02


def test_canonical_partition_retry_respects_shared_candidate_budget():
    owner,pose,_x,_reference,_derivative=_clipped_triangle_mixed_fixture(4)
    owner.friction_settings=replace(owner.friction_settings,integration_refinement=3,
        material=replace(owner.friction_settings.material,maximum_candidate_evaluations=10))
    before=owner.checkpoint_state();evaluations=owner.evaluation_count
    with pytest.raises(MemoryError,match='canonical partition retry exceeds'):
        owner.evaluate(pose)
    assert owner.evaluation_count-evaluations==10
    assert owner._canonical_partition_frames is None
    assert owner.checkpoint_state()==before


@pytest.mark.parametrize('failure',[ValueError,KeyboardInterrupt])
def test_canonical_partition_retry_rolls_back_interrupted_evaluation(monkeypatch,failure):
    owner,pose,_x,_reference,_derivative=_clipped_triangle_mixed_fixture(4)
    owner.friction_settings=replace(owner.friction_settings,integration_refinement=3)
    before=owner.checkpoint_state()
    original=owner._evaluate_once
    calls=0
    def interrupted(u,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise failure('interrupted canonical evaluation')
        return original(u,**kwargs)
    monkeypatch.setattr(owner,'_evaluate_once',interrupted)
    with pytest.raises(failure,match='interrupted canonical evaluation'):
        owner.evaluate(pose)
    assert calls==2
    assert owner._partition_retry_count==1
    assert owner._canonical_partition_frames is None
    assert owner.checkpoint_state()==before


def test_clipped_triangle_derivative_discriminator(tmp_path):
    owner,pose,x,_reference,derivative=_clipped_triangle_mixed_fixture(4)
    rows=[]
    cases=[(3,omit,h) for omit in (False,True) for h in (2e-7,2e-8,2e-9)]
    cases.extend((4,omit,2e-9) for omit in (False,True))
    for level,omit,h in cases:
        owner._separated_partition_enabled=omit
        def value(delta):return owner._integrate(pose+delta*x,level=level)[0]
        coarse=(value(h)-value(-h))/(2*h)
        fine=(value(h/2)-value(-h/2))/h
        actual=float(x@fine)
        rows.append(dict(level=level,omit_history=omit,step=h,actual=actual,
            reference=derivative,relative_oracle_error=abs(actual-derivative)/abs(derivative),
            relative_probe_difference=float(np.linalg.norm(fine-coarse)/np.linalg.norm(fine))))
        (tmp_path/'mixed_derivative_discriminator.json').write_text(json.dumps(rows,indent=2))
    assert all(np.isfinite(row['relative_oracle_error']) for row in rows)


def test_saved_recontact_separated_partition_comparison(tmp_path):
    import time
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True,prepared_partial_pressure_paths=True))
    saved=json.loads((Path(__file__).resolve().parents[1]/
        'docs/d1_review_support/contact_c5_s3_partial_replay_20261002/prepared_partial_full_v1_last_tangent_candidate.json').read_text())
    owner.frames=tuple(owner._frame(owner._full(np.asarray(frame['local'])),check_path=False)
                       for frame in saved['state']['frames'])
    pose=np.asarray(saved['displacement'])
    assert owner._source_strictly_separated(0)
    rows=[]
    for level in (settings.integration_refinement,settings.integration_refinement+1):
        outputs=[]
        for enabled in (False,True):
            owner._separated_partition_enabled=enabled
            start=time.perf_counter();force,_frame,summary=owner._integrate(pose,level=level)
            outputs.append((force,summary))
            rows.append(dict(level=level,omit_history_partition=enabled,
                elapsed_seconds=time.perf_counter()-start,summary=summary,force=force.tolist()))
            (tmp_path/'separated_partition_comparison.json').write_text(json.dumps(rows,indent=2))
        owner._check_integration(*outputs[0],*outputs[1])


def test_saved_recontact_cached_tangent_cost(tmp_path):
    _saved_recontact_tangent_cost(tmp_path,cache_enabled=True)


def test_saved_recontact_uncached_tangent_cost(tmp_path):
    _saved_recontact_tangent_cost(tmp_path,cache_enabled=False)


def _saved_recontact_tangent_cost(tmp_path,*,cache_enabled):
    """One retained coupled tangent, not a continuation or acceptance run."""
    import time
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True,prepared_partial_pressure_paths=True))
    saved=json.loads((Path(__file__).resolve().parents[1]/
        "docs/d1_review_support/contact_c5_s3_partial_replay_20261002/prepared_partial_full_v1_last_tangent_candidate.json").read_text())
    owner.frames=tuple(owner._frame(owner._full(np.asarray(frame['local'])),check_path=False)
                       for frame in saved['state']['frames'])
    # Disposable mechanical replay of the retained poses; no restart-ledger claim.
    owner.reduced_directions=np.asarray(saved['reduced_directions'])
    owner.columns=owner.work_columns
    owner._material_history_cache_enabled=cache_enabled
    before=owner.frames
    start=time.perf_counter()
    try:
        force,tangent,_certificate=owner.evaluate(np.asarray(saved['displacement']))
    finally:
        (tmp_path/'recontact_cached_tangent_cost.json').write_text(json.dumps(dict(
            elapsed_seconds=time.perf_counter()-start,
            cache_enabled=cache_enabled,
            material_cache_points=len(owner._material_history_cache),
            evaluations=owner.evaluation_count),indent=2),encoding='utf-8')
    assert owner.frames is before
    assert np.all(np.isfinite(force)) and np.all(np.isfinite(tangent.data))
    np.savez(tmp_path/'recontact_cached_tangent.npz',force=force,tangent=tangent.toarray())


def test_saved_recontact_history_cache_batch_parity():
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True,prepared_partial_pressure_paths=True))
    saved=json.loads((Path(__file__).resolve().parents[1]/
        "docs/d1_review_support/contact_c5_s3_partial_replay_20261002/prepared_partial_full_v1_last_tangent_candidate.json").read_text())
    owner.frames=tuple(owner._frame(owner._full(np.asarray(frame['local'])),check_path=False)
                       for frame in saved['state']['frames'])
    frame=owner._frame(np.asarray(saved['displacement']),check_path=False)
    bary,_weights=owner._quadrature(0,frame,settings.integration_refinement)
    bary=bary[::max(1,len(bary)//24)]
    owner._material_history_cache_enabled=False
    expected=owner.query_material(0,bary)
    owner._material_history_cache_enabled=True
    owner.query_material(0,bary[::2])
    for selection in (np.arange(len(bary))[::-1],np.arange(len(bary))):
        actual=owner.query_material(0,bary[selection])
        for name in actual:
            np.testing.assert_allclose(actual[name],expected[name][selection],rtol=1e-12,atol=1e-14)


def test_saved_recontact_material_query_reuse(tmp_path):
    """Geometry-only count of exact immutable queries in the retained tangent."""
    from collections import OrderedDict
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True,prepared_partial_pressure_paths=True))
    saved=json.loads((Path(__file__).resolve().parents[1]/
        "docs/d1_review_support/contact_c5_s3_partial_replay_20261002/prepared_partial_full_v1_last_tangent_candidate.json").read_text())
    owner.frames=tuple(owner._frame(owner._full(np.asarray(frame["local"])),check_path=False)
                       for frame in saved["state"]["frames"])
    pose=np.asarray(saved["displacement"]);directions=np.asarray(saved["reduced_directions"])
    cache=OrderedDict();queries=hits=0;rows=[]
    def collect(u,level):
        nonlocal queries,hits
        frame=owner._frame(u,check_path=False)
        bary,_weights=owner._quadrature(0,frame,level)
        found=0
        for point in bary:
            key=point.tobytes()
            if key in cache:found+=1;cache.move_to_end(key)
            else:
                cache[key]=None
                if len(cache)>settings.maximum_integration_points:cache.popitem(last=False)
        queries+=len(bary);hits+=found
        rows.append(dict(points=len(bary),hits=found,level=level))
    collect(pose,settings.integration_refinement)
    collect(pose,settings.integration_refinement+1)
    limits=np.asarray(settings.material.steps_by_dof)[owner.work_columns]
    for direction in directions.T:
        nonzero=direction!=0;step=np.min(limits[nonzero]/abs(direction[nonzero]))
        delta=np.zeros(owner.n_dof);delta[owner.work_columns]=step*direction
        for factor in (1.,-1.,.5,-.5):collect(pose+factor*delta,settings.integration_refinement)
    result=dict(queries=queries,hits=hits,hit_fraction=hits/queries,cache_points=len(cache),
        cache_cap=settings.maximum_integration_points,evaluations=rows,
        scope="exact material coordinate reuse count, no mechanical/performance acceptance")
    (tmp_path/"recontact_query_reuse.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    assert len(rows)==2+4*directions.shape[1]


def test_saved_partial_replay_residual_cost(tmp_path):
    """One residual at retained physical poses; no continuation/ledger claim."""
    import cProfile
    import pstats
    import time
    path=Path(__file__).resolve().parents[1]/"docs/d1_review_support/contact_c5_s3_partial_replay_20261002/unresolved_opening_candidate.json"
    saved=json.loads(path.read_text(encoding="utf-8"))
    model,definition,normal,settings,_unused=_case()
    owner=S3ReplayFrictionManager(model,GeneralS3PatchManager(model,definition,normal),
        replace(settings,event_subdivisions=4,prepared_pressure_paths=True))
    frames=[]
    for item in saved["state"]["frames"][:2]:
        frame=owner._frame(owner._full(np.asarray(item["local"])),check_path=False)
        np.testing.assert_allclose(frame["pressure"],item["pressure"],rtol=1e-12,atol=1e-12)
        frames.append(frame)
    owner.frames=tuple(frames)
    candidate=owner._full(np.asarray(saved["state"]["frames"][2]["local"]))
    profile=cProfile.Profile();start=time.perf_counter()
    try:
        with profile:force,_frame,summary=owner._integrate(candidate)
    finally:
        elapsed=time.perf_counter()-start
        profile.dump_stats(str(tmp_path/"partial_residual.prof"))
        with (tmp_path/"partial_residual_profile.txt").open("w",encoding="utf-8") as stream:
            pstats.Stats(profile,stream=stream).strip_dirs().sort_stats("cumulative").print_stats(35)
    (tmp_path/"partial_residual_cost.json").write_text(json.dumps(dict(
        elapsed_seconds=elapsed,force=force.tolist(),summary=summary,
        eligible_paths=sum(value is not None for value in owner._prepared_path_cache.values()),
        total_paths=len(owner._prepared_path_cache)),indent=2),encoding="utf-8")
    assert np.all(np.isfinite(force))


def test_saved_opening_candidate_derivative_discriminator(tmp_path):
    """One retained failure: vary only integration, with identical history."""
    path=Path(__file__).resolve().parents[1]/"docs/d1_review_support/contact_c5_s3_partial_replay_20261002/unresolved_opening_candidate.json"
    saved=json.loads(path.read_text(encoding="utf-8"))
    _m,_d,_n,_s,owner=_case()
    # The retained input uses the previous quadrature ledger. Reconstruct a
    # disposable candidate from its identical physical poses/pressure fields;
    # this is not continuation of that incompatible checkpoint.
    for frame in saved["state"]["frames"]:
        prepared=owner.prepare_commit(owner._full(np.array(frame["local"])))
        np.testing.assert_allclose(prepared[1]["pressure"],frame["pressure"],rtol=1e-12,atol=1e-12)
        owner.commit(prepared)
    u=np.array(saved["displacement"]);directions=np.array(saved["reduced_directions"])
    def derivative(direction,level,h=2e-7):
        delta=np.zeros(42);delta[owner.work_columns]=h*direction
        coarse=(owner._integrate(u+delta,level=level)[0]-owner._integrate(u-delta,level=level)[0])/(2*h)
        fine=(owner._integrate(u+delta/2,level=level)[0]-owner._integrate(u-delta/2,level=level)[0])/h
        return coarse,fine
    initial=[derivative(v,1) for v in directions.T]
    worst=int(np.argmax([np.linalg.norm(a-b) for a,b in initial]))
    values=[]
    for level in (1,2,3):
        coarse,fine=initial[worst] if level==1 else derivative(directions[:,worst],level)
        values.append(dict(level=level,direction=worst,relative=float(np.linalg.norm(coarse-fine)/max(np.linalg.norm(fine),1e-30)),
            derivative_norm=float(np.linalg.norm(fine))))
    for h in (2e-8,2e-9):
        coarse,fine=derivative(directions[:,worst],1,h)
        values.append(dict(level=1,step=h,direction=worst,relative=float(np.linalg.norm(coarse-fine)/max(np.linalg.norm(fine),1e-30)),
            derivative_norm=float(np.linalg.norm(fine))))
    (tmp_path/"tangent_resolution.json").write_text(json.dumps(values,indent=2),encoding="utf-8")
    assert all(np.isfinite(v["relative"]) and v["relative"]<=1e-5 for v in values)
