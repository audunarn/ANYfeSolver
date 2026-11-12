"""Bounded profiling and exact-control candidate/baseline parity."""
import cProfile,json,os,pstats
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'full-baseline-1.json').is_file(),
    reason='archived baseline input AUTOMATIC_WORKFLOW_EVIDENCE/full-baseline-1.json absent: saved-candidate replay requires the original evidence root')
def test_saved_candidate_profile():
    from examples.general_contact_automatic_q4_pair import build
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.q4_q4_overlap_derivatives import q4_q4_overlap_derivatives
    model,domain,settings,resolution,path=build()
    manager=GeneralS3PatchManager(model,resolution.definition,settings)
    root=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    report=json.loads((root/'full-baseline-1.json').read_text());u=np.array(report['displacements'])
    source,target=manager._positions(u);si=manager.law.source_face_indices[0];ti=manager.law.target_face_indices[0]
    reference=manager.law.geometry.source_faces[0].reference_positions
    CONTROLS=dict(area_tolerance=settings.q4_source_area_tolerance,weighted_gap_tolerance=settings.q4_source_weighted_gap_tolerance,maximum_triangles=settings.maximum_q4_source_integration_triangles,support_gradient_tolerance=settings.q4_source_support_gradient_tolerance,support_hessian_tolerance=settings.q4_source_support_hessian_tolerance,weighted_gap_gradient_tolerance=settings.q4_source_weighted_gap_gradient_tolerance,weighted_gap_hessian_tolerance=settings.q4_source_weighted_gap_hessian_tolerance)
    profiler=cProfile.Profile();profiler.enable()
    result=q4_q4_overlap_derivatives(reference,source[si],target[ti],**CONTROLS)
    profiler.disable();profiler.dump_stats(str(root/'saved-candidate.prof'))
    with (root/'saved-candidate-profile.txt').open('w') as stream:pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(35)
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(integration_triangles=result.integration_triangles,branch=repr(result.branch_signature)),indent=2),encoding='utf-8')


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'install.json').is_file() or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'outputs/full-baseline-1.json').is_file(),
    reason='archived baseline inputs AUTOMATIC_WORKFLOW_EVIDENCE install.json/outputs/full-baseline-1.json absent: baseline-peer parity replay requires the original evidence root')
def test_smooth_parity_and_real_both_body_work(monkeypatch):
    import importlib.util,sys
    from anyfesolver.general_contact import q4_q4_overlap_derivatives as current
    raw=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE']).parent
    install=json.loads((raw/'install.json').read_text())
    module_name='_measured_cost_baseline'
    spec=importlib.util.spec_from_file_location(module_name,Path(install['site'])/'anyfesolver/general_contact/q4_q4_overlap_derivatives.py')
    baseline=importlib.util.module_from_spec(spec);baseline.__package__='anyfesolver.general_contact';sys.modules[module_name]=baseline;spec.loader.exec_module(baseline)
    from examples.general_contact_automatic_q4_pair import build
    model,domain,settings,resolution,path=build()
    controls=dict(area_tolerance=settings.q4_source_area_tolerance,weighted_gap_tolerance=settings.q4_source_weighted_gap_tolerance,maximum_triangles=512,support_gradient_tolerance=settings.q4_source_support_gradient_tolerance,support_hessian_tolerance=settings.q4_source_support_hessian_tolerance,weighted_gap_gradient_tolerance=settings.q4_source_weighted_gap_gradient_tolerance,weighted_gap_hessian_tolerance=settings.q4_source_weighted_gap_hessian_tolerance)
    reference=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
    target=np.array([[.1,.1,-.003],[.9,.1,-.002],[.9,.9,-.003],[.1,.9,-.002]])
    cases=[('rectangle',reference,reference.copy(),target)]
    trapezoid=reference.copy();trapezoid[2,0]=.8
    cases.append(('nonaffine_planar_reference',trapezoid,trapezoid.copy(),target))
    warped=reference.copy();warped[2,2]=.0003;warped[1,2]=.0001
    cases.append(('warped_current',reference,warped,target))
    fields=('reference_support','reference_weighted_gap','support_gradient','support_hessian','weighted_gap_gradient','weighted_gap_hessian','derivative_quadrature_differences')
    report=[]
    for name,ref,source,target in cases:
        old=baseline.q4_q4_overlap_derivatives(ref,source,target,**controls)
        new=current.q4_q4_overlap_derivatives(ref,source,target,**controls)
        errors={}
        for field in fields:
            a,b=np.asarray(getattr(old,field)),np.asarray(getattr(new,field));err=float(np.max(np.abs(a-b))/max(1.,float(np.max(np.abs(a)))))
            assert err<=1e-13,(name,field,err);errors[field]=err
        assert old.branch_signature==new.branch_signature and old.integration_triangles==new.integration_triangles
        assert abs(old.overlap_area-new.overlap_area)<=1e-13
        report.append(dict(case=name,scaled_errors=errors,integration_triangles=new.integration_triangles,branch_equal=True))
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    manager=GeneralS3PatchManager(model,resolution.definition,settings)
    pose=np.array(json.loads((raw/'outputs/full-baseline-1.json').read_text())['displacements'])
    optimized=current.q4_q4_overlap_derivatives
    old_call=current.q4_q4_overlap_derivatives
    monkeypatch.setattr(current,'q4_q4_overlap_derivatives',baseline.q4_q4_overlap_derivatives)
    old=manager.evaluate(pose,tangent=True)
    monkeypatch.setattr(current,'q4_q4_overlap_derivatives',optimized)
    new=manager.evaluate(pose,tangent=True)
    # Actual contributor returns full prescribed-coordinate reactions too.
    force_error=float(np.max(np.abs(old[0]-new[0]))/max(1.,np.max(np.abs(old[0]))))
    a,b=old[1].toarray(),new[1].toarray();tangent_error=float(np.max(np.abs(a-b))/max(1.,np.max(np.abs(a))))
    assert force_error<=1e-13 and tangent_error<=1e-13
    assert np.count_nonzero(new[0][[model.mesh.dof_manager.get_node_dofs(n)[2] for n in range(1,16)]])>1
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(local_cases=report,actual_force_error=force_error,actual_tangent_error=tangent_error,process_cache=False,baseline_peer_source=str(spec.origin),baseline_peer_sha256=__import__('hashlib').sha256(Path(spec.origin).read_bytes()).hexdigest(),baseline_wheel_sha256=install['wheel_sha256']),indent=2),encoding='utf-8')


def test_actual_q4_backend_scope_and_legacy_dispatch(monkeypatch):
    import anyfesolver.general_contact.q4_s3_overlap_derivatives as helper
    from anyfesolver.general_contact.q4_q4_overlap_derivatives import q4_q4_overlap_derivatives
    calls=[]
    def compiled21(av,ag,ah,bv,bg,bh):
        assert ag.shape[1]==bg.shape[1]==21,'24-coordinate Q4 route reached compiled backend'
        calls.append((ag.shape[1],bg.shape[1]))
        # Exact existing NumPy algebra as the backend spy; no JIT execution.
        outer=ag[:,:,None]*bg[:,None,:]
        return av*bv,ag*bv[:,None]+bg*av[:,None],ah*bv[:,None,None]+bh*av[:,None,None]+outer+np.swapaxes(outer,1,2)
    monkeypatch.setattr(helper,'_compiled_bjet_mul',compiled21)
    monkeypatch.setattr(helper,'_COMPILED_BJET_MUL_ENABLED',True)
    ref=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
    target=np.array([[.1,.1,-.003],[.9,.1,-.003],[.9,.9,-.003],[.1,.9,-.003]])
    controls=dict(area_tolerance=1e-9,weighted_gap_tolerance=1e-10,maximum_triangles=128,support_gradient_tolerance=1e-5,support_hessian_tolerance=1e-5,weighted_gap_gradient_tolerance=1e-5,weighted_gap_hessian_tolerance=1e-5)
    enabled=q4_q4_overlap_derivatives(ref,ref,target,**controls);assert not calls
    monkeypatch.setattr(helper,'_COMPILED_BJET_MUL_ENABLED',False)
    disabled=q4_q4_overlap_derivatives(ref,ref,target,**controls)
    for field in ('reference_support','reference_weighted_gap','support_gradient','support_hessian','weighted_gap_gradient','weighted_gap_hessian'):
        assert np.array_equal(getattr(enabled,field),getattr(disabled,field))
    triangle=np.array([[-1.,-1.,-.003],[4.,-1.,-.003],[-1.,4.,-.003]])
    old_numpy=helper.q4_s3_overlap_derivatives(ref,ref,triangle,**controls)
    monkeypatch.setattr(helper,'_COMPILED_BJET_MUL_ENABLED',True)
    old_dispatch=helper.q4_s3_overlap_derivatives(ref,ref,triangle,**controls)
    assert calls and all(x==(21,21) for x in calls)
    maximum=0.
    for field in ('reference_support','reference_weighted_gap','support_gradient','support_hessian','weighted_gap_gradient','weighted_gap_hessian'):
        a,b=getattr(old_numpy,field),getattr(old_dispatch,field);error=float(np.max(np.abs(a-b))/max(1.,np.max(np.abs(a))));maximum=max(maximum,error);assert error<=1e-13
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(actual_24_coordinate_compiled_trap_pass=True,flag_enabled_disabled_q4_exact=True,actual_legacy_21_dispatch_calls=len(calls),legacy_scaled_parity=maximum,jit_campaign=False),indent=2),encoding='utf-8')
