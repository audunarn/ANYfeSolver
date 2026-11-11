"""Independent swept geometry checks and bounded automatic lifecycle evidence."""
import json,os
from pathlib import Path
import numpy as np
import pytest

from anyfesolver.general_contact.q4_overlap_local_bound import (
    q4_s3_whole_pair_swept_lower,overlap_local_swept_lower,q4_s3_swept_chart_admissible,
    _normal_controls_interval)

Q4=np.array(((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)))
S3=np.array(((.2,.2,0.),(.6,.2,0.),(.2,.6,0.)))

@pytest.mark.parametrize('source_z,target_z',[
    ((-.000005,.015),((0.,0.,.000288),(0.,0.,0.))),
    ((.01,.02),((-.03,-.01,-.02),(.002,.004,.003))),
    ((0.,0.),((-.1,-.1,-.1),(.1,.1,.1))),
])
@pytest.mark.parametrize('rotation',[
    np.eye(3),np.array(((0.,0.,1.),(1.,0.,0.),(0.,1.,0.))),
    np.array(((.6,-.8,0.),(.8,.6,0.),(0.,0.,1.))),
])
def test_exact_affine_plane_triangular_paths(source_z,target_z,rotation):
    """Independent closed-form min of affine transverse distances.

    All source points share a plane; each target vertex height is affine in
    time. The exact whole-pair minimum is at one of six vertex/time endpoints.
    Rotation exercises covariant normals without choosing a world-axis law.
    """
    a0=Q4+(0.,0.,source_z[0]);a1=Q4+(0.,0.,source_z[1])
    b0=S3.copy();b1=S3.copy();b0[:,2]=target_z[0];b1[:,2]=target_z[1]
    exact=min(0.,*(source_z[k]-z for k in (0,1) for z in target_z[k]))
    lower=q4_s3_whole_pair_swept_lower(*(a@rotation.T for a in (a0,a1,b0,b1)))
    assert lower<=exact+2e-15
    assert exact-lower<2e-12


def test_interior_crossing_with_positive_endpoint_gaps():
    # Source plane z=t*x; target centre x=t,z=t-.01. Independent exact gap
    # is (t*t-t+.01)/sqrt(1+t*t), positive at endpoints, negative at t=.5.
    a0=Q4.copy();a1=Q4.copy();a1[:,2]=a1[:,0]
    b0=np.array(((0.,.2,-.01),(.002,.2,-.01),(0.,.4,-.01)))
    b1=b0+(1.,0.,1.)
    assert .01>0 and .008/np.sqrt(2.)>0
    assert (.25-.5+.01)/np.sqrt(1.25)<-.2
    assert q4_s3_whole_pair_swept_lower(a0,a1,b0,b1)<-.2


def test_middle_only_overlap_crossing_is_not_pruned_by_endpoint_union():
    assert q4_s3_whole_pair_swept_lower(Q4,Q4,S3+(-2.,0.,.1),S3+(2.,0.,.1))<=-.1


@pytest.mark.parametrize('body',['source','target'])
def test_endpoint_polarity_does_not_hide_interior_collapse(body):
    a1=Q4.copy();b1=S3.copy()
    if body=='source':a1=Q4*np.array((-1.,-1.,1.))
    else:b1=S3*np.array((-1.,-1.,1.))
    assert q4_s3_whole_pair_swept_lower(Q4,a1,S3,b1)==-np.inf


def test_original_endpoint_error_enclosures_cannot_hide_penetration():
    error=np.array((0.,0.,.001))
    lower=q4_s3_whole_pair_swept_lower(Q4,Q4,S3,S3,endpoint_errors=(error,)*4)
    assert lower<=-.002
    with pytest.raises(ValueError,match='nonnegative'):
        q4_s3_whole_pair_swept_lower(Q4,Q4,S3,S3,endpoint_errors=(-1.,0.,0.,0.))


def test_invalid_and_unresolved_geometry_refuses():
    bad=S3.copy();bad[2]=bad[1]
    assert q4_s3_whole_pair_swept_lower(Q4,Q4,bad,bad)==-np.inf
    bad=Q4.copy();bad[2]=(.2,-.2,0.)
    assert q4_s3_whole_pair_swept_lower(bad,bad,S3,S3)==-np.inf
    bad=Q4.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        q4_s3_whole_pair_swept_lower(bad,Q4,S3,S3)


def test_warped_q1_gap_against_independent_exact_material_points():
    """Adverse controls from exact Q1 derivatives, not corner-normal bounds.

    This oracle is supporting counterexample search; the all-point proof is
    the convex/Bernstein enclosure, not this finite set of rational points.
    """
    from fractions import Fraction as F
    from decimal import Decimal,localcontext
    a0=Q4.copy();a1=Q4.copy();a0[2,2]=.125;a1[2,2]=-.125;a1[1,2]=.03125
    b0=S3.copy();b1=S3.copy();b0[:,2]=(.0625,.03125,0.);b1[:,2]=(.03125,-.0625,.0625)
    lower=q4_s3_whole_pair_swept_lower(a0,a1,b0,b1)
    assert np.isfinite(lower) and lower<0.
    exact=lambda a:[[F(float(x)) for x in row] for row in a]
    first,last,target0,target1=map(exact,(a0,a1,b0,b1))
    def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    for t in (F(0),F(3,7),F(1)):
        nodes=[[x+t*(y-x) for x,y in zip(row0,row1)] for row0,row1 in zip(first,last)]
        target=[[x+t*(y-x) for x,y in zip(row0,row1)] for row0,row1 in zip(target0,target1)]
        for u,v in ((F(1,3),F(2,5)),(F(3,4),F(1,4)),(F(1),F(1))):
            weights=((1-u)*(1-v),u*(1-v),u*v,(1-u)*v)
            du=(-(1-v),1-v,v,-v);dv=(-(1-u),-u,u,1-u)
            at=lambda coefficients:[sum(c*row[j] for c,row in zip(coefficients,nodes)) for j in range(3)]
            point=at(weights);normal=cross(at(du),at(dv))
            target_point=[sum(c*row[j] for c,row in zip((F(1,5),F(3,10),F(1,2)),target)) for j in range(3)]
            numerator=sum((x-y)*n for x,y,n in zip(point,target_point,normal))
            norm_squared=sum(n*n for n in normal)
            with localcontext() as context:
                context.prec=80
                decimal=lambda x:Decimal(x.numerator)/Decimal(x.denominator)
                true_gap=decimal(numerator)/decimal(norm_squared).sqrt()
                assert Decimal.from_float(lower)<=true_gap


def test_large_coordinate_cancellation_encloses_exact_transverse_gap():
    shift=np.array((2.**30,-2.**30,2.**30))
    distance=2.**-12
    assert q4_s3_whole_pair_swept_lower(Q4+shift,Q4+shift,S3+shift+(0.,0.,distance),S3+shift+(0.,0.,distance))<=-distance


def test_public_lifecycle_precancel_is_atomic():
    from anyfesolver import CancellationToken,SolveCancelled
    from examples.general_contact_automatic_domain import build_projected_shell_pair,solve_shell_pair
    model,domain,settings,resolution,path=build_projected_shell_pair(lifecycle=True)
    before=repr(model)
    token=CancellationToken();token.cancel('independent lifecycle cancellation control')
    with pytest.raises(SolveCancelled):
        solve_shell_pair(model,domain,settings,resolution,path,cancellation_token=token)
    assert repr(model)==before


def test_individual_polarity_does_not_certify_mutual_chart():
    # Exact dyadic affine edges: both target endpoint z-normal components
    # are +1, and all target normal controls have positive own-axis dot.
    # The middle target z-normal is negative, facing away from the source.
    b0=np.array(((0.,0.,-20.),(1.,0.,-28.),(0.,1.,-20.)))
    b1=np.array(((0.,0.,-20.),(-16.,1.,-21.9375),(-2.,.0625,-20.2421875)))
    target=_normal_controls_interval(b0,b1)
    exact=np.array(((8.,0.,1.),(1.21875,8.12109375,-7.96875),(-.12109375,0.,1.)))
    assert np.all(target[0]<=exact) and np.all(exact<=target[1])
    own=np.array((8.,0.,1.))
    np.testing.assert_array_equal(exact@own,(65.,1.78125,.03125))
    assert .25*exact[0,2]+.5*exact[1,2]+.25*exact[2,2]==-3.484375
    assert not q4_s3_swept_chart_admissible(Q4,Q4,b0,b1)
    assert q4_s3_whole_pair_swept_lower(Q4,Q4,b0,b1)==-np.inf


def test_endpoint_skin_map_error_encloses_exact_sparse_arithmetic():
    from fractions import Fraction as F
    from examples.general_contact_automatic_domain import build_projected_shell_pair
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    model,_,settings,resolution,_=build_projected_shell_pair(lifecycle=True)
    manager=GeneralS3PatchManager(model,resolution.definition,settings)
    matrix=manager.map.tocsr();reference=np.concatenate(manager.reference).reshape(-1)
    for scale in (0.,2.**-7,2.**45):
        u=scale*np.arange(manager.n_dof)*np.where(np.arange(manager.n_dof)%2,1.,-1.)
        positions=np.concatenate(manager._positions(u)).reshape(-1)
        errors=np.concatenate(manager._position_errors(u)).reshape(-1)
        for row in range(matrix.shape[0]):
            exact=F(float(reference[row]))+sum(F(float(matrix.data[k]))*F(float(u[matrix.indices[k]]))
                for k in range(matrix.indptr[row],matrix.indptr[row+1]))
            centre=F(float(positions[row]));error=F(float(errors[row]))
            assert centre-error<=exact<=centre+error


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_SWEPT_TRACE_FILE'),reason='retained actual proposal oracle')
def test_independent_affine_plane_oracle_discriminates_actual_proposals():
    report=json.loads(Path(os.environ['AUTOMATIC_SWEPT_TRACE_FILE']).read_text())
    # Independent vertical plane criterion applies to these captured source
    # faces: equal z at all corners for both endpoints. Each target triangle
    # is P1 and physically affine in time. No midpoint sampling proves safety.
    from examples.general_contact_automatic_domain import build_projected_shell_pair
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    model,_,settings,resolution,_=build_projected_shell_pair(lifecycle=True)
    manager=GeneralS3PatchManager(model,resolution.definition,settings)
    outcomes=[]
    for index in (3,24):
        trial=report['trace'][index]
        all_lowers=[];old_lowers=[];exact_lowers=[]
        for si in manager.law.source_face_indices:
            for ti in manager.law.target_face_indices:
                arrays=[np.asarray(trial[k])[idx] for k,idx in (
                    ('source_start',si),('source_end',si),('target_start',ti),('target_end',ti))]
                a0,a1,b0,b1=arrays
                assert np.ptp(a0[:,2])<1e-16 and np.ptp(a1[:,2])<1e-16
                exact=min(0.,float(a0[0,2]-np.max(b0[:,2])),float(a1[0,2]-np.max(b1[:,2])))
                bound=q4_s3_whole_pair_swept_lower(*arrays)
                assert bound<=exact+1e-13
                all_lowers.append(bound);old_lowers.append(overlap_local_swept_lower(*arrays,3)[0]);exact_lowers.append(exact)
        outcomes.append(dict(trace_index=index,new_lower=min(all_lowers),old_lower=min(old_lowers),exact_lower=min(exact_lowers)))
    assert outcomes[0]['new_lower']<-5e-5
    assert outcomes[1]['new_lower']>=-5e-5 and outcomes[1]['old_lower']<-5e-5
    Path(os.environ['AUTOMATIC_DOMAIN_RESULT']).write_text(json.dumps(dict(discriminating_oracle=outcomes),indent=2))

@pytest.mark.skipif(os.environ.get('AUTOMATIC_SWEPT_CAPTURE')!='1',reason='bounded old-candidate diagnosis')
def test_capture_unchanged_automatic_lifecycle(monkeypatch):
    from examples.general_contact_automatic_domain import build_projected_shell_pair,solve_shell_pair
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    model,domain,settings,resolution,path=build_projected_shell_pair(lifecycle=True)
    trace=[]
    original=GeneralS3PatchManager._swept_admissible
    from dataclasses import asdict
    report=dict(identity=resolution.identity,domain=domain.to_dict(),settings=asdict(settings),
        maximum_swept_subdivisions=settings.maximum_swept_subdivisions,profile='unchanged-guided-5um',trace=trace)
    output=Path(os.environ['AUTOMATIC_DOMAIN_RESULT'])
    def save():output.write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
    def observed(manager,start,end):
        answer=original(manager,start,end)
        if len(trace)<256:
            first=manager._positions(start);last=manager._positions(end)
            trace.append(dict(admissible=answer,start_u=start,end_u=end,
                source_start=first[0],target_start=first[1],source_end=last[0],target_end=last[1],
                start_lower=manager._gap_lower(first),end_lower=manager._gap_lower(last),
                start_errors=manager._position_errors(start),end_errors=manager._position_errors(end)))
        save();return answer
    monkeypatch.setattr(GeneralS3PatchManager,'_swept_admissible',observed)
    result=solve_shell_pair(model,domain,settings,resolution,path,emit_restart_checkpoint=True)
    report.update(status=result.status,failure_reason=result.failure_reason,phases=result.info.get('force_displacement_history'),
                  accepted_displacements=result.displacements,checkpoint=result.restart_checkpoint)
    save()
    assert trace

@pytest.mark.parametrize('row_scales', [(1.,1.), (2.**-400,2.**400)])
def test_control_projection_retains_independent_nullspace(row_scales):
    from anyfesolver.general_contact.static_path import _control_projection_seed
    # Independently prescribed sums; differences and fifth coordinate are free.
    rows=np.array(((1.,1.,0.,0.,0.),(0.,0.,1.,-1.,0.)))
    start=np.array((2.,-3.,5.,7.,11.))
    target=np.array((3.,4.))
    seed=_control_projection_seed(rows*np.array(row_scales)[:,None],start,
                                  target*np.array(row_scales))
    np.testing.assert_allclose(seed,(4.,-1.,8.,4.,11.),rtol=0.,atol=2e-15)
    assert seed[0]-seed[1]==start[0]-start[1]
    assert seed[2]+seed[3]==start[2]+start[3]


def test_predictor_validation_and_ignored_route_refusal():
    from dataclasses import replace
    from anyfesolver import ContactSolverSettings,solve_static_nonlinear
    from anyfesolver.general_contact.core import validate_general_contact
    from examples.general_contact_automatic_domain import build_projected_shell_pair
    from anyfesolver.general_contact.static_path import _control_projection_seed
    with pytest.raises(ValueError,match='unsupported path_predictor'):
        ContactSolverSettings(path_predictor='silent')
    with pytest.raises(ValueError,match='ill-conditioned'):
        _control_projection_seed(np.array(((1.,0.),(1.,1e-12))),np.zeros(2),np.ones(2))
    model,_,settings,resolution,_=build_projected_shell_pair(lifecycle=True,
        path_predictor='control_projection_dev')
    before=repr(model)
    for mode in ('force','displacement','arc_length','transient'):
        report=validate_general_contact(model,resolution.definition,settings,analysis_control=mode)
        assert not report.valid and 'path_predictor requires' in str(report.issues)
    with pytest.raises(ValueError,match='requires contact_path_program'):
        solve_static_nonlinear(model,contact=resolution.definition,contact_settings=settings)
    assert repr(model)==before


def test_predictor_default_keeps_historical_automatic_identity():
    from examples.general_contact_automatic_domain import build_projected_shell_pair
    from anyfesolver.general_contact.automatic_domain import _settings_contract
    _,_,settings,resolution,_=build_projected_shell_pair(lifecycle=True)
    assert 'path_predictor' not in _settings_contract(settings)
    baseline_path=os.environ.get('AUTOMATIC_PREDICTOR_BASELINE')
    if not baseline_path:pytest.skip('archived predictor baseline AUTOMATIC_PREDICTOR_BASELINE absent: historical identity comparison requires the original baseline witness')
    baseline=json.loads(Path(baseline_path).read_text())
    assert resolution.identity==baseline['identity']


def test_predictor_public_precancel_and_invalid_seed_are_atomic(monkeypatch):
    from dataclasses import replace
    from anyfesolver import CancellationToken,SolveCancelled
    from examples.general_contact_automatic_domain import build_projected_shell_pair,solve_shell_pair
    from anyfesolver.general_contact import static_path
    model,domain,settings,resolution,path=build_projected_shell_pair(lifecycle=True,
        path_predictor='control_projection_dev')
    before=repr(model)
    token=CancellationToken();token.cancel('predictor pre-cancel')
    with pytest.raises(SolveCancelled):
        solve_shell_pair(model,domain,settings,resolution,path,cancellation_token=token)
    assert repr(model)==before
    # An actual through-penetration control seed cannot reach structural evaluation.
    invalid=replace(path,waypoints=((0.,0.),(0.,-.02)),steps_per_segment=1)
    def prohibited(*args,**kwargs):raise AssertionError('unsafe seed reached structural evaluation')
    monkeypatch.setattr(static_path,'_assemble_nonlinear_system',prohibited)
    result=solve_shell_pair(model,domain,settings,resolution,invalid,emit_restart_checkpoint=True)
    assert result.status=='diverged' and result.failure_reason=='contact_path_predictor_gap_exceeded'
    assert not result.steps and np.array_equal(result.displacements,np.zeros_like(result.displacements))
    assert repr(model)==before


def test_predictor_refuses_unexercised_profile_and_friction():
    from dataclasses import replace
    from anyfesolver.general_contact.core import validate_general_contact
    from anyfesolver.general_contact.static_path import solve_static_contact_path
    from examples.general_contact_automatic_domain import build_projected_shell_pair
    model,_,settings,resolution,path=build_projected_shell_pair(lifecycle=True,
        path_predictor='control_projection_dev')
    report=validate_general_contact(model,resolution.definition,
        replace(settings,q4_source_gap_certificate='whole_face'),analysis_control='contact_path')
    assert not report.valid and 'fixed-map' in str(report.issues)
    before=repr(model)
    with pytest.raises(ValueError,
            match='requires S3SlidingFrictionSettings and the refined S3 projected-penalty owner'):
        solve_static_contact_path(model,resolution.definition,settings,path,friction=object())
    assert repr(model)==before
