"""Bounded saved Newton trial discriminator for connected physical bodies."""
import json, os
from pathlib import Path
import numpy as np
import pytest
from examples.general_contact_native_retained_body_pair import analysis


@pytest.mark.skipif(os.environ.get("CONTACT_NATIVE_BODY_DIAGNOSTIC")!="1",reason="explicit bounded evidence selector")
def test_first_newton_trial_descriptor():
    owner=analysis();state=owner.initial.mechanical;origins=owner.initial.histories
    residual,hessian,metrics,responses,contact=owner._assemble(state,.05,origins)
    step,correction=owner.context.step(state,residual,hessian)
    candidate=owner.context.advance(state,step)
    report={'mechanical':candidate.descriptor(),'initial_metrics':metrics,
            'step_norm':float(np.linalg.norm(step)),'correction':float(correction)}
    from anyfesolver.general_contact.native_retained_pair_static import _gap_limited_fraction
    fraction=1.
    for _ in range(2):
        candidate=owner.context.advance(state,fraction*step)
        report['mechanical']=candidate.descriptor();report['trial_fraction']=fraction
        try:
            result,gap=owner._endpoint_contact(candidate)
            report['physical_gap_lower_m']=gap
            if result is None:
                fraction=_gap_limited_fraction(fraction,contact.conservative_lower_gap_m,gap,owner.gap_tolerance)
                continue
            report['potential_j']=result.potential_j
        except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
        break
    Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(Path(__file__).parent/'temporary-body-report.json'))).write_text(json.dumps(report,indent=2,default=lambda x:x.tolist()))
    assert 'failure' in report, 'discriminator expected existing finite ownership refusal'


@pytest.mark.skipif(os.environ.get("CONTACT_NATIVE_BODY_DIAGNOSTIC")!="1",reason="explicit bounded evidence selector")
def test_saved_trial_junction_stationarity_inventory():
    from anyfesolver.general_contact.native_curved_pair_partition import _exact_curve,_dot
    owner=analysis()
    raw=json.loads(((Path(__file__).parent/'data/general_contact_native_retained_body_trial_v1.json')).read_text())
    pose=owner.context.make(raw['mechanical']);states=owner.contact._states(pose)
    rows=[]
    for chain in owner.contact.chains:
        for cell in chain:
            source=states[cell].positions
            for targets in (owner.contact.chains[1],owner.contact.chains[0]):
                for i in range(len(targets)-1):
                    first=states[targets[i]].positions;last=states[targets[i+1]].positions
                    junction,tl=_exact_curve(first,1.);_,tr=_exact_curve(last,0.)
                    phis=[[float(_dot(tuple(j-p for j,p in zip(junction,_exact_curve(source,s)[0])),t)) for t in (tl,tr)] for s in (.5,1.)]
                    rows.append({'source':[cell.element,cell.cell],'targets':[[targets[i].element,targets[i].cell],[targets[i+1].element,targets[i+1].cell]],'phi_half_end':phis})
    Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(Path(__file__).parent/'temporary-body-report.json'))).write_text(json.dumps(rows,indent=2))


@pytest.mark.skipif(os.environ.get("CONTACT_NATIVE_BODY_DIAGNOSTIC")!="1",reason="explicit bounded evidence selector")
def test_saved_candidate_independent_projection_and_bounds():
    from anyfesolver.general_contact.native_retained_whole_pair import _active_gap_correspondence_upper,_exact_tiny_cell_omission_bound
    from anyfesolver.general_contact.beam_curve_pair import _source_interval_target_gap_lower
    owner=analysis();raw=json.loads((Path(__file__).parent/'data/general_contact_native_retained_body_trial_v1.json').read_text())
    pose=owner.context.make(raw['mechanical']);states=owner.contact._states(pose)
    source=states[owner.contact.chains[0][0]].positions;target=states[owner.contact.chains[1][0]].positions
    coefficients=lambda points:np.array((points[0],-3*points[0]+4*points[1]-points[2],2*points[0]-4*points[1]+2*points[2]))
    a=coefficients(source);b=coefficients(target)
    def independent_gap(s):
        point=a[0]+s*a[1]+s*s*a[2];z=b.copy();z[0]-=point
        stationary=np.zeros(4)
        for j in range(3):stationary=np.polynomial.polynomial.polyadd(stationary,np.polynomial.polynomial.polymul(z[:,j],(b[1,j],2*b[2,j])))
        roots=np.polynomial.polynomial.polyroots(stationary)
        ts=[0.,1.,*(float(t.real) for t in roots if abs(t.imag)<1e-12 and 0<t.real<1)]
        return min(np.linalg.norm(point-b[0]-t*b[1]-t*t*b[2]) for t in ts)-.1
    left=.00189208984375;right=.001953125
    gaps=[independent_gap(s) for s in (left,.5*(left+right),right)]
    all_bounds=[];all_gaps=[];micro=[]
    for target_cell in owner.contact.chains[1]:
        target=states[target_cell].positions;b=coefficients(target)
        all_gaps.append([independent_gap(s) for s in (left,.5*(left+right),right)])
        all_bounds.append(_source_interval_target_gap_lower(source,left,right,target,.1))
    for source_cell in owner.contact.cells:
        source=states[source_cell].positions
        ref=source_cell.geometry.evaluate(source_cell.geometry.reference.coordinates,np.zeros((3,3)),np.tile(np.eye(3),(2,1,1))).positions
        for target_cell in owner.contact.cells:
            if source_cell in owner.contact.chains[0] and target_cell in owner.contact.chains[0]:continue
            if source_cell in owner.contact.chains[1] and target_cell in owner.contact.chains[1]:continue
            value=_exact_tiny_cell_omission_bound(source,states[target_cell].positions,ref,.1,4000.,.44564208840060615,.4456420884006065)
            micro.append({'source':[source_cell.element,source_cell.cell],'target':[target_cell.element,target_cell.cell],'bound':value})
    report={'same_pose_global_gap_lower_m':owner._physical_gap(pose),'independent_all_target_local_gaps_m':all_gaps,
        'all_target_complete_chord_lowers_m':all_bounds,'union_strip_lower_m':min(all_bounds),'micro_bounds':micro}
    Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(Path(__file__).parent/'temporary-body-report.json'))).write_text(json.dumps(report,indent=2,default=lambda x:x.tolist()))
    assert min(map(min,all_gaps))>0
    assert min(all_bounds)>0


@pytest.mark.skipif(os.environ.get("CONTACT_NATIVE_BODY_DIAGNOSTIC")!="1",reason="explicit bounded evidence selector")
def test_saved_trial_complete_contact_after_open_proof():
    owner=analysis();raw=json.loads((Path(__file__).parent/'data/general_contact_native_retained_body_trial_v1.json').read_text())
    pose=owner.context.make(raw['mechanical']);report={'mechanical':raw['mechanical'],'physical_gap_lower_m':owner._physical_gap(pose),'diagnostics':{}}
    try:
        result=owner.contact.evaluate(pose,_diagnostics=report['diagnostics']);report.update(potential_j=result.potential_j,force_a=result.beam_a_force_n,force_b=result.beam_b_force_n,moment_a=result.beam_a_moment_nm,moment_b=result.beam_b_moment_nm,native_error=(result.native_residual_component_error,result.native_tangent_component_error,result.native_spatial_tangent_component_error))
    except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
    Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(Path(__file__).parent/'temporary-body-report.json'))).write_text(json.dumps(report,indent=2,default=lambda x:x.tolist()))
    assert 'failure' not in report,report.get('failure')


@pytest.mark.skipif(os.environ.get('CONTACT_NATIVE_BODY_DIAGNOSTIC')!='1',reason='explicit bounded evidence selector')
def test_saved_front_exact_gap_resolution():
    from anyfesolver.general_contact.native_curved_pair_partition import _exact_curve,_dot,_convexity_lower
    from anyfesolver.general_contact.beam_curve_pair import b3_distance_jet,closest_on_b3,_curve,_bezier_controls
    from fractions import Fraction
    owner=analysis();raw=json.loads((Path(__file__).parent/'data/general_contact_native_retained_body_trial_v1.json').read_text())
    pose=owner.context.make(raw['mechanical']);states=owner.contact._states(pose)
    source=states[owner.contact.chains[0][0]].positions;target=states[owner.contact.chains[1][0]].positions
    rows=[]
    for s in (.4456420884005717,.44564208840057173,.44564208840060615,.4456420884006065):
        projection=closest_on_b3(_curve(source,s)[0],target);t=projection.minimizers[0]
        p,_=_exact_curve(source,s);r,v=_exact_curve(target,t);offset=tuple(x-y for x,y in zip(p,r))
        square=_dot(offset,offset);delta=square-Fraction(.1)**2
        phi=_dot(tuple(-x for x in offset),v)
        convexity=_convexity_lower(_bezier_controls(source,0.,.5),target,1e-12)
        robust=float(delta)/(projection.distance+.1)
        surplus=float(phi*phi)/(convexity*projection.distance)
        rows.append(dict(station=s,float_gap=b3_distance_jet(source,target,s,.1,source_is_a=True).gap,robust_feasible_gap=robust,stationarity_distance_surplus=surplus))
    Path(os.environ['CONTACT_APP_RESULT_RECORD']).write_text(json.dumps(rows,indent=2))
    assert all(row['stationarity_distance_surplus']<1e-27 for row in rows)
    from anyfesolver.general_contact.native_retained_whole_pair import _private_interval_open
    assert all(_private_interval_open(source,states[cell].positions,.4456420884005717,.44564208840057173,.1) for cell in owner.contact.chains[1][:2])


@pytest.mark.skipif(os.environ.get('CONTACT_NATIVE_BODY_DIAGNOSTIC')!='1',reason='explicit bounded evidence selector')
def test_factored_gap_independent_straight_law():
    from anyfesolver.general_contact.native_retained_whole_pair import _private_factored_gap
    source=np.array(((0.,0.,0.),(.5,0.,0.),(1.,0.,0.)))
    for delta in (-2e-5,-1e-10,1e-10,2e-5):
        target=source.copy();target[:,1]=.1+delta
        lower,upper,value=_private_factored_gap(source,target,.37,.1)
        expected=float(target[0,1])-.1
        assert lower<=expected<=upper
        assert value==pytest.approx(expected,rel=1e-10,abs=1e-30)


def test_factored_offstation_lower_encloses_true_minimum():
    from anyfesolver.general_contact.native_retained_whole_pair import _private_factored_gap
    source=np.array(((0.,0.,0.),(.5,0.,0.),(1.,0.,0.)))
    target=source.copy();target[:,1]=.09998
    lower,upper,_=_private_factored_gap(source,target,.37,.1,target_station=.3701)
    true_gap=.09998-.1
    assert lower<=true_gap<=upper


@pytest.mark.skipif(os.environ.get('CONTACT_NATIVE_BODY_DIAGNOSTIC')!='1',reason='explicit bounded evidence selector')
def test_saved_complete_energy_work_and_directional_hessian():
    from scipy.integrate import quad
    curved=os.environ.get('CONTACT_NATIVE_BODY_CURVED_WORK')=='1'
    if curved:
        from dataclasses import replace
        owner=analysis(separation=.101,reference_rise_m=.001)
        positions=owner.initial.mechanical.positions.copy();positions[5:,1]-=.00102
        pose=replace(owner.initial.mechanical,positions=positions);ctx=owner.context
    else:
        owner=analysis();raw=json.loads((Path(__file__).parent/'data/general_contact_native_retained_body_trial_v1.json').read_text())
        pose=owner.context.make(raw['mechanical']);ctx=owner.context
    def scalar(mechanical):
        # Independent material Q2 geometry and stationary polynomial projection.
        # No production curve/projection/partition is used by this scalar oracle.
        curves=[]
        for element in ctx.elements:
            index=ctx.elements.index(element);ref=element[1].operator.reference
            nodes=ctx.nodes[index];positions=mechanical.positions[nodes]+mechanical.position_low[nodes]
            for cell in (0,1):
                lift=mechanical.cell_rotations[index,cell]@ref.half_cell_lift(cell,.5)
                points=np.array((positions[cell],.5*(positions[cell]+positions[cell+1])+lift,positions[cell+1]))
                curves.append(np.array((points[0],-3*points[0]+4*points[1]-points[2],2*points[0]-4*points[1]+2*points[2])))
        def distance(point,b):
            z=b.copy();z[0]-=point;poly=np.zeros(4)
            for axis in range(3):poly=np.polynomial.polynomial.polyadd(poly,np.polynomial.polynomial.polymul(z[:,axis],(b[1,axis],2*b[2,axis])))
            roots=np.polynomial.polynomial.polyroots(poly)
            stations=(0.,1.,*(float(t.real) for t in roots if abs(t.imag)<1e-9 and 0<t.real<1))
            return min(float(np.linalg.norm(point-b[0]-t*b[1]-t*t*b[2])) for t in stations)
        value=0.
        for side in (0,1):
            for source_cell,curve in enumerate(curves[4*side:4*side+4]):
                def density(s):
                    point=curve[0]+s*curve[1]+s*s*curve[2]
                    gap=min(distance(point,target) for target in curves[4*(1-side):4*(1-side)+4])-.1
                    reference_points=ctx.elements[(4*side+source_cell)//2][1].operator.reference.coordinates
                    cell=source_cell%2
                    # Independent reference Q2 derivative restricted to half cell.
                    coeff=np.array((reference_points[0],-3*reference_points[0]+4*reference_points[1]-reference_points[2],2*reference_points[0]-4*reference_points[1]+2*reference_points[2]))
                    reference_t=.5*(cell+s)
                    measure=.5*np.linalg.norm(coeff[1]+2*reference_t*coeff[2])
                    return .25*4000.*measure*min(0.,gap)**2
                value+=quad(density,0.,1.,epsabs=1e-19,epsrel=1e-10,limit=200,points=np.linspace(0,1,21))[0]
        return value
    diagnostics={}
    try:
        result=owner.contact.evaluate(pose,_diagnostics=diagnostics)
    except Exception as exc:
        report=dict(curved=curved,mechanical=pose.descriptor(),diagnostics=diagnostics,
                    failure=type(exc).__name__+': '+str(exc))
        Path(os.environ['CONTACT_APP_RESULT_RECORD']).write_text(json.dumps(report,indent=2,default=lambda x:x.tolist()))
        raise
    direction=np.zeros(ctx.count);direction[ctx.nodal_count+2 if curved else 7]=1.;h=float(os.environ.get("CONTACT_NATIVE_BODY_WORK_STEP","1e-7"))
    if curved:
        assert np.linalg.norm(owner.contact.cells[0].geometry.reference.half_cell_lift(0,.5))>1e-6
        assert abs(result.residual@direction)>1e-9
    e0=scalar(pose);ep=scalar(ctx.advance(pose,h*direction));em=scalar(ctx.advance(pose,-h*direction))
    work=(ep-em)/(2*h);tangent=(ep+em-2*e0)/(h*h)
    report=dict(potential_j=result.potential_j,independent_energy_j=e0,work=work,native_work=float(result.residual@direction),
        hessian=tangent,native_hessian=float(direction@result.tangent@direction),step=h,curved=curved)
    Path(os.environ['CONTACT_APP_RESULT_RECORD']).write_text(json.dumps(report,indent=2))
    assert result.potential_j==pytest.approx(e0,rel=1e-8,abs=1e-15)
    assert result.residual@direction==pytest.approx(work,rel=1e-6,abs=1e-10)
    assert direction@result.tangent@direction==pytest.approx(tangent,rel=1e-5,abs=1e-6)
