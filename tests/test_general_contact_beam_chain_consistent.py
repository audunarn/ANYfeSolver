"""Moving nearest-cell interfaces in the opt-in B2 chain tangent."""
from dataclasses import replace
from pathlib import Path
from fractions import Fraction
import json,os,time
from itertools import permutations

import numpy as np
import pytest
from scipy.integrate import quad

from anyfesolver import (BoundaryCondition,ContactSolverSettings,DisplacementControl,
    LoadCase,solve_static_nonlinear,validate_general_contact)
from anyfesolver.general_contact.beam_chain_pair import _BeamChainPairManager,PROFILE_CONSISTENT
from anyfesolver.general_contact.beam_chain_consistent import moving_owner_block
from anyfesolver.general_contact.beam_chain_consistent import _check_partition_knot,_windows,_roots,_nearest_indices,_coefficients,_rational_geometry
from anyfesolver.general_contact.beam_pair import _Segment
from anyfesolver.elements import BeamElement
from test_general_contact_beam_chain_pair import _chains,_closed


def _settings(settings):
    return replace(settings,beam_chain_tangent='consistent_dev',sliding_difference_step=None,
                   line_integration_relative_tolerance=1e-9)


def _v_case():
    model,definition,settings=_chains()
    for node,point in zip(range(1,7),[(-.1,0.,0.),(0.,0.,0.),(.1,0.,0.),
            (-1.,.06,0.),(0.,.03,0.),(1.,.06,0.)]):
        model.mesh.nodes[node].x,model.mesh.nodes[node].y,model.mesh.nodes[node].z=point
    manager=_BeamChainPairManager(model,definition,_settings(settings))
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1,2,3): u[model.mesh.dof_manager.get_node_dofs(node)[:2]]=(.023,.049)
    return model,definition,settings,manager,u


def _energy_oracle(manager,u):
    # Independent clamped closest-point geometry; reference measure stays fixed.
    total=0.
    poses=[[cell.reference+u[cell.dofs] for cell in side] for side in manager.segments]
    for direction in (0,1):
        for cell,source in zip(manager.segments[direction],poses[direction]):
            def density(s):
                point=(1-s)*source[0]+s*source[1]
                distances=[]
                for target in poses[1-direction]:
                    v=target[1]-target[0]
                    t=max(0.,min(1.,np.dot(point-target[0],v)/np.dot(v,v)))
                    distances.append(np.linalg.norm(point-target[0]-t*v))
                return .25*manager.stiffness*min(min(distances)-manager.radius,0.)**2*cell.length
            total+=quad(density,0.,1.,points=np.linspace(.05,.95,19),
                        epsabs=1e-12,epsrel=1e-9,limit=300)[0]
    return total


def test_v_switch_tangent_includes_positive_cross_owner_interface():
    model,_,_,manager,u=_v_case()
    force,K,report=manager.evaluate(u)
    assert manager.profile==PROFILE_CONSISTENT
    assert manager.last_moving_owner_interfaces>0 and manager.last_force_assemblies==1
    np.testing.assert_allclose(K.toarray(),K.toarray().T,atol=1e-9)
    assert report.contact_energy==pytest.approx(_energy_oracle(manager,u),rel=2e-7)
    for chosen in ((1,2,3),(4,),(5,),(6,)):
        direction=np.zeros_like(u)
        for node in chosen: direction[model.mesh.dof_manager.get_node_dofs(node)[0]]=1.
        errors=[]
        for step in (2e-6,5e-7,1e-7):
            plus=manager.evaluate(u+step*direction,tangent=False)[0]
            minus=manager.evaluate(u-step*direction,tangent=False)[0]
            expected=K@direction
            errors.append(np.linalg.norm((plus-minus)/(2*step)-expected)/max(np.linalg.norm(expected),1e-12))
        assert min(errors)<1e-5,errors
    # The branch-bulk Hessian alone must fail this discriminator.
    positions=manager._positions(u)
    cell=manager.segments[0][0]; source=positions[0][0]
    s=-Fraction.from_float(float(source[0,0]))/(Fraction.from_float(float(source[1,0]))-Fraction.from_float(float(source[0,0])))
    dofs,block,nonzero=moving_owner_block(cell,source,manager.segments[1],positions[1],0,1,s,
                                        manager.radius,manager.stiffness)
    assert nonzero and np.linalg.eigvalsh(block).min()>-1e-9
    left_node=model.mesh.dof_manager.get_node_dofs(4)[:3]
    right_node=model.mesh.dof_manager.get_node_dofs(6)[:3]
    assert np.linalg.norm(block[np.ix_(np.searchsorted(dofs,left_node),np.searchsorted(dofs,right_node))])>0.
    direction=np.zeros_like(u)
    for node in (1,2,3): direction[model.mesh.dof_manager.get_node_dofs(node)[0]]=1.
    h=5e-7
    numerical=(_energy_oracle(manager,u+h*direction)-_energy_oracle(manager,u-h*direction))/(2*h)
    assert numerical==pytest.approx(force@direction,rel=1e-5,abs=1e-8)


def test_parallel_chain_legacy_force_and_tangent_agree():
    model,definition,settings=_chains()
    old=_BeamChainPairManager(model,definition,settings)
    new=_BeamChainPairManager(model,definition,_settings(settings))
    u=_closed(model)
    f0,k0,r0=old.evaluate(u); f1,k1,r1=new.evaluate(u)
    np.testing.assert_allclose(f1,f0,rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(k1.toarray(),k0.toarray(),rtol=2e-5,atol=1e-3)
    assert r1.contact_energy==pytest.approx(r0.contact_energy,rel=1e-8)
    assert old.last_force_assemblies>1 and new.last_force_assemblies==1


def test_opt_in_selection_and_legacy_restart_contract_are_explicit():
    model,definition,settings=_chains()
    old=_BeamChainPairManager(model,definition,settings)
    new=_BeamChainPairManager(model,definition,_settings(settings))
    assert 'beam_chain_tangent' not in old.restart_contract()['settings']
    assert new.restart_contract()['settings']['beam_chain_tangent']=='consistent_dev'
    with pytest.raises(ValueError,match='beam_chain_tangent'):
        ContactSolverSettings(beam_chain_tangent='automatic')
    single=replace(definition,surfaces=tuple(replace(s,element_ids=(s.element_ids[0],)) for s in definition.surfaces))
    report=validate_general_contact(model,single,_settings(settings))
    assert not report.valid and any('multi-cell beam pair' in x for x in report.issues)


@pytest.mark.parametrize('shift',(-.1,.1))
def test_unequal_work_owner_tie_at_source_end_rejects(shift):
    model,_,_,manager,u=_v_case()
    for node in (1,2,3): u[model.mesh.dof_manager.get_node_dofs(node)[0]]=shift
    with pytest.raises(ValueError,match='domain-end ownership tie'):
        manager.evaluate(u)


def _vertex_case(flat=False):
    model,definition,settings=_chains()
    points=[(0.,-.04,0.),(0.,-.03,0.),(0.,-.02,0.),
            (-1.,.03 if flat else .06,0.),(0.,.03,0.),(1.,.03 if flat else .06,0.)]
    for node,point in enumerate(points,1):
        model.mesh.nodes[node].x,model.mesh.nodes[node].y,model.mesh.nodes[node].z=point
    manager=_BeamChainPairManager(model,definition,_settings(settings))
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,y in zip((1,2,3),(.011,.012,.013)):
        u[model.mesh.dof_manager.get_node_dofs(node)[1]]=y-model.mesh.nodes[node].y
    return model,manager,u


def test_strict_shared_material_vertex_cone_has_complete_tangent():
    model,manager,u=_vertex_case()
    force,K,report=manager.evaluate(u)
    assert report.contact_energy>0.
    direction=np.zeros_like(u)
    direction[model.mesh.dof_manager.get_node_dofs(2)[0]]=1.
    direction[model.mesh.dof_manager.get_node_dofs(5)[0]]=.2
    errors=[]
    for step in (1e-6,2e-7,5e-8):
        plus=manager.evaluate(u+step*direction,tangent=False)[0]
        minus=manager.evaluate(u-step*direction,tangent=False)[0]
        errors.append(np.linalg.norm((plus-minus)/(2*step)-K@direction)/np.linalg.norm(K@direction))
    assert min(errors)<1e-5,errors
    assert np.linalg.norm(np.asarray(report.beam_a_force)+report.beam_b_force)<1e-8


def test_collinear_material_vertex_projection_plateau_is_not_smooth_admission():
    _,manager,u=_vertex_case(flat=True)
    with pytest.raises(ValueError,match='projection-clamp plateau'):
        manager.evaluate(u)


@pytest.mark.parametrize('reverse_ids',(False,True))
def test_tied_competitor_projection_plateau_rejects_independently_of_ids(reverse_ids):
    model,definition,settings=_chains()
    corners=[(1.,0.,0.),(0.,0.,0.),(0.,-1.,0.)]
    if reverse_ids: corners=corners[::-1]
    for node,point in enumerate([(0.,.06,0.),(0.,.07,0.),(0.,.08,0.),*corners],1):
        model.mesh.nodes[node].x,model.mesh.nodes[node].y,model.mesh.nodes[node].z=point
    manager=_BeamChainPairManager(model,definition,_settings(settings))
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,y in zip((1,2,3),(.008,.009,.010)):
        u[model.mesh.dof_manager.get_node_dofs(node)[1]]=y-model.mesh.nodes[node].y
    with pytest.raises(ValueError,match='projection-clamp plateau'):
        manager.evaluate(u)


def test_positive_projection_panels_below_double_parameter_spacing_are_retained():
    model,definition,settings=_chains()
    manager=_BeamChainPairManager(model,definition,_settings(settings))
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,y in zip((4,5,6),(.019,.018,.017)):
        u[model.mesh.dof_manager.get_node_dofs(node)[1]]=y-model.mesh.nodes[node].y
    positions=manager._positions(u)
    windows=_windows(positions[0][0],positions[1],(3,4),manager.radius)
    assert any(right>left and float(right)==float(left) for _,left,right in windows)
    force,K,report=manager.evaluate(u)
    assert report.contact_energy>0 and np.all(np.isfinite(force))
    assert np.all(np.isfinite(K.data))


@pytest.mark.parametrize('factor',(1,2))
def test_close_roots_are_not_filtered_away_after_80_digit_cancellation(factor):
    epsilon=Fraction(1,10**100)
    coefficients=(Fraction(1),Fraction(-2),1-factor*epsilon**2)
    roots=_roots(coefficients,Fraction(0),Fraction(1))
    assert len(roots)==1
    root=roots[0]
    if factor==1: assert root==1-epsilon
    else:
        residual=(1-root)**2-2*epsilon**2
        assert abs(residual)<Fraction(1,10**75)*epsilon**2


def test_unresolved_irrational_root_domain_end_returns_precision_failure():
    high=Fraction('1.414213562373095048801688724209698078569671875376948073176679737990732478462107038850387534327642')
    with pytest.raises(ValueError,match='enclosure overlaps a nonroot domain end'):
        _roots((Fraction(1),Fraction(0),Fraction(-2)),Fraction(0),high)


def test_active_repeated_owner_root_with_unchanged_owner_rejects():
    model,definition,settings=_chains()
    points=[(-.12,0.,.04),(-.02,0.,.04),(.08,0.,.04),
            (0.,0.,0.),(0.,1.,0.),(-1.,0.,0.),(1.,0.,0.)]
    for node,point in enumerate(points[:6],1):
        model.mesh.nodes[node].x,model.mesh.nodes[node].y,model.mesh.nodes[node].z=point
    model.add_node(7,*points[6])
    model.add_element(5,BeamElement(5,[6,7],'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    definition=replace(definition,surfaces=(definition.surfaces[0],replace(definition.surfaces[1],element_ids=(3,4,5))))
    manager=_BeamChainPairManager(model,definition,_settings(settings))
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1,2,3): u[model.mesh.dof_manager.get_node_dofs(node)[2]]=-.025
    with pytest.raises(ValueError,match='repeated owner root'):
        manager.evaluate(u)


def test_three_unequal_work_features_do_not_get_an_arbitrary_simple_switch():
    source=np.array([[-.1,0.,0.],[.1,0.,0.]])
    cell=_Segment(1,(1,2),source,np.arange(6).reshape(2,3),.2)
    targets=tuple(np.array([[-1.,y,z],[1.,y,z]]) for y,z in ((.015,0.),(-.015,0.),(0.,.015)))
    cells=tuple(_Segment(i+2,(2*i+3,2*i+4),p,np.arange(6+6*i,12+6*i).reshape(2,3),2.)
                for i,p in enumerate(targets))
    with pytest.raises(ValueError,match='triple ownership'):
        _check_partition_knot(cell,source,cells,targets,((0,0.,.5),(1,.5,1.)),.5,.02)


def _coupled_case():
    model,definition,settings=_chains()
    model.add_boundary_condition(BoundaryCondition('a_fixed',(1,2,3),
        {name:0. for name in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('b_root',(4,),
        {name:0. for name in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('b_guide',(5,6),
        {name:0. for name in ('ux','uz','rx','ry','rz')}))
    load=LoadCase('chain drive');load.add_nodal_load(6,forces=(0.,-1.,0.))
    return model,definition,settings,load


def test_coupled_opt_in_force_displacement_active_restart_and_selection_rejection():
    model,definition,settings,load=_coupled_case()
    settings=_settings(settings)
    control=DisplacementControl(node_id=6,dof='uy',target_displacement=-.008)
    common=dict(contact=definition,contact_settings=settings,control='displacement',
        displacement_control=control,max_iterations=15,tolerance=1e-8)
    full=solve_static_nonlinear(model,load,num_steps=4,**common)
    assert full.status=='completed',full.failure_reason
    assert full.info['general_contact']['profile']==PROFILE_CONSISTENT
    assert full.info['general_contact']['gap_tolerance_met']
    model,definition,_,load=_coupled_case()
    half=solve_static_nonlinear(model,load,num_steps=2,max_load_factor=.75,
        emit_restart_checkpoint=True,**common)
    assert half.status=='completed',half.failure_reason
    assert half.info['general_contact']['active_source_intervals']>0
    checkpoint=json.loads(json.dumps(half.to_restart_checkpoint()))
    model,definition,_,load=_coupled_case()
    resumed=solve_static_nonlinear(model,load,num_steps=2,restart_checkpoint=checkpoint,**common)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,full.displacements,rtol=1e-6,atol=1e-8)
    legacy=replace(settings,beam_chain_tangent='finite_difference',sliding_difference_step=1e-7)
    with pytest.raises(ValueError,match='restart|incompatible|contract'):
        solve_static_nonlinear(model,load,num_steps=2,restart_checkpoint=checkpoint,
            **{**common,'contact_settings':legacy})
    model,definition,_,load=_coupled_case()
    forced=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=20,tolerance=1e-8)
    assert forced.status=='completed',forced.failure_reason
    assert forced.info['general_contact']['active_source_intervals']>0


def test_balanced_displacement_route_comparison_preserves_mechanics():
    rows=[]; solutions=[]
    # Fresh models; equal loads, controls, integration and equilibrium tolerances.
    # One import/cache warm-up per selection precedes three balanced pairs.
    for tag in ('finite_difference','consistent_dev'):
        model,definition,settings,load=_coupled_case()
        settings=replace(settings,beam_chain_tangent=tag,line_integration_relative_tolerance=1e-6)
        warm=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
            control='displacement',displacement_control=DisplacementControl(node_id=6,dof='uy',target_displacement=-.008),
            num_steps=4,max_iterations=20,tolerance=1e-8)
        assert warm.status=='completed',warm.failure_reason
    for pair,order in enumerate((('consistent_dev','finite_difference'),
                                 ('finite_difference','consistent_dev'),
                                 ('consistent_dev','finite_difference'))):
        values={}
        for tag in order:
            started=time.perf_counter()
            model,definition,settings,load=_coupled_case()
            settings=replace(settings,beam_chain_tangent=tag,line_integration_relative_tolerance=1e-6)
            result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
                control='displacement',displacement_control=DisplacementControl(node_id=6,dof='uy',target_displacement=-.008),
                num_steps=4,max_iterations=20,tolerance=1e-8)
            elapsed=time.perf_counter()-started
            assert result.status=='completed',result.failure_reason
            values[tag]=result
            rows.append({'pair':pair,'selection':tag,'elapsed_s':elapsed,
                'contact':result.info['general_contact'],'solver_info':result.info})
        old,new=(values[tag] for tag in ('finite_difference','consistent_dev'))
        np.testing.assert_allclose(new.displacements,old.displacements,rtol=1e-6,atol=1e-10)
        np.testing.assert_allclose(new.info['general_contact']['beam_a_force'],
            old.info['general_contact']['beam_a_force'],rtol=1e-6,atol=1e-8)
        solutions.append(float(np.linalg.norm(new.displacements-old.displacements)))
    times={tag:[r['elapsed_s'] for r in rows if r['selection']==tag]
           for tag in ('finite_difference','consistent_dev')}
    median={tag:float(np.median(values)) for tag,values in times.items()}
    evidence={'scope':'guided four-step displacement-controlled B2 two-chain development comparison; not universal speed qualification',
        'times_s':times,'median_s':median,'speed_ratio':median['finite_difference']/median['consistent_dev'],
        'displacement_difference_norm_m':solutions,'rows':rows}
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:
        Path(record).write_text(json.dumps(evidence,indent=2,default=lambda x:np.asarray(x).tolist()),encoding='utf8')


def test_public_consistent_chain_example(capsys):
    from examples.general_contact_round_beam_chain_pair import main
    main(beam_chain_tangent='consistent_dev')
    output=capsys.readouterr().out
    assert PROFILE_CONSISTENT in output
    assert 'checkpoint version:' in output


def test_capability_does_not_admit_development_or_foreign_geometry():
    from anyfesolver import RigidContactBody, report_general_contact_capability
    model,definition,settings=_chains()
    controls=_settings(settings)
    report=report_general_contact_capability(model,definition,controls)
    assert report.implemented and report.validation.profile==PROFILE_CONSISTENT
    assert not report.tested and not report.independently_accepted
    assert not report.released and not report.admitted_for_application
    foreign=replace(definition,rigid_bodies=(RigidContactBody(
        'foreign','plane',(0.,0.,0.),normal=(0.,1.,0.)),))
    invalid=validate_general_contact(model,foreign,controls)
    assert not invalid.valid and any('explicit multi-cell beam pair' in x for x in invalid.issues)


def test_near_projection_cut_does_not_create_a_false_unequal_work_tie():
    source=np.array([[0.,0.,0.],[1.,0.,0.]])
    targets=(np.array([[0.,.001999991,0.],[.5,.00199999,0.]]),
             np.array([[.5,.00199999,0.],[1.,.001999988,0.]]))
    rational,(second,)=_rational_geometry(source,(targets[1],))
    _,raw,slope=_coefficients(rational,second,Fraction(1,2))
    cut=-raw/slope
    # Float distances are too close to classify ownership using a geometric
    # tolerance; exact squared distances prove the first feature is nearest.
    assert _nearest_indices(source,targets,cut,partition=True)==(0,)
    cell=_Segment(1,(1,2),source,np.arange(6).reshape(2,3),1.)
    cells=(_Segment(2,(3,4),targets[0],np.arange(6,12).reshape(2,3),.5),
           _Segment(3,(4,5),targets[1],np.arange(9,15).reshape(2,3),.5))
    windows=_windows(source,targets,(2,3),.002)
    _check_partition_knot(cell,source,cells,targets,windows,cut,.002)


def test_irrational_root_identity_is_invariant_under_rational_scaling():
    polynomial=(Fraction(1),Fraction(0),Fraction(-1,2))
    root=_roots(polynomial,Fraction(0),Fraction(1))
    for scale in (Fraction(-2),Fraction(3),Fraction(1,7)):
        assert _roots(tuple(scale*v for v in polynomial),Fraction(0),Fraction(1))==root


@pytest.mark.parametrize('order',list(permutations(range(3))))
def test_irrational_triple_tie_rejects_all_target_permutations(order):
    source=np.array([[0.,0.,0.],[10.,0.,0.]])
    # Distances squared: 100s²+1, 50s²+25 and 49. Three unequal
    # material work maps meet at s=sqrt(.48); differences are proportional.
    raw=(np.array([[0.,1.,-20.],[0.,1.,20.]]),
         np.array([[-20.,5.,-20.],[20.,5.,20.]]),
         np.array([[-20.,7.,0.],[20.,7.,0.]]))
    targets=tuple(raw[i] for i in order)
    cell=_Segment(1,(1,2),source,np.arange(6).reshape(2,3),10.)
    cells=tuple(_Segment(i+2,(2*i+3,2*i+4),pose,np.arange(6+6*i,12+6*i).reshape(2,3),40.)
                for i,pose in enumerate(targets))
    windows=_windows(source,targets,(2,3,4),8.)
    cut=_roots((Fraction(100),Fraction(0),Fraction(-48)),Fraction(0),Fraction(1))[0]
    assert len(_nearest_indices(source,targets,cut,partition=True))==3
    with pytest.raises(ValueError,match='triple ownership'):
        _check_partition_knot(cell,source,cells,targets,windows,cut,8.)
