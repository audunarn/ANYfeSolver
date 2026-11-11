"""Declared-domain precedence/role/refusal and actual ordinary restart checks."""
from dataclasses import replace,asdict
import json,os
from pathlib import Path
import numpy as np
import pytest

from anyfesolver import FEModel,BoundaryCondition,create_shell_element
from anyfesolver.general_contact import (ContactSurface,ContactInteraction,RigidContactBody,ContactSolverSettings,
    AutomaticContactParticipant as Participant,AutomaticContactDefaults as Defaults,
    AutomaticContactDomain as Domain,AutomaticContactDomainError,
    resolve_automatic_contact_domain as resolve,verify_automatic_contact_resolution as verify)
from examples.general_contact_automatic_domain import build,solve


def shell_pair(*,size_a=2.,size_b=1.):
    model=FEModel('role-reference')
    model.add_material('elastic',1e5,.3,density=1.)
    for face,(size,z) in enumerate(((size_a,.1),(size_b,0.))):
        nodes=[]
        for local,(x,y) in enumerate(((0.,0.),(size,0.),(size,size),(0.,size))):
            node=4*face+local+1;nodes.append(node);model.add_node(node,x,y,z)
        model.add_element(face+1,create_shell_element(face+1,nodes,'elastic',thickness=.01))
    model.add_boundary_condition(BoundaryCondition('guides',tuple(range(1,9)),
        {name:0. for name in ('ux','uy','rx','ry','rz')}))
    domain=Domain('two-shells',(
        Participant(ContactSurface('a',(1,),side='bottom'),'body-a'),
        Participant(ContactSurface('b',(2,),side='top'),'body-b')),Defaults(1e5,.001))
    return model,domain,ContactSolverSettings()


def test_complete_pairing_precedence_permutation_and_identity():
    model,_,domain,settings,baseline=build()
    assert len(baseline.definition.interactions)==2
    assert [r.disposition for r in baseline.selections].count('outside_family')==1
    assert all(pair.surface=='skin' for pair in baseline.definition.interactions)
    assert all(baseline.identity in pair.name for pair in baseline.definition.interactions)
    reversed_domain=replace(domain,participants=tuple(reversed(domain.participants)))
    assert resolve(model,reversed_domain,settings)==baseline
    explicit=ContactInteraction('user-name','skin','floor',2e6,2e-6)
    overridden=resolve(model,replace(domain,pair_overrides=(explicit,)),settings,explicit_pairs=(explicit,))
    row=next(row for row in overridden.selections if row.target=='floor')
    assert row.original_interaction_name=='user-name'
    pair=next(pair for pair in overridden.definition.interactions if pair.target=='floor')
    assert pair.normal_stiffness==2e6 and pair.gap_tolerance==2e-6
    excluded=resolve(model,replace(domain,excluded_pairs=(('floor','skin'),),pair_overrides=(explicit,)),settings)
    assert len(excluded.definition.interactions)==1
    assert next(row for row in excluded.selections if row.participants==('floor','skin')).disposition=='excluded'
    assert len(resolve(model,domain,settings).definition.interactions)==2  # complete even initially open/inactive
    with pytest.raises(AutomaticContactDomainError,match='conflict'):
        resolve(model,replace(domain,pair_overrides=(explicit,replace(explicit,normal_stiffness=3e6))),settings)
    # Exclusion suppresses even conflicting mechanical overrides, whose full
    # input provenance remains hashed; unknown declarations still refuse.
    suppressed=replace(domain,excluded_pairs=(('floor','skin'),),pair_overrides=(explicit,replace(explicit,normal_stiffness=3e6)))
    assert len(resolve(model,suppressed,settings).definition.interactions)==1
    with pytest.raises(AutomaticContactDomainError,match='no pairs dropped'):
        resolve(model,replace(domain,maximum_pairs=1),settings)
    for changed in (replace(domain,name='changed'),replace(domain,defaults=Defaults(2e6,1e-6))):
        assert resolve(model,changed,settings).identity!=baseline.identity
    assert resolve(model,domain,replace(settings,maximum_candidate_pairs=200)).identity!=baseline.identity


def test_coarser_median_reference_role_and_stable_tie():
    model,domain,settings=shell_pair()
    result=resolve(model,domain,settings)
    row=result.selections[0]
    assert (row.source,row.target)==('a','b')
    assert row.source_reference_median_area_m2==pytest.approx(4.)
    assert row.target_reference_median_area_m2==pytest.approx(1.)
    # Explicit orientation is never reversed to recover an executable profile.
    explicit=ContactInteraction('explicit','b','a',1e5,.001)
    result=resolve(model,domain,settings,explicit_pairs=(explicit,))
    assert result.definition.interactions[0].surface=='b'
    model,domain,settings=shell_pair(size_a=1.,size_b=1.)
    result=resolve(model,replace(domain,participants=tuple(reversed(domain.participants))),settings)
    assert result.definition.interactions[0].surface=='a'


def test_q4_bilinear_area_is_not_diagonal_area():
    from anyfesolver.general_contact.automatic_domain import _face_area
    corners=np.array(((0.,0.,0.),(1.,0.,0.),(1.,1.,.3),(0.,1.,0.)))
    # Independent graph z=.3*x*y on unit square, high-order 1D quadrature product.
    nodes,weights=np.polynomial.legendre.leggauss(40);x=(nodes+1)/2
    expected=sum(wx*wy*np.sqrt(1+.09*(a*a+b*b)) for a,wx in zip(x,weights/2) for b,wy in zip(x,weights/2))
    diagonal=.5*np.linalg.norm(np.cross(corners[1]-corners[0],corners[2]-corners[0]))+.5*np.linalg.norm(np.cross(corners[2]-corners[0],corners[3]-corners[0]))
    assert _face_area(corners)==pytest.approx(expected,rel=1e-12)
    assert abs(expected-diagonal)>1e-3


def test_exact_equal_q4_s3_area_role_and_rigid_transform():
    from anyfesolver.general_contact.automatic_domain import _face_area,_coarser_source
    q4=np.array(((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)))
    s3=np.array(((0.,0.,0.),(2.,0.,0.),(0.,1.,0.)))
    rotation=np.array(((0.,-1.,0.),(1.,0.,0.),(0.,0.,1.)))
    for r,shift in ((np.eye(3),np.zeros(3)),(rotation,np.array((7.,-4.,2.)))):
        areas={'a-s3':_face_area(s3@r.T+shift),'z-q4':_face_area(q4@r.T+shift)}
        assert areas['a-s3']==areas['z-q4']==1.
        assert _coarser_source(('z-q4','a-s3'),areas)=='a-s3'


def test_reference_fold_nonfinite_and_degenerate_refusal():
    from anyfesolver.general_contact.automatic_domain import _face_area
    for corners in (
        ((0.,0.,0.),(1.,1.,0.),(1.,0.,0.),(0.,1.,0.)),
        ((0.,0.,0.),(1.,0.,0.),(2.,0.,0.),(3.,0.,0.)),
        ((0.,0.,0.),(1.,0.,0.),(1.,float('nan'),0.),(0.,1.,0.)),
        ((0.,0.,0.),(1.,0.,0.),(2.,0.,0.))):
        with pytest.raises(AutomaticContactDomainError):_face_area(corners)


def test_s3_formulation_owned_selected_skin_offset():
    model=FEModel('s3-reference-offset');model.add_material('elastic',1e5,.3,density=1.)
    for node,point in enumerate(((0.,0.,.035),(1.,0.,.035),(0.,1.,.035)),1):model.add_node(node,*point)
    model.add_element(1,create_shell_element(1,(1,2,3),'elastic',formulation='e4-pl-s3-v2d',thickness=.05,
                          reference_normal=(0.,0.,1.),reference_surface_offset=.01))
    domain=Domain('s3-floor',(
        Participant(ContactSurface('selected',(1,),side='bottom'),'s3-body'),
        Participant(RigidContactBody('floor','plane',(0.,0.,0.),(0.,0.,1.)),'tool-body')),Defaults(1e6,1e-6))
    result=resolve(model,domain)
    row=next(r for r in json.loads(result.reference_json) if r['name']=='selected')
    assert row['faces'][0]['reference_positions_m'][0][2]==pytest.approx(.035-.025-.01)
    assert row['faces'][0]['reference_area_m2']==pytest.approx(.5)


def test_physical_skin_binding_median_and_persistence():
    from anyfesolver.general_contact.automatic_domain import _reference,_coarser_source
    model=FEModel('median-not-total');model.add_material('elastic',1e5,.3,density=1.)
    participants=[];eid=0
    for name,count,area,z in (('many-fine',2,.5,.1),('one-coarse',1,.75,0.)):
        ids=[]
        for face in range(count):
            eid+=1;side=np.sqrt(area);nodes=[]
            for local,(x,y) in enumerate(((0.,0.),(side,0.),(side,side),(0.,side))):
                node=4*(eid-1)+local+1;nodes.append(node);model.add_node(node,x+face,y,z)
            model.add_element(eid,create_shell_element(eid,nodes,'elastic',thickness=.05));ids.append(eid)
        participants.append(Participant(ContactSurface(name,tuple(ids),side='bottom'),name+'-body'))
    rows,areas=_reference(model,tuple(participants))
    assert _coarser_source(tuple(areas),areas)=='one-coarse'
    assert sum(face['reference_area_m2'] for face in rows[0]['faces'])>sum(face['reference_area_m2'] for face in rows[1]['faces'])
    assert rows[0]['faces'][0]['reference_positions_m'][0][2]==pytest.approx(.075)
    model,_,domain,settings,result=build()
    assert Domain.from_dict(json.loads(json.dumps(domain.to_dict())))==domain
    malformed=domain.to_dict();malformed['unknown']=1
    with pytest.raises(AutomaticContactDomainError):Domain.from_dict(malformed)
    explicit=tuple(ContactInteraction('original-'+target,'skin',target,2e6,2e-6) for target in ('floor','second-floor'))
    all_explicit=replace(domain,pair_overrides=explicit)
    baseline=resolve(model,all_explicit,settings)
    changed=resolve(model,replace(all_explicit,defaults=Defaults(3e6,2e-6)),settings)
    assert baseline.identity!=changed.identity
    assert [asdict(p)|{'name':''} for p in baseline.definition.interactions]==[asdict(p)|{'name':''} for p in changed.definition.interactions]
    with pytest.raises(AutomaticContactDomainError):verify(model,replace(all_explicit,pair_overrides=()),baseline,settings)
    changed_definition=replace(result.definition,interactions=(replace(result.definition.interactions[0],normal_stiffness=2e6),*result.definition.interactions[1:]))
    poisoned=replace(result,definition=changed_definition,definition_json=json.dumps(asdict(changed_definition),sort_keys=True,separators=(',',':')))
    with pytest.raises(AutomaticContactDomainError):verify(model,domain,poisoned,settings)
    poisoned=replace(result,selections=tuple(replace(row,reason='changed') for row in result.selections))
    with pytest.raises(AutomaticContactDomainError):verify(model,domain,poisoned,settings)
    with pytest.raises(AutomaticContactDomainError):resolve(model,replace(domain,excluded_pairs=(('floor','second-floor'),)),settings)


def test_refusals_and_fixed_reference_binding():
    model,_,domain,settings,result=build()
    assert verify(model,domain,result,settings)==result.definition
    with pytest.raises(AutomaticContactDomainError):
        replace(domain,excluded_pairs=(('unknown','skin'),))
    with pytest.raises(AutomaticContactDomainError):
        replace(domain,participants=domain.participants+(domain.participants[0],))
    with pytest.raises(AutomaticContactDomainError):
        resolve(model,domain,settings,explicit_pairs=(ContactInteraction('bad','floor','skin',1e5,.001),))
    with pytest.raises(AutomaticContactDomainError):
        resolve(model,domain,settings,explicit_pairs=(ContactInteraction('bad','skin','unknown',1e5,.001),))
    with pytest.raises(AutomaticContactDomainError):
        verify(model,replace(domain,defaults=Defaults(2e6,1e-6)),result,settings)
    with pytest.raises(AutomaticContactDomainError):
        verify(model,domain,result,replace(settings,maximum_candidate_pairs=200))
    model.mesh.nodes[3].x+=.01
    with pytest.raises(AutomaticContactDomainError,match='changed'):
        verify(model,domain,result,settings)
    model,domain,settings=shell_pair()
    same=replace(domain,participants=tuple(replace(p,body_id='one-body') for p in domain.participants))
    with pytest.raises(AutomaticContactDomainError,match='self-contact'):
        resolve(model,same,settings)
    with pytest.raises(AutomaticContactDomainError,match='no executable'):
        resolve(model,replace(same,excluded_pairs=(('a','b'),)),settings)
    beam=replace(domain,participants=(replace(domain.participants[0],participant=ContactSurface('a',(1,),beam_radius=.01)),domain.participants[1]))
    with pytest.raises(AutomaticContactDomainError,match='beam/native'):
        resolve(model,beam,settings)


def test_actual_nonlinear_static_restart_and_provenance_refusal():
    model,load,domain,settings,resolution=build()
    half=solve(model,load,domain,settings,resolution,max_load_factor=.5,emit_restart_checkpoint=True)
    assert half.status=='completed',half.failure_reason
    capsule=half.to_restart_checkpoint()
    resumed=solve(model,load,domain,settings,resolution,restart_checkpoint=capsule,emit_restart_checkpoint=True)
    full=solve(model,load,domain,settings,resolution,emit_restart_checkpoint=True)
    report=dict(identity=resolution.identity,half_status=half.status,resumed_status=resumed.status,full_status=full.status,
                contact=resumed.info.get('general_contact'),checkpoint=capsule)
    mapping={row.compiled_interaction_name:row.target for row in resolution.selections if row.disposition=='resolved'}
    pair_errors={}
    for pair in resumed.info['general_contact']['pairs']:
        target=mapping[pair['interaction']];contact=pair['contact']
        force=.25 if target=='floor' else .15
        expected_force=np.array((0.,0.,force));expected_moment=np.array((.5*force,-.5*force,0.))
        errors=dict(shell_force=float(np.linalg.norm(np.array(contact['shell_force'])-expected_force)/.4),
                    rigid_force=float(np.linalg.norm(np.array(contact['rigid_force'])+expected_force)/.4),
                    shell_moment=float(np.linalg.norm(np.array(contact['shell_moment'])-expected_moment)/.4),
                    rigid_moment=float(np.linalg.norm(np.array(contact['rigid_moment'])+expected_moment)/.4))
        pair_errors[target]=dict(compiled_name=pair['interaction'],expected_force_n=expected_force,
                                expected_moment_nm=expected_moment,normalized_errors=errors,
                                actual={name:contact[name] for name in ('shell_force','rigid_force','shell_moment','rigid_moment')})
    report['pair_resultants_about_global_origin']=pair_errors
    if os.environ.get('AUTOMATIC_DOMAIN_RESULT'):
        Path(os.environ['AUTOMATIC_DOMAIN_RESULT']).write_text(json.dumps(report,indent=2,default=lambda x:x.tolist()),encoding='utf-8')
    assert resumed.status==full.status=='completed'
    assert set(pair_errors)=={'floor','second-floor'}
    assert max(error for row in pair_errors.values() for error in row['normalized_errors'].values())<=1e-6
    assert resumed.info['general_contact']['penalty_energy']>0
    assert resumed.info['general_contact']['gap_tolerance_met']
    # Two distinct parallel tools separated by1e-7m: their pressure difference
    # k*delta=0.1N; total prescribed force0.4N gives resultants0.25N/0.15N.
    assert resumed.info['general_contact']['shell_force'][2]==pytest.approx(.4,abs=1e-8)
    expected_gap=-(.4+1e6*1e-7)/(2e6)
    assert resumed.info['general_contact']['minimum_gap']==pytest.approx(expected_gap,abs=1e-10)
    np.testing.assert_allclose(resumed.displacements,full.displacements,rtol=1e-8,atol=1e-10)
    changed=replace(domain,name='new-provenance')
    changed_resolution=resolve(model,changed,settings)
    with pytest.raises(ValueError,match='incompatible'):
        solve(model,load,changed,settings,changed_resolution,restart_checkpoint=capsule)


def test_oblique_rigid_component_persistence_is_bit_exact():
    model,_,domain,settings,_=build()
    for kind in ('plane','cylinder'):
        body=RigidContactBody('oblique',kind,(0.,0.,-.025),(1.,1.,0.),radius=.2 if kind=='cylinder' else None)
        selected=replace(domain,participants=(domain.participants[-1],Participant(body,'oblique-body')))
        stored=selected.to_dict()
        restored=Domain.from_dict(json.loads(json.dumps(stored)))
        assert restored.to_dict()==stored
        assert restored==selected
        if kind=='plane':
            resolution=resolve(model,restored,settings)
            assert resolution.identity==resolve(model,selected,settings).identity
            assert verify(model,restored,resolution,settings)==resolution.definition
        for invalid in ((2.,2.,0.),(float('nan'),1.,0.),('1',0.,0.),(1.,0.)):
            malformed=json.loads(json.dumps(stored))
            rigid=next(row for row in malformed['participants'] if row['kind']=='rigid')
            rigid['participant']['normal']=invalid
            with pytest.raises(AutomaticContactDomainError):Domain.from_dict(malformed)


@pytest.mark.skipif(os.environ.get('ANY_RUN_AUTOMATIC_SHELL_PAIR')!='1',
                   reason='bounded ordinary Q4/S3 automatic-domain integration')
def test_actual_nonmatching_shell_pair_path_parity_and_restart():
    from anyfesolver import ContactDefinition,solve_static_nonlinear
    from examples.general_contact_automatic_domain import build_shell_pair,solve_shell_pair
    model,domain,settings,resolution,path=build_shell_pair()
    selected=next(row for row in resolution.selections if row.disposition=='resolved')
    assert (selected.source,selected.target)==('q4_source','s3_target')
    assert selected.source_reference_median_area_m2==pytest.approx(1.)
    assert selected.target_reference_median_area_m2==pytest.approx(.5)
    report=dict(identity=resolution.identity,target_scope='one free S3 uz; remaining target coordinates guided',
                selection=asdict(selected),settings=asdict(settings),stages={},
                criteria=dict(force_scale_n=.02,moment_scale_nm=.02,energy_scale_j=2e-7,
                    scale_basis='k1000 * Q4 area1 * nominal 20um penetration; moment lever1m',
                    wrench_normalized_tolerance=1e-6,explicit_response_normalized_tolerance=1e-6,
                    displacement_rtol=1e-7,displacement_atol_m=1e-9,gap_budget_m=5e-5))
    def save():
        if os.environ.get('AUTOMATIC_DOMAIN_RESULT'):
            Path(os.environ['AUTOMATIC_DOMAIN_RESULT']).write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
    def record(name,result):
        report['stages'][name]=dict(status=result.status,failure_reason=result.failure_reason,
            contact=result.info.get('general_contact'),phases=result.info.get('force_displacement_history'))
        if result.restart_checkpoint is not None:report['stages'][name]['checkpoint']=result.to_restart_checkpoint()
        save()
        assert result.status=='completed',result.failure_reason
    import anyfesolver,numba
    from threadpoolctl import threadpool_info
    report['runtime_before_solve']=dict(anyfesolver_path=anyfesolver.__file__,
        automatic_domain_path=__import__('inspect').getfile(resolve),
        jit_disabled=bool(numba.config.DISABLE_JIT),numba_threads=int(numba.get_num_threads()),
        pools=threadpool_info())
    save()
    full=solve_shell_pair(model,domain,settings,resolution,path,emit_restart_checkpoint=True)
    record('automatic_full',full)
    phases=full.info['force_displacement_history']
    assert len(phases)==3
    assert [row['contact']['active_pressure_dofs']>0 for row in phases]==[True,False,True]
    wrench_errors=[]
    for row in phases:
        contact=row['contact'];force=np.array(contact['integration_force']);target=np.array(contact['target_force'])
        moment=np.array(contact['integration_moment']);target_moment=np.array(contact['target_moment'])
        wrench_errors.append(dict(force=float(np.linalg.norm(force+target)/.02),
                                  moment=float(np.linalg.norm(moment+target_moment)/.02)))
        assert contact['wrench_origin_m']==[0.,0.,0.]
        assert contact['gap_tolerance_met']
        assert contact['minimum_gap']>=-5e-5
    report['phase_wrench_errors']=wrench_errors
    target_dof=model.mesh.dof_manager.get_node_dofs(7)[2]
    report['target_node7_uz_m']=float(full.displacements[target_dof]);save()
    assert abs(full.displacements[target_dof])>1e-10
    assert max(error for row in wrench_errors for error in row.values())<=1e-6
    assert phases[1]['contact']['penalty_energy']==0.
    assert all(phases[i]['contact']['penalty_energy']>0 for i in (0,2))
    original=ContactInteraction('ordinary-explicit','q4_source','s3_target',1000.,5e-5)
    explicit_model,explicit_domain,explicit_settings,explicit_resolution,explicit_path=build_shell_pair()
    assert explicit_resolution.identity==resolution.identity
    explicit=ContactDefinition(explicit_resolution.definition.surfaces,(),(original,))
    verify(explicit_model,explicit_domain,explicit_resolution,explicit_settings)
    plain=solve_static_nonlinear(explicit_model,contact=explicit,contact_settings=explicit_settings,
        contact_path_program=explicit_path,max_iterations=15,emit_restart_checkpoint=True)
    record('explicit_full',plain)
    np.testing.assert_allclose(full.displacements,plain.displacements,rtol=1e-7,atol=1e-9)
    parity_errors={}
    for key in ('integration_force','target_force','integration_moment','target_moment','minimum_gap','penalty_energy'):
        np.testing.assert_allclose(full.info['general_contact'][key],plain.info['general_contact'][key],rtol=1e-6,atol=1e-9)
        scale=2e-7 if key=='penalty_energy' else (5e-5 if key=='minimum_gap' else .02)
        parity_errors[key]=float(np.linalg.norm(np.asarray(full.info['general_contact'][key])-plain.info['general_contact'][key])/scale)
    report['explicit_normalized_response_errors']=parity_errors;save()
    assert max(parity_errors.values())<=1e-6
    prefix_model,prefix_domain,prefix_settings,prefix_resolution,prefix_path=build_shell_pair()
    assert prefix_resolution.identity==resolution.identity
    half=solve_shell_pair(prefix_model,prefix_domain,prefix_settings,prefix_resolution,prefix_path,
        max_load_factor=2/3,emit_restart_checkpoint=True)
    record('automatic_open_prefix',half)
    assert half.info['general_contact']['active_pressure_dofs']==0
    restart_model,restart_domain,restart_settings,restart_resolution,restart_path=build_shell_pair()
    assert restart_resolution.identity==resolution.identity
    resumed=solve_shell_pair(restart_model,restart_domain,restart_settings,restart_resolution,restart_path,
        restart_checkpoint=half.to_restart_checkpoint(),emit_restart_checkpoint=True)
    record('automatic_resumed',resumed)
    np.testing.assert_allclose(full.displacements,resumed.displacements,rtol=1e-7,atol=1e-9)
    changed_model,base_domain,changed_settings,_,changed_path=build_shell_pair()
    changed=replace(base_domain,name='changed-shell-provenance')
    changed_resolution=resolve(changed_model,changed,changed_settings,analysis_control='displacement')
    with pytest.raises(ValueError,match='incompatible'):
        solve_shell_pair(changed_model,changed,changed_settings,changed_resolution,changed_path,
                         restart_checkpoint=half.to_restart_checkpoint())
    report['explicit_parity']=True;report['split_restart_parity']=True;report['provenance_refusal']=True;save()


@pytest.mark.skipif(os.environ.get('ANY_RUN_AUTOMATIC_PROJECTED_SHELL_PAIR')!='1',
                   reason='bounded refined projected Q4/S3 automatic-domain press')
def test_actual_projected_shell_pair_press_and_checkpoint(monkeypatch):
    from examples.general_contact_automatic_domain import build_projected_shell_pair,solve_shell_pair
    import anyfesolver,numba
    from threadpoolctl import threadpool_info
    model,domain,settings,resolution,path=build_projected_shell_pair()
    report=dict(fixture='refined_projected_penalty_separate_from_single_Q4_penalty',
        identity=resolution.identity,selections=[asdict(x) for x in resolution.selections],
        domain=domain.to_dict(),settings=asdict(settings),
        criteria=dict(force_scale_n=.02,moment_scale_nm=.02,energy_scale_j=2e-7,
            scale_basis='k1000 * total Q4 area1 * nominal20um penetration; lever1m',
            wrench_normalized_tolerance=1e-6,gap_budget_m=5e-5),
        runtime_before_solve=dict(anyfesolver_path=anyfesolver.__file__,
            automatic_domain_path=__import__('inspect').getfile(resolve),
            jit_disabled=bool(numba.config.DISABLE_JIT),numba_threads=int(numba.get_num_threads()),
            pools=threadpool_info()))
    def save():
        if os.environ.get('AUTOMATIC_DOMAIN_RESULT'):
            Path(os.environ['AUTOMATIC_DOMAIN_RESULT']).write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
    save()
    if os.environ.get('AUTOMATIC_WORKFLOW_STAGE','').startswith('press-trace'):
        from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
        original=GeneralS3PatchManager._swept_admissible
        trace=[];report['proposal_trace']=trace
        def observed(manager,start,end):
            answer=original(manager,start,end)
            if len(trace)<24:
                trace.append(dict(admissible=answer,
                    start_lower=manager._gap_lower(manager._positions(start)),
                    end_lower=manager._gap_lower(manager._positions(end)),
                    source_uz=[float(end[model.mesh.dof_manager.get_node_dofs(n)[2]]) for n in range(1,7)],
                    target_uz=float(end[model.mesh.dof_manager.get_node_dofs(9)[2]])))
            save()
            return answer
        monkeypatch.setattr(GeneralS3PatchManager,'_swept_admissible',observed)
    result=solve_shell_pair(model,domain,settings,resolution,path,emit_restart_checkpoint=True)
    report.update(status=result.status,failure_reason=result.failure_reason,
        contact=result.info.get('general_contact'),phases=result.info.get('force_displacement_history'))
    if result.restart_checkpoint is not None:report['checkpoint']=result.to_restart_checkpoint()
    save()
    assert result.status=='completed',result.failure_reason
    if settings.path_predictor!='none':
        assert result.info['contact_path']['path_predictor']==settings.path_predictor
    contact=result.info['general_contact']
    force_error=float(np.linalg.norm(np.array(contact['integration_force'])+contact['target_force'])/.02)
    moment_error=float(np.linalg.norm(np.array(contact['integration_moment'])+contact['target_moment'])/.02)
    target_dof=model.mesh.dof_manager.get_node_dofs(9)[2]
    report.update(force_balance_error=force_error,moment_balance_error=moment_error,
        target_node9_uz_m=float(result.displacements[target_dof]));save()
    assert contact['active_pressure_dofs']>0 and contact['regularization_storage_j']>0.
    assert contact['contact_work_convention']=='projected_weak_gap_nonpotential_v1'
    assert contact['gap_tolerance_met'] and contact['minimum_gap']>=-5e-5
    assert contact['wrench_origin_m']==[0.,0.,0.]
    assert max(force_error,moment_error)<=1e-6
    assert abs(result.displacements[target_dof])>1e-10
    assert report['checkpoint']


@pytest.mark.skipif(os.environ.get('ANY_RUN_AUTOMATIC_PROJECTED_WORKFLOW')!='1',
                   reason='bounded installed refined automatic deformable lifecycle')
def test_actual_projected_shell_pair_lifecycle_and_restart():
    """One guarded stage per invocation, sharing only retained JSON evidence."""
    import anyfesolver,numba
    from threadpoolctl import threadpool_info
    from anyfesolver import ContactDefinition,solve_static_nonlinear
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from examples.general_contact_automatic_domain import build_projected_shell_pair,solve_shell_pair
    raw_stage=os.environ.get('AUTOMATIC_WORKFLOW_STAGE','full-press-v3')
    press_only='-press' in raw_stage
    stage=raw_stage.split('-')[0] if press_only else raw_stage
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    output=Path(os.environ['AUTOMATIC_DOMAIN_RESULT'])
    model,domain,settings,resolution,path=build_projected_shell_pair(lifecycle=not press_only,
        path_predictor=os.environ.get('AUTOMATIC_PATH_PREDICTOR','none'))
    report=dict(stage=stage,identity=resolution.identity,selections=[asdict(x) for x in resolution.selections],
        domain=domain.to_dict(),settings=asdict(settings),criteria=dict(
            force_scale_n=.02,moment_scale_nm=.02,virtual_translation_m=1e-6,
            force_moment_normalized_tolerance=1e-6,gap_budget_m=5e-5,
            displacement_parity_rtol=1e-7,displacement_parity_atol_m=1e-9,
            scale_basis='k1000 * area1 * nominal20um penetration; lever1m'),
        runtime_before_solve=dict(anyfesolver_path=anyfesolver.__file__,
            automatic_domain_path=__import__('inspect').getfile(resolve),
            jit_disabled=bool(numba.config.DISABLE_JIT),numba_threads=int(numba.get_num_threads()),
            pools=threadpool_info()))
    def save():
        output.write_text(json.dumps(report,indent=2,
            default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
    save()
    assert report['runtime_before_solve']['jit_disabled']
    assert report['runtime_before_solve']['numba_threads']==1
    assert all(pool['num_threads']==1 for pool in report['runtime_before_solve']['pools'])
    controls=dict(emit_restart_checkpoint=True)
    prefix_file='prefix-press-v2.json' if press_only else 'prefix.json'
    full_file='full-press-v3.json' if press_only else 'full.json'
    if stage=='prefix':controls['max_load_factor']=.5 if press_only else 2/3
    if stage=='resume':
        prefix=json.loads((evidence/prefix_file).read_text(encoding='utf-8'))
        assert prefix['identity']==resolution.identity and prefix['verified']
        controls['restart_checkpoint']=prefix['checkpoint']
    if stage=='refusal':
        prefix=json.loads((evidence/prefix_file).read_text(encoding='utf-8'))
        before=repr(model)
        changed=replace(domain,name='changed-shell-provenance')
        changed_resolution=resolve(model,changed,settings,analysis_control='contact_path' if settings.path_predictor!='none' else 'displacement')
        with pytest.raises(ValueError,match='incompatible'):
            solve_shell_pair(model,changed,settings,changed_resolution,path,
                restart_checkpoint=prefix['checkpoint'])
        assert repr(model)==before
        if settings.path_predictor!='none':
            changed_settings=replace(settings,path_predictor='none')
            changed_resolution=resolve(model,domain,changed_settings,analysis_control='displacement')
            with pytest.raises(ValueError,match='incompatible'):
                solve_shell_pair(model,domain,changed_settings,changed_resolution,path,
                    restart_checkpoint=prefix['checkpoint'])
            assert repr(model)==before
            report['predictor_refusal']=True
        report.update(verified=True,provenance_refusal=True,atomic_refusal=True);save();return
    if stage=='explicit':
        definition=ContactDefinition(resolution.definition.surfaces,(),(
            ContactInteraction('ordinary-explicit','q4_source','s3_target',1000.,5e-5,
                               normal_law='projected_penalty'),))
        result=solve_static_nonlinear(model,contact=definition,contact_settings=settings,
            contact_path_program=path,max_iterations=15,**controls)
    else:result=solve_shell_pair(model,domain,settings,resolution,path,**controls)
    report.update(status=result.status,failure_reason=result.failure_reason,
        contact=result.info.get('general_contact'),phases=result.info.get('force_displacement_history'),
        displacements=result.displacements)
    if result.restart_checkpoint is not None:report['checkpoint']=result.to_restart_checkpoint()
    save()
    assert result.status=='completed',result.failure_reason
    phases=report['phases']
    if settings.path_predictor!='none':
        assert result.info['contact_path']['path_predictor']==settings.path_predictor
        report['executed_path_predictor']=result.info['contact_path']['path_predictor'];save()
    expected=({'full':[False,True],'explicit':[False,True],'prefix':[False],'resume':[False,True]} if press_only
              else {'full':[True,False,True],'explicit':[True,False,True],
                    'prefix':[True,False],'resume':[True,False,True]})[stage]
    assert [row['contact']['active_pressure_dofs']>0 for row in phases]==expected
    wrench=[]
    for row,active in zip(phases,expected):
        contact=row['contact']
        errors=dict(force=float(np.linalg.norm(np.asarray(contact['integration_force'])+contact['target_force'])/.02),
                    moment=float(np.linalg.norm(np.asarray(contact['integration_moment'])+contact['target_moment'])/.02))
        wrench.append(errors)
        assert max(errors.values())<=1e-6
        assert contact['wrench_origin_m']==[0.,0.,0.]
        assert contact['gap_tolerance_met'] and contact['minimum_gap']>=-5e-5
        assert contact['contact_work_convention']=='projected_weak_gap_nonpotential_v1'
        assert (contact['regularization_storage_j']>0.)==active
        assert contact['penalty_energy']==0.
    report['phase_wrench_errors']=wrench;save()
    target_dof=model.mesh.dof_manager.get_node_dofs(9)[2]
    report['target_node9_uz_m']=float(result.displacements[target_dof])
    if stage!='prefix':
        assert abs(result.displacements[target_dof])>1e-10
        manager=GeneralS3PatchManager(model,resolution.definition,settings)
        residual,_,contact_result=manager.evaluate(result.displacements,tangent=False)
        from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
        structural,_,states=_assemble_nonlinear_system(model,result.displacements,result.element_states,
            5,tangent=False,kinematics='von_karman')
        _discard_nonlinear_state_candidate(states)
        equilibrium=(structural+residual)[target_dof]
        report['target_equilibrium_normalized_error']=abs(float(equilibrium))/.02
        report['support_reactions']=result.steps[-1].support_reactions
        report['actuator_reaction_n']=float(np.sum((structural+residual)[
            [model.mesh.dof_manager.get_node_dofs(n)[2] for n in range(1,7)]]))
        report['target_support_reaction_n']=float(np.sum((structural+residual)[
            [model.mesh.dof_manager.get_node_dofs(n)[2] for n in (7,8,10)]]))
        save()
        assert abs(equilibrium)/.02<=1e-6
        assert abs(report['actuator_reaction_n'])>1e-6
        assert abs(report['target_support_reaction_n'])>1e-6
        assert abs(report['actuator_reaction_n']+report['target_support_reaction_n'])/.02<=1e-6
        source_probe=np.zeros(manager.n_dof);target_probe=np.zeros(manager.n_dof)
        for node in range(1,7):source_probe[model.mesh.dof_manager.get_node_dofs(node)[2]]=1e-6
        for node in range(7,11):target_probe[model.mesh.dof_manager.get_node_dofs(node)[2]]=1e-6
        ws=float(residual@source_probe);wt=float(residual@target_probe)
        source_work=float(residual@(result.displacements*(source_probe!=0)))
        target_work=float(residual@(result.displacements*(target_probe!=0)))
        report['both_body_work']=dict(source_virtual_work_j=ws,target_virtual_work_j=wt,
            common_translation_normalized_error=abs(ws+wt)/(.02*1e-6),
            source_residual_dot_displacement_j=source_work,target_residual_dot_displacement_j=target_work,
            interpretation='instantaneous residual work; not path-integrated work or regularization storage')
        save()
        assert abs(ws)>1e-12 and abs(wt)>1e-12
        assert abs(ws+wt)/(.02*1e-6)<=1e-6
        assert abs(source_work)>1e-12 and abs(target_work)>1e-12
        np.testing.assert_allclose(ws,-contact_result.integration_force[2]*1e-6,rtol=1e-6,atol=2e-14)
        np.testing.assert_allclose(wt,-contact_result.target_force[2]*1e-6,rtol=1e-6,atol=2e-14)
    if stage in ('explicit','resume'):
        full=json.loads((evidence/full_file).read_text(encoding='utf-8'))
        assert full['identity']==resolution.identity and full['verified']
        np.testing.assert_allclose(result.displacements,full['displacements'],rtol=1e-7,atol=1e-9)
        response_errors={}
        for key,scale in (('integration_force',.02),('target_force',.02),
                          ('integration_moment',.02),('target_moment',.02),
                          ('minimum_gap',5e-5),('regularization_storage_j',2e-7)):
            error=float(np.linalg.norm(np.asarray(report['contact'][key])-full['contact'][key])/scale)
            response_errors[key]=error
            assert error<=1e-6
        report['normalized_response_parity']=response_errors;save()
    assert report['checkpoint']
    report['verified']=True;save()
