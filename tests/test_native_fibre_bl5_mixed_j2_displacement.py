"""Public yielded all-S3/mixed native BL5 histories and exact predecessor checks."""
from dataclasses import replace,asdict
from pathlib import Path
import os
from math import fsum
import json
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver._stiffened_shell_displacement_checkpoint import identities,decode_snapshot
from examples.stiffened_shell_native_fibre_bl5_mixed_j2_displacement import definition
from examples.stiffened_shell_native_fibre_bl5_displacement import ACTIONS,CONTROL,OPTIONS,TARGETS

def family_history(owner,result):
    """Model-declared families; native fibre rows never include generalized rows."""
    from anyfesolver._ge_beam3_fibre_section import PhysicalFibreSection
    _,arrays=decode_snapshot(owner,owner.checkpoint())
    shell_records={r.element_id:(family,r) for family,records in
                   (("Q4",result.shell_q4_material),("S3",result.shell_s3_material)) for r in records}
    shells=[dict(family=shell_records[eid][0],element_id=eid,node_ids=list(element.node_ids),
                 alpha=shell_records[eid][1].alpha.tolist(),
                 plastic_strain=shell_records[eid][1].plastic_strain.tolist())
            for eid,element,_,_ in owner._shell_parts]
    fibre_rows=arrays['beam_fibres'].tolist();generalized_rows=arrays['beam_accumulated'].tolist()
    physical=[];generalized=[];offset=0;generalized_offset=0
    for element,_,_ in owner._beam_parts:
        count=len(element.operator.cell.stations)
        if type(element.section) is PhysicalFibreSection:
            records=[r for r in result.beam_fibre_material if r.element_id==element.element_id]
            assert len(records)==count
            rows=[]
            for station in records:
                for fibre in element.section.fibres:
                    history=fibre_rows[offset];offset+=1
                    rows.append(dict(cell=station.cell,station=station.station,xi=station.xi,
                        fibre_id=fibre.fibre_id,plastic_high_low=history[:2],
                        accumulated_high_low=history[2:],accumulated=fsum(history[2:])))
            physical.append(dict(family='PHYSICAL_AXIAL_BIAXIAL_FIBRE',element_id=element.element_id,
                                 node_ids=list(element.node_ids),stations=rows))
        else:
            rows=generalized_rows[generalized_offset:generalized_offset+count];generalized_offset+=count
            generalized.append(dict(family='RESULTANT_ELLIPSOID',element_id=element.element_id,
                node_ids=list(element.node_ids),accumulated_high_low=rows))
    assert offset==len(fibre_rows) and generalized_offset==len(generalized_rows)
    return dict(generation=result.generation,target=result.target,amplitude=result.amplitude,
                shells=shells,physical_fibre_members=physical,generalized_members=generalized,
                s3_alpha_max=max((max(r['alpha']) for r in shells if r['family']=='S3'),default=0.),
                physical_fibre_alpha_max=max((r['accumulated'] for m in physical for r in m['stations']),default=0.))


@pytest.fixture(scope='module',params=(('s3','biaxial','mixed','p0'),('mixed','biaxial','mixed','continuous_p1_coarse'),('s3','axial','fibre','p0')))
def loaded(request,tmp_path_factory):
    folder=tmp_path_factory.mktemp("loaded-"+request.param[0]+"-"+request.param[1])
    model=definition(*request.param);owner=ss.create_analysis(definition=model)
    try:
        seed=owner.solve(ACTIONS,(.001,),**OPTIONS);assert seed.status=='completed'
        raw=owner.checkpoint();fresh=ss.create_analysis(definition=model)
        try:fresh.restore(raw);assert fresh.checkpoint()==raw
        finally:fresh.close()
        observed=[]
        try:
            result=owner.solve(ACTIONS,TARGETS[:2],displacement_control=CONTROL,on_accepted=observed.append,**OPTIONS)
        except Exception as error:
            (folder/('failure-'+str(request.param[0])+'-'+request.param[1]+'.json')).write_text(json.dumps(dict(configuration=request.param,accepted=[dict(target=v.target,amplitude=v.amplitude,generation=v.generation,relative=v.free_equilibrium_ratio,gap=v.constraint_max,shell_alpha=v.shell_layer_alpha,fibre_alpha=v.beam_station_accumulated) for v in observed],error=str(error),detail=asdict(error.detail) if hasattr(error,'detail') else None),indent=2))
            raise
        assert result.status=='completed'
        histories=family_history(owner,result)
        assert histories['physical_fibre_alpha_max']>0
        if request.param[1]=='biaxial':assert histories['s3_alpha_max']>0
        (folder/('loaded-'+request.param[0]+'-'+request.param[1]+'.json')).write_text(json.dumps(dict(configuration=request.param,histories=histories),indent=2))
        assert result.shell_s3_material and result.beam_fibre_material
        assert bool(result.shell_q4_material)==bool(model.shell.q4_elements)
        if request.param[2]=='mixed':assert result.beam_generalized_material
        yield model,owner.checkpoint()
    finally:owner.close()


def test_public_restart_continue_and_atomic_refusals(loaded,tmp_path,monkeypatch):
    model,saved=loaded;owner=ss.create_analysis(definition=model);owner.restore(saved)
    fresh=ss.create_analysis(definition=owner._definition)
    try:
        restored=fresh.restore(saved);assert fresh.checkpoint()==saved
        assert restored.shell_layer_alpha==owner.recover().shell_layer_alpha
        assert owner.capabilities['displacement_control']=='experimental_native_fibre_bl5_mixed_shell_j2_fixed_translation_v1'
        assert 'Q4/S3 J2' in owner.capabilities['native_bl5_material_domain']
        assert identities(owner)[1]=='B3_GE_NATIVE_FIBRE_BL5_MIXED_SHELL_J2_DISPLACEMENT_V1_DEV'
        metadata,arrays=decode_snapshot(owner,saved)
        assert len(arrays['s3_states'])==2*len(owner._definition.shell.s3_elements)
        for opts in (dict(increment_control=ss.AdaptiveIncrementControl()),dict(displacement_control=replace(CONTROL,trial_policy=ss.GENERALIZED_BRANCH_EVENT_TRIAL))):
            options=dict(displacement_control=CONTROL,**OPTIONS);options.update(opts)
            with pytest.raises(ValueError):owner.solve(ACTIONS,TARGETS[2:],**options)
            assert owner.checkpoint()==saved
        corrupt=ss.create_analysis(definition=model)
        try:
            with pytest.raises(ValueError):corrupt.restore(saved[:-1]+bytes((saved[-1]^1,)))
            assert corrupt._finite_driver is None
        finally:corrupt.close()
        assert owner.checkpoint()==saved
        token=CancellationToken();driver=owner._finite_driver;original=driver._advance
        def cancel_candidate(*args,**kwargs):
            candidate=original(*args,**kwargs);token.cancel('owned candidate before material/publication');return candidate
        with monkeypatch.context() as patch:
            patch.setattr(driver,'_advance',cancel_candidate)
            with pytest.raises(SolveCancelled):owner.solve(ACTIONS,TARGETS[2:],displacement_control=CONTROL,cancellation_token=token,**OPTIONS)
        assert owner.checkpoint()==saved
        from anyfesolver._ge_beam3_retained_fibre import RetainedFibreOperator
        with monkeypatch.context() as patch:
            def changed_material(_operator):raise ValueError('captured retained fibre inventory changed')
            patch.setattr(RetainedFibreOperator,'guard',changed_material)
            with pytest.raises(ValueError,match='captured retained fibre'):
                owner.solve(ACTIONS,TARGETS[2:],displacement_control=CONTROL,**OPTIONS)
        assert owner.checkpoint()==saved
        final=owner.solve(ACTIONS,TARGETS[2:],displacement_control=CONTROL,**OPTIONS)
        resumed=fresh.solve(ACTIONS,TARGETS[2:],displacement_control=CONTROL,**OPTIONS)
        assert final.status==resumed.status=='completed' and owner.checkpoint()==fresh.checkpoint()
        assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
        final_histories=family_history(owner,final)
        assert final_histories['physical_fibre_alpha_max']>0
        if any(abs(fibre.y)>0. for element,_,_ in owner._beam_parts
               if hasattr(element.section,'fibres') for fibre in element.section.fibres):
            assert final_histories['s3_alpha_max']>0
        (tmp_path/'family-history.json').write_text(json.dumps(final_histories,indent=2))
        (tmp_path/'result.json').write_text(json.dumps(dict(shell=owner._definition.shell.s3_elements,generation=final.generation,shell_alpha=final.shell_layer_alpha,fibre_alpha=final.beam_station_accumulated,relative=final.free_equilibrium_ratio,gap=final.constraint_max)))
    finally:fresh.close();owner.close()


def test_yielded_assembled_tangent_and_material_capture(loaded,tmp_path):
    model,saved=loaded;owner=ss.create_analysis(definition=model);owner.restore(saved);driver=owner._finite_driver
    state=driver._state;shell=driver._shell_predecessors;beam=driver._beam_predecessors
    applied=owner.recover().applied_dead_load
    base=driver._assemble(state,0.,shell,beam,applied_load=applied)
    direction=np.zeros(base[1].shape[0]);direction[owner._free]=np.linspace(-1.,1.,len(owner._free))*1e-6
    direction[len(state.q):]=np.linspace(-1.,1.,len(state.multipliers))
    for j in range(len(owner._beam_parts)):
        offset=owner._shell_count+6*len(owner._beam_ids)+24*j
        direction[offset+6:offset+24]=np.linspace(-1.,1.,18)
    captured=driver._q4_boundary_trial[3]
    assert len(captured)==len(shell)
    exact=base[1]@direction;errors=[];strain_errors=[]
    for h in (1e-5,1e-6):
        plus=driver._advance(state,direction,h);minus=driver._advance(state,direction,-h)
        ep=driver._assemble(plus,0.,shell,beam,applied_load=applied)
        em=driver._assemble(minus,0.,shell,beam,applied_load=applied)
        errors.append(float(np.linalg.norm((ep[0]-em[0])/(2*h)-exact)/max(np.linalg.norm(exact),1.)))
        for i,(slots,trial) in enumerate(captured):
            deriv=(ep[4][i]['layer_strain']-em[4][i]['layer_strain'])/(2*h)
            expected=np.asarray(trial.strain_differential)@direction[list(slots)]
            strain_errors.append(float(np.linalg.norm(deriv-expected)/max(np.linalg.norm(expected),1e-12)))
    assert max(errors)<=1e-5,errors
    assert max(strain_errors)<=1e-5,strain_errors
    from anyfesolver._stiffened_shell_s3_material_event import boundary_amount
    from anyfesolver._stiffened_shell_q4_material_event import boundary_amount as q4_boundary
    for slots,trial in captured:
        function=boundary_amount if len(slots)==18 else q4_boundary
        assert function(trial,np.zeros(len(slots)),origin=trial.origin) is None
        with pytest.raises(ValueError):function(trial,np.zeros(17),origin=trial.origin)
        with pytest.raises(ValueError):function(trial,np.zeros(len(slots)),origin=dict(trial.origin))
        if len(slots)==18:
            changed=dict(trial.origin);changed['initial_generalized_prestrain']=np.asarray(changed['initial_generalized_prestrain']).copy();changed['initial_generalized_prestrain'][0,0]+=1e-6
            from anyfesolver._stiffened_shell_q4_material_event import _origin_values
            assert _origin_values(changed)!=trial.origin_values
    assert owner.checkpoint()==saved
    owner.close()
    (tmp_path/'tangent.json').write_text(json.dumps(dict(errors=errors,strain_errors=strain_errors,criterion=1e-5)))

@pytest.mark.parametrize('shell',('s3','mixed'))
def test_distinct_policy_preserves_legacy_guards(shell):
    model=definition(shell)
    for policy in (ss.BL5_NATIVE_FIBRE_MATERIAL_V1,ss.BL5_ELASTIC_NATIVE_FIBRE_MIXED_SHELL_V1):
        with pytest.raises(ValueError):ss.create_analysis(definition=replace(model,bl5_material_policy=policy))
    with pytest.raises(ValueError):ss.create_analysis(definition=replace(model,shell=replace(model.shell,j2_curve=None)))
