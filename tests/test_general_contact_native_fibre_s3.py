"""Physical-fibre retained contact composition, with separate family history."""
from pathlib import Path
from hashlib import sha256
import json,os
import numpy as np
import pytest
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.general_contact.native_multi_s3_static import FIBRE_CURVED_MATERIAL_PROFILE
from test_general_contact_native_curved_s3_static import _analysis
from test_general_contact_native_curved_static import _canonical_fixture_payload


def _independent_material_history(owner,mechanical,origins):
    from types import SimpleNamespace
    from test_ge_beam3_fibre_section import oracle,close_exact,exact_pair
    yielded=0;dissipation=0.;proposed=[]
    for index,(_,element) in enumerate(owner.context.elements):
        rows=element.operator.recover(mechanical.cell_rotations[index],mechanical.resultants[index],
            origin=origins[index],check=owner.context._backend.check)
        section=element.section
        for row,origin in zip(rows,origins[index].stations):
            expected,tangent,potential,fibres=oracle(section,SimpleNamespace(
                strain=row['strain'],strain_low=row['strain_low'],origin=origin))
            for actual,wanted,history,fibre in zip(row['fibres'],fibres,row['history'].rows,section.fibres):
                for pair,key in ((actual.stress,'stress'),(actual.algorithmic_modulus,'modulus'),
                                 (actual.plastic_increment,'delta'),(history[:2],'z'),(history[2:],'p')):
                    close_exact(exact_pair(pair),wanted[key])
                yielded+=int(wanted['delta']!=0)
                dissipation+=row['measure']*fibre.area*fibre.curve.flow_stress[0]*float(abs(wanted['delta']))
            proposed.append(row['history'])
            # Compare complete section resultants, including the additive background.
            from test_ge_beam3_fibre_section import exact_array
            actual=exact_array(row['resultants'],row['resultants_low'])
            for got,wanted in zip(actual,expected):close_exact(got,wanted)
    return dict(yielded=yielded,dissipation_j=dissipation),tuple(proposed)


def test_plastic_fibre_candidates_use_committed_origin_and_independent_return():
    owner=_analysis(physical_fibres=True,plastic_fibres=True)
    fields=json.loads(_canonical_fixture_payload(Path(__file__).parent/'data/general_contact_native_curved_s3_v2_active.checkpoint'))['records'][0]
    context=owner.context;mechanical=context.make(fields['mechanical'])
    origins=owner.initial.native_histories;before=canonical(origins)
    residual,tangent,_,responses,_=context.assemble(mechanical,1.,origins)
    report,proposed=_independent_material_history(owner,mechanical,origins)
    assert report['yielded']>0 and report['dissipation_j']>0.
    assert canonical(origins)==before
    assert tuple(proposed)==responses[0].history.stations
    direction=np.random.default_rng(610).normal(size=context.count)*1e-4
    direction[context._backend.layout.fixed]=0.
    errors=[]
    for h in (1e-5,2e-6):
        values=[context.assemble(context.advance(mechanical,sign*h*direction),1.,origins)[0] for sign in (1,-1)]
        derivative=(values[0]-values[1])/(2*h)
        error=float(np.linalg.norm(tangent@direction-derivative)/np.linalg.norm(derivative));errors.append(error)
        assert error<=1e-5
    # A discarded candidate cannot become the origin of the next material trial.
    small=mechanical.descriptor();small['resultants']=mechanical.resultants*.1
    small=context.make(small)
    reduced=context.assemble(small,1.,origins)[3][0].history
    repeated=context.assemble(mechanical,1.,origins)[3][0].history
    assert repeated==responses[0].history and canonical(origins)==before
    assert reduced!=repeated
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path+'-plastic-local.json').write_text(json.dumps(dict(**report,directional_errors=errors),indent=2))


def test_plastic_fibre_first_contact_commits_real_history_and_bound_restart():
    owner=_analysis(physical_fibres=True,plastic_fibres=True)
    before=canonical(owner.initial.native_histories)
    run=owner.solve(stop_after=1)
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-plastic-active.checkpoint').write_bytes(run.checkpoint)
        Path(path+'-plastic-active.json').write_text(json.dumps(dict(status=run.status,failure=run.failure,
            completed=run.accepted.completed_targets,candidates=owner.candidate_evaluations,
            trial_failures=owner.curved_trial_failures),indent=2))
    assert run.status=='paused',run.failure
    assert run.accepted.completed_targets==1 and run.accepted.contact.contact.minimum_gap_m>=-3e-5
    contact=run.accepted.contact.contact
    assert contact.potential_j>0. and np.linalg.norm(contact.beam_force_n)>0.
    np.testing.assert_allclose(contact.beam_force_n+contact.shell_force_n,0.,atol=1e-10)
    np.testing.assert_allclose(contact.beam_moment_nm+contact.shell_moment_nm,0.,atol=1e-10)
    report,proposed=_independent_material_history(owner,run.accepted.mechanical,owner.initial.native_histories)
    assert report['yielded']>0 and report['dissipation_j']>0.
    assert tuple(proposed)==run.accepted.native_histories[0].stations
    assert canonical(owner.initial.native_histories)==before
    restored,records=_analysis(physical_fibres=True,plastic_fibres=True)._restore(run.checkpoint,run.checkpoint_sha256)
    assert restored.native_histories==run.accepted.native_histories
    assert owner._checkpoint(records)==run.checkpoint
    with pytest.raises(ValueError,match='binding'):
        _analysis(physical_fibres=True)._restore(run.checkpoint,run.checkpoint_sha256)
    if path:Path(path+'-plastic-history.json').write_text(json.dumps(report,indent=2))


def test_fibre_contact_context_complete_spatial_work_and_elastic_section_reference():
    fibre=_analysis(targets=(1.,),physical_fibres=True);general=_analysis(targets=(1.,))
    context=fibre.context;n=context.count
    assert n==42 and fibre.profile==FIBRE_CURVED_MATERIAL_PROFILE
    section=context.elements[0][1].section
    # Independent axial/biaxial sectional elasticity, not the production map.
    expected=section.background_factor.T@section.background_factor
    for row in section.fibres:
        b=np.array((1.,0.,0.,0.,row.z,-row.y))
        expected+=row.young*row.area*np.outer(b,b)
    np.testing.assert_allclose(expected,np.diag((1000.,400.,350.,.8,1.,1.2)),rtol=1e-14,atol=1e-14)
    d=np.random.default_rng(571).normal(size=n)*1e-5
    d[list(context._backend.layout.fixed)]=0.
    state=context.advance(context.initial.mechanical,d)
    gstate=general.context.advance(general.context.initial.mechanical,d)
    origins=context.initial.histories;before=canonical(origins)
    r,k,_,responses,_=context.assemble(state,1.,origins)
    gr,gk,_,_,_=general.context.assemble(gstate,1.,general.context.initial.histories)
    np.testing.assert_allclose(r,gr,rtol=1e-8,atol=1e-11)
    np.testing.assert_allclose(k,gk,rtol=1e-8,atol=1e-10)
    v=np.random.default_rng(573).normal(size=n)*.001;v[list(context._backend.layout.fixed)]=0.
    errors=[]
    for h in (1e-5,2e-6):
        values=[context.assemble(context.advance(state,sign*h*v),1.,origins)[0] for sign in (1,-1)]
        derivative=(values[0]-values[1])/(2*h)
        error=float(np.linalg.norm(k@v-derivative)/np.linalg.norm(derivative));errors.append(error)
        assert error<=1e-5
    assert canonical(origins)==before
    assert context.histories(tuple(value.history for value in responses))!=general.context.initial.histories
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path+'-fibre-context.json').write_text(json.dumps(dict(
        count=n,directional_errors=errors,elastic_residual_max_error=float(np.max(np.abs(r-gr))),
        elastic_tangent_max_error=float(np.max(np.abs(k-gk)))),indent=2))


def test_fibre_contact_rejects_foreign_history_and_changed_layout():
    owner=_analysis(targets=(1.,),physical_fibres=True);general=_analysis(targets=(1.,))
    with pytest.raises(ValueError):owner.context.histories(general.initial.native_histories)
    owner.context._backend.layout.scales[24]*=2.
    with pytest.raises(ValueError,match='changed'):owner.context.guard()


@pytest.mark.parametrize('cache',('force','node_index','reference_positions','reference_frames','probes','backend_probes','started'))
def test_fibre_contact_guards_consumed_caches(cache):
    context=_analysis(targets=(1.,),physical_fibres=True).context
    layout=context._backend.layout
    if cache=='force':
        layout.force=layout.force.copy();layout.force[6]+=.1
    elif cache=='node_index':layout.node_index={key:value+1 for key,value in layout.node_index.items()}
    elif cache=='reference_positions':
        layout.reference_positions=layout.reference_positions.copy();layout.reference_positions[0,0]+=.01
    elif cache=='reference_frames':
        layout.reference_frames=layout.reference_frames.copy();layout.reference_frames[0]=np.eye(3)
    elif cache=='probes':layout.probes=()
    elif cache=='backend_probes':context._backend.probes=()
    else:context._backend.started+=120.
    with pytest.raises(ValueError,match='changed'):context.guard()


def test_fibre_contact_expired_context_is_explicit_resource_failure(monkeypatch):
    import anyfesolver._ge_beam3_retained_fibre_state as backend
    from anyfesolver._ge_beam3_effective_deadline import effective_deadline_seconds
    owner=_analysis(targets=(1.,),physical_fibres=True)
    initial=owner.initial
    histories=canonical(initial.native_histories)
    started=owner.context._backend.started
    monkeypatch.setattr(backend,'monotonic',lambda:started+effective_deadline_seconds(120.)+1.)
    with pytest.raises(RuntimeError,match='cooperative deadline'):owner.solve()
    assert owner.initial is initial and canonical(owner.initial.native_histories)==histories
    assert owner.context._backend.started==started
    assert owner.context._token is None and not owner.context._lock.locked()


@pytest.mark.parametrize('cancelled',(False,True))
def test_fibre_material_candidate_exception_and_cancellation_preserve_origin(monkeypatch,cancelled):
    from anyfesolver.control import CancellationToken
    owner=_analysis(targets=(1.,),physical_fibres=True);token=CancellationToken()
    before=owner._checkpoint(());history=canonical(owner.initial.native_histories)
    probe=owner.context._backend.probes[0];original=type(probe).evaluate
    def interrupted(self,*args,**kwargs):
        if cancelled:token.cancel()
        else:raise RuntimeError('fibre material candidate failure')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(type(probe),'evaluate',interrupted)
    run=owner.solve(cancellation_token=token)
    assert run.status==('cancelled' if cancelled else 'failed')
    assert run.accepted.completed_targets==0 and run.checkpoint==before
    assert canonical(run.accepted.native_histories)==history
    assert owner.context._token is None and not owner.context._cancel_active


def test_fibre_contact_precancel_returns_unchanged_initial_state():
    from anyfesolver.control import CancellationToken
    owner=_analysis(targets=(1.,),physical_fibres=True);before=owner._checkpoint(())
    token=CancellationToken();token.cancel()
    run=owner.solve(cancellation_token=token)
    assert run.status=='cancelled' and run.checkpoint==before


def test_curved_fibre_coupled_elastic_first_contact_and_bound_restart():
    owner=_analysis(physical_fibres=True);run=owner.solve(stop_after=1)
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-fibre-active.checkpoint').write_bytes(run.checkpoint)
        Path(path+'-fibre-active.json').write_text(json.dumps(dict(status=run.status,failure=run.failure,
            profile=run.profile,completed_targets=run.accepted.completed_targets,
            candidate_evaluations=owner.candidate_evaluations,pair_evaluations=owner.pair_evaluations,
            trial_failures=owner.curved_trial_failures),indent=2))
    assert run.status=='paused',run.failure
    assert run.profile==FIBRE_CURVED_MATERIAL_PROFILE and run.accepted.completed_targets==1
    assert run.accepted.contact.contact.minimum_gap_m>=-3e-5
    np.testing.assert_allclose(run.accepted.contact.contact.beam_force_n+
        run.accepted.contact.contact.shell_force_n,0.,atol=1e-10)
    np.testing.assert_allclose(run.accepted.contact.contact.beam_moment_nm+
        run.accepted.contact.contact.shell_moment_nm,0.,atol=1e-10)
    reference=Path(__file__).parent/'data/general_contact_native_curved_s3_v2_active.checkpoint'
    fields=json.loads(_canonical_fixture_payload(reference))['records'][0]
    for key,value in fields['mechanical'].items():
        np.testing.assert_allclose(getattr(run.accepted.mechanical,key),value,rtol=1e-7,atol=1e-10)
    np.testing.assert_allclose(run.accepted.shell_displacement,fields['shell_displacement'],rtol=1e-7,atol=1e-10)
    # History stays in the physical-fibre family, never decoded as ellipsoid data.
    for cell in run.accepted.native_histories:
        for station in cell.stations:
            assert len(station.rows)==4 and np.count_nonzero(np.array(station.rows))==0
    restored,records=_analysis(physical_fibres=True)._restore(run.checkpoint,run.checkpoint_sha256)
    assert restored.completed_targets==1 and owner._checkpoint(records)==run.checkpoint
    with pytest.raises(ValueError,match='binding'):
        _analysis()._restore(run.checkpoint,sha256(run.checkpoint).hexdigest())
