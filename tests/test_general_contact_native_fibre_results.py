"""Committed energy/recovery summaries checked against rational section work."""
from pathlib import Path
from hashlib import sha256
from fractions import Fraction as Q
import json,os
import numpy as np
import pytest
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_general_contact_native_curved_s3_static import _analysis
from test_ge_beam3_fibre_section import exact_array,exact_pair,rational_fibre,integral


def _checkpoint():
    return (Path(__file__).parent/'data/general_contact_native_fibre_s3_plastic_open_v1.checkpoint').read_bytes()


def _stored_dissipation(section,station,origin):
    e=exact_array(np.array(station['strain']),np.array(station['strain_low']))
    factor=np.array([[Q(float(x)) for x in row] for row in section.background_factor],dtype=object)
    stored=sum(e*(factor.T@factor@e))/2;previous=Q(0);dissipated=Q(0)
    for fibre,history in zip(section.fibres,origin.rows):
        strain=e[0]+Q(fibre.z)*e[4]-Q(fibre.y)*e[5]
        value=rational_fibre(fibre,strain,exact_pair(history[:2]),exact_pair(history[2:]))
        area=Q(fibre.area);young=Q(fibre.young);yield_stress=Q(fibre.curve.flow_stress[0])
        stored+=area*(young*value['elastic']**2/2+integral(fibre.curve,Q(0),value['p'])-yield_stress*value['p'])
        previous+=area*(integral(fibre.curve,Q(0),exact_pair(history[2:]))-yield_stress*exact_pair(history[2:]))
        dissipated+=area*yield_stress*abs(value['delta'])
    return stored,dissipated,stored-previous+dissipated


def test_committed_recovery_energy_units_origins_and_separate_storage():
    raw=_checkpoint();owner=_analysis(physical_fibres=True,plastic_fibres=True)
    initial=canonical(owner.initial.native_histories)
    report=owner.recover_material(raw,sha256(raw).hexdigest(),include_stations=True)
    assert report['completed_targets']==2 and not report['energy_balance_qualified']
    assert report['checkpoint_sha256']==sha256(raw).hexdigest()
    origins=owner.initial.native_histories;records=json.loads(raw)['records'];cumulative=0.
    errors=[]
    for entry,record in zip(report['increments'],records):
        section=owner.context.elements[0][1].section
        expected=np.zeros(3)
        for row,origin in zip(entry['elements'][0]['stations'],origins[0].stations):
            independent=_stored_dissipation(section,row,origin)
            for key,value in zip(('stored_energy_per_reference_length_n',
                    'plastic_dissipation_per_reference_length_n',
                    'incremental_material_potential_per_reference_length_n'),independent):
                np.testing.assert_allclose(row[key],float(value),rtol=1e-12,atol=1e-24)
            expected+=row['measure']*np.array(list(map(float,independent)))
        keys=('stored_native_material_energy_j','plastic_dissipation_increment_j','incremental_material_potential_j')
        actual=np.array([entry[k] for k in keys]);np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-24)
        errors.append(float(np.max(np.abs(actual-expected))))
        u=np.array(record['shell_displacement'])
        np.testing.assert_allclose(entry['stored_shell_energy_j'],.5*u@owner.shell_stiffness@u,rtol=1e-14,atol=1e-24)
        assert entry['contact_regularization_storage_j']==record['contact']['potential_j']
        np.testing.assert_allclose(entry['total_stored_energy_j'],entry['stored_native_material_energy_j']+
            entry['stored_shell_energy_j']+entry['contact_regularization_storage_j'],rtol=1e-14,atol=1e-24)
        cumulative+=entry['plastic_dissipation_increment_j']
        np.testing.assert_allclose(entry['plastic_dissipation_cumulative_j'],cumulative,rtol=1e-14,atol=1e-24)
        origins=owner.context.histories(record['histories'],decoded=True)
    assert report['increments'][1]['contact_regularization_storage_j']==0.
    assert report['increments'][1]['stored_native_material_energy_j']>0.
    assert canonical(owner.initial.native_histories)==initial
    assert report['plastic_dissipation_cumulative_j']==report['increments'][-1]['plastic_dissipation_cumulative_j']
    # Output mutation has no authority to mutate the committed owner/checkpoint.
    report['increments'][0]['elements'][0]['stations'][0]['history']['rows'][0][0]=999.
    compact=owner.recover_material(raw,sha256(raw).hexdigest())
    assert 'stations' not in compact['increments'][0]['elements'][0]
    assert canonical(owner.initial.native_histories)==initial
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path+'-material-results.json').write_text(json.dumps(dict(max_energy_errors_j=errors,report=compact),indent=2))


def test_material_recovery_binding_precancel_and_no_partial_publication():
    raw=_checkpoint();owner=_analysis(physical_fibres=True,plastic_fibres=True)
    with pytest.raises(ValueError,match='hash'):owner.recover_material(raw,'0'*64)
    with pytest.raises(ValueError,match='binding'):_analysis(physical_fibres=True).recover_material(raw,sha256(raw).hexdigest())
    with pytest.raises(ValueError,match='family'):_analysis().recover_material(raw,sha256(raw).hexdigest())
    with pytest.raises(ValueError,match='boolean'):owner.recover_material(raw,sha256(raw).hexdigest(),include_stations=1)
    token=CancellationToken();token.cancel()
    with pytest.raises(SolveCancelled):owner.recover_material(raw,sha256(raw).hexdigest(),cancellation_token=token)
    assert owner.context._token is None and not owner.context._lock.locked()


@pytest.mark.parametrize('cancelled',(False,True))
def test_station_recovery_interruption_leaves_committed_state(monkeypatch,cancelled):
    raw=_checkpoint();owner=_analysis(physical_fibres=True,plastic_fibres=True)
    initial=canonical(owner.initial.native_histories);token=CancellationToken()
    section=owner.context.elements[0][1].section;response=type(section).response
    replayed=False;restore=owner._restore
    def validated_prefix(*args,**kwargs):
        nonlocal replayed
        result=restore(*args,**kwargs);replayed=True
        return result
    monkeypatch.setattr(owner,'_restore',validated_prefix)
    def interrupted(self,*args,**kwargs):
        assert replayed
        if cancelled:token.cancel()
        else:raise RuntimeError('recovery material interruption')
        return response(self,*args,**kwargs)
    monkeypatch.setattr(type(section),'response',interrupted)
    with pytest.raises(SolveCancelled if cancelled else RuntimeError):
        owner.recover_material(raw,sha256(raw).hexdigest(),cancellation_token=token)
    assert canonical(owner.initial.native_histories)==initial
    assert owner.context._token is None and not owner.context._lock.locked()


def test_cancellation_at_result_publication_returns_no_partial_report(monkeypatch):
    import anyfesolver.general_contact.native_fibre_results as results
    raw=_checkpoint();owner=_analysis(physical_fibres=True,plastic_fibres=True)
    token=CancellationToken();serialize=results.canonical;attempted=False
    initial=canonical(owner.initial.native_histories)
    def before_publication(value):
        nonlocal attempted
        if value.get('schema')==results.SCHEMA:attempted=True;token.cancel()
        return serialize(value)
    monkeypatch.setattr(results,'canonical',before_publication)
    with pytest.raises(SolveCancelled):owner.recover_material(raw,sha256(raw).hexdigest(),cancellation_token=token)
    assert attempted and canonical(owner.initial.native_histories)==initial
    assert owner.context._token is None and not owner.context._lock.locked()


def test_example_recovers_material_json_without_solving_new_target(monkeypatch,tmp_path,capsys):
    import examples.general_contact_native_curved_material_s3 as example
    from anyfesolver.general_contact.native_multi_s3_static import NativeRetainedMultiS3Analysis
    source=Path(__file__).parent/'data/general_contact_native_fibre_s3_plastic_open_v1.checkpoint'
    output=tmp_path/'material.json'
    def forbidden_solve(*args,**kwargs):raise AssertionError('recovery must not advance the contact path')
    monkeypatch.setattr(NativeRetainedMultiS3Analysis,'solve',forbidden_solve)
    monkeypatch.setattr('sys.argv',['example','--plastic-fibres','--restart',str(source),
        '--recover-material',str(output),'--material-stations'])
    example.main()
    summary=json.loads(capsys.readouterr().out);report=json.loads(output.read_text())
    assert summary['status']=='recovered' and report['completed_targets']==2
    assert len(report['increments'][0]['elements'][0]['stations'])==8
