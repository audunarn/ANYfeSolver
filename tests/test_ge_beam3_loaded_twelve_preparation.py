"""Twelve-macro allocation/replay wiring, without spectral solves."""
from hashlib import sha256
import numpy as np
import pytest
from docs.reference_cases import ge_beam3_loaded_arch_refinement as probe


@pytest.mark.parametrize('bad',[True,6.,0,8,24,None])
def test_unregistered_refinement_size_rejects(bad):
    with pytest.raises(ValueError,match='registered spectral refinement size'): probe.configuration(bad)


def test_twelve_inputs_bind_preserved_chain_and_same_continuum():
    config=probe.configuration(12)
    if not config['input'].exists(): pytest.skip('external preserved twelve-macro state absent')
    data=probe.inputs(12)
    assert len(data['input-twelve.json'])==278640
    assert sha256(data['input-twelve.json']).hexdigest()==config['sha']
    six=probe.inputs()
    for name in ('equilibrium-1.json','equilibrium-4.json','reference-1.json','reference-4.json'):
        assert data[name]==six[name]
    assert config['spectral_dimension']==222 and config['dynamic_dimension']==141


def test_complete_twelve_worker_wiring_without_eigensolves(tmp_path,monkeypatch):
    config=probe.configuration(12)
    if not config['input'].exists(): pytest.skip('external preserved twelve-macro state absent')
    from anyfesolver import _ge_beam3_controlled_fibre_modes as spectral
    calls=[]
    def fake_modes(model,program,raw,inertias,**options):
        assert options['coordinate_limit']==256 and options['exact_dimension_limit']==160 and options['max_coordinates']==512
        assert options['bounds']==(-1e6,1e8) and options['num_modes']==6
        packet,guard,load=spectral.prepare(model,program,raw,inertias,material_policy=spectral.FROZEN,
            expected_checkpoint_sha256=options['expected_checkpoint_sha256'],coordinate_limit=256,max_coordinates=512)
        guard(); calls.append(load)
        roots=np.arange(1.,7.)
        return packet,spectral.Analysis(spectral.FROZEN,'TEST_STUB_NOT_SCIENTIFIC',roots,np.zeros((222,6)),
            np.column_stack((roots-1e-11,roots+1e-11)),0.,0.,0.,packet.identity,sha256(raw).hexdigest(),
            equilibrium_load_parameter=load)
    monkeypatch.setattr(spectral,'solve_modes',fake_modes)
    monkeypatch.setattr(probe,'guard',lambda _:None)
    for key,value in probe.THREAD_ENVIRONMENT.items(): monkeypatch.setenv(key,value)
    probe.worker('0'*40,tmp_path,12)
    raw=probe.read(tmp_path/'comparison.pending.json'); result=probe.validate(tmp_path,raw,'0'*40,12)
    assert len(calls)==2 and result['production_qualified'] is False and result['retained_coordinate_limit']==512
    assert len(result['artifacts'])==10 and [row['macros'] for row in result['rows']]==[12,12]
    with pytest.raises(ValueError,match='scope'): probe.validate(tmp_path,raw,'0'*40)


def test_twelve_saved_replay_rejects_default_retained_budget():
    config=probe.configuration(12)
    if not config['input'].exists(): pytest.skip('external preserved twelve-macro state absent')
    from docs.reference_cases import ge_beam3_fibre_arch_refinement as family
    from anyfesolver import _ge_beam3_controlled_fibre_modes as spectral
    model=family.model(12); program=family.program(12)
    inertias={eid:np.diag([1.,1.,1.,.02,.01,.01]) for eid in model.mesh.elements}
    with pytest.raises(ValueError,match='bounded standalone retained fibre model'):
        spectral.prepare(model,program,probe.read(config['input']),inertias,material_policy=spectral.FROZEN,coordinate_limit=256)
