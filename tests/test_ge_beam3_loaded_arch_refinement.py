"""Refinement admission, saved-state wiring and bounded publication guards."""
from hashlib import sha256
from pathlib import Path
import numpy as np
import pytest
from docs.reference_cases import ge_beam3_loaded_arch_refinement as probe


def test_complete_worker_wiring_without_eigensolves(tmp_path,monkeypatch):
    if not probe.INPUT.exists(): pytest.skip('external preserved state absent')
    from anyfesolver import _ge_beam3_controlled_fibre_modes as spectral
    calls=[]
    def fake_modes(model,program,raw,inertias,**options):
        assert options['coordinate_limit']==128 and options['exact_dimension_limit']==96
        assert options['bounds']==(-1e6,1e8) and options['num_modes']==6
        packet,guard,load=spectral.prepare(model,program,raw,inertias,material_policy=spectral.FROZEN,
            expected_checkpoint_sha256=options['expected_checkpoint_sha256'],coordinate_limit=128)
        guard(); calls.append(load)
        roots=np.arange(1.,7.)
        return packet,spectral.Analysis(spectral.FROZEN,'TEST_STUB_NOT_SCIENTIFIC',roots,np.zeros((114,6)),
            np.column_stack((roots-1e-11,roots+1e-11)),0.,0.,0.,packet.identity,sha256(raw).hexdigest(),
            equilibrium_load_parameter=load)
    monkeypatch.setattr(spectral,'solve_modes',fake_modes)
    monkeypatch.setattr(probe,'guard',lambda _:None)
    for key,value in probe.THREAD_ENVIRONMENT.items(): monkeypatch.setenv(key,value)
    probe.worker('0'*40,tmp_path)
    raw=probe.read(tmp_path/'comparison.pending.json')
    result=probe.validate(tmp_path,raw,'0'*40)
    assert len(calls)==2 and result['production_qualified'] is False
    assert len(result['artifacts'])==10


def test_original_default_packet_bytes_unchanged():
    if not probe.OLD.exists(): pytest.skip('external preserved state absent')
    from docs.reference_cases import ge_beam3_fibre_arch_refinement as family
    from anyfesolver import _ge_beam3_controlled_fibre_modes as spectral
    model=family.model(4); program=family.program(4); raw=probe.read(probe.OLD/'checkpoint-1.json')
    inertias={eid:np.diag([1.,1.,1.,.02,.01,.01]) for eid in model.mesh.elements}
    packet,guard,_=spectral.prepare(model,program,raw,inertias,material_policy=spectral.FROZEN,
        expected_checkpoint_sha256=sha256(raw).hexdigest())
    guard()
    expected=probe.parse(probe.read(probe.OLD/'native-1.json'))['packet']
    from anyfesolver._ge_beam3_p5_seeded.core import canonical as native_canonical
    assert native_canonical(packet)==probe.canonical(expected)


def test_six_macros_rejected_by_original_default_budget():
    if not probe.INPUT.exists(): pytest.skip('external preserved state absent')
    from docs.reference_cases import ge_beam3_fibre_arch_refinement as family
    from anyfesolver import _ge_beam3_controlled_fibre_modes as spectral
    model=family.model(6); program=family.program(6)
    inertias={eid:np.diag([1.,1.,1.,.02,.01,.01]) for eid in model.mesh.elements}
    with pytest.raises(ValueError,match='small fibre spectral correctness model'):
        spectral.prepare(model,program,probe.read(probe.INPUT),inertias,material_policy=spectral.FROZEN)


@pytest.mark.parametrize('incident',['success','exit','memory','timeout','inactivity','invalid','cleanup','descendant','mutation'])
def test_refinement_cleanup_before_publication(tmp_path,monkeypatch,incident):
    calls=[]; packet=probe.canonical(dict(development=True))
    ticks=iter([0.,601. if incident=='timeout' else 121. if incident=='inactivity' else 1.,601.])
    monkeypatch.setattr(probe.time,'monotonic',lambda:next(ticks)); monkeypatch.setattr(probe.time,'sleep',lambda _:None)
    monkeypatch.setattr(probe,'guard',lambda _:calls.append('guard'))
    def validate(output,raw,revision,macros=6):
        calls.append('validate'); assert raw==packet
        if incident=='invalid': raise ValueError('invalid')
    monkeypatch.setattr(probe,'validate',validate)
    class Job:
        def __init__(self,memory): assert memory==24*(1<<30)
        def launch(self,command,*,cwd,env,stdout,stderr):
            assert command[3]=='docs.reference_cases.ge_beam3_loaded_arch_refinement'
            assert all(env[k]==v for k,v in probe.THREAD_ENVIRONMENT.items()) and 'PYTHONPATH' not in env
            self.root=Path(stdout.name).parent; (self.root/'comparison.pending.json').write_bytes(packet); return self
        def accounting(self): return (0 if incident=='inactivity' else 1,1 if incident=='descendant' else 0,24*(1<<30) if incident=='memory' else 1)
        def poll(self): return 1 if incident=='exit' else 0
        def terminate(self):
            calls.append('terminate')
            if incident=='mutation': (self.root/'comparison.pending.json').write_bytes(packet+b' ')
            return incident!='cleanup'
        def close(self): calls.append('close')
    monkeypatch.setattr(probe,'_ProcessJob',Job); output=tmp_path/'fresh'
    if incident=='success':
        probe.run('1'*40,output)
        assert (output/'comparison.json').read_bytes()==packet
        with pytest.raises(FileExistsError): probe.run('1'*40,output)
    else:
        with pytest.raises((ValueError,RuntimeError)): probe.run('1'*40,output)
        assert not (output/'comparison.json').exists()
    assert calls.count('terminate')==calls.count('close')==1


def test_input_mutation_rejected_before_model_creation(tmp_path,monkeypatch):
    path=tmp_path/'input.json'; path.write_bytes(b'{}\n'); monkeypatch.setattr(probe,'INPUT',path)
    with pytest.raises(ValueError,match='six-macro input changed'): probe.inputs()
