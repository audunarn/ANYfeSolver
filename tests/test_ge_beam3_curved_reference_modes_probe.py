"""Bounded spectral supervisor mocks; no native spectral solves."""
from pathlib import Path
import pytest
from docs.reference_cases import ge_beam3_curved_reference_modes_probe as probe


@pytest.mark.parametrize('incident',['success','exit','memory','timeout','inactivity','invalid','cleanup','descendant','mutation'])
def test_spectral_cleanup_and_validation_precede_publication(tmp_path,monkeypatch,incident):
    calls=[]; packet=probe.canonical(dict(development=True))
    ticks=iter([0.,601. if incident=='timeout' else 121. if incident=='inactivity' else 1.,601.])
    monkeypatch.setattr(probe.time,'monotonic',lambda:next(ticks)); monkeypatch.setattr(probe.time,'sleep',lambda _:None)
    monkeypatch.setattr(probe,'guard',lambda _:calls.append('guard'))
    def validate(output,raw,revision):
        calls.append('validate'); assert raw==packet
        if incident=='invalid': raise ValueError('invalid')
    monkeypatch.setattr(probe,'validate',validate)
    class Job:
        def __init__(self,memory): assert memory==24*(1<<30)
        def launch(self,command,*,cwd,env,stdout,stderr):
            assert command[3]=='docs.reference_cases.ge_beam3_curved_reference_modes_probe'
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


@pytest.mark.parametrize('raw',[b'',b'{"x":1,"x":2}\n',b'{"x":NaN}\n',b'{"x":Infinity}\n',b' {"x":1}\n'])
def test_noncanonical_spectral_packets_reject(tmp_path,raw):
    path=tmp_path/'bad.json'; path.write_bytes(raw)
    with pytest.raises(ValueError): probe.read(path)


def test_frozen_and_higher_kinetic_rules_are_not_conflated(monkeypatch):
    import numpy as np
    from docs.reference_cases import ge_beam3_curved_reference_modes as comparison
    from docs.reference_cases import ge_beam3_curved_moment_fixture as fixture
    from anyfesolver._ge_beam3_lifted_kinetic_factor import current_lifted_kinetic_factor
    if not comparison.REFERENCE_PATH.is_file(): pytest.skip('Preserved development reference not installed')
    def forbidden(*args,**kwargs): raise AssertionError('reference solve must not be rerun')
    monkeypatch.setattr(comparison.ritz,'solve',forbidden)
    _,references=comparison.preserved_reference(); reference=references[-1]
    model=fixture.model(1); source=model.mesh.elements[1].operator.reference
    b=current_lifted_kinetic_factor(source,comparison.INERTIA,np.tile(np.eye(3),(2,1,1)),order=4)
    basis=np.eye(24)[:,[6,7,8,18,19,20]]
    root=np.linalg.cholesky((b@basis).T@(b@basis)); vectors=np.linalg.solve(root,basis.T).T
    same=comparison.modal_overlap(1,vectors,reference,quadrature=4)
    fine=comparison.modal_overlap(1,vectors,reference,quadrature=16)
    finer=comparison.modal_overlap(1,vectors,reference,quadrature=32)
    assert np.max(abs(same['native_mass_gram']-np.eye(6)))<1e-11
    assert np.max(abs(fine['native_mass_gram']-same['native_mass_gram']))>1e-10
    np.testing.assert_allclose(finer['native_mass_gram'],fine['native_mass_gram'],atol=1e-11,rtol=1e-11)
    assert np.max(abs(fine['reference_mass_gram']-np.eye(6)))<1e-8


def test_preserved_reference_hash_mutation_rejects(tmp_path,monkeypatch):
    from docs.reference_cases import ge_beam3_curved_reference_modes as comparison
    path=tmp_path/'changed.json'; path.write_bytes(b'{}\n'); monkeypatch.setattr(comparison,'REFERENCE_PATH',path)
    with pytest.raises(ValueError,match='identity'): comparison.preserved_reference()
