"""Initializer authority/admission tests, not spatial mechanics evidence."""
import pytest
from docs.reference_cases import ge_beam3_spatial_equilibrium_probe as p

@pytest.mark.parametrize('value',(.003,-.003,.006,-.006))
def test_registered_amplitudes(value):assert p.amplitude(value)==value

@pytest.mark.parametrize('value',(0.,True,1,float('nan'),float('inf'),.01))
def test_unregistered_amplitudes(value):
    with pytest.raises(ValueError):p.amplitude(value)

def test_source_exact_hash_bound():
    raw=p.source_bytes();assert len(raw)==432367 and p.sha256(raw).hexdigest()==p.SOURCE_SHA

def test_changed_manifest_rejected(tmp_path):
    (tmp_path/'archive-manifest.json').write_bytes(b'{}\n')
    with pytest.raises(ValueError,match='manifest'):p.source_bytes(tmp_path)

def test_no_invented_history():
    p.require_elastic(({'plastic':[0.,0.]},),({'plastic':[0.,0.]},))
    with pytest.raises(ValueError,match='history'):
        p.require_elastic(({'plastic':[.1,0.]},),({'plastic':[0.,0.]},))

def test_trial_border_does_not_weaken_virgin_history_owner():
    import numpy as np
    from anyfesolver import _ge_beam3_retained_translation_control as control
    from docs.reference_cases.ge_beam3_refined_controlled_case import model
    programme=control.Program((.003,),2,(0.,0.,1.),control.NodalDeadForces(((3,0.,-1.,0.),)))
    with pytest.raises(np.linalg.LinAlgError):control.Context(model(2),programme)
    trial=p.elastic_control(model(2),programme);trial.guard()
    assert all(not hasattr(trial,name) for name in ('initial','genesis','stage','checkpoint','restore','_record'))
    assert trial.value(trial.physical.initial.mechanical)==0.

@pytest.mark.parametrize('kind',('row','node','model'))
def test_trial_control_guard(kind):
    from anyfesolver import _ge_beam3_retained_translation_control as control
    from docs.reference_cases.ge_beam3_refined_controlled_case import model
    trial=p.elastic_control(model(2),control.Program((.003,),2,(0.,0.,1.),control.NodalDeadForces(((3,0.,-1.,0.),))))
    if kind=='row':trial.row=-trial.row
    elif kind=='node':trial.node=2
    else:trial.physical.model.mesh.nodes[2].y+=.001
    with pytest.raises(ValueError):trial.guard()
