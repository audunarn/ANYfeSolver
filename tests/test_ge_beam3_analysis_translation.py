from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis, NativeBeamAnalysisError, NativeBeamWorkflowError
from anyfesolver import _ge_beam3_native_definition as codec
from anyfesolver import _ge_beam3_analysis_translation as route
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_precise_geometric_work import POLICY
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken
from test_ge_beam3_native_analysis import analysis, definitions
from test_ge_beam3_elastic_seed_continuation import seed, model, programme
from test_ge_beam3_native_generalized_modal import make
from test_ge_beam3_native_fibre_static_element import problem as fibre_problem


def digest(raw):return sha256(raw).hexdigest()


@pytest.mark.parametrize('curved',(False,True))
def test_precise_definition_roundtrip_preserves_exact_operator(curved):
    m,_,_=make(curved,clamped=True);old=m.mesh.elements[1]
    precise=NativeGeneralizedStaticElement(1,tuple(old.node_ids),old.operator.reference,old.section,order=4,arithmetic_policy=POLICY)
    before=codec.NativeBeamDefinition.capture(old,np.eye(6));definition=codec.NativeBeamDefinition.capture(precise,np.eye(6))
    assert json.loads(before.raw)['schema']==codec.SCHEMA and 'arithmetic_policy' not in json.loads(before.raw)
    assert json.loads(definition.raw)['schema']==codec.PRECISE_SCHEMA
    rebuilt,_=definition.instantiate()
    assert rebuilt.identity==precise.identity and rebuilt.operator.arithmetic_policy==POLICY
    assert codec.NativeBeamDefinition.capture(rebuilt,np.eye(6)).raw==definition.raw
    assert codec.NativeBeamDefinition.capture(old,np.eye(6)).raw==before.raw


@pytest.mark.parametrize('kind',('schema','policy','family','missing','old_extra'))
def test_precise_policy_cannot_be_lost_or_relabelled(kind):
    m=model();old=m.mesh.elements[1]
    p=NativeGeneralizedStaticElement(1,tuple(old.node_ids),old.operator.reference,old.section,order=4,arithmetic_policy=POLICY)
    v=json.loads(codec.NativeBeamDefinition.capture(p,np.eye(6)).raw)
    if kind=='schema':v['schema']=codec.SCHEMA
    elif kind=='policy':v['arithmetic_policy']='arbitrary'
    elif kind=='family':v['family']='PHYSICAL_AXIAL_BIAXIAL_FIBRE'
    elif kind=='missing':v.pop('arithmetic_policy')
    else:
        v=json.loads(codec.NativeBeamDefinition.capture(old,np.eye(6)).raw);v['arithmetic_policy']=None
    with pytest.raises(ValueError):codec.NativeBeamDefinition(canonical(v))


@pytest.mark.parametrize('seeded',(False,True))
def test_actual_owned_continuation_import_resume_recover(seed,seeded,tmp_path):
    m=model();made=analysis(m);program=programme();kw=dict(seed=seed,expected_seed_sha256=digest(seed)) if seeded else {}
    whole=made.solve_translation(program,**kw)
    assert whole.status=='completed',whole.backend_result.failure
    backend=whole.backend_result.checkpoint
    direct=(route.seeded.solve(m,program,seed,expected_seed_sha256=digest(seed)) if seeded else route.ordinary.solve(m,program))
    assert backend==direct.checkpoint and canonical(whole.backend_result.state)==canonical(direct.state)
    imported=analysis(m).import_translation_checkpoint(program,backend,expected_sha256=digest(backend),**kw)
    assert imported==whole.checkpoint
    prefix=made.translation_checkpoint_prefix(program,whole.checkpoint,2,expected_sha256=whole.checkpoint_sha256,**kw)
    resumed=analysis(m).solve_translation(program,checkpoint=prefix,expected_sha256=digest(prefix),**kw)
    assert resumed.status=='completed' and resumed.checkpoint==whole.checkpoint
    recovery=made.recover_translation(program,whole.checkpoint,expected_sha256=whole.checkpoint_sha256,**kw)
    assert canonical(recovery)==canonical(analysis(m).recover_translation(program,imported,expected_sha256=digest(imported),**kw))
    with pytest.raises(ValueError):made.recover(whole.checkpoint,expected_sha256=whole.checkpoint_sha256)
    (tmp_path/'checkpoint.json').write_bytes(whole.checkpoint)
    (tmp_path/'recovery.json').write_bytes(canonical(recovery))


@pytest.mark.parametrize('when',('before_commit','committed'))
def test_actual_owned_cancel_then_fresh_resume(seed,when):
    made=analysis(model());p=programme();token=CancellationToken()
    def progress(row):
        if row['stage']=='elastic-continuation.'+when:token.cancel()
    kw=dict(seed=seed,expected_seed_sha256=digest(seed))
    cancelled=made.solve_translation(p,cancellation_token=token,progress=progress,**kw)
    assert cancelled.status=='cancelled' and cancelled.backend_result.completed_targets==(0 if when=='before_commit' else 1)
    resumed=analysis(model()).solve_translation(p,checkpoint=cancelled.checkpoint,expected_sha256=cancelled.checkpoint_sha256,**kw)
    assert resumed.status=='completed'
    assert resumed.checkpoint==analysis(model()).solve_translation(p,**kw).checkpoint


@pytest.fixture(scope='module')
def accepted(seed):
    p=programme();made=analysis(model());r=made.solve_translation(p,seed=seed,expected_seed_sha256=digest(seed),stop_after=1)
    assert r.status=='paused';return r


@pytest.mark.parametrize('kind',('graph','workflow','program','seed','backend_hash','schema','qualification','extra','duplicate','nonfinite','overflow','whitespace'))
def test_envelope_rejects_before_native_context(seed,accepted,monkeypatch,kind):
    v=json.loads(accepted.checkpoint)
    if kind=='graph':v['definition_graph_sha256']='0'*64
    elif kind=='workflow':v['workflow']='RETAINED_FROM_REFERENCE'
    elif kind=='program':v['program']['targets']=[.2]
    elif kind=='seed':v['seed_sha256']='0'*64
    elif kind=='backend_hash':v['backend_sha256']='0'*64
    elif kind=='schema':v['schema']='GE_BEAM3_MODEL_OWNED_ANALYSIS_CHECKPOINT_V1'
    elif kind=='qualification':v['production_qualified']=True
    elif kind=='extra':v['extra']=None
    raw=canonical(v)
    if kind=='duplicate':raw=raw.replace(b'{',b'{"schema":"duplicate",',1)
    elif kind=='nonfinite':raw=raw.replace(b'false',b'NaN',1)
    elif kind=='overflow':raw=raw.replace(b'false',b'1e9999',1)
    elif kind=='whitespace':raw+=b' '
    monkeypatch.setattr(route,'_context',lambda *a: (_ for _ in ()).throw(AssertionError('native context entered')))
    with pytest.raises(ValueError):
        analysis(model()).recover_translation(programme(),raw,expected_sha256=digest(raw),seed=seed,expected_seed_sha256=digest(seed))


def test_wrong_inertia_graph_and_backend_rehash_rejected(seed,accepted):
    m=model();different=analysis(m,{1:2*np.eye(6)})
    kw=dict(seed=seed,expected_seed_sha256=digest(seed))
    with pytest.raises(ValueError,match='owner mismatch'):
        different.recover_translation(programme(),accepted.checkpoint,expected_sha256=accepted.checkpoint_sha256,**kw)
    v=json.loads(accepted.backend_result.checkpoint);v['records'][0]['recovery_sha256']='0'*64
    # Updating outer digest cannot bypass the native record reconstruction.
    raw=canonical(v)
    with pytest.raises(ValueError):analysis(m).import_translation_checkpoint(programme(),raw,expected_sha256=digest(raw),**kw)


def test_workflow_controls_and_explicit_capacity_fail_closed(seed):
    made=analysis(model())
    with pytest.raises(NativeBeamWorkflowError):analysis(fibre_problem()[0]).solve_translation(programme())
    with pytest.raises(ValueError):made.solve_translation(programme(),seed=seed)
    with pytest.raises(ValueError):made.solve_translation(programme(),stop_after=True)
    with pytest.raises(ValueError):made.solve_translation(programme(),progress='bad')
    with pytest.raises(ValueError):NativeBeamAnalysis(definitions(model()),(),retained_refinement=1)
    made._retained_refinement=True
    with pytest.raises(ValueError,match='model changed'):made.solve_translation(programme())


def test_observer_model_mutation_is_rejected_and_lock_released(seed):
    made=analysis(model());p=programme()
    def changed(row):made._inertias[1]=2*made._inertias[1]
    with pytest.raises(ValueError,match='definition/inertia'):
        made.solve_translation(p,seed=seed,expected_seed_sha256=digest(seed),progress=changed)
    assert not made._lock.locked()
