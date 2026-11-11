from hashlib import sha256
import json
import pytest
from anyfesolver import _ge_beam3_elastic_seed_continuation as owner
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement as Element
from anyfesolver._ge_beam3_precise_geometric_work import POLICY
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from docs.reference_cases.ge_beam3_precise_work_enrollment import enroll
from test_ge_beam3_elastic_seed_continuation import seed,model,programme


def precise_model():
    made=model()
    for eid,e in tuple(made.mesh.elements.items()):
        made.mesh.elements[eid]=Element(eid,e.node_ids,e.operator.reference,e.section,
            order=e.operator.order,arithmetic_policy=POLICY)
    return made


@pytest.fixture(scope='module')
def original(seed):
    result=owner.solve(model(),programme((.0002,)),seed,expected_seed_sha256=sha256(seed).hexdigest())
    assert result.status=='completed',result.failure
    return result.checkpoint


def test_new_owner_enrollment_and_foreign_restart_rejection(original,seed):
    digest=sha256(original).hexdigest();p=programme((.0003,))
    context,new_seed=enroll(precise_model(),p,original,digest)
    assert json.loads(new_seed)['source_sha256']==digest
    raw=context.checkpoint(());before=canonical(context.initial);recovery=canonical(context.recover(context.initial))
    state,records=context.restore(raw,expected_sha256=sha256(raw).hexdigest())
    assert not records and canonical(state)==before and canonical(context.recover(state))==recovery
    assert context.checkpoint(())==raw and sha256(original).hexdigest()==digest
    with pytest.raises(ValueError):context.restore(original,expected_sha256=digest)
    with pytest.raises(ValueError):owner.Context(precise_model(),p,seed,expected_seed_sha256=sha256(seed).hexdigest())
    with pytest.raises(ValueError):owner.Context(model(),p,new_seed,expected_seed_sha256=sha256(new_seed).hexdigest())


@pytest.mark.parametrize('kind',('hash','schema','parameter','mechanical','origin','control','policy'))
def test_enrollment_mutation_rejection(original,kind):
    v=json.loads(original);p=programme((.0003,));made=precise_model();digest=None
    if kind=='hash':digest='0'*64
    elif kind=='schema':v['schema']='foreign'
    elif kind=='parameter':v['records'][-1]['parameter']+=.1
    elif kind=='mechanical':v['records'][-1]['mechanical']['positions'][2][0]+=.01
    elif kind=='origin':v['records'][-1]['origins'][0]['stations'][0]['accumulated'][0]=.1
    elif kind=='control':v['program']['control_node']=1
    else:made=model()
    v['checkpoint_sha256']=sha({k:x for k,x in v.items() if k!='checkpoint_sha256'})
    raw=canonical(v)
    with pytest.raises(ValueError):enroll(made,p,raw,digest or sha256(raw).hexdigest())
