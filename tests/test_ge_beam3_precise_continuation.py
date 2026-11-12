from copy import deepcopy
from hashlib import sha256
import pytest
from anyfesolver import _ge_beam3_elastic_seed_continuation as owner
from anyfesolver.control import CancellationToken
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from docs.reference_cases.ge_beam3_precise_continuation import recipe,adjudicate,source,ARCHIVE,INPUTS
from docs.reference_cases.ge_beam3_precise_work_enrollment import enroll
from test_ge_beam3_precise_work_enrollment import original,precise_model
from test_ge_beam3_elastic_seed_continuation import seed,programme


@pytest.mark.parametrize('operation',('advance','cancel-before','cancel-after','resume-before'))
def test_frozen_dispositions(operation):
    status,cursor,event=recipe(operation)
    observed=[] if event is None else [dict(stage=event,target=1,iteration=2)]
    adjudicate(operation,status,cursor,observed)
    for bad in ('status','cursor','bool','event','iteration'):
        s,c,rows=status,cursor,deepcopy(observed)
        if bad=='status':s='failed'
        elif bad=='cursor':c+=1
        elif bad=='bool':c=bool(c)
        elif bad=='event':rows=[dict(stage='unobserved',target=1,iteration=2)]
        else:rows=[dict(stage=event,target=1,iteration=25)]
        with pytest.raises(ValueError):adjudicate(operation,s,c,rows)
    with pytest.raises(ValueError):recipe('retry')


@pytest.mark.parametrize('sign',('plus','minus'))
def test_exact_source_binding_and_mutation(sign,tmp_path):
    expected=source(sign)
    manifest=(ARCHIVE/'manifest.json').read_bytes()
    (tmp_path/'manifest.json').write_bytes(manifest)
    folder=tmp_path/'wave'/(sign+'-a')/'output/diagnostics';folder.mkdir(parents=True)
    for name,raw in expected.items():(folder/name).write_bytes(raw)
    assert source(sign,tmp_path)==expected
    (folder/'checkpoint.json').write_bytes(expected['checkpoint.json']+b' ')
    with pytest.raises(ValueError):source(sign,tmp_path)
    with pytest.raises(ValueError):source('not-registered',tmp_path)


def test_actual_precise_commit_boundaries_and_resume(original):
    p=programme((.0003,));context,new_seed=enroll(precise_model(),p,original,sha256(original).hexdigest())
    initial=context.checkpoint(());digest=sha256(new_seed).hexdigest()
    def run(prior,progress=None,token=None):
        return owner.solve(precise_model(),p,new_seed,expected_seed_sha256=digest,checkpoint=prior,
            expected_sha256=sha256(prior).hexdigest(),progress=progress,cancellation_token=token)
    complete=run(initial);assert complete.status=='completed',complete.failure
    for operation in ('cancel-before','cancel-after'):
        token=CancellationToken();events=[];event=recipe(operation)[2]
        def progress(row):
            if row['stage']==event:events.append(dict(row));token.cancel()
        cancelled=run(initial,progress,token)
        adjudicate(operation,cancelled.status,cancelled.completed_targets,events)
        assert cancelled.checkpoint==(initial if operation=='cancel-before' else complete.checkpoint)
        resumed=run(cancelled.checkpoint)
        assert resumed.status=='completed' and resumed.checkpoint==complete.checkpoint
    replay=owner.Context(precise_model(),p,new_seed,expected_seed_sha256=digest)
    state,records=replay.restore(complete.checkpoint,expected_sha256=sha256(complete.checkpoint).hexdigest())
    before=canonical(state);replay.recover(state)
    assert canonical(state)==before and replay.checkpoint(records)==complete.checkpoint
