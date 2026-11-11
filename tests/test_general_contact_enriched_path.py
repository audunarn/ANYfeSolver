"""Actual public persistence and full registered path/reference delivery."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.e4_pl_enriched import EnrichedProfileError
from anyfesolver.control import SolveCancelled
from scripts.general_contact_enriched_path_run import equal_tree,complete_paths


def test_recursive_full_checkpoint_correspondence_and_typed_scope():
    a=dict(profile='new',records=[dict(coordinates=[0.,1.],states={'43':[2.]})])
    assert equal_tree(a,json.loads(json.dumps(a)))
    b=json.loads(json.dumps(a));b['records'][0]['coordinates'][1]+=2e-10
    assert not equal_tree(a,b)
    b=json.loads(json.dumps(a));b['profile']='legacy'
    assert not equal_tree(a,b)
    assert not equal_tree({'x':float('nan')},{'x':float('nan')})


def test_public_candidate_failure_retains_genuine_prefix(tmp_path,monkeypatch):
    # Pure public persistence caller control: fake owner is explicitly synthetic,
    # while the actual public run/publish/exception envelope is exercised.
    from examples import general_contact_enriched_d1_path as public
    class Owner:
        accepted_step=0
        class Program:total_steps=8
        program=Program()
        @property
        def _state(self):return None if self.accepted_step==0 else self.checkpoint()
        def restore(self,*a,**k):raise AssertionError('not a restart control')
        def checkpoint(self):
            return json.dumps(dict(accepted_step=self.accepted_step,records=[{'step_index':i} for i in range(1,self.accepted_step+1)])).encode()
        def advance(self,*,cancel,before_publish):
            step=self.accepted_step+1
            proposed=json.dumps(dict(accepted_step=step,records=[{'step_index':i} for i in range(1,step+1)])).encode()
            before_publish(proposed)
            if cancel():raise SolveCancelled('late cancellation')
            self.accepted_step=step
            return dict(step_index=step)
    owner=Owner();monkeypatch.setattr(public,'build',lambda mode:(None,None,owner,{}))
    cancelled=[False]
    def late(actual,candidate):
        if json.loads(candidate)['accepted_step']==2:cancelled[0]=True
    output=tmp_path/'public'
    with pytest.raises(SolveCancelled):public.run(output,through=3,cancel=lambda:cancelled[0],on_candidate=late)
    assert owner.accepted_step==1
    assert json.loads((output/'checkpoint.json').read_bytes())['accepted_step']==1
    assert json.loads((output/'candidate-02.json').read_bytes())['status']=='CANDIDATE_NOT_COMMITTED'
    assert not (output/'commit-02.json').exists()
    assert json.loads((output/'failure.json').read_bytes())['owner_accepted_step']==1


def test_registered_full364_force8_displacement8_authentic2plus6_public_delivery(record_property, tmp_path):
    root=Path(os.environ.get('ENRICHED_REFERENCE_OUTPUT', tmp_path))/'full364'
    result=complete_paths(root)
    assert result['status']=='PASS_SCOPED_V16_PUBLIC_PATH_DELIVERY_ONLY'
    assert result['force_poses']==result['displacement_poses']==8
    assert result['authentic_active_prefix']==2 and result['restored_new_poses']==6
    assert result['every_pose_correspondence'] and result['full_checkpoint_correspondence']
    assert result['extended_full364_work_reference']=='PREPARED_REQUIRED_OPEN_UNRUN'
    assert not (root/'force-endpoint-reference').exists()
    record_property('assessment',str(root/'ASSESSMENT.json'))
    record_property('assessment_sha256',hashlib.sha256((root/'ASSESSMENT.json').read_bytes()).hexdigest())
