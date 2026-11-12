"""Bounded staged lifecycle of the verified fine-policy native/S3 candidate."""
from pathlib import Path
from hashlib import sha256
import json,os
import numpy as np
import pytest
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_general_contact_native_curved_s3_static import _analysis
from test_general_contact_native_fibre_s3 import _independent_material_history
from test_general_contact_native_fibre_enforcement import _gap
from test_ge_beam3_fibre_section import exact_pair


def _owner():
    return _analysis(physical_fibres=True,plastic_fibres=True,normal_stiffness_Pa_per_m=8e7,
        integration_budget_policy='certified_active',root_tangent_enclosure_policy='componentwise')


def _anchored_owner():
    return _analysis(physical_fibres=True,plastic_fibres=True,normal_stiffness_Pa_per_m=8e7,
        integration_budget_policy='certified_active',root_tangent_enclosure_policy='anchored')


@pytest.mark.parametrize('stage',(2,3),ids=('opening','recontact'))
def test_anchored_fine_policy_staged_lifecycle(stage):
    if stage==2:
        name='active';digest='98d22bd2aff840df5d7b1034298891b179ed438ec9246a9f76b89445a7a15347'
    else:
        name='open';digest='0ccdfa3cf637d12ca36051e315bd092cc61c53ac9b084676f8f99fac7e092d6e'
    raw=(Path(__file__).parent/f'data/general_contact_native_fibre_s3_plastic_k8e7_anchored_{name}_v1.checkpoint').read_bytes()
    assert sha256(raw).hexdigest()==digest
    _check_lifecycle(stage,raw,_anchored_owner)


def _check_lifecycle(stage,raw,owner_factory):
    """Identical lifecycle gates for an explicitly bound checkpoint/owner."""
    prefix=json.loads(raw);assert prefix['completed_targets']==stage-1
    owner=owner_factory()
    origins=owner.context.histories(prefix['records'][-1]['histories'],decoded=True)
    before=canonical(origins)
    run=owner.solve(checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),stop_after=stage)
    output=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if output:
        Path(output+'-fine-lifecycle.checkpoint').write_bytes(run.checkpoint)
        Path(output+'-fine-lifecycle.json').write_text(json.dumps(dict(stage=stage,status=run.status,
            failure=run.failure,completed=run.accepted.completed_targets,candidates=owner.candidate_evaluations,
            pair_evaluations=owner.pair_evaluations,reuse_hits=owner.contact_reuse_hits,
            trial_failures=owner.curved_trial_failures,checkpoint_sha256=run.checkpoint_sha256),indent=2))
    assert run.status==('paused' if stage==2 else 'completed'),run.failure
    assert run.accepted.completed_targets==stage
    contact=run.accepted.contact.contact
    independent_gap=_gap(owner,run.accepted,129)
    assert contact.minimum_gap_m<=independent_gap+1e-10
    if stage==2:
        assert contact.minimum_gap_m>0. and independent_gap>0.
        assert contact.potential_j==0. and np.linalg.norm(contact.beam_force_n)==0.
    else:
        assert contact.potential_j>0. and np.linalg.norm(contact.beam_force_n)>0.
        assert contact.minimum_gap_m>=-3e-5 and independent_gap>=-3e-5
    assert np.linalg.norm(contact.beam_force_n+contact.shell_force_n)/.005<=1e-6
    assert np.linalg.norm(contact.beam_moment_nm+contact.shell_moment_nm)/.010<=1e-6
    report,proposed=_independent_material_history(owner,run.accepted.mechanical,origins)
    assert tuple(proposed)==run.accepted.native_histories[0].stations and canonical(origins)==before
    for current,previous in zip(proposed,origins[0].stations):
        for new,old in zip(current.rows,previous.rows):assert exact_pair(new[2:])>=exact_pair(old[2:])
    restored,records=owner_factory()._restore(run.checkpoint,run.checkpoint_sha256)
    assert restored.native_histories==run.accepted.native_histories
    assert owner._checkpoint(records)==run.checkpoint
    assert canonical(records[:stage-1])==canonical(prefix['records'])
    assert owner.context._token is None and not owner.context._lock.locked()
    assert owner._contact_reuse_entry is None and not owner._contact_reuse_enabled
    if output:Path(output+'-fine-history.json').write_text(json.dumps(dict(material=report,
        independent_gap_m=independent_gap,production_gap_lower_m=contact.minimum_gap_m),indent=2))
