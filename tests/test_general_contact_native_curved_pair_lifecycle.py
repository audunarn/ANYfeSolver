"""Bounded staged press/open/recontact with authentic preceding material origins."""
import json
import os
import runpy
import time
from pathlib import Path
from hashlib import sha256
from dataclasses import replace
from math import isfinite

import numpy as np
import pytest

from anyfesolver._ge_beam3_p5_seeded.codec import _load
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken
from test_general_contact_native_fibre_s3 import _independent_material_history
from anyfesolver.general_contact.native_replay_diagnostics import replay_difference_message

ROOT=Path(__file__).parents[1]
EVIDENCE=Path(os.environ.get('CONTACT_NATIVE_CURVED_LIFECYCLE_EVIDENCE',
                            'C:/ANYworkspaceWork/contact_native_fibre_s3_20261005/candidate26'))
TARGETS=(.05,0.,.05)
_PRESS_SHA256='a37d8ebe66341e8dc48e83dbc99939153280a8f1a81c3812212c04d4adc6a8ca'
_PRESS_PROVENANCE_SHA256='a32e0997ab8f5808b0bbd2b95b4dc8114094fa6bf1519b9d2fa2357edff70075'
_PRESS_RECORD_SHA256='37d6c1c1c136f277957fd0cc4dc8e6f6744740d071cdef14a8da55c135bbd7ff'
_TEMPLATE_IDENTITIES={}
_MAX_TEMPLATE_IDENTITIES=8


def _template_options_key(options):
    """Cache only ordinary exact builtin inputs; all other inputs use the old path."""
    boolean=('physical_fibres','plastic_fibres','gap_step_control',
             'certify_curved_ownership','reuse_target_trials')
    scalar=('separation','force','line_stiffness_n_per_m2','gap_tolerance_m','reference_rise_m')
    if type(options) is not dict or set(options)!=set(boolean+scalar+('targets',)):
        return None
    if any(type(options[name]) is not bool for name in boolean):
        return None
    if any(type(options[name]) is not float or not isfinite(options[name]) for name in scalar):
        return None
    targets=options['targets']
    if (type(targets) is not tuple or len(targets)>32
            or any(type(value) is not float or not isfinite(value) for value in targets)):
        return None
    return (tuple((name,'bool',options[name]) for name in boolean)
            +tuple((name,'float',options[name].hex()) for name in scalar)
            +(('targets','tuple',tuple(('float',value.hex()) for value in targets)),))


def _build_owner_with_template_identity(build,options,seed_path):
    """Reuse only a string, after a successful fresh actual-owner construction.

    Source is frozen for this process. No owner/context/contact/prepared result
    escapes into this bounded cache; every actual owner retains its own admission,
    lock, cancellation state, history and construction-to-expiry allowance.
    """
    def original_path():
        template=build(**options)
        frozen=_load(seed_path.read_text())
        return build(**options,initial_trial=dict(native_model=template.context.identity,
                      mechanical=frozen['records'][-1]['mechanical']))
    option_key=_template_options_key(options)
    if option_key is None:
        # Preserve the original validation path for nonstandard or invalid input.
        return original_path()
    try:
        seed_raw=seed_path.read_bytes()
    except OSError:
        # A missing/unreadable seed must not precede the original builder's
        # validation failure merely because cache lookup needs a byte digest.
        return original_path()
    key=(option_key,sha256(seed_raw).hexdigest())
    identity=_TEMPLATE_IDENTITIES.get(key)
    if identity is None:
        identity=build(**options).context.identity
    frozen=_load(seed_raw.decode('ascii'))
    owner=build(**options,initial_trial=dict(native_model=identity,
                mechanical=frozen['records'][-1]['mechanical']))
    # Failed decoding/template/actual construction cannot populate the cache.
    if type(identity) is str and len(_TEMPLATE_IDENTITIES)<_MAX_TEMPLATE_IDENTITIES:
        _TEMPLATE_IDENTITIES[key]=identity
    return owner


def _owner(*,reuse_target_trials=True,targets=TARGETS):
    build=runpy.run_path(str(ROOT/'examples/general_contact_native_retained_whole_pair.py'))['analysis']
    options=dict(physical_fibres=True,plastic_fibres=True,targets=targets,
                 separation=.101,force=-.05,line_stiffness_n_per_m2=8000.,
                 gap_tolerance_m=5e-5,gap_step_control=True,reference_rise_m=.005,
                 certify_curved_ownership=True,reuse_target_trials=reuse_target_trials)
    return _build_owner_with_template_identity(build,options,
        ROOT/'tests/data/general_contact_native_curved_fibre_pair_seeded_k8_v1.checkpoint')


def _prefix(stage):return EVIDENCE/f'lifecycle_stage{stage}.checkpoint'


def _press_fixture():
    return ROOT/'tests/data/general_contact_native_curved_pair_lifecycle_press_v1.checkpoint'


def _press_fixture_bytes():
    raw=_press_fixture().read_bytes()
    provenance_raw=_press_fixture().with_name(
        'general_contact_native_curved_pair_lifecycle_press_v1_provenance.json').read_bytes()
    assert sha256(raw).hexdigest()==_PRESS_SHA256
    assert sha256(provenance_raw).hexdigest()==_PRESS_PROVENANCE_SHA256
    provenance=json.loads(provenance_raw)
    assert provenance['provisional'] is True and provenance['coupled_solve_rerun'] is False
    assert provenance['checkpoint_sha256']==_PRESS_SHA256
    assert provenance['record_sha256']==_PRESS_RECORD_SHA256
    assert sha256(canonical(_load(raw.decode('ascii'))['records'][0])).hexdigest()==_PRESS_RECORD_SHA256
    return raw


@pytest.mark.skipif(not os.environ.get('CONTACT_NATIVE_CURVED_LIFECYCLE_STAGE'),
                    reason='explicit bounded lifecycle stage')
def test_staged_curved_pair_press_open_recontact():
    stage=int(os.environ['CONTACT_NATIVE_CURVED_LIFECYCLE_STAGE'])
    assert stage in (1,2,3)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    owner=_owner();assert owner.context.count==84
    output=EVIDENCE/f'lifecycle_stage{stage}.json'
    contacts=[];actual=owner._contact
    def measured(state):
        started=time.perf_counter()
        try:
            result=actual(state)
            contacts.append(dict(gap=float(result.conservative_lower_gap_m),
                                 potential=result.potential_j,elapsed_s=time.perf_counter()-started))
            return result
        finally:
            output.with_suffix('.contacts.json').write_text(json.dumps(contacts,indent=2))
    owner._contact=measured
    if stage==1:
        preceding=owner.initial.histories
        checkpoint=None
    else:
        checkpoint=_prefix(stage-1).read_bytes()
        row=_load(checkpoint.decode('ascii'))['records'][-1]
        preceding=owner.context.histories(row['histories'],decoded=True)
    before=canonical(preceding)
    started=time.perf_counter()
    run=owner.solve(checkpoint=checkpoint,
                    expected_sha256=sha256(checkpoint).hexdigest() if checkpoint else None,
                    stop_after=stage)
    # Persist the accepted capsule before any potentially costly postprocessing.
    _prefix(stage).write_bytes(run.checkpoint)
    if checkpoint:
        assert canonical(_load(run.checkpoint.decode('ascii'))['records'][:stage-1]) == canonical(
            _load(checkpoint.decode('ascii'))['records'])
    report=dict(stage=stage,status=run.status,failure=run.failure,
                completed=run.accepted.completed_targets,elapsed_s=time.perf_counter()-started,
                checkpoint_sha256=run.checkpoint_sha256,contacts=contacts)
    output.write_text(json.dumps(report,indent=2))
    assert run.status==('completed' if stage==3 else 'paused'),run.failure
    assert run.accepted.completed_targets==stage
    contact=run.accepted.contact
    assert contact.conservative_lower_gap_m>=-5e-5
    force_error=float(np.linalg.norm(contact.beam_a_force_n+contact.beam_b_force_n)/.0025)
    moment_error=float(np.linalg.norm(contact.beam_a_moment_nm+contact.beam_b_moment_nm)/.005)
    assert max(force_error,moment_error)<=1e-6
    material,proposed=_independent_material_history(owner,run.accepted.mechanical,preceding)
    assert proposed==tuple(station for cell in run.accepted.histories for station in cell.stations)
    assert canonical(preceding)==before
    report.update(material=material,force_error=force_error,moment_error=moment_error,
                  gap_lower_m=float(contact.conservative_lower_gap_m),potential_j=contact.potential_j,
                  force_a_n=contact.beam_a_force_n.tolist(),force_b_n=contact.beam_b_force_n.tolist(),
                  moment_a_nm=contact.beam_a_moment_nm.tolist(),moment_b_nm=contact.beam_b_moment_nm.tolist(),
                  actual_contact='active' if contact.potential_j>0 else 'open',
                  positions=(run.accepted.mechanical.positions+run.accepted.mechanical.position_low).tolist())
    output.write_text(json.dumps(report,indent=2))
    assert owner.context._token is None and not owner.context._lock.locked()
    if stage in (1,3):assert contact.potential_j>0
    # An actual failure to open is classified from the accepted mechanics.
    if stage==2:
        assert contact.potential_j==0 and contact.conservative_lower_gap_m>0,report
    print('lifecycle_stage',report)


def test_repeated_trial_identity_mutation_and_incompatible_bindings():
    owner=_owner();raw=_press_fixture_bytes()
    with pytest.raises(ValueError,match='external checkpoint hash mismatch'):
        owner._restore(raw,'0'*64)
    tampered=_load(raw.decode('ascii'))
    tampered['records'][0]['contact']['energy_error_j']=1.
    tampered_raw=canonical(tampered)
    with pytest.raises(ValueError,match='definition/settings binding'):
        owner._restore(tampered_raw,sha256(tampered_raw).hexdigest())
    for value in (1,'yes'):
        owner.reuse_target_trials=value
        with pytest.raises(ValueError,match='setting changed'):owner._guard()
    owner.reuse_target_trials=True;owner._guard()
    for other in (_owner(reuse_target_trials=False),_owner(targets=(.05,0.,.06))):
        with pytest.raises(ValueError,match='binding'):
            other._restore(raw,sha256(raw).hexdigest())


def test_curved_owner_rejects_reassigned_program_before_prefix_replay():
    owner=_owner();raw=_press_fixture_bytes()
    owner.program=replace(owner.program,targets=(.05,0.,.06))
    with pytest.raises(ValueError,match='bound program changed'):
        owner._restore(raw,sha256(raw).hexdigest())


def test_nonvirgin_rejected_trial_cancellation_and_exception_roll_back(monkeypatch):
    owner=_owner();raw=_press_fixture_bytes()
    archived=_load(raw.decode('ascii'))
    # Provisional same-pose contact rerecord; every saved field remains exact.
    # The adjacent pinned provenance does not claim a new coupled trajectory.
    assert archived['completed_targets']==1 and len(archived['records'])==1
    record=archived['records'][0]
    mechanical=owner.context.make(record['mechanical'],decoded=True)
    _,replayed=owner._record(record['target'],mechanical,owner.initial.histories,
                            record['previous_sha256'])
    assert canonical(record)==canonical(replayed), replay_difference_message(record,replayed,'complete saved record')
    expected_records=_load(raw.decode('ascii'))['records']
    actual_record=owner._record
    def traced_record(target,*args,**kwargs):
        proposed,row=actual_record(target,*args,**kwargs)
        expected=expected_records[target-1]
        if canonical(expected)!=canonical(row):
            changes=[]
            def walk(a,b,path):
                if isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b):
                    for key in a:walk(a[key],b[key],path+'.'+key)
                elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
                    for index,(left,right) in enumerate(zip(a,b)):walk(left,right,path+f'[{index}]')
                elif a!=b:
                    change={'path':path,'types':[type(a).__name__,type(b).__name__]}
                    if type(a) in (int,float) and type(b) in (int,float):
                        change['absolute_difference']=abs(a-b)
                    changes.append(change)
            walk(json.loads(canonical(expected)),json.loads(canonical(row)),'record')
            print('strict_fixture_replay_difference',json.dumps(dict(
                count=len(changes),first=changes[:12],
                max_numeric_difference=max((c.get('absolute_difference',0.) for c in changes),default=0.))))
        return proposed,row
    monkeypatch.setattr(owner,'_record',traced_record)
    accepted,records=owner._restore(raw,sha256(raw).hexdigest())
    assert owner._checkpoint(records)==raw
    origin=canonical(accepted.histories)
    assert origin!=canonical(owner.initial.histories)
    # Real unsupported-contact candidate, with material evaluated from the
    # current committed origin. Neither proposal nor gap rejection commits it.
    step=np.zeros(owner.context.count)
    node=owner.context.nodes[owner.indices[1]][1]
    step[6*node+1]=-.001
    first=owner.context.nodal_count+24*owner.indices[1]
    step[first+6:first+24]=np.linspace(-.001,.001,18)
    candidate=owner.context.advance(accepted.mechanical,step)
    owner.context.assemble(candidate,0.,accepted.histories)
    rejected,gap=owner._endpoint_contact(candidate)
    assert rejected is None and gap < -owner.gap_tolerance
    assert canonical(accepted.histories)==origin and owner._checkpoint(records)==raw
    token=CancellationToken();token.cancel('nonvirgin lifecycle rollback')
    failed=owner.solve(checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),
                       stop_after=2,cancellation_token=token)
    assert failed.status=='failed' and 'SolveCancelled' in failed.failure
    assert failed.checkpoint==raw and canonical(failed.accepted.histories)==origin
    def interrupt(*args,**kwargs):raise RuntimeError('nonvirgin lifecycle interruption')
    monkeypatch.setattr(owner,'_assemble',interrupt)
    # Restore itself legitimately recomputes the accepted prefix; inject the
    # exception only after authenticated replay has completed.
    restore=owner._restore
    def restored(*args):
        monkeypatch.setattr(owner,'_assemble',original)
        value=restore(*args)
        monkeypatch.setattr(owner,'_assemble',interrupt)
        return value
    original=type(owner)._assemble.__get__(owner)
    monkeypatch.setattr(owner,'_restore',restored)
    failed=owner.solve(checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),stop_after=2)
    assert failed.status=='failed' and 'nonvirgin lifecycle interruption' in failed.failure
    assert failed.checkpoint==raw and canonical(failed.accepted.histories)==origin
    assert owner.context._token is None and not owner.context._lock.locked()


@pytest.mark.skipif(os.environ.get('CONTACT_NATIVE_CURVED_LIFECYCLE_REPLAY')!='1',
                    reason='explicit accepted complete-chain replay; no solve')
def test_complete_lifecycle_canonical_replay_and_prefix_history_chain():
    raw=Path(os.environ.get('CONTACT_NATIVE_CURVED_LIFECYCLE_CHECKPOINT',
                           str(_prefix(3)))).read_bytes()
    owner=_owner()
    virgin=canonical(owner.initial.histories)
    replay_assemblies=[]
    actual=owner._assemble
    def capture(*args,**kwargs):
        evaluated=actual(*args,**kwargs)
        replay_assemblies.append((args[0],evaluated))
        return evaluated
    owner._assemble=capture
    accepted,records=owner._restore(raw,sha256(raw).hexdigest())
    frozen=_load(raw.decode('ascii'))
    assert accepted.completed_targets==3 and len(records)==3
    assert owner._checkpoint(records)==raw
    assert canonical(records)==canonical(frozen['records'])
    final_history=owner.context.histories(frozen['records'][-1]['histories'],decoded=True)
    assert accepted.histories==final_history
    assert canonical(accepted.mechanical.descriptor())==canonical(records[-1]['mechanical'])
    assert canonical(owner.initial.histories)==virgin
    assert canonical(accepted.histories)!=virgin
    for stage in (1,2):
        prefix=_load(_prefix(stage).read_text())
        assert prefix['completed_targets']==stage
        assert canonical(records[:stage])==canonical(prefix['records'])
    # Existing full replay validates every response against its actual previous
    # origin. Cross-check the accepted records retain their measured lifecycle.
    assert records[0]['contact']['potential_j']>0
    assert records[1]['contact']['potential_j']==0
    assert records[1]['contact']['conservative_lower_gap_m']>0
    assert records[2]['contact']['potential_j']>0
    # Spatial root reactions come from the actual complete replay residual,
    # including retained work. External load is declared independently below.
    mechanical,evaluated=replay_assemblies[-1]
    residual=evaluated[0];contact=evaluated[4]
    positions=mechanical.positions+mechanical.position_low
    body_errors=[]
    for body,index in enumerate(owner.indices):
        nodes=owner.context.nodes[index];root,tip=nodes[0],nodes[-1]
        force=residual[6*root:6*root+3]
        moment=residual[6*root+3:6*root+6]+np.cross(positions[root],force)
        load=np.array((0.,-.05*TARGETS[2],0.)) if body==1 else np.zeros(3)
        contact_force=(contact.beam_a_force_n if body==0 else contact.beam_b_force_n)
        contact_moment=(contact.beam_a_moment_nm if body==0 else contact.beam_b_moment_nm)
        force_error=float(np.linalg.norm(force+contact_force+load)/.0025)
        moment_error=float(np.linalg.norm(moment+contact_moment+np.cross(positions[tip],load))/.005)
        body_errors.append(dict(force_error=force_error,moment_error=moment_error,
            root_position_m=positions[root].tolist(),tip_position_m=positions[tip].tolist(),
            support_force_n=force.tolist(),support_moment_about_origin_nm=moment.tolist(),
            contact_force_n=contact_force.tolist(),contact_moment_about_origin_nm=contact_moment.tolist(),
            applied_tip_force_n=load.tolist(),applied_tip_moment_about_origin_nm=np.cross(positions[tip],load).tolist()))
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/'lifecycle_final_replay.json').write_text(json.dumps(dict(
        checkpoint_sha256=sha256(raw).hexdigest(),completed=3,
        canonical_capsule=True,canonical_records=True,
        final_history_matches=True,earlier_prefix_records_match=True,
        support_wrench_errors=body_errors),indent=2))
    assert max(max(row['force_error'],row['moment_error']) for row in body_errors)<=1e-6
    assert owner.context._token is None and not owner.context._lock.locked()
