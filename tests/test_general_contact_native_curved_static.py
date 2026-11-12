"""Curved material contact through the existing retained static transaction."""
from hashlib import sha256
import json,os
from pathlib import Path
import pytest

import numpy as np
from anyfesolver.general_contact.native_retained_static import CURVED_MATERIAL_PROFILE
from examples.general_contact_native_curved_material_plane import analysis


def _canonical_fixture_payload(path):
    raw=path.read_bytes()
    # A text-container checkout may terminate the canonical payload's final
    # newline as CRLF; drop only that container carriage return so strict
    # restores and exact checkpoint byte equality see the producer payload,
    # which itself ends with a final LF. No-op on LF checkouts.
    return raw[:-2]+b'\n' if raw.endswith(b'\r\n') else raw


def _analysis(targets=(1.,),iterations=24,stiffness=1e7):
    return analysis(targets,normal_stiffness_Pa_per_m=stiffness,max_iterations=iterations)


def test_example_binds_verified_cycle_definition():
    reference=Path(__file__).parent/'data/general_contact_native_curved_plane_v7.checkpoint'
    assert analysis().identity==json.loads(reference.read_bytes())['identity']


def test_frozen_curved_cycle_independent_physical_skin_gaps():
    # Analytic parabola and this fixture's (+z, tangent cross +z) section frame.
    # No production station, projection, bounds, area or contact evaluator.
    reference=Path(__file__).parent/'data/general_contact_native_curved_plane_v7.checkpoint'
    gaps=[]
    for record in json.loads(reference.read_bytes())['records']:
        mechanical=record['mechanical']
        p=np.array(mechanical['positions'])+np.array(mechanical['position_low'])
        U=np.array(mechanical['cell_rotations'])[0,1]
        minimum=np.inf
        for xi in np.linspace(.1,.9,401):
            tangent=np.array((1.,-.4*xi,0.));tangent/=np.linalg.norm(tangent)
            third=np.cross(tangent,(0.,0.,1.))
            center=(1-xi)*p[1]+xi*p[2]+U@np.array((0.,.2*xi*(1-xi),0.))
            cosine=.03*U[2,2];sine=.03*float(U[2]@third)
            phase=np.arctan2(sine,cosine)
            angles=[np.pi-.3,np.pi+.3]+[phase+np.pi+2*k*np.pi for k in (-1,0,1)
                if np.pi-.3<phase+np.pi+2*k*np.pi<np.pi+.3]
            minimum=min(minimum,*(center[2]+.0301+cosine*np.cos(a)+sine*np.sin(a) for a in angles))
        assert minimum>=-3e-5
        assert record['contact']['minimum_gap_m']<=minimum+1e-12
        gaps.append(float(minimum))
    assert gaps[0]<0 and gaps[1]>0 and gaps[2]<0
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path+'-curved-independent-gaps.json').write_text(json.dumps(dict(
        sampled_axial_station_count=401,exact_angular_extrema=True,gap_estimates_m=gaps,
        whole_gap_certificate=False),indent=2))


def test_curved_material_coupled_retained_response_and_complete_work():
    owner=_analysis();run=owner.solve()
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-curved-owner.json').write_text(json.dumps(dict(status=run.status,
            failure=run.failure,diagnostics=run.diagnostics,completed=run.accepted.completed_targets,rejected_trials=owner.curved_rejected_trials),indent=2))
    assert run.status=='completed',run.failure
    assert run.profile==CURVED_MATERIAL_PROFILE
    contact=run.accepted.contact;state=run.accepted.mechanical
    assert contact.beam_force_n[2]>0 and contact.minimum_gap_m>=-owner.gap_tolerance
    assert np.linalg.norm(contact.residual[18:24])>0
    assert np.linalg.norm(state.cell_rotations-owner.initial.mechanical.cell_rotations)>0
    r,j,metrics,_,_=owner._assemble(state,1.,run.accepted.histories)
    assert max(metrics)<1e-10 and np.linalg.norm(r[owner.context.free])<1e-10
    load=np.array((0.,0.,-.005))
    balance=r[:3]+load+contact.beam_force_n
    wrench=r[3:6]+np.cross(state.positions[0]+state.position_low[0],r[:3])+np.cross(
        state.positions[2]+state.position_low[2],load)+contact.beam_moment_nm
    assert np.linalg.norm(balance)/.005<=1e-6
    assert np.linalg.norm(wrench)/.01<=1e-6
    direction=np.zeros(owner.context.count);direction[14]=-.002;direction[18]=.001;direction[21]=-.001
    errors=[]
    for h in (1e-5,2e-6):
        plus=owner._assemble(owner.context.advance(state,h*direction),1.,run.accepted.histories)[0]
        minus=owner._assemble(owner.context.advance(state,-h*direction),1.,run.accepted.histories)[0]
        observed=(plus-minus)/(2*h);error=np.linalg.norm(observed-j@direction)/np.linalg.norm(observed)
        errors.append(float(error));assert error<=1e-5
    record=json.loads(run.checkpoint)
    assert record['records'][0]['contact']['minimum_gap_kind']=='continuous_whole_material_patch_lower_bound'
    if path:
        Path(path+'-curved-owner.json').write_text(json.dumps(dict(status=run.status,
            failure=run.failure,completed=run.accepted.completed_targets,
            gap_bound_m=contact.minimum_gap_m,force_n=contact.beam_force_n.tolist(),
            normalized_force_balance=float(np.linalg.norm(balance)/.005),
            normalized_moment_balance=float(np.linalg.norm(wrench)/.01),tangent_errors=errors,
            checkpoint_sha256=sha256(run.checkpoint).hexdigest()),indent=2))


def test_curved_material_press_open_recontact():
    owner=_analysis((1.,0.,1.));run=owner.solve()
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-curved-cycle.json').write_text(json.dumps(dict(status=run.status,
            failure=run.failure,diagnostics=run.diagnostics,rejected_trials=owner.curved_rejected_trials),indent=2))
    assert run.status=='completed',run.failure
    records=json.loads(run.checkpoint)['records']
    if path:Path(path+'-curved-cycle.checkpoint').write_bytes(run.checkpoint)
    assert len(records)==3
    for row in (records[0],records[2]):
        assert row['contact']['beam_force_n'][2]>0
        assert row['contact']['minimum_gap_m']>=-3e-5
    assert records[1]['contact']['beam_force_n']==[0.,0.,0.]
    assert records[1]['contact']['potential_j']==0.
    assert records[1]['contact']['minimum_gap_m']>0
    # Unloading/reloading is a new equilibrium solve, not checkpoint replay.
    # Preserve exact-byte comparison for authentic split restart below.
    for key,initial in records[0]['mechanical'].items():
        np.testing.assert_allclose(records[2]['mechanical'][key],initial,rtol=1e-12,atol=1e-14)


def test_curved_material_authentic_split_restart_and_atomic_failure():
    from anyfesolver.control import CancellationToken
    reference=Path(__file__).parent/'data/general_contact_native_curved_plane_v7.checkpoint'
    full=_canonical_fixture_payload(reference)
    owner=_analysis((1.,0.,1.));prefix=owner.solve(stop_after=1)
    assert prefix.status=='paused',prefix.failure
    for cancelled in (False,True):
        changed=_analysis((1.,0.,1.));record=changed._record;token=CancellationToken()
        def injected(target,*args):
            staged=record(target,*args)
            if target==2:
                if cancelled:token.cancel('curved precommit injection')
                else:raise RuntimeError('curved precommit injection')
            return staged
        changed._record=injected
        result=changed.solve(checkpoint=prefix.checkpoint,expected_sha256=prefix.checkpoint_sha256,
                             cancellation_token=token)
        assert result.status==('cancelled' if cancelled else 'failed')
        assert result.accepted.completed_targets==1 and result.checkpoint==prefix.checkpoint
        np.testing.assert_array_equal(result.accepted.mechanical.cell_rotations,
                                      prefix.accepted.mechanical.cell_rotations)
    continued=_analysis((1.,0.,1.)).solve(checkpoint=prefix.checkpoint,expected_sha256=prefix.checkpoint_sha256)
    assert continued.status=='completed',continued.failure
    assert continued.checkpoint==full
    changed=_analysis((1.,0.,1.),stiffness=2e7)
    with pytest.raises(ValueError,match='definition/settings binding'):
        changed.solve(checkpoint=prefix.checkpoint,expected_sha256=prefix.checkpoint_sha256)
