"""Changed-algorithm checks; existing failed/default evidence stays unchanged."""
import json
import os
from hashlib import sha256
from pathlib import Path
import numpy as np
import pytest
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.general_contact.native_activation_static import NativeContactActivationSettings
from test_general_contact_native_material_s3 import _analysis, _accumulated, _two_analysis


def test_yielding_first_target_with_explicit_area_activation():
    kwargs = dict(native_yield=1e-5, stiffness=1e7, targets=(.25,.5,.75,1.),
                  activation_settings=NativeContactActivationSettings(1e-6,256))
    owner = _analysis(**kwargs)
    initial = canonical((owner.initial.mechanical.descriptor(), owner.initial.native_histories))
    result = owner.solve(stop_after=1)
    output = os.environ.get('CONTACT_APP_RESULT_RECORD')
    if output:
        Path(output).write_text(json.dumps(dict(status=result.status, failure=result.failure,
            completed_targets=result.accepted.completed_targets,
            queries=owner.activation_total_queries,
            minimum_gap_m=result.accepted.contact.contact.minimum_gap_m,
            maximum_accumulated=max(_accumulated(result.accepted.native_histories))),indent=2))
    assert result.status=='paused' and result.accepted.completed_targets==1, result.failure
    assert owner.activation_total_queries>0
    assert max(_accumulated(result.accepted.native_histories))>0.
    assert initial==canonical((owner.initial.mechanical.descriptor(), owner.initial.native_histories))
    assert result.accepted.contact.contact.minimum_gap_m>=-owner.gap_tolerance
    a=result.accepted
    _,_,metrics,_,contact=owner._assemble(a.mechanical,a.shell_displacement,.25,a.native_histories)
    assert max(metrics)<=owner.equilibrium_tolerance
    assert np.linalg.norm(contact.contact.beam_force_n+contact.contact.shell_force_n)<=1e-12


def test_yielding_full_load_cycle_and_exact_split_restart():
    kwargs = dict(native_yield=1e-5, stiffness=1e7,
                  targets=(.25,.5,.75,1.,0.,.25,.5,.75,1.),
                  activation_settings=NativeContactActivationSettings(1e-6,256))
    owner=_analysis(**kwargs);run=owner.solve()
    output=os.environ.get('CONTACT_APP_RESULT_RECORD')
    summary=dict(status=run.status,failure=run.failure,completed_targets=run.accepted.completed_targets,
                 queries=owner.activation_total_queries,checkpoint_sha256=run.checkpoint_sha256)
    if output:Path(output+'-cycle.json').write_text(json.dumps(summary,indent=2))
    assert run.status=='completed',run.failure
    records=json.loads(run.checkpoint)['records'];previous=np.asarray(_accumulated(owner.initial.native_histories))
    for row in records:
        now=np.asarray(_accumulated(row['histories']))
        assert np.all(now>=previous);previous=now
        assert row['contact']['minimum_gap_m']>=-owner.gap_tolerance
    assert max(previous)>0. and records[4]['contact']['potential_j']==0.
    split=_analysis(**kwargs).solve(stop_after=5)
    assert split.status=='paused' and split.accepted.contact.contact.minimum_gap_m>0.
    resumed=_analysis(**kwargs).solve(checkpoint=split.checkpoint,expected_sha256=split.checkpoint_sha256)
    assert resumed.status=='completed' and resumed.checkpoint==run.checkpoint
    assert canonical(resumed.accepted.mechanical.descriptor())==canonical(run.accepted.mechanical.descriptor())
    # Check the tangent actually used for the first full-load plastic increment,
    # from the .75 physical origin (not an already committed yield kink).
    raw=owner._checkpoint(tuple(records[:3]))
    origin,_=owner._restore(raw,sha256(raw).hexdigest())
    pose=owner.context.make(records[3]['mechanical'],decoded=True)
    u=np.asarray(records[3]['shell_displacement'])
    _,tangent,_,responses,_=owner._assemble(pose,u,1.,origin.native_histories)
    assert max(_accumulated(tuple(row.history for row in responses)))>max(_accumulated(origin.native_histories))
    direction=np.random.default_rng(924).normal(size=60)*.001
    direction[list(owner.context.fixed)]=0.;direction[42+owner.shell_fixed]=0.
    errors=[]
    for h in (1e-5,2e-6):
        states=[owner.context.advance(pose,sign*h*direction[:42]) for sign in (1,-1)]
        values=[owner._assemble(s,u+sign*h*direction[42:],1.,origin.native_histories)[0]
                for sign,s in zip((1,-1),states)]
        fd=(values[0]-values[1])/(2*h)
        error=float(np.linalg.norm(tangent@direction-fd)/np.linalg.norm(fd))
        assert error<=1e-5;errors.append(error)
    # Explicit algorithm/settings participate in restart identity; defaults
    # and different activation budgets cannot consume this checkpoint.
    for setting in (None,NativeContactActivationSettings(2e-6,256),NativeContactActivationSettings(1e-6,128)):
        changed=_analysis(**dict(kwargs,activation_settings=setting))
        with pytest.raises(ValueError,match='binding'):
            changed.solve(checkpoint=split.checkpoint,expected_sha256=split.checkpoint_sha256,stop_after=5)
    a=run.accepted;r,_,metrics,_,contact=owner._assemble(a.mechanical,a.shell_displacement,1.,a.native_histories)
    mids=(owner.contact.shell.reference_positions-owner.contact.shell.physical_offset*owner.contact.shell.director
          +a.shell_displacement.reshape(3,6)[:,:3])
    shell_r=r[42:].reshape(3,6);root=owner.context.nodes[0][0];tip=owner.context.nodes[0][2]
    load=np.array((0.,0.,-.005))
    balance=shell_r[:,:3].sum(axis=0)+r[6*root:6*root+3]+load
    moment=(np.cross(mids,shell_r[:,:3]).sum(axis=0)+shell_r[:,3:].sum(axis=0)+r[6*root+3:6*root+6]
            +np.cross(a.mechanical.positions[root],r[6*root:6*root+3])+np.cross(a.mechanical.positions[tip],load))
    nf=float(np.linalg.norm(balance)/.005);nm=float(np.linalg.norm(moment)/.01)
    assert max(metrics)<=owner.equilibrium_tolerance and max(nf,nm)<=1e-6
    assert np.linalg.norm(contact.contact.beam_force_n+contact.contact.shell_force_n)<=1e-12
    summary.update(maximum_accumulated=float(max(previous)),
                   maximum_penetration_m=-min(row['contact']['minimum_gap_m'] for row in records),
                   normalized_force_balance=nf,normalized_moment_balance=nm,
                   metrics=metrics,owner_plastic_tangent_errors=errors,
                   storage_j=[row['contact']['potential_j'] for row in records])
    if output:Path(output+'-cycle.json').write_text(json.dumps(summary,indent=2))


@pytest.mark.parametrize('cancel',(False,True))
def test_activation_query_exhaustion_or_cancellation_preserves_virgin_prefix(cancel,monkeypatch):
    from anyfesolver.control import CancellationToken
    owner=_analysis(native_yield=1e-5,stiffness=1e7,
        activation_settings=NativeContactActivationSettings(1e-6,256 if cancel else 1))
    capsule=owner._checkpoint(());before=canonical(owner.initial.native_histories)
    mechanical=canonical(owner.initial.mechanical.descriptor());shell=owner.initial.shell_displacement.copy()
    token=CancellationToken()
    if cancel:
        query=owner.contact.geometry_gap
        def interrupted(*args):
            value=query(*args);token.cancel('during material activation');return value
        monkeypatch.setattr(owner.contact,'geometry_gap',interrupted)
    result=owner.solve(cancellation_token=token)
    assert result.status==('cancelled' if cancel else 'failed')
    assert 'activation geometry' in result.failure
    assert result.accepted.completed_targets==0 and result.checkpoint==capsule
    assert canonical(result.accepted.native_histories)==before and max(_accumulated(result.accepted.native_histories))==0.
    assert canonical(result.accepted.mechanical.descriptor())==mechanical
    np.testing.assert_array_equal(result.accepted.shell_displacement,shell)
    assert owner.activation_total_queries==1


def test_material_radius_hidden_crossing_limits_internal_rotation():
    owner=_analysis(radius=.03,shell_z=-.0299,angles=(np.pi-.31,np.pi-.29),
                    activation_settings=NativeContactActivationSettings(1e-6,256))
    start=owner.initial.mechanical;u=np.zeros(18);step=np.zeros(60);step[21]=1.2
    origins=canonical(owner.initial.native_histories)
    fraction=owner._activation_fraction(start,u,step,1.,owner.initial.contact,None)
    assert 0.<fraction<.25
    end=owner.context.advance(start,fraction*step[:42]);row=owner._contact(end,u).contact
    assert row.minimum_gap_m>=-1e-6
    assert owner._swept_admissible(start,u,fraction*step,end,u,
        owner.initial.contact.contact.minimum_gap_m,row.minimum_gap_m,
        owner.initial.contact.contact.minimum_rim_clearance_m,row.minimum_rim_clearance_m)
    assert origins==canonical(owner.initial.native_histories)


def test_multiple_pair_activation_uses_second_retained_block():
    from anyfesolver.general_contact import create_native_contact_analysis
    base=_two_analysis(targets=(.5,1.))
    from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
    owner=create_native_contact_analysis(tuple(NativeBeamDefinition(raw) for raw in base.native._definitions),
        tuple(base.native.model.boundary_conditions),base.program,base.shell_model,base.pairs,
        equilibrium_tolerance=1e-10,activation_settings=NativeContactActivationSettings(1e-6,256))
    n=owner.context.count;initial=np.zeros(n)
    for node in owner.context.nodes[1][1:]:initial[6*node+2]=-.0003
    index=owner.context.nodal_count+24+3;initial[index]=-.6
    start=owner.context.advance(owner.initial.mechanical,initial);u=np.zeros(18)
    step=np.zeros(n+18);step[index]=1.2;contact=owner._contact(start,u)
    fraction=owner._activation_fraction(start,u,step,1.,contact,None)
    assert 0.<fraction<.5
    end=owner.context.advance(start,fraction*step[:n]);last=owner._contact(end,u)
    assert all(r.contact.minimum_gap_m>=-1e-6 for _,r in last.pairs)
    assert owner._swept_admissible(start,u,fraction*step,end,u)


def test_hidden_finite_edge_crossing_limits_material_prefix():
    triangle=((-5.,.0299),(0.,-5.),(5.,.0299))
    owner=_analysis(radius=.03,shell_z=-.05,angles=(np.pi+1.27,np.pi+1.29),triangle=triangle,
                    activation_settings=NativeContactActivationSettings(1e-6,256))
    start=owner.initial.mechanical;u=np.zeros(18);step=np.zeros(60);step[21]=1.2
    fraction=owner._activation_fraction(start,u,step,1.,owner.initial.contact,None)
    assert 0.<fraction<.25
    end=owner.context.advance(start,fraction*step[:42]);row=owner._contact(end,u).contact
    assert row.minimum_rim_clearance_m>owner.contact.contact.rim_clearance
    assert owner._swept_admissible(start,u,fraction*step,end,u,
        owner.initial.contact.contact.minimum_gap_m,row.minimum_gap_m,
        owner.initial.contact.contact.minimum_rim_clearance_m,row.minimum_rim_clearance_m)


@pytest.mark.parametrize('gap',(1e-10,5e-5))
def test_activation_budget_must_lie_inside_physical_gap_tolerance(gap):
    with pytest.raises(ValueError,match='inside every pair gap budget'):
        _analysis(activation_settings=NativeContactActivationSettings(gap,256))


def test_multiple_material_facets_activation_cycle_and_staged_rollback(monkeypatch):
    from anyfesolver.general_contact import create_native_contact_analysis
    from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
    from anyfesolver.control import CancellationToken
    base=_two_analysis(facets=2)
    def make():
        return create_native_contact_analysis(tuple(NativeBeamDefinition(raw) for raw in base.native._definitions),
            tuple(base.native.model.boundary_conditions),base.program,base.shell_model,base.pairs,
            equilibrium_tolerance=1e-10,activation_settings=NativeContactActivationSettings(1e-6,256))
    owner=make();run=owner.solve()
    assert run.status=='completed' and run.accepted.completed_targets==9,run.failure
    assert owner.activation_total_queries>0
    rows=json.loads(run.checkpoint)['records']
    assert rows[4]['contact']['potential_j']==0. and rows[8]['contact']['potential_j']>0.
    # Restore the exact open physical origin, then cancel a staged recontact
    # commit. Neither native, shell, history nor checkpoint may advance.
    prefix=make().solve(stop_after=5);assert prefix.status=='paused'
    owner=make();checkpoint=owner._checkpoint;token=CancellationToken()
    def staged(records):
        raw=checkpoint(records)
        if len(records)>5:token.cancel('staged material recontact')
        return raw
    monkeypatch.setattr(owner,'_checkpoint',staged)
    rejected=owner.solve(checkpoint=prefix.checkpoint,expected_sha256=prefix.checkpoint_sha256,cancellation_token=token)
    assert rejected.status=='cancelled' and rejected.accepted.completed_targets==5
    assert rejected.checkpoint==prefix.checkpoint
    assert canonical(rejected.accepted.mechanical.descriptor())==canonical(prefix.accepted.mechanical.descriptor())
    assert canonical(rejected.accepted.native_histories)==canonical(prefix.accepted.native_histories)
    np.testing.assert_array_equal(rejected.accepted.shell_displacement,prefix.accepted.shell_displacement)
    resumed=make().solve(checkpoint=prefix.checkpoint,expected_sha256=prefix.checkpoint_sha256)
    assert resumed.status=='completed' and resumed.checkpoint==run.checkpoint
