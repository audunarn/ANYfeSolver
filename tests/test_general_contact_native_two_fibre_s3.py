"""Two physical-fibre beams coupled through one actual deformable S3 patch."""
from pathlib import Path
from hashlib import sha256
import json,os
import numpy as np
import pytest
from test_general_contact_native_curved_s3_static import _analysis
from test_general_contact_native_fibre_s3 import _independent_material_history
from anyfesolver._ge_beam3_p5_seeded.core import canonical


def _owner(two_facets=False):
    return _analysis(targets=(1.,),two_beams=True,two_facets=two_facets,physical_fibres=True,plastic_fibres=True,
        normal_stiffness_Pa_per_m=8e7,integration_budget_policy='certified_active',
        root_tangent_enclosure_policy='anchored')


def _independent_energy(work,p,low,U,u,order=24):
    """Analytic fixture parabola/circle, independent adaptive axial measure."""
    from math import acos,atan2,ceil,floor,pi
    from scipy.integrate import quad_vec
    from numpy.polynomial.legendre import leggauss
    skin=work.shell
    q=np.array([u[b:b+6] for b in work.shell_translation_dofs[:,0]])
    corners=skin.reference_positions+q[:,:3]+skin.physical_offset*np.cross(q[:,3:],skin.director)
    normal=np.cross(corners[1]-corners[0],corners[2]-corners[0]);normal/=np.linalg.norm(normal)
    al,ar=work.contact.rule.definition['angle_range'];lo,hi=work.contact.rule.definition['xi_range']
    radius=.03;k=work.contact.stiffness;gl,gw=leggauss(order)
    def axial(xi):
        cell=0 if xi<0. else 1;t=xi+1 if cell==0 else xi
        j=np.sqrt(1+.16*xi*xi);transverse=np.array((.4*xi,1.,0.))/j
        centre=(1-t)*(p[cell]+low[cell])+t*(p[cell+1]+low[cell+1])+U[cell]@np.array((0.,.2*t*(1-t),0.))
        A=float(normal@(centre-corners[0]));B=float(normal@(U[cell]@np.array((0.,0.,radius))))
        C=float(normal@(U[cell]@(-radius*transverse)));amplitude=np.hypot(B,C)
        cuts=[al,ar]
        if amplitude and abs(A)<=amplitude:
            phase=atan2(C,B);root=acos(np.clip(-A/amplitude,-1.,1.))
            for angle in (phase-root,phase+root):
                cuts.extend(angle+turn*2*pi for turn in range(ceil((al-angle)/(2*pi)),floor((ar-angle)/(2*pi))+1))
        value=0.;cuts=sorted(set(cuts))
        for left,right in zip(cuts,cuts[1:]):
            mid=(left+right)/2
            if A+B*np.cos(mid)+C*np.sin(mid)>=0.:continue
            alpha=mid+(right-left)*gl/2;gap=A+B*np.cos(alpha)+C*np.sin(alpha)
            value+=float(np.sum(.5*k*gap**2*gw*(right-left)/2*radius*(j-.4*radius*np.sin(alpha)/j**2)))
        return value
    value,error,info=quad_vec(axial,lo,hi,epsabs=1e-19,epsrel=1e-12,limit=64,full_output=True)
    assert info.success,info.message
    return float(value)


@pytest.mark.parametrize('two_facets',(False,True),ids=('one-patch','two-patches'))
def test_two_fibre_beams_shared_target_complete_local_work(two_facets):
    owner=_owner(two_facets);n=owner.context.count
    assert n==84 and owner.beam_indices==((0,0,1,1) if two_facets else (0,1))
    assert len(owner.pairs)==(4 if two_facets else 2) and len(owner.initial.native_histories)==2
    # Coordinates only from the verified partial-front single-beam fixture.
    # This is local assembly verification, not a two-beam equilibrium or a
    # transfer of physical material history between owner identities.
    raw=(Path(__file__).parent/'data/general_contact_native_fibre_s3_plastic_k8e7_anchored_active_v1.checkpoint').read_bytes()
    assert sha256(raw).hexdigest()=='98d22bd2aff840df5d7b1034298891b179ed438ec9246a9f76b89445a7a15347'
    source=json.loads(raw)['records'][0];saved=source['mechanical']
    data={key:np.array(value,copy=True) for key,value in owner.initial.mechanical.descriptor().items()}
    from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum
    for i,nodes in enumerate(owner.context.nodes):
        for local,node in enumerate(nodes):
            for axis in range(3):
                data['positions'][node,axis],data['position_low'][node,axis]=split_sum((
                    float(saved['positions'][local][axis]),float(saved['position_low'][local][axis]),
                    .25 if i==1 and axis==1 else 0.))
            data['nodal_frames'][node]=saved['nodal_frames'][local]
        data['cell_rotations'][i]=saved['cell_rotations'][0]
        data['resultants'][i]=saved['resultants'][0]
    pose=owner.context.make(data)
    u=owner.initial.shell_displacement.copy()
    u[owner.shell_model.mesh.dof_manager.get_node_dofs(5)[2]]=source['shell_displacement'][8]
    result=owner._contact(pose,u)
    independent=[]
    for i,pair in zip(owner.beam_indices,owner.pairs):
        args=(pair.work,pose.positions[owner.context.nodes[i]],pose.position_low[owner.context.nodes[i]],pose.cell_rotations[i],u)
        coarse=_independent_energy(*args,12);fine=_independent_energy(*args,24)
        assert abs(coarse-fine)/max(abs(fine),1e-15)<=1e-6
        independent.append(fine)
    assert abs(sum(independent)-result.contact.potential_j)/result.contact.potential_j<=1e-6
    shared=n+owner.shell_model.mesh.dof_manager.get_node_dofs(5)[2]
    k=result.tangent.toarray()
    for i,(pair,pair_result) in zip(owner.beam_indices,result.pairs):
        slots=owner.context.slots[i]
        if not two_facets or pair.work.shell.element_id==22:
            assert pair_result.contact.potential_j>0.
            assert np.count_nonzero(k[slots,shared]) and np.count_nonzero(k[shared,slots])
            assert np.count_nonzero(pair_result.tangent[18:24,:].toarray())
        np.testing.assert_allclose(pair_result.contact.beam_force_n+pair_result.contact.shell_force_n,0.,atol=1e-12)
        np.testing.assert_allclose(pair_result.contact.beam_moment_nm+pair_result.contact.shell_moment_nm,0.,atol=1e-12)
    direction=np.random.default_rng(630).normal(size=n+len(u))*.001
    direction[list(owner.context._backend.layout.fixed)]=0.
    # Resultants have no geometric work; all physical internal rotations remain.
    for slots in owner.context.slots:direction[np.asarray(slots)[24:42]]=0.
    errors=[];work_errors=[]
    for h in (2e-6,1e-6):
        values=[];energies=[]
        for sign in (1,-1):
            moved=owner.context.advance(pose,sign*h*direction[:n]);su=u+sign*h*direction[n:]
            values.append(owner._contact(moved,su))
            energies.append(sum(_independent_energy(pair.work,
                moved.positions[owner.context.nodes[i]],moved.position_low[owner.context.nodes[i]],
                moved.cell_rotations[i],su) for i,pair in zip(owner.beam_indices,owner.pairs)))
        fd=(values[0].residual-values[1].residual)/(2*h)
        error=float(np.linalg.norm(k@direction-fd)/np.linalg.norm(fd));errors.append(error)
        derivative=(energies[0]-energies[1])/(2*h)
        work_error=float(abs(result.residual@direction-derivative)/max(abs(derivative),1e-12));work_errors.append(work_error)
        assert error<=1e-5 and work_error<=1e-5
    output=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if output:Path(output+'-two-beam-local.json').write_text(json.dumps(dict(
        count=n,shared_target=shared,two_facets=two_facets,directional_errors=errors,
        independent_virtual_work_errors=work_errors),indent=2))


@pytest.mark.parametrize('two_facets',(False,True),ids=('one-patch','two-patches'))
def test_two_fibre_beams_actual_coupled_history_and_restart(two_facets):
    owner=_owner(two_facets);origins=owner.initial.native_histories;before=canonical(origins)
    run=owner.solve();output=os.environ.get('CONTACT_APP_RESULT_RECORD')
    report=dict(status=run.status,failure=run.failure,completed=run.accepted.completed_targets,two_facets=two_facets,
        candidates=owner.candidate_evaluations,pair_evaluations=owner.pair_evaluations,
        reuse_hits=owner.contact_reuse_hits,assembly_reuse_hits=owner.assembly_reuse_hits,
        trial_failures=owner.curved_trial_failures)
    if output:
        Path(output+'-two-beam.json').write_text(json.dumps(report,indent=2))

        Path(output+'-two-beam.checkpoint').write_bytes(run.checkpoint)
    assert run.status=='completed' and run.accepted.completed_targets==1,run.failure
    state=run.accepted;contact=state.contact.contact
    assert len(state.native_histories)==2
    assert np.linalg.norm(contact.beam_force_n+contact.shell_force_n)/.008<=1e-6
    assert np.linalg.norm(contact.beam_moment_nm+contact.shell_moment_nm)/.02<=1e-6
    for pair,result in state.contact.pairs:
        assert result.contact.minimum_gap_m>=-pair.gap_tolerance_m
    for i in range(2):
        assert any(result.contact.potential_j>0. and result.contact.beam_force_n[2]>0.
            for index,(_,result) in zip(owner.beam_indices,state.contact.pairs) if index==i)
    # The original context's construction-to-expiry allowance is never renewed,
    # so every original-owner guarded operation runs immediately after solve
    # inside explicit operation boundaries. Independent restart verification
    # then proceeds on the restart owner's own fresh context.
    with owner.context.operation(None):
        residual,_,metrics=owner._assemble_structural(state.mechanical,state.shell_displacement,1.,origins)
        residual+=state.contact.residual
        material,proposed=_independent_material_history(owner,state.mechanical,origins)
    n=owner.context.count
    scaled=residual[:n]/owner.context.scale
    equilibrium=(np.linalg.norm(scaled[owner.context.equilibrium]),np.linalg.norm(scaled[owner.context.compatibility]),
        np.linalg.norm(residual[n+owner.shell_free]))
    assert max(equilibrium)<=owner.equilibrium_tolerance
    assert material['yielded']>0 and material['dissipation_j']>0.
    assert proposed==tuple(station for cell in state.native_histories for station in cell.stations)
    for cell in state.native_histories:
        assert any(row[2]+row[3]>0. for station in cell.stations for row in station.rows)
    assert canonical(origins)==before
    restart=_owner(two_facets)
    restored,records=restart._restore(run.checkpoint,run.checkpoint_sha256)
    assert restored.native_histories==state.native_histories
    with owner.context.operation(None):
        assert owner._checkpoint(records)==run.checkpoint
    assert owner.context._token is None and not owner.context._lock.locked()
    assert owner._contact_reuse_entry is None and not owner._contact_reuse_enabled
    assert owner._assembly_reuse_entry is None and not owner._assembly_reuse_enabled
    if output:
        report.update(material=material,equilibrium=list(map(float,equilibrium)),checkpoint_sha256=sha256(run.checkpoint).hexdigest(),
            pair_force_z=[float(result.contact.beam_force_n[2]) for _,result in state.contact.pairs])
        Path(output+'-two-beam.json').write_text(json.dumps(report,indent=2))


@pytest.mark.parametrize('two_facets',(False,True),ids=('one-patch','two-patches'))
def test_two_fibre_saved_example_cli_recovery(tmp_path,monkeypatch,two_facets):
    import runpy,sys
    from test_general_contact_native_fibre_results import _stored_dissipation
    name='general_contact_native_fibre_s3_plastic_two_beams'+('_two_patches' if two_facets else '')+'_v1.checkpoint'
    fixture=Path(__file__).parent/'data'/name
    digest=sha256(fixture.read_bytes()).hexdigest()
    assert digest==('38263a956e22496693d968c8c3b6a59f91120c94d4a9e412c4624919814bb28b' if two_facets else
        'a8211bce3cf3468fa2c4f10328b031b635f88bb240f9fee0bdb98b09653282da')
    example=Path(__file__).parents[1]/'examples/general_contact_native_curved_material_s3.py'
    output=tmp_path/'two_beam_material.json'
    argv=[str(example),'--targets','1','--two-beams','--plastic-fibres',
        '--normal-stiffness','80000000','--integration-budget-policy','certified_active',
        '--root-tangent-enclosure-policy','anchored','--restart',str(fixture),
        '--recover-material',str(output),'--material-stations']
    if two_facets:argv.append('--two-facets')
    monkeypatch.setattr(sys,'argv',argv)
    runpy.run_path(str(example),run_name='__main__')
    report=json.loads(output.read_text())
    assert report['completed_targets']==1 and report['checkpoint_sha256']==digest
    assert not report['independent_engineering_acceptance'] and not report['energy_balance_qualified']
    elements=report['increments'][0]['elements'];assert [row['element_id'] for row in elements]==[1,2]
    owner=_owner(two_facets)
    state,_=owner._restore(fixture.read_bytes(),digest)
    from types import SimpleNamespace
    from test_general_contact_native_fibre_enforcement import _gap
    gaps=[]
    for i,(pair,result) in zip(owner.beam_indices,state.contact.pairs):
        nodes=owner.context.nodes[i]
        local=SimpleNamespace(shell_displacement=state.shell_displacement,mechanical=SimpleNamespace(
            positions=state.mechanical.positions[nodes],position_low=state.mechanical.position_low[nodes],
            cell_rotations=state.mechanical.cell_rotations[i:i+1]))
        gap=_gap(SimpleNamespace(pairs=(pair,)),local,129);gaps.append(gap)
        assert gap>=-pair.gap_tolerance_m and result.contact.minimum_gap_m<=gap+1e-10
    residual,_,_=owner._assemble_structural(state.mechanical,state.shell_displacement,1.,owner.initial.native_histories)
    residual+=state.contact.residual;n=owner.context.count
    shell_reactions=residual[n:].reshape(-1,6)
    shell_positions=np.array([node.coords() for _,node in sorted(owner.shell_model.mesh.nodes.items())])
    shell_positions+=state.shell_displacement.reshape(-1,6)[:,:3]
    force=shell_reactions[:,:3].sum(axis=0)
    moment=np.cross(shell_positions,shell_reactions[:,:3]).sum(axis=0)+shell_reactions[:,3:].sum(axis=0)
    loads=owner.context._backend.layout.force.reshape(-1,6)
    for nodes in owner.context.nodes:
        root=nodes[0]
        force+=residual[6*root:6*root+3]
        moment+=np.cross(state.mechanical.positions[root],residual[6*root:6*root+3])+residual[6*root+3:6*root+6]
    force+=loads[:,:3].sum(axis=0)
    moment+=np.cross(state.mechanical.positions+state.mechanical.position_low,loads[:,:3]).sum(axis=0)+loads[:,3:].sum(axis=0)
    assert np.linalg.norm(force)/.008<=1e-6 and np.linalg.norm(moment)/.02<=1e-6
    for i,row in enumerate(elements):
        expected=np.zeros(3)
        section=owner.context.elements[i][1].section
        for station,origin in zip(row['stations'],owner.initial.native_histories[i].stations):
            independent=_stored_dissipation(section,station,origin)
            expected+=station['measure']*np.array(list(map(float,independent)))
        actual=[row[key] for key in ('stored_material_energy_j','plastic_dissipation_increment_j','incremental_material_potential_j')]
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-24)
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record+'-two-beam-recovery.json').write_text(json.dumps(dict(
        independent_gap_m=gaps,normalized_force_balance=float(np.linalg.norm(force)/.008),
        normalized_moment_balance=float(np.linalg.norm(moment)/.02),checkpoint_sha256=digest,
        element_ids=[row['element_id'] for row in elements],two_facets=two_facets),indent=2))
    if two_facets:
        single=Path(__file__).parent/'data/general_contact_native_fibre_s3_plastic_two_beams_v1.checkpoint'
        raw=single.read_bytes()
        with pytest.raises(ValueError,match='binding'):owner._restore(raw,sha256(raw).hexdigest())


def _fixture_owner(two_facets):
    name='general_contact_native_fibre_s3_plastic_two_beams'+('_two_patches' if two_facets else '')+'_v1.checkpoint'
    raw=(Path(__file__).parent/'data'/name).read_bytes()
    owner=_owner(two_facets)
    state,_=owner._restore(raw,sha256(raw).hexdigest())
    return owner,state,raw


@pytest.mark.parametrize('two_facets',(False,True),ids=('one-patch','two-patches'))
def test_two_fibre_assembly_reuse_parity_origins_identity_and_refusal(two_facets):
    owner,state,raw=_fixture_owner(two_facets)
    m=state.mechanical;u=state.shell_displacement
    structural=[]
    assemble=owner._assemble_structural
    def capture(*args):
        structural.append(args[3]);return assemble(*args)
    owner._assemble_structural=capture
    with owner.context.operation(None):
        owner._begin_candidate_context(None)
        try:
            first=owner._assemble(m,u,1.,owner.initial.native_histories)
            assert owner.assembly_reuse_hits==0 and owner._assembly_reuse_entry is not None
            again=owner._assemble(m,u.copy(),1.,owner.initial.native_histories)
            assert again is first and owner.assembly_reuse_hits==1
            # Exact pose/origins/parameter identity only: different constitutive
            # origins must rerun the structural assembly and never reuse it.
            del structural[:]
            contact_hits=owner.contact_reuse_hits
            second=owner._assemble(m,u,1.,state.native_histories)
            assert structural==[state.native_histories] and owner.assembly_reuse_hits==1
            assert owner.contact_reuse_hits==contact_hits+1
            # Cached assembly buffers are read-only.
            with pytest.raises(ValueError,match='read-only'):first[0][0]=0.
            with pytest.raises(ValueError,match='read-only'):first[1][0,0]=0.
            # A mutated cached assembly is refused and the entry is dropped.
            cached=owner._assemble(m,u,1.,owner.initial.native_histories)
            cached[4].tangent.data=cached[4].tangent.data.copy()
            cached[4].tangent.data[0]+=1.
            with pytest.raises(ValueError,match='mutated'):
                owner._assemble(m,u,1.,owner.initial.native_histories)
            assert owner._assembly_reuse_entry is None
        finally:owner._end_candidate_context()
    assert owner._assembly_reuse_entry is None and not owner._assembly_reuse_enabled
    assert owner._contact_reuse_entry is None and not owner._contact_reuse_enabled
    # Outside the candidate context every assembly is authoritative and the
    # cached candidate is exactly reproducible.
    owner._assemble_structural=assemble
    plain=owner._assemble(m,u,1.,owner.initial.native_histories)
    np.testing.assert_array_equal(plain[0],first[0])
    np.testing.assert_array_equal(plain[1],first[1])
    assert plain[2]==first[2] and canonical(plain[4].contact.to_dict())==canonical(first[4].contact.to_dict())
    assert [canonical(r.history) for r in plain[3]]==[canonical(r.history) for r in first[3]]


@pytest.mark.parametrize('two_facets',(False,True),ids=('one-patch','two-patches'))
def test_two_fibre_restart_restore_reevaluates_and_operation_boundary_refusal(two_facets):
    owner,state,raw=_fixture_owner(two_facets)
    # Restored records are independently reevaluated by a fresh owner.
    fresh=_owner(two_facets)
    pairs,candidates=fresh.pair_evaluations,fresh.candidate_evaluations
    restored,records=fresh._restore(raw,sha256(raw).hexdigest())
    assert fresh.pair_evaluations>pairs and fresh.candidate_evaluations>candidates
    assert restored.native_histories==state.native_histories
    assert fresh._checkpoint(records)==raw
    assert not fresh._contact_reuse_enabled and fresh._contact_reuse_entry is None
    assert not fresh._assembly_reuse_enabled and fresh._assembly_reuse_entry is None
    # Cancellation requires its bound operation; operations do not nest.
    with pytest.raises(ValueError,match='bound operation'):
        owner.context.activate_cancellation(None)
    with owner.context.operation(None):
        with pytest.raises(RuntimeError,match='already in operation'):
            with owner.context.operation(None):pass
    assert owner.context._token is None and not owner.context._lock.locked()
    assert not owner._contact_reuse_enabled and owner._contact_reuse_entry is None
    assert not owner._assembly_reuse_enabled and owner._assembly_reuse_entry is None
