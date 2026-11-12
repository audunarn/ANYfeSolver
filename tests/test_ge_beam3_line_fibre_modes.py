"""Loaded spectrum development from preserved states; no new nonlinear solves."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest

from anyfesolver import _ge_beam3_line_fibre_modes as native
from anyfesolver import _ge_beam3_fibre_line_program as control
from anyfesolver._ge_beam3_fibre_line_work import ReferenceLineForces
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from docs.reference_cases import ge_beam3_curved_moment_fixture as fixture
from docs.reference_cases import ge_beam3_centered_line_load_oracle as oracle
from test_ge_beam3_retained_fibre_modes import dense_reference

ROOT=Path('C:/Users/AudunArnesenNyhus/AppData/Local/ANYrelease')
BINDINGS={
    1:(22929,'680bbf90abb5311c5e606ce33418354ccbe78e2c7d186968577a115e22f61e36'),
    2:(41811,'1f11cef7d65542a764defb660200d385ad53b2be1be4d0d1d00f2384e604bca2'),
    4:(79669,'8cfae0b45a24defd313e60638aa9b3c88e6ec3ea8ea06789f84469970d55655b'),
}


def read(path,size,digest):
    if not path.is_file(): pytest.skip('Preserved external development state not installed; not qualification')
    raw=path.read_bytes()
    assert len(raw)==size and sha256(raw).hexdigest()==digest
    return raw


def case(macros=1):
    raw=read(ROOT/'ge-beam3-dead-line-20260907-v1'/f'checkpoint-{macros}.json',*BINDINGS[macros])
    model=fixture.model(macros); program=ForceProgram((.25,.5,1.),(),max_iterations=24)
    line=ReferenceLineForces(tuple((i,.3,-.2,.1) for i in range(1,macros+1)))
    inertias={i:np.diag([1.,1.,1.,.02,.01,.01]) for i in model.mesh.elements}
    return model,program,line,raw,inertias


@pytest.fixture(autouse=True)
def forbid_new_nonlinear_solves(monkeypatch):
    def fail(*args,**kwargs): raise AssertionError('Use the preserved accepted state; no nonlinear rerun')
    monkeypatch.setattr(control,'solve_force_program',fail)


@pytest.mark.parametrize('macros',[1,2,4])
def test_complete_potential_schur_and_independent_external_work(macros):
    model,p,line,raw,inertias=case(macros)
    packet,guard,parameter=native.prepare(model,p,raw,inertias,line_forces=line,material_policy=native.ALGORITHMIC)
    context=control.Context(model,p,line_forces=line); state,_=context.restore(raw)
    _,full,metrics,_=context.assemble(state.mechanical,parameter,state.origins)
    nodal=context.layout.nodal_count
    geometry=list(range(nodal))+[nodal+24*i+j for i in range(macros) for j in range(6)]
    resultants=[nodal+24*i+j for i in range(macros) for j in range(6,24)]
    reduced=full[np.ix_(geometry,geometry)]-full[np.ix_(geometry,resultants)]@np.linalg.solve(
        full[np.ix_(resultants,resultants)],full[np.ix_(resultants,geometry)])
    factor=packet.left@packet.right
    candidate=factor.T@factor+packet.geometric
    assert np.linalg.norm(candidate-reduced)<=1e-11*max(1.,np.linalg.norm(reduced))
    expected=np.zeros_like(packet.geometric)
    for i,element in enumerate(model.mesh.elements.values()):
        nodes=context.layout.nodes[i]; ref=element.operator.reference
        x=state.mechanical.positions[nodes]+state.mechanical.position_low[nodes]
        work=oracle.evaluate(ref.coordinates,x-ref.coordinates,state.mechanical.cell_rotations[i],line.rows[i][1:],order=4)
        slots=list(element.get_dof_mapping(model.mesh))+list(range(nodal+6*i,nodal+6*i+6))
        expected[np.ix_(slots,slots)]-=parameter*work.hessian[:24,:24]
    np.testing.assert_allclose(packet.external_potential_hessian,expected,atol=1e-14,rtol=1e-11)
    assert np.linalg.norm(expected)>1e-5
    assert not np.any(packet.kinetic[:,packet.algebraic]) and max(metrics)<=1e-11
    assert np.linalg.norm(candidate-candidate.T)<=1e-11*max(1.,np.linalg.norm(candidate))
    guard()


@pytest.mark.parametrize('macros',[1,2,4])
def test_saved_loaded_spectra_signed_ritz_and_dense_pencil(macros,tmp_path):
    model,p,line,raw,inertias=case(macros)
    packet,modes=native.solve_modes(model,p,raw,inertias,line_forces=line,material_policy=native.FROZEN,
        expected_checkpoint_sha256=sha256(raw).hexdigest(),bounds=(-1000.,10000.))
    (tmp_path/f'modes-{macros}.json').write_bytes(canonical(dict(packet=packet,modes=modes)))
    reference=dense_reference(packet)
    assert np.max(abs(reference-modes.eigenvalues)/np.maximum(1.,abs(reference)))<=1e-11
    assert modes.original_ritz_residual<=1e-11 and modes.spectral_residual<=1e-11
    assert modes.equilibrium_load_parameter==1. and modes.external_potential_hessian_included
    assert not modes.state_advanced and not modes.checkpoint_converted and not modes.production_qualified
    assert not modes.buckling_factor_authorized and not modes.certified_intervals
    assert modes.negative_eigenvalues_retained
    # Removing the required external contribution changes the actual spectrum.
    missing=replace(packet,geometric=packet.geometric-packet.external_potential_hessian)
    omitted=dense_reference(missing)
    assert np.max(abs(omitted-reference))>1e-7
    with pytest.raises(ValueError): packet.external_potential_hessian.setflags(write=True)
    assert sha256(raw).hexdigest()==BINDINGS[macros][1]


def test_accepted_plastic_history_has_distinct_frozen_and_algorithmic_policies(tmp_path):
    from test_ge_beam3_fibre_line_program import model
    raw=read(ROOT/'ge-beam3-native-line-20260907-065c21d/cycle-a/pytest/test_physical_plasticity_resta0/paused.json',
        25593,'679c09a857c12cdfefd9ea96d25cce590b7a37ae2e0f6633afd8a7c5716f4736')
    p=ForceProgram((.25,.5,1.,.5,0.,-.5,0.),(),max_iterations=24)
    line=ReferenceLineForces(((1,.7,.02,.03),)); inertias={1:np.diag([1.,1.,1.,.02,.01,.01])}
    spectra=[]
    for label,policy in (('frozen',native.FROZEN),('algorithmic',native.ALGORITHMIC)):
        packet,modes=native.solve_modes(model(plastic=True),p,raw,inertias,line_forces=line,
            material_policy=policy,bounds=(-1000.,10000.))
        (tmp_path/(label+'.json')).write_bytes(canonical(dict(packet=packet,modes=modes)))
        assert modes.equilibrium_load_parameter==1. and not modes.state_advanced
        assert np.max(abs(dense_reference(packet)-modes.eigenvalues)/np.maximum(1.,abs(modes.eigenvalues)))<=1e-11
        if policy==native.ALGORITHMIC:
            assert modes.interpretation=='LAST_INCREMENT_OPERATOR_ONLY_NOT_PHYSICAL_VIBRATION_AUTHORITY'
        spectra.append(modes.eigenvalues)
    assert np.all(spectra[1]<=spectra[0]+1e-10) and np.max(spectra[0]-spectra[1])>1e-3
    context=control.Context(model(plastic=True),p,line_forces=line); state,records=context.restore(raw)
    assert state.completed_targets==3 and context.checkpoint(records)==raw
    assert any(row[2]>0 for cell in state.histories for station in cell.stations for row in station.rows)


@pytest.mark.parametrize('mutation',['policy','line','hash','bytes','inertia','inertia_key','coordinate','exact','retained','cancel'])
def test_load_state_inertia_and_budget_guards(mutation):
    from anyfesolver.control import CancellationToken,SolveCancelled
    model,p,line,raw,inertias=case(); policy=native.FROZEN; kw={}; token=CancellationToken()
    if mutation=='policy': policy='AUTO'
    if mutation=='line': line=ReferenceLineForces(((1,.31,-.2,.1),))
    if mutation=='hash': kw['expected_checkpoint_sha256']='0'*64
    if mutation=='bytes': raw=bytearray(raw)
    if mutation=='inertia': inertias[1][0,0]=-1.
    if mutation=='inertia_key': inertias={True:inertias[1]}
    if mutation=='coordinate': kw['coordinate_limit']=True
    if mutation=='exact': kw['exact_dimension_limit']=65
    if mutation=='retained': kw['max_coordinates']=9999
    if mutation=='cancel': token.cancel()
    with pytest.raises(SolveCancelled if mutation=='cancel' else ValueError):
        native.solve_modes(model,p,raw,inertias,line_forces=line,material_policy=policy,
            bounds=(-1000.,10000.),cancellation_token=token,**kw)


@pytest.mark.parametrize('mutation',['model','load','cancel'])
def test_final_publication_guard_rejects_changes(mutation,monkeypatch):
    from anyfesolver.control import CancellationToken,SolveCancelled
    model,p,line,raw,inertias=case(); token=CancellationToken()
    def changed(*args,**kwargs):
        if mutation=='model': model.mesh.nodes[3].x+=.001
        if mutation=='load': object.__setattr__(line,'rows',((1,.31,-.2,.1),))
        if mutation=='cancel': token.cancel()
        return SimpleNamespace()
    monkeypatch.setattr(native,'solve_factor_chain_modes',changed)
    with pytest.raises(SolveCancelled if mutation=='cancel' else ValueError):
        native.solve_modes(model,p,raw,inertias,line_forces=line,material_policy=native.FROZEN,
            bounds=(-1000.,10000.),cancellation_token=token)


def test_saved_common_rotation_covariance(tmp_path):
    from test_ge_beam3_fibre_line_program import model
    from anyfesolver._ge_beam3_p5.algebra import rotation
    common=rotation([2.5,-.7,.8]); f=np.array([.03,-.02,.01]); p=ForceProgram((.5,1.),(),max_iterations=24)
    results=[]; folder=ROOT/'ge-beam3-native-line-20260907-065c21d/cycle-a/pytest/test_curved_load_reactions_cov0'
    inputs=(('E','curved.json',17169,'f8837ab5ded0d5aafe5ba1f9a1641800f84a0f2a3c5d66982c7773232ee39081',np.eye(3)),
            ('R','curved-rotated.json',17683,'5cd81db86d6f38d8bc99443e67ea3d56648aa599bc55d9134c648afa60e82a77',common))
    for label,name,size,digest,g in inputs:
        raw=read(folder/name,size,digest); line=ReferenceLineForces(((1,*map(float,g@f)),))
        packet,modes=native.solve_modes(model(curved=True,common=g),p,raw,{1:np.diag([1.,1.,1.,.02,.01,.01])},
            line_forces=line,material_policy=native.FROZEN,bounds=(-1000.,10000.))
        (tmp_path/(label+'.json')).write_bytes(canonical(dict(packet=packet,modes=modes)))
        results.append(modes.eigenvalues)
    assert np.max(abs(results[0]-results[1])/np.maximum(1.,abs(results[0])))<=1e-11
