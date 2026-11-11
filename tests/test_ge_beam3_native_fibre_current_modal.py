"""Actual force-controlled fibre modal ownership, not translated histories."""
from dataclasses import replace
from hashlib import sha256
import numpy as np
import pytest
from scipy.linalg import eigh
from anyfesolver import _ge_beam3_native_fibre_current_modal as modal
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
from anyfesolver._ge_beam3_native_fibre_restart import decode_checkpoint,_forces
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver.control import CancellationToken,SolveCancelled
from test_ge_beam3_fibre_line_program import model as source_model

FORCES=((3,.35,-.012,.006),)


def make(curved=True,plastic=True,common=None):
    source=source_model(curved=curved,plastic=plastic,common=common)
    definitions=tuple(NativeBeamDefinition.capture(NativeFibreStaticElement(i,tuple(e.node_ids),
        e.operator.reference,e.section,order=4),np.diag([2.,2.,2.,.07,.09,.11]))
        for i,e in sorted(source.mesh.elements.items()))
    return NativeBeamAnalysis(definitions,tuple(source.boundary_conditions))


def decode(made,run):
    raw,digest=made._backend(run.checkpoint,run.checkpoint_sha256)
    forces,chain=decode_checkpoint(made.model,raw,expected_sha256=digest)
    state=chain[-1]
    external=state['load_factor']*_forces(forces,tuple(sorted(made.model.mesh.nodes)),
        made.model.mesh.dof_manager.total_dofs)
    return state,external


@pytest.fixture(scope='module')
def loaded():
    made=make();peak=made.solve_nodal(FORCES,steps=2)
    assert peak.status=='completed',peak.backend_result.info
    state,_=decode(made,peak)
    assert any(row[2]>0 for station in state['states'][1]['response'].history.stations for row in station.rows)
    unload=made.solve_nodal(FORCES,target_factor=.5,steps=1,checkpoint=peak.checkpoint,
        expected_sha256=peak.checkpoint_sha256)
    assert unload.status=='completed',unload.backend_result.info
    return peak,unload


def test_actual_unloaded_force_history_and_full_stationary_schur(loaded,tmp_path):
    peak,run=loaded;made=make();state,external=decode(made,run);before=canonical(state)
    packet,modes=made.current_modes(run.checkpoint,expected_sha256=run.checkpoint_sha256,
        bounds=(-100.,1e6),num_modes=6)
    accepted=state['states'][1]['response'];h=accepted.full_hessian
    expected=h[:24,:24]-h[:24,24:]@np.linalg.solve(h[24:,24:],h[24:,:24])
    np.testing.assert_allclose(packet.base.stiffness,expected,rtol=1e-11,atol=1e-11)
    a=list(packet.base.algebraic_dofs);p=[d for d in packet.base.free_dofs if d not in a]
    k=packet.base.stiffness;m=packet.base.mass
    reduced=k[np.ix_(p,p)]-k[np.ix_(p,a)]@np.linalg.solve(k[np.ix_(a,a)],k[np.ix_(a,p)])
    eigen=eigh(reduced,m[np.ix_(p,p)],eigvals_only=True)[:6]
    np.testing.assert_allclose(modes.eigenvalues,eigen,rtol=1e-11,atol=1e-11)
    assert packet.policy==modal.POLICY and packet.base.material_policy.startswith('PHYSICAL_FIBRE_')
    assert not packet.production_qualified and not packet.buckling_factor_authorized
    assert packet.base.mass.shape==(24,24) and not np.any(packet.base.mass[:,a])
    assert 'load_pattern' not in state['states'][1] and canonical(state)==before
    repeated=make().current_modes(run.checkpoint,expected_sha256=run.checkpoint_sha256,bounds=(-100.,1e6))
    assert canonical((packet,modes))==canonical(repeated)
    assert made.recover(run.checkpoint,expected_sha256=run.checkpoint_sha256)[1]['fibre_stress_status'].startswith('PHYSICAL_SECTION')
    (tmp_path/'force-checkpoint.json').write_bytes(run.checkpoint)
    (tmp_path/'force-modes.json').write_bytes(canonical(dict(packet=packet,modes=modes)))


def test_active_yield_not_relabelled_as_elastic(loaded):
    peak,_=loaded
    with pytest.raises(ValueError,match='history|yield|elastic'):
        make().current_modes(peak.checkpoint,expected_sha256=peak.checkpoint_sha256,bounds=(-100.,1e6))


def test_loaded_physical_mass_matches_independent_velocity_integral(loaded,tmp_path):
    _,run=loaded;made=make();state,external=decode(made,run)
    packet,guard=modal.prepare(made.model,state['states'],state['displacements'],made._inertias,external)
    e=made._elements[0];reference=e.operator.reference;rotations=state['states'][1]['response'].rotations
    curvature=reference.coordinates[0]-2*reference.coordinates[1]+reference.coordinates[2]
    speed=np.sin(np.arange(24)+.2);energy=0.;points,weights=np.polynomial.legendre.leggauss(64)
    for cell in (0,1):
        spin=speed[18+3*cell:21+3*cell]
        for point,weight in zip(points,weights):
            t=float((point+1)/2);xi=cell-1+t
            offset=rotations[cell]@(.5*t*(t-1)*curvature)
            velocity=(1-t)*speed[6*cell:6*cell+3]+t*speed[6*(cell+1):6*(cell+1)+3]+np.cross(spin,offset)
            frame=rotations[cell]@reference.frame(xi)
            field=np.r_[frame.T@velocity,frame.T@spin]
            energy+=float(weight*reference.jacobian(xi)*(field@made._inertias[1]@field)/4)
    error=abs(speed@packet.base.mass@speed/2-energy)/max(1.,abs(energy))
    assert error<=1e-11;guard()
    (tmp_path/'current-kinetic.json').write_bytes(canonical(dict(energy=energy,error=float(error))))


@pytest.mark.parametrize('curved',(False,True))
def test_elastic_force_path_and_reference_limit(curved,tmp_path):
    made=make(curved,False);run=made.solve_nodal(((3,.08,-.006,.003),),steps=1)
    assert run.status=='completed'
    packet,modes=made.current_modes(run.checkpoint,expected_sha256=run.checkpoint_sha256,bounds=(-100.,1e6))
    assert np.all(modes.eigenvalues>0.)
    raw=made.checkpoint_prefix(run.checkpoint,0,expected_sha256=run.checkpoint_sha256)
    virgin,virgin_modes=made.current_modes(raw,expected_sha256=sha256(raw).hexdigest(),bounds=(-100.,1e6))
    reference,reference_modes=made.reference_modes(bounds=(-100.,1e6))
    np.testing.assert_array_equal(virgin.base.mass,reference.mass)
    np.testing.assert_allclose(virgin.base.stiffness,reference.stiffness,atol=1e-11,rtol=1e-11)
    np.testing.assert_allclose(virgin_modes.eigenvalues,reference_modes.eigenvalues,atol=1e-11,rtol=1e-11)
    (tmp_path/'elastic-force-modes.json').write_bytes(canonical(dict(packet=packet,modes=modes)))


def test_loaded_covariance_under_arbitrary_common_rotation(tmp_path):
    transform=rotation([2.6,.8,-.3]);base=make(True,False);changed=make(True,False,transform)
    forces=((3,.08,-.006,.003),);moved=tuple((n,*map(float,transform@np.array(f))) for n,*f in forces)
    results=[]
    for made,load in ((base,forces),(changed,moved)):
        run=made.solve_nodal(load,steps=1);assert run.status=='completed'
        results.append(made.current_modes(run.checkpoint,expected_sha256=run.checkpoint_sha256,bounds=(-100.,1e6)))
    a,b=(r[0].base for r in results);mapping=np.kron(np.eye(8),transform)
    errors={}
    for name in ('stiffness','mass'):
        expected=mapping@getattr(a,name)@mapping.T
        error=np.linalg.norm(getattr(b,name)-expected)/max(1.,np.linalg.norm(expected))
        assert error<=1e-11;errors[name]=float(error)
    np.testing.assert_allclose(results[0][1].eigenvalues,results[1][1].eigenvalues,atol=1e-11,rtol=1e-11)
    (tmp_path/'force-covariance.json').write_bytes(canonical(errors))


@pytest.mark.parametrize('mutation',('state','inertia','model','packet','force','section'))
def test_full_custody_mutations_rejected(loaded,mutation):
    _,run=loaded;made=make();state,force=decode(made,run)
    packet,guard=modal.prepare(made.model,state['states'],state['displacements'],made._inertias,force)
    if mutation=='state':state['states'][1]['state_sha256']='0'*64
    elif mutation=='inertia':made._inertias[1]=2*made._inertias[1]
    elif mutation=='model':made.model.mesh.nodes[1].x+=.1
    elif mutation=='force':force[12]+=.1
    elif mutation=='section':
        section=made.model.mesh.elements[1].section
        object.__setattr__(section,'fibres',(replace(section.fibres[0],young=60.),*section.fibres[1:]))
    else:object.__setattr__(packet,'production_qualified',True)
    with pytest.raises(ValueError):guard()


def test_cancellation_and_explicit_bounds_precede_decode(monkeypatch):
    from anyfesolver import _ge_beam3_native_fibre_restart as restart
    made=make();token=CancellationToken();token.cancel()
    monkeypatch.setattr(restart,'decode_checkpoint',lambda *a,**kw:pytest.fail('decoded rejected request'))
    with pytest.raises(ValueError,match='explicit spectral bounds'):
        made.current_modes(b'invalid',expected_sha256='0'*64)
    with pytest.raises(SolveCancelled):
        made.current_modes(b'invalid',expected_sha256='0'*64,bounds=(-100.,1e6),cancellation_token=token)
    with pytest.raises(ValueError):
        made.current_modes(b'invalid',expected_sha256='0'*64,bounds=(1.,0.))
