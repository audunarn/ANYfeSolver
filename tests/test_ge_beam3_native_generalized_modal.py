"""Current native generalized mass/pencil development, not full qualification."""
from copy import deepcopy
import numpy as np
import pytest
from anyfesolver import _ge_beam3_native_generalized_modal as modal
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_p5_centered.mass import reference_kinetic_factors
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_native_generalized_program import solve_distributed_model
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.control import CancellationToken,SolveCancelled
from test_ge_beam3_schur_line_program import save


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def inertia():
    s=skew([.03,-.02,.01]);m=2.
    return np.block([[m*np.eye(3),-m*s],[m*s,np.diag([.07,.09,.11])-m*s@s]])


def make(curved=False,macros=1,*,clamped=False,transform=None,reversed=False,coupled=True):
    c=np.diag([1000.,400.,350.,.8,1.,1.2])
    if coupled:
        c[0,3]=c[3,0]=.04;c[1,5]=c[5,1]=.03
    s=np.eye(3) if transform is None else transform
    x=np.linspace(0.,2.,2*macros+1);points=[];frames=[]
    for xi in x:
        position=np.array([xi,.2*(1-(xi-1)**2) if curved else 0.,0.])
        tangent=np.array([1.,.4*(1-xi) if curved else 0.,0.]);tangent/=np.linalg.norm(tangent)
        second=np.array([0.,0.,1.])
        points.append(s@position);frames.append(s@np.column_stack((tangent,second,np.cross(tangent,second))))
    mass=inertia() if coupled else np.diag([2.,2.,2.,.07,.09,.11])
    if reversed:
        d=np.diag([-1.,1.,-1.]);work=np.kron(np.eye(2),d)
        points=points[::-1];frames=[f@d for f in frames[::-1]]
        c=work@c@work;mass=work@mass@work
    model=FEModel('native-generalized-modal')
    for i,point in enumerate(points,1):model.add_node(i,*point)
    law=EllipsoidalGeneralizedSection(c,np.eye(6),1e6,1.)
    for index in range(macros):
        ids=tuple(range(2*index+1,2*index+4))
        ref=CenteredCurvedBeam3ReferenceGeometry(np.array(points[2*index:2*index+3]),np.array(frames[2*index:2*index+3]))
        e=NativeGeneralizedStaticElement(index+1,ids,ref,law,order=4)
        model.add_element(index+1,e);model.materials[e.material_name]=e.section
    if clamped:model.add_boundary_condition(BoundaryCondition('clamped',[1],{k:0. for k in ('ux','uy','uz','rx','ry','rz')}))
    states={i:e.init_model_bound_nonlinear_state(model.mesh,e.section,1) for i,e in model.mesh.elements.items()}
    inertias={i:mass for i in model.mesh.elements}
    return model,states,inertias


def close(a,b):
    error=float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(b)))
    assert error<=1e-11
    return error


@pytest.mark.parametrize('curved', (False,True))
def test_reference_mass_rigids_and_modal_pencil(curved,tmp_path):
    m,states,masses=make(curved);before=canonical(states)
    packet,modes=modal.solve_modes(m,states,np.zeros(18),masses,np.zeros(18),num_modes=15)
    e=m.mesh.elements[1]
    factors=reference_kinetic_factors(e.operator.reference,e.section.elastic,masses[1],order=4)
    kinetic=reference_kinetic_factors(e.operator.reference,e.section.elastic,masses[1],order=24)
    errors=dict(stiffness=close(packet.stiffness,factors.uncondensed_stiffness_factor.T@factors.uncondensed_stiffness_factor),
        mass=close(packet.mass,kinetic.full.T@kinetic.full))
    rigid=np.zeros((24,6))
    for i,x in enumerate(e.operator.reference.coordinates):
        rigid[6*i:6*i+3,:3]=np.eye(3);rigid[6*i:6*i+3,3:]=-skew(x)
        rigid[6*i+3:6*i+6,3:]=np.eye(3)
    rigid[18:21,3:]=rigid[21:24,3:]=np.eye(3)
    errors['rigid_null']=close(packet.stiffness@rigid,np.zeros((24,6)))
    np.linalg.cholesky(rigid.T@packet.mass@rigid)
    assert np.linalg.matrix_rank(packet.mass)==15 and modes.dynamic_map.shape==(24,15)
    assert np.array_equal(packet.mass[:,packet.algebraic_dofs],np.zeros((24,9)))
    scale=max(1.,np.linalg.norm(packet.stiffness)/np.linalg.norm(packet.mass))
    assert np.max(np.abs(modes.eigenvalues[:6]))<=1e-11*scale
    assert np.min(modes.eigenvalues[6:])>1e-11*scale
    assert modes.normalized_residual<=1e-11 and canonical(states)==before
    assert not packet.production_qualified and not packet.buckling_factor_authorized
    save(tmp_path/'reference.json',dict(packet=packet,modes=modes,errors=errors,rigid_kinetic=rigid.T@packet.mass@rigid))


@pytest.mark.parametrize('curved', (False,True))
def test_independent_velocity_integral(curved,tmp_path):
    m,states,masses=make(curved);e=m.mesh.elements[1]
    packet,_=modal.prepare(m,states,np.zeros(18),masses,np.zeros(18))
    speed=np.sin(np.arange(24)+.2);energy=0.
    points,weights=np.polynomial.legendre.leggauss(64)
    coords=e.operator.reference.coordinates;b=coords[0]-2*coords[1]+coords[2]
    for cell in (0,1):
        spin=speed[18+3*cell:21+3*cell]
        for p,w in zip(points,weights):
            t=float((p+1)/2);xi=cell-1+t
            lift=.5*t*(t-1)*b
            velocity=(1-t)*speed[6*cell:6*cell+3]+t*speed[6*(cell+1):6*(cell+1)+3]+np.cross(spin,lift)
            frame=e.operator.reference.frame(xi)
            field=np.r_[frame.T@velocity,frame.T@spin]
            energy+=float(w*e.operator.reference.jacobian(xi)*(field@masses[1]@field)/4)
    error=close(np.array(speed@packet.mass@speed/2),np.array(energy))
    save(tmp_path/'kinetic.json',dict(energy=energy,matrix_energy=float(speed@packet.mass@speed/2),error=error))


def test_connected_shared_trace_elimination(tmp_path):
    m,states,masses=make(True,2)
    n=m.mesh.dof_manager.total_dofs
    packet,modes=modal.solve_modes(m,states,np.zeros(n),masses,np.zeros(n),num_modes=15)
    assert packet.mass.shape==(42,42) and len(packet.algebraic_dofs)==15
    assert modes.dynamic_map.shape==(42,27)
    assert np.max(np.abs(modes.eigenvalues[:6]))<1e-8 and np.min(modes.eigenvalues[6:])>1e-8
    np.linalg.cholesky(modes.dynamic_map.T@packet.mass@modes.dynamic_map)
    save(tmp_path/'connected.json',dict(packet=packet,modes=modes))


@pytest.mark.parametrize('reversed', (False,True))
def test_frame_covariance_and_reversal(reversed,tmp_path):
    m,states,masses=make(True);base,_=modal.prepare(m,states,np.zeros(18),masses,np.zeros(18))
    s=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    other,history,inertias=make(True,transform=None if reversed else s,reversed=reversed)
    changed,_=modal.prepare(other,history,np.zeros(18),inertias,np.zeros(18))
    if reversed:
        slots=list(range(12,18))+list(range(6,12))+list(range(6))+list(range(21,24))+list(range(18,21))
        transform=np.eye(24)[slots]
    else:transform=np.kron(np.eye(8),s)
    errors=dict(mass=close(changed.mass,transform@base.mass@transform.T),
        stiffness=close(changed.stiffness,transform@base.stiffness@transform.T))
    save(tmp_path/'covariance.json',errors)


def test_loaded_conservative_state_preserves_history(tmp_path):
    m,_,masses=make(True,clamped=True)
    pattern=DistributedPattern(LinePattern(((1,.05,-.02,.03),)),())
    result,_=solve_distributed_model(m,pattern,steps=1)
    assert result.status=='completed',result.info
    states=result.element_states;before=canonical(states)
    packet,modes=modal.solve_modes(m,states,result.displacements,masses,np.zeros(18),num_modes=8)
    assert np.all(modes.eigenvalues>0.) and canonical(states)==before
    assert all(op.history_unchanged for _,op in packet.operators)
    save(tmp_path/'loaded.json',dict(packet=packet,modes=modes,state_sha256={i:s['state_sha256'] for i,s in states.items()}))


@pytest.mark.parametrize('mutation',('inertia','nodal_moment','state','map','equilibrium','couple','support','caller_change'))
def test_input_fail_closed(mutation,tmp_path):
    m,states,masses=make(clamped=True);total=np.zeros(18);force=np.zeros(18)
    if mutation=='inertia':masses[1]=np.zeros((6,6))
    elif mutation=='nodal_moment':force[9]=.1
    elif mutation=='state':states[1]['state_sha256']='0'*64
    elif mutation=='map':masses={}
    elif mutation=='equilibrium':force[12]=.1
    elif mutation=='couple':states[1]['load_pattern']=DistributedPattern(LinePattern(()),((1,.1,0.,0.),))
    elif mutation=='support':m.add_boundary_condition(BoundaryCondition('partial',[3],{'rx':0.}))
    else:
        _,guard=modal.prepare(m,states,total,masses,force)
        force[12]=.1
        with pytest.raises(ValueError,match='inputs changed'):guard()
        save(tmp_path/'rejection.json',dict(mutation=mutation,rejected=True));return
    with pytest.raises((ValueError,np.linalg.LinAlgError)):
        modal.prepare(m,states,total,masses,force)
    save(tmp_path/'rejection.json',dict(mutation=mutation,rejected=True))


def test_cancellation_and_static_default_remain_closed(tmp_path):
    m,states,masses=make();token=CancellationToken();token.cancel()
    with pytest.raises(SolveCancelled):
        modal.prepare(m,states,np.zeros(18),masses,np.zeros(18),cancellation_token=token)
    with pytest.raises(ValueError,match='workflow not qualified'):
        m.mesh.elements[1].compute_mass_matrix()
    save(tmp_path/'boundary.json',dict(cancelled=True,static_mass_route_unchanged=True))


def test_straight_axial_and_torsional_frequency_convergence(tmp_path):
    from anyfesolver._native_stationary_spectrum import solve_stationary_spectrum
    rows=[]
    for macros in (1,2,4,8):
        m,states,masses=make(macros=macros,clamped=True,coupled=False)
        n=m.mesh.dof_manager.total_dofs
        packet,_=modal.prepare(m,states,np.zeros(n),masses,np.zeros(n))
        axial=tuple(d for node in sorted(m.mesh.nodes) for d in [m.mesh.dof_manager.get_node_dofs(node)[0]] if d in packet.free_dofs)
        traces=tuple(d for node in sorted(m.mesh.nodes) for d in [m.mesh.dof_manager.get_node_dofs(node)[3]] if d in packet.free_dofs)
        cells=tuple(d for _,slots in packet.internal_layout for d in slots[::3])
        torsion=traces+cells
        errors={}
        for family,free,algebraic,rigidity,density in (
            ('axial',axial,(),1000.,2.),('torsion',torsion,traces,.8,.07)):
            excluded=tuple(i for i in packet.free_dofs if i not in free)
            assert np.max(np.abs(packet.stiffness[np.ix_(free,excluded)]))<=1e-11
            assert np.max(np.abs(packet.mass[np.ix_(free,excluded)]))<=1e-11
            modes=solve_stationary_spectrum(packet.stiffness,packet.mass,free,algebraic,num_modes=1)
            omega=float(np.sqrt(modes.eigenvalues[0]));reference=float(np.pi/4*np.sqrt(rigidity/density))
            errors[family]=abs(omega/reference-1.)
        rows.append(dict(macros=macros,errors=errors));print(rows[-1],flush=True)
    for family in ('axial','torsion'):
        assert all(b['errors'][family]<a['errors'][family] for a,b in zip(rows,rows[1:]))
        assert rows[-1]['errors'][family]<.02
    save(tmp_path/'frequency.json',dict(rows=rows,reference='FIXED_FREE_UNIFORM_ROD_PI_OVER_2L_SQRT_RIGIDITY_OVER_INERTIA',
        bending_and_curved_continuum_qualified=False))
