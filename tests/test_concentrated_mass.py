import copy
import numpy as np
import pytest
from anysolver import FEModel, LoadCase, calculate_mass_properties, solve_inertia_relief, AnalysisSession
from anysolver.concentrated_mass import ConcentratedMass
from anysolver.matrix_assembly import assemble_mass_matrix
from test_inertia_relief import bar, rigid, load_vector


def test_spatial_mass_energy_gravity_and_properties():
    m=FEModel('mass'); m.add_node(1,3.,4.,5.)
    offset=np.array([2.,-1.,3.]); J=np.array([[5.,1.,0.],[1.,6.,.5],[0.,.5,7.]])
    m.add_concentrated_mass(1,10.,offset=offset,inertia_cg=J)
    M,_=assemble_mass_matrix(m)
    v=np.array([.2,.3,.4]); w=np.array([.5,-.2,.8]); q=np.r_[v,w]
    assert q@M@q == pytest.approx(10*np.dot(v+np.cross(w,offset),v+np.cross(w,offset))+w@J@w,rel=1e-13)
    props=calculate_mass_properties(m)
    assert props.total_mass == pytest.approx(10.)
    np.testing.assert_allclose(props.center_of_mass,[5.,3.,8.],atol=1e-13)
    np.testing.assert_allclose(props.inertia_tensor_center_of_mass,J,atol=1e-12)
    g=np.array([1.,2.,-9.81]); load=LoadCase('g'); load.gravity=g
    F=load.get_load_vector(m.mesh,m.mesh.dof_manager,m.get_material)
    np.testing.assert_allclose(F,np.r_[10*g,np.cross(offset,10*g)],atol=1e-12)
    np.testing.assert_array_equal(assemble_mass_matrix(copy.deepcopy(m))[0].toarray(),M.toarray())


@pytest.mark.parametrize('consistent',[True,False])
def test_eccentric_mass_relief_and_direct_mutation_cache(consistent):
    m=bar(consistent=consistent)
    m.add_concentrated_mass(4,10.,offset=(1.,2.,3.),inertia_cg=np.diag([2.,3.,4.]))
    session=AnalysisSession(m)
    M,_=assemble_mass_matrix(m); a=rigid(m,np.zeros(3))@np.array([.2,.1,-.3,.4,.5,-.2])
    result=solve_inertia_relief(m,load_vector(m,M@a),session=session)
    assert result.converged,result.to_dict()
    np.testing.assert_allclose(result.rigid_acceleration,a,atol=1e-10,rtol=1e-10)
    assert np.linalg.norm(result.displacements)<1e-12
    m.mesh.concentrated_masses[4][0]=ConcentratedMass(20.,(1.,2.,3.),np.diag([3.,4.,5.]))
    M,_=assemble_mass_matrix(m)
    result=solve_inertia_relief(m,load_vector(m,M@a),session=session)
    assert result.converged,result.to_dict()
    np.testing.assert_allclose(result.rigid_acceleration,a,atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('mass,J',[(1.,np.diag([-1.,2.,2.])),(1.,np.diag([1.,1.,3.])),(0.,np.eye(3)),(-1.,np.eye(3)),(1.,[[1,1,0],[0,1,0],[0,0,1]])])
def test_invalid_mass(mass,J):
    with pytest.raises(ValueError): ConcentratedMass(mass,inertia_cg=J)


def test_batch_matches_scalar_and_refuses_midstream_mass_mutation():
    from anysolver import iter_linear_batches, StaticCase, StaticBatchError
    m=bar(); m.add_concentrated_mass(4,5.,offset=(1.,2.,3.),inertia_cg=np.diag([2.,3.,4.]))
    M,_=assemble_mass_matrix(m)
    a=rigid(m,np.zeros(3))@np.array([.2,.1,-.3,.4,.5,-.2])
    load=load_vector(m,M@a)
    blocks=list(iter_linear_batches(m,[StaticCase('a',load)],inertia_relief=True))
    assert np.linalg.norm(blocks[0].displacements)<1e-12
    def cases():
        yield StaticCase('a',load)
        m.mesh.concentrated_masses[4][0]=ConcentratedMass(10.,(1.,2.,3.),np.diag([2.,3.,4.]))
        yield StaticCase('b',load)
    stream=iter_linear_batches(m,cases(),inertia_relief=True,batch_size=1)
    next(stream)
    with pytest.raises(StaticBatchError): next(stream)


def test_mesh_positional_constructor_stays_compatible():
    from anysolver.fe_core import FEMesh,DOFManager
    activity=object(); revisions={'mass':0}
    mesh=FEMesh({}, {}, DOFManager(), {}, activity, revisions)
    assert mesh.element_activity is activity and mesh.revisions is revisions
    assert mesh.concentrated_masses=={}


def test_legacy_pickle_state_without_concentrated_masses():
    import pickle
    from anysolver.fe_core import FEMesh
    m=bar(); state=m.mesh.__getstate__(); state.pop('concentrated_masses')
    mesh=FEMesh.__new__(FEMesh); mesh.__setstate__(state)
    assert mesh.concentrated_masses=={}
    m.add_concentrated_mass(4,5.,offset=(1.,0.,0.))
    restored=pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(assemble_mass_matrix(restored)[0].toarray(),assemble_mass_matrix(m)[0].toarray())


def test_exact_canonical_record_types():
    from anysolver.concentrated_mass import identity
    class Sub(ConcentratedMass): pass
    good=ConcentratedMass(1.)
    for bad in ({4:[Sub(1.)]},{4.0:[good]},{True:[good]},{4:good},{4:{good}}):
        m=bar(); m.mesh.concentrated_masses=bad
        with pytest.raises(ValueError): identity(m.mesh)
    m=bar(); m.mesh.concentrated_masses=[(4,[good])]
    with pytest.raises(ValueError): identity(m.mesh)
    m=bar(); m.mesh.concentrated_masses={4:(good,)}
    assert identity(m.mesh)==((4,1.,(0.,0.,0.),((0.,)*3,)*3),)


def test_rotated_inertia_roundoff_is_symmetrized():
    c,s=np.cos(.7),np.sin(.7); Rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    c,s=np.cos(-1.3),np.sin(-1.3); Rx=np.array([[1.,0,0],[0,c,-s],[0,s,c]])
    R=Rz@Rx; J=R@np.diag([3.,4.,5.])@R.T
    J[0,1]+=4*np.finfo(float).eps*5.
    value=ConcentratedMass(2.,inertia_cg=J)
    Js=np.array(value.inertia_cg)
    np.testing.assert_array_equal(Js,Js.T)
    np.testing.assert_allclose(np.linalg.eigvalsh(Js),[3.,4.,5.],rtol=1e-13)
    J[0,1]+=1e-9*5.
    with pytest.raises(ValueError): ConcentratedMass(2.,inertia_cg=J)


def test_empty_concentrated_mass_keeps_legacy_relief_identity():
    from anysolver.inertia_relief import _mass_identity
    m=bar(); m.add_point_mass(4,3.)
    assert _mass_identity(m)==((4,3.),)
    m.add_concentrated_mass(4,1.)
    assert _mass_identity(m)!=((4,3.),)


def test_pure_gravity_inertia_relief_with_eccentric_mass():
    m=bar(); m.add_concentrated_mass(4,10.,offset=(1.,2.,3.),inertia_cg=np.diag([2.,3.,4.]))
    g=np.array([.3,-.2,-9.81]); load=LoadCase('g'); load.gravity=g
    result=solve_inertia_relief(m,load)
    assert result.converged,result.to_dict()
    # Uniform gravity on a free body: rigid translation at g about the mass centre.
    np.testing.assert_allclose(result.acceleration_at_center_of_mass,g,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(result.angular_acceleration,0.,atol=1e-10)
    assert result.total_mass == pytest.approx(10.+5.*.2*2.)


def _native_mesh(concentrated):
    from types import SimpleNamespace
    from anysolver.fe_core import DOFManager
    dm=DOFManager(); dm.add_node(1)
    return SimpleNamespace(elements={1:object()},dof_manager=dm,nodes={1:None},point_masses={},
                           element_activity=None,concentrated_masses=concentrated)


def test_native_distributed_identity_rejects_concentrated_masses():
    from types import SimpleNamespace
    from anysolver._ge_beam3_native_generalized_program import _model_identity
    model=SimpleNamespace(mesh=_native_mesh({1:[ConcentratedMass(1.)]}),constraint_equations=[])
    with pytest.raises(ValueError,match='activity/dynamics'): _model_identity(model,16)
    # Control: without the mass admission proceeds to the element-authority check.
    model=SimpleNamespace(mesh=_native_mesh({}),constraint_equations=[])
    with pytest.raises(ValueError,match='exact distributed element'): _model_identity(model,16)


def test_native_fibre_load_admission_rejects_concentrated_masses(monkeypatch):
    import anysolver._ge_beam3_native_fibre_static_element as fibre
    from anysolver._ge_beam3_native_load_admission import assemble_nodal_forces
    class Fake:
        def _check(self,mesh): pass
    monkeypatch.setattr(fibre,'NativeFibreStaticElement',Fake)
    def mesh(concentrated):
        made=_native_mesh(concentrated); made.elements={1:Fake()}; return made
    guard=lambda **kwargs: None
    bad=mesh({1:[ConcentratedMass(1.)]})
    with pytest.raises(ValueError,match='activity/dynamics'):
        assemble_nodal_forces(LoadCase('n'),bad,bad.dof_manager,guard=guard)
    good=mesh({})
    np.testing.assert_array_equal(assemble_nodal_forces(LoadCase('n'),good,good.dof_manager,guard=guard),np.zeros(6))
    gravity=LoadCase('g'); gravity.gravity=np.array([0.,0.,-9.81])
    with pytest.raises(ValueError,match='gravity'):
        assemble_nodal_forces(gravity,good,good.dof_manager,guard=guard)


def test_whole_dict_replacement_reaches_cached_relief():
    m=bar(); m.add_concentrated_mass(4,10.,offset=(1.,2.,3.),inertia_cg=np.diag([2.,3.,4.]))
    session=AnalysisSession(m)
    a=rigid(m,np.zeros(3))@np.array([.2,.1,-.3,.4,.5,-.2])
    M,_=assemble_mass_matrix(m)
    assert solve_inertia_relief(m,load_vector(m,M@a),session=session).converged
    m.mesh.concentrated_masses={4:[ConcentratedMass(25.,(-1.,0.,2.),np.diag([3.,4.,5.]))]}
    M2,_=assemble_mass_matrix(m)
    assert abs(M2-M).max()>0
    result=solve_inertia_relief(m,load_vector(m,M2@a),session=session)
    assert result.converged,result.to_dict()
    np.testing.assert_allclose(result.rigid_acceleration,a,atol=1e-10,rtol=1e-10)
