"""Integrated shell, MPC, lifecycle and refusal contracts for inertia relief."""
import json

import numpy as np
import pytest

from anyfesolver import (AnalysisSession, BeamElement, CancellationToken, FEModel, FixedSupport,
                       InertiaReliefError, LoadCase, SolveCancelled, create_shell_element,
                       solve_inertia_relief, solve_linear)
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.elements import CoupledBeamShellElement
from anyfesolver.matrix_assembly import assemble_mass_matrix, assemble_stiffness_matrix


def beam():
    model = FEModel("relief-integration")
    model.add_material("steel", 1.e7, .3, density=5.)
    model.add_node(1, 0., 0., 0.)
    model.add_node(2, 2., 0., 0.)
    model.add_element(1, BeamElement(1, [1,2], "steel",
        {"area": .2,"Iy": .001,"Iz": .002,"J": .001,"consistent_mass":True}))
    return model


def panel(thickness=.05):
    model = FEModel("relief-panel")
    model.add_material("steel", 1.e7, .3, density=5.)
    for i, xyz in enumerate(((0.,0.,0.),(2.,0.,0.),(2.,1.,0.),(0.,1.,0.)),1):
        model.add_node(i,*xyz)
    model.add_element(1,create_shell_element(1,[1,2,3,4],"steel",thickness=thickness))
    return model


def vector_load(model, vector):
    load = LoadCase("manufactured")
    for node in model.mesh.nodes.values():
        load.add_nodal_load(node.id,np.asarray(vector)[node.dofs])
    return load


def physical_rigid(model, origin):
    R=np.zeros((model.mesh.dof_manager.total_dofs,6))
    for node in model.mesh.nodes.values():
        R[np.ix_(node.dofs[:3],range(3))]=np.eye(3)
        for j in range(3):
            R[node.dofs[:3],j+3]=np.cross(np.eye(3)[j],node.coords()-origin)
        R[np.ix_(node.dofs[3:],range(3,6))]=np.eye(3)
    return R


@pytest.mark.parametrize("thickness", [.05, .001])
def test_shell_manufactured_elastic_strain_and_rigid_acceleration(thickness):
    model=panel(thickness)
    K,_=assemble_stiffness_matrix(model)
    M,_=assemble_mass_matrix(model)
    strain=1e-4
    u=np.zeros(K.shape[0])
    for node in model.mesh.nodes.values(): u[node.dofs[0]]=strain*(node.x-1.)
    R=physical_rigid(model,np.array([1.,.5,0.]))
    a=np.array([.2,-.3,.4,.1,.2,-.1])
    result=solve_inertia_relief(model,vector_load(model,K@u+M@(R@a)))
    assert result.converged, result.to_dict()
    np.testing.assert_allclose(result.angular_acceleration,a[3:],rtol=1e-10,atol=1e-12)
    # Independent physical strain from differences, independent of displacement gauge.
    np.testing.assert_allclose(result.displacements[6]-result.displacements[0],2.*strain,rtol=1e-8,atol=1e-12)
    recovered=result.recover()
    assert recovered.element_stresses
    assert recovered.reactions=={}
    assert result.diagnostics["mechanism"]["eigenpair_residual"]<=1e-13
    assert result.result_case["analysis_case"]["analysis_type"]=="inertia_relief"
    json.dumps(result.to_dict(),allow_nan=False)


def test_uniform_shell_gravity_is_free_fall_without_strain():
    model=panel()
    load=LoadCase("gravity",gravity=np.array([0.,0.,-9.81]))
    result=solve_inertia_relief(model,load)
    assert result.converged,result.to_dict()
    np.testing.assert_allclose(result.acceleration_at_center_of_mass,[0.,0.,-9.81],atol=1e-10)
    assert np.linalg.norm(result.displacements)<1e-12


def test_eccentric_mpc_slave_load_matches_independent_dense_kkt():
    model=panel()
    model.add_node(5,0.,0.,.2)
    model.add_node(6,2.,0.,.2)
    section={"area":.03,"Iy":.0001,"Iz":.0002,"J":.0001,"consistent_mass":True}
    model.add_element(2,BeamElement(2,[5,6],"steel",section))
    model.add_element(3,CoupledBeamShellElement(3,5,1,"steel"))
    model.add_element(4,CoupledBeamShellElement(4,6,2,"steel"))
    load=LoadCase("slave-force")
    load.add_nodal_load(6,forces=[1.,2.,3.],moments=[.2,.1,.3])
    result=solve_inertia_relief(model,load)
    assert result.converged,result.to_dict()
    K,_=assemble_stiffness_matrix(model)
    M,_=assemble_mass_matrix(model)
    # Independent projection of the full applied vector and dense constrained system.
    R=physical_rigid(model,np.zeros(3))
    a=np.linalg.solve(R.T@(M@R),R.T@result.applied_loads)
    expected_load=result.applied_loads-M@(R@a)
    Kr,Fr,T,_,ind,_=build_constraint_transformation(K,expected_load,model)
    Q,_=np.linalg.qr(R[ind,:])
    B=np.block([[Kr.toarray(),Q],[Q.T,np.zeros((6,6))]])
    q=np.linalg.solve(B,np.r_[Fr,np.zeros(6)])[:len(ind)]
    np.testing.assert_allclose(result.displacements,T@q,rtol=1e-8,atol=1e-11)
    np.testing.assert_allclose(result.inertial_loads,-M@(R@a),rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(T.T@(K@result.displacements-result.effective_loads),0.,atol=1e-9)
    assert result.constraint_forces["support_reactions"]=={}
    assert result.constraint_forces["mpc_slave_force_norm"]>0
    assert np.linalg.norm(result.constraint_forces["residual"])>1e-5
    assert result.recover().reactions=={}


def test_rigid_rotation_and_large_translation_preserve_physics():
    original=beam()
    load=LoadCase("force")
    force=np.array([2.,3.,4.]); moment=np.array([.3,.2,.1])
    load.add_nodal_load(2,forces=force,moments=moment)
    first=solve_inertia_relief(original,load)
    assert first.converged,first.to_dict()
    angle=.7
    C=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    rotated=beam()
    offset=np.array([1e5,-2e5,3e5])
    for node in rotated.mesh.nodes.values():
        rotated.set_node_coordinates(node.id,*(C@node.coords()+offset))
    other_load=LoadCase("rotated")
    other_load.add_nodal_load(2,forces=C@force,moments=C@moment)
    second=solve_inertia_relief(rotated,other_load)
    assert second.converged,second.to_dict()
    np.testing.assert_allclose(second.acceleration_at_center_of_mass,C@first.acceleration_at_center_of_mass,rtol=1e-10,atol=1e-10)
    # Section orientation is covariant for rotation about global Z.
    np.testing.assert_allclose(second.angular_acceleration,C@first.angular_acceleration,rtol=1e-10,atol=1e-9)
    for i in range(0,len(first.inertial_loads),3):
        np.testing.assert_allclose(second.inertial_loads[i:i+3],C@first.inertial_loads[i:i+3],rtol=1e-9,atol=1e-9)


def test_session_load_mass_geometry_material_and_constraint_invalidation():
    model=beam()
    load=LoadCase("axial"); load.add_nodal_load(2,forces=[2.,0.,0.])
    with AnalysisSession(model) as session:
        initial=solve_inertia_relief(model,load,session=session)
        assert initial.converged,initial.to_dict()
        load.nodal_loads[2]*=2.
        doubled=solve_inertia_relief(model,load,session=session)
        np.testing.assert_allclose(doubled.displacements,2.*initial.displacements,rtol=1e-8)
        model.mesh.point_masses[2]=3.
        changed=solve_inertia_relief(model,load,session=session)
        oracle=solve_inertia_relief(model,load)
        assert changed.converged and oracle.converged
        np.testing.assert_allclose(changed.displacements,oracle.displacements,rtol=1e-8,atol=1e-12)
        assert changed.total_mass==5.
        assert changed.acceleration_at_center_of_mass[0]==pytest.approx(.8)
        with pytest.raises(InertiaReliefError,match="unchanged model"):
            initial.recover()
        model.materials["steel"].density=10.; model.bump_revision("material")
        model.set_node_coordinates(2,3.,0.,0.)
        updated=solve_inertia_relief(model,load,session=session)
        assert updated.converged,updated.to_dict()
        assert updated.total_mass==pytest.approx(9.)
        fresh=solve_inertia_relief(model,load)
        np.testing.assert_allclose(updated.inertial_loads,fresh.inertial_loads,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(updated.displacements,fresh.displacements,rtol=1e-8,atol=1e-12)


@pytest.mark.parametrize("case",["support","offset","restraint","disconnected","point_only","follower","load_mass","solid","native"])
def test_unsupported_inputs_refuse_without_definition_mutation(case):
    model=beam(); load=LoadCase("bad")
    if case=="support": model.add_boundary_condition(FixedSupport("fixed",[1]))
    elif case in ("offset","restraint"):
        a=model.mesh.nodes[1].dofs[0]; b=model.mesh.nodes[2].dofs[0]
        model.add_constraint_equation(terms=((b,1.),(a,-1. if case=="offset" else -2.)),rhs=.1 if case=="offset" else 0.,source_id=case)
    elif case in ("disconnected","point_only"):
        model.add_node(3,10.,0.,0.)
        if case=="point_only": model.add_point_mass(3,1.)
    elif case=="follower": load.follower_pressure=True
    elif case=="load_mass": load.added_node_masses[2]=1.
    elif case=="solid":
        class Solid: _solid_model_protocol=True
        with pytest.raises(InertiaReliefError): solve_inertia_relief(Solid())
        return
    elif case=="native": model.mesh.elements[1].formulation_native_total_lagrangian=True
    before=model.revision_signature().copy()
    with pytest.raises(InertiaReliefError): solve_inertia_relief(model,load)
    assert model.revision_signature()==before


@pytest.mark.parametrize("stage",["before","mechanism","after_assembly"])
def test_cancellation_propagates_without_accepted_result(stage):
    model=beam(); token=CancellationToken()
    if stage=="before": token.cancel("test")
    def callback(event):
        if stage=="after_assembly" and event["stage"]=="inertia_relief.assembly": token.cancel("assembly")
        if stage=="mechanism" and event["stage"]=="inertia_relief.mechanism_check": token.cancel("mechanism")
    before=model.revision_signature().copy()
    with pytest.raises(SolveCancelled): solve_inertia_relief(model,cancellation_token=token,progress_callback=callback)
    assert model.revision_signature()==before


def test_original_free_free_refusal_and_supported_static_behavior_remain():
    model=beam(); load=LoadCase("force"); load.add_nodal_load(2,forces=[2.,0.,0.])
    _,info=solve_linear(model,load)
    assert info["convergence_info"]["status"]=="incompatible_free_free_load"
    model.add_boundary_condition(FixedSupport("fixed",[1]))
    u,info=solve_linear(model,load)
    assert info["convergence_info"]["status"]=="converged"
    assert u[6]==pytest.approx(2.*2./(1.e7*.2))


@pytest.mark.parametrize("mass_policy,load_policy,ratio",[
    ("delete_only","activity",.375),
    ("activity","delete_only",1./.375),
    ("activity","activity",1.),
])
def test_distinct_activity_policies_and_session_updates(mass_policy,load_policy,ratio):
    from anyfesolver.activity import ElementActivity, ElementActivityPolicy
    model=beam()
    activity=ElementActivity([1],.375,policy=ElementActivityPolicy(mass=mass_policy,load=load_policy))
    model.set_element_activity(activity)
    load=LoadCase("gravity",gravity=np.array([1.,-2.,3.]))
    with AnalysisSession(model) as session:
        first=solve_inertia_relief(model,load,session=session)
        assert first.converged,first.to_dict()
        np.testing.assert_allclose(first.acceleration_at_center_of_mass,ratio*load.gravity,rtol=1e-10,atol=1e-12)
        activity.set_activity([1],.125,reason="test update")
        with pytest.raises(InertiaReliefError,match="unchanged model"):
            first.recover()
        changed=solve_inertia_relief(model,load,session=session)
        expected=solve_inertia_relief(model,load)
        assert changed.converged and expected.converged
        np.testing.assert_allclose(changed.inertial_loads,expected.inertial_loads,rtol=1e-10,atol=1e-12)


def test_hard_deleted_stiffness_with_retained_mass_is_unresolved():
    from anyfesolver.activity import ElementActivity,ElementActivityPolicy
    model=beam()
    activity=ElementActivity([1],0.,policy=ElementActivityPolicy(mass="retain"))
    model.set_element_activity(activity)
    result=solve_inertia_relief(model)
    assert result.status=="unresolved_elastic_stiffness"
    assert result.displacements is None


def test_semidefinite_full_mass_with_massless_rotations_is_supported():
    class TranslationMassBeam(BeamElement):
        def compute_mass_matrix(self,mesh,material):
            matrix=super().compute_mass_matrix(mesh,material).copy()
            rotations=[3,4,5,9,10,11]
            matrix[rotations,:]=0.;matrix[:,rotations]=0.
            return matrix
    model=beam()
    model.add_node(3,2.,1.,0.)
    model.mesh.elements[1]=TranslationMassBeam(1,[1,2],"steel",dict(model.mesh.elements[1].cross_section))
    model.add_element(2,TranslationMassBeam(2,[2,3],"steel",dict(model.mesh.elements[1].cross_section)))
    M,_=assemble_mass_matrix(model)
    assert np.linalg.matrix_rank(M.toarray())<M.shape[0]
    R=physical_rigid(model,np.zeros(3)); a=np.array([1.,2.,3.,.1,.2,.3])
    result=solve_inertia_relief(model,vector_load(model,M@(R@a)))
    assert result.converged,result.to_dict()
    np.testing.assert_allclose(result.rigid_acceleration,R@a,rtol=1e-10,atol=1e-11)


def test_lanczos_nonconvergence_fails_closed(monkeypatch):
    import anyfesolver.inertia_relief as module
    from scipy.sparse.linalg import ArpackNoConvergence
    def fail(*args,**kwargs):
        raise ArpackNoConvergence("forced nonconvergence",np.empty(0),np.empty((12,0)))
    monkeypatch.setattr(module,"eigsh",fail)
    result=solve_inertia_relief(beam())
    assert result.status=="unresolved_elastic_stiffness"
    assert result.diagnostics["mechanism"]["error"]=="Lanczos did not converge"
    assert result.displacements is None


def test_linearity_and_cancellation_inside_lanczos():
    model=beam()
    first=LoadCase("first");first.add_nodal_load(2,forces=[2.,3.,1.])
    second=LoadCase("second");second.add_nodal_load(1,moments=[.3,.1,.2])
    summed=LoadCase("sum")
    summed.add_nodal_load(2,first.nodal_loads[2]);summed.add_nodal_load(1,second.nodal_loads[1])
    a=solve_inertia_relief(model,first);b=solve_inertia_relief(model,second);c=solve_inertia_relief(model,summed)
    assert a.converged and b.converged and c.converged
    np.testing.assert_allclose(c.displacements,a.displacements+b.displacements,rtol=1e-8,atol=1e-12)
    np.testing.assert_allclose(c.rigid_acceleration,a.rigid_acceleration+b.rigid_acceleration,rtol=1e-10,atol=1e-11)
    class IterationCancel(CancellationToken):
        count=0
        def raise_if_cancelled(self,stage=""):
            if stage=="inertia_relief.mechanism_iteration":
                self.count+=1
                if self.count==3:self.cancel("inside Lanczos")
            super().raise_if_cancelled(stage)
    token=IterationCancel()
    with pytest.raises(SolveCancelled,match="inside Lanczos"):
        solve_inertia_relief(model,cancellation_token=token)
    assert token.count==3


@pytest.mark.parametrize("factor",[1.,1e6,1e12])
def test_rigid_dominated_loads_are_invariant_to_density_scale(factor):
    model=beam()
    model.materials["steel"].density*=factor
    model.bump_revision("material")
    load=LoadCase("scaled gravity",gravity=np.array([1.23,-2.34,3.45]))
    result=solve_inertia_relief(model,load)
    assert result.converged,result.to_dict()
    np.testing.assert_allclose(result.acceleration_at_center_of_mass,load.gravity,rtol=1e-10,atol=1e-11)
    assert result.diagnostics["elastic_load_cancellation_ratio"]<1e-14
    assert result.diagnostics["elastic_backward_error"]<=1e-10


def test_recovery_refuses_revision_and_direct_constraint_changes():
    from anyfesolver import ConstraintEquation
    model=beam()
    result=solve_inertia_relief(model)
    assert result.converged
    model.bump_revision("material")
    with pytest.raises(InertiaReliefError,match="unchanged model"):
        result.recover()
    a=model.mesh.nodes[1].dofs[0];b=model.mesh.nodes[2].dofs[0]
    model.add_constraint_equation(terms=((b,1.),(a,-1.)),source_id="internal link")
    result=solve_inertia_relief(model)
    assert result.converged,result.to_dict()
    model.constraint_equations[0]=ConstraintEquation(terms=((b,1.),(a,-2.)),source_id="changed link")
    with pytest.raises(InertiaReliefError,match="unchanged model constraints"):
        result.recover()


def test_nonfinite_operator_produces_typed_refusal_and_arpack_error_is_numerical(monkeypatch):
    import anyfesolver.inertia_relief as module
    from scipy.sparse import eye
    from scipy.sparse.linalg import ArpackError
    original=module.assemble_mass_matrix
    def bad_mass(model):
        matrix=eye(model.mesh.dof_manager.total_dofs,format="csr")
        matrix.data[0]=np.nan
        return matrix,{}
    monkeypatch.setattr(module,"assemble_mass_matrix",bad_mass)
    with pytest.raises(InertiaReliefError,match="nonfinite"):
        solve_inertia_relief(beam())
    monkeypatch.setattr(module,"assemble_mass_matrix",original)
    def arpack_error(*args,**kwargs):raise ArpackError(-9999)
    monkeypatch.setattr(module,"eigsh",arpack_error)
    result=solve_inertia_relief(beam())
    assert result.status=="unresolved_elastic_stiffness"
    assert result.displacements is None
    json.dumps(result.to_dict(),allow_nan=False)


def test_refined_thin_shell_spectral_classification():
    model=FEModel("thin-panel")
    model.add_material("steel",1.e7,.3,density=5.)
    n=4
    for j in range(n+1):
        for i in range(n+1):model.add_node(1+j*(n+1)+i,2.*i/n,2.*j/n,0.)
    for j in range(n):
        for i in range(n):
            a=1+j*(n+1)+i
            model.add_element(1+j*n+i,create_shell_element(1+j*n+i,[a,a+1,a+n+2,a+n+1],"steel",thickness=.0002))
    result=solve_inertia_relief(model)
    assert result.converged,result.to_dict()
    assert result.diagnostics["mechanism"]["eigenvalue_lower_margin"]>1e-12


def test_declared_linear_capability_restriction_is_preserved():
    model=beam()
    model.mesh.elements[1].capability_restrictions={"linear_static":"prestressed_profile_only"}
    with pytest.raises(InertiaReliefError,match="capability disposition"):
        solve_inertia_relief(model)


@pytest.mark.parametrize("bad",["duplicate","outside","nonnumeric"])
def test_invalid_node_dof_mappings_have_typed_refusal(bad):
    model=beam()
    if bad=="duplicate":model.mesh.nodes[2].dofs[0]=model.mesh.nodes[1].dofs[0]
    elif bad=="outside":model.mesh.nodes[2].dofs[0]=99
    else:model.mesh.nodes[2].dofs[0]="invalid"
    with pytest.raises(InertiaReliefError):solve_inertia_relief(model)


def test_result_case_records_mass_and_constraint_input_identity():
    model=beam()
    first=solve_inertia_relief(model)
    model.mesh.point_masses[2]=1.
    second=solve_inertia_relief(model)
    assert first.converged and second.converged
    a=first.result_case["metadata"];b=second.result_case["metadata"]
    assert a["point_mass_identity"]!=b["point_mass_identity"]
    assert a["constraint_identity"]==b["constraint_identity"]
    left=model.mesh.nodes[1].dofs[0];right=model.mesh.nodes[2].dofs[0]
    model.add_constraint_equation(terms=((right,1.),(left,-1.)),source_id="identity link")
    third=solve_inertia_relief(model)
    assert third.converged,third.to_dict()
    c=third.result_case["metadata"]
    assert c["constraint_identity"]!=b["constraint_identity"]
    assert c["model_revisions"]==model.revision_signature()


def test_default_linear_s3_reference_profile_and_semidefinite_mass():
    model=FEModel("s3-relief")
    model.add_material("steel",1e7,.3,density=5.)
    for i,xyz in enumerate(((0.,0.,0.),(2.,0.,0.),(0.,1.,0.)),1):model.add_node(i,*xyz)
    element=create_shell_element(1,[1,2,3],"steel",thickness=.02,
                                 reference_normal=np.array([0.,0.,1.]))
    model.add_element(1,element)
    assert not getattr(element,"formulation_native_total_lagrangian",False)
    K,_=assemble_stiffness_matrix(model);M,_=assemble_mass_matrix(model)
    assert np.linalg.matrix_rank(M.toarray())<M.shape[0]
    R=physical_rigid(model,np.array([2./3.,1./3.,0.]))
    a=np.array([.2,-.3,.4,.1,.2,-.1]);u=np.zeros(18)
    for node in model.mesh.nodes.values():u[node.dofs[0]]=1e-4*node.x
    result=solve_inertia_relief(model,vector_load(model,K@u+M@(R@a)))
    assert result.converged,result.to_dict()
    np.testing.assert_allclose(result.angular_acceleration,a[3:],rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(result.displacements[6]-result.displacements[0],2e-4,rtol=1e-8,atol=1e-12)
    assert result.recover().element_stresses


def test_quadratic_beam_uses_its_consistent_mass_profile():
    from anyfesolver import QuadraticBeamElement
    model=beam()
    model.set_node_coordinates(2,1.,0.,0.)
    model.add_node(3,2.,0.,0.)
    section=dict(model.mesh.elements[1].cross_section)
    section["consistent_mass"]=False
    model.mesh.elements[1]=QuadraticBeamElement(1,[1,2,3],"steel",section)
    model.bump_revision("topology")
    load=LoadCase("quadratic free fall",gravity=np.array([.2,-.3,.4]))
    result=solve_inertia_relief(model,load)
    assert result.converged,result.to_dict()
    np.testing.assert_allclose(result.acceleration_at_center_of_mass,load.gravity,rtol=1e-10,atol=1e-12)
    assert result.diagnostics["beam_mass_selection"]["1"]=="consistent"
