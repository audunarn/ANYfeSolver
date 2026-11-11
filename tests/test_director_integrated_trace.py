"""Independent invariants for the opt-in director-integrated edge trace."""
import copy
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from anyfesolver import FEModel, InterpolatedBeamShellMPCElement
from anyfesolver._director_trace import director_trace_state
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.corotational import (build_nonlinear_rotating_mpc_map,rotation_matrix_from_vector,
    rotation_vector_from_matrix,spatial_moment_from_rotation_covector,SO3MeanDomainError)
from anyfesolver.nonlinear_restart import model_configuration_fingerprint
from anyfesolver.elements import BeamElement


def fixture(offset=(0.,0.,.03),xi=.37):
    X=np.array([[.2,-.1,.3],[1.1,.15,.1]])
    e=np.array(offset); slave=(1-xi)*X[0]+xi*X[1]+e
    return X,slave,e,xi


def model_map(offset=(0.,0.,.03),xi=.37,policy="director_integrated"):
    X,slave,e,xi=fixture(offset,xi); model=FEModel("trace")
    for n,p in ((1,X[0]),(2,X[1]),(3,slave)): model.add_node(n,*p)
    el=InterpolatedBeamShellMPCElement(1,3,[1,2],np.array([1-xi,xi]),e,connection_trace=policy)
    model.add_element(1,el); model.apply_boundary_conditions()
    _,_,T,*_=build_constraint_transformation(sparse.eye(18,format="csr"),np.zeros(18),model)
    return model,el,T,build_nonlinear_rotating_mpc_map(model,T)


def test_reference_linearization_has_compatible_bending_and_shear():
    X=np.array([[0.,0.,0.],[1.,0.,0.]]); xi=.37; slave=np.array([xi,0.,0.])
    y,B,H=director_trace_state(X,slave,np.zeros(3),xi,np.zeros(12))
    np.testing.assert_allclose(y,0.,atol=1e-15)
    state=np.zeros(12); k=.2
    state[8]=-.5*k; state[10]=k
    # uz=-k*x^2/2, ry=k*x is the exact zero-shear bending patch.
    assert abs((B@state)[2]+.5*k*xi**2)<1e-14
    # Adding constant physical shear and axial strain retains affine motion.
    state[6]=.03; state[8]+=.07
    assert abs((B@state)[0]-.03*xi)<1e-14
    assert abs((B@state)[2]-(-.5*k*xi**2+.07*xi))<1e-14
    # Legacy translational interpolation has a nonzero bending trace error.
    assert abs(xi*(-.5*k)+.5*k*xi**2)>.01


def test_actual_beam2_patch_energy_removes_only_extra_mean_shear():
    model=FEModel("energy"); material=model.add_material("elastic",210e9,.3)
    xs=np.linspace(0.,1.,4); k=.2; X=np.array([[0.,0.,0.],[1.,0.,0.]])
    old=np.zeros((4,6)); new=np.zeros((4,6)); endpoint=np.zeros(12)
    endpoint[8]=-.5*k; endpoint[10]=k
    for i,x in enumerate(xs):
        model.add_node(i+1,x,0.,0.)
        old[i,2]=-.5*k*x; old[i,4]=k*x
        _,B,_=director_trace_state(X,np.array([x,0.,0.]),np.zeros(3),x,np.zeros(12))
        new[i]=B@endpoint
    energies=[]
    for values in (old,new):
        energy=0.
        for i in range(3):
            element=BeamElement(i+1,[i+1,i+2],"elastic",dict(area=.001,Iy=1e-6,Iz=1e-6,J=1e-6))
            stiffness=element.compute_stiffness_matrix(model.mesh,model.get_material("elastic"))
            ue=values[i:i+2].reshape(-1); energy+=.5*ue@stiffness@ue
        energies.append(energy)
    bending=.5*210e9*1e-6*k*k
    assert energies[0]>bending*1.01
    assert abs(energies[1]-bending)/bending<1e-12


@pytest.mark.parametrize("angle", (.000999,.001001,.3,np.pi-2e-6))
def test_equal_finite_rotations_and_chart_boundary_derivatives(angle):
    args=fixture(); x=np.zeros(12); axis=np.array([.3,-.4,.5]); axis/=np.linalg.norm(axis)
    x[3:6]=x[9:12]=angle*axis
    y,B,H=director_trace_state(*args,x)
    d=np.linspace(-.2,.3,12); h=2e-7 if angle>3. else 2e-4
    yp,Bp,_=director_trace_state(*args,x+h*d); ym,Bm,_=director_trace_state(*args,x-h*d)
    ypp,Bpp,_=director_trace_state(*args,x+2*h*d); ymm,Bmm,_=director_trace_state(*args,x-2*h*d)
    np.testing.assert_allclose((-ypp+8*yp-8*ym+ymm)/(12*h),B@d,rtol=1e-7,atol=2e-9)
    np.testing.assert_allclose((-Bpp+8*Bp-8*Bm+Bmm)/(12*h),np.einsum("ijk,k->ij",H,d),rtol=1e-7,atol=2e-9)


def test_order_reversal_is_the_same_material_trace():
    X,slave,e,xi=fixture(); x=np.linspace(-.15,.21,12)
    a=director_trace_state(X,slave,e,xi,x)
    perm=np.r_[np.arange(6,12),np.arange(6)]
    b=director_trace_state(X[::-1],slave,e,1.-xi,x[perm])
    np.testing.assert_allclose(a[0],b[0],atol=3e-15)
    np.testing.assert_allclose(a[1],b[1][:,perm],atol=3e-14)
    np.testing.assert_allclose(a[2],b[2][:,perm][:,:,perm],atol=3e-13)


def test_relative_log_near_pi_derivatives_and_compiled_admission():
    args=fixture(); x=np.zeros(12); axis=np.array([.3,-.4,.5]); axis/=np.linalg.norm(axis)
    x[3:6]=-.5*(np.pi-2e-6)*axis; x[9:12]=-x[3:6]
    y,B,H=director_trace_state(*args,x); d=np.linspace(-.1,.15,12)
    # Tangential rotation perturbations have zero first-order change in the
    # relative angle here, resolving the derivative without crossing its cut.
    for idx in (slice(3,6),slice(9,12)): d[idx]-=axis*np.dot(axis,d[idx])
    h=2e-4
    yp,Bp,_=director_trace_state(*args,x+h*d); ym,Bm,_=director_trace_state(*args,x-h*d)
    ypp,Bpp,_=director_trace_state(*args,x+2*h*d); ymm,Bmm,_=director_trace_state(*args,x-2*h*d)
    np.testing.assert_allclose((-ypp+8*yp-8*ym+ymm)/(12*h),B@d,rtol=1e-7,atol=2e-9)
    np.testing.assert_allclose((-Bpp+8*Bp-8*Bm+Bmm)/(12*h),np.einsum("ijk,k->ij",H,d),rtol=1e-7,atol=2e-9)
    model,el,T,m=model_map(); linear=np.asarray(T@x).reshape(-1)
    mapped=m.apply_kinematics(linear)
    np.testing.assert_allclose(mapped[12:18],y,atol=2e-14)


@pytest.mark.parametrize("angle", (0.,.6,-.8))
@pytest.mark.parametrize("xi", (0.,.37,1.))
def test_exact_circular_trace_and_endpoints(angle,xi):
    X=np.array([[0.,0.,0.],[1.,0.,0.]]); e=np.array([0.,0.,.03]); slave=np.array([xi,0.,.03])
    state=np.zeros(12); state[10]=angle
    if angle:
        state[6:9]=[np.sin(angle)/angle-1.,0.,(np.cos(angle)-1.)/angle]
        centre=np.array([np.sin(xi*angle)/angle,0.,(np.cos(xi*angle)-1.)/angle])
    else: centre=np.array([xi,0.,0.])
    y,_,_=director_trace_state(X,slave,e,xi,state)
    R=rotation_matrix_from_vector(np.array([0.,xi*angle,0.]))
    np.testing.assert_allclose(y[:3]+slave,centre+R@e,atol=3e-15)
    np.testing.assert_allclose(y[3:],[0.,xi*angle,0.],atol=3e-15)


def test_mixed_axis_jacobian_hessian_and_se3_objectivity():
    args=fixture(); x=np.array([.03,-.02,.01,.17,-.12,.09,-.01,.04,.02,-.13,.21,.08])
    y,B,H=director_trace_state(*args,x); step=2e-4
    for j in range(12):
        d=np.eye(12)[j]*step
        yp,Bp,_=director_trace_state(*args,x+d); ym,Bm,_=director_trace_state(*args,x-d)
        ypp,Bpp,_=director_trace_state(*args,x+2*d); ymm,Bmm,_=director_trace_state(*args,x-2*d)
        # Fourth-order symmetric stencil avoids roundoff in small Hessian
        # entries without changing the independent derivative bounds.
        np.testing.assert_allclose((-ypp+8*yp-8*ym+ymm)/(12*step),B[:,j],rtol=1e-7,atol=2e-10)
        np.testing.assert_allclose((-Bpp+8*Bp-8*Bm+Bmm)/(12*step),H[:,:,j],rtol=1e-7,atol=2e-10)
    assert np.linalg.norm(H[:3,:3,9:12])>0.
    np.testing.assert_allclose(H,H.transpose(0,2,1),atol=1e-14)
    Q=rotation_matrix_from_vector(np.array([.2,-.1,.15])); c=np.array([.4,-.3,.2]); xx=x.copy()
    for i in range(2):
        xx[6*i:6*i+3]=Q@(args[0][i]+x[6*i:6*i+3])+c-args[0][i]
        xx[6*i+3:6*i+6]=rotation_vector_from_matrix(Q@rotation_matrix_from_vector(x[6*i+3:6*i+6]))
    yy,_,_=director_trace_state(*args,xx)
    np.testing.assert_allclose(yy[:3]+args[1],Q@(y[:3]+args[1])+c,atol=2e-14)
    np.testing.assert_allclose(rotation_matrix_from_vector(yy[3:]),Q@rotation_matrix_from_vector(y[3:]),atol=2e-14)


def test_compiled_map_inverse_work_full_hessian_and_wrench():
    model,el,T,m=model_map(offset=(0.,0.,0.)); assert m is not None
    q=np.linspace(-.04,.06,T.shape[1]); linear=np.asarray(T@q).reshape(-1); u=m.apply_kinematics(linear)
    np.testing.assert_allclose(m.to_linear_displacement(u),linear,atol=1e-14)
    f=np.linspace(-2.,3.,18); J=m.jacobian(u); direction=np.linspace(.3,-.2,len(q)); h=2e-6
    up=m.apply_kinematics(np.asarray(T@(q+h*direction)).reshape(-1)); um=m.apply_kinematics(np.asarray(T@(q-h*direction)).reshape(-1))
    np.testing.assert_allclose((up-um)/(2*h),J@direction,rtol=1e-7,atol=1e-10)
    np.testing.assert_allclose(m.reduce_force(u,f),J.T@f,atol=1e-14)
    Hg=m.geometric_tangent(u,f)
    np.testing.assert_allclose((m.reduce_force(up,f)-m.reduce_force(um,f))/(2*h),Hg@direction,rtol=1e-7,atol=2e-9)
    transferred=m.transfer_slave_imbalance_to_masters(u,f)
    def wrench(force):
        total=np.zeros(6)
        for n in model.mesh.nodes.values():
            d=n.dofs; total[:3]+=force[d[:3]]
            total[3:]+=np.cross(n.coords()+u[d[:3]],force[d[:3]])+spatial_moment_from_rotation_covector(u[d[3:]],force[d[3:]])
        return total
    np.testing.assert_allclose(wrench(f),wrench(transferred),rtol=1e-12,atol=2e-13)
    # Static/reference MPC compiler differentiates this same new map.
    zero=np.zeros(18); np.testing.assert_allclose(m.jacobian(zero).toarray(),T.toarray(),atol=2e-14)


def test_policy_admission_identity_and_no_input_mutation():
    model,el,T,m=model_map(); original=copy.deepcopy(vars(el)); before=model_configuration_fingerprint(model)
    el.connection_trace="linear"; assert model_configuration_fingerprint(model)!=before
    el.connection_trace="invalid"
    with pytest.raises(ValueError,match="unsupported connection_trace"): el.get_mpc_constraints(model.mesh)
    el.connection_trace="director_integrated"
    el.shell_node_ids.reverse(); el.shape_weights=el.shape_weights[::-1].copy()
    assert model_configuration_fingerprint(model)!=before
    el.shell_node_ids.reverse(); el.shape_weights=el.shape_weights[::-1].copy()
    node=model.mesh.get_node(2); node.x+=.001
    assert model_configuration_fingerprint(model)!=before
    node.x-=.001
    el.shape_weights[:]=[.7,.3]; assert model_configuration_fingerprint(model)!=before
    with pytest.raises(ValueError,match="reference position"): el.get_mpc_constraints(model.mesh)
    el.shape_weights[:]=original["shape_weights"]
    u=np.zeros(18); u[3:6]=[np.pi,0.,0.]; saved=u.copy()
    with pytest.raises(SO3MeanDomainError): m.apply_kinematics(u)
    np.testing.assert_array_equal(u,saved)
    with pytest.raises(ValueError,match="requires rotating_eccentricity"):
        InterpolatedBeamShellMPCElement(1,3,[1,2],np.array([.5,.5]),np.zeros(3),rotating_eccentricity=False,connection_trace="director_integrated")


@pytest.mark.parametrize("fault", ("negative","three","length","geometry"))
def test_unsupported_trace_geometry_is_refused_without_mutation(fault):
    model,el,T,m=model_map()
    if fault=="negative": el.shape_weights[:]=[-.1,1.1]
    elif fault=="three": el.shell_node_ids=[1,2,3]; el.shape_weights=np.array([.2,.3,.5])
    elif fault=="length":
        a,b=model.mesh.get_node(1),model.mesh.get_node(2); b.x,b.y,b.z=a.x,a.y,a.z
        slave=model.mesh.get_node(3); slave.x,slave.y,slave.z=a.coords()+el.eccentricity
    else: model.mesh.get_node(3).z+=.01
    before=copy.deepcopy(vars(el)); nodes=np.array([n.coords() for n in model.mesh.nodes.values()])
    with pytest.raises(ValueError): el.get_mpc_constraints(model.mesh)
    np.testing.assert_array_equal(el.shape_weights,before["shape_weights"])
    np.testing.assert_array_equal([n.coords() for n in model.mesh.nodes.values()],nodes)


def elastic_trace_model(policy="director_integrated"):
    from anyfesolver import BoundaryCondition,LoadCase
    model=FEModel("elastic_trace"); model.add_material("elastic",210e9,.3)
    for n,xyz in ((1,(0.,0.,0.)),(2,(1.,0.,0.)),(3,(0.,0.,.03)),(4,(.5,0.,.03)),(5,(1.,0.,.03))): model.add_node(n,*xyz)
    for i,(slave,masters,weights) in enumerate(((3,[1],[1.]),(4,[1,2],[.5,.5]),(5,[2],[1.]))):
        model.add_element(10+i,InterpolatedBeamShellMPCElement(10+i,slave,masters,np.array(weights),np.array([0.,0.,.03]),"elastic",connection_trace=policy))
    for i,pair in enumerate(([3,4],[4,5])):
        model.add_element(i+1,BeamElement(i+1,pair,"elastic",dict(area=.001,Iy=1e-6,Iz=1e-6,J=1e-6)))
    model.add_boundary_condition(BoundaryCondition("root",[1],dict(ux=0.,uy=0.,uz=0.,rx=0.,ry=0.,rz=0.)))
    model.add_boundary_condition(BoundaryCondition("tip",[2],dict(uy=0.,uz=-.1,rx=0.,ry=.2,rz=0.)))
    model.apply_boundary_conditions(); load=LoadCase("probe"); load.add_nodal_load(5,forces=(1.,0.,-1.)); model.add_load_case(load)
    return model,load


@pytest.mark.parametrize("fault", ("input","relative","output"))
def test_all_new_trace_chart_bounds_refused_without_mutation(fault):
    model,el,T,m=model_map(xi=.5); x=np.zeros(12)
    if fault=="input": x[3]=np.pi
    elif fault=="relative": x[3]=-(np.pi-.5e-6)/2; x[9]=-x[3]
    else: x[3]=np.pi-.1; x[9]=-x[3]
    linear=np.asarray(T@x).reshape(-1); saved=linear.copy()
    with pytest.raises(SO3MeanDomainError): m.apply_kinematics(linear)
    np.testing.assert_array_equal(linear,saved)


def test_baseline_style_linear_checkpoint_and_explicit_linear_identity():
    from anyfesolver import solve_static_nonlinear
    baseline,load=elastic_trace_model("linear")
    for element in baseline.mesh.elements.values():
        if isinstance(element,InterpolatedBeamShellMPCElement):
            # Baseline V1 instances had exactly this public namespace, with
            # no trace-policy instance field. The class default is not hashed.
            assert "connection_trace" not in vars(element)
            assert element.connection_trace=="linear"
    common=dict(kinematics="corotational",num_steps=2,max_iterations=20,tolerance=1e-8)
    saved=solve_static_nonlinear(baseline,load,emit_restart_checkpoint=True,**common)
    assert saved.status=="completed"
    current,current_load=elastic_trace_model("linear")
    for eid,element in tuple(current.mesh.elements.items()):
        if isinstance(element,InterpolatedBeamShellMPCElement):
            current.mesh.elements[eid]=InterpolatedBeamShellMPCElement(
                element.element_id,element.beam_node_id,list(element.shell_node_ids),
                element.shape_weights.copy(),element.eccentricity.copy(),element.material_name)
            assert "connection_trace" not in vars(current.mesh.elements[eid])
    assert model_configuration_fingerprint(current)==model_configuration_fingerprint(baseline)
    resumed=solve_static_nonlinear(current,current_load,restart_checkpoint=saved.restart_checkpoint,**common)
    assert resumed.status=="completed"
    np.testing.assert_allclose(resumed.displacements,saved.displacements,rtol=1e-9,atol=1e-12)


def test_documented_experimental_constructor_executes():
    document=Path(__file__).resolve().parents[1]/"docs"/"BEAM_SHELL_CONNECTION.md"
    code=document.read_text(encoding="utf-8").split("```python\n")[-1].split("```",1)[0]
    scope={}; exec(compile(code,str(document),"exec"),scope)
    model=scope["trace_model"]; element=model.mesh.elements[10]
    assert element.connection_trace=="director_integrated"
    assert element.get_nonlinear_mpc_descriptor(model.mesh) is not None


@pytest.mark.parametrize("origin", (0.,.75,1e6))
def test_trace_retains_displacements_below_world_coordinate_ulp(origin):
    X=np.array([[origin,0.,0.],[origin+.0625,0.,0.]])
    slave=np.array([origin+.03125,0.,.002]); offset=np.array([0.,0.,.002])
    zero=np.zeros(12); _,B,_=director_trace_state(X,slave,offset,.5,zero)
    x=zero.copy(); x[6]=2.**-56; x[10]=2.**-56; before=x.copy()
    actual,_,_=director_trace_state(X,slave,offset,.5,x)
    expected=B@x  # Higher-order terms are O(2**-112), below this bound.
    np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-34)
    assert actual[0]>0. and actual[2]>0.
    np.testing.assert_array_equal(x,before)


def test_trace_preserves_exact_nonzero_reference_closure():
    X=np.array([[1.,0.,0.],[2.,0.,0.]])
    # Exact real arithmetic on these binary floats gives -2**-54, even
    # though the rounded absolute-position expression returns zero.
    slave=np.array([1.3,0.,0.]); before=X.copy()
    value,_,_=director_trace_state(X,slave,np.zeros(3),.3,np.zeros(12))
    assert value[0]==-2.**-54
    np.testing.assert_array_equal(value[1:],np.zeros(5))
    np.testing.assert_array_equal(X,before)


def test_public_solve_copied_restart_and_cancel_integrity():
    from anyfesolver import solve_static_nonlinear
    from anyfesolver.control import CancellationToken,SolveCancelled
    model,load=elastic_trace_model()
    common=dict(kinematics="corotational",num_steps=2,max_iterations=20,tolerance=1e-8,emit_restart_checkpoint=False)
    first=solve_static_nonlinear(model,load,**common)
    assert first.status=="completed"
    u=first.displacements.copy(); H=copy.deepcopy(dict(first.element_states)); u0=u.copy(); H0=copy.deepcopy(H)
    second=solve_static_nonlinear(model,load,initial_displacements=u,initial_element_states=H,equilibrate_initial_state=False,**common)
    assert second.status=="completed"
    np.testing.assert_allclose(second.displacements,first.displacements,rtol=1e-9,atol=1e-12)
    token=CancellationToken(); token.cancel("trace test")
    with pytest.raises(SolveCancelled):
        solve_static_nonlinear(model,load,initial_displacements=u,initial_element_states=H,cancellation_token=token,**common)
    np.testing.assert_array_equal(u,u0)
    for eid,state in H.items():
        for key,value in state.items():
            if isinstance(value,np.ndarray): np.testing.assert_array_equal(value,H0[eid][key])
    # A state from this trace must not be silently consumed under the legacy
    # interior translation law; ordinary copied restart revalidates its map.
    model.mesh.elements[11].connection_trace="linear"
    with pytest.raises(ValueError):
        solve_static_nonlinear(model,load,initial_displacements=u,initial_element_states=H,equilibrate_initial_state=False,**common)
    model.mesh.elements[11].connection_trace="director_integrated"
    with pytest.raises(ValueError,match="require corotational"):
        solve_static_nonlinear(model,load,kinematics="von_karman")
    with pytest.raises(ValueError,match="rotating-MPC"):
        solve_static_nonlinear(model,load,kinematics="corotational",corotational_tangent="rotated")
    from anyfesolver.nonlinear_restart import NonlinearCheckpointError
    checkpointed=solve_static_nonlinear(model,load,**dict(common,emit_restart_checkpoint=True))
    assert checkpointed.status=="completed" and checkpointed.restart_checkpoint is not None
    resumed=solve_static_nonlinear(model,load,restart_checkpoint=checkpointed.restart_checkpoint,**common)
    assert resumed.status=="completed"
    np.testing.assert_allclose(resumed.displacements,checkpointed.displacements,rtol=1e-9,atol=1e-12)
    model.mesh.elements[11].connection_trace="linear"
    with pytest.raises(NonlinearCheckpointError,match="fingerprint"):
        solve_static_nonlinear(model,load,restart_checkpoint=checkpointed.restart_checkpoint,**common)
