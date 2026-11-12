"""Independent finite-work and actual public physical-footprint development checks."""
from dataclasses import replace
import json,os
import numpy as np
import pytest
from scipy import sparse,linalg
from scipy.spatial.transform import Rotation
from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
from anyfesolver._stiffened_shell_b2_reference_buckling import compile_geometric
from examples.stiffened_shell_b2_physical_rigid_footprint import definition,preload,workflow,traction_sources

@pytest.fixture
def owner():
    value=api.create_analysis(definition=definition())
    yield value
    value.close()

def test_public_static_wrench_and_normal_spin(owner):
    load=np.zeros(54);load[13]=1.
    result=owner.solve_linear(load,selected_footprints=api.B2PhysicalRigidFootprintSelection(("physical-left",)))
    assert result.status=="completed",result.failure_reason
    assert result.footprint_law_id==api.B2_PHYSICAL_RIGID_FOOTPRINT_DIRECTOR_REFERENCE_V1_EXPERIMENTAL
    assert len(result.constraint_multipliers)==20 and len(result.constraint_rows)==20
    assert max(abs(np.asarray(result.constraint_gaps)))<=1e-9
    w=result.footprint_wrenches[0]
    assert np.linalg.norm(np.array(w.shell_force_N)+w.beam_force_N)<=1e-8
    assert np.linalg.norm(np.array(w.shell_moment_Nm)+w.beam_moment_Nm)<=1e-8
    assert abs(w.beam_moment_Nm[2])>.1
    assert owner.capabilities.normal_spin.startswith("independent")
    for item in owner._footprint_incidence:
        q=np.r_[np.zeros(3),item.director,np.zeros(6)]
        assert np.max(abs(item.constraint@q))==0.
        assert np.max(abs(item.functional(q[:6],q[6:])))==0.

def test_finite_first_second_work_plateau(owner):
    q=np.array((.3,-.2,.5,.4,-.6,.2,-.1,.7,.2,-.3,.5,.6))
    lam=np.array((2.,-3.,4.,.7,-.9))
    for item in owner._footprint_incidence:
        first=lam@item.constraint@q;second=q@item.multiplier_hessian(lam)@q
        errors=[]
        for h in (1e-3,3e-4,1e-4):
            f=lambda t:lam@item.functional(t*q[:6],t*q[6:])
            errors.append((abs((f(h)-f(-h))/(2*h)-first)/max(1.,abs(first)),abs((f(h)-2*f(0)+f(-h))/h**2-second)/max(1.,abs(second))))
        assert sum(max(e)<=1e-6 for e in errors)>=2,errors

def test_finite_rigid_motion_gradient_acceleration(owner):
    omega=np.array((.4,-.2,.3));r=Rotation.from_rotvec(omega).as_matrix();translation=np.array((.2,-.3,.7));lam=np.array((2.,-3.,4.,.7,-.9))
    for item in owner._footprint_incidence:
        xs=np.array(item.shell_position);xb=np.array(item.beam_position)
        shell=np.r_[r@xs+translation-xs,omega];beam=np.r_[r@xb+translation-xb,omega]
        assert np.max(abs(item.functional(shell,beam)))<=1e-9
        v=np.r_[np.cross(omega,xs)+translation,omega,np.cross(omega,xb)+translation,omega]
        a=np.r_[np.cross(omega,np.cross(omega,xs)),np.zeros(3),np.cross(omega,np.cross(omega,xb)),np.zeros(3)]
        h=item.multiplier_hessian(lam);g=item.constraint.T@lam
        bound=abs(v)@abs(h)@abs(v)+abs(g)@abs(a)
        assert np.isfinite(bound) and bound>0
        assert abs(v@h@v+g@a)<=1e-10*bound

def test_tangent_basis_covariance_and_owned_arrays(owner):
    from anyfesolver._stiffened_shell_b2_physical_rigid_footprint import _node
    group=definition().physical_footprints[0];orth=np.array(((0.,-1.),(1.,0.)))
    changed=replace(group,tangent_basis=group.tangent_basis@orth)
    for old in owner._footprint_incidence:
        new=_node(changed,old,old.row_start);lam=np.array((2.,-3.,4.,.7,-.9));mapped=np.r_[lam[:3],orth.T@lam[3:]]
        np.testing.assert_allclose(new.constraint.T@mapped,old.constraint.T@lam,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(new.multiplier_hessian(mapped),old.multiplier_hessian(lam),rtol=1e-10,atol=1e-12)
        with pytest.raises(ValueError):old.constraint.setflags(write=True)
        with pytest.raises(ValueError):group.reference_director.setflags(write=True)

def test_public_buckling_full_dense_pencil(owner):
    request=api.B2PhysicalFootprintBucklingRequest(preload(),traction_sources());result=owner.buckling(request)
    assert result.status=="ok",result.failure_reason
    assert max(result.equation_residuals)<=1e-8 and max(result.constraint_residuals)<=1e-9
    assert result.policy=="B2_PHYSICAL_RIGID_FOOTPRINT_GREEN_BUCKLING_V1_DEV"
    g,_=compile_geometric(owner,request,result.preload_result);t=owner._T;c=owner._C@t;k=t.T@owner._K@t;b=-(t.T@g@t)
    a=sparse.bmat(((k,c.T),(c,None)),format="csr").toarray();fullb=sparse.block_diag((b,sparse.csr_matrix((c.shape[0],c.shape[0])))).toarray()
    # Independent full singular indefinite pencil, all multiplier coordinates retained.
    norms=np.max(abs(a),axis=1);scale=np.exp2(-np.rint(np.log2(norms)/2));aa=scale[:,None]*a*scale[None,:];bb=scale[:,None]*fullb*scale[None,:]
    beta=np.exp2(np.rint(np.log2(np.max(abs(bb)))));vals=linalg.eigvals(aa,bb/beta)/beta
    positive=sorted(v.real for v in vals if np.isfinite(v) and v.real>0 and abs(v.imag)<=1e-8*abs(v.real))
    np.testing.assert_allclose(result.load_factors[0],positive[0],rtol=1e-8,atol=0.)

def test_explicit_refusals_and_legacy_request(owner):
    assert not owner.reference_buckling_capabilities.eligible
    assert owner.physical_footprint_buckling_capabilities.eligible
    with pytest.raises(ValueError,match="footprint second variation"):
        owner.buckling(api.B2ReferenceBucklingRequest(preload(),traction_sources()))
    with pytest.raises(ValueError,match="mass"):
        owner.modes(api.B2ReferenceModesRequest(count=1))
    with pytest.raises(ValueError):owner.solve_linear(preload(),selected_footprints=api.B2NodalRigidFootprintSelection(("physical-left",)))
    with pytest.raises(ValueError,match="patch_cells"):
        replace(definition().physical_footprints[0],patch_cells=())
    d=definition();bad=replace(d.physical_footprints[0],reference_director=(0.,1.,0.),tangent_basis=((1.,0.),(0.,0.),(0.,1.)))
    with pytest.raises(ValueError):api.create_analysis(definition=replace(d,physical_footprints=(bad,)))

def test_cancelled_buckling(owner):
    token=CancellationToken();token.cancel()
    result=owner.buckling(api.B2PhysicalFootprintBucklingRequest(preload(),traction_sources()),cancellation=token)
    assert result.status=="cancelled" and result.load_factors==()

def test_public_default_workflow():
    result=workflow();assert result["static_status"]=="completed" and result["status"]=="ok",result
    assert result["load_factors"][0]>0 and result["production_qualified"] is False
    if os.environ.get("B2_FOOTPRINT_EXAMPLE_RECORD"):
        from pathlib import Path
        Path(os.environ["B2_FOOTPRINT_EXAMPLE_RECORD"]).write_text(json.dumps(result,indent=2))


def test_native_physical_first_work_and_exp_chart_scalar(owner):
    from anyfesolver._stiffened_shell_b2_physical_rigid_footprint import q4_exp_director_chart_correction
    from anyfesolver.e4_pl_element import _compatible,_coefficients
    shell,element,material_id,slots=owner._buckling_shells[0]
    mesh=owner._buckling_model.mesh;material=owner._buckling_model.get_material(material_id)
    displacement=np.array((.01,.02,.03,.04,-.02,.07,-.03,.02,-.01,.03,.05,-.04,.04,-.01,.02,-.05,.03,.06,-.02,.03,.04,.02,-.04,-.03))
    rec=element._recover_planar_mixed_fields(mesh,displacement,material,element.gauss_points)
    covector=rec["stationary_coupling"]@rec["stationary_parameters"]
    physical=np.zeros(20);coeff=_coefficients(rec["local_nodes"])
    for point,weight,det,stress in zip(element.gauss_points,element.gauss_weights,rec["jacobian_determinants"],rec["resultants"]):
        physical+=_compatible(rec["local_nodes"],coeff,*point).T@stress*weight*det
    expanded=np.zeros(24)
    for i in range(4):expanded[6*i:6*i+5]=physical[5*i:5*i+5]
    assert np.linalg.norm(expanded-covector)/np.linalg.norm(covector)<=1e-10
    h=q4_exp_director_chart_correction(element,mesh,material,displacement)
    q=np.tile((0.,0.,0.,.4,-.7,.6),4);n=rec["frame"][:,2]
    def scalar(t):
        work=0.
        for i in range(4):
            m=rec["frame"]@expanded[6*i+3:6*i+6]
            beta=np.cross(n,Rotation.from_rotvec(t*q[6*i+3:6*i+6]).apply(n)-n)
            work+=m@beta
        return work
    target=q@h@q;errors=[]
    for step in (1e-3,3e-4,1e-4):
        errors.append(abs((scalar(step)+scalar(-step)-2*scalar(0.))/step**2-target)/max(1.,abs(target)))
    assert sum(error<=1e-6 for error in errors)>=2,errors


def test_midassembly_cancellation_no_identity(owner,monkeypatch):
    import anyfesolver._stiffened_shell_b2_reference_buckling as service
    token=CancellationToken();original=service.compile_geometric
    def interrupted(model,request,pre,safe):
        def checkpoint(stage):
            if stage=="buckling physical footprint multiplier curvature":token.cancel("physical pad assembly")
            safe(stage)
        return original(model,request,pre,checkpoint)
    monkeypatch.setattr(service,"compile_geometric",interrupted)
    result=owner.buckling(api.B2PhysicalFootprintBucklingRequest(preload(),traction_sources()),cancellation=token)
    assert result.status=="cancelled" and result.identity=="" and result.load_factors==()


def test_full_shell_frozen_green_scalar_second_variation(owner):
    from anyfesolver._stiffened_shell_b2_physical_rigid_footprint import q4_exp_director_chart_correction
    shell,element,mid,slots=owner._buckling_shells[0];mesh=owner._buckling_model.mesh;material=owner._buckling_model.get_material(mid)
    coords=np.array([definition().shell_nodes_m[i] for i in shell.node_ids])
    u=np.zeros((4,6));u[:,2]=-.5*coords[:,0]*coords[:,1];u[:,3]=-.5*coords[:,0];u[:,4]=.5*coords[:,1]
    rec=element.compute_stresses(mesh,u.ravel(),material)
    assert np.all(rec["transverse_shear_resultants"]==0.),rec["transverse_shear_resultants"]
    n=rec["membrane_resultants"];m=rec["bending_resultants"];thickness=shell.thickness_m
    state=dict(membrane_compression_at_gauss=-n,bending_compression_at_gauss=-m,stress_second_moment_at_gauss=-n*thickness**2/12.)
    base=-element.compute_geometric_stiffness_matrix(mesh,material,state)
    correction=q4_exp_director_chart_correction(element,mesh,material,u.ravel())
    q=np.array((.3,-.2,.5,.4,-.6,.2,-.1,.7,.2,-.3,.5,.6,.2,.1,-.3,.7,.4,-.5,-.4,.6,.3,-.2,.5,.8)).reshape(4,6)
    def scalar(t):
        total=0.;directors=Rotation.from_rotvec(t*q[:,3:]).apply(np.tile((0.,0.,1.),(4,1)))
        for k,((xi,eta),weight) in enumerate(zip(element.gauss_points,element.gauss_weights)):
            shape,dxi,deta=element.compute_shape_functions(xi,eta)
            dx=2*dxi;dy=2*deta
            for z in (-thickness/(2*np.sqrt(3)),thickness/(2*np.sqrt(3))):
                stress=n[k]/thickness+12*z*m[k]/thickness**3
                gx=np.array((1.,0.,0.))+t*dx@q[:,:3]+z*dx@directors
                gy=np.array((0.,1.,0.))+t*dy@q[:,:3]+z*dy@directors
                total+=.5*(stress[0]*(gx@gx-1.)+stress[1]*(gy@gy-1.)+2*stress[2]*(gx@gy))*.25*weight*thickness/2
        return total
    expected=q.ravel()@(base+correction)@q.ravel();errors=[]
    print("SHELL_ORACLE_DIAGNOSTIC="+json.dumps(dict(expected=float(expected),baseline=float(q.ravel()@base@q.ravel()),chart=float(q.ravel()@correction@q.ravel()),scalar_fd=float((scalar(1e-4)+scalar(-1e-4)-2*scalar(0))/1e-8),N=n.tolist(),M=m.tolist(),frame=element._recover_planar_mixed_fields(mesh,u.ravel(),material,element.gauss_points)["frame"].tolist())))
    for step in (1e-3,3e-4,1e-4):
        errors.append(abs((scalar(step)+scalar(-step)-2*scalar(0))/step**2-expected)/max(1.,abs(expected)))
    assert sum(error<=1e-6 for error in errors)>=2,(errors,expected,q.ravel()@base@q.ravel(),q.ravel()@correction@q.ravel(),m.tolist(),rec["physical_director"].tolist(),element._recover_planar_mixed_fields(mesh,u.ravel(),material,element.gauss_points)["frame"].tolist())
    assert np.max(abs(correction))>1e-4


def test_rigid_geometry_covariance_director_reversal(owner):
    from anyfesolver._stiffened_shell_b2_physical_rigid_footprint import _node
    group=definition().physical_footprints[0];r=Rotation.from_rotvec((.3,-.2,.4)).as_matrix();lam=np.array((2.,-3.,4.,.7,-.9));block=sparse.block_diag((r,r,r,r)).toarray()
    for old in owner._footprint_incidence:
        changed=replace(group,reference_director=r@group.reference_director,tangent_basis=r@group.tangent_basis)
        item=replace(old,shell_position=tuple(r@old.shell_position),beam_position=tuple(r@old.beam_position))
        new=_node(changed,item,item.row_start);mapped=np.r_[r@lam[:3],lam[3:]]
        np.testing.assert_allclose(new.constraint@block,sparse.block_diag((r,np.eye(2))).toarray()@old.constraint,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(block.T@new.multiplier_hessian(mapped)@block,old.multiplier_hessian(lam),rtol=1e-10,atol=1e-12)
        reverse=_node(replace(group,reference_director=-group.reference_director),old,old.row_start);mapped=np.r_[lam[:3],-lam[3:]]
        np.testing.assert_allclose(reverse.constraint.T@mapped,old.constraint.T@lam,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(reverse.multiplier_hessian(mapped),old.multiplier_hessian(lam),rtol=1e-10,atol=1e-12)


def test_nonzero_director_multiplier_combined_shell_pad_chart(owner):
    from anyfesolver._stiffened_shell_b2_physical_rigid_footprint import q4_exp_director_chart_correction
    shell,element,mid,slots=owner._buckling_shells[0];mesh=owner._buckling_model.mesh;material=owner._buckling_model.get_material(mid)
    coords=np.array([definition().shell_nodes_m[i] for i in shell.node_ids])
    u=np.zeros((4,6));u[:,2]=-.5*coords[:,0]*coords[:,1];u[:,3]=-.5*coords[:,0];u[:,4]=.5*coords[:,1]
    recovered=element.compute_stresses(mesh,u.ravel(),material)
    assert np.all(recovered["transverse_shear_resultants"]==0.)
    native=element._recover_planar_mixed_fields(mesh,u.ravel(),material,element.gauss_points)
    f=native["stationary_coupling"]@native["stationary_parameters"]
    correction=q4_exp_director_chart_correction(element,mesh,material,u.ravel())
    pad=np.zeros((24,24));nonzero=[]
    for node in range(4):
        item=next(v for v in owner._footprint_incidence if v.shell_node==shell.node_ids[node])
        m=native["frame"]@f[6*node+3:6*node+6]
        # Static virtual-work balance: native m + C_director.T lambda = 0.
        director_rows=item.constraint[3:,3:6]
        lam=np.linalg.lstsq(director_rows.T,-m,rcond=None)[0]
        assert np.linalg.norm(m+director_rows.T@lam)<=1e-10*np.linalg.norm(m)
        nonzero.extend(lam)
        pad[6*node:6*node+6,6*node:6*node+6]=item.multiplier_hessian(np.r_[np.zeros(3),lam])[:6,:6]
    assert max(abs(np.array(nonzero)))>1e-4
    spin=np.zeros(24);spin[5::6]=(.4,-.3,.7,-.5)
    tilt=np.zeros(24);tilt[3::6]=(.3,.5,-.2,.7);tilt[4::6]=(-.4,.2,.6,-.3)
    bound=abs(spin)@(abs(correction)+abs(pad))@abs(tilt)
    assert bound>0 and abs(spin@(correction+pad)@tilt)<=1e-10*bound
    assert abs(spin@correction@tilt)>1e-4
    # Omission or duplication of the chart term leaves finite mixed spin work.
    assert abs(spin@pad@tilt)>1e-4 and abs(spin@(2*correction+pad)@tilt)>1e-4


def test_new_family_point_coexistence_and_selection_refuse():
    d=definition()
    from anyfesolver._stiffened_shell_b2_q4_panel_experimental import ExperimentalB2Q4PanelAnalysis
    with pytest.raises(ValueError,match="separate physical footprint static family required"):
        ExperimentalB2Q4PanelAnalysis(d.shell_nodes_m,d.shells,d.members,(api.Q4B2Point("p","shell-1","beam-right",0.),),transformation=d.transformation,prescribed_offset=d.prescribed_offset,physical_footprints=d.physical_footprints)
    with pytest.raises(ValueError,match="sequence"):
        api.B2PhysicalRigidFootprintSelection("x")


@pytest.mark.parametrize("reverse",(False,True))
def test_distorted_skew_frozen_shell_tensor_frame_and_scalar_work(owner,reverse):
    from anyfesolver._stiffened_shell_b2_physical_rigid_footprint import q4_kernel_frame_resultants,_exp_director_covector_hessian
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    from anyfesolver.fe_core import FEModel
    r=Rotation.from_rotvec((.3,-.2,.4)).as_matrix()
    plane=np.array(((0.,0.,0.),(1.3,.2,0.),(1.7,1.1,0.),(.1,.8,0.)))
    coords=plane@r.T;n=r[:,2]*(-1. if reverse else 1.)
    model=FEModel("independent-distorted-frame-work")
    for i,x in enumerate(coords,1):model.add_node(i,*x)
    model.add_material("m",1000.,.3)
    element=QualifiedE4PLShellElement(1,(1,2,3,4),"m",thickness=.15,reference_normal=n)
    material=model.get_material("m");mesh=model.mesh
    native=element._recover_planar_mixed_fields(mesh,np.zeros(24),material,element.gauss_points)
    physical=element._physical_director_context(native["frame"])[0]
    stress=np.array(((2.,-1.,.7,.3,-.2,.4,0.,0.),(1.5,-.6,.4,.2,-.3,.5,0.,0.),(2.3,-.8,.2,.4,-.1,.3,0.,0.),(1.8,-.4,.6,.5,-.4,.2,0.,0.)))
    mapped=q4_kernel_frame_resultants(element,mesh,material,np.zeros(24),stress)
    h=.15
    state=dict(membrane_compression_at_gauss=-mapped[:,:3],bending_compression_at_gauss=-mapped[:,3:],stress_second_moment_at_gauss=-mapped[:,:3]*h*h/12.)
    base=-element.compute_geometric_stiffness_matrix(mesh,material,state)
    q=np.array((.3,-.2,.5,.4,-.6,.2,-.1,.7,.2,-.3,.5,.6,.2,.1,-.3,.7,.4,-.5,-.4,.6,.3,-.2,.5,.8)).reshape(4,6)@sparse.block_diag((r,r)).toarray().T
    data=[];moments=np.zeros((4,3))
    local=coords@physical[:,:2]
    for k,((xi,eta),weight) in enumerate(zip(element.gauss_points,element.gauss_weights)):
        shape,dxi,deta=element.compute_shape_functions(xi,eta)
        jac=np.array(((dxi@local[:,0],dxi@local[:,1]),(deta@local[:,0],deta@local[:,1])))
        gradient=np.linalg.solve(jac,np.vstack((dxi,deta)));det=abs(np.linalg.det(jac))
        nt=np.array(((stress[k,0],stress[k,2]),(stress[k,2],stress[k,1])))
        mt=np.array(((stress[k,3],stress[k,5]),(stress[k,5],stress[k,4])))
        for i in range(4):moments[i]+=np.cross(n,physical[:,:2]@mt@gradient[:,i])*det*weight
        data.append((gradient,det*weight,nt,mt))
    correction=_exp_director_covector_hessian(n,moments)
    def scalar(t):
        directors=Rotation.from_rotvec(t*q[:,3:]).apply(np.tile(n,(4,1)));total=0.
        for grad,area,nt,mt in data:
            for z in (-h/(2*np.sqrt(3)),h/(2*np.sqrt(3))):
                # Physical material-coordinate metric, not the kernel GP basis.
                tangent=physical[:,:2]+t*q[:,:3].T@grad.T+z*directors.T@grad.T
                metric=tangent.T@tangent-np.eye(2)
                total+=.5*np.sum((nt/h+12*z*mt/h**3)*metric)*area*h/2
        return total
    target=q.ravel()@(base+correction)@q.ravel();errors=[]
    for step in (1e-3,3e-4,1e-4):errors.append(abs((scalar(step)+scalar(-step)-2*scalar(0))/step**2-target)/max(1.,abs(target)))
    assert sum(e<=1e-6 for e in errors)>=2,errors


def test_point_square_reference_factor_preserved():
    from examples.stiffened_shell_b2_reference_buckling import workflow as point_workflow
    result=point_workflow()
    assert result["status"]=="ok",result
    np.testing.assert_allclose(result["load_factors"],(24.60872920125391,),rtol=1e-10,atol=0.)
