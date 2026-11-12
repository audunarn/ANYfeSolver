"""Development contracts for the whole-edge shear investigation (not buckling qualification)."""
import importlib.util,json,pathlib
import numpy as np
import pytest
from anyfesolver import solve_static_nonlinear
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from anyfesolver.corotational import _corotational_cache,_corotational_deformation_state,spatial_force_and_moment_resultants

spec=importlib.util.spec_from_file_location('plate_fixture',pathlib.Path(__file__).parents[1]/'scripts/investigate_plate_edge_shear.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

def assemble(m,u,tangent):
    return _assemble_nonlinear_system(m,u,{},5,tangent=tangent,require_full_coordinates=True,kinematics='corotational',corotational_tangent='work_conjugate')

def energy(m,u):
    # Virgin ideal-elastic VK force is cubic in its local displacement ray;
    # 3-point Gauss integrates this degree3 integrand exactly (up to roundoff).
    points,weights=np.polynomial.legendre.leggauss(3);total=0.
    refs=_corotational_cache(m)
    for eid,e in m.mesh.elements.items():
        dofs=e.get_dof_mapping(m.mesh);ud,_,_=_corotational_deformation_state(refs[eid],e,u[dofs])
        for x,w in zip(points,weights):
            f,_,_=e.compute_nonlinear_response(m.mesh,m.get_material(e.material_name),.5*(x+1)*ud,None,5,False)
            total+=.5*w*float(ud@f)
    return total

def test_assembled_work_conjugate_energy_tangent_and_spatial_equilibrium():
    m=fixture.plate(.01,True);rng=np.random.default_rng(20261022)
    u=.003*rng.standard_normal(m.mesh.dof_manager.total_dofs)
    p,k,_=assemble(m,u,True);k=k.toarray()
    symmetry=np.linalg.norm(k-k.T)/np.linalg.norm(k);assert symmetry<1e-12
    xyz=np.array([[n.x,n.y,n.z] for n in m.mesh.nodes.values()])+u.reshape(-1,6)[:,:3]
    force,moment=spatial_force_and_moment_resultants(xyz,u.reshape(-1,6)[:,3:],p)
    scale=max(np.linalg.norm(p),1.)
    assert np.linalg.norm(force)/scale<1e-12
    assert np.linalg.norm(moment)/scale<1e-11
    mixed=rng.standard_normal(len(u));mixed/=np.linalg.norm(mixed)
    edge=np.zeros_like(u)
    edge[[m.mesh.nodes[n].dofs[1] for n in [3,6,9]]]=1
    edge/=np.linalg.norm(edge)
    metrics=[]
    for d in [mixed,edge]:
        h=1e-5;plus=assemble(m,u+h*d,False)[0];minus=assemble(m,u-h*d,False)[0]
        fd=(plus-minus)/(2*h);error=np.linalg.norm(k@d-fd)/np.linalg.norm(fd)
        assert error<1e-7
        h=1e-6;du=(energy(m,u+h*d)-energy(m,u-h*d))/(2*h)
        energy_error=abs(du-p@d)/max(abs(p@d),1.)
        assert energy_error<1e-6
        metrics.append({'tangent_fd_relative':error,'energy_gradient_relative':energy_error})
    print(json.dumps({'symmetry':symmetry,'force_relative':np.linalg.norm(force)/scale,'moment_relative':np.linalg.norm(moment)/scale,'directions':metrics}))

@pytest.fixture(scope='module')
def solved():
    m=fixture.plate(.01,True)
    r=solve_static_nonlinear(m,None,num_steps=4,max_iterations=25,kinematics='corotational',corotational_tangent='work_conjugate',record_increment_snapshots=True)
    return m,r

def test_whole_driven_edge_clamp_and_imperfect_branch(solved):
    m,r=solved;assert r.converged and r.load_factor==1.
    driven=[m.mesh.nodes[n].dofs[1] for n in [3,6,9]]
    fixed=[d for n in [1,4,7] for d in m.mesh.nodes[n].dofs]
    for s in r.snapshots:
        np.testing.assert_allclose(s.displacements[driven],s.load_factor*.01,atol=1e-12,rtol=0)
        np.testing.assert_allclose(s.displacements[fixed],0,atol=1e-12,rtol=0)
    assert m.mesh.nodes[5].z==.001
    assert np.max(abs(r.displacements[2::6]))>1e-4
    p,_,_=assemble(m,r.displacements,False)
    constrained=fixed+driven;free=np.setdiff1d(np.arange(len(p)),constrained)
    assert np.linalg.norm(p[free])/max(np.linalg.norm(p[constrained]),1.)<=1e-6

def test_adaptive_cutback_keeps_only_accepted_edge_states(solved):
    m,r=solved
    failed=[v for v in r.info['convergence_adaptation'] if v['action']=='cutback_after_nonconvergence']
    assert failed and r.info['total_newton_iterations']>sum(s.iterations for s in r.steps)
    assert len(r.snapshots)==len(r.steps)
    factors=[s.load_factor for s in r.snapshots]
    assert all(b>a for a,b in zip(factors,factors[1:]))
    assert factors[-1]==1.
