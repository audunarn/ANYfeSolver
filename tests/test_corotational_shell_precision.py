"""Small-shell precision and public convergence regressions."""
import importlib.util,pathlib,decimal
import numpy as np
import pytest
from anyfesolver import solve_static_nonlinear,solve_linear
from anyfesolver import corotational as cr
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
spec=importlib.util.spec_from_file_location('abuse_fixture',pathlib.Path(__file__).parents[1]/'scripts/explore_nonlinear_abuse.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

@pytest.mark.parametrize('inputs',[{'E':1e12},{'thickness':.1},{'E':1e12,'origin':1e6},{'E':1e12,'scale':.01}])
def test_small_stiff_plate_converges_without_failed_increments(inputs):
    m,_=fixture.plate_model(dict(name='precision',**inputs));linear,_=solve_linear(m,None)
    r=solve_static_nonlinear(m,None,num_steps=2,max_iterations=20,min_step_fraction=.0625,kinematics='corotational',corotational_tangent='work_conjugate')
    assert r.converged and r.load_factor==1
    assert r.info['corotational_shell_rotation_stable'] is True
    assert r.info['total_newton_iterations']<=6
    assert r.info['nonlinear_performance']['solver']['failed_increments']==0
    free=np.setdiff1d(np.arange(len(r.displacements)),list(m.mesh.nodes[1].dofs)+list(m.mesh.nodes[3].dofs)+[m.mesh.nodes[2].dofs[1],m.mesh.nodes[4].dofs[1]])
    force,_,_=_assemble_nonlinear_system(m,r.displacements,{},5,tangent=False,kinematics='corotational',corotational_tangent='work_conjugate',stable_shell_rotation=r.info['corotational_shell_rotation_stable'])
    assert np.linalg.norm(force[free])<=1e-6
    np.testing.assert_allclose(r.displacements,linear,atol=2e-12,rtol=1e-6)

@pytest.mark.parametrize('angle',[1e-9,1e-7,1e-4,.3])
def test_rotation_difference_matches_high_precision_rodrigues(angle):
    decimal.getcontext().prec=80;D=decimal.Decimal
    axis=np.array([2.,-3.,4.]);axis/=np.linalg.norm(axis);v=angle*axis
    vd=[D(float(x)) for x in v];a2=sum(x*x for x in vd)
    c1=sum((-1)**j*a2**j/D(__import__('math').factorial(2*j+1)) for j in range(20))
    c2=sum((-1)**j*a2**j/D(__import__('math').factorial(2*j+2)) for j in range(20))
    K=[[D(0),-vd[2],vd[1]],[vd[2],D(0),-vd[0]],[-vd[1],vd[0],D(0)]]
    exact=np.array([[float(c1*K[j][i]+c2*sum(K[j][k]*K[k][i] for k in range(3))) for j in range(3)] for i in range(3)])
    result=cr._stable_shell_rotation_difference(cr.rotation_matrix_from_vector(v))
    # Matrix products/normalization accumulate several ulps; use a normwise
    # forward-roundoff bound rather than relative error of near-zero entries.
    assert np.max(abs(result-exact)) <= 16*np.finfo(float).eps*np.linalg.norm(exact)

def test_rotation_difference_retains_near_pi_and_nonrotation_fallback():
    for R in [cr.rotation_matrix_from_vector(np.array([0.,0.,np.pi-1e-8])),np.diag([-1.,1.,1.])]:
        np.testing.assert_array_equal(cr._stable_shell_rotation_difference(R),R.T-np.eye(3))

@pytest.mark.parametrize('mode',['rotated','consistent','work_conjugate'])
def test_automatic_precision_keeps_explicit_route_policies(mode):
    m,_=fixture.plate_model({'name':'policy','E':1e12});e=m.mesh.elements[1];u=np.zeros(24);u[[7,19]]=5e-7
    explicit=False
    auto=cr.corotational_element_response(m,1,e,u,tangent=False,tangent_mode=mode)[0]
    fixed=cr.corotational_element_response(m,1,e,u,tangent=False,tangent_mode=mode,stable_shell_rotation=explicit)[0]
    np.testing.assert_array_equal(auto,fixed)
    if mode=='work_conjugate':
        direct=cr.corotational_work_conjugate_response(m,1,e,u,tangent=False)[0]
        np.testing.assert_array_equal(auto,direct)


def test_history_owner_keeps_historical_policy_and_recovery_reads_result_policy():
    from anyfesolver.recovery import recover_stress_result
    for initial,expected in [(None,True),({},False)]:
        m,_=fixture.plate_model({'name':'recovery','E':1e5})
        r=solve_static_nonlinear(m,None,num_steps=2,kinematics='corotational',corotational_tangent='work_conjugate',initial_element_states=initial)
        assert r.converged
        assert r.info['corotational_shell_rotation_stable'] is expected
        recovered=recover_stress_result(m,nonlinear_result=r)
        assert recovered.provenance.analysis_context['corotational_shell_rotation_stable'] is expected
    del r.info['corotational_shell_rotation_stable']
    old=recover_stress_result(m,nonlinear_result=r)
    assert old.provenance.analysis_context['corotational_shell_rotation_stable'] is False


@pytest.mark.parametrize('scale,origin,angle',[(1.,0.,1e-6),(.01,0.,1e-6),(1.,1000.,.3)])
def test_selected_precision_mixed_motion_tangent_and_balance(scale,origin,angle):
    m,_=fixture.plate_model({'name':'derivative','scale':scale,'origin':origin})
    e=m.mesh.elements[1];coords=e.get_node_coordinates(m.mesh);arm=coords-coords.mean(axis=0)
    rng=np.random.default_rng(5);u=np.zeros((4,6));u[:,:3]=1e-6*scale*rng.normal(size=(4,3));u[:,3:]=1e-6*rng.normal(size=(4,3))
    R=cr.rotation_matrix_from_vector(angle*np.array([.2,-.3,.4]))
    u[:,:3]=(arm+u[:,:3])@R.T-arm
    for i in range(4):u[i,3:]=cr.rotation_vector_from_matrix(R@cr.rotation_matrix_from_vector(u[i,3:]))
    u=u.ravel();direction=rng.normal(size=24);direction.reshape(4,6)[:,:3]*=scale;direction/=np.linalg.norm(direction)
    force,k,_=cr.corotational_element_response(m,1,e,u,tangent=True,tangent_mode='work_conjugate',stable_shell_rotation=True)
    h=1e-5*scale
    plus=cr.corotational_element_response(m,1,e,u+h*direction,tangent=False,tangent_mode='work_conjugate',stable_shell_rotation=True)[0]
    minus=cr.corotational_element_response(m,1,e,u-h*direction,tangent=False,tangent_mode='work_conjugate',stable_shell_rotation=True)[0]
    fd=(plus-minus)/(2*h)
    assert np.linalg.norm(k@direction-fd)/np.linalg.norm(fd)<1e-7
    assert np.linalg.norm(k-k.T)/np.linalg.norm(k)<1e-12
    current=coords+u.reshape(4,6)[:,:3]
    f,moment=cr.spatial_force_and_moment_resultants(current,u.reshape(4,6)[:,3:],force,current.mean(axis=0))
    assert np.linalg.norm(f)/max(np.linalg.norm(force),1.)<1e-12
    assert np.linalg.norm(moment)/max(np.linalg.norm(force)*scale,1.)<1e-11
