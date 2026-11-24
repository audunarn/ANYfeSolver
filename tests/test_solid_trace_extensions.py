"""Actual production MIN3 edge-field oracles; no coupling qualification claim."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver import FEModel
from anyfesolver.elements import create_shell_element, ShellElement
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.e4_pl_s3_v2c_element import HAMMER_POINTS
from anyfesolver.solid_trace import ShellEdgeTrace
from anyfesolver._coupled_certificate import CoupledSolveError as SolidModelError


def triangle(*, normal_sign=1, polarity=1, offset=.025, rotated=True):
    model=FEModel('production-S3-trace')
    model.add_material('m',1000.,.3)
    frame=Rotation.from_rotvec([.3,-.4,.2]).as_matrix() if rotated else np.eye(3)
    coords=np.array([[0.,0.,0.],[2.,0.,0.],[0.,1.,0.]])@frame.T+np.array([.7,-.2,.4])
    for nid,x in enumerate(coords,1):
        model.add_node(nid,*x)
    element=NativeParityE4PLS3V2DShellElement(1,[1,2,3],'m',thickness=.2,
        reference_normal=normal_sign*frame[:,2],director_polarity=polarity,
        reference_surface_offset=offset)
    model.add_element(1,element)
    return model,element


def native_surface(model,element,reference_point,point,q):
    """Independent field from the actual pressure covector's l_i/m_i values.

    Uses production geometry/order/global-DOF transformation, with the explicit
    pressure displacement field, not the compact boundary cross-product rule.
    """
    geometry=element._geometry(model.mesh)
    frame=geometry['frame']
    local=geometry['local_coordinates']
    origin=geometry['external_coordinates'][geometry['internal_order'][0]]
    jac=np.column_stack((local[1]-local[0],local[2]-local[0]))
    r,s=np.linalg.solve(jac,frame[:,:2].T@(reference_point-origin))
    N=np.array([1-r-s,r,s])
    dofs=[d for nid in element.node_ids for d in model.mesh.dof_manager.get_node_dofs(nid)]
    local_q=(geometry['local_from_external']@q[dofs]).reshape(3,6)
    a=np.array([local[k,0]-local[j,0] for i,j,k in ((0,1,2),(1,2,0),(2,0,1))])
    b=np.array([local[j,1]-local[k,1] for i,j,k in ((0,1,2),(1,2,0),(2,0,1))])
    w=N@local_q[:,2]
    for i,j,k in ((0,1,2),(1,2,0),(2,0,1)):
        l=.5*N[i]*(b[k]*N[j]-b[j]*N[k])
        m=.5*N[i]*(a[j]*N[k]-a[k]*N[j])
        w+=-l*local_q[i,3]+m*local_q[i,4]
    displacement=N@local_q[:,:3]
    displacement[2]=w
    physical_rotation=frame@(N@local_q[:,3:])
    return frame@displacement+np.cross(physical_rotation,point-reference_point),geometry,local_q,N


@pytest.mark.parametrize('normal_sign,polarity',[(1,1),(-1,1),(1,-1),(-1,-1)])
@pytest.mark.parametrize('edge',[0,1,2])
def test_s3_matches_production_pressure_field_and_nodal_constants(normal_sign,polarity,edge):
    model,element=triangle(normal_sign=normal_sign,polarity=polarity)
    trace=ShellEdgeTrace(model,[(1,edge)],offset=.04)
    a,b=((0,1),(1,2),(2,0))[edge]
    xyz=element.get_node_coordinates(model.mesh)
    rng=np.random.default_rng(872)
    q=rng.normal(size=trace.ndof)
    alphas=np.array([0.,.19,.53,1.])
    references=(1-alphas[:,None])*xyz[a]+alphas[:,None]*xyz[b]
    points=references+np.array([-.04,.01,.05,.07])[:,None]*trace.normal
    actual=(trace.displacement_operator(points)@q).reshape(-1,3)
    expected=np.array([native_surface(model,element,x,p,q)[0] for x,p in zip(references,points)])
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    # At vertices the l/m integration constants vanish exactly.
    for index,node in ((0,element.node_ids[a]),(-1,element.node_ids[b])):
        dofs=model.mesh.dof_manager.get_node_dofs(node)
        np.testing.assert_allclose(actual[index],q[dofs[:3]]+np.cross(q[dofs[3:]],points[index]-references[index]),atol=1e-12)


@pytest.mark.parametrize('normal_sign,polarity',[(1,1),(-1,1),(1,-1)])
@pytest.mark.parametrize('edge',[0,1,2])
def test_s3_edge_derivative_and_retained_rotation_match_actual_native_operators(normal_sign,polarity,edge):
    model,element=triangle(normal_sign=normal_sign,polarity=polarity)
    trace=ShellEdgeTrace(model,[(1,edge)],offset=.04)
    xyz=element.get_node_coordinates(model.mesh)
    a,b=((0,1),(1,2),(2,0))[edge]
    alpha=.43
    x=(1-alpha)*xyz[a]+alpha*xyz[b]
    rng=np.random.default_rng(85)
    q=rng.normal(size=trace.ndof)
    _,geometry,local_q,N=native_surface(model,element,x,x,q)
    operators=element._operators(geometry,{})
    # Production B_s is affine in barycentric coordinates. Reconstruct its
    # value at the edge from the actual three production Hammer stations.
    local=geometry['local_coordinates']
    origin=geometry['external_coordinates'][geometry['internal_order'][0]]
    r,s=np.linalg.solve(np.column_stack((local[1]-local[0],local[2]-local[0])),geometry['frame'][:,:2].T@(x-origin))
    station=np.column_stack((np.ones(3),np.asarray(HAMMER_POINTS)))
    Bs=(np.array([1.,r,s])@np.linalg.solve(station,operators['B_s'].reshape(3,-1))).reshape(2,18)
    length=np.linalg.norm(xyz[b]-xyz[a])
    tangent=(xyz[b]-xyz[a])/length
    normal=geometry['frame'][:,2]
    h=1e-5
    values=(trace.displacement_operator(np.array([x-h*tangent,x+h*tangent]))@q).reshape(2,3)
    dw_ds=((values[1]-values[0])/(2*h))@normal
    theta=geometry['frame']@(N@local_q[:,3:])
    physical_gamma=dw_ds+np.dot(np.cross(theta,normal),tangent)
    local_t=geometry['frame'][:,:2].T@tangent
    np.testing.assert_allclose(physical_gamma,local_t@(Bs@local_q.ravel()),rtol=2e-9,atol=2e-9)
    dof_a=model.mesh.dof_manager.get_node_dofs(element.node_ids[a])
    dof_b=model.mesh.dof_manager.get_node_dofs(element.node_ids[b])
    rotation_derivative=(q[dof_b[3:]]-q[dof_a[3:]])/length
    curvature=np.array([local_t[0]**2,local_t[1]**2,local_t[0]*local_t[1]])@(operators['B_b'][0]@local_q.ravel())
    np.testing.assert_allclose(curvature,np.dot(np.cross(rotation_derivative,normal),tangent),atol=1e-12)


def mixed():
    model=FEModel('mixed-production-trace')
    model.add_material('m',1000.,.3)
    for nid,x in enumerate([[-1.,0.,0.],[0.,0.,0.],[0.,1.,0.],[-1.,1.,0.],[0.,2.,0.]],1):
        model.add_node(nid,*x)
    model.add_element(1,create_shell_element(1,[1,2,3,4],'m',thickness=.2,reference_normal=(0.,0.,1.)))
    model.add_element(2,create_shell_element(2,[4,3,5],'m',thickness=.2,reference_normal=(0.,0.,1.)))
    assert type(model.mesh.elements[2]) is NativeParityE4PLS3V2DShellElement
    return model


@pytest.mark.parametrize('mixed_chain',[False,True])
def test_global_rigid_work_and_resultants_with_permuted_actual_dofs(mixed_chain):
    model=mixed() if mixed_chain else triangle()[0]
    rng=np.random.default_rng(192)
    permutation=rng.permutation(model.mesh.dof_manager.total_dofs)
    for i,nid in enumerate(model.mesh.nodes):
        model.mesh.dof_manager.get_node_dofs(nid)[:]=permutation[6*i:6*i+6].tolist()
    trace=ShellEdgeTrace(model,[(1,1),(2,1)] if mixed_chain else [(1,1)],offset=.037)
    points=trace.points(np.array([[.1*trace.length,-.05],[.4*trace.length,.02],[.9*trace.length,.07]]))
    A=trace.displacement_operator(points)
    for mode in range(6):
        tr,omega=np.zeros(3),np.zeros(3)
        (tr if mode<3 else omega)[mode%3]=1.
        q=np.zeros(trace.ndof)
        for node in model.mesh.nodes.values():
            dofs=model.mesh.dof_manager.get_node_dofs(node.id)
            q[dofs[:3]]=tr+np.cross(omega,node.coords())
            q[dofs[3:]]=omega
        np.testing.assert_allclose((A@q).reshape(-1,3),tr+np.cross(omega,points),atol=1e-12)
    q=rng.normal(size=trace.ndof)
    f=rng.normal(size=(len(points),3))
    nodal=A.T@f.ravel()
    np.testing.assert_allclose((A@q)@f.ravel(),q@nodal,atol=1e-12)
    resultant,moment=np.zeros(3),np.zeros(3)
    for node in model.mesh.nodes.values():
        dofs=model.mesh.dof_manager.get_node_dofs(node.id)
        resultant+=nodal[dofs[:3]]
        moment+=np.cross(node.coords(),nodal[dofs[:3]])+nodal[dofs[3:]]
    np.testing.assert_allclose(resultant,f.sum(axis=0),atol=1e-12)
    np.testing.assert_allclose(moment,np.cross(points,f).sum(axis=0),atol=1e-12)
    if mixed_chain:
        assert trace.length==2.
        assert trace.displacement_operator(trace.points([[1.,0.]])).shape==(3,trace.ndof)


@pytest.mark.parametrize('mutation',['coordinates','thickness','director','polarity','reference_offset','dofs','replacement','removal'])
def test_s3_stale_state_is_refused(mutation):
    model,element=triangle()
    trace=ShellEdgeTrace(model,[(1,0)])
    if mutation=='coordinates': model.mesh.nodes[1].x+=.01
    elif mutation=='thickness': element.thickness=.3
    elif mutation=='director': element.reference_normal=-element.reference_normal
    elif mutation=='polarity': element.director_polarity=-element.director_polarity
    elif mutation=='reference_offset': element.reference_surface_offset=.03
    elif mutation=='dofs':
        dofs=model.mesh.dof_manager.get_node_dofs(1)
        dofs[0],dofs[1]=dofs[1],dofs[0]
    elif mutation=='replacement':
        model.mesh.elements[1]=NativeParityE4PLS3V2DShellElement(1,[1,2,3],'m',thickness=.2,
            reference_normal=element.reference_normal,reference_surface_offset=.025)
    else: del model.mesh.elements[1]
    with pytest.raises(SolidModelError,match='stale'):
        trace.displacement_operator(np.empty((0,3)))


def test_admission_geometry_and_sampling_refusals():
    model,element=triangle()
    for edges in ([],[(1,3)],[(1,0),(1,0)]):
        with pytest.raises(SolidModelError): ShellEdgeTrace(model,edges)
    trace=ShellEdgeTrace(model,[(1,0)])
    with pytest.raises(AttributeError): trace.offset=.1
    with pytest.raises(ValueError): trace.tangent.setflags(write=True)
    for points in ([[np.nan,0.,0.]],trace.points([[-1.,0.]]),trace.points([[0.,.2]])):
        with pytest.raises(SolidModelError): trace.displacement_operator(points)
    model.mesh.elements[1]=ShellElement(1,[1,2,3],'m',thickness=.2)
    with pytest.raises(SolidModelError,match='exact production'):
        ShellEdgeTrace(model,[(1,0)])
    model=mixed()
    model.mesh.nodes[1].z=.1
    with pytest.raises(SolidModelError,match='planar'):
        ShellEdgeTrace(model,[(1,1)])
    model=mixed()
    # Connected edge makes a right-angle corner rather than a straight trace.
    with pytest.raises(SolidModelError): ShellEdgeTrace(model,[(1,1),(2,0)])
    with pytest.raises(SolidModelError): ShellEdgeTrace(model,[(2,1),(1,1)])
    model.mesh.elements[2].thickness=.3
    with pytest.raises(SolidModelError,match='thickness'):
        ShellEdgeTrace(model,[(1,1),(2,1)])


def test_s3_trace_retains_nonzero_production_min3_edge_displacement():
    model,element=triangle(rotated=False)
    trace=ShellEdgeTrace(model,[(1,0)])
    q=np.zeros(trace.ndof)
    q[model.mesh.dof_manager.get_node_dofs(1)[4]]=1.
    xyz=element.get_node_coordinates(model.mesh)
    point=(xyz[0]+xyz[1])/2
    # Linear geometric translation interpolation alone would give zero.
    # Production MIN3 gives .5*.5*.5*(theta_a-theta_b) cross edge = -.25z.
    np.testing.assert_allclose(trace.displacement_operator([point])@q,[0.,0.,-.25],atol=1e-13)


@pytest.mark.parametrize('mode',['corner','overlap','director','thickness_nan','reference_offset_nan'])
def test_connected_mixed_corners_overlaps_and_director_jumps_are_refused(mode):
    model=mixed()
    if mode=='corner':
        model.mesh.nodes[5].x=1.
    elif mode=='overlap':
        model.mesh.nodes[5].y=.5
    elif mode=='thickness_nan':
        model.mesh.elements[2].thickness=np.nan
    elif mode=='reference_offset_nan':
        model.mesh.elements[2].reference_surface_offset=np.nan
    else:
        model.mesh.elements[2].reference_normal=np.array([0.,0.,-1.])
    with pytest.raises(SolidModelError,match='straight|directors|thickness|offset'):
        ShellEdgeTrace(model,[(1,1),(2,1)])


@pytest.mark.parametrize('setting',['default','raise'])
@pytest.mark.parametrize('tilt',[0.,.5],ids=['aligned','tilted'])
def test_trace_geometry_out_of_float_range_is_refused_with_a_code(tilt,setting):
    """Class sweep (round 5): coordinates ~1e160 overflow the facet normal; 717c8e003 raised an uncoded
    FloatingPointError under np.errstate(all='raise') (fails there). Ordinary geometry is unchanged."""
    def build(scale):
        model=FEModel('huge-S3-trace')
        model.add_material('m',1000.,.3)
        for nid,x in enumerate(np.array([[0.,0.,0.],[2.,0.,0.],[0.,1.,0.]])*scale,1):
            model.add_node(nid,*x)
        normal=np.array([0.,tilt,1.])/np.hypot(tilt,1.)
        model.add_element(1,NativeParityE4PLS3V2DShellElement(1,[1,2,3],'m',thickness=.2*scale,reference_normal=normal))
        return ShellEdgeTrace(model,[(1,0)],tolerance=1e-10*scale)
    with np.errstate(**({'all':'raise'} if setting=='raise' else {})):
        with pytest.raises(SolidModelError) as info:
            build(1e160)
        assert info.value.code=='COUPLED_TRACE_INVALID'
        if not tilt:
            assert build(1.).length==2.


def _codex_r6_triangle(coords, tolerance):
    model=FEModel('anisotropic')
    model.add_material('m',1000.,.3)
    for nid,xyz in enumerate(coords,1):
        model.add_node(nid,*xyz)
    model.add_element(1,NativeParityE4PLS3V2DShellElement(1,[1,2,3],'m',thickness=.2,reference_normal=(0.,0.,1.)))
    return ShellEdgeTrace(model,[(1,0)],tolerance=tolerance)


@pytest.mark.parametrize('setting',['default','raise'])
def test_inward_vector_with_overflowing_norm_cannot_give_a_zero_outward(setting):
    """Codex round-6 P2: the inward side vector (0, 3.3e154, 0) is finite but its norm overflowed under the
    local errstate(all='ignore'); the division gave solid_outward = 0 and displacement_operator accepted a
    point one unit outside the interface plane (tolerance 1e-160). Fails on 6c44eec02."""
    with np.errstate(**({'all':'raise'} if setting=='raise' else {})):
        trace=_codex_r6_triangle([[0.,0.,0.],[1e-150,0.,0.],[0.,1e155,0.]],1e-160)
        assert np.array_equal(trace.solid_outward,[0.,1.,0.])
        with pytest.raises(SolidModelError) as info:
            trace.displacement_operator([[5e-151,1.,0.]])
        assert info.value.code=='COUPLED_TRACE_INVALID'
        assert trace.displacement_operator([[5e-151,0.,0.]]).shape==(3,18)


@pytest.mark.parametrize('setting',['default','raise'])
def test_trace_unit_vectors_do_not_lose_digits_to_underflow(setting):
    """Class sweep (round 6): an edge of length ~2^-528 has squares ~2^-1056 (subnormal), so the plain norm lost
    digits and the 'unit' tangent and the length were not the exactly scaled ones (6c44eec02 differs: fails
    there). With an exact power-of-two prescale they equal those of the unscaled edge."""
    scale=2.0**-530
    reference=_codex_r6_triangle([[0.,0.,0.],[3.,.7,0.],[0.,1.,0.]],1e-10)
    with np.errstate(**({'all':'raise'} if setting=='raise' else {})):
        scaled=_codex_r6_triangle([[0.,0.,0.],[3.*scale,.7*scale,0.],[0.,1.,0.]],1e-10*scale)
    assert scaled.tangent.tobytes()==reference.tangent.tobytes()
    assert scaled.length==reference.length*scale
    assert abs(np.linalg.norm(scaled.solid_outward)-1.)<=4e-16


def test_ordinary_trace_unit_vectors_keep_the_original_division_bit_for_bit():
    """In range, _scaled_norm/_scaled_unit are exactly np.linalg.norm and v / np.linalg.norm(v)."""
    from anyfesolver._coupled_certificate import _scaled_norm,_scaled_unit
    rng=np.random.default_rng(6)
    for v in rng.normal(size=(200,3))*np.exp(rng.uniform(-300,300,size=(200,1))):
        assert _scaled_norm(v)==np.linalg.norm(v)
        assert _scaled_unit(v).tobytes()==(v/np.linalg.norm(v)).tobytes()
    assert _scaled_norm([1.5e308,1.5e308,0.])==np.inf
    assert np.allclose(_scaled_unit([1.5e308,-1.5e308,0.]),np.array([1.,-1.,0.])/np.sqrt(2.),rtol=4e-16,atol=0)
    assert _scaled_unit([0.,0.,0.]) is None and _scaled_unit([np.inf,0.,0.]) is None
    assert np.isnan(_scaled_norm([np.nan,0.,0.]))


def _flat_trace():
    m=FEModel('trace'); m.add_material('m',1000.,.3)
    for i,xyz in enumerate([[0.,0.,0.],[1.,1.,0.],[-1.,1.,0.]],1): m.add_node(i,*xyz)
    m.add_element(1,NativeParityE4PLS3V2DShellElement(1,[1,2,3],'m',thickness=.2,reference_normal=(0.,0.,1.)))
    return ShellEdgeTrace(m,[(1,0)])


@pytest.mark.parametrize('mode',['ignore','raise'])
@pytest.mark.parametrize('points',[[[1.7e308,1.7e308,0.]],[[np.nan,0.,0.]],[[np.inf,0.,0.]],[[1.7e308,1.7e308,1.7e308]]])
def test_chart_refuses_nonfinite_or_overflowing_points_under_any_errstate(mode,points):
    """Codex round 5 P2 (pre-existing): chart() gave inf (ignore) or an uncoded FloatingPointError (raise)."""
    trace=_flat_trace()
    with np.errstate(all=mode),pytest.raises(SolidModelError) as info:
        trace.chart(points)
    assert info.value.code=='COUPLED_TRACE_INVALID'


@pytest.mark.parametrize('mode',['ignore','raise'])
@pytest.mark.parametrize('chart',[[[np.nan,0.]],[[0.,np.inf]],[1.,2.],[['a','b']]])
def test_points_refuses_nonfinite_or_malformed_chart_under_any_errstate(mode,chart):
    trace=_flat_trace()
    with np.errstate(all=mode),pytest.raises(SolidModelError) as info:
        trace.points(chart)
    assert info.value.code=='COUPLED_TRACE_INVALID'


@pytest.mark.parametrize('mode',['ignore','raise'])
def test_ordinary_chart_and_points_are_unchanged_under_any_errstate(mode):
    trace=_flat_trace()
    pts=np.array([[.1,.3,.05],[.5,.5,-.04]])
    relative=pts-trace.reference_start
    expected=np.column_stack((relative@trace.tangent,relative@trace.normal-trace.offset))
    chart_ref=np.array([[.2,.01],[1.,-.05]])
    expected_pts=trace.reference_start+chart_ref[:,0,None]*trace.tangent+(chart_ref[:,1,None]+trace.offset)*trace.normal
    with np.errstate(all=mode):
        assert trace.chart(pts).tobytes()==expected.tobytes()
        assert trace.points(chart_ref).tobytes()==expected_pts.tobytes()


@pytest.mark.parametrize('mode',['ignore','raise'])
def test_scaled_norm_and_unit_survive_a_raising_errstate(mode):
    """ldexp underflow of tiny components next to huge ones raised FloatingPointError under 'raise'."""
    from anyfesolver._coupled_certificate import _scaled_norm,_scaled_unit
    with np.errstate(all=mode):
        assert _scaled_norm([1e-300,1e300])==1e300
        assert _scaled_unit([1e-300,1e300]) is not None
