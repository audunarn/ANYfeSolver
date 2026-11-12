"""Installed native reference controls, no continuum/native response score."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from scripts import r8_bl5_matched_native_reference as tool


@pytest.fixture(scope='module')
def owner():
    analysis=ss.create_analysis(definition=tool.definition())
    yield analysis
    analysis.close()


def test_actual_installed_runtime():
    if 'ANYSOLVER_MATCHED_INSTALLED' not in os.environ:
        pytest.skip('ANYSOLVER_MATCHED_INSTALLED installed-runtime binding absent')
    import anyfesolver, numba
    from threadpoolctl import threadpool_info
    expected=Path(os.environ['ANYSOLVER_MATCHED_INSTALLED']).resolve()
    assert Path(anyfesolver.__file__).resolve().is_relative_to(expected)
    assert all(Path(m.__file__).resolve().is_relative_to(expected) for n,m in sys.modules.items()
        if n.startswith('anyfesolver.') and getattr(m,'__file__',None))
    assert numba.config.DISABLE_JIT and numba.get_num_threads()==1
    assert all(p['num_threads']==1 for p in threadpool_info())


def test_rectangle_area_inertia_shear_and_torsion_tail():
    d=tool.rectangular_section(); r=tool.rectangular_section(.06,.04,terms=2000)
    assert d['area']==.0024
    np.testing.assert_allclose((d['Iy'],d['Iz']),(7.2e-7,3.2e-7),rtol=2e-15)
    assert abs(d['J']-r['J'])<=d['J_tail_bound']+r['J_tail_bound']
    assert 0<d['J']<.06*.04**3/3
    # Independent polynomial integration of squared parabolic traction.
    nodes,weights=np.polynomial.legendre.leggauss(4)
    coefficient=sum(weights*(1.5*(1-nodes**2))**2)/2
    assert abs(coefficient-6/5)<1e-14 and d['shear_factor']==5/6
    np.testing.assert_allclose(np.diag(d['mass']),7850*np.array((.0024,.0024,.0024,1.04e-6,7.2e-7,3.2e-7)),rtol=2e-15)


@pytest.mark.parametrize('sign',(-1,1))
def test_actual_fibre_section_matches_analytic_elastic_axes(owner,sign):
    element,inertia=owner._definition.beams[0].instantiate()
    section=element.operator.section
    d=tool.rectangular_section(); stiffness=section.background_factor.T@section.background_factor
    # Independent native axial strain epsilon_x = epsilon + z*kappa_y - y*kappa_z.
    for f in section.fibres:
        b=np.array((1.,0.,0.,0.,f.z,-f.y)); stiffness+=f.area*f.young*np.outer(b,b)
    np.testing.assert_allclose(stiffness,np.diag(d['stiffness']),rtol=3e-15,atol=1e-7)
    vector=sign*np.array((1e-8,0.,0.,0.,2e-5,-3e-5))
    np.testing.assert_allclose(stiffness@vector,d['stiffness']*vector,rtol=3e-15,atol=1e-10)
    response=section.response(vector,section.virgin())
    for f,recovered in zip(section.fibres,response.fibres):
        expected=vector[0]+f.z*vector[4]-f.y*vector[5]
        assert abs(sum(recovered.total_strain)-expected)<=1e-20
        assert abs(sum(recovered.stress)-f.young*expected)<=1e-9
        assert recovered.branch=='ELASTIC' and not any(recovered.plastic_increment)
    np.testing.assert_allclose(inertia,d['mass'],rtol=0,atol=0)


def test_geometry_supports_measures_and_reference_offsets(owner):
    p=owner.preflight()
    assert owner._K.shape==(168,168) and owner._C.shape==(20,168)
    assert len(owner._line_samples)==8 and len(owner._definition.shell.q4_elements)==8
    assert len(owner._fixed)==18 and p['retained_dofs']==48
    assert set(owner._fixed)=={6*(n-1)+j for n in (1,6,11) for j in range(6)}
    assert sum(s[2] for s in owner._line_samples)==1.
    for op,*_ in owner._line_samples:
        np.testing.assert_allclose(op.reference_arm,(0.,0.,-.03),rtol=0,atol=1e-16)
        np.testing.assert_array_equal(op.reference_director,(0.,0.,1.))
        np.testing.assert_array_equal(op.director_basis,((1.,0.,0.),(0.,1.,0.)))
    _,model=tool.external.build_reference()
    assert model['model_sha256']=='a22170179b6e257a47288ff825a3b69ad503e5e3c4a040fcd5f3b53933c86463'
    assert model['deck_sha256']=='54d72909717c07a82bdd4bb70e87b5084a402e529e60ba1a4576b86d99c0b154'


def test_local_five_channel_jacobian_and_transpose_wrench(owner):
    state=tool.reference_state(owner)
    for op,slots,weight,edge,alpha in owner._line_samples:
        actual=op.evaluate(np.zeros((2,6)),state,np.array((2.,-3.,5.,7.,-11.)))
        independent=np.zeros((5,21))
        for i in range(21):
            v=np.eye(21)[i]; s=v[:12].reshape(2,6); b=v[12:]
            us=(1-op.station)*s[0,:3]+op.station*s[1,:3]
            ts=(1-op.station)*s[0,3:]+op.station*s[1,3:]
            ub=(1-op.port.weight)*b[:3]+op.port.weight*b[3:6]
            independent[:,i]=-tool.external_gap_and_work(us,ts,ub,b[6:],np.zeros(5))[0]
        np.testing.assert_allclose(actual.gap_jacobian,independent,rtol=0,atol=2e-15)
        np.testing.assert_allclose(actual.gradient,independent.T@np.array((2.,-3.,5.,7.,-11.)),rtol=0,atol=2e-14)
        # Native nodal/cell covectors give zero total interface wrench at any origin.
        xyz=np.vstack((op.reference,np.array([owner._beam_nodes[owner._beam_ids[n]][0] for n in op.port.nodes])))
        force=np.vstack((actual.gradient[:3],actual.gradient[6:9],actual.gradient[12:15],actual.gradient[15:18]))
        moment=actual.gradient[3:6]+actual.gradient[9:12]+actual.gradient[18:21]
        for origin in ((0.,0.,0.),(7.,-11.,3.)):
            np.testing.assert_allclose(force.sum(axis=0),0.,atol=1e-14)
            np.testing.assert_allclose(moment+np.cross(xyz-origin,force).sum(axis=0),0.,atol=1e-13)


@pytest.mark.parametrize('supported',(False,True))
def test_actual_global_continuum_common_field_and_integrated_dual(owner,supported):
    sa=np.array(((0.,.13),(0.,-.21),(0.,.37)))
    ba=np.array(((.09,-.11),(-.08,.17),(.02,-.31)))
    ts=np.zeros(3) if supported else np.array((.07,-.12,.05)); tb=np.array((.09,.14,-.04))
    q,model,field=tool.global_common_field(owner,sa,ts,ba,tb,supported=supported)
    if supported: assert not np.any(q[owner._fixed])
    # Actual original 102-node mesh, not four synthetic face points.
    assert len(field)==102
    rows=tool.global_correspondence(owner,q,model,field)
    integrated=np.zeros((5,4)); density=np.arange(1.,21.).reshape(5,4)
    direct_work=0.
    for ng,eg,w,edge in rows:
        np.testing.assert_allclose(ng,-eg,rtol=0,atol=3e-15)
        integrated[:,edge]-=w*eg
        direct_work+=float((-w*density[:,edge])@eg)
    np.testing.assert_allclose(owner._C@q,integrated.ravel(),rtol=0,atol=2e-15)
    assert abs(float(density.ravel()@(owner._C@q))-direct_work)<1e-13
    assert abs(float((owner._C.T@density.ravel())@q)-direct_work)<1e-13
    concentrated=tool.concentrated_from_density(density)
    action={n:np.zeros(3) for n in field}; station_work=0.
    for station,multiplier in zip(model['stations'],concentrated):
        station_work+=float(multiplier@tool.external.gap(station,field))
        for n,f in tool.external.transpose_action(station,multiplier).items(): action[n]+=f
    assert abs(station_work-direct_work)<1e-13
    assert abs(sum(float(action[n]@field[n]) for n in field)-direct_work)<1e-13
    for origin in ((0.,0.,0.),(7.,-11.,3.)):
        np.testing.assert_allclose(tool.common_origin_wrench(model['physical_nodes'],action,origin),0.,atol=2e-13)


def test_common_supported_lift_rejects_root_rotation_clipping(owner):
    with pytest.raises(ValueError,match='zero shell constant and spin'):
        tool.global_common_field(owner,np.zeros((3,2)),(.1,0.,0.),np.zeros((3,2)),(0.,0.,0.),supported=True)


def test_broken_native_station_has_two_physical_directors(owner):
    assert [len(tool.station_sides(owner,x)) for x in tool.XS]==[1,2,2,2,1]
    sides=tool.station_sides(owner,.25); state=tool.reference_state(owner)
    cells=state.cell_rotations.copy(); cells[0,0]=rotation((.1,0.,0.)); cells[0,1]=rotation((-.1,0.,0.))
    varied=Mechanical(state.positions,state.position_low,state.nodal_frames,cells,state.resultants)
    directors=[op.port.pose_pair(varied)[2][:,2] for op in sides]
    assert np.linalg.norm(directors[0]-directors[1])>.1
    # One shared C3D20 station cannot equal both; neither averaging nor dropping a side.


def test_free_normal_spin_and_independent_supported_transverse_shear(owner):
    zero=np.zeros((3,2)); zspin=(0.,0.,.17)
    yaw,model,field=tool.global_common_field(owner,zero,(0.,0.,0.),zero,zspin,supported=True)
    np.testing.assert_allclose(owner._C@yaw,0.,atol=1e-15)
    affine=np.zeros((3,2)); affine[2,1]=.2
    shear,model,field=tool.global_common_field(owner,affine,(0.,0.,0.),affine,(0.,0.,0.),supported=True)
    np.testing.assert_allclose(owner._C@shear,0.,atol=1e-15)
    for native,eg,*_ in tool.global_correspondence(owner,shear,model,field):
        np.testing.assert_allclose(native,0.,atol=1e-15); np.testing.assert_allclose(eg,0.,atol=1e-15)
    # Actual retained compensated chord, not a normal-reconstructed director.
    from anyfesolver import _ge_beam3_mixed_ad as ad
    from anyfesolver._ge_beam3_p5.compensated import compensated_strain
    for element,*_ in owner._beam_parts:
        ref=element.operator.reference.coordinates; current=ref.copy(); current[:,2]+=.2*ref[:,0]
        jets=[ad.Jet2.constant(0.,36) for _ in range(36)]
        for left,right in ((0,1),(1,2)):
            z=compensated_strain(ad.constant_matrix(np.eye(3),36),ref,current,np.zeros_like(ref),jets,left,right)
            assert abs(z[2].value/(ref[right,0]-ref[left,0])-.2)<1e-15


def test_finite_common_rigid_pose_and_actual_spatial_chart_derivatives(owner):
    state=tool.reference_state(owner); r=rotation((.17,-.11,.23)); shift=np.array((.2,-.3,.1))
    moved=Mechanical(state.positions@r.T+shift,state.position_low,np.broadcast_to(r,(5,3,3)).copy(),
        np.broadcast_to(r,(2,2,3,3)).copy(),state.resultants)
    multiplier=np.array((2.,-3.,5.,7.,-11.)); v=np.arange(1.,22.)/31
    for op,*_ in owner._line_samples[::2]:
        total=np.zeros((2,6)); total[:,:3]=op.reference@r.T+shift-op.reference
        frames=np.tile(r,(2,1,1)); packet=op.evaluate_spatial(total,frames,moved,multiplier)
        np.testing.assert_allclose(packet.gap,0.,atol=2e-15)
        assert np.max(abs(packet.hessian-packet.hessian.T))<=1e-9
        def gradient(step):
            t=total.copy(); t[:,:3]+=step*v[:12].reshape(2,6)[:,:3]
            f=np.array([rotation(step*v[6*i+3:6*i+6])@r for i in range(2)])
            p=moved.positions.copy(); cells=moved.cell_rotations.copy()
            for i,n in enumerate(op.port.nodes): p[n]+=step*v[12+3*i:15+3*i]
            cells[op.port.element_index,op.port.cell]=rotation(step*v[18:])@r
            s=Mechanical(p,moved.position_low,moved.nodal_frames,cells,moved.resultants)
            g=op.evaluate_spatial(t,f,s,multiplier).gradient.copy()
            for start in (3,9,18): g[start:start+3]=_exp_terms(step*v[start:start+3])[1].T@g[start:start+3]
            return g
        exact=packet.hessian@v
        errors=[np.max(abs((gradient(h)-gradient(-h))/(2*h)-exact))/max(1.,np.max(abs(exact))) for h in (3e-6,1e-6,3e-7)]
        assert max(errors)<=1e-6,errors


@pytest.mark.skipif('ANYSOLVER_MATCHED_RESULT' not in os.environ,
    reason='ANYSOLVER_MATCHED_RESULT installed calibration binding absent')
def test_one_virgin_public_reference_linear_calibration(owner):
    Path(os.environ['ANYSOLVER_MATCHED_RESULT']).with_name('input-v2.json').write_text(json.dumps(tool.input_record(owner._definition),indent=2)+'\n')
    result=tool.calibration(owner)
    assert result['status']=='completed' and not result['comparison_permitted']
    assert result['free_equilibrium_ratio']<=1e-8 and result['constraint_max']<=1e-9
    assert max(abs(v) for v in result['external_wrench_residual'])<=1e-8
    assert abs(result['internal_pairing']-result['applied_virtual_work'])<=1e-9
    assert abs(result['interface_pairing'])<=1e-9
    assert result['internal_work_relative_error']<=1e-8
    assert result['interface_work_relative_error']<=1e-8
    Path(os.environ['ANYSOLVER_MATCHED_RESULT']).write_text(json.dumps(result,indent=2)+'\n')
