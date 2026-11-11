"""Independent linear centroidal force/rigid-arm controls."""
import numpy as np
import pytest
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement

def beam(offsets=None):
 m=FEModel('beam controls');m.add_node(1,0,0,0);m.add_node(2,2,0,0)
 b=BeamElement(1,[1,2],cross_section={'area':.001,'Iy':8e-7,'Iz':8e-9,'J':3e-8,'orientation':(0,0,1),'shear_factor_y':2/3,'shear_factor_z':2/3})
 m.add_element(1,b)
 if offsets is not None:b.set_linear_nodal_offsets(offsets)
 return m,b,m.get_material('default')

@pytest.mark.parametrize('component',[0,1,2,3])
def test_axial_both_bending_planes_torsion_tip_force_equilibrium(component):
 m,b,mat=beam();K=b.compute_stiffness_matrix(m.mesh,mat);force=np.zeros(6);force[component]=100.
 u=np.zeros(12);u[6:]=np.linalg.solve(K[6:,6:],force)
 E=mat.elastic_modulus;G=E/(2*(1+mat.poisson_ratio));L=2.;P=100.;A=.001
 if component==0:assert u[6]==pytest.approx(P*L/(E*A),rel=1e-12)
 elif component==1:
  assert u[7]==pytest.approx(P*L**3/(3*E*8e-9)+P*L/(G*(2/3)*A),rel=1e-12)
  assert u[11]==pytest.approx(P*L**2/(2*E*8e-9),rel=1e-12)
 elif component==2:
  assert u[8]==pytest.approx(P*L**3/(3*E*8e-7)+P*L/(G*(2/3)*A),rel=1e-12)
  assert u[10]==pytest.approx(-P*L**2/(2*E*8e-7),rel=1e-12)
 else:assert u[9]==pytest.approx(P*L/(G*3e-8),rel=1e-12)
 result=b.recover_linear_section_forces(m.mesh,u,mat,[[0,0,0],[2,0,0]])
 np.testing.assert_allclose(result[-1],force,atol=1e-7)
 expected=force.copy();expected[3:]+=np.cross([2,0,0],force[:3])
 np.testing.assert_allclose(result[0],expected,atol=1e-7)

def test_offset_work_duality_and_rigid_body_kernel():
 m,b,mat=beam([(0,0,-.05),(0,0,-.08)]);coords=b.get_node_coordinates(m.mesh)
 L,T=b._beam_frame_and_transform(coords);K=b.compute_stiffness_matrix(m.mesh,mat)
 np.testing.assert_allclose(K,T.T@b._local_linear_stiffness(L,mat)@T)
 A=b._linear_rigid_arm_map();rng=np.random.default_rng(74);u=rng.normal(size=12);f=rng.normal(size=12)
 assert (A@u)@f==pytest.approx(u@(A.T@f))
 for axis in np.eye(3):
  rigid=np.r_[np.cross(axis,[0,0,0]),axis,np.cross(axis,[2,0,0]),axis]
  np.testing.assert_allclose(K@rigid,0,atol=2e-8)
 with pytest.raises(ValueError,match='linear static'):b.compute_mass_matrix(m.mesh,mat)
 with pytest.raises(ValueError,match='linear static'):b.compute_nonlinear_response(m.mesh,mat,np.zeros(12))
 with pytest.raises(ValueError,match='linear static'):b.compute_geometric_stiffness_matrix(m.mesh,mat,{'axial_compression':1.})
 with pytest.raises(ValueError,match='finite Nx3'):b.recover_linear_section_forces(m.mesh,np.zeros(12),mat,[[float('nan'),0,0]])
 with pytest.raises(ValueError,match='centroid line'):b.recover_linear_section_forces(m.mesh,np.zeros(12),mat,[[3,0,0]])

def test_legacy_zero_offset_mass_is_unchanged():
 m,b,mat=beam();expected=b.compute_mass_matrix(m.mesh,mat)
 b.set_linear_nodal_offsets(np.zeros((2,3)))
 np.testing.assert_array_equal(b.compute_mass_matrix(m.mesh,mat),expected)
 with pytest.raises(ValueError,match='near-zero length'):
  b.set_linear_nodal_offsets([(0,0,0),(-2,0,0)]);b.compute_stiffness_matrix(m.mesh,mat)

def test_rectangular_admission_rounding_bound_has_length_units():
 from anyfesolver.sestra_mixed_verification import rectangular_serialization_admission
 xyz=np.array([[1.,1,0],[2,1,0],[2,2,0],[1,2,0]])
 xyz[2,0]+=1e-7
 assert rectangular_serialization_admission(xyz)[0]
 xyz[2,0]+=1e-4
 assert not rectangular_serialization_admission(xyz)[0]

def shell(distorted=True):
 from anyfesolver.elements import create_shell_element
 m=FEModel('sampling controls');xy=[(0,0),(2,0),(2.2 if distorted else 2,1),(0,1)]
 for nid,(x,y) in enumerate(xy,1):m.add_node(nid,x,y,0)
 e=create_shell_element(1,[1,2,3,4],thickness=.02);m.add_element(1,e)
 mat=m.get_material();e.compute_stiffness_matrix(m.mesh,mat)
 return m,e,mat

def test_custom_qualified_points_match_affine_membrane_and_symmetric_bending():
 m,e,mat=shell();u=np.zeros(24);a,b=1e-5,-2e-5
 for i,nid in enumerate(e.node_ids):
  x,y,_=m.mesh.nodes[nid].coords();u[6*i]=a*x;u[6*i+1]=b*y;u[6*i+4]=-1e-4*x
 saved=np.array(e.gauss_points,copy=True);points=[(-1,-1),(1,-1),(0,0),(1,1),(-1,1)]
 result=e.compute_stresses(m.mesh,u,mat,return_global=True,natural_points=points)
 E,nu=mat.elastic_modulus,mat.poisson_ratio
 expected=np.diag([E/(1-nu**2)*(a+nu*b),E/(1-nu**2)*(b+nu*a),0.])
 np.testing.assert_allclose(result['global_membrane_resultant_tensors']/.02,np.broadcast_to(expected,(5,3,3)),rtol=1e-10,atol=1e-6)
 np.testing.assert_allclose(.5*(result['local_xx_top']+result['local_xx_bot']),result['membrane_xx'],atol=1e-7)
 assert np.max(np.abs(result['bending_xx']))>0
 np.testing.assert_array_equal(e.gauss_points,saved)

def test_explicit_default_points_bitwise_equivalent_and_bad_points_refuse():
 from anyfesolver.e4_pl_element import _GAUSS
 m,e,mat=shell();u=np.arange(24)*1e-8
 default=e.compute_stresses(m.mesh,u,mat,return_global=True)
 explicit=e.compute_stresses(m.mesh,u,mat,return_global=True,natural_points=_GAUSS)
 for k,v in default.items():
  if isinstance(v,np.ndarray):np.testing.assert_array_equal(v,explicit[k])
 for points in ([],[[float('nan'),0]],[[2,0]],[[0,0,0]]):
  with pytest.raises(ValueError,match='natural stress points'):e.compute_stresses(m.mesh,u,mat,natural_points=points)

def test_custom_points_refuse_warped_and_generalized_routes():
 from anyfesolver.elements import create_shell_element
 from anyfesolver.shell_sections import GeneralizedShellSection
 m,e,mat=shell();m.mesh.nodes[3].z=.2
 with pytest.raises(ValueError,match='warped'):e.compute_stresses(m.mesh,np.zeros(24),mat,natural_points=[[0,0]])
 m,e,mat=shell(False)
 section=GeneralizedShellSection(A=np.eye(3),B=np.zeros((3,3)),D=np.eye(3),As=np.eye(2))
 generalized=create_shell_element(2,[1,2,3,4],thickness=.02,shell_section=section)
 with pytest.raises(ValueError,match='generalized'):generalized.compute_stresses(m.mesh,np.zeros(24),mat,natural_points=[[0,0]])
