"""Inactive-branch elimination never assigns a nonunique closest point."""
from pathlib import Path
from types import SimpleNamespace, MethodType
import copy
import numpy as np
import pytest
from anysolver.general_contact import corner_contact_geometry as geometry
from anysolver.general_contact.corner_contact_certificate import CornerContactRefusal
from anysolver.general_contact.corner_contact_indentation import CoupledIndentation

FIXTURE=Path(__file__).parent/'data/general_contact/s11_unload_first_separated_tie.npz'

def captured(fixture=FIXTURE):
    with np.load(fixture,allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    panel=SimpleNamespace(face_slices=tuple(map(tuple,data['face_slices'])),
        faces=[SimpleNamespace(node_ids=tuple(map(int,row))) for row in data['face_nodes']])
    pose=SimpleNamespace(positions=data['panel_positions'],position_errors=data['panel_errors'],
        director_gradient=data['panel_director_gradient'])
    corner=int(data['matching_corner_ids'][0])
    error=data['sphere_errors'][corner:corner+1]
    return data,panel,pose,error

def qualified(data,panel,pose,error,lam=None):
    return geometry.corner_gaps(data['x'][None,:],panel,pose,multipliers={0:0.} if lam is None else lam,
        query_position_errors=error,return_audit=True)

@pytest.mark.parametrize("fixture",[FIXTURE,FIXTURE.parent/"s11_unload_faces44_45_separated_tie.npz"])
def test_actual_unload_ambiguous_projection_has_only_certified_inactive_branch(fixture):
    data,panel,pose,error=captured(fixture); before=copy.deepcopy((data,pose.__dict__))
    with pytest.raises(CornerContactRefusal,match='nonunique|distinct'):
        geometry.closest_panel_point(data['x'],pose.positions,panel.face_slices,
            [f.node_ids for f in panel.faces],pose.director_gradient,faces=data['faces'])
    with pytest.raises(CornerContactRefusal):
        geometry.corner_gaps(data['x'][None,:],panel,pose)
    result,audit=qualified(data,panel,pose,error)
    assert result=={} and len(audit)==1 and audit[0]['corner_id']==0
    proof=audit[0]
    assert proof['gap_positive'] and proof['gap_strictly_greater_than_m']==geometry.ZERO_DISTANCE_M
    assert proof['lambda_N']==proof['ncp_phi']==proof['contact_work']==0.
    assert proof['no_closest_owner_or_gradient'] and len(proof['face_proofs'])==4
    from fractions import Fraction
    assert all(Fraction(r['oriented_side_lower'])>0 for r in proof['face_proofs'])
    for k,v in before[0].items(): np.testing.assert_array_equal(data[k],v)
    for k,v in before[1].items(): np.testing.assert_array_equal(getattr(pose,k),v)

@pytest.mark.parametrize('force',[1.,-1.,np.nan])
def test_nonzero_or_nonfinite_force_retains_original_refusal(force):
    data,panel,pose,error=captured()
    with pytest.raises(CornerContactRefusal): qualified(data,panel,pose,error,{0:force})

def test_unavailable_or_uncertain_separation_retains_original_refusal(monkeypatch):
    data,panel,pose,error=captured()
    with pytest.raises(CornerContactRefusal): qualified(data,panel,pose,np.full_like(error,.001))
    monkeypatch.setattr(geometry,'_face_minimum_bounds',lambda *args:None)
    with pytest.raises(CornerContactRefusal): qualified(data,panel,pose,error)

def test_error_boxes_touching_or_crossing_panel_do_not_certify():
    from anysolver.general_contact.corner_contact_projection_inactive import separated_zero_force_proof
    data,panel,pose,error=captured()
    # Move the query below every captured local skin; the positive-side proof
    # must fail even though its unsigned distance is still strictly positive.
    below=data['x'].copy();below[2]=.003
    assert separated_zero_force_proof(below,np.zeros(3),pose.positions,pose.position_errors,
        panel.face_slices,data['faces'],radius=.05,zero_distance=geometry.ZERO_DISTANCE_M) is None
    at=pose.positions[panel.face_slices[54][0]].copy()
    assert separated_zero_force_proof(at,np.zeros(3),pose.positions,pose.position_errors,
        panel.face_slices,[54],radius=.05,zero_distance=geometry.ZERO_DISTANCE_M) is None

def test_protected_actual_44_45_unique_point_contract_remains_strict():
    path=FIXTURE.parent/'s11_faces_44_45_tie.npz'
    with np.load(path,allow_pickle=False) as z:
        with pytest.raises(CornerContactRefusal) as exc:
            geometry.closest_panel_point(z['x'],z['panel_positions'],z['face_slices'],z['face_nodes'],
                z['panel_director_gradient'],faces=z['faces'])
    assert exc.value.code=='nonunique_closest_point'

@pytest.mark.parametrize("fixture",[FIXTURE,FIXTURE.parent/"s11_unload_faces44_45_separated_tie.npz"])
def test_evaluate_production_path_retains_proof_and_zero_force_without_gradient(fixture):
    from scipy import sparse
    data,panel,pose,error=captured(fixture)
    sphere_pose=SimpleNamespace(positions=data['x'][None,:],position_errors=error)
    panel.pose=lambda u:pose
    sim=SimpleNamespace(n_s=3,total=3,sphere_map=SimpleNamespace(pose=lambda u:sphere_pose),panel_map=panel,
        _split=lambda u:(u[:3],u[3:]),guide_rows_full=np.zeros((6,3)),
        _trial_target=np.zeros(6),f_ext=np.zeros(3),
        timing=dict(evaluations=0,structural_s=0.,geometry_s=0.,tangent_reduction_s=0.))
    sim.bridge=SimpleNamespace(_owner=object(),expand=lambda q:q.copy(),
        response=lambda u,tangent:(np.zeros(3),[None,None],({},{})),
        block_jacobian=lambda u:sparse.eye(3,format='csr'),_check_models=lambda:None)
    sim._contact_geometry=MethodType(CoupledIndentation._contact_geometry,sim)
    # Default route remains the original strict three-tuple API and refuses.
    with pytest.raises(CornerContactRefusal):sim._contact_geometry(np.zeros(3))
    ev=CoupledIndentation.evaluate(sim,np.zeros(3),np.zeros(6),{0:0.},tangent=False)
    assert ev.ids==[] and ev.gaps.size==0 and ev.gradients.shape==(0,3)
    assert ev.extra['inactive_corner_proofs'][0]['corner_id']==0
    np.testing.assert_array_equal(ev.extra['f_contact'],0.)
    np.testing.assert_array_equal(ev.residual,0.)
    with pytest.raises(CornerContactRefusal):
        CoupledIndentation.evaluate(sim,np.zeros(3),np.zeros(6),{0:1.},tangent=False)

def test_unambiguous_gap_work_gradient_is_unchanged():
    positions=np.array([[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]])
    x=np.array([.1,.2,.01]);slices=((0,4),);nodes=((0,1,2,3),);gradient=np.zeros((4,3,3))
    info=geometry.closest_panel_point(x,positions,slices,nodes,gradient)
    h=1e-6
    for axis in range(3):
        d=np.eye(3)[axis]*h
        plus=geometry.closest_panel_point(x+d,positions,slices,nodes,gradient)['gap']
        minus=geometry.closest_panel_point(x-d,positions,slices,nodes,gradient)['gap']
        assert (plus-minus)/(2*h)==pytest.approx(info['direction'][axis],abs=1e-8)

def test_hidden_closer_omitted_face_cannot_be_excluded_by_float_broadphase_assumption():
    from anysolver.general_contact.corner_contact_projection_inactive import separated_zero_force_proof
    first=np.array([[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]])
    second=first.copy();second[:,2]=.009
    positions=np.vstack([first,second]);errors=np.zeros_like(positions)
    # The passed face enumeration intentionally omits a closer patch; an exact
    # omitted-box lower must defeat the otherwise-positive first-face proof.
    assert separated_zero_force_proof(np.array([.1,.2,.01]),np.zeros(3),positions,errors,
        ((0,4),(4,8)),[0],radius=.05,zero_distance=geometry.ZERO_DISTANCE_M) is None
    # An actually distant omitted face gives a recorded exact lower bound.
    second[:,2]=1.;positions=np.vstack([first,second])
    proof=separated_zero_force_proof(np.array([.1,.2,.01]),np.zeros(3),positions,errors,
        ((0,4),(4,8)),[0],radius=.05,zero_distance=geometry.ZERO_DISTANCE_M)
    assert proof is not None and len(proof['omitted_face_proofs'])==1
    from fractions import Fraction
    assert Fraction(proof['omitted_face_proofs'][0]['physical_distance_lower_m'])>Fraction(.05)

@pytest.mark.parametrize('radius,zero',[(np.inf,1e-9),(.05,np.nan),(-1.,1e-9),(.05,.05),(.05,0.)])
def test_invalid_proof_thresholds_refuse(radius,zero):
    from anysolver.general_contact.corner_contact_projection_inactive import separated_zero_force_proof
    data,panel,pose,error=captured()
    assert separated_zero_force_proof(data['x'],error[0],pose.positions,pose.position_errors,
        panel.face_slices,data['faces'],radius=radius,zero_distance=zero) is None

def test_nonfinite_or_wrong_shape_positions_do_not_certify():
    from anysolver.general_contact.corner_contact_projection_inactive import separated_zero_force_proof
    data,panel,pose,error=captured()
    for positions in (np.zeros(3),np.zeros((4,2)),np.full_like(pose.positions,np.nan)):
        assert separated_zero_force_proof(data['x'],error[0],positions,np.zeros_like(positions),
            panel.face_slices,data['faces'],radius=.05,zero_distance=geometry.ZERO_DISTANCE_M) is None
