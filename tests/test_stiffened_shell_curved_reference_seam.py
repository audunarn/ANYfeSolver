"""Tilted-facet and point-pose foundation; public curved admission stays closed."""
import numpy as np
from scipy.spatial.transform import Rotation
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import create_shell_element
from anyfesolver._stiffened_shell_q4_algorithmic_pullback import response,local_coordinates
from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from docs.reference_cases.stiffened_shell_e04_roof import build
import pytest


def facet():
    roof=build(8,8);nodes=roof.quads[2];xyz=roof.shell_positions[nodes]
    model=FEModel('curved-reference-local-seam')
    model.add_material('shell',roof.young,roof.poisson,density=1.)
    for i,p in enumerate(xyz,1):model.add_node(i,*p)
    normal=np.cross(xyz[1]-xyz[0],xyz[3]-xyz[0]);normal/=np.linalg.norm(normal)
    element=create_shell_element(1,(1,2,3,4),'shell',formulation='e4-pl',
        thickness=roof.thickness,reference_normal=tuple(normal))
    model.add_element(1,element)
    return model,element,xyz,element.init_nonlinear_state(3)


def test_tilted_reference_zero_strain_and_finite_rigid_motion():
    model,e,xyz,origin=facet();zero=np.zeros(24)
    f,k,state=response(model,e,zero,origin,3)
    assert np.linalg.norm(f)<1e-12
    np.testing.assert_allclose(local_coordinates(model,e,zero),0.,atol=1e-15)
    motion=Rotation.from_rotvec([.4,-.3,.2]);u=np.zeros((4,6))
    u[:,:3]=motion.apply(xyz)+[.1,-.2,.3]-xyz;u[:,3:]=motion.as_rotvec()
    np.testing.assert_allclose(local_coordinates(model,e,u.ravel()),0.,atol=2e-15)
    rotated,_,state=response(model,e,u.ravel(),origin,3)
    assert np.linalg.norm(rotated)/max(1.,np.linalg.norm(k))<1e-12


def test_tilted_force_map_actual_chart_derivative_and_objectivity():
    model,e,xyz,origin=facet()
    u=(np.sin(np.arange(24)+.1)*1e-5).reshape(4,6)
    motion=Rotation.from_rotvec([.4,-.3,.2]);moved=u.copy()
    moved[:,:3]=motion.apply(xyz+u[:,:3])+[.1,-.2,.3]-xyz
    moved[:,3:]=(motion*Rotation.from_rotvec(u[:,3:])).as_rotvec()
    f,_,_=response(model,e,u.ravel(),origin,3)
    g,k,_=response(model,e,moved.ravel(),origin,3)
    np.testing.assert_allclose(g.reshape(-1,3),motion.apply(f.reshape(-1,3)),rtol=1e-8,atol=1e-7)
    direction=np.cos(np.arange(24));direction/=np.linalg.norm(direction)
    predicted=k@direction
    for h in (1e-6,3e-7):
        plus=response(model,e,moved.ravel()+h*direction,origin,3)[0]
        minus=response(model,e,moved.ravel()-h*direction,origin,3)[0]
        assert np.linalg.norm((plus-minus)/(2*h)-predicted)/np.linalg.norm(predicted)<1e-6


def test_curved_point_reference_pose_is_incremental_not_facet_normal():
    c=build(8,8);s,b=c.point_pairs[3]
    positions=np.array([c.shell_positions[s],c.beam_positions[b]])
    # Shell Q is a global incremental physical rotation. Q*n_ref transports
    # each adjacent facet director; Q=I does not assert that n_ref is global z.
    frames=np.array([np.eye(3),c.beam_frames[b]])
    joint=RigidPoseJoint(positions,frames)
    multipliers=np.array([11.,-7.,3.,.4,-.2,.1])
    original=joint.evaluate(positions,frames,multipliers)
    motion=Rotation.from_rotvec([.4,-.3,.2]).as_matrix()
    moved=joint.evaluate(positions@motion.T+[.1,-.2,.3],motion@frames,multipliers)
    np.testing.assert_allclose(original.constraints,0.,atol=1e-14)
    np.testing.assert_allclose(moved.constraints,0.,atol=1e-14)
    np.testing.assert_allclose(moved.residual[:12].reshape(-1,3),
        original.residual[:12].reshape(-1,3)@motion.T,rtol=1e-12,atol=1e-12)
    wrench=original.residual[:12].reshape(2,6)
    np.testing.assert_allclose(wrench[:,:3].sum(axis=0),0.,atol=1e-12)
    moment=wrench[:,3:].sum(axis=0)+np.cross(positions,wrench[:,:3]).sum(axis=0)
    np.testing.assert_allclose(moment,0.,atol=1e-12)


def test_curved_policy_rejects_physical_fibres():
    from dataclasses import replace
    from anyfesolver import stiffened_shell as ss
    from test_stiffened_shell_ge_physical_fibre_linear import _definition
    definition = replace(_definition(),
        q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
        shell_reference_policy=ss.CURVED_Q4_POINT_REFERENCE_V1)
    with pytest.raises(ValueError, match='generalized elastic beam'):
        ss.create_analysis(definition=definition)


def curved_panel():
    from dataclasses import replace
    from anyfesolver import b3_ge,stiffened_shell as ss
    from test_stiffened_shell_ge_experimental import _panel
    base=_panel();xyz=np.array(base.shell.coordinates_m,copy=True)
    xyz[:,2]+=.02*xyz[:,0]**2
    beams=[]
    for old in base.beams:
        element,inertia=old.instantiate()
        p=element.operator.reference.coordinates.copy();p[:,2]+=.02*p[:,0]**2
        frames=Rotation.from_rotvec(np.column_stack((np.zeros(3),-np.arctan(.04*p[:,0]),np.zeros(3)))).as_matrix()
        beams.append(b3_ge.define_beam('b3-ge',element.element_id,element.node_ids,p,frames,
            element.section,inertia,quadrature=element.operator.order))
    return replace(base,shell=replace(base.shell,coordinates_m=xyz),beams=tuple(beams),
        q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
        shell_reference_policy=ss.CURVED_Q4_POINT_REFERENCE_V1)


@pytest.mark.parametrize('force',(1.,100.))
@pytest.mark.parametrize('curved',(False,True))
def test_explicit_curved_point_owner_reference_finite_restart(request,force,curved,tmp_path):
    from dataclasses import replace
    from anyfesolver import stiffened_shell as ss
    definition=curved_panel()
    if not curved:
        from test_stiffened_shell_ge_experimental import _panel
        definition=replace(_panel(),q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    def make(d=definition):
        a=ss.create_analysis(definition=d);request.addfinalizer(a.close);return a
    if curved:
        with pytest.raises(ValueError,match='flat'):make(replace(definition,shell_reference_policy='flat'))
    owner=make();linear=make()
    if curved:
        assert not owner.reference_modal_capabilities['eligible']
        assert owner._definition.shell_reference_policy==ss.CURVED_Q4_POINT_REFERENCE_V1
    assert owner.preflight()['stress_free_scaled_max']<=1e-10
    action=(ss.NodalAction('beam',105,(0.,0.,-force,0.,0.,0.)),)
    result=linear.solve_linear(action)
    assert result.free_equilibrium_ratio<=1e-8
    from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
    from dataclasses import asdict
    try:a=owner.solve(action,(.1,.2))
    except FiniteIncrementFailure as error:
        import json
        (tmp_path/'rejected.json').write_text(json.dumps(asdict(error.attempt),indent=2))
        # Retain the observed tiny-load failure as a non-success result, never
        # an accepted equilibrium. A later numerical improvement may solve it.
        if not(curved and force==1.):raise
        assert owner._generation==0
        with pytest.raises(RuntimeError,match='no accepted'):owner.recover()
        return
    assert a.free_equilibrium_ratio<=1e-8 and a.constraint_max<=1e-9
    raw=owner.checkpoint();restored=make();restored.restore(raw)
    assert restored.checkpoint()==raw
    a=owner.solve(action,(.3,));b=restored.solve(action,(.3,))
    np.testing.assert_array_equal(a.cell_resultants,b.cell_resultants)
    assert owner.checkpoint()==restored.checkpoint()
    if curved:
        import json
        raw_linear=json.loads(linear.checkpoint())
        raw_linear['load'][3]=1.
        with pytest.raises(ValueError,match='translational load'):
            make().restore(json.dumps(raw_linear).encode())
        from anyfesolver._stiffened_shell_ge_body_load import MAGIC
        with pytest.raises(ValueError,match='external load checkpoints'):make().restore(MAGIC+b'invalid')
        assert not owner.capabilities['body_acceleration']
        assert not owner.capabilities['shell_pressure_loads']
        with pytest.raises(ValueError,match='spectra'):make().modes(None)
        with pytest.raises(ValueError,match='translations'):
            make().solve((ss.NodalAction('beam',105,(0.,0.,0.,0.,1.,0.)),))
