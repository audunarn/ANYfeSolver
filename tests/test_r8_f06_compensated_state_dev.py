"""Root-supervised actual state/derivative/lifecycle development controls."""
import os
from pathlib import Path
import numpy as np
import pytest
pytestmark=pytest.mark.skipif('F06_COMPENSATED_OUTPUT' not in os.environ,reason='bounded supervisor only')

def test_control_error_free_owned_accumulation_and_refusal():
    from anyfesolver._stiffened_shell_f06_increment_arithmetic import Pair,add_state,validate_frames,validate_pairs,Jet,log
    high=np.array([1.]);low=np.zeros(1);delta=np.array([2.**-55])
    h,l=add_state(high,low,delta)
    assert h[0]==1. and l[0]==delta[0] and high[0]==1. and low[0]==0.
    assert (Pair(h,l)-1.)[0].scalar()==2.**-55
    with pytest.raises(ValueError):validate_pairs(np.ones(1),np.ones(1),'unnormalized')
    with pytest.raises(ValueError):validate_frames(np.eye(3)[None],np.diag([0.,0.,.1])[None],'bad physical frame')
    with pytest.raises(ValueError):Pair(np.inf)
    a=[[Jet.constant(float(i==j)*.8,1) for j in range(3)] for i in range(3)]
    with pytest.raises(ValueError,match='near-identity'):log(a)

def test_control_direct_material_resultant_boundary():
    from scripts.r8_f06_compensated_state_route_dev import definition
    e,_=definition().beams[0].instantiate();cell=e.operator.cell;origin=cell.virgin()
    for p,low in (([0.]*19,[0.]*18),([0.]*18,[0.]*17),([0.]*18,[float('nan')]+[0.]*17)):
        with pytest.raises(ValueError):cell.response(p,origin,resultants_low=low)
    assert cell.virgin()==origin

def test_control_point_transpose_work_and_moving_rows():
    from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
    from anyfesolver._stiffened_shell_f06_increment_arithmetic import updated_rotation
    joint=RigidPoseJoint(np.array([[0.,0.,0.],[1.,0.,0.]]),np.tile(np.eye(3),(2,1,1)))
    x=np.array([[0.,0.,0.],[1.,0.,0.]])
    low=np.zeros((2,3));low[1,1]=2.**-55
    frames=np.tile(np.eye(3),(2,1,1));fl=np.zeros_like(frames)
    multipliers=np.array([0.,2.,0.,0.,0.,0.])
    r=joint.evaluate(x,frames,multipliers,position_low=low,frames_low=fl)
    assert r.constraints[1]==2.**-55 and r.residual[5]==-2.
    delta=np.zeros(18);delta[5]=1.;h=1e-6
    endpoints=[]
    for sign in (-1,1):
        q,ql=updated_rotation(frames[0],fl[0],sign*h*delta[3:6]);qs=frames.copy();ls=fl.copy();qs[0]=q;ls[0]=ql
        endpoints.append(joint.evaluate(x,qs,multipliers,position_low=low,frames_low=ls).residual)
    observed=(endpoints[1]-endpoints[0])/(2*h)
    np.testing.assert_allclose(observed,r.spatial_jacobian@delta,rtol=1e-8,atol=1e-10)
    assert np.linalg.norm(r.chart_hessian-r.spatial_jacobian)>0
    v=np.arange(12,dtype=float)/13.;mu=np.arange(6,dtype=float)/7.
    assert abs(float(mu@(r.constraint_jacobian@v))-float(v@(r.constraint_jacobian.T@mu)))<1e-14

def test_actual_original_full48_updater_and_evaluator():
    from scripts.r8_f06_compensated_state_route_dev import actual
    actual(Path(os.environ['F06_COMPENSATED_OUTPUT']))

def test_lifecycle_public_accepted_correction_cancel_rebase_and_cold_restore():
    from scripts.r8_f06_compensated_state_route_dev import owner,definition,SAVED
    from anyfesolver import stiffened_shell as ss
    from test_stiffened_shell_ge_finite_experimental import _actions
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import COMPENSATED_MAGIC
    o=owner();d=o._finite_driver
    try:
        old=o.checkpoint()
        with pytest.raises(ValueError,match='legacy checkpoint'):
            fresh=ss.create_analysis(definition=definition())
            try:fresh.restore((SAVED/'accepted-before.bin').read_bytes())
            finally:fresh.close()
        result=o.solve(_actions(),(.050001,),max_iterations=12,max_backtracks=8)
        state=d._state
        assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
        assert np.count_nonzero(state.frames_low)+np.count_nonzero(state.q_low)>0
        saved=o.checkpoint();assert saved.startswith(COMPENSATED_MAGIC)
        # An immediate endpoint evaluation uses committed origins; constitutive
        # branch/history identity across generations is intentionally not asserted.
        immediate=d._assemble(state,d._load_factor,d._shell_origins,d._beam_origins)
        assert immediate[2]<=1e-8 and immediate[3]<=1e-9
        restored=ss.create_analysis(definition=definition())
        try:
            restored.restore(saved)
            assert restored.checkpoint()==saved
            rs=restored._finite_driver._state
            for name in ('q','frames','cells','resultants','q_low','frames_low','cells_low','resultants_low'):
                np.testing.assert_array_equal(getattr(state,name),getattr(rs,name))
        finally:restored.close()
        correction={n:getattr(state,n).copy() for n in ('q_low','frames_low','cells_low','resultants_low')}
        o.rebase_rotations(o._generation)
        for n,v in correction.items():np.testing.assert_array_equal(getattr(d._state,n),v)
        raw=o.checkpoint()
        rebased=ss.create_analysis(definition=definition())
        try:
            rebased.restore(raw);assert rebased.checkpoint()==raw
        finally:rebased.close()
        from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated,encode_compensated
        shapes=(('q_low',state.q.shape),('frames_low',state.frames.shape),('cells_low',state.cells.shape),('resultants_low',state.resultants.shape))
        inner,lows=decode_compensated(saved,o._identity,shapes)
        lows['q_low'][o._shell_count]=1.
        invalid=encode_compensated(inner,o._identity,lows)
        fresh=ss.create_analysis(definition=definition())
        try:
            with pytest.raises(ValueError,match='normalized'):fresh.restore(invalid)
            assert fresh._finite_driver is None and fresh._generation==0
            with pytest.raises(ValueError,match='integrity'):fresh.restore(saved[:-1]+bytes([saved[-1]^1]))
            with pytest.raises(ValueError,match='spectra'):fresh.modes(None)
            with pytest.raises(ValueError,match='controllers'):fresh.solve(_actions(),(.1,),increment_control=object())
        finally:fresh.close()
        from anyfesolver.control import CancellationToken,SolveCancelled
        class CancelAfterCandidate(CancellationToken):
            calls=0
            def raise_if_cancelled(self,stage=''):
                if stage=='stiffened-shell.finite.trial':
                    self.calls+=1
                    if self.calls==2:self.cancel('cancel after evaluated provisional candidate')
                super().raise_if_cancelled(stage)
        cancel=CancelAfterCandidate()
        with pytest.raises(SolveCancelled):o.solve(_actions(),(.050002,),cancellation_token=cancel)
        assert cancel.calls==2
        assert o.checkpoint()==raw
        out=Path(os.environ['F06_COMPENSATED_OUTPUT']);(out/'accepted-compensated.bin').write_bytes(saved)
        (out/'lifecycle.txt').write_text('accepted correction, immediate equilibrium, exact cold restore, rebase, cancellation passed\n')
    finally:o.close()


def test_legacy_affected_definition_default_pose_retained_and_checkpoint_controls():
    from scripts.r8_f06_quiet_caller_repair_dev import definition as original
    from scripts.r8_f06_compensated_state_route_dev import definition,SAVED
    from anyfesolver import stiffened_shell as ss
    from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition,STABLE_SO3_SCHEMA,COMPENSATED_SCHEMA
    from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
    from anyfesolver._ge_beam3_centered_so3_strain import POLICY as STABLE_POLICY
    from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
    import json
    default=original().beams[0];element,inertia=default.instantiate();op=element.operator
    assert op.arithmetic_policy is None
    assert NativeBeamDefinition.capture(element,inertia).raw==default.raw
    for policy,schema in ((STABLE_POLICY,STABLE_SO3_SCHEMA),
            (__import__('anyfesolver._stiffened_shell_f06_increment_arithmetic',fromlist=['POLICY']).POLICY,COMPENSATED_SCHEMA)):
        made=NativeGeneralizedStaticElement(element.element_id,element.node_ids,op.reference,op.section,
            order=op.order,arithmetic_policy=policy)
        capture=NativeBeamDefinition.capture(made,inertia)
        assert json.loads(capture.raw)['schema']==schema
        rebuilt,_=capture.instantiate();assert rebuilt.identity==made.identity
    native=dict(np.load(SAVED/'before-native-0.npz'))
    legacy=ss.create_analysis(definition=original())
    try:
        raw=(SAVED/'accepted-before.bin').read_bytes();legacy.restore(raw)
        assert legacy.checkpoint()==raw
        # Same accepted predecessor and physical binary base; new state
        # arithmetic agrees with the unchanged default mechanical operator.
        origin=legacy._finite_driver._beam_origins[0]
        oldop=legacy._beam_parts[0][0].operator
        selected,_=definition().beams[0].instantiate()
        args=(native['x'],native['low'],native['frames'],native['cells'],native['p'])
        a=oldop.evaluate(*args,origin=origin);b=selected.operator.evaluate(*args,origin=origin)
        np.testing.assert_allclose(a.residual,b.residual,rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(a.hessian+a.hessian_low,b.hessian+b.hessian_low,rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(a.kinematics,b.kinematics,rtol=1e-10,atol=1e-16)
    finally:legacy.close()
    x=np.array([[0.,0.,0.],[1.,0.,0.]]);q=np.tile(np.eye(3),(2,1,1));joint=RigidPoseJoint(x,q)
    lam=np.array([1.,2.,3.,4.,5.,6.]);low=np.zeros_like(x)
    a=joint.evaluate(x,q,lam,position_low=low)
    b=joint.evaluate(x,q,lam,position_low=low,frames_low=np.zeros_like(q))
    np.testing.assert_array_equal(a.residual,b.residual)
    np.testing.assert_array_equal(a.spatial_jacobian,b.spatial_jacobian)
