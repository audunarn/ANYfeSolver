"""Original Log supports and actual native update; ordinary development only."""
from dataclasses import replace
from pathlib import Path
import json,os
import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.transform import Rotation
from anyfesolver import stiffened_shell as api
from anyfesolver import _stiffened_shell_coordinate_support as support
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_chart,encode_chart
from anyfesolver.control import CancellationToken,SolveCancelled
from test_stiffened_shell_ge_experimental import _panel
from test_stiffened_shell_ge_finite_experimental import _actions
from test_stiffened_shell_ge_owned_rebase import _poses


def definition(s3=False,policy=support.POLICY):
    d=replace(_panel(),q4_pullback_policy=api.Q4_ALGORITHMIC_PULLBACK_V1,shell_support_policy=policy)
    if s3:d=replace(d,q4_pullback_policy='historical_corotational',shell=replace(d.shell,q4_elements=(),s3_elements=((1,2,5),(1,5,4),(2,3,6),(2,6,5))))
    return replace(d,fixed=d.fixed+(api.FixedCoordinate('shell',2,3),))


@pytest.fixture(scope='module',params=(False,True))
def loaded(request):
    d=definition(request.param);o=api.create_analysis(definition=d)
    try:o.solve(_actions(),(.5,1.));yield d,o.checkpoint()
    finally:o.close()


def restored(loaded):
    o=api.create_analysis(definition=loaded[0]);o.restore(loaded[1]);return o


def observe(o):
    r=o.recover();xyz=np.asarray(o._definition.shell.coordinates_m)+r.shell_displacement.reshape(-1,6)[:,:3]
    rows=r.support_reactions[:o._shell_count].reshape(-1,6);f=np.array([0.,0.,1000*r.load_factor])
    force=rows[:,:3].sum(0)+f
    moment=(np.cross(xyz,rows[:,:3])+rows[:,3:]).sum(0)+np.cross(r.beam_positions[o._beam_index[105]],f)
    return dict(force_N=force.tolist(),moment_Nm=moment.tolist(),force_ratio=float(np.linalg.norm(force)/1000),
        moment_ratio=float(np.linalg.norm(moment)/2000),free_equilibrium_ratio=float(r.free_equilibrium_ratio),gap=float(r.constraint_max))


def compare(a,b):
    observations=[observe(a),observe(b)]
    if os.environ.get('GE_SUPPORT_OBSERVATIONS'):
        p=Path(os.environ['GE_SUPPORT_OBSERVATIONS']);old=json.loads(p.read_text()) if p.exists() else []
        old.append(dict(s3=bool(a._definition.shell.s3_elements),branches=observations));p.write_text(json.dumps(old,indent=2))
    assert all(x['force_ratio']<=1e-8 and x['moment_ratio']<=1e-8 and x['free_equilibrium_ratio']<=1e-8 and x['gap']<=1e-9 for x in observations),observations
    np.testing.assert_allclose(_poses(a),_poses(b),rtol=0,atol=1e-10)
    for name in ('beam_positions','beam_frames','cell_frames','cell_resultants','multipliers'):
        np.testing.assert_allclose(getattr(a.recover(),name),getattr(b.recover(),name),rtol=1e-8,atol=1e-6)
    ra,rb=a.recover(),b.recover()
    np.testing.assert_allclose(ra.shell_displacement.reshape(-1,6)[:,:3],rb.shell_displacement.reshape(-1,6)[:,:3],rtol=1e-8,atol=1e-10)
    assert np.max(np.abs(ra.support_reactions-rb.support_reactions)/np.tile([1000]*3+[2000]*3,len(ra.support_reactions)//6))<=1e-8
    assert len(ra.point_wrenches)==len(rb.point_wrenches)>0
    for x,y in zip(ra.point_wrenches,rb.point_wrenches):
        assert x.point_index==y.point_index
        for field in ('shell_action','beam_action'):
            assert np.max(np.abs(np.asarray(getattr(x,field))-getattr(y,field))/np.array([1000]*3+[2000]*3))<=1e-8
        assert np.max(np.abs(np.asarray(x.gap)-y.gap)/[2,2,2,1,1,1])<=1e-9


def test_public_full_and_selective_rebase_restore_continue(loaded):
    o,c=restored(loaded),restored(loaded);new=None
    try:
        if loaded[0].shell.s3_elements:
            assert o._finite_driver._stable_s3 is True
            assert c._finite_driver._stable_s3 is True
        before=o._finite_driver._state;origins=o._finite_driver._shell_origins,o._finite_driver._beam_origins
        o.rebase_rotations(o.recover().generation,preserve_supported_charts=True);compare(o,c)
        # Full event now transports the original support equations, including partial support.
        o.rebase_rotations(o.recover().generation);compare(o,c)
        r=o.recover();np.testing.assert_array_equal(r.shell_chart_coordinates.reshape(-1,6)[:,3:],0.)
        assert np.linalg.norm(r.shell_rotation_anchors[1]-np.eye(3))>0
        assert o._finite_driver._shell_origins is origins[0] and o._finite_driver._beam_origins is origins[1]
        for field in ('frames','cells','resultants','multipliers'):assert getattr(o._finite_driver._state,field) is getattr(before,field)
        new=api.create_analysis(definition=loaded[0]);new.restore(o.checkpoint());compare(o,new)
        if loaded[0].shell.s3_elements:
            assert new._finite_driver._stable_s3 is True
        for x in (new,c):x.solve(_actions(),(1.1,))
        compare(new,c)
        assert support.support_gap(new._finite_driver._state.q,new._finite_driver._state.shell_anchors,new._shell_count,new._fixed)<=1e-9
        # Actual physical reaction is dual to original psi, not masked spatial axes.
        state=new._finite_driver._state
        for slots,psi,J,_,_,mask in support.blocks(state.q,state.shell_anchors,new._shell_count,new._fixed):
            if not mask.any():continue
            m=new.recover().support_reactions[slots];eta=J.T@m
            assert np.max(np.abs(eta[~mask]),initial=0.)/2000<=1e-8
            v=np.array([.2,-.3,.4]);assert abs(m@(J@v)-eta@v)/2000<=1e-8
    finally:
        o.close();c.close()
        if new:new.close()


def test_actual_update_directional_tangent_with_nonzero_support_reaction(loaded):
    o=restored(loaded)
    try:
        o.rebase_rotations(o.recover().generation);d=o._finite_driver;state=d._state
        r,K,*_=d._assemble(state,d._load_factor,d._shell_origins,d._beam_origins)
        assert abs(r[9])>1 and np.linalg.norm(o.recover().support_reactions[9:12])>1
        vectors={name:np.zeros(len(r)) for name in ('shell','beam_translation','retained_force','multiplier')}
        vectors['shell'][10:12]=(.3,-.2);vectors['shell'][16:18]=(.1,.2)
        vectors['beam_translation'][o._shell_count+6*o._beam_index[105]+2]=2.
        vectors['retained_force'][o._shell_count+d._beam_nodal+6]=1000.
        vectors['multiplier'][o._K.shape[0]+3]=2000.
        errors={}
        for name,v in vectors.items():
            analytic=K@v;plateau=[]
            # Mechanical rows have force/moment scales; constraint rows have length/angle scales.
            row_scales=np.r_[2000/o._coordinate_scale,o._constraint_scale]
            for h in (3e-5,1e-5,3e-6):
                plus=d._advance(state,v,h);minus=d._advance(state,v,-h)
                rp=d._assemble(plus,d._load_factor,d._shell_origins,d._beam_origins)[0]
                rm=d._assemble(minus,d._load_factor,d._shell_origins,d._beam_origins)[0]
                error=((rp-rm)/(2*h)-analytic)/row_scales;reference=analytic/row_scales
                # Separate mechanics/constraints; fixed unit coefficient action norm for zeroexpected blocks.
                n=o._K.shape[0]
                plateau.append(max(float(np.linalg.norm(error[:n])/max(np.linalg.norm(reference[:n]),1.)),
                    float(np.linalg.norm(error[n:])/max(np.linalg.norm(reference[n:]),1.))))
            errors[name]=plateau
        if os.environ.get('GE_SUPPORT_OBSERVATIONS'):
            p=Path(os.environ['GE_SUPPORT_OBSERVATIONS']);x=json.loads(p.read_text());x.append(dict(tangent_plateau=errors,s3=bool(o._definition.shell.s3_elements)));p.write_text(json.dumps(x,indent=2))
        assert all(max(x)<=1e-6 for x in errors.values()),errors
    finally:o.close()


def test_local_counterexample_dual_and_complete_residual_curvature():
    q=np.zeros(6);q[4]=.4;fixed=np.array([3]);_,J,_=_exp_terms(q[3:])
    spatial=np.array([0,0,0,2.,0,0]);assert abs((J.T@spatial[3:])[2])>.1
    r,K,f,g,*_=support.pullback(q,None,6,fixed,spatial,sparse.csc_matrix((6,6)),np.zeros(6))
    assert sparse.issparse(K) and np.linalg.norm(K.data)>0 and g==0
    m=support.reactions(q,None,6,fixed,np.array([0,0,0,2.,0,0]),6)[3:]
    eta=J.T@m;np.testing.assert_allclose(eta,[2,0,0],rtol=0,atol=1e-14)
    assert abs(m[2])>.1
    for v in (np.array([0,.3,.7]),np.array([.2,-.4,.1])):
        assert abs(m@(J@v)-eta@v)<1e-14


def test_physical_equilibrium_guard_is_not_a_chart_covector_norm():
    from anyfesolver._stiffened_shell_ge_experimental import _linear_equilibrium
    q=np.zeros(6);q[3]=np.pi-2e-7;fixed=np.array([],dtype=int)
    applied=np.array([0,0,1.,0,0,0]);spatial=np.array([0,0,0,0,2.2e-8,0])
    r,_,f,_,physical,physical_f=support.pullback(q,None,6,fixed,spatial,sparse.csc_matrix((6,6)),applied)
    scales=np.array([2,2,2,1,1,1])
    assert _linear_equilibrium(r,f,scales)<=1e-8
    assert _linear_equilibrium(physical,physical_f,scales)>1e-8


def test_stale_cancel_restore_tamper_and_policy_identity(loaded):
    o=restored(loaded);fresh=api.create_analysis(definition=loaded[0])
    try:
        raw=o.checkpoint();gen=o.recover().generation
        with pytest.raises(ValueError,match='stale'):o.rebase_rotations(gen-1)
        class Cancel(CancellationToken):
            def raise_if_cancelled(self,stage=''):
                if stage=='stiffened-shell.rebase.commit':self.cancel('owned cancellation')
                super().raise_if_cancelled(stage)
        with pytest.raises(SolveCancelled):o.rebase_rotations(gen,cancellation_token=Cancel())
        assert o.checkpoint()==raw
        o.rebase_rotations(gen);g,A,W,inner=decode_chart(o.checkpoint(),o._identity,len(o._shell_ids))
        A=A.copy();W=W.copy();A[1]=Rotation.from_rotvec([.01,.02,0]).as_matrix();W[1]=A[1]
        with pytest.raises(ValueError,match='Log support'):fresh.restore(encode_chart(o._identity,g,A,W,inner))
        assert fresh._generation==0 and fresh._finite_driver is None
        legacy=api.create_analysis(definition=replace(loaded[0],shell_support_policy=support.LEGACY))
        try:
            assert legacy._identity!=o._identity
            with pytest.raises(ValueError):legacy.restore(o.checkpoint())
        finally:legacy.close()
    finally:o.close();fresh.close()


@pytest.mark.parametrize('change',({'shell_support_policy':None},{'shell_support_policy':1},{'q4_pullback_policy':'historical_corotational'},
    {'lines':(object(),)},{'modal_boundary_policy':'free_free_modal_only_v1'},
    {'fixed':(api.FixedCoordinate('beam',101,3),)},{'shell':None},{'fixed':(object(),)}))
def test_admission_refuses_unverified_combinations(change):
    with pytest.raises(ValueError):api.create_analysis(definition=replace(definition(),**change))


def test_operation_and_private_context_refusal_before_work(monkeypatch):
    o=api.create_analysis(definition=definition())
    try:
        for call in (lambda:o.solve_linear(_actions()),lambda:o.modes(object()),lambda:o.buckling(object()),
                lambda:o.solve(_actions(),increment_control=object()),lambda:o.solve(_actions(),displacement_control=object()),
                lambda:o.solve((api.NodalAction('beam',105,(0,0,1,1,0,0)),))):
            with pytest.raises(ValueError):call()
        from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
        o._external_context=object()
        with pytest.raises(ValueError,match='contexts'):FinitePointDriver(o)
    finally:o.close()


def test_branch_and_legacy_default_identity():
    with pytest.raises(ValueError,match='principal'):support.principal(Rotation.from_rotvec([np.pi,0,0]).as_matrix())
    q=np.zeros(6);q[3]=2*np.pi-1e-8
    with pytest.raises(ValueError,match='local principal'):support.blocks(q,None,6,np.array([],dtype=int))
    a=api.create_analysis(definition=_panel());b=api.create_analysis(definition=replace(_panel(),shell_support_policy=support.LEGACY))
    try:assert a._identity==b._identity and a._definition.shell_support_policy==support.LEGACY
    finally:a.close();b.close()


@pytest.mark.parametrize('s3',(False,True))
def test_capabilities_describe_only_admitted_coordinate_support_operations(s3):
    owner=api.create_analysis(definition=definition(s3))
    try:
        cap=owner.capabilities
        assert cap['status']=='EXPERIMENTAL_NOT_QUALIFIED'
        assert cap['shell_support_policy']==support.POLICY
        assert cap['analysis']==('experimental_elastic_finite_point',)
        assert cap['increment_control']==('fixed',)
        assert 'elastic response only' in cap['beam']
        assert 'original principal Log supports' in cap['rotation_chart']
        assert cap['zero_load_normalization']==()
        for key in ('body_acceleration','beam_distributed_loads','staged_beam_distributed_loads','shell_pressure_loads','staged_dead_loads'):
            assert cap[key]==()
        for key in ('displacement_control','arc_length'):
            assert cap[key].startswith('not admitted')
    finally:owner.close()


@pytest.mark.parametrize('s3',(False,True))
def test_historical_capability_inventory_unchanged(s3):
    owner=api.create_analysis(definition=definition(s3,policy=support.LEGACY))
    try:
        cap=owner.capabilities
        assert 'reference_linear' in cap['analysis']
        assert 'experimental_generalized_plastic_finite_point' in cap['analysis']
        assert cap['increment_control']==('fixed','experimental_adaptive_halving')
        assert 'shell_support_policy' not in cap
    finally:owner.close()

