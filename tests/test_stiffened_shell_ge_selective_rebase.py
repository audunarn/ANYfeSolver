"""Public supported-chart preservation, without support-equation transport."""
from dataclasses import replace
from types import SimpleNamespace
import json,os
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_ge_experimental import FixedCoordinate
from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_chart,encode_chart
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver._stiffened_shell_q4_algorithmic_pullback import POLICY as Q4_POLICY
from test_stiffened_shell_ge_experimental import _panel
from test_stiffened_shell_ge_finite_experimental import _actions
from test_stiffened_shell_ge_owned_rebase import _poses


@pytest.fixture(scope='module',params=(False,True))
def loaded(request):
    d=_panel()
    if request.param:d=replace(d,shell=replace(d.shell,q4_elements=(),s3_elements=((1,2,5),(1,5,4),(2,3,6),(2,6,5))))
    else:d=replace(d,q4_pullback_policy=Q4_POLICY)
    d=replace(d,fixed=d.fixed+(FixedCoordinate('shell',2,3),))
    owner=stiffened_shell.create_analysis(definition=d)
    try:owner.solve(_actions(),(.5,1.));yield d,owner.checkpoint()
    finally:owner.close()


def owner_from(loaded):
    owner=stiffened_shell.create_analysis(definition=loaded[0]);owner.restore(loaded[1]);return owner


def compare(left,right):
    a,b=left.recover(),right.recover();F=1000.;L=2.;observed=[]
    for label,o,result in (('event',left,a),('control',right,b)):
        rows=result.support_reactions[:o._shell_count].reshape(-1,6)
        # These are spatial rows. Do not pull them back a second time.
        positions=np.asarray(o._definition.shell.coordinates_m)+result.shell_displacement.reshape(-1,6)[:,:3]
        force=rows[:,:3].sum(axis=0)+np.array([0.,0.,F*result.load_factor])
        moment=(np.cross(positions,rows[:,:3])+rows[:,3:]).sum(axis=0)
        moment+=np.cross(result.beam_positions[o._beam_index[105]],np.array([0.,0.,F*result.load_factor]))
        observed.append(dict(label=label,q4_policy=o._definition.q4_pullback_policy,s3=bool(o._definition.shell.s3_elements),
            generation=result.generation,force_N=force.tolist(),moment_Nm=moment.tolist(),
            force_ratio=float(np.linalg.norm(force)/F),moment_ratio=float(np.linalg.norm(moment)/(F*L)),
            force_scale_N=F,moment_scale_Nm=F*L,free_work_ratio=float(result.free_equilibrium_ratio),gap=float(result.constraint_max)))
    if os.environ.get('GE_REBASE_OBSERVATIONS'):
        path=Path(os.environ['GE_REBASE_OBSERVATIONS']);records=json.loads(path.read_text()) if path.exists() else []
        records.append(observed);path.write_text(json.dumps(records,indent=2))
    assert all(x['force_ratio']<=1e-8 and x['moment_ratio']<=1e-8 for x in observed),observed
    np.testing.assert_allclose(_poses(left), _poses(right), rtol=0., atol=1e-10)
    for name in ("beam_positions", "beam_frames", "cell_frames",
                 "cell_resultants", "multipliers"):
        np.testing.assert_allclose(getattr(left.recover(), name), getattr(right.recover(), name),
                                   rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(left.recover().shell_displacement.reshape(-1, 6)[:, :3],
                               right.recover().shell_displacement.reshape(-1, 6)[:, :3],
                               rtol=1e-8, atol=1e-10)
    assert left.recover().accepted_factors == right.recover().accepted_factors
    assert a.free_equilibrium_ratio<=1e-8 and b.free_equilibrium_ratio<=1e-8
    assert a.constraint_max<=1e-9 and b.constraint_max<=1e-9
    scales=np.tile([F]*3+[F*L]*3,len(a.support_reactions)//6)
    assert np.max(np.abs(a.support_reactions-b.support_reactions)/scales)<=1e-8
    assert len(a.point_wrenches)==len(b.point_wrenches)>0
    for actual,expected in zip(a.point_wrenches,b.point_wrenches):
        assert actual.point_index==expected.point_index
        scale=np.array([F]*3+[F*L]*3)
        assert np.max(np.abs(np.asarray(actual.shell_action)-expected.shell_action)/scale)<=1e-8
        assert np.max(np.abs(np.asarray(actual.beam_action)-expected.beam_action)/scale)<=1e-8
        assert np.max(np.abs(np.asarray(actual.gap)-expected.gap)/np.array([L]*3+[1.]*3))<=1e-9


def test_public_selective_event_restore_and_continuation(loaded):
    owner,control=owner_from(loaded),owner_from(loaded)
    resumed=None
    try:
        driver=owner._finite_driver;previous=driver._state;origins=driver._shell_origins,driver._beam_origins
        fixed=set(owner._fixed);supported=np.array([any(6*i+j in fixed for j in (3,4,5)) for i in range(len(owner._shell_ids))])
        q=previous.q[:owner._shell_count].reshape(-1,6)
        assert np.any(q[supported,3:]!=0) and np.any(q[~supported,3:]!=0)
        oldanchors=np.tile(np.eye(3),(len(q),1,1)) if previous.shell_anchors is None else previous.shell_anchors.copy()
        with pytest.raises(ValueError,match='partial supports'):owner.rebase_rotations(owner.recover().generation)
        raw=owner.checkpoint();result=owner.rebase_rotations(owner.recover().generation,preserve_supported_charts=True)
        np.testing.assert_array_equal(result.shell_chart_coordinates.reshape(-1,6)[supported],q[supported])
        np.testing.assert_array_equal(result.shell_rotation_anchors[supported],oldanchors[supported])
        np.testing.assert_array_equal(result.shell_chart_coordinates.reshape(-1,6)[~supported,3:],0.)
        assert driver._shell_origins is origins[0] and driver._beam_origins is origins[1]
        for name in ('frames','cells','resultants','multipliers'):assert getattr(driver._state,name) is getattr(previous,name)
        assert owner.checkpoint()!=raw and result.generation==control.recover().generation+1
        compare(owner,control)
        resumed=stiffened_shell.create_analysis(definition=loaded[0]);resumed.restore(owner.checkpoint());compare(owner,resumed)
        for target in (resumed,control):target.solve(_actions(),(1.1,))
        compare(resumed,control)
    finally:
        owner.close();control.close()
        if resumed:resumed.close()


def test_selective_stale_cancel_and_nonboolean_leave_exact_prefix(loaded):
    owner=owner_from(loaded)
    try:
        raw=owner.checkpoint();generation=owner.recover().generation
        for bad in (None,0,1,np.bool_(True),'yes'):
            with pytest.raises(ValueError,match='strict boolean'):owner.rebase_rotations(generation,preserve_supported_charts=bad)
            assert owner.checkpoint()==raw
        with pytest.raises(ValueError,match='stale'):owner.rebase_rotations(generation-1,preserve_supported_charts=True)
        class AtCommit(CancellationToken):
            def raise_if_cancelled(self,stage=''):
                if stage=='stiffened-shell.rebase.commit':self.cancel('selective cancel')
                super().raise_if_cancelled(stage)
        with pytest.raises(SolveCancelled):owner.rebase_rotations(generation,preserve_supported_charts=True,cancellation_token=AtCommit())
        assert owner.checkpoint()==raw
    finally:owner.close()


def test_supported_anchor_corruption_is_not_new_restore_admission(loaded):
    owner=owner_from(loaded);fresh=None
    try:
        owner.rebase_rotations(owner.recover().generation,preserve_supported_charts=True)
        generation,anchors,witness,inner=decode_chart(owner.checkpoint(),owner._identity,len(owner._shell_ids))
        anchors=anchors.copy();witness=witness.copy();anchors[1]=Rotation.from_rotvec((.001,0.,0.)).as_matrix()
        witness[1]=Rotation.from_rotvec(owner.recover().shell_chart_coordinates.reshape(-1,6)[1,3:].copy()).as_matrix()@anchors[1]
        fresh=stiffened_shell.create_analysis(definition=loaded[0])
        with pytest.raises(ValueError,match='support'):fresh.restore(encode_chart(owner._identity,generation,anchors,witness,inner))
        assert fresh._generation==0
    finally:
        owner.close()
        if fresh:fresh.close()


def test_empty_eligible_event_refuses_before_assembly(loaded,monkeypatch):
    owner=owner_from(loaded)
    try:
        driver=owner._finite_driver;q=driver._state.q.copy();rows=q[:owner._shell_count].reshape(-1,6)
        for i in range(len(rows)):
            if not any(6*i+j in set(owner._fixed) for j in (3,4,5)):rows[i,3:]=0.
        driver._state=replace(driver._state,q=q)
        def forbidden(*a,**k):raise AssertionError('empty event assembled')
        monkeypatch.setattr(driver,'_assemble',forbidden)
        with pytest.raises(ValueError,match='no nonzero'):owner.rebase_rotations(owner.recover().generation,preserve_supported_charts=True)
    finally:owner.close()


@pytest.mark.parametrize('kind',('arc','displacement','adaptive-displacement','autoincrement','line','footprint','staged','external'))
def test_unsupported_dispatchers_refuse_flag_before_work(loaded,monkeypatch,kind):
    owner=owner_from(loaded)
    try:
        if kind=='arc':owner._arc_snapshot=object()
        elif kind=='displacement':owner._displacement_snapshot=object()
        elif kind=='adaptive-displacement':owner._displacement_adaptive=object()
        elif kind=='autoincrement':owner._finite_driver._adaptive_state=SimpleNamespace(completed=True)
        elif kind=='staged':owner._finite_driver._programme=object()
        elif kind=='external':owner._external_context=object()
        else:owner._definition=replace(owner._definition,**{('lines' if kind=='line' else 'footprints'):(object(),)})
        def forbidden(*a,**k):raise AssertionError('excluded dispatcher forwarded flag')
        monkeypatch.setattr(owner._finite_driver,'rebase_rotations',forbidden)
        with pytest.raises(ValueError,match='fixed force-controlled'):owner.rebase_rotations(owner._generation,preserve_supported_charts=True)
    finally:owner.close()


def test_main_capability_metadata_does_not_modify_spectral_delegates(monkeypatch):
    from anyfesolver import _stiffened_shell_ge_reference_modal as modal
    from anyfesolver import _stiffened_shell_ge_point_buckling as buckling
    owner=stiffened_shell.create_analysis(definition=replace(_panel(),q4_pullback_policy=Q4_POLICY))
    calls=[]
    def delegated(target,**kwargs):calls.append((target,kwargs));return {'delegated':kwargs}
    monkeypatch.setattr(modal,'capabilities',delegated);monkeypatch.setattr(buckling,'capabilities',delegated)
    try:
        assert 'supported anchor/coordinates retained exactly' in owner.capabilities['selective_rotation_chart_rebase']
        for field,args in (('reference_modal_capabilities',{}),('point_buckling_capabilities',{}),
                ('line_buckling_capabilities',{'lines':True}),('mixed_buckling_capabilities',{'mixed':True}),
                ('native_line_buckling_capabilities',{'lines':True,'mixed':True,'native_lines':True})):
            assert getattr(owner,field)=={'delegated':args}
        assert len(calls)==5 and all(item[0] is owner for item in calls)
    finally:owner.close()


def test_historical_q4_refuses_opt_in_before_work(monkeypatch):
    owner=stiffened_shell.create_analysis(definition=_panel())
    try:
        def forbidden(*a,**k):raise AssertionError('historical Q4 performed event work')
        monkeypatch.setattr(owner,'_require_stateful_scope',forbidden)
        with pytest.raises(ValueError,match='explicit algorithmic Q4'):owner.rebase_rotations(0,preserve_supported_charts=True)
    finally:owner.close()
