"""Native distributed loads through the public mixed owner and durable state."""
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_distributed_load import decode_checkpoint, encode_checkpoint
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from test_stiffened_shell_ge_experimental import _panel


def owner(request, definition=None):
    a=ss.create_analysis(definition=definition or replace(_panel(),q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1))
    request.addfinalizer(a.close)
    return a


def load(a, force=(0.,0.,-100.), couple=(3.,5.,2.)):
    return ss.GEBeamDistributedLoad(a.preflight()['identity'],tuple(
        ss.GEBeamLineAction(b.element_id,force,couple) for b,_,_ in a._beam_parts))


def test_compiled_stage_density_scatter_reuses_geometry(request,monkeypatch):
    from types import SimpleNamespace
    import anyfesolver._stiffened_shell_native_distributed_load as native
    a=owner(request);context=load(a).compile(a)
    state=SimpleNamespace(q=np.zeros(context.count),
        cells=np.tile(np.eye(3),(context.beam_count,2,1,1)))
    densities=np.array([[1.,-2.,3.,-.5,.2,.1],[-3.,4.,-5.,.7,-.4,.8]])
    descriptor=ss.GEBeamDistributedLoad(a._identity,tuple(
        ss.GEBeamLineAction(row[3],densities[i,:3],densities[i,3:])
        for i,row in enumerate(context.beams)))
    expected=descriptor.compile(a).evaluate(state)
    def forbidden(*args,**kwargs):raise AssertionError('stage rebuilt native geometry')
    monkeypatch.setattr(native,'compile_native_distributed',forbidden)
    force,jac,recovery=context.evaluate(state,densities=densities)
    np.testing.assert_array_equal(force,expected[0])
    np.testing.assert_array_equal(jac.toarray(),expected[1].toarray())
    assert recovery.applied_force==expected[2].applied_force
    assert recovery.applied_moment_about_origin==expected[2].applied_moment_about_origin
    densities[:]=0.
    zero,zjac,zrecovery=context.evaluate(state,densities=densities)
    assert not np.any(zero) and zjac.nnz==0 and zrecovery.potential==0.
    np.testing.assert_array_equal(context.evaluate(state)[0],context.reference_gradient)
    assert np.any(force)
    for bad in (np.zeros((1,6)),np.full((2,6),np.nan),np.ones((2,6))*1j):
        with pytest.raises(ValueError):context.evaluate(state,densities=bad)


def balance(a,r):
    reactions=r.support_reactions if hasattr(r,'support_reactions') else r.reactions
    points=np.asarray(a._shell_xyz)
    if hasattr(r,'beam_positions'):points=points+r.shell_displacement.reshape(-1,6)[:,:3]
    w=reactions[:a._shell_count].reshape(-1,6)
    np.testing.assert_allclose(w[:,:3].sum(0),-np.asarray(r.distributed_load.applied_force),atol=1e-6,rtol=1e-8)
    np.testing.assert_allclose((np.cross(points,w[:,:3])+w[:,3:]).sum(0),
        -np.asarray(r.distributed_load.applied_moment_about_origin),atol=2e-5,rtol=1e-7)


def test_descriptor_owned_canonical_and_model_admission(request):
    a=owner(request);f=np.array([1.,2.,3.]);row=ss.GEBeamLineAction(1,f)
    f[:]=99
    assert row.force_N_per_reference_m==(1.,2.,3.)
    q=ss.GEBeamDistributedLoad(a._identity,(ss.GEBeamLineAction(2,(1,0,0)),row))
    assert [v.element_id for v in q.actions]==[1,2]
    assert ss.GEBeamDistributedLoad.deserialize(q.serialize())==q
    for bad in (0,True,1.5):
        with pytest.raises(ValueError):ss.GEBeamLineAction(bad,(1,0,0))
    for bad in ((0,0,0),(float('nan'),0,1),(True,0,1)):
        with pytest.raises(ValueError):ss.GEBeamLineAction(1,bad)
    with pytest.raises(ValueError):ss.GEBeamDistributedLoad(a._identity,(row,row))
    with pytest.raises(ValueError):ss.GEBeamDistributedLoad('0'*64,(row,)).compile(a)
    with pytest.raises(ValueError):ss.GEBeamDistributedLoad(a._identity,(ss.GEBeamLineAction(99,(1,0,0)),)).compile(a)
    with pytest.raises(ValueError):ss.GEBeamDistributedLoad.deserialize(q.serialize()+b' ')


@pytest.mark.parametrize('force,couple', [((0,0,-100),(0,0,0)),((0,0,0),(3,5,2)),((0,0,-100),(3,5,2))])
def test_linear_resultants_factor_reuse_and_checkpoint(request,force,couple):
    a,b=owner(request),owner(request);q=load(a,force,couple)
    r=a.solve_linear(q);assert r.status=='completed';balance(a,r)
    np.testing.assert_allclose(r.distributed_load.applied_force,2*np.asarray(force),atol=1e-12)
    expected=np.cross((1.,-.5,.1),2*np.asarray(force))+2*np.asarray(couple)
    np.testing.assert_allclose(r.distributed_load.applied_moment_about_origin,expected,atol=1e-12)
    assert (r.distributed_load.potential is None)==bool(any(couple))
    assert r.body_load is None
    saved=a.checkpoint();restored=b.restore(saved)
    np.testing.assert_array_equal(restored.beam_coordinates,r.beam_coordinates)
    assert saved==b.checkpoint()
    a.solve_linear(q);assert a._linear_factorizations==1
    descriptor,inner=decode_checkpoint(saved);assert descriptor==q
    target=owner(request)
    wrong=load(target,(0,0,-101),couple).compile(target)
    with pytest.raises(ValueError):target.restore(encode_checkpoint(wrong,inner))
    with pytest.raises(ValueError):target.restore(inner)
    assert target._accepted is None and target._generation==0
    legacy=a.solve_linear((ss.NodalAction('beam',105,(0,0,1,0,0,0)),))
    assert legacy.distributed_load is None and legacy.body_load is None


def test_finite_balance_atomic_rejection_cancel_chart_restart(request,monkeypatch):
    a,b=owner(request),owner(request);q=load(a)
    first=a.solve(q,(.25,),max_iterations=15);balance(a,first)
    saved=a.checkpoint();token=CancellationToken();token.cancel()
    with pytest.raises(SolveCancelled):a.solve(q,(.5,),cancellation_token=token)
    with pytest.raises(ValueError):a.solve(ss.GEBodyLoad(a._identity,(0,0,-9.81)),(.5,))
    original=FinitePointDriver._assemble
    def broken(self,*args,**kwargs):
        original(self,*args,**kwargs)
        raise RuntimeError('injected distributed trial failure')
    with monkeypatch.context() as scope:
        scope.setattr(FinitePointDriver,'_assemble',broken)
        with pytest.raises(RuntimeError,match='injected'):a.solve(q,(.5,))
    assert a.checkpoint()==saved
    b.restore(saved)
    last=a.solve(q,(.5,),max_iterations=15);resumed=b.solve(q,(.5,),max_iterations=15)
    np.testing.assert_array_equal(last.beam_positions,resumed.beam_positions)
    assert a.checkpoint()==b.checkpoint();balance(a,last)
    a.rebase_rotations(last.generation)
    c=owner(request);c.restore(a.checkpoint())
    final=a.solve(q,(1.,),max_iterations=15);continued=c.solve(q,(1.,),max_iterations=15)
    np.testing.assert_array_equal(final.beam_positions,continued.beam_positions)
    balance(a,final)


def test_global_load_derivative_has_no_couple_chart_skew(request):
    from anyfesolver._ge_beam3_p5.algebra import rotation
    a=owner(request);driver=FinitePointDriver(a)
    pattern,_=driver._prepare_load(load(a),(.1,))
    with pytest.raises(ValueError):driver._prepare_load(load(a,couple=(3.,5.,3.)),(.1,))
    cells=np.tile(rotation([.1,-.08,.12]),(2,2,1,1))
    state=replace(driver._state,cells=cells)
    r1,k1,*_=driver._assemble(state,1.,driver._shell_origins,driver._beam_origins,pattern=pattern)
    r0,k0,*_=driver._assemble(state,0.,driver._shell_origins,driver._beam_origins,pattern=pattern)
    force,jac,_=driver._external.evaluate(state);n=len(force)
    np.testing.assert_allclose(r1[:n]-r0[:n],-force,rtol=1e-7,atol=1e-6)
    np.testing.assert_allclose((k1-k0)[:n,:n].toarray(),-jac.toarray(),atol=1e-6)
    direction=np.zeros(len(r1));direction[a._shell_count+30:a._shell_count+36]=(.3,-.2,.1,.2,.4,-.1)
    h=1e-5
    fp=driver._external.evaluate(driver._advance(state,direction,h))[0]
    fm=driver._external.evaluate(driver._advance(state,direction,-h))[0]
    np.testing.assert_allclose((fp-fm)/(2*h),jac@direction[:n],rtol=1e-7,atol=1e-8)


@pytest.mark.parametrize('triangles',[False,True])
def test_mixed_s3_distributed_restart(request,triangles):
    base=_panel();shell=replace(base.shell,q4_elements=() if triangles else ((1,2,5,4),),
        s3_elements=((1,2,5),(1,5,4),(2,3,6),(2,6,5)) if triangles else ((2,3,6),(2,6,5)))
    d=replace(base,shell=shell,**({} if triangles else {'q4_pullback_policy':ss.Q4_ALGORITHMIC_PULLBACK_V1}))
    a,b=owner(request,d),owner(request,d);q=load(a)
    r=a.solve(q,(.1,),max_iterations=15);balance(a,r)
    restored=b.restore(a.checkpoint())
    np.testing.assert_array_equal(r.beam_positions,restored.beam_positions)
    assert a.checkpoint()==b.checkpoint()


@pytest.mark.parametrize('mixed',[False,True])
def test_physical_fibre_and_mixed_cancelled_prefix(request,mixed):
    from test_stiffened_shell_ge_physical_fibre_linear import _definition
    d=replace(_definition(mixed=mixed),q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    a,b=owner(request,d),owner(request,d);q=load(a);token=CancellationToken()
    control=ss.AdaptiveIncrementControl(2,16)
    def cancel(result):token.cancel('complete distributed prefix')
    with pytest.raises(SolveCancelled):
        a.solve(q,(.1,.2),increment_control=control,on_accepted=cancel,cancellation_token=token)
    assert a.recover().accepted_factors==(.1,)
    b.restore(a.checkpoint())
    last=a.solve(q,(.1,.2),increment_control=control)
    resumed=b.solve(q,(.1,.2),increment_control=control)
    np.testing.assert_array_equal(last.beam_positions,resumed.beam_positions)
    assert a.checkpoint()==b.checkpoint()
    assert resumed.distributed_load.potential is None
    balance(a,last)


def test_nonsequential_ids_partial_loading_independent_scatter(request):
    from anyfesolver import b3_ge
    from anyfesolver._stiffened_shell_native_distributed_load import compile_native_distributed
    base=_panel();beams=[]
    for old,newid in zip(base.beams,(71,13)):
        element,inertia=old.instantiate();ref=element.operator.reference
        beams.append(b3_ge.define_beam('b3-ge',newid,element.node_ids,ref.coordinates,
            ref.nodal_triads,element.section,inertia,quadrature=element.operator.order))
    a=owner(request,replace(base,beams=tuple(beams)))
    q=ss.GEBeamDistributedLoad(a._identity,(ss.GEBeamLineAction(13,(1,2,3),(4,5,6)),))
    context=q.compile(a);n=a._K.shape[0];expected=np.zeros(n)
    element,slots,_=a._beam_parts[1]
    kernel=compile_native_distributed(element.operator,(1,2,3),(4,5,6))
    local=kernel.evaluate(kernel.coordinates,np.zeros((3,3)),np.tile(np.eye(3),(2,1,1)))
    np.add.at(expected,np.asarray(slots)+a._shell_count,local.external_force)
    np.testing.assert_array_equal(context.reference_gradient,expected)
    r=a.solve_linear(q);balance(a,r)
    _,inner=decode_checkpoint(a.checkpoint());fresh=owner(request,replace(base,beams=tuple(beams)))
    wrong=ss.GEBeamDistributedLoad(a._identity,(ss.GEBeamLineAction(71,(1,2,3),(4,5,6)),)).compile(fresh)
    with pytest.raises(ValueError):fresh.restore(encode_checkpoint(wrong,inner))
    assert fresh._accepted is None
    assert fresh.solve_linear(q).status=='completed'
