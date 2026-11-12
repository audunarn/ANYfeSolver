"""Capture completion, separate immutable spectral ownership and old routing."""
from hashlib import sha256
import pytest
from anyfesolver import _ge_beam3_analysis_modal_snapshot as snapshot
from anyfesolver import _ge_beam3_analysis_translation_modal as route
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken,SolveCancelled
from test_ge_beam3_native_analysis import definitions
from test_ge_beam3_elastic_seed_continuation import seed,model,programme


def made():
    source=model()
    return NativeBeamAnalysis(definitions(source),tuple(source.boundary_conditions),retained_refinement=True)


@pytest.mark.parametrize('seeded',(False,True))
def test_model_owned_large_profile_preserves_packet_and_small_modes(seed,seeded,tmp_path):
    analysis=made();p=programme((.0002,))
    kw=dict(seed=seed,expected_seed_sha256=sha256(seed).hexdigest()) if seeded else {}
    state=analysis.solve_translation(p,**kw);assert state.status=='completed'
    controls=dict(expected_sha256=state.checkpoint_sha256,bounds=(-100.,1e6),num_modes=6,**kw)
    old=analysis.translation_modes(p,state.checkpoint,**controls)
    large=analysis.translation_modes(p,state.checkpoint,modal_policy=route.LARGE_POLICY,**controls)
    assert canonical(old.packet)==canonical(large.packet)
    assert canonical(old.modes)==canonical(large.modes)
    assert large.modal_capacity_policy==route.LARGE_POLICY and len(large.snapshot_sha256)==64
    assert not large.production_qualified
    (tmp_path/'owned-large-modes.json').write_bytes(canonical(large))


def test_snapshot_requires_successful_original_guard():
    def expired():raise RuntimeError('original capture deadline')
    with pytest.raises(RuntimeError,match='original capture deadline'):
        snapshot.seal(None,None,None,expired)


@pytest.mark.parametrize('mutation',('program','packet','model','cancel'))
def test_detached_snapshot_checks_all_owned_inputs_without_reusing_capture(mutation):
    analysis=made();p={'load':1};packet={'factor':[1,2]};calls=[];token=CancellationToken()
    identity,guard=snapshot.seal(analysis,p,packet,lambda:calls.append('valid'),token)
    assert len(identity)==64 and calls==['valid']
    guard();assert calls==['valid']  # Numerical phase never resets or reuses context.
    if mutation=='program':p['load']=2
    elif mutation=='packet':packet['factor'][0]=3
    elif mutation=='model':analysis.model.mesh.nodes[1].x+=1
    else:token.cancel()
    with pytest.raises((ValueError,SolveCancelled)):guard()


def test_unknown_or_implicit_large_policy_rejected_before_capture():
    analysis=made()
    with pytest.raises(ValueError,match='registered modal'):
        route._controls(analysis,(-1.,20.),6,1e-10,1e-12,'unknown')
    analysis._retained_refinement=False
    with pytest.raises(ValueError,match='explicit retained'):
        route._controls(analysis,(-1.,20.),6,1e-10,1e-12,route.LARGE_POLICY)


def test_n32_size_requires_explicit_large_policy():
    class N32:
        _elements=tuple(range(32));_retained_refinement=True
        model=type('Model',(),{'mesh':type('Mesh',(),{'dof_manager':type('Dofs',(),{'total_dofs':390})()})()})()
    with pytest.raises(ValueError,match='coordinate bound'):
        route._controls(N32(),(-1.,20.),6,1e-10,1e-12)
    route._controls(N32(),(-1.,20.),6,1e-10,1e-12,route.LARGE_POLICY)
