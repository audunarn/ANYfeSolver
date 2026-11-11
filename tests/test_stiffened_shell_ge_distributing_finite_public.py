"""Experimental finite area workflow, actual state boundary and legacy guards."""
from dataclasses import replace
import numpy as np
import pytest

from anyfesolver import b3_ge, stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_distributing_pose import DistributingPoseDomainError
from anyfesolver._stiffened_shell_ge_distributing_finite import PreparedFiniteDistributingAttachment
from anyfesolver._stiffened_shell_ge_checkpoint_binary import MAGIC, encode
from test_stiffened_shell_ge_distributing_reference import _definition, _selection, _multi_definition, _multi_selection


def definition(multi=False):
    base = _multi_definition() if multi else _definition()
    geometry = _multi_selection() if multi else _selection()
    return replace(base, points=(), finite_distributing_footprints=(ss.GEFiniteDistributingFootprint(geometry),))


def actions():
    return (ss.NodalAction("beam",103,(0.,20.,100.,0.,0.,0.)),)


@pytest.fixture(scope="module")
def completed():
    model=definition();owner=ss.create_analysis(definition=model)
    try:
        result=owner.solve(actions(),(.25,.5,1.))
        yield model,owner,result
    finally:owner.close()


def test_public_force_route_and_physical_wrenches(completed):
    model,owner,result=completed
    assert result.status=="completed" and result.accepted_factors==(.25,.5,1.)
    assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
    assert owner.capabilities["checkpoints"]==("finite_distributing", "finite_distributing_displacement")
    record,=owner.recover().distributing_footprint_wrenches
    assert isinstance(record,ss.GEFiniteDistributingFootprintWrench)
    assert record.configuration=="current" and record.generation==result.generation
    assert record.law==ss.GE_DISTRIBUTING_FINITE_DIRECTOR_POLAR_V1_EXPERIMENTAL
    assert not record.production_qualified
    arm=(record.beam_position_m-record.centroid_m)+(record.beam_position_low_m-record.centroid_low_m)
    assert np.linalg.norm(record.shell_action[:3]+record.beam_action[:3])<1e-8
    assert np.linalg.norm(record.shell_action[3:]+record.beam_action[3:]+np.cross(arm,record.beam_action[:3]))<1e-8
    physical=owner._shell_count+6*len(owner._beam_ids)
    reactions=result.support_reactions[:physical].reshape(-1,6)
    assert np.linalg.norm(reactions[:,:3].sum(axis=0)+np.array((0.,20.,100.)))<1e-6
    selected,=owner.recover(selection=ss.GEReferenceDistributingFootprintSelection((record.footprint_id,)))
    assert selected is record
    with pytest.raises(ValueError):record.shell_node_actions.setflags(write=True)
    assert owner.preflight()["finite_distributing_local_curvature_nnz_upper_bound"]==(6*(len(record.shell_node_ids)+1))**2


def test_checkpoint_restart_continuation_matches_uninterrupted(completed):
    model,owner,result=completed
    resumed=b3_ge.create_stiffened_shell_analysis(formulation="b3-ge",definition=model)
    uninterrupted=ss.create_analysis(definition=model)
    prefix=ss.create_analysis(definition=model)
    try:
        prefix.solve(actions(),(.25,.5))
        checkpoint=prefix.checkpoint();assert checkpoint.startswith(MAGIC)
        restored=resumed.restore(checkpoint)
        assert restored.accepted_factors==(.25,.5)
        assert resumed.checkpoint()==checkpoint
        a=resumed.solve(actions(),(1.,))
        b=uninterrupted.solve(actions(),(.25,.5,1.))
        for name in ("shell_displacement","beam_positions","beam_frames","cell_frames","cell_resultants","multipliers","support_reactions"):
            np.testing.assert_allclose(getattr(a,name),getattr(b,name),rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(a.distributing_footprint_wrenches[0].shell_action,
                                   b.distributing_footprint_wrenches[0].shell_action,rtol=1e-10,atol=1e-10)
    finally:resumed.close();uninterrupted.close();prefix.close()


def test_cancel_after_commit_preserves_complete_prefix():
    owner=ss.create_analysis(definition=definition());token=CancellationToken()
    try:
        def accepted(result):
            assert owner.recover().generation==result.generation
            assert owner.checkpoint().startswith(MAGIC)
            token.cancel("accepted finite pad prefix")
        with pytest.raises(SolveCancelled):owner.solve(actions(),(.25,.5),cancellation_token=token,on_accepted=accepted)
        assert owner.recover().accepted_factors==(.25,)
        owner.solve(actions(),(.5,))
        assert owner.recover().accepted_factors==(.25,.5)
    finally:owner.close()


@pytest.mark.parametrize("failure",(DistributingPoseDomainError,MemoryError,RuntimeError))
def test_rejected_or_programming_candidate_preserves_checkpoint(monkeypatch,failure):
    owner=ss.create_analysis(definition=definition())
    try:
        owner.solve(actions(),(.25,));before=owner.checkpoint();generation=owner.recover().generation
        q=owner._finite_driver._state.q.copy();original=PreparedFiniteDistributingAttachment.evaluate
        def broken(self,actual,state,multipliers,**kwargs):
            if not np.array_equal(state.q,q):raise failure("injected candidate failure")
            return original(self,actual,state,multipliers,**kwargs)
        monkeypatch.setattr(PreparedFiniteDistributingAttachment,"evaluate",broken)
        with pytest.raises(RuntimeError if failure is DistributingPoseDomainError else failure):
            owner.solve(actions(),(.5,),max_backtracks=2)
        assert owner.recover().generation==generation and owner.checkpoint()==before
    finally:owner.close()


def test_multihost_finite_owner_and_preparation_reuse(monkeypatch):
    owner=ss.create_analysis(definition=definition(True))
    try:
        import anyfesolver._stiffened_shell_ge_distributing_reference as reference
        def forbidden(*args,**kwargs):raise AssertionError("reference search repeated during solve")
        monkeypatch.setattr(reference,"_compile_reference_geometry",forbidden)
        result=owner.solve(actions(),(.25,.5))
        record,=result.distributing_footprint_wrenches
        assert len(record.host_element_ids)==8 and len(record.shell_node_ids)==15
        assert result.constraint_max<=1e-9
    finally:owner.close()


def test_disjoint_point_and_pad_offsets_recovery_and_restart():
    model=replace(definition(),points=(ss.PointConnection(1,101),))
    owner=ss.create_analysis(definition=model);resumed=ss.create_analysis(definition=model)
    try:
        a=owner.solve(actions(),(.25,.5))
        assert len(a.point_wrenches)==1 and len(a.distributing_footprint_wrenches)==1
        assert a.multipliers.shape==(12,)
        assert owner._finite_distributing_bindings[0][1]==6
        b=resumed.restore(owner.checkpoint())
        np.testing.assert_allclose(b.multipliers,a.multipliers,rtol=1e-10,atol=1e-12)
        a=owner.solve(actions(),(1.,));b=resumed.solve(actions(),(1.,))
        np.testing.assert_allclose(b.shell_displacement,a.shell_displacement,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(b.distributing_footprint_wrenches[0].beam_action,
                                  a.distributing_footprint_wrenches[0].beam_action,rtol=1e-10,atol=1e-10)
    finally:owner.close();resumed.close()


@pytest.mark.parametrize("operation",("linear","adaptive","displacement","arc","chart","wrapper","moment","reverse","modal"))
def test_excluded_operations_fail_before_state_mutation(operation):
    owner=ss.create_analysis(definition=definition())
    try:
        if operation=="modal":assert not owner.reference_modal_capabilities["eligible"]
        else:
            with pytest.raises(ValueError):
                if operation=="linear":owner.solve_linear(actions())
                elif operation=="adaptive":owner.solve(actions(),increment_control=object())
                elif operation=="displacement":owner.solve(actions(),displacement_control=object())
                elif operation=="arc":owner.continue_arc_length(actions(),object())
                elif operation=="chart":owner.rebase_rotations(0)
                elif operation=="wrapper":owner.solve(object())
                elif operation=="moment":owner.solve((ss.NodalAction("beam",103,(0.,0.,0.,0.,1.,0.)),))
                elif operation=="reverse":owner.solve(actions(),(.5,.25))
        assert owner._generation==0 and owner._finite_driver is None
    finally:owner.close()


@pytest.mark.parametrize("change",("mixed","untyped","overlap","s3","material","policy"))
def test_explicit_records_and_admission_negatives(change):
    model=definition()
    if change=="mixed":model=replace(model,distributing_footprints=(_selection(),))
    elif change=="untyped":model=replace(model,finite_distributing_footprints=(_selection(),))
    elif change=="overlap":model=replace(model,points=(ss.PointConnection(3,102),))
    elif change=="s3":model=replace(model,shell=replace(model.shell,s3_elements=((1,2,6),)))
    elif change=="material":model=replace(model,bl5_material_policy=ss.BL5_GENERALIZED_MATERIAL_V1)
    elif change=="policy":model=replace(model,q4_pullback_policy="historical")
    with pytest.raises(ValueError):ss.create_analysis(definition=model)


def test_checkpoint_kind_corruption_and_cross_model_do_not_publish(completed):
    model,accepted,result=completed
    original=accepted.checkpoint()
    for raw in (original[:-1]+bytes((original[-1]^1,)),b"ANYSGECP6\0"+original,b"{}"):
        owner=ss.create_analysis(definition=model)
        try:
            with pytest.raises(ValueError):owner.restore(raw)
            assert owner._generation==0 and owner._finite_driver is None
        finally:owner.close()
    state=accepted._finite_driver._state
    arrays=(("q",state.q),("frames",state.frames),("cells",state.cells),("resultants",state.resultants),
        ("multipliers",state.multipliers),("pattern",accepted._finite_driver._pattern),
        ("accepted_factors",np.array(result.accepted_factors)))
    forged=encode("finite",accepted.preflight()["identity"],result.generation,arrays)
    owner=ss.create_analysis(definition=model)
    try:
        with pytest.raises(ValueError,match="kind"):owner.restore(forged)
        assert owner._generation==0
    finally:owner.close()
    reference=ss.create_analysis(definition=replace(model,finite_distributing_footprints=(),distributing_footprints=(_selection(),)))
    try:
        with pytest.raises(ValueError):reference.restore(original)
    finally:reference.close()
