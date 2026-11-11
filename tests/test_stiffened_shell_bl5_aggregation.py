"""Whole-halfcell P0 aggregation; no interface stability qualification."""
from dataclasses import replace
import numpy as np
import pytest
from scipy import sparse
from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from test_stiffened_shell_l1_assembly_experimental import _bl5_joint_mesh_definition


def _definition(span=1, supports=True):
    base = _bl5_joint_mesh_definition(4, 8, 1)
    fixed = base.fixed
    if supports:
        fixed += tuple(stiffened_shell.FixedCoordinate("beam", 101, i) for i in range(5))
    return replace(base, fixed=fixed, lines=(replace(base.lines[0], multiplier_span=span),))


def _aggregation(cells, span):
    # Independent density-work rule: each coarse constant multiplier acts on
    # the sum of whole fine support integrals, in every physical channel.
    groups=cells//span
    return sparse.coo_matrix((np.ones(5*cells),
        ([channel*groups+edge//span for channel in range(5) for edge in range(cells)],
         np.arange(5*cells))),shape=(5*groups,5*cells)).tocsr()


@pytest.mark.parametrize("span", (4,8))
def test_whole_halfcell_groups_preserve_weak_density_work_and_reference_state(span):
    fine=stiffened_shell.create_analysis(definition=_definition())
    coarse=stiffened_shell.create_analysis(definition=_definition(span))
    try:
        aggregation=_aggregation(8,span)
        assert coarse._C.shape[0]==40//span
        assert coarse.capabilities["experimental_line_spaces"]==(f"BL5_P0_span_{span}",)
        np.testing.assert_allclose(coarse._C.toarray(),(aggregation@fine._C).toarray(),atol=1e-14,rtol=1e-13)
        np.testing.assert_array_equal(coarse._K.toarray(),fine._K.toarray())
        assert len(coarse._line_samples)==len(fine._line_samples)
        rng=np.random.default_rng(931)
        density=rng.normal(size=coarse._C.shape[0]); q=rng.normal(size=coarse._K.shape[0])
        np.testing.assert_allclose(density@(coarse._C@q),(aggregation.T@density)@(fine._C@q),atol=1e-13,rtol=1e-13)
        assert fine.preflight()["identity"]!=coarse.preflight()["identity"]
        action=(stiffened_shell.NodalAction("beam",109,(0.,100.,0.,0.,0.,0.)),)
        solved=coarse.solve_linear(action)
        assert solved.status=="completed" and solved.free_equilibrium_ratio<1e-8
        restored=stiffened_shell.create_analysis(definition=_definition(span))
        try:
            np.testing.assert_array_equal(restored.restore(coarse.checkpoint()).beam_coordinates,solved.beam_coordinates)
        finally: restored.close()
        with pytest.raises(ValueError,match="incompatible mixed linear checkpoint"):
            fine.restore(coarse.checkpoint())
    finally:
        fine.close();coarse.close()


def test_finite_aggregation_preserves_constraint_curvature_and_same_parent_work():
    fine=stiffened_shell.create_analysis(definition=_definition())
    coarse=stiffened_shell.create_analysis(definition=_definition(4))
    try:
        first,second=FinitePointDriver(fine),FinitePointDriver(coarse)
        count=fine._K.shape[0]; aggregation=_aggregation(8,4)
        rng=np.random.default_rng(318)
        physical=rng.normal(size=count)*1e-4
        density=rng.normal(size=coarse._C.shape[0])*5e4
        state_f=first._advance(first._state,np.r_[physical,aggregation.T@density],1.)
        state_c=second._advance(second._state,np.r_[physical,density],1.)
        pattern=np.zeros(count)
        rf,kf,*_=first._assemble(state_f,0.,first._shell_origins,first._beam_origins,pattern=pattern)
        rc,kc,*_=second._assemble(state_c,0.,second._shell_origins,second._beam_origins,pattern=pattern)
        np.testing.assert_allclose(rc[:count],rf[:count],rtol=1e-12,atol=1e-7)
        np.testing.assert_allclose(rc[count:],aggregation@rf[count:],rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(kc[:count,:count].toarray(),kf[:count,:count].toarray(),rtol=1e-12,atol=1e-7)
        np.testing.assert_allclose(kc[count:,:count].toarray(),(aggregation@kf[count:,:count]).toarray(),rtol=1e-12,atol=1e-14)
        assert not np.shares_memory(state_c.q,state_f.q)
    finally:
        fine.close();coarse.close()


@pytest.mark.parametrize("span", (0,-1,3,9,1.5,True))
def test_invalid_or_partial_aggregation_is_not_inferred(span):
    with pytest.raises(ValueError,match="ordered even-edge"):
        _definition(span)


def test_general_single_support_does_not_hide_released_native_rigid_mode():
    owner=stiffened_shell.create_analysis(definition=_definition(8,supports=False))
    try:
        with pytest.raises(ValueError,match="single-support BL5 leaves a free native beam rigid mode"):
            owner.solve_linear((stiffened_shell.NodalAction("beam",109,(0.,100.,0.,0.,0.,0.)),))
        assert owner._generation==0
    finally:owner.close()


@pytest.mark.parametrize("span", (2,8))
@pytest.mark.parametrize("adaptive", (False,True))
@pytest.mark.parametrize("empty", (False,True))
def test_zero_pattern_cannot_publish_known_unsupported_finite_layout(span,adaptive,empty):
    # Span2 is the pre-existing two-halfcell restriction; span8 exercises
    # the extension. Zero load must not bypass admission before factorization.
    base=_bl5_joint_mesh_definition(4,span,1)
    base=replace(base,lines=(replace(base.lines[0],multiplier_span=span),))
    owner=stiffened_shell.create_analysis(definition=base)
    try:
        actions=() if empty else (stiffened_shell.NodalAction("beam",101+span,(0.,)*6),)
        control=stiffened_shell.AdaptiveIncrementControl(2,8) if adaptive else None
        with pytest.raises(ValueError,match="single-support BL5 leaves a free native beam rigid mode"):
            owner.solve(actions,increment_control=control)
        assert owner._generation==0 and owner._accepted is None and owner._finite_driver is None
        with pytest.raises(ValueError,match="single-support BL5 leaves a free native beam rigid mode"):
            owner.restore(b"obsolete unsupported state")
        assert owner._generation==0
    finally:owner.close()
