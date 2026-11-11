"""Explicit stable elastic S3/mixed-shell native BL5 public development slice."""
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from examples.stiffened_shell_native_fibre_bl5 import definition,workflow
from test_stiffened_shell_native_fibre_bl5 import (
 ACTIONS,test_public_recovery_fresh_restart_continue_and_spectrum as _public,
 test_full42_and_direct_bl5_scalar_curvature_oracle as _oracle,
 test_public_refusals_atomic_corruption_cancel_mutation as _refusals)
from anyfesolver import _stiffened_shell_ge_point_buckling as buckling


@pytest.fixture(scope='module',params=(('s3','biaxial','fibre','p0'),
    ('mixed','biaxial','mixed','continuous_p1_coarse'),('s3','axial','fibre','p0')))
def accepted(request):
    shell,family,beams,basis=request.param
    owner=ss.create_analysis(definition=definition(family,beams,basis,shell=shell))
    result=owner.solve(ACTIONS,(.5,1.),max_iterations=24,max_backtracks=12)
    assert result.status=='completed'
    assert owner._finite_driver._stable_s3
    yield owner
    owner.close()


def test_public_complete_s3_workflow(accepted):
    _public(accepted)
    assert accepted.native_line_buckling_capabilities['policy']==buckling.NATIVE_MIXED_SHELL_LINE_POLICY


def test_independent_full_coordinate_s3_and_bl5_work_oracle(accepted):
    _oracle(accepted)


def test_atomic_public_refusals_and_guards(accepted):
    _refusals(accepted)


def test_chart_snapshot_fresh_restore_continuation_and_prestress(accepted):
    fresh=ss.create_analysis(definition=accepted._definition)
    other=ss.create_analysis(definition=accepted._definition)
    try:
        before=accepted.checkpoint();fresh.restore(before)
        initial=fresh.buckling(ss.GENativeLineBucklingRequest(fresh._generation,count=2))
        assert initial.status=='completed',initial.failure_reason
        fresh.rebase_rotations(expected_generation=fresh._generation)
        charted=fresh.checkpoint();other.restore(charted)
        assert other.checkpoint()==charted
        rebased=fresh.buckling(ss.GENativeLineBucklingRequest(fresh._generation,count=2))
        assert rebased.status=='completed',rebased.failure_reason
        for a,b in zip(initial.prestress_multipliers,rebased.prestress_multipliers,strict=True):
            assert abs(a-b)/max(abs(a),abs(b),np.finfo(float).tiny)<=1e-9
        a=fresh.solve(ACTIONS,(1.2,),max_iterations=24,max_backtracks=12)
        b=other.solve(ACTIONS,(1.2,),max_iterations=24,max_backtracks=12)
        assert a.status==b.status=='completed' and fresh.checkpoint()==other.checkpoint()
        assert accepted.checkpoint()==before
    finally:fresh.close();other.close()


@pytest.mark.parametrize('shell',('s3','mixed'))
def test_topology_and_distinct_policy_admission(shell):
    model=definition(shell=shell)
    with pytest.raises(ValueError):ss.create_analysis(definition=replace(model,bl5_material_policy=ss.BL5_ELASTIC_NATIVE_FIBRE_V1))
    if shell=='mixed':
        with pytest.raises(ValueError):ss.create_analysis(definition=replace(model,q4_pullback_policy='historical_corotational'))
    line=model.lines[0]
    # All nodes exist and remain collinear, but this skips actual cell edges.
    with pytest.raises(ValueError,match='element edges'):
        ss.create_analysis(definition=replace(model,lines=(replace(line,shell_nodes=(1,3,4)),)))
    q4=definition()
    with pytest.raises(ValueError):ss.create_analysis(definition=replace(q4,bl5_material_policy=ss.BL5_ELASTIC_NATIVE_FIBRE_MIXED_SHELL_V1))


@pytest.mark.parametrize('shell',('s3','mixed'))
def test_headless_s3_public_example(shell):
    result=workflow(count=1,shell=shell)
    assert result['status']=='completed' and result['policy']==buckling.NATIVE_MIXED_SHELL_LINE_POLICY,result
    assert result['fresh_restore_bytes_equal'] and result['continued_bytes_equal']
