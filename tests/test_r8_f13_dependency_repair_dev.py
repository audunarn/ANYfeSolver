"""Frozen F13 affine-offset regression through the actual public producer."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from anyfesolver import stiffened_shell as api
from anyfesolver._stiffened_shell_b2_q4_panel_experimental import ExperimentalB2Q4PanelAnalysis
from scripts import r8_f13_static_feasibility as frozen

ROOT = Path(__file__).resolve().parents[1]

def context():
    return SimpleNamespace(np=np,sparse=sparse,api=api,inputs=json.loads((ROOT/'docs/agent_plans/development/r8-f13-static-feasibility-v5-20261009/inputs-v5.json').read_text()))

def test_frozen_affine_offset_preserves_physical_work_and_public_lifecycle():
    ctx=context();owner=api.create_analysis(definition=frozen.definition(ctx,'physical-footprint','affine-offset'))
    try:
        applied=frozen.load(ctx)
        result=owner.solve_linear(applied,**frozen.full_selection(ctx,'physical-footprint'))
        metrics=frozen.physical_checks(ctx,owner,result,applied)
        assert metrics['parity']<=1e-10 and metrics['constraint']<=1e-9
        assert result.factorization_count==result.solve_count==1
        assert result.constraint_gaps[9]==0.
        # Nonzero prescribed translation rows must remain inhomogeneous:
        # zero substitution there would erase the actual affine offset.
        assert any(owner._cu0 != 0.)
        assert all(owner._cu0[row]==0. for row,_,_ in owner._physical_singletons)
        assert result.displacements[0]==pytest.approx(1.3e-5,rel=1e-10)
        assert owner.solve_linear(applied).displacements==result.displacements
    finally:owner.close();owner.close()
    with pytest.raises(RuntimeError):owner.solve_linear(applied)

def test_exact_singleton_cleanup_leaves_material_errors_for_guards():
    ctx=context();owner=api.create_analysis(definition=frozen.definition(ctx,'physical-footprint','affine-offset'))
    try:
        applied=frozen.load(ctx)
        real=owner._handle
        class Corrupt:
            def __getattr__(self,name):return getattr(real,name)
            def solve(self,rhs):
                answer=real.solve(rhs).copy()
                answer[0]+=1./owner._factor_scale[0]
                return answer
        owner._handle=Corrupt()
        result=owner.solve_linear(applied)
        assert result.status=='failed'
        assert result.failure_reason in ('panel equilibrium correction did not improve','panel point gap or equilibrium failed')
    finally:owner.close()

def test_conflicting_fixed_footprint_pose_is_refused_before_substitution():
    ctx=context();base=frozen.definition(ctx,'physical-footprint')
    # Prescribe shell rotation about x while its attached beam rotation stays
    # fixed to zero. Removing that free coordinate leaves an inconsistent
    # supported compatibility equation, which must retain admission refusal.
    offset=np.zeros(54);offset[3]=.001
    transform=base.transformation[:,[i for i in range(base.transformation.shape[1]) if i!=3]]
    with pytest.raises(ValueError,match='independent supported point rows'):
        api.create_analysis(definition=replace(base,transformation=transform,prescribed_offset=offset))

def test_rounded_cancellation_never_proves_homogeneous_singleton():
    select=ExperimentalB2Q4PanelAnalysis._structural_physical_singletons
    C=sparse.csr_matrix([[1.,1.,-1.]])
    T=sparse.csr_matrix([[1.,1e16],[0.,1.],[0.,1e16]])
    # Binary input exact coefficient of the second column is one even though
    # sequential floating point summation can erase it.
    assert select(C,T,np.zeros(3))==()
    singleton=sparse.csr_matrix([[1.],[0.],[0.]])
    assert select(C,singleton,np.array([1e16,1.,1e16]))==()
    assert select(C,singleton,np.zeros(3))==((0,0,1.),)
    assert select(C,sparse.csr_matrix([[1.],[1.],[1.]]),np.zeros(3))==()
