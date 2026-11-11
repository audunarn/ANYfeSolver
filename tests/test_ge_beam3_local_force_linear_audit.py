"""Diagnose captured failures without changing or repeating the force program."""
import json
import os
from pathlib import Path
from decimal import Decimal as D,localcontext
import numpy as np
import pytest
from docs.reference_cases.ge_beam3_decimal_linear_audit import solve_linear
from anyfesolver._ge_beam3_p5_seeded.core import canonical


def test_decimal_pivoting_keeps_weak_rhs_and_indefinite_blocks():
    solution=solve_linear([[0.,1.,0.],[1.,1e16,1.],[0.,1.,-1.]], [1.,1e16,0.])
    assert tuple(D(x) for x in solution)==(D(-1),D(1),D(1))


@pytest.mark.parametrize('index',[0,1,2])
def test_captured_local_newton_step_against_decimal(index,tmp_path):
    setting=os.environ.get('ANY_GE_BEAM3_FORCE_DIAGNOSTIC')
    if setting is None:pytest.skip('explicit captured force diagnostic required')
    root=Path(setting)/('test_curved_high_contrast_nati'+str(index))
    with np.load(root/'last-local-evaluation.npz',allow_pickle=False) as packet:
        h=packet['hessian'];residual=packet['residual'];coordinates=packet['reference']
    data=json.loads((root/'diagnostic.json').read_text())
    a=h[18:,18:];b=-residual[18:]
    x=np.linalg.solve(a,b)
    scale=np.sqrt(np.abs(np.diag(a)))
    assert np.all(scale>0)
    scaled=np.linalg.solve(a/scale[:,None]/scale[None,:],b/scale)/scale
    low=solve_linear(a.tolist(),b.tolist(),digits=60)
    high=solve_linear(a.tolist(),b.tolist(),digits=90)
    with localcontext() as context:
        context.prec=100
        assert max(abs(D(l)-D(r)) for l,r in zip(low,high))<D('1e-40')
    exact=np.array(high,dtype=float)
    length=max(np.linalg.norm(coordinates[j]-coordinates[i]) for i,j in ((0,1),(1,2),(0,2)))
    def effect(step):
        force=(h[:18,18:]@step).reshape(3,6);force[:,3:]/=length
        return float(np.linalg.norm(force))
    record=dict(slenderness=data['slenderness'],force_program_status=data['status'],
        raw_condition_diagnostic=float(np.linalg.cond(a)),scaled_condition_diagnostic=float(np.linalg.cond(a/scale[:,None]/scale[None,:])),
        residual_inf=float(np.linalg.norm(residual[18:],np.inf)),
        raw_step_relative_error=float(np.linalg.norm(x-exact)/np.linalg.norm(exact)),
        scaled_step_relative_error=float(np.linalg.norm(scaled-exact)/np.linalg.norm(exact)),
        raw_external_error_estimate=effect(x),scaled_external_error_estimate=effect(scaled),
        decimal_step_external_error_estimate=effect(exact),
        decimal_step=high,production_qualified=False,diagnostic_only=True)
    with (tmp_path/'linear-audit.json').open('xb') as stream:stream.write(canonical(record))
    print(canonical(record).decode(),flush=True)
