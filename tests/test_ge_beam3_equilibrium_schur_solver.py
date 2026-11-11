"""Bounded saved-state linear-system gate; no native path or modal reruns."""
import json
from pathlib import Path
from hashlib import sha256
import numpy as np
import pytest
from anyfesolver._ge_beam3_equilibrium_schur_solver import EquilibriumSchurSolver
from anyfesolver._ge_beam3_fibre_schur_solver import ResultantSchurSolver as Previous
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.boundary import BoundaryCondition
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._ge_beam3_fibre_line_work import ReferenceLineForces
from anyfesolver import _ge_beam3_fibre_line_program as physical
from anyfesolver import _ge_beam3_schur_line_program as controller
from docs.reference_cases.ge_beam3_exact_linear_witness import solve as exact_solve
from docs.reference_cases.ge_beam3_curved_moment_fixture import model as curved_model
from test_ge_beam3_schur_line_program import inputs,save

ARCHIVE=Path('C:/Users/AudunArnesenNyhus/AppData/Local/ANYrelease/ge-beam3-schur-controller-2756a7b-20260907')
MANIFEST=Path(__file__).resolve().parents[1]/'docs/reference_cases/ge_beam3_schur_controller_archive.json'


@pytest.fixture(autouse=True)
def forbid_path_reruns(monkeypatch):
    def reject(*args,**kw): raise AssertionError('native path rerun forbidden')
    monkeypatch.setattr(controller,'solve_force_program',reject)
    monkeypatch.setattr(physical,'solve_force_program',reject)


def check(c,a,rhs,tmp_path,name,*,rational=False):
    solver=EquilibriumSchurSolver(c.layout,a); answer=solver.solve(rhs)
    free=c.layout.free; expected=np.zeros_like(rhs)
    expected[free]=np.linalg.solve(a[np.ix_(free,free)],rhs[free])
    error=float(np.linalg.norm(answer.increment-expected)/max(np.linalg.norm(expected),np.finfo(float).tiny))
    assert error<=1e-11 and answer.backward_error<=1e-11
    assert not answer.full_mixed_fallback and not answer.production_qualified
    row=dict(answer=answer,direct_solve_difference=error,diagnostics=solver.diagnostics(),matrix=a,rhs=rhs)
    if rational:
        values=exact_solve(a[np.ix_(free,free)].tolist(),rhs[free].tolist())
        oracle=np.array([float(v) for v in values]); row['exact_solution']=[str(v) for v in values]
        exact_error=float(np.linalg.norm(answer.increment[free]-oracle)/np.linalg.norm(oracle))
        assert exact_error<=1e-11; row['exact_relative_difference']=exact_error
    save(tmp_path/name,row)
    np.testing.assert_array_equal(solver.solve(rhs).increment,answer.increment)
    return solver


@pytest.mark.parametrize('case',['straight-elastic','curved-elastic','plastic-cycle'])
def test_initial_homogeneous_equilibrium_without_full_fallback(case,tmp_path):
    made,p,f=inputs(case); c=controller.Context(made,p,line_forces=f)
    r,h,_,_=c.assemble(c.initial.mechanical,p.targets[0],c.initial.histories); a=c.tangent(r,h,p.targets[0])
    with pytest.raises(ValueError,match='recovered full'): Previous(c.layout,a).solve(-r)
    check(c,a,-r,tmp_path,'initial.json',rational=case=='curved-elastic')


@pytest.mark.parametrize('case',['straight-elastic','curved-elastic','plastic-cycle'])
def test_preserved_accepted_state_and_nonstationary_trial(case,tmp_path):
    path='cycle-a/pytest/'+case+'0/schur.json'
    item=next(r for r in json.loads(MANIFEST.read_bytes())['files'] if r['path']==path)
    raw=(ARCHIVE/path).read_bytes(); assert len(raw)==item['bytes'] and sha256(raw).hexdigest()==item['sha256']
    made,p,f=inputs(case); c=controller.Context(made,p,line_forces=f)
    _,records=c.restore(raw,expected_sha256=item['sha256'])
    record=json.loads(records[1]); state=c.layout.make(record['mechanical'],decoded=True)
    origins=c.history(record['origins'],decoded=True); parameter=record['parameter']
    # A fresh local trial, not a nonlinear load-path solve.
    step=np.zeros(c.layout.count); step[c.layout.free]=.001*np.sin(np.arange(len(c.layout.free)))
    trial=c.layout.advance(state,step)
    r,h,_,_=c.assemble(trial,parameter,origins)
    check(c,c.tangent(r,h,parameter),-r,tmp_path,'trial.json')
    assert c.checkpoint(records)==raw


@pytest.mark.parametrize('both_ends',[False,True])
def test_multielement_and_statically_indeterminate_compatibility_completion(both_ends,tmp_path):
    m=curved_model(2)
    if both_ends: m.add_boundary_condition(BoundaryCondition('tip',[5],{k:0. for k in ('ux','uy','uz','rx','ry','rz')}))
    p=ForceProgram((.3,)); f=ReferenceLineForces(((1,.1,-.04,.02),(2,.1,-.04,.02)))
    c=physical.Context(m,p,line_forces=f)
    r,h,_,_=c.assemble(c.initial.mechanical,.3,c.initial.histories); a=c.tangent(r,h,.3)
    rhs=np.column_stack([-r,np.cos(np.arange(c.layout.count))])
    solver=check(c,a,rhs,tmp_path,'multi.json')
    assert solver.diagnostics()['compatibility_completion_rows']==(6 if both_ends else 0)
    assert solver.diagnostics()['force_recovery_coordinates']==36


def test_exact_oracle_and_ownership_guards():
    assert exact_solve([[2,1],[1,3]],[1,2])==( __import__('fractions').Fraction(1,5),__import__('fractions').Fraction(3,5))
    with pytest.raises(ValueError): exact_solve([[0,0],[0,0]],[0,0])
    with pytest.raises(ValueError): exact_solve([[True]],[1])
    made,p,f=inputs('curved-elastic'); c=controller.Context(made,p,line_forces=f)
    r,h,_,_=c.assemble(c.initial.mechanical,p.targets[0],c.initial.histories)
    solver=EquilibriumSchurSolver(c.layout,c.tangent(r,h,p.targets[0]))
    with pytest.raises(ValueError): solver.recovery_matrix.setflags(write=True)
    with pytest.raises(AttributeError): solver.recovery_rows=()
    with pytest.raises(ValueError): solver.solve(np.ones(len(r),dtype=bool))
    c.layout.free=c.layout.free[:-1]
    with pytest.raises(ValueError,match='layout changed'): solver.solve(-r)
