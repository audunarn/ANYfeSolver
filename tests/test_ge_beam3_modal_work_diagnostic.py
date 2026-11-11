from decimal import Decimal as D
from fractions import Fraction as F
import ast
from pathlib import Path
import pytest
from docs.reference_cases.ge_beam3_modal_work_diagnostic import work_record,differences,factor_work
from docs.reference_cases.ge_beam3_modal_work_worker import source

def test_signed_work_mass_normalization_and_cancellation():
    r=work_record(D('3'),D('-4'),D('2'),D('-.5'))
    assert r['total_per_mass']==-.5 and r['cancellation_ratio']==7 and r['relative_rayleigh_error']==0
    a=work_record(D('5'),D('-3'),D('4'),D('.5'))
    assert a['total_per_mass']==.5 and a['material_per_mass']==1.25
@pytest.mark.parametrize('args',[(1,1,0,1),(-1,2,1,1),(1,float('nan'),1,1),(1,-1,1,1),(1,1,1,0),(0,1,1,1)])
def test_invalid_modal_work(args):
    with pytest.raises(ValueError):work_record(*args)
def test_fraction_factor_work_independent_of_decimal_record():
    q=[F(1,3),F(2,7)];left=[[F(2),F(0)],[F(1),F(3)]];right=[[F(1),F(2)],[F(0),F(1)]]
    mv=lambda a,x:[sum(v*y for v,y in zip(row,x)) for row in a]
    f=mv(left,mv(right,q));material=sum(v*v for v in f);geometric=-q[0]*q[0];mass=sum(v*v for v in q)
    r=work_record(material,geometric,mass,(material+geometric)/mass)
    assert r['relative_rayleigh_error']==0
def test_actual_sparse_factor_work_against_fraction():
    # Dyadic inputs are exactly representable in both implementations.
    q=[F(1,2),F(3,4)];l=[[F(2),F(0)],[F(1),F(3)]];r=[[F(1),F(2)],[F(0),F(1)]]
    g=[[F(-2),F(1,2)],[F(1,2),F(1)]];b=[[F(1),F(0)],[F(0),F(2)]]
    mv=lambda a,x:[sum(v*y for v,y in zip(row,x)) for row in a]
    expected=(sum(x*x for x in mv(l,mv(r,q))),sum(x*y for x,y in zip(q,mv(g,q))),sum(x*x for x in mv(b,q)))
    cv=lambda x:D(x.numerator)/D(x.denominator)
    sparse=lambda a:[{j:cv(v) for j,v in enumerate(row) if v} for row in a]
    assert factor_work(*[sparse(a) for a in (l,r,g,b)],[cv(v) for v in q])==tuple(map(cv,expected))
    def fail():raise ValueError('deadline')
    with pytest.raises(ValueError,match='deadline'):factor_work(*[sparse(a) for a in (l,r,g,b)],[cv(v) for v in q],fail)
def test_additive_difference_and_mutation():
    n=work_record(3.,-2.,1.,1.);r=work_record(2.,-1.5,1.,.5)
    values=differences([n]*6,[r]*6)
    assert values[0]['total_difference']==.5
    bad=dict(n,total_per_mass=9.)
    with pytest.raises(ValueError):differences([bad]*6,[r]*6)
    with pytest.raises(ValueError):differences([n]*5,[r]*5)
def test_saved_input_and_no_solver_calls():
    for sign in ('plus','minus'):assert source(sign)['comparison']['signed_rate_errors'][2]>.02
    root=Path(__file__).parents[1]/'docs/reference_cases'
    for name in ('ge_beam3_modal_work_diagnostic.py','ge_beam3_modal_work_worker.py'):
        tree=ast.parse((root/name).read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any(x.startswith('anyfesolver') for x in imports)
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        assert not set(calls)&{'eigh','eig','solve_bvp','prepare'}
