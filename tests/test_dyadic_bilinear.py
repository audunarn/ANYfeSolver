"""Independent Fraction reconstruction of the integer bilinear kernel."""
from fractions import Fraction as F
import numpy as np
import pytest
from anyfesolver._dyadic_bilinear import reassemble_exact_binary64 as reassemble


def dot(a,b): return sum((x*y for x,y in zip(a,b)),F())
def mm(a,b): return [[dot(row,col) for col in zip(*b)] for row in a]
def trans(a): return list(map(list,zip(*a)))
def rational(a): return [[F(float(x)) for x in row] for row in a]


@pytest.mark.parametrize('seed',[14,18,72])
def test_each_rounded_entry_equals_exact_fraction_congruence(seed):
    rng=np.random.default_rng(seed)
    f=rng.normal(size=(8,5)); b=rng.normal(size=(7,5)); v=rng.normal(size=(5,3))
    g=rng.normal(size=(5,5)); g=(g+g.T)/2
    actual_h,actual_m=reassemble(f,g,b,v)
    ff,bb,vv,gg=map(rational,(f,b,v,g))
    strain,speed=mm(ff,vv),mm(bb,vv)
    a=mm(trans(strain),strain); c=mm(trans(vv),mm(gg,vv))
    expected_h=np.array([[float(x+y) for x,y in zip(r,s)] for r,s in zip(a,c)])
    expected_m=np.array([[float(x) for x in row] for row in mm(trans(speed),speed)])
    np.testing.assert_array_equal(actual_h,expected_h)
    np.testing.assert_array_equal(actual_m,expected_m)


def test_large_cancellation_is_exact_before_output_rounding():
    a=float(2**40); f=np.vstack((a*np.array([[1.,1.],[0.,1.]]),np.eye(2)))
    g=-a*a*np.array([[1.,1.],[1.,2.]])
    v=np.array([[.1,.3],[.2,.4]])
    h,m=reassemble(f,g,np.eye(2),v)
    np.testing.assert_array_equal(h,m)


@pytest.mark.parametrize('kind',['nonfinite','shape','underflow','overflow'])
def test_invalid_or_unrepresentable_bilinear_result_fails_closed(kind):
    f=np.eye(2); b=np.eye(2); g=np.zeros((2,2)); v=np.eye(2)
    if kind=='nonfinite': f[0,0]=np.nan
    elif kind=='shape': v=np.ones((3,2))
    elif kind=='underflow': f[0,0]=1e-300
    else: f[0,0]=1e300
    with pytest.raises(ValueError): reassemble(f,g,b,v)


def test_checkpoint_interrupts_before_integer_work():
    calls=[]
    def stop():
        calls.append(1); raise RuntimeError('explicit stop')
    with pytest.raises(RuntimeError,match='explicit stop'):
        reassemble(np.eye(2),np.zeros((2,2)),np.eye(2),np.eye(2),stop)
    assert len(calls)==1
