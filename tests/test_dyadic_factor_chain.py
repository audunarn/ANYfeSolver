"""Unexpanded bilinear chains against separately coded exact Fraction products."""
from fractions import Fraction as F
import numpy as np
import pytest
from anyfesolver._dyadic_factor_chain import reassemble_chain_exact_binary64


def exact(a): return [[F(float(x)) for x in row] for row in a]
def transpose(a): return list(map(list,zip(*a)))
def product(a,b): return [[sum((x*y for x,y in zip(row,col)),F(0)) for col in zip(*b)] for row in a]


@pytest.mark.parametrize('seed',[12,39,71])
def test_factor_chain_congruence_matches_exact_rational_entries(seed):
    rng=np.random.default_rng(seed)
    l=rng.normal(size=(5,4));r=rng.normal(size=(4,3));b=rng.normal(size=(5,3));v=rng.normal(size=(3,2))
    g=rng.normal(size=(3,3));g=g+g.T
    h,m=reassemble_chain_exact_binary64(l,r,g,b,v)
    strain=product(exact(l),product(exact(r),exact(v)))
    material=product(transpose(strain),strain)
    geometric=product(transpose(exact(v)),product(exact(g),exact(v)))
    speed=product(exact(b),exact(v));mass=product(transpose(speed),speed)
    expected=np.array([[float(x+y) for x,y in zip(a,c)] for a,c in zip(material,geometric)])
    assert np.array_equal(h,expected) and np.array_equal(m,np.array(mass,dtype=float))


def test_chain_cancellation_retains_work_that_expansion_loses():
    l=np.array([[1e16,1.],[1e16,0.]])
    r=np.array([[1.,1.],[0.,1.]])
    v=np.array([[-1.],[1.]])
    assert np.array_equal((l@r)@v,np.zeros((2,1)))
    h,m=reassemble_chain_exact_binary64(l,r,np.zeros((2,2)),np.eye(2),v)
    assert np.array_equal(h,[[1.]]) and np.array_equal(m,[[2.]])


@pytest.mark.parametrize('bad',['shape','nan','empty','checkpoint'])
def test_chain_validation_and_cancellation_are_not_bypassed(bad):
    l=r=b=v=np.eye(2);g=np.zeros((2,2))
    def checkpoint():
        if bad=='checkpoint': raise RuntimeError('cancel')
    if bad=='shape':r=np.ones((3,2))
    elif bad=='nan':l=np.full((2,2),np.nan)
    elif bad=='empty':l=np.zeros((0,2))
    with pytest.raises((ValueError,RuntimeError)):
        reassemble_chain_exact_binary64(l,r,g,b,v,checkpoint)
