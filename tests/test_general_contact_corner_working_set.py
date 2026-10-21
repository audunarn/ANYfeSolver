"""Linearized contact sets must satisfy compression before nonlinear line search."""
import numpy as np
import pytest
from anysolver.general_contact.corner_contact_solver import Evaluation, solve_ncp, NewtonFailure


class NearParallelContact:
    def __init__(self):
        self.G = np.array([[1.,0.,0.],[0.,1.,0.],[.5,.5,1e-4]])
        self.b = np.array([0.,0.,1e-6])
        self.load = np.array([-1.,-2.,0.])
        self.trials = []
    def evaluate(self,q,mu,lam):
        values=np.array([lam.get(i,0.) for i in range(3)])
        return Evaluation(q-self.load-self.G.T@values,np.eye(3),list(range(3)),
                          self.b+self.G@q,self.G,np.zeros(0),np.zeros((0,3)))
    def evaluate_trial(self,q,mu,lam):
        self.trials.append((q.copy(),dict(lam)))
        return self.evaluate(q,mu,lam)


def test_overclosed_near_parallel_corner_is_removed_when_it_requires_tension():
    problem=NearParallelContact()
    q=np.array([-.1,-.1,0.]); before=q.copy()
    result=solve_ncp(problem,q,np.zeros(0),{},f_ref=3.,force_tolerance=3e-8)
    np.testing.assert_array_equal(q,before)
    np.testing.assert_allclose(result['q'],0.,rtol=0,atol=1e-12)
    assert result['lam']==pytest.approx({0:1.,1:2.},rel=1e-8)
    assert result['ev'].gaps[2]==pytest.approx(1e-6,abs=1e-12)
    assert len(problem.trials)==1
    assert all(v>=0. for _,lam in problem.trials for v in lam.values())
    assert abs(problem.trials[0][0][2])<=1e-12


def test_inconsistent_linearized_contact_cycles_refuse_before_any_trial():
    class Impossible:
        def evaluate(self,q,mu,lam):
            return Evaluation(-q-1.-lam.get(0,0.),np.array([[-1.]]),[0],q.copy(),
                              np.ones((1,1)),np.zeros(0),np.zeros((0,1)))
        def evaluate_trial(self,*args):
            pytest.fail('inconsistent direction must not reach nonlinear trial')
    q=np.ones(1)
    with pytest.raises(NewtonFailure) as caught:
        solve_ncp(Impossible(),q,np.zeros(0),{},f_ref=1.,force_tolerance=1e-8)
    assert caught.value.code=='working_set_cycle'
    np.testing.assert_array_equal(q,np.ones(1))
