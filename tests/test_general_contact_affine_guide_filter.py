"""Trial filtering cannot relax final contact/control/equilibrium acceptance."""
import numpy as np
import pytest
from anysolver.general_contact.corner_contact_solver import (
    Evaluation, solve_ncp, _affine_guide_progress, NewtonFailure, GAP_TOL_M,
)


class GuidedNonlinearContact:
    affine_guide_filter = True
    def __init__(self):
        self.trials = []
    def evaluate(self,q,mu,lam):
        force = lam.get(0,0.)
        return Evaluation(np.array([-force-mu[0],100*q[1]+1e12*q[1]**3-force]),
                          np.diag([0.,100+3e12*q[1]**2]),[0],np.array([q.sum()]),
                          np.ones((1,2)),np.array([q[0]+.00025]),np.array([[1.,0.]]))
    def evaluate_trial(self,q,mu,lam):
        self.trials.append((q.copy(),mu.copy(),dict(lam)))
        return self.evaluate(q,mu,lam)


def test_filter_reaches_exact_guide_before_equilibrating_nonlinear_contact():
    p = GuidedNonlinearContact()
    result = solve_ncp(p,np.zeros(2),np.zeros(1),{},f_ref=50.,force_tolerance=5e-7)
    np.testing.assert_allclose(result['q'],[-.00025,.00025],rtol=0,atol=1e-12)
    expected = 100*.00025+1e12*.00025**3
    assert result['lam'][0] == pytest.approx(expected,rel=1e-10)
    assert abs(result['ev'].guide_error[0]) <= 1e-12
    assert np.linalg.norm(result['ev'].residual) <= 5e-7
    assert abs(result['ev'].gaps[0]) <= 1e-12
    # The first full control step initially raises force merit by orders of
    # magnitude, then the next step equilibrates on the fixed control manifold.
    assert p.trials[0][0][0] == pytest.approx(-.00025,abs=1e-12)
    assert all(abs(q[0]+.00025)<=1e-12 for q,_,_ in p.trials)


def test_affine_declaration_and_feasible_guide_are_checked_not_assumed():
    p = GuidedNonlinearContact()
    ev = p.evaluate(np.zeros(2),np.zeros(1),{})
    bad = p.evaluate(np.array([-.0001,0.]),np.zeros(1),{})
    with pytest.raises(NewtonFailure) as exc:
        _affine_guide_progress(ev,bad,1.)
    assert exc.value.code == 'nonaffine_guides'
    feasible = p.evaluate(np.array([-.00025,.00025]),np.zeros(1),{})
    assert _affine_guide_progress(feasible,feasible,1.) == (False,True)


def test_guide_progress_cannot_admit_nonfinite_force_residual():
    class Broken(GuidedNonlinearContact):
        def evaluate_trial(self,q,mu,lam):
            ev = super().evaluate_trial(q,mu,lam)
            ev.residual[:] = np.nan
            return ev
    with pytest.raises(NewtonFailure) as exc:
        solve_ncp(Broken(),np.zeros(2),np.zeros(1),{},f_ref=50.,force_tolerance=5e-7)
    assert exc.value.code == 'line_search_failed'


def test_failed_scientific_gate_is_preserved_without_increment_retries():
    from types import SimpleNamespace
    from anysolver.general_contact.corner_contact_indentation import CoupledIndentation, GateFailure
    calls = []
    def reject(target):
        calls.append(target)
        raise GateFailure('fixture_gate','unchanged acceptance refused',dict(measured=2.,bound=1.))
    sim = SimpleNamespace(solve_to=reject,bisections_left=4,target=np.array([0.,0.,1.]))
    with pytest.raises(GateFailure) as exc:
        CoupledIndentation.advance(sim,.5)
    assert exc.value.code == 'fixture_gate' and calls == [.5]
    assert sim.bisections_left == 4
    assert sim.last_gate_failure == dict(code='fixture_gate',detail=dict(measured=2.,bound=1.),target_z=.5)
