"""Focused runtime-path backstops; the frozen mechanical fixtures stay unchanged."""
import numpy as np
import pytest
from types import SimpleNamespace
from test_general_contact_corner_contact import Spring
from anysolver.general_contact.corner_contact_solver import solve_ncp
from anysolver.general_contact.corner_contact_certificate import CornerContactRefusal
from anysolver.general_contact.corner_contact_indentation import CoupledIndentation


def test_residual_only_trials_rebuild_tangent_only_if_another_step_is_needed():
    class TrialSpring(Spring):
        full = 0
        trials = 0

        def evaluate(self, q, mu, lam):
            self.full += 1
            ev = super().evaluate(q, mu, lam)
            ev.residual[0] += 100. * q[0] ** 3
            ev.stiffness[0, 0] += 300. * q[0] ** 2
            return ev

        def evaluate_trial(self, q, mu, lam):
            self.trials += 1
            ev = self.evaluate(q, mu, lam)
            self.full -= 1
            ev.stiffness = None
            return ev

    problem = TrialSpring()
    result = solve_ncp(problem, np.zeros(1), np.zeros(0), {},
                       f_ref=50., force_tolerance=5e-7)
    assert result['q'][0] == pytest.approx(-0.3, abs=1e-12)
    assert result['lam'][0] == pytest.approx(17.3, rel=1e-10)
    assert problem.full == 2 and problem.trials == 2
    # A converged residual-only trial needs no unused final tangent.
    assert result['ev'].stiffness is None


def test_inadmissible_trial_pose_refuses_before_any_structural_assembly():
    called = []

    def refuse(u):
        raise CornerContactRefusal('invalid_pose', 'fixture pose refusal')

    def response(*a, **kw):
        called.append(True)
        raise AssertionError('assembly must not precede pose admission')

    sim = SimpleNamespace(bridge=SimpleNamespace(expand=lambda q: q.copy(), response=response,
                                                 _owner=object()),
                          n_s=1, sphere_map=SimpleNamespace(pose=refuse))
    sim._contact_geometry = lambda full, *, multipliers=None, return_audit=False: refuse(full[:1])
    with pytest.raises(CornerContactRefusal) as caught:
        CoupledIndentation.evaluate(sim, np.zeros(2), np.zeros(0), {})
    assert caught.value.code == 'invalid_pose' and not called

def test_inflight_cancellation_is_observed_before_line_search_exhaustion():
    from anysolver.general_contact.corner_contact_solver import Cancelled
    problem = Spring()
    calls = []
    def cancel():
        calls.append(1)
        return len(calls)>=2
    q = np.zeros(1)
    with pytest.raises(Cancelled):
        solve_ncp(problem,q,np.zeros(0),{},f_ref=50.,force_tolerance=5e-7,cancel=cancel)
    assert len(calls)==2
    np.testing.assert_array_equal(q,np.zeros(1))


def test_failed_acceptance_gate_in_solve_to_preserves_accepted_state(monkeypatch):
    import anysolver.general_contact.corner_contact_indentation as module
    target = np.arange(6, dtype=float)
    commits = []
    sim = SimpleNamespace(
        q=np.array([1., 2.]), mu=np.array([3.]), lam={7:4.},
        target=target.copy(), _trial_target=target.copy(),
        f_ref=1., force_tolerance=1e-6, cancel=None, max_iterations=20,
        bridge=SimpleNamespace(commit=lambda *args:commits.append(args)))
    sim._initial_guess=lambda:sim.q.copy()
    result=dict(q=np.array([5.,6.]),mu=np.array([7.]),lam={7:8.},
                ev=SimpleNamespace(extra={}))
    monkeypatch.setattr(module,'solve_ncp',lambda *args,**kwargs:result)
    def refuse(result):
        raise module.GateFailure('whole_face_certificate','fixture refusal')
    sim._gates=refuse
    with pytest.raises(module.GateFailure):
        CoupledIndentation.solve_to(sim,9.)
    np.testing.assert_array_equal(sim.q,[1.,2.])
    np.testing.assert_array_equal(sim.mu,[3.])
    np.testing.assert_array_equal(sim.target,target)
    np.testing.assert_array_equal(sim._trial_target,target)
    assert sim.lam=={7:4.} and not commits


def test_acceptance_gate_failure_is_not_retried_or_bisected():
    from anysolver.general_contact.corner_contact_indentation import GateFailure
    calls=[]
    sim=SimpleNamespace(target=np.zeros(6),bisections_left=4)
    def refuse(z):
        calls.append(z)
        raise GateFailure('whole_face_certificate','fixture refusal',detail='fixture refusal')
    sim.solve_to=refuse
    with pytest.raises(GateFailure):
        CoupledIndentation.advance(sim,1.)
    assert calls==[1.] and sim.bisections_left==4
    assert sim.last_gate_failure==dict(code='whole_face_certificate',
                                      detail='fixture refusal',target_z=1.)
