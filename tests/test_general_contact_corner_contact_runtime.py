"""Focused runtime-path backstops; the frozen mechanical fixtures stay unchanged."""
import numpy as np
import pytest
from types import SimpleNamespace
from test_general_contact_corner_contact import Spring
from anyfesolver.general_contact.corner_contact_solver import solve_ncp
from anyfesolver.general_contact.corner_contact_certificate import CornerContactRefusal
from anyfesolver.general_contact.corner_contact_indentation import CoupledIndentation


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
    from anyfesolver.general_contact.corner_contact_solver import Cancelled
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
    import anyfesolver.general_contact.corner_contact_indentation as module
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


def _ev(**kw):
    from anyfesolver.general_contact.corner_contact_solver import Evaluation
    base=dict(residual=np.zeros(2),stiffness=None,ids=[],gaps=np.zeros(0),
              gradients=np.zeros((0,2)),guide_error=np.zeros(0),guide_rows=np.zeros((0,2)))
    base.update(kw)
    if 'gradients' not in kw:base['gradients']=np.zeros((len(base['ids']),2))
    return Evaluation(**base)


@pytest.mark.parametrize('field',['residual','guide_error','gap','multiplier','force_scale',
                                  'force_scale_inf'])
def test_non_finite_state_is_never_converged(field):
    """Review finding (High): NaN fails every '>' test, so it must be refused explicitly."""
    from anyfesolver.general_contact.corner_contact_solver import converged,NewtonFailure
    kw,lam={},{}
    if field=='residual':kw['residual']=np.array([0.,np.nan])
    elif field=='guide_error':kw['guide_error']=np.array([np.nan])
    elif field=='gap':kw.update(ids=[3],gaps=np.array([np.nan]))
    elif field=='multiplier':kw.update(ids=[3],gaps=np.array([0.]));lam={3:np.nan}
    elif field=='force_scale':kw['force_scale']=np.nan
    else:kw.update(force_scale=np.inf,residual=np.array([1e300,0.]))
    with pytest.raises(NewtonFailure) as caught:
        converged(_ev(**kw),lam,1.,1.,1e-6)
    assert caught.value.code=='non_finite_state'
    # Ordinary finite states keep their previous verdicts.
    assert converged(_ev(),{},1.,1.,1e-6) is True
    assert converged(_ev(residual=np.array([1.,0.])),{},1.,1.,1e-6) is False


@pytest.mark.parametrize('value',[np.nan,np.inf])
def test_balance_gate_refuses_non_finite_wrench_norms(value):
    import anyfesolver.general_contact.corner_contact_indentation as module
    from examples.elastic_sphere_stiffened_panel import FORCE_BALANCE_TOLERANCE as F,MOMENT_BALANCE_TOLERANCE as M
    finite={'sphere':(0.,0.),'panel':(0.,0.)}
    assert module._balance_exceeded(finite,F,M) is False
    assert module._balance_exceeded({**finite,'panel':(value,0.)},F,M) is True
    assert module._balance_exceeded({**finite,'sphere':(0.,value)},F,M) is True
    assert module._balance_exceeded({**finite,'panel':(2*F,0.)},F,M) is True
    assert module._balance_exceeded({**finite,'panel':(F,M)},F,M) is False   # bound is inclusive, as before


def test_cancellation_during_acceptance_gates_does_not_commit(monkeypatch):
    """Review finding: a cancel raised while the gates run must stop before commit."""
    import anyfesolver.general_contact.corner_contact_indentation as module
    from anyfesolver.general_contact.corner_contact_solver import Cancelled
    target=np.arange(6,dtype=float);commits=[];flag=[]
    sim=SimpleNamespace(q=np.array([1.,2.]),mu=np.array([3.]),lam={7:4.},
        target=target.copy(),_trial_target=target.copy(),f_ref=1.,force_tolerance=1e-6,
        cancel=lambda:bool(flag),max_iterations=20,
        bridge=SimpleNamespace(commit=lambda *args:commits.append(args)))
    sim._initial_guess=lambda:sim.q.copy()
    result=dict(q=np.array([5.,6.]),mu=np.array([7.]),lam={7:8.},
                ev=SimpleNamespace(extra=dict(state=object(),histories=())))
    monkeypatch.setattr(module,'solve_ncp',lambda *args,**kwargs:result)
    def gates(result):
        flag.append(1)                      # cancel arrives while certificates run
        return None,{},None,None,None
    sim._gates=gates
    with pytest.raises(Cancelled):
        CoupledIndentation.solve_to(sim,9.)
    assert not commits
    np.testing.assert_array_equal(sim.q,[1.,2.])
    np.testing.assert_array_equal(sim.target,target)
    np.testing.assert_array_equal(sim._trial_target,target)


def test_committed_bisection_records_survive_a_later_refusal(monkeypatch):
    """Review finding: an accepted midpoint must stay in the report when the rest refuses."""
    import anyfesolver.general_contact.corner_contact_indentation as module
    from anyfesolver.general_contact.corner_contact_solver import NewtonFailure
    sim=SimpleNamespace(target=np.zeros(6),bisections_left=1)
    def solve_to(z):
        if z==1.:raise NewtonFailure('fixture','full step refuses')
        if z==0.5:
            sim.target[2]=z
            return dict(z=z)
        raise AssertionError(z)
    sim.solve_to=solve_to
    accepted=[]
    with pytest.raises(NewtonFailure):
        CoupledIndentation.advance(sim,1.,accepted)
    assert accepted==[dict(z=0.5)] and sim.target[2]==0.5


def test_run_cycle_reports_accepted_increments_before_a_refusal(monkeypatch):
    import anyfesolver.general_contact.corner_contact_indentation as module
    from anyfesolver.general_contact.corner_contact_solver import NewtonFailure
    class Fake:
        f_ref=1.;penetration_limit=1.;force_tolerance=1.;bisections_left=module.MAX_BISECTIONS
        z_open=1.
        def __init__(self,case):self.target=np.zeros(6);self.target[2]=1.;self.q=np.zeros(1)
        def prepare_open(self):return dict(open=True)
        def first_touch_height(self):return 0.,[],[]
        def advance(self,z,records=None):
            records=[] if records is None else records
            records.append(dict(z=z));self.target[2]=z
            raise NewtonFailure('fixture','refused after one accepted midpoint')
    monkeypatch.setattr(module,'CoupledIndentation',Fake)
    report,_=module.run_cycle(case=object())
    assert report['status']==module.STATUS_REFUSED
    assert [r['z'] for r in report['increments']]==[0.] and report['increments'][0]['stage']=='to_first_touch'


def test_acceptance_gate_failure_is_not_retried_or_bisected():
    from anyfesolver.general_contact.corner_contact_indentation import GateFailure
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
