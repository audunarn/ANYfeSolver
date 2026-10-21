"""Proof-refined closest-face work and linearized blocking-contact regressions."""
from fractions import Fraction
import numpy as np
import pytest
from test_general_contact_corner_contact import make_panel, gap
from anysolver.general_contact.corner_contact_geometry import (
    _face_candidates, _face_minimum_bounds,
)
from anysolver.general_contact.corner_contact_solver import Evaluation, solve_ncp


def test_distinct_edge_candidates_do_not_hide_a_provably_unique_face_minimum():
    panel, pose = make_panel(lambda x, y: 0.)
    x = np.array((1.+1e-5, .5, .3))
    info = gap(x,panel,pose)
    assert info['face'] == 1 and info['owners'] == [1]
    assert info['gap'] == pytest.approx(.3,abs=1e-14)
    np.testing.assert_allclose(info['direction'],(0,0,1),atol=1e-14)
    # Shared structural nodes move both physical face copies together.
    h = 1e-6
    for node in {n for face in panel.faces for n in face.node_ids}:
        plus, minus = pose.positions.copy(), pose.positions.copy()
        for j,face in enumerate(panel.faces):
            for k,n in enumerate(face.node_ids):
                if n == node:
                    plus[4*j+k,2] += h
                    minus[4*j+k,2] -= h
        pp = type(pose)(positions=plus,director_gradient=pose.director_gradient)
        pm = type(pose)(positions=minus,director_gradient=pose.director_gradient)
        fd = (gap(x,panel,pp)['gap']-gap(x,panel,pm)['gap'])/(2*h)
        expected = sum(-info['weights'][k] for k,n in enumerate(panel.faces[1].node_ids)
                       if n == node)
        assert fd == pytest.approx(expected,abs=1e-8)


def test_rational_face_bounds_enclose_known_plane_distance_and_same_face_edges():
    panel, pose = make_panel(lambda x,y:0.,nx=1)
    x = np.array((1e-5,.5,.3))
    candidates = [(d,0,xi,eta) for d,xi,eta in _face_candidates(x,pose.positions)]
    upper,lower,row = _face_minimum_bounds(x,pose.positions,candidates)
    exact = Fraction(float(.3)) ** 2 / 2
    assert lower <= exact <= upper
    assert gap(x,panel,pose)['gap'] == pytest.approx(.3,abs=1e-14)


def test_working_set_anticipates_closing_contact_without_first_overclosing_it():
    class GuidedSpring:
        trial_gaps = []
        def evaluate(self,q,mu,lam):
            force = lam.get(0,0.)
            return Evaluation(np.array([-force-mu[0],100*q[1]-force]),
                              np.diag([0.,100.]),[0],np.array([1e-8+q.sum()]),
                              np.ones((1,2)),np.array([q[0]+.00025]),np.array([[1.,0.]]))
        def evaluate_trial(self,q,mu,lam):
            ev = self.evaluate(q,mu,lam)
            self.trial_gaps.append(ev.gaps[0])
            return ev
    p = GuidedSpring()
    result = solve_ncp(p,np.zeros(2),np.zeros(1),{},f_ref=50.,force_tolerance=5e-7)
    assert result['q'][0] == pytest.approx(-.00025,abs=1e-12)
    assert result['q'][1] == pytest.approx(.00025-1e-8,abs=1e-12)
    assert result['lam'][0] == pytest.approx(100*(.00025-1e-8),rel=1e-10)
    assert min(p.trial_gaps) >= -1e-12
