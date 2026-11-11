"""Independent chord/AD reconstruction; not continuation qualification."""

import math

import numpy as np
import pytest

from anyfesolver._ge_beam3_arc_geometry import frame_chord_terms, constraint
from anyfesolver._ge_beam3_p5.algebra import rotation, skew
from anyfesolver._ge_beam3_p5.chart import exp_chart_terms


def direct(v,w,q0):
    return .5*float(np.sum(((rotation(v)-np.eye(3))@q0)*(skew(w)@q0)))


@pytest.mark.parametrize('magnitude',[0.,2.**-54,1e-8,1e-4,.000999999,.001000001,.1,1.3,2.7])
def test_chord_and_chart_row_match_independent_exp_jet(magnitude):
    from anyfesolver._ge_beam3_mixed_ad import Jet2, so3_exp
    axis = np.array([1.,-2.,3.]); axis /= np.linalg.norm(axis)
    v = magnitude*axis; w = np.array([.2,.6,-.5])
    q = so3_exp([Jet2.variable(float(x),i,3) for i,x in enumerate(v)])
    # Frobenius reconstruction from Exp Jets, not the sinc/beta formulas.
    h = skew(w); oracle = Jet2.constant(0.,3)
    for i in range(3):
        for j in range(3): oracle += .5*h[i,j]*(q[i][j]-float(i == j))
    value,gradient = frame_chord_terms(v,w)
    assert abs(value-oracle.value) <= 1e-14
    np.testing.assert_allclose(gradient,oracle.gradient,rtol=0.,atol=1e-13)
    assert not gradient.flags.writeable


def test_tiny_chord_survives_dense_frame_subtraction_loss():
    v = np.array([2.**-54,0.,0.]); w = np.array([1.,0.,0.])
    q0 = rotation(np.array([.7,-.9,.4]))
    lost = .5*float(np.sum((rotation(v)@q0-q0)*(skew(w)@q0)))
    value,row = frame_chord_terms(v,w)
    assert value == 2.**-54 and value != lost
    np.testing.assert_array_equal(row,w)


@pytest.mark.parametrize('seed',range(4))
def test_chord_matches_frame_hyperplane_and_spatial_pullback(seed):
    from docs.reference_cases.ge_beam3_curved_p5_continuation_probe import frame_constraint
    rng = np.random.default_rng(seed)
    v = rng.normal(size=3)*.3; w = rng.normal(size=3)
    q0 = rotation(rng.normal(size=3)*.5); q = rotation(v)@q0
    d = np.r_[np.zeros(3),w]; weights = np.ones(6)
    original,spatial = frame_constraint((np.zeros((1,3)),q[None]),(np.zeros((1,3)),q0[None]),d,weights,np.arange(6))
    value,row = frame_chord_terms(v,w)
    assert abs(value-original) <= 1e-14
    np.testing.assert_allclose(row,exp_chart_terms(v)[0].T@spatial[3:],atol=1e-13,rtol=0.)
    # A raw spatial gradient is not the native increment-coordinate row.
    assert np.linalg.norm(row-spatial[3:]) > 1e-4


def test_common_rigid_rotation_preserves_value_and_rotates_gradient():
    v = np.array([.2,-.1,.4]); w = np.array([-.5,.8,.6])
    s = rotation(np.array([1.2,-.3,.4]))
    value,row = frame_chord_terms(v,w); moved,moved_row = frame_chord_terms(s@v,s@w)
    assert abs(value-moved) <= 1e-14
    np.testing.assert_allclose(moved_row,s@row,atol=1e-13,rtol=0.)
    for q0 in (np.eye(3),rotation(np.array([.5,.2,-.7]))):
        assert abs(value-direct(v,w,q0)) <= 1e-14


def specimen():
    v = np.array([[.02,-.01,.03,.2,-.1,.4],[-.01,.04,-.02,-.3,.1,.2]])
    d = np.array([[.5,-.2,.1,.3,.1,-.4],[.2,.3,-.5,-.1,.2,.6]])
    weights = np.array([[.5,.5,.5,2.,2.,2.],[1.,1.,1.,3.,3.,3.]])
    return v,d,weights


def test_complete_hyperplane_matches_finite_difference_and_rigid_frame_change():
    v,d,weights = specimen()
    value,row = constraint(v,d,weights,.02,.3,2.,.1)
    direction = np.linspace(-.3,.2,13); epsilon = 1e-6
    plus = constraint(v+epsilon*direction[:-1].reshape(2,6),d,weights,float(.02+epsilon*direction[-1]),.3,2.,.1)[0]
    minus = constraint(v-epsilon*direction[:-1].reshape(2,6),d,weights,float(.02-epsilon*direction[-1]),.3,2.,.1)[0]
    assert abs((plus-minus)/(2*epsilon)-row@direction) <= 1e-7
    s = rotation(np.array([.7,-.4,.5]))
    rotate = lambda a: np.c_[a[:,:3]@s.T,a[:,3:]@s.T]
    changed,changed_row = constraint(rotate(v),rotate(d),weights,.02,.3,2.,.1)
    assert abs(value-changed) <= 1e-13
    np.testing.assert_allclose(changed_row[:-1].reshape(2,6),rotate(row[:-1].reshape(2,6)),atol=1e-13,rtol=0.)
    assert changed_row[-1] == row[-1]


def test_zero_increment_row_is_the_metric_predictor():
    _,d,weights = specimen()
    value,row = constraint(np.zeros((2,6)),d,weights,0.,.3,2.,.1)
    assert value == -.1
    np.testing.assert_array_equal(row,np.r_[(d*weights).ravel(),.6])


def test_finite_rotation_chord_is_not_accumulated_vector_dot_product():
    v = np.array([1.2,0.,0.]); w = np.array([1.,0.,0.])
    value,row = frame_chord_terms(v,w)
    assert value == math.sin(1.2) and abs(value-v@w) > .2
    assert abs(row[0]-math.cos(1.2)) <= 1e-14


@pytest.mark.parametrize('kind',['rotation_bound','nonfinite','anisotropic','anisotropic_translation','negative','unmasked','zero_step','implicit_scalar'])
def test_unregistered_constraint_inputs_fail_closed(kind):
    v,d,weights = specimen(); step = .1
    if kind == 'rotation_bound': v[0,3:] = [.9*math.pi,0.,0.]
    elif kind == 'nonfinite': v[0,0] = np.nan
    elif kind == 'anisotropic': weights[0,3] = 1.
    elif kind == 'anisotropic_translation': weights[0,0] = 1.
    elif kind == 'negative': weights[0,0] = -1.
    elif kind == 'unmasked': weights[0,:] = 0.
    elif kind == 'zero_step': step = 0.
    elif kind == 'implicit_scalar': step = 1
    with pytest.raises(ValueError): constraint(v,d,weights,.02,.3,2.,step)


def test_zero_predictor_does_not_create_an_unsolvable_constant_constraint():
    v,_,weights = specimen()
    with pytest.raises(ValueError,match='nonzero metric predictor'):
        constraint(v,np.zeros_like(v),weights,.02,0.,2.,.1)


def test_fixed_translation_or_rotation_triplets_contribute_zero():
    v,d,weights = specimen(); v[0,:3] = 0.; d[0,:3] = 0.
    v[1,3:] = 0.; d[1,3:] = 0.
    a,row = constraint(v,d,weights,.02,.3,2.,.1)
    weights[0,:3] = 0.; weights[1,3:] = 0.
    b,masked_row = constraint(v,d,weights,.02,.3,2.,.1)
    assert a == b
    np.testing.assert_array_equal(row,masked_row)
