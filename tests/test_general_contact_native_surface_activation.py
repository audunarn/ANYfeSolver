"""Independent scalar first-touch oracle and complete mixed-work derivatives."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from anyfesolver.general_contact.native_surface_activation import evaluate_first_touch
from test_general_contact_native_surface_s3_friction import _case


def _approach(slide):
    _,pair,start,history=_case()
    positions=start.positions.copy();positions[:,2]=0.
    start=replace(start,positions=positions)
    translation=np.tile((slide,0.,-.02),(3,1))
    angular=np.array(((0.,.05,0.),(0.,0.,0.)))
    return pair,start,history,translation,angular,np.zeros(18)


@pytest.mark.parametrize('slide,branch',[(.003,'stick'),(.02,'slip')])
def test_first_touch_has_full_endpoint_friction_force_and_physical_section_torque(slide,branch):
    pair,start,history,translation,angular,shell=_approach(slide)
    value=evaluate_first_touch(pair,start,history,translation,angular,shell)
    h,d,r,theta=.01,.02,.03,.05
    root=brentq(lambda a:h-d*a+r*(1-np.cos(theta*a)),0.,1.,xtol=1e-14)
    slip=slide*(1-root)-r*(np.sin(theta)-np.sin(theta*root))
    pressure=-1000*(h-d+r*(1-np.cos(theta)))
    traction=min(1000*slip,.3*pressure)
    assert value.event_fraction==pytest.approx(root,abs=2e-14)
    assert value.trial.branch==branch
    assert sum(value.residual[6*i] for i in range(3))==pytest.approx(.01*traction,rel=1e-10)
    # No tangential-law normal force from differentiating the event fraction.
    assert sum(value.residual[6*i+2] for i in range(3))==pytest.approx(-.01*pressure,rel=1e-10)
    assert value.residual[19]==pytest.approx(-.01*r*(pressure*np.sin(theta)+traction*np.cos(theta)),rel=1e-10)
    if branch=='stick':
        x=np.zeros(60);x[[0,6,12]]=1.
        z=np.zeros(60);z[[2,8,14]]=1.
        assert x@value.spatial_tangent()@x==pytest.approx(.01*1000*(1-root),rel=1e-10)
        assert abs(z@value.spatial_tangent()@x)<1e-12
        assert abs(x@value.spatial_tangent()@z)>1e-4
    assert history.partner_key is None


@pytest.mark.parametrize('slide',(.003,.02))
def test_first_touch_both_body_spatial_tangent_with_event_sensitivity(slide):
    pair,start,history,translation,angular,shell=_approach(slide)
    value=evaluate_first_touch(pair,start,history,translation,angular,shell)
    direction=np.random.default_rng(951).normal(size=60)
    def moved(step):
        t=translation+step*np.array([direction[6*i:6*i+3] for i in range(3)])
        w=np.array([(Rotation.from_rotvec(step*direction[18+3*i:21+3*i])
                     *Rotation.from_rotvec(angular[i])).as_rotvec() for i in range(2)])
        return evaluate_first_touch(pair,start,history,t,w,shell+step*direction[42:])
    for step in (1e-5,2e-6):
        plus,minus=moved(step),moved(-step)
        assert plus.trial.branch==minus.trial.branch==value.trial.branch
        fd=(plus.residual-minus.residual)/(2*step)
        assert np.linalg.norm(value.spatial_tangent()@direction-fd)/np.linalg.norm(fd)<=1e-5


def test_first_touch_rejects_unresolved_normal_path_and_existing_partner():
    pair,start,history,translation,angular,shell=_approach(.003)
    with pytest.raises(ValueError,match='open and closed endpoints'):
        evaluate_first_touch(pair,start,history,np.tile((0.,0.,-.001),(3,1)),np.zeros((2,3)),shell)
    angular[0,1]=1.
    with pytest.raises(ValueError,match='monotone path unresolved'):
        evaluate_first_touch(pair,start,history,translation,angular,shell)
    with pytest.raises(ValueError,match='released physical history'):
        evaluate_first_touch(pair,start,replace(history,partner_key=pair.partner_key),translation,np.zeros((2,3)),shell)


def test_first_touch_finite_deforming_target_frame_and_both_edge_changes():
    pair,start,history,translation,angular,shell=_approach(.003)
    shell[6:9]=(.002,-.001,.001)
    shell[12:15]=(-.001,.002,-.0007)
    value=evaluate_first_touch(pair,start,history,translation,angular,shell)
    assert value.trial.branch=='stick'
    direction=np.random.default_rng(619).normal(size=60)
    for step in (1e-5,2e-6):
        values=[]
        for sign in (1,-1):
            t=translation+sign*step*np.array([direction[6*i:6*i+3] for i in range(3)])
            w=np.array([(Rotation.from_rotvec(sign*step*direction[18+3*i:21+3*i])
                         *Rotation.from_rotvec(angular[i])).as_rotvec() for i in range(2)])
            values.append(evaluate_first_touch(pair,start,history,t,w,shell+sign*step*direction[42:]))
        fd=(values[0].residual-values[1].residual)/(2*step)
        assert np.linalg.norm(value.spatial_tangent()@direction-fd)/np.linalg.norm(fd)<=1e-5
    with pytest.raises(ValueError,match='(ownership unresolved|monotone path unresolved)'):
        evaluate_first_touch(pair,start,history,translation,angular,1000*shell)


def test_first_touch_rejects_changed_bound_definition_before_history_evaluation():
    pair,start,history,translation,angular,shell=_approach(.003)
    pair.normal_stiffness_Pa_per_m *= 2.
    with pytest.raises(ValueError,match='definition changed'):
        evaluate_first_touch(pair,start,history,translation,angular,shell)
    assert history.partner_key is None

@pytest.mark.parametrize('size',(16,30))
def test_prepared_event_rotation_derivatives_match_uncached_jet_algebra(size):
    import anyfesolver.general_contact.native_surface_activation as module
    from anyfesolver._ge_beam3_mixed_ad import Jet2,so3_exp,so3_log,matmul,constant_matrix
    for vector in ((0.,0.,0.),(.03,-.02,.01),(-.2,.1,.3)):
        rotation=Rotation.from_rotvec(vector).as_matrix()
        entries=tuple(rotation.ravel())
        value=module._event_rotation_chart(entries,size)
        q=[Jet2.variable(0.,3+i,size) for i in range(3)]
        exact=so3_log(matmul(so3_exp(q),constant_matrix(rotation,size)))
        for actual,expected in zip(value,exact):
            assert actual.value==expected.value
            np.testing.assert_array_equal(actual.gradient,expected.gradient)
            np.testing.assert_array_equal(actual.hessian,expected.hessian)
            assert not actual.gradient.flags.writeable and not actual.hessian.flags.writeable
        assert module._event_rotation_chart(entries,size) is value
    assert module._event_rotation_chart.cache_info().maxsize==32


def test_prepared_mixed_rotation_matches_uncached_virtual_work_map():
    import anyfesolver.general_contact.native_surface_activation as module
    from anyfesolver._ge_beam3_mixed_ad import Jet2,so3_exp,matmul
    q=[Jet2.variable(0.,3+i,30) for i in range(3)]
    v=[Jet2.variable(0.,18+i,30) for i in range(3)]
    exact=matmul(so3_exp(v),so3_exp(q))
    actual=module._endpoint_virtual_rotation()
    for row,expected in zip(actual,exact):
        for a,b in zip(row,expected):
            assert a.value==b.value
            np.testing.assert_array_equal(a.gradient,b.gradient)
            np.testing.assert_array_equal(a.hessian,b.hessian)
    assert module._endpoint_virtual_rotation.cache_info().maxsize==1


def test_prepared_root_retains_list_form_motion_inputs():
    pair,start,history,translation,angular,shell=_approach(.003)
    expected=evaluate_first_touch(pair,start,history,translation,angular,shell)
    actual=evaluate_first_touch(pair,start,history,translation.tolist(),angular.tolist(),shell.tolist())
    assert actual.event_fraction==expected.event_fraction
    np.testing.assert_array_equal(actual.residual,expected.residual)
    np.testing.assert_array_equal(actual.tangent,expected.tangent)
