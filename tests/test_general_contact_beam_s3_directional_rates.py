"""One consolidated source discriminator, not a general impact campaign."""
import json,os,copy
from pathlib import Path
from fractions import Fraction as Q
from decimal import Decimal,localcontext
import numpy as np
import pytest
from examples.general_contact_beam_s3_transient_directional import build,run
from anyfesolver.general_contact.beam_s3_directional_rates import (
    DirectionalBeamS3Manager,exact_gap_polynomial,directional_coefficients,
    directional_operator,solve_directional_velocity)


def test_directional_oracle_and_actual_first_touch_restart():
    # Independent scalar: a=2*(.25-a)^(3/2), p(s)=s^2-.25+a.
    # Its single active interval [0,sqrt(.25-a)] and integrals are analytical.
    B=np.array([[Q(-1,4),Q(1)],[Q(0),Q(0)],[Q(1),Q(0)]],object)
    with localcontext() as ctx:
        ctx.prec=55;lo=Decimal(0);hi=Decimal('.25')
        for _ in range(150):
            a=(lo+hi)/2;b=(Decimal('.25')-a).sqrt()
            if a-2*b**3>0:hi=a
            else:lo=a
        expected=float((lo+hi)/2)
    velocity=solve_directional_velocity(np.eye(2),np.array([[0.],[1.]]),np.array([1.,0.]),B,Q(3))
    D,H,eD,eH=directional_operator(B,Q(3),velocity);a=velocity[1];b=np.sqrt(.25-a)
    expectedH=3*np.array([[b**5/5-.5*b**3/3+b/16,b**3/3-b/4],[b**3/3-b/4,b]])
    assert abs(a-expected)<1.e-9 and velocity[0]==1.
    np.testing.assert_allclose(H,expectedH,rtol=1.e-12,atol=1.e-14)
    assert abs(D[1]+2*b**3)<1.e-12 and abs(a+D[1])<1.e-9
    assert eD<1.e-10 and eH<1.e-10
    m,d,s,v=build();manager=DirectionalBeamS3Manager(m,d,s);u=np.zeros_like(v)
    assert exact_gap_polynomial(manager,u)[:2]==(True,False)
    coefficients,factor=directional_coefficients(manager,u)
    assert coefficients.shape==(3,len(v)) and factor>0
    near=u.copy();near[manager.beam_dofs[:,2]]=np.nextafter(0.,1.)
    # Exact stored polynomial distinguishes even a subnormal displacement.
    assert not exact_gap_polynomial(manager,near)[0]
    with pytest.raises(ValueError,match='genuinely exact'):directional_coefficients(manager,near)
    isolated=u.copy();isolated[manager.beam_dofs[[0,2],2]]=1.e-4
    assert exact_gap_polynomial(manager,isolated)[:2]==(False,True)
    _,result=run();_,prefix=run(end=1.e-5);_,resumed=run(restart=prefix.restart_checkpoint)
    assert result.restart_checkpoint['version']==19 and result.contact[0].minimum_gap==0.
    assert result.contact[-1].active_line_length>0 and result.contact[-1].minimum_gap>=-1.e-5
    assert resumed.restart_checkpoint==result.restart_checkpoint
    np.testing.assert_array_equal(resumed.displacements,result.displacements[1:])
    from anyfesolver.general_contact.transient import solve_transient_contact,ContactTransientSolverControls
    from anyfesolver.dynamics import TransientConfig
    m,d,s,v=build()
    with pytest.raises(ValueError,match='analysis|version|incompatible'):
        solve_transient_contact(m,d,TransientConfig(1.e-5,3.e-5,initial_velocity=v),contact_settings=s,
            solver_controls=ContactTransientSolverControls(allow_rotational_beam_s3_dev=True),restart_checkpoint=prefix.restart_checkpoint)
    if os.environ.get('CONTACT_APP_RESULT_RECORD'):
        Path(os.environ['CONTACT_APP_RESULT_RECORD']).write_text(json.dumps(dict(
            scalar_expected=expected,scalar_actual=float(a),velocity_residual=float(abs(a+D[1])),
            root_force_error_bound=eD,root_jacobian_error_bound=eH,
            states=len(result.times),initial_gap=result.contact[0].minimum_gap,
            active=result.contact[-1].active_line_length,minimum_gap=result.contact[-1].minimum_gap,
            restart_equal=True,nonexact_refused=True,isolated_touch_classified=True),indent=2))


def test_legacy_load_constructor_directional_controls_and_null_error_transfer():
    from anyfesolver.general_contact.transient import ContactNodalLoadHistory,ContactTransientSolverControls
    from anyfesolver.general_contact.beam_s3_directional_rates import _null_error_factor
    history=ContactNodalLoadHistory(((0.,0.),(1.,1.)))
    assert history.factor(.5)==.5
    assert ContactTransientSolverControls().allow_directional_beam_s3_dev is False
    assert ContactTransientSolverControls(allow_directional_beam_s3_dev=True).allow_directional_beam_s3_dev
    for bad in (1,None,'yes'):
        with pytest.raises(ValueError,match='boolean'):ContactTransientSolverControls(allow_directional_beam_s3_dev=bad)
    with pytest.raises(ValueError):ContactTransientSolverControls(allow_directional_beam_s3_dev=True,allow_rotational_beam_s3_dev=True)
    Z=np.array(((1.,0.),(0.,1.),(1.,1.)))
    factor=_null_error_factor(Z)
    assert Q(float(factor))**2>=Q(8) and factor-np.sqrt(8)<1.e-14
    assert np.linalg.norm(Z.T@np.ones(3))<=factor
    if os.environ.get('CONTACT_APP_RESULT_RECORD'):
        Path(os.environ['CONTACT_APP_RESULT_RECORD']).write_text(json.dumps(dict(legacy_load_constructed=True,directional_bool_conflicts_refused=True,rank_two_null_box_factor=factor,required_factor_squared=8),indent=2))
