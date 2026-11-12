"""Actual mixed-obstacle solves with an independently balanced B3 bar."""
from dataclasses import replace
import json
import numpy as np
import pytest
from scipy.optimize import brentq
from anyfesolver import (RigidContactBody, solve_static_nonlinear,
    validate_general_contact, report_general_contact_capability)
from anyfesolver.general_contact.multiple_beam_rigid import (
    MultipleBeamRigidManager, PROFILE_SUPPORT_CYLINDER)
from anyfesolver.general_contact.beam_hemisphere_cylinder import HemisphereCylinderManager
from test_general_contact_multiple_beam_ends import fixture


def cylinder_bars(left_kind='plane'):
    model, definition, settings, load = fixture('cylinder')
    if left_kind != 'plane':
        left = RigidContactBody('left', left_kind, (0., 0., 0.), radius=.1,
            normal=(0., 0., 1.) if left_kind=='cylinder' else None)
        definition = replace(definition, rigid_bodies=(left, definition.rigid_bodies[1]))
    return model, definition, replace(settings, beam_end_cap_work='hemisphere_support_dev'), load


@pytest.mark.parametrize('left_kind', ('plane', 'sphere', 'cylinder'))
def test_complete_mixed_static_equilibrium_work_and_restart(left_kind):
    model, definition, settings, load = cylinder_bars(left_kind)
    assert validate_general_contact(model, definition, settings).valid
    capability = report_general_contact_capability(model, definition, settings)
    assert capability.validation.profile == PROFILE_SUPPORT_CYLINDER
    assert not capability.independently_accepted and not capability.admitted_for_application
    manager = MultipleBeamRigidManager(model, definition, settings)
    assert isinstance(manager.pairs[-1][1], HemisphereCylinderManager)
    common = dict(contact=definition, contact_settings=settings, num_steps=4,
        max_iterations=20, tolerance=1e-10, emit_restart_checkpoint=True)
    result = solve_static_nonlinear(model, load, **common)
    assert result.status == 'completed', result.failure_reason
    # Direct integration of the quadratic axial shape N1=1-3s+2s².
    # For this symmetric axial fixture each sphere/cylinder has the same radial
    # gap and left plane the same signed gap. This is not a general equivalence.
    def balance(delta):
        a=.002+delta; D=.2+2*a
        line=1e4*.2*(delta**2/(2*D)-delta**3/(2*D**2)+delta**4/(6*D**3))
        return 10*a+1e5*delta+line-.2
    delta=brentq(balance,0.,1e-5,xtol=1e-18); a=.002+delta
    dofs=[model.mesh.dof_manager.get_node_dofs(n)[0] for n in (1,3)]
    np.testing.assert_allclose(result.displacements[dofs],(-a,a),rtol=1e-6,atol=1e-10)
    info=result.info['general_contact']
    assert info['profile']==PROFILE_SUPPORT_CYLINDER and info['active_endpoints']==2
    assert info['candidate_endpoints']==4 and info['gap_tolerance_met']
    np.testing.assert_allclose(np.array(info['beam_force'])+info['rigid_force'],0.,atol=2e-12)
    np.testing.assert_allclose(np.array(info['beam_moment'])+info['rigid_moment'],0.,atol=2e-12)
    u=result.displacements; f,K,fields=manager.evaluate(u)
    direction=np.random.default_rng(423).normal(size=len(u)); direction/=np.linalg.norm(direction)
    for h in (1e-7,3e-8):
        fp,_,rp=manager.evaluate(u+h*direction,tangent=False)
        fm,_,rm=manager.evaluate(u-h*direction,tangent=False)
        assert np.linalg.norm((fp-fm)/(2*h)-K@direction)/np.linalg.norm(K@direction)<1e-5
        ep=rp.to_dict()['algorithmic_contact_potential']; em=rm.to_dict()['algorithmic_contact_potential']
        assert (ep-em)/(2*h)==pytest.approx(f@direction,rel=1e-5,abs=1e-9)
    for row,(_,child) in zip(info['pairs'],manager.pairs):
        own=child.evaluate(u)[2].to_dict()
        np.testing.assert_allclose(row['contact']['rigid_force'],own['rigid_force'],atol=2e-12)
    fresh,_,_,_=cylinder_bars(left_kind)
    prefix=solve_static_nonlinear(fresh,load,**{**common,'num_steps':2},max_load_factor=.75)
    assert prefix.status=='completed',prefix.failure_reason
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_=cylinder_bars(left_kind)
    resumed=solve_static_nonlinear(fresh,load,**{**common,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(fresh,load,**{**common,'contact_settings':replace(settings,
            beam_end_cap_work='owned_point')},restart_checkpoint=checkpoint)


def test_public_installed_example():
    from examples.general_contact_hemisphere_cylinder import run
    result=run()
    assert result.status=='completed' and result.info['general_contact']['gap_tolerance_met']
    assert result.info['general_contact']['active_endpoints']==1
