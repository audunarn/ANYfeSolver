"""Coupled sphere/plane work and same-path complete static restart control."""
from dataclasses import replace
import json
import numpy as np
import pytest
from scipy.integrate import quad
from anyfesolver import LoadCase, RigidContactBody, solve_static_nonlinear
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver.general_contact.multiple_beam_rigid import MultipleBeamRigidManager
from test_general_contact_multiple_beam_rigid import fixture as two_plane_fixture


def fixture():
    model, definition, settings = two_plane_fixture()
    sphere = RigidContactBody("y_tool","sphere",(0.,-.005,.015),radius=.005)
    return model, replace(definition,rigid_bodies=(sphere,definition.rigid_bodies[1]),
        interactions=(replace(definition.interactions[0],line_stiffness=1e7),definition.interactions[1])),settings


def independent_force(yz):
    y,z = yz
    n = np.array((0.,4.,-4.))
    dy = np.array((.02,4*y,-4*y)); dz = n*z
    squared = np.polynomial.polynomial.polyadd(np.polynomial.polynomial.polymul(dy,dy),np.polynomial.polynomial.polymul(dz,dz))
    squared[:3] += np.array((.25,-1.,1.))
    squared[0] -= .015**2
    roots = np.polynomial.polynomial.polyroots(squared)
    sphere_splits = sorted(float(a.real) for a in roots if abs(a.imag)<1e-10 and 0<a.real<1)
    plane_coeff = np.array((.005,4*z,-4*z))
    roots = np.polynomial.polynomial.polyroots(plane_coeff)
    plane_splits = sorted(float(a.real) for a in roots if abs(a.imag)<1e-10 and 0<a.real<1)
    def radial(t,i):
        shape=4*t*(1-t);d=np.array((t-.5,.02+shape*y,shape*z));rho=np.linalg.norm(d)
        g=rho-.015
        return 1e7*min(g,0.)*shape*d[i+1]/rho
    result=np.array([quad(lambda t:radial(t,i),0.,1.,points=sphere_splits,epsabs=1e-9,epsrel=1e-11)[0] for i in (0,1)])
    result[1] += quad(lambda t:2e5*min(.005+4*t*(1-t)*z,0.)*4*t*(1-t),0.,1.,points=plane_splits,epsabs=1e-9,epsrel=1e-11)[0]
    return result


def test_sphere_plane_coupling_cancelled_path_and_full_checkpoint():
    model,definition,settings=fixture()
    load=LoadCase("sphere_plane");load.add_nodal_load(2,forces=(0.,-10.,-12.))
    controls=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=22,
                  tolerance=1e-8,emit_restart_checkpoint=True)
    full=solve_static_nonlinear(model,load,**controls)
    assert full.status=="completed",full.failure_reason
    contact=full.info["general_contact"]
    assert contact["gap_tolerance_met"]
    assert all(p["contact"]["active_line_length"]>0 for p in contact["pairs"])
    indices=np.array(model.mesh.dof_manager.get_node_dofs(2)[1:3])
    manager=MultipleBeamRigidManager(model,definition,settings)
    residual,_,_=manager.evaluate(full.displacements)
    expected=independent_force(full.displacements[indices])
    np.testing.assert_allclose(residual[indices],expected,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(np.array(contact["beam_force"])+contact["rigid_force"],0.,atol=1e-9)
    np.testing.assert_allclose(np.array(contact["beam_moment"])+contact["rigid_moment"],0.,atol=1e-9)
    force_scale=np.hypot(10.,12.)
    assert full.steps[-1].residual_norm/force_scale<1e-8
    token=CancellationToken()
    def stop(event):
        if event.event_type=="nonlinear_static_step" and event.metadata["load_factor"]>=.5:
            token.cancel("same-path restart control")
    prefix_model,_,_=fixture()
    with pytest.raises(SolveCancelled,match="same-path restart control"):
        solve_static_nonlinear(prefix_model,load,cancellation_token=token,progress_callback=stop,**controls)
    # Existing cancellation contract raises; checkpoint is produced by a normal
    # bounded prefix on the same reusable model and exact .25 load partition.
    prefix=solve_static_nonlinear(prefix_model,load,max_load_factor=.5,
                                 **dict(controls,num_steps=2))
    assert prefix.status=="completed"
    assert prefix.steps[-1].load_factor==.5
    saved=json.loads(json.dumps(prefix.to_restart_checkpoint()))
    fresh,_,_=fixture()
    resumed=solve_static_nonlinear(fresh,load,restart_checkpoint=saved,**controls)
    assert resumed.status=="completed",resumed.failure_reason
    np.testing.assert_array_equal(resumed.displacements,full.displacements)
    assert resumed.restart_checkpoint_bytes()==full.restart_checkpoint_bytes()
    print("SPHERE_PLANE_COMPLETE_RESTART "+json.dumps(dict(coordinates=full.displacements[indices].tolist(),
        independent_contact_force=expected.tolist(),complete_checkpoint_sha256=full.restart_checkpoint["checkpoint_sha256"],
        accepted_steps=len(full.steps),prefix_steps=len(prefix.steps),contact=contact),sort_keys=True))
