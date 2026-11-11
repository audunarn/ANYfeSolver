"""Coupled pressure, physical history and static-path friction controls."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import BoundaryCondition, create_shell_element, solve_static_nonlinear
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact import (
    ContactDefinition, ContactInteraction, ContactSurface, ContactSolverSettings,
    ContactPathControl, ContactPathProgram, solve_static_contact_path,
)
from anyfesolver.general_contact.s3_friction_point import S3SkinMaterialPoint
from anyfesolver.general_contact.s3_friction_pair import S3FrictionPointPair
from anyfesolver.general_contact.s3_friction_static import (
    S3MaterialFrictionSettings, S3MaterialFrictionManager,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


def _case():
    model = FEModel("pressure_coupled_material_friction")
    model.add_material("elastic", 1e5, 0.3, density=1)
    points = ((.2,.2,.065), (.6,.2,.065), (.2,.6,.065),
              (0,0,0), (1,0,0), (0,1,0), (1,1,0))
    for i, xyz in enumerate(points, 1):
        model.add_node(i, *xyz)
    for i, nodes in ((1,(1,2,3)), (2,(4,5,6)), (3,(5,7,6))):
        model.add_element(i, create_shell_element(i, nodes, "elastic", thickness=.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0,0,1),
            reference_surface_offset=.01))
    definition = ContactDefinition((ContactSurface("a",(1,),"bottom"),
        ContactSurface("b",(2,3),"top")), (),
        (ContactInteraction("pair","a","b",1000.,.02,normal_law="projected_penalty"),))
    settings = ContactSolverSettings(pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2e-5, general_s3_differentiation="jet2_s3_dev")
    pairs = []
    for bary in ((2/3,1/6,1/6), (1/6,2/3,1/6), (1/6,1/6,2/3)):
        xy = np.asarray(bary) @ np.asarray(points[:3])[:,:2]
        pairs.append(S3FrictionPointPair(
            S3SkinMaterialPoint.from_model(model,1,1,"bottom",bary),
            S3SkinMaterialPoint.from_model(model,2,2,"top",(1-xy.sum(),*xy)),
            reference_weight_m2=.08/3, friction_coefficient=.3,
            tangential_stiffness_Pa_per_m=1000., maximum_active_columns=42))
    friction = S3MaterialFrictionSettings(tuple(pairs),
        (2e-7,)*model.mesh.dof_manager.total_dofs, maximum_candidate_evaluations=100)
    return model, definition, settings, friction


def _mode(model, nodes, axis):
    v = np.zeros(model.mesh.dof_manager.total_dofs)
    for n in nodes:
        v[model.mesh.dof_manager.get_node_dofs(n)[axis]] = 1
    return v


def test_coupled_slip_pressure_derivative_and_disposable_history():
    model, definition, settings, friction = _case()
    manager = S3MaterialFrictionManager(model, GeneralS3PatchManager(model,definition,settings),friction)
    x = _mode(model,(1,2,3),0)
    z = _mode(model,(1,2,3),2)
    target_x = _mode(model,(4,5,6,7),0)
    manager.commit(manager.prepare_commit(-.02*z))
    old = manager.checkpoint_state()
    u = -.02*z + .008*x
    force, K, _ = manager.evaluate(u)
    base_force, _, _ = manager.normal.evaluate(u, tangent=False)
    # Independent flat, uniform gap law: p=1000*.005, |t|=.3*p.
    assert float((force-base_force)@x) == pytest.approx(.08*.3*5, abs=1e-10)
    assert float((force-base_force)@(x+target_x)) == pytest.approx(0, abs=1e-12)
    normal_K = manager.normal.evaluate(u)[1]
    assert float(x @ ((K-normal_K)@z)) == pytest.approx(-.08*.3*1000, rel=1e-7)
    assert manager.checkpoint_state() == old
    direction = .01*x+.003*z
    h = 1e-5
    difference = (manager.evaluate(u+h*direction,tangent=False)[0]
                  - manager.evaluate(u-h*direction,tangent=False)[0])/(2*h)
    np.testing.assert_allclose(K@direction,difference,rtol=1e-5,atol=1e-7)
    candidate = manager.prepare_commit(u)
    assert set(candidate[2]["branch"]) == {"slip"}
    assert candidate[2]["dissipation_increment_j"] > 0
    assert manager.checkpoint_state() == old
    manager.commit(candidate)
    manager.commit(manager.prepare_commit(.008*x))
    assert all(h.partner_key is None and h.elastic_slip == (0,0,0) for h in manager.history)


def test_real_static_stick_slip_reversal_opening_and_split_restart():
    model, definition, settings, friction = _case()
    model.add_boundary_condition(BoundaryCondition("target",(4,5,6,7),
        {d:0 for d in ("ux","uy","uz","rx","ry","rz")}))
    model.add_boundary_condition(BoundaryCondition("guide",(1,2,3),
        {d:0 for d in ("uy","rx","ry","rz")}))
    program = ContactPathProgram((ContactPathControl.node(1,"ux"),ContactPathControl.node(1,"uz")),
        ((0,0),(0,-.02),(.0002,-.02),(.008,-.02),(.004,-.02),(.004,0),(.004,-.02)),1)
    kwargs = dict(model=model,contact=definition,contact_settings=settings,contact_path_program=program,
        contact_friction=friction,max_iterations=18,tolerance=1e-8,emit_restart_checkpoint=True,
        record_increment_snapshots=True)
    full = solve_static_nonlinear(**kwargs)
    assert full.status == "completed", full.info
    history = full.info["force_displacement_history"]
    assert set(history[1]["friction"]["branch"]) == {"stick"}
    assert set(history[2]["friction"]["branch"]) == {"slip"}
    assert set(history[4]["friction"]["branch"]) == {"open"}
    assert all(r["friction"]["dissipation_increment_j"] >= 0 for r in history)
    assert any(abs(full.snapshots[2].displacements[model.mesh.dof_manager.get_node_dofs(n)[0]]-.008)>1e-7
               for n in (2,3))  # Structural tangential response is actually solved.
    half = solve_static_nonlinear(max_load_factor=.5,**kwargs)
    assert half.status == "completed", half.info
    changed = replace(friction,pairs=tuple(replace(p,friction_coefficient=.4) for p in friction.pairs))
    with pytest.raises(ValueError,match="contract|fingerprint|incompatible"):
        solve_static_nonlinear(restart_checkpoint=half.to_restart_checkpoint(),
            **{**kwargs,"contact_friction":changed})
    resumed = solve_static_nonlinear(restart_checkpoint=half.to_restart_checkpoint(),**kwargs)
    np.testing.assert_allclose(resumed.displacements,full.displacements,rtol=1e-7,atol=1e-9)
    assert resumed.info["force_displacement_history"] == full.info["force_displacement_history"]


def test_friction_restore_rejects_wrong_partner_and_model():
    model, definition, settings, friction = _case()
    manager = S3MaterialFrictionManager(model,GeneralS3PatchManager(model,definition,settings),friction)
    u = -.02*_mode(model,(1,2,3),2)
    manager.commit(manager.prepare_commit(u))
    state = copy.deepcopy(manager.checkpoint_state())
    state["history"][0]["partner_key"] = "other"
    with pytest.raises(ValueError,match="partner"):
        manager.restore(state,u)
    assert manager.checkpoint_state() != state


def test_friction_invalid_candidate_and_cancel_restore_context():
    from anyfesolver.control import CancellationToken, SolveCancelled
    from anyfesolver.general_contact.runtime import active_contact_manager
    model, definition, settings, friction = _case()
    manager = S3MaterialFrictionManager(model,GeneralS3PatchManager(model,definition,settings),friction)
    old = manager.checkpoint_state()
    with pytest.raises(ValueError,match="target facet"):
        manager.prepare_commit(2*_mode(model,(1,2,3),0))
    assert manager.checkpoint_state() == old
    with pytest.raises(ValueError,match="explicit contact path"):
        solve_static_nonlinear(model,contact_friction=friction)
    token = CancellationToken()
    token.cancel("friction state control")
    model.add_boundary_condition(BoundaryCondition("target",(4,5,6,7),
        {d:0 for d in ("ux","uy","uz","rx","ry","rz")}))
    model.add_boundary_condition(BoundaryCondition("guide",(1,2,3),
        {d:0 for d in ("uy","rx","ry","rz")}))
    program = ContactPathProgram((ContactPathControl.node(1,"ux"),ContactPathControl.node(1,"uz")),
        ((0,0),(0,-.02)),1)
    with pytest.raises(SolveCancelled):
        solve_static_nonlinear(model,contact=definition,contact_settings=settings,
            contact_path_program=program,contact_friction=friction,cancellation_token=token)
    assert active_contact_manager() is None


def test_friction_rejects_cut_face_even_with_interior_quadrature():
    model, definition, settings, friction = _case()
    manager = S3MaterialFrictionManager(model,GeneralS3PatchManager(model,definition,settings),friction)
    old = manager.checkpoint_state()
    with pytest.raises(ValueError,match="whole source facet"):
        manager.prepare_commit(.25*_mode(model,(1,2,3),0))
    assert manager.checkpoint_state() == old


def test_precancelled_entry_skips_assembly_predictor_and_keeps_context(monkeypatch):
    from anyfesolver.control import CancellationToken, SolveCancelled
    from anyfesolver.general_contact.runtime import active_contact_manager
    import anyfesolver.general_contact.static_path as path_driver
    import anyfesolver.nonlinear_static as driver
    model, definition, settings, friction = _case()
    model.add_boundary_condition(BoundaryCondition("target",(4,5,6,7),
        {d:0 for d in ("ux","uy","uz","rx","ry","rz")}))
    model.add_boundary_condition(BoundaryCondition("guide",(1,2,3),
        {d:0 for d in ("uy","rx","ry","rz")}))
    program = ContactPathProgram((ContactPathControl.node(1,"ux"),ContactPathControl.node(1,"uz")),
        ((0,0),(0,-.02)),1)
    token = CancellationToken()
    token.cancel("entry precancel")
    def sentinel(*args, **kwargs):
        raise AssertionError("assembly or predictor reached before entry cancellation")
    before = repr(model)
    context = active_contact_manager()
    with monkeypatch.context() as patch:
        patch.setattr(path_driver, "assemble_stiffness_matrix", sentinel)
        patch.setattr(path_driver, "_elastic_displacement_force_predictor", sentinel)
        patch.setattr(driver, "_assemble_nonlinear_system", sentinel)
        with pytest.raises(SolveCancelled):
            solve_static_nonlinear(model,contact=definition,contact_settings=settings,
                contact_path_program=program,contact_friction=friction,
                cancellation_token=token)
    assert repr(model) == before
    assert active_contact_manager() is context


def test_uncancelled_legacy_friction_reports_zero_reaction_predictor(monkeypatch):
    # The uncancelled one-step press on the unchanged legacy fixture: the real
    # helper runs first with the projected-profile zero allowance, its finite
    # nonnegative result is checked, and the reported diagnostic is the zero
    # the caller consumes. Step one stays in the expected stick branch.
    import anyfesolver.general_contact.static_path as path_driver
    real = path_driver._elastic_displacement_force_predictor
    seen = []
    def spy(*args, **kwargs):
        seen.append(kwargs.get("allow_zero", False))
        value = real(*args, **kwargs)
        assert np.isfinite(value) and value >= 0.
        return 0.
    monkeypatch.setattr(path_driver, "_elastic_displacement_force_predictor", spy)
    model, definition, settings, friction = _case()
    model.add_boundary_condition(BoundaryCondition("target",(4,5,6,7),
        {d:0 for d in ("ux","uy","uz","rx","ry","rz")}))
    model.add_boundary_condition(BoundaryCondition("guide",(1,2,3),
        {d:0 for d in ("uy","rx","ry","rz")}))
    program = ContactPathProgram((ContactPathControl.node(1,"ux"),ContactPathControl.node(1,"uz")),
        ((0,0),(0,-.02)),1)
    result = solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        contact_path_program=program,contact_friction=friction)
    assert seen == [True]
    assert result.status == "completed", result.info
    assert result.info["contact_free_elastic_force_predictor_n"] == 0.
    assert set(result.info["force_displacement_history"][0]["friction"]["branch"]) == {"stick"}
