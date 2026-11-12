"""Independent first-touch slider reference, not a general friction admission."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver.general_contact.s3_sliding_friction import S3SlidingFrictionManager
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_s3_sliding_friction import _manager
from test_general_contact_s3_friction_static import _mode


def _owner(policy="linear_constant_normal_touch_dev",refinement=3):
    model,definition,normal_settings,friction,_unused=_manager()
    manager=S3SlidingFrictionManager(model,GeneralS3PatchManager(model,definition,normal_settings),
        replace(friction,source_quadrature_refinement=refinement,activation_history=policy))
    return model,manager


def _slider(steps,policy="linear_constant_normal_touch_dev"):
    model,manager=_owner(policy)
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    dissipation=0.
    for tau in np.linspace(0,1,steps+1)[1:]:
        u=tau*(.1*x-.02*z)
        candidate=manager.prepare_commit(u)
        dissipation+=candidate[2]["dissipation_increment_j"]
        manager.commit(candidate)
    reaction=x@manager._friction(u)[0]
    return float(reaction),dissipation


def test_simultaneous_approach_slide_resolves_first_touch_reaction():
    # g(tau)=.015-.02*tau, so tau_touch=.75. In active contact
    # p=20*(tau-.75), cap=6*(tau-.75), elastic_gap=.006*(tau-.75).
    # dx/dtau=.1 > d(elastic_gap)/dtau=.006: slip starts at touch.
    reaction,dissipation=_slider(1)
    assert reaction==pytest.approx(.08*.3*5,rel=1e-3),(reaction,dissipation)


def test_endpoint_rule_is_preserved_and_touch_work_has_measured_step_accuracy(record_property):
    assert _slider(1,"endpoint")==pytest.approx((0.,0.),abs=1e-13)
    exact=.08*.094*6*.25**2/2
    errors=[]
    for steps in (4,8,16,32,256):
        reaction,work=_slider(steps)
        assert reaction==pytest.approx(.12,rel=1e-10)
        assert work==pytest.approx(exact*(1+4/steps),rel=1e-9)
        errors.append(abs(work/exact-1))
    assert errors[-1]<.02
    record_property("steps_4_8_16_32_256_relative_work_error",str(errors))


def test_unloading_slide_work_converges_to_independent_continuous_reference(record_property):
    # Contact opens at tau=.25. Stick -> slip occurs at
    # 100*tau=1.5-6*tau, tau_slip=1.5/106. Thereafter plastic rate=.106.
    tau_slip=1.5/106
    exact=.08*.106*3*(.25-tau_slip)**2
    errors=[]
    for steps in (4,16,64,256):
        model,manager=_owner()
        x,z=(_mode(model,(1,2,3),a) for a in (0,2))
        press=-.02*z
        manager.commit(manager.prepare_commit(press))
        work=0.
        for tau in np.linspace(0,1,steps+1)[1:]:
            u=press+tau*(.1*x+.02*z)
            state=manager.prepare_commit(u)
            work+=state[2]["dissipation_increment_j"]
            assert state[2]["dissipation_increment_j"]>=0
            manager.commit(state)
        assert all(h.target_element is None for h in manager.history)
        np.testing.assert_allclose(manager._friction(u)[0],0,atol=1e-13)
        errors.append(abs(work/exact-1))
    assert errors[0]>.9 and errors[-1]<.02,errors
    assert all(b<a for a,b in zip(errors,errors[1:])),errors
    record_property("unloading_steps_4_16_64_256_relative_work_error",str(errors))


def test_first_touch_fraction_tangent_includes_moving_event_and_anchor():
    model,manager=_owner()
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    active=[model.mesh.dof_manager.get_node_dofs(n)[a] for n in range(1,8) for a in (0,1)]
    active += [model.mesh.dof_manager.get_node_dofs(n)[2] for n in (1,2,3)]
    T=sparse.coo_matrix((np.ones(len(active)),(active,np.arange(len(active)))),shape=(42,len(active)))
    manager.restrict_tangent_to_reduced_variations(T)
    u=.0002*x-.02*z
    residual,K,_certificate=manager.evaluate(u)
    normal_K=manager.normal.evaluate(u)[1]
    assert x@manager._friction(u)[0]==pytest.approx(.004,rel=1e-9)
    assert x@((K-normal_K)@z)==pytest.approx(-.6,rel=1e-7)
    records=manager._friction(u)[2]
    assert all(r["first_touch"][2]==pytest.approx(.75,abs=1e-13) for r in records)
    direction=.003*x+.007*z
    for h in (1e-4,2e-5):
        plus=manager.evaluate(u+h*direction,tangent=False)[0]
        minus=manager.evaluate(u-h*direction,tangent=False)[0]
        fd=(plus-minus)/(2*h)
        assert np.linalg.norm(K@direction-fd)/np.linalg.norm(fd)<=1e-5
    # The physical work map still holds the returned anchor fixed. Its storage
    # is the independently derived post-touch elastic displacement .00005m.
    state=manager.prepare_commit(u)
    assert state[2]["storage_j"]==pytest.approx(.08*.5*1000*.00005**2,rel=1e-9)
    assert state[2]["dissipation_increment_j"]==0.
    assert np.all(np.isfinite(residual))


def test_first_touch_translation_restart_and_geometry_rejection_are_atomic():
    model,manager=_owner()
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    shift=.2*_mode(model,range(1,8),0)+.07*_mode(model,range(1,8),2)
    manager.commit(manager.prepare_commit(shift))
    u=shift+.1*x-.02*z
    state=manager.prepare_commit(u)
    assert state[2]["dissipation_increment_j"]==pytest.approx(.00282,rel=1e-10)
    manager.commit(state)
    before=manager.checkpoint_state()
    restored=S3SlidingFrictionManager(model,
        GeneralS3PatchManager(model,manager.normal.definition,manager.normal.settings),manager.friction_settings)
    restored.restore(before,u)
    np.testing.assert_allclose(manager._friction(u)[0],restored._friction(u)[0],atol=1e-12)
    assert restored.checkpoint_state()==before
    assert manager.restart_contract()["friction_settings"]["activation_history"]=="linear_constant_normal_touch_dev"
    model,invalid=_owner()
    old=invalid.checkpoint_state()
    tilted=np.zeros(42)
    for node,y in ((5,0.),(6,1.),(7,1.)):
        tilted[model.mesh.dof_manager.get_node_dofs(node)[2]]=.02*y
    with pytest.raises(ValueError,match="constant target normal"):
        invalid.prepare_commit(tilted)
    assert invalid.checkpoint_state()==old
    collapsed=np.zeros(42)
    xy=((.2,.2),(.6,.2),(.2,.6),(0,0),(1,0),(0,1),(1,1))
    for node,p in enumerate(xy,1):
        collapsed[model.mesh.dof_manager.get_node_dofs(node)[:2]]=-2*np.array(p)
    # Valid endpoints, but collapse at alpha=.5; no sampled certificate.
    with pytest.raises(ValueError,match="unresolved or inverted"):
        invalid.prepare_commit(collapsed)
    assert invalid.checkpoint_state()==old
    contract=manager.restart_contract()
    assert contract["recontact"]=="stress_free_first_touch_projection"
    assert contract["activation_path"]=="affine_full_FE_constant_target_normal_v1"


def test_first_touch_seam_probes_are_bounded_disposable_and_share_event_facet():
    model,manager=_owner(refinement=0)
    x,z=(_mode(model,(1,2,3),a) for a in (0,2))
    u=(1-11/15)/.75*x-.02*z
    _force,_states,records=manager._friction(u)
    touch=[r["first_touch"] for r in records]
    assert sum(min(t[1])<1e-13 for t in touch)==2
    assert all(t[0]==2 for t in touch)
    before=manager.checkpoint_state()
    kwargs=dict(branches=[r["branch"] for r in records],validate=False,
        linearized_elements=[r["target_element"] for r in records],
        linearized_barycentric=[r["target_barycentric"] for r in records],linearized_touch=touch)
    _force,probe_states,probe_records=manager._friction(u+1e-5*x,**kwargs)
    assert probe_states==() and all(r["first_touch"][0]==2 for r in probe_records)
    assert any(min(r["first_touch"][1])<0 for r in probe_records)
    physical=manager.prepare_commit(u+1e-5*x)
    assert all(min(h.barycentric)>=0 for h in physical[1])
    with pytest.raises(ValueError,match="fraction resolution"):
        manager._friction(u+.001*z,**kwargs)
    assert manager.checkpoint_state()==before
