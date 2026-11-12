"""Independent uniform-pressure wrench and existing-driver multi-pair work."""
from dataclasses import replace
import json
import numpy as np
import pytest
from scipy import sparse

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction, ContactPathControl,
    ContactPathProgram, ContactSurface, FEModel, create_shell_element,
    DisplacementControl, LoadCase, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.general_contact.multiple_projected_pairs import (
    MultipleProjectedPairManager, MultiplePairResults, PROFILE,
)
from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes, load_nonlinear_checkpoint
from test_general_contact_projected_penalty import _candidate


def build(*, rigid_normal=True):
    settings = _candidate()[2]
    model = FEModel("two_explicit_partial_patches")
    model.add_material("elastic", 1e5, .3, density=1.)
    surfaces, interactions = [], []
    xy = ((.1,.1),(.6,.1),(1.1,.1),(.1,1.1),(.6,1.1),(1.1,1.1),
          (0.,0.),(1.,0.),(1.,1.),(0.,1.))
    for part in range(2):
        shift, node_offset, element_offset = 3.*part, 10*part, 4*part
        for node, (x,y) in enumerate(xy, 1):
            model.add_node(node+node_offset, x+shift, y, .0401 if node<=6 else 0.)
        for eid, nodes in ((1,(1,2,5,4)),(2,(2,3,6,5)),(3,(7,8,10)),(4,(8,9,10))):
            extra = {} if eid<=2 else dict(formulation="e4-pl-s3-v2d",reference_surface_offset=.01)
            model.add_element(eid+element_offset,create_shell_element(
                eid+element_offset,tuple(n+node_offset for n in nodes),"elastic",thickness=.05,
                reference_normal=(0.,0.,1.),**extra))
        source_nodes=tuple(n+node_offset for n in range(1,7))
        target_nodes=tuple(n+node_offset for n in range(7,11))
        model.add_boundary_condition(BoundaryCondition(f"guides_{part}",source_nodes,
            {name:0. for name in ("ux","uy","rx","ry","rz")}))
        model.add_boundary_condition(BoundaryCondition(f"fixed_target_{part}",target_nodes,
            {name:0. for name in ("ux","uy","uz","rx","ry","rz")}))
        master=int(model.mesh.dof_manager.get_node_dofs(source_nodes[0])[2])
        if rigid_normal:
            for node in source_nodes[1:]:
                slave=int(model.mesh.dof_manager.get_node_dofs(node)[2])
                model.add_constraint_equation(terms=((slave,1.),(master,-1.)),source_id=f"rigid_press_{node}")
        else:
            model.add_boundary_condition(BoundaryCondition(
                f"physical_clamp_{part}",(1+node_offset,4+node_offset),{"uz":0.}))
        surfaces.extend((ContactSurface(f"source_{part}",(1+element_offset,2+element_offset),"bottom"),
                         ContactSurface(f"target_{part}",(3+element_offset,4+element_offset),"top")))
        interactions.append(ContactInteraction(f"pair_{part}",f"source_{part}",f"target_{part}",
            1e7,5e-5,normal_law="projected_penalty"))
    definition=ContactDefinition(tuple(surfaces),(),tuple(interactions))
    program=ContactPathProgram(controls=(ContactPathControl.node(1,"uz"),ContactPathControl.node(11,"uz")),
        waypoints=((0.,0.),(-130e-6,-120e-6),(0.,0.),(-130e-6,-120e-6)),steps_per_segment=1)
    return model,definition,settings,program


def test_uniform_pressure_wrench_complete_both_body_tangent_and_pair_identity():
    model,definition,settings,_=build()
    manager=MultipleProjectedPairManager(model,definition,settings)
    assert validate_general_contact(model,definition,settings).profile==PROFILE
    u=np.zeros(manager.n_dof)
    direction=np.zeros_like(u)
    variations=[]
    for part,press in enumerate((-130e-6,-120e-6)):
        for node in range(1+10*part,7+10*part):
            dof=model.mesh.dof_manager.get_node_dofs(node)[2]
            u[dof]=press;direction[dof]=.1*(part+1);variations.append(dof)
        for node in range(7+10*part,11+10*part):
            dof=model.mesh.dof_manager.get_node_dofs(node)[2]
            direction[dof]=-.05;variations.append(dof)
    manager.restrict_tangent_to_reduced_variations(sparse.eye(manager.n_dof,format="csr")[:,variations])
    residual,tangent,cert=manager.evaluate(u)
    assert [name for name,_ in cert.pairs]==["pair_0","pair_1"]
    for part,(_,contact) in enumerate(cert.pairs):
        # Original rectangular geometry: .9*.9 overlap, fixed skins, constant gap.
        force=(300. if part==0 else 200.)*.81
        np.testing.assert_allclose(contact.integration_force,(0,0,force),atol=1e-9)
        np.testing.assert_allclose(contact.target_force,(0,0,-force),atol=1e-9)
        np.testing.assert_allclose(contact.integration_moment,(.55*force,-(.55+3*part)*force,0),atol=1e-8)
    np.testing.assert_allclose(np.add(cert.integration_force,cert.target_force),0,atol=1e-9)
    np.testing.assert_allclose(np.add(cert.integration_moment,cert.target_moment),0,atol=1e-8)
    assert len(cert.to_dict()["pairs"])==2
    analytic=tangent@direction
    for step in (1e-7,3e-8):
        plus=manager.evaluate(u+step*direction,tangent=False)[0]
        minus=manager.evaluate(u-step*direction,tangent=False)[0]
        error=np.linalg.norm((plus-minus)/(2*step)-analytic)/np.linalg.norm(analytic)
        assert error<1e-5
    # Pair order is not a candidate-list identity or a different restart model.
    reordered=replace(definition,interactions=definition.interactions[::-1],surfaces=definition.surfaces[::-1])
    other=MultipleProjectedPairManager(model,reordered,settings)
    assert other.restart_contract()==manager.restart_contract()
    np.testing.assert_array_equal(other.evaluate(u,tangent=False)[0],residual)


def test_duplicate_features_and_aggregate_limits_fail_before_a_solve():
    model,definition,settings,_=build()
    duplicate=replace(definition,interactions=(definition.interactions[0],replace(
        definition.interactions[0],name="duplicate")))
    assert any("duplicate physical" in issue for issue in validate_general_contact(model,duplicate,settings).issues)
    for change in (dict(maximum_candidate_pairs=7),dict(maximum_patch_normal_dofs=11)):
        report=validate_general_contact(model,definition,replace(settings,**change))
        assert not report.valid and any("aggregate" in issue for issue in report.issues)
    different=replace(definition,interactions=(definition.interactions[0],replace(
        definition.interactions[1],normal_law="augmented")))
    assert not validate_general_contact(model,different,settings).valid


def test_every_pair_keeps_its_dimensional_gap_gate(monkeypatch):
    model,definition,settings,_=build()
    manager=MultipleProjectedPairManager(model,definition,settings)
    calls=[]
    for index,(_,child) in enumerate(manager.pairs):
        def gate(*args,index=index):
            calls.append(index)
            return index!=0
        monkeypatch.setattr(child,"swept_increment_admissible",gate)
    zero=np.zeros(manager.n_dof)
    assert not manager.swept_increment_admissible(zero,zero,0.,1.)
    assert calls==[0,1]


def test_nested_numpy_scalar_certificates_export_with_standard_json():
    from types import SimpleNamespace
    zero=(np.float64(0.),)*3
    payload={"gap_tolerance_met":np.bool_(True),"active_pressure_dofs":np.int64(2),
             "numerical_pressure_pa":[np.float64(300.)]}
    child=SimpleNamespace(minimum_gap=np.float64(-3e-5),gap_tolerance_met=np.bool_(True),
        active_pressure_dofs=np.int64(2),integration_force=zero,target_force=zero,
        integration_moment=zero,target_moment=zero,pair_geometry_evaluations=np.int64(4),
        regularization_storage_j=np.float64(.001),to_dict=lambda:payload)
    saved=json.loads(json.dumps(MultiplePairResults(PROFILE,(("pair_0",child),)).to_dict()))
    assert saved["pairs"]==[{"interaction":"pair_0","contact":{
        "gap_tolerance_met":True,"active_pressure_dofs":2,"numerical_pressure_pa":[300.]}}]


def test_actual_analysis_control_is_forwarded_to_composite_admission():
    from anyfesolver.general_contact.core import contact_scope
    model,definition,settings,_=build()
    assert validate_general_contact(model,definition,settings).valid
    report=validate_general_contact(model,definition,settings,analysis_control="arc_length")
    assert not report.valid
    with pytest.raises(ValueError) as caught:
        with contact_scope(model,definition,settings,analysis_control="arc_length"):
            raise AssertionError("unsupported contributor must never activate")
    assert report.issues[0] in str(caught.value)


def test_shared_source_coordinates_sum_distinct_target_work_and_tangents():
    model,definition,settings,_=build()
    # Adjacent, independently named target patches support different portions
    # of the same source. Their common geometric edge is not a shared node.
    for node in range(17,21):
        model.mesh.get_node(node).x-=2.
    definition=replace(definition,surfaces=(definition.surfaces[0],
        definition.surfaces[1],definition.surfaces[3]),interactions=(
        definition.interactions[0],replace(definition.interactions[1],surface="source_0")))
    manager=MultipleProjectedPairManager(model,definition,settings)
    u=np.zeros(manager.n_dof); direction=np.zeros_like(u); variations=[]
    for nodes,displacement,variation in ((range(1,7),-130e-6,.2),
                                       (range(7,11),0.,-.1),
                                       (range(17,21),-10e-6,.05)):
        for node in nodes:
            dof=model.mesh.dof_manager.get_node_dofs(node)[2]
            u[dof]=displacement;direction[dof]=variation;variations.append(dof)
    manager.restrict_tangent_to_reduced_variations(sparse.eye(manager.n_dof,format="csr")[:,variations])
    residual,tangent,cert=manager.evaluate(u)
    np.testing.assert_allclose(cert.integration_force,(0.,0.,300.*.9*.9+200.*.1*.9),atol=1e-8)
    np.testing.assert_allclose(np.add(cert.integration_force,cert.target_force),0.,atol=1e-8)
    np.testing.assert_allclose(np.add(cert.integration_moment,cert.target_moment),0.,atol=1e-8)
    source_dofs=[model.mesh.dof_manager.get_node_dofs(node)[2] for node in range(1,7)]
    np.testing.assert_allclose(-sum(residual[source_dofs]),261.,atol=1e-8)
    analytic=tangent@direction
    for step in (1e-7,3e-8):
        difference=(manager.evaluate(u+step*direction,tangent=False)[0]
                   -manager.evaluate(u-step*direction,tangent=False)[0])/(2*step)
        assert np.linalg.norm(difference-analytic)/np.linalg.norm(analytic)<1e-5


def test_ordinary_force_control_activates_both_supported_pairs():
    model,definition,settings,_=build(rigid_normal=False)
    load=LoadCase("two_clamped_sources")
    # Independent structural K and exact Q1 overlap integrals: maximum weak
    # excursion is 105 um and physical excursion is 124.342 um, against 100 um
    # clearance and the unchanged 50 um physical-gap budget. A 105 um tip-only
    # excursion remains weakly open (88.667 um weighted excursion).
    force_per_node=0.066422064777328
    for part in range(2):
        for node in (2,3,5,6):
            load.add_nodal_load(node+10*part,forces=(0.,0.,-force_per_node))
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        control="force",num_steps=2,max_iterations=12,tolerance=1e-8,
        emit_restart_checkpoint=True)
    assert result.status=="completed",result.failure_reason
    summary=result.info["general_contact"]
    print(json.dumps({"status":result.status,"force_per_node_n":force_per_node,
        "minimum_gap_m":float(summary["minimum_gap"]),
        "pair_active_pressure_dofs":[int(pair["contact"]["active_pressure_dofs"])
                                     for pair in summary["pairs"]],
        "integration_force_n":[float(value) for value in summary["integration_force"]],
        "displacements":result.displacements.tolist()}))
    assert summary["gap_tolerance_met"]
    assert all(pair["contact"]["active_pressure_dofs"]>0 for pair in summary["pairs"])
    np.testing.assert_allclose(np.add(summary["integration_force"],summary["target_force"]),0.,atol=1e-8)
    np.testing.assert_allclose(np.add(summary["integration_moment"],summary["target_moment"]),0.,atol=1e-8)


def test_ordinary_displacement_publication_preserves_distinct_pair_settings():
    model,definition,settings,_=build()
    definition=replace(definition,interactions=(definition.interactions[0],replace(
        definition.interactions[1],normal_stiffness=2e7,gap_tolerance=8e-5)))
    # One declared displacement control drives two otherwise independent
    # source patches through an explicit normal tie. No hidden support.
    first=int(model.mesh.dof_manager.get_node_dofs(1)[2])
    second=int(model.mesh.dof_manager.get_node_dofs(11)[2])
    model.add_constraint_equation(terms=((second,1.),(first,-1.)),source_id="common_press")
    control=DisplacementControl(node_id=1,dof="uz",target_displacement=-130e-6)
    load=LoadCase("common_press_reference_force")
    load.add_nodal_load(1,forces=(0.,0.,-1.))
    result=solve_static_nonlinear(model,load_case=load,control="displacement",displacement_control=control,
        contact=definition,contact_settings=settings,num_steps=2,tolerance=1e-8,
        emit_restart_checkpoint=True)
    assert result.status=="completed",result.failure_reason
    summary=result.info["general_contact"]
    assert "gap_tolerance" not in summary and "normal_stiffness" not in summary
    assert summary["pair_enforcement"]==[
        {"interaction":"pair_0","normal_stiffness_pa_per_m":1e7,"gap_tolerance_m":5e-5},
        {"interaction":"pair_1","normal_stiffness_pa_per_m":2e7,"gap_tolerance_m":8e-5}]
    np.testing.assert_allclose(summary["integration_force"],(0,0,729.),rtol=1e-8,atol=1e-8)


def test_existing_static_driver_press_open_recontact_and_authentic_restart(tmp_path):
    model,definition,settings,program=build()
    result=solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        contact_path_program=program,tolerance=1e-8,emit_restart_checkpoint=True)
    assert result.status=="completed",result.failure_reason
    history=result.info["force_displacement_history"]
    assert [row["contact"]["active_pressure_dofs"] for row in history]==[12,0,12]
    # Control coordinates are coupled through MPCs, so reactions integrate all
    # six physical source-node forces for each independently pressed patch.
    # Actuators apply downward forces balancing upward contact forces.
    np.testing.assert_allclose(history[-1]["reaction_generalized"],(-243.,-162.),rtol=1e-8,atol=1e-8)
    model,definition,settings,program=build()
    opened=solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        contact_path_program=program,max_load_factor=2/3,tolerance=1e-8,emit_restart_checkpoint=True)
    assert opened.status=="completed",opened.failure_reason
    path=tmp_path/"open.checkpoint.json"
    path.write_bytes(canonical_checkpoint_json_bytes(opened.restart_checkpoint))
    checkpoint=load_nonlinear_checkpoint(path.read_bytes())
    model,definition,settings,program=build()
    resumed=solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        contact_path_program=program,restart_checkpoint=checkpoint,tolerance=1e-8,emit_restart_checkpoint=True)
    assert resumed.status=="completed",resumed.failure_reason
    np.testing.assert_array_equal(resumed.displacements,result.displacements)
    assert resumed.restart_checkpoint["element_states"]==result.restart_checkpoint["element_states"]
    assert resumed.info["force_displacement_history"]==history
