"""Bounded G5 actual-route integration and regression evidence."""
from hashlib import sha256

import numpy as np
import pytest

import anyfesolver
from anyfesolver import (
    BoundaryCondition, FEModel, FixedSupport, LoadCase, TransientConfig,
    assemble_damping_matrix, assemble_load_vector, assemble_mass_matrix,
    assemble_stiffness_matrix, solve_eigenvalue_buckling,
    solve_free_vibration, solve_linear, solve_transient_newmark,
)
from anyfesolver.beam_sections import GeneralizedBeamSection
from anyfesolver.boundary import LoadCombination
from anyfesolver.elements import BeamElement, QuadraticBeamElement, create_element
from anyfesolver.ge_beam3_element import (
    GE_BEAM3_QUALIFIED_FORMULATION_ID, GeBeam3IntegrationError,
    GeometricallyExactBeam3D3NElement,
)
from anyfesolver.ge_beam3_mixed_element import GeBeam3MixedStateError
from anyfesolver._ge_beam3_g5_completion import evidence
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver import ge_beam3_native as native
from test_ge_beam3_public_workflows import make as native_owner


SCIENTIFIC_RECORDS = []


def _scalar_section():
    return {"area": 1.0, "Iy": 2.0e-3, "Iz": 1.5e-3, "J": 1.0e-3,
            "consistent_mass": True}


def _generalized_section():
    return GeneralizedBeamSection(
        np.diag((1.2e5, 4.0e4, 3.5e4, 2.0e3, 4.5e3, 5.0e3)),
        mass_matrix=np.diag((2.0, 2.0, 2.0, .04, .05, .06)),
        name="G5_REFERENCE_SECTION",
    )


def _model(family):
    model = FEModel("g5-" + family)
    model.add_material("mat", 2.1e5, .3, density=2.0)
    if family == "B2":
        model.add_node(1, 0., 0., 0.); model.add_node(2, 2., 0., 0.)
        element = BeamElement(1, [1, 2], "mat", _scalar_section())
        free_nodes = [2]; end = 2
    else:
        model.add_node(1, 0., 0., 0.); model.add_node(2, 1., 0., 0.)
        model.add_node(3, 2., 0., 0.)
        if family == "B3":
            element = QuadraticBeamElement(1, [1, 2, 3], "mat", _scalar_section())
        else:
            element = create_element("ge-beam3", 1, [1, 2, 3], "mat",
                section=_generalized_section(), reference_orientation=(0., 1., 0.),
                reference_axis_direction=(1., 0., 0.))
        free_nodes = [2, 3]; end = 3
    model.add_element(1, element)
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("axial", free_nodes,
        {name: 0. for name in ("uy", "uz", "rx", "ry", "rz")}))
    return model, element, end


def _exercise_reference_family(family):
    model, element, end = _model(family)
    stiffness, _ = assemble_stiffness_matrix(model)
    mass, _ = assemble_mass_matrix(model)
    assert stiffness.shape == mass.shape
    assert np.all(np.isfinite(stiffness.data)) and np.all(np.isfinite(mass.data))
    load = LoadCase(family + "-nodal")
    load.add_nodal_load(end, forces=(1., 0., 0.), moments=(0., 0., .25))
    solution, info = solve_linear(model, load)
    assert info["convergence_info"]["status"] == "converged"
    assert np.all(np.isfinite(solution))
    modes = solve_free_vibration(model, num_modes=1)
    assert modes.solver_status == "ok" and modes.frequencies_hz[0] > 0.
    transient = solve_transient_newmark(model,
        TransientConfig(dt=2.0e-4, t_end=1.0e-3), base_load_case=load)
    assert transient.status == "completed"
    assert np.all(np.isfinite(transient.displacements))
    damping, _ = assemble_damping_matrix(model, .01, .002)
    assert np.all(np.isfinite(damping.data))
    return dict(stiffness=stiffness, mass=mass, load=load, element=element,
                modal=modes, transient=transient)


def _record(tmp_path, route, **checks):
    row = evidence(route, **checks)
    SCIENTIFIC_RECORDS.append(row)
    (tmp_path / (route.lower() + ".json")).write_bytes(canonical(row))


def test_actual_legacy_b2_reference_routes(tmp_path):
    result = _exercise_reference_family("B2")
    _record(tmp_path, "LEGACY_B2_REFERENCE_ANALYSIS", reference_static=True,
            consistent_mass=True, modal=True, rayleigh_damping=True,
            linear_transient=True, unchanged_concrete_type=type(result["element"]) is BeamElement)


def test_actual_legacy_b3_reference_routes(tmp_path):
    result = _exercise_reference_family("B3")
    _record(tmp_path, "LEGACY_B3_REFERENCE_ANALYSIS", reference_static=True,
            consistent_mass=True, modal=True, rayleigh_damping=True,
            linear_transient=True,
            unchanged_concrete_type=type(result["element"]) is QuadraticBeamElement)


def test_ge_reference_load_inertia_and_combinations(tmp_path):
    model, element, end = _model("GE")
    spatial = element.compute_reference_line_load(model.mesh, {
        "classification": "SPATIAL_DEAD",
        "force_per_reference_length_at_nodes": ((1., 0., 0.),) * 3,
        "couple_per_reference_length_at_nodes": ((0., 0., .5),) * 3,
    })
    material = element.compute_reference_line_load(model.mesh, {
        "classification": "MATERIAL_DEAD",
        "force_per_reference_length_at_nodes": ((1., 0., 0.),) * 3,
        "couple_per_reference_length_at_nodes": ((0., 0., .5),) * 3,
    })
    np.testing.assert_allclose(spatial, material, rtol=0., atol=1.e-13)
    dead = LoadCase("dead"); dead.element_loads[1] = spatial
    dead.add_nodal_load(end, forces=(.5, 0., 0.), moments=(0., .25, 0.))
    dead.set_gravity(0., 0., -3.)
    assembled, _ = assemble_load_vector(model, dead)
    assert np.all(np.isfinite(assembled)) and np.linalg.norm(assembled) > 0.
    combined = LoadCombination("uls", {"dead": 1.5}).get_combined_load_vector(
        [dead], model.mesh, model.mesh.dof_manager,
        material_getter=model.get_material)
    np.testing.assert_allclose(combined, 1.5 * assembled, rtol=0., atol=1.e-12)
    before, _ = assemble_mass_matrix(model)
    model.add_point_mass(end, .75)
    after, info = assemble_mass_matrix(model)
    assert info["diagnostics"]["point_mass_count"] == 1
    assert after.diagonal().sum() - before.diagonal().sum() == pytest.approx(2.25)
    edge = LoadCase("edge-mass"); edge.add_distributed_edge_mass((2, 3), 1.0)
    edge.set_acceleration(1., 0., 0.)
    edge_load, _ = assemble_load_vector(model, edge)
    assert np.sum(edge_load[[model.mesh.get_node(2).dofs[0], model.mesh.get_node(3).dofs[0]]]) > 0.
    _record(tmp_path, "GE_REFERENCE_LOAD_INERTIA", spatial_dead=True,
            material_dead=True, nodal_force_and_couple=True, gravity=True,
            load_combination=True, consistent_generalized_mass=True,
            point_mass=True, distributed_edge_mass=True)


def test_ge_reference_modal_buckling_and_linear_transient(tmp_path):
    result = _exercise_reference_family("GE")
    model, element, _end = _model("GE")
    # Retain only the root clamp for a bending-admissible reference column;
    # the axial-only constraints used by the short transient would otherwise
    # remove every coordinate on which the geometric operator acts.
    model.boundary_conditions = model.boundary_conditions[:1]
    states = {1: {"axial_compression": 1.0}}
    buckling = solve_eigenvalue_buckling(model, states, num_modes=1)
    assert buckling.solver_status == "ok"
    assert buckling.critical_load_factor is not None
    gaps = element.capability_gaps
    assert not ({"buckling", "linear_transient_dynamics",
                 "reference_elastic_prestressed_modal"} & gaps)
    _record(tmp_path, "GE_REFERENCE_MODAL_BUCKLING_TRANSIENT", modal=True,
            physical_consistent_mass=True, reference_buckling=True,
            linear_newmark=True, rayleigh_damping=True,
            capability_declaration_matches=True,
            finite_outputs=bool(np.all(np.isfinite(result["transient"].displacements))))


def test_explicit_native_static_restart_recovery_and_provenance(tmp_path):
    definition, owner, direct = native_owner("generalized", False)
    pattern = native.DistributedPattern(
        native.LinePattern(((1, .001, -.0002, .0001),)), ())
    first = owner.solve_distributed(pattern, steps=1)
    expected = direct.solve_distributed(pattern, steps=1)
    assert first.status == expected.status == "completed"
    assert first.checkpoint == expected.checkpoint
    fresh = native.create_analysis(native.SELECTOR,
        (native.BeamDefinition.from_bytes(definition.raw,
            expected_sha256=definition.sha256),), tuple(owner.model.boundary_conditions))
    recovery = fresh.recover(first.checkpoint,
        expected_sha256=first.checkpoint_sha256)
    provenance = native.workflow_provenance(fresh)
    assert sha256(provenance).hexdigest() == sha256(native.workflow_provenance(fresh)).hexdigest()
    assert canonical(recovery) == canonical(direct.recover(expected.checkpoint,
        expected_sha256=expected.checkpoint_sha256))
    _record(tmp_path, "GE_NATIVE_STATIC_RESTART_RECOVERY", finite_static=True,
            authenticated_restart=True, physical_recovery=True,
            deterministic_provenance=True, accepted_g1_g4_owner=True)


def test_unsupported_routes_fail_closed_before_state_mutation(tmp_path, monkeypatch):
    model, element, _ = _model("GE")
    entered = False
    def forbidden(*args, **kwargs):
        nonlocal entered; entered = True
        raise AssertionError("candidate mechanics entered")
    monkeypatch.setattr(element, "evaluate_candidate", forbidden)
    with pytest.raises(GeBeam3MixedStateError, match="supports only"):
        element.compute_reference_line_load(model.mesh, {
            "classification": "CONSERVATIVE_FOLLOWER",
            "force_per_reference_length_at_nodes": ((1., 0., 0.),) * 3,
            "couple_per_reference_length_at_nodes": ((0., 0., 0.),) * 3,
        })
    assert entered is False
    for route in ("FINITE_ROTATION_TRANSIENT", "GYROSCOPIC_TERMS",
                  "CURRENT_STATE_MODAL", "CURRENT_STATE_BUCKLING"):
        with pytest.raises(GeBeam3MixedStateError):
            element.require_private_analysis_route(route)
    with pytest.raises(ValueError):
        create_element("ge-beam3", 2, [1, 2], "mat", section=_generalized_section(),
            reference_orientation=(0., 1., 0.), reference_axis_direction=(1., 0., 0.))
    _record(tmp_path, "UNSUPPORTED_ROUTES_FAIL_CLOSED", follower_before_mechanics=True,
            finite_rotation_transient=True, gyroscopic_terms=True,
            current_state_generic_modes=True, wrong_topology=True)


def test_selector_and_provenance_boundary_remains_additive(tmp_path):
    _gm, ge, _ = _model("GE"); _b2m, b2, _ = _model("B2"); _b3m, b3, _ = _model("B3")
    assert type(ge) is GeometricallyExactBeam3D3NElement
    assert ge.formulation_id == GE_BEAM3_QUALIFIED_FORMULATION_ID
    assert type(b2) is BeamElement and type(b3) is QuadraticBeamElement
    assert create_element("beam", 7, [1, 2], "mat").__class__ is BeamElement
    assert create_element("quadratic_beam", 8, [1, 2, 3], "mat").__class__ is QuadraticBeamElement
    assert "ge-beam3" not in anyfesolver.elements.ELEMENT_TYPES
    with pytest.raises((ValueError, GeBeam3IntegrationError)):
        native.create_analysis("beam", (), ())
    _record(tmp_path, "SELECTOR_AND_PROVENANCE_BOUNDARY", explicit_ge_only=True,
            legacy_aliases_unchanged=True, no_default_change=True,
            no_production_claim=True, native_selector_distinct=True)
