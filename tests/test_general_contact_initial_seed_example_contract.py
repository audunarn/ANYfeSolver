"""Static public-example contract: model, route, seed wiring, fixture parity.

The end-to-end execution of the example (one real coupled half solve) is
executed by the sibling public-example subprocess check, not here. This
module verifies the example's static public contract on the same
immutable snapshot: the built model/settings/program are the fixture's
D1 overlap-local route, the default-none selector keeps the legacy
shape, and the enabled selector is admitted by real validation on the
example's own model. It also pins the focused physical parity demanded
by this delivery: the example's independently constructed fixture IS
the root-adopted 4x4 Q4 resolution candidate of the restart's
small-patch driver - identical physical node coordinates, element
connectivity and shell thickness, materials, perimeter/lateral/target
supports, contact definition, effective settings and two-step -0.04 N
ramp program, with the remapped load node 13 at (.6,.6) and ux anchors
1 and 11 at (.1,.1)/(.1,.6) asserted against the actual mesh
coordinates. The example file itself imports nothing from the tests
tree; only this test-side check imports the driver fixture. No solve
is executed here and no trajectory claim is made.
"""
import importlib.util
import os
from dataclasses import asdict, replace

import numpy as np

from anyfesolver import validate_general_contact
from anyfesolver.general_contact.core import _contact_settings_contract

from test_general_contact_admission_globalization_driver_small_patch import (
    _ANCHOR_NODES,
    _LOAD_NODE,
    _case as _driver_case,
    _enabled as _driver_enabled,
)

_EXAMPLE_NAME = "general_contact_initial_contact_seed_dev.py"


def _load_example():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo, "examples", _EXAMPLE_NAME)
    assert os.path.isfile(path), path
    spec = importlib.util.spec_from_file_location(
        "initial_contact_seed_dev_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_builds_the_valid_d1_route_with_seed_wiring():
    example = _load_example()
    model, definition, settings, program = example.build_model()
    # The example's own settings carry the D1 reference-linear fine-Q1
    # monolithic Q4/S3 overlap-local route with the joint certificate.
    assert settings.initial_contact_seed == "none"
    assert settings.q4_source_gap_certificate == "overlap_local_dev"
    assert settings.q4_source_overlap_dual_dev is True
    assert settings.general_s3_differentiation == "jet2_q4_s3_dev"
    assert settings.enforcement_solver == "monolithic_active_set"
    assert settings.d1_interior_profile is True
    # Default-none keeps the exact legacy serialized contract shape.
    contract = _contact_settings_contract(settings)
    assert "initial_contact_seed" not in contract
    # The enabled selector is admitted by real validation on the
    # example's own model, program and settings.
    seeded = replace(settings, initial_contact_seed="linearized_kkt_dev")
    report = validate_general_contact(model, definition, seeded,
                                      analysis_control="contact_path")
    assert report.valid, report.issues
    enabled_contract = _contact_settings_contract(seeded)
    assert enabled_contract["initial_contact_seed"] == "linearized_kkt_dev"
    # The program is the fixture's two-step 0.04 N ramp with two
    # controls; the settings serialize the mandated route fields.
    record = asdict(seeded)
    assert record["complementarity_force_scale_n"] == 0.04
    assert program.steps_per_segment == 2
    assert len(program.controls) == 2
    assert len(program.nodal_forces) == 1


def test_example_fixture_parity_with_root_adopted_4x4_driver():
    """The example's fixture is the root-adopted 4x4 driver fixture."""
    example = _load_example()
    model, definition, settings, program = example.build_model()
    ref_model, ref_definition, ref_settings, ref_program = _driver_case()
    ref_settings = _driver_enabled(ref_settings)
    # Physical coordinates: every node position is bit-identical.
    coordinates = model.mesh.get_node_coordinates()
    ref_coordinates = ref_model.mesh.get_node_coordinates()
    assert coordinates.shape == ref_coordinates.shape
    np.testing.assert_array_equal(coordinates, ref_coordinates)
    # Mesh: same element ids, connectivity and shell thickness
    # (16 Q4 source elements + 2 S3 target faces).
    connectivity = model.mesh.get_connectivity()
    assert connectivity == ref_model.mesh.get_connectivity()
    assert len(connectivity) == 18
    for element_id, element in model.mesh.elements.items():
        ref_element = ref_model.mesh.elements[element_id]
        assert element.node_ids == ref_element.node_ids
        assert element.thickness == ref_element.thickness == 0.05
    # Materials: the elastic material matches exactly.
    assert model.materials == ref_model.materials
    elastic = model.materials["elastic"]
    assert elastic.elastic_modulus == 1.0e5
    assert elastic.poisson_ratio == 0.3
    assert elastic.density == 1.0
    # Supports: identical boundary conditions (source lateral uy
    # guide, source perimeter uz support, target control clamp).
    assert model.boundary_conditions == ref_model.boundary_conditions
    # Contact definition: same surfaces, interaction, stiffness,
    # gap tolerance and normal law.
    assert definition == ref_definition
    # Parameters: the effective settings are exactly the driver's
    # enabled D1/Jet2/fine-Q1/monolithic overlap-local route.
    assert asdict(settings) == asdict(ref_settings)
    # Load: the remapped physical positions on the actual mesh, not
    # grid indices.
    assert program.nodal_forces[0].node_id == _LOAD_NODE == 13
    assert [term[0] for control in program.controls
            for term in control.terms] == list(_ANCHOR_NODES) == [1, 11]
    loaded = model.mesh.get_node(_LOAD_NODE)
    assert abs(loaded.x - 0.6) < 1e-12 and abs(loaded.y - 0.6) < 1e-12
    anchors = [model.mesh.get_node(node) for node in _ANCHOR_NODES]
    assert [(anchor.x, anchor.y) for anchor in anchors] == [
        (0.1, 0.1), (0.1, 0.6)]
    # Program: the unchanged two-step -0.04 N ramp and controls.
    assert program.controls == ref_program.controls
    assert program.waypoints == ref_program.waypoints
    assert program.steps_per_segment == ref_program.steps_per_segment == 2
    assert program.nodal_forces == ref_program.nodal_forces
    assert program.nodal_forces[0].dof == "uz"
    assert program.nodal_forces[0].values_n == (0.0, -0.04)
