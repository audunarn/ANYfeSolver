"""Native-package extraction and equivalence; not element qualification."""

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.nonlinear_static import solve_static_nonlinear
from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes
from anyfesolver._ge_beam3_p5 import NativeP5BeamElement, DirectedHardeningSection
from anyfesolver._ge_beam3_p5 import codec as package_codec
from anyfesolver._ge_beam3_p5.reference_modal import solve as package_modes
from anyfesolver._ge_beam3_p5.committed_modal import solve_elastic_modes as package_current_modes
from docs.reference_cases.ge_beam3_curved_p5_native_driver_probe import DriverP5Element
from docs.reference_cases.ge_beam3_curved_p5_native_material_probe import canonical
from docs.reference_cases.ge_beam3_curved_p5_native_modal_probe import solve as research_modes
from docs.reference_cases.ge_beam3_curved_p5_native_committed_modal_probe import solve_elastic_modes as research_current_modes
from docs.reference_cases import ge_beam3_curved_p5_package_extract as extraction
from test_ge_beam3_curved_p5_algebra_probe import reference
from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law
from test_ge_beam3_curved_p5_mass_probe import section_mass


def problem(packaged=True, *, yield_force=.02):
    model = FEModel('p5-package-equivalence'); ref = reference(.4); material = law(yield_force)
    if packaged:
        material = DirectedHardeningSection(material._elastic, material._direction, material._yield, material._hardening)
    for i, point in enumerate(ref.coordinates, 1): model.add_node(i, *point)
    cls = NativeP5BeamElement if packaged else DriverP5Element
    element = cls(1, (1,2,3), ref, material)
    model.add_element(1, element); model.materials[element.material_name] = element.core.section
    model.add_boundary_condition(BoundaryCondition('root', [1],
        {name:0. for name in ('ux','uy','uz','rx','ry','rz')}))
    load = LoadCase('tip'); load.add_nodal_load(3, forces=np.array([.025,-.01,.005]))
    return model, element, load


def run(model, load, **kwargs):
    result = solve_static_nonlinear(model, load, num_steps=kwargs.pop('num_steps',2), max_iterations=12, tolerance=1e-12,
        num_layers=1, min_step_fraction=1., emit_restart_checkpoint=True, **kwargs)
    assert result.status == 'completed', (result.status, result.info)
    return result


def test_exact_extraction_and_closed_package_dependency_graph():
    files = extraction.extract()
    for path, expected in files.items():
        assert (extraction.ROOT/path).read_text(encoding='utf-8') == expected
        if path.endswith('.py'):
            tree = ast.parse(expected)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or '').startswith(('docs','tests'))
                if isinstance(node, ast.Import):
                    assert not any(n.name.startswith(('docs','tests')) for n in node.names)
    package = extraction.ROOT/extraction.TARGET
    assert {p.name for p in package.glob('*.py')} == {Path(p).name for p in files if p.endswith('.py')}
    assert not (package/'history_path.py').exists()
    assert 'NativeMaterialSession' not in (package/'core.py').read_text()


@pytest.mark.parametrize('shallow', [True,False])
def test_missing_source_base_is_accepted_only_at_bound_shallow_boundary(monkeypatch, shallow):
    from types import SimpleNamespace
    monkeypatch.setattr(extraction.subprocess,'run',lambda *args,**kwargs: SimpleNamespace(returncode=1))
    monkeypatch.setattr(extraction.subprocess,'check_output',lambda *args,**kwargs: b'true\n' if shallow else b'false\n')
    if shallow:
        for path, content in extraction.extract().items():
            assert (extraction.ROOT/path).read_text(encoding='utf-8') == content
    else:
        with pytest.raises(ValueError,match='ordinary local repository'): extraction.extract()


@pytest.fixture(scope='module', params=[.02,1000.])
def compared(request):
    old, old_element, old_load = problem(False, yield_force=request.param)
    new, new_element, new_load = problem(True, yield_force=request.param)
    before = run(old, old_load); after = run(new, new_load)
    return old, old_element, before, new, new_element, after, request.param


def test_actual_newton_and_material_response_are_byte_identical(compared):
    _, _, before, _, element, after, _ = compared
    assert np.array_equal(before.displacements, after.displacements)
    assert element.formulation_id == 'CANDIDATE_GE_BEAM3_P5_NATIVE_PACKAGE_V1'
    original = before.element_states[1]['material_state']; made = after.element_states[1]['material_state']
    for key in ('committed_total_u','committed_positions','committed_nodal_rotation_matrices','origins','histories','response'):
        assert canonical(made[key]) == canonical(original[key]), key
    assert made['model_sha256'] != original['model_sha256']
    assert made['schema'] != original['schema']


def test_recovery_values_equal_while_provenance_is_a_successor(compared):
    old, old_element, before, new, new_element, after, _ = compared
    source = old_element.recover_native_fields(old.mesh, before.element_states[1])
    made = new_element.recover_native_fields(new.mesh, after.element_states[1])
    identity_keys = {'formulation_id','state_schema','driver_sha256','state_sha256'}
    for key in source.keys()-identity_keys: assert canonical(made[key]) == canonical(source[key]), key
    assert source['formulation_id'] != made['formulation_id']
    assert not made['production_qualified']


def test_old_research_history_cannot_hot_restart_into_packaged_candidate(compared):
    old, old_element, before, new, new_element, after, _ = compared
    with pytest.raises(ValueError):
        new_element.validate_model_bound_nonlinear_state(new.mesh, new_element.core.section, before.element_states[1], 1)
    encoded = new_element.serialize_native_material_state(new.mesh, after.element_states[1])
    assert encoded['schema'] == package_codec.SCHEMA
    with pytest.raises(ValueError):
        old_element.validate_model_bound_nonlinear_state(old.mesh, old_element.core.section, encoded, 1)


def test_reference_and_current_modal_operators_equal(compared):
    old, _, before, new, _, after, yield_force = compared
    source = research_modes(old, {1:section_mass()}); made = package_modes(new, {1:section_mass()})
    for field in ('full_stiffness','full_mass','eigenvalues','full_modes','dynamic_map'):
        assert np.array_equal(getattr(made,field), getattr(source,field)), field
    force = np.zeros(18); force[12:15] = [.025,-.01,.005]
    if yield_force == .02:
        with pytest.raises(ValueError, match='vibration branch'):
            package_current_modes(new, after.element_states, after.displacements, {1:section_mass()}, force)
    else:
        p0, m0 = research_current_modes(old,before.element_states,before.displacements,{1:section_mass()},force)
        p1, m1 = package_current_modes(new,after.element_states,after.displacements,{1:section_mass()},force)
        for field in ('stiffness','mass','internal_force'):
            assert np.array_equal(getattr(p0,field), getattr(p1,field))
        assert np.array_equal(m0.eigenvalues,m1.eigenvalues)


def test_packaged_native_checkpoint_split_continuation_is_exact():
    model, _, load = problem(); continuous = run(model, load)
    first_model, _, first_load = problem(); first = run(first_model, first_load, num_steps=1, max_load_factor=.5)
    resumed_model, element, resumed_load = problem()
    resumed = run(resumed_model, resumed_load, restart_checkpoint=canonical_checkpoint_json_bytes(first.restart_checkpoint))
    assert np.array_equal(continuous.displacements,resumed.displacements)
    assert canonical(continuous.element_states) == canonical(resumed.element_states)
    assert canonical_checkpoint_json_bytes(continuous.restart_checkpoint) == canonical_checkpoint_json_bytes(resumed.restart_checkpoint)
    with pytest.raises(ValueError): element.compute_mass_matrix(resumed_model.mesh, element.core.section)


def test_copy_is_owned_and_codec_still_rejects_noncanonical_state(compared):
    _, _, _, model, element, result, _ = compared
    state = result.element_states[1]
    encoded = element.serialize_native_material_state(model.mesh,state)
    roundtrip = package_codec.decode(encoded,order=element.core.order)
    assert canonical(roundtrip) == canonical(state)
    bad = deepcopy(encoded); bad['payload'] = bad['payload'].rstrip('\n')
    with pytest.raises(ValueError): package_codec.decode(bad,order=element.core.order)
    bad = deepcopy(encoded); bad['payload'] = '{"schema":0,"schema":0}'
    with pytest.raises(ValueError): package_codec.decode(bad,order=element.core.order)
