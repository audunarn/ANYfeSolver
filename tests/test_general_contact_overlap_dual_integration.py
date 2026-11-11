"""Opt-in settings, exact default contracts and production caller wiring."""
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from anyfesolver.general_contact.core import ContactSolverSettings, _contact_settings_contract
from anyfesolver.general_contact import s3_patch_static as owner
from test_general_contact_predictor_compatibility import legacy_settings


def settings(**changes):
    values = dict(d1_interior_profile=True,
        d1_interior_coverage_tolerance_m2=1e-12,
        d1_interior_boundary_force_tolerance_n=1e-10,
        q4_source_gap_certificate='overlap_local_dev',
        general_s3_differentiation='jet2_q4_s3_dev',
        enforcement_solver='monolithic_active_set',
        complementarity_force_scale_n=1., pair_motion='general')
    values.update(changes)
    return ContactSolverSettings(**values)


def test_default_generic_settings_contract_retains_exact_old_fieldset():
    default = ContactSolverSettings()
    expected = legacy_settings(default)
    for name, value in (('path_predictor', 'none'), ('shell_pair_orientation', 'legacy'),
                        ('shell_skin_kinematics', 'reference_linear'),
                        ('finite_frame_seed', 'common_reference_frame_dev')):
        assert getattr(default, name) == value
    assert _contact_settings_contract(default) == expected
    enabled = settings(q4_source_overlap_dual_dev=True)
    assert _contact_settings_contract(enabled)['q4_source_overlap_dual_dev'] is True


@pytest.mark.parametrize('invalid', (None, 0, 1, 'true'))
def test_exact_boolean_setting_refuses_invalid_types(invalid):
    with pytest.raises(TypeError, match='overlap_dual_dev must be bool'):
        settings(q4_source_overlap_dual_dev=invalid)


@pytest.mark.parametrize('changed', (
    {'d1_interior_profile': False},
    {'q4_source_gap_certificate': 'whole_face'},
    {'general_s3_differentiation': 'jet2_s3_dev'},
    {'q4_source_pressure_basis': 'macro_q1_pair_dev'},
    {'enforcement_solver': 'nested_augmented'},
    {'normal_discretization': 'pointwise_penalty'},
    {'shell_skin_kinematics': 'corotational_nodal_director_dev'},
    {'shell_pair_orientation': 'physical_sides_dev'},
))
def test_opt_in_refuses_other_setting_profiles(changed):
    with pytest.raises(ValueError, match='overlap dual requires'):
        settings(q4_source_overlap_dual_dev=True, **changed)


def caller_fixture(enabled):
    source = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), float)
    target = np.array(((-2, -2, 0), (4, -2, 0), (-2, 4, 0)), float)
    law = SimpleNamespace(source_face_indices=(np.arange(4),), target_face_indices=(np.arange(3),),
        pair_laws=(0,), _check_same_side_overlap=Mock(), _candidate_pair_indices=Mock(return_value=None))
    instance = SimpleNamespace(settings=settings(q4_source_overlap_dual_dev=enabled),
        projected_work=False, law=law, gap_tolerance=5e-5,
        profile='D1_normal', _mesh_physical_ray_owner=None)
    return instance, (source, target)


@pytest.mark.parametrize('enabled', (False, True))
def test_actual_point_and_full_caller_use_one_method_with_distinct_schedules(monkeypatch, enabled):
    instance, positions = caller_fixture(enabled)
    monkeypatch.setattr(owner, 'q4_source_projected_gap_lower', Mock(return_value=-1e-3))
    monkeypatch.setattr(owner, 'overlap_local_lower', Mock(return_value=(-1e-3, 1)))
    matching = Mock(return_value=(-1e-6, 1)); monkeypatch.setattr(owner, 'matching_projection_swept_lower', matching)
    assert owner.GeneralS3PatchManager._point_gap_budget_admissible(instance, positions)
    point = matching.call_args.kwargs
    assert point['gap_budget'] == 5e-5 and 'refinement_gap_budget' not in point
    assert point.get('overlap_dual', False) is enabled
    assert ('overlap_dual' in point) is enabled  # default preserves exact old keyword set
    matching.reset_mock()
    assert owner.GeneralS3PatchManager._gap_lower(instance, positions) == -1e-6
    complete = matching.call_args.kwargs
    assert 'gap_budget' not in complete and complete['refinement_gap_budget'] == 5e-5
    assert complete.get('overlap_dual', False) is enabled
    assert ('overlap_dual' in complete) is enabled
    assert point['adaptive_depth'] == complete['adaptive_depth'] == 7
    assert point['node_budget'] == complete['node_budget'] == 1024


def test_enabled_actual_caller_refuses_unresolved_and_preserves_exceptions(monkeypatch):
    instance, positions = caller_fixture(True)
    monkeypatch.setattr(owner, 'q4_source_projected_gap_lower', Mock(return_value=-1e-3))
    monkeypatch.setattr(owner, 'overlap_local_lower', Mock(return_value=(-1e-3, 1)))
    matching = Mock(return_value=(-np.inf, 1)); monkeypatch.setattr(owner, 'matching_projection_swept_lower', matching)
    assert not owner.GeneralS3PatchManager._point_gap_budget_admissible(instance, positions)
    matching.side_effect = ValueError('original chart refusal')
    with pytest.raises(ValueError, match='original chart refusal'):
        owner.GeneralS3PatchManager._point_gap_budget_admissible(instance, positions)


def test_default_public_request_settings_preserve_original_asdict_shape():
    from test_general_contact_q4_s3_patch_static import _case
    from anyfesolver import identify_general_contact_request, ContactPathControl, ContactPathProgram
    model, definition, config = _case()
    assert config.q4_source_overlap_dual_dev is False
    program = ContactPathProgram((ContactPathControl.node(1, 'uz'),), ((0.,), (-.01,)))
    request = identify_general_contact_request(model, definition, config, contact_path_program=program).to_dict()['request']
    original = legacy_settings(config)
    # Public request identity retains these defaults, unlike the older
    # checkpoint contract. Later disabled seed/continuation fields stay absent.
    original.update(path_predictor='none', shell_pair_orientation='legacy',
                    shell_skin_kinematics='reference_linear',
                    finite_frame_seed='common_reference_frame_dev')
    assert request['settings'] == original
    assert 'contact_numerical_method_revision' not in request


def test_enabled_actual_dispatch_refuses_reversed_body_family():
    from test_general_contact_q4_s3_patch_static import _case
    from anyfesolver import validate_general_contact
    model, definition, _ = _case()
    pair = replace(definition.interactions[0], surface='s3_target', target='q4_source')
    incompatible = replace(definition, interactions=(pair,))
    config = settings(q4_source_overlap_dual_dev=True)
    report = validate_general_contact(model, incompatible, config, analysis_control='contact_path')
    assert not report.valid
    assert any('D1' in item or 'overlap dual' in item for item in report.issues)
    with pytest.raises(ValueError, match='D1|overlap dual'):
        owner.GeneralS3PatchManager(model, incompatible, config)
