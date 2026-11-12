"""Legacy settings byte contracts and real opt-in public request routing."""
import copy
from dataclasses import replace
import hashlib,json,os
from pathlib import Path
import numpy as np
import pytest


# Independent pre-selector schema reference, transcribed from the local
# historical source (the pre-selector parent of the consolidation commit):
# ContactSolverSettings ended at beam_end_cap_work and no settings
# contract filter existed yet. path_predictor and the nine later opt-in
# fields below did not exist; they are admitted only when explicitly
# opted in, so the historical identity/default compat claims keep
# holding: default-disabled fields stay absent (prior serialized
# settings shapes are preserved byte-for-byte) and opted-in fields are
# retained. This reference is independent of the production filter; the
# tests cross-check the production contracts against it.
LEGACY_FIELDS=(
    'maximum_candidate_pairs','pair_motion','sliding_difference_step',
    'maximum_gap_triangles','maximum_patch_normal_dofs','maximum_augmentations',
    'pressure_tolerance','maximum_shell_tilt','line_integration_relative_tolerance',
    'maximum_line_subdivisions','maximum_swept_subdivisions','normal_discretization',
    'pointwise_integration_subdivisions','maximum_pointwise_triangles',
    'beam_shell_rim_clearance','beam_shell_difference_step','warped_q4_plane',
    'warped_q4_quadrature_order','warped_q4_control_order',
    'warped_q4_integration_tolerance','warped_q4_gap_accuracy',
    'warped_q4_maximum_gap_cells','warped_q4_maximum_gap_depth',
    'beam_pair_gap_accuracy','maximum_beam_pair_gap_cells','maximum_beam_pair_gap_depth',
    'general_s3_difference_step','general_s3_differentiation',
    'q4_s3_shared_edge_tie_step_m','q4_s3_shared_edge_tie_direction',
    'q4_s3_shared_edge_tie_relative_tolerance','maximum_general_s3_energy_evaluations',
    'q4_target_weighted_gap_tolerance','maximum_q4_target_integration_triangles',
    'q4_source_area_tolerance','q4_source_weighted_gap_tolerance',
    'maximum_q4_source_integration_triangles','q4_source_gap_certificate',
    'q4_source_pressure_basis','q4_source_pressure_knot_columns',
    'augmentation_acceleration','augmentation_pivot_max_pressure_pa',
    'augmentation_pivot_difference_step_m','augmentation_equilibrated_pressure_step_pa',
    'augmentation_schur_maximum_modes','augmentation_schur_maximum_pressure_pa',
    'augmentation_schur_maximum_step_pa','augmentation_schur_difference_step_m',
    'augmentation_schur_maximum_attempts','augmentation_schur_maximum_relinearizations',
    'augmentation_schur_nonlinear_backtrack_dev','augmentation_support_resolution',
    'augmentation_support_force_budget_n','enforcement_solver',
    'complementarity_force_scale_n','monolithic_maximum_pivots',
    'monolithic_linear_solver','q4_source_support_gradient_tolerance',
    'q4_source_support_hessian_tolerance','q4_source_weighted_gap_gradient_tolerance',
    'q4_source_weighted_gap_hessian_tolerance','d1_interior_profile',
    'd1_interior_coverage_tolerance_m2','d1_interior_boundary_force_tolerance_n',
    'beam_shell_geometry','beam_shell_maximum_swept_intervals','beam_chain_tangent',
    'beam_end_cap_work')
LATER_DEFAULTS={'shell_pair_orientation':'legacy',
                'shell_skin_kinematics':'reference_linear',
                'finite_frame_seed':'common_reference_frame_dev',
                'q4_source_overlap_dual_dev':False,
                'initial_contact_seed':'none'}
CONTINUATION_BLOCK=('path_continuation_maximum_halvings','path_stagnation_patience',
                    'path_stagnation_fraction','path_stagnation_merit_rate')


def legacy_settings(settings):
    # Independent pre-selector dataclass schema, preserving all prior fields.
    value={name:copy.deepcopy(getattr(settings,name)) for name in LEGACY_FIELDS}
    for name,default in LATER_DEFAULTS.items():
        if getattr(settings,name)!=default:
            value[name]=copy.deepcopy(getattr(settings,name))
    if getattr(settings,'path_continuation_maximum_halvings')!=0:
        for name in CONTINUATION_BLOCK:
            value[name]=copy.deepcopy(getattr(settings,name))
    return value


@pytest.mark.parametrize('kind',('caps','transient','rigid'))
def test_actual_manager_contract_matches_pre_selector_schema(kind,monkeypatch):
    if kind=='caps':
        from examples.general_contact_multiple_beam_s3_caps import build
        from anyfesolver.general_contact import multiple_beam_s3_caps as module
        model,definition,settings,_=build()
        manager=module.MultipleBeamS3CapManager(model,definition,settings)
        contract=manager.restart_contract()
        assert 'path_predictor' not in contract['settings']
        assert all('path_predictor' not in p['effective_settings'] for p in contract['pairs'])
        monkeypatch.setattr(module,'_contact_settings_contract',legacy_settings)
        assert manager.restart_contract()==contract
    elif kind=='transient':
        from examples.general_contact_multiple_beam_s3_transient import build
        from anyfesolver.general_contact import multiple_beam_s3_transient as module
        model,definition,settings,_=build()
        manager=module.MultipleBeamS3TransientManager(model,definition,settings)
        contract=manager.restart_contract()
        assert 'path_predictor' not in contract['settings']
        monkeypatch.setattr(module,'_contact_settings_contract',legacy_settings)
        assert manager.restart_contract()==contract
    else:
        from examples.general_contact_triangulated_rigid import case
        from anyfesolver.general_contact import triangulated_rigid_static as module
        manager=module.TriangulatedRigidManager(*case())
        contract=manager._contract_bytes()
        assert 'path_predictor' not in json.loads(contract)['settings']
        monkeypatch.setattr(module,'_contact_settings_contract',legacy_settings)
        assert manager._contract_bytes()==contract
        assert manager.restart_contract()['rigid_mesh_contract_sha256']==hashlib.sha256(contract).hexdigest()


def test_common_settings_contract_keeps_optin_and_all_prior_fields():
    from anyfesolver import ContactSolverSettings
    from anyfesolver.general_contact.core import _contact_settings_contract
    from anyfesolver.general_contact.automatic_domain import _settings_contract
    settings=ContactSolverSettings(path_predictor='control_projection_dev',maximum_swept_subdivisions=9)
    expected={**legacy_settings(settings),'path_predictor':'control_projection_dev'}
    assert _contact_settings_contract(settings)==expected==_settings_contract(settings)
    default=replace(settings,path_predictor='none')
    assert _contact_settings_contract(default)==legacy_settings(default)==_settings_contract(default)
    value=_contact_settings_contract(settings);value['maximum_swept_subdivisions']=999
    assert settings.maximum_swept_subdivisions==9


def test_public_optin_request_identity_then_actual_first_press():
    from anyfesolver import identify_general_contact_request,verify_general_contact_request,LoadCase,ContactRequestIdentityError
    from examples.general_contact_automatic_domain import build_projected_shell_pair,solve_shell_pair
    model,domain,settings,resolution,path=build_projected_shell_pair(lifecycle=True,
        path_predictor='control_projection_dev')
    controls=dict(contact_path_program=path,max_load_factor=1/3,max_iterations=15)
    identity=identify_general_contact_request(model,resolution.definition,settings,**controls)
    assert verify_general_contact_request(identity,model,resolution.definition,settings,**controls)==identity
    record=identity.to_dict()
    assert record['request']['loading']['kind']=='contact_path'
    assert record['request']['settings']['path_predictor']=='control_projection_dev'
    assert not identity.admitted_for_application
    load=LoadCase('predictor must not convert dead loads');load.add_nodal_load(1,forces=(0.,0.,-.1))
    with pytest.raises(ContactRequestIdentityError,match='path_predictor requires'):
        identify_general_contact_request(model,resolution.definition,settings,load_case=load)
    result=solve_shell_pair(model,domain,settings,resolution,path,max_load_factor=1/3)
    assert result.status=='completed' and len(result.steps)==1
    assert result.info['contact_path']['path_predictor']=='control_projection_dev'
    contact=result.info['general_contact']
    assert contact['active_pressure_dofs']>0 and contact['minimum_gap']>=-5e-5
    target=model.mesh.dof_manager.get_node_dofs(9)[2]
    assert abs(result.displacements[target])>1e-10
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(verified=True,
        identity=record,status=result.status,contact=contact,target_uz=float(result.displacements[target]),
        executed_selector=result.info['contact_path']['path_predictor']),indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
