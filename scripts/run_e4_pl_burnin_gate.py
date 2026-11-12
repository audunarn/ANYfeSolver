"""Run and validate the frozen E4-PL Q1M burn-in gates.

The resource-heavy ``functional`` and ``performance`` lanes must be invoked
only after the repository resource manager has approved and acquired their
registered request.  This script deliberately does not mutate that manager.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import tomllib
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


# A direct gate invocation must not contaminate any frozen source tree even if
# its caller omitted the hash-bound child environment.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
BURNIN_AUTHORITY_CYCLE = 22
_GATE_SOURCE_ROOT = ROOT
TESTS = ROOT / "tests"
CONTRACT_PATH = ROOT / "docs" / "reference_cases" / "e4_pl_q1m_burnin_contract.json"
S3_Q4_CONTRACT_PATH = (
    ROOT / "docs" / "reference_cases" / "e4_pl_s3_q4_burnin_contract.json"
)
S3_Q4_ACTIVE_AUTHORITY_GENERATION = "v20"

FUNCTIONAL_WAVE_SCHEMA = "anysolver.e4-pl-s3-q4-functional-wave-v2"
FUNCTIONAL_WAVE_V19_SCHEMA = "anysolver.e4-pl-s3-q4-functional-wave-v3"
FUNCTIONAL_SHARD_SCHEMA = "anysolver.e4-pl-s3-q4-functional-shard-v2"
FUNCTIONAL_READY_SCHEMA = "anysolver.e4-pl-s3-q4-functional-ready-v1"
FUNCTIONAL_RELEASE_SCHEMA = "anysolver.e4-pl-s3-q4-functional-release-v1"
FUNCTIONAL_WAVE_AGGREGATE_SCHEMA = (
    "anysolver.e4-pl-s3-q4-functional-wave-aggregate-v2"
)
FUNCTIONAL_WAVE_V19_AGGREGATE_SCHEMA = (
    "anysolver.e4-pl-s3-q4-functional-wave-aggregate-v3"
)
FUNCTIONAL_WAVE_DIAGNOSTICS_SCHEMA = (
    "anysolver.e4-pl-s3-q4-functional-wave-diagnostics-v2"
)
FUNCTIONAL_WAVE_V19_DIAGNOSTICS_SCHEMA = (
    "anysolver.e4-pl-s3-q4-functional-wave-diagnostics-v3"
)
FUNCTIONAL_WAVE_ROUTING = {
    "aggregate_filename": "functional-wave-aggregate.json",
    "archive_filename": "functional-wave-source.tar",
    "directory_name": "functional-wave",
    "raw_diagnostics_filename": "functional-wave-raw-diagnostics.json",
    "shard_directory_prefix": "shard-",
    "source_directory_name": "source",
}
FUNCTIONAL_SHARD_DIRECTORIES = (
    "cwd",
    "basetemp",
    "python_cache",
    "numba_cache",
    "temp",
    "reports",
    "logs",
)
FUNCTIONAL_COLLECTION_ARTIFACT = {
    "bytes": 116046,
    "sha256": "244132e6294f3dc37f5bf865bd800c7402d9ff6b3300af670ca954ad24bb5c15",
}
FUNCTIONAL_V19_COLLECTION_ARTIFACT = {
    "bytes": 119327,
    "sha256": "53a5cc7a39489f4eb0d33db4abab3ae3cb0798efe110345129a2d80d49d52e96",
}
FUNCTIONAL_V19_FULL_NODE_AUTHORITY = {
    "node_count": 1065,
    "node_ids_sha256": "53a5cc7a39489f4eb0d33db4abab3ae3cb0798efe110345129a2d80d49d52e96",
}
FUNCTIONAL_SHARD_AUTHORITIES = {
    "P01": {
        "node_count": 1,
        "node_ids_sha256": "9a2ff93c333465e9815679ef6e9f971cbee782f2b833934c765447867a5a5704",
    },
    "P02": {
        "node_count": 362,
        "node_ids_sha256": "c6d005d0e69b2be5bbe1b197c71f22af598081ec2a1fc9e31cf217f7b5ef37ce",
    },
    "P03": {
        "node_count": 361,
        "node_ids_sha256": "c7dad7210a3bebcdd7a497db946b574e05c5d2a1694fa9426ee7b62da271b8b3",
    },
    "P04": {
        "node_count": 312,
        "node_ids_sha256": "38cb6d54b4c0e163308cdbcde38eff5a8372cc5a3c3702b1c0edbdf173b2118a",
    },
}
FUNCTIONAL_V19_SHARD_IDS = (
    "H01",
    "V01",
    "G01",
    "B01",
    "B02",
    "B03",
    "D01",
)
FUNCTIONAL_V19_SHARD_AUTHORITIES = {
    "H01": {
        "node_count": 10,
        "node_ids_sha256": "3fef0dace7e7d5ace263ab29ae016e3d18864f27ac92b258b216efed489c05a5",
    },
    "V01": {
        "node_count": 8,
        "node_ids_sha256": "bb78a167e46dba569a001002e6b37aacc9e737fb73ff04d45f9d3784c7cf6a8f",
    },
    "G01": {
        "node_count": 6,
        "node_ids_sha256": "cb9326084e3d7cb531fbc4cc7bf123e045c28cd6b455d508e1afaf1a53604221",
    },
    "B01": {
        "node_count": 262,
        "node_ids_sha256": "7330ad1a643c0811f026c3e632cc3265daf7768271b9ea0b7c05f31ca6968c67",
    },
    "B02": {
        "node_count": 292,
        "node_ids_sha256": "49784c9327ca270986103fa4ed3ebe8d776d5e54f56729d9da6c0e92bcb2a6ad",
    },
    "B03": {
        "node_count": 238,
        "node_ids_sha256": "ae1330121594e6b9d5993b389256e2c2eb8c9d6e645764e3aeb7a1792ec7a3e4",
    },
    "D01": {
        "node_count": 249,
        "node_ids_sha256": "f87ca0b65519aedb3427c75dc019a8a06166da8f4e71d9cf0ba0a4a6ec703a17",
    },
}
FUNCTIONAL_V19_SHARD_MODULE_AUTHORITIES = {
    "H01": {
        "module_count": 1,
        "modules_sha256": "98849593ac93845b998d2dfdb64be01d31e2752ce61b562038a4f66807e24c22",
    },
    "V01": {
        "module_count": 1,
        "modules_sha256": "5f0064fb99008360515a758066fc35b60c2c03225460e26d7bafe9e4c8cb14ad",
    },
    "G01": {
        "module_count": 2,
        "modules_sha256": "41551e23750c140ac0bfb0cef159bc8b739e47cc4a5a0657ecc6ee72fee07ba5",
    },
    "B01": {
        "module_count": 22,
        "modules_sha256": "a67f8bfad4c6ab582d1b51ac240be5362497b48bccd852a6377bfcc1973e29bb",
    },
    "B02": {
        "module_count": 21,
        "modules_sha256": "eaa9e5ea6f6383cdb28b107d0339edba2a1c25e68e0fffcd5dc54063384d3937",
    },
    "B03": {
        "module_count": 19,
        "modules_sha256": "e9c3bbcfec8bb681ea60c4bb7e60ab08de3b59e826a8da12543bb27ab35a4594",
    },
    "D01": {
        "module_count": 21,
        "modules_sha256": "19102d408e3d38fd13198a68636847c8ea304d9c60f7853967e28a15016e3a1f",
    },
}
FUNCTIONAL_V19_FULL_MODULES = {
    "H01": ("tests/test_fe_solver_nonlinear_static.py",),
    "V01": ("tests/test_beam_shell_verification.py",),
    "G01": ("tests/test_geometry_panel.py",),
    "B01": (
        "tests/test_anyfileio_version_compatibility.py",
        "tests/test_constraint_audit.py",
        "tests/test_corotational.py",
        "tests/test_e4_pl_orchestrator_operation_lease.py",
        "tests/test_e4_pl_q4_stationary_conditioning.py",
        "tests/test_external_references.py",
        "tests/test_fe_solver_buckling.py",
        "tests/test_fe_solver_corotational_beam.py",
        "tests/test_fe_solver_cylinder_benchmarks.py",
        "tests/test_fe_solver_infrastructure.py",
        "tests/test_fe_solver_nonlinear_dnv.py",
        "tests/test_fe_solver_nonlinear_limit_point.py",
        "tests/test_fe_solver_shell_verification.py",
        "tests/test_fe_solver_triangular_shell_backend.py",
        "tests/test_generalized_section_mass_properties.py",
        "tests/test_material_history_recovery.py",
        "tests/test_nodal_stress_recovery.py",
        "tests/test_nonlinear_analysis_diagnostics.py",
        "tests/test_nonlinear_impact_plasticity.py",
        "tests/test_orthotropic_workflow_integration.py",
        "tests/test_qualified_assembly_exception_precedence.py",
    ),
    "B02": (
        "tests/test_beam_axis_conventions.py",
        "tests/test_e4_pl_current_state_input_ownership.py",
        "tests/test_e4_pl_mixed_current_state_route.py",
        "tests/test_element_activity_integration.py",
        "tests/test_fe_solver_contact.py",
        "tests/test_fe_solver_fracture.py",
        "tests/test_fe_solver_recovery_policy.py",
        "tests/test_fe_solver_reference_cases.py",
        "tests/test_fe_solver_theory.py",
        "tests/test_follower_pressure.py",
        "tests/test_hht_alpha.py",
        "tests/test_hill48_plane_stress_plasticity.py",
        "tests/test_mesh_load_bc_verification.py",
        "tests/test_namespace_neutral_deepcopy.py",
        "tests/test_nonlinear_restart_checkpoint.py",
        "tests/test_orthotropic_material_contract.py",
        "tests/test_production_readiness.py",
        "tests/test_qualified_q4_assembly_authority.py",
        "tests/test_runtime_api.py",
        "tests/test_s4_director_field.py",
        "tests/test_sesam_fem.py",
    ),
    "B03": (
        "tests/test_arc_length.py",
        "tests/test_e4_pl_component_snapshot_integrity.py",
        "tests/test_e4_pl_guarded_observations.py",
        "tests/test_e4_pl_q4_current_tangent.py",
        "tests/test_e4_pl_transient_authority.py",
        "tests/test_fe_solver_anystructure_fem_mode.py",
        "tests/test_fe_solver_architecture.py",
        "tests/test_fe_solver_mass_modal.py",
        "tests/test_fe_solver_triangular_shell.py",
        "tests/test_generalized_section_api_workflows.py",
        "tests/test_generalized_shell_sections.py",
        "tests/test_mesher_improvements.py",
        "tests/test_mixed_shell_quadrature_grouping.py",
        "tests/test_nonlinear_static_state_lifecycle.py",
        "tests/test_phase2.py",
        "tests/test_quadratic_beam_nonlinear.py",
        "tests/test_qualified_mutation_epoch.py",
        "tests/test_transient_reduced_output.py",
    ),
    "D01": (
        "tests/test_anymesher_version_compatibility.py",
        "tests/test_e4_pl_planar_physical_recovery.py",
        "tests/test_e4_pl_q4_state_lifecycle.py",
        "tests/test_element_activity.py",
        "tests/test_extracted_package_wiring.py",
        "tests/test_fe_solver_capacity_workflow.py",
        "tests/test_fe_solver_dynamics.py",
        "tests/test_fe_solver_element_qualification.py",
        "tests/test_fe_solver_element_quality.py",
        "tests/test_fe_solver_plasticity_qualification.py",
        "tests/test_fe_solver_s4_validity.py",
        "tests/test_generalized_beam_sections.py",
        "tests/test_generalized_state_recovery.py",
        "tests/test_initial_fields_and_staged_history.py",
        "tests/test_native_rotation_state.py",
        "tests/test_nonlinear_state_lifecycle.py",
        "tests/test_orthotropic_elements.py",
        "tests/test_plane_stress_analytical_tangent.py",
        "tests/test_production_validation.py",
        "tests/test_recovery_qualification.py",
        "tests/test_solver_control_contracts.py",
    ),
}
FUNCTIONAL_V19_SPLIT_MODULE = "tests/test_local_patch_transition.py"
FUNCTIONAL_V19_SPLIT_NODE_IDS = {
    "H01": (),
    "V01": (),
    "G01": (
        "tests/test_local_patch_transition.py::"
        "test_axial_cylinder_stress_is_mesh_style_invariant",
    ),
    "B01": (
        "tests/test_local_patch_transition.py::"
        "test_axial_flat_plate_stress_is_mesh_style_invariant",
    ),
    "B02": (),
    "B03": (
        "tests/test_local_patch_transition.py::test_flat_local_patch_is_conforming_and_local",
        "tests/test_local_patch_transition.py::test_cylinder_local_patch_quality_beats_graded",
        "tests/test_local_patch_transition.py::test_cylinder_local_patch_window_on_seam_is_conforming",
        "tests/test_local_patch_transition.py::test_cylinder_local_patch_keeps_normal_winding",
        "tests/test_local_patch_transition.py::test_local_patch_falls_back_for_shell_web_members",
        "tests/test_local_patch_transition.py::test_local_patch_model_solves_with_collision",
        "tests/test_local_patch_transition.py::test_point_refinement_uses_local_patch_when_selected",
    ),
    "D01": (),
}
FUNCTIONAL_V20_DEFERRED_ACTIVATION_NODES = (
    "tests/test_local_patch_transition.py::"
    "test_local_patch_buckling_has_no_spurious_flat_facet_modes",
    "tests/test_local_patch_transition.py::"
    "test_stiffened_cylinder_keeps_mesh_style_invariance_with_beams",
)
FUNCTIONAL_V20_DEFERRED_ACTIVATION_AUTHORITY = {
    "disposition": "MANDATORY_DEFAULT_ACTIVATION_GATE_NOT_OPT_IN_BURN_IN",
    "node_count": 2,
    "node_ids_sha256": "a21a8a44724a3d1601dd54ff8c7a8fd0c97ac4b8572f1731db93cf8767ce277d",
    "reason": "GENERATED_TRANSITION_TRIANGLES_RESOLVE_TO_LEGACY_S3_BEFORE_DEFAULT_ACTIVATION",
}
FUNCTIONAL_V20_INNER_WAVE_POLICY = {
    "cleanup_cutoff_seconds": 385,
    "cleanup_deadline_environment": (
        "ANYSOLVER_BURNIN_S3_Q4_STYLE_CLEANUP_DEADLINE"
    ),
    "default_mode": "SEQUENTIAL",
    "disposition": "DEFERRED_ACTIVATION_DIAGNOSTIC_ONLY",
    "enabled": False,
    "enabled_shard_id": "H01",
    "max_concurrent_workers": 3,
    "opt_in_environment": "ANYSOLVER_BURNIN_S3_Q4_STYLE_WAVE",
    "opt_in_value": "1",
    "result_cutoff_seconds": 370,
    "result_deadline_environment": (
        "ANYSOLVER_BURNIN_S3_Q4_STYLE_RESULT_DEADLINE"
    ),
    "worker_wall_limit_seconds": 360,
}
FUNCTIONAL_V19_STAGE_POLICY = {
    "command_watchdog_cutoff_seconds": 510,
    "emergency_publication_cutoff_seconds": 480,
    "execution_mode": "concurrent_seven_shards_no_inner_style_wave",
    "inner_wave_policy": FUNCTIONAL_V20_INNER_WAVE_POLICY,
    "max_concurrent_workers": 7,
    "normal_evidence_cutoff_seconds": 460,
    "preparation_cutoff_seconds": 50,
    "ready_barrier_cutoff_seconds": 90,
    "shard_execution_cutoff_seconds": 430,
    "shard_execution_ids": ["H01", "V01", "G01", "B01", "B02", "B03", "D01"],
    "watchdog_termination_start_cutoff_seconds": 500,
}
FUNCTIONAL_V19_TREE_CLEANUP_STATES = {
    "NOT_LAUNCHED",
    "CHILD_EXIT_AND_TREE_CLOSED",
    "TREE_TERMINATION_PROVEN",
    "TREE_TERMINATION_FAILED",
}
FUNCTIONAL_V19_LAUNCH_STATES = {
    "ALL_LAUNCHED",
    "NOT_LAUNCHED",
    "PARTIALLY_LAUNCHED",
}
FUNCTIONAL_V19_TERMINAL_STATES = {
    "ALL_TERMINAL",
    "NOT_LAUNCHED",
    "UNPROVEN_DESCENDANT_TREE",
}
_FUNCTIONAL_RESULT_ENV = "ANYSOLVER_FUNCTIONAL_SHARD_RESULT"
_FUNCTIONAL_EXPECTED_ENV = "ANYSOLVER_FUNCTIONAL_EXPECTED_NODES"
_FUNCTIONAL_PROGRESS_ENV = "ANYSOLVER_FUNCTIONAL_PROGRESS"
_FUNCTIONAL_SHARD_ENV = "ANYSOLVER_FUNCTIONAL_SHARD_ID"
_FUNCTIONAL_READY_ENV = "ANYSOLVER_FUNCTIONAL_READY"
_FUNCTIONAL_RELEASE_ENV = "ANYSOLVER_FUNCTIONAL_RELEASE"
_FUNCTIONAL_BARRIER_ID_ENV = "ANYSOLVER_FUNCTIONAL_BARRIER_ID"
_FUNCTIONAL_READY_NONCE_ENV = "ANYSOLVER_FUNCTIONAL_READY_NONCE"
_FUNCTIONAL_BARRIER_DEADLINE_ENV = "ANYSOLVER_FUNCTIONAL_BARRIER_DEADLINE"
_S3_Q4_STYLE_WAVE_ENV = "ANYSOLVER_BURNIN_S3_Q4_STYLE_WAVE"
_S3_Q4_STYLE_RESULT_DEADLINE_ENV = (
    "ANYSOLVER_BURNIN_S3_Q4_STYLE_RESULT_DEADLINE"
)
_S3_Q4_STYLE_CLEANUP_DEADLINE_ENV = (
    "ANYSOLVER_BURNIN_S3_Q4_STYLE_CLEANUP_DEADLINE"
)
_S3_Q4_STYLE_WAVE_ENVIRONMENT = (
    _S3_Q4_STYLE_WAVE_ENV,
    _S3_Q4_STYLE_RESULT_DEADLINE_ENV,
    _S3_Q4_STYLE_CLEANUP_DEADLINE_ENV,
)
_CI_CAPTURE_ENV = "ANYSOLVER_CI_CAPTURE_NODES"
_CI_EXPECTED_ENV = "ANYSOLVER_CI_EXPECTED_NODES"
_CI_SHARD_ENV = "ANYSOLVER_CI_SHARD_ID"
_ACTIVE_TEST_LANE_ENV = "ANYSOLVER_BURNIN_ACTIVE_TEST_LANE"
_GATE_WATCHDOG_ENV = "ANYSOLVER_BURNIN_GATE_WATCHDOG_CHILD"
_FUNCTIONAL_UNPROVEN_TREE = threading.Event()
_FUNCTIONAL_SHARD_TREE_STATE: dict[str, str] = {}
_FUNCTIONAL_SHARD_TREE_STATE_LOCK = threading.Lock()

# Bounded CI is an independent four-worker partition.  It must never inherit
# functional-wave shard IDs: the functional wave has seven timing-balanced
# workers, while CI owns four separately frozen P01--P04 node lists.
CI_SHARD_IDS = ("P01", "P02", "P03", "P04")
CI_SPLIT_MODULE = "tests/test_local_patch_transition.py"
CI_FUNCTIONAL_MODULES = {
    "P01": (
        "tests/test_beam_shell_verification.py",
        "tests/test_e4_pl_guarded_observations.py",
        "tests/test_e4_pl_planar_physical_recovery.py",
        "tests/test_e4_pl_q4_stationary_conditioning.py",
        "tests/test_element_activity.py",
        "tests/test_element_activity_integration.py",
        "tests/test_external_references.py",
        "tests/test_extracted_package_wiring.py",
        "tests/test_fe_solver_capacity_workflow.py",
        "tests/test_fe_solver_element_quality.py",
        "tests/test_fe_solver_triangular_shell_backend.py",
        "tests/test_generalized_section_mass_properties.py",
        "tests/test_native_rotation_state.py",
        "tests/test_nonlinear_analysis_diagnostics.py",
        "tests/test_phase2.py",
        "tests/test_s4_director_field.py",
    ),
    "P02": (
        "tests/test_anyfileio_version_compatibility.py",
        "tests/test_constraint_audit.py",
        "tests/test_e4_pl_component_snapshot_integrity.py",
        "tests/test_fe_solver_contact.py",
        "tests/test_fe_solver_cylinder_benchmarks.py",
        "tests/test_fe_solver_dynamics.py",
        "tests/test_fe_solver_element_qualification.py",
        "tests/test_fe_solver_fracture.py",
        "tests/test_fe_solver_mass_modal.py",
        "tests/test_fe_solver_nonlinear_limit_point.py",
        "tests/test_fe_solver_recovery_policy.py",
        "tests/test_fe_solver_s4_validity.py",
        "tests/test_fe_solver_triangular_shell.py",
        "tests/test_follower_pressure.py",
        "tests/test_generalized_state_recovery.py",
        "tests/test_geometry_panel.py",
        "tests/test_mixed_shell_quadrature_grouping.py",
        "tests/test_namespace_neutral_deepcopy.py",
        "tests/test_nonlinear_restart_checkpoint.py",
        "tests/test_qualified_q4_assembly_authority.py",
        "tests/test_runtime_api.py",
        "tests/test_solver_control_contracts.py",
    ),
    "P03": (
        "tests/test_anymesher_version_compatibility.py",
        "tests/test_arc_length.py",
        "tests/test_corotational.py",
        "tests/test_e4_pl_current_state_input_ownership.py",
        "tests/test_e4_pl_mixed_current_state_route.py",
        "tests/test_e4_pl_q4_state_lifecycle.py",
        "tests/test_e4_pl_transient_authority.py",
        "tests/test_fe_solver_anystructure_fem_mode.py",
        "tests/test_fe_solver_infrastructure.py",
        "tests/test_fe_solver_nonlinear_dnv.py",
        "tests/test_fe_solver_plasticity_qualification.py",
        "tests/test_fe_solver_shell_verification.py",
        "tests/test_hill48_plane_stress_plasticity.py",
        "tests/test_initial_fields_and_staged_history.py",
        "tests/test_nodal_stress_recovery.py",
        "tests/test_nonlinear_state_lifecycle.py",
        "tests/test_nonlinear_static_state_lifecycle.py",
        "tests/test_orthotropic_elements.py",
        "tests/test_plane_stress_analytical_tangent.py",
        "tests/test_production_readiness.py",
        "tests/test_qualified_assembly_exception_precedence.py",
        "tests/test_recovery_qualification.py",
        "tests/test_sesam_fem.py",
        "tests/test_transient_reduced_output.py",
    ),
    "P04": (
        "tests/test_beam_axis_conventions.py",
        "tests/test_e4_pl_orchestrator_operation_lease.py",
        "tests/test_e4_pl_q4_current_tangent.py",
        "tests/test_fe_solver_architecture.py",
        "tests/test_fe_solver_buckling.py",
        "tests/test_fe_solver_corotational_beam.py",
        "tests/test_fe_solver_nonlinear_static.py",
        "tests/test_fe_solver_reference_cases.py",
        "tests/test_fe_solver_theory.py",
        "tests/test_generalized_beam_sections.py",
        "tests/test_generalized_section_api_workflows.py",
        "tests/test_generalized_shell_sections.py",
        "tests/test_hht_alpha.py",
        "tests/test_material_history_recovery.py",
        "tests/test_mesh_load_bc_verification.py",
        "tests/test_mesher_improvements.py",
        "tests/test_nonlinear_impact_plasticity.py",
        "tests/test_orthotropic_material_contract.py",
        "tests/test_orthotropic_workflow_integration.py",
        "tests/test_production_validation.py",
        "tests/test_quadratic_beam_nonlinear.py",
        "tests/test_qualified_mutation_epoch.py",
    ),
}
CI_FUNCTIONAL_MODULE_AUTHORITIES = {
    "P01": {"module_count": 16, "modules_sha256": "15a8990c153655d9203bd64a7d6096e6cc1f2d912d499501127912a07b62730e"},
    "P02": {"module_count": 22, "modules_sha256": "97c29f84a355b5794c5c0599089a095e1a399b03bbe74a80a41fbc72d6d4b848"},
    "P03": {"module_count": 24, "modules_sha256": "79618a1e82536d4ee69b818b1ff941c4736a349f9825692d03942aa225ce709f"},
    "P04": {"module_count": 22, "modules_sha256": "c22f089c1b0a89b32b104805864aaac7f0f3bb6b2662557967e4ac2787eab983"},
}
CI_FUNCTIONAL_EXACT_NODES = {
    "P01": (),
    "P02": (
        "tests/test_local_patch_transition.py::"
        "test_axial_flat_plate_stress_is_mesh_style_invariant",
    ),
    "P03": (),
    "P04": (
        "tests/test_local_patch_transition.py::test_flat_local_patch_is_conforming_and_local",
        "tests/test_local_patch_transition.py::test_cylinder_local_patch_quality_beats_graded",
        "tests/test_local_patch_transition.py::test_cylinder_local_patch_window_on_seam_is_conforming",
        "tests/test_local_patch_transition.py::test_cylinder_local_patch_keeps_normal_winding",
        "tests/test_local_patch_transition.py::test_local_patch_falls_back_for_shell_web_members",
        "tests/test_local_patch_transition.py::test_local_patch_model_solves_with_collision",
        "tests/test_local_patch_transition.py::test_point_refinement_uses_local_patch_when_selected",
        "tests/test_local_patch_transition.py::test_axial_cylinder_stress_is_mesh_style_invariant",
    ),
}
CI_FUNCTIONAL_EXACT_NODE_AUTHORITIES = {
    "P01": {"node_count": 0, "node_ids_sha256": "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570"},
    "P02": {"node_count": 1, "node_ids_sha256": "8238c1b9ac48397eac227af41cb3639cd4c3419a94df636642d857a72f301377"},
    "P03": {"node_count": 0, "node_ids_sha256": "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570"},
    "P04": {"node_count": 8, "node_ids_sha256": "4218b7c97abc9903a6b86dc899b8ce51b8a32050473454f5f70f6d64ae77d43b"},
}
CI_FUNCTIONAL_NODE_AUTHORITIES = {
    "P01": {"node_count": 180, "node_ids_sha256": "dd40cfcd3a4669d9124eecb590231863ffa3b36fc21f8dade293522a0f53ba81"},
    "P02": {"node_count": 236, "node_ids_sha256": "8edc0691fa6af896622130673d73d4bdc272f1adcdde3eae337cebacfeaba36e"},
    "P03": {"node_count": 364, "node_ids_sha256": "c9afdeece65a339e87a47d5a1558c4b1aa8009949b27cc7da244b4349d62d88b"},
    "P04": {"node_count": 285, "node_ids_sha256": "b157590645ef8417b6b833cd606b7ece2991b5a06026a8ccea8020036665b929"},
}
CI_NONFUNCTIONAL_NODE_AUTHORITY = {
    "node_count": 910,
    "node_ids_sha256": "4fb785042836255467776a617940aff34c82fbcbc7f0a40437c1eb43ae786bab",
}
CI_NONFUNCTIONAL_MODULE_AUTHORITIES = {
    "P01": {"module_count": 7, "modules_sha256": "6798a1e7bf8d796c210b60cb2600ed0a161545ea9f239a3720975ae591800304"},
    "P02": {"module_count": 15, "modules_sha256": "40400dcb491f55c9efe745b9c484234664c840300ec5329461b9e1e6fc8100be"},
    "P03": {"module_count": 14, "modules_sha256": "270b2ca721b706b228ed137413caf198ca46dd8e6430095a481376798ff376de"},
    "P04": {"module_count": 14, "modules_sha256": "5d8e4143c2d635fe7a1c739cd5af651d6d2724342a10ac88930f464b0f68138b"},
}
CI_NONFUNCTIONAL_NODE_AUTHORITIES = {
    "P01": {"node_count": 92, "node_ids_sha256": "f97fa883bd2ba01e13a18d7ae413ac2fe341ec26b7722f6748c170239ed9d502"},
    "P02": {"node_count": 260, "node_ids_sha256": "742afa90066e70cd3706551a449b59f0fe5bd65e9fb453e9a56797477deac31f"},
    "P03": {"node_count": 248, "node_ids_sha256": "05a654eba6b1b8a515ca8066c9f637ab8d1f59f5f57327d9f8aad0daa0f4f744"},
    "P04": {"node_count": 310, "node_ids_sha256": "fec37150c16a184d022d2d98bac9ffdfa63f59ecaf9f50b8fba226f2c3b0350f"},
}
CI_FULL_NODE_AUTHORITY = {
    "node_count": 1975,
    "node_ids_sha256": "1b44d5b7472a037a549699ac2aa305dd83e8da393553986cca173abb9ac232da",
}
CI_SHARD_NODE_AUTHORITIES = {
    "P01": {
        "node_count": 272,
        "node_ids_sha256": "038e2eb049cae0bfb874dfb6494e977d3691860b70b446d3ccc6d30fe238aabe",
    },
    "P02": {
        "node_count": 496,
        "node_ids_sha256": "80159dd214c1861c1235f128dff0f792c4518041e2a0f52ab191ef6daf5c86cd",
    },
    "P03": {
        "node_count": 612,
        "node_ids_sha256": "8286fc6302f0a281d46e8701fa71d03405be68dfe66f5303d6db72d089d5b69c",
    },
    "P04": {
        "node_count": 595,
        "node_ids_sha256": "91dfd702f478e53c202694f88b8461e069e99803dc6d853b78832255de06d795",
    },
}

# The parent watchdog includes import/argument handling, inventory discovery,
# preparation, process launch/wait, tree termination, and cleanup.  Functional
# The current staged functional command is bounded at 510 seconds and returns
# before the cycle coordinator's 650-second cumulative functional boundary.
# The longer constants remain solely for historical v18 and CI compatibility.
CI_COMMAND_WALL_LIMIT_SECONDS = 1200
CI_COMMAND_TERMINATION_RESERVE_SECONDS = 30
FUNCTIONAL_COMMAND_WALL_LIMIT_SECONDS = 840
FUNCTIONAL_V19_COMMAND_WALL_LIMIT_SECONDS = 510
FUNCTIONAL_V19_WATCHDOG_TERMINATION_START_SECONDS = 500
FUNCTIONAL_COMMAND_TERMINATION_RESERVE_SECONDS = 10
_GATE_COMMAND_ENTRY = time.monotonic()
_GATE_WATCHDOG_WINDOWS_TERMINATION = {
    "arguments": ["/PID", "{pid}", "/T", "/F"],
    "bytes": 118784,
    "path": "C:\\Windows\\System32\\taskkill.exe",
    "sha256": "1249717315fc8f4d2df17d5db9da0444795fdb9fb83dfb1f763c3f39282244f7",
}

QUICK = {
    "test_e4_pl_burnin.py",
    "test_e4_pl_default_activation.py",
    "test_e4_pl_dormant_element.py",
    "test_e4_pl_functional_constructor_policy.py",
    "test_e4_pl_parity_matrix.py",
    "test_e4_pl_warped_qualification.py",
    "test_e4_pl_workflow_parity.py",
}

PERFORMANCE_TOKENS = (
    "_batch",
    "performance",
    "sol_ultra",
    "threading_policy",
    "vectorized",
)

PERFORMANCE_EXACT = {
    "test_analysis_session.py",
    "test_contact_work_buffer.py",
    "test_experimental_csr_assembly.py",
    "test_impact_reduced_assembly.py",
    "test_impact_tangent_reuse.py",
    "test_recovery_batches.py",
}

EXTENDED_PREFIXES = (
    "test_e3_",
    "test_e4_pl_q1",
    "test_s4_candidate_",
    "test_s4_e2_",
    "test_s4_eq21_",
    "test_s4_geometry_handoff",
    "test_s4_improved_",
    "test_s4_nullspace_",
    "test_s4_restricted_",
    "test_s4_stage_m_",
)

EXTENDED_EXACT = {
    "test_e4_baseline.py",
    "test_e4_closeout.py",
    "test_e4_core_identity.py",
    "test_e4_pl_identity.py",
    "test_e4_route.py",
    "test_e4_ws_feasibility.py",
    "test_s4_drill_constraint_derivation.py",
}

# This preserved mixed-eigen study carries historical external evidence and is
# never a merge or serialized burn-in prerequisite.  Keep it out of the
# performance lane even though its filename contains ``performance``.
HISTORICAL_EXTENDED_EXACT = {
    "test_e4_pl_s3_mixed_eigen_performance.py",
    "test_e4_pl_s3_mixed_structural_qualification.py",
}

# Successor element-family tests must run in pull requests without changing
# the immutable Q1M gate inventories recorded before those files existed.
ADDITIVE_CI_PREFIXES = ("test_e4_pl_s3_",)
# This reference-batch file carries correctness and authority checks despite
# its performance-shaped name, so exercise this exact file in pull requests.
ADDITIVE_CI_EXACT = {"test_e4_pl_s3_reference_batch.py", "test_general_contact_enriched_moment_batch.py"}

PACKAGE_CHECKS = (
    "BUILD_LOCAL_WHEELS_WITHOUT_ISOLATION",
    "INSTALL_FRESH_TARGET_WITHOUT_DEPENDENCY_RESOLUTION",
    "REJECT_SOURCE_TREE_IMPORTS",
    "VERIFY_QUALIFIED_Q4_DEFAULT",
    "VERIFY_NON_Q4_PRESERVATION",
    "VERIFY_DIAGNOSTICS_EXPORT",
    "VERIFY_EXPLICIT_LEGACY_WARNING",
)

GATE_LANES = ("quick", "package", "functional", "anyfem", "performance")
SIBLING_NAMES = ("ANYfem", "ANYmesh", "ANYgeometry", "ANYmaterial", "ANYfileIO")
# The five blocked contracts are immutable authorities from earlier Q1M
# correction cycles.  Their inventories predate additive lifecycle tests, so
# validating them against today's discovery would rewrite their meaning.  Bind
# each semantic contract identity to the lane inventories frozen in those
# accepted records instead.
_HISTORICAL_CONTRACT_IDENTITIES = frozenset(
    {
        "bdf622aaa4c370e078d947fdd38b194279d2bc757862ef9b2ed6f1f09bce9efe",
        "b74c6a41d67dd123ba7d8c35bf68f6a25975b63fe2a5a8d93b5eef00dec8d661",
        "f8f03f3db1e163784e6d948b5832a9cb1f22a04d7d8d14c45b2166ace1db74f6",
        "3e9bc597585a9ed48fc149197a27eb31641521ae5958e79396dfbb46242a3f7a",
        "3059b8aabb0518b188a8b29bf5d21248ea439a32f88603bf70a2072b7b6a4293",
    }
)
_HISTORICAL_INVENTORY_IDENTITIES = {
    "anyfem": "f0665ef5fd92da79b8f24691617574a7380bb1e93d2e520e480039df80f7b0c1",
    "functional": "6d3c46bfeb207d66417cda61129d5ae104f6fee93b1cb084c068c3c88c041860",
    "package": "31c7624758906e5e7501e06a9d1d81b520924cd7ce87f291d8ce44613c82a7b3",
    "performance": "0f6b51790108adc744e77d1827c11ee46ff4609ea3d821588c1510984f9e67aa",
    "quick": "6798a1e7bf8d796c210b60cb2600ed0a161545ea9f239a3720975ae591800304",
}
LOCAL_DISTRIBUTIONS = (
    ("ANYgeometry", "ANYgeometry", "anygeometry"),
    ("ANYmaterial", "ANYmaterial", "anymaterial"),
    ("ANYmesh", "ANYmesher", "anymesher"),
    ("ANYfileIO", "ANYfileio", "anyfileio"),
    ("ANYfeSolver", "ANYfeSolver", "anyfesolver"),
)

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}\Z")
REQUEST_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
PROJECT_NAME_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
PROJECT_VERSION_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._+-]*[A-Za-z0-9])?\Z")
PERFORMANCE_OBSERVATION_SCHEMA = (
    "anysolver.s4.e4-pl-q1m-performance-observation-v1"
)
PERFORMANCE_BASELINE_MARKER = b"Q1M_PERFORMANCE_BASELINE_JSON="


class EvidenceError(ValueError):
    """Raised when a Q1M contract or gate record is not strict/canonical."""


class FunctionalDeadlineExpired(EvidenceError):
    """Raised when the frozen functional-wave internal deadline is exhausted."""


def classify_test(path: Path) -> str:
    """Return the single burn-in lane for a test file."""

    name = path.name
    if name in QUICK:
        return "quick"
    if name in ADDITIVE_CI_EXACT:
        return "additive"
    if name in HISTORICAL_EXTENDED_EXACT:
        return "extended"
    if name in PERFORMANCE_EXACT or any(token in name for token in PERFORMANCE_TOKENS):
        return "performance"
    if name in EXTENDED_EXACT or name.startswith(EXTENDED_PREFIXES):
        return "extended"
    if name.startswith(ADDITIVE_CI_PREFIXES):
        return "additive"
    return "functional"


def inventory() -> dict[str, list[str]]:
    result = {
        lane: []
        for lane in ("quick", "functional", "performance", "extended", "additive")
    }
    for path in sorted(TESTS.glob("test_*.py")):
        result[classify_test(path)].append(path.relative_to(ROOT).as_posix())
    return result


def gate_inventories() -> dict[str, list[str]]:
    """Return the exact evidence inventory for all five required gate lanes."""

    lanes = inventory()
    return {
        "quick": lanes["quick"],
        "package": list(PACKAGE_CHECKS),
        "functional": lanes["functional"],
        "anyfem": [
            "tests/test_e4_pl_default_routing.py",
            "tests/test_verification.py",
        ],
        "performance": lanes["performance"],
    }


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON number is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(raw: str | bytes) -> Any:
    """Decode JSON while rejecting duplicate keys and non-finite numbers."""

    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvidenceError("JSON must be UTF-8") from exc
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"invalid JSON: {exc.msg}") from exc
    _reject_nonfinite(value)
    return value


def strict_json_load(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise EvidenceError(f"JSON input must be a regular non-symlink file: {path}")
    return strict_json_loads(path.read_bytes())


def _reject_nonfinite(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError(f"non-finite value at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{location}[{index}]")


def canonical_json_bytes(value: Any) -> bytes:
    _reject_nonfinite(value)
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash_record(path: Path) -> dict[str, Any]:
    """Return the canonical byte count/hash for a regular external artifact."""

    if not path.is_file() or path.is_symlink() or _is_reparse_point(path):
        raise EvidenceError(f"artifact must be a regular non-reparse file: {path}")
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def wheel_hash_record(path: Path) -> dict[str, Any]:
    record = file_hash_record(path)
    if path.name != str(path.name) or not path.name.endswith(".whl"):
        raise EvidenceError("wheel artifact must have a .whl basename")
    if record["bytes"] <= 0:
        raise EvidenceError("wheel artifact must be nonempty")
    return {"bytes": record["bytes"], "filename": path.name, "sha256": record["sha256"]}


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _sanitized_git_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    result = dict(os.environ if environment is None else environment)
    removed = {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_EXEC_PATH",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_OPTIONAL_LOCKS",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    }
    for name in list(result):
        if name in removed or name.startswith("GIT_CONFIG_"):
            result.pop(name, None)
    result.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "NUL",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_GRAFT_FILE": "NUL",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    return result


def _git_executable() -> str:
    """Use the coordinator-bound absolute Git launcher during formal execution."""

    configured = os.environ.get("ANYSOLVER_FROZEN_GIT")
    if configured is not None:
        path = Path(configured)
        if (
            not path.is_absolute()
            or not path.is_file()
            or path.is_symlink()
            or _is_reparse_point(path)
        ):
            raise EvidenceError("formal Git launcher is not a regular absolute file")
        return str(path)
    discovered = shutil.which("git")
    if discovered is None:
        raise EvidenceError("Git launcher is unavailable")
    return str(Path(discovered).resolve(strict=True))


def _git_command_prefix(repository: Path) -> list[str]:
    """Return the single frozen Git command prefix used by the validator."""

    return [
        _git_executable(),
        "--no-replace-objects",
        "-c",
        f"safe.directory={repository}",
        "-c",
        "core.autocrlf=true",
        "-c",
        "core.attributesFile=NUL",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.quotepath=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "status.showUntrackedFiles=all",
        "-C",
        str(repository),
    ]


def _exact_keys(value: Any, expected: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{location} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EvidenceError(f"{location} keys mismatch; missing={missing}, extra={extra}")
    return value


def _require_bool(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise EvidenceError(f"{location} must be a boolean")
    return value


def _require_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EvidenceError(f"{location} must be an integer >= {minimum}")
    return value


def _require_match(value: Any, pattern: re.Pattern[str], location: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise EvidenceError(f"{location} has invalid format")
    return value


def _require_timestamp(value: Any, location: str) -> dt.datetime:
    raw = _require_match(value, UTC_RE, location)
    try:
        parsed = dt.datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as exc:
        raise EvidenceError(f"{location} is not a valid UTC timestamp") from exc
    return parsed


def _validate_hash_record(value: Any, location: str, *, allow_empty: bool = True) -> None:
    value = _exact_keys(value, {"bytes", "sha256"}, location)
    minimum = 0 if allow_empty else 1
    _require_int(value["bytes"], f"{location}.bytes", minimum=minimum)
    _require_match(value["sha256"], SHA256_RE, f"{location}.sha256")
    if value["bytes"] == 0 and value["sha256"] != hashlib.sha256(b"").hexdigest():
        raise EvidenceError(f"{location} has the wrong SHA-256 for empty content")


def _validate_process_record(
    value: Any, location: str, *, allow_empty: bool = True
) -> None:
    value = _exact_keys(value, {"bytes", "returncode", "sha256"}, location)
    _require_int(value["returncode"], f"{location}.returncode")
    if value["returncode"] != 0:
        raise EvidenceError(f"{location}.returncode must be zero")
    _validate_hash_record(
        {"bytes": value["bytes"], "sha256": value["sha256"]},
        location,
        allow_empty=allow_empty,
    )


def validate_package_result(
    value: Any, *, contract: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate the canonical installed-wheel package result."""

    contract = dict(_load_contract() if contract is None else contract)
    value = _exact_keys(
        value,
        {
            "build_logs",
            "install_log",
            "schema",
            "smoke",
            "smoke_log",
            "sources",
            "status",
            "wheels",
        },
        "$package",
    )
    if value["schema"] != contract["package_result_schema"]:
        raise EvidenceError("package result schema mismatch")
    if value["status"] != "PASS":
        raise EvidenceError("package result status must be PASS")
    names = {name for name, _distribution, _package in LOCAL_DISTRIBUTIONS}
    logs = _exact_keys(value["build_logs"], names, "$package.build_logs")
    for name in names:
        _validate_process_record(logs[name], f"$package.build_logs.{name}", allow_empty=False)
    _validate_process_record(value["install_log"], "$package.install_log", allow_empty=False)
    _validate_hash_record(value["smoke_log"], "$package.smoke_log", allow_empty=False)

    sources = _exact_keys(value["sources"], names, "$package.sources")
    for name in names:
        location = f"$package.sources.{name}"
        source = _exact_keys(
            sources[name],
            {"archive", "archive_log", "commit", "content", "tree"},
            location,
        )
        _require_match(source["commit"], GIT_OBJECT_RE, f"{location}.commit")
        _require_match(source["tree"], GIT_OBJECT_RE, f"{location}.tree")
        _validate_hash_record(source["archive"], f"{location}.archive", allow_empty=False)
        _validate_process_record(source["archive_log"], f"{location}.archive_log")
        content = _exact_keys(source["content"], {"files", "sha256"}, f"{location}.content")
        _require_int(content["files"], f"{location}.content.files", minimum=1)
        _require_match(content["sha256"], SHA256_RE, f"{location}.content.sha256")

    wheels = _exact_keys(value["wheels"], names, "$package.wheels")
    for name in names:
        wheel = _exact_keys(
            wheels[name], {"bytes", "filename", "sha256"}, f"$package.wheels.{name}"
        )
        if (
            not isinstance(wheel["filename"], str)
            or Path(wheel["filename"]).name != wheel["filename"]
            or not wheel["filename"].endswith(".whl")
        ):
            raise EvidenceError(f"$package.wheels.{name}.filename is invalid")
        _require_int(wheel["bytes"], f"$package.wheels.{name}.bytes", minimum=1)
        _require_match(wheel["sha256"], SHA256_RE, f"$package.wheels.{name}.sha256")

    smoke = _exact_keys(
        value["smoke"],
        {"diagnostics_schema", "legacy_warning", "non_q4_types", "origins", "q4_type"},
        "$package.smoke",
    )
    if smoke["diagnostics_schema"] != contract["diagnostics_schema"]:
        raise EvidenceError("package diagnostics schema mismatch")
    if smoke["legacy_warning"] != contract["legacy_q4"]["warning"]:
        raise EvidenceError("package legacy warning mismatch")
    if smoke["q4_type"] != "QualifiedE4PLShellElement":
        raise EvidenceError("package Q4 type mismatch")
    if smoke["non_q4_types"] != [
        "ShellElement",
        "ShellElement",
        "ShellElement",
        "ShellElement",
    ]:
        raise EvidenceError("package non-Q4 type preservation mismatch")
    expected_packages = {package for _name, _distribution, package in LOCAL_DISTRIBUTIONS}
    origins = _exact_keys(smoke["origins"], expected_packages, "$package.smoke.origins")
    for package, origin in origins.items():
        if (
            not isinstance(origin, str)
            or PurePosixPath(origin).is_absolute()
            or ".." in PurePosixPath(origin).parts
            or PurePosixPath(origin).parts[0] != package
        ):
            raise EvidenceError(f"package origin is invalid for {package}")
    return value


def _validate_timing_summary(
    value: Any,
    location: str,
    *,
    repetitions: int,
) -> None:
    value = _exact_keys(
        value,
        {"mad_ns", "median_ns", "p95_ns", "samples_ns"},
        location,
    )
    samples = value["samples_ns"]
    if not isinstance(samples, list) or len(samples) != repetitions:
        raise EvidenceError(f"{location}.samples_ns must contain {repetitions} samples")
    for index, sample in enumerate(samples):
        _require_int(sample, f"{location}.samples_ns[{index}]")
    ordered = sorted(samples)
    median_ns = int(statistics.median(ordered))
    mad_ns = int(statistics.median(abs(sample - median_ns) for sample in ordered))
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    expected = {
        "mad_ns": mad_ns,
        "median_ns": median_ns,
        "p95_ns": int(ordered[p95_index]),
    }
    for key, expected_value in expected.items():
        if _require_int(value[key], f"{location}.{key}") != expected_value:
            raise EvidenceError(f"{location}.{key} does not match its timing samples")


def _validate_performance_evidence(
    hard_gates: Any,
    baseline: Any,
    *,
    contract: Mapping[str, Any],
    lanes: Mapping[str, Any] | None = None,
) -> bool:
    expected_gates = contract["hard_performance_gates"]
    hard_gates = _exact_keys(hard_gates, set(expected_gates), "$.hard_gates")
    all_pass = True
    inventories = gate_inventories()
    for name in sorted(expected_gates):
        location = f"$.hard_gates.{name}"
        gate = _exact_keys(
            hard_gates[name],
            {"evidence_nodes", "observed", "status"},
            location,
        )
        if gate["evidence_nodes"] != expected_gates[name]:
            raise EvidenceError(f"{location}.evidence_nodes does not match the contract")
        observed = _require_bool(gate["observed"], f"{location}.observed")
        expected_status = "PASS" if observed else "FAIL"
        if gate["status"] != expected_status:
            raise EvidenceError(f"{location}.status contradicts observed")
        all_pass = all_pass and observed
        for node in gate["evidence_nodes"]:
            if not isinstance(node, str) or "::" not in node:
                raise EvidenceError(f"{location}.evidence_nodes contains an invalid node")
            test_path = node.split("::", 1)[0]
            matching_lanes = [
                lane for lane, paths in inventories.items() if test_path in paths
            ]
            if len(matching_lanes) != 1:
                raise EvidenceError(f"{location} evidence node has no unique gate lane")
            if lanes is not None and lanes[matching_lanes[0]]["status"] != "PASS":
                raise EvidenceError(f"{location} cannot pass when its evidence lane failed")

    expected_baseline = contract["performance_baseline"]
    baseline = _exact_keys(
        baseline,
        {"measurements", "repetitions", "schema", "speed_claim", "warmups"},
        "$.performance_baseline",
    )
    for key in ("repetitions", "schema", "speed_claim", "warmups"):
        if baseline[key] != expected_baseline[key]:
            raise EvidenceError(f"$.performance_baseline.{key} does not match the contract")
    repetitions = _require_int(
        baseline["repetitions"], "$.performance_baseline.repetitions", minimum=1
    )
    _require_int(baseline["warmups"], "$.performance_baseline.warmups", minimum=1)
    measurements = _exact_keys(
        baseline["measurements"],
        set(expected_baseline["measurement_names"]),
        "$.performance_baseline.measurements",
    )
    for name in expected_baseline["measurement_names"]:
        _validate_timing_summary(
            measurements[name],
            f"$.performance_baseline.measurements.{name}",
            repetitions=repetitions,
        )
    return all_pass


def extract_performance_observation(path: Path) -> dict[str, Any]:
    """Extract the single strict Q1M observation from a performance log."""

    if not path.is_file() or path.is_symlink():
        raise EvidenceError(f"performance log must be a regular non-symlink file: {path}")
    payloads = [
        line[len(PERFORMANCE_BASELINE_MARKER) :]
        for line in path.read_bytes().splitlines()
        if line.startswith(PERFORMANCE_BASELINE_MARKER)
    ]
    if len(payloads) != 1:
        raise EvidenceError("performance log must contain exactly one Q1M baseline marker")
    observation = _exact_keys(
        strict_json_loads(payloads[0]),
        {"hard_gates", "performance_baseline", "schema"},
        "$performance_observation",
    )
    if observation["schema"] != PERFORMANCE_OBSERVATION_SCHEMA:
        raise EvidenceError("performance observation schema mismatch")
    return observation


def _load_contract() -> dict[str, Any]:
    contract = strict_json_load(CONTRACT_PATH)
    if not isinstance(contract, dict):
        raise EvidenceError("Q1M contract must be an object")
    return contract


def validate_gate_result(
    record: Any,
    *,
    contract: Mapping[str, Any] | None = None,
    repository_paths: Mapping[str, Path] | None = None,
    lane_log_paths: Mapping[str, Path] | None = None,
    package_result_path: Path | None = None,
    request_paths: Mapping[str, Path] | None = None,
    wheel_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a prospective Q1M gate result without creating evidence.

    Passing ``repository_paths`` enables the final clean-tree and Git-object
    identity check. Its keys must name ANYfeSolver and all five siblings.
    """

    contract = dict(_load_contract() if contract is None else contract)
    required_top_keys = {
        "candidate",
        "clean_gate_index",
        "lanes",
        "legacy_removal_authorized",
        "production_boundary",
        "resource_requests",
        "rollback",
        "schema",
        "siblings",
    }
    optional_top_keys = {
        "hard_gates",
        "package_result",
        "performance_baseline",
        "wheel",
    }
    if not isinstance(record, dict):
        raise EvidenceError("$ must be an object")
    missing = required_top_keys - set(record)
    extra = set(record) - required_top_keys - optional_top_keys
    if missing or extra:
        raise EvidenceError(
            f"$ keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if record["schema"] != contract["gate_result_schema"]:
        raise EvidenceError("$.schema does not match the contract")

    candidate = _exact_keys(
        record["candidate"], {"clean", "commit", "repository", "tree"}, "$.candidate"
    )
    contract_identity = _sha256_bytes(canonical_json_bytes(contract))
    historical_cycle = contract_identity in _HISTORICAL_CONTRACT_IDENTITIES
    expected_repository = "ANYsolver" if historical_cycle else "ANYfeSolver"
    if candidate["repository"] != expected_repository:
        raise EvidenceError(f"$.candidate.repository must be {expected_repository}")
    if not _require_bool(candidate["clean"], "$.candidate.clean"):
        raise EvidenceError("the candidate repository must be clean")
    _require_match(candidate["commit"], GIT_OBJECT_RE, "$.candidate.commit")
    _require_match(candidate["tree"], GIT_OBJECT_RE, "$.candidate.tree")

    siblings = _exact_keys(record["siblings"], set(SIBLING_NAMES), "$.siblings")
    for name in SIBLING_NAMES:
        identity = _exact_keys(
            siblings[name], {"clean", "commit", "tree"}, f"$.siblings.{name}"
        )
        if not _require_bool(identity["clean"], f"$.siblings.{name}.clean"):
            raise EvidenceError(f"sibling repository {name} must be clean")
        _require_match(identity["commit"], GIT_OBJECT_RE, f"$.siblings.{name}.commit")
        _require_match(identity["tree"], GIT_OBJECT_RE, f"$.siblings.{name}.tree")
    sibling_authority = contract.get("sibling_authority")
    if sibling_authority is None:
        # Immutable cycle-0/1/2 authorities predate the complete sibling
        # graph and bind the paired adapter commit directly.
        if siblings["ANYfem"]["commit"] != contract["anyfem_commit"]:
            raise EvidenceError("ANYfem commit does not match the contract")
    else:
        sibling_authority = _exact_keys(
            sibling_authority,
            set(SIBLING_NAMES),
            "$.contract.sibling_authority",
        )
        for name in SIBLING_NAMES:
            expected = _exact_keys(
                sibling_authority[name],
                {"commit", "tree"},
                f"$.contract.sibling_authority.{name}",
            )
            _require_match(
                expected["commit"],
                GIT_OBJECT_RE,
                f"$.contract.sibling_authority.{name}.commit",
            )
            _require_match(
                expected["tree"],
                GIT_OBJECT_RE,
                f"$.contract.sibling_authority.{name}.tree",
            )
            if siblings[name]["commit"] != expected["commit"]:
                raise EvidenceError(
                    f"sibling repository {name} commit does not match the contract"
                )
            if siblings[name]["tree"] != expected["tree"]:
                raise EvidenceError(
                    f"sibling repository {name} tree does not match the contract"
                )
        if contract.get("anyfem_commit") != sibling_authority["ANYfem"]["commit"]:
            raise EvidenceError("ANYfem authority aliases disagree")

    expected_requests = contract["resource_requests"]
    requests = record["resource_requests"]
    if not isinstance(requests, list) or len(requests) != len(expected_requests):
        raise EvidenceError("$.resource_requests must contain every contracted request")
    seen_ids: set[str] = set()
    seen_lanes: set[str] = set()
    for index, request in enumerate(requests):
        location = f"$.resource_requests[{index}]"
        request = _exact_keys(
            request, {"lane", "request_id", "request_sha256"}, location
        )
        lane = request["lane"]
        if lane not in expected_requests:
            raise EvidenceError(f"{location}.lane is not contracted")
        request_id = _require_match(
            request["request_id"], REQUEST_ID_RE, f"{location}.request_id"
        )
        _require_match(request["request_sha256"], SHA256_RE, f"{location}.request_sha256")
        if request_id in seen_ids or lane in seen_lanes:
            raise EvidenceError("resource request IDs and lanes must be unique")
        seen_ids.add(request_id)
        seen_lanes.add(lane)
        if request_id != expected_requests[lane]["request_id"]:
            raise EvidenceError(f"{location}.request_id does not match the contract")
        if request["request_sha256"] != expected_requests[lane]["request_sha256"]:
            raise EvidenceError(f"{location}.request_sha256 does not match the contract")
    if seen_lanes != set(expected_requests):
        raise EvidenceError("resource request lanes are incomplete")

    lanes = _exact_keys(record["lanes"], set(GATE_LANES), "$.lanes")
    historical_inventory_authority = (
        _HISTORICAL_INVENTORY_IDENTITIES
        if historical_cycle
        else None
    )
    expected_inventories = (
        None if historical_inventory_authority is not None else gate_inventories()
    )
    all_pass = True
    statuses: list[str] = []
    for lane in GATE_LANES:
        location = f"$.lanes.{lane}"
        raw_result = lanes[lane]
        if not isinstance(raw_result, dict) or raw_result.get("status") not in {
            "PASS",
            "FAIL",
            "NOT_RUN",
        }:
            raise EvidenceError(f"{location}.status must be PASS, FAIL, or NOT_RUN")
        status = raw_result["status"]
        base_keys = {
            "command_sha256",
            "inventory",
            "resource_request_id",
            "status",
        }
        result = _exact_keys(
            raw_result,
            base_keys
            if status == "NOT_RUN"
            else base_keys | {"exit_code", "finished_at_utc", "log", "started_at_utc"},
            location,
        )
        statuses.append(status)
        if status == "NOT_RUN":
            all_pass = False
        else:
            exit_code = _require_int(result["exit_code"], f"{location}.exit_code")
            if (status == "PASS") != (exit_code == 0):
                raise EvidenceError(f"{location}.exit_code contradicts status")
            all_pass = all_pass and status == "PASS"
            started = _require_timestamp(result["started_at_utc"], f"{location}.started_at_utc")
            finished = _require_timestamp(result["finished_at_utc"], f"{location}.finished_at_utc")
            if finished < started:
                raise EvidenceError(f"{location} finishes before it starts")
            _validate_hash_record(result["log"], f"{location}.log", allow_empty=False)
        _require_match(
            result["command_sha256"], SHA256_RE, f"{location}.command_sha256"
        )
        expected_command_sha256 = (
            contract["non_resource_commands"][lane]["command_sha256"]
            if lane in contract["non_resource_commands"]
            else expected_requests[lane]["command_sha256"]
        )
        if result["command_sha256"] != expected_command_sha256:
            raise EvidenceError(f"$.lanes.{lane}.command_sha256 does not match authority")
        if historical_inventory_authority is not None:
            inventory_identity = _sha256_bytes(
                canonical_json_bytes(result["inventory"])
            )
            if inventory_identity != historical_inventory_authority[lane]:
                raise EvidenceError(
                    f"{location}.inventory does not match historical authority"
                )
        elif result["inventory"] != expected_inventories[lane]:
            raise EvidenceError(f"{location}.inventory does not match discovery")
        expected_id = None if lane in {"quick", "package"} else expected_requests[lane]["request_id"]
        if result["resource_request_id"] != expected_id:
            raise EvidenceError(f"{location}.resource_request_id is invalid")

    failed_indices = [index for index, status in enumerate(statuses) if status == "FAIL"]
    if failed_indices:
        if len(failed_indices) != 1:
            raise EvidenceError("a fail-fast result must contain exactly one failed lane")
        failed_index = failed_indices[0]
        if statuses[:failed_index] != ["PASS"] * failed_index or statuses[failed_index + 1 :] != [
            "NOT_RUN"
        ] * (len(statuses) - failed_index - 1):
            raise EvidenceError("lane statuses violate fail-fast execution order")
    elif "NOT_RUN" in statuses:
        raise EvidenceError("NOT_RUN lanes require one preceding failed lane")

    if lanes["performance"]["status"] == "PASS":
        if not {"hard_gates", "performance_baseline"} <= set(record):
            raise EvidenceError("successful performance lane requires its exact evidence")
        hard_gates_pass = _validate_performance_evidence(
            record["hard_gates"],
            record["performance_baseline"],
            contract=contract,
            lanes=lanes,
        )
        all_pass = all_pass and hard_gates_pass
    elif {"hard_gates", "performance_baseline"} & set(record):
        raise EvidenceError("performance evidence is forbidden when the lane did not pass")

    wheel: dict[str, Any] | None = None
    if lanes["package"]["status"] == "PASS":
        if not {"package_result", "wheel"} <= set(record):
            raise EvidenceError("successful package lane requires result and wheel identities")
        wheel = _exact_keys(record["wheel"], {"bytes", "filename", "sha256"}, "$.wheel")
        if not isinstance(wheel["filename"], str) or not wheel["filename"].endswith(".whl"):
            raise EvidenceError("$.wheel.filename must be a wheel basename")
        if Path(wheel["filename"]).name != wheel["filename"]:
            raise EvidenceError("$.wheel.filename must not contain a path")
        _require_int(wheel["bytes"], "$.wheel.bytes", minimum=1)
        _require_match(wheel["sha256"], SHA256_RE, "$.wheel.sha256")
        _validate_hash_record(record["package_result"], "$.package_result", allow_empty=False)
    elif {"package_result", "wheel"} & set(record):
        raise EvidenceError("package artifacts are forbidden when the lane did not pass")

    rollback = _exact_keys(record["rollback"], {"state", "unresolved_incidents"}, "$.rollback")
    incidents = rollback["unresolved_incidents"]
    if not isinstance(incidents, list) or any(
        not isinstance(item, str) or not item for item in incidents
    ):
        raise EvidenceError("$.rollback.unresolved_incidents must be a string list")
    if len(incidents) != len(set(incidents)):
        raise EvidenceError("rollback incidents must be unique")
    expected_rollback_state = (
        "NO_UNRESOLVED_ROLLBACK_INCIDENT"
        if not incidents
        else "UNRESOLVED_LEGACY_Q4_ROLLBACK_INCIDENT"
    )
    if rollback["state"] != expected_rollback_state:
        raise EvidenceError("$.rollback.state contradicts unresolved incidents")
    if not all_pass and not incidents:
        raise EvidenceError("a failed required gate must record a rollback incident")

    expected_gate_index = 1 if all_pass and not incidents else 0
    if _require_int(record["clean_gate_index"], "$.clean_gate_index") != expected_gate_index:
        raise EvidenceError("$.clean_gate_index contradicts lane/rollback state")
    if _require_bool(record["legacy_removal_authorized"], "$.legacy_removal_authorized"):
        raise EvidenceError("Q1M gate 1 cannot authorize legacy removal")

    boundary = _exact_keys(
        record["production_boundary"],
        {"default", "legacy_q4_available_through", "mechanics_changed", "removal_not_before"},
        "$.production_boundary",
    )
    if boundary != contract["production_boundary"]:
        raise EvidenceError("$.production_boundary does not match the Q1M contract")
    if _require_bool(boundary["mechanics_changed"], "$.production_boundary.mechanics_changed"):
        raise EvidenceError("Q1M may not change qualified Q4 mechanics")

    if repository_paths is not None:
        _validate_repository_identities(record, repository_paths, contract)
    if lane_log_paths is not None:
        executed_lanes = {
            lane for lane in GATE_LANES if lanes[lane]["status"] != "NOT_RUN"
        }
        if set(lane_log_paths) != executed_lanes:
            raise EvidenceError("lane_log_paths must name exactly the executed gate lanes")
        for lane in executed_lanes:
            if file_hash_record(Path(lane_log_paths[lane])) != lanes[lane]["log"]:
                raise EvidenceError(f"external log mismatch for lane {lane}")
        if lanes["performance"]["status"] == "PASS":
            observation = extract_performance_observation(
                Path(lane_log_paths["performance"])
            )
            if observation["hard_gates"] != record["hard_gates"]:
                raise EvidenceError("performance hard gates do not match their external log")
            if observation["performance_baseline"] != record["performance_baseline"]:
                raise EvidenceError("performance baseline does not match its external log")
    if request_paths is not None:
        if set(request_paths) != set(expected_requests):
            raise EvidenceError("request_paths must name every resource lane")
        for lane, expected in expected_requests.items():
            path = Path(request_paths[lane])
            if path.name != f"{expected['request_id']}.json":
                raise EvidenceError(f"request filename mismatch for lane {lane}")
            if file_hash_record(path)["sha256"] != expected["request_sha256"]:
                raise EvidenceError(f"immutable request hash mismatch for lane {lane}")
            request = strict_json_load(path)
            if not isinstance(request, dict) or request.get("request_id") != expected["request_id"]:
                raise EvidenceError(f"immutable request identity mismatch for lane {lane}")
            command = request.get("command")
            if not isinstance(command, str) or _sha256_bytes(command.encode("utf-8")) != expected[
                "command_sha256"
            ]:
                raise EvidenceError(f"immutable request command mismatch for lane {lane}")
    if lanes["package"]["status"] == "PASS" and package_result_path is not None:
        package_path = Path(package_result_path)
        package_value = strict_json_load(package_path)
        package_raw = package_path.read_bytes()
        package = validate_package_result(package_value, contract=contract)
        if package_raw != canonical_json_bytes(package):
            raise EvidenceError("package result is not canonical JSON")
        if file_hash_record(package_path) != record["package_result"]:
            raise EvidenceError("package result identity mismatch")
        expected_source_identities = {
            "ANYfeSolver": record["candidate"],
            "ANYmesh": record["siblings"]["ANYmesh"],
            "ANYgeometry": record["siblings"]["ANYgeometry"],
            "ANYmaterial": record["siblings"]["ANYmaterial"],
            "ANYfileIO": record["siblings"]["ANYfileIO"],
        }
        for name, identity in expected_source_identities.items():
            source = package["sources"][name]
            if source["commit"] != identity["commit"] or source["tree"] != identity["tree"]:
                raise EvidenceError(f"package source identity mismatch for {name}")
        if package["wheels"]["ANYfeSolver"] != record["wheel"]:
            raise EvidenceError("package result ANYfeSolver wheel mismatch")
    elif package_result_path is not None:
        raise EvidenceError("package result path is forbidden when package did not pass")
    if lanes["package"]["status"] == "PASS" and wheel_path is not None:
        if wheel_hash_record(Path(wheel_path)) != wheel:
            raise EvidenceError("installed ANYfeSolver wheel identity mismatch")
    elif wheel_path is not None:
        raise EvidenceError("wheel path is forbidden when package did not pass")
    return record


def _git(
    repository: Path, *args: str, timeout_seconds: float | None = None
) -> str:
    timeout_options: dict[str, Any] = {}
    if timeout_seconds is not None:
        timeout_options["timeout"] = timeout_seconds
    try:
        completed = subprocess.run(
            [
                *_git_command_prefix(repository),
                *args,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=_sanitized_git_environment(),
            **timeout_options,
        )
    except subprocess.TimeoutExpired as exc:
        raise EvidenceError(f"Git identity check timed out for {repository}") from exc
    if completed.returncode:
        raise EvidenceError(f"Git identity check failed for {repository}")
    return completed.stdout.strip()


def _validate_repository_identities(
    record: Mapping[str, Any],
    repository_paths: Mapping[str, Path],
    contract: Mapping[str, Any],
) -> None:
    expected_names = {"ANYfeSolver", *SIBLING_NAMES}
    if set(repository_paths) != expected_names:
        raise EvidenceError("repository_paths must name the candidate and all siblings")
    identities = {"ANYfeSolver": record["candidate"], **record["siblings"]}
    if "execution" in contract:
        empty_status = contract["execution"]["clean_status_empty"]
        repository_order = contract["execution"]["clean_status_scope"]
        if (
            not isinstance(repository_order, list)
            or any(not isinstance(name, str) for name in repository_order)
            or len(repository_order) != len(set(repository_order))
            or set(repository_order) != expected_names
        ):
            raise EvidenceError(
                f"{S3_Q4_ACTIVE_AUTHORITY_GENERATION} clean-status repository scope "
                "is malformed, duplicated, or incomplete"
            )
    else:
        # Historical Q1M contracts predate the named S3/Q4 clean-status policy.
        empty_status = {"bytes": 0, "sha256": _sha256_bytes(b"")}
        repository_order = sorted(expected_names)
    for name in repository_order:
        repository = Path(repository_paths[name]).resolve()
        if _functional_source_status(repository) != empty_status:
            raise EvidenceError(f"repository {name} is not completely clean")
        if _git(repository, "rev-parse", "HEAD") != identities[name]["commit"]:
            raise EvidenceError(f"repository {name} commit mismatch")
        if _git(repository, "rev-parse", "HEAD^{tree}") != identities[name]["tree"]:
            raise EvidenceError(f"repository {name} tree mismatch")


def validate_final_gate_result(
    record: Any,
    *,
    repository_paths: Mapping[str, Path],
    lane_log_paths: Mapping[str, Path],
    package_result_path: Path | None,
    request_paths: Mapping[str, Path],
    wheel_path: Path | None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authorize a gate result only after every external binding is checked."""

    package_passed = record.get("lanes", {}).get("package", {}).get("status") == "PASS"
    if package_passed and (package_result_path is None or wheel_path is None):
        raise EvidenceError("a passed package lane requires package-result and wheel paths")
    if not package_passed and (package_result_path is not None or wheel_path is not None):
        raise EvidenceError("an unpassed package lane forbids package-result and wheel paths")

    return validate_gate_result(
        record,
        contract=contract,
        repository_paths=repository_paths,
        lane_log_paths=lane_log_paths,
        package_result_path=package_result_path,
        request_paths=request_paths,
        wheel_path=wheel_path,
    )


def validate_adjudication_files(
    gate_result_path: Path,
    status_path: Path,
    review_path: Path,
    *,
    contract_path: Path = CONTRACT_PATH,
    repository_root: Path | None = ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the frozen status and independent five-key review."""

    contract = strict_json_load(contract_path)
    gate_value = strict_json_load(gate_result_path)
    status_value = strict_json_load(status_path)
    review_value = strict_json_load(review_path)
    gate_raw = gate_result_path.read_bytes()
    status_raw = status_path.read_bytes()
    review_raw = review_path.read_bytes()
    gate = validate_gate_result(gate_value, contract=contract)
    status = _exact_keys(
        status_value,
        set(contract["adjudication"]["status_required_keys"]),
        "$status",
    )
    review = _exact_keys(
        review_value,
        set(contract["adjudication"]["review_required_keys"]),
        "$review",
    )
    for raw, value, label in (
        (gate_raw, gate, "gate result"),
        (status_raw, status, "status"),
        (review_raw, review, "review"),
    ):
        if raw != canonical_json_bytes(value):
            raise EvidenceError(f"{label} is not canonical JSON")
    if status["schema"] != contract["status_schema"]:
        raise EvidenceError("status schema mismatch")
    if status["gate_result_sha256"] != _sha256_bytes(gate_raw):
        raise EvidenceError("status gate-result hash mismatch")
    _require_match(status["gate_result_sha256"], SHA256_RE, "$status.gate_result_sha256")
    expected_success = gate["clean_gate_index"] == 1
    if repository_root is not None:
        root = repository_root.resolve()
        try:
            actual_paths = [
                path.resolve().relative_to(root).as_posix()
                for path in (gate_result_path, status_path, review_path)
            ]
        except ValueError as exc:
            raise EvidenceError("adjudication output is outside the repository") from exc
        route = "success_paths" if expected_success else "blocked_paths"
        if actual_paths != contract["adjudication"][route]:
            raise EvidenceError("adjudication output extent mismatch")
    expected_terminal = contract["adjudication"][
        "success_terminal" if expected_success else "blocked_terminal"
    ]
    if status["terminal"] != expected_terminal:
        raise EvidenceError("status terminal mismatch")
    if status["clean_gate_index"] != gate["clean_gate_index"]:
        raise EvidenceError("status clean-gate index mismatch")
    if status["legacy_removal_authorized"] is not False:
        raise EvidenceError("status cannot authorize legacy removal")

    if review["schema"] != contract["review_schema"]:
        raise EvidenceError("review schema mismatch")
    if review["findings"] != []:
        raise EvidenceError("accepted independent review must have no findings")
    if review["reviewer_independence"] != contract["adjudication"]["review_independence"]:
        raise EvidenceError("review independence mismatch")
    expected_verdict = contract["adjudication"][
        "accepted_success_verdict" if expected_success else "accepted_blocked_verdict"
    ]
    if review["verdict"] != expected_verdict:
        raise EvidenceError("review verdict mismatch")
    reviewed_inputs = _exact_keys(
        review["reviewed_inputs"],
        set(contract["adjudication"]["reviewed_input_hashes"]),
        "$review.reviewed_inputs",
    )
    expected_inputs = {
        "contract_sha256": _sha256_file(contract_path),
        "gate_result_sha256": _sha256_bytes(gate_raw),
        "status_sha256": _sha256_bytes(status_raw),
    }
    if reviewed_inputs != expected_inputs:
        raise EvidenceError("reviewed-input hashes mismatch")
    return gate, status, review


def canonical_gate_result_bytes(
    record: Any,
    *,
    contract: Mapping[str, Any] | None = None,
    repository_paths: Mapping[str, Path] | None = None,
    lane_log_paths: Mapping[str, Path] | None = None,
    package_result_path: Path | None = None,
    request_paths: Mapping[str, Path] | None = None,
    wheel_path: Path | None = None,
) -> bytes:
    validated = validate_gate_result(
        record,
        contract=contract,
        repository_paths=repository_paths,
        lane_log_paths=lane_log_paths,
        package_result_path=package_result_path,
        request_paths=request_paths,
        wheel_path=wheel_path,
    )
    return canonical_json_bytes(validated)


def write_gate_result_exclusive(
    path: Path,
    record: Any,
    *,
    contract: Mapping[str, Any] | None = None,
    repository_paths: Mapping[str, Path] | None = None,
    lane_log_paths: Mapping[str, Path] | None = None,
    package_result_path: Path | None = None,
    request_paths: Mapping[str, Path] | None = None,
    wheel_path: Path | None = None,
) -> None:
    if repository_paths is None or lane_log_paths is None or request_paths is None:
        raise EvidenceError("final gate-result creation requires repository/request/log bindings")
    package_passed = record.get("lanes", {}).get("package", {}).get("status") == "PASS"
    if package_passed and (package_result_path is None or wheel_path is None):
        raise EvidenceError("final passed-package result requires package and wheel bindings")
    if not package_passed and (package_result_path is not None or wheel_path is not None):
        raise EvidenceError("blocked pre-package result forbids package and wheel bindings")
    payload = canonical_gate_result_bytes(
        record,
        contract=contract,
        repository_paths=repository_paths,
        lane_log_paths=lane_log_paths,
        package_result_path=package_result_path,
        request_paths=request_paths,
        wheel_path=wheel_path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise EvidenceError(f"refusing to replace existing evidence: {path}") from exc


def _github_root() -> Path:
    if len(ROOT.parents) < 3:
        raise RuntimeError("cannot locate the common GitHub repository directory")
    return ROOT.parents[2]


def _local_roots() -> dict[str, Path]:
    github = _github_root()
    roots = {
        "ANYgeometry": github / "ANYgeometry",
        "ANYmaterial": github / "ANYmaterial",
        "ANYmesh": github / "ANYmesh",
        "ANYfileIO": github / "ANYfileIO",
        "ANYfeSolver": ROOT,
    }
    for name in tuple(roots):
        override = os.environ.get(f"Q1M_{name.upper()}_ROOT")
        if override:
            roots[name] = Path(override).resolve()
    for name, path in roots.items():
        if not (path / "pyproject.toml").is_file():
            raise RuntimeError(f"required local source snapshot is unavailable: {name}")
    return roots


def _normalise_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _write_source_metadata_overlay(
    roots: Mapping[str, Path], destination: Path
) -> dict[str, str]:
    """Create deterministic distribution metadata for the exact source graph.

    Source worktrees intentionally contain no generated distribution metadata.
    Without this overlay, importlib.metadata can report an unrelated globally
    installed distribution even while Python imports the frozen source tree.
    Each record is derived solely from that source tree's tracked pyproject
    file and is deleted with the lane-local pytest temp directory.
    """

    expected_roots = {name for name, _distribution, _package in LOCAL_DISTRIBUTIONS}
    if set(roots) != expected_roots:
        raise EvidenceError("source metadata roots do not match the package graph")
    destination.mkdir()
    versions: dict[str, str] = {}
    for repository_name, distribution_name, _package_name in LOCAL_DISTRIBUTIONS:
        project_path = roots[repository_name] / "pyproject.toml"
        if not project_path.is_file() or project_path.is_symlink():
            raise EvidenceError(
                f"source project metadata is unavailable: {repository_name}"
            )
        try:
            document = tomllib.loads(project_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise EvidenceError(
                f"source project metadata is invalid: {repository_name}"
            ) from exc
        project = document.get("project")
        if not isinstance(project, dict):
            raise EvidenceError(f"source project table is missing: {repository_name}")
        name = project.get("name")
        version = project.get("version")
        if (
            not isinstance(name, str)
            or not PROJECT_NAME_RE.fullmatch(name)
            or _normalise_distribution_name(name)
            != _normalise_distribution_name(distribution_name)
        ):
            raise EvidenceError(
                f"source distribution name mismatch: {repository_name}"
            )
        if not isinstance(version, str) or not PROJECT_VERSION_RE.fullmatch(version):
            raise EvidenceError(
                f"source distribution version is invalid: {repository_name}"
            )
        normalised = _normalise_distribution_name(distribution_name).replace("-", "_")
        record = destination / f"{normalised}-{version}.dist-info"
        record.mkdir()
        metadata = (
            "Metadata-Version: 2.1\n"
            f"Name: {distribution_name}\n"
            f"Version: {version}\n\n"
        ).encode("ascii")
        with (record / "METADATA").open("xb") as stream:
            stream.write(metadata)
        versions[distribution_name] = version
    return versions


def _pytest_environment(
    *, roots: Mapping[str, Path], metadata_overlay: Path
) -> dict[str, str]:
    """Provide the exact source set required by the frozen preflight command."""

    environment = os.environ.copy()
    for name in _S3_Q4_STYLE_WAVE_ENVIRONMENT:
        environment.pop(name, None)
    environment["PYTHONPATH"] = os.pathsep.join(
        str(path)
        for path in (
            metadata_overlay,
            ROOT / "src",
            roots["ANYmesh"] / "src",
            roots["ANYgeometry"] / "src",
            roots["ANYmaterial"] / "src",
            roots["ANYfileIO"] / "src",
        )
    )
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _functional_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{location} must be a nonempty string")
    return value


def _functional_route_name(value: Any, location: str, *, prefix: bool = False) -> str:
    name = _functional_string(value, location)
    if (
        name in {".", ".."}
        or Path(name).name != name
        or "/" in name
        or "\\" in name
        or (not prefix and name.endswith(("/", "\\")))
    ):
        raise EvidenceError(f"{location} must be a safe {'prefix' if prefix else 'basename'}")
    return name


def _functional_list_hash(values: Sequence[str]) -> str:
    return _sha256_bytes(canonical_json_bytes(list(values)))


def _functional_remaining_seconds(absolute_deadline: float, stage: str) -> float:
    remaining = absolute_deadline - time.monotonic()
    if remaining <= 0:
        raise FunctionalDeadlineExpired(
            f"functional internal deadline expired before {stage}"
        )
    return max(0.001, remaining)


def _functional_deadline_check(absolute_deadline: float, stage: str) -> None:
    _functional_remaining_seconds(absolute_deadline, stage)


def _functional_bounded_hash_record(
    path: Path, absolute_deadline: float, stage: str
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or _is_reparse_point(path):
        raise EvidenceError(f"functional bounded artifact is not regular: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while True:
            _functional_deadline_check(absolute_deadline, stage)
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
    _functional_deadline_check(absolute_deadline, f"{stage} completion")
    return {"bytes": byte_count, "sha256": digest.hexdigest()}


def _functional_bounded_read_bytes(
    path: Path, absolute_deadline: float, stage: str
) -> bytes:
    if not path.is_file() or path.is_symlink() or _is_reparse_point(path):
        raise EvidenceError(f"functional bounded artifact is not regular: {path}")
    chunks: list[bytes] = []
    with path.open("rb") as stream:
        while True:
            _functional_deadline_check(absolute_deadline, stage)
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    _functional_deadline_check(absolute_deadline, f"{stage} completion")
    return b"".join(chunks)


def _validate_functional_timeout_policy(
    contract: Mapping[str, Any], *, schema: str = FUNCTIONAL_WAVE_SCHEMA
) -> dict[str, Any]:
    policy = _exact_keys(
        contract["execution"]["timeout_policy"],
        {
            "automatic_retry",
            "evidence_reserve_seconds",
            "scope",
            "termination_grace_seconds",
            "timeout_exit_code",
            "wall_limit_seconds",
            "windows_job",
            "windows_termination",
        },
        "$contract.execution.timeout_policy",
    )
    if policy["scope"] != "COMPLETE_RESOURCE_INVOCATION_AND_CHILD_PROCESS_TREE":
        raise EvidenceError("functional timeout scope must cover the complete child tree")
    v19 = schema == FUNCTIONAL_WAVE_V19_SCHEMA
    expected_wall_limit = 900 if v19 else 1200
    expected_watchdog_start = 890 if v19 else 1190
    expected_evidence_reserve = 10 if v19 else 20
    if policy["windows_job"] != {
        "assignment": "CREATE_SUSPENDED_ASSIGN_RESUME",
        "limit": "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE",
        "watchdog_termination_start_seconds": expected_watchdog_start,
    }:
        raise EvidenceError("functional Windows job-object policy mismatch")
    if _require_bool(
        policy["automatic_retry"],
        "$contract.execution.timeout_policy.automatic_retry",
    ):
        raise EvidenceError("functional timeout policy forbids automatic retry")
    if _require_int(
        policy["evidence_reserve_seconds"],
        "$contract.execution.timeout_policy.evidence_reserve_seconds",
        minimum=1,
    ) != expected_evidence_reserve:
        raise EvidenceError(
            "functional evidence reserve differs from the schema authority"
        )
    if _require_int(
        policy["wall_limit_seconds"],
        "$contract.execution.timeout_policy.wall_limit_seconds",
        minimum=1,
    ) != expected_wall_limit:
        raise EvidenceError(
            "functional outer wall limit differs from the schema authority"
        )
    if _require_int(
        policy["timeout_exit_code"],
        "$contract.execution.timeout_policy.timeout_exit_code",
        minimum=1,
    ) != 124:
        raise EvidenceError("functional timeout exit code must be 124")
    if _require_int(
        policy["termination_grace_seconds"],
        "$contract.execution.timeout_policy.termination_grace_seconds",
        minimum=1,
    ) != 10:
        raise EvidenceError("functional termination grace must be ten seconds")
    windows = _exact_keys(
        policy["windows_termination"],
        {"arguments", "bytes", "path", "sha256"},
        "$contract.execution.timeout_policy.windows_termination",
    )
    path = Path(
        _functional_string(
            windows["path"], "$contract.execution.timeout_policy.windows_termination.path"
        )
    )
    if not path.is_absolute():
        raise EvidenceError("taskkill authority must be an absolute path")
    _validate_hash_record(
        {"bytes": windows["bytes"], "sha256": windows["sha256"]},
        "$contract.execution.timeout_policy.windows_termination",
        allow_empty=False,
    )
    if windows["arguments"] != ["/PID", "{pid}", "/T", "/F"]:
        raise EvidenceError("taskkill arguments differ from frozen authority")
    return policy


def validate_functional_wave_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact bounded functional-wave authority."""

    raw_wave = contract["functional_wave"]
    if not isinstance(raw_wave, dict):
        raise EvidenceError("$contract.functional_wave must be an object")
    schema = raw_wave.get("schema")
    v19 = schema == FUNCTIONAL_WAVE_V19_SCHEMA
    wave = _exact_keys(
        raw_wave,
        (
            {
                "aggregate",
                "deferred_activation",
                "execution",
                "functional_wave_schema_version",
                "manifest",
                "schema",
                "source",
            }
            if v19
            else {"aggregate", "execution", "manifest", "schema", "source"}
        ),
        "$contract.functional_wave",
    )
    if wave["schema"] not in {FUNCTIONAL_WAVE_SCHEMA, FUNCTIONAL_WAVE_V19_SCHEMA}:
        raise EvidenceError("functional-wave schema mismatch")
    if v19 and _require_int(
        wave["functional_wave_schema_version"],
        "$contract.functional_wave.functional_wave_schema_version",
        minimum=1,
    ) != 3:
        raise EvidenceError("v19 functional-wave schema version must be three")

    deferred_activation: dict[str, Any] | None = None
    if v19:
        deferred_activation = _exact_keys(
            wave["deferred_activation"],
            {"disposition", "node_count", "node_ids", "node_ids_sha256", "reason"},
            "$contract.functional_wave.deferred_activation",
        )
        deferred_nodes = deferred_activation["node_ids"]
        if (
            not isinstance(deferred_nodes, list)
            or deferred_nodes != list(FUNCTIONAL_V20_DEFERRED_ACTIVATION_NODES)
            or len(set(deferred_nodes)) != len(deferred_nodes)
        ):
            raise EvidenceError("deferred activation node authority mismatch")
        observed_deferred_authority = {
            "disposition": deferred_activation["disposition"],
            "node_count": _require_int(
                deferred_activation["node_count"],
                "$contract.functional_wave.deferred_activation.node_count",
                minimum=1,
            ),
            "node_ids_sha256": deferred_activation["node_ids_sha256"],
            "reason": deferred_activation["reason"],
        }
        if (
            observed_deferred_authority
            != FUNCTIONAL_V20_DEFERRED_ACTIVATION_AUTHORITY
            or deferred_activation["node_ids_sha256"]
            != _functional_list_hash(deferred_nodes)
        ):
            raise EvidenceError("deferred activation disposition differs from authority")

    source = _exact_keys(
        wave["source"],
        {
            "archive_filename",
            "commit_role",
            "file_graph_filename",
            "file_graph_schema",
            "tree_role",
        },
        "$contract.functional_wave.source",
    )
    if source["commit_role"] != "EXECUTION_AUTHORIZATION_COMMIT":
        raise EvidenceError("functional source commit role mismatch")
    if source["tree_role"] != "EXECUTION_AUTHORIZATION_TREE":
        raise EvidenceError("functional source tree role mismatch")
    for key in ("archive_filename", "file_graph_filename"):
        _functional_route_name(
            source[key], f"$contract.functional_wave.source.{key}"
        )
    _functional_string(
        source["file_graph_schema"],
        "$contract.functional_wave.source.file_graph_schema",
    )
    if source["archive_filename"] != "functional-wave-source.tar":
        raise EvidenceError("functional source archive filename mismatch")
    if source["file_graph_filename"] != "functional-wave-source-file-graph.json":
        raise EvidenceError("functional source file-graph filename mismatch")
    if (
        source["file_graph_schema"]
        != "anysolver.e4-pl-s3-q4-functional-wave-file-graph-v1"
    ):
        raise EvidenceError("functional source file-graph schema mismatch")

    manifest = _exact_keys(
        wave["manifest"],
        {
            "collection_artifact",
            "full_node_ids",
            "full_node_ids_sha256",
            "module_count",
            "modules",
            "modules_sha256",
            "node_count",
            "shards",
        },
        "$contract.functional_wave.manifest",
    )
    modules = manifest["modules"]
    if (
        not isinstance(modules, list)
        or not modules
        or len(set(modules)) != len(modules)
        or not all(
            isinstance(item, str)
            and item.startswith("tests/test_")
            and item.endswith(".py")
            and "\\" not in item
            for item in modules
        )
    ):
        raise EvidenceError("functional module manifest is malformed")
    if _require_int(
        manifest["module_count"],
        "$contract.functional_wave.manifest.module_count",
        minimum=1,
    ) != len(modules):
        raise EvidenceError("functional module count mismatch")
    if len(modules) != 85:
        raise EvidenceError("functional module authority must contain 85 modules")
    if manifest["modules_sha256"] != _functional_list_hash(modules):
        raise EvidenceError("functional module-list hash mismatch")
    discovered_modules = inventory()["functional"]
    if modules != discovered_modules:
        raise EvidenceError("functional module manifest differs from current inventory")

    full_nodes = manifest["full_node_ids"]
    module_set = set(modules)
    if (
        not isinstance(full_nodes, list)
        or not full_nodes
        or len(set(full_nodes)) != len(full_nodes)
        or not all(
            isinstance(item, str)
            and "\\" not in item
            and item.partition("::")[0] in module_set
            and bool(item.partition("::")[1])
            for item in full_nodes
        )
    ):
        raise EvidenceError("functional full-node manifest is malformed")
    if v19 and deferred_activation is not None:
        deferred_nodes = deferred_activation["node_ids"]
        if set(full_nodes).intersection(deferred_nodes):
            raise EvidenceError("activation-only nodes cannot enter opt-in burn-in")
        if not all(node.partition("::")[0] in module_set for node in deferred_nodes):
            raise EvidenceError("deferred activation node is outside functional inventory")
    if _require_int(
        manifest["node_count"],
        "$contract.functional_wave.manifest.node_count",
        minimum=1,
    ) != len(full_nodes):
        raise EvidenceError("functional node count mismatch")
    if not v19 and len(full_nodes) != 1036:
        raise EvidenceError("functional node authority must contain 1036 nodes")
    if manifest["full_node_ids_sha256"] != _functional_list_hash(full_nodes):
        raise EvidenceError("functional full-node hash mismatch")
    if v19 and {
        "node_count": manifest["node_count"],
        "node_ids_sha256": manifest["full_node_ids_sha256"],
    } != FUNCTIONAL_V19_FULL_NODE_AUTHORITY:
        raise EvidenceError("v19 full-node manifest differs from frozen authority")
    _validate_hash_record(
        manifest["collection_artifact"],
        "$contract.functional_wave.manifest.collection_artifact",
        allow_empty=False,
    )
    if v19:
        canonical_node_inventory = canonical_json_bytes(full_nodes)
        if manifest["collection_artifact"] != {
            "bytes": len(canonical_node_inventory),
            "sha256": _sha256_bytes(canonical_node_inventory),
        }:
            raise EvidenceError(
                "v19 collection artifact must be the canonical full-node inventory"
            )
    expected_collection_artifact = (
        FUNCTIONAL_V19_COLLECTION_ARTIFACT
        if v19
        else FUNCTIONAL_COLLECTION_ARTIFACT
    )
    if manifest["collection_artifact"] != expected_collection_artifact:
        raise EvidenceError("functional collection artifact identity mismatch")

    shards = manifest["shards"]
    expected_shard_ids = list(FUNCTIONAL_V19_SHARD_IDS) if v19 else [
        "P01",
        "P02",
        "P03",
        "P04",
    ]
    if not isinstance(shards, list) or len(shards) != len(expected_shard_ids):
        raise EvidenceError(
            "v20 functional wave requires exactly seven concurrent shards"
            if v19
            else "functional wave requires exactly four shards"
        )
    global_order = {node_id: index for index, node_id in enumerate(full_nodes)}
    assigned: list[str] = []
    for index, raw_shard in enumerate(shards):
        location = f"$contract.functional_wave.manifest.shards[{index}]"
        shard = _exact_keys(
            raw_shard,
            {"node_count", "node_ids", "node_ids_sha256", "shard_id"},
            location,
        )
        expected_id = expected_shard_ids[index]
        if shard["shard_id"] != expected_id:
            raise EvidenceError(
                f"functional shard order must be {'/'.join(FUNCTIONAL_V19_SHARD_IDS) if v19 else 'P01-P04'}: {location}"
            )
        nodes = shard["node_ids"]
        if (
            not isinstance(nodes, list)
            or not nodes
            or len(set(nodes)) != len(nodes)
            or not all(isinstance(item, str) and item in global_order for item in nodes)
        ):
            raise EvidenceError(f"{location}.node_ids is malformed")
        if _require_int(shard["node_count"], f"{location}.node_count", minimum=1) != len(nodes):
            raise EvidenceError(f"{location}.node_count mismatch")
        if shard["node_ids_sha256"] != _functional_list_hash(nodes):
            raise EvidenceError(f"{location}.node_ids_sha256 mismatch")
        if not v19 and {
            "node_count": shard["node_count"],
            "node_ids_sha256": shard["node_ids_sha256"],
        } != FUNCTIONAL_SHARD_AUTHORITIES[expected_id]:
            raise EvidenceError(f"{location} differs from the frozen shard authority")
        if v19 and {
            "node_count": shard["node_count"],
            "node_ids_sha256": shard["node_ids_sha256"],
        } != FUNCTIONAL_V19_SHARD_AUTHORITIES[expected_id]:
            raise EvidenceError(f"{location} differs from the frozen v19 authority")
        assigned.extend(nodes)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(full_nodes):
        raise EvidenceError("functional shards are not a disjoint complete node partition")
    if v19:
        nodes_by_shard = {
            shard["shard_id"]: list(shard["node_ids"]) for shard in shards
        }
        hot_module = "tests/test_fe_solver_nonlinear_static.py"
        hot_nodes = [
            node_id
            for node_id in full_nodes
            if node_id.partition("::")[0] == hot_module
        ]
        expected_h01 = hot_nodes + list(FUNCTIONAL_V19_SPLIT_NODE_IDS["H01"])
        if nodes_by_shard["H01"] != expected_h01 or len(hot_nodes) != 10:
            raise EvidenceError(
                "v20 H01 must own the complete nonlinear module only"
            )
        module_owner: dict[str, str] = {}
        observed_split_nodes: list[str] = []
        for shard_id in expected_shard_ids:
            shard_nodes = nodes_by_shard[shard_id]
            shard_modules = list(
                dict.fromkeys(
                    node_id.partition("::")[0]
                    for node_id in shard_nodes
                )
            )
            if {
                "module_count": len(shard_modules),
                "modules_sha256": _functional_list_hash(shard_modules),
            } != FUNCTIONAL_V19_SHARD_MODULE_AUTHORITIES[shard_id]:
                raise EvidenceError(
                    f"v19 {shard_id} module set differs from frozen authority"
                )
            full_modules = tuple(
                module
                for module in shard_modules
                if module != FUNCTIONAL_V19_SPLIT_MODULE
            )
            if full_modules != FUNCTIONAL_V19_FULL_MODULES[shard_id]:
                raise EvidenceError(
                    f"v19 {shard_id} full-module list differs from authority"
                )
            split_nodes = tuple(
                node_id
                for node_id in shard_nodes
                if node_id.partition("::")[0] == FUNCTIONAL_V19_SPLIT_MODULE
            )
            if split_nodes != FUNCTIONAL_V19_SPLIT_NODE_IDS[shard_id]:
                raise EvidenceError(
                    f"v19 {shard_id} exact split-module extent differs from authority"
                )
            selectors, selector_summary = _functional_shard_selectors(
                shard_nodes, full_nodes
            )
            if (
                tuple(selector for selector in selectors if "::" not in selector)
                != FUNCTIONAL_V19_FULL_MODULES[shard_id]
                or tuple(selector for selector in selectors if "::" in selector)
                != FUNCTIONAL_V19_SPLIT_NODE_IDS[shard_id]
                or selector_summary["full_module_count"] != len(full_modules)
                or selector_summary["exact_node_count"] != len(split_nodes)
            ):
                raise EvidenceError(
                    f"v19 {shard_id} selector sequence differs from authority"
                )
            observed_split_nodes.extend(split_nodes)
            for node_id in shard_nodes:
                module = node_id.partition("::")[0]
                if module == FUNCTIONAL_V19_SPLIT_MODULE:
                    continue
                owner = module_owner.setdefault(module, shard_id)
                if owner != shard_id:
                    raise EvidenceError(
                        "v19 functional shards split an unapproved test module"
                    )
        full_split_nodes = [
            node_id
            for node_id in full_nodes
            if node_id.partition("::")[0] == FUNCTIONAL_V19_SPLIT_MODULE
        ]
        if (
            len(observed_split_nodes) != len(set(observed_split_nodes))
            or set(observed_split_nodes) != set(full_split_nodes)
            or len(full_split_nodes) != 9
        ):
            raise EvidenceError(
                "v22 exact split-module assignments must partition all nine burn-in nodes"
            )

    execution = _exact_keys(
        wave["execution"],
        (
            {
                "artifact_routing",
                "automatic_retry",
                "command_watchdog_cutoff_seconds",
                "emergency_publication_cutoff_seconds",
                "environment",
                "execution_mode",
                "inner_wave_policy",
                "max_concurrent_workers",
                "normal_evidence_cutoff_seconds",
                "numerical_library_threads",
                "preparation_cutoff_seconds",
                "raw_observability",
                "ready_barrier_cutoff_seconds",
                "selector_safety",
                "shard_execution_cutoff_seconds",
                "shard_execution_ids",
                "source_mode",
                "source_status_must_match",
                "unproven_tree_action",
                "watchdog_termination_start_cutoff_seconds",
            }
            if v19
            else {
                "artifact_routing",
                "automatic_retry",
                "environment",
                "internal_deadline_seconds",
                "max_workers",
                "numerical_library_threads",
                "raw_observability",
                "selector_safety",
                "source_mode",
                "source_status_must_match",
                "unproven_tree_action",
            }
        ),
        "$contract.functional_wave.execution",
    )
    if v19:
        for key in (
            "command_watchdog_cutoff_seconds",
            "emergency_publication_cutoff_seconds",
            "max_concurrent_workers",
            "normal_evidence_cutoff_seconds",
            "preparation_cutoff_seconds",
            "ready_barrier_cutoff_seconds",
            "shard_execution_cutoff_seconds",
            "watchdog_termination_start_cutoff_seconds",
        ):
            _require_int(
                execution[key],
                f"$contract.functional_wave.execution.{key}",
                minimum=1,
            )
        for key, expected in FUNCTIONAL_V19_STAGE_POLICY.items():
            if execution[key] != expected:
                raise EvidenceError(
                    f"v19 functional concurrent policy mismatch: {key}"
                )
        ordered_cutoffs = [
            execution["preparation_cutoff_seconds"],
            execution["ready_barrier_cutoff_seconds"],
            execution["inner_wave_policy"]["result_cutoff_seconds"],
            execution["inner_wave_policy"]["cleanup_cutoff_seconds"],
            execution["shard_execution_cutoff_seconds"],
            execution["normal_evidence_cutoff_seconds"],
            execution["emergency_publication_cutoff_seconds"],
            execution["watchdog_termination_start_cutoff_seconds"],
            execution["command_watchdog_cutoff_seconds"],
        ]
        if ordered_cutoffs != sorted(set(ordered_cutoffs)):
            raise EvidenceError(
                "v20 absolute functional cutoffs must be strictly ordered"
            )
    elif _require_int(
        execution["max_workers"],
        "$contract.functional_wave.execution.max_workers",
        minimum=1,
    ) != 4:
        raise EvidenceError("functional max-workers authority must be four")
    if _require_int(
        execution["numerical_library_threads"],
        "$contract.functional_wave.execution.numerical_library_threads",
        minimum=1,
    ) != 1:
        raise EvidenceError("functional numerical-library thread count must be one")
    if _require_bool(
        execution["automatic_retry"],
        "$contract.functional_wave.execution.automatic_retry",
    ):
        raise EvidenceError("functional automatic retry is forbidden")
    if not v19 and _require_int(
        execution["internal_deadline_seconds"],
        "$contract.functional_wave.execution.internal_deadline_seconds",
        minimum=1,
    ) != 780:
        raise EvidenceError("functional internal deadline must be 780 seconds")
    environment = _exact_keys(
        execution["environment"],
        {"NUMBA_NUM_THREADS", "scope"},
        "$contract.functional_wave.execution.environment",
    )
    if environment != {
        "NUMBA_NUM_THREADS": "1",
        "scope": "FUNCTIONAL_SHARDS_ONLY",
    }:
        raise EvidenceError("functional shard-only environment authority mismatch")
    selector_safety = _exact_keys(
        execution["selector_safety"],
        {
            "extra_nodes",
            "full_module_selector",
            "missing_nodes",
            "split_module_selector",
        },
        "$contract.functional_wave.execution.selector_safety",
    )
    if selector_safety != {
        "extra_nodes": "REJECT",
        "full_module_selector": "ONLY_WHEN_ALL_COLLECTED_MODULE_NODES_ARE_SHARD_OWNED",
        "missing_nodes": "REJECT",
        "split_module_selector": "EXACT_NODE_IDS_ONLY",
    }:
        raise EvidenceError("functional selector-safety authority mismatch")
    raw_observability = _exact_keys(
        execution["raw_observability"],
        {"canonical_timings", "lifecycle_progress", "pytest_durations"},
        "$contract.functional_wave.execution.raw_observability",
    )
    if raw_observability != {
        "canonical_timings": False,
        "lifecycle_progress": True,
        "pytest_durations": True,
    }:
        raise EvidenceError("functional raw-observability authority mismatch")
    if execution["source_mode"] != "GIT_ARCHIVE_HEAD":
        raise EvidenceError("functional source mode must be Git archive HEAD")
    if not _require_bool(
        execution["source_status_must_match"],
        "$contract.functional_wave.execution.source_status_must_match",
    ):
        raise EvidenceError("functional source pre/post status equality must be required")
    if (
        execution["unproven_tree_action"]
        != "WAIT_FOR_OUTER_RESOURCE_TREE_TERMINATION"
    ):
        raise EvidenceError("functional unproven-tree action mismatch")
    routing = _exact_keys(
        execution["artifact_routing"],
        {
            "aggregate_filename",
            "archive_filename",
            "directory_name",
            "raw_diagnostics_filename",
            "shard_directory_prefix",
            "source_directory_name",
        },
        "$contract.functional_wave.execution.artifact_routing",
    )
    for key in (
        "aggregate_filename",
        "archive_filename",
        "directory_name",
        "raw_diagnostics_filename",
        "source_directory_name",
    ):
        _functional_route_name(
            routing[key], f"$contract.functional_wave.execution.artifact_routing.{key}"
        )
    _functional_route_name(
        routing["shard_directory_prefix"],
        "$contract.functional_wave.execution.artifact_routing.shard_directory_prefix",
        prefix=True,
    )
    if routing["archive_filename"] != source["archive_filename"]:
        raise EvidenceError("functional source/routing archive filenames differ")
    if routing != FUNCTIONAL_WAVE_ROUTING:
        raise EvidenceError("functional-wave artifact routing mismatch")
    reserved_names = {
        routing["aggregate_filename"],
        routing["archive_filename"],
        routing["raw_diagnostics_filename"],
        source["file_graph_filename"],
    }
    if len(reserved_names) != 4:
        raise EvidenceError("functional routed artifact names must be distinct")

    aggregate = _exact_keys(
        wave["aggregate"],
        {"blocked_terminal", "schema", "success_terminal"},
        "$contract.functional_wave.aggregate",
    )
    for key in ("schema", "success_terminal", "blocked_terminal"):
        _functional_string(aggregate[key], f"$contract.functional_wave.aggregate.{key}")
    if aggregate != {
        "blocked_terminal": "BLOCKED_E4_PL_S3_Q4_FUNCTIONAL_WAVE",
        "schema": (
            FUNCTIONAL_WAVE_V19_AGGREGATE_SCHEMA
            if v19
            else FUNCTIONAL_WAVE_AGGREGATE_SCHEMA
        ),
        "success_terminal": "PASS_E4_PL_S3_Q4_FUNCTIONAL_WAVE",
    }:
        raise EvidenceError("functional aggregate authority mismatch")
    _validate_functional_timeout_policy(contract, schema=wave["schema"])
    return wave


def _functional_source_status(
    repository: Path, *, absolute_deadline: float | None = None
) -> dict[str, Any]:
    command = [
        *_git_command_prefix(repository),
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
    ]
    timeout = (
        None
        if absolute_deadline is None
        else _functional_remaining_seconds(
            absolute_deadline, "functional source-status subprocess"
        )
    )
    timeout_options: dict[str, Any] = {}
    if timeout is not None:
        timeout_options["timeout"] = timeout
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=_sanitized_git_environment(),
            **timeout_options,
        )
    except subprocess.TimeoutExpired as exc:
        raise EvidenceError("functional source status timed out") from exc
    if completed.returncode:
        raise EvidenceError("functional source status command failed")
    return {
        "bytes": len(completed.stdout),
        "sha256": _sha256_bytes(completed.stdout),
    }


def _functional_file_graph(
    root: Path, *, absolute_deadline: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for item in root.rglob("*"):
        _functional_deadline_check(absolute_deadline, "source-graph enumeration")
        if item.is_file():
            paths.append(item)
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        _functional_deadline_check(absolute_deadline, "source-graph hashing")
        if path.is_symlink() or _is_reparse_point(path):
            raise EvidenceError(f"functional source graph contains a reparse file: {path}")
        record = _functional_bounded_hash_record(
            path, absolute_deadline, "source-graph file hashing"
        )
        rows.append(
            {
                "bytes": record["bytes"],
                "path": path.relative_to(root).as_posix(),
                "sha256": record["sha256"],
            }
        )
    if not rows:
        raise EvidenceError("functional Git archive extracted no files")
    return rows, {"files": len(rows), "sha256": _sha256_bytes(canonical_json_bytes(rows))}


def _extract_functional_tar(
    archive: Path, destination: Path, *, absolute_deadline: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _functional_deadline_check(absolute_deadline, "tar extraction initialization")
    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise EvidenceError(f"functional source sandbox already exists: {destination}") from exc
    seen: set[str] = set()
    with tarfile.open(archive, mode="r:") as source:
        for member in source:
            _functional_deadline_check(absolute_deadline, "tar member extraction")
            name = member.name
            relative = PurePosixPath(name)
            if (
                not name
                or "\\" in name
                or relative.is_absolute()
                or any(part in {"", ".", ".."} for part in relative.parts)
                or name in seen
            ):
                raise EvidenceError(f"unsafe or duplicate Git tar member: {name!r}")
            seen.add(name)
            if member.issym() or member.islnk():
                raise EvidenceError(f"functional Git tar link is forbidden: {name}")
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise EvidenceError(f"unsupported functional Git tar member: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = source.extractfile(member)
            if extracted is None:
                raise EvidenceError(f"functional Git tar member has no data: {name}")
            try:
                with extracted, target.open("xb") as output_stream:
                    while True:
                        _functional_deadline_check(
                            absolute_deadline, "tar member copy"
                        )
                        chunk = extracted.read(1024 * 1024)
                        if not chunk:
                            break
                        output_stream.write(chunk)
            except FileExistsError as exc:
                raise EvidenceError(f"duplicate functional Git tar target: {name}") from exc
    _functional_deadline_check(absolute_deadline, "tar extraction completion")
    return _functional_file_graph(
        destination, absolute_deadline=absolute_deadline
    )


def _create_functional_archive(
    repository: Path,
    archive: Path,
    *,
    absolute_deadline: float,
    environment: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    _functional_deadline_check(absolute_deadline, "archive identity preparation")
    identity = _tracked_head_identity(
        repository, absolute_deadline=absolute_deadline
    )
    attributes_path = Path(
        _git(
            repository,
            "rev-parse",
            "--git-path",
            "info/attributes",
            timeout_seconds=_functional_remaining_seconds(
                absolute_deadline, "archive attribute-path query"
            ),
        )
    )
    if not attributes_path.is_absolute():
        attributes_path = repository / attributes_path
    if attributes_path.exists() or attributes_path.is_symlink() or _is_reparse_point(attributes_path):
        raise EvidenceError("Git info attributes are forbidden during functional archiving")
    if os.path.lexists(archive):
        raise EvidenceError("functional source archive output is not exclusive")
    _functional_deadline_check(absolute_deadline, "Git archive subprocess")
    process = _run_captured(
        [
            *_git_command_prefix(repository),
            "archive",
            "--format=tar",
            f"--output={archive}",
            "HEAD",
        ],
        cwd=archive.parent,
        env=_sanitized_git_environment(environment),
        timeout_seconds=_functional_remaining_seconds(
            absolute_deadline, "Git archive subprocess"
        ),
    )
    _functional_deadline_check(absolute_deadline, "Git archive validation")
    if not archive.is_file() or archive.is_symlink() or _is_reparse_point(archive):
        raise EvidenceError("functional Git archive is not a regular file")
    if (
        _tracked_head_identity(repository, absolute_deadline=absolute_deadline)
        != identity
    ):
        raise EvidenceError("functional source identity changed during archiving")
    return identity, process


_FUNCTIONAL_PLUGIN_STATE: dict[str, Any] | None = None


def _functional_progress_event(path: Path, event: str, **values: Any) -> None:
    """Append one raw, noncanonical lifecycle/timing observation."""

    payload = {
        "event": event,
        "recorded_at": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        **values,
    }
    with path.open("ab") as stream:
        stream.write(canonical_json_bytes(payload))
        stream.flush()


def _functional_plugin_progress(
    state: Mapping[str, Any], event: str, **values: Any
) -> None:
    payload = {
        "event": event,
        "recorded_at": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        **values,
    }
    stream = state["progress_stream"]
    stream.write(canonical_json_bytes(payload))
    stream.flush()


def _validate_functional_ready_record(
    value: Any,
    *,
    shard_id: str | None = None,
    pid: int | None = None,
    barrier_id: str | None = None,
    ready_nonce: str | None = None,
) -> dict[str, Any]:
    record = _exact_keys(
        value,
        {
            "barrier_id",
            "node_count",
            "node_ids_sha256",
            "pid",
            "ready_nonce",
            "schema",
            "shard_id",
        },
        "$functional_ready",
    )
    if record["schema"] != FUNCTIONAL_READY_SCHEMA:
        raise EvidenceError("functional READY schema mismatch")
    _require_match(record["barrier_id"], SHA256_RE, "$functional_ready.barrier_id")
    _require_match(record["ready_nonce"], SHA256_RE, "$functional_ready.ready_nonce")
    _require_match(
        record["node_ids_sha256"], SHA256_RE, "$functional_ready.node_ids_sha256"
    )
    _require_int(record["node_count"], "$functional_ready.node_count", minimum=1)
    _require_int(record["pid"], "$functional_ready.pid", minimum=1)
    if not isinstance(record["shard_id"], str) or not record["shard_id"]:
        raise EvidenceError("functional READY shard identity is invalid")
    expected = {
        "barrier_id": barrier_id,
        "pid": pid,
        "ready_nonce": ready_nonce,
        "shard_id": shard_id,
    }
    for key, expected_value in expected.items():
        if expected_value is not None and record[key] != expected_value:
            raise EvidenceError(f"functional READY {key} mismatch")
    return dict(record)


def _validate_functional_release_record(value: Any) -> dict[str, Any]:
    release = _exact_keys(
        value,
        {"barrier_id", "liveness", "ready_records", "schema"},
        "$functional_release",
    )
    if release["schema"] != FUNCTIONAL_RELEASE_SCHEMA:
        raise EvidenceError("functional release schema mismatch")
    _require_match(
        release["barrier_id"], SHA256_RE, "$functional_release.barrier_id"
    )
    if release["liveness"] != "ALL_REGISTERED_CHILDREN_LIVE":
        raise EvidenceError("functional release liveness disposition mismatch")
    rows = release["ready_records"]
    if not isinstance(rows, list) or len(rows) != len(FUNCTIONAL_V19_SHARD_IDS):
        raise EvidenceError("functional release READY extent mismatch")
    validated: list[dict[str, Any]] = []
    for index, (row, expected_id) in enumerate(
        zip(rows, FUNCTIONAL_V19_SHARD_IDS, strict=True)
    ):
        row = _exact_keys(
            row,
            {"ready", "shard_id"},
            f"$functional_release.ready_records[{index}]",
        )
        if row["shard_id"] != expected_id:
            raise EvidenceError("functional release READY order mismatch")
        ready = _exact_keys(
            row["ready"],
            {"bytes", "sha256"},
            f"$functional_release.ready_records[{index}].ready",
        )
        _validate_hash_record(
            ready, f"$functional_release.ready_records[{index}].ready"
        )
        validated.append({"ready": ready, "shard_id": expected_id})
    return {
        "barrier_id": release["barrier_id"],
        "liveness": release["liveness"],
        "ready_records": validated,
        "schema": release["schema"],
    }


def _functional_publish_ready_and_wait(state: dict[str, Any]) -> None:
    """Publish one fsynced READY record and wait for the common release."""

    ready = {
        "barrier_id": state["barrier_id"],
        "node_count": len(state["expected"]),
        "node_ids_sha256": _functional_list_hash(state["expected"]),
        "pid": os.getpid(),
        "ready_nonce": state["ready_nonce"],
        "schema": FUNCTIONAL_READY_SCHEMA,
        "shard_id": state["shard_id"],
    }
    ready_raw = canonical_json_bytes(ready)
    ready_path = state["ready_path"]
    try:
        with ready_path.open("xb") as stream:
            stream.write(ready_raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError("functional READY path is not exclusive") from exc

    ready_record = {"bytes": len(ready_raw), "sha256": _sha256_bytes(ready_raw)}
    release_path = state["release_path"]
    deadline = state["barrier_deadline"]
    while True:
        if release_path.exists() or release_path.is_symlink():
            if (
                not release_path.is_file()
                or release_path.is_symlink()
                or _is_reparse_point(release_path)
            ):
                raise RuntimeError("functional release is not a regular file")
            release_raw = release_path.read_bytes()
            # An exclusive writer makes the name visible before fsync.  Treat a
            # non-newline-terminated prefix as in-flight rather than malformed.
            if release_raw.endswith(b"\n"):
                release = _validate_functional_release_record(
                    strict_json_loads(release_raw)
                )
                if release_raw != canonical_json_bytes(release):
                    raise RuntimeError("functional release is noncanonical")
                if release["barrier_id"] != state["barrier_id"]:
                    raise RuntimeError("functional release barrier identity mismatch")
                own_rows = [
                    row
                    for row in release["ready_records"]
                    if row["shard_id"] == state["shard_id"]
                ]
                if len(own_rows) != 1 or own_rows[0]["ready"] != ready_record:
                    raise RuntimeError("functional release omits this READY identity")
                state["barrier_released"] = True
                return
        if time.monotonic() >= deadline:
            raise RuntimeError("functional READY barrier deadline expired")
        time.sleep(0.01)


def _functional_plugin_state() -> dict[str, Any] | None:
    global _FUNCTIONAL_PLUGIN_STATE
    ci_capture_path = os.environ.get(_CI_CAPTURE_ENV)
    ci_expected_path = os.environ.get(_CI_EXPECTED_ENV)
    ci_shard_id = os.environ.get(_CI_SHARD_ENV)
    result_path = os.environ.get(_FUNCTIONAL_RESULT_ENV)
    expected_path = os.environ.get(_FUNCTIONAL_EXPECTED_ENV)
    progress_path = os.environ.get(_FUNCTIONAL_PROGRESS_ENV)
    shard_id = os.environ.get(_FUNCTIONAL_SHARD_ENV)
    ready_path = os.environ.get(_FUNCTIONAL_READY_ENV)
    release_path = os.environ.get(_FUNCTIONAL_RELEASE_ENV)
    barrier_id = os.environ.get(_FUNCTIONAL_BARRIER_ID_ENV)
    ready_nonce = os.environ.get(_FUNCTIONAL_READY_NONCE_ENV)
    barrier_deadline = os.environ.get(_FUNCTIONAL_BARRIER_DEADLINE_ENV)
    barrier_values = (
        ready_path,
        release_path,
        barrier_id,
        ready_nonce,
        barrier_deadline,
    )
    if ci_capture_path:
        if any(
            value
            for value in (
                ci_expected_path,
                ci_shard_id,
                result_path,
                expected_path,
                progress_path,
                shard_id,
                *barrier_values,
            )
        ):
            raise RuntimeError("CI collection-capture environment is ambiguous")
        if _FUNCTIONAL_PLUGIN_STATE is None:
            _FUNCTIONAL_PLUGIN_STATE = {
                "capture_path": Path(ci_capture_path),
                "mode": "ci-capture",
            }
        return _FUNCTIONAL_PLUGIN_STATE
    if ci_expected_path or ci_shard_id:
        if (
            not ci_expected_path
            or not ci_shard_id
            or any(
                value
                for value in (
                    result_path,
                    expected_path,
                    progress_path,
                    shard_id,
                    *barrier_values,
                )
            )
        ):
            raise RuntimeError("CI expected-node environment is incomplete or ambiguous")
        if _FUNCTIONAL_PLUGIN_STATE is None:
            expected = strict_json_load(Path(ci_expected_path))
            if (
                not isinstance(expected, list)
                or not expected
                or len(set(expected)) != len(expected)
                or not all(isinstance(item, str) and item for item in expected)
            ):
                raise RuntimeError("CI pytest expected-node file is malformed")
            authority = CI_SHARD_NODE_AUTHORITIES.get(ci_shard_id)
            if authority != {
                "node_count": len(expected),
                "node_ids_sha256": _functional_list_hash(expected),
            }:
                raise RuntimeError("CI pytest expected-node authority mismatch")
            _FUNCTIONAL_PLUGIN_STATE = {
                "collection_matches": False,
                "expected": expected,
                "expected_set": set(expected),
                "mode": "ci-guard",
                "shard_id": ci_shard_id,
            }
        return _FUNCTIONAL_PLUGIN_STATE
    functional_values = (result_path, expected_path, progress_path, shard_id)
    if not any((*functional_values, *barrier_values)):
        return None
    if not all(functional_values) or (any(barrier_values) and not all(barrier_values)):
        raise RuntimeError("functional pytest plugin environment is incomplete")
    if _FUNCTIONAL_PLUGIN_STATE is None:
        expected = strict_json_load(Path(expected_path))
        if (
            not isinstance(expected, list)
            or not expected
            or len(set(expected)) != len(expected)
            or not all(isinstance(item, str) and item for item in expected)
        ):
            raise RuntimeError("functional pytest expected-node file is malformed")
        state: dict[str, Any] = {
            "collection_matches": False,
            "expected": expected,
            "expected_set": set(expected),
            "durations": {},
            "outcomes": {},
            "progress_path": Path(progress_path),
            "progress_stream": Path(progress_path).open("ab"),
            "result_path": Path(result_path),
            "shard_id": shard_id,
            "mode": "functional",
        }
        if all(barrier_values):
            try:
                parsed_deadline = float(str(barrier_deadline))
            except ValueError as exc:
                raise RuntimeError("functional barrier deadline is malformed") from exc
            if not math.isfinite(parsed_deadline) or parsed_deadline <= 0.0:
                raise RuntimeError("functional barrier deadline is invalid")
            state.update(
                {
                    "barrier_deadline": parsed_deadline,
                    "barrier_id": _require_match(
                        barrier_id, SHA256_RE, "$functional_environment.barrier_id"
                    ),
                    "barrier_released": False,
                    "ready_nonce": _require_match(
                        ready_nonce, SHA256_RE, "$functional_environment.ready_nonce"
                    ),
                    "ready_path": Path(str(ready_path)),
                    "release_path": Path(str(release_path)),
                }
            )
        _FUNCTIONAL_PLUGIN_STATE = state
    return _FUNCTIONAL_PLUGIN_STATE


def pytest_sessionstart(session) -> None:  # pragma: no cover - child plugin
    state = _functional_plugin_state()
    if (
        state is not None
        and state.get("mode") == "functional"
        and "barrier_id" in state
    ):
        _functional_publish_ready_and_wait(state)


def pytest_collection_modifyitems(session, config, items) -> None:  # pragma: no cover - child plugin
    state = _functional_plugin_state()
    if state is None:
        return
    mode = state.get("mode", "functional")
    if mode == "ci-capture":
        node_ids = [item.nodeid for item in items]
        if len(node_ids) != len(set(node_ids)):
            raise RuntimeError("CI collection contains duplicate node IDs")
        capture_path = state["capture_path"]
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with capture_path.open("xb") as stream:
                stream.write(canonical_json_bytes(node_ids))
        except FileExistsError as exc:
            raise RuntimeError("CI collection capture path is not exclusive") from exc
        return
    by_id: dict[str, Any] = {}
    duplicates: set[str] = set()
    for item in items:
        if item.nodeid in by_id:
            duplicates.add(item.nodeid)
        by_id[item.nodeid] = item
    expected = state["expected"]
    expected_ids = state["expected_set"]
    missing = [node_id for node_id in expected if node_id not in by_id]
    unexpected = [node_id for node_id in by_id if node_id not in expected_ids]
    state["collection_matches"] = (
        not missing
        and not unexpected
        and not duplicates
        and len(items) == len(expected)
    )
    selected = [by_id[node_id] for node_id in expected if node_id in by_id]
    deselected = [item for item in items if item.nodeid not in expected_ids]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
    if mode == "functional":
        _functional_plugin_progress(
            state,
            "COLLECTION_FINISHED",
            collected_count=len(items) + len(deselected),
            collection_matches=state["collection_matches"],
            duplicate_count=len(duplicates),
            expected_count=len(expected),
            missing_count=len(missing),
            selected_count=len(selected),
            shard_id=state["shard_id"],
            unexpected_count=len(unexpected),
        )


def pytest_runtest_logreport(report) -> None:  # pragma: no cover - child plugin
    state = _functional_plugin_state()
    if (
        state is None
        or state.get("mode", "functional") != "functional"
        or report.nodeid not in state["expected_set"]
    ):
        return
    durations = state["durations"]
    durations[report.nodeid] = durations.get(report.nodeid, 0.0) + float(
        report.duration
    )
    outcomes = state["outcomes"]
    was_xfail = bool(getattr(report, "wasxfail", False))
    if report.when == "setup":
        if report.failed:
            outcomes[report.nodeid] = "ERROR"
        elif report.skipped:
            outcomes[report.nodeid] = "XFAIL" if was_xfail else "SKIPPED"
    elif report.when == "call":
        if report.passed:
            outcomes[report.nodeid] = "XPASS" if was_xfail else "PASSED"
        elif report.skipped:
            outcomes[report.nodeid] = "XFAIL" if was_xfail else "SKIPPED"
        else:
            outcomes[report.nodeid] = "XFAIL" if was_xfail else "FAILED"
    elif report.when == "teardown" and report.failed:
        outcomes[report.nodeid] = "ERROR"
    if report.when == "teardown":
        _functional_plugin_progress(
            state,
            "PYTEST_ITEM_FINISHED",
            duration_seconds=durations[report.nodeid],
            node_id=report.nodeid,
            outcome=outcomes.get(report.nodeid, "NOT_RUN"),
            shard_id=state["shard_id"],
        )


def pytest_sessionfinish(session, exitstatus) -> None:  # pragma: no cover - child plugin
    state = _functional_plugin_state()
    if state is None:
        return
    mode = state.get("mode", "functional")
    if mode == "ci-capture":
        return
    if mode == "ci-guard":
        if not state["collection_matches"]:
            session.exitstatus = 3
        return
    _functional_plugin_progress(
        state,
        "SESSION_FINISHED",
        exit_code=int(exitstatus),
        node_count=len(state["expected"]),
        shard_id=state["shard_id"],
    )
    payload = {
        "collection_matches": state["collection_matches"],
        "exit_code": int(exitstatus),
        "nodes": [
            {
                "node_id": node_id,
                "outcome": state["outcomes"].get(node_id, "NOT_RUN"),
            }
            for node_id in state["expected"]
        ],
        "schema": FUNCTIONAL_SHARD_SCHEMA,
        "shard_id": state["shard_id"],
    }
    result_path = state["result_path"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with result_path.open("xb") as stream:
            stream.write(canonical_json_bytes(payload))
    except FileExistsError as exc:
        raise RuntimeError("functional shard result path is not exclusive") from exc
    finally:
        state["progress_stream"].close()


def _validate_functional_shard_result(
    value: Any,
    *,
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    value = _exact_keys(
        value,
        {"collection_matches", "exit_code", "nodes", "schema", "shard_id"},
        "$functional_shard_result",
    )
    if value["schema"] != FUNCTIONAL_SHARD_SCHEMA or value["shard_id"] != shard["shard_id"]:
        raise EvidenceError("functional shard result identity mismatch")
    _require_bool(value["collection_matches"], "$functional_shard_result.collection_matches")
    _require_int(value["exit_code"], "$functional_shard_result.exit_code")
    nodes = value["nodes"]
    if not isinstance(nodes, list) or len(nodes) != shard["node_count"]:
        raise EvidenceError("functional shard result node count mismatch")
    expected = shard["node_ids"]
    allowed = {"ERROR", "FAILED", "NOT_RUN", "PASSED", "SKIPPED", "XFAIL", "XPASS"}
    for index, row in enumerate(nodes):
        row = _exact_keys(row, {"node_id", "outcome"}, f"$functional_shard_result.nodes[{index}]")
        if row["node_id"] != expected[index] or row["outcome"] not in allowed:
            raise EvidenceError("functional shard result node identity/outcome mismatch")
    return value


def _functional_taskkill_authority(policy: Mapping[str, Any]) -> Path:
    authority = policy["windows_termination"]
    path = Path(authority["path"])
    if (
        not path.is_file()
        or path.is_symlink()
        or _is_reparse_point(path)
        or file_hash_record(path)
        != {"bytes": authority["bytes"], "sha256": authority["sha256"]}
    ):
        raise EvidenceError("taskkill executable differs from frozen authority")
    return path


def _terminate_functional_process_tree(
    worker: subprocess.Popen[bytes], policy: Mapping[str, Any]
) -> dict[str, Any]:
    grace = int(policy["termination_grace_seconds"])
    termination_deadline = time.monotonic() + grace

    def remaining_termination_time() -> float:
        return max(0.001, termination_deadline - time.monotonic())

    if os.name != "nt":
        try:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=remaining_termination_time())
        except (OSError, subprocess.TimeoutExpired) as exc:
            raw = str(exc).encode("utf-8", errors="replace")
            return {
                "bytes": len(raw),
                "returncode": 259,
                "sha256": _sha256_bytes(raw),
            }
        return {"bytes": 0, "returncode": 0, "sha256": _sha256_bytes(b"")}
    taskkill = _functional_taskkill_authority(policy)
    arguments = [str(worker.pid) if item == "{pid}" else item for item in policy["windows_termination"]["arguments"]]
    try:
        completed = subprocess.run(
            [str(taskkill), *arguments],
            capture_output=True,
            check=False,
            timeout=remaining_termination_time(),
        )
        raw = completed.stdout + completed.stderr
        result = {
            "bytes": len(raw),
            "returncode": int(completed.returncode),
            "sha256": _sha256_bytes(raw),
        }
    except subprocess.TimeoutExpired as exc:
        raw = (exc.stdout or b"") + (exc.stderr or b"")
        result = {
            "bytes": len(raw),
            "returncode": 258,
            "sha256": _sha256_bytes(raw),
        }
    except OSError as exc:
        raw = str(exc).encode("utf-8", errors="replace")
        result = {
            "bytes": len(raw),
            "returncode": 257,
            "sha256": _sha256_bytes(raw),
        }
    try:
        worker.wait(timeout=remaining_termination_time())
    except subprocess.TimeoutExpired:
        try:
            worker.kill()
            worker.wait(timeout=remaining_termination_time())
        except (OSError, subprocess.TimeoutExpired):
            pass
        result["returncode"] = 259
    return result


def _launch_functional_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout: Any,
    stderr: Any,
) -> tuple[subprocess.Popen[bytes], int | None]:
    """Launch one shard inside its own kill-on-close containment boundary."""

    if os.name != "nt":
        return (
            subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            ),
            None,
        )

    import ctypes
    from ctypes import wintypes

    job_handle = _create_gate_job()
    assert job_handle is not None
    worker: subprocess.Popen[bytes] | None = None
    assigned = False
    try:
        worker = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=0x00000004,  # CREATE_SUSPENDED
        )
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        process_handle = int(worker._handle)
        if not kernel32.AssignProcessToJobObject(job_handle, process_handle):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        assigned = True
        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        status = int(ntdll.NtResumeProcess(process_handle))
        if status != 0:
            raise OSError(status, "NtResumeProcess failed")
        return worker, job_handle
    except BaseException:
        if assigned:
            _close_gate_job(job_handle)
        else:
            if worker is not None:
                try:
                    worker.kill()
                    worker.wait(timeout=1.0)
                except BaseException:
                    pass
            _close_gate_job(job_handle)
        raise


def _close_functional_process_tree(
    worker: subprocess.Popen[bytes],
    *,
    child_exit_observed: bool,
    job_handle: int | None,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Close a shard tree on every path and return an explicit proof enum."""

    deadline = time.monotonic() + int(policy["termination_grace_seconds"])

    def remaining() -> float:
        return max(0.001, deadline - time.monotonic())

    error: BaseException | None = None
    if os.name == "nt":
        try:
            if job_handle is None or not _close_gate_job(job_handle):
                raise RuntimeError("functional shard job closure failed")
            try:
                worker.wait(timeout=remaining())
            except BaseException:
                raise
        except BaseException as exc:
            error = exc
    else:
        try:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if worker.poll() is None:
                worker.wait(timeout=remaining())
            while True:
                try:
                    os.killpg(worker.pid, 0)
                except ProcessLookupError:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("functional shard process group remains live")
                time.sleep(0.01)
        except BaseException as exc:
            error = exc

    if error is None:
        return (
            {"bytes": 0, "returncode": 0, "sha256": _sha256_bytes(b"")},
            (
                "CHILD_EXIT_AND_TREE_CLOSED"
                if child_exit_observed
                else "TREE_TERMINATION_PROVEN"
            ),
        )
    raw = repr(error).encode("utf-8", errors="replace")
    return (
        {"bytes": len(raw), "returncode": 260, "sha256": _sha256_bytes(raw)},
        "TREE_TERMINATION_FAILED",
    )


def _functional_wave_has_unproven_tree(shards: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        shard.get("descendant_tree_cleanup") == "TREE_TERMINATION_FAILED"
        or shard["status"] == "TIMEOUT_TREE_TERMINATION_FAILED"
        for shard in shards
    )


def _set_functional_shard_tree_state(shard_id: str, state: str) -> None:
    with _FUNCTIONAL_SHARD_TREE_STATE_LOCK:
        _FUNCTIONAL_SHARD_TREE_STATE[shard_id] = state


def _get_functional_shard_tree_state(shard_id: str) -> str:
    with _FUNCTIONAL_SHARD_TREE_STATE_LOCK:
        return _FUNCTIONAL_SHARD_TREE_STATE.get(shard_id, "NOT_LAUNCHED")


def _new_functional_ready_barrier(
    prepared_by_id: Mapping[str, Mapping[str, Any]], *, deadline: float
) -> dict[str, Any]:
    """Create one in-memory controller for the seven isolated child records."""

    if set(prepared_by_id) != set(FUNCTIONAL_V19_SHARD_IDS):
        raise EvidenceError("functional READY barrier shard extent mismatch")
    return {
        "abort": threading.Event(),
        "barrier_id": secrets.token_hex(32),
        "deadline": deadline,
        "lock": threading.Lock(),
        "ready_nonces": {
            shard_id: secrets.token_hex(32) for shard_id in FUNCTIONAL_V19_SHARD_IDS
        },
        "ready_paths": {
            shard_id: prepared_by_id[shard_id]["shard_root"]
            / "logs"
            / "barrier-ready.json"
            for shard_id in FUNCTIONAL_V19_SHARD_IDS
        },
        # The release is common to every child but remains below an already
        # registered shard directory, so no new cycle-root route is invented.
        "release_path": prepared_by_id[FUNCTIONAL_V19_SHARD_IDS[0]]["shard_root"]
        / "logs"
        / "barrier-release.json",
        "workers": {},
    }


def _publish_functional_ready_release(
    barrier: Mapping[str, Any],
    *,
    futures_by_id: Mapping[str, concurrent.futures.Future[Any]],
    prepared_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate seven live children, then exclusively publish their release."""

    ready_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    deadline = float(barrier["deadline"])
    while len(ready_by_id) != len(FUNCTIONAL_V19_SHARD_IDS):
        _functional_deadline_check(deadline, "functional READY barrier")
        for shard_id in FUNCTIONAL_V19_SHARD_IDS:
            with barrier["lock"]:
                worker = barrier["workers"].get(shard_id)
            if worker is None:
                if futures_by_id[shard_id].done():
                    raise EvidenceError(
                        f"functional {shard_id} exited before registering its child"
                    )
                continue
            if int(worker.pid) <= 0:
                raise EvidenceError("functional READY worker PID is invalid")
            if worker.poll() is not None:
                raise EvidenceError(
                    f"functional {shard_id} exited before common release"
                )
            path = barrier["ready_paths"][shard_id]
            if not path.exists() and not path.is_symlink():
                continue
            if not path.is_file() or path.is_symlink() or _is_reparse_point(path):
                raise EvidenceError("functional READY is not a regular file")
            raw = path.read_bytes()
            if not raw.endswith(b"\n"):
                continue
            record = _validate_functional_ready_record(
                strict_json_loads(raw),
                barrier_id=barrier["barrier_id"],
                pid=int(worker.pid),
                ready_nonce=barrier["ready_nonces"][shard_id],
                shard_id=shard_id,
            )
            authority = prepared_by_id[shard_id]["shard"]
            if (
                record["node_count"] != authority["node_count"]
                or record["node_ids_sha256"] != authority["node_ids_sha256"]
            ):
                raise EvidenceError("functional READY node authority mismatch")
            if raw != canonical_json_bytes(record):
                raise EvidenceError("functional READY is noncanonical")
            ready_by_id[shard_id] = (
                record,
                {"bytes": len(raw), "sha256": _sha256_bytes(raw)},
            )
        if len(ready_by_id) != len(FUNCTIONAL_V19_SHARD_IDS):
            time.sleep(0.01)

    # The lock prevents a launch thread from changing the registration set
    # during this final liveness observation.  Every child is blocked on the
    # as-yet-absent common release at this point.
    with barrier["lock"]:
        workers = dict(barrier["workers"])
        if set(workers) != set(FUNCTIONAL_V19_SHARD_IDS):
            raise EvidenceError("functional READY worker registry is incomplete")
        pids = [int(workers[shard_id].pid) for shard_id in FUNCTIONAL_V19_SHARD_IDS]
        if len(set(pids)) != len(pids) or any(pid <= 0 for pid in pids):
            raise EvidenceError("functional READY workers need distinct positive PIDs")
        if any(workers[shard_id].poll() is not None for shard_id in FUNCTIONAL_V19_SHARD_IDS):
            raise EvidenceError("functional READY worker exited before release")
        release = {
            "barrier_id": barrier["barrier_id"],
            "liveness": "ALL_REGISTERED_CHILDREN_LIVE",
            "ready_records": [
                {"ready": ready_by_id[shard_id][1], "shard_id": shard_id}
                for shard_id in FUNCTIONAL_V19_SHARD_IDS
            ],
            "schema": FUNCTIONAL_RELEASE_SCHEMA,
        }
        release_raw = canonical_json_bytes(release)
        try:
            with barrier["release_path"].open("xb") as stream:
                stream.write(release_raw)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise EvidenceError("functional common release path is not exclusive") from exc
    return release


def _functional_barrier_raw_record(
    prepared: Mapping[str, Any],
    *,
    pid: int | None,
    absolute_deadline: float,
) -> dict[str, Any] | None:
    """Bind the exact READY/release files into raw, noncanonical diagnostics."""

    barrier = prepared.get("barrier")
    if not isinstance(barrier, Mapping):
        return None
    shard_id = prepared["shard"]["shard_id"]

    def observe(
        path: Path, validator: Any, label: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not path.exists() and not path.is_symlink():
            return None, None
        if not path.is_file() or path.is_symlink() or _is_reparse_point(path):
            return None, None
        raw = _functional_bounded_read_bytes(path, absolute_deadline, label)
        artifact = {"bytes": len(raw), "sha256": _sha256_bytes(raw)}
        try:
            record = validator(strict_json_loads(raw))
        except (EvidenceError, OSError, UnicodeError, ValueError):
            return artifact, None
        if raw != canonical_json_bytes(record):
            return artifact, None
        return artifact, record

    ready_artifact, ready_record = observe(
        barrier["ready_paths"][shard_id],
        _validate_functional_ready_record,
        f"{shard_id} READY hashing",
    )
    release_artifact, release_record = observe(
        barrier["release_path"],
        _validate_functional_release_record,
        f"{shard_id} release hashing",
    )
    state = "NOT_LAUNCHED" if pid is None else "NOT_READY"
    if ready_artifact is not None:
        state = "INVALID" if ready_record is None else "READY_NOT_RELEASED"
    if release_artifact is not None:
        state = "INVALID"
        if ready_record is not None and release_record is not None:
            own_rows = [
                row
                for row in release_record["ready_records"]
                if row["shard_id"] == shard_id
            ]
            if (
                len(own_rows) == 1
                and own_rows[0]["ready"] == ready_artifact
                and ready_record["barrier_id"] == barrier["barrier_id"]
                and release_record["barrier_id"] == barrier["barrier_id"]
                and ready_record["shard_id"] == shard_id
                and ready_record["pid"] == pid
                and ready_record["ready_nonce"]
                == barrier["ready_nonces"][shard_id]
            ):
                state = "RELEASED"
    return {
        "barrier_id": barrier["barrier_id"],
        "ready_artifact": ready_artifact,
        "ready_record": ready_record,
        "release_artifact": release_artifact,
        "release_record": release_record,
        "state": state,
    }


def _functional_v19_workers_overlapped(
    raw_by_id: Mapping[str, Mapping[str, Any]],
    canonical_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Prove seven-way overlap from the identity-bound common release."""

    pids: list[int] = []
    nonces: list[str] = []
    ready_artifacts: dict[str, dict[str, Any]] = {}
    release_records: list[dict[str, Any]] = []
    release_artifacts: list[dict[str, Any]] = []
    barrier_ids: list[str] = []
    for shard_id in FUNCTIONAL_V19_SHARD_IDS:
        row = raw_by_id.get(shard_id)
        if not isinstance(row, Mapping) or row.get("attempts") != 1:
            return False
        pid = _require_int(row.get("pid"), f"$raw.{shard_id}.pid", minimum=1)
        barrier = _exact_keys(
            row.get("barrier"),
            {
                "barrier_id",
                "ready_artifact",
                "ready_record",
                "release_artifact",
                "release_record",
                "state",
            },
            f"$raw.{shard_id}.barrier",
        )
        if barrier["state"] != "RELEASED":
            return False
        barrier_id = _require_match(
            barrier["barrier_id"], SHA256_RE, f"$raw.{shard_id}.barrier_id"
        )
        ready_artifact = _exact_keys(
            barrier["ready_artifact"],
            {"bytes", "sha256"},
            f"$raw.{shard_id}.barrier.ready_artifact",
        )
        _validate_hash_record(
            ready_artifact, f"$raw.{shard_id}.barrier.ready_artifact"
        )
        ready_record = _validate_functional_ready_record(
            barrier["ready_record"],
            barrier_id=barrier_id,
            pid=pid,
            shard_id=shard_id,
        )
        if canonical_by_id is not None and (
            ready_record["node_count"] != canonical_by_id[shard_id]["node_count"]
            or ready_record["node_ids_sha256"]
            != canonical_by_id[shard_id]["node_ids_sha256"]
        ):
            raise EvidenceError("functional READY canonical node authority mismatch")
        if ready_artifact != {
            "bytes": len(canonical_json_bytes(ready_record)),
            "sha256": _sha256_bytes(canonical_json_bytes(ready_record)),
        }:
            raise EvidenceError("functional READY artifact/content mismatch")
        release_artifact = _exact_keys(
            barrier["release_artifact"],
            {"bytes", "sha256"},
            f"$raw.{shard_id}.barrier.release_artifact",
        )
        _validate_hash_record(
            release_artifact, f"$raw.{shard_id}.barrier.release_artifact"
        )
        release_record = _validate_functional_release_record(
            barrier["release_record"]
        )
        release_raw = canonical_json_bytes(release_record)
        if release_artifact != {
            "bytes": len(release_raw),
            "sha256": _sha256_bytes(release_raw),
        }:
            raise EvidenceError("functional release artifact/content mismatch")
        pids.append(pid)
        nonces.append(ready_record["ready_nonce"])
        ready_artifacts[shard_id] = dict(ready_artifact)
        release_records.append(release_record)
        release_artifacts.append(dict(release_artifact))
        barrier_ids.append(barrier_id)
    if len(set(pids)) != len(pids):
        raise EvidenceError("v19 workers must have distinct positive process IDs")
    if len(set(nonces)) != len(nonces):
        raise EvidenceError("v19 READY nonces must be distinct")
    if len(set(barrier_ids)) != 1:
        raise EvidenceError("v19 READY barrier identities disagree")
    if any(record != release_records[0] for record in release_records[1:]):
        raise EvidenceError("v19 workers do not bind one common release")
    if any(record != release_artifacts[0] for record in release_artifacts[1:]):
        raise EvidenceError("v19 workers do not bind one release artifact")
    expected_ready = [
        {"ready": ready_artifacts[shard_id], "shard_id": shard_id}
        for shard_id in FUNCTIONAL_V19_SHARD_IDS
    ]
    if (
        release_records[0]["barrier_id"] != barrier_ids[0]
        or release_records[0]["ready_records"] != expected_ready
    ):
        raise EvidenceError("v19 common release does not bind every READY record")
    return True


def _functional_v19_schedule(
    canonical_by_id: Mapping[str, Mapping[str, Any]],
    raw_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the timing-free canonical schedule disposition for a v19 wave.

    ``terminal`` describes every process that was actually launched; ``launch``
    separately records whether the required seven-process extent was complete.
    """

    missing = set(FUNCTIONAL_V19_SHARD_IDS) - set(canonical_by_id)
    if missing:
        raise EvidenceError(
            f"v19 schedule is missing canonical shard rows: {sorted(missing)}"
        )

    def cleanup(shard_id: str) -> str:
        value = canonical_by_id[shard_id].get("descendant_tree_cleanup")
        if value not in FUNCTIONAL_V19_TREE_CLEANUP_STATES:
            raise EvidenceError(f"v19 {shard_id} tree-cleanup state is invalid")
        return str(value)

    cleanup_by_id = {
        shard_id: cleanup(shard_id) for shard_id in FUNCTIONAL_V19_SHARD_IDS
    }

    def launched(shard_id: str) -> bool:
        if raw_by_id is not None:
            row = raw_by_id.get(shard_id)
            if isinstance(row, Mapping):
                attempts = row.get("attempts")
                if type(attempts) is int and attempts in {0, 1}:
                    return attempts == 1
        return cleanup_by_id[shard_id] != "NOT_LAUNCHED"

    launched_by_id = {
        shard_id: launched(shard_id) for shard_id in FUNCTIONAL_V19_SHARD_IDS
    }
    launched_count = sum(launched_by_id.values())
    launch = (
        "ALL_LAUNCHED"
        if launched_count == len(FUNCTIONAL_V19_SHARD_IDS)
        else "NOT_LAUNCHED"
        if launched_count == 0
        else "PARTIALLY_LAUNCHED"
    )
    if any(
        cleanup_by_id[shard_id] == "TREE_TERMINATION_FAILED"
        for shard_id in FUNCTIONAL_V19_SHARD_IDS
    ):
        terminal = "UNPROVEN_DESCENDANT_TREE"
    elif launched_count == 0:
        terminal = "NOT_LAUNCHED"
    else:
        terminal = "ALL_TERMINAL"

    overlap = (
        _functional_v19_workers_overlapped(raw_by_id, canonical_by_id)
        if raw_by_id is not None and launch == "ALL_LAUNCHED"
        else False
    )
    return {
        "all_workers_overlapped": overlap,
        "launch": launch,
        "terminal": terminal,
    }


def _await_outer_resource_tree_termination() -> None:
    """Hold an uncertain child tree inside both enclosing watchdogs.

    The historical v18 gate and the shorter v19 gate both retain an enclosing
    command watchdog plus an independent resource-cycle watchdog.  Returning
    after an inner tree-kill failure could orphan descendants and falsely look
    like a normal resource exit, so this child remains alive only until one of
    those bounded parents terminates the complete process tree.
    """

    while True:
        try:
            time.sleep(1.0)
        except BaseException:
            # Only termination of the outer resource process tree may release
            # this containment hold; an inner signal/exception is not proof.
            continue


def _functional_shard_environment(
    *,
    sandbox: Path,
    shard_root: Path,
    expected_path: Path,
    progress_path: Path,
    result_path: Path,
    shard_id: str,
    barrier: Mapping[str, Any] | None = None,
    inner_wave_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    roots = _local_roots()
    roots["ANYfeSolver"] = sandbox
    metadata = shard_root / "temp" / "metadata"
    _write_source_metadata_overlay(roots, metadata)
    environment = os.environ.copy()
    for inherited_barrier_name in (
        _FUNCTIONAL_BARRIER_DEADLINE_ENV,
        _FUNCTIONAL_BARRIER_ID_ENV,
        _FUNCTIONAL_READY_ENV,
        _FUNCTIONAL_READY_NONCE_ENV,
        _FUNCTIONAL_RELEASE_ENV,
    ):
        environment.pop(inherited_barrier_name, None)
    for name in _S3_Q4_STYLE_WAVE_ENVIRONMENT:
        environment.pop(name, None)
    environment.update(
        {
            "NUMBA_CACHE_DIR": str(shard_root / "numba_cache"),
            "NUMBA_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(shard_root / "python_cache"),
            "PYTEST_ADDOPTS": "-p no:cacheprovider",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TEMP": str(shard_root / "temp"),
            "TMP": str(shard_root / "temp"),
            "TMPDIR": str(shard_root / "temp"),
            _FUNCTIONAL_EXPECTED_ENV: str(expected_path),
            _FUNCTIONAL_PROGRESS_ENV: str(progress_path),
            _FUNCTIONAL_RESULT_ENV: str(result_path),
            _FUNCTIONAL_SHARD_ENV: shard_id,
        }
    )
    if barrier is not None:
        environment.update(
            {
                _FUNCTIONAL_BARRIER_DEADLINE_ENV: repr(float(barrier["deadline"])),
                _FUNCTIONAL_BARRIER_ID_ENV: str(barrier["barrier_id"]),
                _FUNCTIONAL_READY_ENV: str(barrier["ready_paths"][shard_id]),
                _FUNCTIONAL_READY_NONCE_ENV: str(barrier["ready_nonces"][shard_id]),
                _FUNCTIONAL_RELEASE_ENV: str(barrier["release_path"]),
            }
        )
    if inner_wave_environment is not None:
        if not FUNCTIONAL_V20_INNER_WAVE_POLICY["enabled"]:
            raise EvidenceError(
                "functional inner-wave environment is disabled for opt-in burn-in"
            )
        if set(inner_wave_environment) != set(_S3_Q4_STYLE_WAVE_ENVIRONMENT):
            raise EvidenceError("functional inner-wave environment is incomplete")
        if shard_id != FUNCTIONAL_V20_INNER_WAVE_POLICY["enabled_shard_id"]:
            raise EvidenceError("functional inner-wave environment is restricted to H01")
        if inner_wave_environment[_S3_Q4_STYLE_WAVE_ENV] != "1":
            raise EvidenceError("functional inner-wave opt-in value is invalid")
        try:
            inner_result_deadline = float(
                inner_wave_environment[_S3_Q4_STYLE_RESULT_DEADLINE_ENV]
            )
            inner_cleanup_deadline = float(
                inner_wave_environment[_S3_Q4_STYLE_CLEANUP_DEADLINE_ENV]
            )
        except (TypeError, ValueError) as exc:
            raise EvidenceError("functional inner-wave deadline is malformed") from exc
        if (
            not math.isfinite(inner_result_deadline)
            or not math.isfinite(inner_cleanup_deadline)
            or inner_result_deadline <= 0.0
            or inner_cleanup_deadline - inner_result_deadline != 15.0
        ):
            raise EvidenceError("functional inner-wave deadlines are invalid")
        environment.update(inner_wave_environment)
    routed_paths = [
        metadata,
        sandbox / "src",
        roots["ANYmesh"] / "src",
        roots["ANYgeometry"] / "src",
        roots["ANYmaterial"] / "src",
        roots["ANYfileIO"] / "src",
    ]
    for raw_path in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not raw_path:
            continue
        path = Path(raw_path)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved == (ROOT / "src").resolve() or resolved in {
            item.resolve() for item in routed_paths
        }:
            continue
        routed_paths.append(resolved)
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in routed_paths)
    return environment


def _functional_shard_selectors(
    shard_nodes: Sequence[str], full_nodes: Sequence[str]
) -> tuple[list[str], dict[str, Any]]:
    """Select complete modules, but address every split module by exact node ID."""

    full_by_module: dict[str, list[str]] = {}
    for node_id in full_nodes:
        full_by_module.setdefault(node_id.partition("::")[0], []).append(node_id)
    shard_by_module: dict[str, list[str]] = {}
    for node_id in shard_nodes:
        shard_by_module.setdefault(node_id.partition("::")[0], []).append(node_id)

    selectors: list[str] = []
    full_module_count = 0
    exact_node_count = 0
    for module, nodes in shard_by_module.items():
        if nodes == full_by_module.get(module):
            selectors.append(module)
            full_module_count += 1
        else:
            selectors.extend(nodes)
            exact_node_count += len(nodes)

    expanded: list[str] = []
    for selector in selectors:
        if "::" in selector:
            expanded.append(selector)
        else:
            expanded.extend(full_by_module[selector])
    if expanded != list(shard_nodes):
        raise EvidenceError("functional shard selectors do not preserve exact node order")
    return selectors, {
        "exact_node_count": exact_node_count,
        "full_module_count": full_module_count,
        "selector_count": len(selectors),
        "selectors_sha256": _functional_list_hash(selectors),
    }


def _run_functional_shard(
    prepared: Mapping[str, Any],
    *,
    absolute_deadline: float,
    deadline_seconds: int,
    timeout_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # V19 reserves bounded time inside each shard cutoff for descendant-tree
    # closure and evidence finalization.  Without this reserve a pytest child
    # could consume the entire shard budget and leave log hashing/result
    # validation running beyond the registered absolute cutoff.
    process_deadline = absolute_deadline
    if prepared.get("v19") is True:
        process_deadline -= int(timeout_policy["termination_grace_seconds"])
        process_deadline -= int(timeout_policy["evidence_reserve_seconds"])
    deadline_expired_on_entry = time.monotonic() >= process_deadline
    shard = prepared["shard"]
    _set_functional_shard_tree_state(shard["shard_id"], "NOT_LAUNCHED")
    shard_root = prepared["shard_root"]
    sandbox = prepared["sandbox"]
    logs = shard_root / "logs"
    expected_path = logs / "expected-nodes.json"
    progress_path = logs / "progress.ndjson"
    result_path = logs / "shard-result.json"
    stdout_path = logs / "stdout.txt"
    stderr_path = logs / "stderr.txt"
    started_at = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    artifact_deadline_expired = False

    def artifact(path: Path) -> dict[str, Any] | None:
        nonlocal artifact_deadline_expired
        try:
            if (
                path.is_file()
                and not path.is_symlink()
                and not _is_reparse_point(path)
            ):
                if prepared.get("v19") is True:
                    return _functional_bounded_hash_record(
                        path,
                        absolute_deadline,
                        f"{shard['shard_id']} failure-artifact hashing",
                    )
                return file_hash_record(path)
        except FunctionalDeadlineExpired:
            artifact_deadline_expired = True
            return None
        except OSError:
            return None
        return None

    def failure(
        status: str,
        *,
        error: BaseException | None,
        command: Sequence[str],
        selector_summary: Mapping[str, Any],
        timed_out: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        ended_at = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        raw: dict[str, Any] = {
            "attempts": 0,
            "command": list(command),
            "command_sha256": _functional_list_hash(command),
            "deadline_seconds": deadline_seconds,
            "disposition": {
                "PREPARATION_FAILED": "PREPARATION_FAILED_BEFORE_LAUNCH",
                "START_FAILED": "PROCESS_START_FAILED_NO_CHILD",
                "TIMED_OUT_NOT_STARTED": "DEADLINE_EXPIRED_NOT_STARTED",
            }[status],
            "duration_seconds": time.perf_counter() - started,
            "ended_at": ended_at,
            "pid": None,
            "progress": artifact(progress_path),
            "pytest_durations": {"enabled": True, "minimum_seconds": 0.0},
            "returncode": (
                int(timeout_policy["timeout_exit_code"]) if timed_out else 250
            ),
            "selector_summary": dict(selector_summary),
            "shard_id": shard["shard_id"],
            "started_at": started_at,
            "stderr": artifact(stderr_path),
            "stdout": artifact(stdout_path),
            "termination": None,
            "timed_out": timed_out,
        }
        if prepared.get("v19") is True:
            raw.update(
                {
                    "barrier": _functional_barrier_raw_record(
                        prepared, pid=None, absolute_deadline=absolute_deadline
                    ),
                    "finalization_deadline_expired": artifact_deadline_expired,
                    "process_ended_at": None,
                    "process_ended_monotonic_ns": None,
                    "process_started_at": None,
                    "process_started_monotonic_ns": None,
                }
            )
        if error is not None:
            raw["error"] = repr(error)
        canonical = {
            "exit_code": raw["returncode"],
            "node_count": shard["node_count"],
            "node_ids_sha256": shard["node_ids_sha256"],
            "result": None,
            "shard_id": shard["shard_id"],
            "status": status,
        }
        if prepared.get("v19") is True:
            canonical["descendant_tree_cleanup"] = "NOT_LAUNCHED"
        return canonical, raw

    command: list[str] = []
    selector_summary: dict[str, Any] = {
        "exact_node_count": 0,
        "full_module_count": 0,
        "selector_count": 0,
        "selectors_sha256": _functional_list_hash([]),
    }
    if deadline_expired_on_entry:
        try:
            with (
                stdout_path.open("xb"),
                stderr_path.open("xb"),
                progress_path.open("xb"),
            ):
                pass
            _functional_progress_event(
                progress_path,
                "TIMED_OUT",
                launched=False,
                shard_id=shard["shard_id"],
            )
        except BaseException:
            pass
        return failure(
            "TIMED_OUT_NOT_STARTED",
            error=None,
            command=command,
            selector_summary=selector_summary,
            timed_out=True,
        )
    try:
        _functional_deadline_check(
            process_deadline, f"{shard['shard_id']} selector preparation"
        )
        selectors, selector_summary = _functional_shard_selectors(
            shard["node_ids"], prepared["full_node_ids"]
        )
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "scripts.run_e4_pl_burnin_gate",
            "--durations=0",
            "--durations-min=0.0",
            f"--basetemp={shard_root / 'basetemp'}",
            *selectors,
        ]
        _functional_deadline_check(
            process_deadline, f"{shard['shard_id']} log preparation"
        )
        with stdout_path.open("xb"), stderr_path.open("xb"), progress_path.open("xb"):
            pass
        _functional_progress_event(
            progress_path,
            "PREPARATION_STARTED",
            selector_count=len(selectors),
            shard_id=shard["shard_id"],
        )
        _functional_deadline_check(
            process_deadline, f"{shard['shard_id']} environment preparation"
        )
        environment = _functional_shard_environment(
            sandbox=sandbox,
            shard_root=shard_root,
            expected_path=expected_path,
            progress_path=progress_path,
            result_path=result_path,
            shard_id=shard["shard_id"],
            barrier=prepared.get("barrier"),
            inner_wave_environment=prepared.get("inner_wave_environment"),
        )
        _functional_deadline_check(
            process_deadline, f"{shard['shard_id']} environment validation"
        )
    except BaseException as exc:
        timed_out_during_preparation = time.monotonic() >= process_deadline
        try:
            if progress_path.exists():
                _functional_progress_event(
                    progress_path,
                    (
                        "TIMED_OUT"
                        if timed_out_during_preparation
                        else "PREPARATION_FAILED"
                    ),
                    error=repr(exc),
                    launched=False,
                    shard_id=shard["shard_id"],
                )
        except BaseException:
            pass
        return failure(
            (
                "TIMED_OUT_NOT_STARTED"
                if timed_out_during_preparation
                else "PREPARATION_FAILED"
            ),
            error=None if timed_out_during_preparation else exc,
            command=command,
            selector_summary=selector_summary,
            timed_out=timed_out_during_preparation,
        )

    remaining = process_deadline - time.monotonic()
    if remaining <= 0:
        try:
            _functional_progress_event(
                progress_path,
                "TIMED_OUT",
                launched=False,
                shard_id=shard["shard_id"],
            )
        except BaseException:
            pass
        return failure(
            "TIMED_OUT_NOT_STARTED",
            error=None,
            command=command,
            selector_summary=selector_summary,
            timed_out=True,
        )

    timed_out = False
    termination: dict[str, Any] | None = None
    termination_error: str | None = None
    process_error: str | None = None
    pid: int | None = None
    process_started_at: str | None = None
    process_started_monotonic_ns: int | None = None
    process_ended_at: str | None = None
    process_ended_monotonic_ns: int | None = None
    returncode = 250
    start_error: BaseException | None = None
    launch_deadline_expired = False
    worker: subprocess.Popen[bytes] | None = None
    job_handle: int | None = None
    descendant_tree_cleanup = "NOT_LAUNCHED"
    with stdout_path.open("ab") as stdout_stream, stderr_path.open("ab") as stderr_stream:
        if time.monotonic() >= process_deadline:
            launch_deadline_expired = True
        else:
            try:
                worker, job_handle = _launch_functional_process(
                    command,
                    cwd=sandbox,
                    env=environment,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                )
                process_started_at = (
                    dt.datetime.now(dt.timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                process_started_monotonic_ns = time.monotonic_ns()
                _set_functional_shard_tree_state(
                    shard["shard_id"], "TREE_LIVE_UNPROVEN"
                )
                barrier = prepared.get("barrier")
                if isinstance(barrier, Mapping):
                    with barrier["lock"]:
                        if shard["shard_id"] in barrier["workers"]:
                            raise EvidenceError(
                                "functional READY worker registration is not exclusive"
                            )
                        barrier["workers"][shard["shard_id"]] = worker
            except BaseException as exc:
                start_error = exc
        if worker is not None:
            pid = worker.pid
            child_exit_observed = False
            try:
                try:
                    _functional_progress_event(
                        progress_path,
                        "STARTED",
                        pid=pid,
                        shard_id=shard["shard_id"],
                    )
                    barrier = prepared.get("barrier")
                    if isinstance(barrier, Mapping):
                        while True:
                            if barrier["abort"].is_set():
                                process_error = "functional READY barrier aborted"
                                returncode = 250
                                break
                            remaining_wait = process_deadline - time.monotonic()
                            if remaining_wait <= 0.0:
                                raise subprocess.TimeoutExpired(command, 0.0)
                            try:
                                returncode = int(
                                    worker.wait(timeout=min(0.05, remaining_wait))
                                )
                            except subprocess.TimeoutExpired:
                                continue
                            child_exit_observed = True
                            break
                    else:
                        returncode = int(
                            worker.wait(
                                timeout=max(0.001, process_deadline - time.monotonic())
                            )
                        )
                        child_exit_observed = True
                except subprocess.TimeoutExpired:
                    timed_out = True
                    returncode = int(timeout_policy["timeout_exit_code"])
                except BaseException as exc:
                    process_error = repr(exc)
                    returncode = 250
            finally:
                try:
                    termination, descendant_tree_cleanup = (
                        _close_functional_process_tree(
                            worker,
                            child_exit_observed=child_exit_observed,
                            job_handle=job_handle,
                            policy=timeout_policy,
                        )
                    )
                except BaseException as exc:  # outer watchdog remains containment
                    termination_error = repr(exc)
                    error_raw = termination_error.encode("utf-8", errors="replace")
                    termination = {
                        "bytes": len(error_raw),
                        "returncode": 260,
                        "sha256": _sha256_bytes(error_raw),
                    }
                    descendant_tree_cleanup = "TREE_TERMINATION_FAILED"
                process_ended_at = (
                    dt.datetime.now(dt.timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                process_ended_monotonic_ns = time.monotonic_ns()
                if descendant_tree_cleanup == "TREE_TERMINATION_FAILED":
                    _FUNCTIONAL_UNPROVEN_TREE.set()
                _set_functional_shard_tree_state(
                    shard["shard_id"], descendant_tree_cleanup
                )
        _functional_deadline_check(
            absolute_deadline, f"{shard['shard_id']} stream finalization"
        )
        stdout_stream.flush()
        stderr_stream.flush()
        os.fsync(stdout_stream.fileno())
        os.fsync(stderr_stream.fileno())
        _functional_deadline_check(
            absolute_deadline, f"{shard['shard_id']} stream finalization completion"
        )
    if launch_deadline_expired:
        try:
            _functional_progress_event(
                progress_path,
                "TIMED_OUT",
                launched=False,
                shard_id=shard["shard_id"],
            )
        except BaseException:
            pass
        return failure(
            "TIMED_OUT_NOT_STARTED",
            error=None,
            command=command,
            selector_summary=selector_summary,
            timed_out=True,
        )
    if start_error is not None:
        try:
            _functional_progress_event(
                progress_path,
                "START_FAILED",
                error=repr(start_error),
                shard_id=shard["shard_id"],
            )
        except BaseException:
            pass
        return failure(
            "START_FAILED",
            error=start_error,
            command=command,
            selector_summary=selector_summary,
            timed_out=False,
        )
    _functional_deadline_check(
        absolute_deadline, f"{shard['shard_id']} progress finalization"
    )
    try:
        _functional_progress_event(
            progress_path,
            "TIMED_OUT" if timed_out else "PROCESS_EXITED",
            launched=True,
            returncode=returncode,
            shard_id=shard["shard_id"],
            termination_returncode=(
                None if termination is None else termination["returncode"]
            ),
        )
    except BaseException as exc:
        if process_error is None:
            process_error = f"progress-finalization: {exc!r}"
    _functional_deadline_check(
        absolute_deadline, f"{shard['shard_id']} progress finalization completion"
    )
    ended_at = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    progress_artifact = (
        _functional_bounded_hash_record(
            progress_path,
            absolute_deadline,
            f"{shard['shard_id']} progress hashing",
        )
        if prepared.get("v19") is True
        else file_hash_record(progress_path)
    )
    stderr_artifact = (
        _functional_bounded_hash_record(
            stderr_path,
            absolute_deadline,
            f"{shard['shard_id']} stderr hashing",
        )
        if prepared.get("v19") is True
        else file_hash_record(stderr_path)
    )
    stdout_artifact = (
        _functional_bounded_hash_record(
            stdout_path,
            absolute_deadline,
            f"{shard['shard_id']} stdout hashing",
        )
        if prepared.get("v19") is True
        else file_hash_record(stdout_path)
    )
    raw: dict[str, Any] = {
        "attempts": 1 if pid is not None else 0,
        "command": command,
        "command_sha256": _functional_list_hash(command),
        "deadline_seconds": deadline_seconds,
        "duration_seconds": time.perf_counter() - started,
        "ended_at": ended_at,
        "pid": pid,
        "progress": progress_artifact,
        "pytest_durations": {"enabled": True, "minimum_seconds": 0.0},
        "returncode": returncode,
        "selector_summary": selector_summary,
        "shard_id": shard["shard_id"],
        "started_at": started_at,
        "stderr": stderr_artifact,
        "stdout": stdout_artifact,
        "termination": termination,
        "timed_out": timed_out,
    }
    if prepared.get("v19") is True:
        raw.update(
            {
                "barrier": _functional_barrier_raw_record(
                    prepared, pid=pid, absolute_deadline=absolute_deadline
                ),
                "process_ended_at": process_ended_at,
                "process_ended_monotonic_ns": process_ended_monotonic_ns,
                "process_started_at": process_started_at,
                "process_started_monotonic_ns": process_started_monotonic_ns,
            }
        )
    if termination_error is not None:
        raw["termination_error"] = termination_error
    if process_error is not None:
        raw["process_error"] = process_error
    canonical: dict[str, Any] = {
        "exit_code": returncode,
        "node_count": shard["node_count"],
        "node_ids_sha256": shard["node_ids_sha256"],
        "result": None,
        "shard_id": shard["shard_id"],
        "status": (
            "TIMEOUT_TREE_TERMINATION_FAILED"
            if descendant_tree_cleanup == "TREE_TERMINATION_FAILED"
            else "TIMED_OUT_NOT_STARTED"
            if timed_out and pid is None
            else "TIMED_OUT"
            if timed_out
            else "PROCESS_ERROR"
            if process_error is not None
            else "PROCESS_FAILED"
        ),
    }
    if prepared.get("v19") is True:
        canonical["descendant_tree_cleanup"] = descendant_tree_cleanup
    result_is_regular = (
        result_path.is_file()
        and not result_path.is_symlink()
        and not _is_reparse_point(result_path)
    )
    result_raw: bytes | None = None
    if result_is_regular:
        # A timed-out pytest process can finish writing its session record just
        # before tree termination.  Bind every such partial even though it may
        # never be treated as a passing result.
        if prepared.get("v19") is True:
            result_raw = _functional_bounded_read_bytes(
                result_path,
                absolute_deadline,
                f"{shard['shard_id']} result hashing",
            )
            canonical["result"] = {
                "bytes": len(result_raw),
                "sha256": _sha256_bytes(result_raw),
            }
        else:
            canonical["result"] = file_hash_record(result_path)
    if canonical["status"] == "PROCESS_FAILED" and result_is_regular:
        try:
            result = _validate_functional_shard_result(
                (
                    strict_json_loads(result_raw)
                    if result_raw is not None
                    else strict_json_load(result_path)
                ),
                shard=shard,
            )
        except (EvidenceError, OSError, UnicodeError) as exc:
            raw["result_error"] = str(exc)
            canonical["status"] = "MALFORMED_RESULT"
        else:
            outcomes = [row["outcome"] for row in result["nodes"]]
            passed = (
                returncode == 0
                and descendant_tree_cleanup == "CHILD_EXIT_AND_TREE_CLOSED"
                and result["exit_code"] == 0
                and result["collection_matches"] is True
                and all(outcome in {"PASSED", "SKIPPED", "XFAIL"} for outcome in outcomes)
            )
            canonical["status"] = "PASS" if passed else "TEST_FAILURE"
    elif canonical["status"] == "PROCESS_FAILED":
        canonical["status"] = "MISSING_RESULT"
    _functional_deadline_check(
        absolute_deadline, f"{shard['shard_id']} result validation completion"
    )
    return canonical, raw


def _run_functional_wave_unprotected(
    cycle: int,
    *,
    contract_path: Path = S3_Q4_CONTRACT_PATH,
    wave_started: float | None = None,
) -> int:
    """Run v18 or the seven-worker single-split successor wave."""

    if cycle not in {1, 2}:
        raise EvidenceError("functional-wave cycle must be 1 or 2")
    if wave_started is None:
        wave_started = time.monotonic()
    contract = strict_json_load(contract_path)
    if not isinstance(contract, dict):
        raise EvidenceError("S3/Q4 burn-in contract must be an object")
    wave = validate_functional_wave_contract(contract)
    v19 = wave["schema"] == FUNCTIONAL_WAVE_V19_SCHEMA
    if v19:
        execution = wave["execution"]
        preparation_deadline = wave_started + int(
            execution["preparation_cutoff_seconds"]
        )
        ready_barrier_deadline = wave_started + int(
            execution["ready_barrier_cutoff_seconds"]
        )
        shard_execution_deadline = wave_started + int(
            execution["shard_execution_cutoff_seconds"]
        )
        normal_evidence_deadline = wave_started + int(
            execution["normal_evidence_cutoff_seconds"]
        )
        emergency_publication_deadline = wave_started + int(
            execution["emergency_publication_cutoff_seconds"]
        )
        deadline = int(execution["emergency_publication_cutoff_seconds"])
    else:
        deadline = int(wave["execution"]["internal_deadline_seconds"])
        preparation_deadline = wave_started + deadline
        ready_barrier_deadline = preparation_deadline
        shard_execution_deadline = preparation_deadline
        normal_evidence_deadline = preparation_deadline
        emergency_publication_deadline = preparation_deadline
    absolute_deadline = emergency_publication_deadline
    _functional_deadline_check(preparation_deadline, "functional-wave routing")
    routing = wave["execution"]["artifact_routing"]
    external_root = Path(contract["non_resource_commands"]["output_root"])
    if not external_root.is_absolute():
        raise EvidenceError("functional-wave output authority must be absolute")
    wave_parent = external_root / routing["directory_name"]
    resolved_wave_parent = wave_parent.resolve()
    resolved_repository = ROOT.resolve()
    if (
        resolved_wave_parent == resolved_repository
        or resolved_repository in resolved_wave_parent.parents
        or resolved_wave_parent in resolved_repository.parents
    ):
        raise EvidenceError("functional-wave output must be outside the repository")
    _functional_deadline_check(preparation_deadline, "functional-wave parent creation")
    wave_parent.mkdir(parents=True, exist_ok=True)
    _functional_deadline_check(preparation_deadline, "functional-wave parent validation")
    if wave_parent.is_symlink() or _is_reparse_point(wave_parent):
        raise EvidenceError("functional-wave parent must not be a reparse point")
    wave_root = wave_parent / f"cycle-{cycle}"
    try:
        _functional_deadline_check(preparation_deadline, "functional-wave cycle creation")
        wave_root.mkdir()
    except FileExistsError as exc:
        raise EvidenceError("functional-wave cycle output is one-shot and already exists") from exc
    if wave_root.is_symlink() or _is_reparse_point(wave_root):
        raise EvidenceError("functional-wave cycle output must not be a reparse point")

    _functional_deadline_check(preparation_deadline, "source-status preparation")
    pre_status = _functional_source_status(
        ROOT, absolute_deadline=preparation_deadline
    )
    _functional_deadline_check(preparation_deadline, "source-status validation")
    if pre_status["bytes"] != 0:
        raise EvidenceError("functional source repository is not completely clean")
    timeout_policy = _validate_functional_timeout_policy(
        contract, schema=wave["schema"]
    )
    if os.name == "nt":
        _functional_deadline_check(preparation_deadline, "taskkill authority check")
        _functional_taskkill_authority(timeout_policy)
        _functional_deadline_check(preparation_deadline, "taskkill authority validation")
    archive_path = wave_root / routing["archive_filename"]
    _functional_deadline_check(preparation_deadline, "source archive creation")
    identity, archive_process = _create_functional_archive(
        ROOT,
        archive_path,
        absolute_deadline=preparation_deadline,
        environment=os.environ,
    )
    _functional_deadline_check(preparation_deadline, "source archive hashing")
    archive_record = _functional_bounded_hash_record(
        archive_path, preparation_deadline, "source archive hashing"
    )

    source_path = wave_root / routing["source_directory_name"]
    _functional_deadline_check(preparation_deadline, "source extraction")
    graph_rows, graph_summary = _extract_functional_tar(
        archive_path, source_path, absolute_deadline=preparation_deadline
    )
    _functional_deadline_check(preparation_deadline, "source extraction validation")
    prepared: list[dict[str, Any]] = []
    for shard in wave["manifest"]["shards"]:
        shard_id = shard["shard_id"]
        _functional_deadline_check(
            preparation_deadline, f"{shard_id} sandbox preparation"
        )
        shard_root = wave_root / f"{routing['shard_directory_prefix']}{shard_id}"
        shard_root.mkdir()
        for child in FUNCTIONAL_SHARD_DIRECTORIES:
            _functional_deadline_check(
                preparation_deadline, f"{shard_id} directory preparation"
            )
            if child == "cwd":
                continue
            (shard_root / child).mkdir()
        expected_path = shard_root / "logs" / "expected-nodes.json"
        _functional_deadline_check(
            preparation_deadline, f"{shard_id} expected-node preparation"
        )
        with expected_path.open("xb") as stream:
            stream.write(canonical_json_bytes(shard["node_ids"]))
        sandbox = shard_root / "cwd"
        prepared.append(
            {
                "full_node_ids": wave["manifest"]["full_node_ids"],
                "sandbox": sandbox,
                "shard": shard,
                "shard_root": shard_root,
                "v19": v19,
            }
        )

    def extract_and_verify_sandbox(item: Mapping[str, Any]) -> str:
        shard_id = item["shard"]["shard_id"]
        _functional_deadline_check(
            preparation_deadline, f"{shard_id} sandbox extraction"
        )
        rows, summary = _extract_functional_tar(
            archive_path, item["sandbox"], absolute_deadline=preparation_deadline
        )
        _functional_deadline_check(
            preparation_deadline, f"{shard_id} sandbox validation"
        )
        if rows != graph_rows or summary != graph_summary:
            raise EvidenceError(
                "functional source sandboxes have different initial graphs"
            )
        return shard_id

    _functional_deadline_check(preparation_deadline, "sandbox extraction wave")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=(
            int(wave["execution"]["max_concurrent_workers"])
            if v19
            else 4
        ),
        thread_name_prefix="functional-extract",
    ) as pool:
        extraction_futures: list[concurrent.futures.Future[Any]] = []
        for item in prepared:
            shard_id = item["shard"]["shard_id"]
            _functional_deadline_check(
                preparation_deadline, f"{shard_id} sandbox extraction submission"
            )
            extraction_futures.append(
                pool.submit(extract_and_verify_sandbox, item)
            )
        for item, future in zip(prepared, extraction_futures, strict=True):
            shard = item["shard"]
            shard_id = shard["shard_id"]
            _functional_deadline_check(
                preparation_deadline, f"{shard_id} sandbox extraction collection"
            )
            if future.result() != shard_id:
                raise EvidenceError("functional sandbox extraction identity mismatch")
            print(
                f"[functional-wave] prepared {shard_id} "
                f"({shard['node_count']} nodes)",
                file=sys.stderr,
                flush=True,
            )
    graph_payload = {
        "files": graph_rows,
        "schema": wave["source"]["file_graph_schema"],
        "summary": graph_summary,
    }
    graph_path = wave_root / wave["source"]["file_graph_filename"]
    _functional_deadline_check(preparation_deadline, "source-graph publication")
    with graph_path.open("xb") as stream:
        stream.write(canonical_json_bytes(graph_payload))
    _functional_deadline_check(preparation_deadline, "functional shard launch wave")

    canonical_by_id: dict[str, dict[str, Any]] = {}
    raw_by_id: dict[str, dict[str, Any]] = {}

    prepared_by_id = {
        item["shard"]["shard_id"]: item for item in prepared
    }
    ready_barrier: dict[str, Any] | None = None
    if v19:
        ready_barrier = _new_functional_ready_barrier(
            prepared_by_id,
            deadline=ready_barrier_deadline,
        )
        for item in prepared:
            item["barrier"] = ready_barrier
        inner_policy = execution["inner_wave_policy"]
        if inner_policy["enabled"]:
            inner_shard_id = str(inner_policy["enabled_shard_id"])
            prepared_by_id[inner_shard_id]["inner_wave_environment"] = {
                str(inner_policy["opt_in_environment"]): str(
                    inner_policy["opt_in_value"]
                ),
                str(inner_policy["result_deadline_environment"]): repr(
                    wave_started + int(inner_policy["result_cutoff_seconds"])
                ),
                str(inner_policy["cleanup_deadline_environment"]): repr(
                    wave_started + int(inner_policy["cleanup_cutoff_seconds"])
                ),
            }

    def collect_worker(
        shard_id: str, future: concurrent.futures.Future[Any]
    ) -> None:
        try:
            canonical, raw = future.result()
        except BaseException as exc:  # preserve peer results; never retry
            tree_state = _get_functional_shard_tree_state(shard_id)
            tree_unproven = tree_state in {
                "TREE_LIVE_UNPROVEN",
                "TREE_TERMINATION_FAILED",
            } or not v19
            if tree_unproven:
                _FUNCTIONAL_UNPROVEN_TREE.set()
            shard = prepared_by_id[shard_id]["shard"]
            canonical = {
                "exit_code": 250,
                "node_count": shard["node_count"],
                "node_ids_sha256": shard["node_ids_sha256"],
                "result": None,
                "shard_id": shard_id,
                "status": (
                    "TIMEOUT_TREE_TERMINATION_FAILED"
                    if tree_unproven
                    else "PROCESS_ERROR"
                ),
            }
            if v19:
                canonical["descendant_tree_cleanup"] = (
                    "TREE_TERMINATION_FAILED" if tree_unproven else tree_state
                )
            raw = {
                "attempts": 0 if tree_state == "NOT_LAUNCHED" else 1,
                "error": repr(exc),
                "shard_id": shard_id,
            }
        canonical_by_id[shard_id] = canonical
        raw_by_id[shard_id] = raw
        print(
            f"[functional-wave] {shard_id} finished: {canonical['status']}",
            file=sys.stderr,
            flush=True,
        )

    if v19:
        execution = wave["execution"]
        shard_ids = tuple(str(value) for value in execution["shard_execution_ids"])
        if shard_ids != FUNCTIONAL_V19_SHARD_IDS:
            raise EvidenceError("v20 shard-execution order differs from authority")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(execution["max_concurrent_workers"]),
            thread_name_prefix="functional-bounded-concurrent",
        ) as pool:
            futures: dict[concurrent.futures.Future[Any], str] = {}
            for shard_id in shard_ids:
                _functional_deadline_check(
                    shard_execution_deadline,
                    f"{shard_id} concurrent launch submission",
                )
                future = pool.submit(
                    _run_functional_shard,
                    prepared_by_id[shard_id],
                    absolute_deadline=shard_execution_deadline,
                    deadline_seconds=int(
                        execution["shard_execution_cutoff_seconds"]
                    ),
                    timeout_policy=timeout_policy,
                )
                futures[future] = shard_id
            assert ready_barrier is not None
            try:
                _publish_functional_ready_release(
                    ready_barrier,
                    futures_by_id={
                        shard_id: future for future, shard_id in futures.items()
                    },
                    prepared_by_id=prepared_by_id,
                )
            except BaseException as exc:
                # Wake every launch thread so its existing finally path closes
                # the complete child tree.  No shard is retried or cancelled.
                ready_barrier["abort"].set()
                print(
                    f"[functional-wave] READY barrier blocked: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
            for future in concurrent.futures.as_completed(futures):
                collect_worker(futures[future], future)
        _functional_deadline_check(
            shard_execution_deadline, "concurrent shard terminalization"
        )
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(wave["execution"]["max_workers"]),
            thread_name_prefix="functional-wave",
        ) as pool:
            futures = {}
            for item in prepared:
                shard_id = item["shard"]["shard_id"]
                _functional_deadline_check(
                    absolute_deadline, f"{shard_id} launch submission"
                )
                future = pool.submit(
                    _run_functional_shard,
                    item,
                    absolute_deadline=absolute_deadline,
                    deadline_seconds=deadline,
                    timeout_policy=timeout_policy,
                )
                futures[future] = shard_id
            for future in concurrent.futures.as_completed(futures):
                collect_worker(futures[future], future)

    evidence_deadline = normal_evidence_deadline if v19 else absolute_deadline
    _functional_deadline_check(evidence_deadline, "post-wave source status")
    post_status = _functional_source_status(
        ROOT, absolute_deadline=evidence_deadline
    )
    _functional_deadline_check(evidence_deadline, "post-wave source validation")
    if post_status != pre_status:
        raise EvidenceError("functional source repository status changed during the wave")
    ordered_canonical = [
        canonical_by_id[row["shard_id"]] for row in wave["manifest"]["shards"]
    ]
    if _functional_wave_has_unproven_tree(ordered_canonical):
        _FUNCTIONAL_UNPROVEN_TREE.set()
    schedule = (
        _functional_v19_schedule(canonical_by_id, raw_by_id) if v19 else None
    )
    success = all(row["status"] == "PASS" for row in ordered_canonical)
    if v19:
        success = (
            success
            and schedule["launch"] == "ALL_LAUNCHED"
            and schedule["terminal"] == "ALL_TERMINAL"
            and schedule["all_workers_overlapped"] is True
        )
    aggregate = {
        "candidate": identity,
        "manifest": {
            "module_count": wave["manifest"]["module_count"],
            "modules_sha256": wave["manifest"]["modules_sha256"],
            "node_count": wave["manifest"]["node_count"],
            "node_ids_sha256": wave["manifest"]["full_node_ids_sha256"],
            "shard_count": len(wave["manifest"]["shards"]),
        },
        "schema": wave["aggregate"]["schema"],
        "shards": ordered_canonical,
        "source": {
            "archive": archive_record,
            "file_graph": _functional_bounded_hash_record(
                graph_path, evidence_deadline, "source-graph final hashing"
            ),
            "file_graph_content": graph_summary,
            "repository_status": pre_status,
        },
        "terminal": (
            wave["aggregate"]["success_terminal"]
            if success
            else wave["aggregate"]["blocked_terminal"]
        ),
    }
    if v19:
        aggregate["schedule"] = schedule
    aggregate_path = wave_root / routing["aggregate_filename"]
    diagnostics_path = wave_root / routing["raw_diagnostics_filename"]
    diagnostics = {
        "archive_process": archive_process,
        "artifact_extent": _functional_artifact_extent(
            wave_root,
            exclude={
                routing["aggregate_filename"],
                routing["raw_diagnostics_filename"],
            },
            absolute_deadline=evidence_deadline,
        ),
        "cycle": cycle,
        "schema": (
            FUNCTIONAL_WAVE_V19_DIAGNOSTICS_SCHEMA
            if v19
            else FUNCTIONAL_WAVE_DIAGNOSTICS_SCHEMA
        ),
        "shards": [raw_by_id[row["shard_id"]] for row in wave["manifest"]["shards"]],
        "source_status_after": post_status,
        "source_status_before": pre_status,
    }
    if v19:
        aggregate_raw = _publish_functional_evidence_atomically(
            wave_root,
            routing=routing,
            wave=wave,
            cycle=cycle,
            diagnostics=diagnostics,
            aggregate=aggregate,
            decision_deadline=normal_evidence_deadline,
            publication_deadline=emergency_publication_deadline,
        )
    else:
        _functional_deadline_check(absolute_deadline, "raw diagnostics serialization")
        diagnostics_raw = canonical_json_bytes(diagnostics)
        _functional_deadline_check(absolute_deadline, "raw diagnostics publication")
        with diagnostics_path.open("xb") as stream:
            stream.write(diagnostics_raw)
            stream.flush()
            os.fsync(stream.fileno())
        _functional_deadline_check(absolute_deadline, "raw diagnostics hashing")
        aggregate["diagnostics"] = _functional_bounded_hash_record(
            diagnostics_path, absolute_deadline, "raw diagnostics hashing"
        )
        _functional_deadline_check(absolute_deadline, "aggregate serialization")
        aggregate_raw = canonical_json_bytes(aggregate)
        _functional_deadline_check(absolute_deadline, "aggregate publication")
        with aggregate_path.open("xb") as stream:
            stream.write(aggregate_raw)
            stream.flush()
            os.fsync(stream.fileno())
        _functional_deadline_check(absolute_deadline, "aggregate stdout publication")
    sys.stdout.buffer.write(aggregate_raw)
    sys.stdout.buffer.flush()
    if not v19:
        _functional_deadline_check(absolute_deadline, "functional-wave completion")
    return 0 if success else 1


def _functional_artifact_extent(
    root: Path,
    *,
    exclude: set[str],
    absolute_deadline: float | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        if absolute_deadline is not None:
            _functional_deadline_check(
                absolute_deadline, "functional artifact-extent enumeration"
            )
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        if path.is_symlink() or _is_reparse_point(path):
            raise EvidenceError(f"functional-wave artifact is a reparse file: {relative}")
        if absolute_deadline is not None:
            record = _functional_bounded_hash_record(
                path, absolute_deadline, "functional artifact-extent hashing"
            )
        else:
            record = file_hash_record(path)
        records.append(
            {
                "bytes": record["bytes"],
                "path": relative,
                "sha256": record["sha256"],
            }
        )
    return {
        "files": len(records),
        "records": records,
        "sha256": _sha256_bytes(canonical_json_bytes(records)),
    }


def _validate_staged_functional_evidence(
    diagnostics_raw: bytes,
    aggregate_raw: bytes,
    *,
    wave: Mapping[str, Any],
    cycle: int,
    success: bool,
) -> None:
    """Fail closed on staged v19 evidence before either canonical rename."""

    diagnostics = strict_json_loads(diagnostics_raw)
    aggregate = strict_json_loads(aggregate_raw)
    if diagnostics_raw != canonical_json_bytes(diagnostics):
        raise EvidenceError("staged functional diagnostics are noncanonical")
    if aggregate_raw != canonical_json_bytes(aggregate):
        raise EvidenceError("staged functional aggregate is noncanonical")
    if not isinstance(diagnostics, dict) or not isinstance(aggregate, dict):
        raise EvidenceError("staged functional evidence must contain JSON objects")

    diagnostics_keys = set(diagnostics)
    normal_diagnostics_keys = {
        "archive_process",
        "artifact_extent",
        "cycle",
        "schema",
        "shards",
        "source_status_after",
        "source_status_before",
    }
    failure_diagnostics_keys = {
        "artifact_extent",
        "cycle",
        "failure",
        "schema",
    }
    if frozenset(diagnostics_keys) not in {
        frozenset(normal_diagnostics_keys),
        frozenset(failure_diagnostics_keys),
    }:
        raise EvidenceError("staged v19 diagnostics keys mismatch")
    if diagnostics.get("schema") != FUNCTIONAL_WAVE_V19_DIAGNOSTICS_SCHEMA:
        raise EvidenceError("staged v19 diagnostics schema mismatch")
    if diagnostics.get("cycle") != cycle:
        raise EvidenceError("staged v19 diagnostics cycle mismatch")
    if diagnostics_keys == failure_diagnostics_keys:
        failure = _exact_keys(
            diagnostics["failure"],
            {"message", "repository_status", "type"},
            "$staged_diagnostics.failure",
        )
        if not isinstance(failure["message"], str) or not failure["message"]:
            raise EvidenceError("staged v19 failure message must be nonempty")
        if not isinstance(failure["type"], str) or not failure["type"]:
            raise EvidenceError("staged v19 failure type must be nonempty")
        if failure["repository_status"] is not None:
            _validate_hash_record(
                failure["repository_status"],
                "$staged_diagnostics.failure.repository_status",
            )

    aggregate = _exact_keys(
        aggregate,
        {
            "candidate",
            "diagnostics",
            "manifest",
            "schedule",
            "schema",
            "shards",
            "source",
            "terminal",
        },
        "$staged_aggregate",
    )
    if aggregate["schema"] != FUNCTIONAL_WAVE_V19_AGGREGATE_SCHEMA:
        raise EvidenceError("staged v19 aggregate schema mismatch")
    if aggregate["diagnostics"] != {
        "bytes": len(diagnostics_raw),
        "sha256": _sha256_bytes(diagnostics_raw),
    }:
        raise EvidenceError("staged aggregate diagnostics binding mismatch")
    expected_terminal = wave["aggregate"][
        "success_terminal" if success else "blocked_terminal"
    ]
    if aggregate["terminal"] != expected_terminal:
        raise EvidenceError("staged aggregate terminal disagrees with disposition")

    authorities = wave["manifest"]["shards"]
    shards = aggregate["shards"]
    if not isinstance(shards, list) or len(shards) != len(authorities):
        raise EvidenceError("staged v19 aggregate has the wrong shard count")
    canonical_by_id: dict[str, Mapping[str, Any]] = {}
    all_pass = True
    for index, (row, authority) in enumerate(zip(shards, authorities, strict=True)):
        row = _exact_keys(
            row,
            {
                "descendant_tree_cleanup",
                "exit_code",
                "node_count",
                "node_ids_sha256",
                "result",
                "shard_id",
                "status",
            },
            f"$staged_aggregate.shards[{index}]",
        )
        if (
            row["shard_id"] != authority["shard_id"]
            or row["node_count"] != authority["node_count"]
            or row["node_ids_sha256"] != authority["node_ids_sha256"]
        ):
            raise EvidenceError("staged v19 shard authority mismatch")
        if row["descendant_tree_cleanup"] not in FUNCTIONAL_V19_TREE_CLEANUP_STATES:
            raise EvidenceError("staged v19 shard cleanup enum is invalid")
        if row["status"] == "PASS" and (
            row["exit_code"] != 0
            or row["descendant_tree_cleanup"] != "CHILD_EXIT_AND_TREE_CLOSED"
            or row["result"] is None
        ):
            raise EvidenceError("staged v19 PASS lacks a clean terminal child tree")
        all_pass = all_pass and row["status"] == "PASS" and row["exit_code"] == 0
        canonical_by_id[row["shard_id"]] = row

    schedule = _exact_keys(
        aggregate["schedule"],
        {"all_workers_overlapped", "launch", "terminal"},
        "$staged_aggregate.schedule",
    )
    if schedule["launch"] not in FUNCTIONAL_V19_LAUNCH_STATES:
        raise EvidenceError("staged v19 launch enum is invalid")
    if schedule["terminal"] not in FUNCTIONAL_V19_TERMINAL_STATES:
        raise EvidenceError("staged v19 terminal-state enum is invalid")
    _require_bool(
        schedule["all_workers_overlapped"],
        "$staged_aggregate.schedule.all_workers_overlapped",
    )

    raw_by_id: dict[str, Mapping[str, Any]] | None = None
    if diagnostics_keys == normal_diagnostics_keys:
        raw_rows = diagnostics["shards"]
        if not isinstance(raw_rows, list) or len(raw_rows) != len(authorities):
            raise EvidenceError("staged v19 raw shard diagnostics are incomplete")
        raw_by_id = {}
        for expected, raw_row in zip(authorities, raw_rows, strict=True):
            if not isinstance(raw_row, dict) or raw_row.get("shard_id") != expected["shard_id"]:
                raise EvidenceError("staged v19 raw shard diagnostics are misordered")
            raw_by_id[expected["shard_id"]] = raw_row
    expected_schedule = _functional_v19_schedule(canonical_by_id, raw_by_id)
    if schedule != expected_schedule:
        raise EvidenceError("staged v19 schedule disagrees with raw process evidence")

    expected_success = (
        all_pass
        and schedule["launch"] == "ALL_LAUNCHED"
        and schedule["terminal"] == "ALL_TERMINAL"
        and schedule["all_workers_overlapped"] is True
    )
    if success != expected_success:
        raise EvidenceError("staged v19 terminal does not follow shard/overlap evidence")


def _functional_pending_path(path: Path) -> Path:
    return path.with_name(f".pending-{path.name}")


def _discard_functional_pending_evidence(
    wave_root: Path, *, routing: Mapping[str, Any]
) -> None:
    """Remove only known unpublished staging files in the fresh cycle root."""

    resolved_root = wave_root.resolve(strict=True)
    for key in ("raw_diagnostics_filename", "aggregate_filename"):
        pending = _functional_pending_path(wave_root / str(routing[key]))
        if not pending.exists() and not pending.is_symlink():
            continue
        if pending.parent.resolve(strict=True) != resolved_root:
            raise EvidenceError("functional pending evidence escaped its cycle root")
        if pending.is_dir() or pending.is_symlink() or _is_reparse_point(pending):
            raise EvidenceError("functional pending evidence is not a regular file")
        pending.unlink()


def _recover_functional_diagnostics_first_publication(
    wave_root: Path,
    *,
    routing: Mapping[str, Any],
    wave: Mapping[str, Any],
    cycle: int,
    publication_deadline: float,
) -> bytes | None:
    """Finish an already-decided diagnostics-first atomic publication.

    The normal publisher checks the decision deadline before promoting the
    diagnostics.  Therefore an existing canonical diagnostics file plus its
    still-pending aggregate proves that the pass/block decision was timely and
    may be recovered without rerunning a worker.  A merely staged diagnostics
    file does not carry that proof and is never promoted here.
    """

    aggregate_path = wave_root / str(routing["aggregate_filename"])
    diagnostics_path = wave_root / str(routing["raw_diagnostics_filename"])
    aggregate_pending = _functional_pending_path(aggregate_path)
    diagnostics_pending = _functional_pending_path(diagnostics_path)
    if aggregate_path.exists() or aggregate_path.is_symlink():
        return None
    if not diagnostics_path.exists() or not aggregate_pending.exists():
        return None
    if diagnostics_pending.exists() or diagnostics_pending.is_symlink():
        raise EvidenceError(
            "promoted functional diagnostics coexist with pending diagnostics"
        )
    diagnostics_raw = _functional_bounded_read_bytes(
        diagnostics_path,
        publication_deadline,
        "functional recovery diagnostics read",
    )
    aggregate_raw = _functional_bounded_read_bytes(
        aggregate_pending,
        publication_deadline,
        "functional recovery aggregate read",
    )
    aggregate_value = strict_json_loads(aggregate_raw)
    if not isinstance(aggregate_value, dict):
        raise EvidenceError("pending functional aggregate is not an object")
    success = (
        aggregate_value.get("terminal") == wave["aggregate"]["success_terminal"]
    )
    _validate_staged_functional_evidence(
        diagnostics_raw,
        aggregate_raw,
        wave=wave,
        cycle=cycle,
        success=success,
    )
    _functional_deadline_check(
        publication_deadline, "functional recovery aggregate promotion"
    )
    if aggregate_path.exists() or aggregate_path.is_symlink():
        raise EvidenceError("functional aggregate appeared during recovery")
    os.replace(aggregate_pending, aggregate_path)
    return aggregate_raw


def _publish_functional_evidence_atomically(
    wave_root: Path,
    *,
    routing: Mapping[str, Any],
    wave: Mapping[str, Any],
    cycle: int,
    diagnostics: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    decision_deadline: float | None,
    publication_deadline: float,
    allow_existing_diagnostics: bool = False,
) -> bytes:
    """Stage, validate, then atomically promote v19 diagnostics and aggregate."""

    aggregate_path = wave_root / str(routing["aggregate_filename"])
    diagnostics_path = wave_root / str(routing["raw_diagnostics_filename"])
    aggregate_pending = _functional_pending_path(aggregate_path)
    diagnostics_pending = _functional_pending_path(diagnostics_path)
    if aggregate_path.exists() or aggregate_path.is_symlink():
        raise EvidenceError("functional aggregate already exists")
    if aggregate_pending.exists() or diagnostics_pending.exists():
        raise EvidenceError("functional pending evidence already exists")

    diagnostics_raw = canonical_json_bytes(dict(diagnostics))
    diagnostics_staged = False
    if diagnostics_path.exists() or diagnostics_path.is_symlink():
        if not allow_existing_diagnostics:
            raise EvidenceError("functional diagnostics already exist")
        if (
            not diagnostics_path.is_file()
            or diagnostics_path.is_symlink()
            or _is_reparse_point(diagnostics_path)
        ):
            raise EvidenceError("existing functional diagnostics are not regular")
        existing_raw = _functional_bounded_read_bytes(
            diagnostics_path,
            publication_deadline,
            "existing functional diagnostics read",
        )
        if existing_raw != diagnostics_raw:
            raise EvidenceError("existing functional diagnostics differ during recovery")
        diagnostics_raw = existing_raw
    else:
        with diagnostics_pending.open("xb") as stream:
            stream.write(diagnostics_raw)
            stream.flush()
            os.fsync(stream.fileno())
        diagnostics_staged = True

    aggregate_value = dict(aggregate)
    aggregate_value["diagnostics"] = {
        "bytes": len(diagnostics_raw),
        "sha256": _sha256_bytes(diagnostics_raw),
    }
    aggregate_raw = canonical_json_bytes(aggregate_value)
    with aggregate_pending.open("xb") as stream:
        stream.write(aggregate_raw)
        stream.flush()
        os.fsync(stream.fileno())

    staged_diagnostics_raw = (
        _functional_bounded_read_bytes(
            diagnostics_pending,
            publication_deadline,
            "staged functional diagnostics read",
        )
        if diagnostics_staged
        else diagnostics_raw
    )
    staged_aggregate_raw = _functional_bounded_read_bytes(
        aggregate_pending,
        publication_deadline,
        "staged functional aggregate read",
    )
    expected_success = (
        aggregate_value["terminal"] == wave["aggregate"]["success_terminal"]
    )
    _validate_staged_functional_evidence(
        staged_diagnostics_raw,
        staged_aggregate_raw,
        wave=wave,
        cycle=cycle,
        success=expected_success,
    )
    if decision_deadline is not None:
        _functional_deadline_check(
            decision_deadline, "v19 normal evidence decision"
        )

    if diagnostics_staged:
        _functional_deadline_check(
            publication_deadline, "v19 diagnostics atomic promotion"
        )
        if diagnostics_path.exists() or diagnostics_path.is_symlink():
            raise EvidenceError("functional diagnostics appeared before promotion")
        os.replace(diagnostics_pending, diagnostics_path)
    _functional_deadline_check(
        publication_deadline, "v19 aggregate atomic promotion"
    )
    if aggregate_path.exists() or aggregate_path.is_symlink():
        raise EvidenceError("functional aggregate appeared before promotion")
    os.replace(aggregate_pending, aggregate_path)
    return aggregate_raw


def _record_functional_wave_failure(
    cycle: int,
    contract_path: Path,
    error: Exception,
    *,
    publication_deadline: float | None = None,
) -> int:
    """Bind every partial external artifact without retrying a failed wave."""

    try:
        contract = (
            strict_json_loads(
                _functional_bounded_read_bytes(
                    contract_path,
                    publication_deadline,
                    "emergency contract read",
                )
            )
            if publication_deadline is not None
            else strict_json_load(contract_path)
        )
        wave = validate_functional_wave_contract(contract)
        v19 = wave["schema"] == FUNCTIONAL_WAVE_V19_SCHEMA
        if v19 and publication_deadline is None:
            publication_deadline = time.monotonic() + int(
                wave["execution"]["emergency_publication_cutoff_seconds"]
            )
        routing = wave["execution"]["artifact_routing"]
        wave_root = (
            Path(contract["non_resource_commands"]["output_root"])
            / routing["directory_name"]
            / f"cycle-{cycle}"
        )
    except Exception:
        raise error
    if not wave_root.is_dir() or wave_root.is_symlink() or _is_reparse_point(wave_root):
        raise error
    aggregate_path = wave_root / routing["aggregate_filename"]
    diagnostics_path = wave_root / routing["raw_diagnostics_filename"]
    if v19:
        assert publication_deadline is not None
        _functional_deadline_check(
            publication_deadline, "v19 emergency publication recovery"
        )
        recovered = _recover_functional_diagnostics_first_publication(
            wave_root,
            routing=routing,
            wave=wave,
            cycle=cycle,
            publication_deadline=publication_deadline,
        )
        if recovered is not None:
            recovered_value = strict_json_loads(recovered)
            sys.stdout.buffer.write(recovered)
            sys.stdout.buffer.flush()
            return (
                0
                if recovered_value.get("terminal")
                == wave["aggregate"]["success_terminal"]
                else 1
            )
        _discard_functional_pending_evidence(wave_root, routing=routing)
        if diagnostics_path.exists() or diagnostics_path.is_symlink():
            raise EvidenceError(
                "functional diagnostics exist without a recoverable pending aggregate"
            )
    if aggregate_path.exists():
        aggregate_raw = (
            _functional_bounded_read_bytes(
                aggregate_path,
                publication_deadline,
                "existing emergency aggregate read",
            )
            if v19
            else aggregate_path.read_bytes()
        )
        existing_aggregate = strict_json_loads(aggregate_raw)
        diagnostics_record = (
            _functional_bounded_hash_record(
                diagnostics_path,
                publication_deadline,
                "existing emergency diagnostics hashing",
            )
            if v19 and diagnostics_path.is_file()
            else file_hash_record(diagnostics_path)
            if diagnostics_path.is_file()
            else None
        )
        if (
            not isinstance(existing_aggregate, dict)
            or not diagnostics_path.is_file()
            or existing_aggregate.get("diagnostics")
            != diagnostics_record
        ):
            raise EvidenceError("existing functional aggregate lacks its diagnostics binding")
    else:
        try:
            identity = {
                "commit": _git(
                    ROOT,
                    "rev-parse",
                    "HEAD",
                    timeout_seconds=(
                        _functional_remaining_seconds(
                            publication_deadline,
                            "emergency commit identity query",
                        )
                        if v19
                        else None
                    ),
                ),
                "tree": _git(
                    ROOT,
                    "rev-parse",
                    "HEAD^{tree}",
                    timeout_seconds=(
                        _functional_remaining_seconds(
                            publication_deadline,
                            "emergency tree identity query",
                        )
                        if v19
                        else None
                    ),
                ),
            }
        except Exception:
            identity = {"commit": "0" * 40, "tree": "0" * 40}
        try:
            observed_status = _functional_source_status(
                ROOT,
                absolute_deadline=(publication_deadline if v19 else None),
            )
        except Exception:
            observed_status = None
        empty_status = {"bytes": 0, "sha256": _sha256_bytes(b"")}
        status = observed_status if observed_status == empty_status else None
        shards = []
        for shard in wave["manifest"]["shards"]:
            result_path = (
                wave_root
                / f"{routing['shard_directory_prefix']}{shard['shard_id']}"
                / "logs"
                / "shard-result.json"
            )
            result_record = (
                _functional_bounded_hash_record(
                    result_path,
                    publication_deadline,
                    "emergency shard-result hashing",
                )
                if v19
                and result_path.is_file()
                and not result_path.is_symlink()
                and not _is_reparse_point(result_path)
                else file_hash_record(result_path)
                if result_path.is_file()
                and not result_path.is_symlink()
                and not _is_reparse_point(result_path)
                else None
            )
            shards.append(
                {
                    **(
                        {
                            "descendant_tree_cleanup": (
                                "TREE_TERMINATION_FAILED"
                                if _get_functional_shard_tree_state(
                                    shard["shard_id"]
                                )
                                == "TREE_LIVE_UNPROVEN"
                                else _get_functional_shard_tree_state(
                                    shard["shard_id"]
                                )
                            )
                        }
                        if v19
                        else {}
                    ),
                    "exit_code": 250,
                    "node_count": shard["node_count"],
                    "node_ids_sha256": shard["node_ids_sha256"],
                    "result": result_record,
                    "shard_id": shard["shard_id"],
                    "status": "NOT_STARTED_OR_UNBOUND_PARTIAL",
                }
            )
        archive_path = wave_root / routing["archive_filename"]
        graph_path = wave_root / wave["source"]["file_graph_filename"]
        try:
            graph_value = (
                strict_json_loads(
                    _functional_bounded_read_bytes(
                        graph_path,
                        publication_deadline,
                        "emergency graph read",
                    )
                )
                if v19
                else strict_json_load(graph_path)
            )
            graph_content = graph_value["summary"]
        except Exception:
            graph_content = None
        aggregate = {
            "candidate": identity,
            "manifest": {
                "module_count": wave["manifest"]["module_count"],
                "modules_sha256": wave["manifest"]["modules_sha256"],
                "node_count": wave["manifest"]["node_count"],
                "node_ids_sha256": wave["manifest"]["full_node_ids_sha256"],
                "shard_count": len(wave["manifest"]["shards"]),
            },
            "schema": wave["aggregate"]["schema"],
            "shards": shards,
            "source": {
                "archive": (
                    _functional_bounded_hash_record(
                        archive_path,
                        publication_deadline,
                        "emergency archive hashing",
                    )
                    if archive_path.is_file() and not archive_path.is_symlink()
                    and v19
                    else file_hash_record(archive_path)
                    if archive_path.is_file() and not archive_path.is_symlink()
                    else None
                ),
                "file_graph": (
                    _functional_bounded_hash_record(
                        graph_path,
                        publication_deadline,
                        "emergency graph hashing",
                    )
                    if graph_path.is_file() and not graph_path.is_symlink()
                    and v19
                    else file_hash_record(graph_path)
                    if graph_path.is_file() and not graph_path.is_symlink()
                    else None
                ),
                "file_graph_content": graph_content,
                "repository_status": status,
            },
            "terminal": wave["aggregate"]["blocked_terminal"],
        }
        if v19:
            aggregate["schedule"] = _functional_v19_schedule(
                {row["shard_id"]: row for row in shards}
            )
        if not diagnostics_path.exists():
            diagnostics = {
                "artifact_extent": _functional_artifact_extent(
                    wave_root,
                    exclude={
                        routing["aggregate_filename"],
                        routing["raw_diagnostics_filename"],
                    },
                    absolute_deadline=(publication_deadline if v19 else None),
                ),
                "cycle": cycle,
                "failure": {
                    "message": str(error),
                    "repository_status": observed_status,
                    "type": type(error).__name__,
                },
                "schema": (
                    FUNCTIONAL_WAVE_V19_DIAGNOSTICS_SCHEMA
                    if v19
                    else FUNCTIONAL_WAVE_DIAGNOSTICS_SCHEMA
                ),
            }
        else:
            diagnostics = (
                strict_json_loads(
                    _functional_bounded_read_bytes(
                        diagnostics_path,
                        publication_deadline,
                        "existing emergency diagnostics read",
                    )
                )
                if v19
                else strict_json_load(diagnostics_path)
            )
        if v19:
            assert publication_deadline is not None
            aggregate_raw = _publish_functional_evidence_atomically(
                wave_root,
                routing=routing,
                wave=wave,
                cycle=cycle,
                diagnostics=diagnostics,
                aggregate=aggregate,
                decision_deadline=None,
                publication_deadline=publication_deadline,
                allow_existing_diagnostics=diagnostics_path.exists(),
            )
        else:
            if not diagnostics_path.exists():
                with diagnostics_path.open("xb") as stream:
                    stream.write(canonical_json_bytes(diagnostics))
            aggregate["diagnostics"] = file_hash_record(diagnostics_path)
            aggregate_raw = canonical_json_bytes(aggregate)
            with aggregate_path.open("xb") as stream:
                stream.write(aggregate_raw)
    sys.stdout.buffer.write(aggregate_raw)
    sys.stdout.buffer.flush()
    return 1


def run_functional_wave(
    cycle: int,
    *,
    contract_path: Path = S3_Q4_CONTRACT_PATH,
) -> int:
    _FUNCTIONAL_UNPROVEN_TREE.clear()
    with _FUNCTIONAL_SHARD_TREE_STATE_LOCK:
        _FUNCTIONAL_SHARD_TREE_STATE.clear()
    wave_started = time.monotonic()
    try:
        try:
            return _run_functional_wave_unprotected(
                cycle,
                contract_path=contract_path,
                wave_started=wave_started,
            )
        except FunctionalDeadlineExpired as exc:
            print(f"[functional-wave] {exc}", file=sys.stderr, flush=True)
            return _record_functional_wave_failure(
                cycle,
                contract_path,
                exc,
                publication_deadline=wave_started
                + FUNCTIONAL_V19_STAGE_POLICY[
                    "emergency_publication_cutoff_seconds"
                ],
            )
        except Exception as exc:
            return _record_functional_wave_failure(
                cycle,
                contract_path,
                exc,
                publication_deadline=wave_started
                + FUNCTIONAL_V19_STAGE_POLICY[
                    "emergency_publication_cutoff_seconds"
                ],
            )
    finally:
        if _FUNCTIONAL_UNPROVEN_TREE.is_set():
            _await_outer_resource_tree_termination()


def _run_pytest_lane(lane: str, selected: Sequence[str]) -> int:
    """Run one lane with a fresh external pytest and metadata root."""

    if lane not in {"quick", "functional", "performance", "extended", "ci"}:
        raise EvidenceError(f"pytest lane does not support basetemp isolation: {lane}")
    if ROOT != _GATE_SOURCE_ROOT:
        # Existing unit tests replace ROOT with a temporary synthetic project.
        # That seam never applies to a CLI execution from the real gate file.
        parent = ROOT / ".pytest_tmp_q1m_runtime"
    else:
        contract = strict_json_load(S3_Q4_CONTRACT_PATH)
        if not isinstance(contract, dict):
            raise EvidenceError("S3/Q4 contract must be an object")
        external_root = Path(contract["non_resource_commands"]["output_root"])
        if not external_root.is_absolute():
            raise EvidenceError("pytest runtime root authority must be absolute")
        parent = external_root / "common-runtime"
        resolved_parent = parent.resolve()
        resolved_repository = ROOT.resolve()
        if (
            resolved_parent == resolved_repository
            or resolved_repository in resolved_parent.parents
            or resolved_parent in resolved_repository.parents
        ):
            raise EvidenceError("pytest runtime root must be outside the repository")
    if parent.exists() and (parent.is_symlink() or _is_reparse_point(parent)):
        raise EvidenceError("pytest runtime parent must not be a reparse point")
    parent.mkdir(parents=True, exist_ok=True)
    basetemp: Path | None = None
    metadata_workspace: Path | None = None
    try:
        basetemp = Path(tempfile.mkdtemp(prefix=f"{lane}-", dir=parent))
        if basetemp.resolve().parent != parent.resolve():
            raise EvidenceError("Q1M pytest basetemp escaped its parent")
        metadata_workspace = Path(
            tempfile.mkdtemp(prefix=f"{lane}-metadata-", dir=parent)
        )
        if metadata_workspace.resolve().parent != parent.resolve():
            raise EvidenceError("Q1M metadata workspace escaped its parent")
        roots = _local_roots()
        metadata_overlay = metadata_workspace / "source-distributions"
        _write_source_metadata_overlay(roots, metadata_overlay)
        environment = _pytest_environment(
            roots=roots,
            metadata_overlay=metadata_overlay,
        )
        if lane == "quick":
            environment[_ACTIVE_TEST_LANE_ENV] = "quick"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"--basetemp={basetemp}",
                *selected,
            ],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        return int(completed.returncode)
    finally:
        for temporary_path in (basetemp, metadata_workspace):
            if temporary_path is None or not os.path.lexists(temporary_path):
                continue
            if temporary_path.is_symlink() or not temporary_path.is_dir():
                temporary_path.unlink()
                continue

            def make_writable_and_retry(function, path, _excinfo):
                os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
                function(path)

            if sys.version_info >= (3, 12):
                shutil.rmtree(temporary_path, onexc=make_writable_and_retry)
            else:
                # ``onexc`` was added in Python 3.12. ANYfeSolver also supports
                # Python 3.11, whose equivalent callback is ``onerror``.
                shutil.rmtree(temporary_path, onerror=make_writable_and_retry)
        try:
            parent.rmdir()
        except OSError:
            # Concurrent common commands may share only this external parent;
            # every randomized child remains isolated.
            pass


def _validate_ci_policy(contract: Mapping[str, Any]) -> dict[str, Any]:
    policy = _exact_keys(
        contract["ci_policy"],
        {
            "coordinator_wall_limit_seconds",
            "extent",
            "required_lanes",
            "smoke_or_representative_only_forbidden",
        },
        "$contract.ci_policy",
    )
    if policy != {
        "coordinator_wall_limit_seconds": 1200,
        "extent": "COMPLETE_FROZEN_INVENTORIES",
        "required_lanes": ["quick", "functional", "additive"],
        "smoke_or_representative_only_forbidden": True,
    }:
        raise EvidenceError("bounded CI policy differs from the frozen authority")
    return policy


def _ci_deadline_check(absolute_deadline: float, stage: str) -> None:
    if time.monotonic() >= absolute_deadline:
        raise subprocess.TimeoutExpired(["bounded-ci", stage], 1200)


def _terminate_ci_workers(
    workers: Sequence[subprocess.Popen[bytes]],
    timeout_policy: Mapping[str, Any],
) -> bool:
    """Terminate live CI trees and return whether absence was proven."""

    live = [worker for worker in workers if worker.poll() is None]
    if not live:
        return True
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(live), thread_name_prefix="ci-tree-termination"
    ) as pool:
        results = list(
            pool.map(
                lambda worker: _terminate_functional_process_tree(
                    worker, timeout_policy
                ),
                live,
            )
        )
    return all(
        result["returncode"] == 0 and worker.poll() is not None
        for worker, result in zip(live, results, strict=True)
    )


def _collect_ci_nonfunctional_nodes(
    selectors: Sequence[str],
    *,
    ci_root: Path,
    roots: Mapping[str, Path],
    absolute_deadline: float,
    timeout_policy: Mapping[str, Any],
) -> list[str]:
    """Collect and bind the exact quick/additive node IDs before execution."""

    capture_path = ci_root / "nonfunctional-node-ids.json"
    metadata_overlay = ci_root / "collector-metadata"
    _ci_deadline_check(absolute_deadline, "collection metadata preparation")
    _write_source_metadata_overlay(roots, metadata_overlay)
    environment = _pytest_environment(roots=roots, metadata_overlay=metadata_overlay)
    environment.update(
        {
            _CI_CAPTURE_ENV: str(capture_path),
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_ADDOPTS": "-p no:cacheprovider",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "scripts.run_e4_pl_burnin_gate",
        *selectors,
    ]
    options: dict[str, Any] = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    _ci_deadline_check(absolute_deadline, "collection process launch")
    collector = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **options,
    )
    try:
        returncode = collector.wait(
            timeout=max(0.001, absolute_deadline - time.monotonic())
        )
    except subprocess.TimeoutExpired:
        if not _terminate_ci_workers([collector], timeout_policy):
            _FUNCTIONAL_UNPROVEN_TREE.set()
            _await_outer_resource_tree_termination()
        raise
    if returncode != 0:
        raise EvidenceError(f"bounded CI collection failed with exit {returncode}")
    _ci_deadline_check(absolute_deadline, "collection result validation")
    nodes = strict_json_load(capture_path)
    if (
        not isinstance(nodes, list)
        or not all(isinstance(node, str) and node for node in nodes)
        or len(nodes) != len(set(nodes))
        or {
            "node_count": len(nodes),
            "node_ids_sha256": _functional_list_hash(nodes),
        }
        != CI_NONFUNCTIONAL_NODE_AUTHORITY
    ):
        raise EvidenceError("bounded CI nonfunctional node set differs from authority")
    return nodes


def _ci_nonfunctional_module_buckets(
    lanes: Mapping[str, Sequence[str]],
) -> list[list[str]]:
    """Return the frozen complete quick/additive assignment for P01--P04."""

    quick = list(lanes["quick"])
    additive = list(lanes["additive"])
    return [quick, additive[0::3], additive[1::3], additive[2::3]]


def _ci_authority(values: Sequence[str], *, count_key: str) -> dict[str, Any]:
    return {
        count_key: len(values),
        ("modules_sha256" if count_key == "module_count" else "node_ids_sha256"): (
            _functional_list_hash(values)
        ),
    }


def _ci_functional_node_partitions(
    functional_modules: Sequence[str], full_nodes: Sequence[str]
) -> dict[str, list[str]]:
    """Build CI's four-worker functional extent independently of wave shards."""

    if list(dict.fromkeys(functional_modules)) != list(functional_modules):
        raise EvidenceError("bounded CI functional inventory contains duplicates")
    if CI_SPLIT_MODULE not in functional_modules:
        raise EvidenceError("bounded CI split module is absent from functional inventory")
    full_module_rows: list[str] = []
    exact_rows: list[str] = []
    for shard_id in CI_SHARD_IDS:
        modules = list(CI_FUNCTIONAL_MODULES[shard_id])
        exact_nodes = list(CI_FUNCTIONAL_EXACT_NODES[shard_id])
        if _ci_authority(modules, count_key="module_count") != (
            CI_FUNCTIONAL_MODULE_AUTHORITIES[shard_id]
        ):
            raise EvidenceError(f"bounded CI {shard_id} functional modules differ from authority")
        if _ci_authority(exact_nodes, count_key="node_count") != (
            CI_FUNCTIONAL_EXACT_NODE_AUTHORITIES[shard_id]
        ):
            raise EvidenceError(f"bounded CI {shard_id} exact nodes differ from authority")
        if CI_SPLIT_MODULE in modules:
            raise EvidenceError("bounded CI split module cannot be a full-module selector")
        if any(node.partition("::")[0] != CI_SPLIT_MODULE for node in exact_nodes):
            raise EvidenceError("bounded CI exact selector is outside the split module")
        full_module_rows.extend(modules)
        exact_rows.extend(exact_nodes)
    if len(full_module_rows) != len(set(full_module_rows)):
        raise EvidenceError("bounded CI functional full-module assignments overlap")
    expected_full_modules = set(functional_modules) - {CI_SPLIT_MODULE}
    if set(full_module_rows) != expected_full_modules:
        raise EvidenceError("bounded CI functional full-module assignment is incomplete")
    split_nodes = [
        node for node in full_nodes if node.partition("::")[0] == CI_SPLIT_MODULE
    ]
    if (
        len(exact_rows) != len(set(exact_rows))
        or set(exact_rows) != set(split_nodes)
        or len(split_nodes) != 9
    ):
        raise EvidenceError("bounded CI exact selectors do not partition the split module")

    partitions: dict[str, list[str]] = {}
    assigned: list[str] = []
    for shard_id in CI_SHARD_IDS:
        modules = set(CI_FUNCTIONAL_MODULES[shard_id])
        exact_nodes = set(CI_FUNCTIONAL_EXACT_NODES[shard_id])
        nodes = [
            node
            for node in full_nodes
            if node.partition("::")[0] in modules or node in exact_nodes
        ]
        if _ci_authority(nodes, count_key="node_count") != (
            CI_FUNCTIONAL_NODE_AUTHORITIES[shard_id]
        ):
            raise EvidenceError(f"bounded CI {shard_id} functional nodes differ from authority")
        partitions[shard_id] = nodes
        assigned.extend(nodes)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(full_nodes):
        raise EvidenceError("bounded CI functional node partition is not exact")
    return partitions


def _run_bounded_ci(lanes: Mapping[str, Sequence[str]]) -> int:
    """Run the complete CI extent in four processes under one 20-minute cap."""

    coordinator_started = _GATE_COMMAND_ENTRY
    contract = strict_json_load(S3_Q4_CONTRACT_PATH)
    if not isinstance(contract, dict):
        raise EvidenceError("S3/Q4 contract must be an object")
    ci_policy = _validate_ci_policy(contract)
    wave = validate_functional_wave_contract(contract)
    timeout_policy = _validate_functional_timeout_policy(
        contract, schema=wave["schema"]
    )
    required_lanes = ci_policy["required_lanes"]
    inventories = contract["lane_inventories"]
    for lane in required_lanes:
        selected = list(lanes[lane])
        authority = inventories[lane]
        if (
            authority["count"] != len(selected)
            or authority["sha256"] != _functional_list_hash(selected)
        ):
            raise EvidenceError(f"CI {lane} inventory differs from frozen authority")
    selected_modules = [
        module for lane in required_lanes for module in lanes[lane]
    ]
    if len(selected_modules) != len(set(selected_modules)):
        raise EvidenceError("CI lane inventories overlap")

    full_nodes = wave["manifest"]["full_node_ids"]
    commands: list[list[str]] = []
    nonfunctional = [*lanes["quick"], *lanes["additive"]]
    functional_partitions = _ci_functional_node_partitions(
        lanes["functional"], full_nodes
    )
    # CI owns its own balanced four-worker partition.  Quick remains coherent
    # in P01 and additive modules remain coherently striped over P02--P04.
    nonfunctional_buckets = _ci_nonfunctional_module_buckets(lanes)
    assigned_nonfunctional: list[str] = []
    selector_rows: list[list[str]] = []
    for index, shard_id in enumerate(CI_SHARD_IDS):
        functional_selectors, _summary = _functional_shard_selectors(
            functional_partitions[shard_id], full_nodes
        )
        extras = nonfunctional_buckets[index]
        if _ci_authority(extras, count_key="module_count") != (
            CI_NONFUNCTIONAL_MODULE_AUTHORITIES[shard_id]
        ):
            raise EvidenceError(
                f"bounded CI {shard_id} nonfunctional modules differ from authority"
            )
        assigned_nonfunctional.extend(extras)
        selector_rows.append([*functional_selectors, *extras])
    if sorted(assigned_nonfunctional) != sorted(nonfunctional):
        raise EvidenceError("bounded CI did not assign the complete nonfunctional extent")

    if ROOT != _GATE_SOURCE_ROOT:
        parent = ROOT / ".pytest_tmp_q1m_runtime"
    else:
        external_root = Path(contract["non_resource_commands"]["output_root"])
        if not external_root.is_absolute():
            raise EvidenceError("bounded CI runtime root authority must be absolute")
        parent = external_root / "common-runtime"
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or _is_reparse_point(parent):
        raise EvidenceError("bounded CI runtime parent must not be a reparse point")
    ci_root = Path(tempfile.mkdtemp(prefix="ci-bounded-", dir=parent))
    roots = _local_roots()
    workers: list[subprocess.Popen[bytes]] = []
    cleanup_allowed = True
    try:
        wall_limit = int(ci_policy["coordinator_wall_limit_seconds"])
        termination_grace = int(timeout_policy["termination_grace_seconds"])
        absolute_deadline = coordinator_started + wall_limit
        # Finish child-tree termination before the command-entry parent begins
        # its final 30-second containment reserve.
        termination_start = absolute_deadline - (
            CI_COMMAND_TERMINATION_RESERVE_SECONDS + termination_grace
        )
        _ci_deadline_check(termination_start, "exact node collection")
        captured_nonfunctional = _collect_ci_nonfunctional_nodes(
            nonfunctional,
            ci_root=ci_root,
            roots=roots,
            absolute_deadline=termination_start,
            timeout_policy=timeout_policy,
        )
        if _ci_authority(
            [*full_nodes, *captured_nonfunctional], count_key="node_count"
        ) != CI_FULL_NODE_AUTHORITY:
            raise EvidenceError("bounded CI full node extent differs from authority")
        expected_rows: list[list[str]] = []
        assigned_nodes: list[str] = []
        for index, shard_id in enumerate(CI_SHARD_IDS):
            extra_modules = set(nonfunctional_buckets[index])
            nonfunctional_nodes = [
                node_id
                for node_id in captured_nonfunctional
                if node_id.partition("::")[0] in extra_modules
            ]
            if _ci_authority(nonfunctional_nodes, count_key="node_count") != (
                CI_NONFUNCTIONAL_NODE_AUTHORITIES[shard_id]
            ):
                raise EvidenceError(
                    f"bounded CI {shard_id} nonfunctional nodes differ from authority"
                )
            expected = [*functional_partitions[shard_id], *nonfunctional_nodes]
            if _ci_authority(expected, count_key="node_count") != (
                CI_SHARD_NODE_AUTHORITIES[shard_id]
            ):
                raise EvidenceError(f"bounded CI {shard_id} node set differs from authority")
            expected_rows.append(expected)
            assigned_nodes.extend(expected)
        complete_nodes = [*full_nodes, *captured_nonfunctional]
        if (
            len(assigned_nodes) != len(set(assigned_nodes))
            or set(assigned_nodes) != set(complete_nodes)
        ):
            raise EvidenceError("bounded CI worker nodes are not a disjoint complete extent")

        environments: list[dict[str, str]] = []
        for shard_id, selectors, expected in zip(
            CI_SHARD_IDS, selector_rows, expected_rows, strict=True
        ):
            _ci_deadline_check(termination_start, f"{shard_id} preparation")
            shard_root = ci_root / shard_id
            shard_root.mkdir()
            basetemp = shard_root / "basetemp"
            metadata_overlay = shard_root / "metadata"
            expected_path = shard_root / "expected-nodes.json"
            with expected_path.open("xb") as stream:
                stream.write(canonical_json_bytes(expected))
            _write_source_metadata_overlay(roots, metadata_overlay)
            environment = _pytest_environment(
                roots=roots, metadata_overlay=metadata_overlay
            )
            environment.update(
                {
                    "NUMBA_CACHE_DIR": str(shard_root / "numba-cache"),
                    "NUMBA_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "PYTHONHASHSEED": "0",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPYCACHEPREFIX": str(shard_root / "python-cache"),
                    "PYTEST_ADDOPTS": "-p no:cacheprovider",
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    _CI_EXPECTED_ENV: str(expected_path),
                    _CI_SHARD_ENV: shard_id,
                }
            )
            environments.append(environment)
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "-p",
                    "scripts.run_e4_pl_burnin_gate",
                    "--durations=20",
                    "--durations-min=0.0",
                    f"--basetemp={basetemp}",
                    *selectors,
                ]
            )

        options: dict[str, Any] = {}
        if os.name == "nt":
            _functional_taskkill_authority(timeout_policy)
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        for shard_id, command, environment in zip(
            CI_SHARD_IDS, commands, environments, strict=True
        ):
            if time.monotonic() >= termination_start:
                cleanup_allowed = False
                raise subprocess.TimeoutExpired(command, wall_limit)
            print(
                f"[ci-bounded] starting {shard_id}/{CI_SHARD_IDS[-1]}",
                file=sys.stderr,
                flush=True,
            )
            workers.append(
                subprocess.Popen(command, cwd=ROOT, env=environment, **options)
            )

        while any(worker.poll() is None for worker in workers):
            if time.monotonic() >= termination_start:
                cleanup_allowed = False
                break
            time.sleep(0.1)
        if not cleanup_allowed:
            if not _terminate_ci_workers(workers, timeout_policy):
                _FUNCTIONAL_UNPROVEN_TREE.set()
                _await_outer_resource_tree_termination()
            print(
                "[ci-bounded] 1200-second coordinator limit reached",
                file=sys.stderr,
                flush=True,
            )
            return int(timeout_policy["timeout_exit_code"])
        returncodes = [int(worker.returncode) for worker in workers]
        for shard_id, returncode in zip(CI_SHARD_IDS, returncodes, strict=True):
            print(
                f"[ci-bounded] {shard_id} finished: exit {returncode}",
                file=sys.stderr,
                flush=True,
            )
        return next((code for code in returncodes if code != 0), 0)
    except subprocess.TimeoutExpired:
        cleanup_allowed = False
        if not _terminate_ci_workers(workers, timeout_policy):
            _FUNCTIONAL_UNPROVEN_TREE.set()
            _await_outer_resource_tree_termination()
        return int(timeout_policy["timeout_exit_code"])
    except BaseException:
        if not _terminate_ci_workers(workers, timeout_policy):
            _FUNCTIONAL_UNPROVEN_TREE.set()
            _await_outer_resource_tree_termination()
        raise
    finally:
        if cleanup_allowed and ci_root.exists():
            shutil.rmtree(ci_root, ignore_errors=True)
        try:
            parent.rmdir()
        except OSError:
            pass


def _tracked_head_identity(
    repository: Path, *, absolute_deadline: float | None = None
) -> dict[str, str]:
    """Return HEAD authority after rejecting tracked/index modifications.

    Untracked paths are intentionally ignored: sibling repositories may carry
    unrelated work, and the package source is obtained exclusively from HEAD.
    """

    def remaining(stage: str) -> float | None:
        if absolute_deadline is None:
            return None
        return _functional_remaining_seconds(absolute_deadline, stage)

    if _git(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=no",
        timeout_seconds=remaining("tracked-status identity check"),
    ):
        raise RuntimeError(f"tracked or index changes exist in {repository}")
    return {
        "commit": _git(
            repository,
            "rev-parse",
            "HEAD",
            timeout_seconds=remaining("commit identity check"),
        ),
        "tree": _git(
            repository,
            "rev-parse",
            "HEAD^{tree}",
            timeout_seconds=remaining("tree identity check"),
        ),
    }


def _extract_git_archive(archive: Path, destination: Path) -> dict[str, Any]:
    """Safely extract a Git-created ZIP and hash its complete file graph."""

    destination.mkdir()
    seen: set[str] = set()
    with zipfile.ZipFile(archive, "r") as source:
        for member in source.infolist():
            name = member.filename
            relative = PurePosixPath(name)
            if (
                not name
                or "\\" in name
                or relative.is_absolute()
                or any(part in {"", ".", ".."} for part in relative.parts)
                or name in seen
            ):
                raise RuntimeError(f"unsafe or duplicate Git archive member: {name!r}")
            seen.add(name)
            mode = (member.external_attr >> 16) & 0xFFFF
            kind = stat.S_IFMT(mode)
            if kind == stat.S_IFLNK:
                raise RuntimeError(f"symlink Git archive member is forbidden: {name}")
            if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise RuntimeError(f"unsupported Git archive member type: {name}")
            target = destination.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with source.open(member, "r") as input_stream, target.open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream)
            except FileExistsError as exc:
                raise RuntimeError(f"duplicate Git archive target: {name}") from exc

    graph = []
    for path in sorted(
        (item for item in destination.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(destination).as_posix(),
    ):
        relative = path.relative_to(destination).as_posix()
        graph.append(
            {"bytes": path.stat().st_size, "path": relative, "sha256": _sha256_file(path)}
        )
    if not graph:
        raise RuntimeError("Git archive extracted no regular files")
    return {"files": len(graph), "sha256": _sha256_bytes(canonical_json_bytes(graph))}


def _archive_head_snapshot(
    repository: Path,
    archive: Path,
    destination: Path,
    *,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Materialize only committed HEAD and bind both archive and file graph."""

    attributes_path = Path(
        _git(repository, "rev-parse", "--git-path", "info/attributes")
    )
    if not attributes_path.is_absolute():
        attributes_path = repository / attributes_path
    if attributes_path.exists() or attributes_path.is_symlink() or _is_reparse_point(attributes_path):
        raise RuntimeError("Git info attributes are forbidden during package archiving")
    before = _tracked_head_identity(repository)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive_log = _run_captured(
        [
            *_git_command_prefix(repository),
            "archive",
            "--format=zip",
            f"--output={archive}",
            "HEAD",
        ],
        cwd=archive.parent,
        env=_sanitized_git_environment(environment),
    )
    if not archive.is_file() or archive.is_symlink() or _is_reparse_point(archive):
        raise RuntimeError("git archive did not create a regular file")
    content = _extract_git_archive(archive, destination)
    after = _tracked_head_identity(repository)
    if after != before:
        raise RuntimeError(f"repository HEAD changed while archiving {repository}")
    return {
        "archive": file_hash_record(archive),
        "archive_log": archive_log,
        "commit": before["commit"],
        "content": content,
        "tree": before["tree"],
    }


def _run_captured(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    timeout_options: dict[str, Any] = {}
    if timeout_seconds is not None:
        timeout_options["timeout"] = timeout_seconds
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        check=False,
        capture_output=True,
        **timeout_options,
    )
    combined = completed.stdout + completed.stderr
    result = {
        "bytes": len(combined),
        "returncode": int(completed.returncode),
        "sha256": _sha256_bytes(combined),
    }
    if completed.returncode:
        raise RuntimeError(
            f"package-lane subprocess failed ({completed.returncode}); "
            f"log_sha256={result['sha256']}"
        )
    return result


PACKAGE_SMOKE = r'''
import importlib
import json
import pathlib
import sys
import warnings

target = pathlib.Path(sys.argv[1]).resolve()
forbidden = [pathlib.Path(item).resolve() for item in json.loads(sys.argv[2])]

def under(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

origins = {}
for package in ("anyfesolver", "anygeometry", "anymaterial", "anymesher", "anyfileio"):
    module = importlib.import_module(package)
    origin = pathlib.Path(module.__file__).resolve()
    if not under(origin, target):
        raise AssertionError(f"{package} imported outside isolated target")
    origins[package] = origin.relative_to(target).as_posix()

for entry in sys.path:
    if not entry:
        continue
    resolved = pathlib.Path(entry).resolve()
    if any(under(resolved, root) for root in forbidden):
        raise AssertionError("source repository leaked onto sys.path")
if any(under(pathlib.Path.cwd().resolve(), root) for root in forbidden):
    raise AssertionError("smoke process cwd is inside a source repository")

from anyfesolver import (
    LegacyQ4DeprecationWarning,
    QualifiedE4PLShellElement,
    ShellElement,
    create_shell_element,
    shell_formulation_diagnostics,
)

default = create_shell_element(1, [1, 2, 3, 4], "steel")
non_q4 = [
    create_shell_element(2, [1, 2, 3], "steel"),
    create_shell_element(3, [1, 2, 3, 4, 5, 6], "steel"),
    create_shell_element(4, [1, 2, 3, 4, 5, 6, 7, 8], "steel"),
    create_shell_element(
        5, [1, 2, 3, 4, 5, 6, 7, 8], "steel", reduced_integration=True
    ),
]
if type(default) is not QualifiedE4PLShellElement:
    raise AssertionError("installed wheel did not select qualified Q4")
if any(type(item) is not ShellElement for item in non_q4):
    raise AssertionError("installed wheel changed a non-Q4 topology")
diagnostic = shell_formulation_diagnostics(node_count=4)
if diagnostic["schema"] != "anysolver.shell-formulation-diagnostics-v1":
    raise AssertionError("installed wheel did not export diagnostics")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    rollback = create_shell_element(6, [1, 2, 3, 4], "steel", formulation="legacy")
if type(rollback) is not ShellElement:
    raise AssertionError("legacy selector did not preserve ShellElement")
if not any(item.category is LegacyQ4DeprecationWarning for item in caught):
    raise AssertionError("legacy selector did not emit LegacyQ4DeprecationWarning")

print(json.dumps({
    "diagnostics_schema": diagnostic["schema"],
    "legacy_warning": "LegacyQ4DeprecationWarning",
    "non_q4_types": [type(item).__name__ for item in non_q4],
    "origins": origins,
    "q4_type": type(default).__name__,
}, sort_keys=True, separators=(",", ":")))
'''


def run_package_lane(
    *, output: Path | None = None, wheel_output: Path | None = None
) -> dict[str, Any]:
    """Build verified committed-HEAD archives and smoke an isolated target."""

    for artifact in (output, wheel_output):
        if artifact is not None and os.path.lexists(artifact):
            raise EvidenceError(f"refusing to replace existing package artifact: {artifact}")
    if output is not None and wheel_output is not None and output.resolve() == wheel_output.resolve():
        raise EvidenceError("package result and wheel outputs must be distinct")

    roots = _local_roots()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"

    with tempfile.TemporaryDirectory(prefix="anyfesolver-q1m-package-") as temporary:
        workspace = Path(temporary).resolve()
        if any(
            workspace == path.resolve() or workspace.is_relative_to(path.resolve())
            for path in roots.values()
        ):
            raise RuntimeError("package workspace must be external to every source repository")
        snapshots = workspace / "snapshots"
        archives = workspace / "archives"
        wheelhouse = workspace / "wheelhouse"
        target = workspace / "target"
        smoke_cwd = workspace / "smoke"
        snapshots.mkdir()
        archives.mkdir()
        wheelhouse.mkdir()
        target.mkdir()
        smoke_cwd.mkdir()

        build_logs: dict[str, dict[str, Any]] = {}
        source_records: dict[str, dict[str, Any]] = {}
        wheels: dict[str, Path] = {}
        for repository_name, distribution_name, _package_name in LOCAL_DISTRIBUTIONS:
            snapshot = snapshots / repository_name
            source_records[repository_name] = _archive_head_snapshot(
                roots[repository_name],
                archives / f"{repository_name}.zip",
                snapshot,
                environment=environment,
            )
            before = set(wheelhouse.glob("*.whl"))
            build_logs[repository_name] = _run_captured(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheelhouse),
                    str(snapshot),
                ],
                cwd=workspace,
                env=environment,
            )
            created = set(wheelhouse.glob("*.whl")) - before
            if len(created) != 1:
                raise RuntimeError(f"expected one new wheel for {distribution_name}")
            wheel = created.pop()
            if not wheel.is_file() or wheel.is_symlink() or _is_reparse_point(wheel):
                raise RuntimeError(f"wheel for {distribution_name} is not a regular file")
            if wheel.stat().st_size <= 0:
                raise RuntimeError(f"wheel for {distribution_name} is empty")
            wheels[repository_name] = wheel

        install_log = _run_captured(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-compile",
                "--no-deps",
                "--target",
                str(target),
                *[str(wheels[name]) for name, _, _ in LOCAL_DISTRIBUTIONS],
            ],
            cwd=workspace,
            env=environment,
        )

        # The system environment may contain editable-install ``.pth`` entries
        # for a different checkout.  Remove the entire common repository root,
        # not merely the five snapshots selected above.
        forbidden_roots = [str(_github_root().resolve())]
        smoke_command = [
            sys.executable,
            "-I",
            "-c",
            (
                "import json,pathlib,sys;"
                "roots=[pathlib.Path(p).resolve() for p in json.loads(sys.argv[2])];"
                "inside=lambda p:any(p==r or p.is_relative_to(r) for r in roots);"
                "sys.path[:]=[p for p in sys.path if p and not inside(pathlib.Path(p).resolve())];"
                "sys.path.insert(0,sys.argv[1]);exec(sys.argv[3])"
            ),
            str(target),
            json.dumps(forbidden_roots, separators=(",", ":")),
            PACKAGE_SMOKE,
        ]
        smoke_completed = subprocess.run(
            smoke_command,
            cwd=smoke_cwd,
            env=environment,
            check=False,
            capture_output=True,
        )
        smoke_log = smoke_completed.stdout + smoke_completed.stderr
        if smoke_completed.returncode:
            raise RuntimeError(
                f"installed-wheel smoke failed ({smoke_completed.returncode}); "
                f"log_sha256={_sha256_bytes(smoke_log)}; "
                f"tail={smoke_log[-4000:].decode('utf-8', errors='replace')}"
            )
        try:
            smoke = strict_json_loads(smoke_completed.stdout)
        except EvidenceError as exc:
            raise RuntimeError("installed-wheel smoke did not emit strict JSON") from exc

        wheel_records = {
            name: {
                "bytes": wheel.stat().st_size,
                "filename": wheel.name,
                "sha256": _sha256_file(wheel),
            }
            for name, wheel in sorted(wheels.items())
        }
        result = {
            "build_logs": build_logs,
            "install_log": install_log,
            "schema": _load_contract()["package_result_schema"],
            "smoke": smoke,
            "smoke_log": {
                "bytes": len(smoke_log),
                "sha256": _sha256_bytes(smoke_log),
            },
            "sources": source_records,
            "status": "PASS",
            "wheels": wheel_records,
        }
        validate_package_result(result)
        payload = canonical_json_bytes(result)
        if wheel_output is not None:
            if wheel_output.name != wheels["ANYfeSolver"].name:
                raise EvidenceError(
                    "preserved wheel path basename must match the built ANYfeSolver wheel"
                )
            wheel_output.parent.mkdir(parents=True, exist_ok=True)
            with wheels["ANYfeSolver"].open("rb") as source, wheel_output.open("xb") as target_stream:
                shutil.copyfileobj(source, target_stream)
            if wheel_hash_record(wheel_output) != wheel_hash_record(wheels["ANYfeSolver"]):
                raise EvidenceError("preserved wheel identity differs from built wheel")
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(payload)
            if file_hash_record(output) != {
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }:
                raise EvidenceError("preserved package result identity mismatch")
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
        return result


def _path_bindings(values: Sequence[str] | None, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values or ():
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in result:
            raise EvidenceError(f"invalid or duplicate {label} binding: {value!r}")
        result[name] = Path(raw_path)
    return result


def _create_gate_job() -> int | None:
    """Create a Windows kill-on-close job for one watchdog child tree."""

    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    information = ExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x00002000
    if not kernel32.SetInformationJobObject(
        handle, 9, ctypes.byref(information), ctypes.sizeof(information)
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(handle)
        raise OSError(error, "SetInformationJobObject failed")
    return int(handle)


def _close_gate_job(handle: int | None) -> bool:
    if handle is None:
        return True
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return bool(kernel32.CloseHandle(handle))


def _launch_gate_watchdog_child(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str]
) -> tuple[subprocess.Popen[bytes], int | None]:
    """Launch suspended, bind to the watchdog job, then resume on Windows."""

    if os.name != "nt":
        return (
            subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True),
            None,
        )
    import ctypes
    from ctypes import wintypes

    job_handle = _create_gate_job()
    assert job_handle is not None
    worker: subprocess.Popen[bytes] | None = None
    assigned = False
    try:
        worker = subprocess.Popen(
            command, cwd=cwd, env=env, creationflags=0x00000004
        )
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        process_handle = int(worker._handle)
        if not kernel32.AssignProcessToJobObject(job_handle, process_handle):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        assigned = True
        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        status = int(ntdll.NtResumeProcess(process_handle))
        if status != 0:
            raise OSError(status, "NtResumeProcess failed")
        return worker, job_handle
    except BaseException:
        if assigned:
            _close_gate_job(job_handle)
        else:
            if worker is not None:
                try:
                    worker.kill()
                    worker.wait(timeout=1.0)
                except BaseException:
                    pass
            _close_gate_job(job_handle)
        raise


def _terminate_gate_posix_group(
    worker: subprocess.Popen[bytes], *, absolute_deadline: float
) -> bool:
    """Kill and prove absence of the watchdog child's POSIX process group."""

    try:
        os.killpg(worker.pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    while time.monotonic() < absolute_deadline:
        try:
            os.killpg(worker.pid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        time.sleep(0.01)
    return False


def _run_gate_cli_watchdog(
    lane: str,
    *,
    cycle: int | None,
    command_started: float,
) -> int:
    """Run one long gate behind a command-entry complete-tree watchdog."""

    termination_start_cutoff: int | None = None
    if lane == "ci":
        wall_limit = CI_COMMAND_WALL_LIMIT_SECONDS
        reserve = CI_COMMAND_TERMINATION_RESERVE_SECONDS
        child_arguments = ["ci"]
    elif lane == "functional-wave" and cycle in {1, 2}:
        try:
            watchdog_contract = strict_json_load(S3_Q4_CONTRACT_PATH)
            watchdog_wave = watchdog_contract["functional_wave"]
            watchdog_schema = watchdog_wave["schema"]
        except (EvidenceError, KeyError, OSError, TypeError, UnicodeError, ValueError):
            # A malformed or unreadable successor contract must fail inside a
            # short bound; it cannot inherit the longer historical allowance.
            watchdog_schema = FUNCTIONAL_WAVE_V19_SCHEMA
        wall_limit = (
            FUNCTIONAL_V19_COMMAND_WALL_LIMIT_SECONDS
            if watchdog_schema == FUNCTIONAL_WAVE_V19_SCHEMA
            else FUNCTIONAL_COMMAND_WALL_LIMIT_SECONDS
        )
        if watchdog_schema == FUNCTIONAL_WAVE_V19_SCHEMA:
            termination_start_cutoff = (
                FUNCTIONAL_V19_WATCHDOG_TERMINATION_START_SECONDS
            )
        reserve = FUNCTIONAL_COMMAND_TERMINATION_RESERVE_SECONDS
        child_arguments = ["functional-wave", "--cycle", str(cycle)]
    else:
        raise EvidenceError("gate watchdog received an unsupported invocation")
    absolute_deadline = command_started + wall_limit
    termination_start = (
        command_started + termination_start_cutoff
        if termination_start_cutoff is not None
        else absolute_deadline - reserve
    )
    if time.monotonic() >= termination_start:
        return 124
    if os.environ.get(_GATE_WATCHDOG_ENV) is not None:
        raise EvidenceError("preexisting gate watchdog context is forbidden")

    watchdog_stop = threading.Event()
    ownership_lock = threading.Lock()
    ownership: dict[str, Any] = {
        "job_handle": None,
        "tree_close_result": None,
        "worker": None,
    }

    def close_owned_tree() -> bool:
        with ownership_lock:
            if ownership["tree_close_result"] is not None:
                return bool(ownership["tree_close_result"])
            owned_worker = ownership["worker"]
            owned_job = ownership["job_handle"]
            if owned_worker is None:
                result = True
            elif os.name == "nt":
                result = _close_gate_job(owned_job)
            else:
                result = _terminate_gate_posix_group(
                    owned_worker, absolute_deadline=absolute_deadline
                )
            ownership["job_handle"] = None
            ownership["tree_close_result"] = result
            return result

    def enforce_absolute_deadline() -> None:
        if watchdog_stop.wait(
            max(0.0, termination_start - time.monotonic())
        ):
            return
        proven = close_owned_tree()
        owned_worker = ownership["worker"]
        if proven and owned_worker is not None:
            try:
                owned_worker.wait(
                    timeout=max(0.001, absolute_deadline - time.monotonic())
                )
            except BaseException:
                proven = False
        # This independent thread bounds even a stuck inventory/preparation,
        # Popen, wait, tree-termination, or cleanup operation in the parent.
        os._exit(124 if proven else 250)

    watchdog_thread = threading.Thread(
        target=enforce_absolute_deadline,
        name=f"{lane}-command-entry-watchdog",
        daemon=True,
    )
    watchdog_thread.start()

    token = secrets.token_hex(32)
    parent_pid = os.getpid()
    environment = os.environ.copy()
    environment[_GATE_WATCHDOG_ENV] = f"{lane}:{parent_pid}:{token}"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *child_arguments,
        "--_watchdog-token",
        token,
        "--_watchdog-parent",
        str(parent_pid),
    ]
    worker: subprocess.Popen[bytes] | None = None
    job_handle: int | None = None
    try:
        worker, job_handle = _launch_gate_watchdog_child(
            command, cwd=ROOT, env=environment
        )
        with ownership_lock:
            ownership["worker"] = worker
            ownership["job_handle"] = job_handle
        try:
            returncode = int(
                worker.wait(
                    timeout=max(0.001, termination_start - time.monotonic())
                )
            )
            tree_closed = close_owned_tree()
            if not tree_closed:
                print(
                    f"[{lane}] command watchdog could not close its process-tree job",
                    file=sys.stderr,
                    flush=True,
                )
                watchdog_stop.set()
                return 250
            watchdog_stop.set()
            return returncode
        except subprocess.TimeoutExpired:
            pass
    except BaseException:
        if worker is None:
            watchdog_stop.set()
            raise

    assert worker is not None
    if os.name == "nt":
        if not close_owned_tree():
            print(
                f"[{lane}] command watchdog could not close its process-tree job",
                file=sys.stderr,
                flush=True,
            )
            watchdog_stop.set()
            return 250
        try:
            worker.wait(timeout=max(0.001, absolute_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            print(
                f"[{lane}] command watchdog job closure did not terminate its child",
                file=sys.stderr,
                flush=True,
            )
            watchdog_stop.set()
            return 250
        proven = True
    else:
        proven = close_owned_tree()
    if not proven:
        print(
            f"[{lane}] command watchdog tree termination is unproven",
            file=sys.stderr,
            flush=True,
        )
        watchdog_stop.set()
        return 250
    print(
        f"[{lane}] command-entry wall limit reached; complete tree terminated",
        file=sys.stderr,
        flush=True,
    )
    watchdog_stop.set()
    return 124


def main(argv: list[str] | None = None) -> int:
    global _GATE_COMMAND_ENTRY
    _GATE_COMMAND_ENTRY = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lane",
        choices=(
            "quick",
            "package",
            "functional",
            "functional-wave",
            "performance",
            "extended",
            "ci",
            "list",
            "validate-evidence",
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--repository", action="append")
    parser.add_argument("--request", action="append")
    parser.add_argument("--log", action="append")
    parser.add_argument("--package-result", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--wheel-output", type=Path)
    parser.add_argument("--cycle", type=int, choices=(1, 2))
    parser.add_argument("--_watchdog-token", help=argparse.SUPPRESS)
    parser.add_argument("--_watchdog-parent", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    active_test_lane = os.environ.get(_ACTIVE_TEST_LANE_ENV)
    if active_test_lane is not None:
        if active_test_lane != "quick":
            raise EvidenceError("active burn-in test lane marker is invalid")
        if args.lane not in {"list", "validate-evidence"}:
            raise EvidenceError(
                "nested burn-in execution from the quick pytest lane is forbidden"
            )
    watchdog_context = os.environ.get(_GATE_WATCHDOG_ENV)
    private_arguments_present = (
        args._watchdog_token is not None or args._watchdog_parent is not None
    )
    if watchdog_context is None:
        if private_arguments_present:
            raise EvidenceError("gate watchdog private arguments lack parent context")
        watchdog_child = False
    else:
        if not private_arguments_present:
            raise EvidenceError("preexisting gate watchdog context is forbidden")
        if (
            args._watchdog_token is None
            or not re.fullmatch(r"[0-9a-f]{64}", args._watchdog_token)
            or args._watchdog_parent is None
            or args._watchdog_parent != os.getppid()
            or watchdog_context
            != f"{args.lane}:{args._watchdog_parent}:{args._watchdog_token}"
        ):
            raise EvidenceError("gate watchdog child context is invalid")
        watchdog_child = True
    if watchdog_child and args.lane not in {"ci", "functional-wave"}:
        raise EvidenceError("gate watchdog child context is invalid for this lane")
    if args.lane == "list":
        if any(
            value
            for value in (
                args.output,
                args.evidence,
                args.final,
                args.repository,
                args.request,
                args.log,
                args.package_result,
                args.wheel,
                args.wheel_output,
                args.cycle,
            )
        ):
            parser.error("list accepts no options")
        print(json.dumps(gate_inventories(), sort_keys=True, separators=(",", ":")))
        return 0
    if args.lane == "package":
        if any(
            value
            for value in (
                args.evidence,
                args.final,
                args.repository,
                args.request,
                args.log,
                args.package_result,
                args.wheel,
                args.cycle,
            )
        ):
            parser.error("validation options are invalid for the package lane")
        output = args.output
        wheel_output = args.wheel_output
        if output is None and os.environ.get("Q1M_PACKAGE_RESULT_PATH"):
            output = Path(os.environ["Q1M_PACKAGE_RESULT_PATH"])
        if wheel_output is None and os.environ.get("Q1M_PACKAGE_WHEEL_PATH"):
            wheel_output = Path(os.environ["Q1M_PACKAGE_WHEEL_PATH"])
        run_package_lane(output=output, wheel_output=wheel_output)
        return 0
    if args.lane == "functional-wave":
        if args.cycle is None:
            parser.error("functional-wave requires --cycle {1,2}")
        if any(
            value
            for value in (
                args.output,
                args.evidence,
                args.final,
                args.repository,
                args.request,
                args.log,
                args.package_result,
                args.wheel,
                args.wheel_output,
            )
        ):
            parser.error("functional-wave accepts only --cycle")
        if not watchdog_child:
            return _run_gate_cli_watchdog(
                "functional-wave",
                cycle=args.cycle,
                command_started=_GATE_COMMAND_ENTRY,
            )
        return run_functional_wave(args.cycle)
    if args.lane == "functional":
        parser.error(
            "the monolithic functional lane is retired; use functional-wave --cycle {1,2}"
        )
    if args.lane == "validate-evidence":
        if args.evidence is None:
            parser.error("validate-evidence requires --evidence")
        if args.cycle is not None:
            parser.error("validate-evidence does not accept --cycle")
        record = strict_json_load(args.evidence)
        if args.final:
            package_passed = record.get("lanes", {}).get("package", {}).get("status") == "PASS"
            if package_passed and (args.package_result is None or args.wheel is None):
                parser.error("passed-package validation requires --package-result and --wheel")
            if not package_passed and (args.package_result is not None or args.wheel is not None):
                parser.error("blocked pre-package validation forbids package/wheel paths")
            validate_final_gate_result(
                record,
                repository_paths=_path_bindings(args.repository, "repository"),
                lane_log_paths=_path_bindings(args.log, "log"),
                package_result_path=args.package_result,
                request_paths=_path_bindings(args.request, "request"),
                wheel_path=args.wheel,
            )
        else:
            if any(
                value
                for value in (
                    args.repository,
                    args.request,
                    args.log,
                    args.package_result,
                    args.wheel,
                    args.wheel_output,
                    args.output,
                    args.cycle,
                )
            ):
                parser.error("external bindings require --final")
            validate_gate_result(record)
        return 0
    if any(
        value
        for value in (
            args.output,
            args.evidence,
            args.final,
            args.repository,
            args.request,
            args.log,
            args.package_result,
            args.wheel,
            args.wheel_output,
            args.cycle,
        )
    ):
        parser.error("evidence/package options are invalid for this lane")
    if args.lane == "ci" and not watchdog_child:
        return _run_gate_cli_watchdog(
            "ci", cycle=None, command_started=_GATE_COMMAND_ENTRY
        )
    lanes = inventory()
    if args.lane == "ci":
        return _run_bounded_ci(lanes)
    else:
        selected = lanes[args.lane]
    if not selected:
        raise SystemExit(f"burn-in lane {args.lane!r} is empty")
    returncode = _run_pytest_lane(args.lane, selected)
    if returncode:
        return returncode
    if args.lane == "performance":
        from measure_e4_pl_q1m_baseline import collect_performance_observation

        observation = collect_performance_observation()
        payload = canonical_json_bytes(observation).rstrip(b"\n")
        sys.stdout.buffer.write(PERFORMANCE_BASELINE_MARKER + payload + b"\n")
        sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
