"""Explicit development factory for one native beam plus one S3 descriptor."""
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_shell_contact import NativeShellContactAnalysis
from .native_retained_line_s3 import NativeRetainedRoundLineS3Interior
from .native_retained_line_s3_work import NativeRetainedLineS3Work
from .s3_skin_work import S3SkinWorkAdapter
from .overlap import reference_shell_face

SELECTOR='ge_beam3_native_s3_small_motion_transient_dev'


def create_native_transient_shell_analysis(formulation,definitions,boundaries,shell_model,*,
        shell_body_id,shell_element_id,shell_side,radius_m,line_stiffness_n_per_m2,
        rim_clearance_m,geometry_tolerance_m,energy_tolerance_j,force_tolerance_n,
        tangent_tolerance_n_per_m,maximum_integration_intervals,maximum_penetration_m,
        maximum_swept_intervals,maximum_shell_displacement_m,maximum_shell_rotation_rad,
        inertia_order=24,max_iterations=12,max_backtracks=8):
    if formulation!=SELECTOR:raise ValueError('exact native/S3 small-motion development selector required')
    if (type(definitions) is not tuple or len(definitions)!=1
            or type(definitions[0]) is not NativeBeamDefinition):
        raise ValueError('one exact immutable native definition required')
    if shell_side!='top':raise ValueError('native/S3 development profile requires top physical skin')
    element,_=definitions[0].instantiate()
    skin=S3SkinWorkAdapter.from_model(shell_model,shell_body_id,shell_element_id,shell_side)
    kernel=NativeRetainedRoundLineS3Interior(element.element_id,element.operator.reference,element.node_ids,
        shell_element_id,reference_shell_face(shell_model,shell_body_id,shell_element_id,shell_side),skin.node_ids,
        radius_m=radius_m,line_stiffness_n_per_m2=line_stiffness_n_per_m2,
        rim_clearance_m=rim_clearance_m,geometry_tolerance_m=geometry_tolerance_m,
        energy_tolerance_j=energy_tolerance_j,force_tolerance_n=force_tolerance_n,
        tangent_tolerance_n_per_m=tangent_tolerance_n_per_m,
        maximum_integration_intervals=maximum_integration_intervals)
    return NativeShellContactAnalysis(definitions,boundaries,shell_model,NativeRetainedLineS3Work(kernel,skin),
        maximum_penetration_m=maximum_penetration_m,maximum_swept_intervals=maximum_swept_intervals,
        maximum_shell_displacement_m=maximum_shell_displacement_m,
        maximum_shell_rotation_rad=maximum_shell_rotation_rad,inertia_order=inertia_order,
        max_iterations=max_iterations,max_backtracks=max_backtracks)


def solve_native_transient_shell(analysis,config,*,checkpoint=None,expected_sha256=None,
        max_steps=10000,max_cutbacks=8,minimum_dt=None,cancellation_token=None):
    if type(analysis) is not NativeShellContactAnalysis:raise TypeError('exact native/S3 development owner required')
    from anyfesolver._ge_beam3_connected_trajectory import _solve_retained_transient
    return _solve_retained_transient(analysis,config,checkpoint=checkpoint,expected_sha256=expected_sha256,
        max_steps=max_steps,max_cutbacks=max_cutbacks,minimum_dt=minimum_dt,cancellation_token=cancellation_token)
