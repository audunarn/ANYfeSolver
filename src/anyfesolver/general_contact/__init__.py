"""Opt-in contact definitions and coupled static contact profiles."""

from importlib import import_module


_CORE_EXPORTS = frozenset({
    "ContactDefinition", "ContactInteraction", "ContactResults",
    "ContactSolverSettings", "ContactSurface", "ContactValidationReport",
    "RigidContactBody", "validate_general_contact",
})
_CAPABILITY_EXPORTS = frozenset({
    "ContactCapabilityReport", "report_general_contact_capability",
})
_STATIC_EXPORTS = frozenset({
    "ContactPathControl", "ContactPathNodalForce", "ContactPathProgram",
    "solve_static_contact_path",
})
_REQUEST_EXPORTS = frozenset({
    "ContactRequestIdentity", "ContactRequestIdentityError",
    "identify_general_contact_request", "verify_general_contact_request",
})
_SKIN_EXPORTS = frozenset({
    "ContactSkinGeometryError", "ContactShellSkinFace", "ContactShellSkinGeometry",
    "recover_general_contact_shell_skins",
})
_TRANSIENT_EXPORTS = frozenset({
    "ContactNodalLoadHistory", "ContactTransientSolverControls",
    "TransientContactResults", "TransientContactSettings",
    "solve_transient_contact",
})
_NATIVE_EXPORTS = frozenset({'NativeS3ContactPair', 'create_native_contact_analysis'})
_NATIVE_TRANSIENT_PAIR_EXPORTS = frozenset({'create_native_transient_pair_analysis',
    'solve_native_transient_pair'})
_NATIVE_TRANSIENT_SHELL_EXPORTS = frozenset({'create_native_transient_shell_analysis',
    'solve_native_transient_shell'})
_NATIVE_TRANSIENT_MULTI_SHELL_EXPORTS = frozenset({
    'NativeContactBody', 'S3ContactBody', 'NativeTransientS3ContactPair',
    'create_native_transient_multi_shell_analysis', 'solve_native_transient_multi_shell',
})
_ARCHIVE_EXPORTS = frozenset({'TransientContactArchiveError','ContactTransientArchive',
    'serialize_transient_contact_results','read_transient_contact_results'})
_STATIC_ARCHIVE_EXPORTS = frozenset({'StaticContactArchiveError','ContactStaticArchive',
    'serialize_static_contact_results','read_static_contact_results'})
_ARC_ARCHIVE_EXPORTS = frozenset({'ArcContactArchiveError','ContactArcArchive',
    'serialize_arc_contact_results','read_arc_contact_results'})
_NATIVE_RESULTS_EXPORTS = frozenset({'NativeContactCheckpointResults',
    'NativeContactCheckpointResultError', 'read_native_contact_checkpoint_results'})
_AUTOMATIC_EXPORTS = frozenset({
    'AutomaticContactDomainError', 'AutomaticContactParticipant',
    'AutomaticContactDefaults', 'AutomaticContactDomain',
    'AutomaticContactPairSelection', 'AutomaticContactResolution',
    'resolve_automatic_contact_domain', 'verify_automatic_contact_resolution',
})


def __getattr__(name: str):
    if name in {"S3SlidingFrictionSettings", "S3MovingNormalFrictionSettings", "S3MeshMovingNormalFrictionSettings", "S3MeshWeakPressureFrictionSettings", "S3MeshWholeChordGapFrictionSettings", "S3MeshHistoryPathFrictionSettings"}:
        module = "s3_sliding_friction"
    elif name == "TriangulatedRigidContactBody":
        module = "triangulated_rigid"
    elif name == "RigidCappedCylinder":
        module = "capped_cylinder"
    elif name in _CORE_EXPORTS:
        module = "core"
    elif name in _CAPABILITY_EXPORTS:
        module = "capability"
    elif name in _STATIC_EXPORTS:
        module = "static_path"
    elif name in _REQUEST_EXPORTS:
        module = "request_identity"
    elif name in _SKIN_EXPORTS:
        module = "skin_geometry"
    elif name in _TRANSIENT_EXPORTS:
        module = "transient"
    elif name in _NATIVE_EXPORTS:
        module = "native_multi_s3_static"
    elif name in _NATIVE_TRANSIENT_PAIR_EXPORTS:
        module = 'native_transient_pair'
    elif name in _NATIVE_TRANSIENT_SHELL_EXPORTS:
        module = 'native_transient_shell'
    elif name in _NATIVE_TRANSIENT_MULTI_SHELL_EXPORTS:
        module = 'native_transient_multi_shell'
    elif name in _ARCHIVE_EXPORTS:
        module = 'transient_results_archive'
    elif name in _STATIC_ARCHIVE_EXPORTS:
        module = 'static_results_archive'
    elif name in _ARC_ARCHIVE_EXPORTS:
        module = 'arc_results_archive'
    elif name in _NATIVE_RESULTS_EXPORTS:
        module = 'native_checkpoint_results'
    elif name in _AUTOMATIC_EXPORTS:
        module = 'automatic_domain'
    elif name == 'NativeContactActivationSettings':
        module = 'native_activation_static'
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "S3SlidingFrictionSettings",
    "S3MovingNormalFrictionSettings",
    "S3MeshMovingNormalFrictionSettings",
    "S3MeshWeakPressureFrictionSettings",
    "S3MeshWholeChordGapFrictionSettings",
    "S3MeshHistoryPathFrictionSettings",
    'create_native_transient_pair_analysis', 'solve_native_transient_pair',
    'create_native_transient_shell_analysis', 'solve_native_transient_shell',
    "TriangulatedRigidContactBody",
    *sorted(_NATIVE_TRANSIENT_MULTI_SHELL_EXPORTS),
    *sorted(_ARCHIVE_EXPORTS),
    *sorted(_STATIC_ARCHIVE_EXPORTS),
    *sorted(_ARC_ARCHIVE_EXPORTS),
    *sorted(_NATIVE_RESULTS_EXPORTS),
    *sorted(_AUTOMATIC_EXPORTS),
    "NativeS3ContactPair", "create_native_contact_analysis",
    "NativeContactActivationSettings",
    "RigidCappedCylinder",
    "ContactCapabilityReport",
    "ContactDefinition",
    "ContactInteraction",
    "ContactResults",
    "ContactSolverSettings",
    "ContactSurface",
    "ContactValidationReport",
    "RigidContactBody",
    "validate_general_contact",
    "report_general_contact_capability",
    "ContactPathControl",
    "ContactPathNodalForce",
    "ContactPathProgram",
    "solve_static_contact_path",
    "ContactRequestIdentity",
    "ContactRequestIdentityError",
    "identify_general_contact_request",
    "verify_general_contact_request",
    "ContactSkinGeometryError",
    "ContactShellSkinFace",
    "ContactShellSkinGeometry",
    "recover_general_contact_shell_skins",
    "ContactNodalLoadHistory",
    "ContactTransientSolverControls",
    "TransientContactResults",
    "TransientContactSettings",
    "solve_transient_contact",
]
