"""NumPy-only runtime for ANYfeSolver recovery capsules (contracts v1 and v2).

This package reconstructs ANYfeSolver response fields offline.  It is plain
data in, arrays out: no executable or pickle payloads, and no imports from
``anyfesolver``, scipy, ANYfem or ANYpost.  ANYpost resolves dependencies and
calls this runtime; the closed derived-operation registry and the physics
live here.
"""
from __future__ import annotations

from .capsule import (SCHEMA, VERSION, RUNTIME_ID, SUPPORTED_RUNTIME_IDS,
                      SUPPORTED_COMBINATIONS, PreparedResponse, prepare_response,
                      reconstruct_prepared_response, reconstruct_response,
                      validate_capsule)
from .errors import (CapsuleError, CapsuleValidationError, DerivationError,
                     ReconstructionError)
from .registry import OPERATIONS, DERIVATION_EXECUTION_API, derive_field
from .storage_validation import STORAGE_PREDICTION_API, validate_storage_prediction
from .formulas import shell_surface_component

__all__ = [
    'shell_surface_component',
    'STORAGE_PREDICTION_API',
    'validate_storage_prediction',
    'RUNTIME_ID',
    'SUPPORTED_RUNTIME_IDS',
    'SUPPORTED_COMBINATIONS',
    'SCHEMA',
    'VERSION',
    'OPERATIONS',
    'DERIVATION_EXECUTION_API',
    'PreparedResponse',
    'prepare_response',
    'reconstruct_prepared_response',
    'reconstruct_response',
    'derive_field',
    'validate_capsule',
    'CapsuleError',
    'CapsuleValidationError',
    'ReconstructionError',
    'DerivationError',
]
