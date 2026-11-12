"""Owner-observed static request identity, not a qualification certificate.

The record identifies files and serialized inputs. It does not prove that a
solve ran, validate all live Python bytecode, or confer application admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

from ..current_state_tangent import require_exact_qualified_component_lifecycle_api as _guard
from ..fe_core import FEModel
from ..nonlinear_restart import (
    canonical_checkpoint_json_bytes, load_nonlinear_checkpoint,
    model_configuration_fingerprint, load_case_descriptor,
)
from .core import (ContactDefinition, ContactSolverSettings, validate_general_contact,
                   _Q4_SOURCE_OVERLAP_DUAL_REVISION,
                   _CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION,
                   _INITIAL_CONTACT_SEED_REVISION,
                   _INITIAL_CONTACT_SEED_GLOBALIZATION_REVISION)
from .static_path import ContactPathProgram, _program_from_dead_nodal_load_case


class ContactRequestIdentityError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _package_manifest():
    root = Path(__file__).resolve().parents[1]
    for name, module in tuple(sys.modules.items()):
        if name == "anyfesolver" or name.startswith("anyfesolver."):
            filename = vars(module).get("__file__") if module is not None else None
            if filename and not Path(filename).resolve().is_relative_to(root):
                raise ContactRequestIdentityError("mixed_source", "loaded ANYfeSolver modules use different package roots")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".py", ".pyd", ".so", ".dll"}:
            if not path.resolve().is_relative_to(root):
                raise ContactRequestIdentityError("mixed_source", "package file resolves outside its source root")
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "__init__.py" not in files:
        raise ContactRequestIdentityError("source_unavailable", "complete file-backed ANYfeSolver package required")
    return files


@dataclass(frozen=True)
class ContactRequestIdentity:
    """Immutable request snapshot; construction carries no evidence authority."""

    canonical_request: bytes

    def __post_init__(self):
        if type(self.canonical_request) is not bytes:
            raise ContactRequestIdentityError("invalid_identity", "immutable canonical request bytes required")
        try:
            value = json.loads(self.canonical_request)
            valid = (type(value) is dict and value.get("schema") == "ANYSOLVER_STATIC_CONTACT_REQUEST_V1"
                     and canonical_checkpoint_json_bytes(value) == self.canonical_request)
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise ContactRequestIdentityError("invalid_identity", "canonical static contact request required")

    @property
    def request_sha256(self):
        return hashlib.sha256(self.canonical_request).hexdigest()

    @property
    def admitted_for_application(self):
        return False

    def to_dict(self):
        return {"schema": "ANYSOLVER_CONTACT_REQUEST_IDENTITY_V1",
                "purpose": "request_identity_only", "admitted_for_application": False,
                "request_sha256": self.request_sha256,
                "request": json.loads(self.canonical_request)}


def identify_general_contact_request(
    model, definition, settings, *, load_case=None, contact_path_program=None,
    num_steps=10, max_load_factor=1.0, max_iterations=25, tolerance=1e-6,
    restart_checkpoint=None, emit_restart_checkpoint=False,
):
    """Observe one static developer request without solving or admission.

    This first binding supports an explicit public contact path or dead nodal
    loads. Other load/control routes are not silently converted. Definition,
    settings and numerical controls are explicit. Model fingerprints use the
    existing guarded checkpoint owner. A checkpoint is integrity-checked here;
    its model/analysis compatibility remains the actual solver's responsibility.
    File hashes identify package bytes, not Git names supplied by callers.
    Dependency versions are observations, not a complete environment certificate.
    """
    if (type(model) is not FEModel or type(definition) is not ContactDefinition
            or type(settings) is not ContactSolverSettings):
        raise ContactRequestIdentityError("invalid_request", "exact model, contact definition and settings required")
    if ((load_case is None) == (contact_path_program is None)):
        raise ContactRequestIdentityError("invalid_loading", "select exactly one dead load case or contact path")
    if (type(num_steps) is not int or num_steps < 1
            or type(max_iterations) is not int or max_iterations < 1
            or type(emit_restart_checkpoint) is not bool
            or type(max_load_factor) not in (float, int)
            or not np.isfinite(max_load_factor) or not 0. < max_load_factor <= 1.
            or type(tolerance) not in (float, int)
            or not np.isfinite(tolerance) or tolerance <= 0.):
        raise ContactRequestIdentityError("invalid_controls", "finite explicit static solve controls required")
    _guard(model, context="contact request before observation")
    validation = validate_general_contact(model, definition, settings,
        analysis_control="contact_path" if contact_path_program is not None else "force")
    if not validation.valid:
        raise ContactRequestIdentityError("unsupported_profile", "; ".join(validation.issues))
    if contact_path_program is not None:
        if type(contact_path_program) is not ContactPathProgram:
            raise ContactRequestIdentityError("invalid_loading", "exact public contact path required")
        loading = {"kind": "contact_path", "program": contact_path_program.contract()}
    else:
        program = _program_from_dead_nodal_load_case(load_case, num_steps)
        loading = {"kind": "dead_nodal_load", "program": program.contract(),
                   "load_case": load_case_descriptor(load_case)}
    # The canonical loader verifies the complete checkpoint hash rather than
    # trusting a caller-provided hash string or dropping its histories.
    checkpoint = None if restart_checkpoint is None else load_nonlinear_checkpoint(
        restart_checkpoint, _exact_guard=_guard, _guard_model=model)
    files = _package_manifest()
    settings_record = asdict(settings)
    if settings_record.get('q4_source_overlap_dual_dev') is False:
        settings_record.pop('q4_source_overlap_dual_dev')
    if settings_record.get('initial_contact_seed') == 'none':
        # Default-none keeps the exact legacy serialized request shape.
        settings_record.pop('initial_contact_seed')
    if settings_record.get('path_continuation_maximum_halvings') == 0:
        # Explicit OFF omits the whole continuation block so prior
        # serialized request identities keep their exact shape.
        for name in ('path_continuation_maximum_halvings',
                     'path_stagnation_patience', 'path_stagnation_fraction',
                     'path_stagnation_merit_rate'):
            settings_record.pop(name, None)
    request = {
        "schema": "ANYSOLVER_STATIC_CONTACT_REQUEST_V1",
        "solver_package_files": files,
        "solver_package_sha256": hashlib.sha256(canonical_checkpoint_json_bytes(files)).hexdigest(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__},
        "model_fingerprint": model_configuration_fingerprint(model, _exact_guard=_guard),
        "definition": asdict(definition), "settings": settings_record,
        **({'contact_numerical_method_revision':_Q4_SOURCE_OVERLAP_DUAL_REVISION}
           if settings.q4_source_overlap_dual_dev else {}),
        **({'contact_path_continuation_revision':
            _CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION}
           if settings.path_continuation_maximum_halvings else {}),
        **({'contact_initial_seed_revision': (
               _INITIAL_CONTACT_SEED_GLOBALIZATION_REVISION
               if settings.initial_contact_seed ==
               'linearized_kkt_merit_globalization_dev'
               else _INITIAL_CONTACT_SEED_REVISION)}
           if settings.initial_contact_seed != 'none' else {}),
        "implemented_profile": validation.profile, "loading": loading,
        "solve_controls": {"max_load_factor": max_load_factor,
                           "max_iterations": max_iterations, "tolerance": tolerance,
                           "num_steps": num_steps if contact_path_program is None else None,
                           "emit_restart_checkpoint": emit_restart_checkpoint},
        "restart_checkpoint_sha256": None if checkpoint is None else checkpoint["checkpoint_sha256"],
    }
    canonical = canonical_checkpoint_json_bytes(request, _exact_guard=_guard, _guard_model=model)
    _guard(model, context="contact request after observation")
    return ContactRequestIdentity(canonical)


def verify_general_contact_request(identity, model, definition, settings, **controls):
    """Reject changed files or inputs; a match is not acceptance or solve proof."""
    if type(identity) is not ContactRequestIdentity:
        raise ContactRequestIdentityError("invalid_identity", "exact request identity required")
    current = identify_general_contact_request(model, definition, settings, **controls)
    if current.canonical_request != identity.canonical_request:
        previous, observed = json.loads(identity.canonical_request), json.loads(current.canonical_request)
        changed = sorted(key for key in set(previous) | set(observed) if previous.get(key) != observed.get(key))
        raise ContactRequestIdentityError("request_mismatch", "contact request changed: " + ", ".join(changed))
    return current


__all__ = ["ContactRequestIdentity", "ContactRequestIdentityError",
           "identify_general_contact_request", "verify_general_contact_request"]
