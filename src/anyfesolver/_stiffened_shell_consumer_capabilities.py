"""Detached wire records for the existing experimental stiffened-shell owners."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from math import isfinite

import numpy as np

from ._b2_physical_linear_definition import PROFILE_ID as B2_PROFILE
from ._stiffened_shell_b2_experimental import ExperimentalB2Analysis
from ._stiffened_shell_b2_q4_point_experimental import ExperimentalB2Q4PointAnalysis
from ._stiffened_shell_b2_q4_panel_experimental import (
    ExperimentalB2Q4PanelAnalysis, ExperimentalB2MixedPanelAnalysis,
)
from ._stiffened_shell_b2_nodal_footprint import B2NodalRigidFootprint
from ._stiffened_shell_b2_physical_rigid_footprint import B2PhysicalRigidFootprint
from ._stiffened_shell_ge_experimental import ExperimentalGEAnalysis, PROFILE as GE_PROFILE


SCHEMA = "STIFFENED_SHELL_CONSUMER_CAPABILITIES_V1"
_B2_OWNERS = (ExperimentalB2Analysis, ExperimentalB2Q4PointAnalysis,
              ExperimentalB2Q4PanelAnalysis, ExperimentalB2MixedPanelAnalysis)
_FOOTPRINTS = (B2NodalRigidFootprint, B2PhysicalRigidFootprint)


def _primitive(value, active, depth=0):
    """Copy capability values without guessing representations or losing fields."""
    if depth > 64:
        raise ValueError("capability report nesting exceeds the bounded record depth")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("capability report contains a nonfinite number")
        return value
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biufU":
            raise TypeError("unsupported NumPy capability scalar")
        scalar = value.item()
        if isinstance(scalar, np.generic):
            raise TypeError("capability scalar has no lossless Python representation")
        return _primitive(scalar, active, depth+1)

    identity = id(value)
    if identity in active:
        raise ValueError("cyclic capability report")
    active.add(identity)
    try:
        if type(value) in _FOOTPRINTS:
            # These descriptors carry the established law and admission semantics.
            return _primitive(value.descriptor(), active, depth+1)
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("capability report keys must be strings")
                result[key] = _primitive(item, active, depth+1)
            return result
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: _primitive(getattr(value, field.name), active, depth+1)
                    for field in fields(value)}
        if type(value) in (tuple, list):
            return [_primitive(item, active, depth+1) for item in value]
        if type(value) is np.ndarray:
            return _primitive(value.tolist(), active, depth+1)
        raise TypeError("unsupported capability report value")
    finally:
        active.remove(identity)


def _check_claims(value, profile):
    if type(value) is dict:
        for key, item in value.items():
            if key in ("profile", "profile_id") and item != profile:
                raise ValueError("capability report conflicts with the owner profile")
            if key in ("production_qualified", "qualified") and item is not False:
                raise ValueError("qualified capability claims are not admitted")
            if key == "status" and type(item) is str and item.upper() in (
                    "QUALIFIED", "PRODUCTION_QUALIFIED", "ACCEPTED", "RELEASED"):
                raise ValueError("qualified capability status is not admitted")
            _check_claims(item, profile)
    elif type(value) is list:
        for item in value:
            _check_claims(item, profile)


def capabilities_record(analysis):
    """Read only an existing owner's capabilities; never compile or run requests."""
    if type(analysis) is ExperimentalGEAnalysis:
        profile = GE_PROFILE
    elif type(analysis) in _B2_OWNERS:
        profile = B2_PROFILE
    else:
        raise TypeError("an existing stiffened-shell GE or B2 owner is required")
    report = _primitive(analysis.capabilities, set())
    if type(report) is not dict:
        raise TypeError("a mapping or dataclass capability report is required")
    profiles = [report[key] for key in ("profile", "profile_id") if key in report]
    if not profiles or any(value != profile for value in profiles):
        raise ValueError("explicit matching capability profile required")
    if report.get("status") not in ("experimental", "EXPERIMENTAL_NOT_QUALIFIED"):
        raise ValueError("explicit experimental capability status required")
    _check_claims(report, profile)
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "profile": profile,
        "status": "EXPERIMENTAL_NOT_QUALIFIED",
        "raw_status": report["status"],
        "raw_report": report,
        "qualification": {
            "production_qualified": False,
            "accepted_evidence": {"status": "UNAVAILABLE", "ids": []},
        },
        "platform_qualification": {"status": "NOT_BOUND"},
        "resource_qualification": {"status": "NOT_BOUND"},
        "resources": {"construction": "create_analysis(resources=None)",
                      "resources": None,
                      "scope": "owner_resource_argument_admission_only"},
        "admission": {"authority": "existing_typed_owner_preflight_and_request_admission",
                      "capability_authority": "raw_report"},
        "exports": {"status": "NO_EXPORT_OPERATION_SUPPLIED",
                    "scope": "this_capability_record_only"},
    }
