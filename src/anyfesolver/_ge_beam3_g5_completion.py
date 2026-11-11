"""Strict G5 integration evidence vocabulary; contains no mechanics."""
from hashlib import sha256

from ._ge_beam3_p5_seeded.core import canonical


POLICY = "GE_BEAM3_G5_GENERAL_STATIC_INTEGRATION_V1"
TERMINAL = "PROVISIONAL_GO_GE_BEAM3_G5_GENERAL_STATIC_INTEGRATION_ONLY"
PRODUCTION_RESTRICTION = "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"

ROUTES = (
    "LEGACY_B2_REFERENCE_ANALYSIS",
    "LEGACY_B3_REFERENCE_ANALYSIS",
    "GE_REFERENCE_LOAD_INERTIA",
    "GE_REFERENCE_MODAL_BUCKLING_TRANSIENT",
    "GE_NATIVE_STATIC_RESTART_RECOVERY",
    "UNSUPPORTED_ROUTES_FAIL_CLOSED",
    "SELECTOR_AND_PROVENANCE_BOUNDARY",
)


def evidence(route, **checks):
    """Return one immutable, canonical-friendly route record."""
    if type(route) is not str or route not in ROUTES:
        raise ValueError("registered G5 route required")
    if not checks or any(type(name) is not str or type(value) is not bool
                         for name, value in checks.items()):
        raise ValueError("nonempty Boolean G5 checks required")
    if not all(checks.values()):
        raise ValueError("failed G5 route cannot be recorded as evidence")
    body = dict(policy=POLICY, route=route, checks=dict(sorted(checks.items())),
                production_qualified=False)
    return {**body, "record_sha256": sha256(canonical(body)).hexdigest()}


def adjudicate(records, *, installed_artifact):
    """Adjudicate a complete unique route set and an external wheel witness."""
    if type(records) is not list or any(type(row) is not dict for row in records):
        raise ValueError("G5 records must be a list of mappings")
    by_route = {row.get("route"): row for row in records}
    if len(by_route) != len(records) or tuple(sorted(by_route)) != tuple(sorted(ROUTES)):
        raise ValueError("complete unique G5 route inventory required")
    for route, row in by_route.items():
        body = {key: value for key, value in row.items() if key != "record_sha256"}
        if (set(row) != {"policy", "route", "checks", "production_qualified",
                         "record_sha256"}
                or row["policy"] != POLICY or row["route"] != route
                or row["production_qualified"] is not False
                or row["record_sha256"] != sha256(canonical(body)).hexdigest()
                or type(row["checks"]) is not dict or not row["checks"]
                or any(type(value) is not bool or not value
                       for value in row["checks"].values())):
            raise ValueError("invalid G5 route record")
    if (type(installed_artifact) is not dict
            or set(installed_artifact) != {"wheel_sha256", "probe_sha256",
                                           "replicas_byte_identical"}
            or type(installed_artifact["wheel_sha256"]) is not str
            or len(installed_artifact["wheel_sha256"]) != 64
            or type(installed_artifact["probe_sha256"]) is not str
            or len(installed_artifact["probe_sha256"]) != 64
            or installed_artifact["replicas_byte_identical"] is not True):
        raise ValueError("complete installed-artifact witness required")
    return dict(policy=POLICY, terminal=TERMINAL,
                closed_rows=["S25", "S26"], g5_qualified=True,
                general_static_integration_closed=True,
                installed_artifact=installed_artifact,
                production_qualified=False,
                production_restriction=PRODUCTION_RESTRICTION)
