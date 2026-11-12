"""Production-readiness artifacts tied to verification release gates.

These helpers implement the safe part of the post-verification production plan:
capability and scope reporting.  They deliberately do not create a qualification
tag or claim a production release while verification gates remain blocked.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .beam_shell_verification import run_beam_shell_verification


DEFAULT_PRODUCTION_READINESS_DIR = Path("reports/production_readiness/current")


@dataclass(frozen=True)
class CapabilityEntry:
    feature: str
    status: str
    release_gate: str
    verification_cases: List[str]
    limits: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    required_solver_configuration: Dict[str, Any] = field(default_factory=dict)
    gate_blockers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _gate(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    gates = report.get("release_gates") or {}
    value = gates.get(name) or {}
    return value if isinstance(value, Mapping) else {}


def _gate_blockers(report: Mapping[str, Any], name: str) -> List[str]:
    return [str(item.get("case_id")) for item in (_gate(report, name).get("blockers") or []) if item.get("case_id")]


def _status_from_gate(report: Mapping[str, Any], name: str) -> str:
    status = str(_gate(report, name).get("status", "not_evaluated"))
    if status == "passed":
        return "qualified"
    if status == "blocked":
        return "not_qualified"
    return "not_evaluated"


def _qualified_s3_companion_entry() -> CapabilityEntry:
    """Report the qualified V2D companion selected for current models."""

    return CapabilityEntry(
        feature="qualified_s3_companion_shell",
        status="qualified",
        release_gate="qualified_s3_companion_activation",
        verification_cases=["S3-V6W"],
        limits={
            "default_formulation": "e4-pl-s3-v2d",
            "explicit_selectors": [
                "e4-pl-s3",
                "qualified-s3",
                "e4-pl-s3-v2d",
                "qualified-s3-v2d",
            ],
            "formulation_id": "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1",
            "shell_topology": "S3",
        },
        limitations=[
            "Historical S3 records without formulation identity remain legacy-s3.",
            "Switching a history-bearing model requires complete load-history replay.",
            "Qualified construction requires authoritative physical owner-normal input.",
        ],
        required_solver_configuration={
            "authoritative_reference_normal": True,
            "formulation": "e4-pl-s3-v2d",
            "geometry_admission": "ANYMESHER_QUALIFIED_S3_ADMISSION_V1",
        },
        gate_blockers=[],
    )


def build_capability_matrix(report: Optional[Mapping[str, Any]] = None) -> List[CapabilityEntry]:
    """Return production capability entries derived from verification gates."""

    report = report or run_beam_shell_verification()
    flat_shell = _status_from_gate(report, "flat_thin_shell")
    flat_stiffened = _status_from_gate(report, "flat_thin_stiffened_shell")
    curved_stiffened = _status_from_gate(report, "curved_thin_stiffened_shell")
    nonlinear = _status_from_gate(report, "nonlinear_capacity")
    contact = _status_from_gate(report, "contact")
    fracture = _status_from_gate(report, "simplified_fracture")
    full_release = _status_from_gate(report, "fully_documented_verified_release")

    return [
        CapabilityEntry(
            feature="flat_thin_shell_linear_static_modal_buckling",
            status=flat_shell,
            release_gate="flat_thin_shell",
            verification_cases=["SHELL-009", "SHELL-010", "SHELL-011"],
            limits={
                "shell_formulations": ["Q4", "Q8"],
                "units": "SI",
                "geometry": "flat thin plates within verified mesh and thickness ranges",
            },
            limitations=[
                "Q8R is experimental pending thickness-robust hourglass and inertia qualification.",
                "Follower pressure is a nonlinear-static/arc-length capability and is outside this linear-analysis scope.",
                "Arbitrary contact and unverified material laws are unsupported.",
            ],
            required_solver_configuration={"analysis": ["linear_static", "modal", "linear_buckling"]},
            gate_blockers=_gate_blockers(report, "flat_thin_shell"),
        ),
        CapabilityEntry(
            feature="flat_thin_stiffened_shell_beam_shell_mpc",
            status=flat_stiffened,
            release_gate="flat_thin_stiffened_shell",
            verification_cases=["COUP-012", "COUP-013", "COUP-014", "COUP-015", "COUP-016", "COUP-017"],
            limits={
                "shell_formulations": ["Q4", "Q8"],
                "beam_formulations": ["B2"],
                "coupling": "interpolated eccentric MPC",
                "stiffeners": "beam stiffeners inside verified eccentricity and mesh-ratio ranges",
            },
            limitations=[
                "Q8R is experimental pending thickness-robust hourglass and inertia qualification.",
                "Equivalent all-shell model-pair checks must pass before production qualification.",
            ],
            required_solver_configuration={"member_model": "plates as shell, stiffeners/girders as beams"},
            gate_blockers=_gate_blockers(report, "flat_thin_stiffened_shell"),
        ),
        CapabilityEntry(
            feature="curved_thin_stiffened_shell",
            status=curved_stiffened,
            release_gate="curved_thin_stiffened_shell",
            verification_cases=["SHELL-008", "COUP-020", "COUP-021", "CYL-001", "CYL-002", "CYL-003"],
            limits={"geometry": "cylindrical and ring-stiffened shells within verified curvature ranges"},
            limitations=[
                "Curved scope is qualified only for the V3 cylindrical/curved-panel fixtures and documented mesh/curvature ranges."
            ],
            required_solver_configuration={"geometry": ["cylinder", "curved_panel"]},
            gate_blockers=_gate_blockers(report, "curved_thin_stiffened_shell"),
        ),
        CapabilityEntry(
            feature="nonlinear_capacity_and_buckling_stop",
            status=nonlinear,
            release_gate="nonlinear_capacity",
            verification_cases=["NLG-006", "NLG-007", "NLG-008", "MAT-008", "DYN-001"],
            limits={"nonlinear_scope": "monotonic ultimate capacity with verified buckling-stop diagnostics"},
            limitations=[
                "Unrestricted post-buckling continuation is not a production target.",
                "Follower-pressure support is limited to supported shell topologies and requires the NLG-008 load/tangent gate.",
                "Nonsymmetric follower-load eigenvalue pencils remain outside the qualified buckling scope.",
            ],
            required_solver_configuration={"buckling_stop": "required", "post_buckling_target": False},
            gate_blockers=_gate_blockers(report, "nonlinear_capacity"),
        ),
        CapabilityEntry(
            feature="limited_rigid_sphere_shell_contact",
            status=contact,
            release_gate="contact",
            verification_cases=["CONTACT-001", "CONTACT-002", "CONTACT-003", "CONTACT-004", "CONTACT-005", "CONTACT-006", "CONTACT-007", "CONTACT-008", "CONTACT-009", "CONTACT-010", "CONTACT-011", "CONTACT-012"],
            limits={
                "contact_scope": "single rigid sphere to shell midsurface or shell-thickness offset surface",
                "contact_law": "frictionless normal penalty",
                "beam_contact": "not direct; beams respond through existing shell-beam coupling",
                "impact_damage": "optional engineering shell damage/erosion; capacity-based mode is gated by FRACT-007..012",
            },
            limitations=[
                "Arbitrary contact, shell-shell contact, direct beam contact, friction, spin, rolling, crack propagation, material nonlinear impact fracture and FSI are unsupported.",
                "Contact remains linear structural transient dynamics plus nonlinear contact-force iteration.",
            ],
            required_solver_configuration={"analysis": "sphere_impact_transient", "contact_gate": "CONTACT-001..012"},
            gate_blockers=_gate_blockers(report, "contact"),
        ),
        CapabilityEntry(
            feature="simplified_element_erosion_and_impact_damage",
            status=fracture,
            release_gate="simplified_fracture",
            verification_cases=[
                "FRACT-001",
                "FRACT-002",
                "FRACT-003",
                "FRACT-004",
                "FRACT-005",
                "FRACT-006",
                "FRACT-007",
                "FRACT-008",
                "FRACT-009",
                "FRACT-010",
                "FRACT-011",
                "FRACT-012",
            ],
            limits={
                "static_fracture_scope": "nonlinear static equivalent-plastic-strain erosion only",
                "impact_damage_scope": "linear sphere-shell transient with engineering capacity-based shell softening/erosion",
                "erosion": "post-converged-increment or post-converged-contact-substep residual-stiffness element erosion",
            },
            limitations=[
                "Not crack propagation, cohesive fracture, remeshing or validated fracture mechanics.",
                "Direct member separation, node deletion, MPC deletion and material nonlinear impact fracture are unsupported.",
            ],
            required_solver_configuration={"analysis": ["nonlinear_static", "sphere_impact_transient"], "fracture_gate": "FRACT-001..012"},
            gate_blockers=_gate_blockers(report, "simplified_fracture"),
        ),
        CapabilityEntry(
            feature="fully_documented_verified_release",
            status=full_release,
            release_gate="fully_documented_verified_release",
            verification_cases=["V1:*", "V2:*", "V3:*", "V4:*", "V5:*", "MLBC:*", "CONTACT:*", "FRACTURE:*"],
            limits={"claim": "documented ANYfeSolver beam-shell analysis scope only"},
            limitations=[
                "General-purpose commercial-solver replacement claims are prohibited.",
                "External-reference decks are reproducible handoff artifacts unless executed external-solver result files are supplied.",
            ],
            required_solver_configuration={"qualification_evidence": "complete immutable release package"},
            gate_blockers=_gate_blockers(report, "fully_documented_verified_release"),
        ),
        _qualified_s3_companion_entry(),
        CapabilityEntry(
            feature="unsupported_general_purpose_fe",
            status="unsupported",
            release_gate="none",
            verification_cases=[],
            limits={},
            limitations=[
                "Arbitrary CAD topology, arbitrary contact beyond limited sphere-shell impact, fluid-structure interaction, unsupported residual-stress fields and unverified element distortion levels are outside scope.",
            ],
            required_solver_configuration={"expert_override": "not sufficient for production qualification"},
        ),
    ]


def build_verification_scope_statement(report: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return a concise production scope statement from the live capability matrix."""

    report = report or run_beam_shell_verification()
    matrix = build_capability_matrix(report)
    by_status: Dict[str, List[str]] = {}
    for entry in matrix:
        by_status.setdefault(entry.status, []).append(entry.feature)
    return {
        "schema_version": 1,
        "production_release_status": "qualified"
        if _status_from_gate(report, "fully_documented_verified_release") == "qualified"
        else "not_qualified",
        "verified": by_status.get("qualified", []),
        "conditionally_supported": by_status.get("not_qualified", []) + by_status.get("not_evaluated", []),
        "experimental": [],
        "unsupported": by_status.get("unsupported", []),
        "explicit_limitations": [
            (
                "Follower pressure is limited to nonlinear static and arc-length equilibrium on supported "
                "shell elements; nonsymmetric follower-load eigenvalue pencils are unsupported."
            ),
            "Arbitrary contact beyond limited rigid-sphere-to-shell contact.",
            "Fracture outside simplified nonlinear-static erosion and limited sphere-impact shell damage.",
            "Fluid-structure interaction.",
            "Unrestricted post-buckling continuation.",
            "Arbitrary general-purpose CAD topology.",
            "Unverified material laws.",
            "Unsupported residual-stress fields.",
            "Unsupported element distortion levels.",
            "External-reference decks without executed external-solver result comparisons.",
        ],
        "release_gates": {
            name: {
                "status": gate.get("status"),
                "blockers": [item.get("case_id") for item in (gate.get("blockers") or [])],
            }
            for name, gate in (report.get("release_gates") or {}).items()
            if name != "thin_stiffened_shell"
        },
    }


def _markdown_list(items: Iterable[str]) -> List[str]:
    values = [str(item) for item in items]
    return [f"- {item}" for item in values] if values else ["- None"]


def scope_statement_markdown(scope: Mapping[str, Any]) -> str:
    lines = [
        "# ANYfeSolver Production Scope Statement",
        "",
        f"- Production release status: {scope.get('production_release_status')}",
        "",
        "## Verified",
        "",
        *_markdown_list(scope.get("verified") or []),
        "",
        "## Conditionally Supported",
        "",
        *_markdown_list(scope.get("conditionally_supported") or []),
        "",
        "## Experimental",
        "",
        *_markdown_list(scope.get("experimental") or []),
        "",
        "## Unsupported",
        "",
        *_markdown_list(scope.get("unsupported") or []),
        "",
        "## Explicit Limitations",
        "",
        *_markdown_list(scope.get("explicit_limitations") or []),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_production_readiness_artifacts(
    output_dir: Path | str = DEFAULT_PRODUCTION_READINESS_DIR,
    *,
    report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write capability matrix and scope-statement artifacts."""

    report = report or run_beam_shell_verification()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    matrix = [entry.to_dict() for entry in build_capability_matrix(report)]
    scope = build_verification_scope_statement(report)

    matrix_path = output_path / "capability_matrix.json"
    scope_json_path = output_path / "verification_scope.json"
    scope_md_path = output_path / "verification_scope.md"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")
    scope_json_path.write_text(json.dumps(scope, indent=2, sort_keys=True), encoding="utf-8")
    scope_md_path.write_text(scope_statement_markdown(scope), encoding="utf-8")
    return {
        "output_dir": str(output_path),
        "capability_matrix": str(matrix_path),
        "verification_scope_json": str(scope_json_path),
        "verification_scope_markdown": str(scope_md_path),
        "production_release_status": scope["production_release_status"],
        "capability_count": len(matrix),
    }
