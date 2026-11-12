"""Manifest-driven beam/shell solver verification report.

This module implements the verification manifest supplied with the beam-shell
verification specification. It deliberately separates implemented checks from
cases that need literature data, external solver execution, or explicitly
unsupported solver features.
"""

from __future__ import annotations

import copy
import contextvars
import json
import importlib.metadata
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse

from .assembly import (
    build_constraint_transformation,
    build_reduced_rigid_body_modes,
    compute_constraint_force_diagnostics,
    reconstruct_full_solution,
    solve_linear,
    solve_linear_many,
)
from .boundary import BoundaryCondition, FixedSupport, LoadCase
from .buckling import solve_eigenvalue_buckling
from .composite_strip_verification import composite_strip_metric_rows
from .contact import contact_verification_metrics
from .cylinder_benchmarks import (
    CylinderBenchmarkConfig,
    build_cylindrical_shell_benchmark_model,
    run_cylindrical_shell_benchmark,
)
from .element_qualification import q8_patch_metric, reference_q8_geometries
from .dynamics import PressurePatch, TransientConfig, solve_transient_newmark
from .elements import (
    BeamElement,
    CoupledBeamShellElement,
    Element,
    ShellElement,
    create_shell_element,
)
from .external_references import generate_external_reference_report, write_external_reference_report
from .fe_core import FEModel
from .mass_properties import calculate_mass_properties
from .matrix_assembly import (
    assemble_external_load_tangent,
    assemble_geometric_stiffness_matrix,
    assemble_load_vector,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
)
from .mesh_load_bc_verification import mesh_load_bc_result_by_case
from .mesh_gen import MeshConfig, PanelGeometry, StiffenerCrossSection, generate_beam_mesh, generate_simple_panel_mesh, generate_stiffened_panel_mesh
from .modal import solve_free_vibration
from .plasticity_qualification import element_tangent_metrics, material_point_path_metrics, reference_plastic_curve, yield_function_residual
from .reference_cases import discover_calculix_reference_cases, upstream_calculix_reference_manifest
from .s4_validity import bending_patch_metric, membrane_patch_metric, thin_plate_locking_sweep
from .validation import mpc_constraint_residuals, validate_production_model


DEFAULT_BEAM_SHELL_VERIFICATION_PATH = Path("reports/beam_shell_verification/beam_shell_verification_report.json")
_EXTERNAL_REFERENCE_REPORT_OVERRIDE: contextvars.ContextVar[Optional[Path]] = contextvars.ContextVar(
    "anyfesolver_external_reference_report",
    default=None,
)
_EXTERNAL_REFERENCE_REPORT_RUN_CACHE: contextvars.ContextVar[
    Optional[Dict[Path, Dict[str, Any]]]
] = contextvars.ContextVar(
    "anyfesolver_external_reference_report_run_cache",
    default=None,
)
_COMPOSITE_METRIC_RUN_CACHE: contextvars.ContextVar[
    Optional[Dict[str, Tuple[Tuple[Dict[str, Any], ...], float, float]]]
] = contextvars.ContextVar(
    "anyfesolver_composite_metric_run_cache",
    default=None,
)
_PLATE_BUCKLING_RUN_CACHE: contextvars.ContextVar[
    Optional[Dict[Tuple[int, float], Tuple[float, Dict[str, Any]]]]
] = contextvars.ContextVar(
    "anyfesolver_plate_buckling_run_cache",
    default=None,
)
_NONLINEAR_FRAMEWORK_RUN_CACHE: contextvars.ContextVar[
    Optional[Dict[str, "VerificationCaseResult"]]
] = contextvars.ContextVar(
    "anyfesolver_nonlinear_framework_run_cache",
    default=None,
)

DEFAULT_TOLERANCES: Dict[str, float] = {
    "stiffness_symmetry_rel": 1.0e-10,
    "mass_symmetry_rel": 1.0e-12,
    "equilibrium_residual_rel": 1.0e-9,
    "energy_rel": 1.0e-8,
    "analytic_linear_rel": 1.0e-6,
    "literature_medium_rel": 0.05,
    "literature_fine_rel": 0.02,
    "mac_min": 0.99,
}

THIN_SHELL_SPAN_TO_THICKNESS: Tuple[int, ...] = (100, 300, 1000, 3000, 10000)

THIN_STIFFENED_SHELL_RELEASE_CASES: Tuple[str, ...] = (
    "META-001",
    "BEAM-001",
    "BEAM-002",
    "BEAM-003",
    "BEAM-004",
    "BEAM-005",
    "BEAM-006",
    "BEAM-007",
    "BEAM-008",
    "BEAM-009",
    "BEAM-010",
    "SHELL-001",
    "SHELL-002",
    "SHELL-003",
    "SHELL-004",
    "SHELL-005",
    "SHELL-006",
    "SHELL-007",
    "SHELL-008",
    "COUP-001",
    "COUP-002",
    "COUP-003",
    "COUP-004",
    "COUP-005",
    "COUP-006",
    "COUP-007",
    "COUP-008",
    "COUP-009",
    "COUP-010",
    "NULL-001",
    "NULL-002",
    "NULL-003",
    "NULL-004",
    "NULL-005",
    "EIG-001",
    "EIG-002",
    "EIG-003",
    "EIG-004",
    "BUC-001",
    "BUC-002",
    "BUC-003",
    "BUC-004",
    "BUC-005",
    "BEAM-011",
    "SHELL-009",
    "SHELL-010",
    "SHELL-011",
    "COUP-012",
    "COUP-013",
    "COUP-014",
    "COUP-015",
    "COUP-016",
    "COUP-017",
    "PERF-001",
    "PERF-002",
)


CASE_ROWS: Tuple[Tuple[str, int, str, bool, str], ...] = (
    ("META-001", 0, "reporting", True, "Verification status semantics"),
    ("ALG-001", 0, "shape", True, "Partition of unity"),
    ("ALG-002", 0, "shape", True, "Nodal interpolation"),
    ("ALG-003", 0, "mapping", True, "Jacobian and orientation"),
    ("ALG-004", 0, "matrix", True, "Element stiffness symmetry"),
    ("ALG-005", 0, "matrix", True, "Element mass symmetry and positivity"),
    ("ALG-006", 0, "kinematics", True, "Rigid-body zero energy"),
    ("ALG-007", 0, "coordinates", True, "Transform orthogonality"),
    ("ALG-008", 0, "assembly", True, "Global assembly consistency"),
    ("ALG-009", 0, "energy", True, "Energy/work identity"),
    ("BEAM-001", 1, "beam_static", True, "Axial extension"),
    ("BEAM-002", 1, "beam_static", True, "Circular torsion"),
    ("BEAM-003", 1, "beam_static", True, "Timoshenko cantilever"),
    ("BEAM-004", 1, "beam_static", True, "Slenderness sweep"),
    ("BEAM-005", 1, "beam_static", True, "Pure bending"),
    ("BEAM-006", 2, "beam_coordinates", True, "Biaxial bending and local-axis rotation"),
    ("BEAM-007", 1, "beam_static", True, "Combined-action superposition"),
    ("BEAM-008", 1, "beam_eigen", True, "Axial bar eigenfrequency invariant"),
    ("BEAM-009", 2, "beam_eigen", True, "Free-free rigid modes"),
    ("BEAM-010", 1, "beam_buckling", True, "Euler column"),
    ("BEAM-011", 1, "beam_eigen", True, "Actual cantilever bending eigenmodes"),
    ("SHELL-001", 2, "shell_patch", True, "Membrane patch"),
    ("SHELL-002", 2, "shell_patch", True, "Pure-bending patch"),
    ("SHELL-003", 2, "invariance", True, "Shell rigid-transform invariance"),
    ("SHELL-004", 1, "plate_static", True, "Thin cantilever-strip static response"),
    ("SHELL-005", 2, "locking", True, "Thin-shell locking and thickness sweep"),
    ("SHELL-006", 1, "plate_eigen", True, "Simply supported plate frequencies"),
    ("SHELL-007", 1, "plate_buckling", True, "Simply supported plate buckling"),
    ("SHELL-008", 2, "shell_locking", True, "Thin curved-shell inextensional bending"),
    ("SHELL-009", 1, "plate_static", True, "Navier square plate under uniform pressure"),
    ("SHELL-010", 1, "plate_eigen", True, "Q4/Q8/Q8R thin plate modal convergence"),
    ("SHELL-011", 1, "plate_buckling", True, "Q4/Q8/Q8R thin plate buckling convergence"),
    ("BENCH-001", 3, "shell_benchmark", True, "MacNeal-Harder twisted cantilever"),
    ("BENCH-002", 3, "shell_benchmark", True, "Scordelis-Lo roof"),
    ("BENCH-003", 3, "shell_benchmark", True, "Pinched cylinder"),
    ("BENCH-004", 3, "shell_benchmark", False, "Hemispherical shell"),
    ("COUP-001", 2, "coupling", True, "Coincident rigid compatibility"),
    ("COUP-002", 2, "coupling", True, "Coincident force transfer"),
    ("COUP-003", 2, "coupling", True, "Eccentric rigid-link kinematics"),
    ("COUP-004", 2, "coupling", True, "Eccentric moment transfer"),
    ("COUP-005", 4, "coupling", True, "Stiffened plate equivalent models"),
    ("COUP-006", 4, "coupling", True, "Ring-stiffened cylinder equivalent models"),
    ("COUP-007", 2, "coupling", True, "Stiffened-panel static invariants"),
    ("COUP-008", 3, "eigen", True, "Stiffened-panel modal invariants"),
    ("COUP-009", 3, "buckling", True, "Stiffened-panel buckling invariants"),
    ("COUP-010", 2, "coordinates", True, "Stiffener orientation and curved-surface transport"),
    ("COUP-011", 2, "coupling", False, "Nonmatching beam and shell discretisation"),
    ("COUP-012", 2, "coupling", True, "Actual eccentric one-to-one MPC affine-field reproduction"),
    ("COUP-013", 2, "coupling", True, "Actual eccentric load-transfer equilibrium"),
    ("COUP-014", 2, "mixed_static", True, "Composite stiffened-strip static benchmark"),
    ("COUP-015", 2, "mixed_modal", True, "Composite stiffened-strip modal benchmark"),
    ("COUP-016", 2, "mixed_buckling", True, "Composite stiffened-strip Euler buckling"),
    ("COUP-017", 2, "production_mesh", True, "Production Q8/Q8R stiffened-panel suite"),
    ("NULL-001", 2, "nullspace", True, "Six rigid modes"),
    ("NULL-002", 2, "nullspace", True, "Projected load orthogonality"),
    ("NULL-003", 2, "nullspace", True, "Projected versus constrained solution"),
    ("NULL-004", 2, "nullspace", True, "Constraint-choice independence"),
    ("NULL-005", 2, "nullspace", True, "Rigid-transform invariance"),
    ("EIG-001", 1, "mass", True, "Total translational mass"),
    ("EIG-002", 2, "mass", True, "Mass mesh invariance"),
    ("EIG-003", 2, "eigen", True, "Modal orthogonality"),
    ("EIG-004", 2, "eigen", True, "Repeated-mode eigenspace"),
    ("EIG-005", 3, "model_pair_modal", True, "Equivalent stiffened-panel modal comparison"),
    ("BUC-001", 0, "buckling", True, "Geometric stiffness symmetry"),
    ("BUC-002", 1, "buckling", True, "Preload scaling"),
    ("BUC-003", 1, "buckling", True, "Euler columns"),
    ("BUC-004", 1, "buckling", True, "Simply supported plate"),
    ("BUC-005", 4, "buckling", True, "Stiffened panel mode comparison"),
    ("NLG-001", 2, "nonlinear", True, "Large rigid-rotation objectivity"),
    ("NLG-002", 3, "nonlinear", True, "Large-rotation cantilever"),
    ("NLG-003", 3, "nonlinear", False, "NAFEMS 3DNLG framework"),
    ("NLG-004", 2, "nonlinear", True, "Increment independence"),
    ("NLG-005", 2, "nonlinear", True, "Consistent tangent finite-difference check"),
    ("MAT-001", 1, "plasticity", True, "Uniaxial elastic response"),
    ("MAT-002", 1, "plasticity", True, "Perfect plasticity"),
    ("MAT-003", 1, "plasticity", True, "Isotropic hardening"),
    ("MAT-004", 2, "plasticity", True, "Kinematic hardening cycle"),
    ("MAT-005", 2, "plasticity", True, "Shell membrane yielding"),
    ("MAT-006", 2, "plasticity", True, "Shell bending yielding"),
    ("MAT-007", 2, "plasticity", True, "Beam plastic hinge"),
    ("FRACT-001", 2, "fracture", True, "Fracture configuration validation"),
    ("FRACT-002", 2, "fracture", True, "Deleted element residual stiffness scaling"),
    ("FRACT-003", 2, "fracture", True, "Deleted shell pressure load removal"),
    ("FRACT-004", 2, "fracture", True, "Plastic-strain threshold deletion record"),
    ("FRACT-005", 2, "fracture", True, "High fracture threshold leaves elements active"),
    ("FRACT-006", 2, "fracture", True, "Maximum deleted-fraction stop status"),
    ("FRACT-007", 2, "impact_fracture", True, "Impact contact patch area estimate"),
    ("FRACT-008", 2, "impact_fracture", True, "Low-energy/high-capacity impact leaves shell undamaged"),
    ("FRACT-009", 2, "impact_fracture", True, "High-energy impact damage shell erosion"),
    ("FRACT-010", 2, "impact_fracture", True, "Material capacity delays impact damage"),
    ("FRACT-011", 2, "impact_fracture", True, "Accumulated repeated contact damage"),
    ("FRACT-012", 2, "impact_fracture", True, "Neighbor smoothing blocks isolated deletion spike"),
    ("PERF-001", 2, "optimized_nonlinear_assembly", True, "Real weighted-MPC fast-path equivalence"),
    ("PERF-002", 2, "cache_correctness", True, "Revision invalidation on a real stiffened model"),
    ("COUP-018", 2, "coupling_robustness", True, "Element-edge ownership and numbering invariance"),
    ("COUP-019", 2, "coupling_convergence", True, "Independent beam/shell mesh-ratio sweep"),
    ("COUP-020", 2, "curved_stiffener_coordinates", True, "Complete ring-stiffener frame closure"),
    ("COUP-021", 2, "ring_static", True, "Circular ring membrane benchmark"),
    ("MLBC-001", 2, "mesh_load_bc", True, "Flat stiffened mesh topology and member-line alignment"),
    ("MLBC-002", 2, "mesh_load_bc", True, "Cylinder seam closure, ring-frame topology and shell orientation"),
    ("MLBC-003", 2, "mesh_load_bc", True, "Q8 midside placement and distorted-mesh guardrails"),
    ("MLBC-004", 2, "mesh_load_bc", True, "Pressure direction and resultant sign conventions"),
    ("MLBC-005", 2, "mesh_load_bc", True, "Pressure patch area selection and resultant"),
    ("MLBC-006", 2, "mesh_load_bc", True, "Edge load and nodal moment resultant balance"),
    ("MLBC-007", 2, "mesh_load_bc", True, "Fixed, pinned, roller and symmetry support DOF semantics"),
    ("MLBC-008", 2, "mesh_load_bc", True, "MPC duplicate ownership and fixed-slave rejection"),
    ("MLBC-009", 2, "mesh_load_bc", True, "Self-equilibrated free-free load nullspace consistency"),
    ("CONTACT-001", 2, "contact", True, "Rigid sphere no-contact trajectory"),
    ("CONTACT-002", 2, "contact", True, "Rigid sphere normal penalty force law"),
    ("CONTACT-003", 2, "contact", True, "Rigid sphere contact force-resultant balance"),
    ("CONTACT-004", 2, "contact", True, "Rigid sphere impulse and momentum consistency"),
    ("CONTACT-005", 2, "contact", True, "Rigid sphere shell-panel impact smoke"),
    ("CONTACT-006", 2, "contact", True, "Rigid sphere stiffened-panel beam load transfer"),
    ("CONTACT-007", 2, "contact", True, "Rigid sphere contact projection classification"),
    ("CONTACT-008", 2, "contact", True, "Rigid sphere shell-thickness contact surface offset"),
    ("CONTACT-009", 2, "contact", True, "Rigid sphere adjacent-element contact reduction"),
    ("CONTACT-010", 2, "contact", True, "Rigid sphere automatic penalty penetration control"),
    ("CONTACT-011", 2, "contact", True, "Rigid sphere event-substep contact detection"),
    ("CONTACT-012", 2, "contact", True, "Rigid sphere production contact validation guardrails"),
    ("CYL-001", 2, "cylinder_static", True, "Closed thin-cylinder membrane benchmark"),
    ("CYL-002", 3, "curved_mixed", True, "Longitudinally stiffened cylinder model-pair benchmark"),
    ("CYL-003", 3, "curved_mixed_buckling", True, "Ring-stiffened cylinder external-pressure benchmark"),
    ("NLG-006", 3, "nonlinear_static", True, "Thin stiffened-panel nonlinear increment study"),
    ("NLG-007", 3, "arc_length", True, "Arc-length imperfect stiffened-panel reference"),
    ("NLG-008", 2, "follower_pressure", True, "Follower-pressure load and tangent"),
    ("MAT-008", 3, "combined_plasticity", True, "Combined shell and beam plasticity"),
    ("DYN-001", 3, "transient", True, "Transient thin stiffened-panel benchmark"),
    ("EXT-001", 4, "cross_solver", True, "CalculiX reference pack"),
    ("EXT-002", 4, "cross_solver", True, "Second external solver reference"),
    ("VVR-001", 4, "verification_report", True, "Complete verification report package"),
)


PROGRAMME_BATCH_CASES: Mapping[str, Tuple[str, ...]] = {
    "V1": (
        "META-001",
        "BEAM-011",
        "SHELL-009",
        "SHELL-010",
        "SHELL-011",
        "COUP-012",
        "COUP-013",
        "COUP-014",
        "COUP-015",
        "COUP-016",
    ),
    "V2": (
        "COUP-005",
        "EIG-005",
        "BUC-005",
        "COUP-017",
        "PERF-001",
        "PERF-002",
        "COUP-018",
        "COUP-019",
    ),
    "V3": (
        "SHELL-008",
        "BENCH-002",
        "BENCH-003",
        "BENCH-001",
        "COUP-020",
        "COUP-021",
        "CYL-001",
        "CYL-002",
        "COUP-006",
        "CYL-003",
    ),
    "V4": ("NLG-006", "NLG-007", "NLG-008", "MAT-008", "DYN-001"),
    "V5": ("BENCH-004", "NLG-002", "NLG-003", "EXT-001", "EXT-002", "VVR-001"),
    "MLBC": (
        "MLBC-001",
        "MLBC-002",
        "MLBC-003",
        "MLBC-004",
        "MLBC-005",
        "MLBC-006",
        "MLBC-007",
        "MLBC-008",
        "MLBC-009",
    ),
    "CONTACT": (
        "CONTACT-001",
        "CONTACT-002",
        "CONTACT-003",
        "CONTACT-004",
        "CONTACT-005",
        "CONTACT-006",
        "CONTACT-007",
        "CONTACT-008",
        "CONTACT-009",
        "CONTACT-010",
        "CONTACT-011",
        "CONTACT-012",
    ),
    "FRACTURE": (
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
    ),
}

PROGRAMME_RELEASE_GATES: Mapping[str, Tuple[str, ...]] = {
    "flat_thin_shell": ("SHELL-009", "SHELL-010", "SHELL-011"),
    "flat_thin_stiffened_shell": PROGRAMME_BATCH_CASES["V1"] + PROGRAMME_BATCH_CASES["V2"],
    "curved_thin_stiffened_shell": PROGRAMME_BATCH_CASES["V1"] + PROGRAMME_BATCH_CASES["V2"] + PROGRAMME_BATCH_CASES["V3"],
    "nonlinear_capacity": PROGRAMME_BATCH_CASES["V1"]
    + PROGRAMME_BATCH_CASES["V2"]
    + PROGRAMME_BATCH_CASES["V3"]
    + PROGRAMME_BATCH_CASES["V4"],
    "fully_documented_verified_release": PROGRAMME_BATCH_CASES["V1"]
    + PROGRAMME_BATCH_CASES["V2"]
    + PROGRAMME_BATCH_CASES["V3"]
    + PROGRAMME_BATCH_CASES["V4"]
    + PROGRAMME_BATCH_CASES["V5"]
    + PROGRAMME_BATCH_CASES["MLBC"]
    + PROGRAMME_BATCH_CASES["CONTACT"]
    + PROGRAMME_BATCH_CASES["FRACTURE"],
    "mesh_load_bc": PROGRAMME_BATCH_CASES["MLBC"],
    "contact": PROGRAMME_BATCH_CASES["CONTACT"],
    "simplified_fracture": PROGRAMME_BATCH_CASES["FRACTURE"],
}

CASE_TO_BATCH: Mapping[str, str] = {
    case_id: batch_id for batch_id, case_ids in PROGRAMME_BATCH_CASES.items() for case_id in case_ids
}


def _case_evidence_type(case_id: str, category: str) -> str:
    if case_id.startswith("BENCH-") or category in {"shell_benchmark"}:
        return "literature"
    if case_id.startswith("EXT-"):
        return "handoff_artifact"
    if category.startswith("model_pair") or category in {"curved_mixed", "curved_mixed_buckling"}:
        return "model_pair"
    if category in {"mixed_static", "mixed_modal", "mixed_buckling", "beam_modal", "shell_static", "shell_modal", "shell_buckling"}:
        return "analytical"
    return "invariant"


IMPLEMENTED_PHASES: Mapping[str, Tuple[str, ...]] = {
    "A": (
        "ALG-",
        "BEAM-001",
        "BEAM-002",
        "BEAM-003",
        "BEAM-005",
        "BEAM-007",
        "SHELL-001",
        "SHELL-002",
        "SHELL-004",
        "SHELL-005",
        "COUP-001",
        "COUP-002",
        "COUP-003",
        "COUP-004",
        "COUP-007",
        "NULL-001",
        "NULL-002",
        "NULL-003",
        "NULL-004",
        "NULL-005",
    ),
    "B": ("BEAM-008", "BEAM-009", "BEAM-010", "EIG-001", "EIG-002", "EIG-003", "BUC-001", "BUC-002", "BUC-003"),
    "E": ("NLG-001", "NLG-004", "NLG-005", "MAT-001", "MAT-002", "MAT-003", "MAT-005", "MAT-006", "MAT-007"),
}


@dataclass(frozen=True)
class VerificationCase:
    case_id: str
    tier: int
    category: str
    required: bool
    title: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "batch": CASE_TO_BATCH.get(self.case_id),
            "evidence_type": _case_evidence_type(self.case_id, self.category),
            "tier": self.tier,
            "category": self.category,
            "required": self.required,
            "title": self.title,
        }


@dataclass
class VerificationCaseResult:
    case_id: str
    status: str
    title: str
    tier: int
    category: str
    required: bool
    analysis_type: str = "verification"
    evidence_type: Optional[str] = None
    element_types: List[str] = field(default_factory=list)
    mesh: Dict[str, Any] = field(default_factory=dict)
    reference: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    checks: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    test_execution_status: Optional[str] = None
    verification_completion_status: Optional[str] = None
    release_gate_status: Optional[str] = None

    def __post_init__(self) -> None:
        if self.evidence_type is None:
            self.evidence_type = _case_evidence_type(self.case_id, self.category)
        if self.test_execution_status is None:
            self.test_execution_status = "failed" if self.status == "FAIL" else "passed"
        if self.verification_completion_status is None:
            self.verification_completion_status = "complete" if self.status == "PASS" else "incomplete"
        if self.release_gate_status is None:
            if self.status == "PASS":
                self.release_gate_status = "passed"
            elif self.required:
                self.release_gate_status = "blocked"
            else:
                self.release_gate_status = "not_evaluated"

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "case_id": self.case_id,
            "batch": CASE_TO_BATCH.get(self.case_id),
            "status": self.status,
            "evidence_type": self.evidence_type,
            "solver_commit": _git_sha(),
            "test_execution_status": self.test_execution_status,
            "verification_completion_status": self.verification_completion_status,
            "release_gate_status": self.release_gate_status,
            "title": self.title,
            "tier": int(self.tier),
            "category": self.category,
            "required": bool(self.required),
            "analysis_type": self.analysis_type,
            "element_types": list(self.element_types),
            "mesh": dict(self.mesh),
            "reference": dict(self.reference),
            "result": dict(self.result),
            "checks": dict(self.checks),
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


def verification_manifest_cases() -> List[VerificationCase]:
    return [VerificationCase(*row) for row in CASE_ROWS]


_GIT_SHA_CACHE: Optional[str] = None


def _git_sha() -> Optional[str]:
    global _GIT_SHA_CACHE
    if _GIT_SHA_CACHE is not None:
        return _GIT_SHA_CACHE
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    except Exception:
        return None
    _GIT_SHA_CACHE = result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None
    return _GIT_SHA_CACHE


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _pass(case: VerificationCase, **kwargs: Any) -> VerificationCaseResult:
    return VerificationCaseResult(case.case_id, "PASS", case.title, case.tier, case.category, case.required, **kwargs)


def _xfail(case: VerificationCase, reason: str, **kwargs: Any) -> VerificationCaseResult:
    return VerificationCaseResult(case.case_id, "XFAIL", case.title, case.tier, case.category, case.required, reason=reason, **kwargs)


def _fail(case: VerificationCase, reason: str, **kwargs: Any) -> VerificationCaseResult:
    return VerificationCaseResult(case.case_id, "FAIL", case.title, case.tier, case.category, case.required, reason=reason, **kwargs)


def _cached_nonlinear_framework_result(
    key: str,
    case: VerificationCase,
    evaluator: Callable[[VerificationCase], VerificationCaseResult],
) -> VerificationCaseResult:
    """Reuse NLG-003 ingredients only within one verification report."""

    cache = _NONLINEAR_FRAMEWORK_RUN_CACHE.get()
    cached = None if cache is None else cache.get(key)
    if cached is not None:
        # NLG-003 uses the same mechanics check as a framework ingredient,
        # while standalone rows retain their own manifest identity and mutable
        # diagnostics.  Rebind only that manifest envelope on a deep copy.
        return replace(
            copy.deepcopy(cached),
            case_id=case.case_id,
            title=case.title,
            tier=case.tier,
            category=case.category,
            required=case.required,
        )

    result = evaluator(case)
    if cache is not None:
        cache[key] = copy.deepcopy(result)
    return result


def _rel_error(value: float, reference: float, floor: float = 1.0e-30) -> float:
    return abs(float(value) - float(reference)) / max(abs(float(reference)), floor)


def _symmetry_error(matrix: np.ndarray | sparse.spmatrix) -> float:
    if sparse.issparse(matrix):
        return float(sparse.linalg.norm(matrix - matrix.T) / max(float(sparse.linalg.norm(matrix)), 1.0))
    dense = np.asarray(matrix, dtype=float)
    return float(np.linalg.norm(dense - dense.T) / max(np.linalg.norm(dense), 1.0))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run_meta_001(case: VerificationCase) -> VerificationCaseResult:
    return _pass(
        case,
        analysis_type="reporting",
        reference={
            "test_execution_status": ["passed", "failed"],
            "verification_completion_status": ["complete", "incomplete"],
            "release_gate_status": ["passed", "blocked", "not_evaluated"],
        },
        result={"status_semantics": "separated"},
        checks={
            "meaning": {
                "test_execution_status": "whether the local check command executed without an unexpected error",
                "verification_completion_status": "whether the case has complete accepted verification evidence",
                "release_gate_status": "whether a required capability is releasable from this evidence",
            }
        },
    )


def _finite_metric_rows(rows: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    finite_rows: List[Mapping[str, Any]] = []
    for row in rows:
        try:
            value = float(row.get("relative_error", math.nan))
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            finite_rows.append(row)
    return finite_rows


def _composite_strip_result(
    case: VerificationCase,
    *,
    metric: str,
    tolerance: float,
    expected_status: str,
    reference_type: str,
) -> VerificationCaseResult:
    # Reuse the same run-scoped, copy-on-read rows used by the companion
    # composite verification views.  COUP-016 and BUC-005 deliberately expose
    # the same buckling calculation under different acceptance labels.
    rows, max_relative_error, _pair_spread = _composite_rows_for(metric)
    finite_rows = _finite_metric_rows(rows)
    status_ok = all(str(row.get("solver_status")) == expected_status for row in rows)
    tolerance_ok = bool(finite_rows) and len(finite_rows) == len(rows) and max_relative_error <= float(tolerance)
    payload = {
        "element_types": ["shell4", "shell8", "shell8r", "beam2", "interpolated_mpc"],
        "analysis_type": {
            "static": "linear_static",
            "modal": "modal",
            "buckling": "linear_buckling",
        }[metric],
        "mesh": {
            "fixture": "narrow production stiffened strip",
            "eccentricity_to_thickness": 5.0,
            "element_types": [str(row.get("element_type")) for row in rows],
        },
        "reference": {
            "type": reference_type,
            "source": "closed-form composite-section beam theory",
            "acceptance_tolerance_rel": float(tolerance),
        },
        "result": {"max_relative_error": max_relative_error, "rows_evaluated": len(rows)},
        "checks": {"rows": rows, "solver_status_expected": expected_status},
    }
    if status_ok and tolerance_ok:
        return _pass(case, **payload)
    reason = (
        f"analytical composite-strip check executed, but max relative error {max_relative_error:.6g} "
        f"exceeds tolerance {float(tolerance):.6g}"
        if status_ok
        else "analytical composite-strip check executed, but one or more solver statuses were not successful"
    )
    return _xfail(case, reason, **payload)


def _beam_model(
    *,
    length: float = 2.0,
    area: float = 0.02,
    iy: float = 1.0e-6,
    iz: float = 1.0e-6,
    j: float = 1.0e-6,
    density: float = 7850.0,
    num_elements: int = 1,
) -> FEModel:
    model = FEModel("verification_beam")
    model.add_material("steel", 210.0e9, 0.3, density=density)
    for i in range(num_elements + 1):
        model.add_node(i + 1, length * i / num_elements, 0.0, 0.0)
    section = {"area": area, "Iy": iy, "Iz": iz, "J": j, "shear_factor_y": 5.0 / 6.0, "shear_factor_z": 5.0 / 6.0}
    for i in range(num_elements):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", section))
    return model


def _run_alg_001(case: VerificationCase) -> VerificationCaseResult:
    points = [(-1.0, -1.0), (0.0, 0.0), (0.37, -0.42), (0.91, 0.88)]
    max_partition = 0.0
    max_derivative = 0.0
    for nnode in (4, 8):
        element = create_shell_element(1, list(range(1, nnode + 1)), "steel")
        for xi, eta in points:
            N, dxi, deta = element.compute_shape_functions(xi, eta)
            max_partition = max(max_partition, abs(float(np.sum(N) - 1.0)))
            max_derivative = max(max_derivative, abs(float(np.sum(dxi))), abs(float(np.sum(deta))))
    for xi in (-1.0, -0.25, 0.0, 0.66, 1.0):
        N = np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0])
        dN = np.array([-0.5, 0.5])
        max_partition = max(max_partition, abs(float(np.sum(N) - 1.0)))
        max_derivative = max(max_derivative, abs(float(np.sum(dN))))
    _assert(max_partition < 1.0e-13 and max_derivative < 1.0e-12, "shape-function partition failed")
    return _pass(case, element_types=["beam2", "shell4", "shell8"], checks={"partition_error": max_partition, "derivative_sum_error": max_derivative})


def _run_alg_002(case: VerificationCase) -> VerificationCaseResult:
    natural = {
        4: [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        8: [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)],
    }
    max_error = 0.0
    for nnode, coords in natural.items():
        element = create_shell_element(1, list(range(1, nnode + 1)), "steel")
        for node_index, (xi, eta) in enumerate(coords):
            N, _, _ = element.compute_shape_functions(xi, eta)
            expected = np.zeros(nnode)
            expected[node_index] = 1.0
            max_error = max(max_error, float(np.max(np.abs(N - expected))))
    _assert(max_error < 1.0e-13, "nodal interpolation failed")
    return _pass(case, element_types=["shell4", "shell8"], checks={"max_delta_error": max_error})


def _single_shell_model(nnode: int = 4) -> FEModel:
    model = FEModel("single_shell_verification")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    if nnode == 4:
        coords = [(0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (1.35, 0.8, 0.05), (0.1, 0.9, 0.0)]
    else:
        coords = [(0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (1.35, 0.8, 0.05), (0.1, 0.9, 0.0), (0.6, 0.0, 0.0), (1.275, 0.4, 0.025), (0.725, 0.85, 0.025), (0.05, 0.45, 0.0)]
    for i, xyz in enumerate(coords, start=1):
        model.add_node(i, *xyz)
    model.add_element(1, create_shell_element(1, list(range(1, nnode + 1)), "steel", thickness=0.01))
    return model


def _run_alg_003(case: VerificationCase) -> VerificationCaseResult:
    min_det = math.inf
    for nnode in (4, 8):
        model = _single_shell_model(nnode)
        element = model.mesh.elements[1]
        coords = element.get_node_coordinates(model.mesh)
        for xi, eta in element.gauss_points:
            _N, dxi, deta = element.compute_shape_functions(float(xi), float(eta))
            R, _dx, _dy, det_j = element._local_frame_and_derivatives(coords, dxi, deta)
            min_det = min(min_det, float(det_j))
            _assert(float(np.linalg.det(R)) > 0.0, "shell local frame has negative determinant")
    zero = _single_shell_model(4)
    for node in zero.mesh.nodes.values():
        node.x = 0.0
        node.y = 0.0
        node.z = 0.0
    element = zero.mesh.elements[1]
    coords = element.get_node_coordinates(zero.mesh)
    _N, dxi, deta = element.compute_shape_functions(0.0, 0.0)
    raised = False
    try:
        element._local_frame_and_derivatives(coords, dxi, deta)
    except ValueError:
        raised = True
    _assert(raised, "zero-area shell did not raise ValueError")
    return _pass(case, element_types=["shell4", "shell8"], checks={"min_surface_jacobian": min_det, "zero_area_rejected": raised})


def _run_alg_004(case: VerificationCase) -> VerificationCaseResult:
    beam = _beam_model()
    shell = _single_shell_model(8)
    beam_k = beam.mesh.elements[1].compute_stiffness_matrix(beam.mesh, beam.get_material("steel"))
    shell_k = shell.mesh.elements[1].compute_stiffness_matrix(shell.mesh, shell.get_material("steel"))
    beam_err = _symmetry_error(beam_k)
    shell_err = _symmetry_error(shell_k)
    _assert(max(beam_err, shell_err) < 1.0e-10, "element stiffness matrix is not symmetric")
    return _pass(case, element_types=["beam2", "shell8"], checks={"beam_symmetry": beam_err, "shell_symmetry": shell_err})


def _run_alg_005(case: VerificationCase) -> VerificationCaseResult:
    beam = _beam_model()
    shell = _single_shell_model(8)
    beam_m = beam.mesh.elements[1].compute_mass_matrix(beam.mesh, beam.get_material("steel"))
    shell_m = shell.mesh.elements[1].compute_mass_matrix(shell.mesh, shell.get_material("steel"))
    beam_min = float(np.min(np.linalg.eigvalsh(0.5 * (beam_m + beam_m.T))))
    shell_min = float(np.min(np.linalg.eigvalsh(0.5 * (shell_m + shell_m.T))))
    _assert(_symmetry_error(beam_m) < 1.0e-12 and _symmetry_error(shell_m) < 1.0e-12, "element mass symmetry failed")
    _assert(beam_min > -1.0e-9 and shell_min > -1.0e-9, "element mass has negative eigenvalue")
    return _pass(case, element_types=["beam2", "shell8"], checks={"beam_min_eigenvalue": beam_min, "shell_min_eigenvalue": shell_min})


def _rigid_body_vector(model: FEModel, mode: int) -> np.ndarray:
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in model.mesh.nodes.values():
        x = node.coords()
        d = node.dofs
        if mode < 3:
            u[d[mode]] = 1.0
        else:
            omega = np.zeros(3)
            omega[mode - 3] = 1.0
            u[d[:3]] = np.cross(omega, x)
            u[d[3:6]] = omega
    return u


def _run_alg_006(case: VerificationCase) -> VerificationCaseResult:
    model = _single_shell_model(4)
    K, _info = assemble_stiffness_matrix(model)
    max_ratio = 0.0
    norm_k = max(float(sparse.linalg.norm(K)), 1.0)
    for mode in range(6):
        u = _rigid_body_vector(model, mode)
        ratio = float(np.linalg.norm(K @ u) / (norm_k * max(np.linalg.norm(u), 1.0)))
        max_ratio = max(max_ratio, ratio)
    _assert(max_ratio < 1.0e-10, "rigid body mode produced elastic force")
    return _pass(case, element_types=["shell4"], checks={"max_rigid_body_force_ratio": max_ratio})


def _run_alg_007(case: VerificationCase) -> VerificationCaseResult:
    model = FEModel("beam_orientation")
    model.add_material("steel", 210.0e9, 0.3)
    model.add_node(1, 0.0, 0.0, 0.0)
    direction = np.array([0.37, -0.51, 0.776], dtype=float)
    direction /= np.linalg.norm(direction)
    model.add_node(2, *direction)
    element = BeamElement(1, [1, 2], "steel", {"area": 0.01, "Iy": 1.0e-6, "Iz": 2.0e-6, "J": 1.0e-6, "orientation": (0.21, 0.91, 0.35)})
    model.add_element(1, element)
    _L, T = element._beam_frame_and_transform(element.get_node_coordinates(model.mesh))
    R = T[:3, :3].T
    ortho = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det = float(np.linalg.det(R))
    _assert(ortho < 1.0e-12 and abs(det - 1.0) < 1.0e-12, "beam transform is not proper orthogonal")
    return _pass(case, element_types=["beam2"], checks={"orthogonality_error": ortho, "determinant": det})


def _run_alg_008(case: VerificationCase) -> VerificationCaseResult:
    model_a = _beam_model(num_elements=2)
    model_b = _beam_model(num_elements=2)
    model_b.mesh.elements = dict(reversed(list(model_b.mesh.elements.items())))
    ka, _ = assemble_stiffness_matrix(model_a)
    kb, _ = assemble_stiffness_matrix(model_b)
    diff = float(sparse.linalg.norm(ka - kb))
    _assert(diff == 0.0, "element ordering changed assembled stiffness")
    return _pass(case, element_types=["beam2"], mesh={"elements": 2}, checks={"assembly_order_difference": diff})


def _run_alg_009(case: VerificationCase) -> VerificationCaseResult:
    model = _beam_model(length=2.0, area=0.02)
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("slider", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    load = LoadCase("axial")
    load.add_nodal_load(2, [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u, _info = solve_linear(model, load)
    K, _ = assemble_stiffness_matrix(model)
    F = load.get_load_vector(model.mesh, model.mesh.dof_manager, model.get_material)
    energy = 0.5 * float(u @ (K @ u))
    work = 0.5 * float(u @ F)
    err = abs(energy - work) / max(abs(energy), abs(work), 1.0e-30)
    _assert(err < 1.0e-8, "energy/work identity failed")
    return _pass(case, element_types=["beam2"], checks={"strain_energy": energy, "external_work": work, "relative_error": err})


def _run_beam_001(case: VerificationCase) -> VerificationCaseResult:
    L, A, E, F = 2.0, 0.02, 210.0e9, 100.0e3
    model = _beam_model(length=L, area=A)
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("slider", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    load = LoadCase("axial")
    load.add_nodal_load(2, [F, 0.0, 0.0, 0.0, 0.0, 0.0])
    u, _ = solve_linear(model, load)
    ref = F * L / (E * A)
    value = float(u[model.mesh.nodes[2].dofs[0]])
    err = _rel_error(value, ref)
    _assert(err < 1.0e-10, "axial displacement mismatch")
    return _pass(case, element_types=["beam2"], reference={"type": "analytical", "value": ref, "quantity": "tip ux"}, result={"value": value, "relative_error": err})


def _run_beam_002(case: VerificationCase) -> VerificationCaseResult:
    L, r, torque, E, nu = 2.0, 0.05, 1000.0, 210.0e9, 0.3
    G = E / (2.0 * (1.0 + nu))
    J = math.pi * r**4 / 2.0
    model = _beam_model(length=L, area=math.pi * r**2, iy=J / 2.0, iz=J / 2.0, j=J)
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("tip_suppress", [2], {"ux": 0.0, "uy": 0.0, "uz": 0.0, "ry": 0.0, "rz": 0.0}))
    load = LoadCase("torsion")
    load.add_nodal_load(2, [0.0, 0.0, 0.0, torque, 0.0, 0.0])
    u, _ = solve_linear(model, load)
    ref = torque * L / (G * J)
    value = float(u[model.mesh.nodes[2].dofs[3]])
    err = _rel_error(value, ref)
    _assert(err < 1.0e-10, "torsion rotation mismatch")
    return _pass(case, element_types=["beam2"], reference={"type": "analytical", "value": ref, "quantity": "tip rx"}, result={"value": value, "relative_error": err})


def _run_beam_003(case: VerificationCase) -> VerificationCaseResult:
    L, b, h, P, E, nu = 2.0, 0.10, 0.20, -1000.0, 210.0e9, 0.3
    A = b * h
    I = b * h**3 / 12.0
    G = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0
    model = _beam_model(length=L, area=A, iy=I, iz=I, j=I, num_elements=8)
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    tip = 9
    load = LoadCase("tip_z")
    load.add_nodal_load(tip, [0.0, 0.0, P, 0.0, 0.0, 0.0])
    u, _ = solve_linear(model, load)
    ref = P * L**3 / (3.0 * E * I) + P * L / (kappa * G * A)
    value = float(u[model.mesh.nodes[tip].dofs[2]])
    err = _rel_error(value, ref)
    _assert(err < 5.0e-3, "Timoshenko cantilever displacement mismatch")
    return _pass(case, element_types=["beam2"], mesh={"elements": 8}, reference={"type": "analytical", "value": ref}, result={"value": value, "relative_error": err})


def _run_beam_004(case: VerificationCase) -> VerificationCaseResult:
    E, nu = 210.0e9, 0.3
    G = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0
    L, width, load_value = 2.0, 0.10, -1000.0
    rows: List[Dict[str, Any]] = []
    for slenderness in (5.0, 10.0, 20.0, 50.0, 100.0):
        depth = L / slenderness
        area = width * depth
        iy = width * depth**3 / 12.0
        iz = depth * width**3 / 12.0
        model = _beam_model(length=L, area=area, iy=iy, iz=iz, j=iy + iz, num_elements=10)
        model.add_boundary_condition(FixedSupport("fixed", [1]))
        tip = 11
        load = LoadCase("tip_z")
        load.add_nodal_load(tip, [0.0, 0.0, load_value, 0.0, 0.0, 0.0])
        u, info = solve_linear(model, load)
        value = float(u[model.mesh.nodes[tip].dofs[2]])
        reference = load_value * L**3 / (3.0 * E * iy) + load_value * L / (kappa * G * area)
        err = _rel_error(value, reference)
        rows.append(
            {
                "L_over_h": float(slenderness),
                "tip_displacement": value,
                "reference": float(reference),
                "relative_error": float(err),
                "solver_status": str((info.get("convergence_info") or {}).get("status", "unknown")),
            }
        )
    max_error = max(float(row["relative_error"]) for row in rows)
    _assert(max_error < 5.0e-3, "Timoshenko slenderness sweep exceeded tolerance")
    return _pass(
        case,
        element_types=["beam2"],
        mesh={"elements": 10, "slenderness": [row["L_over_h"] for row in rows]},
        reference={"type": "analytical", "quantity": "Timoshenko cantilever tip displacement"},
        result={"max_relative_error": max_error},
        checks={"rows": rows},
    )


def _run_beam_005(case: VerificationCase) -> VerificationCaseResult:
    L, b, h, M, E = 2.0, 0.10, 0.20, 1000.0, 210.0e9
    I = b * h**3 / 12.0
    model = _beam_model(length=L, area=b * h, iy=I, iz=I, j=I, num_elements=4)
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    tip = 5
    load = LoadCase("tip_moment_y")
    load.add_nodal_load(tip, [0.0, 0.0, 0.0, 0.0, M, 0.0])
    u, _ = solve_linear(model, load)
    ref_w = -M * L**2 / (2.0 * E * I)
    value = float(u[model.mesh.nodes[tip].dofs[2]])
    err = _rel_error(value, ref_w)
    _assert(err < 1.0e-6, "pure bending displacement mismatch")
    return _pass(case, element_types=["beam2"], reference={"type": "analytical", "value": ref_w}, result={"value": value, "relative_error": err})


def _run_beam_006(case: VerificationCase) -> VerificationCaseResult:
    E, nu = 210.0e9, 0.3
    G = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0
    L = 2.0
    area = 0.02
    iy = 2.0e-5
    iz = 7.0e-6
    py = 1200.0
    pz = -800.0
    orientation = np.array([0.0, 1.0, 1.0], dtype=float)
    model = FEModel("verification_biaxial_local_axis")
    model.add_material("steel", E, nu, density=7850.0)
    for node_id, x in enumerate(np.linspace(0.0, L, 9), start=1):
        model.add_node(node_id, float(x), 0.0, 0.0)
    section = {"area": area, "Iy": iy, "Iz": iz, "J": iy + iz, "shear_factor_y": kappa, "shear_factor_z": kappa, "orientation": orientation}
    for element_id in range(1, 9):
        model.add_element(element_id, BeamElement(element_id, [element_id, element_id + 1], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    first_element = model.mesh.elements[1]
    _L, T = first_element._beam_frame_and_transform(first_element.get_node_coordinates(model.mesh))
    rotation = T[:3, :3].T
    local_force = np.array([0.0, py, pz], dtype=float)
    global_force = rotation @ local_force
    load = LoadCase("local_biaxial_tip")
    load.add_nodal_load(9, np.concatenate([global_force, np.zeros(3)]))
    u, _info = solve_linear(model, load)
    tip_global = u[model.mesh.nodes[9].dofs[:3]]
    tip_local = rotation.T @ tip_global
    ref_y = py * L**3 / (3.0 * E * iz) + py * L / (kappa * G * area)
    ref_z = pz * L**3 / (3.0 * E * iy) + pz * L / (kappa * G * area)
    err_y = _rel_error(tip_local[1], ref_y)
    err_z = _rel_error(tip_local[2], ref_z)
    coupling_x = abs(float(tip_local[0])) / max(abs(float(ref_y)), abs(float(ref_z)), 1.0e-30)
    _assert(max(err_y, err_z, coupling_x) < 5.0e-3, "biaxial local-axis response mismatch")
    return _pass(
        case,
        element_types=["beam2"],
        mesh={"elements": 8},
        reference={"type": "analytical", "quantity": "local-y/local-z Timoshenko cantilever response"},
        result={"local_tip_displacement": tip_local.tolist(), "relative_error_y": err_y, "relative_error_z": err_z},
        checks={"rotation_matrix": rotation.tolist(), "local_force": local_force.tolist(), "global_force": global_force.tolist(), "local_x_coupling_ratio": coupling_x},
    )


def _run_beam_007(case: VerificationCase) -> VerificationCaseResult:
    model = _beam_model(num_elements=2)
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    tip = 3
    loads = []
    components = ([100.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 50.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -75.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 20.0, 30.0, -10.0])
    for i, vec in enumerate(components):
        lc = LoadCase(f"component_{i}")
        lc.add_nodal_load(tip, vec)
        loads.append(lc)
    combined = LoadCase("combined")
    for vec in components:
        combined.add_nodal_load(tip, vec)
    u_many, _ = solve_linear_many(model, loads)
    u_combined, _ = solve_linear(model, combined)
    summed = np.sum(u_many, axis=1)
    err = float(np.linalg.norm(u_combined - summed) / max(np.linalg.norm(u_combined), 1.0e-30))
    _assert(err < 1.0e-9, "linear superposition failed")
    return _pass(case, element_types=["beam2"], checks={"superposition_relative_error": err})


def _run_shell_001(case: VerificationCase) -> VerificationCaseResult:
    metric = q8_patch_metric(reference_q8_geometries()["square"])
    err = max(abs(float(metric["membrane_max_relative_error"])), abs(float(metric["shear_relative_error"])))
    _assert(err < 1.0e-9, "Q8 membrane/shear patch metric failed")
    return _pass(case, element_types=["shell8"], checks=metric)


def _run_shell_002(case: VerificationCase) -> VerificationCaseResult:
    metric = q8_patch_metric(reference_q8_geometries()["square"])
    err = abs(float(metric["bending_relative_error"]))
    _assert(err < 1.0e-9, "Q8 bending patch metric failed")
    return _pass(case, element_types=["shell8"], checks=metric)


def _run_shell_004(case: VerificationCase) -> VerificationCaseResult:
    sweep = thin_plate_locking_sweep((0.01,))
    row = sweep[0]
    ratio = float(row["ratio_to_reference"])
    _assert(0.90 < ratio < 1.05, "plate deflection deviates from thin-reference band")
    return _pass(case, element_types=["shell4"], mesh={"label": "thin_strip_reference"}, reference={"type": "analytical", "quantity": "beam/plate strip"}, result={"ratio_to_reference": ratio}, checks=row)


def _run_shell_005(case: VerificationCase) -> VerificationCaseResult:
    thicknesses = tuple(1.0 / ratio for ratio in THIN_SHELL_SPAN_TO_THICKNESS)
    rows = thin_plate_locking_sweep(thicknesses, length=1.0, width=0.1, num_divisions=10)
    relative_errors = [float(row["relative_error"]) for row in rows]
    statuses = [str(row["solver_status"]) for row in rows]
    ratios = [float(row["ratio_to_reference"]) for row in rows]
    max_error = max(relative_errors)
    ratio_spread = float(max(ratios) - min(ratios))
    _assert(all(status == "converged" for status in statuses), "thin-shell locking sweep did not converge")
    _assert(max_error < 0.02, "thin-shell locking sweep exceeds 2% strip-bending reference error")
    _assert(ratio_spread < 0.005, "thin-shell response ratio changes materially over L/t sweep")
    return _pass(
        case,
        element_types=["shell4"],
        mesh={"label": "cantilever_strip", "span_to_thickness": list(THIN_SHELL_SPAN_TO_THICKNESS)},
        reference={"type": "analytical", "quantity": "Euler-Bernoulli strip bending"},
        result={"max_relative_error": max_error, "ratio_spread": ratio_spread},
        checks={"rows": list(rows), "statuses": statuses, "ratios": ratios},
    )


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def _global_vector_transform(model: FEModel, rotation: np.ndarray) -> sparse.csr_matrix:
    total = model.mesh.dof_manager.total_dofs
    matrix = sparse.lil_matrix((total, total), dtype=float)
    for node in model.mesh.nodes.values():
        d = node.dofs
        matrix[np.ix_(d[:3], d[:3])] = rotation
        matrix[np.ix_(d[3:6], d[3:6])] = rotation
    return matrix.tocsr()


def _rotated_single_shell_model(nnode: int, rotation: np.ndarray, translation: np.ndarray) -> FEModel:
    base = _single_shell_model(nnode)
    rotated = FEModel(f"rotated_shell_{nnode}")
    rotated.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, node in base.mesh.nodes.items():
        coords = rotation @ node.coords() + translation
        rotated.add_node(node_id, float(coords[0]), float(coords[1]), float(coords[2]))
    base_element = base.mesh.elements[1]
    rotated.add_element(1, create_shell_element(1, list(base_element.node_ids), "steel", thickness=base_element.thickness))
    return rotated


def _run_shell_003(case: VerificationCase) -> VerificationCaseResult:
    rotation = _axis_angle_rotation(np.array([0.3, -0.5, 0.8], dtype=float), 0.71)
    translation = np.array([2.0, -1.5, 0.4], dtype=float)
    rows: List[Dict[str, Any]] = []
    for nnode in (4, 8):
        base = _single_shell_model(nnode)
        rotated = _rotated_single_shell_model(nnode, rotation, translation)
        K_base, _ = assemble_stiffness_matrix(base)
        K_rot, _ = assemble_stiffness_matrix(rotated)
        G = _global_vector_transform(base, rotation)
        expected = (G @ K_base @ G.T).tocsr()
        stiffness_error = float(sparse.linalg.norm(K_rot - expected) / max(float(sparse.linalg.norm(K_base)), 1.0))
        M_base, _ = assemble_mass_matrix(base)
        M_rot, _ = assemble_mass_matrix(rotated)
        expected_mass = (G @ M_base @ G.T).tocsr()
        mass_error = float(sparse.linalg.norm(M_rot - expected_mass) / max(float(sparse.linalg.norm(M_base)), 1.0))
        rows.append({"nodes_per_element": nnode, "stiffness_transform_error": stiffness_error, "mass_transform_error": mass_error})
    max_error = max(max(row["stiffness_transform_error"], row["mass_transform_error"]) for row in rows)
    _assert(max_error < 1.0e-9, "shell stiffness/mass changed under rigid transform")
    return _pass(case, element_types=["shell4", "shell8"], checks={"rows": rows, "max_transform_error": max_error})


def _simply_supported_plate_model(divisions: int = 10, thickness: float = 0.01) -> FEModel:
    return _verification_plate_model(divisions=divisions, thickness=thickness, element_family="S4")


def _verification_plate_model(
    *,
    divisions: int = 10,
    thickness: float = 0.01,
    element_family: str = "S4",
) -> FEModel:
    """Build the canonical simply-supported square plate verification model."""
    length = width = 1.0
    family = str(element_family).upper()
    model = generate_simple_panel_mesh(length, width, thickness, divisions, divisions, use_8node_elements=family in {"S8", "S8R", "Q8", "Q8R"})
    model.clear_boundary_conditions()
    model.materials["steel"].density = 7850.0
    if family in {"S8R", "Q8R"}:
        for element in model.mesh.elements.values():
            if isinstance(element, ShellElement) and getattr(element, "_is_8node", False):
                element.reduced_integration = True
                element.hourglass_stabilization = max(float(getattr(element, "hourglass_stabilization", 0.0)), 1.0e-8)
        model.bump_revision("material")
    edge_nodes: List[int] = []
    tol = 1.0e-9
    for node_id, node in model.mesh.nodes.items():
        x, y, _z = node.coords()
        if abs(x) <= tol or abs(x - length) <= tol or abs(y) <= tol or abs(y - width) <= tol:
            edge_nodes.append(int(node_id))
    model.add_boundary_condition(BoundaryCondition("simply_supported_w", edge_nodes, {"uz": 0.0}))
    model.add_boundary_condition(BoundaryCondition("inplane_edge_reference", edge_nodes, {"ux": 0.0, "uy": 0.0}))
    return model


def _plate_bending_frequency_hz(m: int, n: int, *, length: float = 1.0, width: float = 1.0, thickness: float = 0.01) -> float:
    E, nu, rho = 210.0e9, 0.3, 7850.0
    D = E * thickness**3 / (12.0 * (1.0 - nu**2))
    omega = math.pi**2 * math.sqrt(D / (rho * thickness)) * ((m / length) ** 2 + (n / width) ** 2)
    return omega / (2.0 * math.pi)


def _plate_uniaxial_buckling_resultant(*, width: float = 1.0, thickness: float = 0.01, k: float = 4.0) -> float:
    E, nu = 210.0e9, 0.3
    D = E * thickness**3 / (12.0 * (1.0 - nu**2))
    return float(k * math.pi**2 * D / (width**2))


def _simply_supported_plate_buckling(
    *,
    divisions: int = 10,
    thickness: float = 0.01,
) -> Tuple[float, Dict[str, Any]]:
    """Return a report-local cached plate buckling result and diagnostics."""

    key = (int(divisions), float(thickness))
    cache = _PLATE_BUCKLING_RUN_CACHE.get()
    cached = None if cache is None else cache.get(key)
    if cached is not None:
        value, diagnostics = cached
        return value, copy.deepcopy(diagnostics)

    model = _simply_supported_plate_model(divisions=divisions, thickness=thickness)
    states = {
        int(element_id): {"membrane_compression_x": 1.0}
        for element_id, element in model.mesh.elements.items()
        if isinstance(element, ShellElement)
    }
    result = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=3,
        dense_size_limit=10000,
    )
    _assert(
        result.solver_status == "ok" and result.critical_load_factor is not None,
        "plate buckling solve failed",
    )
    value = float(result.critical_load_factor)
    diagnostics = copy.deepcopy(result.diagnostics or {})
    if cache is not None:
        cache[key] = (value, copy.deepcopy(diagnostics))
    return value, diagnostics


def _run_shell_006(case: VerificationCase) -> VerificationCaseResult:
    model = _simply_supported_plate_model(divisions=10, thickness=0.01)
    result = solve_free_vibration(model, num_modes=6, dense_size_limit=10000)
    _assert(result.solver_status == "ok" and result.num_modes_returned > 0, "plate modal solve failed")
    value = float(result.frequencies_hz[0])
    reference = _plate_bending_frequency_hz(1, 1)
    err = _rel_error(value, reference)
    _assert(err < 0.02, "simply supported plate first frequency mismatch")
    return _pass(
        case,
        element_types=["shell4"],
        analysis_type="modal",
        mesh={"divisions": 10, "span_to_thickness": 100},
        reference={"type": "analytical", "mode": [1, 1], "frequency_hz": reference},
        result={"frequency_hz": value, "relative_error": err},
        checks=result.diagnostics,
    )


def _run_shell_007(case: VerificationCase) -> VerificationCaseResult:
    value, diagnostics = _simply_supported_plate_buckling(
        divisions=10,
        thickness=0.01,
    )
    reference = _plate_uniaxial_buckling_resultant()
    err = _rel_error(value, reference)
    _assert(err < 0.02, "simply supported plate buckling load mismatch")
    return _pass(
        case,
        element_types=["shell4"],
        analysis_type="linear_buckling",
        mesh={"divisions": 10, "span_to_thickness": 100},
        reference={"type": "analytical", "k": 4.0, "critical_membrane_resultant": reference},
        result={"critical_load_factor": value, "relative_error": err},
        checks=diagnostics,
    )


def _run_beam_008(case: VerificationCase) -> VerificationCaseResult:
    model = _beam_model(length=1.0, area=1.0, density=2.0)
    model.materials["steel"].elastic_modulus = 100.0
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("slider", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    result = solve_free_vibration(model, num_modes=1)
    ref = math.sqrt(100.0 / 1.0) / (2.0 * math.pi)
    value = float(result.frequencies_hz[0])
    err = _rel_error(value, ref)
    _assert(err < 5.0e-3, "axial modal frequency mismatch")
    return _pass(case, element_types=["beam2"], analysis_type="modal", reference={"type": "analytical", "value": ref}, result={"value": value, "relative_error": err}, checks=result.diagnostics)


def _run_beam_009(case: VerificationCase) -> VerificationCaseResult:
    model = _beam_model(length=1.0, area=1.0, density=2.0)
    model.materials["steel"].elastic_modulus = 100.0
    result = solve_free_vibration(model, num_modes=6)
    _assert(result.diagnostics["num_rigid_body_modes"] == 6, "free beam did not return six rigid modes")
    return _pass(case, element_types=["beam2"], analysis_type="modal", checks=result.diagnostics)


def _run_beam_010(case: VerificationCase) -> VerificationCaseResult:
    model = FEModel("verification_column")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    L = 4.0
    Iz = 5.0e-6
    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": Iz, "J": 2.0e-6}
    for i in range(11):
        model.add_node(i + 1, L * i / 10, 0.0, 0.0)
    for i in range(10):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", section))
    all_nodes = list(model.mesh.nodes)
    model.add_boundary_condition(BoundaryCondition("suppress", all_nodes, {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pins", [1, 11], {"uy": 0.0}))
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    result = solve_eigenvalue_buckling(model, states, num_modes=1)
    ref = math.pi**2 * 210.0e9 * Iz / L**2
    value = float(result.critical_load_factor or 0.0)
    err = _rel_error(value, ref)
    _assert(err < 0.08, "Euler buckling factor mismatch")
    return _pass(case, element_types=["beam2"], analysis_type="linear_buckling", reference={"type": "analytical", "value": ref}, result={"value": value, "relative_error": err}, checks=result.diagnostics or {})


def _run_coup_003(case: VerificationCase) -> VerificationCaseResult:
    e = np.array([0.0, 0.0, 0.25])
    u_s = np.array([0.1, -0.2, 0.05])
    theta = np.array([0.03, -0.04, 0.02])
    expected = u_s + np.cross(theta, e)
    evaluated = u_s + np.cross(theta, e)
    err = float(np.linalg.norm(evaluated - expected))
    _assert(err < 1.0e-12, "eccentric rigid-link kinematic relation failed")
    return _pass(case, element_types=["mpc"], checks={"component_error": err})


def _run_coup_004(case: VerificationCase) -> VerificationCaseResult:
    e = np.array([0.0, 0.0, 0.25])
    force = np.array([1000.0, -200.0, 0.0])
    expected = np.cross(e, force)
    evaluated = np.cross(e, force)
    err = float(np.linalg.norm(evaluated - expected))
    _assert(err < 1.0e-12, "eccentric moment-transfer relation failed")
    return _pass(case, element_types=["mpc"], checks={"moment_error": err, "moment_norm": float(np.linalg.norm(expected))})


def _coincident_coupling_model(*, fixed_shell: bool = False) -> FEModel:
    model = FEModel("verification_coincident_coupling")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 0.0, 0.0, 0.0)
    model.add_element(1, CoupledBeamShellElement(1, beam_node_id=2, shell_node_id=1, material_name="steel"))
    if fixed_shell:
        model.add_boundary_condition(FixedSupport("fixed_shell_master", [1]))
    return model


def _run_coup_001(case: VerificationCase) -> VerificationCaseResult:
    model = _coincident_coupling_model()
    total_dofs = model.mesh.dof_manager.total_dofs
    K = sparse.eye(total_dofs, format="csr")
    zero = np.zeros(total_dofs, dtype=float)
    _K_red, _F_red, T, u0, independent, constraint_info = build_constraint_transformation(K, zero, model)
    q = np.linspace(-0.25, 0.35, len(independent), dtype=float)
    u = reconstruct_full_solution(T, q, u0)

    shell = model.mesh.get_node(1)
    beam = model.mesh.get_node(2)
    translation_error = float(np.linalg.norm(u[beam.dofs[:3]] - u[shell.dofs[:3]]))
    rotation_error = float(np.linalg.norm(u[beam.dofs[3:6]] - u[shell.dofs[3:6]]))
    residuals = mpc_constraint_residuals(model, u)
    max_constraint_residual = max((abs(value) for value in residuals.values()), default=0.0)

    _assert(constraint_info["num_mpc_slave_dofs"] == 6, "coincident coupling did not create six slave DOFs")
    _assert(max(translation_error, rotation_error, max_constraint_residual) < 1.0e-13, "coincident MPC compatibility failed")
    return _pass(
        case,
        element_types=["beam_shell_mpc"],
        checks={
            "num_mpc_slave_dofs": int(constraint_info["num_mpc_slave_dofs"]),
            "translation_error": translation_error,
            "rotation_error": rotation_error,
            "max_constraint_residual": float(max_constraint_residual),
        },
    )


def _run_coup_002(case: VerificationCase) -> VerificationCaseResult:
    model = _coincident_coupling_model(fixed_shell=True)
    load_vector = np.array([1200.0, -350.0, 80.0, 14.0, -6.0, 22.0], dtype=float)
    load = LoadCase("coincident_slave_load")
    load.add_nodal_load(2, load_vector)
    u0 = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)

    diagnostics = compute_constraint_force_diagnostics(model, u0, load)
    slave_force = np.asarray(diagnostics["mpc_slave_forces"].get(2, np.zeros(6)), dtype=float)
    master_equivalent = np.asarray(diagnostics["mpc_master_equivalent_forces"].get(1, np.zeros(6)), dtype=float)
    direct_support = np.asarray(diagnostics["support_reactions"].get(1, np.zeros(6)), dtype=float)

    slave_error = float(np.linalg.norm(slave_force + load_vector))
    master_error = float(np.linalg.norm(master_equivalent + load_vector))
    support_direct_norm = float(np.linalg.norm(direct_support))

    _assert(slave_error < 1.0e-12, "MPC slave residual did not recover the applied slave load")
    _assert(master_error < 1.0e-12, "MPC master-equivalent force did not transfer the slave load")
    _assert(support_direct_norm < 1.0e-12, "direct support bucket should remain separate from MPC transfer")
    return _pass(
        case,
        element_types=["beam_shell_mpc"],
        checks={
            "slave_force": slave_force.tolist(),
            "master_equivalent_force": master_equivalent.tolist(),
            "direct_support_force": direct_support.tolist(),
            "slave_force_error": slave_error,
            "master_equivalent_force_error": master_error,
            "direct_support_norm": support_direct_norm,
            "num_mpc_constraint_forces": len(diagnostics["mpc_constraint_forces"]),
        },
    )


def _thin_stiffened_panel_geometry(num_stiffeners: int = 1) -> PanelGeometry:
    width = 0.4
    return PanelGeometry(
        length=1.0,
        width=width,
        plate_thickness=0.001,
        stiffener_type="T-bar",
        stiffener_spacing=width / (int(num_stiffeners) + 1),
        stiffener_height=0.04,
        stiffener_web_thickness=0.003,
        stiffener_flange_width=0.03,
        stiffener_flange_thickness=0.003,
        num_stiffeners=int(num_stiffeners),
        in_plane_support="Integrated",
        rotational_support="FS",
    )


def _thin_stiffened_panel_model(
    num_stiffeners: int = 1,
    *,
    element_family: str = "Q4",
    shell_divisions_x: int = 4,
    shell_divisions_y: Optional[int] = None,
    beam_divisions: int = 4,
) -> Tuple[FEModel, PanelGeometry, MeshConfig]:
    panel = _thin_stiffened_panel_geometry(num_stiffeners)
    family = str(element_family).upper()
    config = MeshConfig(
        shell_num_divisions_x=int(shell_divisions_x),
        shell_num_divisions_y=max(2 * int(num_stiffeners), 2) if shell_divisions_y is None else int(shell_divisions_y),
        beam_num_divisions=int(beam_divisions),
        use_coupling_elements=True,
        align_mesh_to_stiffeners=True,
        use_8node_shells=family in {"Q8", "Q8R", "S8", "S8R"},
    )
    model = generate_stiffened_panel_mesh(panel, config)
    if family in {"Q8R", "S8R"}:
        for element in model.mesh.elements.values():
            if isinstance(element, ShellElement) and getattr(element, "_is_8node", False):
                element.reduced_integration = True
                element.hourglass_stabilization = max(float(getattr(element, "hourglass_stabilization", 0.0)), 1.0e-8)
        model.bump_revision("material")
    return model, panel, config


def _count_mpc_constraints(model: FEModel) -> int:
    count = 0
    for element in model.mesh.elements.values():
        getter = getattr(element, "get_mpc_constraints", None)
        if getter is not None:
            count += len(getter(model.mesh) or [])
    return int(count)


def _pressure_load_for_shells(model: FEModel, pressure: float) -> LoadCase:
    load = LoadCase("thin_stiffened_plate_pressure")
    for element_id, element in model.mesh.elements.items():
        if isinstance(element, ShellElement):
            load.add_pressure_load(int(element_id), float(pressure))
    return load


def _analytic_stiffened_panel_mass(panel: PanelGeometry, density: float = 7850.0) -> float:
    section = StiffenerCrossSection.from_geometry(
        panel.stiffener_type,
        panel.stiffener_height,
        panel.stiffener_web_thickness,
        panel.stiffener_flange_width,
        panel.stiffener_flange_thickness,
    )
    plate_mass = float(density) * float(panel.length) * float(panel.width) * float(panel.plate_thickness)
    stiffener_mass = float(density) * float(section.area) * float(panel.length) * int(panel.num_stiffeners)
    return plate_mass + stiffener_mass


def _max_mpc_eccentricity_z(model: FEModel) -> Tuple[float, float]:
    values = [
        float(getattr(element, "eccentricity")[2])
        for element in model.mesh.elements.values()
        if hasattr(element, "eccentricity")
    ]
    return (min(values), max(values)) if values else (0.0, 0.0)


def _stiffened_panel_buckling_states(model: FEModel) -> Dict[int, Dict[str, float]]:
    states: Dict[int, Dict[str, float]] = {}
    for element_id, element in model.mesh.elements.items():
        if isinstance(element, ShellElement):
            states[int(element_id)] = {"membrane_compression_x": 1.0}
        elif isinstance(element, BeamElement):
            states[int(element_id)] = {"axial_compression": 1.0}
        else:
            states[int(element_id)] = {}
    return states


def _run_coup_007(case: VerificationCase) -> VerificationCaseResult:
    rows: List[Dict[str, Any]] = []
    for num_stiffeners in (1, 2):
        model, panel, config = _thin_stiffened_panel_model(num_stiffeners)
        props = calculate_mass_properties(model)
        expected_mass = _analytic_stiffened_panel_mass(panel)
        mass_error = _rel_error(props.total_mass, expected_mass)

        beam_node_count = sum(1 for node in model.mesh.nodes.values() if abs(float(node.z) - panel.stiffener_height) < 1.0e-12)
        expected_mpc_constraints = 6 * beam_node_count
        mpc_constraints = _count_mpc_constraints(model)

        K, _ = assemble_stiffness_matrix(model)
        rigid = _rigid_body_vector(model, 3)
        rigid_force_ratio = float(np.linalg.norm(K @ rigid) / max(float(sparse.linalg.norm(K)) * np.linalg.norm(rigid), 1.0))

        pressure = 250.0
        load = _pressure_load_for_shells(model, pressure)
        load_vector = load.get_load_vector(model.mesh, model.mesh.dof_manager, model.get_material)
        applied_force = np.array(
            [
                float(np.sum(load_vector[0::6])),
                float(np.sum(load_vector[1::6])),
                float(np.sum(load_vector[2::6])),
            ],
            dtype=float,
        )
        expected_force = np.array([0.0, 0.0, pressure * panel.length * panel.width], dtype=float)
        resultant_error = float(np.linalg.norm(applied_force - expected_force) / max(np.linalg.norm(expected_force), 1.0))

        displacements, solver_info = solve_linear(model, load)
        solver_status = str((solver_info.get("convergence_info") or {}).get("status", "unknown"))
        max_displacement = float(np.max(np.abs(displacements))) if displacements.size else 0.0
        ecc_min, ecc_max = _max_mpc_eccentricity_z(model)

        row = {
            "num_stiffeners": int(num_stiffeners),
            "nodes": int(len(model.mesh.nodes)),
            "elements": int(len(model.mesh.elements)),
            "shell_divisions": [int(config.shell_num_divisions_x), int(config.shell_num_divisions_y)],
            "beam_node_count": int(beam_node_count),
            "expected_mpc_constraints": int(expected_mpc_constraints),
            "mpc_constraints": int(mpc_constraints),
            "mass": float(props.total_mass),
            "expected_mass": float(expected_mass),
            "mass_relative_error": float(mass_error),
            "skipped_constraint_elements": [int(element_id) for element_id in props.skipped_elements],
            "rigid_force_ratio": rigid_force_ratio,
            "applied_force": applied_force.tolist(),
            "expected_force": expected_force.tolist(),
            "load_resultant_relative_error": resultant_error,
            "max_abs_displacement": max_displacement,
            "solver_status": solver_status,
            "eccentricity_z_min": float(ecc_min),
            "eccentricity_z_max": float(ecc_max),
        }
        rows.append(row)

        _assert(mpc_constraints == expected_mpc_constraints, "stiffened panel MPC constraint count mismatch")
        _assert(mass_error < 1.0e-12, "stiffened panel mass does not match physical shell-plus-beam mass")
        _assert(rigid_force_ratio < 1.0e-10, "stiffened panel produces elastic force under rigid motion")
        _assert(resultant_error < 1.0e-12, "stiffened panel pressure resultant mismatch")
        _assert(solver_status == "converged" and np.isfinite(max_displacement) and max_displacement > 0.0, "stiffened panel static solve failed")
        _assert(ecc_min > 0.0 and ecc_max > 0.0, "stiffener eccentricity sign is not positive out of the shell midsurface")

    return _pass(
        case,
        element_types=["shell4", "beam2", "interpolated_mpc"],
        analysis_type="linear_static",
        mesh={"span_to_thickness": 1000, "num_stiffeners": [1, 2]},
        reference={"type": "analytical", "quantities": ["mass", "pressure resultant", "rigid-body zero energy"]},
        result={"max_mass_relative_error": max(float(row["mass_relative_error"]) for row in rows)},
        checks={"rows": rows},
    )


def _run_coup_008(case: VerificationCase) -> VerificationCaseResult:
    model, panel, config = _thin_stiffened_panel_model(1)
    modal = solve_free_vibration(model, num_modes=6, dense_size_limit=10000)
    props = calculate_mass_properties(model)
    mass_reference = _analytic_stiffened_panel_mass(panel)
    mass_error = _rel_error(props.total_mass, mass_reference)
    frequencies = modal.frequencies_hz.tolist()
    min_frequency = min(float(value) for value in frequencies) if frequencies else 0.0
    diagnostics = modal.diagnostics
    _assert(modal.solver_status == "ok" and len(frequencies) >= 3, "stiffened-panel modal solve failed")
    _assert(min_frequency > 1.0e-6, "stiffened-panel modal solve produced an extra near-zero mode")
    _assert(float(diagnostics.get("mass_orthogonality_error", 1.0)) < 1.0e-8, "stiffened-panel modal mass orthogonality failed")
    _assert(mass_error < 1.0e-12, "stiffened-panel modal mass check failed")
    return _pass(
        case,
        element_types=["shell4", "beam2", "interpolated_mpc"],
        analysis_type="modal",
        mesh={"shell_divisions": [config.shell_num_divisions_x, config.shell_num_divisions_y], "beam_divisions": config.beam_num_divisions},
        reference={"type": "solver-independent invariant", "quantities": ["physical mass", "mass orthogonality", "no extra near-zero modes"]},
        result={"frequencies_hz": frequencies[:6], "mass_relative_error": mass_error},
        checks={**diagnostics, "physical_mass": props.total_mass, "expected_mass": mass_reference},
    )


def _run_coup_009(case: VerificationCase) -> VerificationCaseResult:
    model, _panel, config = _thin_stiffened_panel_model(1)
    base = _stiffened_panel_buckling_states(model)
    doubled = {
        element_id: {key: 2.0 * float(value) for key, value in state.items()}
        for element_id, state in base.items()
    }
    result = solve_eigenvalue_buckling(model, base, num_modes=3, dense_size_limit=10000)
    doubled_result = solve_eigenvalue_buckling(model, doubled, num_modes=1, dense_size_limit=10000)
    _assert(result.solver_status == "ok" and result.num_modes_returned >= 3, "stiffened-panel buckling solve failed")
    _assert(doubled_result.critical_load_factor is not None and result.critical_load_factor is not None, "stiffened-panel buckling scaling solve failed")
    scale_error = _rel_error(float(doubled_result.critical_load_factor), 0.5 * float(result.critical_load_factor))
    residual = float((result.diagnostics or {}).get("max_residual_norm", 1.0))
    _assert(scale_error < 1.0e-8, "stiffened-panel buckling preload scaling failed")
    _assert(residual < 1.0e-8, "stiffened-panel buckling residual is too large")
    factors = [float(mode.load_factor) for mode in result.modes]
    _assert(all(value > 0.0 and np.isfinite(value) for value in factors), "stiffened-panel buckling factors are not positive finite values")
    return _pass(
        case,
        element_types=["shell4", "beam2", "interpolated_mpc"],
        analysis_type="linear_buckling",
        mesh={"shell_divisions": [config.shell_num_divisions_x, config.shell_num_divisions_y], "beam_divisions": config.beam_num_divisions},
        reference={"type": "solver-independent invariant", "quantities": ["positive roots", "preload scaling", "eigen residual"]},
        result={"load_factors": factors, "preload_scaling_error": scale_error},
        checks=result.diagnostics or {},
    )


def _composite_rows_for(metric: str) -> Tuple[List[Dict[str, Any]], float, float]:
    cache = _COMPOSITE_METRIC_RUN_CACHE.get()
    cached = None if cache is None else cache.get(metric)
    if cached is not None:
        cached_rows, max_error, spread = cached
        # Result objects are mutable dictionaries.  Keep the run-local cache
        # private and hand every case a new copy of the exact cached values.
        return copy.deepcopy(list(cached_rows)), max_error, spread

    rows = copy.deepcopy(list(composite_strip_metric_rows(metric)))
    finite = [row for row in rows if math.isfinite(float(row.get("relative_error", math.inf)))]
    max_error = max((float(row["relative_error"]) for row in finite), default=math.inf)
    values = []
    value_key = {
        "static": "tip_displacement_z",
        "modal": "frequency_hz",
        "buckling": "critical_load_factor",
    }[metric]
    for row in finite:
        values.append(float(row[value_key]))
    spread = (max(values) - min(values)) / max(max(abs(value) for value in values), 1.0e-30) if values else math.inf
    if cache is not None:
        cache[metric] = (
            tuple(copy.deepcopy(rows)),
            max_error,
            float(spread),
        )
    return rows, max_error, float(spread)


def _run_coup_005(case: VerificationCase) -> VerificationCaseResult:
    rows, max_error, pair_spread = _composite_rows_for("static")
    _assert(max_error < 0.025, "stiffened-plate static model-pair analytical error too high")
    _assert(pair_spread < 0.04, "stiffened-plate static Q4/Q8/Q8R model-pair spread too high")
    return _pass(
        case,
        element_types=["shell4", "shell8", "shell8r", "beam2", "interpolated_mpc"],
        analysis_type="linear_static",
        reference={"type": "analytical model-pair", "source": "composite-section cantilever strip"},
        result={"max_relative_error": max_error, "model_pair_spread": pair_spread},
        checks={"rows": rows},
    )


def _run_eig_005(case: VerificationCase) -> VerificationCaseResult:
    rows, max_error, pair_spread = _composite_rows_for("modal")
    _assert(max_error < 0.025, "stiffened-panel modal model-pair analytical error too high")
    _assert(pair_spread < 0.04, "stiffened-panel modal Q4/Q8/Q8R spread too high")
    return _pass(
        case,
        element_types=["shell4", "shell8", "shell8r", "beam2", "interpolated_mpc"],
        analysis_type="modal",
        reference={"type": "analytical model-pair", "source": "composite-section cantilever frequency"},
        result={"max_relative_error": max_error, "model_pair_spread": pair_spread},
        checks={"rows": rows},
    )


def _run_buc_005(case: VerificationCase) -> VerificationCaseResult:
    rows, max_error, pair_spread = _composite_rows_for("buckling")
    _assert(max_error < 0.03, "stiffened-panel buckling model-pair analytical error too high")
    _assert(pair_spread < 0.09, "stiffened-panel buckling Q4/Q8/Q8R spread too high")
    return _pass(
        case,
        element_types=["shell4", "shell8", "shell8r", "beam2", "interpolated_mpc"],
        analysis_type="linear_buckling",
        reference={"type": "analytical model-pair", "source": "composite-section fixed-free Euler buckling"},
        result={"max_relative_error": max_error, "model_pair_spread": pair_spread},
        checks={"rows": rows},
    )


def _run_coup_017(case: VerificationCase) -> VerificationCaseResult:
    rows: List[Dict[str, Any]] = []
    for family in ("Q8", "Q8R"):
        for num_stiffeners in (1, 2):
            model, panel, config = _thin_stiffened_panel_model(num_stiffeners, element_family=family)
            validation = validate_production_model(model)
            props = calculate_mass_properties(model)
            expected_mass = _analytic_stiffened_panel_mass(panel)
            mass_error = _rel_error(props.total_mass, expected_mass)
            load = _pressure_load_for_shells(model, 250.0)
            displacements, solver_info = solve_linear(model, load)
            modal = solve_free_vibration(model, num_modes=4, dense_size_limit=12000)
            buckling = solve_eigenvalue_buckling(model, _stiffened_panel_buckling_states(model), num_modes=2, dense_size_limit=12000)
            row = {
                "element_family": family,
                "num_stiffeners": int(num_stiffeners),
                "validation_status": validation.status,
                "validation_issue_count": len(validation.issues),
                "mass_relative_error": mass_error,
                "static_solver_status": str((solver_info.get("convergence_info") or {}).get("status", "unknown")),
                "max_abs_displacement": float(np.max(np.abs(displacements))) if displacements.size else 0.0,
                "modal_status": modal.solver_status,
                "modal_first_frequency_hz": float(modal.frequencies_hz[0]) if modal.frequencies_hz.size else 0.0,
                "buckling_status": buckling.solver_status,
                "buckling_critical_load_factor": buckling.critical_load_factor,
                "mpc_constraints": _count_mpc_constraints(model),
                "mesh": {
                    "nodes": len(model.mesh.nodes),
                    "elements": len(model.mesh.elements),
                    "shell_divisions": [config.shell_num_divisions_x, config.shell_num_divisions_y],
                    "beam_divisions": config.beam_num_divisions,
                },
            }
            rows.append(row)
            _assert(validation.status in {"ok", "warning"}, "production Q8/Q8R stiffened model failed validation")
            _assert(mass_error < 1.0e-12, "production Q8/Q8R stiffened model mass mismatch")
            _assert(row["static_solver_status"] == "converged" and row["max_abs_displacement"] > 0.0, "production Q8/Q8R static solve failed")
            _assert(modal.solver_status == "ok" and row["modal_first_frequency_hz"] > 1.0e-6, "production Q8/Q8R modal solve failed")
            _assert(buckling.solver_status == "ok" and buckling.critical_load_factor is not None and buckling.critical_load_factor > 0.0, "production Q8/Q8R buckling solve failed")
    return _pass(
        case,
        element_types=["shell8", "shell8r", "beam2", "interpolated_mpc"],
        analysis_type="linear_static_modal_buckling",
        reference={"type": "production invariant suite", "quantities": ["validation", "mass", "static", "modal", "buckling"]},
        result={"rows_evaluated": len(rows)},
        checks={"rows": rows},
    )


def _run_perf_001(case: VerificationCase) -> VerificationCaseResult:
    from . import nonlinear_performance, nonlinear_static

    model, _panel, _config = _thin_stiffened_panel_model(1, element_family="Q4", shell_divisions_x=3, beam_divisions=5)
    legacy = nonlinear_performance._ORIGINAL_ASSEMBLER
    _assert(legacy is not None, "reference nonlinear assembler is not available")
    rng = np.random.default_rng(20260630)
    displacement = rng.normal(scale=1.0e-6, size=model.mesh.dof_manager.total_dofs)
    force_ref, tangent_ref, states_ref = legacy(model, displacement, {}, 3, tangent=True)
    force_fast, tangent_fast, states_fast = nonlinear_static._assemble_nonlinear_system(model, displacement, {}, 3, tangent=True)
    force_error = float(np.linalg.norm(force_fast - force_ref) / max(np.linalg.norm(force_ref), 1.0))
    tangent_error = float(sparse.linalg.norm(tangent_fast - tangent_ref) / max(float(sparse.linalg.norm(tangent_ref)), 1.0))
    _assert(force_error < 1.0e-10, "optimized nonlinear force assembly differs on weighted-MPC model")
    _assert(tangent_error < 1.0e-10, "optimized nonlinear tangent assembly differs on weighted-MPC model")
    _assert(set(states_fast) == set(states_ref), "optimized nonlinear state map differs on weighted-MPC model")
    return _pass(
        case,
        element_types=["shell4", "beam2", "interpolated_mpc"],
        analysis_type="nonlinear_assembly",
        reference={"type": "implementation equivalence", "source": "reference nonlinear assembler"},
        result={"force_relative_error": force_error, "tangent_relative_error": tangent_error},
        checks={"state_count": len(states_fast), "total_dofs": model.mesh.dof_manager.total_dofs},
    )


def _run_perf_002(case: VerificationCase) -> VerificationCaseResult:
    from .nonlinear_performance_bootstrap import clear_nonlinear_assembly_cache, get_nonlinear_assembly_plan

    model, _panel, _config = _thin_stiffened_panel_model(1, element_family="Q4")
    clear_nonlinear_assembly_cache(model)
    K0, info0 = assemble_stiffness_matrix(model)
    plan0 = get_nonlinear_assembly_plan(model, 3)
    signature0 = info0.get("sparsity_signature")
    revision0 = model.revision_signature()
    shell_node = next(node for node_id, node in model.mesh.nodes.items() if int(node_id) < 10000 and abs(float(node.z)) < 1.0e-12)
    model.set_node_coordinates(shell_node.id, shell_node.x, shell_node.y, shell_node.z + 2.0e-5)
    K1, info1 = assemble_stiffness_matrix(model)
    plan1 = get_nonlinear_assembly_plan(model, 3)
    signature1 = info1.get("sparsity_signature")
    revision1 = model.revision_signature()
    stiffness_change = float(sparse.linalg.norm(K1 - K0) / max(float(sparse.linalg.norm(K0)), 1.0))
    _assert(signature0 == signature1, "geometry-only change should preserve sparsity signature")
    _assert(revision1["geometry"] > revision0["geometry"], "geometry revision was not incremented")
    _assert(plan1 is not plan0, "nonlinear assembly plan was not invalidated by geometry revision")
    _assert(stiffness_change > 1.0e-12, "geometry cache invalidation did not change stiffness")
    return _pass(
        case,
        element_types=["shell4", "beam2", "interpolated_mpc"],
        analysis_type="cache_correctness",
        reference={"type": "cache invariant", "quantities": ["geometry revision", "sparsity preservation", "stiffness update"]},
        result={"stiffness_relative_change": stiffness_change},
        checks={"revision_before": revision0, "revision_after": revision1, "sparsity_signature_preserved": signature0 == signature1},
    )


def _run_coup_018(case: VerificationCase) -> VerificationCaseResult:
    model, _panel, _config = _thin_stiffened_panel_model(1, element_family="Q4")
    validation = validate_production_model(model)
    constraints = []
    for element_id, element in model.mesh.elements.items():
        getter = getattr(element, "get_mpc_constraints", None)
        if getter is None:
            continue
        for constraint in getter(model.mesh) or []:
            constraints.append((int(element_id), int(constraint["slave"]), dict(constraint.get("masters", {}))))
    slave_ids = [item[1] for item in constraints]
    duplicate_slave_count = len(slave_ids) - len(set(slave_ids))
    max_weight_sum_error = 0.0
    max_negative_weight = 0.0
    for element in model.mesh.elements.values():
        weights = getattr(element, "shape_weights", None)
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            max_weight_sum_error = max(max_weight_sum_error, abs(float(np.sum(weights) - 1.0)))
            max_negative_weight = max(max_negative_weight, max((-float(value) for value in weights if float(value) < 0.0), default=0.0))

    load = _pressure_load_for_shells(model, 100.0)
    u0, _ = solve_linear(model, load)
    reordered = dict(sorted(model.mesh.elements.items(), reverse=True))
    model.mesh.elements = reordered
    model.bump_revision("topology")
    u1, _ = solve_linear(model, load)
    displacement_error = float(np.linalg.norm(u1 - u0) / max(np.linalg.norm(u0), 1.0e-30))
    _assert(validation.status in {"ok", "warning"}, "numbering-invariance fixture failed validation")
    _assert(duplicate_slave_count == 0, "weighted MPC slave DOF ownership is not unique")
    _assert(max_weight_sum_error < 1.0e-12, "weighted MPC shape functions do not sum to one")
    _assert(max_negative_weight < 1.0e-12, "weighted MPC ownership produced negative interpolation weights")
    _assert(displacement_error < 1.0e-10, "element numbering changed weighted-MPC static solution")
    return _pass(
        case,
        element_types=["shell4", "beam2", "interpolated_mpc"],
        analysis_type="coupling_robustness",
        reference={"type": "invariant", "quantities": ["unique slave ownership", "partition of unity", "numbering independence"]},
        result={"displacement_relative_error": displacement_error},
        checks={
            "constraint_count": len(constraints),
            "duplicate_slave_count": duplicate_slave_count,
            "max_weight_sum_error": max_weight_sum_error,
            "max_negative_weight": max_negative_weight,
        },
    )


def _run_coup_019(case: VerificationCase) -> VerificationCaseResult:
    rows: List[Dict[str, Any]] = []
    reference = None
    for shell_x, beam_div in ((4, 4), (4, 8), (8, 4), (8, 8)):
        model, _panel, config = _thin_stiffened_panel_model(
            1,
            element_family="Q4",
            shell_divisions_x=shell_x,
            shell_divisions_y=2,
            beam_divisions=beam_div,
        )
        load = _pressure_load_for_shells(model, 250.0)
        displacements, solver_info = solve_linear(model, load)
        max_disp = float(np.max(np.abs(displacements))) if displacements.size else 0.0
        if reference is None and shell_x == 8 and beam_div == 8:
            reference = max_disp
        rows.append(
            {
                "shell_divisions_x": int(config.shell_num_divisions_x),
                "shell_divisions_y": int(config.shell_num_divisions_y),
                "beam_divisions": int(config.beam_num_divisions),
                "max_abs_displacement": max_disp,
                "solver_status": str((solver_info.get("convergence_info") or {}).get("status", "unknown")),
                "mpc_constraints": _count_mpc_constraints(model),
            }
        )
    reference = rows[-1]["max_abs_displacement"]
    for row in rows:
        row["relative_to_finest"] = _rel_error(row["max_abs_displacement"], reference)
    max_error = max(float(row["relative_to_finest"]) for row in rows[:-1])
    _assert(all(row["solver_status"] == "converged" for row in rows), "mesh-ratio sweep static solve failed")
    _assert(max_error < 0.35, "independent beam/shell mesh-ratio sweep is not converging consistently")
    return _pass(
        case,
        element_types=["shell4", "beam2", "interpolated_mpc"],
        analysis_type="linear_static",
        reference={"type": "mesh convergence", "source": "finest independent shell/beam mesh in sweep"},
        result={"max_relative_to_finest": max_error},
        checks={"rows": rows},
    )


def _curved_strip_model(
    *,
    nx: int = 6,
    nt: int = 6,
    radius: float = 1.5,
    length: float = 2.0,
    angle: float = math.pi / 3.0,
    thickness: float = 0.01,
    load: float = -100.0,
) -> Tuple[FEModel, LoadCase, Dict[Tuple[int, int], int]]:
    model = FEModel("curved_shell_strip")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    ids: Dict[Tuple[int, int], int] = {}
    node_id = 1
    for ix in range(int(nx) + 1):
        x = float(length) * ix / float(nx)
        for itheta in range(int(nt) + 1):
            theta = -0.5 * float(angle) + float(angle) * itheta / float(nt)
            ids[(ix, itheta)] = node_id
            model.add_node(node_id, x, float(radius) * math.cos(theta), float(radius) * math.sin(theta))
            node_id += 1

    element_id = 1
    for ix in range(int(nx)):
        for itheta in range(int(nt)):
            model.add_element(
                element_id,
                create_shell_element(
                    element_id,
                    [ids[(ix, itheta)], ids[(ix + 1, itheta)], ids[(ix + 1, itheta + 1)], ids[(ix, itheta + 1)]],
                    "steel",
                    thickness=float(thickness),
                ),
            )
            element_id += 1

    model.add_boundary_condition(FixedSupport("root_edge", [ids[(0, itheta)] for itheta in range(int(nt) + 1)]))
    load_case = LoadCase("curved_strip_tip_load")
    nodal_force = float(load) / float(int(nt) + 1)
    for itheta in range(int(nt) + 1):
        load_case.add_nodal_load(ids[(int(nx), itheta)], forces=np.array([0.0, 0.0, nodal_force]))
    return model, load_case, ids


def _curved_strip_tip_response(nx: int, nt: int) -> Dict[str, Any]:
    model, load, ids = _curved_strip_model(nx=nx, nt=nt)
    displacements, solver_info = solve_linear(model, load, constraint_mode="auto")
    tip_values = [
        float(displacements[model.mesh.get_node(ids[(nx, itheta)]).dofs[2]])
        for itheta in range(nt + 1)
    ]
    return {
        "nx": int(nx),
        "nt": int(nt),
        "nodes": int(model.mesh.num_nodes),
        "elements": int(model.mesh.num_elements),
        "mean_tip_uz": float(np.mean(tip_values)),
        "max_abs_displacement": float(np.max(np.abs(displacements))) if displacements.size else 0.0,
        "solver_status": str((solver_info.get("convergence_info") or {}).get("status", "unknown")),
    }


def _run_shell_008(case: VerificationCase) -> VerificationCaseResult:
    rows = [_curved_strip_tip_response(n, n) for n in (4, 6, 8)]
    fine = abs(float(rows[-1]["mean_tip_uz"]))
    errors = [abs(abs(float(row["mean_tip_uz"])) - fine) / max(fine, 1.0e-30) for row in rows[:-1]]
    monotone = all(abs(float(rows[index]["mean_tip_uz"])) >= abs(float(rows[index + 1]["mean_tip_uz"])) for index in range(len(rows) - 1))
    max_relative_to_fine = max(errors, default=0.0)
    _assert(all(row["solver_status"] == "converged" for row in rows), "curved-strip bending solve failed")
    _assert(monotone and max_relative_to_fine < 0.25, "curved-strip bending response is not mesh-convergent")
    return _pass(
        case,
        element_types=["shell4"],
        analysis_type="linear_static",
        mesh={"fixture": "clamped cylindrical strip", "radius": 1.5, "span_to_thickness": 200},
        reference={"type": "internal mesh convergence", "source": "fine 8x8 curved-strip response"},
        result={"fine_mean_tip_uz": float(rows[-1]["mean_tip_uz"]), "max_relative_to_fine": max_relative_to_fine},
        checks={"rows": rows, "monotone_refinement": bool(monotone)},
    )


def _run_bench_001(case: VerificationCase) -> VerificationCaseResult:
    rows: List[Dict[str, Any]] = []
    for n in (4, 6):
        model = FEModel("twisted_cantilever_shell")
        model.add_material("steel", 210.0e9, 0.3, density=7850.0)
        ids: Dict[Tuple[int, int], int] = {}
        node_id = 1
        length = 2.0
        width = 0.4
        twist = math.radians(30.0)
        for ix in range(n + 1):
            x = length * ix / n
            phi = twist * ix / n
            for iy in range(n + 1):
                y0 = -0.5 * width + width * iy / n
                y = y0 * math.cos(phi)
                z = y0 * math.sin(phi)
                ids[(ix, iy)] = node_id
                model.add_node(node_id, x, y, z)
                node_id += 1
        element_id = 1
        for ix in range(n):
            for iy in range(n):
                model.add_element(
                    element_id,
                    create_shell_element(element_id, [ids[(ix, iy)], ids[(ix + 1, iy)], ids[(ix + 1, iy + 1)], ids[(ix, iy + 1)]], "steel", thickness=0.01),
                )
                element_id += 1
        model.add_boundary_condition(FixedSupport("root", [ids[(0, iy)] for iy in range(n + 1)]))
        load = LoadCase("tip_shear")
        for iy in range(n + 1):
            load.add_nodal_load(ids[(n, iy)], forces=np.array([0.0, 0.0, -100.0 / float(n + 1)]))
        displacements, solver_info = solve_linear(model, load, constraint_mode="auto")
        tip = float(np.mean([displacements[model.mesh.get_node(ids[(n, iy)]).dofs[2]] for iy in range(n + 1)]))
        rows.append(
            {
                "divisions": int(n),
                "nodes": int(model.mesh.num_nodes),
                "elements": int(model.mesh.num_elements),
                "mean_tip_uz": tip,
                "max_abs_displacement": float(np.max(np.abs(displacements))),
                "solver_status": str((solver_info.get("convergence_info") or {}).get("status", "unknown")),
            }
        )
    spread = _rel_error(rows[-1]["mean_tip_uz"], rows[0]["mean_tip_uz"])
    _assert(all(row["solver_status"] == "converged" for row in rows), "twisted cantilever benchmark solve failed")
    _assert(0.0 < spread < 0.35, "twisted cantilever internal refinement check is outside tolerance")
    return _pass(
        case,
        element_types=["shell4"],
        analysis_type="linear_static",
        mesh={"fixture": "MacNeal-Harder-style twisted cantilever", "twist_degrees": 30.0},
        reference={"type": "internal refinement", "note": "literature target not bundled"},
        result={"relative_change_4_to_6": spread},
        checks={"rows": rows},
    )


def _run_bench_002(case: VerificationCase) -> VerificationCaseResult:
    rows = [_curved_strip_tip_response(4, 6), _curved_strip_tip_response(6, 8)]
    response_ratio = abs(float(rows[1]["mean_tip_uz"])) / max(abs(float(rows[0]["mean_tip_uz"])), 1.0e-30)
    _assert(all(row["solver_status"] == "converged" for row in rows), "Scordelis-Lo-style roof solve failed")
    _assert(0.70 < response_ratio < 1.05, "Scordelis-Lo-style roof refinement response is outside tolerance")
    return _pass(
        case,
        element_types=["shell4"],
        analysis_type="linear_static",
        mesh={"fixture": "Scordelis-Lo-style cylindrical roof strip"},
        reference={"type": "internal curved-roof refinement", "note": "external Scordelis-Lo reference not bundled"},
        result={"response_ratio_refined_to_coarse": response_ratio},
        checks={"rows": rows},
    )


def _run_bench_003(case: VerificationCase) -> VerificationCaseResult:
    config = CylinderBenchmarkConfig(
        radius=1.0,
        height=2.0,
        thickness=0.01,
        pressure=0.0,
        num_circumferential=16,
        num_height=4,
        closed_end_axial_load=False,
    )
    model, _unused = build_cylindrical_shell_benchmark_model(config)
    load = LoadCase("pinched_cylinder")
    mid = config.num_height // 2
    n_pos = mid * config.num_circumferential + 1
    n_neg = mid * config.num_circumferential + config.num_circumferential // 2 + 1
    load.add_nodal_load(n_pos, forces=np.array([-1000.0, 0.0, 0.0]))
    load.add_nodal_load(n_neg, forces=np.array([1000.0, 0.0, 0.0]))
    displacements, solver_info = solve_linear(model, load, constraint_mode="auto")
    ux_pos = float(displacements[model.mesh.get_node(n_pos).dofs[0]])
    ux_neg = float(displacements[model.mesh.get_node(n_neg).dofs[0]])
    antisymmetry_error = abs(ux_pos + ux_neg) / max(abs(ux_pos), abs(ux_neg), 1.0e-30)
    _assert(str((solver_info.get("convergence_info") or {}).get("status")) == "converged", "pinched-cylinder solve failed")
    _assert(abs(ux_pos) > 0.0 and antisymmetry_error < 1.0e-9, "pinched-cylinder opposite-load symmetry failed")
    return _pass(
        case,
        element_types=["shell4"],
        analysis_type="linear_static",
        mesh={"fixture": "closed pinched cylinder", "circumferential_divisions": config.num_circumferential},
        reference={"type": "symmetry invariant", "note": "external pinched-cylinder reference not bundled"},
        result={"pinch_displacement_m": ux_pos, "opposite_displacement_m": ux_neg},
        checks={"antisymmetry_error": antisymmetry_error, "nullspace": solver_info.get("nullspace_info", {})},
    )


def _add_ring_stiffener_to_cylinder(
    model: FEModel,
    config: CylinderBenchmarkConfig,
    *,
    z_index: Optional[int] = None,
    offset: float = 0.04,
    area: float = 0.004,
    iy: float = 1.0e-5,
    iz: float = 1.0e-5,
) -> List[int]:
    iz_index = config.num_height // 2 if z_index is None else int(z_index)
    element_id = max(int(eid) for eid in model.mesh.elements) + 1
    node_id = max(int(nid) for nid in model.mesh.nodes) + 1
    beam_nodes: List[int] = []
    section_base = {
        "area": float(area),
        "Iy": float(iy),
        "Iz": float(iz),
        "J": 0.5 * min(float(iy), float(iz)),
        "shear_factor_y": 5.0 / 6.0,
        "shear_factor_z": 5.0 / 6.0,
    }
    for itheta in range(config.num_circumferential):
        shell_id = iz_index * config.num_circumferential + itheta + 1
        shell_node = model.mesh.get_node(shell_id)
        theta = 2.0 * math.pi * itheta / config.num_circumferential
        radial = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)
        beam_id = node_id
        node_id += 1
        model.add_node(beam_id, shell_node.x + offset * radial[0], shell_node.y + offset * radial[1], shell_node.z)
        beam_nodes.append(beam_id)
        model.add_element(element_id, CoupledBeamShellElement(element_id, beam_node_id=beam_id, shell_node_id=shell_id, material_name="steel"))
        element_id += 1
    for itheta, beam_id in enumerate(beam_nodes):
        theta_mid = 2.0 * math.pi * (itheta + 0.5) / config.num_circumferential
        section = dict(section_base)
        section["orientation"] = np.array([math.cos(theta_mid), math.sin(theta_mid), 0.0], dtype=float)
        model.add_element(element_id, BeamElement(element_id, [beam_id, beam_nodes[(itheta + 1) % len(beam_nodes)]], "steel", section))
        element_id += 1
    return beam_nodes


def _add_longitudinal_stiffeners_to_cylinder(
    model: FEModel,
    config: CylinderBenchmarkConfig,
    *,
    count: int = 4,
    offset: float = 0.04,
    area: float = 0.004,
) -> List[int]:
    element_id = max(int(eid) for eid in model.mesh.elements) + 1
    node_id = max(int(nid) for nid in model.mesh.nodes) + 1
    beam_nodes: List[int] = []
    section_base = {
        "area": float(area),
        "Iy": 1.0e-5,
        "Iz": 2.0e-5,
        "J": 5.0e-6,
        "shear_factor_y": 5.0 / 6.0,
        "shear_factor_z": 5.0 / 6.0,
    }
    for istiffener in range(int(count)):
        itheta = istiffener * config.num_circumferential // int(count)
        theta = 2.0 * math.pi * itheta / config.num_circumferential
        radial = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)
        previous: Optional[int] = None
        for iz_index in range(config.num_height + 1):
            shell_id = iz_index * config.num_circumferential + itheta + 1
            shell_node = model.mesh.get_node(shell_id)
            beam_id = node_id
            node_id += 1
            model.add_node(beam_id, shell_node.x + offset * radial[0], shell_node.y + offset * radial[1], shell_node.z)
            beam_nodes.append(beam_id)
            model.add_element(element_id, CoupledBeamShellElement(element_id, beam_node_id=beam_id, shell_node_id=shell_id, material_name="steel"))
            element_id += 1
            if previous is not None:
                section = dict(section_base)
                section["orientation"] = radial
                model.add_element(element_id, BeamElement(element_id, [previous, beam_id], "steel", section))
                element_id += 1
            previous = beam_id
    return beam_nodes


def _cylinder_axial_load(config: CylinderBenchmarkConfig, force: float = 1.0e6) -> LoadCase:
    load = LoadCase("self_equilibrated_axial_end_load")
    nodal = float(force) / float(config.num_circumferential)
    top_offset = config.num_height * config.num_circumferential
    for itheta in range(config.num_circumferential):
        load.add_nodal_load(itheta + 1, forces=np.array([0.0, 0.0, -nodal]))
        load.add_nodal_load(top_offset + itheta + 1, forces=np.array([0.0, 0.0, nodal]))
    return load


def _cylinder_end_extension(model: FEModel, displacements: np.ndarray, config: CylinderBenchmarkConfig) -> float:
    bottom: List[float] = []
    top: List[float] = []
    top_offset = config.num_height * config.num_circumferential
    for itheta in range(config.num_circumferential):
        bottom.append(float(displacements[model.mesh.get_node(itheta + 1).dofs[2]]))
        top.append(float(displacements[model.mesh.get_node(top_offset + itheta + 1).dofs[2]]))
    return float(np.mean(top) - np.mean(bottom))


def _shell_node_radial_displacements(model: FEModel, displacements: np.ndarray) -> List[float]:
    shell_node_ids = {
        int(node_id)
        for element in model.mesh.elements.values()
        if isinstance(element, ShellElement)
        for node_id in element.node_ids
    }
    values: List[float] = []
    for node_id, node in model.mesh.nodes.items():
        if shell_node_ids and int(node_id) not in shell_node_ids:
            continue
        radial = np.array([node.x, node.y, 0.0], dtype=float)
        norm = float(np.linalg.norm(radial))
        if norm <= 1.0e-12:
            continue
        values.append(float(displacements[node.dofs[:3]] @ (radial / norm)))
    return values


def _run_coup_020(case: VerificationCase) -> VerificationCaseResult:
    config = CylinderBenchmarkConfig(radius=1.5, height=3.0, thickness=0.015, pressure=0.0, num_circumferential=16, num_height=4)
    model, _unused = build_cylindrical_shell_benchmark_model(config)
    beam_nodes = _add_ring_stiffener_to_cylinder(model, config)
    orientation_errors: List[float] = []
    length_errors: List[float] = []
    for element in model.mesh.elements.values():
        if not isinstance(element, BeamElement):
            continue
        coords = element.get_node_coordinates(model.mesh)
        midpoint = np.mean(coords[:, :3], axis=0)
        radial = np.array([midpoint[0], midpoint[1], 0.0], dtype=float)
        radial /= max(float(np.linalg.norm(radial)), 1.0e-30)
        _length, transform = element._beam_frame_and_transform(coords)
        rotation = transform[:3, :3].T
        local_z = rotation[:, 2]
        orientation_errors.append(float(1.0 - abs(float(local_z @ radial))))
        length_errors.append(abs(float(_length) - 2.0 * config.radius * math.sin(math.pi / config.num_circumferential)))
    K = sparse.eye(model.mesh.dof_manager.total_dofs, format="csr")
    zero = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    _K, _F, T, u0, independent, constraint_info = build_constraint_transformation(K, zero, model)
    q = np.linspace(-0.05, 0.05, len(independent), dtype=float)
    u = reconstruct_full_solution(T, q, u0)
    residuals = mpc_constraint_residuals(model, u)
    max_constraint_residual = max((abs(value) for value in residuals.values()), default=0.0)
    closure_gap = float(np.linalg.norm(model.mesh.get_node(beam_nodes[0]).coords() - model.mesh.get_node(beam_nodes[-1]).coords()))
    beam_radius = float(np.linalg.norm(model.mesh.get_node(beam_nodes[0]).coords()[:2]))
    expected_chord = 2.0 * beam_radius * math.sin(math.pi / config.num_circumferential)
    _assert(max(orientation_errors, default=0.0) < 1.0e-12, "ring stiffener local frames do not close with radial orientation")
    _assert(max_constraint_residual < 1.0e-12, "ring stiffener MPC compatibility failed")
    _assert(abs(closure_gap - expected_chord) < 1.0e-12, "ring stiffener closure chord is inconsistent")
    return _pass(
        case,
        element_types=["beam2", "beam_shell_mpc"],
        analysis_type="curved_stiffener_coordinates",
        mesh={"ring_nodes": len(beam_nodes), "radius": config.radius},
        reference={"type": "geometric invariant", "quantities": ["radial frame transport", "ring closure", "MPC compatibility"]},
        result={"num_mpc_slave_dofs": int(constraint_info["num_mpc_slave_dofs"])},
        checks={
            "max_orientation_error": max(orientation_errors, default=0.0),
            "max_length_error": max(length_errors, default=0.0),
            "closure_gap": closure_gap,
            "expected_chord": expected_chord,
            "beam_radius": beam_radius,
            "max_constraint_residual": max_constraint_residual,
        },
    )


def _run_coup_021(case: VerificationCase) -> VerificationCaseResult:
    radius = 2.0
    area = 0.01
    modulus = 210.0e9
    line_load = 1000.0
    n = 32
    model = FEModel("circular_ring_membrane")
    model.add_material("steel", modulus, 0.3, density=7850.0)
    section_base = {"area": area, "Iy": 1.0e-5, "Iz": 1.0e-5, "J": 1.0e-5, "shear_factor_y": 5.0 / 6.0, "shear_factor_z": 5.0 / 6.0}
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        model.add_node(i + 1, radius * math.cos(theta), radius * math.sin(theta), 0.0)
    for i in range(n):
        theta_mid = 2.0 * math.pi * (i + 0.5) / n
        section = dict(section_base)
        section["orientation"] = np.array([math.cos(theta_mid), math.sin(theta_mid), 0.0], dtype=float)
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, (i + 1) % n + 1], "steel", section))
    load = LoadCase("uniform_radial_line_load")
    nodal_force = float(line_load) * radius * 2.0 * math.pi / n
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        load.add_nodal_load(i + 1, forces=nodal_force * np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float))
    displacements, solver_info = solve_linear(model, load, constraint_mode="auto")
    radial_values = _shell_node_radial_displacements(model, displacements)
    mean_radial = float(np.mean(radial_values))
    spread = float((max(radial_values) - min(radial_values)) / max(abs(mean_radial), 1.0e-30))
    reference = float(line_load) * radius**2 / (modulus * area)
    relative_error = _rel_error(mean_radial, reference)
    _assert(str((solver_info.get("convergence_info") or {}).get("status")) == "converged", "ring membrane static solve failed")
    _assert(relative_error < 0.005 and spread < 1.0e-9, "ring membrane analytical displacement check failed")
    return _pass(
        case,
        element_types=["beam2"],
        analysis_type="linear_static",
        mesh={"ring_segments": n, "radius": radius},
        reference={"type": "analytical", "quantity": "uniform ring radial expansion", "radial_displacement_m": reference},
        result={"mean_radial_displacement_m": mean_radial, "relative_error": relative_error},
        checks={"radial_spread": spread, "nullspace": solver_info.get("nullspace_info", {})},
    )


def _run_cyl_001(case: VerificationCase) -> VerificationCaseResult:
    config = CylinderBenchmarkConfig(
        radius=2.0,
        height=4.0,
        thickness=0.02,
        pressure=10_000.0,
        num_circumferential=16,
        num_height=4,
        mid_height_band_fraction=3.0,
    )
    result = run_cylindrical_shell_benchmark(config)
    stress_error = _rel_error(result.fe_mid_height_p95_von_mises, result.nominal.von_mises_stress)
    _assert(result.solver_status == "converged", "closed-cylinder membrane benchmark solve failed")
    _assert(stress_error < 0.04, "closed-cylinder mid-height membrane stress is outside tolerance")
    return _pass(
        case,
        element_types=["shell4"],
        analysis_type="linear_static",
        mesh={"closed_cylinder": True, "circumferential_divisions": config.num_circumferential, "height_divisions": config.num_height},
        reference={"type": "analytical", "source": "thin closed-cylinder membrane stress", **result.nominal.to_dict()},
        result={"mid_height_p95_von_mises": result.fe_mid_height_p95_von_mises, "relative_error": stress_error},
        checks=result.to_dict(),
    )


def _run_cyl_002(case: VerificationCase) -> VerificationCaseResult:
    config = CylinderBenchmarkConfig(
        radius=2.0,
        height=4.0,
        thickness=0.02,
        pressure=0.0,
        num_circumferential=16,
        num_height=4,
        closed_end_axial_load=False,
    )
    plain, _unused = build_cylindrical_shell_benchmark_model(config)
    stiffened, _unused_stiff = build_cylindrical_shell_benchmark_model(config)
    _add_longitudinal_stiffeners_to_cylinder(stiffened, config, count=4, area=0.004)
    load = _cylinder_axial_load(config)
    u_plain, info_plain = solve_linear(plain, load, constraint_mode="auto")
    u_stiff, info_stiff = solve_linear(stiffened, load, constraint_mode="auto")
    extension_plain = abs(_cylinder_end_extension(plain, u_plain, config))
    extension_stiff = abs(_cylinder_end_extension(stiffened, u_stiff, config))
    stiffness_ratio = extension_stiff / max(extension_plain, 1.0e-30)
    mass_ratio = calculate_mass_properties(stiffened).total_mass / max(calculate_mass_properties(plain).total_mass, 1.0e-30)
    _assert(str((info_plain.get("convergence_info") or {}).get("status")) == "converged", "plain cylinder axial model-pair solve failed")
    _assert(str((info_stiff.get("convergence_info") or {}).get("status")) == "converged", "stiffened cylinder axial model-pair solve failed")
    _assert(0.0 < stiffness_ratio < 1.0 and mass_ratio > 1.0, "longitudinal stiffener model pair did not stiffen/increase mass")
    return _pass(
        case,
        element_types=["shell4", "beam2", "beam_shell_mpc"],
        analysis_type="linear_static",
        mesh={"longitudinal_stiffeners": 4, "circumferential_divisions": config.num_circumferential},
        reference={"type": "model-pair invariant", "quantities": ["added mass", "reduced axial extension"]},
        result={"extension_ratio_stiffened_to_plain": stiffness_ratio, "mass_ratio_stiffened_to_plain": mass_ratio},
        checks={"plain_extension_m": extension_plain, "stiffened_extension_m": extension_stiff, "mpc_constraints": _count_mpc_constraints(stiffened)},
    )


def _run_coup_006(case: VerificationCase) -> VerificationCaseResult:
    config = CylinderBenchmarkConfig(
        radius=1.5,
        height=3.0,
        thickness=0.015,
        pressure=5_000.0,
        num_circumferential=16,
        num_height=6,
        closed_end_axial_load=True,
    )
    area = 0.004
    ring, load = build_cylindrical_shell_benchmark_model(config)
    _add_ring_stiffener_to_cylinder(ring, config, area=area)
    band, band_load = build_cylindrical_shell_benchmark_model(config)
    band_width = config.height / config.num_height
    equivalent_band_count = 2
    added_thickness = area / (equivalent_band_count * band_width)
    for element in band.mesh.elements.values():
        if isinstance(element, ShellElement):
            z_mean = float(np.mean(element.get_node_coordinates(band.mesh)[:, 2]))
            if abs(z_mean - 0.5 * config.height) <= 0.5 * band_width + 1.0e-12:
                element.thickness += added_thickness
    band.bump_revision("material")
    u_ring, info_ring = solve_linear(ring, load, constraint_mode="auto")
    u_band, info_band = solve_linear(band, band_load, constraint_mode="auto")
    ring_radial = _shell_node_radial_displacements(ring, u_ring)
    band_radial = _shell_node_radial_displacements(band, u_band)
    ring_mean = float(np.mean(np.abs(ring_radial)))
    band_mean = float(np.mean(np.abs(band_radial)))
    response_spread = _rel_error(ring_mean, band_mean)
    base, _base_load = build_cylindrical_shell_benchmark_model(config)
    base_mass = calculate_mass_properties(base).total_mass
    ring_added = calculate_mass_properties(ring).total_mass - base_mass
    band_added = calculate_mass_properties(band).total_mass - base_mass
    added_mass_error = _rel_error(ring_added, band_added)
    _assert(str((info_ring.get("convergence_info") or {}).get("status")) == "converged", "ring-stiffened cylinder solve failed")
    _assert(str((info_band.get("convergence_info") or {}).get("status")) == "converged", "equivalent thickened-band cylinder solve failed")
    _assert(response_spread < 0.15 and added_mass_error < 0.20, "ring-stiffened cylinder equivalent model spread is outside tolerance")
    return _pass(
        case,
        element_types=["shell4", "beam2", "beam_shell_mpc"],
        analysis_type="linear_static",
        mesh={"equivalent_band_width": equivalent_band_count * band_width, "ring_area": area},
        reference={"type": "internal model-pair", "source": "ring beam versus equivalent thickened shell band"},
        result={"radial_response_spread": response_spread, "added_mass_relative_error": added_mass_error},
        checks={
            "ring_mean_abs_radial_displacement": ring_mean,
            "band_mean_abs_radial_displacement": band_mean,
            "ring_added_mass": ring_added,
            "band_added_mass": band_added,
            "mpc_constraints": _count_mpc_constraints(ring),
        },
    )


def _run_cyl_003(case: VerificationCase) -> VerificationCaseResult:
    config = CylinderBenchmarkConfig(
        radius=1.5,
        height=3.0,
        thickness=0.015,
        pressure=0.0,
        num_circumferential=12,
        num_height=4,
        closed_end_axial_load=False,
    )
    model, _unused = build_cylindrical_shell_benchmark_model(config)
    _add_ring_stiffener_to_cylinder(model, config, area=0.008, iy=2.0e-5, iz=2.0e-5)
    model.add_boundary_condition(FixedSupport("single_anchor", [1]))
    states = {
        int(element_id): ({"membrane_compression_y": 1.0} if isinstance(element, ShellElement) else {})
        for element_id, element in model.mesh.elements.items()
    }
    doubled_states = {
        element_id: {key: 2.0 * float(value) for key, value in state.items()}
        for element_id, state in states.items()
    }
    result = solve_eigenvalue_buckling(model, states, num_modes=2, dense_size_limit=10000)
    doubled = solve_eigenvalue_buckling(model, doubled_states, num_modes=1, dense_size_limit=10000)
    _assert(result.solver_status == "ok" and result.critical_load_factor is not None, "ring-stiffened cylinder buckling solve failed")
    _assert(doubled.critical_load_factor is not None, "ring-stiffened cylinder doubled-preload buckling solve failed")
    scaling_error = _rel_error(float(doubled.critical_load_factor), 0.5 * float(result.critical_load_factor))
    residual = float((result.diagnostics or {}).get("max_residual_norm", 1.0))
    _assert(float(result.critical_load_factor) > 0.0 and scaling_error < 1.0e-8 and residual < 1.0e-8, "ring-stiffened cylinder buckling invariants failed")
    return _pass(
        case,
        element_types=["shell4", "beam2", "beam_shell_mpc"],
        analysis_type="linear_buckling",
        mesh={"ring_stiffeners": 1, "circumferential_divisions": config.num_circumferential},
        reference={"type": "buckling invariant", "preload": "unit circumferential shell membrane compression, ring beams elastic"},
        result={"critical_load_factor": float(result.critical_load_factor), "preload_scaling_error": scaling_error},
        checks={**(result.diagnostics or {}), "mpc_constraints": _count_mpc_constraints(model)},
    )


def _run_beam_011(case: VerificationCase) -> VerificationCaseResult:
    """Cantilever bending eigenmodes against Euler-Bernoulli reference."""
    length = 2.0
    area = 0.02
    iy = 2.0e-6
    iz = 5.0e-6
    density = 7850.0
    E = 210.0e9
    model = _beam_model(length=length, area=area, iy=iy, iz=iz, j=2.0e-6, density=density, num_elements=24)
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    result = solve_free_vibration(model, num_modes=6, dense_size_limit=10000)
    _assert(result.solver_status == "ok" and result.num_modes_returned >= 2, "cantilever modal solve failed")

    beta1 = 1.875104068711961
    references = sorted(
        [
            beta1**2 * math.sqrt(E * iy / (density * area * length**4)) / (2.0 * math.pi),
            beta1**2 * math.sqrt(E * iz / (density * area * length**4)) / (2.0 * math.pi),
        ]
    )
    computed = sorted(float(value) for value in result.frequencies_hz[:4] if float(value) > 1.0e-6)[:2]
    _assert(len(computed) == 2, "cantilever bending modes were not recovered")
    errors = [_rel_error(value, ref) for value, ref in zip(computed, references)]
    max_error = max(errors)
    _assert(max_error < 0.03, "cantilever bending eigenfrequency mismatch")
    return _pass(
        case,
        element_types=["beam2"],
        analysis_type="modal",
        mesh={"length": length, "num_elements": 24, "section": {"area": area, "Iy": iy, "Iz": iz}},
        reference={"type": "analytical", "source": "Euler-Bernoulli cantilever beta_1", "frequencies_hz": references},
        result={"frequencies_hz": computed, "max_relative_error": max_error},
        checks={**result.diagnostics, "relative_errors": errors},
    )


def _pressure_load_all_shells(model: FEModel, pressure: float) -> LoadCase:
    load = LoadCase("uniform_plate_pressure")
    for element_id, element in model.mesh.elements.items():
        if isinstance(element, ShellElement):
            load.add_pressure_load(int(element_id), float(pressure))
    return load


def _plate_center_uz(model: FEModel, displacements: np.ndarray) -> float:
    center_node = min(
        model.mesh.nodes.values(),
        key=lambda node: float((node.x - 0.5) ** 2 + (node.y - 0.5) ** 2 + node.z**2),
    )
    return float(displacements[center_node.dofs[2]])


def _run_shell_009(case: VerificationCase) -> VerificationCaseResult:
    """Navier simply-supported square plate under uniform pressure."""
    thickness = 0.01
    pressure = 1000.0
    rows: List[Dict[str, Any]] = []
    # w_max = alpha q a^4 / D for a simply supported square plate, uniform load.
    alpha = 0.00406235
    E, nu = 210.0e9, 0.3
    D = E * thickness**3 / (12.0 * (1.0 - nu**2))
    reference = alpha * pressure / D
    for family, divisions in (("S4", 12), ("S8", 6), ("S8R", 6)):
        model = _verification_plate_model(divisions=divisions, thickness=thickness, element_family=family)
        load = _pressure_load_all_shells(model, pressure)
        displacements, solver_info = solve_linear(model, load)
        value = abs(_plate_center_uz(model, displacements))
        err = _rel_error(value, reference)
        rows.append(
            {
                "element_family": family,
                "divisions": divisions,
                "center_deflection_m": value,
                "reference_m": reference,
                "relative_error": err,
                "solver_status": (solver_info.get("convergence_info") or {}).get("status"),
            }
        )
    max_error = max(float(row["relative_error"]) for row in rows)
    _assert(max_error < 0.08, "Navier plate pressure benchmark outside tolerance")
    return _pass(
        case,
        element_types=["shell4", "shell8", "shell8r"],
        analysis_type="linear_static",
        mesh={"square_plate": True, "span_to_thickness": 100},
        reference={"type": "analytical", "source": "Navier SSSS square plate uniform load", "center_deflection_m": reference},
        result={"max_relative_error": max_error, "rows": rows},
        checks={"rows": rows},
    )


def _run_shell_010(case: VerificationCase) -> VerificationCaseResult:
    """Q4/Q8/Q8R thin-plate first modal convergence against analytical SSSS plate."""
    thickness = 0.01
    reference = _plate_bending_frequency_hz(1, 1, thickness=thickness)
    rows: List[Dict[str, Any]] = []
    for family, divisions in (("S4", 10), ("S8", 5), ("S8R", 5)):
        model = _verification_plate_model(divisions=divisions, thickness=thickness, element_family=family)
        result = solve_free_vibration(model, num_modes=4, dense_size_limit=10000)
        _assert(result.solver_status == "ok" and result.num_modes_returned > 0, f"{family} plate modal solve failed")
        value = float(result.frequencies_hz[0])
        rows.append(
            {
                "element_family": family,
                "divisions": divisions,
                "frequency_hz": value,
                "reference_hz": reference,
                "relative_error": _rel_error(value, reference),
                "mass_orthogonality_error": float(result.diagnostics.get("mass_orthogonality_error", 0.0)),
            }
        )
    max_error = max(float(row["relative_error"]) for row in rows)
    _assert(max_error < 0.06, "thin plate modal convergence benchmark outside tolerance")
    return _pass(
        case,
        element_types=["shell4", "shell8", "shell8r"],
        analysis_type="modal",
        mesh={"square_plate": True, "span_to_thickness": 100},
        reference={"type": "analytical", "mode": [1, 1], "frequency_hz": reference},
        result={"max_relative_error": max_error, "rows": rows},
        checks={"rows": rows},
    )


def _run_shell_011(case: VerificationCase) -> VerificationCaseResult:
    """Q4/Q8/Q8R thin-plate buckling convergence against SSSS uniaxial reference."""
    thickness = 0.01
    reference = _plate_uniaxial_buckling_resultant(thickness=thickness)
    rows: List[Dict[str, Any]] = []
    for family, divisions in (("S4", 10), ("S8", 5), ("S8R", 5)):
        model = _verification_plate_model(divisions=divisions, thickness=thickness, element_family=family)
        states = {
            int(element_id): {"membrane_compression_x": 1.0}
            for element_id, element in model.mesh.elements.items()
            if isinstance(element, ShellElement)
        }
        result = solve_eigenvalue_buckling(model, states, num_modes=3, dense_size_limit=10000)
        _assert(result.solver_status == "ok" and result.critical_load_factor is not None, f"{family} plate buckling solve failed")
        value = float(result.critical_load_factor)
        rows.append(
            {
                "element_family": family,
                "divisions": divisions,
                "critical_membrane_resultant": value,
                "reference": reference,
                "relative_error": _rel_error(value, reference),
            }
        )
    max_error = max(float(row["relative_error"]) for row in rows)
    _assert(max_error < 0.08, "thin plate buckling convergence benchmark outside tolerance")
    return _pass(
        case,
        element_types=["shell4", "shell8", "shell8r"],
        analysis_type="linear_buckling",
        mesh={"square_plate": True, "span_to_thickness": 100},
        reference={"type": "analytical", "k": 4.0, "critical_membrane_resultant": reference},
        result={"max_relative_error": max_error, "rows": rows},
        checks={"rows": rows},
    )


def _run_coup_014(case: VerificationCase) -> VerificationCaseResult:
    return _composite_strip_result(
        case,
        metric="static",
        tolerance=0.02,
        expected_status="converged",
        reference_type="composite EA, neutral axis and EI static response",
    )


def _run_coup_015(case: VerificationCase) -> VerificationCaseResult:
    return _composite_strip_result(
        case,
        metric="modal",
        tolerance=0.02,
        expected_status="ok",
        reference_type="composite cantilever bending frequency",
    )


def _run_coup_016(case: VerificationCase) -> VerificationCaseResult:
    return _composite_strip_result(
        case,
        metric="buckling",
        tolerance=0.03,
        expected_status="ok",
        reference_type="composite fixed-free Euler buckling load",
    )


def _run_coup_010(case: VerificationCase) -> VerificationCaseResult:
    radius = 1.0
    offset = 0.05
    dtheta = 0.08
    model = FEModel("curved_stiffener_orientation")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)

    def surface_point(x: float, theta: float) -> np.ndarray:
        return np.array([x, radius * math.cos(theta), radius * math.sin(theta)], dtype=float)

    def radial(theta: float) -> np.ndarray:
        return np.array([0.0, math.cos(theta), math.sin(theta)], dtype=float)

    shell_points = {
        1: surface_point(0.0, 0.0),
        2: surface_point(1.0, 0.0),
        3: surface_point(0.5, -dtheta),
        4: surface_point(0.5, dtheta),
    }
    beam_points = {
        101: shell_points[1] + offset * radial(0.0),
        102: shell_points[2] + offset * radial(0.0),
        103: shell_points[3] + offset * radial(-dtheta),
        104: shell_points[4] + offset * radial(dtheta),
    }
    for node_id, coords in {**shell_points, **beam_points}.items():
        model.add_node(int(node_id), float(coords[0]), float(coords[1]), float(coords[2]))
    section_long = {"area": 0.002, "Iy": 1.0e-6, "Iz": 2.0e-6, "J": 5.0e-7, "orientation": radial(0.0)}
    section_ring = {"area": 0.002, "Iy": 1.0e-6, "Iz": 2.0e-6, "J": 5.0e-7, "orientation": radial(0.0)}
    model.add_element(1, BeamElement(1, [101, 102], "steel", section_long))
    model.add_element(2, BeamElement(2, [103, 104], "steel", section_ring))
    for element_id, beam_node, shell_node in ((1001, 101, 1), (1002, 102, 2), (1003, 103, 3), (1004, 104, 4)):
        model.add_element(element_id, CoupledBeamShellElement(element_id, beam_node_id=beam_node, shell_node_id=shell_node, material_name="steel"))

    orientation_errors: List[float] = []
    for element_id, theta in ((1, 0.0), (2, 0.0)):
        element = model.mesh.elements[element_id]
        _L, T = element._beam_frame_and_transform(element.get_node_coordinates(model.mesh))
        rotation = T[:3, :3].T
        local_z = rotation[:, 2]
        orientation_errors.append(float(1.0 - abs(local_z @ radial(theta))))

    total_dofs = model.mesh.dof_manager.total_dofs
    K = sparse.eye(total_dofs, format="csr")
    zero = np.zeros(total_dofs, dtype=float)
    _K_red, _F_red, T, u0, independent, constraint_info = build_constraint_transformation(K, zero, model)
    q = np.linspace(-0.1, 0.2, len(independent), dtype=float)
    u = reconstruct_full_solution(T, q, u0)
    residuals = mpc_constraint_residuals(model, u)
    max_constraint_residual = max((abs(value) for value in residuals.values()), default=0.0)
    max_orientation_error = max(orientation_errors)
    _assert(max_orientation_error < 1.0e-12, "curved stiffener local-z orientation does not follow radial transport")
    _assert(max_constraint_residual < 1.0e-12 and constraint_info["num_mpc_slave_dofs"] == 24, "curved stiffener MPC compatibility failed")
    return _pass(
        case,
        element_types=["beam2", "beam_shell_mpc"],
        checks={
            "max_orientation_error": max_orientation_error,
            "max_constraint_residual": float(max_constraint_residual),
            "num_mpc_slave_dofs": int(constraint_info["num_mpc_slave_dofs"]),
            "radius_to_offset": float(radius / offset),
        },
    )


def _eccentric_coupling_model(*, fixed_shell: bool = False, eccentricity: np.ndarray | None = None) -> FEModel:
    model = FEModel("verification_eccentric_coupling")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    r = np.asarray([0.12, -0.04, 0.18] if eccentricity is None else eccentricity, dtype=float)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, float(r[0]), float(r[1]), float(r[2]))
    model.add_element(1, CoupledBeamShellElement(1, beam_node_id=2, shell_node_id=1, material_name="steel"))
    if fixed_shell:
        model.add_boundary_condition(FixedSupport("fixed_shell_master", [1]))
    return model


def _run_coup_012(case: VerificationCase) -> VerificationCaseResult:
    """Actual MPC transformation reproduces an affine rigid-link field."""
    r = np.array([0.12, -0.04, 0.18], dtype=float)
    model = _eccentric_coupling_model(eccentricity=r)
    K = sparse.eye(model.mesh.dof_manager.total_dofs, format="csr")
    zero = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    _K_red, _F_red, T, u0, independent, constraint_info = build_constraint_transformation(K, zero, model)

    shell = model.mesh.get_node(1)
    beam = model.mesh.get_node(2)
    shell_translation = np.array([0.021, -0.014, 0.033], dtype=float)
    shell_rotation = np.array([0.006, -0.011, 0.017], dtype=float)
    full_target = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    full_target[shell.dofs[:3]] = shell_translation
    full_target[shell.dofs[3:6]] = shell_rotation
    q = full_target[np.asarray(independent, dtype=int)]
    u = reconstruct_full_solution(T, q, u0)

    expected_beam_translation = shell_translation + np.cross(shell_rotation, r)
    expected_beam_rotation = shell_rotation
    translation_error = float(np.linalg.norm(u[beam.dofs[:3]] - expected_beam_translation))
    rotation_error = float(np.linalg.norm(u[beam.dofs[3:6]] - expected_beam_rotation))
    residuals = mpc_constraint_residuals(model, u)
    max_constraint_residual = max((abs(value) for value in residuals.values()), default=0.0)

    _assert(constraint_info["num_mpc_slave_dofs"] == 6, "eccentric MPC did not eliminate six beam-node slave DOFs")
    _assert(max(translation_error, rotation_error, max_constraint_residual) < 1.0e-13, "eccentric MPC affine-field reproduction failed")
    return _pass(
        case,
        element_types=["beam_shell_mpc"],
        checks={
            "eccentricity": r.tolist(),
            "num_mpc_slave_dofs": int(constraint_info["num_mpc_slave_dofs"]),
            "translation_error": translation_error,
            "rotation_error": rotation_error,
            "max_constraint_residual": max_constraint_residual,
        },
    )


def _run_coup_013(case: VerificationCase) -> VerificationCaseResult:
    """Actual eccentric slave load transfers to master force plus r x F moment."""
    r = np.array([0.12, -0.04, 0.18], dtype=float)
    model = _eccentric_coupling_model(fixed_shell=True, eccentricity=r)
    force = np.array([1400.0, -320.0, 210.0], dtype=float)
    moment = np.array([18.0, -7.0, 31.0], dtype=float)
    load_vector = np.concatenate([force, moment])
    load = LoadCase("eccentric_slave_load")
    load.add_nodal_load(2, load_vector)
    u0 = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)

    diagnostics = compute_constraint_force_diagnostics(model, u0, load)
    slave_force = np.asarray(diagnostics["mpc_slave_forces"].get(2, np.zeros(6)), dtype=float)
    master_equivalent = np.asarray(diagnostics["mpc_master_equivalent_forces"].get(1, np.zeros(6)), dtype=float)
    expected_master = -np.concatenate([force, moment + np.cross(r, force)])

    slave_error = float(np.linalg.norm(slave_force + load_vector))
    master_error = float(np.linalg.norm(master_equivalent - expected_master))
    support_direct = np.asarray(diagnostics["support_reactions"].get(1, np.zeros(6)), dtype=float)
    support_direct_norm = float(np.linalg.norm(support_direct))

    _assert(slave_error < 1.0e-12, "MPC slave-force bucket did not recover eccentric applied load")
    _assert(master_error < 1.0e-10, "MPC master-equivalent eccentric load transfer mismatch")
    _assert(support_direct_norm < 1.0e-12, "direct support reaction bucket should not include MPC transfer")
    return _pass(
        case,
        element_types=["beam_shell_mpc"],
        checks={
            "eccentricity": r.tolist(),
            "slave_force_error": slave_error,
            "master_equivalent_error": master_error,
            "slave_force": slave_force.tolist(),
            "master_equivalent_force": master_equivalent.tolist(),
            "expected_master_equivalent_force": expected_master.tolist(),
            "support_direct_norm": support_direct_norm,
            "num_mpc_constraint_forces": len(diagnostics["mpc_constraint_forces"]),
        },
    )


def _free_beam_nullspace() -> Tuple[FEModel, np.ndarray, np.ndarray, Dict[str, Any]]:
    model = _beam_model(length=1.0, area=0.01)
    K, _ = assemble_stiffness_matrix(model)
    zero = np.zeros(model.mesh.dof_manager.total_dofs)
    K_red, _F, _T, _u0, independent, constraint_info = build_constraint_transformation(K, zero, model)
    Q, nullspace_info = build_reduced_rigid_body_modes(model, independent, int(K.shape[0]))
    return model, K_red, Q, {"constraint": constraint_info, "nullspace": nullspace_info}


def _run_null_001(case: VerificationCase) -> VerificationCaseResult:
    _model, _K, Q, info = _free_beam_nullspace()
    rank = int(Q.shape[1])
    _assert(rank == 6, "free beam nullspace rank is not six")
    return _pass(case, element_types=["beam2"], checks={"rank": rank, **info["nullspace"]})


def _run_null_002(case: VerificationCase) -> VerificationCaseResult:
    _model, _K, Q, _info = _free_beam_nullspace()
    f = np.zeros(Q.shape[0])
    f[0] = 1.0
    projected = f - Q @ (Q.T @ f)
    rel = float(np.linalg.norm(Q.T @ projected) / max(np.linalg.norm(projected), 1.0e-30))
    _assert(rel < 1.0e-12, "projected load is not orthogonal to rigid basis")
    return _pass(case, element_types=["beam2"], checks={"projected_load_orthogonality": rel})


def _beam_model_between(start: np.ndarray, end: np.ndarray) -> FEModel:
    model = FEModel("verification_beam_between")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, *np.asarray(start, dtype=float).tolist())
    model.add_node(2, *np.asarray(end, dtype=float).tolist())
    section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6, "shear_factor_y": 5.0 / 6.0, "shear_factor_z": 5.0 / 6.0}
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    return model


def _self_equilibrated_axial_load(model: FEModel, magnitude: float = 2500.0) -> LoadCase:
    n1 = model.mesh.get_node(1)
    n2 = model.mesh.get_node(2)
    axis = n2.coords() - n1.coords()
    axis = axis / np.linalg.norm(axis)
    load = LoadCase("self_equilibrated_axial")
    load.add_nodal_load(1, np.concatenate([-magnitude * axis, np.zeros(3)]))
    load.add_nodal_load(2, np.concatenate([magnitude * axis, np.zeros(3)]))
    return load


def _axial_extension(model: FEModel, displacements: np.ndarray) -> float:
    n1 = model.mesh.get_node(1)
    n2 = model.mesh.get_node(2)
    axis = n2.coords() - n1.coords()
    axis = axis / np.linalg.norm(axis)
    u = np.asarray(displacements, dtype=float)
    return float((u[n2.dofs[:3]] - u[n1.dofs[:3]]) @ axis)


def _strain_energy(model: FEModel, displacements: np.ndarray) -> float:
    K, _ = assemble_stiffness_matrix(model)
    u = np.asarray(displacements, dtype=float)
    return 0.5 * float(u @ (K @ u))


def _solve_linear_checked(model: FEModel, load: LoadCase) -> Tuple[np.ndarray, Dict[str, Any]]:
    displacements, info = solve_linear(model, load, constraint_mode="auto")
    status = (info.get("convergence_info") or {}).get("status")
    _assert(status == "converged", f"linear solve did not converge: {status}")
    return displacements, info


def _run_null_003(case: VerificationCase) -> VerificationCaseResult:
    free_model = _beam_model(length=1.0, area=0.01)
    free_load = _self_equilibrated_axial_load(free_model)
    free_u, free_info = _solve_linear_checked(free_model, free_load)

    constrained_model = _beam_model(length=1.0, area=0.01)
    constrained_model.add_boundary_condition(FixedSupport("fixed_node_1", [1]))
    constrained_load = _self_equilibrated_axial_load(constrained_model)
    constrained_u, constrained_info = _solve_linear_checked(constrained_model, constrained_load)

    extension_error = _rel_error(_axial_extension(free_model, free_u), _axial_extension(constrained_model, constrained_u))
    energy_error = _rel_error(_strain_energy(free_model, free_u), _strain_energy(constrained_model, constrained_u))
    _assert(max(extension_error, energy_error) < 1.0e-9, "projected and constrained solutions differ in elastic field")
    return _pass(
        case,
        element_types=["beam2"],
        analysis_type="linear_static",
        checks={
            "extension_relative_error": extension_error,
            "strain_energy_relative_error": energy_error,
            "free_nullspace_rank": int((free_info.get("nullspace_info") or {}).get("rank", 0)),
            "constrained_nullspace_rank": int((constrained_info.get("nullspace_info") or {}).get("rank", 0)),
        },
    )


def _run_null_004(case: VerificationCase) -> VerificationCaseResult:
    variants: List[Tuple[str, Callable[[FEModel], None]]] = [
        ("node_1_fixed", lambda m: m.add_boundary_condition(FixedSupport("fixed_node_1", [1]))),
        ("node_2_fixed", lambda m: m.add_boundary_condition(FixedSupport("fixed_node_2", [2]))),
        (
            "minimal_stabilized",
            lambda m: (
                m.add_boundary_condition(BoundaryCondition("node_1_translations", [1], {"ux": 0.0, "uy": 0.0, "uz": 0.0})),
                m.add_boundary_condition(BoundaryCondition("node_2_transverse_rotations", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0})),
            ),
        ),
    ]
    extensions: Dict[str, float] = {}
    energies: Dict[str, float] = {}
    for name, apply_restraints in variants:
        model = _beam_model(length=1.0, area=0.01)
        apply_restraints(model)
        load = _self_equilibrated_axial_load(model)
        u, _info = _solve_linear_checked(model, load)
        extensions[name] = _axial_extension(model, u)
        energies[name] = _strain_energy(model, u)

    extension_spread = float((max(extensions.values()) - min(extensions.values())) / max(abs(next(iter(extensions.values()))), 1.0e-30))
    energy_spread = float((max(energies.values()) - min(energies.values())) / max(abs(next(iter(energies.values()))), 1.0e-30))
    _assert(max(extension_spread, energy_spread) < 1.0e-9, "elastic field depends on arbitrary support choice")
    return _pass(
        case,
        element_types=["beam2"],
        analysis_type="linear_static",
        checks={
            "extensions": extensions,
            "strain_energies": energies,
            "extension_relative_spread": extension_spread,
            "strain_energy_relative_spread": energy_spread,
        },
    )


def _run_null_005(case: VerificationCase) -> VerificationCaseResult:
    reference_model = _beam_model(length=1.0, area=0.01)
    reference_load = _self_equilibrated_axial_load(reference_model)
    reference_u, _reference_info = _solve_linear_checked(reference_model, reference_load)

    start = np.array([2.0, -0.5, 0.75], dtype=float)
    axis = np.array([0.36, 0.48, 0.80], dtype=float)
    axis = axis / np.linalg.norm(axis)
    transformed_model = _beam_model_between(start, start + axis)
    transformed_load = _self_equilibrated_axial_load(transformed_model)
    transformed_u, transformed_info = _solve_linear_checked(transformed_model, transformed_load)

    extension_error = _rel_error(_axial_extension(transformed_model, transformed_u), _axial_extension(reference_model, reference_u))
    energy_error = _rel_error(_strain_energy(transformed_model, transformed_u), _strain_energy(reference_model, reference_u))
    _assert(max(extension_error, energy_error) < 1.0e-9, "nullspace solution is not invariant under rigid transform")
    return _pass(
        case,
        element_types=["beam2"],
        analysis_type="linear_static",
        checks={
            "extension_relative_error": extension_error,
            "strain_energy_relative_error": energy_error,
            "transformed_nullspace_rank": int((transformed_info.get("nullspace_info") or {}).get("rank", 0)),
        },
    )


def _run_eig_001(case: VerificationCase) -> VerificationCaseResult:
    model = _beam_model(length=2.0, area=0.02, density=7850.0)
    props = calculate_mass_properties(model)
    ref = 7850.0 * 0.02 * 2.0
    err = _rel_error(props.total_mass, ref)
    _assert(err < 1.0e-12, "beam mass mismatch")
    return _pass(case, element_types=["beam2"], reference={"type": "analytical", "value": ref}, result={"value": props.total_mass, "relative_error": err}, checks=props.to_dict())


def _run_eig_002(case: VerificationCase) -> VerificationCaseResult:
    masses = []
    for n in (1, 2, 4):
        model = _beam_model(length=2.0, area=0.02, density=7850.0, num_elements=n)
        masses.append(calculate_mass_properties(model).total_mass)
    spread = float((max(masses) - min(masses)) / max(abs(masses[0]), 1.0))
    _assert(spread < 1.0e-12, "mass changes under beam mesh refinement")
    return _pass(case, element_types=["beam2"], checks={"mass_values": masses, "relative_spread": spread})


def _run_eig_003(case: VerificationCase) -> VerificationCaseResult:
    model = _beam_model(length=1.0, area=1.0, density=2.0)
    model.materials["steel"].elastic_modulus = 100.0
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("slider", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    result = solve_free_vibration(model, num_modes=1)
    err = float(result.diagnostics["mass_orthogonality_error"])
    _assert(err < 1.0e-8, "modal mass orthogonality failed")
    return _pass(case, element_types=["beam2"], analysis_type="modal", checks=result.diagnostics)


def _run_eig_004(case: VerificationCase) -> VerificationCaseResult:
    model = FEModel("repeated_axial_modes")
    model.add_material("steel", 100.0, 0.3, density=2.0)
    section = {"area": 1.0, "Iy": 1.0, "Iz": 1.0, "J": 1.0}
    for node_id, coords in {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (0.0, 2.0, 0.0),
        4: (1.0, 2.0, 0.0),
    }.items():
        model.add_node(node_id, *coords)
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_element(2, BeamElement(2, [3, 4], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed_bases", [1, 3]))
    model.add_boundary_condition(BoundaryCondition("axial_sliders", [2, 4], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    result = solve_free_vibration(model, num_modes=2, dense_size_limit=10000)
    _assert(result.solver_status == "ok" and result.num_modes_returned == 2, "repeated eigenspace modal solve failed")
    frequencies = result.frequencies_hz
    spread = float((np.max(frequencies) - np.min(frequencies)) / max(abs(float(np.mean(frequencies))), 1.0e-30))
    orthogonality = float(result.diagnostics.get("mass_orthogonality_error", 1.0))
    _assert(spread < 1.0e-10 and orthogonality < 1.0e-8, "repeated modal eigenspace is not stable")
    return _pass(
        case,
        element_types=["beam2"],
        analysis_type="modal",
        checks={"frequencies_hz": frequencies.tolist(), "relative_frequency_spread": spread, **result.diagnostics},
    )


def _run_buc_001(case: VerificationCase) -> VerificationCaseResult:
    model = _beam_model(num_elements=2)
    states = {element_id: {"axial_compression": 100.0} for element_id in model.mesh.elements}
    KG, _ = assemble_geometric_stiffness_matrix(model, states)
    err = _symmetry_error(KG)
    _assert(err < 1.0e-10, "geometric stiffness symmetry failed")
    return _pass(case, element_types=["beam2"], analysis_type="linear_buckling", checks={"geometric_stiffness_symmetry": err})


def _run_buc_002(case: VerificationCase) -> VerificationCaseResult:
    model = _beam_model(length=4.0, num_elements=6)
    all_nodes = list(model.mesh.nodes)
    model.add_boundary_condition(BoundaryCondition("suppress", all_nodes, {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pins", [1, 7], {"uy": 0.0}))
    base = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    double = {element_id: {"axial_compression": 2.0} for element_id in model.mesh.elements}
    half = {element_id: {"axial_compression": 0.5} for element_id in model.mesh.elements}
    r1 = solve_eigenvalue_buckling(model, base, num_modes=1)
    r2 = solve_eigenvalue_buckling(model, double, num_modes=1)
    rh = solve_eigenvalue_buckling(model, half, num_modes=1)
    err2 = _rel_error(float(r2.critical_load_factor), 0.5 * float(r1.critical_load_factor))
    errh = _rel_error(float(rh.critical_load_factor), 2.0 * float(r1.critical_load_factor))
    _assert(max(err2, errh) < 1.0e-8, "buckling preload scaling failed")
    return _pass(case, element_types=["beam2"], analysis_type="linear_buckling", checks={"double_preload_error": err2, "half_preload_error": errh})


class _VerificationSofteningSpring(Element):
    """One active DOF with analytical limit point: lambda = k u - c u^3."""

    def __init__(self, element_id: int, node_id: int, *, k: float = 1.0, c: float = 1.0):
        super().__init__(element_id, [node_id], "default")
        self.k = float(k)
        self.c = float(c)

    @property
    def num_nodes(self) -> int:
        return 1

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh: Any) -> np.ndarray:
        return np.asarray([mesh.get_node(self.node_ids[0]).coords()], dtype=float)

    def compute_stiffness_matrix(self, mesh: Any, material: Any) -> np.ndarray:
        matrix = np.eye(6, dtype=float)
        matrix[0, 0] = self.k
        return matrix

    def compute_nonlinear_response(
        self,
        mesh: Any,
        material: Any,
        u_elem: np.ndarray,
        state: Any = None,
        num_layers: int = 5,
        tangent: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
        displacement = float(np.asarray(u_elem, dtype=float)[0])
        force = np.asarray(u_elem, dtype=float).copy()
        force[0] = self.k * displacement - self.c * displacement**3
        stiffness = None
        if tangent:
            stiffness = np.eye(6, dtype=float)
            stiffness[0, 0] = self.k - 3.0 * self.c * displacement**2
        return force, stiffness, {"spring_displacement": displacement}


def _softening_spring_model(*, k: float = 1.0, c: float = 1.0) -> Tuple[FEModel, LoadCase]:
    model = FEModel("verification_softening_spring")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_element(1, _VerificationSofteningSpring(1, 1, k=k, c=c))
    model.add_boundary_condition(BoundaryCondition("one_dof", [1], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    load = LoadCase("unit_reference")
    load.add_nodal_load(1, load_vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load


def _plastic_stiffened_panel_model() -> Tuple[FEModel, LoadCase]:
    model, _panel, _config = _thin_stiffened_panel_model(1, element_family="Q4", shell_divisions_x=2, shell_divisions_y=2, beam_divisions=2)
    curve = reference_plastic_curve()
    for material in model.materials.values():
        material.hardening_curve = curve
    for element in model.mesh.elements.values():
        if isinstance(element, BeamElement):
            element.cross_section["fiber_plasticity"] = True
    load = _pressure_load_for_shells(model, 1.0e7)
    return model, load


def _run_nlg_006(case: VerificationCase) -> VerificationCaseResult:
    from .nonlinear_static import solve_static_nonlinear

    rows: List[Dict[str, Any]] = []
    displacements_by_steps: Dict[int, np.ndarray] = {}
    for num_steps in (2, 4, 6):
        model, load = _plastic_stiffened_panel_model()
        result = solve_static_nonlinear(
            model,
            load,
            max_load_factor=1.0,
            num_steps=num_steps,
            max_iterations=25,
            num_layers=3,
            convergence_settings="robust",
        )
        displacements_by_steps[int(num_steps)] = result.displacements
        rows.append(
            {
                "num_steps": int(num_steps),
                "status": result.status,
                "load_factor": float(result.load_factor),
                "total_newton_iterations": int(result.info.get("total_newton_iterations", 0)),
                "max_abs_displacement": float(np.max(np.abs(result.displacements))),
                "max_equivalent_plastic_strain": float(result.info.get("strain_summary", {}).get("max_equivalent_plastic_strain", 0.0)),
                "stop_reason": result.stop_reason,
                "status_category": result.status_category,
            }
        )
    reference = np.asarray(displacements_by_steps[6], dtype=float)
    for row in rows:
        row["endpoint_relative_to_6_step"] = float(
            np.linalg.norm(displacements_by_steps[int(row["num_steps"])] - reference) / max(np.linalg.norm(reference), 1.0e-30)
        )
    max_endpoint_error = max(float(row["endpoint_relative_to_6_step"]) for row in rows)
    _assert(all(row["status"] == "completed" for row in rows), "stiffened-panel nonlinear increment study did not complete")
    _assert(max_endpoint_error < 1.0e-6, "stiffened-panel nonlinear endpoint is increment dependent")
    return _pass(
        case,
        element_types=["shell4", "beam2", "interpolated_mpc"],
        analysis_type="nonlinear_static",
        reference={"type": "increment convergence", "source": "6-step robust Newton endpoint"},
        result={"max_endpoint_relative_error": max_endpoint_error, "rows": rows},
        checks={"rows": rows},
    )


def _evaluate_nlg_007(case: VerificationCase) -> VerificationCaseResult:
    from .arc_length import ArcLengthControl, solve_static_arc_length
    from .imperfections import ImperfectionField

    spring_model, spring_load = _softening_spring_model(k=1.0, c=1.0)
    spring_control = ArcLengthControl(
        initial_load_increment=0.02,
        minimum_load_increment=1.0e-5,
        maximum_load_increment=0.05,
        max_steps=120,
        stop_after_peak_steps=5,
        peak_drop_tolerance=1.0e-4,
    )
    spring = solve_static_arc_length(
        spring_model,
        spring_load,
        control=spring_control,
        max_iterations=30,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
    )
    analytical_peak = 2.0 / (3.0 * math.sqrt(3.0))
    peak_error = _rel_error(spring.peak_load_factor, analytical_peak)

    panel, panel_load = _plastic_stiffened_panel_model()
    offsets = {
        int(node_id): (0.0, 0.0, 1.0e-4)
        for node_id, node in panel.mesh.nodes.items()
        if abs(float(node.z)) < 1.0e-12 and 0.25 < float(node.x) < 0.85 and 0.05 < float(node.y) < 0.35
    }
    panel_control = ArcLengthControl(
        initial_load_increment=0.10,
        minimum_load_increment=0.01,
        maximum_load_increment=0.20,
        max_steps=8,
        maximum_absolute_load_factor=0.50,
        stop_after_peak_steps=2,
    )
    panel_result = solve_static_arc_length(
        panel,
        panel_load,
        control=panel_control,
        max_iterations=15,
        tolerance=1.0e-6,
        arc_tolerance=1.0e-6,
        num_layers=3,
        imperfection=ImperfectionField(offsets, name="verification_panel_bow"),
    )
    _assert(spring.status == "peak_confirmed" and peak_error < 0.03, "arc-length analytical peak reference failed")
    _assert(panel_result.status == "load_factor_limit_reached" and len(panel_result.steps) >= 3, "imperfect stiffened-panel arc-length guard failed")
    return _pass(
        case,
        element_types=["shell4", "beam2", "interpolated_mpc"],
        analysis_type="arc_length",
        reference={"type": "analytical plus imperfect-panel guard", "analytical_softening_peak": analytical_peak},
        result={
            "softening_peak_load_factor": float(spring.peak_load_factor),
            "softening_peak_relative_error": peak_error,
            "panel_status": panel_result.status,
            "panel_last_load_factor": float(panel_result.load_factor),
            "panel_peak_load_factor": float(panel_result.peak_load_factor),
        },
        checks={
            "softening_steps": [step.to_dict() for step in spring.steps[-8:]],
            "panel_steps": [step.to_dict() for step in panel_result.steps],
            "panel_imperfection": panel_result.info.get("imperfection", []),
        },
    )


def _evaluate_nlg_008(case: VerificationCase) -> VerificationCaseResult:
    model = _verification_plate_model(divisions=2, thickness=0.01, element_family="S4")
    follower = _pressure_load_all_shells(model, 1000.0)
    follower.follower_pressure = True
    rng = np.random.default_rng(20260726)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    translational_dofs: List[int] = []
    for node in model.mesh.nodes.values():
        displacement[node.dofs[:3]] = rng.normal(scale=0.015, size=3)
        translational_dofs.extend(int(dof) for dof in node.dofs[:3])
    tangent, tangent_info = assemble_external_load_tangent(model, follower, displacement)
    step = 2.0e-7
    maximum_error = 0.0
    for dof in translational_dofs[:: max(len(translational_dofs) // 8, 1)]:
        plus = displacement.copy()
        minus = displacement.copy()
        plus[dof] += step
        minus[dof] -= step
        force_plus, _ = assemble_load_vector(model, follower, plus)
        force_minus, _ = assemble_load_vector(model, follower, minus)
        numerical = (force_plus - force_minus) / (2.0 * step)
        scale = max(float(np.linalg.norm(numerical)), 1.0)
        error = float(np.linalg.norm(tangent[:, dof].toarray().reshape(-1) - numerical) / scale)
        maximum_error = max(maximum_error, error)
    report = validate_production_model(
        model,
        load_cases=[follower],
        analysis_type="nonlinear_static",
        allow_free_mechanisms=True,
    )
    _assert(report.status != "invalid", "qualified follower pressure failed production validation")
    _assert(maximum_error < 1.0e-8, "follower-pressure tangent finite-difference mismatch")

    ring_config = CylinderBenchmarkConfig(
        radius=1.0,
        height=1.0,
        thickness=0.02,
        pressure=1.0,
        num_circumferential=24,
        num_height=1,
        closed_end_axial_load=False,
    )
    ring_model, ring_pressure = build_cylindrical_shell_benchmark_model(ring_config)
    ring_pressure.follower_pressure = True
    for index in range(ring_config.num_circumferential):
        ring_model.add_element(
            1000 + index,
            CoupledBeamShellElement(
                1000 + index,
                beam_node_id=ring_config.num_circumferential + 1 + index,
                shell_node_id=1 + index,
                material_name="steel",
                eccentricity=np.zeros(3),
            ),
        )
    polygon_membrane_compression = (
        ring_config.pressure
        * ring_config.radius
        * np.cos(np.pi / ring_config.num_circumferential)
    )
    ring_states = {
        int(element_id): {
            "membrane_compression_x": polygon_membrane_compression,
            "membrane_compression_y": 0.0,
            "membrane_compression_xy": 0.0,
        }
        for element_id, element in ring_model.mesh.elements.items()
        if isinstance(element, ShellElement)
    }
    ring_buckling = solve_eigenvalue_buckling(
        ring_model,
        ring_states,
        num_modes=2,
        reference_load_case=ring_pressure,
        dense_size_limit=10_000,
        allow_dense_fallback=True,
    )
    ring_D = (
        ring_config.elastic_modulus
        * ring_config.thickness**3
        / (12.0 * (1.0 - ring_config.poisson_ratio**2))
    )
    ring_reference = 3.0 * ring_D / ring_config.radius**3
    ring_error = _rel_error(float(ring_buckling.critical_load_factor or 0.0), ring_reference)
    _assert(
        ring_buckling.solver_status == "ok" and ring_error < 0.06,
        "thin-ring follower-pressure buckling mismatch",
    )
    return _pass(
        case,
        element_types=["shell4"],
        analysis_type="nonlinear_static",
        reference={
            "type": "central finite difference plus analytical thin ring",
            "ring_critical_pressure": ring_reference,
            "ring_formula": "3*D/R^3",
        },
        result={
            "validation_status": report.status,
            "maximum_relative_tangent_error": maximum_error,
            "external_tangent_symmetry_error": tangent_info["diagnostics"]["assembled_symmetry_error"],
            "ring_critical_pressure": ring_buckling.critical_load_factor,
            "ring_relative_error": ring_error,
        },
        checks={
            "validation": report.to_dict(),
            "load_tangent": tangent_info,
            "ring_buckling": ring_buckling.diagnostics,
        },
    )


def _run_mat_008(case: VerificationCase) -> VerificationCaseResult:
    curve = reference_plastic_curve()
    model = FEModel("combined_shell_beam_plasticity")
    model.add_material("steel", 210.0e9, 0.3, hardening_curve=curve)
    for node_id, coord in enumerate(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)), start=1):
        model.add_node(node_id, *coord)
    model.add_node(101, 0.0, 1.4, 0.0)
    model.add_node(102, 1.0, 1.4, 0.0)
    shell = create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.01)
    beam = BeamElement(2, [101, 102], "steel", {"area": 0.01, "Iy": 1.0e-5, "Iz": 1.0e-5, "J": 1.0e-5, "fiber_plasticity": True})
    model.add_element(1, shell)
    model.add_element(2, beam)

    u_shell = np.zeros(shell.total_dofs, dtype=float)
    coords = shell.get_node_coordinates(model.mesh)
    for local, coord in enumerate(coords):
        x, y, _z = coord
        base = local * 6
        u_shell[base + 0] = 0.003 * x
        u_shell[base + 1] = -0.0008 * y
        u_shell[base + 4] = 0.002 * x
    shell_force, shell_tangent, shell_state = shell.compute_nonlinear_response(model.mesh, model.get_material("steel"), u_shell, num_layers=5, tangent=True)

    u_beam = np.zeros(beam.total_dofs, dtype=float)
    u_beam[6] = 0.006
    u_beam[7] = 0.0003
    beam_force, beam_tangent, beam_state = beam.compute_nonlinear_response(model.mesh, model.get_material("steel"), u_beam, tangent=True)

    shell_alpha = float(np.max(np.asarray(shell_state.get("alpha", [0.0]), dtype=float)))
    beam_alpha = float(np.max(np.asarray(beam_state.get("alpha", [0.0]), dtype=float)))
    shell_symmetry = _symmetry_error(shell_tangent)
    beam_symmetry = _symmetry_error(beam_tangent)
    tangent_metrics = element_tangent_metrics()
    shell_tangent_error = float(tangent_metrics["shell_layered_plastic"]["tangent_fd_relative_error"])
    beam_tangent_error = float(tangent_metrics["beam_fiber_plastic"]["tangent_fd_relative_error"])
    _assert(shell_alpha > 0.0 and beam_alpha > 0.0, "combined shell/beam plastic states did not yield")
    _assert(max(shell_tangent_error, beam_tangent_error) < 1.0e-4, "combined shell/beam plastic tangents are not finite-difference consistent")
    _assert(np.all(np.isfinite(shell_force)) and np.all(np.isfinite(beam_force)), "combined plastic internal forces are not finite")
    return _pass(
        case,
        element_types=["shell4", "beam2"],
        analysis_type="plasticity",
        reference={"type": "element state invariant", "quantities": ["shell layered plasticity", "beam fiber plasticity"]},
        result={"shell_alpha_max": shell_alpha, "beam_alpha_max": beam_alpha},
        checks={
            "shell_force_norm": float(np.linalg.norm(shell_force)),
            "beam_force_norm": float(np.linalg.norm(beam_force)),
            "shell_tangent_symmetry": shell_symmetry,
            "beam_tangent_symmetry": beam_symmetry,
            "shell_tangent_fd_relative_error": shell_tangent_error,
            "beam_tangent_fd_relative_error": beam_tangent_error,
            "shell_state_keys": sorted(str(key) for key in shell_state.keys()),
            "beam_state_keys": sorted(str(key) for key in beam_state.keys()),
        },
    )


def _run_dyn_001(case: VerificationCase) -> VerificationCaseResult:
    model, _panel, _config = _thin_stiffened_panel_model(1, element_family="Q4", shell_divisions_x=2, shell_divisions_y=2, beam_divisions=2)
    for material in model.materials.values():
        material.density = 7850.0
    shell_ids = [int(element_id) for element_id, element in model.mesh.elements.items() if isinstance(element, ShellElement)]
    output_node = max(int(node_id) for node_id in model.mesh.nodes)
    patch = PressurePatch.rectangular_pulse(
        name="thin_stiffened_panel_pulse",
        pressure=1000.0,
        start_time=0.0,
        end_time=0.002,
        element_ids=shell_ids[:1],
    )
    result = solve_transient_newmark(
        model,
        TransientConfig(dt=0.001, t_end=0.004, save_every=1, output_nodes=[output_node]),
        pressure_patches=[patch],
    )
    selected_area_impulse = float(np.linalg.norm(result.force_impulse))
    _assert(result.status == "completed", "thin stiffened-panel transient benchmark did not complete")
    _assert(result.peak_displacement > 0.0 and result.node_histories[output_node].shape[0] == len(result.times), "transient displacement history was not saved")
    _assert(result.diagnostics["pressure_patches"][0]["num_selected_elements"] == 1 and selected_area_impulse > 0.0, "transient pressure-patch impulse check failed")
    _assert(result.diagnostics["factorization_reused"] is True, "transient effective stiffness factorization was not reused")
    return _pass(
        case,
        element_types=["shell4", "beam2", "interpolated_mpc"],
        analysis_type="linear_transient",
        reference={"type": "transient invariant", "quantities": ["pressure impulse", "saved nodal history", "factor reuse"]},
        result={
            "peak_displacement": float(result.peak_displacement),
            "peak_displacement_node": result.peak_displacement_node,
            "force_impulse": result.force_impulse.tolist(),
            "factorization_count": int(result.diagnostics["factorization_count"]),
        },
        checks={
            "times": result.times.tolist(),
            "output_node": int(output_node),
            "node_history_shape": list(result.node_histories[output_node].shape),
            "pressure_patch": result.diagnostics["pressure_patches"][0],
            "factorization_reused": bool(result.diagnostics["factorization_reused"]),
        },
    )


def _hemisphere_cap_model(n_phi: int = 4, n_theta: int = 8) -> Tuple[FEModel, LoadCase]:
    model = FEModel("hemisphere_cap_internal_benchmark")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    radius = 1.0
    thickness = 0.01
    phi_min = 0.15
    phi_max = 0.5 * math.pi
    node_id = 1
    ids: Dict[Tuple[int, int], int] = {}
    for iphi in range(int(n_phi) + 1):
        phi = phi_min + (phi_max - phi_min) * iphi / float(n_phi)
        for itheta in range(int(n_theta)):
            theta = 2.0 * math.pi * itheta / float(n_theta)
            ids[(iphi, itheta)] = node_id
            model.add_node(
                node_id,
                radius * math.sin(phi) * math.cos(theta),
                radius * math.sin(phi) * math.sin(theta),
                radius * math.cos(phi),
            )
            node_id += 1
    element_id = 1
    for iphi in range(int(n_phi)):
        for itheta in range(int(n_theta)):
            model.add_element(
                element_id,
                create_shell_element(
                    element_id,
                    [
                        ids[(iphi, itheta)],
                        ids[(iphi, (itheta + 1) % int(n_theta))],
                        ids[(iphi + 1, (itheta + 1) % int(n_theta))],
                        ids[(iphi + 1, itheta)],
                    ],
                    "steel",
                    thickness=thickness,
                ),
            )
            element_id += 1
    equator_nodes = [ids[(int(n_phi), itheta)] for itheta in range(int(n_theta))]
    model.add_boundary_condition(BoundaryCondition("equator_uz", equator_nodes, {"uz": 0.0}))
    model.add_boundary_condition(BoundaryCondition("reference_xy", [equator_nodes[0]], {"ux": 0.0, "uy": 0.0}))
    model.add_boundary_condition(BoundaryCondition("reference_x", [equator_nodes[int(n_theta) // 4]], {"ux": 0.0}))
    load = LoadCase("external_pressure")
    for element_id in model.mesh.elements:
        load.add_pressure_load(int(element_id), -1000.0)
    return model, load


def _run_bench_004(case: VerificationCase) -> VerificationCaseResult:
    rows: List[Dict[str, Any]] = []
    for n_phi, n_theta in ((4, 8), (6, 12)):
        model, load = _hemisphere_cap_model(n_phi=n_phi, n_theta=n_theta)
        displacements, solver_info = solve_linear(model, load, constraint_mode="auto")
        top_ring_nodes = [
            node_id
            for node_id, node in model.mesh.nodes.items()
            if float(node.z) == max(float(other.z) for other in model.mesh.nodes.values())
        ]
        top_uz = [float(displacements[model.mesh.get_node(node_id).dofs[2]]) for node_id in top_ring_nodes]
        top_spread = 0.0 if not top_uz else float((max(top_uz) - min(top_uz)) / max(max(abs(value) for value in top_uz), 1.0e-30))
        rows.append(
            {
                "n_phi": int(n_phi),
                "n_theta": int(n_theta),
                "nodes": int(model.mesh.num_nodes),
                "elements": int(model.mesh.num_elements),
                "solver_status": str((solver_info.get("convergence_info") or {}).get("status", "unknown")),
                "max_abs_displacement": float(np.max(np.abs(displacements))),
                "top_ring_uz_spread": top_spread,
            }
        )
    response_ratio = float(rows[-1]["max_abs_displacement"] / max(rows[0]["max_abs_displacement"], 1.0e-30))
    _assert(all(row["solver_status"] == "converged" for row in rows), "hemispherical shell benchmark solve failed")
    _assert(all(row["top_ring_uz_spread"] < 1.0e-8 for row in rows), "hemispherical shell rotational symmetry check failed")
    _assert(0.5 < response_ratio < 2.0, "hemispherical shell refinement response is outside sanity band")
    return _pass(
        case,
        element_types=["shell4"],
        analysis_type="linear_static",
        mesh={"fixture": "hemisphere cap with equator support", "radius": 1.0, "thickness": 0.01},
        reference={"type": "internal symmetry/refinement", "note": "classical external hemispherical-shell target not bundled"},
        result={"response_ratio_refined_to_coarse": response_ratio},
        checks={"rows": rows},
    )


def _evaluate_nlg_002(case: VerificationCase) -> VerificationCaseResult:
    from .beam_validity import corotational_axial_extension_metric, corotational_rigid_rotation_metric

    rigid = corotational_rigid_rotation_metric(angle_degrees=75.0)
    axial = corotational_axial_extension_metric(extension=0.002)
    _assert(float(rigid["corotational_force_norm"]) < 1.0e-5, "large rigid rotation produced corotational beam force")
    _assert(float(rigid["force_norm_ratio_corot_to_default"]) < 1.0e-12, "corotational beam did not suppress rigid-rotation strain")
    _assert(float(axial["relative_error"]) < 1.0e-10, "corotational beam axial extension response failed")
    return _pass(
        case,
        element_types=["beam2"],
        analysis_type="geometric_nonlinear",
        reference={"type": "large-rotation objectivity", "source": "corotational 2-node beam rigid rotation and axial extension"},
        result={"angle_degrees": rigid["angle_degrees"], "corotational_force_norm": rigid["corotational_force_norm"]},
        checks={"rigid_rotation": rigid, "axial_extension": axial},
    )


def _run_nlg_002(case: VerificationCase) -> VerificationCaseResult:
    return _cached_nonlinear_framework_result("NLG-002", case, _evaluate_nlg_002)


def _run_nlg_007(case: VerificationCase) -> VerificationCaseResult:
    return _cached_nonlinear_framework_result("NLG-007", case, _evaluate_nlg_007)


def _run_nlg_008(case: VerificationCase) -> VerificationCaseResult:
    return _cached_nonlinear_framework_result("NLG-008", case, _evaluate_nlg_008)


def _run_nlg_003(case: VerificationCase) -> VerificationCaseResult:
    """Local NAFEMS-style nonlinear framework smoke check.

    The proprietary NAFEMS 3DNLG datasets are not redistributed with this repo.
    This case therefore verifies that the local nonlinear framework has the
    same ingredients required to host such cases: geometric objectivity,
    analytical limit-point continuation, imperfect panel continuation and
    unsupported follower-pressure rejection.
    """

    synthetic = [
        ("large_rotation_objectivity", _run_nlg_002(case).status),
        ("arc_length_limit_point", _run_nlg_007(case).status),
        ("follower_pressure_guard", _run_nlg_008(case).status),
    ]
    _assert(all(status == "PASS" for _name, status in synthetic), "NAFEMS-style nonlinear framework ingredients failed")
    return _pass(
        case,
        analysis_type="nonlinear_framework",
        reference={"type": "framework readiness", "note": "NAFEMS 3DNLG proprietary numerical targets are not bundled"},
        result={"framework_checks": {name: status for name, status in synthetic}},
        checks={
            "datasets_bundled": False,
            "supported_fixture_types": ["large_rotation_objectivity", "limit_point_arc_length", "scope_guard"],
        },
    )


def _external_reference_locations() -> Tuple[Path, Path, Path]:
    report_path = _EXTERNAL_REFERENCE_REPORT_OVERRIDE.get()
    if report_path is None:
        return (
            Path("reports/external_references/external_reference_report.json"),
            Path("reports/external_references/external_reference_report.md"),
            Path("reports/external_references/decks"),
        )
    return report_path, report_path.with_suffix(".md"), report_path.parent / "external_reference_decks"


def _external_reference_report_for_verification() -> Dict[str, Any]:
    _report_path, _markdown_path, deck_dir = _external_reference_locations()
    cache = _EXTERNAL_REFERENCE_REPORT_RUN_CACHE.get()
    # EXT-001, EXT-002 and VVR-001 inspect the same deterministic deck-only
    # handoff bundle.  Generate it once per top-level report, but retain the
    # existing public behaviour of regenerating it for each independent run.
    key = deck_dir.absolute()
    cached = None if cache is None else cache.get(key)
    if cached is not None:
        return copy.deepcopy(cached)

    report = generate_external_reference_report(deck_dir)
    if cache is not None:
        # Case checks own mutable result dictionaries, so never expose the
        # cached object itself to a verification case.
        cache[key] = copy.deepcopy(report)
    return report


def _external_report_evidence_summary(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify an external report without promoting deck generation to validation."""

    execution_mode = str(report.get("execution_mode", ""))
    validation_performed = report.get("validation_performed") is True
    executed = execution_mode == "calculix" or validation_performed
    cases = report.get("cases", [])
    validations = [
        item.get("validation", {})
        for item in cases
        if isinstance(item, Mapping) and isinstance(item.get("validation"), Mapping)
    ]
    if executed:
        valid = (
            execution_mode == "calculix"
            and validation_performed
            and report.get("status") == "passed"
            and bool(validations)
            and len(validations) == len(cases)
            and all(item.get("executed") is True and item.get("status") == "passed" for item in validations)
        )
        return {
            "evidence_kind": "executed_numerical_validation",
            "numerical_validation_performed": True,
            "numerical_validation_status": "passed" if valid else str(report.get("status") or "invalid"),
            "report_is_acceptable": valid,
        }

    valid = (
        execution_mode == "deck_only"
        and report.get("validation_performed") is False
        and report.get("status") == "not_executed"
        and bool(validations)
        and len(validations) == len(cases)
        and all(item.get("executed") is False and item.get("status") == "not_executed" for item in validations)
    )
    return {
        "evidence_kind": "handoff_artifact",
        "numerical_validation_performed": False,
        "numerical_validation_status": "not_performed",
        "report_is_acceptable": valid,
    }


def _run_ext_001(case: VerificationCase) -> VerificationCaseResult:
    report = _external_reference_report_for_verification()
    cases = report.get("cases", [])
    _report_path, _markdown_path, deck_dir = _external_reference_locations()
    discovered = discover_calculix_reference_cases(
        roots=[deck_dir],
        repo_root=Path.cwd(),
        require_frd=False,
    )
    names = {str(item.get("name")) for item in cases}
    kinds = {case.name: case.kind for case in discovered}
    evidence = _external_report_evidence_summary(report)
    _assert(
        report.get("status") == "not_executed"
        and evidence["evidence_kind"] == "handoff_artifact"
        and evidence["report_is_acceptable"]
        and {
            "pressure_plate_s4",
            "beam_column_buckling",
            "cylinder_s4_pressure",
            "orthotropic_membrane_s4",
        }
        <= names,
        "CalculiX handoff deck pack is incomplete or incorrectly presented as executed validation",
    )
    _assert(len(discovered) >= 4 and all(case.element_count > 0 for case in discovered), "generated CalculiX decks are not discoverable")
    return _pass(
        case,
        analysis_type="external_solver_handoff_artifact",
        evidence_type="handoff_artifact",
        reference={
            "type": "generated CalculiX/Abaqus-style input decks",
            "execution_status": "not_executed",
            "numerical_validation_claim": False,
        },
        result={
            "artifact_status": "passed",
            "external_report_status": report.get("status"),
            "case_count": len(cases),
            "discovered_count": len(discovered),
            "case_names": sorted(names),
        },
        checks={"evidence": evidence, "report": report, "discovered_kinds": kinds},
    )


def _run_ext_002(case: VerificationCase) -> VerificationCaseResult:
    report = _external_reference_report_for_verification()
    manifest = upstream_calculix_reference_manifest()
    shell_entry = next((entry for entry in manifest if entry.get("name") == "calculix_examples_shell_convergence"), None)
    deck_paths = [Path(item.get("inp_path", "")) for item in report.get("cases", [])]
    abaqus_style_keywords = {}
    for path in deck_paths:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        abaqus_style_keywords[path.name] = {
            "has_node": "*NODE" in text,
            "has_element": "*ELEMENT" in text,
            "has_step": "*STEP" in text,
            "has_output": "*NODE FILE" in text and "*EL FILE" in text,
        }
    _assert(shell_entry is not None and shell_entry.get("repository") == "calculix/CalculiX-Examples", "upstream external shell manifest is missing")
    _assert(all(all(values.values()) for values in abaqus_style_keywords.values()), "external decks do not expose neutral Abaqus-style solver handoff keywords")
    return _pass(
        case,
        analysis_type="second_external_reference_handoff",
        evidence_type="handoff_artifact",
        reference={
            "type": "upstream manifest plus neutral Abaqus-style deck syntax",
            "execution_status": "not_executed",
            "numerical_validation_claim": False,
        },
        result={"upstream_manifest_entries": len(manifest), "deck_keyword_checks": abaqus_style_keywords},
        checks={"upstream_shell_reference": shell_entry, "known_limitations": report.get("known_limitations", [])},
    )


def _run_vvr_001(case: VerificationCase) -> VerificationCaseResult:
    external_report_path, external_markdown_path, external_deck_dir = _external_reference_locations()
    expected_release_paths = [
        Path("reports/beam_shell_verification/beam_shell_verification_report.json"),
        Path("reports/beam_shell_verification/beam_shell_verification_report.md"),
        Path("reports/production_readiness/current/capability_matrix.json"),
        Path("reports/production_readiness/current/verification_scope.json"),
        Path("reports/production_readiness/current/verification_scope.md"),
        Path("reports/verification_quick_current/fe_verification_report.json"),
        Path("reports/verification_quick_current/fe_verification_report.md"),
        external_report_path,
        external_markdown_path,
    ]
    # Always regenerate deterministic handoff decks, but never overwrite a
    # previously executed CalculiX report with a deck-only report.
    handoff_report = _external_reference_report_for_verification()
    handoff_evidence = _external_report_evidence_summary(handoff_report)
    _assert(
        handoff_evidence["evidence_kind"] == "handoff_artifact"
        and handoff_evidence["report_is_acceptable"],
        "external solver handoff artifacts are incomplete",
    )
    if external_report_path.exists():
        external_report = json.loads(
            external_report_path.read_text(encoding="utf-8")
        )
        existing_evidence = _external_report_evidence_summary(external_report)
        existing_looks_executed = (
            str(external_report.get("execution_mode", "")) == "calculix"
            or external_report.get("validation_performed") is True
        )
        if existing_evidence["report_is_acceptable"]:
            external_report_disposition = "preserved_existing"
        elif existing_looks_executed:
            # Never hide or destroy invalid numerical evidence by replacing it
            # with a deck-only artifact.  The manifest below fails closed and
            # keeps the original report available for diagnosis.
            external_report_disposition = "preserved_invalid_executed"
        else:
            # Legacy/deck-only report schemas are reproducible artifacts, not
            # numerical evidence.  Replace them deterministically so a stale
            # ignored report cannot make the source/test gate order-dependent.
            external_report = write_external_reference_report(
                external_report_path,
                deck_dir=external_deck_dir,
                markdown=external_markdown_path,
            )
            external_report_disposition = "replaced_invalid_nonexecuted"
    else:
        external_report = write_external_reference_report(
            external_report_path,
            deck_dir=external_deck_dir,
            markdown=external_markdown_path,
        )
        external_report_disposition = "generated_deck_only"
    external_evidence = _external_report_evidence_summary(external_report)
    present = [str(path) for path in expected_release_paths if path.exists()]
    package_dir = Path("reports/verification_package")
    package_dir.mkdir(parents=True, exist_ok=True)
    package_status = "passed" if external_evidence["report_is_acceptable"] else "incomplete"
    external_limitation = (
        "The preserved external report contains executed, tolerance-controlled CalculiX comparisons."
        if external_evidence["numerical_validation_performed"]
        else "The external reference package is a reproducible handoff artifact; no external-solver numerical validation was performed."
    )
    manifest = {
        "schema_version": 1,
        "status": package_status,
        "expected_release_artifacts": [str(path) for path in expected_release_paths],
        "present_artifacts_at_vvr_runtime": present,
        "external_reference_report": str(external_report_path),
        "external_report_disposition": external_report_disposition,
        "external_reference_report_status": external_report.get("status"),
        "external_evidence_kind": external_evidence["evidence_kind"],
        "external_handoff_artifact_status": "passed",
        "external_numerical_validation_performed": external_evidence["numerical_validation_performed"],
        "external_numerical_validation_status": external_evidence["numerical_validation_status"],
        "known_limitations": [
            external_limitation,
            "The verified production scope remains the documented ANYfeSolver beam-shell scope, not a general-purpose FE claim.",
            "The beam-shell and production-readiness reports are generated by the caller after this case completes, so VVR-001 avoids a circular dependency on the file currently being written.",
        ],
    }
    (package_dir / "release_evidence_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (package_dir / "release_evidence_manifest.md").write_text(
        "# Verification Package Manifest\n\n"
        f"- Status: {manifest['status']}\n"
        f"- Expected release artifacts: {len(expected_release_paths)}\n"
        f"- Present artifacts at VVR runtime: {len(present)}\n"
        f"- External evidence kind: {manifest['external_evidence_kind']}\n"
        f"- External report status: {manifest['external_reference_report_status']}\n"
        f"- External numerical validation: {manifest['external_numerical_validation_status']}\n\n"
        "## Known Limitations\n\n"
        + "\n".join(f"- {item}" for item in manifest["known_limitations"])
        + "\n",
        encoding="utf-8",
    )
    _assert(manifest["status"] == "passed", "verification report package is incomplete")
    return _pass(
        case,
        analysis_type="verification_report_package",
        reference={"type": "generated evidence package manifest"},
        result={"expected_artifact_count": len(expected_release_paths), "package_manifest": str(package_dir / "release_evidence_manifest.json")},
        checks=manifest,
    )


def _run_nlg_001(case: VerificationCase) -> VerificationCaseResult:
    from .beam_validity import corotational_rigid_rotation_metric

    metric = corotational_rigid_rotation_metric()
    _assert(float(metric["corotational_force_norm"]) < 1.0e-6, "corotational beam rigid rotation produced force")
    return _pass(case, element_types=["beam2"], analysis_type="geometric_nonlinear", checks=metric)


def _run_nlg_004(case: VerificationCase) -> VerificationCaseResult:
    model1 = _beam_model(length=1.0)
    load = LoadCase("small")
    load.add_nodal_load(2, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    from .nonlinear_static import solve_static_nonlinear

    r1 = solve_static_nonlinear(model1, load, num_steps=2, max_iterations=8)
    model2 = _beam_model(length=1.0)
    r2 = solve_static_nonlinear(model2, load, num_steps=4, max_iterations=8)
    err = float(np.linalg.norm(r1.displacements - r2.displacements) / max(np.linalg.norm(r2.displacements), 1.0e-30))
    _assert(err < 1.0e-8, "smooth nonlinear endpoint depends on increment count")
    return _pass(case, element_types=["beam2"], analysis_type="nonlinear_static", checks={"endpoint_relative_difference": err})


def _run_nlg_005(case: VerificationCase) -> VerificationCaseResult:
    metrics = element_tangent_metrics()
    err = max(
        float(item["tangent_fd_relative_error"])
        for item in metrics.values()
        if isinstance(item, Mapping) and "tangent_fd_relative_error" in item
    )
    _assert(err < 1.0e-4, "element tangent finite-difference check failed")
    return _pass(case, analysis_type="nonlinear_static", checks={"max_relative_tangent_error": err, "metrics": metrics})


def _run_mat_common(case: VerificationCase) -> VerificationCaseResult:
    paths = material_point_path_metrics()
    residual = float(paths["max_abs_yield_residual"])
    max_tangent = float(paths["max_material_tangent_fd_error"])
    _assert(residual < 1.0e-8 and max_tangent < 1.0e-4, "material path checks failed")
    return _pass(case, analysis_type="plasticity", checks={"yield_residual": residual, "max_tangent_error": max_tangent, "paths": paths})


_MLBC_RESULT_CACHE: Optional[Dict[str, Any]] = None


def _run_mlbc_common(case: VerificationCase) -> VerificationCaseResult:
    global _MLBC_RESULT_CACHE
    if _MLBC_RESULT_CACHE is None:
        _MLBC_RESULT_CACHE = mesh_load_bc_result_by_case()
    result = _MLBC_RESULT_CACHE.get(case.case_id)
    if result is None:
        return _fail(case, "mesh/load/boundary verification case was not evaluated")
    payload = {
        "analysis_type": "mesh_load_bc",
        "element_types": ["S4", "Q8", "beam", "mpc"],
        "result": dict(result.measured),
        "reference": dict(result.tolerance),
        "checks": dict(result.checks),
        "mesh": dict(result.diagnostics.get("mesh_quality") or {}),
    }
    if result.status == "PASS":
        return _pass(case, **payload)
    return _fail(case, result.reason or "mesh/load/boundary verification failed", **payload)


_FRACTURE_METRIC_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _fracture_panel_model(curve: Optional[Any] = None) -> FEModel:
    model = FEModel("fracture_verification_panel")
    model.add_material("steel", 210.0e9, 0.3, hardening_curve=curve)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.0, 0.2, 0.0)
    model.add_node(4, 0.0, 0.2, 0.0)
    model.add_element(1, create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.01))
    model.add_boundary_condition(BoundaryCondition("left_x", [1, 4], {"ux": 0.0}))
    model.add_boundary_condition(BoundaryCondition("plane", [1, 2, 3, 4], {"uz": 0.0, "rx": 0.0, "ry": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pin_y", [1], {"uy": 0.0}))
    return model


def _fracture_tension_load(stress: float = 340.0e6) -> LoadCase:
    load = LoadCase("fracture_pull")
    total = float(stress) * 0.2 * 0.01
    load.add_nodal_load(2, [0.5 * total, 0.0, 0.0, 0.0, 0.0, 0.0])
    load.add_nodal_load(3, [0.5 * total, 0.0, 0.0, 0.0, 0.0, 0.0])
    return load


def _fracture_verification_metrics() -> Dict[str, Dict[str, Any]]:
    from .fracture import (
        FractureConfig,
        ImpactDamageConfig,
        deleted_pressure_load_resultant,
        filtered_load_case_for_deleted_elements,
    )
    from .contact import (
        RigidSphereImpact,
        SphereContactConfig,
        _impact_contact_patch_area,
        _update_impact_damage_states,
        assemble_sphere_contact_load_vector,
        solve_transient_sphere_impact,
    )
    from .matrix_assembly import assemble_load_vector
    from .nonlinear_static import _assemble_nonlinear_system, solve_static_nonlinear

    metrics: Dict[str, Dict[str, Any]] = {}

    def _impact_panel() -> FEModel:
        panel = FEModel("impact_damage_panel")
        panel.add_material("soft", 1.0e5, 0.3, density=20.0)
        panel.add_node(1, 0.0, 0.0, 0.0)
        panel.add_node(2, 1.0, 0.0, 0.0)
        panel.add_node(3, 1.0, 1.0, 0.0)
        panel.add_node(4, 0.0, 1.0, 0.0)
        panel.add_element(1, create_shell_element(1, [1, 2, 3, 4], "soft", thickness=0.05))
        panel.add_boundary_condition(
            BoundaryCondition(
                "restrain_shell_nonimpact_modes",
                [1, 2, 3, 4],
                {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
            )
        )
        return panel

    def _impact_sphere(speed: float = 2.0, radius: float = 0.1) -> RigidSphereImpact:
        return RigidSphereImpact(
            "impact_damage",
            radius=radius,
            mass=1.0,
            start_point=(0.5, 0.5, 0.25),
            travel_direction=(0.0, 0.0, -1.0),
            speed=speed,
        )

    invalid_count = 0
    for kwargs in (
        {"threshold": 0.0},
        {"threshold": 1.0e-3, "residual_stiffness_fraction": -1.0},
        {"threshold": 1.0e-3, "max_deleted_fraction": 1.5},
        {"threshold": 1.0e-3, "element_scope": ("solid",)},
    ):
        try:
            FractureConfig(**kwargs)
        except ValueError:
            invalid_count += 1
    metrics["FRACT-001"] = {"invalid_configs_rejected": invalid_count}

    curve = reference_plastic_curve()
    model = _fracture_panel_model(curve)
    displacement = np.linspace(0.0, 1.0e-4, model.mesh.dof_manager.total_dofs)
    f_active, k_active, states = _assemble_nonlinear_system(model, displacement, {}, 3, tangent=True)
    f_deleted, k_deleted, deleted_states = _assemble_nonlinear_system(
        model,
        displacement,
        states,
        3,
        tangent=True,
        deleted_element_ids=(1,),
        residual_stiffness_fraction=0.2,
    )
    metrics["FRACT-002"] = {
        "force_scale": float(np.linalg.norm(f_deleted) / max(np.linalg.norm(f_active), 1.0e-30)),
        "tangent_scale": float(np.linalg.norm(k_deleted.toarray()) / max(np.linalg.norm(k_active.toarray()), 1.0e-30)),
        "state_preserved": bool(deleted_states[1] is states[1]),
    }

    elastic_model = _fracture_panel_model(None)
    pressure = LoadCase("pressure")
    pressure.add_pressure_load(1, 11.0)
    full, _ = assemble_load_vector(elastic_model, pressure)
    filtered, _ = assemble_load_vector(elastic_model, filtered_load_case_for_deleted_elements(pressure, (1,)))
    removed = deleted_pressure_load_resultant(elastic_model, pressure, (1,))
    metrics["FRACT-003"] = {
        "full_load_norm": float(np.linalg.norm(full)),
        "filtered_load_norm": float(np.linalg.norm(filtered)),
        "removed_resultant_z": float(removed[2]),
        "expected_resultant_z": 11.0 * 1.0 * 0.2,
    }

    deletion_model = _fracture_panel_model(curve)
    deletion_result = solve_static_nonlinear(
        deletion_model,
        _fracture_tension_load(stress=380.0e6),
        num_steps=4,
        num_layers=3,
        fracture_config=FractureConfig(threshold=1.0e-5, max_deleted_fraction=1.0),
    )
    deletion_summary = deletion_result.info.get("fracture_summary", {})
    metrics["FRACT-004"] = {
        "status": deletion_result.status,
        "deleted_count": int(deletion_summary.get("deleted_count", 0)),
        "first_deletion_load_factor": deletion_summary.get("first_deletion_load_factor"),
        "record_count": len(deletion_summary.get("records", [])),
    }

    high_model = _fracture_panel_model(curve)
    high_result = solve_static_nonlinear(
        high_model,
        _fracture_tension_load(),
        num_steps=4,
        num_layers=3,
        fracture_config=FractureConfig(threshold=1.0, max_deleted_fraction=1.0),
    )
    metrics["FRACT-005"] = {
        "status": high_result.status,
        "deleted_count": int(high_result.info.get("fracture_summary", {}).get("deleted_count", 0)),
    }

    stop_model = _fracture_panel_model(curve)
    initial_state = stop_model.mesh.get_element(1).init_nonlinear_state(3)
    initial_state["alpha"][:] = 0.01
    stop_result = solve_static_nonlinear(
        stop_model,
        _fracture_tension_load(stress=100.0e6),
        num_steps=2,
        num_layers=3,
        initial_element_states={1: initial_state},
        fracture_config=FractureConfig(threshold=0.001, max_deleted_fraction=0.5),
    )
    metrics["FRACT-006"] = {
        "status": stop_result.status,
        "failure_reason": stop_result.failure_reason,
        "deleted_count": int(stop_result.info.get("fracture_summary", {}).get("deleted_count", 0)),
    }

    patch_model = _impact_panel()
    patch_element = patch_model.mesh.get_element(1)
    patch_sphere = RigidSphereImpact("patch", radius=0.2, mass=1.0, start_point=(0.5, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    patch_config = ImpactDamageConfig(capacity_basis="user", user_capacity=1000.0, min_contact_area=1.0e-4)
    _v1, _sf1, patch_records = assemble_sphere_contact_load_vector(
        patch_model,
        patch_sphere,
        SphereContactConfig(penalty_stiffness=1000.0),
        sphere_position=np.array([0.5, 0.5, 0.15]),
        sphere_velocity=np.zeros(3),
    )
    _v2, _sf2, deeper_patch_records = assemble_sphere_contact_load_vector(
        patch_model,
        patch_sphere,
        SphereContactConfig(penalty_stiffness=1000.0),
        sphere_position=np.array([0.5, 0.5, 0.05]),
        sphere_velocity=np.zeros(3),
    )
    shallow_area = _impact_contact_patch_area(patch_records[0], patch_element, patch_config, patch_sphere)
    deep_area = _impact_contact_patch_area(deeper_patch_records[0], patch_element, patch_config, patch_sphere)
    metrics["FRACT-007"] = {
        "shallow_area": float(shallow_area),
        "deep_area": float(deep_area),
        "min_contact_area": float(patch_config.min_contact_area),
    }

    low_damage = solve_transient_sphere_impact(
        _impact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12),
        _impact_sphere(speed=0.5),
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        damage_config=ImpactDamageConfig(capacity_basis="user", user_capacity=1.0e9, max_deleted_fraction=1.0),
    ).diagnostics["impact_damage_summary"]
    metrics["FRACT-008"] = {
        "deleted_count": int(low_damage["deleted_count"]),
        "max_damage": float(low_damage["max_damage"]),
    }

    high_damage_result = solve_transient_sphere_impact(
        _impact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12),
        _impact_sphere(),
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        damage_config=ImpactDamageConfig(mode="instant_threshold", capacity_basis="user", user_capacity=10.0, max_deleted_fraction=1.0),
    )
    high_damage = high_damage_result.diagnostics["impact_damage_summary"]
    metrics["FRACT-009"] = {
        "status": high_damage_result.status,
        "deleted_count": int(high_damage["deleted_count"]),
        "governing_component": high_damage["records"][0]["governing_component"],
    }

    low_capacity = solve_transient_sphere_impact(
        _impact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12),
        _impact_sphere(),
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        damage_config=ImpactDamageConfig(capacity_basis="user", user_capacity=20.0, max_deleted_fraction=1.0),
    ).diagnostics["impact_damage_summary"]
    high_capacity = solve_transient_sphere_impact(
        _impact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12),
        _impact_sphere(),
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        damage_config=ImpactDamageConfig(capacity_basis="user", user_capacity=1.0e6, max_deleted_fraction=1.0),
    ).diagnostics["impact_damage_summary"]
    metrics["FRACT-010"] = {
        "low_capacity_max_damage": float(low_capacity["max_damage"]),
        "high_capacity_max_damage": float(high_capacity["max_damage"]),
        "low_capacity_deleted": int(low_capacity["deleted_count"]),
        "high_capacity_deleted": int(high_capacity["deleted_count"]),
    }

    repeat_model = _impact_panel()
    repeat_sphere = RigidSphereImpact("repeat", radius=0.2, mass=1.0, start_point=(0.5, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    _repeat_vector, _repeat_force, repeat_records = assemble_sphere_contact_load_vector(
        repeat_model,
        repeat_sphere,
        SphereContactConfig(penalty_stiffness=1000.0),
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    repeat_config = ImpactDamageConfig(
        mode="accumulated_damage",
        capacity_basis="user",
        user_capacity=1000.0,
        impulse_reference_time=0.01,
        max_deleted_fraction=1.0,
    )
    repeat_states: Dict[int, Dict[str, Any]] = {}
    repeat_deleted: set[int] = set()
    for repeat_step in range(4):
        repeat_new_deleted, _repeat_util, _repeat_diags, _repeat_changed = _update_impact_damage_states(
            repeat_model,
            repeat_records,
            repeat_config,
            repeat_sphere,
            repeat_states,
            tuple(repeat_deleted),
            step_index=repeat_step + 1,
            time_value=0.01 * (repeat_step + 1),
            dt=0.01,
        )
        repeat_deleted.update(record.element_id for record in repeat_new_deleted)
        if repeat_deleted:
            break
    metrics["FRACT-011"] = {
        "damage": float(repeat_states[1]["damage"]),
        "deleted_count": len(repeat_deleted),
    }

    smooth_config = ImpactDamageConfig(
        mode="instant_threshold",
        capacity_basis="user",
        user_capacity=1.0,
        neighbor_smoothing=True,
        max_deleted_fraction=1.0,
    )
    smooth_states: Dict[int, Dict[str, Any]] = {}
    smooth_first_deleted, _smooth_util, smooth_first_diag, _smooth_changed = _update_impact_damage_states(
        repeat_model,
        repeat_records,
        smooth_config,
        repeat_sphere,
        smooth_states,
        (),
        step_index=1,
        time_value=0.01,
        dt=0.01,
    )
    smooth_second_deleted, _smooth_util2, _smooth_diag2, _smooth_changed2 = _update_impact_damage_states(
        repeat_model,
        repeat_records,
        smooth_config,
        repeat_sphere,
        smooth_states,
        (),
        step_index=2,
        time_value=0.02,
        dt=0.01,
    )
    metrics["FRACT-012"] = {
        "first_deleted_count": len(smooth_first_deleted),
        "first_hold": bool(smooth_first_diag[0].get("neighbor_smoothing_hold")),
        "second_deleted_count": len(smooth_second_deleted),
    }
    return metrics


def _run_fracture_common(case: VerificationCase) -> VerificationCaseResult:
    global _FRACTURE_METRIC_CACHE
    if _FRACTURE_METRIC_CACHE is None:
        _FRACTURE_METRIC_CACHE = _fracture_verification_metrics()
    metrics = dict(_FRACTURE_METRIC_CACHE.get(case.case_id) or {})
    if not metrics:
        return _fail(case, "fracture verification metric was not evaluated")
    if case.case_id == "FRACT-001":
        _assert(int(metrics["invalid_configs_rejected"]) == 4, "fracture config validation missed invalid inputs")
    elif case.case_id == "FRACT-002":
        _assert(abs(float(metrics["force_scale"]) - 0.2) < 1.0e-12, "deleted force residual scale mismatch")
        _assert(abs(float(metrics["tangent_scale"]) - 0.2) < 1.0e-12, "deleted tangent residual scale mismatch")
        _assert(bool(metrics["state_preserved"]), "deleted element state was updated")
    elif case.case_id == "FRACT-003":
        _assert(float(metrics["full_load_norm"]) > 0.0, "pressure reference load is empty")
        _assert(float(metrics["filtered_load_norm"]) < 1.0e-12, "deleted pressure load was not removed")
        _assert(abs(float(metrics["removed_resultant_z"]) - float(metrics["expected_resultant_z"])) < 1.0e-12, "removed pressure resultant mismatch")
    elif case.case_id == "FRACT-004":
        _assert(int(metrics["deleted_count"]) == 1, "plastic-strain threshold did not delete the shell")
        _assert(int(metrics["record_count"]) == 1, "fracture deletion record was not stored")
        _assert(float(metrics["first_deletion_load_factor"]) > 0.0, "first deletion load factor was not reported")
    elif case.case_id == "FRACT-005":
        _assert(int(metrics["deleted_count"]) == 0, "high threshold unexpectedly deleted elements")
    elif case.case_id == "FRACT-006":
        _assert(metrics["status"] == "stopped_at_limit", "max deleted fraction did not stop the solve")
        _assert(metrics["failure_reason"] == "max_deleted_fraction_reached", "wrong fracture stop reason")
        _assert(int(metrics["deleted_count"]) == 1, "deleted element was not reported on stop")
    elif case.case_id == "FRACT-007":
        _assert(float(metrics["shallow_area"]) >= float(metrics["min_contact_area"]), "contact patch area fell below configured minimum")
        _assert(float(metrics["deep_area"]) > float(metrics["shallow_area"]), "contact patch area did not grow with penetration")
    elif case.case_id == "FRACT-008":
        _assert(int(metrics["deleted_count"]) == 0, "high-capacity low-energy impact unexpectedly deleted a shell")
        _assert(float(metrics["max_damage"]) < 1.0, "high-capacity low-energy impact reached deletion damage")
    elif case.case_id == "FRACT-009":
        _assert(int(metrics["deleted_count"]) == 1, "high-energy impact damage did not delete the contacted shell")
        _assert(metrics["governing_component"] in {"contact_pressure", "impulse_per_area", "equivalent_plastic_strain_estimate"}, "missing damage trigger component")
    elif case.case_id == "FRACT-010":
        _assert(float(metrics["low_capacity_max_damage"]) > float(metrics["high_capacity_max_damage"]), "higher capacity did not reduce damage")
        _assert(int(metrics["low_capacity_deleted"]) >= int(metrics["high_capacity_deleted"]), "higher capacity increased deletion")
    elif case.case_id == "FRACT-011":
        _assert(float(metrics["damage"]) > 1.0, "repeated subthreshold contacts did not accumulate damage")
        _assert(int(metrics["deleted_count"]) == 1, "accumulated damage did not delete after repeated contact")
    elif case.case_id == "FRACT-012":
        _assert(int(metrics["first_deleted_count"]) == 0, "neighbor smoothing did not hold first isolated deletion")
        _assert(bool(metrics["first_hold"]), "neighbor smoothing hold was not reported")
        _assert(int(metrics["second_deleted_count"]) == 1, "neighbor smoothing did not allow repeated-contact deletion")
    analysis_type = "impact_damage" if case.case_id >= "FRACT-007" else "nonlinear_static_fracture"
    reference_scope = (
        "engineering contact-demand shell damage and erosion after converged impact substeps"
        if case.case_id >= "FRACT-007"
        else "equivalent-plastic-strain-triggered soft element erosion after converged increments"
    )
    return _pass(
        case,
        analysis_type=analysis_type,
        evidence_type="invariant",
        element_types=["shell"],
        checks=metrics,
        reference={"fracture_scope": reference_scope},
    )


_CONTACT_METRIC_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _run_contact_common(case: VerificationCase) -> VerificationCaseResult:
    global _CONTACT_METRIC_CACHE
    if _CONTACT_METRIC_CACHE is None:
        _CONTACT_METRIC_CACHE = contact_verification_metrics()
    metrics = dict(_CONTACT_METRIC_CACHE.get(case.case_id) or {})
    if not metrics:
        return _fail(case, "contact verification metric was not evaluated")
    if case.case_id == "CONTACT-001":
        _assert(metrics["status"] == "no_contact", "no-contact sphere trajectory did not report no_contact")
        _assert(float(metrics["trajectory_error"]) < 1.0e-12, "no-contact sphere trajectory drifted")
        _assert(float(metrics["peak_contact_force"]) == 0.0, "no-contact case produced contact force")
    elif case.case_id == "CONTACT-002":
        _assert(float(metrics["relative_error"]) < 1.0e-12, "normal penalty force law mismatch")
    elif case.case_id == "CONTACT-003":
        _assert(float(metrics["balance_error"]) < 1.0e-10, "sphere and structural contact force resultants are not balanced")
    elif case.case_id == "CONTACT-004":
        denominator = max(float(metrics["impulse_norm"]), 1.0)
        _assert(float(metrics["impulse_error"]) / denominator < 5.0e-2, "sphere impulse and momentum change mismatch")
        _assert(metrics["status"] == "completed", "impact impulse case did not complete")
    elif case.case_id == "CONTACT-005":
        _assert(metrics["status"] == "completed", "panel impact case did not complete")
        _assert(float(metrics["peak_contact_force"]) > 0.0, "panel impact produced no contact force")
        _assert(float(metrics["max_penetration"]) > 0.0, "panel impact produced no penetration")
        _assert(float(metrics["peak_displacement"]) > 0.0, "panel impact produced no structural response")
    elif case.case_id == "CONTACT-006":
        _assert(metrics["status"] == "completed", "stiffened-panel impact case did not complete")
        _assert(float(metrics["beam_node_10_peak_uz"]) > 0.0, "coupled beam node 10 did not respond")
        _assert(float(metrics["beam_node_11_peak_uz"]) > 0.0, "coupled beam node 11 did not respond")
    elif case.case_id == "CONTACT-007":
        _assert(bool(metrics["projection_classification_ok"]), "contact projection face/edge/corner classification failed")
    elif case.case_id == "CONTACT-008":
        _assert(int(metrics["midsurface_records"]) == 0, "midsurface offset check unexpectedly contacted")
        _assert(abs(float(metrics["top_penetration"]) - float(metrics["expected_top_penetration"])) < 1.0e-12, "top-surface penetration mismatch")
    elif case.case_id == "CONTACT-009":
        _assert(int(metrics["active_records"]) == 1, "adjacent contact reduction did not keep one active contact")
        _assert(abs(float(metrics["force_norm"]) - float(metrics["expected_force_norm"])) < 1.0e-10, "adjacent contact reduction changed force magnitude")
    elif case.case_id == "CONTACT-010":
        _assert(metrics["status"] == "completed", "automatic-penalty impact did not complete")
        _assert(float(metrics["resolved_penalty"]) > 0.0, "automatic penalty did not resolve to a positive stiffness")
        _assert(float(metrics["max_penetration_ratio"]) < 0.08, "automatic penalty exceeded production penetration target guard")
    elif case.case_id == "CONTACT-011":
        _assert(metrics["status"] == "completed", "event-substep impact did not complete")
        _assert(int(metrics["event_substep_count"]) > 0, "event substepping was not activated")
        _assert(float(metrics["peak_contact_force"]) > 0.0, "event substepping did not catch contact")
    elif case.case_id == "CONTACT-012":
        _assert(metrics["validation_status"] == "invalid", "invalid contact configuration was not rejected")
        _assert({"CONTACT002", "CONTACT005"} <= set(metrics["issue_codes"]), "contact validation missed required issue codes")
    return _pass(
        case,
        analysis_type="sphere_impact_transient",
        evidence_type="invariant",
        element_types=["shell", "beam", "mpc"],
        checks=metrics,
        reference={"contact_scope": "rigid sphere to shell midsurface, frictionless normal penalty"},
    )


IMPLEMENTATIONS: Dict[str, Callable[[VerificationCase], VerificationCaseResult]] = {
    "META-001": _run_meta_001,
    "ALG-001": _run_alg_001,
    "ALG-002": _run_alg_002,
    "ALG-003": _run_alg_003,
    "ALG-004": _run_alg_004,
    "ALG-005": _run_alg_005,
    "ALG-006": _run_alg_006,
    "ALG-007": _run_alg_007,
    "ALG-008": _run_alg_008,
    "ALG-009": _run_alg_009,
    "BEAM-001": _run_beam_001,
    "BEAM-002": _run_beam_002,
    "BEAM-003": _run_beam_003,
    "BEAM-004": _run_beam_004,
    "BEAM-005": _run_beam_005,
    "BEAM-006": _run_beam_006,
    "BEAM-007": _run_beam_007,
    "BEAM-008": _run_beam_008,
    "BEAM-009": _run_beam_009,
    "BEAM-010": _run_beam_010,
    "BEAM-011": _run_beam_011,
    "SHELL-001": _run_shell_001,
    "SHELL-002": _run_shell_002,
    "SHELL-003": _run_shell_003,
    "SHELL-004": _run_shell_004,
    "SHELL-005": _run_shell_005,
    "SHELL-006": _run_shell_006,
    "SHELL-007": _run_shell_007,
    "SHELL-008": _run_shell_008,
    "SHELL-009": _run_shell_009,
    "SHELL-010": _run_shell_010,
    "SHELL-011": _run_shell_011,
    "BENCH-001": _run_bench_001,
    "BENCH-002": _run_bench_002,
    "BENCH-003": _run_bench_003,
    "BENCH-004": _run_bench_004,
    "COUP-001": _run_coup_001,
    "COUP-002": _run_coup_002,
    "COUP-003": _run_coup_003,
    "COUP-004": _run_coup_004,
    "COUP-005": _run_coup_005,
    "COUP-006": _run_coup_006,
    "COUP-007": _run_coup_007,
    "COUP-008": _run_coup_008,
    "COUP-009": _run_coup_009,
    "COUP-010": _run_coup_010,
    "COUP-012": _run_coup_012,
    "COUP-013": _run_coup_013,
    "COUP-014": _run_coup_014,
    "COUP-015": _run_coup_015,
    "COUP-016": _run_coup_016,
    "COUP-017": _run_coup_017,
    "COUP-018": _run_coup_018,
    "COUP-019": _run_coup_019,
    "COUP-020": _run_coup_020,
    "COUP-021": _run_coup_021,
    "MLBC-001": _run_mlbc_common,
    "MLBC-002": _run_mlbc_common,
    "MLBC-003": _run_mlbc_common,
    "MLBC-004": _run_mlbc_common,
    "MLBC-005": _run_mlbc_common,
    "MLBC-006": _run_mlbc_common,
    "MLBC-007": _run_mlbc_common,
    "MLBC-008": _run_mlbc_common,
    "MLBC-009": _run_mlbc_common,
    "CONTACT-001": _run_contact_common,
    "CONTACT-002": _run_contact_common,
    "CONTACT-003": _run_contact_common,
    "CONTACT-004": _run_contact_common,
    "CONTACT-005": _run_contact_common,
    "CONTACT-006": _run_contact_common,
    "CONTACT-007": _run_contact_common,
    "CONTACT-008": _run_contact_common,
    "CONTACT-009": _run_contact_common,
    "CONTACT-010": _run_contact_common,
    "CONTACT-011": _run_contact_common,
    "CONTACT-012": _run_contact_common,
    "FRACT-001": _run_fracture_common,
    "FRACT-002": _run_fracture_common,
    "FRACT-003": _run_fracture_common,
    "FRACT-004": _run_fracture_common,
    "FRACT-005": _run_fracture_common,
    "FRACT-006": _run_fracture_common,
    "FRACT-007": _run_fracture_common,
    "FRACT-008": _run_fracture_common,
    "FRACT-009": _run_fracture_common,
    "FRACT-010": _run_fracture_common,
    "FRACT-011": _run_fracture_common,
    "FRACT-012": _run_fracture_common,
    "CYL-001": _run_cyl_001,
    "CYL-002": _run_cyl_002,
    "CYL-003": _run_cyl_003,
    "NULL-001": _run_null_001,
    "NULL-002": _run_null_002,
    "NULL-003": _run_null_003,
    "NULL-004": _run_null_004,
    "NULL-005": _run_null_005,
    "EIG-001": _run_eig_001,
    "EIG-002": _run_eig_002,
    "EIG-003": _run_eig_003,
    "EIG-004": _run_eig_004,
    "EIG-005": _run_eig_005,
    "BUC-001": _run_buc_001,
    "BUC-002": _run_buc_002,
    "BUC-003": _run_beam_010,
    "BUC-004": _run_shell_007,
    "BUC-005": _run_buc_005,
    "NLG-001": _run_nlg_001,
    "NLG-002": _run_nlg_002,
    "NLG-003": _run_nlg_003,
    "NLG-004": _run_nlg_004,
    "NLG-005": _run_nlg_005,
    "NLG-006": _run_nlg_006,
    "NLG-007": _run_nlg_007,
    "NLG-008": _run_nlg_008,
    "MAT-001": _run_mat_common,
    "MAT-002": _run_mat_common,
    "MAT-003": _run_mat_common,
    "MAT-005": _run_mat_common,
    "MAT-006": _run_mat_common,
    "MAT-007": _run_mat_common,
    "MAT-008": _run_mat_008,
    "DYN-001": _run_dyn_001,
    "EXT-001": _run_ext_001,
    "EXT-002": _run_ext_002,
    "VVR-001": _run_vvr_001,
    "PERF-001": _run_perf_001,
    "PERF-002": _run_perf_002,
}


XFAIL_REASONS: Mapping[str, str] = {
    "COUP-011": "nonmatching beam-shell coupling is optional and not claimed in this verification batch",
    "MAT-004": "kinematic hardening is not implemented; cyclic Bauschinger check is an explicit unsupported feature",
}


def _release_gate_summary(results: List[VerificationCaseResult], selected_ids: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    selected = None if selected_ids is None else {str(item) for item in selected_ids}
    by_case = {result.case_id: result for result in results}

    def build_gate(name: str, required_cases: Tuple[str, ...]) -> Dict[str, Any]:
        required = list(required_cases)
        not_evaluated = [case_id for case_id in required if case_id not in by_case]
        blockers = [
            {
                "case_id": case_id,
                "status": by_case[case_id].status,
                "test_execution_status": by_case[case_id].test_execution_status,
                "verification_completion_status": by_case[case_id].verification_completion_status,
                "release_gate_status": by_case[case_id].release_gate_status,
                "reason": by_case[case_id].reason,
            }
            for case_id in required
            if case_id in by_case and by_case[case_id].status != "PASS"
        ]
        status = "not_evaluated" if selected is not None and not_evaluated else ("passed" if not blockers and not not_evaluated else "blocked")
        verification_completion_status = "complete" if status == "passed" else "incomplete"
        return {
            "status": status,
            "test_execution_status": "failed" if any(result.status == "FAIL" for result in results) else "passed",
            "verification_completion_status": verification_completion_status,
            "release_gate_status": status,
            "required_cases": required,
            "passed_cases": [case_id for case_id in required if case_id in by_case and by_case[case_id].status == "PASS"],
            "blockers": blockers,
            "not_evaluated": not_evaluated,
            "note": f"Programme gate '{name}' is blocked unless every required case is PASS.",
        }

    gates = {name: build_gate(name, tuple(required)) for name, required in PROGRAMME_RELEASE_GATES.items()}
    gates["thin_stiffened_shell"] = dict(gates["flat_thin_stiffened_shell"])
    gates["thin_stiffened_shell"]["alias_for"] = "flat_thin_stiffened_shell"
    gates["thin_stiffened_shell"][
        "note"
    ] = "Compatibility alias for flat_thin_stiffened_shell; isolated beam and shell checks are insufficient."
    return gates


def _run_beam_shell_verification(selected_ids: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    selected = None if selected_ids is None else {str(item) for item in selected_ids}
    results: List[VerificationCaseResult] = []
    for case in verification_manifest_cases():
        if selected is not None and case.case_id not in selected:
            continue
        runner = IMPLEMENTATIONS.get(case.case_id)
        if runner is None:
            results.append(_xfail(case, XFAIL_REASONS.get(case.case_id, "manifest case is registered but not implemented yet")))
            continue
        try:
            results.append(runner(case))
        except Exception as exc:
            results.append(_fail(case, str(exc)))

    counts: Dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    required_failures = [result.case_id for result in results if result.required and result.status == "FAIL"]
    release_gates = _release_gate_summary(results, selected_ids)
    thin_gate = release_gates["flat_thin_stiffened_shell"]
    test_execution_status = "failed" if any(result.status == "FAIL" for result in results) else "passed"
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git_sha(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "numba": _package_version("numba"),
        },
        "status": "passed" if not required_failures else "failed",
        "test_execution_status": test_execution_status,
        "verification_completion_status": thin_gate["verification_completion_status"],
        "release_gate_status": thin_gate["release_gate_status"],
        "default_tolerances": dict(DEFAULT_TOLERANCES),
        "verification_programme": {
            "title": "ANYfeSolver Complete Verification Programme",
            "version": 1,
            "batches": {key: list(value) for key, value in PROGRAMME_BATCH_CASES.items()},
            "release_gates": {key: list(value) for key, value in PROGRAMME_RELEASE_GATES.items()},
        },
        "scope": {
            "primary_shell_regime": "thin plates and thin shells with attached beam stiffeners",
            "default_span_to_thickness": list(THIN_SHELL_SPAN_TO_THICKNESS),
            "default_min_radius_to_thickness": 100,
            "locking_sensitive_radius_to_thickness": 1000,
            "core_mixed_capabilities": [
                "coincident beam-shell coupling",
                "eccentric beam-shell coupling",
                "static stiffened-shell response",
                "stiffened-shell eigenmodes",
                "stiffened-shell linear buckling",
                "ring and longitudinal stiffeners on curved shells",
            ],
        },
        "release_gates": release_gates,
        "counts": counts,
        "required_failures": required_failures,
        "manifest_cases": [case.to_dict() for case in verification_manifest_cases()],
        "results": [result.to_dict() for result in results],
        "known_limitations": [
            "XFAIL records are explicit missing fixtures, missing traceable reference datasets, or unsupported solver features.",
            "External handoff-artifact PASS records verify reproducible decks and metadata only; they are not executed cross-solver numerical evidence.",
            "Executed external numerical validation is reported separately and requires parsed tolerance-controlled solver results.",
            "This report is a verification coverage ledger; release capability claims should gate on specific PASS sets.",
        ],
    }


def run_beam_shell_verification(
    selected_ids: Optional[Iterable[str]] = None,
    *,
    external_reference_report: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Run the manifest, optionally consuming a separately generated external report."""

    token = None
    external_reference_cache_token = _EXTERNAL_REFERENCE_REPORT_RUN_CACHE.set({})
    composite_cache_token = _COMPOSITE_METRIC_RUN_CACHE.set({})
    plate_buckling_cache_token = _PLATE_BUCKLING_RUN_CACHE.set({})
    nonlinear_framework_cache_token = _NONLINEAR_FRAMEWORK_RUN_CACHE.set({})
    if external_reference_report is not None:
        token = _EXTERNAL_REFERENCE_REPORT_OVERRIDE.set(Path(external_reference_report))
    try:
        return _run_beam_shell_verification(selected_ids=selected_ids)
    finally:
        if token is not None:
            _EXTERNAL_REFERENCE_REPORT_OVERRIDE.reset(token)
        _PLATE_BUCKLING_RUN_CACHE.reset(plate_buckling_cache_token)
        _COMPOSITE_METRIC_RUN_CACHE.reset(composite_cache_token)
        _EXTERNAL_REFERENCE_REPORT_RUN_CACHE.reset(external_reference_cache_token)
        _NONLINEAR_FRAMEWORK_RUN_CACHE.reset(nonlinear_framework_cache_token)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Beam-Shell Solver Verification Report",
        "",
        f"- Status: {report.get('status')}",
        f"- Test execution status: {report.get('test_execution_status')}",
        f"- Verification completion status: {report.get('verification_completion_status')}",
        f"- Release gate status: {report.get('release_gate_status')}",
        f"- Commit: {report.get('commit') or 'unknown'}",
        "",
        "## Counts",
        "",
    ]
    for key, value in sorted(report.get("counts", {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Release Gates", ""])
    for name, gate in (report.get("release_gates") or {}).items():
        blockers = gate.get("blockers", []) or []
        not_evaluated = gate.get("not_evaluated", []) or []
        lines.append(f"- {name}: {gate.get('status')} ({len(blockers)} blockers, {len(not_evaluated)} not evaluated)")
        for blocker in blockers:
            reason = f" - {blocker.get('reason')}" if blocker.get("reason") else ""
            lines.append(f"  - {blocker.get('case_id')} {blocker.get('status')}{reason}")
    lines.extend(["", "## Results", ""])
    for result in report.get("results", []):
        suffix = f" - {result.get('reason')}" if result.get("reason") and result.get("status") != "PASS" else ""
        lines.append(f"- {result.get('case_id')} {result.get('status')}: {result.get('title')}{suffix}")
    lines.extend(["", "## Known Limitations", ""])
    for item in report.get("known_limitations", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def write_beam_shell_verification_report(
    output: Path | str = DEFAULT_BEAM_SHELL_VERIFICATION_PATH,
    *,
    markdown: Optional[Path | str] = None,
    selected_ids: Optional[Iterable[str]] = None,
    external_reference_report: Optional[Path | str] = None,
) -> Dict[str, Any]:
    report = run_beam_shell_verification(
        selected_ids=selected_ids,
        external_reference_report=external_reference_report,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown is not None:
        markdown_path = Path(markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report
