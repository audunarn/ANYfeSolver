"""Run and report the frozen S4-improved qualification contract.

The case inventory and numerical tolerances in this module are intentionally
declared before the improved element kernels.  A release may add evidence, but
must not relax this contract in response to a failed implementation.  Full
qualification is deliberately lease-gated because it includes the repository
test suite and several nonlinear verification programs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT_ROOT = ROOT / "reports" / "s4_improved"
SCHEMA_NAME = "anyfesolver.s4_improved.qualification"
SCHEMA_VERSION = 1
PERF_LEASE_ENV = "ANYSOLVER_PERF_LEASE"


@dataclass(frozen=True)
class QualificationCase:
    """One immutable scientific claim in the qualification inventory."""

    case_id: str
    family: str
    level: str
    reference: str
    tolerance_ids: tuple[str, ...] = ()
    requires_external: bool = False


def _family(
    family: str,
    level: str,
    reference: str,
    names: Iterable[str],
    *tolerance_ids: str,
    requires_external: bool = False,
) -> tuple[QualificationCase, ...]:
    return tuple(
        QualificationCase(
            case_id=f"{family}.{name}",
            family=family,
            level=level,
            reference=reference,
            tolerance_ids=tuple(tolerance_ids),
            requires_external=requires_external,
        )
        for name in names
    )


# Dimensionless tolerances are based on a norm or problem-specific reference
# scale.  Absolute floors are used only for quantities whose units are recorded
# by the executing case.  These values form part of the hashed contract below.
TOLERANCES: dict[str, dict[str, Any]] = {
    "compiled_relative_linf": {
        "limit": 2.0e-12,
        "definition": "max_abs(candidate-reference)/max(max_abs(reference), scale_floor)",
    },
    "compiled_absolute_floor": {
        "limit": 2.0e-12,
        "definition": "dimensionless operator floor after case scaling",
    },
    "symmetry_relative_linf": {
        "limit": 2.0e-12,
        "definition": "max_abs(K-K.T)/max(max_abs(K), scale_floor)",
    },
    "rigid_mode_relative_eigenvalue": {
        "limit": 1.0e-9,
        "definition": "abs(lambda_i)/max(abs(non_rigid_spectrum)) for exactly six modes",
    },
    "negative_mode_relative_eigenvalue": {
        "limit": 2.0e-10,
        "definition": "most_negative_lambda/max(abs(non_rigid_spectrum))",
    },
    "patch_relative_l2": {
        "limit": 2.0e-9,
        "definition": "L2 numerical-minus-prescribed generalized strain/resultant",
    },
    "distorted_patch_relative_l2": {
        "limit": 1.0e-8,
        "definition": "L2 numerical-minus-prescribed field on frozen valid distortion",
    },
    "rigid_motion_relative_residual": {
        "limit": 2.0e-9,
        "definition": "norm(internal_force)/max(characteristic_force, scale_floor)",
    },
    "objectivity_relative_linf": {
        "limit": 2.0e-8,
        "definition": "rotated response versus push-forward of reference response",
    },
    "tangent_directional_relative_l2": {
        "limit": 2.0e-5,
        "definition": "analytic directional tangent versus centered-difference residual oracle",
    },
    "virtual_work_relative": {
        "limit": 2.0e-10,
        "definition": "abs(delta_u dot residual-internal_work)/reference_work_scale",
    },
    "mass_total_relative": {
        "limit": 2.0e-12,
        "definition": "assembled translational mass versus analytic area-density mass",
    },
    "mass_moment_relative": {
        "limit": 2.0e-10,
        "definition": "centre-of-mass and inertia moments versus analytic values",
    },
    "modal_frequency_relative": {
        "limit": 5.0e-4,
        "definition": "qualified-mesh frequency versus analytic or published reference",
    },
    "modal_mac": {
        "limit": 0.995,
        "comparison": "greater_equal",
        "definition": "minimum modal assurance criterion after repeated-mode subspace matching",
    },
    "buckling_factor_relative": {
        "limit": 2.0e-2,
        "definition": "qualified-mesh eigenvalue versus analytic or published reference",
    },
    "recovery_relative_linf": {
        "limit": 2.0e-11,
        "definition": "compiled recovered numeric field versus scalar improved-theory oracle",
    },
    "recovery_absolute_scaled": {
        "limit": 5.0e-10,
        "definition": "absolute field error after case stress/resultant scaling",
    },
    "full_reduced_scatter_relative_linf": {
        "limit": 5.0e-12,
        "definition": "direct reduced result versus explicit T.T/full/T projection",
    },
    "convergence_slope_fraction": {
        "limit": 0.85,
        "comparison": "greater_equal",
        "definition": "observed asymptotic slope divided by published/analytic expected slope",
    },
}


SWEEPS: dict[str, Any] = {
    "patch_geometries": [
        "square",
        "parallelogram",
        "skew",
        "tapered",
        "high_aspect_ratio",
        "mild_warp",
        "strong_valid_warp",
        "global_rotation",
        "singly_curved",
        "doubly_curved",
    ],
    "patch_fields": [
        "membrane_extension",
        "in_plane_shear",
        "pure_bending",
        "transverse_shear",
    ],
    "slenderness_L_over_t": [10, 30, 100, 300, 1000, 3000, 10000, 30000],
    "aspect_ratio": [1, 2, 5, 10, 20],
    "distortion_sequences": {
        "skew_offset_over_height": [0.0, 0.10, 0.25, 0.50, 0.75],
        "taper_top_over_bottom": [1.0, 0.80, 0.60, 0.40, 0.25],
        "warpage_over_short_edge": [0.0, 0.01, 0.03, 0.07, 0.12],
    },
    "mesh_sequences": ["regular", "irregular"],
    "uniform_refinements": [1, 2, 4, 8, 16, 32],
}


# JSON-native fixtures are deliberately kept in this runner instead of the
# implementation modules. They are independent inputs, not values generated
# by the candidate element. Parametric curved patches are sampled by the test
# harness without calling a live geometry model.
FROZEN_FIXTURES: dict[str, Any] = {
    "units": "SI",
    "random_seed": 20260812,
    "linear_material": {
        "elastic_modulus": 210.0e9,
        "poisson_ratio": 0.30,
        "density": 7850.0,
        "thickness": 0.010,
    },
    "generalized_section": {
        "A": [
            [2.30e9, 6.20e8, 0.0],
            [6.20e8, 1.85e9, 0.0],
            [0.0, 0.0, 7.20e8],
        ],
        "B": [
            [1.10e5, -2.00e4, 1.50e4],
            [3.00e4, -8.00e4, 9.00e3],
            [-1.20e4, 7.00e3, 4.00e4],
        ],
        "D": [
            [2.10e4, 5.00e3, 0.0],
            [5.00e3, 1.70e4, 0.0],
            [0.0, 0.0, 6.50e3],
        ],
        "As": [[7.50e8, 4.00e7], [4.00e7, 6.20e8]],
        "mass_per_area": 78.50,
        "rotary_inertia_per_area": 6.541666666666667e-4,
    },
    "planar_coordinates": {
        "square": [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "parallelogram": [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.4, 1.0, 0.0],
            [0.4, 1.0, 0.0],
        ],
        "skew": [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.55, 1.0, 0.0],
            [-0.45, 1.0, 0.0],
        ],
        "tapered": [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.70, 1.0, 0.0],
            [0.20, 1.0, 0.0],
        ],
        "high_aspect_ratio": [
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    },
    "warped_coordinates": {
        "mild_warp": [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.01],
            [2.0, 1.0, 0.03],
            [0.0, 1.0, -0.01],
        ],
        "strong_valid_warp": [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.10],
            [2.0, 1.0, 0.25],
            [0.0, 1.0, -0.08],
        ],
    },
    "global_rigid_transform": {
        "axis": [1.0, 2.0, 3.0],
        "angle_radians": 0.71,
        "translation": [1.0e6, -2.0e6, 3.0e6],
    },
    "curved_patches": {
        "singly_curved": {
            "surface": "cylinder",
            "radius": 10.0,
            "angle_radians": [-0.05, 0.05],
            "axial_coordinate": [0.0, 1.0],
            "directors": "analytic_outward_radial_at_corners",
        },
        "doubly_curved": {
            "surface": "sphere",
            "radius": 10.0,
            "longitude_radians": [-0.04, 0.04],
            "latitude_radians": [0.18, 0.26],
            "directors": "analytic_outward_radial_at_corners",
        },
    },
    "prescribed_fields": {
        "membrane_extension": {
            "u_x": "1.0e-4*x",
            "u_y": "-3.0e-5*y",
        },
        "in_plane_shear": {"u_x": "2.0e-4*y", "u_y": "0"},
        "pure_bending": {
            "w": "5.0e-4*x*x",
            "rotation_y": "-1.0e-3*x",
        },
        "transverse_shear": {"w": "2.0e-4*x", "rotation_y": "0"},
    },
    "mass_case": {
        "coordinates_fixture": "square",
        "thickness": 0.020,
        "density": 7850.0,
        "expected_total_mass": 314.0,
        "expected_centre_of_mass": [1.0, 0.5, 0.0],
    },
    "recovery_case": {
        "coordinates_fixture": "strong_valid_warp",
        "displacement_rule": "(arange(24)-11.5)*1.0e-7",
        "surface_coordinates": [-1.0, 1.0],
        "return_local_and_global": True,
    },
}


QUALIFICATION_CASES = (
    *_family(
        "algebra",
        "focused",
        "analytic_identity",
        (
            "partition_of_unity",
            "derivative_sums",
            "nodal_interpolation",
            "basis_orthogonality",
            "jacobian_orientation",
            "director_normalization",
            "face_use_orientation",
            "material_axis_transformation",
            "cyclic_numbering",
            "global_translation",
            "arbitrary_rigid_rotation",
            "isotropy",
            "six_rigid_modes",
            "no_extra_zero_modes",
            "no_negative_elastic_modes",
            "symmetry",
            "virtual_work",
            "scalar_compiled_equality",
            "no_live_geometry_hot_loop",
        ),
        "compiled_relative_linf",
        "symmetry_relative_linf",
        "rigid_mode_relative_eigenvalue",
        "negative_mode_relative_eigenvalue",
        "virtual_work_relative",
    ),
    *_family(
        "patch",
        "focused",
        "prescribed_constant_field",
        ("geometry_field_matrix",),
        "patch_relative_l2",
        "distorted_patch_relative_l2",
    ),
    *_family(
        "locking_distortion",
        "full",
        "analytic_limit_and_convergence",
        ("slenderness_aspect_distortion_matrix",),
        "convergence_slope_fraction",
    ),
    *_family(
        "linear",
        "full",
        "analytic_or_published_primary_reference",
        (
            "cook_membrane",
            "twisted_cantilever",
            "scordelis_lo",
            "pinched_cylinder",
            "hemisphere_with_opening",
            "navier_plate",
            "published_mitc4_plus_d",
            "warped_q4_convergence",
            "stiffened_panel",
            "ring_stiffened_cylinder",
        ),
        "convergence_slope_fraction",
        requires_external=True,
    ),
    *_family(
        "modal_mass",
        "full",
        "analytic_mass_and_independent_modal_reference",
        (
            "total_mass",
            "centre_of_mass",
            "inertia",
            "free_free_modes",
            "plate_frequency",
            "curved_shell_frequency",
            "stiffened_panel_frequency",
            "repeated_modes",
            "drilling_participation",
            "no_spurious_low_modes",
        ),
        "mass_total_relative",
        "mass_moment_relative",
        "modal_frequency_relative",
        "modal_mac",
    ),
    *_family(
        "buckling",
        "full",
        "analytic_or_published_primary_reference",
        (
            "uniaxial_plate",
            "biaxial_plate",
            "shear_plate",
            "axial_cylinder",
            "pressure_cylinder",
            "pressure_ring",
            "stiffened_panel",
            "ring_stiffened_cylinder",
            "distorted_mesh",
            "repeated_modes",
            "eigen_nonlinear_trend",
        ),
        "buckling_factor_relative",
        "modal_mac",
        requires_external=True,
    ),
    *_family(
        "nonlinear",
        "full",
        "analytic_identity_or_independent_path_reference",
        (
            "rigid_rotation",
            "consistent_tangent",
            "large_rotation_cantilever",
            "slit_annular_plate",
            "hemisphere",
            "snap_through",
            "follower_pressure",
            "first_limit_point",
            "imperfect_cylinder",
            "controlled_descending_branch",
            "cutback_sensitivity",
            "distorted_mesh",
        ),
        "rigid_motion_relative_residual",
        "objectivity_relative_linf",
        "tangent_directional_relative_l2",
        requires_external=True,
    ),
    *_family(
        "materials",
        "full",
        "independent_constitutive_driver_and_scalar_element_oracle",
        (
            "isotropic_elastic",
            "j2_perfect_plastic",
            "j2_hardening",
            "orthotropic_elastic",
            "hill48",
            "symmetric_generalized_section",
            "nonsymmetric_B",
            "mass_overrides",
            "initial_stress_prestrain",
            "staged_loading",
            "restart",
            "displacement_control",
            "state_commit_reject",
        ),
        "compiled_relative_linf",
        "tangent_directional_relative_l2",
    ),
    *_family(
        "coupling_source_intent",
        "full",
        "explicit_neutral_FE_fixture_and_transformation_oracle",
        (
            "coincident_beam_shell",
            "eccentric_beam_shell",
            "shell_shell_intersection",
            "torsion_through_intersection",
            "ring_stiffener",
            "independent_mesh_ratio",
            "numbering_invariance",
            "weighted_mpc_parity",
            "member_split_identity",
            "attachment_intent",
            "junction_intent",
            "separate_part_non_coupling",
        ),
        "full_reduced_scatter_relative_linf",
    ),
    *_family(
        "geometry_handoff",
        "full",
        "synthetic_neutral_contract_fixture",
        (
            "model_bound_handle",
            "wrong_model_rejection",
            "active_replaced_deleted_unknown",
            "schema4_forward_boundary",
            "schema3_legacy_boundary",
            "source_revision_mismatch",
            "source_audit_status",
            "face_use_orientation",
            "exact_source_directors",
            "mesh_reconstructed_directors",
            "sharp_intersection",
            "no_cross_sheet_averaging",
            "large_translation_invariance",
            "independent_FE_numbering",
            "compact_provenance_serialization",
        ),
        "objectivity_relative_linf",
    ),
    *_family(
        "recovery",
        "focused",
        "same_improved_operator_scalar_oracle",
        (
            "raw_resultants",
            "top_bottom_stresses",
            "local_global_frames",
            "plastic_layers",
            "generalized_resultants",
            "discontinuity_separation",
            "scalar_compiled_equality",
            "error_indicator_convergence",
            "source_provenance",
        ),
        "recovery_relative_linf",
        "recovery_absolute_scaled",
    ),
)


def qualification_contract() -> dict[str, Any]:
    """Return the canonical, JSON-safe release contract."""

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "theory": "full_2x2_mitc4_plus_d",
        "geometry_boundary": {
            "api": ">=0.2,<0.3",
            "live_observed": "0.2.1",
            "forward_schema": 4,
            "legacy_schema": 3,
            "solver_document_parsing": False,
            "live_geometry_hot_loop_calls_allowed": False,
        },
        "sweeps": SWEEPS,
        "fixtures": FROZEN_FIXTURES,
        "tolerances": TOLERANCES,
        "cases": [asdict(case) for case in QUALIFICATION_CASES],
        "external_status_vocabulary": ["executed", "failed", "unavailable"],
        "full_requires_perf_lease": True,
    }


def qualification_contract_sha256() -> str:
    encoded = json.dumps(
        qualification_contract(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


# Updated only by an intentional review of the pre-implementation contract.
QUALIFICATION_CONTRACT_SHA256 = (
    "1591906DB90B83A7018E51D1C6CF35A545BCE28787ED3108713CDEECC51103F1"
)


def validate_contract() -> None:
    case_ids = [case.case_id for case in QUALIFICATION_CASES]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("qualification case IDs are not unique")
    unknown = sorted(
        {
            tolerance_id
            for case in QUALIFICATION_CASES
            for tolerance_id in case.tolerance_ids
            if tolerance_id not in TOLERANCES
        }
    )
    if unknown:
        raise RuntimeError(f"qualification cases use unknown tolerances: {unknown}")
    actual = qualification_contract_sha256()
    if actual != QUALIFICATION_CONTRACT_SHA256:
        raise RuntimeError(
            "qualification contract changed without updating its reviewed hash: "
            f"expected {QUALIFICATION_CONTRACT_SHA256}, actual {actual}"
        )


def _pytest_command(*paths: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        *paths,
        "-q",
        "-p",
        "no:cacheprovider",
    ]


def _quick_commands() -> list[tuple[str, list[str]]]:
    paths = [
        path
        for path in (
            "tests/test_s4_improved_qualification.py",
            "tests/test_s4_improved_batch.py",
            "tests/test_s4_improved_recovery.py",
        )
        if (ROOT / path).exists()
    ]
    return [("s4_improved_focused", _pytest_command(*paths))]


def _full_commands() -> list[tuple[str, list[str]]]:
    return [
        *_quick_commands(),
        (
            "s4_regression_families",
            _pytest_command(
                "tests/test_fe_solver_s4_validity.py",
                "tests/test_advanced_s4_batches.py",
                "tests/test_recovery_batches.py",
                "tests/test_fe_solver_mass_modal.py",
                "tests/test_fe_solver_buckling.py",
                "tests/test_fe_solver_nonlinear_static.py",
                "tests/test_fe_solver_plasticity_qualification.py",
                "tests/test_beam_shell_verification.py",
            ),
        ),
        ("full_pytest", _pytest_command("tests")),
        (
            "fe_verification",
            [sys.executable, "scripts/run_fe_verification.py"],
        ),
        (
            "beam_shell_verification",
            [sys.executable, "scripts/run_beam_shell_verification.py"],
        ),
        (
            "buckling_validity",
            [sys.executable, "scripts/run_buckling_validity.py"],
        ),
        (
            "mass_modal_validity",
            [sys.executable, "scripts/run_mass_modal_validity.py"],
        ),
        (
            "plasticity_qualification",
            [sys.executable, "scripts/run_plasticity_qualification.py"],
        ),
    ]


def _activation_diagnostics() -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested_formulation": "mitc4_plus_d",
        "legacy_theory_fallback_allowed": False,
        "live_geometry_objects_allowed": False,
    }
    try:
        jit = importlib.import_module("anyfesolver.jit_compiler")
        result["jit"] = jit.jit_diagnostics()
    except Exception as exc:  # pragma: no cover - environment diagnostic
        result["jit"] = {"enabled": False, "disabled_reason": repr(exc)}
    for name in ("mitc4_plus_d_batch", "mitc4_plus_d_recovery"):
        module_name = f"anyfesolver.shell_formulations.{name}"
        try:
            module = importlib.import_module(module_name)
            diagnostics = getattr(module, "kernel_diagnostics", None)
            result[name] = (
                diagnostics() if callable(diagnostics) else {"imported": True}
            )
        except Exception as exc:
            result[name] = {"imported": False, "reason": repr(exc)}
    return result


def _run_command(name: str, command: Sequence[str]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(SRC), environment.get("PYTHONPATH", ""))
        if value
    )
    start = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "name": name,
        "command": list(command),
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": int(completed.returncode),
        "elapsed_seconds": float(time.perf_counter() - start),
        "stdout_tail": completed.stdout[-12000:],
        "stderr_tail": completed.stderr[-12000:],
    }


def _git_sha() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _write_report(report: dict[str, Any], output: Path, markdown: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown is None:
        return
    lines = [
        "# S4 improved qualification",
        "",
        f"- Status: `{report['summary']['status']}`",
        f"- Revision: `{report['revision']['head_sha']}`",
        f"- Contract SHA-256: `{report['contract_sha256']}`",
        f"- PERF lease: `{report['execution']['perf_lease'] or 'not supplied'}`",
        "",
        "| Gate | Status | Seconds |",
        "| --- | --- | ---: |",
    ]
    for gate in report["gates"]:
        lines.append(
            f"| `{gate['name']}` | {gate['status']} | {gate['elapsed_seconds']:.6f} |"
        )
    lines.extend(
        [
            "",
            "External cases remain `unavailable` unless an independent executable or "
            "published-data comparison was actually run; absence is never reported as a pass.",
            "",
        ]
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print the frozen contract only")
    mode.add_argument("--quick", action="store_true", help="run focused owned tests")
    mode.add_argument("--full", action="store_true", help="run the lease-gated full hierarchy")
    parser.add_argument(
        "--perf-lease",
        default=os.environ.get(PERF_LEASE_ENV),
        help=f"ecosystem PERF lease identifier (or ${PERF_LEASE_ENV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "qualification.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPORT_ROOT / "qualification.md",
    )
    parser.add_argument("--no-markdown", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_contract()
    if args.list or (not args.quick and not args.full):
        print(
            json.dumps(
                {
                    **qualification_contract(),
                    "contract_sha256": QUALIFICATION_CONTRACT_SHA256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.full and not args.perf_lease:
        raise SystemExit(
            f"--full requires an ecosystem PERF lease via --perf-lease or {PERF_LEASE_ENV}"
        )

    commands = _full_commands() if args.full else _quick_commands()
    gates = [_run_command(name, command) for name, command in commands]
    failed = sum(gate["status"] == "failed" for gate in gates)
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "report_kind": "full" if args.full else "quick",
        "revision": {"head_sha": _git_sha()},
        "contract_sha256": QUALIFICATION_CONTRACT_SHA256,
        "contract": qualification_contract(),
        "execution": {
            "perf_lease": args.perf_lease,
            "activation": _activation_diagnostics(),
            "external_default_status": "unavailable",
        },
        "gates": gates,
        "summary": {
            "status": "passed" if failed == 0 else "failed",
            "passed": len(gates) - failed,
            "failed": failed,
        },
    }
    _write_report(
        report,
        args.output,
        None if args.no_markdown else args.markdown,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
