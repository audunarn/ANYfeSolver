"""One joint32 projected BL5 supplementary external development fixture.

Fixed 32x8 ASDShellQ4/64 classic ElasticTimoshenkoBeam intervals, eight complete
five-channel [1, xi_halfcell_mean] supports spanning four shell intervals each.
Physical rows include all endpoints and use the unchanged independent exact
reference integrals. One Fy100N/Mx10Nm/Mz10Nm case runs per process.
Experimental and not qualified: no policy selection, accuracy/locking, growing
family, M2 or F03 acceptance. This fixture has its own discretization; no
cross-level nesting assertion or native operator input is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback


AFFINE_SOURCE = "r8_bl5_external_affine_reference_dev.py"
AFFINE_SHA256 = "04c1b930258d5911c500ab3430b196e9488324ae8cafaebbfa30584420a26b08"
HELPER_SHA256 = {
    "b3_e4_bl5_external_p0_refinement_dev.py": "6f3d19d0c95a851d7e1d7dd694f1c1dce051716a811fc4fb7c733e314c4c733a",
    "b3_e4_bl5_external_p0_enforcement_dev.py": "11e593497dab40ae052a491740a6ed50bfe51a3580f275fa87d950dc1004a004",
    "b3_e4_bl5_matched_mz_reference_dev.py": "5b6eb1dac1bb4f58548c7555e10cc798ef4727406ca25efb735808b776444702",
    "b3_e4_bl5_matched_reference_actions.py": "a42d4bdeb96c7106f296e92129104eb896fbc7cb2904d9867bc61c7a9ca169c6",
    "b3_e4_f03_bl5_opensees_line_reference.py": "fe654b7a7c384620dae65f0b0981bb18e512513c090abdfe0dbf6c62c43d15cc",
}
BASIS = "discontinuous_p1_cell_average"
BALANCE_LIMIT, GAP_LIMIT, DERIVATIVE_LIMIT = 1e-8, 1e-9, 1e-6
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
OPTIONAL_THREAD_VARIABLES = ("BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
IDENTITY = "BL5_PROJECTED_FOUR_SHELL_INTERVALS_JOINT32_DEV_V1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbid_product_import():
    if any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in sys.modules):
        raise ValueError("independent external runner must not import product modules")


def services():
    """Verify exact six-source graph before loading unchanged generic services."""
    forbid_product_import()
    directory = Path(__file__).resolve().parent
    expected = {AFFINE_SOURCE: AFFINE_SHA256, **HELPER_SHA256}
    actual = {name: sha(directory/name) for name in expected}
    if actual != expected:
        raise ValueError("joint32 external source identity changed: "+str(actual))
    spec = importlib.util.spec_from_file_location("_joint32_projected_external_services", directory/AFFINE_SOURCE)
    affine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(affine)
    if (affine.HELPER_SHA256 != HELPER_SHA256 or affine.PROJECTED_BASIS != BASIS or
            (affine.BALANCE_LIMIT, affine.GAP_LIMIT, affine.DERIVATIVE_LIMIT) !=
            (BALANCE_LIMIT, GAP_LIMIT, DERIVATIVE_LIMIT)):
        raise ValueError("unchanged projected basis and physical budgets required")
    loaded = affine.services()
    forbid_product_import()
    return affine, *loaded


def aggregate_rows(model, reference, affine=None):
    """Expose the strict fixture using the unchanged generic row assembler."""
    if any(type(model[key]) is not int for key in ("nx", "ny", "beam_intervals", "support_count")) or (
            model["nx"], model["ny"], model["beam_intervals"], model["support_count"]) != (32, 8, 64, 8):
        raise ValueError("one32x8/64-interval/eight-support projected fixture required")
    if affine is None:
        affine = services()[0]
    return affine.aggregate_rows(model, reference, basis=BASIS)


def write_summary(output, summary):
    temporary = output/("summary."+str(os.getpid())+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, output/"summary.json")


def finalize(recorder, service, affine, output, summary):
    """Use the unchanged raw-retention/wipe lifecycle, preserving primary errors."""
    earlier_error = "error" in summary
    try:
        if affine is None:
            write_summary(output, summary)
        else:
            affine.finalize(recorder, service, output, summary, earlier_error=earlier_error)
    except BaseException:
        if not earlier_error:
            raise
        summary["finalization_error"] = traceback.format_exc()
        try:
            write_summary(output, summary)
        except BaseException:
            pass  # An existing primary/cancellation error must still propagate.


def main(site, output, case):
    if case not in ("fy", "mx", "mz"):
        raise ValueError("one fixed Fy/Mx/Mz joint32 case required")
    output.mkdir(parents=True, exist_ok=False)
    recorder, service, affine = None, None, None
    summary = {"status": "failed", "identity": IDENTITY, "nx": 32, "ny": 8, "beam_intervals": 64,
        "case": case, "basis": BASIS, "supports": 8, "rows": 80,
        "fixture": {"nx": 32, "ny": 8, "halfcells": 64, "supports": 8},
        "policy": "four shell intervals per support; complete five-channel pair; joint32 fixture",
        "classification": "EXPERIMENTAL_NOT_QUALIFIED; supplementary independent development reference only; "
            "no accuracy/locking/policy selection/M2/growing-family/F03 acceptance",
        "formal_f03_authority_consumed": False,
        "source_sha256": sha(__file__), "affine_source": {AFFINE_SOURCE: AFFINE_SHA256},
        "helper_sources": HELPER_SHA256, "python": sys.version,
        "requested_threads": {name: os.environ.get(name) for name in THREAD_VARIABLES+OPTIONAL_THREAD_VARIABLES},
        "thread_scope": "One-thread environment requests required; effective external OpenSees threads unresolved; "
            "no resource-qualified claim."}
    try:
        affine, refinement, p0, frozen, service, reference = services()
        if (any(os.environ.get(name) != "1" for name in THREAD_VARIABLES) or
                any(os.environ.get(name) not in (None, "1") for name in OPTIONAL_THREAD_VARIABLES)):
            raise ValueError("one numerical thread environment required")
        if sys.version_info[:2] != (3, 12):
            raise ValueError("verified external OpenSees binary requires Python3.12")
        model = refinement.geometry(32, reference)
        model["support_count"] = 8
        pairing = aggregate_rows(model, reference, affine)
        commands = refinement.command_recipe(model, reference, pairing["emitted_rows"], case)
        expected = refinement.planned_inventory(commands)
        if (len(model["nodes"]), len(model["shell_cells"]), len(model["beam_cells"]),
                len(pairing["physical_rows"]), len(expected["equations"])) != (362, 256, 64, 80, 80):
            raise ValueError("fixed joint32 physical inventory required")
        if [s["interval_m"] for s in pairing["supports"]] != [[i/8, (i+1)/8] for i in range(8)]:
            raise ValueError("eight complete fixed physical supports required")
        service._atomic(output/"inputs.json", {"model": model, **pairing, "load": refinement.LOADS[case],
            "command_recipe": commands, "sections": {name: getattr(reference, name) for name in
                ("E_PA", "NU", "G_PA", "SHELL_THICKNESS_M", "BEAM_AREA_M2", "BEAM_TORSION_M4",
                 "BEAM_BENDING_M4", "BEAM_SHEAR_AREA_M2")}})
        controls = affine.preparation_controls(model, pairing, refinement, p0, frozen, service, reference)
        service._atomic(output/"preparation-controls.json", controls)
        values = controls["six_common_rigid_field_average_gap_scaled"]+list(controls["objective_average_gap_scaled"].values())
        if (not controls["normal_spin_columns_zero"] or any(not math.isfinite(v) or v > GAP_LIMIT for v in values)
                or any(not math.isfinite(v) or v > DERIVATIVE_LIMIT for v in controls["actual_left_exp_derivative_relative"])
                or not math.isfinite(controls["row_transform_max_abs"]) or controls["row_transform_max_abs"] > BALANCE_LIMIT):
            raise ValueError("unchanged rigid/objectivity/actual-rotation derivative controls failed")
        sys.path.insert(0, str(site.resolve(strict=True)))
        import openseespy.opensees as ops
        from openseespywin import opensees as core
        forbid_product_import()
        summary.update(opensees=ops.version(), binary=str(Path(core.__file__).resolve()), binary_sha256=sha(core.__file__),
            expected_binary_sha256=service.BINARY_SHA256)
        if not summary["opensees"].startswith("3.8") or summary["binary_sha256"] != service.BINARY_SHA256:
            raise ValueError("verified OpenSees3.8 binary identity required")
        recorder = service.RecordingOps(ops)
        refinement.solve(recorder, commands, model, reference)
        raw = recorder.raw
        service._atomic(output/"raw-inventory.json", raw)
        inventory = {name: raw[name] == value for name, value in expected.items()}
        inventory["all_primary_poses"] = set(raw["poses"]) == set(model["nodes"])
        inventory["all_element_covectors"] = set(raw["element_global_covectors"]) == set(expected["elements"])
        service._atomic(output/"inventory-predicates.json", inventory)
        if not all(inventory.values()):
            raise ValueError("complete fixed physical inventory required: "+str(inventory))
        derived = affine.recover(raw, model, reference, pairing, case, refinement, service)
        service._atomic(output/"recovered-actions.json", derived)
        service._atomic(output/"interface-recovery-blocks.json", derived["local_recovery"])
        for name, value in derived["checks"].items():
            limit = GAP_LIMIT if name in ("average_five_channel_gap_scaled", "emitted_coordinate_gap_scaled") else BALANCE_LIMIT
            if not math.isfinite(value) or value > limit:
                raise ValueError(f"{name}={value} exceeds unchanged physical budget {limit}")
        forbid_product_import()
        summary.update(status="completed", physical_nodes=362, shell_elements=256, beam_elements=64,
            applied_work_J=derived["work_J"]["applied"], checks=derived["checks"])
    except BaseException as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        finalize(recorder, service, affine, output, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("fy", "mx", "mz"))
    args = parser.parse_args()
    main(args.site, args.output_dir, args.case)
