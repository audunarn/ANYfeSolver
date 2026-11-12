"""One nested40 projected BL5 external fixture, supplementary development only.

Four complete five-channel [1, xi_halfcell_mean] supports span four shell
intervals each on the SAME 16x4 ASDShellQ4/32 classic ElasticTimoshenkoBeam
fixture. Physical actions include endpoints and remain independent of native C.
This experimental, not-qualified reference does not select a policy, establish
accuracy/locking, or consume formal F03 authority. One case runs per process;
there is no external refinement family or product/global/API/default change.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
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
BALANCE_LIMIT, GAP_LIMIT, DERIVATIVE_LIMIT = 1e-8, 1e-9, 1e-6
BASIS = "discontinuous_p1_cell_average"
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
OPTIONAL_THREAD_VARIABLES = ("BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbid_product_import():
    if any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in sys.modules):
        raise ValueError("independent external runner must not import product modules")


def services():
    """Verify all six sources before importing the unchanged generic helpers."""
    forbid_product_import()
    directory = Path(__file__).resolve().parent
    expected = {AFFINE_SOURCE: AFFINE_SHA256, **HELPER_SHA256}
    actual = {name: sha(directory/name) for name in expected}
    if actual != expected:
        raise ValueError("nested external source identity changed: "+str(actual))
    spec = importlib.util.spec_from_file_location("_nested40_affine_services", directory/AFFINE_SOURCE)
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
    """Standalone four-support admission; unchanged exact helper row algebra."""
    if affine is None:
        affine = services()[0]
    nx, nb, ns = model["nx"], model["beam_intervals"], model["support_count"]
    if (any(type(v) is not int for v in (nx, model["ny"], nb, ns)) or
            (nx, model["ny"], nb, ns) != (16, 4, 32, 4)):
        raise ValueError("one 16x4/32-interval/four-support nested40 fixture required")
    first = model["beam_root"]
    rb, rs = F(reference.BEAM_BOTTOM_ARM_M), F(reference.SHELL_THICKNESS_M)/2
    if (tuple(model["beam_node_ids"]) != tuple(range(first, first+nb+1)) or
            any(tuple(model["nodes"][first+i]) != (float(F(i, nb)), 0., reference.BEAM_AXIS_Z_M)
                for i in range(nb+1)) or
            any(tuple(model["nodes"][1+i]) != (float(F(i, nx)), 0., 0.) for i in range(nx+1))):
        raise ValueError("declared matching physical traces required")
    stride, hb = nb//ns, F(1, nb)
    physical, emitted, supports, all_dependent, exact_blocks = [], [], [], [], []
    for index in range(ns):
        left, right = F(index, ns), F(index+1, ns)
        bw, bp = affine.integrated_projected_weights(left, right, nb, nb, first)
        sw, sp = affine.integrated_projected_weights(left, right, nx, nb, 1)
        block = [{} for _ in range(10)]
        for weights, sign, arm in ((bw, F(1), rb), (sw, F(-1), rs)):
            for degree, nodes in enumerate(weights):
                for n, w in nodes.items():
                    for d in range(5):
                        block[2*d+degree][n, d+1] = sign*w
                    block[degree][n, 5] = sign*arm*w
                    block[2+degree][n, 4] = -sign*arm*w
        quarter_nodes = (first+stride*index+stride//4, first+stride*index+3*stride//4)
        if any((bw[0][n], bw[1][n]) != (hb, sign*hb/2)
               for n, sign in zip(quarter_nodes, (-1, 1))):
            raise ValueError("exact quarter-station shape integrals required")
        dependent = [(n, d) for n in quarter_nodes for d in range(1, 6)]
        if any(model["fixes"].get(n, (0,)*6)[d-1] for n, d in dependent):
            raise ValueError("selected interior dependent coordinates must be free")
        dmatrix = [[row.get(key, F(0)) for key in dependent] for row in block]
        e = affine.inverse_fraction(dmatrix)
        columns = sorted({key for row in block for key in row})
        transformed = [{key: sum(e[i][j]*block[j].get(key, F(0)) for j in range(10))
                        for key in columns} for i in range(10)]
        if any(transformed[i].get(key, F(0)) != (i == j)
               for i in range(10) for j, key in enumerate(dependent)):
            raise ValueError("exact dependent-column identity failed")
        physical.extend(affine.flat_row(row) for row in block)
        emitted.extend(affine.flat_row(row, key) for row, key in zip(transformed, dependent))
        supports.append({"support_id": index, "basis": BASIS, "interval_m": [float(left), float(right)],
            "width_m": float(right-left), "row_indices": list(range(10*index, 10*index+10)),
            "beam_spacing_m": float(hb), "physical_row_order": "channel0..4, degree0/1 per channel",
            "beam_shape_integrals_m": [{n: float(v) for n, v in w.items()} for w in bw],
            "shell_shape_integrals_m": [{n: float(v) for n, v in w.items()} for w in sw],
            "beam_pieces": bp, "shell_pieces": sp, "recovery_beam_nodes": list(quarter_nodes),
            "dependent_node_dofs": dependent, "dependent_coefficient_block_D": [[float(v) for v in row] for row in dmatrix],
            "emitted_from_physical_E": [[float(v) for v in row] for row in e],
            "exact_E_rationals": [[str(v) for v in row] for row in e]})
        all_dependent.extend(dependent)
        exact_blocks.append(block)
    if len(set(all_dependent)) != 40:
        raise ValueError("forty distinct dependent physical coordinates required")
    for index, block in enumerate(exact_blocks):
        other = all_dependent[:10*index]+all_dependent[10*index+10:]
        if any(row.get(key, 0) for row in block for key in other):
            raise ValueError("other-support selected columns must be absent")
    return {"basis": BASIS, "physical_rows": physical, "emitted_rows": emitted, "supports": supports}


def nested_row_correspondence(model, reference, pairing, affine):
    """Compare all coordinates with admitted old80 physical rows by congruence."""
    old = affine.aggregate_rows({**model, "support_count": 8}, reference, basis=BASIS)
    old_maps = affine.coefficient_maps(old["physical_rows"])
    new_maps = affine.coefficient_maps(pairing["physical_rows"])
    error = 0.
    columns_checked = 0
    for support in range(4):
        for channel in range(5):
            left0, left1 = 20*support+2*channel, 20*support+2*channel+1
            right0, right1 = left0+10, left1+10
            for degree, terms in enumerate((((left0, F(1)), (right0, F(1))),
                    ((left0, F(-1, 2)), (left1, F(1, 2)), (right0, F(1, 2)), (right1, F(1, 2))))):
                row = new_maps[10*support+2*channel+degree]
                columns = set(row) | {key for r, _ in terms for key in old_maps[r]}
                for key in columns:
                    predicted = sum(w*F(old_maps[r].get(key, 0.)) for r, w in terms)
                    error = max(error, float(abs(predicted-F(row.get(key, 0.)))))
                    columns_checked += 1
    return {"nested_physical_row_max_abs": error, "nested_physical_columns_checked": columns_checked,
        "nested_physical_row_correspondence": "constant=left0+right0; affine=.5*(-left0+left1+right0+right1); "
            "all five channels, all physical coordinates including support endpoints; no native C input"}


def main(site, output, case):
    if case not in ("fy", "mx", "mz"):
        raise ValueError("one Fy/Mx/Mz nested40 case required")
    output.mkdir(parents=True, exist_ok=False)
    recorder, service, affine = None, None, None
    summary = {"status": "failed", "nx": 16, "ny": 4, "beam_intervals": 32,
        "case": case, "basis": BASIS, "supports": 4, "rows": 40,
        "identity": "BL5_P1_discontinuous_cell_average_span_8",
        "policy": "four shell intervals per support; complete five-channel pair; same coarse model",
        "classification": "EXPERIMENTAL_NOT_QUALIFIED; supplementary independent development reference only; "
            "no accuracy/locking/policy selection/F03 acceptance",
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
        model = refinement.geometry(16, reference)
        model["support_count"] = 4
        pairing = aggregate_rows(model, reference, affine)
        commands = refinement.command_recipe(model, reference, pairing["emitted_rows"], case)
        expected = refinement.planned_inventory(commands)
        if (len(model["nodes"]), len(model["shell_cells"]), len(model["beam_cells"]),
                len(pairing["physical_rows"]), len(expected["equations"])) != (118, 64, 32, 40, 40):
            raise ValueError("fixed nested40 inventory required")
        service._atomic(output/"inputs.json", {"model": model, **pairing, "load": refinement.LOADS[case],
            "command_recipe": commands, "sections": {name: getattr(reference, name) for name in
                ("E_PA", "NU", "G_PA", "SHELL_THICKNESS_M", "BEAM_AREA_M2", "BEAM_TORSION_M4",
                 "BEAM_BENDING_M4", "BEAM_SHEAR_AREA_M2")}})
        controls = affine.preparation_controls(model, pairing, refinement, p0, frozen, service, reference)
        controls.update(nested_row_correspondence(model, reference, pairing, affine))
        service._atomic(output/"preparation-controls.json", controls)
        values = controls["six_common_rigid_field_average_gap_scaled"]+list(controls["objective_average_gap_scaled"].values())
        if (not controls["normal_spin_columns_zero"] or any(not math.isfinite(v) or v > GAP_LIMIT for v in values)
                or any(not math.isfinite(v) or v > DERIVATIVE_LIMIT for v in controls["actual_left_exp_derivative_relative"])
                or any(not math.isfinite(controls[name]) or controls[name] > BALANCE_LIMIT
                       for name in ("row_transform_max_abs", "nested_physical_row_max_abs"))):
            raise ValueError("unchanged physical work/derivative and nested row controls failed")
        sys.path.insert(0, str(site.resolve(strict=True)))
        import openseespy.opensees as ops
        from openseespywin import opensees as core
        forbid_product_import()
        summary.update(opensees=ops.version(), binary=str(Path(core.__file__).resolve()), binary_sha256=sha(core.__file__))
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
        summary.update(status="completed", physical_nodes=118, shell_elements=64, beam_elements=32,
            applied_work_J=derived["work_J"]["applied"], checks=derived["checks"])
    except BaseException as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        if affine is not None:
            affine.finalize(recorder, service, output, summary, earlier_error="error" in summary)
        else:
            # Identity failures cannot execute an unverified helper, even to retain a summary.
            import json
            temporary = output/("summary."+str(os.getpid())+".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output/"summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("fy", "mx", "mz"))
    args = parser.parse_args()
    main(args.site, args.output_dir, args.case)
