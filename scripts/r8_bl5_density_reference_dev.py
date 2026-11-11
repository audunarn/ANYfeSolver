"""Fixed Ns64/h16 S4 or S2 Mz supplementary physical-density reference.

One fresh process/load; unchanged generic reference services, historical guards
and physical budgets. No product import, policy, convergence or qualification.
prepare() performs only owned geometry/rational row/command preparation and
does not load OpenSees. Native retained fields are never external nodal fields.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback


IDENTITY = "BL5_FIXED_NS64_H16_DENSITY_REFERENCE_DEV_V1"
AFFINE_SOURCE = "r8_bl5_external_affine_reference_dev.py"
AFFINE_SHA256 = "04c1b930258d5911c500ab3430b196e9488324ae8cafaebbfa30584420a26b08"
HELPER_SHA256 = {
    "b3_e4_bl5_external_p0_refinement_dev.py": "6f3d19d0c95a851d7e1d7dd694f1c1dce051716a811fc4fb7c733e314c4c733a",
    "b3_e4_bl5_external_p0_enforcement_dev.py": "11e593497dab40ae052a491740a6ed50bfe51a3580f275fa87d950dc1004a004",
    "b3_e4_bl5_matched_mz_reference_dev.py": "5b6eb1dac1bb4f58548c7555e10cc798ef4727406ca25efb735808b776444702",
    "b3_e4_bl5_matched_reference_actions.py": "a42d4bdeb96c7106f296e92129104eb896fbc7cb2904d9867bc61c7a9ca169c6",
    "b3_e4_f03_bl5_opensees_line_reference.py": "fe654b7a7c384620dae65f0b0981bb18e512513c090abdfe0dbf6c62c43d15cc",
}
BINARY_SHA256 = "25ba42e1672526137d765d9c3631d1403e091f54f5afcdb9242f9243c63c3a96"
BASIS = "discontinuous_p1_cell_average"
BALANCE_LIMIT, GAP_LIMIT, DERIVATIVE_LIMIT = 1e-8, 1e-9, 1e-6
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
OPTIONAL_THREAD_VARIABLES = ("BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
SECTIONS = dict(E_PA=210000000000., NU=.3, G_PA=80769230769.23077,
    SHELL_THICKNESS_M=.035, BEAM_AREA_M2=.01, BEAM_TORSION_M4=1.6666666666666667e-5,
    BEAM_BENDING_M4=8.333333333333334e-6, BEAM_SHEAR_AREA_M2=.008)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbid_product_import():
    if any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in sys.modules):
        raise ValueError("independent density reference must not import product modules")


def services():
    """Verify all six unchanged sources before loading only generic services."""
    forbid_product_import()
    directory = Path(__file__).resolve().parent
    expected = {AFFINE_SOURCE: AFFINE_SHA256, **HELPER_SHA256}
    actual = {name: sha(directory/name) for name in expected}
    if actual != expected:
        raise ValueError("density reference source identity changed: "+str(actual))
    spec = importlib.util.spec_from_file_location("_fixed_density_reference_services", directory/AFFINE_SOURCE)
    affine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(affine)
    if (affine.HELPER_SHA256 != HELPER_SHA256 or affine.PROJECTED_BASIS != BASIS or
            (affine.BALANCE_LIMIT, affine.GAP_LIMIT, affine.DERIVATIVE_LIMIT) !=
            (BALANCE_LIMIT, GAP_LIMIT, DERIVATIVE_LIMIT)):
        raise ValueError("unchanged projected basis and physical budgets required")
    loaded = affine.services()
    if loaded[3].BINARY_SHA256 != BINARY_SHA256:
        raise ValueError("unchanged verified external DLL identity required")
    forbid_product_import()
    return affine, *loaded


def _support_count(value):
    if type(value) is not int or value not in (2, 4):
        raise ValueError("plain integer fixed S4 or S2 density required")
    return value


def _reference(reference):
    if ({name: getattr(reference, name) for name in SECTIONS} != SECTIONS or
            reference.WIDTH_M != .2 or reference.BEAM_AXIS_Z_M != .0675 or
            reference.BEAM_BOTTOM_ARM_M != -.05):
        raise ValueError("exact fixed physical sections/width/beam axis/offset required")


def geometry(reference, support_count, refinement=None):
    support_count = _support_count(support_count)
    _reference(reference)
    if refinement is None:
        refinement = services()[1]
    model = refinement.geometry(64, reference)
    first, nb = refinement.BEAM_FIRST, 16
    beam_ids = tuple(first+i for i in range(nb+1))
    nodes = {n: model["nodes"][n] for n in model["shell_node_ids"]}
    if set(nodes) & set(beam_ids):
        raise ValueError("disjoint physical shell/beam namespaces required")
    nodes.update({n: (float(F(i, nb)), 0., reference.BEAM_AXIS_Z_M) for i, n in enumerate(beam_ids)})
    fixes = {n: model["fixes"][n] for n in model["root_shell_nodes"]}
    fixes[first] = (1, 1, 1, 1, 1, 0)
    model.update(beam_intervals=nb, support_count=support_count, nodes=nodes,
        beam_node_ids=beam_ids, beam_cells=tuple(zip(beam_ids, beam_ids[1:])),
        beam_root=first, beam_tip=beam_ids[-1], fixes=fixes,
        beam_element_tags=tuple(refinement.BEAM_ELEMENT_FIRST+i for i in range(nb)),
        physical_node_order=sorted(nodes), moment_origin_m=nodes[first],
        fixture_case="s"+str(support_count),
        density_policy="fixed h16 halfcell means; complete five-channel S4/H4 or nested S2/H8; development discriminator")
    _inventory(model, reference)
    return model


def _inventory(model, reference):
    _reference(reference)
    keys = ("nx", "ny", "beam_intervals", "support_count")
    if (any(type(model[key]) is not int for key in keys) or
            tuple(model[key] for key in keys) not in ((64, 16, 16, 4), (64, 16, 16, 2))):
        raise ValueError("only fixed64x16/h16/S4 or S2 physical tuple admitted")
    first = model["beam_root"]
    shell_ids = tuple(range(1, 1106))
    beam_ids = tuple(first+i for i in range(17))
    roots = tuple(1+65*j for j in range(17))
    expected_fixes = {n: (1, 1, 1, 1, 1, 1) for n in roots}
    expected_fixes[first] = (1, 1, 1, 1, 1, 0)
    shell_cells = tuple((1+65*j+i, 2+65*j+i, 2+65*(j+1)+i, 1+65*(j+1)+i)
                        for j in range(16) for i in range(64))
    if (type(first) is not int or first != 100001 or model["beam_tip"] != first+16 or
            tuple(model["shell_node_ids"]) != shell_ids or tuple(model["beam_node_ids"]) != beam_ids or
            set(model["nodes"]) != set(shell_ids) | set(beam_ids) or
            tuple(model["root_shell_nodes"]) != roots or model["fixes"] != expected_fixes or
            tuple(map(tuple, model["shell_cells"])) != shell_cells or
            tuple(map(tuple, model["beam_cells"])) != tuple(zip(beam_ids, beam_ids[1:])) or
            tuple(model["beam_element_tags"]) != tuple(range(100001, 100017)) or
            model["physical_node_order"] != sorted(model["nodes"]) or
            tuple(model["moment_origin_m"]) != (0., 0., .0675)):
        raise ValueError("complete fixed node/element/ground inventory required")
    if (any(tuple(model["nodes"][1+65*j+i]) != (i/64, reference.WIDTH_M*j/16, 0.)
            for j in range(17) for i in range(65)) or
            any(tuple(model["nodes"][first+i]) != (float(F(i, 16)), 0., .0675) for i in range(17))):
        raise ValueError("unchanged exact physical shell/beam coordinates required")
    if sum(sum(flags) for flags in model["fixes"].values()) != 107:
        raise ValueError("107 prescribed physical coordinates required")


def aggregate_rows(model, reference, affine=None):
    """Exact rational independent action, with a new fixed tuple guard only.

    Support-major local channel/degree rows are required by unchanged recovery;
    The permutation only reorders external rows. External rx/ry channels and
    gap signs do not directly identify the native projected director channels.
    """
    _inventory(model, reference)
    if affine is None:
        affine = services()[0]
    nx, nb, ns, first = 64, 16, model["support_count"], model["beam_root"]
    rb, rs, hb, stride = F(reference.BEAM_BOTTOM_ARM_M), F(reference.SHELL_THICKNESS_M)/2, F(1, 16), 16//ns
    physical, emitted, supports, all_dependent, exact_blocks = [], [], [], [], []
    for index in range(ns):
        left, right = F(index, ns), F(index+1, ns)
        bw, bp = affine.integrated_projected_weights(left, right, nb, nb, first)
        sw, sp = affine.integrated_projected_weights(left, right, nx, nb, 1)
        block = [{} for _ in range(10)]
        for weights, sign, arm in ((bw, F(1), rb), (sw, F(-1), rs)):
            for degree, nodes in enumerate(weights):
                for n, weight in nodes.items():
                    for d in range(5):
                        block[2*d+degree][n, d+1] = sign*weight
                    block[degree][n, 5] = sign*arm*weight
                    block[2+degree][n, 4] = -sign*arm*weight
        quarter = (first+stride*index+stride//4, first+stride*index+3*stride//4)
        if any((bw[0][n], bw[1][n]) != (hb, sign*hb/2) for n, sign in zip(quarter, (-1, 1))):
            raise ValueError("exact quarter-station shape integrals required")
        dependent = [(n, d) for n in quarter for d in range(1, 6)]
        if any(model["fixes"].get(n, (0,)*6)[d-1] for n, d in dependent):
            raise ValueError("interior dependent physical coordinates must be free")
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
        supports.append(dict(support_id=index, basis=BASIS, interval_m=[float(left), float(right)],
            width_m=float(right-left), row_indices=list(range(10*index, 10*index+10)),
            beam_spacing_m=float(hb), physical_row_order="channel0..4, degree0/1 per support",
            beam_shape_integrals_m=[{n: float(v) for n, v in weights.items()} for weights in bw],
            shell_shape_integrals_m=[{n: float(v) for n, v in weights.items()} for weights in sw],
            beam_pieces=bp, shell_pieces=sp, recovery_beam_nodes=list(quarter),
            dependent_node_dofs=dependent, dependent_coefficient_block_D=[[float(v) for v in row] for row in dmatrix],
            emitted_from_physical_E=[[float(v) for v in row] for row in e],
            exact_E_rationals=[[str(v) for v in row] for row in e]))
        all_dependent.extend(dependent)
        exact_blocks.append(block)
    if len(set(all_dependent)) != 10*ns:
        raise ValueError("complete distinct dependent physical coordinate inventory required")
    for index, block in enumerate(exact_blocks):
        other = all_dependent[:10*index]+all_dependent[10*index+10:]
        if any(row.get(key, 0) for row in block for key in other):
            raise ValueError("foreign support dependent columns must be absent")
    return dict(basis=BASIS, physical_rows=physical, emitted_rows=emitted, supports=supports,
        row_labels=[[s, c, m] for s in range(ns) for c in range(5) for m in range(2)],
        external_channels=["translation_x", "translation_y", "translation_z", "rotation_x", "rotation_y"],
        external_channel_major_from_support_major=[10*s+2*c+m for c in range(5) for s in range(ns) for m in range(2)],
        row_correspondence="External ordering only; no direct native-row identification. External translation gaps oppose native gap signs; external physical rx/ry are not native director basis components.",
        exact_physical_row_coefficients=[[[n, d, str(v)] for (n, d), v in sorted(row.items()) if v]
                                         for block in exact_blocks for row in block])


def prepare(support_count):
    """No OpenSees import or solve; fresh owned inputs for one admitted density."""
    support_count = _support_count(support_count)
    affine, refinement, _, _, _, reference = services()
    model = geometry(reference, support_count, refinement)
    pairing = aggregate_rows(model, reference, affine)
    load = refinement.LOADS["mz"]
    if tuple(load) != (0., 0., 0., 0., 0., 10.):
        raise ValueError("only fixed tip Mz10Nm load admitted")
    commands = refinement.command_recipe(model, reference, pairing["emitted_rows"], "mz")
    expected = refinement.planned_inventory(commands)
    if (len(expected["nodes"]), len(expected["elements"]), len(expected["equations"]),
            sum(sum(flags) for flags in expected["fixes"].values())) != (1122, 1040, 10*support_count, 107):
        raise ValueError("complete1122-node/1040-element/107-fixed physical command inventory required")
    if expected["loads"] != {model["beam_tip"]: list(load)}:
        raise ValueError("one dead tip Mz10Nm load and no other loads required")
    forbid_product_import()
    return dict(model=model, **pairing, load=list(load), command_recipe=commands,
                sections=dict(SECTIONS), expected_inventory=expected)


def write_summary(output, summary):
    temporary = output/("summary."+str(os.getpid())+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, output/"summary.json")


def finalize(recorder, service, affine, output, summary):
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
            pass  # Preserve the already propagating primary/cancellation error.


def _affinity():
    import ctypes
    if os.name != "nt":
        raise ValueError("verified Windows external runtime required")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
    kernel.GetProcessAffinityMask.restype = ctypes.c_int
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(process_mask=process.value, system_mask=system.value, processor_count=process.value.bit_count())


def main(site, output, support_count, case="mz"):
    support_count = _support_count(support_count)
    if case != "mz":
        raise ValueError("one fixed Mz10Nm reference load required")
    output.mkdir(parents=True, exist_ok=False)
    recorder, service, affine = None, None, None
    summary = dict(status="failed", identity=IDENTITY, case=case, basis=BASIS,
        fixture=dict(nx=64, ny=16, halfcells=16, supports=support_count), fixture_case="s"+str(support_count),
        nx=64, ny=16, beam_intervals=16, supports=support_count, rows=10*support_count,
        source_sha256=sha(__file__), affine_source={AFFINE_SOURCE: AFFINE_SHA256}, helper_sources=HELPER_SHA256,
        classification="EXPERIMENTAL_NOT_QUALIFIED; supplementary same-geometry density discriminator only; no accuracy/convergence/policy/M2/F03 acceptance",
        formal_f03_authority_consumed=False, python=sys.version,
        requested_threads={name: os.environ.get(name) for name in THREAD_VARIABLES+OPTIONAL_THREAD_VARIABLES},
        thread_scope="one-thread environment and affinity1 required; effective internal OpenSees pools remain supervisor-owned/unresolved; no resource-qualified claim")
    try:
        affine, refinement, p0, frozen, service, reference = services()
        if (any(os.environ.get(name) != "1" for name in THREAD_VARIABLES) or
                any(os.environ.get(name) not in (None, "1") for name in OPTIONAL_THREAD_VARIABLES)):
            raise ValueError("one numerical thread environment required")
        if sys.version_info[:2] != (3, 12):
            raise ValueError("verified external OpenSees binary requires Python3.12")
        summary["affinity"] = _affinity()
        if summary["affinity"]["processor_count"] != 1:
            raise ValueError("supervised single-processor affinity required")
        prepared = prepare(support_count)
        model, commands, expected = prepared["model"], prepared["command_recipe"], prepared["expected_inventory"]
        pairing = {key: prepared[key] for key in ("basis", "physical_rows", "emitted_rows", "supports")}
        service._atomic(output/"inputs.json", prepared)
        controls = affine.preparation_controls(model, pairing, refinement, p0, frozen, service, reference)
        service._atomic(output/"preparation-controls.json", controls)
        values = controls["six_common_rigid_field_average_gap_scaled"]+list(controls["objective_average_gap_scaled"].values())
        if (not controls["normal_spin_columns_zero"] or any(not math.isfinite(v) or v > GAP_LIMIT for v in values)
                or any(not math.isfinite(v) or v > DERIVATIVE_LIMIT for v in controls["actual_left_exp_derivative_relative"])
                or not math.isfinite(controls["row_transform_max_abs"]) or controls["row_transform_max_abs"] > BALANCE_LIMIT):
            raise ValueError("unchanged rigid/objectivity/actual-rotation derivative controls failed")
        resolved_site = site.resolve(strict=True)
        summary["site"] = str(resolved_site)
        sys.path.insert(0, str(resolved_site))
        import openseespy.opensees as ops
        from openseespywin import opensees as core
        forbid_product_import()
        binary = Path(core.__file__).resolve()
        summary.update(opensees=ops.version(), binary=str(binary), binary_sha256=sha(binary), expected_binary_sha256=BINARY_SHA256)
        if (not summary["opensees"].startswith("3.8") or summary["binary_sha256"] != BINARY_SHA256 or
                not binary.is_relative_to(resolved_site)):
            raise ValueError("verified OpenSees3.8 DLL identity within requested site required")
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
        summary.update(status="completed", physical_nodes=1122, shell_elements=1024, beam_elements=16,
            fixed_coordinates=107, applied_work_J=derived["work_J"]["applied"], checks=derived["checks"])
    except BaseException as error:
        summary["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        finalize(recorder, service, affine, output, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--supports", required=True, type=int, choices=(4, 2))
    parser.add_argument("--case", required=True, choices=("mz",))
    args = parser.parse_args()
    main(args.site, args.output_dir, args.supports, args.case)
