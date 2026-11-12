"""Retain one frozen native 16x4/32-halfcell/c2 Fy or Mz comparison.

No new mesh, mechanics or convergence gate. The matched external strong nodal
reference and native weak retained-director operator are different discretizations.
All raw poses/retained variables/covectors survive large response discrepancies.
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
import tempfile
import traceback
import zipfile

import numpy as np
import numba
from threadpoolctl import threadpool_info
import anyfesolver
from anyfesolver import stiffened_shell
from b3_e4_bl5_coherent_aggregation_dev import model, physical_transfer


LOADS = {"fy": (0., 100., 0., 0., 0., 0.), "mz": (0., 0., 0., 0., 0., 10.)}
LREF_M = 1.
WHEEL_SHA = "0c45666609355e654cb726a0e4e0780d386bccfaafef1209ffd994bdd3b9d09c"


class NativeReferenceContractError(ValueError):
    """Inputs or retained evidence do not describe the declared comparison."""


class NativePhysicalCheckFailure(RuntimeError):
    """Native physical/recovery criteria failed; discrepancy is not this error."""


def require(condition, message):
    if not condition:
        raise NativeReferenceContractError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_services():
    path = Path(__file__).with_name("b3_e4_bl5_matched_mz_reference_dev.py")
    spec = importlib.util.spec_from_file_location("_matched_reference_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    service, service_path, reference, reference_path = module.services()
    return module, path, service, service_path, reference, reference_path


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def save_npz(path, arrays):
    descriptor, temporary = tempfile.mkstemp(prefix=".native-raw-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **{name: np.array(value, copy=True) for name, value in arrays.items()})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def external_contract(directory, case, module, path, service, service_path, reference, reference_path):
    files = {name: directory/(name+".json") for name in
             ("summary", "inputs", "controls", "raw-inventory", "recovered-actions")}
    data = {name: json.loads(path.read_text()) for name, path in files.items()}
    summary, inputs = data["summary"], data["inputs"]
    require(summary["status"] == "completed" and summary["case"] == case,
            "completed declared external case required")
    require(summary["source_sha256"] == sha(path)
            and summary["service_sha256"] == sha(service_path)
            and summary["reference_source_sha256"] == sha(reference_path),
            "external source/services changed since execution")
    require(summary["binary_sha256"] == service.BINARY_SHA256, "declared external binary required")
    require(inputs["case"] == case and inputs["load"] == list(LOADS[case]), "external load mismatch")
    expected = module.geometry(reference)
    require(inputs["model"] == plain(expected), "external model inventory mismatch")
    require(inputs["constraint_rows"] == module.equations(expected, reference),
            "external five-channel rows mismatch")
    require(data["raw-inventory"]["nodes"] == plain(expected["nodes"])
            and len(data["raw-inventory"]["poses"]) == 118,
            "complete external physical pose inventory required")
    commands = data["raw-inventory"]["model_commands"]
    require(["geomTransf", ["Linear", 1, 0., 0., 1.], {}] in commands,
            "external global-y section frame command required")
    for name, value in data["recovered-actions"]["checks"].items():
        limit = 1e-9 if name == "linear_five_channel_gap_scaled" else 1e-8
        require(isinstance(value, (int, float)) and math.isfinite(value) and value <= limit,
                "external failed physical check: "+name)
    data["source_binding"] = {name: sha(file) for name, file in files.items()}
    return data, expected


def match(owner, definition, policy, external, expected, reference):
    require(tuple(owner._shell_ids) == tuple(range(1, 86))
            and tuple(owner._beam_ids) == tuple(range(101, 134)), "native physical node order mismatch")
    require(owner._C.shape == (40, 1092) and policy["span_halfcells"] == 4
            and len(owner._beam_parts) == 16, "native c2/retained dimensions mismatch")
    xyz = np.vstack((owner._shell_xyz, [owner._beam_nodes[n][0] for n in owner._beam_ids]))
    external_ids = list(owner._shell_ids)+[n+900 for n in owner._beam_ids]
    expected_xyz = np.array([expected["nodes"][n] for n in external_ids])
    require(np.allclose(xyz, expected_xyz, rtol=0., atol=1e-13), "physical coordinates mismatch")
    fixed = {(f.domain, f.node, f.component) for f in definition.fixed}
    expected_fixed = {("shell" if n < 1001 else "beam", n if n < 1001 else n-900, d)
                      for n, flags in expected["fixes"].items() for d, flag in enumerate(flags) if flag}
    require(fixed == expected_fixed, "root support/free spin mismatch")
    sec = external["inputs"]["sections"]
    require(definition.shell.thickness_m == sec["SHELL_THICKNESS_M"]
            and definition.shell.young_pa == sec["E_PA"]
            and definition.shell.poisson == sec["NU"]
            and tuple(definition.shell.q4_elements) == tuple(expected["shell_cells"]), "shell law/topology mismatch")
    elastic = np.diag((sec["E_PA"]*sec["BEAM_AREA_M2"], sec["G_PA"]*sec["BEAM_SHEAR_AREA_M2"],
                       sec["G_PA"]*sec["BEAM_SHEAR_AREA_M2"], sec["G_PA"]*sec["BEAM_TORSION_M4"],
                       sec["E_PA"]*sec["BEAM_BENDING_M4"], sec["E_PA"]*sec["BEAM_BENDING_M4"]))
    for beam in definition.beams:
        element, _ = beam.instantiate()
        require(np.array_equal(element.section.elastic, elastic), "beam generalized stiffness mismatch")
    frame, arm = np.eye(3), np.array((0., 0., reference.BEAM_BOTTOM_ARM_M))
    records = []
    for local, slots, weight, _, _ in owner._line_samples:
        require(np.allclose(local.port.frame, frame, rtol=0., atol=1e-13)
                and np.allclose(local.reference_director, (0., 0., 1.), rtol=0., atol=1e-13)
                and np.allclose(local.director_basis, ((1., 0., 0.), (0., 1., 0.)), rtol=0., atol=1e-13)
                and np.allclose(local.reference_arm, arm, rtol=0., atol=1e-13)
                and np.linalg.norm(local.port.lift) <= 1e-13
                and local.thickness == definition.shell.thickness_m,
                "native line frame/director/arm/lift mismatch")
        records.append(dict(shell_reference=local.reference, shell_alpha=local.station,
                            beam_weight=local.port.weight, frame=local.port.frame,
                            arm=local.reference_arm, director=local.reference_director,
                            director_basis=local.director_basis,
                            slots=slots, measure_m=weight))
    require(np.allclose([owner._beam_nodes[n][1] for n in owner._beam_ids], frame,
                        rtol=0., atol=1e-13), "beam reference frames mismatch")
    return xyz, external_ids, records


def rigid_maps(owner, xyz):
    """Domain wrench work includes retained cell rotations, not only nodal slots."""
    maps = {name: np.zeros((6, owner._K.shape[0])) for name in ("shell", "beam")}
    count = len(owner._shell_ids)+len(owner._beam_ids)
    origin = xyz[len(owner._shell_ids)]
    for index in range(count):
        domain = "shell" if index < len(owner._shell_ids) else "beam"
        slot = 6*index
        maps[domain][:3, slot:slot+3] = np.eye(3)
        for axis in range(3):
            omega = np.eye(3)[axis]
            maps[domain][axis+3, slot:slot+3] = np.cross(omega, xyz[index]-origin)
            maps[domain][axis+3, slot+3:slot+6] = omega
    retained = 6*count
    for member in range(len(owner._beam_parts)):
        for cell in range(2):
            maps["beam"][3:, retained+24*member+3*cell:retained+24*member+3*cell+3] = np.eye(3)
    return maps


def windows(transfer):
    # P0 straight/reference-frame wrench is affine on every native halfcell.
    # Three-point integration exactly integrates the quartic smooth window
    # times this affine wrench, unlike reusing the two native Gauss samples.
    records = transfer["station_records"]
    require(len(records) == 64, "complete two-Gauss native halfcell inventory required")
    smooth = np.zeros(6)
    for edge in range(32):
        a, b = records[2*edge:2*edge+2]
        left, right = edge/32, (edge+1)/32
        require(left < a["x_m"] < b["x_m"] < right
                and abs(a["measure_m"]-(right-left)/2) < 1e-13
                and abs(b["measure_m"]-(right-left)/2) < 1e-13, "native quadrature mapping mismatch")
        fa, fb = np.array(a["beam_wrench_about_root_density"]), np.array(b["beam_wrench_about_root_density"])
        slope = (fb-fa)/(b["x_m"]-a["x_m"])
        require(np.allclose(slope, np.r_[np.zeros(3), np.cross((1., 0., 0.), fa[:3])],
                            rtol=1e-10, atol=1e-8), "non-affine reference P0 wrench")
        if left >= .25 and right <= .75:
            for abscissa, weight in ((-math.sqrt(3/5), 5/9), (0., 8/9), (math.sqrt(3/5), 5/9)):
                x = (left+right)/2+(right-left)*abscissa/2
                smooth += weight*(right-left)/2*256*(x-.25)**2*(.75-x)**2*(fa+slope*(x-a["x_m"]))
    closed = transfer["fixed_region_025_075_wrench"]
    return {"open": closed, "closed": closed, "endpoint_atoms": [0.]*6,
            "smooth_unit_peak": smooth.tolist()}


def field_errors(native, external, cells, coordinates, beam=False):
    """Exact two-Gauss L2 of common nodal P1/Q1 fields; no cell-rotation identification."""
    errors, references = np.zeros(6), np.zeros(6)
    for cell in cells:
        ids = list(cell)
        local_n, local_e, xyz = native[ids], external[ids], coordinates[ids]
        points = ((-.5773502691896258, .5773502691896258) if beam else
                  ((-.5773502691896258, -.5773502691896258),
                   (-.5773502691896258, .5773502691896258),
                   (.5773502691896258, -.5773502691896258),
                   (.5773502691896258, .5773502691896258)))
        for point in points:
            if beam:
                weights = np.array(((1-point)/2, (1+point)/2))
                measure = np.linalg.norm(xyz[1]-xyz[0])/2
            else:
                x, y = point
                weights = np.array(((1-x)*(1-y), (1+x)*(1-y), (1+x)*(1+y), (1-x)*(1+y)))/4
                measure = np.linalg.norm(np.cross(xyz[1]-xyz[0], xyz[3]-xyz[0]))/4
            difference, value = weights@(local_n-local_e), weights@local_e
            errors += measure*difference*difference
            references += measure*value*value
    scaled_error = math.fsum(errors[:3])+LREF_M**2*math.fsum(errors[3:])
    scaled_reference = math.fsum(references[:3])+LREF_M**2*math.fsum(references[3:])
    return dict(component_error_squared=errors.tolist(), component_reference_squared=references.tolist(),
                translation_relative_l2=math.sqrt(math.fsum(errors[:3])/max(math.fsum(references[:3]), 1e-24)),
                rotation_relative_l2=math.sqrt(math.fsum(errors[3:])/max(math.fsum(references[3:]), 1e-24)),
                combined_1m_relative_l2=math.sqrt(scaled_error/max(scaled_reference, 1e-24)),
                denominator_floor=1e-24, component_units="translations m; rotations rad; shell measure m2/beam measure m")


def comparison(owner, solved, q, load, xyz, external_ids, definition, transfer, external):
    count = len(external_ids)
    native = q[:6*count].reshape(count, 6)
    other = np.array([external["raw-inventory"]["poses"][str(n)] for n in external_ids])
    require(other.shape == native.shape and np.isfinite(other).all(), "finite six-coordinate external poses required")
    index = {n: i for i, n in enumerate(owner._shell_ids)}
    shell_cells = [tuple(index[n] for n in cell) for cell in definition.shell.q4_elements]
    beam_cells = [(i, i+1) for i in range(32)]
    ns = len(owner._shell_ids)
    fields = {"shell": field_errors(native[:ns], other[:ns], shell_cells, xyz[:ns]),
              "beam": field_errors(native[ns:], other[ns:], beam_cells, xyz[ns:], True)}
    work = float(q@load)
    external_work = external["recovered-actions"]["work_j"]["applied"]
    reference = external["recovered-actions"]
    native_windows = windows(transfer)
    differences = {name: (np.array(value)-np.array(reference["beam_connection_fixed_region_025_075"][name])).tolist()
                   for name, value in native_windows.items()}
    return dict(nodal_field_errors=fields, native_work_j=work, external_work_j=external_work,
                work_relative_signed=(work-external_work)/max(abs(external_work), 1e-12),
                polynomial_transfer_difference=(np.array(transfer["polynomial_transfer_moments"])
                    -np.array([reference["beam_connection_polynomial_wrench_moments"][str(p)] for p in range(3)])).tolist(),
                native_fixed_region=native_windows, fixed_region_difference=differences,
                fixed_region_convention="Native continuous P0 density has no endpoint atoms. External open/closed/endpoint and smooth windows all retained. No chosen endpoint convention hides another.",
                scope="Common Q1 shell/P1 beam nodal displacement and infinitesimal orientation fields. Native retained cell orientation is separate; no weak/strong discrete equality or converged1% claim.")


def main(output, directory, site, case):
    output.mkdir(parents=True, exist_ok=False)
    module, ref_path, service, service_path, reference, reference_path = reference_services()
    write = lambda name, value: service._atomic(output/(name+".json"), plain(value))
    owner, original_error = None, None
    summary = dict(status="failed", classification="fixed-mesh physical-reference development comparison",
                   case=case, source_sha256=sha(__file__), expected_wheel_sha256=WHEEL_SHA,
                   limits="No converged1% accuracy, locking, M2/F03 acceptance or equal discrete operators")
    try:
        site = site.resolve(strict=True)
        imports = {n: str(Path(m.__file__).resolve()) for n, m in sys.modules.copy().items()
                   if n.split('.')[0] == "anyfesolver" and getattr(m, "__file__", None)}
        require(all(Path(p).is_relative_to(site) for p in imports.values()), "installed product imports required")
        identity_path = site.parent/"identity.json"
        installed = json.loads(identity_path.read_text())
        require(installed["wheel_sha256"] == WHEEL_SHA and installed["matching_modules"] == 388,
                "declared unchanged installed wheel required")
        wheel = next((site.parent/"wheels").glob("*.whl"))
        require(sha(wheel) == WHEEL_SHA, "installed wheel bytes changed")
        with zipfile.ZipFile(wheel) as archive:
            modules = [name for name in archive.namelist() if name.startswith("anyfesolver/") and name.endswith(".py")]
            require(len(modules) == 388 and all(archive.read(name) == (site/name).read_bytes() for name in modules),
                    "complete installed module/wheel byte binding required")
        fixture = Path(sys.modules["test_stiffened_shell_l1_assembly_experimental"].__file__).resolve()
        require(sha(fixture) == installed["fixtures_sha256"][fixture.name],
                "unchanged installed-campaign model fixture required")
        pools = threadpool_info()
        require(pools and all(p["num_threads"] == 1 for p in pools)
                and numba.config.DISABLE_JIT and numba.config.NUMBA_NUM_THREADS == 1,
                "one effective BLAS/Numba thread and disabled JIT required")
        summary.update(python=sys.version, pools=pools, jit_disabled=True, numba_threads=1,
                       model_fixture_path=str(fixture), model_fixture_sha256=sha(fixture),
                       identity_sha256=sha(identity_path), helper_sha256={p: sha(Path(__file__).with_name(p)) for p in
                       ("b3_e4_bl5_coherent_aggregation_dev.py", ref_path.name, service_path.name, reference_path.name)})
        external, expected = external_contract(directory, case, module, ref_path, service, service_path, reference, reference_path)
        summary["external_thread_scope"] = external["summary"]["thread_scope"]
        definition, policy = model(16, 4, 32, 2)
        owner = stiffened_shell.create_analysis(definition=definition)
        xyz, external_ids, line_frames = match(owner, definition, policy, external, expected, reference)
        write("inputs", dict(case=case, load=LOADS[case], policy=policy, preflight=owner.preflight(),
            external_binding=external["source_binding"], external_ids=external_ids,
            native_ids=list(owner._shell_ids)+list(owner._beam_ids), xyz_m=xyz,
            fixed=[(f.domain, f.node, f.component) for f in definition.fixed], line=definition.lines[0].descriptor(),
            beam_descriptors=[json.loads(b.raw) for b in definition.beams], line_frame_slots=line_frames))
        result = owner.solve_linear((stiffened_shell.NodalAction("beam", 133, LOADS[case]),))
        q, load = np.r_[result.shell_displacement, result.beam_coordinates], np.array(owner._accepted[1], copy=True)
        g, ct = owner._K@q, owner._C.T@result.multipliers
        arrays = dict(q=q, load=load, shell_poses=result.shell_displacement.reshape(85, 6),
            beam_nodal_poses=result.beam_coordinates[:198].reshape(33, 6),
            beam_retained=result.beam_coordinates[198:].reshape(16, 24), multipliers=result.multipliers,
            reactions=result.reactions, Kq=g, constraint_resisting=ct, connection_exerted=-ct,
            weak_gaps=owner._C@q, xyz_m=xyz, external_ids=external_ids,
            native_ids=list(owner._shell_ids)+list(owner._beam_ids),
            C_data=owner._C.data, C_indices=owner._C.indices, C_indptr=owner._C.indptr,
            C_shape=owner._C.shape, fixed=owner._fixed, free=owner._free,
            coordinate_scale=owner._coordinate_scale, constraint_scale=owner._constraint_scale)
        save_npz(output/"raw-state.npz", arrays)
        write("raw-state", dict(status=result.status, generation=result.generation, arrays_sha256=sha(output/"raw-state.npz"),
            counts=dict(shell_nodes=85, beam_nodes=33, retained_members=16, retained_cell_rotations=96,
                        retained_resultants=288, multipliers=40, full_q=1092),
            role="First708 physical nodal coordinates;16 retained24 blocks=two3D cell rotations+18 resultants. Full mixedK is not SPD."))
        transfer = physical_transfer(owner, result)
        write("physical-transfer", transfer)
        maps = rigid_maps(owner, xyz)
        internal, exerted, ground, applied = ({domain: matrix@vector for domain, matrix in maps.items()}
            for vector in (g, -ct, result.reactions, load))
        moment0 = np.array(transfer["polynomial_transfer_moments"][0])
        cuts = {domain: applied[domain]+exerted[domain]+ground[domain] for domain in maps}
        scale = max(map(abs, LOADS[case]))
        work = float(q@load)
        checks = dict(equilibrium_ratio=result.free_equilibrium_ratio, weak_gap=result.constraint_max,
            energy_work_relative=abs(float(q@g)-work)/max(abs(work), 1e-12),
            beam_physical_workmap_scaled=float(np.max(np.abs(exerted["beam"]-moment0)))/scale,
            shell_physical_workmap_scaled=float(np.max(np.abs(exerted["shell"]+moment0)))/scale,
            complete_domain_cuts_scaled=max(float(np.max(np.abs(v)))/scale for v in cuts.values()),
            global_ground_applied_scaled=float(np.max(np.abs(sum(ground.values())+sum(applied.values()))))/scale)
        compared = comparison(owner, result, q, load, xyz, external_ids, definition, transfer, external)
        compared.update(native_domain_wrenches=dict(internal=internal, connection_exerted=exerted,
            ground=ground, applied=applied, complete_cuts=cuts),
            wrench_origin_m=xyz[len(owner._shell_ids)],
            wrench_map_provenance="Full domain rigid virtual work includes native retained cell rotations; beam nodal-force sum alone omits cell torque covectors.",
            external_domain_cuts=external["recovered-actions"]["complete_domain_cuts"],
            domain_ground_wrench_difference={domain: (ground[domain]-np.array(external["recovered-actions"][
                domain+"_ground_wrench_about_beam_root"])).tolist() for domain in maps},
            ground_wrench_difference=(sum(ground.values())-np.array(external["recovered-actions"]["ground_wrench_about_beam_root"])).tolist(),
            physical_checks=checks, preflight_after=owner.preflight())
        write("comparison", compared)  # Preserve large discrepancies before checks.
        for name, value in checks.items():
            limit = 1e-9 if name == "weak_gap" else 1e-8
            if not math.isfinite(value) or value > limit:
                raise NativePhysicalCheckFailure(f"{name}={value} > {limit}")
        require(owner.preflight()["reference_linear_factorizations"] == 1, "one cached factor required")
        summary.update(status="completed", work_relative_signed=compared["work_relative_signed"],
                       field_errors=compared["nodal_field_errors"], checks=checks)
        final_imports = {n: Path(m.__file__).resolve() for n, m in sys.modules.copy().items()
                        if n.split('.')[0] == "anyfesolver" and getattr(m, "__file__", None)}
        require(all(path.is_relative_to(site) for path in final_imports.values()),
                "all final product imports must remain on the canonical installed site")
    except BaseException as error:
        original_error = error
        summary["status"] = "failed"
        summary["failure"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        imports = {n: str(Path(m.__file__).resolve()) for n, m in sys.modules.copy().items()
                   if n.split('.')[0] == "anyfesolver" and getattr(m, "__file__", None)}
        cleanup_error = None
        actions = [("retain_imports", lambda: write("imports", imports))]
        if owner is not None:
            actions.append(("close", owner.close))
        for name, action in actions:
            try:
                action()
            except BaseException as error:
                cleanup_error = error
                summary["status"] = "failed"
                summary.setdefault("retention_cleanup_failures", []).append(
                    dict(operation=name, type=type(error).__name__, message=str(error)))
        try:
            write("summary", summary)
        except BaseException:
            if original_error is None:
                raise
            traceback.print_exc()
        if cleanup_error is not None and original_error is None:
            raise cleanup_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--expected-site", type=Path, required=True)
    parser.add_argument("--case", choices=("fy", "mz"), required=True)
    args = parser.parse_args()
    main(args.output_dir, args.external_dir, args.expected_site, args.case)
