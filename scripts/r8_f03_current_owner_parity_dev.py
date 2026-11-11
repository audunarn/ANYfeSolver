"""Bounded current-owner c2/P0 source and work parity; no F03 quotient."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/agent_plans/development/r8-f03-current-owner-parity-20261009"
FAMILY = ROOT / "docs/agent_plans/development/r8-bl5-native-stability-20260930/r8-bl5-native-stability-family-v1"
AUDIT = ROOT / "docs/agent_plans/development/r8-f03-actual-family-20261009/reuse-audit-20261009.md"
PYTHON = Path(r"C:\Python\Python314\python.exe")
SITE = Path(r"C:\Python\Python314\Lib\site-packages")
ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1", "NUMBA_DISABLE_JIT": "1"}
LEVELS = {32: (32, 8, 32), 64: (64, 16, 64)}
MAX_WALL = 90.0
MAX_RSS = 2*1024**3
OLD_PRODUCT = "855610d94e65570d360f6716d2fda447b54bf186"
FAMILY_SHA = "8af69fa6448e864c6c8c6bb9eddd9d29bfe34295e3e7e78de2c45c7ff8efa0e4"
REFERENCE_FUNCTIONS = {
    "b3_e4_bl5_reference_actions_dev.py": ("spatial_action", "integrated_p1",
                                             "physical_loads", "native_slots"),
    "b3_e4_bl5_riesz_controls_dev.py": ("beam_norm", "shell_scalar"),
}
SOURCE_FILES = ("scripts/b3_e4_bl5_coherent_aggregation_dev.py",
                "scripts/b3_e4_bl5_reference_actions_dev.py",
                "scripts/b3_e4_bl5_riesz_controls_dev.py",
                "scripts/r8_bl5_actual_primal_work_dev.py",
                "scripts/r8_bl5_mixed_shell_h1_dev.py",
                "src/anyfesolver/_stiffened_shell_ge_experimental.py",
                "src/anyfesolver/_stiffened_shell_l1_assembly_experimental.py")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def campaign_folder(level: int, campaign: str) -> Path:
    require(level in LEVELS and re.fullmatch(r"v[0-9]{1,3}", campaign) is not None,
            "bounded versioned campaign name required")
    nx, ny, cells = LEVELS[level]
    return OUT / f"run-{campaign}-nx{nx}-ny{ny}-half{cells}"


def require_fresh_output(folder: Path) -> None:
    require(not folder.exists(), "fresh per-level diagnostic directory required")


def validate_prior_32(receipt: dict, outcome: dict, binding: dict,
                      campaign: str) -> float:
    wall = receipt.get("process_wall_seconds")
    require(receipt.get("campaign") == outcome.get("campaign") == campaign and
            receipt.get("status") == "completed" and receipt.get("exit_code") == 0 and
            outcome.get("status") == "current_owner_frozen_c2_work_parity_passed_development_only" and
            outcome.get("level") == 32 and
            (outcome.get("nx"), outcome.get("ny"), outcome.get("halfcells")) == LEVELS[32] and
            receipt.get("source_dependencies_sha256") == binding["current_sources_sha256"] and
            outcome.get("bound_inputs", {}).get("family_result_sha256") == FAMILY_SHA and
            isinstance(wall, (int, float)) and 0.0 < wall <= MAX_WALL,
            "same-campaign bound 32x8 pass required before 64x16")
    return float(wall)


def function_ast_hash(source: bytes, name: str) -> str:
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                 and node.name == name]
    require(len(functions) == 1, f"unique {name} source function required")
    return hashlib.sha256(ast.dump(functions[0], include_attributes=False).encode()).hexdigest()


def compare_reference_helpers(current: Path, frozen: Path, names: tuple[str, ...]) -> dict:
    require(current.is_file() and frozen.is_file(), "current/frozen reference helper missing")
    result = {}
    for name in names:
        actual = function_ast_hash(current.read_bytes(), name)
        expected = function_ast_hash(frozen.read_bytes(), name)
        require(actual == expected, f"frozen independent {name} construction changed")
        result[name] = actual
    return result


def check_frozen_case(nx: int, ny: int, cells: int) -> dict:
    folder = FAMILY / f"nx{nx}-ny{ny}-half{cells}"
    result_path, identity_path = folder / "result.json", folder / "identity.json"
    frozen = json.loads(result_path.read_text())
    seal = json.loads(identity_path.read_text())
    require(frozen["status"] == "COMPLETED_DEVELOPMENT_ONLY" and
            all(frozen["predicates"].values()) and
            (frozen["nx"], frozen["ny"], frozen["beam_halfcells"]) == (nx, ny, cells),
            "representative frozen positive case absent")
    require(sha(folder / "run_case.py") == frozen["source_sha256"] ==
            seal["frozen_sha256"]["run_case.py"] and
            sha(folder / "scalar.json") == frozen["frozen_scalar_sha256"] ==
            seal["frozen_sha256"]["scalar.json"],
            "frozen case source/scalar bytes changed")
    require(frozen["policy"]["c_lambda"] == 2 and
            frozen["policy"]["span_halfcells"] == 2 and
            frozen["constraint_rows"] == 5*cells//2,
            "frozen c2/P0 row policy changed")
    helper_hashes = {}
    for filename, names in REFERENCE_FUNCTIONS.items():
        helper = folder / "scripts" / filename
        require(sha(helper) == seal["frozen_sha256"][f"scripts/{filename}"],
                f"frozen {filename} bytes changed")
        helper_hashes[filename] = compare_reference_helpers(
            ROOT / "scripts" / filename, helper, names)
    return {"frozen_result_sha256": sha(result_path),
            "frozen_identity_sha256": sha(identity_path),
            "frozen_norm_rows": frozen["norm_rows"],
            "frozen_constraint_rows": frozen["constraint_rows"],
            "frozen_span": frozen["policy"]["span_halfcells"],
            "frozen_reference_function_ast_sha256": helper_hashes}


def preflight(level: int) -> dict:
    require(level in LEVELS, "only representative 32/64 levels allowed")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", "c64d189e", "HEAD"],
                              cwd=ROOT, capture_output=True)
    changed = subprocess.run(["git", "diff", "--name-only", "c64d189e", "HEAD", "--",
                              *SOURCE_FILES], cwd=ROOT, check=True, capture_output=True, text=True)
    require(ancestor.returncode == 0 and not changed.stdout.strip(),
            "current GE owner/source dependencies changed since c64d189e")
    require(sha(FAMILY / "family-result.json") == FAMILY_SHA and AUDIT.is_file(),
            "frozen family/audit identity changed")
    nx, ny, cells = LEVELS[level]
    old_owner = subprocess.run(["git", "show", f"{OLD_PRODUCT}:src/anyfesolver/_stiffened_shell_ge_experimental.py"],
                               cwd=ROOT, check=True, capture_output=True).stdout
    current_owner = (ROOT / "src/anyfesolver/_stiffened_shell_ge_experimental.py").read_bytes()
    require(function_ast_hash(old_owner, "_assemble_beam") ==
            function_ast_hash(current_owner, "_assemble_beam"),
            "native beam assembly changed from frozen product")
    # The changed owner and shell reference paths prevent source-only full
    # equality; the child compares the supported physical work directly.
    return {"status": "source_preflight_passed", "head": head, "level": level,
            "nx": nx, "ny": ny, "halfcells": cells,
            "frozen_product_commit": OLD_PRODUCT,
            "frozen_product_owner_sha256": hashlib.sha256(old_owner).hexdigest(),
            "current_sources_sha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
            "family_result_sha256": FAMILY_SHA, "reuse_audit_sha256": sha(AUDIT),
            "frozen_case": check_frozen_case(nx, ny, cells),
            "expected_span": 2, "expected_rows": 5*cells//2,
            "comparison_tolerances": {"integer_layout": "exact",
                                      "selector_and_omitted_action": "exact or <=1e-13",
                                      "H_and_reference_action": "1e-10 relative + 1e-12 original-unit floor"}}


def max_sparse_abs(matrix) -> float:
    copy = matrix.tocsr()
    copy.sum_duplicates()
    copy.eliminate_zeros()
    return float(max(abs(copy.data), default=0.0))


def signed_physical_basis_map(embedding, frozen_selector, np, sparse):
    """E_current P = E_frozen with one signed physical coordinate per column."""
    p = (embedding.T@frozen_selector).tocsc()
    require(p.shape == (embedding.shape[1], frozen_selector.shape[1]) and
            p.shape[0] == p.shape[1] and p.nnz == p.shape[0] and
            np.array_equal(abs(p.data), np.ones(p.nnz)) and
            np.array_equal(np.asarray(abs(p).sum(axis=0)).ravel(), np.ones(p.shape[1])) and
            np.array_equal(np.asarray(abs(p).sum(axis=1)).ravel(), np.ones(p.shape[0])) and
            max_sparse_abs(p.T@p-sparse.eye(p.shape[0], format="csc")) == 0.0 and
            max_sparse_abs(embedding@p-frozen_selector) == 0.0,
            "current E lacks exact signed/orthogonal frozen physical basis map")
    return p


def child(level: int, folder: Path, campaign: str) -> None:
    result = {"status": "failed", "classification": "current-owner mapping parity development only",
              "source_sha256": sha(Path(__file__)), "level": level, "campaign": campaign,
              "environment": {key: os.environ.get(key) for key in ENV}}
    try:
        require(result["environment"] == ENV, "one-thread/JIT-off environment changed")
        import numpy as np
        import scipy
        from scipy import sparse
        import numba
        import psutil
        from threadpoolctl import threadpool_info
        from anyfesolver import stiffened_shell as api
        from scripts.b3_e4_bl5_coherent_aggregation_dev import model
        from scripts.b3_e4_bl5_reference_actions_dev import native_slots, physical_loads
        from scripts.b3_e4_bl5_riesz_controls_dev import beam_norm, shell_scalar
        from scripts.r8_bl5_actual_primal_work_dev import actual_flat_primal_work
        pools = threadpool_info()
        require(psutil.Process().cpu_affinity() == [0] and pools and
                all(pool["num_threads"] == 1 for pool in pools) and
                bool(numba.config.DISABLE_JIT), "effective CPU0/BLAS1/JIT-off missing")
        result["runtime"] = {"affinity": [0], "blas_pools": pools,
                             "jit_disabled": True, "numpy": np.__version__,
                             "scipy": scipy.__version__}
        binding = preflight(level)
        result["bound_inputs"] = binding
        nx, ny, cells = LEVELS[level]
        definition, policy = model(nx, ny, cells, 2)
        require(definition.lines[0].basis == "p0" and
                policy["span_halfcells"] == binding["expected_span"] and
                cells//policy["span_halfcells"]*5 == binding["expected_rows"],
                "current c2/P0 support rule differs from frozen")
        owner = api.create_analysis(definition=definition)
        try:
            local = owner._line_samples[0][0]
            frame = local.port.frame
            director = frame@local.reference_director
            arm_b = frame@local.reference_arm
            arm_s = local.thickness*director/2
            require(np.max(abs(arm_b-np.array((0., 0., -.05)))) <= 1e-13 and
                    np.max(abs(arm_s-np.array((0., 0., .0175)))) <= 1e-13 and
                    np.max(abs(director-np.array((0., 0., 1.)))) <= 1e-13,
                    "frozen flat reference arm/director changed")
            h, embedding, row_work = actual_flat_primal_work(owner, .4, .1)
            shell_knots, _, scalar, scalar_active = shell_scalar(1., .2, .4, nx, ny)
            beam_knots, beam_h, beam_active, _ = beam_norm(1., .4, cells, 1, arm_b, .1)
            slots, absent, all_fields = native_slots(owner, scalar_active, beam_active)
            require(len(set(slots.tolist())) == len(slots) and
                    len(set(all_fields.tolist())) == len(all_fields),
                    "frozen physical/retained coordinate ordering duplicated")
            old_e = sparse.coo_matrix((np.ones(len(slots)),
                                       (slots, np.arange(len(slots)))),
                                      shape=(owner._K.shape[0], len(slots))).tocsc()
            p = signed_physical_basis_map(embedding, old_e, np, sparse)
            require(p.shape == (h.shape[0], len(slots)),
                    "current/frozen supported primal dimension changed")
            old_h = sparse.block_diag((scalar, scalar, scalar, .4**2*scalar,
                                       .4**2*scalar, beam_h), format="csc")
            require(old_h.shape == h.shape and h.shape[0] ==
                    binding["frozen_case"]["frozen_norm_rows"],
                    "current H dimension differs from frozen c2 construction")
            h_error = max_sparse_abs(p.T@h@p-old_h)
            h_scale = max(1., max_sparse_abs(old_h))
            require(h_error <= 1e-10*h_scale+1e-12,
                    "current physical H differs from frozen independent norm")
            work_old_order = (row_work@p).tocsc()
            ce_error = max_sparse_abs(work_old_order-owner._C@old_e)
            require(ce_error <= 1e-13 and row_work.shape[0] ==
                    binding["frozen_case"]["frozen_constraint_rows"],
                    "actual C E or P0 row ordering changed")
            omitted = owner._C[:, absent].copy().tocsr()
            omitted.eliminate_zeros()
            omitted_max = max_sparse_abs(omitted)
            require(omitted_max <= 1e-13,
                    "actual C reaches omitted retained/drilling roles")
            supports = cells//policy["span_halfcells"]
            density = np.diag(np.repeat(np.array((1., 1., 1., .4, .4)), supports))
            reference = [physical_loads(col.reshape(5, supports).T, frame, director,
                                        local.director_basis, arm_b, arm_s,
                                        beam_knots, shell_knots, ny, cells,
                                        policy["span_halfcells"])
                         for col in density.T]
            independent = np.column_stack([np.r_[shell, beam] for beam, shell, _ in reference])
            require(independent.shape == (len(all_fields), row_work.shape[0]),
                    "independent frozen physical reference action layout changed")
            actual_full = -np.asarray(owner._C[:, all_fields].T@density)
            reference_error = float(np.max(abs(independent-actual_full)))
            reference_scale = max(1., float(np.max(abs(independent))))
            require(reference_error <= 1e-10*reference_scale+1e-12,
                    "current actual row work differs from frozen independent action")
            lookup = {int(slot): i for i, slot in enumerate(all_fields)}
            expected_selected = independent[[lookup[int(slot)] for slot in slots]]
            selected_error = float(np.max(abs(expected_selected+
                                              np.asarray(work_old_order.T@density))))
            require(selected_error <= 1e-10*reference_scale+1e-12,
                    "C E differs from independent supported reference action")
            sparse.save_npz(folder / "E.npz", embedding.tocsr())
            sparse.save_npz(folder / "H.npz", h.tocsr())
            sparse.save_npz(folder / "CE.npz", row_work.tocsr())
            np.save(folder / "independent-action.npy", independent)
            result.update({"status": "current_owner_frozen_c2_work_parity_passed_development_only",
                           "nx": nx, "ny": ny, "halfcells": cells,
                           "span": policy["span_halfcells"], "supports": supports,
                           "owner_coordinates": int(owner._K.shape[0]),
                           "retained_coordinates": int(owner._beam_count-6*len(owner._beam_ids)),
                           "fixed_count": int(len(owner._fixed)),
                           "E_shape": list(embedding.shape), "E_nnz": int(embedding.nnz),
                           "signed_basis_negative_entries": int(np.count_nonzero(p.data < 0.)),
                           "H_shape": list(h.shape), "H_nnz": int(h.nnz),
                           "CE_shape": list(row_work.shape), "CE_nnz": int(row_work.nnz),
                           "frozen_selector_exact_error": 0.0,
                           "H_max_abs_error": h_error, "H_scale": h_scale,
                           "CE_selector_max_abs_error": ce_error,
                           "omitted_C_max_abs": omitted_max,
                           "independent_action_max_abs_error": reference_error,
                           "selected_action_max_abs_error": selected_error,
                           "independent_action_scale": reference_scale})
        finally:
            owner.close()
    except BaseException as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error),
                             "traceback": traceback.format_exc()}
    finally:
        save(folder / "result.json", result)
    if result["status"] != "current_owner_frozen_c2_work_parity_passed_development_only":
        raise RuntimeError(result["failure"]["message"])


def supervise(level: int, campaign: str) -> None:
    binding = preflight(level)
    folder = campaign_folder(level, campaign)
    require_fresh_output(folder)
    prior_wall = 0.0
    if level == 64:
        prior = campaign_folder(32, campaign)
        receipt = json.loads((prior / "receipt.json").read_text())
        outcome = json.loads((prior / "result.json").read_text())
        require(sha(prior / "result.json") == receipt.get("result_sha256") and
                sha(prior / "executed-source.py") == receipt.get("source_sha256") and
                sha(prior / "preflight.json") == receipt.get("input_preflight_sha256"),
                "same-campaign predecessor bytes changed")
        prior_wall = validate_prior_32(receipt, outcome, binding, campaign)
    require(240.0-prior_wall-MAX_WALL >= 60.0,
            "fresh family ledger cannot retain 60s reserve")
    sys.path.insert(0, str(SITE))
    import psutil
    require(PYTHON.is_file() and 0 in psutil.Process().cpu_affinity(),
            "Python3.14/CPU0 unavailable")
    folder.mkdir(parents=True)
    shutil.copyfile(__file__, folder / "executed-source.py")
    for name in SOURCE_FILES:
        snapshot = folder / name.replace("/", "__")
        shutil.copyfile(ROOT / name, snapshot)
        require(sha(snapshot) == binding["current_sources_sha256"][name],
                f"source changed before {level} launch: {name}")
    save(folder / "preflight.json", binding)
    parent = psutil.Process()
    original_affinity = parent.cpu_affinity()
    parent.cpu_affinity([0])
    env = os.environ.copy()
    env.update(ENV)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT),
                                          str(ROOT / "tests")))
    command = [str(PYTHON), "-u", str(Path(__file__).resolve()), "--child",
               "--level", str(level), "--campaign", campaign,
               "--output-dir", str(folder)]
    started = time.perf_counter()
    peak_rss, peak_threads, reason, affinity = 0, 0, None, None
    proc = None
    try:
        with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout,
                                    stderr=stderr, creationflags=0x4)
            process = psutil.Process(proc.pid)
            process.cpu_affinity([0])
            affinity = process.cpu_affinity()
            require(affinity == [0], "child CPU0 affinity missing")
            process.resume()
            while proc.poll() is None:
                elapsed = time.perf_counter()-started
                try:
                    live = [p for p in [process]+process.children(recursive=True) if p.is_running()]
                    peak_rss = max(peak_rss, sum(p.memory_info().rss for p in live))
                    peak_threads = max(peak_threads, sum(p.num_threads() for p in live))
                    if any(p.cpu_affinity() != [0] for p in live):
                        reason = "affinity_changed"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if elapsed >= MAX_WALL or peak_rss > MAX_RSS:
                    reason = "timeout" if elapsed >= MAX_WALL else "memory_limit"
                if reason:
                    for p in reversed([process]+process.children(recursive=True)):
                        try:
                            p.kill()
                        except psutil.NoSuchProcess:
                            pass
                    break
                time.sleep(0.01)
            proc.wait(timeout=3)
    finally:
        elapsed = time.perf_counter()-started
        parent.cpu_affinity(original_affinity)
    result_path = folder / "result.json"
    files = ["E.npz", "H.npz", "CE.npz", "independent-action.npy"]
    receipt = {"campaign": campaign,
               "status": reason or ("completed" if proc and proc.returncode == 0 and
                result_path.exists() else "failed"), "command": command,
               "source_sha256": sha(folder / "executed-source.py"),
               "source_dependencies_sha256": binding["current_sources_sha256"],
               "input_preflight_sha256": sha(folder / "preflight.json"),
               "child_affinity_before_resume": affinity,
               "requested_environment": ENV, "process_wall_seconds": elapsed,
               "max_wall_seconds": MAX_WALL, "peak_process_tree_rss_bytes": peak_rss,
               "max_process_tree_rss_bytes": MAX_RSS,
               "peak_process_tree_threads": peak_threads,
               "exit_code": proc.returncode if proc else None,
               "result_sha256": sha(result_path) if result_path.exists() else None,
               "output_sha256": {name: sha(folder / name) for name in files if (folder / name).exists()},
               "prior_32_wall_seconds": prior_wall,
               "fresh_240_second_ledger_consumed": prior_wall+elapsed,
               "remaining_of_240_seconds": 240.0-prior_wall-elapsed,
               "reserved_final_seconds": 60.0}
    save(folder / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in
                      ("status", "process_wall_seconds", "peak_process_tree_rss_bytes",
                       "exit_code", "remaining_of_240_seconds")}))
    if receipt["status"] != "completed":
        raise RuntimeError("current-owner parity stopped; inspect retained result/receipt/logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--level", type=int, choices=(32, 64), required=True)
    parser.add_argument("--campaign", default="v2")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.preflight:
        save(args.preflight, preflight(args.level))
    elif args.child and args.output_dir:
        child(args.level, args.output_dir, args.campaign)
    else:
        supervise(args.level, args.campaign)
