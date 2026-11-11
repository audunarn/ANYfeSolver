"""Saved-operator supported-work discriminator; no assembly or physical solve.

--prepare is stdlib IO/hash only. --evaluate requires an explicit root-issued
execution plan bound to the preparation digest. Formal F03 remains unconsumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/agent_plans/development/r8-f03-supported-work-20261010"
NATIVE = ROOT / "docs/agent_plans/development/r8-bl5-matched-comparison-20261001/refinement-v1-raw/native/n16"
EXTERNAL = ROOT / "docs/agent_plans/development/r8-bl5-external-operator-20261009/fy-fullgeneral-v3"
ADMISSION = ROOT / ".development/r8-f03-reference-operator-admission-20261009/saved-reuse-io-v3/saved-reuse-io.json"
ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}
ENV.update(NUMBA_DISABLE_JIT="1")
CLAIM = "declared shared shell Q1 family on admitted saved operators; DEVELOPMENT ONLY"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def source_binding():
    """No recipe evaluation, numerical import, matrix materialization or solve."""
    inputs = json.loads((NATIVE / "inputs.json").read_text())
    layout = json.loads((NATIVE / "layout.json").read_text())
    admission = json.loads(ADMISSION.read_text())
    require(admission["status"] == "IO_SOURCE_APPLICABILITY_ESTABLISHED", "saved source admission unavailable")
    require(sha(EXTERNAL / "raw.json") == "ebacac2ea27341decc25a1a8ace6ca388b9b99391331f10f33297106804255b5", "external raw capture changed")
    require(sha(NATIVE / "operators.npz") == inputs["operators_sha256"] ==
            "dd14400f93f5eaf53f54cc89300917c87326a8913aec2c1c1588d39c0af5c0d5", "native saved operator changed")
    require(layout["counts"]["full_q"] == 1092 and layout["counts"]["physical_dofs"] == 708 and
            layout["counts"]["shell_nodes"] == 85 and layout["counts"]["retained_dofs"] == 384,
            "saved full retained layout changed")
    require(inputs["line"]["multiplier_span"] == 4 and len(inputs["line"]["shell_nodes"]) == 17,
            "native saved integrated P0 support layout changed")
    paths = [Path(__file__), ROOT / "src/anyfesolver/_stiffened_shell_l1_assembly_experimental.py",
             ROOT / "src/anyfesolver/_stiffened_shell_l1_supported_work_experimental.py",
             NATIVE / "operators.npz", NATIVE / "inputs.json", NATIVE / "layout.json",
             EXTERNAL / "raw.json", EXTERNAL / "receipt.json", ADMISSION]
    return {"claim": CLAIM, "qualification": False,
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "inputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths},
            "family": "six shell interior Q1 component fields; phi nodal generator; exact boundary zero",
            "units": {"amplitude": "dimensionless", "translation": "m", "rotation": "rad", "bilinear_work": "J"},
            "limits": ["no physical solve/assembly/eigenproblem", "no complete-space transfer", "no mechanism or accuracy verdict",
                       "raw captured operator all-column tangent semantics not independently certified"]}


def check_plan(plan, binding):
    require(plan.get("authorized_by") == "root" and plan.get("scope") == "saved-supported-work-only" and
            plan.get("binding_sha256") == digest(binding) and plan.get("physical_runs") == 0 and
            plan.get("assembly_runs") == 0 and plan.get("solves") == 0 and plan.get("eigenproblems") == 0,
            "fresh root-bound saved-work authorization required")


def evaluate(folder, plan_path):
    """Bounded saved-data products only; the external raw matrix stays unsymmetric."""
    folder = Path(folder)
    result = {"status": "failed", "claim": CLAIM, "qualification": False}
    try:
        binding = source_binding()
        check_plan(json.loads(Path(plan_path).read_text()), binding)
        require({key: os.environ.get(key) for key in ENV} == ENV, "one-thread/JIT-off environment required")
        import numpy as np
        from scipy import sparse
        import psutil
        from threadpoolctl import threadpool_info
        from anyfesolver._stiffened_shell_l1_assembly_experimental import compile_supported_shell_work

        pools = threadpool_info()
        require(psutil.Process().cpu_affinity() == [0] and pools and all(p["num_threads"] == 1 for p in pools),
                "effective CPU0 and one numerical thread required")
        result.update(binding=binding, runtime={"environment": ENV, "affinity": [0], "blas_pools": pools,
                      "numpy": np.__version__, "scipy": __import__("scipy").__version__})
        inputs = json.loads((NATIVE / "inputs.json").read_text())
        layout = json.loads((NATIVE / "layout.json").read_text())
        raw = json.loads((EXTERNAL / "raw.json").read_text())
        inventory = raw["raw_inventory"]
        typed = layout["typed_physical_nodes"]
        require([row["q_offset"] for row in typed] == list(range(0, 708, 6)), "native physical order changed")
        external_nodes = [row["external_node"] for row in typed]
        require(len(set(external_nodes)) == 118, "physical node identity collision")
        for row, xyz in zip(typed, inputs["xyz_m"]):
            require(inventory["nodes"][str(row["external_node"])] == xyz, "native/external physical coordinates differ")
        physical_ids = [raw["node_dofs"][str(node)][d] for node in external_nodes for d in range(6)]
        require(len(set(physical_ids)) == 708 and min(physical_ids) >= 0 and max(physical_ids) < 783,
                "complete captured physical equation map required")
        ke_raw = np.asarray(raw["printA"], dtype=float).reshape((783, 783))
        ke = ke_raw[np.ix_(physical_ids, physical_ids)]
        with np.load(NATIVE / "operators.npz", allow_pickle=False) as archive:
            op = {key: np.array(archive[key], copy=True) for key in archive.files}
        kn = sparse.csr_matrix((op["K_data"], op["K_indices"], op["K_indptr"]), shape=tuple(op["K_shape"]))
        cn = sparse.csr_matrix((op["C_data"], op["C_indices"], op["C_indptr"]), shape=tuple(op["C_shape"]))
        require(kn.shape == (1092, 1092) and cn.shape == (40, 1092), "native saved full dimensions changed")
        shell_ids = inputs["shell"]["node_ids"]
        shell_map = {row["native_node"]: row["external_node"] for row in typed if row["domain"] == "shell"}
        captured_cells = [element["nodes"] for element in inventory["elements"].values() if element["kind"] == "ASDShellQ4"]
        require(captured_cells == [[shell_map[node] for node in cell] for cell in inputs["shell"]["q4_elements"]],
                "exact shell cell correspondence required")
        fixed_n = op["fixed"].tolist()
        physical_index = {(node, dof+1): 6*i+dof for i, node in enumerate(external_nodes) for dof in range(6)}
        fixed_e = [physical_index[(int(node), dof+1)] for node, flags in inventory["fixes"].items()
                   for dof, flag in enumerate(flags) if flag]
        require(len(fixed_e) == 35 and set(fixed_e) == set(fixed_n) and 515 not in fixed_e,
                "35 physical root supports and released beam yaw required")
        kwargs = dict(shell_ids=shell_ids, shell_xyz=inputs["shell"]["coordinates_m"],
                      shell_cells=inputs["shell"]["q4_elements"], trace_nodes=inputs["line"]["shell_nodes"])
        pn = compile_supported_shell_work(**kwargs, full_count=1092, fixed_slots=fixed_n)
        pe = compile_supported_shell_work(**kwargs, full_count=708, fixed_slots=fixed_e)
        require(np.array_equal(pn.evaluation[:708], pe.evaluation) and np.all(pn.evaluation[708:] == 0.),
                "shared physical family and all retained zero rows required")
        ce = np.zeros((40, 708))
        require(len(inventory["equations"]) == 40, "40 integrated physical external rows required")
        for i, row in enumerate(inventory["equations"]):
            for node, dof, coefficient in zip(row[::3], row[1::3], row[2::3]):
                ce[i, physical_index[(node, dof)]] += coefficient
        pn.require_constraint_kernel(cn)
        pe.require_constraint_kernel(ce)
        wn, we = pn.body_work(kn), pe.body_work(ke)
        difference = wn-we
        # Do not average/symmetrize either raw operator or its projected work.
        result.update(status="saved_supported_work_evaluated_development_only", names=list(pn.names),
                      body_work_J={"native": wn.tolist(), "external": we.tolist(), "native_minus_external": difference.tolist()},
                      raw_asymmetry_max={"external": float(np.max(abs(ke-ke.T))),
                                         "native_projected_J": float(np.max(abs(wn-wn.T))),
                                         "external_projected_J": float(np.max(abs(we-we.T)))},
                      exact_checks={"native_trace_kernel": True, "external_trace_kernel": True,
                                    "native_structural_exclusion": True, "external_structural_exclusion": True,
                                    "root_and_boundary_zero": True, "retained_zero": True},
                      interpretation="Differences are saved-shell-operator response on these shared supported fields; connection work is identically zero. No attribution of the original coupled response.")
        np.savez(folder / "arrays.npz", native_evaluation=pn.evaluation, external_evaluation=pe.evaluation,
                 native_work=wn, external_work=we, difference_work=difference)
        result["arrays_sha256"] = sha(folder / "arrays.npz")
    except Exception:
        result["exception"] = traceback.format_exc()
        raise
    finally:
        save(folder / "result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    require(args.prepare != args.evaluate, "choose exactly one mode")
    require(not args.output.exists(), "preserve every unique attempt directory")
    args.output.mkdir(parents=True)
    if args.prepare:
        binding = source_binding()
        save(args.output / "binding.json", binding)
        save(args.output / "preparation.json", {"status": "prepared_io_only", "binding_sha256": digest(binding), "qualification": False})
    else:
        require(args.plan is not None, "root execution plan path required")
        evaluate(args.output, args.plan)


if __name__ == "__main__":
    main()
