"""Fixed-fixture full native KKT sensitivity; diagnostic copies only.

No production coefficient, multiplier policy, geometry or load is changed.
Retains native algebraic variables. Infinitesimal response is not evidence of
drilling independence over thickness, refinement or arbitrary coefficient ranges.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def csr(arrays, name):
    return sparse.csr_matrix((arrays[name+"_data"], arrays[name+"_indices"],
                              arrays[name+"_indptr"]), shape=tuple(arrays[name+"_shape"]))


def dot(a, b):
    return math.fsum(float(x)*float(y) for x, y in zip(a, b))


def role_parity(current_q, current_lambda, saved_q, saved_lambda, layout, coordinate_scale, constraint_scale):
    """Check every native role, not just the scalar applied work."""
    shell = 6*layout["counts"]["shell_nodes"]
    physical = layout["counts"]["physical_dofs"]
    members = layout["counts"]["members"]
    slots = {
        "shell": np.arange(shell), "beam_nodal": np.arange(shell,physical),
        "retained_rotation": np.concatenate([physical+24*i+np.arange(6) for i in range(members)]),
        "retained_resultant": np.concatenate([physical+24*i+np.arange(6,24) for i in range(members)])}
    results = {}
    for name,indices in slots.items():
        a, b = current_q[indices]/coordinate_scale[indices], saved_q[indices]/coordinate_scale[indices]
        error = float(np.linalg.norm(a-b)/max(np.linalg.norm(b),np.finfo(float).tiny))
        if error > 1e-10: raise ValueError("saved baseline role parity failed: "+name)
        results[name] = error
    error = float(np.linalg.norm((current_lambda-saved_lambda)*constraint_scale)/
                  max(np.linalg.norm(saved_lambda*constraint_scale),np.finfo(float).tiny))
    if error > 1e-10: raise ValueError("saved multiplier parity failed")
    results["multiplier"] = error
    return results


def run(saved_root, output, site):
    if output.exists(): raise FileExistsError(output)
    output.mkdir()
    sys.path.insert(0, str(site.resolve()))
    import anyfesolver
    import numba
    from threadpoolctl import threadpool_info
    from anyfesolver import stiffened_shell as api
    from anyfesolver._stiffened_shell_ge_experimental import _shell_reference
    from anyfesolver._b2_physical_linear_batch_dev import _accurate_sparse_residual
    if not Path(anyfesolver.__file__).resolve().is_relative_to(site.resolve()):
        raise ValueError("isolated installed solver required")
    pools = threadpool_info()
    if not pools or any(p["num_threads"] != 1 for p in pools) or not numba.config.DISABLE_JIT:
        raise ValueError("effective one-thread/JIT-off environment required")
    report = dict(status="IN_PROGRESS", imported_solver=anyfesolver.__file__, pools=pools,
        jit_disabled=True, numba_threads=numba.get_num_threads(), checks={}, cases={},
        scope="fixed flat Q4/c2 elastic native mixed fixture, development sensitivity only",
        perturbations=[1e-3, 1e-4], factorization_count=0, linear_solve_count=0)
    def write():
        (output/"result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    write()
    try:
        manifest = json.loads((saved_root/"refinement-v1-manifest.json").read_text(encoding="utf-8-sig"))
        fixture = saved_root/"refinement-v1-raw/native/n16"
        def checked(path):
            name = path.relative_to(saved_root).as_posix()
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest["files"][name]:
                raise ValueError("saved input identity changed: "+name)
            return hashlib.sha256(raw).hexdigest()
        inputs = [fixture/"operators.npz", fixture/"layout.json"]
        for name in ("fy", "mx", "mz"):
            inputs += [fixture/name/"raw-state.npz", fixture/name/"physical-transfer.json"]
        report["saved_inputs"] = {str(p.relative_to(saved_root)): checked(p) for p in inputs}
        layout = json.loads((fixture/"layout.json").read_text(encoding="utf-8-sig"))
        with np.load(fixture/"operators.npz", allow_pickle=False) as arrays:
            K, C, A = (csr(arrays, n) for n in ("K", "C", "A"))
            free, fixed = arrays["free"].copy(), arrays["fixed"].copy()
            coordinate_scale, constraint_scale = arrays["coordinate_scale"].copy(), arrays["constraint_scale"].copy()
            scale = np.r_[coordinate_scale[free], 1./constraint_scale]
        nx, ny = layout["nx"], layout["ny"]
        if (nx, ny, layout["halfcell_count"], layout["policy"]["c_lambda"]) != (16, 4, 32, 2):
            raise ValueError("exact declared fixed fixture required")
        cases = {}
        for name in ("fy", "mx", "mz"):
            with np.load(fixture/name/"raw-state.npz", allow_pickle=False) as state:
                cases[name] = {k: state[k].copy() for k in ("q", "load", "multipliers", "shell_xyz_m")}
        vertices = cases["mx"]["shell_xyz_m"]
        np.testing.assert_array_equal(vertices, cases["fy"]["shell_xyz_m"])
        row = nx+1
        cells = tuple((1+j*row+i, 2+j*row+i, 2+(j+1)*row+i, 1+(j+1)*row+i)
                      for j in range(ny) for i in range(nx))
        shell = api.ShellPanel(tuple(layout["shell_node_ids"]), tuple(map(tuple,vertices)),
                              cells, .035, 210e9, .3, 7850.)
        model, ids, xyz, index, pieces = _shell_reference(shell)
        rr, cc, values = [], [], {n: [] for n in ("core", "pl", "hourglass", "total")}
        for _, element, slots, _ in pieces:
            components = element.compute_stiffness_components(model.mesh, model.get_material("shell"))
            rr.extend(np.repeat(slots,24)); cc.extend(np.tile(slots,24))
            for name in values: values[name].extend(components[name].reshape(-1))
        blocks = {name: sparse.coo_matrix((v,(rr,cc)),shape=K.shape).tocsr() for name,v in values.items()}
        shell_size = 6*len(ids)
        nominal = K[:shell_size,:shell_size]
        delta = blocks["total"][:shell_size,:shell_size]-nominal
        parity = float(sparse.linalg.norm(delta)/sparse.linalg.norm(nominal))
        skew = float(sparse.linalg.norm(A-A.T)/sparse.linalg.norm(A))
        report["checks"].update(shell_total_relative=parity, native_kkt_relative_skew=skew)
        if parity > 1e-10 or skew > 1e-9:
            raise ValueError("saved operator/component or symmetric work premise failed")
        for name, matrix in blocks.items(): sparse.save_npz(output/(name+".npz"), matrix)
        scaling = sparse.diags(scale)
        def factor(matrix):
            report["factorization_count"] += 1
            return splu((scaling@matrix@scaling).tocsc())
        def solve(matrix, lu, rhs):
            report["linear_solve_count"] += 1
            value = scale*lu.solve(scale*rhs)
            for iteration in range(5):
                residual = _accurate_sparse_residual(matrix,value[:,None],rhs[:,None],exact_products=True)[:,0]
                ratio = float(np.linalg.norm(scale*residual)/max(np.linalg.norm(scale*rhs),np.finfo(float).tiny))
                # A small residual alone need not resolve physical work in the
                # native indefinite representation. Require compensated refinement
                # before the saved-state/finite-difference controls are evaluated.
                if iteration >= 2 and ratio <= 1e-10: return value, ratio
                value -= scale*lu.solve(scale*residual)
                report["linear_solve_count"] += 1
            raise ValueError("full native KKT residual unresolved: "+str(ratio))
        supports = np.array(layout["policy"]["support_endpoints_m"])
        n_support = len(supports)-1
        def transfer(multiplier, records):
            density = multiplier.reshape(5,n_support)
            moments = np.zeros((3,6))
            out = []
            for record in records:
                x, weight = record["x_m"], record["measure_m"]
                i = min(int(np.searchsorted(supports,x,side="right")-1),n_support-1)
                f = density[:3,i]
                # director=e_z, basis=(e_x,e_y), axis arm=-.05*e_z.
                m = np.array((.05*f[1]-density[4,i], -.05*f[0]+density[3,i], 0.))
                wrench = np.r_[f,m+np.cross((x,0.,0.),f)]
                out.append(wrench)
                for power in range(3): moments[power] += weight*x**power*wrench
            return moments, np.array(out)
        lu = factor(A)
        arrays_out = {}
        perturbations = {}
        for block_name in ("pl", "hourglass"):
            d = blocks[block_name][free,:][:,free]
            augmented = sparse.block_diag((d,sparse.csr_matrix((C.shape[0],C.shape[0]))),format="csr")
            perturbations[block_name] = [(step, factor(A+step*augmented),factor(A-step*augmented))
                                        for step in report["perturbations"]]
        for name, state in cases.items():
            q, f, multiplier = state["q"], state["load"], state["multipliers"]
            rhs = np.r_[f[free],np.zeros(C.shape[0])]
            base, residual = solve(A,lu,rhs)
            full_base = np.zeros(K.shape[0]); full_base[free] = base[:len(free)]
            saved_work, work = dot(q,f), dot(full_base,f)
            if work <= 0. or abs(work/saved_work-1.) > 1e-10:
                raise ValueError("saved baseline work not reproduced")
            roles = role_parity(full_base,base[len(free):],q,multiplier,layout,coordinate_scale,constraint_scale)
            records = json.loads((fixture/name/"physical-transfer.json").read_text(encoding="utf-8"))["station_records"]
            _, nominal_wrenches = transfer(multiplier, records)
            np.testing.assert_allclose(nominal_wrenches,
                np.array([r["beam_wrench_about_root_density"] for r in records]),rtol=1e-10,atol=1e-9)
            _, solved_wrenches = transfer(base[len(free):],records)
            np.testing.assert_allclose(solved_wrenches,nominal_wrenches,rtol=1e-10,atol=1e-9)
            item = dict(baseline_work_J=work, baseline_kkt_residual=residual, baseline_role_parity=roles, blocks={})
            report["cases"][name] = item
            for block_name in ("pl", "hourglass"):
                D = blocks[block_name]
                slope_rhs = np.r_[-(D@full_base)[free],np.zeros(C.shape[0])]
                dx, dr = solve(A,lu,slope_rhs)
                dq = np.zeros(K.shape[0]); dq[free] = dx[:len(free)]
                ds = dx[len(free):]
                energy = dot(full_base,D@full_base)
                work_slope = dot(f,dq)
                identity = abs(work_slope+energy)/work
                gap = float(np.max(abs(C@dq)))
                if identity > 1e-8 or gap > 1e-9:
                    raise ValueError("sensitivity work/compatibility failed")
                moment_slope,_ = transfer(ds,records)
                finite = []
                for step, plus, minus in perturbations[block_name]:
                    dfree = D[free,:][:,free]
                    aug = sparse.block_diag((dfree,sparse.csr_matrix((C.shape[0],C.shape[0]))),format="csr")
                    xp,rp = solve(A+step*aug,plus,rhs)
                    xm,rm = solve(A-step*aug,minus,rhs)
                    difference = (xp-xm)/(2*step)
                    # Report the complete stored-coordinate norm. Near-zero derivatives
                    # require a separate base-normalized absolute control, not a false relative claim.
                    base_norm = np.linalg.norm(base/scale)
                    slope_norm = np.linalg.norm(dx/scale)
                    absolute = float(np.linalg.norm((difference-dx)/scale)/max(base_norm,np.finfo(float).tiny))
                    relative = float(np.linalg.norm((difference-dx)/scale)/max(slope_norm,np.finfo(float).tiny))
                    if absolute > 1e-6*max(slope_norm/base_norm,1e-6):
                        raise ValueError("symmetric perturbation control failed: "+str((name,block_name,step,absolute,relative)))
                    finite.append(dict(step=step, derivative_relative=relative,
                        derivative_base_normalized=absolute, residual_plus=rp,residual_minus=rm))
                shell_slope = dq[:shell_size].reshape(-1,6)
                shell_base = full_base[:shell_size].reshape(-1,6)
                item["blocks"][block_name] = dict(work_slope_J=work_slope,
                    numerical_energy_J=energy, work_slope_relative=work_slope/work,
                    work_identity_applied_work_scale=identity, kkt_residual=dr, differentiated_gap_max=gap,
                    shell_translation_max_m=float(np.max(abs(shell_slope[:,:3]))),
                    shell_director_max_rad=float(np.max(abs(shell_slope[:,3:5]))),
                    shell_drilling_max_rad=float(np.max(abs(shell_slope[:,5]))),
                    shell_translation_baseline_max_m=float(np.max(abs(shell_base[:,:3]))),
                    shell_director_baseline_max_rad=float(np.max(abs(shell_base[:,3:5]))),
                    beam_translation_max_m=float(np.max(abs(dq[shell_size:shell_size+6*layout["counts"]["beam_nodes"]].reshape(-1,6)[:,:3]))),
                    physical_transfer_moment_slopes=moment_slope.tolist(),finite_difference=finite)
                arrays_out[name+"_"+block_name+"_dq"] = dq
                arrays_out[name+"_"+block_name+"_dlambda"] = ds
                arrays_out[name+"_q"] = full_base
                arrays_out[name+"_lambda"] = base[len(free):]
        np.savez_compressed(output/"states.npz",**arrays_out)
        report["status"] = "COMPLETED_DEVELOPMENT_ONLY"
        write()
        return report
    except BaseException as error:
        report.update(status="ERROR_PROPAGATED", failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        write()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-root",type=Path,required=True)
    parser.add_argument("--site",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = run(args.saved_root,args.output,args.site)
    print(json.dumps(dict(status=result["status"],checks=result["checks"],cases={name:{
        block:item["work_slope_relative"] for block,item in case["blocks"].items()}
        for name,case in result["cases"].items()})))
