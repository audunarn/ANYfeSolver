"""bench.py CASE [--profile] [--json-extra '{}'] ; run with py -3.14, PYTHONPATH=<worktree>/src, OMP_NUM_THREADS=1.
Cases: lin_plate nl_elastic_plate nl_plastic_plate stiffened_beam_shell arc_panel prepared_envelope"""
import sys, os, json, time, hashlib, argparse, cProfile, runpy
import numpy as np
import pathlib; WT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, WT + r'\scripts')
import anyfesolver as fs
from anyfesolver.elements import create_shell_element
from anyfesolver.boundary import FixedSupport, LoadCase, BoundaryCondition
from anyfesolver.material_curves import DNVC208MaterialCurve

def H(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a, dtype=float)); h.update(a.tobytes())
    return h.hexdigest()[:16]

def plate(n, t, p, plastic=0):
    m = fs.FEModel('p')
    curve = DNVC208MaterialCurve(sigma_prop=3e7, sigma_yield=3.2e7, sigma_yield_2=3.4e7, eps_p_y1=.005, eps_p_y2=.01, K=4e8, n=.2) if plastic else None
    m.add_material('mat', 210e9, .3, density=7850., yield_stress=3e7 if curve else 0., hardening_curve=curve)
    for j in range(n+1):
        for i in range(n+1): m.add_node(1+j*(n+1)+i, i/n, j/n, 0.)
    for j in range(n):
        for i in range(n):
            v = 1+j*(n+1)+i; e = 1+j*n+i
            m.add_element(e, create_shell_element(e, [v, v+1, v+n+2, v+n+1], 'mat', thickness=t))
    edge = [1+j*(n+1)+i for j in range(n+1) for i in range(n+1) if i in (0, n) or j in (0, n)]
    m.add_boundary_condition(FixedSupport('clamp', edge))
    L = LoadCase('p')
    for e in m.mesh.elements: L.add_pressure_load(e, p)
    m.apply_boundary_conditions(); return m, L

def panel(n=6, t=.01, rise=.05, p=1e4):
    m = fs.FEModel('a'); m.add_material('mat', 210e9, .3, density=7850.)
    for j in range(n+1):
        for i in range(n+1):
            x = i/n; m.add_node(1+j*(n+1)+i, x, j/n, rise*(1-(2*x-1)**2))
    for j in range(n):
        for i in range(n):
            v = 1+j*(n+1)+i; e = 1+j*n+i
            m.add_element(e, create_shell_element(e, [v, v+1, v+n+2, v+n+1], 'mat', thickness=t))
    m.add_boundary_condition(FixedSupport('c', [1+j*(n+1)+i for j in range(n+1) for i in (0, n)]))
    L = LoadCase('p')
    for e in m.mesh.elements: L.add_pressure_load(e, p)
    m.apply_boundary_conditions(); return m, L

def nl_result(r):
    u = np.asarray(r.displacements)
    return dict(status=str(r.status), conv=bool(r.converged), f=float(r.load_factor), it=r.info.get('total_newton_iterations')), H(u)

def case_lin_plate():
    m, L = plate(24, .01, 1e4)
    t0 = time.perf_counter(); u, info = fs.solve_linear(m, L); t1 = time.perf_counter()
    from anyfesolver.assembly import compute_stresses
    s = compute_stresses(m, u); t2 = time.perf_counter()
    arrs = [u]
    for e in sorted(s):
        for k in sorted(s[e]):
            try: arrs.append(np.asarray(s[e][k], dtype=float))
            except Exception: pass
    return dict(status='ok', solve_s=round(t1-t0, 4), stress_recovery_s=round(t2-t1, 4), ndof=int(u.size)), H(*arrs)

def case_nl_elastic_plate():
    m, L = plate(12, .01, 3e4)
    r = fs.solve_static_nonlinear(m, L, num_steps=5, max_iterations=25, kinematics='von_karman')
    return nl_result(r)

def case_nl_plastic_plate():
    m, L = plate(6, .005, 3e4, plastic=1)
    r = fs.solve_static_nonlinear(m, L, num_steps=5, max_iterations=25, kinematics='von_karman')
    return nl_result(r)

def case_stiffened_beam_shell():
    ex = runpy.run_path(WT + r'\examples\beam_shell_coupling.py')
    case = ex['build_model'](ex['DEFAULT_COUPLING'])
    r = fs.solve_static_nonlinear(case.model, case.load, kinematics='corotational', num_steps=ex['NUM_STEPS'],
                                  max_iterations=ex['MAX_ITERATIONS'], tolerance=ex['TOLERANCE'], record_increment_snapshots=True)
    return nl_result(r)

def case_arc_panel():
    m, L = panel()
    ctl = fs.ArcLengthControl(initial_load_increment=.05, max_steps=25, stop_after_peak_steps=4)
    r = fs.solve_static_arc_length(m, L, control=ctl, kinematics='von_karman')
    ls = [s.load_factor for s in r.steps]
    arrs = [np.asarray(r.displacements)] if hasattr(r, 'displacements') else []
    arrs.append(np.array(ls))
    return dict(status=str(r.status), nsteps=len(ls), peak=max(ls) if ls else None, final=ls[-1] if ls else None), H(*arrs)

def case_prepared_envelope():
    from benchmark_static_batches import mixed_free_model, free_cases, positions
    from anyfesolver import prepare_static_evaluator, StaticCase, RecoveryConfig, ResourceConfig
    model, groups, balancing = mixed_free_model(1)
    idx = [positions('pilot')[i] for i in np.linspace(0, len(positions('pilot'))-1, 4, dtype=int)]
    nodes = tuple(model.mesh.nodes); els = tuple(model.mesh.elements)
    sn = tuple(dict.fromkeys((nodes[0], nodes[len(nodes)//2], nodes[-1])))
    se = tuple(dict.fromkeys((els[0], els[len(els)//2], els[-1])))
    dofs = np.array([d for n in sn for d in model.mesh.nodes[n].dofs], dtype=int)
    rec = RecoveryConfig(node_ids=sn, element_ids=se, history_mode='selected', store_full_histories=False)
    res = ResourceConfig(solver_threads=1, assembly_threads=1, recovery_threads=1)
    owner = prepare_static_evaluator(model, output_dofs=dofs, recovery_config=rec, resource_config=res)
    arrs = []; st = 'ok'
    try:
        for block in owner.iter_batches((StaticCase(n, c, t) for n, t, c in free_cases(model, groups, balancing, idx)), batch_size=8):
            arrs.append(block.displacements)
            for r in block.case_results:
                if r['status'] != 'converged': st = 'uncertified'
    finally: owner.close()
    return dict(status=st, ncases=len(idx)), H(*arrs)

CASES = {k[5:]: v for k, v in list(globals().items()) if k.startswith('case_')}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('case'); ap.add_argument('--profile', action='store_true')
    ap.add_argument('--outdir', default='.')
    a = ap.parse_args(); f = CASES[a.case]
    try:
        t = time.perf_counter(); f(); cold = time.perf_counter()-t
        t = time.perf_counter(); extra, h = f(); wall = time.perf_counter()-t
        out = dict(case=a.case, wall_s=round(wall, 3), cold_s=round(cold, 2), sha256=h, status=extra.get('status'), extra=extra)
        if a.profile:
            pr = cProfile.Profile(); pr.enable(); f(); pr.disable()
            pr.dump_stats(os.path.join(a.outdir, a.case + '.prof')); out['profiled'] = True
    except Exception as e:
        import traceback; traceback.print_exc()
        out = dict(case=a.case, wall_s=None, sha256=None, status='ERROR:' + type(e).__name__ + ':' + str(e)[:200])
    print(json.dumps(out))
