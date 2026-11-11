"""Warmed public-route pairs, optionally against frozen prior source modules."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_info
import anyfesolver as fs
import anyfesolver.contact as current_contact
import anyfesolver._impact_coupling as current_coupling


def load_snapshot(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def run(method, cutbacks, *, stiffness=40000., damping=0., baseline=None, deform=False):
    solver=current_contact
    if method=='baseline-anderson':
        solver,helper=baseline
        sys.modules['anyfesolver._impact_coupling']=helper
    start=time.perf_counter()
    try:
        model=solver._verification_contact_panel()
        if deform:
            from anyfesolver.boundary import BoundaryCondition
            model.add_boundary_condition(BoundaryCondition('pin-one-z',[1],{'uz':0.}))
        result=solver.solve_transient_sphere_impact(
            model,fs.TransientConfig(dt=.01,t_end=.03),
            solver.RigidSphereImpact('coupled',radius=.2,mass=1.,start_point=(.5,.5,.205),travel_direction=(0,0,-1),speed=1.),
            solver.SphereContactConfig(penalty_stiffness=stiffness,contact_damping=damping),
            nonlinear_config=solver.NonlinearTransientConfig(enabled=True,max_iterations=25,max_cutbacks=cutbacks,
                coupled_contact_acceleration='anderson' if method=='baseline-anderson' else method))
    finally:
        sys.modules['anyfesolver._impact_coupling']=current_coupling
    return {'method':method,'deform':deform,'k':stiffness,'damping':damping,'max_cutbacks':cutbacks,
        'wall_seconds':time.perf_counter()-start,'status':result.status,'times':result.times.tolist(),
        'x':result.sphere_positions.tolist(),'v':result.sphere_velocities.tolist(),
        'f':result.contact_force_history.tolist(),'u':result.displacements.tolist(),
        'iterations':result.diagnostics['iteration_counts'],'factorizations':result.diagnostics['factorization_count'],
        'cutbacks':result.diagnostics['cutback_count'],'steps':result.diagnostics['contact_step_diagnostics'],
        'momentum_error':result.sphere_momentum_balance_error,
        'activation':result.diagnostics['coupled_contact_acceleration']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--stiffness',type=float,default=40000.)
    p.add_argument('--baseline-dir',type=Path)
    p.add_argument('--candidate-method',choices=['anderson','contact_newton'],default='anderson')
    p.add_argument('--deform',action='store_true')
    p.add_argument('--validation-json',type=Path,help='Reuse already checked immutable validation trajectories')
    a=p.parse_args()
    if a.repeats<3:p.error('at least three paired repeats are required')
    paths=[Path(current_contact.__file__),Path(current_coupling.__file__),Path(__file__)]
    baseline=None
    if a.baseline_dir:
        old_contact=a.baseline_dir/'contact_before.py';old_helper=a.baseline_dir/'coupling_before.py'
        paths.extend([old_contact,old_helper])
        baseline=(load_snapshot('anyfesolver._benchmark_prior_contact',old_contact),
                  load_snapshot('anyfesolver._benchmark_prior_coupling',old_helper))
    methods=['baseline-anderson' if baseline else 'none',a.candidate_method]
    import numba
    data={'source_hashes':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']},
        'runtime':{'threadpools':threadpool_info(),'numba_threads':numba.get_num_threads(),
            'versions':{name:importlib.metadata.version(name) for name in ['numpy','scipy','numba']},'anyfesolver_module':fs.__file__},
        'warmup':[],'runs':[],'validation':[]}
    output=Path(a.output)
    def save():output.write_text(json.dumps(data,indent=2))
    # Frozen baseline uses its own original contact caller and helper together.
    for method in methods:
        data['warmup'].append(run(method,4,stiffness=a.stiffness,baseline=baseline,deform=a.deform));save()
        if data['warmup'][-1]['status']!='completed':
            raise RuntimeError('Cannot time completed-result pairs: warmup did not complete')
    for i in range(a.repeats):
        for method in (methods if i%2==0 else methods[::-1]):
            row=run(method,4,stiffness=a.stiffness,baseline=baseline,deform=a.deform)
            data['runs'].append(row);save()
            print(method,row['status'],row['wall_seconds'],row['factorizations'],row['cutbacks'],flush=True)
    # Complete same-dt trajectories for independent unchanged-equations checks.
    if a.validation_json:
        data['validation']=json.loads(a.validation_json.read_text())
        data['validation_source']={'path':str(a.validation_json),'sha256':hashlib.sha256(a.validation_json.read_bytes()).hexdigest()}
        save()
    else:
        for damping in [0.,10.]:
            data['validation'].append(run(a.candidate_method,0,stiffness=a.stiffness,damping=damping,deform=a.deform));save()
    for method in methods:
        rows=[r for r in data['runs'] if r['method']==method]
        print(method,'median_seconds',float(np.median([r['wall_seconds'] for r in rows])))

if __name__=='__main__':main()
