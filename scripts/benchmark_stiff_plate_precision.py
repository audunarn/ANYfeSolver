"""Fresh-process warmed source-frozen stiff-plate comparison.

Each invocation loads one complete source overlay before ANYfeSolver imports,
warms it once and measures once. The outer ledger alternates variants by pair.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,pathlib,platform,sys,time
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['baseline','candidate'],required=True);p.add_argument('--baseline-dir',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--pair',type=int,required=True);a=p.parse_args()
repo=pathlib.Path(__file__).resolve().parents[1];source=repo/'src/anyfesolver'
if a.variant=='baseline':
    overlay=a.baseline_dir/'overlay';overlay.mkdir(exist_ok=True)
    for dest,old in [('corotational.py','corotational_before.py'),('_corotational_batch.py','batch_before.py'),('nonlinear_static.py','nonlinear_static_before.py'),('recovery.py','recovery_before.py'),('_plate_stability_continuation.py','stability_before.py')]:
        (overlay/dest).write_bytes((a.baseline_dir/old).read_bytes())
    spec=importlib.util.spec_from_file_location('anyfesolver',source/'__init__.py',submodule_search_locations=[str(overlay),str(source)])
    package=importlib.util.module_from_spec(spec);sys.modules['anyfesolver']=package;spec.loader.exec_module(package)
import numpy as np
import scipy
import anyfesolver as fs
import anyfesolver.nonlinear_static as ns
import anyfesolver.corotational as cr
import anyfesolver.recovery as recovery
from threadpoolctl import threadpool_info
from explore_nonlinear_abuse import plate_model

def sample():
    m,_=plate_model({'name':'stiff','E':1e12});linear,_=fs.solve_linear(m,None)
    start=time.perf_counter();r=fs.solve_static_nonlinear(m,None,num_steps=2,max_iterations=20,min_step_fraction=.0625,kinematics='corotational',corotational_tangent='work_conjugate',record_increment_snapshots=True);wall=time.perf_counter()-start
    stable=r.info.get('corotational_shell_rotation_stable',False)
    free=np.setdiff1d(np.arange(len(r.displacements)),list(m.mesh.nodes[1].dofs)+list(m.mesh.nodes[3].dofs)+[m.mesh.nodes[2].dofs[1],m.mesh.nodes[4].dofs[1]])
    checks=[]
    for snapshot in r.snapshots:
        opts={'stable_shell_rotation':True} if stable else {}
        force,_,_=ns._assemble_nonlinear_system(m,snapshot.displacements,{},5,tangent=False,kinematics='corotational',corotational_tangent='work_conjugate',**opts)
        error=max(abs(snapshot.displacements[m.mesh.nodes[n].dofs[1]]-snapshot.load_factor*1e-6) for n in [2,4]);clamp=max(np.max(abs(snapshot.displacements[m.mesh.nodes[n].dofs])) for n in [1,3]);residual=float(np.linalg.norm(force[free]));assert residual<=1e-6 and error==0. and clamp==0.
        checks.append({'factor':snapshot.load_factor,'u':snapshot.displacements.tolist(),'residual':residual,'driven_error':error,'clamp_error':float(clamp)})
    assert r.converged and r.load_factor==1.
    linear_error=float(np.max(abs(r.displacements-linear)));assert linear_error<2e-12
    payload={'status':r.status,'u':r.displacements.tolist(),'snapshots':checks,'steps':[v.to_dict() for v in r.steps],'policy':stable,'iterations':r.info['total_newton_iterations'],'work':r.info['nonlinear_performance'],'adaptation':r.info['convergence_adaptation'],'linear_max_error':linear_error}
    return dict(wall=wall,payload=payload)
files=[pathlib.Path(__file__),pathlib.Path(ns.__file__),pathlib.Path(cr.__file__),pathlib.Path(recovery.__file__),source/'_corotational_batch.py',repo/'scripts/explore_nonlinear_abuse.py',source/'e4_pl_element.py',source/'elements.py']
result={'variant':a.variant,'pair':a.pair,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'module_paths':{'static':ns.__file__,'corotational':cr.__file__,'recovery':recovery.__file__},'controls':{'num_steps':2,'max_iterations':20,'tolerance':1e-6,'min_step_fraction':.0625,'E':1e12,'thickness':.01,'target_uy':1e-6,'kinematics':'corotational','corotational_tangent':'work_conjugate'},'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']}}
def save():a.output.write_text(json.dumps(result,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),encoding='utf8')
result['warmup']=sample();result['threadpools']=threadpool_info();save();result['sample']=sample();save();print(json.dumps({'variant':a.variant,'pair':a.pair,'wall':result['sample']['wall'],'iterations':result['sample']['payload']['iterations'],'failed':result['sample']['payload']['work']['solver']['failed_increments']}))
