"""Warmed, alternating stable-path pairs against a frozen helper source.

The public caller must remain byte-identical to its frozen snapshot. Each process
warms both variants, then measures one alternating pair to respect process bounds.
"""
from __future__ import annotations
import argparse,contextlib,hashlib,importlib.util,json,os,pathlib,platform,sys,time
import numpy as np
import scipy
from threadpoolctl import threadpool_info
from anyfesolver import solve_static_nonlinear
import anyfesolver._plate_stability_continuation as current_helper
from investigate_plate_edge_shear import plate,jsonable


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark(output,baseline_dir,pair):
    repo=pathlib.Path(__file__).resolve().parents[1]
    caller=repo/'src/anyfesolver/nonlinear_static.py'
    assert digest(caller)==digest(baseline_dir/'nonlinear_static_before.py'),'Caller changed; frozen reference must be reconsidered'
    name='anyfesolver._plate_stability_frozen_reference'
    spec=importlib.util.spec_from_file_location(name,baseline_dir/'helper_before.py')
    baseline=importlib.util.module_from_spec(spec);sys.modules[name]=baseline;spec.loader.exec_module(baseline)
    files=[pathlib.Path(__file__),repo/'scripts/investigate_plate_edge_shear.py',caller,repo/'src/anyfesolver/_plate_stability_continuation.py',baseline_dir/'helper_before.py',baseline_dir/'nonlinear_static_before.py']
    files += [repo/'src/anyfesolver'/f for f in ['corotational.py','elements.py','e4_pl_element.py','assembly.py']]
    result={'comparison':'Same public stable path, exact cubic local-work quadrature reduction3to2','pair':pair,'caller_binding':'Current caller is SHA256-identical to frozen caller; helper loaded separately without changing its seal','controls':{'num_steps':4,'max_iterations':25,'tolerance':1e-6,'min_step_fraction':1/1024,'kinematics':'corotational','corotational_tangent':'work_conjugate','stability_control':'plate_energy'},'fixture':'4x4,1m square,.01thickness,E210GPa,nu.3,1mm sampled sine imperfection,clamp left,uy=.01 right','source_hashes':{str(p):digest(p) for p in files},'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'module_path':str(caller),'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']},'warmups':[],'samples':[]}
    def save():output.write_text(json.dumps(result,default=jsonable,indent=2))
    def sample(mode):
        helper=baseline if mode=='baseline' else current_helper
        old=sys.modules['anyfesolver._plate_stability_continuation']
        sys.modules['anyfesolver._plate_stability_continuation']=helper
        try:
            m=plate(.01,True,4);start=time.perf_counter()
            r=solve_static_nonlinear(m,None,num_steps=4,max_iterations=25,kinematics='corotational',corotational_tangent='work_conjugate',stability_control='plate_energy',record_increment_snapshots=True)
            wall=time.perf_counter()-start
            physical={'status':r.status,'u':r.displacements.tolist(),'snapshots':[{'factor':s.load_factor,'u':s.displacements.tolist()} for s in r.snapshots],'steps':[s.to_dict() for s in r.steps],'stability':r.info['stability_control']}
            assert r.converged and r.load_factor==1.
            key=hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest()
            return {'mode':mode,'wall':wall,'digest':key,'physical_result':physical}
        finally:sys.modules['anyfesolver._plate_stability_continuation']=old
    modes=['baseline','candidate']
    for mode in modes:
        result['warmups'].append(sample(mode));save()
    result['threadpools']=threadpool_info()
    try:
        import numba
        result['numba']={'version':numba.__version__,'threads':numba.get_num_threads()}
    except ImportError:result['numba']=None
    assert len({s['digest'] for s in result['warmups']})==1,'Physical path/counter drift'
    save()
    for mode in modes if pair%2==0 else modes[::-1]:
        item=sample(mode);assert item['digest']==result['warmups'][0]['digest']
        result['samples'].append(item);save()
    result['wall_reduction']=1-next(s['wall'] for s in result['samples'] if s['mode']=='candidate')/next(s['wall'] for s in result['samples'] if s['mode']=='baseline');save()
    print(json.dumps({'pair':pair,'times':{s['mode']:s['wall'] for s in result['samples']},'reduction':result['wall_reduction'],'digest':result['samples'][0]['digest']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--baseline-dir',type=pathlib.Path,required=True);p.add_argument('--pair',type=int,choices=[0,1,2],required=True);a=p.parse_args();benchmark(a.output,a.baseline_dir,a.pair)
