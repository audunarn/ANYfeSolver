"""Three alternating warmed pairs: TWO DIFFERENT corotational force formulations.

This is not an optimization/same-equations benchmark. Geometry, controls and
material are identical; reproducibility is checked separately within each mode.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,platform,statistics,time
import numpy as np
from threadpoolctl import threadpool_info
from anyfesolver import solve_static_nonlinear
from investigate_plate_edge_shear import plate,jsonable

def sample(mode):
    model=plate(.01,True,2)
    start=time.perf_counter()
    r=solve_static_nonlinear(model,None,num_steps=4,max_iterations=25,kinematics='corotational',corotational_tangent=mode,record_increment_snapshots=True)
    wall=time.perf_counter()-start
    payload={'status':r.status,'u':r.displacements.tolist(),'snapshots':[{'factor':s.load_factor,'u':s.displacements.tolist()} for s in r.snapshots],'steps':[s.to_dict() for s in r.steps]}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return {'mode':mode,'wall':wall,'iterations':r.info['total_newton_iterations'],'accepted_steps':len(r.steps),'failed_attempts':sum(a['action']=='cutback_after_nonconvergence' for a in r.info['convergence_adaptation']),'reaction':r.steps[-1].support_reactions['right-shear'][1],'max_incremental_w':float(np.max(abs(r.displacements[2::6]))),'digest':digest,'physical_result':payload}

def benchmark(output):
    repo=pathlib.Path(__file__).resolve().parents[1]
    files=[pathlib.Path(__file__),repo/'scripts/investigate_plate_edge_shear.py']+[repo/'src/anyfesolver'/n for n in ['corotational.py','_corotational_batch.py','nonlinear_static.py','elements.py','e4_pl_element.py']]
    result={'comparison':'different force formulations, NOT physics-preserving optimization','fixture':'2x2 square1m,t.01,E210GPa,nu.3,initial1mmsine-bump,leftallfixed,rightuy.01only','controls':{'num_steps':4,'max_iterations':25,'tolerance':1e-6,'min_step_fraction':1/1024,'control':'force'},'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'python':platform.python_version(),'numpy':np.__version__,'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']},'warmups':[],'samples':[]}
    def save():output.write_text(json.dumps(result,default=jsonable,indent=2))
    modes=['consistent','work_conjugate']
    for mode in modes:
        result['warmups'].append(sample(mode));save()
    result['threadpools']=threadpool_info();save()
    for pair in range(3):
        for mode in modes if pair%2==0 else modes[::-1]:
            s=sample(mode);s['pair']=pair
            assert s['physical_result']['status']=='completed'
            assert s['digest']==next(w['digest'] for w in result['warmups'] if w['mode']==mode)
            result['samples'].append(s);save()
    result['medians']={mode:statistics.median(s['wall'] for s in result['samples'] if s['mode']==mode) for mode in modes}
    result['wall_reduction_wc_vs_consistent']=1-result['medians']['work_conjugate']/result['medians']['consistent'];save()
    print(json.dumps({k:result[k] for k in ['medians','wall_reduction_wc_vs_consistent']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);benchmark(p.parse_args().output)
