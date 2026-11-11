"""Warmed assembly-only timing of exact solve-local impact baseline reuse."""
import argparse,hashlib,json,os,pathlib,platform,statistics,time
from contextlib import nullcontext
import numpy as np
from threadpoolctl import threadpool_info
from anyfesolver import FEModel,e4_pl_element
from anyfesolver.elements import create_shell_element
from anyfesolver.nonlinear_static import _assemble_nonlinear_system

def model():
    m=FEModel('impact-baseline-reuse-assembly');m.add_material('soft',1e5,.3,density=20.)
    for j in range(3):
        for i in range(3):m.add_node(1+3*j+i,i/2,j/2,0.)
    for j in range(2):
        for i in range(2):
            n=1+3*j+i;eid=1+2*j+i
            m.add_element(eid,create_shell_element(eid,[n,n+1,n+4,n+3],'soft',thickness=.05))
    m.apply_boundary_conditions();return m

def run(out):
    root=pathlib.Path(__file__).resolve().parents[1]
    sources=[pathlib.Path(__file__),root/'src/anyfesolver/e4_pl_element.py',root/'src/anyfesolver/contact.py',root/'src/anyfesolver/elements.py']
    result={'scope':'assembly only, 2x2 Q4, three hot assemblies per sample; identical current code with vs without solve-local scope','python':platform.python_version(),'numpy':np.__version__,'module':e4_pl_element.__file__,'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']},'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'samples':[]}
    def save():out.write_text(json.dumps(result,indent=2))
    m=model();u=np.zeros(m.mesh.dof_manager.total_dofs);u[2::6]=.003*np.sin(np.arange(9))
    def assemble():return _assemble_nonlinear_system(m,u,{},5,tangent=True,require_full_coordinates=True)
    reference=assemble();result['warmup_completed']=True;result['threadpools']=threadpool_info();save()
    for pair in range(3):
        for cached in ([False,True] if pair%2==0 else [True,False]):
            with e4_pl_element._elastic_batch_baseline_scope() if cached else nullcontext():
                assemble()
                start=time.perf_counter()
                for _ in range(3): actual=assemble()
                elapsed=time.perf_counter()-start
            exact=bool(np.array_equal(reference[0],actual[0]) and np.array_equal(reference[1].toarray(),actual[1].toarray()))
            assert exact
            result['samples'].append({'pair':pair,'cached':cached,'wall':elapsed,'exact_force_tangent':exact});save()
    old=statistics.median(s['wall'] for s in result['samples'] if not s['cached']);new=statistics.median(s['wall'] for s in result['samples'] if s['cached'])
    result.update(median_uncached=old,median_cached=new,reduction=1-new/old);save();print(json.dumps({k:result[k] for k in ['median_uncached','median_cached','reduction']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);run(p.parse_args().output)
