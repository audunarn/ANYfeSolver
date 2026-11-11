"""Bounded G0B F05 Q4 J2 algorithmic pullback; no product admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if not os.environ.get('NUMBA_CACHE_DIR'):
    raise RuntimeError('set an explicit writable NUMBA_CACHE_DIR before import')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import anyfesolver
import numba
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from anyfesolver._ge_beam3_variational_shell import deformation
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve


REFERENCE=np.array(((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)))
STEPS=(2e-6,2e-7,2e-8)


def measure():
    model=FEModel('g0-f05-q4-algorithmic-pullback-v1')
    curve=DNVC208MaterialCurve(sigma_prop=100e6,sigma_yield=105e6,
                              sigma_yield_2=110e6,eps_p_y1=.005,
                              eps_p_y2=.010,K=400e6,n=.20)
    material=model.add_material('steel',210e9,.3,density=7850.,yield_stress=100e6,
                                hardening_curve=curve)
    for node_id,point in enumerate(REFERENCE,start=1):
        model.add_node(node_id,*point)
    element=create_shell_element(1,[1,2,3,4],'steel',formulation='e4-pl',
                                 thickness=.02,reference_normal=(0.,0.,1.))
    model.add_element(1,element)
    committed=np.zeros(24)
    committed[6]=committed[12]=.004
    committed[8]=.0004;committed[14]=.0007
    local_parent=deformation(REFERENCE,committed)[0]
    _,_,parent_trial=element.compute_nonlinear_response(
        model.mesh,material,local_parent,None,3,True)
    predecessor=element.seal_committed_current_tangent_state(
        model.mesh,material,local_parent,parent_trial,3)
    sealed=canonical_json_bytes(predecessor)

    trial=committed.copy()
    trial[6]+=.0002;trial[12]+=.00015
    trial[8]+=.00004;trial[14]+=.00005
    trial.reshape(4,6)[:,3:]=np.array(((.003,-.002,.001),
                                        (-.001,.002,-.003),
                                        (.001,.001,.002),
                                        (.002,-.001,.003)))

    def evaluate(q,tangent):
        d,D,H=deformation(REFERENCE,q)
        f,k,state=element.compute_nonlinear_response(
            model.mesh,material,d,predecessor,3,tangent)
        g=D.T@f
        if not tangent:return g,None,state
        return g,D.T@k@D+np.einsum('i,ijk->jk',f,H),state

    force,tangent,candidate=evaluate(trial,True)
    alpha_parent=np.asarray(predecessor['alpha'])
    alpha_trial=np.asarray(candidate['alpha'])
    if not all(np.isfinite(values).all() for values in
               (force,tangent,alpha_parent,alpha_trial)):
        raise ValueError('nonfinite Q4 base response or material state')
    base_active_mask=alpha_trial>alpha_parent
    directions=(np.sin(np.arange(24)+.3),
                np.cos(.4+np.arange(24)*.73))
    checks=[]
    for vector in directions:
        v=vector/np.linalg.norm(vector)
        expected=tangent@v
        errors=[]
        mask_stable=[]
        for h in STEPS:
            plus,_,plus_state=evaluate(trial+h*v,False)
            minus,_,minus_state=evaluate(trial-h*v,False)
            if not all(np.isfinite(values).all() for values in
                       (plus,minus,np.asarray(plus_state['alpha']),
                        np.asarray(minus_state['alpha']))):
                raise ValueError('nonfinite Q4 perturbed response or material state')
            numerical=(plus-minus)/(2*h)
            error=float(np.linalg.norm(expected-numerical)/
                        max(1.,np.linalg.norm(numerical)))
            if not np.isfinite(error):
                raise ValueError('nonfinite Q4 directional derivative error')
            errors.append(error)
            mask_stable.append(bool(np.array_equal(
                np.asarray(plus_state['alpha'])>alpha_parent,base_active_mask)
                and np.array_equal(np.asarray(minus_state['alpha'])>alpha_parent,
                                   base_active_mask)))
        checks.append({'direction':v.tolist(),'errors':errors,
                       'sampled_active_mask_stable':mask_stable,
                       'adjacent_plateau':any(a<=1e-5 and b<=1e-5 and
                                              ma and mb for a,b,ma,mb in zip(
                                                  errors,errors[1:],mask_stable,
                                                  mask_stable[1:]))})
    active=bool(np.any(base_active_mask))
    immutable=canonical_json_bytes(predecessor)==sealed
    finite=bool(np.isfinite(force).all() and np.isfinite(tangent).all())
    status=('DIAGNOSTIC_Q4_PASS_NOT_G0B_ACCEPTANCE' if active and immutable
            and finite and all(c['adjacent_plateau'] for c in checks)
            else 'DIAGNOSTIC_Q4_FAIL_NOT_G0B_ACCEPTANCE')
    return {'status':status,'active_plasticity':active,
            'predecessor_unchanged':immutable,'finite':finite,
            'predecessor_alpha_max':float(np.max(alpha_parent)),
            'candidate_alpha_max':float(np.max(alpha_trial)),
            'force_norm':float(np.linalg.norm(force)),
            'relative_skew':float(np.linalg.norm(tangent-tangent.T)/
                                  max(1.,np.linalg.norm(tangent))),
            'steps':STEPS,'directions':checks}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    source=Path(__file__).read_bytes()
    repository=Path(__file__).resolve().parents[1]
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],
                                          cwd=repository,text=True).strip()
    modules=('_ge_beam3_variational_shell.py','_ge_beam3_mixed_ad.py',
             '_ge_beam3_pose_joint.py','e4_pl_element.py','e4_pl_s3_state.py',
             'elements.py','fe_core.py','material_curves.py','plasticity.py')
    module_hashes={name:hashlib.sha256((repository/'src'/'anyfesolver'/name).read_bytes()).hexdigest()
                   for name in modules}
    tracked=['scripts/'+Path(__file__).name]+['src/anyfesolver/'+name for name in modules]
    dirty=subprocess.check_output(['git','status','--porcelain','--',*tracked],
                                  cwd=repository,text=True).strip()
    if dirty:
        raise RuntimeError('frozen F05 probe or mechanical source changed')
    manifest={'purpose':'F05_Q4_FIXED_PREDECESSOR_DIAGNOSTIC_NOT_G0B',
              'source_sha256':hashlib.sha256(source).hexdigest(),
              'source_commit':source_commit,'module_sha256':module_hashes,
              'anyfesolver_import':str(Path(anyfesolver.__file__).resolve()),
              'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
              'numba':numba.__version__,'blas_threads':1,
              'numba_cache':os.environ['NUMBA_CACHE_DIR'],
              'geometry_m':REFERENCE.tolist(),'thickness_m':.02,'layers':3,
              'directions':('sin(i+.3)','cos(.4+.73*i)'),
              'finite_difference_steps':STEPS,'relative_tolerance':1e-5}
    args.root.mkdir(parents=True,exist_ok=False)
    (args.root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',
                                          encoding='utf-8')
    try:
        with threadpool_limits(limits=1):
            result=measure()
    except Exception as exc:
        result={'status':'DIAGNOSTIC_Q4_EXCEPTION_NOT_G0B_ACCEPTANCE',
                'failure_type':type(exc).__name__,'failure':str(exc)}
    (args.root/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2,
                                                   allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],
                      'errors':[x['errors'] for x in result.get('directions',[])],
                      'failure':result.get('failure')},sort_keys=True),flush=True)
    return 0 if result['status']=='DIAGNOSTIC_Q4_PASS_NOT_G0B_ACCEPTANCE' else 1


if __name__=='__main__':
    raise SystemExit(main())
