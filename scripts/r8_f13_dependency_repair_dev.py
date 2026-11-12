"""Private bounded F13 dependency capture; never replays the frozen campaign."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/agent_plans/development/r8-f13-static-feasibility-v5-20261009'
EVIDENCE = ROOT / '.development/r8-f13-dependency-repair-20261009'
PRODUCT = 'anyfesolver/_stiffened_shell_b2_q4_panel_experimental.py'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str)+'\n', encoding='utf-8')

def bounded(name, command, *, reserve=True):
    """Own one child tree and retain exact elapsed accounting and diagnostics."""
    import psutil
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ledger_path = EVIDENCE / 'ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'status':'OPEN', 'budget_s':240., 'reserve_s':60., 'used_s':0., 'attempts':[]}
    if ledger.get('status', 'OPEN') != 'OPEN':
        raise RuntimeError('F13 dependency repair ledger is closed')
    cap = min(30., ledger['budget_s']-ledger['used_s']-(ledger['reserve_s'] if reserve else 0.))
    if cap <= 0: raise RuntimeError('fresh F13 allowance exhausted')
    out = EVIDENCE/name
    out.mkdir()
    env = os.environ.copy()
    temporary=out/'tmp';temporary.mkdir()
    settings = dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMBA_DISABLE_JIT='1',PYTHONOPTIMIZE='0',PYTHONNOUSERSITE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',TEMP=str(temporary),TMP=str(temporary))
    env.update(settings)
    psutil.Process().cpu_affinity([0])
    receipt = dict(command=command,cwd=str(ROOT),environment=settings,cap_s=cap,prior_used_s=ledger['used_s'],memory_cap_bytes=2147483648,script_sha256=sha(__file__),input_sha256=sha(DOC/'inputs-v5.json'),product_sha256=sha(ROOT/'src'/PRODUCT))
    save(out/'receipt.json',receipt)
    failure=None; peak=0; start=time.perf_counter_ns()
    with (out/'stdout.log').open('wb') as stdout, (out/'stderr.log').open('wb') as stderr:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        proc=psutil.Process(child.pid); proc.cpu_affinity([0])
        while child.poll() is None:
            try: members=[proc]+proc.children(recursive=True)
            except psutil.NoSuchProcess: members=[]
            rss=0
            for member in members:
                try:rss+=member.memory_info().rss
                except psutil.NoSuchProcess:pass
            peak=max(peak,rss)
            if rss>2147483648:failure='PROCESS_TREE_MEMORY_LIMIT'
            if (time.perf_counter_ns()-start)/1e9>=cap:failure='PROCESS_TREE_TIMEOUT'
            if failure:
                for member in reversed(members):
                    try:member.kill()
                    except psutil.NoSuchProcess:pass
                break
            time.sleep(.01)
        code=child.wait()
    wall=(time.perf_counter_ns()-start)/1e9
    ledger['used_s']+=wall; ledger['attempts'].append(dict(name=name,wall_s=wall,exit_code=code,failure=failure))
    save(ledger_path,ledger)
    receipt.update(wall_s=wall,exit_code=code,failure=failure,peak_sampled_tree_rss_bytes=peak,cumulative_used_s=ledger['used_s'],remaining_s=240.-ledger['used_s'])
    save(out/'receipt.json',receipt)
    print(json.dumps(dict(attempt=name,wall_s=wall,exit_code=code,failure=failure,used_s=ledger['used_s'])))
    return code or bool(failure)

def capture(site, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(site).resolve()));sys.path.insert(1,str(ROOT))
    import numpy as np
    import psutil
    from scipy import sparse
    from threadpoolctl import threadpool_info
    from types import SimpleNamespace
    from anyfesolver import stiffened_shell as api
    from anyfesolver import _stiffened_shell_b2_q4_panel_experimental as module
    from anyfesolver.control import CancellationToken
    spec=importlib.util.spec_from_file_location('f13_frozen_ownership',ROOT/'.development/r8-f13-static-feasibility-v5-20261009/runner.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    inputs=json.loads((DOC/'inputs-v5.json').read_text())
    ctx=SimpleNamespace(inputs=inputs,np=np,sparse=sparse,api=api,module=module,Token=CancellationToken)
    imports={name:str(Path(m.__file__).resolve()) for name,m in sys.modules.items() if name.startswith('anyfesolver') and getattr(m,'__file__',None)}
    assert all(Path(p).is_relative_to(Path(site).resolve()) for p in imports.values())
    runtime=dict(affinity=psutil.Process().cpu_affinity(),optimization=sys.flags.optimize,jit=os.environ.get('NUMBA_DISABLE_JIT'),pools=threadpool_info(),imports=imports,import_sha256={n:sha(p) for n,p in imports.items()},input_sha256=sha(DOC/'inputs-v5.json'),frozen_runner_sha256=sha(frozen.__file__))
    assert runtime['affinity']==[0] and runtime['optimization']==0 and runtime['jit']=='1'
    assert all(p['num_threads']==1 for p in runtime['pools'])
    save(out/'runtime.json',runtime)
    original=frozen.legacy.definition;active={};counts={};records=[]
    def definition(context,domain,variant=None):
        active.clear();active.update(domain=domain,variant=variant or 'base',input_sha256=sha(DOC/'inputs-v5.json'))
        save(out/'active-variant.json',active)
        return original(context,domain,variant)
    frozen.legacy.definition=definition
    module_path=str(Path(module.__file__).resolve())
    guard_line=next(i for i,line in enumerate(Path(module_path).read_text().splitlines(),1) if 'raise _NumericalFailure("panel point gap or equilibrium failed")' in line)
    def tracer(frame,event,arg):
        if frame.f_code.co_filename!=module_path or frame.f_code.co_name!='solve_linear':return tracer
        if event=='line' and frame.f_lineno==guard_line:
            local=frame.f_locals
            keys=('answer','rhs','u','lam','gap','reaction','projected','gap_scale','physical_force_scale','equilibrium_scale','projected_roundoff','equilibrium_tolerance','gap_tolerance','explicit_rhs','resolution_floor')
            arrays={key:np.array(local[key],copy=True) for key in keys}
            np.savez(out/('guard-'+active['variant']+'.npz'),**arrays)
            save(out/('guard-'+active['variant']+'.json'),dict(active=active.copy(),producer=module_path,line=guard_line,producer_sha256=sha(module_path),roundoff_fraction=local['roundoff_fraction'],certified=local['certified'],gap_failed_rows=np.flatnonzero(np.abs(arrays['gap'])>arrays['gap_tolerance']).tolist(),equilibrium_failed_rows=np.flatnonzero(np.abs(arrays['projected'])>arrays['equilibrium_tolerance']).tolist(),nonfinite=[k for k,a in arrays.items() if not np.isfinite(a).all()],predicate='nonfinite OR abs(gap)>gap_tolerance OR abs(projected)>equilibrium_tolerance'))
        if event=='return' and hasattr(arg,'status'):
            variant=active['variant'];count=counts.get(variant,0)+1;counts[variant]=count
            save(out/f'public-{variant}-{count}.json',dataclasses.asdict(arg))
            records.append(dict(variant=variant,status=arg.status,failure_reason=arg.failure_reason))
        return tracer
    sys.settrace(tracer)
    try:
        result=frozen.ownership_case(ctx,'physical-footprint')
        save(out/'ownership-result.json',result);status='PASS';failure=None
    except Exception as exc:
        status='FAIL';failure=f'{type(exc).__name__}: {exc}'
    finally:sys.settrace(None);frozen.legacy.definition=original
    save(out/'summary.json',dict(status=status,failure=failure,active=active,public_records=records))
    print(json.dumps(dict(status=status,failure=failure,variant=active.get('variant'),public_count=len(records))))
    return int(status!='PASS')

def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',action='store_true');p.add_argument('--site');p.add_argument('--out');p.add_argument('--attempt');p.add_argument('--verification',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.capture:return capture(a.site,a.out)
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    return bounded(a.attempt,command,reserve=not a.verification)

if __name__=='__main__':raise SystemExit(main())
