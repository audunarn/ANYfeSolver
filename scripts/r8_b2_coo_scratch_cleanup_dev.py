"""Bounded development caller/build/install checks using existing supervisor."""
import argparse
import base64
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/agent_plans/development/r8-b2-coo-scratch-cleanup-20261010'
ENV={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
 'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMBA_NUM_THREADS','NUMBA_DISABLE_JIT')};ENV['PYTHONOPTIMIZE']='0'
TESTS=['tests/test_r8_b2_coo_scratch_cleanup_dev.py','tests/test_r8_b2_assembly_metrics_dev.py','tests/test_stiffened_shell_b2_physical_footprint_batch.py','tests/test_r8_b2_assembly_series_dev.py']
OWNED=['src/anyfesolver/_stiffened_shell_b2_q4_panel_experimental.py',
 'src/anyfesolver/_stiffened_shell_b2_assembly_diagnostics.py','tests/test_r8_b2_coo_scratch_cleanup_dev.py',
 'scripts/r8_b2_coo_scratch_cleanup_dev.py']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False,default=str)+'\n')
def require(ok,m):
 if not ok:raise RuntimeError(m)

def verify_install(site,a):
 record=[n for n in a['package'] if n.endswith('.dist-info/RECORD')]
 require(len(record)==1,'one installed RECORD');record=record[0]
 metadata=record.rsplit('/',1)[0]
 extras={metadata+'/'+n for n in ('direct_url.json','INSTALLER','REQUESTED')}
 actual={q.relative_to(site).as_posix() for q in site.rglob('*') if q.is_file()}
 require(actual==set(a['package'])|extras,'installed file inventory mismatch')
 require(all(sha(site/n)==h for n,h in a['package'].items() if n!=record),'installed product bytes mismatch')
 rows=list(csv.reader((site/record).read_text().splitlines()));seen=set()
 for name,digest,size in rows:
  require(name in actual and name not in seen,'installed RECORD path/duplicate');seen.add(name)
  if name==record:require(digest==size=='','RECORD self entry')
  else:
   data=(site/name).read_bytes();expected='sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode('ascii')
   require(digest==expected and size==str(len(data)),'installed RECORD digest/size '+name)
 require(seen==actual,'installed RECORD completeness')
 return dict(package_files=len(a['package']),wheel_unchanged_files=len(a['package'])-1,
  installed_files={n:sha(site/n) for n in sorted(actual)},installer_generated=sorted(extras),rewritten_record=record)

def freeze():
 inputs=OWNED+TESTS[1:]+['scripts/r8_b2_assembly_series_dev.py','examples/stiffened_shell_b2_physical_rigid_footprint.py','pyproject.toml',
  'scripts/_b3_e4_f03_local_scientific_supervisor.py','scripts/_b3_e4_f03_local_scientific_cases.py']
 p=dict(source={n:sha(ROOT/n) for n in inputs},all_solver_source={p.relative_to(ROOT/'src').as_posix():sha(p)
  for p in (ROOT/'src/anyfesolver').rglob('*.py')},tests=TESTS,scope='SMALL_DEVELOPMENT_ONLY',caps=dict(objects=2000000,edges=8000000,buffers=500000,depth=40))
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'PREPARED.json').exists():
  version=OUT/'preparations';version.mkdir(exist_ok=True);old=OUT/'PREPARED.json'
  (version/(sha(old)+'.json')).write_bytes(old.read_bytes())
 save(OUT/'PREPARED.json',p);return p

def runtime(p,installed=False,scientific=False):
 import psutil
 r=dict(affinity=psutil.Process().cpu_affinity(),optimize=sys.flags.optimize,executable=sys.executable,
  environment={k:os.environ.get(k) for k in ENV},imports={})
 require(r['affinity']==[0] and r['optimize']==0 and r['environment']==ENV,'effective basic runtime')
 if scientific:
  import numba
  from threadpoolctl import threadpool_info
  r.update(numba=dict(disable_jit=int(numba.config.DISABLE_JIT),threads=numba.get_num_threads()),pools=threadpool_info())
  require(r['numba']==dict(disable_jit=1,threads=1) and r['pools'] and all(x['num_threads']==1 for x in r['pools']),'effective numerical runtime')
  base=(OUT/'package/site' if installed else ROOT/'src').resolve()
  expected=p['package'] if installed else p['all_solver_source']
  for n,m in tuple(sys.modules.items()):
   if n.split('.')[0]=='anyfesolver' and getattr(m,'__file__',None):
    path=Path(m.__file__).resolve();require(path.is_relative_to(base),'wrong solver import '+n)
    rel=path.relative_to(base).as_posix();require(rel in expected and sha(path)==expected[rel],'import hash '+n)
    r['imports'][n]=dict(path=str(path),sha256=sha(path))
 return r

def child(phase,attempt):
 import psutil
 psutil.Process().cpu_affinity([0]);p=json.loads((attempt/'binding.json').read_text())['preparation']
 save(attempt/'runtime-before.json',runtime(p))
 try:
  if phase=='build':
   wheels=OUT/'package/wheels';wheels.mkdir(parents=True,exist_ok=True)
   command=[sys.executable,'-m','pip','wheel','--no-index','--no-deps','--no-build-isolation','--no-cache-dir','--wheel-dir',str(wheels),str(ROOT)]
   save(attempt/'build-command.json',command);require(subprocess.run(command,check=False).returncode==0,'wheel build failed')
   found=list(wheels.glob('*.whl'));require(len(found)==1,'one isolated candidate wheel')
   wheel=found[0]
   with ZipFile(wheel) as z:package={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/')}
   save(OUT/'ARTIFACT.json',dict(wheel=str(wheel),sha256=sha(wheel),package=package))
   result=dict(status='PASS_BUILD')
  elif phase=='install':
   a=json.loads((OUT/'ARTIFACT.json').read_text());require(sha(a['wheel'])==a['sha256'],'wheel drift')
   site=OUT/'package/site'
   command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-compile','--target',str(site),a['wheel']]
   if site.exists():
    prior=OUT/'attempt-install-01'
    require(json.loads((prior/'result.json').read_text())==dict(status='FAILED',error='installed bytes mismatch'),'only known metadata adjudication')
    require(json.loads((prior/'install-command.json').read_text())==command,'installed predecessor command drift')
    save(attempt/'reused-install.json',dict(command=command,prior_receipt_sha256=sha(prior/'receipt.json'),prior_result_sha256=sha(prior/'result.json'),reinstalled=False))
   else:
    save(attempt/'install-command.json',command);require(subprocess.run(command,check=False).returncode==0,'install failure')
   inventory=verify_install(site,a);save(attempt/'installed-inventory.json',inventory)
   result=dict(status='PASS_INSTALL',package_files=len(a['package']),reinstalled=(attempt/'install-command.json').exists())
  else:
   installed=phase=='installed';base=OUT/'package/site' if installed else ROOT/'src'
   sys.path[:]=[str(base),str(ROOT)]+[x for x in sys.path if Path(x or '.').resolve() not in (ROOT/'src',Path('C:/Github/ANYfeSolver/src'))]
   import numpy,scipy,numba,pytest
   save(attempt/'runtime-before.json',runtime(p,installed,True))
   class Counts:
    def __init__(self):self.passed=self.failed=self.skipped=0
    def pytest_runtest_logreport(self,report):
     if report.when=='call':
      self.passed+=report.passed;self.failed+=report.failed;self.skipped+=report.skipped
     elif report.failed:self.failed+=1
   selected=TESTS
   counts=Counts();code=pytest.main(['-q','-o','pythonpath=','--junitxml='+str(attempt/'junit.xml'),
    *(str(ROOT/n) for n in selected)],plugins=[counts])
   result=dict(status='PASS_CONTROLS' if code==0 and counts.passed==42 and counts.failed==counts.skipped==0 else 'FAILED_CONTROLS',
    passed=counts.passed,failed=counts.failed,skipped=counts.skipped,exit_code=code)
   save(attempt/'runtime-after.json',runtime(p,installed,True))
  save(attempt/'result.json',result)
  if phase in ('build','install'):save(attempt/'runtime-after.json',runtime(p))
  return result['status'].startswith('PASS')
 except BaseException as exc:
  (attempt/'failure.txt').write_text(traceback.format_exc());save(attempt/'result.json',dict(status='FAILED',error=str(exc)));return False

def run(phase):
 a=json.loads((OUT/'ROOT_DEVELOPMENT_AUTHORITY.json').read_text());require(a['status']=='AUTHORIZED' and a['allowance_seconds']==180 and a['reserve_seconds']==60,'fresh authority')
 p=json.loads((OUT/'PREPARED.json').read_text());require(all(sha(ROOT/n)==h for n,h in p['source'].items()),'source drift')
 if phase=='installed':p.update(json.loads((OUT/'ARTIFACT.json').read_text()))
 ledger_path=OUT/'ledger.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(status='OPEN',used_seconds=0.,attempts=[])
 require(ledger['status']=='OPEN','closed ledger');remaining=180.-60.-ledger['used_seconds'];cap=min(30.,remaining);require(cap>0,'reserve exhausted')
 ordinal=1+sum(x['phase']==phase for x in ledger['attempts']);attempt=OUT/f'attempt-{phase}-{ordinal:02d}';attempt.mkdir(exist_ok=False)
 env=dict(os.environ);env.update(ENV);env['PYTHONPATH']=str(OUT/'package/site' if phase=='installed' else ROOT/'src')
 env['B2_DIAGNOSTIC_EVIDENCE']=str(attempt)
 temp=attempt/'temp';temp.mkdir();tracker=temp/'build-tracker';tracker.mkdir()
 env.update(TEMP=str(temp),TMP=str(temp),PIP_BUILD_TRACKER=str(tracker),PIP_DISABLE_PIP_VERSION_CHECK='1')
 command=[sys.executable,'-u',str(Path(__file__).resolve()),'--child',phase,str(attempt)]
 save(attempt/'binding.json',dict(command=command,cwd=str(attempt),preparation=p,authority_sha256=sha(OUT/'ROOT_DEVELOPMENT_AUTHORITY.json'),environment={k:env[k] for k in (*ENV,'PYTHONPATH','TEMP','TMP','PIP_BUILD_TRACKER','PIP_DISABLE_PIP_VERSION_CHECK')}))
 for n,h in p['source'].items():
  dest=attempt/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/n).read_bytes());require(sha(dest)==h,'snapshot mismatch')
 sys.path.insert(0,str(ROOT));from scripts._b3_e4_f03_local_scientific_supervisor import supervise,SupervisorLimits
 save(ledger_path,ledger)
 with (attempt/'stdout.txt').open('w') as stdout,(attempt/'stderr.txt').open('w') as stderr:
  receipt=supervise(command,cwd=attempt,environment=env,stdout=stdout,stderr=stderr,limits=SupervisorLimits(cap,2*1024**3,.05))
 save(attempt/'receipt.json',asdict(receipt));result=json.loads((attempt/'result.json').read_text()) if (attempt/'result.json').exists() else {}
 passed=receipt.exit_code==0 and receipt.reason=='completed' and result.get('status','').startswith('PASS')
 ledger['used_seconds']+=receipt.elapsed_seconds;ledger['attempts'].append(dict(phase=phase,passed=passed,receipt=asdict(receipt),candidate_sha256=sha(OUT/'PREPARED.json')))
 save(ledger_path,ledger);print(json.dumps(dict(phase=phase,passed=passed,used_seconds=ledger['used_seconds'],remaining_seconds=180-ledger['used_seconds'])));return passed

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',choices=('source','build','install','installed'));parser.add_argument('--child',nargs=2);args=parser.parse_args()
 if args.prepare:freeze()
 elif args.child:raise SystemExit(0 if child(args.child[0],Path(args.child[1])) else 1)
 elif args.run:raise SystemExit(0 if run(args.run) else 1)
 else:parser.error('prepare or bounded authorized phase')
