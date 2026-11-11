"""Current-artifact installed B2 remaining scale slice: full111 ordered 1/10/100
plus conditional registered beam-rich control.

Reuses scripts/r8_b2_installed_batch_persistence_dev.py execute_installed and its
existing consumer/builders/validators/profile counters/accepted-prefix unchanged.
The completed V9 assembly series is not rerun; its verified 761-file installed
site is reused as the binding current artifact (solver 00385aa8, consumer
84b73806). The installed graph is checked against the V9-verified inventory
before/after every phase; no duplicate raw-wheel RECORD re-derivation.
Preparation is stdlib-only; every executing phase needs explicit root authority.
No backend override, solver-state checkpoint, qualification or performance
claim; the aggregate G5 mechanically identical warm baseline stays open.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/agent_plans/development/r8-b2-installed-batch-persistence-20261010'
OUT=BASE/'v4-remaining'
V9=ROOT/'docs/agent_plans/development/r8-b2-assembly-series-20261010/execution-v9'
PRIOR=ROOT/'scripts/r8_b2_installed_batch_persistence_dev.py'
ORIGINAL=ROOT/'scripts/r8_b2_installed_original_target_dev.py'
CONTROL=ROOT/'tests/test_r8_b2_installed_batch_persistence_dev.py'
TEST=ROOT/'tests/test_r8_b2_remaining_scale_dev.py'
SOLVER_SHA='00385aa8dcfaa472b1f0860257cf83baad263db958a250b0d0c22e9ebe0bff15'
CONSUMER_SHA='84b738061f0627da5c68fb2e8efb0a0882a2c19ce0f1750595050923f58bd66b'
FULL_SHA='cf7970c30839256a2eabc0f9fc2a5521e61b641b92b4a6bca2ee68a8ab70f8b4'
ENV={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS',
    'NUMBA_NUM_THREADS','NUMBA_DISABLE_JIT')}
ENV['PYTHONOPTIMIZE']='0'
PHASES={'controls':20.,'full':240.,'beam':60.}
ALLOWANCE=420.
RSS_BYTES=2*1024**3
G5_NOTE=('aggregate G5 warm <=5% time / <=10% memory mechanically identical baseline '
    'is missing on 00385aa8; this slice does not close aggregate G5')


def harness():
    """Load the consumed installed-batch-persistence runner for reuse only."""
    spec=importlib.util.spec_from_file_location('b2_installed_batch_persistence_harness',PRIOR)
    result=importlib.util.module_from_spec(spec);sys.modules[spec.name]=result;spec.loader.exec_module(result)
    return result


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False,default=str)+'\n',encoding='utf-8')


def product_package(inventory):
    """PRODUCT namespace map (anyfesolver/anyfem) only; never a raw wheel RECORD."""
    package={n:h for n,h in inventory['files'].items() if n.startswith(('anyfesolver/','anyfem/'))}
    require(package and not any('.dist-info' in n or n.startswith('bin/') for n in package),
        'product namespace contamination')
    return package


def graph_mismatch(site,inventory):
    """Exact installed-file closure/bytes drift against the V9-verified graph."""
    site=Path(site).resolve();expected=dict(inventory['files']);problems=[];actual=set()
    for q in site.rglob('*'):
        if q.is_symlink() or getattr(q,'is_junction',lambda:False)():
            problems.append('reparse '+q.name);continue
        if q.is_file():actual.add(q.relative_to(site).as_posix())
    for name in sorted(set(expected)-actual):problems.append('missing '+name)
    for name in sorted(actual-set(expected)):problems.append('extra '+name)
    for name in sorted(set(expected)&actual):
        if sha(site/name)!=expected[name]:problems.append('bytes '+name)
    return problems


def v9_bindings():
    plan=json.loads((V9/'EXECUTABLE_PREPARED.json').read_text())
    require(plan['solver_sha256']==SOLVER_SHA and plan['consumer_sha256']==CONSUMER_SHA,
        'V9 current artifact identity')
    require(plan['installation']==dict(reuse_install=False,bytecode_disabled=True),
        'V9 fresh no-cache installation contract')
    receipt=json.loads((V9/'attempt-install-01/receipt.json').read_text())
    result=json.loads((V9/'attempt-install-01/result.json').read_text())
    require(receipt['reason']=='completed' and receipt['exit_code']==0
        and result['status']=='PASS_FRESH_INSTALL','V9 installation PASS required')
    ledger=json.loads((V9/'SERIES_LEDGER.json').read_text())
    require(ledger['status']=='CLOSED','V9 series ledger must remain closed')
    inventory=json.loads((V9/'attempt-install-01/installed-inventory.json').read_text())
    require(inventory['reinstalled'] is False and inventory['launchers']
        and inventory['interpretation'],'V9 installed graph shape')
    return plan,inventory


def verify_installed_graph(p,label,attempt):
    """Before/after strict site check reusing the V9-verified inventory.

    The inventory already covers non-RECORD wheel bytes, the actual RECORD and
    the declared launcher roles; raw wheels are not re-read here.
    """
    inventory=json.loads((V9/'attempt-install-01/installed-inventory.json').read_text())
    require(sha(V9/'attempt-install-01/installed-inventory.json')==p['installed_graph']['inventory_sha256'],
        'V9 installed inventory drift')
    require(inventory['launchers']==p['installed_graph']['launchers'],'declared launcher roles drift')
    problems=graph_mismatch(p['site'],inventory)
    require(not problems,'installed site graph drift: '+', '.join(problems[:8]))
    save(attempt/('installed-graph-'+label+'.json'),dict(label=label,files=len(inventory['files']),
        inventory_sha256=p['installed_graph']['inventory_sha256'],launchers=inventory['launchers'],
        interpretation=inventory['interpretation'],reused_v9_verified_graph=True,
        raw_wheel_record_recheck=False))


def connected_caller_ast():
    """Static admission that the reused harness keeps the connected public path."""
    tree=ast.parse(PRIOR.read_text())
    names={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
    names|={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    required=('solve_linear_many','close','create_analysis','definition','from_definition',
        'save_case','load_case','save_project','load_project','point_panel_from_analysis')
    require(all(name in names for name in required),'reused connected public caller path missing')
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_installed')
    return dict(status='REUSED_CONNECTED_CALLER_AST',required=list(required),
        execute_installed_line=function.lineno,not_execution_evidence=True)


def matrix_contracts():
    original=harness().module('b2_public_static_identity',ORIGINAL)
    return dict(original.EXPECTED_MATRICES)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan,inventory=v9_bindings()
    solver=Path(plan['solver_wheel']);consumer=Path(plan['consumer_wheel'])
    require(solver.is_file() and consumer.is_file(),'V9 bound wheels missing')
    require(sha(solver)==SOLVER_SHA and sha(consumer)==CONSUMER_SHA,'adopted wheel identity')
    h=harness()
    require(sha(h.BUILDER)==json.loads((h.BUILDER.parent/'preflight.json').read_text())['builder_sha256'],
        'original builder drift')
    require(h.BEAM_BUILDER.is_file(),'registered beam-rich builder missing')
    package=product_package(inventory)
    original=h.module('b2_public_static_identity',ORIGINAL)
    require(all('anyfesolver/'+name in package for name in original.B2_FILES),
        'product package closure missing B2 files')
    problems=graph_mismatch(plan['package_site'],inventory)
    require(not problems,'installed site graph drift before preparation: '+', '.join(problems[:8]))
    prior_full=BASE/'v2/attempt-full-01';prior_beam=BASE/'v3-beam/attempt-beam-01'
    full=json.loads((prior_full/'result.json').read_text());beam=json.loads((prior_beam/'result.json').read_text())
    require(full['status']=='PASS_INSTALLED_FUNCTIONAL' and full['validated']==111
        and beam['status']=='PASS_INSTALLED_FUNCTIONAL' and beam['primary_count']==100,
        'prior analogous functional evidence required')
    paths=(Path(__file__),TEST,PRIOR,CONTROL,ORIGINAL,h.BUILDER,h.BEAM_BUILDER,
        ROOT/'scripts/_b3_e4_f03_local_scientific_supervisor.py',
        ROOT/'scripts/_b3_e4_f03_local_scientific_cases.py',
        V9/'attempt-install-01/installed-inventory.json',V9/'EXECUTABLE_PREPARED.json',
        V9/'SERIES_LEDGER.json',V9/'attempt-install-01/receipt.json',
        V9/'attempt-install-01/result.json',V9/'EXECUTION_REPORT.md')
    for path in (Path(__file__),TEST):ast.parse(path.read_text())
    p=dict(status='PREPARED_SLOT_HELD',solver_wheel=str(solver.resolve()),consumer_wheel=str(consumer.resolve()),
        solver_sha256=SOLVER_SHA,consumer_sha256=CONSUMER_SHA,package=package,
        site=str((V9/'package/site').resolve()),
        installed_graph=dict(inventory_sha256=sha(V9/'attempt-install-01/installed-inventory.json'),
            files=len(inventory['files']),launchers=inventory['launchers'],
            interpretation=inventory['interpretation'],
            install_receipt_sha256=sha(V9/'attempt-install-01/receipt.json'),
            install_result_sha256=sha(V9/'attempt-install-01/result.json'),
            series_ledger_sha256=sha(V9/'SERIES_LEDGER.json'),
            executable_prepared_sha256=sha(V9/'EXECUTABLE_PREPARED.json'),
            execution_report_sha256=sha(V9/'EXECUTION_REPORT.md'),
            reused_v9_verified_graph=True,raw_wheel_record_recheck=False,bytecode_disabled=True),
        source={str(path.relative_to(ROOT)):sha(path) for path in paths},
        matrix_sha256=matrix_contracts(),
        full=dict(nx=100,ny=100,shells=10000,members=1000,pads=100,incidences=400,rows=2000,
            dofs=67266,free=66660,batch_sizes=[1,10,100],primary_count=111,
            selected_dofs=[30902,30900],selected_pad='pad-0-5',selected_member='b2-0-0',
            stations=[0.,.5,1.],load_coordinate=30902,canonical_sha256=FULL_SHA),
        beam=dict(nx=20,ny=20,shells=400,members=1000,pads=100,incidences=400,rows=2000,
            batch_sizes=[100],primary_count=100),
        prior_analogous_pass=dict(artifact_solver_sha256=h.SOLVER_SHA,
            full_result_sha256=sha(prior_full/'result.json'),full_receipt_sha256=sha(prior_full/'receipt.json'),
            beam_result_sha256=sha(prior_beam/'result.json'),beam_receipt_sha256=sha(prior_beam/'receipt.json'),
            scope='historical evidence on the prior artifact only, never execution authority'),
        caller=connected_caller_ast(),consumer_choice_required='frozen84b738-B2-route-GE-delta-explicit',
        persistence='experimental physical sparse-v2 standalone AND nativeProject definition-only',
        policy='unchanged eligible current GENERAL/default; no factor/backend override',
        limits=dict(total_seconds=ALLOWANCE,reserve_seconds=60.,phase_seconds=PHASES,rss_bytes=RSS_BYTES,
            cpu=[0],threads=1,jit_disabled=True,optimize=0,one_job=True,process_cap_seconds=600.),
        g5_aggregate=dict(warm_baseline='missing on 00385aa8',closes_aggregate_g5=False,note=G5_NOTE),
        qualification=False,performance=False,checkpoint=False,
        worktree_source_assertion='none; the reused V9 installed artifact is the binding current artifact')
    save(OUT/'PREPARED.json',p)
    return p


def assert_prepared(p):
    require(sha(p['solver_wheel'])==p['solver_sha256'] and sha(p['consumer_wheel'])==p['consumer_sha256'],
        'wheel drift')
    for name,value in p['source'].items():require(sha(ROOT/name)==value,'prepared source drift '+name)
    require(sha(V9/'attempt-install-01/installed-inventory.json')==p['installed_graph']['inventory_sha256'],
        'V9 installed inventory drift')
    for name,value in p['package'].items():
        require(sha(Path(p['site'])/name)==value,'installed product bytes drift '+name)


def child(phase,attempt):
    p=json.loads((attempt/'binding.json').read_text())['preparation'];assert_prepared(p)
    import psutil
    psutil.Process().cpu_affinity([0])
    basic=dict(affinity=psutil.Process().cpu_affinity(),environment={k:os.environ.get(k) for k in ENV},
        optimize=sys.flags.optimize,executable=sys.executable,python=sys.version,phase=phase)
    require(basic['affinity']==[0] and basic['environment']==ENV and basic['optimize']==0,
        'basic child runtime refusal')
    save(attempt/'runtime-basic.json',basic)
    verify_installed_graph(p,'before',attempt)
    if phase=='controls':
        import unittest
        suites={}
        for name in (CONTROL,TEST):
            suite=unittest.defaultTestLoader.discover(str(name.parent),pattern=name.name)
            result=unittest.TextTestRunner(verbosity=2).run(suite)
            suites[name.name]=dict(tests=result.testsRun,errors=len(result.errors),
                failures=len(result.failures))
            require(result.wasSuccessful(),'control suite failed '+name.name)
        save(attempt/'result.json',dict(status='PASS_CONTROLS',suites=suites))
        verify_installed_graph(p,'after',attempt)
        save(attempt/'runtime-after.json',dict(basic,affinity=psutil.Process().cpu_affinity()))
        return True
    h=harness()
    passed=h.execute_installed(phase,attempt,p)
    if phase=='full' and passed:
        record=json.loads((attempt/'result.json').read_text());persistence=record.get('persistence',{})
        require(persistence.get('canonical_sha256')==FULL_SHA and persistence.get('standalone_sha256')==FULL_SHA,
            'current-artifact full canonical identity')
        require(record.get('validated')==111 and record.get('primary_count')==111,'full111 counters')
    verify_installed_graph(p,'after',attempt)
    return passed


def predecessors(phase):
    """Beam-rich stays conditional on the current-artifact full111 PASS."""
    return {'controls':(),'full':('controls',),'beam':('full',)}[phase]


def run(phase,authority):
    a=json.loads(Path(authority).read_text());p=json.loads((OUT/'PREPARED.json').read_text());assert_prepared(p)
    require(a['status']=='AUTHORIZED' and a['numerical_slot_transferred'] is True
        and a['prepared_sha256']==sha(OUT/'PREPARED.json') and phase in a['phases'],
        'root reviewed authority/slot required')
    require(a['consumer_choice']==p['consumer_choice_required'],'unadopted consumer artifact delta')
    require(a['reused_v9_install_receipt_sha256']==p['installed_graph']['install_receipt_sha256'],
        'root verified reused V9 install contract required')
    ledger_path=OUT/'ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(status='OPEN',
        allowance_seconds=ALLOWANCE,reserve_seconds=60.,used_seconds=0.,attempts=[])
    require(ledger['status']=='OPEN' and not any(item['phase']==phase for item in ledger['attempts']),
        'closed/consumed phase')
    for previous in predecessors(phase):
        require(any(x['phase']==previous and x['passed'] for x in ledger['attempts']),
            'missing predecessor '+previous)
    available=ALLOWANCE-60.-ledger['used_seconds'];limit=min(PHASES[phase],available)
    require(limit==PHASES[phase],'insufficient authorized phase budget/reserve')
    attempt=OUT/('attempt-'+phase+'-01');attempt.mkdir(exist_ok=False)
    temp=OUT/'package/temp';temp.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ);env.update(ENV);env.update(PYTHONPATH=p['site'],TEMP=str(temp),TMP=str(temp),
        PIP_DISABLE_PIP_VERSION_CHECK='1')
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--child',phase,str(attempt)]
    save(attempt/'binding.json',dict(command=command,cwd=str(attempt),preparation=p,
        authority_sha256=sha(authority),
        environment={k:env[k] for k in (*ENV,'PYTHONPATH','TEMP','TMP')}))
    for name,value in p['source'].items():
        path=attempt/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes());require(sha(path)==value,'snapshot drift')
    sys.path.insert(0,str(ROOT))
    from scripts._b3_e4_f03_local_scientific_supervisor import supervise,SupervisorLimits
    save(ledger_path,ledger)
    with (attempt/'stdout.txt').open('w') as stdout,(attempt/'stderr.txt').open('w') as stderr:
        receipt=supervise(command,cwd=attempt,environment=env,stdout=stdout,stderr=stderr,
            limits=SupervisorLimits(limit,RSS_BYTES,.05))
    save(attempt/'receipt.json',asdict(receipt))
    result=json.loads((attempt/'result.json').read_text()) if (attempt/'result.json').exists() else {}
    passed=receipt.reason=='completed' and receipt.exit_code==0 and result.get('status','').startswith('PASS')
    ledger['used_seconds']+=receipt.elapsed_seconds
    ledger['attempts'].append(dict(phase=phase,passed=passed,process_seconds=receipt.elapsed_seconds,
        reason=receipt.reason,exit_code=receipt.exit_code))
    if not passed or phase=='beam':
        ledger.update(status='CLOSED',discarded_unused_seconds=ALLOWANCE-ledger['used_seconds'])
    save(ledger_path,ledger)
    print(json.dumps(dict(status=ledger['status'],phase=phase,passed=passed,used_seconds=ledger['used_seconds'])))
    return passed


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',choices=tuple(PHASES));parser.add_argument('--authority')
    parser.add_argument('--child',nargs=2)
    args=parser.parse_args()
    if args.prepare:
        prepare();print(json.dumps({'status':'PREPARED_SLOT_HELD','sha256':sha(OUT/'PREPARED.json')}))
    elif args.child:sys.exit(0 if child(args.child[0],Path(args.child[1])) else 1)
    elif args.run and args.authority:sys.exit(0 if run(args.run,args.authority) else 1)
    else:parser.error('explicit prepare or reviewed authority phase required')
