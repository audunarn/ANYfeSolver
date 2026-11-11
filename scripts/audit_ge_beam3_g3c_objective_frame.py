"""Inert authority audit for the private G3c direct-anchor equation candidate."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE='10623490e83928d561349072566b9dc2ffbcd325'
TREE='d8422a3b3feafc19b8bd71e2ca65ecadd3f594f9'
POLICY='GE_BEAM3_G3C_ANCHOR_DIRECTOR_LOCAL_POTENTIAL_V1'
PLAN='docs/GE_BEAM3_G3C_OBJECTIVE_BEAM_FRAME_CONTRACT.md'
CONTRACT='docs/reference_cases/ge_beam3_g3c_objective_beam_frame_contract_v1.json'
REVIEW='docs/reference_cases/ge_beam3_g3c_blocked_design_review_v1.json'
EXTENT=[PLAN,CONTRACT,REVIEW,'scripts/audit_ge_beam3_g3c_objective_frame.py',
        'tests/test_ge_beam3_g3c_objective_frame_contract.py']
SOURCES=['docs/GE_BEAM3_GENERAL_STATIC_INTEGRATION_CONTRACT.md',
 'docs/GE_BEAM3_G3_COMPLETION_PLAN.md',
 'docs/reference_cases/ge_beam3_g3c_contract_v1.json',
 'docs/reference_cases/ge_beam3_g3c_fixtures_v1.json',
 'src/anyfesolver/elements.py','src/anyfesolver/corotational.py',
 'src/anyfesolver/_ge_beam3_mixed_ad.py','src/anyfesolver/_ge_beam3_pose_joint.py']
ANCHORS=[['J_B2_PAIR',11,101],['J_B3_PAIR',11,102],
         ['J_MULTIFAMILY_LOOP',11,101],['J_MULTIFAMILY_LOOP',12,202]]
LIMITS=dict(child_seconds=600,wave_seconds=1800,inactivity_seconds=120,
 max_workers=3,numerical_threads=1,memory_bytes=25769803776,automatic_retry=False)
TESTS=['REFERENCE_OPERATOR_EQUALITY','DIRECT_ANCHOR_OBJECTIVITY',
 'ALL_FROZEN_COMMON_ROTATIONS','REJECTED_FRAME_COUNTEREXAMPLES',
 'COMPLETE_FIRST_SECOND_PULLBACK','ACTUAL_LOCAL_POTENTIAL_AND_RECOVERY',
 'ANCHOR_RENUMBER_AND_REVERSAL','RELATIVE_DOMAIN_AND_INPUT_REJECTION',
 'ACCEPTED_MATRIX_REBASE_EQUALITY','DETACHED_ORIGIN_AND_NO_COMMIT']
TOP={'schema','status','base','policy','anchors','limits','extent','sources',
     'payloads','next_tests','admission'}

def canonical(v):
    return (json.dumps(v,sort_keys=True,separators=(',', ':'),allow_nan=False,
                       ensure_ascii=True)+'\n').encode('ascii')

def require(ok,message):
    if not ok: raise ValueError(message)

def strict(raw):
    require(type(raw) is bytes and 0<len(raw)<=4*1024**2,'bounded JSON bytes')
    def pairs(items):
        d={}
        for k,v in items:
            require(k not in d,'duplicate key'); d[k]=v
        return d
    def bad(_): raise ValueError('nonfinite JSON')
    v=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
    require(canonical(v)==raw,'canonical JSON')
    return v

def read(path): return (ROOT/path).read_bytes().replace(b'\r\n',b'\n')

def git(*args):
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,
      GIT_CONFIG_SYSTEM=os.devnull,GIT_NO_REPLACE_OBJECTS='1',GIT_ATTR_NOSYSTEM='1')
    return subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),
       '-c','core.autocrlf=true','-c','core.eol=crlf',
       '-c','core.attributesFile='+os.devnull,*args],cwd=ROOT,env=env,timeout=30).decode().strip()

def validate(c):
    require(type(c) is dict and set(c)==TOP,'contract keys')
    require(c['schema']=='GE_BEAM3_G3C_OBJECTIVE_FRAME_CONTRACT_V1' and
      c['status']=='FROZEN_EQUATIONS_REQUIRING_INDEPENDENT_REVIEW','equations only')
    require(c['base']==dict(commit=BASE,tree=TREE) and c['policy']==POLICY,'base/policy')
    require(canonical(c['anchors'])==canonical(ANCHORS),'physical anchor registry')
    require(canonical(c['limits'])==canonical(LIMITS),'resource limits')
    require(c['extent']==EXTENT and c['next_tests']==TESTS,'closed extent/inventory')
    require(canonical(c['admission'])==canonical(dict(accepted_adapter_ids=[],G3_complete=False,
       public_legacy_changed=False,Q4_S3_changed=False,defaults_changed=False,
       full_domain_parity=False,runtime_implementation_authorized=False)),'no acceptance')
    require([s['path'] for s in c['sources']]==SOURCES,'source registry')
    require(set(c['payloads'])==set(EXTENT)-{CONTRACT},'payload registry')
    for s in c['sources']:
        require(set(s)=={'path','blob','sha256','bytes'} and
                type(s['bytes']) is int and s['bytes']>0,'source row')
    return c

def audit():
    c=validate(strict(read(CONTRACT)))
    require(git('rev-parse',BASE+'^{tree}')==TREE,'base tree')
    git('merge-base','--is-ancestor',BASE,'HEAD')
    for s in c['sources']:
        raw=read(s['path'])
        require(len(raw)==s['bytes'] and sha256(raw).hexdigest()==s['sha256'],'source hash')
        require(git('rev-parse',BASE+':'+s['path'])==s['blob'],'source blob')
    for p,row in c['payloads'].items():
        raw=read(p)
        require(row==dict(bytes=len(raw),sha256=sha256(raw).hexdigest()),'payload hash')
    review=strict(read(REVIEW))
    require(set(review)=={'decision','findings','reviewer','scope','subject_commit'} and
      review['subject_commit']==BASE and
      review['decision']=='BLOCKED_G3C_POSITIVE_READINESS_INHERITED_ROUTE' and
      len(review['findings'])==1 and review['findings'][0]['id']=='G3C-DR-01','preserved incident')
    require(sha256(read(REVIEW)).hexdigest()=='13fa0505d461389371b4849fc7c4e2a250117036690502316e1b285f2d87c3a7','unaltered independent review')
    changed=set(git('diff','--name-only',BASE).splitlines())
    extra=set(git('ls-files','--others','--exclude-standard').splitlines())
    require(changed|extra==set(EXTENT),'exact design-only delta')
    git('diff','--check',BASE)
    return dict(status='EQUATION_DESIGN_ONLY',mechanics_executed=False,source_count=len(SOURCES))

if __name__=='__main__': print(canonical(audit()).decode(),end='')
