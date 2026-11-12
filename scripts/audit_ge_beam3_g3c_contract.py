"""Inert G3c design/source/fixture auditor. Does not execute mechanics."""
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE='773b61b5370a5f8365a376e958244defe1e9cb79'
TREE='768eb0d9d3b302020ee336eecbc4451244b208f9'
CONTRACT='docs/reference_cases/ge_beam3_g3c_contract_v1.json'
FIXTURES='docs/reference_cases/ge_beam3_g3c_fixtures_v1.json'
EXTENT=('docs/GE_BEAM3_G3_COMPLETION_PLAN.md', CONTRACT, FIXTURES,
        'scripts/audit_ge_beam3_g3c_contract.py','tests/test_ge_beam3_g3c_contract.py')
FAMILIES=('B2','B3','Q4','S3')
SOURCE_PATHS=["docs/GE_BEAM3_GENERAL_STATIC_INTEGRATION_CONTRACT.md","docs/GE_BEAM3_G3_GRAPH_JUNCTION_CONTRACT.md","docs/reference_cases/ge_beam3_g3_graph_fixtures_v1.json","docs/reference_cases/ge_beam3_g3a_confirmation_v1.json","docs/reference_cases/ge_beam3_g3a_confirmation_review_v1.json","docs/reference_cases/ge_beam3_g3b_confirmation_v1.json","docs/reference_cases/ge_beam3_g3b_confirmation_review_v1.json","src/anyfesolver/_ge_beam3_g1_elastic.py","src/anyfesolver/_ge_beam3_g1_element.py","src/anyfesolver/_ge_beam3_g1_operator.py","src/anyfesolver/_ge_beam3_g3_analysis.py","src/anyfesolver/_ge_beam3_g3_constraints.py","src/anyfesolver/_ge_beam3_g3b_owner.py","src/anyfesolver/_ge_beam3_g3b_result_schema.py","src/anyfesolver/_ge_beam3_pose_joint.py","src/anyfesolver/_ge_beam3_shell_joint_trial.py","src/anyfesolver/_ge_beam3_shell_joint_state.py","src/anyfesolver/_ge_beam3_variational_shell.py","src/anyfesolver/_ge_beam3_mixed_ad.py","src/anyfesolver/corotational.py","src/anyfesolver/elements.py","src/anyfesolver/e4_pl_element.py","src/anyfesolver/e4_pl_s3_v2d_element.py","src/anyfesolver/nonlinear_state.py","docs/agent_plans/GE_BEAM3_VARIATIONAL_SHELL_MAP.md","tests/test_ge_beam3_pose_joint.py","tests/test_ge_beam3_shell_joint_trial.py","tests/test_ge_beam3_shell_joint_state.py","docs/reference_cases/e4_pl_s3_v2_bounded_process.py"]
REQUIRED_CHECKS=["INHERITED_ROUTE_ADMISSIBILITY","EXPLICIT_POSE_AND_MATERIAL_FRAMES","ANALYTIC_CHART_FIRST_SECOND_DERIVATIVES","COMMON_RIGID_MOTION_AND_REBASE","JOINT_WORK_ACTION_REACTION","REAL_FINITE_FAMILY_RESPONSE","NATIVE_FULL_STATIONARY_SCHUR","PHYSICAL_RECOVERY","CONSERVATIVE_SYMMETRY_WHERE_APPLICABLE","ALL_DIRECTIONAL_STEPS","ALL_FIVE_GRAPHS_AND_VARIANTS","NONCOMMUTING_ROOTS_AND_LOAD_HISTORY","ATOMIC_PREPARE_ALL_COMMIT","ROLLBACK_CANCELLATION_AND_TOKEN_REJECTION","CACHE_AND_DEFINITION_IDENTITY","AUTHENTICATED_PREFIX_REPLAY","RESOURCE_BOUNDS_AND_FAILURES","TWO_DETERMINISTIC_FORMAL_CYCLES","INDEPENDENT_IMPLEMENTATION_AND_EVIDENCE_REVIEW"]
DEFINITIONS={"legacy":{"E":100,"Iy":0.2,"Iz":0.3,"J":0.1,"area":1,"fiber_plasticity":False,"generalized_section":False,"nu":0.25,"orientation":[0,0,1],"shear_factor_y":0.8,"shear_factor_z":0.7,"geometric_nonlinearity":"von_karman","wrapper":"corotational.corotational_element_response","tangent_mode":"consistent"},"native":{"E":100,"G":40,"Iy":0.2,"Iz":0.3,"J":0.1,"area":1,"shear_y":0.8,"shear_z":0.7},"shell":{"E":100,"generalized_section":False,"layers":3,"nu":0.25,"reference_normal":[0,0,1],"thickness":0.1,"yield_stress":0}}
PROGRAMS={"common_rotation_vectors":[[0,0,0],[0.4,-0.3,0.2],[3.141592653589793,0,0],[0,4.39822971502571,0]],"directional_steps":[0.0001,0.00001,0.000001],"force":[0.1,0.2,-0.3],"force_scales":[0.01,1,10],"joint_multipliers":[0.3,-0.2,0.4,-0.1,0.25,-0.15],"load_factors":[0,0.5,1,0.25,0],"root_increment_vectors":[[0.12,0,0],[0,-0.09,0.04]]}
NEGATIVE_CASES=["unregistered_adapter","wrong_family","stale_authority","wrong_normal","shared_id_without_adapter","rotational_affine_row","legacy_checkpoint","foreign_token","replayed_token","changed_graph","changed_section","changed_frame","changed_chart","changed_constraint","changed_cached_factor","bad_relative_chart","unsupported_component","unsupported_section_history","bounds_overflow"]
GRAPH_IDS=tuple('J_'+f+'_PAIR' for f in FAMILIES)+('J_MULTIFAMILY_LOOP',)
STAGES=('CONTRACT_REVIEW','LOCAL_ADAPTER_SMOKE','REAL_FAMILY_TRIAL_BRIDGE',
        'GLOBAL_GRAPH_OWNER','AUTHENTICATED_REPLAY','REVIEWED_FORMAL_CONFIRMATION')
LIMITS=dict(elements=8,nodes=32,external_dofs=192,external_plus_internal=384,
            constraints=96,restart_bytes=8*1024**2,accepted_history=128,
            child_seconds=600,wave_seconds=1800,inactivity_seconds=120,
            process_tree_memory_bytes=24*1024**3,max_workers=3,numerical_threads=1,
            formal_cycles=2,automatic_retry=False)


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=True,
                       allow_nan=False)+'\n').encode('ascii')


def strict(raw):
    if type(raw) is not bytes or not 0<len(raw)<=4*1024**2: raise ValueError('bounded bytes')
    def pairs(items):
        result={}
        for k,v in items:
            if k in result: raise ValueError('duplicate key')
            result[k]=v
        return result
    def bad(value): raise ValueError('nonfinite number')
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
    if canonical(value)!=raw: raise ValueError('noncanonical JSON')
    return value


def text(path): return Path(path).read_bytes().replace(b'\r\n',b'\n')


def git(*args):
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,GIT_CONFIG_SYSTEM=os.devnull,
               GIT_NO_REPLACE_OBJECTS='1',GIT_ATTR_NOSYSTEM='1')
    return subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),
        '-c','core.attributesFile='+os.devnull,'-c','core.autocrlf=true','-c','core.eol=crlf',
        *args],cwd=ROOT,env=env,timeout=30).decode().strip()


def require(condition, message):
    if not condition: raise ValueError(message)


def number(v):
    require(type(v) in (int,float) and math.isfinite(v),'finite numeric value')
    return Fraction(v)


def frame(rows):
    require(type(rows) is list and len(rows)==3 and all(type(r) is list and len(r)==3 for r in rows),'frame shape')
    m=[[number(v) for v in r] for r in rows]
    require(all(sum(m[k][i]*m[k][j] for k in range(3))==int(i==j)
                for i in range(3) for j in range(3)),'orthonormal explicit frame')
    det=sum(m[0][i]*(m[1][(i+1)%3]*m[2][(i+2)%3]-m[1][(i+2)%3]*m[2][(i+1)%3]) for i in range(3))
    require(det==1,'proper physical frame')


def validate_fixtures(data):
    require(set(data)=={'schema','status','definitions','graphs','programs','variants','negative_cases'},'fixture schema')
    require(canonical(data['definitions'])==canonical(DEFINITIONS),'complete frozen section definitions')
    require(canonical(data['programs'])==canonical(PROGRAMS),'complete frozen load/chart programs')
    require(data['negative_cases']==NEGATIVE_CASES,'complete negative-case registry')
    require(data['schema']=='GE_BEAM3_G3C_FIXTURES_V1' and data['status']=='PLANNED_UNEXECUTED','no fixture acceptance')
    require([g['id'] for g in data['graphs']]==list(GRAPH_IDS),'complete ordered graph inventory')
    require(data['variants']==['BASE','SHUFFLED_INSERTION','RENUMBERED','CONNECTIVITY_REVERSED','PROPER_GLOBAL_TRANSFORM'],'all transport variants')
    require(data['programs']['load_factors']==[0,.5,1,.25,0] and
            data['programs']['directional_steps']==[1e-4,1e-5,1e-6] and
            data['programs']['force_scales']==[.01,1,10],'frozen histories and steps')
    require(data['definitions']['shell']['reference_normal']==[0,0,1] and
            data['definitions']['shell']['yield_stress']==0 and
            data['definitions']['legacy']['generalized_section'] is False and
            data['definitions']['legacy']['fiber_plasticity'] is False,'exact elastic family boundaries')
    for graph in data['graphs']:
        require(set(graph)=={'id','nodes','elements','fixed_nodes','joints','load_node','reference_only','qualification'},'graph schema')
        require(graph['qualification'] is False and graph['reference_only'] is False,'positive finite cases required but unqualified')
        ids=[r[0] for r in graph['nodes']]
        require(all(type(n) is int and n>0 for n in ids) and ids==sorted(set(ids)),'unique sorted node IDs')
        require(len(ids)<=32 and len(graph['elements'])<=8,'graph bounds')
        positions={n:tuple(number(v) for v in xyz) for n,xyz in graph['nodes']}
        require(all(len(p)==3 for p in positions.values()),'position shape')
        elements=graph['elements']; eids=[e['id'] for e in elements]
        require(eids==sorted(set(eids)) and all(type(i) is int and i>0 for i in eids),'unique element IDs')
        incidence={n:set() for n in ids}; adjacency={n:set() for n in ids}
        for e in elements:
            require(set(e)=={'id','family','nodes','orientation','roll_radians'},'element schema')
            f=e['family']; expected={'NATIVE':3,'B2':2,'B3':3,'Q4':4,'S3':3}
            require(f in expected and len(e['nodes'])==expected[f] and len(set(e['nodes']))==expected[f],'exact family topology')
            require(all(type(n) is int and n>0 and n in positions for n in e['nodes']),'known exact-integer connectivity')
            for n in e['nodes']:
                incidence[n].add(f); adjacency[n].update(set(e['nodes'])-{n})
            require(len(e['orientation'])==3 and sum(number(v)**2 for v in e['orientation'])>0,'physical orientation')
            number(e['roll_radians'])
            if f in ('NATIVE','B3'):
                a,m,b=(positions[n] for n in e['nodes'])
                require(all(2*m[i]==a[i]+b[i] for i in range(3)) and a!=b,'exact midpoint')
        require(all(len(f)==1 for f in incidence.values()),'no cross-family ID alias or orphan')
        native_count=sum(e['family']=='NATIVE' for e in elements)
        require(6*len(ids)+24*native_count<=384,'internal-coordinate bound')
        require(6*(len(graph['joints'])+len(graph['fixed_nodes']))<=96,'constraint bound')
        js=[j['id'] for j in graph['joints']]
        require(js==sorted(set(js)) and all(type(i) is int and i>0 for i in js),'joint ordering')
        for j in graph['joints']:
            require(set(j)=={'id','adapter','master','slave','master_frame','slave_frame'},'joint schema')
            m,s=j['master'],j['slave']
            require(type(m) is int and m>0 and type(s) is int and s>0,'exact-integer joint references')
            require(m in incidence and s in incidence and m!=s and incidence[s]=={'NATIVE'},'explicit native slave')
            family=next(iter(incidence[m]))
            require(family in FAMILIES and j['adapter']=='G3C_NATIVE_'+family+'_POSE_V1','exact family adapter')
            frame(j['master_frame']); frame(j['slave_frame'])
            adjacency[m].add(s); adjacency[s].add(m)
        require(all(type(n) is int and n>0 for n in graph['fixed_nodes']) and
                type(graph['load_node']) is int and graph['load_node']>0 and
                graph['fixed_nodes']==sorted(set(graph['fixed_nodes'])) and
                set(graph['fixed_nodes'])<=set(ids) and graph['load_node'] in ids,'explicit supports/load')
        remaining=set(ids)
        while remaining:
            todo=[min(remaining)]; found=set()
            while todo:
                n=todo.pop()
                if n in found: continue
                found.add(n); todo.extend(adjacency[n]-found)
            require(bool(found & set(graph['fixed_nodes'])),'supported connected component')
            remaining-=found
    return data


def audit():
    contract=strict(text(ROOT/CONTRACT)); fixtures=validate_fixtures(strict(text(ROOT/FIXTURES)))
    require(set(contract)=={'schema','status','base','extent','limits','stages','admission',
        'required_checks','source_bindings','payload_bindings'},'contract schema')
    require(contract['schema']=='GE_BEAM3_G3C_CONTRACT_V1' and
            contract['status']=='FROZEN_DESIGN_BLOCKED_INHERITED_ROUTE','blocked design only')
    require(contract['base']==dict(commit=BASE,tree=TREE),'accepted G3b base')
    require(contract['extent']==list(EXTENT) and canonical(contract['limits'])==canonical(LIMITS),'extent/limits')
    require(contract['stages']==list(STAGES),'complete development funnel')
    require(contract['required_checks']==REQUIRED_CHECKS,'complete required-check registry')
    require([r['path'] for r in contract['source_bindings']]==SOURCE_PATHS,'complete ordered source registry')
    require(contract['admission']==dict(accepted_adapter_ids=[],
        planned_adapter_ids=['G3C_NATIVE_'+f+'_POSE_V1' for f in FAMILIES],
        defaults_changed=False,full_G3_complete=False,
        runtime_implementation_authorized=False,inherited_route_blocked=True),'no premature admission')
    require(git('rev-parse',BASE+'^{tree}')==TREE,'base tree')
    git('merge-base','--is-ancestor',BASE,'HEAD')
    for row in contract['source_bindings']:
        require(set(row)=={'path','bytes','sha256','blob'},'source binding schema')
        raw=text(ROOT/row['path'])
        require(len(raw)==row['bytes'] and sha256(raw).hexdigest()==row['sha256'],'unchanged source '+row['path'])
        require(git('rev-parse',BASE+':'+row['path'])==row['blob'],'original blob '+row['path'])
    require(set(contract['payload_bindings'])==set(EXTENT)-{CONTRACT},'complete non-self-referencing payload DAG')
    for path,b in contract['payload_bindings'].items():
        raw=text(ROOT/path)
        require(b==dict(bytes=len(raw),sha256=sha256(raw).hexdigest()),'frozen payload '+path)
    changed=set(git('diff','--name-only',BASE).splitlines())
    untracked=set(git('ls-files','--others','--exclude-standard').splitlines())
    require(changed|untracked==set(EXTENT),'exact five-path design extent')
    git('diff','--check',BASE)
    return dict(status='CONTRACT_STATIC_ONLY_NOT_QUALIFICATION',graphs=len(fixtures['graphs']),
                graph_variants=len(fixtures['graphs'])*len(fixtures['variants']),source_bindings=len(contract['source_bindings']))


if __name__=='__main__': print(canonical(audit()).decode(),end='')
