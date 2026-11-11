"""Inert mixed-owner contract audit. No mechanics or authority generation."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '64dc8c42d91126dfef4322c15c0bd314975ffe4d'
TREE = 'ad073a00324e5d0adc8dd16425a966cbb092090e'
PLAN = 'docs/GE_BEAM3_G3C_MIXED_OWNER_CONTRACT.md'
CONTRACT = 'docs/reference_cases/ge_beam3_g3c_mixed_owner_contract_v1.json'
EXTENT = [PLAN, CONTRACT, 'scripts/audit_ge_beam3_g3c_mixed_owner.py',
          'tests/test_ge_beam3_g3c_mixed_owner_contract.py']
SOURCES = [
 'docs/GE_BEAM3_GENERAL_STATIC_INTEGRATION_CONTRACT.md',
 'docs/GE_BEAM3_G3_COMPLETION_PLAN.md',
 'docs/reference_cases/ge_beam3_g3c_contract_v1.json',
 'docs/reference_cases/ge_beam3_g3c_fixtures_v1.json',
 'docs/reference_cases/ge_beam3_g3c_finite_local_review_v1.json',
 'docs/reference_cases/ge_beam3_g3c_matrix_shell_contract_v1.json',
 'docs/reference_cases/ge_beam3_g3c_matrix_shell_equation_review_v1.json',
 'docs/reference_cases/ge_beam3_g3c_local_shell_implementation_review_v1.json',
 'docs/GE_BEAM3_G3C_LOCAL_SHELL_ACCEPTANCE.md',
 'src/anyfesolver/_ge_beam3_g3c_local_beam.py',
 'src/anyfesolver/_ge_beam3_g3c_recovery.py',
 'src/anyfesolver/_ge_beam3_g3c_local_shell.py',
 'src/anyfesolver/_ge_beam3_g1_element.py',
 'src/anyfesolver/_ge_beam3_g1_operator.py',
 'src/anyfesolver/_ge_beam3_g1_elastic.py',
 'src/anyfesolver/_ge_beam3_g1_analysis.py',
 'src/anyfesolver/_ge_beam3_g3_analysis.py',
 'src/anyfesolver/_ge_beam3_g3_constraints.py',
 'src/anyfesolver/_ge_beam3_generalized_static_boundary.py',
 'src/anyfesolver/_ge_beam3_pose_joint.py',
 'src/anyfesolver/_native_material_protocol.py',
 'src/anyfesolver/_native_rotation_state.py',
 'src/anyfesolver/nonlinear_state.py',
 'src/anyfesolver/elements.py', 'src/anyfesolver/e4_pl_element.py',
 'src/anyfesolver/e4_pl_s3_v2d_element.py',
 'docs/reference_cases/e4_pl_s3_v2_bounded_process.py',
 'scripts/ge_beam3_g3b_environment.py']
ENVIRONMENT = '2ce154b866565e1d03019651b1ae7fdd070e5554f38d72adf21d8080558b6756'
POLICIES = {'NATIVE': 'CANDIDATE_GE_BEAM3_G1_ELASTIC_STATIC_V1',
 'B2': 'GE_BEAM3_G3C_ANCHOR_DIRECTOR_LOCAL_POTENTIAL_V1',
 'B3': 'GE_BEAM3_G3C_ANCHOR_DIRECTOR_LOCAL_POTENTIAL_V1',
 'Q4': 'GE_BEAM3_G3C_MATRIX_POSE_SHELL_PULLBACK_V1',
 'S3': 'GE_BEAM3_G3C_MATRIX_POSE_SHELL_PULLBACK_V1'}
LIMITS = dict(child_seconds=600, wave_seconds=1800, inactivity_seconds=120,
 max_workers=3, numerical_threads=1, memory_bytes=25769803776,
 automatic_retry=False, restart_bytes=8388608, accepted_history=128,
 elements=8, nodes=32, external_dofs=192, external_plus_internal=384, constraints=96)
TESTS = [
 'MO01_REGISTERED_25_EXPANSIONS_AND_BOUNDARY_REJECTION',
 'MO02_ISSUED_NATIVE_CONTEXT_AND_ACCEPTED_ORIGIN',
 'MO03_NATIVE_FULL_SCHUR_NONZERO_INTERNAL_AND_LOAD',
 'MO04_FOUR_ACTUAL_ADAPTERS_SINGLE_CHART_PULLBACK',
 'MO05_NONIDENTITY_PORT_C_WORK_AND_WRENCH_BALANCE',
 'MO06_SUPPORT_KKT_MULTIPLIER_HESSIANS_ALL_STEPS',
 'MO07_ALL_GRAPH_LOAD_SCALES_AND_NONCOMMUTING_HISTORY',
 'MO08_ALL_VARIANTS_COMMON_MOTIONS_AND_REBASE',
 'MO09_LAST_FAMILY_AND_LATE_SANDBOX_FAILURE_ATOMICITY',
 'MO10_CALLBACK_INPUT_MUTATION_CANCELLATION_AND_REENTRY',
 'MO11_FOREIGN_STALE_COPIED_REPLAYED_TOKEN',
 'MO12_CHANGED_DEFINITION_RUNTIME_GRAPH_AND_CACHE',
 'MO13_COMPLETE_NESTED_PREFLIGHT_BEFORE_CONSTRUCTION',
 'MO14_EVERY_PREFIX_REPLAY_AND_CONTINUATION_EQUALITY',
 'MO15_REHASHED_STATE_HISTORY_POSE_ORIGIN_MUTATIONS',
 'MO16_REAL_RECOVERY_AND_OPEN_OBLIGATION_ENFORCEMENT',
 'MO17_PROCESS_BOUNDS_NO_PARTIAL_EVIDENCE',
 'MO18_REHEARSAL_REVIEW_AND_TWO_FORMAL_CYCLES']
# Closed structural layout. Semantic constraints and shape variables are below.
LAYOUTS = {
 'checkpoint': dict(schema='literal:GE_BEAM3_G3C_MIXED_ELASTIC_RESTART_V1',
  policy='literal:GE_BEAM3_G3C_MIXED_ELASTIC_OWNER_V1', runtime_sha256='sha256',
  definition='definition', definition_sha256='sha256', initial_sha256='sha256',
  history='entry[0..128]', final_state='state', final_sha256='sha256'),
 'definition': dict(fixture_id='registered_graph', variant='registered_variant',
  common_motion='enum:NONE,CM0,CM1,CM2,CM3', fixture_sha256='sha256',
  policy_sha256='sha256', expanded_sha256='sha256'),
 'entry': dict(epoch='int[1..128]', previous_entry_sha256='sha256|null',
  origin_sha256='sha256', command='load_command|prepare_command', accepted_state_sha256='sha256'),
 'load_command': dict(kind='literal:LOAD_STAGE', load_factor='f64',
  force_scale='f64', root_stage='int[0..4]'),
 'prepare_command': dict(kind='literal:PREPARE_COMMON_MOTION', step='int[1..4]'),
 'state': dict(schema='literal:GE_BEAM3_G3C_MIXED_ELASTIC_STATE_V1',
  epoch='int[0..128]', definition_sha256='sha256', previous_sha256='sha256|null',
  total_u='f64[6*n]', rotations='proper_rotation[n]',
  native_rows='native_row[native_count]', adapter_rows='adapter_row[adapter_count]',
  multipliers='f64[constraint_count]'),
 'native_row': dict(element_id='positive_int', payload='native_payload'),
 'native_payload': dict(schema='literal:GE_BEAM3_G1_ELASTIC_STATE_V1',
  element_identity='sha256', epoch='int[0..128]', committed_total_u='f64[18]',
  committed_nodal_rotation_matrices='proper_rotation[3]', response='native_response',
  history='empty_list', previous_state_sha256='sha256|null', line='zero_f64[3]',
  couple='zero_f64[3]', state_sha256='sha256'),
 'native_response': dict(rotations='proper_rotation[2]', resultants='f64[18]',
  residual='f64[18]', tangent='f64[18,18]', lift='f64[24,18]', correction='f64[24]',
  full='native_full', internal_error='f64[0..1e-11]', history='empty_list',
  iterations='int[0..24]'),
 'native_full': dict(residual='f64[42]', hessian='f64[42,42]',
  jacobian='f64[42,42]', potential='f64', conservative='literal:true'),
 'adapter_row': dict(element_id='positive_int', policy='registered_local_policy',
  definition_sha256='sha256', epoch='int[0..128]', previous_sha256='sha256|null',
  origin_sha256='sha256|null', pose_sha256='sha256', deformation='f64[6*family_nodes]',
  diagnostic_sha256='sha256', source_material_committed='literal:false',
  physical_recovery_complete='literal:false')}
SEMANTICS = dict(
 shape_variables='recomputed from exact registered expanded fixture before arrays',
 command='NONE:load programs only; CM0..CM3:four genuine preparation commits then unchanged load programs',
 hash_dag='state->previous_state; entry->origin+accepted_state+previous_entry; no cycle',
 publication='one immutable generation pointer; sandbox commits cannot change published state',
 replay='external digest plus full preflight then actual every-prefix replay; exact final bytes',
 native_origin='owner-captured payload validated by fresh actual issued material context',
 adapter_origin='graph-owned envelope; source virgin diagnostic never relabeled committed',
 token='nonserializable identity capability; owner+generation+definition+runtime+serial',
 cache='no persistent numerical factor cache',
 recovery='Q4 conventional finite recovery and B2 clamp-domain recovery remain open')
ADMISSION = dict(mechanics_executed=False, runtime_implementation_authorized=False,
 full_G3_complete=False, physical_recovery_complete=False, full_domain_parity=False,
 defaults_changed=False, production_changed=False)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False)+'\n').encode('ascii')

def strict(raw):
    require(type(raw) is bytes and 0 < len(raw) <= 4*1024**2, 'JSON bound')
    def pairs(rows):
        result = {}
        for k, v in rows:
            require(k not in result, 'duplicate key')
            result[k] = v
        return result
    def bad(_):
        raise ValueError('nonfinite JSON')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    require(canonical(value) == raw, 'canonical JSON')
    return value

def read(path):
    return (ROOT/path).read_bytes().replace(b'\r\n', b'\n')

def fingerprint(raw):
    return dict(bytes=len(raw), sha256=sha256(raw).hexdigest())

def git(*args):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_CONFIG_SYSTEM=os.devnull, GIT_NO_REPLACE_OBJECTS='1', GIT_ATTR_NOSYSTEM='1')
    return subprocess.check_output(['git', '-c', 'safe.directory='+ROOT.as_posix(),
        '-c', 'core.autocrlf=true', '-c', 'core.eol=crlf',
        '-c', 'core.attributesFile='+os.devnull, *args], cwd=ROOT, env=env, timeout=30).decode().strip()

def fixed():
    return dict(schema='GE_BEAM3_G3C_MIXED_OWNER_CONTRACT_V1',
     status='FROZEN_DESIGN_REQUIRING_INDEPENDENT_REVIEW', base=dict(commit=BASE, tree=TREE),
     extent=EXTENT, policies=POLICIES, environment_sha256=ENVIRONMENT,
     limits=LIMITS, layouts=LAYOUTS, semantics=SEMANTICS, next_tests=TESTS, admission=ADMISSION)

def validate(value):
    expected = fixed()
    require(type(value) is dict and set(value) == set(expected)|{'sources', 'payloads'}, 'contract keys')
    for key, wanted in expected.items():
        require(canonical(value[key]) == canonical(wanted), 'frozen '+key)
    sources = value['sources']
    require(type(sources) is list and all(type(r) is dict for r in sources)
            and [r.get('path') for r in sources] == SOURCES, 'source inventory')
    for row in sources:
        require(type(row) is dict and set(row) == {'path', 'blob', 'bytes', 'sha256'}, 'source row')
        validate_fingerprint({k: row[k] for k in ('bytes', 'sha256')})
        require(hexstring(row['blob'], 40), 'source blob')
    require(type(value['payloads']) is dict and set(value['payloads']) == set(EXTENT)-{CONTRACT}, 'payload inventory')
    for row in value['payloads'].values():
        validate_fingerprint(row)

def hexstring(value, n):
    return type(value) is str and len(value) == n and all(v in '0123456789abcdef' for v in value)

def validate_fingerprint(row):
    require(type(row) is dict and set(row) == {'bytes', 'sha256'}, 'fingerprint keys')
    require(type(row['bytes']) is int and row['bytes'] > 0 and hexstring(row['sha256'], 64), 'fingerprint values')

def audit():
    c = strict(read(CONTRACT)); validate(c)
    require(git('rev-parse', BASE+'^{tree}') == TREE, 'baseline tree')
    git('merge-base', '--is-ancestor', BASE, 'HEAD')  # Missing local base is fatal.
    blobs = {}
    for line in git('ls-tree', '-r', BASE).splitlines():
        meta, path = line.split('\t', 1)
        blobs[path] = meta.split()[2]
    for row in c['sources']:
        require(blobs.get(row['path']) == row['blob'], 'base blob')
        require(fingerprint(read(row['path'])) == {k: row[k] for k in ('bytes', 'sha256')}, 'source hash')
    for path, row in c['payloads'].items():
        require(fingerprint(read(path)) == row, 'payload hash')
    changed = set(git('diff', '--name-only', BASE).splitlines())
    extra = set(git('ls-files', '--others', '--exclude-standard').splitlines())
    require(changed | extra == set(EXTENT), 'exact contract extent')
    git('diff', '--check', BASE)
    f = strict(read('docs/reference_cases/ge_beam3_g3c_fixtures_v1.json'))
    require(len(f['graphs']) == 5 and len(f['variants']) == 5, '25 fixture coverage')
    return dict(status='DESIGN_ONLY', mechanics_executed=False, sources=len(SOURCES),
                graph_variants=25, physical_recovery_complete=False)

if __name__ == '__main__':
    print(canonical(audit()).decode(), end='')
