"""Standard-library-only equation authority audit; never imports mechanics."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '44dc811ccf620536216c8bc97c2930f2502454b4'
TREE = '624ecc2bc931c5467f59a1cb74c5ff71f1b5efce'
POLICY = 'GE_BEAM3_G3C_MATRIX_POSE_SHELL_PULLBACK_V1'
CONTRACT = 'docs/reference_cases/ge_beam3_g3c_matrix_shell_contract_v1.json'
PLAN = 'docs/GE_BEAM3_G3C_MATRIX_SHELL_CONTRACT.md'
ASSESSMENT = 'docs/GE_BEAM3_G3C_Q4_WORK_CHANNEL_ASSESSMENT.md'
EXTENT = [PLAN, ASSESSMENT, CONTRACT,
          'scripts/audit_ge_beam3_g3c_matrix_shell.py',
          'tests/test_ge_beam3_g3c_matrix_shell_contract.py']
SOURCES = [
 'docs/GE_BEAM3_GENERAL_STATIC_INTEGRATION_CONTRACT.md',
 'docs/GE_BEAM3_G3_COMPLETION_PLAN.md',
 'docs/reference_cases/ge_beam3_g3c_contract_v1.json',
 'docs/reference_cases/ge_beam3_g3c_fixtures_v1.json',
 'docs/GE_BEAM3_G3C_MATRIX_POSE_SHELL_ASSESSMENT.md',
 'docs/reference_cases/ge_beam3_g3c_finite_local_review_v1.json',
 'src/anyfesolver/_ge_beam3_variational_shell.py',
 'src/anyfesolver/_ge_beam3_mixed_ad.py',
 'src/anyfesolver/_ge_beam3_pose_joint.py',
 'src/anyfesolver/elements.py', 'src/anyfesolver/e4_pl_element.py',
 'src/anyfesolver/e4_pl_s3_v2c_element.py',
 'src/anyfesolver/e4_pl_s3_v2d_element.py',
 'src/anyfesolver/e4_pl_s3_state.py', 'src/anyfesolver/e4_pl_s3_v2d_state.py',
 'src/anyfesolver/nonlinear_state.py', 'src/anyfesolver/fe_core.py',
 'docs/reference_cases/e4_pl_s3_v2_bounded_process.py',
 'scripts/ge_beam3_g3b_environment.py']
LIMITS = dict(child_seconds=600, wave_seconds=1800, inactivity_seconds=120,
              max_workers=3, numerical_threads=1, memory_bytes=25769803776,
              automatic_retry=False)
CHANNELS = [['QUALIFIED_MIXED_PHYSICAL', 1], ['NUMERICAL_PL', 1],
            ['NUMERICAL_HOURGLASS', 1], ['COMPATIBLE_VK_MEMBRANE', 1],
            ['REMOVED_COMPATIBLE_LINEAR_MEMBRANE', -1]]
TESTS = ['INDEPENDENT_MATRIX_POSE_VALUES', 'COMPLETE_FIRST_SECOND_MAP',
 'ACTUAL_REFERENCE_OPERATOR_AND_RIGID_NULLS', 'ACTUAL_FINITE_FORCE_TANGENT_WORK',
 'Q4_SIGNED_CHANNEL_SUMS', 'Q4_CHECKERBOARD_AND_SINGLE_FIELD_COUNTERSENTINEL',
 'S3_SOURCE_NATIVE_STATION_WORK', 'D4_D3_AND_PHYSICAL_DIRECTOR_TRANSPORT',
 'COMMON_PI_AND_1_4PI_AND_PASSIVE_TRANSFORM', 'REPEATED_LOWER_AND_TOP_GAP',
 'SAME_POSE_REBASE_AND_LOCAL_CANDIDATE', 'OLD_MAP_COMMON_CHART_COMPARISON',
 'OWNERSHIP_NONFINITE_AND_GUARD_BEFORE_EVALUATION']
ADMISSION = dict(runtime_implementation_authorized=False, mechanics_executed=False,
 G3_complete=False, full_domain_parity=False, physical_recovery_complete=False,
 public_Q4_S3_changed=False, defaults_changed=False, state_committed=False)
TOP = {'schema', 'status', 'base', 'policy', 'limits', 'extent', 'sources',
       'payloads', 'channels', 'next_tests', 'admission', 'environment_sha256'}
ENVIRONMENT = '2ce154b866565e1d03019651b1ae7fdd070e5554f38d72adf21d8080558b6756'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False, ensure_ascii=True)+'\n').encode('ascii')

def strict(raw):
    require(type(raw) is bytes and 0 < len(raw) <= 4*1024**2, 'bounded JSON')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate key')
            result[key] = value
        return result
    def bad(_):
        raise ValueError('nonfinite JSON')
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    require(canonical(result) == raw, 'canonical JSON')
    return result

def read(path):
    return (ROOT/path).read_bytes().replace(b'\r\n', b'\n')

def git(*args):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_CONFIG_SYSTEM=os.devnull, GIT_NO_REPLACE_OBJECTS='1',
               GIT_ATTR_NOSYSTEM='1')
    return subprocess.check_output(['git', '-c', 'safe.directory='+ROOT.as_posix(),
        '-c', 'core.autocrlf=true', '-c', 'core.eol=crlf',
        '-c', 'core.attributesFile='+os.devnull, *args],
        cwd=ROOT, env=env, timeout=30).decode().strip()

def fingerprint(raw):
    return dict(bytes=len(raw), sha256=sha256(raw).hexdigest())

def valid_hash(value, length):
    return (type(value) is str and len(value) == length
            and all(c in '0123456789abcdef' for c in value))

def validate(c):
    require(type(c) is dict and set(c) == TOP, 'contract keys')
    require(c['schema'] == 'GE_BEAM3_G3C_MATRIX_SHELL_CONTRACT_V1'
            and c['status'] == 'FROZEN_EQUATIONS_REQUIRING_INDEPENDENT_REVIEW',
            'equation status')
    require(c['base'] == dict(commit=BASE, tree=TREE) and c['policy'] == POLICY,
            'base/policy')
    for field, expected in [('limits', LIMITS), ('extent', EXTENT),
                            ('channels', CHANNELS), ('next_tests', TESTS),
                            ('admission', ADMISSION)]:
        require(canonical(c[field]) == canonical(expected), field)
    require(c['environment_sha256'] == ENVIRONMENT, 'environment')
    require(type(c['sources']) is list and all(type(s) is dict for s in c['sources'])
            and [s.get('path') for s in c['sources']] == SOURCES, 'sources')
    require(type(c['payloads']) is dict and set(c['payloads']) == set(EXTENT)-{CONTRACT},
            'payload registry')
    for s in c['sources']:
        require(set(s) == {'path', 'blob', 'bytes', 'sha256'} and valid_hash(s['blob'], 40),
                'source row')
    for row in [*c['sources'], *c['payloads'].values()]:
        require(type(row) is dict and type(row.get('bytes')) is int and row['bytes'] > 0
                and valid_hash(row.get('sha256'), 64), 'hash row')
    for row in c['payloads'].values():
        require(set(row) == {'bytes', 'sha256'}, 'payload row')
    return c

def audit():
    c = validate(strict(read(CONTRACT)))
    require(git('rev-parse', BASE+'^{tree}') == TREE, 'base tree')
    git('merge-base', '--is-ancestor', BASE, 'HEAD')
    # One immutable tree query, not one process for each source row.
    tree = {}
    for line in git('ls-tree', '-r', BASE, '--', *SOURCES).splitlines():
        metadata, path = line.split('\t', 1)
        mode, kind, blob = metadata.split()
        require(mode == '100644' and kind == 'blob', 'regular source')
        tree[path] = blob
    for s in c['sources']:
        require(tree.get(s['path']) == s['blob'], 'source blob')
        require(fingerprint(read(s['path'])) == {k: s[k] for k in ('bytes', 'sha256')},
                'source hash')
    for path, row in c['payloads'].items():
        require(fingerprint(read(path)) == row, 'payload hash')
    changed = set(git('diff', '--name-only', BASE).splitlines())
    extra = set(git('ls-files', '--others', '--exclude-standard').splitlines())
    require(changed | extra == set(EXTENT), 'exact design-only extent')
    git('diff', '--check', BASE)
    return dict(status='EQUATION_DESIGN_ONLY', mechanics_executed=False,
                source_count=len(SOURCES), physical_recovery_complete=False)

if __name__ == '__main__':
    print(canonical(audit()).decode(), end='')
