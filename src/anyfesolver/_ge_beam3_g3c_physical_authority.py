"""Fixed private authority wrapper; no mechanics execution or dynamic cloning."""
from hashlib import sha256
from math import factorial
from pathlib import Path
import os
import stat
import subprocess
import sys

# The inert preflight executes this exact hash-bound file with its separately
# hash-bound standard-library definition module injected as _original. Normal
# package use imports the same fixed module; no checkpoint selects a module.
if '_original' not in globals():
    from anyfesolver import _ge_beam3_g3c_definition as _original
original = _original

ROOT = Path(__file__).resolve().parents[2]
MAP = 'docs/GE_BEAM3_G3C_PHYSICAL_MIXED_OWNER_CONTRACT.md'
CONTRACT_SHA = 'f6638863e6e222c795e980194c9a71cf42eb83bdcba6af2a8c409164ad4d957b'
REVIEW = 'docs/reference_cases/ge_beam3_g3c_physical_owner_contract_review_v1.json'
REVIEW_SHA = '47829c5c1c896ed2eb28131ffedadd25cc83bc5bed79bc2517cfe308f7dc8988'
U = original.U
SHIFT = original.SHIFT
canonical = original.canonical
command = original.command
_LEASE = None


def regular(path):
    for item in (path, *path.parents):
        if item.exists() and item.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError('reparse authority input')
    if not path.is_file():
        raise ValueError('regular authority input required')
    return path


def normalized(path):
    return regular(path).read_bytes().replace(b'\r\n', b'\n')


def bound(path, expected):
    raw = normalized(ROOT/path)
    if sha256(raw).hexdigest() != expected:
        raise ValueError('frozen authority changed: '+path)
    return raw if path == MAP else original.strict(raw)


def capture(lease_bytes):
    """One-shot installation by the reviewed runner after full input validation.

    This is private process plumbing, never a public authorization API. The
    runner independently verifies review, Git blobs and environment BEFORE
    importing anyfesolver; no mechanics are admitted merely by importing us.
    """
    global _LEASE
    if _LEASE is not None:
        raise ValueError('private authority already captured')
    lease = original.strict(lease_bytes)
    if lease.get('kind') != 'G3C_PHYSICAL_PRIVATE_DEVELOPMENT' or lease.get('contract_sha256') != CONTRACT_SHA:
        raise ValueError('private implementation lease required')
    review = lease.get('implementation_review')
    if (type(review) is not dict or
            set(review) != {'decision', 'findings', 'reviewer', 'scope', 'subject_commit'} or
            sha256(canonical(review)).hexdigest() != lease.get('review_sha256') or
            review['decision'] != 'ACCEPTED_G3C_PHYSICAL_IMPLEMENTATION_FOR_BOUNDED_DEVELOPMENT' or
            review['findings'] or review['reviewer'].get('independent') is not True or
            review['subject_commit'] != lease['candidate']['commit'] or
            review['scope'].get('subject_tree') != lease['candidate']['tree'] or
            review['scope'].get('contract_sha256') != CONTRACT_SHA or
            review['scope'].get('inputs_sha256') != sha256(canonical(lease['inputs'])).hexdigest()):
        raise ValueError('review-bound private implementation lease required')
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_CONFIG_SYSTEM=os.devnull, GIT_NO_REPLACE_OBJECTS='1', GIT_ATTR_NOSYSTEM='1')
    args = ['git', '-c', 'safe.directory='+ROOT.as_posix(), '-c', 'core.autocrlf=true',
            '-c', 'core.eol=crlf', '-c', 'core.attributesFile='+os.devnull]
    identity = subprocess.check_output(args+['rev-parse', 'HEAD', 'HEAD^{tree}'],
                                       cwd=ROOT, env=env, timeout=10).decode().splitlines()
    if identity != [lease['candidate']['commit'], lease['candidate']['tree']]:
        raise ValueError('private lease candidate identity changed')
    if subprocess.check_output(args+['status', '--porcelain', '--untracked-files=all'],
                               cwd=ROOT, env=env, timeout=10).strip():
        raise ValueError('dirty private lease candidate')
    rows = []
    for path, row in sorted(lease['inputs'].items()):
        if Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError('lease path escapes root')
        if path.startswith('src/') or path in (MAP, REVIEW):
            if set(row) != {'bytes', 'sha256'}:
                raise ValueError('lease source row schema')
            rows.append((path, row['bytes'], row['sha256']))
    required={'src/anyfesolver/_ge_beam3_g3c_physical_authority.py',
        'src/anyfesolver/_ge_beam3_g3c_physical_owner.py',
        'src/anyfesolver/_ge_beam3_g3c_b2_physical.py',
        'src/anyfesolver/_ge_beam3_g3c_b2_physical_adapter.py',
        'src/anyfesolver/_ge_beam3_g3c_affine_q4_recovery.py',
        'src/anyfesolver/_ge_beam3_g3c_affine_q4_registry.py',
        'src/anyfesolver/_ge_beam3_g3c_affine_q4_increment_chart.py',MAP,REVIEW}
    if not required.issubset({r[0] for r in rows}):
        raise ValueError('missing private source lease')
    made = (sha256(lease_bytes).hexdigest(), tuple(rows))
    _verify_rows(made[1])
    _LEASE = made


def _verify_rows(rows):
    for path, size, expected in rows:
        raw = normalized(ROOT/path)
        if len(raw) != size or sha256(raw).hexdigest() != expected:
            raise ValueError('leased runtime source changed: '+path)


DEFINITION_SCHEMA='GE_BEAM3_G3C_PHYSICAL_GRAPH_DEFINITION_V1'
OPERATOR_GRAPH='GE_BEAM3_G3C_PHYSICAL_OPERATOR_GRAPH_V1'
ENVELOPE_SCHEMA='GE_BEAM3_G3C_PHYSICAL_ADAPTER_ENVELOPE_V1'
STATE_SCHEMA='GE_BEAM3_G3C_PHYSICAL_MIXED_ELASTIC_STATE_V1'
RESTART_SCHEMA='GE_BEAM3_G3C_PHYSICAL_GRAPH_RESTART_V1'
OWNER_POLICY='GE_BEAM3_G3C_PHYSICAL_MIXED_ELASTIC_OWNER_V1'
B2_POLICY='GE_BEAM3_G3C_B2_PHYSICAL_MATRIX_ADAPTER_V1'
Q4_POLICY='GE_BEAM3_G3C_AFFINE_Q4_PHYSICAL_FACADE_V1'

def envelope_identities(family):
    operators={'B2':'GE_BEAM3_G3C_B2_PHYSICAL_FLEXIBILITY_V1',
        'B3':'GE_BEAM3_G3C_ANCHOR_DIRECTOR_LOCAL_POTENTIAL_V1',
        'Q4':'E4_PL_QUALIFIED_Q4_HYBRID_V2',
        'S3':'CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1'}
    if type(family) is not str or family not in operators: raise ValueError('registered envelope family')
    made=dict(schema=ENVELOPE_SCHEMA,family=family,operator_id=operators[family])
    if family=='Q4':
        made.update(recovery_id='GE_BEAM3_Q4_AFFINE_CHART_PHYSICAL_RECOVERY_V1',
            representation_id='GE_BEAM3_Q4_AFFINE_STATION_RECOVERY_64_V1',
            chart_numerics_id='GE_BEAM3_Q4_AFFINE_INCREMENT_RESOLVED_CHART_NUMERICS_V1',
            station_association_id='GE_BEAM3_Q4_NATURAL_COORDINATE_BIJECTION_V1')
    return made

def envelope_keys(family):
    return frozenset(envelope_identities(family)) | frozenset((
        'element_id','policy','definition_sha256','epoch','previous_sha256','origin_sha256',
        'pose_sha256','deformation','diagnostic_sha256','source_material_committed',
        'physical_recovery_complete','recovery_witness_sha256'))

def expand_inert(fixture_id,variant='BASE',common_motion='NONE'):
    """Exact selector/identity expansion only; no numerical imports or constructors."""
    bound(MAP,CONTRACT_SHA)
    review=bound(REVIEW,REVIEW_SHA)
    if (review['findings'] or review['decision']!='ACCEPTED_GE_BEAM3_G3C_PHYSICAL_OWNER_CONTRACT_DESIGN_ONLY'
            or review['scope']['contract_sha256']!=CONTRACT_SHA):
        raise ValueError('physical design not accepted')
    definition,expanded,programs=original.expand(fixture_id,variant,common_motion)
    expanded['local_policies']['B2']=B2_POLICY
    expanded['local_policies']['Q4']=Q4_POLICY
    definition.update(schema=DEFINITION_SCHEMA,operator_graph_id=OPERATOR_GRAPH,
        policy_sha256=CONTRACT_SHA,expanded_sha256=original.digest(expanded))
    return definition,expanded,programs

def expand(fixture_id,variant='BASE',common_motion='NONE'):
    if _LEASE is None: raise ValueError('physical runner lease required before construction')
    _verify_rows(_LEASE[1])
    return expand_inert(fixture_id,variant,common_motion)

def runtime_identity():
    if _LEASE is None: raise ValueError('missing physical runtime lease')
    # Attempt/lease UUID is external administration, not scientific state.
    return sha256(canonical(dict(contract=CONTRACT_SHA,inputs=_LEASE[1]))).hexdigest()


def runtime_modules():
    # Fixed imports only, never modules selected by a lease or fixture.
    from ._ge_beam3_g3c_stable import boundary, chart, compensated, line_work, fit
    from anyfesolver import _ge_beam3_g3c_so3_numerics as kernel
    from anyfesolver import _ge_beam3_mixed_ad as ad
    from anyfesolver import _ge_beam3_g3c_recovery as recovery
    from anyfesolver import _ge_beam3_g3c_b2_physical as physical_core
    from anyfesolver import _ge_beam3_g3c_affine_q4_increment_chart as increment_chart
    from anyfesolver import _ge_beam3_g3c_affine_q4_chart as connection_chart
    from anyfesolver import _ge_beam3_g3c_local_shell as local_shell
    from anyfesolver import e4_pl_element as q4_source
    return (sys.modules[__name__], original, boundary, chart, compensated,
            line_work, fit, kernel, ad, recovery,physical_core,increment_chart,
            connection_chart,local_shell,q4_source)


def runtime_constants():
    # Exercise the real authority-read hook before dispatch comparison. Read
    # failures poison the owner even when a callback has replaced this hook.
    bound(REVIEW, REVIEW_SHA)
    if _LEASE is None:
        raise ValueError('missing private lease')
    from anyfesolver import _ge_beam3_g3c_so3_numerics as kernel
    sinc = tuple((-1.0)**n/factorial(2*n+1) for n in range(17))
    cosc = tuple((-1.0)**n/factorial(2*n+2) for n in range(17))
    log = [1.0]
    for n in range(1, 41):
        log.append(log[-1]*n/(2*n+1))
    found = []
    for name, expected in (('SINC', sinc), ('COSC', cosc), ('LOG', tuple(log))):
        value = getattr(kernel, name)
        if type(value) is not tuple or any(type(v) is not float for v in value) or value != expected:
            raise ValueError('frozen kernel coefficient authority changed: '+name)
        found.append((kernel.__name__, name, value, id(value)))
    found.append((__name__, '_LEASE', _LEASE, id(_LEASE)))
    return tuple(found)
