"""Fixed private authority wrapper; no mechanics execution or dynamic cloning."""
from hashlib import sha256
from math import factorial
from pathlib import Path
import os
import stat
import subprocess
import sys

from anyfesolver import _ge_beam3_g3c_definition as original

ROOT = Path(__file__).resolve().parents[3]
MAP = 'docs/reference_cases/ge_beam3_g3c_stable_source_map_v1.json'
CONTRACT_SHA = '30848dea0530f5a49b44cea184f4ff38d73f7e3449131086d9f2c9f312a51ca8'
REVIEW = 'docs/reference_cases/ge_beam3_g3c_stable_source_map_review_v1.json'
REVIEW_SHA = 'd700a1284eb8324442e523d04b21da1c8c24f1c92d19da97d9e37922528a787f'
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
    return original.strict(raw)


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
    if lease.get('kind') != 'G3C_STABLE_PRIVATE_DEVELOPMENT' or lease.get('source_map_sha256') != CONTRACT_SHA:
        raise ValueError('private implementation lease required')
    review = lease.get('implementation_review')
    if (type(review) is not dict or
            set(review) != {'decision', 'findings', 'reviewer', 'scope', 'subject_commit'} or
            sha256(canonical(review)).hexdigest() != lease.get('review_sha256') or
            review['decision'] != 'ACCEPTED_G3C_STABLE_IMPLEMENTATION_FOR_BOUNDED_DEVELOPMENT' or
            review['findings'] or review['reviewer'].get('independent') is not True or
            review['subject_commit'] != lease['candidate']['commit'] or
            review['scope'].get('subject_tree') != lease['candidate']['tree'] or
            review['scope'].get('source_map_sha256') != CONTRACT_SHA or
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
    if not rows or 'src/anyfesolver/_ge_beam3_g3c_stable/authority.py' not in {r[0] for r in rows}:
        raise ValueError('missing private source lease')
    made = (sha256(lease_bytes).hexdigest(), tuple(rows))
    _verify_rows(made[1])
    _LEASE = made


def _verify_rows(rows):
    for path, size, expected in rows:
        raw = normalized(ROOT/path)
        if len(raw) != size or sha256(raw).hexdigest() != expected:
            raise ValueError('leased runtime source changed: '+path)


def expand(fixture_id, variant='BASE', common_motion='NONE'):
    if _LEASE is None:
        raise ValueError('reviewed runner lease required before graph construction')
    mapping = bound(MAP, CONTRACT_SHA)
    review = bound(REVIEW, REVIEW_SHA)
    if review['findings'] or review['decision'] != 'ACCEPTED_G3C_PRIVATE_SOURCE_MAP_DESIGN_ONLY':
        raise ValueError('private source-map design not accepted')
    _verify_rows(_LEASE[1])
    # Revalidate the mapped dependencies, including exact copied destinations.
    rows = [(r['source'], r['source_fingerprint']) for r in mapping['copies']]
    rows += [(r['destination'], r['generated_fingerprint']) for r in mapping['copies']]
    rows += [(r['path'], r['fingerprint']) for r in mapping['bindings']]
    for path, expected in rows:
        raw = normalized(ROOT/path)
        if dict(bytes=len(raw), sha256=sha256(raw).hexdigest()) != expected:
            raise ValueError('private map source mismatch: '+path)
    definition, expanded, programs = original.expand(fixture_id, variant, common_motion)
    definition['policy_sha256'] = CONTRACT_SHA
    return definition, expanded, programs


def runtime_modules():
    # Fixed imports only, never modules selected by a lease or fixture.
    from . import boundary, chart, compensated, line_work, fit
    from anyfesolver import _ge_beam3_g3c_so3_numerics as kernel
    from anyfesolver import _ge_beam3_mixed_ad as ad
    from anyfesolver import _ge_beam3_g3c_recovery as recovery
    return (sys.modules[__name__], original, boundary, chart, compensated,
            line_work, fit, kernel, ad, recovery)


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
