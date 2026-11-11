"""Strict development checkpoint for one committed native midpoint state.

Integrity requires the caller's trusted external digest. This codec is not a
signature system, trajectory replay, candidate/session credential or static codec.
"""
from hashlib import sha256
import json
import os
import stat
from pathlib import Path
import numpy as np

from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5_seeded.codec import _load, _keys, MAX_BYTES
from ._ge_beam3_native_generalized_restart import _history
from ._ge_beam3_retained_generalized_state import Mechanical
from ._ge_beam3_centered_finite_inertia import PHYSICAL_DOFS, ALGEBRAIC_DOFS
from ._native_reference_modal import _owned

SCHEMA = 'GE_BEAM3_NATIVE_CONTACT_FREE_MIDPOINT_CHECKPOINT_DEV_V1'
IMPLEMENTATION_ID = 'DEVELOPMENT_NATIVE_MIDPOINT_RESTART_V1'
MAX_PACKAGE_FILES = 1024
MAX_PACKAGE_FILE_BYTES = 2*1024*1024
MAX_PACKAGE_TOTAL_BYTES = 64*1024*1024
MAX_PACKAGE_DIRECTORIES = 512
MAX_PACKAGE_ENTRIES = 8192


def _source_is_link(path):
    """Python3.11 lstat: reject symlinks and every Windows reparse entry.

    FILE_ATTRIBUTE_REPARSE_POINT includes junctions. Windows missing attribute
    metadata fails closed; it is not a reason to ignore the junction check.
    POSIX uses the original lstat symbolic-link mode and follows no link.
    """
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        return True
    if os.name == 'nt':
        attributes = getattr(metadata, 'st_file_attributes', None)
        if type(attributes) is not int:
            raise ValueError('native dynamic Windows reparse metadata required')
        return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return False


def _package_manifest():
    """Exact anyfesolver .py bytes; stable relative paths, no installation path.

    SHA256 covers canonical sorted [relative-path, byte-length, file-SHA256]
    records. Only Python source artifacts are claimed, not external dependencies
    or compiled caches. Traversal, file count and bytes have admission bounds;
    symlink entries and case-ambiguous source paths fail closed. This runs at
    owner operation boundaries, never in Newton/stage evaluations.
    """
    root = Path(__file__).absolute().parent
    if not root.is_dir() or _source_is_link(root):
        raise ValueError('native dynamic package source directory required')
    rows = []
    total = directories = entries = 0
    seen = set()
    def walk_error(error):
        raise ValueError('native dynamic package source traversal failed') from error
    for directory, names, files in os.walk(root, followlinks=False, onerror=walk_error):
        directories += 1
        entries += len(names)+len(files)
        if directories > MAX_PACKAGE_DIRECTORIES or entries > MAX_PACKAGE_ENTRIES:
            raise ValueError('native dynamic package traversal bound')
        names.sort()
        for name in names:
            if _source_is_link(Path(directory, name)):
                raise ValueError('native dynamic package symlink directory unsupported')
        for name in sorted(files):
            if not name.lower().endswith('.py'):
                continue
            path = Path(directory, name)
            if _source_is_link(path):
                raise ValueError('native dynamic package symlink source unsupported')
            relative = path.relative_to(root).as_posix()
            folded = relative.casefold()
            if folded in seen:
                raise ValueError('native dynamic package source path ambiguity')
            seen.add(folded)
            if len(rows) >= MAX_PACKAGE_FILES:
                raise ValueError('native dynamic package source file count bound')
            with path.open('rb') as stream:
                raw = stream.read(MAX_PACKAGE_FILE_BYTES+1)
            if len(raw) > MAX_PACKAGE_FILE_BYTES:
                raise ValueError('native dynamic package source file byte bound')
            total += len(raw)
            if total > MAX_PACKAGE_TOTAL_BYTES:
                raise ValueError('native dynamic package source total byte bound')
            rows.append((relative, len(raw), sha256(raw).hexdigest()))
    if not rows:
        raise ValueError('native dynamic package source manifest empty')
    rows.sort(key=lambda row: row[0])
    return dict(format='CANONICAL_SORTED_RELATIVE_PATH_LENGTH_SHA256_V1',
        source_glob='anyfesolver/**/*.py', files=len(rows), total_bytes=total,
        manifest_sha256=sha(rows), limits=dict(files=MAX_PACKAGE_FILES,
            file_bytes=MAX_PACKAGE_FILE_BYTES, total_bytes=MAX_PACKAGE_TOTAL_BYTES,
            directories=MAX_PACKAGE_DIRECTORIES, entries=MAX_PACKAGE_ENTRIES))


_CHECK_KEYS = ('algebraic_rank', 'algebraic_sigma_min', 'algebraic_sigma_max',
               'algebraic_condition', 'physical_mass_min_eigenvalue',
               'physical_mass_max_eigenvalue', 'physical_mass_condition')


def contract(context):
    """Existing profile/graph contracts plus all bounded anyfesolver .py sources.

    No new generic package importer, signing framework or version migration.
    Source files are required for this explicitly development-only codec.
    """
    from . import ge_beam3_native_dynamics as profile
    from .ge_beam3_native import PROFILE_SHA256 as native_profile
    from ._ge_beam3_native_midpoint import IMPLEMENTATION_ID as midpoint_id
    from ._ge_beam3_dynamic_elastic_state import IMPLEMENTATION_ID as dynamic_id
    from ._ge_beam3_centered_finite_inertia import IMPLEMENTATION_ID as inertia_id
    if context.structural_owner != 'dynamic_elastic':
        raise ValueError('dynamic elastic midpoint restart only')
    return dict(schema=SCHEMA, codec=IMPLEMENTATION_ID,
        development_profile_id=profile.PROFILE_ID,
        development_profile_sha256=profile.PROFILE_SHA256,
        native_profile_sha256=native_profile, selector=profile.SELECTOR,
        package_source_manifest=_package_manifest(),
        backend=dict(midpoint=midpoint_id, dynamic_elastic=dynamic_id, finite_inertia=inertia_id),
        analysis_sha256=context.analysis.identity, midpoint_sha256=context._identity,
        dynamic_sha256=context._dynamic._identity,
        facade_sha256=sha(dict(profile_sha256=profile.PROFILE_SHA256,
            native=context.analysis.identity, midpoint=context._identity)),
        settings=dict(inertia_order=context.inertia_order, max_iterations=context.max_iterations,
            max_backtracks=context.max_backtracks, stage_residual_tolerance=context.stage_residual_tolerance,
            **({} if context.stage_coordinates == 'increment' else
               dict(stage_coordinates=context.stage_coordinates)),
            correction_tolerance=1e-11, endpoint_algebraic_residual_tolerance=1e-11,
            endpoint_algebraic_correction_tolerance=1e-11, endpoint_physical_tolerance=1e-11,
            lifetime_seconds=context.runtime_limit_seconds),
        node_ids=tuple(sorted(context.analysis.model.mesh.nodes)),
        element_ids=tuple(e.element_id for e in context.analysis._elements),
        physical_order=PHYSICAL_DOFS, algebraic_order=ALGEBRAIC_DOFS,
        free=context.free, fixed=context.fixed,
        momentum_origin_high=context.momentum_origin_high,
        momentum_origin_low=context.momentum_origin_low,
        constant_load_pattern=context.load_pattern)


def require_contract(context):
    if (type(context._restart_contract) is not bytes
            or context._restart_contract != context._captured_restart_contract
            or context._restart_contract != canonical(contract(context))):
        raise ValueError('native dynamic restart artifact/model/settings changed')
    return json.loads(context._restart_contract)


def _digest(value, label):
    if (type(value) is not str or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('canonical '+label+' digest required')
    return value


def _array(value, shape, label):
    if type(value) is not list:
        raise ValueError('typed checkpoint '+label+' array required')
    try:
        made = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('finite checkpoint '+label+' array required') from error
    if made.shape != shape or not np.isfinite(made).all():
        raise ValueError('finite checkpoint '+label+' extent required')
    return _owned(made)


def _initial_checks(value, context):
    _keys(value, _CHECK_KEYS)
    if type(value['algebraic_rank']) is not int or value['algebraic_rank'] != len(context.algebraic_free):
        raise ValueError('checkpoint historical initial rank extent')
    for name in _CHECK_KEYS[1:]:
        if type(value[name]) is not float or not np.isfinite(value[name]) or value[name] <= 0.:
            raise ValueError('finite positive checkpoint historical initial checks')
    return canonical(value)


def encode(context, state):
    header = require_contract(context)
    context._require_current(state)
    if context._candidates:
        raise ValueError('commit/discard pending candidates before dynamic checkpoint')
    return _encode_body(context, state, header=header)


def _encode_body(context, state, *, header):
    """Existing serialization formula; caller must authenticate its state."""
    if type(context._initial_checks) is not bytes:
        raise ValueError('complete original dynamic initial checks required')
    initial = json.loads(context._initial_checks)
    _initial_checks(initial, context)
    body = dict(header=header, state=state, state_sha256=sha(state),
                original_initial_checks=initial, production_qualified=False)
    raw = canonical(dict(body, checkpoint_sha256=sha(body)))
    if len(raw) > MAX_BYTES:
        raise ValueError('native dynamic checkpoint byte bound')
    return raw


def decode(context, raw, expected_sha256):
    """Typed/integrity validation only; issuer owns semantic checks/publication."""
    from ._ge_beam3_native_midpoint import MidpointState
    if (type(raw) is not bytes or not 0 < len(raw) <= MAX_BYTES
            or sha256(raw).hexdigest() != _digest(expected_sha256, 'external checkpoint')):
        raise ValueError('native dynamic external checkpoint authority mismatch')
    try:
        value = _load(raw.decode('ascii'))
    except UnicodeError as error:
        raise ValueError('canonical ASCII dynamic checkpoint required') from error
    _keys(value, ('header', 'state', 'state_sha256', 'original_initial_checks',
                  'production_qualified', 'checkpoint_sha256'))
    body = {key:item for key,item in value.items() if key != 'checkpoint_sha256'}
    if (value['production_qualified'] is not False
            or value['checkpoint_sha256'] != sha(body)
            or canonical(value['header']) != canonical(require_contract(context))):
        raise ValueError('native dynamic checkpoint header/body authority mismatch')
    row = value['state']
    _keys(row, ('mechanical','origins','histories','physical_velocity24','physical_acceleration24',
                'time','epoch','predecessor_sha256','context_sha256'))
    m = row['mechanical']
    _keys(m, tuple(Mechanical.__dataclass_fields__))
    made = Mechanical(*(_array(m[name], shape, name) for name,shape in (
        ('positions',(3,3)),('position_low',(3,3)),('nodal_frames',(3,3,3)),
        ('cell_rotations',(1,2,3,3)),('resultants',(1,18)))))
    # Existing owner validates split pairs, SO(3), full mixed order and supports.
    mechanical = context._make(made.descriptor())
    histories = []
    for name in ('origins','histories'):
        if type(row[name]) is not list or len(row[name]) != 1:
            raise ValueError('complete one-element dynamic '+name+' required')
        typed = (_history(row[name][0], context.element),)
        if canonical(typed) != canonical(context._reference_origins):
            raise ValueError('unchanged virgin elastic '+name+' required')
        histories.append(typed)
    if type(row['time']) is not float or not np.isfinite(row['time']):
        raise ValueError('finite typed dynamic time required')
    if type(row['epoch']) is not int or row['epoch'] < 0:
        raise ValueError('nonnegative exact dynamic epoch required')
    predecessor = _digest(row['predecessor_sha256'], 'predecessor')
    if row['context_sha256'] != context._identity:
        raise ValueError('native dynamic state context mismatch')
    if row['epoch'] == 0 and predecessor != context._identity:
        raise ValueError('dynamic genesis predecessor mismatch')
    velocity = _array(row['physical_velocity24'], (24,), 'velocity24')
    acceleration = _array(row['physical_acceleration24'], (24,), 'acceleration24')
    inactive = [i for i in range(24) if i not in context.physical_free]
    if np.any(velocity[inactive]) or np.any(acceleration[inactive]):
        raise ValueError('exact zero fixed/algebraic inertial v/a placeholders required')
    state = MidpointState(mechanical, histories[0], histories[1], velocity, acceleration,
        row['time'], row['epoch'], predecessor, context._identity)
    if canonical(state) != canonical(row) or sha(state) != _digest(value['state_sha256'], 'state'):
        raise ValueError('complete dynamic state canonical round trip mismatch')
    initial = _initial_checks(value['original_initial_checks'], context)
    return state, initial
